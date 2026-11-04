"""USP's Ulysses attention with the all-to-alls on SDMA copy engines (ROCm).

Selected with XDIT_ULYSSES_A2A=sdma (default rccl). XDIT_ULYSSES_HEAD_GROUPS (default
2) splits each rank's heads into groups whose exchanges overlap the other groups'
attention, and XDIT_ULYSSES_GROUP_STREAMS=1 runs each group's attention on a stream of
its own; see xfuser/core/distributed/sdma/ulysses.py.

The call is one custom op, opaque to torch.compile, that runs the selected
attention backend per head group. It covers the plain Ulysses case: no ring, no
joint tensors, no FP8 communication, no extra per-head inputs, no grouped-query
repetition, no PipeFusion KV cache, no head balancing and a dense backend.
Anything else, and a call made while a CUDA graph is being captured (the SDMA
copies are issued from the host), runs on RCCL as before.
"""

from typing import Optional

import torch
import torch.distributed as dist

import xfuser.envs as envs
from xfuser.core.attention import registry as attention_registry
from xfuser.core.attention.spec import AttentionBackendType
from xfuser.logger import init_logger

logger = init_logger(__name__)

_MODES = ("rccl", "sdma")
# Keys of attention_kwargs that dense backends read and the op does not carry to them
# (it carries valid_kv_len and the varlen key packing): a call that sets one stays on
# RCCL. Every other key configures sparse backends, which never take this path.
_UNFORWARDED_KWARGS = ("attn_mask", "pre_quantized", "q_descale", "k_descale", "v_descale", "seqlens_k", "return_lse")


def _from_env():
    mode = envs.environment_variables["XDIT_ULYSSES_A2A"]().strip().lower()
    if mode not in _MODES:
        raise ValueError(f"XDIT_ULYSSES_A2A must be one of {_MODES}; got {mode!r}")
    groups = int(envs.environment_variables["XDIT_ULYSSES_HEAD_GROUPS"]())
    return mode, groups, envs.environment_variables["XDIT_ULYSSES_GROUP_STREAMS"]().strip() == "1"


_mode, _head_groups, _concurrent = _from_env()
_warned = set()


def configure(mode: str, head_groups: int, concurrent: bool = False):
    """Select the Ulysses all-to-all transport for later calls; returns the previous
    (mode, head_groups, concurrent)."""
    global _mode, _head_groups, _concurrent
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}; got {mode!r}")
    previous = (_mode, _head_groups, _concurrent)
    _mode, _head_groups, _concurrent = mode, head_groups, concurrent
    return previous


def selected() -> bool:
    return _mode == "sdma"


def unsupported_reason(
    query,
    key,
    value,
    *,
    backend,
    ring_world_size: int,
    joint_strategy,
    fp8_comms,
    extra_inputs,
    kv_head_repeat: int,
    has_kv_cache: bool,
    head_balanced: bool,
    attention_kwargs,
) -> Optional[str]:
    """Why this USP call cannot take the SDMA path, or None if it can."""
    if torch.version.hip is None:
        return "the SDMA all-to-all needs ROCm"
    if ring_world_size > 1:
        return "ring attention"
    if joint_strategy:
        return "joint tensors"
    if fp8_comms is not None:
        return "FP8 communication"
    if extra_inputs:
        return "extra per-head inputs"
    if kv_head_repeat > 1:
        return "grouped-query KV repetition"
    if has_kv_cache:
        return "PipeFusion KV cache"
    if head_balanced:
        return "head balancing"
    if not (query.shape == key.shape == value.shape and query.dtype == key.dtype == value.dtype):
        return "Q, K and V of different shapes or dtypes"
    spec = attention_registry.find(backend)
    if spec is None or spec.sparsity is not None:
        return f"backend {getattr(backend, 'name', backend)} is not a dense backend"
    unforwarded = [k for k in _UNFORWARDED_KWARGS if (attention_kwargs or {}).get(k) is not None]
    if unforwarded:
        return f"attention_kwargs {unforwarded}"
    return None


def warn_fallback(reason: str) -> None:
    if reason not in _warned:
        _warned.add(reason)
        logger.warning(f"XDIT_ULYSSES_A2A=sdma: this attention runs on RCCL ({reason})")


# Dynamo drops a logging call traced inside a compiled region; registered here, the call
# runs after the graph instead, without a graph break.
_reorderable = getattr(torch._dynamo.config, "reorderable_logging_functions", None)
if _reorderable is not None:
    _reorderable.add(warn_fallback)


def attention(query, key, value, *, backend, dropout_p, is_causal, attention_kwargs, group: dist.ProcessGroup):
    """Ulysses attention on SDMA engines. Inputs are USP's [b, H, s, d]; returns [b, H, s, d]
    (a transposed view of a contiguous [b, s, H, d])."""
    kwargs = attention_kwargs or {}
    out = torch.ops.xfuser.sdma_ulysses_attention(
        query,
        key,
        value,
        kwargs.get("indices_k"),
        kwargs.get("cu_seqlens_k"),
        backend.name if isinstance(backend, AttentionBackendType) else str(backend),
        float(dropout_p),
        bool(is_causal),
        kwargs.get("valid_kv_len"),
        kwargs.get("max_seqlen_k"),
        _head_groups,
        _concurrent,
        group.group_name,
    )
    return out.transpose(1, 2)


def _attend_fn(backend, dropout_p, is_causal, attention_kwargs):
    """Run the backend on one head group as USP would after its input all-to-all."""
    from xfuser.model_executor.layers import usp

    attention_function = usp._get_attention_function(backend=backend)

    def attend(q, k, v):
        kwargs = attention_kwargs
        if kwargs["valid_kv_len"] is not None and not is_causal and not usp._backend_strips_sp_pad(backend, kwargs):
            k, v, kwargs = usp._trim_trailing_kv_padding(k, v, kwargs, backend)
        out, _ = attention_function(
            q, k, v, dropout_p=dropout_p, is_causal=is_causal, joint_attn_kwargs=None, attention_kwargs=kwargs
        )
        return out

    return attend


@torch.library.custom_op("xfuser::sdma_ulysses_attention", mutates_args=())
def _sdma_ulysses_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices_k: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    backend: str,
    dropout_p: float,
    is_causal: bool,
    valid_kv_len: Optional[int],
    max_seqlen_k: Optional[int],
    head_groups: int,
    concurrent: bool,
    group_name: str,
) -> torch.Tensor:
    backend_type = AttentionBackendType[backend]
    attention_kwargs = {
        "valid_kv_len": valid_kv_len,
        "indices_k": indices_k,
        "cu_seqlens_k": cu_seqlens_k,
        "max_seqlen_k": max_seqlen_k,
    }
    attend = _attend_fn(backend_type, dropout_p, is_causal, attention_kwargs)
    if torch.cuda.is_current_stream_capturing():
        # Host-issued copies cannot be captured: this call takes USP's RCCL exchange.
        from xfuser.model_executor.layers import usp

        q, k, v = usp._combined_qkv_all_to_all(query, key, value)
        out = usp._ft_c_output_all_to_all(attend(q, k, v))
        return out.transpose(1, 2).contiguous()
    from xfuser.core.distributed.sdma import ulysses

    group = dist.distributed_c10d._resolve_process_group(group_name)
    return ulysses.attention(query, key, value, attend, group, head_groups, concurrent)


@_sdma_ulysses_attention.register_fake
def _(
    query,
    key,
    value,
    indices_k,
    cu_seqlens_k,
    backend,
    dropout_p,
    is_causal,
    valid_kv_len,
    max_seqlen_k,
    head_groups,
    concurrent,
    group_name,
):
    b, heads, s, d = query.shape
    return query.new_empty(b, s, heads, d)
