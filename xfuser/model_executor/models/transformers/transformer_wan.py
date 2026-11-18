import math

import torch
from typing import Optional, Union, Dict, Any, Tuple

from diffusers.models.transformers.transformer_wan import WanAttnProcessor
from diffusers.models.transformers.transformer_wan import WanAttention
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.models.transformers.transformer_wan import WanTransformerBlock
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from xfuser.model_executor.layers.usp import (
    USP,
    attention,
    usp_fused_a2a_input_k,
    usp_fused_a2a_input_q,
    usp_fused_a2a_input_v,
)
from xfuser.core.distributed.attention_a2a import (
    fused_a2a_pad_multiple,
    get_fused_a2a_hadamard_placement,
    get_fused_a2a_mode,
    use_fused_a2a_interleave,
)
from xfuser.model_executor.layers.tensor_parallel import (
    quantize_linear_fp8_,
    shard_linear_,
    shard_rms_norm_,
    tp_column_parallel,
    tp_fp8_column_parallel_split,
    tp_gather_rows,
    tp_rms_norm,
    tp_row_parallel,
    tp_split_rows,
)
from xfuser.core.distributed import ipc_workspace
from xfuser.core.distributed.fp8_comms import register_fp8_comms_eligible_modules
from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_ulysses_parallel_world_size,
    get_ring_parallel_world_size,
    get_sp_group,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_runtime_state,
)
from xfuser.model_executor.layers.attention_processor import xFuserAttentionProcessorRegister
from xfuser.model_executor.models.transformers.transformers_utils import local_sp_pad_rows, zero_sp_pad_rows_hook
import xfuser.envs as envs
from xfuser.envs import PACKAGES_CHECKER
from xfuser.core.vsa_attention import jenga_scheduled_drop_rate
from xfuser.model_executor.layers.fused_qk_norm_rope_wan_flydsl import (
    fused_qk_norm_rope,
    fused_qk_norm_rope_packed,
    wan_flydsl_hadamard,
    _HAS_FLYDSL,
)
from xfuser.model_executor.layers import usp_sdma

env_info = PACKAGES_CHECKER.get_packages_info()
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]


@xFuserAttentionProcessorRegister.register(WanAttnProcessor)
class xFuserWanAttnProcessor(WanAttnProcessor):
    def __init__(
        self,
        use_ulysses_parallel_attention: bool = True,
        is_cross_attention: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if use_ulysses_parallel_attention:
            self.attention_function = USP
        else:
            self.attention_function = attention
        self.is_cross_attention = is_cross_attention
        # attention_kwargs is the shared, mutable dict used by sparse backends
        # (SSTA / sparge) to receive layout info like `thw`. Cross-attention and
        # the I2V image-context sub-call below are dense, so they don't read it.
        self.attention_kwargs = attention_kwargs if attention_kwargs is not None else {}
        # Set by xFuserWanTransformer3DWrapper.shard_for_tensor_parallel: the layer
        # holds 1/tp of the heads, and its Q/K norms span all of them. It is then
        # given this rank's rows and gathers them into its Q/K/V projections in
        # `tp_chunks` pipelined chunks, or through the fused path `tp_fused`
        # names, "symm_mem" or "sdma" (see layers/tensor_parallel.py).
        self.tensor_parallel = False
        self.tp_chunks = 1
        self.tp_fused = None

    def _get_qkv_projections(
        self, attn: "WanAttention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor
    ):
        # encoder_hidden_states is only passed for cross-attention
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if attn.fused_projections:
            if attn.cross_attention_dim_head is None:
                # In self-attention layers, we can fuse the entire QKV projection into a single linear
                query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
            else:
                # In cross-attention layers, we can only fuse the KV projections into a single linear
                query = attn.to_q(hidden_states)
                key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        return query, key, value

    def _get_tp_qkv_projections(
        self, attn: "WanAttention", hidden_states: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor]
    ):
        """Q/K/V of this rank's heads over all rows: the rows are gathered from the TP group into Q (and K/V
        for self-attention). Cross-attention's text K/V are whole on every rank. Projections are fused."""
        if self.is_cross_attention:
            query = tp_column_parallel(hidden_states, attn.to_q, "attn2_q", self.tp_chunks, self.tp_fused)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
            return query, key, value
        qkv = tp_column_parallel(hidden_states, attn.to_qkv, "attn1_qkv", self.tp_chunks, self.tp_fused)
        return qkv.chunk(3, dim=-1)

    def _get_added_kv_projections(self, attn: "WanAttention", encoder_hidden_states_img: torch.Tensor):
        if attn.fused_projections:
            key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
        else:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)
        return key_img, value_img

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # The fused attention all-to-all projects full Q/K/V itself; TP's sharded projections take the standard path.
        if get_fused_a2a_mode() > 0 and not self.is_cross_attention and not self.tensor_parallel:
            return self._call_attention_a2a(
                attn,
                hidden_states,
                encoder_hidden_states,
                rotary_emb,
            )
        return self._call_standard(
            attn,
            hidden_states,
            encoder_hidden_states,
            rotary_emb,
        )

    def _call_standard(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        backend = None
        if self.is_cross_attention:
            # We allow specifying a different backend for cross-attention than the main attention backend
            # as some backends may have too much overhead for cross-attention.
            backend = get_runtime_state().get_cross_attention_backend()

        activation_dtype = hidden_states.dtype
        encoder_hidden_states, encoder_hidden_states_img = self._split_i2v_context(
            attn,
            encoder_hidden_states,
        )
        if self.tensor_parallel:
            query, key, value = self._get_tp_qkv_projections(attn, hidden_states, encoder_hidden_states)
        else:
            query, key, value = self._get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        # Collapse norm_q -> norm_k -> apply_rotary_emb(q) -> apply_rotary_emb(k)
        # into a single FlyDSL kernel: inductor cannot fuse RoPE into the norm
        # (the RMS reduction sits between them) and the reference RoPE writes
        # through two stride-2 scatters, so the reference is four uncoalesced
        # bandwidth-bound passes over a [1, S, H*D] bf16 tensor.  FlyDSL is used
        # automatically when importable (no env flag, no Triton path); the entry
        # self-falls-back to the diffusers reference for out-of-envelope shapes.
        # value carries no norm/rope -- it just needs the head split.
        if self.tensor_parallel:
            # norm_q/norm_k normalize across all heads, which TP splits over ranks,
            # so the RMS needs the TP group's sum; the FlyDSL kernel sees one rank's.
            site = "attn2_qk_norm" if self.is_cross_attention else "attn1_qk_norm"
            query, key = tp_rms_norm((query, key), (attn.norm_q, attn.norm_k), site)
            query, key, value = self._rope_reference(attn, query, key, value, rotary_emb)
        elif _HAS_FLYDSL and rotary_emb is not None:
            query, key = fused_qk_norm_rope(
                query,
                key,
                attn.norm_q,
                attn.norm_k,
                rotary_emb[0],
                rotary_emb[1],
                attn.heads,
            )
            value = value.unflatten(2, (attn.heads, -1))
        else:
            query, key, value = self._qk_norm_rope_reference(attn, query, key, value, rotary_emb)

        hidden_states_img = self._call_i2v_attention(
            attn,
            query,
            encoder_hidden_states_img,
            backend,
            activation_dtype,
        )
        hidden_states = self.attention_function(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            backend=backend,
            attention_kwargs=self.attention_kwargs,
            head_balance_layer=attn,
            attn_layer=None if self.is_cross_attention else attn,
        ).transpose(1, 2)

        return self._finish_attention(
            attn,
            hidden_states,
            activation_dtype,
            hidden_states_img,
        )

    def _call_attention_a2a(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if self.attention_function is not USP:
            raise RuntimeError("Attention A2A requires USP")
        if encoder_hidden_states is not None or attn.add_k_proj is not None:
            raise RuntimeError("Attention A2A supports Wan self-attention only")
        if rotary_emb is None or not _HAS_FLYDSL:
            raise RuntimeError("Attention A2A requires the Wan FlyDSL preprocessing path")
        if get_ulysses_parallel_world_size() <= 1 or get_ring_parallel_world_size() != 1:
            raise RuntimeError("Attention A2A requires Ulysses-only sequence parallelism")
        if get_runtime_state().runtime_config.use_sparseattn_head_balance:
            raise RuntimeError("Attention A2A does not support sparse-attention head balancing")

        activation_dtype = hidden_states.dtype
        hadamard_placement = get_fused_a2a_hadamard_placement()
        valid_kv_len = self.attention_kwargs.get("valid_kv_len")
        pending = None
        interleave = use_fused_a2a_interleave() and not attn.fused_projections

        if interleave:
            # Submit each role on the transport stream while the next
            # projection is still running on the compute stream.
            query = attn.to_q(hidden_states)
            query, _ = fused_qk_norm_rope(
                query,
                query,
                attn.norm_q,
                attn.norm_q,
                rotary_emb[0],
                rotary_emb[1],
                attn.heads,
                single_role=True,
                apply_hadamard=hadamard_placement == "preprocess",
            )
            if hadamard_placement == "epilogue":
                query = wan_flydsl_hadamard(query)
            pending = usp_fused_a2a_input_q(query, valid_kv_len)

            key = attn.to_k(hidden_states)
            key, _ = fused_qk_norm_rope(
                key,
                key,
                attn.norm_k,
                attn.norm_k,
                rotary_emb[0],
                rotary_emb[1],
                attn.heads,
                single_role=True,
                apply_hadamard=hadamard_placement == "preprocess",
            )
            if hadamard_placement == "epilogue":
                key = wan_flydsl_hadamard(key)
            pending = usp_fused_a2a_input_k(key, pending, valid_kv_len)

            value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
            pending = usp_fused_a2a_input_v(value, pending, valid_kv_len)
        else:
            query, key, value = self._get_qkv_projections(attn, hidden_states, None)
            query, key = fused_qk_norm_rope(
                query,
                key,
                attn.norm_q,
                attn.norm_k,
                rotary_emb[0],
                rotary_emb[1],
                attn.heads,
                apply_hadamard=hadamard_placement == "preprocess",
            )
            if hadamard_placement == "epilogue":
                query = wan_flydsl_hadamard(query)
                key = wan_flydsl_hadamard(key)
            value = value.unflatten(2, (attn.heads, -1))

        output = self.attention_function(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            backend=None,
            attention_kwargs=self.attention_kwargs,
            head_balance_layer=attn,
            attn_layer=attn,
            attention_a2a_enabled=True,
            attention_a2a_pending=pending,
        ).transpose(1, 2)
        return self._finish_attention(attn, output, activation_dtype)

    @staticmethod
    def _split_i2v_context(attn, encoder_hidden_states):
        if attn.add_k_proj is None:
            return encoder_hidden_states, None
        # 512 is the context length of the text encoder, hardcoded for now.
        image_context_length = encoder_hidden_states.shape[1] - 512
        return (
            encoder_hidden_states[:, image_context_length:],
            encoder_hidden_states[:, :image_context_length],
        )

    def _call_i2v_attention(
        self,
        attn,
        query,
        encoder_hidden_states_img,
        backend,
        activation_dtype,
    ):
        if encoder_hidden_states_img is None:
            return None
        key_img, value_img = self._get_added_kv_projections(attn, encoder_hidden_states_img)
        key_img = attn.norm_added_k(key_img)
        key_img = key_img.unflatten(2, (attn.heads, -1))
        value_img = value_img.unflatten(2, (attn.heads, -1))
        output = self.attention_function(
            query.transpose(1, 2),
            key_img.transpose(1, 2),
            value_img.transpose(1, 2),
            backend=backend,
            attention_kwargs=self.attention_kwargs,
        ).transpose(1, 2)
        return output.flatten(2, 3).to(activation_dtype)

    def _finish_attention(self, attn, hidden_states, activation_dtype, hidden_states_img=None):
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(activation_dtype)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        if self.tensor_parallel:
            # Row-parallel: partial sums over this rank's heads, reduced back to its rows.
            site = "attn2_out" if self.is_cross_attention else "attn1_out"
            hidden_states = tp_row_parallel(
                hidden_states, attn.to_out[0], attn.tp_out_bias, site, self.tp_chunks, self.tp_fused
            )
        else:
            hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    def _qk_norm_rope_reference(
        self,
        attn: "WanAttention",
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]],
    ):
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        return self._rope_reference(attn, query, key, value, rotary_emb)

    def _rope_reference(
        self,
        attn: "WanAttention",
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]],
    ):
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        return query, key, value


@usp_sdma.sdma_processor(xFuserWanAttnProcessor)
class xFuserWanSdmaAttnProcessor(xFuserWanAttnProcessor):
    """xFuserWanAttnProcessor under --ulysses_a2a sdma --ulysses_sdma_overlap_qkv.

    The QK norm spans every head, so Q/K/V cannot be projected one head group at a time.
    Self-attention instead sends Q and K, normed, rotated and packed peer by peer in one
    kernel, before V is projected, in one head group (usp_sdma.qk_first_attention). Calls
    that path does not cover take the base path.
    """

    def sdma_qk_first(self, attn) -> bool:
        """Whether `attn`'s self-attention can take the path, as far as its modules tell: plain
        Ulysses self-attention with BF16 nn.Linear projections and nn.RMSNorm QK norms. The
        call checks the rest (_qk_first_eligible)."""
        if (
            self.is_cross_attention
            or self.attention_function is not USP
            or attn.add_k_proj is not None
            or getattr(attn, "fused_projections", False)
        ):
            return False
        linears = (attn.to_q, attn.to_k, attn.to_v)
        if not all(usp_sdma.plain_linear(m) and m.weight.dtype == torch.bfloat16 for m in linears):
            return False
        return all(
            isinstance(norm, torch.nn.RMSNorm) and norm.weight is not None and norm.weight.dtype == torch.bfloat16
            for norm in (attn.norm_q, attn.norm_k)
        )

    def _call_standard(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if usp_sdma.overlap_qkv() and self._tp_qk_first_eligible(attn, encoder_hidden_states, rotary_emb):
            output = self._call_tp_qk_first(attn, hidden_states, rotary_emb)
            if output is not None:
                return self._finish_attention(attn, output.transpose(1, 2), hidden_states.dtype, None)
        if usp_sdma.overlap_qkv() and self._qk_first_eligible(attn, hidden_states, encoder_hidden_states, rotary_emb):
            output = self._call_qk_first(attn, hidden_states, rotary_emb)
            if output is not None:
                return self._finish_attention(attn, output.transpose(1, 2), hidden_states.dtype, None)
        return super()._call_standard(attn, hidden_states, encoder_hidden_states, rotary_emb)

    def _tp_qk_first_eligible(self, attn, encoder_hidden_states, rotary_emb) -> bool:
        """Q/K before V under TP: self-attention with the FP8 fused TP sites (XDIT_TP_FP8 with
        XDIT_TP_COMM_OVERLAP=symm_mem), whose all-gather can keep the gathered rows for V."""
        return (
            self.tensor_parallel
            and not self.is_cross_attention
            and self.attention_function is USP
            and self.tp_fused == "symm_mem"
            and encoder_hidden_states is None
            and attn.add_k_proj is None
            and rotary_emb is not None
            and not torch.is_grad_enabled()
            and getattr(attn, "fused_projections", False)
            and getattr(attn.to_qkv, "tp_fp8_weight", None) is not None
        )

    def _call_tp_qk_first(self, attn, hidden_states, rotary_emb):
        """TP x Ulysses Q/K before V: the TP all-gather feeds the Q/K columns of the QKV GEMM and keeps
        the gathered rows; Q/K are normed over the TP group and rotated, head group 0's are sent, and
        V's GEMM reads the kept rows while they are in flight (usp_sdma.qk_first_grouped_attention).
        USP's [b, H, s, d], or None."""
        from yunchang.globals import PROCESS_GROUP

        width = attn.to_qkv.tp_fp8_weight.shape[1] // 3
        qk, project_rest = tp_fp8_column_parallel_split(hidden_states, attn.to_qkv, 2 * width, "attn1_qk")
        query, key = qk.chunk(2, dim=-1)
        query, key = tp_rms_norm((query, key), (attn.norm_q, attn.norm_k), "attn1_qk_norm")
        query, key, _ = self._rope_reference(attn, query, key, key, rotary_emb)

        def project_v(token):
            return project_rest(token).unflatten(2, (attn.heads, -1)).transpose(1, 2)

        output = usp_sdma.qk_first_grouped_attention(
            hidden_states,
            query.transpose(1, 2),
            key.transpose(1, 2),
            project_v,
            backend=None,
            attention_kwargs=self.attention_kwargs,
            group=PROCESS_GROUP.ULYSSES_PG,
            rows=qk.shape[1],
        )
        if output is None:
            # The rows the split all-gather kept must be used before the next TP all-gather.
            value = project_v(qk.new_empty(0)).transpose(1, 2)
            output = self.attention_function(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                backend=None,
                attention_kwargs=self.attention_kwargs,
                head_balance_layer=attn,
                attn_layer=attn,
            )
        return output

    def _qk_first_eligible(self, attn, hidden_states, encoder_hidden_states, rotary_emb) -> bool:
        """Whether this call can send Q/K before V: plain self-attention in inference, batch one and
        BF16 (the packed norm/RoPE producer's layout), with resident unhooked nn.Linear
        projections and nn.RMSNorm QK norms."""
        if (
            self.is_cross_attention
            or self.attention_function is not USP
            or encoder_hidden_states is not None
            or attn.add_k_proj is not None
            or rotary_emb is None
            or torch.is_grad_enabled()
            or getattr(attn, "fused_projections", False)
        ):
            return False
        if hidden_states.device.type != "cuda" or hidden_states.dtype != torch.bfloat16 or hidden_states.shape[0] != 1:
            return False
        linears = (attn.to_q, attn.to_k, attn.to_v)
        if len({m.weight.shape for m in linears}) != 1 or attn.to_q.weight.shape[0] % attn.heads:
            return False
        for module in linears:
            if (
                type(module) is not torch.nn.Linear
                or hasattr(module.weight, "placements")
                or module.weight.device != hidden_states.device
                or module.weight.dtype != hidden_states.dtype
                or module._forward_hooks
                or module._forward_pre_hooks
                or hasattr(module, "_hf_hook")
            ):
                return False
        return all(
            isinstance(norm, torch.nn.RMSNorm) and norm.weight is not None and norm.weight.dtype == hidden_states.dtype
            for norm in (attn.norm_q, attn.norm_k)
        )

    def _call_qk_first(self, attn, hidden_states, rotary_emb):
        """Q/K normed, rotated and packed peer by peer in one kernel, sent; then V. USP's
        [b, H, s, d], or None if the call cannot take the SDMA path."""
        from yunchang.globals import PROCESS_GROUP

        group = PROCESS_GROUP.ULYSSES_PG
        world = torch.distributed.get_world_size(group)

        def project_qk():
            query, key = attn.to_q(hidden_states), attn.to_k(hidden_states)
            return fused_qk_norm_rope_packed(query, key, attn.norm_q, attn.norm_k, *rotary_emb, attn.heads, world)

        def project_v(token):
            value = torch.ops.xfuser.sdma_ulysses_linear(token, hidden_states, attn.to_v.weight, attn.to_v.bias)
            return value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        return usp_sdma.qk_first_attention(
            hidden_states,
            project_qk,
            project_v,
            heads=attn.heads,
            head_dim=attn.to_q.weight.shape[0] // attn.heads,
            backend=None,
            attention_kwargs=self.attention_kwargs,
            group=group,
        )


def _local_tp_pad_rows(seq_len: int, sp_rank: int, sp_len: int, tp_rank: int, tp_size: int, chunks: int) -> int:
    """How many of this rank's rows are pad under TP, for a real length `seq_len`.

    Of its SP rank's `sp_len` rows, a TP rank holds blocks tp_rank, tp_size +
    tp_rank, ... of tp_size * chunks (see tp_split_rows). Its rows keep global
    order, so the pad, which trails the global sequence, trails them too.
    """
    block = sp_len // (tp_size * chunks)
    first_row = sp_rank * sp_len
    return sum(min(block, max(0, first_row + (c * tp_size + tp_rank + 1) * block - seq_len)) for c in range(chunks))


class xFuserWanTPTransformerBlock(WanTransformerBlock):
    """WanTransformerBlock over a TP-sharded residual stream.

    `xFuserWanTransformer3DWrapper.shard_for_tensor_parallel` gives blocks this
    class. The residual rows are split across the TP group. Each attention and
    the FFN all-gather the group's rows into their column-parallel projections
    and reduce-scatter their row-parallel output projections (the attention
    processors do so inside attn1/attn2), in `tp_chunks` pipelined chunks or
    through the fused path `tp_fused` names. Row-parallel biases are added once, after the reduction. Cross-attention
    instead runs unsharded on the local rows when `shard_cross_attention` is
    False. The modulation, norms and gated residuals are diffusers', applied to
    the local rows.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention; replicated, it also runs on the local rows, against the whole text.
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward: diffusers' GELU(proj) -> dropout -> linear, its two linears split.
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        act = self.ffn.net[0]
        ff_hidden = act.gelu(tp_column_parallel(norm_hidden_states, act.proj, "ffn_up", self.tp_chunks, self.tp_fused))
        ff_output = tp_row_parallel(
            self.ffn.net[1](ff_hidden), self.ffn.net[2], self.ffn_out_bias, "ffn_down", self.tp_chunks, self.tp_fused
        )
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class xFuserWanTransformer3DWrapper(WanTransformer3DModel):
    def __init__(
        self,
        patch_size: Tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            patch_size,
            num_attention_heads,
            attention_head_dim,
            in_channels,
            out_channels,
            text_dim,
            freq_dim,
            ffn_dim,
            num_layers,
            cross_attn_norm,
            qk_norm,
            eps,
            image_dim,
            added_kv_proj_dim,
            rope_max_seq_len,
            pos_embed_seq_len,
        )
        self.attention_kwargs = attention_kwargs
        # Self-attention also needs the SP pad length, published per forward. A
        # model built without attention_kwargs gets a dict holding only that, so
        # the layout options it never had (`thw`) stay unpublished. The key is
        # always present because torch.compile guards on the dict's key set.
        # Cross-attention is not given this dict: its keys are the text, which
        # carries no SP pad.
        self._self_attn_kwargs = attention_kwargs if attention_kwargs is not None else {}
        self._self_attn_kwargs.setdefault("valid_kv_len", None)
        for block in self.blocks:
            block.attn1.processor = xFuserWanAttnProcessor(attention_kwargs=self._self_attn_kwargs)
            block.attn2.processor = xFuserWanAttnProcessor(
                use_ulysses_parallel_attention=False, is_cross_attention=True
            )
            # Per-layer head permutation buffer for the Ulysses block-sparse head
            # balancer (read/updated in-place inside USP; identity = no balancing).
            # Registered pre-compile and non-persistent so it stays out of the
            # state_dict and is captured as a graph input by torch.compile.
            block.attn1.register_buffer(
                "head_perm",
                torch.arange(num_attention_heads, dtype=torch.long),
                persistent=False,
            )
        # attn2 is cross-attention over the text encoder: no Ulysses collective.
        register_fp8_comms_eligible_modules(self, [block.attn1 for block in self.blocks])
        # Pad rows of this rank's chunk, set per forward; 0 leaves the blocks untouched.
        self._sp_pad = {"local_rows": 0}
        for block in self.blocks:
            block.register_forward_hook(zero_sp_pad_rows_hook(self._sp_pad))
        # Tensor parallelism is on once shard_for_tensor_parallel has split the weights.
        self._tp_size = 1
        self._tp_rank = 0
        self._tp_chunks = 1

    def shard_for_tensor_parallel(
        self, shard_cross_attention: bool = True, overlap_chunks: int = 1, fused_comm: Optional[str] = None
    ) -> None:
        """Split the blocks' weights across the TP group, Megatron style.

        Q/K/V and the FFN's first projection are column-parallel and their
        output projections row-parallel, so each rank holds 1/tp of the heads
        and of the FFN's inner features. Q/K norm weights split with the heads,
        and each attention's projections are fused (`to_qkv`, or `to_kv` with
        `to_q` for cross-attention) so one gather feeds one GEMM. The
        embeddings, modulation tables and output head stay replicated. Blocks
        become xFuserWanTPTransformerBlock, and forward() splits the residual
        rows over the TP group as well as the SP group.

        With `shard_cross_attention` False, cross-attention keeps its full
        weights and runs on each rank's own rows, without communication.
        `overlap_chunks` > 1 pipelines each collective against its GEMM in that
        many chunks, with the residual rows block-cyclic (see layers/tensor_parallel.py).
        `fused_comm="symm_mem"` runs each pair through torch's symm_mem fused
        ops instead, on contiguous rows; `fused_comm="sdma"` (ROCm) gathers
        each all-gather + GEMM on SDMA copy engines in `overlap_chunks` pieces
        (layers/sdma_all_gather.py), reduce-scatters as with chunks alone.
        """
        if fused_comm not in (None, "symm_mem", "sdma"):
            raise ValueError(f"fused_comm must be None, 'symm_mem' or 'sdma'; got {fused_comm!r}")
        if fused_comm == "symm_mem" and overlap_chunks != 1:
            raise ValueError("fused_comm='symm_mem' pipelines on its own; it takes no overlap_chunks")
        if fused_comm == "symm_mem":
            # On torch's patched ROCm SDMA ops: inboxes over hipIpc, not torch's rendezvoused workspace.
            ipc_workspace.install_torch_sdma_inbox()
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size == 1 or self._tp_size > 1:
            return
        tp_rank = get_tensor_model_parallel_rank()
        if self.config.num_attention_heads % tp_size != 0:
            raise ValueError(
                f"{self.config.num_attention_heads} attention heads do not split across {tp_size} TP ranks"
            )
        for block in self.blocks:
            block.shard_cross_attention = shard_cross_attention
            block.tp_chunks = overlap_chunks
            block.tp_fused = fused_comm
            for name in ("attn1", "attn2") if shard_cross_attention else ("attn1",):
                attn = getattr(block, name)
                if getattr(attn, "fused_projections", False) or attn.add_k_proj is not None:
                    raise NotImplementedError(
                        "Wan tensor parallelism supports neither fused projections nor I2V image keys"
                    )
                for linear in (attn.to_q, attn.to_k, attn.to_v):
                    shard_linear_(linear, 0, tp_rank, tp_size)
                out_bias = shard_linear_(attn.to_out[0], 1, tp_rank, tp_size)
                attn.register_parameter("tp_out_bias", torch.nn.Parameter(out_bias, requires_grad=False))
                shard_rms_norm_(attn.norm_q, tp_rank, tp_size)
                shard_rms_norm_(attn.norm_k, tp_rank, tp_size)
                attn.fuse_projections()
                # fuse_projections keeps the separate linears; the TP processor reads only the fused ones.
                for unused in ("to_k", "to_v") if attn.is_cross_attention else ("to_q", "to_k", "to_v"):
                    delattr(attn, unused)
                attn.heads //= tp_size
                attn.inner_dim //= tp_size
                attn.kv_inner_dim //= tp_size
                attn.processor.tensor_parallel = True
                attn.processor.tp_chunks = overlap_chunks
                attn.processor.tp_fused = fused_comm
            # The Ulysses head balancer permutes this rank's heads only.
            block.attn1.head_perm = torch.arange(block.attn1.heads, device=block.attn1.head_perm.device)
            shard_linear_(block.ffn.net[0].proj, 0, tp_rank, tp_size)
            out_bias = shard_linear_(block.ffn.net[2], 1, tp_rank, tp_size)
            block.register_parameter("ffn_out_bias", torch.nn.Parameter(out_bias, requires_grad=False))
            block.__class__ = xFuserWanTPTransformerBlock
            if fused_comm == "symm_mem" and envs.environment_variables["XDIT_TP_FP8"]().strip() == "1":
                # Row-wise FP8 GEMMs at every TP site (the linears tp_column/row_parallel take).
                for name in ("attn1", "attn2") if shard_cross_attention else ("attn1",):
                    attn = getattr(block, name)
                    # Cross-attention's K/V come from the text context on each rank, not through a TP site.
                    quantize_linear_fp8_(attn.to_q if attn.is_cross_attention else attn.to_qkv)
                    quantize_linear_fp8_(attn.to_out[0])
                quantize_linear_fp8_(block.ffn.net[0].proj)
                quantize_linear_fp8_(block.ffn.net[2])
        self._tp_size = tp_size
        self._tp_rank = tp_rank
        self._tp_chunks = overlap_chunks

    def _update_vsa_attention_kwargs(self, timestep: torch.LongTensor) -> None:
        """Publish the current AITER VSA schedule values to its backend."""
        if self.attention_kwargs is None or not self.attention_kwargs.get("vsa_drop_rates"):
            return

        runtime_state = get_runtime_state()
        step_index, num_steps = runtime_state.advance_vsa_schedule(float(timestep.reshape(-1)[0].item()))
        self.attention_kwargs["vsa_step_index"] = step_index
        self.attention_kwargs["vsa_num_steps"] = num_steps
        effective_drop_rate = jenga_scheduled_drop_rate(
            step_index,
            num_steps,
            self.attention_kwargs["vsa_drop_rates"],
        )
        self.attention_kwargs["vsa_effective_drop_rate"] = effective_drop_rate
        self.attention_kwargs["vsa_use_dense"] = effective_drop_rate <= 0.25

    def _chunk_and_pad_sequence(
        self, x: torch.Tensor, sp_world_rank: int, sp_world_size: int, pad_amount: int, dim: int
    ) -> torch.Tensor:
        if pad_amount > 0:
            if dim < 0:
                dim = x.ndim + dim
            pad_shape = list(x.shape)
            pad_shape[dim] = pad_amount
            x = torch.cat(
                [
                    x,
                    torch.zeros(
                        pad_shape,
                        dtype=x.dtype,
                        device=x.device,
                    ),
                ],
                dim=dim,
            )
        x = torch.chunk(x, sp_world_size, dim=dim)[sp_world_rank]
        return x

    def _gather_and_unpad(self, x: torch.Tensor, pad_amount: int, dim: int) -> torch.Tensor:
        x = get_sp_group().all_gather(x, dim=dim)
        size = x.size(dim)
        return x.narrow(dim=dim, start=0, length=size - pad_amount)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        self._update_vsa_attention_kwargs(timestep)
        get_runtime_state().increment_step_counter()

        sp_world_rank = get_sequence_parallel_rank()
        sp_world_size = get_sequence_parallel_world_size()

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        if self.attention_kwargs is not None:
            self.attention_kwargs["thw"] = (
                post_patch_num_frames,
                post_patch_height,
                post_patch_width,
            )

        # 1. RoPE
        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        # Without the copy, the strided layout slows every elementwise op in the residual stream.
        hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            # We only reach this for Wan2.1, when doing cross attention with image embeddings
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # Under tensor parallelism the residual rows split once more, over the TP
        # ranks of each SP rank (in tp_chunks blocks each, see tp_split_rows): the
        # blocks all-gather the TP group's rows back into this SP rank's chunk
        # before attention, which RoPE is chunked for.
        rows_multiple = sp_world_size * self._tp_size * self._tp_chunks
        if self._tp_size > 1:
            # XDIT_TP_ROW_ALIGN: each TP rank's rows a multiple of it (e.g. a GEMM tile height, for kernels that
            # take whole tiles per rank); the extra rows are SP pad, masked like any other.
            rows_multiple *= int(envs.environment_variables["XDIT_TP_ROW_ALIGN"]())

        # Part of sequence parallel: given the resolution, we may need to pad the sequence length to match this prior to chunking
        seq_len = hidden_states.shape[1]
        attention_a2a_padding = (
            get_fused_a2a_mode() > 0
            and get_ulysses_parallel_world_size() > 1
            and get_ring_parallel_world_size() == 1
            and not get_runtime_state().runtime_config.use_sparseattn_head_balance
        )
        pad_multiple = fused_a2a_pad_multiple(sp_world_size) if attention_a2a_padding else sp_world_size
        # Under TP the rows must also split into tp_size * tp_chunks blocks per SP rank.
        pad_multiple = math.lcm(pad_multiple, rows_multiple)
        pad_amount = (pad_multiple - (seq_len % pad_multiple)) % pad_multiple
        # Padded keys would act as an attention sink. Publish the real global
        # length for ordinary USP slicing or direct packed K/V output.
        self._self_attn_kwargs["valid_kv_len"] = seq_len if pad_amount else None

        def split_rows(x):
            x = self._chunk_and_pad_sequence(x, sp_world_rank, sp_world_size, pad_amount, dim=1)
            if self._tp_size > 1:
                x = tp_split_rows(x, self._tp_rank, self._tp_size, self._tp_chunks)
            return x

        sp_len = (seq_len + pad_amount) // sp_world_size
        hidden_states = split_rows(hidden_states)
        if self._tp_size > 1:
            self._sp_pad["local_rows"] = _local_tp_pad_rows(
                seq_len, sp_world_rank, sp_len, self._tp_rank, self._tp_size, self._tp_chunks
            )
        else:
            self._sp_pad["local_rows"] = local_sp_pad_rows(seq_len, hidden_states.shape[1], sp_world_rank)

        if ts_seq_len is not None:  # (wan2.2 ti2v)
            temb = split_rows(temb)
            timestep_proj = split_rows(timestep_proj)

        freqs_cos, freqs_sin = rotary_emb

        def get_rotary_emb_chunk(freqs, pad_amount):
            freqs = self._chunk_and_pad_sequence(freqs, sp_world_rank, sp_world_size, pad_amount, dim=1)
            return freqs

        freqs_cos = get_rotary_emb_chunk(freqs_cos, pad_amount)
        freqs_sin = get_rotary_emb_chunk(freqs_sin, pad_amount)
        rotary_emb = (freqs_cos, freqs_sin)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        if self._tp_size > 1:
            hidden_states = tp_gather_rows(hidden_states, self._tp_chunks)
        hidden_states = self._gather_and_unpad(hidden_states, pad_amount, dim=-2)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
