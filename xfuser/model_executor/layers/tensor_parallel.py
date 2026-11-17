"""Megatron-style tensor parallelism over a sequence-sharded residual stream.

Between layers each TP rank holds 1/tp of the sequence rows with every
feature. A column-parallel linear all-gathers its input rows over the TP group
and produces 1/tp of the output features; a row-parallel linear consumes those
features and reduce-scatters its partial sums back to the rank's rows.

`tp_column_parallel` and `tp_row_parallel` run those pairs. With one chunk
they are a plain collective next to the GEMM (`tp_all_gather_seq` /
`tp_reduce_scatter_seq`, labelled `tp_ag:<site>` / `tp_rs:<site>` in eager
traces). With more, the collective is pipelined against the GEMM: RCCL runs
chunk by chunk on a side stream, so chunk c's GEMM overlaps chunk c+1's
all-gather, or chunk c's reduce-scatter overlaps chunk c+1's GEMM.

Chunking needs the rows block-cyclic within the TP group: of a sequence split
into tp*chunks blocks, TP rank r holds blocks r, tp+r, 2tp+r, ... Each chunk's
all-gather output and reduce-scatter input are then contiguous rows of the
natural order. `tp_split_rows` / `tp_gather_rows` convert between the layouts;
with one chunk the layout is plain contiguous chunks.

With `fused="symm_mem"`, torch's symmetric-memory ops run each pair instead
(`symm_mem.fused_all_gather_matmul` / `fused_matmul_reduce_scatter`). They
pipeline peer-to-peer copies against per-shard GEMMs themselves, on the
contiguous layout. They work over NVLink; on torch 2.13 / ROCm 7.16 they
segfault beyond ~10 MB. A torch patched with patches/pytorch runs them on
ROCm's SDMA engines (TORCH_SYMM_MEM_ENABLE_ROCM_SDMA=1). With `fused="sdma"`
(ROCm), each all-gather + GEMM gathers on SDMA copy engines, all peers at
once, in `chunks` pieces on the block-cyclic layout, each piece's GEMM starting
as it lands
(layers/sdma_all_gather.py); reduce-scatters run as with `fused` unset, or with
XDIT_TP_SDMA_REDUCE_SCATTER=on push their partial sums on SDMA engines and reduce
them chunk by chunk.

With `fused="symm_mem"`, a linear given a row-wise FP8 weight by `quantize_linear_fp8_`
(XDIT_TP_FP8=1) runs the FP8 ops instead (`fused_all_gather_scaled_matmul` /
`fused_scaled_matmul_reduce_scatter`): activations are quantized row-wise per call, the
gathered rows move as FP8 (half the bytes), and the GEMMs are scaled FP8.
"""

from typing import List, Optional, Sequence

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory  # noqa: F401  registers torch.ops.symm_mem
import torch.nn.functional as F
from torch import nn
from torch.profiler import record_function

import xfuser.envs as envs
from xfuser.core.distributed.parallel_state import get_tp_group
from xfuser.model_executor.layers import sdma_all_gather  # noqa: F401  registers xfuser::tp_sdma_all_gather_linear


def shard_linear_(linear: nn.Linear, dim: int, tp_rank: int, tp_size: int) -> Optional[torch.Tensor]:
    """Keep this rank's 1/tp_size of `linear` along weight `dim`, in place.

    dim 0 splits the output features (column-parallel); the bias splits with
    them. dim 1 splits the input features (row-parallel); each rank's output
    is then a partial sum, so the bias is removed from the linear and
    returned, to be added once after the reduction.
    """
    weight = linear.weight.data
    if weight.size(dim) % tp_size != 0:
        raise ValueError(f"{tuple(weight.shape)} weight does not split {tp_size} ways along dim {dim}")
    linear.weight.data = weight.chunk(tp_size, dim=dim)[tp_rank].contiguous()
    if dim == 0:
        linear.out_features //= tp_size
        if linear.bias is not None:
            linear.bias.data = linear.bias.data.chunk(tp_size)[tp_rank].contiguous()
        return None
    linear.in_features //= tp_size
    bias = linear.bias
    linear.bias = None
    return None if bias is None else bias.data


def shard_rms_norm_(norm: nn.RMSNorm, tp_rank: int, tp_size: int) -> None:
    """Keep this rank's 1/tp_size of an RMSNorm's weight, for `tp_rms_norm`."""
    full = norm.normalized_shape[-1]
    if full % tp_size != 0:
        raise ValueError(f"RMSNorm over {full} features does not split {tp_size} ways")
    norm.full_features = full
    norm.normalized_shape = (*norm.normalized_shape[:-1], full // tp_size)
    if norm.weight is not None:
        norm.weight.data = norm.weight.data.chunk(tp_size)[tp_rank].contiguous()


FP8 = torch.float8_e4m3fn


def _rowwise_fp8(x: torch.Tensor):
    """Row-wise FP8 E4M3 over the last dim: values and fp32 scales [..., 1]."""
    amax = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12)
    scale = amax / torch.finfo(FP8).max
    return (x.float() / scale).to(FP8), scale


def quantize_linear_fp8_(linear: nn.Linear) -> None:
    """Give a (sharded) linear a row-wise FP8 copy of its weight for the FP8 fused TP ops: per output
    channel scales, stored as [in, out] column-major (the B layout of torch's scaled matmuls). The bf16
    weight stays for the paths that do not use it."""
    w_q, w_s = _rowwise_fp8(linear.weight.data)
    linear.tp_fp8_weight = w_q.t()  # [in, out], column-major
    linear.tp_fp8_scale = w_s.view(1, -1)  # [1, out]


def _seq_rows(x: torch.Tensor) -> torch.Tensor:
    """(batch, rows, features) -> (rows * batch, features), sequence-major (a view when batch is 1)."""
    return x.transpose(0, 1).reshape(-1, x.shape[-1])


def _fp8_column_parallel(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    batch, rows, _ = x.shape
    xq, xs = _rowwise_fp8(_seq_rows(x))
    _, (out,) = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
        xq,
        [linear.tp_fp8_weight],
        xs,
        [linear.tp_fp8_scale],
        gather_dim=0,
        biases=[None],
        result_scales=[None],
        # hipBLASLt's row-wise scaled GEMM writes bf16 only
        out_dtypes=[torch.bfloat16],
        use_fast_accum=[False],
        group_name=get_tp_group().device_group.group_name,
    )
    out = out.view(-1, batch, out.shape[-1]).transpose(0, 1).to(x.dtype)
    return out if linear.bias is None else out + linear.bias


def _fp8_row_parallel(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    batch = x.shape[0]
    xq, xs = _rowwise_fp8(_seq_rows(x))
    n = linear.tp_fp8_weight.shape[1]
    out = torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter(
        xq,
        linear.tp_fp8_weight,
        xs,
        linear.tp_fp8_scale,
        "sum",
        0,
        0,
        get_tp_group().device_group.group_name,
        [xq.shape[0], n],
        out_dtype=torch.bfloat16,
    )
    return out.view(-1, batch, n).transpose(0, 1).to(x.dtype)


def tp_all_gather_seq(x: torch.Tensor, site: str) -> torch.Tensor:
    """All-gather the TP group's sequence rows (dim 1) ahead of a column-parallel GEMM."""
    with record_function(f"tp_ag:{site}"):
        return get_tp_group().all_gather(x, dim=1)


def tp_reduce_scatter_seq(x: torch.Tensor, site: str) -> torch.Tensor:
    """Sum a row-parallel GEMM's partial outputs and keep this rank's sequence rows (dim 1)."""
    with record_function(f"tp_rs:{site}"):
        return get_tp_group().reduce_scatter(x, dim=1)


def tp_rms_norm(tensors: Sequence[torch.Tensor], norms: Sequence[nn.RMSNorm], site: str) -> List[torch.Tensor]:
    """RMSNorm each tensor over features split across the TP group.

    Each tensor holds this rank's 1/tp of its norm's features and each norm
    was sharded by `shard_rms_norm_`. The squares are summed locally in fp32
    and one all-reduce completes the sums for every tensor at once.
    """
    sums = [t.float().pow(2).sum(dim=-1, keepdim=True) for t in tensors]
    flat = torch.cat([s.flatten() for s in sums])
    with record_function(f"tp_ar:{site}"):
        flat = get_tp_group().all_reduce(flat)
    outputs = []
    for t, local, total, norm in zip(tensors, sums, flat.split([s.numel() for s in sums]), norms):
        eps = norm.eps if norm.eps is not None else torch.finfo(t.dtype).eps
        out = t.float() * torch.rsqrt(total.view_as(local) / norm.full_features + eps)
        if norm.weight is not None:
            out = out * norm.weight.float()
        outputs.append(out.to(t.dtype))
    return outputs


def tp_split_rows(x: torch.Tensor, tp_rank: int, tp_size: int, chunks: int) -> torch.Tensor:
    """This TP rank's rows (dim 1) of `x`: blocks r, tp+r, ... of tp*chunks, stacked."""
    batch, rows, *rest = x.shape
    blocks = x.view(batch, chunks, tp_size, rows // (chunks * tp_size), *rest)
    return blocks[:, :, tp_rank].reshape(batch, rows // tp_size, *rest)


def tp_gather_rows(x: torch.Tensor, chunks: int) -> torch.Tensor:
    """All-gather the TP group's rows (dim 1) back into natural order; the inverse of `tp_split_rows`."""
    tp_size = get_tp_group().world_size
    gathered = get_tp_group().all_gather(x, dim=1)
    if chunks == 1:
        return gathered
    batch, rows, *rest = gathered.shape
    blocks = gathered.view(batch, tp_size, chunks, rows // (tp_size * chunks), *rest)
    return blocks.transpose(1, 2).reshape(batch, rows, *rest)


def _rocm_sdma_ops():
    """torch's patched ROCm SDMA symm_mem module when it is enabled, else None."""
    import os

    if torch.version.hip is None or os.environ.get("TORCH_SYMM_MEM_ENABLE_ROCM_SDMA") != "1":
        return None
    try:
        import torch.distributed._symmetric_memory._rocm_sdma as rocm_sdma
    except ImportError:
        return None
    return rocm_sdma


def tp_gather_apply_scatter(
    tensors: Sequence[torch.Tensor], fn, out_features: int, out_dtype: torch.dtype, site: str, chunks: int = 4
) -> torch.Tensor:
    """reduce_scatter(fn(all_gather(tensors))) over the TP group, pipelined on SDMA engines.

    For token-parallel functions such as an MoE whose experts are TP-sharded: every row's output
    is a partial sum over the TP ranks. `tensors` are this rank's rows, (rows, features_i) each
    (e.g. activations, or quantized activations and their scales); `fn(*pieces)` maps rows of
    every tensor to (rows, out_features) partial sums. Returns this rank's (rows, out_features),
    summed in fp32 and cast to out_dtype.

    On torch's patched ROCm SDMA path one call pipelines both collectives: piece c of every rank's
    rows lands, fn runs on it, and its partial sums are pushed to their owners while fn runs on
    piece c + 1. Otherwise: RCCL all-gather, fn, RCCL reduce-scatter.
    """
    group = get_tp_group()
    rows = tensors[0].shape[0]
    rocm_sdma = _rocm_sdma_ops()
    while chunks > 1 and rows % chunks:
        chunks -= 1
    if rocm_sdma is None or torch.cuda.is_current_stream_capturing():
        with record_function(f"tp_ag:{site}"):
            full = [group.all_gather(t.contiguous(), dim=0) for t in tensors]
        partial = fn(*full)
        with record_function(f"tp_rs:{site}"):
            return group.reduce_scatter(partial.to(out_dtype), dim=0)
    with record_function(f"tp_gas:{site}"):
        return _sdma_gather_apply_scatter(rocm_sdma, tensors, fn, out_features, out_dtype, chunks)


def _sdma_gather_apply_scatter(rs, tensors, fn, out_features, out_dtype, chunks):
    """Schedule: fn on this rank's own rows of the first half of the chunks (no wait: hides the first
    gathered piece), then on the peers' rows chunk by chunk as they land (their partial sums pushed
    to their owners), then on the own rows of the second half (hides the last pushes in flight).
    Chunk c of this rank's output is reduced once its own partial and the peers' partials are in."""
    group_name = get_tp_group().device_group.group_name
    g = rs._group(group_name)
    world, rank = g.world, g.rank
    peers = list(g.peers)
    xs = [t.contiguous() for t in tensors]
    rows = xs[0].shape[0]
    block = rows // chunks
    N = out_features
    lib = rs._load()
    cur = torch.cuda.current_stream()
    # one inbox: the gathered inputs [world, rows, f_i] each (this rank's slot unused), then the partial
    # sums [chunks, world, block, N]; flag rows 0..chunks-1 gather pieces, chunks..2*chunks-1 scatter pieces
    regions = [((world, rows, t.shape[1]), t.dtype) for t in xs] + [((chunks, world, block, N), out_dtype)]
    ws, inboxes, offsets, flags, bases, flag_offset = rs._inbox(group_name, rank, world, regions, 2 * chunks)
    gathered, recv = inboxes[:-1], inboxes[-1]
    row_bytes = [t.shape[1] * t.element_size() for t in xs]
    block_bytes = block * N * recv.element_size()
    # this rank's partial sums for the peers' rows, piece by piece, owner-major within a piece
    partials = torch.empty(chunks, world, block, N, dtype=out_dtype, device=xs[0].device)
    # Two plans: a plan's pushes must all carry the same number of copies per peer (its completion
    # signals are indexed by piece x peer x copy), and the gather moves len(xs) tensors, the scatter one.
    gather = rs._Plan(lib, g, chunks, len(xs))
    scatter = rs._Plan(lib, g, chunks, 1)
    for c in range(chunks):
        # this rank's piece c straight from its own tensors into every peer's inbox slot for it
        dst = [off + (rank * rows + c * block) * size for off, size in zip(offsets[:-1], row_bytes)]
        src = [t.data_ptr() + c * block * size for t, size in zip(xs, row_bytes)]
        gather.push(
            0,
            [bases[p] + a for p in peers for a in dst],
            [a for _ in peers for a in src],
            [block * size for size in row_bytes],
            [bases[p] + flag_offset + (c * world + rank) * 4 for p in peers],
        )
        scatter.push(
            c,
            [bases[p] + offsets[-1] + (c * world + rank) * block_bytes for p in peers],
            [partials[c, p].data_ptr() for p in peers],
            [block_bytes],
            [bases[p] + flag_offset + ((chunks + c) * world + rank) * 4 for p in peers],
        )
    flags.zero_()
    flags[:, rank] = 1
    out = torch.empty(rows, N, dtype=out_dtype, device=xs[0].device)
    own = torch.empty(rows, N, dtype=out_dtype, device=xs[0].device)
    pieces = [inbox.view(world, chunks, block, -1) for inbox in gathered]
    first = list(range((chunks + 1) // 2))
    last = list(range((chunks + 1) // 2, chunks))
    own_done = set()
    reduced = []

    def own_rows(cs):
        if not cs:
            return
        lo, hi = cs[0] * block, (cs[-1] + 1) * block
        own[lo:hi].copy_(fn(*[t[lo:hi] for t in xs]))
        own_done.update(cs)

    def reduce_ready(peers_done):
        for c in range(chunks):
            if c in reduced or c not in own_done or c >= peers_done:
                continue
            scatter.wait(cur, flags[chunks + c], c)
            recv[c, rank].copy_(own[c * block : (c + 1) * block])
            out[c * block : (c + 1) * block] = torch.sum(recv[c], dim=0, dtype=torch.float32)
            reduced.append(c)

    rs._barrier(ws)
    in_last_barrier = False
    try:
        gather.start(cur)
        scatter.start(cur)
        gather.release(0, cur)
        own_rows(first)
        for c in range(chunks):
            gather.wait(cur, flags[c], c)
            y = fn(*[p[peers, c].reshape(len(peers) * block, -1) for p in pieces]).view(len(peers), block, N)
            for i, p in enumerate(peers):  # (advanced-index assignment would write into a copy)
                partials[c, p].copy_(y[i])
            scatter.release(c, cur)
            reduce_ready(c)  # chunks whose peer partials were released one step earlier
        own_rows(last)
        reduce_ready(chunks)
        in_last_barrier = True
        rs._barrier(ws)
    except BaseException as e:
        if not (gather.capturing or scatter.capturing):
            rs._abandon(e, ws, in_last_barrier)
        raise
    return out


def tp_column_parallel(
    x: torch.Tensor, linear: nn.Linear, site: str, chunks: int, fused: Optional[str] = None
) -> torch.Tensor:
    """`linear` over the TP group's rows of `x` (dim 1): all-gather, then the column-parallel GEMM."""
    if fused == "sdma":
        with record_function(f"tp_ag:{site}"):
            return torch.ops.xfuser.tp_sdma_all_gather_linear(
                x,
                linear.weight,
                linear.bias,
                get_tp_group().device_group.group_name,
                chunks,
                envs.environment_variables["XDIT_TP_SDMA_TRIGGER"]().strip().lower(),
                envs.environment_variables["XDIT_TP_SDMA_DIRECTION"]().strip().lower(),
            )
    if fused == "symm_mem" and getattr(linear, "tp_fp8_weight", None) is not None:
        with record_function(f"tp_ag_fp8:{site}"):
            return _fp8_column_parallel(x, linear)
    if fused:
        with record_function(f"tp_ag:{site}"):
            _, (out,) = torch.ops.symm_mem.fused_all_gather_matmul(
                x, [linear.weight.t()], 1, get_tp_group().device_group.group_name, return_A=False
            )
        return out if linear.bias is None else out + linear.bias
    if chunks == 1:
        return linear(tp_all_gather_seq(x, site))
    group = get_tp_group()
    with record_function(f"tp_ag:{site}"):
        return torch.ops.xfuser.tp_chunked_all_gather_linear(
            x, linear.weight, linear.bias, group.device_group.group_name, group.world_size, chunks
        )


def tp_row_parallel(
    x: torch.Tensor,
    linear: nn.Linear,
    bias: Optional[torch.Tensor],
    site: str,
    chunks: int,
    fused: Optional[str] = None,
) -> torch.Tensor:
    """Row-parallel `linear` over all rows of `x`, reduce-scattered to this rank's rows, plus `bias` once."""
    if fused == "symm_mem" and getattr(linear, "tp_fp8_weight", None) is not None:
        with record_function(f"tp_rs_fp8:{site}"):
            out = _fp8_row_parallel(x, linear)
    elif fused == "symm_mem":
        with record_function(f"tp_rs:{site}"):
            out = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                x, linear.weight.t(), "sum", 1, get_tp_group().device_group.group_name
            )
    elif chunks == 1:
        out = tp_reduce_scatter_seq(linear(x), site)
    elif fused == "sdma" and envs.environment_variables["XDIT_TP_SDMA_REDUCE_SCATTER"]().strip().lower() == "on":
        group = get_tp_group()
        with record_function(f"tp_rs:{site}"):
            out = torch.ops.xfuser.tp_sdma_linear_reduce_scatter(
                x,
                linear.weight,
                group.device_group.group_name,
                group.world_size,
                chunks,
                envs.environment_variables["XDIT_TP_SDMA_TRIGGER"]().strip().lower(),
            )
    else:
        group = get_tp_group()
        with record_function(f"tp_rs:{site}"):
            out = torch.ops.xfuser.tp_chunked_linear_reduce_scatter(
                x, linear.weight, group.device_group.group_name, group.world_size, chunks
            )
    return out if bias is None else out + bias


# The pipelined collectives run on a side stream, which dynamo does not trace reliably,
# so each is one opaque custom op to torch.compile.
_comm_streams = {}


def _comm_stream() -> torch.cuda.Stream:
    device = torch.cuda.current_device()
    if device not in _comm_streams:
        _comm_streams[device] = torch.cuda.Stream(device=device)
    return _comm_streams[device]


@torch.library.custom_op("xfuser::tp_chunked_all_gather_linear", mutates_args=())
def _tp_chunked_all_gather_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], group_name: str, tp_size: int, chunks: int
) -> torch.Tensor:
    """all_gather(x, dim 1) @ weight.T + bias, chunk c's GEMM overlapping chunk c+1's all-gather.

    `x` holds block-cyclic rows (see the module docstring); the output is in natural order.
    """
    group = dist.distributed_c10d._resolve_process_group(group_name)
    batch, rows, features = x.shape
    block = rows // chunks
    span = tp_size * block
    compute, comm = torch.cuda.current_stream(), _comm_stream()
    gathered = x.new_empty(chunks, tp_size, batch, block, features)
    out = x.new_empty(batch, tp_size * rows, weight.shape[0])
    events = [torch.cuda.Event() for _ in range(chunks)]
    comm.wait_stream(compute)
    with torch.cuda.stream(comm):
        for c in range(chunks):
            dist.all_gather_into_tensor(gathered[c], x[:, c * block : (c + 1) * block].contiguous(), group=group)
            events[c].record()
    for c in range(chunks):
        compute.wait_event(events[c])
        # Rank-major blocks of chunk c are the natural rows c*span onward; a view when batch is 1.
        rows_c = gathered[c].transpose(0, 1).reshape(batch, span, features)
        if batch == 1:
            out_c = out[0, c * span : (c + 1) * span]
            if bias is None:
                torch.mm(rows_c[0], weight.t(), out=out_c)
            else:
                torch.addmm(bias, rows_c[0], weight.t(), out=out_c)
        else:
            out[:, c * span : (c + 1) * span] = F.linear(rows_c, weight, bias)
    # Every chunk's event was waited on, so the side stream is done with `x` and `gathered`.
    return out


@_tp_chunked_all_gather_linear.register_fake
def _(x, weight, bias, group_name, tp_size, chunks):
    return x.new_empty(x.shape[0], tp_size * x.shape[1], weight.shape[0])


@torch.library.custom_op("xfuser::tp_chunked_linear_reduce_scatter", mutates_args=())
def _tp_chunked_linear_reduce_scatter(
    x: torch.Tensor, weight: torch.Tensor, group_name: str, tp_size: int, chunks: int
) -> torch.Tensor:
    """reduce_scatter(x @ weight.T, dim 1), chunk c's reduce-scatter overlapping chunk c+1's GEMM.

    `x` holds rows in natural order; the output holds this rank's block-cyclic rows.
    """
    group = dist.distributed_c10d._resolve_process_group(group_name)
    batch, rows, features = x.shape
    block = rows // (tp_size * chunks)
    span = tp_size * block
    out_features = weight.shape[0]
    compute, comm = torch.cuda.current_stream(), _comm_stream()
    partial = x.new_empty(chunks, tp_size, batch, block, out_features)
    out = x.new_empty(chunks, batch, block, out_features)
    comm.wait_stream(compute)
    for c in range(chunks):
        rows_c = x[:, c * span : (c + 1) * span]
        if batch == 1:
            torch.mm(rows_c[0], weight.t(), out=partial[c].view(span, out_features))
        else:
            partial[c] = F.linear(rows_c, weight).view(batch, tp_size, block, out_features).transpose(0, 1)
        ready = torch.cuda.Event()
        ready.record()
        comm.wait_event(ready)
        with torch.cuda.stream(comm):
            dist.reduce_scatter_tensor(out[c], partial[c], group=group)
    compute.wait_stream(comm)
    return out.transpose(0, 1).reshape(batch, chunks * block, out_features)


@_tp_chunked_linear_reduce_scatter.register_fake
def _(x, weight, group_name, tp_size, chunks):
    return x.new_empty(x.shape[0], x.shape[1] // tp_size, weight.shape[0])
