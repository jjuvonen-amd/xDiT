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
"""

from typing import List, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.profiler import record_function

from xfuser.core.distributed.parallel_state import get_tp_group


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


def tp_column_parallel(x: torch.Tensor, linear: nn.Linear, site: str, chunks: int) -> torch.Tensor:
    """`linear` over the TP group's rows of `x` (dim 1): all-gather, then the column-parallel GEMM."""
    if chunks == 1:
        return linear(tp_all_gather_seq(x, site))
    group = get_tp_group()
    with record_function(f"tp_ag:{site}"):
        return torch.ops.xfuser.tp_chunked_all_gather_linear(
            x, linear.weight, linear.bias, group.device_group.group_name, group.world_size, chunks
        )


def tp_row_parallel(
    x: torch.Tensor, linear: nn.Linear, bias: Optional[torch.Tensor], site: str, chunks: int
) -> torch.Tensor:
    """Row-parallel `linear` over all rows of `x`, reduce-scattered to this rank's rows, plus `bias` once."""
    if chunks == 1:
        out = tp_reduce_scatter_seq(linear(x), site)
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
