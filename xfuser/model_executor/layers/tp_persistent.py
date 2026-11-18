"""Persistent FP8 TP ops on the SDMA transport of torch's patched ROCm symm_mem ops (XDIT_TP_PERSISTENT=1).

Same transfers as torch's fused FP8 ops on that path (`_rocm_sdma.fused_all_gather_scaled_matmul` /
`fused_scaled_matmul_reduce_scatter`: inbox, pieces pushed on SDMA engines, flag words, barriers), with
one persistent FlyDSL GEMM (layers/flydsl_tp_gemm.py) instead of a GEMM per piece:

- all-gather + GEMM: the kernel walks its output tiles in readiness order and waits in-kernel on the
  piece flags, so this rank's rows compute while the peers' pieces land;
- GEMM + reduce-scatter: the kernel writes bf16 partial sums owner by owner in send order and counts
  finished tiles per (chunk, peer); a poster stream releases each chunk's SDMA pushes once its tiles
  are in; the received partials are summed in fp32 as torch's op does.

Each returns None when the call does not fit (rows per piece not whole 256-row tiles, K not a multiple
of 128 or below 256, ...); the caller then runs torch's op. XDIT_TP_ROW_ALIGN=256 pads rows so that they fit.
"""

import functools
from typing import Optional

import torch

from xfuser.logger import init_logger

logger = init_logger(__name__)

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 128
_warned = set()
_rs_state = {}
# calls that ran the persistent kernels (tests check these)
calls = {"all_gather_gemm": 0, "gemm_reduce_scatter": 0, "all_gather_gemm_bf16": 0}


def _fallback(kind: str, why: str) -> None:
    if kind not in _warned:
        _warned.add(kind)
        logger.warning(f"persistent FP8 {kind}: running torch's op instead ({why}); XDIT_TP_ROW_ALIGN=256 pads rows")


def _tile_chunks(want: int, rows: int) -> Optional[int]:
    """The largest piece count <= want whose pieces are whole BLOCK_M-row tiles."""
    for c in range(max(want, 1), 0, -1):
        if rows % (c * BLOCK_M) == 0:
            return c
    return None


@functools.lru_cache(maxsize=None)
def _wait_counts_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def wait_counts(CNT, W, me, target):
        """Spin (system-scope acquire) until CNT[p] >= target for every p != me (own rows are not counted)."""
        for p in range(W):
            if p != me:
                v = tl.atomic_add(CNT + p, 0, sem="acquire", scope="sys")
                while v < target:
                    v = tl.atomic_add(CNT + p, 0, sem="acquire", scope="sys")

    return wait_counts


def all_gather_gemm_fp8(rs, xq, xs, w_t, w_s, group_name: str) -> Optional[torch.Tensor]:
    """all_gather(xq rows) @ w_t, row-wise scaled: xq [rows, K] fp8, xs [rows, 1] fp32, w_t [K, N] fp8 column-major,
    w_s [1, N] fp32 -> [world * rows, N] bf16 (rank r's rows at r * rows), or None if the call does not fit."""
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    from xfuser.model_executor.layers.flydsl_tp_gemm import compile_persistent_ag_gemm_fp8

    g = rs._group(group_name)
    world, rank = g.world, g.rank
    rows, K = xq.shape
    N = w_t.shape[1]
    chunks = _tile_chunks(rs._chunks(rows, K * xq.element_size()), rows)
    # the kernel's K pipeline needs at least two K blocks (K = 128 computes wrong results)
    if K % BLOCK_K or K < 2 * BLOCK_K or chunks is None or not w_t.t().is_contiguous():
        _fallback("all-gather + GEMM", f"rows {rows}, K {K}")
        return None
    block = rows // chunks
    lib = rs._load()
    stream = torch.cuda.current_stream()
    tensors = [xq.contiguous(), xs.reshape(rows, 1).float().contiguous()]
    regions = [((world, rows, t.shape[1]), t.dtype) for t in tensors]
    ws, inboxes, offsets, flags, bases, flag_offset = rs._inbox(group_name, rank, world, regions, chunks)
    row_bytes = [t.shape[1] * t.element_size() for t in tensors]
    plan = rs._Plan(lib, g, chunks, len(tensors))
    for c in range(chunks):
        at = [offset + (rank * rows + c * block) * size for offset, size in zip(offsets, row_bytes)]
        plan.push(
            0,
            [bases[p] + a for p in g.peers for a in at],
            [bases[rank] + a for _ in g.peers for a in at],
            [block * size for size in row_bytes],
            [bases[p] + flag_offset + (c * world + rank) * 4 for p in g.peers],
        )
    flags.zero_()
    flags[:, rank] = 1
    for inbox, t in zip(inboxes, tensors):
        inbox[rank].copy_(t)
    out = torch.empty(world * rows, N, dtype=torch.bfloat16, device=xq.device)
    exe = compile_persistent_ag_gemm_fp8(K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    progs = torch.cuda.get_device_properties(xq.device).multi_processor_count
    a_view = inboxes[0].view(-1)[: rows * K].view(torch.int8)  # rank-0-sized views: the kernel rebases per rank
    s_view = inboxes[1].view(-1)[:rows]
    rs._barrier(ws)
    in_last_barrier = False
    try:
        plan.start(stream)
        plan.release(0, stream)
        _run_compiled(
            exe,
            a_view,
            w_t.t().view(torch.int8).reshape(-1),
            out.view(-1)[: rows * N],
            s_view,
            w_s.reshape(-1).float().contiguous(),
            bases[rank],
            1,
            rank,
            world,
            rows,
            block,
            rows,
            rows,
            flag_offset // 4,
            1,
            N,
            progs,
            fx.Stream(stream),
        )
        in_last_barrier = True
        rs._barrier(ws)
    except BaseException as err:
        if not plan.capturing:
            rs._abandon(err, ws, in_last_barrier)
        raise
    calls["all_gather_gemm"] += 1
    return out


def gemm_reduce_scatter_fp8(rs, xq, xs, w_t, w_s, group_name: str) -> Optional[torch.Tensor]:
    """reduce_scatter(xq @ w_t), row-wise scaled: xq [S, K] fp8 (all rows, this rank's K shard), xs [S, 1] fp32,
    w_t [K, N] fp8 column-major, w_s [1, N] -> this rank's [S / world, N] bf16, or None if the call does not fit."""
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    from xfuser.model_executor.layers.flydsl_tp_gemm import compile_persistent_ag_gemm_fp8

    g = rs._group(group_name)
    world, rank = g.world, g.rank
    S, K = xq.shape
    N = w_t.shape[1]
    rows = S // world
    chunks = _tile_chunks(rs._chunks(rows, N * 2), rows) if S % world == 0 else None
    if K % BLOCK_K or K < 2 * BLOCK_K or chunks is None or chunks > 64 or not w_t.t().is_contiguous():
        _fallback("GEMM + reduce-scatter", f"rows {S}, K {K}")
        return None
    block = rows // chunks
    lib = rs._load()
    cur = torch.cuda.current_stream()
    block_bytes = block * N * 2
    ws, (inbox,), _, flags, bases, flag_offset = rs._inbox(
        group_name, rank, world, [((chunks, world, block, N), torch.bfloat16)], chunks
    )
    partials = torch.empty(S, N, dtype=torch.bfloat16, device=xq.device)
    pv = partials.view(world, chunks, block, N)
    key = (xq.device.index, group_name)
    if key not in _rs_state:
        _rs_state[key] = (torch.zeros(64 * world, dtype=torch.int32, device=xq.device), torch.cuda.Stream(xq.device))
    cnt, post = _rs_state[key]
    plan = rs._Plan(lib, g, chunks, 1)
    for c in range(chunks):
        plan.push(
            c,
            [bases[p] + (c * world + rank) * block_bytes for p in g.peers],
            [pv[p, c].data_ptr() for p in g.peers],
            [block_bytes],
            [bases[p] + flag_offset + (c * world + rank) * 4 for p in g.peers],
        )
    flags.zero_()
    flags[:, rank] = 1
    cnt.zero_()
    out = torch.empty(rows, N, dtype=torch.bfloat16, device=xq.device)
    exe = compile_persistent_ag_gemm_fp8(K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, mode="rs")
    # leave CUs for the poster's spin kernels
    progs = torch.cuda.get_device_properties(xq.device).multi_processor_count * 7 // 8
    tiles = (block // BLOCK_M) * ((N + BLOCK_N - 1) // BLOCK_N)  # tiles per (chunk, peer)
    a_scale = xs.reshape(-1).float().contiguous()
    wait_counts = _wait_counts_kernel()
    rs._barrier(ws)
    in_last_barrier = False
    try:
        plan.start(cur)
        post.wait_stream(cur)  # counters zeroed before the poster reads them
        # A: rank-0-sized view, the kernel rebases per owner with 64-bit offsets (same for C and the scales)
        _run_compiled(
            exe,
            xq.contiguous().view(torch.int8).view(-1)[: rows * K],
            w_t.t().view(torch.int8).reshape(-1),
            partials.view(-1)[: rows * N],
            a_scale[:rows],
            w_s.reshape(-1).float().contiguous(),
            cnt.data_ptr(),
            1,
            rank,
            world,
            rows,
            block,
            rows,
            rows,
            0,
            1,
            N,
            progs,
            fx.Stream(cur),
        )
        for c in range(chunks):
            with torch.cuda.stream(post):
                wait_counts[(1,)](cnt[c * world :], world, rank, tiles)
            plan.release(c, post)
        for c in range(chunks):
            plan.wait(cur, flags[c], c)
            inbox[c, rank].copy_(pv[rank, c])  # after the GEMM on cur (own rows are its last tiles)
            out[c * block : (c + 1) * block] = torch.sum(inbox[c], dim=0, dtype=torch.float32)
        cur.wait_stream(post)
        in_last_barrier = True
        rs._barrier(ws)
    except BaseException as err:
        if not plan.capturing:
            rs._abandon(err, ws, in_last_barrier)
        raise
    partials.record_stream(post)
    calls["gemm_reduce_scatter"] += 1
    return out


def all_gather_gemm_bf16(rs, x, w_t, group_name: str) -> Optional[torch.Tensor]:
    """all_gather(x rows) @ w_t in bf16: x [rows, K], w_t [K, N] (e.g. weight.t()) -> [world * rows, N], with one
    flag-gated FlyDSL GEMM over the gathered inbox (layers/flydsl_tp_gemm_bf16.py); None if the call does not fit."""
    from xfuser.model_executor.layers.flydsl_tp_gemm_bf16 import ag_gemm_bf16

    g = rs._group(group_name)
    world, rank = g.world, g.rank
    rows, K = x.shape
    chunks = rs._chunks(rows, K * x.element_size())
    if x.dtype != torch.bfloat16 or w_t.dtype != x.dtype or rows % chunks or K % 64:
        _fallback("bf16 all-gather + GEMM", f"rows {rows}, K {K}, dtype {x.dtype}")
        return None
    block = rows // chunks
    lib = rs._load()
    stream = torch.cuda.current_stream()
    ws, (inbox,), offsets, flags, bases, flag_offset = rs._inbox(
        group_name, rank, world, [((world, rows, K), x.dtype)], chunks
    )
    row_bytes = K * x.element_size()
    plan = rs._Plan(lib, g, chunks, 1)
    for c in range(chunks):
        at = offsets[0] + (rank * rows + c * block) * row_bytes
        plan.push(
            0,
            [bases[p] + at for p in g.peers],
            [bases[rank] + at for _ in g.peers],
            [block * row_bytes],
            [bases[p] + flag_offset + (c * world + rank) * 4 for p in g.peers],
        )
    flags.zero_()
    flags[:, rank] = 1
    inbox[rank].copy_(x)
    out = x.new_empty(world * rows, w_t.shape[1])
    rs._barrier(ws)
    in_last_barrier = False
    try:
        plan.start(stream)
        plan.release(0, stream)
        ag_gemm_bf16(inbox.view(world * rows, K), w_t, out, flags.data_ptr(), world, rank, rows, block, stream=stream)
        in_last_barrier = True
        rs._barrier(ws)
    except BaseException as err:
        if not plan.capturing:
            rs._abandon(err, ws, in_last_barrier)
        raise
    calls["all_gather_gemm_bf16"] = calls.get("all_gather_gemm_bf16", 0) + 1
    return out
