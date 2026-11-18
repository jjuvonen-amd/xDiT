"""Flag-gated bf16 all-gather + GEMM in one launch (gfx950), for the TP sites on the SDMA transport of torch's
patched ROCm symm_mem ops (XDIT_TP_PERSISTENT=1, layers/tp_persistent.py).

AITER's FlyDSL bf16 GEMM (aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py, both its plain and its
half-tile-interleaved kernel), patched at load time:
  * the output tile of a workgroup comes from a host-built order table instead of the XCD/group swizzle:
    row blocks that need only this rank's rows first, then the others by when their pieces arrive
    (chunk-major, peers me+1, me+2, ...);
  * before its K loop, thread 0 of a workgroup spins (system scope, acquire) until every piece flag its rows
    overlap reads >= 1, then the workgroup barriers. Flags are the transport's (chunks, world) int32 words:
    piece c of rank r covers gathered rows r * rows + c * block .. + block; this rank's own are set first.
A is the gathered inbox [world * rows, K], so tiles may straddle ranks or chunks (no row alignment needed).
The patched module is written to XDIT_CACHE_DIR (default ~/.cache/xfuser), as FlyDSL reads kernel sources.
"""

import functools
import importlib.util
import os
import sys
from pathlib import Path

import torch

_SRC = "aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py"
_NAME = "aiter.ops.flydsl.kernels.gemm_a16w16_gfx950"
_mods = {}

_SIG_OLD = "    b_leading_stride: fx.Int32,\n    tiled_mma: fx.TiledMma,\n"
_SIG_NEW = (
    "    b_leading_stride: fx.Int32,\n    ag_order: fx.Tensor,\n    flags_addr: fx.Int64,\n"
    "    ag_block: fx.Int32,\n    ag_chunks: fx.Int32,\n    tiled_mma: fx.TiledMma,\n"
)
_MAP_OLD = "    bid_m, bid_n = block_swizzle.swizzle(num_pid_m, num_pid_n, fx.block_idx.x)\n"
_MAP_COMMON = """    _t = fx.Int32(fx.block_idx.x)
    _order_buf = fx.rocdl.make_buffer_tensor(ag_order, max_size=True)
    bid_m = fx.Int32(_order_buf[_t // num_pid_n])
    bid_n = _t % num_pid_n
    _row0 = bid_m * block_m
    _row1 = (_row0 + block_m < m).select(_row0 + block_m, m) - 1
"""
# ag: wait for every piece the tile's rows overlap (own pieces are pre-set); flags (chunks, world): c * W + r
_MAP_AG = (
    _MAP_COMMON
    + """    if fx.thread_idx.x == 0:
        for _g in range(_row0 // ag_block, _row1 // ag_block + 1, fx.Int32(1)):
            _r = _g // ag_chunks
            _c = _g % ag_chunks
            _comm_ops.spin_until_ge_i32_system(
                flags_addr + fx.Int64(_c * (m // (ag_block * ag_chunks)) + _r) * fx.Int64(4), 1, acquire=True
            )
        _comm_ops.fence_system_acquire()
    gpu.barrier()
"""
)
_LSIG_OLD = "    split_k: fx.Int32,\n    param: GemmA16W16Gfx950Param,\n"
_LSIG_NEW = (
    "    split_k: fx.Int32,\n    ag_order: fx.Tensor,\n    flags_addr: fx.Int64,\n    ag_block: fx.Int32,\n"
    "    ag_chunks: fx.Int32,\n    param: GemmA16W16Gfx950Param,\n"
)
_CALL_OLD = "        b_leading_stride,\n        tiled_mma,\n"
_CALL_NEW = "        b_leading_stride,\n        ag_order,\n        flags_addr,\n        ag_block,\n        ag_chunks,\n        tiled_mma,\n"
_KNAME_OLD = "make_gemm_a16w16_gfx950_kernel_name(param)\n"


def _module(mode="ag"):
    """The patched copy of AITER's bf16 GEMM module (flag-gated tiles)."""
    if mode in _mods:
        return _mods[mode]
    import aiter

    src = (Path(aiter.__file__).parent.parent / _SRC).read_text()
    kname_new = f'make_gemm_a16w16_gfx950_kernel_name(param) + "_{mode}flags"\n'
    for old, new, count in (
        (_SIG_OLD, _SIG_NEW, 2),
        (_MAP_OLD, _MAP_AG, 2),
        (_LSIG_OLD, _LSIG_NEW, 1),
        (_CALL_OLD, _CALL_NEW, 1),
        (_KNAME_OLD, kname_new, 1),
    ):
        assert src.count(old) == count, f"AITER's bf16 GEMM changed: {old.strip()[:60]!r} x{src.count(old)}"
        src = src.replace(old, new)
    src = src.replace(
        "from .kernels_common import run_cached\n",
        "from .kernels_common import run_cached\nfrom . import communication_ops_utils as _comm_ops\n",
        1,
    )
    # FlyDSL reads kernel sources from files: the patched module goes to a cache directory
    out_dir = Path(os.environ.get("XDIT_CACHE_DIR", Path.home() / ".cache" / "xfuser")) / "flydsl_tp_gemm_bf16"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"gemm_a16w16_gfx950_{mode}.py"
    if not path.exists() or path.read_text() != src:
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(src)
        tmp.replace(path)
    name = f"{_NAME}_{mode}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "aiter.ops.flydsl.kernels"
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _mods[mode] = mod
    return mod


@functools.lru_cache(maxsize=64)
def _order(world, me, rows, block, block_m, device_index):
    """Row-block order: blocks whose pieces are all mine first, then by the latest (chunk, peer distance) they need."""
    chunks = rows // block
    M = world * rows
    keys = []
    for b in range((M + block_m - 1) // block_m):
        r0, r1 = b * block_m, min(b * block_m + block_m, M) - 1
        need = []
        for g in range(r0 // block, r1 // block + 1):
            r, c = divmod(g, chunks)
            if r != me:
                need.append((c, (r - me) % world))
        keys.append((1, max(need), b) if need else (0, (0, 0), b))
    order = [k[2] for k in sorted(keys)]
    return torch.tensor(order, dtype=torch.int32, device=torch.device("cuda", device_index))


def _launch(mode, A, B, out, flags_addr, order, block, chunks, stream, user_kwargs):
    mod = _module(mode)
    M, K = A.shape
    N = B.shape[1]
    b_is_transposed = B.stride(0) == 1 and B.stride(1) != 1
    kwargs = {
        "block_m": 256,
        "block_n": 256,
        "block_k": 64,
        "stages": 2,
        "split_k": 1,
        "m_waves": 2,
        "n_waves": 4,
        "k_waves": 1,
        "group_m": 0,
        "use_half_tile_interleaved": True,
    }
    kwargs.update(user_kwargs or {})
    kwargs.update(
        a_is_transposed=False,
        b_is_transposed=b_is_transposed,
        in_dtype_id=mod.GEMM_A16W16_DTYPE_BF16,
        out_dtype_id=mod.GEMM_A16W16_DTYPE_BF16,
        has_bias=False,
    )
    mod.assert_no_k_tail(K, kwargs)
    param = mod.make_gemm_a16w16_param_and_validate(M, N, K, kwargs)
    assert param is not None, "unsupported shape/config"
    semaphore, signal = mod.get_split_k_buffers(stream, A.device)
    a_arg = mod._dynamic_tensor_arg(A, 1)
    b_arg = mod._dynamic_tensor_arg(B, 0 if b_is_transposed else 1)
    out_arg = mod._dynamic_tensor_arg(out, 1)
    dispatch_args = (
        out_arg,
        a_arg,
        b_arg,
        a_arg,
        semaphore,
        signal,
        1,
        order,
        flags_addr,
        block,
        chunks,
        param,
        stream,
    )
    mod.run_cached(
        mod.gemm_a16w16_gfx950,
        *dispatch_args,
        constexpr_param=param,
        compiler=mod.flyc.compile,
        dispatch_args=dispatch_args,
    )
    return out


def ag_gemm_bf16(A, B, out, flags_addr, world, me, rows, block, stream=None, user_kwargs=None):
    """out[world * rows, N] = A[world * rows, K] @ B[K, N]; tile rows wait on Joni's piece flags.
    B row-major ("nn") or column-major (e.g. weight.t(), "nt")."""
    M, K = A.shape
    assert M == world * rows and rows % block == 0 and B.shape[0] == K and out.shape == (M, B.shape[1])
    order = _order(world, me, rows, block, (user_kwargs or {}).get("block_m", 256), A.device.index)
    return _launch(
        "ag", A, B, out, flags_addr, order, block, rows // block, stream or torch.cuda.current_stream(), user_kwargs
    )
