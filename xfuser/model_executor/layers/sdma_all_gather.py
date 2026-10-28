"""All-gather + column-parallel GEMM with the gather on SDMA copy engines (ROCm).

torch's fused all-gather + GEMM reads its peers' symmetric-memory buffers with
tensor copies. On ROCm, HIP runs concurrent copies from those buffers one at a
time, so the gather moves at one link's rate. This op keeps torch's symmetric
memory and its barrier and replaces the transport: each peer's shard is copied
by its own SDMA engine, all peers at once, with ROCr's
`hsa_amd_memory_async_copy_on_engine`. The copies use no compute units.

The shard moves in `chunks` pieces queued in order on each engine, and a
dependent 4-byte copy on the same engine sets each piece's flag. The rows are
block-cyclic, as for xDiT's chunked overlap (layers/tensor_parallel.py): of
tp*chunks row blocks, rank r holds blocks r, tp+r, ..., so piece c of every rank
is one contiguous range of the gathered rows. The compute stream waits for
piece c's flags and runs that range's GEMM while later pieces are in flight.

Copies wait for a trigger the compute stream releases after the
symmetric-memory barrier (`trigger="gpu"`). `trigger="host"` instead waits on
the host for the barrier and then releases them, a fallback if ROCr cannot gate
SDMA copies on a GPU-written signal. XDIT_TP_SDMA_ENGINES picks the engine plan
(see sag_engine_plan in sdma_all_gather.hip).

Limits: ROCm only; batch 1. The transport is built with hipcc on first use
(sdma_all_gather.hip).
"""

import ctypes
import fcntl
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

_SOURCE = Path(__file__).with_name("sdma_all_gather.hip")
_SLOTS = 4
MAX_CHUNKS = 16
_ENGINE_PLANS = {"fixed": 0, "pairfree": 1}

_lib = None
_groups = {}
_slot = 0
_ones = {}


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed with {rc}")


def _load():
    """Build (once per source version, under a file lock) and load the transport."""
    global _lib
    if _lib is not None:
        return _lib
    source = _SOURCE.read_bytes()
    cache = Path(os.environ.get("XDIT_CACHE_DIR", Path.home() / ".cache" / "xfuser"))
    cache.mkdir(parents=True, exist_ok=True)
    so = cache / f"sdma_all_gather_{hashlib.sha1(source).hexdigest()[:12]}.so"
    with open(cache / "sdma_all_gather.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not so.exists():
            tmp = so.with_suffix(f".{os.getpid()}.tmp")
            subprocess.run(
                ["hipcc", "-O2", "-shared", "-fPIC", "-o", str(tmp), str(_SOURCE), "-lhsa-runtime64"], check=True
            )
            tmp.rename(so)
    lib = ctypes.CDLL(str(so))
    i, u32, u64, vp = ctypes.c_int, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_void_p
    ip, u64p = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint64)
    for name, argtypes in {
        "sag_init": [i, i, i],
        "sag_engine_plan": [i, ip, i, i, ip],
        "sag_arm": [i, i],
        "sag_release": [i, vp],
        "sag_open": [i],
        "sag_issue": [i, i, u64p, u64p, u64, ip, ip, i, u64, u64],
        "sag_stream_wait_value32": [vp, u64, u32],
    }.items():
        getattr(lib, name).argtypes = argtypes
        getattr(lib, name).restype = i
    _lib = lib
    return lib


class _Group:
    """Per TP group and engine plan: rank -> device map and one SDMA engine per peer."""

    def __init__(self, group_name: str, plan: str):
        lib = _load()
        group = dist.distributed_c10d._resolve_process_group(group_name)
        self.rank, self.world = group.rank(), group.size()
        device = torch.cuda.current_device()
        devices = [None] * self.world
        dist.all_gather_object(devices, device, group=group)
        _check(lib.sag_init(device, self.world * MAX_CHUNKS, _SLOTS), "sag_init")
        # Peers in step order from this rank, as torch's pipelined all-gather walks them.
        self.peers = [(self.rank + step) % self.world for step in range(1, self.world)]
        peer_devs = (ctypes.c_int * len(self.peers))(*[devices[p] for p in self.peers])
        engines = (ctypes.c_int * len(self.peers))()
        _check(lib.sag_engine_plan(device, peer_devs, len(self.peers), _ENGINE_PLANS[plan], engines), "sag_engine_plan")
        self.engines = list(engines)


def _group(group_name: str, plan: str) -> _Group:
    if (group_name, plan) not in _groups:
        _groups[group_name, plan] = _Group(group_name, plan)
    return _groups[group_name, plan]


def engine_plan(group_name: str, plan: str = "fixed") -> dict:
    """The SDMA engine each peer's shard is copied on, for this rank: {peer rank: engine}."""
    g = _group(group_name, plan)
    return dict(zip(g.peers, g.engines))


def _one(device: torch.device) -> torch.Tensor:
    if device not in _ones:
        _ones[device] = torch.ones(1, dtype=torch.int32, device=device)
    return _ones[device]


def sdma_all_gather_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    group_name: str,
    chunks: int = 4,
    trigger: str = "gpu",
    plan: str = "fixed",
) -> torch.Tensor:
    """all_gather(x, dim 1) @ weight.T + bias over the TP group, gather on SDMA engines in `chunks` pieces.

    `x` is (1, rows, features) holding this rank's block-cyclic rows (see the
    module docstring); the output is (1, world * rows, out_features) in natural order.
    """
    global _slot
    if torch.version.hip is None:
        raise RuntimeError("the SDMA all-gather needs ROCm")
    if trigger not in ("gpu", "host") or plan not in _ENGINE_PLANS:
        raise ValueError(f"trigger must be gpu or host and plan one of {list(_ENGINE_PLANS)}; got {trigger}, {plan}")
    batch, rows, features = x.shape
    if batch != 1:
        raise NotImplementedError("the SDMA all-gather handles batch 1")
    if not 1 <= chunks <= MAX_CHUNKS or rows % chunks:
        raise ValueError(f"chunks must divide the {rows} rows per rank and be at most {MAX_CHUNKS}; got {chunks}")
    lib = _load()
    g = _group(group_name, plan)
    world, rank = g.world, g.rank
    x2d = x[0].contiguous()
    block = rows // chunks
    elt = x2d.element_size()
    piece_bytes = block * features * elt
    ws = symm_mem.get_symm_mem_workspace(group_name, min_size=x2d.numel() * elt)
    ws.get_buffer(rank, x2d.shape, x2d.dtype).copy_(x2d)

    # gathered[c, p] = piece c of rank p = natural rows (c * world + p) * block onward.
    gathered = x2d.new_empty(chunks, world, block, features)
    flags = torch.zeros(chunks, world, dtype=torch.int32, device=x.device)
    gathered[:, rank] = x2d.view(chunks, block, features)
    out = x2d.new_empty(chunks, world * block, weight.shape[0])

    stream = torch.cuda.current_stream()
    handle = ctypes.c_void_p(stream.cuda_stream)
    slot = _slot
    _slot = (_slot + 1) % _SLOTS
    n_peers = len(g.peers)
    rc = lib.sag_arm(slot, chunks * n_peers)
    if rc <= -1000:
        kind, i = ("data", -rc - 1000) if rc > -2000 else ("flag", -rc - 2000)
        peer = g.peers[i % n_peers]
        raise RuntimeError(
            f"rank {rank}: the {kind} copy of piece {i // n_peers} from peer {peer} (SDMA engine "
            f"{g.engines[i % n_peers]}) of an earlier call on transport slot {slot} never finished"
        )
    _check(rc, "sag_arm")
    ws.barrier(channel=0)
    if trigger == "gpu":
        _check(lib.sag_release(slot, handle), "sag_release")
    else:
        stream.synchronize()
        _check(lib.sag_open(slot), "sag_open")

    src_base = [ws.get_buffer(p, x2d.shape, x2d.dtype).data_ptr() for p in g.peers]
    engines = (ctypes.c_int * n_peers)(*g.engines)
    one = _one(x.device).data_ptr()
    w_t = weight.t()
    for c in range(chunks):
        # Piece c of every peer, queued behind piece c - 1 on each peer's engine.
        dst = (ctypes.c_uint64 * n_peers)(*[gathered[c, p].data_ptr() for p in g.peers])
        src = (ctypes.c_uint64 * n_peers)(*[base + c * piece_bytes for base in src_base])
        flag_index = (ctypes.c_int * n_peers)(*[c * world + p for p in g.peers])
        _check(
            lib.sag_issue(
                slot, c * n_peers, dst, src, piece_bytes, engines, flag_index, n_peers, flags.data_ptr(), one
            ),
            "sag_issue",
        )
        for p in g.peers:
            _check(lib.sag_stream_wait_value32(handle, flags[c, p].data_ptr(), 1), "hipStreamWaitValue32")
        rows_c = gathered[c].view(world * block, features)
        if bias is None:
            torch.mm(rows_c, w_t, out=out[c])
        else:
            torch.addmm(bias, rows_c, w_t, out=out[c])
    # Every flag has been waited on, so every copy into `gathered` and from the peers'
    # buffers is done once the last GEMM is; the barrier keeps the peers from refilling theirs early.
    ws.barrier(channel=0)
    return out.view(1, chunks * world * block, weight.shape[0])


@torch.library.custom_op("xfuser::tp_sdma_all_gather_linear", mutates_args=())
def _tp_sdma_all_gather_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    group_name: str,
    chunks: int,
    trigger: str,
    plan: str,
) -> torch.Tensor:
    return sdma_all_gather_linear(x, weight, bias, group_name, chunks, trigger, plan)


@_tp_sdma_all_gather_linear.register_fake
def _(x, weight, bias, group_name, chunks, trigger, plan):
    tp_size = dist.distributed_c10d._resolve_process_group(group_name).size()
    return x.new_empty(x.shape[0], tp_size * x.shape[1], weight.shape[0])
