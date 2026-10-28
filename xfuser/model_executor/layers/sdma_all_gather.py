"""All-gather + column-parallel GEMM with the gather on SDMA copy engines (ROCm).

torch's fused all-gather + GEMM reads its peers' symmetric-memory buffers with
tensor copies. On ROCm, HIP runs concurrent copies from those buffers one at a
time, so the gather moves at one link's rate. This op keeps torch's symmetric
memory, its barrier and its ROCm `_async_input_mm` GEMM (whose row tiles wait
on per-rank flags), and replaces only the transport: each peer's shard is
copied by its own SDMA engine, all peers at once, with ROCr's
`hsa_amd_memory_async_copy_on_engine`, and a dependent 4-byte copy on the same
engine sets that peer's flag. The copies use no compute units; the GEMM starts
on the local shard and consumes the others as they land.

Copies wait for a trigger the compute stream releases after the
symmetric-memory barrier (`trigger="gpu"`). `trigger="host"` instead waits on
the host for the barrier and then issues the copies, a fallback if ROCr cannot
gate SDMA copies on a GPU-written signal. `consumer="per_peer"` replaces
`_async_input_mm` with one stream wait + GEMM per shard.

Limits: ROCm only; batch 1; with `consumer="async_mm"` each rank's rows are a
multiple of 256 (the GEMM's row tile). The transport is built with hipcc on
first use (sdma_all_gather.hip).
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

ROW_TILE = 256
_SOURCE = Path(__file__).with_name("sdma_all_gather.hip")
_SLOTS = 4

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
        "sag_engine_plan": [i, ip, i, ip],
        "sag_arm": [i, i],
        "sag_release": [i, vp],
        "sag_issue": [i, u64p, u64p, u64, ip, ip, i, u64, u64, i],
        "sag_stream_wait_value32": [vp, u64, u32],
    }.items():
        getattr(lib, name).argtypes = argtypes
        getattr(lib, name).restype = i
    _lib = lib
    return lib


class _Group:
    """Per TP group: rank -> device map and one SDMA engine per peer."""

    def __init__(self, group_name: str):
        lib = _load()
        group = dist.distributed_c10d._resolve_process_group(group_name)
        self.rank, self.world = group.rank(), group.size()
        device = torch.cuda.current_device()
        devices = [None] * self.world
        dist.all_gather_object(devices, device, group=group)
        _check(lib.sag_init(device, self.world, _SLOTS), "sag_init")
        # Peers in step order from this rank, as torch's pipelined all-gather walks them.
        self.peers = [(self.rank + step) % self.world for step in range(1, self.world)]
        peer_devs = (ctypes.c_int * len(self.peers))(*[devices[p] for p in self.peers])
        engines = (ctypes.c_int * len(self.peers))()
        _check(lib.sag_engine_plan(device, peer_devs, len(self.peers), engines), "sag_engine_plan")
        self.engines = list(engines)
        self.engine_arr = engines
        self.flag_index = (ctypes.c_int * len(self.peers))(*self.peers)


def engine_plan(group_name: str) -> dict:
    """The SDMA engine each peer's shard is copied on, for this rank: {peer rank: engine}."""
    g = _group(group_name)
    return dict(zip(g.peers, g.engines))


def _group(group_name: str) -> _Group:
    if group_name not in _groups:
        _groups[group_name] = _Group(group_name)
    return _groups[group_name]


def _one(device: torch.device) -> torch.Tensor:
    if device not in _ones:
        _ones[device] = torch.ones(1, dtype=torch.int32, device=device)
    return _ones[device]


def sdma_all_gather_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    group_name: str,
    consumer: str = "async_mm",
    trigger: str = "gpu",
) -> torch.Tensor:
    """all_gather(x, dim 1) @ weight.T + bias over the TP group, gather on SDMA engines.

    `x` is (1, rows, features) with this rank's contiguous rows; the output is
    (1, world * rows, out_features) in rank order.
    """
    global _slot
    if torch.version.hip is None:
        raise RuntimeError("the SDMA all-gather needs ROCm")
    if consumer not in ("async_mm", "per_peer") or trigger not in ("gpu", "host"):
        raise ValueError(f"consumer must be async_mm or per_peer and trigger gpu or host; got {consumer}, {trigger}")
    batch, rows, features = x.shape
    if batch != 1:
        raise NotImplementedError("the SDMA all-gather handles batch 1")
    if consumer == "async_mm" and rows % ROW_TILE != 0:
        raise ValueError(f"consumer async_mm needs rows per rank to be a multiple of {ROW_TILE}; got {rows}")
    lib = _load()
    g = _group(group_name)
    x2d = x[0].contiguous()
    nbytes = x2d.numel() * x2d.element_size()
    ws = symm_mem.get_symm_mem_workspace(group_name, min_size=nbytes)
    ws.get_buffer(g.rank, x2d.shape, x2d.dtype).copy_(x2d)

    gathered = x2d.new_empty(g.world * rows, features)
    shards = gathered.chunk(g.world)
    flags = torch.zeros(g.world, dtype=torch.uint32, device=x.device)
    shards[g.rank].copy_(x2d)
    torch._C._distributed_c10d._SymmetricMemory.stream_write_value32(flags, g.rank, 1)

    stream = torch.cuda.current_stream()
    slot = _slot
    _slot = (_slot + 1) % _SLOTS
    rc = lib.sag_arm(slot, len(g.peers))
    if rc <= -1000:
        kind, i = ("data", -rc - 1000) if rc > -2000 else ("flag", -rc - 2000)
        raise RuntimeError(
            f"rank {g.rank}: the {kind} copy from peer {g.peers[i]} (SDMA engine {g.engines[i]}) of an earlier "
            f"call on transport slot {slot} never finished"
        )
    _check(rc, "sag_arm")
    ws.barrier(channel=0)
    if trigger == "gpu":
        _check(lib.sag_release(slot, ctypes.c_void_p(stream.cuda_stream)), "sag_release")
    else:
        stream.synchronize()
    n = len(g.peers)
    dst = (ctypes.c_uint64 * n)(*[shards[p].data_ptr() for p in g.peers])
    src = (ctypes.c_uint64 * n)(*[ws.get_buffer(p, x2d.shape, x2d.dtype).data_ptr() for p in g.peers])
    _check(
        lib.sag_issue(
            slot,
            dst,
            src,
            nbytes,
            g.engine_arr,
            g.flag_index,
            n,
            flags.data_ptr(),
            _one(x.device).data_ptr(),
            int(trigger == "gpu"),
        ),
        "sag_issue",
    )

    if consumer == "async_mm":
        out = torch.ops.symm_mem._async_input_mm(gathered, weight.t(), flags, g.rank)
    else:
        out = gathered.new_empty(g.world * rows, weight.shape[0])
        out_shards = out.chunk(g.world)
        torch.mm(shards[g.rank], weight.t(), out=out_shards[g.rank])
        for p in g.peers:
            _check(
                lib.sag_stream_wait_value32(ctypes.c_void_p(stream.cuda_stream), flags.data_ptr() + 4 * p, 1),
                "hipStreamWaitValue32",
            )
            torch.mm(shards[p], weight.t(), out=out_shards[p])
    # Every flag has been waited on, so every copy into `gathered` and from the peers'
    # buffers is done once the GEMM is; the barrier keeps the peers from refilling theirs early.
    ws.barrier(channel=0)
    if bias is not None:
        out += bias
    return out.unsqueeze(0)


@torch.library.custom_op("xfuser::tp_sdma_all_gather_linear", mutates_args=())
def _tp_sdma_all_gather_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], group_name: str, consumer: str, trigger: str
) -> torch.Tensor:
    return sdma_all_gather_linear(x, weight, bias, group_name, consumer, trigger)


@_tp_sdma_all_gather_linear.register_fake
def _(x, weight, bias, group_name, consumer, trigger):
    tp_size = dist.distributed_c10d._resolve_process_group(group_name).size()
    return x.new_empty(x.shape[0], tp_size * x.shape[1], weight.shape[0])
