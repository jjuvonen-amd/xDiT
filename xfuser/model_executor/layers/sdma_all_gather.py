"""All-gather + column-parallel GEMM with the gather on SDMA copy engines (ROCm).

torch's fused all-gather + GEMM reads its peers' symmetric-memory buffers with
tensor copies. On ROCm, HIP runs concurrent copies from those buffers one at a
time, so the gather moves at one link's rate. This op keeps torch's symmetric
memory and its barrier and replaces the transport: each peer's shard is copied
by its own SDMA engine, all peers at once, with ROCr's
`hsa_amd_memory_async_copy_on_engine`. The copies use no compute units.

The shard moves in `chunks` pieces queued in order on one engine per peer.
The rows are block-cyclic, as for xDiT's chunked overlap
(layers/tensor_parallel.py): of tp*chunks row blocks, rank r holds blocks r,
tp+r, ..., so piece c of every rank is one contiguous range of the gathered
rows. The compute stream waits for piece c and runs that range's GEMM while
later pieces are in flight.

`direction="push"` (default): each rank's engines write its pieces into every
peer's symmetric-memory inbox, on the engines ROCr prefers for writing to that
peer, each piece followed by a 1-byte write of the sender's byte in the
receiver's flag word for that piece; the receiver waits for all bytes at once.
`direction="pull"`: each rank's engines read the peers' pieces, and a 4-byte
copy on a spare engine flags piece c once every peer's has landed. On MI3xx,
writing on the write-preferred engines moves the gather faster than reading.

Copies wait for a trigger the compute stream releases after the
symmetric-memory barrier (`trigger="gpu"`). `trigger="host"` instead waits on
the host for the barrier and then releases them, a fallback if ROCr cannot gate
SDMA copies on a GPU-written signal. Engines: see sag_engine_plan in
sdma_all_gather.hip.

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
_FLAG_WORD = 8  # bytes per piece in a push receiver's flags: one per rank

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
        "sag_arm": [i, i, i],
        "sag_release": [i, vp],
        "sag_open": [i],
        "sag_issue": [i, i, u64p, u64p, u64, ip, i, u64, u64],
        "sag_issue_push": [i, i, u64p, u64p, u64, ip, i, u64p, u64],
        "sag_stream_wait_value32": [vp, u64, u32],
        "sag_stream_wait_value64": [vp, u64, u64, u64],
    }.items():
        getattr(lib, name).argtypes = argtypes
        getattr(lib, name).restype = i
    _lib = lib
    return lib


class _Group:
    """Per TP group and direction: rank -> device map, one SDMA engine per peer and a spare one."""

    def __init__(self, group_name: str, direction: str):
        lib = _load()
        group = dist.distributed_c10d._resolve_process_group(group_name)
        self.rank, self.world = group.rank(), group.size()
        device = torch.cuda.current_device()
        devices = [None] * self.world
        dist.all_gather_object(devices, device, group=group)
        _check(lib.sag_init(device, self.world * MAX_CHUNKS, _SLOTS), "sag_init")
        # Peers in step order from this rank, as torch's pipelined all-gather walks them.
        self.peers = [(self.rank + step) % self.world for step in range(1, self.world)]
        n = len(self.peers)
        peer_devs = (ctypes.c_int * n)(*[devices[p] for p in self.peers])
        self.engine_arr = (ctypes.c_int * (n + 1))()
        push = int(direction == "push")
        _check(lib.sag_engine_plan(device, peer_devs, n, push, self.engine_arr), "sag_engine_plan")
        self.engines = list(self.engine_arr[:n])
        self.flag_engine = self.engine_arr[n]


def _group(group_name: str, direction: str) -> _Group:
    if (group_name, direction) not in _groups:
        _groups[group_name, direction] = _Group(group_name, direction)
    return _groups[group_name, direction]


def engine_plan(group_name: str, direction: str = "push") -> dict:
    """This rank's SDMA engines: {peer rank: engine, "flags": engine}."""
    g = _group(group_name, direction)
    return {**dict(zip(g.peers, g.engines)), "flags": g.flag_engine}


def _one(device: torch.device) -> torch.Tensor:
    """A device int32 1, whose first byte is the 1 a push flag write copies."""
    if device not in _ones:
        _ones[device] = torch.ones(1, dtype=torch.int32, device=device)
    return _ones[device]


def _arm(lib, g: _Group, slot: int, chunks: int, n_flags: int) -> None:
    n = len(g.peers)
    rc = lib.sag_arm(slot, chunks * n, n_flags)
    if rc <= -2000:
        raise RuntimeError(
            f"rank {g.rank}: flag copy {-rc - 2000} of an earlier call on transport slot {slot} never landed"
        )
    if rc <= -1000:
        i = -rc - 1000
        raise RuntimeError(
            f"rank {g.rank}: piece {i // n} for peer {g.peers[i % n]} (SDMA engine {g.engines[i % n]}) of an "
            f"earlier call on transport slot {slot} never landed"
        )
    _check(rc, "sag_arm")


def sdma_all_gather_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    group_name: str,
    chunks: int = 4,
    trigger: str = "gpu",
    direction: str = "push",
) -> torch.Tensor:
    """all_gather(x, dim 1) @ weight.T + bias over the TP group, gather on SDMA engines in `chunks` pieces.

    `x` is (1, rows, features) holding this rank's block-cyclic rows (see the
    module docstring); the output is (1, world * rows, out_features) in natural order.
    """
    global _slot
    if torch.version.hip is None:
        raise RuntimeError("the SDMA all-gather needs ROCm")
    if trigger not in ("gpu", "host") or direction not in ("push", "pull"):
        raise ValueError(f"trigger must be gpu or host and direction push or pull; got {trigger}, {direction}")
    batch, rows, features = x.shape
    if batch != 1:
        raise NotImplementedError("the SDMA all-gather handles batch 1")
    if not 1 <= chunks <= MAX_CHUNKS or rows % chunks:
        raise ValueError(f"chunks must divide the {rows} rows per rank and be at most {MAX_CHUNKS}; got {chunks}")
    lib = _load()
    g = _group(group_name, direction)
    world, rank, n_peers = g.world, g.rank, len(g.peers)
    if direction == "push" and world > _FLAG_WORD:
        raise NotImplementedError(f"the push direction flags up to {_FLAG_WORD} ranks per word; got {world}")
    x2d = x[0].contiguous()
    block = rows // chunks
    elt = x2d.element_size()
    piece_bytes = block * features * elt
    gathered_shape = (chunks, world, block, features)  # [c, p] = piece c of rank p
    gathered_bytes = chunks * world * piece_bytes
    flag_offset = -(-gathered_bytes // 64) * 64

    stream = torch.cuda.current_stream()
    handle = ctypes.c_void_p(stream.cuda_stream)
    slot = _slot
    _slot = (_slot + 1) % _SLOTS
    one = _one(x.device).data_ptr()

    if direction == "push":
        # This rank's inbox and flag words live in the symmetric workspace, where peers write.
        ws = symm_mem.get_symm_mem_workspace(group_name, min_size=flag_offset + chunks * _FLAG_WORD)
        gathered = ws.get_buffer(rank, gathered_shape, x2d.dtype)
        flag_words = ws.get_buffer(rank, (chunks, _FLAG_WORD), torch.uint8, storage_offset=flag_offset)
        flag_words.zero_()
        flag_words[:, rank] = 1
        gathered[:, rank] = x2d.view(chunks, block, features)
        _arm(lib, g, slot, chunks, chunks * n_peers)
    else:
        ws = symm_mem.get_symm_mem_workspace(group_name, min_size=x2d.numel() * elt)
        ws.get_buffer(rank, x2d.shape, x2d.dtype).copy_(x2d)
        gathered = x2d.new_empty(gathered_shape)
        flags = torch.zeros(chunks, dtype=torch.int32, device=x.device)
        gathered[:, rank] = x2d.view(chunks, block, features)
        _arm(lib, g, slot, chunks, chunks)
    out = x2d.new_empty(chunks, world * block, weight.shape[0])

    # Every rank's pieces and flags are in place, and its inbox is free, past this barrier.
    ws.barrier(channel=0)
    if trigger == "gpu":
        _check(lib.sag_release(slot, handle), "sag_release")
    else:
        stream.synchronize()
        _check(lib.sag_open(slot), "sag_open")

    if direction == "push":
        peer_base = [ws.get_buffer(p, (flag_offset + chunks * _FLAG_WORD,), torch.uint8).data_ptr() for p in g.peers]
        all_ranks = sum(1 << (8 * r) for r in range(world))
        mask = (1 << (8 * world)) - 1
    else:
        src_base = [ws.get_buffer(p, x2d.shape, x2d.dtype).data_ptr() for p in g.peers]
    w_t = weight.t()
    for c in range(chunks):
        # Piece c for or from every peer, queued behind piece c - 1 on each peer's engine.
        if direction == "push":
            own = x2d.data_ptr() + c * piece_bytes
            dst = (ctypes.c_uint64 * n_peers)(*[b + (c * world + rank) * piece_bytes for b in peer_base])
            src = (ctypes.c_uint64 * n_peers)(*[own] * n_peers)
            flag_dst = (ctypes.c_uint64 * n_peers)(*[b + flag_offset + c * _FLAG_WORD + rank for b in peer_base])
            _check(
                lib.sag_issue_push(slot, c, dst, src, piece_bytes, g.engine_arr, n_peers, flag_dst, one),
                "sag_issue_push",
            )
            _check(
                lib.sag_stream_wait_value64(handle, flag_words[c].data_ptr(), all_ranks, mask),
                "hipStreamWaitValue64",
            )
        else:
            dst = (ctypes.c_uint64 * n_peers)(*[gathered[c, p].data_ptr() for p in g.peers])
            src = (ctypes.c_uint64 * n_peers)(*[base + c * piece_bytes for base in src_base])
            _check(
                lib.sag_issue(slot, c, dst, src, piece_bytes, g.engine_arr, n_peers, flags[c].data_ptr(), one),
                "sag_issue",
            )
            _check(lib.sag_stream_wait_value32(handle, flags[c].data_ptr(), 1), "hipStreamWaitValue32")
        rows_c = gathered[c].view(world * block, features)
        if bias is None:
            torch.mm(rows_c, w_t, out=out[c])
        else:
            torch.addmm(bias, rows_c, w_t, out=out[c])
    # Every rank waited for every piece before its last GEMM, so past this barrier no copy reads
    # or writes this call's buffers (`x` included, which this rank's pushes read).
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
    direction: str,
) -> torch.Tensor:
    return sdma_all_gather_linear(x, weight, bias, group_name, chunks, trigger, direction)


@_tp_sdma_all_gather_linear.register_fake
def _(x, weight, bias, group_name, chunks, trigger, direction):
    tp_size = dist.distributed_c10d._resolve_process_group(group_name).size()
    return x.new_empty(x.shape[0], tp_size * x.shape[1], weight.shape[0])
