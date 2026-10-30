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
peer; once a piece lands, a 4-byte write on a spare engine sets the sender's own
flag word for that piece at the receiver. `direction="pull"`: each rank's
engines read the peers' pieces, and a 4-byte copy on a spare engine flags piece
c once every peer's has landed. On MI3xx, writing on the write-preferred engines
moves the gather faster than reading.

Waits are bounded. The compute stream waits for a piece's flags with a
one-thread kernel that gives up after XDIT_TP_SDMA_WAIT_S seconds (default 30)
and records which flags never landed; the next call on this process raises an
error naming them. The symmetric-memory barriers time out too, after long
enough for a peer's own waits to give up first.

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

import xfuser.envs as envs

_SOURCE = Path(__file__).with_name("sdma_all_gather.hip")
_SLOTS = 4
MAX_CHUNKS = 16

_lib = None
_groups = {}
_slot = 0
_call = 0
_ones = {}


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed with {rc}")


def _wait_seconds() -> float:
    return float(envs.environment_variables["XDIT_TP_SDMA_WAIT_S"]())


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
    i, u64, vp, dbl = ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_double
    ip, u64p = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint64)
    for name, argtypes in {
        "sag_init": [i, i, i],
        "sag_engine_plan": [i, ip, i, i, ip],
        "sag_arm": [i, i, i],
        "sag_release": [i, i, vp],
        "sag_open": [i, i],
        "sag_issue": [i, i, i, u64p, u64p, u64, ip, i, u64, u64],
        "sag_issue_push": [i, i, i, u64p, u64p, u64, ip, i, u64p, u64],
        "sag_wait_flags": [vp, u64, i, dbl, i, i],
        "sag_take_error": [ip, ip, u64p],
        "sag_debug_drop_flag": [i, i],
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


def debug_drop_flag(piece: int, peer_index: int) -> None:
    """Test hook: the next push of `piece` loses its flag write to peer g.peers[peer_index]."""
    _load().sag_debug_drop_flag(piece, peer_index)


def _one(device: torch.device) -> torch.Tensor:
    """A device int32 1, the value every flag write copies."""
    if device not in _ones:
        _ones[device] = torch.ones(1, dtype=torch.int32, device=device)
    return _ones[device]


def _raise_timeouts(lib, g: _Group) -> None:
    """Raise if a bounded wait of an earlier call ran out (recorded on the GPU by sag_wait_flags)."""
    piece, call, missing = ctypes.c_int(), ctypes.c_int(), ctypes.c_uint64()
    count = lib.sag_take_error(ctypes.byref(piece), ctypes.byref(call), ctypes.byref(missing))
    if count:
        words = [i for i in range(64) if missing.value >> i & 1]
        raise RuntimeError(
            f"rank {g.rank}: {count} SDMA flag wait(s) gave up after {_wait_seconds():g} s. First: call "
            f"{call.value}, piece {piece.value}, flag words {words} never set (for a push or reduce-scatter, "
            f"word r is rank r's; for a pull, word 0 is the whole piece). Peer engines (rank: engine): "
            f"{dict(zip(g.peers, g.engines))}"
        )


def raise_if_waits_gave_up(group_name: str, direction: str = "push") -> None:
    """Raise if a bounded flag wait of an earlier call on this process ran out; every call checks this first."""
    _raise_timeouts(_load(), _group(group_name, direction))


def _start(lib, g: _Group, n_data: int, n_flags: int):
    """Begin a call: check earlier calls' waits, take the next slot and arm it. Returns (slot, call id)."""
    global _slot, _call
    _raise_timeouts(lib, g)
    slot, _slot = _slot, (_slot + 1) % _SLOTS
    _call += 1
    n = len(g.peers)
    rc = lib.sag_arm(slot, n_data, n_flags)
    if rc <= -2000:
        raise RuntimeError(f"rank {g.rank}: flag copy {-rc - 2000} of an earlier call on slot {slot} never landed")
    if rc <= -1000:
        i = -rc - 1000
        raise RuntimeError(
            f"rank {g.rank}: piece {i // n} for peer {g.peers[i % n]} (SDMA engine {g.engines[i % n]}) of an "
            f"earlier call on slot {slot} never landed"
        )
    _check(rc, "sag_arm")
    return slot, _call


def _release(lib, slot: int, k: int, stream: torch.cuda.Stream, trigger: str) -> None:
    """Let slot's copies gated on trigger k start once `stream` reaches this point."""
    if trigger == "gpu":
        _check(lib.sag_release(slot, k, ctypes.c_void_p(stream.cuda_stream)), "sag_release")
    else:
        stream.synchronize()
        _check(lib.sag_open(slot, k), "sag_open")


def _wait(lib, stream: torch.cuda.Stream, flags: torch.Tensor, piece: int, call: int) -> None:
    """Make `stream` wait (bounded) until every int32 in `flags` reads 1."""
    _check(
        lib.sag_wait_flags(
            ctypes.c_void_p(stream.cuda_stream), flags.data_ptr(), flags.numel(), _wait_seconds(), piece, call
        ),
        "sag_wait_flags",
    )


def _inbox(group_name: str, rank: int, world: int, shape, dtype, chunks: int):
    """This rank's symmetric inbox of `shape` and its (chunks, world) int32 flag words, plus every
    rank's workspace base pointer and the flags' byte offset in it."""
    elt = torch.empty((), dtype=dtype).element_size()
    nbytes = elt
    for d in shape:
        nbytes *= d
    flag_offset = -(-nbytes // 64) * 64
    size = flag_offset + chunks * world * 4
    ws = symm_mem.get_symm_mem_workspace(group_name, min_size=size)
    inbox = ws.get_buffer(rank, tuple(shape), dtype)
    flags = ws.get_buffer(rank, (chunks, world), torch.int32, storage_offset=flag_offset // 4)
    bases = [ws.get_buffer(r, (size,), torch.uint8).data_ptr() for r in range(world)]
    return ws, inbox, flags, bases, flag_offset


def _barrier(ws) -> None:
    # A peer may reach the barrier only after all of its own bounded waits gave up, one per piece.
    ws.barrier(channel=0, timeout_ms=int(_wait_seconds() * (MAX_CHUNKS + 2) * 1000))


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
    x2d = x[0].contiguous()
    block = rows // chunks
    piece_bytes = block * features * x2d.element_size()
    stream = torch.cuda.current_stream()
    one = _one(x.device).data_ptr()

    if direction == "push":
        # gathered[c, p] = piece c of rank p = natural rows (c * world + p) * block onward.
        ws, gathered, flags, bases, flag_offset = _inbox(
            group_name, rank, world, (chunks, world, block, features), x2d.dtype, chunks
        )
        flags.zero_()
        flags[:, rank] = 1
        gathered[:, rank] = x2d.view(chunks, block, features)
        slot, call = _start(lib, g, chunks * n_peers, chunks * n_peers)
    else:
        ws = symm_mem.get_symm_mem_workspace(group_name, min_size=x2d.numel() * x2d.element_size())
        ws.get_buffer(rank, x2d.shape, x2d.dtype).copy_(x2d)
        gathered = x2d.new_empty(chunks, world, block, features)
        flags = torch.zeros(chunks, 1, dtype=torch.int32, device=x.device)
        gathered[:, rank] = x2d.view(chunks, block, features)
        slot, call = _start(lib, g, chunks * n_peers, chunks)
    out = x2d.new_empty(chunks, world * block, weight.shape[0])

    # Every rank's pieces and flags are in place, and its inbox is free, past this barrier.
    _barrier(ws)
    _release(lib, slot, 0, stream, trigger)

    if direction == "pull":
        src_base = [ws.get_buffer(p, x2d.shape, x2d.dtype).data_ptr() for p in g.peers]
    w_t = weight.t()
    for c in range(chunks):
        # Piece c for or from every peer, queued behind piece c - 1 on each peer's engine.
        if direction == "push":
            dst = (ctypes.c_uint64 * n_peers)(*[bases[p] + (c * world + rank) * piece_bytes for p in g.peers])
            src = (ctypes.c_uint64 * n_peers)(*[x2d.data_ptr() + c * piece_bytes] * n_peers)
            flag_dst = (ctypes.c_uint64 * n_peers)(*[bases[p] + flag_offset + (c * world + rank) * 4 for p in g.peers])
            _check(
                lib.sag_issue_push(slot, c, 0, dst, src, piece_bytes, g.engine_arr, n_peers, flag_dst, one),
                "sag_issue_push",
            )
        else:
            dst = (ctypes.c_uint64 * n_peers)(*[gathered[c, p].data_ptr() for p in g.peers])
            src = (ctypes.c_uint64 * n_peers)(*[base + c * piece_bytes for base in src_base])
            _check(
                lib.sag_issue(slot, c, 0, dst, src, piece_bytes, g.engine_arr, n_peers, flags[c].data_ptr(), one),
                "sag_issue",
            )
        _wait(lib, stream, flags[c], c, call)
        rows_c = gathered[c].view(world * block, features)
        if bias is None:
            torch.mm(rows_c, w_t, out=out[c])
        else:
            torch.addmm(bias, rows_c, w_t, out=out[c])
    # Every rank waited for every piece before its last GEMM, so past this barrier no copy reads
    # or writes this call's buffers (`x` included, which this rank's pushes read).
    _barrier(ws)
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
