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
one-thread kernel that records, after XDIT_TP_SDMA_WAIT_S seconds (default 30),
which flags never landed. With XDIT_TP_SDMA_ON_TIMEOUT=abort (default) the
kernel keeps the stream held, so nothing runs on data that never came, and a
watchdog thread logs what was missing and aborts the process, as NCCL's async
error handling does. With XDIT_TP_SDMA_ON_TIMEOUT=raise the stream moves on
(that call's output is wrong) and the next call on this process raises an
error naming the missing flags. The symmetric-memory barriers time out too,
after long enough for a peer's own waits to give up first; torch's barrier
kernel then traps (logging which rank never signalled) and the process aborts.
Being a GPU trap, it can also leave a large GPU core dump (gpucore.<pid>.gpu)
in the working directory.

Copies wait for a trigger the compute stream releases after the
symmetric-memory barrier (`trigger="gpu"`). `trigger="host"` instead waits on
the host for the barrier and then releases them, a fallback if ROCr cannot gate
SDMA copies on a GPU-written signal. Engines: see sag_engine_plan in
sdma_all_gather.hip.

`sdma_linear_reduce_scatter` is the row-parallel counterpart: GEMM chunk c
writes this rank's partial sums for every owner's block of chunk c; once it is
done, the engines push each owner's block into that owner's inbox, and each
rank sums the partials it received for chunk c (in fp32) one chunk behind, so
the next GEMM chunk is never held up by a wait.

Any batch size: every batch item's piece c moves with the others, and each
(piece, peer) flag waits for all of them.

The transport (sdma_all_gather.hip) is a small library loaded with ctypes,
from the first of: XDIT_SDMA_LIB; a library built into this package's directory
from the current source (`python xfuser/model_executor/layers/sdma_build.py
--out <this directory>`, for image builds: no GPU and no xfuser import needed);
a library built with hipcc for this GPU on first use and cached. The ops run on the GPU architectures in VALIDATED_ARCHS; others
need XDIT_TP_SDMA_ALLOW_UNTESTED=1. ROCm only.
"""

import ctypes
import fcntl
import hashlib
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import xfuser.envs as envs
from xfuser.logger import init_logger
from xfuser.model_executor.layers.sdma_build import LIB_NAME, SOURCE, build, source_hash

logger = init_logger(__name__)

# GPU architectures the ops are validated on; others need XDIT_TP_SDMA_ALLOW_UNTESTED=1.
VALIDATED_ARCHS = ("gfx950",)
_SLOTS = 4
MAX_CHUNKS = 16
_INITIAL_COPIES = 128  # signals per slot to start with; a slot grows when a call needs more

_lib = None
_lib_path = None
_groups = {}
_slot = 0
_call = 0
_ones = {}
_watchdog = None


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed with {rc}")


def _wait_seconds() -> float:
    return float(envs.environment_variables["XDIT_TP_SDMA_WAIT_S"]())


def _env(name: str) -> str:
    return envs.environment_variables[name]().strip()


def _on_timeout() -> str:
    mode = _env("XDIT_TP_SDMA_ON_TIMEOUT").lower()
    if mode not in ("abort", "raise"):
        raise ValueError(f"XDIT_TP_SDMA_ON_TIMEOUT must be abort or raise; got {mode!r}")
    return mode


def _arch() -> str:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]


def _library_path() -> Path:
    """XDIT_SDMA_LIB if set; else a library built for this GPU, cached by source, arch and HIP version."""
    prebuilt = _env("XDIT_SDMA_LIB")
    if prebuilt:
        if not Path(prebuilt).is_file():
            raise RuntimeError(f"XDIT_SDMA_LIB={prebuilt} is not a file")
        return Path(prebuilt)
    packaged = SOURCE.with_name(LIB_NAME)
    stamp = SOURCE.with_name(LIB_NAME + ".source-sha1")
    if packaged.is_file() and stamp.is_file() and stamp.read_text().strip() == source_hash():
        return packaged
    arch = _arch()
    key = hashlib.sha1(SOURCE.read_bytes() + f" {arch} {torch.version.hip}".encode()).hexdigest()[:12]
    cache = Path(os.environ.get("XDIT_CACHE_DIR", Path.home() / ".cache" / "xfuser")) / f"sdma_all_gather_{key}"
    so = cache / LIB_NAME
    if not so.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with open(cache / "build.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not so.exists():
                build(cache, [arch])
    return so


def _load():
    """Load the transport library (prebuilt or built on first use)."""
    global _lib, _lib_path
    if _lib is not None:
        return _lib
    _lib_path = _library_path()
    lib = ctypes.CDLL(str(_lib_path))
    i, u64, vp, dbl = ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_double
    ip, u64p = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint64)
    for name, argtypes in {
        "sag_init": [i, i, i],
        "sag_engine_plan": [i, ip, i, i, ip],
        "sag_arm": [i, i, i],
        "sag_release": [i, i, vp],
        "sag_open": [i, i],
        "sag_issue": [i, i, i, u64p, u64p, u64, ip, i, i, u64, u64],
        "sag_issue_push": [i, i, i, u64p, u64p, u64, ip, i, i, u64p, u64],
        "sag_wait_flags": [vp, u64, i, dbl, i, i, i],
        "sag_take_error": [ip, ip, u64p],
        "sag_peek_error": [ip, ip, u64p],
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
        arch = _arch()
        if arch not in VALIDATED_ARCHS and _env("XDIT_TP_SDMA_ALLOW_UNTESTED") != "1":
            raise RuntimeError(
                f"the SDMA TP ops are validated on {', '.join(VALIDATED_ARCHS)}, not {arch}; set "
                "XDIT_TP_SDMA_ALLOW_UNTESTED=1 to run them anyway"
            )
        devices = [None] * self.world
        dist.all_gather_object(devices, device, group=group)
        rc = lib.sag_init(device, _INITIAL_COPIES, _SLOTS)
        if rc == -11:
            raise RuntimeError(f"{_lib_path} has no GPU code for {arch}; rebuild it with --offload-arch {arch}")
        _check(rc, "sag_init")
        # Peers in step order from this rank, as torch's pipelined all-gather walks them.
        self.peers = [(self.rank + step) % self.world for step in range(1, self.world)]
        n = len(self.peers)
        peer_devs = (ctypes.c_int * n)(*[devices[p] for p in self.peers])
        self.engine_arr = (ctypes.c_int * (n + 1))()
        push = int(direction == "push")
        _check(lib.sag_engine_plan(device, peer_devs, n, push, self.engine_arr), "sag_engine_plan")
        self.engines = list(self.engine_arr[:n])
        self.flag_engine = self.engine_arr[n]
        _start_watchdog(lib)


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


def _timeout_message(count: int, piece: int, call: int, missing: int) -> str:
    words = [i for i in range(64) if missing >> i & 1]
    plans = {f"{name} ({direction})": dict(zip(g.peers, g.engines)) for (name, direction), g in _groups.items()}
    rank = dist.get_rank() if dist.is_initialized() else "?"
    return (
        f"rank {rank}: {count} SDMA flag wait(s) gave up after {_wait_seconds():g} s. First: call {call}, "
        f"piece {piece}, flag words {words} never set (for a push or reduce-scatter, word r is TP rank r's; for "
        f"a pull, word 0 is the whole piece). Peer engines per TP group (peer rank: engine): {plans}"
    )


def _start_watchdog(lib) -> None:
    """In abort mode, a thread that aborts the process once a flag wait has given up."""
    global _watchdog
    if _watchdog is not None or _on_timeout() != "abort":
        return

    def watch():
        piece, call, missing = ctypes.c_int(), ctypes.c_int(), ctypes.c_uint64()
        while True:
            count = lib.sag_peek_error(ctypes.byref(piece), ctypes.byref(call), ctypes.byref(missing))
            if count:
                message = _timeout_message(count, piece.value, call.value, missing.value) + "; aborting"
                logger.error(message)
                print(message, file=sys.stderr, flush=True)
                os.abort()
            time.sleep(0.1)

    _watchdog = threading.Thread(target=watch, name="xfuser-sdma-watchdog", daemon=True)
    _watchdog.start()


def _raise_timeouts(lib, g: _Group) -> None:
    """Raise if a bounded wait of an earlier call ran out (recorded on the GPU by sag_wait_flags)."""
    piece, call, missing = ctypes.c_int(), ctypes.c_int(), ctypes.c_uint64()
    count = lib.sag_take_error(ctypes.byref(piece), ctypes.byref(call), ctypes.byref(missing))
    if count:
        raise RuntimeError(_timeout_message(count, piece.value, call.value, missing.value))


def raise_if_waits_gave_up(group_name: str, direction: str = "push") -> None:
    """Raise if a bounded flag wait of an earlier call on this process ran out; every call checks this first."""
    _raise_timeouts(_load(), _group(group_name, direction))


def _start(lib, g: _Group, n_data: int, n_flags: int):
    """Begin a call: check earlier calls' waits, take the next slot and arm it. Returns (slot, call id)."""
    global _slot, _call
    if _on_timeout() == "raise":
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
            ctypes.c_void_p(stream.cuda_stream),
            flags.data_ptr(),
            flags.numel(),
            _wait_seconds(),
            piece,
            call,
            int(_on_timeout() == "abort"),
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

    `x` is (batch, rows, features) holding this rank's block-cyclic rows (see the
    module docstring); the output is (batch, world * rows, out_features) in natural order.
    """
    if torch.version.hip is None:
        raise RuntimeError("the SDMA all-gather needs ROCm")
    if trigger not in ("gpu", "host") or direction not in ("push", "pull"):
        raise ValueError(f"trigger must be gpu or host and direction push or pull; got {trigger}, {direction}")
    batch, rows, features = x.shape
    if not 1 <= chunks <= MAX_CHUNKS or rows % chunks:
        raise ValueError(f"chunks must divide the {rows} rows per rank and be at most {MAX_CHUNKS}; got {chunks}")
    lib = _load()
    g = _group(group_name, direction)
    world, rank, n_peers = g.world, g.rank, len(g.peers)
    x3 = x.contiguous()
    elt = x3.element_size()
    block = rows // chunks
    piece_bytes = block * features * elt
    out_features = weight.shape[0]
    stream = torch.cuda.current_stream()
    one = _one(x.device).data_ptr()

    def piece_of(base, item, c):
        # Piece c of batch item `item` in a (batch, rows, features) buffer starting at `base`.
        return base + (item * rows + c * block) * features * elt

    # gathered[c, i, p] = piece c of rank p for batch item i; [c] is (batch, world * block) natural rows.
    gathered_shape = (chunks, batch, world, block, features)
    if direction == "push":
        # This rank's inbox and flag words live in the symmetric workspace, where peers write.
        ws, gathered, flags, bases, flag_offset = _inbox(group_name, rank, world, gathered_shape, x3.dtype, chunks)
        flags.zero_()
        flags[:, rank] = 1
        gathered[:, :, rank] = x3.view(batch, chunks, block, features).transpose(0, 1)
        slot, call = _start(lib, g, chunks * n_peers * batch, chunks * n_peers)
    else:
        ws = symm_mem.get_symm_mem_workspace(group_name, min_size=x3.numel() * elt)
        ws.get_buffer(rank, x3.shape, x3.dtype).copy_(x3)
        gathered = x3.new_empty(gathered_shape)
        flags = torch.zeros(chunks, 1, dtype=torch.int32, device=x.device)
        gathered[:, :, rank] = x3.view(batch, chunks, block, features).transpose(0, 1)
        slot, call = _start(lib, g, chunks * n_peers * batch, chunks)
    out = x3.new_empty(chunks, batch, world * block, out_features)

    # Every rank's pieces and flags are in place, and its inbox is free, past this barrier.
    _barrier(ws)
    _release(lib, slot, 0, stream, trigger)

    copies = n_peers * batch
    if direction == "pull":
        src_base = [ws.get_buffer(p, x3.shape, x3.dtype).data_ptr() for p in g.peers]
    else:
        own = x3.data_ptr()
    w_t = weight.t()
    for c in range(chunks):
        # Piece c for or from every peer, queued behind piece c - 1 on each peer's engine.
        if direction == "push":
            dst = (ctypes.c_uint64 * copies)(
                *[bases[p] + ((c * batch + i) * world + rank) * piece_bytes for p in g.peers for i in range(batch)]
            )
            src = (ctypes.c_uint64 * copies)(*[piece_of(own, i, c) for _ in g.peers for i in range(batch)])
            flag_dst = (ctypes.c_uint64 * n_peers)(*[bases[p] + flag_offset + (c * world + rank) * 4 for p in g.peers])
            _check(
                lib.sag_issue_push(slot, c, 0, dst, src, piece_bytes, g.engine_arr, n_peers, batch, flag_dst, one),
                "sag_issue_push",
            )
        else:
            dst = (ctypes.c_uint64 * copies)(*[gathered[c, i, p].data_ptr() for p in g.peers for i in range(batch)])
            src = (ctypes.c_uint64 * copies)(*[piece_of(base, i, c) for base in src_base for i in range(batch)])
            _check(
                lib.sag_issue(
                    slot, c, 0, dst, src, piece_bytes, g.engine_arr, n_peers, batch, flags[c].data_ptr(), one
                ),
                "sag_issue",
            )
        _wait(lib, stream, flags[c], c, call)
        rows_c = gathered[c].view(batch * world * block, features)
        out_c = out[c].view(batch * world * block, out_features)
        if bias is None:
            torch.mm(rows_c, w_t, out=out_c)
        else:
            torch.addmm(bias, rows_c, w_t, out=out_c)
    # Every rank waited for every piece before its last GEMM, so past this barrier no copy reads
    # or writes this call's buffers (`x` included, which this rank's pushes read).
    _barrier(ws)
    return out.transpose(0, 1).reshape(batch, chunks * world * block, out_features)


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


def sdma_linear_reduce_scatter(
    x: torch.Tensor, weight: torch.Tensor, group_name: str, chunks: int = 4, trigger: str = "gpu"
) -> torch.Tensor:
    """reduce_scatter(x @ weight.T, dim 1) over the TP group, with the partial sums pushed on SDMA engines.

    `x` is (batch, rows, features) in natural order, the same rows on every rank
    with this rank's slice of the features; the output is (batch, rows / world,
    out_features) holding this rank's block-cyclic rows (see the module
    docstring), as xfuser::tp_chunked_linear_reduce_scatter returns them.
    """
    if torch.version.hip is None:
        raise RuntimeError("the SDMA reduce-scatter needs ROCm")
    if trigger not in ("gpu", "host"):
        raise ValueError(f"trigger must be gpu or host; got {trigger}")
    batch, rows, features = x.shape
    lib = _load()
    g = _group(group_name, "push")
    world, rank, n_peers = g.world, g.rank, len(g.peers)
    if not 1 <= chunks <= MAX_CHUNKS or rows % (world * chunks):
        raise ValueError(f"world * chunks must divide the {rows} rows and chunks be at most {MAX_CHUNKS}")
    block = rows // (world * chunks)
    span = world * block
    out_features = weight.shape[0]
    block_bytes = block * out_features * x.element_size()
    stream = torch.cuda.current_stream()
    one = _one(x.device).data_ptr()

    # inbox[c, i, s] = sender s's partial sums for this rank's block of chunk c, batch item i.
    shape = (chunks, batch, world, block, out_features)
    ws, inbox, flags, bases, flag_offset = _inbox(group_name, rank, world, shape, x.dtype, chunks)
    partials = x.new_empty(shape)
    out = x.new_empty(chunks, batch, block, out_features)
    flags.zero_()
    flags[:, rank] = 1
    slot, call = _start(lib, g, chunks * n_peers * batch, chunks * n_peers)
    # Every rank's inbox is free and its flags cleared past this barrier.
    _barrier(ws)

    w_t = weight.t()
    copies = n_peers * batch

    def reduce(c):
        _wait(lib, stream, flags[c], c, call)
        out[c].copy_(torch.sum(inbox[c], dim=1, dtype=torch.float32))

    for c in range(chunks):
        torch.matmul(x[:, c * span : (c + 1) * span], w_t, out=partials[c].view(batch, span, out_features))
        inbox[c, :, rank].copy_(partials[c, :, rank])
        # Chunk c's pushes start once its GEMM is done: trigger c.
        _release(lib, slot, c, stream, trigger)
        dst = (ctypes.c_uint64 * copies)(
            *[bases[p] + ((c * batch + i) * world + rank) * block_bytes for p in g.peers for i in range(batch)]
        )
        src = (ctypes.c_uint64 * copies)(*[partials[c, i, p].data_ptr() for p in g.peers for i in range(batch)])
        flag_dst = (ctypes.c_uint64 * n_peers)(*[bases[p] + flag_offset + (c * world + rank) * 4 for p in g.peers])
        _check(
            lib.sag_issue_push(slot, c, c, dst, src, block_bytes, g.engine_arr, n_peers, batch, flag_dst, one),
            "sag_issue_push",
        )
        if c > 0:
            reduce(c - 1)
    reduce(chunks - 1)
    # Every rank waited for every chunk from every peer, so past this barrier no copy reads
    # `partials` or writes an inbox.
    _barrier(ws)
    return out.transpose(0, 1).reshape(batch, chunks * block, out_features)


@torch.library.custom_op("xfuser::tp_sdma_linear_reduce_scatter", mutates_args=())
def _tp_sdma_linear_reduce_scatter(
    x: torch.Tensor, weight: torch.Tensor, group_name: str, tp_size: int, chunks: int, trigger: str
) -> torch.Tensor:
    return sdma_linear_reduce_scatter(x, weight, group_name, chunks, trigger)


@_tp_sdma_linear_reduce_scatter.register_fake
def _(x, weight, group_name, tp_size, chunks, trigger):
    return x.new_empty(x.shape[0], x.shape[1] // tp_size, weight.shape[0])


def _main(argv=None) -> None:
    """`python -m xfuser.model_executor.layers.sdma_all_gather build ...`: as sdma_build.py, for hosts where
    importing xfuser is fine."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "build":
        raise SystemExit(
            "usage: python -m xfuser.model_executor.layers.sdma_all_gather build --out DIR [--offload-arch ARCHS]"
        )
    from xfuser.model_executor.layers.sdma_build import main

    main(argv[1:], prog="python -m xfuser.model_executor.layers.sdma_all_gather build")


if __name__ == "__main__":
    _main()
