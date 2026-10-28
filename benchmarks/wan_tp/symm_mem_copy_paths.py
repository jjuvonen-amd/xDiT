"""Which copy path lets torch symmetric-memory peer reads scale on ROCm?

    torchrun --nproc_per_node 8 benchmarks/wan_tp/symm_mem_copy_paths.py [--mib 92] [--output-dir /output/p2p]

torch's fused TP ops read peers' symmetric-memory buffers with tensor copies,
and on MI3xx those reads stay at one link's rate even when issued
concurrently, while ROCr copies between hipMalloc buffers with both GPUs named
scale with the number of peers. This reads the same symmetric-memory peer buffers, all ranks
at once (an all-gather's traffic), through each path:

  torch          torch copy_: what the fused ops do
  hip_default    hipMemcpyAsync, kind Default
  hip_d2d        hipMemcpyAsync, kind DeviceToDevice
  hsa_local      ROCr hsa_amd_memory_async_copy with the local GPU as source agent:
                 the peer buffer is mapped for it. ROCr treats this as a same-device
                 copy; on MI355X concurrent ones run one at a time (rocprofv3)
  hsa_local_sdma the same copy forced onto SDMA (hsa_amd_memory_async_copy_on_engine),
                 on the engine ROCr prefers for that peer link
  hsa_local_sdma_fixed     the same, with peers that share a preferred engine moved
                 to a free engine: with the local GPU on both sides all copies use
                 the local GPU's engines, so a shared engine runs its copies in turn
  hsa_local_sdma_pairfree  the same, also keeping peers off both engines of a pair
                 2k/2k+1, in case such a pair shares bandwidth
  hsa_local_sdma_distinct  one distinct available engine per peer (lowest first)
  hsa_local_sdma_distinct_rev  the same engines, issued in reverse order: tells a
                 slow engine from a slow issue position
  hsa_local_sdma_split     each peer's copy in halves on two distinct engines
  hip_peer      hipMemcpyPeerAsync with the peer's device id       (opt-in)
  hsa_peer       ROCr async copy with the peer GPU as source agent  (opt-in)

The opt-in paths name the peer GPU for a buffer this process maps only for the
local GPU; on MI355X they faulted the GPU (memory access fault, core dump), so
run them alone with --paths. Each path runs for one peer and for all peers at
once (one stream or signal per peer), with RCCL's all-gather as the reference.
It also reports how HIP and ROCr classify a peer buffer pointer. Builds
symm_copy_helper.cpp with hipcc on first use.
GB/s = bytes received per rank / time, slowest rank, median. Every mode checks what it received.
"""

import argparse
import ctypes
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

HERE = Path(__file__).resolve().parent
HIP_TYPES = {-1: "error", 0: "unregistered", 1: "host", 2: "device", 3: "managed", 10: "array", 11: "unified"}
HSA_TYPES = {0: "unknown", 1: "HSA", 2: "locked", 3: "graphics", 4: "IPC", 5: "reserved addr", 6: "HSA VMEM"}
HIP_MEMCPY_DEFAULT, HIP_MEMCPY_D2D = 4, 3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mib", type=int, default=92)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--output-dir", default="/output/p2p")
    p.add_argument("--helper", default="/tmp/symm_copy_helper.so")
    p.add_argument(
        "--paths",
        default="torch,hip_default,hip_d2d,hsa_local,hsa_local_sdma,hsa_local_sdma_fixed,hsa_local_sdma_pairfree,"
        "hsa_local_sdma_distinct,hsa_local_sdma_distinct_rev,hsa_local_sdma_split",
        type=lambda s: s.split(","),
        help="comma list of: torch, hip_default, hip_d2d, hsa_local, hsa_local_sdma, hsa_local_sdma_fixed, "
        "hsa_local_sdma_pairfree, hsa_local_sdma_distinct, hsa_local_sdma_distinct_rev, hsa_local_sdma_split, hip_peer, hsa_peer "
        "(the last two name the peer GPU and can fault; run them alone)",
    )
    return p.parse_args()


def load_helper(path, rank):
    so = Path(path)
    src = HERE / "symm_copy_helper.cpp"
    if rank == 0 and (not so.exists() or so.stat().st_mtime < src.stat().st_mtime):
        subprocess.run(["hipcc", "-O2", "-shared", "-fPIC", "-o", str(so), str(src), "-lhsa-runtime64"], check=True)
    dist.barrier()
    lib = ctypes.CDLL(str(so))
    vp, i, sz = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
    ip = ctypes.POINTER(ctypes.c_int)
    for name, args in {
        "sch_init": [i, i],
        "sch_hip_copy": [vp, vp, sz, i, i],
        "sch_hip_copy_peer": [vp, i, vp, i, sz, i],
        "sch_hip_sync": [i],
        "sch_hsa_copy": [vp, i, vp, i, sz, i],
        "sch_hsa_copy_on_engine": [vp, i, vp, i, sz, i, i],
        "sch_engine_for": [i, i],
        "sch_engine_mask": [i, i],
        "sch_hsa_wait": [i],
        "sch_hip_pointer": [vp, ip, ip],
        "sch_hsa_pointer": [vp, ip, ip],
        "sch_bus_of_device": [i],
    }.items():
        getattr(lib, name).argtypes = args
        getattr(lib, name).restype = i
    return lib


def check(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed with {rc}")


def sdma_plans(lib, local_rank, peers):
    """SDMA engines (bit indices) per peer for copies that name the local GPU on both sides.

    preferred  ROCr's preferred engine for each peer link (peers can share one)
    fixed      preferred, with each clash moved to a free engine no peer prefers
    pairfree   fixed, but no two peers on engines 2k and 2k+1 (which may share bandwidth)
    distinct   the lowest available engine not yet taken, one per peer
    split      two distinct engines per peer, each copying half
    """
    preferred = [lib.sch_engine_for(local_rank, p) for p in peers]
    masks = [lib.sch_engine_mask(local_rank, p) for p in peers]
    avail = [[e for e in range(31) if m > 0 and m >> e & 1] for m in masks]

    def take(i, used, avoid=()):
        free = [e for e in avail[i] if e not in used and e not in avoid] or [e for e in avail[i] if e not in used]
        return free[0] if free else -1

    distinct, used = [], set()
    for i in range(len(peers)):
        distinct.append(take(i, used))
        used.add(distinct[-1])
    fixed, used = [-1] * len(peers), set()
    for i, e in enumerate(preferred):
        if e >= 0 and e not in used:
            fixed[i] = e
            used.add(e)
    for i in range(len(peers)):
        if fixed[i] < 0:
            fixed[i] = take(i, used, avoid=set(preferred))
            used.add(fixed[i])
    pairfree, used = [-1] * len(peers), set()
    for i, e in enumerate(preferred):
        if e >= 0 and e not in used and e ^ 1 not in used:
            pairfree[i] = e
            used.add(e)
    for i in range(len(peers)):
        if pairfree[i] < 0:
            partners = used | {e ^ 1 for e in used} | set(preferred) | {e ^ 1 for e in preferred}
            pairfree[i] = take(i, used, avoid=partners)
            used.add(pairfree[i])
    second, used = [], set(distinct)
    for i in range(len(peers)):
        second.append(take(i, used))
        used.add(second[-1])
    return {
        "preferred": [[e] for e in preferred],
        "fixed": [[e] for e in fixed],
        "pairfree": [[e] for e in pairfree],
        "distinct": [[e] for e in distinct],
        "split": [[a, b] for a, b in zip(distinct, second)],
    }


def issue_sdma(lib, local_rank, dst_ptrs, src_ptrs, nbytes, plan, order):
    """Start one ROCr copy per (peer, engine) of `plan` for the peers in `order`; returns the signal count."""
    slot = 0
    for i in order:
        engines = plan[i]
        if min(engines) < 0:
            raise RuntimeError(f"no SDMA engine for peer index {i} ({engines})")
        part = nbytes // len(engines)
        for j, engine in enumerate(engines):
            off = j * part
            size = nbytes - off if j == len(engines) - 1 else part
            check(
                lib.sch_hsa_copy_on_engine(
                    dst_ptrs[i] + off, local_rank, src_ptrs[i] + off, local_rank, size, slot, engine
                ),
                "hsa_amd_memory_async_copy_on_engine",
            )
            slot += 1
    return slot


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    t = torch.tensor([statistics.median(times)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    world, rank = dist.get_world_size(), dist.get_rank()
    lib = load_helper(args.helper, rank)
    check(lib.sch_init(2 * world, world), "sch_init")

    numel = args.mib * (1 << 20) // 2
    nbytes = numel * 2
    src = symm_mem.empty(numel, dtype=torch.bfloat16, device="cuda")
    handle = symm_mem.rendezvous(src, dist.group.WORLD.group_name)
    src.fill_(rank)
    peers = [(rank + step) % world for step in range(1, world)]
    remote = {p: handle.get_buffer(p, (numel,), torch.bfloat16) for p in peers}
    dst = torch.empty(len(peers), numel, dtype=torch.bfloat16, device="cuda")
    streams = [torch.cuda.Stream() for _ in peers]
    torch.cuda.synchronize()
    dist.barrier()

    # How HIP and ROCr see a local and a peer buffer pointer.
    t, d = ctypes.c_int(), ctypes.c_int()
    views = {}
    for label, ptr in (
        ("local symm buffer", src.data_ptr()),
        (f"peer {peers[0]} symm buffer", remote[peers[0]].data_ptr()),
    ):
        lib.sch_hip_pointer(ptr, ctypes.byref(t), ctypes.byref(d))
        hip_view = f"{HIP_TYPES.get(t.value, t.value)} (device {d.value})"
        rc = lib.sch_hsa_pointer(ptr, ctypes.byref(t), ctypes.byref(d))
        hsa_view = "query failed" if rc else f"{HSA_TYPES.get(t.value, t.value)} (owner bus {d.value:02x})"
        views[label] = (hip_view, hsa_view)
    bus = {r: lib.sch_bus_of_device(r) for r in range(world)}

    def torch_copy(count):
        current = torch.cuda.current_stream()
        for i in range(count):
            streams[i].wait_stream(current)
            with torch.cuda.stream(streams[i]):
                dst[i].copy_(remote[peers[i]])
        for i in range(count):
            current.wait_stream(streams[i])
        torch.cuda.synchronize()

    def hip_copy(count, kind):
        for i in range(count):
            check(lib.sch_hip_copy(dst[i].data_ptr(), remote[peers[i]].data_ptr(), nbytes, i, kind), "hipMemcpyAsync")
        check(lib.sch_hip_sync(count), "hipStreamSynchronize")

    def hip_copy_peer(count):
        for i in range(count):
            check(
                lib.sch_hip_copy_peer(dst[i].data_ptr(), local_rank, remote[peers[i]].data_ptr(), peers[i], nbytes, i),
                "hipMemcpyPeerAsync",
            )
        check(lib.sch_hip_sync(count), "hipStreamSynchronize")

    def hsa_copy(count, src_is_peer):
        # The peer buffer is mapped in this process for the local GPU only, so naming the
        # peer GPU as source agent can send ROCr to an engine without that mapping.
        for i in range(count):
            src_dev = peers[i] if src_is_peer else local_rank
            check(
                lib.sch_hsa_copy(dst[i].data_ptr(), local_rank, remote[peers[i]].data_ptr(), src_dev, nbytes, i),
                "hsa_amd_memory_async_copy",
            )
        check(lib.sch_hsa_wait(count), "hsa wait")

    plans = sdma_plans(lib, local_rank, peers)
    dst_ptrs = [dst[i].data_ptr() for i in range(len(peers))]
    src_ptrs = [remote[p].data_ptr() for p in peers]

    def hsa_copy_sdma(count, plan, reverse=False):
        order = range(count - 1, -1, -1) if reverse else range(count)
        n = issue_sdma(lib, local_rank, dst_ptrs, src_ptrs, nbytes, plans[plan], order)
        check(lib.sch_hsa_wait(n), "hsa wait")

    all_paths = {
        "torch": ("torch copy_", torch_copy),
        "hip_default": ("hipMemcpyAsync, kind Default", lambda c: hip_copy(c, HIP_MEMCPY_DEFAULT)),
        "hip_d2d": ("hipMemcpyAsync, kind DeviceToDevice", lambda c: hip_copy(c, HIP_MEMCPY_D2D)),
        "hsa_local": ("ROCr async copy, src agent = local GPU", lambda c: hsa_copy(c, False)),
        "hsa_local_sdma": (
            "ROCr copy on preferred SDMA engine, src agent = local GPU",
            lambda c: hsa_copy_sdma(c, "preferred"),
        ),
        "hsa_local_sdma_fixed": (
            "ROCr copy on preferred SDMA engine, clashes moved to a free engine",
            lambda c: hsa_copy_sdma(c, "fixed"),
        ),
        "hsa_local_sdma_pairfree": (
            "ROCr copy on preferred SDMA engine, clashes moved, no engine pair 2k/2k+1 shared",
            lambda c: hsa_copy_sdma(c, "pairfree"),
        ),
        "hsa_local_sdma_distinct": (
            "ROCr copy on one distinct SDMA engine per peer, src agent = local GPU",
            lambda c: hsa_copy_sdma(c, "distinct"),
        ),
        "hsa_local_sdma_distinct_rev": (
            "ROCr copy on one distinct SDMA engine per peer, issued in reverse order",
            lambda c: hsa_copy_sdma(c, "distinct", reverse=True),
        ),
        "hsa_local_sdma_split": (
            "ROCr copy split in halves on two distinct SDMA engines per peer",
            lambda c: hsa_copy_sdma(c, "split"),
        ),
        # Opt-in: these name the peer GPU for a buffer only the local GPU maps and have
        # faulted the GPU (memory access fault, GPU core dump) on MI355X.
        "hip_peer": ("hipMemcpyPeerAsync (peer device id)", hip_copy_peer),
        "hsa_peer": ("ROCr async copy, src agent = peer GPU", lambda c: hsa_copy(c, True)),
    }
    unknown = set(args.paths) - set(all_paths)
    if unknown:
        raise SystemExit(f"unknown --paths {sorted(unknown)}; choose from {sorted(all_paths)}")
    rows = []
    for key in args.paths:
        name, fn = all_paths[key]
        for count in (1, len(peers)):
            if rank == 0:
                print(f"running: {name}, {count} peer(s)", flush=True)
            dst.zero_()
            torch.cuda.synchronize()
            try:
                fn(count)
                ok = all(bool((dst[i] == peers[i]).all()) for i in range(count))
                ms = timed(lambda: fn(count), args.warmup, args.iters)
                err = ""
            except RuntimeError as e:
                ok, ms, err = False, float("nan"), str(e)
            ok_all = torch.tensor([int(ok)], device="cuda")
            dist.all_reduce(ok_all, op=dist.ReduceOp.MIN)
            rows.append((name, count, ms, count * nbytes / (ms * 1e-3) / 1e9, bool(ok_all.item()), err))
            if rank == 0:
                # Printed as it goes: a GPU fault in a later path kills the job before the table.
                print(f"  {rows[-1][2]:.2f} ms, {rows[-1][3]:.0f} GB/s, data ok: {rows[-1][4]} {err}", flush=True)

    gathered = torch.empty(world * numel, dtype=torch.bfloat16, device="cuda")
    local = src.clone()
    ms = timed(
        lambda: (dist.all_gather_into_tensor(gathered, local), torch.cuda.synchronize()), args.warmup, args.iters
    )
    rows.append(("RCCL all-gather (reference)", len(peers), ms, len(peers) * nbytes / (ms * 1e-3) / 1e9, True, ""))

    if rank == 0:
        lines = [
            f"# Symmetric-memory peer reads by copy path, {world} x {torch.cuda.get_device_name()}, {args.mib} MiB per peer",
            "",
            f"torch {torch.__version__}, HIP {torch.version.hip}. All ranks read at once; "
            "GB/s = bytes received per rank / wall time (slowest rank, median).",
            "",
            "Pointer classification (rank 0):",
            "",
            "| buffer | HIP (hipPointerGetAttributes) | ROCr (hsa_amd_pointer_info) |",
            "|---|---|---|",
        ]
        lines += [f"| {k} | {v[0]} | {v[1]} |" for k, v in views.items()]
        lines += ["", f"PCI bus per HIP device: {', '.join(f'{r}:{b:02x}' for r, b in bus.items())}", ""]
        for plan, engines in plans.items():
            lines += [f"SDMA engines per peer, {plan} (rank 0): {dict(zip(peers, engines))}", ""]
        lines += ["| copy path | peers read | ms | GB/s per rank | data ok |", "|---|---|---|---|---|"]
        for name, count, t_ms, gbps, ok, err in rows:
            note = f" ({err})" if err else ""
            lines.append(f"| {name} | {count} | {t_ms:.2f} | {gbps:.0f} | {'yes' if ok else 'NO' + note} |")
        text = "\n".join(lines) + "\n"
        print(text, flush=True)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"symm_copy_paths_tp{world}.md").write_text(text)
    handle.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
