"""Does all-gather traffic slow a concurrent GEMM less on SDMA than on RCCL?

    torchrun --nproc_per_node 8 benchmarks/wan_tp/sdma_gemm_contention.py [--output-dir /output/p2p]

Each rank runs a Wan2.2-14B TP GEMM over the full sequence while its all-gather
traffic (7 peer shards of seq/P x hidden bf16 from torch symmetric memory) moves
through one of:

  rccl            RCCL all_gather_into_tensor on a side stream: uses CUs, like chunked overlap
  torch           torch copy_ of each peer shard, one stream per peer: what the fused ops do
  sdma_<plan>     ROCr copies forced onto SDMA engines (plans from symm_mem_copy_paths.py:
                  preferred, fixed, distinct, split), naming the local GPU on both sides

The two halves are independent (the GEMM doesn't wait for the data), so `rccl`
is the best case for chunked overlap and `sdma_*` the best case for an SDMA-based
fused AG+GEMM. Reported per site: each half alone, both at once (wall, GEMM and
comm time), the GEMM's slowdown, and the wall time against running RCCL then the
GEMM. Times are the slowest rank's median. Comm data is checked on every path.
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

sys.path.insert(0, str(Path(__file__).resolve().parent))
from symm_mem_copy_paths import check, issue_sdma, load_helper, sdma_plans  # noqa: E402

# GEMM shape per site at TP degree P: (output features, input features).
SITES = {
    "qkv": lambda h, f, p: (3 * h // p, h),
    "ffn_up": lambda h, f, p: (f // p, h),
    "ffn_down": lambda h, f, p: (h, f // p),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq", type=int, default=75_600, help="tokens (720p, 81 frames)")
    p.add_argument("--hidden", type=int, default=5120)
    p.add_argument("--ffn", type=int, default=13_824)
    p.add_argument("--sites", nargs="+", default=list(SITES), choices=list(SITES))
    p.add_argument("--gemm-reps", type=int, default=1, help="back-to-back GEMMs per measurement")
    p.add_argument(
        "--comms",
        default="rccl,torch,sdma_fixed,sdma_distinct,sdma_split",
        type=lambda s: s.split(","),
        help="comma list of: rccl, torch, sdma_preferred, sdma_fixed, sdma_distinct, sdma_split",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--output-dir", default="/output/p2p")
    p.add_argument("--helper", default="/tmp/symm_copy_helper.so")
    return p.parse_args()


def slowest(values):
    t = torch.tensor([statistics.median(values)], device="cuda")
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

    rows = args.seq // world
    numel = rows * args.hidden
    nbytes = numel * 2
    src = symm_mem.empty(numel, dtype=torch.bfloat16, device="cuda")
    handle = symm_mem.rendezvous(src, dist.group.WORLD.group_name)
    src.fill_(rank)
    peers = [(rank + step) % world for step in range(1, world)]
    remote = {p: handle.get_buffer(p, (numel,), torch.bfloat16) for p in peers}
    dst = torch.empty(len(peers), numel, dtype=torch.bfloat16, device="cuda")
    local = src.clone()
    gathered = torch.empty(world, numel, dtype=torch.bfloat16, device="cuda")
    comm_stream = torch.cuda.Stream()
    streams = [torch.cuda.Stream() for _ in peers]
    plans = sdma_plans(lib, local_rank, peers)
    dst_ptrs = [dst[i].data_ptr() for i in range(len(peers))]
    src_ptrs = [remote[p].data_ptr() for p in peers]
    torch.cuda.synchronize()
    dist.barrier()

    # Each comm is (issue, wait, check): issue returns a token that wait consumes.
    def rccl_issue():
        comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(comm_stream):
            dist.all_gather_into_tensor(gathered, local)
        ev = torch.cuda.Event()
        ev.record(comm_stream)
        return ev

    def torch_issue():
        events = []
        for i, s in enumerate(streams):
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                dst[i].copy_(remote[peers[i]])
            ev = torch.cuda.Event()
            ev.record(s)
            events.append(ev)
        return events

    def sdma_issue(plan):
        return issue_sdma(lib, local_rank, dst_ptrs, src_ptrs, nbytes, plans[plan], range(len(peers)))

    def peers_ok():
        return all(bool((dst[i] == peers[i]).all()) for i in range(len(peers)))

    comms = {
        "rccl": (
            "RCCL all-gather",
            rccl_issue,
            lambda ev: ev.synchronize(),
            lambda: all(bool((gathered[r] == r).all()) for r in range(world)),
        ),
        "torch": ("torch copy_, 7 streams", torch_issue, lambda evs: [e.synchronize() for e in evs], peers_ok),
    }
    for plan in plans:
        comms[f"sdma_{plan}"] = (
            f"SDMA, {plan} engines",
            lambda plan=plan: sdma_issue(plan),
            lambda n: check(lib.sch_hsa_wait(n), "hsa wait"),
            peers_ok,
        )
    unknown = set(args.comms) - set(comms)
    if unknown:
        raise SystemExit(f"unknown --comms {sorted(unknown)}; choose from {sorted(comms)}")

    def measure(gemm, comm):
        """One run: (wall ms, GEMM ms or None, comm ms or None), timed from a common start."""
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        token = comm[1]() if comm else None
        g_ms = None
        if gemm:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.gemm_reps):
                gemm()
            end.record()
        c_ms = None
        if comm:
            comm[2](token)
            c_ms = (time.perf_counter() - t0) * 1e3
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1e3
        if gemm:
            g_ms = start.elapsed_time(end)
        return wall, g_ms, c_ms

    def run(gemm, comm):
        for _ in range(args.warmup):
            measure(gemm, comm)
        samples = [measure(gemm, comm) for _ in range(args.iters)]
        return [slowest([s[k] for s in samples]) if samples[0][k] is not None else None for k in range(3)]

    comm_alone = {}
    for key in args.comms:
        if rank == 0:
            print(f"running: {comms[key][0]} alone", flush=True)
        dst.zero_()
        gathered.zero_()
        comm_alone[key] = run(None, comms[key])[2]
        ok = torch.tensor([int(comms[key][3]())], device="cuda")
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        comm_alone[key] = (comm_alone[key], bool(ok.item()))

    results = []
    for site in args.sites:
        n, k = SITES[site](args.hidden, args.ffn, world)
        a = torch.randn(args.seq, k, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(args.seq, n, dtype=torch.bfloat16, device="cuda")

        def gemm(a=a, w=w, out=out):
            torch.matmul(a, w.t(), out=out)

        if rank == 0:
            print(f"running: {site} GEMM alone", flush=True)
        gemm_alone = run(gemm, None)[1]
        for key in args.comms:
            if rank == 0:
                print(f"running: {site} GEMM with {comms[key][0]}", flush=True)
            wall, g_ms, c_ms = run(gemm, comms[key])
            results.append((site, n, k, key, gemm_alone, wall, g_ms, c_ms))

    if rank == 0:
        gb = len(peers) * nbytes / 1e9
        lines = [
            f"# GEMM under all-gather traffic, {world} x {torch.cuda.get_device_name()}",
            "",
            f"torch {torch.__version__}, HIP {torch.version.hip}. seq {args.seq}, hidden {args.hidden}, "
            f"ffn {args.ffn}, bf16; each rank receives {len(peers)} x {nbytes / 2**20:.1f} MiB. "
            f"GEMM reps per measurement: {args.gemm_reps}. Slowest rank's median of {args.iters}.",
            "",
            "Comm alone:",
            "",
            "| comm | ms | GB/s per rank | data ok |",
            "|---|---|---|---|",
        ]
        for key in args.comms:
            ms, ok = comm_alone[key]
            lines.append(f"| {comms[key][0]} | {ms:.2f} | {gb / (ms * 1e-3):.0f} | {'yes' if ok else 'NO'} |")
        lines += [
            "",
            "GEMM and comm at once (independent: the GEMM doesn't wait for the data). "
            "`vs RCCL then GEMM` = (RCCL alone + GEMM alone) / wall.",
            "",
            "| site | GEMM (M x N x K) | comm | GEMM alone ms | GEMM with comm ms | GEMM slowdown "
            "| comm with GEMM ms | wall ms | vs RCCL then GEMM |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for site, n, k, key, g_alone, wall, g_ms, c_ms in results:
            base = comm_alone["rccl"][0] + g_alone if "rccl" in comm_alone else float("nan")
            lines.append(
                f"| {site} | {args.seq} x {n} x {k} | {comms[key][0]} | {g_alone:.2f} | {g_ms:.2f} "
                f"| {g_ms / g_alone:.2f}x | {c_ms:.2f} | {wall:.2f} | {base / wall:.2f}x |"
            )
        text = "\n".join(lines) + "\n"
        print(text, flush=True)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"sdma_gemm_contention_tp{world}.md").write_text(text)
    handle.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
