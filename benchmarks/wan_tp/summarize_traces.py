"""Summarize run_profiles.sh traces: where GPU time goes and how much of the TP communication is hidden.

    python benchmarks/wan_tp/summarize_traces.py [--root ./output/profiles] [--rank 0] [CONFIG ...]

For each config's trace (one profiled pipeline iteration): GPU time per kernel
category, each NCCL/RCCL collective's count, mean time and bus bandwidth, and
how much collective time overlaps compute kernels on another stream. Exposed
collective time is what a comm/GEMM overlap can still win back. Collectives
are identified by the "Collective name" the profiler attaches to their kernels,
on CUDA and ROCm alike.
"""

import argparse
import collections
import gzip
import json
import pathlib
import re

# First match wins: CK and cuDNN convolutions (the VAE) carry "gemm" in their names.
CATEGORIES = [
    ("conv(vae)", re.compile(r"conv|implicit_gemm", re.I)),
    ("attention", re.compile(r"fmha|flash|sdpa|attn_fwd", re.I)),
    ("gemm", re.compile(r"Cijk_|nvjet|cutlass|xmma|gemm|matmul", re.I)),
]


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _length(intervals):
    return sum(end - start for start, end in intervals)


def _intersection(a, b):
    total, i, j = 0.0, 0, 0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def summarize(path):
    kernels = [e for e in json.load(gzip.open(path))["traceEvents"] if e.get("cat") == "kernel"]
    by_category = collections.Counter()
    collectives = collections.defaultdict(list)
    comm_spans, compute_spans = [], []
    for kernel in kernels:
        span = (kernel["ts"], kernel["ts"] + kernel["dur"])
        collective = kernel["args"].get("Collective name")
        if collective:
            collectives[collective].append(kernel)
            comm_spans.append(span)
            continue
        compute_spans.append(span)
        by_category[next((c for c, rx in CATEGORIES if rx.search(kernel["name"])), "other")] += kernel["dur"]
    comm, compute = _union(comm_spans), _union(compute_spans)
    busy = _length(_union(comm_spans + compute_spans))

    print(f"\n=== {path.parent.name}  ({path.name}, GPU busy {busy / 1e6:.2f} s)")
    for name, dur in by_category.most_common():
        print(f"  {name:12s} {dur / 1e6:8.2f} s")
    for name, ks in sorted(collectives.items()):
        args = ks[0]["args"]
        n = int(args.get("Group size", 1))
        total = sum(k["dur"] for k in ks)
        size = max(int(args["In msg nelems"]), int(args["Out msg nelems"])) * (
            2 if args.get("dtype") == "BFloat16" else 4
        )
        # Bus bandwidth as nccl-tests defines it: the bytes each rank moves over its links.
        factor = (2 if name.startswith("allreduce") else 1) * (n - 1) / max(n, 1)
        busbw = size * factor / (total / len(ks) * 1e-6) / 1e9
        print(
            f"  coll {name:34s} x{len(ks):5d}  total {total / 1e6:6.2f} s  mean {total / len(ks) / 1e3:7.2f} ms"
            f"  msg {size / 1e6:8.1f} MB  busbw {busbw:6.1f} GB/s"
        )
    comm_total = _length(comm)
    hidden = _intersection(comm, compute)
    print(
        f"  collectives wall {comm_total / 1e6:.2f} s, overlapping compute {hidden / 1e6:.2f} s"
        f" ({100 * hidden / max(comm_total, 1):.0f}%), exposed {(comm_total - hidden) / 1e6:.2f} s"
        f" = {100 * (comm_total - hidden) / max(busy, 1):.0f}% of GPU busy time"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("output/profiles"))
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("configs", nargs="*", help="default: every config under --root with a trace")
    args = parser.parse_args()
    trace = f"profile_trace_rank_{args.rank}.json.gz"
    configs = args.configs or sorted(p.name for p in args.root.iterdir() if (p / trace).exists())
    for config in configs:
        summarize(args.root / config / trace)


if __name__ == "__main__":
    main()
