"""Summarize GPU kernel time of a torch profiler chrome trace by category.

python trace_summary.py trace.json [--top 25]
"""

import argparse
import collections
import gzip
import json
import re

CATEGORIES = [
    ("attention", re.compile(r"fmha|mha_|flash|attn_fwd|_attention", re.I)),
    ("collective", re.compile(r"nccl|rccl|all_?to_?all|allgather|all_gather|reduce_scatter|allreduce", re.I)),
    ("gemm", re.compile(r"Cijk_|gemm|hipblaslt|matmul|_mm_|wvSplitK|bf16gemm", re.I)),
    ("flydsl", re.compile(r"flydsl|fused_qk", re.I)),
    ("triton/inductor", re.compile(r"^triton_|_kernel$|^fused_", re.I)),
    ("copy/elementwise", re.compile(r"elementwise|copy|Memcpy|Memset|cat|index", re.I)),
]


def category(name):
    for label, pattern in CATEGORIES:
        if pattern.search(name):
            return label
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    opener = gzip.open if args.trace.endswith(".gz") else open
    with opener(args.trace, "rt") as handle:
        events = json.load(handle)["traceEvents"]

    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not kernels:
        print("no kernel events")
        return
    start = min(e["ts"] for e in kernels)
    end = max(e["ts"] + e["dur"] for e in kernels)
    by_cat = collections.Counter()
    by_name = collections.Counter()
    counts = collections.Counter()
    for e in kernels:
        by_cat[category(e["name"])] += e["dur"]
        by_name[e["name"]] += e["dur"]
        counts[e["name"]] += 1
    busy = sum(by_cat.values())
    print(f"span {(end - start) / 1000:.1f} ms, kernel time {busy / 1000:.1f} ms, {len(kernels)} kernels")
    for label, dur in by_cat.most_common():
        print(f"  {label:18s} {dur / 1000:9.2f} ms  {100 * dur / busy:5.1f}%")
    print("top kernels:")
    for name, dur in by_name.most_common(args.top):
        print(f"  {dur / 1000:9.2f} ms  x{counts[name]:5d}  [{category(name)}] {name[:110]}")


if __name__ == "__main__":
    main()
