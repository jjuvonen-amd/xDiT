"""Benchmark overlapping Megatron-TP collectives with their GEMMs at Wan2.2-14B shapes.

Megatron tensor parallelism with a sequence-sharded residual all-gathers rows
into each column-parallel GEMM and reduce-scatters each row-parallel GEMM's
output. This times those pairs per site of a Wan block, run serially as the
model does today and with the collective pipelined against the GEMM:

  baseline    all_gather -> GEMM, GEMM -> reduce_scatter, one stream.
  chunked_kN  RCCL on a side stream in N chunks: the GEMM of chunk c runs while
              chunk c+1 is gathered (AG), or chunk c is reduce-scattered while
              the GEMM of chunk c+1 runs (RS). Rows are block-cyclic: rank r
              holds blocks r, P+r, 2P+r, ... of P*N, so a chunk's gather output
              and reduce-scatter input are contiguous rows.
  symm_mem    torch.ops.symm_mem.fused_all_gather_matmul / fused_matmul_reduce_scatter.
              Opt-in: peer access through torch symmetric memory segfaulted
              beyond ~10 MB on torch 2.13 / ROCm 7.16.
  sdma_async  ROCm, opt-in (--sdma): xfuser's all-gather + GEMM with the gather on SDMA
  sdma_per_peer  copy engines, all peers at once, consumed by torch's ROCm GEMM that waits
              per row tile (async) or by a stream wait + GEMM per shard (per_peer); see
              xfuser/model_executor/layers/sdma_all_gather.py. AG sites only: an RS site
              reports its baseline, as XDIT_TP_COMM_OVERLAP=sdma runs it. Rows per rank
              are padded to the GEMM's 256-row tile.

`comm_only` and `gemm_only` time each half alone; `ideal` is their max. Each
variant is checked against baseline. Times are the slowest rank's median.
--hidden / --ffn set other model widths, e.g. to find where the GEMM outlasts the gather.

  torchrun --nproc_per_node 4 benchmarks/tp_comm_gemm_overlap.py
"""

import argparse
import importlib
import math
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

# Per Wan2.2-14B block: (site, kind, output features, input features) at TP degree P.
# Self- and cross-attention outputs share a shape; "attn_out" counts for both.
_SITES = {
    "qkv": ("ag", lambda h, f, p: 3 * h // p, lambda h, f, p: h),
    "attn2_q": ("ag", lambda h, f, p: h // p, lambda h, f, p: h),
    "ffn_up": ("ag", lambda h, f, p: f // p, lambda h, f, p: h),
    "attn_out": ("rs", lambda h, f, p: h, lambda h, f, p: h // p),
    "ffn_down": ("rs", lambda h, f, p: h, lambda h, f, p: f // p),
}
# How often each site runs per block, with cross-attention sharded or replicated.
_PER_BLOCK = {
    "sharded": {"qkv": 1, "attn2_q": 1, "ffn_up": 1, "attn_out": 2, "ffn_down": 1},
    "replicated": {"qkv": 1, "ffn_up": 1, "attn_out": 1, "ffn_down": 1},
}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq", type=int, default=75_600, help="tokens (720p, 81 frames)")
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--ffn", type=int, default=13_824)
    parser.add_argument("--sites", nargs="+", default=list(_SITES), choices=list(_SITES))
    parser.add_argument("--chunks", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--symm-mem", action="store_true", help="also time torch's fused symm_mem ops")
    parser.add_argument("--sdma", action="store_true", help="also time the SDMA all-gather + GEMM (ROCm)")
    parser.add_argument("--sdma-trigger", default="gpu", choices=["gpu", "host"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output-dir", default="/output/fusion_bench")
    return parser.parse_args()


class Bench:
    def __init__(self, args):
        self.args = args
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.group_name = dist.group.WORLD.group_name
        self.comm = torch.cuda.Stream()
        # Pad so every chunk count splits the sequence into whole blocks per rank, and
        # with --sdma so each rank's rows are whole 256-row GEMM tiles.
        align = self.world * (math.lcm(max(args.chunks), 256) if args.sdma else max(args.chunks))
        self.seq = -(-args.seq // align) * align
        self.rows = self.seq // self.world

    # ---- layouts -------------------------------------------------------------------------------
    def natural_rows(self, x_full):
        return x_full[self.rank * self.rows : (self.rank + 1) * self.rows]

    def cyclic_rows(self, x_full, k):
        """This rank's blocks r, P+r, ... of P*k, stacked: the block-cyclic local layout."""
        blocks = x_full.view(k, self.world, self.rows // k, -1)
        return blocks[:, self.rank].reshape(self.rows, -1)

    # ---- all-gather + GEMM ---------------------------------------------------------------------
    def ag_baseline(self, a_local, weight, out_buf):
        full = torch.empty(self.seq, a_local.shape[1], dtype=a_local.dtype, device=a_local.device)
        dist.all_gather_into_tensor(full, a_local)
        return torch.matmul(full, weight.t(), out=out_buf)

    def ag_chunked(self, a_cyclic, weight, out_buf, k):
        compute = torch.cuda.current_stream()
        full = torch.empty(self.seq, a_cyclic.shape[1], dtype=a_cyclic.dtype, device=a_cyclic.device)
        local_chunks, full_chunks, out_chunks = a_cyclic.chunk(k), full.chunk(k), out_buf.chunk(k)
        events = [torch.cuda.Event() for _ in range(k)]
        self.comm.wait_stream(compute)
        with torch.cuda.stream(self.comm):
            for c in range(k):
                dist.all_gather_into_tensor(full_chunks[c], local_chunks[c])
                events[c].record()
        for c in range(k):
            compute.wait_event(events[c])
            torch.matmul(full_chunks[c], weight.t(), out=out_chunks[c])
        full.record_stream(self.comm)
        return out_buf

    def ag_symm_mem(self, a_local, weight, out_buf):
        _, (out,) = torch.ops.symm_mem.fused_all_gather_matmul(a_local, [weight.t()], 0, self.group_name)
        return out

    def ag_sdma(self, a_local, weight, consumer):
        from xfuser.model_executor.layers.sdma_all_gather import sdma_all_gather_linear

        out = sdma_all_gather_linear(a_local[None], weight, None, self.group_name, consumer, self.args.sdma_trigger)
        return out[0]

    # ---- GEMM + reduce-scatter -----------------------------------------------------------------
    def rs_baseline(self, a_full, weight, out_buf):
        partial = torch.matmul(a_full, weight.t())
        dist.reduce_scatter_tensor(out_buf, partial)
        return out_buf

    def rs_chunked(self, a_full, weight, out_buf, k):
        compute = torch.cuda.current_stream()
        partial = torch.empty(self.seq, weight.shape[0], dtype=a_full.dtype, device=a_full.device)
        in_chunks, partial_chunks, out_chunks = a_full.chunk(k), partial.chunk(k), out_buf.chunk(k)
        self.comm.wait_stream(compute)
        for c in range(k):
            torch.matmul(in_chunks[c], weight.t(), out=partial_chunks[c])
            event = torch.cuda.Event()
            event.record()
            self.comm.wait_event(event)
            with torch.cuda.stream(self.comm):
                dist.reduce_scatter_tensor(out_chunks[c], partial_chunks[c])
        compute.wait_stream(self.comm)
        partial.record_stream(self.comm)
        return out_buf

    def rs_symm_mem(self, a_full, weight, out_buf):
        return torch.ops.symm_mem.fused_matmul_reduce_scatter(a_full, weight.t(), "sum", 0, self.group_name)

    # ---- timing --------------------------------------------------------------------------------
    def time(self, fn):
        for _ in range(self.args.warmup):
            fn()
        times = []
        for _ in range(self.args.iters):
            torch.cuda.synchronize()
            dist.barrier()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        slowest = torch.tensor([statistics.median(times)], device="cuda")
        dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
        return slowest.item()

    def check(self, name, actual, expected):
        error = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
        if error > 1e-2:
            raise AssertionError(f"rank {self.rank}: {name} differs from baseline by {error:.3e}")

    def run_site(self, site):
        kind, out_features, in_features = _SITES[site]
        h, f, p = self.args.hidden, self.args.ffn, self.world
        n, k_in = out_features(h, f, p), in_features(h, f, p)
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        # Same global activations on every rank for AG; a different partial-sum input per rank for RS.
        torch.manual_seed(0 if kind == "ag" else 1 + self.rank)
        weight = torch.randn(n, k_in, **bf16) / k_in**0.5
        results = {}
        if kind == "ag":
            x_full = torch.randn(self.seq, k_in, **bf16)
            out = torch.empty(self.seq, n, **bf16)
            a_local = self.natural_rows(x_full).contiguous()
            expected = self.ag_baseline(a_local, weight, out).clone()
            results["baseline"] = self.time(lambda: self.ag_baseline(a_local, weight, out))
            gathered = torch.empty(self.seq, k_in, **bf16)
            results["comm_only"] = self.time(lambda: dist.all_gather_into_tensor(gathered, a_local))
            results["gemm_only"] = self.time(lambda: torch.matmul(x_full, weight.t(), out=out))
            for k in self.args.chunks:
                a_cyclic = self.cyclic_rows(x_full, k).contiguous()
                self.check(f"chunked_k{k}", self.ag_chunked(a_cyclic, weight, out, k), expected)
                results[f"chunked_k{k}"] = self.time(lambda: self.ag_chunked(a_cyclic, weight, out, k))
            if self.args.symm_mem:
                self.check("symm_mem", self.ag_symm_mem(a_local, weight, out), expected)
                results["symm_mem"] = self.time(lambda: self.ag_symm_mem(a_local, weight, out))
            if self.args.sdma:
                for variant, consumer in (("sdma_async", "async_mm"), ("sdma_per_peer", "per_peer")):
                    self.check(variant, self.ag_sdma(a_local, weight, consumer), expected)
                    results[variant] = self.time(lambda: self.ag_sdma(a_local, weight, consumer))
        else:
            a_full = torch.randn(self.seq, k_in, **bf16)
            out = torch.empty(self.rows, n, **bf16)
            expected_local = self.rs_baseline(a_full, weight, out).clone()
            results["baseline"] = self.time(lambda: self.rs_baseline(a_full, weight, out))
            partial = torch.matmul(a_full, weight.t())
            results["comm_only"] = self.time(lambda: dist.reduce_scatter_tensor(out, partial))
            results["gemm_only"] = self.time(lambda: torch.matmul(a_full, weight.t(), out=partial))
            # The reduced rows in natural order, to compare the block-cyclic outputs against.
            expected_full = torch.empty(self.seq, n, **bf16)
            dist.all_gather_into_tensor(expected_full, expected_local)
            for k in self.args.chunks:
                # Chunk c reduce-scatters global rows c*S/k onward, leaving this rank block c*P + r.
                self.check(f"chunked_k{k}", self.rs_chunked(a_full, weight, out, k), self.cyclic_rows(expected_full, k))
                results[f"chunked_k{k}"] = self.time(lambda: self.rs_chunked(a_full, weight, out, k))
            if self.args.symm_mem:
                self.check("symm_mem", self.rs_symm_mem(a_full, weight, out), expected_local)
                results["symm_mem"] = self.time(lambda: self.rs_symm_mem(a_full, weight, out))
            if self.args.sdma:
                results["sdma_async"] = results["sdma_per_peer"] = results["baseline"]
        results["ideal"] = max(results["comm_only"], results["gemm_only"])
        return results


def _report(bench, table):
    args = bench.args
    variants = (
        ["baseline"]
        + [f"chunked_k{k}" for k in args.chunks]
        + (["symm_mem"] if args.symm_mem else [])
        + (["sdma_async", "sdma_per_peer"] if args.sdma else [])
    )
    columns = ["comm_only", "gemm_only", "ideal", *variants]
    channels = os.environ.get("NCCL_MAX_NCHANNELS", "default")
    lines = [
        f"# TP{bench.world} comm/GEMM overlap, hidden {args.hidden}, ffn {args.ffn}",
        "",
        f"seq {bench.seq} ({bench.rows} rows/rank), hidden {args.hidden}, ffn {args.ffn}, bf16, "
        f"NCCL_MAX_NCHANNELS={channels}. Slowest rank's median ms over {args.iters} iterations.",
        "",
        "| site | " + " | ".join(columns) + " | best vs baseline |",
        "|---|" + "---|" * (len(columns) + 1),
    ]
    for site, results in table.items():
        best = min(variants[1:], key=lambda v: results[v]) if len(variants) > 1 else "baseline"
        cells = " | ".join(f"{results[c]:.2f}" for c in columns)
        lines.append(f"| {site} | {cells} | {best} {results['baseline'] / results[best]:.2f}x |")
    for mode, counts in _PER_BLOCK.items():
        if not set(counts) <= set(table):
            continue
        totals = {c: sum(table[s][c] * n for s, n in counts.items()) for c in columns}
        cells = " | ".join(f"{totals[c]:.2f}" for c in columns)
        best = min(variants[1:], key=lambda v: totals[v]) if len(variants) > 1 else "baseline"
        lines.append(f"| **block, cross-attn {mode}** | {cells} | {best} {totals['baseline'] / totals[best]:.2f}x |")
    text = "\n".join(lines) + "\n"
    print(text, flush=True)
    path = Path(args.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    suffix = "" if channels == "default" else f"_ch{channels}"
    (path / f"tp{bench.world}{suffix}.md").write_text(text)


def main():
    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    if args.symm_mem:
        importlib.import_module("torch.distributed._symmetric_memory")  # registers torch.ops.symm_mem
    bench = Bench(args)
    table = {}
    with torch.no_grad():
        for site in args.sites:
            table[site] = bench.run_site(site)
            if bench.rank == 0:
                print(f"{site}: " + ", ".join(f"{k} {v:.2f}" for k, v in table[site].items()), flush=True)
    if bench.rank == 0:
        _report(bench, table)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
