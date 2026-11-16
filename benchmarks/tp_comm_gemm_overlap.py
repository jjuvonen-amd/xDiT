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
  symm_mem_sdma_kN  ROCm, opt-in (--symm-mem-sdma): the same torch ops on a torch patched with
              patches/pytorch (TORCH_SYMM_MEM_ENABLE_ROCM_SDMA=1 for these calls only), N pieces
              per rank. Natural row order, so the same inputs as symm_mem. symm_mem_sdma_auto: the
              patch's default piece count for the shape. With --graph also symm_mem_sdma_graph_kN /
              _graph_auto: the same call captured once in a CUDA graph and replayed (each replay
              issues the copies from a host-function node).
  sdma_kN     ROCm, opt-in (--sdma): xfuser's all-gather + GEMM with the gather on SDMA
              copy engines, all peers at once, in N pieces on the block-cyclic rows; each
              piece's GEMM starts as it lands (xfuser/model_executor/layers/sdma_all_gather.py).
              At RS sites, the GEMM in N chunks with each chunk's partial sums pushed to their
              owners on SDMA engines and reduced chunk by chunk (sdma_linear_reduce_scatter).

The all-gathers are called with return_A=False, as inference calls them.

--fp8 adds the same pairs in fp8 with row-wise scales (activation rows and weight rows quantized
to e4m3, bf16 output), checked against fp8_baseline:

  fp8_baseline    all_gather of the fp8 rows and their scales -> _scaled_mm, or _scaled_mm ->
                  reduce_scatter.
  fp8_symm_mem    torch.ops.symm_mem.fused_all_gather_scaled_matmul /
                  fused_scaled_matmul_reduce_scatter (with --symm-mem).
  fp8_symm_mem_sdma_kN  the same ops on their patched SDMA path (with --symm-mem-sdma), N pieces
                  per rank, or fp8_symm_mem_sdma_auto; fp8_symm_mem_sdma_graph_* with --graph.

`comm_only` and `gemm_only` time each half alone; `ideal` is their max. Each
variant is checked against baseline. Times are the slowest rank's median.
--hidden / --ffn set other model widths, e.g. to find where the GEMM outlasts the gather.

  torchrun --nproc_per_node 4 benchmarks/tp_comm_gemm_overlap.py
"""

import argparse
import contextlib
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
_TORCH_SDMA_ENV = ("TORCH_SYMM_MEM_ENABLE_ROCM_SDMA", "TORCH_SYMM_MEM_ROCM_SDMA_CHUNKS")
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
    parser.add_argument(
        "--symm-mem-sdma", action="store_true", help="also time torch's symm_mem ops on their patched SDMA path (ROCm)"
    )
    parser.add_argument("--graph", action="store_true", help="with --symm-mem-sdma, also time it replayed from a graph")
    parser.add_argument("--fp8", action="store_true", help="also time each pair in fp8 with row-wise scales")
    parser.add_argument("--sdma-trigger", default="gpu", choices=["gpu", "host"])
    parser.add_argument("--sdma-direction", default="push", choices=["push", "pull"])
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
        # Pad so every chunk count splits the sequence into whole blocks per rank.
        align = self.world * math.lcm(*args.chunks)
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
        _, (out,) = torch.ops.symm_mem.fused_all_gather_matmul(
            a_local, [weight.t()], 0, self.group_name, return_A=False
        )
        return out

    def ag_sdma(self, a_cyclic, weight, k):
        from xfuser.model_executor.layers.sdma_all_gather import sdma_all_gather_linear

        args = self.args
        return sdma_all_gather_linear(
            a_cyclic[None], weight, None, self.group_name, k, args.sdma_trigger, args.sdma_direction
        )[0]

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

    def rs_sdma(self, a_full, weight, k):
        from xfuser.model_executor.layers.sdma_all_gather import sdma_linear_reduce_scatter

        return sdma_linear_reduce_scatter(a_full[None], weight, self.group_name, k, self.args.sdma_trigger)[0]

    def rs_symm_mem(self, a_full, weight, out_buf):
        return torch.ops.symm_mem.fused_matmul_reduce_scatter(a_full, weight.t(), "sum", 0, self.group_name)

    @contextlib.contextmanager
    def torch_sdma(self, k, op):
        """Calls inside run torch's symm_mem ops on their SDMA path in k pieces per rank, or the
        patch's default count if k is None (patched torch)."""
        from torch.distributed._symmetric_memory import _rocm_sdma

        saved = {name: os.environ.get(name) for name in _TORCH_SDMA_ENV}
        os.environ.update({"TORCH_SYMM_MEM_ENABLE_ROCM_SDMA": "1", "TORCH_SYMM_MEM_ROCM_SDMA_CHUNKS": str(k or "")})
        calls = _rocm_sdma.num_calls[op]
        try:
            yield
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        if _rocm_sdma.num_calls[op] == calls:
            raise RuntimeError(
                f"torch's {op} did not take its SDMA path (unvalidated GPU? set "
                "TORCH_SYMM_MEM_ROCM_SDMA_ALLOW_UNTESTED=1)"
            )

    @staticmethod
    def graphed(fn):
        """fn captured once into a CUDA graph (eager calls first sized the workspace): (replay, its output)."""
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fn()
        return graph.replay, out

    def time_torch_sdma(self, results, op, fn, expected, prefix=""):
        """{prefix}symm_mem_sdma_k{k} for `fn` (a stock symm_mem call) for each --chunks k and the
        default count, and replayed from a graph with --graph."""
        for k in [*self.args.chunks, None]:
            tag = _sdma_tag(k)
            with self.torch_sdma(k, op):
                self.check(f"{prefix}symm_mem_sdma_{tag}", fn(), expected)
                results[f"{prefix}symm_mem_sdma_{tag}"] = self.time(fn)
                if self.args.graph:
                    replay, out = self.graphed(fn)
                    replay()
                    self.check(f"{prefix}symm_mem_sdma_graph_{tag}", out, expected)
                    results[f"{prefix}symm_mem_sdma_graph_{tag}"] = self.time(replay)

    # ---- fp8, row-wise scales ------------------------------------------------------------------
    @staticmethod
    def quantize(x):
        """x as e4m3 rows and their fp32 scales, (rows, 1): x ~= q * scale."""
        dtype = (
            torch.float8_e4m3fnuz if "gfx94" in torch.cuda.get_device_properties().gcnArchName else torch.float8_e4m3fn
        )
        scale = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / torch.finfo(dtype).max
        return (x.float() / scale).to(dtype), scale

    def scaled_mm(self, a, a_scale, w, w_scale):
        return torch._scaled_mm(a, w.t(), a_scale, w_scale, out_dtype=torch.bfloat16)

    def ag_fp8_baseline(self, a_local, a_scale, w, w_scale):
        full = torch.empty(self.seq, a_local.shape[1], dtype=torch.uint8, device=a_local.device)
        dist.all_gather_into_tensor(full, a_local.view(torch.uint8))
        scales = torch.empty(self.seq, 1, device=a_local.device)
        dist.all_gather_into_tensor(scales, a_scale)
        return self.scaled_mm(full.view(a_local.dtype), scales, w, w_scale)

    def ag_fp8_symm_mem(self, a_local, a_scale, w, w_scale):
        _, (out,) = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
            a_local, [w.t()], a_scale, [w_scale], 0, self.group_name,
            biases=[None], result_scales=[None], out_dtypes=[torch.bfloat16], use_fast_accum=[False],
        )  # fmt: skip
        return out

    def rs_fp8_baseline(self, a_full, a_scale, w, w_scale):
        partial = self.scaled_mm(a_full, a_scale, w, w_scale)
        out = partial.new_empty(self.rows, partial.shape[1])
        dist.reduce_scatter_tensor(out, partial)
        return out

    def rs_fp8_symm_mem(self, a_full, a_scale, w, w_scale):
        return torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter(
            a_full, w.t(), a_scale, w_scale, "sum", 0, 0, self.group_name,
            [self.seq, w.shape[0]], out_dtype=torch.bfloat16,
        )  # fmt: skip

    def run_site_fp8(self, results, kind, weight, a):
        """fp8 variants of the site; `a` is this rank's rows for AG, its partial-sum input for RS."""
        w, w_scale = self.quantize(weight)
        w_scale = w_scale.t().contiguous()  # (1, out features), as _scaled_mm takes it
        a, a_scale = self.quantize(a)
        if kind == "ag":
            baseline, symm_mem, op = self.ag_fp8_baseline, self.ag_fp8_symm_mem, "fused_all_gather_scaled_matmul"
        else:
            baseline, symm_mem, op = self.rs_fp8_baseline, self.rs_fp8_symm_mem, "fused_scaled_matmul_reduce_scatter"
        expected = baseline(a, a_scale, w, w_scale).clone()
        results["fp8_baseline"] = self.time(lambda: baseline(a, a_scale, w, w_scale))
        fn = lambda: symm_mem(a, a_scale, w, w_scale)  # noqa: E731
        if self.args.symm_mem:
            self.check("fp8_symm_mem", fn(), expected)
            results["fp8_symm_mem"] = self.time(fn)
        if self.args.symm_mem_sdma:
            self.time_torch_sdma(results, op, fn, expected, prefix="fp8_")

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
            if self.args.symm_mem_sdma:
                fn = lambda: self.ag_symm_mem(a_local, weight, out)  # noqa: E731
                self.time_torch_sdma(results, "fused_all_gather_matmul", fn, expected)
            if self.args.sdma:
                for k in self.args.chunks:
                    a_cyclic = self.cyclic_rows(x_full, k).contiguous()
                    self.check(f"sdma_k{k}", self.ag_sdma(a_cyclic, weight, k), expected)
                    results[f"sdma_k{k}"] = self.time(lambda: self.ag_sdma(a_cyclic, weight, k))
            if self.args.fp8:
                self.run_site_fp8(results, kind, weight, a_local)
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
            if self.args.symm_mem_sdma:
                fn = lambda: self.rs_symm_mem(a_full, weight, out)  # noqa: E731
                self.time_torch_sdma(results, "fused_matmul_reduce_scatter", fn, expected_local)
            if self.args.sdma:
                for k in self.args.chunks:
                    self.check(f"sdma_k{k}", self.rs_sdma(a_full, weight, k), self.cyclic_rows(expected_full, k))
                    results[f"sdma_k{k}"] = self.time(lambda: self.rs_sdma(a_full, weight, k))
            if self.args.fp8:
                self.run_site_fp8(results, kind, weight, a_full)
        results["ideal"] = max(results["comm_only"], results["gemm_only"])
        return results


def _sdma_tag(k):
    return f"k{k}" if k else "auto"


def _report(bench, table):
    args = bench.args
    sdma_ks = [*args.chunks, None]
    variants = (
        ["baseline"]
        + [f"chunked_k{k}" for k in args.chunks]
        + (["symm_mem"] if args.symm_mem else [])
        + ([f"symm_mem_sdma_{_sdma_tag(k)}" for k in sdma_ks] if args.symm_mem_sdma else [])
        + ([f"symm_mem_sdma_graph_{_sdma_tag(k)}" for k in sdma_ks] if args.symm_mem_sdma and args.graph else [])
        + ([f"sdma_k{k}" for k in args.chunks] if args.sdma else [])
    )
    fp8 = (
        ["fp8_baseline"]
        + (["fp8_symm_mem"] if args.symm_mem else [])
        + ([f"fp8_symm_mem_sdma_{_sdma_tag(k)}" for k in sdma_ks] if args.symm_mem_sdma else [])
        + ([f"fp8_symm_mem_sdma_graph_{_sdma_tag(k)}" for k in sdma_ks] if args.symm_mem_sdma and args.graph else [])
        if args.fp8
        else []
    )
    columns = ["comm_only", "gemm_only", "ideal", *variants, *fp8]
    channels = os.environ.get("NCCL_MAX_NCHANNELS", "default")
    lines = [
        f"# TP{bench.world} comm/GEMM overlap, hidden {args.hidden}, ffn {args.ffn}",
        "",
        f"seq {bench.seq} ({bench.rows} rows/rank), hidden {args.hidden}, ffn {args.ffn}, bf16"
        + (" and fp8 row-wise (fp8_*)" if args.fp8 else "")
        + ", "
        f"NCCL_MAX_NCHANNELS={channels}"
        + (f", SDMA {args.sdma_direction}, trigger {args.sdma_trigger}" if args.sdma else "")
        + f". Slowest rank's median ms over {args.iters} iterations.",
        "",
        "| site | " + " | ".join(columns) + " | best vs baseline |" + (" fp8 best vs fp8_baseline |" if fp8 else ""),
        "|---|" + "---|" * (len(columns) + 1 + bool(fp8)),
    ]

    def best(results, names):
        name = min(names[1:], key=lambda v: results[v]) if len(names) > 1 else names[0]
        return f"{name} {results[names[0]] / results[name]:.2f}x"

    def row(label, results):
        cells = " | ".join(f"{results[c]:.2f}" for c in columns)
        return f"| {label} | {cells} | {best(results, variants)} |" + (f" {best(results, fp8)} |" if fp8 else "")

    for site, results in table.items():
        lines.append(row(site, results))
    for mode, counts in _PER_BLOCK.items():
        if not set(counts) <= set(table):
            continue
        totals = {c: sum(table[s][c] * n for s, n in counts.items()) for c in columns}
        lines.append(row(f"**block, cross-attn {mode}**", totals))
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
    if args.symm_mem or args.symm_mem_sdma:
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
