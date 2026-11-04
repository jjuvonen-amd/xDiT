"""Stress the SDMA Ulysses all-to-all: new data every call, every call checked against RCCL.

    torchrun --nproc_per_node 8 benchmarks/ulysses_a2a/sdma_stress.py [--cfg 2] [--calls 200] \
        [--seqs 96 512 2048] [--head-groups 1 2 3] [--output /output/stress/name.md]

Each call draws new Q/K/V and compares the SDMA path with RCCL exactly. A flag that
lands before its data, or a read of stale data, shows up as a mismatch; the report
counts mismatching calls per shape and head-group count, over all ranks, with the
first one's details.

  --attend elementwise  (default) the transport alone: the SDMA path with an elementwise
                        stand-in for attention (q + k / 2 + v / 4, exact in any layout),
                        against RCCL all-to-alls around the same function
  --attend usp          USP with --backend on RCCL and on SDMA. Kernels may round
                        differently for the two input layouts (the SDMA path hands them
                        bshd views, RCCL contiguous bhsd), so use a backend that makes its
                        inputs contiguous first, such as AITER.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg", type=int, default=1, help="CFG degree; Ulysses takes the other ranks")
    p.add_argument("--calls", type=int, default=200, help="calls per shape and head-group count")
    p.add_argument("--seqs", type=int, nargs="+", default=[96, 512, 2048], help="local sequence lengths")
    p.add_argument("--heads-per-rank", type=int, default=5)
    p.add_argument("--head-groups", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--concurrent", action="store_true", help="each head group on a stream of its own")
    p.add_argument("--attend", default="elementwise", choices=["elementwise", "usp"])
    p.add_argument("--backend", default="AITER")
    p.add_argument("--output", default="")
    return p.parse_args()


def main():
    args = parse_args()
    from xfuser.config.args import xFuserArgs
    from xfuser.core.distributed import (
        get_runtime_state,
        init_distributed_environment,
        initialize_model_parallel,
        initialize_runtime_state,
    )
    from xfuser.model_executor.layers import usp_sdma
    from xfuser.model_executor.layers.usp import USP

    local_rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    init_distributed_environment(rank=int(os.environ["RANK"]), world_size=world, local_rank=local_rank)
    ulysses = world // args.cfg
    initialize_model_parallel(classifier_free_guidance_degree=args.cfg, ulysses_degree=ulysses)
    engine_config, _ = xFuserArgs(
        attention_backend=args.backend, ulysses_degree=ulysses, use_cfg_parallel=args.cfg > 1
    ).create_config()
    initialize_runtime_state(engine_config=engine_config)
    get_runtime_state().set_attention_backend(args.backend)
    rank, device = dist.get_rank(), torch.device("cuda", local_rank)
    from xfuser.core.distributed.sdma import ulysses as sdma_ulysses
    from xfuser.model_executor.layers.usp import _combined_qkv_all_to_all, _ft_c_output_all_to_all
    from yunchang.globals import PROCESS_GROUP

    def elementwise(q, k, v):
        return q + k / 2 + v / 4

    def run(q, k, v, mode, groups):
        if args.attend == "usp":
            usp_sdma.configure(mode, groups, args.concurrent)
            try:
                return USP(q, k, v)
            finally:
                usp_sdma.configure("rccl", 1)
        if mode == "rccl":
            qa, ka, va = _combined_qkv_all_to_all(q, k, v)
            return _ft_c_output_all_to_all(elementwise(qa, ka, va))
        out = sdma_ulysses.attention(q, k, v, elementwise, PROCESS_GROUP.ULYSSES_PG, groups, args.concurrent)
        return out.transpose(1, 2)

    heads = args.heads_per_rank * ulysses

    lines = []

    def say(text=""):
        if rank == 0:
            print(text, flush=True)
            lines.append(text)

    what = "elementwise stand-in" if args.attend == "elementwise" else f"USP, {args.backend}"
    say(
        f"# SDMA Ulysses stress ({what}): CFG {args.cfg} x Ulysses {ulysses}, {heads} heads, {args.calls} calls per row"
        f"{', concurrent groups' if args.concurrent else ''}; torch {torch.__version__}, HIP {torch.version.hip}"
    )
    say()
    say("| local seq | head groups | mismatching calls (all ranks) | first: call, rank, elements, max diff |")
    say("|---|---|---|---|")
    for seq in args.seqs:
        for groups in args.head_groups:
            bad, first = 0, None
            for call in range(args.calls):
                g = torch.Generator(device=device).manual_seed(call * 1000 + rank)
                q, k, v = (
                    torch.randn(1, seq, heads, 64, generator=g, device=device, dtype=torch.bfloat16).transpose(1, 2)
                    for _ in range(3)
                )
                expected = run(q, k, v, "rccl", 1)
                actual = run(q, k, v, "sdma", groups)
                wrong = (actual != expected).sum().item()
                if wrong:
                    bad += 1
                    if first is None:
                        first = f"{call}, {rank}, {wrong}, {(actual.float() - expected.float()).abs().max().item():.3g}"
            counts = [None] * world
            dist.all_gather_object(counts, (bad, first))
            total = sum(c[0] for c in counts)
            first = next((c[1] for c in counts if c[1]), "")
            say(f"| {seq} | {groups} | {total} | {first} |")
    if args.output and rank == 0:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text("\n".join(lines) + "\n")
    dist.barrier()


if __name__ == "__main__":
    main()
