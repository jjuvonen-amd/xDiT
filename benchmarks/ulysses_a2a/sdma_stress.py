"""Stress the SDMA Ulysses all-to-all: new data every call, every call checked exactly.

    torchrun --nproc_per_node 8 benchmarks/ulysses_a2a/sdma_stress.py [--cfg 2] [--calls 200] \
        [--seqs 96 512 2048] [--head-groups 1 2 3] [--concurrent] [--output /output/stress/name.md]

  --attend exchange  (default) the exchange alone. Each rank all-gathers the inputs and checks
                     every element of every view attention gets (Q, K, V, an extra tensor,
                     joint rows) and of the output, which a stand-in returns as 2 * Q + 1:
                     exact in any layout. Shapes: plain, joint text in front and behind, and
                     grouped-query K/V with an extra tensor.
  --attend usp       USP with --backend on RCCL and on SDMA, compared exactly. Kernels may round
                     differently for the two input layouts (the SDMA path hands them bshd
                     views, RCCL contiguous bhsd), so use a backend that makes its inputs
                     contiguous first, such as AITER.

A flag that lands before its data, or a read of stale data, shows up as a mismatch; the
report counts mismatching calls per shape and head-group count, over all ranks.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist

HEAD_DIM = 64


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg", type=int, default=1, help="CFG degree; Ulysses takes the other ranks")
    p.add_argument("--calls", type=int, default=200, help="calls per row")
    p.add_argument("--seqs", type=int, nargs="+", default=[96, 512, 2048], help="local sequence lengths")
    p.add_argument("--heads-per-rank", type=int, default=6)
    p.add_argument("--head-groups", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--concurrent", action="store_true", help="each head group on a stream of its own")
    p.add_argument("--alternate", action="store_true", help="every other call with group streams")
    p.add_argument(
        "--switch-groups", action="store_true", help="cycle through --head-groups call by call, in one row per shape"
    )
    p.add_argument("--switch-shapes", action="store_true", help="cycle through the shapes call by call, in one row")
    p.add_argument("--attend", default="exchange", choices=["exchange", "usp"])
    p.add_argument("--backend", default="AITER")
    p.add_argument("--output", default="")
    return p.parse_args()


def _bhsd(generator, device, seq, heads):
    x = torch.randn(1, seq, heads, HEAD_DIM, generator=generator, device=device, dtype=torch.bfloat16)
    return x.transpose(1, 2)


def exchange_mismatches(group, shape, seq, hq, groups, concurrent, seed):
    """Wrong elements in one exchange of `shape` (plain, joint_front, joint_rear, gqa)."""
    from xfuser.core.distributed.sdma import ulysses

    world, rank = dist.get_world_size(group), dist.get_rank(group)
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(seed * 1000 + dist.get_rank())
    hkv = hq // 2 if shape == "gqa" else hq
    joint_rows = 77 if shape.startswith("joint") else 0
    front = shape != "joint_rear"
    extras = 1 if shape == "gqa" else 0
    q = _bhsd(generator, device, seq + joint_rows, hq * world)
    k, v = (_bhsd(generator, device, seq, hkv * world) for _ in range(2))
    tensors = [q, k, v, *(_bhsd(generator, device, seq + joint_rows, hq * world) for _ in range(extras))]
    joint = [_bhsd(generator, device, joint_rows, hkv).contiguous() for _ in range(2)] if joint_rows else None
    full = []
    for x in tensors:
        parts = [torch.empty_like(x.contiguous()) for _ in range(world)]
        dist.all_gather(parts, x.contiguous(), group=group)
        full.append(torch.cat(parts, dim=2))
    heads = [x.shape[1] // world for x in tensors]
    sizes = ulysses.head_split(heads, groups)
    counts = [torch.zeros((), dtype=torch.int64, device=device) for _ in sizes]
    calls = []

    def attend(*views):
        gi = len(calls)
        calls.append(gi)
        for t, view in enumerate(views):
            off, h = sum(sizes[i][t] for i in range(gi)), sizes[gi][t]
            want = full[t][:, rank * heads[t] + off : rank * heads[t] + off + h]
            if joint and t in (1, 2):
                own = joint[t - 1][:, off : off + h]
                want = torch.cat([own, want] if front else [want, own], dim=2)
            counts[gi].add_((view != want).sum())
        return views[0] * 2 + 1

    kv = (joint_rows, 0) if front else (0, joint_rows)
    reserved = [(0, 0), kv, kv] + [(0, 0)] * extras
    local = [None, *joint] + [None] * extras if joint else None
    out = ulysses.attention(tensors, attend, group, groups, concurrent, reserved, local)
    return int(((out != q.transpose(1, 2) * 2 + 1).sum() + sum(counts)).item())


def main():
    args = parse_args()
    from yunchang.globals import PROCESS_GROUP

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
    group = PROCESS_GROUP.ULYSSES_PG

    lines = []

    def say(text=""):
        if rank == 0:
            print(text, flush=True)
            lines.append(text)

    what = "the exchange alone" if args.attend == "exchange" else f"USP, {args.backend}"
    say(
        f"# SDMA Ulysses stress ({what}): CFG {args.cfg} x Ulysses {ulysses}, {args.heads_per_rank} heads per rank, "
        f"{args.calls} calls per row{', group streams' if args.concurrent else ''}; torch {torch.__version__}, "
        f"HIP {torch.version.hip}"
    )
    say()
    say("| shape | local seq | head groups | mismatching calls (all ranks) |")
    say("|---|---|---|---|")
    shapes = ["plain", "joint_front", "joint_rear", "gqa"] if args.attend == "exchange" else ["plain"]
    group_rows = [args.head_groups] if args.switch_groups else [[g] for g in args.head_groups]
    if args.switch_shapes:
        shapes = ["/".join(shapes)]
    for shape in shapes:
        for seq in args.seqs:
            for cycle in group_rows:
                groups = "/".join(map(str, cycle))
                bad = 0
                for call in range(args.calls):
                    if args.attend == "exchange":
                        hq = args.heads_per_rank
                        concurrent = args.concurrent or (args.alternate and call % 2 == 1)
                        g = cycle[call % len(cycle)]
                        names = shape.split("/")
                        name = names[call % len(names)]
                        bad += exchange_mismatches(group, name, seq, hq, g, concurrent, call) > 0
                        continue
                    g = torch.Generator(device=device).manual_seed(call * 1000 + rank)
                    q, k, v = (_bhsd(g, device, seq, args.heads_per_rank * ulysses) for _ in range(3))
                    usp_sdma.configure("rccl", 1)
                    expected = USP(q, k, v)
                    usp_sdma.configure("sdma", cycle[call % len(cycle)], args.concurrent)
                    actual = USP(q, k, v)
                    usp_sdma.configure("rccl", 1)
                    bad += bool((actual != expected).any())
                total = torch.tensor([bad], device=device)
                dist.all_reduce(total)
                say(f"| {shape} | {seq} | {groups} | {int(total.item())} |")
    if args.output and rank == 0:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text("\n".join(lines) + "\n")
    dist.barrier()


if __name__ == "__main__":
    main()
