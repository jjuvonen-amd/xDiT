"""Peer-to-peer read bandwidth: one peer, all peers in turn, all peers at once.

    torchrun --nproc_per_node 8 benchmarks/wan_tp/p2p_bandwidth.py [--mib 92] [--output-dir /output/p2p]

Separates what the interconnect can do from what torch's fused TP ops use of
it. Every rank holds a buffer in torch symmetric memory (the mechanism
symm_mem.fused_all_gather_matmul uses) and all ranks read from their peers at
the same time, as an all-gather does:

  one_peer  read one peer's buffer (rank + 1): a single link
  seq_peers read every peer's buffer one after another on one stream, the
            order torch's AsyncTP all-gather uses
  conc_peers read every peer's buffer at once, one stream per peer

Each mode runs as one copy per buffer ("dma": HIP and CUDA move large peer
copies on copy engines) and as copies of at most 1 MiB ("pieces": what torch's
native ROCm AsyncTP path does, which HIP runs as kernels). RCCL's all-gather of
the same bytes is the reference. Rates are bytes received per rank / time,
for the slowest rank; every mode checks what it received.
"""

import argparse
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

PIECE_BYTES = 1 << 20


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mib", type=int, default=92, help="buffer per rank (Wan2.2 TP8 shard: ~92 MiB)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--output-dir", default="/output/p2p")
    return p.parse_args()


def copy_dma(dst, src):
    dst.copy_(src)


def copy_pieces(dst, src):
    rows = PIECE_BYTES // src.element_size()
    for d, s in zip(dst.split(rows), src.split(rows)):
        d.copy_(s)


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
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


def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    world, rank = dist.get_world_size(), dist.get_rank()
    group = dist.group.WORLD.group_name
    numel = args.mib * (1 << 20) // 2
    bf16 = torch.bfloat16

    src = symm_mem.empty(numel, dtype=bf16, device="cuda")
    handle = symm_mem.rendezvous(src, group)
    src.fill_(rank)
    peers = [(rank + step) % world for step in range(1, world)]
    remote = {p: handle.get_buffer(p, (numel,), bf16) for p in peers}
    dst = torch.empty(len(peers), numel, dtype=bf16, device="cuda")
    streams = [torch.cuda.Stream() for _ in peers]
    torch.cuda.synchronize()
    dist.barrier()

    def one_peer(copy):
        copy(dst[0], remote[peers[0]])

    def seq_peers(copy):
        for i, p in enumerate(peers):
            copy(dst[i], remote[p])

    def conc_peers(copy):
        current = torch.cuda.current_stream()
        for i, (p, stream) in enumerate(zip(peers, streams)):
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                copy(dst[i], remote[p])
        for stream in streams:
            current.wait_stream(stream)

    shard_bytes = numel * 2
    rows = []
    for mode, fn, received in (
        ("one_peer", one_peer, 1),
        ("seq_peers", seq_peers, len(peers)),
        ("conc_peers", conc_peers, len(peers)),
    ):
        for method, copy in (("dma", copy_dma), ("pieces", copy_pieces)):
            dst.zero_()
            fn(copy)
            torch.cuda.synchronize()
            ok = all(bool((dst[i] == p).all()) for i, p in enumerate(peers[:received]))
            ok_all = torch.tensor([int(ok)], device="cuda")
            dist.all_reduce(ok_all, op=dist.ReduceOp.MIN)
            ms = timed(lambda: fn(copy), args.warmup, args.iters)
            rows.append((mode, method, received, ms, received * shard_bytes / (ms * 1e-3) / 1e9, bool(ok_all.item())))

    gathered = torch.empty(world * numel, dtype=bf16, device="cuda")
    local = src.clone()
    ms = timed(lambda: dist.all_gather_into_tensor(gathered, local), args.warmup, args.iters)
    ok = bool((gathered.view(world, numel)[:, ::4096] == torch.arange(world, device="cuda")[:, None]).all())
    ok_all = torch.tensor([int(ok)], device="cuda")
    dist.all_reduce(ok_all, op=dist.ReduceOp.MIN)
    rows.append(
        ("rccl_all_gather", "-", len(peers), ms, len(peers) * shard_bytes / (ms * 1e-3) / 1e9, bool(ok_all.item()))
    )

    if rank == 0:
        name = torch.cuda.get_device_name()
        lines = [
            f"# P2P read bandwidth, {world} x {name}, {args.mib} MiB per rank",
            "",
            f"torch {torch.__version__}, {'HIP ' + torch.version.hip if torch.version.hip else 'CUDA ' + str(torch.version.cuda)}. "
            "All ranks read at once; GB/s = bytes received per rank / time (slowest rank, median).",
            "",
            "| mode | copy | peers read | ms | GB/s per rank | data ok |",
            "|---|---|---|---|---|---|",
        ]
        for mode, method, received, t, gbps, ok in rows:
            lines.append(f"| {mode} | {method} | {received} | {t:.2f} | {gbps:.0f} | {'yes' if ok else 'NO'} |")
        text = "\n".join(lines) + "\n"
        print(text, flush=True)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"p2p_tp{world}.md").write_text(text)
    handle.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
