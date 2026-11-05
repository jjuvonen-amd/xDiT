"""Does an all-to-all on SDMA engines work, and keep up with RCCL, on this ROCm?

    torchrun --nproc_per_node 8 benchmarks/ulysses_a2a/sdma_ipc_probe.py \
        [--mapping ipc|symm_mem] [--memory coarse|fine|uncached] [--groups 8|2x4] [--mib 8 64 320] \
        [--output /output/step0/name.md]

Each rank pushes 1/P of a buffer to each peer of its group on the SDMA engines
(xfuser/core/distributed/sdma/transport.py): one engine per peer, all peers at
once, a GPU trigger released on the compute stream, and a flag per peer that the
receiving stream waits for. The reference is RCCL's all_to_all_single on the same
bytes.

  --mapping ipc       peers' inboxes are mapped with HIP IPC handles (the transport's own)
  --mapping symm_mem  peers' inboxes come from torch's symmetric-memory workspace, which
                      needed a ROCr fix for copies on ROCm 10.2
  --memory            inbox memory for the IPC mapping: hipMalloc (coarse), fine-grained or uncached
  --groups 2x4        two groups of 4 at once, as Ulysses 4 under CFG parallelism

Every iteration writes a new byte pattern, and the receiving GPU checks every byte
of its inbox after its flag wait, so stale reads and lost copies both show up as
mismatches. Ranks are lined up with a small all-reduce before each iteration.
Time = the slowest rank's median, from after that all-reduce and the refill to the end of the flag wait
(SDMA) or of the all-to-all (RCCL); GB/s = bytes received from peers per rank / time.
"""

import argparse
import ctypes
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from xfuser.core.distributed.sdma import transport


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mapping", default="ipc", choices=["ipc", "symm_mem"])
    p.add_argument("--memory", default="coarse", choices=["coarse", "fine", "uncached"])
    p.add_argument("--groups", default="8", choices=["8", "2x4"])
    p.add_argument("--mib", type=float, nargs="+", default=[1, 4, 16, 64, 128, 256, 320], help="buffer per rank")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--output", default="")
    return p.parse_args()


class SymmBuffer:
    """The transport's SharedBuffer interface over torch's symmetric-memory workspace."""

    def __init__(self, g, process_group, nbytes):
        import torch.distributed._symmetric_memory as symm_mem

        name = process_group.group_name
        if hasattr(symm_mem, "enable_symm_mem_for_group"):
            symm_mem.enable_symm_mem_for_group(name)
        ws = symm_mem.get_symm_mem_workspace(name, min_size=nbytes)
        self.local = ws.get_buffer(g.rank, (nbytes,), torch.uint8)
        self.ptr = self.local.data_ptr()
        self.peer_bases = (ctypes.c_uint64 * len(g.peers))(
            *[ws.get_buffer(p, (nbytes,), torch.uint8).data_ptr() for p in g.peers]
        )
        self.nbytes = nbytes
        self.ws = ws


def pattern(it, src, dst):
    return (it * 31 + src * 7 + dst * 3) % 251 + 1


def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    world_rank, world = dist.get_rank(), dist.get_world_size()
    if args.groups == "2x4":
        assert world == 8, "--groups 2x4 needs 8 ranks"
        groups = [dist.new_group([0, 1, 2, 3]), dist.new_group([4, 5, 6, 7])]
        pg = groups[world_rank // 4]
    else:
        pg = dist.group.WORLD
    g = transport.group_for(pg)
    rank, P = g.rank, g.world
    device = g.device
    stream = torch.cuda.current_stream()

    lines = []

    def say(text=""):
        if world_rank == 0:
            print(text, flush=True)
            lines.append(text)

    props = torch.cuda.get_device_properties(device)
    say(f"# SDMA all-to-all probe: {world} x {props.name}, torch {torch.__version__}, HIP {torch.version.hip}")
    say()
    say(
        f"mapping {args.mapping}, inbox memory {args.memory if args.mapping == 'ipc' else 'symm_mem'}, "
        f"groups {args.groups}; engines on rank 0 (peer: engine): {g.engine_plan()}"
    )

    flags = g.buffer("flags", 64 * 4, memory="uncached")
    flag_words = flags.local.view(torch.int32)
    tick = torch.zeros(1, device=device)
    mismatches = torch.zeros(1, dtype=torch.int64, device=device)

    say()
    say("| MiB per rank | path | ms | GB/s per rank | host issue µs | data ok |")
    say("|---|---|---|---|---|---|")
    for mib in args.mib:
        chunk = int(mib * 2**20) // P // 256 * 256
        total = chunk * P
        if args.mapping == "ipc":
            inbox = g.buffer("inbox", total, memory=args.memory)
        else:
            inbox = SymmBuffer(g, pg, total)
        if world_rank == 0 and mib == args.mib[0]:
            say(f"ROCr pointer type of a peer inbox (4 = IPC): {transport.pointer_type(inbox.peer_bases[0])}")
        src = torch.empty(P, chunk, dtype=torch.uint8, device=device)
        inbox_rows = inbox.local[:total].view(P, chunk)
        recv = torch.empty_like(src)
        mismatches.zero_()

        def sdma_once(it):
            # Past this all-reduce every rank has received the previous iteration, so every copy
            # out of `src` and into every inbox has landed.
            dist.all_reduce(tick, group=pg)
            vals = torch.tensor([pattern(it, rank, d) for d in range(P)], dtype=torch.uint8, device=device)
            src.copy_(vals[:, None].expand(P, chunk))
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            t0 = time.perf_counter()
            slot, call = transport.start(len(g.peers), len(g.peers))
            transport.release(slot, 0, stream)
            transport.push(
                g, slot, 0, 0, 0, src.data_ptr(), chunk, 0, inbox, rank * chunk, 0, chunk, 1, flags, rank * 4
            )
            host = time.perf_counter() - t0
            inbox_rows[rank].copy_(src[rank])
            flag_words[rank].fill_(1)
            transport.wait(stream, flags.ptr, P, 0, call)
            end.record()
            want = torch.tensor([pattern(it, s, rank) for s in range(P)], dtype=torch.uint8, device=device)
            mismatches.add_((inbox_rows != want[:, None]).sum())
            return start, end, host

        def rccl_once(it):
            # Past this all-reduce every rank has received the previous iteration, so every copy
            # out of `src` and into every inbox has landed.
            dist.all_reduce(tick, group=pg)
            vals = torch.tensor([pattern(it, rank, d) for d in range(P)], dtype=torch.uint8, device=device)
            src.copy_(vals[:, None].expand(P, chunk))
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            t0 = time.perf_counter()
            dist.all_to_all_single(recv, src, group=pg)
            host = time.perf_counter() - t0
            end.record()
            want = torch.tensor([pattern(it, s, rank) for s in range(P)], dtype=torch.uint8, device=device)
            mismatches.add_((recv != want[:, None]).sum())
            return start, end, host

        for name, once in (("rccl", rccl_once), ("sdma", sdma_once)):
            mismatches.zero_()
            for it in range(args.warmup):
                once(it)
            runs = [once(args.warmup + it) for it in range(args.iters)]
            torch.cuda.synchronize()
            ms = statistics.median(s.elapsed_time(e) for s, e, _ in runs)
            host_us = statistics.median(h for _, _, h in runs) * 1e6
            stats = torch.tensor([ms, host_us, float(mismatches.item())], device=device)
            every = [torch.empty_like(stats) for _ in range(world)]
            dist.all_gather(every, stats)
            every = torch.stack(every).cpu()
            worst_ms = every[:, 0].max().item()
            gbps = (P - 1) * chunk / (worst_ms * 1e-3) / 1e9
            bad = int(every[:, 2].sum().item())
            say(
                f"| {mib:g} | {name} | {worst_ms:.3f} | {gbps:.0f} | {every[:, 1].max().item():.0f} | "
                f"{'yes' if bad == 0 else f'NO ({bad} bytes)'} |"
            )

    if args.output and world_rank == 0:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text("\n".join(lines) + "\n")
    g.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
