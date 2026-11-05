"""Do kernels read stale cache lines of a buffer that a peer's SDMA engine rewrote?

    torchrun --nproc_per_node 2 benchmarks/ulysses_a2a/sdma_coherence_probe.py [--memory coarse|fine|uncached]

Rank 1 pushes a new byte pattern into rank 0's inbox on an SDMA engine every round, with
a flag behind it. Before each round, rank 0 reads the whole inbox with a wide kernel, so
its caches (one L2 per XCD) hold the previous round's lines; after the flag wait it checks
every byte. Between rounds the ranks line up on the host only (gloo and a device
synchronize): an RCCL collective on the device might clean the caches and hide the effect.
"""

import argparse
import os

import torch
import torch.distributed as dist

from xfuser.core.distributed.sdma import transport


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--memory", default="coarse", choices=list(transport.MEMORY_KINDS))
    p.add_argument("--mib", type=float, nargs="+", default=[0.5, 2, 8])
    p.add_argument("--rounds", type=int, default=300)
    args = p.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    cpu = dist.new_group(backend="gloo")
    rank = dist.get_rank()
    g = transport.group_for(dist.group.WORLD)
    flags = g.buffer("flags", 64 * 4, memory="uncached")
    words = flags.local[:8].view(torch.int32)
    stream = torch.cuda.current_stream()
    for mib in args.mib:
        n = int(mib * 2**20) // 256 * 256
        inbox = g.buffer("inbox", n, memory=args.memory)
        src = torch.empty(n, dtype=torch.uint8, device=g.device)
        bad_rounds = 0
        for r in range(args.rounds):
            value = (r * 37) % 251 + 1
            if rank == 0:
                inbox.local[:n].float().sum()  # warm every XCD's cache with this round's old lines
                words[0].fill_(1)  # own word
            else:
                src.fill_(value)
            torch.cuda.synchronize()
            dist.barrier(group=cpu)
            if rank == 1:
                slot, call = transport.start(1, 1)
                transport.release(slot, 0, stream)
                transport.push(g, slot, 0, 0, 0, src.data_ptr(), 0, 0, inbox, 0, 0, n, 1, flags, 1 * 4)
            else:
                slot, call = transport.start(0, 0)
                transport.wait(stream, flags.ptr, 2, 0, call)
                wrong = int((inbox.local[:n] != value).sum())
                bad_rounds += wrong > 0
            torch.cuda.synchronize()
            dist.barrier(group=cpu)
        if rank == 0:
            print(f"memory {args.memory}, {mib} MiB: {bad_rounds} of {args.rounds} rounds read stale bytes", flush=True)
    g.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
