"""Stress the SDMA push transport's ordering: lost flags, early or stale data.

    torchrun --nproc_per_node 8 benchmarks/wan_tp/sdma_flag_stress.py [--modes flags order] [--output-dir DIR]

Runs the push protocol of xfuser/model_executor/layers/sdma_all_gather.py
directly, without the GEMM, for many calls:

  flags  tiny pieces (64 B), MAX_CHUNKS pieces per call: almost only flag
         writes, so the receivers' flag words take concurrent writes from all
         senders. A flag that never lands makes a wait never finish.
  order  pieces of 64 KiB to the size of a Wan2.2 TP8 piece. Each sender
         rewrites its rows with a value unique to (call, sender) right before
         releasing the copies; each receiver compares every piece with that
         value right after its wait. A mismatch means data read before it
         landed (a flag ahead of its data), a stale cache line from an earlier
         call, or a piece copied before the release.

--background-rccl keeps RCCL all-gathers running on a side stream, loading
the same links. A watchdog bounds every call: if a wait does not finish, the
rank prints the flag bytes it is missing and exits with status 3, instead of
hanging. Writes <output-dir>/sdma_flag_stress_tp<P>.md.
"""

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from xfuser.model_executor.layers import sdma_all_gather as sag

WATCHDOG_S = 20.0
# Bounded flag waits give up well before the watchdog, so a lost flag is counted, not a freeze.
os.environ.setdefault("XDIT_TP_SDMA_WAIT_S", "5")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", nargs="+", default=["flags", "order"], choices=["flags", "order"])
    p.add_argument("--flag-calls", type=int, default=3000)
    p.add_argument("--order-kib", nargs="+", type=int, default=[64, 1024, 23680])
    p.add_argument("--order-calls", type=int, default=400, help="calls per piece size (scaled up for small pieces)")
    p.add_argument("--chunks", type=int, default=4)
    p.add_argument("--background-rccl", action="store_true")
    p.add_argument("--output-dir", default="/output/sdma_stress")
    return p.parse_args()


class Transport:
    """The op's push protocol (sdma_all_gather_linear, direction="push"), step by step, with hooks."""

    def __init__(self, group_name):
        self.lib = sag._load()
        self.g = sag._group(group_name, "push")
        self.group_name = group_name
        self.stream = torch.cuda.current_stream()
        self.one = sag._one(torch.device("cuda", torch.cuda.current_device())).data_ptr()
        self.gave_up = 0

    def call(self, src, chunks, piece_bytes, before_release=None, after_wait=None):
        """One push all-gather of `src` (chunks pieces of piece_bytes); returns the local inbox and flags."""
        lib, g = self.lib, self.g
        world, rank, n = g.world, g.rank, len(g.peers)
        ws, inbox, flags, bases, flag_offset = sag._inbox(
            self.group_name, rank, world, (chunks, world, piece_bytes), torch.uint8, chunks
        )
        flags.zero_()
        flags[:, rank] = 1
        slot, call = sag._start(lib, g, chunks * n, chunks * n)
        sag._barrier(ws)
        if before_release is not None:
            before_release()
        # This rank's own pieces, as the op copies them into its inbox.
        inbox[:, rank] = src.view(torch.uint8).view(chunks, piece_bytes)
        sag._release(lib, slot, 0, self.stream, "gpu")
        for c in range(chunks):
            dst = (ctypes.c_uint64 * n)(*[bases[p] + (c * world + rank) * piece_bytes for p in g.peers])
            srcs = (ctypes.c_uint64 * n)(*[src.data_ptr() + c * piece_bytes] * n)
            flag_dst = (ctypes.c_uint64 * n)(*[bases[p] + flag_offset + (c * world + rank) * 4 for p in g.peers])
            sag._check(
                lib.sag_issue_push(slot, c, 0, dst, srcs, piece_bytes, g.engine_arr, n, flag_dst, self.one), "push"
            )
            sag._wait(lib, self.stream, flags[c], c, call)
            if after_wait is not None:
                after_wait(c, inbox)
        sag._barrier(ws)
        return inbox, flags

    def take_gave_up(self):
        """Count waits that ran out of time (recorded on the GPU); they would have been hangs before."""
        piece, call, missing = ctypes.c_int(), ctypes.c_int(), ctypes.c_uint64()
        count = self.lib.sag_take_error(ctypes.byref(piece), ctypes.byref(call), ctypes.byref(missing))
        if count:
            print(
                f"rank {self.g.rank}: {count} wait(s) gave up, first call {call.value} piece {piece.value} "
                f"missing words {missing.value:#x}",
                flush=True,
            )
        self.gave_up += count


def watch(event, what, flags, rank, world):
    """Wait for `event` up to WATCHDOG_S; on timeout report the missing flags and exit."""
    deadline = time.monotonic() + WATCHDOG_S
    while not event.query():
        if time.monotonic() > deadline:
            side = torch.cuda.Stream()
            with torch.cuda.stream(side):
                words = flags.to("cpu", non_blocking=False)
            missing = {c: [r for r in range(world) if words[c, r] != 1] for c in range(words.shape[0])}
            missing = {c: m for c, m in missing.items() if m}
            print(f"FREEZE rank {rank} at {what}: pieces missing flags from {missing}", flush=True)
            os._exit(3)
        time.sleep(0.0005)


def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    world, rank = dist.get_world_size(), dist.get_rank()
    group_name = dist.group.WORLD.group_name
    t = Transport(group_name)
    rows = []

    def all_ranks(count):
        total = torch.tensor([count], device="cuda")
        dist.all_reduce(total)
        return int(total.item())

    background = None
    if args.background_rccl:
        bg_in = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
        bg_out = torch.empty(world * (64 << 20), dtype=torch.uint8, device="cuda")
        bg_stream = torch.cuda.Stream()
        bg_group = dist.new_group(list(range(world)))

        def background():
            with torch.cuda.stream(bg_stream):
                dist.all_gather_into_tensor(bg_out, bg_in, group=bg_group)

    if "flags" in args.modes:
        chunks, piece = sag.MAX_CHUNKS, 64
        src = torch.zeros(chunks * piece, dtype=torch.uint8, device="cuda")
        t0 = time.perf_counter()
        for call in range(args.flag_calls):
            if background is not None and call % 8 == 0:
                background()
            _, flags = t.call(src, chunks, piece)
            done = torch.cuda.Event()
            done.record()
            watch(done, f"flags call {call}", flags, rank, world)
            t.take_gave_up()
        ms = (time.perf_counter() - t0) * 1e3 / args.flag_calls
        rows.append(("flags", "64 B", chunks, args.flag_calls, 0, all_ranks(t.gave_up), ms))
        t.gave_up = 0
        if rank == 0:
            print(f"flags: {args.flag_calls} calls x {chunks} pieces, {ms:.3f} ms/call, no freeze", flush=True)

    if "order" in args.modes:
        chunks = args.chunks
        for kib in args.order_kib:
            elems = kib * 1024 // 2  # bf16 elements per piece
            piece = elems * 2
            calls = max(args.order_calls, args.order_calls * 1024 // max(kib, 1) // 4)
            src = torch.empty(chunks, elems, dtype=torch.bfloat16, device="cuda")
            bad = torch.zeros((), dtype=torch.int64, device="cuda")
            ranks = torch.arange(world, device="cuda")
            t0 = time.perf_counter()
            for call in range(calls):
                # Exact in bf16: integers below 256. Unique per (call, sender) modulo 251.
                value = float((call * world + rank) % 251)

                def before_release(value=value):
                    src.fill_(value)

                want = ((call * world + ranks) % 251).to(torch.bfloat16)[:, None]

                def after_wait(c, inbox, want=want):
                    got = inbox[c].view(world, piece).view(torch.bfloat16)
                    bad.add_((got != want).sum())

                if background is not None and call % 8 == 0:
                    background()
                _, flags = t.call(src, chunks, piece, before_release, after_wait)
                done = torch.cuda.Event()
                done.record()
                watch(done, f"order {kib} KiB call {call}", flags, rank, world)
                t.take_gave_up()
            mismatches = int(bad.item())
            total = torch.tensor([mismatches], device="cuda")
            dist.all_reduce(total)
            ms = (time.perf_counter() - t0) * 1e3 / calls
            rows.append(("order", f"{kib} KiB", chunks, calls, int(total.item()), all_ranks(t.gave_up), ms))
            t.gave_up = 0
            if rank == 0:
                print(
                    f"order {kib} KiB: {calls} calls, {int(total.item())} mismatched elements, {ms:.3f} ms/call",
                    flush=True,
                )

    torch.cuda.synchronize()
    if rank == 0:
        lines = [
            f"# SDMA push transport stress, {world} x {torch.cuda.get_device_name()}",
            "",
            f"torch {torch.__version__}, HIP {torch.version.hip}. Background RCCL traffic: "
            f"{'yes' if args.background_rccl else 'no'}. Watchdog {WATCHDOG_S:.0f} s per call: none fired.",
            "",
            "| mode | piece | pieces per call | calls | mismatched elements | flag waits that gave up | ms per call |",
            "|---|---|---|---|---|---|---|",
        ]
        lines += [f"| {m} | {p} | {c} | {n} | {b} | {w} | {ms:.3f} |" for m, p, c, n, b, w, ms in rows]
        lines += ["", "Mismatches and waits that gave up are summed over all ranks."]
        text = "\n".join(lines) + "\n"
        print(text, flush=True)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        suffix = "_bg" if args.background_rccl else ""
        (out / f"sdma_flag_stress_tp{world}{suffix}.md").write_text(text)
    dist.destroy_process_group()
    sys.exit(0)


if __name__ == "__main__":
    main()
