"""TP collectives relayed through the GPUs of the other TP group (XDIT_TP_RELAY=1), with the persistent FP8 GEMMs.

For layouts with two TP groups on one node running the same TP calls at the same time (U2 x TP4: groups
{0..3} and {4..7}), a TP group's own links carry only part of the node's bandwidth. Here every chunk a
rank sends to a TP peer goes in 1 + H pieces: one over the direct link and one through each GPU of the
other group (a helper), which receives it into a staging buffer and forwards it to the peer from a side
stream once it has landed. Each rank is the helper of the other group's transfers, symmetrically.

Transport: xDiT's SDMA transport (core/distributed/sdma/transport.py) over the world group. Copies are
registered once as plans and released per call; every plan writes one flag word per peer behind its
copies on that peer's engine (peers a plan does not target get a scratch word). The flag words live in
one buffer per rank, zeroed before a world barrier at the start of each call; a world barrier ends it.

All-gather + GEMM (FP8): the row scales go first, direct; chunk c of every rank's rows lands in P = 1 + H
pieces, and the persistent kernel (layers/flydsl_tp_gemm.py, "ag") waits in-kernel for all P pieces of a
(chunk, rank) before its tiles. Needs rows per piece of whole 256-row tiles (XDIT_TP_ROW_ALIGN=256).
"""

import functools
from typing import Dict, Optional

import torch
import torch.distributed as dist

import xfuser.envs as envs
from xfuser.logger import init_logger

logger = init_logger(__name__)

BLOCK_M = 256
_ALIGN = 256
_relays: Dict[str, "TPRelay"] = {}
_side: Dict[int, torch.cuda.Stream] = {}
calls = {"all_gather_gemm": 0, "gemm_reduce_scatter": 0}


def enabled() -> bool:
    return envs.environment_variables["XDIT_TP_RELAY"]().strip() == "1"


@functools.lru_cache(maxsize=None)
def _wait_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def wait_flags(F, IDX, n):
        """Spin (system-scope acquire) until every int32 word F[IDX[i]] (i < n) is >= 1."""
        for i in range(n):
            w = tl.load(IDX + i)
            v = tl.atomic_add(F + w, 0, sem="acquire", scope="sys")
            while v < 1:
                v = tl.atomic_add(F + w, 0, sem="acquire", scope="sys")

    return wait_flags


def _pieces(nbytes: int, helpers: int, direct_frac: float):
    """[(offset, size)] of a chunk's pieces: the direct one first, then one per helper; 256-byte aligned."""
    d = int(nbytes * direct_frac) // _ALIGN * _ALIGN
    hp = ((nbytes - d) // helpers) // _ALIGN * _ALIGN
    sizes = [d] + [hp] * (helpers - 1) + [nbytes - d - hp * (helpers - 1)]
    return [(sum(sizes[:j]), s) for j, s in enumerate(sizes)]


class TPRelay:
    """The relay state of one TP group: the world transport, the helpers, buffers and plans by shape."""

    def __init__(self, group_name: str):
        from xfuser.core.distributed.parallel_state import get_world_group
        from xfuser.core.distributed.sdma import transport

        self.transport = transport
        group = dist.distributed_c10d._resolve_process_group(group_name)
        self.ranks = dist.get_process_group_ranks(group)
        self.Wt = len(self.ranks)
        self.w = transport.group_for(get_world_group().device_group)
        world = self.w.world
        if world != 2 * self.Wt:
            raise ValueError(f"the TP relay needs two TP groups of {self.Wt} on {world} GPUs")
        everyone = [None] * world
        dist.all_gather_object(everyone, tuple(self.ranks), group=self.w.process_group)
        self.groups = sorted(set(everyone))
        if len(self.groups) != 2 or any(len(g) != self.Wt for g in self.groups):
            raise ValueError(f"the TP relay needs two disjoint TP groups of {self.Wt}; got {self.groups}")
        self.me = self.ranks.index(self.w.rank)
        # helper j of this group is rank j of the other one; this rank is helper `me` of the other group
        self.helpers = list(next(g for g in self.groups if g != tuple(self.ranks)))
        self.H = len(self.helpers)
        self.P = 1 + self.H
        self.direct_frac = float(envs.environment_variables["XDIT_TP_RELAY_DIRECT"]())
        self._plans: Dict[tuple, dict] = {}
        logger.info(f"TP relay for group {self.ranks} through {self.helpers}")

    def _one(self, dev):
        one = getattr(self, "_one_t", None)
        if one is None:
            one = self._one_t = torch.zeros(1, dtype=torch.int32, device=dev)
        return one

    # -- addresses ----------------------------------------------------------------------------------------
    def base(self, buf, rank: int) -> int:
        return buf.ptr if rank == self.w.rank else buf.peer_bases[self.w.peers.index(rank)]

    def flag_words(self, C: int):
        """Word offsets in the flag buffer: data [C][Wt][P], scales [Wt], stage [C][Wt], scratch."""
        data = 0
        scale = data + C * self.Wt * self.P
        stage = scale + self.Wt
        scratch = stage + C * self.Wt
        return data, scale, stage, scratch, scratch + 1

    # -- all-gather + GEMM ------------------------------------------------------------------------------
    def _ag_plans(self, rows: int, K: int, C: int) -> dict:
        key = ("ag", rows, K, C)
        cached = self._plans.get(key)
        w, T = self.w, self.transport
        Wt, H, P = self.Wt, self.H, self.P
        block = rows // C
        cb = block * K  # bytes of a chunk (fp8)
        pieces = _pieces(cb, H, self.direct_frac)
        stage_slot = max(s for _, s in pieces[1:])
        x_buf = w.buffer("tp_relay_ag_x", Wt * rows * K)
        s_buf = w.buffer("tp_relay_ag_s", Wt * rows * 4)
        st_buf = w.buffer("tp_relay_ag_stage", Wt * Wt * C * stage_slot)  # [client sender][client dest][c]
        nwords = self.flag_words(C)[-1]
        f_buf = w.buffer("tp_relay_flags", nwords * 4)
        bufs = (x_buf, s_buf, st_buf, f_buf)
        if cached is not None and all(b.live for b in bufs) and cached["bufs"] == [id(b) for b in bufs]:
            return cached
        data, scale, stage, scratch, _ = self.flag_words(C)
        me, ranks, helpers = self.me, self.ranks, self.helpers

        def flag(rank, word):
            return self.base(f_buf, rank) + word * 4

        def stage_addr(helper_rank, sender, dest, c):
            return self.base(st_buf, helper_rank) + ((sender * Wt + dest) * C + c) * stage_slot

        peers = w.peers
        # row scales: rows * 4 bytes to every TP peer, flag scale[me] there
        copies, flags = [], []
        for q in peers:
            if q in ranks:
                copies.append([(0, self.base(s_buf, q) + me * rows * 4, rows * 4)])
                flags.append(flag(q, scale + me))
            else:
                copies.append([])
                flags.append(flag(q, scratch))
        scale_plan = T.Plan(w, copies, flags)
        # chunk c: the direct piece to each TP peer, the helper pieces (for every dest) to each helper
        chunk_plans = []
        for c in range(C):
            copies, flags = [], []
            for q in peers:
                if q in ranks:
                    off, size = pieces[0]
                    copies.append([(c * cb + off, self.base(x_buf, q) + me * rows * K + c * cb + off, size)])
                    flags.append(flag(q, data + (c * Wt + me) * P + 0))
                else:
                    j = helpers.index(q)
                    off, size = pieces[1 + j]
                    mine = []
                    for p_idx, p in enumerate(ranks):
                        if p_idx != me:
                            mine.append((c * cb + off, stage_addr(q, me, p_idx, c), size))
                    copies.append(mine)
                    flags.append(flag(q, stage + c * Wt + me))
            chunk_plans.append(T.Plan(w, copies, flags))
        # as a helper of the other group: forward client sender i's pieces of chunk c to each client dest
        clients = helpers  # the other group, in its own rank order
        hidx = me  # this rank's index among that group's helpers
        off, size = pieces[1 + hidx]
        fwd_plans = []
        for c in range(C):
            per_sender = []
            for i in range(Wt):
                copies, flags = [], []
                for q in peers:
                    if q in clients and clients.index(q) != i:
                        d = clients.index(q)
                        copies.append(
                            [
                                (
                                    stage_addr(w.rank, i, d, c) - st_buf.ptr,
                                    self.base(x_buf, q) + i * rows * K + c * cb + off,
                                    size,
                                )
                            ]
                        )
                        flags.append(flag(q, data + (c * Wt + i) * P + 1 + hidx))
                    else:
                        copies.append([])
                        flags.append(flag(q, scratch))
                per_sender.append(T.Plan(w, copies, flags))
            fwd_plans.append(per_sender)
        if cached is not None:
            for plan in [cached["scale"], *cached["chunks"], *(p for row in cached["fwd"] for p in row)]:
                plan.free()
        plans = {
            "scale": scale_plan,
            "chunks": chunk_plans,
            "fwd": fwd_plans,
            "bufs": [id(b) for b in bufs],
            "x": x_buf,
            "s": s_buf,
            "stage": st_buf,
            "flags": f_buf,
        }
        self._plans[key] = plans
        return plans

    def all_gather_gemm_fp8(self, xq, xs, w_t, w_s, chunks: int) -> torch.Tensor:
        import flydsl.expr as fx
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

        from xfuser.model_executor.layers.flydsl_tp_gemm import compile_persistent_ag_gemm_fp8

        rows, K = xq.shape
        N = w_t.shape[1]
        C, Wt, P = chunks, self.Wt, self.P
        block = rows // C
        pl = self._ag_plans(rows, K, C)
        data, scale, stage, scratch, nwords = self.flag_words(C)
        dev = xq.device
        cur = torch.cuda.current_stream()
        side = _side.setdefault(dev.index, torch.cuda.Stream(device=dev, priority=-1))
        flags = pl["flags"].local.view(torch.int32)[:nwords]
        x_in = pl["x"].local[: Wt * rows * K].view(Wt, rows * K)
        s_in = pl["s"].local[: Wt * rows * 4].view(torch.float32).view(Wt, rows)
        xq = xq.contiguous()
        xs = xs.reshape(rows).float().contiguous()
        flags.zero_()
        f3 = flags[data : data + C * Wt * P].view(C, Wt, P)
        f3[:, self.me, :] = 1
        flags[scale + self.me] = 1
        x_in[self.me].copy_(xq.view(torch.uint8).view(-1))
        s_in[self.me].copy_(xs)
        wait = _wait_kernel()
        if "idx" not in pl:  # built once per shape: a per-call host->device copy would stall the host
            pl["idx"] = (
                torch.tensor([scale + i for i in range(Wt) if i != self.me], dtype=torch.int32, device=dev),
                [
                    torch.tensor([stage + c * Wt + i], dtype=torch.int32, device=dev)
                    for c in range(C)
                    for i in range(Wt)
                ],
            )
        scale_idx, stage_idx = pl["idx"]
        # every rank's flags zeroed and slots free past this (stream-ordered) barrier; both TP groups call together
        dist.all_reduce(self._one(dev), group=self.w.process_group)
        call = self.w.begin()
        call.send(pl["scale"], xs, cur)
        for c in range(C):
            call.send(pl["chunks"][c], xq, cur)
        # helper side: forward each client sender's piece once it has landed in this rank's staging
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            for c in range(C):
                for i in range(Wt):
                    wait[(1,)](flags, stage_idx[c * Wt + i], 1)
                    call.send(pl["fwd"][c][i], pl["stage"].local, side)
        out = torch.empty(Wt * rows, N, dtype=torch.bfloat16, device=dev)
        wait[(1,)](flags, scale_idx, scale_idx.numel())
        exe = compile_persistent_ag_gemm_fp8(K=K, BLOCK_M=BLOCK_M, BLOCK_N=256)
        # leave CUs for the forwarders on the side stream: other ranks wait for what they forward
        progs = torch.cuda.get_device_properties(dev).multi_processor_count * 7 // 8
        _run_compiled(
            exe,
            x_in[0][: rows * K].view(torch.int8),  # rank-0-sized views: the kernel rebases per rank
            w_t.t().view(torch.int8).reshape(-1),
            out.view(-1)[: rows * N],
            s_in[0][:rows],
            w_s.reshape(-1).float().contiguous(),
            pl["flags"].ptr,
            1,
            self.me,
            Wt,
            rows,
            block,
            rows,
            rows,
            data,
            P,
            N,
            progs,
            fx.Stream(cur),
        )
        cur.wait_stream(side)
        call.end()
        # every copy into or out of this call's buffers is done past this barrier
        dist.all_reduce(self._one(dev), group=self.w.process_group)
        calls["all_gather_gemm"] += 1
        return out


def relay_for(group_name: str) -> Optional[TPRelay]:
    """The relay of TP group `group_name` (collective over the world on first use), or None if it does not apply."""
    if not enabled():
        return None
    r = _relays.get(group_name)
    if r is None:
        try:
            r = TPRelay(group_name)
        except ValueError as e:
            logger.warning(f"TP relay off: {e}")
            r = False
        _relays[group_name] = r
    return r or None
