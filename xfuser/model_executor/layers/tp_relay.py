"""TP collectives relayed through the GPUs of the other TP group (XDIT_TP_RELAY=1), with the persistent FP8 GEMMs.

For layouts with two TP groups on one node running the same TP calls at the same time (U2 x TP4: groups
{0..3} and {4..7}), a TP group's own links carry only part of the node's bandwidth. Here every chunk a
rank sends to a TP peer goes in 1 + H pieces: one over the direct link and one through each GPU of the
other group (a helper), which receives it into a staging buffer and forwards it to the peer from a side
stream once it has landed. Each rank is the helper of the other group's transfers, symmetrically.

Transport: xDiT's SDMA transport (core/distributed/sdma/transport.py) over the world group. Copies are
registered once as plans and released per call; every plan writes one flag word per peer behind its
copies on that peer's engine (peers a plan does not target get a scratch word).

Protocol (no barriers): flag words are never reset; each op kind numbers its calls, every flag write of
call n writes n, and every wait is "word >= n". Buffers are doubled by call parity: a rank sends call
n + 2 into a parity only after it received every peer's call n + 1 data, which each peer (and helper)
sends only after it finished with call n's (stream order), so nobody writes a buffer still in use.
All ranks of the node make the same relayed calls in the same order (both TP groups run the same step).

All-gather + GEMM (FP8): the row scales go first, direct; chunk c of every rank's rows lands in P = 1 + H
pieces, and the persistent kernel (layers/flydsl_tp_gemm.py, "ag") waits in-kernel for all P pieces of a
(chunk, rank) before its tiles. GEMM + reduce-scatter (FP8): the persistent kernel ("rs") writes partial
sums in send order and counts tiles per (chunk, dest); one side stream, per chunk, posts this rank's
pieces once counted and then forwards the other group's staged pieces of that chunk; each destination
reduces a chunk in fp32 once all pieces from all senders are in. Rows per piece must be whole 256-row
tiles (XDIT_TP_ROW_ALIGN=256).
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
_VALUES = 1 << 22  # call numbers available as flag values before the counters wrap
_relays: Dict[str, "TPRelay"] = {}
_side: Dict[int, torch.cuda.Stream] = {}  # helper forwards
_post: Dict[int, torch.cuda.Stream] = {}  # GEMM + reduce-scatter: this rank's chunks, once their tiles are counted
calls = {"all_gather_gemm": 0, "gemm_reduce_scatter": 0}


def enabled() -> bool:
    return envs.environment_variables["XDIT_TP_RELAY"]().strip() == "1"


@functools.lru_cache(maxsize=None)
def _wait_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def wait_words(F, IDX, n, target):
        """Spin (system-scope acquire) until F[IDX[i]] >= target for every i < n."""
        for i in range(n):
            w = tl.load(IDX + i)
            v = tl.atomic_add(F + w, 0, sem="acquire", scope="sys")
            while v < target:
                v = tl.atomic_add(F + w, 0, sem="acquire", scope="sys")

    return wait_words


@functools.lru_cache(maxsize=None)
def _reduce_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_chunk(RECV, OWN, OUT, sender_stride, me, n, W: tl.constexpr, BLOCK: tl.constexpr):
        """OUT[i] = bf16(sum over r < W in rank order of part_r[i]) in fp32: part_r = RECV + r * sender_stride
        (this chunk of sender r's partials), or OWN for r == me."""
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for r in tl.static_range(W):
            if r == me:
                v = tl.load(OWN + offs, mask=m, other=0.0)
            else:
                v = tl.load(RECV + r * sender_stride + offs, mask=m, other=0.0)
            acc += v.to(tl.float32)
        tl.store(OUT + offs, acc.to(tl.bfloat16), mask=m)

    return reduce_chunk


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

        self.T = transport
        group = dist.distributed_c10d._resolve_process_group(group_name)
        self.ranks = dist.get_process_group_ranks(group)
        self.Wt = len(self.ranks)
        self.w = transport.group_for(get_world_group().device_group)
        world = self.w.world
        if world != 2 * self.Wt:
            raise ValueError(f"the TP relay needs two TP groups of {self.Wt} on {world} GPUs")
        everyone = [None] * world
        dist.all_gather_object(everyone, tuple(self.ranks), group=self.w.process_group)
        groups = sorted(set(everyone))
        if len(groups) != 2 or any(len(g) != self.Wt for g in groups):
            raise ValueError(f"the TP relay needs two disjoint TP groups of {self.Wt}; got {groups}")
        self.me = self.ranks.index(self.w.rank)
        # helper j of this group is rank j of the other one; this rank is helper `me` of the other group
        self.helpers = list(next(g for g in groups if g != tuple(self.ranks)))
        self.H = len(self.helpers)
        self.P = 1 + self.H
        self.direct_frac = float(envs.environment_variables["XDIT_TP_RELAY_DIRECT"]())
        self.n = {"ag": 0, "rs": 0}  # calls per op kind (flag values)
        self._plans: Dict[tuple, dict] = {}
        dev = torch.device("cuda", torch.cuda.current_device())
        self.values = torch.arange(_VALUES, dtype=torch.int32, device=dev)  # flag source of call n: values[n]
        logger.info(f"TP relay for group {self.ranks} through {self.helpers}")

    def _next(self, kind: str) -> int:
        self.n[kind] += 1
        if self.n[kind] >= _VALUES:
            raise RuntimeError("TP relay call counter wrapped")
        return self.n[kind]

    def base(self, buf, rank: int) -> int:
        return buf.ptr if rank == self.w.rank else buf.peer_bases[self.w.peers.index(rank)]

    def _flags_layout(self, C: int):
        """Word offsets in a flag buffer: data [C][Wt][P], scales [Wt], stage [C][Wt], scratch."""
        data = 0
        scale = data + C * self.Wt * self.P
        stage = scale + self.Wt
        scratch = stage + C * self.Wt
        return data, scale, stage, scratch, scratch + 1

    def _plan(self, copies, flags):
        return self.T.Plan(self.w, copies, flags)

    @staticmethod
    def _free(entry):
        for pp in entry["par"]:
            plans = [*pp["chunks"], *(x for row in pp["fwd"] for x in row)]
            if "scale" in pp:
                plans.append(pp["scale"])
            for plan in plans:
                plan.free()

    # -- all-gather + GEMM ------------------------------------------------------------------------------
    def _ag(self, rows: int, K: int, C: int) -> dict:
        key = ("ag", rows, K, C)
        w, Wt, H, P = self.w, self.Wt, self.H, self.P
        block = rows // C
        cb = block * K  # bytes of a chunk (fp8)
        pieces = _pieces(cb, H, self.direct_frac)
        slot = max(s for _, s in pieces[1:])
        xh, sh, sth = Wt * rows * K, Wt * rows * 4, Wt * Wt * C * slot  # one parity's bytes
        bufs = {
            "x": w.buffer("tp_relay_ag_x", 2 * xh),
            "s": w.buffer("tp_relay_ag_s", 2 * sh),
            "stage": w.buffer("tp_relay_ag_stage", 2 * sth),  # [parity][client sender][client dest][c]
            "flags": w.buffer("tp_relay_ag_flags", self._flags_layout(C)[-1] * 4),
        }
        cached = self._plans.get(key)
        if cached is not None and all(cached["bufs"][k] is b and b.live for k, b in bufs.items()):
            return cached
        data, scale, stage, scratch, _ = self._flags_layout(C)
        me, ranks, helpers, peers = self.me, self.ranks, self.helpers, w.peers
        f = bufs["flags"]

        def flag(rank, word):
            return self.base(f, rank) + word * 4

        par_plans = []
        for par in range(2):

            def x_at(rank, slot_rank, c, off, par=par):
                return self.base(bufs["x"], rank) + par * xh + slot_rank * rows * K + c * cb + off

            def stage_at(helper_rank, sender, dest, c, par=par):
                return self.base(bufs["stage"], helper_rank) + par * sth + ((sender * Wt + dest) * C + c) * slot

            copies, flags = [], []  # row scales: to every TP peer, direct
            for q in peers:
                if q in ranks:
                    copies.append([(0, self.base(bufs["s"], q) + par * sh + me * rows * 4, rows * 4)])
                    flags.append(flag(q, scale + me))
                else:
                    copies.append([])
                    flags.append(flag(q, scratch))
            scale_plan = self._plan(copies, flags)
            chunk_plans = []  # chunk c: the direct piece to each TP peer, every peer's helper piece to each helper
            for c in range(C):
                copies, flags = [], []
                for q in peers:
                    if q in ranks:
                        off, size = pieces[0]
                        copies.append([(c * cb + off, x_at(q, me, c, off), size)])
                        flags.append(flag(q, data + (c * Wt + me) * P))
                    else:
                        off, size = pieces[1 + helpers.index(q)]
                        copies.append([(c * cb + off, stage_at(q, me, d, c), size) for d in range(Wt) if d != me])
                        flags.append(flag(q, stage + c * Wt + me))
                chunk_plans.append(self._plan(copies, flags))
            # as helper `me` of the other group: forward client sender i's pieces of chunk c to each client dest
            off, size = pieces[1 + me]
            fwd = []
            for c in range(C):
                row = []
                for i in range(Wt):
                    copies, flags = [], []
                    for q in peers:
                        if q in helpers and helpers.index(q) != i:
                            d = helpers.index(q)
                            src = stage_at(w.rank, i, d, c) - bufs["stage"].ptr
                            copies.append([(src, x_at(q, i, c, off), size)])
                            flags.append(flag(q, data + (c * Wt + i) * P + 1 + me))
                        else:
                            copies.append([])
                            flags.append(flag(q, scratch))
                    row.append(self._plan(copies, flags))
                fwd.append(row)
            par_plans.append({"scale": scale_plan, "chunks": chunk_plans, "fwd": fwd})
        if cached is not None:
            self._free(cached)
        dev = self.values.device
        entry = {
            "par": par_plans,
            "bufs": bufs,
            "halves": (xh, sh, sth),
            # per-shape index lists (a per-call host->device copy would stall the host)
            "scale_idx": torch.tensor([scale + i for i in range(Wt) if i != me], dtype=torch.int32, device=dev),
            "stage_idx": [
                torch.tensor([stage + c * Wt + i], dtype=torch.int32, device=dev) for c in range(C) for i in range(Wt)
            ],
            "own_words": torch.tensor(
                [data + (c * Wt + me) * P + k for c in range(C) for k in range(P)] + [scale + me],
                dtype=torch.long,
                device=dev,
            ),
        }
        self._plans[key] = entry
        return entry

    def all_gather_gemm_fp8(self, xq, xs, w_t, w_s, chunks: int) -> torch.Tensor:
        """xq [rows, K] fp8, xs [rows, 1] fp32, w_t [K, N] column-major, w_s [1, N] -> [Wt * rows, N] bf16."""
        import flydsl.expr as fx
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

        from xfuser.model_executor.layers.flydsl_tp_gemm import compile_persistent_ag_gemm_fp8

        rows, K = xq.shape
        N = w_t.shape[1]
        C, Wt, P = chunks, self.Wt, self.P
        e = self._ag(rows, K, C)
        n = self._next("ag")
        par = n % 2
        pp = e["par"][par]
        xh, sh, sth = e["halves"]
        data, _, _, _, nwords = self._flags_layout(C)
        dev = xq.device
        cur = torch.cuda.current_stream()
        side = _side.setdefault(dev.index, torch.cuda.Stream(device=dev, priority=-1))
        flags = e["bufs"]["flags"].local.view(torch.int32)[:nwords]
        x_in = e["bufs"]["x"].local[par * xh : (par + 1) * xh].view(Wt, rows * K)
        s_in = e["bufs"]["s"].local[par * sh : (par + 1) * sh].view(torch.float32).view(Wt, rows)
        stage_all = e["bufs"]["stage"].local  # forward plans hold offsets from the buffer start (parity included)
        flag_src = self.values[n : n + 1].data_ptr()
        xq = xq.contiguous()
        xs = xs.reshape(rows).float().contiguous()
        # own slot and own flag words for this call (no reset: peers' words only ever grow)
        x_in[self.me].copy_(xq.view(torch.uint8).view(-1))
        s_in[self.me].copy_(xs)
        flags.index_fill_(0, e["own_words"], n)
        call = self.w.begin()
        call.send(pp["scale"], xs, cur, flag_src)
        for c in range(C):
            call.send(pp["chunks"][c], xq, cur, flag_src)
        wait = _wait_kernel()
        # helper side: forward each client sender's piece once it has landed in this rank's staging
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            for c in range(C):
                for i in range(Wt):
                    wait[(1,)](flags, e["stage_idx"][c * Wt + i], 1, n)
                    call.send(pp["fwd"][c][i], stage_all, side, flag_src)
        out = torch.empty(Wt * rows, N, dtype=torch.bfloat16, device=dev)
        wait[(1,)](flags, e["scale_idx"], e["scale_idx"].numel(), n)
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
            e["bufs"]["flags"].ptr,
            n,
            self.me,
            Wt,
            rows,
            rows // C,
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
        xq.record_stream(side)
        calls["all_gather_gemm"] += 1
        return out

    # -- GEMM + reduce-scatter ------------------------------------------------------------------------------
    def _rs(self, rows: int, N: int, C: int) -> dict:
        key = ("rs", rows, N, C)
        w, Wt, H, P = self.w, self.Wt, self.H, self.P
        block = rows // C
        cb = block * N * 2  # bytes of a chunk of bf16 partial sums
        pieces = _pieces(cb, H, self.direct_frac)
        slot = max(s for _, s in pieces[1:])
        rh, sth = Wt * C * cb, Wt * Wt * C * slot
        bufs = {
            "recv": w.buffer("tp_relay_rs_recv", 2 * rh),  # [parity][sender][c] at each destination
            "stage": w.buffer("tp_relay_rs_stage", 2 * sth),
            "flags": w.buffer("tp_relay_rs_flags", self._flags_layout(C)[-1] * 4),
        }
        cached = self._plans.get(key)
        if cached is not None and all(cached["bufs"][k] is b and b.live for k, b in bufs.items()):
            return cached
        data, _, stage, scratch, _ = self._flags_layout(C)
        me, ranks, helpers, peers = self.me, self.ranks, self.helpers, w.peers
        f = bufs["flags"]

        def flag(rank, word):
            return self.base(f, rank) + word * 4

        def src(dest, c):  # this rank's partials for dest's chunk c: rows dest * rows + c * block of [S, N]
            return (dest * rows + c * block) * N * 2

        par_plans = []
        for par in range(2):

            def recv_at(rank, sender, c, off, par=par):
                return self.base(bufs["recv"], rank) + par * rh + (sender * C + c) * cb + off

            def stage_at(helper_rank, sender, dest, c, par=par):
                return self.base(bufs["stage"], helper_rank) + par * sth + ((sender * Wt + dest) * C + c) * slot

            chunk_plans = []
            for c in range(C):
                copies, flags = [], []
                for q in peers:
                    if q in ranks:
                        d = ranks.index(q)
                        off, size = pieces[0]
                        copies.append([(src(d, c) + off, recv_at(q, me, c, off), size)])
                        flags.append(flag(q, data + (c * Wt + me) * P))
                    else:
                        off, size = pieces[1 + helpers.index(q)]
                        copies.append([(src(d, c) + off, stage_at(q, me, d, c), size) for d in range(Wt) if d != me])
                        flags.append(flag(q, stage + c * Wt + me))
                chunk_plans.append(self._plan(copies, flags))
            off, size = pieces[1 + me]
            fwd = []
            for c in range(C):
                row = []
                for i in range(Wt):
                    copies, flags = [], []
                    for q in peers:
                        if q in helpers and helpers.index(q) != i:
                            d = helpers.index(q)
                            s_off = stage_at(w.rank, i, d, c) - bufs["stage"].ptr
                            copies.append([(s_off, recv_at(q, i, c, off), size)])
                            flags.append(flag(q, data + (c * Wt + i) * P + 1 + me))
                        else:
                            copies.append([])
                            flags.append(flag(q, scratch))
                    row.append(self._plan(copies, flags))
                fwd.append(row)
            par_plans.append({"chunks": chunk_plans, "fwd": fwd})
        if cached is not None:
            self._free(cached)
        dev = self.values.device
        peers_of = [i for i in range(Wt) if i != me]
        entry = {
            "par": par_plans,
            "bufs": bufs,
            "halves": (rh, sth),
            "tiles": (block // BLOCK_M) * ((N + 255) // 256),  # GEMM tiles per (chunk, dest)
            "cnt_idx": [torch.tensor([c * Wt + p for p in peers_of], dtype=torch.int32, device=dev) for c in range(C)],
            "stage_idx": [
                torch.tensor([stage + c * Wt + i], dtype=torch.int32, device=dev) for c in range(C) for i in range(Wt)
            ],
            "data_idx": [
                torch.tensor(
                    [data + (c * Wt + s) * P + k for s in peers_of for k in range(P)], dtype=torch.int32, device=dev
                )
                for c in range(C)
            ],
            "cnt": torch.zeros(C * Wt, dtype=torch.int32, device=dev),  # tile counters [c][dest], zeroed per call
        }
        self._plans[key] = entry
        return entry

    def gemm_reduce_scatter_fp8(self, xq, xs, w_t, w_s, chunks: int) -> torch.Tensor:
        """xq [S, K] fp8 (all rows of the group, this rank's K shard), xs [S, 1], w_t [K, N] column-major,
        w_s [1, N] -> this rank's [S / Wt, N] bf16, partial sums relayed through the other group."""
        import flydsl.expr as fx
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

        from xfuser.model_executor.layers.flydsl_tp_gemm import compile_persistent_ag_gemm_fp8

        S, K = xq.shape
        N = w_t.shape[1]
        C, Wt = chunks, self.Wt
        rows = S // Wt
        block = rows // C
        e = self._rs(rows, N, C)
        n = self._next("rs")
        par = n % 2
        pp = e["par"][par]
        rh, sth = e["halves"]
        nwords = self._flags_layout(C)[-1]
        dev = xq.device
        cur = torch.cuda.current_stream()
        side = _side.setdefault(dev.index, torch.cuda.Stream(device=dev, priority=-1))
        flags = e["bufs"]["flags"].local.view(torch.int32)[:nwords]
        recv = e["bufs"]["recv"].local[par * rh : (par + 1) * rh].view(torch.bfloat16).view(Wt, C, block, N)
        stage_all = e["bufs"]["stage"].local  # forward plans hold offsets from the buffer start (parity included)
        flag_src = self.values[n : n + 1].data_ptr()
        cnt = e["cnt"]
        partials = torch.empty(S, N, dtype=torch.bfloat16, device=dev)
        out = torch.empty(rows, N, dtype=torch.bfloat16, device=dev)
        cnt.zero_()
        call = self.w.begin()
        post = _post.setdefault(dev.index, torch.cuda.Stream(device=dev, priority=-1))
        post.wait_stream(cur)  # counters zeroed before the poster reads them
        side.wait_stream(cur)
        wait = _wait_kernel()
        # Each engine runs its copies in issue (host) order, so issue in expected-ready order: this rank's chunk
        # c (poster stream, once its tiles are counted), then the other group's staged pieces of chunk c (forwarder
        # stream). Separate streams: a forwarder's wait never holds back the next post's release, nor vice versa.
        for c in range(C):
            with torch.cuda.stream(post):
                wait[(1,)](cnt, e["cnt_idx"][c], e["cnt_idx"][c].numel(), e["tiles"])
                call.send(pp["chunks"][c], partials, post, flag_src)
            with torch.cuda.stream(side):
                for i in range(Wt):
                    wait[(1,)](flags, e["stage_idx"][c * Wt + i], 1, n)
                    call.send(pp["fwd"][c][i], stage_all, side, flag_src)
        exe = compile_persistent_ag_gemm_fp8(K=K, BLOCK_M=BLOCK_M, BLOCK_N=256, mode="rs")
        progs = torch.cuda.get_device_properties(dev).multi_processor_count * 7 // 8
        _run_compiled(
            exe,
            xq.contiguous().view(torch.int8).view(-1)[: rows * K],
            w_t.t().view(torch.int8).reshape(-1),
            partials.view(-1)[: rows * N],
            xs.reshape(-1).float().contiguous()[:rows],
            w_s.reshape(-1).float().contiguous(),
            cnt.data_ptr(),
            1,
            self.me,
            Wt,
            rows,
            block,
            rows,
            rows,
            0,
            1,
            N,
            progs,
            fx.Stream(cur),
        )
        own = partials.view(Wt, C, block * N)[self.me]
        recv_flat = recv.view(Wt, C, block * N)
        reduce = _reduce_kernel()
        RB = 4096
        nel = block * N
        for c in range(C):
            wait[(1,)](flags, e["data_idx"][c], e["data_idx"][c].numel(), n)
            reduce[((nel + RB - 1) // RB,)](
                recv_flat[0, c], own[c], out[c * block :], C * nel, self.me, nel, W=Wt, BLOCK=RB
            )
        cur.wait_stream(post)
        cur.wait_stream(side)
        call.end()
        partials.record_stream(post)
        calls["gemm_reduce_scatter"] += 1
        return out


def relay_for(group_name: str) -> Optional[TPRelay]:
    """The relay of TP group `group_name` (collective over the world on first use), or None if it does not apply."""
    if not enabled():
        return None
    r = _relays.get(group_name)
    if r is None:
        try:
            r = TPRelay(group_name)
        except ValueError as err:
            logger.warning(f"TP relay off: {err}")
            r = False
        _relays[group_name] = r
    return r or None
