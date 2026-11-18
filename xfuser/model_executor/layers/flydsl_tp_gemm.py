"""Persistent FP8 row-wise GEMMs for tensor parallelism, in FlyDSL (gfx950).

A fork of AITER's 8-wave FP8 GEMM (aiter/ops/flydsl/kernels/gemm_a8w8_8wave.py: 2x2 waves of 4x2
16x16x128 scaled MFMA tiles, global->LDS DMA double buffering), persistent: one workgroup per CU,
each looping over output tiles. Two modes, used by layers/tp_persistent.py on the SDMA transport of
torch's patched ROCm symm_mem ops:

- "ag" (all-gather + GEMM): tiles in readiness order (this rank's row blocks first, then chunk by
  chunk over the peers); before a peer tile thread 0 waits (system scope, acquire) until every
  piece flag of that (rank, chunk) reached `val`. A rows and A row scales come from the gathered
  inbox (rank r's rows at r * a_slot_rows / r * s_slot_rows; descriptors rebased per rank with
  64-bit offsets, so no 2 GiB window).
- "rs" (GEMM + reduce-scatter): A is this rank's full [S, K] input; tiles in send order (chunk-major
  over the peers me+1, me+2, ..., own rows last); after each finished peer tile thread 0 releases
  (agent scope) and adds 1 to counter word f0 + c * W + p, which a poster stream waits on before the
  SDMA push of (chunk c, peer p).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    wait_barrier,
)
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

BLOCK_K = 128


def _lds_barrier():
    """s_barrier that first drains this wave's LDS reads, so no other wave's global->LDS DMA overwrites a buffer this
    wave is still reading (race fixed in the SDXL forks of the same pipeline)."""
    from flydsl._mlir.dialects import llvm as _llvm

    _llvm.inline_asm(
        res=None, operands_=[], asm_string="s_waitcnt lgkmcnt(0)\ns_barrier", constraints="", has_side_effects=True
    )


def _rank_buffer(t, elem_off, nbytes, recast_ir_t=None):
    """Buffer tensor whose descriptor starts at element `elem_off` (64-bit) of `t` and covers `nbytes`: every tile
    reads one rank's slot and writes one rank's output rows, so in-tile offsets stay 32-bit for any S."""
    ptr = fx.get_iter(t) + fx.Int64(elem_off)
    bp = fx.rocdl.make_buffer_ptr(ptr, num_records_bytes=nbytes)
    if recast_ir_t is not None:
        bp = fx.recast_iter(
            fx.PointerType.get(
                elem_ty=recast_ir_t,
                address_space=rocdl.TargetAddressSpace.BufferDesc,
                alignment=fx.PointerType(bp.type).alignment,
            ),
            bp,
        )
    return fx.Tensor(fx.make_view(bp, fx.get_layout(t)))


class _StoreC:
    """aiter's StoreC with the A-scale row decoupled from the output row."""

    def __init__(self, A_scale, B_scale, C, c_rows, c_cols, sa_rows, c_idx_fn, n_tiles_a, n_tiles_b, c_off, sa_off):
        self.c_rows, self.c_cols = c_rows, c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn, self.n_tiles_a, self.n_tiles_b = c_idx_fn, n_tiles_a, n_tiles_b
        gC = _rank_buffer(C, c_off, c_rows * c_cols * 2)  # this rank's output rows only
        gSA = _rank_buffer(A_scale, sa_off, sa_rows * 4)  # this rank's row scales only
        gSB = fx.rocdl.make_buffer_tensor(B_scale, max_size=False, num_records_bytes=c_cols * 4)
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.sa_div = fx.logical_divide(gSA, fx.make_layout(1, 1))
        self.sb_div = fx.logical_divide(gSB, fx.make_layout(1, 1))
        self.scale_atom_4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        self.scale_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        self.out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        self.reg_f32_4 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
        self.reg_f32_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        self.reg_bf16_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.BFloat16)

    def _sa4(self, row):
        fx.copy(self.scale_atom_4, fx.slice(self.sa_div, (None, fx.Int32(row))), self.reg_f32_4)
        return Vec(fx.memref_load_vec(self.reg_f32_4))

    def _sb1(self, col):
        fx.copy(self.scale_atom_1, fx.slice(self.sb_div, (None, fx.Int32(col))), self.reg_f32_1)
        return Vec(fx.memref_load_vec(self.reg_f32_1))[0]

    def _st(self, v, idx):
        fx.memref_store_vec(Vec.filled(1, v, fx.BFloat16), self.reg_bf16_1)
        fx.copy(self.out_atom_1, self.reg_bf16_1, fx.slice(self.c_div, (None, fx.Int32(idx))))

    def store(self, frag, c_row, s_row, col0):
        sa = [self._sa4(s_row + i * 16 + (self.lane_id // 16) * 4) for i in range_constexpr(self.n_tiles_a)]
        sb = [self._sb1(col0 + i * 16 + self.lane_id % 16) for i in range_constexpr(self.n_tiles_b)]
        for ti in range_constexpr(self.n_tiles_a):
            row = c_row + ti * 16 + (self.lane_id // 16) * 4
            for tj in range_constexpr(self.n_tiles_b):
                col = col0 + tj * 16 + self.lane_id % 16
                ok = col < self.c_cols
                oob = fx.Int32(self.c_rows * self.c_cols)
                v = Vec(frag[self.c_idx_fn(ti, tj)])
                for i in range_constexpr(4):
                    self._st(
                        (v[i] * (sa[ti][i] * sb[tj])).to(fx.BFloat16), ok.select((row + i) * self.c_cols + col, oob)
                    )


_CACHE = {}


def compile_persistent_ag_gemm_fp8(*, K, BLOCK_M=256, BLOCK_N=256, waves_per_eu=1, mode="ag"):
    """mode "ag": A rows of peer r come from the receive slots, tiles wait on the chunk's flags.
    mode "rs": A = my full [S, K] input (a_slot_rows = s_slot_rows = rows); tiles in send order (chunk-major over
    the peers me+1, me+2, ..., my own rows last); after each finished peer tile, thread 0 releases (agent) and adds
    1 to counter word f0 + c * W + p, which a poster stream waits on before the SDMA push of (chunk c, peer p)."""
    assert mode in ("ag", "rs")
    key = (K, BLOCK_M, BLOCK_N, waves_per_eu, mode)
    if key in _CACHE:
        return _CACHE[key]
    assert BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0 and K % BLOCK_K == 0
    K_ITERS = K // BLOCK_K
    N_TILES_A, N_TILES_B = BLOCK_M // 64, BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M, LDS_BLOCK_N = BLOCK_M // 2, BLOCK_N // 2
    N_LDS_STEPS_A, N_LDS_STEPS_B = LDS_BLOCK_M // 64, LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    a_lds, b_lds = LDS_BLOCK_M * BLOCK_K, LDS_BLOCK_N * BLOCK_K

    @fx.struct
    class SharedStorage:
        A0: fx.Array[fx.Float8E4M3FN, a_lds, 16]
        A1: fx.Array[fx.Float8E4M3FN, a_lds, 16]
        A2: fx.Array[fx.Float8E4M3FN, a_lds, 16]
        A3: fx.Array[fx.Float8E4M3FN, a_lds, 16]
        B0: fx.Array[fx.Float8E4M3FN, b_lds, 16]
        B1: fx.Array[fx.Float8E4M3FN, b_lds, 16]
        B2: fx.Array[fx.Float8E4M3FN, b_lds, 16]
        B3: fx.Array[fx.Float8E4M3FN, b_lds, 16]

    kname = f"persistent_{mode}_gemm_fp8_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_k{K}_w{waves_per_eu}"

    @flyc.kernel(name=kname, known_block_size=[512, 1, 1])
    def kernel(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        flags_addr: fx.Int64,
        val: fx.Int32,
        me: fx.Int32,
        W: fx.Int32,
        rows: fx.Int32,
        rows_chunk: fx.Int32,
        a_slot_rows: fx.Int32,
        s_slot_rows: fx.Int32,
        f0: fx.Int32,
        P: fx.Int32,
        c_n: fx.Int32,
        n_progs: fx.Int32,
    ):
        F8 = fx.Float8E4M3FN.ir_type
        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m, wave_n = wave_id // 4, wave_id % 4
        n_blocks_n = (c_n + BLOCK_N - 1) // BLOCK_N
        bpr = rows // BLOCK_M  # row blocks per rank
        bpc = rows_chunk // BLOCK_M  # row blocks per chunk
        peers = W - 1
        total = W * bpr * n_blocks_n
        for t in range(fx.Int32(fx.block_idx.x), total, n_progs):
            # trace-time helper objects live inside the loop body: FlyDSL carries outer names through scf.for
            mo = t // n_blocks_n
            block_n = t % n_blocks_n
            if const_expr(mode == "ag"):
                # readiness order -> (rank r, local row block lmb, chunk c)
                j = mo - bpr
                own = mo < bpr
                jj = own.select(fx.Int32(0), j)
                c = jj // (bpc * peers)
                q = jj % (bpc * peers)
                pi = q // bpc
                r = own.select(me, pi + (pi >= me).select(fx.Int32(1), fx.Int32(0)))
                lmb = own.select(mo, c * bpc + q % bpc)
            else:
                # send order -> (destination rank r, local row block lmb, chunk c); own rows last
                own = mo >= peers * bpr
                jj = own.select(fx.Int32(0), mo)
                c = jj // (bpc * peers)
                q = jj % (bpc * peers)
                pi = q // bpc
                r = own.select(me, (me + 1 + pi) % W)
                lmb = own.select(mo - peers * bpr, c * bpc + q % bpc)
            r64 = fx.Int64(r)
            gA = _rank_buffer(A, r64 * fx.Int64(a_slot_rows) * fx.Int64(K), fx.Int64(rows) * fx.Int64(K), F8)
            gB = make_fp8_buffer_tensor(B_T, F8)
            a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
            b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
            gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
            gl_off_b = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
            mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
            a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8, wave_id)
            b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8, wave_id)
            a_s2r = S2RLoader(wave_m, N_TILES_A)
            b_s2r = S2RLoader(wave_n, N_TILES_B)
            store_c = _StoreC(
                A_scale,
                B_scale,
                C,
                rows,
                c_n,
                rows,
                mfma.idx,
                N_TILES_A,
                N_TILES_B,
                r64 * fx.Int64(rows) * fx.Int64(c_n),
                r64 * fx.Int64(s_slot_rows),
            )
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            # wait for the chunk (all pieces) unless the rows are mine
            if const_expr(mode == "ag") and r != me:
                if fx.thread_idx.x == 0:
                    for k in range(fx.Int32(0), P, fx.Int32(1)):
                        idx = f0 + (c * W + r) * P + k
                        comm_ops.spin_until_ge_i32_system(flags_addr + fx.Int64(idx) * fx.Int64(4), val, acquire=True)
                    comm_ops.fence_system_acquire()
                _lds_barrier()

            a_cur0, a_cur1, a_next0, a_next1 = lds.A0, lds.A1, lds.A2, lds.A3
            b_cur0, b_cur1, b_next0, b_next1 = lds.B0, lds.B1, lds.B2, lds.B3
            a_row = lmb * BLOCK_M  # relative to rank r's slot
            A0_off = a_row * K
            A1_off = (a_row + LDS_BLOCK_M) * K
            B0_off = (block_n * BLOCK_N) * K
            B1_off = (block_n * BLOCK_N + LDS_BLOCK_N) * K
            c00 = [mfma.zero_value] * N_ACCUMS
            c01 = [mfma.zero_value] * N_ACCUMS
            c10 = [mfma.zero_value] * N_ACCUMS
            c11 = [mfma.zero_value] * N_ACCUMS

            b_g2s.load(b_cur0, B0_off)
            a_g2s.load(a_cur0, A0_off)
            b_g2s.load(b_cur1, B1_off)
            a_g2s.load(a_cur1, A1_off)
            if wave_m == 1:
                _lds_barrier()
            wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)
            b_g2s.load(b_next0, B0_off + BLOCK_K)
            a_g2s.load(a_next0, A0_off + BLOCK_K)
            b_g2s.load(b_next1, B1_off + BLOCK_K)
            wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)
            for k in range_constexpr(K_ITERS - 2):
                b0 = b_s2r.load(b_cur0)
                a0 = a_s2r.load(a_cur0)
                a_g2s.load(a_next1, A1_off + (k + 1) * BLOCK_K)
                _lds_barrier()
                c00 = mfma.call(a0, b0, c00)
                b1 = b_s2r.load(b_cur1)
                b_g2s.load(b_cur0, B0_off + (k + 2) * BLOCK_K)
                _lds_barrier()
                c01 = mfma.call(a0, b1, c01)
                a1 = a_s2r.load(a_cur1)
                a_g2s.load(a_cur0, A0_off + (k + 2) * BLOCK_K)
                _lds_barrier()
                c10 = mfma.call(a1, b0, c10)
                b_g2s.load(b_cur1, B1_off + (k + 2) * BLOCK_K)
                wait_barrier(min(2 * N_LDS_STEPS_A + N_LDS_STEPS_B, N_LDS_STEPS_A + 2 * N_LDS_STEPS_B))
                c11 = mfma.call(a1, b1, c11)
                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1
            b0 = b_s2r.load(b_cur0)
            a0 = a_s2r.load(a_cur0)
            _lds_barrier()
            c00 = mfma.call(a0, b0, c00)
            b1 = b_s2r.load(b_cur1)
            _lds_barrier()
            c01 = mfma.call(a0, b1, c01)
            a1 = a_s2r.load(a_cur1)
            a_g2s.load(a_next1, A1_off + (K_ITERS - 1) * BLOCK_K)
            _lds_barrier()
            c10 = mfma.call(a1, b0, c10)
            wait_barrier(0)  # tail: the last K tile's loads are not guaranteed to have landed
            b0 = b_s2r.load(b_next0)
            _lds_barrier()
            c11 = mfma.call(a1, b1, c11)
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1
            a0 = a_s2r.load(a_cur0)
            wait_barrier(0)
            c00 = mfma.call(a0, b0, c00)
            b1 = b_s2r.load(b_cur1)
            _lds_barrier()
            c01 = mfma.call(a0, b1, c01)
            a1 = a_s2r.load(a_cur1)
            _lds_barrier()
            rocdl.s_setprio(1)
            c10 = mfma.call(a1, b0, c10, set_prio=False)
            c11 = mfma.call(a1, b1, c11, set_prio=False)
            rocdl.s_setprio(0)
            _lds_barrier()

            wn = wave_n * (N_TILES_B * 16)
            wm = wave_m * (N_TILES_A * 16)
            c_row = lmb * BLOCK_M + wm  # relative to rank r's output rows / scales
            s_row = lmb * BLOCK_M + wm
            col0 = block_n * BLOCK_N + wn
            store_c.store(c00, c_row, s_row, col0)
            store_c.store(c01, c_row, s_row, col0 + LDS_BLOCK_N)
            store_c.store(c10, c_row + LDS_BLOCK_M, s_row + LDS_BLOCK_M, col0)
            store_c.store(c11, c_row + LDS_BLOCK_M, s_row + LDS_BLOCK_M, col0 + LDS_BLOCK_N)
            # the tile started with an extra barrier for wave_m == 1 (ping-pong offset): re-align the two wave
            # groups, then drain this tile's C stores (vmcnt also counts stores on CDNA) before the next tile
            if wave_m == 0:
                _lds_barrier()
            wait_barrier(0)
            if const_expr(mode == "rs"):
                # every wave's stores of this tile are done (vmcnt(0) + barrier): publish the tile
                if r != me:
                    if fx.thread_idx.x == 0:
                        comm_ops.fence_agent_release()
                        comm_ops.atomic_add_system(flags_addr + fx.Int64(f0 + c * W + r) * fx.Int64(4), fx.Int32(1))

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        flags_addr: fx.Int64,
        val: fx.Int32,
        me: fx.Int32,
        W: fx.Int32,
        rows: fx.Int32,
        rows_chunk: fx.Int32,
        a_slot_rows: fx.Int32,
        s_slot_rows: fx.Int32,
        f0: fx.Int32,
        P: fx.Int32,
        c_n: fx.Int32,
        n_progs: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            flags_addr,
            val,
            me,
            W,
            rows,
            rows_chunk,
            a_slot_rows,
            s_slot_rows,
            f0,
            P,
            c_n,
            n_progs,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu, "rocdl.flat_work_group_size": "512,512"},
        ).launch(grid=(n_progs, 1, 1), block=(512, 1, 1), stream=stream)

    _CACHE[key] = launch
    return launch
