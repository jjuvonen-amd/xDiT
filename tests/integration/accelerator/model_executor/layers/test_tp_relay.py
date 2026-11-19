"""The relayed FP8 TP all-gather + GEMM and GEMM + reduce-scatter (XDIT_TP_RELAY=1, two TP4 groups on 8 GPUs)
against the direct FP8 paths."""

import pytest

pytestmark = [pytest.mark.multi_gpu, pytest.mark.rocm]


def _worker(rank, world_size, init_method):
    import os

    import torch
    import torch.distributed as dist

    # bf16 partial sums first (tight check); the FP8 ones are checked per shape below
    os.environ.update(TORCH_SYMM_MEM_ENABLE_ROCM_SDMA="1", XDIT_TP_IPC_WORKSPACE="1", XDIT_TP_RELAY_RS_FP8="0")
    pytest.importorskip("torch.distributed._symmetric_memory._rocm_sdma")
    pytest.importorskip("flydsl")
    from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel, ipc_workspace
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.model_executor.layers import tp_relay
    from xfuser.core.distributed.parallel_state import get_tp_group
    from xfuser.model_executor.layers.tensor_parallel import (
        _rowwise_fp8,
        quantize_linear_fp8_,
        tp_column_parallel,
        tp_row_parallel,
    )

    try:
        torch.cuda.set_device(rank)
        init_distributed_environment(
            rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method
        )
        initialize_model_parallel(tensor_parallel_degree=4, ulysses_degree=2)
        assert ipc_workspace.install_torch_sdma_inbox()
        gen = torch.Generator(device="cuda").manual_seed(0)
        # repeated calls, a larger shape (plans rebuilt, buffers grown), the first shape again, then 9 tiles per rank
        # (no equal split into 4 pieces: 2 + 2 + 2 + 3 tiles) and 37 tiles (Wan 720p's 9472 rows: 9 + 9 + 9 + 10)
        shapes = ((1024, 1024, 768), (1024, 1024, 768), (2048, 2048, 1280), (1024, 1024, 768), (2304, 1024, 768))
        for rows, k, n in shapes + ((9472, 1280, 1024),):
            linear = torch.nn.Linear(k, n, device="cuda", dtype=torch.bfloat16)
            torch.nn.init.normal_(linear.weight, std=k**-0.5, generator=gen)
            dist.broadcast(linear.weight.data, 0)
            dist.broadcast(linear.bias.data, 0)
            quantize_linear_fp8_(linear)
            x = torch.randn(1, rows, k, device="cuda", dtype=torch.bfloat16, generator=gen)
            with torch.inference_mode():
                os.environ["XDIT_TP_RELAY"] = "0"
                ref = tp_column_parallel(x, linear, "test", 1, "symm_mem")
                os.environ["XDIT_TP_RELAY"] = "1"
                before = tp_relay.calls["all_gather_gemm"]
                out = tp_column_parallel(x, linear, "test", 1, "symm_mem")
                assert tp_relay.calls["all_gather_gemm"] == before + 1, "the relayed path did not run"
            error = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            assert error < 1e-2, f"rows {rows} k {k} n {n}: relative error {error:.3e} vs the direct FP8 path"
            # GEMM + reduce-scatter: all of the group's rows, this rank's K shard
            row_linear = torch.nn.Linear(k, n, bias=False, device="cuda", dtype=torch.bfloat16)
            torch.nn.init.normal_(row_linear.weight, std=k**-0.5, generator=gen)
            quantize_linear_fp8_(row_linear)
            xr = torch.randn(1, 4 * rows, k, device="cuda", dtype=torch.bfloat16, generator=gen)
            with torch.inference_mode():
                os.environ["XDIT_TP_RELAY"] = "0"
                ref = tp_row_parallel(xr, row_linear, None, "test", 1, "symm_mem")
                os.environ["XDIT_TP_RELAY"] = "1"
                before = tp_relay.calls["gemm_reduce_scatter"]
                out = tp_row_parallel(xr, row_linear, None, "test", 1, "symm_mem")
                assert tp_relay.calls["gemm_reduce_scatter"] == before + 1, "the relayed RS path did not run"
            error = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            assert error < 1e-2, f"RS rows {rows} k {k} n {n}: relative error {error:.3e} vs the direct FP8 path"
            # the same with the peers' partial sums sent as FP8 (XDIT_TP_RELAY_RS_FP8): e4m3 rounding of 3 of 4 parts
            relays = [r for r in tp_relay._relays.values() if r]
            for r in relays:
                r.rs_fp8 = True
            with torch.inference_mode():
                out8 = tp_row_parallel(xr, row_linear, None, "test", 1, "symm_mem")
            for r in relays:
                r.rs_fp8 = False
            error8 = ((out8.float() - ref.float()).norm() / ref.float().norm()).item()
            if rank == 0:
                print(f"RS rows {rows} n {n}: bf16 partials {error:.2e}, fp8 partials {error8:.2e}", flush=True)
            assert error8 < 4e-2, f"RS rows {rows} k {k} n {n}: FP8 partials relative error {error8:.3e}"
            # GEMM + RS pipelined into the next AG + GEMM, piece by piece, through a row-wise function: bitwise the
            # same as the two ops one after the other with the function in between
            relay = tp_relay.relay_for(get_tp_group().device_group.group_name)
            if relay is not None and rows % 256 == 0:
                nxt = torch.nn.Linear(n, 512, bias=False, device="cuda", dtype=torch.bfloat16)
                torch.nn.init.normal_(nxt.weight, std=n**-0.5, generator=gen)
                quantize_linear_fp8_(nxt)
                xq, xs = _rowwise_fp8(xr[0])
                bias = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=gen)

                def then(r0, y):
                    return (y.float() * 0.5 + bias.float()).to(torch.bfloat16)

                w_t, w_s, w2_t, w2_s = (
                    row_linear.tp_fp8_weight,
                    row_linear.tp_fp8_scale,
                    nxt.tp_fp8_weight,
                    nxt.tp_fp8_scale,
                )
                chunks = relay.chunks(rows)
                with torch.inference_mode():
                    red = relay.gemm_reduce_scatter_fp8(xq, xs, w_t, w_s, chunks).clone()
                    q2, s2 = _rowwise_fp8(then(0, red))
                    seq_out = relay.all_gather_gemm_fp8(q2, s2, w2_t, w2_s, chunks)
                    red_f, fused_out = relay.gemm_rs_then_ag_fp8(xq, xs, w_t, w_s, then, w2_t, w2_s, chunks)
                assert torch.equal(red_f, red), f"rows {rows}: fused RS rows differ"
                assert torch.equal(fused_out, seq_out), f"rows {rows}: fused RS -> AG output differs"
        dist.barrier()
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def test_relayed_fp8_tp_ops_match_direct(accelerator_ranks):
    accelerator_ranks(_worker, world_size=8, timeout=900)
