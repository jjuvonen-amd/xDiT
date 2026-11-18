"""The relayed FP8 TP all-gather + GEMM and GEMM + reduce-scatter (XDIT_TP_RELAY=1, two TP4 groups on 8 GPUs)
against the direct FP8 paths."""

import pytest

pytestmark = [pytest.mark.multi_gpu, pytest.mark.rocm]


def _worker(rank, world_size, init_method):
    import os

    import torch
    import torch.distributed as dist

    os.environ.update(TORCH_SYMM_MEM_ENABLE_ROCM_SDMA="1", XDIT_TP_IPC_WORKSPACE="1")
    pytest.importorskip("torch.distributed._symmetric_memory._rocm_sdma")
    pytest.importorskip("flydsl")
    from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel, ipc_workspace
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.model_executor.layers import tp_relay
    from xfuser.model_executor.layers.tensor_parallel import quantize_linear_fp8_, tp_column_parallel, tp_row_parallel

    try:
        torch.cuda.set_device(rank)
        init_distributed_environment(
            rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method
        )
        initialize_model_parallel(tensor_parallel_degree=4, ulysses_degree=2)
        assert ipc_workspace.install_torch_sdma_inbox()
        gen = torch.Generator(device="cuda").manual_seed(0)
        # repeated calls, a larger shape (plans rebuilt, buffers grown), then the first shape again
        for rows, k, n in ((1024, 1024, 768), (1024, 1024, 768), (2048, 2048, 1280), (1024, 1024, 768)):
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
        dist.barrier()
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def test_relayed_fp8_tp_ops_match_direct(accelerator_ranks):
    accelerator_ranks(_worker, world_size=8, timeout=900)
