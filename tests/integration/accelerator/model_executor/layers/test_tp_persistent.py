"""Persistent bf16 TP all-gather + GEMM (XDIT_TP_PERSISTENT=1) against RCCL all-gather + linear."""

import pytest

pytestmark = [pytest.mark.multi_gpu, pytest.mark.rocm]


def _worker(rank, world_size, init_method):
    import os

    import torch
    import torch.distributed as dist

    os.environ.update(TORCH_SYMM_MEM_ENABLE_ROCM_SDMA="1", XDIT_TP_IPC_WORKSPACE="1", XDIT_TP_PERSISTENT="1")
    pytest.importorskip("torch.distributed._symmetric_memory._rocm_sdma")
    pytest.importorskip("flydsl")
    from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel, ipc_workspace
    from xfuser.core.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
    )
    from xfuser.model_executor.layers import tp_persistent
    from xfuser.model_executor.layers.tensor_parallel import tp_column_parallel

    try:
        torch.cuda.set_device(rank)
        init_distributed_environment(
            rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method
        )
        initialize_model_parallel(tensor_parallel_degree=world_size)
        assert ipc_workspace.install_torch_sdma_inbox()
        gen = torch.Generator(device="cuda").manual_seed(0)
        # rows per rank that are and are not whole 256-row tiles (tiles then straddle ranks); batch 1 and 2
        for batch, rows, k, n in ((1, 1024, 1024, 768), (1, 1000, 512, 384), (2, 512, 1024, 512)):
            linear = torch.nn.Linear(k, n, device="cuda", dtype=torch.bfloat16)
            torch.nn.init.normal_(linear.weight, std=k**-0.5, generator=gen)
            dist.broadcast(linear.weight.data, 0)
            dist.broadcast(linear.bias.data, 0)
            x = torch.randn(batch, rows, k, device="cuda", dtype=torch.bfloat16, generator=gen)
            before = tp_persistent.calls["all_gather_gemm_bf16"]
            with torch.inference_mode():
                out = tp_column_parallel(x, linear, "test", 1, "symm_mem")
                ref = linear(get_tp_group().all_gather(x, dim=1))
            assert tp_persistent.calls["all_gather_gemm_bf16"] == before + 1, "the persistent kernel did not run"
            error = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            assert error < 1e-2, f"batch {batch} rows {rows}: relative error {error:.3e}"
        dist.barrier()
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def test_persistent_bf16_all_gather_gemm_matches_rccl(accelerator_ranks):
    accelerator_ranks(_worker, world_size=4, timeout=600)
