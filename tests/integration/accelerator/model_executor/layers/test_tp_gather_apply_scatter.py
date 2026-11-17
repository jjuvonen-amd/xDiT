"""tp_gather_apply_scatter on torch's patched ROCm SDMA path against RCCL all-gather -> fn -> reduce-scatter."""

import pytest

pytestmark = [pytest.mark.multi_gpu, pytest.mark.rocm]


def _worker(rank, world_size, init_method):
    import os

    import torch
    import torch.distributed as dist

    os.environ["TORCH_SYMM_MEM_ENABLE_ROCM_SDMA"] = "1"
    os.environ["XDIT_TP_IPC_WORKSPACE"] = "1"
    pytest.importorskip("torch.distributed._symmetric_memory._rocm_sdma")
    from xfuser.core.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        ipc_workspace,
    )
    from xfuser.core.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
    )
    from xfuser.model_executor.layers.tensor_parallel import tp_gather_apply_scatter

    try:
        torch.cuda.set_device(rank)
        init_distributed_environment(
            rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method
        )
        initialize_model_parallel(tensor_parallel_degree=world_size)
        assert ipc_workspace.install_torch_sdma_inbox()
        tp = get_tp_group()
        n = 256
        weight = torch.randn(n, n, device="cuda", dtype=torch.bfloat16) / 16
        dist.broadcast(weight, 0)

        def fn(*tensors):
            # every input reaches the output: the first tensor's leading features plus each other tensor's row sums
            acc = tensors[0][:, :n].float()
            for t in tensors[1:]:
                acc = acc + t.float().sum(-1, keepdim=True) * 1e-3
            return acc.to(torch.bfloat16) @ weight

        gen = torch.Generator(device="cuda").manual_seed(rank)
        rows = 512
        cases = [
            [torch.randn(rows, 384, device="cuda", dtype=torch.bfloat16, generator=gen)],
            # mixed dtypes, as quantized activations + scales + routing travel: several copies per peer per piece
            [
                torch.randint(0, 255, (rows, 320), device="cuda", dtype=torch.uint8, generator=gen),
                torch.randint(0, 1 << 10, (rows, 16), device="cuda", dtype=torch.int32, generator=gen),
                torch.randint(0, 64, (rows, 4), device="cuda", dtype=torch.int32, generator=gen),
                torch.rand(rows, 4, device="cuda", generator=gen),
            ],
        ]
        for tensors in cases + cases:  # each case twice in a row, and switching between them
            for chunks in (4, 1):
                with torch.inference_mode():
                    out = tp_gather_apply_scatter(tensors, fn, n, torch.bfloat16, "test", chunks=chunks)
                    full = [tp.all_gather(t, dim=0) for t in tensors]
                    ref = tp.reduce_scatter(fn(*full).float(), dim=0).to(torch.bfloat16)
                error = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                assert error < 1e-2, f"{[t.dtype for t in tensors]} chunks {chunks}: relative error {error:.3e}"
        dist.barrier()
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def test_gather_apply_scatter_matches_rccl(accelerator_ranks):
    accelerator_ranks(_worker, world_size=4, timeout=600)
