"""The hipIpc workspace: every rank's buffer readable and writable from every rank, growth, and
torch's ROCm SDMA ops on it."""

import pytest

pytestmark = [pytest.mark.multi_gpu, pytest.mark.rocm]


def _workspace_worker(rank, world_size, init_method):
    import torch
    import torch.distributed as dist

    from xfuser.core.distributed import ipc_workspace

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world_size)
    try:
        group_name = dist.group.WORLD.group_name
        for size in (1 << 20, 3 << 20, 64 << 20, 2 << 20):  # growth, then reuse of the larger buffer
            ws = ipc_workspace.get_ipc_workspace(group_name, size)
            assert ws.size >= size
            n = size // 4
            ws.get_buffer(rank, (n,), torch.int32).copy_(torch.arange(n, device="cuda", dtype=torch.int32) + rank * n)
            ws.barrier()
            # Write into the next rank's buffer at an offset, read every rank's buffer.
            nxt = (rank + 1) % world_size
            ws.get_buffer(nxt, (16,), torch.int32, storage_offset=n - 16).fill_(-1 - rank)
            ws.barrier()
            for r in range(world_size):
                got = ws.get_buffer(r, (n - 16,), torch.int32)
                assert torch.equal(got, torch.arange(n - 16, device="cuda", dtype=torch.int32) + r * n)
            mine = ws.get_buffer(rank, (16,), torch.int32, storage_offset=n - 16)
            assert bool((mine == -1 - (rank - 1) % world_size).all())
            ws.barrier()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_ipc_workspace(accelerator_ranks, world_size):
    accelerator_ranks(_workspace_worker, world_size=world_size, timeout=300)


def _torch_sdma_worker(rank, world_size, init_method):
    import os

    import torch
    import torch.distributed as dist

    os.environ["TORCH_SYMM_MEM_ENABLE_ROCM_SDMA"] = "1"
    os.environ["XDIT_TP_IPC_WORKSPACE"] = "1"
    rocm_sdma = pytest.importorskip("torch.distributed._symmetric_memory._rocm_sdma")
    import torch.distributed._symmetric_memory as symm_mem

    from xfuser.core.distributed import ipc_workspace

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world_size)
    try:
        assert ipc_workspace.install_torch_sdma_inbox()

        def forbidden(*args, **kwargs):
            raise AssertionError("symmetric-memory rendezvous used")

        symm_mem.get_symm_mem_workspace = forbidden
        symm_mem.rendezvous = forbidden
        group_name = dist.group.WORLD.group_name
        gen = torch.Generator(device="cuda").manual_seed(rank)
        for rows in (256, 1024, 1024):
            a = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16, generator=gen)
            w = torch.randn(512, 768, device="cuda", dtype=torch.bfloat16, generator=gen)
            dist.broadcast(w, 0)
            gathered, (out,) = torch.ops.symm_mem.fused_all_gather_matmul(a, [w], 0, group_name)
            ref = torch.empty(world_size * rows, 512, device="cuda", dtype=torch.bfloat16)
            dist.all_gather_into_tensor(ref, a)
            # The rows move exactly; the op's per-piece GEMMs round like any bf16 GEMM.
            assert torch.equal(gathered, ref)
            want = ref.float() @ w.float()
            assert float((out.float() - want).norm() / want.norm()) < 1e-2
        rocm_sdma.raise_if_failed()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_torch_sdma_ops_on_the_ipc_workspace(accelerator_ranks):
    accelerator_ranks(_torch_sdma_worker, world_size=4, timeout=300)
