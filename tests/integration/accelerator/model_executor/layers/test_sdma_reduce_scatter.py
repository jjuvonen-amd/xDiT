"""The SDMA GEMM + reduce-scatter matches a GEMM whose partial sums are all-reduced.

Each rank holds different input features, so every owner's output block is a
sum over all ranks' partials: a partial pushed to the wrong owner or block, left
out, or read before it landed shows up in the output. The output holds this
rank's block-cyclic rows, as under chunked overlap. Each case runs more calls
than the transport has signal slots, with fresh inputs.
"""

import pytest
import torch

from _spawn import spawn_accelerator_ranks

pytestmark = [pytest.mark.multi_gpu, pytest.mark.rocm]

FEATURES, OUT_FEATURES = 512, 384
BLOCK_ROWS = 64
CALLS = 6
# bf16 partials summed in fp32 against an fp32 reference of the same bf16 inputs.
TOLERANCE = 1e-2


def _worker(rank, world_size, init_method, chunks, trigger):
    import torch.distributed as dist

    from xfuser.model_executor.layers.sdma_all_gather import sdma_linear_reduce_scatter

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world_size, device_id=device)
    try:
        group_name = dist.group.WORLD.group_name
        rows = world_size * chunks * BLOCK_ROWS
        for call in range(CALLS):
            generator = torch.Generator().manual_seed(1000 * call + rank)
            x = torch.randn(1, rows, FEATURES, generator=generator).to(device, torch.bfloat16)
            weight = torch.randn(OUT_FEATURES, FEATURES, generator=generator).to(device, torch.bfloat16)
            full = (x[0].float() @ weight.float().t()).contiguous()
            dist.all_reduce(full)
            # This rank's blocks rank, world + rank, ... of world * chunks, stacked.
            expected = full.view(chunks, world_size, BLOCK_ROWS, OUT_FEATURES)[:, rank].reshape(-1, OUT_FEATURES)

            actual = sdma_linear_reduce_scatter(x, weight, group_name, chunks, trigger)[0].float()

            assert actual.shape == expected.shape
            error = ((actual - expected).norm() / expected.norm()).item()
            assert error <= TOLERANCE, f"call {call}: relative error {error:.3e}"
        torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("trigger", ["gpu", "host"])
@pytest.mark.parametrize("chunks", [1, 2, 4])
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_sdma_linear_reduce_scatter_matches_all_reduced_gemm(tmp_path, world_size, chunks, trigger):
    if torch.version.hip is None:
        pytest.skip("SDMA copy engines need ROCm")

    spawn_accelerator_ranks(_worker, tmp_path, world_size=world_size, timeout=300, args=(chunks, trigger))


def _lost_flag_worker(rank, world_size, init_method):
    import os

    os.environ["XDIT_TP_SDMA_WAIT_S"] = "2"
    import torch.distributed as dist

    from xfuser.model_executor.layers import sdma_all_gather as sdma

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world_size, device_id=device)
    try:
        group_name = dist.group.WORLD.group_name
        x = torch.randn(1, world_size * 2 * BLOCK_ROWS, FEATURES, device=device, dtype=torch.bfloat16)
        weight = torch.randn(OUT_FEATURES, FEATURES, device=device, dtype=torch.bfloat16)
        sdma.sdma_linear_reduce_scatter(x, weight, group_name, 2, "gpu")
        torch.cuda.synchronize()
        if rank == 0:
            # Rank 0's partials of chunk 1 for its first peer, rank 1, lose their flag.
            sdma.debug_drop_flag(1, 0)
        sdma.sdma_linear_reduce_scatter(x, weight, group_name, 2, "gpu")
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 1:
            with pytest.raises(RuntimeError, match=r"piece 1, flag words \[0\] never set"):
                sdma.raise_if_waits_gave_up(group_name)
        else:
            sdma.raise_if_waits_gave_up(group_name)
    finally:
        dist.destroy_process_group()


def test_lost_partial_flag_gives_up_and_is_reported_instead_of_hanging(tmp_path):
    """A partial-sum flag that never lands ends its wait after XDIT_TP_SDMA_WAIT_S and is reported."""
    if torch.version.hip is None:
        pytest.skip("SDMA copy engines need ROCm")

    spawn_accelerator_ranks(_lost_flag_worker, tmp_path, world_size=2, timeout=300)
