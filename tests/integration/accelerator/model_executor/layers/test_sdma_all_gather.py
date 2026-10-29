"""The SDMA all-gather + GEMM matches the GEMM over the whole gathered input.

Each rank holds its block-cyclic rows of one global input, as under chunked
overlap, and the op returns the GEMM over all rows in natural order. Rows
differ across ranks and pieces, so a piece copied to the wrong place, left
unwritten, or read before its peer filled it shows up in the output. Each case
runs more calls than the transport has signal slots, with fresh inputs, so a
reused slot that gates or flags the wrong call fails too.
"""

import pytest
import torch

from _spawn import spawn_accelerator_ranks

pytestmark = [pytest.mark.multi_gpu, pytest.mark.rocm]

ROWS, FEATURES, OUT_FEATURES = 256, 1024, 512
CALLS = 6
# bf16 inputs; the per-piece GEMMs and the reference differ only in accumulation order.
TOLERANCE = 1e-2


def _worker(rank, world_size, init_method, chunks, trigger, bias):
    import torch.distributed as dist

    from xfuser.model_executor.layers.sdma_all_gather import sdma_all_gather_linear

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world_size, device_id=device)
    try:
        group_name = dist.group.WORLD.group_name
        generator = torch.Generator().manual_seed(0)
        weight = torch.randn(OUT_FEATURES, FEATURES, generator=generator).to(device, torch.bfloat16)
        b = torch.randn(OUT_FEATURES, generator=generator).to(device, torch.bfloat16) if bias else None
        block = ROWS // chunks
        for call in range(CALLS):
            # The same global rows on every rank, each row offset by its block so misplaced pieces show.
            full = torch.randn(world_size * ROWS, FEATURES, generator=torch.Generator().manual_seed(100 + call))
            full += torch.arange(world_size * chunks).repeat_interleave(block)[:, None]
            full = full.to(device, torch.bfloat16)
            # Blocks rank, world + rank, ... of world * chunks: this rank's block-cyclic rows.
            local = full.view(chunks, world_size, block, FEATURES)[:, rank].reshape(1, ROWS, FEATURES)
            expected = torch.nn.functional.linear(full, weight, b).float()

            actual = sdma_all_gather_linear(local, weight, b, group_name, chunks, trigger)[0].float()

            assert actual.shape == expected.shape
            error = ((actual - expected).norm() / expected.norm()).item()
            assert error <= TOLERANCE, f"call {call}: relative error {error:.3e}"
        torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("trigger", ["gpu", "host"])
@pytest.mark.parametrize("chunks", [1, 2, 4])
@pytest.mark.parametrize("world_size", [2, 4])
def test_sdma_all_gather_linear_matches_full_gemm(tmp_path, world_size, chunks, trigger):
    if torch.version.hip is None:
        pytest.skip("SDMA copy engines need ROCm")
    # The bias goes into every piece's GEMM; one piece checks the plain GEMM.
    bias = chunks > 1

    spawn_accelerator_ranks(_worker, tmp_path, world_size=world_size, timeout=300, args=(chunks, trigger, bias))
