"""The SDMA all-gather + GEMM matches an RCCL all-gather followed by the same GEMM.

Each rank's rows are distinct, so a peer shard copied to the wrong place, left
unwritten, or read before the peer filled it shows up in the output. Each case
runs more calls than the transport has signal slots, with fresh inputs, so a
reused slot that gates or flags the wrong call fails too.
"""

import pytest
import torch

from _spawn import spawn_accelerator_ranks

pytestmark = [pytest.mark.multi_gpu, pytest.mark.rocm]

ROWS, FEATURES, OUT_FEATURES = 256, 1024, 512
CALLS = 6
# bf16 inputs; the consumers' GEMMs and the reference differ only in accumulation order.
TOLERANCE = 1e-2


def _worker(rank, world_size, init_method, consumer, trigger, bias):
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
        for call in range(CALLS):
            local = torch.randn(1, ROWS, FEATURES, generator=torch.Generator().manual_seed(100 * call + rank))
            local = (local + rank).to(device, torch.bfloat16)
            gathered = torch.empty(world_size * ROWS, FEATURES, dtype=torch.bfloat16, device=device)
            dist.all_gather_into_tensor(gathered, local[0])
            expected = torch.nn.functional.linear(gathered, weight, b).float()

            actual = sdma_all_gather_linear(local, weight, b, group_name, consumer, trigger)[0].float()

            assert actual.shape == expected.shape
            error = ((actual - expected).norm() / expected.norm()).item()
            assert error <= TOLERANCE, f"call {call}: relative error {error:.3e}"
        torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("bias", [False, True], ids=["no-bias", "bias"])
@pytest.mark.parametrize("trigger", ["gpu", "host"])
@pytest.mark.parametrize("consumer", ["async_mm", "per_peer"])
@pytest.mark.parametrize("world_size", [2, 4])
def test_sdma_all_gather_linear_matches_rccl(tmp_path, world_size, consumer, trigger, bias):
    if torch.version.hip is None:
        pytest.skip("SDMA copy engines need ROCm")

    spawn_accelerator_ranks(_worker, tmp_path, world_size=world_size, timeout=600, args=(consumer, trigger, bias))
