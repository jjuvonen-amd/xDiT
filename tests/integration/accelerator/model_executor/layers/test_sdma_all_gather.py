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


def _worker(rank, world_size, init_method, chunks, trigger, direction, bias, batch=1):
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
            gen = torch.Generator().manual_seed(100 + call)
            full = torch.randn(batch, world_size * ROWS, FEATURES, generator=gen)
            full += torch.arange(world_size * chunks).repeat_interleave(block)[:, None]
            full += 100 * torch.arange(batch)[:, None, None]
            full = full.to(device, torch.bfloat16)
            # Blocks rank, world + rank, ... of world * chunks: this rank's block-cyclic rows.
            local = full.view(batch, chunks, world_size, block, FEATURES)[:, :, rank].reshape(batch, ROWS, FEATURES)
            expected = torch.nn.functional.linear(full, weight, b).float()

            actual = sdma_all_gather_linear(local, weight, b, group_name, chunks, trigger, direction).float()

            assert actual.shape == expected.shape
            error = ((actual - expected).norm() / expected.norm()).item()
            assert error <= TOLERANCE, f"call {call}: relative error {error:.3e}"
        torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("direction", ["push", "pull"])
@pytest.mark.parametrize("trigger", ["gpu", "host"])
@pytest.mark.parametrize("chunks", [1, 2, 4])
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_sdma_all_gather_linear_matches_full_gemm(tmp_path, world_size, chunks, trigger, direction):
    if torch.version.hip is None:
        pytest.skip("SDMA copy engines need ROCm")
    # The bias goes into every piece's GEMM; one piece checks the plain GEMM.
    bias = chunks > 1

    spawn_accelerator_ranks(
        _worker, tmp_path, world_size=world_size, timeout=300, args=(chunks, trigger, direction, bias)
    )


@pytest.mark.parametrize("direction", ["push", "pull"])
@pytest.mark.parametrize("world_size", [2, 4])
def test_sdma_all_gather_linear_batch_of_two(tmp_path, world_size, direction):
    """Two batch items: every item's piece moves with the others, and the output keeps them apart."""
    if torch.version.hip is None:
        pytest.skip("SDMA copy engines need ROCm")

    spawn_accelerator_ranks(_worker, tmp_path, world_size=world_size, timeout=300, args=(2, "gpu", direction, True, 2))


def _two_groups_worker(rank, world_size, init_method):
    import torch.distributed as dist

    from xfuser.model_executor.layers.sdma_all_gather import sdma_all_gather_linear

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world_size, device_id=device)
    try:
        pairs = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        pair = pairs[rank // 2]
        weight = torch.randn(OUT_FEATURES, FEATURES, generator=torch.Generator().manual_seed(0)).to(
            device, torch.bfloat16
        )
        # The TP4 group first, so the TP2 groups come second; then alternate.
        for call, (group, size, index) in enumerate([(dist.group.WORLD, 4, rank), (pair, 2, rank % 2)] * 3):
            full = torch.randn(1, size * ROWS, FEATURES, generator=torch.Generator().manual_seed(call))
            full = (full + torch.arange(size * 2).repeat_interleave(ROWS // 2)[:, None]).to(device, torch.bfloat16)
            local = full.view(1, 2, size, ROWS // 2, FEATURES)[:, :, index].reshape(1, ROWS, FEATURES)
            expected = torch.nn.functional.linear(full, weight).float()

            actual = sdma_all_gather_linear(local, weight, None, group.group_name, 2, "gpu", "push").float()

            error = ((actual - expected).norm() / expected.norm()).item()
            assert error <= TOLERANCE, f"call {call} on a group of {size}: relative error {error:.3e}"
        torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()


def test_sdma_all_gather_linear_on_two_group_sizes(tmp_path):
    """Calls alternate between a TP4 group and two TP2 groups in the same processes."""
    if torch.version.hip is None:
        pytest.skip("SDMA copy engines need ROCm")

    spawn_accelerator_ranks(_two_groups_worker, tmp_path, world_size=4, timeout=300)


def _lost_flag_worker(rank, world_size, init_method):
    import os

    os.environ["XDIT_TP_SDMA_WAIT_S"] = "2"
    os.environ["XDIT_TP_SDMA_ON_TIMEOUT"] = "raise"
    import time

    import torch.distributed as dist

    from xfuser.model_executor.layers import sdma_all_gather as sdma

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world_size, device_id=device)
    try:
        group_name = dist.group.WORLD.group_name
        weight = torch.randn(OUT_FEATURES, FEATURES, device=device, dtype=torch.bfloat16)
        local = torch.randn(1, ROWS, FEATURES, device=device, dtype=torch.bfloat16)
        sdma.sdma_all_gather_linear(local, weight, None, group_name, 2, "gpu", "push")
        torch.cuda.synchronize()
        sdma.raise_if_waits_gave_up(group_name)
        if rank == 0:
            # Rank 0's push of piece 1 to its first peer, rank 1, loses its flag.
            sdma.debug_drop_flag(1, 0)
        start = time.monotonic()
        sdma.sdma_all_gather_linear(local, weight, None, group_name, 2, "gpu", "push")
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        dist.barrier()
        if rank == 1:
            assert 2 <= elapsed < 30, f"the call took {elapsed:.1f} s: its wait should give up after 2 s"
            with pytest.raises(RuntimeError, match=r"piece 1, flag words \[0\] never set"):
                sdma.raise_if_waits_gave_up(group_name)
        else:
            sdma.raise_if_waits_gave_up(group_name)
        # Recorded once: the next check is clean.
        sdma.raise_if_waits_gave_up(group_name)
    finally:
        dist.destroy_process_group()


def test_lost_flag_gives_up_and_is_reported_instead_of_hanging(tmp_path):
    """A flag that never lands ends its wait after XDIT_TP_SDMA_WAIT_S and is reported, not a GPU hang."""
    if torch.version.hip is None:
        pytest.skip("SDMA copy engines need ROCm")

    spawn_accelerator_ranks(_lost_flag_worker, tmp_path, world_size=2, timeout=300)


def _abort_worker(rank, world_size, init_method):
    import os

    os.environ["XDIT_TP_SDMA_WAIT_S"] = "2"
    os.environ["XDIT_TP_SDMA_ON_TIMEOUT"] = "abort"
    import torch.distributed as dist

    from xfuser.model_executor.layers import sdma_all_gather as sdma

    os.environ["RANK"], os.environ["LOCAL_RANK"], os.environ["WORLD_SIZE"] = str(rank), str(rank), str(world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world_size, device_id=device)
    group_name = dist.group.WORLD.group_name
    weight = torch.randn(OUT_FEATURES, FEATURES, device=device, dtype=torch.bfloat16)
    local = torch.randn(1, ROWS, FEATURES, device=device, dtype=torch.bfloat16)
    sdma.sdma_all_gather_linear(local, weight, None, group_name, 2, "gpu", "push")
    torch.cuda.synchronize()
    if rank == 0:
        sdma.debug_drop_flag(1, 0)
    sdma.sdma_all_gather_linear(local, weight, None, group_name, 2, "gpu", "push")
    # Rank 1's stream is held by its wait; its watchdog aborts the process during this synchronize.
    torch.cuda.synchronize()


def test_lost_flag_aborts_the_waiting_process(tmp_path):
    """With XDIT_TP_SDMA_ON_TIMEOUT=abort, a flag that never lands aborts the rank waiting for it, never hangs it."""
    import signal
    import time

    if torch.version.hip is None:
        pytest.skip("SDMA copy engines need ROCm")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2 accelerator devices")
    context = torch.multiprocessing.get_context("spawn")
    init_method = f"file://{tmp_path / 'dist-init'}"
    processes = [context.Process(target=_abort_worker, args=(r, 2, init_method)) for r in range(2)]
    for process in processes:
        process.start()
    start = time.monotonic()
    processes[1].join(240)
    elapsed = time.monotonic() - start
    for process in processes:
        if process.is_alive():
            process.kill()
        process.join(10)
    assert processes[1].exitcode == -signal.SIGABRT, f"rank 1 exit code {processes[1].exitcode}, expected SIGABRT"
    assert elapsed < 240, "rank 1 did not abort"
