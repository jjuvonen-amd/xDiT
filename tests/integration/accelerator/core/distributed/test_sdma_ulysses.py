"""Ulysses attention with its all-to-alls on SDMA engines matches USP on RCCL."""

import pytest

pytestmark = [pytest.mark.multi_gpu, pytest.mark.rocm]

BACKEND = "SDPA"


def _setup(rank, world_size, init_method, ulysses, cfg=1):
    import torch

    from xfuser.config.args import xFuserArgs
    from xfuser.core.distributed import (
        get_runtime_state,
        init_distributed_environment,
        initialize_model_parallel,
        initialize_runtime_state,
    )

    torch.cuda.set_device(rank)
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(classifier_free_guidance_degree=cfg, ulysses_degree=ulysses)
    engine_config, _ = xFuserArgs(
        attention_backend=BACKEND, ulysses_degree=ulysses, use_cfg_parallel=cfg > 1
    ).create_config()
    initialize_runtime_state(engine_config=engine_config)
    get_runtime_state().set_attention_backend(BACKEND)
    return torch.device("cuda", rank)


def _qkv(device, batch, heads, local_seq, seed):
    """This rank's Q, K and V as models hand them to USP: [b, H, s, d] views of [b, s, H, d]."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(batch, local_seq, heads, 64, generator=generator).to(device, torch.bfloat16).transpose(1, 2)
        for _ in range(3)
    ]


def _usp(q, k, v, mode, head_groups, concurrent=False, attention_kwargs=None):
    from xfuser.model_executor.layers import usp_sdma
    from xfuser.model_executor.layers.usp import USP

    previous = usp_sdma.configure(mode, head_groups, concurrent)
    try:
        return USP(q, k, v, attention_kwargs=attention_kwargs)
    finally:
        usp_sdma.configure(*previous)


def _elementwise(q, k, v):
    # Stands in for attention: exact in any layout, so the exchange alone is compared.
    return q + k / 2 + v / 4


def _transport_worker(rank, world_size, init_method, cfg):
    import torch
    import torch.distributed as dist
    from yunchang.globals import PROCESS_GROUP

    from xfuser.core.distributed import get_ulysses_parallel_rank
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.core.distributed.sdma import ulysses as sdma_ulysses
    from xfuser.model_executor.layers.usp import _combined_qkv_all_to_all, _ft_c_output_all_to_all

    try:
        ulysses = world_size // cfg
        device = _setup(rank, world_size, init_method, ulysses, cfg)
        # 5 heads per rank: 2 groups split them 3 + 2, 5 groups one by one.
        heads = 5 * ulysses
        cases = [
            # (batch, local sequence, head groups, concurrent)
            (1, 96, 1, False),
            (1, 96, 2, False),
            (1, 96, 5, False),
            (1, 96, 2, True),
            (2, 64, 3, False),
            # Larger than every call before it: the shared buffers grow.
            (1, 1024, 2, False),
        ]
        for index, (batch, seq, groups, concurrent) in enumerate(cases):
            # Several calls in a row, as layers make them, each with new data; each CFG group
            # has inputs of its own, as the two branches do.
            for call in range(3):
                seed = 1000 * index + 100 * call + 10 * (rank // ulysses) + get_ulysses_parallel_rank()
                q, k, v = _qkv(device, batch, heads, seq, seed)
                qa, ka, va = _combined_qkv_all_to_all(q, k, v)
                expected = _ft_c_output_all_to_all(_elementwise(qa, ka, va))
                actual = sdma_ulysses.attention(
                    q, k, v, _elementwise, PROCESS_GROUP.ULYSSES_PG, groups, concurrent
                ).transpose(1, 2)
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=0,
                    atol=0,
                    msg=lambda m: f"batch {batch}, seq {seq}, {groups} head groups, concurrent {concurrent}: {m}",
                )
        dist.barrier()
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_sdma_exchange_matches_rccl(accelerator_ranks, world_size):
    accelerator_ranks(_transport_worker, world_size=world_size, timeout=600, args=(1,))


@pytest.mark.parametrize("world_size", [4, 8])
def test_sdma_exchange_matches_rccl_under_cfg_parallel(accelerator_ranks, world_size):
    accelerator_ranks(_transport_worker, world_size=world_size, timeout=600, args=(2,))


def _usp_worker(rank, world_size, init_method):
    import torch
    import torch.distributed as dist

    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

    try:
        device = _setup(rank, world_size, init_method, world_size)
        heads = 5 * world_size
        # Without and with a trailing SP pad that USP drops from the keys.
        for valid in (None, world_size * 256 - 7):
            q, k, v = _qkv(device, 1, heads, 256, rank)
            kwargs = {"valid_kv_len": valid} if valid is not None else None
            expected = _usp(q, k, v, "rccl", 1, attention_kwargs=kwargs)
            actual = _usp(q, k, v, "sdma", 2, attention_kwargs=kwargs)
            # The backend reads bshd views on the SDMA path and contiguous bhsd on RCCL's, and
            # may round differently for the two.
            torch.testing.assert_close(actual, expected, msg=lambda m: f"valid_kv_len {valid}: {m}")
        dist.barrier()
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def test_usp_on_sdma_matches_rccl(accelerator_ranks):
    accelerator_ranks(_usp_worker, world_size=4, timeout=600)


def _lost_flag_worker(rank, world_size, init_method):
    import os

    import torch
    import torch.distributed as dist

    os.environ["XDIT_SDMA_ON_TIMEOUT"] = "raise"
    os.environ["XDIT_SDMA_WAIT_S"] = "2"
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.core.distributed.sdma import transport

    try:
        device = _setup(rank, world_size, init_method, world_size)
        q, k, v = _qkv(device, 1, 5 * world_size, 64, rank)
        _usp(q, k, v, "sdma", 1)
        torch.cuda.synchronize()
        dist.barrier()
        # Rank 0's next push to its first peer (rank 1) loses its flag: rank 1's wait gives up.
        if rank == 0:
            transport.debug_drop_flag(1, 0)
        _usp(q, k, v, "sdma", 1)
        torch.cuda.synchronize()
        try:
            transport.raise_if_waits_gave_up()
            message = ""
        except RuntimeError as error:
            message = str(error)
        every = [None] * world_size
        dist.all_gather_object(every, message)
        # Rank 1's Q/K/V wait (wait 0) gives up on rank 0's word. Rank 0's own wait for rank 1's
        # output may give up too, since that output comes only after rank 1's timeout.
        assert "wait 0, flag words [0] never set" in every[1], every
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def test_lost_flag_is_reported_by_the_rank_that_waited(accelerator_ranks):
    accelerator_ranks(_lost_flag_worker, world_size=2, timeout=300)
