"""Tensor parallelism over gloo: process groups next to SP, and the sharded layers against unsharded ones.

TP is the innermost rank dimension, so an SP group is strided by the TP degree
([0, 2] and [1, 3] for TP 2 on four ranks). The Ulysses and ring groups must
stay inside those, never pairing TP peers. The layer check runs the Megatron
pattern the Wan blocks use: all-gather rows, column-parallel linear, RMSNorm
over the split features, row-parallel linear, reduce-scatter rows. Rows split
block-cyclically, as the pipelined overlap holds them, must gather back into
natural order.
"""

import functools
import traceback
from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.gloo, pytest.mark.slow]

_WORLD_SIZE = 4
_TP_DEGREE = 2
# (ulysses, ring) sharing the SP degree of 2 left by TP 2 on four ranks.
_SP_LAYOUTS = {"ulysses": (2, 1), "ring": (1, 2)}
_ROWS, _IN, _HIDDEN = 6, 8, 12


def _worker(rank, world_size, init_method, result_queue, *, ulysses, ring):
    dist = None
    try:
        import torch
        import torch.distributed as dist
        from yunchang.globals import PROCESS_GROUP

        from xfuser.core.distributed import parallel_state
        from xfuser.model_executor.layers.tensor_parallel import (
            shard_linear_,
            shard_rms_norm_,
            tp_all_gather_seq,
            tp_gather_rows,
            tp_reduce_scatter_seq,
            tp_rms_norm,
            tp_split_rows,
        )

        with patch.object(parallel_state, "set_device"):
            parallel_state.init_distributed_environment(
                backend="gloo", distributed_init_method=init_method, local_rank=rank, rank=rank, world_size=world_size
            )
        parallel_state.initialize_model_parallel(
            backend="gloo", tensor_parallel_degree=_TP_DEGREE, ulysses_degree=ulysses, ring_degree=ring
        )
        tp_group = parallel_state.get_tp_group()
        tp_rank = tp_group.rank_in_group

        # Seeded, so every rank builds the same layers and input.
        torch.manual_seed(0)
        column = torch.nn.Linear(_IN, _HIDDEN)
        norm = torch.nn.RMSNorm(_HIDDEN, eps=1e-6)
        torch.nn.init.normal_(norm.weight)
        row = torch.nn.Linear(_HIDDEN, _IN)
        x = torch.randn(1, _ROWS, _IN)
        with torch.no_grad():
            expected = row(norm(column(x))).chunk(_TP_DEGREE, dim=1)[tp_rank]

            shard_linear_(column, 0, tp_rank, _TP_DEGREE)
            shard_rms_norm_(norm, tp_rank, _TP_DEGREE)
            row_bias = shard_linear_(row, 1, tp_rank, _TP_DEGREE)
            gathered = tp_all_gather_seq(x.chunk(_TP_DEGREE, dim=1)[tp_rank], "test")
            (hidden,) = tp_rms_norm([column(gathered)], [norm], "test")
            actual = tp_reduce_scatter_seq(row(hidden), "test") + row_bias
            # Block-cyclic rows, as the chunked overlap holds them, gather back into natural order.
            rows = torch.arange(2 * _ROWS * _TP_DEGREE, dtype=torch.float32).view(1, -1, 1)
            round_trips = all(
                torch.equal(tp_gather_rows(tp_split_rows(rows, tp_rank, _TP_DEGREE, chunks), chunks), rows)
                for chunks in (1, 2, 3)
            )

        # xDiT builds yunchang's groups only where yunchang's attention can run, which needs an accelerator.
        has_seq_parallel_pgs = parallel_state.HAS_LONG_CTX_ATTN
        result_queue.put(
            (
                "returned",
                rank,
                tp_group.ranks,
                parallel_state.get_sp_group().ranks,
                dist.get_process_group_ranks(PROCESS_GROUP.ULYSSES_PG) if has_seq_parallel_pgs else None,
                dist.get_process_group_ranks(PROCESS_GROUP.RING_PG) if has_seq_parallel_pgs else None,
                (actual - expected).abs().max().item(),
                round_trips,
            )
        )
    except Exception:  # noqa: BLE001 - report arbitrary child failures to the parent
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("layout", _SP_LAYOUTS)
def test_tensor_parallel_groups_and_layers_over_gloo(tmp_path, run_spawned, layout):
    torch = pytest.importorskip("torch")
    pytest.importorskip("yunchang", reason="the Ulysses and ring groups are yunchang's")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")
    ulysses, ring = _SP_LAYOUTS[layout]

    processes, hung, survivors, results = run_spawned(
        torch,
        functools.partial(_worker, ulysses=ulysses, ring=ring),
        f"file://{tmp_path / 'tp-gloo-init'}",
        world_size=_WORLD_SIZE,
    )

    assert not survivors, f"workers survived SIGKILL: {survivors}"
    assert not hung, f"workers hung: {hung}"
    errors = [result[2] for result in results if result[0] == "error"]
    assert not errors, "\n".join(errors)
    assert [process.exitcode for process in processes] == [0] * _WORLD_SIZE
    assert len(results) == _WORLD_SIZE
    sp_groups = {0: [0, 2], 1: [1, 3], 2: [0, 2], 3: [1, 3]}
    for _, rank, tp_ranks, sp_ranks, ulysses_ranks, ring_ranks, _, round_trips in results:
        assert round_trips, f"rank {rank}: block-cyclic rows did not gather back into natural order"
        assert tp_ranks == [rank - rank % 2, rank - rank % 2 + 1]
        assert sp_ranks == sp_groups[rank]
        if ulysses_ranks is not None:
            assert ulysses_ranks == (sp_groups[rank] if ulysses > 1 else [rank])
            assert ring_ranks == (sp_groups[rank] if ring > 1 else [rank])
    for _, rank, *_, error, _ in results:
        assert error <= 1e-5, f"rank {rank}: sharded layers differ from unsharded ones by {error:.3e}"
