"""Which USP calls the SDMA Ulysses all-to-all takes, and how it splits heads into groups."""

import pytest
import torch

from xfuser.core.attention.spec import AttentionBackendType
from xfuser.core.distributed.sdma.ulysses import MAX_HEAD_GROUPS, split_heads
from xfuser.model_executor.layers import usp_sdma


@pytest.mark.parametrize(
    ("heads", "groups", "sizes"),
    [
        (5, 1, [5]),
        (5, 2, [3, 2]),
        (5, 5, [1, 1, 1, 1, 1]),
        # More groups than heads: one head per group.
        (3, 8, [1, 1, 1]),
        (7, 3, [3, 2, 2]),
    ],
)
def test_split_heads(heads, groups, sizes):
    assert split_heads(heads, groups) == sizes


def test_split_heads_caps_the_group_count():
    assert len(split_heads(4 * MAX_HEAD_GROUPS, 4 * MAX_HEAD_GROUPS)) == MAX_HEAD_GROUPS


def _plain_call(**overrides):
    q = torch.zeros(1, 8, 4, 16)
    call = dict(
        query=q,
        key=q,
        value=q,
        backend=AttentionBackendType.SDPA,
        ring_world_size=1,
        joint_strategy=None,
        fp8_comms=None,
        extra_inputs=[],
        kv_head_repeat=1,
        has_kv_cache=False,
        head_balanced=False,
        # As Wan's runner publishes them: sparse-backend options a dense backend does not read.
        attention_kwargs={"valid_kv_len": 30, "indices_k": None, "thw": (2, 2, 2), "spargeattn_simthreshold": 0.6},
    )
    call.update(overrides)
    return call


@pytest.fixture
def rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.0")


def _reason(**overrides):
    call = _plain_call(**overrides)
    return usp_sdma.unsupported_reason(call.pop("query"), call.pop("key"), call.pop("value"), **call)


def test_plain_ulysses_call_takes_the_sdma_path(rocm):
    assert _reason() is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"ring_world_size": 2},
        {"joint_strategy": "rear"},
        {"fp8_comms": object()},
        {"extra_inputs": [("vsa_h3_gate", torch.zeros(1))]},
        {"kv_head_repeat": 2},
        {"has_kv_cache": True},
        {"head_balanced": True},
        {"value": torch.zeros(1, 8, 4, 32)},
        {"backend": AttentionBackendType.AITER_SPARGE},
        {"attention_kwargs": {"attn_mask": torch.ones(4, 4, dtype=torch.bool)}},
    ],
)
def test_calls_the_sdma_path_does_not_cover_stay_on_rccl(rocm, overrides):
    assert _reason(**overrides) is not None


def test_cuda_builds_stay_on_rccl(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    assert _reason() is not None


def test_configure_returns_the_previous_selection():
    previous = usp_sdma.configure("sdma", 3, True)
    try:
        assert usp_sdma.selected()
        assert usp_sdma.configure("rccl", 1) == ("sdma", 3, True)
        assert not usp_sdma.selected()
    finally:
        usp_sdma.configure(*previous)
    with pytest.raises(ValueError):
        usp_sdma.configure("nccl", 1)
