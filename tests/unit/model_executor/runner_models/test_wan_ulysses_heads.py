"""Wan runners refuse a Ulysses degree that does not divide their heads before any weights load."""

import pytest


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


def _build(model_name, **config):
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models import wan  # noqa: F401  registers
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    return MODEL_REGISTRY[model_name](xFuserArgs(model=model_name, **config))


@pytest.mark.parametrize(
    ("name", "task", "heads", "accepted", "refused"),
    [
        ("Wan2.1-T2V", None, 40, 8, 3),
        ("Wan2.2-T2V", None, 40, 5, 16),
        ("Wan2.1-I2V", None, 40, 4, 6),
        ("Wan2.2-I2V", None, 40, 2, 3),
        ("Wan2.2-TI2V", "t2v", 24, 8, 5),
        ("Wan2.1-VACE-14B", None, 40, 8, 12),
        ("Wan2.1-VACE-1.3B", None, 12, 4, 8),
    ],
)
def test_wan_ulysses_degree_must_divide_heads_before_loading(name, task, heads, accepted, refused):
    task_args = {"task": task} if task else {}

    _build(name, ulysses_degree=accepted, **task_args)
    with pytest.raises(ValueError, match=rf"has {heads} attention heads.*got {refused}\..*--ring_degree"):
        _build(name, ulysses_degree=refused, **task_args)


def test_wan22_t2v_tensor_parallel_times_ulysses_must_divide_heads():
    """TP takes 1/tp of the heads and Ulysses splits each rank's share; 8 divides 40 but not the 20 of TP 2."""
    _build("Wan2.2-T2V", tensor_parallel_degree=2, ulysses_degree=4)
    with pytest.raises(ValueError, match=r"must divide 40; got 2 x 8\."):
        _build("Wan2.2-T2V", tensor_parallel_degree=2, ulysses_degree=8)


def test_wan_tensor_parallel_refuses_what_its_sharding_does_not_split():
    with pytest.raises(ValueError, match=r"does not support --use_fp8_gemms, --cache_method\."):
        _build("Wan2.2-T2V", tensor_parallel_degree=2, use_fp8_gemms=True, cache_method="fbcache")
    with pytest.raises(ValueError, match=r"does not support tensor_parallel_degree"):
        _build("Wan2.1-T2V", tensor_parallel_degree=2)
