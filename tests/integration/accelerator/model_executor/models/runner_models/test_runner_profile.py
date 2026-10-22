"""The runner's --profile trace holds the active iteration's device kernels.

The runner steps the profiler through exactly wait + warmup + active
iterations. With the schedule's default endless repeat, the last step opened a
new cycle that dropped the active iteration's device events before export, so
on torch 2.13 / ROCm every trace held only its header.
"""

import gzip
import json
from types import SimpleNamespace

import pytest
import torch

KERNELS_PER_ITERATION = 5


def _model():
    from xfuser.model_executor.models.runner_models.base_model import xFuserModel

    class StubModel(xFuserModel):
        _load_model = _run_pipe = None

    model = object.__new__(StubModel)
    model.config = SimpleNamespace(
        profile_wait=0, profile_warmup=1, profile_active=1, profile_with_stack=False, batch_size=None
    )
    model._validate_args = lambda input_args: None
    model._split_prompts_for_dp = lambda input_args: input_args
    model._gather_dp_outputs = lambda output: output
    weight = torch.randn(1024, 1024, device="cuda")

    def run_timed_pipe(input_args):
        for _ in range(KERNELS_PER_ITERATION):
            weight @ weight
        torch.cuda.synchronize()
        return None, 0.0

    model._run_timed_pipe = run_timed_pipe
    return model


def test_profile_trace_holds_the_active_iterations_kernels(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    # The runner's log reads these to pick the process that prints.
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    model = _model()

    _, _, profile = model.profile({"height": 64, "width": 64})

    path = tmp_path / "trace.json.gz"
    profile.export_chrome_trace(str(path))
    events = json.load(gzip.open(path))["traceEvents"]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    assert len(kernels) == KERNELS_PER_ITERATION, "expected the one active iteration's GEMMs, not the warmup's"
