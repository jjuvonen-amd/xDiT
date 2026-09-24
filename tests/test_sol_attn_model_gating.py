"""Which models may select a Sol-Attn backend, and what they must not silently acquire with it.

Config validation only -- no kernel, no device. Whether the machine actually carries the manifest
row for a recipe is a separate check that runtime_state makes against the device.
"""

import pytest

from xfuser.config import xFuserArgs
from xfuser.model_executor.models.runner_models.minimax_h3 import (
    xFuserFastH3DenseModel,
    xFuserFastH3Model,
    xFuserMiniMaxH3Model,
)

SOL_BACKENDS = ["aiter_fp8_sol", "aiter_i8fp8_sol"]

# The model class, a name it accepts, and any extra args its own validation demands.
MODELS = [
    (xFuserMiniMaxH3Model, "MiniMax-H3", {"task": "t2va"}),
    (xFuserFastH3Model, "FastH3", {"task": "t2va"}),
    (xFuserFastH3DenseModel, "FastH3-Dense", {"task": "t2va"}),
]


def _build(cls, model, **kwargs):
    return cls(xFuserArgs(model=model, **kwargs))


@pytest.mark.parametrize("cls,model,extra", MODELS)
@pytest.mark.parametrize("backend", SOL_BACKENDS)
def test_sol_backends_are_accepted(cls, model, extra, backend):
    _build(cls, model, attention_backend=backend, **extra)


@pytest.mark.parametrize("cls,model,extra", MODELS)
@pytest.mark.parametrize("backend", ["aiter_sparge", "aiter_fp8_sparge"])
def test_sol_does_not_drag_in_sparge(cls, model, extra, backend):
    """Opting into Sol-Attn must not also opt a model into Sparge.

    The two are separate capabilities: Sparge builds a mask from the operands, Sol-Attn routes its
    own. Sharing one capability would hand a model every Sparge row unchecked.
    """
    with pytest.raises(ValueError, match="does not support Sparge"):
        _build(cls, model, attention_backend=backend, **extra)


@pytest.mark.parametrize("cls,model,extra", MODELS)
def test_ulysses_stays_available(cls, model, extra):
    """Ulysses is the sequence parallelism Sol-Attn does support, so it must survive the gate."""
    _build(cls, model, attention_backend="aiter_fp8_sol", ulysses_degree=2, **extra)


@pytest.mark.parametrize("cls,model,extra", MODELS)
def test_ring_is_refused_before_the_denoise_loop(cls, model, extra):
    """Sol-Attn cannot be a ring rank, and finding that out mid-denoise is too late.

    Ring merges each rank's partial output by its LSE, which stops being valid once a rank has
    folded in a pooled correction for the blocks it skipped. Every model wired for Sol-Attn here
    refuses ring_degree on its own capability first, which is a stricter answer to the same
    question; the Sol-specific check below is what covers a model that does advertise ring.
    """
    with pytest.raises(ValueError, match="ring"):
        _build(
            cls, model, attention_backend="aiter_fp8_sol", ring_degree=2, **extra
        )


def test_the_sol_ring_check_names_the_reason():
    """Exercised directly: no model currently both serves Sol-Attn and advertises ring_degree."""
    from xfuser.model_executor.models.runner_models.base_model import (
        _validate_ring_for_sol,
    )

    _validate_ring_for_sol(xFuserArgs(model="MiniMax-H3", ring_degree=1))
    with pytest.raises(ValueError, match="does not support ring parallelism"):
        _validate_ring_for_sol(xFuserArgs(model="MiniMax-H3", ring_degree=2))


def test_a_model_without_the_capability_still_refuses_sol():
    """The gate is opt-in, so a model that has not been checked for Sol-Attn keeps saying no."""
    from xfuser.model_executor.models.runner_models.flux import xFuserFluxModel

    with pytest.raises(ValueError, match="does not support Sol-Attn"):
        xFuserFluxModel(
            xFuserArgs(model="FLUX.1-dev", attention_backend="aiter_fp8_sol")
        )
