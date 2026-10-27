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
from xfuser.model_executor.models.runner_models.wan import (
    xFuserWan21I2VModel,
    xFuserWan21T2VModel,
    xFuserWan22TI2VModel,
)

SOL_BACKENDS = [
    "aiter_fp8_sol", "aiter_i8fp8_sol",
    # The MXFP6 rows aiter's mha_v4-sol-attn branch added.
    "aiter_mxfp6_sol", "aiter_f8f6_sol", "aiter_f6f8_sol", "aiter_f6f4_sol",
]

# The model class, a name it accepts, and any extra args its own validation demands.
#
# The H3 family has no cross-attention -- it packs text, audio and video into one sequence and
# attends them together -- so it must keep being accepted without --cross_attention_backend.
H3_MODELS = [
    (xFuserMiniMaxH3Model, "MiniMax-H3", {"task": "t2va"}),
    (xFuserFastH3Model, "FastH3", {"task": "t2va"}),
    (xFuserFastH3DenseModel, "FastH3-Dense", {"task": "t2va"}),
]

# Wan does have cross-attention, so every Sol row it selects has to name a backend for it.
# Wan2.2-TI2V serves two tasks and insists one be named.
WAN_MODELS = [
    (xFuserWan21T2VModel, "Wan2.1-T2V", {"cross_attention_backend": "aiter"}),
    (xFuserWan21I2VModel, "Wan2.1-I2V", {"cross_attention_backend": "aiter"}),
    (
        xFuserWan22TI2VModel,
        "Wan2.2-TI2V",
        {"cross_attention_backend": "aiter", "task": "t2v"},
    ),
]

MODELS = H3_MODELS + WAN_MODELS


def _build(cls, model, **kwargs):
    return cls(xFuserArgs(model=model, **kwargs))


@pytest.mark.parametrize("cls,model,extra", MODELS)
@pytest.mark.parametrize("backend", SOL_BACKENDS)
def test_sol_backends_are_accepted(cls, model, extra, backend):
    _build(cls, model, attention_backend=backend, **extra)


@pytest.mark.parametrize("cls,model,extra", H3_MODELS)
@pytest.mark.parametrize("backend", ["aiter_sparge", "aiter_fp8_sparge"])
def test_sol_does_not_drag_in_sparge(cls, model, extra, backend):
    """Opting into Sol-Attn must not also opt a model into Sparge.

    The two are separate capabilities: Sparge builds a mask from the operands, Sol-Attn routes its
    own. Sharing one capability would hand a model every Sparge row unchecked. Only the H3 family
    can show this -- Wan is wired for both, so it has nothing to be refused.
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
    folded in a pooled correction for the blocks it skipped. The H3 family refuses ring_degree on
    its own capability first, which is a stricter answer to the same question; Wan advertises ring
    and so reaches the Sol-specific check.
    """
    with pytest.raises(ValueError, match="ring"):
        _build(
            cls, model, attention_backend="aiter_fp8_sol", ring_degree=2, **extra
        )


@pytest.mark.parametrize("cls,model,extra", WAN_MODELS)
def test_the_sol_ring_check_names_the_reason(cls, model, extra):
    """Wan is the model that actually reaches it: it serves Sol-Attn and advertises ring_degree.

    The H3 family bails out earlier on its own ring capability, so without Wan this check would
    only ever be exercised by calling it directly.
    """
    with pytest.raises(ValueError, match="Sol-Attn does not support ring parallelism"):
        _build(
            cls, model, attention_backend="aiter_fp8_sol", ring_degree=2, **extra
        )


@pytest.mark.parametrize("cls,model,extra", WAN_MODELS)
def test_wan_sol_demands_a_cross_attention_backend(cls, model, extra):
    """A routed backend must not be left to serve Wan's cross-attention by falling back.

    Wan's text KV is 512 tokens -- four KV blocks -- which is too few for a per-tile threshold to
    select meaningfully and too few for a pooled correction to carry what it skips. It would not
    fail, it would quietly answer with a worse number.
    """
    without_cross = {k: v for k, v in extra.items() if k != "cross_attention_backend"}
    with pytest.raises(ValueError, match="cross_attention_backend must be"):
        _build(cls, model, attention_backend="aiter_fp8_sol", **without_cross)


@pytest.mark.parametrize("cls,model,extra", WAN_MODELS)
def test_wan_cross_attention_may_not_itself_be_routed(cls, model, extra):
    """Naming the same Sol row for cross-attention is the same hazard spelled differently."""
    routed = {**extra, "cross_attention_backend": "aiter_fp8_sol"}
    with pytest.raises(ValueError, match="cross_attention_backend cannot be"):
        _build(cls, model, attention_backend="aiter_fp8_sol", **routed)


@pytest.mark.parametrize("cls,model,extra", H3_MODELS)
def test_h3_needs_no_cross_attention_backend(cls, model, extra):
    """The mirror of the Wan case: a model with no cross-attention must not be asked for one.

    H3 index_copies text, video and audio into one packed sequence, so there is no second
    attention call for a routed backend to fall back into.
    """
    _build(cls, model, attention_backend="aiter_fp8_sol", **extra)


def test_models_wired_for_sol_are_the_ones_we_expect():
    """Sol-Attn is opt-in per family. This pins which runners carry the capability today.

    Wan VACE and Causal Wan are deliberately out: VACE's vace_blocks take a different path through
    attention, and Causal Wan is causal, which Sol-Attn has no variant for.
    """
    from xfuser.model_executor.models.runner_models.causal_wan import (
        xFuserCausalWanModel,
    )
    from xfuser.model_executor.models.runner_models.wan import xFuserWan21VACEModel

    for cls, _, _ in MODELS:
        assert cls.capabilities.supports_sol_attention_backends, cls.__name__
    for cls in (xFuserWan21VACEModel, xFuserCausalWanModel):
        assert not cls.capabilities.supports_sol_attention_backends, cls.__name__


def test_a_model_without_the_capability_still_refuses_sol():
    """The gate is opt-in, so a model that has not been checked for Sol-Attn keeps saying no."""
    from xfuser.model_executor.models.runner_models.flux import xFuserFluxModel

    with pytest.raises(ValueError, match="does not support Sol-Attn"):
        xFuserFluxModel(
            xFuserArgs(model="FLUX.1-dev", attention_backend="aiter_fp8_sol")
        )


def test_wan_edge_frame_flag_reaches_the_attention_kwargs():
    """--solattn_exact_edge_frames is read by the transformer out of the shared attention kwargs."""
    from xfuser.model_executor.models.runner_models.wan import _build_attention_kwargs

    for flag in (False, True):
        args = xFuserArgs(
            model="Wan2.1-T2V",
            attention_backend="aiter_fp8_sol",
            cross_attention_backend="aiter",
            solattn_exact_edge_frames=flag,
        )
        assert _build_attention_kwargs(args)["solattn_exact_edge_frames"] is flag


def test_wan_edge_frame_tokens_flag_the_first_and_last_latent_frames():
    """Frame-major tokens: frame t owns [t*HW, (t+1)*HW). The sequence-parallel pad stays unflagged."""
    import torch

    from xfuser.model_executor.models.transformers.transformer_wan import _edge_frame_tokens

    mask = _edge_frame_tokens(num_frames=3, frame_tokens=4, pad_amount=2, device="cpu")
    expected = [True] * 4 + [False] * 4 + [True] * 4 + [False] * 2
    assert mask.dtype == torch.bool and mask.tolist() == expected


@pytest.mark.parametrize(
    "argv,router,jensen",
    [([], "mean", False),
     (["--solattn_router", "error", "--solattn_jensen"], "error", True)],
)
def test_sol_router_and_jensen_flags_reach_the_runtime_config(argv, router, jensen):
    """Defaults leave Sol exactly as it was; the flags land where the backend reads them."""
    from xfuser.config.args import FlexibleArgumentParser
    from xfuser.config.config import RuntimeConfig

    parser = xFuserArgs.add_runner_args(FlexibleArgumentParser(description="xDiT"))
    args = xFuserArgs.from_cli_args(parser.parse_args(["--model", "Wan2.1-T2V", *argv]))
    assert (args.solattn_router, args.solattn_jensen) == (router, jensen)
    runtime = RuntimeConfig(solattn_router=args.solattn_router, solattn_jensen=args.solattn_jensen)
    assert (runtime.solattn_router, runtime.solattn_jensen) == (router, jensen)


def test_an_unknown_sol_router_is_refused_by_the_parser():
    from xfuser.config.args import FlexibleArgumentParser

    parser = xFuserArgs.add_runner_args(FlexibleArgumentParser(description="xDiT"))
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "Wan2.1-T2V", "--solattn_router", "nearest"])
