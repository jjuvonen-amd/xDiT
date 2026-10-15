"""fp4_gemm_include_suffixes keeps every other linear of an FP4 target in bf16, on each FP4 path.

Base Z-Image needs this: per-tensor quantization of feed_forward.w2 destroys its output, so FP4
must leave that layer (and adaLN_modulation) in bf16 exactly as FP8 does.
"""

from types import SimpleNamespace

import pytest

from xfuser.model_executor.models.runner_models.loading import (
    placement,
    shard,
    transformer_load,
)
from xfuser.model_executor.models.runner_models.loading.contracts import QuantizationFormat
from xfuser.model_executor.models.runner_models.loading.quantization_ledger import QuantizationLedger

SUFFIXES = ("attention.to_q", "feed_forward.w1")


class _Plan:
    def __init__(self, targets):
        self.targets = targets

    def module_list(self, format_name="fp8"):
        return list(self.targets)

    def targets_for(self, component_name, format_name="fp8"):
        prefix = f"{component_name}."
        return [t[len(prefix):] for t in self.targets if t.startswith(prefix)]


class _FormatAdapter:
    def __init__(self):
        self.block_calls = []
        self.module_calls = []

    def convert_block(self, block, **kwargs):
        self.block_calls.append(kwargs)

    def convert_module(self, module, **kwargs):
        self.module_calls.append(kwargs)


def _loader(include_suffixes):
    targets = ["transformer.layers"]
    adapter = _FormatAdapter()
    model = SimpleNamespace(
        config=SimpleNamespace(
            use_fp4_gemms=True,
            use_fp8_gemms=False,
            use_int8_gemms=False,
            use_hybrid_gemm_schedule=False,
        ),
        settings=SimpleNamespace(
            fp4_gemm_module_list=targets,
            fp4_gemm_include_suffixes=include_suffixes,
            fp8_gemm_include_suffixes=None,
            fp8_precision_overrides=None,
            fp8_precision_override_suffixes=None,
        ),
        pipe=SimpleNamespace(transformer=SimpleNamespace(layers=object())),
    )
    loader = SimpleNamespace(
        model=model,
        quantization_plan=_Plan(targets),
        quantization_ledger=QuantizationLedger(),
        backends=SimpleNamespace(
            format=adapter,
            format_targets_for=lambda name: _Plan(targets).targets_for(name),
        ),
    )
    return loader, adapter


def _assert_bf16_outside_suffixes(filter_fn, prefix=""):
    assert filter_fn(object(), f"{prefix}attention.to_q")
    assert filter_fn(object(), f"{prefix}feed_forward.w1")
    assert not filter_fn(object(), f"{prefix}feed_forward.w2")
    assert not filter_fn(object(), f"{prefix}adaLN_modulation.0")


def test_blockwise_fp4_converts_only_included_suffixes():
    loader, adapter = _loader(SUFFIXES)

    quantize = shard.build_block_quantize_fn(loader, "transformer", ["layers"], local_rank=0)
    quantize(object(), 0)

    _assert_bf16_outside_suffixes(adapter.block_calls[0]["filter_fn"])


def test_blockwise_fp4_without_suffixes_converts_every_linear():
    loader, adapter = _loader(None)

    quantize = shard.build_block_quantize_fn(loader, "transformer", ["layers"], local_rank=0)
    quantize(object(), 0)

    assert adapter.block_calls[0]["filter_fn"](object(), "feed_forward.w2")


def test_eager_fp4_converts_only_included_suffixes(monkeypatch):
    loader, adapter = _loader(SUFFIXES)
    monkeypatch.setattr(placement, "log", lambda *_args: None)
    monkeypatch.setattr(
        placement,
        "prepare_native_transformer_format_load",
        lambda *_args, **_kwargs: SimpleNamespace(descriptor=SimpleNamespace(log_message=lambda: "")),
    )

    placement.setup_mxfp4_gemms(loader, local_rank=0)

    _assert_bf16_outside_suffixes(adapter.module_calls[0]["filter_fn"], prefix="0.")


@pytest.mark.parametrize(("include_suffixes", "streams"), [(SUFFIXES, False), (None, True)])
def test_native_fp4_streaming_is_disabled_for_suffix_restricted_targets(monkeypatch, include_suffixes, streams):
    observed = {}
    from xfuser.model_executor.models.runner_models.loading import format_backends

    monkeypatch.setattr(
        format_backends,
        "prepare_native_transformer_format_load",
        lambda _adapter, **kwargs: observed.update(kwargs),
    )
    model = SimpleNamespace(
        settings=SimpleNamespace(
            fp4_gemm_include_suffixes=include_suffixes,
            fp8_precision_overrides=None,
            fp8_precision_override_suffixes=None,
        ),
        config=SimpleNamespace(use_hybrid_gemm_schedule=False),
    )

    transformer_load._prepare_native_load(
        model,
        SimpleNamespace(format=QuantizationFormat.FP4),
        "transformer",
        ("layers",),
        True,
        lambda: object(),
    )

    assert observed["stream_quant"] is streams


def test_flux2_klein_4b_does_not_inherit_fp4_support():
    from xfuser.model_executor.models.runner_models.flux import (
        xFuserFlux2Klein4BModel,
        xFuserFlux2Klein9BModel,
    )

    assert xFuserFlux2Klein9BModel.capabilities.use_fp4_gemms
    assert not xFuserFlux2Klein4BModel.capabilities.use_fp4_gemms
