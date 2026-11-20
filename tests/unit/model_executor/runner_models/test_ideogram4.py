from types import SimpleNamespace

import pytest
import torch

from xfuser.model_executor.pipelines.pipeline_ideogram4 import (
    _broadcast_object_in_group,
)


def test_ideogram4_broadcast_object_uses_cpu_subgroup(monkeypatch):
    cpu_group = object()
    coordinator = SimpleNamespace(first_rank=3, cpu_group=cpu_group)

    def fake_broadcast(payload, src, group):
        assert src == 3
        assert group is cpu_group
        payload[0] = ["broadcast caption"]

    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        fake_broadcast,
    )

    assert _broadcast_object_in_group(None, coordinator) == ["broadcast caption"]


@pytest.fixture
def build_ideogram4(monkeypatch):
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models import ideogram4  # noqa: F401  registers
    from xfuser.runner import xFuserModelRunner

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    runner = object.__new__(xFuserModelRunner)

    def _build(name):
        return runner._select_model(name, xFuserArgs(model=name))

    return _build


@pytest.mark.parametrize(
    ("name", "checkpoint"),
    [
        ("Ideogram-4", "ideogram-ai/ideogram-4-fp8"),
        ("ideogram-ai/ideogram-v4", "ideogram-ai/ideogram-4-fp8"),
        ("ideogram-ai/ideogram-4-fp8", "ideogram-ai/ideogram-4-fp8"),
        (
            "CalamitousFelicitousness/Ideogram-4-bf16-Diffusers",
            "CalamitousFelicitousness/Ideogram-4-bf16-Diffusers",
        ),
    ],
)
def test_ideogram4_name_selects_its_checkpoint(build_ideogram4, name, checkpoint):
    model = build_ideogram4(name)

    assert model.loader.checkpoint_request().model_name_or_path == checkpoint


@pytest.mark.parametrize("name", ["ideogram-ai/ideogram-4-nf4", "ideogram-ai/ideogram-4-nf4-diffusers"])
def test_ideogram4_nf4_checkpoints_are_not_registered(build_ideogram4, name):
    with pytest.raises(ValueError, match="not found in registry"):
        build_ideogram4(name)


def _tiny_ideogram4():
    from xfuser.model_executor.models.transformers.transformer_ideogram4 import (
        get_ideogram4_transformer_wrapper_class,
    )

    cls = get_ideogram4_transformer_wrapper_class()
    model = cls.from_config(
        dict(
            in_channels=8,
            num_layers=2,
            attention_head_dim=32,
            num_attention_heads=2,
            intermediate_size=64,
            adaln_dim=16,
            llm_features_dim=32,
            mrope_section=(6, 5, 5),
        )
    ).eval()
    model._init_sp_state()
    return model


def _forward(model, pad, text, image):
    from diffusers.models.transformers.transformer_ideogram4 import (
        LLM_TOKEN_INDICATOR,
        OUTPUT_IMAGE_INDICATOR,
    )

    length = pad + text + image
    indicator = torch.tensor([0] * pad + [LLM_TOKEN_INDICATOR] * text + [OUTPUT_IMAGE_INDICATOR] * image)
    with torch.no_grad():
        return model(
            hidden_states=torch.randn(1, length, 8),
            timestep=torch.full((1,), 0.5),
            encoder_hidden_states=torch.randn(1, length, 32),
            position_ids=torch.arange(length).expand(1, 3, length).transpose(1, 2),
            segment_ids=torch.zeros(1, length, dtype=torch.long),
            indicator=indicator[None],
            return_dict=False,
        )[0]


def test_ideogram4_marks_the_prompt_exact_for_a_block_selecting_backend():
    """Every layer's attention sees the prompt, which leads the joint sequence, as exact keys and
    exact query rows; an image-only forward, the unconditional branch, has none to mark."""
    from xfuser.core.attention.spec import EXACT_KV_TOKENS_KEY, EXACT_Q_TOKENS_KEY

    model = _tiny_ideogram4()
    seen = []

    def attend(query, key, value, attention_kwargs=None):
        seen.append(attention_kwargs)
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    model._xfuser_processor.attention_function = attend

    _forward(model, pad=3, text=5, image=16)
    assert len(seen) == 2
    for kwargs in seen:
        assert kwargs["valid_kv_len"] is None
        assert kwargs[EXACT_KV_TOKENS_KEY] is kwargs[EXACT_Q_TOKENS_KEY]
        assert kwargs[EXACT_KV_TOKENS_KEY].tolist() == [True] * 5 + [False] * 16

    seen.clear()
    del model._num_image_tokens
    _forward(model, pad=0, text=0, image=16)
    assert seen == [None, None]
