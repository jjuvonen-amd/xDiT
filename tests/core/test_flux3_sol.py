"""FLUX.3 under Sol-Attn: the short text stage stays dense and the joint blocks keep the prompt exact.

CPU only. The attention call itself is replaced, so these check what the processor asks for --
which backend, which Sol keys -- not what any kernel returns.
"""

import pytest
import torch

transformer_flux3 = pytest.importorskip("xfuser.model_executor.models.transformers.transformer_flux3")

from xfuser.core.distributed.attention_backend import (  # noqa: E402
    SOL_EXACT_QUERY_TOKENS_KEY,
    SOL_EXACT_TOKENS_KEY,
)

HEADS, HEAD_DIM = 2, 128
CROSS_BACKEND = object()


def test_the_joint_exact_tokens_are_the_text_prefix_over_the_padded_sequence():
    tokens = transformer_flux3._joint_exact_tokens(txt_seq_len=80, joint_seq_len=4176, pad=4, device="cpu")
    assert tokens.shape == (4180,)
    assert tokens[:80].all()
    assert not tokens[80:].any()


@pytest.fixture
def calls(monkeypatch):
    """Record every attention call the processor makes, answering with zeros."""
    recorded = []

    def fake(query, key, value, *args, backend=None, attention_kwargs=None, **kwargs):
        recorded.append({"backend": backend, "attention_kwargs": attention_kwargs})
        return torch.zeros_like(query)

    class _State:
        def get_cross_attention_backend(self):
            return CROSS_BACKEND

    monkeypatch.setattr(transformer_flux3, "USP", fake)
    monkeypatch.setattr(transformer_flux3, "attention", fake)
    monkeypatch.setattr(transformer_flux3, "get_runtime_state", lambda: _State())
    return recorded


def _run(processor, seq_len=16):
    from diffusers.models.transformers.transformer_flux3 import Flux3Attention

    attn = Flux3Attention(query_dim=HEADS * HEAD_DIM, heads=HEADS, dim_head=HEAD_DIM)
    with torch.no_grad():
        processor(attn, torch.randn(1, seq_len, HEADS * HEAD_DIM))


@pytest.mark.parametrize("local", [False, True])
def test_the_text_stage_runs_on_the_cross_attention_backend(calls, local):
    processor = transformer_flux3.xFuserFlux3AttnProcessor(text_only=True)
    processor.local = local
    _run(processor)
    assert calls[-1]["backend"] is CROSS_BACKEND


def test_the_other_stages_keep_the_main_backend(calls):
    _run(transformer_flux3.xFuserFlux3AttnProcessor())
    assert calls[-1]["backend"] is None
    assert calls[-1]["attention_kwargs"] is None


def test_the_joint_stage_hands_the_prompt_to_sol_on_both_sides(calls):
    processor = transformer_flux3.xFuserFlux3AttnProcessor()
    processor.exact_tokens = transformer_flux3._joint_exact_tokens(4, 16, 0, "cpu")
    processor.valid_kv_len = 15
    _run(processor)
    kwargs = calls[-1]["attention_kwargs"]
    assert kwargs["valid_kv_len"] == 15
    assert kwargs[SOL_EXACT_TOKENS_KEY] is processor.exact_tokens
    assert kwargs[SOL_EXACT_QUERY_TOKENS_KEY] is processor.exact_tokens
