import pytest
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

import xfuser.model_executor.layers.flux2.vae_attention as vae_attention
from xfuser.model_executor.layers.flux2.vae_attention import Flux2VAEAttnProcessor


def _vae_attention(heads):
    """A VAE-style spatial self-attention block (residual, rescale), as the FLUX.2 decoder builds it.

    No group norm: importing xfuser can swap torch.nn.GroupNorm for a GPU-only implementation, and the norm runs
    identically in both processors anyway.
    """
    torch.manual_seed(0)
    return Attention(
        query_dim=32,
        heads=heads,
        dim_head=32 // heads,
        rescale_output_factor=1.0,
        eps=1e-6,
        norm_num_groups=None,
        spatial_norm_dim=None,
        residual_connection=True,
        bias=True,
        upcast_softmax=True,
        _from_deprecated_attn_block=True,
    ).eval()


def _run(attn, processor, x):
    attn.set_processor(processor)
    with torch.no_grad():
        return attn(x)


@pytest.fixture
def small_chunks(monkeypatch):
    # 10x10 = 100 tokens with 32-row chunks: four chunks, the last one ragged.
    monkeypatch.setattr(vae_attention, "CHUNKED_MIN_TOKENS", 64)
    monkeypatch.setattr(vae_attention, "CHUNK", 32)


def test_single_head_long_sequence_matches_sdpa(small_chunks):
    attn = _vae_attention(heads=1)
    x = torch.randn(2, 32, 10, 10)
    expected = _run(attn, AttnProcessor2_0(), x)
    actual = _run(attn, Flux2VAEAttnProcessor(), x)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_chunked_path_is_taken_for_single_head(small_chunks, monkeypatch):
    calls = []
    real = vae_attention.chunked_single_head_attention
    monkeypatch.setattr(vae_attention, "chunked_single_head_attention", lambda *a: calls.append(1) or real(*a))
    _run(_vae_attention(heads=1), Flux2VAEAttnProcessor(), torch.randn(1, 32, 10, 10))
    assert calls


@pytest.mark.parametrize("heads, hw", [(2, 10), (1, 4)])
def test_other_shapes_fall_back_to_sdpa_exactly(small_chunks, monkeypatch, heads, hw):
    # Several heads, or fewer tokens than the threshold: SDPA, bit for bit what diffusers computes.
    monkeypatch.setattr(vae_attention, "chunked_single_head_attention", lambda *a: pytest.fail("chunked path used"))
    attn = _vae_attention(heads=heads)
    x = torch.randn(1, 32, hw, hw)
    torch.testing.assert_close(_run(attn, Flux2VAEAttnProcessor(), x), _run(attn, AttnProcessor2_0(), x),
                               rtol=0, atol=0)
