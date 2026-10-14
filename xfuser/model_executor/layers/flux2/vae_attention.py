"""Single-head VAE self-attention as two GEMMs and a fused softmax, for the FLUX.2 VAE decoder.

What it replaces
----------------
The FLUX.2 VAE decoder's mid-block runs one self-attention over every latent pixel with a single head of
dim 512: ``[1, 1, S, 512]`` with S = 16384 at 1024x1024 and 65536 at 2048x2048. Fused attention kernels are
rarely tuned for, or support at all, a head dim that large, so SDPA's flash path runs this shape far below the
GPU's GEMM throughput.

With one head, attention is ``softmax(q k^T * scale) v``: two large, well-shaped GEMMs plus a softmax. Done per
chunk of queries, the score block stays ``chunk x S`` instead of ``S x S``. Inside the compiled decoder Inductor
fuses the softmax into one pass (bf16 scores in, fp32 math, bf16 probabilities out); without that fusion this
path is *slower* than flash (~0.7x), so the processor is only installed together with the decoder compile.
Measured against SDPA's flash path at the two shapes, chunk 8192:

    1024x1024  S=16384   2.90 ms -> 1.99 ms   (1.46x)
    2048x2048  S=65536  49.8 ms  -> 30.1 ms   (1.66x)

These numbers come from the hardware the path was developed on; how SDPA handles head_dim 512 differs between
GPUs and backends, so re-measure before relying on the gain elsewhere.

Numerics
--------
Exact attention, but the scores are rounded to bf16 between the first GEMM and the softmax (flash keeps them in
fp32): relative error vs an fp32 reference is ~4.7e-3 against flash's ~2.3e-3. Probabilities are bf16 for the
second GEMM, as in flash kernels.

Anything else -- more than one head, an attention mask, cross-attention, short sequences -- goes to SDPA exactly
as diffusers' ``AttnProcessor2_0`` would.
"""

import torch
import torch.nn.functional as F

# Below this many tokens flash is fine and the extra launches aren't worth it.
CHUNKED_MIN_TOKENS = 8192
# Query rows per chunk: the score block is CHUNK x S bf16 (1 GiB at S=65536).
CHUNK = 8192


def chunked_single_head_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """``softmax(q k^T / sqrt(d)) v`` for ``[B, 1, S, D]`` tensors, one query chunk at a time."""
    scale = query.shape[-1] ** -0.5
    out = torch.empty_like(query)
    for b in range(query.shape[0]):
        q = query[b, 0] * scale
        k_t = key[b, 0].transpose(0, 1)
        v = value[b, 0]
        for start in range(0, q.shape[0], CHUNK):
            scores = torch.mm(q[start:start + CHUNK], k_t)
            probs = torch.softmax(scores.float(), dim=-1).to(query.dtype)
            out[b, 0, start:start + CHUNK] = torch.mm(probs, v)
    return out


class Flux2VAEAttnProcessor:
    """diffusers ``AttnProcessor2_0`` with the single-head, long-sequence case routed to the chunked path."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        is_self_attention = encoder_hidden_states is None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if (attn.heads == 1 and attention_mask is None and is_self_attention
                and query.shape[2] >= CHUNKED_MIN_TOKENS):
            hidden_states = chunked_single_head_attention(query, key, value)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor
