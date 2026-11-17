"""Per-call error of Sol-Attn on dumped MiniMax-H3 operands, with and without the text/audio blocks forced.

Usage (one GPU, in the container, from /work/xDiT-b):
    python3 /work/tools/h3_exact_tokens.py /output/h3_dumps/call*.pt

For each call: relative L2 error against fp32 dense attention over the real rows, split into the
rows exact_tokens flags (text + audio) and the rest (video), plus the share of KV blocks computed
exactly. beta 0.5, recipes bf16 and fp8.
"""

import sys

import torch

from xfuser.core.attention.sparsity.sol import sol_attn_bhsd


def dense(query, key, value, chunk=4096):
    """fp32 attention, BHSD, chunked over the query rows."""
    scale = query.shape[-1] ** -0.5
    k, v = key.float(), value.float()
    out = torch.empty(query.shape, dtype=torch.float32, device=query.device)
    for start in range(0, query.shape[2], chunk):
        q = query[:, :, start : start + chunk].float()
        out[:, :, start : start + chunk] = torch.softmax(q @ k.transpose(-1, -2) * scale, dim=-1) @ v
    return out


def rel(out, ref, rows):
    diff = (out.float() - ref)[:, :, rows]
    return (diff.norm() / ref[:, :, rows].norm()).item()


def main(paths):
    print(f"{'call':>10} {'step':>4} {'recipe':6} {'forced':6} {'blocks':>6} {'err all':>8} {'text+audio':>10} {'video':>8}")
    for path in paths:
        dump = torch.load(path)
        q, k, v = (dump[n].cuda() for n in ("query", "key", "value"))
        n = dump["valid_kv_len"] or k.shape[2]
        exact = dump["exact_tokens"].cuda()[:n]
        q, k, v = q[:, :, :n], k[:, :, :n], v[:, :, :n]
        ref = dense(q, k, v)
        minority, video = exact.nonzero().squeeze(1), (~exact).nonzero().squeeze(1)
        for recipe in ("bf16", "fp8"):
            for forced in (False, True):
                with torch.no_grad():
                    out, cost = sol_attn_bhsd(
                        q, k, v, beta=0.5, recipe=recipe, return_head_cost=True, exact_tokens=exact if forced else None
                    )
                q_tiles = -(-n // (256))
                kv_tiles = -(-n // (64 if recipe == "bf16" else 128))
                share = cost.mean().item() / (q_tiles * kv_tiles)
                name = path.rsplit("/", 1)[-1].removesuffix(".pt")
                print(
                    f"{name:>10} {str(dump['step']):>4} {recipe:6} {str(forced):6} {share:6.3f} "
                    f"{rel(out, ref, slice(None)):8.4f} {rel(out, ref, minority):10.4f} {rel(out, ref, video):8.4f}"
                )


if __name__ == "__main__":
    main(sys.argv[1:])
