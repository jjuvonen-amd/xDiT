"""Should FLUX.3's prompt query rows be forced exact, beside its prompt KV blocks?

Usage (one GPU, from a checkout root): python3 flux3_exact_rows.py /output/flux3_dumps/call*.pt

On dumped joint-block operands (rank 0's heads, full sequence), for recipes bf16 and fp8 at beta
0.5: relative L2 error against fp32 dense attention on the prompt rows and the image rows, the
share of KV blocks computed exactly, and compiled per-call time, for
  none  - no forcing (the raw entry point);
  kv    - the prompt's KV blocks exact for every query (exact_kv_tokens);
  rows  - kv plus every KV block for the query tiles holding prompt rows (exact_q_tokens); what
          the FLUX.3 PR ships.
"""

import sys

import torch

from xfuser.core.attention.sparsity import sol


def dense(query, key, value, chunk=4096):
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


def timed(fn, *args, iters=10):
    for _ in range(2):
        fn(*args)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main(paths):
    print(f"{'call':>9} {'recipe':6} {'variant':5} {'blocks':>6} {'err all':>8} {'prompt':>8} {'image':>8} {'ms':>7}")
    for path in paths:
        dump = torch.load(path)
        n = dump["valid_kv_len"] or dump["key"].shape[2]
        q, k, v = (dump[name][:, :, :n].cuda() for name in ("query", "key", "value"))
        exact = dump["exact_tokens"][:n].cuda()
        prompt, image = exact.nonzero().squeeze(1), (~exact).nonzero().squeeze(1)
        ref = dense(q, k, v)
        name = path.rsplit("/", 1)[-1].removesuffix(".pt")
        for recipe in ("bf16", "fp8"):
            q_tile, kv_tile = sol.sol_attn_block_tile(sol._RECIPES[recipe])
            total = -(-n // q_tile) * -(-n // kv_tile)
            for variant in ("none", "kv", "rows"):
                forced = {
                    "exact_tokens": None if variant == "none" else exact,
                    "exact_query_tokens": exact if variant == "rows" else None,
                }

                def run(q, k, v):
                    return sol.sol_attn_bhsd(q, k, v, beta=0.5, recipe=recipe, **forced)[0]

                with torch.no_grad():
                    _, cost = sol.sol_attn_bhsd(q, k, v, beta=0.5, recipe=recipe, return_head_cost=True, **forced)
                    out = run(q, k, v)
                    torch._dynamo.reset()
                    ms = timed(torch.compile(run, fullgraph=True), q, k, v)
                share = cost.mean().item() / total
                print(
                    f"{name:>9} {recipe:6} {variant:5} {share:6.3f} {rel(out, ref, slice(None)):8.4f} "
                    f"{rel(out, ref, prompt):8.4f} {rel(out, ref, image):8.4f} {ms:7.2f}"
                )


if __name__ == "__main__":
    main(sys.argv[1:])
