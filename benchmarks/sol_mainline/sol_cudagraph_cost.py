"""Per-call cost of Sol's raw and packed paths under torch.compile, with and without CUDA graphs.

Usage (one GPU, from a checkout root): python3 sol_cudagraph_cost.py DUMP.pt
"""

import sys

import torch

from xfuser.core.attention.sparsity.sol import sol_attn_bhsd


def timed(fn, *args, iters=20):
    for _ in range(3):
        fn(*args)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


dump = torch.load(sys.argv[1])
n = dump["valid_kv_len"] or dump["key"].shape[2]
q, k, v = (dump[name][:, :, :n].cuda() for name in ("query", "key", "value"))
exact = dump["exact_tokens"][:n].cuda()
print(f"q {tuple(q.shape)}")
for recipe in ("bf16", "fp8"):

    def raw(q, k, v):
        return sol_attn_bhsd(q, k, v, beta=0.5, recipe=recipe)[0]

    def packed(q, k, v, tokens):
        return sol_attn_bhsd(q, k, v, beta=0.5, recipe=recipe, exact_tokens=tokens, exact_query_tokens=tokens)[0]

    row = [recipe]
    for mode in ("default", "reduce-overhead"):
        torch._dynamo.reset()
        with torch.no_grad():
            row.append(f"{mode}: raw {timed(torch.compile(raw, mode=mode), q, k, v):.2f} ms")
            torch._dynamo.reset()
            row.append(f"packed {timed(torch.compile(packed, mode=mode), q, k, v, exact):.2f} ms")
    print("  ".join(row))
