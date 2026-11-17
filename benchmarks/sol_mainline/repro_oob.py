"""Does the MXFP8 Sol row read past the pooled K scale (mean_k_scale)?

Places mean_k_scale's storage so it ends exactly at the end of a 2 MiB allocation (run with
PYTORCH_NO_CUDA_MEMORY_CACHING=1, so the page after it is unmapped), then launches mha_v4_packed
through xDiT's sol_attn_bhsd. A read past the storage faults the GPU.

Usage: repro_oob.py MODE TILE SLACK [RECIPE] [TENSOR]
  MODE    jensen | plain   (capped Jensen term, or the packed call with no second-order term)
  TILE    64 | default
  SLACK   bytes left after the storage inside the allocation
  RECIPE  Sol recipe (default mxfp8)
  TENSOR  packed-call keyword to relocate (default mean_k_scale)
"""

import sys

import torch

from xfuser.core.attention.sparsity import sol

mode, tile, slack = sys.argv[1], sys.argv[2], int(sys.argv[3])
recipe = sys.argv[4] if len(sys.argv) > 4 else "mxfp8"
name = sys.argv[5] if len(sys.argv) > 5 else "mean_k_scale"
packed = sol._AITER.packed


def at_end(t, slack):
    st = t.untyped_storage()
    n = st.nbytes()
    block = torch.zeros(2 << 20, dtype=torch.uint8, device=t.device)
    start = block.numel() - slack - n
    block[start : start + n].copy_(torch.empty(0, dtype=torch.uint8, device=t.device).set_(st, 0, (n,)))
    out = torch.empty(0, dtype=t.dtype, device=t.device)
    out.set_(block[start:].untyped_storage(), start // t.element_size() + t.storage_offset(), t.shape, t.stride())
    return out


def relocated(*args, **kwargs):
    if kwargs.get(name) is None:
        print("absent", name, flush=True)
    else:
        kwargs[name] = at_end(kwargs[name], slack)
        print(name, tuple(kwargs[name].shape), kwargs[name].stride(), flush=True)
    return packed(*args, **kwargs)


sol._AITER.packed = relocated
g = torch.Generator(device="cuda").manual_seed(0)
q, k, v = (torch.randn(1, 2, 2048, 128, generator=g, device="cuda").bfloat16() for _ in range(3))
options = dict(jensen=True) if mode == "jensen" else dict(return_head_cost=True)
out, _ = sol.sol_attn_bhsd(
    q, k, v, beta=0.5, recipe=recipe, block_tile=(64, 64) if tile == "64" else None, **options
)
torch.cuda.synchronize()
print("ok", bool(torch.isfinite(out).all()), flush=True)
