"""Find which length (query or key) breaks AITER MHA v4 dense, per format."""
import torch
from aiter.ops.mha_v4 import mha_v4  # noqa: F401
from xfuser.core.attention.backends.aiter_mha_v4 import kernel
from xfuser.core.attention.backends.aiter_mha_v4.spec import FORMATS

torch.manual_seed(0)
def run(fmt, sq, sk):
    q = torch.randn(1, sq, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, sk, 8, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, sk, 8, 128, device="cuda", dtype=torch.bfloat16)
    out = kernel._launch(q, k, v, fmt)
    out = out[0] if isinstance(out, tuple) else out
    ref = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float()).transpose(1, 2)
    return ((out.float() - ref).norm() / ref.norm()).item()
for fmt in FORMATS:
    if not fmt.dense:
        continue
    try:
        row = {f"q{sq}k{sk}": run(fmt, sq, sk) for sq, sk in [(53, 53), (64, 64), (65, 65), (53, 4096), (4096, 53), (4096, 64), (4096, 65), (1, 1)]}
        print(fmt.name, " ".join(f"{k}={v:.2e}" for k, v in row.items()), flush=True)
    except Exception as e:
        print(fmt.name, "error", type(e).__name__, str(e)[:120], flush=True)
