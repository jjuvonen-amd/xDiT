"""Compare xDiT attention backends against SDPA on one shape, with and without a trailing pad.

torchrun --nproc-per-node 1 attn_probe.py [--seq 4160 --valid 4149 --heads 56]
"""

import argparse
import os
from types import SimpleNamespace

import torch

from xfuser.config.args import xFuserArgs
from xfuser.core.attention.spec import AttentionBackendType
from xfuser.core.distributed import (
    get_runtime_state,
    init_distributed_environment,
    initialize_model_parallel,
    initialize_runtime_state,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=4160)
    parser.add_argument("--valid", type=int, nargs="*", default=[4149, 4160])
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--backends", nargs="*", default=["AITER", "AITER_BF16"])
    args = parser.parse_args()

    torch.cuda.set_device(0)
    init_distributed_environment(rank=0, world_size=1, local_rank=0)
    initialize_model_parallel()
    from xfuser.model_executor.layers.usp import attention

    config = SimpleNamespace(patch_size=(1, 2, 2), in_channels=24, num_attention_heads=args.heads, attention_head_dim=128)
    engine_config, _ = xFuserArgs(attention_backend="SDPA").create_config()
    initialize_runtime_state(SimpleNamespace(transformer=SimpleNamespace(config=config)), engine_config=engine_config)

    generator = torch.Generator(device="cuda").manual_seed(0)
    shape = (1, args.heads, args.seq, 128)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(3))
    for valid in args.valid:
        kwargs = {"indices_k": None, "cu_seqlens_k": None, "max_seqlen_k": None, "valid_kv_len": None}
        if valid < args.seq:
            kwargs.update(
                indices_k=torch.arange(valid, device="cuda"),
                cu_seqlens_k=torch.tensor([0, valid], dtype=torch.int32, device="cuda"),
                max_seqlen_k=valid,
                valid_kv_len=valid,
            )
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k[:, :, :valid].float(), v[:, :, :valid].float()
        )
        for name in ["SDPA", *args.backends]:
            get_runtime_state().set_attention_backend(name)
            out = attention(
                q, k, v, attention_kwargs=dict(kwargs), backend=AttentionBackendType[name]
            ).float()
            rel = ((out - ref).norm() / ref.norm()).item()
            print(f"seq {args.seq} valid {valid} heads {args.heads} {name:12s} rel-L2 vs fp32 {rel:.3e}", flush=True)


if __name__ == "__main__":
    main()
