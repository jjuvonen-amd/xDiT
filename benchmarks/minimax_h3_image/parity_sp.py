"""Teacher-forced parity of the xDiT MiniMax-H3-Image wrapper against reference dumps.

torchrun --nproc-per-node N parity_sp.py --dump /output/ref/t2i_2048/r1 [--backend SDPA] [--no-flydsl]

Every rank loads the wrapper, runs Ulysses N on each dumped step's exact inputs and
rank 0 prints rel-L2 / max-abs of the output against the dumped diffusers output.
"""

import argparse
import glob
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

MODEL = "MiniMaxAI/MiniMax-H3-Image"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True)
    parser.add_argument("--backend", default="SDPA")
    parser.add_argument("--no-flydsl", action="store_true")
    parser.add_argument("--no-fuse", action="store_true")
    args = parser.parse_args()

    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=rank)
    initialize_model_parallel(ulysses_degree=world_size)

    from xfuser.model_executor.models.transformers import transformer_minimax_h3, transformer_minimax_h3_image
    from xfuser.model_executor.models.transformers.transformer_minimax_h3_image import (
        xFuserMiniMaxH3ImageTransformer2DWrapper,
    )

    if args.no_flydsl:
        transformer_minimax_h3._HAS_FLYDSL = False
        transformer_minimax_h3_image._HAS_FLYDSL = False

    model = xFuserMiniMaxH3ImageTransformer2DWrapper.from_pretrained(
        MODEL,
        subfolder="transformer_diffusers",
        dtype=torch.bfloat16,
        local_files_only=True,
        attention_backend=AttentionBackendType[args.backend],
    )
    engine_config, _ = xFuserArgs(attention_backend=args.backend, ulysses_degree=world_size).create_config()
    initialize_runtime_state(SimpleNamespace(transformer=model), engine_config=engine_config)
    get_runtime_state().set_attention_backend(args.backend)
    model = model.to(f"cuda:{rank}").eval()
    if not args.no_fuse:
        model.fuse_qkv_projections()

    for path in sorted(glob.glob(os.path.join(args.dump, "step*.pt"))):
        dump = torch.load(path)
        kwargs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in dump["kwargs"].items()}
        with torch.no_grad():
            out = model(**kwargs)[0].float().cpu()
        ref = dump["output"].float()
        if rank == 0:
            rel = ((out - ref).norm() / ref.norm()).item()
            print(
                f"u{world_size} {args.backend} flydsl={not args.no_flydsl} fused={not args.no_fuse} "
                f"{os.path.basename(path)}: rel-L2 {rel:.3e} max-abs {(out - ref).abs().max().item():.3e}",
                flush=True,
            )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
