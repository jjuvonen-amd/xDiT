"""Unmodified diffusers MiniMax-H3-Image run that dumps the tensors parity checks need.

python ref_run.py --out /output/ref/t2i_2048 [--image examples/reference1.png] [--repeats 2]

Per repeat r it writes r{r}/: step{i}.pt (transformer kwargs + output of every step),
latents.pt (the decoder input), image.png. The text encoder output is step0's
encoder_hidden_states.
"""

import argparse
import functools
import os
import time

import torch
from diffusers import ModularPipeline
from diffusers.utils import load_image

MODEL = "MiniMaxAI/MiniMax-H3-Image"
PROMPT = (
    "An overhead photograph of a rustic farm-to-table spread on a weathered wooden table: a round loaf of sourdough "
    "on burlap, wedges of cheese on a dark board, a copper bowl of vegetables, vine tomatoes, olives and walnuts."
)
EDIT_PROMPT = "Keep everything in <Picture 1> as it is, but remove the headphones and have the two hosts fist bump."


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--fuse-qkv", action="store_true")
    args = parser.parse_args()

    pipe = ModularPipeline.from_pretrained(MODEL)
    pipe.load_components(dtype=torch.bfloat16, tokenizer_type={"processor": "qwen2"}, local_files_only=True)
    missing = [name for name in pipe.pretrained_component_names if getattr(pipe, name, None) is None]
    assert not missing, f"components failed to load: {missing}"
    pipe.to("cuda")
    if args.fuse_qkv:
        pipe.transformer.fuse_qkv_projections()

    state = {"dir": None, "step": 0}
    forward = pipe.transformer.forward

    @functools.wraps(forward)
    def dumping_forward(*f_args, **kwargs):
        out = forward(*f_args, **kwargs)
        sample = out[0] if isinstance(out, tuple) else out.sample
        torch.save(
            {"kwargs": {k: cpu(v) for k, v in kwargs.items()}, "output": cpu(sample)},
            os.path.join(state["dir"], f"step{state['step']}.pt"),
        )
        state["step"] += 1
        return out

    pipe.transformer.forward = dumping_forward
    decode = pipe.vae.decode

    @functools.wraps(decode)
    def dumping_decode(z, *d_args, **d_kwargs):
        torch.save(cpu(z), os.path.join(state["dir"], "latents.pt"))
        return decode(z, *d_args, **d_kwargs)

    pipe.vae.decode = dumping_decode

    call = {
        "prompt": args.prompt or (EDIT_PROMPT if args.image else PROMPT),
        "height": args.height,
        "width": args.width,
        "output": "images",
    }
    if args.image:
        call["image"] = load_image(args.image)

    for repeat in range(args.repeats):
        state["dir"] = os.path.join(args.out, f"r{repeat}")
        state["step"] = 0
        os.makedirs(state["dir"], exist_ok=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        image = pipe(generator=torch.Generator().manual_seed(args.seed), **call)[0]
        torch.cuda.synchronize()
        print(f"repeat {repeat}: {time.perf_counter() - start:.1f} s, size {image.size}", flush=True)
        image.save(os.path.join(state["dir"], "image.png"))
    print(f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
