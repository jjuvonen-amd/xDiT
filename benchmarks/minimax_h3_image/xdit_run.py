"""Run the xDiT runner with optional dumps and per-stage timings for MiniMax-H3-Image.

[DUMP_DIR=dir] [TIMING=1] [NO_FUSE=1] [NO_FLYDSL=1] torchrun ... xdit_run.py <xfuser.runner args>

DUMP_DIR: save every transformer call's kwargs + output (rank 0).
TIMING:   time the text encoder, each transformer call and the VAE decode
          with CUDA events and print a per-request breakdown on rank 0.
"""

import functools
import os
import sys

import torch
from diffusers import AutoencoderKLMiniMaxH3Image
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

from xfuser.model_executor.models.transformers.transformer_minimax_h3_image import (
    xFuserMiniMaxH3ImageTransformer2DWrapper as Wrapper,
)

RANK = int(os.environ.get("RANK", "0"))
dump_dir = os.environ.get("DUMP_DIR")
timing = bool(os.environ.get("TIMING"))
counter = {"step": 0}


events = []


def timed(stage):
    """Bracket each call with CUDA events, which time the GPU without syncing it;
    they are read once per request, after the VAE decode."""

    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            if not timing:
                return fn(*args, **kwargs)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(*args, **kwargs)
            end.record()
            events.append((stage, start, end))
            if stage == "vae":
                end.synchronize()
                totals = {"text": 0.0, "denoise": 0.0, "vae": 0.0}
                for name, s, e in events:
                    totals[name] += s.elapsed_time(e) / 1000
                steps = sum(1 for name, _, _ in events if name == "denoise")
                request = events[0][1].elapsed_time(end) / 1000
                if RANK == 0:
                    print(
                        f"[timing] request {request:.3f} s | text {totals['text']:.3f} s | denoise "
                        f"{totals['denoise']:.3f} s ({totals['denoise'] / max(steps, 1):.3f} s/step x {steps}) "
                        f"| vae {totals['vae']:.3f} s | peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                        flush=True,
                    )
                events.clear()
            return out

        return inner

    return wrap


forward = Wrapper.forward


teacher_dir = os.environ.get("TEACHER_DIR")


@functools.wraps(forward)
def dumping_forward(self, *args, **kwargs):
    if teacher_dir:
        # Teacher forcing: feed this step the reference's input and score the output
        # against the reference's, so a config's per-step error is measured without
        # the trajectory drifting. Steps are counted per request.
        step = kwargs["pdd_step_index"]
        ref = torch.load(os.path.join(teacher_dir, f"step{step}.pt"))
        device = kwargs["hidden_states"].device
        kwargs["hidden_states"] = ref["kwargs"]["hidden_states"].to(device)
        out = forward(self, *args, **kwargs)
        sample = (out[0] if isinstance(out, tuple) else out.sample).float().cpu()
        expected = ref["output"].float()
        if RANK == 0:
            rel = ((sample - expected).norm() / expected.norm()).item()
            print(f"[teacher] step {step}: rel-L2 {rel:.3e}", flush=True)
        return out
    out = forward(self, *args, **kwargs)
    if dump_dir and RANK == 0:
        os.makedirs(dump_dir, exist_ok=True)
        sample = out[0] if isinstance(out, tuple) else out.sample
        torch.save(
            {
                "kwargs": {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()},
                "output": sample.detach().cpu(),
            },
            os.path.join(dump_dir, f"step{counter['step']}.pt"),
        )
    counter["step"] += 1
    return out


Wrapper.forward = timed("denoise")(dumping_forward)
Qwen3VLModel.forward = timed("text")(Qwen3VLModel.forward)
AutoencoderKLMiniMaxH3Image.decode = timed("vae")(AutoencoderKLMiniMaxH3Image.decode)

# Exactness switches: the two optimizations that change bf16 rounding.
if os.environ.get("NO_FUSE"):
    Wrapper.fuse_qkv_projections = lambda self: None
if os.environ.get("NO_FLYDSL"):
    from xfuser.model_executor.models.transformers import transformer_minimax_h3, transformer_minimax_h3_image

    transformer_minimax_h3._HAS_FLYDSL = False
    transformer_minimax_h3_image._HAS_FLYDSL = False

from xfuser.model_executor.models.runner_models.minimax_h3_image import xFuserMiniMaxH3ImageModel  # noqa: E402

# A/B switches for optimizations that are not runner options.
if os.environ.get("COMPILE_MODE"):
    xFuserMiniMaxH3ImageModel._get_compile_mode = lambda self: os.environ["COMPILE_MODE"]
if os.environ.get("AITER_RMSNORM"):
    from xfuser.model_executor.layers.norms import _AITERRMSNorm

    load_model = xFuserMiniMaxH3ImageModel._load_model

    def load_model_with_aiter_norms(self):
        pipe = load_model(self)
        swapped = 0
        for block in pipe.transformer.transformer_blocks:
            for name in ("norm1", "norm2"):
                norm = getattr(block, name)
                setattr(block, name, _AITERRMSNorm(norm.weight, norm.eps, norm.weight.shape[0]))
                swapped += 1
        print(f"[ab] swapped {swapped} block RMSNorms for AITER rms_norm", flush=True)
        return pipe

    xFuserMiniMaxH3ImageModel._load_model = load_model_with_aiter_norms

from xfuser.runner import main  # noqa: E402

sys.argv = ["xfuser.runner", *sys.argv[1:]]
main()
