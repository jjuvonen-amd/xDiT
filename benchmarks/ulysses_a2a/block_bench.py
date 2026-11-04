"""Time a few real transformer blocks under Ulysses with each all-to-all transport.

    torchrun --nproc_per_node 8 benchmarks/ulysses_a2a/block_bench.py --model wan \
        [--backend AITER] [--layers 2] [--variants rccl nocomm sdma] [--head-groups 1 2] \
        [--compile] [--output /output/step1/wan.md]

Models, at the shapes of their benchmark runs, with random bf16 weights and the
number of transformer blocks cut to --layers:
  wan      Wan2.2-T2V-A14B, 1280x720, 81 frames (75,600 tokens), Ulysses 8
  minimax  MiniMax-H3 t2va, 1344x768, 124 frames (video, stereo audio and 256 text
           tokens in one packed sequence), Ulysses 8
  flux3    FLUX.3 klein base, --resolution 2048 or 4096 square, 512 text tokens, CFG 2 x
           Ulysses 4 (both CFG groups run the same forward at once, as their traffic does)

The xDiT wrapper of each model runs its whole forward (embeddings, the blocks, the
output projection and the sequence-parallel gather), so the blocks see exactly what
the runners hand them.

Variants:
  rccl     USP's all-to-alls on RCCL, as shipped
  nocomm   each all-to-all replaced by a local copy of the same size: what the blocks
           cost with the exchange free. Its output is meaningless.
  sdma     the Ulysses all-to-alls on SDMA engines (xfuser/core/distributed/sdma), once
           for each --head-groups value
  sdmac    the same with each head group attending on a stream of its own

`exposed` = (variant - nocomm) / variant: the share of the forward spent waiting for
the exchange. Every sdma variant is checked against rccl's output. Variants take turns
every iteration. Time = the slowest rank's median forward, ranks lined up by a barrier
before each. `blocks` is
the span from the first block's start to the last block's end, slowest and fastest
rank: without per-layer exchanges (nocomm) the ranks drift apart and the forward's
closing all-gather absorbs the drift, so a spread there is the GPUs' speed spread.
"""

import argparse
import contextlib
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

HF = Path(os.environ.get("HF_HUB_CACHE", "/app/huggingface_models/hub"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["wan", "minimax", "flux3"])
    p.add_argument("--resolution", type=int, default=2048, help="flux3: square image side")
    p.add_argument("--backend", default="AITER", help="attention backend (AttentionBackendType name)")
    p.add_argument("--layers", type=int, default=2, help="transformer blocks (flux3: single-stream blocks)")
    p.add_argument("--variants", nargs="+", default=["rccl", "nocomm"], choices=["rccl", "nocomm", "sdma", "sdmac"])
    p.add_argument("--head-groups", type=int, nargs="+", default=[1, 2])
    p.add_argument(
        "--compile", action="store_true", help="torch.compile each block (default mode, the runners' comm reordering)"
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--profile", default="", help="directory for a torch profiler trace of rank 0, per variant")
    p.add_argument("--output", default="")
    return p.parse_args()


def _snapshot(repo: str) -> Path:
    snapshots = HF / f"models--{repo.replace('/', '--')}" / "snapshots"
    return next(snapshots.iterdir())


def _setup(args):
    from xfuser.config.args import xFuserArgs
    from xfuser.core.distributed import (
        get_runtime_state,
        init_distributed_environment,
        initialize_model_parallel,
        initialize_runtime_state,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    init_distributed_environment(rank=int(os.environ["RANK"]), world_size=world, local_rank=local_rank)
    cfg = 2 if args.model == "flux3" else 1
    ulysses = world // cfg
    initialize_model_parallel(classifier_free_guidance_degree=cfg, ulysses_degree=ulysses)
    engine_config, _ = xFuserArgs(
        attention_backend=args.backend, ulysses_degree=ulysses, use_cfg_parallel=cfg > 1
    ).create_config()
    initialize_runtime_state(engine_config=engine_config)
    runtime = get_runtime_state()
    runtime.set_attention_backend(args.backend)
    # The runners build a pipeline-backed runtime state; of it the blocks only need the step counter.
    if not hasattr(runtime, "increment_step_counter"):
        runtime.increment_step_counter = lambda: None
    return torch.device("cuda", local_rank), cfg, ulysses


def _build(model_cls, device, **config):
    # Same weights on every rank: built on the CPU from one seed.
    torch.manual_seed(0)
    model = model_cls.from_config(config) if config.pop("_from_config", True) else model_cls(**config)
    return model.to(device=device, dtype=torch.bfloat16).eval()


def wan(args, device):
    from xfuser.model_executor.models.transformers.transformer_wan import xFuserWanTransformer3DWrapper

    config = json.loads((_snapshot("Wan-AI/Wan2.2-T2V-A14B-Diffusers") / "transformer" / "config.json").read_text())
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    config["num_layers"] = args.layers
    model = _build(xFuserWanTransformer3DWrapper, device, **config)
    g = torch.Generator().manual_seed(1)
    inputs = {
        "hidden_states": torch.randn(1, 16, 21, 90, 160, generator=g),
        "timestep": torch.tensor([500.0]),
        "encoder_hidden_states": torch.randn(1, 512, 4096, generator=g),
    }
    tokens = 21 * 45 * 80
    return (
        model,
        {
            k: v.to(device, torch.bfloat16 if v.is_floating_point() and v.ndim > 1 else v.dtype)
            for k, v in inputs.items()
        },
        tokens,
        model.blocks,
    )


def minimax(args, device):
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        MINIMAX_H3_AUDIO_CHANNELS,
        MINIMAX_H3_AUDIO_TAG,
        MINIMAX_H3_TEXT_TAG,
        MINIMAX_H3_VIDEO_TAG,
        audio_latent_num_frames,
        video_latent_num_frames,
    )

    from xfuser.model_executor.models.transformers.transformer_minimax_h3 import xFuserMiniMaxH3Transformer3DWrapper

    model = _build(xFuserMiniMaxH3Transformer3DWrapper, device, _from_config=False, num_layers=args.layers)
    # As the runner does.
    model.fuse_qkv_projections()
    height, width, frames, text = 768, 1344, 124, 256
    latent_frames = video_latent_num_frames(frames, 17, 5)
    latent_h, latent_w = height // 16, width // 16
    audio_latents = audio_latent_num_frames(frames)
    patch = model.config.patch_size
    position_ids, token_tags, video_indices, audio_indices, text_indices, _, _ = (
        MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            torch.full((text,), MINIMAX_H3_TEXT_TAG),
            latent_frames,
            latent_h,
            latent_w,
            audio_latents,
            tuple(patch),
            MINIMAX_H3_AUDIO_CHANNELS,
            MINIMAX_H3_AUDIO_TAG,
            MINIMAX_H3_VIDEO_TAG,
        )
    )
    tokens = position_ids.shape[0]
    g = torch.Generator().manual_seed(1)
    inputs = {
        "hidden_states": torch.randn(1, video_indices.numel(), model.proj_in.in_features, generator=g),
        "audio_hidden_states": torch.randn(1, audio_indices.numel(), model.audio_proj_in.in_features, generator=g),
        "encoder_hidden_states": torch.randn(1, text, model.context_embedder.in_features, generator=g),
        "timestep": torch.tensor([500.0]),
        "timestep_indices": torch.zeros(tokens, dtype=torch.long),
        "token_tags": token_tags,
        "position_ids": position_ids,
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "text_indices": text_indices,
    }
    floats = ("hidden_states", "audio_hidden_states", "encoder_hidden_states")
    inputs = {k: v.to(device, torch.bfloat16) if k in floats else v.to(device) for k, v in inputs.items()}
    inputs["return_dict"] = False
    return model, inputs, tokens, model.transformer_blocks


def flux3(args, device):
    from xfuser.model_executor.models.transformers.transformer_flux3 import xFuserFlux3Transformer2DWrapper

    root = _snapshot("black-forest-labs/flux-3-image-klein-base") / "transformer"
    config = {k: v for k, v in json.loads((root / "config.json").read_text()).items() if not k.startswith("_")}
    config["num_layers"] = 1
    config["num_single_layers"] = args.layers
    model = _build(xFuserFlux3Transformer2DWrapper, device, **config)
    side = args.resolution // 16
    text = 512
    g = torch.Generator().manual_seed(1)

    def ids(height, width):
        grid = torch.meshgrid(
            torch.arange(1), torch.arange(height), torch.arange(width), torch.tensor([0]), indexing="ij"
        )
        return torch.stack(grid, dim=-1).reshape(-1, 4).float()

    inputs = {
        "hidden_states": torch.randn(1, side * side, config["in_channels"], generator=g).to(device, torch.bfloat16),
        "encoder_hidden_states": torch.randn(1, text, config["joint_attention_dim"], generator=g).to(
            device, torch.bfloat16
        ),
        "timestep": torch.tensor([0.7], device=device),
        "img_ids": ids(side, side).to(device),
        "txt_ids": torch.stack([torch.zeros(text)] * 3 + [torch.arange(text).float()], dim=-1).to(device),
    }
    if model.config.get("flow_map", False):
        inputs["target_timestep"] = torch.tensor([0.5], device=device)
    blocks = [*model.txt_mode_blocks, *model.mode_blocks["image"], *model.single_transformer_blocks]
    return model, inputs, side * side + text, blocks


def _event():
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


@contextlib.contextmanager
def variant(name, head_groups):
    """Run the block forward with the named all-to-all transport."""
    import xfuser.model_executor.layers.usp as usp

    if name == "rccl":
        yield
    elif name == "nocomm":
        real = usp._sdpa_all_to_all_single
        usp._sdpa_all_to_all_single = lambda x: x.clone()
        try:
            yield
        finally:
            usp._sdpa_all_to_all_single = real
    else:
        from xfuser.model_executor.layers import usp_sdma

        previous = usp_sdma.configure("sdma", head_groups, concurrent=name == "sdmac")
        try:
            yield
        finally:
            usp_sdma.configure(*previous)


def main():
    args = parse_args()
    device, cfg, ulysses = _setup(args)
    rank, world = dist.get_rank(), dist.get_world_size()
    model, inputs, tokens, blocks = {"wan": wan, "minimax": minimax, "flux3": flux3}[args.model](args, device)
    if args.compile:
        # As the runners compile: Inductor may move collectives against compute
        # (xFuserModel._enable_compute_comm_overlap).
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        torch._inductor.config.reorder_for_compute_comm_overlap_passes = [
            "reorder_compute_for_overlap",
            "sink_waits",
            "raise_comms",
        ]
        # Every variant is its own set of guards on every block.
        torch._dynamo.config.recompile_limit = 64
        torch._dynamo.config.accumulated_recompile_limit = 4096
        for block in blocks:
            block.compile()

    lines = []

    def say(text=""):
        if rank == 0:
            print(text, flush=True)
            lines.append(text)

    props = torch.cuda.get_device_properties(device)
    what = f"{args.model}" + (f" {args.resolution}²" if args.model == "flux3" else "")
    say(
        f"# {what}: {args.layers} block(s), {tokens} tokens, CFG {cfg} x Ulysses {ulysses}, {args.backend}"
        f"{', compiled' if args.compile else ''}; {world} x {props.name}, torch {torch.__version__}"
    )
    say()
    say("| variant | forward ms | exposed | blocks ms, slowest / fastest rank | max abs diff vs rccl |")
    say("|---|---|---|---|---|")

    # Events at the first block's start and the last block's end, per forward.
    marks = []
    blocks[0].register_forward_pre_hook(lambda *_: marks.append(_event()))
    blocks[-1].register_forward_hook(lambda *_: marks.append(_event()))

    def forward():
        with torch.no_grad():
            out = model(**inputs)
        return out[0] if isinstance(out, (tuple, list)) else out.sample

    runs = [("rccl", 0)] if "rccl" in args.variants else []
    runs += [("nocomm", 0)] if "nocomm" in args.variants else []
    runs += [("sdma", h) for h in args.head_groups] if "sdma" in args.variants else []
    runs += [("sdmac", h) for h in args.head_groups if h > 1] if "sdmac" in args.variants else []
    labels = [name if name not in ("sdma", "sdmac") else f"{name} g{h}" for name, h in runs]

    def timed(run):
        """One forward under `run`: (forward events, block-span events, output)."""
        with variant(*run):
            dist.barrier()
            marks.clear()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            out = forward()
            end.record()
            return (start, end), (marks[0], marks[-1]), out

    # Variants take turns every iteration, so clock and thermal drift falls on all of them alike.
    for _ in range(args.warmup):
        for run in runs:
            timed(run)
    times = {label: [] for label in labels}
    outputs = {}
    for _ in range(args.iters):
        for label, run in zip(labels, runs):
            fwd, span, outputs[label] = timed(run)
            times[label].append((fwd, span))
    torch.cuda.synchronize()

    results = {}
    reference = outputs["rccl"].float() if "rccl" in outputs else None
    per_rank = {}
    for label, run in zip(labels, runs):
        ms = statistics.median(s.elapsed_time(e) for (s, e), _ in times[label])
        block_ms = statistics.median(s.elapsed_time(e) for _, (s, e) in times[label])
        worst = torch.tensor([ms, block_ms, -block_ms], device=device)
        dist.all_reduce(worst, op=dist.ReduceOp.MAX)
        every = [torch.zeros(1, device=device) for _ in range(world)]
        dist.all_gather(every, torch.tensor([block_ms], device=device))
        per_rank[label] = [round(t.item(), 1) for t in every]
        results[label] = worst[0].item()
        results[label + " blocks"] = f"{worst[1].item():.2f} / {-worst[2].item():.2f}"
        diff = ""
        if run[0] in ("sdma", "sdmac") and reference is not None:
            d = torch.tensor([(outputs[label].float() - reference).abs().max().item()], device=device)
            dist.all_reduce(d, op=dist.ReduceOp.MAX)
            diff = f"{d.item():.3g}"
        results[label + " diff"] = diff
        if args.profile:
            from torch.profiler import ProfilerActivity, profile

            with variant(*run), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                forward()
                torch.cuda.synchronize()
            if rank == 0:
                Path(args.profile).mkdir(parents=True, exist_ok=True)
                prof.export_chrome_trace(str(Path(args.profile) / f"{label.replace(' ', '_')}.json"))
    base = results.get("nocomm")
    for label in [k for k in results if not k.endswith((" diff", " blocks"))]:
        exposed = "" if base is None or label == "nocomm" else f"{(results[label] - base) / results[label]:.1%}"
        say(
            f"| {label} | {results[label]:.2f} | {exposed} | {results[label + ' blocks']} | {results[label + ' diff']} |"
        )
    say()
    say("Blocks ms per rank: " + "; ".join(f"{label} {times}" for label, times in per_rank.items()))

    if args.output and rank == 0:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text("\n".join(lines) + "\n")
    dist.barrier()


if __name__ == "__main__":
    main()
