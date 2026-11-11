"""Time and profile MiniMax-H3-Image's prompt encoding alone, on one GPU.

python te_bench.py [--compile] [--profile out.json]
"""

import argparse
import time

import torch
from diffusers import ModularPipeline
from diffusers.modular_pipelines.minimax_h3_image import encoders

from xfuser.model_executor.models.runner_models.flux3 import trim_text_encoder

PROMPT = (
    "An overhead photograph of a rustic farm-to-table spread on a weathered wooden table: a round loaf of sourdough "
    "on burlap, wedges of cheese on a dark board, a copper bowl of vegetables, vine tomatoes, olives and walnuts."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--profile", default=None)
    args = parser.parse_args()

    pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3-Image")
    pipe.load_components(
        names=["text_encoder", "tokenizer", "processor"],
        dtype=torch.bfloat16,
        tokenizer_type={"processor": "qwen2"},
        local_files_only=True,
    )
    text_encoder = pipe.text_encoder.to("cuda").eval()
    trim_text_encoder(text_encoder, 50)
    if args.compile:
        text_encoder.model.language_model.compile()

    token_ids = pipe.tokenizer(PROMPT)["input_ids"]
    print(f"{len(token_ids)} tokens")

    def encode():
        return encoders.get_qwen3vl_prompt_embeds(
            text_encoder, pipe.processor, token_ids, device=torch.device("cuda"), dtype=torch.bfloat16
        )

    with torch.no_grad():
        for _ in range(3):
            encode()
        torch.cuda.synchronize()
        times = []
        for _ in range(10):
            start = time.perf_counter()
            encode()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        print(f"compile={args.compile}: median {sorted(times)[5] * 1000:.1f} ms, min {min(times) * 1000:.1f} ms")
        if args.profile:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                encode()
                torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))
            prof.export_chrome_trace(args.profile)


if __name__ == "__main__":
    main()
