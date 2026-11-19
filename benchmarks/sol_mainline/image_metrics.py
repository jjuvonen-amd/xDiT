"""Score each FLUX.3 matrix image against the dense image of the same model, size and prompt.

    python3 image_metrics.py /output/matrix [--csv out.csv]

Row ids are <model><size>_<prompt>_<attention>, e.g. b4k_hands_fp8sol; the reference is
<model><size>_<prompt>_dense. Per image:
- PSNR and SSIM at full resolution.
- LPIPS (AlexNet) on a 1024² downscale -- the image as a whole.
- LPIPS on the full-resolution 1024² center crop -- fine detail.
- s/image: the steady-state entry of timings.json.

The summary averages over prompts per model, size and attention. Read Sol rows against
`fp8dense`: dense FP8 attention is an ordinary numeric change, so it marks how far an image
drifts without any approximation in which keys are attended.
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

_LPIPS = None


def _lpips(a, b):
    global _LPIPS
    if _LPIPS is None:
        _LPIPS = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
    with torch.no_grad():
        return float(_LPIPS(a, b))


def _load(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0


def _down(x, size=1024):
    return torch.nn.functional.interpolate(x, size=(size, size), mode="area")


def _center(x, size=1024):
    h, w = x.shape[-2:]
    top, left = (h - size) // 2, (w - size) // 2
    return x[..., top:top + size, left:left + size]


def _image(row_dir):
    found = glob.glob(os.path.join(row_dir, "*.png"))
    return found[0] if found else None


def _seconds(row_dir):
    try:
        return json.load(open(os.path.join(row_dir, "timings.json")))[-1]
    except (OSError, ValueError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    rows = []
    for row_dir in sorted(glob.glob(os.path.join(args.matrix, "*"))):
        row_id = os.path.basename(row_dir)
        parts = row_id.split("_", 2)
        if len(parts) != 3 or _image(row_dir) is None:
            continue
        model_size, prompt, attention = parts
        row = {"id": row_id, "model_size": model_size, "prompt": prompt, "attention": attention,
               "s_per_image": _seconds(row_dir)}
        ref_dir = os.path.join(args.matrix, f"{model_size}_{prompt}_dense")
        if attention != "dense" and _image(ref_dir):
            x, ref = _load(_image(row_dir)), _load(_image(ref_dir))
            row["psnr"] = float(peak_signal_noise_ratio(x, ref, data_range=1.0))
            row["ssim"] = float(structural_similarity_index_measure(x, ref, data_range=1.0))
            row["lpips_whole"] = _lpips(_down(x), _down(ref))
            row["lpips_crop"] = _lpips(_center(x), _center(ref))
        rows.append(row)

    fields = ["id", "model_size", "prompt", "attention", "psnr", "ssim", "lpips_whole", "lpips_crop",
              "s_per_image"]
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["model_size"], row["attention"])].append(row)

    def mean(group, key):
        values = [r[key] for r in group if r.get(key) is not None]
        return sum(values) / len(values) if values else None

    def fmt(value, spec):
        return "" if value is None else format(value, spec)

    print("| model+size | attention | n | PSNR dB | SSIM | LPIPS whole | LPIPS crop | s/image |")
    print("|---|---|---|---|---|---|---|---|")
    order = ["dense", "fp8dense", "bf16sol", "fp8sol", "mxfp6sol"]
    for (model_size, attention), group in sorted(
            groups.items(), key=lambda kv: (kv[0][0], order.index(kv[0][1]) if kv[0][1] in order else 99, kv[0][1])):
        print(f"| {model_size} | {attention} | {len(group)} | {fmt(mean(group, 'psnr'), '.2f')} | "
              f"{fmt(mean(group, 'ssim'), '.4f')} | {fmt(mean(group, 'lpips_whole'), '.4f')} | "
              f"{fmt(mean(group, 'lpips_crop'), '.4f')} | {fmt(mean(group, 's_per_image'), '.2f')} |")


if __name__ == "__main__":
    main()
