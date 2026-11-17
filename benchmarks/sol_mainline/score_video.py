"""Per-latent-frame artifact scores for Wan 81-frame videos.

Usage: python3 score_video.py VIDEO [VIDEO ...] [--ref REF_VIDEO]

Wan's causal VAE maps latent frame 0 to video frame 0 and latent t>0 to video frames 4t-3..4t,
so every metric is reported per latent frame. Edge latents (t=0, t=T-1) are where the Sol-Attn
artifacts showed up, hence the edge/mid ratios at the end of each table.

speckle: isolated bright points, i.e. connected components of 2-80 px whose gray level exceeds
         the 7x7 median by more than 60. Scene highlights also count, so read it as a ratio
         against the same video's mid frames, not as an absolute.
lapvar:  variance of the Laplacian, a proxy for high-frequency energy (over-sharpening).
psnr:    against --ref, frame by frame, averaged over the latent's frames.
"""
import argparse

import cv2
import numpy as np


def load(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    return frames


def latent_groups(num_frames):
    groups = [[0]]
    for start in range(1, num_frames, 4):
        groups.append(list(range(start, min(start + 4, num_frames))))
    return groups


def speckle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tophat = gray.astype(np.int16) - cv2.medianBlur(gray, 7).astype(np.int16) > 60
    _, _, stats, _ = cv2.connectedComponentsWithStats(tophat.astype(np.uint8))
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int(((areas >= 2) & (areas <= 80)).sum())


def lapvar(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def score(path, ref_frames=None):
    frames = load(path)
    groups = latent_groups(len(frames))
    rows = []
    for t, idx in enumerate(groups):
        row = {
            "t": t,
            "speckle": np.mean([speckle(frames[i]) for i in idx]),
            "lapvar": np.mean([lapvar(frames[i]) for i in idx]),
        }
        if ref_frames is not None:
            row["psnr"] = np.mean([psnr(frames[i], ref_frames[i]) for i in idx])
        rows.append(row)
    return rows


def summarize(rows):
    mid = rows[2:-2]
    out = {}
    for key in ("speckle", "lapvar", "psnr"):
        if key not in rows[0]:
            continue
        mid_mean = np.mean([r[key] for r in mid])
        out[key] = (rows[0][key], rows[-1][key], mid_mean)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--ref")
    parser.add_argument("--per-frame", action="store_true", help="print the per-latent table")
    args = parser.parse_args()

    ref_frames = load(args.ref) if args.ref else None
    header = f"{'video':<40} {'spk t0':>7} {'spk tN':>7} {'spk mid':>7} {'lap t0':>7} {'lap tN':>7} {'lap mid':>7}"
    if ref_frames is not None:
        header += f" {'psnr t0':>7} {'psnr tN':>7} {'psnr mid':>8}"
    summaries = []
    for path in args.videos:
        rows = score(path, ref_frames)
        if args.per_frame:
            print(path)
            for r in rows:
                print("  " + "  ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in r.items()))
        summaries.append((path, summarize(rows)))

    print(header)
    for path, s in summaries:
        name = "/".join(path.rstrip("/").split("/")[-2:])[:40]
        line = f"{name:<40}"
        for key in ("speckle", "lapvar", "psnr"):
            if key in s:
                line += "".join(f" {v:>7.1f}" for v in s[key])
        print(line)


if __name__ == "__main__":
    main()
