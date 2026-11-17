"""Compare videos: frame montage, mean frame PSNR against a reference, and bitwise equality.

Usage: python3 compare.py --out montage.jpg --frames 0,40,80 REF.mp4 OTHER.mp4 [...]
Each row of the montage is one video (label = its parent directory); PSNR is against the first.
"""

import argparse
import os

import cv2
import numpy as np


def read(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    return np.stack(frames)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--out")
    parser.add_argument("--frames", default="0")
    parser.add_argument("--scale", type=float, default=0.25)
    args = parser.parse_args()

    videos = [read(v) for v in args.videos]
    ref = videos[0]
    rows = []
    for path, video in zip(args.videos, videos):
        label = os.path.basename(os.path.dirname(path))
        n = min(len(ref), len(video))
        per_frame = [psnr(ref[i], video[i]) for i in range(n)]
        finite = [p for p in per_frame if np.isfinite(p)]
        identical = len(ref) == len(video) and all(np.isinf(p) for p in per_frame)
        mean = np.mean(finite) if finite else float("inf")
        print(f"{label:40s} frames {len(video):4d}  PSNR vs ref {mean:6.2f} dB  frame0 {per_frame[0]:6.2f}  identical {identical}")
        if args.out:
            picks = [int(f) if int(f) >= 0 else len(video) + int(f) for f in args.frames.split(",")]
            tiles = [cv2.resize(video[i], None, fx=args.scale, fy=args.scale) for i in picks]
            row = np.concatenate(tiles, axis=1)
            cv2.putText(row, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            rows.append(row)
    if args.out:
        cv2.imwrite(args.out, np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("wrote", args.out)


if __name__ == "__main__":
    main()
