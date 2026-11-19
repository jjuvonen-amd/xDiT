"""Labelled side-by-side images, for judging Sol-Attn against dense by eye.

    python3 image_montage.py out.jpg LABEL=image.png [LABEL=image.png ...] [--crop x,y,w,h] [--width 640]

Without --crop each image is shown whole, downscaled to --width. With --crop the same
full-resolution region of every image is shown, which is where fine detail (text, fingers,
fibres) gets judged. --cols wraps the tiles into a grid.
"""
import argparse

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("images", nargs="+", help="LABEL=path")
    parser.add_argument("--crop", default=None, help="x,y,w,h in full-resolution pixels")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--cols", type=int, default=0)
    args = parser.parse_args()

    tiles = []
    for spec in args.images:
        label, path = spec.split("=", 1)
        image = cv2.imread(path)
        if args.crop:
            x, y, w, h = map(int, args.crop.split(","))
            image = image[y:y + h, x:x + w]
        scale = args.width / image.shape[1]
        tile = cv2.resize(image, (args.width, round(image.shape[0] * scale)),
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
        cv2.putText(tile, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(tile, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)

    cols = args.cols or len(tiles)
    height = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, height - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(args.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == "__main__":
    main()
