"""PSNR (dB) of images against a reference image: psnr.py ref.png a.png [b.png ...]"""

import sys

import numpy as np
from PIL import Image

reference = np.asarray(Image.open(sys.argv[1]).convert("RGB"), dtype=np.float64)
for path in sys.argv[2:]:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    if image.shape != reference.shape:
        print(f"{path}: shape {image.shape} != {reference.shape}")
        continue
    mse = np.mean((image - reference) ** 2)
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
    print(f"{psnr:7.2f} dB  {path}")
