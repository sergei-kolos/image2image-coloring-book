from __future__ import annotations

import cv2
import numpy as np

from .models import PaletteColor


def quantize(image: np.ndarray, palette_size: int):
    """Run k-means color quantization on an RGB uint8 image.

    Returns (palette, labels):
      palette: list[PaletteColor] of length palette_size (1-based indices)
      labels:  HxW int32 array of 0-based cluster ids
    """
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels_flat, centers = cv2.kmeans(
        pixels, palette_size, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    centers = centers.astype(np.uint8)

    palette = [
        PaletteColor(
            index=i + 1,
            hex=_to_hex(centers[i]),
            rgb=(int(centers[i][0]), int(centers[i][1]), int(centers[i][2])),
        )
        for i in range(palette_size)
    ]
    labels = labels_flat.reshape(h, w).astype(np.int32)
    return palette, labels


def _to_hex(rgb) -> str:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return "#%02X%02X%02X" % (r, g, b)
