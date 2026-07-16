from __future__ import annotations

import cv2
import numpy as np

from .models import PaletteColor


def quantize(image: np.ndarray, palette_size: int):
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels_flat, centers_lab = cv2.kmeans(
        pixels, palette_size, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    centers_lab = centers_lab.reshape(-1, 1, 3).astype(np.uint8)
    centers_rgb = cv2.cvtColor(centers_lab, cv2.COLOR_LAB2RGB).reshape(-1, 3)

    palette = [
        PaletteColor(
            index=i + 1,
            hex=_to_hex(centers_rgb[i]),
            rgb=(int(centers_rgb[i][0]), int(centers_rgb[i][1]), int(centers_rgb[i][2])),
        )
        for i in range(palette_size)
    ]
    labels = labels_flat.reshape(h, w).astype(np.int32)
    return palette, labels


def merge_similar_colors(
    palette: list[PaletteColor],
    labels: np.ndarray,
    threshold: float = 15.0,
) -> tuple[list[PaletteColor], np.ndarray]:
    """Merge palette entries that are within *threshold* Euclidean RGB distance.

    Duplicates (distance == 0) are always merged.  Similar colors are merged
    using a greedy pass: the first colour in the palette acts as the "anchor",
    and every subsequent colour within *threshold* is absorbed into it.

    Returns (new_palette, new_labels) with re-indexed 0-based labels.
    """
    if len(palette) < 2:
        return palette, labels

    rgb_array = np.array([c.rgb for c in palette], dtype=np.float64)

    old_to_new: dict[int, int] = {}
    new_colors: list[tuple[int, int, int]] = []
    new_index = 0

    for old_idx in range(len(palette)):
        if old_idx in old_to_new:
            continue

        old_rgb = rgb_array[old_idx]
        old_to_new[old_idx] = new_index

        # Collect all colors that will merge into this new entry
        merged_rgbs = [old_rgb]
        for other_idx in range(old_idx + 1, len(palette)):
            if other_idx in old_to_new:
                continue
            other_rgb = rgb_array[other_idx]
            dist = float(np.linalg.norm(old_rgb - other_rgb))
            if dist <= threshold:
                old_to_new[other_idx] = new_index
                merged_rgbs.append(other_rgb)

        # Use centroid (mean) of all merged colors, not just the first
        centroid_rgb = np.mean(merged_rgbs, axis=0)
        new_colors.append((
            int(round(centroid_rgb[0])),
            int(round(centroid_rgb[1])),
            int(round(centroid_rgb[2])),
        ))

        new_index += 1

    new_palette = [
        PaletteColor(index=i + 1, hex=_to_hex(rgb), rgb=rgb)
        for i, rgb in enumerate(new_colors)
    ]

    remap = np.array([old_to_new.get(i, i) for i in range(len(palette))], dtype=np.int32)
    new_labels = remap[labels]

    return new_palette, new_labels


def _to_hex(rgb) -> str:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return "#%02X%02X%02X" % (r, g, b)
