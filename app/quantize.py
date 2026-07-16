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
    threshold: float = 5.0,
) -> tuple[list[PaletteColor], np.ndarray]:
    """Merge palette entries that are within *threshold* CIEDE2000 distance.

    Duplicates (distance == 0) are always merged.  Similar colors are merged
    using a greedy pass: colours are sorted by luminance, then the first
    colour anchors the group and all subsequent colours within *threshold*
    are absorbed.

    Returns (new_palette, new_labels) with re-indexed 0-based labels.
    """
    if len(palette) < 2:
        return palette, labels

    # Sort palette by luminance (L* in Lab) for deterministic merge order
    rgb_arr = np.array([[c.rgb for c in palette]], dtype=np.uint8)
    lab_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB)
    sort_order = np.argsort(lab_arr[0, :, 0])
    palette = [palette[i] for i in sort_order]
    remap_sorted = np.zeros(len(palette), dtype=np.int32)
    remap_sorted[sort_order] = np.arange(len(palette))
    labels = remap_sorted[labels]

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
            dist = _ciede2000(tuple(old_rgb.astype(int)), tuple(other_rgb.astype(int)))
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


def _rgb_to_lab(rgb: tuple[int, int, int]) -> np.ndarray:
    pixels = np.array([[rgb]], dtype=np.uint8)
    return cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float64)


def _ciede2000(rgb1: tuple[int, int, int], rgb2: tuple[int, int, int]) -> float:
    """CIEDE2000 color difference between two RGB colors."""
    lab1 = _rgb_to_lab(rgb1)
    lab2 = _rgb_to_lab(rgb2)
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    kL = 1.0
    kC = 1.0
    kH = 1.0

    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    C_avg = (C1 + C2) / 2.0

    G = 0.5 * (1.0 - np.sqrt(C_avg ** 7 / (C_avg ** 7 + 25.0 ** 7)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.sqrt(a1p ** 2 + b1 ** 2)
    C2p = np.sqrt(a2p ** 2 + b2 ** 2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0:
        dhp = 0.0
    else:
        if abs(h2p - h1p) <= 180.0:
            dhp = h2p - h1p
        elif h2p - h1p > 180.0:
            dhp = h2p - h1p - 360.0
        else:
            dhp = h2p - h1p + 360.0

    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    L_avg = (L1 + L2) / 2.0
    C_avg_p = (C1p + C2p) / 2.0

    if C1p * C2p == 0:
        h_avg_p = h1p + h2p
    else:
        if abs(h2p - h1p) <= 180.0:
            h_avg_p = (h1p + h2p) / 2.0
        elif h1p + h2p < 360.0:
            h_avg_p = (h1p + h2p + 360.0) / 2.0
        else:
            h_avg_p = (h1p + h2p - 360.0) / 2.0

    T = (
        1.0
        - 0.17 * np.cos(np.radians(h_avg_p - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_avg_p))
        + 0.32 * np.cos(np.radians(3.0 * h_avg_p + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_avg_p - 63.0))
    )

    dTheta = 30.0 * np.exp(-((h_avg_p - 275.0) / 25.0) ** 2)
    RC = 2.0 * np.sqrt(C_avg_p ** 7 / (C_avg_p ** 7 + 25.0 ** 7))
    SL = 1.0 + (0.015 * (L_avg - 50.0) ** 2) / np.sqrt(20.0 + (L_avg - 50.0) ** 2)
    SC = 1.0 + 0.045 * C_avg_p
    SH = 1.0 + 0.015 * C_avg_p * T

    RT = -np.sin(np.radians(2.0 * dTheta)) * RC

    return np.sqrt(
        (dLp / (kL * SL)) ** 2
        + (dCp / (kC * SC)) ** 2
        + (dHp / (kH * SH)) ** 2
        + RT * (dCp / (kC * SC)) * (dHp / (kH * SH))
    )


def _to_hex(rgb) -> str:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return "#%02X%02X%02X" % (r, g, b)
