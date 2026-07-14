from __future__ import annotations

import cv2
import numpy as np

from .models import PaletteColor, Region

_CLOSE_KERNEL = np.ones((5, 5), np.uint8)
_SMOOTH_KERNEL = (9, 9)


def extract_regions(labels: np.ndarray, palette, min_region_area: float):
    """Extract paintable regions from a cluster label map.

    For each palette color, build a binary mask, smooth boundaries
    (morphological close + Gaussian blur + re-threshold to remove
    pixel-level staircasing), then take external contours as regions.
    Drops regions whose area < min_region_area percent of the image area.
    """
    total_pixels = labels.size
    min_px = total_pixels * (min_region_area / 100.0)

    regions = []
    for color in palette:
        cluster_id = color.index - 1  # 0-based
        mask = (labels == cluster_id).astype(np.uint8)
        if mask.sum() == 0:
            continue

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLOSE_KERNEL)
        mask = cv2.GaussianBlur(mask.astype(np.float32), _SMOOTH_KERNEL, 0)
        mask = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_px:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            epsilon = max(2.0, 0.01 * cv2.arcLength(contour, True))
            approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            regions.append(
                Region(
                    color_index=color.index,
                    area=int(area),
                    centroid=(cx, cy),
                    contour=approx,
                )
            )
    return regions
