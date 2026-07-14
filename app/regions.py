from __future__ import annotations

import cv2
import numpy as np

from .models import PaletteColor, Region
from .postprocess import adaptive_smooth, clean_mask

_SMALL_AREA = 5000
_MEDIUM_AREA = 20000


def extract_regions(labels: np.ndarray, palette, min_region_area: float):
    """Extract paintable regions from a cluster label map.

    For each palette color, build a binary mask, clean it morphologically
    (median -> open -> close -> blur), then take external contours.
    Each contour is simplified (approxPolyDP) then smoothed with
    adaptive_smooth — corner-preserving, displacement-limited, size-adaptive.
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

        mask = clean_mask(mask)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

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
            sigma, max_disp = _smoothing_params(area)
            smooth = adaptive_smooth(
                approx, sigma=sigma, max_displacement=max_disp
            )
            regions.append(
                Region(
                    color_index=color.index,
                    area=int(area),
                    centroid=(cx, cy),
                    contour=smooth,
                )
            )
    return regions


def _smoothing_params(area: float) -> tuple[float, float]:
    """Return (sigma, max_displacement) scaled to region size.

    Small regions (eyes, facial features) get gentle smoothing to preserve
    detail; large regions (background, clothes) get stronger cleanup.
    """
    if area < _SMALL_AREA:
        return 0.5, 0.8
    if area < _MEDIUM_AREA:
        return 0.8, 1.2
    return 1.5, 2.0
