from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .models import PaletteColor

_KERNEL_3 = np.ones((3, 3), np.uint8)
_KERNEL_5 = np.ones((5, 5), np.uint8)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Advanced morphological cleaning of a binary region mask.

    Pipeline: median blur -> morphological open (remove spikes/needles) ->
    morphological close (fill holes) -> Gaussian blur + re-threshold
    (smooth pixel-level staircasing into paintable boundaries).
    """
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL_3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL_5)
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
    return (blurred > 0.5).astype(np.uint8)


def merge_small_regions(
    labels: np.ndarray, palette: list[PaletteColor], min_area_px: int
) -> np.ndarray:
    """Merge connected regions below min_area_px into their most color-similar neighbor.

    Uses a custom Region Adjacency Graph approach: for each iteration, build a
    segment map (unique ID per connected same-cluster region), find segments
    below the area threshold, and reassign their pixels to the adjacent segment
    with the closest palette color. Iterates until no small regions remain or
    a merge cap is reached.
    """
    labels = labels.copy()
    palette_rgb = [np.array(p.rgb, dtype=np.float64) for p in palette]

    for _ in range(10):
        seg_map, seg_clusters = _build_segment_map(labels)
        if not seg_clusters:
            break

        small_ids = [
            sid for sid, info in seg_clusters.items() if info[1] < min_area_px
        ]
        if not small_ids:
            break

        small_ids.sort(key=lambda sid: seg_clusters[sid][1])

        merged = False
        for sid in small_ids:
            seg_mask = seg_map == sid
            dilated = cv2.dilate(seg_mask.astype(np.uint8), _KERNEL_3)
            border = (dilated > 0) & (seg_map != sid) & (seg_map > 0)
            neighbor_ids = set(int(v) for v in np.unique(seg_map[border]))

            if not neighbor_ids:
                continue

            my_rgb = palette_rgb[seg_clusters[sid][0]]
            best_nid = min(
                neighbor_ids,
                key=lambda nid: float(
                    np.linalg.norm(my_rgb - palette_rgb[seg_clusters[nid][0]])
                ),
            )
            labels[seg_mask] = seg_clusters[best_nid][0]
            merged = True

        if not merged:
            break

    return labels


def _build_segment_map(labels: np.ndarray):
    """Assign a unique ID to each connected same-cluster region.

    Returns (seg_map, seg_info) where seg_info[sid] = (cluster_id, area_px).
    """
    seg_map = np.zeros_like(labels, dtype=np.int32)
    seg_info: dict[int, tuple[int, int]] = {}
    next_id = 1

    for cid in np.unique(labels):
        mask = (labels == cid).astype(np.uint8)
        n, comps = cv2.connectedComponents(mask)
        for i in range(1, n):
            region = comps == i
            area = int(region.sum())
            seg_map[region] = next_id
            seg_info[next_id] = (int(cid), area)
            next_id += 1

    return seg_map, seg_info


def chaikin_smooth(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Apply Chaikin's corner-cutting algorithm to smooth a closed contour.

    Each iteration replaces every point pair (P, Q) with two points at
    3/4·P + 1/4·Q and 1/4·P + 3/4·Q, cutting sharp corners into rounded
    curves. Produces smooth, paintable outlines suitable for print.
    """
    pts = np.asarray(points, dtype=np.float64)
    for _ in range(iterations):
        n = len(pts)
        if n < 4:
            break
        p0 = pts
        p1 = np.roll(pts, -1, axis=0)
        q = 0.75 * p0 + 0.25 * p1
        r = 0.25 * p0 + 0.75 * p1
        pts = np.empty((n * 2, 2), dtype=np.float64)
        pts[0::2] = q
        pts[1::2] = r
    return pts


def adaptive_smooth(
    points: np.ndarray,
    max_angle_deg: float = 130,
    sigma: float = 1.0,
    max_displacement: float = 1.5,
) -> np.ndarray:
    """Smooth a closed contour while preserving sharp corners and limiting displacement.

    Three protections against topology drift:

    1. **Corner preservation:** points where the turn angle is below
       *max_angle_deg* (e.g. 90° phone corners, sharp facial features)
       are treated as structural anchors and kept exactly — no rounding.
    2. **Bounded Gaussian smoothing:** circular 1-D Gaussian filter on
       x/y coordinates smooths pixel-level staircasing between anchors.
    3. **Displacement clamp:** no point moves more than *max_displacement*
       pixels from its original position, preventing drift.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 4:
        return pts

    anchor_mask = _detect_sharp_corners(pts, max_angle_deg)

    sx = gaussian_filter1d(pts[:, 0], sigma=sigma, mode="wrap")
    sy = gaussian_filter1d(pts[:, 1], sigma=sigma, mode="wrap")
    smoothed = np.column_stack([sx, sy])

    smoothed[anchor_mask] = pts[anchor_mask]

    disp = np.linalg.norm(smoothed - pts, axis=1)
    exceed = (~anchor_mask) & (disp > max_displacement)
    if np.any(exceed):
        scale = max_displacement / disp[exceed]
        smoothed[exceed] = pts[exceed] + (smoothed[exceed] - pts[exceed]) * scale[:, np.newaxis]

    return smoothed


def _detect_sharp_corners(pts: np.ndarray, max_angle_deg: float) -> np.ndarray:
    """Return boolean mask: True at points whose turn angle < max_angle_deg.

    The turn angle is the angle between the vectors to the previous and next
    points. A 90° rectangle corner has turn angle ≈ 90° (sharp, preserved).
    A point on a smooth curve has turn angle ≈ 180° (not preserved).
    """
    prev = np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0)

    v1 = prev - pts
    v2 = nxt - pts

    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    dot = np.einsum("ij,ij->i", v1, v2)
    cos_angle = np.clip(dot / (n1 * n2 + 1e-10), -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_angle))

    mask = angles < max_angle_deg
    if mask.sum() < 3:
        step = max(1, len(pts) // 8)
        mask = np.zeros(len(pts), dtype=bool)
        mask[::step] = True
    return mask
