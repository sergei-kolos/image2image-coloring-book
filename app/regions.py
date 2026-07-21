from __future__ import annotations

import cv2
import numpy as np

from .models import PaletteColor, Region, SharedEdge
from .postprocess import clean_mask, chaikin_smooth, organic_smooth

_BORDER_PADDING = 8        # px — clamp label positions away from edge
_R_MIN_NUMERIC = 6.0       # px — smallest inscribed-circle radius that fits a number
_LETTER_POOL = [chr(65 + i) for i in range(26)]
_LETTER_COUNTER: list[int] = [0]


def _next_letter() -> str:
    """Return the next letter label (A–Z, then Aa–Zz)."""
    n = _LETTER_COUNTER[0]
    _LETTER_COUNTER[0] += 1
    if n < 26:
        return _LETTER_POOL[n]
    m = n - 26
    if m < 26:
        return _LETTER_POOL[m] * 2
    return str(n)


def _chaikin_closed(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Chaikin corner‑cutting for a closed polygon (first == last)."""
    pts = np.asarray(points, dtype=np.float64)
    for _ in range(iterations):
        n = len(pts) - 1
        if n < 3:
            break
        p0 = pts[:-1]
        p1 = pts[1:]
        q = 0.75 * p0 + 0.25 * p1
        r = 0.25 * p0 + 0.75 * p1
        new_pts = np.empty((2 * n, 2), dtype=np.float64)
        new_pts[0::2] = q
        new_pts[1::2] = r
        pts = np.vstack([new_pts, new_pts[:1]])
    return pts


def _pole_of_inaccessibility(contour: np.ndarray, shape: tuple) -> tuple[tuple[float, float], float]:
    """Find the point inside *contour* maximally distant from all boundaries.

    Uses ``cv2.distanceTransform`` to locate the centre of the largest
    inscribed circle (the "pole of inaccessibility").  This guarantees the
    label number stays inside the paint region even for C‑shaped, concave,
    or highly elongated contours where the geometric centroid would fall
    outside the polygon.

    Returns ``((cx, cy), max_radius)``.  The coordinates are clamped to
    ``[_BORDER_PADDING, size - padding]`` so that numbers never drift under
    the page trim.  ``max_radius`` is the radius of the largest inscribed
    circle; when it falls below ``_R_MIN_NUMERIC == 5`` px the caller
    switches to a single‑letter label.
    """
    h, w = shape
    vis = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(vis, [contour.astype(np.int32)], -1, 255, -1)

    dist = cv2.distanceTransform(vis, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    if max_val > 0:
        x = float(max_loc[0])
        y = float(max_loc[1])
    else:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            x = M["m10"] / M["m00"]
            y = M["m01"] / M["m00"]
        else:
            x, y = float(contour[0, 0]), float(contour[0, 1])
    return ((max(x, _BORDER_PADDING), max(y, _BORDER_PADDING)), float(max_val))


def _simplify_and_smooth(pts: np.ndarray) -> np.ndarray:
    """Simplify with approxPolyDP then organic-smooth preserving corners.

    approxPolyDP (epsilon=0.5) removes pixel staircasing while keeping real
    geometry.  organic_smooth then rounds organic curves via Chaikin between
    detected sharp corners, preserving structural corners exactly.

    This prevents both edge clipping (corners stay put) and jagged contours
    (organic segments get smoothed).
    """
    pts_f32 = pts.astype(np.float32).reshape(-1, 1, 2)
    epsilon = 0.5
    simplified = cv2.approxPolyDP(pts_f32, epsilon, True).reshape(-1, 2)
    if len(simplified) >= 2 and not np.array_equal(simplified[0], simplified[-1]):
        simplified = np.vstack([simplified, simplified[:1]])
    if len(simplified) >= 4:
        smoothed = organic_smooth(simplified, max_angle_deg=130, iterations=2)
        if len(smoothed) >= 4:
            return smoothed
    return simplified


def _smooth_closed(pts: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Simplify (TC89) then Chaikin-smooth a closed contour.

    Uses a fixed *epsilon* of 0.5 px so that shared boundaries between
    adjacent regions are simplified *identically*, preventing 1–2 px gaps
    after vector rendering.
    """
    pts_f32 = pts.astype(np.float32).reshape(-1, 1, 2)
    epsilon = 0.5  # fixed — shared edges simplify identically
    simplified = cv2.approxPolyDP(pts_f32, epsilon, True).reshape(-1, 2)
    if len(simplified) >= 2 and not np.array_equal(simplified[0], simplified[-1]):
        simplified = np.vstack([simplified, simplified[:1]])
    if len(simplified) >= 4:
        simplified = _chaikin_closed(simplified, iterations=iterations)
    return simplified


def extract_shared_edges(labels: np.ndarray) -> list[SharedEdge]:
    """Shared edges disabled — renderer uses raster boundaries from the label
    map directly, which guarantees clean 1px lines with no double strokes and
    no missing segments. Returns empty list so renderers fall back to
    per-region contours for PDF (vector) output.
    """
    return []


def extract_regions(labels: np.ndarray, palette, morph_kernel: int = 3) -> list[Region]:
    """Extract every contour from the label map via vector contour model.

    Uses ``RETR_CCOMP`` + ``CHAIN_APPROX_TC89_KCOS`` to obtain clean
    exterior rings and holes, then smooths with closed‑polygon Chaikin.
    Each ``Region`` carries both its exterior (``.contour``) and its
    interior cut‑outs (``.holes``).
    """
    h, w = labels.shape
    regions: list[Region] = []
    _LETTER_COUNTER[0] = 0  # reset so labelling is deterministic

    for color in palette:
        cluster_id = color.index - 1
        mask = (labels == cluster_id).astype(np.uint8)
        if mask.sum() == 0:
            continue

        mask = clean_mask(mask, kernel_size=max(1, morph_kernel))

        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS
        )
        if hierarchy is None:
            continue
        hierarchy = hierarchy[0]

        for i, contour in enumerate(contours):
            if hierarchy[i][3] != -1:
                continue  # hole — handled together with its parent

            # Collect child holes.
            holes: list[np.ndarray] = []
            child = hierarchy[i][2]
            while child >= 0:
                holes.append(
                    _smooth_closed(contours[child].reshape(-1, 2), iterations=1)
                )
                child = hierarchy[child][0]

            exterior = _simplify_and_smooth(contour.reshape(-1, 2))
            centroid, max_radius = _pole_of_inaccessibility(exterior, (h, w))
            label = _next_letter() if max_radius < _R_MIN_NUMERIC else str(color.index)
            regions.append(
                Region(
                    color_index=color.index,
                    area=int(cv2.contourArea(
                        exterior.astype(np.float32).reshape(-1, 1, 2)
                    )),
                    centroid=centroid,
                    label=label,
                    contour=exterior,
                    max_radius=float(max_radius),
                    holes=holes,
                )
            )
    return regions
