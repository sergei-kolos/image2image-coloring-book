from __future__ import annotations

import cv2
import numpy as np

from app.models import PaletteColor
from app.postprocess import (
    adaptive_smooth,
    chaikin_smooth,
    clean_mask,
    merge_small_regions,
)


def test_clean_mask_removes_isolated_noise():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 10:40] = 1
    mask[0, 0] = 1
    cleaned = clean_mask(mask)
    assert cleaned[0, 0] == 0
    assert cleaned[25, 25] == 1


def test_clean_mask_preserves_large_region():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 1
    cleaned = clean_mask(mask)
    assert cleaned.sum() > 2500


def test_merge_small_regions_absorbs_island():
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[:50, :] = 0
    labels[50:, :] = 1
    labels[70:74, 70:74] = 0

    palette = [
        PaletteColor(index=1, hex="#FF0000", rgb=(255, 0, 0)),
        PaletteColor(index=2, hex="#0000FF", rgb=(0, 0, 255)),
    ]

    merged = merge_small_regions(labels, palette, min_area_px=100)
    assert merged[71, 71] == 1


def test_merge_small_regions_keeps_large():
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[:50, :] = 0
    labels[50:, :] = 1

    palette = [
        PaletteColor(index=1, hex="#FF0000", rgb=(255, 0, 0)),
        PaletteColor(index=2, hex="#0000FF", rgb=(0, 0, 255)),
    ]

    merged = merge_small_regions(labels, palette, min_area_px=100)
    assert merged[25, 50] == 0
    assert merged[75, 50] == 1


def test_chaikin_smooth_doubles_points():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    smooth = chaikin_smooth(square, iterations=2)
    assert len(smooth) == 16


def test_chaikin_smooth_preserves_centroid():
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
    smooth = chaikin_smooth(square, iterations=3)
    cx, cy = smooth.mean(axis=0)
    assert abs(cx - 50.0) < 5.0
    assert abs(cy - 50.0) < 5.0


def test_chaikin_smooth_short_contour_unchanged():
    triangle = np.array([[0, 0], [5, 0], [3, 3]], dtype=np.float64)
    smooth = chaikin_smooth(triangle, iterations=2)
    assert len(smooth) == 3


def test_adaptive_smooth_preserves_rectangle_corners():
    rect = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
    smooth = adaptive_smooth(rect, sigma=2.0, max_displacement=5.0)
    for corner in rect:
        matches = np.any(np.all(np.abs(smooth - corner) < 0.5, axis=1))
        assert matches, f"Corner {corner} not preserved"


def test_adaptive_smooth_limits_displacement():
    pts = np.array(
        [[0, 0], [50, 0], [100, 0], [100, 50], [100, 100],
         [50, 100], [0, 100], [0, 50]],
        dtype=np.float64,
    )
    smooth = adaptive_smooth(pts, sigma=5.0, max_displacement=1.0)
    dists = np.linalg.norm(smooth - pts, axis=1)
    assert dists.max() <= 1.0 + 1e-6


def test_adaptive_smooth_preserves_centroid():
    square = np.array(
        [[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64
    )
    smooth = adaptive_smooth(square, sigma=1.0)
    cx, cy = smooth.mean(axis=0)
    assert abs(cx - 50.0) < 2.0
    assert abs(cy - 50.0) < 2.0


def test_adaptive_smooth_short_contour_unchanged():
    triangle = np.array([[0, 0], [5, 0], [3, 3]], dtype=np.float64)
    smooth = adaptive_smooth(triangle)
    assert len(smooth) == 3
