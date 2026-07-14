from __future__ import annotations

import numpy as np

from app.quantize import quantize
from app.regions import extract_regions
from .conftest import make_two_color_array


def test_two_color_yields_two_regions():
    img = make_two_color_array()
    palette, labels = quantize(img, palette_size=2)
    regions = extract_regions(labels, palette, min_region_area=0.1)
    assert len(regions) == 2
    assert {r.color_index for r in regions} == {1, 2}


def test_centroids_inside_image():
    img = make_two_color_array()
    palette, labels = quantize(img, palette_size=2)
    regions = extract_regions(labels, palette, min_region_area=0.1)
    for r in regions:
        cx, cy = r.centroid
        assert 0 <= cx <= 100
        assert 0 <= cy <= 100


def test_contour_is_polygon_array():
    img = make_two_color_array()
    palette, labels = quantize(img, palette_size=2)
    regions = extract_regions(labels, palette, min_region_area=0.1)
    for r in regions:
        assert r.contour.ndim == 2
        assert r.contour.shape[1] == 2
        assert r.contour.shape[0] >= 3  # a polygon


def test_min_region_area_filters_small_regions():
    # 4-quadrant image: each quadrant 50x50 = 2500px (25% of 10000)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50] = (255, 0, 0)
    img[:50, 50:] = (0, 255, 0)
    img[50:, :50] = (0, 0, 255)
    img[50:, 50:] = (255, 255, 0)
    palette, labels = quantize(img, palette_size=4)
    # 30% threshold drops everything (each quadrant is 25%)
    regions = extract_regions(labels, palette, min_region_area=30.0)
    assert regions == []
    # 10% threshold keeps all four
    regions = extract_regions(labels, palette, min_region_area=10.0)
    assert len(regions) == 4
