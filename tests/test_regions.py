from __future__ import annotations

import numpy as np

from app.quantize import quantize
from app.regions import extract_regions
from .conftest import make_two_color_array


def test_two_color_yields_two_regions():
    img = make_two_color_array()
    palette, labels = quantize(img, palette_size=2)
    regions = extract_regions(labels, palette)
    assert len(regions) == 2
    assert {r.color_index for r in regions} == {1, 2}


def test_centroids_inside_image():
    img = make_two_color_array()
    palette, labels = quantize(img, palette_size=2)
    regions = extract_regions(labels, palette)
    for r in regions:
        cx, cy = r.centroid
        assert 0 <= cx <= 100
        assert 0 <= cy <= 100


def test_contour_is_polygon_array():
    img = make_two_color_array()
    palette, labels = quantize(img, palette_size=2)
    regions = extract_regions(labels, palette)
    for r in regions:
        assert r.contour.ndim == 2
        assert r.contour.shape[1] == 2
        assert r.contour.shape[0] >= 3  # a polygon
