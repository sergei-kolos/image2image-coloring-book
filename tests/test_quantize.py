from __future__ import annotations

import numpy as np

from app.quantize import quantize
from .conftest import make_two_color_array


def test_palette_has_requested_size():
    img = make_two_color_array()
    palette, labels = quantize(img, palette_size=8)
    assert len(palette) == 8
    assert [c.index for c in palette] == [1, 2, 3, 4, 5, 6, 7, 8]


def test_palette_hex_format():
    img = make_two_color_array()
    palette, _ = quantize(img, palette_size=4)
    for c in palette:
        assert c.hex.startswith("#")
        assert len(c.hex) == 7


def test_labels_shape_and_range():
    img = make_two_color_array()
    palette, labels = quantize(img, palette_size=4)
    assert labels.shape == (100, 100)
    assert labels.min() >= 0
    assert labels.max() <= 3


def test_two_color_image_splits_into_two_dominant_regions():
    img = make_two_color_array()
    _, labels = quantize(img, palette_size=2)
    # Top half one cluster, bottom half another (cluster id order is arbitrary)
    top = labels[:50, :]
    bottom = labels[50:, :]
    assert len(np.unique(top)) == 1
    assert len(np.unique(bottom)) == 1
    assert top[0, 0] != bottom[0, 0]
