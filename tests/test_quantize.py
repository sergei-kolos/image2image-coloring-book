from __future__ import annotations

import numpy as np

from app.quantize import quantize, merge_similar_colors
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
    top = labels[:50, :]
    bottom = labels[50:, :]
    assert len(np.unique(top)) == 1
    assert len(np.unique(bottom)) == 1
    assert top[0, 0] != bottom[0, 0]


def test_merge_similar_colors_removes_duplicates():
    from app.models import PaletteColor
    palette = [
        PaletteColor(index=1, hex="#FF0000", rgb=(255, 0, 0)),
        PaletteColor(index=2, hex="#FF0000", rgb=(255, 0, 0)),
        PaletteColor(index=3, hex="#00FF00", rgb=(0, 255, 0)),
        PaletteColor(index=4, hex="#00FF00", rgb=(0, 255, 0)),
    ]
    labels = np.array([[0, 1], [2, 3]], dtype=np.int32)

    new_palette, new_labels = merge_similar_colors(palette, labels, threshold=0)

    assert len(new_palette) == 2
    assert new_labels.shape == labels.shape
    assert set(np.unique(new_labels)) == {0, 1}


def test_merge_similar_colors_merges_close_colors():
    from app.models import PaletteColor
    palette = [
        PaletteColor(index=1, hex="#FF0000", rgb=(255, 0, 0)),
        PaletteColor(index=2, hex="#FE0101", rgb=(254, 1, 1)),
        PaletteColor(index=3, hex="#0000FF", rgb=(0, 0, 255)),
    ]
    labels = np.array([[0, 1], [2, 2]], dtype=np.int32)

    new_palette, new_labels = merge_similar_colors(palette, labels, threshold=2.0)

    assert len(new_palette) == 2
    assert new_labels[0, 0] == new_labels[0, 1]


def test_merge_similar_colors_reindexes_labels():
    from app.models import PaletteColor
    palette = [
        PaletteColor(index=1, hex="#FF0000", rgb=(255, 0, 0)),
        PaletteColor(index=2, hex="#FF0000", rgb=(255, 0, 0)),
        PaletteColor(index=3, hex="#00FF00", rgb=(0, 255, 0)),
        PaletteColor(index=4, hex="#0000FF", rgb=(0, 0, 255)),
    ]
    labels = np.array([[0, 1], [2, 3]], dtype=np.int32)

    new_palette, new_labels = merge_similar_colors(palette, labels, threshold=0)

    assert len(new_palette) == 3
    assert new_labels.min() == 0
    assert new_labels.max() == 2


def test_merge_similar_colors_preserves_palette_indices():
    from app.models import PaletteColor
    palette = [
        PaletteColor(index=1, hex="#FF0000", rgb=(255, 0, 0)),
        PaletteColor(index=2, hex="#FF0000", rgb=(255, 0, 0)),
        PaletteColor(index=3, hex="#00FF00", rgb=(0, 255, 0)),
    ]
    labels = np.array([[0, 1], [2, 2]], dtype=np.int32)

    new_palette, _ = merge_similar_colors(palette, labels, threshold=0)

    assert len(new_palette) == 2
    assert new_palette[0].index == 1
    assert new_palette[1].index == 2
