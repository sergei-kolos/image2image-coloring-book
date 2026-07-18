from __future__ import annotations

import numpy as np
import pytest


class TestMasksToLabels:
    def test_no_overlap(self):
        from app.sam_segment import _masks_to_labels
        h, w = 10, 10
        m1 = np.zeros((h, w), dtype=bool)
        m1[:5, :] = True
        m2 = np.zeros((h, w), dtype=bool)
        m2[5:, :] = True
        masks = [
            {"segmentation": m1, "area": 50},
            {"segmentation": m2, "area": 50},
        ]
        result = _masks_to_labels(masks, h, w)
        assert result[0, 0] != 0
        assert result[9, 0] != 0
        assert result[0, 0] != result[9, 0]

    def test_overlap_small_wins(self):
        from app.sam_segment import _masks_to_labels
        h, w = 10, 10
        m1 = np.ones((h, w), dtype=bool)
        m2 = np.zeros((h, w), dtype=bool)
        m2[3:7, 3:7] = True
        masks = [
            {"segmentation": m1, "area": 100},
            {"segmentation": m2, "area": 16},
        ]
        result = _masks_to_labels(masks, h, w)
        small_label = result[5, 5]
        assert small_label != result[0, 0]

    def test_background_pixels(self):
        from app.sam_segment import _masks_to_labels
        h, w = 10, 10
        m1 = np.zeros((h, w), dtype=bool)
        m1[:5, :5] = True
        masks = [{"segmentation": m1, "area": 25}]
        result = _masks_to_labels(masks, h, w)
        assert result[9, 9] == 0
        assert result[0, 0] == 1


class TestIsSamAvailable:
    def test_returns_bool(self):
        from app.sam_segment import is_sam_hq_available
        result = is_sam_hq_available()
        assert isinstance(result, bool)
