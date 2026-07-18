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


class TestSamHqIntegration:
    def test_segment_with_stub(self, monkeypatch):
        """Verify segment_with_sam_hq returns correct shape when model is stubbed."""
        import numpy as np
        from app.models import PipelineParams

        pp = PipelineParams(
            max_working_side=1600,
            mean_shift_sp=5,
            mean_shift_sr=20,
            felzenszwalb_scale=30,
            palette_size=16,
            min_region_area_pct=0.5,
            morph_kernel=3,
            boundary_sigma=1.0,
        )

        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        class FakePredictor:
            model = None
            def set_image(self, img):
                pass

        def fake_load():
            import app.sam_segment as mod
            mod._MODEL = FakePredictor()
            mod._MASK_GENERATOR = FakeGenerator()

        class FakeGenerator:
            def generate(self, img):
                h, w = img.shape[:2]
                m1 = np.zeros((h, w), dtype=bool)
                m1[: h // 2, :] = True
                m2 = np.zeros((h, w), dtype=bool)
                m2[h // 2 :, :] = True
                return [
                    {"segmentation": m1, "area": m1.sum()},
                    {"segmentation": m2, "area": m2.sum()},
                ]

        import app.sam_segment as sam_mod
        monkeypatch.setattr(sam_mod, "_load_sam_hq_model", fake_load)
        monkeypatch.setattr(sam_mod, "is_sam_hq_available", lambda: True)

        result = sam_mod.segment_with_sam_hq(image, pp)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8
        assert not np.array_equal(result[:32, 0], result[32:, 0])
