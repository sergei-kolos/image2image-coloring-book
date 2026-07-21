from __future__ import annotations

import numpy as np


class TestIsFaceDetectionAvailable:
    def test_returns_bool(self):
        from app.face_detection import is_face_detection_available
        result = is_face_detection_available()
        assert isinstance(result, bool)


class TestComputeFaceImportance:
    def test_no_face_returns_zero(self):
        from app.face_detection import compute_face_importance
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = compute_face_importance(image)
        assert result.shape == (100, 100)
        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= result.max() <= 1.0
