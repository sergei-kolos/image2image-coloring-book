from __future__ import annotations

import numpy as np
import pytest


class TestClassImportance:
    def test_array_length(self):
        from app.semantic import _CLASS_IMPORTANCE
        assert len(_CLASS_IMPORTANCE) == 150

    def test_values_in_range(self):
        from app.semantic import _CLASS_IMPORTANCE
        assert _CLASS_IMPORTANCE.min() >= 0.0
        assert _CLASS_IMPORTANCE.max() <= 1.0

    def test_person_has_high_importance(self):
        from app.semantic import _CLASS_IMPORTANCE
        assert _CLASS_IMPORTANCE[12] > 0.8

    def test_sky_has_low_importance(self):
        from app.semantic import _CLASS_IMPORTANCE
        assert _CLASS_IMPORTANCE[2] < 0.2


class TestIsSemanticAvailable:
    def test_returns_bool(self):
        from app.semantic import is_semantic_available
        result = is_semantic_available()
        assert isinstance(result, bool)
