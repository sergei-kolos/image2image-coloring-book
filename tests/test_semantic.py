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


class TestImportanceMapIntegration:
    def test_compute_importance_with_stub(self, monkeypatch):
        """Verify compute_importance_map returns correct shape/range with stubbed model."""
        import numpy as np

        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        class FakeProcessor:
            @staticmethod
            def from_pretrained(name):
                return FakeProcessor()

            def __call__(self, images, return_tensors):
                import torch
                return {"pixel_values": torch.randn(1, 3, 256, 256)}

        class FakeModel:
            def to(self, device):
                return self
            def eval(self):
                return self
            def __call__(self, **kwargs):
                import torch
                logits = torch.zeros(1, 150, 16, 16)
                logits[:, 2, :, :] = 10.0
                logits[:, 12, 8:12, 8:12] = 20.0
                return type("FakeOutput", (), {"logits": logits, "loss": None})()

        def fake_load():
            import app.semantic as sem_mod
            sem_mod._PROCESSOR = FakeProcessor()
            sem_mod._MODEL = FakeModel()

        import app.semantic as sem_mod
        monkeypatch.setattr(sem_mod, "_load_semantic_model", fake_load)
        monkeypatch.setattr(sem_mod, "is_semantic_available", lambda: True)

        result = sem_mod.compute_importance_map(image)
        assert result.shape == (64, 64)
        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= result.max() <= 1.0
        assert result[32, 32] > result[0, 0]
