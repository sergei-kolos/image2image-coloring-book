from __future__ import annotations

import cv2
import numpy as np

_FACE_MESH = None

# MediaPipe Face Mesh landmark indices for key features
_NOSE = [1, 2, 3, 4, 5, 19, 49, 220]
_LEFT_EYE = [33, 133, 7, 163]
_RIGHT_EYE = [362, 263, 382, 381]
_MOUTH = [0, 17, 61, 291, 13, 14]
_LEFT_BROW = [46, 70, 105, 66]
_RIGHT_BROW = [276, 300, 335, 296]
_FACE_ALL = list(range(468))

_FEATURES = [
    (_NOSE, 0.50, 6),
    (_LEFT_EYE, 0.35, 5),
    (_RIGHT_EYE, 0.35, 5),
    (_MOUTH, 0.30, 7),
    (_LEFT_BROW, 0.20, 5),
    (_RIGHT_BROW, 0.20, 5),
    (_FACE_ALL, 0.15, 20),
]


def is_face_detection_available() -> bool:
    try:
        import mediapipe
        return True
    except ImportError:
        return False


def _load_face_mesh():
    global _FACE_MESH
    if _FACE_MESH is not None:
        return
    import mediapipe as mp
    _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=5,
        min_detection_confidence=0.5,
    )


def compute_face_importance(image: np.ndarray) -> np.ndarray:
    """Return per-pixel face detail importance map (H x W, float32, 0..1).

    Detects human faces via MediaPipe Face Mesh and boosts importance
    around nose, eyes, mouth, and eyebrows so these small features are
    preserved during region merging.

    Returns zero map if no faces detected.
    """
    h, w = image.shape[:2]
    importance = np.zeros((h, w), dtype=np.float32)

    _load_face_mesh()

    results = _FACE_MESH.process(image)

    if not results or not results.multi_face_landmarks:
        return importance

    y, x = np.ogrid[:h, :w]

    for face_landmarks in results.multi_face_landmarks:
        landmarks_px = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]

        for indices, boost, sigma in _FEATURES:
            cx = float(np.mean([landmarks_px[i][0] for i in indices]))
            cy = float(np.mean([landmarks_px[i][1] for i in indices]))
            dist2 = (x - cx) ** 2 + (y - cy) ** 2
            blob = boost * np.exp(-dist2 / (2.0 * sigma ** 2))
            np.maximum(importance, blob, out=importance)

    importance = cv2.GaussianBlur(importance, (5, 5), 1)
    return np.clip(importance, 0.0, 1.0)
