from __future__ import annotations

import numpy as np
from io import BytesIO
from PIL import Image


def make_two_color_array() -> np.ndarray:
    """100x100 RGB image: red top half, blue bottom half."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :] = (220, 20, 20)
    img[50:, :] = (20, 20, 220)
    return img


def make_two_color_bytes() -> bytes:
    """PNG bytes of make_two_color_array."""
    buf = BytesIO()
    Image.fromarray(make_two_color_array()).save(buf, format="PNG")
    return buf.getvalue()
