from __future__ import annotations

from app.models import ConvertParams, RenderData
from app.pipeline import render_data_from_image
from .conftest import make_two_color_bytes


def test_pipeline_returns_render_data():
    png = make_two_color_bytes()
    params = ConvertParams(palette_size=4, smoothing=0)
    data = render_data_from_image(png, params)

    assert isinstance(data, RenderData)
    assert data.width > 0
    assert data.height > 0
    assert len(data.palette) == 4
    assert len(data.regions) >= 1


def test_pipeline_downsamples_large_side():
    import numpy as np
    from io import BytesIO
    from PIL import Image
    from app import config

    big = np.zeros((3000, 4000, 3), dtype=np.uint8)
    big[:, :2000] = (200, 30, 30)
    big[:, 2000:] = (30, 30, 200)
    buf = BytesIO()
    Image.fromarray(big).save(buf, format="PNG")

    data = render_data_from_image(buf.getvalue(), ConvertParams(palette_size=2, smoothing=0))
    # 2‑px border on each side adds 4 to the working side.
    assert max(data.width, data.height) <= config.MAX_WORKING_SIDE + 4
