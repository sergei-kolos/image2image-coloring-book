from __future__ import annotations

from app.models import ConvertParams, RenderData
from app.pipeline import render_data_from_image
from .conftest import make_two_color_bytes


def test_pipeline_returns_render_data():
    png = make_two_color_bytes()
    params = ConvertParams(smoothing=0, color_merge_threshold=10.0)
    data = render_data_from_image(png, params)

    assert isinstance(data, RenderData)
    assert data.width > 0
    assert data.height > 0
    assert len(data.palette) >= 1
    assert len(data.regions) >= 1


def test_pipeline_downsamples_large_side():
    import numpy as np
    from io import BytesIO
    from PIL import Image

    big = np.zeros((3000, 4000, 3), dtype=np.uint8)
    big[:, :2000] = (200, 30, 30)
    big[:, 2000:] = (30, 30, 200)
    buf = BytesIO()
    Image.fromarray(big).save(buf, format="PNG")

    params = ConvertParams(palette_size=2, smoothing=0, detail_level=1)
    data = render_data_from_image(buf.getvalue(), params)
    pp = params.pipeline_params()
    assert max(data.width, data.height) <= pp.max_working_side
