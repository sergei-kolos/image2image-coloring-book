from __future__ import annotations

from app.models import ConvertParams, PaletteColor, Region, RenderData
from app.pdf_layout import (
    LayoutGeometry,
    compute_layout,
    legend_rows,
    render_pdf,
    resolve_orientation,
)
import numpy as np


def _palette(n):
    return [PaletteColor(index=i + 1, hex="#FF0000", rgb=(255, 0, 0)) for i in range(n)]


def _data(width=200, height=100, palette_size=4):
    contour = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    regions = [Region(color_index=1, area=100, centroid=(5.0, 5.0), contour=contour)]
    return RenderData(width=width, height=height, palette=_palette(palette_size), regions=regions)


def test_resolve_orientation_auto():
    p = ConvertParams(orientation="auto")
    assert resolve_orientation(p, 100, 200) is True   # tall image -> portrait
    assert resolve_orientation(p, 200, 100) is False  # wide image -> landscape


def test_resolve_orientation_explicit():
    assert resolve_orientation(ConvertParams(orientation="portrait"), 200, 100) is True
    assert resolve_orientation(ConvertParams(orientation="landscape"), 100, 200) is False


def test_legend_rows_wraps():
    palette = _palette(10)
    rows = legend_rows(palette, available_width_pt=100, item_width_pt=30)
    # 100 // 30 == 3 per row -> 4 rows (3,3,3,1)
    assert len(rows) == 4
    assert len(rows[0]) == 3
    assert len(rows[-1]) == 1


def test_compute_layout_fits_within_page():
    data = _data(width=200, height=100, palette_size=4)
    geo = compute_layout(data, ConvertParams(paper_size="A4"))
    assert isinstance(geo, LayoutGeometry)
    assert geo.scale > 0
    assert geo.image_w <= geo.page_w
    assert geo.image_h <= geo.page_h
    assert geo.image_x >= 0
    assert geo.image_y >= 0


def test_render_pdf_returns_valid_pdf_bytes():
    data = _data(palette_size=8)
    pdf = render_pdf(data, ConvertParams())
    assert isinstance(pdf, bytes)
    assert pdf[:4] == b"%PDF"


def test_render_pdf_without_numbers():
    data = _data(palette_size=8)
    pdf = render_pdf(data, ConvertParams(show_numbers=False))
    assert pdf[:4] == b"%PDF"
