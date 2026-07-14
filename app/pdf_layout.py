from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

# Python 3.8 compat: reportlab 4.x calls md5(usedforsecurity=False) which
# is only supported on Python 3.9+.  Strip the keyword so it works on 3.8.
import hashlib as _hashlib
import reportlab.pdfbase.pdfdoc as _pdfdoc
_orig = _pdfdoc.md5
_pdfdoc.md5 = lambda data=b'', **kw: _orig(data)

from reportlab.lib import colors
from reportlab.pdfgen import canvas

from . import config
from .models import ConvertParams, RenderData

_NUMBER_GRAY = colors.HexColor("#888888")


@dataclass
class LayoutGeometry:
    page_w: float
    page_h: float
    margin_pt: float
    image_x: float    # bottom-left x of rendered image
    image_y: float    # bottom-left y of rendered image
    image_w: float
    image_h: float
    scale: float
    legend_h: float


def resolve_orientation(params: ConvertParams, width: int, height: int) -> bool:
    """Return True for portrait. `auto` picks portrait when image is taller than wide."""
    if params.orientation == "portrait":
        return True
    if params.orientation == "landscape":
        return False
    return height >= width


def legend_rows(palette, available_width_pt: float, item_width_pt: float):
    per_row = max(1, int(available_width_pt // item_width_pt))
    return [palette[i:i + per_row] for i in range(0, len(palette), per_row)]


def compute_layout(data: RenderData, params: ConvertParams) -> LayoutGeometry:
    margin_pt = config.PAGE_MARGIN_MM * config.MM_TO_PT
    item_w_pt = config.LEGEND_ITEM_MM * config.MM_TO_PT
    row_h_pt = config.LEGEND_ROW_MM * config.MM_TO_PT
    top_gap_pt = config.LEGEND_TOP_GAP_MM * config.MM_TO_PT

    w_mm, h_mm = config.PAPER_SIZES_MM[params.paper_size]
    portrait = resolve_orientation(params, data.width, data.height)
    page_w = w_mm * config.MM_TO_PT
    page_h = h_mm * config.MM_TO_PT
    if not portrait:
        page_w, page_h = page_h, page_w

    legend_avail_w = page_w - 2 * margin_pt
    rows = legend_rows(data.palette, legend_avail_w, item_w_pt)
    legend_h = len(rows) * row_h_pt + top_gap_pt

    avail_w = page_w - 2 * margin_pt
    avail_h = page_h - 2 * margin_pt - legend_h
    scale = min(avail_w / data.width, avail_h / data.height)
    image_w = data.width * scale
    image_h = data.height * scale
    image_x = margin_pt + (avail_w - image_w) / 2
    image_y = margin_pt + legend_h + (avail_h - image_h) / 2

    return LayoutGeometry(
        page_w=page_w,
        page_h=page_h,
        margin_pt=margin_pt,
        image_x=image_x,
        image_y=image_y,
        image_w=image_w,
        image_h=image_h,
        scale=scale,
        legend_h=legend_h,
    )


def _to_pdf_pt(geo: LayoutGeometry, px: float, py: float):
    """Convert image pixel coords (origin top-left) to PDF points (origin bottom-left)."""
    return (geo.image_x + px * geo.scale, geo.image_y + geo.image_h - py * geo.scale)


def render_pdf(data: RenderData, params: ConvertParams) -> bytes:
    geo = compute_layout(data, params)
    number_color = colors.black if params.number_color == "black" else _NUMBER_GRAY

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(geo.page_w, geo.page_h))

    # Region outlines (white fill + black stroke so the sheet is paintable).
    for region in data.regions:
        pts = [_to_pdf_pt(geo, float(p[0]), float(p[1])) for p in region.contour]
        if len(pts) < 2:
            continue
        path = c.beginPath()
        path.moveTo(*pts[0])
        for x, y in pts[1:]:
            path.lineTo(x, y)
        path.close()
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.black)
        c.setLineWidth(params.line_thickness)
        c.drawPath(path, fill=1, stroke=1)

    # Region numbers at centroids.
    if params.show_numbers:
        c.setFillColor(number_color)
        for region in data.regions:
            cx, cy = _to_pdf_pt(geo, region.centroid[0], region.centroid[1])
            rendered_area_pt = region.area * geo.scale * geo.scale
            side = rendered_area_pt ** 0.5
            font_size = max(5.0, min(16.0, side * 0.4))
            c.setFont("Helvetica", font_size)
            c.drawCentredString(cx, cy - font_size / 2, str(region.color_index))

    _draw_legend(c, geo, data)
    c.showPage()
    c.save()
    return buf.getvalue()


def _draw_legend(c, geo: LayoutGeometry, data: RenderData) -> None:
    margin = geo.margin_pt
    item_w = config.LEGEND_ITEM_MM * config.MM_TO_PT
    square = config.LEGEND_SQUARE_MM * config.MM_TO_PT
    row_h = config.LEGEND_ROW_MM * config.MM_TO_PT
    top_gap = config.LEGEND_TOP_GAP_MM * config.MM_TO_PT
    avail_w = geo.page_w - 2 * margin

    rows = legend_rows(data.palette, avail_w, item_w)
    y_top = margin + geo.legend_h - top_gap

    for r_idx, row in enumerate(rows):
        y = y_top - r_idx * row_h - square
        x = margin
        for color in row:
            c.setFillColor(colors.HexColor(color.hex))
            c.setStrokeColor(colors.black)
            c.setLineWidth(0.5)
            c.rect(x, y, square, square, fill=1, stroke=1)
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 8)
            c.drawString(x + square + 2, y + 2, str(color.index))
            x += item_w
