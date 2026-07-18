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
    if data.width == 0 or data.height == 0:
        raise ValueError(f"RenderData has zero dimension: width={data.width}, height={data.height}")

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

    # colour index → hex lookup
    color_map = {pc.index: pc.hex for pc in data.palette}

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(geo.page_w, geo.page_h))

    # ── Pass 1: outlines only (no colour fill — this is a coloring book) ──
    outline_color = colors.black if params.number_color == "black" else _NUMBER_GRAY
    c.setStrokeColor(outline_color)
    outline_width = max(0.5, params.line_thickness * 1.0)
    c.setLineWidth(outline_width)

    outlines = c.beginPath()
    if data.edges:
        # Outer border (image rectangle)
        x0, y0 = _to_pdf_pt(geo, 0, 0)
        x1, y1 = _to_pdf_pt(geo, data.width, data.height)
        c.rect(x0, y0, x1 - x0, y1 - y0, stroke=1, fill=0)
        # Internal shared edges
        for edge in data.edges:
            pts = [_to_pdf_pt(geo, float(p[0]), float(p[1])) for p in edge.polyline]
            if len(pts) >= 2:
                outlines.moveTo(*pts[0])
                for x, y in pts[1:]:
                    outlines.lineTo(x, y)
    else:
        # Fallback: per-region contours
        for region in data.regions:
            pts = [_to_pdf_pt(geo, float(p[0]), float(p[1])) for p in region.contour]
            if len(pts) >= 2:
                outlines.moveTo(*pts[0])
                for x, y in pts[1:]:
                    outlines.lineTo(x, y)
                outlines.close()
            for hole in region.holes:
                pts = [_to_pdf_pt(geo, float(p[0]), float(p[1])) for p in hole]
                if len(pts) >= 2:
                    outlines.moveTo(*pts[0])
                    for x, y in pts[1:]:
                        outlines.lineTo(x, y)
                    outlines.close()
    c.drawPath(outlines, fill=0, stroke=1)

    # ── Region labels at centroids ──
    if params.show_numbers:
        c.setFillColor(number_color)
        for region in data.regions:
            cx, cy = _to_pdf_pt(geo, region.centroid[0], region.centroid[1])
            rendered_area_pt = region.area * geo.scale * geo.scale
            side = rendered_area_pt ** 0.5
            # Cap font size by both region side and label width
            label_w_factor = max(1.0, len(region.label) * 0.55)
            font_size = max(3.0, min(16.0, side * 0.7 / label_w_factor))
            c.setFont("Helvetica", font_size)
            c.drawCentredString(cx, cy - font_size / 2, region.label)

    _draw_legend(c, geo, data, params.number_color)
    c.showPage()
    c.save()
    return buf.getvalue()


def _draw_legend(c, geo: LayoutGeometry, data: RenderData, number_color: str) -> None:
    margin = geo.margin_pt
    item_w = config.LEGEND_ITEM_MM * config.MM_TO_PT
    square = config.LEGEND_SQUARE_MM * config.MM_TO_PT
    row_h = config.LEGEND_ROW_MM * config.MM_TO_PT
    top_gap = config.LEGEND_TOP_GAP_MM * config.MM_TO_PT
    avail_w = geo.page_w - 2 * margin

    rows = legend_rows(data.palette, avail_w, item_w)
    y_top = margin + geo.legend_h - top_gap

    legend_label_color = colors.black if number_color == "black" else _NUMBER_GRAY

    for r_idx, row in enumerate(rows):
        y = y_top - r_idx * row_h - square
        x = margin
        for color in row:
            c.setFillColor(colors.HexColor(color.hex))
            c.setStrokeColor(colors.black)
            c.setLineWidth(0.5)
            c.rect(x, y, square, square, fill=1, stroke=1)
            c.setFillColor(legend_label_color)
            c.setFont("Helvetica", 8)
            c.drawString(x + square + 2, y + 2, str(color.index))
            x += item_w
