from __future__ import annotations

import cv2
import numpy as np

from .models import ConvertParams, RenderData


def render_visualization(data: RenderData, params: ConvertParams) -> bytes:
    """Render the paint‑by‑numbers result as a PNG byte string.

    Each region is filled with its palette colour; smoothed edges are
    drawn on top in black/gray; labels are placed at centroids.
    """
    h, w = data.height, data.width

    # Fill each region with its palette colour (OpenCV uses BGR).
    vis = np.ones((h, w, 3), dtype=np.uint8) * 255
    for region in data.regions:
        r, g, b = data.palette[region.color_index - 1].rgb
        cv2.fillPoly(vis, [region.contour.astype(np.int32)], (b, g, r))

    # Draw region boundaries (exterior + holes).
    thickness = max(1, int(round(params.line_thickness * 1.5)))
    edge_color = (0, 0, 0) if params.number_color == "black" else (100, 100, 100)
    for region in data.regions:
        ext = region.contour.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [ext], isClosed=True, color=edge_color, thickness=thickness)
        for hole in region.holes:
            h = hole.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [h], isClosed=True, color=edge_color, thickness=thickness)

    # Draw labels at region centroids.
    if params.show_numbers:
        label_color = (0, 0, 0) if params.number_color == "black" else (100, 100, 100)
        for region in data.regions:
            cx, cy = int(round(region.centroid[0])), int(round(region.centroid[1]))
            font_scale = max(0.25, min(0.6, (region.area ** 0.5) * 0.008))
            cv2.putText(
                vis, region.label, (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, 1, cv2.LINE_AA,
            )

    _, buf = cv2.imencode(".png", vis)  # already BGR
    return buf.tobytes()
