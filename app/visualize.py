from __future__ import annotations

import cv2
import numpy as np

from .models import ConvertParams, RenderData


def _draw_boundaries(vis: np.ndarray, regions, thickness: int, edge_color, edges=None):
    """Draw region boundaries as raster lines from reconstructed label map.

    Guarantees clean 1px-wide lines with no double strokes and no missing
    segments. The ``edges`` parameter is ignored — raster approach is always
    used for PNG output.
    """
    h, w = vis.shape[:2]
    # Reconstruct label map from region contours
    label_map = np.zeros((h, w), dtype=np.int32)
    for region in regions:
        cv2.fillPoly(label_map, [region.contour.astype(np.int32)], region.color_index)
    # Compute boundary mask (both sides of each label transition)
    boundary = np.zeros((h, w), dtype=bool)
    h_diff = label_map[:, :-1] != label_map[:, 1:]
    v_diff = label_map[:-1, :] != label_map[1:, :]
    boundary[:, :-1] |= h_diff
    boundary[:, 1:] |= h_diff
    boundary[:-1, :] |= v_diff
    boundary[1:, :] |= v_diff
    # Draw boundary pixels
    vis[boundary] = edge_color
    # Dilate for thicker lines
    if thickness > 1:
        kernel = np.ones((3, 3), np.uint8)
        boundary_u8 = boundary.astype(np.uint8) * 255
        for _ in range(thickness - 1):
            boundary_u8 = cv2.dilate(boundary_u8, kernel)
        vis[boundary_u8 > 0] = edge_color


def _draw_labels(vis: np.ndarray, regions, label_color):
    for region in regions:
        r = region.max_radius or (region.area ** 0.5) * 0.3
        if r < 3:
            continue
        cx, cy = int(round(region.centroid[0])), int(round(region.centroid[1]))
        font_scale = max(0.15, min(0.6, r * 0.035))
        cv2.putText(
            vis, region.label, (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, 1, cv2.LINE_AA,
        )


def render_visualization(data: RenderData, params: ConvertParams) -> bytes:
    """Render the paint‑by‑numbers result as a PNG byte string.

    Each region is filled with its palette colour; smoothed edges are
    drawn on top in black/gray; labels are placed at centroids.
    """
    h, w = data.height, data.width

    vis = np.ones((h, w, 3), dtype=np.uint8) * 255
    for region in data.regions:
        r, g, b = data.palette[region.color_index - 1].rgb
        cv2.fillPoly(vis, [region.contour.astype(np.int32)], (b, g, r))

    thickness = max(1, int(round(params.line_thickness * 1.5)))
    edge_color = (0, 0, 0) if params.number_color == "black" else (100, 100, 100)
    _draw_boundaries(vis, data.regions, thickness, edge_color, data.edges)
    if params.show_numbers:
        label_color = (0, 0, 0) if params.number_color == "black" else (100, 100, 100)
        _draw_labels(vis, data.regions, label_color)

    _, buf = cv2.imencode(".png", vis)
    return buf.tobytes()


def render_outline(data: RenderData, params: ConvertParams) -> bytes:
    """Render only the black outlines on white background (no colour fills)."""
    h, w = data.height, data.width

    vis = np.ones((h, w, 3), dtype=np.uint8) * 255
    thickness = max(1, int(round(params.line_thickness * 1.5)))
    edge_color = (0, 0, 0) if params.number_color == "black" else (100, 100, 100)
    _draw_boundaries(vis, data.regions, thickness, edge_color, data.edges)
    if params.show_numbers:
        label_color = (0, 0, 0) if params.number_color == "black" else (100, 100, 100)
        _draw_labels(vis, data.regions, label_color)

    _, buf = cv2.imencode(".png", vis)
    return buf.tobytes()
