from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image
from skimage.segmentation import felzenszwalb

from .models import ConvertParams, PipelineParams, RenderData
from .quantize import quantize, merge_similar_colors
from .regions import extract_regions, extract_shared_edges
from .postprocess import merge_small_regions, smooth_label_boundaries


def render_data_from_image(image_bytes: bytes, params: ConvertParams) -> RenderData:
    """Run the full conversion pipeline and return data ready for PDF rendering."""
    pp = params.pipeline_params()
    image = _load_and_normalize(image_bytes, pp)

    # ── Segmentation engine branch ──
    if params.engine == "sam_hq":
        from .sam_segment import segment_with_sam_hq, is_sam_hq_available
        if not is_sam_hq_available():
            raise RuntimeError(
                "SAM-HQ engine requested but not available. "
                "Install torch + sam-hq + download weights."
            )
        image = segment_with_sam_hq(image, pp)
    else:
        image = _segment_and_flatten(image, pp)

    palette, labels = quantize(image, pp.palette_size)
    # AI engine produces more sub-regions — boost merge threshold for cleaner palette
    merge_thresh = params.color_merge_threshold
    if params.engine == "sam_hq":
        merge_thresh = max(merge_thresh, 15.0)
    palette, labels = merge_similar_colors(palette, labels, threshold=merge_thresh)
    min_area_px = int(image.shape[0] * image.shape[1] * pp.min_region_area_pct / 100)
    edge_density = _compute_edge_density(image)

    # ── Semantic importance (AI engine only; optional even then) ──
    importance_map = None
    if params.engine == "sam_hq":
        from .semantic import compute_importance_map, is_semantic_available
        if is_semantic_available():
            importance_map = compute_importance_map(image)

    labels = merge_small_regions(labels, palette, min_area_px, edge_density, importance_map)
    if pp.boundary_sigma >= 0.5:
        labels = smooth_label_boundaries(labels, sigma=pp.boundary_sigma)
    regions = extract_regions(labels, palette, morph_kernel=pp.morph_kernel)
    edges = extract_shared_edges(labels)
    # Strip the 2px border added in _load_and_normalize (always added)
    regions = _strip_border_from_regions(regions, border=2)
    edges = _strip_border_from_edges(edges, border=2)
    final_w = image.shape[1] - 4
    final_h = image.shape[0] - 4
    edges = _clip_edges_to_bounds(edges, final_w, final_h)
    return RenderData(
        width=final_w,
        height=final_h,
        palette=palette,
        regions=regions,
        edges=edges,
    )


def _load_and_normalize(image_bytes: bytes, pp: PipelineParams) -> np.ndarray:
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(pil_image)

    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > pp.max_working_side:
        scale = pp.max_working_side / longest
        new_size = (int(w * scale), int(h * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    if pp.mean_shift_sp > 1:
        d = max(5, min(9, pp.mean_shift_sp))
        sigma_color = float(pp.mean_shift_sr)
        sigma_space = float(pp.mean_shift_sp * 5)
        image = cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    image = cv2.copyMakeBorder(image, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    return image


def _segment_and_flatten(image: np.ndarray, pp: PipelineParams) -> np.ndarray:
    sigma = 0.5
    h, w = image.shape[:2]
    area = h * w

    # Enhance local contrast before segmentation — helps Felzenszwalb
    # detect boundaries in low-contrast zones (shadows, skin tones).
    clahe_clip = max(0.5, min(1.5, 1.5 - (pp.felzenszwalb_scale - 10) * 0.02))
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Use smaller min_size to preserve fine details, then let merge_small_regions
    # clean up noise in uniform zones (it already does color-similarity merging).
    min_size = max(5, int(area * 0.00002))

    segments = felzenszwalb(
        image, scale=pp.felzenszwalb_scale, sigma=sigma, min_size=min_size
    )

    flat_seg = segments.ravel()
    flat_img = image.reshape(-1, 3).astype(np.float64)
    n = int(segments.max()) + 1
    medians = np.zeros((n, 3), dtype=np.float64)
    for sid in range(n):
        mask = flat_seg == sid
        if mask.any():
            medians[sid] = np.median(flat_img[mask], axis=0)
    return medians[flat_seg].reshape(image.shape).astype(np.uint8)


def _compute_edge_density(image: np.ndarray) -> np.ndarray:
    """Return per-pixel edge density map (0..1) for edge-aware merging."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150).astype(np.float32)
    return cv2.boxFilter(edges / 255.0, -1, (15, 15), normalize=True)


def _strip_border_from_regions(regions, border: int):
    """Shift all region contours inward by *border* pixels."""
    for region in regions:
        region.contour = region.contour - border
        region.centroid = (region.centroid[0] - border, region.centroid[1] - border)
        region.holes = [h - border for h in region.holes]
    return regions


def _strip_border_from_edges(edges, border: int):
    """Shift all shared edge polylines inward by *border* pixels."""
    for edge in edges:
        edge.polyline = edge.polyline - border
    return edges


def _clip_edges_to_bounds(edges, width: int, height: int):
    """Clip shared edge polylines to [0, width-1] x [0, height-1]."""
    for edge in edges:
        edge.polyline[:, 0] = np.clip(edge.polyline[:, 0], 0, width - 1)
        edge.polyline[:, 1] = np.clip(edge.polyline[:, 1], 0, height - 1)
    return edges


def _apply_global_morphology(
    labels: np.ndarray, palette, kernel_size: int
) -> np.ndarray:
    clean = np.zeros_like(labels)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    for color_idx in range(len(palette)):
        mask = (labels == color_idx).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        clean[mask == 1] = color_idx
    return clean
