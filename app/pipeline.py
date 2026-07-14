from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image
from skimage.segmentation import felzenszwalb

from . import config
from .models import ConvertParams, RenderData
from .quantize import quantize
from .regions import extract_regions
from .postprocess import merge_small_regions, smooth_label_boundaries


_KERNEL_CIRCLE_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _apply_global_morphology(labels: np.ndarray, palette) -> np.ndarray:
    """Per‑label MORPH_OPEN that writes back into a single label map.

    Narrow isthmuses are cut *before* region extraction so that contour
    finding sees cleanly separated colour islands.
    """
    clean = np.zeros_like(labels)
    for color_idx in range(len(palette)):
        mask = (labels == color_idx).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL_CIRCLE_5)
        clean[mask == 1] = color_idx
    return clean


def render_data_from_image(image_bytes: bytes, params: ConvertParams) -> RenderData:
    """Run the full conversion pipeline and return data ready for PDF rendering."""
    image = _load_and_normalize(image_bytes, params.smoothing)
    image = _segment_and_flatten(image, params.smoothing)
    palette, labels = quantize(image, params.palette_size)
    min_area_px = int(image.shape[0] * image.shape[1] * params.min_region_area / 100)
    labels = merge_small_regions(labels, palette, min_area_px)
    labels = smooth_label_boundaries(labels, sigma=1.0)
    labels = _apply_global_morphology(labels, palette)
    regions = extract_regions(labels, palette)
    return RenderData(
        width=image.shape[1],
        height=image.shape[0],
        palette=palette,
        regions=regions,
    )


def _load_and_normalize(image_bytes: bytes, smoothing: int) -> np.ndarray:
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(pil_image)

    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > config.MAX_WORKING_SIDE:
        scale = config.MAX_WORKING_SIDE / longest
        new_size = (int(w * scale), int(h * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    if smoothing > 0:
        # 1. Edge‑preserving smoothing (pyrMeanShiftFiltering is ~10× faster
        #    than bilateralFilter for the same quality on large images).
        sp = max(1, smoothing)
        sr = max(5, smoothing * 3)
        image = cv2.pyrMeanShiftFiltering(image, sp=sp, sr=sr)

        # 2. CLAHE on LAB L‑channel — adaptive local contrast.
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # 3. Unsharp masking — final edge sharpening.
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
        image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    # 2‑px black border so no real content touches the image edge.
    # This eliminates unclosed contours and boundary death rays.
    image = cv2.copyMakeBorder(image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return image


def _segment_and_flatten(image: np.ndarray, smoothing: int) -> np.ndarray:
    """Segment image with Felzenszwalb (edge-aware) and flatten each region to its mean color.

    Felzenszwalb follows real image boundaries instead of creating a regular
    grid like SLIC. Flattening ensures k-means produces clean, contiguous
    regions whose edges trace actual objects.
    """
    scale = 30 + smoothing * 5
    sigma = 0.5
    min_size = max(20, int(image.shape[0] * image.shape[1] * 0.0001))

    segments = felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)

    flat_seg = segments.ravel()
    flat_img = image.reshape(-1, 3).astype(np.float64)
    n = int(segments.max()) + 1
    sums = np.zeros((n, 3), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    np.add.at(sums, flat_seg, flat_img)
    np.add.at(counts, flat_seg, 1)
    means = sums / np.maximum(counts[:, np.newaxis], 1)
    return means[flat_seg].reshape(image.shape).astype(np.uint8)
