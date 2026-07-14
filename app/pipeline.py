from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from . import config
from .models import ConvertParams, RenderData
from .quantize import quantize
from .regions import extract_regions


def render_data_from_image(image_bytes: bytes, params: ConvertParams) -> RenderData:
    """Run the full conversion pipeline and return data ready for PDF rendering."""
    image = _load_and_normalize(image_bytes, params.smoothing)
    palette, labels = quantize(image, params.palette_size)
    regions = extract_regions(labels, palette, params.min_region_area)
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
        sigma = smoothing * 15
        image = cv2.bilateralFilter(image, d=9, sigmaColor=sigma, sigmaSpace=sigma)

    return image
