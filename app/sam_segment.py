from __future__ import annotations

import numpy as np

from .models import PipelineParams

_MODEL = None
_MASK_GENERATOR = None


def is_sam_hq_available() -> bool:
    """True if torch + sam-hq + weights can be loaded."""
    try:
        import torch
        from .config import SAM_HQ_MODEL_PATH
        import os
        _ = torch.nn.Module
        return os.path.isfile(SAM_HQ_MODEL_PATH)
    except ImportError:
        return False


def _load_sam_hq_model():
    """Lazy-load SAM-HQ predictor and mask generator (cached globally)."""
    global _MODEL, _MASK_GENERATOR
    if _MODEL is not None:
        return

    import torch
    from .config import AI_DEVICE, SAM_HQ_MODEL_PATH
    from segment_anything import sam_model_registry, SamPredictor

    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=SAM_HQ_MODEL_PATH)
    sam.to(device=AI_DEVICE)
    _MODEL = SamPredictor(sam)


def segment_with_sam_hq(
    image: np.ndarray, pp: PipelineParams
) -> np.ndarray:
    """Run SAM-HQ automatic mask generation and return a flattened RGB image.

    The returned image has each region filled with its median colour, matching
    the output format of `_segment_and_flatten` so the rest of the pipeline
    (k-means, merge, regions) works unchanged.
    """
    from .config import AI_DEVICE

    h, w = image.shape[:2]

    _load_sam_hq_model()
    _MODEL.set_image(image)

    from segment_anything import SamAutomaticMaskGenerator
    global _MASK_GENERATOR
    if _MASK_GENERATOR is None:
        min_area_px = max(100, int(h * w * pp.min_region_area_pct / 100))
        _MASK_GENERATOR = SamAutomaticMaskGenerator(
            model=_MODEL.model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=min_area_px,
        )

    masks = _MASK_GENERATOR.generate(image)
    labels = _masks_to_labels(masks, h, w)

    flat_img = image.reshape(-1, 3).astype(np.float64)
    flat_lbl = labels.ravel()
    n = int(labels.max()) + 1
    medians = np.zeros((n, 3), dtype=np.float64)
    for sid in range(n):
        mask = flat_lbl == sid
        if mask.any():
            medians[sid] = np.median(flat_img[mask], axis=0)
    return medians[flat_lbl].reshape(image.shape).astype(np.uint8)


def _masks_to_labels(
    masks: list[dict], h: int, w: int
) -> np.ndarray:
    """Convert SAM's overlapping masks to a non-overlapping label map.

    Masks are sorted by area ascending (smallest painted last), so smaller
    masks overwrite larger ones — fine details win over background.
    """
    labels = np.zeros((h, w), dtype=np.int32)
    ordered = sorted(masks, key=lambda m: m["area"], reverse=True)
    for idx, m in enumerate(ordered, start=1):
        labels[m["segmentation"]] = idx
    return labels
