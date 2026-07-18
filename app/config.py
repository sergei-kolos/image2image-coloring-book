from __future__ import annotations

MAX_PALETTE_SIZE = 64
DEFAULT_PALETTE_SIZE = 16

MAX_WORKING_SIDE = 1600  # px, longest side after normalization

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

PAGE_MARGIN_MM = 15
MM_TO_PT = 2.834645669
LEGEND_SQUARE_MM = 8
LEGEND_ITEM_MM = 20   # width per legend entry (square + number + gap)
LEGEND_ROW_MM = 12    # row height
LEGEND_TOP_GAP_MM = 5

PAPER_SIZES_MM = {
    "A3": (297, 420),
    "A4": (210, 297),
    "A5": (148, 210),
    "Letter": (215.9, 279.4),
    "Legal": (215.9, 355.6),
}

# ── AI Engine config ──────────────────────────────────────────────
import os as _os

SAM_HQ_MODEL_PATH = _os.environ.get(
    "SAM_HQ_MODEL_PATH", "models/sam_hq_vit_l.pth"
)
SAM_HQ_MODEL_TYPE = _os.environ.get("SAM_HQ_MODEL_TYPE", "vit_l")

SEMANTIC_MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"

# Cap GPU memory usage (fraction of total VRAM). Default: 50%.
GPU_MEMORY_FRACTION = float(_os.environ.get("GPU_MEMORY_FRACTION", "0.5"))


def _detect_ai_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


AI_DEVICE = _detect_ai_device()
