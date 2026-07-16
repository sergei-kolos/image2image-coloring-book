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
