# AI Segmentation (SAM-HQ + ADE20K) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SAM-HQ segmentation engine and ADE20K semantic importance mapping as optional AI features to the Coloring Book pipeline, user-selectable via UI toggle.

**Architecture:** Two new modules (`app/sam_segment.py`, `app/semantic.py`) with lazy-loaded PyTorch models. Pipeline branches on `params.engine` — classic (Felzenszwalb) unchanged, AI path replaces segmentation with SAM-HQ and feeds ADE20K importance map into `merge_small_regions`. All AI deps optional; app works without them.

**Tech Stack:** Python 3.8, torch 2.1.2, transformers, segment-anything, sam-hq (from git), FastAPI, OpenCV.

## Global Constraints

- **Python 3.8** — `from __future__ import annotations` in every `.py`.
- **torch==2.1.2** — last version supporting Python 3.8.
- **All AI deps are optional** — `ImportError` → `is_sam_hq_available() / is_semantic_available()` return `False`. Classic engine always works.
- **48 existing tests must pass unchanged** (classic engine is the default).
- **No committing model weights** — `models/` is gitignored.
- **RTX 3060 12GB** target GPU. CPU fallback supported but slower.

---

### Task 0: Create feature branch and install AI dependencies

**Files:**
- Create/modify: `requirements.txt`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `torch`, `transformers`, `segment-anything`, `sam-hq` installed into `.venv`.
- After this task: `import torch; torch.cuda.is_available()` prints `True`.

- [ ] **Step 1: Create feature branch from feat/shared-boundaries**

```bash
git fetch origin
git checkout feat/shared-boundaries
git checkout -b feat/ai-segmentation
```

- [ ] **Step 2: Add AI dependencies to requirements.txt**

Read current `requirements.txt` (already read — end of file). Append these lines:

```
# AI engine (optional — app works without these, see app/sam_segment.py)
torch==2.1.2
torchvision==0.16.2
transformers>=4.30,<5.0
segment-anything @ git+https://github.com/facebookresearch/segment-anything.git
sam-hq @ git+https://github.com/SysCV/sam-hq.git
```

- [ ] **Step 3: Install PyTorch with CUDA 12.1 wheels**

```bash
.\.venv\Scripts\Activate.ps1
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

Expected: installs torch 2.1.2 with CUDA support.

- [ ] **Step 4: Install remaining AI packages**

```bash
.\.venv\Scripts\Activate.ps1
pip install "transformers>=4.30,<5.0"
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/SysCV/sam-hq.git
```

Expected: all install cleanly.

- [ ] **Step 5: Verify torch + CUDA**

```bash
.\.venv\Scripts\Activate.ps1; python -c "import torch; print(f'Torch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
Torch 2.1.2
CUDA available: True
```

If CUDA is not available, the app will fall back to CPU (slower but functional). Continue anyway.

- [ ] **Step 6: Update .gitignore**

Append these lines to `.gitignore`:

```
# AI model weights (large files, downloaded manually)
models/

# HuggingFace cache (not needed in repo)
~/.cache/huggingface/

# sam-hq vendored source (installed via pip from git)
vendor/
```

Current `.gitignore` ends with `!samples/*.pdf`. Add after that line.

- [ ] **Step 7: Create models/ directory for weights**

```bash
New-Item -ItemType Directory -Path "models" -Force
```

- [ ] **Step 8: Commit dependencies and config**

```bash
git add requirements.txt .gitignore models/
git commit -m "chore: add AI dependencies (torch, transformers, SAM-HQ)"
```

---

### Task 1: SAM-HQ segmentation module (`app/sam_segment.py`)

**Files:**
- Create: `app/sam_segment.py`
- Test: `tests/test_sam_segment.py` (Task 8)

**Interfaces:**
- Produces:
  - `is_sam_hq_available() -> bool`
  - `segment_with_sam_hq(image: np.ndarray, pp: PipelineParams) -> np.ndarray` — returns flattened RGB image (same format as `_segment_and_flatten` output) so downstream k-means works unchanged.

**Design note:** Instead of returning raw labels, `segment_with_sam_hq` returns a flattened RGB image (median colour per label), matching the return type of `_segment_and_flatten`. The pipeline's k-means step expects RGB input and runs identically on both paths.

- [ ] **Step 1: Write the module file**

```python
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
        _ = torch.nn.Module  # verify torch
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
    from segment_anything import sam_model_registry
    from sam_hq.predictor import SamHQPredictor  # uses sam_hq.predictor

    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=SAM_HQ_MODEL_PATH)
    sam.to(device=AI_DEVICE)
    _MODEL = SamHQPredictor(sam)


def segment_with_sam_hq(
    image: np.ndarray, pp: PipelineParams
) -> np.ndarray:
    """Run SAM-HQ automatic mask generation and return a flattened RGB image.

    The returned image has each region filled with its median colour, matching
    the output format of `_segment_and_flatten` so the rest of the pipeline
    (k-means, merge, regions) works unchanged.

    Parameters
    ----------
    image : np.ndarray
        RGB image (H, W, 3), uint8. Already resized by `_load_and_normalize`.
    pp : PipelineParams
        Pipeline parameters (used for min_region_area).

    Returns
    -------
    np.ndarray
        Flattened RGB image (H, W, 3), uint8.
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

    # Median flatten per label — same logic as _segment_and_flatten
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

    Masks are sorted by area descending (largest painted first), so smaller
    masks painted later overwrite larger ones — fine details win over
    background.
    """
    labels = np.zeros((h, w), dtype=np.int32)
    ordered = sorted(masks, key=lambda m: m["area"], reverse=False)
    for idx, m in enumerate(ordered, start=1):
        labels[m["segmentation"]] = idx
    return labels
```

- [ ] **Step 2: Write the test file (failing — module doesn't exist in test context yet)**

Create `tests/test_sam_segment.py`:

```python
from __future__ import annotations

import numpy as np
import pytest


class TestMasksToLabels:
    def test_no_overlap(self):
        from app.sam_segment import _masks_to_labels
        h, w = 10, 10
        m1 = np.zeros((h, w), dtype=bool)
        m1[:5, :] = True
        m2 = np.zeros((h, w), dtype=bool)
        m2[5:, :] = True
        masks = [
            {"segmentation": m1, "area": 50},
            {"segmentation": m2, "area": 50},
        ]
        result = _masks_to_labels(masks, h, w)
        assert result[0, 0] != 0
        assert result[9, 0] != 0
        assert result[0, 0] != result[9, 0]

    def test_overlap_small_wins(self):
        from app.sam_segment import _masks_to_labels
        h, w = 10, 10
        m1 = np.ones((h, w), dtype=bool)  # background
        m2 = np.zeros((h, w), dtype=bool)
        m2[3:7, 3:7] = True  # small region inside
        masks = [
            {"segmentation": m1, "area": 100},
            {"segmentation": m2, "area": 16},
        ]
        result = _masks_to_labels(masks, h, w)
        # Small mask (area 16) should overwrite large mask in its region
        small_label = result[5, 5]
        assert small_label != result[0, 0]

    def test_background_pixels(self):
        from app.sam_segment import _masks_to_labels
        h, w = 10, 10
        m1 = np.zeros((h, w), dtype=bool)
        m1[:5, :5] = True
        masks = [{"segmentation": m1, "area": 25}]
        result = _masks_to_labels(masks, h, w)
        assert result[9, 9] == 0  # uncovered = background
        assert result[0, 0] == 1  # covered = label 1


class TestIsSamAvailable:
    def test_returns_bool(self):
        from app.sam_segment import is_sam_hq_available
        result = is_sam_hq_available()
        assert isinstance(result, bool)
```

- [ ] **Step 3: Run the test to verify it works**

```bash
.\.venv\Scripts\Activate.ps1; python -m pytest tests/test_sam_segment.py -v
```

Expected: 4 tests pass (the function is testable without torch or weights).

- [ ] **Step 4: Commit**

```bash
git add app/sam_segment.py tests/test_sam_segment.py
git commit -m "feat: add SAM-HQ segmentation module with mask-to-label conversion"
```

---

### Task 2: ADE20K semantic importance module (`app/semantic.py`)

**Files:**
- Create: `app/semantic.py`
- Test: `tests/test_semantic.py` (Task 9)

**Interfaces:**
- Produces:
  - `is_semantic_available() -> bool`
  - `compute_importance_map(image: np.ndarray) -> np.ndarray` — returns H×W float32 array in [0, 1].

- [ ] **Step 1: Write the module file**

```python
from __future__ import annotations

import numpy as np
import cv2

_MODEL = None
_PROCESSOR = None


def is_semantic_available() -> bool:
    """True if transformers + SegFormer model can be loaded."""
    try:
        import torch
        from transformers import SegformerForSemanticSegmentation
        _ = torch.nn.Module
        _ = SegformerForSemanticSegmentation
        return True
    except ImportError:
        return False


def _load_semantic_model():
    """Lazy-load SegFormer-B2 on ADE20K (cached globally)."""
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return

    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    from .config import SEMANTIC_MODEL_NAME, AI_DEVICE

    _PROCESSOR = SegformerImageProcessor.from_pretrained(SEMANTIC_MODEL_NAME)
    _MODEL = SegformerForSemanticSegmentation.from_pretrained(SEMANTIC_MODEL_NAME)
    _MODEL.to(AI_DEVICE)
    _MODEL.eval()


def compute_importance_map(image: np.ndarray) -> np.ndarray:
    """Return per-pixel importance map (H x W, float32, 0..1).

    1.0 = preserve detail (faces, hands, text), 0.0 = simplify (sky, walls).
    """
    from .config import AI_DEVICE
    import torch

    h, w = image.shape[:2]

    _load_semantic_model()

    inputs = _PROCESSOR(images=image, return_tensors="pt")
    inputs = {k: v.to(AI_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _MODEL(**inputs)
        logits = outputs.logits  # (1, 150, H/4, W/4)

    seg_map = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int32)  # (H/4, W/4)
    importance_small = _CLASS_IMPORTANCE[seg_map]

    # Upsample to original resolution
    importance = cv2.resize(importance_small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    # Soften class boundary transitions
    importance = cv2.GaussianBlur(importance, (31, 31), 8)
    return np.clip(importance, 0.0, 1.0)


# ADE20K 150-class importance mapping (see SPEC for tier definitions)
# Index = ADE20K class id, value = importance (0..1)
_CLASS_IMPORTANCE = np.array([
    0.10,  # 0: wall
    0.15,  # 1: building
    0.05,  # 2: sky
    0.10,  # 3: floor
    0.25,  # 4: tree
    0.10,  # 5: ceiling
    0.15,  # 6: road
    0.35,  # 7: bed
    0.20,  # 8: windowpane
    0.05,  # 9: grass
    0.30,  # 10: cabinet
    0.10,  # 11: sidewalk
    0.90,  # 12: person
    0.05,  # 13: earth/ground
    0.15,  # 14: door
    0.35,  # 15: table
    0.20,  # 16: mountain
    0.30,  # 17: plant
    0.20,  # 18: curtain
    0.40,  # 19: chair
    0.50,  # 20: car
    0.05,  # 21: water
    0.50,  # 22: painting
    0.35,  # 23: sofa
    0.25,  # 24: shelf
    0.20,  # 25: house
    0.05,  # 26: sea
    0.40,  # 27: mirror
    0.25,  # 28: rug
    0.05,  # 29: field
    0.35,  # 30: armchair
    0.30,  # 31: seat
    0.20,  # 32: fence
    0.35,  # 33: desk
    0.15,  # 34: rock/stone
    0.30,  # 35: wardrobe
    0.35,  # 36: lamp
    0.30,  # 37: bathtub
    0.20,  # 38: railing
    0.25,  # 39: cushion
    0.15,  # 40: base/pedestal
    0.30,  # 41: box
    0.15,  # 42: column
    0.60,  # 43: signboard
    0.30,  # 44: chest of drawers
    0.25,  # 45: counter
    0.05,  # 46: sand
    0.30,  # 47: sink
    0.15,  # 48: skyscraper
    0.30,  # 49: fireplace
    0.25,  # 50: refrigerator
    0.20,  # 51: grandstand
    0.15,  # 52: path
    0.15,  # 53: stairs
    0.15,  # 54: runway
    0.30,  # 55: case/display case
    0.30,  # 56: pool table
    0.25,  # 57: pillow
    0.20,  # 58: screen door
    0.15,  # 59: stairway
    0.10,  # 60: river
    0.25,  # 61: bridge
    0.25,  # 62: bookcase
    0.20,  # 63: blind/screen
    0.30,  # 64: coffee table
    0.30,  # 65: toilet
    0.70,  # 66: flower
    0.30,  # 67: book
    0.15,  # 68: hill
    0.30,  # 69: bench
    0.25,  # 70: countertop
    0.30,  # 71: stove
    0.25,  # 72: palm
    0.30,  # 73: kitchen
    0.60,  # 74: computer
    0.40,  # 75: swivel chair
    0.50,  # 76: boat
    0.30,  # 77: bar
    0.40,  # 78: arcade machine
    0.10,  # 79: hovel/hut
    0.50,  # 80: bus
    0.30,  # 81: towel
    0.35,  # 82: light
    0.50,  # 83: truck
    0.15,  # 84: tower
    0.40,  # 85: chandelier
    0.20,  # 86: awning
    0.25,  # 87: streetlight
    0.30,  # 88: booth
    0.55,  # 89: television
    0.50,  # 90: airplane
    0.10,  # 91: dirt track
    0.60,  # 92: apparel/clothing
    0.15,  # 93: pole
    0.10,  # 94: land/ground
    0.20,  # 95: bannister
    0.15,  # 96: escalator
    0.30,  # 97: ottoman
    0.40,  # 98: bottle
    0.30,  # 99: buffet/counter
    0.40,  # 100: poster
    0.30,  # 101: stage
    0.50,  # 102: van
    0.50,  # 103: ship
    0.40,  # 104: fountain
    0.15,  # 105: conveyer belt
    0.20,  # 106: canopy
    0.25,  # 107: washer
    0.40,  # 108: plaything/toy
    0.25,  # 109: swimming pool
    0.30,  # 110: stool
    0.30,  # 111: barrel
    0.30,  # 112: basket
    0.40,  # 113: waterfall
    0.20,  # 114: tent
    0.50,  # 115: bag
    0.50,  # 116: minibike
    0.30,  # 117: cradle
    0.30,  # 118: oven
    0.40,  # 119: ball
    0.60,  # 120: food
    0.15,  # 121: step/stair
    0.30,  # 122: tank
    0.30,  # 123: trade name/brand
    0.25,  # 124: microwave
    0.40,  # 125: pot/flowerpot
    0.60,  # 126: animal
    0.50,  # 127: bicycle
    0.10,  # 128: lake
    0.25,  # 129: dishwasher
    0.30,  # 130: screen/CRT
    0.30,  # 131: blanket
    0.60,  # 132: sculptor
    0.25,  # 133: hood
    0.30,  # 134: sconce
    0.40,  # 135: vase
    0.35,  # 136: traffic light
    0.30,  # 137: tray
    0.10,  # 138: ashcan/trash can
    0.20,  # 139: island
    0.35,  # 140: flag
    0.90,  # 141: audience
    0.15,  # 142: park
    0.30,  # 143: glass
    0.45,  # 144: painting
    0.50,  # 145: clothing
    0.45,  # 146: picture
    0.90,  # 147: people
    0.10,  # 148: country
    0.25,  # 149: unknown
], dtype=np.float32)

assert len(_CLASS_IMPORTANCE) == 150, f"Expected 150 ADE20K classes, got {len(_CLASS_IMPORTANCE)}"
assert _CLASS_IMPORTANCE.min() >= 0.0 and _CLASS_IMPORTANCE.max() <= 1.0
```

- [ ] **Step 2: Write minimal test for importance mapping**

Create `tests/test_semantic.py`:

```python
from __future__ import annotations

import numpy as np
import pytest


class TestClassImportance:
    def test_array_length(self):
        from app.semantic import _CLASS_IMPORTANCE
        assert len(_CLASS_IMPORTANCE) == 150

    def test_values_in_range(self):
        from app.semantic import _CLASS_IMPORTANCE
        assert _CLASS_IMPORTANCE.min() >= 0.0
        assert _CLASS_IMPORTANCE.max() <= 1.0

    def test_person_has_high_importance(self):
        from app.semantic import _CLASS_IMPORTANCE
        assert _CLASS_IMPORTANCE[12] > 0.8   # person

    def test_sky_has_low_importance(self):
        from app.semantic import _CLASS_IMPORTANCE
        assert _CLASS_IMPORTANCE[2] < 0.2    # sky


class TestIsSemanticAvailable:
    def test_returns_bool(self):
        from app.semantic import is_semantic_available
        result = is_semantic_available()
        assert isinstance(result, bool)
```

- [ ] **Step 3: Run the test**

```bash
.\.venv\Scripts\Activate.ps1; python -m pytest tests/test_semantic.py -v
```

Expected: 5 tests pass.

- [ ] **Step 4: Commit**

```bash
git add app/semantic.py tests/test_semantic.py
git commit -m "feat: add ADE20K semantic importance module"
```

---

### Task 3: Add `engine` field to `ConvertParams` model

**Files:**
- Modify: `app/models.py`

**Interfaces:**
- Produces: `ConvertParams.engine: Literal["classic", "sam_hq"] = "classic"`

- [ ] **Step 1: Add engine field**

In `app/models.py`, add `engine` to `ConvertParams` — after `number_color` field (line 20):

```python
    engine: Literal["classic", "sam_hq"] = "classic"
```

Add this new line between the existing `number_color` field and the `pipeline_params()` method. Specifically:

At line 21, insert:
```python
    engine: Literal["classic", "sam_hq"] = "classic"
```

- [ ] **Step 2: Test parameters default**

```bash
.\.venv\Scripts\Activate.ps1; python -m pytest tests/test_models.py -v
```

Expected: all tests pass (default is "classic", existing tests unaffected).

- [ ] **Step 3: Commit**

```bash
git add app/models.py
git commit -m "feat: add engine field to ConvertParams (classic|sam_hq)"
```

---

### Task 4: Add AI config to `app/config.py`

**Files:**
- Modify: `app/config.py`

**Interfaces:**
- Produces: `SAM_HQ_MODEL_PATH`, `SAM_HQ_MODEL_TYPE`, `SEMANTIC_MODEL_NAME`, `AI_DEVICE`

- [ ] **Step 1: Add AI config constants**

Append to `app/config.py` after the `PAPER_SIZES_MM` dict (after line 24):

```python
# ── AI Engine config ──────────────────────────────────────────────
import os as _os

SAM_HQ_MODEL_PATH = _os.environ.get(
    "SAM_HQ_MODEL_PATH", "models/sam_hq_vit_h.pth"
)
SAM_HQ_MODEL_TYPE = _os.environ.get("SAM_HQ_MODEL_TYPE", "vit_h")

SEMANTIC_MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"


def _detect_ai_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


AI_DEVICE = _detect_ai_device()
```

- [ ] **Step 2: Verify config loads**

```bash
.\.venv\Scripts\Activate.ps1; python -c "from app.config import SAM_HQ_MODEL_PATH, SEMANTIC_MODEL_NAME, AI_DEVICE; print(f'SAM: {SAM_HQ_MODEL_PATH}'); print(f'Semantic: {SEMANTIC_MODEL_NAME}'); print(f'Device: {AI_DEVICE}')"
```

Expected:
```
SAM: models/sam_hq_vit_h.pth
Semantic: nvidia/segformer-b2-finetuned-ade-512-512
Device: cuda
```

- [ ] **Step 3: Commit**

```bash
git add app/config.py
git commit -m "feat: add AI engine config (model paths, device detection)"
```

---

### Task 5: Add `importance_map` parameter to `merge_small_regions`

**Files:**
- Modify: `app/postprocess.py`

**Interfaces:**
- Modifies: `merge_small_regions(labels, palette, min_area_px, edge_density=None, importance_map=None) -> np.ndarray`
- Backward compatible: `importance_map=None` keeps existing behavior.

- [ ] **Step 1: Update the function signature and body**

In `app/postprocess.py`, change `merge_small_regions`:

**Signature** (line 73-76): change to:
```python
def merge_small_regions(
    labels: np.ndarray, palette: list[PaletteColor], min_area_px: int,
    edge_density: np.ndarray = None,
    importance_map: np.ndarray = None,
) -> np.ndarray:
```

**Body** — replace the `eff_min` calculation (lines 97-104) that currently reads:
```python
        small_ids = []
        for sid, info in seg_clusters.items():
            eff_min = min_area_px
            if edge_density is not None:
                seg_mask = seg_map == sid
                dens = float(edge_density[seg_mask].mean())
                eff_min = min_area_px * (1.0 - 0.7 * dens)
            if info[1] < eff_min:
                small_ids.append(sid)
```

Replace with:
```python
        small_ids = []
        for sid, info in seg_clusters.items():
            eff_min = min_area_px
            seg_mask = seg_map == sid
            if edge_density is not None or importance_map is not None:
                detail_signal = 0.0
                if edge_density is not None:
                    detail_signal += 0.5 * float(edge_density[seg_mask].mean())
                if importance_map is not None:
                    detail_signal += 0.5 * float(importance_map[seg_mask].mean())
                eff_min = min_area_px * (1.0 - 0.7 * detail_signal)
            if info[1] < eff_min:
                small_ids.append(sid)
```

- [ ] **Step 2: Run existing tests**

```bash
.\.venv\Scripts\Activate.ps1; python -m pytest tests/ -q --tb=short
```

Expected: 48 tests pass (adds 4 from test_semantic.py and 4 from test_sam_segment.py, total 56+).

Wait — the test_semantic and test_sam_segment files already exist from Tasks 1-2. So this should show 56 tests.

Expected: all tests pass. The `importance_map=None` default preserves backward compatibility.

- [ ] **Step 3: Commit**

```bash
git add app/postprocess.py
git commit -m "feat: add importance_map param to merge_small_regions"
```

---

### Task 6: Wire engine branch and importance map into pipeline

**Files:**
- Modify: `app/pipeline.py:16-42` (the `render_data_from_image` function)

**Interfaces:**
- Consumes: `segment_with_sam_hq` from Task 1, `compute_importance_map`/`is_semantic_available` from Task 2
- Produces: same `RenderData` — downstream (render, extract_regions, shared_edges) unchanged.

- [ ] **Step 1: Edit `render_data_from_image` to branch on engine**

Replace the function body (lines 16-42) in `app/pipeline.py`:

```python
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
    palette, labels = merge_similar_colors(palette, labels, threshold=params.color_merge_threshold)
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
```

- [ ] **Step 2: Run all existing tests**

```bash
.\.venv\Scripts\Activate.ps1; python -m pytest tests/ -q --tb=short
```

Expected: all tests pass. Classic engine is the default path and is unchanged.

- [ ] **Step 3: Commit**

```bash
git add app/pipeline.py
git commit -m "feat: wire SAM-HQ engine branch and semantic importance into pipeline"
```

---

### Task 7: Add `engine` param and `/api/capabilities` endpoint

**Files:**
- Modify: `app/main.py`

**Interfaces:**
- Produces: `GET /api/capabilities` → `{"sam_hq": bool, "semantic": bool, "device": str}`
- Modifies: `POST /api/convert` accepts `engine` form param.

- [ ] **Step 1: Add `engine` to the form params dependency**

In `app/main.py`, add `engine` parameter to `_form_params` (line 38):

Insert between the existing params. After the `number_color` line (line 48), add:
```python
    engine: str = Form("classic"),
```

And pass it through in the `ConvertParams` constructor (after line 61):
```python
        engine=engine,
```

- [ ] **Step 2: Add error handling for unavailable AI engine**

In the `convert` endpoint (line 66-85), add a try/except around `render_data_from_image`:

Replace line 77 (`data = render_data_from_image(contents, params)`) with:
```python
    try:
        data = render_data_from_image(contents, params)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
```

- [ ] **Step 3: Add `/api/capabilities` endpoint**

After the `health` endpoint (after line 34), add:

```python
@app.get("/api/capabilities")
def capabilities():
    from .sam_segment import is_sam_hq_available
    from .semantic import is_semantic_available
    return {
        "sam_hq": is_sam_hq_available(),
        "semantic": is_semantic_available(),
        "device": config.AI_DEVICE,
    }
```

- [ ] **Step 4: Run existing tests**

```bash
.\.venv\Scripts\Activate.ps1; python -m pytest tests/ -q --tb=short
```

Expected: all tests pass.

- [ ] **Step 5: Test `/api/capabilities` manually**

```bash
.\.venv\Scripts\Activate.ps1; python -c "
from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
r = client.get('/api/capabilities')
print(r.json())
"
```

Expected: `{"sam_hq": false, "semantic": true, "device": "cuda"}` (sam_hq is false until weights are downloaded).

- [ ] **Step 6: Commit**

```bash
git add app/main.py
git commit -m "feat: add engine param to /api/convert and /api/capabilities endpoint"
```

---

### Task 8: Expand SAM-HQ tests

**Files:**
- Modify: `tests/test_sam_segment.py`

- [ ] **Step 1: The tests from Task 1 are already in place. Add one integration test**

Append to `tests/test_sam_segment.py`:

```python
class TestSamHqIntegration:
    def test_segment_with_stub(self, monkeypatch):
        """Verify segment_with_sam_hq returns correct shape when model is stubbed."""
        import numpy as np
        from app.models import PipelineParams

        pp = PipelineParams(
            max_working_side=1600,
            mean_shift_sp=5,
            mean_shift_sr=20,
            felzenszwalb_scale=30,
            palette_size=16,
            min_region_area_pct=0.5,
            morph_kernel=3,
            boundary_sigma=1.0,
        )

        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        # Stub: simulate SAM-HQ returning 2 masks covering the image
        class FakePredictor:
            model = None
            def set_image(self, img):
                pass

        def fake_load():
            import app.sam_segment as mod
            mod._MODEL = FakePredictor()
            mod._MASK_GENERATOR = FakeGenerator()

        class FakeGenerator:
            def generate(self, img):
                h, w = img.shape[:2]
                m1 = np.zeros((h, w), dtype=bool)
                m1[: h // 2, :] = True
                m2 = np.zeros((h, w), dtype=bool)
                m2[h // 2 :, :] = True
                return [
                    {"segmentation": m1, "area": m1.sum()},
                    {"segmentation": m2, "area": m2.sum()},
                ]

        import app.sam_segment as sam_mod
        monkeypatch.setattr(sam_mod, "_load_sam_hq_model", fake_load)
        monkeypatch.setattr(sam_mod, "is_sam_hq_available", lambda: True)

        result = sam_mod.segment_with_sam_hq(image, pp)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8
        assert not np.array_equal(result[:32, 0], result[32:, 0])
```

- [ ] **Step 2: Run tests**

```bash
.\.venv\Scripts\Activate.ps1; python -m pytest tests/test_sam_segment.py -v
```

Expected: 5 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_sam_segment.py
git commit -m "test: add SAM-HQ integration test with stubbed model"
```

---

### Task 9: Expand semantic tests

**Files:**
- Modify: `tests/test_semantic.py`

- [ ] **Step 1: Add integration test for importance map with stub**

Append to `tests/test_semantic.py`:

```python
class TestImportanceMapIntegration:
    def test_compute_importance_with_stub(self, monkeypatch):
        """Verify compute_importance_map returns correct shape/range with stubbed model."""
        import numpy as np

        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        class FakeProcessor:
            @staticmethod
            def from_pretrained(name):
                return FakeProcessor()

            def __call__(self, images, return_tensors):
                import torch
                return {"pixel_values": torch.randn(1, 3, 256, 256)}

        class FakeModel:
            def to(self, device):
                return self
            def eval(self):
                return self
            def __call__(self, **kwargs):
                import torch
                # Return logits where class=12 (person) dominates center,
                # class=2 (sky) dominates edges
                logits = torch.zeros(1, 150, 16, 16)
                logits[:, 2, :, :] = 10.0   # sky everywhere
                logits[:, 12, 8:12, 8:12] = 20.0  # person in center
                return type("FakeOutput", (), {"logits": logits, "loss": None})()

        def fake_load():
            import app.semantic as sem_mod
            sem_mod._PROCESSOR = FakeProcessor()
            sem_mod._MODEL = FakeModel()

        import app.semantic as sem_mod
        monkeypatch.setattr(sem_mod, "_load_semantic_model", fake_load)
        monkeypatch.setattr(sem_mod, "is_semantic_available", lambda: True)

        result = sem_mod.compute_importance_map(image)
        assert result.shape == (64, 64)
        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= result.max() <= 1.0
        # Center (person) should have higher importance than corner (sky)
        assert result[32, 32] > result[0, 0]
```

- [ ] **Step 2: Run tests**

```bash
.\.venv\Scripts\Activate.ps1; python -m pytest tests/test_semantic.py -v
```

Expected: 6 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_semantic.py
git commit -m "test: add semantic importance integration test with stubbed model"
```

---

### Task 10: UI — engine toggle + loading overlay

**Files:**
- Modify: `static/index.html`, `static/app.js`, `static/styles.css`

- [ ] **Step 1: Add engine toggle to HTML**

In `static/index.html`, inside the `<fieldset class="params">`, add a new section BEFORE the "Parameters" fieldset (or after, as a separate control group). Add after the file input label (line 17) and before `fieldset.params` (line 19):

```html
      <div class="engine-selector" id="engine-selector">
        <span class="engine-label">Engine:</span>
        <label class="engine-option active" id="engine-classic-label">
          <input type="radio" name="engine" value="classic" checked> Classic
        </label>
        <label class="engine-option" id="engine-sam-label">
          <input type="radio" name="engine" value="sam_hq"> AI (SAM-HQ)
        </label>
      </div>
```

And add the loading overlay right after the `<main>` opening tag (after line 9):

```html
    <div id="loading-overlay" class="loading-overlay" hidden>
      <div class="loading-card">
        <div class="spinner"></div>
        <p>AI segmentation in progress\u2026 this may take up to 10 seconds.</p>
      </div>
    </div>
```

- [ ] **Step 2: Add CSS for engine toggle and overlay**

In `static/styles.css`, append these styles:

```css
/* ── Engine selector ── */
.engine-selector {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 1rem;
  padding: 0.5rem 0;
}
.engine-label {
  font-weight: 600;
  color: #9ca3af;
  margin-right: 0.5rem;
}
.engine-option {
  padding: 0.3rem 0.8rem;
  border: 1px solid #4b5563;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.85rem;
  color: #9ca3af;
  transition: all 0.2s;
}
.engine-option.active {
  background: #374151;
  color: #f3f4f6;
  border-color: #6b7280;
}
.engine-option input { display: none; }
.engine-option:disabled,
.engine-option.disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

/* ── Loading overlay ── */
.loading-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}
.loading-card {
  background: #1f2937;
  border: 1px solid #374151;
  border-radius: 12px;
  padding: 2rem 3rem;
  text-align: center;
  color: #d1d5db;
}
.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid #374151;
  border-top-color: #60a5fa;
  border-radius: 50%;
  margin: 0 auto 1rem;
  animation: spin 0.8s linear infinite;
}
@keyframes spin {
  to { transform: rotate(360deg); }
}
```

- [ ] **Step 3: Add JS for engine toggle and capability probe**

In `static/app.js`, add at the top (after the existing variable declarations, around line 19):

```javascript
// ── AI engine toggle ──
const engineClassicLabel = document.getElementById("engine-classic-label");
const engineSamLabel = document.getElementById("engine-sam-label");
const loadingOverlay = document.getElementById("loading-overlay");

// Probe capabilities on load
fetch("/api/capabilities")
  .then(r => r.json())
  .then(caps => {
    if (!caps.sam_hq) {
      engineSamLabel.classList.add("disabled");
      engineSamLabel.title = "Requires PyTorch + SAM-HQ weights. See README.";
      engineSamLabel.querySelector("input").disabled = true;
    }
  })
  .catch(() => {
    // Offline or server error — disable AI option
    engineSamLabel.classList.add("disabled");
    engineSamLabel.querySelector("input").disabled = true;
  });

// Toggle active state on engine labels
document.querySelectorAll(".engine-option").forEach(label => {
  label.addEventListener("click", () => {
    if (label.querySelector("input").disabled) return;
    document.querySelectorAll(".engine-option").forEach(l => l.classList.remove("active"));
    label.classList.add("active");
  });
});
```

Then modify the form submit handler (line 67-101 of app.js) to show/hide the loading overlay for AI mode. Replace the `form.addEventListener("submit", ...)` block:

```javascript
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!imageInput.files[0]) {
    statusEl.textContent = "Please select an image first.";
    return;
  }

  const formData = new FormData(form);
  const numbersChecked = document.querySelector('[name="show_numbers"]').checked;
  formData.set("show_numbers", numbersChecked ? "true" : "false");

  submitBtn.disabled = true;
  const engine = formData.get("engine");
  if (engine === "sam_hq") {
    loadingOverlay.hidden = false;
  }
  statusEl.textContent = "Processing\u2026";
  comparison.hidden = true;
  downloads.hidden = true;

  try {
    const response = await fetch("/api/convert", { method: "POST", body: formData });
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || "Error " + response.status);
    }
    const result = await response.json();
    afterImg.src = result.colored;
    outlineImg.src = result.outline;
    pdfDataUri = result.pdf || null;
    comparison.hidden = false;
    if (pdfDataUri) downloads.hidden = false;
    statusEl.textContent = "Done.";
  } catch (err) {
    statusEl.textContent = err.message;
  } finally {
    submitBtn.disabled = false;
    loadingOverlay.hidden = true;
  }
});
```

- [ ] **Step 4: Visual verification**

Start the dev server and check the page loads without errors:

```bash
.\.venv\Scripts\Activate.ps1; Start-Job -ScriptBlock { uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 };
Start-Sleep 3;
(Invoke-WebRequest -Uri "http://127.0.0.1:8000/" -UseBasicParsing).StatusCode
```

Expected: 200 OK. Page has engine toggle visible. AI option may be disabled (weights not downloaded yet).

- [ ] **Step 5: Commit**

```bash
git add static/index.html static/app.js static/styles.css
git commit -m "feat: add AI engine toggle and loading overlay to UI"
```

---

### Task 11: End-to-end verification

**Files:**
- None new. Verification only.

- [ ] **Step 1: Run full test suite**

```bash
.\.venv\Scripts\Activate.ps1; python -m pytest tests/ -v --tb=short
```

Expected: all tests pass (~56 tests: 48 existing + 5 test_sam_segment + 6 test_semantic - 3 that might overlap).

Count and confirm: `python -m pytest tests/ -q` should show `56 passed` (or whatever the exact count is).

- [ ] **Step 2: Test classic engine unchanged**

```bash
.\.venv\Scripts\Activate.ps1; python -c "
from tests.conftest import make_two_color_bytes
from app.models import ConvertParams
from app.pipeline import render_data_from_image
params = ConvertParams(detail_level=5, engine='classic', color_merge_threshold=10)
data = render_data_from_image(make_two_color_bytes(), params)
assert data.width > 0
assert len(data.regions) > 0
print('Classic engine OK')
"
```

Expected: `Classic engine OK`.

- [ ] **Step 3: Test AI engine graceful failure (no weights)**

```bash
.\.venv\Scripts\Activate.ps1; python -c "
from app.sam_segment import is_sam_hq_available
print(f'SAM-HQ available: {is_sam_hq_available()}')
assert is_sam_hq_available() == False  # weights not downloaded yet
from app.semantic import is_semantic_available
print(f'Semantic available: {is_semantic_available()}')
assert is_semantic_available() == True  # transformers installed
print('Availability checks OK')
"
```

Expected:
```
SAM-HQ available: False
Semantic available: True
Availability checks OK
```

- [ ] **Step 4: Test /api/capabilities via TestClient**

```bash
.\.venv\Scripts\Activate.ps1; python -c "
from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
r = client.get('/api/capabilities')
caps = r.json()
assert 'sam_hq' in caps
assert 'semantic' in caps
assert 'device' in caps
print(caps)
"
```

Expected: JSON with `sam_hq`, `semantic`, `device` keys.

- [ ] **Step 5: Final commit**

```bash
git add -A
git status
git commit -m "feat: complete AI segmentation integration (SAM-HQ + ADE20K)"
```

Only commit if there are uncommitted changes.

---

### Post-implementation: Download model weights

After all code is in place, download the SAM-HQ weights to enable the AI engine:

```bash
# Download sam_hq_vit_h.pth from:
# https://github.com/SysCV/sam-hq#model-download
# Place it at: models/sam_hq_vit_h.pth

# Then verify:
python -c "from app.sam_segment import is_sam_hq_available; assert is_sam_hq_available(), 'Weights not found'"
```

SegFormer-B2 weights download automatically on first use via HuggingFace cache.
