# SAM-HQ + ADE20K Semantic AI Integration

**Date:** 2026-07-18
**Status:** Approved design, ready for implementation plan
**Branch:** `feat/ai-segmentation` (to be created from `feat/shared-boundaries`)

## Goal

Add AI-based image segmentation (SAM-HQ) and semantic importance mapping
(ADE20K) to the coloring book pipeline, as optional engines selectable from
the UI. The classic Felzenszwalb pipeline stays as the default and fallback.

### Problems solved

1. **Better object boundaries.** Felzenszwalb segments on color/texture
   gradients and loses semantic object edges (hair strands, fingers, fabric
   folds). SAM-HQ produces object-aware masks with high-quality boundaries.
2. **Importance-aware detail preservation.** Currently `merge_small_regions`
   uses only Canny edge density to decide where to keep details. ADE20K
   semantic classes add a second signal: faces/hands = preserve, sky/walls =
   simplify.

## Non-goals

- Cloud APIs (Replicate, HuggingFace Inference) — local GPU only.
- Text-guided segmentation (Grounding DINO + SAM).
- Automatic palette suggestion via AI.
- Python upgrade (stay on 3.8; use torch 2.1.x).

## Architecture

### New modules

- **`app/sam_segment.py`** — SAM-HQ integration. Lazy-loaded, globally
  cached model. Produces a label map in the same format as Felzenszwalb
  output (integer label per pixel).
- **`app/semantic.py`** — ADE20K semantic segmentation via SegFormer-B2.
  Produces a per-pixel importance map (float 0..1).

### Data flow

```
load image → resize → bilateralFilter (denoise)
                ↓
       ┌──── engine branch ─────┐
       │ "classic"               │ "sam_hq"
       │ Felzenszwalb            │ SAM-HQ automatic mask gen
       │   → segments            │   → masks → label map
       └─────────┬───────────────┘
                 ↓
         median flatten (per-label)
                 ↓
         k-means quantization (Lab, palette_size)
                 ↓
         merge_similar_colors (CIEDE2000)
                 ↓
    compute edge_density (Canny)
    compute importance_map (ADE20K, if semantic available)
                 ↓
    merge_small_regions(min_area, edge_density, importance_map)
                 ↓
    smooth_label_boundaries (if sigma >= 0.5)
                 ↓
    extract_regions + extract_shared_edges
                 ↓
    strip border + clip edges
                 ↓
    render (PDF + PNG)
```

### Key invariant

SAM-HQ and ADE20K are **optional**. The app must boot, serve `/api/convert`
with `engine="classic"`, and pass all 48 existing tests when `torch` is not
installed. AI modules are imported lazily inside their functions.

## SAM-HQ integration (`app/sam_segment.py`)

### Public API

```python
def segment_with_sam_hq(image: np.ndarray, pp: PipelineParams) -> np.ndarray:
    """Return label map (H×W int32) from SAM-HQ automatic mask generation."""

def is_sam_hq_available() -> bool:
    """True if torch + sam-hq + weights are loadable."""
```

### Model loading

- Lazy: first call triggers `load_sam_hq_model()` which caches the model in
  a module-level `_MODEL` variable.
- Source: `sam-hq` package from
  [SysCV/sam-hq](https://github.com/SysCV/sam-hq), `SamHQPredictor` class.
- Weights: `models/sam_hq_vit_h.pth` (~2.5 GB). Downloaded manually (URL in
  config). Not committed to git.
- Device: `cuda` if available else `cpu`. Configurable via `AI_DEVICE`.

### Automatic mask generation

Use `SamAutomaticMaskGenerator` from `segment_anything` with SAM-HQ model:

```python
mask_generator = SamAutomaticMaskGenerator(
    model,
    points_per_side=32,            # grid density
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    min_mask_region_area=min_area_px,  # from PipelineParams
)
masks = mask_generator.generate(image)  # list of dicts
```

### Masks → label map conversion

SAM produces overlapping masks. Convert to non-overlapping label map:

1. Sort masks by area ascending (smallest first).
2. Paint each mask onto the label canvas with its index; smaller masks
   overwrite larger ones (most specific region wins).
3. Uncovered pixels get label `0` (background).

```python
def _masks_to_labels(masks: list[dict], h: int, w: int) -> np.ndarray:
    labels = np.zeros((h, w), dtype=np.int32)
    ordered = sorted(masks, key=lambda m: m["area"], reverse=True)  # large first
    for idx, m in enumerate(ordered, start=1):
        labels[m["segmentation"]] = idx
    return labels
```

*Large-first assignment* means small masks painted later overwrite the
background label, preserving fine details.

### Performance budget

- Model load (one-time): ~10 s on RTX 3060.
- Inference per image (1024×1024): ~3–5 s on CUDA, ~20–30 s on CPU.

## ADE20K semantic importance (`app/semantic.py`)

### Public API

```python
def compute_importance_map(image: np.ndarray) -> np.ndarray:
    """Return H×W float32 array in [0, 1]. 1 = preserve detail."""

def is_semantic_available() -> bool:
    """True if transformers + model are loadable."""
```

### Model

- SegFormer-B2 fine-tuned on ADE20K
  (`nvidia/segformer-b2-finetuned-ade-512-512`) via HuggingFace
  `transformers`.
- Lazy load, module-level cache (`_MODEL`, `_PROCESSOR`).
- Device: same `AI_DEVICE` as SAM-HQ.

```python
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(AI_DEVICE)
```

### ADE20K class → importance mapping

150 ADE20K semantic classes mapped to a 0..1 importance scalar. Values
stored as a length-150 lookup table indexed by class id.

| Tier | Importance | Example classes |
|---|---|---|
| Critical | 0.90 | person, man, woman, child, face |
| High | 0.75 | hand, arm, foot, head, hair |
| Medium-high | 0.60 | dog, cat, bird, animal, fish |
| Medium | 0.40 | clothing, shirt, car, chair, food |
| Low-medium | 0.25 | tree, plant, building, window |
| Low | 0.10 | sky, wall, floor, ceiling, road |
| Minimal | 0.05 | grass, sand, water, earth, background |

Full table will be authored in `app/semantic.py` as `_CLASS_IMPORTANCE`
(array of 150 floats).

### Post-processing

1. Vectorized lookup: `importance = _CLASS_IMPORTANCE[seg_logits.argmax(1)]`
2. Upsample from 512×512 (model output) to image H×W via `cv2.resize`.
3. Gaussian blur (31×31, sigma ~8) to soften class boundary transitions.

### Performance budget

- Model load (one-time): ~3 s.
- Inference per image: ~0.5 s on CUDA.

## Pipeline changes (`app/pipeline.py`)

### Engine branch

```python
def render_data_from_image(image_bytes: bytes, params: ConvertParams) -> RenderData:
    pp = params.pipeline_params()
    image = _load_and_normalize(image_bytes, pp)
    if params.engine == "sam_hq":
        from .sam_segment import segment_with_sam_hq
        image = segment_with_sam_hq(image, pp)  # returns flattened RGB
    else:
        image = _segment_and_flatten(image, pp)  # Felzenszwalb path (unchanged)
    # ...rest unchanged: kmeans, merge, regions, edges, render
```

### `merge_small_regions` signature

Current:
```python
def merge_small_regions(labels, palette, min_area_px, edge_density=None):
```

New (backward compatible — `importance_map=None` preserves old behavior):
```python
def merge_small_regions(
    labels, palette, min_area_px,
    edge_density=None,
    importance_map=None,
):
    if importance_map is not None:
        detail_signal = 0.5 * edge_density + 0.5 * importance_map
    elif edge_density is not None:
        detail_signal = edge_density
    else:
        detail_signal = None
    # rest unchanged, uses detail_signal in place of edge_density
```

### Importance map wiring

```python
edge_density = _compute_edge_density(image)
importance_map = None
if params.engine == "sam_hq":
    from .semantic import compute_importance_map, is_semantic_available
    if is_semantic_available():
        importance_map = compute_importance_map(image)
labels = merge_small_regions(labels, palette, min_area_px, edge_density, importance_map)
```

Semantic importance runs **only when `engine="sam_hq"`** to keep classic mode
fast and dependency-free. (Can be enabled for classic mode later if desired.)

## Models / API changes

### `app/models.py`

Add `engine` field to `ConvertParams`:
```python
class ConvertParams(BaseModel):
    engine: Literal["classic", "sam_hq"] = "classic"
    # ...existing fields unchanged
```

### `app/config.py`

```python
import os

SAM_HQ_MODEL_PATH = os.environ.get(
    "SAM_HQ_MODEL_PATH", "models/sam_hq_vit_h.pth"
)
SEMANTIC_MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"

def _detect_ai_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

AI_DEVICE = _detect_ai_device()
```

### `app/main.py`

- `POST /api/convert` reads `engine` from form data (defaults to `"classic"`).
- New `GET /api/capabilities` endpoint:
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
- On `engine="sam_hq"` request when unavailable: HTTP 503 with JSON body
  `{"detail": "SAM-HQ not available: torch not installed"}`.

## UI changes (`static/index.html`, `static/styles.css`, `static/app.js`)

### Engine selector

Add a segmented toggle below the file input:
- **Классический** (default, active)
- **ИИ (SAM-HQ)** — disabled if `/api/capabilities` reports `sam_hq: false`.

### Capability probe on load

```javascript
fetch("/api/capabilities")
  .then(r => r.json())
  .then(caps => {
    if (!caps.sam_hq) {
      document.getElementById("engine-sam").disabled = true;
      document.getElementById("engine-sam").title =
        "Требуется PyTorch и веса SAM-HQ. См. README.";
    }
  });
```

### Loading overlay

Since AI mode takes ~4–6 s, show a full-card overlay with spinner and the
text «ИИ-сегментация… может занять до 10 секунд». Hidden on response or
error.

## Dependencies and installation

### `requirements.txt` (additions)

```
# AI engine (optional — app works without these)
torch==2.1.2
torchvision==0.16.2
transformers>=4.30,<5.0
segment-anything @ git+https://github.com/facebookresearch/segment-anything.git
sam-hq @ git+https://github.com/SysCV/sam-hq.git
```

These are **optional**: the app detects `ImportError` at runtime and reports
`sam_hq: false` via `/api/capabilities`. Classic engine is always available.

### Model weights

- `models/sam_hq_vit_h.pth` — download manually from
  [SAM-HQ releases](https://github.com/SysCV/sam-hq#model-download).
  Place at path defined by `SAM_HQ_MODEL_PATH` (default `models/`).
- SegFormer-B2 weights download automatically on first use via HuggingFace
  `transformers` cache (`~/.cache/huggingface/`).
- `models/` directory is gitignored.

### Installation task (first implementation step)

1. `pip install torch==2.1.2 torchvision==0.16.2 --index-url
   https://download.pytorch.org/whl/cu121` (CUDA 12.1 wheels for RTX 3060).
2. `pip install transformers>=4.30 segment-anything` and clone `sam-hq` into
   `vendor/sam-hq` (pip install from git).
3. Verify `python -c "import torch; print(torch.cuda.is_available())"`
   prints `True`.
4. Download `sam_hq_vit_h.pth` into `models/`.
5. Smoke test:
   ```python
   from app.sam_segment import is_sam_hq_available
   assert is_sam_hq_available()
   ```

## Error handling

| Condition | Behavior |
|---|---|
| `torch` not installed | `is_sam_hq_available()` → False. UI disables AI toggle. Classic engine works. |
| CUDA not available | Falls back to CPU (`AI_DEVICE = "cpu"`). ~5× slower. UI warns. |
| SAM-HQ weights missing | `is_sam_hq_available()` → False. Logs path it expects. |
| SegFormer download fails | `is_semantic_available()` → False. SAM-HQ runs without importance map. |
| Inference runtime error | Caught in API layer, returns HTTP 500 with message. Classic engine can be retried. |

## Testing

### New tests

- **`tests/test_sam_segment.py`**
  - `_masks_to_labels()` — synthetic masks (overlap, containment,
    background). Verify label assignment: small masks win over large.
  - `segment_with_sam_hq()` — monkeypatch `load_sam_hq_model` with a stub
    returning a fixed label map. Verify the function returns a label map of
    the right shape.
  - `is_sam_hq_available()` — True/False paths via mocking imports.

- **`tests/test_semantic.py`**
  - `_CLASS_IMPORTANCE` — all 150 entries in [0, 1], lookup returns sane
    values for known class ids (person → high, sky → low).
  - `compute_importance_map()` — monkeypatch model with a stub returning
    fixed class ids. Verify output shape, range, smoothing.
  - Importance map integration with `merge_small_regions`: high importance
    preserves small regions, low importance merges them.

### Existing tests

All 48 existing tests run unchanged with `engine="classic"` (the default).
No test requires torch or network access.

## File structure (new + modified)

```
app/
  sam_segment.py        NEW — SAM-HQ integration
  semantic.py           NEW — ADE20K importance map
  pipeline.py           MODIFIED — engine branch + importance wiring
  models.py             MODIFIED — ConvertParams.engine field
  config.py             MODIFIED — SAM_HQ_MODEL_PATH, AI_DEVICE
  main.py               MODIFIED — engine param + /api/capabilities
  postprocess.py        MODIFIED — merge_small_regions importance_map arg
static/
  index.html            MODIFIED — engine toggle + loading overlay
  styles.css            MODIFIED — toggle styles + overlay
  app.js                MODIFIED — capability probe + engine param
tests/
  test_sam_segment.py   NEW
  test_semantic.py      NEW
models/
  sam_hq_vit_h.pth      NEW (gitignored, downloaded manually)
requirements.txt        MODIFIED — optional AI deps
.gitignore              MODIFIED — models/, vendor/, HF cache
README.md               MODIFIED — AI engine setup instructions
```

## Open questions for implementation plan

1. Whether to add a `semantic_only` mode (classic Felzenszwalb + ADE20K
   importance, without SAM-HQ). Out of scope for this spec — can be added
   later by extending the `engine` enum.
2. Whether to expose SAM-HQ params (`points_per_side`,
   `stability_score_thresh`) in the UI. Defer — keep `detail_level` as the
   single user-facing knob and derive SAM params from it in
   `pipeline_params()`.
3. Memory budget: SAM-HQ ViT-H uses ~6 GB VRAM at 1024×1024. RTX 3060 has
   12 GB, so there is headroom. If user uploads a very large image, the
   existing `max_working_side` resize protects against OOM.
