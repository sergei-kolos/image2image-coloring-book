# AGENTS.md

## Commands
```powershell
.\.venv\Scripts\Activate.ps1            # activate venv (Windows)
uvicorn app.main:app --reload            # dev server
python -m pytest tests/ -q --tb=short   # all tests (48 pass)
```

## Architecture
- **FastAPI**, no DB, no build step. Python 3.8.
- **Every `.py`** has `from __future__ import annotations`.
- **`/api/convert` returns JSON** with base64-encoded colored PNG, outline PNG, and PDF.
- Pipeline entrypoint: `render_data_from_image()` in `app/pipeline.py`.
- **`detail_level` (1-15)** is the primary pipeline control. All pipeline parameters are derived from it via `ConvertParams.pipeline_params() → PipelineParams`.
- **`color_merge_threshold` (0-30, default 5)** controls `merge_similar_colors()` — user-adjustable via UI slider. Uses CIEDE2000 distance.
- **`palette_size` in `ConvertParams`** is ignored by `pipeline_params()` — `derived_palette` is computed from `detail_level` + `max_colors` cap.

## detail_level mapping (1=coarse, 15=finest)
| Parameter | Level 1 | Level 15 |
|---|---|---|
| `max_working_side` | 1200px | 3200px |
| `mean_shift_sp` | 12 | 1 |
| `mean_shift_sr` | 35 | 15 |
| `felzenszwalb_scale` | 60 | 10 |
| `palette_size` (derived) | 12 | 48 |
| `min_region_area_pct` | 1.0% | ~0.05% |
| `morph_kernel` | 5 | 0 (skip) |
| `boundary_sigma` | 1.5 | 0.3 |

## Pipeline order
1. Resize to `max_working_side` (dynamic from `detail_level`)
2. `pyrMeanShiftFilter` (sp/sr from `detail_level`) → CLAHE → unsharp masking + 2px border
3. Felzenszwalb segmentation (`scale` from `detail_level`) + flatten to mean colour
4. k‑means quantization (`palette_size` from `detail_level`, hard cap 64)
5. `merge_similar_colors` (user-controlled threshold via `color_merge_threshold`)
6. RAG merge (`min_region_area_pct` from `detail_level`) + `smooth_label_boundaries` (sigma from `detail_level`)
7. Per‑label `MORPH_OPEN` (kernel from `detail_level`; skipped at level >= 10)
8. `extract_regions` (RETR_CCOMP exterior+holes, TC89_KCOS, approxPolyDP for exterior, Chaikin for holes, pole‑of‑inaccessibility centroid, letter labels for <5px radius)
9. Strip 2px border from region coordinates
10. `render_visualization` (OpenCV: colour‑fill each region + black outlines + number labels)

## Key modules
- `app/pipeline.py` — `render_data_from_image(image_bytes, params) → RenderData`. Uses `params.pipeline_params()` for all derived values.
- `app/models.py` — `ConvertParams` (Pydantic, has `detail_level` + `color_merge_threshold` + `pipeline_params()`), `PipelineParams` (dataclass of derived values), `Region`, `PaletteColor`, `RenderData`
- `app/config.py` — paper sizes, `MAX_PALETTE_SIZE = 64`, upload limits. (`MAX_WORKING_SIDE = 1600` is dead code — `pipeline_params()` computes dynamically.)
- `app/quantize.py` — k-means → `(palette, labels)` + `merge_similar_colors(threshold)`
- `app/postprocess.py` — `merge_small_regions`, `smooth_label_boundaries`, `clean_mask(kernel_size)`, Chaikin/adaptive smoothing
- `app/regions.py` — `extract_regions(labels, palette, morph_kernel)`, `_pole_of_inaccessibility`, `_simplify_only`, `_smooth_closed`
- `app/pdf_layout.py` — `render_pdf` (ReportLab). Vector PDF with outlines, palette legend, region labels.
- `app/visualize.py` — `render_visualization` (OpenCV PNG preview) + `render_outline` (outline-only PNG)

## API
- `POST /api/convert` — multipart form: `image` file + `detail_level` (1-15, default 5) + `color_merge_threshold` (0-30, default 5) + other params. Returns JSON with colored, outline, pdf data URIs.
- `GET /api/health`
- `GET /` — serves `static/index.html` (Liquid Glass dark theme UI)

## Tests
- `tests/conftest.py` — `make_two_color_array()` (100×100, red/blue halves) and `make_two_color_bytes()` (PNG bytes).
- API tests use `httpx.TestClient`.
- Pipeline tests call `render_data_from_image()` with `color_merge_threshold=80` (aggressive merge) for deterministic palette size.
- `test_quantize.py` covers `merge_similar_colors()` (duplicates, close colors, reindexing).

## Known issues (planned for feat/pipeline-v2)
- ~~k-means operates in RGB, not perceptual Lab space~~ → fixed in feat/pipeline-v2
- ~~`merge_similar_colors` uses RGB Euclidean distance, not CIEDE2000~~ → fixed in feat/pipeline-v2
- ~~`_apply_global_morphology` creates gaps between regions~~ → removed in feat/pipeline-v2
- ~~Mean-shift + CLAHE + unsharp can destroy fine details~~ → skipped at high detail in feat/pipeline-v2
- Felzenszwalb min_size is fixed, not edge-aware (improved but not fully solved)
- `clean_mask` in `extract_regions` still processes each color independently
- See `SPEC_PIPELINE_V2.md` for remaining improvements

## Known issues (planned for feat/shared-boundaries)
- ~~Per-region contour stroke causes double lines at shared boundaries~~ → fixed with SharedEdge model
- ~~Per-color `clean_mask` morph shifts boundaries independently~~ → replaced with medianBlur only
- ~~pyrMeanShiftFiltering destroys fine details~~ → replaced with bilateralFilter
- ~~Mean flatten loses color fidelity~~ → replaced with median flatten
- ~~Aggressive merge_small wipes small important regions~~ → edge-aware threshold
- ~~Rendering artifacts from independent contour smoothing~~ → shared edges extracted from label map
- See `SPEC_SHARED_BOUNDARIES.md` for remaining improvements
