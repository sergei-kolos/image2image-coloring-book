# AGENTS.md

## Commands
```powershell
.\.venv\Scripts\Activate.ps1            # activate venv (Windows)
uvicorn app.main:app --reload            # dev server
```

## Architecture
- **FastAPI**, no DB, no build step. Python 3.8.
- **Every `.py`** has `from __future__ import annotations`.
- **`/api/convert` returns JSON** with base64-encoded colored PNG, outline PNG, and PDF.
- Pipeline entrypoint: `render_data_from_image()` in `app/pipeline.py`.
- **`detail_level` (1-15)** is the primary pipeline control. All pipeline parameters are derived from it via `ConvertParams.pipeline_params() → PipelineParams`.

## detail_level mapping (1=coarse, 15=finest)
| Parameter | Level 1 | Level 15 |
|---|---|---|
| `max_working_side` | 1200px | 3200px |
| `mean_shift_sp` | 12 | 1 |
| `mean_shift_sr` | 35 | 15 |
| `felzenszwalb_scale` | 60 | 10 |
| `palette_size` | 12 | 48 |
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
- `app/models.py` — `ConvertParams` (Pydantic, has `detail_level` + `pipeline_params()`), `PipelineParams` (dataclass of derived values), `Region`, `PaletteColor`, `RenderData`
- `app/config.py` — paper sizes, thresholds, `MAX_PALETTE_SIZE = 64`, `MAX_WORKING_SIDE = 1600` (fallback)
- `app/quantize.py` — k-means → `(palette, labels)` + `merge_similar_colors`
- `app/postprocess.py` — `merge_small_regions`, `smooth_label_boundaries`, `clean_mask(kernel_size)`, Chaikin/adaptive smoothing
- `app/regions.py` — `extract_regions(labels, palette, morph_kernel)`, `_pole_of_inaccessibility`, `_simplify_only`, `_smooth_closed`
- `app/pdf_layout.py` — `render_pdf` (ReportLab). Vector PDF with outlines, palette legend, region labels.
- `app/visualize.py` — `render_visualization` (OpenCV PNG preview) + `render_outline` (outline-only PNG)

## API
- `POST /api/convert` — multipart form: `image` file + `detail_level` (1-15, default 5) + other params. Returns JSON with colored, outline, pdf data URIs.
- `GET /api/health`
- `GET /` — serves `static/index.html` (Liquid Glass dark theme UI)
