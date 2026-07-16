# AGENTS.md

## Commands
```powershell
uvicorn app.main:app --reload          # dev server
python -m pytest -q --tb=short         # all tests (44 pass)
python -m pytest tests/test_X.py::test_Y -xvs  # single test
```

## Architecture
- **FastAPI**, no DB, no build step.
- Pipeline: `render_data_from_image()` in `app/pipeline.py`.
- Every `.py` has `from __future__ import annotations` (Python 3.8).
- `.venv\Scripts\Activate.ps1` to activate venv on Windows.

## Pipeline order
1. `pyrMeanShiftFilter` (noise reduction)
2. Felzenszwalb segmentation + flatten
3. k‑means quantization (`palette_size`, hard cap 64)
4. RAG merge + `smooth_label_boundaries` (adaptive sigma)
5. `extract_regions` (RETR_CCOMP exterior+holes, TC89_KCOS, Chaikin smooth, pole‑of‑inaccessibility centroid, letter labels for <5px radius)
6. `render_pdf` (draw exterior + holes via ReportLab paths)

## Key modules
- `app/models.py` — `ConvertParams` (Pydantic), dataclasses (`Region` with `contour`+`holes`, `PaletteColor`, `RenderData`)
- `app/config.py` — paper sizes, thresholds, `MAX_PALETTE_SIZE = 64`
- `app/postprocess.py` — `merge_small_regions`, `smooth_label_boundaries`
- `app/regions.py` — `extract_regions`, `_pole_of_inaccessibility` (distance‑transform centroid), `_chaikin_closed`
- `app/pdf_layout.py` — `render_pdf` (ReportLab)

## API
- `POST /api/convert` — multipart form: `image` file + params (`palette_size`, `min_region_area`, `smoothing`, etc.). Returns `application/pdf`.
- `GET /api/health`
- `GET /` — serves `static/index.html`

## Tests
- `tests/conftest.py` — `make_two_color_array()` (100×100, red/blue halves) and `make_two_color_bytes()` (PNG bytes).
- API tests use `httpx.TestClient`.
- Pipeline tests call `render_data_from_image()` with `smoothing=0` for speed.

