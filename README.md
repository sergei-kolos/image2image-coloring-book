# Paint-by-Numbers Generator

Turn any photo or image into a printable paint-by-numbers PDF. Upload an image, tweak the settings, and get a vector coloring page with numbered regions, color palette, and crisp outlines.

## Features

- **Color quantization** — k-means reduction to 2–64 colors
- **Region segmentation** — Felzenszwalb superpixels → RAG merge → contour extraction with RETR_CCOMP (exterior + holes)
- **Contour smoothing** — TC89_KCOS approximation + Chaikin corner-cutting for clean vector paths
- **Stroke trapping** — adjacent regions overlap by 0.5pt to eliminate white gaps between fills
- **Vector PDF** — ReportLab rendering with single-pass black outlines, palette legend, and region labels
- **Configurable** — palette size, minimum region area, smoothing strength, line thickness, paper size & orientation

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000`, upload an image, adjust parameters, and click "Generate".

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/api/health` | Health check |
| POST | `/api/convert` | Upload image → download PDF (multipart form: `image` + `palette_size`, `min_region_area`, `smoothing`, etc.) |

## Tests

```powershell
python -m pytest -q --tb=short
```

44 tests covering: pipeline integration, region extraction, contour geometry, pole-of-inaccessibility centroid, PDF layout, and full API round-trip.

## Project Structure

```
app/
  main.py            — FastAPI server & routes
  pipeline.py        — full conversion pipeline
  models.py          — Pydantic params, dataclasses (Region, PaletteColor, RenderData)
  config.py          — paper sizes, thresholds
  quantize.py        — k-means color quantization
  regions.py         — contour extraction (RETR_CCOMP, TC89_KCOS, Chaikin)
  postprocess.py     — RAG merge, label boundary smoothing
  pdf_layout.py      — ReportLab PDF rendering
  visualize.py       — OpenCV preview rendering
static/
  index.html, app.js, styles.css — single-page UI
tests/
  conftest.py        — test fixtures (two-color synthetic images)
  test_pipeline.py
  test_regions.py
  test_pdf_layout.py
  test_api.py
```

## Pipeline

1. `pyrMeanShiftFilter` — edge-preserving noise reduction
2. Felzenszwalb segmentation + flatten
3. k-means quantization
4. RAG merge of small regions + `smooth_label_boundaries`
5. `extract_regions` — RETR_CCOMP hierarchy, TC89_KCOS approximation, Chaikin smoothing, pole-of-inaccessibility centroids
6. `render_pdf` — color fills with stroke trapping + single-pass black outlines + palette legend
