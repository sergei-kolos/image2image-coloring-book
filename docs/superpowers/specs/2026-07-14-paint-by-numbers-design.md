# Paint-by-Numbers Converter — Design Spec

**Date:** 2026-07-14
**Status:** Approved (pending implementation)
**Author:** Sergei Kolos

## 1. Purpose

A local web application that converts a photo or picture into a **paint-by-numbers** (картины по номерам) page: a print-ready PDF containing the region outlines with numbers plus a color palette legend. The user prints the PDF and paints it by hand.

## 2. Goals & Non-Goals

### Goals
- Convert a single uploaded image into a single-page PDF in a selectable paper format (A3, A4, A5, Letter, Legal).
- Produce a fixed-size color palette (user-configurable, **max 64 colors**) with numbered regions.
- Render the result as vector outlines + numbers + palette legend for high-quality printing.
- Run locally as a personal tool (no auth, no cloud, no async queues).

### Non-Goals (v1)
- Public multi-user deployment, accounts, rate limiting.
- Neural-network-based line-art extraction (classical CV only).
- A plain "line-art only" coloring mode (paint-by-numbers only).
- Batch/multi-page album UI (interface reserved, UI deferred — see §8).
- In-browser image cropping/rotation in the UI.

## 3. User Stories

1. As a user, I open `http://localhost:8000`, upload a JPG/PNG, pick a paper format and palette size, and download a print-ready PDF.
2. As a user, I can tune detail (min region area), smoothing, line thickness, and whether numbers are shown.
3. As a user, I receive a clear error if I request more than 64 colors or upload an unsupported file.

## 4. Architecture

**Stack:** FastAPI (backend) + vanilla JS/HTML/CSS (frontend, no bundler). Single process, synchronous processing.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Serve `static/index.html` |
| `POST` | `/api/convert` | Accept image + params, return `application/pdf` |
| `GET` | `/api/health` | Liveness check (`{"status":"ok"}`) |

### Request (`POST /api/convert`, `multipart/form-data`)
- `image` — file (JPG/JPEG/PNG/WebP), max size 25 MB
- `palette_size` — int, 2–64 (default 16)
- `min_region_area` — float, 0.1–10 (% of image area, default 0.5)
- `smoothing` — int, 0–10 (bilateral filter strength, default 5)
- `line_thickness` — float, 0.5–3.0 pt (default 1.0)
- `paper_size` — enum: `A3 | A4 | A5 | Letter | Legal` (default `A4`)
- `orientation` — enum: `auto | portrait | landscape` (default `auto`)
- `show_numbers` — bool (default `true`)
- `number_color` — enum: `black | gray` (default `black`)

### Response
- Success: `200 OK`, `Content-Type: application/pdf`, body = PDF bytes.
- Validation error: `422`, JSON with field-level messages.
- Unsupported/oversized file: `413` / `415`.

### Processing flow (single page)
1. Frontend posts the image and parameters with a "processing…" indicator.
2. Server runs the conversion pipeline synchronously (seconds for typical photos).
3. Server returns the PDF; the browser previews/downloads it.

## 5. Conversion Pipeline (Approach A: color-quantization-first)

Input: RGB image (Pillow) + parameters.

### Step 1 — Load & normalize (`app/pipeline.py`)
- Convert to RGB.
- Resize so the longest side ≤ `MAX_WORKING_SIDE` (default 1600 px), preserving aspect ratio.
- Apply bilateral filter to reduce noise while preserving edges; strength derived from `smoothing`.

### Step 2 — Quantize colors (`app/quantize.py`)
- Run OpenCV `k-means` on the pixel data, `K = palette_size`.
- Output: cluster centers (the palette, converted to hex/RGB) and a per-pixel label map.
- Palette indices are the **numbers** used in the output (1..N).

### Step 3 — Segment regions (`app/regions.py`)
- `cv2.connectedComponents` on the label map → connected regions of identical color.
- For each region record: cluster index (color number), area, centroid, contour.

### Step 4 — Region cleanup (`app/regions.py`)
- Drop regions with area < `min_region_area` (percent of total image area).
- Optional morphological closing to remove speckle holes inside regions.
- Final numbers stay bound to their palette color index.

### Step 5 — Prepare rendering data (`app/regions.py` + `app/pdf_layout.py`)
- Contours: `cv2.findContours` + polygon approximation → vector paths.
- Numbers: one label per region at its centroid; font size scaled to region area.
- Palette: ordered list `{number, hex, rgb}` for the legend.

## 6. PDF Layout (`app/pdf_layout.py`, ReportLab)

- Paper size from `paper_size`; orientation resolved (`auto` → chosen by image aspect ratio).
- Margins: 15 mm all sides.
- **Image area**: centered, `contain`-fit within `(pageWidth - margins, pageHeight - margins - legendHeight)`.
- **Outlines**: vector strokes (`line_thickness` pt). Region interiors left white (for painting).
- **Numbers**: vector text at each region centroid, clipped to its region, colored per `number_color`.
- **Palette legend**: bottom strip — colored squares with numbers, wrapping to additional rows as needed.
- Vector output ensures print quality independent of raster DPI.

## 7. Parameters & Validation (`app/models.py`)

Pydantic model validates form fields:

| Field | Range | Default | Notes |
|---|---|---|---|
| `palette_size` | 2–64 | 16 | **Hard cap 64**; >64 → 422 |
| `min_region_area` | 0.1–10.0 | 0.5 | percent of image area |
| `smoothing` | 0–10 | 5 | bilateral filter strength |
| `line_thickness` | 0.5–3.0 | 1.0 | PDF points |
| `paper_size` | enum | `A4` | A3/A4/A5/Letter/Legal |
| `orientation` | enum | `auto` | auto/portrait/landscape |
| `show_numbers` | bool | `true` | toggle numbers |
| `number_color` | enum | `black` | black/gray |

Server-side validation is authoritative; the frontend mirrors ranges in the UI.

## 8. Album Mode (reserved, deferred)

`POST /api/convert/album` (multipart, multiple files) → multi-page PDF. The module boundary is defined now (pipeline is reusable per-image), but **no UI and no endpoint implementation in v1**. Build only after v1 is validated.

## 9. Project Structure

```
Image2Image-ColoringBook/
├── app/
│   ├── main.py              # FastAPI app + endpoints
│   ├── pipeline.py          # Orchestrate steps 1–5
│   ├── quantize.py          # k-means quantization, palette
│   ├── regions.py           # connected components, cleanup, contours
│   ├── pdf_layout.py        # ReportLab page, legend, vector drawing
│   ├── models.py            # Pydantic schemas (palette_size <= 64)
│   └── config.py            # Constants (working side, margins, MAX_PALETTE=64)
├── static/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── tests/
│   ├── test_quantize.py
│   ├── test_regions.py
│   ├── test_pdf_layout.py
│   ├── test_api.py
│   └── fixtures/            # synthetic test images
├── samples/                 # example inputs for manual checks
├── requirements.txt
├── README.md
└── .gitignore
```

## 10. Dependencies (`requirements.txt`)

Runtime:
- `fastapi`
- `uvicorn[standard]`
- `opencv-python-headless`
- `pillow`
- `numpy`
- `reportlab`
- `python-multipart`

Dev:
- `pytest`
- `pytest-cov`

**Python:** compatible with 3.8 (installed version). No framework upgrade required.

## 11. Testing Strategy

- **Unit (pure functions):**
  - `quantize`: palette has exactly N colors; `palette_size=64` works; `palette_size>64` rejected by the model.
  - `regions`: connected-component count correct; regions below `min_region_area` removed; centroids lie inside their regions.
  - `pdf_layout`: PDF builds without error; page count = 1; legend contains all N palette entries.
- **API:** `TestClient` — valid upload → 200 `application/pdf`; out-of-range params → 422; bad file → 415.
- **Fixtures:** synthetic images (solid color blocks, gradient) for deterministic assertions.
- **Manual:** spot-check on `samples/` photos.

## 12. Open Questions

None blocking. (Album UI and possible Approach B segmentation are explicitly deferred.)

## 13. Out of Scope / Future

- Album (multi-page) UI.
- Plain line-art coloring mode.
- Neural segmentation (Approach B / SLIC / Felzenszwalb).
- Public deployment concerns (auth, cloud storage, queues).
- In-browser crop/rotate.
