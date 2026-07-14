# Paint-by-Numbers Converter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local FastAPI web app that converts an uploaded image into a print-ready PDF paint-by-numbers page (numbered region outlines + color palette legend).

**Architecture:** FastAPI backend serves a vanilla-JS frontend and exposes `POST /api/convert`. A classical-CV pipeline (k-means color quantization → connected-component regions → vector PDF via ReportLab) runs synchronously per request. Palette size is capped at 64 colors.

**Tech Stack:** Python 3.8, FastAPI, OpenCV (headless), Pillow, NumPy, ReportLab, Pydantic. Tests with pytest + FastAPI TestClient.

## Global Constraints

- **Python 3.8 compatible** — every module starts with `from __future__ import annotations`; use typing.Literal (3.8+); numpy/opencv versions must resolve on 3.8 (pip handles).
- **Palette hard cap: 64 colors** — `MAX_PALETTE_SIZE = 64`; `ConvertParams.palette_size` is `Field(ge=2, le=64)`.
- **Local-only tool** — no auth, no cloud, no async queues; processing is synchronous.
- **v1 scope = single-image single-page PDF only** — album/multi-page UI and endpoint are deferred (do NOT implement in this plan).
- **Paint-by-numbers only** — no plain line-art mode.
- **Paper formats:** A3, A4, A5, Letter, Legal (selectable); orientation auto/portrait/landscape.
- **PDF output is vector** (ReportLab paths + text) for print quality.
- **TDD discipline** — write failing test first, then implement; commit after each task.
- **Conventional commits** — `feat:`, `test:`, `chore:`, `docs:`.
- **Working directory:** `E:\WorkRepos\Image2Image-ColoringBook` (git already initialized, design spec committed at `eba54ae`).

## File Structure

| File | Responsibility |
|---|---|
| `requirements.txt` | Pinned-by-name runtime + dev dependencies |
| `.gitignore` | Ignore venv, caches, pytest cache, uploads |
| `app/__init__.py` | Package marker |
| `app/config.py` | Constants (palette cap, paper sizes, layout metrics) |
| `app/models.py` | `ConvertParams` (Pydantic) + `PaletteColor`/`Region`/`RenderData` dataclasses |
| `app/quantize.py` | k-means color quantization → palette + label map |
| `app/regions.py` | Connected components via contours, cleanup, contour approximation |
| `app/pipeline.py` | Orchestrate load → normalize → quantize → regions → `RenderData` |
| `app/pdf_layout.py` | Layout geometry + ReportLab vector rendering → PDF bytes |
| `app/main.py` | FastAPI app, endpoints, form parsing, file validation |
| `static/index.html`, `static/app.js`, `static/styles.css` | Frontend UI |
| `tests/conftest.py` | Shared synthetic image fixtures |
| `tests/test_models.py`, `test_quantize.py`, `test_regions.py`, `test_pipeline.py`, `test_pdf_layout.py`, `test_api.py` | Unit + API tests |
| `README.md` | Setup and usage docs |

---

## Task 1: Project scaffolding & dependencies

**Files:**
- Create: `requirements.txt`, `.gitignore`, `app/__init__.py`, `tests/__init__.py`, `app/config.py`, `tests/conftest.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `app.config.MAX_PALETTE_SIZE` (int = 64), `app.config.PAPER_SIZES_MM` (dict), `app.config.DEFAULT_PALETTE_SIZE` (int = 16), shared fixtures `make_two_color_array()` and `make_two_color_bytes()` in `tests/conftest.py`.

- [ ] **Step 1: Create `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
.venv/
venv/
env/
.pytest_cache/
.coverage
htmlcov/
*.egg-info/
.DS_Store
uploads/
*.pdf
!samples/*.pdf
```

- [ ] **Step 2: Create `requirements.txt`**

```text
fastapi
uvicorn[standard]
opencv-python-headless
pillow
numpy
reportlab
python-multipart
pydantic
pytest
pytest-cov
httpx
```

- [ ] **Step 3: Create package markers**

`app/__init__.py` (empty) and `tests/__init__.py` (empty) — create both as empty files.

- [ ] **Step 4: Create `app/config.py`**

```python
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
```

- [ ] **Step 5: Create `tests/conftest.py` with shared fixtures**

```python
from __future__ import annotations

import numpy as np
from io import BytesIO
from PIL import Image


def make_two_color_array() -> np.ndarray:
    """100x100 RGB image: red top half, blue bottom half."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :] = (220, 20, 20)
    img[50:, :] = (20, 20, 220)
    return img


def make_two_color_bytes() -> bytes:
    """PNG bytes of make_two_color_array."""
    buf = BytesIO()
    Image.fromarray(make_two_color_array()).save(buf, format="PNG")
    return buf.getvalue()
```

- [ ] **Step 6: Write the failing smoke test `tests/test_config.py`**

```python
from app import config


def test_palette_cap_is_64():
    assert config.MAX_PALETTE_SIZE == 64


def test_default_palette_is_16():
    assert config.DEFAULT_PALETTE_SIZE == 16


def test_paper_sizes_present():
    for name in ("A3", "A4", "A5", "Letter", "Legal"):
        assert name in config.PAPER_SIZES_MM
        assert len(config.PAPER_SIZES_MM[name]) == 2
```

- [ ] **Step 7: Create virtualenv and install dependencies**

Run:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
Expected: all packages install without error on Python 3.8.

- [ ] **Step 8: Run the smoke test to verify the harness works**

Run: `python -m pytest tests/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 9: Commit**

```powershell
git add requirements.txt .gitignore app/ tests/
git commit -m "chore: scaffold project, config, and test harness"
```

---

## Task 2: Pydantic params + shared dataclasses

**Files:**
- Create: `app/models.py`, `tests/test_models.py`

**Interfaces:**
- Produces: `ConvertParams` (Pydantic `BaseModel`) with fields `palette_size (2..64)`, `min_region_area (0.1..10)`, `smoothing (0..10)`, `line_thickness (0.5..3.0)`, `paper_size (Literal)`, `orientation (Literal)`, `show_numbers (bool)`, `number_color (Literal)`; dataclasses `PaletteColor(index:int, hex:str, rgb:tuple)`, `Region(color_index:int, area:int, centroid:tuple, contour)`, `RenderData(width:int, height:int, palette:list, regions:list)`.

- [ ] **Step 1: Write the failing test `tests/test_models.py`**

```python
import pytest
from pydantic import ValidationError

from app.models import ConvertParams


def test_defaults():
    p = ConvertParams()
    assert p.palette_size == 16
    assert p.paper_size == "A4"
    assert p.orientation == "auto"
    assert p.show_numbers is True
    assert p.number_color == "black"
    assert p.line_thickness == 1.0


def test_palette_size_64_ok():
    assert ConvertParams(palette_size=64).palette_size == 64


def test_palette_size_65_rejected():
    with pytest.raises(ValidationError):
        ConvertParams(palette_size=65)


def test_palette_size_1_rejected():
    with pytest.raises(ValidationError):
        ConvertParams(palette_size=1)


def test_bad_paper_size_rejected():
    with pytest.raises(ValidationError):
        ConvertParams(paper_size="A2")


def test_line_thickness_bounds():
    with pytest.raises(ValidationError):
        ConvertParams(line_thickness=0.1)
    with pytest.raises(ValidationError):
        ConvertParams(line_thickness=5.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models'`.

- [ ] **Step 3: Create `app/models.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


class ConvertParams(BaseModel):
    palette_size: int = Field(default=16, ge=2, le=64)
    min_region_area: float = Field(default=0.5, ge=0.1, le=10.0)
    smoothing: int = Field(default=5, ge=0, le=10)
    line_thickness: float = Field(default=1.0, ge=0.5, le=3.0)
    paper_size: Literal["A3", "A4", "A5", "Letter", "Legal"] = "A4"
    orientation: Literal["auto", "portrait", "landscape"] = "auto"
    show_numbers: bool = True
    number_color: Literal["black", "gray"] = "black"


@dataclass
class PaletteColor:
    index: int        # 1-based number shown to the user
    hex: str          # "#RRGGBB"
    rgb: tuple        # (r, g, b) 0..255


@dataclass
class Region:
    color_index: int          # 1-based, matches PaletteColor.index
    area: int                 # pixel area
    centroid: tuple           # (x, y) in image pixel coords
    contour: object           # np.ndarray of shape (N, 2)


@dataclass
class RenderData:
    width: int
    height: int
    palette: list             # list[PaletteColor]
    regions: list             # list[Region]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_models.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```powershell
git add app/models.py tests/test_models.py
git commit -m "feat: add ConvertParams schema and render dataclasses"
```

---

## Task 3: Color quantization (k-means)

**Files:**
- Create: `app/quantize.py`, `tests/test_quantize.py`

**Interfaces:**
- Consumes: `app.models.PaletteColor`
- Produces: `quantize(image: np.ndarray, palette_size: int) -> tuple[list[PaletteColor], np.ndarray]` where `image` is `HxWx3` RGB uint8 and the returned `labels` array is `HxW` int32 with 0-based cluster ids; `palette` is length `palette_size` with 1-based `index` (1..N).

- [ ] **Step 1: Write the failing test `tests/test_quantize.py`**

```python
import numpy as np

from app.quantize import quantize
from .conftest import make_two_color_array


def test_palette_has_requested_size():
    img = make_two_color_array()
    palette, labels = quantize(img, palette_size=8)
    assert len(palette) == 8
    assert [c.index for c in palette] == [1, 2, 3, 4, 5, 6, 7, 8]


def test_palette_hex_format():
    img = make_two_color_array()
    palette, _ = quantize(img, palette_size=4)
    for c in palette:
        assert c.hex.startswith("#")
        assert len(c.hex) == 7


def test_labels_shape_and_range():
    img = make_two_color_array()
    palette, labels = quantize(img, palette_size=4)
    assert labels.shape == (100, 100)
    assert labels.min() >= 0
    assert labels.max() <= 3


def test_two_color_image_splits_into_two_dominant_regions():
    img = make_two_color_array()
    _, labels = quantize(img, palette_size=2)
    # Top half one cluster, bottom half another (cluster id order is arbitrary)
    top = labels[:50, :]
    bottom = labels[50:, :]
    assert len(np.unique(top)) == 1
    assert len(np.unique(bottom)) == 1
    assert top[0, 0] != bottom[0, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_quantize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.quantize'`.

- [ ] **Step 3: Create `app/quantize.py`**

```python
from __future__ import annotations

import cv2
import numpy as np

from .models import PaletteColor


def quantize(image: np.ndarray, palette_size: int):
    """Run k-means color quantization on an RGB uint8 image.

    Returns (palette, labels):
      palette: list[PaletteColor] of length palette_size (1-based indices)
      labels:  HxW int32 array of 0-based cluster ids
    """
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels_flat, centers = cv2.kmeans(
        pixels, palette_size, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    centers = centers.astype(np.uint8)

    palette = [
        PaletteColor(
            index=i + 1,
            hex=_to_hex(centers[i]),
            rgb=(int(centers[i][0]), int(centers[i][1]), int(centers[i][2])),
        )
        for i in range(palette_size)
    ]
    labels = labels_flat.reshape(h, w).astype(np.int32)
    return palette, labels


def _to_hex(rgb) -> str:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return "#%02X%02X%02X" % (r, g, b)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_quantize.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```powershell
git add app/quantize.py tests/test_quantize.py
git commit -m "feat: add k-means color quantization"
```

---

## Task 4: Region extraction (connected components via contours)

**Files:**
- Create: `app/regions.py`, `tests/test_regions.py`

**Interfaces:**
- Consumes: `app.models.PaletteColor`, `app.models.Region`, labels array + palette from `quantize`
- Produces: `extract_regions(labels: np.ndarray, palette: list[PaletteColor], min_region_area: float) -> list[Region]`. `min_region_area` is a percent of total image area; regions smaller than that are dropped. Each returned `Region` has `.color_index` (1-based), `.area`, `.centroid`, `.contour` (np.ndarray shape `(N,2)`).

- [ ] **Step 1: Write the failing test `tests/test_regions.py`**

```python
import numpy as np

from app.quantize import quantize
from app.regions import extract_regions
from .conftest import make_two_color_array


def test_two_color_yields_two_regions():
    img = make_two_color_array()
    palette, labels = quantize(img, palette_size=2)
    regions = extract_regions(labels, palette, min_region_area=0.1)
    assert len(regions) == 2
    assert {r.color_index for r in regions} == {1, 2}


def test_centroids_inside_image():
    img = make_two_color_array()
    palette, labels = quantize(img, palette_size=2)
    regions = extract_regions(labels, palette, min_region_area=0.1)
    for r in regions:
        cx, cy = r.centroid
        assert 0 <= cx <= 100
        assert 0 <= cy <= 100


def test_contour_is_polygon_array():
    img = make_two_color_array()
    palette, labels = quantize(img, palette_size=2)
    regions = extract_regions(labels, palette, min_region_area=0.1)
    for r in regions:
        assert r.contour.ndim == 2
        assert r.contour.shape[1] == 2
        assert r.contour.shape[0] >= 3  # a polygon


def test_min_region_area_filters_small_regions():
    # 4-quadrant image: each quadrant 50x50 = 2500px (25% of 10000)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50] = (255, 0, 0)
    img[:50, 50:] = (0, 255, 0)
    img[50:, :50] = (0, 0, 255)
    img[50:, 50:] = (255, 255, 0)
    palette, labels = quantize(img, palette_size=4)
    # 30% threshold drops everything (each quadrant is 25%)
    regions = extract_regions(labels, palette, min_region_area=30.0)
    assert regions == []
    # 10% threshold keeps all four
    regions = extract_regions(labels, palette, min_region_area=10.0)
    assert len(regions) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_regions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.regions'`.

- [ ] **Step 3: Create `app/regions.py`**

```python
from __future__ import annotations

import cv2
import numpy as np

from .models import PaletteColor, Region

_CLOSE_KERNEL = np.ones((3, 3), np.uint8)


def extract_regions(labels: np.ndarray, palette, min_region_area: float):
    """Extract paintable regions from a cluster label map.

    For each palette color, build a binary mask, apply a light morphological
    close to remove speckle holes, then take external contours as regions.
    Drops regions whose area < min_region_area percent of the image area.
    """
    total_pixels = labels.size
    min_px = total_pixels * (min_region_area / 100.0)

    regions = []
    for color in palette:
        cluster_id = color.index - 1  # 0-based
        mask = (labels == cluster_id).astype(np.uint8)
        if mask.sum() == 0:
            continue

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLOSE_KERNEL)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_px:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            approx = cv2.approxPolyDP(
                contour, 0.02 * cv2.arcLength(contour, True), True
            ).reshape(-1, 2)
            regions.append(
                Region(
                    color_index=color.index,
                    area=int(area),
                    centroid=(cx, cy),
                    contour=approx,
                )
            )
    return regions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_regions.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```powershell
git add app/regions.py tests/test_regions.py
git commit -m "feat: add region extraction via contours"
```

---

## Task 5: Pipeline orchestration

**Files:**
- Create: `app/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `app.config`, `app.models.ConvertParams`/`RenderData`, `app.quantize.quantize`, `app.regions.extract_regions`
- Produces: `render_data_from_image(image_bytes: bytes, params: ConvertParams) -> RenderData`

- [ ] **Step 1: Write the failing test `tests/test_pipeline.py`**

```python
from app.models import ConvertParams, RenderData
from app.pipeline import render_data_from_image
from .conftest import make_two_color_bytes


def test_pipeline_returns_render_data():
    png = make_two_color_bytes()
    params = ConvertParams(palette_size=4, smoothing=0)
    data = render_data_from_image(png, params)

    assert isinstance(data, RenderData)
    assert data.width > 0
    assert data.height > 0
    assert len(data.palette) == 4
    assert len(data.regions) >= 1


def test_pipeline_downsamples_large_side():
    import numpy as np
    from io import BytesIO
    from PIL import Image
    from app import config

    big = np.zeros((3000, 4000, 3), dtype=np.uint8)
    big[:, :2000] = (200, 30, 30)
    big[:, 2000:] = (30, 30, 200)
    buf = BytesIO()
    Image.fromarray(big).save(buf, format="PNG")

    data = render_data_from_image(buf.getvalue(), ConvertParams(palette_size=2, smoothing=0))
    assert max(data.width, data.height) <= config.MAX_WORKING_SIDE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.pipeline'`.

- [ ] **Step 3: Create `app/pipeline.py`**

```python
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from . import config
from .models import ConvertParams, RenderData
from .quantize import quantize
from .regions import extract_regions


def render_data_from_image(image_bytes: bytes, params: ConvertParams) -> RenderData:
    """Run the full conversion pipeline and return data ready for PDF rendering."""
    image = _load_and_normalize(image_bytes, params.smoothing)
    palette, labels = quantize(image, params.palette_size)
    regions = extract_regions(labels, palette, params.min_region_area)
    return RenderData(
        width=image.shape[1],
        height=image.shape[0],
        palette=palette,
        regions=regions,
    )


def _load_and_normalize(image_bytes: bytes, smoothing: int) -> np.ndarray:
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(pil_image)

    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > config.MAX_WORKING_SIDE:
        scale = config.MAX_WORKING_SIDE / longest
        new_size = (int(w * scale), int(h * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    if smoothing > 0:
        sigma = smoothing * 15
        image = cv2.bilateralFilter(image, d=9, sigmaColor=sigma, sigmaSpace=sigma)

    return image
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add app/pipeline.py tests/test_pipeline.py
git commit -m "feat: add conversion pipeline orchestration"
```

---

## Task 6: PDF layout & vector rendering

**Files:**
- Create: `app/pdf_layout.py`, `tests/test_pdf_layout.py`

**Interfaces:**
- Consumes: `app.models.RenderData`, `app.models.ConvertParams`, `app.config`
- Produces:
  - `resolve_orientation(params, width, height) -> bool` (True = portrait)
  - `legend_rows(palette, available_width_pt, item_width_pt) -> list[list[PaletteColor]]`
  - `compute_layout(data, params) -> LayoutGeometry`
  - `render_pdf(data: RenderData, params: ConvertParams) -> bytes`
  - dataclass `LayoutGeometry(page_w, page_h, margin_pt, image_x, image_y, image_w, image_h, scale, legend_h)` where `image_x/image_y` is the bottom-left corner of the rendered image rectangle in PDF points (origin bottom-left).

- [ ] **Step 1: Write the failing test `tests/test_pdf_layout.py`**

```python
from app.models import ConvertParams, PaletteColor, Region, RenderData
from app.pdf_layout import (
    LayoutGeometry,
    compute_layout,
    legend_rows,
    render_pdf,
    resolve_orientation,
)
import numpy as np


def _palette(n):
    return [PaletteColor(index=i + 1, hex="#FF0000", rgb=(255, 0, 0)) for i in range(n)]


def _data(width=200, height=100, palette_size=4):
    contour = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    regions = [Region(color_index=1, area=100, centroid=(5.0, 5.0), contour=contour)]
    return RenderData(width=width, height=height, palette=_palette(palette_size), regions=regions)


def test_resolve_orientation_auto():
    p = ConvertParams(orientation="auto")
    assert resolve_orientation(p, 100, 200) is True   # tall image -> portrait
    assert resolve_orientation(p, 200, 100) is False  # wide image -> landscape


def test_resolve_orientation_explicit():
    assert resolve_orientation(ConvertParams(orientation="portrait"), 200, 100) is True
    assert resolve_orientation(ConvertParams(orientation="landscape"), 100, 200) is False


def test_legend_rows_wraps():
    palette = _palette(10)
    rows = legend_rows(palette, available_width_pt=100, item_width_pt=30)
    # 100 // 30 == 3 per row -> 4 rows (3,3,3,1)
    assert len(rows) == 4
    assert len(rows[0]) == 3
    assert len(rows[-1]) == 1


def test_compute_layout_fits_within_page():
    data = _data(width=200, height=100, palette_size=4)
    geo = compute_layout(data, ConvertParams(paper_size="A4"))
    assert isinstance(geo, LayoutGeometry)
    assert geo.scale > 0
    assert geo.image_w <= geo.page_w
    assert geo.image_h <= geo.page_h
    assert geo.image_x >= 0
    assert geo.image_y >= 0


def test_render_pdf_returns_valid_pdf_bytes():
    data = _data(palette_size=8)
    pdf = render_pdf(data, ConvertParams())
    assert isinstance(pdf, bytes)
    assert pdf[:4] == b"%PDF"


def test_render_pdf_without_numbers():
    data = _data(palette_size=8)
    pdf = render_pdf(data, ConvertParams(show_numbers=False))
    assert pdf[:4] == b"%PDF"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pdf_layout.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.pdf_layout'`.

- [ ] **Step 3: Create `app/pdf_layout.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from reportlab.lib import colors
from reportlab.pdfgen import canvas

from . import config
from .models import ConvertParams, RenderData

_NUMBER_GRAY = colors.HexColor("#888888")


@dataclass
class LayoutGeometry:
    page_w: float
    page_h: float
    margin_pt: float
    image_x: float    # bottom-left x of rendered image
    image_y: float    # bottom-left y of rendered image
    image_w: float
    image_h: float
    scale: float
    legend_h: float


def resolve_orientation(params: ConvertParams, width: int, height: int) -> bool:
    """Return True for portrait. `auto` picks portrait when image is taller than wide."""
    if params.orientation == "portrait":
        return True
    if params.orientation == "landscape":
        return False
    return height >= width


def legend_rows(palette, available_width_pt: float, item_width_pt: float):
    per_row = max(1, int(available_width_pt // item_width_pt))
    return [palette[i:i + per_row] for i in range(0, len(palette), per_row)]


def compute_layout(data: RenderData, params: ConvertParams) -> LayoutGeometry:
    margin_pt = config.PAGE_MARGIN_MM * config.MM_TO_PT
    item_w_pt = config.LEGEND_ITEM_MM * config.MM_TO_PT
    row_h_pt = config.LEGEND_ROW_MM * config.MM_TO_PT
    top_gap_pt = config.LEGEND_TOP_GAP_MM * config.MM_TO_PT

    w_mm, h_mm = config.PAPER_SIZES_MM[params.paper_size]
    portrait = resolve_orientation(params, data.width, data.height)
    page_w = w_mm * config.MM_TO_PT
    page_h = h_mm * config.MM_TO_PT
    if not portrait:
        page_w, page_h = page_h, page_w

    legend_avail_w = page_w - 2 * margin_pt
    rows = legend_rows(data.palette, legend_avail_w, item_w_pt)
    legend_h = len(rows) * row_h_pt + top_gap_pt

    avail_w = page_w - 2 * margin_pt
    avail_h = page_h - 2 * margin_pt - legend_h
    scale = min(avail_w / data.width, avail_h / data.height)
    image_w = data.width * scale
    image_h = data.height * scale
    image_x = margin_pt + (avail_w - image_w) / 2
    image_y = margin_pt + legend_h + (avail_h - image_h) / 2

    return LayoutGeometry(
        page_w=page_w,
        page_h=page_h,
        margin_pt=margin_pt,
        image_x=image_x,
        image_y=image_y,
        image_w=image_w,
        image_h=image_h,
        scale=scale,
        legend_h=legend_h,
    )


def _to_pdf_pt(geo: LayoutGeometry, px: float, py: float):
    """Convert image pixel coords (origin top-left) to PDF points (origin bottom-left)."""
    return (geo.image_x + px * geo.scale, geo.image_y + geo.image_h - py * geo.scale)


def render_pdf(data: RenderData, params: ConvertParams) -> bytes:
    geo = compute_layout(data, params)
    number_color = colors.black if params.number_color == "black" else _NUMBER_GRAY

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(geo.page_w, geo.page_h))

    # Region outlines (white fill + black stroke so the sheet is paintable).
    for region in data.regions:
        pts = [_to_pdf_pt(geo, float(p[0]), float(p[1])) for p in region.contour]
        if len(pts) < 2:
            continue
        path = c.beginPath()
        path.moveTo(*pts[0])
        for x, y in pts[1:]:
            path.lineTo(x, y)
        path.close()
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.black)
        c.setLineWidth(params.line_thickness)
        c.drawPath(path, fill=1, stroke=1)

    # Region numbers at centroids.
    if params.show_numbers:
        c.setFillColor(number_color)
        for region in data.regions:
            cx, cy = _to_pdf_pt(geo, region.centroid[0], region.centroid[1])
            rendered_area_pt = region.area * geo.scale * geo.scale
            side = rendered_area_pt ** 0.5
            font_size = max(5.0, min(16.0, side * 0.4))
            c.setFont("Helvetica", font_size)
            c.drawCentredString(cx, cy - font_size / 2, str(region.color_index))

    _draw_legend(c, geo, data)
    c.showPage()
    c.save()
    return buf.getvalue()


def _draw_legend(c, geo: LayoutGeometry, data: RenderData) -> None:
    margin = geo.margin_pt
    item_w = config.LEGEND_ITEM_MM * config.MM_TO_PT
    square = config.LEGEND_SQUARE_MM * config.MM_TO_PT
    row_h = config.LEGEND_ROW_MM * config.MM_TO_PT
    top_gap = config.LEGEND_TOP_GAP_MM * config.MM_TO_PT
    avail_w = geo.page_w - 2 * margin

    rows = legend_rows(data.palette, avail_w, item_w)
    y_top = margin + geo.legend_h - top_gap

    for r_idx, row in enumerate(rows):
        y = y_top - r_idx * row_h - square
        x = margin
        for color in row:
            c.setFillColor(colors.HexColor(color.hex))
            c.setStrokeColor(colors.black)
            c.setLineWidth(0.5)
            c.rect(x, y, square, square, fill=1, stroke=1)
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 8)
            c.drawString(x + square + 2, y + 2, str(color.index))
            x += item_w
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pdf_layout.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```powershell
git add app/pdf_layout.py tests/test_pdf_layout.py
git commit -m "feat: add PDF layout and vector rendering"
```

---

## Task 7: FastAPI app, endpoints, validation

**Files:**
- Create: `app/main.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: all prior modules
- Produces: FastAPI `app` with `GET /api/health`, `POST /api/convert` (returns `application/pdf`), `GET /` (serves index). Form parsing via `_form_params` dependency that builds `ConvertParams` (Pydantic validation → 422 on bad input). File validation: 415 unsupported type, 413 too large.

- [ ] **Step 1: Write the failing test `tests/test_api.py`**

```python
from fastapi.testclient import TestClient

from app.main import app
from .conftest import make_two_color_bytes


def test_health():
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_convert_returns_pdf():
    client = TestClient(app)
    r = client.post(
        "/api/convert",
        files={"image": ("t.png", make_two_color_bytes(), "image/png")},
        data={"palette_size": "4", "paper_size": "A4"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:4] == b"%PDF"


def test_convert_rejects_palette_over_64():
    client = TestClient(app)
    r = client.post(
        "/api/convert",
        files={"image": ("t.png", make_two_color_bytes(), "image/png")},
        data={"palette_size": "999"},
    )
    assert r.status_code == 422


def test_convert_rejects_bad_content_type():
    client = TestClient(app)
    r = client.post(
        "/api/convert",
        files={"image": ("t.txt", b"hello", "text/plain")},
        data={"palette_size": "4"},
    )
    assert r.status_code == 415
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'`.

- [ ] **Step 3: Create `app/main.py`**

```python
from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config
from .models import ConvertParams
from .pdf_layout import render_pdf
from .pipeline import render_data_from_image

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Paint-by-Numbers Converter")
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/health")
def health():
    return {"status": "ok"}


async def _form_params(
    palette_size: int = Form(config.DEFAULT_PALETTE_SIZE),
    min_region_area: float = Form(0.5),
    smoothing: int = Form(5),
    line_thickness: float = Form(1.0),
    paper_size: str = Form("A4"),
    orientation: str = Form("auto"),
    show_numbers: bool = Form(True),
    number_color: str = Form("black"),
) -> ConvertParams:
    return ConvertParams(
        palette_size=palette_size,
        min_region_area=min_region_area,
        smoothing=smoothing,
        line_thickness=line_thickness,
        paper_size=paper_size,
        orientation=orientation,
        show_numbers=show_numbers,
        number_color=number_color,
    )


@app.post("/api/convert")
async def convert(
    image: UploadFile = File(...),
    params: ConvertParams = Depends(_form_params),
):
    if image.content_type not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type")

    contents = await image.read()
    if len(contents) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    data = render_data_from_image(contents, params)
    pdf = render_pdf(data, params)
    return Response(content=pdf, media_type="application/pdf")


@app.get("/", response_class=HTMLResponse)
def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h1>Frontend not built yet</h1>"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```powershell
git add app/main.py tests/test_api.py
git commit -m "feat: add FastAPI endpoints and file validation"
```

---

## Task 8: Frontend UI

**Files:**
- Create: `static/index.html`, `static/app.js`, `static/styles.css`
- Modify: `tests/test_api.py` (append two smoke tests)

**Interfaces:**
- Consumes: `POST /api/convert` (multipart form fields match `ConvertParams`)
- Produces: a usable single-page UI; `GET /` returns the HTML.

- [ ] **Step 1: Append frontend smoke tests to `tests/test_api.py`**

Add to the end of `tests/test_api.py`:

```python
def test_index_page_served():
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "Картины по номерам" in r.text


def test_static_appjs_served():
    client = TestClient(app)
    r = client.get("/static/app.js")
    assert r.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_api.py::test_index_page_served tests/test_api.py::test_static_appjs_served -v`
Expected: FAIL (index placeholder text / 404 for app.js).

- [ ] **Step 3: Create `static/index.html`**

```html
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Картины по номерам</title>
  <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
  <main class="container">
    <h1>Картины по номерам</h1>
    <form id="convert-form">
      <label class="file">
        <input type="file" id="image" name="image" accept="image/png,image/jpeg,image/webp" required>
        <span id="file-label">Выберите изображение</span>
      </label>
      <div class="preview"><img id="preview" alt="" hidden></div>

      <fieldset class="params">
        <legend>Параметры</legend>
        <label>Цветов: <output id="palette_out">16</output>
          <input type="range" id="palette_size" name="palette_size" min="2" max="64" value="16"></label>
        <label>Мин. область (%): <input type="number" name="min_region_area" min="0.1" max="10" step="0.1" value="0.5"></label>
        <label>Сглаживание: <input type="range" name="smoothing" min="0" max="10" value="5"></label>
        <label>Толщина линий: <input type="number" name="line_thickness" min="0.5" max="3" step="0.1" value="1.0"></label>
        <label>Формат:
          <select name="paper_size">
            <option>A3</option>
            <option selected>A4</option>
            <option>A5</option>
            <option>Letter</option>
            <option>Legal</option>
          </select>
        </label>
        <label>Ориентация:
          <select name="orientation">
            <option value="auto" selected>auto</option>
            <option value="portrait">portrait</option>
            <option value="landscape">landscape</option>
          </select>
        </label>
        <label>Номера: <input type="checkbox" name="show_numbers" checked></label>
        <label>Цвет номеров:
          <select name="number_color">
            <option value="black" selected>black</option>
            <option value="gray">gray</option>
          </select>
        </label>
      </fieldset>

      <button type="submit" id="submit">Сгенерировать PDF</button>
      <p id="status" class="status"></p>
    </form>

    <iframe id="result" class="result" hidden></iframe>
  </main>
  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 4: Create `static/app.js`**

```javascript
const form = document.getElementById("convert-form");
const statusEl = document.getElementById("status");
const result = document.getElementById("result");
const palette = document.getElementById("palette_size");
const paletteOut = document.getElementById("palette_out");
const imageInput = document.getElementById("image");
const fileLabel = document.getElementById("file-label");
const preview = document.getElementById("preview");
const submitBtn = document.getElementById("submit");

palette.addEventListener("input", () => {
  paletteOut.textContent = palette.value;
});

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (file) {
    fileLabel.textContent = file.name;
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!imageInput.files[0]) {
    statusEl.textContent = "Сначала выберите изображение.";
    return;
  }

  const formData = new FormData(form);
  const numbersChecked = document.querySelector('[name="show_numbers"]').checked;
  formData.set("show_numbers", numbersChecked ? "true" : "false");

  submitBtn.disabled = true;
  statusEl.textContent = "Обработка…";
  result.hidden = true;

  try {
    const response = await fetch("/api/convert", { method: "POST", body: formData });
    if (!response.ok) {
      const text = await response.text();
      throw new Error("Ошибка " + response.status + ": " + text);
    }
    const blob = await response.blob();
    result.src = URL.createObjectURL(blob);
    result.hidden = false;
    statusEl.textContent = "Готово. Можно скачать или распечатать из просмотра.";
  } catch (err) {
    statusEl.textContent = err.message;
  } finally {
    submitBtn.disabled = false;
  }
});
```

- [ ] **Step 5: Create `static/styles.css`**

```css
:root {
  --bg: #f5f5f7;
  --panel: #ffffff;
  --accent: #2f6df6;
  --border: #d8d8e0;
  --text: #1d1d1f;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
}

.container {
  max-width: 720px;
  margin: 0 auto;
  padding: 24px 16px 64px;
}

h1 { margin: 0 0 16px; }

form { display: flex; flex-direction: column; gap: 16px; }

.file {
  display: flex; align-items: center; gap: 12px;
  padding: 14px; background: var(--panel);
  border: 1px dashed var(--border); border-radius: 10px; cursor: pointer;
}

.preview img {
  max-width: 100%; max-height: 260px; border-radius: 8px;
  border: 1px solid var(--border);
}

.params {
  border: 1px solid var(--border); border-radius: 10px;
  background: var(--panel); padding: 16px;
  display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
}
.params legend { padding: 0 6px; font-weight: 600; }
.params label { display: flex; flex-direction: column; gap: 4px; font-size: 14px; }

input, select { padding: 6px 8px; border: 1px solid var(--border); border-radius: 6px; }

button {
  align-self: flex-start; padding: 12px 20px; font-size: 16px;
  background: var(--accent); color: white; border: none; border-radius: 8px;
  cursor: pointer;
}
button:disabled { opacity: 0.6; cursor: progress; }

.status { min-height: 1.2em; color: #444; }

.result {
  width: 100%; height: 80vh; margin-top: 16px;
  border: 1px solid var(--border); border-radius: 8px;
}
```

- [ ] **Step 6: Run the full test suite**

Run: `python -m pytest -v`
Expected: all tests pass (including the two new frontend smoke tests).

- [ ] **Step 7: Commit**

```powershell
git add static/ tests/test_api.py
git commit -m "feat: add frontend UI for single-image conversion"
```

---

## Task 9: README + manual verification

**Files:**
- Create: `README.md`, `samples/.gitkeep`

- [ ] **Step 1: Create `samples/.gitkeep`** (empty file so the directory is tracked).

- [ ] **Step 2: Create `README.md`**

```markdown
# Paint-by-Numbers Converter

Локальное веб-приложение: превращает фотографию или картинку в PDF-лист
«картины по номерам» для печати и раскрашивания.

## Возможности

- Квантование цветов в палитру из 2–64 цветов (k-means)
- Автоматическая сегментация на области с номерами
- Векторный PDF (контуры + номера + легенда палитры) форматов A3/A4/A5/Letter/Legal
- Параметры: размер палитры, мин. область, сглаживание, толщина линий, ориентация

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Запуск

```powershell
uvicorn app.main:app --reload
```

Откройте `http://localhost:8000`, загрузите изображение, настройте параметры
и нажмите «Сгенерировать PDF».

## Тесты

```powershell
python -m pytest -v
```

## Структура

- `app/` — FastAPI-сервер и конвейер обработки (квантование, сегментация, PDF)
- `static/` — фронтенд (HTML/JS/CSS, без сборщика)
- `tests/` — unit- и API-тесты
- `samples/` — примеры изображений для ручной проверки
- `docs/superpowers/specs/` — спецификация дизайна
- `docs/superpowers/plans/` — план реализации
```

- [ ] **Step 3: Manual end-to-end verification**

Run: `uvicorn app.main:app --reload`
Then in a browser at `http://localhost:8000`:
1. Upload a photo from `samples/` (drop one in first).
2. Set palette size to 24, format A4.
3. Click «Сгенерировать PDF».
4. Confirm the PDF shows numbered regions and a palette legend.
5. Try `palette_size=64` and confirm it works; try a non-image file and confirm a clean error.

Expected: PDF renders correctly; no server errors.

- [ ] **Step 4: Commit**

```powershell
git add README.md samples/.gitkeep
git commit -m "docs: add README and samples directory"
```

---

## Self-Review Checklist (run after all tasks)

- Run `python -m pytest -v` — full suite green.
- Run `python -m pytest --cov=app` — review coverage; core modules (quantize, regions, pipeline, pdf_layout, models, main) should be exercised.
- Spot-check: `palette_size=64` accepted, `65` rejected (422), bad file type rejected (415).
- Confirm `.venv/`, `__pycache__/`, `*.pdf` are gitignored (no stray commits).
