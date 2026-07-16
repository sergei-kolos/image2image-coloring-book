from __future__ import annotations

import base64 as base64mod
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pydantic import ValidationError

from . import config
from .models import ConvertParams
from .pipeline import render_data_from_image
from .visualize import render_visualization, render_outline
from .pdf_layout import render_pdf

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Paint-by-Numbers Converter")
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(ValidationError)
async def pydantic_validation_handler(request, exc: ValidationError):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=422, content={"detail": exc.errors(include_url=False)})


@app.get("/api/health")
def health():
    return {"status": "ok"}


async def _form_params(
    detail_level: int = Form(5),
    max_colors: int = Form(64),
    palette_size: int = Form(16),
    min_region_area: float = Form(0.5),
    smoothing: int = Form(5),
    line_thickness: float = Form(1.0),
    color_merge_threshold: float = Form(5.0),
    paper_size: str = Form("A4"),
    orientation: str = Form("auto"),
    show_numbers: bool = Form(True),
    number_color: str = Form("black"),
) -> ConvertParams:
    return ConvertParams(
        detail_level=detail_level,
        max_colors=max_colors,
        palette_size=palette_size,
        min_region_area=min_region_area,
        smoothing=smoothing,
        line_thickness=line_thickness,
        color_merge_threshold=color_merge_threshold,
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
    colored_png = render_visualization(data, params)
    outline_png = render_outline(data, params)
    pdf_bytes = render_pdf(data, params)
    return JSONResponse({
        "colored": "data:image/png;base64," + base64mod.b64encode(colored_png).decode(),
        "outline": "data:image/png;base64," + base64mod.b64encode(outline_png).decode(),
        "pdf": "data:application/pdf;base64," + base64mod.b64encode(pdf_bytes).decode(),
    })


@app.get("/", response_class=HTMLResponse)
def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h1>Frontend not built yet</h1>"
