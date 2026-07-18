from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


class ConvertParams(BaseModel):
    detail_level: int = Field(default=5, ge=1, le=15)
    max_colors: int = Field(default=64, ge=2, le=64)
    palette_size: int = Field(default=16, ge=2, le=64)
    min_region_area: float = Field(default=0.5, ge=0.1, le=10.0)
    smoothing: int = Field(default=5, ge=0, le=10)
    line_thickness: float = Field(default=1.0, ge=0.5, le=3.0)
    color_merge_threshold: float = Field(default=5.0, ge=0.0, le=30.0)
    paper_size: Literal["A3", "A4", "A5", "Letter", "Legal"] = "A4"
    orientation: Literal["auto", "portrait", "landscape"] = "auto"
    show_numbers: bool = True
    number_color: Literal["black", "gray"] = "black"
    engine: Literal["classic", "sam_hq"] = "classic"

    def pipeline_params(self) -> PipelineParams:
        t = (self.detail_level - 1) / 14.0
        derived_palette = max(2, min(self.max_colors, round(12 + t * 36)))
        return PipelineParams(
            max_working_side=int(1200 + t * 2000),
            mean_shift_sp=max(1, round(12 - t * 11)),
            mean_shift_sr=max(10, round(35 - t * 20)),
            felzenszwalb_scale=max(10, round(60 - t * 50)),
            palette_size=derived_palette,
            min_region_area_pct=max(0.05, 1.0 * (10 ** (-1.3 * t))),
            morph_kernel=max(0, round(5 - t * 5)),
            boundary_sigma=max(0.3, round((1.5 - t * 1.2) * 10) / 10),
        )


@dataclass
class PipelineParams:
    max_working_side: int
    mean_shift_sp: int
    mean_shift_sr: int
    felzenszwalb_scale: int
    palette_size: int
    min_region_area_pct: float
    morph_kernel: int
    boundary_sigma: float


@dataclass
class PaletteColor:
    index: int        # 1-based number shown to the user
    hex: str          # "#RRGGBB"
    rgb: tuple        # (r, g, b) 0..255


@dataclass
class SharedEdge:
    label_a: int               # 0-based cluster id (or -1 for image border)
    label_b: int               # 0-based cluster id (or -1 for image border)
    polyline: object           # np.ndarray (N, 2) float — shared boundary path


@dataclass
class Region:
    color_index: int          # 1-based, matches PaletteColor.index
    area: int                 # pixel area
    centroid: tuple           # (x, y) in image pixel coords
    label: str                # display label (digit or letter for tiny zones)
    contour: object           # np.ndarray of shape (N, 2) — exterior ring
    holes: list = field(default_factory=list)  # list[np.ndarray] interior cut‑outs


@dataclass
class RenderData:
    width: int
    height: int
    palette: list             # list[PaletteColor]
    regions: list             # list[Region]
    edges: list = field(default_factory=list)  # list[SharedEdge]
