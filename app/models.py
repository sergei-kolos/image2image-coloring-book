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
