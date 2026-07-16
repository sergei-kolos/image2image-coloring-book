from __future__ import annotations

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
