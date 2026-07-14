from app import config


def test_palette_cap_is_64():
    assert config.MAX_PALETTE_SIZE == 64


def test_default_palette_is_16():
    assert config.DEFAULT_PALETTE_SIZE == 16


def test_paper_sizes_present():
    for name in ("A3", "A4", "A5", "Letter", "Legal"):
        assert name in config.PAPER_SIZES_MM
        assert len(config.PAPER_SIZES_MM[name]) == 2
