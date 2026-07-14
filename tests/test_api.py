from __future__ import annotations

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
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic


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


def test_index_page_served():
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "Картины по номерам" in r.text


def test_static_appjs_served():
    client = TestClient(app)
    r = client.get("/static/app.js")
    assert r.status_code == 200
