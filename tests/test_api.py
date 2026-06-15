"""Lightweight API tests that do NOT load the heavy OCR models.

These tests stub the engine so they run on CI / locally in <1s. End-to-end
recognition quality is validated manually with sample images (see README).
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture()
def client(monkeypatch):
    # Stub OCREngine before importing the app so lifespan warmup is a no-op.
    from app import ocr_engine

    class _StubEngine:
        ready = True

        def warmup(self) -> None:
            return None

        def recognize(self, image_bgr) -> list[str]:
            assert isinstance(image_bgr, np.ndarray)
            return ["Xin chào", "Việt Nam"]

    stub = _StubEngine()
    monkeypatch.setattr(ocr_engine.OCREngine, "instance", classmethod(lambda cls: stub))

    from app.main import app

    with TestClient(app) as c:
        yield c


def _png_bytes() -> bytes:
    img = Image.new("RGB", (32, 16), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine_ready"] is True


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()


def test_ocr_multipart(client):
    r = client.post(
        "/ocr",
        files={"file": ("sample.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["lines"] == ["Xin chào", "Việt Nam"]
    assert body["text"] == "Xin chào\nViệt Nam"
    assert body["elapsed_ms"] >= 0


def test_ocr_base64(client):
    import base64

    payload = {"image_base64": base64.b64encode(_png_bytes()).decode()}
    r = client.post("/ocr/base64", json=payload)
    assert r.status_code == 200
    assert r.json()["lines"] == ["Xin chào", "Việt Nam"]


def test_ocr_rejects_non_image(client):
    r = client.post(
        "/ocr",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
    )
    assert r.status_code == 415


def test_ocr_rejects_empty(client):
    r = client.post(
        "/ocr",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert r.status_code == 400


def test_ocr_base64_invalid(client):
    r = client.post("/ocr/base64", json={"image_base64": "@@@not-base64@@@"})
    assert r.status_code == 400
