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
    from app import ocr_engine

    class _StubResult:
        text = "Xin chào\nViệt Nam"
        lines = ["Xin chào", "Việt Nam"]
        blocks = [
            ocr_engine.TextBlock(text="Xin", bbox=[10, 20, 50, 40], confidence=1.0, line_index=0),
            ocr_engine.TextBlock(text="chào", bbox=[55, 20, 90, 40], confidence=1.0, line_index=0),
            ocr_engine.TextBlock(text="Việt", bbox=[10, 50, 50, 70], confidence=1.0, line_index=1),
            ocr_engine.TextBlock(text="Nam", bbox=[55, 50, 90, 70], confidence=1.0, line_index=1),
        ]
        image_size = [100, 80]
        elapsed_ms = 0

    class _StubEngine:
        ready = True

        def warmup(self) -> None:
            return None

        def recognize(self, image_bgr) -> object:
            assert isinstance(image_bgr, np.ndarray)
            return _StubResult()

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
    assert "extraction_provider" in body


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
    assert len(body["blocks"]) == 4
    assert body["blocks"][0]["text"] == "Xin"
    assert body["blocks"][0]["line_index"] == 0
    assert body["image_size"] == [100, 80]
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


# --------------------------------------------------------------------- Extract
def test_extract_with_text_only(client):
    r = client.post("/extract", json={"text": "Hello world"})
    assert r.status_code == 200
    body = r.json()
    assert body["method"] == "none"
    assert body["raw_text"] == "Hello world"


def test_extract_no_input(client):
    r = client.post("/extract", json={})
    assert r.status_code == 400


def test_extract_auto_detect_acb(client):
    text = (
        "ACB\n"
        "Chuyển tiền thành công\n"
        "30.000VND\n"
        "Ba mươi nghìn đồng\n"
        "Từ NGUYEN HUY HOANG\n"
        "44444%87\n"
        "Đến NGO THI KHUYEN\n"
        "BIDV - NH TMCP Dau tu va phat\n"
        "Mã giao dịch 5696\n"
        "Chuyển lúc 15/06/2026, 19:06:10\n"
        "Phí Miễn phí\n"
        "Nội dung\n"
        "NGUYEN HUY HOANG CHUYEN KHOAN"
    )
    r = client.post("/extract", json={"text": text})
    assert r.status_code == 200
    body = r.json()
    assert body["method"] == "pattern"
    assert body["pattern_id"] == "acb-transfer"
    assert body["fields"]["amount"] == 30000
    assert body["fields"]["transaction_id"] == "5696"
    assert "datetime" in body["fields"]


def test_extract_auto_detect_momo(client):
    text = (
        "Chi Tiết Giao Dịch\n"
        "MUA VÉ XEM PHIM\n"
        "-110.000\n"
        "Trạng thái Thành công\n"
        "Thời gian 0920:09-29/05/2026\n"
        "Mã giao dịch 131189602811\n"
        "Tài khoản/thẻ Ví MoMo\n"
        "Tổng phí Miễn phí\n"
        "Danh mục Giải trí"
    )
    r = client.post("/extract", json={"text": text})
    assert r.status_code == 200
    body = r.json()
    assert body["method"] == "pattern"
    assert body["pattern_id"] == "momo-receipt"
    assert body["fields"]["transaction_id"] == "131189602811"


def test_extract_pattern_not_found(client):
    r = client.post("/extract", json={"text": "test", "pattern_id": "nonexistent"})
    assert r.status_code == 404


def test_extract_llm_not_configured(client):
    r = client.post("/extract", json={
        "text": "random text with no pattern match",
        "schema": {"type": "object", "properties": {"amount": {"type": "number"}}},
    })
    assert r.status_code == 503


# --------------------------------------------------------------------- Patterns
def test_list_patterns(client):
    r = client.get("/patterns")
    assert r.status_code == 200
    body = r.json()
    ids = [p["id"] for p in body["patterns"]]
    assert "acb-transfer" in ids
    assert "momo-receipt" in ids
    assert "id-card-vn" in ids
