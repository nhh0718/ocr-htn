"""Unit tests for preprocess utilities (no heavy deps required)."""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.preprocess import decode_base64_image, decode_image, resize_max_side, warp_crop


def _png_bytes(size=(64, 32), color=(120, 200, 50)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_decode_image_returns_bgr_ndarray():
    arr = decode_image(_png_bytes())
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 3 and arr.shape[2] == 3
    # PIL was RGB(120,200,50) -> BGR(50,200,120)
    assert tuple(arr[0, 0]) == (50, 200, 120)


def test_decode_image_empty():
    with pytest.raises(ValueError):
        decode_image(b"")


def test_decode_base64_with_data_url():
    import base64

    raw = _png_bytes()
    b64 = "data:image/png;base64," + base64.b64encode(raw).decode()
    arr = decode_base64_image(b64)
    assert arr.shape[:2] == (32, 64)


def test_resize_max_side_no_op_for_small():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out, scale = resize_max_side(img, max_side=400)
    assert scale == 1.0
    assert out.shape == img.shape


def test_resize_max_side_downscales():
    img = np.zeros((1000, 2000, 3), dtype=np.uint8)
    out, scale = resize_max_side(img, max_side=1000)
    assert scale == pytest.approx(0.5)
    assert out.shape[1] == 1000


def test_warp_crop_axis_aligned():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[10:30, 20:80] = (0, 0, 255)  # red rectangle in BGR
    box = np.array([[20, 10], [80, 10], [80, 30], [20, 30]], dtype=np.float32)
    crop = warp_crop(img, box)
    assert crop.shape[0] in {19, 20}  # height ~= 20
    assert crop.shape[1] in {59, 60}  # width ~= 60
    # Center pixel should be red.
    cy, cx = crop.shape[0] // 2, crop.shape[1] // 2
    assert tuple(crop[cy, cx]) == (0, 0, 255)
