"""Image decoding & preprocessing utilities."""
from __future__ import annotations

import base64
from io import BytesIO
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import settings


def decode_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR ndarray, honoring EXIF orientation."""
    if not data:
        raise ValueError("Empty image payload")
    try:
        with Image.open(BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            arr = np.array(im)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Cannot decode image: {exc}") from exc
    # PIL gives RGB, OpenCV/Paddle expect BGR
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def decode_base64_image(b64: str) -> np.ndarray:
    """Decode a base64 string (with optional data URL prefix) into a BGR ndarray."""
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64, validate=False)
    except Exception as exc:
        raise ValueError(f"Invalid base64 payload: {exc}") from exc
    return decode_image(raw)


def resize_max_side(img: np.ndarray, max_side: int | None = None) -> Tuple[np.ndarray, float]:
    """Resize so that the longer side equals `max_side`. Returns (image, scale)."""
    target = max_side or settings.max_image_side
    h, w = img.shape[:2]
    longer = max(h, w)
    if longer <= target:
        return img, 1.0
    scale = target / float(longer)
    new_size = (int(round(w * scale)), int(round(h * scale)))
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


def warp_crop(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Perspective-warp a quadrilateral region into an upright rectangle.

    `box` is a (4, 2) array of points ordered as [tl, tr, br, bl] from PaddleOCR.
    """
    box = np.asarray(box, dtype=np.float32)
    if box.shape != (4, 2):
        raise ValueError(f"Expected (4,2) box, got {box.shape}")
    (tl, tr, br, bl) = box
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    width = max(width, 1)
    height = max(height, 1)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(img, matrix, (width, height), flags=cv2.INTER_CUBIC)
    # If the crop is taller than wide by a large margin it's likely rotated 90°.
    if height > width * 1.5:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped
