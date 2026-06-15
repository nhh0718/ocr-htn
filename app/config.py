"""Runtime configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # VietOCR weights: "vgg_transformer" (accurate, heavier) or "vgg_seq2seq"
    vietocr_weights: str = os.getenv("VIETOCR_WEIGHTS", "vgg_transformer")
    # CPU only on Render free/standard plans
    device: str = os.getenv("OCR_DEVICE", "cpu")
    # PaddleOCR detection language (multilingual model handles Vietnamese well)
    paddle_lang: str = os.getenv("PADDLE_LANG", "en")
    # Maximum side of the input image (longer side resized to this; preserves aspect)
    max_image_side: int = int(os.getenv("MAX_IMAGE_SIDE", "1600"))
    # Y-axis tolerance ratio for grouping detected boxes into lines
    line_y_tolerance: float = float(os.getenv("LINE_Y_TOLERANCE", "0.7"))
    # Where to cache models inside the container
    model_dir: str = os.getenv("MODEL_DIR", "/app/models")
    # Whether to log per-request timings
    verbose: bool = _get_bool("OCR_VERBOSE", False)


settings = Settings()
