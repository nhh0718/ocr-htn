"""OCR pipeline: PaddleOCR detection + VietOCR recognition.

The engine is a process-wide singleton. Models are loaded lazily on first use
and reused across requests to avoid the ~10-15s cold start cost.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from PIL import Image

from .config import settings
from .preprocess import resize_max_side, warp_crop

logger = logging.getLogger(__name__)


@dataclass
class _Box:
    points: np.ndarray  # (4, 2) float32
    y_center: float
    x_center: float
    height: float


class OCREngine:
    """Singleton wrapping PaddleOCR (det) + VietOCR (rec)."""

    _instance: "OCREngine | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._detector = None
        self._recognizer = None
        self._ready = False
        self._init_lock = threading.Lock()

    # ------------------------------------------------------------------ public
    @classmethod
    def instance(cls) -> "OCREngine":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def ready(self) -> bool:
        return self._ready

    def warmup(self) -> None:
        """Force model load. Safe to call multiple times."""
        self._ensure_loaded()

    def recognize(self, image_bgr: np.ndarray) -> List[str]:
        """Run the full pipeline on a BGR ndarray and return ordered text lines."""
        self._ensure_loaded()
        if image_bgr is None or image_bgr.size == 0:
            return []
        image_bgr, _ = resize_max_side(image_bgr)
        boxes = self._detect(image_bgr)
        if not boxes:
            return []
        crops_rgb = [self._crop_to_rgb(image_bgr, b.points) for b in boxes]
        texts = self._recognize_batch(crops_rgb)
        # Pair texts with boxes, drop empties, group into lines preserving order.
        items = [(b, t) for b, t in zip(boxes, texts) if t and t.strip()]
        return self._group_lines(items)

    # ----------------------------------------------------------------- private
    def _ensure_loaded(self) -> None:
        if self._ready:
            return
        with self._init_lock:
            if self._ready:
                return
            self._load_detector()
            self._load_recognizer()
            self._ready = True

    def _load_detector(self) -> None:
        from paddleocr import PaddleOCR

        logger.info("Loading PaddleOCR detector (lang=%s)...", settings.paddle_lang)
        # We only use detection; rec=False skips loading the (lower quality for VN)
        # PaddleOCR recognizer entirely.
        self._detector = PaddleOCR(
            lang=settings.paddle_lang,
            use_angle_cls=False,
            use_gpu=settings.device == "cuda",
            show_log=False,
            rec=False,
        )
        logger.info("PaddleOCR detector ready.")

    def _load_recognizer(self) -> None:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        logger.info("Loading VietOCR recognizer (weights=%s)...", settings.vietocr_weights)
        cfg = Cfg.load_config_from_name(settings.vietocr_weights)
        cfg["device"] = settings.device
        cfg["predictor"]["beamsearch"] = False
        # vietocr will download pretrained weights to its default cache on first use
        self._recognizer = Predictor(cfg)
        logger.info("VietOCR recognizer ready.")

    def _detect(self, image_bgr: np.ndarray) -> List[_Box]:
        # PaddleOCR.ocr with rec=False returns: [[box1, box2, ...]]  (one entry per image)
        result = self._detector.ocr(image_bgr, det=True, rec=False, cls=False)
        if not result:
            return []
        raw_boxes = result[0] or []
        boxes: List[_Box] = []
        for raw in raw_boxes:
            pts = np.asarray(raw, dtype=np.float32)
            if pts.shape != (4, 2):
                continue
            ys = pts[:, 1]
            xs = pts[:, 0]
            height = float(ys.max() - ys.min())
            if height < 2:
                continue
            boxes.append(
                _Box(
                    points=pts,
                    y_center=float(ys.mean()),
                    x_center=float(xs.mean()),
                    height=height,
                )
            )
        return boxes

    @staticmethod
    def _crop_to_rgb(image_bgr: np.ndarray, box: np.ndarray) -> Image.Image:
        crop_bgr = warp_crop(image_bgr, box)
        crop_rgb = crop_bgr[:, :, ::-1]
        return Image.fromarray(crop_rgb)

    def _recognize_batch(self, crops: Sequence[Image.Image]) -> List[str]:
        if not crops:
            return []
        # VietOCR Predictor.predict_batch is faster but optional depending on version.
        predict_batch = getattr(self._recognizer, "predict_batch", None)
        try:
            if callable(predict_batch):
                return list(predict_batch(list(crops)))
            return [self._recognizer.predict(c) for c in crops]
        except Exception as exc:
            logger.warning("Batch predict failed (%s); falling back to per-image.", exc)
            return [self._safe_predict(c) for c in crops]

    def _safe_predict(self, crop: Image.Image) -> str:
        try:
            return self._recognizer.predict(crop) or ""
        except Exception as exc:
            logger.warning("VietOCR predict failed on a crop: %s", exc)
            return ""

    @staticmethod
    def _group_lines(items: Sequence[tuple[_Box, str]]) -> List[str]:
        if not items:
            return []
        # Sort top-to-bottom first.
        sorted_items = sorted(items, key=lambda it: (it[0].y_center, it[0].x_center))
        avg_height = float(np.mean([it[0].height for it in sorted_items])) or 1.0
        tolerance = avg_height * settings.line_y_tolerance

        lines: List[List[tuple[_Box, str]]] = []
        for box, text in sorted_items:
            if not lines:
                lines.append([(box, text)])
                continue
            current = lines[-1]
            current_y = float(np.mean([b.y_center for b, _ in current]))
            if abs(box.y_center - current_y) <= tolerance:
                current.append((box, text))
            else:
                lines.append([(box, text)])

        out: List[str] = []
        for line in lines:
            line.sort(key=lambda it: it[0].x_center)
            joined = " ".join(t.strip() for _, t in line if t.strip())
            if joined:
                out.append(joined)
        return out


def get_engine() -> OCREngine:
    return OCREngine.instance()
