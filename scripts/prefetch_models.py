"""Download PaddleOCR and VietOCR weights at image build time.

Failing here should NOT abort the Docker build (the Dockerfile guards this with
`|| echo`); models will then download lazily on first request.
"""
from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("prefetch")


def prefetch_paddle() -> None:
    from paddleocr import PaddleOCR

    log.info("Prefetching PaddleOCR detection model...")
    PaddleOCR(lang="en", use_angle_cls=False, show_log=False, rec=False)
    log.info("PaddleOCR ready.")


def prefetch_vietocr() -> None:
    import os

    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor

    weights = os.getenv("VIETOCR_WEIGHTS", "vgg_transformer")
    log.info("Prefetching VietOCR weights: %s", weights)
    cfg = Cfg.load_config_from_name(weights)
    cfg["device"] = "cpu"
    cfg["predictor"]["beamsearch"] = False
    Predictor(cfg)
    log.info("VietOCR ready.")


def main() -> int:
    try:
        prefetch_paddle()
        prefetch_vietocr()
        return 0
    except Exception as exc:  # noqa: BLE001
        log.warning("Prefetch failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
