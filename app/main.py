"""FastAPI entrypoint for the Vietnamese OCR API."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .config import settings
from .ocr_engine import get_engine
from .preprocess import decode_base64_image, decode_image
from .schemas import HealthResponse, OCRBase64Request, OCRResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ocr.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up models at startup so the first real request is fast.
    try:
        get_engine().warmup()
        logger.info("OCR engine warmup complete.")
    except Exception as exc:  # pragma: no cover - log and continue, healthz reflects state
        logger.exception("OCR engine warmup failed: %s", exc)
    yield


app = FastAPI(
    title="Vietnamese OCR API",
    version=__version__,
    description="Self-hosted OCR API (PaddleOCR detection + VietOCR recognition).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "name": "Vietnamese OCR API",
        "version": __version__,
        "docs": "/docs",
        "endpoints": ["/ocr", "/ocr/base64", "/healthz"],
    }


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    engine = get_engine()
    return HealthResponse(version=__version__, engine_ready=engine.ready)


def _run_ocr(image_bgr) -> OCRResponse:
    start = time.perf_counter()
    try:
        lines = get_engine().recognize(image_bgr)
    except Exception as exc:
        logger.exception("OCR pipeline error")
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    if settings.verbose:
        logger.info("OCR done: %d lines in %d ms", len(lines), elapsed_ms)
    return OCRResponse(text="\n".join(lines), lines=lines, elapsed_ms=elapsed_ms)


@app.post("/ocr", response_model=OCRResponse)
async def ocr_multipart(file: UploadFile = File(...)) -> OCRResponse:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        image = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _run_ocr(image)


@app.post("/ocr/base64", response_model=OCRResponse)
async def ocr_base64(payload: OCRBase64Request) -> OCRResponse:
    try:
        image = decode_base64_image(payload.image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _run_ocr(image)
