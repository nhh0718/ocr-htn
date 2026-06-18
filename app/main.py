"""FastAPI entrypoint for the Vietnamese OCR & extraction API."""
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
from .schemas import (
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    OCRBase64Request,
    OCRResponse,
    PatternInfo,
    PatternListResponse,
    TextBlockOut,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ocr.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_engine().warmup()
        logger.info("OCR engine warmup complete.")
    except Exception as exc:
        logger.exception("OCR engine warmup failed: %s", exc)
    yield


app = FastAPI(
    title="Vietnamese OCR API",
    version=__version__,
    description="Self-hosted OCR (PaddleOCR + VietOCR) with pattern-based and LLM extraction.",
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
        "endpoints": ["/ocr", "/ocr/base64", "/extract", "/extract/preset/{name}", "/patterns", "/healthz"],
    }


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    engine = get_engine()
    return HealthResponse(
        version=__version__,
        engine_ready=engine.ready,
        extraction_provider=settings.extraction_provider,
    )


# --------------------------------------------------------------------------- OCR
def _run_ocr(image_bgr) -> OCRResponse:
    start = time.perf_counter()
    try:
        result = get_engine().recognize(image_bgr)
    except Exception as exc:
        logger.exception("OCR pipeline error")
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    if settings.verbose:
        logger.info("OCR done: %d lines, %d blocks in %d ms", len(result.lines), len(result.blocks), elapsed_ms)
    blocks_out = [
        TextBlockOut(text=b.text, bbox=b.bbox, confidence=b.confidence, line_index=b.line_index)
        for b in result.blocks
    ]
    return OCRResponse(
        text=result.text,
        lines=result.lines,
        blocks=blocks_out,
        image_size=result.image_size,
        elapsed_ms=elapsed_ms,
    )


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


# --------------------------------------------------------------------- Extraction
def _do_extract(req: ExtractRequest) -> ExtractResponse:
    start = time.perf_counter()

    # 1. Get text: either from request or OCR an image
    raw_text = req.text or ""
    if req.image_base64:
        try:
            image = decode_base64_image(req.image_base64)
            result = get_engine().recognize(image)
            raw_text = result.text
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc
    elif not raw_text:
        raise HTTPException(status_code=400, detail="Provide either image_base64 or text.")

    # 2. Try pattern-based extraction first
    from . import extractor as ext_mod
    from . import llm_extractor as llm_mod

    pattern = None
    if req.pattern_id:
        pattern = ext_mod.load_pattern(req.pattern_id)
        if not pattern:
            raise HTTPException(status_code=404, detail=f"Pattern '{req.pattern_id}' not found.")
    elif req.pattern:
        pattern = ext_mod.LoadedPattern(
            id="inline",
            name=req.pattern.get("name", "inline"),
            description=req.pattern.get("description", ""),
            detect_keywords=req.pattern.get("detect", {}).get("any_keyword", []),
            fields=req.pattern.get("fields", {}),
        )
    else:
        # Auto-detect
        pattern = ext_mod.auto_detect_pattern(raw_text)

    if pattern:
        fields = ext_mod.extract_with_pattern(raw_text, pattern)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return ExtractResponse(
            method="pattern",
            pattern_id=pattern.id,
            fields=fields,
            raw_text=raw_text,
            elapsed_ms=elapsed_ms,
        )

    # 3. Fall back to LLM extraction if schema is provided
    if req.json_schema:
        if not llm_mod.is_available():
            raise HTTPException(
                status_code=503,
                detail="No pattern matched and LLM extraction is not configured. "
                       "Set EXTRACTION_PROVIDER and EXTRACTION_API_KEY, or provide a pattern_id.",
            )
        try:
            fields = llm_mod.extract_via_llm(raw_text, req.json_schema, req.instructions)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"LLM extraction failed: {exc}") from exc
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return ExtractResponse(
            method="llm",
            pattern_id=None,
            fields=fields,
            raw_text=raw_text,
            elapsed_ms=elapsed_ms,
        )

    # 4. No pattern, no schema → return raw text only
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return ExtractResponse(
        method="none",
        pattern_id=None,
        fields={},
        raw_text=raw_text,
        elapsed_ms=elapsed_ms,
    )


@app.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest) -> ExtractResponse:
    return _do_extract(req)


@app.post("/extract/preset/{pattern_id}", response_model=ExtractResponse)
async def extract_preset(pattern_id: str, file: UploadFile = File(...)) -> ExtractResponse:
    """Shortcut: upload image + use a named pattern. Equivalent to POST /extract with pattern_id."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        image = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result = get_engine().recognize(image)
    req = ExtractRequest(text=result.text, pattern_id=pattern_id)
    return _do_extract(req)


# ------------------------------------------------------------------------ Patterns
@app.get("/patterns", response_model=PatternListResponse)
def get_patterns() -> PatternListResponse:
    from . import extractor as ext_mod
    patterns = ext_mod.list_patterns()
    return PatternListResponse(
        patterns=[
            PatternInfo(
                id=p.id,
                name=p.name,
                description=p.description,
                detect_keywords=p.detect_keywords,
                field_count=len(p.fields),
            )
            for p in patterns
        ]
    )
