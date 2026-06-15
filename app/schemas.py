"""Pydantic models for the OCR API."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class OCRBase64Request(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image bytes (with or without data URL prefix).")


class OCRResponse(BaseModel):
    text: str = Field(..., description="Full recognized text, lines joined by \\n.")
    lines: List[str] = Field(default_factory=list, description="Recognized lines in reading order.")
    elapsed_ms: int = Field(..., description="Server-side processing time in milliseconds.")


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    engine_ready: bool
