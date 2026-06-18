"""Pydantic models for the OCR & extraction API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- OCR
class OCRBase64Request(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image bytes (with or without data URL prefix).")


class TextBlockOut(BaseModel):
    text: str
    bbox: List[float] = Field(..., description="[x_min, y_min, x_max, y_max] in image pixels.")
    confidence: float = Field(1.0, description="Recognition confidence (0-1).")
    line_index: int = Field(..., description="Line number (0-based, top-to-bottom).")


class OCRResponse(BaseModel):
    text: str = Field(..., description="Full recognized text, lines joined by \\n.")
    lines: List[str] = Field(default_factory=list, description="Recognized lines in reading order.")
    blocks: List[TextBlockOut] = Field(
        default_factory=list,
        description="Per-word/phrase blocks with bbox, confidence, and line index.",
    )
    image_size: List[int] = Field(..., description="[width, height] of the processed image.")
    elapsed_ms: int = Field(..., description="Server-side processing time in milliseconds.")


# --------------------------------------------------------------------- Extraction
class ExtractRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image_base64: Optional[str] = Field(None, description="Base64 image to OCR first.")
    text: Optional[str] = Field(None, description="Pre-OCR'd text. Used if image_base64 is absent.")
    pattern_id: Optional[str] = Field(None, description="Named pattern from patterns/ dir.")
    pattern: Optional[Dict[str, Any]] = Field(None, description="Inline pattern definition.")
    json_schema: Optional[Dict[str, Any]] = Field(
        None,
        alias="schema",
        description="JSON Schema for LLM extraction (requires EXTRACTION_PROVIDER set).",
    )
    instructions: Optional[str] = Field(
        None,
        description="Extra hint for LLM extraction, e.g. 'This is a Vietnamese bank transfer receipt'.",
    )


class ExtractResponse(BaseModel):
    method: str = Field(..., description="pattern | llm | none")
    pattern_id: Optional[str] = None
    fields: Dict[str, Any] = Field(default_factory=dict, description="Extracted key-value pairs.")
    raw_text: str = Field("", description="OCR text used for extraction.")
    elapsed_ms: int = Field(..., description="Total processing time in milliseconds.")


# ------------------------------------------------------------------------ Patterns
class PatternInfo(BaseModel):
    id: str
    name: str
    description: str
    detect_keywords: List[str] = Field(default_factory=list)
    field_count: int


class PatternListResponse(BaseModel):
    patterns: List[PatternInfo]


# ------------------------------------------------------------------------ Health
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    engine_ready: bool
    extraction_provider: str = "none"
