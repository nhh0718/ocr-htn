"""LLM-based extraction layer (schema-driven).

Supports Gemini and OpenAI providers. When provider is "none" or no API key
is set, extraction is unavailable and the caller should fall back to patterns.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from .config import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a document extraction assistant. Extract structured data from the "
    "given OCR text. Return ONLY valid JSON matching the provided JSON Schema. "
    "Do not include markdown fences or explanations. If a field cannot be "
    "determined, use null."
)


def is_available() -> bool:
    return settings.extraction_provider != "none" and bool(settings.extraction_api_key)


def extract_via_llm(text: str, schema: Dict[str, Any], instructions: Optional[str] = None) -> Dict[str, Any]:
    provider = settings.extraction_provider
    if provider == "gemini":
        return _extract_gemini(text, schema, instructions)
    elif provider == "openai":
        return _extract_openai(text, schema, instructions)
    raise RuntimeError(f"Extraction provider '{provider}' is not configured.")


def _extract_gemini(text: str, schema: Dict[str, Any], instructions: Optional[str]) -> Dict[str, Any]:
    import urllib.request

    api_key = settings.extraction_api_key
    model = settings.extraction_model or "gemini-2.0-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    user_prompt = f"Extract structured data from this OCR text.\n"
    if instructions:
        user_prompt += f"Context: {instructions}\n"
    user_prompt += f"\nJSON Schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n"
    user_prompt += f"\nOCR Text:\n{text}\n"
    user_prompt += "\nReturn ONLY a JSON object matching the schema."

    body = json.dumps({
        "contents": [{"parts": [{"text": _SYSTEM_PROMPT + "\n\n" + user_prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.0,
        },
    }).encode("utf-8")

    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    candidates = data.get("candidates", [])
    if not candidates:
        return {}
    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        return {}
    raw_json = parts[0].get("text", "{}")
    return json.loads(raw_json)


def _extract_openai(text: str, schema: Dict[str, Any], instructions: Optional[str]) -> Dict[str, Any]:
    import urllib.request

    api_key = settings.extraction_api_key
    model = settings.extraction_model or "gpt-4o-mini"
    url = "https://api.openai.com/v1/chat/completions"

    user_prompt = f"Extract structured data from this OCR text.\n"
    if instructions:
        user_prompt += f"Context: {instructions}\n"
    user_prompt += f"\nJSON Schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n"
    user_prompt += f"\nOCR Text:\n{text}\n"
    user_prompt += "\nReturn ONLY a JSON object matching the schema."

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    choices = data.get("choices", [])
    if not choices:
        return {}
    content = choices[0].get("message", {}).get("content", "{}")
    return json.loads(content)
