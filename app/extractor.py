"""Pattern-based field extraction engine.

Patterns are JSON files that define:
  - detect: keywords to auto-detect the document type
  - fields: rules to extract each field (regex, anchor, transform)

This is a deterministic, no-AI extraction layer that works on raw OCR text.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class LoadedPattern:
    id: str
    name: str
    description: str
    detect_keywords: List[str]
    fields: Dict[str, Dict[str, Any]]


def _patterns_dir() -> str:
    return settings.patterns_dir


def load_pattern(pattern_id: str) -> Optional[LoadedPattern]:
    """Load a pattern by id from the patterns directory."""
    path = os.path.join(_patterns_dir(), f"{pattern_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return LoadedPattern(
        id=pattern_id,
        name=data.get("name", pattern_id),
        description=data.get("description", ""),
        detect_keywords=data.get("detect", {}).get("any_keyword", []),
        fields=data.get("fields", {}),
    )


def list_patterns() -> List[LoadedPattern]:
    """List all available patterns."""
    d = _patterns_dir()
    if not os.path.isdir(d):
        return []
    out: List[LoadedPattern] = []
    for fname in sorted(os.listdir(d)):
        if not fname.endswith(".json"):
            continue
        pid = fname[:-5]
        p = load_pattern(pid)
        if p:
            out.append(p)
    return out


def auto_detect_pattern(text: str) -> Optional[LoadedPattern]:
    """Find the first pattern whose detect keywords match the text."""
    for p in list_patterns():
        if not p.detect_keywords:
            continue
        text_lower = text.lower()
        if any(kw.lower() in text_lower for kw in p.detect_keywords):
            return p
    return None


def _apply_regex(text: str, rule: Dict[str, Any]) -> Optional[str]:
    pattern = rule.get("regex")
    if not pattern:
        return None
    flags = re.MULTILINE
    if rule.get("ignore_case"):
        flags |= re.IGNORECASE
    m = re.search(pattern, text, flags)
    if not m:
        return None
    if m.groups():
        return m.group(1).strip()
    return m.group(0).strip()


def _apply_anchor(text: str, lines: List[str], rule: Dict[str, Any]) -> Optional[str]:
    anchor = rule.get("anchor")
    if not anchor:
        return None
    direction = rule.get("direction", "below")
    stop_words = rule.get("stop_at", [])
    ignore_words = rule.get("ignore", [])

    for i, line in enumerate(lines):
        if anchor.lower() in line.lower():
            if direction == "below":
                for j in range(i + 1, len(lines)):
                    candidate = lines[j].strip()
                    if not candidate:
                        continue
                    if any(sw.lower() in candidate.lower() for sw in stop_words):
                        break
                    if any(iw.lower() in candidate.lower() for iw in ignore_words):
                        continue
                    return candidate
            elif direction == "right":
                idx = line.lower().find(anchor.lower())
                if idx >= 0:
                    rest = line[idx + len(anchor):].strip(" :|\t")
                    if rest:
                        return rest
                if i + 1 < len(lines):
                    return lines[i + 1].strip()
            elif direction == "same_line":
                idx = line.lower().find(anchor.lower())
                if idx >= 0:
                    rest = line[idx + len(anchor):].strip(" :|\t")
                    if rest:
                        return rest
    return None


def _transform(value: str, transform: str) -> Any:
    if transform == "vnd_to_number":
        cleaned = value.replace(".", "").replace(",", "").replace("VND", "").replace("vnd", "").strip()
        try:
            return int(cleaned)
        except ValueError:
            try:
                return float(cleaned)
            except ValueError:
                return value
    elif transform == "to_number":
        cleaned = value.replace(".", "").replace(",", "").strip()
        try:
            return int(cleaned)
        except ValueError:
            try:
                return float(cleaned)
            except ValueError:
                return value
    elif transform == "strip":
        return value.strip()
    elif transform == "lower":
        return value.lower()
    elif transform == "upper":
        return value.upper()
    return value


def extract_with_pattern(text: str, pattern: LoadedPattern) -> Dict[str, Any]:
    """Extract fields from text using a loaded pattern."""
    lines = text.split("\n")
    result: Dict[str, Any] = {}
    for field_name, rule in pattern.fields.items():
        value: Optional[str] = None
        if "regex" in rule:
            value = _apply_regex(text, rule)
        if value is None and "anchor" in rule:
            value = _apply_anchor(text, lines, rule)
        if value is not None:
            transform = rule.get("transform")
            if transform:
                value = _transform(value, transform)
            result[field_name] = value
        elif "default" in rule:
            result[field_name] = rule["default"]
    return result
