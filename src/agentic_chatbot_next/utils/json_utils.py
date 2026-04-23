from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_REASONING_TAG_RE = re.compile(r"<(think|analysis|reasoning)>\s*.*?\s*</\1>", re.DOTALL | re.IGNORECASE)


def _sanitize_json_text(text: str) -> str:
    cleaned = _REASONING_TAG_RE.sub("", text or "")
    return cleaned.strip()


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object from model output."""

    if not isinstance(text, str):
        text = str(text or "")
    text = _sanitize_json_text(text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    match = _JSON_RE.search(text)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def make_json_compatible(value: Any) -> Any:
    """Recursively normalize common runtime objects into JSON-safe values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return make_json_compatible(asdict(value))
    if isinstance(value, Mapping):
        normalized: Dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, (str, int, float, bool)) or key is None:
                normalized_key = key
            else:
                normalized_key = str(key)
            normalized[normalized_key] = make_json_compatible(item)
        return normalized
    if isinstance(value, (list, tuple, set, frozenset)):
        return [make_json_compatible(item) for item in value]
    return value
