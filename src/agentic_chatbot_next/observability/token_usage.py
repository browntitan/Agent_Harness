from __future__ import annotations

from typing import Any, Dict


def _coerce_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def extract_token_usage(payload: Any) -> Dict[str, int]:
    data = dict(payload or {}) if isinstance(payload, dict) else {}
    usage = data.get("token_usage")
    if isinstance(usage, dict):
        prompt_tokens = _coerce_int(usage.get("prompt_tokens"))
        completion_tokens = _coerce_int(usage.get("completion_tokens"))
        total_tokens = _coerce_int(usage.get("total_tokens"))
        if prompt_tokens or completion_tokens or total_tokens:
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens or prompt_tokens + completion_tokens,
            }

    nested = data.get("usage")
    if isinstance(nested, dict):
        prompt_tokens = _coerce_int(nested.get("prompt_tokens"))
        completion_tokens = _coerce_int(nested.get("completion_tokens"))
        total_tokens = _coerce_int(nested.get("total_tokens"))
        if prompt_tokens or completion_tokens or total_tokens:
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens or prompt_tokens + completion_tokens,
            }

    prompt_tokens = _coerce_int(
        data.get("prompt_tokens")
        or data.get("input_tokens")
        or data.get("prompt_token_count")
    )
    completion_tokens = _coerce_int(
        data.get("completion_tokens")
        or data.get("output_tokens")
        or data.get("completion_token_count")
    )
    total_tokens = _coerce_int(
        data.get("total_tokens")
        or data.get("token_count")
        or data.get("tokens")
    )
    if prompt_tokens or completion_tokens or total_tokens:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens or prompt_tokens + completion_tokens,
        }
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


__all__ = ["extract_token_usage"]
