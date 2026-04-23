from __future__ import annotations

from typing import Any


def coerce_optional_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def resolve_chat_output_cap(
    settings: Any,
    *,
    agent_name: str = "",
    request_max_tokens: Any = None,
    demo_mode: bool = False,
) -> int | None:
    request_cap = coerce_optional_positive_int(request_max_tokens)
    if request_cap is not None:
        return request_cap

    normalized_agent = str(agent_name or "").strip().lower()
    agent_overrides = dict(getattr(settings, "agent_chat_max_output_tokens", {}) or {})
    agent_cap = coerce_optional_positive_int(agent_overrides.get(normalized_agent))
    if agent_cap is not None:
        return agent_cap

    if demo_mode:
        demo_cap = coerce_optional_positive_int(getattr(settings, "demo_chat_max_output_tokens", None))
        if demo_cap is not None:
            return demo_cap

    global_cap = coerce_optional_positive_int(getattr(settings, "chat_max_output_tokens", None))
    if global_cap is not None:
        return global_cap

    if demo_mode:
        demo_legacy_cap = coerce_optional_positive_int(getattr(settings, "demo_ollama_num_predict", None))
        if demo_legacy_cap is not None:
            return demo_legacy_cap

    return coerce_optional_positive_int(getattr(settings, "ollama_num_predict", None))


def resolve_judge_output_cap(settings: Any) -> int | None:
    configured = coerce_optional_positive_int(getattr(settings, "judge_max_output_tokens", None))
    if configured is not None:
        return configured
    return coerce_optional_positive_int(getattr(settings, "ollama_num_predict", None))

