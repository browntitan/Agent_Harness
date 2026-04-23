from __future__ import annotations

from typing import Any, Iterable, Sequence


def resolver_query_char_limit(settings: Any, *, default: int = 4000) -> int:
    try:
        limit = int(getattr(settings, "skill_context_max_chars", default) or default)
    except (TypeError, ValueError):
        limit = default
    return max(1, limit)


def build_skill_resolver_query(
    settings: Any,
    *,
    base_query: str,
    skill_queries: Sequence[str] | None = None,
) -> str:
    limit = resolver_query_char_limit(settings)
    remaining = limit
    parts: list[str] = []
    seen: set[str] = set()

    for raw_part in _iter_parts(base_query, skill_queries or ()):
        if remaining <= 0:
            break
        key = raw_part.casefold()
        if key in seen:
            continue
        seen.add(key)
        budget = remaining if not parts else remaining - 1
        if budget <= 0:
            break
        snippet = raw_part[:budget].strip()
        if not snippet:
            continue
        parts.append(snippet)
        remaining = limit - len("\n".join(parts))

    return "\n".join(parts).strip()


def _iter_parts(base_query: str, skill_queries: Sequence[str]) -> Iterable[str]:
    base = str(base_query or "").strip()
    if base:
        yield base
    for item in skill_queries:
        text = str(item or "").strip()
        if text:
            yield text


__all__ = ["build_skill_resolver_query", "resolver_query_char_limit"]
