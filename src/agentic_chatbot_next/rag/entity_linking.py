from __future__ import annotations

import re
from typing import Any, Dict, List

from agentic_chatbot_next.persistence.postgres.entities import normalize_entity_text

_QUERY_ENTITY_STOPWORDS = {
    "compare",
    "describe",
    "does",
    "explain",
    "find",
    "how",
    "identify",
    "list",
    "show",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


def extract_entity_candidates(query: str) -> List[str]:
    entities: List[str] = []
    seen: set[str] = set()
    for match in re.findall(r"\b(?:[A-Z][A-Za-z0-9_-]{2,}(?:\s+[A-Z][A-Za-z0-9_-]{2,})*)\b", str(query or "")):
        value = str(match).strip()
        if not value:
            continue
        parts = [part for part in value.split() if part]
        if len(parts) == 1 and parts[0].casefold() in _QUERY_ENTITY_STOPWORDS:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        entities.append(value)
    return entities[:8]


def resolve_query_entities(
    *,
    query: str,
    stores: Any,
    settings: Any,
    session: Any,
    collection_id: str = "",
    controller_hints: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    hints = dict(controller_hints or {})
    tenant_id = str(getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev")) or "local-dev")
    effective_collection = str(collection_id or getattr(settings, "default_collection_id", "default") or "default")
    entity_store = getattr(stores, "entity_store", None)

    resolved: List[Dict[str, Any]] = []
    if entity_store is not None and bool(getattr(settings, "entity_linking_enabled", True)):
        try:
            resolved.extend(
                entity_store.resolve_aliases(
                    query,
                    tenant_id=tenant_id,
                    collection_id=effective_collection,
                    limit=8,
                )
            )
        except Exception:
            resolved = []

    seen_ids = {str(item.get("entity_id") or "") for item in resolved if str(item.get("entity_id") or "")}
    for hint in [str(item) for item in (hints.get("entity_candidates") or []) if str(item)] + extract_entity_candidates(query):
        normalized = normalize_entity_text(hint)
        if not normalized:
            continue
        entity_id = f"candidate:{normalized}"
        if entity_id in seen_ids:
            continue
        seen_ids.add(entity_id)
        resolved.append(
            {
                "entity_id": entity_id,
                "canonical_name": hint,
                "entity_type": "",
                "description": "",
                "graph_id": "",
                "matched_alias": hint,
                "score": 1.0,
            }
        )
    return resolved[:8]


__all__ = ["extract_entity_candidates", "resolve_query_entities"]
