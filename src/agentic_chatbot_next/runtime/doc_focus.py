from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from agentic_chatbot_next.rag.hints import normalize_structured_query
from agentic_chatbot_next.rag.inventory import INVENTORY_QUERY_NONE, classify_inventory_query

_DOCSET_REFERENCE_HINTS = re.compile(
    r"\b("
    r"candidate\s+(?:documents|docs)|"
    r"documents?\s+you\s+provided|docs?\s+you\s+provided|"
    r"documents?\s+you\s+gave|docs?\s+you\s+gave|"
    r"documents?\s+above|docs?\s+above|"
    r"those\s+documents|those\s+docs|"
    r"look\s+through\s+(?:them|those\s+documents|those\s+docs|the\s+candidate\s+documents)"
    r")\b",
    re.IGNORECASE,
)
_FOLLOWUP_SYNTHESIS_HINTS = re.compile(
    r"\b("
    r"summary|summari[sz]e|explain|walk\s+me\s+through|walk\s+through|"
    r"look\s+through|review|analy[sz]e|detailed|comprehensive|verbose|"
    r"thorough|major\s+subsystems?|architect(?:ure|ural)|subsystems?"
    r")\b",
    re.IGNORECASE,
)
_DISCOVERY_FOLLOWUP_HINTS = re.compile(
    r"\b("
    r"identify|find|list|which|what"
    r")\s+(?:more\s+)?(?:documents|docs|files)\b",
    re.IGNORECASE,
)


def _normalize_documents(raw_documents: Iterable[Any]) -> List[Dict[str, str]]:
    documents: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_documents:
        item = dict(raw or {}) if isinstance(raw, dict) else {}
        doc_id = str(item.get("doc_id") or "").strip()
        title = str(item.get("title") or "").strip()
        source_path = str(item.get("source_path") or "").strip()
        source_type = str(item.get("source_type") or "").strip()
        key = (doc_id, title.casefold())
        if key == ("", "") or key in seen:
            continue
        seen.add(key)
        documents.append(
            {
                "doc_id": doc_id,
                "title": title,
                "source_path": source_path,
                "source_type": source_type,
            }
        )
    return documents


def build_doc_focus_result(
    *,
    collection_id: str,
    documents: Iterable[Any],
    source_query: str = "",
    result_mode: str = "",
) -> Dict[str, Any] | None:
    normalized_docs = _normalize_documents(documents)
    if not normalized_docs:
        return None
    return {
        "collection_id": str(collection_id or "default").strip() or "default",
        "documents": normalized_docs,
        "source_query": str(source_query or "").strip(),
        "result_mode": str(result_mode or "").strip(),
    }


def coerce_doc_focus_result(raw: Any) -> Dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    payload = build_doc_focus_result(
        collection_id=str(raw.get("collection_id") or "default"),
        documents=raw.get("documents") or [],
        source_query=str(raw.get("source_query") or ""),
        result_mode=str(raw.get("result_mode") or ""),
    )
    return payload


def active_doc_focus_from_metadata(metadata: Dict[str, Any] | None) -> Dict[str, Any] | None:
    raw = dict(metadata or {}).get("active_doc_focus")
    return coerce_doc_focus_result(raw)


def doc_focus_result_from_metadata(metadata: Dict[str, Any] | None) -> Dict[str, Any] | None:
    raw = dict(metadata or {}).get("doc_focus_result")
    return coerce_doc_focus_result(raw)


def active_doc_focus_doc_ids(metadata: Dict[str, Any] | None) -> List[str]:
    payload = active_doc_focus_from_metadata(metadata)
    if payload is None:
        return []
    return [str(item.get("doc_id") or "").strip() for item in payload.get("documents") or [] if str(item.get("doc_id") or "").strip()]


def active_doc_focus_prompt_block(metadata: Dict[str, Any] | None, *, limit: int = 6) -> str:
    payload = active_doc_focus_from_metadata(metadata)
    if payload is None:
        return ""
    documents = [dict(item) for item in (payload.get("documents") or []) if isinstance(item, dict)]
    if not documents:
        return ""
    rows = []
    for item in documents[:limit]:
        title = str(item.get("title") or item.get("doc_id") or "").strip()
        doc_id = str(item.get("doc_id") or "").strip()
        if title and doc_id and title != doc_id:
            rows.append(f"- {title} ({doc_id})")
        elif title:
            rows.append(f"- {title}")
    if not rows:
        return ""
    return (
        "## Active Document Focus\n"
        "The conversation currently has a stored candidate-document set from a prior discovery step. "
        "If the user refers to candidate documents, those docs, or the docs above, treat this set as the default scope unless the user explicitly broadens scope.\n"
        f"Collection: {payload.get('collection_id')}\n"
        + "\n".join(rows)
    ).strip()


def is_active_doc_focus_followup(query: str, metadata: Dict[str, Any] | None) -> bool:
    payload = active_doc_focus_from_metadata(metadata)
    if payload is None or not list(payload.get("documents") or []):
        return False
    normalized_query = normalize_structured_query(query) or str(query or "")
    if not normalized_query.strip():
        return False
    if classify_inventory_query(normalized_query) != INVENTORY_QUERY_NONE:
        return False
    if _DISCOVERY_FOLLOWUP_HINTS.search(normalized_query):
        return False
    if not _DOCSET_REFERENCE_HINTS.search(normalized_query):
        return False
    return bool(_FOLLOWUP_SYNTHESIS_HINTS.search(normalized_query))


def active_doc_focus_controller_hints(
    query: str,
    metadata: Dict[str, Any] | None,
) -> Dict[str, Any]:
    payload = active_doc_focus_from_metadata(metadata)
    if payload is None or not is_active_doc_focus_followup(query, metadata):
        return {}
    doc_ids = [str(item.get("doc_id") or "").strip() for item in payload.get("documents") or [] if str(item.get("doc_id") or "").strip()]
    collection_id = str(payload.get("collection_id") or "default").strip() or "default"
    return {
        "summary_scope": "active_doc_focus",
        "prefer_detailed_synthesis": True,
        "final_output_mode": "detailed_subsystem_summary",
        "active_doc_focus_doc_ids": doc_ids,
        "active_doc_focus_documents": [dict(item) for item in (payload.get("documents") or []) if isinstance(item, dict)],
        "requested_kb_collection_id": collection_id,
        "kb_collection_id": collection_id,
        "search_collection_ids": [collection_id],
        "retrieval_scope_mode": "kb_only",
        "strict_kb_scope": True,
    }


__all__ = [
    "active_doc_focus_controller_hints",
    "active_doc_focus_doc_ids",
    "active_doc_focus_from_metadata",
    "active_doc_focus_prompt_block",
    "build_doc_focus_result",
    "coerce_doc_focus_result",
    "doc_focus_result_from_metadata",
    "is_active_doc_focus_followup",
]
