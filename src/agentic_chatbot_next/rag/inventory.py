from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Sequence

from agentic_chatbot_next.authz import access_summary_authz_enabled
from agentic_chatbot_next.rag.ingest import get_collection_readiness_status, get_kb_coverage_status
from agentic_chatbot_next.rag.retrieval_scope import (
    has_upload_evidence,
    resolve_available_kb_collection_ids,
    resolve_kb_collection_confirmed,
    resolve_kb_collection_id,
    resolve_upload_collection_id,
)
from agentic_chatbot_next.utils.json_utils import make_json_compatible

INVENTORY_QUERY_NONE = "none"
INVENTORY_QUERY_KB_FILE = "kb_file_inventory"
INVENTORY_QUERY_KB_COLLECTIONS = "kb_collection_access_inventory"
INVENTORY_QUERY_SESSION_ACCESS = "session_access_inventory"
INVENTORY_QUERY_GRAPH_INDEXES = "graph_index_inventory"
INVENTORY_QUERY_GRAPH_FILE = "graph_file_inventory"

NAMESPACE_SCOPE_SELECTION_REASON = "namespace_scope_selection"

SESSION_ACCESS_NEXT_ACTIONS = (
    "search uploaded docs",
    "search the KB",
    "search both",
    "list KB files",
)

_COLLECTION_NAME_PATTERN = r"(?:`[^`]+`|'[^']+'|\"[^\"]+\"|[A-Za-z0-9][A-Za-z0-9_.:-]*)"
_COLLECTION_SCOPED_DOC_LIST_PATTERNS = (
    re.compile(
        r"\b(?:what|which|show|list)\b.*\b(?:documents|docs|files|individual\s+files)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:list|show)\b.*\b(?:all\s+of\s+the\s+)?(?:documents|docs|files)\b",
        re.IGNORECASE,
    ),
)
_COLLECTION_REFERENCE_PATTERNS = (
    re.compile(
        rf"\b(?:in|inside|within)\s+(?:the\s+)?(?P<collection>{_COLLECTION_NAME_PATTERN})\s+collection\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bfrom\s+(?:the\s+)?collection\s+(?P<collection>{_COLLECTION_NAME_PATTERN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bcollection\s+(?P<collection>{_COLLECTION_NAME_PATTERN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:inside|in)\s+(?:the\s+)?(?P<collection>{_COLLECTION_NAME_PATTERN})\s+(?:knowledge\s*base|kb)\b",
        re.IGNORECASE,
    ),
)
_GRAPH_REFERENCE_PATTERNS = (
    re.compile(
        rf"\b(?:in|inside|within|from|for)\s+(?:the\s+)?(?P<graph>{_COLLECTION_NAME_PATTERN})\s+(?:knowledge\s+graph|graph(?:\s+index)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:knowledge\s+graph|graph(?:\s+index)?)\s+(?P<graph>{_COLLECTION_NAME_PATTERN})\b",
        re.IGNORECASE,
    ),
)
_NAMESPACE_REFERENCE_PATTERNS = (
    re.compile(
        rf"\b(?:in|inside|within|from)\s+(?:the\s+)?(?P<namespace>{_COLLECTION_NAME_PATTERN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:use|search|consult|refer(?:\s+to)?|look\s+in|within|inside)\s+(?:the\s+)?(?P<namespace>{_COLLECTION_NAME_PATTERN})\b",
        re.IGNORECASE,
    ),
)

_KB_FILE_INVENTORY_PATTERNS = (
    re.compile(r"\bwhat(?:'s|\s+is)\s+indexed\b", re.IGNORECASE),
    re.compile(r"\bknowledge\s*base\s+inventory\b", re.IGNORECASE),
    re.compile(
        r"\b(?:what|which)\b.*\b(?:knowledge\s*base|kb)\s+(?:documents|docs|files)\b.*\b(?:have\s+access\s+to|are\s+available|are\s+in)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:what|which)\s+(?:documents|docs|files)\b.*\b(?:have\s+access\s+to|are\s+available|are\s+in)\b.*\b(?:knowledge\s*base|kb)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:show|list)\b.*\b(?:knowledge\s*base|kb)\b.*\b(?:documents|docs|files)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:show|list)\b.*\b(?:documents|docs|files)\b.*\b(?:knowledge\s*base|kb)\b",
        re.IGNORECASE,
    ),
)
_GRAPH_FILE_INVENTORY_PATTERNS = (
    re.compile(
        r"\b(?:what|which|show|list)\b.*\b(?:documents|docs|files|sources?)\b.*\b(?:knowledge\s+graph|graph(?:\s+index)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:what|which|show|list)\b.*\b(?:knowledge\s+graph|graph(?:\s+index)?)\b.*\b(?:documents|docs|files|sources?)\b",
        re.IGNORECASE,
    ),
)
_KB_COLLECTION_ACCESS_PATTERNS = (
    re.compile(
        r"\b(?:what|which)\b.*\b(?:knowledge\s*bases?|kbs?|kb\s+collections?)\b.*\b(?:have\s+access\s+to|are\s+available)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:show|list)(?:\s+out)?\b.*\b(?:knowledge\s*bases?|kbs?|kb\s+collections?)\b.*\b(?:have\s+access\s+to|available)?\b",
        re.IGNORECASE,
    ),
)
_SESSION_ACCESS_PATTERNS = (
    re.compile(
        r"\bwhat\s+(?:documents|docs|files)\s+do\s+(?:we|you)\s+have(?:\s+access\s+to)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+(?:documents|docs|files)\s+are\s+available\b",
        re.IGNORECASE,
    ),
)
_GRAPH_INDEX_INVENTORY_PATTERNS = (
    re.compile(
        r"\b(?:what|which)\s+(?:knowledge\s+graphs?|graphs?|graph\s+indexes?)\b.*\b(?:do\s+(?:we|you|i)\s+have(?:\s+available|\s+access\s+to)?|are\s+available|exist)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:show|list)\b.*\b(?:my|available|the)?\s*(?:knowledge\s+graphs?|graphs?|graph\s+indexes?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bwhat\s+graphs\s+exist\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+graphs\s+exist\b", re.IGNORECASE),
)
_FILTER_HINTS = re.compile(
    r"\b("
    r"mention|mentions|mentioned|contain|contains|containing|cover|covers|covering|"
    r"describe|describes|describing|discuss|discusses|discussing|"
    r"information\s+about|details\s+about|"
    r"about|for|with|related\s+to|matching|that\s+(?:mention|mentions|contain|contains|cover|covers|have)|"
    r"workflow|workflows|process|processes|policy|policies|onboarding|pricing|security|architecture|"
    r"approval|incident|contract|contracts|requirement|requirements"
    r")\b",
    re.IGNORECASE,
)
_DISCOVERY_OVERRIDE_HINTS = re.compile(
    r"\b("
    r"identify\s+all|which\s+documents|find\s+all|across\s+(?:the\s+)?(?:corpus|documents|policies|sops)|"
    r"every\s+document|exhaustive|compare|difference|differences|versus|vs\.?|contrast|"
    r"process\s+flows?|flowcharts?|approval\s+flows?|handoff|escalation|"
    r"investigate|potential\s+(?:documents|files)|"
    r"(?:provide|give|return)\s+(?:me\s+)?(?:only\s+)?(?:a\s+)?list\s+of\s+(?:potential\s+)?(?:documents|files)|"
    r"(?:documents|files)\s+that\s+(?:have|contain)\s+information\s+about|"
    r"major\s+subsystems?"
    r")\b",
    re.IGNORECASE,
)
_INVENTORY_ANALYSIS_HINTS = re.compile(
    r"\b("
    r"analy(?:s|z)e|analysis|compare|comparison|diff|difference|versus|vs\.?|"
    r"explain|describe|summari(?:s|z)e|summary|synthesi(?:s|ze)|walk\s*through|"
    r"deep|detailed|thorough|comprehensive|"
    r"evidence|grounded|supporting\s+documents?|citations?|cite"
    r")\b",
    re.IGNORECASE,
)
_NAMESPACE_DOC_LIST_HINTS = re.compile(
    r"\b(?:what|which|show|list)\b.*\b(?:documents|docs|files|titles?|source\s+docs?)\b",
    re.IGNORECASE,
)
_NAMESPACE_SCOPE_HINTS = re.compile(
    r"\b(?:use|search|consult|refer(?:\s+to)?|look\s+in|inside|within)\b",
    re.IGNORECASE,
)
_NAMESPACE_STOPWORDS = {
    "",
    "the",
    "a",
    "an",
    "knowledge",
    "knowledge base",
    "kb",
    "collection",
    "graph",
    "knowledge graph",
    "it",
    "them",
    "this",
    "that",
}

_COLLECTION_TOPIC_LABELS = (
    "product overviews",
    "pricing",
    "security",
    "integrations",
    "release notes",
    "legal contracts",
    "ai-ops standards",
    "incident playbooks",
    "vendor security",
    "architecture",
    "api docs",
    "internal process guides",
)


def _clean_collection_reference(value: str) -> str:
    text = str(value or "").strip().strip(".,!?;:")
    if len(text) >= 2 and text[:1] == text[-1:] and text[0] in {"'", '"', "`"}:
        text = text[1:-1].strip()
    return " ".join(text.split())


def _normalize_namespace_text(value: str) -> str:
    text = str(value or "").strip().casefold()
    if not text:
        return ""
    text = re.sub(r"[_\-.:/]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())


def _namespace_tokens(value: str) -> tuple[str, ...]:
    normalized = _normalize_namespace_text(value)
    if not normalized:
        return ()
    return tuple(token for token in normalized.split() if token)


def _is_namespace_stopword(value: str) -> bool:
    normalized = _normalize_namespace_text(value)
    return normalized in _NAMESPACE_STOPWORDS


def _stable_namespace_boundary_pattern(value: str) -> str:
    escaped = re.escape(str(value or "").strip())
    if not escaped:
        return r"$^"
    return rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"


def extract_requested_kb_collection_id(query: str) -> str:
    text = str(query or "").strip()
    if not text:
        return ""
    for pattern in _COLLECTION_REFERENCE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        candidate = _clean_collection_reference(match.group("collection"))
        if candidate.casefold() not in {"", "the", "a", "an"}:
            return candidate
    return ""


def extract_requested_graph_reference(query: str) -> str:
    text = str(query or "").strip()
    if not text:
        return ""
    for pattern in _GRAPH_REFERENCE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        candidate = _clean_collection_reference(match.group("graph"))
        if not _is_namespace_stopword(candidate):
            return candidate
    return ""


def extract_requested_namespace_query(query: str) -> str:
    text = str(query or "").strip()
    if not text:
        return ""
    graph_ref = extract_requested_graph_reference(text)
    if graph_ref:
        return graph_ref
    collection_ref = extract_requested_kb_collection_id(text)
    if collection_ref:
        return collection_ref
    for pattern in _NAMESPACE_REFERENCE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        candidate = _clean_collection_reference(match.group("namespace"))
        if not _is_namespace_stopword(candidate):
            return candidate
    return ""


def _is_collection_scoped_kb_file_inventory_query(query: str) -> bool:
    text = str(query or "").strip()
    if not text or not extract_requested_kb_collection_id(text):
        return False
    return any(pattern.search(text) for pattern in _COLLECTION_SCOPED_DOC_LIST_PATTERNS)


def _is_graph_scoped_file_inventory_query(query: str) -> bool:
    text = str(query or "").strip()
    if not text or not extract_requested_graph_reference(text):
        return False
    return any(pattern.search(text) for pattern in _GRAPH_FILE_INVENTORY_PATTERNS)


def _is_namespace_scoped_doc_inventory_query(query: str) -> bool:
    text = str(query or "").strip()
    if not text or extract_requested_kb_collection_id(text) or extract_requested_graph_reference(text):
        return False
    if not extract_requested_namespace_query(text):
        return False
    return bool(_NAMESPACE_DOC_LIST_HINTS.search(text))


def _is_namespace_scope_query(query: str) -> bool:
    text = str(query or "").strip()
    if not text:
        return False
    if str(text).strip().lower() in {"use all", "all", "collections only", "graphs only"}:
        return True
    if not extract_requested_namespace_query(text):
        return False
    return bool(_NAMESPACE_SCOPE_HINTS.search(text))


def classify_inventory_query(query: str) -> str:
    text = str(query or "").strip()
    if not text:
        return INVENTORY_QUERY_NONE
    if _is_graph_scoped_file_inventory_query(text):
        return INVENTORY_QUERY_NONE if _FILTER_HINTS.search(text) else INVENTORY_QUERY_GRAPH_FILE
    if _is_collection_scoped_kb_file_inventory_query(text):
        return INVENTORY_QUERY_NONE if _FILTER_HINTS.search(text) else INVENTORY_QUERY_KB_FILE
    if _is_namespace_scoped_doc_inventory_query(text):
        return INVENTORY_QUERY_NONE if _FILTER_HINTS.search(text) else INVENTORY_QUERY_KB_FILE
    if any(pattern.search(text) for pattern in _GRAPH_INDEX_INVENTORY_PATTERNS):
        return INVENTORY_QUERY_NONE if _FILTER_HINTS.search(text) else INVENTORY_QUERY_GRAPH_INDEXES
    if _DISCOVERY_OVERRIDE_HINTS.search(text):
        return INVENTORY_QUERY_NONE
    if any(pattern.search(text) for pattern in _KB_FILE_INVENTORY_PATTERNS):
        return INVENTORY_QUERY_NONE if _FILTER_HINTS.search(text) else INVENTORY_QUERY_KB_FILE
    if any(pattern.search(text) for pattern in _KB_COLLECTION_ACCESS_PATTERNS):
        return INVENTORY_QUERY_NONE if _FILTER_HINTS.search(text) else INVENTORY_QUERY_KB_COLLECTIONS
    if any(pattern.search(text) for pattern in _SESSION_ACCESS_PATTERNS):
        return INVENTORY_QUERY_NONE if _FILTER_HINTS.search(text) else INVENTORY_QUERY_SESSION_ACCESS
    return INVENTORY_QUERY_NONE


def is_inventory_query(query: str) -> bool:
    return classify_inventory_query(query) != INVENTORY_QUERY_NONE


def is_authoritative_inventory_query_type(query_type: str) -> bool:
    return str(query_type or "").strip() in {
        INVENTORY_QUERY_KB_FILE,
        INVENTORY_QUERY_KB_COLLECTIONS,
        INVENTORY_QUERY_SESSION_ACCESS,
        INVENTORY_QUERY_GRAPH_INDEXES,
        INVENTORY_QUERY_GRAPH_FILE,
    }


def inventory_scope_kind(query_type: str) -> str:
    normalized = str(query_type or "").strip()
    if normalized == INVENTORY_QUERY_SESSION_ACCESS:
        return "session_access"
    if normalized in {INVENTORY_QUERY_GRAPH_INDEXES, INVENTORY_QUERY_GRAPH_FILE}:
        return "graph_indexes"
    if normalized in {INVENTORY_QUERY_KB_FILE, INVENTORY_QUERY_KB_COLLECTIONS}:
        return "knowledge_base"
    return "none"


def inventory_answer_origin(query_type: str) -> str:
    normalized = str(query_type or "").strip()
    if normalized == INVENTORY_QUERY_GRAPH_INDEXES:
        return "parametric"
    if is_authoritative_inventory_query_type(normalized):
        return "conversation"
    return "parametric"


def inventory_query_requests_grounded_analysis(query: str, *, query_type: str | None = None) -> bool:
    resolved_type = str(query_type or classify_inventory_query(query) or "").strip()
    if not is_authoritative_inventory_query_type(resolved_type):
        return False
    return bool(_INVENTORY_ANALYSIS_HINTS.search(str(query or "")))


def _tenant_id(settings: Any, session: Any) -> str:
    return str(
        getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev"))
        or getattr(settings, "default_tenant_id", "local-dev")
        or "local-dev"
    )


def _record_sort_key(record: Any) -> tuple[str, str]:
    return (
        str(getattr(record, "title", "") or "").casefold(),
        str(getattr(record, "doc_id", "") or ""),
    )


def _record_to_dict(record: Any) -> dict[str, Any]:
    return {
        "doc_id": str(getattr(record, "doc_id", "") or ""),
        "title": str(getattr(record, "title", "") or ""),
        "source_type": str(getattr(record, "source_type", "") or ""),
        "collection_id": str(getattr(record, "collection_id", "") or ""),
        "num_chunks": int(getattr(record, "num_chunks", 0) or 0),
        "file_type": str(getattr(record, "file_type", "") or ""),
        "doc_structure_type": str(getattr(record, "doc_structure_type", "") or ""),
        "source_path": str(getattr(record, "source_path", "") or ""),
        "source_display_path": str(getattr(record, "source_display_path", "") or ""),
    }


def _list_documents(
    stores: Any,
    *,
    tenant_id: str,
    source_type: str,
    collection_id: str,
) -> list[Any]:
    doc_store = getattr(stores, "doc_store", None)
    if doc_store is None or not hasattr(doc_store, "list_documents"):
        return []
    try:
        records = doc_store.list_documents(
            source_type=source_type,
            tenant_id=tenant_id,
            collection_id=collection_id,
        )
    except Exception:
        return []
    return sorted(list(records or []), key=_record_sort_key)


def _dedupe_collection_ids(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized


def _document_collection_summaries(stores: Any, *, tenant_id: str) -> dict[str, dict[str, Any]]:
    doc_store = getattr(stores, "doc_store", None)
    if doc_store is None:
        return {}

    rows: list[Any] = []
    if hasattr(doc_store, "list_collections"):
        try:
            rows = list(doc_store.list_collections(tenant_id=tenant_id) or [])
        except Exception:
            rows = []
    elif hasattr(doc_store, "list_documents"):
        try:
            records = list(
                doc_store.list_documents(
                    source_type="",
                    tenant_id=tenant_id,
                    collection_id="",
                )
                or []
            )
        except Exception:
            records = []
        grouped: dict[str, dict[str, Any]] = {}
        for record in records:
            collection_id = str(getattr(record, "collection_id", "") or "").strip()
            if not collection_id:
                continue
            source_type = str(getattr(record, "source_type", "") or "unknown").strip() or "unknown"
            item = grouped.setdefault(
                collection_id,
                {
                    "collection_id": collection_id,
                    "document_count": 0,
                    "source_type_counts": {},
                },
            )
            item["document_count"] = int(item.get("document_count") or 0) + 1
            counts = dict(item.get("source_type_counts") or {})
            counts[source_type] = int(counts.get(source_type) or 0) + 1
            item["source_type_counts"] = counts
        rows = list(grouped.values())

    summaries: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = dict(row) if isinstance(row, dict) else {}
        collection_id = str(payload.get("collection_id") or "").strip()
        if not collection_id:
            continue
        summaries[collection_id] = {
            "collection_id": collection_id,
            "kb_doc_count": int(payload.get("document_count") or 0),
            "latest_ingested_at": payload.get("latest_ingested_at"),
            "source_type_counts": dict(payload.get("source_type_counts") or {}),
        }
    return summaries


def _json_safe_inventory_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = make_json_compatible(payload)
    return dict(normalized) if isinstance(normalized, dict) else {}


def _summary_is_upload_only(summary: dict[str, Any]) -> bool:
    counts = {
        str(key or "").strip().lower(): int(value or 0)
        for key, value in dict(summary.get("source_type_counts") or {}).items()
        if str(key or "").strip()
    }
    if not counts:
        return False
    positive_types = {key for key, value in counts.items() if value > 0}
    return bool(positive_types) and positive_types <= {"upload"}


def _catalog_collection_ids(stores: Any, *, tenant_id: str) -> list[str]:
    collection_store = getattr(stores, "collection_store", None)
    if collection_store is None or not hasattr(collection_store, "list_collections"):
        return []
    try:
        records = list(collection_store.list_collections(tenant_id=tenant_id) or [])
    except Exception:
        return []
    return [
        str(getattr(record, "collection_id", "") or "").strip()
        for record in records
        if str(getattr(record, "collection_id", "") or "").strip()
    ]


def _human_join(values: Sequence[str]) -> str:
    items = [str(value or "").strip() for value in values if str(value or "").strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _record_title(record: Any) -> str:
    return str(getattr(record, "title", "") or "").strip()


def _record_source_path(record: Any) -> str:
    return str(
        getattr(record, "source_display_path", "")
        or getattr(record, "source_path", "")
        or ""
    ).strip()


def _record_topic_text(record: Any, *, collection_id: str = "") -> str:
    parts = [
        str(collection_id or "").strip(),
        _record_title(record),
        _record_source_path(record),
        str(getattr(record, "file_type", "") or "").strip(),
    ]
    return " ".join(part for part in parts if part).casefold()


def _collection_title_samples(records: Sequence[Any], *, limit: int = 3) -> list[str]:
    samples: list[str] = []
    seen: set[str] = set()
    for record in records:
        title = _record_title(record)
        if not title:
            continue
        key = title.casefold()
        if key in seen:
            continue
        seen.add(key)
        samples.append(title)
        if len(samples) >= limit:
            break
    return samples


def _record_topic_labels(record: Any, *, collection_id: str = "") -> set[str]:
    text = _record_topic_text(record, collection_id=collection_id)
    if not text:
        return set()
    labels: set[str] = set()
    if any(token in text for token in ("overview", "solution brief", "capabilities", "catalog")):
        labels.add("product overviews")
    if any(token in text for token in ("pricing", "price", "rate card", "quote")):
        labels.add("pricing")
    if ("vendor" in text or "supplier" in text or "third party" in text or "third-party" in text) and (
        "security" in text or "risk" in text
    ):
        labels.add("vendor security")
    if "vendor security" not in labels and any(
        token in text for token in ("security", "privacy", "compliance", "control")
    ):
        labels.add("security")
    if any(token in text for token in ("integration", "connector", "webhook", "sync")):
        labels.add("integrations")
    if any(token in text for token in ("release note", "release-notes", "changelog", "release_notes")):
        labels.add("release notes")
    if any(token in text for token in ("contract", "agreement", "msa", "dpa", "sow", "addendum", "terms")):
        labels.add("legal contracts")
    if any(token in text for token in ("ai-ops", "ai ops", "llmops", "mlops", "model ops", "standard")):
        labels.add("ai-ops standards")
    if any(token in text for token in ("incident", "playbook", "runbook", "outage", "sev")):
        labels.add("incident playbooks")
    if any(token in text for token in ("architecture", "c4", "topology", "system design")):
        labels.add("architecture")
    if any(token in text for token in ("api", "openapi", "swagger", "endpoint", "sdk", "reference")):
        labels.add("api docs")
    if any(
        token in text
        for token in ("process", "workflow", "guide", "handbook", "procedure", "onboarding", "policy", "checklist", "sop")
    ):
        labels.add("internal process guides")
    return labels


def _collection_summary_topics(records: Sequence[Any], *, collection_id: str = "") -> list[str]:
    counts = {label: 0 for label in _COLLECTION_TOPIC_LABELS}
    for record in records:
        for label in _record_topic_labels(record, collection_id=collection_id):
            counts[label] += 1
    ranked = [
        label
        for label in _COLLECTION_TOPIC_LABELS
        if int(counts.get(label) or 0) > 0
    ]
    return sorted(
        ranked,
        key=lambda label: (-int(counts.get(label) or 0), _COLLECTION_TOPIC_LABELS.index(label)),
    )


def _collection_domain_opening(
    collection_id: str,
    *,
    title_samples: Sequence[str],
    summary_topics: Sequence[str],
) -> str:
    clean_collection_id = str(collection_id or "").strip()
    if clean_collection_id.casefold() == "default":
        return "The primary corporate knowledge base"
    hint_text = _normalize_namespace_text(
        " ".join(
            [
                clean_collection_id,
                *[str(item or "").strip() for item in title_samples],
                *[str(item or "").strip() for item in summary_topics],
            ]
        )
    )
    if any(token in hint_text for token in ("test", "validation", "sandbox", "demo")):
        return "A secondary test collection"
    if "rfp" in hint_text or "request for proposal" in hint_text or "proposal" in hint_text:
        return "A specialized collection for request-for-proposal (RFP) material"
    if "vendor security" in hint_text:
        return "A vendor-security collection"
    if "security" in hint_text:
        return "A security knowledge-base collection"
    if "contract" in hint_text or "legal" in hint_text or "agreement" in hint_text:
        return "A legal-document collection"
    if "architecture" in hint_text:
        return "An architecture reference collection"
    if "api docs" in hint_text or "api" in hint_text:
        return "An API documentation collection"
    return "A knowledge-base collection"


def _collection_summary_text(
    *,
    collection_id: str,
    kb_doc_count: int,
    title_samples: Sequence[str],
    summary_topics: Sequence[str],
    documents_enumerated: bool,
) -> tuple[str, str]:
    opening = _collection_domain_opening(
        collection_id,
        title_samples=title_samples,
        summary_topics=summary_topics,
    )
    if kb_doc_count <= 0:
        return (
            "empty",
            f"{opening}. It is listed as available but currently has no indexed documents shown in the inventory.",
        )
    if not documents_enumerated:
        return (
            "not_enumerated",
            f"{opening}. It is listed as available, but the inventory payload does not enumerate its individual files.",
        )
    suffix = "" if kb_doc_count == 1 else "s"
    if summary_topics:
        return (
            "indexed",
            f"{opening} - {kb_doc_count} indexed document{suffix} covering {_human_join(summary_topics)}.",
        )
    if title_samples:
        return (
            "indexed",
            f"{opening} - {kb_doc_count} indexed document{suffix}. Representative titles include {_human_join(title_samples)}.",
        )
    return ("indexed", f"{opening} - {kb_doc_count} indexed document{suffix}.")


def _graph_source_document_count(stores: Any, *, tenant_id: str, graph_id: str, fallback_doc_ids: Sequence[str]) -> int:
    source_store = getattr(stores, "graph_source_store", None)
    if source_store is not None and hasattr(source_store, "list_sources"):
        try:
            return len(list(source_store.list_sources(graph_id, tenant_id=tenant_id) or []))
        except Exception:
            pass
    return len([str(item) for item in (fallback_doc_ids or []) if str(item).strip()])


def _graph_summary_text(
    *,
    collection_id: str,
    status: str,
    query_ready: bool,
    domain_summary: str,
    source_document_count: int,
) -> tuple[str, str]:
    clean_domain_summary = str(domain_summary or "").strip()
    if clean_domain_summary:
        if clean_domain_summary[-1:] not in {".", "!", "?"}:
            clean_domain_summary += "."
        return ("domain_summary", clean_domain_summary)
    readiness = "query-ready" if query_ready else (str(status or "").strip() or "draft")
    suffix = "" if source_document_count == 1 else "s"
    collection_label = str(collection_id or "unknown").strip() or "unknown"
    return (
        "fallback",
        f"Graph index over {collection_label}, {readiness}, covering {source_document_count} source document{suffix}.",
    )


def _visible_kb_collections(settings: Any, stores: Any, session: Any) -> list[dict[str, Any]]:
    tenant_id = _tenant_id(settings, session)
    metadata = dict(getattr(session, "metadata", {}) or {})
    access_summary = dict(metadata.get("access_summary") or {})
    doc_summaries = _document_collection_summaries(stores, tenant_id=tenant_id)
    fallback_doc_collection_ids = [
        collection_id
        for collection_id, summary in doc_summaries.items()
        if not _summary_is_upload_only(summary)
    ]
    if access_summary_authz_enabled(access_summary):
        visible_ids = _dedupe_collection_ids(
            [
                *resolve_available_kb_collection_ids(settings, session),
                resolve_kb_collection_id(settings, session),
            ]
        )
    else:
        visible_ids = _dedupe_collection_ids(
            [
                *resolve_available_kb_collection_ids(settings, session),
                *[str(item) for item in _catalog_collection_ids(stores, tenant_id=tenant_id) if str(item)],
                *fallback_doc_collection_ids,
                resolve_kb_collection_id(settings, session),
            ]
        )
    if not visible_ids:
        fallback = str(metadata.get("kb_collection_id") or getattr(settings, "default_collection_id", "default") or "default").strip()
        if fallback:
            visible_ids = [fallback]

    payloads: list[dict[str, Any]] = []
    for collection_id in visible_ids:
        records = _list_documents(
            stores,
            tenant_id=tenant_id,
            source_type="",
            collection_id=collection_id,
        )
        summary = dict(doc_summaries.get(collection_id) or {})
        readiness = get_collection_readiness_status(
            settings,
            stores,
            tenant_id=tenant_id,
            collection_id=collection_id,
        )
        payloads.append(
            {
                "collection_id": collection_id,
                "maintenance_policy": str(getattr(readiness, "maintenance_policy", "") or ""),
                "kb_available": bool(getattr(readiness, "ready", False)),
                "kb_doc_count": int(summary.get("kb_doc_count") or len(records)),
                "latest_ingested_at": summary.get("latest_ingested_at"),
                "source_type_counts": dict(summary.get("source_type_counts") or {}),
            }
        )
    return payloads


def _visible_graph_indexes(settings: Any, stores: Any, session: Any) -> list[dict[str, Any]]:
    tenant_id = _tenant_id(settings, session)
    try:
        from agentic_chatbot_next.graph.service import GraphService
    except Exception:
        return []

    try:
        payloads = list(GraphService(settings, stores, session=session).list_indexes(limit=250) or [])
    except Exception:
        return []

    graphs: list[dict[str, Any]] = []
    for item in payloads:
        payload = dict(item) if isinstance(item, dict) else {}
        graph_id = str(payload.get("graph_id") or "").strip()
        if not graph_id:
            continue
        display_name = str(payload.get("display_name") or graph_id).strip() or graph_id
        graphs.append(
            {
                "graph_id": graph_id,
                "display_name": display_name,
                "collection_id": str(payload.get("collection_id") or "").strip(),
                "status": str(payload.get("status") or "draft").strip() or "draft",
                "query_ready": bool(payload.get("query_ready")),
                "domain_summary": str(payload.get("domain_summary") or "").strip(),
                "source_document_count": _graph_source_document_count(
                    stores,
                    tenant_id=tenant_id,
                    graph_id=graph_id,
                    fallback_doc_ids=[str(item) for item in (payload.get("source_doc_ids") or []) if str(item).strip()],
                ),
            }
        )
    return sorted(
        graphs,
        key=lambda item: (
            str(item.get("display_name") or item.get("graph_id") or "").casefold(),
            str(item.get("graph_id") or ""),
        ),
    )


def _namespace_match_payload(
    *,
    kind: str,
    namespace_query: str,
    namespace_id: str,
    display_name: str,
    aliases: Sequence[str],
    collection_id: str = "",
    graph_id: str = "",
    status: str = "",
    query_ready: bool = False,
) -> dict[str, Any] | None:
    query_text = str(namespace_query or "").strip()
    if not query_text:
        return None
    exactness = ""
    score = 0.0
    normalized_query = _normalize_namespace_text(query_text)
    query_tokens = set(_namespace_tokens(query_text))
    if not normalized_query:
        return None
    for raw_alias in aliases:
        alias = str(raw_alias or "").strip()
        if not alias:
            continue
        if alias.casefold() == query_text.casefold():
            exactness = "exact"
            score = max(score, 1.0)
            continue
        normalized_alias = _normalize_namespace_text(alias)
        if not normalized_alias:
            continue
        if normalized_alias == normalized_query:
            if score < 0.96:
                exactness = "normalized_exact"
                score = 0.96
            continue
        if normalized_alias.startswith(normalized_query) or normalized_query.startswith(normalized_alias):
            if score < 0.92:
                exactness = "prefix"
                score = 0.92
            continue
        alias_tokens = set(_namespace_tokens(alias))
        if query_tokens and alias_tokens and query_tokens <= alias_tokens:
            if score < 0.86:
                exactness = "substring"
                score = 0.86
            continue
        if normalized_query in normalized_alias or normalized_alias in normalized_query:
            if score < 0.86:
                exactness = "substring"
                score = 0.86
            continue
        ratio = SequenceMatcher(None, normalized_query, normalized_alias).ratio()
        if ratio >= 0.78:
            fuzzy_score = min(0.84, float(ratio))
            if fuzzy_score > score:
                exactness = "fuzzy"
                score = fuzzy_score
    if score <= 0.0:
        return None
    payload = {
        "kind": kind,
        "namespace_id": namespace_id,
        "display_name": display_name,
        "aliases": [str(item) for item in aliases if str(item).strip()],
        "score": round(score, 4),
        "exactness": exactness or "fuzzy",
    }
    if collection_id:
        payload["collection_id"] = collection_id
    if graph_id:
        payload["graph_id"] = graph_id
    if status:
        payload["status"] = status
    if kind == "graph":
        payload["query_ready"] = bool(query_ready)
    return payload


def build_namespace_search_payload(
    settings: Any,
    stores: Any,
    session: Any,
    *,
    namespace_query: str,
) -> dict[str, Any]:
    query_text = str(namespace_query or "").strip()
    collections = _visible_kb_collections(settings, stores, session)
    graphs = _visible_graph_indexes(settings, stores, session)
    collection_matches: list[dict[str, Any]] = []
    graph_matches: list[dict[str, Any]] = []

    for item in collections:
        collection_id = str(item.get("collection_id") or "").strip()
        if not collection_id:
            continue
        payload = _namespace_match_payload(
            kind="collection",
            namespace_query=query_text,
            namespace_id=collection_id,
            display_name=collection_id,
            aliases=[collection_id],
            collection_id=collection_id,
        )
        if payload is not None:
            collection_matches.append(payload)

    for item in graphs:
        graph_id = str(item.get("graph_id") or "").strip()
        if not graph_id:
            continue
        display_name = str(item.get("display_name") or graph_id).strip() or graph_id
        collection_id = str(item.get("collection_id") or "").strip()
        payload = _namespace_match_payload(
            kind="graph",
            namespace_query=query_text,
            namespace_id=graph_id,
            display_name=display_name,
            aliases=[graph_id, display_name, collection_id],
            graph_id=graph_id,
            collection_id=collection_id,
            status=str(item.get("status") or ""),
            query_ready=bool(item.get("query_ready")),
        )
        if payload is not None:
            graph_matches.append(payload)

    sort_key = lambda item: (
        -float(item.get("score") or 0.0),
        str(item.get("display_name") or item.get("namespace_id") or "").casefold(),
        str(item.get("namespace_id") or ""),
    )
    collection_matches = sorted(collection_matches, key=sort_key)
    graph_matches = sorted(graph_matches, key=sort_key)
    return _json_safe_inventory_payload({
        "view": "namespace_search",
        "namespace_query": query_text,
        "collections": collection_matches,
        "graphs": graph_matches,
        "match_count": len(collection_matches) + len(graph_matches),
    })


def _namespace_pending_candidates(session: Any) -> dict[str, Any]:
    metadata = dict(getattr(session, "metadata", {}) or {})
    pending = dict(metadata.get("pending_namespace_candidates") or {})
    return {
        "namespace_query": str(pending.get("namespace_query") or "").strip(),
        "collections": [dict(item) for item in (pending.get("collections") or []) if isinstance(item, dict)],
        "graphs": [dict(item) for item in (pending.get("graphs") or []) if isinstance(item, dict)],
    }


def _candidate_options(candidates: dict[str, Any]) -> tuple[list[str], list[str]]:
    collection_ids = [
        str(item.get("namespace_id") or item.get("collection_id") or "").strip()
        for item in (candidates.get("collections") or [])
        if str(item.get("namespace_id") or item.get("collection_id") or "").strip()
    ]
    graph_ids = [
        str(item.get("graph_id") or item.get("namespace_id") or "").strip()
        for item in (candidates.get("graphs") or [])
        if str(item.get("graph_id") or item.get("namespace_id") or "").strip()
    ]
    return (_dedupe_collection_ids(collection_ids), _dedupe_collection_ids(graph_ids))


def _extract_namespace_selection_ids(
    query: str,
    *,
    candidates: dict[str, Any],
) -> tuple[list[str], list[str]]:
    text = str(query or "").strip()
    lowered = text.casefold()
    collection_ids, graph_ids = _candidate_options(candidates)
    if not text:
        return ([], [])
    if lowered in {"use all", "all"} or re.search(r"\buse\s+all\b", text, flags=re.I):
        return (list(collection_ids), list(graph_ids))
    if re.search(r"\bcollections?\s+only\b", text, flags=re.I):
        return (list(collection_ids), [])
    if re.search(r"\bgraphs?\s+only\b", text, flags=re.I):
        return ([], list(graph_ids))

    selected_collections: list[str] = []
    selected_graphs: list[str] = []

    for collection_id in collection_ids:
        if re.search(_stable_namespace_boundary_pattern(collection_id), text, flags=re.I):
            selected_collections.append(collection_id)
    for graph_id in graph_ids:
        if re.search(_stable_namespace_boundary_pattern(graph_id), text, flags=re.I):
            selected_graphs.append(graph_id)

    if not selected_collections:
        for item in (candidates.get("collections") or []):
            namespace_id = str(item.get("namespace_id") or item.get("collection_id") or "").strip()
            display_name = str(item.get("display_name") or namespace_id).strip() or namespace_id
            if namespace_id and display_name and display_name.casefold() != namespace_id.casefold():
                if re.search(_stable_namespace_boundary_pattern(display_name), text, flags=re.I):
                    selected_collections.append(namespace_id)
    if not selected_graphs:
        for item in (candidates.get("graphs") or []):
            graph_id = str(item.get("graph_id") or item.get("namespace_id") or "").strip()
            display_name = str(item.get("display_name") or graph_id).strip() or graph_id
            if graph_id and display_name and display_name.casefold() != graph_id.casefold():
                if re.search(_stable_namespace_boundary_pattern(display_name), text, flags=re.I):
                    selected_graphs.append(graph_id)

    return (_dedupe_collection_ids(selected_collections), _dedupe_collection_ids(selected_graphs))


def resolve_namespace_scope(
    settings: Any,
    stores: Any,
    session: Any,
    *,
    query: str = "",
) -> dict[str, Any]:
    raw_query = str(query or "").strip()
    metadata = dict(getattr(session, "metadata", {}) or {})
    pending = dict(metadata.get("pending_clarification") or {})
    pending_reason = str(pending.get("reason") or "").strip().lower()
    pending_candidates = _namespace_pending_candidates(session)

    if pending_reason == NAMESPACE_SCOPE_SELECTION_REASON:
        selected_collection_ids, selected_graph_ids = _extract_namespace_selection_ids(
            raw_query,
            candidates=pending_candidates,
        )
        if selected_collection_ids or selected_graph_ids:
            return {
                "mode": "selected",
                "namespace_query": str(pending_candidates.get("namespace_query") or "").strip(),
                "selected_collection_ids": selected_collection_ids,
                "selected_graph_ids": selected_graph_ids,
                "collections": [dict(item) for item in (pending_candidates.get("collections") or []) if isinstance(item, dict)],
                "graphs": [dict(item) for item in (pending_candidates.get("graphs") or []) if isinstance(item, dict)],
                "selection_from_pending": True,
            }

    namespace_query = extract_requested_namespace_query(raw_query)
    if not namespace_query:
        return {
            "mode": "none",
            "namespace_query": "",
            "selected_collection_ids": [],
            "selected_graph_ids": [],
            "collections": [],
            "graphs": [],
            "selection_from_pending": False,
        }

    candidates = build_namespace_search_payload(
        settings,
        stores,
        session,
        namespace_query=namespace_query,
    )
    collections = [dict(item) for item in (candidates.get("collections") or []) if isinstance(item, dict)]
    graphs = [dict(item) for item in (candidates.get("graphs") or []) if isinstance(item, dict)]
    selected_collection_ids: list[str] = []
    selected_graph_ids: list[str] = []
    explicit_graph_reference = extract_requested_graph_reference(raw_query)

    if _is_namespace_scope_query(raw_query):
        selected_collection_ids, selected_graph_ids = _extract_namespace_selection_ids(
            raw_query,
            candidates=candidates,
        )

    if not selected_collection_ids and not selected_graph_ids:
        if explicit_graph_reference and len(graphs) == 1 and float(graphs[0].get("score") or 0.0) >= 0.92:
            selected_graph_ids = [str(graphs[0].get("graph_id") or graphs[0].get("namespace_id") or "").strip()]
        elif len(collections) == 1 and not graphs and float(collections[0].get("score") or 0.0) >= 0.92:
            selected_collection_ids = [str(collections[0].get("namespace_id") or collections[0].get("collection_id") or "").strip()]
        elif len(graphs) == 1 and not collections and float(graphs[0].get("score") or 0.0) >= 0.92:
            selected_graph_ids = [str(graphs[0].get("graph_id") or graphs[0].get("namespace_id") or "").strip()]

    if selected_collection_ids or selected_graph_ids:
        return {
            "mode": "selected",
            "namespace_query": namespace_query,
            "selected_collection_ids": _dedupe_collection_ids(selected_collection_ids),
            "selected_graph_ids": _dedupe_collection_ids(selected_graph_ids),
            "collections": collections,
            "graphs": graphs,
            "selection_from_pending": False,
        }

    return {
        "mode": "clarify" if collections or graphs else "none",
        "namespace_query": namespace_query,
        "selected_collection_ids": [],
        "selected_graph_ids": [],
        "collections": collections,
        "graphs": graphs,
        "selection_from_pending": False,
    }


def resolve_namespace_selection_from_metadata(
    query: str,
    metadata: dict[str, Any] | None,
) -> dict[str, list[str]]:
    payload = dict(metadata or {})
    pending_candidates = {
        "namespace_query": str(dict(payload.get("pending_namespace_candidates") or {}).get("namespace_query") or "").strip(),
        "collections": [dict(item) for item in (dict(payload.get("pending_namespace_candidates") or {}).get("collections") or []) if isinstance(item, dict)],
        "graphs": [dict(item) for item in (dict(payload.get("pending_namespace_candidates") or {}).get("graphs") or []) if isinstance(item, dict)],
    }
    selected_collection_ids, selected_graph_ids = _extract_namespace_selection_ids(
        query,
        candidates=pending_candidates,
    )
    return {
        "collection_ids": selected_collection_ids,
        "graph_ids": selected_graph_ids,
    }


def match_requested_kb_collection_id(
    query: str,
    collection_ids: Sequence[str],
    *,
    pending_reason: str = "",
) -> str:
    options = _dedupe_collection_ids([str(item) for item in collection_ids if str(item).strip()])
    if not options:
        return ""

    option_map = {item.casefold(): item for item in options}
    raw_query = str(query or "").strip()
    clean_query = _clean_collection_reference(raw_query)
    if clean_query.casefold() in option_map:
        return option_map[clean_query.casefold()]

    requested_collection_id = extract_requested_kb_collection_id(raw_query)
    if requested_collection_id.casefold() in option_map:
        return option_map[requested_collection_id.casefold()]

    for collection_id in options:
        escaped = re.escape(collection_id)
        if re.search(rf'["\'`]{escaped}["\'`]', raw_query, flags=re.I):
            return option_map[collection_id.casefold()]
        if re.search(
            rf"\b(?:use|choose|select|pick|switch(?:\s+to)?|search|look\s+in|within|inside)\s+"
            rf"(?:the\s+)?(?:knowledge\s*base\s+|kb\s+|collection\s+)?{escaped}\b",
            raw_query,
            flags=re.I,
        ):
            return option_map[collection_id.casefold()]

    if str(pending_reason or "").strip().lower() == "kb_collection_selection":
        for collection_id in options:
            if re.search(rf"\b{re.escape(collection_id)}\b", raw_query, flags=re.I):
                return option_map[collection_id.casefold()]
    return ""


def sync_session_kb_collection_state(
    settings: Any,
    stores: Any,
    session: Any,
    *,
    query: str = "",
    requested_collection_id: str = "",
) -> dict[str, Any]:
    visible_collections = _visible_kb_collections(settings, stores, session)
    visible_graphs = _visible_graph_indexes(settings, stores, session)
    visible_ids = [str(item.get("collection_id") or "").strip() for item in visible_collections if str(item.get("collection_id") or "").strip()]
    metadata = dict(getattr(session, "metadata", {}) or {})
    active_collection_id = resolve_kb_collection_id(settings, session)
    confirmed = resolve_kb_collection_confirmed(session)
    pending = dict(metadata.get("pending_clarification") or {})
    raw_query = str(requested_collection_id or query or "").strip()
    pending_reason = str(pending.get("reason") or "").strip()
    should_resolve_namespace = (
        pending_reason == NAMESPACE_SCOPE_SELECTION_REASON
        or _is_namespace_scoped_doc_inventory_query(raw_query)
        or _is_namespace_scope_query(raw_query)
        or _is_graph_scoped_file_inventory_query(raw_query)
    )
    namespace_resolution = (
        resolve_namespace_scope(settings, stores, session, query=raw_query)
        if should_resolve_namespace
        else {
            "mode": "none",
            "namespace_query": "",
            "selected_collection_ids": [],
            "selected_graph_ids": [],
            "collections": [],
            "graphs": [],
            "selection_from_pending": False,
        }
    )
    selected_collection_ids = [
        str(item).strip()
        for item in (namespace_resolution.get("selected_collection_ids") or [])
        if str(item).strip()
    ]
    selected_graph_ids = [
        str(item).strip()
        for item in (namespace_resolution.get("selected_graph_ids") or [])
        if str(item).strip()
    ]
    selected_collection_id = selected_collection_ids[0] if selected_collection_ids else ""
    if not selected_collection_id and str(namespace_resolution.get("mode") or "") != "clarify":
        selected_collection_id = match_requested_kb_collection_id(
            raw_query,
            visible_ids,
            pending_reason=pending_reason,
        )
    if selected_collection_id and not selected_collection_ids:
        selected_collection_ids = [selected_collection_id]
    if selected_collection_id:
        active_collection_id = selected_collection_id
        confirmed = True
    elif active_collection_id not in visible_ids and visible_ids:
        default_collection_id = str(getattr(settings, "default_collection_id", "default") or "default").strip() or "default"
        active_collection_id = default_collection_id if default_collection_id in visible_ids else visible_ids[0]
        confirmed = False

    if not active_collection_id:
        active_collection_id = str(getattr(settings, "default_collection_id", "default") or "default").strip() or "default"
    if active_collection_id and active_collection_id not in visible_ids:
        visible_ids = _dedupe_collection_ids([*visible_ids, active_collection_id])

    patch: dict[str, Any] = {
        "kb_collection_id": active_collection_id,
        "available_kb_collection_ids": list(visible_ids),
        "available_graph_indexes": [dict(item) for item in visible_graphs],
        "kb_collection_confirmed": confirmed,
    }
    next_metadata = {**metadata, **patch}
    if selected_collection_ids:
        patch["requested_kb_collection_id"] = selected_collection_id
        patch["search_collection_ids"] = list(selected_collection_ids)
    if selected_graph_ids:
        patch["active_graph_ids"] = list(selected_graph_ids)
    next_metadata = {**metadata, **patch}
    if str(namespace_resolution.get("mode") or "") == "clarify":
        next_metadata["pending_namespace_candidates"] = {
            "namespace_query": str(namespace_resolution.get("namespace_query") or "").strip(),
            "collections": [dict(item) for item in (namespace_resolution.get("collections") or []) if isinstance(item, dict)],
            "graphs": [dict(item) for item in (namespace_resolution.get("graphs") or []) if isinstance(item, dict)],
        }
    elif selected_collection_ids or selected_graph_ids:
        next_metadata.pop("pending_namespace_candidates", None)
    session.metadata = next_metadata
    return {
        "kb_collection_id": active_collection_id,
        "available_kb_collection_ids": list(visible_ids),
        "kb_collection_confirmed": confirmed,
        "selected_kb_collection_id": selected_collection_id,
        "selected_kb_collection_ids": list(selected_collection_ids),
        "selected_graph_ids": list(selected_graph_ids),
        "namespace_mode": str(namespace_resolution.get("mode") or "none"),
        "namespace_query": str(namespace_resolution.get("namespace_query") or "").strip(),
        "namespace_collections": [dict(item) for item in (namespace_resolution.get("collections") or []) if isinstance(item, dict)],
        "namespace_graphs": [dict(item) for item in (namespace_resolution.get("graphs") or []) if isinstance(item, dict)],
        "collections": visible_collections,
        "graphs": visible_graphs,
    }


def _kb_available(settings: Any, stores: Any, *, tenant_id: str, collection_id: str, kb_records: list[Any]) -> bool:
    try:
        status = get_collection_readiness_status(
            settings,
            stores,
            tenant_id=tenant_id,
            collection_id=collection_id,
        )
    except Exception:
        return bool(kb_records)
    return bool(getattr(status, "ready", False))


def build_session_access_inventory_payload(settings: Any, stores: Any, session: Any) -> dict[str, Any]:
    tenant_id = _tenant_id(settings, session)
    kb_scope = sync_session_kb_collection_state(settings, stores, session)
    kb_collection_id = str(kb_scope.get("kb_collection_id") or resolve_kb_collection_id(settings, session))
    upload_collection_id = resolve_upload_collection_id(settings, session)
    kb_records = _list_documents(
        stores,
        tenant_id=tenant_id,
        source_type="",
        collection_id=kb_collection_id,
    )
    upload_records = _list_documents(
        stores,
        tenant_id=tenant_id,
        source_type="upload",
        collection_id=upload_collection_id,
    )
    return _json_safe_inventory_payload({
        "view": "session_access",
        "kb_collection_id": kb_collection_id,
        "available_kb_collection_ids": list(kb_scope.get("available_kb_collection_ids") or []),
        "kb_collections": [dict(item) for item in (kb_scope.get("collections") or []) if isinstance(item, dict)],
        "kb_collection_confirmed": bool(kb_scope.get("kb_collection_confirmed")),
        "kb_available": _kb_available(
            settings,
            stores,
            tenant_id=tenant_id,
            collection_id=kb_collection_id,
            kb_records=kb_records,
        ),
        "kb_doc_count": len(kb_records),
        "upload_collection_id": upload_collection_id,
        "has_uploads": bool(upload_records or has_upload_evidence(session)),
        "uploaded_documents": [_record_to_dict(record) for record in upload_records],
        "next_actions": list(SESSION_ACCESS_NEXT_ACTIONS),
    })


def build_kb_file_inventory_payload(
    settings: Any,
    stores: Any,
    session: Any,
    *,
    records: list[Any] | None = None,
    collection_id: str = "",
) -> dict[str, Any]:
    tenant_id = _tenant_id(settings, session)
    kb_scope = sync_session_kb_collection_state(
        settings,
        stores,
        session,
        requested_collection_id=collection_id,
    )
    visible_collection_ids = list(kb_scope.get("available_kb_collection_ids") or [])
    session_kb_collection_id = str(kb_scope.get("kb_collection_id") or resolve_kb_collection_id(settings, session))
    requested_collection_id = str(collection_id or session_kb_collection_id).strip() or session_kb_collection_id
    requested_collection_available = requested_collection_id in set(visible_collection_ids)
    kb_records = (
        sorted(list(records or []), key=_record_sort_key)
        if records is not None
        else (
            _list_documents(
                stores,
                tenant_id=tenant_id,
                source_type="",
                collection_id=requested_collection_id,
            )
            if requested_collection_available
            else []
        )
    )
    return _json_safe_inventory_payload({
        "view": "kb_file_inventory",
        "kb_collection_id": requested_collection_id,
        "requested_collection_id": requested_collection_id,
        "requested_collection_available": requested_collection_available,
        "session_kb_collection_id": session_kb_collection_id,
        "available_kb_collection_ids": visible_collection_ids,
        "kb_available": (
            _kb_available(
                settings,
                stores,
                tenant_id=tenant_id,
                collection_id=requested_collection_id,
                kb_records=kb_records,
            )
            if requested_collection_available
            else False
        ),
        "kb_doc_count": len(kb_records),
        "documents": [_record_to_dict(record) for record in kb_records],
    })


def _graph_source_sort_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("title") or "").casefold(),
        str(record.get("doc_id") or ""),
        str(record.get("source_path") or ""),
    )


def _graph_source_to_dict(record: Any) -> dict[str, Any]:
    payload = dict(record) if isinstance(record, dict) else {}
    source_path = str(payload.get("source_path") or getattr(record, "source_path", "") or "").strip()
    doc_id = str(payload.get("doc_id") or payload.get("source_doc_id") or getattr(record, "source_doc_id", "") or "").strip()
    title = str(payload.get("title") or payload.get("source_title") or getattr(record, "source_title", "") or "").strip()
    if not title:
        title = str(
            doc_id
            or source_path.rsplit("/", 1)[-1]
            or "Untitled document"
        ).strip()
    return {
        "doc_id": doc_id,
        "title": title,
        "source_type": str(payload.get("source_type") or getattr(record, "source_type", "") or "").strip(),
        "source_path": source_path,
    }


def build_graph_document_inventory_payload(
    settings: Any,
    stores: Any,
    session: Any,
    *,
    graph_id: str,
) -> dict[str, Any]:
    requested_graph_id = str(graph_id or "").strip()
    tenant_id = _tenant_id(settings, session)
    try:
        from agentic_chatbot_next.graph.service import GraphService
    except Exception:
        return _json_safe_inventory_payload({
            "view": "graph_file_inventory",
            "requested_graph_id": requested_graph_id,
            "requested_graph_available": False,
            "documents": [],
        })

    try:
        payload = GraphService(settings, stores, session=session).inspect_index(requested_graph_id)
    except Exception:
        payload = {}

    graph_payload = dict(payload.get("graph") or {})
    raw_sources = [dict(item) for item in (payload.get("sources") or []) if isinstance(item, dict)]

    if not graph_payload:
        record = None
        store = getattr(stores, "graph_index_store", None)
        if store is not None and hasattr(store, "get_index"):
            try:
                record = store.get_index(requested_graph_id, tenant_id, user_id=str(getattr(session, "user_id", "") or ""))
            except TypeError:
                try:
                    record = store.get_index(requested_graph_id, tenant_id)
                except TypeError:
                    record = store.get_index(requested_graph_id)
        if record is None and store is not None and hasattr(store, "list_indexes"):
            try:
                records = list(store.list_indexes(tenant_id=tenant_id, limit=250) or [])
            except TypeError:
                records = list(store.list_indexes(limit=250) or [])
            for item in records:
                if str(getattr(item, "display_name", "") or "").strip().casefold() == requested_graph_id.casefold():
                    record = item
                    break
        if record is not None:
            graph_payload = {
                "graph_id": str(getattr(record, "graph_id", "") or ""),
                "display_name": str(getattr(record, "display_name", "") or ""),
                "collection_id": str(getattr(record, "collection_id", "") or ""),
                "status": str(getattr(record, "status", "") or ""),
                "query_ready": bool(getattr(record, "query_ready", False)),
            }
            source_store = getattr(stores, "graph_source_store", None)
            if source_store is not None and hasattr(source_store, "list_sources"):
                raw_sources = [
                    _graph_source_to_dict(item)
                    for item in (source_store.list_sources(graph_payload.get("graph_id") or requested_graph_id, tenant_id=tenant_id) or [])
                ]

    if not graph_payload or str(payload.get("error") or "").strip():
        return _json_safe_inventory_payload({
            "view": "graph_file_inventory",
            "requested_graph_id": requested_graph_id,
            "requested_graph_available": False,
            "documents": [],
        })
    raw_sources = [_graph_source_to_dict(item) for item in raw_sources]
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in sorted(raw_sources, key=_graph_source_sort_key):
        key = (
            str(item.get("doc_id") or "").strip(),
            str(item.get("source_path") or "").strip(),
            str(item.get("title") or "").strip().casefold(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    resolved_graph_id = str(graph_payload.get("graph_id") or requested_graph_id).strip() or requested_graph_id
    return _json_safe_inventory_payload({
        "view": "graph_file_inventory",
        "requested_graph_id": requested_graph_id,
        "requested_graph_available": True,
        "graph_id": resolved_graph_id,
        "display_name": str(graph_payload.get("display_name") or resolved_graph_id).strip() or resolved_graph_id,
        "collection_id": str(graph_payload.get("collection_id") or "").strip(),
        "status": str(graph_payload.get("status") or "draft").strip() or "draft",
        "query_ready": bool(graph_payload.get("query_ready")),
        "documents": deduped,
    })


def build_namespace_document_inventory_payload(
    settings: Any,
    stores: Any,
    session: Any,
    *,
    query: str,
) -> dict[str, Any]:
    kb_scope = sync_session_kb_collection_state(settings, stores, session, query=query)
    namespace_mode = str(kb_scope.get("namespace_mode") or "none")
    namespace_query = str(kb_scope.get("namespace_query") or "").strip()
    selected_collection_ids = [
        str(item).strip()
        for item in (kb_scope.get("selected_kb_collection_ids") or [])
        if str(item).strip()
    ]
    selected_graph_ids = [
        str(item).strip()
        for item in (kb_scope.get("selected_graph_ids") or [])
        if str(item).strip()
    ]
    explicit_graph_ref = extract_requested_graph_reference(query)

    if namespace_mode == "clarify":
        return _json_safe_inventory_payload({
            "view": "namespace_clarification",
            "namespace_query": namespace_query,
            "collections": [dict(item) for item in (kb_scope.get("namespace_collections") or []) if isinstance(item, dict)],
            "graphs": [dict(item) for item in (kb_scope.get("namespace_graphs") or []) if isinstance(item, dict)],
        })

    if explicit_graph_ref and not selected_graph_ids:
        return build_graph_document_inventory_payload(
            settings,
            stores,
            session,
            graph_id=explicit_graph_ref,
        )

    collection_payloads = [
        build_kb_file_inventory_payload(settings, stores, session, collection_id=collection_id)
        for collection_id in selected_collection_ids
    ]
    graph_payloads = [
        build_graph_document_inventory_payload(settings, stores, session, graph_id=graph_id)
        for graph_id in selected_graph_ids
    ]
    if collection_payloads or graph_payloads:
        if len(collection_payloads) == 1 and not graph_payloads:
            return collection_payloads[0]
        if len(graph_payloads) == 1 and not collection_payloads:
            return graph_payloads[0]
        return _json_safe_inventory_payload({
            "view": "namespace_combined_inventory",
            "namespace_query": namespace_query,
            "collections": collection_payloads,
            "graphs": graph_payloads,
        })

    requested_collection_id = extract_requested_kb_collection_id(query)
    if requested_collection_id:
        return build_kb_file_inventory_payload(
            settings,
            stores,
            session,
            collection_id=requested_collection_id,
        )
    if namespace_query:
        return _json_safe_inventory_payload({
            "view": "namespace_not_found",
            "namespace_query": namespace_query,
            "collections": [dict(item) for item in (kb_scope.get("namespace_collections") or []) if isinstance(item, dict)],
            "graphs": [dict(item) for item in (kb_scope.get("namespace_graphs") or []) if isinstance(item, dict)],
        })
    return build_kb_file_inventory_payload(settings, stores, session)


def build_kb_collection_access_payload(settings: Any, stores: Any, session: Any) -> dict[str, Any]:
    kb_scope = sync_session_kb_collection_state(settings, stores, session)
    tenant_id = _tenant_id(settings, session)
    graphs = _visible_graph_indexes(settings, stores, session)
    graph_counts_by_collection: dict[str, int] = {}
    for item in graphs:
        collection_id = str(item.get("collection_id") or "").strip()
        if not collection_id:
            continue
        graph_counts_by_collection[collection_id] = int(graph_counts_by_collection.get(collection_id) or 0) + 1

    collections: list[dict[str, Any]] = []
    for item in (kb_scope.get("collections") or []):
        if not isinstance(item, dict):
            continue
        payload = dict(item)
        collection_id = str(payload.get("collection_id") or "").strip()
        records = (
            _list_documents(
                stores,
                tenant_id=tenant_id,
                source_type="",
                collection_id=collection_id,
            )
            if collection_id
            else []
        )
        title_samples = _collection_title_samples(records)
        kb_doc_count = int(payload.get("kb_doc_count") or len(records))
        documents_enumerated = kb_doc_count <= 0 or bool(records)
        summary_topics = _collection_summary_topics(records, collection_id=collection_id)
        summary_mode, summary_text = _collection_summary_text(
            collection_id=collection_id,
            kb_doc_count=kb_doc_count,
            title_samples=title_samples,
            summary_topics=summary_topics,
            documents_enumerated=documents_enumerated,
        )
        payload.update(
            {
                "kb_doc_count": kb_doc_count,
                "graph_count": int(graph_counts_by_collection.get(collection_id) or 0),
                "title_samples": title_samples,
                "summary_topics": summary_topics,
                "summary_mode": summary_mode,
                "summary": summary_text,
            }
        )
        collections.append(payload)

    enriched_graphs: list[dict[str, Any]] = []
    for item in graphs:
        payload = dict(item)
        summary_mode, summary_text = _graph_summary_text(
            collection_id=str(payload.get("collection_id") or "").strip(),
            status=str(payload.get("status") or "draft").strip() or "draft",
            query_ready=bool(payload.get("query_ready")),
            domain_summary=str(payload.get("domain_summary") or "").strip(),
            source_document_count=int(payload.get("source_document_count") or 0),
        )
        payload.update(
            {
                "summary_mode": summary_mode,
                "summary": summary_text,
            }
        )
        enriched_graphs.append(payload)

    return _json_safe_inventory_payload({
        "view": "kb_collections",
        "kb_collection_id": str(kb_scope.get("kb_collection_id") or resolve_kb_collection_id(settings, session)),
        "kb_collection_confirmed": bool(kb_scope.get("kb_collection_confirmed")),
        "collections": sorted(
            collections,
            key=lambda item: str(item.get("collection_id") or "").casefold(),
        ),
        "graphs": sorted(
            enriched_graphs,
            key=lambda item: (
                str(item.get("display_name") or item.get("graph_id") or "").casefold(),
                str(item.get("graph_id") or ""),
            ),
        ),
    })


def build_graph_index_access_payload(settings: Any, stores: Any, session: Any) -> dict[str, Any]:
    kb_scope = sync_session_kb_collection_state(settings, stores, session)
    graphs = _visible_graph_indexes(settings, stores, session)

    enriched_graphs: list[dict[str, Any]] = []
    for item in graphs:
        payload = dict(item)
        summary_mode, summary_text = _graph_summary_text(
            collection_id=str(payload.get("collection_id") or "").strip(),
            status=str(payload.get("status") or "draft").strip() or "draft",
            query_ready=bool(payload.get("query_ready")),
            domain_summary=str(payload.get("domain_summary") or "").strip(),
            source_document_count=int(payload.get("source_document_count") or 0),
        )
        payload.update(
            {
                "summary_mode": summary_mode,
                "summary": summary_text,
            }
        )
        enriched_graphs.append(payload)

    return _json_safe_inventory_payload({
        "view": "graph_indexes",
        "kb_collection_id": str(kb_scope.get("kb_collection_id") or resolve_kb_collection_id(settings, session)),
        "kb_collection_confirmed": bool(kb_scope.get("kb_collection_confirmed")),
        "graphs": sorted(
            enriched_graphs,
            key=lambda item: (
                str(item.get("display_name") or item.get("graph_id") or "").casefold(),
                str(item.get("graph_id") or ""),
            ),
        ),
    })


def build_session_access_inventory_answer(payload: dict[str, Any]) -> dict[str, Any]:
    kb_collection_id = str(payload.get("kb_collection_id") or "default")
    kb_doc_count = int(payload.get("kb_doc_count") or 0)
    kb_available = bool(payload.get("kb_available"))
    kb_collections = [dict(item) for item in (payload.get("kb_collections") or []) if isinstance(item, dict)]
    uploaded_documents = [dict(item) for item in (payload.get("uploaded_documents") or []) if isinstance(item, dict)]
    has_uploads = bool(payload.get("has_uploads"))

    lines = ["This chat can currently use:"]
    if len(kb_collections) > 1:
        collection_descriptions: list[str] = []
        for item in kb_collections:
            collection_id = str(item.get("collection_id") or "default")
            count = int(item.get("kb_doc_count") or 0)
            suffix = "indexed document" if count == 1 else "indexed documents"
            if bool(item.get("kb_available")):
                collection_descriptions.append(f"{collection_id} ({count} {suffix})")
            else:
                collection_descriptions.append(f"{collection_id} (not indexed yet)")
        lines.append("- Knowledge base collections: " + ", ".join(collection_descriptions))
        lines.append(f"- Active KB collection for this chat: {kb_collection_id}")
    elif kb_available:
        lines.append(
            f"- Knowledge base collection: {kb_collection_id} ({kb_doc_count} indexed document"
            f"{'' if kb_doc_count == 1 else 's'})"
        )
    else:
        lines.append(f"- Knowledge base collection: {kb_collection_id} (not indexed yet)")

    if uploaded_documents:
        lines.append(f"- Current chat uploads ({len(uploaded_documents)} total):")
        for document in uploaded_documents[:10]:
            title = str(document.get("title") or document.get("doc_id") or "Untitled document")
            details = [f"doc_id={document.get('doc_id', '')}"]
            file_type = str(document.get("file_type") or "").strip()
            if file_type:
                details.append(f"file_type={file_type}")
            num_chunks = int(document.get("num_chunks", 0) or 0)
            if num_chunks > 0:
                details.append(f"chunks={num_chunks}")
            lines.append(f"  - {title} ({'; '.join(details)})")
        remaining = len(uploaded_documents) - 10
        if remaining > 0:
            lines.append(f"  - ... and {remaining} more upload(s)")
    elif has_uploads:
        lines.append("- Current chat uploads are available, but I could not enumerate their metadata.")
    else:
        lines.append("- Current chat uploads: none")

    lines.append(
        "Tell me what you want next: `search uploaded docs`, `search the KB`, `search both`, or `list KB files`."
    )
    return {
        "answer": "\n".join(lines),
        "followups": list(SESSION_ACCESS_NEXT_ACTIONS),
        "warnings": [],
        "confidence_hint": 0.95 if kb_available or uploaded_documents or has_uploads else 0.7,
    }


def build_kb_file_inventory_answer(payload: dict[str, Any]) -> dict[str, Any]:
    requested_collection_id = str(payload.get("requested_collection_id") or payload.get("kb_collection_id") or "default")
    session_kb_collection_id = str(payload.get("session_kb_collection_id") or requested_collection_id or "default")
    if not bool(payload.get("requested_collection_available", True)):
        available_collections = [
            str(item).strip()
            for item in (payload.get("available_kb_collection_ids") or [])
            if str(item).strip()
        ]
        availability_text = (
            "Available KB collections: " + ", ".join(f"`{item}`" for item in available_collections)
            if available_collections
            else f"Active KB collection: `{session_kb_collection_id}`."
        )
        return {
            "answer": (
                f"Requested KB collection `{requested_collection_id}` is not available to this chat.\n"
                f"{availability_text}"
            ),
            "followups": [],
            "warnings": ["KB_COLLECTION_NOT_AVAILABLE"],
            "confidence_hint": 0.95,
        }
    kb_collection_id = str(payload.get("kb_collection_id") or requested_collection_id or "default")
    documents = [dict(item) for item in (payload.get("documents") or []) if isinstance(item, dict)]
    lines = [
        f"Knowledge base documents currently indexed in collection `{kb_collection_id}` ({len(documents)} total):"
    ]
    for document in documents:
        title = str(document.get("title") or document.get("doc_id") or "Untitled document")
        details = [f"doc_id={document.get('doc_id', '')}"]
        file_type = str(document.get("file_type") or "").strip()
        if file_type:
            details.append(f"file_type={file_type}")
        source_type = str(document.get("source_type") or "").strip()
        if source_type and source_type.casefold() != "kb":
            details.append(f"source_type={source_type}")
        num_chunks = int(document.get("num_chunks", 0) or 0)
        if num_chunks > 0:
            details.append(f"chunks={num_chunks}")
        source_path = str(document.get("source_display_path") or document.get("source_path") or "").strip()
        if source_path:
            details.append(f"path={source_path}")
        lines.append(f"- {title} ({'; '.join(details)})")
    if len(lines) == 1:
        lines.append("- No knowledge-base documents are currently indexed for this collection.")
    return {
        "answer": "\n".join(lines),
        "followups": [],
        "warnings": [],
        "confidence_hint": 0.95 if documents else 0.6,
    }


def _namespace_followup_options(payload: dict[str, Any]) -> list[str]:
    collection_ids = [
        str(item.get("namespace_id") or item.get("collection_id") or "").strip()
        for item in (payload.get("collections") or [])
        if str(item.get("namespace_id") or item.get("collection_id") or "").strip()
    ]
    graph_ids = [
        str(item.get("graph_id") or item.get("namespace_id") or "").strip()
        for item in (payload.get("graphs") or [])
        if str(item.get("graph_id") or item.get("namespace_id") or "").strip()
    ]
    followups = [*collection_ids, *graph_ids]
    if collection_ids:
        followups.append("collections only")
    if graph_ids:
        followups.append("graphs only")
    if collection_ids or graph_ids:
        followups.append("use all")
    return _dedupe_collection_ids(followups)


def build_graph_document_inventory_answer(payload: dict[str, Any]) -> dict[str, Any]:
    requested_graph_id = str(payload.get("requested_graph_id") or payload.get("graph_id") or "").strip()
    if not bool(payload.get("requested_graph_available", True)):
        return {
            "answer": f"Requested graph `{requested_graph_id}` is not available to this chat.",
            "followups": [],
            "warnings": ["GRAPH_NOT_AVAILABLE"],
            "confidence_hint": 0.95,
        }
    graph_id = str(payload.get("graph_id") or requested_graph_id or "").strip()
    display_name = str(payload.get("display_name") or graph_id).strip() or graph_id
    collection_id = str(payload.get("collection_id") or "").strip()
    readiness = "query-ready" if bool(payload.get("query_ready")) else str(payload.get("status") or "draft").strip() or "draft"
    documents = [dict(item) for item in (payload.get("documents") or []) if isinstance(item, dict)]
    graph_title = display_name if display_name.casefold() == graph_id.casefold() else f"{display_name} (`{graph_id}`)"
    suffix_parts: list[str] = []
    if collection_id:
        suffix_parts.append(f"collection: {collection_id}")
    if readiness:
        suffix_parts.append(readiness)
    suffix = f" [{'; '.join(suffix_parts)}]" if suffix_parts else ""
    lines = [
        f"Knowledge graph source documents currently indexed in graph {graph_title}{suffix} ({len(documents)} total):"
    ]
    for document in documents:
        title = str(document.get("title") or document.get("doc_id") or "Untitled document")
        details = [f"doc_id={document.get('doc_id', '')}"] if str(document.get("doc_id") or "").strip() else []
        source_type = str(document.get("source_type") or "").strip()
        if source_type:
            details.append(f"source_type={source_type}")
        source_path = str(document.get("source_path") or "").strip()
        if source_path:
            details.append(f"path={source_path}")
        lines.append(f"- {title}" + (f" ({'; '.join(details)})" if details else ""))
    if len(lines) == 1:
        lines.append("- No graph source documents are currently recorded for this graph.")
    return {
        "answer": "\n".join(lines),
        "followups": [],
        "warnings": [],
        "confidence_hint": 0.95 if documents else 0.6,
    }


def build_namespace_document_inventory_answer(payload: dict[str, Any]) -> dict[str, Any]:
    view = str(payload.get("view") or "").strip()
    if view == "namespace_clarification":
        namespace_query = str(payload.get("namespace_query") or "").strip()
        collections = [dict(item) for item in (payload.get("collections") or []) if isinstance(item, dict)]
        graphs = [dict(item) for item in (payload.get("graphs") or []) if isinstance(item, dict)]
        lines = [
            f"I found multiple visible namespaces matching `{namespace_query}`. Tell me which ones to use."
        ]
        if collections:
            lines.append("Collections:")
            for item in collections:
                collection_id = str(item.get("namespace_id") or item.get("collection_id") or "").strip()
                lines.append(
                    f"- `{collection_id}` (score={float(item.get('score') or 0.0):.2f}; {str(item.get('exactness') or 'match')})"
                )
        if graphs:
            lines.append("Graphs:")
            for item in graphs:
                graph_id = str(item.get("graph_id") or item.get("namespace_id") or "").strip()
                display_name = str(item.get("display_name") or graph_id).strip() or graph_id
                collection_id = str(item.get("collection_id") or "").strip()
                title = display_name if display_name.casefold() == graph_id.casefold() else f"{display_name} (`{graph_id}`)"
                suffix = f"; collection={collection_id}" if collection_id else ""
                lines.append(
                    f"- {title} (score={float(item.get('score') or 0.0):.2f}; {str(item.get('exactness') or 'match')}{suffix})"
                )
        lines.append("Reply with one or more ids, `collections only`, `graphs only`, or `use all`.")
        return {
            "answer": "\n".join(lines),
            "followups": _namespace_followup_options(payload),
            "warnings": ["NAMESPACE_SCOPE_SELECTION_REQUIRED"],
            "confidence_hint": 0.8,
        }
    if view == "namespace_combined_inventory":
        lines: list[str] = []
        for collection_payload in (payload.get("collections") or []):
            answer = build_kb_file_inventory_answer(dict(collection_payload))
            lines.append(str(answer.get("answer") or "").strip())
        for graph_payload in (payload.get("graphs") or []):
            answer = build_graph_document_inventory_answer(dict(graph_payload))
            lines.append(str(answer.get("answer") or "").strip())
        return {
            "answer": "\n\n".join(line for line in lines if line),
            "followups": [],
            "warnings": [],
            "confidence_hint": 0.95,
        }
    if view == "graph_file_inventory":
        return build_graph_document_inventory_answer(payload)
    if view == "namespace_not_found":
        namespace_query = str(payload.get("namespace_query") or "").strip()
        return {
            "answer": f"No visible knowledge-base collection or graph matched `{namespace_query}`.",
            "followups": [],
            "warnings": ["NAMESPACE_NOT_FOUND"],
            "confidence_hint": 0.85,
        }
    return build_kb_file_inventory_answer(payload)


def build_kb_collection_access_answer(payload: dict[str, Any]) -> dict[str, Any]:
    collections = [dict(item) for item in (payload.get("collections") or []) if isinstance(item, dict)]
    graphs = [dict(item) for item in (payload.get("graphs") or []) if isinstance(item, dict)]
    lines = ["Knowledge base collections available to this chat:"]
    for item in collections:
        collection_id = str(item.get("collection_id") or "default")
        lines.append(collection_id)
        lines.append(str(item.get("summary") or "").strip() or "A knowledge-base collection.")
        lines.append("")
    if len(lines) == 1:
        lines.extend(["default", "The primary corporate knowledge base.", ""])

    lines.append("Knowledge graphs available to this chat:")
    if graphs:
        for item in graphs:
            graph_id = str(item.get("graph_id") or "").strip()
            display_name = str(item.get("display_name") or graph_id).strip() or graph_id
            title = display_name if display_name.casefold() == graph_id.casefold() else f"{display_name} (`{graph_id}`)"
            lines.append(title)
            lines.append(str(item.get("summary") or "").strip() or "Graph index metadata is available.")
            lines.append("")
    else:
        lines.append("- none")

    if collections or graphs:
        if lines and not lines[-1]:
            lines.pop()
        lines.append("Reply with one or more ids or `use all` if you want me to scope later searches to specific namespaces.")

    return {
        "answer": "\n".join(lines),
        "followups": _namespace_followup_options(payload),
        "warnings": [],
        "confidence_hint": 0.95 if collections or graphs else 0.7,
    }


def build_graph_index_access_answer(payload: dict[str, Any]) -> dict[str, Any]:
    graphs = [dict(item) for item in (payload.get("graphs") or []) if isinstance(item, dict)]
    lines = ["Knowledge graphs available to this chat:"]
    followups: list[str] = []
    if graphs:
        for item in graphs:
            graph_id = str(item.get("graph_id") or "").strip()
            display_name = str(item.get("display_name") or graph_id).strip() or graph_id
            title = display_name if display_name.casefold() == graph_id.casefold() else f"{display_name} (`{graph_id}`)"
            lines.append(title)
            lines.append(str(item.get("summary") or "").strip() or "Graph index metadata is available.")
            lines.append("")
            if graph_id:
                followups.append(graph_id)
    else:
        lines.append("- none")

    if followups:
        if lines and not lines[-1]:
            lines.pop()
        lines.append("Reply with a graph id if you want me to inspect one in more detail.")

    return {
        "answer": "\n".join(lines),
        "followups": _dedupe_collection_ids(followups),
        "warnings": [],
        "confidence_hint": 0.95 if graphs else 0.7,
    }


def dispatch_authoritative_inventory(
    settings: Any,
    stores: Any,
    session: Any,
    *,
    query: str = "",
    query_type: str = "",
) -> dict[str, Any]:
    resolved_query_type = str(query_type or classify_inventory_query(query) or "").strip()
    if not is_authoritative_inventory_query_type(resolved_query_type):
        return {
            "handled": False,
            "query_type": resolved_query_type,
            "view": "",
            "payload": {},
            "answer": {},
            "provenance": "",
        }

    if resolved_query_type == INVENTORY_QUERY_SESSION_ACCESS:
        payload = build_session_access_inventory_payload(settings, stores, session)
        answer = build_session_access_inventory_answer(payload)
    elif resolved_query_type == INVENTORY_QUERY_KB_COLLECTIONS:
        payload = build_kb_collection_access_payload(settings, stores, session)
        answer = build_kb_collection_access_answer(payload)
    elif resolved_query_type == INVENTORY_QUERY_GRAPH_INDEXES:
        payload = build_graph_index_access_payload(settings, stores, session)
        answer = build_graph_index_access_answer(payload)
    elif resolved_query_type in {INVENTORY_QUERY_KB_FILE, INVENTORY_QUERY_GRAPH_FILE}:
        payload = (
            build_namespace_document_inventory_payload(settings, stores, session, query=query)
            if str(query or "").strip()
            else build_kb_file_inventory_payload(settings, stores, session)
        )
        answer = build_namespace_document_inventory_answer(payload)
    else:
        return {
            "handled": False,
            "query_type": resolved_query_type,
            "view": "",
            "payload": {},
            "answer": {},
            "provenance": "",
        }

    return _json_safe_inventory_payload({
        "handled": True,
        "query_type": resolved_query_type,
        "view": str(payload.get("view") or "").strip(),
        "payload": make_json_compatible(payload),
        "answer": make_json_compatible(answer),
        "provenance": "authoritative_inventory",
    })


__all__ = [
    "INVENTORY_QUERY_GRAPH_FILE",
    "INVENTORY_QUERY_GRAPH_INDEXES",
    "INVENTORY_QUERY_KB_FILE",
    "INVENTORY_QUERY_KB_COLLECTIONS",
    "NAMESPACE_SCOPE_SELECTION_REASON",
    "INVENTORY_QUERY_NONE",
    "INVENTORY_QUERY_SESSION_ACCESS",
    "SESSION_ACCESS_NEXT_ACTIONS",
    "build_graph_document_inventory_answer",
    "build_graph_document_inventory_payload",
    "build_graph_index_access_answer",
    "build_graph_index_access_payload",
    "build_kb_collection_access_answer",
    "build_kb_collection_access_payload",
    "build_kb_file_inventory_answer",
    "build_kb_file_inventory_payload",
    "build_namespace_document_inventory_answer",
    "build_namespace_document_inventory_payload",
    "build_namespace_search_payload",
    "build_session_access_inventory_answer",
    "build_session_access_inventory_payload",
    "classify_inventory_query",
    "dispatch_authoritative_inventory",
    "extract_requested_graph_reference",
    "extract_requested_kb_collection_id",
    "extract_requested_namespace_query",
    "inventory_answer_origin",
    "inventory_query_requests_grounded_analysis",
    "inventory_scope_kind",
    "is_inventory_query",
    "is_authoritative_inventory_query_type",
    "match_requested_kb_collection_id",
    "resolve_namespace_scope",
    "resolve_namespace_selection_from_metadata",
    "sync_session_kb_collection_state",
]
