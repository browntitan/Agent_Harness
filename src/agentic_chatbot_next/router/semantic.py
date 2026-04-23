from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from agentic_chatbot_next.persistence.postgres.entities import normalize_entity_text
from agentic_chatbot_next.rag.retrieval_scope import (
    document_source_policy_requires_repository,
    has_upload_evidence,
)
from agentic_chatbot_next.rag.inventory import (
    INVENTORY_QUERY_GRAPH_INDEXES,
    INVENTORY_QUERY_KB_COLLECTIONS,
    INVENTORY_QUERY_KB_FILE,
    INVENTORY_QUERY_NONE,
    INVENTORY_QUERY_SESSION_ACCESS,
    classify_inventory_query,
    inventory_answer_origin,
    inventory_query_requests_grounded_analysis,
    inventory_scope_kind,
    is_authoritative_inventory_query_type,
    match_requested_kb_collection_id,
)
from agentic_chatbot_next.runtime.doc_focus import is_active_doc_focus_followup

_ALLOWED_ANSWER_ORIGINS = {"parametric", "conversation", "retrieval", "ambiguous"}
_ALLOWED_SCOPE_KINDS = {
    "knowledge_base",
    "uploads",
    "active_doc_focus",
    "session_access",
    "graph_indexes",
    "none",
}
_OBVIOUS_BASIC_TURNS = {
    "hello",
    "hello there",
    "hi",
    "hey",
    "how are you",
    "how are you today",
    "thanks",
    "thank you",
}


def _normalize_answer_origin(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in _ALLOWED_ANSWER_ORIGINS else "parametric"


def _normalize_scope_kind(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in _ALLOWED_SCOPE_KINDS else "none"


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _dedupe_strings(values: Iterable[Any]) -> list[str]:
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


@dataclass(frozen=True)
class SemanticRoutingContract:
    route: str = "BASIC"
    suggested_agent: str = ""
    requires_external_evidence: bool = False
    answer_origin: str = "parametric"
    requested_scope_kind: str = "none"
    requested_collection_id: str = ""
    confidence: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": str(self.route or "BASIC").strip().upper() or "BASIC",
            "suggested_agent": str(self.suggested_agent or "").strip().lower(),
            "requires_external_evidence": bool(self.requires_external_evidence),
            "answer_origin": _normalize_answer_origin(self.answer_origin),
            "requested_scope_kind": _normalize_scope_kind(self.requested_scope_kind),
            "requested_collection_id": str(self.requested_collection_id or "").strip(),
            "confidence": max(0.0, min(1.0, float(self.confidence or 0.0))),
            "reasoning": str(self.reasoning or "").strip(),
        }

    @classmethod
    def from_value(
        cls,
        value: Mapping[str, Any] | "SemanticRoutingContract" | None,
        *,
        default_route: str = "BASIC",
        default_confidence: float = 0.0,
        default_reasoning: str = "",
        default_suggested_agent: str = "",
    ) -> "SemanticRoutingContract":
        if isinstance(value, SemanticRoutingContract):
            return value
        payload = dict(value or {})
        return cls(
            route=str(payload.get("route") or default_route or "BASIC").strip().upper() or "BASIC",
            suggested_agent=str(payload.get("suggested_agent") or default_suggested_agent or "").strip().lower(),
            requires_external_evidence=_coerce_bool(payload.get("requires_external_evidence"), default=False),
            answer_origin=_normalize_answer_origin(payload.get("answer_origin")),
            requested_scope_kind=_normalize_scope_kind(payload.get("requested_scope_kind")),
            requested_collection_id=str(payload.get("requested_collection_id") or "").strip(),
            confidence=max(0.0, min(1.0, float(payload.get("confidence") or default_confidence or 0.0))),
            reasoning=str(payload.get("reasoning") or default_reasoning or "").strip(),
        )


def visible_kb_collection_ids(session_metadata: Mapping[str, Any] | None) -> list[str]:
    metadata = dict(session_metadata or {})
    return _dedupe_strings(metadata.get("available_kb_collection_ids") or [])


def has_visible_uploads(session_metadata: Mapping[str, Any] | None) -> bool:
    metadata = dict(session_metadata or {})
    if has_upload_evidence(metadata):
        return True
    if document_source_policy_requires_repository(metadata):
        return False
    if str(metadata.get("upload_collection_id") or "").strip() and str(metadata.get("upload_collection_id") or "").strip() != str(metadata.get("kb_collection_id") or "").strip():
        return True
    return False


def obvious_basic_turn(query: str) -> bool:
    normalized = normalize_entity_text(query)
    if not normalized:
        return True
    if normalized in _OBVIOUS_BASIC_TURNS:
        return True
    return normalized.startswith("hello ") or normalized.startswith("hi ") or normalized.startswith("hey ")


def select_requested_collection_id(
    query: str,
    visible_collection_ids: Sequence[str],
) -> str:
    options = _dedupe_strings(visible_collection_ids)
    if not options:
        return ""
    matched = match_requested_kb_collection_id(query, options)
    if matched:
        return matched

    normalized_query = f" {normalize_entity_text(query)} "
    for collection_id in sorted(options, key=lambda item: (-len(item), item.casefold())):
        normalized_collection = normalize_entity_text(collection_id)
        if normalized_collection and f" {normalized_collection} " in normalized_query:
            return collection_id
    return ""


def build_deterministic_semantic_contract(
    *,
    user_text: str,
    route: str,
    suggested_agent: str = "",
    confidence: float = 0.0,
    reasoning: str = "",
    session_metadata: Mapping[str, Any] | None = None,
) -> SemanticRoutingContract:
    metadata = dict(session_metadata or {})
    visible_collections = visible_kb_collection_ids(metadata)
    requested_collection_id = select_requested_collection_id(user_text, visible_collections)
    inventory_query_type = classify_inventory_query(user_text)
    inventory_metadata_only = (
        is_authoritative_inventory_query_type(inventory_query_type)
        and not inventory_query_requests_grounded_analysis(user_text, query_type=inventory_query_type)
    )
    active_doc_focus = is_active_doc_focus_followup(user_text, metadata)
    has_uploads = has_visible_uploads(metadata)
    requested_scope_kind = "none"
    answer_origin = "parametric"
    requires_external_evidence = False

    if active_doc_focus:
        requested_scope_kind = "active_doc_focus"
        answer_origin = "retrieval"
        requires_external_evidence = True
    elif inventory_metadata_only:
        requested_scope_kind = inventory_scope_kind(inventory_query_type)
        answer_origin = inventory_answer_origin(inventory_query_type)
        requires_external_evidence = False
    elif requested_collection_id:
        requested_scope_kind = "knowledge_base"
        answer_origin = "retrieval" if len(visible_collections) <= 1 or requested_collection_id else "ambiguous"
        requires_external_evidence = True
    elif str(suggested_agent or "").strip().lower() == "graph_manager":
        requested_scope_kind = "graph_indexes"
        answer_origin = "retrieval"
        requires_external_evidence = True
    elif str(suggested_agent or "").strip().lower() in {"rag_worker", "coordinator"}:
        requested_scope_kind = "knowledge_base"
        answer_origin = "retrieval"
        requires_external_evidence = True
    elif str(suggested_agent or "").strip().lower() == "data_analyst" or has_uploads:
        requested_scope_kind = "uploads"
        answer_origin = "retrieval"
        requires_external_evidence = True

    return SemanticRoutingContract(
        route=str(route or "BASIC").strip().upper() or "BASIC",
        suggested_agent=str(suggested_agent or "").strip().lower(),
        requires_external_evidence=requires_external_evidence,
        answer_origin=answer_origin,
        requested_scope_kind=requested_scope_kind,
        requested_collection_id=requested_collection_id,
        confidence=confidence,
        reasoning=reasoning,
    )


def semantic_contract_requires_agent(contract: SemanticRoutingContract) -> bool:
    payload = contract.to_dict()
    if bool(payload["requires_external_evidence"]):
        return True
    if str(payload["requested_scope_kind"] or "") in {
        "knowledge_base",
        "uploads",
        "active_doc_focus",
        "session_access",
        "graph_indexes",
    }:
        return True
    if str(payload["requested_collection_id"] or "").strip():
        return True
    if str(payload["answer_origin"] or "") in {"retrieval", "ambiguous"}:
        return True
    return False


def default_agent_for_semantic_contract(
    contract: SemanticRoutingContract,
    *,
    fallback_suggested_agent: str = "",
) -> str:
    preferred = str(contract.suggested_agent or fallback_suggested_agent or "").strip().lower()
    if preferred:
        return preferred
    scope_kind = _normalize_scope_kind(contract.requested_scope_kind)
    if scope_kind == "active_doc_focus":
        return "coordinator"
    if scope_kind == "session_access":
        return "general"
    if scope_kind == "graph_indexes":
        if bool(contract.requires_external_evidence) or _normalize_answer_origin(contract.answer_origin) in {"retrieval", "ambiguous"}:
            return "graph_manager"
        return "general"
    if scope_kind == "uploads":
        return "data_analyst"
    if scope_kind == "knowledge_base":
        if not bool(contract.requires_external_evidence) and _normalize_answer_origin(contract.answer_origin) not in {"retrieval", "ambiguous"}:
            return "general"
        return "rag_worker"
    if _normalize_answer_origin(contract.answer_origin) in {"retrieval", "ambiguous"}:
        return "rag_worker"
    return "general"


__all__ = [
    "SemanticRoutingContract",
    "build_deterministic_semantic_contract",
    "default_agent_for_semantic_contract",
    "has_visible_uploads",
    "obvious_basic_turn",
    "select_requested_collection_id",
    "semantic_contract_requires_agent",
    "visible_kb_collection_ids",
]
