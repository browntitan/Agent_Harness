from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from agentic_chatbot_next.rag.hints import infer_rag_execution_hints, normalize_structured_query
from agentic_chatbot_next.rag.inventory import (
    INVENTORY_QUERY_GRAPH_FILE,
    INVENTORY_QUERY_GRAPH_INDEXES,
    INVENTORY_QUERY_KB_COLLECTIONS,
    INVENTORY_QUERY_KB_FILE,
    NAMESPACE_SCOPE_SELECTION_REASON,
    INVENTORY_QUERY_NONE,
    INVENTORY_QUERY_SESSION_ACCESS,
    classify_inventory_query,
    extract_requested_kb_collection_id,
    inventory_query_requests_grounded_analysis,
    is_authoritative_inventory_query_type,
    match_requested_kb_collection_id,
    resolve_namespace_selection_from_metadata,
)
from agentic_chatbot_next.rag.retrieval_scope import has_upload_evidence
from agentic_chatbot_next.rag.requirements_service import (
    REQUIREMENTS_WORKFLOW_KIND,
    is_requirements_extraction_request,
)
from agentic_chatbot_next.runtime.doc_focus import active_doc_focus_from_metadata, is_active_doc_focus_followup
from agentic_chatbot_next.runtime.openwebui_helpers import is_openwebui_helper_message

_ANALYSIS_VERBS_RE = re.compile(
    r"\b(identify|investigate|explain|describe|summari[sz]e|analy[sz]e|synthesi[sz]e|map|walk\s*through|ground)\b",
    flags=re.I,
)
_LIST_ONLY_RE = re.compile(
    r"\b(only\s+(?:return|give|provide|list)|just\s+list|titles?\s+only|inventory|flat\s+inventory|list\s+(?:all\s+)?(?:documents|files|titles))\b",
    flags=re.I,
)
_COMPARISON_RE = re.compile(r"\b(compare|comparison|diff|difference|versus|vs\.)\b", flags=re.I)
_DEEP_RE = re.compile(r"\b(deep|detailed|thorough|comprehensive|sophisticated|walkthrough)\b", flags=re.I)
_MERMAID_RE = re.compile(r"\b(mermaid|diagram|sequence\s+diagram|flowchart)\b", flags=re.I)
_SYSTEM_ORIENTED_RE = re.compile(
    r"\b(architecture|architectural|system\s+design|control\s+flow|workflow|subsystem|interface|boundary|boundaries)\b",
    flags=re.I,
)
_BROAD_SCOPE_RE = re.compile(
    r"\b(across|whole|entire|every|all\s+(?:documents|docs|files|kb|knowledge\s*base|repository|repo|corpus))\b",
    flags=re.I,
)
_REPOSITORY_SCOPE_RE = re.compile(
    r"\b(repo|repository|codebase|code\s*base|project|runtime|application)\b",
    flags=re.I,
)
_HOLISTIC_REPOSITORY_RE = re.compile(
    r"\b("
    r"major\s+(?:repo\s+|repository\s+|codebase\s+|system\s+)?subsystems?|"
    r"subsystem\s+(?:map|overview|walkthrough|architecture|inventory)|"
    r"architectural\s+walkthrough|architecture\s+(?:walkthrough|overview|map)|"
    r"system\s+(?:overview|map|walkthrough)|component\s+map|"
    r"(?:agents?|tools?|skills?|runtime|persistence|observability|routing|coordinator)\s+(?:architecture|overview|map|walkthrough|subsystems?)"
    r")\b",
    flags=re.I,
)
_CROSS_CUTTING_SYSTEM_TERM_RE = re.compile(
    r"\b(agents?|tools?|skills?|rag|retrieval|routing|router|coordinator|runtime|persistence|observability|gateway|memory|graph)\b",
    flags=re.I,
)
_NAMED_DOCUMENT_REFERENCE_RE = re.compile(r"\b[a-z0-9._/-]+\.(?:md|pdf|docx|txt|csv|xlsx|xls)\b", flags=re.I)


def _is_holistic_repository_research(query: str, metadata: Mapping[str, Any] | None = None) -> bool:
    normalized_query = normalize_structured_query(query) or str(query or "").strip()
    if not normalized_query:
        return False
    if is_active_doc_focus_followup(normalized_query, dict(metadata or {})):
        return False
    if _COMPARISON_RE.search(normalized_query):
        return False
    if _NAMED_DOCUMENT_REFERENCE_RE.search(normalized_query) and not _BROAD_SCOPE_RE.search(normalized_query):
        return False
    has_holistic_language = bool(_HOLISTIC_REPOSITORY_RE.search(normalized_query))
    if not has_holistic_language:
        if not _REPOSITORY_SCOPE_RE.search(normalized_query) or not _ANALYSIS_VERBS_RE.search(normalized_query):
            return False
        cross_cutting_terms = {
            match.group(1).casefold()
            for match in _CROSS_CUTTING_SYSTEM_TERM_RE.finditer(normalized_query)
        }
        return len(cross_cutting_terms) >= 2
    has_repo_scope = bool(_REPOSITORY_SCOPE_RE.search(normalized_query))
    if has_repo_scope:
        return True
    return bool(re.search(r"\bmajor\s+(?:repo\s+|repository\s+|codebase\s+|system\s+)?subsystems?\b", normalized_query, flags=re.I))


def _coverage_profile_for_query(query: str, metadata: Mapping[str, Any] | None = None) -> str:
    if _is_holistic_repository_research(query, metadata):
        return "holistic_repository"
    return ""


def _message_metadata(message: Any) -> Dict[str, Any]:
    raw = getattr(message, "metadata", None)
    if isinstance(raw, Mapping):
        return dict(raw)
    additional = getattr(message, "additional_kwargs", None)
    if isinstance(additional, Mapping):
        return dict(additional)
    return {}


def _message_role(message: Any) -> str:
    role = str(getattr(message, "role", "") or "").strip().lower()
    if role:
        return role
    role = str(getattr(message, "type", "") or "").strip().lower()
    if role == "human":
        return "user"
    if role == "ai":
        return "assistant"
    return role


def message_is_context_eligible(
    message: Any,
    *,
    allowed_roles: Sequence[str] = ("user", "assistant"),
) -> bool:
    metadata = _message_metadata(message)
    if metadata.get("openwebui_internal"):
        return False
    if is_openwebui_helper_message(message):
        return False
    role = _message_role(message)
    if allowed_roles and role not in {str(item).strip().lower() for item in allowed_roles if str(item).strip()}:
        return False
    content = getattr(message, "content", "")
    if isinstance(content, list):
        text = " ".join(str(item) for item in content if str(item).strip()).strip()
    else:
        text = str(content or "").strip()
    return bool(text)


def filter_context_messages(
    messages: Iterable[Any],
    *,
    allowed_roles: Sequence[str] = ("user", "assistant"),
) -> List[Any]:
    return [message for message in messages if message_is_context_eligible(message, allowed_roles=allowed_roles)]


@dataclass
class AnswerContract:
    kind: str = "analysis"
    depth: str = "standard"
    broad_coverage: bool = False
    coverage_profile: str = ""
    requires_supporting_evidence: bool = False
    requires_authoritative_inventory: bool = False
    prefer_list_only: bool = False
    final_output_mode: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "depth": self.depth,
            "broad_coverage": self.broad_coverage,
            "coverage_profile": self.coverage_profile,
            "requires_supporting_evidence": self.requires_supporting_evidence,
            "requires_authoritative_inventory": self.requires_authoritative_inventory,
            "prefer_list_only": self.prefer_list_only,
            "final_output_mode": self.final_output_mode,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "AnswerContract":
        payload = dict(raw or {})
        return cls(
            kind=str(payload.get("kind") or "analysis").strip() or "analysis",
            depth=str(payload.get("depth") or "standard").strip() or "standard",
            broad_coverage=bool(payload.get("broad_coverage")),
            coverage_profile=str(payload.get("coverage_profile") or "").strip(),
            requires_supporting_evidence=bool(payload.get("requires_supporting_evidence")),
            requires_authoritative_inventory=bool(payload.get("requires_authoritative_inventory")),
            prefer_list_only=bool(payload.get("prefer_list_only")),
            final_output_mode=str(payload.get("final_output_mode") or "").strip(),
        )


@dataclass
class EvidenceContract:
    source_scope: str = "auto"
    collection_id: str = ""
    authoritative_inventory_required: bool = False
    grounding_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_scope": self.source_scope,
            "collection_id": self.collection_id,
            "authoritative_inventory_required": self.authoritative_inventory_required,
            "grounding_required": self.grounding_required,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "EvidenceContract":
        payload = dict(raw or {})
        return cls(
            source_scope=str(payload.get("source_scope") or "auto").strip() or "auto",
            collection_id=str(payload.get("collection_id") or "").strip(),
            authoritative_inventory_required=bool(payload.get("authoritative_inventory_required")),
            grounding_required=bool(payload.get("grounding_required", True)),
        )


@dataclass
class PresentationPreferences:
    diagram_policy: str = "auto"
    preferred_structure: str = "default"
    verify_overclaim: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagram_policy": self.diagram_policy,
            "preferred_structure": self.preferred_structure,
            "verify_overclaim": self.verify_overclaim,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "PresentationPreferences":
        payload = dict(raw or {})
        return cls(
            diagram_policy=str(payload.get("diagram_policy") or "auto").strip() or "auto",
            preferred_structure=str(payload.get("preferred_structure") or "default").strip() or "default",
            verify_overclaim=bool(payload.get("verify_overclaim")),
        )


@dataclass
class ResolvedTurnIntent:
    source_user_text: str
    normalized_user_objective: str
    effective_user_text: str
    clarification_response: str = ""
    requested_scope: Dict[str, Any] = field(default_factory=dict)
    answer_contract: AnswerContract = field(default_factory=AnswerContract)
    evidence_contract: EvidenceContract = field(default_factory=EvidenceContract)
    presentation_preferences: PresentationPreferences = field(default_factory=PresentationPreferences)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_user_text": self.source_user_text,
            "normalized_user_objective": self.normalized_user_objective,
            "effective_user_text": self.effective_user_text,
            "clarification_response": self.clarification_response,
            "requested_scope": dict(self.requested_scope),
            "answer_contract": self.answer_contract.to_dict(),
            "evidence_contract": self.evidence_contract.to_dict(),
            "presentation_preferences": self.presentation_preferences.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "ResolvedTurnIntent | None":
        payload = dict(raw or {})
        normalized_user_objective = str(payload.get("normalized_user_objective") or "").strip()
        effective_user_text = str(payload.get("effective_user_text") or "").strip()
        source_user_text = str(payload.get("source_user_text") or "").strip()
        if not (normalized_user_objective or effective_user_text or source_user_text):
            return None
        return cls(
            source_user_text=source_user_text,
            normalized_user_objective=normalized_user_objective or source_user_text,
            effective_user_text=effective_user_text or normalized_user_objective or source_user_text,
            clarification_response=str(payload.get("clarification_response") or "").strip(),
            requested_scope=dict(payload.get("requested_scope") or {}),
            answer_contract=AnswerContract.from_dict(payload.get("answer_contract") or {}),
            evidence_contract=EvidenceContract.from_dict(payload.get("evidence_contract") or {}),
            presentation_preferences=PresentationPreferences.from_dict(payload.get("presentation_preferences") or {}),
        )


@dataclass
class ExecutionDigest:
    user_request: str
    planner_summary: str
    answer_contract: Dict[str, Any] = field(default_factory=dict)
    evidence_contract: Dict[str, Any] = field(default_factory=dict)
    presentation_preferences: Dict[str, Any] = field(default_factory=dict)
    task_summaries: List[Dict[str, Any]] = field(default_factory=list)
    artifact_summaries: List[Dict[str, Any]] = field(default_factory=list)
    partial_answer: str = ""
    final_answer: str = ""
    verification: Dict[str, Any] = field(default_factory=dict)
    revision_feedback: str = ""
    truncated: bool = False
    estimated_chars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "user_request": self.user_request,
            "planner_summary": self.planner_summary,
            "answer_contract": dict(self.answer_contract),
            "evidence_contract": dict(self.evidence_contract),
            "presentation_preferences": dict(self.presentation_preferences),
            "task_summaries": [dict(item) for item in self.task_summaries],
            "artifact_summaries": [dict(item) for item in self.artifact_summaries],
            "partial_answer": self.partial_answer,
            "final_answer": self.final_answer,
            "verification": dict(self.verification),
            "revision_feedback": self.revision_feedback,
            "truncated": self.truncated,
            "estimated_chars": self.estimated_chars,
        }
        payload["estimated_chars"] = len(json.dumps(payload, ensure_ascii=False))
        return payload


def resolved_turn_intent_from_metadata(metadata: Mapping[str, Any] | None) -> ResolvedTurnIntent | None:
    payload = dict(metadata or {})
    return ResolvedTurnIntent.from_dict(payload.get("resolved_turn_intent") or {})


def _is_broad_scope(
    query: str,
    *,
    coverage_goal: str,
    requested_scope: Dict[str, Any],
    coverage_profile: str = "",
) -> bool:
    if coverage_goal in {"corpus_wide", "exhaustive"}:
        return True
    if coverage_profile == "holistic_repository":
        return True
    if str(requested_scope.get("coverage") or "").strip().lower() in {"broad", "corpus"}:
        return True
    return bool(_BROAD_SCOPE_RE.search(query))


def _semantic_routing_payload(metadata: Mapping[str, Any] | None) -> Dict[str, Any]:
    session_metadata = dict(metadata or {})
    payload = session_metadata.get("semantic_routing") or {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _infer_requested_scope(query: str, metadata: Mapping[str, Any] | None) -> Dict[str, Any]:
    session_metadata = dict(metadata or {})
    semantic_routing = _semantic_routing_payload(session_metadata)
    normalized_query = normalize_structured_query(query) or str(query or "").strip()
    active_doc_focus = active_doc_focus_from_metadata(session_metadata)
    available_kb_collection_ids = [
        str(item).strip()
        for item in (session_metadata.get("available_kb_collection_ids") or [])
        if str(item).strip()
    ]
    collection_id = (
        str(semantic_routing.get("requested_collection_id") or "").strip()
        or match_requested_kb_collection_id(normalized_query, available_kb_collection_ids)
        or extract_requested_kb_collection_id(normalized_query)
        or str(session_metadata.get("requested_kb_collection_id") or "").strip()
    )
    inventory_type = classify_inventory_query(normalized_query)
    active_graph_ids = [
        str(item).strip()
        for item in (session_metadata.get("active_graph_ids") or [])
        if str(item).strip()
    ]
    scope_kind = str(semantic_routing.get("requested_scope_kind") or "").strip().lower() or "auto"
    if active_doc_focus is not None and is_active_doc_focus_followup(normalized_query, session_metadata):
        scope_kind = "active_doc_focus"
    elif inventory_type == INVENTORY_QUERY_SESSION_ACCESS:
        scope_kind = "session_access"
    elif inventory_type in {INVENTORY_QUERY_GRAPH_INDEXES, INVENTORY_QUERY_GRAPH_FILE}:
        scope_kind = "graph_indexes"
    elif inventory_type in {INVENTORY_QUERY_KB_COLLECTIONS, INVENTORY_QUERY_KB_FILE}:
        scope_kind = "knowledge_base"
    elif is_requirements_extraction_request(normalized_query):
        if has_upload_evidence(session_metadata):
            scope_kind = "uploads"
        elif collection_id:
            scope_kind = "knowledge_base"
    elif collection_id and scope_kind in {"auto", "none"}:
        scope_kind = "knowledge_base"
    elif scope_kind == "none":
        scope_kind = "auto"
    coverage_profile = _coverage_profile_for_query(normalized_query, session_metadata)
    coverage = "broad" if _BROAD_SCOPE_RE.search(normalized_query) or coverage_profile else ""
    return {
        "scope_kind": scope_kind,
        "collection_id": collection_id,
        "graph_ids": active_graph_ids,
        "coverage": coverage,
        "coverage_profile": coverage_profile,
        "scope_text": normalized_query,
        "inventory_query_type": inventory_type,
        "workflow": REQUIREMENTS_WORKFLOW_KIND if is_requirements_extraction_request(normalized_query) else "",
    }


def _infer_answer_contract(query: str, metadata: Mapping[str, Any] | None) -> AnswerContract:
    normalized_query = normalize_structured_query(query) or str(query or "").strip()
    lowered = normalized_query.casefold()
    semantic_routing = _semantic_routing_payload(metadata)
    requested_scope = _infer_requested_scope(normalized_query, metadata)
    inventory_type = classify_inventory_query(normalized_query)
    authoritative_inventory = is_authoritative_inventory_query_type(inventory_type)
    inventory_requires_supporting_evidence = inventory_query_requests_grounded_analysis(
        normalized_query,
        query_type=inventory_type,
    )
    hints = infer_rag_execution_hints(normalized_query)
    requirements_extraction = is_requirements_extraction_request(normalized_query)
    coverage_profile = _coverage_profile_for_query(normalized_query, metadata)
    broad_coverage = _is_broad_scope(
        normalized_query,
        coverage_goal=str(hints.coverage_goal or "").strip().lower(),
        requested_scope=requested_scope,
        coverage_profile=coverage_profile,
    )
    prefer_list_only = bool(_LIST_ONLY_RE.search(normalized_query)) and not bool(_ANALYSIS_VERBS_RE.search(normalized_query))
    explicit_grounding_request = bool(
        re.search(r"\b(evidence|grounded|supporting\s+documents?|citations?|overclaim)\b", lowered, flags=re.I)
    )
    requires_supporting_evidence = (
        broad_coverage
        or explicit_grounding_request
        or bool(_ANALYSIS_VERBS_RE.search(normalized_query))
        or bool(semantic_routing.get("requires_external_evidence"))
        or str(semantic_routing.get("answer_origin") or "").strip().lower() in {"retrieval", "ambiguous"}
    )
    if authoritative_inventory and not inventory_requires_supporting_evidence:
        requires_supporting_evidence = False
    if requirements_extraction:
        kind = REQUIREMENTS_WORKFLOW_KIND
        requires_supporting_evidence = True
        prefer_list_only = True
    elif _COMPARISON_RE.search(normalized_query):
        kind = "comparison"
    elif authoritative_inventory and not inventory_requires_supporting_evidence:
        kind = "inventory"
    elif bool(semantic_routing.get("requires_external_evidence")) and not prefer_list_only:
        kind = "grounded_synthesis"
    elif broad_coverage and not prefer_list_only:
        kind = "grounded_synthesis"
    elif is_active_doc_focus_followup(normalized_query, metadata):
        kind = "grounded_synthesis"
    elif requires_supporting_evidence and not prefer_list_only:
        kind = "grounded_synthesis"
    else:
        kind = "analysis"
    depth = "deep" if _DEEP_RE.search(normalized_query) or kind == "grounded_synthesis" and broad_coverage else "standard"
    final_output_mode = ""
    if kind == REQUIREMENTS_WORKFLOW_KIND:
        final_output_mode = "requirement_inventory"
    elif kind == "inventory":
        final_output_mode = "document_titles_only"
    elif coverage_profile == "holistic_repository" and kind == "grounded_synthesis":
        final_output_mode = "detailed_subsystem_summary"
    return AnswerContract(
        kind=kind,
        depth=depth,
        broad_coverage=broad_coverage,
        coverage_profile=coverage_profile,
        requires_supporting_evidence=requires_supporting_evidence,
        requires_authoritative_inventory=(authoritative_inventory and kind == "inventory") or requirements_extraction,
        prefer_list_only=prefer_list_only,
        final_output_mode=final_output_mode,
    )


def _infer_evidence_contract(query: str, metadata: Mapping[str, Any] | None, answer_contract: AnswerContract) -> EvidenceContract:
    semantic_routing = _semantic_routing_payload(metadata)
    requested_scope = _infer_requested_scope(query, metadata)
    inventory_type = classify_inventory_query(query)
    authoritative_inventory = is_authoritative_inventory_query_type(inventory_type) and answer_contract.kind == "inventory"
    scope_kind = (
        str(semantic_routing.get("requested_scope_kind") or "").strip()
        or str(requested_scope.get("scope_kind") or "auto").strip()
        or "auto"
    )
    if answer_contract.kind == REQUIREMENTS_WORKFLOW_KIND and scope_kind in {"", "none"}:
        scope_kind = str(requested_scope.get("scope_kind") or "auto").strip() or "auto"
    collection_id = (
        str(semantic_routing.get("requested_collection_id") or "").strip()
        or str(requested_scope.get("collection_id") or "").strip()
    )
    return EvidenceContract(
        source_scope=scope_kind,
        collection_id=collection_id,
        authoritative_inventory_required=answer_contract.requires_authoritative_inventory,
        grounding_required=(
            False
            if authoritative_inventory
            else bool(semantic_routing.get("requires_external_evidence"))
            or answer_contract.kind != "inventory"
            or answer_contract.requires_supporting_evidence
        ),
    )


def _infer_presentation_preferences(
    query: str,
    *,
    answer_contract: AnswerContract,
) -> PresentationPreferences:
    normalized_query = normalize_structured_query(query) or str(query or "").strip()
    if re.search(r"\bmermaid\b", normalized_query, flags=re.I):
        diagram_policy = "require_mermaid"
    elif answer_contract.kind == "grounded_synthesis" and _SYSTEM_ORIENTED_RE.search(normalized_query):
        diagram_policy = "auto_mermaid"
    elif _MERMAID_RE.search(normalized_query):
        diagram_policy = "prefer_diagram"
    else:
        diagram_policy = "auto"
    preferred_structure = "inventory" if answer_contract.kind == "inventory" else "synthesis"
    return PresentationPreferences(
        diagram_policy=diagram_policy,
        preferred_structure=preferred_structure,
        verify_overclaim=bool(re.search(r"\boverclaim|verify\b", normalized_query, flags=re.I)),
    )


def _merge_scope(base: Mapping[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base or {})
    for key, value in dict(update or {}).items():
        if value in (None, "", [], {}):
            continue
        merged[key] = value
    return merged


def resolve_turn_intent(user_text: str, metadata: Mapping[str, Any] | None) -> ResolvedTurnIntent:
    session_metadata = dict(metadata or {})
    normalized_user_text = normalize_structured_query(user_text) or str(user_text or "").strip()
    pending = dict(session_metadata.get("pending_clarification") or {})
    previous = resolved_turn_intent_from_metadata(session_metadata)
    if previous is None and pending:
        previous = ResolvedTurnIntent.from_dict(pending.get("resolved_turn_intent") or {})
    if pending and previous is not None:
        normalized_objective = previous.normalized_user_objective or previous.source_user_text or normalized_user_text
        requested_scope = _merge_scope(previous.requested_scope, _infer_requested_scope(normalized_user_text, session_metadata))
        requested_scope["clarification_resolution"] = normalized_user_text
        pending_reason = str(pending.get("reason") or "").strip().lower()
        requirement_candidate_documents = [
            dict(item)
            for item in (session_metadata.get("requirements_candidate_documents") or [])
            if isinstance(item, Mapping)
        ]
        selected_collection_id = ""
        selected_collection_ids: list[str] = []
        selected_graph_ids: list[str] = []
        selected_document_names: list[str] = []
        selected_document_ids: list[str] = []
        if pending_reason == "kb_collection_selection":
            selected_collection_id = match_requested_kb_collection_id(
                normalized_user_text,
                [str(item) for item in (pending.get("options") or []) if str(item).strip()],
                pending_reason=pending_reason,
            )
            if selected_collection_id:
                requested_scope["collection_id"] = selected_collection_id
                requested_scope["requested_kb_collection_id"] = selected_collection_id
                requested_scope["kb_collection_confirmed"] = True
        elif pending_reason == NAMESPACE_SCOPE_SELECTION_REASON:
            selection = resolve_namespace_selection_from_metadata(normalized_user_text, session_metadata)
            selected_collection_ids = [
                str(item).strip()
                for item in (selection.get("collection_ids") or [])
                if str(item).strip()
            ]
            selected_graph_ids = [
                str(item).strip()
                for item in (selection.get("graph_ids") or [])
                if str(item).strip()
            ]
            if selected_collection_ids:
                selected_collection_id = selected_collection_ids[0]
                requested_scope["collection_id"] = selected_collection_id
                requested_scope["requested_kb_collection_id"] = selected_collection_id
                requested_scope["search_collection_ids"] = list(selected_collection_ids)
                requested_scope["kb_collection_confirmed"] = True
            if selected_graph_ids:
                requested_scope["graph_ids"] = list(selected_graph_ids)
        elif pending_reason == "requirements_document_selection":
            options = [str(item).strip() for item in (pending.get("options") or []) if str(item).strip()]
            lowered_response = normalized_user_text.casefold()
            if re.search(r"\ball\s+(?:documents?|docs?|files?)\b", lowered_response):
                requested_scope["requirements_all_documents"] = True
                selected_document_ids = [
                    str(item.get("doc_id") or "").strip()
                    for item in requirement_candidate_documents
                    if str(item.get("doc_id") or "").strip()
                ]
                if selected_document_ids:
                    requested_scope["document_ids"] = list(selected_document_ids)
            else:
                ordinal_map = {
                    "first": 0,
                    "1": 0,
                    "one": 0,
                    "second": 1,
                    "2": 1,
                    "two": 1,
                    "third": 2,
                    "3": 2,
                    "three": 2,
                }
                selected_index = None
                for token, index in ordinal_map.items():
                    if re.search(rf"\b{re.escape(token)}\b", lowered_response):
                        selected_index = index
                        break
                if selected_index is not None and selected_index < len(options):
                    selected_document_names = [options[selected_index]]
                else:
                    for option in options:
                        if option.casefold() in lowered_response or lowered_response in option.casefold():
                            selected_document_names = [option]
                            break
                if selected_document_names:
                    requested_scope["document_names"] = list(selected_document_names)
                    selected_titles = {item.casefold() for item in selected_document_names}
                    selected_document_ids = [
                        str(item.get("doc_id") or "").strip()
                        for item in requirement_candidate_documents
                        if str(item.get("doc_id") or "").strip()
                        and (
                            str(item.get("title") or "").strip().casefold() in selected_titles
                            or str(item.get("doc_id") or "").strip().casefold() in selected_titles
                        )
                    ]
                    if selected_document_ids:
                        requested_scope["document_ids"] = list(selected_document_ids)
                        requested_scope["selected_doc_ids"] = list(selected_document_ids)
        answer_contract = previous.answer_contract
        evidence_contract = previous.evidence_contract
        if selected_collection_id:
            evidence_contract = EvidenceContract(
                source_scope="knowledge_base",
                collection_id=selected_collection_id,
                authoritative_inventory_required=previous.evidence_contract.authoritative_inventory_required,
                grounding_required=True,
            )
        presentation_preferences = previous.presentation_preferences
        effective_user_text = normalized_objective
        if normalized_user_text:
            clarification_resolution = normalized_user_text
            if selected_document_names:
                clarification_resolution = "Use document " + ", ".join(f"`{item}`" for item in selected_document_names) + "."
            elif bool(requested_scope.get("requirements_all_documents")):
                clarification_resolution = "Extract from all documents."
            elif selected_graph_ids or len(selected_collection_ids) > 1:
                resolution_parts: list[str] = []
                if selected_collection_ids:
                    resolution_parts.append(
                        "Use knowledge base collections " + ", ".join(f"`{item}`" for item in selected_collection_ids) + "."
                    )
                if selected_graph_ids:
                    resolution_parts.append(
                        "Use graphs " + ", ".join(f"`{item}`" for item in selected_graph_ids) + "."
                    )
                clarification_resolution = " ".join(resolution_parts).strip()
            elif selected_collection_id:
                clarification_resolution = f"Use knowledge base collection `{selected_collection_id}`."
            effective_user_text = (
                f"{normalized_objective}\n\n"
                "Clarification resolution:\n"
                f"- {clarification_resolution}"
            ).strip()
        return ResolvedTurnIntent(
            source_user_text=str(user_text or "").strip(),
            normalized_user_objective=normalized_objective,
            effective_user_text=effective_user_text,
            clarification_response=normalized_user_text,
            requested_scope=requested_scope,
            answer_contract=answer_contract,
            evidence_contract=evidence_contract,
            presentation_preferences=presentation_preferences,
        )
    answer_contract = _infer_answer_contract(normalized_user_text, session_metadata)
    evidence_contract = _infer_evidence_contract(normalized_user_text, session_metadata, answer_contract)
    presentation_preferences = _infer_presentation_preferences(
        normalized_user_text,
        answer_contract=answer_contract,
    )
    return ResolvedTurnIntent(
        source_user_text=str(user_text or "").strip(),
        normalized_user_objective=normalized_user_text,
        effective_user_text=normalized_user_text,
        clarification_response="",
        requested_scope=_infer_requested_scope(normalized_user_text, session_metadata),
        answer_contract=answer_contract,
        evidence_contract=evidence_contract,
        presentation_preferences=presentation_preferences,
    )


def resolved_turn_intent_prompt_block(metadata: Mapping[str, Any] | None) -> str:
    intent = resolved_turn_intent_from_metadata(metadata)
    if intent is None:
        return ""
    lines = [
        "## Resolved Turn Intent",
        f"objective: {intent.normalized_user_objective}",
        f"effective_request: {intent.effective_user_text}",
    ]
    if intent.clarification_response:
        lines.append(f"clarification_response: {intent.clarification_response}")
    requested_scope = {key: value for key, value in dict(intent.requested_scope).items() if value not in ("", [], {}, None)}
    if requested_scope:
        lines.append("requested_scope: " + json.dumps(requested_scope, ensure_ascii=False))
    lines.append("answer_contract: " + json.dumps(intent.answer_contract.to_dict(), ensure_ascii=False))
    lines.append("evidence_contract: " + json.dumps(intent.evidence_contract.to_dict(), ensure_ascii=False))
    lines.append("presentation_preferences: " + json.dumps(intent.presentation_preferences.to_dict(), ensure_ascii=False))
    return "\n".join(lines)


def _has_holistic_research_campaign(tasks: Sequence[Mapping[str, Any]]) -> bool:
    normalized_tasks = [dict(task) for task in tasks if isinstance(task, Mapping)]
    has_title_scan = any(
        "title_candidates" in [str(item) for item in (task.get("produces_artifacts") or []) if str(item)]
        for task in normalized_tasks
    )
    has_seed_scan = any(
        str(task.get("executor") or "").strip() == "rag_worker"
        and str(task.get("answer_mode") or "").strip().lower() == "evidence_only"
        for task in normalized_tasks
    )
    has_facet_phase = any(
        bool(dict(task.get("controller_hints") or {}).get("dynamic_facet_fanout"))
        or "research_facets" in [str(item) for item in (task.get("produces_artifacts") or []) if str(item)]
        for task in normalized_tasks
    )
    has_doc_review_phase = any(
        bool(dict(task.get("controller_hints") or {}).get("dynamic_doc_review_fanout"))
        or "doc_digest" in [str(item) for item in (task.get("produces_artifacts") or []) if str(item)]
        for task in normalized_tasks
    )
    has_subsystem_inventory = any(
        "subsystem_inventory" in [str(item) for item in (task.get("produces_artifacts") or []) if str(item)]
        for task in normalized_tasks
    )
    return bool(has_title_scan and has_seed_scan and has_facet_phase and has_doc_review_phase and has_subsystem_inventory)


def plan_satisfies_intent(task_plan: Sequence[Mapping[str, Any]], intent: ResolvedTurnIntent | None) -> bool:
    if intent is None:
        return True
    tasks = [dict(item) for item in task_plan if isinstance(item, Mapping)]
    if not tasks:
        return False
    if (
        intent.answer_contract.kind == "grounded_synthesis"
        and intent.answer_contract.coverage_profile == "holistic_repository"
    ):
        return _has_holistic_research_campaign(tasks)
    if intent.answer_contract.kind == "grounded_synthesis" and intent.answer_contract.broad_coverage:
        has_evidence_phase = any(
            str(task.get("executor") or "").strip() == "rag_worker"
            and (
                str(task.get("answer_mode") or "").strip().lower() == "evidence_only"
                or "doc_focus" in [str(item) for item in (task.get("produces_artifacts") or []) if str(item)]
                or "research_facets" in [str(item) for item in (task.get("produces_artifacts") or []) if str(item)]
            )
            for task in tasks
        )
        has_synthesis_phase = any(
            str(task.get("executor") or "").strip() in {"general", "finalizer", "rag_worker"}
            and (
                [str(item) for item in (task.get("depends_on") or []) if str(item)]
                or "subsystem_inventory" in [str(item) for item in (task.get("produces_artifacts") or []) if str(item)]
                or "doc_digest" in [str(item) for item in (task.get("produces_artifacts") or []) if str(item)]
                or str(task.get("answer_mode") or "").strip().lower() == "answer"
                and str(task.get("executor") or "").strip() == "general"
            )
            for task in tasks
        )
        return has_evidence_phase and has_synthesis_phase
    if intent.answer_contract.kind == "inventory" and intent.answer_contract.requires_authoritative_inventory:
        return any(
            str(task.get("result_mode") or "").strip().lower() == "inventory"
            or bool(dict(task.get("controller_hints") or {}).get("prefer_inventory_output"))
            for task in tasks
        )
    return True


def infer_result_provenance(result: Mapping[str, Any] | None) -> str:
    payload = dict(result or {})
    metadata = dict(payload.get("metadata") or {})
    rag_contract = dict(metadata.get("rag_contract") or {})
    retrieval_summary = dict(rag_contract.get("retrieval_summary") or {})
    search_mode = str(retrieval_summary.get("search_mode") or "").strip().lower()
    worker_request = dict(metadata.get("worker_request") or {})
    if search_mode == "metadata_inventory":
        return "authoritative_inventory"
    if metadata.get("rag_search_result") or rag_contract:
        if str(worker_request.get("answer_mode") or metadata.get("answer_mode") or "").strip().lower() == "evidence_only":
            return "retrieval_candidates"
        if str(worker_request.get("result_mode") or "").strip().lower() == "inventory":
            return "retrieval_candidates"
        return "grounded_retrieval"
    if metadata.get("doc_focus_result"):
        return "structured_doc_focus"
    return "agent_output"


def infer_artifact_provenance(artifact: Mapping[str, Any] | None) -> str:
    payload = dict(artifact or {})
    data = dict(payload.get("data") or {})
    artifact_type = str(payload.get("artifact_type") or "").strip().lower()
    if str(data.get("view") or "").strip().lower() in {
        "kb_file_inventory",
        "kb_collection_access_inventory",
        "session_access_inventory",
        "graph_index_inventory",
    }:
        return "authoritative_inventory"
    if artifact_type in {"doc_focus", "facet_matches", "subsystem_evidence"}:
        return "grounded_retrieval"
    if artifact_type in {"title_candidates", "doc_digest", "subsystem_inventory", "research_facets", "research_coverage_ledger"}:
        return "structured_synthesis"
    if artifact_type in {"evidence_request", "evidence_response"}:
        return "workflow_signal"
    return "artifact"


def _truncate_text(value: Any, *, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def build_execution_digest(
    execution_state: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None,
    artifacts: Sequence[Mapping[str, Any]] | None = None,
    revision_feedback: str = "",
    task_output_char_limit: int = 900,
    artifact_data_char_limit: int = 500,
) -> ExecutionDigest:
    state = dict(execution_state or {})
    intent = resolved_turn_intent_from_metadata(metadata)
    task_summaries: List[Dict[str, Any]] = []
    for item in state.get("task_results") or []:
        if not isinstance(item, Mapping):
            continue
        task = dict(item)
        task_metadata = dict(task.get("metadata") or {})
        rag_contract = dict(task_metadata.get("rag_contract") or {})
        retrieval_summary = dict(rag_contract.get("retrieval_summary") or {})
        task_summaries.append(
            {
                "task_id": str(task.get("task_id") or ""),
                "title": str(task.get("title") or ""),
                "executor": str(task.get("executor") or ""),
                "status": str(task.get("status") or ""),
                "provenance_kind": infer_result_provenance(task),
                "search_mode": str(retrieval_summary.get("search_mode") or "").strip(),
                "warnings": [str(value) for value in (task.get("warnings") or []) if str(value)],
                "output_excerpt": _truncate_text(task.get("output") or "", limit=task_output_char_limit),
            }
        )
    artifact_summaries: List[Dict[str, Any]] = []
    for item in artifacts or []:
        if not isinstance(item, Mapping):
            continue
        artifact = dict(item)
        artifact_summaries.append(
            {
                "artifact_id": str(artifact.get("artifact_id") or ""),
                "artifact_type": str(artifact.get("artifact_type") or ""),
                "producer_task_id": str(artifact.get("producer_task_id") or ""),
                "summary": _truncate_text(artifact.get("summary") or artifact.get("artifact_type") or "", limit=220),
                "provenance_kind": infer_artifact_provenance(artifact),
                "data_preview": _truncate_text(json.dumps(dict(artifact.get("data") or {}), ensure_ascii=False), limit=artifact_data_char_limit),
            }
        )
    digest = ExecutionDigest(
        user_request=str(
            state.get("user_request")
            or (intent.normalized_user_objective if intent is not None else "")
        ).strip(),
        planner_summary=str(state.get("planner_summary") or "").strip(),
        answer_contract=intent.answer_contract.to_dict() if intent is not None else {},
        evidence_contract=intent.evidence_contract.to_dict() if intent is not None else {},
        presentation_preferences=intent.presentation_preferences.to_dict() if intent is not None else {},
        task_summaries=task_summaries,
        artifact_summaries=artifact_summaries,
        partial_answer=_truncate_text(state.get("partial_answer") or "", limit=2000),
        final_answer=_truncate_text(state.get("final_answer") or "", limit=2400),
        verification=dict(state.get("verification") or {}),
        revision_feedback=_truncate_text(revision_feedback, limit=1200),
    )
    payload = digest.to_dict()
    digest.truncated = any(len(str(item.get("output_excerpt") or "")) >= task_output_char_limit for item in task_summaries)
    digest.estimated_chars = int(payload.get("estimated_chars") or 0)
    return digest


__all__ = [
    "AnswerContract",
    "EvidenceContract",
    "ExecutionDigest",
    "PresentationPreferences",
    "ResolvedTurnIntent",
    "build_execution_digest",
    "filter_context_messages",
    "infer_artifact_provenance",
    "infer_result_provenance",
    "message_is_context_eligible",
    "plan_satisfies_intent",
    "resolve_turn_intent",
    "resolved_turn_intent_from_metadata",
    "resolved_turn_intent_prompt_block",
]
