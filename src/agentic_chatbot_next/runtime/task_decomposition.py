from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from agentic_chatbot_next.rag.hints import normalize_structured_query
from agentic_chatbot_next.rag.doc_targets import extract_named_document_targets
from agentic_chatbot_next.rag.inventory import (
    INVENTORY_QUERY_GRAPH_INDEXES,
    INVENTORY_QUERY_NONE,
    classify_inventory_query,
    inventory_query_requests_grounded_analysis,
    is_authoritative_inventory_query_type,
)
from agentic_chatbot_next.runtime.doc_focus import is_active_doc_focus_followup

_ARITHMETIC_RE = re.compile(
    r"\b(calculate|compute|math|arithmetic|percentage|percent|subtotal|total|average|mean|sum|convert)\b"
    r"|(?:\d[\d,]*(?:\.\d+)?\s*(?:%|percent)\s+of\s+\d)"
    r"|(?:\d[\d,]*(?:\.\d+)?\s*(?:\+|-|\*|/|x|times|divided\s+by)\s*\d)",
    re.IGNORECASE,
)
_RETRIEVAL_RE = re.compile(
    r"\b(search|find|look\s+up|lookup|retrieve|read|summari[sz]e|analy[sz]e|compare|ground|cite|citation|evidence)\b.*"
    r"\b(indexed\s+docs?|documents?|docs?|knowledge\s+base|kb|policy|policies|contract|contracts|clause|clauses)\b"
    r"|\b(indexed\s+docs?|documents?|docs?|knowledge\s+base|kb|policy|policies|contract|contracts|clause|clauses)\b.*"
    r"\b(search|find|look\s+up|lookup|retrieve|read|summari[sz]e|analy[sz]e|compare|ground|cite|citation|evidence)\b",
    re.IGNORECASE,
)
_DIRECT_DOC_HINT_RE = re.compile(
    r"\b(indexed\s+docs?|search\s+docs?|search\s+documents?|rag|knowledge\s+base|kb|policy|policies)\b",
    re.IGNORECASE,
)
_DOC_ACTION_RE = re.compile(
    r"\b(search|find|look\s+up|lookup|retrieve|read|summari[sz]e|analy[sz]e|compare|ground|cite)\b",
    re.IGNORECASE,
)
_DATA_ANALYSIS_RE = re.compile(r"\b(csv|excel|spreadsheet|workbook|dataframe|pandas|dataset)\b", re.IGNORECASE)
_REQUIREMENTS_RE = re.compile(
    r"\b(?:extract|pull|list|inventory|organize|find|harvest|export|download|return)\b"
    r".*\b(?:shall|must|requirement|requirements|obligation|obligations|clause|clauses)\b"
    r"|\b(?:shall|must)\s+statements?\b"
    r"|\brequirement\s+statements?\b"
    r"|\b(?:far|dfars)\b.*\b(?:clause|clauses|requirement|requirements|obligation|obligations)\b",
    re.IGNORECASE | re.DOTALL,
)
_BROAD_OR_STAGED_RE = re.compile(
    r"\b(across|whole|entire|every|all\s+(?:documents|docs|files|kb|knowledge\s*base|corpus)|"
    r"deep|detailed|thorough|comprehensive|verify|validate|cross[-\s]?check|multi[-\s]?stage|"
    r"plan|planner|background|worker|workers|orchestrate|coordinate)\b",
    re.IGNORECASE,
)
_CLAUSE_REVIEW_RE = re.compile(
    r"\b(clause|clauses|redline|redlines|marked\s+changes?|tracked\s+changes?|supplier\s+position)\b",
    re.IGNORECASE,
)
_POLICY_EVIDENCE_RE = re.compile(
    r"\b(policy|policies|guidance|knowledge\s+base|kb|collection|internal\s+policy)\b",
    re.IGNORECASE,
)
_PER_ITEM_LOOP_RE = re.compile(
    r"\b(each|every|all|per[-\s]?item|loop|fan\s*out|for\s+each)\b",
    re.IGNORECASE,
)
_BUYER_RESPONSE_RE = re.compile(
    r"\b(buyer|supplier|write\s+back|recommended\s+action|recommendation|risk\s+level)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TaskDecompositionSlice:
    kind: str
    executor: str
    description: str
    independent: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "executor": self.executor,
            "description": self.description,
            "independent": self.independent,
        }


@dataclass(frozen=True)
class TaskDecompositionDecision:
    is_mixed_intent: bool
    selected_agent: str
    route_kind: str = "none"
    reason: str = ""
    slices: tuple[TaskDecompositionSlice, ...] = ()
    original_agent: str = ""

    @property
    def applied(self) -> bool:
        return (
            self.route_kind in {"general_direct", "coordinator"}
            and bool(self.selected_agent)
            and self.selected_agent != self.original_agent
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_mixed_intent": self.is_mixed_intent,
            "selected_agent": self.selected_agent,
            "route_kind": self.route_kind,
            "reason": self.reason,
            "slice_count": len(self.slices),
            "slices": [item.to_dict() for item in self.slices],
            "original_agent": self.original_agent,
            "applied": self.applied,
        }


def _normalized_query(query: str) -> str:
    return normalize_structured_query(query) or str(query or "").strip()


def _is_preserved_fast_path(query: str, session_metadata: Mapping[str, Any] | None = None) -> bool:
    normalized = _normalized_query(query)
    if not normalized:
        return True
    if _REQUIREMENTS_RE.search(normalized) and not is_clause_policy_workflow(
        normalized,
        session_metadata=session_metadata,
    ):
        return True
    inventory_type = classify_inventory_query(normalized)
    if inventory_type == INVENTORY_QUERY_GRAPH_INDEXES:
        return True
    if (
        inventory_type != INVENTORY_QUERY_NONE
        and is_authoritative_inventory_query_type(inventory_type)
        and not inventory_query_requests_grounded_analysis(normalized, query_type=inventory_type)
    ):
        return True
    return is_active_doc_focus_followup(normalized, session_metadata)


def is_clause_policy_workflow(
    query: str,
    *,
    session_metadata: Mapping[str, Any] | None = None,
) -> bool:
    normalized = _normalized_query(query)
    if not normalized:
        return False
    metadata = dict(session_metadata or {})
    has_upload_context = bool(metadata.get("uploaded_doc_ids") or metadata.get("has_uploads"))
    mixed_sources = bool(_CLAUSE_REVIEW_RE.search(normalized)) and bool(_POLICY_EVIDENCE_RE.search(normalized))
    needs_loop_or_response = bool(_PER_ITEM_LOOP_RE.search(normalized)) or bool(_BUYER_RESPONSE_RE.search(normalized))
    return bool(mixed_sources and needs_loop_or_response and (has_upload_context or "uploaded" in normalized.casefold()))


def build_planner_input_packet(
    query: str,
    *,
    session_metadata: Mapping[str, Any] | None = None,
    available_agents: Sequence[str] | None = None,
    available_tools: Sequence[str] | None = None,
    available_skill_packs: Sequence[str] | None = None,
) -> dict[str, Any]:
    metadata = dict(session_metadata or {})
    effective = dict(metadata.get("effective_capabilities") or {})
    risk_flags: list[str] = []
    if is_clause_policy_workflow(query, session_metadata=metadata):
        risk_flags.extend(["mixed_evidence_scopes", "requires_per_item_loop", "buyer_response_policy_review"])
    if metadata.get("uploaded_doc_ids"):
        risk_flags.append("has_upload_artifacts")
    if metadata.get("requested_kb_collection_id") or metadata.get("kb_collection_id"):
        risk_flags.append("selected_kb_collection")
    return {
        "user_request": _normalized_query(query),
        "attachments": list(metadata.get("uploaded_doc_ids") or []),
        "selected_kb_collections": [
            str(item)
            for item in (
                metadata.get("search_collection_ids")
                or [metadata.get("requested_kb_collection_id") or metadata.get("kb_collection_id")]
            )
            if str(item or "").strip()
        ],
        "effective_capability_profile": effective,
        "available_agents": list(available_agents or effective.get("enabled_agents") or []),
        "available_tools": list(available_tools or effective.get("enabled_tools") or []),
        "available_skill_packs": list(available_skill_packs or effective.get("enabled_skill_pack_ids") or []),
        "permission_mode": str(effective.get("permission_mode") or metadata.get("permission_mode") or "default"),
        "preserved_fast_path_policy": str(
            effective.get("fast_path_policy") or metadata.get("fast_path_policy") or "inventory_plus_simple"
        ),
        "risk_flags": sorted(set(risk_flags)),
    }


def detect_task_slices(
    query: str,
    *,
    session_metadata: Mapping[str, Any] | None = None,
) -> tuple[TaskDecompositionSlice, ...]:
    normalized = _normalized_query(query)
    if _is_preserved_fast_path(normalized, session_metadata):
        return ()

    slices: list[TaskDecompositionSlice] = []
    if is_clause_policy_workflow(normalized, session_metadata=session_metadata):
        return (
            TaskDecompositionSlice(
                kind="document_clause_redline_extraction",
                executor="general",
                description="Extract structured clauses and redlines from uploaded document artifacts.",
                independent=False,
            ),
            TaskDecompositionSlice(
                kind="policy_guidance_fanout",
                executor="rag_worker",
                description="Search the selected policy collection for each extracted clause or redline.",
                independent=False,
            ),
            TaskDecompositionSlice(
                kind="buyer_response_synthesis",
                executor="general",
                description="Synthesize buyer-facing actions, risk levels, and unresolved questions with evidence.",
                independent=False,
            ),
        )
    if _ARITHMETIC_RE.search(normalized):
        slices.append(
            TaskDecompositionSlice(
                kind="utility_calculation",
                executor="utility",
                description="Complete the arithmetic or calculator slice.",
            )
        )
    has_named_doc_target = bool(extract_named_document_targets(normalized))
    if (
        _RETRIEVAL_RE.search(normalized)
        or (_DIRECT_DOC_HINT_RE.search(normalized) and "search" in normalized.casefold())
        or (has_named_doc_target and _DOC_ACTION_RE.search(normalized))
    ):
        slices.append(
            TaskDecompositionSlice(
                kind="indexed_document_search",
                executor="rag_worker",
                description="Search indexed documents and preserve grounded evidence.",
            )
        )
    if _DATA_ANALYSIS_RE.search(normalized):
        slices.append(
            TaskDecompositionSlice(
                kind="tabular_analysis",
                executor="data_analyst",
                description="Analyze structured tabular or workbook data.",
            )
        )
    return tuple(slices)


def is_mixed_utility_retrieval_request(
    query: str,
    *,
    session_metadata: Mapping[str, Any] | None = None,
) -> bool:
    kinds = {item.kind for item in detect_task_slices(query, session_metadata=session_metadata)}
    return {"utility_calculation", "indexed_document_search"}.issubset(kinds)


def decide_task_decomposition(
    query: str,
    *,
    current_agent: str,
    route: str = "AGENT",
    suggested_agent: str = "",
    session_metadata: Mapping[str, Any] | None = None,
    explicit_override: bool = False,
) -> TaskDecompositionDecision:
    original_agent = str(current_agent or "").strip().lower()
    normalized_route = str(route or "").strip().upper()
    if explicit_override or normalized_route != "AGENT":
        return TaskDecompositionDecision(
            is_mixed_intent=False,
            selected_agent=original_agent,
            reason="preserve_explicit_or_non_agent_route",
            original_agent=original_agent,
        )

    slices = detect_task_slices(query, session_metadata=session_metadata)
    executors = {item.executor for item in slices}
    if len(executors) < 2:
        return TaskDecompositionDecision(
            is_mixed_intent=False,
            selected_agent=original_agent,
            reason="single_or_no_task_slice",
            slices=slices,
            original_agent=original_agent,
        )

    normalized = _normalized_query(query)
    if is_clause_policy_workflow(normalized, session_metadata=session_metadata):
        return TaskDecompositionDecision(
            is_mixed_intent=True,
            selected_agent="coordinator",
            route_kind="coordinator",
            reason="clause_redline_policy_workflow_requires_task_graph",
            slices=slices,
            original_agent=original_agent,
        )
    if _BROAD_OR_STAGED_RE.search(normalized) or len(slices) > 3:
        orchestration_agent = "research_coordinator" if original_agent == "research_coordinator" else "coordinator"
        return TaskDecompositionDecision(
            is_mixed_intent=True,
            selected_agent=orchestration_agent,
            route_kind="coordinator",
            reason="mixed_intent_requires_orchestration",
            slices=slices,
            original_agent=original_agent,
        )

    # V1 keeps simple mixed requests on direct tools rather than introducing a
    # second multi-agent execution path for general.
    return TaskDecompositionDecision(
        is_mixed_intent=True,
        selected_agent="general",
        route_kind="general_direct",
        reason="simple_independent_direct_tool_slices",
        slices=slices,
        original_agent=original_agent or str(suggested_agent or "").strip().lower(),
    )


__all__ = [
    "TaskDecompositionDecision",
    "TaskDecompositionSlice",
    "build_planner_input_packet",
    "decide_task_decomposition",
    "detect_task_slices",
    "is_mixed_utility_retrieval_request",
    "is_clause_policy_workflow",
]
