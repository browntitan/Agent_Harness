from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from agentic_chatbot_next.graph.structured_search import StructuredSearchAdapter
from agentic_chatbot_next.rag.entity_linking import resolve_query_entities


@dataclass
class SourcePlan:
    query: str
    sources_considered: List[str] = field(default_factory=list)
    sources_chosen: List[str] = field(default_factory=list)
    graph_ids: List[str] = field(default_factory=list)
    graph_methods: List[str] = field(default_factory=list)
    preferred_doc_ids: List[str] = field(default_factory=list)
    sql_views_used: List[str] = field(default_factory=list)
    graph_shortlist: List[Dict[str, Any]] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    decomposition: Dict[str, Any] = field(default_factory=dict)
    required_hops: List[str] = field(default_factory=list)
    claim_checklist: List[Dict[str, Any]] = field(default_factory=list)
    prioritized_sections: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "sources_considered": list(self.sources_considered),
            "sources_chosen": list(self.sources_chosen),
            "graph_ids": list(self.graph_ids),
            "graph_methods": list(self.graph_methods),
            "preferred_doc_ids": list(self.preferred_doc_ids),
            "sql_views_used": list(self.sql_views_used),
            "graph_shortlist": [dict(item) for item in self.graph_shortlist],
            "reasons": list(self.reasons),
            "decomposition": dict(self.decomposition),
            "required_hops": list(self.required_hops),
            "claim_checklist": [dict(item) for item in self.claim_checklist],
            "prioritized_sections": [dict(item) for item in self.prioritized_sections],
        }

    def to_controller_hints(self) -> Dict[str, Any]:
        return {
            "planned_sources": list(self.sources_chosen),
            "planned_graph_ids": list(self.graph_ids),
            "planned_graph_methods": list(self.graph_methods),
            "preferred_doc_ids_from_sql": list(self.preferred_doc_ids),
            "source_plan_reasons": list(self.reasons),
            "canonical_entities": list(self.decomposition.get("canonical_entities") or []),
            "required_hops": list(self.required_hops),
            "claim_checklist": [dict(item) for item in self.claim_checklist],
            "prioritized_sections": [dict(item) for item in self.prioritized_sections],
        }


def _dedupe(items: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        clean = str(item or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result


def _graph_capable_query(query: str, controller_hints: Dict[str, Any]) -> bool:
    lower = str(query or "").lower()
    return bool(
        controller_hints.get("needs_multihop")
        or controller_hints.get("prefer_graph")
        or re.search(
            r"\b(relationship|relationships|depends on|connected to|cross-document|entity|entities|graph|knowledge graph|who reports to|approval chain|multi hop|multihop)\b",
            lower,
        )
        or len(re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", query or "")) >= 2
    )


def _sql_capable_query(query: str, controller_hints: Dict[str, Any]) -> bool:
    lower = str(query or "").lower()
    return bool(
        controller_hints.get("needs_structured_join")
        or controller_hints.get("prefer_sql")
        or re.search(
            r"\b(count|how many|list|show me|which|exact|status|metadata|collection|title|latest graph|existing graph|what graphs)\b",
            lower,
        )
    )


def _graph_methods_for_query(query: str, *, default_method: str = "local") -> List[str]:
    lower = str(query or "").lower()
    methods: List[str] = []
    if re.search(r"\b(summary|overview|themes|across the corpus|global)\b", lower):
        methods.append("global")
    if re.search(r"\b(drift|change|evolution|timeline)\b", lower):
        methods.append("drift")
    if re.search(r"\b(relationship|entity|entities|depends on|who|which)\b", lower):
        methods.append("local")
    if not methods:
        methods.append(default_method or "local")
    return _dedupe(methods)


def _extract_prioritized_sections(query: str) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for clause in re.findall(r"\b(?:clause|section|article)\s+([a-z0-9.]+)\b", str(query or ""), re.IGNORECASE):
        key = ("clause_number", clause.casefold())
        if key in seen:
            continue
        seen.add(key)
        sections.append({"match_type": "clause_number", "value": clause})
    for sheet in re.findall(r"\bsheet\s+([A-Za-z0-9 _-]{2,})\b", str(query or ""), re.IGNORECASE):
        key = ("sheet_name", sheet.casefold())
        if key in seen:
            continue
        seen.add(key)
        sections.append({"match_type": "sheet_name", "value": sheet.strip()})
    return sections[:8]


def plan_sources(
    query: str,
    *,
    settings: Any,
    stores: Any,
    session: Any,
    controller_hints: Dict[str, Any] | None = None,
    collection_id: str = "",
    preferred_doc_ids: Sequence[str] | None = None,
) -> SourcePlan:
    hints = dict(controller_hints or {})
    plan = SourcePlan(
        query=str(query or ""),
        sources_considered=["vector", "keyword", "graph", "sql"],
        sources_chosen=["vector", "keyword"],
    )
    structured = StructuredSearchAdapter(stores, settings=settings, session=session)

    explicit_graph_ids = _dedupe(
        [
            *[str(item) for item in (hints.get("graph_ids") or []) if str(item)],
            *[str(item) for item in (hints.get("planned_graph_ids") or []) if str(item)],
        ]
    )
    graph_methods = _dedupe(
        [
            *[str(item) for item in (hints.get("graph_query_methods") or []) if str(item)],
            *[str(item) for item in (hints.get("planned_graph_methods") or []) if str(item)],
        ]
    )
    if not graph_methods:
        graph_methods = _graph_methods_for_query(
            query,
            default_method=str(getattr(settings, "graphrag_default_query_method", "local") or "local"),
        )

    if bool(getattr(settings, "graph_source_planning_enabled", True)):
        graph_shortlist = (
            structured.search_graph_metadata(query, collection_id=collection_id, limit=4)
            if _graph_capable_query(query, hints) or explicit_graph_ids
            else []
        )
        plan.graph_shortlist = [dict(item) for item in graph_shortlist]
        if explicit_graph_ids:
            plan.graph_ids = explicit_graph_ids
            plan.graph_methods = graph_methods
            plan.sources_chosen.append("graph")
            plan.reasons.append("explicit_graph_scope")
        elif graph_shortlist:
            plan.graph_ids = _dedupe(str(item.get("graph_id") or "") for item in graph_shortlist)
            plan.graph_methods = graph_methods
            plan.sources_chosen.append("graph")
            plan.reasons.append("graph_shortlist_match")

    canonical_entities = resolve_query_entities(
        query=query,
        stores=stores,
        settings=settings,
        session=session,
        collection_id=collection_id,
        controller_hints=hints,
    )
    plan.decomposition = {
        "canonical_entities": canonical_entities,
        "preferred_sources": list(plan.sources_chosen),
        "answer_goal": str(query or ""),
    }
    plan.required_hops = [
        f"{canonical_entities[index]['canonical_name']} -> {canonical_entities[index + 1]['canonical_name']}"
        for index in range(0, max(0, len(canonical_entities) - 1))
    ][:4]
    plan.claim_checklist = [
        {
            "claim_id": f"claim_{index + 1}",
            "question": str(query or ""),
            "entity": str(entity.get("canonical_name") or ""),
            "priority": "high" if index == 0 else "medium",
        }
        for index, entity in enumerate(canonical_entities[:4])
    ]
    plan.prioritized_sections = _extract_prioritized_sections(query)

    if bool(getattr(settings, "graph_sql_enabled", True)) and _sql_capable_query(query, hints):
        doc_matches = structured.search_documents(query, collection_id=collection_id, limit=6)
        plan.preferred_doc_ids = _dedupe(
            [*(preferred_doc_ids or []), *[str(item.get("doc_id") or "") for item in doc_matches]]
        )
        if doc_matches:
            plan.sources_chosen.append("sql")
            plan.sql_views_used = structured.allowed_views()
            plan.reasons.append("structured_metadata_lookup")

    plan.sources_chosen = _dedupe(plan.sources_chosen)
    return plan


__all__ = ["SourcePlan", "plan_sources"]
