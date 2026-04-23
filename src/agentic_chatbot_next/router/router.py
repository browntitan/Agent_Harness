from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agentic_chatbot_next.rag.inventory import (
    INVENTORY_QUERY_GRAPH_INDEXES,
    INVENTORY_QUERY_KB_FILE,
    INVENTORY_QUERY_KB_COLLECTIONS,
    INVENTORY_QUERY_NONE,
    INVENTORY_QUERY_SESSION_ACCESS,
    classify_inventory_query,
)
from agentic_chatbot_next.rag.hints import normalize_structured_query
from agentic_chatbot_next.router.feedback_loop import build_router_decision_id
from agentic_chatbot_next.router.patterns import (
    CompiledRouterPatterns,
    load_router_patterns,
    normalize_router_text,
)
from agentic_chatbot_next.router.semantic import (
    SemanticRoutingContract,
    build_deterministic_semantic_contract,
)
from agentic_chatbot_next.runtime.doc_focus import is_active_doc_focus_followup

_REQUIREMENTS_EXTRACTION_RE = re.compile(
    r"\b(?:extract|pull|list|inventory|organize|find|harvest|export|download|return)\b"
    r".*\b(?:shall|must|requirement|requirements|obligation|obligations|clause|clauses)\b"
    r"|\b(?:shall|must)\s+statements?\b"
    r"|\brequirement\s+statements?\b"
    r"|\b(?:far|dfars)\b.*\b(?:clause|clauses|requirement|requirements|obligation|obligations)\b",
    re.IGNORECASE | re.DOTALL,
)
_GRAPH_RETRIEVAL_RE = re.compile(
    r"\b(?:knowledge\s+graph|graphrag|graph\s+rag|graph\s+index|graph\s+indexes|"
    r"use\s+(?:the\s+)?(?:knowledge\s+)?graph|query\s+(?:the\s+)?(?:knowledge\s+)?graph|"
    r"search\s+(?:the\s+)?(?:knowledge\s+)?graph|inspect\s+(?:the\s+)?(?:knowledge\s+)?graph)\b"
    r"|(?:\bgraph\b.*\b(?:relationships?|entities|entity|multi[-\s]?hop|connected|dependencies|network|evidence)\b)"
    r"|(?:\b(?:relationships?|entities|entity|multi[-\s]?hop|connected|dependencies|network)\b.*\bgraph\b)",
    re.IGNORECASE,
)
_GRAPH_ADMIN_RE = re.compile(
    r"\b(?:create|build|index|import|refresh|rebuild|update|delete|remove)\b.*\b(?:knowledge\s+)?graph\b"
    r"|\bgraph\s+(?:build|index|import|refresh|rebuild|delete|remove)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RouterDecision:
    route: str
    confidence: float
    reasons: list[str]
    suggested_agent: str = ""
    router_method: str = "deterministic"
    router_decision_id: str = field(default_factory=build_router_decision_id)
    router_evidence: dict[str, Any] = field(default_factory=dict)
    semantic_contract: SemanticRoutingContract = field(default_factory=SemanticRoutingContract)


@dataclass(frozen=True)
class RouterTargets:
    default_agent: str = "general"
    basic_agent: str = "basic"
    coordinator_agent: str = "coordinator"
    data_analyst_agent: str = "data_analyst"
    rag_agent: str = "rag_worker"
    graph_agent: str = "graph_manager"
    suggested_agents: tuple[str, ...] = ("coordinator", "data_analyst", "rag_worker", "graph_manager")
    descriptions: dict[str, str] = field(default_factory=dict)


def is_requirements_inventory_request(user_text: str) -> bool:
    return bool(_REQUIREMENTS_EXTRACTION_RE.search(str(user_text or "")))


def build_router_targets(registry: Any | None = None) -> RouterTargets:
    if registry is None:
        return RouterTargets(
            descriptions={
                "coordinator": "Manager-only role for explicit worker orchestration.",
                "data_analyst": "Tabular data analysis specialist using sandboxed Python tools.",
                "rag_worker": "Grounded document worker that returns the stable RAG contract.",
                "graph_manager": "Managed graph retrieval and source-planning specialist.",
            }
        )

    default_agent = getattr(registry, "get_default_agent_name", lambda: "general")()
    basic_agent = getattr(registry, "get_basic_agent_name", lambda: "basic")()
    coordinator_agent = getattr(registry, "get_manager_agent_name", lambda: "coordinator")()
    data_analyst_agent = getattr(registry, "get_data_analyst_agent_name", lambda: "data_analyst")()
    rag_agent = getattr(registry, "get_rag_agent_name", lambda: "rag_worker")()
    graph_agent = getattr(registry, "get_graph_agent_name", lambda: "graph_manager")()

    descriptions: dict[str, str] = {}
    suggested: list[str] = []
    list_routable = getattr(registry, "list_routable", lambda: [])
    for agent in list_routable():
        if agent.mode == "basic":
            continue
        descriptions[agent.name] = agent.description
        if agent.name not in suggested:
            suggested.append(agent.name)
    for name in (coordinator_agent, data_analyst_agent, rag_agent, graph_agent, default_agent):
        if name and name != basic_agent and name not in suggested:
            suggested.append(name)
    return RouterTargets(
        default_agent=default_agent or "general",
        basic_agent=basic_agent or "basic",
        coordinator_agent=coordinator_agent or "coordinator",
        data_analyst_agent=data_analyst_agent or "data_analyst",
        rag_agent=rag_agent or "rag_worker",
        graph_agent=graph_agent or "graph_manager",
        suggested_agents=tuple(suggested),
        descriptions=descriptions,
    )


def is_graph_retrieval_request(user_text: str) -> bool:
    if _GRAPH_ADMIN_RE.search(str(user_text or "")):
        return False
    if classify_inventory_query(user_text) != INVENTORY_QUERY_NONE:
        return False
    return bool(_GRAPH_RETRIEVAL_RE.search(str(user_text or "")))

def route_message(
    user_text: str,
    *,
    has_attachments: bool,
    explicit_force_agent: bool = False,
    registry: Any | None = None,
    patterns: CompiledRouterPatterns | None = None,
    patterns_path: str | None = None,
    session_metadata: dict[str, Any] | None = None,
) -> RouterDecision:
    targets = build_router_targets(registry)
    compiled = patterns or load_router_patterns(patterns_path)
    routing_text = normalize_structured_query(user_text) or str(user_text or "")
    normalized = normalize_router_text(routing_text)
    inventory_query_type = classify_inventory_query(routing_text)
    metadata = dict(session_metadata or {})
    reasons: list[str] = []

    if explicit_force_agent:
        return RouterDecision(
            route="AGENT",
            confidence=1.0,
            reasons=["explicit_force_agent"],
            semantic_contract=build_deterministic_semantic_contract(
                user_text=routing_text,
                route="AGENT",
                confidence=1.0,
                reasoning="explicit_force_agent",
                session_metadata=metadata,
            ),
        )

    if has_attachments:
        suggested = ""
        if is_requirements_inventory_request(routing_text):
            suggested = targets.default_agent
        elif compiled.data_analysis_intent.matches(routing_text, normalized):
            suggested = targets.data_analyst_agent
        elif inventory_query_type != INVENTORY_QUERY_NONE:
            suggested = targets.default_agent
        elif compiled.coordinator_campaign_intent.matches(routing_text, normalized):
            suggested = targets.coordinator_agent
        elif compiled.rag_grounding_intent.matches(routing_text, normalized):
            suggested = targets.rag_agent
        return RouterDecision(
            route="AGENT",
            confidence=1.0,
            reasons=["attachments_present"],
            suggested_agent=suggested,
            semantic_contract=build_deterministic_semantic_contract(
                user_text=routing_text,
                route="AGENT",
                suggested_agent=suggested,
                confidence=1.0,
                reasoning="attachments_present",
                session_metadata=metadata,
            ),
        )

    if compiled.tool_or_multistep_intent.matches(routing_text, normalized):
        reasons.append("tool_or_multistep_intent")

    if compiled.citation_grounding_intent.matches(routing_text, normalized):
        reasons.append("citation_or_grounding_requested")

    if compiled.high_stakes_intent.matches(routing_text, normalized):
        reasons.append("high_stakes_topic")

    if is_requirements_inventory_request(routing_text):
        reasons.append("requirements_inventory_intent")
        return RouterDecision(
            route="AGENT",
            confidence=0.92,
            reasons=reasons,
            suggested_agent=targets.default_agent,
            semantic_contract=build_deterministic_semantic_contract(
                user_text=routing_text,
                route="AGENT",
                suggested_agent=targets.default_agent,
                confidence=0.92,
                reasoning="requirements_inventory_intent",
                session_metadata=metadata,
            ),
        )

    if compiled.data_analysis_intent.matches(routing_text, normalized):
        reasons.append("data_analysis_intent")
        return RouterDecision(
            route="AGENT",
            confidence=0.90,
            reasons=reasons,
            suggested_agent=targets.data_analyst_agent,
            semantic_contract=build_deterministic_semantic_contract(
                user_text=routing_text,
                route="AGENT",
                suggested_agent=targets.data_analyst_agent,
                confidence=0.90,
                reasoning="data_analysis_intent",
                session_metadata=metadata,
            ),
        )

    if is_graph_retrieval_request(routing_text):
        reasons.append("graph_retrieval_intent")
        return RouterDecision(
            route="AGENT",
            confidence=0.92 if len(reasons) <= 1 else 0.95,
            reasons=reasons,
            suggested_agent=targets.graph_agent,
            semantic_contract=build_deterministic_semantic_contract(
                user_text=routing_text,
                route="AGENT",
                suggested_agent=targets.graph_agent,
                confidence=0.92 if len(reasons) <= 1 else 0.95,
                reasoning="; ".join(reasons),
                session_metadata=metadata,
            ),
        )

    if inventory_query_type != INVENTORY_QUERY_NONE:
        if inventory_query_type == INVENTORY_QUERY_SESSION_ACCESS:
            reasons.append("session_access_inventory_intent")
        elif inventory_query_type == INVENTORY_QUERY_KB_FILE:
            reasons.append("kb_inventory_intent")
        elif inventory_query_type == INVENTORY_QUERY_KB_COLLECTIONS:
            reasons.append("kb_collection_inventory_intent")
        elif inventory_query_type == INVENTORY_QUERY_GRAPH_INDEXES:
            reasons.append("graph_inventory_intent")
        confidence = 0.75 if len(reasons) == 1 else 0.9
        return RouterDecision(
            route="AGENT",
            confidence=confidence,
            reasons=reasons,
            suggested_agent=targets.default_agent,
            semantic_contract=build_deterministic_semantic_contract(
                user_text=routing_text,
                route="AGENT",
                suggested_agent=targets.default_agent,
                confidence=confidence,
                reasoning="; ".join(reasons),
                session_metadata=metadata,
            ),
        )

    if is_active_doc_focus_followup(routing_text, session_metadata):
        reasons.append("active_doc_focus_followup")
        return RouterDecision(
            route="AGENT",
            confidence=0.95,
            reasons=reasons,
            suggested_agent=targets.coordinator_agent,
            semantic_contract=build_deterministic_semantic_contract(
                user_text=routing_text,
                route="AGENT",
                suggested_agent=targets.coordinator_agent,
                confidence=0.95,
                reasoning="active_doc_focus_followup",
                session_metadata=metadata,
            ),
        )

    if compiled.coordinator_campaign_intent.matches(routing_text, normalized):
        reasons.append("document_research_campaign")
        suggested_agent = targets.coordinator_agent
    elif compiled.rag_grounding_intent.matches(routing_text, normalized):
        reasons.append("document_grounding_intent")
        suggested_agent = targets.rag_agent
    else:
        suggested_agent = ""

    if len(routing_text.strip()) > 600:
        reasons.append("long_input")

    if reasons:
        confidence = 0.75 if len(reasons) == 1 else 0.9
        return RouterDecision(
            route="AGENT",
            confidence=confidence,
            reasons=reasons,
            suggested_agent=suggested_agent,
            semantic_contract=build_deterministic_semantic_contract(
                user_text=routing_text,
                route="AGENT",
                suggested_agent=suggested_agent,
                confidence=confidence,
                reasoning="; ".join(reasons),
                session_metadata=metadata,
            ),
        )

    return RouterDecision(
        route="BASIC",
        confidence=0.85,
        reasons=["general_knowledge_or_small_talk"],
        semantic_contract=build_deterministic_semantic_contract(
            user_text=routing_text,
            route="BASIC",
            confidence=0.85,
            reasoning="general_knowledge_or_small_talk",
            session_metadata=metadata,
        ),
    )
