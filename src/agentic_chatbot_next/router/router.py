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
from agentic_chatbot_next.router.mcp_intent import mcp_intent_detected
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
_GRAPH_METADATA_INTENT_RE = re.compile(
    r"\b(?:relationships?|entities|entity|multi[-\s]?hop|connected|dependencies|network|evidence|"
    r"vendors?|suppliers?|risks?|approvals?|causes?|causal|outcomes?|source[-\s]?resolve|cross[-\s]?document)\b",
    re.IGNORECASE,
)
_GRAPH_ADMIN_RE = re.compile(
    r"\b(?:create|build|index|import|refresh|rebuild|update|delete|remove)\b.*\b(?:knowledge\s+)?graph\b"
    r"|\bgraph\s+(?:build|index|import|refresh|rebuild|delete|remove)\b",
    re.IGNORECASE,
)
_DEEP_RESEARCH_RE = re.compile(
    r"\b("
    r"deep\s+research|deep\s+rag|long[-\s]?running\s+research|multi[-\s]?hop|"
    r"identify\s+all\s+(?:documents|docs|files)|"
    r"which\s+(?:documents|docs|files)\s+.*\b(?:mention|contain|discuss|describe|cover)\b|"
    r"(?:documents|docs|files)\s+that\s+(?:mention|contain|discuss|describe|cover)|"
    r"organize\s+(?:this\s+|the\s+)?(?:repository|corpus|documents|docs|files)|"
    r"synthesi[sz]e\s+(?:across|over)\s+(?:all\s+|the\s+)?(?:documents|docs|files|corpus|repository)|"
    r"across\s+(?:all\s+|the\s+whole\s+|the\s+entire\s+)?(?:documents|docs|files|corpus|repository)|"
    r"whole\s+(?:repository|corpus|knowledge\s*base)|entire\s+(?:repository|corpus|knowledge\s*base)|"
    r"defense\s+(?:program|repository|corpus)|large\s+repository\s+of\s+(?:documents|files)|"
    r"major\s+subsystems?|corpus[-\s]?scale"
    r")\b",
    re.IGNORECASE,
)
_MERMAID_DIAGRAM_RE = re.compile(
    r"\b(mermaid|diagram|flowchart|sequence\s+diagram|state\s+diagram|class\s+diagram|er\s+diagram|erd|gantt)\b",
    re.IGNORECASE,
)
_SKILL_SEARCH_RE = re.compile(
    r"\b(search|find|load|use)\s+(?:your\s+)?skills?\b|\bskills?\s+(?:search|guidance)\b|\bsearch_skills\b",
    re.IGNORECASE,
)
_DIAGRAM_ORCHESTRATION_RE = re.compile(
    r"\b(first|then|after|before|multi[-\s]?step|research|rag|grounded|knowledge\s+base|default\s+kb|"
    r"indexed\s+(?:documents|knowledge\s+base)|cite|cites?|citations?|sources?)\b",
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
    research_agent: str = "research_coordinator"
    data_analyst_agent: str = "data_analyst"
    rag_agent: str = "rag_worker"
    graph_agent: str = "graph_manager"
    suggested_agents: tuple[str, ...] = ("coordinator", "research_coordinator", "data_analyst", "rag_worker", "graph_manager")
    descriptions: dict[str, str] = field(default_factory=dict)


def is_requirements_inventory_request(user_text: str) -> bool:
    return bool(_REQUIREMENTS_EXTRACTION_RE.search(str(user_text or "")))


def build_router_targets(registry: Any | None = None) -> RouterTargets:
    if registry is None:
        return RouterTargets(
            descriptions={
                "coordinator": "Manager-only role for explicit worker orchestration.",
                "research_coordinator": "Manager role for long-running deep research over indexed corpora.",
                "data_analyst": "Tabular data analysis specialist using sandboxed Python tools.",
                "rag_worker": "Grounded document worker that returns the stable RAG contract.",
                "graph_manager": "Managed graph retrieval and source-planning specialist.",
            }
        )

    default_agent = getattr(registry, "get_default_agent_name", lambda: "general")()
    basic_agent = getattr(registry, "get_basic_agent_name", lambda: "basic")()
    coordinator_agent = getattr(registry, "get_manager_agent_name", lambda: "coordinator")()
    research_agent = getattr(registry, "get_research_agent_name", lambda: "research_coordinator")()
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
    for name in (coordinator_agent, research_agent, data_analyst_agent, rag_agent, graph_agent, default_agent):
        if name and name != basic_agent and name not in suggested:
            suggested.append(name)
    return RouterTargets(
        default_agent=default_agent or "general",
        basic_agent=basic_agent or "basic",
        coordinator_agent=coordinator_agent or "coordinator",
        research_agent=research_agent or "research_coordinator",
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


def _metadata_graph_ids(metadata: dict[str, Any]) -> list[str]:
    values: list[str] = []
    route_context = metadata.get("route_context") if isinstance(metadata.get("route_context"), dict) else {}
    for source in (metadata, route_context):
        for key in (
            "graph_id",
            "active_graph_id",
            "selected_graph_id",
            "active_graph_ids",
            "selected_graph_ids",
            "graph_ids",
            "planned_graph_ids",
        ):
            value = source.get(key)
            if isinstance(value, str):
                clean = value.strip()
                if clean:
                    values.append(clean)
            elif isinstance(value, list):
                values.extend(str(item).strip() for item in value if str(item).strip())
    return list(dict.fromkeys(values))


def is_graph_metadata_request(user_text: str, session_metadata: dict[str, Any] | None = None) -> bool:
    metadata = dict(session_metadata or {})
    if not _metadata_graph_ids(metadata):
        return False
    if _GRAPH_ADMIN_RE.search(str(user_text or "")):
        return False
    if classify_inventory_query(user_text) != INVENTORY_QUERY_NONE:
        return False
    return bool(is_graph_retrieval_request(user_text) or _GRAPH_METADATA_INTENT_RE.search(str(user_text or "")))


def is_deep_research_request(user_text: str) -> bool:
    text = str(user_text or "")
    if not text.strip():
        return False
    return bool(_DEEP_RESEARCH_RE.search(text))


def is_mermaid_diagram_workflow_request(user_text: str) -> bool:
    text = str(user_text or "")
    if not _MERMAID_DIAGRAM_RE.search(text):
        return False
    return bool(_SKILL_SEARCH_RE.search(text) or _DIAGRAM_ORCHESTRATION_RE.search(text))


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

    if mcp_intent_detected(metadata):
        mcp_intent = dict(metadata.get("mcp_intent") or {})
        reasons.append("mcp_intent")
        trigger = str(mcp_intent.get("trigger") or "").strip()
        if trigger:
            reasons.append(f"mcp_intent_{trigger}")
        return RouterDecision(
            route="AGENT",
            confidence=0.94,
            reasons=reasons,
            suggested_agent=targets.default_agent,
            router_evidence={"mcp_intent": mcp_intent},
            semantic_contract=build_deterministic_semantic_contract(
                user_text=routing_text,
                route="AGENT",
                suggested_agent=targets.default_agent,
                confidence=0.94,
                reasoning="; ".join(reasons),
                session_metadata=metadata,
            ),
        )

    if has_attachments:
        suggested = ""
        if is_mermaid_diagram_workflow_request(routing_text):
            suggested = targets.default_agent
        elif is_requirements_inventory_request(routing_text):
            suggested = targets.default_agent
        elif compiled.data_analysis_intent.matches(routing_text, normalized):
            suggested = targets.data_analyst_agent
        elif is_graph_metadata_request(routing_text, metadata) or is_graph_retrieval_request(routing_text):
            suggested = targets.graph_agent
        elif inventory_query_type != INVENTORY_QUERY_NONE:
            suggested = targets.default_agent
        elif is_deep_research_request(routing_text):
            suggested = targets.research_agent
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

    if is_mermaid_diagram_workflow_request(routing_text):
        reasons.append("mermaid_diagram_workflow")
        return RouterDecision(
            route="AGENT",
            confidence=0.93,
            reasons=reasons,
            suggested_agent=targets.default_agent,
            semantic_contract=build_deterministic_semantic_contract(
                user_text=routing_text,
                route="AGENT",
                suggested_agent=targets.default_agent,
                confidence=0.93,
                reasoning="; ".join(reasons),
                session_metadata=metadata,
            ),
        )

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

    if is_graph_metadata_request(routing_text, metadata) or is_graph_retrieval_request(routing_text):
        reasons.append("graph_metadata_intent" if is_graph_metadata_request(routing_text, metadata) else "graph_retrieval_intent")
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
        suggested_agent = targets.research_agent if is_deep_research_request(routing_text) else targets.coordinator_agent
        return RouterDecision(
            route="AGENT",
            confidence=0.95,
            reasons=reasons,
            suggested_agent=suggested_agent,
            semantic_contract=build_deterministic_semantic_contract(
                user_text=routing_text,
                route="AGENT",
                suggested_agent=suggested_agent,
                confidence=0.95,
                reasoning="active_doc_focus_followup",
                session_metadata=metadata,
            ),
        )

    if is_deep_research_request(routing_text):
        reasons.append("deep_research_campaign")
        suggested_agent = targets.research_agent
    elif compiled.coordinator_campaign_intent.matches(routing_text, normalized):
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
