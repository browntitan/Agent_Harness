from __future__ import annotations

import logging
import re
from typing import Any, Iterable

from pydantic import BaseModel, Field, field_validator

from agentic_chatbot_next.providers.circuit_breaker import CircuitBreakerOpenError
from agentic_chatbot_next.rag.inventory import (
    INVENTORY_QUERY_GRAPH_INDEXES,
    INVENTORY_QUERY_NONE,
    classify_inventory_query,
    inventory_answer_origin,
    inventory_query_requests_grounded_analysis,
    inventory_scope_kind,
    is_authoritative_inventory_query_type,
)
from agentic_chatbot_next.rag.hints import normalize_structured_query
from agentic_chatbot_next.router.patterns import (
    CompiledRouterPatterns,
    load_router_patterns,
    normalize_router_text,
    patterns_path_from_settings,
)
from agentic_chatbot_next.router.semantic import (
    SemanticRoutingContract,
    build_deterministic_semantic_contract,
    default_agent_for_semantic_contract,
    obvious_basic_turn,
    select_requested_collection_id,
    semantic_contract_requires_agent,
    has_visible_uploads,
    visible_kb_collection_ids,
)
from agentic_chatbot_next.utils.json_utils import extract_json
from agentic_chatbot_next.router.router import (
    RouterDecision,
    build_router_targets,
    is_deep_research_request,
    is_graph_retrieval_request,
    is_requirements_inventory_request,
    route_message,
)

logger = logging.getLogger(__name__)

_ROUTER_SYSTEM_PROMPT = """\
You are a message router for an enterprise document-intelligence assistant.

## Your task
Classify the incoming user message as either BASIC or AGENT.

### Route to AGENT when the message:
- Asks about documents, contracts, policies, requirements, or procedures
- Requests search, retrieval, citations, or evidence
- Involves comparison or analysis of multiple documents
- Asks what documents are indexed or available in the knowledge base
- Asks what documents are in a specific knowledge base collection such as `default`
- Asks which knowledge base collections or KBs are available to this chat
- Asks what documents are available to this chat right now or what documents we have access to
- Asks what knowledge graphs or graph indexes are available, accessible, or currently exist
- Asks to use, query, search, or inspect a knowledge graph for entities, relationships, dependencies, source planning, or graph-backed evidence
- Asks for a fact, date, requirement, decision, approval, milestone, or summary that must be looked up from the knowledge base, a named collection, or uploaded files, even if the user does not explicitly say "search", "retrieve", or "cite"
- Is a high-stakes domain (legal, medical, financial, compliance, security)
- Requires multi-step reasoning or tool use
- Contains file attachments or references to uploaded documents
- Involves data analysis, spreadsheets, Excel, CSV files, statistics, or pandas operations

### Route to BASIC when the message:
- Is a greeting or small talk
- Asks for general-knowledge information not tied to a specific document
- Is a simple conversational follow-up that was already answered

## Also suggest the best starting runtime agent
Choose from the runtime agents listed below, or return an empty string when the default top-level
agent is sufficient:
{agent_options}

Agent suggestion guidance:
- Suggest `general` for KB inventory, graph inventory, and access questions
- Suggest `general` for requirements extraction, shall-statement inventories, FAR/DFARS clause obligations, and mandatory-language harvesting from prose documents
- Suggest `graph_manager` for graph-backed evidence, GraphRAG, graph relationship, entity network, dependency, source-planning, and named graph query requests
- Suggest `rag_worker` for direct grounded questions answered from a specific document or focused KB lookup
- Suggest `research_coordinator` for deep research, multi-hop, repository organization, corpus-wide synthesis, or long-running document research campaigns
- Suggest `coordinator` for broad research campaigns, corpus-wide document discovery, multi-step investigation, or structured `Goal/Context/Deliverable` prompts that ask for a list of relevant documents
- Suggest `coordinator` for follow-up requests that ask to summarize or explain the candidate documents already identified earlier in the conversation
- Suggest `data_analyst` for spreadsheet, CSV, workbook, or tabular-analysis requests

Return additional semantic routing fields:
- `requires_external_evidence`: true when the answer must come from KB documents, uploads, or prior active-document scope rather than parametric chat knowledge
- `answer_origin`: one of `parametric`, `conversation`, `retrieval`, or `ambiguous`
- `requested_scope_kind`: one of `knowledge_base`, `uploads`, `active_doc_focus`, `session_access`, `graph_indexes`, or `none`
- `requested_collection_id`: copy the exact collection id from the visible collection list when the user clearly names or refers to one; otherwise return `""`

Graph inventory rules:
- For plain graph availability questions, set `route="AGENT"`, `suggested_agent="general"`, `answer_origin="parametric"`, `requires_external_evidence=false`, and `requested_scope_kind="graph_indexes"`
- For graph-backed evidence or relationship questions, set `route="AGENT"`, `suggested_agent="graph_manager"`, `answer_origin="retrieval"`, `requires_external_evidence=true`, and `requested_scope_kind="graph_indexes"`
- Do not turn graph availability or access questions into knowledge-base retrieval

Authoritative inventory rules:
- For plain KB collection inventory, KB file inventory, and session-access inventory questions, keep `route="AGENT"` and `suggested_agent="general"`
- Use `requested_scope_kind="knowledge_base"` for KB collection/file inventory and `requested_scope_kind="session_access"` for session-access inventory
- For these plain inventory questions, set `requires_external_evidence=false` and `answer_origin="conversation"`
- Do not turn pure availability, listing, or access questions into grounded retrieval unless the user explicitly asks for evidence, citations, analysis, comparison, or synthesis

When a visible collection list is provided:
- prefer selecting from that closed set instead of inventing a collection name
- if the user clearly asks about a named collection, set `requested_collection_id` and route to `AGENT`
- if multiple collections are visible and the user asks a KB/content question without clearly picking one, set `answer_origin="ambiguous"` and route to `AGENT`
"""

_ROUTER_HUMAN_TEMPLATE = """\
Research packet:
{history_summary}

Deterministic prior:
{deterministic_prior}

Visible KB collections:
{visible_collections}

Active KB collection: {active_collection_id}
Has visible uploads: {has_uploads}
Pending clarification reason: {pending_clarification_reason}
Active document focus available: {active_doc_focus}

Current user message:
{user_text}
"""


class LLMRouterOutput(BaseModel):
    route: str = Field(..., description="'BASIC' or 'AGENT'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Routing confidence 0.0-1.0")
    reasoning: str = Field(..., description="One-sentence explanation of the routing decision")
    suggested_agent: str = Field(
        default="",
        description="Best starting runtime agent from the listed options, or '' for the default top-level agent.",
    )
    requires_external_evidence: bool = Field(
        default=False,
        description="Whether the answer depends on KB/uploads/active-doc evidence rather than parametric chat knowledge.",
    )
    answer_origin: str = Field(
        default="parametric",
        description="One of parametric, conversation, retrieval, or ambiguous.",
    )
    requested_scope_kind: str = Field(
        default="none",
        description="One of knowledge_base, uploads, active_doc_focus, session_access, graph_indexes, or none.",
    )
    requested_collection_id: str = Field(
        default="",
        description="Exact requested collection id chosen from the visible collection list, or ''.",
    )

    @field_validator("route")
    @classmethod
    def _validate_route(cls, value: str) -> str:
        upper = value.strip().upper()
        if upper not in {"BASIC", "AGENT"}:
            raise ValueError(f"route must be 'BASIC' or 'AGENT', got {value!r}")
        return upper

    @field_validator("suggested_agent")
    @classmethod
    def _validate_suggested_agent(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("answer_origin")
    @classmethod
    def _validate_answer_origin(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        return normalized if normalized in {"parametric", "conversation", "retrieval", "ambiguous"} else "parametric"

    @field_validator("requested_scope_kind")
    @classmethod
    def _validate_requested_scope_kind(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        return normalized if normalized in {"knowledge_base", "uploads", "active_doc_focus", "session_access", "graph_indexes", "none"} else "none"


def _semantic_contract_from_llm_output(
    output: LLMRouterOutput,
    *,
    user_text: str,
    session_metadata: dict[str, Any] | None,
) -> SemanticRoutingContract:
    metadata = dict(session_metadata or {})
    inventory_query_type = classify_inventory_query(user_text)
    inventory_metadata_only = (
        is_authoritative_inventory_query_type(inventory_query_type)
        and not inventory_query_requests_grounded_analysis(user_text, query_type=inventory_query_type)
    )
    visible_collections = visible_kb_collection_ids(metadata)
    requested_collection_id = (
        select_requested_collection_id(output.requested_collection_id, visible_collections)
        if output.requested_collection_id
        else ""
    )
    if not requested_collection_id:
        requested_collection_id = select_requested_collection_id(user_text, visible_collections)
    requested_scope_kind = output.requested_scope_kind
    answer_origin = output.answer_origin
    requires_external_evidence = bool(output.requires_external_evidence)
    if inventory_metadata_only:
        requested_scope_kind = inventory_scope_kind(inventory_query_type)
        answer_origin = inventory_answer_origin(inventory_query_type)
        requires_external_evidence = False
    if requested_collection_id and requested_scope_kind == "none":
        requested_scope_kind = "knowledge_base"
    if requested_scope_kind in {"knowledge_base", "uploads", "active_doc_focus", "session_access", "graph_indexes"} and not inventory_metadata_only:
        requires_external_evidence = True
    if requires_external_evidence and answer_origin in {"parametric", "conversation"}:
        answer_origin = "retrieval"
    return SemanticRoutingContract(
        route=output.route,
        suggested_agent=output.suggested_agent,
        requires_external_evidence=requires_external_evidence,
        answer_origin=answer_origin,
        requested_scope_kind=requested_scope_kind,
        requested_collection_id=requested_collection_id,
        confidence=output.confidence,
        reasoning=output.reasoning,
    )


def _agent_options_text(valid_suggested_agents: Iterable[str], descriptions: dict[str, str]) -> str:
    lines = ["- (empty string) - use the default top-level agent"]
    for agent_name in valid_suggested_agents:
        if not agent_name:
            continue
        description = descriptions.get(agent_name, "").strip() or "runtime specialist"
        lines.append(f"- {agent_name} - {description}")
    return "\n".join(lines)


def _sanitize_suggested_agent(value: str, valid_suggested_agents: Iterable[str]) -> str:
    valid = {str(item).strip().lower() for item in valid_suggested_agents if str(item).strip()}
    clean = str(value or "").strip().lower()
    return clean if clean in valid else ""


def route_turn(
    settings: object,
    providers: object,
    *,
    user_text: str,
    has_attachments: bool,
    history_summary: str = "",
    force_agent: bool = False,
    registry: Any | None = None,
    session_id: str = "",
    session_metadata: dict[str, Any] | None = None,
):
    routing_text = normalize_structured_query(user_text) or str(user_text or "")
    patterns = load_router_patterns(patterns_path_from_settings(settings))
    if not bool(getattr(settings, "llm_router_enabled", True)):
        return route_message(
            routing_text,
            has_attachments=has_attachments,
            explicit_force_agent=force_agent,
            registry=registry,
            patterns=patterns,
            session_metadata=session_metadata,
        )

    router_mode = str(getattr(settings, "llm_router_mode", "hybrid") or "hybrid").strip().lower()
    if router_mode == "llm_only":
        return route_message_llm_only(
            routing_text,
            has_attachments=has_attachments,
            judge_llm=getattr(providers, "judge"),
            history_summary=history_summary,
            explicit_force_agent=force_agent,
            registry=registry,
            patterns=patterns,
            session_id=session_id,
            session_metadata=session_metadata,
        )
    return route_message_hybrid(
        routing_text,
        has_attachments=has_attachments,
        judge_llm=getattr(providers, "judge"),
        history_summary=history_summary,
        explicit_force_agent=force_agent,
        llm_confidence_threshold=float(getattr(settings, "llm_router_confidence_threshold", 0.70)),
        registry=registry,
        patterns=patterns,
        session_id=session_id,
        session_metadata=session_metadata,
    )


__all__ = ["route_turn"]


def _deterministic_fast_path(
    user_text: str,
    *,
    has_attachments: bool,
    explicit_force_agent: bool,
    registry: Any | None,
    patterns: CompiledRouterPatterns | None,
    session_metadata: dict[str, Any] | None = None,
) -> RouterDecision | None:
    targets = build_router_targets(registry)
    inventory_query_type = classify_inventory_query(user_text)
    metadata = dict(session_metadata or {})

    if explicit_force_agent:
        return RouterDecision(
            route="AGENT",
            confidence=1.0,
            reasons=["explicit_force_agent"],
            suggested_agent=(
                targets.default_agent
                if inventory_query_type != INVENTORY_QUERY_NONE
                else (
                    targets.rag_agent
                    if patterns is not None
                    and patterns.rag_grounding_intent.matches(user_text, normalize_router_text(user_text))
                    else ""
                )
            ),
            router_method="deterministic",
            semantic_contract=build_deterministic_semantic_contract(
                user_text=user_text,
                route="AGENT",
                suggested_agent=(
                    targets.default_agent
                    if inventory_query_type != INVENTORY_QUERY_NONE
                    else (
                        targets.rag_agent
                        if patterns is not None
                        and patterns.rag_grounding_intent.matches(user_text, normalize_router_text(user_text))
                        else ""
                    )
                ),
                confidence=1.0,
                reasoning="explicit_force_agent",
                session_metadata=metadata,
            ),
        )

    if has_attachments:
        suggested = ""
        if patterns is not None:
            normalized = normalize_router_text(user_text)
            if is_requirements_inventory_request(user_text):
                suggested = targets.default_agent
            elif patterns.data_analysis_intent.matches(user_text, normalized):
                suggested = targets.data_analyst_agent
            elif inventory_query_type != INVENTORY_QUERY_NONE:
                suggested = targets.default_agent
            elif is_deep_research_request(user_text):
                suggested = targets.research_agent
            elif patterns.coordinator_campaign_intent.matches(user_text, normalized):
                suggested = targets.coordinator_agent
            elif patterns.rag_grounding_intent.matches(user_text, normalized):
                suggested = targets.rag_agent
        return RouterDecision(
            route="AGENT",
            confidence=1.0,
            reasons=["attachments_present"],
            suggested_agent=suggested,
            router_method="deterministic",
            semantic_contract=build_deterministic_semantic_contract(
                user_text=user_text,
                route="AGENT",
                suggested_agent=suggested,
                confidence=1.0,
                reasoning="attachments_present",
                session_metadata=metadata,
            ),
        )
    if is_requirements_inventory_request(user_text):
        return RouterDecision(
            route="AGENT",
            confidence=0.92,
            reasons=["requirements_inventory_intent"],
            suggested_agent=targets.default_agent,
            router_method="deterministic",
            semantic_contract=build_deterministic_semantic_contract(
                user_text=user_text,
                route="AGENT",
                suggested_agent=targets.default_agent,
                confidence=0.92,
                reasoning="requirements_inventory_intent",
                session_metadata=metadata,
            ),
        )
    if is_graph_retrieval_request(user_text):
        return RouterDecision(
            route="AGENT",
            confidence=0.92,
            reasons=["graph_retrieval_intent"],
            suggested_agent=targets.graph_agent,
            router_method="deterministic",
            semantic_contract=build_deterministic_semantic_contract(
                user_text=user_text,
                route="AGENT",
                suggested_agent=targets.graph_agent,
                confidence=0.92,
                reasoning="graph_retrieval_intent",
                session_metadata=metadata,
            ),
        )
    if inventory_query_type == INVENTORY_QUERY_GRAPH_INDEXES:
        return RouterDecision(
            route="AGENT",
            confidence=0.90,
            reasons=["graph_inventory_intent"],
            suggested_agent=targets.default_agent,
            router_method="deterministic",
            semantic_contract=build_deterministic_semantic_contract(
                user_text=user_text,
                route="AGENT",
                suggested_agent=targets.default_agent,
                confidence=0.90,
                reasoning="graph_inventory_intent",
                session_metadata=metadata,
            ),
        )
    if obvious_basic_turn(user_text):
        return RouterDecision(
            route="BASIC",
            confidence=0.95,
            reasons=["obvious_small_talk"],
            router_method="deterministic",
            semantic_contract=build_deterministic_semantic_contract(
                user_text=user_text,
                route="BASIC",
                confidence=0.95,
                reasoning="obvious_small_talk",
                session_metadata=metadata,
            ),
        )
    return None


def _deterministic_fallback(
    user_text: str,
    *,
    has_attachments: bool,
    registry: Any | None,
    patterns: CompiledRouterPatterns | None,
    reason: str,
    router_method: str,
    session_metadata: dict[str, Any] | None = None,
) -> RouterDecision:
    deterministic = route_message(
        user_text,
        has_attachments=has_attachments,
        explicit_force_agent=False,
        registry=registry,
        patterns=patterns,
        session_metadata=session_metadata,
    )
    contract = SemanticRoutingContract.from_value(
        getattr(deterministic, "semantic_contract", None),
        default_route=deterministic.route,
        default_confidence=deterministic.confidence,
        default_reasoning="; ".join(deterministic.reasons),
        default_suggested_agent=deterministic.suggested_agent,
    )
    route = deterministic.route
    suggested_agent = deterministic.suggested_agent
    reasons = list(deterministic.reasons)
    router_evidence = {
        "fallback_reason": reason,
        "deterministic_reasons": list(deterministic.reasons),
        "fallback_router_method": router_method,
    }
    if route == "BASIC" and not obvious_basic_turn(user_text):
        route = "AGENT"
        suggested_agent = default_agent_for_semantic_contract(
            contract,
            fallback_suggested_agent=deterministic.suggested_agent,
        )
        reasons.append("llm_router_unavailable_promoted_to_agent")
        router_evidence["basic_candidate_upgraded"] = True
        contract = SemanticRoutingContract.from_value(
            {
                **contract.to_dict(),
                "route": "AGENT",
                "suggested_agent": suggested_agent,
                "reasoning": contract.reasoning or reason,
            },
            default_route="AGENT",
            default_confidence=deterministic.confidence,
            default_reasoning=reason,
            default_suggested_agent=suggested_agent,
        )
    return RouterDecision(
        route=route,
        confidence=deterministic.confidence,
        reasons=reasons + [reason] if reason not in reasons else reasons,
        suggested_agent=suggested_agent,
        router_method=router_method,
        router_evidence=router_evidence,
        semantic_contract=contract,
    )


def _should_defer_deterministic_to_llm(
    user_text: str,
    *,
    deterministic: RouterDecision,
    patterns: CompiledRouterPatterns | None,
) -> bool:
    if deterministic.route != "AGENT":
        return False
    if re.search(r"\bgoal\s*:", user_text, flags=re.I) and re.search(r"\bdeliverable\s*:", user_text, flags=re.I):
        return True
    if any(
        reason in set(deterministic.reasons)
        for reason in (
            "session_access_inventory_intent",
            "kb_inventory_intent",
            "kb_collection_inventory_intent",
            "graph_inventory_intent",
            "data_analysis_intent",
            "active_doc_focus_followup",
            "requirements_inventory_intent",
        )
    ):
        return False
    if any(reason in set(deterministic.reasons) for reason in ("citation_or_grounding_requested", "document_grounding_intent")):
        return True
    normalized = normalize_router_text(user_text)
    if patterns is not None and patterns.coordinator_campaign_intent.matches(user_text, normalized):
        return True
    return bool(
        re.search(
            r"\b("
            r"investigate|identify\s+documents|identify\s+all\s+documents|"
            r"list\s+of\s+(?:potential\s+)?documents|"
            r"documents\s+that\s+(?:discuss|describe|cover|contain)|"
            r"search\s+across\s+(?:the\s+)?documents"
            r")\b",
            user_text,
            flags=re.I,
        )
    )


def route_message_hybrid(
    user_text: str,
    *,
    has_attachments: bool,
    judge_llm: Any,
    history_summary: str = "",
    explicit_force_agent: bool = False,
    llm_confidence_threshold: float = 0.70,
    registry: Any | None = None,
    patterns: CompiledRouterPatterns | None = None,
    session_id: str = "",
    session_metadata: dict[str, Any] | None = None,
) -> RouterDecision:
    fast_path = _deterministic_fast_path(
        user_text,
        has_attachments=has_attachments,
        explicit_force_agent=explicit_force_agent,
        registry=registry,
        patterns=patterns,
        session_metadata=session_metadata,
    )
    if fast_path is not None:
        return fast_path

    deterministic = route_message(
        user_text,
        has_attachments=has_attachments,
        explicit_force_agent=False,
        registry=registry,
        patterns=patterns,
        session_metadata=session_metadata,
    )
    contract = SemanticRoutingContract.from_value(
        getattr(deterministic, "semantic_contract", None),
        default_route=deterministic.route,
        default_confidence=deterministic.confidence,
        default_reasoning="; ".join(deterministic.reasons),
        default_suggested_agent=deterministic.suggested_agent,
    )
    threshold = max(0.0, min(1.0, float(llm_confidence_threshold)))
    if (
        deterministic.confidence >= threshold
        and not semantic_contract_requires_agent(contract)
        and not _should_defer_deterministic_to_llm(
            user_text,
            deterministic=deterministic,
            patterns=patterns,
        )
    ):
        return deterministic

    targets = build_router_targets(registry)
    try:
        return _call_llm_router(
            judge_llm,
            user_text=user_text,
            history_summary=history_summary,
            valid_suggested_agents=targets.suggested_agents,
            descriptions=targets.descriptions,
            session_id=session_id,
            session_metadata=session_metadata,
            deterministic=deterministic,
            router_method="llm",
        )
    except CircuitBreakerOpenError:
        logger.warning("LLM router circuit is open; falling back to deterministic route.")
        return _deterministic_fallback(
            user_text,
            has_attachments=has_attachments,
            registry=registry,
            patterns=patterns,
            reason="llm_router_circuit_open",
            router_method="llm_circuit_fallback",
            session_metadata=session_metadata,
        )
    except Exception as exc:
        logger.warning("LLM router failed (%s); falling back to deterministic route.", exc)
        return _deterministic_fallback(
            user_text,
            has_attachments=has_attachments,
            registry=registry,
            patterns=patterns,
            reason="llm_router_failed",
            router_method="llm_fallback",
            session_metadata=session_metadata,
        )


def route_message_llm_only(
    user_text: str,
    *,
    has_attachments: bool,
    judge_llm: Any,
    history_summary: str = "",
    explicit_force_agent: bool = False,
    registry: Any | None = None,
    patterns: CompiledRouterPatterns | None = None,
    session_id: str = "",
    session_metadata: dict[str, Any] | None = None,
) -> RouterDecision:
    fast_path = _deterministic_fast_path(
        user_text,
        has_attachments=has_attachments,
        explicit_force_agent=explicit_force_agent,
        registry=registry,
        patterns=patterns,
        session_metadata=session_metadata,
    )
    if fast_path is not None:
        return fast_path

    targets = build_router_targets(registry)
    try:
        return _call_llm_router(
            judge_llm,
            user_text=user_text,
            history_summary=history_summary,
            valid_suggested_agents=targets.suggested_agents,
            descriptions=targets.descriptions,
            session_id=session_id,
            session_metadata=session_metadata,
            deterministic=route_message(
                user_text,
                has_attachments=has_attachments,
                explicit_force_agent=False,
                registry=registry,
                patterns=patterns,
                session_metadata=session_metadata,
            ),
            router_method="llm",
        )
    except CircuitBreakerOpenError:
        logger.warning("LLM router circuit is open; falling back to deterministic route.")
        return _deterministic_fallback(
            user_text,
            has_attachments=has_attachments,
            registry=registry,
            patterns=patterns,
            reason="llm_router_circuit_open",
            router_method="llm_circuit_fallback",
            session_metadata=session_metadata,
        )
    except Exception as exc:
        logger.warning("LLM router failed (%s); falling back to deterministic route.", exc)
        return _deterministic_fallback(
            user_text,
            has_attachments=has_attachments,
            registry=registry,
            patterns=patterns,
            reason="llm_router_failed",
            router_method="llm_fallback",
            session_metadata=session_metadata,
        )


def _call_llm_router(
    judge_llm: Any,
    *,
    user_text: str,
    history_summary: str,
    valid_suggested_agents: Iterable[str],
    descriptions: dict[str, str],
    session_id: str = "",
    session_metadata: dict[str, Any] | None = None,
    deterministic: RouterDecision | None = None,
    router_method: str = "llm",
) -> RouterDecision:
    from langchain_core.messages import HumanMessage, SystemMessage
    metadata = dict(session_metadata or {})
    deterministic = deterministic or RouterDecision(
        route="BASIC",
        confidence=0.0,
        reasons=["no_deterministic_prior"],
        semantic_contract=build_deterministic_semantic_contract(
            user_text=user_text,
            route="BASIC",
            confidence=0.0,
            reasoning="no_deterministic_prior",
            session_metadata=metadata,
        ),
    )
    visible_collections = visible_kb_collection_ids(metadata)
    active_doc_focus = bool(metadata.get("active_doc_focus"))
    active_collection_id = str(metadata.get("kb_collection_id") or "").strip()
    pending_clarification_reason = str(
        dict(metadata.get("pending_clarification") or {}).get("reason") or ""
    ).strip()
    deterministic_prior = (
        f"route={deterministic.route}; "
        f"suggested_agent={str(deterministic.suggested_agent or '').strip() or '(none)'}; "
        f"confidence={deterministic.confidence:.2f}; "
        f"reasons={', '.join(deterministic.reasons) or '(none)'}"
    )

    messages = [
        SystemMessage(
            content=_ROUTER_SYSTEM_PROMPT.format(
                agent_options=_agent_options_text(valid_suggested_agents, descriptions),
            )
        ),
        HumanMessage(
            content=_ROUTER_HUMAN_TEMPLATE.format(
                history_summary=history_summary or "(no prior context)",
                deterministic_prior=deterministic_prior,
                visible_collections=", ".join(visible_collections) or "(none)",
                active_collection_id=active_collection_id or "(none)",
                has_uploads="yes" if has_visible_uploads(metadata) else "no",
                pending_clarification_reason=pending_clarification_reason or "(none)",
                active_doc_focus="yes" if active_doc_focus else "no",
                user_text=user_text,
            )
        ),
    ]

    try:
        structured_llm = judge_llm.with_structured_output(LLMRouterOutput)
        raw_result = _invoke_with_optional_config(
            structured_llm,
            messages,
            config={"metadata": {"session_id": session_id}},
        )
        if isinstance(raw_result, LLMRouterOutput):
            parsed = raw_result
        else:
            parsed = None
    except CircuitBreakerOpenError:
        raise
    except (AttributeError, NotImplementedError, Exception):
        parsed = None

    if parsed is None:
        response = _invoke_with_optional_config(
            judge_llm,
            messages,
            config={"metadata": {"session_id": session_id}},
        )
        text = getattr(response, "content", None) or str(response)
        parsed = _parse_llm_response_text(text, valid_suggested_agents)

    contract = _semantic_contract_from_llm_output(
        parsed,
        user_text=user_text,
        session_metadata=metadata,
    )
    route = parsed.route
    if semantic_contract_requires_agent(contract):
        route = "AGENT"
    suggested_agent = _sanitize_suggested_agent(parsed.suggested_agent, valid_suggested_agents)
    if route == "AGENT":
        if is_requirements_inventory_request(user_text):
            suggested_agent = "general"
        elif is_deep_research_request(user_text):
            valid_agents = {str(item).strip().lower() for item in valid_suggested_agents if str(item).strip()}
            if "research_coordinator" in valid_agents:
                suggested_agent = "research_coordinator"
        suggested_agent = suggested_agent or default_agent_for_semantic_contract(
            contract,
            fallback_suggested_agent=deterministic.suggested_agent,
        )
        contract = SemanticRoutingContract.from_value(
            {
                **contract.to_dict(),
                "route": "AGENT",
                "suggested_agent": suggested_agent,
            },
            default_route="AGENT",
            default_confidence=parsed.confidence,
            default_reasoning=parsed.reasoning,
            default_suggested_agent=suggested_agent,
        )
    return RouterDecision(
        route=route,
        confidence=parsed.confidence,
        reasons=[f"llm_router: {parsed.reasoning}"],
        suggested_agent=suggested_agent,
        router_method=router_method,
        router_evidence={
            "llm_reasoning": parsed.reasoning,
            "llm_confidence": parsed.confidence,
            "llm_suggested_agent": parsed.suggested_agent,
            "visible_kb_collection_ids": visible_collections,
            "active_kb_collection_id": active_collection_id,
            "pending_clarification_reason": pending_clarification_reason,
            "deterministic_route": deterministic.route,
            "deterministic_reasons": list(deterministic.reasons),
        },
        semantic_contract=contract,
    )


def _invoke_with_optional_config(model: Any, messages: list[Any], *, config: dict[str, Any]) -> Any:
    try:
        return model.invoke(messages, config=config)
    except TypeError as exc:
        if "unexpected keyword argument 'config'" not in str(exc):
            raise
        return model.invoke(messages)


def _parse_llm_response_text(text: str, valid_suggested_agents: Iterable[str]) -> LLMRouterOutput:
    obj = extract_json(text) or {}
    route = str(obj.get("route", "")).strip().upper()
    if route not in {"BASIC", "AGENT"}:
        lower = text.lower()
        route = (
            "AGENT"
            if any(keyword in lower for keyword in ("agent", "document", "search", "rag", "retrieval"))
            else "BASIC"
        )

    confidence = float(obj.get("confidence", 0.65))
    confidence = max(0.0, min(1.0, confidence))

    suggested = _sanitize_suggested_agent(str(obj.get("suggested_agent", "")), valid_suggested_agents)

    reasoning = str(obj.get("reasoning", "parsed from text"))
    return LLMRouterOutput(
        route=route,
        confidence=confidence,
        reasoning=reasoning,
        suggested_agent=suggested,
        requires_external_evidence=bool(obj.get("requires_external_evidence", False)),
        answer_origin=str(obj.get("answer_origin", "parametric")),
        requested_scope_kind=str(obj.get("requested_scope_kind", "none")),
        requested_collection_id=str(obj.get("requested_collection_id", "")),
    )
