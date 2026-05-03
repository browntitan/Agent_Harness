from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from agentic_chatbot_next.agents.registry import AgentRegistry
from agentic_chatbot_next.providers.circuit_breaker import CircuitBreakerOpenError
from agentic_chatbot_next.router.patterns import clear_router_patterns_cache, load_router_patterns
from agentic_chatbot_next.router.router import RouterDecision
from agentic_chatbot_next.router.llm_router import route_turn
from agentic_chatbot_next.router.policy import choose_agent_name


def test_route_turn_suggests_data_analyst_for_csv_queries() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="Analyze this CSV and compute total revenue by region.",
        has_attachments=False,
        force_agent=False,
    )
    assert decision.route == "AGENT"
    assert decision.suggested_agent == "data_analyst"
    assert choose_agent_name(settings, decision) == "data_analyst"


def test_route_turn_suggests_data_analyst_for_sentiment_column_requests() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="Provide sentiment analysis of all of the reviews in the reviews column and add a new column with the label.",
        has_attachments=True,
        force_agent=False,
    )
    assert decision.route == "AGENT"
    assert decision.suggested_agent == "data_analyst"
    assert choose_agent_name(settings, decision) == "data_analyst"


def test_route_turn_suggests_data_analyst_for_workbook_tab_requests() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="Create a new tab that summarizes the correlation analysis between column A and column B in this workbook.",
        has_attachments=True,
        force_agent=False,
    )
    assert decision.route == "AGENT"
    assert decision.suggested_agent == "data_analyst"
    assert choose_agent_name(settings, decision) == "data_analyst"


def test_route_turn_suggests_coordinator_for_multistep_comparison() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="Compare the uploaded contracts, verify the differences, then synthesize a recommendation.",
        has_attachments=True,
        force_agent=False,
    )
    assert decision.route == "AGENT"
    assert decision.suggested_agent == "coordinator"
    assert choose_agent_name(settings, decision) == "coordinator"


def test_route_turn_suggests_rag_worker_for_grounded_document_query() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="What are the key implementation details in the architecture docs? Cite your sources.",
        has_attachments=False,
        force_agent=False,
    )
    assert decision.route == "AGENT"
    assert decision.suggested_agent == "rag_worker"


def test_route_turn_suggests_general_for_mermaid_skill_workflow() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text=(
            "Search your skills for Mermaid diagram guidance, then create a Mermaid flowchart "
            "showing how a control-panel-created skill becomes available to the general agent."
        ),
        has_attachments=False,
        force_agent=False,
    )

    assert decision.route == "AGENT"
    assert decision.suggested_agent == "general"
    assert "mermaid_diagram_workflow" in decision.reasons


def test_route_turn_suggests_general_for_grounded_mermaid_research_workflow() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text=(
            "Use the default KB, research how skills are indexed and resolved, "
            "then produce a Mermaid flowchart with citations below it."
        ),
        has_attachments=False,
        force_agent=False,
    )

    assert decision.route == "AGENT"
    assert decision.suggested_agent == "general"
    assert "mermaid_diagram_workflow" in decision.reasons


def test_route_turn_keeps_simple_mermaid_prompt_on_basic_path() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text=(
            "Make a Mermaid state diagram for a skill lifecycle: draft, active, archived, "
            "updated version, rollback target, and dependency-blocked activation."
        ),
        has_attachments=False,
        force_agent=False,
    )

    assert decision.route == "BASIC"
    assert decision.suggested_agent == ""


def test_route_turn_sends_mcp_intent_to_general_before_rag_patterns() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text=(
            "using mcp tooling do the following: Find 10 active SAM.gov contract opportunities related "
            "to artificial intelligence, AI, machine learning, or ML, posted from 04/02/2026 to 05/02/2026."
        ),
        has_attachments=False,
        force_agent=False,
        session_metadata={
            "mcp_intent": {
                "detected": True,
                "trigger": "explicit_mcp",
                "discover_query": "Find 10 active SAM.gov contract opportunities related to AI.",
                "matched_connections": [{"connection_id": "mcp_conn_sam", "display_name": "SAM.gov", "connection_slug": "sam_gov"}],
                "matched_tools": [{"registry_name": "mcp__sam_gov__search_open_contracts", "raw_tool_name": "search_open_contracts", "connection_id": "mcp_conn_sam"}],
            }
        },
    )

    assert decision.route == "AGENT"
    assert decision.suggested_agent == "general"
    assert "mcp_intent" in decision.reasons
    assert decision.router_evidence["mcp_intent"]["trigger"] == "explicit_mcp"


def test_route_turn_suggests_general_for_requirements_inventory_requests() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="Extract all shall statements from this uploaded document and export them as a CSV.",
        has_attachments=True,
        force_agent=False,
    )

    assert decision.route == "AGENT"
    assert decision.suggested_agent == "general"
    assert choose_agent_name(settings, decision) == "general"


def test_route_turn_unwraps_openwebui_context_for_requirements_inventory_requests() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    wrapped = """### Task: Respond to the user query using the provided context.

<context>
Section 4 verification context. Numeric example: 3-5 and 20%.
</context>

extract all requirements/ shall statements from the uploaded document
"""
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text=wrapped,
        has_attachments=False,
        force_agent=False,
        session_metadata={"uploaded_doc_ids": ["UPLOAD_0dc122b0e8"], "upload_collection_id": "owui-chat-1"},
    )

    assert decision.route == "AGENT"
    assert decision.suggested_agent == "general"
    assert "requirements_inventory_intent" in decision.reasons


def test_route_turn_suggests_research_coordinator_for_corpus_discovery_campaign() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="Identify all documents that have process flows outlined in them.",
        has_attachments=False,
        force_agent=False,
    )

    assert decision.route == "AGENT"
    assert decision.suggested_agent == "research_coordinator"
    assert choose_agent_name(settings, decision) == "research_coordinator"


def test_route_turn_suggests_research_coordinator_for_deep_repository_research() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="Organize this repository of documents and synthesize across all files with citations.",
        has_attachments=False,
        force_agent=False,
    )

    assert decision.route == "AGENT"
    assert decision.suggested_agent == "research_coordinator"
    assert "deep_research_campaign" in decision.reasons


def test_route_turn_uses_coordinator_for_active_doc_focus_followup() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="Can you look through the candidate documents you provided and give me a detailed summary of the major subsystems involved?",
        has_attachments=False,
        force_agent=False,
        session_metadata={
            "active_doc_focus": {
                "collection_id": "default",
                "documents": [
                    {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md"},
                    {"doc_id": "KB_control_flow", "title": "CONTROL_FLOW.md"},
                ],
            }
        },
    )

    assert decision.route == "AGENT"
    assert decision.suggested_agent == "research_coordinator"
    assert "active_doc_focus_followup" in decision.reasons


def test_route_turn_keeps_discovery_queries_off_active_doc_focus_followup_path() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="Identify more documents about subsystems in the default collection.",
        has_attachments=False,
        force_agent=False,
        session_metadata={
            "active_doc_focus": {
                "collection_id": "default",
                "documents": [
                    {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md"},
                    {"doc_id": "KB_control_flow", "title": "CONTROL_FLOW.md"},
                ],
            }
        },
    )

    assert decision.route == "AGENT"
    assert "active_doc_focus_followup" not in decision.reasons


@pytest.mark.parametrize(
    "query",
    [
        "what documents do we have access to",
        "what files do we have access to",
        "what docs do you have",
        "what knowledge bases do you have access to",
        "list out the kbs i have access to",
        "can you list out what documents we have access to in the knowledge base",
        "can you list out all of the documents in the default collection",
        "what documents are in the default collection",
        "what docs are in the KB",
        "what's indexed",
        "show me the knowledge base inventory",
    ],
)
def test_route_turn_suggests_general_for_kb_inventory_queries(query: str) -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )

    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text=query,
        has_attachments=False,
        force_agent=False,
    )

    assert decision.route == "AGENT"
    assert decision.suggested_agent == "general"
    assert choose_agent_name(settings, decision) == "general"


@pytest.mark.parametrize(
    "query",
    [
        "what knowledge bases do i have access to",
        "which knowledge bases do i have access to",
        "list out the knowledge bases i have access to",
    ],
)
def test_route_turn_keeps_kb_inventory_semantic_contract_off_retrieval_flags(query: str) -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )

    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text=query,
        has_attachments=False,
        force_agent=False,
    )

    assert decision.route == "AGENT"
    assert decision.semantic_contract.requested_scope_kind == "knowledge_base"
    assert decision.semantic_contract.answer_origin == "conversation"
    assert decision.semantic_contract.requires_external_evidence is False
    assert decision.suggested_agent == "general"


@pytest.mark.parametrize(
    "query",
    [
        "what knowledge graphs do i have available to me",
        "which knowledge graphs are available",
        "list my graph indexes",
        "what graphs exist",
    ],
)
def test_route_turn_suggests_general_for_graph_inventory_queries(query: str) -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )

    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text=query,
        has_attachments=False,
        force_agent=False,
    )

    assert decision.route == "AGENT"
    assert decision.suggested_agent == "general"
    assert choose_agent_name(settings, decision) == "general"
    assert "graph_inventory_intent" in decision.reasons


def test_route_turn_suggests_graph_manager_for_graph_relationship_query() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )

    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text=(
            "Use the knowledge graph defense_rag_v2_graph to find cross-document "
            "relationships between vendors, risks, approvals, dependencies, and program outcomes."
        ),
        has_attachments=False,
        force_agent=False,
    )

    assert decision.route == "AGENT"
    assert decision.suggested_agent == "graph_manager"
    assert choose_agent_name(settings, decision) == "graph_manager"
    assert decision.semantic_contract.requested_scope_kind == "graph_indexes"
    assert decision.semantic_contract.requires_external_evidence is True
    assert "graph_retrieval_intent" in decision.reasons


def test_route_turn_uses_graph_metadata_for_relationship_query() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )

    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="Find the relationships between vendors, risks, approvals, and outcomes.",
        has_attachments=False,
        force_agent=False,
        session_metadata={"active_graph_ids": ["program_graph"]},
    )

    assert decision.route == "AGENT"
    assert decision.suggested_agent == "graph_manager"
    assert choose_agent_name(settings, decision) == "graph_manager"
    assert "graph_metadata_intent" in decision.reasons


@pytest.mark.parametrize(
    ("query", "expected_agent"),
    [
        ("which documents contain onboarding workflows?", "research_coordinator"),
        ("identify all documents that mention onboarding", "research_coordinator"),
        ("what docs are available about onboarding", "rag_worker"),
        ("which documents in the default collection mention onboarding?", "research_coordinator"),
    ],
)
def test_route_turn_keeps_filtered_document_discovery_off_inventory_path(query: str, expected_agent: str) -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )

    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text=query,
        has_attachments=False,
        force_agent=False,
    )

    assert decision.route == "AGENT"
    assert decision.suggested_agent == expected_agent
    assert choose_agent_name(settings, decision) == expected_agent


def test_route_turn_keeps_smalltalk_with_explanation_on_basic_path() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="Hello there. Give me a brief explanation of what this assistant can help with.",
        has_attachments=False,
        force_agent=False,
    )

    assert decision.route == "BASIC"
    assert decision.suggested_agent == ""


def test_route_turn_keeps_plain_smalltalk_on_basic_path() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="Hello there, how are you today?",
        has_attachments=False,
        force_agent=False,
    )

    assert decision.route == "BASIC"
    assert decision.suggested_agent == ""


def test_route_turn_phrase_boundaries_prevent_plan_matching_explanation() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="Please provide an explanation of the assistant capabilities.",
        has_attachments=False,
        force_agent=False,
    )

    assert decision.route == "BASIC"


def test_route_turn_phrase_boundaries_prevent_tab_matching_stable() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="Is the system stable today?",
        has_attachments=False,
        force_agent=False,
    )

    assert decision.route == "BASIC"


def test_route_turn_phrase_boundaries_prevent_source_matching_resource() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="What resources are available for onboarding?",
        has_attachments=False,
        force_agent=False,
    )

    assert decision.route == "BASIC"


def test_route_turn_still_suggests_coordinator_for_explicit_planning_request() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="Plan the work, gather evidence, synthesize the answer, and verify it before responding with citations.",
        has_attachments=False,
        force_agent=False,
    )

    assert decision.route == "AGENT"
    assert decision.suggested_agent == "coordinator"


def test_choose_agent_name_accepts_registry_defined_top_level_specialist(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "general.md").write_text(
        """---
name: general
mode: react
description: default generalist
prompt_file: general_agent.md
skill_scope: general
allowed_tools: ["calculator"]
allowed_worker_agents: []
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 3
max_tool_calls: 3
allow_background_jobs: false
metadata: {"role_kind": "top_level", "entry_path": "default", "expected_output": "user_text"}
---
general
""",
        encoding="utf-8",
    )
    (agents_dir / "policy_specialist.md").write_text(
        """---
name: policy_specialist
mode: react
description: policy-focused top-level specialist
prompt_file: general_agent.md
skill_scope: general
allowed_tools: ["rag_agent_tool"]
allowed_worker_agents: []
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 3
max_tool_calls: 3
allow_background_jobs: false
metadata: {"role_kind": "top_level", "entry_path": "router_or_delegated", "expected_output": "user_text"}
---
policy specialist
""",
        encoding="utf-8",
    )
    registry = AgentRegistry(agents_dir)
    settings = SimpleNamespace(enable_coordinator_mode=False)
    decision = RouterDecision(
        route="AGENT",
        confidence=0.82,
        reasons=["llm_router"],
        suggested_agent="policy_specialist",
        router_method="llm",
    )

    assert choose_agent_name(settings, decision, registry=registry) == "policy_specialist"


def test_route_turn_allows_llm_router_to_suggest_registry_defined_specialist(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "general.md").write_text(
        """---
name: general
mode: react
description: default generalist
prompt_file: general_agent.md
skill_scope: general
allowed_tools: ["calculator"]
allowed_worker_agents: []
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 3
max_tool_calls: 3
allow_background_jobs: false
metadata: {"role_kind": "top_level", "entry_path": "default", "expected_output": "user_text"}
---
general
""",
        encoding="utf-8",
    )
    (agents_dir / "policy_specialist.md").write_text(
        """---
name: policy_specialist
mode: react
description: policy-focused top-level specialist
prompt_file: general_agent.md
skill_scope: general
allowed_tools: ["rag_agent_tool"]
allowed_worker_agents: []
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 3
max_tool_calls: 3
allow_background_jobs: false
metadata: {"role_kind": "top_level", "entry_path": "router_or_delegated", "expected_output": "user_text"}
---
policy specialist
""",
        encoding="utf-8",
    )
    registry = AgentRegistry(agents_dir)

    class FakeJudge:
        def invoke(self, messages):
            del messages
            return '{"route":"AGENT","confidence":0.92,"reasoning":"policy specialist requested","suggested_agent":"policy_specialist"}'

        def with_structured_output(self, schema):
            del schema
            raise NotImplementedError

    settings = SimpleNamespace(
        llm_router_enabled=True,
        llm_router_mode="hybrid",
        llm_router_confidence_threshold=0.95,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=FakeJudge()),
        user_text="Review the policy changes.",
        has_attachments=False,
        force_agent=False,
        registry=registry,
    )

    assert decision.route == "AGENT"
    assert decision.suggested_agent == "policy_specialist"
    assert choose_agent_name(settings, decision, registry=registry) == "policy_specialist"


def test_route_turn_hybrid_defers_structured_research_prompt_to_llm_router() -> None:
    class FakeJudge:
        def with_structured_output(self, schema):
            del schema
            raise NotImplementedError

        def invoke(self, messages):
            del messages
            return '{"route":"AGENT","confidence":0.93,"reasoning":"broad research campaign","suggested_agent":"coordinator"}'

    settings = SimpleNamespace(
        llm_router_enabled=True,
        llm_router_mode="hybrid",
        llm_router_confidence_threshold=0.70,
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=FakeJudge()),
        user_text=(
            "Goal: Investigate the major subsystems in this repo and provide a list of potential documents.\n"
            'Context: Search only the "default" collection in the knowledge base.\n'
            "Deliverable: Return only the document titles."
        ),
        has_attachments=False,
        force_agent=False,
    )

    assert decision.router_method == "llm"
    assert decision.route == "AGENT"
    assert decision.suggested_agent == "research_coordinator"


def test_route_turn_hybrid_skips_judge_for_high_confidence_graph_inventory() -> None:
    class CountingJudge:
        def __init__(self) -> None:
            self.calls = 0

        def with_structured_output(self, schema):
            del schema
            raise NotImplementedError

        def invoke(self, messages):
            del messages
            self.calls += 1
            return '{"route":"BASIC","confidence":0.10,"reasoning":"unused","suggested_agent":""}'

    judge = CountingJudge()
    settings = SimpleNamespace(
        llm_router_enabled=True,
        llm_router_mode="hybrid",
        llm_router_confidence_threshold=0.70,
        enable_coordinator_mode=False,
    )

    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=judge),
        user_text="what knowledge graphs do i have available to me",
        has_attachments=False,
        force_agent=False,
    )

    assert judge.calls == 0
    assert decision.router_method == "deterministic"
    assert decision.route == "AGENT"
    assert decision.suggested_agent == "general"
    assert "graph_inventory_intent" in decision.reasons


def test_route_turn_llm_only_still_short_circuits_graph_inventory() -> None:
    class CountingJudge:
        def __init__(self) -> None:
            self.calls = 0

        def with_structured_output(self, schema):
            del schema
            raise NotImplementedError

        def invoke(self, messages):
            del messages
            self.calls += 1
            return '{"route":"BASIC","confidence":0.10,"reasoning":"unused","suggested_agent":""}'

    judge = CountingJudge()
    settings = SimpleNamespace(
        llm_router_enabled=True,
        llm_router_mode="llm_only",
        llm_router_confidence_threshold=0.10,
        enable_coordinator_mode=False,
    )

    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=judge),
        user_text="list my graph indexes",
        has_attachments=False,
        force_agent=False,
    )

    assert judge.calls == 0
    assert decision.router_method == "deterministic"
    assert decision.route == "AGENT"
    assert decision.suggested_agent == "general"


def test_route_turn_force_agent_preserves_rag_worker_for_grounded_query() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=True,
        llm_router_mode="hybrid",
        enable_coordinator_mode=False,
    )
    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="Summarize the architecture docs and cite your sources.",
        has_attachments=False,
        force_agent=True,
    )
    assert decision.route == "AGENT"
    assert decision.suggested_agent == "rag_worker"


def test_route_turn_matches_accent_insensitive_data_analysis_phrase() -> None:
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )

    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text="Necesito análisis de datos del CSV con totales por región.",
        has_attachments=False,
        force_agent=False,
    )

    assert decision.route == "AGENT"
    assert decision.suggested_agent == "data_analyst"


def test_route_turn_falls_back_to_deterministic_when_llm_router_circuit_is_open() -> None:
    class OpenJudge:
        def with_structured_output(self, schema):
            del schema
            return self

        def invoke(self, messages, config=None):
            del messages, config
            raise CircuitBreakerOpenError(
                key="judge:test:model",
                provider_role="judge",
                provider_name="test",
                model_name="model",
            )

    settings = SimpleNamespace(
        llm_router_enabled=True,
        llm_router_mode="hybrid",
        llm_router_confidence_threshold=0.95,
        enable_coordinator_mode=False,
    )

    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=OpenJudge()),
        user_text="Review the policy changes.",
        has_attachments=False,
        force_agent=False,
    )

    assert decision.router_method == "llm_circuit_fallback"
    assert "llm_router_circuit_open" in decision.reasons
    assert decision.route == "AGENT"
    assert decision.suggested_agent == "rag_worker"


def test_route_turn_llm_only_uses_llm_as_primary_router_for_non_trivial_turns() -> None:
    class CountingJudge:
        def __init__(self) -> None:
            self.calls = 0

        def with_structured_output(self, schema):
            del schema
            raise NotImplementedError

        def invoke(self, messages):
            del messages
            self.calls += 1
            return '{"route":"AGENT","confidence":0.91,"reasoning":"use the analyst","suggested_agent":"data_analyst"}'

    judge = CountingJudge()
    settings = SimpleNamespace(
        llm_router_enabled=True,
        llm_router_mode="llm_only",
        llm_router_confidence_threshold=0.10,
        enable_coordinator_mode=False,
    )

    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=judge),
        user_text="Please explain what this assistant can help with.",
        has_attachments=False,
        force_agent=False,
    )

    assert judge.calls == 1
    assert decision.router_method == "llm"
    assert decision.route == "AGENT"
    assert decision.suggested_agent == "data_analyst"


def test_route_turn_llm_only_keeps_obvious_smalltalk_on_fast_path() -> None:
    class CountingJudge:
        def __init__(self) -> None:
            self.calls = 0

        def with_structured_output(self, schema):
            del schema
            raise NotImplementedError

        def invoke(self, messages):
            del messages
            self.calls += 1
            return '{"route":"AGENT","confidence":0.91,"reasoning":"unused","suggested_agent":"data_analyst"}'

    judge = CountingJudge()
    settings = SimpleNamespace(
        llm_router_enabled=True,
        llm_router_mode="llm_only",
        enable_coordinator_mode=False,
    )

    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=judge),
        user_text="Hello there, how are you today?",
        has_attachments=False,
        force_agent=False,
    )

    assert judge.calls == 0
    assert decision.router_method == "deterministic"
    assert decision.route == "BASIC"
    assert "obvious_small_talk" in decision.reasons


def test_route_turn_llm_only_still_respects_force_agent_fast_path() -> None:
    class CountingJudge:
        def __init__(self) -> None:
            self.calls = 0

        def with_structured_output(self, schema):
            del schema
            raise NotImplementedError

        def invoke(self, messages):
            del messages
            self.calls += 1
            return '{"route":"BASIC","confidence":0.60,"reasoning":"unused","suggested_agent":""}'

    judge = CountingJudge()
    settings = SimpleNamespace(
        llm_router_enabled=True,
        llm_router_mode="llm_only",
        enable_coordinator_mode=False,
    )

    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=judge),
        user_text="Summarize the architecture docs and cite your sources.",
        has_attachments=False,
        force_agent=True,
    )

    assert judge.calls == 0
    assert decision.router_method == "deterministic"
    assert decision.route == "AGENT"
    assert decision.suggested_agent == "rag_worker"


def test_route_turn_llm_only_still_respects_attachment_fast_path() -> None:
    class CountingJudge:
        def __init__(self) -> None:
            self.calls = 0

        def with_structured_output(self, schema):
            del schema
            raise NotImplementedError

        def invoke(self, messages):
            del messages
            self.calls += 1
            return '{"route":"BASIC","confidence":0.60,"reasoning":"unused","suggested_agent":""}'

    judge = CountingJudge()
    settings = SimpleNamespace(
        llm_router_enabled=True,
        llm_router_mode="llm_only",
        enable_coordinator_mode=False,
    )

    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=judge),
        user_text="Analyze this CSV and compute total revenue by region.",
        has_attachments=True,
        force_agent=False,
    )

    assert judge.calls == 0
    assert decision.router_method == "deterministic"
    assert decision.route == "AGENT"
    assert decision.suggested_agent == "data_analyst"


def test_route_turn_llm_only_falls_back_to_deterministic_when_llm_router_fails() -> None:
    class BrokenJudge:
        def with_structured_output(self, schema):
            del schema
            raise NotImplementedError

        def invoke(self, messages):
            del messages
            raise RuntimeError("judge offline")

    settings = SimpleNamespace(
        llm_router_enabled=True,
        llm_router_mode="llm_only",
        enable_coordinator_mode=False,
    )

    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=BrokenJudge()),
        user_text="What are the key implementation details in the architecture docs? Cite your sources.",
        has_attachments=False,
        force_agent=False,
    )

    assert decision.router_method == "llm_fallback"
    assert "llm_router_failed" in decision.reasons
    assert decision.route == "AGENT"
    assert decision.suggested_agent == "rag_worker"


@pytest.mark.parametrize(
    "query",
    [
        'What is the approved current CDR date for Asterion from the "rfp-corpus"',
        "What is the approved current CDR date for Asterion from the rfp-corpus collection",
        "What is the approved current CDR date for Asterion in the rfp-corpus collection",
        "What is the approved current CDR date for Asterion from the “rfp-corpus”",
    ],
)
def test_route_turn_semantically_routes_rfp_corpus_grounded_queries(query: str) -> None:
    class FakeJudge:
        def with_structured_output(self, schema):
            del schema
            return self

        def invoke(self, messages, config=None):
            del messages, config
            return (
                '{"route":"AGENT","confidence":0.94,'
                '"reasoning":"named KB collection requires grounded retrieval",'
                '"suggested_agent":"rag_worker",'
                '"requires_external_evidence":true,'
                '"answer_origin":"retrieval",'
                '"requested_scope_kind":"knowledge_base",'
                '"requested_collection_id":"rfp-corpus"}'
            )

    settings = SimpleNamespace(
        llm_router_enabled=True,
        llm_router_mode="hybrid",
        llm_router_confidence_threshold=0.70,
        enable_coordinator_mode=False,
    )

    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=FakeJudge()),
        user_text=query,
        has_attachments=False,
        force_agent=False,
        session_metadata={
            "available_kb_collection_ids": ["default", "rfp-corpus"],
            "kb_collection_id": "default",
            "kb_collection_confirmed": False,
        },
    )

    assert decision.router_method == "llm"
    assert decision.route == "AGENT"
    assert decision.suggested_agent == "rag_worker"
    assert decision.semantic_contract.requested_collection_id == "rfp-corpus"
    assert decision.semantic_contract.requested_scope_kind == "knowledge_base"
    assert decision.semantic_contract.requires_external_evidence is True


def test_route_turn_promotes_nontrivial_basic_candidate_to_agent_when_llm_router_is_unavailable() -> None:
    class OpenJudge:
        def with_structured_output(self, schema):
            del schema
            return self

        def invoke(self, messages, config=None):
            del messages, config
            raise CircuitBreakerOpenError(
                key="judge:test:model",
                provider_role="judge",
                provider_name="test",
                model_name="model",
            )

    settings = SimpleNamespace(
        llm_router_enabled=True,
        llm_router_mode="hybrid",
        llm_router_confidence_threshold=0.70,
        enable_coordinator_mode=False,
    )

    decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=OpenJudge()),
        user_text='What is the approved current CDR date for Asterion from the "rfp-corpus"?',
        has_attachments=False,
        force_agent=False,
        session_metadata={
            "available_kb_collection_ids": ["default", "rfp-corpus"],
            "kb_collection_id": "default",
            "kb_collection_confirmed": False,
        },
    )

    assert decision.router_method == "llm_circuit_fallback"
    assert "llm_router_circuit_open" in decision.reasons
    assert "llm_router_unavailable_promoted_to_agent" in decision.reasons
    assert decision.route == "AGENT"
    assert decision.suggested_agent == "rag_worker"
    assert decision.semantic_contract.requested_collection_id == "rfp-corpus"


def test_load_router_patterns_rejects_invalid_config(tmp_path: Path) -> None:
    bad_path = tmp_path / "intent_patterns.json"
    bad_path.write_text('{"tool_or_multistep_intent":"not-an-object"}', encoding="utf-8")

    clear_router_patterns_cache()
    with pytest.raises(ValueError) as excinfo:
        load_router_patterns(bad_path)
    assert "intent_patterns.json" in str(excinfo.value)
    assert "- tool_or_multistep_intent:" in str(excinfo.value)
