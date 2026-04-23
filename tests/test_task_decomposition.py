from __future__ import annotations

from types import SimpleNamespace

from agentic_chatbot_next.router.llm_router import route_turn
from agentic_chatbot_next.router.policy import choose_agent_name
from agentic_chatbot_next.runtime.task_decomposition import decide_task_decomposition


def test_task_decomposition_admits_simple_mixed_query_to_general_direct_tools() -> None:
    query = (
        "Calculate 18% of 4.2 million and, separately, search indexed docs for rate limit policy; "
        "return both results together."
    )
    settings = SimpleNamespace(
        llm_router_enabled=False,
        enable_coordinator_mode=False,
    )
    router_decision = route_turn(
        settings,
        providers=SimpleNamespace(judge=None),
        user_text=query,
        has_attachments=False,
        force_agent=False,
    )
    initial_agent = choose_agent_name(settings, router_decision)

    decomposition = decide_task_decomposition(
        query,
        current_agent=initial_agent or "",
        route=router_decision.route,
        suggested_agent=router_decision.suggested_agent,
    )

    assert initial_agent == "rag_worker"
    assert decomposition.is_mixed_intent is True
    assert decomposition.route_kind == "general_direct"
    assert decomposition.selected_agent == "general"
    assert [item.executor for item in decomposition.slices] == ["utility", "rag_worker"]


def test_task_decomposition_preserves_requirements_inventory_fast_path() -> None:
    decomposition = decide_task_decomposition(
        "Extract all shall statements from this uploaded document and export them as a CSV.",
        current_agent="general",
        route="AGENT",
        suggested_agent="general",
    )

    assert decomposition.is_mixed_intent is False
    assert decomposition.selected_agent == "general"


def test_task_decomposition_preserves_openwebui_wrapped_requirements_fast_path() -> None:
    query = """### Task: Respond to the user query using the provided context.

<context>
Example requirement identifier 3.3.2 RC-SOW-33-02 and percentage 20%.
</context>

extract all requirements/ shall statements from the uploaded document
"""
    decomposition = decide_task_decomposition(
        query,
        current_agent="general",
        route="AGENT",
        suggested_agent="general",
    )

    assert decomposition.is_mixed_intent is False
    assert decomposition.selected_agent == "general"
    assert decomposition.slices == ()


def test_task_decomposition_preserves_graph_inventory_fast_path() -> None:
    decomposition = decide_task_decomposition(
        "What graphs do I have access to?",
        current_agent="general",
        route="AGENT",
        suggested_agent="general",
    )

    assert decomposition.is_mixed_intent is False
    assert decomposition.selected_agent == "general"
