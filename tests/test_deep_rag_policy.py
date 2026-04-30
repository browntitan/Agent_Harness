from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_chatbot_next.runtime.deep_rag import (
    decide_deep_rag_policy,
    deep_rag_controller_hints,
    deep_rag_search_mode,
)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        deep_rag_default_mode="auto",
        deep_rag_max_parallel_lanes=3,
        deep_rag_full_read_chunk_threshold=24,
        deep_rag_sync_reflection_rounds=1,
        deep_rag_background_threshold=4,
    )


def test_deep_rag_policy_prefers_coordinator_for_broad_research_prompt() -> None:
    policy = decide_deep_rag_policy(
        _settings(),
        judge_llm=None,
        user_text=(
            "Investigate the full lifecycle of an AGENT request across the default collection "
            "and produce a comprehensive subsystem map."
        ),
        route="AGENT",
        suggested_agent="rag_worker",
        has_attachments=False,
        research_packet="",
        session_metadata={},
        request_metadata={},
    )

    assert policy.search_mode == "deep"
    assert policy.preferred_agent == "research_coordinator"


def test_deep_rag_policy_prefers_full_reads_for_doc_focus_followup() -> None:
    policy = decide_deep_rag_policy(
        _settings(),
        judge_llm=None,
        user_text="Look through the candidate documents you provided and give me a detailed summary.",
        route="AGENT",
        suggested_agent="coordinator",
        has_attachments=False,
        research_packet="",
        session_metadata={"active_doc_focus": {"documents": [{"doc_id": "doc-1", "title": "ARCHITECTURE.md"}]}},
        request_metadata={},
    )

    assert policy.search_mode == "deep"
    assert policy.preferred_agent == "coordinator"
    assert policy.prefer_full_reads is True


def test_deep_rag_controller_hints_round_trip() -> None:
    route_context = {
        "deep_rag": {
            "search_mode": "deep",
            "prefer_full_reads": True,
            "prefer_section_first": True,
            "max_parallel_lanes": 4,
            "max_reflection_rounds": 2,
            "full_read_chunk_threshold": 30,
        }
    }

    hints = deep_rag_controller_hints(route_context)

    assert hints["force_deep_search"] is True
    assert hints["prefer_full_reads"] is True
    assert hints["doc_read_depth"] == "full"
    assert hints["max_parallel_lanes"] == 4
    assert hints["max_reflection_rounds"] == 2
    assert hints["full_read_chunk_threshold"] == 30
    assert deep_rag_search_mode(route_context) == "deep"


def test_deep_rag_policy_keeps_graph_inventory_on_fast_path() -> None:
    policy = decide_deep_rag_policy(
        _settings(),
        judge_llm=None,
        user_text="what graphs do i have access to",
        route="AGENT",
        suggested_agent="general",
        has_attachments=False,
        research_packet="",
        session_metadata={"semantic_routing": {"requested_scope_kind": "graph_indexes"}},
        request_metadata={},
    )

    assert policy.search_mode == "fast"
    assert policy.preferred_agent == ""
    assert policy.background_recommended is False


@pytest.mark.parametrize(
    ("query", "scope_kind"),
    [
        ("what knowledge bases do i have access to", "knowledge_base"),
        ("what documents do we have access to", "session_access"),
    ],
)
def test_deep_rag_policy_keeps_authoritative_inventory_on_fast_path(query: str, scope_kind: str) -> None:
    policy = decide_deep_rag_policy(
        _settings(),
        judge_llm=None,
        user_text=query,
        route="AGENT",
        suggested_agent="general",
        has_attachments=False,
        research_packet="",
        session_metadata={"semantic_routing": {"requested_scope_kind": scope_kind}},
        request_metadata={},
    )

    assert policy.search_mode == "fast"
    assert policy.preferred_agent == ""
    assert policy.background_recommended is False
