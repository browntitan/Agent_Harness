from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_chatbot_next.rag.hints import infer_rag_execution_hints
from agentic_chatbot_next.persistence.postgres.graphs import GraphIndexRecord
from agentic_chatbot_next.persistence.postgres.skills import SkillPackRecord
from agentic_chatbot_next.persistence.postgres.skills import SkillChunkMatch
from agentic_chatbot_next.rag.skill_policy import resolve_rag_execution_hints


class _SkillStore:
    def vector_search(self, query, *, tenant_id, top_k, agent_scope, tool_tags=None, task_tags=None, enabled_only=True, owner_user_id="", graph_ids=None):
        del query, tenant_id, top_k, agent_scope, tool_tags, task_tags, enabled_only, owner_user_id, graph_ids
        return [
            SkillChunkMatch(
                skill_id="rag-corpus-discovery",
                name="Corpus Discovery Campaigns",
                agent_scope="rag",
                content="Prefer a per-document inventory.",
                chunk_index=0,
                score=0.98,
                tool_tags=["search_all_documents"],
                task_tags=["inventory"],
                retrieval_profile="corpus_discovery",
                controller_hints={"prefer_inventory_output": True, "force_deep_search": True},
                coverage_goal="corpus_wide",
                result_mode="inventory",
            )
        ]

    def get_skill_packs_by_ids(self, skill_ids, *, tenant_id, owner_user_id=""):
        del skill_ids, tenant_id, owner_user_id
        return []

    def list_skill_packs(self, *, tenant_id="tenant", agent_scope="", owner_user_id="", graph_id="", **kwargs):
        del tenant_id, agent_scope, owner_user_id, graph_id, kwargs
        return []


class _GraphBoundSkillStore(_SkillStore):
    def list_skill_packs(self, *, tenant_id="tenant", agent_scope="", owner_user_id="", graph_id="", **kwargs):
        del kwargs
        records = [
            SkillPackRecord(
                skill_id="defense-graph-guidance",
                name="Defense Graph Guidance",
                agent_scope="rag",
                checksum="abc123",
                tenant_id=tenant_id,
                graph_id="defense_rag_test_corpus",
                retrieval_profile="targeted",
                controller_hints={"prefer_graph": True, "planned_graph_ids": ["defense_rag_test_corpus"]},
                coverage_goal="targeted",
                result_mode="answer",
                body_markdown="# Defense Graph Guidance\nUse the defense graph first.\n",
                visibility="tenant",
                status="active",
                version_parent="defense-graph-guidance",
            )
        ]
        if graph_id:
            return [record for record in records if record.graph_id == graph_id]
        return [record for record in records if not agent_scope or record.agent_scope == agent_scope]


class _GraphIndexStore:
    def get_index(self, graph_id: str, tenant_id: str, user_id: str = ""):
        del tenant_id, user_id
        if graph_id != "defense_rag_test_corpus":
            return None
        return GraphIndexRecord(
            graph_id="defense_rag_test_corpus",
            tenant_id="tenant",
            display_name="Defense RAG Test Corpus",
            owner_admin_user_id="owner",
            visibility="tenant",
            graph_skill_ids=["defense-graph-guidance"],
        )


class _CaptureOnlySkillStore:
    def __init__(self) -> None:
        self.query = ""

    def vector_search(self, query, *, tenant_id, top_k, agent_scope, tool_tags=None, task_tags=None, enabled_only=True, owner_user_id="", graph_ids=None, accessible_skill_family_ids=None):
        del tenant_id, top_k, agent_scope, tool_tags, task_tags, enabled_only, owner_user_id, graph_ids, accessible_skill_family_ids
        self.query = query
        return []

    def get_skill_packs_by_ids(self, skill_ids, *, tenant_id, owner_user_id=""):
        del skill_ids, tenant_id, owner_user_id
        return []

    def list_skill_packs(self, *, tenant_id="tenant", agent_scope="", owner_user_id="", graph_id="", **kwargs):
        del tenant_id, agent_scope, owner_user_id, graph_id, kwargs
        return []


def test_resolve_rag_execution_hints_uses_skill_metadata():
    hints = resolve_rag_execution_hints(
        SimpleNamespace(default_tenant_id="tenant"),
        SimpleNamespace(skill_store=_SkillStore()),
        session=SimpleNamespace(tenant_id="tenant"),
        query="Which documents contain onboarding workflows?",
        skill_queries=["corpus discovery"],
    )

    assert hints.research_profile == "corpus_discovery"
    assert hints.coverage_goal == "corpus_wide"
    assert hints.result_mode == "inventory"
    assert hints.controller_hints["prefer_inventory_output"] is True
    assert hints.matched_skill_ids == ["rag-corpus-discovery"]


def test_resolve_rag_execution_hints_keeps_explicit_overrides():
    hints = resolve_rag_execution_hints(
        SimpleNamespace(default_tenant_id="tenant"),
        SimpleNamespace(skill_store=_SkillStore()),
        session=SimpleNamespace(tenant_id="tenant"),
        query="Compare all SOPs covering escalation.",
        skill_queries=["comparison campaign"],
        research_profile="comparison_campaign",
        coverage_goal="cross_document",
        result_mode="comparison",
        controller_hints={"compare_across_documents": True},
    )

    assert hints.research_profile == "comparison_campaign"
    assert hints.coverage_goal == "cross_document"
    assert hints.result_mode == "comparison"
    assert hints.controller_hints["compare_across_documents"] is True
    assert hints.controller_hints["prefer_inventory_output"] is True


def test_resolve_rag_execution_hints_prefers_bounded_synthesis_for_architecture_summary():
    hints = resolve_rag_execution_hints(
        SimpleNamespace(default_tenant_id="tenant"),
        SimpleNamespace(skill_store=_SkillStore()),
        session=SimpleNamespace(tenant_id="tenant"),
        query="What are the key implementation details in the architecture docs? Cite your sources.",
        skill_queries=["architecture docs"],
    )

    assert hints.research_profile == ""
    assert hints.coverage_goal == "targeted"
    assert hints.result_mode == "answer"
    assert "prefer_inventory_output" not in hints.controller_hints
    assert "force_deep_search" not in hints.controller_hints


def test_resolve_rag_execution_hints_uses_active_graph_bound_skill_metadata():
    hints = resolve_rag_execution_hints(
        SimpleNamespace(default_tenant_id="tenant", default_user_id="user"),
        SimpleNamespace(skill_store=_GraphBoundSkillStore(), graph_index_store=_GraphIndexStore()),
        session=SimpleNamespace(
            tenant_id="tenant",
            user_id="user",
            metadata={"active_graph_ids": ["defense_rag_test_corpus"]},
        ),
        query="What relationship connects Iron Vale to endurance testing evidence?",
        skill_queries=["graph follow-up"],
    )

    assert "defense-graph-guidance" in hints.matched_skill_ids
    assert hints.controller_hints["prefer_graph"] is True


def test_resolve_rag_execution_hints_caps_resolver_query_but_keeps_inventory_mode() -> None:
    skill_store = _CaptureOnlySkillStore()

    hints = resolve_rag_execution_hints(
        SimpleNamespace(default_tenant_id="tenant", skill_context_max_chars=80),
        SimpleNamespace(skill_store=skill_store),
        session=SimpleNamespace(tenant_id="tenant"),
        query="what knowledge bases do i have access to\n" + ("coordinator handoff context\n" * 20),
        skill_queries=["inventory lookup", "available collections"],
    )

    assert hints.result_mode == "inventory"
    assert hints.controller_hints["prefer_inventory_output"] is True
    assert skill_store.query.startswith("what knowledge bases do i have access to")
    assert len(skill_store.query) <= 80


@pytest.mark.parametrize(
    "query",
    [
        "what documents do we have access to",
        "what files do we have access to",
        "what docs do you have",
        "what knowledge bases do you have access to",
        "list out the kbs i have access to",
        "can you list out what documents we have access to in the knowledge base",
        "what docs are in the KB",
        "what's indexed",
        "show me the knowledge base inventory",
    ],
)
def test_infer_rag_execution_hints_uses_inventory_mode_for_kb_inventory_queries(query: str) -> None:
    hints = infer_rag_execution_hints(query)

    assert hints.research_profile == "corpus_discovery"
    assert hints.coverage_goal == "corpus_wide"
    assert hints.result_mode == "inventory"
    assert hints.controller_hints["prefer_inventory_output"] is True


def test_infer_rag_execution_hints_keeps_filtered_access_query_on_search_path() -> None:
    hints = infer_rag_execution_hints("what docs are available about onboarding")

    assert hints.research_profile == ""
    assert hints.coverage_goal == "targeted"
    assert hints.result_mode == "answer"
    assert "prefer_inventory_output" not in hints.controller_hints


def test_infer_rag_execution_hints_recognizes_structured_document_discovery_prompt() -> None:
    hints = infer_rag_execution_hints(
        """
        Goal: Investigate the major subsystems in this repo and provide me a list of documents that have information about the major sub systems
        Context: The documents i want you to pull this information from is provided to you internally in the knowledge base we have access to in the "default" collection, conduct your search accross the documents that are in there
        Deliverable: you will provide me only a list of potential documents that have information about major subsystems
        """
    )

    assert hints.research_profile == "corpus_discovery"
    assert hints.coverage_goal == "corpus_wide"
    assert hints.result_mode == "inventory"
    assert hints.controller_hints["prefer_inventory_output"] is True
