from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from agentic_chatbot_next.persistence.postgres.graphs import GraphIndexRecord, GraphIndexSourceRecord
from agentic_chatbot_next.rag.doc_targets import (
    AmbiguousIndexedDocMatch,
    IndexedDocResolution,
    MissingIndexedDocMatch,
    ResolvedIndexedDoc,
)
from agentic_chatbot_next.rag.engine import _answer_context, _requested_doc_resolution_answer, run_rag_contract
from agentic_chatbot_next.rag.retrieval import GradedChunk


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        default_collection_id="default",
        default_tenant_id="tenant",
        rag_min_evidence_chunks=1,
        prompts_backend="local",
        judge_grading_prompt_path=Path("missing"),
        grounded_answer_prompt_path=Path("missing"),
    )


class _GraphIndexStore:
    def __init__(self, records):
        self._records = list(records)

    def list_indexes(self, *, tenant_id="tenant", user_id="", collection_id="", status="", backend="", limit=100):
        del user_id, status, backend
        return [
            record
            for record in self._records
            if str(getattr(record, "tenant_id", "tenant") or "tenant") == tenant_id
            and (not collection_id or str(getattr(record, "collection_id", "") or "") == collection_id)
        ][:limit]

    def get_index(self, graph_id, tenant_id="tenant", user_id=""):
        del user_id
        for record in self._records:
            if str(getattr(record, "tenant_id", "tenant") or "tenant") != tenant_id:
                continue
            if str(getattr(record, "graph_id", "") or "") == str(graph_id or ""):
                return record
        return None


class _GraphSourceStore:
    def __init__(self, records):
        self._records = list(records)

    def list_sources(self, graph_id, *, tenant_id="tenant"):
        return [
            record
            for record in self._records
            if str(getattr(record, "tenant_id", "tenant") or "tenant") == tenant_id
            and str(getattr(record, "graph_id", "") or "") == str(graph_id or "")
        ]


def test_run_rag_contract_uses_inventory_mode_answer(monkeypatch):
    selected_doc = Document(
        page_content="Workflow with approval handoff for onboarding.",
        metadata={
            "doc_id": "doc-1",
            "chunk_id": "doc-1#chunk0001",
            "title": "onboarding_workflow.md",
            "source_type": "kb",
            "chunk_index": 1,
        },
    )

    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.run_retrieval_controller",
        lambda *args, **kwargs: SimpleNamespace(
            selected_docs=[selected_doc],
            candidate_docs=[selected_doc],
            graded=[],
            query_used="Which documents contain onboarding workflows?",
            search_mode="deep",
            rounds=2,
            tool_calls_used=4,
            tool_call_log=["round1:search_corpus[hybrid]:workflow"],
            strategies_used=["hybrid", "keyword"],
            candidate_counts={"unique_docs": 1, "selected_docs": 1},
            evidence_ledger={"round_summaries": [], "entries": []},
            parallel_workers_used=False,
            to_summary=lambda citations_found: SimpleNamespace(
                to_dict=lambda: {},
                query_used="Which documents contain onboarding workflows?",
                steps=4,
                tool_calls_used=4,
                tool_call_log=["round1:search_corpus[hybrid]:workflow"],
                citations_found=citations_found,
                search_mode="deep",
                rounds=2,
                strategies_used=["hybrid", "keyword"],
                candidate_counts={"unique_docs": 1, "selected_docs": 1},
                parallel_workers_used=False,
            ),
        ),
    )
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.get_kb_coverage_status",
        lambda *args, **kwargs: SimpleNamespace(ready=True),
    )

    contract = run_rag_contract(
        _settings(),
        SimpleNamespace(),
        providers=SimpleNamespace(chat=SimpleNamespace(invoke=lambda *args, **kwargs: None), judge=SimpleNamespace()),
        session=SimpleNamespace(tenant_id="tenant", scratchpad={}),
        query="Which documents contain onboarding workflows?",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
        result_mode="inventory",
        controller_hints={"prefer_inventory_output": True},
    )

    assert "Documents with grounded evidence relevant to the request" in contract.answer
    assert "onboarding_workflow.md" in contract.answer
    assert contract.used_citation_ids == ["doc-1#chunk0001"]


def test_answer_contract_prevents_inventory_output_shape(monkeypatch):
    captured: dict[str, object] = {}
    selected_doc = Document(
        page_content="Alpha, beta, and gamma are covered by this source.",
        metadata={
            "doc_id": "doc-1",
            "chunk_id": "doc-1#chunk0001",
            "title": "source.md",
            "source_type": "kb",
            "chunk_index": 1,
        },
    )

    def fake_run_retrieval_controller(*args, **kwargs):
        del args
        captured["result_mode"] = kwargs.get("result_mode")
        captured["controller_hints"] = dict(kwargs.get("controller_hints") or {})
        return SimpleNamespace(
            selected_docs=[selected_doc],
            candidate_docs=[selected_doc],
            graded=[GradedChunk(doc=selected_doc, relevance=3, reason="test")],
            query_used="Draft an answer about alpha, beta, and gamma",
            search_mode="deep",
            rounds=1,
            tool_calls_used=2,
            tool_call_log=[],
            strategies_used=["hybrid"],
            candidate_counts={"unique_docs": 1, "selected_docs": 1},
            evidence_ledger={"round_summaries": [], "entries": []},
            parallel_workers_used=False,
            retrieval_verification={},
            to_summary=lambda citations_found: SimpleNamespace(
                to_dict=lambda: {},
                query_used="Draft an answer about alpha, beta, and gamma",
                steps=2,
                tool_calls_used=2,
                tool_call_log=[],
                citations_found=citations_found,
                search_mode="deep",
                rounds=1,
                strategies_used=["hybrid"],
                candidate_counts={"unique_docs": 1, "selected_docs": 1},
                parallel_workers_used=False,
                retrieval_verification={},
            ),
        )

    monkeypatch.setattr("agentic_chatbot_next.rag.engine.run_retrieval_controller", fake_run_retrieval_controller)
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.get_collection_readiness_status",
        lambda *args, **kwargs: SimpleNamespace(ready=True),
    )
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.generate_grounded_answer",
        lambda *args, **kwargs: {
            "answer": "Grounded answer. (doc-1#chunk0001)",
            "used_citation_ids": ["doc-1#chunk0001"],
            "followups": [],
            "warnings": [],
            "confidence_hint": 0.8,
        },
    )

    contract = run_rag_contract(
        _settings(),
        SimpleNamespace(),
        providers=SimpleNamespace(chat=SimpleNamespace(invoke=lambda *args, **kwargs: None), judge=SimpleNamespace()),
        session=SimpleNamespace(tenant_id="tenant", scratchpad={}, metadata={"kb_collection_id": "default"}),
        query="Draft a concise grounded answer about alpha, beta, and gamma. Cite sources.",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
        result_mode="inventory",
        controller_hints={"prefer_inventory_output": True},
        answer_contract=SimpleNamespace(kind="grounded_synthesis", broad_coverage=False),
    )

    assert "Documents with grounded evidence relevant to the request" not in contract.answer
    assert contract.answer.startswith("Grounded answer.")
    assert captured["result_mode"] == "answer"
    assert "prefer_inventory_output" not in captured["controller_hints"]


def test_answer_mode_does_not_attach_uncited_synthesis_to_first_citation(monkeypatch):
    selected_doc = Document(
        page_content="Alpha policy evidence is present in this retrieved chunk.",
        metadata={
            "doc_id": "doc-1",
            "chunk_id": "doc-1#chunk0001",
            "title": "alpha.md",
            "source_type": "kb",
            "chunk_index": 1,
        },
    )

    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.run_retrieval_controller",
        lambda *args, **kwargs: SimpleNamespace(
            selected_docs=[selected_doc],
            candidate_docs=[selected_doc],
            graded=[GradedChunk(doc=selected_doc, relevance=3, reason="test")],
            query_used="Explain alpha",
            search_mode="deep",
            rounds=1,
            tool_calls_used=2,
            tool_call_log=[],
            strategies_used=["hybrid"],
            candidate_counts={"unique_docs": 1, "selected_docs": 1},
            evidence_ledger={"round_summaries": [], "entries": []},
            parallel_workers_used=False,
            retrieval_verification={},
            to_summary=lambda citations_found: SimpleNamespace(
                to_dict=lambda: {},
                query_used="Explain alpha",
                steps=2,
                tool_calls_used=2,
                tool_call_log=[],
                citations_found=citations_found,
                search_mode="deep",
                rounds=1,
                strategies_used=["hybrid"],
                candidate_counts={"unique_docs": 1, "selected_docs": 1},
                parallel_workers_used=False,
                retrieval_verification={},
            ),
        ),
    )
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.get_collection_readiness_status",
        lambda *args, **kwargs: SimpleNamespace(ready=True),
    )
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.generate_grounded_answer",
        lambda *args, **kwargs: {
            "answer": "This uncited synthesis should not be shipped as-is.",
            "used_citation_ids": [],
            "followups": [],
            "warnings": [],
            "confidence_hint": 0.7,
        },
    )

    contract = run_rag_contract(
        _settings(),
        SimpleNamespace(),
        providers=SimpleNamespace(chat=SimpleNamespace(invoke=lambda *args, **kwargs: None), judge=SimpleNamespace()),
        session=SimpleNamespace(tenant_id="tenant", scratchpad={}, metadata={"kb_collection_id": "default"}),
        query="Explain alpha. Cite sources.",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
        answer_contract=SimpleNamespace(kind="grounded_synthesis", broad_coverage=False),
    )

    assert "uncited synthesis" not in contract.answer
    assert "could not produce a fully cited synthesis" in contract.answer
    assert contract.used_citation_ids == ["doc-1#chunk0001"]
    assert "MISSING_VALID_CITATIONS_FALLBACK" in contract.warnings


def test_answer_mode_uses_extractive_fallback_when_budget_exhausted(monkeypatch):
    selected_doc = Document(
        page_content="The alpha retention setting is 30 days for standard plans.",
        metadata={
            "doc_id": "doc-1",
            "chunk_id": "doc-1#chunk0001",
            "title": "alpha.md",
            "source_type": "kb",
            "chunk_index": 1,
        },
    )

    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.run_retrieval_controller",
        lambda *args, **kwargs: SimpleNamespace(
            selected_docs=[selected_doc],
            candidate_docs=[selected_doc],
            graded=[GradedChunk(doc=selected_doc, relevance=3, reason="test")],
            query_used="What is the alpha retention setting?",
            search_mode="fast",
            rounds=1,
            tool_calls_used=2,
            tool_call_log=[],
            strategies_used=["hybrid"],
            candidate_counts={"unique_docs": 1, "selected_docs": 1},
            evidence_ledger={"round_summaries": [], "entries": []},
            parallel_workers_used=False,
            retrieval_verification={},
            stage_timings_ms={"hybrid_retrieval": 205000.0},
            budget_ms=210000,
            budget_exhausted=True,
            slow_stages=["hybrid_retrieval"],
            to_summary=lambda citations_found: SimpleNamespace(
                to_dict=lambda: {},
                query_used="What is the alpha retention setting?",
                steps=2,
                tool_calls_used=2,
                tool_call_log=[],
                citations_found=citations_found,
                search_mode="fast",
                rounds=1,
                strategies_used=["hybrid"],
                candidate_counts={"unique_docs": 1, "selected_docs": 1},
                parallel_workers_used=False,
                retrieval_verification={},
                stage_timings_ms={"hybrid_retrieval": 205000.0},
                budget_ms=210000,
                budget_exhausted=True,
                slow_stages=["hybrid_retrieval"],
            ),
        ),
    )
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.get_collection_readiness_status",
        lambda *args, **kwargs: SimpleNamespace(ready=True),
    )

    def fail_synthesis(*args, **kwargs):
        raise AssertionError("synthesis should be skipped after budget exhaustion")

    monkeypatch.setattr("agentic_chatbot_next.rag.engine.generate_grounded_answer", fail_synthesis)

    contract = run_rag_contract(
        _settings(),
        SimpleNamespace(),
        providers=SimpleNamespace(chat=SimpleNamespace(invoke=lambda *args, **kwargs: None), judge=SimpleNamespace()),
        session=SimpleNamespace(tenant_id="tenant", scratchpad={}, metadata={"kb_collection_id": "default"}),
        query="What is the alpha retention setting? Cite sources.",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
        answer_contract=SimpleNamespace(kind="grounded_synthesis", broad_coverage=False),
    )

    assert "30 days" in contract.answer
    assert contract.used_citation_ids == ["doc-1#chunk0001"]
    assert "BUDGET_EXTRACTIVE_FALLBACK" in contract.warnings


def test_run_rag_contract_applies_deep_rag_route_context(monkeypatch):
    captured: dict[str, object] = {}
    selected_doc = Document(
        page_content="Architecture summary.",
        metadata={
            "doc_id": "doc-1",
            "chunk_id": "doc-1#chunk0001",
            "title": "ARCHITECTURE.md",
            "source_type": "kb",
            "chunk_index": 1,
        },
    )

    def fake_run_retrieval_controller(*args, **kwargs):
        del args
        captured["search_mode"] = kwargs.get("search_mode")
        captured["controller_hints"] = dict(kwargs.get("controller_hints") or {})
        return SimpleNamespace(
            selected_docs=[selected_doc],
            candidate_docs=[selected_doc],
            graded=[GradedChunk(doc=selected_doc, relevance=3, reason="test")],
            query_used="Summarize the architecture docs",
            search_mode="deep",
            rounds=2,
            tool_calls_used=4,
            tool_call_log=[],
            strategies_used=["hybrid", "document_read"],
            candidate_counts={"unique_docs": 1, "selected_docs": 1},
            evidence_ledger={"round_summaries": [], "entries": []},
            parallel_workers_used=False,
            retrieval_verification={},
            to_summary=lambda citations_found: SimpleNamespace(
                to_dict=lambda: {},
                query_used="Summarize the architecture docs",
                steps=4,
                tool_calls_used=4,
                tool_call_log=[],
                citations_found=citations_found,
                search_mode="deep",
                rounds=2,
                strategies_used=["hybrid", "document_read"],
                candidate_counts={"unique_docs": 1, "selected_docs": 1},
                parallel_workers_used=False,
                retrieval_verification={},
            ),
        )

    monkeypatch.setattr("agentic_chatbot_next.rag.engine.run_retrieval_controller", fake_run_retrieval_controller)
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.get_kb_coverage_status",
        lambda *args, **kwargs: SimpleNamespace(ready=True),
    )
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.generate_grounded_answer",
        lambda *args, **kwargs: {
            "answer": "Architecture summary.",
            "used_citation_ids": ["doc-1#chunk0001"],
            "followups": [],
            "warnings": [],
            "confidence_hint": 0.8,
        },
    )

    contract = run_rag_contract(
        _settings(),
        SimpleNamespace(),
        providers=SimpleNamespace(chat=SimpleNamespace(invoke=lambda *args, **kwargs: None), judge=SimpleNamespace()),
        session=SimpleNamespace(
            tenant_id="tenant",
            scratchpad={},
            metadata={
                "route_context": {
                    "deep_rag": {
                        "search_mode": "deep",
                        "prefer_full_reads": True,
                        "max_parallel_lanes": 4,
                        "max_reflection_rounds": 2,
                    }
                }
            },
        ),
        query="Summarize the architecture docs.",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
    )

    assert contract.answer == "Architecture summary."
    assert captured["search_mode"] == "deep"
    controller_hints = dict(captured["controller_hints"] or {})
    assert controller_hints["force_deep_search"] is True
    assert controller_hints["prefer_full_reads"] is True
    assert controller_hints["doc_read_depth"] == "full"
    assert controller_hints["max_parallel_lanes"] == 4


def test_run_rag_contract_bridges_active_graph_ids_into_controller_hints(monkeypatch):
    captured: dict[str, object] = {}
    selected_doc = Document(
        page_content="Graph-backed architecture summary.",
        metadata={
            "doc_id": "doc-1",
            "chunk_id": "doc-1#chunk0001",
            "title": "ARCHITECTURE.md",
            "source_type": "kb",
            "chunk_index": 1,
        },
    )

    def fake_run_retrieval_controller(*args, **kwargs):
        del args
        captured["controller_hints"] = dict(kwargs.get("controller_hints") or {})
        return SimpleNamespace(
            selected_docs=[selected_doc],
            candidate_docs=[selected_doc],
            graded=[GradedChunk(doc=selected_doc, relevance=3, reason="test")],
            query_used="Summarize the architecture docs",
            search_mode="deep",
            rounds=1,
            tool_calls_used=2,
            tool_call_log=[],
            strategies_used=["hybrid", "graph"],
            candidate_counts={"unique_docs": 1, "selected_docs": 1},
            evidence_ledger={"round_summaries": [], "entries": []},
            parallel_workers_used=False,
            retrieval_verification={},
            to_summary=lambda citations_found: SimpleNamespace(
                to_dict=lambda: {},
                query_used="Summarize the architecture docs",
                steps=2,
                tool_calls_used=2,
                tool_call_log=[],
                citations_found=citations_found,
                search_mode="deep",
                rounds=1,
                strategies_used=["hybrid", "graph"],
                candidate_counts={"unique_docs": 1, "selected_docs": 1},
                parallel_workers_used=False,
                retrieval_verification={},
            ),
        )

    monkeypatch.setattr("agentic_chatbot_next.rag.engine.run_retrieval_controller", fake_run_retrieval_controller)
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.get_collection_readiness_status",
        lambda *args, **kwargs: SimpleNamespace(ready=True),
    )
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.generate_grounded_answer",
        lambda *args, **kwargs: {
            "answer": "Architecture summary.",
            "used_citation_ids": ["doc-1#chunk0001"],
            "followups": [],
            "warnings": [],
            "confidence_hint": 0.8,
        },
    )

    contract = run_rag_contract(
        _settings(),
        SimpleNamespace(),
        providers=SimpleNamespace(chat=SimpleNamespace(invoke=lambda *args, **kwargs: None), judge=SimpleNamespace()),
        session=SimpleNamespace(
            tenant_id="tenant",
            scratchpad={},
            metadata={"kb_collection_id": "default", "active_graph_ids": ["rfp_corpus", "vendor_risk"]},
        ),
        query="Summarize the architecture docs.",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
    )

    assert contract.answer == "Architecture summary."
    controller_hints = dict(captured["controller_hints"] or {})
    assert controller_hints["graph_ids"] == ["rfp_corpus", "vendor_risk"]
    assert controller_hints["planned_graph_ids"] == ["rfp_corpus", "vendor_risk"]


def test_run_rag_contract_downgrades_generic_workflow_candidates_to_no_confirmed_matches(monkeypatch):
    weak_doc = Document(
        page_content="Workflow description for routing approval handoffs between teams.",
        metadata={
            "doc_id": "doc-weak",
            "chunk_id": "doc-weak#chunk0001",
            "title": "TOOLS_AND_TOOL_CALLING.md",
            "source_type": "kb",
            "chunk_index": 1,
        },
    )

    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.run_retrieval_controller",
        lambda *args, **kwargs: SimpleNamespace(
            selected_docs=[weak_doc],
            candidate_docs=[weak_doc],
            graded=[GradedChunk(doc=weak_doc, relevance=2, reason="test")],
            query_used="Which documents contain onboarding workflows?",
            search_mode="deep",
            rounds=2,
            tool_calls_used=4,
            tool_call_log=["round1:search_corpus[hybrid]:workflow"],
            strategies_used=["hybrid", "keyword"],
            candidate_counts={"unique_docs": 1, "selected_docs": 1},
            evidence_ledger={"round_summaries": [{"queries": ["keyword:workflow approval handoff"]}], "entries": []},
            parallel_workers_used=False,
            retrieval_verification={},
            to_summary=lambda citations_found: SimpleNamespace(
                to_dict=lambda: {},
                query_used="Which documents contain onboarding workflows?",
                steps=4,
                tool_calls_used=4,
                tool_call_log=["round1:search_corpus[hybrid]:workflow"],
                citations_found=citations_found,
                search_mode="deep",
                rounds=2,
                strategies_used=["hybrid", "keyword"],
                candidate_counts={"unique_docs": 1, "selected_docs": 1, "confirmed_match_count": 0},
                parallel_workers_used=False,
                retrieval_verification={
                    "downgraded_to_negative_evidence": True,
                    "downgrade_reason": "no_confirmed_topic_matches",
                },
            ),
        ),
    )
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.get_kb_coverage_status",
        lambda *args, **kwargs: SimpleNamespace(ready=True),
    )

    contract = run_rag_contract(
        _settings(),
        SimpleNamespace(),
        providers=SimpleNamespace(chat=SimpleNamespace(invoke=lambda *args, **kwargs: None), judge=SimpleNamespace()),
        session=SimpleNamespace(tenant_id="tenant", scratchpad={}),
        query="Which documents in the knowledge base contain onboarding workflows?",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
        research_profile="corpus_discovery",
        coverage_goal="corpus_wide",
        result_mode="inventory",
        controller_hints={"prefer_inventory_output": True, "prefer_process_flow_docs": True},
    )

    assert "No confirmed knowledge-base documents were found that explicitly mention onboarding workflows" in contract.answer
    assert "TOOLS_AND_TOOL_CALLING.md" not in contract.answer
    assert contract.citations == []
    assert contract.used_citation_ids == []
    assert "INSUFFICIENT_CORPUS_EVIDENCE" in contract.warnings
    assert "RETRIEVAL_VERIFICATION_ISSUES" not in contract.warnings
    assert contract.retrieval_summary.retrieval_verification["downgraded_to_negative_evidence"] is True
    assert contract.retrieval_summary.retrieval_verification["downgrade_reason"] == "no_confirmed_topic_matches"


def test_run_rag_contract_suppresses_inventory_list_when_verification_fails(monkeypatch):
    supported_doc = Document(
        page_content="The onboarding workflow routes approvals to HR and IT before day one.",
        metadata={
            "doc_id": "doc-1",
            "chunk_id": "doc-1#chunk0001",
            "title": "onboarding_workflow.md",
            "source_type": "kb",
            "chunk_index": 1,
        },
    )

    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.run_retrieval_controller",
        lambda *args, **kwargs: SimpleNamespace(
            selected_docs=[supported_doc],
            candidate_docs=[supported_doc],
            graded=[GradedChunk(doc=supported_doc, relevance=3, reason="test")],
            query_used="Which documents contain onboarding workflows?",
            search_mode="deep",
            rounds=2,
            tool_calls_used=4,
            tool_call_log=["round1:search_corpus[hybrid]:onboarding workflow"],
            strategies_used=["hybrid", "keyword"],
            candidate_counts={"unique_docs": 1, "selected_docs": 1},
            evidence_ledger={"round_summaries": [{"queries": ["keyword:onboarding workflow"]}], "entries": []},
            parallel_workers_used=False,
            retrieval_verification={"status": "revise", "issues": [{"check": "citation_topic_mismatch"}]},
            to_summary=lambda citations_found: SimpleNamespace(
                to_dict=lambda: {},
                query_used="Which documents contain onboarding workflows?",
                steps=4,
                tool_calls_used=4,
                tool_call_log=["round1:search_corpus[hybrid]:onboarding workflow"],
                citations_found=citations_found,
                search_mode="deep",
                rounds=2,
                strategies_used=["hybrid", "keyword"],
                candidate_counts={"unique_docs": 1, "selected_docs": 1, "confirmed_match_count": 1},
                parallel_workers_used=False,
                retrieval_verification={
                    "downgraded_to_negative_evidence": True,
                    "downgrade_reason": "retrieval_verification_failed",
                },
            ),
        ),
    )
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.get_kb_coverage_status",
        lambda *args, **kwargs: SimpleNamespace(ready=True),
    )

    contract = run_rag_contract(
        _settings(),
        SimpleNamespace(),
        providers=SimpleNamespace(chat=SimpleNamespace(invoke=lambda *args, **kwargs: None), judge=SimpleNamespace()),
        session=SimpleNamespace(tenant_id="tenant", scratchpad={}),
        query="Which documents in the knowledge base contain onboarding workflows?",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
        research_profile="corpus_discovery",
        coverage_goal="corpus_wide",
        result_mode="inventory",
        controller_hints={"prefer_inventory_output": True, "prefer_process_flow_docs": True},
    )

    assert "No confirmed knowledge-base documents were found that explicitly mention onboarding workflows" in contract.answer
    assert "onboarding_workflow.md" not in contract.answer
    assert contract.citations == []
    assert "RETRIEVAL_VERIFICATION_ISSUES" not in contract.warnings
    assert contract.retrieval_summary.retrieval_verification["downgrade_reason"] == "retrieval_verification_failed"


def test_answer_context_uses_detailed_directive_for_active_doc_focus_summary():
    context = _answer_context(
        "Can you look through the candidate documents you provided and give me a detailed summary of the major subsystems involved?",
        "assistant: Candidate docs ready.",
        {},
        coverage_goal="targeted",
        result_mode="answer",
        controller_hints={
            "summary_scope": "active_doc_focus",
            "prefer_detailed_synthesis": True,
        },
    )

    assert "detailed subsystem-organized synthesis" in context
    assert "one short synthesis paragraph" not in context


def test_run_rag_contract_uses_metadata_inventory_answer_for_kb_listing_queries(monkeypatch):
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.run_retrieval_controller",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieval should not run for metadata inventory")),
    )
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.get_kb_coverage_status",
        lambda *args, **kwargs: SimpleNamespace(ready=True),
    )

    stores = SimpleNamespace(
        doc_store=SimpleNamespace(
            list_documents=lambda source_type="", tenant_id="tenant", collection_id="default": [
                SimpleNamespace(
                    doc_id="doc-arch",
                    title="ARCHITECTURE.md",
                    source_type=source_type or "kb",
                    source_path="/repo/docs/ARCHITECTURE.md",
                    collection_id=collection_id,
                    file_type="md",
                    doc_structure_type="general",
                    num_chunks=12,
                ),
                SimpleNamespace(
                    doc_id="doc-c4",
                    title="C4_ARCHITECTURE.md",
                    source_type=source_type or "kb",
                    source_path="/repo/docs/C4_ARCHITECTURE.md",
                    collection_id=collection_id,
                    file_type="md",
                    doc_structure_type="general",
                    num_chunks=8,
                ),
            ]
        )
    )

    contract = run_rag_contract(
        _settings(),
        stores,
        providers=SimpleNamespace(chat=SimpleNamespace(invoke=lambda *args, **kwargs: None), judge=SimpleNamespace()),
        session=SimpleNamespace(tenant_id="tenant", scratchpad={}, metadata={}),
        query="can you list out what documents we have access to in the knowledge base",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
    )

    assert "Knowledge base documents currently indexed in collection `default` (2 total):" in contract.answer
    assert "ARCHITECTURE.md (doc_id=doc-arch; file_type=md; chunks=12; path=/repo/docs/ARCHITECTURE.md)" in contract.answer
    assert "C4_ARCHITECTURE.md (doc_id=doc-c4; file_type=md; chunks=8; path=/repo/docs/C4_ARCHITECTURE.md)" in contract.answer
    assert contract.used_citation_ids == []
    assert contract.citations == []
    assert contract.warnings == []
    assert contract.retrieval_summary.search_mode == "metadata_inventory"


def test_run_rag_contract_uses_metadata_inventory_answer_for_named_collection_queries(monkeypatch):
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.run_retrieval_controller",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieval should not run for named collection inventory")),
    )
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.get_kb_coverage_status",
        lambda *args, **kwargs: SimpleNamespace(ready=True),
    )

    stores = SimpleNamespace(
        doc_store=SimpleNamespace(
            list_documents=lambda source_type="", tenant_id="tenant", collection_id="default": [
                record
                for record in [
                    SimpleNamespace(
                        doc_id="doc-arch",
                        title="ARCHITECTURE.md",
                        source_type="kb",
                        source_path="/repo/docs/ARCHITECTURE.md",
                        collection_id="default",
                        file_type="md",
                        doc_structure_type="general",
                        num_chunks=12,
                    ),
                    SimpleNamespace(
                        doc_id="doc-c4",
                        title="C4_ARCHITECTURE.md",
                        source_type="kb",
                        source_path="/repo/docs/C4_ARCHITECTURE.md",
                        collection_id="default",
                        file_type="md",
                        doc_structure_type="general",
                        num_chunks=8,
                    ),
                    SimpleNamespace(
                        doc_id="doc-other",
                        title="Other.md",
                        source_type="kb",
                        source_path="/repo/docs/Other.md",
                        collection_id="other",
                        file_type="md",
                        doc_structure_type="general",
                        num_chunks=3,
                    ),
                ]
                if (not source_type or record.source_type == source_type)
                and (not collection_id or record.collection_id == collection_id)
            ]
        )
    )

    contract = run_rag_contract(
        _settings(),
        stores,
        providers=SimpleNamespace(chat=SimpleNamespace(invoke=lambda *args, **kwargs: None), judge=SimpleNamespace()),
        session=SimpleNamespace(tenant_id="tenant", scratchpad={}, metadata={"kb_collection_id": "default"}),
        query="can you list out all of the documents in the default collection",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
    )

    assert "Knowledge base documents currently indexed in collection `default` (2 total):" in contract.answer
    assert "ARCHITECTURE.md (doc_id=doc-arch; file_type=md; chunks=12; path=/repo/docs/ARCHITECTURE.md)" in contract.answer
    assert "C4_ARCHITECTURE.md (doc_id=doc-c4; file_type=md; chunks=8; path=/repo/docs/C4_ARCHITECTURE.md)" in contract.answer
    assert "Other.md" not in contract.answer
    assert contract.warnings == []
    assert contract.retrieval_summary.search_mode == "metadata_inventory"


def test_run_rag_contract_reports_unavailable_named_collection_as_metadata_inventory(monkeypatch):
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.run_retrieval_controller",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieval should not run for unavailable named collection inventory")),
    )

    stores = SimpleNamespace(doc_store=SimpleNamespace(list_documents=lambda **kwargs: []))

    contract = run_rag_contract(
        _settings(),
        stores,
        providers=SimpleNamespace(chat=SimpleNamespace(invoke=lambda *args, **kwargs: None), judge=SimpleNamespace()),
        session=SimpleNamespace(tenant_id="tenant", scratchpad={}, metadata={"kb_collection_id": "default"}),
        query="what documents are in the other collection",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
    )

    assert "Requested KB collection `other` is not available to this chat." in contract.answer
    assert "Available KB collections: `default`" in contract.answer
    assert contract.warnings == ["KB_COLLECTION_NOT_AVAILABLE"]
    assert contract.retrieval_summary.search_mode == "metadata_inventory"


def test_run_rag_contract_uses_kb_collection_access_answer_for_collection_queries(monkeypatch):
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.run_retrieval_controller",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieval should not run for KB collection inventory")),
    )
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.inventory.get_kb_coverage_status",
        lambda *args, **kwargs: SimpleNamespace(ready=True, configured_source_paths=("/repo/docs/ARCHITECTURE.md",)),
    )

    def list_documents(source_type="", tenant_id="tenant", collection_id="default"):
        del tenant_id
        records = [
            SimpleNamespace(
                doc_id="doc-arch",
                title="Architecture Overview.md",
                source_type="kb",
                source_path="/repo/docs/ARCHITECTURE.md",
                collection_id="default",
                file_type="md",
                doc_structure_type="general",
                num_chunks=12,
            ),
            SimpleNamespace(
                doc_id="doc-other",
                title="RFP Overview.docx",
                source_type="host_path",
                source_path="/repo/docs/RFP Overview.docx",
                collection_id="rfp-corpus",
                file_type="docx",
                doc_structure_type="general",
                num_chunks=3,
            ),
        ]
        return [
            record
            for record in records
            if (not source_type or record.source_type == source_type)
            and (not collection_id or record.collection_id == collection_id)
        ]

    stores = SimpleNamespace(
        doc_store=SimpleNamespace(list_documents=list_documents),
        graph_index_store=SimpleNamespace(
            list_indexes=lambda **kwargs: [
                GraphIndexRecord(
                    graph_id="default_ops_graph",
                    tenant_id="tenant",
                    collection_id="default",
                    display_name="Default Ops Graph",
                    status="draft",
                    query_ready=False,
                    source_doc_ids=["doc-arch"],
                ),
                GraphIndexRecord(
                    graph_id="rfp_corpus",
                    tenant_id="tenant",
                    collection_id="rfp-corpus",
                    display_name="RFP Corpus Graph",
                    status="ready",
                    query_ready=True,
                    domain_summary="Graph index for cross-document RFP entity and requirement analysis",
                    source_doc_ids=["doc-other"],
                )
            ]
        ),
    )

    contract = run_rag_contract(
        _settings(),
        stores,
        providers=SimpleNamespace(chat=SimpleNamespace(invoke=lambda *args, **kwargs: None), judge=SimpleNamespace()),
        session=SimpleNamespace(tenant_id="tenant", scratchpad={}, metadata={"kb_collection_id": "default"}),
        query="what knowledge bases do you have access to",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
    )

    assert "Knowledge base collections available to this chat:" in contract.answer
    assert "default\nThe default knowledge-base collection - 1 indexed document covering product overviews and architecture." in contract.answer
    assert (
        "rfp-corpus\nA knowledge-base collection - 1 indexed document covering product overviews."
    ) in contract.answer
    assert "Knowledge graphs available to this chat:" in contract.answer
    assert (
        "Default Ops Graph (`default_ops_graph`)\nGraph index over default, draft, covering 1 source document."
    ) in contract.answer
    assert (
        "RFP Corpus Graph (`rfp_corpus`)\nGraph index for cross-document RFP entity and requirement analysis."
    ) in contract.answer
    assert "Reply with one or more ids or `use all`" in contract.answer


def test_run_rag_contract_uses_graph_inventory_answer_for_graph_access_queries(monkeypatch):
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.run_retrieval_controller",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieval should not run for graph inventory")),
    )
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.inventory.get_collection_readiness_status",
        lambda *args, **kwargs: SimpleNamespace(ready=True, maintenance_policy=""),
    )

    stores = SimpleNamespace(
        doc_store=SimpleNamespace(
            list_documents=lambda source_type="", tenant_id="tenant", collection_id="": [
                record
                for record in [
                    SimpleNamespace(
                        doc_id="doc-default",
                        title="ARCHITECTURE.md",
                        source_type="kb",
                        source_path="/repo/docs/ARCHITECTURE.md",
                        collection_id="default",
                        file_type="md",
                        doc_structure_type="general",
                        num_chunks=12,
                    ),
                    SimpleNamespace(
                        doc_id="doc-rfp",
                        title="RFP Overview.docx",
                        source_type="host_path",
                        source_path="/repo/docs/RFP Overview.docx",
                        collection_id="rfp-corpus",
                        file_type="docx",
                        doc_structure_type="general",
                        num_chunks=9,
                    ),
                ]
                if (not source_type or record.source_type == source_type)
                and (not collection_id or record.collection_id == collection_id)
            ]
        ),
        graph_index_store=_GraphIndexStore(
            [
                GraphIndexRecord(
                    graph_id="default_ops_graph",
                    tenant_id="tenant",
                    collection_id="default",
                    display_name="Default Ops Graph",
                    status="draft",
                    query_ready=False,
                    source_doc_ids=["doc-default"],
                ),
                GraphIndexRecord(
                    graph_id="rfp_corpus",
                    tenant_id="tenant",
                    collection_id="rfp-corpus",
                    display_name="RFP Corpus Graph",
                    status="ready",
                    query_ready=True,
                    domain_summary="Graph index for cross-document RFP entity and requirement analysis",
                    source_doc_ids=["doc-rfp"],
                ),
            ]
        ),
    )

    contract = run_rag_contract(
        _settings(),
        stores,
        providers=SimpleNamespace(chat=SimpleNamespace(invoke=lambda *args, **kwargs: None), judge=SimpleNamespace()),
        session=SimpleNamespace(tenant_id="tenant", scratchpad={}, metadata={"kb_collection_id": "default"}),
        query="what graphs do i have access to",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
    )

    assert "Knowledge graphs available to this chat:" in contract.answer
    assert "Default Ops Graph (`default_ops_graph`)" in contract.answer
    assert "RFP Corpus Graph (`rfp_corpus`)" in contract.answer
    assert "Knowledge base collections available to this chat:" not in contract.answer


def test_run_rag_contract_returns_namespace_clarification_for_ambiguous_bare_namespace(monkeypatch):
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.run_retrieval_controller",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieval should not run for namespace clarification")),
    )
    stores = SimpleNamespace(
        doc_store=SimpleNamespace(
            list_documents=lambda source_type="", tenant_id="tenant", collection_id="": [
                record
                for record in [
                    SimpleNamespace(
                        doc_id="doc-rfp",
                        title="RFP Overview.docx",
                        source_type="host_path",
                        source_path="/rfp/RFP Overview.docx",
                        collection_id="rfp-corpus",
                        file_type="docx",
                        doc_structure_type="general",
                        num_chunks=9,
                    )
                ]
                if (not source_type or record.source_type == source_type)
                and (not collection_id or record.collection_id == collection_id)
            ]
        ),
        graph_index_store=_GraphIndexStore(
            [
                GraphIndexRecord(
                    graph_id="rfp_corpus",
                    tenant_id="tenant",
                    collection_id="rfp-corpus",
                    display_name="RFP Corpus Graph",
                    status="ready",
                    query_ready=True,
                )
            ]
        ),
    )
    session = SimpleNamespace(tenant_id="tenant", scratchpad={}, metadata={"kb_collection_id": "default"})

    contract = run_rag_contract(
        _settings(),
        stores,
        providers=SimpleNamespace(chat=SimpleNamespace(invoke=lambda *args, **kwargs: None), judge=SimpleNamespace()),
        session=session,
        query="can you list out the documents in rfp-corpus. I just want to know the titles of the documents",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
    )

    assert "multiple visible namespaces matching `rfp-corpus`" in contract.answer
    assert "Collections:" in contract.answer
    assert "Graphs:" in contract.answer
    assert "use all" in contract.followups
    assert contract.warnings == ["NAMESPACE_SCOPE_SELECTION_REQUIRED"]


def test_run_rag_contract_returns_direct_kb_inventory_for_single_namespace_match(monkeypatch):
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.run_retrieval_controller",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieval should not run for bare namespace inventory")),
    )
    stores = SimpleNamespace(
        doc_store=SimpleNamespace(
            list_documents=lambda source_type="", tenant_id="tenant", collection_id="": [
                record
                for record in [
                    SimpleNamespace(
                        doc_id="doc-rfp",
                        title="RFP Overview.docx",
                        source_type="host_path",
                        source_path="/rfp/RFP Overview.docx",
                        collection_id="rfp-corpus",
                        file_type="docx",
                        doc_structure_type="general",
                        num_chunks=9,
                    )
                ]
                if (not source_type or record.source_type == source_type)
                and (not collection_id or record.collection_id == collection_id)
            ]
        ),
        graph_index_store=_GraphIndexStore([]),
    )

    contract = run_rag_contract(
        _settings(),
        stores,
        providers=SimpleNamespace(chat=SimpleNamespace(invoke=lambda *args, **kwargs: None), judge=SimpleNamespace()),
        session=SimpleNamespace(tenant_id="tenant", scratchpad={}, metadata={"kb_collection_id": "default"}),
        query="can you list out the documents in rfp-corpus. I just want to know the titles of the documents",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
    )

    assert "Knowledge base documents currently indexed in collection `rfp-corpus`" in contract.answer
    assert "RFP Overview.docx" in contract.answer
    assert contract.warnings == []


def test_run_rag_contract_returns_graph_source_inventory_for_explicit_graph_queries(monkeypatch):
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.run_retrieval_controller",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieval should not run for graph inventory")),
    )
    stores = SimpleNamespace(
        doc_store=SimpleNamespace(list_documents=lambda **kwargs: []),
        graph_index_store=_GraphIndexStore(
            [
                GraphIndexRecord(
                    graph_id="rfp_corpus",
                    tenant_id="tenant",
                    collection_id="rfp-corpus",
                    display_name="RFP Corpus Graph",
                    status="ready",
                    query_ready=True,
                )
            ]
        ),
        graph_source_store=_GraphSourceStore(
            [
                GraphIndexSourceRecord(
                    graph_source_id="src-1",
                    graph_id="rfp_corpus",
                    tenant_id="tenant",
                    source_doc_id="DOC-1",
                    source_path="/rfp/A.pdf",
                    source_title="A.pdf",
                    source_type="host_path",
                )
            ]
        ),
    )

    contract = run_rag_contract(
        _settings(),
        stores,
        providers=SimpleNamespace(chat=SimpleNamespace(invoke=lambda *args, **kwargs: None), judge=SimpleNamespace()),
        session=SimpleNamespace(tenant_id="tenant", scratchpad={}, metadata={"kb_collection_id": "default"}),
        query="what documents are in the rfp_corpus graph",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
    )

    assert "Knowledge graph source documents currently indexed in graph RFP Corpus Graph (`rfp_corpus`)" in contract.answer
    assert "A.pdf (doc_id=DOC-1; source_type=host_path; path=/rfp/A.pdf)" in contract.answer
    assert "RFP Corpus Graph" in contract.answer
    assert "collection: rfp-corpus" in contract.answer
    assert "query-ready" in contract.answer
    assert contract.used_citation_ids == []
    assert contract.citations == []
    assert contract.warnings == []
    assert contract.retrieval_summary.search_mode == "metadata_inventory"


def test_run_rag_contract_requests_collection_selection_when_multiple_kb_collections_are_visible():
    stores = SimpleNamespace(
        doc_store=SimpleNamespace(
            list_documents=lambda source_type="", tenant_id="tenant", collection_id="": [
                record
                for record in [
                    SimpleNamespace(
                        doc_id="doc-default",
                        title="ARCHITECTURE.md",
                        source_type="kb",
                        source_path="/repo/docs/ARCHITECTURE.md",
                        collection_id="default",
                        file_type="md",
                        doc_structure_type="general",
                        num_chunks=12,
                    ),
                    SimpleNamespace(
                        doc_id="doc-rfp",
                        title="Asterion Overview.docx",
                        source_type="host_path",
                        source_path="/rfp/Asterion Overview.docx",
                        collection_id="rfp-corpus",
                        file_type="docx",
                        doc_structure_type="general",
                        num_chunks=9,
                    ),
                ]
                if (not source_type or record.source_type == source_type)
                and (not collection_id or record.collection_id == collection_id)
            ]
        )
    )

    session = SimpleNamespace(
        tenant_id="tenant",
        scratchpad={},
        metadata={"kb_collection_id": "default", "kb_collection_confirmed": False},
    )

    contract = run_rag_contract(
        _settings(),
        stores,
        providers=SimpleNamespace(chat=SimpleNamespace(invoke=lambda *args, **kwargs: None), judge=SimpleNamespace()),
        session=session,
        query="Explain the approval workflow and cite your sources.",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
    )

    assert "multiple knowledge base collections" in contract.answer
    assert "`default`" in contract.answer
    assert "`rfp-corpus`" in contract.answer
    assert contract.followups == ["default", "rfp-corpus"]
    assert contract.warnings == ["KB_COLLECTION_SELECTION_REQUIRED"]


def test_run_rag_contract_uses_session_access_inventory_answer_for_short_access_queries(monkeypatch):
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.run_retrieval_controller",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieval should not run for session access inventory")),
    )
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.inventory.get_kb_coverage_status",
        lambda *args, **kwargs: SimpleNamespace(ready=True, configured_source_paths=("/repo/docs/ARCHITECTURE.md",)),
    )

    def list_documents(source_type="", tenant_id="tenant", collection_id="default"):
        del tenant_id
        records = [
            SimpleNamespace(
                doc_id="doc-arch",
                title="ARCHITECTURE.md",
                source_type="kb",
                source_path="/repo/docs/ARCHITECTURE.md",
                collection_id="default",
                file_type="md",
                doc_structure_type="general",
                num_chunks=12,
            ),
            SimpleNamespace(
                doc_id="upload-1",
                title="contract.pdf",
                source_type="upload",
                source_path="/uploads/contract.pdf",
                collection_id="owui-chat-1",
                file_type="pdf",
                doc_structure_type="general",
                num_chunks=4,
            ),
            SimpleNamespace(
                doc_id="upload-other",
                title="other.pdf",
                source_type="upload",
                source_path="/uploads/other.pdf",
                collection_id="owui-chat-2",
                file_type="pdf",
                doc_structure_type="general",
                num_chunks=3,
            ),
        ]
        return [
            record
            for record in records
            if (not source_type or record.source_type == source_type)
            and (not collection_id or record.collection_id == collection_id)
        ]

    stores = SimpleNamespace(doc_store=SimpleNamespace(list_documents=list_documents))

    contract = run_rag_contract(
        _settings(),
        stores,
        providers=SimpleNamespace(chat=SimpleNamespace(invoke=lambda *args, **kwargs: None), judge=SimpleNamespace()),
        session=SimpleNamespace(
            tenant_id="tenant",
            scratchpad={},
            metadata={"upload_collection_id": "owui-chat-1", "kb_collection_id": "default"},
            uploaded_doc_ids=["upload-1"],
        ),
        query="what documents do we have access to",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
    )

    assert "This chat can currently use:" in contract.answer
    assert "Knowledge base collection: default (1 indexed document)" in contract.answer
    assert "Current chat uploads (1 total):" in contract.answer
    assert "contract.pdf (doc_id=upload-1; file_type=pdf; chunks=4)" in contract.answer
    assert "other.pdf" not in contract.answer
    assert "search uploaded docs" in contract.followups
    assert contract.used_citation_ids == []
    assert contract.citations == []
    assert contract.warnings == []
    assert contract.retrieval_summary.search_mode == "metadata_inventory"


def test_run_rag_contract_reports_negative_evidence_for_exhaustive_search(monkeypatch):
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.run_retrieval_controller",
        lambda *args, **kwargs: SimpleNamespace(
            selected_docs=[],
            candidate_docs=[],
            graded=[],
            query_used="Find all policies that mention moonlight approvals.",
            search_mode="deep",
            rounds=3,
            tool_calls_used=5,
            tool_call_log=["round1:search_corpus[hybrid]:moonlight approvals"],
            strategies_used=["hybrid", "keyword"],
            candidate_counts={"unique_docs": 4, "selected_docs": 0},
            evidence_ledger={"round_summaries": [], "entries": []},
            parallel_workers_used=False,
            to_summary=lambda citations_found: SimpleNamespace(
                to_dict=lambda: {},
                query_used="Find all policies that mention moonlight approvals.",
                steps=5,
                tool_calls_used=5,
                tool_call_log=["round1:search_corpus[hybrid]:moonlight approvals"],
                citations_found=citations_found,
                search_mode="deep",
                rounds=3,
                strategies_used=["hybrid", "keyword"],
                candidate_counts={"unique_docs": 4, "selected_docs": 0},
                parallel_workers_used=False,
            ),
        ),
    )
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.get_kb_coverage_status",
        lambda *args, **kwargs: SimpleNamespace(ready=True),
    )

    contract = run_rag_contract(
        _settings(),
        SimpleNamespace(),
        providers=SimpleNamespace(chat=SimpleNamespace(invoke=lambda *args, **kwargs: None), judge=SimpleNamespace()),
        session=SimpleNamespace(tenant_id="tenant", scratchpad={}),
        query="Find all policies that mention moonlight approvals.",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
        coverage_goal="exhaustive",
        controller_hints={"prefer_negative_evidence_reporting": True},
    )

    assert "could not find enough grounded evidence" in contract.answer.lower()
    assert "moonlight approvals" in contract.answer
    assert contract.used_citation_ids == []
    assert "INSUFFICIENT_CORPUS_EVIDENCE" in contract.warnings


def test_run_rag_contract_scopes_explicit_named_docs_and_keeps_both_docs(monkeypatch):
    left_doc = Document(
        page_content="RuntimeKernel is the stable runtime facade.",
        metadata={
            "doc_id": "doc-arch",
            "chunk_id": "doc-arch#chunk0001",
            "title": "ARCHITECTURE.md",
            "source_type": "kb",
            "chunk_index": 1,
        },
    )
    right_doc = Document(
        page_content="C4 container view shows RuntimeService and AgentRegistry relationships.",
        metadata={
            "doc_id": "doc-c4",
            "chunk_id": "doc-c4#chunk0001",
            "title": "C4_ARCHITECTURE.md",
            "source_type": "kb",
            "chunk_index": 1,
        },
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.resolve_query_document_targets",
        lambda settings, stores, session, *, query: IndexedDocResolution(
            requested_names=("ARCHITECTURE.md", "C4_ARCHITECTURE.md"),
            resolved=(
                ResolvedIndexedDoc(
                    doc_id="doc-arch",
                    title="ARCHITECTURE.md",
                    source_type="kb",
                    source_path="/repo/docs/ARCHITECTURE.md",
                    collection_id="default",
                    file_type="md",
                    doc_structure_type="general",
                    match_name="ARCHITECTURE.md",
                    match_type="title_exact",
                ),
                ResolvedIndexedDoc(
                    doc_id="doc-c4",
                    title="C4_ARCHITECTURE.md",
                    source_type="kb",
                    source_path="/repo/docs/C4_ARCHITECTURE.md",
                    collection_id="default",
                    file_type="md",
                    doc_structure_type="general",
                    match_name="C4_ARCHITECTURE.md",
                    match_type="title_exact",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.run_retrieval_controller",
        lambda *args, **kwargs: (
            captured.update(
                {
                    "preferred_doc_ids": list(kwargs.get("preferred_doc_ids") or []),
                    "controller_hints": dict(kwargs.get("controller_hints") or {}),
                }
            )
            or SimpleNamespace(
                selected_docs=[left_doc],
                candidate_docs=[left_doc, right_doc],
                graded=[
                    GradedChunk(doc=left_doc, relevance=3, reason="title_hint"),
                    GradedChunk(doc=right_doc, relevance=3, reason="title_hint"),
                ],
                query_used="Compare ARCHITECTURE.md and C4_ARCHITECTURE.md",
                search_mode="deep",
                rounds=2,
                tool_calls_used=4,
                tool_call_log=[],
                strategies_used=["hybrid", "read_document"],
                candidate_counts={"unique_docs": 2, "selected_docs": 1},
                evidence_ledger={"round_summaries": [], "entries": []},
                parallel_workers_used=False,
                to_summary=lambda citations_found: SimpleNamespace(
                    to_dict=lambda: {},
                    query_used="Compare ARCHITECTURE.md and C4_ARCHITECTURE.md",
                    steps=4,
                    tool_calls_used=4,
                    tool_call_log=[],
                    citations_found=citations_found,
                    search_mode="deep",
                    rounds=2,
                    strategies_used=["hybrid", "read_document"],
                    candidate_counts={"unique_docs": 2, "selected_docs": 1},
                    parallel_workers_used=False,
                ),
            )
        ),
    )
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.get_kb_coverage_status",
        lambda *args, **kwargs: SimpleNamespace(ready=True),
    )
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.generate_grounded_answer",
        lambda *args, **kwargs: {
            "answer": "ARCHITECTURE.md explains runtime responsibilities, while C4_ARCHITECTURE.md focuses on C4 relationships.",
            "used_citation_ids": ["doc-arch#chunk0001", "doc-c4#chunk0001"],
            "followups": [],
            "warnings": [],
            "confidence_hint": 0.82,
        },
    )

    contract = run_rag_contract(
        _settings(),
        SimpleNamespace(),
        providers=SimpleNamespace(chat=SimpleNamespace(invoke=lambda *args, **kwargs: None), judge=SimpleNamespace()),
        session=SimpleNamespace(tenant_id="tenant", scratchpad={}, metadata={}),
        query="Compare ARCHITECTURE.md and C4_ARCHITECTURE.md in detail.",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
    )

    assert captured["preferred_doc_ids"] == ["doc-arch", "doc-c4"]
    assert captured["controller_hints"]["resolved_doc_ids"] == ["doc-arch", "doc-c4"]
    assert {citation.doc_id for citation in contract.citations} == {"doc-arch", "doc-c4"}


def test_run_rag_contract_returns_clear_answer_when_named_docs_are_missing(monkeypatch):
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.resolve_query_document_targets",
        lambda settings, stores, session, *, query: IndexedDocResolution(
            requested_names=("ARCHITECTURE.md", "C4_ARCHITECTURE.md"),
            missing=(
                MissingIndexedDocMatch(requested_name="ARCHITECTURE.md"),
                MissingIndexedDocMatch(requested_name="C4_ARCHITECTURE.md"),
            ),
        ),
    )
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.get_kb_coverage_status",
        lambda *args, **kwargs: SimpleNamespace(ready=True),
    )

    contract = run_rag_contract(
        _settings(),
        SimpleNamespace(),
        providers=SimpleNamespace(chat=SimpleNamespace(invoke=lambda *args, **kwargs: None), judge=SimpleNamespace()),
        session=SimpleNamespace(tenant_id="tenant", scratchpad={}, metadata={}),
        query="Compare ARCHITECTURE.md and C4_ARCHITECTURE.md in detail.",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
    )

    assert "could not find indexed documents matching" in contract.answer.lower()
    assert "ARCHITECTURE.md" in contract.answer
    assert "C4_ARCHITECTURE.md" in contract.answer
    assert contract.warnings == ["REQUESTED_DOCS_NOT_INDEXED"]


def test_requested_doc_resolution_answer_includes_candidate_scope_collection_and_path() -> None:
    resolution = IndexedDocResolution(
        ambiguous=(
            AmbiguousIndexedDocMatch(
                requested_name="ARCHITECTURE.md",
                candidates=(
                    ResolvedIndexedDoc(
                        doc_id="doc-kb",
                        title="ARCHITECTURE.md",
                        source_type="kb",
                        source_path="/repo/docs/ARCHITECTURE.md",
                        collection_id="default",
                        file_type="md",
                        doc_structure_type="general",
                        match_name="ARCHITECTURE.md",
                        match_type="title_exact",
                    ),
                    ResolvedIndexedDoc(
                        doc_id="doc-upload",
                        title="ARCHITECTURE.md",
                        source_type="upload",
                        source_path="/workspace/ARCHITECTURE.md",
                        collection_id="owui-chat-1",
                        file_type="md",
                        doc_structure_type="general",
                        match_name="ARCHITECTURE.md",
                        match_type="title_exact",
                    ),
                ),
            ),
        ),
    )

    payload = _requested_doc_resolution_answer(resolution)

    assert "ARCHITECTURE.md" in payload["answer"]
    assert "scope kb" in payload["answer"]
    assert "collection default" in payload["answer"]
    assert "path /repo/docs/ARCHITECTURE.md" in payload["answer"]
    assert "scope upload" in payload["answer"]
    assert "collection owui-chat-1" in payload["answer"]
    assert "path /workspace/ARCHITECTURE.md" in payload["answer"]
    assert payload["warnings"] == ["REQUESTED_DOCS_AMBIGUOUS"]
