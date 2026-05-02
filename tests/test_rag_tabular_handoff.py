from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.providers import ProviderBundle
from agentic_chatbot_next.rag.adaptive import RetrievalRun
from agentic_chatbot_next.rag.engine import run_rag_contract
from agentic_chatbot_next.rag.fanout import TabularEvidenceBatchResult, TabularEvidenceResult
from agentic_chatbot_next.rag.retrieval import GradedChunk
from agentic_chatbot_next.rag.tabular import (
    plan_tabular_evidence_tasks,
    tabular_evidence_results_to_documents,
)


class _ChatModel:
    def __init__(self, text: str = "{}") -> None:
        self.text = text

    def invoke(self, messages, config=None):
        del messages, config
        return SimpleNamespace(content=self.text)


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        default_collection_id="default",
        default_tenant_id="tenant",
        rag_tabular_handoff_max_tasks=2,
        rag_extractive_fallback_enabled=True,
        rag_budget_synthesis_reserve_ms=30_000,
        graph_search_enabled=False,
        clarification_sensitivity=50,
        grounded_answer_prompt_path=Path("missing"),
        prompts_backend="local",
    )


def _ready_status() -> SimpleNamespace:
    return SimpleNamespace(
        ready=True,
        sync_error="",
        missing_source_paths=[],
        collection_id="default",
        reason="ready",
        suggested_fix="",
    )


def test_plan_tabular_evidence_tasks_triggers_for_spreadsheet_lookup() -> None:
    doc = Document(
        page_content="Workbook row",
        metadata={
            "doc_id": "doc-xlsx",
            "title": "tracker.xlsx",
            "file_type": "xlsx",
            "sheet_name": "IMS",
            "row_start": 5,
            "row_end": 5,
            "cell_range": "A5:E5",
        },
    )

    tasks = plan_tabular_evidence_tasks("What is the current approved CDR date?", [doc])

    assert len(tasks) == 1
    assert tasks[0].doc_id == "doc-xlsx"
    assert tasks[0].sheet_hints == ["IMS"]
    assert tasks[0].cell_ranges == ["A5:E5"]
    assert "lookup" in tasks[0].requested_operations


def test_plan_tabular_evidence_tasks_skips_non_tabular_docs() -> None:
    doc = Document(page_content="Policy text", metadata={"doc_id": "doc-md", "title": "policy.md", "file_type": "md"})

    assert plan_tabular_evidence_tasks("What is the current status?", [doc]) == []


def test_tabular_evidence_results_to_documents_preserves_source_refs() -> None:
    task = plan_tabular_evidence_tasks(
        "Summarize status by supplier",
        [
            Document(
                page_content="sheet summary",
                metadata={"doc_id": "doc-1", "title": "suppliers.xlsx", "file_type": "xlsx", "sheet_name": "Status"},
            )
        ],
    )[0]
    result = TabularEvidenceResult(
        task_id=task.task_id,
        summary="North Coast is late by 12 days.",
        findings=[{"summary": "North Coast is late by 12 days.", "value": 12}],
        source_refs=[
            {
                "doc_id": "doc-1",
                "title": "suppliers.xlsx",
                "sheet_name": "Status",
                "row_start": 7,
                "row_end": 7,
                "cell_range": "A7:F7",
                "columns": ["Supplier", "Variance"],
            }
        ],
        operations=["profile_dataset", "execute_code"],
        confidence=0.88,
    )

    docs = tabular_evidence_results_to_documents([result], [task])

    assert len(docs) == 1
    assert docs[0].metadata["chunk_type"] == "tabular_analysis"
    assert docs[0].metadata["sheet_name"] == "Status"
    assert docs[0].metadata["row_start"] == 7
    assert docs[0].metadata["cell_range"] == "A7:F7"
    assert "North Coast is late" in docs[0].page_content


def test_run_rag_contract_adds_tabular_analyst_evidence_before_synthesis(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    providers = ProviderBundle(chat=_ChatModel(), judge=_ChatModel(), embeddings=object())
    spreadsheet_doc = Document(
        page_content="Workbook: tracker.xlsx | Sheet: IMS | Row 2: Milestone: CDR; Current Date: 2028-09-26",
        metadata={
            "doc_id": "doc-final",
            "chunk_id": "doc-final#chunk0001",
            "title": "tracker.xlsx",
            "source_type": "kb",
            "source_path": str(tmp_path / "tracker.xlsx"),
            "file_type": "xlsx",
            "sheet_name": "IMS",
            "row_start": 2,
            "row_end": 2,
            "cell_range": "A2:C2",
        },
    )
    retrieval_run = RetrievalRun(
        selected_docs=[spreadsheet_doc],
        candidate_docs=[spreadsheet_doc],
        graded=[GradedChunk(doc=spreadsheet_doc, relevance=3, reason="test")],
        query_used="What is the current approved CDR date?",
        search_mode="fast",
        rounds=1,
        tool_calls_used=1,
        tool_call_log=["fast"],
        strategies_used=["hybrid"],
        candidate_counts={"selected_docs": 1},
    )
    captured: dict[str, object] = {}

    class _Selection:
        resolved = True
        selected_collection_id = "default"

        def to_dict(self):
            return {"selected_collection_id": "default"}

    class _Bridge:
        def run_tabular_evidence_tasks(self, tasks):
            captured["tasks"] = list(tasks)
            return TabularEvidenceBatchResult(
                results=[
                    TabularEvidenceResult(
                        task_id=tasks[0].task_id,
                        summary="The current approved CDR date is 2028-09-26.",
                        findings=[{"summary": "Current approved CDR date: 2028-09-26"}],
                        source_refs=[
                            {
                                "doc_id": "doc-final",
                                "title": "tracker.xlsx",
                                "sheet_name": "IMS",
                                "row_start": 2,
                                "row_end": 2,
                                "cell_range": "A2:C2",
                                "columns": ["Milestone", "Current Date"],
                            }
                        ],
                        operations=["profile_dataset"],
                        confidence=0.9,
                    )
                ]
            )

    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.sync_session_kb_collection_state",
        lambda *args, **kwargs: {
            "kb_collection_id": "default",
            "available_kb_collection_ids": ["default"],
            "kb_collection_confirmed": True,
        },
    )
    monkeypatch.setattr("agentic_chatbot_next.rag.engine.get_collection_readiness_status", lambda *args, **kwargs: _ready_status())
    monkeypatch.setattr("agentic_chatbot_next.rag.engine.select_collection_for_query", lambda *args, **kwargs: _Selection())
    monkeypatch.setattr("agentic_chatbot_next.rag.engine.apply_selection_to_session", lambda *args, **kwargs: None)
    monkeypatch.setattr("agentic_chatbot_next.rag.engine.run_retrieval_controller", lambda *args, **kwargs: retrieval_run)

    def fake_generate_grounded_answer(llm, *, question, evidence_docs, **kwargs):
        del llm, question, kwargs
        tabular_doc = next(doc for doc in evidence_docs if doc.metadata.get("chunk_type") == "tabular_analysis")
        captured["tabular_chunk_id"] = tabular_doc.metadata["chunk_id"]
        return {
            "answer": f"The current approved CDR date is 2028-09-26 ({tabular_doc.metadata['chunk_id']}).",
            "used_citation_ids": [tabular_doc.metadata["chunk_id"]],
            "followups": [],
            "warnings": [],
            "confidence_hint": 0.9,
        }

    monkeypatch.setattr("agentic_chatbot_next.rag.engine.generate_grounded_answer", fake_generate_grounded_answer)

    contract = run_rag_contract(
        settings,
        SimpleNamespace(doc_store=SimpleNamespace(), chunk_store=SimpleNamespace(), graph_store=None),
        providers=providers,
        session=session,
        query="What is the current approved CDR date?",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=2,
        top_k_keyword=2,
        max_retries=1,
        runtime_bridge=_Bridge(),
    )

    assert captured["tasks"][0].doc_id == "doc-final"
    assert captured["tabular_chunk_id"] in contract.used_citation_ids
    assert contract.citations[0].location == "IMS row 2 A2:C2"
    assert "tabular_analyst" in contract.retrieval_summary.strategies_used
