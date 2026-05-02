from __future__ import annotations

import json
from types import SimpleNamespace

from langchain_core.documents import Document

from agentic_chatbot_next.tools.groups import rag_workbench


class FakeRetrievalAdapter:
    calls: list[tuple[str, dict]] = []

    def __init__(self, stores, *, settings, session):
        self.stores = stores
        self.settings = settings
        self.session = session

    def search_corpus(self, query, **kwargs):
        self.calls.append(("search", {"query": query, **kwargs}))
        return [
            SimpleNamespace(
                doc=Document(
                    page_content="hybrid evidence",
                    metadata={"chunk_id": "chunk-1", "doc_id": "doc-1", "chunk_index": 2},
                ),
                score=0.91,
                method="hybrid",
            )
        ]

    def grep_corpus(self, query, **kwargs):
        self.calls.append(("grep", {"query": query, **kwargs}))
        return [
            SimpleNamespace(
                doc=Document(
                    page_content="keyword evidence",
                    metadata={"chunk_id": "chunk-2", "doc_id": "doc-2", "chunk_index": 4},
                ),
                score=0.82,
                method="keyword",
            )
        ]

    def fetch_chunk_window(self, chunk_id, *, before=1, after=1):
        self.calls.append(("window", {"chunk_id": chunk_id, "before": before, "after": after}))
        return [
            Document(
                page_content="neighbor evidence",
                metadata={"chunk_id": "chunk-3", "doc_id": "doc-3", "chunk_index": 5},
            )
        ]

    def search_section_scope(self, query, *, doc_ids, prioritized_sections=None, limit=8):
        self.calls.append(
            (
                "sections",
                {
                    "query": query,
                    "doc_ids": list(doc_ids),
                    "prioritized_sections": list(prioritized_sections or []),
                    "limit": limit,
                },
            )
        )
        return [
            Document(
                page_content="section evidence",
                metadata={
                    "chunk_id": "chunk-4",
                    "doc_id": "doc-4",
                    "chunk_index": 6,
                    "section_title": "Schedule",
                },
            )
        ]

    def outline_scan(self, doc_id, *, max_chunks=8):
        self.calls.append(("outline_scan", {"doc_id": doc_id, "max_chunks": max_chunks}))
        return [
            Document(
                page_content="outline sample",
                metadata={"chunk_id": "chunk-outline", "doc_id": doc_id, "chunk_index": 0},
            )
        ]


class FakeChunkStore:
    def get_structure_outline(self, doc_id, tenant_id="tenant"):
        return [
            {
                "clause_number": "1.0",
                "section_title": "Overview",
                "chunk_type": "section",
                "chunk_index": 0,
            },
            {
                "clause_number": "2.0",
                "section_title": "Schedule",
                "chunk_type": "section",
                "chunk_index": 3,
            },
        ]


class FakeDocRecord:
    doc_id = "doc-filtered"
    title = "Asterion Schedule Workbook"
    source_type = "kb"
    source_path = "asterion_schedule.xlsx"
    file_type = "xlsx"
    doc_structure_type = "workbook"
    collection_id = "defense"
    num_chunks = 12


class FakeDocStore:
    def search_by_metadata(self, **kwargs):
        self.kwargs = dict(kwargs)
        return [FakeDocRecord()]


class FakeJudge:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls: list[str] = []

    def invoke(self, prompt, config=None):
        del config
        self.calls.append(str(prompt))
        payload = self.responses.pop(0) if self.responses else {}
        return SimpleNamespace(content=json.dumps(payload))


def _ctx(*, judge=None) -> SimpleNamespace:
    return SimpleNamespace(
        settings=SimpleNamespace(),
        stores=SimpleNamespace(doc_store=FakeDocStore(), chunk_store=FakeChunkStore()),
        session_handle=SimpleNamespace(tenant_id="tenant", metadata={}),
        providers=SimpleNamespace(judge=judge),
        callbacks=[],
    )


def test_rag_workbench_tools_wrap_adapter(monkeypatch) -> None:
    FakeRetrievalAdapter.calls = []
    monkeypatch.setattr(rag_workbench, "CorpusRetrievalAdapter", FakeRetrievalAdapter)
    tools = {tool.name: tool for tool in rag_workbench.build_rag_workbench_tools(_ctx())}

    search_result = tools["search_corpus_chunks"].invoke(
        {
            "query": "renewal terms",
            "strategy": "hybrid",
            "preferred_doc_ids_csv": "doc-1, doc-2",
            "collection_id": "contracts",
            "limit": 3,
        }
    )
    grep_result = tools["grep_corpus_chunks"].invoke({"query": "force majeure"})
    window_result = tools["fetch_chunk_window"].invoke({"chunk_id": "chunk-1", "before": 2, "after": 9})

    assert search_result["results"][0]["chunk_id"] == "chunk-1"
    assert search_result["results"][0]["score"] == 0.91
    assert grep_result["results"][0]["method"] == "keyword"
    assert window_result["chunks"][0]["content"] == "neighbor evidence"
    assert FakeRetrievalAdapter.calls[0][1]["preferred_doc_ids"] == ["doc-1", "doc-2"]
    assert FakeRetrievalAdapter.calls[2] == ("window", {"chunk_id": "chunk-1", "before": 2, "after": 5})


def test_search_corpus_chunks_rejects_unknown_strategy(monkeypatch) -> None:
    monkeypatch.setattr(rag_workbench, "CorpusRetrievalAdapter", FakeRetrievalAdapter)
    tools = {tool.name: tool for tool in rag_workbench.build_rag_workbench_tools(_ctx())}

    result = tools["search_corpus_chunks"].invoke({"query": "x", "strategy": "graph"})

    assert result["results"] == []
    assert "strategy must be one of" in result["error"]


def test_plan_rag_queries_uses_judge_and_bounds_output(monkeypatch) -> None:
    monkeypatch.setattr(rag_workbench, "CorpusRetrievalAdapter", FakeRetrievalAdapter)
    judge = FakeJudge(
        [
            {
                "queries": [
                    {"facet": "semantic", "query": "approved CDR move", "strategy": "hybrid", "rationale": "broad"},
                    {"facet": "exact_terms", "query": "26 Sep 2028", "strategy": "keyword", "rationale": "date"},
                    {"facet": "bad", "query": "bad", "strategy": "graph", "rationale": "coerce"},
                ]
            }
        ]
    )
    tools = {tool.name: tool for tool in rag_workbench.build_rag_workbench_tools(_ctx(judge=judge))}

    result = tools["plan_rag_queries"].invoke(
        {
            "query": "Why did approved CDR move?",
            "collection_id": "defense",
            "preferred_doc_ids_csv": "doc-a",
            "max_queries": 2,
        }
    )

    assert [item["facet"] for item in result["queries"]] == ["semantic", "exact_terms"]
    assert result["queries"][0]["collection_id"] == "defense"
    assert result["queries"][0]["preferred_doc_ids"] == ["doc-a"]
    assert "answer" not in result


def test_structure_filter_and_section_tools_wrap_existing_primitives(monkeypatch) -> None:
    FakeRetrievalAdapter.calls = []
    monkeypatch.setattr(rag_workbench, "CorpusRetrievalAdapter", FakeRetrievalAdapter)
    tools = {tool.name: tool for tool in rag_workbench.build_rag_workbench_tools(_ctx())}

    structure = tools["inspect_document_structure"].invoke({"doc_id": "doc-a", "max_items": 1})
    filtered = tools["filter_indexed_docs"].invoke({"collection_id": "defense", "file_type": "xlsx"})
    sections = tools["search_document_sections"].invoke(
        {
            "query": "approved current date",
            "doc_ids_csv": "doc-a, doc-b",
            "prioritized_sections_json": '[{"match_type":"sheet_name","value":"IMS"}]',
            "limit": 3,
        }
    )

    assert structure["outline_count"] == 2
    assert structure["truncated"] is True
    assert filtered["documents"][0]["doc_id"] == "doc-filtered"
    assert sections["results"][0]["section_title"] == "Schedule"
    assert FakeRetrievalAdapter.calls[-1][0] == "sections"
    assert FakeRetrievalAdapter.calls[-1][1]["prioritized_sections"] == [{"match_type": "sheet_name", "value": "IMS"}]


def test_grade_prune_validate_and_controller_hints_handle_bad_json(monkeypatch) -> None:
    monkeypatch.setattr(rag_workbench, "CorpusRetrievalAdapter", FakeRetrievalAdapter)
    tools = {tool.name: tool for tool in rag_workbench.build_rag_workbench_tools(_ctx())}

    bad_grade = tools["grade_evidence_candidates"].invoke({"query": "x", "candidates_json": "{bad"})
    assert bad_grade["grades"] == []
    assert "invalid_json_input" in bad_grade["warnings"]

    candidates = {
        "results": [
            {
                "chunk_id": "chunk-a",
                "doc_id": "doc-a",
                "title": "Final status",
                "content": "Approved current CDR date moved to 26 Sep 2028.",
                "score": 0.9,
            },
            {
                "chunk_id": "chunk-b",
                "doc_id": "doc-a",
                "title": "Draft note",
                "content": "Draft CDR date was 14 Aug 2028.",
                "score": 0.6,
                "conflict": True,
            },
            {
                "chunk_id": "chunk-c",
                "doc_id": "doc-b",
                "title": "Unrelated",
                "content": "Supplier training notes.",
                "score": 0.1,
            },
        ]
    }

    graded = tools["grade_evidence_candidates"].invoke(
        {"query": "What is the approved current CDR date?", "candidates_json": json.dumps(candidates)}
    )
    assert graded["grades"][0]["grade"] == "strong"

    pruned = tools["prune_evidence_candidates"].invoke(
        {
            "query": "What is the approved current CDR date?",
            "candidates_json": json.dumps({"grades": graded["grades"]}),
            "keep": 2,
            "max_per_doc": 1,
        }
    )
    kept_ids = {item["candidate_id"] for item in pruned["kept"]}
    assert "chunk-a" in kept_ids
    assert "chunk-b" in kept_ids

    validation = tools["validate_evidence_plan"].invoke(
        {
            "query": "What is the approved current CDR date?",
            "selected_candidates_json": json.dumps({"kept": pruned["kept"]}),
            "expected_scope": "targeted",
        }
    )
    assert validation["status"] == "sufficient"
    assert validation["selected_doc_ids"] == ["doc-a"]

    hints = tools["build_rag_controller_hints"].invoke(
        {
            "query": "What is the approved current CDR date?",
            "selected_doc_ids_csv": "doc-a",
            "selected_chunks_json": json.dumps({"kept": pruned["kept"]}),
            "coverage_goal": "cross_document",
            "result_mode": "comparison",
        }
    )
    parsed = json.loads(hints["controller_hints_json"])
    assert parsed["preferred_doc_ids"] == ["doc-a"]
    assert parsed["force_deep_search"] is True
    assert parsed["authority_version_check"] is True
