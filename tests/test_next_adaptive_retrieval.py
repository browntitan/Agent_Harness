from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from agentic_chatbot_next.persistence.postgres.chunks import ChunkRecord, ScoredChunk
from agentic_chatbot_next.rag.adaptive import (
    CorpusRetrievalAdapter,
    SearchFilters,
    _build_round_queries,
    run_retrieval_controller,
)
from agentic_chatbot_next.rag.fanout import RagSearchBatchResult, RagSearchTaskResult, serialize_document, serialize_graded_chunk
from agentic_chatbot_next.rag.graph_store import GraphSearchHit
from agentic_chatbot_next.rag.retrieval import GradedChunk


def _scored_chunk(
    *,
    doc_id: str,
    chunk_id: str,
    title: str,
    text: str,
    score: float,
    method: str = "vector",
    chunk_type: str = "general",
    section_title: str = "",
) -> ScoredChunk:
    return ScoredChunk(
        doc=Document(
            page_content=text,
            metadata={
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "title": title,
                "chunk_index": 0,
                "chunk_type": chunk_type,
                "section_title": section_title,
                "source_type": "kb",
            },
        ),
        score=score,
        method=method,
    )


def _chunk_record(
    *,
    doc_id: str,
    chunk_id: str,
    index: int,
    content: str,
    chunk_type: str = "general",
    section_title: str = "",
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_index=index,
        content=content,
        tenant_id="tenant-123",
        collection_id="default",
        chunk_type=chunk_type,
        section_title=section_title or None,
    )


class _JudgeModel:
    def invoke(self, prompt, config=None):  # noqa: D401 - test double
        del config
        text = str(prompt)
        if "rewritten_query" in text:
            return SimpleNamespace(content='{"rewritten_query":"process flow workflow approval handoff"}')
        chunk_ids = re.findall(r'"chunk_id":\s*"([^"]+)"', text)
        grades = []
        for chunk_id in chunk_ids:
            relevance = 3 if chunk_id in {"doc-a#chunk0001", "doc-b#chunk0001"} else 1
            grades.append({"chunk_id": chunk_id, "relevance": relevance, "reason": "test"})
        return SimpleNamespace(content=json.dumps({"grades": grades}))


class _FacetJudgeModel:
    def invoke(self, prompt, config=None):  # noqa: D401 - test double
        del config
        text = str(prompt)
        if "rewritten_query" in text:
            return SimpleNamespace(content='{"rewritten_query":""}')
        chunk_ids = re.findall(r'"chunk_id":\s*"([^"]+)"', text)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "grades": [
                        {"chunk_id": chunk_id, "relevance": 3, "reason": "facet evidence"}
                        for chunk_id in chunk_ids
                    ]
                }
            )
        )


def test_retrieval_controller_selects_generic_facet_coverage() -> None:
    chunks = [
        _scored_chunk(
            doc_id="doc-alpha",
            chunk_id="doc-alpha#chunk0001",
            title="alpha.md",
            text="Alpha evidence is covered in this document.",
            score=0.9,
        ),
        _scored_chunk(
            doc_id="doc-beta",
            chunk_id="doc-beta#chunk0001",
            title="beta.md",
            text="Beta evidence is covered in this document.",
            score=0.88,
        ),
        _scored_chunk(
            doc_id="doc-gamma",
            chunk_id="doc-gamma#chunk0001",
            title="gamma.md",
            text="Gamma evidence is covered in this document.",
            score=0.86,
        ),
    ]
    records = {
        chunk.doc.metadata["chunk_id"]: _chunk_record(
            doc_id=chunk.doc.metadata["doc_id"],
            chunk_id=chunk.doc.metadata["chunk_id"],
            index=0,
            content=chunk.doc.page_content,
        )
        for chunk in chunks
    }

    def search(query, **kwargs):
        del kwargs
        lowered = str(query).lower()
        hits = [chunk for chunk in chunks if chunk.doc.metadata["title"].split(".", 1)[0] in lowered]
        return hits or list(chunks)

    stores = SimpleNamespace(
        chunk_store=SimpleNamespace(
            vector_search=search,
            keyword_search=search,
            chunk_count=lambda doc_id=None, tenant_id="tenant-123": 1,
            get_chunk_by_id=lambda chunk_id, tenant_id: records.get(chunk_id),
            get_chunks_by_index_range=lambda doc_id, min_idx, max_idx, tenant_id: [
                record
                for record in records.values()
                if record.doc_id == doc_id and min_idx <= record.chunk_index <= max_idx
            ],
        ),
        doc_store=SimpleNamespace(
            fuzzy_search_title=lambda hint, tenant_id, limit=5, collection_id="": [],
            get_document=lambda doc_id, tenant_id: SimpleNamespace(
                doc_id=doc_id,
                title=f"{doc_id}.md",
                source_type="kb",
                source_path=f"/tmp/{doc_id}.md",
            ),
        ),
    )
    settings = SimpleNamespace(
        default_collection_id="default",
        default_tenant_id="tenant-123",
        rag_min_evidence_chunks=1,
        rag_top_k_vector=3,
        rag_top_k_keyword=3,
        prompts_backend="local",
        judge_rewrite_prompt_path=Path("missing"),
        judge_grading_prompt_path=Path("missing"),
    )

    result = run_retrieval_controller(
        settings,
        stores,
        providers=SimpleNamespace(judge=_FacetJudgeModel(), chat=object()),
        session=SimpleNamespace(tenant_id="tenant-123", metadata={"kb_collection_id": "default"}),
        query="Draft a grounded answer about alpha, beta, and gamma. Cite sources.",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=3,
        top_k_keyword=3,
        max_retries=1,
        callbacks=[],
        search_mode="fast",
        max_search_rounds=1,
    )

    selected_chunk_ids = {doc.metadata["chunk_id"] for doc in result.selected_docs}
    assert selected_chunk_ids == {
        "doc-alpha#chunk0001",
        "doc-beta#chunk0001",
        "doc-gamma#chunk0001",
    }
    assert result.candidate_counts["covered_facets"] == 3
    assert result.evidence_ledger["facet_coverage"]


def test_fast_path_runs_one_hybrid_retrieval_pass() -> None:
    calls: list[str] = []
    chunk = _scored_chunk(
        doc_id="doc-alpha",
        chunk_id="doc-alpha#chunk0001",
        title="alpha.md",
        text="Alpha policy states the release tier is standard for this customer.",
        score=0.91,
    )

    def vector_search(query, *, top_k, tenant_id, doc_id_filter=None, collection_id_filter=None):
        del query, top_k, tenant_id, doc_id_filter, collection_id_filter
        calls.append("vector")
        return [chunk]

    def keyword_search(query, *, top_k, tenant_id, doc_id_filter=None, collection_id_filter=None):
        del query, top_k, tenant_id, doc_id_filter, collection_id_filter
        calls.append("keyword")
        return [ScoredChunk(doc=chunk.doc, score=0.87, method="keyword")]

    stores = SimpleNamespace(
        chunk_store=SimpleNamespace(
            vector_search=vector_search,
            keyword_search=keyword_search,
            get_chunks_by_index_range=lambda doc_id, min_idx, max_idx, tenant_id: [],
        ),
        doc_store=SimpleNamespace(
            fuzzy_search_title=lambda hint, tenant_id, limit=5, collection_id="": [],
            list_documents=lambda tenant_id="tenant-123", collection_id="default", source_type="": [],
            get_document=lambda doc_id, tenant_id: SimpleNamespace(
                doc_id=doc_id,
                title="alpha.md",
                source_type="kb",
                source_path="/tmp/alpha.md",
                file_type="md",
                doc_structure_type="general",
            ),
        ),
    )
    settings = SimpleNamespace(
        default_collection_id="default",
        default_tenant_id="tenant-123",
        rag_min_evidence_chunks=1,
        rag_top_k_vector=4,
        rag_top_k_keyword=4,
        prompts_backend="local",
        judge_rewrite_prompt_path=Path("missing"),
        judge_grading_prompt_path=Path("missing"),
        graph_search_enabled=False,
        rag_budget_ms=0,
    )

    result = run_retrieval_controller(
        settings,
        stores,
        providers=SimpleNamespace(judge=_FacetJudgeModel(), chat=object()),
        session=SimpleNamespace(tenant_id="tenant-123", metadata={"kb_collection_id": "default"}),
        query="What release tier applies to Alpha?",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=1,
        callbacks=[],
        search_mode="fast",
        max_search_rounds=1,
    )

    assert calls.count("vector") == 1
    assert calls.count("keyword") == 1
    assert result.search_mode == "fast"


def test_corpus_retrieval_adapter_full_read_samples_long_document():
    chunks = [
        _chunk_record(
            doc_id="doc-long",
            chunk_id=f"doc-long#chunk{index:04d}",
            index=index,
            content=f"Chunk {index} content",
            section_title=f"Section {index}",
        )
        for index in range(12)
    ]

    stores = SimpleNamespace(
        chunk_store=SimpleNamespace(
            chunk_count=lambda doc_id=None, tenant_id="tenant-123": len(chunks),
            get_chunks_by_index_range=lambda doc_id, min_idx, max_idx, tenant_id: [
                item
                for item in chunks
                if item.doc_id == doc_id and min_idx <= item.chunk_index <= max_idx
            ],
            vector_search=lambda *args, **kwargs: [],
            keyword_search=lambda *args, **kwargs: [],
        ),
        doc_store=SimpleNamespace(
            get_document=lambda doc_id, tenant_id: SimpleNamespace(
                doc_id=doc_id,
                title="ARCHITECTURE.md",
                source_type="kb",
                source_path="/tmp/ARCHITECTURE.md",
                file_type="md",
                doc_structure_type="general",
            )
        ),
    )

    adapter = CorpusRetrievalAdapter(
        stores,
        settings=SimpleNamespace(default_tenant_id="tenant-123"),
        session=SimpleNamespace(tenant_id="tenant-123"),
    )

    docs = adapter.read_document(
        "doc-long",
        focus="architecture overview",
        max_chunks=4,
        read_depth="full",
        full_read_chunk_threshold=6,
    )

    assert len(docs) == 6
    chunk_indexes = {int(doc.metadata["chunk_index"]) for doc in docs}
    assert 0 in chunk_indexes
    assert max(chunk_indexes) >= 10


def test_corpus_retrieval_adapter_uses_selected_kb_collection_not_upload_scope():
    calls = []

    def _record_call(method):
        def _search(query, *, top_k, tenant_id, doc_id_filter=None, collection_id_filter=None):
            del query, top_k, tenant_id, doc_id_filter
            calls.append((method, collection_id_filter))
            return []

        return _search

    stores = SimpleNamespace(
        chunk_store=SimpleNamespace(
            vector_search=_record_call("vector"),
            keyword_search=_record_call("keyword"),
        ),
        doc_store=SimpleNamespace(
            fuzzy_search_title=lambda hint, tenant_id, limit=5, collection_id="": [],
            list_documents=lambda tenant_id="tenant-123", collection_id="", source_type="": [],
        ),
    )
    adapter = CorpusRetrievalAdapter(
        stores,
        settings=SimpleNamespace(default_collection_id="default", rag_top_k_vector=2, rag_top_k_keyword=2),
        session=SimpleNamespace(
            tenant_id="tenant-123",
            metadata={
                "collection_id": "owui-chat-1",
                "upload_collection_id": "owui-chat-1",
                "kb_collection_id": "default",
                "available_kb_collection_ids": ["default", "policy"],
                "selected_kb_collection_id": "policy",
            },
        ),
    )

    adapter.search_corpus("rate limit policy", strategy="hybrid", top_k_vector=2, top_k_keyword=2)

    assert calls == [("vector", "policy"), ("keyword", "policy")]


def test_corpus_retrieval_adapter_expands_structured_neighbors() -> None:
    seed = ScoredChunk(
        doc=Document(
            page_content="Workbook row 3 has the matched beta schedule value.",
            metadata={
                "doc_id": "doc-sheet",
                "chunk_id": "doc-sheet#chunk0003",
                "chunk_index": 3,
                "title": "schedule.xlsx",
                "file_type": "xlsx",
                "sheet_name": "Schedule",
                "source_type": "kb",
            },
        ),
        score=0.92,
        method="vector",
    )
    records = [
        _chunk_record(
            doc_id="doc-sheet",
            chunk_id=f"doc-sheet#chunk{index:04d}",
            index=index,
            content=f"Workbook row {index} adjacent schedule context.",
        )
        for index in range(1, 6)
    ]

    stores = SimpleNamespace(
        chunk_store=SimpleNamespace(
            vector_search=lambda *args, **kwargs: [seed],
            keyword_search=lambda *args, **kwargs: [],
            get_chunks_by_index_range=lambda doc_id, min_idx, max_idx, tenant_id: [
                record
                for record in records
                if record.doc_id == doc_id and min_idx <= record.chunk_index <= max_idx
            ],
        ),
        doc_store=SimpleNamespace(
            fuzzy_search_title=lambda hint, tenant_id, limit=5, collection_id="": [],
            get_document=lambda doc_id, tenant_id: SimpleNamespace(
                doc_id=doc_id,
                title="schedule.xlsx",
                source_type="kb",
                source_path="/tmp/schedule.xlsx",
                file_type="xlsx",
                doc_structure_type="spreadsheet",
            ),
        ),
    )
    adapter = CorpusRetrievalAdapter(
        stores,
        settings=SimpleNamespace(default_collection_id="default", rag_top_k_vector=2, rag_top_k_keyword=2),
        session=SimpleNamespace(tenant_id="tenant-123", metadata={"kb_collection_id": "default"}),
    )

    hits = adapter.search_corpus("beta schedule", strategy="hybrid", limit=8)
    hit_ids = {hit.doc.metadata["chunk_id"] for hit in hits}

    assert "doc-sheet#chunk0003" in hit_ids
    assert {"doc-sheet#chunk0001", "doc-sheet#chunk0002", "doc-sheet#chunk0004", "doc-sheet#chunk0005"} <= hit_ids


def test_run_retrieval_controller_escalates_to_deep_for_process_flow_discovery():
    process_a = _scored_chunk(
        doc_id="doc-a",
        chunk_id="doc-a#chunk0001",
        title="incident_workflow.md",
        text="Process flow for incident escalation and approval handoff.",
        score=0.92,
        chunk_type="process_flow",
        section_title="Incident workflow",
    )
    process_b = _scored_chunk(
        doc_id="doc-b",
        chunk_id="doc-b#chunk0001",
        title="change_workflow.md",
        text="Workflow describing approval flow, process steps, and handoff sequence.",
        score=0.89,
        chunk_type="process_flow",
        section_title="Change workflow",
    )
    unrelated = _scored_chunk(
        doc_id="doc-c",
        chunk_id="doc-c#chunk0001",
        title="pricing.md",
        text="Pricing plan overview and commercial terms.",
        score=0.61,
    )

    chunk_windows = {
        "doc-a#chunk0001": [
            _chunk_record(
                doc_id="doc-a",
                chunk_id="doc-a#chunk0000",
                index=0,
                content="Overview of the incident management workflow.",
                chunk_type="header",
                section_title="Workflow overview",
            ),
            _chunk_record(
                doc_id="doc-a",
                chunk_id="doc-a#chunk0001",
                index=1,
                content="Process flow for incident escalation and approval handoff.",
                chunk_type="process_flow",
                section_title="Incident workflow",
            ),
            _chunk_record(
                doc_id="doc-a",
                chunk_id="doc-a#chunk0002",
                index=2,
                content="Step 3 routes tickets to the approver.",
                chunk_type="process_flow",
                section_title="Approval routing",
            ),
        ],
        "doc-b#chunk0001": [
            _chunk_record(
                doc_id="doc-b",
                chunk_id="doc-b#chunk0000",
                index=0,
                content="Overview of the change request workflow.",
                chunk_type="header",
                section_title="Workflow overview",
            ),
            _chunk_record(
                doc_id="doc-b",
                chunk_id="doc-b#chunk0001",
                index=1,
                content="Workflow describing approval flow, process steps, and handoff sequence.",
                chunk_type="process_flow",
                section_title="Change workflow",
            ),
            _chunk_record(
                doc_id="doc-b",
                chunk_id="doc-b#chunk0002",
                index=2,
                content="Step 4 transitions work to deployment.",
                chunk_type="process_flow",
                section_title="Deployment handoff",
            ),
        ],
    }

    def vector_search(query, *, top_k, tenant_id, doc_id_filter=None, collection_id_filter=None):
        del top_k, tenant_id
        if doc_id_filter == "doc-a":
            return [process_a]
        if doc_id_filter == "doc-b":
            return [process_b]
        if doc_id_filter == "doc-c":
            return [unrelated]
        if collection_id_filter == "default":
            return [process_a, unrelated]
        return []

    def keyword_search(query, *, top_k, tenant_id, doc_id_filter=None, collection_id_filter=None):
        del top_k, tenant_id
        lower = query.lower()
        if doc_id_filter == "doc-a" and ("workflow" in lower or "process" in lower):
            return [ScoredChunk(doc=process_a.doc, score=0.95, method="keyword")]
        if doc_id_filter == "doc-b" and ("workflow" in lower or "process" in lower):
            return [ScoredChunk(doc=process_b.doc, score=0.93, method="keyword")]
        if collection_id_filter == "default":
            return [ScoredChunk(doc=process_a.doc, score=0.94, method="keyword")]
        return []

    def get_chunk_by_id(chunk_id, tenant_id):
        del tenant_id
        for rows in chunk_windows.values():
            for item in rows:
                if item.chunk_id == chunk_id:
                    return item
        return None

    def get_chunks_by_index_range(doc_id, min_idx, max_idx, tenant_id):
        del tenant_id
        rows = []
        for item in chunk_windows.get("doc-a#chunk0001", []) + chunk_windows.get("doc-b#chunk0001", []):
            if item.doc_id == doc_id and min_idx <= item.chunk_index <= max_idx:
                rows.append(item)
        return rows

    stores = SimpleNamespace(
        chunk_store=SimpleNamespace(
            vector_search=vector_search,
            keyword_search=keyword_search,
            get_chunk_by_id=get_chunk_by_id,
            get_chunks_by_index_range=get_chunks_by_index_range,
            chunk_count=lambda doc_id=None, tenant_id="tenant-123": 3,
        ),
        doc_store=SimpleNamespace(
            list_documents=lambda tenant_id="tenant-123", collection_id="default", source_type="": [
                SimpleNamespace(
                    doc_id="doc-a",
                    title="incident_workflow.md",
                    source_type="kb",
                    file_type="md",
                    doc_structure_type="process_flow_doc",
                    num_chunks=3,
                ),
                SimpleNamespace(
                    doc_id="doc-b",
                    title="change_workflow.md",
                    source_type="kb",
                    file_type="md",
                    doc_structure_type="process_flow_doc",
                    num_chunks=3,
                ),
                SimpleNamespace(
                    doc_id="doc-c",
                    title="pricing.md",
                    source_type="kb",
                    file_type="md",
                    doc_structure_type="general",
                    num_chunks=1,
                ),
            ],
            get_document=lambda doc_id, tenant_id: SimpleNamespace(
                doc_id=doc_id,
                title="incident_workflow.md" if doc_id == "doc-a" else "change_workflow.md" if doc_id == "doc-b" else "pricing.md",
                source_type="kb",
                source_path=f"/tmp/{doc_id}.md",
            ),
            fuzzy_search_title=lambda hint, tenant_id, limit=5, collection_id="": [],
        ),
    )
    settings = SimpleNamespace(
        default_collection_id="default",
        default_tenant_id="tenant-123",
        rag_min_evidence_chunks=2,
        rag_top_k_vector=4,
        rag_top_k_keyword=4,
        prompts_backend="local",
        judge_rewrite_prompt_path=Path("missing"),
        judge_grading_prompt_path=Path("missing"),
    )

    result = run_retrieval_controller(
        settings,
        stores,
        providers=SimpleNamespace(judge=_JudgeModel(), chat=object()),
        session=SimpleNamespace(tenant_id="tenant-123"),
        query="Identify all documents that have process flows outlined in them.",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=2,
        callbacks=[],
        search_mode="auto",
        max_search_rounds=3,
    )

    selected_titles = {_title for _title in (doc.metadata.get("title") for doc in result.selected_docs)}

    assert result.search_mode == "deep"
    assert selected_titles >= {"incident_workflow.md", "change_workflow.md"}
    assert result.candidate_counts["unique_docs"] >= 2
    assert any("fetch_chunk_window" in item for item in result.tool_call_log)
    assert "keyword" in result.strategies_used


def test_run_retrieval_controller_respects_corpus_discovery_profile_for_generic_query():
    doc = _scored_chunk(
        doc_id="doc-a",
        chunk_id="doc-a#chunk0001",
        title="onboarding_steps.md",
        text="Step 1 opens the onboarding workflow and routes approval to HR.",
        score=0.91,
        chunk_type="process_flow",
        section_title="Onboarding steps",
    )

    def vector_search(query, *, top_k, tenant_id, doc_id_filter=None, collection_id_filter=None):
        del query, top_k, tenant_id, doc_id_filter, collection_id_filter
        return [doc]

    def keyword_search(query, *, top_k, tenant_id, doc_id_filter=None, collection_id_filter=None):
        del query, top_k, tenant_id, doc_id_filter, collection_id_filter
        return [ScoredChunk(doc=doc.doc, score=0.88, method="keyword")]

    stores = SimpleNamespace(
        chunk_store=SimpleNamespace(
            vector_search=vector_search,
            keyword_search=keyword_search,
            get_chunk_by_id=lambda chunk_id, tenant_id: _chunk_record(
                doc_id="doc-a",
                chunk_id=chunk_id,
                index=1,
                content="Step 1 opens the onboarding workflow and routes approval to HR.",
                chunk_type="process_flow",
                section_title="Onboarding steps",
            ),
            get_chunks_by_index_range=lambda doc_id, min_idx, max_idx, tenant_id: [
                _chunk_record(
                    doc_id=doc_id,
                    chunk_id="doc-a#chunk0001",
                    index=1,
                    content="Step 1 opens the onboarding workflow and routes approval to HR.",
                    chunk_type="process_flow",
                    section_title="Onboarding steps",
                )
            ],
            chunk_count=lambda doc_id=None, tenant_id="tenant-123": 1,
        ),
        doc_store=SimpleNamespace(
            list_documents=lambda tenant_id="tenant-123", collection_id="default", source_type="": [
                SimpleNamespace(
                    doc_id="doc-a",
                    title="onboarding_steps.md",
                    source_type="kb",
                    file_type="md",
                    doc_structure_type="process_flow_doc",
                    num_chunks=1,
                )
            ],
            get_document=lambda doc_id, tenant_id: SimpleNamespace(
                doc_id=doc_id,
                title="onboarding_steps.md",
                source_type="kb",
                source_path="/tmp/onboarding_steps.md",
            ),
            fuzzy_search_title=lambda hint, tenant_id, limit=5, collection_id="": [],
        ),
    )
    settings = SimpleNamespace(
        default_collection_id="default",
        default_tenant_id="tenant-123",
        rag_min_evidence_chunks=1,
        rag_top_k_vector=2,
        rag_top_k_keyword=2,
        prompts_backend="local",
        judge_rewrite_prompt_path=Path("missing"),
        judge_grading_prompt_path=Path("missing"),
    )

    result = run_retrieval_controller(
        settings,
        stores,
        providers=SimpleNamespace(judge=_JudgeModel(), chat=object()),
        session=SimpleNamespace(tenant_id="tenant-123"),
        query="onboarding approvals",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=2,
        top_k_keyword=2,
        max_retries=1,
        callbacks=[],
        search_mode="auto",
        max_search_rounds=2,
        research_profile="corpus_discovery",
        coverage_goal="corpus_wide",
        result_mode="inventory",
        controller_hints={"prefer_process_flow_docs": True},
    )

    assert result.search_mode == "deep"
    assert result.candidate_counts["selected_docs"] >= 1


def test_build_round_queries_preserves_topic_anchor_for_workflow_discovery() -> None:
    queries = _build_round_queries(
        "Which documents in the knowledge base contain onboarding workflows?",
        settings=SimpleNamespace(prompts_backend="local", judge_rewrite_prompt_path=Path("missing")),
        providers=SimpleNamespace(judge=_JudgeModel()),
        conversation_context="",
        callbacks=[],
        round_index=1,
        discovery=True,
        prefer_process_flow=True,
        controller_hints={"prefer_inventory_output": True},
        seen_queries=set(),
    )

    fallback_queries = {rationale: query_text for query_text, _strategy, rationale in queries}

    assert "process_flow_fallback" in fallback_queries
    assert "inventory_fallback" in fallback_queries
    assert any(term in fallback_queries["process_flow_fallback"] for term in ("onboarding", "new hire"))
    assert any(term in fallback_queries["inventory_fallback"] for term in ("onboarding", "new hire"))
    assert "process flow workflow approval flow handoff escalation path" not in fallback_queries.values()


def test_build_round_queries_generates_visible_info_rewrites_for_weak_causal_claims() -> None:
    question = (
        "If someone says Aurora Portal slipped because the payment API was faulty, "
        "what is the better evidence-based answer?"
    )
    queries = _build_round_queries(
        question,
        settings=SimpleNamespace(prompts_backend="local", judge_rewrite_prompt_path=Path("missing")),
        providers=SimpleNamespace(judge=_JudgeModel()),
        conversation_context="Candidate sources: portal_release_readiness.md (doc_id: DOC-1)",
        callbacks=[],
        round_index=1,
        discovery=False,
        prefer_process_flow=False,
        controller_hints={},
        seen_queries=set(),
    )

    assert queries[0] == (question, "hybrid", "original")
    rationales = {rationale for _query_text, _strategy, rationale in queries}
    assert "claim_focused_rewrite" in rationales
    assert "refutation_focused_rewrite" in rationales
    assert "causal_factor_rewrite" in rationales
    assert "status_outcome_rewrite" in rationales
    joined_queries = "\n".join(query_text for query_text, _strategy, _rationale in queries).lower()
    assert "support refute" in joined_queries
    assert "cause reason driver factor evidence" in joined_queries
    assert "certificate" not in joined_queries
    assert "serialized" not in joined_queries


def test_corpus_adapter_excludes_seen_chunks_and_returns_window_context():
    main_chunk = _scored_chunk(
        doc_id="doc-a",
        chunk_id="doc-a#chunk0001",
        title="playbook.md",
        text="Workflow and handoff sequence.",
        score=0.91,
        chunk_type="process_flow",
    )
    neighbours = [
        _chunk_record(doc_id="doc-a", chunk_id="doc-a#chunk0000", index=0, content="Workflow intro", chunk_type="header"),
        _chunk_record(doc_id="doc-a", chunk_id="doc-a#chunk0001", index=1, content="Workflow and handoff sequence.", chunk_type="process_flow"),
        _chunk_record(doc_id="doc-a", chunk_id="doc-a#chunk0002", index=2, content="Approver handoff details.", chunk_type="process_flow"),
    ]

    stores = SimpleNamespace(
        chunk_store=SimpleNamespace(
            vector_search=lambda query, *, top_k, tenant_id, doc_id_filter=None, collection_id_filter=None: [main_chunk],
            keyword_search=lambda query, *, top_k, tenant_id, doc_id_filter=None, collection_id_filter=None: [ScoredChunk(doc=main_chunk.doc, score=0.93, method="keyword")],
            get_chunk_by_id=lambda chunk_id, tenant_id: neighbours[1],
            get_chunks_by_index_range=lambda doc_id, min_idx, max_idx, tenant_id: neighbours,
            chunk_count=lambda doc_id=None, tenant_id="tenant-123": 3,
        ),
        doc_store=SimpleNamespace(
            list_documents=lambda tenant_id="tenant-123", collection_id="default", source_type="": [
                SimpleNamespace(doc_id="doc-a", title="playbook.md", source_type="kb", file_type="md", doc_structure_type="process_flow_doc")
            ],
            get_document=lambda doc_id, tenant_id: SimpleNamespace(doc_id="doc-a", title="playbook.md", source_type="kb", source_path="/tmp/playbook.md"),
        ),
    )
    adapter = CorpusRetrievalAdapter(
        stores,
        settings=SimpleNamespace(default_collection_id="default", rag_top_k_vector=4, rag_top_k_keyword=4, default_tenant_id="tenant-123"),
        session=SimpleNamespace(tenant_id="tenant-123"),
    )

    initial = adapter.search_corpus("workflow handoff", filters=SearchFilters(collection_id="default"), strategy="hybrid")
    excluded = adapter.search_corpus(
        "workflow handoff",
        filters=SearchFilters(collection_id="default"),
        strategy="hybrid",
        exclude_chunk_ids={"doc-a#chunk0001"},
    )
    window = adapter.fetch_chunk_window("doc-a#chunk0001", before=1, after=1)

    assert len(initial) == 1
    assert excluded == []
    assert [doc.metadata["chunk_id"] for doc in window] == [
        "doc-a#chunk0000",
        "doc-a#chunk0001",
        "doc-a#chunk0002",
    ]


def test_run_retrieval_controller_uses_runtime_bridge_for_parallel_discovery():
    process_a = _scored_chunk(
        doc_id="doc-a",
        chunk_id="doc-a#chunk0001",
        title="incident_workflow.md",
        text="Incident process flow and escalation path.",
        score=0.92,
        chunk_type="process_flow",
        section_title="Incident workflow",
    )
    process_b = _scored_chunk(
        doc_id="doc-b",
        chunk_id="doc-b#chunk0001",
        title="change_workflow.md",
        text="Change approval workflow and deployment handoff.",
        score=0.89,
        chunk_type="process_flow",
        section_title="Change workflow",
    )

    def vector_search(query, *, top_k, tenant_id, doc_id_filter=None, collection_id_filter=None):
        del query, top_k, tenant_id, collection_id_filter
        if doc_id_filter == "doc-a":
            return [process_a]
        if doc_id_filter == "doc-b":
            return [process_b]
        return [process_a, process_b]

    def keyword_search(query, *, top_k, tenant_id, doc_id_filter=None, collection_id_filter=None):
        del query, top_k, tenant_id, collection_id_filter
        if doc_id_filter == "doc-a":
            return [ScoredChunk(doc=process_a.doc, score=0.95, method="keyword")]
        if doc_id_filter == "doc-b":
            return [ScoredChunk(doc=process_b.doc, score=0.94, method="keyword")]
        return [ScoredChunk(doc=process_a.doc, score=0.95, method="keyword")]

    stores = SimpleNamespace(
        chunk_store=SimpleNamespace(
            vector_search=vector_search,
            keyword_search=keyword_search,
            get_chunk_by_id=lambda chunk_id, tenant_id: None,
            get_chunks_by_index_range=lambda doc_id, min_idx, max_idx, tenant_id: [],
            chunk_count=lambda doc_id=None, tenant_id="tenant-123": 0,
        ),
        doc_store=SimpleNamespace(
            list_documents=lambda tenant_id="tenant-123", collection_id="default", source_type="": [
                SimpleNamespace(doc_id="doc-a", title="incident_workflow.md", source_type="kb", file_type="md", doc_structure_type="process_flow_doc"),
                SimpleNamespace(doc_id="doc-b", title="change_workflow.md", source_type="kb", file_type="md", doc_structure_type="process_flow_doc"),
            ],
            get_document=lambda doc_id, tenant_id: SimpleNamespace(
                doc_id=doc_id,
                title="incident_workflow.md" if doc_id == "doc-a" else "change_workflow.md",
                source_type="kb",
                source_path=f"/tmp/{doc_id}.md",
            ),
            fuzzy_search_title=lambda hint, tenant_id, limit=5, collection_id="": [],
        ),
    )
    settings = SimpleNamespace(
        default_collection_id="default",
        default_tenant_id="tenant-123",
        rag_min_evidence_chunks=2,
        rag_top_k_vector=4,
        rag_top_k_keyword=4,
        prompts_backend="local",
        judge_rewrite_prompt_path=Path("missing"),
        judge_grading_prompt_path=Path("missing"),
    )

    worker_doc = Document(
        page_content="Worker found a second workflow with approval steps.",
        metadata={
            "doc_id": "doc-b",
            "chunk_id": "doc-b#chunk0002",
            "title": "change_workflow.md",
            "source_type": "kb",
        },
    )

    class _FakeBridge:
        def __init__(self) -> None:
            self.tasks = []

        def can_run_parallel(self, *, task_count: int) -> bool:
            return task_count >= 2

        def run_search_tasks(self, tasks):
            self.tasks = [task.to_dict() for task in tasks]
            results = [
                RagSearchTaskResult(
                    task_id=str(task.task_id),
                    evidence_entries=[
                        {
                            "chunk_id": "doc-b#chunk0002",
                            "doc_id": "doc-b",
                            "title": "change_workflow.md",
                            "query": task.query,
                            "strategy": "worker",
                            "rationale": "parallel_search",
                            "score": 0.88,
                            "relevance": 3,
                            "coverage_state": "strong",
                            "grade_reason": "worker evidence",
                        }
                    ],
                    candidate_docs=[serialize_document(worker_doc)],
                    graded_chunks=[serialize_graded_chunk(GradedChunk(doc=worker_doc, relevance=3, reason="worker evidence"))],
                    warnings=[],
                    doc_focus=[{"doc_id": "doc-b", "title": "change_workflow.md"}],
                )
                for task in tasks
            ]
            return RagSearchBatchResult(results=results, parallel_workers_used=True)

    bridge = _FakeBridge()
    progress_events = []

    class _Progress:
        def emit_progress(self, event_type: str, **payload):
            progress_events.append((event_type, payload))

    result = run_retrieval_controller(
        settings,
        stores,
        providers=SimpleNamespace(judge=_JudgeModel(), chat=object()),
        session=SimpleNamespace(tenant_id="tenant-123"),
        query="Identify all documents that have process flows outlined in them.",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=4,
        top_k_keyword=4,
        max_retries=2,
        callbacks=[],
        search_mode="deep",
        max_search_rounds=2,
        runtime_bridge=bridge,
        progress_emitter=_Progress(),
    )

    assert len(bridge.tasks) >= 2
    assert result.parallel_workers_used is True
    assert result.candidate_counts["worker_tasks"] >= 2
    assert any(":worker:" in item for item in result.tool_call_log)
    assert any(event_type == "task_plan" for event_type, _payload in progress_events)


def test_corpus_retrieval_adapter_uses_graph_hits_when_enabled():
    chunk = _chunk_record(
        doc_id="doc-graph",
        chunk_id="doc-graph#chunk0001",
        index=1,
        content="Vendor Acme requires Finance approval before renewing Clause 7 obligations.",
        chunk_type="requirement",
        section_title="Approval dependencies",
    )

    class _GraphStore:
        available = True

        def local_search(self, query, *, tenant_id, limit=8, doc_ids=None):
            del query, tenant_id, limit, doc_ids
            return [
                GraphSearchHit(
                    doc_id="doc-graph",
                    chunk_ids=["doc-graph#chunk0001"],
                    score=0.91,
                    title="vendor_clause.md",
                    relationship_path=["Vendor", "Finance", "Clause"],
                    summary="Vendor Acme depends on Finance approval.",
                )
            ]

        def global_search(self, query, *, tenant_id, limit=8, doc_ids=None):
            del query, tenant_id, limit, doc_ids
            return []

    stores = SimpleNamespace(
        graph_store=_GraphStore(),
        chunk_store=SimpleNamespace(
            vector_search=lambda *args, **kwargs: [],
            keyword_search=lambda *args, **kwargs: [],
            get_chunk_by_id=lambda chunk_id, tenant_id: chunk if chunk_id == "doc-graph#chunk0001" else None,
        ),
        doc_store=SimpleNamespace(
            get_document=lambda doc_id, tenant_id: SimpleNamespace(
                doc_id=doc_id,
                title="vendor_clause.md",
                source_type="kb",
                source_path="/tmp/vendor_clause.md",
            ),
            list_documents=lambda tenant_id="tenant-123", collection_id="default": [],
        ),
    )
    settings = SimpleNamespace(
        default_collection_id="default",
        default_tenant_id="tenant-123",
        rag_top_k_vector=4,
        rag_top_k_keyword=4,
        graph_search_enabled=True,
    )
    adapter = CorpusRetrievalAdapter(stores, settings=settings, session=SimpleNamespace(tenant_id="tenant-123"))

    hits = adapter.search_corpus(
        "Find all contracts with Vendor Acme that require Finance approval.",
        strategy="hybrid",
        limit=6,
    )

    assert hits
    assert any(item.method == "graph" for item in hits)
    assert any(item.doc.metadata.get("doc_id") == "doc-graph" for item in hits)


def test_corpus_retrieval_adapter_skips_graph_for_simple_fact_queries():
    graph_calls: list[str] = []
    chunk = _scored_chunk(
        doc_id="doc-fact",
        chunk_id="doc-fact#chunk0001",
        title="fact.md",
        text="The alpha service uses the standard retention setting.",
        score=0.9,
    )

    class _GraphStore:
        available = True

        def local_search(self, query, *, tenant_id, limit=8, doc_ids=None):
            del query, tenant_id, limit, doc_ids
            graph_calls.append("local")
            return []

        def global_search(self, query, *, tenant_id, limit=8, doc_ids=None):
            del query, tenant_id, limit, doc_ids
            graph_calls.append("global")
            return []

    stores = SimpleNamespace(
        graph_store=_GraphStore(),
        chunk_store=SimpleNamespace(
            vector_search=lambda *args, **kwargs: [chunk],
            keyword_search=lambda *args, **kwargs: [],
            get_chunks_by_index_range=lambda doc_id, min_idx, max_idx, tenant_id: [],
        ),
        doc_store=SimpleNamespace(
            fuzzy_search_title=lambda hint, tenant_id, limit=5, collection_id="": [],
            get_document=lambda doc_id, tenant_id: SimpleNamespace(
                doc_id=doc_id,
                title="fact.md",
                source_type="kb",
                source_path="/tmp/fact.md",
            ),
            list_documents=lambda tenant_id="tenant-123", collection_id="default": [],
        ),
    )
    adapter = CorpusRetrievalAdapter(
        stores,
        settings=SimpleNamespace(
            default_collection_id="default",
            default_tenant_id="tenant-123",
            rag_top_k_vector=4,
            rag_top_k_keyword=4,
            graph_search_enabled=True,
        ),
        session=SimpleNamespace(tenant_id="tenant-123", metadata={"active_graph_ids": ["graph-a"]}),
    )

    hits = adapter.search_corpus("What retention setting does the alpha service use?", strategy="hybrid", limit=6)

    assert hits
    assert graph_calls == []


def test_run_retrieval_controller_records_decomposition_claims_and_verification():
    doc = _scored_chunk(
        doc_id="doc-a",
        chunk_id="doc-a#chunk0001",
        title="vendor_clause.md",
        text="Vendor Acme depends on Finance approval before renewal under Clause 7.",
        score=0.96,
        chunk_type="requirement",
        section_title="Clause 7",
    )

    stores = SimpleNamespace(
        chunk_store=SimpleNamespace(
            vector_search=lambda query, *, top_k, tenant_id, doc_id_filter=None, collection_id_filter=None: [doc],
            keyword_search=lambda query, *, top_k, tenant_id, doc_id_filter=None, collection_id_filter=None: [ScoredChunk(doc=doc.doc, score=0.94, method="keyword")],
            get_chunk_by_id=lambda chunk_id, tenant_id: _chunk_record(
                doc_id="doc-a",
                chunk_id="doc-a#chunk0001",
                index=1,
                content="Vendor Acme depends on Finance approval before renewal under Clause 7.",
                chunk_type="requirement",
                section_title="Clause 7",
            ),
            get_chunks_by_index_range=lambda doc_id, min_idx, max_idx, tenant_id: [
                _chunk_record(
                    doc_id="doc-a",
                    chunk_id="doc-a#chunk0001",
                    index=1,
                    content="Vendor Acme depends on Finance approval before renewal under Clause 7.",
                    chunk_type="requirement",
                    section_title="Clause 7",
                )
            ],
            search_sections=lambda doc_id, *, tenant_id, section_query="", clause_numbers=None, sheet_names=None, limit=12: [
                _chunk_record(
                    doc_id="doc-a",
                    chunk_id="doc-a#chunk0001",
                    index=1,
                    content="Vendor Acme depends on Finance approval before renewal under Clause 7.",
                    chunk_type="requirement",
                    section_title="Clause 7",
                )
            ],
            chunk_count=lambda doc_id=None, tenant_id="tenant-123": 1,
        ),
        doc_store=SimpleNamespace(
            list_documents=lambda tenant_id="tenant-123", collection_id="default", source_type="": [
                SimpleNamespace(
                    doc_id="doc-a",
                    title="vendor_clause.md",
                    source_type="kb",
                    file_type="md",
                    doc_structure_type="contract",
                    num_chunks=1,
                    ingested_at="2026-04-01T00:00:00Z",
                )
            ],
            get_document=lambda doc_id, tenant_id: SimpleNamespace(
                doc_id=doc_id,
                title="vendor_clause.md",
                source_type="kb",
                source_path="/tmp/vendor_clause.md",
                ingested_at="2026-04-01T00:00:00Z",
            ),
            fuzzy_search_title=lambda hint, tenant_id, limit=5, collection_id="": [],
        ),
        graph_index_store=SimpleNamespace(get_index=lambda graph_id, tenant_id: None),
        entity_store=SimpleNamespace(
            resolve_aliases=lambda query, *, tenant_id, collection_id, limit=8: [
                {"entity_id": "ent_vendor", "canonical_name": "Vendor Acme", "matched_alias": "Vendor Acme", "score": 3.0},
                {"entity_id": "ent_finance", "canonical_name": "Finance Approval", "matched_alias": "Finance Approval", "score": 3.0},
            ]
        ),
    )
    settings = SimpleNamespace(
        default_collection_id="default",
        default_tenant_id="tenant-123",
        rag_min_evidence_chunks=1,
        rag_top_k_vector=2,
        rag_top_k_keyword=2,
        prompts_backend="local",
        judge_rewrite_prompt_path=Path("missing"),
        judge_grading_prompt_path=Path("missing"),
        retrieval_quality_verifier_enabled=True,
        section_first_retrieval_enabled=True,
        entity_linking_enabled=True,
    )

    result = run_retrieval_controller(
        settings,
        stores,
        providers=SimpleNamespace(judge=_JudgeModel(), chat=object()),
        session=SimpleNamespace(tenant_id="tenant-123"),
        query="Which vendor depends on Finance approval in Clause 7?",
        conversation_context="",
        preferred_doc_ids=["doc-a"],
        must_include_uploads=False,
        top_k_vector=2,
        top_k_keyword=2,
        max_retries=2,
        callbacks=[],
        search_mode="deep",
        max_search_rounds=3,
        controller_hints={"resolved_doc_ids": ["doc-a"]},
    )

    assert result.decomposition["canonical_entities"]
    assert result.claim_ledger["claims"]
    assert result.claim_ledger["supported_claim_ids"]
    assert "round1:search_sections:Which vendor depends on Finance approval in Clause 7?" in result.tool_call_log
    assert result.retrieval_verification["status"] in {"pass", "revise"}
