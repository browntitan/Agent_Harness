from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from agentic_chatbot_next.persistence.postgres.chunks import ScoredChunk
from agentic_chatbot_next.rag import retrieval as retrieval_module
from agentic_chatbot_next.rag.query_normalization import normalize_retrieval_question
from agentic_chatbot_next.rag.retrieval import grade_chunks, merge_dedupe, rank_fuse_dedupe, retrieve_candidates


def _scored_chunk(*, doc_id: str, chunk_id: str, title: str, score: float, method: str = "vector") -> ScoredChunk:
    return ScoredChunk(
        doc=Document(
            page_content=f"content from {title}",
            metadata={
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "title": title,
                "source_type": "kb",
            },
        ),
        score=score,
        method=method,
    )


def test_retrieve_candidates_expands_title_matches_for_architecture_queries():
    calls: list[tuple[str, str | None, str | None]] = []

    prompt_chunk = _scored_chunk(
        doc_id="doc-prompts",
        chunk_id="doc-prompts#chunk0001",
        title="TEST_QUERIES.md",
        score=0.91,
    )
    architecture_chunk = _scored_chunk(
        doc_id="doc-arch",
        chunk_id="doc-arch#chunk0001",
        title="ARCHITECTURE.md",
        score=0.55,
    )

    def vector_search(
        query: str,
        *,
        top_k: int,
        tenant_id: str,
        doc_id_filter: str | None = None,
        collection_id_filter: str | None = None,
    ):
        del query, top_k, tenant_id
        calls.append(("vector", doc_id_filter, collection_id_filter))
        if doc_id_filter == "doc-arch":
            return [architecture_chunk]
        return [prompt_chunk]

    def keyword_search(
        query: str,
        *,
        top_k: int,
        tenant_id: str,
        doc_id_filter: str | None = None,
        collection_id_filter: str | None = None,
    ):
        del query, top_k, tenant_id
        calls.append(("keyword", doc_id_filter, collection_id_filter))
        return []

    stores = SimpleNamespace(
        chunk_store=SimpleNamespace(
            vector_search=vector_search,
            keyword_search=keyword_search,
        ),
        doc_store=SimpleNamespace(
            fuzzy_search_title=lambda hint, tenant_id, limit=5, collection_id="": [
                {"doc_id": "doc-arch", "title": "ARCHITECTURE.md", "score": 0.93}
            ]
        ),
    )

    result = retrieve_candidates(
        stores,
        "key implementation details architecture documentation",
        tenant_id="tenant-123",
        preferred_doc_ids=["doc-prompts", "doc-arch"],
        must_include_uploads=False,
        top_k_vector=5,
        top_k_keyword=5,
        collection_id_filter="default",
    )

    merged_doc_ids = {(chunk.doc.metadata or {}).get("doc_id") for chunk in result["merged"]}

    assert ("vector", None, "default") in calls
    assert ("keyword", None, "default") in calls
    assert ("vector", "doc-arch", None) in calls
    assert ("keyword", "doc-arch", None) in calls
    assert "doc-arch" in merged_doc_ids
    boosted_architecture = next(
        chunk for chunk in result["vector"] if (chunk.doc.metadata or {}).get("doc_id") == "doc-arch"
    )
    assert boosted_architecture.score > architecture_chunk.score


def test_retrieve_candidates_searches_across_multiple_collections() -> None:
    calls: list[tuple[str, str | None, str | None]] = []

    kb_chunk = _scored_chunk(
        doc_id="doc-kb",
        chunk_id="doc-kb#chunk0001",
        title="ARCHITECTURE.md",
        score=0.88,
    )
    upload_chunk = _scored_chunk(
        doc_id="doc-upload",
        chunk_id="doc-upload#chunk0001",
        title="uploaded.csv",
        score=0.81,
    )
    upload_chunk.doc.metadata["source_type"] = "upload"

    def vector_search(
        query: str,
        *,
        top_k: int,
        tenant_id: str,
        doc_id_filter: str | None = None,
        collection_id_filter: str | None = None,
    ):
        del query, top_k, tenant_id
        calls.append(("vector", collection_id_filter, doc_id_filter))
        if collection_id_filter == "default":
            return [kb_chunk]
        if collection_id_filter == "owui-chat-1":
            return [upload_chunk]
        return []

    def keyword_search(
        query: str,
        *,
        top_k: int,
        tenant_id: str,
        doc_id_filter: str | None = None,
        collection_id_filter: str | None = None,
    ):
        del query, top_k, tenant_id
        calls.append(("keyword", collection_id_filter, doc_id_filter))
        return []

    stores = SimpleNamespace(
        chunk_store=SimpleNamespace(
            vector_search=vector_search,
            keyword_search=keyword_search,
        ),
        doc_store=SimpleNamespace(
            fuzzy_search_title=lambda hint, tenant_id, limit=5, collection_id="": []
        ),
    )

    result = retrieve_candidates(
        stores,
        "compare the uploaded file with the architecture docs",
        tenant_id="tenant-123",
        preferred_doc_ids=[],
        must_include_uploads=True,
        top_k_vector=4,
        top_k_keyword=4,
        collection_ids_filter=["default", "owui-chat-1"],
    )

    merged_doc_ids = {(chunk.doc.metadata or {}).get("doc_id") for chunk in result["merged"]}

    assert ("vector", "default", None) in calls
    assert ("vector", "owui-chat-1", None) in calls
    assert ("keyword", "default", None) in calls
    assert ("keyword", "owui-chat-1", None) in calls
    assert merged_doc_ids == {"doc-kb", "doc-upload"}


def test_merge_dedupe_returns_results_sorted_by_score_descending():
    low = _scored_chunk(doc_id="doc-low", chunk_id="doc-low#chunk0001", title="LOW.md", score=0.2)
    high = _scored_chunk(doc_id="doc-high", chunk_id="doc-high#chunk0001", title="HIGH.md", score=0.9)

    merged = merge_dedupe([low, high])

    assert [chunk.doc.metadata["doc_id"] for chunk in merged] == ["doc-high", "doc-low"]


def test_rank_fuse_dedupe_uses_lane_rank_not_raw_score_only() -> None:
    vector_top = _scored_chunk(doc_id="doc-vector", chunk_id="doc-vector#chunk0001", title="VECTOR.md", score=0.99)
    shared_vector = _scored_chunk(doc_id="doc-shared", chunk_id="doc-shared#chunk0001", title="SHARED.md", score=0.2)
    shared_keyword = _scored_chunk(doc_id="doc-shared", chunk_id="doc-shared#chunk0001", title="SHARED.md", score=0.2, method="keyword")
    keyword_top = _scored_chunk(doc_id="doc-keyword", chunk_id="doc-keyword#chunk0001", title="KEYWORD.md", score=12.0, method="keyword")

    fused = rank_fuse_dedupe(
        {
            "vector": [vector_top, shared_vector],
            "keyword": [shared_keyword, keyword_top],
        }
    )

    assert fused[0].doc.metadata["doc_id"] == "doc-shared"
    assert set(fused[0].doc.metadata["_retrieval_lanes"]) == {"vector", "keyword"}


def test_normalize_retrieval_question_strips_guided_tool_routing_prefix() -> None:
    assert (
        normalize_retrieval_question(
            "Search the default knowledge base for this product fact and answer briefly with citations: What are the three AAP plans?"
        )
        == "What are the three AAP plans?"
    )


def test_normalize_retrieval_question_preserves_content_prefix() -> None:
    assert (
        normalize_retrieval_question("Security policy: What happens when an uploaded document is deleted?")
        == "Security policy: What happens when an uploaded document is deleted?"
    )


def test_grade_chunks_preserves_architecture_docs_via_title_hint():
    architecture_doc = Document(
        page_content="Implementation details for the next runtime kernel and routing flow.",
        metadata={"chunk_id": "arch#chunk0001", "title": "ARCHITECTURE.md"},
    )
    prompt_doc = Document(
        page_content="Prompt catalog for demo questions.",
        metadata={"chunk_id": "prompt#chunk0001", "title": "TEST_QUERIES.md"},
    )

    judge = SimpleNamespace(
        invoke=lambda prompt, config=None: SimpleNamespace(
            content='{"grades": [{"chunk_id": "arch#chunk0001", "relevance": 0, "reason": "missed"}, {"chunk_id": "prompt#chunk0001", "relevance": 1, "reason": "partial"}]}'
        )
    )

    graded = grade_chunks(
        judge,
        settings=SimpleNamespace(prompts_backend="local", judge_grading_prompt_path=Path("missing")),
        question="What are the key implementation details in the architecture docs? Cite your sources.",
        chunks=[architecture_doc, prompt_doc],
        callbacks=[],
    )

    by_id = {item.doc.metadata["chunk_id"]: item for item in graded}
    assert by_id["arch#chunk0001"].relevance >= 2
    assert by_id["arch#chunk0001"].reason == "title_hint"


def test_grade_chunks_demotes_prompt_catalog_question_echo():
    prompt_doc = Document(
        page_content="What are the key implementation details in the architecture docs? Cite your sources.",
        metadata={"chunk_id": "prompt#chunk0001", "title": "TEST_QUERIES.md"},
    )

    judge = SimpleNamespace(
        invoke=lambda prompt, config=None: SimpleNamespace(
            content='{"grades": [{"chunk_id": "prompt#chunk0001", "relevance": 3, "reason": "exact match"}]}'
        )
    )

    graded = grade_chunks(
        judge,
        settings=SimpleNamespace(prompts_backend="local", judge_grading_prompt_path=Path("missing")),
        question="What are the key implementation details in the architecture docs? Cite your sources.",
        chunks=[prompt_doc],
        callbacks=[],
    )

    assert graded[0].relevance <= 1
    assert graded[0].reason == "question_echo"


def test_grade_chunks_demotes_meta_catalog_for_architecture_queries():
    prompt_doc = Document(
        page_content="Grouped prompt sets for basic chat, citations, architecture docs, and upload analysis.",
        metadata={"chunk_id": "prompt#chunk0002", "title": "TEST_QUERIES.md"},
    )

    judge = SimpleNamespace(
        invoke=lambda prompt, config=None: SimpleNamespace(
            content='{"grades": [{"chunk_id": "prompt#chunk0002", "relevance": 3, "reason": "semantically related"}]}'
        )
    )

    graded = grade_chunks(
        judge,
        settings=SimpleNamespace(prompts_backend="local", judge_grading_prompt_path=Path("missing")),
        question="key implementation details architecture documentation",
        chunks=[prompt_doc],
        callbacks=[],
    )

    assert graded[0].relevance <= 1
    assert graded[0].reason == "meta_catalog"


def test_grade_chunks_demotes_operational_runbooks_for_architecture_queries():
    runbook_doc = Document(
        page_content="Local startup instructions for Docker Compose services and health checks.",
        metadata={"chunk_id": "runbook#chunk0001", "title": "LOCAL_DOCKER_STACK.md"},
    )

    judge = SimpleNamespace(
        invoke=lambda prompt, config=None: SimpleNamespace(
            content='{"grades": [{"chunk_id": "runbook#chunk0001", "relevance": 3, "reason": "related"}]}'
        )
    )

    graded = grade_chunks(
        judge,
        settings=SimpleNamespace(prompts_backend="local", judge_grading_prompt_path=Path("missing")),
        question="What are the key implementation details in the architecture docs? Cite your sources.",
        chunks=[runbook_doc],
        callbacks=[],
    )

    assert graded[0].relevance <= 1
    assert graded[0].reason == "operational_runbook"


def test_grade_chunks_batches_selected_chunks_into_single_judge_call():
    chunks = [
        Document(
            page_content="First chunk with operating procedure details.",
            metadata={"chunk_id": "doc-a#chunk0001", "title": "OPS_A.md"},
        ),
        Document(
            page_content="Second chunk with escalation and approvals.",
            metadata={"chunk_id": "doc-b#chunk0001", "title": "OPS_B.md"},
        ),
        Document(
            page_content="Third chunk with controls and timelines.",
            metadata={"chunk_id": "doc-c#chunk0001", "title": "OPS_C.md"},
        ),
    ]
    invocations: list[tuple[str, object]] = []

    def invoke(prompt, config=None):
        invocations.append((prompt, config))
        return SimpleNamespace(
            content=(
                '{"grades": ['
                '{"chunk_id": "doc-a#chunk0001", "relevance": 3, "reason": "high"}, '
                '{"chunk_id": "doc-b#chunk0001", "relevance": 2, "reason": "medium"}, '
                '{"chunk_id": "doc-c#chunk0001", "relevance": 1, "reason": "low"}'
                "]}"
            )
        )

    judge = SimpleNamespace(invoke=invoke)

    graded = grade_chunks(
        judge,
        settings=SimpleNamespace(prompts_backend="local", judge_grading_prompt_path=Path("missing")),
        question="Which operating procedures include approvals and timelines?",
        chunks=chunks,
        callbacks=["cb-marker"],
    )

    assert len(invocations) == 1
    prompt, config = invocations[0]
    assert "doc-a#chunk0001" in prompt
    assert "doc-b#chunk0001" in prompt
    assert "doc-c#chunk0001" in prompt
    assert config == {"callbacks": ["cb-marker"]}
    assert [item.doc.metadata["chunk_id"] for item in graded] == [
        "doc-a#chunk0001",
        "doc-b#chunk0001",
        "doc-c#chunk0001",
    ]


def test_grade_chunks_caches_identical_candidate_sets():
    retrieval_module._GRADE_CACHE.clear()
    doc = Document(
        page_content="Alpha approval evidence with budget owner and rollout date.",
        metadata={"chunk_id": "doc-a#chunk0001", "title": "alpha.md"},
    )
    invocations = 0

    def invoke(prompt, config=None):
        nonlocal invocations
        del prompt, config
        invocations += 1
        return SimpleNamespace(
            content='{"grades": [{"chunk_id": "doc-a#chunk0001", "relevance": 3, "reason": "exact"}]}'
        )

    judge = SimpleNamespace(invoke=invoke)
    settings = SimpleNamespace(
        prompts_backend="local",
        judge_grading_prompt_path=Path("missing"),
        rag_heuristic_grading_enabled=False,
    )

    first = grade_chunks(judge, settings=settings, question="Who owns Alpha approval?", chunks=[doc], callbacks=[])
    second = grade_chunks(judge, settings=settings, question="Who owns Alpha approval?", chunks=[doc], callbacks=[])

    assert invocations == 1
    assert first[0].relevance == 3
    assert second[0].relevance == 3
