from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document

from agentic_chatbot_next.observability.events import RuntimeEvent
from agentic_chatbot_next.persistence.postgres.chunks import ScoredChunk
from agentic_chatbot_next.rag.collection_selection import select_collection_for_query


def _settings(**overrides):
    payload = {
        "default_collection_id": "default",
        "default_tenant_id": "tenant",
        "max_parallel_collection_probes": 4,
        "max_collection_discovery_collections": 25,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _session(metadata=None):
    return SimpleNamespace(
        tenant_id="tenant",
        session_id="session-1",
        active_agent="general",
        metadata=dict(metadata or {}),
        uploaded_doc_ids=[],
    )


def _record(doc_id: str, title: str, collection_id: str):
    return SimpleNamespace(
        doc_id=doc_id,
        title=title,
        source_path=f"/kb/{collection_id}/{title}",
        source_type="kb",
        collection_id=collection_id,
        doc_structure_type="general",
    )


def _chunk(doc_id: str, title: str, collection_id: str, score: float, method: str = "keyword"):
    return ScoredChunk(
        doc=Document(
            page_content=f"{title} content",
            metadata={
                "doc_id": doc_id,
                "title": title,
                "collection_id": collection_id,
                "source_path": f"/kb/{collection_id}/{title}",
            },
        ),
        score=score,
        method=method,
    )


class _DocStore:
    def __init__(self, records):
        self.records = list(records)

    def list_documents(self, tenant_id="tenant", collection_id="", source_type=""):
        del tenant_id
        return [
            item
            for item in self.records
            if (not collection_id or item.collection_id == collection_id)
            and (not source_type or item.source_type == source_type)
        ]

    def list_collections(self, tenant_id="tenant"):
        del tenant_id
        ids = sorted({item.collection_id for item in self.records})
        return [
            {"collection_id": collection_id, "source_type_counts": {"kb": 1}}
            for collection_id in ids
        ]

    def fuzzy_search_title(self, hint, tenant_id="tenant", limit=5, collection_id=""):
        del tenant_id, limit
        hint_terms = set(str(hint or "").lower().split())
        rows = []
        for record in self.list_documents(collection_id=collection_id, source_type="kb"):
            if hint_terms & set(record.title.lower().replace("_", " ").split()):
                rows.append({"doc_id": record.doc_id, "title": record.title, "score": 0.6})
        return rows


class _ChunkStore:
    def __init__(self, *, keyword_hits=None, vector_hits=None, vector_raises=False):
        self.keyword_hits = dict(keyword_hits or {})
        self.vector_hits = dict(vector_hits or {})
        self.vector_raises = vector_raises

    def keyword_search(self, query, *, top_k, collection_id_filter=None, tenant_id="tenant"):
        del query, top_k, tenant_id
        return list(self.keyword_hits.get(collection_id_filter or "", []))

    def vector_search(self, query, *, top_k, collection_id_filter=None, tenant_id="tenant"):
        del query, top_k, tenant_id
        if self.vector_raises:
            raise RuntimeError("embedding unavailable")
        return list(self.vector_hits.get(collection_id_filter or "", []))


class _Sink:
    def __init__(self):
        self.events = []

    def emit(self, event: RuntimeEvent):
        self.events.append(event.to_dict())


def _stores(records, *, keyword_hits=None, vector_hits=None, vector_raises=False):
    return SimpleNamespace(
        doc_store=_DocStore(records),
        chunk_store=_ChunkStore(
            keyword_hits=keyword_hits,
            vector_hits=vector_hits,
            vector_raises=vector_raises,
        ),
    )


def test_collection_selection_preserves_explicit_collection():
    selection = select_collection_for_query(
        _stores([]),
        _settings(),
        _session(metadata={"available_kb_collection_ids": ["default", "rfp"]}),
        "rate limit policy",
        explicit_collection_id="rfp",
    )

    assert selection.status == "explicit"
    assert selection.selected_collection_id == "rfp"


def test_collection_selection_uses_single_accessible_collection_without_probe():
    selection = select_collection_for_query(
        _stores([]),
        _settings(),
        _session(metadata={"available_kb_collection_ids": ["default"]}),
        "rate limit policy",
    )

    assert selection.status == "single"
    assert selection.selected_collection_id == "default"


def test_collection_selection_picks_clear_parallel_probe_winner():
    records = [
        _record("doc-default", "architecture.md", "default"),
        _record("doc-policy", "rate_limit_policy.md", "policy"),
    ]
    selection = select_collection_for_query(
        _stores(
            records,
            keyword_hits={
                "policy": [_chunk("doc-policy", "rate_limit_policy.md", "policy", 0.9)],
                "default": [],
            },
        ),
        _settings(),
        _session(metadata={"available_kb_collection_ids": ["default", "policy"]}),
        "rate limit policy",
    )

    assert selection.status == "selected"
    assert selection.selected_collection_id == "policy"
    assert selection.ranked_collections[0].keyword_hits == 1


def test_collection_selection_returns_no_match_when_all_probes_are_empty():
    records = [
        _record("doc-default", "architecture.md", "default"),
        _record("doc-policy", "security.md", "policy"),
    ]
    selection = select_collection_for_query(
        _stores(records),
        _settings(),
        _session(metadata={"available_kb_collection_ids": ["default", "policy"]}),
        "rate limit policy",
    )

    assert selection.status == "no_match"
    assert selection.selected_collection_id == ""
    assert selection.clarification_options == ["policy", "default"] or selection.clarification_options == ["default", "policy"]


def test_collection_selection_marks_tied_candidates_ambiguous():
    records = [
        _record("doc-a", "rate_limit_policy.md", "default"),
        _record("doc-b", "rate_limit_policy.md", "policy"),
    ]
    selection = select_collection_for_query(
        _stores(
            records,
            keyword_hits={
                "default": [_chunk("doc-a", "rate_limit_policy.md", "default", 0.8)],
                "policy": [_chunk("doc-b", "rate_limit_policy.md", "policy", 0.8)],
            },
        ),
        _settings(),
        _session(metadata={"available_kb_collection_ids": ["default", "policy"]}),
        "rate limit policy",
    )

    assert selection.status == "ambiguous"
    assert set(selection.clarification_options[:2]) == {"default", "policy"}


def test_collection_selection_falls_back_when_vector_probe_fails():
    records = [
        _record("doc-default", "architecture.md", "default"),
        _record("doc-policy", "rate_limit_policy.md", "policy"),
    ]
    selection = select_collection_for_query(
        _stores(
            records,
            keyword_hits={"policy": [_chunk("doc-policy", "rate_limit_policy.md", "policy", 0.95)]},
            vector_raises=True,
        ),
        _settings(),
        _session(metadata={"available_kb_collection_ids": ["default", "policy"]}),
        "rate limit policy",
    )

    assert selection.status == "selected"
    assert selection.selected_collection_id == "policy"
    assert "keyword" in selection.searched_methods


def test_collection_selection_narrows_large_candidate_sets_with_metadata():
    records = [
        _record("doc-a", "architecture.md", "a"),
        _record("doc-b", "security.md", "b"),
        _record("doc-c", "rate_limit_policy.md", "c"),
        _record("doc-d", "operations.md", "d"),
    ]
    selection = select_collection_for_query(
        _stores(records),
        _settings(max_collection_discovery_collections=2),
        _session(metadata={"available_kb_collection_ids": ["a", "b", "c", "d"]}),
        "rate limit policy",
    )

    assert selection.status == "selected"
    assert selection.selected_collection_id == "c"


def test_collection_selection_emits_trace_events():
    records = [
        _record("doc-default", "architecture.md", "default"),
        _record("doc-policy", "rate_limit_policy.md", "policy"),
    ]
    sink = _Sink()
    selection = select_collection_for_query(
        _stores(records, keyword_hits={"policy": [_chunk("doc-policy", "rate_limit_policy.md", "policy", 0.9)]}),
        _settings(),
        _session(metadata={"available_kb_collection_ids": ["default", "policy"]}),
        "rate limit policy",
        event_sink=sink,
    )

    assert selection.selected_collection_id == "policy"
    assert [event["event_type"] for event in sink.events] == [
        "collection_selection_started",
        "collection_selection_completed",
    ]
    completed_payload = sink.events[-1]["payload"]
    top_documents = completed_payload["ranked_collections"][0]["top_documents"]
    assert top_documents
    assert {item["collection_id"] for item in top_documents} == {"policy"}
