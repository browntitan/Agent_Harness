from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from agentic_chatbot_next.tools.indexed_docs import make_indexed_doc_tools


def _record(doc_id: str, title: str, source_path: str, collection_id: str = "default") -> SimpleNamespace:
    return SimpleNamespace(
        doc_id=doc_id,
        title=title,
        source_type="kb",
        source_path=source_path,
        collection_id=collection_id,
        file_type="md",
        doc_structure_type="general",
        num_chunks=4,
    )


def _stores() -> SimpleNamespace:
    records = {
        "doc-arch": _record("doc-arch", "ARCHITECTURE.md", "/repo/docs/ARCHITECTURE.md"),
        "doc-c4": _record("doc-c4", "C4_ARCHITECTURE.md", "/repo/docs/C4_ARCHITECTURE.md"),
    }
    return SimpleNamespace(
        doc_store=SimpleNamespace(
            list_documents=lambda tenant_id="tenant", collection_id="", source_type="": [
                record
                for record in records.values()
                if (not collection_id or record.collection_id == collection_id)
                and (not source_type or record.source_type == source_type)
            ],
            list_collections=lambda tenant_id="tenant": [
                {"collection_id": "default", "source_type_counts": {"kb": len(records)}}
            ],
            fuzzy_search_title=lambda hint, tenant_id, limit=5, collection_id="": [],
            get_document=lambda doc_id, tenant_id="tenant": records.get(doc_id),
        ),
        chunk_store=SimpleNamespace(
            chunk_count=lambda doc_id=None, tenant_id="tenant": 4,
            get_chunks_by_index_range=lambda doc_id, min_idx, max_idx, tenant_id="tenant": [
                SimpleNamespace(
                    chunk_id=f"{doc_id}#chunk{index:04d}",
                    doc_id=doc_id,
                    chunk_index=index,
                    chunk_type="general",
                    section_title=f"Section {index}",
                    clause_number=None,
                    page_number=None,
                    content=f"content {index}",
                )
                for index in range(min_idx, max_idx + 1)
            ],
            get_structure_outline=lambda doc_id, tenant_id="tenant": [
                {"section_title": "Overview"},
                {"section_title": "Runtime"},
            ],
        ),
    )


def test_make_indexed_doc_tools_exposes_expected_live_tool_names() -> None:
    tools = make_indexed_doc_tools(
        SimpleNamespace(default_collection_id="default", default_tenant_id="tenant"),
        _stores(),
        SimpleNamespace(tenant_id="tenant", metadata={}),
    )

    assert {tool.name for tool in tools} == {
        "resolve_indexed_docs",
        "search_indexed_docs",
        "read_indexed_doc",
        "compare_indexed_docs",
    }


def test_search_indexed_docs_returns_ranked_title_candidates() -> None:
    tools = {
        tool.name: tool
        for tool in make_indexed_doc_tools(
            SimpleNamespace(default_collection_id="default", default_tenant_id="tenant"),
            _stores(),
            SimpleNamespace(tenant_id="tenant", metadata={}),
        )
    }

    result = tools["search_indexed_docs"].invoke({"query": 'Review "C4_ARCHITECTURE.md" for architecture details'})

    assert result["collection_id"] == "default"
    assert result["collection_selection"]["status"] == "single"
    assert len(result["results"]) >= 1
    assert result["results"][0]["doc_id"] == "doc-c4"
    assert result["results"][0]["title"] == "C4_ARCHITECTURE.md"
    assert result["results"][0]["collection_id"] == "default"
    assert result["results"][0]["match_reason"] in {"metadata_title", "metadata_path", "fuzzy_title"}
    assert result["results"][0]["score"] > 0.0


def test_search_indexed_docs_keeps_kb_lookup_out_of_openwebui_upload_collection() -> None:
    tools = {
        tool.name: tool
        for tool in make_indexed_doc_tools(
            SimpleNamespace(default_collection_id="default", default_tenant_id="tenant"),
            _stores(),
            SimpleNamespace(
                tenant_id="tenant",
                metadata={
                    "collection_id": "owui-chat-1",
                    "upload_collection_id": "owui-chat-1",
                    "kb_collection_id": "default",
                    "available_kb_collection_ids": ["default"],
                },
            ),
        )
    }

    result = tools["search_indexed_docs"].invoke({"query": 'Review "C4_ARCHITECTURE.md"', "source_type": "kb"})

    assert result["collection_id"] == "default"
    assert result["collection_selection"]["selected_collection_id"] == "default"
    assert result["results"]
    assert {item["collection_id"] for item in result["results"]} == {"default"}


def test_search_indexed_docs_preserves_explicit_collection_id() -> None:
    tools = {
        tool.name: tool
        for tool in make_indexed_doc_tools(
            SimpleNamespace(default_collection_id="default", default_tenant_id="tenant"),
            _stores(),
            SimpleNamespace(
                tenant_id="tenant",
                metadata={"available_kb_collection_ids": ["default", "rfp"]},
            ),
        )
    }

    result = tools["search_indexed_docs"].invoke(
        {"query": "architecture details", "source_type": "kb", "collection_id": "rfp"}
    )

    assert result["collection_id"] == "rfp"
    assert result["collection_selection"] == {}


def test_read_indexed_doc_full_mode_returns_paginated_chunks() -> None:
    tools = {
        tool.name: tool
        for tool in make_indexed_doc_tools(
            SimpleNamespace(default_collection_id="default", default_tenant_id="tenant"),
            _stores(),
            SimpleNamespace(tenant_id="tenant", metadata={}),
        )
    }

    result = tools["read_indexed_doc"].invoke(
        {"doc_id": "doc-arch", "mode": "full", "cursor": 1, "max_chunks": 2}
    )

    assert result["document"]["title"] == "ARCHITECTURE.md"
    assert [item["chunk_index"] for item in result["chunks"]] == [1, 2]
    assert result["next_cursor"] == 3
    assert result["has_more"] is True


def test_compare_indexed_docs_returns_evidence_and_section_differences(monkeypatch) -> None:
    stores = _stores()
    tools = {
        tool.name: tool
        for tool in make_indexed_doc_tools(
            SimpleNamespace(default_collection_id="default", default_tenant_id="tenant"),
            stores,
            SimpleNamespace(tenant_id="tenant", metadata={}),
        )
    }

    def fake_read_document(self, doc_id: str, *, focus: str = "", max_chunks: int = 6):
        del self, focus, max_chunks
        if doc_id == "doc-arch":
            return [
                Document(
                    page_content="Architecture overview and runtime responsibilities.",
                    metadata={
                        "doc_id": "doc-arch",
                        "chunk_id": "doc-arch#chunk0001",
                        "chunk_index": 1,
                        "section_title": "Overview",
                    },
                )
            ]
        return [
            Document(
                page_content="C4 system context and container relationships.",
                metadata={
                    "doc_id": "doc-c4",
                    "chunk_id": "doc-c4#chunk0001",
                    "chunk_index": 1,
                    "section_title": "System Context",
                },
            )
        ]

    monkeypatch.setattr(
        "agentic_chatbot_next.tools.indexed_docs.CorpusRetrievalAdapter.read_document",
        fake_read_document,
    )
    stores.chunk_store.get_structure_outline = lambda doc_id, tenant_id="tenant": (
        [{"section_title": "Overview"}, {"section_title": "Runtime"}]
        if doc_id == "doc-arch"
        else [{"section_title": "Overview"}, {"section_title": "System Context"}]
    )

    result = tools["compare_indexed_docs"].invoke(
        {"left_doc_id": "doc-arch", "right_doc_id": "doc-c4", "focus": "runtime"}
    )

    assert result["left_document"]["title"] == "ARCHITECTURE.md"
    assert result["right_document"]["title"] == "C4_ARCHITECTURE.md"
    assert result["shared_sections"] == ["Overview"]
    assert result["left_only_sections"] == ["Runtime"]
    assert result["right_only_sections"] == ["System Context"]
    assert result["supporting_citation_ids"]["left"] == ["doc-arch#chunk0001"]
    assert result["supporting_citation_ids"]["right"] == ["doc-c4#chunk0001"]
