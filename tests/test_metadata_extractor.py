from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from agentic_chatbot_next.rag.metadata_extractor import (
    INDEX_METADATA_VERSION,
    build_chunk_index_metadata,
    build_document_index_metadata,
    summarize_index_metadata,
)
from agentic_chatbot_next.rag.ingest import _build_chunk_records
from agentic_chatbot_next.rag.structure_detector import detect_structure


def test_document_index_metadata_detects_requirements_outline_and_entities(tmp_path: Path) -> None:
    path = tmp_path / "system_spec.md"
    text = (
        "# Raven Crest System Specification\n\n"
        "1. Introduction\n"
        "North Coast Systems LLC provides the payload adapter.\n\n"
        "2. Requirements\n"
        "REQ-001 The gateway shall authenticate operators before use.\n"
        "REQ-002 The gateway must not expose admin APIs.\n"
    )
    docs = [Document(page_content=text, metadata={"parser": "unit-test"})]
    structure = detect_structure(text)

    metadata = build_document_index_metadata(
        path=path,
        raw_docs=docs,
        structure=structure,
        metadata_profile="auto",
    )

    assert metadata.extractor_version == INDEX_METADATA_VERSION
    assert metadata.doc_structure_type == "requirements_doc"
    assert "requirements" in metadata.tags
    assert metadata.stats["requirement_signal_count"] >= 2
    assert any(item["title"] == "Requirements" for item in metadata.outline)
    assert "North Coast Systems LLC" in metadata.entities


def test_chunk_index_metadata_preserves_location_tags_and_summary() -> None:
    document_metadata = {
        "extractor_version": INDEX_METADATA_VERSION,
        "metadata_profile": "auto",
        "doc_structure_type": "structured_clauses",
        "tags": ["structured"],
    }
    chunk = Document(
        page_content="3.2 Security Controls\nThe service shall retain audit logs.",
        metadata={
            "chunk_type": "requirement",
            "chunk_index": 2,
            "clause_number": "3.2",
            "section_title": "Security Controls",
            "page": 4,
        },
    )

    metadata = build_chunk_index_metadata(
        chunk,
        chunk_id="doc-1#chunk0002",
        chunk_index=2,
        document_metadata=document_metadata,
    )
    summary = summarize_index_metadata([document_metadata], [metadata], metadata_profile="auto")

    assert metadata.location["clause_number"] == "3.2"
    assert metadata.location["page_number"] == 4
    assert "requirements" in metadata.tags
    assert summary.chunk_count == 1
    assert summary.tag_counts["requirements"] == 1


def test_chunk_records_embed_with_metadata_context_while_storing_clean_content() -> None:
    chunk = Document(
        page_content="The gateway shall authenticate users.",
        metadata={"chunk_type": "requirement", "section_title": "Security Controls", "page": 7},
    )

    records = _build_chunk_records(
        [chunk],
        "doc-1",
        collection_id="default",
        document_index_metadata={
            "extractor_version": INDEX_METADATA_VERSION,
            "metadata_profile": "auto",
            "doc_structure_type": "requirements_doc",
            "tags": ["requirements"],
        },
    )

    assert records[0].content == "The gateway shall authenticate users."
    assert records[0].embedding_text is not None
    assert records[0].embedding_text.startswith("[index metadata]")
    assert "section: Security Controls" in records[0].embedding_text
