from __future__ import annotations

from agentic_chatbot_next.persistence.postgres.chunks import ChunkRecord
from agentic_chatbot_next.persistence.postgres.documents import DocumentRecord
from agentic_chatbot_next.rag.requirements import (
    LEGAL_CLAUSE_MODE,
    MANDATORY_MODE,
    STRICT_SHALL_MODE,
    build_requirement_statement_records,
    format_requirement_location,
)


def test_build_requirement_statement_records_extracts_mandatory_language_only() -> None:
    document = DocumentRecord(
        doc_id="doc-req",
        tenant_id="tenant",
        collection_id="default",
        title="SPEC_A.docx",
        source_type="kb",
        content_hash="hash",
        file_type="docx",
    )
    chunk = ChunkRecord(
        chunk_id="doc-req#chunk0001",
        doc_id="doc-req",
        tenant_id="tenant",
        collection_id="default",
        chunk_index=1,
        content=(
            "3.1 The payload shall include a checksum. "
            "The reviewer should record an observation. "
            "The gateway must not expose admin APIs. "
            "The adapter is required to retry once. "
            "The service may defer non-critical work."
        ),
        chunk_type="requirement",
        page_number=2,
        clause_number="3.1",
        section_title="Gateway Controls",
    )

    statements = build_requirement_statement_records(document, [chunk], mode=MANDATORY_MODE)

    assert [item.modality for item in statements] == ["shall", "must_not", "required_to"]
    assert all("should" not in item.statement_text.casefold() for item in statements)
    assert all("may defer" not in item.statement_text.casefold() for item in statements)
    assert format_requirement_location(statements[0]) == "Page 2 · Clause 3.1 · Gateway Controls · Offset 0"


def test_build_requirement_statement_records_strict_mode_keeps_only_shall_forms() -> None:
    document = DocumentRecord(
        doc_id="doc-req",
        tenant_id="tenant",
        collection_id="default",
        title="SPEC_B.md",
        source_type="kb",
        content_hash="hash",
        file_type="md",
    )
    chunk = ChunkRecord(
        chunk_id="doc-req#chunk0001",
        doc_id="doc-req",
        tenant_id="tenant",
        collection_id="default",
        chunk_index=0,
        content=(
            "The subsystem shall authenticate operators before use. "
            "It must retain audit logs for thirty days. "
            "The operator shall not bypass the interlock."
        ),
        chunk_type="requirement",
    )

    statements = build_requirement_statement_records(document, [chunk], mode=STRICT_SHALL_MODE)

    assert [item.modality for item in statements] == ["shall", "shall_not"]
    assert all("must retain" not in item.statement_text.casefold() for item in statements)


def test_build_requirement_statement_records_flags_multi_requirement_sentences_without_over_splitting() -> None:
    document = DocumentRecord(
        doc_id="doc-req",
        tenant_id="tenant",
        collection_id="default",
        title="SPEC_C.txt",
        source_type="kb",
        content_hash="hash",
        file_type="txt",
    )
    chunk = ChunkRecord(
        chunk_id="doc-req#chunk0001",
        doc_id="doc-req",
        tenant_id="tenant",
        collection_id="default",
        chunk_index=2,
        content="The controller shall authenticate users and shall encrypt all exported telemetry.",
        chunk_type="requirement",
    )

    statements = build_requirement_statement_records(document, [chunk], mode=MANDATORY_MODE)

    assert len(statements) == 1
    assert statements[0].multi_requirement is True
    assert statements[0].statement_text == (
        "The controller shall authenticate users and shall encrypt all exported telemetry."
    )


def test_legal_clause_mode_extracts_far_dfars_style_obligations_without_changing_mandatory_mode() -> None:
    document = DocumentRecord(
        doc_id="doc-legal",
        tenant_id="tenant",
        collection_id="default",
        title="DFARS_252_204_7012.txt",
        source_type="kb",
        content_hash="hash",
        file_type="txt",
    )
    chunk = ChunkRecord(
        chunk_id="doc-legal#chunk0001",
        doc_id="doc-legal",
        tenant_id="tenant",
        collection_id="default",
        chunk_index=0,
        content=(
            "252.204-7012 Safeguarding Covered Defense Information. "
            "(b)(1) The Contractor shall provide adequate security. "
            "(c) The Contractor will rapidly report cyber incidents. "
            "(d) The offeror agrees to flow down this clause."
        ),
        chunk_type="clause",
        clause_number="252.204-7012",
        section_title="Safeguarding Covered Defense Information",
    )

    mandatory = build_requirement_statement_records(document, [chunk], mode=MANDATORY_MODE)
    legal = build_requirement_statement_records(document, [chunk], mode=LEGAL_CLAUSE_MODE)

    assert [item.modality for item in mandatory] == ["shall"]
    assert [item.modality for item in legal] == ["shall", "will", "agrees_to"]
