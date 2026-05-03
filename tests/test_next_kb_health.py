from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_chatbot_next.persistence.postgres.documents import DocumentRecord
from agentic_chatbot_next.rag.ingest import (
    build_collection_health_report,
    build_kb_health_report,
    ingest_paths,
    repair_collection_documents,
    repair_kb_collection,
)


class _FakeDocStore:
    def __init__(self) -> None:
        self.records: dict[str, DocumentRecord] = {}

    def upsert_document(self, record: DocumentRecord) -> None:
        self.records[record.doc_id] = record

    def list_documents(self, source_type: str = "", tenant_id: str = "tenant", collection_id: str = ""):
        matches = []
        for record in self.records.values():
            if record.tenant_id != tenant_id:
                continue
            if source_type and record.source_type != source_type:
                continue
            if collection_id and record.collection_id != collection_id:
                continue
            matches.append(record)
        return sorted(matches, key=lambda item: (item.ingested_at, item.title))

    def delete_document(self, doc_id: str, tenant_id: str) -> None:
        record = self.records.get(doc_id)
        if record is not None and record.tenant_id == tenant_id:
            self.records.pop(doc_id, None)

    def get_document(self, doc_id: str, tenant_id: str) -> DocumentRecord | None:
        record = self.records.get(doc_id)
        if record is None or record.tenant_id != tenant_id:
            return None
        return record


class _FakeChunkStore:
    def __init__(self) -> None:
        self.records: dict[str, list[object]] = {}

    def add_chunks(self, chunks, *, tenant_id: str) -> None:
        del tenant_id
        if not chunks:
            return
        self.records[chunks[0].doc_id] = list(chunks)


def _settings(tmp_path: Path) -> SimpleNamespace:
    kb_dir = tmp_path / "kb"
    docs_dir = tmp_path / "docs"
    kb_dir.mkdir()
    docs_dir.mkdir()
    return SimpleNamespace(
        kb_dir=kb_dir,
        kb_extra_dirs=(docs_dir,),
        default_collection_id="default",
        default_tenant_id="tenant",
        chunk_size=400,
        chunk_overlap=40,
        ocr_enabled=False,
        docling_enabled=False,
        ocr_min_page_chars=20,
        ocr_language="eng",
        ocr_use_gpu=False,
        object_store_backend="local",
        graph_search_enabled=False,
    )


def _stores() -> SimpleNamespace:
    return SimpleNamespace(
        doc_store=_FakeDocStore(),
        chunk_store=_FakeChunkStore(),
        graph_store=None,
    )


def test_ingest_paths_retains_prior_versions_and_marks_latest_active(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    stores = _stores()
    path = settings.kb_dir / "ARCHITECTURE.md"
    path.write_text("# Architecture\n\nVersion one.", encoding="utf-8")

    first_doc_ids = ingest_paths(
        settings,
        stores,
        [path],
        source_type="kb",
        tenant_id="tenant",
        collection_id="default",
    )
    assert len(first_doc_ids) == 1

    path.write_text("# Architecture\n\nVersion two.", encoding="utf-8")
    second_doc_ids = ingest_paths(
        settings,
        stores,
        [path],
        source_type="kb",
        tenant_id="tenant",
        collection_id="default",
    )

    assert len(second_doc_ids) == 1
    records = stores.doc_store.list_documents(source_type="kb", tenant_id="tenant", collection_id="default")
    assert len(records) == 2
    by_id = {record.doc_id: record for record in records}
    assert by_id[first_doc_ids[0]].active is False
    assert by_id[first_doc_ids[0]].superseded_at
    assert by_id[second_doc_ids[0]].active is True
    assert by_id[second_doc_ids[0]].version_ordinal == 2

    health = build_kb_health_report(settings, stores, tenant_id="tenant", collection_id="default")
    assert health.stale_version_count == 1
    assert len(health.duplicate_groups) == 0


def test_ingest_paths_persists_remote_blob_reference(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.object_store_backend = "s3"
    stores = _stores()
    path = tmp_path / "staged-upload.txt"
    path.write_text("Remote-backed upload content.", encoding="utf-8")
    blob_ref = {
        "backend": "s3",
        "uri": "s3://agentic-uploads/uploads/staged-upload.txt",
        "bucket": "agentic-uploads",
        "key": "uploads/staged-upload.txt",
        "etag": "etag-1",
        "size": path.stat().st_size,
        "content_type": "text/plain",
        "sha1": "sha1",
    }

    doc_ids = ingest_paths(
        settings,
        stores,
        [path],
        source_type="upload",
        tenant_id="tenant",
        collection_id="uploads",
        source_display_paths={str(path.resolve()): "staged-upload.txt"},
        source_identities={str(path.resolve()): "upload:staged-upload.txt"},
        source_metadata_by_path={
            str(path.resolve()): {
                "blob_ref": blob_ref,
                "source_uri": blob_ref["uri"],
                "original_filename": "staged-upload.txt",
                "mime_type": "text/plain",
            }
        },
    )

    assert len(doc_ids) == 1
    record = stores.doc_store.get_document(doc_ids[0], tenant_id="tenant")
    assert record is not None
    assert record.source_path == blob_ref["uri"]
    assert record.source_uri == blob_ref["uri"]
    assert record.source_storage_backend == "s3"
    assert record.source_object_bucket == "agentic-uploads"
    assert record.source_object_key == "uploads/staged-upload.txt"


def test_ingest_paths_supports_pptx_parser_provenance(tmp_path: Path) -> None:
    pptx = pytest.importorskip("pptx")
    settings = _settings(tmp_path)
    stores = _stores()
    path = tmp_path / "program_status.pptx"
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "Program Status"
    text_box = slide.shapes.add_textbox(0, 0, 5000000, 1000000)
    text_box.text_frame.text = "CDRL A001 is approved for WBS 1.2.3."
    presentation.save(path)

    doc_ids = ingest_paths(
        settings,
        stores,
        [path],
        source_type="upload",
        tenant_id="tenant",
        collection_id="uploads",
        metadata_enrichment="deterministic",
    )

    assert len(doc_ids) == 1
    record = stores.doc_store.get_document(doc_ids[0], tenant_id="tenant")
    assert record is not None
    assert record.file_type == "pptx"
    assert record.doc_type == "cdrl"
    assert "python-pptx" in record.parser_provenance["chain"]
    assert record.signal_summary["cdrl"]["count"] >= 1


def test_build_kb_health_report_and_repair_handle_same_source_duplicates(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    stores = _stores()
    path = Path(settings.kb_extra_dirs[0]) / "ARCHITECTURE.md"
    text = "# Architecture\n\nCurrent version."
    path.write_text(text, encoding="utf-8")
    current_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()

    stores.doc_store.upsert_document(
        DocumentRecord(
            doc_id="doc-old",
            tenant_id="tenant",
            collection_id="default",
            title="ARCHITECTURE.md",
            source_type="kb",
            content_hash="hash-old",
            source_path=str(path),
            num_chunks=2,
            ingested_at="2026-04-09T02:00:00Z",
            file_type="md",
            doc_structure_type="process_flow_doc",
        )
    )
    stores.doc_store.upsert_document(
        DocumentRecord(
            doc_id="doc-new",
            tenant_id="tenant",
            collection_id="default",
            title="ARCHITECTURE.md",
            source_type="kb",
            content_hash=current_hash,
            source_path=str(path),
            num_chunks=3,
            ingested_at="2026-04-09T03:00:00Z",
            file_type="md",
            doc_structure_type="process_flow_doc",
        )
    )

    health_before = build_kb_health_report(settings, stores, tenant_id="tenant", collection_id="default")
    assert len(health_before.duplicate_groups) == 1
    assert health_before.duplicate_groups[0].active_doc_id == "doc-new"

    result = repair_kb_collection(settings, stores, tenant_id="tenant", collection_id="default")

    assert result.deleted_doc_ids == ("doc-old",)
    health_after = build_kb_health_report(settings, stores, tenant_id="tenant", collection_id="default")
    assert len(health_after.duplicate_groups) == 0


def test_collection_health_reports_extraction_and_metadata_flags(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    stores = _stores()
    stores.doc_store.upsert_document(
        DocumentRecord(
            doc_id="doc-failed",
            tenant_id="tenant",
            collection_id="uploads",
            title="bad.pdf",
            source_type="upload",
            content_hash="hash-failed",
            source_path=str(tmp_path / "bad.pdf"),
            ingested_at="2026-04-09T02:00:00Z",
            file_type="pdf",
            extraction_status="failed",
            extraction_error="parser failed",
            metadata_confidence=0.2,
            parser_provenance={"chain": ["docling", "pypdf"], "steps": []},
        )
    )

    health = build_collection_health_report(settings, stores, tenant_id="tenant", collection_id="uploads")

    assert health.extraction_failure_count == 1
    assert health.low_confidence_metadata_count == 1
    assert health.parser_counts["docling"] == 1
    assert health.source_groups[0].status == "extraction_failed"


def test_build_collection_health_report_and_repair_canonicalize_repo_alias_paths(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    stores = _stores()
    repo_root = Path(__file__).resolve().parents[1]
    repo_path = repo_root / "defense_rag_test_corpus" / "documents" / "rfp-sample.md"
    container_path = Path("/app/defense_rag_test_corpus/documents/rfp-sample.md")

    stores.doc_store.upsert_document(
        DocumentRecord(
            doc_id="doc-host",
            tenant_id="tenant",
            collection_id="rfp-corpus",
            title="rfp-sample.md",
            source_type="upload",
            content_hash="hash-1",
            source_path=str(repo_path),
            source_identity=f"path:{repo_path}",
            num_chunks=1,
            ingested_at="2026-04-09T02:00:00Z",
            file_type="md",
            doc_structure_type="text",
        )
    )
    stores.doc_store.upsert_document(
        DocumentRecord(
            doc_id="doc-container",
            tenant_id="tenant",
            collection_id="rfp-corpus",
            title="rfp-sample.md",
            source_type="upload",
            content_hash="hash-1",
            source_path=str(container_path),
            source_identity=f"path:{container_path}",
            num_chunks=1,
            ingested_at="2026-04-09T03:00:00Z",
            file_type="md",
            doc_structure_type="text",
        )
    )

    health_before = build_collection_health_report(settings, stores, tenant_id="tenant", collection_id="rfp-corpus")
    assert health_before.maintenance_policy == "indexed_documents"
    assert len(health_before.duplicate_groups) == 1
    assert health_before.duplicate_groups[0].active_doc_id == "doc-container"
    assert health_before.duplicate_groups[0].source_identity.startswith("path:repo://")

    result = repair_collection_documents(settings, stores, tenant_id="tenant", collection_id="rfp-corpus")

    assert result.deleted_doc_ids == ("doc-host",)
    health_after = build_collection_health_report(settings, stores, tenant_id="tenant", collection_id="rfp-corpus")
    assert len(health_after.duplicate_groups) == 0
