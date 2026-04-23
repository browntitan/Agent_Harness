from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

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


def test_ingest_paths_replaces_prior_active_doc_for_same_kb_source(tmp_path: Path) -> None:
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
    assert len(records) == 1
    assert records[0].doc_id == second_doc_ids[0]
    assert records[0].doc_id != first_doc_ids[0]


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
