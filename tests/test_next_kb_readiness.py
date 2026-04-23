from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agentic_chatbot_next.rag.engine import run_rag_contract
from agentic_chatbot_next.rag.ingest import (
    build_kb_coverage_status,
    ensure_kb_indexed,
    get_collection_readiness_status,
)


def _settings(tmp_path: Path) -> SimpleNamespace:
    kb_dir = tmp_path / "kb"
    docs_dir = tmp_path / "docs"
    kb_dir.mkdir()
    docs_dir.mkdir()
    (kb_dir / "msa.md").write_text("# Agreement\n", encoding="utf-8")
    (docs_dir / "ARCHITECTURE.md").write_text("# Architecture\n", encoding="utf-8")
    return SimpleNamespace(
        kb_dir=kb_dir,
        kb_extra_dirs=(docs_dir,),
        default_collection_id="default",
        default_tenant_id="tenant-123",
        rag_min_evidence_chunks=1,
        prompts_backend="local",
        judge_grading_prompt_path=Path("missing"),
        grounded_answer_prompt_path=Path("missing"),
        seed_demo_kb_on_startup=True,
    )


def test_build_kb_coverage_status_detects_missing_configured_sources(tmp_path: Path):
    settings = _settings(tmp_path)

    status = build_kb_coverage_status(
        settings,
        [{"source_path": str(settings.kb_dir / "msa.md"), "title": "msa.md"}],
        tenant_id="tenant-123",
        collection_id="default",
    )

    assert status.ready is False
    assert status.reason == "kb_coverage_missing"
    assert status.missing_source_paths == (str((settings.kb_extra_dirs[0] / "ARCHITECTURE.md").resolve()),)


def test_ensure_kb_indexed_auto_syncs_missing_sources(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    records: list[SimpleNamespace] = []

    class _DocStore:
        def list_documents(self, **kwargs):
            del kwargs
            return list(records)

    def fake_ingest_paths(settings_arg, stores_arg, paths, *, source_type, tenant_id, collection_id):
        del settings_arg, stores_arg, source_type, tenant_id, collection_id
        doc_ids: list[str] = []
        for path in paths:
            records.append(SimpleNamespace(source_path=str(path), title=Path(path).name))
            doc_ids.append(f"doc-{Path(path).stem}")
        return doc_ids

    monkeypatch.setattr("agentic_chatbot_next.rag.ingest.ingest_paths", fake_ingest_paths)

    status = ensure_kb_indexed(
        settings,
        SimpleNamespace(doc_store=_DocStore()),
        tenant_id="tenant-123",
        collection_id="default",
        attempt_sync=True,
    )

    assert status.ready is True
    assert status.sync_attempted is True
    assert set(status.synced_doc_ids) == {"doc-msa", "doc-ARCHITECTURE"}


def test_run_rag_contract_returns_operator_fix_when_kb_is_not_ready(tmp_path: Path):
    settings = _settings(tmp_path)
    stores = SimpleNamespace(
        chunk_store=SimpleNamespace(
            vector_search=lambda *args, **kwargs: [],
            keyword_search=lambda *args, **kwargs: [],
        ),
        doc_store=SimpleNamespace(
            list_documents=lambda **kwargs: [],
            fuzzy_search_title=lambda hint, tenant_id, limit=5, collection_id="": [],
        ),
    )
    providers = SimpleNamespace(
        judge=SimpleNamespace(invoke=lambda *args, **kwargs: SimpleNamespace(content='{"grades": []}')),
        chat=SimpleNamespace(invoke=lambda *args, **kwargs: SimpleNamespace(content="unused")),
    )
    session = SimpleNamespace(tenant_id="tenant-123")

    contract = run_rag_contract(
        settings,
        stores,
        providers=providers,
        session=session,
        query="What are the key implementation details in the architecture docs? Cite your sources.",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=5,
        top_k_keyword=5,
        max_retries=1,
        callbacks=[],
    )

    assert "not indexed for collection 'default'" in contract.answer
    assert "python run.py sync-kb --collection-id default" in contract.answer
    assert contract.warnings == ["KB_COVERAGE_MISSING"]


def test_run_rag_contract_keeps_normal_no_evidence_fallback_when_kb_is_ready(tmp_path: Path):
    settings = _settings(tmp_path)
    architecture_path = str((settings.kb_extra_dirs[0] / "ARCHITECTURE.md").resolve())
    msa_path = str((settings.kb_dir / "msa.md").resolve())
    stores = SimpleNamespace(
        chunk_store=SimpleNamespace(
            vector_search=lambda *args, **kwargs: [],
            keyword_search=lambda *args, **kwargs: [],
        ),
        doc_store=SimpleNamespace(
            list_documents=lambda **kwargs: [
                SimpleNamespace(source_path=msa_path, title="msa.md"),
                SimpleNamespace(source_path=architecture_path, title="ARCHITECTURE.md"),
            ],
            fuzzy_search_title=lambda hint, tenant_id, limit=5, collection_id="": [],
        ),
    )
    providers = SimpleNamespace(
        judge=SimpleNamespace(invoke=lambda *args, **kwargs: SimpleNamespace(content='{"grades": []}')),
        chat=SimpleNamespace(invoke=lambda *args, **kwargs: SimpleNamespace(content="unused")),
    )
    session = SimpleNamespace(tenant_id="tenant-123")

    contract = run_rag_contract(
        settings,
        stores,
        providers=providers,
        session=session,
        query="Summarize a non-existent appendix and cite it.",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=5,
        top_k_keyword=5,
        max_retries=1,
        callbacks=[],
    )

    assert "couldn't confidently answer from the retrieved evidence" in contract.answer
    assert "sync-kb" not in contract.answer


def test_run_rag_contract_uses_shared_kb_collection_for_openwebui_chat_scope(tmp_path: Path):
    settings = _settings(tmp_path)
    architecture_path = str((settings.kb_extra_dirs[0] / "ARCHITECTURE.md").resolve())
    msa_path = str((settings.kb_dir / "msa.md").resolve())
    stores = SimpleNamespace(
        chunk_store=SimpleNamespace(
            vector_search=lambda *args, **kwargs: [],
            keyword_search=lambda *args, **kwargs: [],
        ),
        doc_store=SimpleNamespace(
            list_documents=lambda **kwargs: (
                [
                    SimpleNamespace(source_path=msa_path, title="msa.md"),
                    SimpleNamespace(source_path=architecture_path, title="ARCHITECTURE.md"),
                ]
                if kwargs.get("collection_id") == "default"
                else []
            ),
            fuzzy_search_title=lambda hint, tenant_id, limit=5, collection_id="": [],
        ),
    )
    providers = SimpleNamespace(
        judge=SimpleNamespace(invoke=lambda *args, **kwargs: SimpleNamespace(content='{"grades": []}')),
        chat=SimpleNamespace(invoke=lambda *args, **kwargs: SimpleNamespace(content="unused")),
    )
    session = SimpleNamespace(
        tenant_id="tenant-123",
        metadata={
            "collection_id": "owui-chat-1",
            "upload_collection_id": "owui-chat-1",
            "kb_collection_id": "default",
        },
    )

    contract = run_rag_contract(
        settings,
        stores,
        providers=providers,
        session=session,
        query="What are the key implementation details in the architecture docs? Cite your sources.",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=5,
        top_k_keyword=5,
        max_retries=1,
        callbacks=[],
    )

    assert "sync-kb" not in contract.answer
    assert "owui-chat-1" not in contract.answer
    assert "KB_COVERAGE_MISSING" not in contract.warnings


def test_get_collection_readiness_status_marks_indexed_host_path_collection_ready(tmp_path: Path):
    settings = _settings(tmp_path)
    stores = SimpleNamespace(
        doc_store=SimpleNamespace(
            list_documents=lambda **kwargs: (
                [
                    SimpleNamespace(
                        source_path="/corpora/rfp/docx/asterion_ecp_04_rev_c.docx",
                        source_type="host_path",
                        title="asterion_ecp_04_rev_c.docx",
                    )
                ]
                if kwargs.get("collection_id") == "rfp-corpus"
                else []
            )
        )
    )

    status = get_collection_readiness_status(
        settings,
        stores,
        tenant_id="tenant-123",
        collection_id="rfp-corpus",
    )

    assert status.collection_id == "rfp-corpus"
    assert status.maintenance_policy == "indexed_documents"
    assert status.ready is True
    assert status.reason == "indexed"
    assert status.suggested_fix == ""


def test_get_collection_readiness_status_marks_empty_indexed_collection_without_sync_guidance(tmp_path: Path):
    settings = _settings(tmp_path)
    stores = SimpleNamespace(
        collection_store=SimpleNamespace(
            get_collection=lambda collection_id, *, tenant_id: (
                SimpleNamespace(
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    maintenance_policy="indexed_documents",
                )
                if collection_id == "rfp-corpus"
                else None
            )
        ),
        doc_store=SimpleNamespace(list_documents=lambda **kwargs: []),
    )

    status = get_collection_readiness_status(
        settings,
        stores,
        tenant_id="tenant-123",
        collection_id="rfp-corpus",
    )

    assert status.collection_id == "rfp-corpus"
    assert status.maintenance_policy == "indexed_documents"
    assert status.ready is False
    assert status.reason == "empty_collection"
    assert "sync-kb" not in status.suggested_fix
    assert "add documents" in status.suggested_fix


def test_run_rag_contract_returns_upload_prompt_when_upload_only_scope_has_no_files(tmp_path: Path):
    settings = _settings(tmp_path)
    stores = SimpleNamespace(
        chunk_store=SimpleNamespace(
            vector_search=lambda *args, **kwargs: [],
            keyword_search=lambda *args, **kwargs: [],
        ),
        doc_store=SimpleNamespace(
            list_documents=lambda **kwargs: [],
            fuzzy_search_title=lambda hint, tenant_id, limit=5, collection_id="": [],
        ),
    )
    providers = SimpleNamespace(
        judge=SimpleNamespace(invoke=lambda *args, **kwargs: SimpleNamespace(content='{"grades": []}')),
        chat=SimpleNamespace(invoke=lambda *args, **kwargs: SimpleNamespace(content="unused")),
    )
    session = SimpleNamespace(
        tenant_id="tenant-123",
        metadata={"upload_collection_id": "owui-chat-1", "kb_collection_id": "default"},
        uploaded_doc_ids=[],
    )

    contract = run_rag_contract(
        settings,
        stores,
        providers=providers,
        session=session,
        query="Only use the uploaded file to answer this.",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=False,
        top_k_vector=5,
        top_k_keyword=5,
        max_retries=1,
        callbacks=[],
    )

    assert "There are no uploaded files attached to this chat yet." in contract.answer
    assert contract.warnings == ["NO_CHAT_UPLOADS_AVAILABLE"]


def test_run_rag_contract_returns_clarification_when_both_scopes_are_available(tmp_path: Path):
    settings = _settings(tmp_path)
    architecture_path = str((settings.kb_extra_dirs[0] / "ARCHITECTURE.md").resolve())
    msa_path = str((settings.kb_dir / "msa.md").resolve())
    stores = SimpleNamespace(
        chunk_store=SimpleNamespace(
            vector_search=lambda *args, **kwargs: [],
            keyword_search=lambda *args, **kwargs: [],
        ),
        doc_store=SimpleNamespace(
            list_documents=lambda **kwargs: (
                [
                    SimpleNamespace(source_path=msa_path, title="msa.md"),
                    SimpleNamespace(source_path=architecture_path, title="ARCHITECTURE.md"),
                ]
                if kwargs.get("collection_id") == "default"
                else []
            ),
            fuzzy_search_title=lambda hint, tenant_id, limit=5, collection_id="": [],
        ),
    )
    providers = SimpleNamespace(
        judge=SimpleNamespace(invoke=lambda *args, **kwargs: SimpleNamespace(content='{"grades": []}')),
        chat=SimpleNamespace(invoke=lambda *args, **kwargs: SimpleNamespace(content="unused")),
    )
    session = SimpleNamespace(
        tenant_id="tenant-123",
        metadata={
            "collection_id": "owui-chat-1",
            "upload_collection_id": "owui-chat-1",
            "kb_collection_id": "default",
        },
        uploaded_doc_ids=["doc-upload-1"],
    )

    contract = run_rag_contract(
        settings,
        stores,
        providers=providers,
        session=session,
        query="Explain the approval workflow and cite your sources.",
        conversation_context="",
        preferred_doc_ids=[],
        must_include_uploads=True,
        top_k_vector=5,
        top_k_keyword=5,
        max_retries=1,
        callbacks=[],
    )

    assert "Reply with one of:" in contract.answer
    assert contract.warnings == ["RETRIEVAL_SCOPE_AMBIGUOUS"]
