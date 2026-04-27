from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agentic_chatbot_next.rag.retrieval_scope import (
    decide_retrieval_scope,
    has_upload_evidence,
    merge_scope_metadata,
    resolve_available_kb_collection_ids,
    resolve_collection_ids_for_source,
    resolve_kb_collection_confirmed,
    resolve_search_collection_ids,
)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(default_collection_id="default", kb_dir="", kb_extra_dirs=())


def _session(*, metadata: dict[str, object] | None = None, uploaded_doc_ids: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        metadata=dict(metadata or {}),
        uploaded_doc_ids=list(uploaded_doc_ids or []),
    )


def test_merge_scope_metadata_keeps_legacy_single_collection_behavior() -> None:
    merged = merge_scope_metadata(_settings(), {"collection_id": "defense-rag-test"})

    assert merged["collection_id"] == "defense-rag-test"
    assert merged["upload_collection_id"] == "defense-rag-test"
    assert merged["kb_collection_id"] == "defense-rag-test"
    assert merged["available_kb_collection_ids"] == ["defense-rag-test"]
    assert merged["kb_collection_confirmed"] is True


def test_merge_scope_metadata_keeps_bootstrap_kb_collection_unconfirmed_when_requested() -> None:
    merged = merge_scope_metadata(
        _settings(),
        {
            "collection_id": "owui-chat-1",
            "upload_collection_id": "owui-chat-1",
            "kb_collection_id": "default",
            "kb_collection_confirmed": False,
        },
    )

    assert merged["collection_id"] == "owui-chat-1"
    assert merged["upload_collection_id"] == "owui-chat-1"
    assert merged["kb_collection_id"] == "default"
    assert merged["available_kb_collection_ids"] == ["default"]
    assert merged["kb_collection_confirmed"] is False


def test_decide_retrieval_scope_detects_explicit_uploads_only() -> None:
    decision = decide_retrieval_scope(
        _settings(),
        _session(metadata={"upload_collection_id": "owui-chat-1"}, uploaded_doc_ids=["doc-upload-1"]),
        query="Only use the uploaded file for this answer.",
        has_uploads=True,
        kb_available=True,
    )

    assert decision.mode == "uploads_only"


def test_decide_retrieval_scope_detects_explicit_kb_only() -> None:
    decision = decide_retrieval_scope(
        _settings(),
        _session(metadata={"kb_collection_id": "default"}),
        query="Search the knowledge base for architecture docs.",
        has_uploads=False,
        kb_available=True,
    )

    assert decision.mode == "kb_only"


def test_decide_retrieval_scope_keeps_kb_policy_question_in_kb_scope_with_upload_terms() -> None:
    decision = decide_retrieval_scope(
        _settings(),
        _session(
            metadata={
                "upload_collection_id": "capability-benchmark-uploads",
                "kb_collection_id": "default",
                "requested_kb_collection_id": "default",
                "kb_collection_confirmed": True,
                "search_collection_ids": ["default"],
            }
        ),
        query="Search the default knowledge base for this upload lifecycle policy: What happens when an uploaded document is deleted?",
        has_uploads=False,
        kb_available=True,
    )

    assert decision.mode == "kb_only"
    assert decision.search_collection_ids == ("default",)


def test_decide_retrieval_scope_current_uploaded_document_still_uses_upload_scope() -> None:
    decision = decide_retrieval_scope(
        _settings(),
        _session(metadata={"upload_collection_id": "owui-chat-1"}, uploaded_doc_ids=["doc-upload-1"]),
        query="Summarize the attached document.",
        has_uploads=True,
        kb_available=True,
    )

    assert decision.mode == "uploads_only"


def test_decide_retrieval_scope_treats_kb_inventory_queries_as_kb_only_even_with_uploads() -> None:
    decision = decide_retrieval_scope(
        _settings(),
        _session(
            metadata={"upload_collection_id": "owui-chat-1", "kb_collection_id": "default"},
            uploaded_doc_ids=["doc-upload-1"],
        ),
        query="What docs are in the knowledge base?",
        has_uploads=True,
        kb_available=True,
    )

    assert decision.mode == "kb_only"


def test_decide_retrieval_scope_detects_explicit_both() -> None:
    decision = decide_retrieval_scope(
        _settings(),
        _session(
            metadata={"upload_collection_id": "owui-chat-1", "kb_collection_id": "default"},
            uploaded_doc_ids=["doc-upload-1"],
        ),
        query="Use the uploaded file and the docs to answer this.",
        has_uploads=True,
        kb_available=True,
    )

    assert decision.mode == "both"
    assert decision.search_collection_ids == ("default", "owui-chat-1")


def test_decide_retrieval_scope_detects_explicit_none() -> None:
    decision = decide_retrieval_scope(
        _settings(),
        _session(),
        query="Don't look anything up; just rewrite this.",
        has_uploads=False,
        kb_available=True,
    )

    assert decision.mode == "none"


def test_decide_retrieval_scope_marks_grounded_multi_source_queries_as_ambiguous() -> None:
    decision = decide_retrieval_scope(
        _settings(),
        _session(
            metadata={"upload_collection_id": "owui-chat-1", "kb_collection_id": "default"},
            uploaded_doc_ids=["doc-upload-1"],
        ),
        query="Explain the approval workflow and cite your sources.",
        has_uploads=True,
        kb_available=True,
    )

    assert decision.mode == "ambiguous"


def test_decide_retrieval_scope_prefers_kb_when_query_names_configured_repo_doc(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "ARCHITECTURE.md").write_text("# Architecture\n", encoding="utf-8")
    settings = SimpleNamespace(default_collection_id="default", kb_dir="", kb_extra_dirs=(str(docs_dir),))

    decision = decide_retrieval_scope(
        settings,
        _session(
            metadata={"upload_collection_id": "owui-chat-1", "kb_collection_id": "default"},
            uploaded_doc_ids=["doc-upload-1"],
        ),
        query="Summarize the main components of the runtime service described in ARCHITECTURE.md.",
        has_uploads=True,
        kb_available=True,
    )

    assert decision.mode == "kb_only"
    assert decision.reason == "explicit_named_kb_doc"
    assert decision.search_collection_ids == ("default",)


def test_decide_retrieval_scope_keeps_explicit_upload_preference_even_when_kb_doc_is_named(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "ARCHITECTURE.md").write_text("# Architecture\n", encoding="utf-8")
    settings = SimpleNamespace(default_collection_id="default", kb_dir="", kb_extra_dirs=(str(docs_dir),))

    decision = decide_retrieval_scope(
        settings,
        _session(
            metadata={"upload_collection_id": "owui-chat-1", "kb_collection_id": "default"},
            uploaded_doc_ids=["doc-upload-1"],
        ),
        query="Only use the uploaded file, not the KB copy of ARCHITECTURE.md.",
        has_uploads=True,
        kb_available=True,
    )

    assert decision.mode == "uploads_only"
    assert decision.reason == "explicit_uploads_only"


def test_resolve_search_collection_ids_prefers_explicit_runtime_scope() -> None:
    session = _session(
        metadata={
            "search_collection_ids": ["default", "owui-chat-1", "default"],
            "upload_collection_id": "owui-chat-1",
            "kb_collection_id": "default",
        }
    )

    assert resolve_search_collection_ids(_settings(), session) == ("default", "owui-chat-1")


def test_resolve_collection_ids_for_source_keeps_kb_separate_from_openwebui_upload_scope() -> None:
    session = _session(
        metadata={
            "collection_id": "owui-chat-1",
            "upload_collection_id": "owui-chat-1",
            "kb_collection_id": "default",
            "available_kb_collection_ids": ["default"],
        }
    )

    assert resolve_collection_ids_for_source(_settings(), session, source_type="kb") == ("default",)
    assert resolve_collection_ids_for_source(_settings(), session, source_type="upload") == ("owui-chat-1",)


def test_openwebui_thin_mode_requires_internal_upload_doc_ids_for_upload_evidence() -> None:
    session = _session(
        metadata={
            "collection_id": "owui-chat-1",
            "upload_collection_id": "owui-chat-1",
            "kb_collection_id": "default",
            "source_upload_ids": ["owui-file-1"],
            "openwebui_thin_mode": True,
            "document_source_policy": "agent_repository_only",
        }
    )

    assert has_upload_evidence(session) is False
    decision = decide_retrieval_scope(
        _settings(),
        session,
        query="Summarize the uploaded document.",
        kb_available=False,
    )
    assert decision.mode == "uploads_only"
    assert decision.has_uploads is False
    assert decision.search_collection_ids == ()


def test_resolve_collection_ids_for_source_filters_session_upload_when_authz_is_enabled() -> None:
    session = _session(
        metadata={
            "upload_collection_id": "owui-chat-1",
            "access_summary": {
                "authz_enabled": True,
                "session_upload_collection_id": "owui-chat-1",
                "resources": {"collection": {"use": ["default", "rfp-corpus"]}},
            },
        }
    )

    assert resolve_available_kb_collection_ids(_settings(), session) == (
        "default",
        "rfp-corpus",
    )
    assert resolve_collection_ids_for_source(_settings(), session, source_type="kb") == (
        "default",
        "rfp-corpus",
    )
    assert resolve_collection_ids_for_source(_settings(), session, source_type="all") == (
        "default",
        "rfp-corpus",
        "owui-chat-1",
    )


def test_merge_scope_metadata_keeps_authz_available_kb_ids_kb_only() -> None:
    merged = merge_scope_metadata(
        _settings(),
        {
            "collection_id": "owui-chat-1",
            "upload_collection_id": "owui-chat-1",
            "kb_collection_id": "default",
            "access_summary": {
                "authz_enabled": True,
                "session_upload_collection_id": "owui-chat-1",
                "resources": {"collection": {"use": ["default"]}},
            },
        },
    )

    assert merged["upload_collection_id"] == "owui-chat-1"
    assert merged["kb_collection_id"] == "default"
    assert merged["available_kb_collection_ids"] == ["default"]


def test_resolve_available_kb_collection_ids_and_confirmation_from_session_metadata() -> None:
    session = _session(
        metadata={
            "kb_collection_id": "default",
            "available_kb_collection_ids": ["default", "rfp-corpus", "default"],
            "kb_collection_confirmed": True,
        }
    )

    assert resolve_available_kb_collection_ids(_settings(), session) == ("default", "rfp-corpus")
    assert resolve_kb_collection_confirmed(session) is True
