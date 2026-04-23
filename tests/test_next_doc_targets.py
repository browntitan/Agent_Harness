from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agentic_chatbot_next.rag.doc_targets import extract_named_document_targets, resolve_indexed_docs
from agentic_chatbot_next.rag.hints import normalize_structured_query


def _record(
    doc_id: str,
    title: str,
    source_path: str,
    *,
    collection_id: str = "default",
    content_hash: str = "",
    ingested_at: str = "",
    source_identity: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        doc_id=doc_id,
        title=title,
        source_type="kb",
        source_path=source_path,
        source_identity=source_identity,
        collection_id=collection_id,
        content_hash=content_hash,
        ingested_at=ingested_at,
        file_type="md",
        doc_structure_type="general",
        num_chunks=4,
    )


def _stores(records: list[SimpleNamespace]):
    record_map = {item.doc_id: item for item in records}
    return SimpleNamespace(
        doc_store=SimpleNamespace(
            list_documents=lambda tenant_id="tenant", collection_id="": [
                item for item in records if not collection_id or item.collection_id == collection_id
            ],
            fuzzy_search_title=lambda hint, tenant_id, limit=5, collection_id="": [
                {"doc_id": item.doc_id, "title": item.title, "score": 0.42}
                for item in records
                if hint.lower() in item.title.lower()
            ][:limit],
            get_document=lambda doc_id, tenant_id="tenant": record_map.get(doc_id),
        )
    )


def test_extract_named_document_targets_finds_file_names_and_paths() -> None:
    names = extract_named_document_targets(
        'Compare ARCHITECTURE.md and "docs/C4_ARCHITECTURE.md" in detail.'
    )

    assert names == ["ARCHITECTURE.md", "docs/C4_ARCHITECTURE.md"]


def test_extract_named_document_targets_ignores_bare_numeric_decimals() -> None:
    names = extract_named_document_targets(
        "Calculate 18% of 4.2 million and, separately, search indexed docs for rate limit policy."
    )

    assert names == []


def test_extract_named_document_targets_ignores_openwebui_wrapper_noise() -> None:
    names = extract_named_document_targets(
        (
            "If you don't know the answer, say so. "
            "Don't present information that's not present in the context. "
            "Use e.g. citations only when present. "
            "extract all requirements from the uploaded document"
        )
    )

    assert names == []


def test_normalize_structured_query_unwraps_openwebui_rag_task_prompt() -> None:
    wrapped = """### Task:
Respond to the user query using the provided context.

### Guidelines:
- If you don't know the answer, say so.
- Don't present information that's not present in the context.

<context>
Requirement identifier examples: RC-SYS-3.2 and e.g. citation markers.
</context>

### Output:
Provide a clear and direct response.
extract all requirements/ shall statements from the uploaded document
"""

    assert normalize_structured_query(wrapped) == "extract all requirements/ shall statements from the uploaded document"


def test_resolve_indexed_docs_matches_exact_titles_and_basenames() -> None:
    stores = _stores(
        [
            _record("doc-arch", "ARCHITECTURE.md", "/repo/docs/ARCHITECTURE.md"),
            _record("doc-c4", "C4 overview", "/repo/docs/C4_ARCHITECTURE.md"),
        ]
    )

    resolution = resolve_indexed_docs(
        stores,
        tenant_id="tenant",
        names=["ARCHITECTURE.md", "docs/C4_ARCHITECTURE.md"],
        collection_ids=["default"],
    )

    assert resolution.requested_names == ("ARCHITECTURE.md", "docs/C4_ARCHITECTURE.md")
    assert resolution.resolved_doc_ids == ["doc-arch", "doc-c4"]
    assert resolution.missing == ()
    assert resolution.ambiguous == ()


def test_resolve_indexed_docs_reports_ambiguous_and_missing_names() -> None:
    stores = _stores(
        [
            _record("doc-a", "Overview.md", "/repo/docs/Overview.md"),
            _record("doc-b", "Overview.md", "/repo/archive/Overview.md"),
            _record("doc-c", "Architecture Notes.md", "/repo/docs/Architecture Notes.md"),
        ]
    )

    resolution = resolve_indexed_docs(
        stores,
        tenant_id="tenant",
        names=["Overview.md", "C4_ARCHITECTURE.md"],
        collection_ids=["default"],
    )

    assert resolution.resolved == ()
    assert resolution.ambiguous[0].requested_name == "Overview.md"
    assert {item.doc_id for item in resolution.ambiguous[0].candidates} == {"doc-a", "doc-b"}
    assert resolution.missing[0].requested_name == "C4_ARCHITECTURE.md"


def test_resolve_indexed_docs_collapses_same_source_duplicates_to_latest_active() -> None:
    stores = _stores(
        [
            _record(
                "doc-old",
                "ARCHITECTURE.md",
                "/repo/docs/ARCHITECTURE.md",
                content_hash="hash-old",
                ingested_at="2026-04-09T02:00:00Z",
            ),
            _record(
                "doc-new",
                "ARCHITECTURE.md",
                "/repo/docs/ARCHITECTURE.md",
                content_hash="hash-new",
                ingested_at="2026-04-09T03:00:00Z",
            ),
        ]
    )

    resolution = resolve_indexed_docs(
        stores,
        tenant_id="tenant",
        names=["ARCHITECTURE.md"],
        collection_ids=["default"],
    )

    assert resolution.ambiguous == ()
    assert resolution.missing == ()
    assert resolution.resolved_doc_ids == ["doc-new"]
    assert resolution.resolved[0].ignored_duplicate_doc_ids == ("doc-old",)


def test_resolve_indexed_docs_collapses_legacy_raw_path_identity_into_canonical_kb_source() -> None:
    stores = _stores(
        [
            _record(
                "doc-legacy",
                "ARCHITECTURE.md",
                "/repo/docs/ARCHITECTURE.md",
                content_hash="hash-old",
                ingested_at="2026-04-09T02:00:00Z",
                source_identity="/app/docs/ARCHITECTURE.md",
            ),
            _record(
                "doc-canonical",
                "ARCHITECTURE.md",
                "/repo/docs/ARCHITECTURE.md",
                content_hash="hash-new",
                ingested_at="2026-04-09T03:00:00Z",
                source_identity="path:repo://docs/ARCHITECTURE.md",
            ),
        ]
    )

    settings = SimpleNamespace(
        kb_dir="missing",
        kb_extra_dirs=(
            str(Path(__file__).resolve().parents[1] / "docs"),
        ),
    )

    resolution = resolve_indexed_docs(
        stores,
        settings=settings,
        tenant_id="tenant",
        names=["ARCHITECTURE.md"],
        collection_ids=["default"],
    )

    assert resolution.ambiguous == ()
    assert resolution.missing == ()
    assert resolution.resolved_doc_ids == ["doc-canonical"]
    assert resolution.resolved[0].ignored_duplicate_doc_ids == ("doc-legacy",)
