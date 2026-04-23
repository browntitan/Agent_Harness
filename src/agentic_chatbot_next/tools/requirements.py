from __future__ import annotations

import csv
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence

from langchain.tools import tool

from agentic_chatbot_next.rag.doc_targets import resolve_indexed_docs as resolve_named_indexed_docs
from agentic_chatbot_next.rag.ingest import resolve_record_source_identity, select_active_kb_record
from agentic_chatbot_next.rag.requirements import (
    MANDATORY_MODE,
    STRICT_SHALL_MODE,
    build_requirement_statement_records,
    format_requirement_location,
    normalize_requirement_mode,
    requirement_modalities_for_mode,
    supports_requirements_extraction,
)
from agentic_chatbot_next.rag.requirements_service import RequirementExtractionService
from agentic_chatbot_next.rag.retrieval_scope import (
    document_source_policy_requires_repository,
    has_upload_evidence,
    repository_upload_doc_ids,
    resolve_kb_collection_id,
    resolve_search_collection_ids,
    resolve_upload_collection_id,
)
from agentic_chatbot_next.runtime.artifacts import register_workspace_artifact


def _tenant_id(settings: object, session: object) -> str:
    return str(getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev")) or "local-dev")


def _source_scope_default(settings: object, session: object) -> str:
    del settings
    if has_upload_evidence(session):
        return "uploads"
    return "kb"


def _record_to_dict(record: Any) -> Dict[str, Any]:
    return {
        "doc_id": str(getattr(record, "doc_id", "") or ""),
        "title": str(getattr(record, "title", "") or ""),
        "source_type": str(getattr(record, "source_type", "") or ""),
        "source_path": str(getattr(record, "source_path", "") or ""),
        "collection_id": str(getattr(record, "collection_id", "") or ""),
        "file_type": str(getattr(record, "file_type", "") or ""),
        "doc_structure_type": str(getattr(record, "doc_structure_type", "") or ""),
    }


def _build_resolution_stores(records: Sequence[Any]) -> Any:
    record_map = {
        str(getattr(record, "doc_id", "") or ""): record
        for record in records
        if str(getattr(record, "doc_id", "") or "")
    }

    def _list_documents(tenant_id: str = "local-dev", collection_id: str = "", source_type: str = "") -> List[Any]:
        del tenant_id
        filtered = list(record_map.values())
        if collection_id:
            filtered = [
                record
                for record in filtered
                if str(getattr(record, "collection_id", "") or "") == collection_id
            ]
        if source_type:
            filtered = [
                record
                for record in filtered
                if str(getattr(record, "source_type", "") or "").strip().lower() == source_type
            ]
        return filtered

    def _fuzzy_search_title(hint: str, tenant_id: str, limit: int = 5, collection_id: str = "") -> List[Dict[str, Any]]:
        del tenant_id
        lowered = str(hint or "").strip().lower()
        candidates = []
        for record in _list_documents(collection_id=collection_id):
            title = str(getattr(record, "title", "") or "")
            source_path = str(getattr(record, "source_path", "") or "")
            if lowered and lowered not in title.lower() and lowered not in source_path.lower():
                continue
            candidates.append({"doc_id": record.doc_id, "title": title, "score": 0.5})
            if len(candidates) >= max(1, int(limit)):
                break
        return candidates

    return SimpleNamespace(
        doc_store=SimpleNamespace(
            list_documents=_list_documents,
            fuzzy_search_title=_fuzzy_search_title,
            get_document=lambda doc_id, tenant_id="local-dev": record_map.get(doc_id),
        )
    )


def _collapse_active_kb_records(settings: object, records: Sequence[Any]) -> List[Any]:
    grouped: dict[str, list[Any]] = {}
    configured_paths_by_title: dict[str, list[Path]] = {}
    kb_dir = getattr(settings, "kb_dir", None)
    extra_dirs = list(getattr(settings, "kb_extra_dirs", ()) or [])
    for raw_root in [kb_dir, *extra_dirs]:
        if not raw_root:
            continue
        root = Path(str(raw_root)).expanduser()
        if not root.exists() or not root.is_dir():
            continue
        for path in root.glob("*"):
            if path.is_file():
                configured_paths_by_title.setdefault(path.name.casefold(), []).append(path.resolve())
    for record in records:
        identity = resolve_record_source_identity(record, configured_paths_by_title=configured_paths_by_title)
        grouped.setdefault(identity, []).append(record)
    collapsed = [select_active_kb_record(group) for group in grouped.values()]
    return [record for record in collapsed if record is not None]


def _scoped_records(
    settings: object,
    stores: object,
    session: object,
    *,
    source_scope: str,
    collection_id: str = "",
) -> tuple[str, List[Any]]:
    tenant_id = _tenant_id(settings, session)
    normalized_scope = str(source_scope or "").strip().lower()
    if normalized_scope not in {"uploads", "kb"}:
        normalized_scope = _source_scope_default(settings, session)
    if normalized_scope == "uploads":
        effective_collection = str(collection_id or resolve_upload_collection_id(settings, session))
        records = list(
            stores.doc_store.list_documents(
                tenant_id=tenant_id,
                source_type="upload",
                collection_id=effective_collection,
            )
        )
        uploaded_doc_ids = set(repository_upload_doc_ids(session))
        if not uploaded_doc_ids and document_source_policy_requires_repository(session):
            records = []
        if uploaded_doc_ids:
            records = [
                record
                for record in records
                if str(getattr(record, "doc_id", "") or "") in uploaded_doc_ids
            ]
        return effective_collection, records

    available_collections = list(resolve_search_collection_ids(settings, session))
    effective_collection = str(collection_id or resolve_kb_collection_id(settings, session))
    if effective_collection and available_collections and effective_collection not in available_collections:
        available_collections = [effective_collection]
    elif effective_collection:
        available_collections = [effective_collection]
    records: List[Any] = []
    for scoped_collection in available_collections or [effective_collection]:
        records.extend(
            list(
                stores.doc_store.list_documents(
                    tenant_id=tenant_id,
                    source_type="kb",
                    collection_id=str(scoped_collection or effective_collection),
                )
            )
        )
    return effective_collection, _collapse_active_kb_records(settings, records)


def _select_target_records(
    settings: object,
    stores: object,
    session: object,
    *,
    source_scope: str,
    collection_id: str,
    document_names: Sequence[str],
    all_documents: bool,
) -> Dict[str, Any]:
    effective_collection, records = _scoped_records(
        settings,
        stores,
        session,
        source_scope=source_scope,
        collection_id=collection_id,
    )
    if not records:
        return {
            "collection_id": effective_collection,
            "records": [],
            "error": "No indexed documents are available in the requested scope.",
        }

    if document_names:
        resolution = resolve_named_indexed_docs(
            _build_resolution_stores(records),
            settings=settings,
            tenant_id=_tenant_id(settings, session),
            names=document_names,
            collection_ids=[effective_collection] if effective_collection else [],
        )
        if resolution.ambiguous:
            return {
                "collection_id": effective_collection,
                "records": [],
                "error": "Multiple documents matched the requested names.",
                "ambiguous_documents": [item.to_dict() for item in resolution.ambiguous],
                "missing_documents": [item.to_dict() for item in resolution.missing],
            }
        if resolution.missing:
            return {
                "collection_id": effective_collection,
                "records": [],
                "error": "One or more requested documents could not be resolved.",
                "missing_documents": [item.to_dict() for item in resolution.missing],
            }
        resolved_ids = set(resolution.resolved_doc_ids)
        selected = [
            record
            for record in records
            if str(getattr(record, "doc_id", "") or "") in resolved_ids
        ]
        return {
            "collection_id": effective_collection,
            "records": selected,
            "resolved_documents": [item.to_dict() for item in resolution.resolved],
        }

    if all_documents:
        return {"collection_id": effective_collection, "records": list(records)}

    if len(records) == 1:
        return {"collection_id": effective_collection, "records": list(records)}

    supported_records = [
        record
        for record in records
        if supports_requirements_extraction(str(getattr(record, "file_type", "") or ""))
    ]
    if len(supported_records) == 1:
        return {"collection_id": effective_collection, "records": supported_records}

    return {
        "collection_id": effective_collection,
        "records": [],
        "error": "Multiple documents are available in this scope. Specify one or set all_documents=true.",
        "candidate_documents": [_record_to_dict(record) for record in records[:10]],
    }


def _ensure_requirement_inventory(stores: object, record: Any, *, tenant_id: str) -> None:
    requirement_store = getattr(stores, "requirement_store", None)
    if requirement_store is None:
        return
    doc_id = str(getattr(record, "doc_id", "") or "")
    if not doc_id or requirement_store.has_doc_statements(doc_id, tenant_id):
        return
    chunk_records = list(stores.chunk_store.list_document_chunks(doc_id, tenant_id))
    statements = build_requirement_statement_records(
        SimpleNamespace(
            doc_id=doc_id,
            tenant_id=str(getattr(record, "tenant_id", "") or tenant_id),
            collection_id=str(getattr(record, "collection_id", "") or "default"),
            title=str(getattr(record, "title", "") or ""),
            source_type=str(getattr(record, "source_type", "") or ""),
            file_type=str(getattr(record, "file_type", "") or ""),
        ),
        chunk_records,
    )
    requirement_store.replace_doc_statements(doc_id, tenant_id, statements=statements)


def _preview_rows(rows: Sequence[Any], limit: int) -> List[Dict[str, Any]]:
    preview: List[Dict[str, Any]] = []
    for record in list(rows)[: max(1, int(limit))]:
        preview.append(
            {
                "document_title": str(getattr(record, "document_title", "") or ""),
                "modality": str(getattr(record, "modality", "") or ""),
                "location": format_requirement_location(record),
                "statement_text": str(getattr(record, "statement_text", "") or ""),
            }
        )
    return preview


def _summary_text(*, statement_count: int, document_count: int, source_scope: str, collection_id: str, mode: str) -> str:
    extractor_label = "shall statements" if mode == STRICT_SHALL_MODE else "requirement statements"
    if document_count == 1:
        return f"Extracted {statement_count} {extractor_label} from 1 document."
    if source_scope == "kb":
        return (
            f"Extracted {statement_count} {extractor_label} across {document_count} documents "
            f"in collection '{collection_id}'."
        )
    return f"Extracted {statement_count} {extractor_label} across {document_count} uploaded documents."


def _requirements_csv_filename(records: Sequence[Any], *, collection_id: str, all_documents: bool) -> str:
    if len(records) == 1 and not all_documents:
        title = str(getattr(records[0], "title", "") or getattr(records[0], "doc_id", "") or "requirements")
        stem = Path(title).stem or "requirements"
        return f"{stem}__requirement_statements.csv"
    stem = str(collection_id or "requirements_corpus").strip() or "requirements_corpus"
    stem = stem.replace("/", "_").replace("\\", "_")
    return f"{stem}__requirement_statements.csv"


def _write_requirements_csv(session: object, *, filename: str, rows: Sequence[Any]) -> Dict[str, Any]:
    workspace = getattr(session, "workspace", None)
    if workspace is None:
        raise ValueError("No session workspace is available.")
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "requirement_id",
            "document_title",
            "modality",
            "location",
            "statement_text",
            "doc_id",
            "collection_id",
            "source_type",
            "chunk_id",
            "chunk_index",
            "page_number",
            "clause_number",
            "section_title",
            "char_start",
            "char_end",
            "multi_requirement",
        ],
    )
    writer.writeheader()
    for record in rows:
        writer.writerow(
            {
                "requirement_id": str(getattr(record, "requirement_id", "") or ""),
                "document_title": str(getattr(record, "document_title", "") or ""),
                "modality": str(getattr(record, "modality", "") or ""),
                "location": format_requirement_location(record),
                "statement_text": str(getattr(record, "statement_text", "") or ""),
                "doc_id": str(getattr(record, "doc_id", "") or ""),
                "collection_id": str(getattr(record, "collection_id", "") or ""),
                "source_type": str(getattr(record, "source_type", "") or ""),
                "chunk_id": str(getattr(record, "chunk_id", "") or ""),
                "chunk_index": int(getattr(record, "chunk_index", 0) or 0),
                "page_number": getattr(record, "page_number", None),
                "clause_number": str(getattr(record, "clause_number", "") or ""),
                "section_title": str(getattr(record, "section_title", "") or ""),
                "char_start": int(getattr(record, "char_start", 0) or 0),
                "char_end": int(getattr(record, "char_end", 0) or 0),
                "multi_requirement": bool(getattr(record, "multi_requirement", False)),
            }
        )
    workspace.write_text(filename, buffer.getvalue())
    return register_workspace_artifact(session, filename=filename, label=filename)


def make_requirement_tools(settings: object, stores: object, session: object) -> List[Any]:
    def _extract(
        *,
        source_scope: str = "auto",
        collection_id: str = "",
        document_names: Sequence[str] | None = None,
        all_documents: bool = False,
        mode: str = MANDATORY_MODE,
        max_preview_rows: int = 8,
        export: bool = False,
        filename: str = "",
    ) -> Dict[str, Any]:
        return RequirementExtractionService(settings, stores, session).extract(
            source_scope=source_scope,
            collection_id=collection_id,
            document_names=[str(item) for item in (document_names or []) if str(item)],
            all_documents=bool(all_documents),
            mode=mode,
            max_preview_rows=max_preview_rows,
            export=export,
            filename=filename,
        )

    @tool
    def extract_requirement_statements(
        source_scope: str = "auto",
        collection_id: str = "",
        document_names: List[str] | None = None,
        all_documents: bool = False,
        mode: str = MANDATORY_MODE,
        max_preview_rows: int = 8,
    ) -> Dict[str, Any]:
        """Extract persisted requirement statements from uploads or one KB collection."""

        return _extract(
            source_scope=source_scope,
            collection_id=collection_id,
            document_names=document_names,
            all_documents=all_documents,
            mode=mode,
            max_preview_rows=max_preview_rows,
        )

    @tool
    def export_requirement_statements(
        source_scope: str = "auto",
        collection_id: str = "",
        document_names: List[str] | None = None,
        all_documents: bool = False,
        mode: str = MANDATORY_MODE,
        filename: str = "",
        max_preview_rows: int = 8,
    ) -> Dict[str, Any]:
        """Export persisted requirement statements as a downloadable CSV artifact."""

        return _extract(
            source_scope=source_scope,
            collection_id=collection_id,
            document_names=document_names,
            all_documents=all_documents,
            mode=mode,
            max_preview_rows=max_preview_rows,
            export=True,
            filename=filename,
        )

    return [extract_requirement_statements, export_requirement_statements]


__all__ = ["make_requirement_tools"]
