from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence

from agentic_chatbot_next.rag.doc_targets import (
    extract_named_document_targets,
    resolve_indexed_docs as resolve_named_indexed_docs,
)
from agentic_chatbot_next.rag.hints import normalize_structured_query
from agentic_chatbot_next.rag.ingest import resolve_record_source_identity, select_active_kb_record
from agentic_chatbot_next.rag.requirements import (
    LEGAL_CLAUSE_MODE,
    MANDATORY_MODE,
    STRICT_SHALL_MODE,
    build_requirement_statement_records,
    format_requirement_location,
    normalize_requirement_mode,
    normalize_requirement_text,
    requirement_modalities_for_mode,
    supports_requirements_extraction,
)
from agentic_chatbot_next.rag.retrieval_scope import (
    document_source_policy_requires_repository,
    has_upload_evidence,
    repository_upload_doc_ids,
    resolve_kb_collection_id,
    resolve_search_collection_ids,
    resolve_upload_collection_id,
)
from agentic_chatbot_next.runtime.artifacts import register_workspace_artifact


REQUIREMENTS_WORKFLOW_KIND = "requirements_extraction"

_REQUIREMENTS_EXTRACTION_RE = re.compile(
    r"\b(?:extract|pull|list|inventory|organize|find|harvest|export|download|return)\b"
    r".*\b(?:shall|must|requirement|requirements|obligation|obligations|clause|clauses)\b"
    r"|\b(?:shall|must)\s+statements?\b"
    r"|\brequirement\s+statements?\b"
    r"|\b(?:far|dfars)\b.*\b(?:clause|clauses|requirement|requirements|obligation|obligations)\b",
    re.IGNORECASE | re.DOTALL,
)
_STRICT_SHALL_REQUEST_RE = re.compile(r"\bshall\s+statements?\b|\bstrict\s+shall\b", re.IGNORECASE)
_LEGAL_REQUEST_RE = re.compile(
    r"\b(?:far|dfars|cfr|clause|clauses|legal|regulatory|obligation|obligations|"
    r"flow[-\s]?down|contractor\s+shall|offeror\s+shall)\b",
    re.IGNORECASE,
)
_ALL_DOCUMENTS_RE = re.compile(
    r"\b(?:all|every)\s+(?:documents?|docs?|files?)\b|\b(?:entire|whole)\s+(?:corpus|collection)\b|\bacross\s+(?:the\s+)?(?:corpus|documents?|docs?|files?)\b",
    re.IGNORECASE,
)
_FAR_DFARS_CLAUSE_RE = re.compile(r"\b(?:52|252)\.\d{3}-\d+\b|\b(?:FAR|DFARS)\b", re.IGNORECASE)
_PARAGRAPH_PATH_RE = re.compile(r"^\s*((?:\([a-z0-9ivxlcdm]+\))+)", re.IGNORECASE)
_ACTOR_RE = re.compile(
    r"\b(?:the\s+)?(?P<actor>"
    r"contractor|subcontractor|offeror|supplier|vendor|provider|government|agency|"
    r"system|subsystem|platform|service|operator|administrator|user"
    r")\b\s+(?:shall|must|will|is\s+required|are\s+required|is\s+responsible|are\s+responsible|agrees?)\b",
    re.IGNORECASE,
)


def is_requirements_extraction_request(user_text: str) -> bool:
    text = normalize_structured_query(str(user_text or "")) or str(user_text or "")
    return bool(_REQUIREMENTS_EXTRACTION_RE.search(text))


def infer_requirement_mode(user_text: str) -> str:
    text = str(user_text or "")
    if _LEGAL_REQUEST_RE.search(text):
        return LEGAL_CLAUSE_MODE
    if _STRICT_SHALL_REQUEST_RE.search(text):
        return STRICT_SHALL_MODE
    return MANDATORY_MODE


def infer_all_documents(user_text: str) -> bool:
    return bool(_ALL_DOCUMENTS_RE.search(str(user_text or "")))


def _tenant_id(settings: object, session: object) -> str:
    return str(getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev")) or "local-dev")


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


def _upload_filter_doc_ids(session: object, records: Sequence[Any]) -> set[str]:
    del records
    direct_ids = set(repository_upload_doc_ids(session))
    if direct_ids:
        return direct_ids
    if document_source_policy_requires_repository(session):
        return {"__agent_repository_upload_missing__"}
    return set()


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
        normalized_scope = "uploads" if has_upload_evidence(session) else "kb"
    if normalized_scope == "uploads":
        effective_collection = str(collection_id or resolve_upload_collection_id(settings, session))
        records = list(
            stores.doc_store.list_documents(
                tenant_id=tenant_id,
                source_type="upload",
                collection_id=effective_collection,
            )
        )
        filter_doc_ids = _upload_filter_doc_ids(session, records)
        if filter_doc_ids:
            records = [
                record
                for record in records
                if str(getattr(record, "doc_id", "") or "") in filter_doc_ids
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
        "error": "Multiple documents are available in this scope. Specify one document or ask for all documents.",
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
        mode=MANDATORY_MODE,
    )
    requirement_store.replace_doc_statements(doc_id, tenant_id, statements=statements)


def _build_ad_hoc_records(stores: object, record: Any, *, tenant_id: str, mode: str) -> List[Any]:
    doc_id = str(getattr(record, "doc_id", "") or "")
    chunks = list(stores.chunk_store.list_document_chunks(doc_id, tenant_id)) if doc_id else []
    return build_requirement_statement_records(
        SimpleNamespace(
            doc_id=doc_id,
            tenant_id=str(getattr(record, "tenant_id", "") or tenant_id),
            collection_id=str(getattr(record, "collection_id", "") or "default"),
            title=str(getattr(record, "title", "") or ""),
            source_type=str(getattr(record, "source_type", "") or ""),
            file_type=str(getattr(record, "file_type", "") or ""),
        ),
        chunks,
        mode=mode,
    )


def _infer_actor(text: str) -> str:
    match = _ACTOR_RE.search(str(text or ""))
    if not match:
        return ""
    actor = str(match.group("actor") or "").strip()
    return actor[:1].upper() + actor[1:] if actor else ""


def _infer_paragraph_path(text: str) -> str:
    match = _PARAGRAPH_PATH_RE.search(str(text or ""))
    return str(match.group(1) or "").strip() if match else ""


def _confidence_for_row(row: Any, *, mode: str, source_span_valid: bool) -> float:
    if not source_span_valid:
        return 0.35
    modality = str(getattr(row, "modality", "") or "")
    if mode == LEGAL_CLAUSE_MODE and modality in {"will", "agrees_to", "responsible_for", "required"}:
        return 0.78
    if modality in {"shall", "shall_not", "must", "must_not", "required_to", "prohibited"}:
        return 0.92
    return 0.70


def _row_to_output(row: Any, *, mode: str, chunk_text_by_id: Mapping[str, str]) -> Dict[str, Any]:
    statement_text = normalize_requirement_text(str(getattr(row, "statement_text", "") or ""))
    chunk_id = str(getattr(row, "chunk_id", "") or "")
    source_text = str(chunk_text_by_id.get(chunk_id) or "")
    source_span_valid = bool(statement_text and (not source_text or statement_text in normalize_requirement_text(source_text)))
    warnings: list[str] = []
    if bool(getattr(row, "multi_requirement", False)):
        warnings.append("sentence_contains_multiple_mandatory_operators")
    if not source_span_valid:
        warnings.append("source_span_not_verified")
    return {
        "document_title": str(getattr(row, "document_title", "") or ""),
        "modality": str(getattr(row, "modality", "") or ""),
        "location": format_requirement_location(row),
        "statement_text": statement_text,
        "requirement_id": str(getattr(row, "requirement_id", "") or ""),
        "doc_id": str(getattr(row, "doc_id", "") or ""),
        "collection_id": str(getattr(row, "collection_id", "") or ""),
        "source_type": str(getattr(row, "source_type", "") or ""),
        "section": str(getattr(row, "section_title", "") or ""),
        "clause_number": str(getattr(row, "clause_number", "") or ""),
        "paragraph_path": _infer_paragraph_path(statement_text),
        "actor": _infer_actor(statement_text),
        "requirement_text": statement_text,
        "source_excerpt": statement_text,
        "source_location": format_requirement_location(row),
        "extraction_method": str(getattr(row, "extractor_version", "") or "requirements_v1"),
        "extractor_mode": str(getattr(row, "extractor_mode", "") or mode),
        "confidence": _confidence_for_row(row, mode=mode, source_span_valid=source_span_valid),
        "source_span_valid": source_span_valid,
        "chunk_id": chunk_id,
        "chunk_index": int(getattr(row, "chunk_index", 0) or 0),
        "page_number": getattr(row, "page_number", None),
        "char_start": int(getattr(row, "char_start", 0) or 0),
        "char_end": int(getattr(row, "char_end", 0) or 0),
        "warnings": warnings,
    }


def _profile_document(record: Any, chunks: Sequence[Any], *, requested_mode: str) -> Dict[str, Any]:
    sample = "\n".join(str(getattr(chunk, "content", "") or "") for chunk in list(chunks)[:8])
    title = str(getattr(record, "title", "") or "")
    haystack = f"{title}\n{sample}"
    legal_like = bool(_FAR_DFARS_CLAUSE_RE.search(haystack)) or bool(
        re.search(r"\bcontractor\s+shall\b|\bofferor\s+shall\b|\bas\s+prescribed\s+in\b", haystack, re.IGNORECASE)
    )
    parser_strategy = "legal_clause" if requested_mode == LEGAL_CLAUSE_MODE or legal_like else "cached_statement_inventory"
    return {
        "doc_id": str(getattr(record, "doc_id", "") or ""),
        "title": title,
        "file_type": str(getattr(record, "file_type", "") or ""),
        "doc_structure_type": str(getattr(record, "doc_structure_type", "") or ""),
        "legal_clause_like": legal_like,
        "parser_strategy": parser_strategy,
        "chunk_count": len(chunks),
    }


def _safe_stem(value: str) -> str:
    stem = Path(str(value or "requirements")).stem or "requirements"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem[:80] or "requirements"


def _requirements_filename(records: Sequence[Any], *, collection_id: str, all_documents: bool, suffix: str) -> str:
    if len(records) == 1 and not all_documents:
        stem = _safe_stem(str(getattr(records[0], "title", "") or getattr(records[0], "doc_id", "") or "requirements"))
    else:
        stem = _safe_stem(str(collection_id or "requirements_corpus"))
    return f"{stem}__requirement_statements.{suffix}"


_EXPORT_COLUMNS = [
    "document_title",
    "modality",
    "location",
    "statement_text",
    "requirement_id",
    "doc_id",
    "collection_id",
    "source_type",
    "section",
    "clause_number",
    "paragraph_path",
    "actor",
    "requirement_text",
    "source_excerpt",
    "source_location",
    "extraction_method",
    "extractor_mode",
    "confidence",
    "source_span_valid",
    "chunk_id",
    "chunk_index",
    "page_number",
    "char_start",
    "char_end",
    "warnings",
]


def _write_csv(session: object, *, filename: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    workspace = getattr(session, "workspace", None)
    if workspace is None:
        workspace_root = str(getattr(session, "workspace_root", "") or "").strip()
        session_id = str(getattr(session, "session_id", "") or "")
        if workspace_root and session_id:
            from agentic_chatbot_next.sandbox.workspace import SessionWorkspace

            workspace = SessionWorkspace(session_id=session_id, root=Path(workspace_root))
            workspace.open()
            session.workspace = workspace
        else:
            raise ValueError("No session workspace is available.")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_EXPORT_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        payload = dict(row)
        payload["warnings"] = "; ".join(str(item) for item in list(payload.get("warnings") or []) if str(item))
        writer.writerow(payload)
    workspace.write_text(filename, buffer.getvalue())
    return register_workspace_artifact(session, filename=filename, label=filename)


def _write_jsonl(session: object, *, filename: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    workspace = getattr(session, "workspace", None)
    if workspace is None:
        workspace_root = str(getattr(session, "workspace_root", "") or "").strip()
        session_id = str(getattr(session, "session_id", "") or "")
        if workspace_root and session_id:
            from agentic_chatbot_next.sandbox.workspace import SessionWorkspace

            workspace = SessionWorkspace(session_id=session_id, root=Path(workspace_root))
            workspace.open()
            session.workspace = workspace
        else:
            raise ValueError("No session workspace is available.")
    content = "\n".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) for row in rows)
    if content:
        content += "\n"
    workspace.write_text(filename, content)
    return register_workspace_artifact(session, filename=filename, label=filename)


class RequirementExtractionService:
    def __init__(self, settings: object, stores: object, session: object) -> None:
        self.settings = settings
        self.stores = stores
        self.session = session
        self.tenant_id = _tenant_id(settings, session)

    def extract(
        self,
        *,
        source_scope: str = "auto",
        collection_id: str = "",
        document_names: Sequence[str] | None = None,
        all_documents: bool = False,
        mode: str = MANDATORY_MODE,
        max_preview_rows: int = 12,
        export: bool = False,
        filename: str = "",
        source_query: str = "",
    ) -> Dict[str, Any]:
        normalized_scope = str(source_scope or "").strip().lower()
        if normalized_scope not in {"auto", "uploads", "kb"}:
            normalized_scope = "auto"
        if normalized_scope == "auto" and str(collection_id or "").strip():
            effective_scope = "kb"
        elif normalized_scope == "auto":
            effective_scope = "uploads" if has_upload_evidence(self.session) else "kb"
        else:
            effective_scope = normalized_scope

        requested_document_names = [str(item) for item in (document_names or []) if str(item)]
        selection = _select_target_records(
            self.settings,
            self.stores,
            self.session,
            source_scope=effective_scope,
            collection_id=collection_id,
            document_names=requested_document_names,
            all_documents=bool(all_documents),
        )
        ignored_document_targets: list[str] = []
        if selection.get("error") and requested_document_names and selection.get("missing_documents"):
            retry_selection = _select_target_records(
                self.settings,
                self.stores,
                self.session,
                source_scope=effective_scope,
                collection_id=collection_id,
                document_names=[],
                all_documents=bool(all_documents),
            )
            if not retry_selection.get("error"):
                ignored_document_targets = requested_document_names
                selection = retry_selection
        if selection.get("error"):
            return {
                "object": "requirements.extraction_result",
                "handled": False,
                "error": str(selection.get("error") or ""),
                "source_scope": effective_scope,
                "collection_id": str(selection.get("collection_id") or ""),
                "sanitized_user_query": str(source_query or "").strip(),
                "ambiguous_documents": list(selection.get("ambiguous_documents") or []),
                "missing_documents": list(selection.get("missing_documents") or []),
                "candidate_documents": list(selection.get("candidate_documents") or []),
                "ignored_document_targets": ignored_document_targets,
                "active_uploaded_doc_ids": repository_upload_doc_ids(self.session),
            }

        target_records = list(selection.get("records") or [])
        unsupported = [
            _record_to_dict(record)
            for record in target_records
            if not supports_requirements_extraction(str(getattr(record, "file_type", "") or ""))
        ]
        target_records = [
            record
            for record in target_records
            if supports_requirements_extraction(str(getattr(record, "file_type", "") or ""))
        ]
        if not target_records:
            if unsupported:
                return {
                    "object": "requirements.extraction_result",
                    "handled": False,
                    "error": "Spreadsheet-style requirement sources are not supported for exact requirement extraction in v1.",
                    "source_scope": effective_scope,
                    "collection_id": str(selection.get("collection_id") or ""),
                    "unsupported_documents": unsupported,
                }
            return {
                "object": "requirements.extraction_result",
                "handled": False,
                "error": "No supported prose documents were available in the requested scope.",
                "source_scope": effective_scope,
                "collection_id": str(selection.get("collection_id") or ""),
            }

        normalized_mode = normalize_requirement_mode(mode)
        requirement_store = getattr(self.stores, "requirement_store", None)
        if requirement_store is None and normalized_mode != LEGAL_CLAUSE_MODE:
            return {"object": "requirements.extraction_result", "handled": False, "error": "Requirement statement storage is unavailable."}

        rows: List[Any] = []
        chunk_text_by_id: dict[str, str] = {}
        document_profiles: list[dict[str, Any]] = []
        for record in target_records:
            doc_id = str(getattr(record, "doc_id", "") or "")
            chunks = list(self.stores.chunk_store.list_document_chunks(doc_id, self.tenant_id)) if doc_id else []
            chunk_text_by_id.update({str(getattr(chunk, "chunk_id", "") or ""): str(getattr(chunk, "content", "") or "") for chunk in chunks})
            document_profiles.append(_profile_document(record, chunks, requested_mode=normalized_mode))
            if normalized_mode == LEGAL_CLAUSE_MODE:
                rows.extend(_build_ad_hoc_records(self.stores, record, tenant_id=self.tenant_id, mode=LEGAL_CLAUSE_MODE))
            else:
                _ensure_requirement_inventory(self.stores, record, tenant_id=self.tenant_id)

        if normalized_mode != LEGAL_CLAUSE_MODE:
            rows = list(
                requirement_store.list_statements(
                    tenant_id=self.tenant_id,
                    doc_ids=[str(getattr(record, "doc_id", "") or "") for record in target_records],
                    modalities=requirement_modalities_for_mode(normalized_mode),
                )
            )

        output_rows = [_row_to_output(row, mode=normalized_mode, chunk_text_by_id=chunk_text_by_id) for row in rows]
        output_rows.sort(key=lambda item: (item.get("document_title", "").casefold(), int(item.get("chunk_index") or 0), int(item.get("char_start") or 0)))
        artifacts: list[dict[str, Any]] = []
        if export:
            requested_filename = str(filename or "").strip()
            csv_filename = (
                f"{_safe_stem(requested_filename)}.csv"
                if requested_filename
                else _requirements_filename(
                    target_records,
                    collection_id=str(selection.get("collection_id") or ""),
                    all_documents=bool(all_documents),
                    suffix="csv",
                )
            )
            jsonl_filename = re.sub(r"\.csv$", ".jsonl", csv_filename, flags=re.IGNORECASE)
            if jsonl_filename == csv_filename:
                jsonl_filename = _requirements_filename(
                    target_records,
                    collection_id=str(selection.get("collection_id") or ""),
                    all_documents=bool(all_documents),
                    suffix="jsonl",
                )
            csv_artifact = _write_csv(self.session, filename=csv_filename, rows=output_rows)
            jsonl_artifact = _write_jsonl(self.session, filename=jsonl_filename, rows=output_rows)
            artifacts.extend([csv_artifact, jsonl_artifact])

        return {
            "object": "requirements.extraction_result",
            "handled": True,
            "workflow": REQUIREMENTS_WORKFLOW_KIND,
            "source_scope": effective_scope,
            "collection_id": str(selection.get("collection_id") or ""),
            "sanitized_user_query": str(source_query or "").strip(),
            "mode": normalized_mode,
            "document_count": len(target_records),
            "statement_count": len(output_rows),
            "documents": [_record_to_dict(record) for record in target_records],
            "document_profiles": document_profiles,
            "preview_rows": output_rows[: max(1, int(max_preview_rows or 12))],
            "preview_columns": ["document_title", "modality", "location", "statement_text", "actor", "confidence"],
            "artifacts": artifacts,
            "artifact": artifacts[0] if artifacts else {},
            "unsupported_documents": unsupported,
            "summary_text": self.summary_text(
                statement_count=len(output_rows),
                document_count=len(target_records),
                source_scope=effective_scope,
                collection_id=str(selection.get("collection_id") or ""),
                mode=normalized_mode,
            ),
            "warnings": self._result_warnings(
                output_rows,
                unsupported=unsupported,
                ignored_document_targets=ignored_document_targets,
            ),
            "ignored_document_targets": ignored_document_targets,
        }

    @staticmethod
    def summary_text(*, statement_count: int, document_count: int, source_scope: str, collection_id: str, mode: str) -> str:
        if mode == STRICT_SHALL_MODE:
            label = "shall statements"
        elif mode == LEGAL_CLAUSE_MODE:
            label = "legal or clause obligations"
        else:
            label = "requirement statements"
        if document_count == 1:
            return f"Extracted {statement_count} {label} from 1 document."
        if source_scope == "kb":
            return f"Extracted {statement_count} {label} across {document_count} documents in collection '{collection_id}'."
        return f"Extracted {statement_count} {label} across {document_count} uploaded documents."

    @staticmethod
    def _result_warnings(
        rows: Sequence[Mapping[str, Any]],
        *,
        unsupported: Sequence[Mapping[str, Any]],
        ignored_document_targets: Sequence[str] = (),
    ) -> List[str]:
        warnings: list[str] = []
        if unsupported:
            warnings.append(f"Skipped {len(unsupported)} unsupported document(s).")
        if ignored_document_targets:
            warnings.append(
                "Ignored unresolvable document target hint(s) after selecting the active uploaded document: "
                + ", ".join(str(item) for item in ignored_document_targets[:5])
            )
        invalid_spans = sum(1 for row in rows if not bool(row.get("source_span_valid", True)))
        if invalid_spans:
            warnings.append(f"{invalid_spans} extracted row(s) could not be source-span verified.")
        return warnings

    def extract_for_user_request(self, user_text: str, *, max_preview_rows: int = 12) -> Dict[str, Any]:
        effective_user_text = normalize_structured_query(user_text) or str(user_text or "").strip()
        mode = infer_requirement_mode(effective_user_text)
        document_names = extract_named_document_targets(effective_user_text)
        export = True
        return self.extract(
            source_scope="auto",
            document_names=document_names,
            all_documents=infer_all_documents(effective_user_text),
            mode=mode,
            max_preview_rows=max_preview_rows,
            export=export,
            source_query=effective_user_text,
        )


__all__ = [
    "RequirementExtractionService",
    "REQUIREMENTS_WORKFLOW_KIND",
    "infer_all_documents",
    "infer_requirement_mode",
    "is_requirements_extraction_request",
]
