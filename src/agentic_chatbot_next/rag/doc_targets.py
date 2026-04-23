from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from agentic_chatbot_next.rag.ingest import resolve_record_source_identity, select_active_kb_record
from agentic_chatbot_next.rag.retrieval_scope import resolve_search_collection_ids

_FILE_NAME_PATTERN = re.compile(
    r"(?<![\w/-])([A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,8})(?![\w/-])"
)
_QUOTED_PATTERN = re.compile(r'"([^"\n]{1,240})"|(?<!\w)\'([^\'\n]{1,240})\'(?!\w)')
_BARE_NUMERIC_DECIMAL_PATTERN = re.compile(r"^[+-]?\d+(?:\.\d+)+$")
_SUPPORTED_DOCUMENT_TARGET_EXTENSIONS = {
    "csv",
    "doc",
    "docx",
    "htm",
    "html",
    "md",
    "markdown",
    "pdf",
    "pptx",
    "rtf",
    "tsv",
    "txt",
    "xls",
    "xlsx",
}


def _normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.replace("\\", "/").casefold().split())


def _basename(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return Path(text.replace("\\", "/")).name


def _stem(value: str) -> str:
    base = _basename(value)
    if not base:
        return ""
    return Path(base).stem


def _looks_like_document_target(value: str) -> bool:
    candidate = str(value or "").strip()
    if not candidate:
        return False
    if len(candidate) > 240 or "\n" in candidate or "\r" in candidate:
        return False
    # Numeric quantities such as "4.2 million" satisfy the filename regex but
    # are not document names. Keep this guard local to target extraction so
    # arithmetic prompts do not accidentally force exact-document RAG scope.
    if _BARE_NUMERIC_DECIMAL_PATTERN.fullmatch(candidate):
        return False
    extension = Path(_basename(candidate)).suffix.lower().lstrip(".")
    if extension not in _SUPPORTED_DOCUMENT_TARGET_EXTENSIONS:
        return False
    return True


def extract_named_document_targets(query: str) -> list[str]:
    text = str(query or "")
    candidates: list[str] = []
    seen: set[str] = set()

    for match in _FILE_NAME_PATTERN.finditer(text):
        candidate = str(match.group(1) or "").strip()
        if not _looks_like_document_target(candidate):
            continue
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    for match in _QUOTED_PATTERN.finditer(text):
        candidate = str(match.group(1) or match.group(2) or "").strip()
        if not candidate:
            continue
        if "." not in candidate and "/" not in candidate and "\\" not in candidate:
            continue
        if not _looks_like_document_target(candidate):
            continue
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


@dataclass(frozen=True)
class ResolvedIndexedDoc:
    doc_id: str
    title: str
    source_type: str
    source_path: str
    collection_id: str
    file_type: str
    doc_structure_type: str
    match_name: str
    match_type: str
    source_identity: str = ""
    content_hash: str = ""
    ingested_at: str = ""
    ignored_duplicate_doc_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, str]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "collection_id": self.collection_id,
            "file_type": self.file_type,
            "doc_structure_type": self.doc_structure_type,
            "match_name": self.match_name,
            "match_type": self.match_type,
            "source_identity": self.source_identity,
            "content_hash": self.content_hash,
            "ingested_at": self.ingested_at,
            "ignored_duplicate_doc_ids": list(self.ignored_duplicate_doc_ids),
        }


@dataclass(frozen=True)
class AmbiguousIndexedDocMatch:
    requested_name: str
    candidates: tuple[ResolvedIndexedDoc, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_name": self.requested_name,
            "candidates": [item.to_dict() for item in self.candidates],
        }


@dataclass(frozen=True)
class MissingIndexedDocMatch:
    requested_name: str
    suggestions: tuple[ResolvedIndexedDoc, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_name": self.requested_name,
            "suggestions": [item.to_dict() for item in self.suggestions],
        }


@dataclass(frozen=True)
class IndexedDocResolution:
    requested_names: tuple[str, ...] = ()
    resolved: tuple[ResolvedIndexedDoc, ...] = ()
    ambiguous: tuple[AmbiguousIndexedDocMatch, ...] = ()
    missing: tuple[MissingIndexedDocMatch, ...] = ()

    @property
    def resolved_doc_ids(self) -> list[str]:
        seen: set[str] = set()
        doc_ids: list[str] = []
        for item in self.resolved:
            if item.doc_id and item.doc_id not in seen:
                seen.add(item.doc_id)
                doc_ids.append(item.doc_id)
        return doc_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_names": list(self.requested_names),
            "resolved": [item.to_dict() for item in self.resolved],
            "ambiguous": [item.to_dict() for item in self.ambiguous],
            "missing": [item.to_dict() for item in self.missing],
        }


def _record_payload(
    record: Any,
    *,
    match_name: str,
    match_type: str,
    source_identity: str = "",
    ignored_duplicate_doc_ids: Sequence[str] = (),
) -> ResolvedIndexedDoc:
    return ResolvedIndexedDoc(
        doc_id=str(getattr(record, "doc_id", "") or ""),
        title=str(getattr(record, "title", "") or ""),
        source_type=str(getattr(record, "source_type", "") or ""),
        source_path=str(getattr(record, "source_path", "") or ""),
        collection_id=str(getattr(record, "collection_id", "") or ""),
        file_type=str(getattr(record, "file_type", "") or ""),
        doc_structure_type=str(getattr(record, "doc_structure_type", "") or ""),
        match_name=match_name,
        match_type=match_type,
        source_identity=source_identity,
        content_hash=str(getattr(record, "content_hash", "") or ""),
        ingested_at=str(getattr(record, "ingested_at", "") or ""),
        ignored_duplicate_doc_ids=tuple(str(item) for item in ignored_duplicate_doc_ids if str(item)),
    )


def _collection_records(stores: Any, *, tenant_id: str, collection_ids: Sequence[str]) -> list[Any]:
    records: list[Any] = []
    seen: set[str] = set()
    scoped_ids = [str(item) for item in collection_ids if str(item).strip()]
    try:
        if scoped_ids:
            for collection_id in scoped_ids:
                for record in stores.doc_store.list_documents(
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                ):
                    doc_id = str(getattr(record, "doc_id", "") or "")
                    if doc_id and doc_id not in seen:
                        seen.add(doc_id)
                        records.append(record)
        else:
            for record in stores.doc_store.list_documents(tenant_id=tenant_id):
                doc_id = str(getattr(record, "doc_id", "") or "")
                if doc_id and doc_id not in seen:
                    seen.add(doc_id)
                    records.append(record)
    except Exception:
        return []
    return records


def _configured_paths_by_title(settings: Any | None) -> dict[str, list[Path]]:
    if settings is None:
        return {}
    try:
        from agentic_chatbot_next.rag.ingest import iter_kb_source_paths

        grouped: dict[str, list[Path]] = {}
        for path in iter_kb_source_paths(settings):
            grouped.setdefault(path.name.casefold(), []).append(path.resolve())
        return grouped
    except Exception:
        return {}


def _exact_matches(records: Sequence[Any], name: str) -> list[ResolvedIndexedDoc]:
    requested = str(name or "").strip()
    if not requested:
        return []
    normalized_requested = _normalize_name(requested)
    requested_base = _basename(requested)
    normalized_base = _normalize_name(requested_base)
    requested_stem = _stem(requested)
    normalized_stem = _normalize_name(requested_stem)

    matches: list[ResolvedIndexedDoc] = []
    seen: set[str] = set()
    for record in records:
        title = str(getattr(record, "title", "") or "")
        source_path = str(getattr(record, "source_path", "") or "")
        title_base = _basename(title)
        source_base = _basename(source_path)
        candidates = [
            ("title_exact", title),
            ("title_basename", title_base),
            ("source_basename", source_base),
            ("title_normalized", _normalize_name(title)),
            ("title_basename_normalized", _normalize_name(title_base)),
            ("source_basename_normalized", _normalize_name(source_base)),
            ("title_stem", _normalize_name(_stem(title))),
            ("source_stem", _normalize_name(_stem(source_base))),
        ]
        matched = False
        match_type = ""
        for candidate_type, candidate_value in candidates:
            candidate_text = str(candidate_value or "").strip()
            if not candidate_text:
                continue
            if candidate_type.endswith("normalized"):
                if candidate_text == normalized_requested or candidate_text == normalized_base:
                    matched = True
                    match_type = candidate_type
                    break
                if normalized_stem and candidate_text == normalized_stem:
                    matched = True
                    match_type = candidate_type
                    break
                continue
            if candidate_text.casefold() == requested.casefold() or (
                requested_base and candidate_text.casefold() == requested_base.casefold()
            ):
                matched = True
                match_type = candidate_type
                break
            if requested_stem and _normalize_name(candidate_text) == normalized_stem:
                matched = True
                match_type = candidate_type
                break
        if not matched:
            continue
        doc_id = str(getattr(record, "doc_id", "") or "")
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        matches.append(_record_payload(record, match_name=requested, match_type=match_type or "exact"))
    return matches


def _collapse_duplicate_matches(
    matches: Sequence[ResolvedIndexedDoc],
    *,
    records_by_doc_id: dict[str, Any],
    configured_paths_by_title: dict[str, list[Path]] | None = None,
) -> list[ResolvedIndexedDoc]:
    grouped: dict[str, list[Any]] = {}
    fallback_groups: dict[str, list[ResolvedIndexedDoc]] = {}
    for match in matches:
        record = records_by_doc_id.get(match.doc_id)
        if record is None:
            fallback_groups.setdefault(match.doc_id, []).append(match)
            continue
        identity = resolve_record_source_identity(
            record,
            configured_paths_by_title=configured_paths_by_title or {},
        )
        grouped.setdefault(identity, []).append(record)

    collapsed: list[ResolvedIndexedDoc] = []
    seen_doc_ids: set[str] = set()
    for identity, group_records in grouped.items():
        active = select_active_kb_record(group_records)
        if active is None:
            continue
        active_doc_id = str(getattr(active, "doc_id", "") or "")
        if not active_doc_id or active_doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(active_doc_id)
        ignored_duplicate_doc_ids = [
            str(getattr(record, "doc_id", "") or "")
            for record in group_records
            if str(getattr(record, "doc_id", "") or "") and str(getattr(record, "doc_id", "") or "") != active_doc_id
        ]
        template = next((item for item in matches if item.doc_id == active_doc_id), matches[0])
        collapsed.append(
            _record_payload(
                active,
                match_name=template.match_name,
                match_type=template.match_type,
                source_identity=identity,
                ignored_duplicate_doc_ids=ignored_duplicate_doc_ids,
            )
        )
    for match_group in fallback_groups.values():
        for match in match_group:
            if match.doc_id and match.doc_id not in seen_doc_ids:
                seen_doc_ids.add(match.doc_id)
                collapsed.append(match)
    return collapsed


def _suggestions(stores: Any, *, tenant_id: str, collection_ids: Sequence[str], name: str) -> list[ResolvedIndexedDoc]:
    suggestions: list[ResolvedIndexedDoc] = []
    seen: set[str] = set()
    for collection_id in list(collection_ids or [""]):
        try:
            matches = stores.doc_store.fuzzy_search_title(
                name,
                tenant_id=tenant_id,
                limit=5,
                collection_id=str(collection_id or ""),
            )
        except Exception:
            matches = []
        for item in matches:
            doc_id = str(item.get("doc_id") or "")
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            try:
                record = stores.doc_store.get_document(doc_id, tenant_id)
            except Exception:
                record = None
            if record is None:
                continue
            suggestions.append(_record_payload(record, match_name=name, match_type="suggestion"))
    return suggestions


def resolve_indexed_docs(
    stores: Any,
    *,
    settings: Any | None = None,
    tenant_id: str,
    names: Iterable[str],
    collection_ids: Sequence[str] = (),
) -> IndexedDocResolution:
    requested_names = tuple(str(item).strip() for item in names if str(item).strip())
    if not requested_names:
        return IndexedDocResolution()

    records = _collection_records(stores, tenant_id=tenant_id, collection_ids=collection_ids)
    records_by_doc_id = {
        str(getattr(record, "doc_id", "") or ""): record
        for record in records
        if str(getattr(record, "doc_id", "") or "")
    }
    configured_paths_by_title = _configured_paths_by_title(settings)
    resolved: list[ResolvedIndexedDoc] = []
    ambiguous: list[AmbiguousIndexedDocMatch] = []
    missing: list[MissingIndexedDocMatch] = []
    seen_doc_ids: set[str] = set()

    for name in requested_names:
        matches = _collapse_duplicate_matches(
            _exact_matches(records, name),
            records_by_doc_id=records_by_doc_id,
            configured_paths_by_title=configured_paths_by_title,
        )
        if len(matches) == 1:
            match = matches[0]
            if match.doc_id not in seen_doc_ids:
                seen_doc_ids.add(match.doc_id)
                resolved.append(match)
            continue
        if len(matches) > 1:
            ambiguous.append(AmbiguousIndexedDocMatch(requested_name=name, candidates=tuple(matches[:5])))
            continue
        missing.append(
            MissingIndexedDocMatch(
                requested_name=name,
                suggestions=tuple(_suggestions(stores, tenant_id=tenant_id, collection_ids=collection_ids, name=name)[:5]),
            )
        )

    return IndexedDocResolution(
        requested_names=requested_names,
        resolved=tuple(resolved),
        ambiguous=tuple(ambiguous),
        missing=tuple(missing),
    )


def resolve_query_document_targets(settings: Any, stores: Any, session: Any, *, query: str) -> IndexedDocResolution:
    names = extract_named_document_targets(query)
    if not names:
        return IndexedDocResolution()
    collection_ids = resolve_search_collection_ids(settings, session)
    tenant_id = str(getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev")) or "local-dev")
    return resolve_indexed_docs(
        stores,
        settings=settings,
        tenant_id=tenant_id,
        names=names,
        collection_ids=collection_ids,
    )
