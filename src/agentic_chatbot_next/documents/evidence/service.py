from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from agentic_chatbot_next.contracts.messages import utc_now_iso
from agentic_chatbot_next.documents.evidence.models import (
    BinderArtifact,
    BinderHandoffArtifact,
    BinderOpenIssue,
    BinderSourceDocument,
    EvidenceBinderResult,
    EvidenceRow,
)
from agentic_chatbot_next.documents.evidence.renderers import (
    build_zip_bytes,
    render_docx_bytes,
    safe_stem,
    write_binary_artifact,
)
from agentic_chatbot_next.documents.extractors import DocumentExtractionService, DocumentResolutionError, element_location
from agentic_chatbot_next.runtime.artifacts import list_handoff_artifacts, normalize_artifact


CITATION_POLICIES = {"warn_and_include", "exclude_unsupported", "fail_closed"}
_CLASSIFICATION_RE = re.compile(
    r"\b(?:UNCLASSIFIED|CUI|CONTROLLED\s+UNCLASSIFIED|SECRET|TOP\s+SECRET|NOFORN|EXPORT\s+CONTROL|DISTRIBUTION\s+STATEMENT)\b",
    re.IGNORECASE,
)
_EVIDENCE_ID_PREFIX = "EV"


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _short(value: Any, limit: int = 500) -> str:
    text = _clean(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _normalise_policy(value: str) -> str:
    normalized = str(value or "warn_and_include").strip().lower()
    return normalized if normalized in CITATION_POLICIES else "warn_and_include"


def _metadata(session: object) -> dict[str, Any]:
    raw = getattr(session, "metadata", None)
    if isinstance(raw, dict):
        return raw
    session.metadata = {}
    return session.metadata


def _workspace(session: object) -> object | None:
    workspace = getattr(session, "workspace", None)
    if workspace is not None:
        return workspace
    root = str(getattr(session, "workspace_root", "") or "").strip()
    session_id = str(getattr(session, "session_id", "") or "").strip()
    if not root or not session_id:
        return None
    from agentic_chatbot_next.sandbox.workspace import SessionWorkspace

    workspace = SessionWorkspace(session_id=session_id, root=Path(root))
    workspace.open()
    session.workspace = workspace
    return workspace


def _hash_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _location_text(location: Dict[str, Any]) -> str:
    parts = []
    for key in ("section_title", "clause_number", "page_number", "slide_number", "sheet_name", "cell_range", "row_start", "row_end"):
        value = location.get(key)
        if value not in ("", None):
            parts.append(f"{key}={value}")
    return "; ".join(parts)


class EvidenceBinderService:
    def __init__(self, settings: object, stores: object, session: object) -> None:
        self.settings = settings
        self.stores = stores
        self.session = session

    def build(
        self,
        *,
        binder_title: str = "Evidence Binder",
        objective: str = "",
        document_refs: Sequence[str] | None = None,
        artifact_refs: Sequence[str] | None = None,
        handoff_artifact_ids: Sequence[str] | None = None,
        handoff_artifact_types: Sequence[str] | None = None,
        source_scope: str = "auto",
        collection_id: str = "",
        include_latest_artifacts: bool = True,
        include_generated_artifacts: bool = True,
        include_original_sources: bool = False,
        citation_policy: str = "warn_and_include",
        max_source_excerpts: int = 300,
        export: bool = True,
        binder_id: str = "",
    ) -> EvidenceBinderResult:
        binder_id = binder_id or f"evidence_binder_{uuid.uuid4().hex[:12]}"
        title = _clean(binder_title) or "Evidence Binder"
        policy = _normalise_policy(citation_policy)
        max_excerpts = max(1, int(max_source_excerpts or 300))
        warnings = [
            "Evidence binder packages provenance for review; classification marking and redaction review are not automatic in V1."
        ]

        source_documents, source_rows = self._extract_source_documents(
            document_refs or [],
            source_scope=source_scope,
            collection_id=collection_id,
            max_source_excerpts=max_excerpts,
            warnings=warnings,
        )
        artifacts = self._resolve_artifacts(
            artifact_refs or [],
            include_latest_artifacts=include_latest_artifacts,
            include_generated_artifacts=include_generated_artifacts,
        )
        handoffs = self._resolve_handoff_artifacts(
            handoff_artifact_ids=handoff_artifact_ids or [],
            handoff_artifact_types=handoff_artifact_types or [],
            include_latest_artifacts=include_latest_artifacts,
        )

        rows = list(source_rows)
        rows.extend(self._rows_from_artifacts(artifacts, source_rows=source_rows))
        rows.extend(self._rows_from_handoffs(handoffs))
        rows, issues = self._apply_citation_policy(rows, policy)
        if not rows:
            issues.append(
                BinderOpenIssue(
                    issue_id="ISSUE-0001",
                    severity="medium",
                    message="No evidence rows were produced from the selected sources or artifacts.",
                )
            )

        result = EvidenceBinderResult(
            binder_id=binder_id,
            status="completed",
            binder_title=title,
            objective=_clean(objective),
            generated_at=utc_now_iso(),
            source_documents=source_documents,
            artifacts=artifacts,
            handoff_artifacts=handoffs,
            evidence_rows=rows,
            open_issues=issues,
            warnings=warnings,
        )
        if export:
            result.binder_artifacts = self._write_outputs(
                result,
                include_generated_artifacts=include_generated_artifacts,
                include_original_sources=include_original_sources,
            )
        return result

    def _extract_source_documents(
        self,
        document_refs: Sequence[str],
        *,
        source_scope: str,
        collection_id: str,
        max_source_excerpts: int,
        warnings: list[str],
    ) -> tuple[list[BinderSourceDocument], list[EvidenceRow]]:
        refs = [_clean(item) for item in document_refs if _clean(item)]
        if not refs:
            return [], []
        extractor = DocumentExtractionService(self.settings, self.stores, self.session)
        sources: list[BinderSourceDocument] = []
        rows: list[EvidenceRow] = []
        row_index = 1
        remaining = max_source_excerpts
        for ref in refs:
            if remaining <= 0:
                warnings.append(f"Source excerpt extraction was capped at max_source_excerpts={max_source_excerpts}.")
                break
            try:
                result = extractor.extract(
                    document_ref=ref,
                    source_scope=source_scope,
                    collection_id=collection_id,
                    include_tables=True,
                    include_figures=True,
                    include_metadata=True,
                    include_hierarchy=True,
                    max_elements=remaining,
                )
            except DocumentResolutionError as exc:
                warnings.append(f"Could not resolve source document {ref}: {exc.payload.get('error')}")
                continue
            sources.append(
                BinderSourceDocument(
                    doc_id=result.document.doc_id,
                    title=result.document.title,
                    source_type=result.document.source_type,
                    source_path=result.document.source_path,
                    collection_id=result.document.collection_id,
                    file_type=result.document.file_type,
                    content_hash=result.document.content_hash,
                    parser_path=result.document.parser_path,
                    warnings=list(result.warnings),
                    classification_markings=self._classification_markings([element.text for element in result.elements[:20]]),
                )
            )
            for element in result.elements[:remaining]:
                if not _clean(element.text):
                    continue
                evidence_id = f"{_EVIDENCE_ID_PREFIX}-{row_index:05d}"
                rows.append(
                    EvidenceRow(
                        evidence_id=evidence_id,
                        claim=_short(element.text, 320),
                        source_excerpt=_short(element.text, 900),
                        source_document=result.document.title,
                        source_location=_location_text(element_location(element)),
                        source_hash=result.document.content_hash,
                        producing_tool="document_extract",
                        citation_status="source_backed",
                        review_status="ready_for_review",
                        notes=f"Source element {element.element_id}; parser={result.document.parser_path}",
                    )
                )
                row_index += 1
            remaining = max(0, max_source_excerpts - len(rows))
        return sources, rows

    def _classification_markings(self, texts: Iterable[str]) -> list[str]:
        markings: list[str] = []
        for text in texts:
            clean = _short(text, 240)
            if _CLASSIFICATION_RE.search(clean):
                markings.append(clean)
        return list(dict.fromkeys(markings))[:10]

    def _resolve_artifacts(
        self,
        artifact_refs: Sequence[str],
        *,
        include_latest_artifacts: bool,
        include_generated_artifacts: bool,
    ) -> list[BinderArtifact]:
        metadata = _metadata(self.session)
        downloads = [
            normalize_artifact(item)
            for item in dict(metadata.get("downloads") or {}).values()
            if isinstance(item, dict)
        ]
        pending = [
            normalize_artifact(item)
            for item in list(metadata.get("pending_artifacts") or [])
            if isinstance(item, dict)
        ]
        by_key: dict[str, dict[str, Any]] = {}
        for item in [*downloads, *pending]:
            if not item.get("filename") or self._is_binder_generated(str(item.get("filename") or "")):
                continue
            for key in {
                str(item.get("artifact_ref") or ""),
                str(item.get("download_id") or ""),
                str(item.get("filename") or ""),
            }:
                if key:
                    by_key[key] = item

        selected: dict[str, dict[str, Any]] = {}
        workspace = _workspace(self.session)
        for ref in artifact_refs:
            clean = _clean(ref)
            if clean and clean in by_key:
                selected[str(by_key[clean].get("filename"))] = by_key[clean]
            elif clean and workspace is not None:
                workspace_item = self._workspace_artifact_record(clean, workspace)
                if workspace_item is not None:
                    selected[str(workspace_item.get("filename"))] = workspace_item
        if include_latest_artifacts:
            for item in [*downloads, *pending]:
                filename = str(item.get("filename") or "")
                if filename and not self._is_binder_generated(filename):
                    selected.setdefault(filename, item)

        rows: list[BinderArtifact] = []
        for item in selected.values():
            filename = str(item.get("filename") or "")
            path = workspace.root / filename if workspace is not None and filename else None
            path_exists = bool(path is not None and path.exists() and path.is_file())
            rows.append(
                BinderArtifact(
                    artifact_ref=str(item.get("artifact_ref") or ""),
                    filename=filename,
                    label=str(item.get("label") or filename),
                    content_type=str(item.get("content_type") or ""),
                    size_bytes=int((path.stat().st_size if path_exists and path is not None else item.get("size_bytes")) or 0),
                    content_hash=_hash_file(path) if path_exists and path is not None else "",
                    source=str(item.get("source") or "session_download"),
                    included_in_zip=bool(include_generated_artifacts and path_exists),
                    missing_file=not path_exists,
                )
            )
        return rows

    def _workspace_artifact_record(self, ref: str, workspace: object) -> dict[str, Any] | None:
        raw = Path(ref)
        root = Path(getattr(workspace, "root", ""))
        candidate = raw if raw.is_absolute() else root / raw
        try:
            resolved_root = root.resolve()
            resolved_candidate = candidate.resolve()
            resolved_candidate.relative_to(resolved_root)
        except Exception:
            return None
        if not resolved_candidate.exists() or not resolved_candidate.is_file():
            return None
        filename = resolved_candidate.relative_to(resolved_root).as_posix()
        if self._is_binder_generated(filename):
            return None
        return normalize_artifact(
            {
                "artifact_ref": f"workspace://{filename}",
                "filename": filename,
                "label": filename,
                "content_type": mimetypes.guess_type(filename)[0] or "application/octet-stream",
                "size_bytes": resolved_candidate.stat().st_size,
                "session_id": str(getattr(self.session, "session_id", "") or ""),
                "conversation_id": str(getattr(self.session, "conversation_id", "") or ""),
                "source": "workspace_file",
            }
        ) | {"source": "workspace_file"}

    def _is_binder_generated(self, filename: str) -> bool:
        lowered = str(filename or "").casefold()
        return lowered.startswith("evidence_binder_") or "__evidence_binder" in lowered

    def _resolve_handoff_artifacts(
        self,
        *,
        handoff_artifact_ids: Sequence[str],
        handoff_artifact_types: Sequence[str],
        include_latest_artifacts: bool,
    ) -> list[BinderHandoffArtifact]:
        ids = [_clean(item) for item in handoff_artifact_ids if _clean(item)]
        types = [_clean(item) for item in handoff_artifact_types if _clean(item)]
        if not ids and not types and not include_latest_artifacts:
            return []
        records = list_handoff_artifacts(
            self.session,
            artifact_ids=ids or None,
            artifact_types=types or None,
        )
        return [
            BinderHandoffArtifact(
                artifact_id=str(item.get("artifact_id") or ""),
                artifact_ref=str(item.get("artifact_ref") or ""),
                artifact_type=str(item.get("artifact_type") or ""),
                handoff_schema=str(item.get("handoff_schema") or ""),
                producer_task_id=str(item.get("producer_task_id") or ""),
                producer_agent=str(item.get("producer_agent") or ""),
                summary=str(item.get("summary") or ""),
                created_at=str(item.get("created_at") or ""),
            )
            for item in records
        ]

    def _rows_from_artifacts(self, artifacts: Sequence[BinderArtifact], *, source_rows: Sequence[EvidenceRow]) -> list[EvidenceRow]:
        workspace = _workspace(self.session)
        if workspace is None:
            return []
        source_trace = self._source_trace_map(source_rows)
        rows: list[EvidenceRow] = []
        next_index = 1
        for artifact in artifacts:
            path = workspace.root / artifact.filename
            if not path.exists() or not path.is_file():
                rows.append(self._artifact_issue_row(next_index, artifact, "Artifact file was registered but is not present in the workspace."))
                next_index += 1
                continue
            produced = self._parse_artifact_rows(path, artifact, source_trace=source_trace, start_index=next_index)
            if produced:
                rows.extend(produced)
                next_index += len(produced)
            else:
                rows.append(
                    self._artifact_issue_row(
                        next_index,
                        artifact,
                        "Artifact was included, but no source trace or structured claims were recognized.",
                    )
                )
                next_index += 1
        return rows

    def _source_trace_map(self, source_rows: Sequence[EvidenceRow]) -> dict[str, EvidenceRow]:
        return {row.evidence_id: row for row in source_rows}

    def _parse_artifact_rows(
        self,
        path: Path,
        artifact: BinderArtifact,
        *,
        source_trace: dict[str, EvidenceRow],
        start_index: int,
    ) -> list[EvidenceRow]:
        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                return self._rows_from_json_payload(payload, artifact, source_trace=source_trace, start_index=start_index)
            if suffix == ".jsonl":
                values = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                return self._rows_from_json_payload(values, artifact, source_trace=source_trace, start_index=start_index)
            if suffix == ".csv":
                reader = csv.DictReader(path.read_text(encoding="utf-8").splitlines())
                return [
                    self._row_from_claim(
                        start_index + index,
                        claim=self._claim_from_mapping(row),
                        artifact=artifact,
                        source_row=source_trace.get(str(row.get("Evidence/Source") or row.get("Source ID") or "")),
                        fallback_note="CSV row imported from artifact.",
                    )
                    for index, row in enumerate(list(reader)[:120])
                ]
        except Exception as exc:
            return [self._artifact_issue_row(start_index, artifact, f"Artifact parsing failed: {exc}")]
        return []

    def _rows_from_json_payload(
        self,
        payload: Any,
        artifact: BinderArtifact,
        *,
        source_trace: dict[str, EvidenceRow],
        start_index: int,
    ) -> list[EvidenceRow]:
        rows: list[EvidenceRow] = []
        local_trace = dict(source_trace)
        if isinstance(payload, dict):
            local_trace.update(self._trace_from_template_payload(payload))
            for section in list(payload.get("sections") or [])[:80]:
                source_row = self._first_source_row(section, local_trace)
                claim = _short(section.get("body") or "; ".join(section.get("bullets") or []) or section.get("title"))
                rows.append(self._row_from_claim(start_index + len(rows), claim=claim, artifact=artifact, source_row=source_row, fallback_note="Template section imported from JSON sidecar."))
            for slide in list(payload.get("slides") or [])[:80]:
                source_row = self._first_source_row(slide, local_trace)
                claim = _short("; ".join([slide.get("title", ""), *list(slide.get("bullets") or [])]))
                rows.append(self._row_from_claim(start_index + len(rows), claim=claim, artifact=artifact, source_row=source_row, fallback_note="Template slide imported from JSON sidecar."))
            for table in list(payload.get("tables") or [])[:20]:
                for raw in list(table.get("rows") or [])[:80]:
                    source_id = str(raw.get("Evidence/Source") or raw.get("Source ID") or "")
                    source_row = local_trace.get(source_id)
                    rows.append(self._row_from_claim(start_index + len(rows), claim=self._claim_from_mapping(raw), artifact=artifact, source_row=source_row, fallback_note="Template table row imported from JSON sidecar."))
            if rows:
                return rows
            payload = [payload]
        if isinstance(payload, list):
            for raw in payload[:120]:
                if not isinstance(raw, dict):
                    continue
                rows.append(
                    self._row_from_claim(
                        start_index + len(rows),
                        claim=self._claim_from_mapping(raw),
                        artifact=artifact,
                        source_row=None,
                        fallback_note="Structured artifact row has no recognized source trace.",
                    )
                )
        return rows

    def _trace_from_template_payload(self, payload: dict[str, Any]) -> dict[str, EvidenceRow]:
        rows: dict[str, EvidenceRow] = {}
        for doc in list(payload.get("source_documents") or []):
            for entry in list(doc.get("trace_entries") or []):
                source_id = str(entry.get("source_id") or "")
                if not source_id:
                    continue
                rows[source_id] = EvidenceRow(
                    evidence_id=source_id,
                    claim=_short(entry.get("text_excerpt"), 320),
                    source_excerpt=_short(entry.get("text_excerpt"), 900),
                    source_document=str(entry.get("document_title") or doc.get("title") or ""),
                    source_location=_location_text(dict(entry.get("source_location") or {})),
                    source_hash=str(doc.get("content_hash") or ""),
                    producing_tool="template_transform_source_trace",
                    citation_status="source_backed",
                    review_status="ready_for_review",
                    notes=f"Trace element {entry.get('element_id') or ''}; parser={entry.get('parser_path') or doc.get('parser_path') or ''}",
                )
        return rows

    def _first_source_row(self, raw: dict[str, Any], trace: dict[str, EvidenceRow]) -> EvidenceRow | None:
        for source_id in list(raw.get("source_ids") or []):
            found = trace.get(str(source_id))
            if found is not None:
                return found
        return None

    def _claim_from_mapping(self, row: dict[str, Any]) -> str:
        for key in (
            "Requirement Text",
            "Claim",
            "summary",
            "recommendation",
            "Deliverable Title",
            "Notes",
            "title",
            "text",
        ):
            value = _clean(row.get(key))
            if value:
                return _short(value, 420)
        pairs = [f"{key}: {_short(value, 120)}" for key, value in row.items() if _clean(value)]
        return _short("; ".join(pairs), 420) or "Structured artifact row."

    def _row_from_claim(
        self,
        index: int,
        *,
        claim: str,
        artifact: BinderArtifact,
        source_row: EvidenceRow | None,
        fallback_note: str,
    ) -> EvidenceRow:
        if source_row is not None:
            return EvidenceRow(
                evidence_id=f"{_EVIDENCE_ID_PREFIX}-A{index:05d}",
                claim=_short(claim, 500),
                source_excerpt=source_row.source_excerpt,
                source_document=source_row.source_document,
                source_location=source_row.source_location,
                source_hash=source_row.source_hash,
                producing_tool=artifact.label or artifact.filename,
                artifact_ref=artifact.artifact_ref,
                artifact_filename=artifact.filename,
                citation_status="source_backed",
                review_status="ready_for_review",
                notes=f"Supported by {source_row.evidence_id}.",
            )
        return EvidenceRow(
            evidence_id=f"{_EVIDENCE_ID_PREFIX}-A{index:05d}",
            claim=_short(claim, 500),
            producing_tool=artifact.label or artifact.filename,
            artifact_ref=artifact.artifact_ref,
            artifact_filename=artifact.filename,
            citation_status="missing",
            review_status="needs_review",
            notes=fallback_note,
        )

    def _artifact_issue_row(self, index: int, artifact: BinderArtifact, message: str) -> EvidenceRow:
        return EvidenceRow(
            evidence_id=f"{_EVIDENCE_ID_PREFIX}-A{index:05d}",
            claim=f"Included artifact: {artifact.filename}",
            producing_tool=artifact.label or artifact.filename,
            artifact_ref=artifact.artifact_ref,
            artifact_filename=artifact.filename,
            citation_status="missing",
            review_status="needs_review",
            notes=message,
        )

    def _rows_from_handoffs(self, handoffs: Sequence[BinderHandoffArtifact]) -> list[EvidenceRow]:
        rows: list[EvidenceRow] = []
        for index, artifact in enumerate(handoffs, start=1):
            rows.append(
                EvidenceRow(
                    evidence_id=f"{_EVIDENCE_ID_PREFIX}-H{index:05d}",
                    claim=artifact.summary or f"Handoff artifact {artifact.artifact_type}",
                    producing_tool=artifact.producer_agent,
                    artifact_ref=artifact.artifact_ref,
                    artifact_filename=artifact.artifact_type,
                    citation_status="weak",
                    review_status="needs_review",
                    notes=f"Handoff schema={artifact.handoff_schema}; producer_task_id={artifact.producer_task_id}",
                )
            )
        return rows

    def _apply_citation_policy(self, rows: list[EvidenceRow], policy: str) -> tuple[list[EvidenceRow], list[BinderOpenIssue]]:
        issues: list[BinderOpenIssue] = []
        unsupported = [row for row in rows if row.citation_status != "source_backed"]
        for index, row in enumerate(unsupported, start=1):
            issues.append(
                BinderOpenIssue(
                    issue_id=f"ISSUE-{index:04d}",
                    severity="medium" if row.citation_status == "missing" else "low",
                    message=f"Evidence row {row.evidence_id} has citation_status={row.citation_status}.",
                    related_artifact=row.artifact_filename,
                    related_evidence_id=row.evidence_id,
                )
            )
        if unsupported and policy == "fail_closed":
            raise ValueError(f"Evidence binder citation policy failed: {len(unsupported)} unsupported evidence row(s).")
        if unsupported and policy == "exclude_unsupported":
            rows = [row for row in rows if row.citation_status == "source_backed"]
        return rows, issues

    def _write_outputs(
        self,
        result: EvidenceBinderResult,
        *,
        include_generated_artifacts: bool,
        include_original_sources: bool,
    ) -> list[dict[str, Any]]:
        workspace = _workspace(self.session)
        if workspace is None:
            raise ValueError("No session workspace is available.")
        stem = safe_stem(result.binder_title)
        docx_name = f"{stem}__{result.binder_id}.docx"
        docx_artifact = write_binary_artifact(self.session, docx_name, render_docx_bytes(result))
        result.binder_artifacts.append(docx_artifact)

        included_filenames = [
            artifact.filename
            for artifact in result.artifacts
            if include_generated_artifacts and artifact.included_in_zip and artifact.filename
        ]
        original_sources: list[Path] = []
        if include_original_sources:
            for source in result.source_documents:
                path = Path(source.source_path)
                if path.exists() and path.is_file():
                    original_sources.append(path)
        zip_name = f"{stem}__{result.binder_id}.zip"
        zip_bytes = build_zip_bytes(
            result,
            workspace_root=workspace.root,
            included_artifact_filenames=included_filenames,
            original_source_paths=original_sources,
        )
        zip_artifact = write_binary_artifact(self.session, zip_name, zip_bytes)
        result.binder_artifacts.append(zip_artifact)
        return list(result.binder_artifacts)


__all__ = ["CITATION_POLICIES", "EvidenceBinderService"]
