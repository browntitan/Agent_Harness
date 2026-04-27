from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from agentic_chatbot_next.documents.models import DocumentCompareResult, DocumentExtractResult
from agentic_chatbot_next.runtime.artifacts import register_workspace_artifact


_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_stem(value: str, *, fallback: str) -> str:
    stem = Path(str(value or "")).stem or fallback
    clean = _FILENAME_SAFE_RE.sub("_", stem).strip("._")
    return (clean or fallback)[:64]


def _write_workspace_text(session: object, filename: str, content: str) -> Dict[str, Any]:
    workspace = getattr(session, "workspace", None)
    if workspace is None:
        raise ValueError("No session workspace is available.")
    workspace.write_text(filename, content)
    return register_workspace_artifact(session, filename=filename, label=filename)


def extract_to_markdown(result: DocumentExtractResult, *, max_elements: int = 80) -> str:
    doc = result.document
    lines = [
        f"# Document Extraction: {doc.title or doc.doc_id}",
        "",
        "## Document",
        f"- Doc ID: {doc.doc_id}",
        f"- Source: {doc.source_type or doc.source_scope}",
        f"- File type: {doc.file_type}",
        f"- Parser: {doc.parser_path}",
        "",
        "## Counts",
    ]
    for key, value in result.counts().items():
        lines.append(f"- {key}: {value}")
    if result.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {item}" for item in result.warnings)
    if result.metadata:
        lines.extend(["", "## Metadata"])
        for key, value in sorted(result.metadata.items()):
            lines.append(f"- {key}: {value}")
    if result.sections:
        lines.extend(["", "## Section Hierarchy"])
        by_id = {section.section_id: section for section in result.sections}
        for section in result.sections[:max_elements]:
            depth = 0
            cursor = section
            while cursor.parent_id and cursor.parent_id in by_id:
                depth += 1
                cursor = by_id[cursor.parent_id]
            lines.append(f"{'  ' * depth}- {section.title}")
    if result.tables:
        lines.extend(["", "## Tables"])
        for table in result.tables[:max_elements]:
            location = table.sheet_name or (f"Slide {table.slide_number}" if table.slide_number else f"Page {table.page_number}" if table.page_number else "")
            lines.append(f"- {table.table_id}: {table.title or 'Table'} {location}".strip())
    if result.figures:
        lines.extend(["", "## Figures"])
        for figure in result.figures[:max_elements]:
            location = f"Page {figure.page_number}" if figure.page_number else f"Slide {figure.slide_number}" if figure.slide_number else ""
            lines.append(f"- {figure.figure_id}: {figure.title or 'Figure'} {location}".strip())
    if result.elements:
        lines.extend(["", "## Element Preview"])
        for element in result.elements[:max_elements]:
            location = element.section_title or element.sheet_name or (f"Slide {element.slide_number}" if element.slide_number else f"Page {element.page_number}" if element.page_number else "")
            prefix = f"{element.element_id} [{element.element_type}]"
            if location:
                prefix += f" {location}"
            lines.append(f"- {prefix}: {element.text[:400]}")
    return "\n".join(lines).rstrip() + "\n"


def compare_to_markdown(result: DocumentCompareResult, *, max_deltas: int = 120) -> str:
    lines = [
        f"# Document Comparison: {result.left_document.title or result.left_document.doc_id} -> {result.right_document.title or result.right_document.doc_id}",
        "",
        "## Summary",
        f"- Mode: {result.compare_mode}",
        f"- Focus: {result.focus or 'none'}",
    ]
    for key, value in result.counts().items():
        lines.append(f"- {key}: {value}")
    if result.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {item}" for item in result.warnings)
    if result.deltas:
        lines.extend(["", "## Deltas"])
        for delta in result.deltas[:max_deltas]:
            lines.append(f"### {delta.delta_id} - {delta.change_type}")
            lines.append(delta.summary)
            if delta.left_text:
                lines.append(f"- Before: {delta.left_text[:700]}")
            if delta.right_text:
                lines.append(f"- After: {delta.right_text[:700]}")
    if result.changed_obligations:
        lines.extend(["", "## Changed Obligations"])
        for obligation in result.changed_obligations[:max_deltas]:
            lines.append(f"### {obligation.obligation_id} - {obligation.change_type} ({obligation.severity})")
            lines.append(f"- Modality: {obligation.modality}")
            if obligation.before_text:
                lines.append(f"- Before: {obligation.before_text[:700]}")
            if obligation.after_text:
                lines.append(f"- After: {obligation.after_text[:700]}")
            if obligation.rationale:
                lines.append(f"- Rationale: {obligation.rationale}")
    return "\n".join(lines).rstrip() + "\n"


def _obligations_csv(result: DocumentCompareResult) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "obligation_id",
            "change_type",
            "modality",
            "severity",
            "before_text",
            "after_text",
            "location",
            "rationale",
        ],
    )
    writer.writeheader()
    for item in result.changed_obligations:
        writer.writerow(
            {
                "obligation_id": item.obligation_id,
                "change_type": item.change_type,
                "modality": item.modality,
                "severity": item.severity,
                "before_text": item.before_text,
                "after_text": item.after_text,
                "location": json.dumps(item.location, sort_keys=True),
                "rationale": item.rationale,
            }
        )
    return buffer.getvalue()


def write_extract_artifacts(session: object, result: DocumentExtractResult) -> List[Dict[str, Any]]:
    stem = _safe_stem(result.document.title or result.document.doc_id, fallback="document")
    digest = (result.document.content_hash or result.document.doc_id or "extract")[:10].replace("/", "_")
    json_name = f"{stem}__document_extract__{digest}.json"
    md_name = f"{stem}__document_extract__{digest}.md"
    return [
        _write_workspace_text(session, json_name, json.dumps(result.to_dict(), indent=2, sort_keys=True)),
        _write_workspace_text(session, md_name, extract_to_markdown(result)),
    ]


def write_compare_artifacts(session: object, result: DocumentCompareResult) -> List[Dict[str, Any]]:
    left = _safe_stem(result.left_document.title or result.left_document.doc_id, fallback="left")
    right = _safe_stem(result.right_document.title or result.right_document.doc_id, fallback="right")
    stem = f"{left}__vs__{right}"[:72]
    json_name = f"{stem}__document_compare.json"
    md_name = f"{stem}__document_compare.md"
    artifacts = [
        _write_workspace_text(session, json_name, json.dumps(result.to_dict(), indent=2, sort_keys=True)),
        _write_workspace_text(session, md_name, compare_to_markdown(result)),
    ]
    if result.changed_obligations:
        csv_name = f"{stem}__changed_obligations.csv"
        artifacts.append(_write_workspace_text(session, csv_name, _obligations_csv(result)))
    return artifacts


def compact_extract_payload(
    result: DocumentExtractResult,
    *,
    artifacts: List[Dict[str, Any]] | None = None,
    output: str = "summary",
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "document": result.document.to_dict(),
        "counts": result.counts(),
        "warnings": list(result.warnings),
        "truncated": result.truncated,
        "section_hierarchy_preview": [section.to_dict() for section in result.sections[:20]],
        "table_inventory": [table.to_dict() for table in result.tables[:20]],
        "figure_inventory": [figure.to_dict() for figure in result.figures[:20]],
        "artifacts": list(artifacts or []),
    }
    if output == "json":
        payload["element_preview"] = [element.to_dict() for element in result.elements[:25]]
    elif output == "markdown":
        payload["markdown_preview"] = extract_to_markdown(result, max_elements=25)[:8000]
    return payload


def compact_compare_payload(
    result: DocumentCompareResult,
    *,
    artifacts: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    return {
        "left_document": result.left_document.to_dict(),
        "right_document": result.right_document.to_dict(),
        "compare_mode": result.compare_mode,
        "focus": result.focus,
        "counts": result.counts(),
        "redline_summary": [delta.summary for delta in result.deltas if delta.change_type != "unchanged"][:12],
        "clause_deltas": [delta.to_dict() for delta in result.deltas if delta.change_type != "unchanged"][:30],
        "changed_obligations": [item.to_dict() for item in result.changed_obligations[:30]],
        "warnings": list(result.warnings),
        "artifacts": list(artifacts or []),
    }


__all__ = [
    "compact_compare_payload",
    "compact_extract_payload",
    "compare_to_markdown",
    "extract_to_markdown",
    "write_compare_artifacts",
    "write_extract_artifacts",
]
