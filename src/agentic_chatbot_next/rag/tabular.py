from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from langchain_core.documents import Document

from agentic_chatbot_next.rag.fanout import TabularEvidenceResult, TabularEvidenceTask

_SPREADSHEET_TYPES = {"csv", "xls", "xlsx"}
_SPREADSHEET_SUFFIXES = {f".{item}" for item in _SPREADSHEET_TYPES}

_OPERATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "lookup",
        re.compile(
            r"\b(lookup|look\s+up|find|show|list|which|what|where|row|record|cell|value|"
            r"current|latest|approved|status|date|amount|price|cost|variance|milestone|supplier)\b",
            flags=re.I,
        ),
    ),
    (
        "filter",
        re.compile(r"\b(filter|matching|matches|where|with|for|only|exclude|include|greater|less|between)\b", flags=re.I),
    ),
    (
        "aggregate",
        re.compile(r"\b(total|sum|average|avg|mean|median|count|min|max|top|bottom|rank|percent|rate|ratio)\b", flags=re.I),
    ),
    (
        "profile",
        re.compile(
            r"\b(profile|summari[sz]e|summary|overview|columns?|schema|dtype|statistics|stats|"
            r"categorical|numeric|strings?|distribution|missing|nulls?)\b",
            flags=re.I,
        ),
    ),
    (
        "compare",
        re.compile(r"\b(compare|comparison|difference|delta|trend|correlation|relationship|versus|vs\.?)\b", flags=re.I),
    ),
)


def _metadata(doc: Document) -> Dict[str, Any]:
    return dict(getattr(doc, "metadata", {}) or {})


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _unique_strings(items: Iterable[Any], *, limit: int = 8) -> List[str]:
    seen: set[str] = set()
    values: List[str] = []
    for item in items:
        clean = _as_text(item)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        values.append(clean)
        if len(values) >= limit:
            break
    return values


def _safe_id(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return clean.strip("_") or "tabular"


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _doc_suffix(metadata: Dict[str, Any]) -> str:
    for key in ("title", "source_path", "source_display_path"):
        text = _as_text(metadata.get(key))
        if text:
            suffix = Path(text.split("?", 1)[0]).suffix.lower()
            if suffix:
                return suffix
    return ""


def spreadsheet_file_type(metadata: Dict[str, Any]) -> str:
    file_type = _as_text(metadata.get("file_type")).lstrip(".").lower()
    if file_type in _SPREADSHEET_TYPES:
        return file_type
    suffix = _doc_suffix(metadata).lstrip(".").lower()
    return suffix if suffix in _SPREADSHEET_TYPES else file_type


def is_spreadsheet_document(doc: Document) -> bool:
    metadata = _metadata(doc)
    if _as_text(metadata.get("chunk_type")) == "tabular_analysis":
        return False
    file_type = spreadsheet_file_type(metadata)
    if file_type in _SPREADSHEET_TYPES:
        return True
    return _doc_suffix(metadata) in _SPREADSHEET_SUFFIXES


def requested_tabular_operations(query: str) -> List[str]:
    operations = [name for name, pattern in _OPERATION_PATTERNS if pattern.search(str(query or ""))]
    return operations or []


def query_needs_tabular_handoff(query: str) -> bool:
    return bool(requested_tabular_operations(query))


def _row_hint(metadata: Dict[str, Any]) -> Dict[str, Any]:
    hint: Dict[str, Any] = {}
    for key in ("sheet_name", "cell_range"):
        value = _as_text(metadata.get(key))
        if value:
            hint[key] = value
    for key in ("row_start", "row_end"):
        value = _coerce_int(metadata.get(key))
        if value is not None:
            hint[key] = value
    return hint


def plan_tabular_evidence_tasks(
    query: str,
    docs: Sequence[Document],
    *,
    max_tasks: int = 2,
) -> List[TabularEvidenceTask]:
    operations = requested_tabular_operations(query)
    if not operations:
        return []

    grouped: Dict[str, Dict[str, Any]] = {}
    for doc in docs:
        if not is_spreadsheet_document(doc):
            continue
        metadata = _metadata(doc)
        doc_id = _as_text(metadata.get("doc_id")) or _as_text(metadata.get("title")) or _as_text(metadata.get("source_path"))
        if not doc_id:
            continue
        item = grouped.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "title": _as_text(metadata.get("title")) or Path(_as_text(metadata.get("source_path"))).name,
                "source_path": _as_text(metadata.get("source_path")) or _as_text(metadata.get("source_display_path")),
                "file_type": spreadsheet_file_type(metadata),
                "sheet_hints": [],
                "cell_ranges": [],
                "row_hints": [],
            },
        )
        if not item["title"]:
            item["title"] = _as_text(metadata.get("title")) or Path(_as_text(metadata.get("source_path"))).name
        if not item["source_path"]:
            item["source_path"] = _as_text(metadata.get("source_path")) or _as_text(metadata.get("source_display_path"))
        if not item["file_type"]:
            item["file_type"] = spreadsheet_file_type(metadata)
        item["sheet_hints"].append(metadata.get("sheet_name"))
        item["cell_ranges"].append(metadata.get("cell_range"))
        hint = _row_hint(metadata)
        if hint:
            item["row_hints"].append(hint)

    tasks: List[TabularEvidenceTask] = []
    for index, item in enumerate(list(grouped.values())[: max(0, int(max_tasks))], start=1):
        row_hints: List[Dict[str, Any]] = []
        seen_row_hints: set[str] = set()
        for hint in item["row_hints"]:
            key = json.dumps(hint, sort_keys=True)
            if key in seen_row_hints:
                continue
            seen_row_hints.add(key)
            row_hints.append(dict(hint))
            if len(row_hints) >= 8:
                break
        tasks.append(
            TabularEvidenceTask(
                task_id=f"tabular_{index}",
                query=str(query or ""),
                doc_id=str(item["doc_id"]),
                title=str(item["title"] or item["doc_id"]),
                source_path=str(item["source_path"] or ""),
                file_type=str(item["file_type"] or ""),
                sheet_hints=_unique_strings(item["sheet_hints"]),
                cell_ranges=_unique_strings(item["cell_ranges"]),
                row_hints=row_hints,
                requested_operations=list(operations),
            )
        )
    return tasks


def _render_source_ref(ref: Dict[str, Any], task: TabularEvidenceTask) -> str:
    sheet = _as_text(ref.get("sheet_name"))
    cell_range = _as_text(ref.get("cell_range"))
    row_start = ref.get("row_start")
    row_end = ref.get("row_end")
    columns = [str(item) for item in (ref.get("columns") or []) if str(item)]
    parts: List[str] = []
    if sheet:
        parts.append(f"sheet={sheet}")
    if row_start is not None:
        if row_end is not None and str(row_end) != str(row_start):
            parts.append(f"rows={row_start}-{row_end}")
        else:
            parts.append(f"row={row_start}")
    if cell_range:
        parts.append(f"cells={cell_range}")
    if columns:
        parts.append("columns=" + ", ".join(columns[:8]))
    if not parts:
        parts.append(task.title)
    return "; ".join(parts)


def _render_finding(finding: Dict[str, Any]) -> str:
    if not finding:
        return ""
    for key in ("text", "summary", "finding", "answer", "value"):
        value = finding.get(key)
        if value not in (None, ""):
            return str(value)
    compact = {
        str(key): value
        for key, value in finding.items()
        if key not in {"source_refs", "source_ref"} and value not in (None, "", [], {})
    }
    return json.dumps(compact, ensure_ascii=False, sort_keys=True) if compact else ""


def tabular_evidence_results_to_documents(
    results: Sequence[TabularEvidenceResult],
    tasks: Sequence[TabularEvidenceTask],
) -> List[Document]:
    task_by_id = {task.task_id: task for task in tasks}
    documents: List[Document] = []
    for result in results:
        task = task_by_id.get(result.task_id)
        if task is None:
            continue
        if str(result.status or "").lower() not in {"ok", "partial"}:
            continue
        if not result.summary and not result.findings:
            continue
        source_refs = [dict(item) for item in result.source_refs if isinstance(item, dict)]
        if not source_refs:
            source_refs = [
                {
                    "doc_id": task.doc_id,
                    "title": task.title,
                    "sheet_name": task.sheet_hints[0] if task.sheet_hints else "",
                    "cell_range": task.cell_ranges[0] if task.cell_ranges else "",
                    "columns": [],
                }
            ]

        findings = [_render_finding(item) for item in result.findings[:8]]
        findings = [item for item in findings if item]
        source_lines = [_render_source_ref(ref, task) for ref in source_refs[:6]]
        lines = [
            f"Tabular evidence for {task.title}.",
            f"Question: {task.query}",
        ]
        if result.summary:
            lines.append(f"Summary: {result.summary}")
        if result.operations:
            lines.append("Operations: " + ", ".join(result.operations[:8]))
        if findings:
            lines.append("Findings: " + " | ".join(findings))
        if source_lines:
            lines.append("Source refs: " + " | ".join(source_lines))
        if result.warnings:
            lines.append("Warnings: " + ", ".join(result.warnings[:4]))
        page_content = " ".join(line for line in lines if line).strip()

        for index, ref in enumerate(source_refs[:4], start=1):
            doc_id = _as_text(ref.get("doc_id")) or task.doc_id
            title = _as_text(ref.get("title")) or task.title
            chunk_id = f"{doc_id}#tabular_{_safe_id(result.task_id)}_{index:02d}"
            metadata: Dict[str, Any] = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "title": title,
                "source_type": "tabular_analysis",
                "source_path": _as_text(ref.get("source_path")) or task.source_path,
                "file_type": task.file_type,
                "chunk_type": "tabular_analysis",
                "is_synthetic_evidence": True,
                "tabular_task_id": result.task_id,
                "tabular_confidence": float(result.confidence or 0.0),
                "columns": [str(item) for item in (ref.get("columns") or []) if str(item)],
            }
            for key in ("sheet_name", "cell_range"):
                value = _as_text(ref.get(key))
                if value:
                    metadata[key] = value
            for key in ("row_start", "row_end"):
                value = _coerce_int(ref.get(key))
                if value is not None:
                    metadata[key] = value
            documents.append(Document(page_content=page_content, metadata=metadata))
    return documents
