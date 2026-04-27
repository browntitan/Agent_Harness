from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from langchain_core.documents import Document

from agentic_chatbot_next.rag.structure_detector import (
    CLAUSE_PATTERN,
    PROCESS_FLOW_PATTERN,
    REQUIREMENT_PATTERN,
    StructureAnalysis,
)


INDEX_METADATA_VERSION = "document_index_metadata_v1"
SUPPORTED_METADATA_PROFILES = {"auto", "deterministic", "basic", "off"}

_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
_NUMBERED_HEADING_RE = re.compile(r"^\s*(?P<num>\d{1,3}(?:\.\d{1,3}){0,4})\.?\s+(?P<title>[A-Z][^\n]{2,180})")
_CLAUSE_NUM_RE = re.compile(
    r"(?:Clause|Section|Article)\s+([\d\.]+|[IVXLCDM]+)"
    r"|^(\d{1,3}(?:\.\d{1,3}){0,4})\.?\s+[A-Z]",
    re.IGNORECASE,
)
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9&/-]+(?:[ \t]+[A-Z][A-Za-z0-9&/-]+){1,5}\b")
_TABLE_HINT_RE = re.compile(r"\|.+\||\b(table|columns?|rows?|worksheet|spreadsheet)\b", re.IGNORECASE)
_CHECKLIST_HINT_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+\[[ xX]\]", re.MULTILINE)


@dataclass(frozen=True)
class DocumentIndexMetadata:
    extractor_version: str
    metadata_profile: str
    metadata_enrichment: str
    doc_structure_type: str
    confidence: float
    parser_chain: list[str] = field(default_factory=list)
    structural_flags: dict[str, Any] = field(default_factory=dict)
    outline: list[dict[str, Any]] = field(default_factory=list)
    sheets: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChunkIndexMetadata:
    extractor_version: str
    metadata_profile: str
    document_type: str
    chunk_id: str
    chunk_index: int
    chunk_type: str
    tags: list[str] = field(default_factory=list)
    section_path: list[str] = field(default_factory=list)
    location: dict[str, Any] = field(default_factory=dict)
    entities: list[str] = field(default_factory=list)
    confidence: float = 0.5
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusIndexSummary:
    extractor_version: str
    metadata_profile: str
    document_count: int = 0
    chunk_count: int = 0
    structure_type_counts: dict[str, int] = field(default_factory=dict)
    tag_counts: dict[str, int] = field(default_factory=dict)
    parser_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_metadata_profile(profile: str) -> str:
    normalized = str(profile or "auto").strip().lower()
    if normalized in {"none", "disabled", "disable"}:
        return "off"
    return normalized if normalized in SUPPORTED_METADATA_PROFILES else "auto"


def build_document_index_metadata(
    *,
    path: Path,
    raw_docs: Sequence[Document],
    structure: StructureAnalysis,
    metadata_profile: str = "auto",
    metadata_enrichment: str = "deterministic",
    source_metadata: dict[str, Any] | None = None,
) -> DocumentIndexMetadata:
    profile = normalize_metadata_profile(metadata_profile)
    full_text = "\n\n".join(str(doc.page_content or "") for doc in raw_docs)
    parser_chain = _parser_chain(raw_docs)
    outline = [] if profile == "off" else _extract_outline(full_text)
    sheets = [] if profile == "off" else _extract_sheet_summaries(raw_docs)
    entities = [] if profile in {"basic", "off"} else _extract_entities(full_text)
    tags = _document_tags(path, full_text, structure, sheets=sheets, profile=profile)
    warnings: list[str] = []
    if not full_text.strip():
        warnings.append("No extractable text was found.")
    if not parser_chain:
        warnings.append("Parser provenance was not reported by the loader.")

    stats = {
        "char_count": len(full_text),
        "line_count": len([line for line in full_text.splitlines() if line.strip()]),
        "source_document_count": len(raw_docs),
        "outline_count": len(outline),
        "sheet_count": len(sheets),
        "requirement_signal_count": len(REQUIREMENT_PATTERN.findall(full_text)),
        "process_flow_signal_count": len(PROCESS_FLOW_PATTERN.findall(full_text)),
        "clause_density": round(float(structure.clause_density or 0.0), 4),
    }
    source_metadata = dict(source_metadata or {})
    if source_metadata.get("mime_type"):
        stats["mime_type"] = str(source_metadata.get("mime_type") or "")

    return DocumentIndexMetadata(
        extractor_version=INDEX_METADATA_VERSION,
        metadata_profile=profile,
        metadata_enrichment=str(metadata_enrichment or "deterministic").strip().lower() or "deterministic",
        doc_structure_type=str(structure.doc_structure_type or "general"),
        confidence=_document_confidence(structure, outline=outline, sheets=sheets, parser_chain=parser_chain),
        parser_chain=parser_chain,
        structural_flags={
            "has_clauses": bool(structure.has_clauses),
            "has_requirements": bool(structure.has_requirements),
            "has_process_flow": bool(structure.has_process_flow),
        },
        outline=outline[:200],
        sheets=sheets,
        tags=tags,
        entities=entities,
        warnings=warnings,
        stats=stats,
    )


def build_chunk_index_metadata(
    chunk: Document,
    *,
    chunk_id: str,
    chunk_index: int,
    document_metadata: DocumentIndexMetadata | dict[str, Any] | None,
) -> ChunkIndexMetadata:
    source = document_metadata.to_dict() if isinstance(document_metadata, DocumentIndexMetadata) else dict(document_metadata or {})
    metadata = dict(chunk.metadata or {})
    content = str(chunk.page_content or "")
    chunk_type = str(metadata.get("chunk_type") or "general")
    tags = _chunk_tags(content, metadata, document_tags=list(source.get("tags") or []))
    section_title = str(metadata.get("section_title") or "").strip()
    section_path = [section_title] if section_title else []
    location = {
        key: value
        for key, value in {
            "page_number": metadata.get("page"),
            "clause_number": metadata.get("clause_number"),
            "section_title": section_title,
            "sheet_name": metadata.get("sheet_name"),
            "row_start": metadata.get("row_start"),
            "row_end": metadata.get("row_end"),
            "cell_range": metadata.get("cell_range"),
        }.items()
        if value not in ("", None)
    }
    confidence = 0.55
    if section_title or metadata.get("clause_number") or metadata.get("sheet_name"):
        confidence += 0.2
    if tags:
        confidence += 0.1
    return ChunkIndexMetadata(
        extractor_version=str(source.get("extractor_version") or INDEX_METADATA_VERSION),
        metadata_profile=str(source.get("metadata_profile") or "auto"),
        document_type=str(source.get("doc_structure_type") or "general"),
        chunk_id=chunk_id,
        chunk_index=int(chunk_index),
        chunk_type=chunk_type,
        tags=tags,
        section_path=section_path,
        location=location,
        entities=_extract_entities(content, limit=6),
        confidence=round(min(confidence, 0.95), 2),
        stats={
            "char_count": len(content),
            "word_count": len(content.split()),
            "has_requirement_signal": bool(REQUIREMENT_PATTERN.search(content)),
            "has_process_flow_signal": bool(PROCESS_FLOW_PATTERN.search(content)),
        },
    )


def summarize_index_metadata(
    documents: Iterable[DocumentIndexMetadata | dict[str, Any]],
    chunks: Iterable[ChunkIndexMetadata | dict[str, Any]] = (),
    *,
    metadata_profile: str = "auto",
) -> CorpusIndexSummary:
    doc_items = [item.to_dict() if isinstance(item, DocumentIndexMetadata) else dict(item or {}) for item in documents]
    chunk_items = [item.to_dict() if isinstance(item, ChunkIndexMetadata) else dict(item or {}) for item in chunks]
    structure_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    parser_counts: Counter[str] = Counter()
    warnings: list[str] = []

    for item in doc_items:
        structure_counts[str(item.get("doc_structure_type") or "general")] += 1
        for tag in list(item.get("tags") or []):
            tag_counts[str(tag)] += 1
        for parser in list(item.get("parser_chain") or []):
            parser_counts[str(parser)] += 1
        for warning in list(item.get("warnings") or []):
            text = str(warning or "").strip()
            if text and text not in warnings:
                warnings.append(text)
    for item in chunk_items:
        for tag in list(item.get("tags") or []):
            tag_counts[str(tag)] += 1

    return CorpusIndexSummary(
        extractor_version=INDEX_METADATA_VERSION,
        metadata_profile=normalize_metadata_profile(metadata_profile),
        document_count=len(doc_items),
        chunk_count=len(chunk_items),
        structure_type_counts=dict(sorted(structure_counts.items())),
        tag_counts=dict(sorted(tag_counts.items())),
        parser_counts=dict(sorted(parser_counts.items())),
        warnings=warnings[:20],
    )


def _parser_chain(raw_docs: Sequence[Document]) -> list[str]:
    parsers: list[str] = []
    for doc in raw_docs:
        metadata = dict(doc.metadata or {})
        parser = str(metadata.get("parser") or "").strip()
        if not parser and metadata.get("is_prechunked"):
            parser = "prechunked_loader"
        if not parser and metadata.get("source"):
            parser = "langchain_loader"
        if parser and parser not in parsers:
            parsers.append(parser)
    return parsers


def _extract_outline(text: str) -> list[dict[str, Any]]:
    outline: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(str(text or "").splitlines(), start=1):
        line = raw_line.strip()
        if not line or len(line) > 220:
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            outline.append({"level": len(heading.group(1)), "title": heading.group(2).strip(), "line": line_number})
            continue
        if CLAUSE_PATTERN.match(line):
            outline.append(
                {
                    "level": 1,
                    "title": _section_title(line),
                    "line": line_number,
                    "clause_number": _clause_number(line),
                }
            )
            continue
        numbered = _NUMBERED_HEADING_RE.match(line)
        if numbered:
            outline.append(
                {
                    "level": max(1, str(numbered.group("num")).count(".") + 1),
                    "title": numbered.group("title").strip(),
                    "line": line_number,
                    "clause_number": numbered.group("num").strip("."),
                }
            )
        if len(outline) >= 200:
            break
    return outline


def _extract_sheet_summaries(raw_docs: Sequence[Document]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for doc in raw_docs:
        metadata = dict(doc.metadata or {})
        sheet_name = str(metadata.get("sheet_name") or "").strip()
        if not sheet_name:
            continue
        item = grouped.setdefault(
            sheet_name,
            {
                "sheet_name": sheet_name,
                "chunk_count": 0,
                "row_start": None,
                "row_end": None,
                "cell_ranges": [],
            },
        )
        item["chunk_count"] = int(item["chunk_count"]) + 1
        row_start = metadata.get("row_start")
        row_end = metadata.get("row_end")
        if isinstance(row_start, int):
            item["row_start"] = row_start if item["row_start"] is None else min(int(item["row_start"]), row_start)
        if isinstance(row_end, int):
            item["row_end"] = row_end if item["row_end"] is None else max(int(item["row_end"]), row_end)
        cell_range = str(metadata.get("cell_range") or "").strip()
        if cell_range and cell_range not in item["cell_ranges"]:
            item["cell_ranges"].append(cell_range)
    return list(grouped.values())


def _document_tags(
    path: Path,
    text: str,
    structure: StructureAnalysis,
    *,
    sheets: Sequence[dict[str, Any]],
    profile: str,
) -> list[str]:
    tags: list[str] = []
    if profile == "off":
        return tags
    _add_tag(tags, str(structure.doc_structure_type or "general"))
    if structure.has_requirements:
        _add_tag(tags, "requirements")
    if structure.has_process_flow:
        _add_tag(tags, "process_flow")
    if structure.has_clauses:
        _add_tag(tags, "structured")
    if sheets or path.suffix.lower() in {".xls", ".xlsx", ".csv", ".tsv"}:
        _add_tag(tags, "tabular")
    if _TABLE_HINT_RE.search(text):
        _add_tag(tags, "tables")
    if _CHECKLIST_HINT_RE.search(text):
        _add_tag(tags, "checklist")
    return tags


def _chunk_tags(content: str, metadata: dict[str, Any], *, document_tags: Sequence[str]) -> list[str]:
    tags: list[str] = []
    chunk_type = str(metadata.get("chunk_type") or "").strip()
    if chunk_type:
        _add_tag(tags, chunk_type)
    if REQUIREMENT_PATTERN.search(content):
        _add_tag(tags, "requirements")
    if PROCESS_FLOW_PATTERN.search(content):
        _add_tag(tags, "process_flow")
    if metadata.get("sheet_name"):
        _add_tag(tags, "tabular")
    if metadata.get("clause_number"):
        _add_tag(tags, "clause")
    for tag in document_tags:
        if tag in {"requirements", "process_flow", "tabular"}:
            _add_tag(tags, tag)
    return tags


def _extract_entities(text: str, *, limit: int = 12) -> list[str]:
    counts: Counter[str] = Counter()
    for match in _ENTITY_RE.finditer(str(text or "")):
        value = " ".join(match.group(0).split())
        if len(value) < 4 or value.lower() in {"table of", "section"}:
            continue
        counts[value] += 1
    return [value for value, _ in counts.most_common(limit)]


def _document_confidence(
    structure: StructureAnalysis,
    *,
    outline: Sequence[dict[str, Any]],
    sheets: Sequence[dict[str, Any]],
    parser_chain: Sequence[str],
) -> float:
    confidence = 0.55
    if parser_chain:
        confidence += 0.1
    if outline:
        confidence += 0.15
    if sheets:
        confidence += 0.15
    if structure.has_clauses or structure.has_requirements or structure.has_process_flow:
        confidence += 0.15
    return round(min(confidence, 0.98), 2)


def _clause_number(header_line: str) -> str:
    match = _CLAUSE_NUM_RE.search(header_line.strip())
    if not match:
        return ""
    return str(match.group(1) or match.group(2) or "").strip()


def _section_title(header_line: str) -> str:
    title = re.sub(
        r"^(?:Clause|Section|Article)\s+[\d\.IVXLCDMivxlcdm]+\s*[:\-–]?\s*",
        "",
        header_line.strip(),
        flags=re.IGNORECASE,
    )
    title = re.sub(r"^\d{1,3}(?:\.\d{1,3})*\.?\s+", "", title)
    return title.strip() or header_line.strip()


def _add_tag(tags: list[str], tag: str) -> None:
    normalized = str(tag or "").strip().lower().replace(" ", "_")
    if normalized and normalized not in tags:
        tags.append(normalized)


__all__ = [
    "INDEX_METADATA_VERSION",
    "ChunkIndexMetadata",
    "CorpusIndexSummary",
    "DocumentIndexMetadata",
    "build_chunk_index_metadata",
    "build_document_index_metadata",
    "normalize_metadata_profile",
    "summarize_index_metadata",
]
