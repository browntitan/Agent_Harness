from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

from agentic_chatbot_next.persistence.postgres.chunks import ChunkRecord
from agentic_chatbot_next.persistence.postgres.documents import DocumentRecord
from agentic_chatbot_next.persistence.postgres.requirements import RequirementStatementRecord

REQUIREMENT_EXTRACTOR_VERSION = "requirements_v1"
SUPPORTED_REQUIREMENTS_FILE_TYPES = {"pdf", "docx", "md", "txt"}
STRICT_SHALL_MODE = "strict_shall"
MANDATORY_MODE = "mandatory"
LEGAL_CLAUSE_MODE = "legal_clause"
SUPPORTED_REQUIREMENT_MODES = {MANDATORY_MODE, STRICT_SHALL_MODE, LEGAL_CLAUSE_MODE}

_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|[A-Za-z][.)])\s+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=(?:[A-Z0-9(]|[-*•]))")
_TOC_FRAGMENT_RE = re.compile(r"\.{4,}\s*\d+\s*$")
_HEADING_LIKE_RE = re.compile(r"^[A-Z][A-Za-z0-9 /&()_-]{0,100}$")
_EXAMPLE_PREFIX_RE = re.compile(r"^\s*(?:example|note|notes|e\.g\.)[:\s-]", re.IGNORECASE)

_MODALITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("shall_not", re.compile(r"\bshall\s+not\b", re.IGNORECASE)),
    ("must_not", re.compile(r"\bmust\s+not\b", re.IGNORECASE)),
    ("required_to", re.compile(r"\b(?:is|are)\s+required\s+to\b", re.IGNORECASE)),
    (
        "prohibited",
        re.compile(
            r"\b(?:is|are)\s+prohibited\s+from\b"
            r"|\b(?:is|are)\s+not\s+permitted\s+to\b"
            r"|\bmay\s+not\b",
            re.IGNORECASE,
        ),
    ),
    ("shall", re.compile(r"\bshall\b", re.IGNORECASE)),
    ("must", re.compile(r"\bmust\b", re.IGNORECASE)),
)

_STRICT_MODALITIES = {"shall", "shall_not"}
_LEGAL_MODALITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    *_MODALITY_PATTERNS,
    (
        "required",
        re.compile(r"\b(?:is|are|be)\s+required\b", re.IGNORECASE),
    ),
    (
        "responsible_for",
        re.compile(r"\b(?:is|are|be)\s+responsible\s+for\b", re.IGNORECASE),
    ),
    (
        "agrees_to",
        re.compile(r"\b(?:agrees|agree)\s+to\b", re.IGNORECASE),
    ),
    (
        "will",
        re.compile(
            r"\b(?:contractor|subcontractor|offeror|supplier|vendor|provider|agency|government)\s+will\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class TextSlice:
    text: str
    start: int
    end: int

    def trimmed(self) -> "TextSlice":
        raw = self.text
        if not raw:
            return self
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        start = self.start + leading
        end = self.end - trailing
        return TextSlice(text=raw.strip(), start=max(self.start, start), end=max(start, end))


def normalize_requirement_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    return normalized if normalized in SUPPORTED_REQUIREMENT_MODES else MANDATORY_MODE


def requirement_modalities_for_mode(mode: str) -> tuple[str, ...]:
    normalized = normalize_requirement_mode(mode)
    if normalized == STRICT_SHALL_MODE:
        return tuple(sorted(_STRICT_MODALITIES))
    if normalized == LEGAL_CLAUSE_MODE:
        return tuple(dict.fromkeys(name for name, _ in _LEGAL_MODALITY_PATTERNS))
    return tuple(pattern[0] for pattern in _MODALITY_PATTERNS)


def supports_requirements_extraction(file_type: str) -> bool:
    return str(file_type or "").strip().lower() in SUPPORTED_REQUIREMENTS_FILE_TYPES


def normalize_requirement_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def format_requirement_location(record: RequirementStatementRecord | dict[str, Any]) -> str:
    payload = record if isinstance(record, dict) else record.__dict__
    parts: List[str] = []
    page_number = payload.get("page_number")
    clause_number = str(payload.get("clause_number") or "").strip()
    section_title = str(payload.get("section_title") or "").strip()
    chunk_index = int(payload.get("chunk_index") or 0)
    char_start = int(payload.get("char_start") or 0)
    if page_number:
        parts.append(f"Page {page_number}")
    if clause_number:
        parts.append(f"Clause {clause_number}")
    if section_title:
        parts.append(section_title)
    if not parts:
        parts.append(f"Chunk {chunk_index}")
    parts.append(f"Offset {char_start}")
    return " · ".join(parts)


def build_requirement_statement_records(
    document: DocumentRecord,
    chunks: Sequence[ChunkRecord],
    *,
    mode: str = MANDATORY_MODE,
) -> List[RequirementStatementRecord]:
    normalized_mode = normalize_requirement_mode(mode)
    statements: List[RequirementStatementRecord] = []
    next_index = 0
    for chunk in sorted(chunks, key=lambda item: int(getattr(item, "chunk_index", 0) or 0)):
        for statement in _extract_chunk_statement_slices(chunk, mode=normalized_mode):
            requirement_id = f"{document.doc_id}#req{next_index:05d}"
            modality = _detect_modality(statement.text, mode=normalized_mode)
            if not modality:
                continue
            statements.append(
                RequirementStatementRecord(
                    requirement_id=requirement_id,
                    doc_id=document.doc_id,
                    tenant_id=document.tenant_id,
                    collection_id=document.collection_id,
                    source_type=document.source_type,
                    document_title=document.title,
                    statement_index=next_index,
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.chunk_index,
                    statement_text=normalize_requirement_text(statement.text),
                    normalized_statement_text=normalize_requirement_text(statement.text).casefold(),
                    modality=modality,
                    page_number=chunk.page_number,
                    clause_number=str(chunk.clause_number or ""),
                    section_title=str(chunk.section_title or ""),
                    char_start=int(statement.start),
                    char_end=int(statement.end),
                    multi_requirement=_count_mandatory_operators(statement.text, mode=normalized_mode) > 1,
                    extractor_version=REQUIREMENT_EXTRACTOR_VERSION,
                    extractor_mode=normalized_mode,
                )
            )
            next_index += 1
    return statements


def _extract_chunk_statement_slices(chunk: ChunkRecord, *, mode: str) -> List[TextSlice]:
    content = str(chunk.content or "")
    if not content.strip():
        return []
    extracted: List[TextSlice] = []
    for block in _split_blocks(content):
        trimmed_block = block.trimmed()
        if not trimmed_block.text or _should_skip_candidate(trimmed_block.text):
            continue
        units = _split_list_units(trimmed_block)
        if not units:
            units = [trimmed_block]
        for unit in units:
            trimmed_unit = unit.trimmed()
            if not trimmed_unit.text or _should_skip_candidate(trimmed_unit.text):
                continue
            sentences = [
                item.trimmed()
                for item in _split_sentences(trimmed_unit)
                if item.trimmed().text
            ]
            matches = [
                sentence
                for sentence in sentences
                if _detect_modality(sentence.text, mode=mode)
                and not _should_skip_candidate(sentence.text)
            ]
            if matches:
                extracted.extend(matches)
                continue
            if _detect_modality(trimmed_unit.text, mode=mode):
                extracted.append(trimmed_unit)
    deduped: List[TextSlice] = []
    seen: set[tuple[int, int]] = set()
    for item in extracted:
        key = (item.start, item.end)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _split_blocks(text: str) -> List[TextSlice]:
    cursor = 0
    blocks: List[TextSlice] = []
    for match in re.finditer(r"\n\s*\n+", text):
        if match.start() > cursor:
            blocks.append(TextSlice(text=text[cursor:match.start()], start=cursor, end=match.start()))
        cursor = match.end()
    if cursor < len(text):
        blocks.append(TextSlice(text=text[cursor:], start=cursor, end=len(text)))
    return blocks


def _split_list_units(block: TextSlice) -> List[TextSlice]:
    lines = block.text.splitlines(keepends=True)
    if not lines:
        return []
    units: List[TextSlice] = []
    unit_start = 0
    cursor = 0
    saw_list = False
    for index, line in enumerate(lines):
        if index > 0 and _LIST_ITEM_RE.match(line):
            saw_list = True
            units.append(
                TextSlice(
                    text=block.text[unit_start:cursor],
                    start=block.start + unit_start,
                    end=block.start + cursor,
                )
            )
            unit_start = cursor
        cursor += len(line)
    if cursor > unit_start:
        units.append(
            TextSlice(
                text=block.text[unit_start:cursor],
                start=block.start + unit_start,
                end=block.start + cursor,
            )
        )
    return units if saw_list else []


def _split_sentences(unit: TextSlice) -> List[TextSlice]:
    boundaries = [0]
    for match in _SENTENCE_BOUNDARY_RE.finditer(unit.text):
        boundaries.append(match.start())
    boundaries.append(len(unit.text))
    slices: List[TextSlice] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        slices.append(
            TextSlice(
                text=unit.text[start:end],
                start=unit.start + start,
                end=unit.start + end,
            )
        )
    return slices


def _should_skip_candidate(text: str) -> bool:
    normalized = normalize_requirement_text(text)
    if not normalized:
        return True
    lowered = normalized.casefold()
    if _TOC_FRAGMENT_RE.search(normalized):
        return True
    if _EXAMPLE_PREFIX_RE.match(normalized):
        return True
    if len(normalized) <= 80 and _HEADING_LIKE_RE.match(normalized) and not any(char in normalized for char in ".;:!?"):
        return True
    if lowered.startswith(("appendix ", "table ", "figure ", "references", "bibliography")) and not _has_any_modality(normalized):
        return True
    return False


def _has_any_modality(text: str) -> bool:
    return any(pattern.search(text) for _, pattern in _MODALITY_PATTERNS)


def _patterns_for_mode(mode: str) -> tuple[tuple[str, re.Pattern[str]], ...]:
    normalized = normalize_requirement_mode(mode)
    if normalized == LEGAL_CLAUSE_MODE:
        return _LEGAL_MODALITY_PATTERNS
    return _MODALITY_PATTERNS


def _detect_modality(text: str, *, mode: str) -> str:
    allowed = set(requirement_modalities_for_mode(mode))
    best_name = ""
    best_start: int | None = None
    for name, pattern in _patterns_for_mode(mode):
        if name not in allowed:
            continue
        match = pattern.search(text)
        if match is None:
            continue
        if best_start is None or match.start() < best_start:
            best_name = name
            best_start = match.start()
    return best_name


def _count_mandatory_operators(text: str, *, mode: str = MANDATORY_MODE) -> int:
    count = 0
    for _, pattern in _patterns_for_mode(mode):
        count += len(list(pattern.finditer(text)))
    return count


__all__ = [
    "MANDATORY_MODE",
    "LEGAL_CLAUSE_MODE",
    "REQUIREMENT_EXTRACTOR_VERSION",
    "STRICT_SHALL_MODE",
    "SUPPORTED_REQUIREMENT_MODES",
    "SUPPORTED_REQUIREMENTS_FILE_TYPES",
    "build_requirement_statement_records",
    "format_requirement_location",
    "normalize_requirement_mode",
    "normalize_requirement_text",
    "requirement_modalities_for_mode",
    "supports_requirements_extraction",
]
