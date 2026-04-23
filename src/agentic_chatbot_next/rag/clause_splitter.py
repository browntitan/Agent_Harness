"""Clause-aware document splitter.

Splits structured documents (contracts, policy docs, termsets) at clause /
section / article boundaries rather than arbitrary character counts.

Falls back to RecursiveCharacterTextSplitter for oversized clauses.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from agentic_chatbot_next.rag.structure_detector import CLAUSE_PATTERN, PROCESS_FLOW_PATTERN, REQUIREMENT_PATTERN

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLAUSE_NUM_RE = re.compile(
    r"(?:Clause|CLAUSE|Section|SECTION|Article|ARTICLE)\s+([\d\.]+|[IVXLCDM]+)"
    r"|^(\d{1,3}(?:\.\d{1,3}){0,3})\.\s+[A-Z]",
    re.IGNORECASE,
)


def _extract_clause_number(header_line: str) -> Optional[str]:
    """Parse a header line and return a normalised clause number string.

    Examples:
        "Clause 3.2: Definitions" → "3.2"
        "Section 10"               → "10"
        "Article IV"               → "IV"
        "3.1.2 Scope"              → "3.1.2"
    Returns None if no clause number can be extracted.
    """
    m = _CLAUSE_NUM_RE.search(header_line.strip())
    if not m:
        return None
    # Group 1: named clause/section/article number; group 2: bare numeric prefix
    return (m.group(1) or m.group(2) or "").strip() or None


def _extract_section_title(header_line: str) -> str:
    """Return a clean section title from a header line."""
    # Strip leading clause number prefix and punctuation
    title = re.sub(
        r"^(?:Clause|Section|Article|CLAUSE|SECTION|ARTICLE)\s+[\d\.IVXLCDMivxlcdm]+\s*[:\-–]?\s*",
        "",
        header_line.strip(),
        flags=re.IGNORECASE,
    )
    # Also strip bare numeric prefixes like "3.2.1 "
    title = re.sub(r"^\d{1,3}(?:\.\d{1,3})*\.\s+", "", title)
    return title.strip()


def _tag_chunk_type(content: str) -> str:
    """Classify a chunk using the strongest structural signal available."""
    if PROCESS_FLOW_PATTERN.search(content):
        return "process_flow"
    if REQUIREMENT_PATTERN.search(content):
        return "requirement"
    return "clause"


# ---------------------------------------------------------------------------
# Main splitter
# ---------------------------------------------------------------------------

def clause_split(
    doc: Document,
    *,
    max_clause_chars: int = 1500,
    overlap_chars: int = 100,
) -> List[Document]:
    """Split a document at clause/section boundaries.

    Algorithm:
    1. Find all clause header positions using CLAUSE_PATTERN.
    2. Extract text between consecutive headers as individual clause chunks.
    3. Sub-split oversized clause bodies with RecursiveCharacterTextSplitter.
    4. Attach metadata: clause_number, section_title, chunk_type.

    Any text before the first header is returned as a 'header' chunk.

    Args:
        doc:              Source LangChain Document (must have page_content).
        max_clause_chars: Maximum characters per clause chunk before sub-splitting.
        overlap_chars:    Character overlap when sub-splitting an oversized clause.

    Returns:
        List of Documents with enriched metadata.
    """
    text = doc.page_content
    base_meta = dict(doc.metadata or {})

    # Find all clause header positions
    boundaries: List[Tuple[int, str]] = []  # (char_offset, header_line)
    for m in CLAUSE_PATTERN.finditer(text):
        # Walk back to the start of the line
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.start())
        if line_end == -1:
            line_end = len(text)
        header_line = text[line_start:line_end].strip()
        boundaries.append((line_start, header_line))

    # No boundaries detected — return the whole doc as a single 'general' chunk
    if not boundaries:
        return [_make_chunk(doc, text, base_meta, chunk_index=0, chunk_type="general")]

    chunks: List[Document] = []
    chunk_index = 0

    # Text before the first clause header
    pre_text = text[: boundaries[0][0]].strip()
    if pre_text:
        chunks.append(
            _make_chunk(
                doc, pre_text, base_meta,
                chunk_index=chunk_index,
                chunk_type="header",
            )
        )
        chunk_index += 1

    # Iterate clause segments
    for i, (start, header_line) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        clause_text = text[start:end].strip()

        clause_number = _extract_clause_number(header_line)
        section_title = _extract_section_title(header_line)
        chunk_type = _tag_chunk_type(clause_text)

        extra_meta = {
            "clause_number": clause_number,
            "section_title": section_title,
        }

        if len(clause_text) <= max_clause_chars:
            chunks.append(
                _make_chunk(
                    doc, clause_text, {**base_meta, **extra_meta},
                    chunk_index=chunk_index,
                    chunk_type=chunk_type,
                )
            )
            chunk_index += 1
        else:
            # Sub-split oversized clause
            sub_chunks = _sub_split(
                clause_text,
                max_clause_chars,
                overlap_chars,
                base_meta={**base_meta, **extra_meta},
                doc=doc,
                start_index=chunk_index,
                chunk_type=chunk_type,
            )
            chunks.extend(sub_chunks)
            chunk_index += len(sub_chunks)

    return chunks


def _make_chunk(
    doc: Document,
    content: str,
    meta: dict,
    *,
    chunk_index: int,
    chunk_type: str,
) -> Document:
    return Document(
        page_content=content,
        metadata={
            **meta,
            "chunk_index": chunk_index,
            "chunk_type": chunk_type,
        },
    )


def _sub_split(
    text: str,
    max_chars: int,
    overlap: int,
    *,
    base_meta: dict,
    doc: Document,
    start_index: int,
    chunk_type: str,
) -> List[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        add_start_index=True,
    )
    raw = splitter.split_text(text)
    result: List[Document] = []
    for i, part in enumerate(raw):
        # Re-tag each sub-chunk — a sub-clause might be a requirement
        sub_type = _tag_chunk_type(part) if chunk_type != "header" else chunk_type
        result.append(
            _make_chunk(
                doc, part, base_meta,
                chunk_index=start_index + i,
                chunk_type=sub_type,
            )
        )
    return result
