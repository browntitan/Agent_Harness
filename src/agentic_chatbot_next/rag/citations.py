from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from langchain_core.documents import Document

from agentic_chatbot_next.contracts.rag import Citation


def _citation_value(citation: Any, key: str) -> str:
    if isinstance(citation, Citation):
        return str(getattr(citation, key, "") or "").strip()
    if isinstance(citation, dict):
        return str(citation.get(key) or "").strip()
    return str(getattr(citation, key, "") or "").strip()


def _source_basename(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    clean = text.split("?", 1)[0].rstrip("/").replace("\\", "/")
    try:
        return Path(clean).name
    except Exception:
        return clean.rsplit("/", 1)[-1]


def citation_display_label(citation: Any) -> str:
    citation_id = _citation_value(citation, "citation_id")
    for candidate in (
        _citation_value(citation, "title"),
        _source_basename(_citation_value(citation, "source_path")),
        _citation_value(citation, "doc_id"),
        citation_id,
    ):
        if candidate and candidate != citation_id:
            return candidate
    return citation_id or "source"


def replace_inline_citation_ids(
    text: str,
    citations: Sequence[Any],
    *,
    used_citation_ids: Sequence[str] | None = None,
    link_renderer: Callable[[str, str], str] | None = None,
) -> str:
    """Replace raw inline citation ids with user-facing source labels."""

    citation_by_id = {
        _citation_value(citation, "citation_id"): citation
        for citation in citations
        if _citation_value(citation, "citation_id")
    }
    if not citation_by_id:
        return str(text or "")
    used = {str(item) for item in (used_citation_ids or []) if str(item)}
    if used:
        citation_by_id = {
            citation_id: citation
            for citation_id, citation in citation_by_id.items()
            if citation_id in used
        }
    if not citation_by_id:
        return str(text or "")

    def _render(citation: Any) -> str:
        label = citation_display_label(citation)
        url = _citation_value(citation, "url")
        if link_renderer is not None:
            return link_renderer(label, url)
        return label

    token_re = re.compile(r"(\s*(?:[,;]|\band\b)\s*)")

    def _replace_group(match: re.Match[str]) -> str:
        inner = match.group(1)
        parts = token_re.split(inner)
        replaced = False
        rendered_parts: list[str] = []
        for part in parts:
            clean = part.strip()
            if clean in citation_by_id:
                rendered_parts.append(part.replace(clean, _render(citation_by_id[clean])))
                replaced = True
            else:
                rendered_parts.append(part)
        if not replaced:
            return match.group(0)
        return "(" + "".join(rendered_parts) + ")"

    return re.sub(r"(?<!\])\(([^()\n]{1,500})\)", _replace_group, str(text or ""))


def render_citation_location(metadata: Dict[str, Any]) -> str:
    sheet_name = str(metadata.get("sheet_name") or "").strip()
    row_start = metadata.get("row_start")
    row_end = metadata.get("row_end")
    cell_range = str(metadata.get("cell_range") or "").strip()
    if sheet_name:
        parts = [sheet_name]
        if row_start is not None:
            if row_end is not None and int(row_end) != int(row_start):
                parts.append(f"rows {row_start}-{row_end}")
            else:
                parts.append(f"row {row_start}")
        if cell_range:
            parts.append(cell_range)
        return " ".join(part for part in parts if part)
    if "page" in metadata:
        return f"page {metadata.get('page')}"
    if "start_index" in metadata:
        return f"char {metadata.get('start_index')}"
    if "chunk_index" in metadata:
        return f"chunk {metadata.get('chunk_index')}"
    return ""


def build_citations(
    docs: Sequence[Document],
    *,
    max_snippet_chars: int = 320,
    url_resolver: Callable[[str], str] | None = None,
) -> List[Citation]:
    citations: List[Citation] = []
    for doc in docs:
        metadata = doc.metadata or {}
        doc_id = str(metadata.get("doc_id") or "")
        url = (
            str(metadata.get("url") or metadata.get("source_url") or metadata.get("document_url") or "").strip()
        )
        if not url and url_resolver is not None and doc_id:
            try:
                url = str(url_resolver(doc_id) or "").strip()
            except Exception:
                url = ""
        snippet = doc.page_content.strip().replace("\n", " ")
        if len(snippet) > max_snippet_chars:
            snippet = snippet[:max_snippet_chars] + "..."
        citations.append(
            Citation(
                citation_id=str(metadata.get("chunk_id") or ""),
                doc_id=doc_id,
                title=str(metadata.get("title") or ""),
                source_type=str(metadata.get("source_type") or ""),
                location=render_citation_location(metadata),
                snippet=snippet,
                collection_id=str(metadata.get("collection_id") or ""),
                url=url,
                source_path=str(metadata.get("source_path") or metadata.get("source_display_path") or ""),
            )
        )
    return citations
