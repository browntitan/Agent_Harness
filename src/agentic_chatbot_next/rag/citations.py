from __future__ import annotations

from typing import Any, Dict, List, Sequence

from langchain_core.documents import Document

from agentic_chatbot_next.contracts.rag import Citation


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


def build_citations(docs: Sequence[Document], *, max_snippet_chars: int = 320) -> List[Citation]:
    citations: List[Citation] = []
    for doc in docs:
        metadata = doc.metadata or {}
        snippet = doc.page_content.strip().replace("\n", " ")
        if len(snippet) > max_snippet_chars:
            snippet = snippet[:max_snippet_chars] + "..."
        citations.append(
            Citation(
                citation_id=str(metadata.get("chunk_id") or ""),
                doc_id=str(metadata.get("doc_id") or ""),
                title=str(metadata.get("title") or ""),
                source_type=str(metadata.get("source_type") or ""),
                location=render_citation_location(metadata),
                snippet=snippet,
                collection_id=str(metadata.get("collection_id") or ""),
            )
        )
    return citations
