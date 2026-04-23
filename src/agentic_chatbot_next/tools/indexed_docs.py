from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from langchain.tools import tool

from agentic_chatbot_next.rag.adaptive import CorpusRetrievalAdapter
from agentic_chatbot_next.rag.doc_targets import (
    extract_named_document_targets,
    resolve_indexed_docs as resolve_named_indexed_docs,
)
from agentic_chatbot_next.rag.collection_selection import (
    apply_selection_to_session,
    select_collection_for_query,
    selection_answer,
)
from agentic_chatbot_next.rag.retrieval_scope import (
    resolve_collection_ids_for_source,
    resolve_search_collection_ids,
)


def _tenant_id(settings: object, session: object) -> str:
    return str(getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev")) or "local-dev")


def _serialize_chunk(doc: Any) -> Dict[str, Any]:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    return {
        "chunk_id": str(metadata.get("chunk_id") or ""),
        "doc_id": str(metadata.get("doc_id") or ""),
        "chunk_index": int(metadata.get("chunk_index") or 0),
        "chunk_type": str(metadata.get("chunk_type") or ""),
        "section_title": str(metadata.get("section_title") or ""),
        "clause_number": str(metadata.get("clause_number") or ""),
        "page_number": metadata.get("page"),
        "citation_id": str(metadata.get("chunk_id") or ""),
        "content": str(getattr(doc, "page_content", "") or ""),
    }


def _record_to_dict(record: Any) -> Dict[str, Any]:
    return {
        "doc_id": str(getattr(record, "doc_id", "") or ""),
        "title": str(getattr(record, "title", "") or ""),
        "source_type": str(getattr(record, "source_type", "") or ""),
        "source_path": str(getattr(record, "source_path", "") or ""),
        "collection_id": str(getattr(record, "collection_id", "") or ""),
        "file_type": str(getattr(record, "file_type", "") or ""),
        "doc_structure_type": str(getattr(record, "doc_structure_type", "") or ""),
        "num_chunks": int(getattr(record, "num_chunks", 0) or 0),
    }


def _record_collection_id(record: Any, *, fallback: str = "") -> str:
    return str(getattr(record, "collection_id", "") or fallback or "").strip()


_SEARCH_STOPWORDS = {
    "about",
    "across",
    "all",
    "and",
    "architecture",
    "base",
    "collection",
    "corpus",
    "deep",
    "describe",
    "detail",
    "detailed",
    "docs",
    "document",
    "documents",
    "explain",
    "find",
    "for",
    "from",
    "give",
    "identify",
    "information",
    "knowledge",
    "list",
    "major",
    "only",
    "provide",
    "repo",
    "repository",
    "return",
    "search",
    "subsystem",
    "subsystems",
    "summarize",
    "synthesize",
    "system",
    "the",
    "these",
    "thorough",
    "thoroughly",
    "through",
    "what",
    "which",
}


def _normalize_query_fragments(query: str) -> List[str]:
    clean = str(query or "").strip()
    if not clean:
        return []

    fragments: List[str] = [clean]
    seen: set[str] = {clean.casefold()}

    for target in extract_named_document_targets(clean):
        normalized = str(target or "").strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            fragments.append(normalized)

    for match in re.findall(r'"([^"]+)"|\'([^\']+)\'', clean):
        normalized = str(match[0] or match[1] or "").strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            fragments.append(normalized)

    projected_terms: List[str] = []
    for token in re.findall(r"[A-Za-z0-9_]{4,}", clean.lower()):
        if token in _SEARCH_STOPWORDS or token.isdigit():
            continue
        if token not in projected_terms:
            projected_terms.append(token)
        if len(projected_terms) >= 6:
            break
    if projected_terms:
        projected = " ".join(projected_terms)
        if projected.casefold() not in seen:
            fragments.append(projected)
    return fragments


def _metadata_match_score(query: str, record: Any) -> float:
    normalized_query = str(query or "").strip().casefold()
    if not normalized_query:
        return 0.0
    title = str(getattr(record, "title", "") or "")
    source_path = str(getattr(record, "source_path", "") or "")
    source_name = Path(source_path).name if source_path else ""
    title_lower = title.casefold()
    source_name_lower = source_name.casefold()
    source_path_lower = source_path.casefold()
    if normalized_query == title_lower or normalized_query == source_name_lower:
        return 0.96
    if normalized_query in title_lower:
        return 0.84
    if normalized_query in source_name_lower:
        return 0.8
    if normalized_query in source_path_lower:
        return 0.74
    query_terms = {
        token
        for token in re.findall(r"[A-Za-z0-9_]{3,}", normalized_query)
        if token not in _SEARCH_STOPWORDS
    }
    if not query_terms:
        return 0.0
    title_terms = set(re.findall(r"[A-Za-z0-9_]{3,}", title_lower))
    source_terms = set(re.findall(r"[A-Za-z0-9_]{3,}", source_name_lower))
    overlap = len(query_terms & (title_terms | source_terms))
    if overlap == 0:
        return 0.0
    return min(0.7, 0.18 + (overlap * 0.12))


def _search_indexed_documents(
    *,
    stores: Any,
    tenant_id: str,
    query: str,
    collection_id: str = "",
    source_type: str = "kb",
    limit: int = 20,
) -> List[Dict[str, Any]]:
    doc_store = getattr(stores, "doc_store", None)
    if doc_store is None:
        return []

    effective_limit = max(1, min(int(limit or 20), 50))
    normalized_source_type = str(source_type or "kb").strip().lower()
    fragments = _normalize_query_fragments(query)
    if not fragments:
        return []

    ranked: Dict[str, Dict[str, Any]] = {}
    records_by_doc_id: Dict[str, Any] = {}
    for fragment in fragments[:6]:
        try:
            fuzzy = doc_store.fuzzy_search_title(
                fragment,
                tenant_id,
                limit=max(effective_limit, 8),
                collection_id=collection_id,
            )
        except Exception:
            fuzzy = []
        for item in fuzzy or []:
            doc_id = str(item.get("doc_id") or "")
            if not doc_id:
                continue
            try:
                record = doc_store.get_document(doc_id, tenant_id)
            except Exception:
                record = None
            if record is None:
                continue
            if normalized_source_type and str(getattr(record, "source_type", "") or "").strip().lower() != normalized_source_type:
                continue
            existing = ranked.get(doc_id)
            candidate = {
                "doc_id": doc_id,
                "title": str(getattr(record, "title", "") or doc_id),
                "source_path": str(getattr(record, "source_path", "") or ""),
                "collection_id": _record_collection_id(record, fallback=collection_id),
                "match_reason": "fuzzy_title",
                "score": float(item.get("score") or 0.0),
            }
            records_by_doc_id[doc_id] = record
            if existing is None or float(candidate["score"]) > float(existing.get("score") or 0.0):
                ranked[doc_id] = candidate

    try:
        records = doc_store.list_documents(
            tenant_id=tenant_id,
            collection_id=collection_id,
            source_type=normalized_source_type,
        )
    except TypeError:
        records = doc_store.list_documents(
            tenant_id=tenant_id,
            collection_id=collection_id,
        )
        records = [
            record
            for record in records
            if not normalized_source_type
            or str(getattr(record, "source_type", "") or "").strip().lower() == normalized_source_type
        ]
    except Exception:
        records = []

    for record in records:
        doc_id = str(getattr(record, "doc_id", "") or "")
        if not doc_id:
            continue
        records_by_doc_id[doc_id] = record
        best_score = 0.0
        best_reason = ""
        for fragment in fragments[:6]:
            score = _metadata_match_score(fragment, record)
            if score <= best_score:
                continue
            best_score = score
            best_reason = "metadata_path" if score >= 0.74 and score < 0.84 else "metadata_title"
        if best_score <= 0.0:
            continue
        existing = ranked.get(doc_id)
        candidate = {
            "doc_id": doc_id,
            "title": str(getattr(record, "title", "") or doc_id),
            "source_path": str(getattr(record, "source_path", "") or ""),
            "collection_id": _record_collection_id(record, fallback=collection_id),
            "match_reason": best_reason or "metadata_match",
            "score": round(best_score, 4),
        }
        if existing is None or float(candidate["score"]) > float(existing.get("score") or 0.0):
            ranked[doc_id] = candidate

    ordered = sorted(
        ranked.values(),
        key=lambda item: (float(item.get("score") or 0.0), str(item.get("title") or "").lower()),
        reverse=True,
    )
    return ordered[:effective_limit]


def _matching_section_chunks(chunks: Iterable[Any], *, heading: str) -> List[Any]:
    normalized_heading = str(heading or "").strip().casefold()
    if not normalized_heading:
        return list(chunks)
    matched: List[Any] = []
    for doc in chunks:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        section_title = str(metadata.get("section_title") or "").casefold()
        content = str(getattr(doc, "page_content", "") or "")
        if normalized_heading in section_title or normalized_heading in content.casefold():
            matched.append(doc)
    return matched


def make_indexed_doc_tools(
    settings: object,
    stores: object,
    session: object,
    *,
    event_sink: object | None = None,
) -> list[Any]:
    tenant_id = _tenant_id(settings, session)
    collection_ids = resolve_search_collection_ids(settings, session)
    adapter = CorpusRetrievalAdapter(stores, settings=settings, session=session)

    @tool
    def resolve_indexed_docs(names: List[str]) -> Dict[str, Any]:
        """Resolve one or more exact indexed document names to doc_ids."""

        resolution = resolve_named_indexed_docs(
            stores,
            settings=settings,
            tenant_id=tenant_id,
            names=names,
            collection_ids=collection_ids,
        )
        return resolution.to_dict()

    @tool
    def search_indexed_docs(
        query: str,
        collection_id: str = "",
        source_type: str = "kb",
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Search indexed document titles and paths for candidate documents."""

        effective_collection_id = str(collection_id or "").strip()
        normalized_source_type = str(source_type or "kb").strip().lower()
        collection_selection: Dict[str, Any] = {}
        if not effective_collection_id:
            selection = select_collection_for_query(
                stores,
                settings,
                session,
                query,
                source_type=normalized_source_type,
                event_sink=event_sink,
            )
            collection_selection = selection.to_dict()
            if selection.resolved:
                apply_selection_to_session(session, selection)
                effective_collection_id = selection.selected_collection_id
            elif normalized_source_type == "kb":
                return {
                    "query": str(query or ""),
                    "collection_id": "",
                    "source_type": "kb",
                    "results": [],
                    **selection_answer(selection),
                }
            else:
                source_collection_ids = resolve_collection_ids_for_source(
                    settings,
                    session,
                    source_type=normalized_source_type,
                )
                effective_collection_id = str(source_collection_ids[0] if source_collection_ids else "")
        effective_source_type = "" if normalized_source_type in {"all", "*", "any"} else normalized_source_type
        results = _search_indexed_documents(
            stores=stores,
            tenant_id=tenant_id,
            query=query,
            collection_id=effective_collection_id,
            source_type=effective_source_type,
            limit=limit,
        )
        return {
            "query": str(query or ""),
            "collection_id": effective_collection_id,
            "source_type": normalized_source_type or "kb",
            "collection_selection": collection_selection,
            "results": results,
        }

    @tool
    def read_indexed_doc(
        doc_id: str,
        mode: str = "overview",
        focus: str = "",
        heading: str = "",
        cursor: int = 0,
        max_chunks: int = 6,
    ) -> Dict[str, Any]:
        """Read an indexed document by overview, section, or paginated full mode."""

        record = stores.doc_store.get_document(doc_id, tenant_id)
        if record is None:
            return {
                "error": f"Indexed document '{doc_id}' was not found for this tenant.",
                "doc_id": doc_id,
                "chunks": [],
            }

        effective_mode = str(mode or "overview").strip().lower()
        capped = max(1, min(int(max_chunks or 6), 50))
        payload: Dict[str, Any] = {
            "document": _record_to_dict(record),
            "mode": effective_mode,
            "focus": str(focus or ""),
            "heading": str(heading or ""),
            "cursor": max(0, int(cursor or 0)),
        }

        if effective_mode == "full":
            total = int(stores.chunk_store.chunk_count(doc_id=doc_id, tenant_id=tenant_id) or 0)
            start = max(0, int(cursor or 0))
            end = max(start, start + capped - 1)
            raw = stores.chunk_store.get_chunks_by_index_range(doc_id, start, end, tenant_id)
            payload.update(
                {
                    "total_chunks": total,
                    "chunks": [
                        {
                            "chunk_id": item.chunk_id,
                            "doc_id": item.doc_id,
                            "chunk_index": item.chunk_index,
                            "chunk_type": item.chunk_type,
                            "section_title": str(item.section_title or ""),
                            "clause_number": str(item.clause_number or ""),
                            "page_number": item.page_number,
                            "citation_id": item.chunk_id,
                            "content": item.content,
                        }
                        for item in raw
                    ],
                    "next_cursor": start + len(raw) if start + len(raw) < total else None,
                    "has_more": bool(start + len(raw) < total),
                }
            )
            return payload

        if effective_mode == "section":
            docs = adapter.read_document(doc_id, focus=heading or focus, max_chunks=max(capped, 8))
            docs = _matching_section_chunks(docs, heading=heading)[:capped]
            outline = stores.chunk_store.get_structure_outline(doc_id, tenant_id=tenant_id)
            payload.update(
                {
                    "outline": outline,
                    "chunks": [_serialize_chunk(doc) for doc in docs],
                }
            )
            return payload

        docs = adapter.read_document(doc_id, focus=focus, max_chunks=capped)
        outline = stores.chunk_store.get_structure_outline(doc_id, tenant_id=tenant_id)
        payload.update(
            {
                "outline": outline,
                "chunks": [_serialize_chunk(doc) for doc in docs],
            }
        )
        return payload

    @tool
    def compare_indexed_docs(
        left_doc_id: str,
        right_doc_id: str,
        focus: str = "",
    ) -> Dict[str, Any]:
        """Compare two indexed documents using deterministic overview reads from each side."""

        left_record = stores.doc_store.get_document(left_doc_id, tenant_id)
        right_record = stores.doc_store.get_document(right_doc_id, tenant_id)
        if left_record is None or right_record is None:
            missing = [
                doc_id
                for doc_id, record in ((left_doc_id, left_record), (right_doc_id, right_record))
                if record is None
            ]
            return {
                "error": "One or more indexed documents could not be found.",
                "missing_doc_ids": missing,
                "left_doc_id": left_doc_id,
                "right_doc_id": right_doc_id,
            }

        left_docs = adapter.read_document(left_doc_id, focus=focus, max_chunks=6)
        right_docs = adapter.read_document(right_doc_id, focus=focus, max_chunks=6)
        left_outline = stores.chunk_store.get_structure_outline(left_doc_id, tenant_id=tenant_id)
        right_outline = stores.chunk_store.get_structure_outline(right_doc_id, tenant_id=tenant_id)

        left_sections = {
            str(item.get("section_title") or "").strip()
            for item in left_outline
            if str(item.get("section_title") or "").strip()
        } or {
            str((doc.metadata or {}).get("section_title") or "").strip()
            for doc in left_docs
            if str((doc.metadata or {}).get("section_title") or "").strip()
        }
        right_sections = {
            str(item.get("section_title") or "").strip()
            for item in right_outline
            if str(item.get("section_title") or "").strip()
        } or {
            str((doc.metadata or {}).get("section_title") or "").strip()
            for doc in right_docs
            if str((doc.metadata or {}).get("section_title") or "").strip()
        }

        return {
            "focus": str(focus or ""),
            "left_document": _record_to_dict(left_record),
            "right_document": _record_to_dict(right_record),
            "shared_sections": sorted(item for item in (left_sections & right_sections) if item),
            "left_only_sections": sorted(item for item in (left_sections - right_sections) if item),
            "right_only_sections": sorted(item for item in (right_sections - left_sections) if item),
            "left_evidence": [_serialize_chunk(doc) for doc in left_docs],
            "right_evidence": [_serialize_chunk(doc) for doc in right_docs],
            "supporting_citation_ids": {
                "left": [str((doc.metadata or {}).get("chunk_id") or "") for doc in left_docs if str((doc.metadata or {}).get("chunk_id") or "")],
                "right": [str((doc.metadata or {}).get("chunk_id") or "") for doc in right_docs if str((doc.metadata or {}).get("chunk_id") or "")],
            },
        }

    return [
        resolve_indexed_docs,
        search_indexed_docs,
        read_indexed_doc,
        compare_indexed_docs,
    ]
