"""Next-owned RAG specialist tools."""
from __future__ import annotations

import json
import logging
from typing import Any, List

from langchain_core.tools import tool

from agentic_chatbot_next.config import Settings
from agentic_chatbot_next.graph.service import GraphService
from agentic_chatbot_next.persistence.postgres.chunks import ChunkRecord, ScoredChunk
from agentic_chatbot_next.rag.stores import KnowledgeStores

logger = logging.getLogger(__name__)


def _chunk_to_dict(chunk: Any) -> dict:
    if isinstance(chunk, ScoredChunk):
        metadata = chunk.doc.metadata or {}
        return {
            "chunk_id": metadata.get("chunk_id", ""),
            "doc_id": metadata.get("doc_id", ""),
            "chunk_type": metadata.get("chunk_type", "general"),
            "clause_number": metadata.get("clause_number"),
            "section_title": metadata.get("section_title"),
            "page_number": metadata.get("page"),
            "score": round(chunk.score, 4),
            "method": chunk.method,
            "snippet": chunk.doc.page_content[:500],
        }
    if isinstance(chunk, ChunkRecord):
        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "chunk_type": chunk.chunk_type,
            "clause_number": chunk.clause_number,
            "section_title": chunk.section_title,
            "page_number": chunk.page_number,
            "snippet": chunk.content[:500],
        }
    return {}


def make_all_rag_tools(
    stores: KnowledgeStores,
    session: Any,
    *,
    settings: Settings | None = None,
) -> List[Any]:
    top_k_vector = max(1, int(getattr(settings, "rag_top_k_vector", 8)))
    top_k_keyword = max(1, int(getattr(settings, "rag_top_k_keyword", 8)))
    tenant_id = getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev"))

    @tool
    def resolve_document(name_or_hint: str) -> str:
        """Fuzzy-match a document name or description to indexed doc_ids."""
        try:
            from rapidfuzz import fuzz
            from rapidfuzz import process as fuzz_proc
        except ImportError:
            results = stores.doc_store.fuzzy_search_title(name_or_hint, tenant_id=tenant_id, limit=5)
            return json.dumps({"candidates": results})

        all_docs = stores.doc_store.get_all_titles(tenant_id=tenant_id)
        if not all_docs:
            return json.dumps({"candidates": []})

        titles = [doc["title"] for doc in all_docs]
        doc_map = {doc["title"]: doc for doc in all_docs}
        fuzzy_hits = fuzz_proc.extract(name_or_hint, titles, scorer=fuzz.WRatio, limit=5)

        candidates = []
        seen = set()
        for title, score, _ in fuzzy_hits:
            if title in seen:
                continue
            seen.add(title)
            doc = doc_map[title]
            candidates.append(
                {
                    "doc_id": doc["doc_id"],
                    "title": title,
                    "source_type": doc.get("source_type", ""),
                    "score": round(score / 100.0, 3),
                }
            )
        trigram_hits = stores.doc_store.fuzzy_search_title(name_or_hint, tenant_id=tenant_id, limit=5)
        for hit in trigram_hits:
            if hit["doc_id"] not in {candidate["doc_id"] for candidate in candidates}:
                candidates.append(
                    {
                        "doc_id": hit["doc_id"],
                        "title": hit["title"],
                        "source_type": hit.get("source_type", ""),
                        "score": round(float(hit.get("score", 0)), 3),
                    }
                )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return json.dumps({"candidates": candidates[:5]})

    @tool
    def search_document(doc_id: str, query: str, strategy: str = "hybrid") -> str:
        """Search within a single document."""
        results = []
        if strategy in {"vector", "hybrid"}:
            results.extend(
                stores.chunk_store.vector_search(
                    query,
                    top_k=top_k_vector,
                    doc_id_filter=doc_id,
                    tenant_id=tenant_id,
                )
            )
        if strategy in {"keyword", "hybrid"}:
            results.extend(
                stores.chunk_store.keyword_search(
                    query,
                    top_k=top_k_keyword,
                    doc_id_filter=doc_id,
                    tenant_id=tenant_id,
                )
            )
        seen = set()
        unique = []
        for result in results:
            chunk_id = (result.doc.metadata or {}).get("chunk_id", "")
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique.append(result)
        unique.sort(key=lambda item: item.score, reverse=True)
        max_out = min(50, top_k_vector + top_k_keyword)
        return json.dumps([_chunk_to_dict(item) for item in unique[:max_out]])

    @tool
    def search_all_documents(query: str, strategy: str = "hybrid") -> str:
        """Search across all indexed documents."""
        results = []
        if strategy in {"vector", "hybrid"}:
            results.extend(stores.chunk_store.vector_search(query, top_k=top_k_vector, tenant_id=tenant_id))
        if strategy in {"keyword", "hybrid"}:
            results.extend(stores.chunk_store.keyword_search(query, top_k=top_k_keyword, tenant_id=tenant_id))
        seen = set()
        unique = []
        for result in results:
            chunk_id = (result.doc.metadata or {}).get("chunk_id", "")
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique.append(result)
        unique.sort(key=lambda item: item.score, reverse=True)
        max_out = min(80, top_k_vector + top_k_keyword)
        return json.dumps([_chunk_to_dict(item) for item in unique[:max_out]])

    @tool
    def search_collection(collection_id: str, query: str, strategy: str = "hybrid") -> str:
        """Search across one collection."""
        results = []
        if strategy in {"vector", "hybrid"}:
            results.extend(
                stores.chunk_store.vector_search(
                    query,
                    top_k=top_k_vector,
                    collection_id_filter=collection_id,
                    tenant_id=tenant_id,
                )
            )
        if strategy in {"keyword", "hybrid"}:
            results.extend(
                stores.chunk_store.keyword_search(
                    query,
                    top_k=top_k_keyword,
                    collection_id_filter=collection_id,
                    tenant_id=tenant_id,
                )
            )
        seen = set()
        unique = []
        for result in results:
            chunk_id = (result.doc.metadata or {}).get("chunk_id", "")
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique.append(result)
        unique.sort(key=lambda item: item.score, reverse=True)
        max_out = min(80, top_k_vector + top_k_keyword)
        return json.dumps([_chunk_to_dict(item) for item in unique[:max_out]])

    @tool
    def full_text_search_document(doc_id: str, query: str, max_results: int = 20) -> str:
        """Run a deeper keyword search inside a single document and return full chunk content."""
        chunks = stores.chunk_store.full_text_search_document(
            query,
            doc_id,
            top_k=max(1, min(int(max_results), 50)),
            tenant_id=tenant_id,
        )
        return json.dumps(
            {
                "doc_id": doc_id,
                "results": [
                    {
                        **_chunk_to_dict(chunk),
                        "content": chunk.content,
                    }
                    for chunk in chunks
                ],
            }
        )

    @tool
    def search_by_metadata(
        source_type: str = "",
        file_type: str = "",
        doc_structure_type: str = "",
        title_contains: str = "",
        collection_id: str = "",
        limit: int = 50,
    ) -> str:
        """Filter indexed documents by metadata."""
        records = stores.doc_store.search_by_metadata(
            tenant_id=tenant_id,
            collection_id=collection_id,
            source_type=source_type,
            file_type=file_type,
            doc_structure_type=doc_structure_type,
            title_contains=title_contains,
            limit=max(1, min(int(limit), 200)),
        )
        return json.dumps(
            {
                "documents": [
                    {
                        "doc_id": record.doc_id,
                        "title": record.title,
                        "source_type": record.source_type,
                        "file_type": record.file_type,
                        "doc_structure_type": record.doc_structure_type,
                        "num_chunks": record.num_chunks,
                    }
                    for record in records
                ],
                "count": len(records),
            }
        )

    @tool
    def extract_clauses(doc_id: str, clause_numbers: str) -> str:
        """Extract numbered clauses from a document."""
        numbers = [item.strip() for item in clause_numbers.split(",") if item.strip()]
        if not numbers:
            return json.dumps({"error": "No clause numbers provided."})
        chunks = stores.chunk_store.get_chunks_by_clause(doc_id, numbers, tenant_id=tenant_id)
        if not chunks:
            return json.dumps({"message": f"No clauses found for numbers {numbers} in doc {doc_id}.", "chunks": []})
        return json.dumps([_chunk_to_dict(chunk) for chunk in chunks])

    @tool
    def list_document_structure(doc_id: str) -> str:
        """Return the clause or section outline of a document."""
        outline = stores.chunk_store.get_structure_outline(doc_id, tenant_id=tenant_id)
        if not outline:
            return json.dumps({"message": f"No structured outline found for doc {doc_id}.", "outline": []})
        return json.dumps({"doc_id": doc_id, "outline": outline})

    @tool
    def fetch_document_outline(doc_id: str) -> str:
        """Alias for list_document_structure."""
        return list_document_structure.invoke({"doc_id": doc_id})

    @tool
    def extract_requirements(doc_id: str, requirement_filter: str = "") -> str:
        """Find requirement-like chunks in a document."""
        chunks = stores.chunk_store.get_requirement_chunks(
            doc_id,
            semantic_query=requirement_filter.strip() or None,
            top_k=50,
            tenant_id=tenant_id,
        )
        if not chunks:
            return json.dumps({"message": f"No requirement chunks found in doc {doc_id}.", "requirements": []})
        return json.dumps({"requirements": [_chunk_to_dict(chunk) for chunk in chunks]})

    @tool
    def compare_clauses(doc_id_1: str, doc_id_2: str, clause_numbers: str) -> str:
        """Compare matching numbered clauses between two documents."""
        left = json.loads(extract_clauses.invoke({"doc_id": doc_id_1, "clause_numbers": clause_numbers}))
        right = json.loads(extract_clauses.invoke({"doc_id": doc_id_2, "clause_numbers": clause_numbers}))
        return json.dumps({"doc_id_1": doc_id_1, "doc_id_2": doc_id_2, "left": left, "right": right})

    @tool
    def diff_documents(doc_id_1: str, doc_id_2: str) -> str:
        """Provide a coarse structural diff between two documents."""
        left_outline = stores.chunk_store.get_structure_outline(doc_id_1, tenant_id=tenant_id)
        right_outline = stores.chunk_store.get_structure_outline(doc_id_2, tenant_id=tenant_id)
        left_clauses = {str(item.get("clause_number") or "") for item in left_outline}
        right_clauses = {str(item.get("clause_number") or "") for item in right_outline}
        return json.dumps(
            {
                "doc_id_1": doc_id_1,
                "doc_id_2": doc_id_2,
                "shared_clauses": sorted(item for item in (left_clauses & right_clauses) if item),
                "only_in_doc_1": sorted(item for item in (left_clauses - right_clauses) if item),
                "only_in_doc_2": sorted(item for item in (right_clauses - left_clauses) if item),
            }
        )

    @tool
    def fetch_chunk_window(chunk_id: str, before: int = 1, after: int = 1) -> str:
        """Fetch a chunk and neighboring chunks for context."""
        chunk = stores.chunk_store.get_chunk_by_id(chunk_id, tenant_id)
        if chunk is None:
            return json.dumps({"error": f"Chunk {chunk_id!r} not found.", "chunks": []})
        neighbours = stores.chunk_store.get_chunks_by_index_range(
            chunk.doc_id,
            max(0, int(chunk.chunk_index) - int(before)),
            int(chunk.chunk_index) + int(after),
            tenant_id,
        )
        return json.dumps({"chunks": [_chunk_to_dict(item) for item in neighbours], "count": len(neighbours)})

    @tool
    def list_collections() -> str:
        """List known collections."""
        return json.dumps({"collections": stores.doc_store.list_collections(tenant_id=tenant_id)})

    @tool
    def graph_search_local(query: str, limit: int = 8) -> str:
        """Search managed graphs for local entity and relationship matches."""
        graph_service = GraphService(settings, stores, session=session) if settings is not None else None
        collection_id = str(
            getattr(session, "metadata", {}).get("collection_id")
            or getattr(session, "metadata", {}).get("kb_collection_id")
            or ""
        )
        active_graph_ids = [
            str(item)
            for item in (getattr(session, "metadata", {}) or {}).get("active_graph_ids", [])
            if str(item).strip()
        ]
        if graph_service is not None and getattr(stores, "graph_index_store", None) is not None:
            payload = graph_service.query_across_graphs(
                query,
                collection_id=collection_id,
                graph_ids=active_graph_ids or None,
                methods=["local"],
                limit=max(1, min(int(limit), 20)),
                top_k_graphs=max(1, len(active_graph_ids) or 3),
            )
            if payload.get("results"):
                return json.dumps(payload)
        graph_store = getattr(stores, "graph_store", None)
        if graph_store is None or not bool(getattr(graph_store, "available", False)):
            return json.dumps({"message": "Managed graph search is not configured.", "results": []})
        hits = graph_store.local_search(query, tenant_id=tenant_id, limit=max(1, min(int(limit), 20)))
        return json.dumps(
            {
                "results": [
                    {
                        "doc_id": hit.doc_id,
                        "chunk_ids": list(hit.chunk_ids),
                        "title": hit.title,
                        "source_path": hit.source_path,
                        "source_type": hit.source_type,
                        "score": hit.score,
                        "relationship_path": list(hit.relationship_path),
                        "summary": hit.summary,
                    }
                    for hit in hits
                ]
            }
        )

    @tool
    def graph_search_global(query: str, limit: int = 8) -> str:
        """Search managed graphs for cross-document and thematic matches."""
        graph_service = GraphService(settings, stores, session=session) if settings is not None else None
        collection_id = str(
            getattr(session, "metadata", {}).get("collection_id")
            or getattr(session, "metadata", {}).get("kb_collection_id")
            or ""
        )
        active_graph_ids = [
            str(item)
            for item in (getattr(session, "metadata", {}) or {}).get("active_graph_ids", [])
            if str(item).strip()
        ]
        if graph_service is not None and getattr(stores, "graph_index_store", None) is not None:
            payload = graph_service.query_across_graphs(
                query,
                collection_id=collection_id,
                graph_ids=active_graph_ids or None,
                methods=["global"],
                limit=max(1, min(int(limit), 20)),
                top_k_graphs=max(1, len(active_graph_ids) or 3),
            )
            if payload.get("results"):
                return json.dumps(payload)
        graph_store = getattr(stores, "graph_store", None)
        if graph_store is None or not bool(getattr(graph_store, "available", False)):
            return json.dumps({"message": "Managed graph search is not configured.", "results": []})
        hits = graph_store.global_search(query, tenant_id=tenant_id, limit=max(1, min(int(limit), 20)))
        return json.dumps(
            {
                "results": [
                    {
                        "doc_id": hit.doc_id,
                        "chunk_ids": list(hit.chunk_ids),
                        "title": hit.title,
                        "source_path": hit.source_path,
                        "source_type": hit.source_type,
                        "score": hit.score,
                        "relationship_path": list(hit.relationship_path),
                        "summary": hit.summary,
                    }
                    for hit in hits
                ]
            }
        )

    @tool
    def scratchpad_write(key: str, value: str) -> str:
        """Write a scratchpad value for the current session."""
        session.scratchpad[key] = value
        return json.dumps({"saved": key, "length": len(value)})

    @tool
    def scratchpad_read(key: str) -> str:
        """Read a scratchpad value."""
        if key not in session.scratchpad:
            return json.dumps({"error": f"Key {key!r} not found.", "available_keys": sorted(session.scratchpad.keys())})
        return json.dumps({"key": key, "value": session.scratchpad[key]})

    @tool
    def scratchpad_list() -> str:
        """List scratchpad keys."""
        return json.dumps({"keys": sorted(session.scratchpad.keys()), "count": len(session.scratchpad)})

    return [
        resolve_document,
        search_document,
        search_all_documents,
        full_text_search_document,
        search_by_metadata,
        search_collection,
        extract_clauses,
        list_document_structure,
        fetch_document_outline,
        extract_requirements,
        compare_clauses,
        diff_documents,
        fetch_chunk_window,
        list_collections,
        graph_search_local,
        graph_search_global,
        scratchpad_write,
        scratchpad_read,
        scratchpad_list,
    ]
