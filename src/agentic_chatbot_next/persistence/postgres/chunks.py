from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import psycopg2.extras
import psycopg2.extensions

from agentic_chatbot_next.persistence.postgres.connection import get_conn


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    chunk_index: int
    content: str
    tenant_id: str = "local-dev"
    collection_id: str = "default"
    chunk_type: str = "general"          # 'general'|'clause'|'section'|'requirement'|'header'
    page_number: Optional[int] = None
    clause_number: Optional[str] = None  # e.g. '3', '3.2', '10.1.4'
    section_title: Optional[str] = None
    sheet_name: Optional[str] = None
    row_start: Optional[int] = None
    row_end: Optional[int] = None
    cell_range: Optional[str] = None
    metadata_json: Dict[str, Any] = field(default_factory=dict)
    embedding_text: Optional[str] = field(default=None, repr=False)
    embedding: Optional[List[float]] = field(default=None, repr=False)


@dataclass
class ScoredChunk:
    """Retrieval result — mirrors the existing rag/retrieval.py dataclass."""
    doc: Any          # langchain_core.documents.Document — built lazily in to_document()
    score: float
    method: str       # 'vector' | 'keyword'

    @classmethod
    def from_row(cls, row: Dict[str, Any], score: float, method: str) -> "ScoredChunk":
        from langchain_core.documents import Document
        index_metadata = _coerce_metadata_json(row.get("metadata_json"))
        metadata = {
            "chunk_id":      row.get("chunk_id", ""),
            "doc_id":        row.get("doc_id", ""),
            "tenant_id":     row.get("tenant_id", "local-dev"),
            "collection_id": row.get("collection_id", "default"),
            "chunk_index":   row.get("chunk_index", 0),
            "chunk_type":    row.get("chunk_type", "general"),
            "page":          row.get("page_number"),
            "clause_number": row.get("clause_number"),
            "section_title": row.get("section_title"),
            "sheet_name":    row.get("sheet_name"),
            "row_start":     row.get("row_start"),
            "row_end":       row.get("row_end"),
            "cell_range":    row.get("cell_range"),
            # These are populated by DocumentStore join if needed; left empty here
            "title":         row.get("title", ""),
            "source_type":   row.get("source_type", ""),
            "source_path":   row.get("source_path", ""),
            "file_type":     row.get("file_type", ""),
            "doc_structure_type": row.get("doc_structure_type", ""),
            "index_metadata": index_metadata,
        }
        doc = Document(page_content=row.get("content", ""), metadata=metadata)
        return cls(doc=doc, score=score, method=method)


class ChunkStore:
    """All chunk operations backed by PostgreSQL + pgvector."""

    def __init__(self, embed_fn: Callable[[str], List[float]], embedding_dim: int = 768):
        self._embed = embed_fn
        self.embedding_dim = embedding_dim

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[ChunkRecord], tenant_id: str) -> None:
        """Batch-insert chunks. Embeds any chunk that has no pre-computed embedding."""
        if not chunks:
            return

        rows: List[Tuple] = []
        for ch in chunks:
            emb = ch.embedding
            if emb is None:
                emb = self._embed(ch.embedding_text or ch.content)
            rows.append((
                ch.chunk_id,
                ch.doc_id,
                tenant_id,
                ch.collection_id,
                ch.chunk_index,
                ch.page_number,
                ch.clause_number,
                ch.section_title,
                ch.sheet_name,
                ch.row_start,
                ch.row_end,
                ch.cell_range,
                ch.content,
                emb,
                ch.chunk_type,
                psycopg2.extras.Json(dict(ch.metadata_json or {})),
            ))

        with get_conn() as conn:
            # Register vector type so psycopg2 can accept Python lists as vector
            from pgvector.psycopg2 import register_vector
            register_vector(conn)

            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO chunks
                        (chunk_id, doc_id, tenant_id, collection_id, chunk_index, page_number,
                         clause_number, section_title, sheet_name, row_start, row_end, cell_range,
                         content, embedding, chunk_type, metadata_json)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        tenant_id     = EXCLUDED.tenant_id,
                        collection_id = EXCLUDED.collection_id,
                        content       = EXCLUDED.content,
                        embedding     = EXCLUDED.embedding,
                        chunk_type    = EXCLUDED.chunk_type,
                        clause_number = EXCLUDED.clause_number,
                        section_title = EXCLUDED.section_title,
                        sheet_name    = EXCLUDED.sheet_name,
                        row_start     = EXCLUDED.row_start,
                        row_end       = EXCLUDED.row_end,
                        cell_range    = EXCLUDED.cell_range,
                        metadata_json = EXCLUDED.metadata_json
                    """,
                    rows,
                    template=(
                        "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s)"
                    ),
                )
            conn.commit()

    def delete_doc_chunks(self, doc_id: str, tenant_id: str) -> None:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE doc_id = %s AND tenant_id = %s", (doc_id, tenant_id))
            conn.commit()

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def vector_search(
        self,
        query: str,
        top_k: int = 12,
        doc_id_filter: Optional[str] = None,
        collection_id_filter: Optional[str] = None,
        tenant_id: str = "local-dev",
    ) -> List[ScoredChunk]:
        """ANN vector search via HNSW cosine distance."""
        embedding = self._embed(query)

        with get_conn() as conn:
            from pgvector.psycopg2 import register_vector
            register_vector(conn)

            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if doc_id_filter:
                    cur.execute(
                        """
                        SELECT c.*, d.title, d.source_type, d.source_path, d.file_type, d.doc_structure_type,
                               1 - (c.embedding <=> %s::vector) AS score
                        FROM chunks c
                        JOIN documents d USING (doc_id)
                        WHERE c.doc_id = %s
                          AND c.tenant_id = %s
                          AND d.tenant_id = %s
                        ORDER BY c.embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (embedding, doc_id_filter, tenant_id, tenant_id, embedding, top_k),
                    )
                elif collection_id_filter:
                    cur.execute(
                        """
                        SELECT c.*, d.title, d.source_type, d.source_path, d.file_type, d.doc_structure_type,
                               1 - (c.embedding <=> %s::vector) AS score
                        FROM chunks c
                        JOIN documents d USING (doc_id)
                        WHERE c.tenant_id = %s
                          AND d.tenant_id = %s
                          AND c.collection_id = %s
                          AND d.collection_id = %s
                        ORDER BY c.embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (embedding, tenant_id, tenant_id, collection_id_filter, collection_id_filter, embedding, top_k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT c.*, d.title, d.source_type, d.source_path, d.file_type, d.doc_structure_type,
                               1 - (c.embedding <=> %s::vector) AS score
                        FROM chunks c
                        JOIN documents d USING (doc_id)
                        WHERE c.tenant_id = %s AND d.tenant_id = %s
                        ORDER BY c.embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (embedding, tenant_id, tenant_id, embedding, top_k),
                    )
                rows = cur.fetchall()

        return [ScoredChunk.from_row(dict(r), float(r["score"]), "vector") for r in rows]

    # ------------------------------------------------------------------
    # Full-text search
    # ------------------------------------------------------------------

    def keyword_search(
        self,
        query: str,
        top_k: int = 12,
        doc_id_filter: Optional[str] = None,
        collection_id_filter: Optional[str] = None,
        tenant_id: str = "local-dev",
    ) -> List[ScoredChunk]:
        """PostgreSQL full-text search via tsvector/tsquery."""
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if doc_id_filter:
                    cur.execute(
                        """
                        SELECT c.*, d.title, d.source_type, d.source_path, d.file_type, d.doc_structure_type,
                               ts_rank_cd(c.ts, plainto_tsquery('english', %s)) AS score
                        FROM chunks c
                        JOIN documents d USING (doc_id)
                        WHERE c.ts @@ plainto_tsquery('english', %s)
                          AND c.doc_id = %s
                          AND c.tenant_id = %s
                          AND d.tenant_id = %s
                        ORDER BY score DESC
                        LIMIT %s
                        """,
                        (query, query, doc_id_filter, tenant_id, tenant_id, top_k),
                    )
                elif collection_id_filter:
                    cur.execute(
                        """
                        SELECT c.*, d.title, d.source_type, d.source_path, d.file_type, d.doc_structure_type,
                               ts_rank_cd(c.ts, plainto_tsquery('english', %s)) AS score
                        FROM chunks c
                        JOIN documents d USING (doc_id)
                        WHERE c.ts @@ plainto_tsquery('english', %s)
                          AND c.tenant_id = %s
                          AND d.tenant_id = %s
                          AND c.collection_id = %s
                          AND d.collection_id = %s
                        ORDER BY score DESC
                        LIMIT %s
                        """,
                        (query, query, tenant_id, tenant_id, collection_id_filter, collection_id_filter, top_k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT c.*, d.title, d.source_type, d.source_path, d.file_type, d.doc_structure_type,
                               ts_rank_cd(c.ts, plainto_tsquery('english', %s)) AS score
                        FROM chunks c
                        JOIN documents d USING (doc_id)
                        WHERE c.ts @@ plainto_tsquery('english', %s)
                          AND c.tenant_id = %s
                          AND d.tenant_id = %s
                        ORDER BY score DESC
                        LIMIT %s
                        """,
                        (query, query, tenant_id, tenant_id, top_k),
                    )
                rows = cur.fetchall()

        return [ScoredChunk.from_row(dict(r), float(r["score"]), "keyword") for r in rows]

    # ------------------------------------------------------------------
    # Full-text search within a single document (full content)
    # ------------------------------------------------------------------

    def full_text_search_document(
        self,
        query: str,
        doc_id: str,
        top_k: int = 20,
        tenant_id: str = "local-dev",
    ) -> List[ChunkRecord]:
        """Full-text search within a single document, returning full ChunkRecords.

        Unlike keyword_search() which returns ScoredChunks with truncated content
        wrapped in LangChain Documents, this returns raw ChunkRecords with complete
        content text, page numbers, section titles, and clause numbers. Ordered by
        ts_rank_cd relevance score descending.
        """
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT *, ts_rank_cd(ts, plainto_tsquery('english', %s)) AS score
                    FROM chunks
                    WHERE ts @@ plainto_tsquery('english', %s)
                      AND doc_id = %s
                      AND tenant_id = %s
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (query, query, doc_id, tenant_id, top_k),
                )
                rows = cur.fetchall()
        return [_row_to_chunk(dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # Structured retrieval (clause/requirement aware)
    # ------------------------------------------------------------------

    def get_chunks_by_clause(
        self,
        doc_id: str,
        clause_numbers: List[str],
        tenant_id: str,
    ) -> List[ChunkRecord]:
        """Retrieve chunks by exact clause_number match."""
        if not clause_numbers:
            return []
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM chunks
                    WHERE doc_id = %s AND tenant_id = %s AND clause_number = ANY(%s)
                    ORDER BY chunk_index
                    """,
                    (doc_id, tenant_id, clause_numbers),
                )
                rows = cur.fetchall()
        return [_row_to_chunk(dict(r)) for r in rows]

    def get_structure_outline(self, doc_id: str, tenant_id: str) -> List[Dict[str, Any]]:
        """Return distinct clause/section structure of a document, ordered by position."""
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT DISTINCT ON (clause_number)
                           clause_number, section_title, chunk_type, chunk_index
                    FROM chunks
                    WHERE doc_id = %s AND tenant_id = %s AND clause_number IS NOT NULL
                    ORDER BY clause_number, chunk_index
                    """,
                    (doc_id, tenant_id),
                )
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_requirement_chunks(
        self,
        doc_id: str,
        semantic_query: Optional[str] = None,
        top_k: int = 30,
        tenant_id: str = "local-dev",
    ) -> List[ChunkRecord]:
        """Return all chunks tagged chunk_type='requirement' for a document.

        If semantic_query is provided, re-rank by cosine similarity and return top_k.
        """
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if semantic_query:
                    from pgvector.psycopg2 import register_vector
                    register_vector(conn)
                    emb = self._embed(semantic_query)
                    cur.execute(
                        """
                        SELECT *, 1 - (embedding <=> %s::vector) AS _vscore
                        FROM chunks
                        WHERE doc_id = %s AND tenant_id = %s AND chunk_type = 'requirement'
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (emb, doc_id, tenant_id, emb, top_k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT * FROM chunks
                        WHERE doc_id = %s AND tenant_id = %s AND chunk_type = 'requirement'
                        ORDER BY chunk_index
                        LIMIT %s
                        """,
                        (doc_id, tenant_id, top_k),
                    )
                rows = cur.fetchall()
        return [_row_to_chunk(dict(r)) for r in rows]

    def search_sections(
        self,
        doc_id: str,
        *,
        tenant_id: str = "local-dev",
        section_query: str = "",
        clause_numbers: List[str] | None = None,
        sheet_names: List[str] | None = None,
        limit: int = 12,
    ) -> List[ChunkRecord]:
        sql = ["SELECT * FROM chunks WHERE doc_id = %s AND tenant_id = %s"]
        params: List[Any] = [doc_id, tenant_id]
        clause_numbers = [str(item) for item in (clause_numbers or []) if str(item)]
        sheet_names = [str(item) for item in (sheet_names or []) if str(item)]
        if clause_numbers:
            sql.append("AND clause_number = ANY(%s)")
            params.append(clause_numbers)
        if sheet_names:
            sql.append("AND sheet_name = ANY(%s)")
            params.append(sheet_names)
        if section_query:
            sql.append(
                "AND (lower(COALESCE(section_title, '')) LIKE lower(%s) OR lower(COALESCE(content, '')) LIKE lower(%s))"
            )
            params.extend([f"%{section_query}%", f"%{section_query}%"])
        sql.append("ORDER BY chunk_index ASC LIMIT %s")
        params.append(max(1, int(limit)))
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(" ".join(sql), params)
                rows = cur.fetchall()
        return [_row_to_chunk(dict(r)) for r in rows]

    def match_document_outline(
        self,
        doc_id: str,
        *,
        query: str,
        tenant_id: str = "local-dev",
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        lowered = str(query or "").lower()
        matches = []
        for item in self.get_structure_outline(doc_id, tenant_id):
            haystack = " ".join(
                [
                    str(item.get("clause_number") or ""),
                    str(item.get("section_title") or ""),
                    str(item.get("chunk_type") or ""),
                ]
            ).lower()
            if lowered and lowered not in haystack and not any(token in haystack for token in lowered.split()):
                continue
            matches.append(item)
            if len(matches) >= max(1, int(limit)):
                break
        return matches

    # ------------------------------------------------------------------
    # Point-lookup helpers (used by extended RAG tools)
    # ------------------------------------------------------------------

    def get_chunk_by_id(self, chunk_id: str, tenant_id: str) -> Optional[ChunkRecord]:
        """Fetch a single chunk by its exact chunk_id."""
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM chunks WHERE chunk_id = %s AND tenant_id = %s LIMIT 1",
                    (chunk_id, tenant_id),
                )
                row = cur.fetchone()
        return _row_to_chunk(dict(row)) if row else None

    def get_chunks_by_index_range(
        self,
        doc_id: str,
        min_idx: int,
        max_idx: int,
        tenant_id: str,
    ) -> List[ChunkRecord]:
        """Return chunks with chunk_index BETWEEN min_idx AND max_idx, ordered by index.

        Used by the ``chunk_expander`` tool to fetch neighbouring chunks for
        context window expansion around a retrieved snippet.
        """
        clamped_min = max(0, min_idx)
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM chunks
                    WHERE doc_id = %s
                      AND tenant_id = %s
                      AND chunk_index BETWEEN %s AND %s
                    ORDER BY chunk_index
                    """,
                    (doc_id, tenant_id, clamped_min, max_idx),
                )
                rows = cur.fetchall()
        return [_row_to_chunk(dict(r)) for r in rows]

    def list_document_chunks(self, doc_id: str, tenant_id: str) -> List[ChunkRecord]:
        """Return all chunks for a document ordered by chunk index."""
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM chunks
                    WHERE doc_id = %s AND tenant_id = %s
                    ORDER BY chunk_index
                    """,
                    (doc_id, tenant_id),
                )
                rows = cur.fetchall()
        return [_row_to_chunk(dict(r)) for r in rows]

    def chunk_count(self, doc_id: Optional[str] = None, tenant_id: str = "local-dev") -> int:
        """Return total number of chunks (optionally filtered by doc_id)."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                if doc_id:
                    cur.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = %s AND tenant_id = %s", (doc_id, tenant_id))
                else:
                    cur.execute("SELECT COUNT(*) FROM chunks WHERE tenant_id = %s", (tenant_id,))
                row = cur.fetchone()
        return int(row[0]) if row else 0


def _row_to_chunk(row: Dict[str, Any]) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=row.get("chunk_id", ""),
        doc_id=row.get("doc_id", ""),
        tenant_id=row.get("tenant_id") or "local-dev",
        collection_id=row.get("collection_id") or "default",
        chunk_index=row.get("chunk_index", 0),
        content=row.get("content", ""),
        chunk_type=row.get("chunk_type", "general"),
        page_number=row.get("page_number"),
        clause_number=row.get("clause_number"),
        section_title=row.get("section_title"),
        sheet_name=row.get("sheet_name"),
        row_start=row.get("row_start"),
        row_end=row.get("row_end"),
        cell_range=row.get("cell_range"),
        metadata_json=_coerce_metadata_json(row.get("metadata_json")),
        embedding=None,  # don't load embeddings back into Python by default
    )


def _coerce_metadata_json(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}
