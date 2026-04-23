from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import psycopg2.extras

from agentic_chatbot_next.persistence.postgres.connection import get_conn


@dataclass
class DocumentRecord:
    doc_id: str
    title: str
    source_type: str                    # 'kb' | 'upload'
    content_hash: str
    tenant_id: str = "local-dev"
    collection_id: str = "default"
    source_path: str = ""
    num_chunks: int = 0
    ingested_at: str = ""
    file_type: str = ""
    doc_structure_type: str = "general"
    source_display_path: str = ""
    source_identity: str = ""


class DocumentStore:
    """CRUD operations against the `documents` table."""

    def upsert_document(self, doc: DocumentRecord) -> None:
        """Insert or update a document record (keyed on doc_id)."""
        ingested_at = doc.ingested_at or (dt.datetime.utcnow().isoformat() + "Z")
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents
                        (doc_id, tenant_id, collection_id, title, source_type, source_path, content_hash,
                         num_chunks, ingested_at, file_type, doc_structure_type, source_display_path, source_identity)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        tenant_id          = EXCLUDED.tenant_id,
                        collection_id      = EXCLUDED.collection_id,
                        title              = EXCLUDED.title,
                        source_type        = EXCLUDED.source_type,
                        source_path        = EXCLUDED.source_path,
                        content_hash       = EXCLUDED.content_hash,
                        num_chunks         = EXCLUDED.num_chunks,
                        ingested_at        = EXCLUDED.ingested_at,
                        file_type          = EXCLUDED.file_type,
                        doc_structure_type = EXCLUDED.doc_structure_type,
                        source_display_path = EXCLUDED.source_display_path,
                        source_identity    = EXCLUDED.source_identity
                    """,
                    (
                        doc.doc_id,
                        doc.tenant_id,
                        doc.collection_id,
                        doc.title,
                        doc.source_type,
                        doc.source_path,
                        doc.content_hash,
                        doc.num_chunks,
                        ingested_at,
                        doc.file_type,
                        doc.doc_structure_type,
                        doc.source_display_path,
                        doc.source_identity,
                    ),
                )
            conn.commit()

    def get_document(self, doc_id: str, tenant_id: str) -> Optional[DocumentRecord]:
        """Return a DocumentRecord by doc_id, or None if not found."""
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM documents WHERE doc_id = %s AND tenant_id = %s",
                    (doc_id, tenant_id),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return _row_to_record(dict(row))

    def document_exists(
        self,
        doc_id: str,
        content_hash: str,
        tenant_id: str,
        *,
        collection_id: str = "",
        source_type: str = "",
        title: str = "",
    ) -> bool:
        """Return True if a matching document already exists for this collection."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                if collection_id and source_type and title:
                    cur.execute(
                        """
                        SELECT 1
                        FROM documents
                        WHERE tenant_id = %s
                          AND content_hash = %s
                          AND (
                                doc_id = %s
                                OR (
                                    collection_id = %s
                                    AND source_type = %s
                                    AND title = %s
                                )
                          )
                        """,
                        (tenant_id, content_hash, doc_id, collection_id, source_type, title),
                    )
                else:
                    cur.execute(
                        "SELECT 1 FROM documents WHERE doc_id = %s AND content_hash = %s AND tenant_id = %s",
                        (doc_id, content_hash, tenant_id),
                    )
                return cur.fetchone() is not None

    def list_documents(
        self,
        source_type: str = "",
        tenant_id: str = "local-dev",
        collection_id: str = "",
    ) -> List[DocumentRecord]:
        """Return all documents for tenant, optionally filtered by source_type ('kb' or 'upload')."""
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if source_type and collection_id:
                    cur.execute(
                        """
                        SELECT * FROM documents
                        WHERE tenant_id = %s AND source_type = %s AND collection_id = %s
                        ORDER BY ingested_at
                        """,
                        (tenant_id, source_type, collection_id),
                    )
                elif source_type:
                    cur.execute(
                        "SELECT * FROM documents WHERE tenant_id = %s AND source_type = %s ORDER BY ingested_at",
                        (tenant_id, source_type),
                    )
                elif collection_id:
                    cur.execute(
                        "SELECT * FROM documents WHERE tenant_id = %s AND collection_id = %s ORDER BY ingested_at",
                        (tenant_id, collection_id),
                    )
                else:
                    cur.execute(
                        "SELECT * FROM documents WHERE tenant_id = %s ORDER BY ingested_at",
                        (tenant_id,),
                    )
                rows = cur.fetchall()
        return [_row_to_record(dict(r)) for r in rows]

    def search_by_metadata(
        self,
        *,
        tenant_id: str = "local-dev",
        collection_id: str = "",
        source_type: str = "",
        file_type: str = "",
        doc_structure_type: str = "",
        title_contains: str = "",
        limit: int = 100,
    ) -> List[DocumentRecord]:
        sql = ["SELECT * FROM documents WHERE tenant_id = %s"]
        params: List[Any] = [tenant_id]

        if collection_id:
            sql.append("AND collection_id = %s")
            params.append(collection_id)
        if source_type:
            sql.append("AND source_type = %s")
            params.append(source_type)
        if file_type:
            sql.append("AND file_type = %s")
            params.append(file_type)
        if doc_structure_type:
            sql.append("AND doc_structure_type = %s")
            params.append(doc_structure_type)
        if title_contains:
            sql.append("AND lower(title) LIKE lower(%s)")
            params.append(f"%{title_contains}%")

        sql.append("ORDER BY ingested_at DESC, title ASC LIMIT %s")
        params.append(max(1, int(limit)))

        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(" ".join(sql), params)
                rows = cur.fetchall()
        return [_row_to_record(dict(r)) for r in rows]

    def delete_document(self, doc_id: str, tenant_id: str) -> None:
        """Delete a document record (chunks cascade via FK)."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE doc_id = %s AND tenant_id = %s", (doc_id, tenant_id))
            conn.commit()

    def fuzzy_search_title(
        self,
        hint: str,
        tenant_id: str,
        limit: int = 5,
        collection_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Return documents whose title fuzzy-matches hint, ranked by similarity.

        Uses PostgreSQL pg_trgm similarity(). Requires the pg_trgm extension.
        Returns list of dicts: {doc_id, title, source_type, doc_structure_type, score}.
        """
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if collection_id:
                    cur.execute(
                        """
                        SELECT doc_id, title, source_type, doc_structure_type,
                               similarity(lower(title), lower(%s)) AS score
                        FROM documents
                        WHERE tenant_id = %s
                          AND collection_id = %s
                          AND similarity(lower(title), lower(%s)) > 0.1
                        ORDER BY score DESC
                        LIMIT %s
                        """,
                        (hint, tenant_id, collection_id, hint, limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT doc_id, title, source_type, doc_structure_type,
                               similarity(lower(title), lower(%s)) AS score
                        FROM documents
                        WHERE tenant_id = %s
                          AND similarity(lower(title), lower(%s)) > 0.1
                        ORDER BY score DESC
                        LIMIT %s
                        """,
                        (hint, tenant_id, hint, limit),
                    )
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def list_collections(self, tenant_id: str = "local-dev") -> List[Dict[str, Any]]:
        """Return collection IDs and document counts for the tenant."""
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT collection_id,
                           COUNT(*) AS document_count,
                           MAX(ingested_at) AS latest_ingested_at
                    FROM documents
                    WHERE tenant_id = %s
                    GROUP BY collection_id
                    ORDER BY collection_id
                    """,
                    (tenant_id,),
                )
                rows = cur.fetchall()
                cur.execute(
                    """
                    SELECT collection_id, source_type, COUNT(*) AS source_count
                    FROM documents
                    WHERE tenant_id = %s
                    GROUP BY collection_id, source_type
                    """,
                    (tenant_id,),
                )
                source_rows = cur.fetchall()
        source_counts: Dict[str, Dict[str, int]] = {}
        for row in source_rows:
            collection_id = str(row.get("collection_id") or "")
            source_type = str(row.get("source_type") or "unknown")
            source_counts.setdefault(collection_id, {})[source_type] = int(row.get("source_count") or 0)
        payload: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            collection_id = str(item.get("collection_id") or "")
            item["source_type_counts"] = source_counts.get(collection_id, {})
            payload.append(item)
        return payload

    def get_collection_summary(self, collection_id: str, tenant_id: str) -> Dict[str, Any] | None:
        summary = next(
            (
                item
                for item in self.list_collections(tenant_id=tenant_id)
                if str(item.get("collection_id") or "") == collection_id
            ),
            None,
        )
        return dict(summary) if summary is not None else None

    def get_all_titles(self, tenant_id: str) -> List[Dict[str, str]]:
        """Return [{doc_id, title}] for all tenant documents — used by resolve_document tool."""
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT doc_id, title, source_type FROM documents WHERE tenant_id = %s ORDER BY title",
                    (tenant_id,),
                )
                rows = cur.fetchall()
        return [dict(r) for r in rows]


def _row_to_record(row: Dict[str, Any]) -> DocumentRecord:
    return DocumentRecord(
        doc_id=row.get("doc_id", ""),
        tenant_id=row.get("tenant_id") or "local-dev",
        collection_id=row.get("collection_id") or "default",
        title=row.get("title", ""),
        source_type=row.get("source_type", ""),
        content_hash=row.get("content_hash", ""),
        source_path=row.get("source_path") or "",
        num_chunks=row.get("num_chunks") or 0,
        ingested_at=str(row.get("ingested_at") or ""),
        file_type=row.get("file_type") or "",
        doc_structure_type=row.get("doc_structure_type") or "general",
        source_display_path=row.get("source_display_path") or "",
        source_identity=row.get("source_identity") or "",
    )
