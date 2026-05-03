from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
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
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    source_uri: str = ""
    source_storage_backend: str = "local"
    source_object_bucket: str = ""
    source_object_key: str = ""
    source_etag: str = ""
    source_size_bytes: int = 0
    source_content_type: str = ""
    active: bool = True
    version_ordinal: int = 1
    superseded_at: str = ""
    parser_provenance: Dict[str, Any] = field(default_factory=dict)
    extraction_status: str = "success"
    extraction_error: str = ""
    metadata_confidence: float = 0.5
    lifecycle_phase: str = ""
    doc_type: str = ""
    program_entities: List[str] = field(default_factory=list)
    signal_summary: Dict[str, Any] = field(default_factory=dict)


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
                         num_chunks, ingested_at, file_type, doc_structure_type, source_display_path, source_identity,
                         source_metadata, source_uri, source_storage_backend, source_object_bucket,
                         source_object_key, source_etag, source_size_bytes, source_content_type, active,
                         version_ordinal, superseded_at, parser_provenance, extraction_status, extraction_error,
                         metadata_confidence, lifecycle_phase, doc_type, program_entities, signal_summary)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        source_identity    = EXCLUDED.source_identity,
                        source_metadata    = EXCLUDED.source_metadata,
                        source_uri         = EXCLUDED.source_uri,
                        source_storage_backend = EXCLUDED.source_storage_backend,
                        source_object_bucket = EXCLUDED.source_object_bucket,
                        source_object_key  = EXCLUDED.source_object_key,
                        source_etag        = EXCLUDED.source_etag,
                        source_size_bytes  = EXCLUDED.source_size_bytes,
                        source_content_type = EXCLUDED.source_content_type,
                        active             = EXCLUDED.active,
                        version_ordinal    = EXCLUDED.version_ordinal,
                        superseded_at      = EXCLUDED.superseded_at,
                        parser_provenance  = EXCLUDED.parser_provenance,
                        extraction_status  = EXCLUDED.extraction_status,
                        extraction_error   = EXCLUDED.extraction_error,
                        metadata_confidence = EXCLUDED.metadata_confidence,
                        lifecycle_phase    = EXCLUDED.lifecycle_phase,
                        doc_type           = EXCLUDED.doc_type,
                        program_entities   = EXCLUDED.program_entities,
                        signal_summary     = EXCLUDED.signal_summary
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
                        psycopg2.extras.Json(dict(doc.source_metadata or {})),
                        doc.source_uri,
                        doc.source_storage_backend,
                        doc.source_object_bucket,
                        doc.source_object_key,
                        doc.source_etag,
                        doc.source_size_bytes,
                        doc.source_content_type,
                        bool(doc.active),
                        int(doc.version_ordinal or 1),
                        doc.superseded_at or None,
                        psycopg2.extras.Json(dict(doc.parser_provenance or {})),
                        doc.extraction_status or "success",
                        doc.extraction_error or "",
                        float(doc.metadata_confidence or 0.0),
                        doc.lifecycle_phase or "",
                        doc.doc_type or "",
                        psycopg2.extras.Json(list(doc.program_entities or [])),
                        psycopg2.extras.Json(dict(doc.signal_summary or {})),
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
        active_only: bool = True,
    ) -> List[DocumentRecord]:
        """Return all documents for tenant, optionally filtered by source_type ('kb' or 'upload')."""
        sql = ["SELECT * FROM documents WHERE tenant_id = %s"]
        params: List[Any] = [tenant_id]
        if source_type:
            sql.append("AND source_type = %s")
            params.append(source_type)
        if collection_id:
            sql.append("AND collection_id = %s")
            params.append(collection_id)
        if active_only:
            sql.append("AND active = TRUE")
        sql.append("ORDER BY ingested_at")
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(" ".join(sql), params)
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
        active_only: bool = True,
    ) -> List[DocumentRecord]:
        sql = ["SELECT * FROM documents WHERE tenant_id = %s"]
        params: List[Any] = [tenant_id]

        if active_only:
            sql.append("AND active = TRUE")
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

    def supersede_source_versions(
        self,
        *,
        tenant_id: str,
        collection_id: str,
        source_type: str,
        source_identity: str,
        active_doc_id: str,
        superseded_at: str = "",
    ) -> None:
        """Mark older rows for the same logical source inactive after a newer version succeeds."""
        timestamp = superseded_at or (dt.datetime.utcnow().isoformat() + "Z")
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE documents
                    SET active = FALSE,
                        superseded_at = COALESCE(superseded_at, %s::timestamptz)
                    WHERE tenant_id = %s
                      AND collection_id = %s
                      AND source_type = %s
                      AND source_identity = %s
                      AND doc_id <> %s
                      AND active = TRUE
                    """,
                    (timestamp, tenant_id, collection_id, source_type, source_identity, active_doc_id),
                )
            conn.commit()

    def list_document_versions(self, doc_id: str, tenant_id: str) -> List[DocumentRecord]:
        """Return all stored versions for the logical source that owns doc_id."""
        base = self.get_document(doc_id, tenant_id)
        if base is None:
            return []
        sql = [
            "SELECT * FROM documents WHERE tenant_id = %s AND collection_id = %s AND source_type = %s"
        ]
        params: List[Any] = [tenant_id, base.collection_id, base.source_type]
        if base.source_identity:
            sql.append("AND source_identity = %s")
            params.append(base.source_identity)
        else:
            sql.append("AND title = %s")
            params.append(base.title)
        sql.append("ORDER BY version_ordinal DESC, ingested_at DESC, doc_id DESC")
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(" ".join(sql), params)
                rows = cur.fetchall()
        return [_row_to_record(dict(r)) for r in rows]

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
                          AND active = TRUE
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
                          AND active = TRUE
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
                      AND active = TRUE
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
                      AND active = TRUE
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
                    "SELECT doc_id, title, source_type FROM documents WHERE tenant_id = %s AND active = TRUE ORDER BY title",
                    (tenant_id,),
                )
                rows = cur.fetchall()
        return [dict(r) for r in rows]


def _row_to_record(row: Dict[str, Any]) -> DocumentRecord:
    source_metadata = dict(_coerce_json_value(row.get("source_metadata"), {}) or {})
    source_path = row.get("source_path") or ""
    source_uri = row.get("source_uri") or source_metadata.get("source_uri") or ""
    if not source_uri and source_path:
        source_uri = source_path if "://" in str(source_path) else f"file://{source_path}"
    blob_ref = source_metadata.get("blob_ref") if isinstance(source_metadata.get("blob_ref"), dict) else {}
    return DocumentRecord(
        doc_id=row.get("doc_id", ""),
        tenant_id=row.get("tenant_id") or "local-dev",
        collection_id=row.get("collection_id") or "default",
        title=row.get("title", ""),
        source_type=row.get("source_type", ""),
        content_hash=row.get("content_hash", ""),
        source_path=source_path,
        num_chunks=row.get("num_chunks") or 0,
        ingested_at=str(row.get("ingested_at") or ""),
        file_type=row.get("file_type") or "",
        doc_structure_type=row.get("doc_structure_type") or "general",
        source_display_path=row.get("source_display_path") or "",
        source_identity=row.get("source_identity") or "",
        source_metadata=source_metadata,
        source_uri=str(source_uri or ""),
        source_storage_backend=str(
            row.get("source_storage_backend")
            or blob_ref.get("backend")
            or ("local" if source_path else "")
        ),
        source_object_bucket=str(row.get("source_object_bucket") or blob_ref.get("bucket") or ""),
        source_object_key=str(row.get("source_object_key") or blob_ref.get("key") or ""),
        source_etag=str(row.get("source_etag") or blob_ref.get("etag") or ""),
        source_size_bytes=int(row.get("source_size_bytes") or blob_ref.get("size") or 0),
        source_content_type=str(
            row.get("source_content_type")
            or blob_ref.get("content_type")
            or source_metadata.get("mime_type")
            or ""
        ),
        active=bool(True if row.get("active") is None else row.get("active")),
        version_ordinal=int(row.get("version_ordinal") or 1),
        superseded_at=str(row.get("superseded_at") or ""),
        parser_provenance=dict(_coerce_json_value(row.get("parser_provenance"), {}) or {}),
        extraction_status=str(row.get("extraction_status") or "success"),
        extraction_error=str(row.get("extraction_error") or ""),
        metadata_confidence=float(row.get("metadata_confidence") if row.get("metadata_confidence") is not None else 0.5),
        lifecycle_phase=str(row.get("lifecycle_phase") or ""),
        doc_type=str(row.get("doc_type") or ""),
        program_entities=[
            str(item)
            for item in list(_coerce_json_value(row.get("program_entities"), []) or [])
            if str(item)
        ],
        signal_summary=dict(_coerce_json_value(row.get("signal_summary"), {}) or {}),
    )


def _coerce_json_value(value: Any, default: Any) -> Any:
    if value in ("", None):
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return value
