from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from threading import Lock
from typing import List, Optional

import psycopg2.extras

from agentic_chatbot_next.persistence.postgres.connection import get_conn

COLLECTION_MAINTENANCE_CONFIGURED_KB_SOURCES = "configured_kb_sources"
COLLECTION_MAINTENANCE_INDEXED_DOCUMENTS = "indexed_documents"
_VALID_MAINTENANCE_POLICIES = {
    COLLECTION_MAINTENANCE_CONFIGURED_KB_SOURCES,
    COLLECTION_MAINTENANCE_INDEXED_DOCUMENTS,
}
_schema_ready = False
_schema_lock = Lock()


def normalize_collection_maintenance_policy(
    value: str,
    *,
    default: str = COLLECTION_MAINTENANCE_INDEXED_DOCUMENTS,
) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in _VALID_MAINTENANCE_POLICIES else default


def _default_maintenance_policy(collection_id: str) -> str:
    return (
        COLLECTION_MAINTENANCE_CONFIGURED_KB_SOURCES
        if str(collection_id or "").strip() == "default"
        else COLLECTION_MAINTENANCE_INDEXED_DOCUMENTS
    )


def _ensure_collection_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    ALTER TABLE collections
                    ADD COLUMN IF NOT EXISTS maintenance_policy TEXT
                    """
                )
                cur.execute(
                    """
                    UPDATE collections
                    SET maintenance_policy = %s
                    WHERE collection_id = 'default'
                      AND (maintenance_policy IS NULL OR maintenance_policy = '')
                    """,
                    (COLLECTION_MAINTENANCE_CONFIGURED_KB_SOURCES,),
                )
                cur.execute(
                    """
                    UPDATE collections AS c
                    SET maintenance_policy = CASE
                        WHEN EXISTS (
                            SELECT 1
                            FROM documents AS d
                            WHERE d.tenant_id = c.tenant_id
                              AND d.collection_id = c.collection_id
                        )
                        AND NOT EXISTS (
                            SELECT 1
                            FROM documents AS d
                            WHERE d.tenant_id = c.tenant_id
                              AND d.collection_id = c.collection_id
                              AND COALESCE(d.source_type, '') <> 'kb'
                        )
                        THEN %s
                        ELSE %s
                    END
                    WHERE maintenance_policy IS NULL OR maintenance_policy = ''
                    """,
                    (
                        COLLECTION_MAINTENANCE_CONFIGURED_KB_SOURCES,
                        COLLECTION_MAINTENANCE_INDEXED_DOCUMENTS,
                    ),
                )
                cur.execute(
                    """
                    ALTER TABLE collections
                    ALTER COLUMN maintenance_policy SET DEFAULT %s
                    """,
                    (COLLECTION_MAINTENANCE_INDEXED_DOCUMENTS,),
                )
                cur.execute(
                    """
                    UPDATE collections
                    SET maintenance_policy = %s
                    WHERE maintenance_policy IS NULL OR maintenance_policy = ''
                    """,
                    (COLLECTION_MAINTENANCE_INDEXED_DOCUMENTS,),
                )
                cur.execute(
                    """
                    ALTER TABLE collections
                    ALTER COLUMN maintenance_policy SET NOT NULL
                    """
                )
            conn.commit()
        _schema_ready = True


@dataclass
class CollectionRecord:
    tenant_id: str = "local-dev"
    collection_id: str = "default"
    maintenance_policy: str = COLLECTION_MAINTENANCE_CONFIGURED_KB_SOURCES
    created_at: str = ""
    updated_at: str = ""


class CollectionStore:
    """CRUD operations for the persistent collection catalog."""

    def upsert_collection(self, record: CollectionRecord) -> CollectionRecord:
        _ensure_collection_schema()
        created_at = record.created_at or (dt.datetime.utcnow().isoformat() + "Z")
        updated_at = record.updated_at or created_at
        maintenance_policy = normalize_collection_maintenance_policy(
            record.maintenance_policy,
            default=_default_maintenance_policy(record.collection_id),
        )
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO collections (tenant_id, collection_id, maintenance_policy, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (tenant_id, collection_id) DO UPDATE SET
                        maintenance_policy = CASE
                            WHEN NULLIF(EXCLUDED.maintenance_policy, '') IS NOT NULL
                            THEN EXCLUDED.maintenance_policy
                            ELSE collections.maintenance_policy
                        END,
                        updated_at = GREATEST(collections.updated_at, EXCLUDED.updated_at)
                    RETURNING tenant_id, collection_id, maintenance_policy, created_at, updated_at
                    """,
                    (record.tenant_id, record.collection_id, maintenance_policy, created_at, updated_at),
                )
                row = cur.fetchone()
            conn.commit()
        return _row_to_record(dict(row or {}))

    def ensure_collection(
        self,
        *,
        tenant_id: str,
        collection_id: str,
        maintenance_policy: str = "",
    ) -> CollectionRecord:
        return self.upsert_collection(
            CollectionRecord(
                tenant_id=tenant_id,
                collection_id=collection_id,
                maintenance_policy=maintenance_policy or _default_maintenance_policy(collection_id),
            )
        )

    def get_collection(self, collection_id: str, *, tenant_id: str) -> Optional[CollectionRecord]:
        _ensure_collection_schema()
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT tenant_id, collection_id, maintenance_policy, created_at, updated_at
                    FROM collections
                    WHERE tenant_id = %s AND collection_id = %s
                    """,
                    (tenant_id, collection_id),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return _row_to_record(dict(row))

    def list_collections(self, *, tenant_id: str = "local-dev") -> List[CollectionRecord]:
        _ensure_collection_schema()
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT tenant_id, collection_id, maintenance_policy, created_at, updated_at
                    FROM collections
                    WHERE tenant_id = %s
                    ORDER BY collection_id ASC
                    """,
                    (tenant_id,),
                )
                rows = cur.fetchall()
        return [_row_to_record(dict(row)) for row in rows]

    def delete_collection(self, collection_id: str, *, tenant_id: str) -> bool:
        _ensure_collection_schema()
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM collections WHERE tenant_id = %s AND collection_id = %s",
                    (tenant_id, collection_id),
                )
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted


def _row_to_record(row: dict[str, object]) -> CollectionRecord:
    return CollectionRecord(
        tenant_id=str(row.get("tenant_id") or "local-dev"),
        collection_id=str(row.get("collection_id") or "default"),
        maintenance_policy=normalize_collection_maintenance_policy(
            str(row.get("maintenance_policy") or ""),
            default=_default_maintenance_policy(str(row.get("collection_id") or "default")),
        ),
        created_at=str(row.get("created_at") or ""),
        updated_at=str(row.get("updated_at") or ""),
    )
