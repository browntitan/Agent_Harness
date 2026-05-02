from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import psycopg2.extras

from agentic_chatbot_next.persistence.postgres.connection import get_conn


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


@dataclass
class GraphIndexRecord:
    graph_id: str
    tenant_id: str = "local-dev"
    collection_id: str = "default"
    display_name: str = ""
    owner_admin_user_id: str = ""
    visibility: str = "tenant"
    backend: str = "microsoft_graphrag"
    status: str = "draft"
    root_path: str = ""
    artifact_path: str = ""
    domain_summary: str = ""
    entity_samples: List[str] = field(default_factory=list)
    relationship_samples: List[str] = field(default_factory=list)
    source_doc_ids: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    supported_query_methods: List[str] = field(default_factory=list)
    query_ready: bool = False
    query_backend: str = ""
    artifact_tables: List[str] = field(default_factory=list)
    artifact_mtime: str = ""
    graph_context_summary: Dict[str, Any] = field(default_factory=dict)
    config_json: Dict[str, Any] = field(default_factory=dict)
    prompt_overrides_json: Dict[str, Any] = field(default_factory=dict)
    graph_skill_ids: List[str] = field(default_factory=list)
    health: Dict[str, Any] = field(default_factory=dict)
    freshness_score: float = 0.0
    last_indexed_at: str = ""
    created_at: str = ""
    updated_at: str = ""
    summary_embedding: Optional[List[float]] = field(default=None, repr=False)

    def summary_text(self) -> str:
        parts = [
            self.display_name,
            self.domain_summary,
            " ".join(self.entity_samples[:12]),
            " ".join(self.relationship_samples[:12]),
            " ".join(self.supported_query_methods[:8]),
            " ".join(self.artifact_tables[:12]),
        ]
        return " ".join(part for part in parts if str(part).strip()).strip()


@dataclass
class GraphIndexSourceRecord:
    graph_source_id: str
    graph_id: str
    tenant_id: str = "local-dev"
    source_doc_id: str = ""
    source_path: str = ""
    source_title: str = ""
    source_type: str = ""
    created_at: str = ""


@dataclass
class GraphIndexRunRecord:
    run_id: str
    graph_id: str
    tenant_id: str = "local-dev"
    operation: str = ""
    status: str = "queued"
    detail: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    completed_at: str = ""


@dataclass
class GraphQueryCacheRecord:
    cache_id: str
    graph_id: str
    tenant_id: str = "local-dev"
    query_text: str = ""
    query_method: str = "local"
    response_json: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    expires_at: str = ""


def _row_to_graph_index(row: Dict[str, Any]) -> GraphIndexRecord:
    return GraphIndexRecord(
        graph_id=str(row.get("graph_id") or ""),
        tenant_id=str(row.get("tenant_id") or "local-dev"),
        collection_id=str(row.get("collection_id") or "default"),
        display_name=str(row.get("display_name") or ""),
        owner_admin_user_id=str(row.get("owner_admin_user_id") or ""),
        visibility=str(row.get("visibility") or "tenant"),
        backend=str(row.get("backend") or "microsoft_graphrag"),
        status=str(row.get("status") or "draft"),
        root_path=str(row.get("root_path") or ""),
        artifact_path=str(row.get("artifact_path") or ""),
        domain_summary=str(row.get("domain_summary") or ""),
        entity_samples=[str(item) for item in (row.get("entity_samples") or []) if str(item)],
        relationship_samples=[str(item) for item in (row.get("relationship_samples") or []) if str(item)],
        source_doc_ids=[str(item) for item in (row.get("source_doc_ids") or []) if str(item)],
        capabilities=[str(item) for item in (row.get("capabilities") or []) if str(item)],
        supported_query_methods=[str(item) for item in (row.get("supported_query_methods") or []) if str(item)],
        query_ready=bool(row.get("query_ready", False)),
        query_backend=str(row.get("query_backend") or ""),
        artifact_tables=[str(item) for item in (row.get("artifact_tables") or []) if str(item)],
        artifact_mtime=str(row.get("artifact_mtime") or ""),
        graph_context_summary=dict(row.get("graph_context_summary") or {}),
        config_json=dict(row.get("config_json") or {}),
        prompt_overrides_json=dict(row.get("prompt_overrides_json") or {}),
        graph_skill_ids=[str(item) for item in (row.get("graph_skill_ids") or []) if str(item)],
        health=dict(row.get("health") or {}),
        freshness_score=float(row.get("freshness_score") or 0.0),
        last_indexed_at=str(row.get("last_indexed_at") or ""),
        created_at=str(row.get("created_at") or ""),
        updated_at=str(row.get("updated_at") or ""),
    )


def _row_to_graph_source(row: Dict[str, Any]) -> GraphIndexSourceRecord:
    return GraphIndexSourceRecord(
        graph_source_id=str(row.get("graph_source_id") or ""),
        graph_id=str(row.get("graph_id") or ""),
        tenant_id=str(row.get("tenant_id") or "local-dev"),
        source_doc_id=str(row.get("source_doc_id") or ""),
        source_path=str(row.get("source_path") or ""),
        source_title=str(row.get("source_title") or ""),
        source_type=str(row.get("source_type") or ""),
        created_at=str(row.get("created_at") or ""),
    )


def _row_to_graph_run(row: Dict[str, Any]) -> GraphIndexRunRecord:
    return GraphIndexRunRecord(
        run_id=str(row.get("run_id") or ""),
        graph_id=str(row.get("graph_id") or ""),
        tenant_id=str(row.get("tenant_id") or "local-dev"),
        operation=str(row.get("operation") or ""),
        status=str(row.get("status") or "queued"),
        detail=str(row.get("detail") or ""),
        metadata=dict(row.get("metadata") or {}),
        started_at=str(row.get("started_at") or ""),
        completed_at=str(row.get("completed_at") or ""),
    )


class GraphIndexStore:
    def __init__(self, embed_fn: Callable[[str], List[float]] | None = None, embedding_dim: int = 768):
        self._embed = embed_fn
        self.embedding_dim = embedding_dim

    def _access_filter_sql(self, user_id: str = "") -> tuple[str, List[Any]]:
        clean_user_id = str(user_id or "").strip()
        if clean_user_id == "*":
            return ("", [])
        if clean_user_id:
            return (
                "AND (COALESCE(NULLIF(visibility, ''), 'tenant') <> 'private' OR owner_admin_user_id = %s)",
                [clean_user_id],
            )
        return ("AND COALESCE(NULLIF(visibility, ''), 'tenant') <> 'private'", [])

    def _embed_summary(self, record: GraphIndexRecord) -> List[float] | None:
        if record.summary_embedding is not None:
            return record.summary_embedding
        if self._embed is None:
            return None
        summary = record.summary_text()
        if not summary:
            return None
        return self._embed(summary)

    def upsert_index(self, record: GraphIndexRecord) -> None:
        embedding = self._embed_summary(record)
        now = _now_iso()
        created_at = record.created_at or now
        updated_at = record.updated_at or now
        last_indexed_at = record.last_indexed_at or now

        with get_conn() as conn:
            if embedding is not None:
                from pgvector.psycopg2 import register_vector

                register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO graph_indexes
                        (
                            graph_id,
                            tenant_id,
                            collection_id,
                            display_name,
                            owner_admin_user_id,
                            visibility,
                            backend,
                            status,
                            root_path,
                            artifact_path,
                            domain_summary,
                            entity_samples,
                            relationship_samples,
                            source_doc_ids,
                            capabilities,
                            supported_query_methods,
                            query_ready,
                            query_backend,
                            artifact_tables,
                            artifact_mtime,
                            graph_context_summary,
                            config_json,
                            prompt_overrides_json,
                            graph_skill_ids,
                            health,
                            freshness_score,
                            last_indexed_at,
                            created_at,
                            updated_at,
                            summary_embedding
                        )
                    VALUES
                        (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s::vector
                        )
                    ON CONFLICT (graph_id) DO UPDATE SET
                        tenant_id = EXCLUDED.tenant_id,
                        collection_id = EXCLUDED.collection_id,
                        display_name = EXCLUDED.display_name,
                        owner_admin_user_id = EXCLUDED.owner_admin_user_id,
                        visibility = EXCLUDED.visibility,
                        backend = EXCLUDED.backend,
                        status = EXCLUDED.status,
                        root_path = EXCLUDED.root_path,
                        artifact_path = EXCLUDED.artifact_path,
                        domain_summary = EXCLUDED.domain_summary,
                        entity_samples = EXCLUDED.entity_samples,
                        relationship_samples = EXCLUDED.relationship_samples,
                        source_doc_ids = EXCLUDED.source_doc_ids,
                        capabilities = EXCLUDED.capabilities,
                        supported_query_methods = EXCLUDED.supported_query_methods,
                        query_ready = EXCLUDED.query_ready,
                        query_backend = EXCLUDED.query_backend,
                        artifact_tables = EXCLUDED.artifact_tables,
                        artifact_mtime = EXCLUDED.artifact_mtime,
                        graph_context_summary = EXCLUDED.graph_context_summary,
                        config_json = EXCLUDED.config_json,
                        prompt_overrides_json = EXCLUDED.prompt_overrides_json,
                        graph_skill_ids = EXCLUDED.graph_skill_ids,
                        health = EXCLUDED.health,
                        freshness_score = EXCLUDED.freshness_score,
                        last_indexed_at = EXCLUDED.last_indexed_at,
                        updated_at = EXCLUDED.updated_at,
                        summary_embedding = EXCLUDED.summary_embedding
                    """,
                    (
                        record.graph_id,
                        record.tenant_id,
                        record.collection_id,
                        record.display_name or record.graph_id,
                        record.owner_admin_user_id,
                        record.visibility or "tenant",
                        record.backend,
                        record.status,
                        record.root_path,
                        record.artifact_path,
                        record.domain_summary,
                        psycopg2.extras.Json(list(record.entity_samples)),
                        psycopg2.extras.Json(list(record.relationship_samples)),
                        list(record.source_doc_ids),
                        list(record.capabilities),
                        list(record.supported_query_methods),
                        bool(record.query_ready),
                        record.query_backend,
                        list(record.artifact_tables),
                        record.artifact_mtime or None,
                        psycopg2.extras.Json(dict(record.graph_context_summary or {})),
                        psycopg2.extras.Json(dict(record.config_json or {})),
                        psycopg2.extras.Json(dict(record.prompt_overrides_json or {})),
                        list(record.graph_skill_ids),
                        psycopg2.extras.Json(dict(record.health)),
                        float(record.freshness_score or 0.0),
                        last_indexed_at,
                        created_at,
                        updated_at,
                        embedding,
                    ),
                )
            conn.commit()

    def get_index(self, graph_id: str, tenant_id: str, user_id: str = "") -> Optional[GraphIndexRecord]:
        access_sql, access_params = self._access_filter_sql(user_id)
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"SELECT * FROM graph_indexes WHERE graph_id = %s AND tenant_id = %s {access_sql}",
                    (graph_id, tenant_id, *access_params),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return _row_to_graph_index(dict(row))

    def list_indexes(
        self,
        *,
        tenant_id: str = "local-dev",
        user_id: str = "",
        collection_id: str = "",
        status: str = "",
        backend: str = "",
        limit: int = 100,
    ) -> List[GraphIndexRecord]:
        sql = ["SELECT * FROM graph_indexes WHERE tenant_id = %s"]
        params: List[Any] = [tenant_id]
        access_sql, access_params = self._access_filter_sql(user_id)
        sql.append(access_sql)
        params.extend(access_params)
        if collection_id:
            sql.append("AND collection_id = %s")
            params.append(collection_id)
        if status:
            sql.append("AND status = %s")
            params.append(status)
        if backend:
            sql.append("AND backend = %s")
            params.append(backend)
        sql.append("ORDER BY COALESCE(last_indexed_at, created_at) DESC, display_name ASC LIMIT %s")
        params.append(max(1, int(limit)))

        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(" ".join(sql), params)
                rows = cur.fetchall()
        return [_row_to_graph_index(dict(row)) for row in rows]

    def search_indexes(
        self,
        query: str,
        *,
        tenant_id: str = "local-dev",
        user_id: str = "",
        collection_id: str = "",
        limit: int = 6,
    ) -> List[GraphIndexRecord]:
        clean_query = str(query or "").strip()
        if not clean_query:
            return self.list_indexes(tenant_id=tenant_id, user_id=user_id, collection_id=collection_id, limit=limit)

        access_sql, access_params = self._access_filter_sql(user_id)

        if self._embed is not None:
            embedding = self._embed(clean_query)
            with get_conn() as conn:
                from pgvector.psycopg2 import register_vector

                register_vector(conn)
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    if collection_id:
                        cur.execute(
                            f"""
                            SELECT *
                            FROM graph_indexes
                            WHERE tenant_id = %s
                              {access_sql}
                              AND collection_id = %s
                              AND summary_embedding IS NOT NULL
                            ORDER BY summary_embedding <=> %s::vector
                            LIMIT %s
                            """,
                            (tenant_id, *access_params, collection_id, embedding, max(1, int(limit))),
                        )
                    else:
                        cur.execute(
                            f"""
                            SELECT *
                            FROM graph_indexes
                            WHERE tenant_id = %s
                              {access_sql}
                              AND summary_embedding IS NOT NULL
                            ORDER BY summary_embedding <=> %s::vector
                            LIMIT %s
                            """,
                            (tenant_id, *access_params, embedding, max(1, int(limit))),
                        )
                    rows = cur.fetchall()
            if rows:
                return [_row_to_graph_index(dict(row)) for row in rows]

        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if collection_id:
                    cur.execute(
                        f"""
                        SELECT *
                        FROM graph_indexes
                        WHERE tenant_id = %s
                          {access_sql}
                          AND collection_id = %s
                          AND (
                            lower(display_name) LIKE lower(%s)
                            OR lower(domain_summary) LIKE lower(%s)
                          )
                        ORDER BY COALESCE(last_indexed_at, created_at) DESC, display_name ASC
                        LIMIT %s
                        """,
                        (
                            tenant_id,
                            *access_params,
                            collection_id,
                            f"%{clean_query}%",
                            f"%{clean_query}%",
                            max(1, int(limit)),
                        ),
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT *
                        FROM graph_indexes
                        WHERE tenant_id = %s
                          {access_sql}
                          AND (
                            lower(display_name) LIKE lower(%s)
                            OR lower(domain_summary) LIKE lower(%s)
                          )
                        ORDER BY COALESCE(last_indexed_at, created_at) DESC, display_name ASC
                        LIMIT %s
                        """,
                        (
                            tenant_id,
                            *access_params,
                            f"%{clean_query}%",
                            f"%{clean_query}%",
                            max(1, int(limit)),
                        ),
                    )
                rows = cur.fetchall()
        return [_row_to_graph_index(dict(row)) for row in rows]

    def update_index_status(
        self,
        graph_id: str,
        tenant_id: str,
        *,
        status: str,
        health: Dict[str, Any] | None = None,
    ) -> bool:
        with get_conn() as conn:
            with conn.cursor() as cur:
                if health is None:
                    cur.execute(
                        """
                        UPDATE graph_indexes
                        SET status = %s,
                            updated_at = %s
                        WHERE graph_id = %s AND tenant_id = %s
                        """,
                        (status, _now_iso(), graph_id, tenant_id),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE graph_indexes
                        SET status = %s,
                            health = %s,
                            updated_at = %s
                        WHERE graph_id = %s AND tenant_id = %s
                        """,
                        (status, psycopg2.extras.Json(dict(health)), _now_iso(), graph_id, tenant_id),
                    )
                updated = cur.rowcount > 0
            conn.commit()
        return updated

    def delete_index(self, graph_id: str, tenant_id: str) -> Dict[str, int]:
        deleted: Dict[str, int] = {}
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM entity_mentions WHERE graph_id = %s AND tenant_id = %s",
                    (graph_id, tenant_id),
                )
                deleted["entity_mentions"] = cur.rowcount
                cur.execute(
                    "DELETE FROM canonical_entities WHERE graph_id = %s AND tenant_id = %s",
                    (graph_id, tenant_id),
                )
                deleted["canonical_entities"] = cur.rowcount
                cur.execute(
                    "DELETE FROM skills WHERE graph_id = %s AND tenant_id = %s",
                    (graph_id, tenant_id),
                )
                deleted["skills"] = cur.rowcount
                cur.execute(
                    """
                    DELETE FROM auth_role_permissions
                    WHERE tenant_id = %s
                      AND resource_type = 'graph'
                      AND resource_selector = %s
                    """,
                    (tenant_id, graph_id),
                )
                deleted["auth_role_permissions"] = cur.rowcount
                cur.execute("DELETE FROM graph_indexes WHERE graph_id = %s AND tenant_id = %s", (graph_id, tenant_id))
                deleted["graph_indexes"] = cur.rowcount
            conn.commit()
        return deleted


class GraphIndexSourceStore:
    def list_sources(self, graph_id: str, *, tenant_id: str = "local-dev") -> List[GraphIndexSourceRecord]:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM graph_index_sources
                    WHERE tenant_id = %s AND graph_id = %s
                    ORDER BY source_title ASC, source_path ASC
                    """,
                    (tenant_id, graph_id),
                )
                rows = cur.fetchall()
        return [_row_to_graph_source(dict(row)) for row in rows]

    def replace_sources(
        self,
        graph_id: str,
        *,
        tenant_id: str = "local-dev",
        sources: List[GraphIndexSourceRecord] | None = None,
    ) -> None:
        rows = list(sources or [])
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM graph_index_sources WHERE tenant_id = %s AND graph_id = %s",
                    (tenant_id, graph_id),
                )
                for row in rows:
                    cur.execute(
                        """
                        INSERT INTO graph_index_sources
                            (graph_source_id, graph_id, tenant_id, source_doc_id, source_path, source_title, source_type, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (graph_source_id) DO UPDATE SET
                            source_doc_id = EXCLUDED.source_doc_id,
                            source_path = EXCLUDED.source_path,
                            source_title = EXCLUDED.source_title,
                            source_type = EXCLUDED.source_type
                        """,
                        (
                            row.graph_source_id,
                            row.graph_id,
                            row.tenant_id,
                            row.source_doc_id,
                            row.source_path,
                            row.source_title,
                            row.source_type,
                            row.created_at or _now_iso(),
                        ),
                    )
            conn.commit()


class GraphIndexRunStore:
    def upsert_run(self, record: GraphIndexRunRecord) -> None:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO graph_index_runs
                        (run_id, tenant_id, graph_id, operation, status, detail, metadata, started_at, completed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        detail = EXCLUDED.detail,
                        metadata = EXCLUDED.metadata,
                        completed_at = EXCLUDED.completed_at
                    """,
                    (
                        record.run_id,
                        record.tenant_id,
                        record.graph_id,
                        record.operation,
                        record.status,
                        record.detail,
                        psycopg2.extras.Json(dict(record.metadata)),
                        record.started_at or _now_iso(),
                        record.completed_at or None,
                    ),
                )
            conn.commit()

    def list_runs(self, graph_id: str, *, tenant_id: str = "local-dev", limit: int = 20) -> List[GraphIndexRunRecord]:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM graph_index_runs
                    WHERE tenant_id = %s AND graph_id = %s
                    ORDER BY started_at DESC
                    LIMIT %s
                    """,
                    (tenant_id, graph_id, max(1, int(limit))),
                )
                rows = cur.fetchall()
        return [_row_to_graph_run(dict(row)) for row in rows]

    def list_runs_by_status(
        self,
        *,
        tenant_id: str = "local-dev",
        status: str = "",
        graph_id: str = "",
        limit: int = 100,
    ) -> List[GraphIndexRunRecord]:
        clauses = ["tenant_id = %s"]
        params: List[Any] = [tenant_id]
        if status:
            clauses.append("status = %s")
            params.append(status)
        if graph_id:
            clauses.append("graph_id = %s")
            params.append(graph_id)
        params.append(max(1, int(limit)))
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT *
                    FROM graph_index_runs
                    WHERE {' AND '.join(clauses)}
                    ORDER BY started_at DESC
                    LIMIT %s
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()
        return [_row_to_graph_run(dict(row)) for row in rows]

    def delete_run(
        self,
        run_id: str,
        *,
        tenant_id: str = "local-dev",
        graph_id: str = "",
        statuses: Sequence[str] | None = None,
    ) -> int:
        clauses = ["tenant_id = %s", "run_id = %s"]
        params: List[Any] = [tenant_id, run_id]
        if graph_id:
            clauses.append("graph_id = %s")
            params.append(graph_id)
        allowed_statuses = [str(item) for item in (statuses or []) if str(item).strip()]
        if allowed_statuses:
            clauses.append("status = ANY(%s)")
            params.append(allowed_statuses)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM graph_index_runs WHERE {' AND '.join(clauses)}",
                    tuple(params),
                )
                deleted = int(cur.rowcount or 0)
            conn.commit()
        return deleted

    def delete_runs_by_status(
        self,
        *,
        tenant_id: str = "local-dev",
        status: str,
        graph_id: str = "",
    ) -> int:
        clauses = ["tenant_id = %s", "status = %s"]
        params: List[Any] = [tenant_id, status]
        if graph_id:
            clauses.append("graph_id = %s")
            params.append(graph_id)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM graph_index_runs WHERE {' AND '.join(clauses)}",
                    tuple(params),
                )
                deleted = int(cur.rowcount or 0)
            conn.commit()
        return deleted


class GraphQueryCacheStore:
    def get_cached(
        self,
        *,
        graph_id: str,
        tenant_id: str,
        query_text: str,
        query_method: str,
        now: dt.datetime | None = None,
    ) -> Optional[GraphQueryCacheRecord]:
        now_value = now or dt.datetime.now(dt.timezone.utc)
        cache_id = _sha1(f"{tenant_id}:{graph_id}:{query_method}:{query_text.lower().strip()}")
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM graph_query_cache
                    WHERE cache_id = %s
                      AND tenant_id = %s
                      AND graph_id = %s
                      AND (expires_at IS NULL OR expires_at >= %s)
                    """,
                    (cache_id, tenant_id, graph_id, now_value.isoformat()),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return GraphQueryCacheRecord(
            cache_id=str(row.get("cache_id") or ""),
            graph_id=str(row.get("graph_id") or ""),
            tenant_id=str(row.get("tenant_id") or "local-dev"),
            query_text=str(row.get("query_text") or ""),
            query_method=str(row.get("query_method") or "local"),
            response_json=dict(row.get("response_json") or {}),
            created_at=str(row.get("created_at") or ""),
            expires_at=str(row.get("expires_at") or ""),
        )

    def put_cached(
        self,
        *,
        graph_id: str,
        tenant_id: str,
        query_text: str,
        query_method: str,
        response_json: Dict[str, Any],
        ttl_seconds: int = 900,
    ) -> GraphQueryCacheRecord:
        created_at = dt.datetime.now(dt.timezone.utc)
        expires_at = created_at + dt.timedelta(seconds=max(1, int(ttl_seconds)))
        cache_id = _sha1(f"{tenant_id}:{graph_id}:{query_method}:{query_text.lower().strip()}")
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO graph_query_cache
                        (cache_id, tenant_id, graph_id, query_text, query_method, response_json, created_at, expires_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (cache_id) DO UPDATE SET
                        response_json = EXCLUDED.response_json,
                        created_at = EXCLUDED.created_at,
                        expires_at = EXCLUDED.expires_at
                    """,
                    (
                        cache_id,
                        tenant_id,
                        graph_id,
                        query_text,
                        query_method,
                        psycopg2.extras.Json(dict(response_json)),
                        created_at.isoformat(),
                        expires_at.isoformat(),
                    ),
                )
            conn.commit()
        return GraphQueryCacheRecord(
            cache_id=cache_id,
            graph_id=graph_id,
            tenant_id=tenant_id,
            query_text=query_text,
            query_method=query_method,
            response_json=dict(response_json),
            created_at=created_at.isoformat(),
            expires_at=expires_at.isoformat(),
        )


__all__ = [
    "GraphIndexRecord",
    "GraphIndexRunRecord",
    "GraphIndexSourceRecord",
    "GraphIndexStore",
    "GraphIndexRunStore",
    "GraphIndexSourceStore",
    "GraphQueryCacheRecord",
    "GraphQueryCacheStore",
]
