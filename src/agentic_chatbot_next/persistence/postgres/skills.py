from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import psycopg2.extras

from agentic_chatbot_next.persistence.postgres.connection import get_conn

if TYPE_CHECKING:
    from agentic_chatbot_next.skills.telemetry import SkillTelemetryEvent


@dataclass
class SkillPackRecord:
    skill_id: str
    name: str
    agent_scope: str
    checksum: str
    tenant_id: str = "local-dev"
    graph_id: str = ""
    tool_tags: List[str] = field(default_factory=list)
    task_tags: List[str] = field(default_factory=list)
    version: str = "1"
    enabled: bool = True
    source_path: str = ""
    description: str = ""
    retrieval_profile: str = ""
    controller_hints: Dict[str, Any] = field(default_factory=dict)
    coverage_goal: str = ""
    result_mode: str = ""
    body_markdown: str = ""
    owner_user_id: str = ""
    visibility: str = "global"
    status: str = "active"
    version_parent: str = ""
    updated_at: str = ""
    kind: str = "retrievable"
    execution_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillChunkMatch:
    skill_id: str
    name: str
    agent_scope: str
    content: str
    chunk_index: int
    score: float
    graph_id: str = ""
    tool_tags: List[str] = field(default_factory=list)
    task_tags: List[str] = field(default_factory=list)
    retrieval_profile: str = ""
    controller_hints: Dict[str, Any] = field(default_factory=dict)
    coverage_goal: str = ""
    result_mode: str = ""
    owner_user_id: str = ""
    visibility: str = "global"
    status: str = "active"
    version_parent: str = ""
    kind: str = "retrievable"
    execution_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillTelemetryEventRecord:
    event_id: str
    tenant_id: str
    skill_id: str
    skill_family_id: str
    query: str
    answer_quality: str
    agent_name: str = ""
    session_id: str = ""
    created_at: str = ""


_VALID_VISIBILITY = {"private", "tenant", "global"}
_VALID_STATUS = {"draft", "active", "archived"}
_VALID_KINDS = {"retrievable", "executable", "hybrid"}


def _normalize_visibility(value: str) -> str:
    normalized = str(value or "global").strip().lower()
    return normalized if normalized in _VALID_VISIBILITY else "global"


def _normalize_status(value: str) -> str:
    normalized = str(value or "active").strip().lower()
    return normalized if normalized in _VALID_STATUS else "active"


def _normalize_kind(value: str) -> str:
    normalized = str(value or "retrievable").strip().lower()
    return normalized if normalized in _VALID_KINDS else "retrievable"


def _scope_rank(row: Dict[str, Any], *, owner_user_id: str = "") -> int:
    visibility = _normalize_visibility(str(row.get("visibility") or "global"))
    row_owner = str(row.get("owner_user_id") or "")
    if visibility == "private" and owner_user_id and row_owner == owner_user_id:
        return 3
    if visibility == "tenant":
        return 2
    return 1


def _normalize_accessible_skill_family_ids(values: List[str] | None) -> tuple[list[str], bool]:
    normalized = [str(item).strip() for item in list(values or []) if str(item).strip()]
    allow_all = "*" in normalized
    deduped: list[str] = []
    seen: set[str] = set()
    for item in normalized:
        if item == "*" or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped, allow_all


class SkillStore:
    """Persistence for retrievable skill packs and their vectorized chunks."""

    def __init__(self, embed_fn: Callable[[str], List[float]], embedding_dim: int = 768):
        self._embed = embed_fn
        self.embedding_dim = embedding_dim

    def upsert_skill_pack(self, record: SkillPackRecord, chunks: List[str]) -> None:
        timestamp = record.updated_at or (dt.datetime.utcnow().isoformat() + "Z")
        visibility = _normalize_visibility(record.visibility)
        status = _normalize_status(record.status)
        kind = _normalize_kind(record.kind)
        version_parent = str(record.version_parent or record.skill_id)
        rows: List[Tuple[Any, ...]] = []
        for index, content in enumerate(chunks):
            rows.append((
                f"{record.skill_id}#chunk{index:04d}",
                record.skill_id,
                record.tenant_id,
                index,
                content,
                self._embed(content),
            ))

        with get_conn() as conn:
            from pgvector.psycopg2 import register_vector

            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO skills
                        (skill_id, tenant_id, owner_user_id, graph_id, name, agent_scope, tool_tags, task_tags,
                         version, enabled, source_path, checksum, description,
                         retrieval_profile, controller_hints, coverage_goal, result_mode,
                         visibility, status, version_parent, body_markdown, kind, execution_config, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (skill_id) DO UPDATE SET
                        tenant_id = EXCLUDED.tenant_id,
                        owner_user_id = EXCLUDED.owner_user_id,
                        graph_id = EXCLUDED.graph_id,
                        name = EXCLUDED.name,
                        agent_scope = EXCLUDED.agent_scope,
                        tool_tags = EXCLUDED.tool_tags,
                        task_tags = EXCLUDED.task_tags,
                        version = EXCLUDED.version,
                        enabled = EXCLUDED.enabled,
                        source_path = EXCLUDED.source_path,
                        checksum = EXCLUDED.checksum,
                        description = EXCLUDED.description,
                        retrieval_profile = EXCLUDED.retrieval_profile,
                        controller_hints = EXCLUDED.controller_hints,
                        coverage_goal = EXCLUDED.coverage_goal,
                        result_mode = EXCLUDED.result_mode,
                        visibility = EXCLUDED.visibility,
                        status = EXCLUDED.status,
                        version_parent = EXCLUDED.version_parent,
                        body_markdown = EXCLUDED.body_markdown,
                        kind = EXCLUDED.kind,
                        execution_config = EXCLUDED.execution_config,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        record.skill_id,
                        record.tenant_id,
                        record.owner_user_id,
                        record.graph_id,
                        record.name,
                        record.agent_scope,
                        record.tool_tags,
                        record.task_tags,
                        record.version,
                        record.enabled,
                        record.source_path,
                        record.checksum,
                        record.description,
                        record.retrieval_profile,
                        psycopg2.extras.Json(record.controller_hints or {}),
                        record.coverage_goal,
                        record.result_mode,
                        visibility,
                        status,
                        version_parent,
                        record.body_markdown,
                        kind,
                        psycopg2.extras.Json(record.execution_config or {}),
                        timestamp,
                    ),
                )
                cur.execute(
                    "DELETE FROM skill_chunks WHERE tenant_id = %s AND skill_id = %s",
                    (record.tenant_id, record.skill_id),
                )
                if rows:
                    psycopg2.extras.execute_values(
                        cur,
                        """
                        INSERT INTO skill_chunks
                            (skill_chunk_id, skill_id, tenant_id, chunk_index, content, embedding)
                        VALUES %s
                        ON CONFLICT (skill_chunk_id) DO UPDATE SET
                            tenant_id = EXCLUDED.tenant_id,
                            chunk_index = EXCLUDED.chunk_index,
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding
                        """,
                        rows,
                        template="(%s, %s, %s, %s, %s, %s::vector)",
                    )
            conn.commit()

    def list_skill_packs(
        self,
        *,
        tenant_id: str = "local-dev",
        agent_scope: str = "",
        enabled_only: bool = False,
        owner_user_id: str = "",
        visibility: str = "",
        status: str = "",
        graph_id: str = "",
        accessible_skill_family_ids: Optional[List[str]] = None,
    ) -> List[SkillPackRecord]:
        sql = "SELECT * FROM skills WHERE tenant_id = %s"
        params: List[Any] = [tenant_id]
        scoped_skill_family_ids, allow_all_skill_families = _normalize_accessible_skill_family_ids(
            accessible_skill_family_ids
        )
        if accessible_skill_family_ids is not None and not allow_all_skill_families and not scoped_skill_family_ids:
            return []
        if agent_scope:
            sql += " AND agent_scope = %s"
            params.append(agent_scope)
        if enabled_only:
            sql += " AND enabled = TRUE"
        if accessible_skill_family_ids is not None:
            if not allow_all_skill_families:
                sql += " AND COALESCE(NULLIF(version_parent, ''), skill_id) = ANY(%s)"
                params.append(scoped_skill_family_ids)
        elif owner_user_id:
            sql += " AND (visibility <> 'private' OR owner_user_id = %s)"
            params.append(owner_user_id)
        if visibility:
            sql += " AND visibility = %s"
            params.append(_normalize_visibility(visibility))
        if status:
            sql += " AND status = %s"
            params.append(_normalize_status(status))
        if graph_id:
            sql += " AND graph_id = %s"
            params.append(str(graph_id))
        sql += " ORDER BY agent_scope, name"

        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [_row_to_skill_pack(dict(row)) for row in rows]

    def get_skill_pack(
        self,
        skill_id: str,
        *,
        tenant_id: str = "local-dev",
        owner_user_id: str = "",
        accessible_skill_family_ids: Optional[List[str]] = None,
    ) -> Optional[SkillPackRecord]:
        scoped_skill_family_ids, allow_all_skill_families = _normalize_accessible_skill_family_ids(
            accessible_skill_family_ids
        )
        if accessible_skill_family_ids is not None and not allow_all_skill_families and not scoped_skill_family_ids:
            return None
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if accessible_skill_family_ids is not None:
                    if allow_all_skill_families:
                        cur.execute(
                            """
                            SELECT *
                            FROM skills
                            WHERE tenant_id = %s
                              AND skill_id = %s
                            LIMIT 1
                            """,
                            (tenant_id, skill_id),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT *
                            FROM skills
                            WHERE tenant_id = %s
                              AND skill_id = %s
                              AND COALESCE(NULLIF(version_parent, ''), skill_id) = ANY(%s)
                            LIMIT 1
                            """,
                            (tenant_id, skill_id, scoped_skill_family_ids),
                        )
                elif owner_user_id:
                    cur.execute(
                        """
                        SELECT *
                        FROM skills
                        WHERE tenant_id = %s
                          AND skill_id = %s
                          AND (visibility <> 'private' OR owner_user_id = %s)
                        LIMIT 1
                        """,
                        (tenant_id, skill_id, owner_user_id),
                    )
                else:
                    cur.execute(
                        "SELECT * FROM skills WHERE tenant_id = %s AND skill_id = %s LIMIT 1",
                        (tenant_id, skill_id),
                    )
                row = cur.fetchone()
        return _row_to_skill_pack(dict(row)) if row else None

    def get_skill_chunks(self, skill_id: str, *, tenant_id: str = "local-dev") -> List[Dict[str, Any]]:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT skill_chunk_id, skill_id, chunk_index, content
                    FROM skill_chunks
                    WHERE tenant_id = %s AND skill_id = %s
                    ORDER BY chunk_index
                    """,
                    (tenant_id, skill_id),
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def get_skill_packs_by_ids(
        self,
        skill_ids: List[str],
        *,
        tenant_id: str = "local-dev",
        owner_user_id: str = "",
        accessible_skill_family_ids: Optional[List[str]] = None,
    ) -> List[SkillPackRecord]:
        ordered_ids = [str(item).strip() for item in skill_ids if str(item).strip()]
        if not ordered_ids:
            return []
        rows: List[SkillPackRecord] = []
        for skill_id in ordered_ids:
            record = self.get_skill_pack(
                skill_id,
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
                accessible_skill_family_ids=accessible_skill_family_ids,
            )
            if record is None:
                continue
            rows.append(record)
        return rows

    def delete_skill_pack(self, skill_id: str, *, tenant_id: str = "local-dev") -> None:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM skills WHERE tenant_id = %s AND skill_id = %s", (tenant_id, skill_id))
            conn.commit()

    def set_skill_status(
        self,
        skill_id: str,
        *,
        tenant_id: str = "local-dev",
        status: str,
        enabled: bool | None = None,
    ) -> None:
        normalized_status = _normalize_status(status)
        with get_conn() as conn:
            with conn.cursor() as cur:
                if enabled is None:
                    cur.execute(
                        """
                        UPDATE skills
                        SET status = %s,
                            updated_at = %s
                        WHERE tenant_id = %s AND skill_id = %s
                        """,
                        (normalized_status, dt.datetime.utcnow().isoformat() + "Z", tenant_id, skill_id),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE skills
                        SET status = %s,
                            enabled = %s,
                            updated_at = %s
                        WHERE tenant_id = %s AND skill_id = %s
                        """,
                        (normalized_status, bool(enabled), dt.datetime.utcnow().isoformat() + "Z", tenant_id, skill_id),
                    )
            conn.commit()

    def list_skill_versions(
        self,
        version_parent: str,
        *,
        tenant_id: str = "local-dev",
        owner_user_id: str = "",
        accessible_skill_family_ids: Optional[List[str]] = None,
    ) -> List[SkillPackRecord]:
        scoped_skill_family_ids, allow_all_skill_families = _normalize_accessible_skill_family_ids(
            accessible_skill_family_ids
        )
        if accessible_skill_family_ids is not None and not allow_all_skill_families and version_parent not in scoped_skill_family_ids:
            return []
        sql = "SELECT * FROM skills WHERE tenant_id = %s AND version_parent = %s"
        params: List[Any] = [tenant_id, version_parent]
        if accessible_skill_family_ids is None and owner_user_id:
            sql += " AND (visibility <> 'private' OR owner_user_id = %s)"
            params.append(owner_user_id)
        sql += " ORDER BY updated_at DESC, skill_id DESC"
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [_row_to_skill_pack(dict(row)) for row in rows]

    def append_skill_telemetry_event(self, event: SkillTelemetryEvent | SkillTelemetryEventRecord) -> None:
        if hasattr(event, "to_dict"):
            payload = event.to_dict()
        else:
            payload = dict(getattr(event, "__dict__", {}) or {})
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO skill_telemetry_events
                        (event_id, tenant_id, skill_id, skill_family_id, query, answer_quality,
                         agent_name, session_id, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (event_id) DO NOTHING
                    """,
                    (
                        str(payload.get("event_id") or ""),
                        str(payload.get("tenant_id") or "local-dev"),
                        str(payload.get("skill_id") or ""),
                        str(payload.get("skill_family_id") or ""),
                        str(payload.get("query") or ""),
                        str(payload.get("answer_quality") or ""),
                        str(payload.get("agent_name") or ""),
                        str(payload.get("session_id") or ""),
                        str(payload.get("created_at") or dt.datetime.utcnow().isoformat() + "Z"),
                    ),
                )
            conn.commit()

    def list_skill_telemetry_events(
        self,
        *,
        tenant_id: str = "local-dev",
        skill_family_id: str = "",
        skill_id: str = "",
        session_id: str = "",
        limit: int = 200,
    ) -> List[SkillTelemetryEventRecord]:
        sql = "SELECT * FROM skill_telemetry_events WHERE tenant_id = %s"
        params: List[Any] = [tenant_id]
        if skill_family_id:
            sql += " AND skill_family_id = %s"
            params.append(str(skill_family_id))
        if skill_id:
            sql += " AND skill_id = %s"
            params.append(str(skill_id))
        if session_id:
            sql += " AND session_id = %s"
            params.append(str(session_id))
        sql += " ORDER BY created_at DESC LIMIT %s"
        params.append(max(1, int(limit)))
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [
            SkillTelemetryEventRecord(
                event_id=str(row.get("event_id") or ""),
                tenant_id=str(row.get("tenant_id") or "local-dev"),
                skill_id=str(row.get("skill_id") or ""),
                skill_family_id=str(row.get("skill_family_id") or ""),
                query=str(row.get("query") or ""),
                answer_quality=str(row.get("answer_quality") or ""),
                agent_name=str(row.get("agent_name") or ""),
                session_id=str(row.get("session_id") or ""),
                created_at=str(row.get("created_at") or ""),
            )
            for row in rows
        ]

    def vector_search(
        self,
        query: str,
        *,
        tenant_id: str = "local-dev",
        top_k: int = 4,
        agent_scope: str = "",
        tool_tags: Optional[List[str]] = None,
        task_tags: Optional[List[str]] = None,
        enabled_only: bool = True,
        owner_user_id: str = "",
        graph_ids: Optional[List[str]] = None,
        accessible_skill_family_ids: Optional[List[str]] = None,
    ) -> List[SkillChunkMatch]:
        embedding = self._embed(query)
        tool_tags = [tag for tag in (tool_tags or []) if tag]
        task_tags = [tag for tag in (task_tags or []) if tag]
        scoped_graph_ids = [str(item).strip() for item in (graph_ids or []) if str(item).strip()]
        scoped_skill_family_ids, allow_all_skill_families = _normalize_accessible_skill_family_ids(
            accessible_skill_family_ids
        )
        if accessible_skill_family_ids is not None and not allow_all_skill_families and not scoped_skill_family_ids:
            return []

        sql = """
            SELECT sc.skill_id,
                   s.name,
                   s.agent_scope,
                   s.graph_id,
                   s.tool_tags,
                   s.task_tags,
                   s.retrieval_profile,
                   s.controller_hints,
                   s.coverage_goal,
                   s.result_mode,
                   s.owner_user_id,
                   s.visibility,
                   s.status,
                   s.version_parent,
                   s.kind,
                   s.execution_config,
                   s.updated_at,
                   sc.content,
                   sc.chunk_index,
                   1 - (sc.embedding <=> %s::vector) AS score
            FROM skill_chunks sc
            JOIN skills s ON s.skill_id = sc.skill_id AND s.tenant_id = sc.tenant_id
            WHERE sc.tenant_id = %s
              AND s.tenant_id = %s
        """
        params: List[Any] = [embedding, tenant_id, tenant_id]
        if enabled_only:
            sql += " AND s.enabled = TRUE"
            sql += " AND s.status = 'active'"
        if accessible_skill_family_ids is not None:
            if not allow_all_skill_families:
                sql += " AND COALESCE(NULLIF(s.version_parent, ''), s.skill_id) = ANY(%s)"
                params.append(scoped_skill_family_ids)
        elif owner_user_id:
            sql += " AND (s.visibility <> 'private' OR s.owner_user_id = %s)"
            params.append(owner_user_id)
        else:
            sql += " AND s.visibility <> 'private'"
        if agent_scope:
            sql += " AND s.agent_scope = %s"
            params.append(agent_scope)
        if scoped_graph_ids:
            sql += " AND (s.graph_id = '' OR s.graph_id = ANY(%s))"
            params.append(scoped_graph_ids)
        if tool_tags:
            sql += " AND s.tool_tags && %s"
            params.append(tool_tags)
        if task_tags:
            sql += " AND s.task_tags && %s"
            params.append(task_tags)
        sql += " ORDER BY sc.embedding <=> %s::vector LIMIT %s"
        params.extend([embedding, max(int(top_k) * 8, int(top_k))])

        with get_conn() as conn:
            from pgvector.psycopg2 import register_vector

            register_vector(conn)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        scoped_rows = sorted(
            [dict(row) for row in rows],
            key=lambda row: (
                -_scope_rank(row, owner_user_id=owner_user_id),
                -float(row.get("score") or 0.0),
                str(row.get("updated_at") or ""),
            ),
        )

        deduped: List[Dict[str, Any]] = []
        seen_families: set[str] = set()
        seen_chunks: set[tuple[str, int]] = set()
        for row in scoped_rows:
            family = str(row.get("version_parent") or row.get("skill_id") or "")
            chunk_key = (str(row.get("skill_id") or ""), int(row.get("chunk_index") or 0))
            if chunk_key in seen_chunks:
                continue
            if family and family in seen_families and len(deduped) >= int(top_k):
                continue
            seen_chunks.add(chunk_key)
            if family:
                seen_families.add(family)
            deduped.append(row)
            if len(deduped) >= int(top_k):
                break

        return [
            SkillChunkMatch(
                skill_id=row["skill_id"],
                name=row["name"],
                agent_scope=row["agent_scope"],
                content=row["content"],
                chunk_index=int(row["chunk_index"]),
                score=float(row["score"]),
                graph_id=str(row.get("graph_id") or ""),
                tool_tags=list(row.get("tool_tags") or []),
                task_tags=list(row.get("task_tags") or []),
                retrieval_profile=str(row.get("retrieval_profile") or ""),
                controller_hints=dict(row.get("controller_hints") or {}),
                coverage_goal=str(row.get("coverage_goal") or ""),
                result_mode=str(row.get("result_mode") or ""),
                owner_user_id=str(row.get("owner_user_id") or ""),
                visibility=str(row.get("visibility") or "global"),
                status=str(row.get("status") or "active"),
                version_parent=str(row.get("version_parent") or row.get("skill_id") or ""),
                kind=_normalize_kind(str(row.get("kind") or "retrievable")),
                execution_config=dict(row.get("execution_config") or {}),
            )
            for row in deduped
        ]


def _row_to_skill_pack(row: Dict[str, Any]) -> SkillPackRecord:
    return SkillPackRecord(
        skill_id=row.get("skill_id", ""),
        tenant_id=row.get("tenant_id") or "local-dev",
        graph_id=row.get("graph_id") or "",
        name=row.get("name", ""),
        agent_scope=row.get("agent_scope", ""),
        tool_tags=list(row.get("tool_tags") or []),
        task_tags=list(row.get("task_tags") or []),
        version=row.get("version") or "1",
        enabled=bool(row.get("enabled", True)),
        source_path=row.get("source_path") or "",
        checksum=row.get("checksum") or "",
        description=row.get("description") or "",
        retrieval_profile=row.get("retrieval_profile") or "",
        controller_hints=dict(row.get("controller_hints") or {}),
        coverage_goal=row.get("coverage_goal") or "",
        result_mode=row.get("result_mode") or "",
        body_markdown=row.get("body_markdown") or "",
        owner_user_id=row.get("owner_user_id") or "",
        visibility=_normalize_visibility(row.get("visibility") or "global"),
        status=_normalize_status(row.get("status") or "active"),
        version_parent=row.get("version_parent") or row.get("skill_id") or "",
        updated_at=str(row.get("updated_at") or ""),
        kind=_normalize_kind(row.get("kind") or "retrievable"),
        execution_config=dict(row.get("execution_config") or {}),
    )
