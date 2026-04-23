from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import psycopg2.extras

from agentic_chatbot_next.persistence.postgres.connection import get_conn

VALID_MCP_AUTH_TYPES = {"none", "bearer"}
VALID_MCP_CONNECTION_STATUSES = {"active", "disabled", "error"}
VALID_MCP_VISIBILITIES = {"private", "tenant"}
VALID_MCP_TOOL_STATUSES = {"active", "disabled", "missing", "error"}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def _string_list(value: Any) -> List[str]:
    seen: set[str] = set()
    items: List[str] = []
    for item in list(value or []):
        clean = str(item or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        items.append(clean)
    return items


def _normalize_auth_type(value: str) -> str:
    normalized = str(value or "none").strip().lower()
    return normalized if normalized in VALID_MCP_AUTH_TYPES else "none"


def _normalize_connection_status(value: str) -> str:
    normalized = str(value or "active").strip().lower()
    return normalized if normalized in VALID_MCP_CONNECTION_STATUSES else "active"


def _normalize_visibility(value: str) -> str:
    normalized = str(value or "private").strip().lower()
    return normalized if normalized in VALID_MCP_VISIBILITIES else "private"


def _normalize_tool_status(value: str) -> str:
    normalized = str(value or "active").strip().lower()
    return normalized if normalized in VALID_MCP_TOOL_STATUSES else "active"


@dataclass
class McpConnectionRecord:
    connection_id: str
    tenant_id: str = "local-dev"
    owner_user_id: str = "local-cli"
    display_name: str = ""
    connection_slug: str = ""
    server_url: str = ""
    auth_type: str = "none"
    encrypted_secret: str = ""
    status: str = "active"
    allowed_agents: List[str] = field(default_factory=lambda: ["general"])
    visibility: str = "private"
    health: Dict[str, Any] = field(default_factory=dict)
    metadata_json: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    last_tested_at: str = ""
    last_refreshed_at: str = ""

    def to_dict(self, *, include_secret: bool = False) -> Dict[str, Any]:
        payload = {
            "connection_id": self.connection_id,
            "tenant_id": self.tenant_id,
            "owner_user_id": self.owner_user_id,
            "display_name": self.display_name,
            "connection_slug": self.connection_slug,
            "server_url": self.server_url,
            "auth_type": self.auth_type,
            "status": self.status,
            "allowed_agents": list(self.allowed_agents),
            "visibility": self.visibility,
            "health": dict(self.health),
            "metadata_json": dict(self.metadata_json),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_tested_at": self.last_tested_at,
            "last_refreshed_at": self.last_refreshed_at,
            "secret_configured": bool(self.encrypted_secret),
        }
        if include_secret:
            payload["encrypted_secret"] = self.encrypted_secret
        return payload


@dataclass
class McpToolCatalogRecord:
    tool_id: str
    connection_id: str
    tenant_id: str = "local-dev"
    owner_user_id: str = "local-cli"
    raw_tool_name: str = ""
    registry_name: str = ""
    tool_slug: str = ""
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    read_only: bool = False
    destructive: bool = True
    background_safe: bool = False
    should_defer: bool = True
    search_hint: str = ""
    defer_priority: int = 50
    enabled: bool = True
    status: str = "active"
    checksum: str = ""
    metadata_json: Dict[str, Any] = field(default_factory=dict)
    first_seen_at: str = ""
    last_seen_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "connection_id": self.connection_id,
            "tenant_id": self.tenant_id,
            "owner_user_id": self.owner_user_id,
            "raw_tool_name": self.raw_tool_name,
            "registry_name": self.registry_name,
            "tool_slug": self.tool_slug,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "read_only": bool(self.read_only),
            "destructive": bool(self.destructive),
            "background_safe": bool(self.background_safe),
            "should_defer": bool(self.should_defer),
            "search_hint": self.search_hint,
            "defer_priority": int(self.defer_priority or 50),
            "enabled": bool(self.enabled),
            "status": self.status,
            "checksum": self.checksum,
            "metadata_json": dict(self.metadata_json),
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
        }


def _row_to_connection(row: Dict[str, Any]) -> McpConnectionRecord:
    return McpConnectionRecord(
        connection_id=str(row.get("connection_id") or ""),
        tenant_id=str(row.get("tenant_id") or "local-dev"),
        owner_user_id=str(row.get("owner_user_id") or "local-cli"),
        display_name=str(row.get("display_name") or ""),
        connection_slug=str(row.get("connection_slug") or ""),
        server_url=str(row.get("server_url") or ""),
        auth_type=_normalize_auth_type(str(row.get("auth_type") or "none")),
        encrypted_secret=str(row.get("encrypted_secret") or ""),
        status=_normalize_connection_status(str(row.get("status") or "active")),
        allowed_agents=_string_list(row.get("allowed_agents")),
        visibility=_normalize_visibility(str(row.get("visibility") or "private")),
        health=dict(row.get("health") or {}),
        metadata_json=dict(row.get("metadata_json") or {}),
        created_at=str(row.get("created_at") or ""),
        updated_at=str(row.get("updated_at") or ""),
        last_tested_at=str(row.get("last_tested_at") or ""),
        last_refreshed_at=str(row.get("last_refreshed_at") or ""),
    )


def _row_to_tool(row: Dict[str, Any]) -> McpToolCatalogRecord:
    return McpToolCatalogRecord(
        tool_id=str(row.get("tool_id") or ""),
        connection_id=str(row.get("connection_id") or ""),
        tenant_id=str(row.get("tenant_id") or "local-dev"),
        owner_user_id=str(row.get("owner_user_id") or "local-cli"),
        raw_tool_name=str(row.get("raw_tool_name") or ""),
        registry_name=str(row.get("registry_name") or ""),
        tool_slug=str(row.get("tool_slug") or ""),
        description=str(row.get("description") or ""),
        input_schema=dict(row.get("input_schema") or {}),
        read_only=bool(row.get("read_only", False)),
        destructive=bool(row.get("destructive", True)),
        background_safe=bool(row.get("background_safe", False)),
        should_defer=bool(row.get("should_defer", True)),
        search_hint=str(row.get("search_hint") or ""),
        defer_priority=int(row.get("defer_priority", 50) or 50),
        enabled=bool(row.get("enabled", True)),
        status=_normalize_tool_status(str(row.get("status") or "active")),
        checksum=str(row.get("checksum") or ""),
        metadata_json=dict(row.get("metadata_json") or {}),
        first_seen_at=str(row.get("first_seen_at") or ""),
        last_seen_at=str(row.get("last_seen_at") or ""),
    )


class McpConnectionStore:
    def create_connection(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        display_name: str,
        connection_slug: str,
        server_url: str,
        auth_type: str = "none",
        encrypted_secret: str = "",
        allowed_agents: Iterable[str] | None = None,
        visibility: str = "private",
        metadata_json: Optional[Dict[str, Any]] = None,
    ) -> McpConnectionRecord:
        record = McpConnectionRecord(
            connection_id=_new_id("mcp_conn"),
            tenant_id=str(tenant_id or "local-dev"),
            owner_user_id=str(owner_user_id or "local-cli"),
            display_name=str(display_name or "").strip(),
            connection_slug=str(connection_slug or "").strip(),
            server_url=str(server_url or "").strip(),
            auth_type=_normalize_auth_type(auth_type),
            encrypted_secret=str(encrypted_secret or ""),
            allowed_agents=_string_list(allowed_agents or ["general"]),
            visibility=_normalize_visibility(visibility),
            metadata_json=dict(metadata_json or {}),
        )
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO mcp_connections
                        (connection_id, tenant_id, owner_user_id, display_name, connection_slug, server_url,
                         auth_type, encrypted_secret, status, allowed_agents, visibility, health, metadata_json,
                         created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'active', %s, %s, '{}'::jsonb, %s, now(), now())
                    RETURNING *
                    """,
                    (
                        record.connection_id,
                        record.tenant_id,
                        record.owner_user_id,
                        record.display_name,
                        record.connection_slug,
                        record.server_url,
                        record.auth_type,
                        record.encrypted_secret,
                        psycopg2.extras.Json(record.allowed_agents),
                        record.visibility,
                        psycopg2.extras.Json(record.metadata_json),
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return _row_to_connection(dict(row or {}))

    def update_connection(
        self,
        connection_id: str,
        *,
        tenant_id: str,
        owner_user_id: str | None = None,
        display_name: str | None = None,
        server_url: str | None = None,
        auth_type: str | None = None,
        encrypted_secret: str | None = None,
        allowed_agents: Iterable[str] | None = None,
        visibility: str | None = None,
        status: str | None = None,
        metadata_json: Optional[Dict[str, Any]] = None,
    ) -> McpConnectionRecord | None:
        current = self.get_connection(connection_id, tenant_id=tenant_id, owner_user_id=owner_user_id)
        if current is None:
            return None
        updates = {
            "display_name": current.display_name if display_name is None else str(display_name or "").strip(),
            "server_url": current.server_url if server_url is None else str(server_url or "").strip(),
            "auth_type": current.auth_type if auth_type is None else _normalize_auth_type(auth_type),
            "encrypted_secret": current.encrypted_secret if encrypted_secret is None else str(encrypted_secret or ""),
            "allowed_agents": current.allowed_agents if allowed_agents is None else _string_list(allowed_agents),
            "visibility": current.visibility if visibility is None else _normalize_visibility(visibility),
            "status": current.status if status is None else _normalize_connection_status(status),
            "metadata_json": current.metadata_json if metadata_json is None else dict(metadata_json),
        }
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    UPDATE mcp_connections
                    SET display_name=%s, server_url=%s, auth_type=%s, encrypted_secret=%s,
                        allowed_agents=%s, visibility=%s, status=%s, metadata_json=%s, updated_at=now()
                    WHERE connection_id=%s AND tenant_id=%s
                    RETURNING *
                    """,
                    (
                        updates["display_name"],
                        updates["server_url"],
                        updates["auth_type"],
                        updates["encrypted_secret"],
                        psycopg2.extras.Json(updates["allowed_agents"]),
                        updates["visibility"],
                        updates["status"],
                        psycopg2.extras.Json(updates["metadata_json"]),
                        connection_id,
                        tenant_id,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return _row_to_connection(dict(row or {})) if row else None

    def get_connection(
        self,
        connection_id: str,
        *,
        tenant_id: str,
        owner_user_id: str | None = None,
    ) -> McpConnectionRecord | None:
        sql = "SELECT * FROM mcp_connections WHERE connection_id=%s AND tenant_id=%s"
        params: list[Any] = [connection_id, tenant_id]
        if owner_user_id is not None:
            sql += " AND owner_user_id=%s"
            params.append(owner_user_id)
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
        return _row_to_connection(dict(row or {})) if row else None

    def list_connections(
        self,
        *,
        tenant_id: str,
        owner_user_id: str | None = None,
        include_disabled: bool = True,
        include_tenant_visible: bool = True,
    ) -> List[McpConnectionRecord]:
        filters = ["tenant_id=%s"]
        params: list[Any] = [tenant_id]
        if owner_user_id is not None:
            if include_tenant_visible:
                filters.append("(owner_user_id=%s OR visibility='tenant')")
            else:
                filters.append("owner_user_id=%s")
            params.append(owner_user_id)
        if not include_disabled:
            filters.append("status='active'")
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT * FROM mcp_connections
                    WHERE {' AND '.join(filters)}
                    ORDER BY updated_at DESC, display_name, connection_id
                    """,
                    params,
                )
                rows = cur.fetchall()
        return [_row_to_connection(dict(row)) for row in rows]

    def delete_connection(self, connection_id: str, *, tenant_id: str, owner_user_id: str | None = None) -> bool:
        sql = "DELETE FROM mcp_connections WHERE connection_id=%s AND tenant_id=%s"
        params: list[Any] = [connection_id, tenant_id]
        if owner_user_id is not None:
            sql += " AND owner_user_id=%s"
            params.append(owner_user_id)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted

    def update_health(
        self,
        connection_id: str,
        *,
        tenant_id: str,
        health: Dict[str, Any],
        tested: bool = False,
        refreshed: bool = False,
        status: str | None = None,
    ) -> McpConnectionRecord | None:
        assignments = ["health=%s", "updated_at=now()"]
        params: list[Any] = [psycopg2.extras.Json(dict(health or {}))]
        if tested:
            assignments.append("last_tested_at=now()")
        if refreshed:
            assignments.append("last_refreshed_at=now()")
        if status is not None:
            assignments.append("status=%s")
            params.append(_normalize_connection_status(status))
        params.extend([connection_id, tenant_id])
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    UPDATE mcp_connections
                    SET {', '.join(assignments)}
                    WHERE connection_id=%s AND tenant_id=%s
                    RETURNING *
                    """,
                    params,
                )
                row = cur.fetchone()
            conn.commit()
        return _row_to_connection(dict(row or {})) if row else None

    def replace_tool_catalog(
        self,
        connection: McpConnectionRecord,
        tools: Iterable[McpToolCatalogRecord],
    ) -> List[McpToolCatalogRecord]:
        records = list(tools)
        now = _utc_now()
        seen_raw = {record.raw_tool_name for record in records}
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                for record in records:
                    cur.execute(
                        """
                        INSERT INTO mcp_tool_catalog
                            (tool_id, connection_id, tenant_id, owner_user_id, raw_tool_name, registry_name, tool_slug,
                             description, input_schema, read_only, destructive, background_safe, should_defer,
                             search_hint, defer_priority, enabled, status, checksum, metadata_json, first_seen_at, last_seen_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now(), now())
                        ON CONFLICT (connection_id, raw_tool_name) DO UPDATE SET
                            registry_name=EXCLUDED.registry_name,
                            tool_slug=EXCLUDED.tool_slug,
                            description=EXCLUDED.description,
                            input_schema=EXCLUDED.input_schema,
                            checksum=EXCLUDED.checksum,
                            metadata_json=EXCLUDED.metadata_json,
                            status='active',
                            last_seen_at=EXCLUDED.last_seen_at
                        """,
                        (
                            record.tool_id,
                            connection.connection_id,
                            connection.tenant_id,
                            connection.owner_user_id,
                            record.raw_tool_name,
                            record.registry_name,
                            record.tool_slug,
                            record.description,
                            psycopg2.extras.Json(record.input_schema),
                            record.read_only,
                            record.destructive,
                            record.background_safe,
                            record.should_defer,
                            record.search_hint,
                            record.defer_priority,
                            record.enabled,
                            _normalize_tool_status(record.status),
                            record.checksum,
                            psycopg2.extras.Json(record.metadata_json),
                        ),
                    )
                if seen_raw:
                    cur.execute(
                        """
                        UPDATE mcp_tool_catalog
                        SET status='missing', last_seen_at=%s
                        WHERE connection_id=%s AND NOT (raw_tool_name = ANY(%s))
                        """,
                        (now, connection.connection_id, list(seen_raw)),
                    )
                else:
                    cur.execute(
                        "UPDATE mcp_tool_catalog SET status='missing', last_seen_at=%s WHERE connection_id=%s",
                        (now, connection.connection_id),
                    )
            conn.commit()
        return self.list_tool_catalog(
            tenant_id=connection.tenant_id,
            owner_user_id=connection.owner_user_id,
            connection_id=connection.connection_id,
            include_disabled=True,
        )

    def list_tool_catalog(
        self,
        *,
        tenant_id: str,
        owner_user_id: str | None = None,
        connection_id: str = "",
        include_disabled: bool = False,
        include_tenant_visible: bool = True,
    ) -> List[McpToolCatalogRecord]:
        filters = ["t.tenant_id=%s"]
        params: list[Any] = [tenant_id]
        if owner_user_id is not None:
            if include_tenant_visible:
                filters.append("(t.owner_user_id=%s OR c.visibility='tenant')")
            else:
                filters.append("t.owner_user_id=%s")
            params.append(owner_user_id)
        if connection_id:
            filters.append("t.connection_id=%s")
            params.append(connection_id)
        if not include_disabled:
            filters.append("t.enabled=TRUE AND t.status='active' AND c.status='active'")
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT t.*
                    FROM mcp_tool_catalog t
                    JOIN mcp_connections c ON c.connection_id = t.connection_id
                    WHERE {' AND '.join(filters)}
                    ORDER BY t.registry_name
                    """,
                    params,
                )
                rows = cur.fetchall()
        return [_row_to_tool(dict(row)) for row in rows]

    def get_tool_by_registry_name(
        self,
        registry_name: str,
        *,
        tenant_id: str,
        owner_user_id: str | None = None,
    ) -> McpToolCatalogRecord | None:
        filters = ["t.tenant_id=%s", "t.registry_name=%s"]
        params: list[Any] = [tenant_id, registry_name]
        if owner_user_id is not None:
            filters.append("(t.owner_user_id=%s OR c.visibility='tenant')")
            params.append(owner_user_id)
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT t.*
                    FROM mcp_tool_catalog t
                    JOIN mcp_connections c ON c.connection_id = t.connection_id
                    WHERE {' AND '.join(filters)}
                    LIMIT 1
                    """,
                    params,
                )
                row = cur.fetchone()
        return _row_to_tool(dict(row or {})) if row else None

    def update_tool_catalog(
        self,
        tool_id: str,
        *,
        tenant_id: str,
        owner_user_id: str | None = None,
        enabled: bool | None = None,
        read_only: bool | None = None,
        destructive: bool | None = None,
        background_safe: bool | None = None,
        should_defer: bool | None = None,
        search_hint: str | None = None,
        defer_priority: int | None = None,
        status: str | None = None,
    ) -> McpToolCatalogRecord | None:
        current = self._get_tool(tool_id, tenant_id=tenant_id, owner_user_id=owner_user_id)
        if current is None:
            return None
        values = {
            "enabled": current.enabled if enabled is None else bool(enabled),
            "read_only": current.read_only if read_only is None else bool(read_only),
            "destructive": current.destructive if destructive is None else bool(destructive),
            "background_safe": current.background_safe if background_safe is None else bool(background_safe),
            "should_defer": current.should_defer if should_defer is None else bool(should_defer),
            "search_hint": current.search_hint if search_hint is None else str(search_hint or "").strip(),
            "defer_priority": current.defer_priority if defer_priority is None else int(defer_priority or 50),
            "status": current.status if status is None else _normalize_tool_status(status),
        }
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    UPDATE mcp_tool_catalog
                    SET enabled=%s, read_only=%s, destructive=%s, background_safe=%s,
                        should_defer=%s, search_hint=%s, defer_priority=%s, status=%s
                    WHERE tool_id=%s AND tenant_id=%s
                    RETURNING *
                    """,
                    (
                        values["enabled"],
                        values["read_only"],
                        values["destructive"],
                        values["background_safe"],
                        values["should_defer"],
                        values["search_hint"],
                        values["defer_priority"],
                        values["status"],
                        tool_id,
                        tenant_id,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return _row_to_tool(dict(row or {})) if row else None

    def _get_tool(self, tool_id: str, *, tenant_id: str, owner_user_id: str | None = None) -> McpToolCatalogRecord | None:
        sql = "SELECT * FROM mcp_tool_catalog WHERE tool_id=%s AND tenant_id=%s"
        params: list[Any] = [tool_id, tenant_id]
        if owner_user_id is not None:
            sql += " AND owner_user_id=%s"
            params.append(owner_user_id)
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
        return _row_to_tool(dict(row or {})) if row else None
