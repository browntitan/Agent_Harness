from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional

from agentic_chatbot_next.mcp.client import McpClientError, McpStreamableHttpClient
from agentic_chatbot_next.mcp.security import (
    decrypt_mcp_secret,
    encrypt_mcp_secret,
    normalize_mcp_registry_name,
    slugify_mcp_name,
    validate_mcp_server_url,
)
from agentic_chatbot_next.persistence.postgres.mcp import (
    McpConnectionRecord,
    McpConnectionStore,
    McpToolCatalogRecord,
)


def _emit(ctx: Any, event_type: str, payload: Dict[str, Any]) -> None:
    kernel = getattr(ctx, "kernel", None)
    session = getattr(ctx, "session", None)
    if kernel is not None and session is not None and hasattr(kernel, "_emit"):
        kernel._emit(
            event_type,
            getattr(session, "session_id", ""),
            agent_name=str(getattr(ctx, "active_agent", "") or ""),
            payload=payload,
        )


def _json_checksum(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _sanitize_schema(schema: Any) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}, "additionalProperties": True}
    clean = dict(schema)
    if clean.get("type") != "object":
        clean["type"] = "object"
    properties = clean.get("properties")
    if not isinstance(properties, dict):
        clean["properties"] = {}
    required = clean.get("required")
    if required is not None and not isinstance(required, list):
        clean["required"] = []
    clean.setdefault("additionalProperties", True)
    return clean


def _metadata_annotations(raw_tool: Dict[str, Any]) -> Dict[str, Any]:
    annotations = raw_tool.get("annotations")
    if hasattr(annotations, "model_dump"):
        annotations = annotations.model_dump(mode="json")
    if not isinstance(annotations, dict):
        annotations = {}
    return dict(annotations)


def _tool_record_from_mcp(connection: McpConnectionRecord, raw_tool: Dict[str, Any]) -> McpToolCatalogRecord:
    raw_name = str(raw_tool.get("name") or "").strip()
    description = str(raw_tool.get("description") or "").strip()
    annotations = _metadata_annotations(raw_tool)
    input_schema = _sanitize_schema(raw_tool.get("inputSchema") or raw_tool.get("input_schema") or {})
    tool_slug = slugify_mcp_name(raw_name, fallback="tool")
    registry_name = normalize_mcp_registry_name(connection.connection_slug, raw_name)
    read_only = bool(annotations.get("readOnlyHint", False))
    destructive = not read_only
    checksum = _json_checksum(
        {
            "raw_tool_name": raw_name,
            "description": description,
            "input_schema": input_schema,
            "annotations": annotations,
        }
    )
    return McpToolCatalogRecord(
        tool_id=f"mcp_tool_{hashlib.sha1(f'{connection.connection_id}:{raw_name}'.encode('utf-8')).hexdigest()[:20]}",
        connection_id=connection.connection_id,
        tenant_id=connection.tenant_id,
        owner_user_id=connection.owner_user_id,
        raw_tool_name=raw_name,
        registry_name=registry_name,
        tool_slug=tool_slug,
        description=description or f"MCP tool {raw_name}",
        input_schema=input_schema,
        read_only=read_only,
        destructive=destructive,
        background_safe=False,
        should_defer=True,
        search_hint=" ".join(part for part in [raw_name, description] if part),
        defer_priority=50,
        enabled=True,
        status="active",
        checksum=checksum,
        metadata_json={"annotations": annotations},
    )


class McpCatalogService:
    def __init__(
        self,
        settings: Any,
        store: McpConnectionStore,
        *,
        client: McpStreamableHttpClient | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.client = client or McpStreamableHttpClient()

    def create_connection(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        display_name: str,
        server_url: str,
        auth_type: str = "none",
        secret: str = "",
        allowed_agents: Iterable[str] | None = None,
        visibility: str = "private",
        metadata_json: Optional[Dict[str, Any]] = None,
    ) -> McpConnectionRecord:
        clean_url = validate_mcp_server_url(self.settings, server_url)
        encrypted_secret = encrypt_mcp_secret(self.settings, secret) if secret else ""
        slug = slugify_mcp_name(display_name or clean_url, fallback="mcp")
        return self.store.create_connection(
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            display_name=display_name or slug,
            connection_slug=slug,
            server_url=clean_url,
            auth_type=auth_type,
            encrypted_secret=encrypted_secret,
            allowed_agents=list(allowed_agents or ["general"]),
            visibility=visibility,
            metadata_json=metadata_json or {},
        )

    def update_connection(
        self,
        connection_id: str,
        *,
        tenant_id: str,
        owner_user_id: str | None = None,
        display_name: str | None = None,
        server_url: str | None = None,
        auth_type: str | None = None,
        secret: str | None = None,
        allowed_agents: Iterable[str] | None = None,
        visibility: str | None = None,
        status: str | None = None,
        metadata_json: Optional[Dict[str, Any]] = None,
    ) -> McpConnectionRecord | None:
        clean_url = validate_mcp_server_url(self.settings, server_url) if server_url is not None else None
        encrypted_secret = encrypt_mcp_secret(self.settings, secret) if secret is not None and secret else None
        if secret == "":
            encrypted_secret = ""
        return self.store.update_connection(
            connection_id,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            display_name=display_name,
            server_url=clean_url,
            auth_type=auth_type,
            encrypted_secret=encrypted_secret,
            allowed_agents=allowed_agents,
            visibility=visibility,
            status=status,
            metadata_json=metadata_json,
        )

    def test_connection(
        self,
        connection: McpConnectionRecord,
        *,
        secret_override: str = "",
    ) -> Dict[str, Any]:
        secret = secret_override or decrypt_mcp_secret(self.settings, connection.encrypted_secret)
        try:
            tools = self.client.list_tools(
                connection,
                secret=secret,
                timeout_seconds=int(getattr(self.settings, "mcp_connection_timeout_seconds", 15) or 15),
            )
        except Exception as exc:
            health = {"ok": False, "error": str(exc), "tool_count": 0}
            self.store.update_health(connection.connection_id, tenant_id=connection.tenant_id, health=health, tested=True, status="error")
            raise McpClientError(str(exc)) from exc
        health = {"ok": True, "tool_count": len(tools)}
        self.store.update_health(connection.connection_id, tenant_id=connection.tenant_id, health=health, tested=True, status="active")
        return health

    def refresh_tools(
        self,
        connection_id: str,
        *,
        tenant_id: str,
        owner_user_id: str | None = None,
    ) -> List[McpToolCatalogRecord]:
        connection = self.store.get_connection(connection_id, tenant_id=tenant_id, owner_user_id=owner_user_id)
        if connection is None:
            raise ValueError("MCP connection not found.")
        secret = decrypt_mcp_secret(self.settings, connection.encrypted_secret)
        try:
            raw_tools = self.client.list_tools(
                connection,
                secret=secret,
                timeout_seconds=int(getattr(self.settings, "mcp_connection_timeout_seconds", 15) or 15),
            )
        except Exception as exc:
            health = {"ok": False, "error": str(exc), "tool_count": 0}
            self.store.update_health(connection.connection_id, tenant_id=tenant_id, health=health, refreshed=True, status="error")
            raise McpClientError(str(exc)) from exc
        records = [_tool_record_from_mcp(connection, raw_tool) for raw_tool in raw_tools if str(raw_tool.get("name") or "").strip()]
        refreshed = self.store.replace_tool_catalog(connection, records)
        health = {"ok": True, "tool_count": len(records)}
        self.store.update_health(connection.connection_id, tenant_id=tenant_id, health=health, refreshed=True, status="active")
        return refreshed

    def call_tool(self, tool_context: Any, *, registry_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        session = getattr(tool_context, "session", None)
        tenant_id = str(getattr(session, "tenant_id", "") or "local-dev")
        owner_user_id = str(getattr(session, "user_id", "") or "local-cli")
        tool_record = self.store.get_tool_by_registry_name(registry_name, tenant_id=tenant_id, owner_user_id=owner_user_id)
        if tool_record is None or not tool_record.enabled or tool_record.status != "active":
            _emit(tool_context, "mcp_tool_denied", {"registry_name": registry_name, "reason": "tool_not_available"})
            return {"object": "mcp_tool_result", "registry_name": registry_name, "status": "error", "error": "MCP tool is not available."}
        connection = self.store.get_connection(
            tool_record.connection_id,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id if tool_record.owner_user_id == owner_user_id else None,
        )
        if connection is None or connection.status != "active":
            _emit(tool_context, "mcp_tool_denied", {"registry_name": registry_name, "reason": "connection_not_available"})
            return {"object": "mcp_tool_result", "registry_name": registry_name, "status": "error", "error": "MCP connection is not available."}
        secret = decrypt_mcp_secret(self.settings, connection.encrypted_secret)
        try:
            result = self.client.call_tool(
                connection,
                raw_tool_name=tool_record.raw_tool_name,
                arguments=dict(arguments or {}),
                secret=secret,
                timeout_seconds=int(getattr(self.settings, "mcp_tool_call_timeout_seconds", 60) or 60),
            )
        except Exception as exc:
            _emit(tool_context, "mcp_connection_error", {"registry_name": registry_name, "error": str(exc)})
            return {"object": "mcp_tool_result", "registry_name": registry_name, "status": "error", "error": str(exc)}
        payload = {
            "object": "mcp_tool_result",
            "registry_name": registry_name,
            "connection_id": connection.connection_id,
            "raw_tool_name": tool_record.raw_tool_name,
            "status": "ok",
            "result": result,
        }
        _emit(tool_context, "mcp_tool_invoked", {"registry_name": registry_name, "connection_id": connection.connection_id})
        return payload
