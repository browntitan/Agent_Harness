from __future__ import annotations

from typing import Any, Dict

from agentic_chatbot_next.contracts.tools import ToolDefinition
from agentic_chatbot_next.tools.groups.mcp import build_mcp_tools


def _session_identity(ctx: Any) -> tuple[str, str]:
    session = getattr(ctx, "session", None)
    return (
        str(getattr(session, "tenant_id", "") or "local-dev"),
        str(getattr(session, "user_id", "") or "local-cli"),
    )


def build_mcp_tool_definitions(ctx: Any) -> Dict[str, ToolDefinition]:
    if ctx is None:
        return {}
    settings = getattr(ctx, "settings", None)
    if not bool(getattr(settings, "mcp_tool_plane_enabled", False)):
        return {}
    store = getattr(getattr(ctx, "stores", None), "mcp_connection_store", None)
    if store is None:
        return {}
    tenant_id, owner_user_id = _session_identity(ctx)
    connections = {
        record.connection_id: record
        for record in store.list_connections(
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            include_disabled=False,
        )
    }
    definitions: Dict[str, ToolDefinition] = {}
    for record in store.list_tool_catalog(
        tenant_id=tenant_id,
        owner_user_id=owner_user_id,
        include_disabled=False,
    ):
        connection = connections.get(record.connection_id)
        if connection is None:
            continue
        description = record.description or f"MCP tool {record.raw_tool_name}"
        search_hint = record.search_hint or " ".join(
            item
            for item in [
                connection.display_name,
                record.raw_tool_name,
                description,
            ]
            if item
        )
        definitions[record.registry_name] = ToolDefinition(
            name=record.registry_name,
            group="mcp",
            builder=build_mcp_tools,
            description=description,
            args_schema=dict(record.input_schema or {"type": "object", "properties": {}, "additionalProperties": True}),
            when_to_use=f"Use when the connected MCP server '{connection.display_name}' exposes this capability.",
            avoid_when="Avoid unless the user asked for a capability that is unavailable through built-in tools.",
            output_description="Returns a JSON MCP tool-result envelope from the connected remote MCP server.",
            examples=[f"{record.registry_name}(...)"],
            keywords=[
                "mcp",
                "plugin",
                connection.display_name,
                record.raw_tool_name,
                record.tool_slug,
            ],
            read_only=bool(record.read_only),
            destructive=bool(record.destructive),
            background_safe=bool(record.background_safe),
            should_defer=bool(record.should_defer),
            search_hint=search_hint,
            defer_reason="External MCP/plugin tools are loaded on demand to protect context budget and policy boundaries.",
            defer_priority=int(record.defer_priority or 50),
            metadata={
                "source": "mcp",
                "connection_id": connection.connection_id,
                "connection_slug": connection.connection_slug,
                "connection_display_name": connection.display_name,
                "raw_tool_name": record.raw_tool_name,
                "tool_id": record.tool_id,
                "tenant_id": record.tenant_id,
                "owner_user_id": record.owner_user_id,
                "visibility": connection.visibility,
                "allowed_agents": list(connection.allowed_agents),
                "server_url": connection.server_url,
            },
        )
    return definitions
