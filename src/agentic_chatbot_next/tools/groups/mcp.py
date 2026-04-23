from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.tools import StructuredTool

from agentic_chatbot_next.mcp.service import McpCatalogService
from agentic_chatbot_next.persistence.postgres.mcp import McpToolCatalogRecord


def _mcp_store(ctx: Any) -> Any | None:
    return getattr(getattr(ctx, "stores", None), "mcp_connection_store", None)


def _tool_result_to_text(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _build_one_tool(ctx: Any, record: McpToolCatalogRecord) -> Any:
    def _call(**kwargs: Any) -> str:
        store = _mcp_store(ctx)
        if store is None:
            return _tool_result_to_text(
                {
                    "object": "mcp_tool_result",
                    "registry_name": record.registry_name,
                    "status": "error",
                    "error": "MCP connection store is not configured.",
                }
            )
        service = McpCatalogService(ctx.settings, store)
        return _tool_result_to_text(
            service.call_tool(
                ctx,
                registry_name=record.registry_name,
                arguments=dict(kwargs or {}),
            )
        )

    _call.__name__ = record.registry_name
    return StructuredTool.from_function(
        func=_call,
        name=record.registry_name,
        description=record.description or f"MCP tool {record.raw_tool_name}",
        args_schema=dict(record.input_schema or {"type": "object", "properties": {}, "additionalProperties": True}),
        infer_schema=False,
    )


def build_mcp_tools(ctx: Any) -> List[Any]:
    settings = getattr(ctx, "settings", None)
    if not bool(getattr(settings, "mcp_tool_plane_enabled", False)):
        return []
    store = _mcp_store(ctx)
    session = getattr(ctx, "session", None)
    if store is None or session is None:
        return []
    records = store.list_tool_catalog(
        tenant_id=str(getattr(session, "tenant_id", "") or "local-dev"),
        owner_user_id=str(getattr(session, "user_id", "") or "local-cli"),
        include_disabled=False,
    )
    return [_build_one_tool(ctx, record) for record in records]
