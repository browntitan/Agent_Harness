from __future__ import annotations

import json
from typing import Any, List

from langchain_core.tools import tool

from agentic_chatbot_next.tools.discovery import ToolDiscoveryService


def build_discovery_tools(ctx: Any) -> List[Any]:
    @tool
    def discover_tools(query: str, group: str = "", top_k: int = 0) -> str:
        """Search deferred tools available to the current agent."""

        service = ToolDiscoveryService.from_context(ctx)
        return json.dumps(
            service.search(query, group=group, top_k=top_k),
            ensure_ascii=False,
        )

    @tool
    def call_deferred_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> str:
        """Invoke a deferred tool that was discovered in the current turn."""

        service = ToolDiscoveryService.from_context(ctx)
        return json.dumps(
            service.invoke(tool_name, arguments=arguments),
            ensure_ascii=False,
        )

    return [discover_tools, call_deferred_tool]
