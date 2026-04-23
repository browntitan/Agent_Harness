from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any, Dict, List

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from agentic_chatbot_next.persistence.postgres.mcp import McpConnectionRecord


class McpClientError(RuntimeError):
    """Raised when an MCP server cannot be reached or returns an invalid response."""


def _json_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _run_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro)).result()


class McpStreamableHttpClient:
    """Small synchronous wrapper around the official MCP Streamable HTTP client."""

    async def _list_tools_async(
        self,
        connection: McpConnectionRecord,
        *,
        secret: str = "",
        timeout_seconds: int = 15,
    ) -> List[Dict[str, Any]]:
        headers = {"Authorization": f"Bearer {secret}"} if secret else None
        timeout = timedelta(seconds=max(1, int(timeout_seconds or 15)))
        try:
            async with streamablehttp_client(
                connection.server_url,
                headers=headers,
                timeout=timeout,
                sse_read_timeout=timeout,
            ) as (read_stream, write_stream, _get_session_id):
                async with ClientSession(read_stream, write_stream, read_timeout_seconds=timeout) as session:
                    await session.initialize()
                    result = await session.list_tools()
        except Exception as exc:
            raise McpClientError(str(exc)) from exc
        payload = _json_model(result)
        tools = payload.get("tools") if isinstance(payload, dict) else getattr(result, "tools", [])
        return [_json_model(tool) for tool in list(tools or []) if tool is not None]

    async def _call_tool_async(
        self,
        connection: McpConnectionRecord,
        *,
        raw_tool_name: str,
        arguments: Dict[str, Any],
        secret: str = "",
        timeout_seconds: int = 60,
    ) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {secret}"} if secret else None
        timeout = timedelta(seconds=max(1, int(timeout_seconds or 60)))
        try:
            async with streamablehttp_client(
                connection.server_url,
                headers=headers,
                timeout=timeout,
                sse_read_timeout=timeout,
            ) as (read_stream, write_stream, _get_session_id):
                async with ClientSession(read_stream, write_stream, read_timeout_seconds=timeout) as session:
                    await session.initialize()
                    result = await session.call_tool(str(raw_tool_name), arguments=dict(arguments or {}))
        except Exception as exc:
            raise McpClientError(str(exc)) from exc
        payload = _json_model(result)
        return payload if isinstance(payload, dict) else {"result": payload}

    def list_tools(
        self,
        connection: McpConnectionRecord,
        *,
        secret: str = "",
        timeout_seconds: int = 15,
    ) -> List[Dict[str, Any]]:
        return list(_run_sync(self._list_tools_async(connection, secret=secret, timeout_seconds=timeout_seconds)))

    def call_tool(
        self,
        connection: McpConnectionRecord,
        *,
        raw_tool_name: str,
        arguments: Dict[str, Any],
        secret: str = "",
        timeout_seconds: int = 60,
    ) -> Dict[str, Any]:
        return dict(
            _run_sync(
                self._call_tool_async(
                    connection,
                    raw_tool_name=raw_tool_name,
                    arguments=arguments,
                    secret=secret,
                    timeout_seconds=timeout_seconds,
                )
            )
        )
