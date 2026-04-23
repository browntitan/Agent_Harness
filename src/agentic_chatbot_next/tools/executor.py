"""Tool assembly helpers for the next runtime."""
from __future__ import annotations

from typing import Any, List

from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.tools import ToolDefinition
from agentic_chatbot_next.tools.discovery import (
    CALL_DEFERRED_TOOL_NAME,
    DISCOVER_TOOLS_NAME,
    ToolDiscoveryService,
    deferred_tool_discovery_enabled,
    should_defer_for_agent,
)
from agentic_chatbot_next.tools.policy import ToolPolicyService, tool_allowed_by_selectors
from agentic_chatbot_next.tools.registry import build_tool_definitions


def _attach_definition_metadata(candidate: Any, definition: ToolDefinition) -> Any:
    if hasattr(candidate, "description"):
        candidate.description = definition.render_tool_card()
    candidate.metadata = {
        **dict(getattr(candidate, "metadata", {}) or {}),
        "concurrency_key": definition.concurrency_key,
        "group": definition.group,
        "read_only": definition.read_only,
        "destructive": definition.destructive,
        "requires_workspace": definition.requires_workspace,
        "should_defer": definition.should_defer,
        "deferred_facade": definition.name in {DISCOVER_TOOLS_NAME, CALL_DEFERRED_TOOL_NAME},
    }
    return candidate


def _build_named_tool(definition: ToolDefinition, tool_context: Any, tool_name: str) -> Any | None:
    for candidate in definition.builder(tool_context):
        if getattr(candidate, "name", "") == tool_name:
            return _attach_definition_metadata(candidate, definition)
    return None


def _emit_catalog_built(tool_context: Any, summary: dict[str, Any]) -> None:
    kernel = getattr(tool_context, "kernel", None)
    session = getattr(tool_context, "session", None)
    if kernel is not None and session is not None and hasattr(kernel, "_emit"):
        kernel._emit(
            "deferred_tool_catalog_built",
            getattr(session, "session_id", ""),
            agent_name=str(getattr(tool_context, "active_agent", "") or ""),
            payload=summary,
        )


def build_agent_tools(
    agent: AgentDefinition,
    tool_context: Any,
    *,
    policy_service: ToolPolicyService | None = None,
) -> List[Any]:
    policy = policy_service or ToolPolicyService()
    definitions = build_tool_definitions(tool_context)
    deferred_enabled = deferred_tool_discovery_enabled(getattr(tool_context, "settings", None))
    tools: List[Any] = []
    bound_tool_names: set[str] = set()
    for tool_name in agent.allowed_tools:
        if str(tool_name).strip().endswith("*"):
            continue
        definition = definitions.get(tool_name)
        if definition is None:
            continue
        if deferred_enabled and should_defer_for_agent(definition, agent):
            continue
        if not policy.is_allowed(agent, definition, tool_context):
            continue
        candidate = _build_named_tool(definition, tool_context, tool_name)
        if candidate is not None:
            tools.append(candidate)
            bound_tool_names.add(tool_name)
    for definition in definitions.values():
        if definition.name in bound_tool_names:
            continue
        if not tool_allowed_by_selectors(list(agent.allowed_tools or []), definition.name):
            continue
        if deferred_enabled and should_defer_for_agent(definition, agent):
            continue
        if not policy.is_allowed(agent, definition, tool_context):
            continue
        candidate = _build_named_tool(definition, tool_context, definition.name)
        if candidate is not None:
            tools.append(candidate)
            bound_tool_names.add(definition.name)
    if deferred_enabled:
        discovery = ToolDiscoveryService(
            agent=agent,
            tool_context=tool_context,
            definitions=definitions,
            policy_service=policy,
        )
        summary = discovery.summary()
        metadata = getattr(tool_context, "metadata", None)
        if isinstance(metadata, dict):
            metadata["deferred_tool_discovery"] = summary
        if discovery.has_deferred_targets():
            for facade_name in (DISCOVER_TOOLS_NAME, CALL_DEFERRED_TOOL_NAME):
                if any(getattr(tool, "name", "") == facade_name for tool in tools):
                    continue
                definition = definitions.get(facade_name)
                if definition is None:
                    continue
                candidate = _build_named_tool(definition, tool_context, facade_name)
                if candidate is not None:
                    tools.append(candidate)
        _emit_catalog_built(tool_context, summary)
    return tools
