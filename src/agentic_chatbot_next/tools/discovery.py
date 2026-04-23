from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List

from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.tools import ToolDefinition
from agentic_chatbot_next.tools.policy import ToolPolicyService, tool_allowed_by_selectors

DISCOVER_TOOLS_NAME = "discover_tools"
CALL_DEFERRED_TOOL_NAME = "call_deferred_tool"
DISCOVERY_TOOL_NAMES = {DISCOVER_TOOLS_NAME, CALL_DEFERRED_TOOL_NAME}
DISCOVERED_TARGETS_METADATA_KEY = "deferred_tool_discovered_targets"


def deferred_tool_discovery_enabled(settings: Any) -> bool:
    return bool(getattr(settings, "deferred_tool_discovery_enabled", False))


def should_defer_for_agent(definition: ToolDefinition, agent: AgentDefinition) -> bool:
    if not bool(getattr(definition, "should_defer", False)):
        return False
    eager_for = {str(item).strip() for item in (getattr(definition, "eager_for_agents", []) or []) if str(item).strip()}
    return agent.name not in eager_for


def _emit_event(ctx: Any, event_type: str, payload: dict[str, Any]) -> None:
    kernel = getattr(ctx, "kernel", None)
    session = getattr(ctx, "session", None)
    if kernel is not None and session is not None and hasattr(kernel, "_emit"):
        kernel._emit(
            event_type,
            getattr(session, "session_id", ""),
            agent_name=str(getattr(ctx, "active_agent", "") or ""),
            payload=payload,
        )


def _schema_summary(schema: dict[str, Any]) -> dict[str, Any]:
    properties = dict(schema.get("properties") or {}) if isinstance(schema, dict) else {}
    return {
        "required": [str(item) for item in (schema.get("required") or [])] if isinstance(schema, dict) else [],
        "properties": [
            {
                "name": str(name),
                "type": str(details.get("type") or "object") if isinstance(details, dict) else "object",
                "description": str(details.get("description") or "") if isinstance(details, dict) else "",
            }
            for name, details in properties.items()
        ],
    }


def _tool_card(definition: ToolDefinition) -> dict[str, Any]:
    return {
        "name": definition.name,
        "group": definition.group,
        "description": definition.description,
        "when_to_use": definition.when_to_use,
        "avoid_when": definition.avoid_when,
        "output_description": definition.output_description,
        "schema": _schema_summary(dict(definition.args_schema or {})),
        "read_only": bool(definition.read_only),
        "destructive": bool(definition.destructive),
        "background_safe": bool(definition.background_safe),
        "requires_workspace": bool(definition.requires_workspace),
        "search_hint": str(definition.search_hint or ""),
        "defer_reason": str(definition.defer_reason or ""),
    }


def _tokenize(value: Any) -> set[str]:
    return {part.lower() for part in re.findall(r"[A-Za-z0-9_]+", str(value or "")) if part.strip()}


def _schema_text(schema: dict[str, Any]) -> str:
    properties = dict(schema.get("properties") or {}) if isinstance(schema, dict) else {}
    parts: list[str] = []
    for name, details in properties.items():
        parts.append(str(name))
        if isinstance(details, dict):
            parts.append(str(details.get("description") or ""))
            enum = details.get("enum")
            if isinstance(enum, list):
                parts.extend(str(item) for item in enum)
    return " ".join(parts)


def _score_definition(definition: ToolDefinition, query: str) -> int:
    query_terms = _tokenize(query)
    if not query_terms:
        return 0
    name_terms = _tokenize(definition.name)
    group_terms = _tokenize(definition.group)
    keyword_terms = _tokenize(" ".join(definition.keywords or []))
    hint_terms = _tokenize(definition.search_hint)
    body_terms = _tokenize(
        " ".join(
            [
                definition.description,
                definition.when_to_use,
                definition.avoid_when,
                definition.output_description,
                _schema_text(dict(definition.args_schema or {})),
            ]
        )
    )
    score = 0
    score += 12 * len(query_terms & name_terms)
    score += 8 * len(query_terms & keyword_terms)
    score += 6 * len(query_terms & hint_terms)
    score += 4 * len(query_terms & group_terms)
    score += 2 * len(query_terms & body_terms)
    if definition.name.lower() in str(query or "").lower():
        score += 25
    if definition.group.lower() in str(query or "").lower():
        score += 8
    return score


def _stringify_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except Exception:
        return str(result)


class ToolDiscoveryService:
    def __init__(
        self,
        *,
        agent: AgentDefinition,
        tool_context: Any,
        definitions: Dict[str, ToolDefinition],
        policy_service: ToolPolicyService | None = None,
    ) -> None:
        self.agent = agent
        self.tool_context = tool_context
        self.definitions = definitions
        self.policy = policy_service or ToolPolicyService()

    @classmethod
    def from_context(cls, tool_context: Any, *, policy_service: ToolPolicyService | None = None) -> "ToolDiscoveryService":
        from agentic_chatbot_next.tools.registry import build_tool_definitions

        agent = getattr(tool_context, "active_definition", None)
        if agent is None:
            raise ValueError("Deferred tool discovery requires an active agent definition.")
        return cls(
            agent=agent,
            tool_context=tool_context,
            definitions=build_tool_definitions(tool_context),
            policy_service=policy_service,
        )

    def deferred_targets(self) -> list[ToolDefinition]:
        targets: list[ToolDefinition] = []
        for definition in self.definitions.values():
            if definition.name in DISCOVERY_TOOL_NAMES:
                continue
            if not tool_allowed_by_selectors(list(self.agent.allowed_tools or []), definition.name):
                continue
            if not should_defer_for_agent(definition, self.agent):
                continue
            if not self.policy.is_allowed(self.agent, definition, self.tool_context):
                continue
            targets.append(definition)
        return targets

    def has_deferred_targets(self) -> bool:
        return bool(self.deferred_targets())

    def summary(self) -> dict[str, Any]:
        targets = self.deferred_targets()
        groups = sorted({definition.group for definition in targets})
        return {
            "enabled": deferred_tool_discovery_enabled(getattr(self.tool_context, "settings", None)),
            "count": len(targets),
            "groups": groups,
            "tools": [definition.name for definition in sorted(targets, key=lambda item: (item.group, item.name))[:8]],
        }

    def search(self, query: str, *, group: str = "", top_k: int = 0) -> dict[str, Any]:
        default_top_k = max(1, int(getattr(getattr(self.tool_context, "settings", None), "deferred_tool_discovery_top_k", 8) or 8))
        limit = max(1, min(int(top_k or default_top_k), 20))
        group_filter = str(group or "").strip()
        scored: list[tuple[int, int, str, ToolDefinition]] = []
        for definition in self.deferred_targets():
            if group_filter and definition.group != group_filter:
                continue
            score = _score_definition(definition, query)
            if score <= 0 and str(query or "").strip():
                continue
            scored.append((score, int(definition.defer_priority or 50), definition.name, definition))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected = [definition for _, _, _, definition in scored[:limit]]
        discovered = set(self._discovered_targets())
        discovered.update(definition.name for definition in selected)
        self._set_discovered_targets(discovered)
        payload = {
            "object": "deferred_tool_discovery",
            "query": query,
            "group": group_filter,
            "top_k": limit,
            "matches": [_tool_card(definition) for definition in selected],
            "available_count": len(self.deferred_targets()),
        }
        _emit_event(
            self.tool_context,
            "deferred_tool_discovery_searched",
            {
                "query": query,
                "group": group_filter,
                "top_k": limit,
                "match_names": [definition.name for definition in selected],
                "available_count": payload["available_count"],
            },
        )
        return payload

    def invoke(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        target_name = str(tool_name or "").strip()
        args = dict(arguments or {})
        denial = self._validate_invocation(target_name)
        if denial:
            _emit_event(self.tool_context, "deferred_tool_denied", {"tool_name": target_name, "reason": denial})
            return {
                "object": "deferred_tool_result",
                "tool_name": target_name,
                "status": "error",
                "error": denial,
            }
        definition = self.definitions[target_name]
        try:
            tool = self._build_target_tool(definition)
            if hasattr(tool, "invoke"):
                try:
                    result = tool.invoke(args, config={"callbacks": getattr(self.tool_context, "callbacks", [])})
                except TypeError:
                    result = tool.invoke(args)
            else:
                result = tool(**args)
            payload = {
                "object": "deferred_tool_result",
                "tool_name": target_name,
                "status": "ok",
                "result": _stringify_tool_result(result),
                "read_only": bool(definition.read_only),
                "destructive": bool(definition.destructive),
                "requires_workspace": bool(definition.requires_workspace),
            }
            _emit_event(self.tool_context, "deferred_tool_invoked", {"tool_name": target_name, "status": "ok"})
            return payload
        except Exception as exc:
            _emit_event(
                self.tool_context,
                "deferred_tool_denied",
                {"tool_name": target_name, "reason": type(exc).__name__},
            )
            return {
                "object": "deferred_tool_result",
                "tool_name": target_name,
                "status": "error",
                "error": str(exc),
            }

    def _validate_invocation(self, target_name: str) -> str:
        if not target_name:
            return "tool_name is required"
        if target_name in DISCOVERY_TOOL_NAMES:
            return "deferred discovery tools cannot call themselves"
        definition = self.definitions.get(target_name)
        if definition is None:
            return "unknown deferred tool"
        if not tool_allowed_by_selectors(list(self.agent.allowed_tools or []), target_name):
            return "tool is not allowed for the active agent"
        if not should_defer_for_agent(definition, self.agent):
            return "tool is not deferred for the active agent"
        if not self.policy.is_allowed(self.agent, definition, self.tool_context):
            return "tool policy denied the deferred target"
        require_search = bool(getattr(getattr(self.tool_context, "settings", None), "deferred_tool_discovery_require_search", True))
        if require_search and target_name not in self._discovered_targets():
            return "tool must be returned by discover_tools before call_deferred_tool can invoke it"
        return ""

    def _build_target_tool(self, definition: ToolDefinition) -> Any:
        for candidate in definition.builder(self.tool_context):
            if getattr(candidate, "name", "") == definition.name:
                if hasattr(candidate, "description"):
                    candidate.description = definition.render_tool_card()
                candidate.metadata = {
                    **dict(getattr(candidate, "metadata", {}) or {}),
                    "concurrency_key": definition.concurrency_key,
                    "group": definition.group,
                    "read_only": definition.read_only,
                    "destructive": definition.destructive,
                    "requires_workspace": definition.requires_workspace,
                    "deferred_target": True,
                }
                return candidate
        raise ValueError(f"Deferred target builder did not return {definition.name!r}.")

    def _discovered_targets(self) -> set[str]:
        metadata = getattr(self.tool_context, "metadata", None)
        if not isinstance(metadata, dict):
            return set()
        return {str(item) for item in (metadata.get(DISCOVERED_TARGETS_METADATA_KEY) or []) if str(item)}

    def _set_discovered_targets(self, targets: Iterable[str]) -> None:
        metadata = getattr(self.tool_context, "metadata", None)
        if not isinstance(metadata, dict):
            return
        metadata[DISCOVERED_TARGETS_METADATA_KEY] = sorted({str(item) for item in targets if str(item)})
