from __future__ import annotations

import re
from typing import Any, Dict, List


_UNKNOWN_TOOL_RE = re.compile(r"agent\s+'([^']+)'\s+references\s+unknown\s+tool\s+'([^']+)'")
_UNKNOWN_WORKER_RE = re.compile(r"agent\s+'([^']+)'\s+references\s+unknown\s+worker\s+'([^']+)'")


def parse_runtime_registry_error(message: str) -> Dict[str, Any]:
    text = str(message or "").strip()
    missing_tools = [
        {"agent": agent, "tool": tool}
        for agent, tool in _UNKNOWN_TOOL_RE.findall(text)
        if agent and tool
    ]
    missing_workers = [
        {"agent": agent, "worker": worker}
        for agent, worker in _UNKNOWN_WORKER_RE.findall(text)
        if agent and worker
    ]
    affected_agents = sorted(
        {
            str(item.get("agent") or "")
            for item in [*missing_tools, *missing_workers]
            if str(item.get("agent") or "")
        }
    )
    return {
        "missing_tools": missing_tools,
        "missing_workers": missing_workers,
        "affected_agents": affected_agents,
    }


def build_runtime_error_payload(exc: BaseException | str) -> Dict[str, Any]:
    message = str(exc or "").strip() or "Runtime is unavailable."
    diagnostics = parse_runtime_registry_error(message)
    is_registry_error = bool(
        diagnostics["missing_tools"]
        or diagnostics["missing_workers"]
        or "Invalid next-runtime agent configuration" in message
    )
    if is_registry_error:
        return {
            "error_code": "runtime_registry_invalid",
            "message": "Invalid next-runtime agent configuration.",
            "detail": message,
            **diagnostics,
            "remediation": (
                "Rebuild/recreate the app image or container and rerun OpenWebUI bootstrap so "
                "agent definitions and the runtime tool registry come from the same code version."
            ),
        }
    return {
        "error_code": "runtime_unavailable",
        "message": "Runtime is unavailable.",
        "detail": message,
        "missing_tools": [],
        "missing_workers": [],
        "affected_agents": [],
        "remediation": "Check provider, database, and runtime startup logs, then restart the app service.",
    }


def runtime_error_payload_to_text(payload: Dict[str, Any]) -> str:
    code = str(payload.get("error_code") or "").strip()
    detail = str(payload.get("detail") or payload.get("message") or "").strip()
    if code == "runtime_registry_invalid":
        tool_names: List[str] = []
        for item in list(payload.get("missing_tools") or []):
            if isinstance(item, dict):
                tool = str(item.get("tool") or "").strip()
                if tool and tool not in tool_names:
                    tool_names.append(tool)
        affected_agents = [
            str(item).strip()
            for item in list(payload.get("affected_agents") or [])
            if str(item).strip()
        ]
        tool_text = ", ".join(tool_names) if tool_names else "one or more configured tools"
        agent_text = ", ".join(affected_agents) if affected_agents else "one or more agents"
        remediation = str(payload.get("remediation") or "").strip()
        return (
            f"Backend runtime registry is invalid: {agent_text} references missing tool(s) {tool_text}."
            + (f" {remediation}" if remediation else "")
        )
    return detail or str(payload.get("message") or "Runtime is unavailable.")


__all__ = [
    "build_runtime_error_payload",
    "parse_runtime_registry_error",
    "runtime_error_payload_to_text",
]
