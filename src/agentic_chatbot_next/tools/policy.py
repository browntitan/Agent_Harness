from __future__ import annotations

from typing import Any

from agentic_chatbot_next.authz import access_summary_allows, access_summary_authz_enabled
from agentic_chatbot_next.capabilities import coerce_effective_capabilities
from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.tools import ToolDefinition


def tool_selector_matches(selector: str, tool_name: str) -> bool:
    clean_selector = str(selector or "").strip()
    clean_tool = str(tool_name or "").strip()
    if not clean_selector or not clean_tool:
        return False
    if clean_selector == clean_tool or clean_selector == "*":
        return True
    if clean_selector.endswith("*"):
        return clean_tool.startswith(clean_selector[:-1])
    return False


def tool_allowed_by_selectors(selectors: list[str] | tuple[str, ...] | set[str], tool_name: str) -> bool:
    return any(tool_selector_matches(str(selector), tool_name) for selector in selectors or [])


class ToolPolicyService:
    """Central tool policy for the next runtime."""

    READ_ONLY_ONLY_MODES = {"basic", "planner", "finalizer", "verifier", "rag"}
    TEAM_MAILBOX_TOOLS = {
        "create_team_channel",
        "post_team_message",
        "list_team_messages",
        "claim_team_messages",
        "respond_team_message",
    }

    def is_allowed(
        self,
        agent: AgentDefinition,
        tool: ToolDefinition | str,
        tool_context: Any | None = None,
    ) -> bool:
        tool_name = tool if isinstance(tool, str) else tool.name
        if not tool_allowed_by_selectors(list(agent.allowed_tools or []), tool_name):
            return False
        if isinstance(tool, str):
            return True

        metadata = dict((tool_context.metadata if tool_context is not None else {}) or {})
        task_payload = dict(metadata.get("task_payload") or {})
        effective_capabilities = coerce_effective_capabilities(
            metadata.get("effective_capabilities")
            or dict(getattr(getattr(tool_context, "session", None), "metadata", {}) or {}).get("effective_capabilities")
        )
        if effective_capabilities is not None and not effective_capabilities.allows_tool(
            tool_name,
            group=tool.group,
            read_only=bool(tool.read_only),
            destructive=bool(tool.destructive),
            metadata=dict(getattr(tool, "metadata", {}) or {}),
        ):
            return False
        if tool_name in self.TEAM_MAILBOX_TOOLS and not bool(
            getattr(getattr(tool_context, "settings", None), "team_mailbox_enabled", False)
        ):
            return False
        skill_execution = dict(task_payload.get("skill_execution") or {})
        if skill_execution:
            if tool_name == "execute_skill":
                return False
            allowed_by_skill = {
                str(item).strip()
                for item in (skill_execution.get("allowed_tools") or [])
                if str(item).strip()
            }
            if not tool_allowed_by_selectors(allowed_by_skill, tool_name):
                return False
        if tool.group == "mcp":
            if not bool(getattr(getattr(tool_context, "settings", None), "mcp_tool_plane_enabled", False)):
                return False
            tool_metadata = dict(getattr(tool, "metadata", {}) or {})
            allowed_agents = {
                str(item).strip()
                for item in (tool_metadata.get("allowed_agents") or [])
                if str(item).strip()
            }
            if allowed_agents and "*" not in allowed_agents and agent.name not in allowed_agents:
                return False
            session = getattr(tool_context, "session", None) if tool_context is not None else None
            session_tenant = str(getattr(session, "tenant_id", "") or "")
            session_user = str(getattr(session, "user_id", "") or "")
            tool_tenant = str(tool_metadata.get("tenant_id") or "")
            tool_owner = str(tool_metadata.get("owner_user_id") or "")
            visibility = str(tool_metadata.get("visibility") or "private").strip().lower()
            if tool_tenant and session_tenant and tool_tenant != session_tenant:
                return False
            if visibility == "private" and tool_owner and session_user and tool_owner != session_user:
                return False
        if tool_name in {"request_parent_question", "request_parent_approval"} and not task_payload.get("job_id"):
            return False
        session_metadata = dict((getattr(getattr(tool_context, "session", None), "metadata", {}) or {}) if tool_context is not None else {})
        access_summary = dict(metadata.get("access_summary") or session_metadata.get("access_summary") or {})
        mcp_owned = False
        if not isinstance(tool, str) and getattr(tool, "group", "") == "mcp":
            tool_metadata = dict(getattr(tool, "metadata", {}) or {})
            session = getattr(tool_context, "session", None) if tool_context is not None else None
            mcp_owned = (
                str(tool_metadata.get("tenant_id") or "") == str(getattr(session, "tenant_id", "") or "")
                and str(tool_metadata.get("owner_user_id") or "") == str(getattr(session, "user_id", "") or "")
            )
        if access_summary_authz_enabled(access_summary) and not mcp_owned and not access_summary_allows(
            access_summary,
            "tool",
            tool_name,
            action="use",
        ):
            return False

        if tool.requires_workspace:
            workspace_root = getattr(tool_context, "workspace_root", None) if tool_context is not None else None
            if workspace_root is None:
                return False

        if task_payload.get("job_id") and not tool.background_safe:
            return False

        worker_request = dict(task_payload.get("worker_request") or {})
        if (
            tool_name == "rag_agent_tool"
            and agent.name == "general"
            and worker_request
            and (
                str(worker_request.get("handoff_schema") or "").strip() == "research_inventory"
                or [str(item) for item in (worker_request.get("consumes_artifacts") or []) if str(item)]
                or str(dict(worker_request.get("controller_hints") or {}).get("final_output_mode") or "").strip()
            )
        ):
            return False

        if agent.mode in self.READ_ONLY_ONLY_MODES and not tool.read_only:
            return False

        if agent.mode == "memory_maintainer" and tool.group != "memory":
            return False

        if bool((tool_context.metadata if tool_context is not None else {}).get("read_only_only")) and not tool.read_only:
            return False

        return True
