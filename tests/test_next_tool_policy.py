from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.capabilities import EffectiveCapabilities
from agentic_chatbot_next.agents.registry import AgentRegistry
from agentic_chatbot_next.runtime.context import RuntimePaths
from agentic_chatbot_next.tools.base import ToolContext
from agentic_chatbot_next.tools.policy import ToolPolicyService
from agentic_chatbot_next.tools.registry import build_tool_definitions


def _paths(tmp_path: Path) -> RuntimePaths:
    settings = SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        workspace_dir=tmp_path / "workspaces",
        memory_dir=tmp_path / "memory",
    )
    return RuntimePaths.from_settings(settings)


def _tool_context(tmp_path: Path, *, workspace: bool = False, task_payload: dict | None = None) -> ToolContext:
    paths = _paths(tmp_path)
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conversation")
    if workspace:
        session.workspace_root = str(paths.workspace_dir(session.session_id))
    return ToolContext(
        settings=SimpleNamespace(workspace_dir=paths.workspace_root),
        providers=None,
        stores=None,
        session=session,
        paths=paths,
        metadata={"task_payload": dict(task_payload or {})},
    )


def test_tool_policy_requires_workspace(tmp_path: Path) -> None:
    definitions = build_tool_definitions(None)
    policy = ToolPolicyService()
    agent = AgentDefinition(
        name="data_analyst",
        mode="react",
        prompt_file="data_analyst_agent.md",
        allowed_tools=["workspace_read"],
    )
    assert not policy.is_allowed(agent, definitions["workspace_read"], _tool_context(tmp_path, workspace=False))
    assert policy.is_allowed(agent, definitions["workspace_read"], _tool_context(tmp_path, workspace=True))


def test_tool_policy_blocks_non_background_safe_tools_in_job_context(tmp_path: Path) -> None:
    definitions = build_tool_definitions(None)
    policy = ToolPolicyService()
    agent = AgentDefinition(
        name="general",
        mode="react",
        prompt_file="general_agent.md",
        allowed_tools=["spawn_worker", "invoke_agent", "rag_agent_tool"],
    )
    bg_ctx = _tool_context(tmp_path, task_payload={"job_id": "job_123"})
    assert not policy.is_allowed(agent, definitions["spawn_worker"], bg_ctx)
    assert policy.is_allowed(agent, definitions["invoke_agent"], bg_ctx)
    assert policy.is_allowed(agent, definitions["rag_agent_tool"], bg_ctx)


def test_tool_policy_limits_worker_request_tools_to_job_context(tmp_path: Path) -> None:
    definitions = build_tool_definitions(None)
    policy = ToolPolicyService()
    agent = AgentDefinition(
        name="utility",
        mode="react",
        prompt_file="utility_agent.md",
        allowed_tools=["request_parent_question", "request_parent_approval"],
    )

    assert not policy.is_allowed(agent, definitions["request_parent_question"], _tool_context(tmp_path))
    assert not policy.is_allowed(agent, definitions["request_parent_approval"], _tool_context(tmp_path))
    assert policy.is_allowed(
        agent,
        definitions["request_parent_question"],
        _tool_context(tmp_path, task_payload={"job_id": "job_123"}),
    )
    assert policy.is_allowed(
        agent,
        definitions["request_parent_approval"],
        _tool_context(tmp_path, task_payload={"job_id": "job_123"}),
    )


def test_tool_policy_gates_team_mailbox_tools_by_feature_flag(tmp_path: Path) -> None:
    definitions = build_tool_definitions(None)
    policy = ToolPolicyService()
    agent = AgentDefinition(
        name="general",
        mode="react",
        prompt_file="general_agent.md",
        allowed_tools=["create_team_channel", "list_team_messages"],
    )
    disabled = _tool_context(tmp_path)
    enabled = _tool_context(tmp_path)
    enabled.settings.team_mailbox_enabled = True

    assert not policy.is_allowed(agent, definitions["create_team_channel"], disabled)
    assert not policy.is_allowed(agent, definitions["list_team_messages"], disabled)
    assert policy.is_allowed(agent, definitions["create_team_channel"], enabled)
    assert policy.is_allowed(agent, definitions["list_team_messages"], enabled)


def test_tool_policy_blocks_nested_rag_agent_tool_for_research_inventory_synthesis(tmp_path: Path) -> None:
    definitions = build_tool_definitions(None)
    policy = ToolPolicyService()
    agent = AgentDefinition(
        name="general",
        mode="react",
        prompt_file="general_agent.md",
        allowed_tools=["rag_agent_tool", "read_indexed_doc"],
    )
    ctx = _tool_context(
        tmp_path,
        task_payload={
            "worker_request": {
                "task_id": "task_6",
                "title": "Consolidate subsystem inventory",
                "handoff_schema": "research_inventory",
                "consumes_artifacts": ["doc_digest", "facet_matches"],
                "controller_hints": {"final_output_mode": "detailed_subsystem_summary"},
            }
        },
    )

    assert not policy.is_allowed(agent, definitions["rag_agent_tool"], ctx)
    assert policy.is_allowed(agent, definitions["read_indexed_doc"], ctx)


def test_tool_policy_enforces_read_only_modes(tmp_path: Path) -> None:
    definitions = build_tool_definitions(None)
    policy = ToolPolicyService()
    agent = AgentDefinition(
        name="verifier",
        mode="verifier",
        prompt_file="verifier_agent.md",
        allowed_tools=["memory_save", "memory_load"],
    )
    ctx = _tool_context(tmp_path)
    assert not policy.is_allowed(agent, definitions["memory_save"], ctx)
    assert policy.is_allowed(agent, definitions["memory_load"], ctx)


def test_tool_policy_requires_tool_grant_when_authz_is_enabled(tmp_path: Path) -> None:
    definitions = build_tool_definitions(None)
    policy = ToolPolicyService()
    agent = AgentDefinition(
        name="general",
        mode="react",
        prompt_file="general_agent.md",
        allowed_tools=["calculator"],
    )
    ctx = _tool_context(tmp_path)
    ctx.session.metadata["access_summary"] = {
        "authz_enabled": True,
        "resources": {
            "tool": {"use": [], "manage": [], "use_all": False, "manage_all": False},
        },
    }
    assert not policy.is_allowed(agent, definitions["calculator"], ctx)
    ctx.session.metadata["access_summary"]["resources"]["tool"]["use"] = ["calculator"]
    assert policy.is_allowed(agent, definitions["calculator"], ctx)


def test_tool_policy_clips_tools_inside_skill_execution(tmp_path: Path) -> None:
    definitions = build_tool_definitions(None)
    policy = ToolPolicyService()
    agent = AgentDefinition(
        name="utility",
        mode="react",
        prompt_file="utility_agent.md",
        allowed_tools=["calculator", "search_skills", "execute_skill"],
    )
    ctx = _tool_context(
        tmp_path,
        task_payload={
            "skill_execution": {
                "skill_id": "math-skill",
                "allowed_tools": ["calculator"],
            }
        },
    )

    assert policy.is_allowed(agent, definitions["calculator"], ctx)
    assert not policy.is_allowed(agent, definitions["search_skills"], ctx)
    assert not policy.is_allowed(agent, definitions["execute_skill"], ctx)


def test_tool_policy_blocks_disabled_tool_from_effective_capabilities(tmp_path: Path) -> None:
    definitions = build_tool_definitions(None)
    policy = ToolPolicyService()
    agent = AgentDefinition(
        name="utility",
        mode="react",
        prompt_file="utility_agent.md",
        allowed_tools=["calculator", "list_indexed_docs"],
    )
    ctx = _tool_context(tmp_path)
    ctx.metadata["effective_capabilities"] = EffectiveCapabilities(disabled_tools=["calculator"]).to_dict()

    assert not policy.is_allowed(agent, definitions["calculator"], ctx)
    assert policy.is_allowed(agent, definitions["list_indexed_docs"], ctx)


def test_tool_policy_restricted_mode_allows_read_only_tools_only(tmp_path: Path) -> None:
    definitions = build_tool_definitions(None)
    policy = ToolPolicyService()
    agent = AgentDefinition(
        name="general",
        mode="react",
        prompt_file="general_agent.md",
        allowed_tools=["list_jobs", "stop_job"],
    )
    ctx = _tool_context(tmp_path)
    ctx.metadata["effective_capabilities"] = EffectiveCapabilities(permission_mode="restricted").to_dict()

    assert policy.is_allowed(agent, definitions["list_jobs"], ctx)
    assert not policy.is_allowed(agent, definitions["stop_job"], ctx)


def test_research_coordinator_has_no_terminal_or_execute_code_tools_by_default() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    registry = AgentRegistry(repo_root / "data" / "agents")
    agent = registry.get("research_coordinator")

    assert agent is not None
    forbidden = {"execute_code", "workspace_write", "workspace_read", "terminal", "shell"}
    assert forbidden.isdisjoint(set(agent.allowed_tools))
