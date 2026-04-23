from __future__ import annotations

from types import SimpleNamespace

from langchain_core.tools import tool

from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.tools import ToolDefinition
from agentic_chatbot_next.tools.discovery import ToolDiscoveryService, should_defer_for_agent
from agentic_chatbot_next.tools.executor import build_agent_tools
from agentic_chatbot_next.tools.registry import build_tool_definitions


def _settings(*, enabled: bool = True, require_search: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        deferred_tool_discovery_enabled=enabled,
        deferred_tool_discovery_top_k=8,
        deferred_tool_discovery_require_search=require_search,
    )


def _ctx(agent: AgentDefinition, *, settings: SimpleNamespace | None = None, metadata: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        settings=settings or _settings(),
        metadata=dict(metadata or {}),
        callbacks=[],
        active_agent=agent.name,
        active_definition=agent,
        session=SimpleNamespace(session_id="session-1", metadata={}),
        kernel=None,
        workspace_root=None,
    )


@tool("heavy_lookup")
def heavy_lookup(query: str) -> str:
    """Return a deterministic heavy lookup result."""

    return f"hit:{query}"


def _heavy_definition(**overrides) -> ToolDefinition:
    payload = {
        "name": "heavy_lookup",
        "group": "graph_gateway",
        "builder": lambda ctx: [heavy_lookup],
        "description": "Search heavy relationship evidence.",
        "args_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Lookup query."}},
            "required": ["query"],
        },
        "when_to_use": "Use for relationship-heavy graph evidence.",
        "output_description": "Returns graph evidence.",
        "keywords": ["relationship", "graph", "heavy"],
        "read_only": True,
        "background_safe": True,
        "should_defer": True,
        "search_hint": "Use for deferred relationship graph searches.",
        "defer_priority": 10,
    }
    payload.update(overrides)
    return ToolDefinition(**payload)


def test_registry_marks_graph_heavy_tools_deferred_with_specialist_eager_override() -> None:
    definitions = build_tool_definitions(None)
    general = AgentDefinition(name="general", mode="react", allowed_tools=["search_graph_index"])
    graph_manager = AgentDefinition(name="graph_manager", mode="react", allowed_tools=["search_graph_index"])

    assert definitions["list_graph_indexes"].should_defer is False
    assert definitions["inspect_graph_index"].should_defer is False
    assert definitions["search_graph_index"].should_defer is True
    assert should_defer_for_agent(definitions["search_graph_index"], general)
    assert not should_defer_for_agent(definitions["search_graph_index"], graph_manager)
    assert definitions["index_graph_corpus"].should_defer is True
    assert definitions["index_graph_corpus"].eager_for_agents == []


def test_build_agent_tools_flag_disabled_binds_deferred_tool_directly(monkeypatch) -> None:
    definition = _heavy_definition()
    monkeypatch.setattr(
        "agentic_chatbot_next.tools.executor.build_tool_definitions",
        lambda ctx: {"heavy_lookup": definition},
    )
    agent = AgentDefinition(name="general", mode="react", allowed_tools=["heavy_lookup"])

    tools = build_agent_tools(agent, _ctx(agent, settings=_settings(enabled=False)))

    assert [item.name for item in tools] == ["heavy_lookup"]


def test_build_agent_tools_flag_enabled_hides_deferred_tool_and_injects_facades() -> None:
    agent = AgentDefinition(name="general", mode="react", allowed_tools=["search_graph_index"])
    ctx = _ctx(agent, settings=_settings(enabled=True))

    tools = build_agent_tools(agent, ctx)

    assert {item.name for item in tools} == {"discover_tools", "call_deferred_tool"}
    assert all(item.metadata["concurrency_key"] == "deferred_tool_discovery" for item in tools)
    assert ctx.metadata["deferred_tool_discovery"]["tools"] == ["search_graph_index"]


def test_discovery_search_returns_only_allowed_policy_visible_targets() -> None:
    agent = AgentDefinition(name="general", mode="react", allowed_tools=["heavy_lookup"])
    ctx = _ctx(agent)
    service = ToolDiscoveryService(
        agent=agent,
        tool_context=ctx,
        definitions={
            "heavy_lookup": _heavy_definition(),
            "other_heavy": _heavy_definition(name="other_heavy"),
        },
    )

    result = service.search("relationship graph evidence")

    assert [match["name"] for match in result["matches"]] == ["heavy_lookup"]
    assert ctx.metadata["deferred_tool_discovered_targets"] == ["heavy_lookup"]


def test_deferred_invocation_requires_prior_discovery_and_then_calls_target() -> None:
    agent = AgentDefinition(name="general", mode="react", allowed_tools=["heavy_lookup"])
    ctx = _ctx(agent)
    service = ToolDiscoveryService(agent=agent, tool_context=ctx, definitions={"heavy_lookup": _heavy_definition()})

    denied = service.invoke("heavy_lookup", {"query": "dependencies"})
    service.search("relationship dependencies")
    allowed = service.invoke("heavy_lookup", {"query": "dependencies"})

    assert denied["status"] == "error"
    assert "discover_tools" in denied["error"]
    assert allowed["status"] == "ok"
    assert allowed["result"] == "hit:dependencies"


def test_deferred_invocation_rechecks_workspace_job_and_skill_policy() -> None:
    agent = AgentDefinition(name="general", mode="react", allowed_tools=["heavy_lookup"])

    workspace_ctx = _ctx(agent, metadata={"deferred_tool_discovered_targets": ["heavy_lookup"]})
    workspace_service = ToolDiscoveryService(
        agent=agent,
        tool_context=workspace_ctx,
        definitions={"heavy_lookup": _heavy_definition(requires_workspace=True)},
    )
    assert "policy denied" in workspace_service.invoke("heavy_lookup", {"query": "x"})["error"]

    job_ctx = _ctx(
        agent,
        metadata={"deferred_tool_discovered_targets": ["heavy_lookup"], "task_payload": {"job_id": "job-1"}},
    )
    job_service = ToolDiscoveryService(
        agent=agent,
        tool_context=job_ctx,
        definitions={"heavy_lookup": _heavy_definition(background_safe=False)},
    )
    assert "policy denied" in job_service.invoke("heavy_lookup", {"query": "x"})["error"]

    skill_ctx = _ctx(
        agent,
        metadata={
            "deferred_tool_discovered_targets": ["heavy_lookup"],
            "task_payload": {"skill_execution": {"allowed_tools": ["calculator"]}},
        },
    )
    skill_service = ToolDiscoveryService(
        agent=agent,
        tool_context=skill_ctx,
        definitions={"heavy_lookup": _heavy_definition()},
    )
    assert "policy denied" in skill_service.invoke("heavy_lookup", {"query": "x"})["error"]
