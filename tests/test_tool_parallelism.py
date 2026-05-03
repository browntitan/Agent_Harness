from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.tools import ToolDefinition
from agentic_chatbot_next.runtime.tool_parallelism import PolicyAwareToolNode
from agentic_chatbot_next.tools.executor import build_agent_tools


class _ConcurrencyProbe:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.events: list[tuple[str, str, str]] = []

    def enter(self, name: str, value: str) -> None:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.events.append(("start", name, value))

    def exit(self, name: str, value: str) -> None:
        with self._lock:
            self.events.append(("end", name, value))
            self.active -= 1


def _make_probe_tool(
    name: str,
    probe: _ConcurrencyProbe,
    *,
    delay: float = 0.05,
    metadata: dict[str, object] | None = None,
):
    @tool(name)
    def _probe_tool(value: str = "") -> str:
        """Probe tool for parallel scheduling tests."""
        probe.enter(name, value)
        try:
            time.sleep(delay)
            return f"{name}:{value}"
        finally:
            probe.exit(name, value)

    _probe_tool.metadata = dict(metadata or {})
    return _probe_tool


def _tool_call(name: str, call_id: str, *, value: str = "") -> dict[str, object]:
    return {"name": name, "args": {"value": value}, "id": call_id, "type": "tool_call"}


def _event_index(events: list[tuple[str, str, str]], kind: str, name: str) -> int:
    return next(index for index, event in enumerate(events) if event[:2] == (kind, name))


class _EventSink:
    def __init__(self) -> None:
        self.events: list[object] = []

    def emit(self, event) -> None:
        self.events.append(event)


def _tool_context(event_sink: _EventSink) -> SimpleNamespace:
    return SimpleNamespace(
        event_sink=event_sink,
        session=SimpleNamespace(session_id="tenant:user:conv", conversation_id="conv"),
        active_agent="general",
        metadata={"job_id": "job-1"},
    )


def test_policy_aware_tool_node_runs_independent_tools_in_parallel_and_preserves_order() -> None:
    probe = _ConcurrencyProbe()
    alpha = _make_probe_tool("alpha_tool", probe)
    beta = _make_probe_tool("beta_tool", probe)
    node = PolicyAwareToolNode([alpha, beta], max_tool_calls=4, max_parallel_tool_calls=4)

    result = node.invoke(
        {
            "messages": [
                HumanMessage(content="Run the independent tools."),
                AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call("alpha_tool", "tool_alpha", value="first"),
                        _tool_call("beta_tool", "tool_beta", value="second"),
                    ],
                ),
            ]
        }
    )

    outputs = result["messages"]
    assert [message.content for message in outputs] == ["alpha_tool:first", "beta_tool:second"]
    assert probe.peak == 2


def test_policy_aware_tool_node_emits_parallel_tool_wave_audit_events() -> None:
    probe = _ConcurrencyProbe()
    event_sink = _EventSink()
    alpha = _make_probe_tool("alpha_tool", probe)
    beta = _make_probe_tool("beta_tool", probe)
    node = PolicyAwareToolNode(
        [alpha, beta],
        max_tool_calls=4,
        max_parallel_tool_calls=4,
        tool_context=_tool_context(event_sink),
    )

    result = node.invoke(
        {
            "messages": [
                HumanMessage(content="Run the independent tools."),
                AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call("alpha_tool", "tool_alpha", value="first"),
                        _tool_call("beta_tool", "tool_beta", value="second"),
                    ],
                ),
            ]
        }
    )

    assert [message.content for message in result["messages"]] == ["alpha_tool:first", "beta_tool:second"]
    assert [event.event_type for event in event_sink.events] == [
        "tool_parallel_group_started",
        "tool_parallel_group_completed",
    ]
    started = event_sink.events[0]
    completed = event_sink.events[1]
    assert started.session_id == "tenant:user:conv"
    assert started.agent_name == "general"
    assert started.job_id == "job-1"
    assert started.payload["group_kind"] == "tool_wave"
    assert started.payload["execution_mode"] == "parallel"
    assert started.payload["size"] == 2
    assert [member["tool_call_id"] for member in started.payload["members"]] == ["tool_alpha", "tool_beta"]
    assert completed.payload["group_id"] == started.payload["group_id"]
    assert completed.payload["status"] == "completed"
    assert completed.payload["duration_ms"] >= 0


def test_policy_aware_tool_node_serializes_conflicting_concurrency_keys() -> None:
    probe = _ConcurrencyProbe()
    alpha = _make_probe_tool("alpha_tool", probe, metadata={"concurrency_key": "memory"})
    beta = _make_probe_tool("beta_tool", probe, metadata={"concurrency_key": "memory"})
    node = PolicyAwareToolNode([alpha, beta], max_tool_calls=4, max_parallel_tool_calls=4)

    result = node.invoke(
        {
            "messages": [
                HumanMessage(content="Serialize the memory tools."),
                AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call("alpha_tool", "tool_alpha", value="first"),
                        _tool_call("beta_tool", "tool_beta", value="second"),
                    ],
                ),
            ]
        }
    )

    outputs = result["messages"]
    assert [message.content for message in outputs] == ["alpha_tool:first", "beta_tool:second"]
    assert probe.peak == 1


def test_policy_aware_tool_node_marks_conflicting_tool_waves_as_sequential() -> None:
    probe = _ConcurrencyProbe()
    event_sink = _EventSink()
    alpha = _make_probe_tool("alpha_tool", probe, metadata={"concurrency_key": "memory"})
    beta = _make_probe_tool("beta_tool", probe, metadata={"concurrency_key": "memory"})
    node = PolicyAwareToolNode(
        [alpha, beta],
        max_tool_calls=4,
        max_parallel_tool_calls=4,
        tool_context=_tool_context(event_sink),
    )

    result = node.invoke(
        {
            "messages": [
                HumanMessage(content="Serialize the memory tools."),
                AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call("alpha_tool", "tool_alpha", value="first"),
                        _tool_call("beta_tool", "tool_beta", value="second"),
                    ],
                ),
            ]
        }
    )

    assert [message.content for message in result["messages"]] == ["alpha_tool:first", "beta_tool:second"]
    assert probe.peak == 1
    started_events = [event for event in event_sink.events if event.event_type == "tool_parallel_group_started"]
    assert len(started_events) == 2
    assert {event.payload["execution_mode"] for event in started_events} == {"sequential"}
    assert [event.payload["size"] for event in started_events] == [1, 1]


def test_policy_aware_tool_node_respects_max_parallel_tool_calls() -> None:
    probe = _ConcurrencyProbe()
    alpha = _make_probe_tool("alpha_tool", probe)
    beta = _make_probe_tool("beta_tool", probe)
    gamma = _make_probe_tool("gamma_tool", probe)
    node = PolicyAwareToolNode([alpha, beta, gamma], max_tool_calls=6, max_parallel_tool_calls=2)

    result = node.invoke(
        {
            "messages": [
                HumanMessage(content="Run three tools with a cap of two."),
                AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call("alpha_tool", "tool_alpha", value="first"),
                        _tool_call("beta_tool", "tool_beta", value="second"),
                        _tool_call("gamma_tool", "tool_gamma", value="third"),
                    ],
                ),
            ]
        }
    )

    outputs = result["messages"]
    assert [message.content for message in outputs] == [
        "alpha_tool:first",
        "beta_tool:second",
        "gamma_tool:third",
    ]
    assert probe.peak == 2


def test_policy_aware_tool_node_returns_budget_errors_for_overflow_calls() -> None:
    probe = _ConcurrencyProbe()
    alpha = _make_probe_tool("alpha_tool", probe)
    beta = _make_probe_tool("beta_tool", probe)
    node = PolicyAwareToolNode([alpha, beta], max_tool_calls=1, max_parallel_tool_calls=4)

    result = node.invoke(
        {
            "messages": [
                HumanMessage(content="Use too many tools."),
                AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call("alpha_tool", "tool_alpha", value="first"),
                        _tool_call("beta_tool", "tool_beta", value="second"),
                    ],
                ),
            ]
        }
    )

    outputs = result["messages"]
    assert outputs[0].content == "alpha_tool:first"
    assert outputs[1].status == "error"
    assert "budget exceeded" in str(outputs[1].content).lower()


def test_policy_aware_tool_node_counts_prior_tool_messages_against_budget() -> None:
    probe = _ConcurrencyProbe()
    alpha = _make_probe_tool("alpha_tool", probe)
    node = PolicyAwareToolNode([alpha], max_tool_calls=1, max_parallel_tool_calls=4)

    result = node.invoke(
        {
            "messages": [
                HumanMessage(content="Continue the same turn."),
                ToolMessage(content="already ran", tool_call_id="tool_prev", name="existing_tool"),
                AIMessage(
                    content="",
                    tool_calls=[_tool_call("alpha_tool", "tool_alpha", value="first")],
                ),
            ]
        }
    )

    outputs = result["messages"]
    assert outputs[0].status == "error"
    assert "budget exceeded" in str(outputs[0].content).lower()
    assert probe.peak == 0


def test_policy_aware_tool_node_works_inside_langgraph_react_agent() -> None:
    probe = _ConcurrencyProbe()
    alpha = _make_probe_tool("alpha_tool", probe)

    class _ToolBindingFakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            del tools, tool_choice, kwargs
            return self

    model = _ToolBindingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[_tool_call("alpha_tool", "tool_alpha", value="first")],
            ),
            AIMessage(content="Final answer from graph."),
        ]
    )
    graph = create_react_agent(model, tools=PolicyAwareToolNode([alpha], max_tool_calls=2))

    result = graph.invoke({"messages": [HumanMessage(content="Run the alpha tool.")]})

    assert [message.content for message in result["messages"] if isinstance(message, ToolMessage)] == [
        "alpha_tool:first"
    ]
    assert result["messages"][-1].content == "Final answer from graph."


def test_policy_aware_tool_node_accepts_new_langgraph_runtime_shape(monkeypatch) -> None:
    probe = _ConcurrencyProbe()
    alpha = _make_probe_tool("alpha_tool", probe)
    node = PolicyAwareToolNode([alpha], max_tool_calls=2)
    monkeypatch.delattr(node, "messages_key", raising=False)
    node._messages_key = "messages"
    captured: list[object] = []
    extracted: list[object] = []

    class _FakeToolRuntime:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    import langgraph.prebuilt.tool_node as tool_node_module

    monkeypatch.setattr(tool_node_module, "ToolRuntime", _FakeToolRuntime, raising=False)

    def fake_extract_state(input_payload, config):
        extracted.append(config)
        return input_payload
    monkeypatch.setattr(node, "_extract_state", fake_extract_state, raising=False)

    def fake_run_one(call, input_type, tool_runtime):
        del input_type
        captured.append(tool_runtime)
        return ToolMessage(
            content=f"{call['name']}:{tool_runtime.tool_call_id}:{tool_runtime.store}",
            name=call["name"],
            tool_call_id=call["id"],
        )

    node._run_one = fake_run_one
    result = node._func(
        {
            "messages": [
                HumanMessage(content="Run the tool."),
                AIMessage(content="", tool_calls=[_tool_call("alpha_tool", "tool_alpha", value="first")]),
            ]
        },
        {},
        SimpleNamespace(
            context={"tenant": "tenant"},
            store="runtime-store",
            stream_writer=None,
            execution_info=None,
            server_info=None,
        ),
    )

    assert result["messages"][0].content == "alpha_tool:tool_alpha:runtime-store"
    assert captured[0].context == {"tenant": "tenant"}
    assert captured[0].state["messages"][0].content == "Run the tool."
    assert extracted and extracted[0]["metadata"]["agentic_parallel_group_size"] == 1


def test_policy_aware_tool_node_builds_tool_runtime_when_runtime_arg_missing(monkeypatch) -> None:
    probe = _ConcurrencyProbe()
    alpha = _make_probe_tool("alpha_tool", probe)
    node = PolicyAwareToolNode([alpha], max_tool_calls=2)
    captured: list[object] = []

    class _FakeToolRuntime:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    import langgraph.prebuilt.tool_node as tool_node_module

    monkeypatch.setattr(tool_node_module, "ToolRuntime", _FakeToolRuntime, raising=False)

    def fake_run_one(call, input_type, tool_runtime):
        del input_type
        captured.append(tool_runtime)
        return ToolMessage(
            content=f"{call['name']}:{tool_runtime.tool_call_id}:{tool_runtime.store}",
            name=call["name"],
            tool_call_id=call["id"],
        )

    node._run_one = fake_run_one
    result = node._func(
        {
            "messages": [
                HumanMessage(content="Run the tool."),
                AIMessage(content="", tool_calls=[_tool_call("alpha_tool", "tool_alpha", value="first")]),
            ]
        },
        {},
    )

    assert result["messages"][0].content == "alpha_tool:tool_alpha:None"
    assert captured[0].tool_call_id == "tool_alpha"


def test_policy_aware_tool_node_accepts_internal_underscore_tool_map(monkeypatch) -> None:
    probe = _ConcurrencyProbe()
    alpha = _make_probe_tool("alpha_tool", probe)
    node = PolicyAwareToolNode([alpha], max_tool_calls=2)
    monkeypatch.delattr(node, "tools_by_name", raising=False)
    node._tools_by_name = {"alpha_tool": alpha}

    result = node.invoke(
        {
            "messages": [
                HumanMessage(content="Run the tool."),
                AIMessage(content="", tool_calls=[_tool_call("alpha_tool", "tool_alpha", value="first")]),
            ]
        }
    )

    assert [message.content for message in result["messages"]] == ["alpha_tool:first"]


def test_policy_aware_tool_node_serializes_requirements_extract_then_export() -> None:
    probe = _ConcurrencyProbe()
    extract_tool = _make_probe_tool(
        "extract_requirement_statements",
        probe,
        metadata={"concurrency_key": "requirements_inventory"},
    )
    export_tool = _make_probe_tool(
        "export_requirement_statements",
        probe,
        metadata={"concurrency_key": "requirements_inventory"},
    )
    node = PolicyAwareToolNode([extract_tool, export_tool], max_tool_calls=4, max_parallel_tool_calls=4)

    result = node.invoke(
        {
            "messages": [
                HumanMessage(content="Preview and export the shall statements."),
                AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call("extract_requirement_statements", "tool_extract", value="preview"),
                        _tool_call("export_requirement_statements", "tool_export", value="csv"),
                    ],
                ),
            ]
        }
    )

    outputs = result["messages"]
    assert [message.content for message in outputs] == [
        "extract_requirement_statements:preview",
        "export_requirement_statements:csv",
    ]
    assert probe.peak == 1
    assert _event_index(probe.events, "start", "extract_requirement_statements") < _event_index(
        probe.events, "start", "export_requirement_statements"
    )


def test_policy_aware_tool_node_serializes_repeated_requirement_extractions() -> None:
    probe = _ConcurrencyProbe()
    extract_tool = _make_probe_tool(
        "extract_requirement_statements",
        probe,
        metadata={"concurrency_key": "requirements_inventory"},
    )
    node = PolicyAwareToolNode([extract_tool], max_tool_calls=4, max_parallel_tool_calls=4)

    result = node.invoke(
        {
            "messages": [
                HumanMessage(content="Extract requirements twice."),
                AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call("extract_requirement_statements", "tool_extract_one", value="doc-a"),
                        _tool_call("extract_requirement_statements", "tool_extract_two", value="doc-b"),
                    ],
                ),
            ]
        }
    )

    outputs = result["messages"]
    assert [message.content for message in outputs] == [
        "extract_requirement_statements:doc-a",
        "extract_requirement_statements:doc-b",
    ]
    assert probe.peak == 1


def test_policy_aware_tool_node_serializes_requirement_export_even_with_parallel_safe_reads() -> None:
    probe = _ConcurrencyProbe()
    extract_tool = _make_probe_tool(
        "extract_requirement_statements",
        probe,
        metadata={"concurrency_key": "requirements_inventory"},
    )
    export_tool = _make_probe_tool(
        "export_requirement_statements",
        probe,
        metadata={"concurrency_key": "requirements_inventory"},
    )
    search_tool = _make_probe_tool("search_indexed_docs", probe)
    node = PolicyAwareToolNode(
        [extract_tool, export_tool, search_tool],
        max_tool_calls=6,
        max_parallel_tool_calls=4,
    )

    result = node.invoke(
        {
            "messages": [
                HumanMessage(content="Preview, search, and export requirements."),
                AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call("extract_requirement_statements", "tool_extract", value="preview"),
                        _tool_call("search_indexed_docs", "tool_search", value="rfp"),
                        _tool_call("export_requirement_statements", "tool_export", value="csv"),
                    ],
                ),
            ]
        }
    )

    outputs = result["messages"]
    assert [message.content for message in outputs] == [
        "extract_requirement_statements:preview",
        "search_indexed_docs:rfp",
        "export_requirement_statements:csv",
    ]
    assert probe.peak == 2
    assert _event_index(probe.events, "start", "export_requirement_statements") > _event_index(
        probe.events, "end", "extract_requirement_statements"
    )


def test_build_agent_tools_attaches_scheduling_metadata(monkeypatch) -> None:
    @tool("memory_echo")
    def memory_echo(value: str = "") -> str:
        """Echo the provided value."""
        return value

    definition = ToolDefinition(
        name="memory_echo",
        group="memory",
        builder=lambda ctx: [memory_echo],
        description="Echo memory input.",
        args_schema={"type": "object", "properties": {"value": {"type": "string"}}},
        when_to_use="Use for metadata propagation tests.",
        output_description="Returns the provided value.",
        read_only=True,
        requires_workspace=True,
        concurrency_key="memory",
    )

    class _AllowAllPolicy:
        def is_allowed(self, agent, definition, tool_context) -> bool:
            del agent, definition, tool_context
            return True

    monkeypatch.setattr(
        "agentic_chatbot_next.tools.executor.build_tool_definitions",
        lambda ctx: {"memory_echo": definition},
    )

    tools = build_agent_tools(
        AgentDefinition(name="general", mode="react", allowed_tools=["memory_echo"]),
        object(),
        policy_service=_AllowAllPolicy(),
    )

    assert len(tools) == 1
    assert tools[0].metadata["concurrency_key"] == "memory"
    assert tools[0].metadata["group"] == "memory"
    assert tools[0].metadata["read_only"] is True
    assert tools[0].metadata["requires_workspace"] is True
    assert "When to use" in tools[0].description
