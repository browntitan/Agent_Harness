from __future__ import annotations
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import FakeListChatModel, FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from agentic_chatbot_next.app.service import AppContext, RuntimeService
from agentic_chatbot_next.api.live_progress import LiveProgressSink
from agentic_chatbot_next.api.progress_callback import ProgressCallback
from agentic_chatbot_next.observability.callbacks import RuntimeTraceCallbackHandler
from agentic_chatbot_next.observability.events import RuntimeEvent
from agentic_chatbot_next.providers.circuit_breaker import CircuitBreakerOpenError
from agentic_chatbot_next.runtime.context import RuntimePaths
from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.messages import RuntimeMessage, SessionState
from agentic_chatbot_next.runtime.kernel import AgentRunResult, RuntimeKernel
from agentic_chatbot_next.runtime.query_loop import QueryLoop
from agentic_chatbot_next.runtime.tool_parallelism import PolicyAwareToolNode
from agentic_chatbot_next.runtime.turn_contracts import resolve_turn_intent
from agentic_chatbot_next.runtime.transcript_store import RuntimeTranscriptStore
from agentic_chatbot_next.session import ChatSession


def _runtime_settings(tmp_path: Path) -> SimpleNamespace:
    repo_root = Path(__file__).resolve().parents[1]
    return SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        agents_dir=repo_root / "data" / "agents",
        max_worker_concurrency=2,
        max_parallel_tool_calls=4,
        chat_max_output_tokens=512,
        enable_coordinator_mode=False,
        runtime_events_enabled=True,
        planner_max_tasks=4,
        rag_top_k_vector=2,
        rag_top_k_keyword=2,
        rag_max_retries=1,
        llm_router_enabled=False,
        llm_router_confidence_threshold=0.70,
        workspace_dir=None,
        clear_scratchpad_per_turn=False,
        agent_runtime_mode="planner_executor",
        default_tenant_id="local-dev",
        default_user_id="local-cli",
        default_conversation_id="local-session",
    )


def test_basic_turn_persists_router_and_basic_runtime_events(tmp_path: Path, monkeypatch):
    settings = _runtime_settings(tmp_path)
    providers = SimpleNamespace(
        chat=FakeListChatModel(responses=["Hello back from BASIC."]),
        judge=FakeListChatModel(responses=["unused"]),
        embeddings=object(),
    )
    stores = SimpleNamespace()

    monkeypatch.setattr("agentic_chatbot_next.app.service.load_basic_chat_skills", lambda settings: "Be concise.")
    monkeypatch.setattr("agentic_chatbot_next.app.service.ensure_kb_indexed", lambda *args, **kwargs: None)

    class _NoopSkillIndexSync:
        def __init__(self, *args, **kwargs):
            pass

        def sync(self, tenant_id: str) -> None:
            return None

    monkeypatch.setattr("agentic_chatbot_next.app.service.SkillIndexSync", _NoopSkillIndexSync)

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="basic-conv")

    text = app.process_turn(session, user_text="Hello there")

    assert "Hello back from BASIC" in text
    store = RuntimeTranscriptStore(RuntimePaths.from_settings(settings))
    events = store.load_session_events(session.session_id)
    event_types = {row["event_type"] for row in events}
    assert {
        "router_decision",
        "basic_turn_started",
        "model_start",
        "model_end",
        "basic_turn_completed",
    }.issubset(event_types)
    router_event = next(row for row in events if row["event_type"] == "router_decision")
    assert router_event["payload"]["route"] == "BASIC"
    assert router_event["payload"]["router_decision_id"]


def test_process_turn_requested_agent_override_preserves_router_decision_and_changes_starting_agent(tmp_path: Path, monkeypatch):
    settings = _runtime_settings(tmp_path)
    providers = SimpleNamespace(
        chat=object(),
        judge=object(),
        embeddings=object(),
    )
    stores = SimpleNamespace()

    monkeypatch.setattr("agentic_chatbot_next.app.service.load_basic_chat_skills", lambda settings: "Be concise.")
    monkeypatch.setattr("agentic_chatbot_next.app.service.ensure_kb_indexed", lambda *args, **kwargs: None)

    class _NoopSkillIndexSync:
        def __init__(self, *args, **kwargs):
            pass

        def sync(self, tenant_id: str) -> None:
            return None

    monkeypatch.setattr("agentic_chatbot_next.app.service.SkillIndexSync", _NoopSkillIndexSync)
    monkeypatch.setattr(
        "agentic_chatbot_next.app.service.route_turn",
        lambda *args, **kwargs: SimpleNamespace(
            route="AGENT",
            confidence=0.9,
            reasons=["document_grounding_intent"],
            router_method="deterministic",
            suggested_agent="rag_worker",
        ),
    )

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="override-conv")

    monkeypatch.setattr(
        app.kernel,
        "run_agent",
        lambda agent, session_state, *, user_text, callbacks, task_payload=None, chat_max_output_tokens=None: AgentRunResult(
            text="override answer",
            messages=list(session_state.messages)
            + [RuntimeMessage(role="assistant", content="override answer", metadata={"agent_name": agent.name})],
            metadata={"agent_name": agent.name},
        ),
    )

    text = app.process_turn(
        session,
        user_text="Find the grounded answer.",
        requested_agent="general",
    )

    assert text == "override answer"
    store = RuntimeTranscriptStore(RuntimePaths.from_settings(settings))
    events = store.load_session_events(session.session_id)
    router_event = next(row for row in events if row["event_type"] == "router_decision")
    agent_turn_started = next(row for row in events if row["event_type"] == "agent_turn_started")
    router_outcome = next(row for row in events if row["event_type"] == "router_outcome")
    mispick_sample = next(row for row in events if row["event_type"] == "router_mispick_sampled")

    assert router_event["payload"]["route"] == "AGENT"
    assert router_event["payload"]["suggested_agent"] == "rag_worker"
    assert router_event["payload"]["requested_agent_override"] == "general"
    assert router_event["payload"]["requested_agent_override_applied"] is False
    assert router_event["payload"]["router_decision_id"]

    assert agent_turn_started["agent_name"] == "general"
    assert agent_turn_started["payload"]["requested_agent_override"] == "general"
    assert agent_turn_started["payload"]["requested_agent_override_applied"] is True
    assert router_outcome["payload"]["outcome_label"] == "negative"
    assert "manual_agent_override" in router_outcome["payload"]["evidence_signals"]
    assert mispick_sample["payload"]["router_decision_id"] == router_event["payload"]["router_decision_id"]


def test_runtime_trace_callback_records_model_and_tool_events_for_general_agent(tmp_path: Path):
    paths = RuntimePaths(runtime_root=tmp_path / "runtime", workspace_root=tmp_path / "workspaces", memory_root=tmp_path / "memory")
    store = RuntimeTranscriptStore(paths)

    class _Sink:
        def emit(self, event):
            store.append_session_event(event)

    callback = RuntimeTraceCallbackHandler(
        event_sink=_Sink(),
        session_id="tenant:user:general",
        conversation_id="general",
        trace_name="test_general_agent",
        agent_name="general",
        metadata={"route": "AGENT", "router_method": "deterministic"},
    )

    @tool
    def echo_tool(text: str) -> str:
        """Echo the provided text."""
        return f"echo:{text}"

    from agentic_chatbot_next.general_agent import run_general_agent

    llm = FakeListChatModel(
        responses=[
            '{"plan":[{"tool":"echo_tool","args":{"text":"trace me"},"purpose":"demo"}],"notes":"demo"}',
            "Final answer from plan-execute fallback.",
        ]
    )
    final_text, _, _ = run_general_agent(
        llm,
        tools=[echo_tool],
        messages=[],
        user_text="Use the echo tool.",
        system_prompt="Use tools when needed.",
        callbacks=[callback],
        max_tool_calls=2,
    )

    assert "Final answer" in final_text
    events = store.load_session_events("tenant:user:general")
    event_types = [row["event_type"] for row in events]
    assert "model_start" in event_types
    assert "model_end" in event_types
    assert "tool_start" in event_types
    assert "tool_end" in event_types
    tool_event = next(row for row in events if row["event_type"] == "tool_start")
    assert tool_event["tool_name"] == "echo_tool"
    assert tool_event["payload"]["tool_call_id"]
    assert tool_event["payload"]["status"] == "running"
    assert tool_event["payload"]["input_preview"]
    assert tool_event["payload"]["input"]
    completed_event = next(row for row in events if row["event_type"] == "tool_end")
    assert completed_event["payload"]["tool_call_id"] == tool_event["payload"]["tool_call_id"]
    assert completed_event["payload"]["status"] == "completed"
    assert completed_event["payload"]["output_preview"]
    assert completed_event["payload"]["output"]
    assert completed_event["payload"]["duration_ms"] is not None


def test_runtime_trace_callback_redacts_and_truncates_tool_payloads(tmp_path: Path) -> None:
    paths = RuntimePaths(runtime_root=tmp_path / "runtime", workspace_root=tmp_path / "workspaces", memory_root=tmp_path / "memory")
    store = RuntimeTranscriptStore(paths)

    class _Sink:
        def emit(self, event):
            store.append_session_event(event)

    callback = RuntimeTraceCallbackHandler(
        event_sink=_Sink(),
        session_id="tenant:user:redacted",
        conversation_id="redacted",
        trace_name="tool_payload_contract",
        agent_name="general",
    )
    run_id = uuid4()
    callback.on_tool_start(
        {"name": "secret_tool"},
        "",
        run_id=run_id,
        inputs={"query": "hello", "api_key": "sk-live-secret", "nested": {"token": "tok-secret"}},
        metadata={
            "parent_agent": "coordinator",
            "agentic_parallel_group_id": "tool-wave-general-1-call",
            "agentic_parallel_execution_mode": "parallel",
            "agentic_parallel_group_size": 2,
        },
    )
    callback.on_tool_end(
        {"ok": True, "text": "x" * 13_000, "authorization": "Bearer abc"},
        run_id=run_id,
    )

    events = store.load_session_events("tenant:user:redacted")
    start = next(row for row in events if row["event_type"] == "tool_start")
    end = next(row for row in events if row["event_type"] == "tool_end")

    assert start["payload"]["tool_call_id"] == str(run_id)
    assert start["payload"]["input"]["api_key"] == "[redacted]"
    assert start["payload"]["input"]["nested"]["token"] == "[redacted]"
    assert set(start["payload"]["redacted_fields"]) == {"input.api_key", "input.nested.token"}
    assert start["payload"]["payload_limit_chars"] == 12_000
    assert start["payload"]["parent_agent"] == "coordinator"
    assert start["payload"]["parallel_group_id"] == "tool-wave-general-1-call"
    assert start["payload"]["parallel_execution_mode"] == "parallel"
    assert start["payload"]["parallel_group_size"] == 2
    assert "sk-live-secret" not in json.dumps(start["payload"])
    assert end["payload"]["tool_call_id"] == str(run_id)
    assert end["payload"]["output"].startswith("{")
    assert end["payload"]["truncated"] is True
    assert "output" in end["payload"]["truncated_fields"]
    assert "output.authorization" in end["payload"]["redacted_fields"]
    assert "Bearer abc" not in json.dumps(end["payload"])


def test_general_agent_renders_rag_tool_output_when_final_ai_message_is_empty(monkeypatch):
    from agentic_chatbot_next.general_agent import run_general_agent

    class _DummyLLM:
        def bind_tools(self, tools):
            return self

    class _FakeGraph:
        def invoke(self, payload, config=None):
            del payload, config
            return {
                "messages": [
                    AIMessage(content=""),
                    ToolMessage(
                        content=(
                            '{"answer":"Tool-calling reliability improved through stronger retries.",'
                            '"citations":[{"citation_id":"KB_1#chunk0001","doc_id":"KB_1","title":"05_release_notes.md","source_type":"kb","location":"page None","snippet":"retry improvements"}],'
                            '"used_citation_ids":["KB_1#chunk0001"],'
                            '"retrieval_summary":{"query_used":"tool reliability","steps":3,"tool_calls_used":0,"tool_call_log":[],"citations_found":1},'
                            '"followups":[],"warnings":[]}'
                        ),
                        tool_call_id="tool_1",
                    ),
                    AIMessage(content=""),
                ]
            }

    monkeypatch.setattr(
        "langgraph.prebuilt.create_react_agent",
        lambda chat_llm, tools=None: _FakeGraph(),
    )

    final_text, updated_messages, _ = run_general_agent(
        _DummyLLM(),
        tools=[],
        messages=[],
        user_text="Explain the release-note changes with citations.",
        system_prompt="Use tools when needed.",
    )

    assert "Tool-calling reliability improved" in final_text
    assert "Citations:" in final_text
    assert isinstance(updated_messages[-1], AIMessage)
    assert "Citations:" in str(updated_messages[-1].content)


def test_general_agent_builds_policy_aware_tool_node(monkeypatch):
    from agentic_chatbot_next.general_agent import run_general_agent

    captured = {}

    class _DummyLLM:
        def bind_tools(self, tools):
            return self

    class _FakeGraph:
        def invoke(self, payload, config=None):
            del payload, config
            return {"messages": [AIMessage(content="Final answer from graph.")]}

    def fake_create_react_agent(chat_llm, tools=None):
        del chat_llm
        captured["tools"] = tools
        return _FakeGraph()

    monkeypatch.setattr("langgraph.prebuilt.create_react_agent", fake_create_react_agent)

    final_text, _, _ = run_general_agent(
        _DummyLLM(),
        tools=[],
        messages=[],
        user_text="Answer directly.",
        system_prompt="Use tools when needed.",
        max_tool_calls=7,
        max_parallel_tool_calls=3,
    )

    assert final_text == "Final answer from graph."
    assert isinstance(captured["tools"], PolicyAwareToolNode)
    assert captured["tools"].max_tool_calls == 7
    assert captured["tools"].max_parallel_tool_calls == 3


def test_query_loop_forces_plan_execute_for_data_analyst_strategy(monkeypatch):
    captured = {}

    def fake_run_general_agent(*args, **kwargs):
        captured["force_plan_execute"] = kwargs.get("force_plan_execute")
        captured["max_parallel_tool_calls"] = kwargs.get("max_parallel_tool_calls")
        return "done", [], {"steps": 1, "tool_calls": 0}

    monkeypatch.setattr("agentic_chatbot_next.runtime.query_loop.run_general_agent", fake_run_general_agent)

    loop = QueryLoop(
        settings=SimpleNamespace(max_parallel_tool_calls=6),
        providers=SimpleNamespace(chat=object()),
        stores=SimpleNamespace(),
    )
    agent = AgentDefinition(
        name="data_analyst",
        mode="react",
        prompt_file="data_analyst_agent.md",
        metadata={"execution_strategy": "plan_execute"},
    )
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    session.append_message("user", "Analyze the uploaded CSV files.")
    tool_context = SimpleNamespace(callbacks=[], refresh_from_session_handle=lambda: None)

    result = loop.run(
        agent,
        session,
        user_text="Analyze the uploaded CSV files.",
        tool_context=tool_context,
        tools=[],
    )

    assert result.text == "done"
    assert captured["force_plan_execute"] is True
    assert captured["max_parallel_tool_calls"] == 6


def test_runtime_trace_callback_handles_parallel_tool_lifecycle_events(tmp_path: Path) -> None:
    paths = RuntimePaths(runtime_root=tmp_path / "runtime", workspace_root=tmp_path / "workspaces", memory_root=tmp_path / "memory")
    store = RuntimeTranscriptStore(paths)

    class _Sink:
        def emit(self, event):
            store.append_session_event(event)

    callback = RuntimeTraceCallbackHandler(
        event_sink=_Sink(),
        session_id="tenant:user:parallel",
        conversation_id="parallel",
        trace_name="parallel_tool_trace",
        agent_name="general",
    )

    def worker(index: int) -> None:
        run_id = uuid4()
        callback.on_tool_start(
            {"name": f"tool_{index}"},
            '{"value":"demo"}',
            run_id=run_id,
            inputs={"value": "demo"},
        )
        callback.on_tool_end(
            {"ok": True, "index": index},
            run_id=run_id,
        )

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    events = store.load_session_events("tenant:user:parallel")
    assert sum(1 for row in events if row["event_type"] == "tool_start") == 12
    assert sum(1 for row in events if row["event_type"] == "tool_end") == 12
    assert callback._tool_runs == {}


def test_progress_callback_handles_parallel_tool_events() -> None:
    callback = ProgressCallback(LiveProgressSink())

    def worker(index: int) -> None:
        run_id = uuid4()
        callback.on_tool_start({"name": f"tool_{index}"}, '{"query":"demo"}', run_id=run_id)
        callback.on_tool_end({"ok": True, "index": index}, run_id=run_id)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    queued_events = []
    while not callback.events.empty():
        queued_events.append(callback.events.get())

    assert sum(1 for event in queued_events if event.get("type") == "tool_call") == 10
    assert sum(1 for event in queued_events if event.get("type") == "tool_result") == 10
    assert callback._start_times == {}
    assert callback._active_tool_names == {}


def test_general_agent_repairs_non_json_plan_output_before_falling_back(monkeypatch):
    from agentic_chatbot_next.general_agent import run_general_agent

    @tool
    def echo_tool(text: str) -> str:
        """Echo the provided text."""
        return f"echo:{text}"

    class _DummyLLM:
        def bind_tools(self, tools):
            return self

        def __init__(self):
            self.calls = 0

        def invoke(self, messages, config=None):
            del messages, config
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="Use echo_tool with text=trace me.")
            if self.calls == 2:
                return AIMessage(
                    content='{"plan":[{"tool":"echo_tool","args":{"text":"trace me"},"purpose":"demo"}],"notes":"repaired"}'
                )
            return AIMessage(content="Final answer after repaired plan.")

    llm = _DummyLLM()
    final_text, _, metadata = run_general_agent(
        llm,
        tools=[echo_tool],
        messages=[],
        user_text="Use the echo tool.",
        system_prompt="Use tools when needed.",
        force_plan_execute=True,
    )

    assert "Final answer" in final_text
    assert metadata["fallback"] == "plan_execute"


def test_general_agent_sanitizes_null_tool_args_before_invocation():
    from agentic_chatbot_next.general_agent import run_general_agent

    @tool
    def inspect_like_tool(doc_id: str = "", columns: str = "") -> str:
        """Return the received arguments."""
        return f"{doc_id}|{columns}"

    llm = FakeListChatModel(
        responses=[
            '{"plan":[{"tool":"inspect_like_tool","args":{"doc_id":null,"columns":null},"purpose":"demo"}],"notes":"demo"}',
            "Final answer after sanitized tool args.",
        ]
    )

    final_text, messages, metadata = run_general_agent(
        llm,
        tools=[inspect_like_tool],
        messages=[],
        user_text="Inspect the uploaded dataset.",
        system_prompt="Use tools when needed.",
        force_plan_execute=True,
    )

    assert "Final answer" in final_text
    assert metadata["fallback"] == "plan_execute"
    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    assert tool_messages
    assert tool_messages[-1].content == "|"


def test_general_agent_plan_execute_retries_named_graph_request_with_search() -> None:
    from agentic_chatbot_next.general_agent import run_general_agent

    calls: list[tuple[str, dict[str, object]]] = []

    @tool("list_graph_indexes")
    def list_graph_indexes(collection_id: str = "", limit: int = 20) -> str:
        """List graph indexes."""
        calls.append(("list_graph_indexes", {"collection_id": collection_id, "limit": limit}))
        return json.dumps(
            {
                "graphs": [
                    {
                        "graph_id": "defense_rag_v2_graph",
                        "status": "ready",
                        "query_ready": True,
                    }
                ]
            }
        )

    @tool("search_graph_index")
    def search_graph_index(query: str, graph_id: str = "", limit: int = 8) -> str:
        """Search graph evidence."""
        calls.append(("search_graph_index", {"query": query, "graph_id": graph_id, "limit": limit}))
        return json.dumps(
            {
                "graph_id": graph_id,
                "results": [
                    {
                        "title": "Asterion ISR Payload Modernization Program Overview",
                        "summary": "North Coast Systems and Mesa Micro Optics are supplier dependencies tied to moderate schedule risk.",
                        "chunk_ids": ["chunk-1"],
                    }
                ],
            }
        )

    llm = FakeListChatModel(
        responses=[
            '{"plan":[{"tool":"list_graph_indexes","args":{},"purpose":"discover graph"}],"notes":"catalog first"}',
            "Graph evidence answer citing Asterion ISR Payload Modernization Program Overview.",
        ]
    )

    final_text, messages, metadata = run_general_agent(
        llm,
        tools=[list_graph_indexes, search_graph_index],
        messages=[],
        user_text=(
            "Use the knowledge graph defense_rag_v2_graph to find cross-document relationships "
            "between vendors, risks, approvals, dependencies, and program outcomes."
        ),
        system_prompt="Use graph tools for graph evidence.",
        force_plan_execute=True,
    )

    assert "Graph evidence answer" in final_text
    assert [name for name, _ in calls] == ["list_graph_indexes", "search_graph_index"]
    assert calls[-1][1]["graph_id"] == "defense_rag_v2_graph"
    assert metadata["tool_calls"] == 2
    assert [message.name for message in messages if isinstance(message, ToolMessage)] == [
        "list_graph_indexes",
        "search_graph_index",
    ]


def test_general_agent_plan_execute_does_not_finalize_graph_relationships_from_catalog_only() -> None:
    from agentic_chatbot_next.general_agent import run_general_agent

    @tool("list_graph_indexes")
    def list_graph_indexes() -> str:
        """List graph indexes."""
        return json.dumps(
            {
                "graphs": [
                    {"graph_id": "alpha_graph", "status": "ready"},
                    {"graph_id": "beta_graph", "status": "ready"},
                ]
            }
        )

    @tool("search_graph_index")
    def search_graph_index(query: str, graph_id: str = "") -> str:
        """Search graph evidence."""
        return json.dumps({"query": query, "graph_id": graph_id, "results": []})

    llm = FakeListChatModel(
        responses=[
            '{"plan":[{"tool":"list_graph_indexes","args":{},"purpose":"discover graph"}],"notes":"catalog only"}',
            "This response should not be used.",
        ]
    )

    final_text, messages, metadata = run_general_agent(
        llm,
        tools=[list_graph_indexes, search_graph_index],
        messages=[],
        user_text="Use a knowledge graph to find relationships between vendors and risks.",
        system_prompt="Use graph tools for graph evidence.",
        force_plan_execute=True,
    )

    assert "do not have GraphRAG search evidence" in final_text
    assert "should not infer" in final_text
    assert metadata["recovery"] == ["graph_search_missing"]
    assert [message.name for message in messages if isinstance(message, ToolMessage)] == ["list_graph_indexes"]


def test_graph_h06_catalog_candidates_recover_to_grounded_rag_answer() -> None:
    from agentic_chatbot_next.general_agent import run_general_agent

    @tool("search_graph_index")
    def search_graph_index(query: str, graph_id: str = "", methods_csv: str = "", limit: int = 8) -> str:
        """Search graph evidence."""
        del query, methods_csv, limit
        return json.dumps(
            {
                "graph_id": graph_id,
                "evidence_status": "source_candidates_only",
                "requires_source_read": True,
                "results": [
                    {
                        "backend": "catalog",
                        "doc_id": "HOST_PATH_7f8a93a2d7",
                        "title": "blue_mica_crypto_compliance_summary_final.pdf",
                        "summary": "Catalog source candidate from managed graph.",
                        "chunk_ids": [],
                        "relationship_path": [],
                        "metadata": {"fallback": "catalog", "catalog_only": True, "evidence_kind": "source_candidate"},
                        "citation_ids": ["HOST_PATH_7f8a93a2d7#graph"],
                    },
                    {
                        "backend": "catalog",
                        "doc_id": "HOST_PATH_8889337d84",
                        "title": "blue_mica_after_action_notes.txt",
                        "summary": "Catalog source candidate from managed graph.",
                        "chunk_ids": [],
                        "relationship_path": [],
                        "metadata": {"fallback": "catalog", "catalog_only": True, "evidence_kind": "source_candidate"},
                        "citation_ids": ["HOST_PATH_8889337d84#graph"],
                    },
                ],
            }
        )

    @tool("rag_agent_tool")
    def rag_agent_tool(
        query: str,
        conversation_context: str = "",
        preferred_doc_ids_csv: str = "",
        must_include_uploads: bool = False,
        top_k_vector: int = 16,
        top_k_keyword: int = 16,
        max_retries: int = 2,
        search_mode: str = "deep",
        max_search_rounds: int = 2,
    ) -> dict:
        """Run grounded staged retrieval."""
        del must_include_uploads, top_k_vector, top_k_keyword, max_retries, search_mode, max_search_rounds
        assert "HOST_PATH_7f8a93a2d7" in preferred_doc_ids_csv
        assert "certificate/binder" not in conversation_context
        assert "serialized-assignment" not in conversation_context
        assert "training/readiness" not in conversation_context
        assert "06 Feb 2029" not in conversation_context
        assert "claim-focused" in conversation_context
        return {
            "answer": (
                "The better answer is that the main issue was not a demonstrated hardware failure. "
                "ENC-21M hardware was technically acceptable, but certificate/binder timing, "
                "serialized-assignment accuracy, and training/readiness problems drove the move to 06 Feb 2029."
            ),
            "citations": [
                {
                    "citation_id": "bm-compliance",
                    "doc_id": "HOST_PATH_7f8a93a2d7",
                    "title": "blue_mica_crypto_compliance_summary_final.pdf",
                    "source_type": "kb",
                    "location": "Compliance Sections 1-3",
                    "snippet": (
                        "The issue was not a demonstrated hardware failure. ENC-21M hardware was technically acceptable; "
                        "certificate/binder timing, serialized-assignment accuracy, and training/readiness drove the move to 06 Feb 2029."
                    ),
                }
            ],
            "used_citation_ids": ["bm-compliance"],
            "confidence": 0.86,
            "retrieval_summary": {"query_used": query, "search_mode": "deep"},
            "warnings": [],
            "followups": [],
        }

    llm = FakeListChatModel(
        responses=[
            (
                '{"plan":[{"tool":"search_graph_index","args":{"graph_id":"defense_rag_v2_graph",'
                '"query":"Blue Mica Wave 2 slipped hardware bad","methods_csv":"graph"},'
                '"purpose":"find graph source candidates"}],"notes":"graph first"}'
            ),
            "This response should not be used.",
        ]
    )

    final_text, messages, metadata = run_general_agent(
        llm,
        tools=[search_graph_index, rag_agent_tool],
        messages=[],
        user_text=(
            "Use the knowledge graph defense_rag_v2_graph answer the following question: "
            "If someone says Blue Mica Wave 2 slipped because the hardware was bad, "
            "what is the better evidence-based answer?"
        ),
        system_prompt="Use graph tools, but do not answer from catalog-only graph matches.",
        force_plan_execute=True,
    )

    assert "not a demonstrated hardware failure" in final_text
    assert "ENC-21M hardware was technically acceptable" in final_text
    assert "certificate/binder timing" in final_text
    assert "serialized-assignment accuracy" in final_text
    assert "training/readiness" in final_text
    assert "06 Feb 2029" in final_text
    assert metadata["recovery"] == ["graph_catalog_to_rag_recovery"]
    assert [message.name for message in messages if isinstance(message, ToolMessage)] == [
        "search_graph_index",
        "rag_agent_tool",
    ]


def test_graph_final_verifier_rejects_generic_unsupported_causal_claims() -> None:
    from agentic_chatbot_next.general_agent import run_general_agent

    @tool("search_graph_index")
    def search_graph_index(query: str, graph_id: str = "", methods_csv: str = "", limit: int = 8) -> str:
        """Search graph evidence."""
        del query, methods_csv, limit
        return json.dumps(
            {
                "graph_id": graph_id,
                "evidence_status": "grounded_graph_evidence",
                "requires_source_read": False,
                "results": [
                    {
                        "backend": "graphrag",
                        "doc_id": "DOC-portal-readiness",
                        "title": "portal_release_readiness.md",
                        "summary": (
                            "Aurora Portal moved because vendor approval queue timing and load-test readiness "
                            "work were not complete by the earlier date."
                        ),
                        "chunk_ids": ["chunk-1"],
                        "relationship_path": ["Aurora Portal", "release movement", "readiness"],
                        "metadata": {"evidence_kind": "graph_chunk"},
                    }
                ],
            }
        )

    llm = FakeListChatModel(
        responses=[
            (
                '{"plan":[{"tool":"search_graph_index","args":{"graph_id":"operations_graph",'
                '"query":"Aurora Portal slipped because payment API was faulty","methods_csv":"local,global"},'
                '"purpose":"find graph evidence"}],"notes":"graph first"}'
            ),
            "Aurora Portal slipped because of thermal actuator resonance in orbital firmware.",
        ]
    )

    final_text, _messages, metadata = run_general_agent(
        llm,
        tools=[search_graph_index],
        messages=[],
        user_text=(
            "Use the knowledge graph operations_graph answer the following question: "
            "If someone says Aurora Portal slipped because the payment API was faulty, "
            "what is the better evidence-based answer?"
        ),
        system_prompt="Use graph tools and cited evidence only.",
        force_plan_execute=True,
    )

    assert "do not have cited evidence" in final_text
    assert "thermal" in final_text
    assert "orbital" in final_text
    assert "should not present them as the answer" in final_text
    assert metadata["recovery"] == ["unsupported_causal_claims_rejected"]


def test_general_agent_react_missing_graph_search_retries_with_search() -> None:
    from agentic_chatbot_next.general_agent import run_general_agent

    calls: list[dict[str, object]] = []

    @tool("search_graph_index")
    def search_graph_index(query: str, graph_id: str = "", limit: int = 8) -> str:
        """Search graph evidence."""
        calls.append({"query": query, "graph_id": graph_id, "limit": limit})
        return json.dumps(
            {
                "graph_id": graph_id,
                "results": [{"title": "Supplier Risk Evidence", "summary": "Vendor risk relationship found."}],
            }
        )

    class _ToolBindingFakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            del tools, tool_choice, kwargs
            return self

    model = _ToolBindingFakeModel(
        responses=[
            AIMessage(content="I do not have direct access to that graph."),
            AIMessage(
                content=(
                    '{"plan":[{"tool":"search_graph_index","args":{"graph_id":"defense_rag_v2_graph"},'
                    '"purpose":"search graph evidence"}],"notes":"search"}'
                )
            ),
            AIMessage(content="Graph evidence answer from Supplier Risk Evidence."),
        ]
    )

    final_text, _messages, metadata = run_general_agent(
        model,
        tools=[search_graph_index],
        messages=[],
        user_text=(
            "Use the knowledge graph defense_rag_v2_graph to find cross-document relationships "
            "between vendors, risks, approvals, dependencies, and program outcomes."
        ),
        system_prompt="Use graph tools for graph evidence.",
    )

    assert "Graph evidence answer" in final_text
    assert calls and calls[0]["graph_id"] == "defense_rag_v2_graph"
    assert metadata["recovery"][0] == "graph_search_missing_react"


def test_general_agent_plan_execute_shows_tool_schemas_to_planner() -> None:
    from agentic_chatbot_next.general_agent import run_general_agent

    captured: dict[str, str] = {}

    class _SchemaCapturingLLM:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages, config=None):
            del config
            self.calls += 1
            if self.calls == 1:
                captured["planner_system"] = str(messages[0].content)
                return AIMessage(
                    content=(
                        '{"plan":[{"tool":"search_graph_index","args":{"graph_id":"defense_rag_v2_graph"},'
                        '"purpose":"search graph"}],"notes":"search"}'
                    )
                )
            return AIMessage(content="Graph evidence answer.")

    @tool("search_graph_index")
    def search_graph_index(query: str, graph_id: str = "") -> str:
        """Search graph evidence."""
        return json.dumps({"query": query, "graph_id": graph_id, "results": [{"title": "Evidence", "summary": "Found."}]})

    run_general_agent(
        _SchemaCapturingLLM(),
        tools=[search_graph_index],
        messages=[],
        user_text="Use the knowledge graph defense_rag_v2_graph to find supplier relationships.",
        system_prompt="Use graph tools for graph evidence.",
        force_plan_execute=True,
    )

    assert "Tool schemas:" in captured["planner_system"]
    assert "search_graph_index" in captured["planner_system"]
    assert "graph_id" in captured["planner_system"]
    assert "Use inspect_graph_index with graph_id, never collection_id." in captured["planner_system"]


def test_data_analyst_plan_execute_falls_back_to_guided_flow_when_execute_code_fails(monkeypatch):
    from agentic_chatbot_next.general_agent import run_general_agent

    @tool
    def workspace_list() -> str:
        """List files in the workspace."""
        return '{"files":["regional_spend.csv","regional_controls.csv"]}'

    @tool
    def load_dataset(doc_id: str = "") -> str:
        """Load a dataset."""
        return '{"doc_id":"regional_spend.csv","columns":["region","annual_spend_usd","current_reserve_usd"]}'

    @tool
    def inspect_columns(doc_id: str = "", columns: str = "") -> str:
        """Inspect dataset columns."""
        return '{"region":{"dtype":"object"}}'

    @tool
    def execute_code(code: str, doc_ids: str = "") -> str:
        """Execute analysis code."""
        del code, doc_ids
        return '{"success": false, "stdout": "", "stderr": "sandbox failed"}'

    captured = {}

    def fake_guided_fallback(**kwargs):
        captured["called"] = True
        return "guided fallback", [], {"fallback": "data_analyst_guided"}

    monkeypatch.setattr(
        "agentic_chatbot_next.general_agent._run_data_analyst_guided_fallback",
        fake_guided_fallback,
    )

    llm = FakeListChatModel(
        responses=[
            '{"plan":[{"tool":"load_dataset","args":{},"purpose":"load"},'
            '{"tool":"inspect_columns","args":{"doc_id":null,"columns":null},"purpose":"inspect"},'
            '{"tool":"execute_code","args":{"code":"print(1)","doc_ids":null},"purpose":"run"}],'
            '"notes":"demo"}'
        ]
    )

    final_text, _, metadata = run_general_agent(
        llm,
        tools=[workspace_list, load_dataset, inspect_columns, execute_code],
        messages=[],
        user_text="Analyze the uploaded CSV files.",
        system_prompt="Use tools when needed.",
        force_plan_execute=True,
    )

    assert captured["called"] is True
    assert final_text == "guided fallback"
    assert metadata["fallback"] == "data_analyst_guided"


def test_data_analyst_plan_execute_treats_successful_nlp_tool_as_complete(monkeypatch):
    from agentic_chatbot_next.general_agent import run_general_agent

    @tool
    def workspace_list() -> str:
        """List files in the workspace."""
        return '{"files":["customer_reviews_100.csv","sales_performance.xlsx"]}'

    @tool
    def load_dataset(doc_id: str = "") -> str:
        """Load a dataset."""
        del doc_id
        return '{"doc_id":"customer_reviews_100.csv","columns":["reviews"],"dtypes":{"reviews":"object"}}'

    @tool
    def inspect_columns(doc_id: str = "", columns: str = "") -> str:
        """Inspect dataset columns."""
        del doc_id, columns
        return '{"reviews":{"dtype":"object"}}'

    @tool
    def execute_code(code: str, doc_ids: str = "") -> str:
        """Execute analysis code."""
        raise AssertionError("execute_code should not run for bounded NLP requests")

    @tool
    def run_nlp_column_task(
        doc_id: str = "",
        dataset: str = "",
        column: str = "",
        task: str = "",
        output_mode: str = "summary_only",
    ) -> str:
        """Run bounded NLP over one text column."""
        assert (doc_id or dataset) == "customer_reviews_100.csv"
        assert column == "reviews"
        assert task == "sentiment_analysis"
        assert output_mode == "append_columns"
        return (
            '{"task":"sentiment","doc_id":"customer_reviews_100.csv","column":"reviews",'
            '"processed_rows":3,"result_counts":{"positive":2,"negative":1},'
            '"written_file":"customer_reviews_100__analyst_sentiment.csv",'
            '"preview_columns":["row_index","reviews","sentiment_label","sentiment_score"],'
            '"preview_rows":[{"row_index":0,"reviews":"Great support","sentiment_label":"positive","sentiment_score":0.98}],'
            '"summary_text":"Processed 3 rows from \'reviews\' in customer_reviews_100.csv. Sentiment counts: negative: 1, positive: 2."}'
        )

    published: dict[str, str] = {}

    @tool
    def return_file(filename: str = "") -> str:
        """Publish a generated file."""
        published["filename"] = filename
        return '{"filename":"customer_reviews_100__analyst_sentiment.csv","download_url":"/v1/files/dl_123"}'

    monkeypatch.setattr(
        "agentic_chatbot_next.general_agent._run_data_analyst_guided_fallback",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("guided fallback should not run after successful NLP output")),
    )

    llm = FakeListChatModel(
        responses=[
            '{"plan":[{"tool":"run_nlp_column_task","args":{"dataset":"customer_reviews_100.csv","column":"reviews","task":"sentiment_analysis","output_mode":"summary_only"},"purpose":"label sentiment"}],"notes":"demo"}',
            "",
        ]
    )

    final_text, _, metadata = run_general_agent(
        llm,
        tools=[workspace_list, load_dataset, inspect_columns, execute_code, run_nlp_column_task, return_file],
        messages=[],
        user_text="Provide sentiment analysis of all of the reviews in the reviews column.",
        system_prompt="Use tools when needed.",
        force_plan_execute=True,
    )

    assert "Processed 3 rows" in final_text
    assert "Preview:" in final_text
    assert "sentiment_label" in final_text
    assert published["filename"] == "customer_reviews_100__analyst_sentiment.csv"
    assert metadata["fallback"] == "plan_execute"
    assert "render_data_analyst_tool_fallback" in metadata["recovery"]


def test_data_analyst_plan_execute_rewrites_placeholder_dataset_args_from_loaded_schema(monkeypatch):
    from agentic_chatbot_next.general_agent import run_general_agent

    @tool
    def workspace_list() -> str:
        """List files in the workspace."""
        return '{"files":["customer_reviews_100.csv","sales_performance.xlsx"]}'

    @tool
    def load_dataset(doc_id: str = "") -> str:
        """Load a dataset."""
        if doc_id == "customer_reviews_100.csv":
            return '{"doc_id":"customer_reviews_100.csv","columns":["reviews"],"dtypes":{"reviews":"object"}}'
        return '{"doc_id":"sales_performance.xlsx","columns":["region","revenue_usd"],"dtypes":{"region":"object","revenue_usd":"float64"}}'

    @tool
    def inspect_columns(doc_id: str = "", columns: str = "") -> str:
        """Inspect dataset columns."""
        del doc_id, columns
        return '{"reviews":{"dtype":"object"}}'

    @tool
    def execute_code(code: str, doc_ids: str = "") -> str:
        """Execute analysis code."""
        del code, doc_ids
        raise AssertionError("execute_code should not run for bounded NLP requests")

    @tool
    def run_nlp_column_task(doc_id: str = "", column: str = "", task: str = "", output_mode: str = "summary_only") -> str:
        """Run bounded NLP over one text column."""
        assert doc_id == "customer_reviews_100.csv"
        assert column == "reviews"
        assert output_mode == "append_columns"
        return (
            '{"task":"sentiment","doc_id":"customer_reviews_100.csv","column":"reviews",'
            '"processed_rows":3,"result_counts":{"positive":2,"negative":1},'
            '"written_file":"customer_reviews_100__analyst_sentiment.csv",'
            '"preview_columns":["row_index","reviews","sentiment_label","sentiment_score"],'
            '"preview_rows":[{"row_index":0,"reviews":"Great support","sentiment_label":"positive","sentiment_score":0.98}],'
            '"summary_text":"Processed 3 rows from \'reviews\' in customer_reviews_100.csv. Sentiment counts: negative: 1, positive: 2."}'
        )

    @tool
    def return_file(filename: str = "") -> str:
        """Publish a generated file."""
        assert filename == "customer_reviews_100__analyst_sentiment.csv"
        return '{"filename":"customer_reviews_100__analyst_sentiment.csv","download_url":"/v1/files/dl_123"}'

    monkeypatch.setattr(
        "agentic_chatbot_next.general_agent._run_data_analyst_guided_fallback",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("guided fallback should not run after normalized NLP args succeed")),
    )

    llm = FakeListChatModel(
        responses=[
            '{"plan":[{"tool":"workspace_list","args":{},"purpose":"list files"},'
            '{"tool":"load_dataset","args":{"doc_id":"customer_reviews_100.csv"},"purpose":"inspect schema"},'
            '{"tool":"run_nlp_column_task","args":{"doc_id":"df","column":"text","task":"sentiment_analysis","output_mode":"summary_only"},"purpose":"label sentiment"}],"notes":"demo"}'
        ]
    )

    final_text, _, metadata = run_general_agent(
        llm,
        tools=[workspace_list, load_dataset, inspect_columns, execute_code, run_nlp_column_task, return_file],
        messages=[],
        user_text="Provide sentiment analysis of all of the reviews in the reviews column.",
        system_prompt="Use tools when needed.",
        force_plan_execute=True,
    )

    assert "Processed 3 rows" in final_text
    assert "Preview:" in final_text
    assert metadata["fallback"] == "plan_execute"


def test_data_analyst_guided_fallback_prefers_nlp_for_sentiment_requests(monkeypatch):
    from agentic_chatbot_next.general_agent import run_general_agent

    @tool
    def workspace_list() -> str:
        """List files in the workspace."""
        return '{"files":["customer_reviews_100.csv","sales_performance.xlsx"]}'

    @tool
    def load_dataset(doc_id: str = "") -> str:
        """Load a dataset."""
        if doc_id == "customer_reviews_100.csv":
            return '{"doc_id":"customer_reviews_100.csv","columns":["review_id","reviews"],"dtypes":{"review_id":"int64","reviews":"object"}}'
        return '{"doc_id":"sales_performance.xlsx","columns":["region","revenue_usd"],"dtypes":{"region":"object","revenue_usd":"float64"}}'

    inspected: list[str] = []

    @tool
    def inspect_columns(doc_id: str = "", columns: str = "") -> str:
        """Inspect dataset columns."""
        del columns
        inspected.append(doc_id)
        if doc_id == "customer_reviews_100.csv":
            return '{"reviews":{"dtype":"object"},"_meta":{"doc_id":"customer_reviews_100.csv"}}'
        raise AssertionError("Only the selected dataset should be inspected for row-level NLP fallback")

    @tool
    def execute_code(code: str, doc_ids: str = "") -> str:
        """Execute analysis code."""
        raise AssertionError("execute_code should not run for guided NLP fallback")

    @tool
    def run_nlp_column_task(
        doc_id: str = "",
        column: str = "",
        task: str = "",
        output_mode: str = "summary_only",
    ) -> str:
        """Run bounded NLP over one text column."""
        assert doc_id == "customer_reviews_100.csv"
        assert column == "reviews"
        assert task == "sentiment"
        assert output_mode == "append_columns"
        return (
            '{"task":"sentiment","doc_id":"customer_reviews_100.csv","column":"reviews",'
            '"processed_rows":4,"result_counts":{"positive":3,"neutral":1},'
            '"written_file":"customer_reviews_100__analyst_sentiment.csv",'
            '"preview_columns":["row_index","reviews","sentiment_label","sentiment_score"],'
            '"preview_rows":[{"row_index":0,"reviews":"Great support","sentiment_label":"positive","sentiment_score":0.98}],'
            '"summary_text":"Processed 4 rows from \'reviews\' in customer_reviews_100.csv. Sentiment counts: neutral: 1, positive: 3."}'
        )

    published: dict[str, str] = {}

    @tool
    def return_file(filename: str = "") -> str:
        """Publish a generated file."""
        published["filename"] = filename
        return '{"filename":"customer_reviews_100__analyst_sentiment.csv","download_url":"/v1/files/dl_123"}'

    llm = FakeListChatModel(responses=["not json", "still not json"])

    final_text, _, metadata = run_general_agent(
        llm,
        tools=[workspace_list, load_dataset, inspect_columns, execute_code, run_nlp_column_task, return_file],
        messages=[],
        user_text="Provide sentiment analysis of all of the reviews in the reviews column.",
        system_prompt="Use tools when needed.",
        force_plan_execute=True,
    )

    assert "Processed 4 rows" in final_text
    assert "Preview:" in final_text
    assert "sentiment_score" in final_text
    assert inspected == ["customer_reviews_100.csv"]
    assert published["filename"] == "customer_reviews_100__analyst_sentiment.csv"
    assert metadata["fallback"] == "data_analyst_guided"
    assert metadata["guided_mode"] == "nlp"


def test_data_analyst_guided_fallback_keeps_distribution_prompts_summary_only():
    from agentic_chatbot_next.general_agent import run_general_agent

    @tool
    def workspace_list() -> str:
        """List files in the workspace."""
        return '{"files":["customer_reviews_100.csv"]}'

    @tool
    def load_dataset(doc_id: str = "") -> str:
        """Load a dataset."""
        del doc_id
        return '{"doc_id":"customer_reviews_100.csv","columns":["reviews"],"dtypes":{"reviews":"object"}}'

    @tool
    def inspect_columns(doc_id: str = "", columns: str = "") -> str:
        """Inspect dataset columns."""
        del doc_id, columns
        return '{"reviews":{"dtype":"object"}}'

    @tool
    def execute_code(code: str, doc_ids: str = "") -> str:
        """Execute analysis code."""
        raise AssertionError("execute_code should not run for summary-only NLP prompts")

    @tool
    def run_nlp_column_task(
        doc_id: str = "",
        column: str = "",
        task: str = "",
        output_mode: str = "summary_only",
    ) -> str:
        """Run bounded NLP over one text column."""
        assert doc_id == "customer_reviews_100.csv"
        assert column == "reviews"
        assert task == "sentiment"
        assert output_mode == "summary_only"
        return (
            '{"task":"sentiment","doc_id":"customer_reviews_100.csv","column":"reviews",'
            '"processed_rows":4,"result_counts":{"positive":3,"neutral":1},'
            '"preview_columns":["row_index","reviews","sentiment_label","sentiment_score"],'
            '"preview_rows":[{"row_index":0,"reviews":"Great support","sentiment_label":"positive","sentiment_score":0.98}],'
            '"summary_text":"Processed 4 rows from \'reviews\' in customer_reviews_100.csv. Sentiment counts: neutral: 1, positive: 3."}'
        )

    @tool
    def return_file(filename: str = "") -> str:
        """Publish a generated file."""
        del filename
        raise AssertionError("return_file should not run for summary-only analyst prompts")

    llm = FakeListChatModel(responses=["not json", "still not json"])

    final_text, _, metadata = run_general_agent(
        llm,
        tools=[workspace_list, load_dataset, inspect_columns, execute_code, run_nlp_column_task, return_file],
        messages=[],
        user_text="Summarize the sentiment distribution in the reviews column.",
        system_prompt="Use tools when needed.",
        force_plan_execute=True,
    )

    assert "Processed 4 rows" in final_text
    assert "Preview:" in final_text
    assert metadata["fallback"] == "data_analyst_guided"
    assert metadata["guided_mode"] == "nlp"


def test_data_analyst_guided_fallback_reports_ambiguous_row_level_targets():
    from agentic_chatbot_next.general_agent import run_general_agent

    @tool
    def workspace_list() -> str:
        """List files in the workspace."""
        return '{"files":["customer_reviews_100.csv","support_reviews.csv"]}'

    @tool
    def load_dataset(doc_id: str = "") -> str:
        """Load a dataset."""
        return json.dumps({"doc_id": doc_id, "columns": ["reviews"], "dtypes": {"reviews": "object"}})

    @tool
    def inspect_columns(doc_id: str = "", columns: str = "") -> str:
        """Inspect dataset columns."""
        del doc_id, columns
        raise AssertionError("inspect_columns should not run when the target dataset is ambiguous")

    @tool
    def execute_code(code: str, doc_ids: str = "") -> str:
        """Execute analysis code."""
        del code, doc_ids
        raise AssertionError("execute_code should not run for ambiguous row-level NLP prompts")

    @tool
    def run_nlp_column_task(doc_id: str = "", column: str = "", task: str = "", output_mode: str = "summary_only") -> str:
        """Run bounded NLP over one text column."""
        del doc_id, column, task, output_mode
        raise AssertionError("run_nlp_column_task should not run when the target dataset is ambiguous")

    llm = FakeListChatModel(responses=["not json", "still not json"])

    final_text, _, metadata = run_general_agent(
        llm,
        tools=[workspace_list, load_dataset, inspect_columns, execute_code, run_nlp_column_task],
        messages=[],
        user_text="Provide sentiment analysis of all of the reviews in the reviews column.",
        system_prompt="Use tools when needed.",
        force_plan_execute=True,
    )

    assert "Multiple uploaded datasets look equally valid" in final_text
    assert "customer_reviews_100.csv" in final_text
    assert "support_reviews.csv" in final_text
    assert metadata["fallback"] == "data_analyst_guided"
    assert metadata["guided_mode"] == "nlp"


def test_data_analyst_guided_fallback_reports_docker_unavailable_for_code_requests():
    from agentic_chatbot_next.general_agent import run_general_agent

    @tool
    def workspace_list() -> str:
        """List files in the workspace."""
        return '{"files":["sales_performance.xlsx"]}'

    @tool
    def load_dataset(doc_id: str = "") -> str:
        """Load a dataset."""
        del doc_id
        return '{"doc_id":"sales_performance.xlsx","columns":["region","revenue_usd"],"dtypes":{"region":"object","revenue_usd":"float64"}}'

    @tool
    def inspect_columns(doc_id: str = "", columns: str = "") -> str:
        """Inspect dataset columns."""
        del doc_id, columns
        return '{"region":{"dtype":"object"},"revenue_usd":{"dtype":"float64"}}'

    @tool
    def execute_code(code: str, doc_ids: str = "") -> str:
        """Execute analysis code."""
        del code, doc_ids
        return (
            '{"error":"Docker sandbox is not available: Docker is not available: missing daemon","success":false,"stdout":"","stderr":""}'
        )

    @tool
    def run_nlp_column_task(doc_id: str = "", column: str = "", task: str = "", output_mode: str = "summary_only") -> str:
        """Run bounded NLP over one text column."""
        raise AssertionError("run_nlp_column_task should not run for chart requests")

    llm = FakeListChatModel(responses=["not json", "still not json"])

    final_text, _, metadata = run_general_agent(
        llm,
        tools=[workspace_list, load_dataset, inspect_columns, execute_code, run_nlp_column_task],
        messages=[],
        user_text="Generate a revenue-by-region chart from the uploaded workbook.",
        system_prompt="Use tools when needed.",
        force_plan_execute=True,
    )

    assert "Docker sandbox" in final_text
    assert "build-sandbox-image" in final_text
    assert "doctor --strict" in final_text
    assert metadata["fallback"] == "data_analyst_guided"
    assert metadata["guided_mode"] == "code"


def test_job_runner_builds_runtime_trace_callbacks_for_workers(tmp_path: Path, monkeypatch):
    kernel = RuntimeKernel(
        _runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    job = kernel.job_manager.create_job(
        agent_name="utility",
        prompt="original worker prompt",
        session_id="tenant:user:conv",
        description="worker trace propagation",
        metadata={
            "session_state": {
                "tenant_id": "tenant",
                "user_id": "user",
                "conversation_id": "conv",
            },
            "worker_request": {
                "task_id": "task_1",
                "skill_queries": [],
            },
        },
    )
    captured = {}

    def fake_run_agent(agent, session_state, *, user_text, callbacks, task_payload=None):
        captured["callbacks"] = callbacks
        return SimpleNamespace(
            text="worker complete",
            messages=[],
            metadata={},
        )

    monkeypatch.setattr(kernel, "run_agent", fake_run_agent)

    kernel._job_runner(job)

    callbacks = captured["callbacks"]
    assert callbacks
    assert any(isinstance(callback, RuntimeTraceCallbackHandler) for callback in callbacks)


def test_ingest_and_summarize_uploads_uses_langchain_callbacks_without_name_error(
    tmp_path: Path,
    monkeypatch,
):
    settings = _runtime_settings(tmp_path)
    providers = SimpleNamespace(chat=object(), judge=object(), embeddings=object())
    stores = SimpleNamespace()

    monkeypatch.setattr("agentic_chatbot_next.app.service.load_basic_chat_skills", lambda settings: "Be concise.")
    monkeypatch.setattr("agentic_chatbot_next.app.service.ensure_kb_indexed", lambda *args, **kwargs: None)

    class _NoopSkillIndexSync:
        def __init__(self, *args, **kwargs):
            pass

        def sync(self, tenant_id: str) -> None:
            return None

    monkeypatch.setattr("agentic_chatbot_next.app.service.SkillIndexSync", _NoopSkillIndexSync)
    monkeypatch.setattr(
        "agentic_chatbot_next.app.service.ingest_paths",
        lambda *args, **kwargs: ["doc-upload-1"],
    )

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="upload-conv")
    upload_path = tmp_path / "upload.txt"
    upload_path.write_text("uploaded content")

    def fake_rag(*, session, query, conversation_context, preferred_doc_ids, providers, callbacks):
        assert preferred_doc_ids == ["doc-upload-1"]
        assert providers is not None
        assert isinstance(callbacks, list)
        return {
            "answer": "Upload summary",
            "citations": [],
            "used_citation_ids": [],
            "warnings": [],
            "followups": [],
        }

    monkeypatch.setattr(app, "_call_rag_direct", fake_rag)

    doc_ids, rendered = app.ingest_and_summarize_uploads(session, [upload_path])

    assert doc_ids == ["doc-upload-1"]
    assert "Upload summary" in rendered
    assert session.uploaded_doc_ids == ["doc-upload-1"]


def test_ingest_and_summarize_uploads_uses_rag_worker_provider_override(tmp_path: Path, monkeypatch):
    settings = _runtime_settings(tmp_path)
    providers = SimpleNamespace(chat=object(), judge=object(), embeddings=object())
    stores = SimpleNamespace()

    monkeypatch.setattr("agentic_chatbot_next.app.service.load_basic_chat_skills", lambda settings: "Be concise.")
    monkeypatch.setattr("agentic_chatbot_next.app.service.ensure_kb_indexed", lambda *args, **kwargs: None)

    class _NoopSkillIndexSync:
        def __init__(self, *args, **kwargs):
            pass

        def sync(self, tenant_id: str) -> None:
            return None

    monkeypatch.setattr("agentic_chatbot_next.app.service.SkillIndexSync", _NoopSkillIndexSync)
    monkeypatch.setattr(
        "agentic_chatbot_next.app.service.ingest_paths",
        lambda *args, **kwargs: ["doc-upload-1"],
    )

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="upload-conv")
    upload_path = tmp_path / "upload.txt"
    upload_path.write_text("uploaded content")
    rag_override = SimpleNamespace(chat=object(), judge=object(), embeddings=providers.embeddings)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        app.kernel,
        "resolve_providers_for_agent",
        lambda agent_name: rag_override if agent_name == "rag_worker" else providers,
    )

    def fake_rag(*, session, query, conversation_context, preferred_doc_ids, providers, callbacks):
        del session, query, conversation_context, preferred_doc_ids, callbacks
        captured["providers"] = providers
        return {
            "answer": "Upload summary",
            "citations": [],
            "used_citation_ids": [],
            "warnings": [],
            "followups": [],
        }

    monkeypatch.setattr(app, "_call_rag_direct", fake_rag)

    app.ingest_and_summarize_uploads(session, [upload_path])

    assert captured["providers"] is rag_override


def test_process_turn_forwards_extra_callbacks_to_kernel(tmp_path: Path, monkeypatch):
    settings = _runtime_settings(tmp_path)
    providers = SimpleNamespace(chat=object(), judge=object(), embeddings=object())
    stores = SimpleNamespace()

    monkeypatch.setattr("agentic_chatbot_next.app.service.load_basic_chat_skills", lambda settings: "Be concise.")
    monkeypatch.setattr("agentic_chatbot_next.app.service.ensure_kb_indexed", lambda *args, **kwargs: None)

    class _NoopSkillIndexSync:
        def __init__(self, *args, **kwargs):
            pass

        def sync(self, tenant_id: str) -> None:
            return None

    monkeypatch.setattr("agentic_chatbot_next.app.service.SkillIndexSync", _NoopSkillIndexSync)
    monkeypatch.setattr(
        "agentic_chatbot_next.app.service.route_turn",
        lambda *args, **kwargs: SimpleNamespace(
            route="BASIC",
            confidence=1.0,
            reasons=["deterministic"],
            router_method="deterministic",
            suggested_agent="",
        ),
    )

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="callbacks-conv")
    callback = object()
    captured: dict[str, object] = {}

    def fake_basic_turn(session_arg, *, user_text, system_prompt, chat_llm, route_metadata, callbacks):
        captured["session"] = session_arg
        captured["callbacks"] = callbacks
        captured["route"] = dict(route_metadata or {})
        return "ok"

    monkeypatch.setattr(app.kernel, "process_basic_turn", fake_basic_turn)

    text = app.process_turn(session, user_text="hello", extra_callbacks=[callback])

    assert text == "ok"
    assert captured["session"] is session
    assert captured["callbacks"] == [callback]
    assert captured["route"]["route"] == "BASIC"


def test_process_turn_uses_basic_agent_provider_override_for_basic_route(tmp_path: Path, monkeypatch):
    settings = _runtime_settings(tmp_path)
    providers = SimpleNamespace(chat=object(), judge=object(), embeddings=object())
    stores = SimpleNamespace()

    monkeypatch.setattr("agentic_chatbot_next.app.service.load_basic_chat_skills", lambda settings: "Be concise.")
    monkeypatch.setattr("agentic_chatbot_next.app.service.ensure_kb_indexed", lambda *args, **kwargs: None)

    class _NoopSkillIndexSync:
        def __init__(self, *args, **kwargs):
            pass

        def sync(self, tenant_id: str) -> None:
            return None

    monkeypatch.setattr("agentic_chatbot_next.app.service.SkillIndexSync", _NoopSkillIndexSync)
    monkeypatch.setattr(
        "agentic_chatbot_next.app.service.route_turn",
        lambda *args, **kwargs: SimpleNamespace(
            route="BASIC",
            confidence=1.0,
            reasons=["deterministic"],
            router_method="deterministic",
            suggested_agent="",
        ),
    )

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="basic-override-conv")
    basic_override = SimpleNamespace(chat=object(), judge=object(), embeddings=providers.embeddings)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        app.kernel,
        "resolve_providers_for_agent",
        lambda agent_name, chat_max_output_tokens=None: basic_override if agent_name == "basic" else providers,
    )

    def fake_basic_turn(session_arg, *, user_text, system_prompt, chat_llm, route_metadata, callbacks):
        del session_arg, user_text, system_prompt, route_metadata, callbacks
        captured["chat_llm"] = chat_llm
        return "ok"

    monkeypatch.setattr(app.kernel, "process_basic_turn", fake_basic_turn)

    text = app.process_turn(session, user_text="hello")

    assert text == "ok"
    assert captured["chat_llm"] is basic_override.chat


def test_process_turn_emits_router_degraded_event_when_judge_breaker_is_open(tmp_path: Path, monkeypatch):
    settings = _runtime_settings(tmp_path)
    settings.llm_router_enabled = True
    settings.llm_router_confidence_threshold = 0.95

    class _OpenJudge:
        def with_structured_output(self, schema):
            del schema
            return self

        def invoke(self, messages, config=None):
            del messages, config
            raise CircuitBreakerOpenError(
                key="judge:test:model",
                provider_role="judge",
                provider_name="test",
                model_name="model",
            )

    providers = SimpleNamespace(
        chat=FakeListChatModel(responses=["Fallback BASIC answer."]),
        judge=_OpenJudge(),
        embeddings=object(),
    )
    stores = SimpleNamespace()

    monkeypatch.setattr("agentic_chatbot_next.app.service.load_basic_chat_skills", lambda settings: "Be concise.")
    monkeypatch.setattr("agentic_chatbot_next.app.service.ensure_kb_indexed", lambda *args, **kwargs: None)

    class _NoopSkillIndexSync:
        def __init__(self, *args, **kwargs):
            pass

        def sync(self, tenant_id: str) -> None:
            return None

    monkeypatch.setattr("agentic_chatbot_next.app.service.SkillIndexSync", _NoopSkillIndexSync)

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="router-degraded-conv")
    monkeypatch.setattr(app.kernel, "process_agent_turn", lambda *args, **kwargs: "agent fallback answer")

    text = app.process_turn(session, user_text="Review the policy changes.")

    assert text == "agent fallback answer"
    store = RuntimeTranscriptStore(RuntimePaths.from_settings(settings))
    events = store.load_session_events(session.session_id)
    event_types = {row["event_type"] for row in events}
    assert "router_degraded_to_deterministic" in event_types


def test_process_turn_downgrades_agent_to_basic_when_agent_breaker_is_open(tmp_path: Path, monkeypatch):
    settings = _runtime_settings(tmp_path)
    providers = SimpleNamespace(chat=object(), judge=object(), embeddings=object())
    stores = SimpleNamespace()

    monkeypatch.setattr("agentic_chatbot_next.app.service.load_basic_chat_skills", lambda settings: "Be concise.")
    monkeypatch.setattr("agentic_chatbot_next.app.service.ensure_kb_indexed", lambda *args, **kwargs: None)

    class _NoopSkillIndexSync:
        def __init__(self, *args, **kwargs):
            pass

        def sync(self, tenant_id: str) -> None:
            return None

    monkeypatch.setattr("agentic_chatbot_next.app.service.SkillIndexSync", _NoopSkillIndexSync)
    monkeypatch.setattr(
        "agentic_chatbot_next.app.service.route_turn",
        lambda *args, **kwargs: SimpleNamespace(
            route="AGENT",
            confidence=1.0,
            reasons=["deterministic"],
            router_method="deterministic",
            suggested_agent="general",
        ),
    )

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="agent-downgrade-conv")
    general_bundle = SimpleNamespace(chat=object())
    basic_bundle = SimpleNamespace(chat=object())
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        app.kernel,
        "resolve_providers_for_agent",
        lambda agent_name, chat_max_output_tokens=None: basic_bundle if agent_name == "basic" else general_bundle,
    )
    monkeypatch.setattr(
        app.kernel,
        "bundle_role_identity",
        lambda bundle, role: (role, str(id(bundle)), role),
    )
    monkeypatch.setattr(app.kernel, "is_bundle_role_open", lambda bundle, role: False)

    def fail_agent_turn(*args, **kwargs):
        raise CircuitBreakerOpenError(
            key="chat:test:model",
            provider_role="chat",
            provider_name="test",
            model_name="model",
        )

    def fake_basic_turn(
        session_arg,
        *,
        user_text,
        system_prompt,
        chat_llm,
        route_metadata,
        callbacks,
        user_already_recorded=False,
    ):
        del session_arg, user_text, system_prompt, chat_llm, route_metadata, callbacks
        captured["user_already_recorded"] = user_already_recorded
        return "basic fallback"

    monkeypatch.setattr(app.kernel, "process_agent_turn", fail_agent_turn)
    monkeypatch.setattr(app.kernel, "process_basic_turn", fake_basic_turn)

    text = app.process_turn(session, user_text="Find the grounded answer.")

    assert text == "basic fallback"
    assert captured["user_already_recorded"] is True
    store = RuntimeTranscriptStore(RuntimePaths.from_settings(settings))
    events = store.load_session_events(session.session_id)
    assert any(row["event_type"] == "agent_downgraded_to_basic" for row in events)


def test_process_turn_returns_degraded_response_when_agent_and_basic_are_unavailable(tmp_path: Path, monkeypatch):
    settings = _runtime_settings(tmp_path)
    providers = SimpleNamespace(chat=object(), judge=object(), embeddings=object())
    stores = SimpleNamespace()

    monkeypatch.setattr("agentic_chatbot_next.app.service.load_basic_chat_skills", lambda settings: "Be concise.")
    monkeypatch.setattr("agentic_chatbot_next.app.service.ensure_kb_indexed", lambda *args, **kwargs: None)

    class _NoopSkillIndexSync:
        def __init__(self, *args, **kwargs):
            pass

        def sync(self, tenant_id: str) -> None:
            return None

    monkeypatch.setattr("agentic_chatbot_next.app.service.SkillIndexSync", _NoopSkillIndexSync)
    monkeypatch.setattr(
        "agentic_chatbot_next.app.service.route_turn",
        lambda *args, **kwargs: SimpleNamespace(
            route="AGENT",
            confidence=1.0,
            reasons=["deterministic"],
            router_method="deterministic",
            suggested_agent="general",
        ),
    )

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="degraded-conv")
    general_bundle = SimpleNamespace(chat=object())
    basic_bundle = SimpleNamespace(chat=object())

    monkeypatch.setattr(
        app.kernel,
        "resolve_providers_for_agent",
        lambda agent_name, chat_max_output_tokens=None: basic_bundle if agent_name == "basic" else general_bundle,
    )
    monkeypatch.setattr(
        app.kernel,
        "bundle_role_identity",
        lambda bundle, role: (role, str(id(bundle)), role),
    )
    monkeypatch.setattr(app.kernel, "is_bundle_role_open", lambda bundle, role: False)

    def fail_with_open_breaker(*args, **kwargs):
        raise CircuitBreakerOpenError(
            key="chat:test:model",
            provider_role="chat",
            provider_name="test",
            model_name="model",
        )

    monkeypatch.setattr(app.kernel, "process_agent_turn", fail_with_open_breaker)
    monkeypatch.setattr(app.kernel, "process_basic_turn", fail_with_open_breaker)

    text = app.process_turn(session, user_text="Find the grounded answer.")

    assert "temporarily degraded" in text
    assert session.messages[-1].content == text
    store = RuntimeTranscriptStore(RuntimePaths.from_settings(settings))
    events = store.load_session_events(session.session_id)
    event_types = [row["event_type"] for row in events]
    assert "agent_downgraded_to_basic" in event_types
    assert "degraded_response_returned" in event_types


def test_process_turn_upgrades_basic_router_candidate_when_semantic_contract_requires_agent(tmp_path: Path, monkeypatch):
    settings = _runtime_settings(tmp_path)
    providers = SimpleNamespace(chat=object(), judge=object(), embeddings=object())
    stores = SimpleNamespace()

    monkeypatch.setattr("agentic_chatbot_next.app.service.load_basic_chat_skills", lambda settings: "Be concise.")
    monkeypatch.setattr("agentic_chatbot_next.app.service.ensure_kb_indexed", lambda *args, **kwargs: None)

    class _NoopSkillIndexSync:
        def __init__(self, *args, **kwargs):
            pass

        def sync(self, tenant_id: str) -> None:
            return None

    monkeypatch.setattr("agentic_chatbot_next.app.service.SkillIndexSync", _NoopSkillIndexSync)
    monkeypatch.setattr(
        "agentic_chatbot_next.app.service.route_turn",
        lambda *args, **kwargs: SimpleNamespace(
            route="BASIC",
            confidence=0.22,
            reasons=["general_knowledge_or_small_talk"],
            router_method="deterministic",
            suggested_agent="",
            semantic_contract={
                "route": "BASIC",
                "requires_external_evidence": True,
                "answer_origin": "retrieval",
                "requested_scope_kind": "knowledge_base",
                "requested_collection_id": "rfp-corpus",
                "confidence": 0.22,
                "reasoning": "named collection requires retrieval",
            },
        ),
    )

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="semantic-gate-conv")
    captured: dict[str, object] = {"basic_called": False}

    def fake_basic_turn(*args, **kwargs):
        captured["basic_called"] = True
        return "basic answer"

    def fake_agent_turn(*args, **kwargs):
        captured["agent_name"] = kwargs.get("agent_name")
        return "agent answer"

    monkeypatch.setattr(app.kernel, "process_basic_turn", fake_basic_turn)
    monkeypatch.setattr(app.kernel, "process_agent_turn", fake_agent_turn)

    text = app.process_turn(
        session,
        user_text='What is the approved current CDR date for Asterion from the "rfp-corpus"?',
    )

    assert text == "agent answer"
    assert captured["basic_called"] is False
    assert captured["agent_name"] == "rag_worker"
    assert session.metadata["kb_collection_id"] == "rfp-corpus"
    assert session.metadata["requested_kb_collection_id"] == "rfp-corpus"
    assert session.metadata["kb_collection_confirmed"] is True

    store = RuntimeTranscriptStore(RuntimePaths.from_settings(settings))
    events = store.load_session_events(session.session_id)
    router_event = next(row for row in events if row["event_type"] == "router_decision")
    assert router_event["payload"]["route"] == "AGENT"
    assert router_event["payload"]["router_evidence"]["basic_candidate_upgraded"] is True


def test_process_turn_defaults_broad_grounded_analysis_to_coordinator(tmp_path: Path, monkeypatch):
    settings = _runtime_settings(tmp_path)
    providers = SimpleNamespace(chat=object(), judge=object(), embeddings=object())
    stores = SimpleNamespace()

    monkeypatch.setattr("agentic_chatbot_next.app.service.load_basic_chat_skills", lambda settings: "Be concise.")
    monkeypatch.setattr("agentic_chatbot_next.app.service.ensure_kb_indexed", lambda *args, **kwargs: None)

    class _NoopSkillIndexSync:
        def __init__(self, *args, **kwargs):
            pass

        def sync(self, tenant_id: str) -> None:
            return None

    monkeypatch.setattr("agentic_chatbot_next.app.service.SkillIndexSync", _NoopSkillIndexSync)
    monkeypatch.setattr(
        "agentic_chatbot_next.app.service.route_turn",
        lambda *args, **kwargs: SimpleNamespace(
            route="AGENT",
            confidence=0.88,
            reasons=["document_grounding_intent"],
            router_method="deterministic",
            suggested_agent="rag_worker",
        ),
    )

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="broad-analysis-conv")
    captured: dict[str, object] = {}

    def fake_agent_turn(session_arg, *, user_text, callbacks, agent_name, route_metadata, chat_max_output_tokens=None):
        del session_arg, user_text, callbacks, chat_max_output_tokens
        captured["agent_name"] = agent_name
        captured["route_metadata"] = dict(route_metadata or {})
        return "agent answer"

    monkeypatch.setattr(app.kernel, "process_agent_turn", fake_agent_turn)

    text = app.process_turn(
        session,
        user_text="Find the docs about major subsystems and synthesize them thoroughly across the default collection.",
    )

    assert text == "agent answer"
    assert captured["agent_name"] == "coordinator"
    assert captured["route_metadata"]["coordinator_default_applied"] is True


def test_process_turn_keeps_inventory_queries_off_coordinator_default_path(tmp_path: Path, monkeypatch) -> None:
    settings = _runtime_settings(tmp_path)
    providers = SimpleNamespace(
        chat=object(),
        judge=object(),
        embeddings=object(),
    )
    stores = SimpleNamespace()

    monkeypatch.setattr("agentic_chatbot_next.app.service.load_basic_chat_skills", lambda settings: "Be concise.")
    monkeypatch.setattr("agentic_chatbot_next.app.service.ensure_kb_indexed", lambda *args, **kwargs: None)

    class _NoopSkillIndexSync:
        def __init__(self, *args, **kwargs):
            pass

        def sync(self, tenant_id: str) -> None:
            return None

    monkeypatch.setattr("agentic_chatbot_next.app.service.SkillIndexSync", _NoopSkillIndexSync)
    monkeypatch.setattr(
        "agentic_chatbot_next.app.service.route_turn",
        lambda *args, **kwargs: SimpleNamespace(
            route="AGENT",
            confidence=0.88,
            reasons=["kb_inventory_intent"],
            router_method="deterministic",
            suggested_agent="general",
            semantic_contract={
                "route": "AGENT",
                "suggested_agent": "general",
                "requires_external_evidence": True,
                "answer_origin": "retrieval",
                "requested_scope_kind": "knowledge_base",
                "requested_collection_id": "",
                "confidence": 0.88,
                "reasoning": "stale KB inventory evidence flag",
            },
        ),
    )

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="inventory-conv")
    captured: dict[str, object] = {}

    def fake_agent_turn(session_arg, *, user_text, callbacks, agent_name, route_metadata, chat_max_output_tokens=None):
        del session_arg, user_text, callbacks, chat_max_output_tokens
        captured["agent_name"] = agent_name
        captured["route_metadata"] = dict(route_metadata or {})
        return "agent answer"

    monkeypatch.setattr(app.kernel, "process_agent_turn", fake_agent_turn)

    text = app.process_turn(
        session,
        user_text="what knowledge bases do i have access to",
    )

    assert text == "agent answer"
    assert captured["agent_name"] == "general"
    assert captured["route_metadata"]["coordinator_default_applied"] is False


def test_process_turn_routes_graph_evidence_scope_to_graph_manager(tmp_path: Path, monkeypatch) -> None:
    settings = _runtime_settings(tmp_path)
    providers = SimpleNamespace(
        chat=object(),
        judge=object(),
        embeddings=object(),
    )
    stores = SimpleNamespace()

    monkeypatch.setattr("agentic_chatbot_next.app.service.load_basic_chat_skills", lambda settings: "Be concise.")
    monkeypatch.setattr("agentic_chatbot_next.app.service.ensure_kb_indexed", lambda *args, **kwargs: None)

    class _NoopSkillIndexSync:
        def __init__(self, *args, **kwargs):
            pass

        def sync(self, tenant_id: str) -> None:
            return None

    monkeypatch.setattr("agentic_chatbot_next.app.service.SkillIndexSync", _NoopSkillIndexSync)
    monkeypatch.setattr(
        "agentic_chatbot_next.app.service.route_turn",
        lambda *args, **kwargs: SimpleNamespace(
            route="AGENT",
            confidence=0.90,
            reasons=["llm_router: graph relationship evidence"],
            router_method="llm",
            suggested_agent="rag_worker",
            semantic_contract={
                "route": "AGENT",
                "suggested_agent": "rag_worker",
                "requires_external_evidence": True,
                "answer_origin": "retrieval",
                "requested_scope_kind": "graph_indexes",
                "requested_collection_id": "defense-rag-v2",
                "confidence": 0.90,
                "reasoning": "graph-backed relationship analysis",
            },
        ),
    )

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="graph-evidence-conv")
    captured: dict[str, object] = {}

    def fake_agent_turn(session_arg, *, user_text, callbacks, agent_name, route_metadata, chat_max_output_tokens=None):
        del session_arg, user_text, callbacks, chat_max_output_tokens
        captured["agent_name"] = agent_name
        captured["route_metadata"] = dict(route_metadata or {})
        return "graph answer"

    monkeypatch.setattr(app.kernel, "process_agent_turn", fake_agent_turn)

    text = app.process_turn(
        session,
        user_text=(
            "Use the knowledge graph defense_rag_v2_graph to find cross-document relationships "
            "between vendors, risks, approvals, dependencies, and program outcomes."
        ),
    )

    assert text == "graph answer"
    assert captured["agent_name"] == "graph_manager"
    assert captured["route_metadata"]["coordinator_default_applied"] is False


def test_process_turn_resumes_graph_clarification_with_graph_manager(tmp_path: Path, monkeypatch) -> None:
    settings = _runtime_settings(tmp_path)
    providers = SimpleNamespace(
        chat=object(),
        judge=object(),
        embeddings=object(),
    )
    stores = SimpleNamespace()

    monkeypatch.setattr("agentic_chatbot_next.app.service.load_basic_chat_skills", lambda settings: "Be concise.")
    monkeypatch.setattr("agentic_chatbot_next.app.service.ensure_kb_indexed", lambda *args, **kwargs: None)

    class _NoopSkillIndexSync:
        def __init__(self, *args, **kwargs):
            pass

        def sync(self, tenant_id: str) -> None:
            return None

    monkeypatch.setattr("agentic_chatbot_next.app.service.SkillIndexSync", _NoopSkillIndexSync)
    captured: dict[str, object] = {}

    def fake_route_turn(*args, **kwargs):
        routed_text = str(kwargs.get("user_text") or "")
        captured["routed_text"] = routed_text
        return SimpleNamespace(
            route="AGENT",
            confidence=0.92,
            reasons=["graph_retrieval_intent"],
            router_method="deterministic",
            suggested_agent="graph_manager",
            semantic_contract={
                "route": "AGENT",
                "suggested_agent": "graph_manager",
                "requires_external_evidence": True,
                "answer_origin": "retrieval",
                "requested_scope_kind": "graph_indexes",
                "requested_collection_id": "defense-rag-v2",
                "confidence": 0.92,
                "reasoning": "graph clarification resumed",
            },
        )

    monkeypatch.setattr("agentic_chatbot_next.app.service.route_turn", fake_route_turn)

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="graph-clarification-conv")
    original_request = (
        "Use the knowledge graph defense_rag_v2_graph to find cross-document relationships "
        "between vendors, risks, approvals, dependencies, and program outcomes."
    )
    semantic_routing = {
        "route": "AGENT",
        "suggested_agent": "graph_manager",
        "requires_external_evidence": True,
        "answer_origin": "retrieval",
        "requested_scope_kind": "graph_indexes",
        "requested_collection_id": "defense-rag-v2",
    }
    resolved = resolve_turn_intent(original_request, {"semantic_routing": semantic_routing}).to_dict()
    stored = SessionState(tenant_id=session.tenant_id, user_id=session.user_id, conversation_id=session.conversation_id)
    stored.metadata = {
        "route_context": {"semantic_routing": semantic_routing, "suggested_agent": "graph_manager"},
        "semantic_routing": semantic_routing,
        "resolved_turn_intent": resolved,
        "pending_clarification": {
            "question": "What output format should I use?",
            "reason": "answer_format_selection",
            "options": ["Textual synthesis", "Diagram", "Table", "Mixed"],
            "selected_agent": "graph_manager",
            "semantic_routing": semantic_routing,
            "resolved_turn_intent": resolved,
        },
    }
    app.kernel.transcript_store.persist_session_state(stored)

    def fake_agent_turn(session_arg, *, user_text, callbacks, agent_name, route_metadata, chat_max_output_tokens=None):
        del session_arg, callbacks, chat_max_output_tokens
        captured["agent_user_text"] = user_text
        captured["agent_name"] = agent_name
        captured["route_metadata"] = dict(route_metadata or {})
        return "graph synthesis"

    monkeypatch.setattr(app.kernel, "process_agent_turn", fake_agent_turn)

    text = app.process_turn(session, user_text="textual synthesis")

    assert text == "graph synthesis"
    assert "defense_rag_v2_graph" in str(captured["routed_text"])
    assert "Clarification resolution" in str(captured["routed_text"])
    assert captured["agent_user_text"] == "textual synthesis"
    assert captured["agent_name"] == "graph_manager"
    assert captured["route_metadata"]["clarification_resume_applied"] is True


def test_live_progress_sink_translates_peer_dispatch_events() -> None:
    sink = LiveProgressSink()
    sink.emit(
        RuntimeEvent(
            event_type="peer_agent_dispatch",
            session_id="tenant:user:conv",
            job_id="job_peer_123",
            agent_name="data_analyst",
            payload={
                "target_agent": "data_analyst",
                "description": "analyze the evidence",
                "reused_existing_job": False,
            },
        )
    )

    event = sink.events.get_nowait()

    assert event["type"] == "peer_dispatch"
    assert event["label"] == "Queued data_analyst"
    assert event["detail"] == "analyze the evidence"
    assert event["job_id"] == "job_peer_123"


def test_live_progress_sink_translates_tool_lifecycle_to_status_cards() -> None:
    sink = LiveProgressSink()
    sink.emit(
        RuntimeEvent(
            event_id="evt_start",
            event_type="tool_start",
            session_id="tenant:user:conv",
            agent_name="general",
            tool_name="search_indexed_docs",
            payload={
                "tool_call_id": "call-1",
                "tool_name": "search_indexed_docs",
                "status": "running",
                "input_preview": '{"query":"trace"}',
                "input": {"query": "trace"},
                "parent_agent": "coordinator",
                "parallel_group_id": "tool-wave-general-1-call-1",
                "redacted_fields": ["input.api_key"],
                "payload_limit_chars": 12000,
                "started_at": "2026-04-23T10:00:00Z",
            },
        )
    )
    sink.emit(
        RuntimeEvent(
            event_id="evt_end",
            event_type="tool_end",
            session_id="tenant:user:conv",
            agent_name="general",
            tool_name="search_indexed_docs",
            payload={
                "tool_call_id": "call-1",
                "tool_name": "search_indexed_docs",
                "status": "completed",
                "input_preview": '{"query":"trace"}',
                "input": {"query": "trace"},
                "output_preview": '{"hits":1}',
                "output": {"hits": 1},
                "parent_agent": "coordinator",
                "parallel_group_id": "tool-wave-general-1-call-1",
                "redacted_fields": ["input.api_key"],
                "payload_limit_chars": 12000,
                "started_at": "2026-04-23T10:00:00Z",
                "completed_at": "2026-04-23T10:00:01Z",
                "duration_ms": 1000,
            },
        )
    )

    started = sink.events.get_nowait()
    completed = sink.events.get_nowait()

    assert started["type"] == "tool_trace"
    assert started["status_id"] == "tool-call-1"
    assert started["done"] is False
    assert started["agentic_tool_call"]["tool_call_id"] == "call-1"
    assert started["agentic_tool_call"]["input"] == {"query": "trace"}
    assert started["agentic_tool_call"]["parent_agent"] == "coordinator"
    assert started["agentic_tool_call"]["parallel_group_id"] == "tool-wave-general-1-call-1"
    assert started["agentic_tool_call"]["redacted_fields"] == ["input.api_key"]
    assert started["agentic_tool_call"]["payload_limit_chars"] == 12000
    assert completed["status_id"] == "tool-call-1"
    assert completed["done"] is True
    assert completed["agentic_tool_call"]["status"] == "completed"
    assert completed["agentic_tool_call"]["output"] == {"hits": 1}
    assert completed["agentic_tool_call"]["source_event_id"] == "evt_end"


def test_live_progress_sink_translates_worker_agent_activity_payloads() -> None:
    sink = LiveProgressSink()
    sink.emit(
        RuntimeEvent(
            event_id="evt_worker_start",
            event_type="worker_agent_started",
            session_id="tenant:user:conv",
            job_id="job-1",
            agent_name="rag_worker",
            payload={
                "job_id": "job-1",
                "task_id": "T2",
                "title": "Find source evidence",
                "detail": "Searching evidence for task T2.",
                "parent_agent": "coordinator",
                "parallel_group_id": "worker-batch-T1-T2",
            },
        )
    )

    event = sink.events.get_nowait()

    assert event["type"] == "worker_start"
    assert event["status_id"] == "agent-agent-worker-rag_worker-T2-job-1"
    assert event["agentic_status"]["version"] == 1
    assert event["agentic_status"]["kind"] == "agent"
    assert event["agentic_agent_activity"] == {
        "version": 1,
        "activity_id": "agent-worker-rag_worker-T2-job-1",
        "agent_name": "rag_worker",
        "role": "worker",
        "status": "running",
        "title": "rag_worker working on T2",
        "description": "Searching evidence for task T2.",
        "parent_agent": "coordinator",
        "task_id": "T2",
        "job_id": "job-1",
        "parallel_group_id": "worker-batch-T1-T2",
        "started_at": "",
        "completed_at": "",
        "duration_ms": None,
    }


def test_live_progress_sink_translates_top_level_agent_completion_activity() -> None:
    sink = LiveProgressSink()
    sink.emit(
        RuntimeEvent(
            event_id="evt_agent_done",
            event_type="agent_run_completed",
            session_id="tenant:user:conv",
            agent_name="general",
            payload={"status": "completed", "detail": "General agent finished."},
        )
    )

    event = sink.events.get_nowait()

    assert event["type"] == "summary"
    assert event["status"] == "complete"
    assert event["agentic_agent_activity"]["role"] == "top_level"
    assert event["agentic_agent_activity"]["status"] == "completed"
    assert event["agentic_status"]["kind"] == "agent"


def test_live_progress_sink_translates_parallel_group_payloads() -> None:
    sink = LiveProgressSink()
    members = [
        {"agent_name": "rag_worker", "task_id": "T1", "job_id": "job-1"},
        {"agent_name": "table_worker", "task_id": "T2", "job_id": "job-2"},
    ]
    sink.emit(
        RuntimeEvent(
            event_id="evt_group_start",
            event_type="coordinator_worker_batch_started",
            session_id="tenant:user:conv",
            agent_name="coordinator",
            payload={
                "group_id": "worker-batch-T1-T2",
                "group_kind": "worker_batch",
                "status": "running",
                "execution_mode": "parallel",
                "size": 2,
                "members": members,
                "reason": "Coordinator dispatched this worker batch in parallel.",
                "started_at": "2026-04-23T10:00:00Z",
            },
        )
    )

    event = sink.events.get_nowait()

    assert event["type"] == "parallel_group_trace"
    assert event["status_id"] == "group-worker-batch-T1-T2"
    assert event["status"] == "in_progress"
    assert event["agentic_status"]["kind"] == "parallel_group"
    assert event["agentic_parallel_group"] == {
        "version": 1,
        "group_id": "worker-batch-T1-T2",
        "group_kind": "worker_batch",
        "status": "running",
        "execution_mode": "parallel",
        "size": 2,
        "members": members,
        "reason": "Coordinator dispatched this worker batch in parallel.",
        "started_at": "2026-04-23T10:00:00Z",
        "completed_at": "",
        "duration_ms": None,
    }


def test_live_progress_sink_uses_planner_agent_for_coordinator_planning_events() -> None:
    sink = LiveProgressSink()
    sink.emit(
        RuntimeEvent(
            event_type="coordinator_planning_started",
            session_id="tenant:user:conv",
            agent_name="coordinator",
            payload={"planner_agent": "planner"},
        )
    )
    sink.emit(
        RuntimeEvent(
            event_type="coordinator_planning_completed",
            session_id="tenant:user:conv",
            agent_name="coordinator",
            payload={"planner_agent": "planner", "task_count": 3},
        )
    )

    started = sink.events.get_nowait()
    completed = sink.events.get_nowait()

    assert started["type"] == "phase_start"
    assert started["agent"] == "planner"
    assert started["selected_agent"] == "coordinator"
    assert completed["type"] == "phase_end"
    assert completed["agent"] == "planner"
    assert completed["detail"] == "3 task(s)"


def test_live_progress_sink_translates_finalizer_and_verifier_start_events() -> None:
    sink = LiveProgressSink()
    sink.emit(
        RuntimeEvent(
            event_type="coordinator_finalizer_started",
            session_id="tenant:user:conv",
            agent_name="coordinator",
            payload={"finalizer_agent": "finalizer", "revision_round": 2},
        )
    )
    sink.emit(
        RuntimeEvent(
            event_type="coordinator_verifier_started",
            session_id="tenant:user:conv",
            agent_name="coordinator",
            payload={"verifier_agent": "verifier", "revision_round": 2},
        )
    )

    finalizer_event = sink.events.get_nowait()
    verifier_event = sink.events.get_nowait()

    assert finalizer_event["type"] == "phase_start"
    assert finalizer_event["agent"] == "finalizer"
    assert finalizer_event["selected_agent"] == "coordinator"
    assert finalizer_event["detail"] == "Revision round 2"
    assert verifier_event["type"] == "phase_start"
    assert verifier_event["agent"] == "verifier"
    assert verifier_event["selected_agent"] == "coordinator"
    assert verifier_event["detail"] == "Revision round 2"
