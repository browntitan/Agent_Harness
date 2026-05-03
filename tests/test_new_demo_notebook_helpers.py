from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from new_demo_notebook.lib.scenario_runner import (  # noqa: E402
    REQUIRED_AGENT_COVERAGE,
    ScenarioDefinition,
    ScenarioAttempt,
    ScenarioResult,
    ScenarioRunner,
    _ensure_repo_import_roots,
    load_scenarios,
    validate_agent_coverage,
)
from new_demo_notebook.lib.preflight import BootstrapAction, PreflightCheck, run_preflight, bootstrap_local_dependencies  # noqa: E402
from new_demo_notebook.lib.server import BackendServerManager  # noqa: E402
from new_demo_notebook.lib.client import GatewayClient  # noqa: E402
from new_demo_notebook.lib.render import (  # noqa: E402
    build_artifact_rows,
    build_metadata_rows,
    build_progress_rows,
    build_tool_rows,
    build_workspace_preview,
    display_scenario_result,
    stream_event_console_line,
)
from new_demo_notebook.lib.trace_reader import (  # noqa: E402
    TraceBundle,
    cleanup_conversation_artifacts,
    collect_trace_bundle,
    extract_latest_router_decision,
    extract_observed_agents,
    extract_observed_route,
)


def test_server_manager_starts_waits_and_stops(monkeypatch, tmp_path: Path):
    started = {}

    class DummyProcess:
        def __init__(self):
            self._returncode = None

        def poll(self):
            return self._returncode

        def terminate(self):
            self._returncode = 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self._returncode = -9

    def fake_popen(command, cwd, stdout, stderr, text, start_new_session):
        started["command"] = command
        started["cwd"] = cwd
        started["stdout"] = stdout
        return DummyProcess()

    class DummyResponse:
        status_code = 200

        def json(self):
            return {"status": "ready"}

    monkeypatch.setattr("new_demo_notebook.lib.server.subprocess.Popen", fake_popen)
    monkeypatch.setattr("new_demo_notebook.lib.server.httpx.get", lambda *args, **kwargs: DummyResponse())

    manager = BackendServerManager(repo_root=tmp_path, artifacts_dir=tmp_path / ".artifacts", port=8765)
    base_url = manager.start(timeout_seconds=0.1)
    manager.stop()

    assert base_url == "http://127.0.0.1:8765"
    assert started["command"][1:4] == ["run.py", "serve-api", "--host"]
    assert manager.log_path.name == "server.log"


def test_server_manager_uses_env_ready_timeout_by_default(monkeypatch, tmp_path: Path):
    captured = {}

    class DummyProcess:
        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            return None

    monkeypatch.setenv("NEXT_RUNTIME_SERVER_READY_TIMEOUT_SECONDS", "321")
    monkeypatch.setattr("new_demo_notebook.lib.server.subprocess.Popen", lambda *args, **kwargs: DummyProcess())

    manager = BackendServerManager(repo_root=tmp_path, artifacts_dir=tmp_path / ".artifacts", port=8765)
    monkeypatch.setattr(
        manager,
        "wait_until_ready",
        lambda *, timeout_seconds=None: captured.setdefault("timeout_seconds", timeout_seconds),
    )

    manager.start()
    manager.stop()

    assert captured["timeout_seconds"] == 321.0


def test_server_manager_explicit_timeout_overrides_env(monkeypatch, tmp_path: Path):
    captured = {}

    class DummyProcess:
        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            return None

    monkeypatch.setenv("NEXT_RUNTIME_SERVER_READY_TIMEOUT_SECONDS", "321")
    monkeypatch.setattr("new_demo_notebook.lib.server.subprocess.Popen", lambda *args, **kwargs: DummyProcess())

    manager = BackendServerManager(repo_root=tmp_path, artifacts_dir=tmp_path / ".artifacts", port=8765)
    monkeypatch.setattr(
        manager,
        "wait_until_ready",
        lambda *, timeout_seconds=None: captured.setdefault("timeout_seconds", timeout_seconds),
    )

    manager.start(timeout_seconds=12.5)
    manager.stop()

    assert captured["timeout_seconds"] == 12.5


def test_server_manager_waits_on_configured_readiness_path(monkeypatch, tmp_path: Path):
    seen = {}

    class DummyProcess:
        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            return None

    class DummyResponse:
        status_code = 200

    def fake_get(url, timeout):
        seen["url"] = url
        seen["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr("new_demo_notebook.lib.server.subprocess.Popen", lambda *args, **kwargs: DummyProcess())
    monkeypatch.setattr("new_demo_notebook.lib.server.httpx.get", fake_get)

    manager = BackendServerManager(
        repo_root=tmp_path,
        artifacts_dir=tmp_path / ".artifacts",
        port=8765,
        readiness_path="/health/live",
    )
    manager.start(timeout_seconds=0.1)
    manager.stop()

    assert seen["url"] == "http://127.0.0.1:8765/health/live"


def test_gateway_client_uses_extended_default_timeout(monkeypatch):
    monkeypatch.delenv("NEXT_RUNTIME_GATEWAY_TIMEOUT_SECONDS", raising=False)

    client = GatewayClient("http://127.0.0.1:9999")
    try:
        assert client._client.timeout.read == 1800.0
        assert client._client.timeout.connect == 10.0
    finally:
        client.close()


def test_gateway_client_explicit_timeout_overrides_env(monkeypatch):
    monkeypatch.setenv("NEXT_RUNTIME_GATEWAY_TIMEOUT_SECONDS", "1800")

    client = GatewayClient("http://127.0.0.1:9999", timeout_seconds=45.0)
    try:
        assert client._client.timeout.read == 45.0
        assert client._client.timeout.write == 45.0
        assert client._client.timeout.pool == 45.0
    finally:
        client.close()


def test_gateway_client_parses_progress_chunks_artifacts_metadata_and_done() -> None:
    client = GatewayClient("http://127.0.0.1:9999")
    try:
        events = list(
            client._parse_sse_events(
                [
                    'event: progress',
                    'data: {"type":"route_decision","label":"Routed to rag_worker"}',
                    "",
                    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
                    "",
                    'event: artifacts',
                    'data: [{"filename":"report.csv","label":"report.csv"}]',
                    "",
                    'event: metadata',
                    'data: {"long_output":{"output_filename":"long_output_demo.md","section_count":4}}',
                    "",
                    'data: [DONE]',
                    "",
                ]
            )
        )
    finally:
        client.close()

    assert [event.kind for event in events] == ["progress", "content", "artifacts", "metadata", "done"]
    assert events[0].payload["type"] == "route_decision"
    assert events[1].text_delta == "Hello"
    assert events[2].payload[0]["filename"] == "report.csv"
    assert events[3].payload["long_output"]["output_filename"] == "long_output_demo.md"


def test_gateway_client_collect_stream_groups_text_progress_artifacts_and_metadata() -> None:
    client = GatewayClient("http://127.0.0.1:9999")
    try:
        events = list(
            client._parse_sse_events(
                [
                    'event: progress',
                    'data: {"type":"tool_call","tool":"search_skills"}',
                    "",
                    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
                    "",
                    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}',
                    "",
                    'event: artifacts',
                    'data: [{"filename":"joined.csv"}]',
                    "",
                    'event: metadata',
                    'data: {"long_output":{"output_filename":"joined.md","manifest_filename":"joined_manifest.json"}}',
                    "",
                ]
            )
        )
    finally:
        client.close()

    result = GatewayClient.collect_stream(events)

    assert result.text == "Hello world"
    assert result.progress_events[0]["tool"] == "search_skills"
    assert result.artifacts[0]["filename"] == "joined.csv"
    assert result.metadata["long_output"]["output_filename"] == "joined.md"
    assert len(result.raw_chunks) == 2


def test_render_helpers_group_progress_and_tool_rows() -> None:
    progress_events = [
        {
            "type": "decision_point",
            "label": "Planning search",
            "detail": "Preparing next action",
            "agent": "general",
            "why": "The agent is deciding what to do next.",
        },
        {
            "type": "tool_call",
            "tool": "search_skills",
            "input": {"query": "process flow"},
        },
        {
            "type": "tool_result",
            "tool": "search_skills",
            "output": "Found 2 skill packs.",
            "duration_ms": 42,
        },
    ]

    progress_rows = build_progress_rows(progress_events)
    tool_rows = build_tool_rows(progress_events)
    artifact_rows = build_artifact_rows([{"filename": "report.csv", "label": "report.csv"}])
    metadata_rows = build_metadata_rows({"job_id": "job-1", "long_output": {"output_filename": "draft.md"}})

    assert progress_rows == [
        {
            "type": "decision_point",
            "label": "Planning search",
            "detail": "Preparing next action",
            "agent": "general",
            "task_id": "",
            "job_id": "",
            "status": "",
            "why": "The agent is deciding what to do next.",
            "waiting_on": "",
            "docs": [],
        }
    ]
    assert [row["type"] for row in tool_rows] == ["tool_call", "tool_result"]
    assert artifact_rows[0]["filename"] == "report.csv"
    assert metadata_rows["job_id"] == "job-1"
    assert metadata_rows["long_output"]["output_filename"] == "draft.md"
    assert "Planning search" in stream_event_console_line(
        type("Evt", (), {"kind": "progress", "payload": progress_events[0], "text_delta": ""})()
    )
    assert "draft.md" in stream_event_console_line(
        type(
            "Evt",
            (),
            {
                "kind": "metadata",
                "payload": {"long_output": {"output_filename": "draft.md", "section_count": 4}},
                "text_delta": "",
            },
        )()
    )


def test_render_helpers_build_workspace_preview(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True)
    output_path = workspace_root / "long_output_demo.md"
    output_path.write_text("# Demo\n\nPreview text.", encoding="utf-8")

    preview = build_workspace_preview(
        TraceBundle(
            conversation_id="demo-conv",
            workspace_roots={"tenant:user:demo-conv": str(workspace_root)},
        ),
        "long_output_demo.md",
        max_chars=12,
    )

    assert preview["filename"] == "long_output_demo.md"
    assert preview["path"] == str(output_path)
    assert preview["preview"].startswith("# Demo")


def test_trace_reader_merges_session_and_job_artifacts(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    workspace_root = tmp_path / "workspaces"
    session_dir = runtime_root / "sessions" / "tenant:user:demo-conv"
    job_dir = runtime_root / "jobs" / "job_demo"
    session_dir.mkdir(parents=True)
    (job_dir / "artifacts").mkdir(parents=True)
    workspace = workspace_root / "tenant:user:demo-conv"
    workspace.mkdir(parents=True)
    (workspace / ".meta").write_text(json.dumps({"session_id": "tenant:user:demo-conv"}), encoding="utf-8")

    (session_dir / "state.json").write_text(
        json.dumps({"session_id": "tenant:user:demo-conv", "conversation_id": "demo-conv"}),
        encoding="utf-8",
    )
    (session_dir / "transcript.jsonl").write_text(
        json.dumps({"kind": "message", "message": {"role": "user", "content": "hello"}}) + "\n",
        encoding="utf-8",
    )
    (session_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "event_type": "router_decision",
                "session_id": "tenant:user:demo-conv",
                "created_at": "2026-01-01T00:00:00+00:00",
                "agent_name": "router",
                "payload": {
                    "conversation_id": "demo-conv",
                    "route": "BASIC",
                    "confidence": 0.85,
                    "reasons": ["general_knowledge_or_small_talk"],
                    "router_method": "deterministic",
                    "suggested_agent": "",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (job_dir / "state.json").write_text(
        json.dumps(
            {
                "job_id": "job_demo",
                "session_id": "tenant:user:demo-conv",
                "agent_name": "memory_maintainer",
                "status": "completed",
                "prompt": "remember this",
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "event_type": "job_completed",
                "session_id": "tenant:user:demo-conv",
                "job_id": "job_demo",
                "created_at": "2026-01-01T00:00:01+00:00",
                "agent_name": "memory_maintainer",
                "payload": {"conversation_id": "demo-conv"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = collect_trace_bundle(runtime_root, workspace_root, "demo-conv")

    assert bundle.session_ids == ["tenant:user:demo-conv"]
    assert extract_observed_route(bundle) == "BASIC"
    assert extract_latest_router_decision(bundle) == {
        "route": "BASIC",
        "confidence": 0.85,
        "reasons": ["general_knowledge_or_small_talk"],
        "router_method": "deterministic",
        "suggested_agent": "",
    }
    assert "memory_maintainer" in extract_observed_agents(bundle)
    assert bundle.workspace_files["tenant:user:demo-conv"] == []

    cleanup_conversation_artifacts(runtime_root, workspace_root, "demo-conv")
    assert not session_dir.exists()
    assert not job_dir.exists()
    assert not workspace.exists()


def test_display_scenario_result_includes_latest_router_decision(monkeypatch):
    rendered_markdown = []
    rendered_values = []

    monkeypatch.setattr("new_demo_notebook.lib.render._display_markdown", lambda text: rendered_markdown.append(text))
    monkeypatch.setattr("new_demo_notebook.lib.render._display_pretty", lambda value: rendered_values.append(value))

    scenario = ScenarioDefinition.from_dict(
        {
            "id": "basic_route_smalltalk",
            "title": "Basic Route Small Talk",
            "description": "desc",
            "conversation_id": "demo-conv",
            "messages": ["Hello"],
            "force_agent": False,
            "expected_route": "BASIC",
            "expected_agents": ["basic"],
            "expected_event_types": ["router_decision"],
            "fallback_prompts": [],
            "trace_focus": ["router_decision"],
        }
    )
    attempt = ScenarioAttempt(
        attempt_index=1,
        final_prompt="Hello",
        outputs=["Hi there"],
        raw_responses=[],
        observed_route="AGENT",
        observed_agents=["router", "coordinator"],
        observed_event_types=["router_decision"],
        validation_errors=["expected route 'BASIC', observed 'AGENT'"],
        success=False,
    )
    bundle = TraceBundle(
        conversation_id="demo-conv",
        event_rows=[
            {
                "created_at": "2026-01-01T00:00:00+00:00",
                "session_id": "tenant:user:demo-conv",
                "conversation_id": "demo-conv",
                "event_type": "router_decision",
                "route": "AGENT",
                "confidence": 0.75,
                "reasons": ["document_research_campaign"],
                "router_method": "deterministic",
                "suggested_agent": "coordinator",
                "agent_name": "router",
                "job_id": "",
                "tool_name": "",
                "payload": {},
            }
        ],
    )
    result = ScenarioResult(
        scenario=scenario,
        attempts=[attempt],
        history=[],
        bundle=bundle,
    )

    display_scenario_result(result)

    assert "### Latest Router Decision" in rendered_markdown
    assert {
        "route": "AGENT",
        "confidence": 0.75,
        "reasons": ["document_research_campaign"],
        "router_method": "deterministic",
        "suggested_agent": "coordinator",
    } in rendered_values


def test_scenario_runner_extracts_rag_contracts_from_tool_and_event_payloads(tmp_path: Path):
    runner = ScenarioRunner(
        client=type("Client", (), {})(),
        repo_root=REPO_ROOT,
        runtime_root=tmp_path / "runtime",
        workspace_root=tmp_path / "workspaces",
        model_id="enterprise-agent",
    )
    rag_contract = {
        "answer": "Grounded answer",
        "citations": [{"citation_id": "KB_1#chunk0001"}],
        "used_citation_ids": ["KB_1#chunk0001"],
        "followups": [],
        "warnings": [],
    }
    bundle = TraceBundle(
        conversation_id="demo-rag",
        session_states=[
            {
                "messages": [
                    {"role": "tool", "content": json.dumps(rag_contract)},
                    {"role": "assistant", "metadata": {"rag_contract": rag_contract}},
                ]
            }
        ],
        event_rows=[
            {
                "created_at": "2026-01-01T00:00:00+00:00",
                "session_id": "tenant:user:demo-rag",
                "conversation_id": "demo-rag",
                "event_type": "agent_turn_completed",
                "route": "AGENT",
                "confidence": 1.0,
                "reasons": ["document_grounding_intent"],
                "router_method": "deterministic",
                "suggested_agent": "rag_worker",
                "agent_name": "rag_worker",
                "job_id": "",
                "tool_name": "",
                "payload": {"rag_contract": rag_contract},
            }
        ],
    )

    contracts = runner._extract_rag_contracts(bundle)
    direct_contracts = runner._extract_event_rag_contracts(bundle)

    assert len(contracts) == 3
    assert len(direct_contracts) == 2
    assert all(contract["answer"] == "Grounded answer" for contract in contracts)


def test_trace_reader_filters_jobs_to_matching_conversation(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    workspace_root = tmp_path / "workspaces"
    matching_session_dir = runtime_root / "sessions" / "tenant-user-demo-a"
    other_session_dir = runtime_root / "sessions" / "tenant-user-demo-b"
    matching_session_dir.mkdir(parents=True)
    other_session_dir.mkdir(parents=True)

    (matching_session_dir / "state.json").write_text(
        json.dumps({"session_id": "tenant:user:demo-a", "conversation_id": "demo-a"}),
        encoding="utf-8",
    )
    (other_session_dir / "state.json").write_text(
        json.dumps({"session_id": "tenant:user:demo-b", "conversation_id": "demo-b"}),
        encoding="utf-8",
    )

    matching_job_dir = runtime_root / "jobs" / "job_match"
    unrelated_job_dir = runtime_root / "jobs" / "job_other"
    (matching_job_dir / "artifacts").mkdir(parents=True)
    (unrelated_job_dir / "artifacts").mkdir(parents=True)
    (matching_job_dir / "state.json").write_text(
        json.dumps(
            {
                "job_id": "job_match",
                "session_id": "tenant:user:demo-a",
                "agent_name": "utility",
                "status": "completed",
                "metadata": {"session_state": {"conversation_id": "demo-a"}},
            }
        ),
        encoding="utf-8",
    )
    (unrelated_job_dir / "state.json").write_text(
        json.dumps(
            {
                "job_id": "job_other",
                "session_id": "tenant:user:demo-b",
                "agent_name": "utility",
                "status": "completed",
                "metadata": {"session_state": {"conversation_id": "demo-b"}},
            }
        ),
        encoding="utf-8",
    )

    bundle = collect_trace_bundle(runtime_root, workspace_root, "demo-a")

    assert [job["job_id"] for job in bundle.jobs] == ["job_match"]


def test_scenario_runner_reuses_same_conversation_scope_for_ingest_and_chat(tmp_path: Path):
    calls = {"ingest": [], "chat": []}

    class FakeClient:
        def ingest(self, *, paths, conversation_id, source_type="upload", request_id=""):
            calls["ingest"].append((tuple(paths), conversation_id, source_type))
            return {"doc_ids": ["doc-1"]}

        def chat_turn(self, *, history, user_text, conversation_id, model, force_agent=False, request_id="", metadata=None):
            calls["chat"].append((conversation_id, user_text, force_agent))
            return type("Resp", (), {"text": f"reply:{user_text}"})()

    scenario = ScenarioDefinition.from_dict(
        {
            "id": "test-scenario",
            "title": "Test Scenario",
            "description": "desc",
            "conversation_id": "demo-conv",
            "ingest_paths": ["new_demo_notebook/demo_data/regional_spend.csv"],
            "messages": ["Turn one", "Turn two"],
            "force_agent": True,
            "expected_route": "AGENT",
            "expected_agents": ["general"],
            "expected_event_types": [],
            "fallback_prompts": [],
            "trace_focus": ["general"],
        }
    )

    def fake_trace_loader(runtime_root, workspace_root, conversation_id):
        from new_demo_notebook.lib.trace_reader import TraceBundle

        return TraceBundle(
            conversation_id=conversation_id,
            session_ids=["tenant:user:demo-conv"],
            event_rows=[
                {
                    "event_type": "router_decision",
                    "session_id": "tenant:user:demo-conv",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "agent_name": "router",
                    "job_id": "",
                    "tool_name": "",
                    "route": "AGENT",
                    "router_method": "deterministic",
                    "suggested_agent": "",
                    "conversation_id": conversation_id,
                    "payload": {"conversation_id": conversation_id, "route": "AGENT"},
                },
                {
                    "event_type": "agent_turn_completed",
                    "session_id": "tenant:user:demo-conv",
                    "created_at": "2026-01-01T00:00:01+00:00",
                    "agent_name": "general",
                    "job_id": "",
                    "tool_name": "",
                    "route": "AGENT",
                    "router_method": "deterministic",
                    "suggested_agent": "",
                    "conversation_id": conversation_id,
                    "payload": {"conversation_id": conversation_id, "route": "AGENT"},
                },
            ],
            jobs=[],
            job_events=[],
        )

    runner = ScenarioRunner(
        client=FakeClient(),
        repo_root=REPO_ROOT,
        runtime_root=tmp_path / "runtime",
        workspace_root=tmp_path / "workspaces",
        model_id="enterprise-agent",
        trace_loader=fake_trace_loader,
        cleanup_fn=lambda runtime_root, workspace_root, conversation_id: None,
    )

    result = runner.run_scenario(scenario)

    assert result.success is True
    assert calls["ingest"][0][1] == "demo-conv"
    assert [conversation_id for conversation_id, _, _ in calls["chat"]] == ["demo-conv", "demo-conv"]


def test_scenario_runner_uses_long_default_job_wait_timeout(monkeypatch, tmp_path: Path):
    runner = ScenarioRunner(
        client=type("Client", (), {})(),
        repo_root=REPO_ROOT,
        runtime_root=tmp_path / "runtime",
        workspace_root=tmp_path / "workspaces",
        model_id="enterprise-agent",
    )

    seen = {}

    def fake_getenv(name, default=None):
        if name == "NEXT_RUNTIME_JOB_WAIT_SECONDS":
            seen["default"] = default
            return default
        return None

    monkeypatch.setattr("new_demo_notebook.lib.scenario_runner.os.getenv", fake_getenv)
    monkeypatch.setattr(
        runner,
        "trace_loader",
        lambda runtime_root, workspace_root, conversation_id: SimpleNamespace(jobs=[]),
    )

    runner.wait_for_terminal_jobs("demo-conv")

    assert seen["default"] == "900.0"


def test_scenario_runner_explicit_job_wait_timeout_overrides_default(tmp_path: Path):
    runner = ScenarioRunner(
        client=type("Client", (), {})(),
        repo_root=REPO_ROOT,
        runtime_root=tmp_path / "runtime",
        workspace_root=tmp_path / "workspaces",
        model_id="enterprise-agent",
        job_wait_timeout_seconds=321.0,
    )

    assert runner.job_wait_timeout_seconds == 321.0


def test_extract_observed_agents_includes_coordinator_phase_agents_from_event_payload(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    workspace_root = tmp_path / "workspaces"
    session_dir = runtime_root / "sessions" / "tenant-user-demo-coordinator"
    session_dir.mkdir(parents=True)
    (session_dir / "state.json").write_text(
        json.dumps({"session_id": "tenant:user:demo-coordinator", "conversation_id": "demo-coordinator"}),
        encoding="utf-8",
    )
    (session_dir / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_type": "coordinator_planning_started",
                        "session_id": "tenant:user:demo-coordinator",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "agent_name": "coordinator",
                        "payload": {
                            "conversation_id": "demo-coordinator",
                            "planner_agent": "planner",
                        },
                    }
                ),
                json.dumps(
                    {
                        "event_type": "coordinator_finalizer_completed",
                        "session_id": "tenant:user:demo-coordinator",
                        "created_at": "2026-01-01T00:00:01+00:00",
                        "agent_name": "coordinator",
                        "payload": {
                            "conversation_id": "demo-coordinator",
                            "finalizer_agent": "finalizer",
                        },
                    }
                ),
                json.dumps(
                    {
                        "event_type": "coordinator_verifier_completed",
                        "session_id": "tenant:user:demo-coordinator",
                        "created_at": "2026-01-01T00:00:02+00:00",
                        "agent_name": "coordinator",
                        "payload": {
                            "conversation_id": "demo-coordinator",
                            "verifier_agent": "verifier",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = collect_trace_bundle(runtime_root, workspace_root, "demo-coordinator")
    observed = extract_observed_agents(bundle)

    assert "coordinator" in observed
    assert "planner" in observed
    assert "finalizer" in observed
    assert "verifier" in observed


def test_scenario_runner_accepts_markdown_citation_heading_for_general_grounded_rag(tmp_path: Path):
    scenario = ScenarioDefinition.from_dict(
        {
            "id": "general_grounded_rag",
            "title": "General Grounded RAG",
            "description": "desc",
            "conversation_id": "demo-grounded",
            "ingest_paths": [],
            "messages": [{"content": "Explain the release-note changes with citations.", "metadata": {"requested_agent": "general"}}],
            "force_agent": True,
            "expected_route": "AGENT",
            "expected_agents": ["general"],
            "expected_event_types": [],
            "fallback_prompts": [],
            "trace_focus": ["general", "rag_agent_tool"],
        }
    )
    runner = ScenarioRunner(
        client=type("Client", (), {})(),
        repo_root=REPO_ROOT,
        runtime_root=tmp_path / "runtime",
        workspace_root=tmp_path / "workspaces",
        model_id="enterprise-agent",
    )
    rag_contract = {
        "answer": "Grounded answer",
        "citations": [{"citation_id": "KB_1#chunk0001"}],
        "used_citation_ids": ["KB_1#chunk0001"],
        "followups": [],
        "warnings": [],
    }
    bundle = TraceBundle(
        conversation_id="demo-grounded",
        session_states=[
            {
                "messages": [
                    {
                        "role": "tool",
                        "content": json.dumps(rag_contract),
                    }
                ],
            }
        ],
    )

    errors = runner._scenario_specific_errors(
        scenario,
        outputs=["Summary of changes.\n\n### Citations\n- [KB_1#chunk0001] Release Notes (v1.4)"],
        raw_responses=[],
        bundle=bundle,
    )

    assert errors == []


def test_scenario_runner_accepts_contract_backed_citations_for_direct_grounded_rag(tmp_path: Path):
    scenario = ScenarioDefinition.from_dict(
        {
            "id": "direct_grounded_rag",
            "title": "Direct Grounded RAG",
            "description": "desc",
            "conversation_id": "demo-direct-grounded",
            "ingest_paths": [],
            "messages": ["Explain the release-note changes with citations."],
            "force_agent": True,
            "expected_route": "AGENT",
            "expected_agents": ["rag_worker"],
            "expected_event_types": [],
            "fallback_prompts": [],
            "trace_focus": ["rag_worker"],
        }
    )
    runner = ScenarioRunner(
        client=type("Client", (), {})(),
        repo_root=REPO_ROOT,
        runtime_root=tmp_path / "runtime",
        workspace_root=tmp_path / "workspaces",
        model_id="enterprise-agent",
    )
    rag_contract = {
        "answer": "Grounded answer",
        "citations": [{"citation_id": "KB_2#chunk0007"}],
        "used_citation_ids": ["KB_2#chunk0007"],
        "followups": [],
        "warnings": [],
    }
    bundle = TraceBundle(
        conversation_id="demo-direct-grounded",
        event_rows=[
            {
                "created_at": "2026-01-01T00:00:00+00:00",
                "session_id": "tenant:user:demo-direct-grounded",
                "conversation_id": "demo-direct-grounded",
                "event_type": "agent_turn_completed",
                "route": "AGENT",
                "confidence": 1.0,
                "reasons": ["document_grounding_intent"],
                "router_method": "deterministic",
                "suggested_agent": "rag_worker",
                "agent_name": "rag_worker",
                "job_id": "",
                "tool_name": "",
                "payload": {"rag_contract": rag_contract},
            }
        ],
    )

    errors = runner._scenario_specific_errors(
        scenario,
        outputs=["Summary of changes without a separate citations heading."],
        raw_responses=[],
        bundle=bundle,
    )

    assert errors == []


def test_scenario_runner_reports_latest_execute_code_error_for_data_analyst(tmp_path: Path):
    scenario = ScenarioDefinition.from_dict(
        {
            "id": "data_analyst_csv_review",
            "title": "Data Analyst CSV Review",
            "description": "desc",
            "conversation_id": "demo-data-analyst",
            "ingest_paths": ["new_demo_notebook/demo_data/regional_spend.csv"],
            "messages": ["Analyze the uploaded CSVs."],
            "force_agent": False,
            "expected_route": "AGENT",
            "expected_agents": ["data_analyst"],
            "expected_event_types": [],
            "fallback_prompts": [],
            "trace_focus": ["data_analyst"],
        }
    )
    runner = ScenarioRunner(
        client=type("Client", (), {})(),
        repo_root=REPO_ROOT,
        runtime_root=tmp_path / "runtime",
        workspace_root=tmp_path / "workspaces",
        model_id="enterprise-agent",
    )
    bundle = TraceBundle(
        conversation_id="demo-data-analyst",
        session_states=[
            {
                "messages": [
                    {
                        "role": "tool",
                        "content": json.dumps(
                            {
                                "stdout": "",
                                "stderr": "ModuleNotFoundError: No module named 'pandas'",
                                "success": False,
                                "execution_time_seconds": 0.1,
                            }
                        ),
                    }
                ]
            }
        ],
        workspace_files={"tenant:user:demo-data-analyst": ["regional_spend.csv"]},
    )

    errors = runner._scenario_specific_errors(
        scenario,
        outputs=["Data analysis failed."],
        raw_responses=[],
        bundle=bundle,
    )

    assert any("latest execute_code result" in error for error in errors)
    assert any("ModuleNotFoundError" in error for error in errors)


def test_scenario_runner_bootstrap_adds_src_root_to_sys_path(monkeypatch):
    repo_root = REPO_ROOT
    src_root = repo_root / "src"
    custom_sys_path = [entry for entry in sys.path if entry not in {str(repo_root), str(src_root)}]
    monkeypatch.setattr(sys, "path", custom_sys_path)

    _ensure_repo_import_roots()

    assert str(src_root) in sys.path
    assert str(repo_root) in sys.path


def test_scenario_manifest_covers_all_required_agents():
    scenarios = load_scenarios(REPO_ROOT / "new_demo_notebook" / "scenarios" / "scenarios.json")
    coverage = validate_agent_coverage(scenarios, required_agents=REQUIRED_AGENT_COVERAGE)

    assert set(REQUIRED_AGENT_COVERAGE).issubset(set(coverage))


def test_scenario_manifest_includes_both_grounded_rag_paths():
    scenarios = {
        scenario.id: scenario
        for scenario in load_scenarios(REPO_ROOT / "new_demo_notebook" / "scenarios" / "scenarios.json")
    }

    general_grounded = scenarios["general_grounded_rag"]
    direct_grounded = scenarios["direct_grounded_rag"]

    assert general_grounded.messages[-1].metadata["requested_agent"] == "general"
    assert "general" in general_grounded.expected_agents
    assert "tool_start" in general_grounded.expected_event_types

    assert direct_grounded.force_agent is True
    assert direct_grounded.expected_agents == ["rag_worker"]
    assert "tool_start" not in direct_grounded.expected_event_types
    long_form = scenarios["sync_long_form_general"]
    assert long_form.messages[-1].metadata["requested_agent"] == "general"
    assert long_form.messages[-1].metadata["long_output"]["enabled"] is True
    assert long_form.messages[-1].metadata["long_output"]["target_words"] == 1800
    assert long_form.expected_agents == ["general"]
    assert "general_utility_memory" not in scenarios
    assert "memory_maintainer_background" not in scenarios


def test_scenario_runner_validates_sync_long_form_outputs(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True)
    output_filename = "long_output_demo_runtime_brief.md"
    manifest_filename = "long_output_demo_manifest.json"
    (workspace_root / output_filename).write_text("# Runtime Brief\n\nSection content.", encoding="utf-8")
    (workspace_root / manifest_filename).write_text('{"status":"completed"}', encoding="utf-8")

    scenario = ScenarioDefinition.from_dict(
        {
            "id": "sync_long_form_general",
            "title": "Synchronous Long-Form Writing",
            "description": "desc",
            "conversation_id": "demo-conv",
            "messages": [
                {
                    "content": "Write a long-form brief.",
                    "metadata": {
                        "requested_agent": "general",
                        "long_output": {
                            "enabled": True,
                            "target_words": 1800,
                            "target_sections": 4,
                            "delivery_mode": "hybrid",
                            "output_format": "markdown",
                        },
                    },
                }
            ],
            "force_agent": True,
            "expected_route": "AGENT",
            "expected_agents": ["general"],
            "expected_event_types": ["agent_turn_completed"],
            "fallback_prompts": [],
            "trace_focus": ["general"],
        }
    )
    runner = ScenarioRunner(
        client=type("Client", (), {})(),
        repo_root=REPO_ROOT,
        runtime_root=tmp_path / "runtime",
        workspace_root=tmp_path / "workspaces",
        model_id="enterprise-agent",
    )
    bundle = TraceBundle(
        conversation_id="demo-conv",
        workspace_roots={"tenant:user:demo-conv": str(workspace_root)},
        workspace_files={"tenant:user:demo-conv": [manifest_filename, output_filename]},
    )

    errors = runner._scenario_specific_errors(
        scenario,
        outputs=["I attached the full Markdown artifact for the completed draft."],
        raw_responses=[
            {
                "artifacts": [{"filename": output_filename, "label": output_filename}],
                "metadata": {
                    "long_output": {
                        "output_filename": output_filename,
                        "manifest_filename": manifest_filename,
                        "section_count": 4,
                        "background": False,
                    }
                },
            }
        ],
        bundle=bundle,
    )

    assert errors == []


def test_preflight_reports_missing_docker_and_unreachable_db(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PG_DSN", "postgresql://localhost:59999/ragdb")
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "azure")
    monkeypatch.setenv("JUDGE_PROVIDER", "azure")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setattr("new_demo_notebook.lib.preflight.shutil.which", lambda name: None)

    report = run_preflight(
        repo_root=tmp_path,
        runtime_root=tmp_path / "runtime",
        workspace_root=tmp_path / "workspaces",
        memory_root=tmp_path / "memory",
    )

    rows = {row["name"]: row for row in report.to_rows()}
    assert report.ready is False
    assert rows["docker"]["ok"] is False
    assert rows["database"]["ok"] is False
    assert rows["chat_provider"]["ok"] is False


def test_preflight_uses_runtime_settings_loader(monkeypatch, tmp_path: Path):
    repo_root = tmp_path
    for env_name in [
        "PG_DSN",
        "LLM_PROVIDER",
        "EMBEDDINGS_PROVIDER",
        "JUDGE_PROVIDER",
        "OLLAMA_BASE_URL",
        "OLLAMA_CHAT_MODEL",
        "OLLAMA_EMBED_MODEL",
        "OLLAMA_JUDGE_MODEL",
    ]:
        monkeypatch.delenv(env_name, raising=False)
    (repo_root / ".env").write_text(
        "\n".join(
            [
                "PG_DSN=postgresql://demo:secret@db.internal:6543/demo",
                "LLM_PROVIDER=ollama",
                "EMBEDDINGS_PROVIDER=ollama",
                "JUDGE_PROVIDER=ollama",
                "OLLAMA_BASE_URL=http://models.internal:22434",
                "OLLAMA_CHAT_MODEL=demo-chat:1",
                "OLLAMA_EMBED_MODEL=demo-embed:latest",
                "OLLAMA_JUDGE_MODEL=demo-judge:2",
            ]
        ),
        encoding="utf-8",
    )

    captured = {}

    def fake_check_database(settings, *, repo_root):
        captured["pg_dsn"] = settings.pg_dsn
        return PreflightCheck(name="database", ok=True, detail=settings.pg_dsn)

    def fake_check_docker():
        return PreflightCheck(name="docker", ok=True, detail="docker daemon reachable")

    def fake_check_providers(settings, *, repo_root):
        captured["ollama_base_url"] = settings.ollama_base_url
        captured["chat_model"] = settings.ollama_chat_model
        captured["embed_model"] = settings.ollama_embed_model
        captured["judge_model"] = settings.ollama_judge_model
        return [PreflightCheck(name="chat_provider", ok=True, detail="ok")]

    def fake_check_sandbox_image(settings):
        captured["sandbox_docker_image"] = settings.sandbox_docker_image
        return PreflightCheck(name="sandbox_image", ok=True, detail="sandbox ready")

    monkeypatch.setattr("new_demo_notebook.lib.preflight._check_database", fake_check_database)
    monkeypatch.setattr("new_demo_notebook.lib.preflight._check_docker", fake_check_docker)
    monkeypatch.setattr("new_demo_notebook.lib.preflight._check_providers", fake_check_providers)
    monkeypatch.setattr("new_demo_notebook.lib.preflight._check_sandbox_image", fake_check_sandbox_image)

    report = run_preflight(
        repo_root=repo_root,
        runtime_root=tmp_path / "runtime",
        workspace_root=tmp_path / "workspaces",
        memory_root=tmp_path / "memory",
    )

    assert report.ready is True
    assert captured == {
        "pg_dsn": "postgresql://demo:secret@db.internal:6543/demo",
        "ollama_base_url": "http://models.internal:22434",
        "chat_model": "demo-chat:1",
        "embed_model": "demo-embed:latest",
        "judge_model": "demo-judge:2",
        "sandbox_docker_image": "agentic-chatbot-sandbox:py312",
    }
    assert report.resolved_settings["pg_dsn"] == "postgresql://demo:secret@db.internal:6543/demo"
    assert report.resolved_settings["memory_enabled"] is True


def test_preflight_flags_legacy_sandbox_image_with_targeted_detail(monkeypatch, tmp_path: Path):
    repo_root = tmp_path
    (repo_root / ".env").write_text(
        "\n".join(
            [
                "PG_DSN=postgresql://demo:secret@db.internal:6543/demo",
                "LLM_PROVIDER=ollama",
                "EMBEDDINGS_PROVIDER=ollama",
                "JUDGE_PROVIDER=ollama",
                "OLLAMA_BASE_URL=http://models.internal:22434",
                "OLLAMA_CHAT_MODEL=demo-chat:1",
                "OLLAMA_EMBED_MODEL=demo-embed:latest",
                "OLLAMA_JUDGE_MODEL=demo-judge:2",
                "SANDBOX_DOCKER_IMAGE=python:3.12-slim",
                "MEMORY_ENABLED=false",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "new_demo_notebook.lib.preflight._check_database",
        lambda settings, *, repo_root: PreflightCheck(name="database", ok=True, detail=settings.pg_dsn),
    )
    monkeypatch.setattr(
        "new_demo_notebook.lib.preflight._check_docker",
        lambda: PreflightCheck(name="docker", ok=True, detail="docker daemon reachable"),
    )
    monkeypatch.setattr(
        "new_demo_notebook.lib.preflight._check_providers",
        lambda settings, *, repo_root: [PreflightCheck(name="chat_provider", ok=True, detail="ok")],
    )

    report = run_preflight(
        repo_root=repo_root,
        runtime_root=tmp_path / "runtime",
        workspace_root=tmp_path / "workspaces",
        memory_root=tmp_path / "memory",
    )

    rows = {row["name"]: row for row in report.to_rows()}
    assert report.ready is False
    assert rows["sandbox_image"]["ok"] is False
    assert "legacy/stale" in rows["sandbox_image"]["detail"]
    assert "python:3.12-slim" in rows["sandbox_image"]["detail"]
    assert "agentic-chatbot-sandbox:py312" in rows["sandbox_image"]["detail"]
    assert "build-sandbox-image" in rows["sandbox_image"]["hint"]
    assert report.resolved_settings["memory_enabled"] is False
    assert "sandbox_image:" in report.failure_summary()
    assert "agentic-chatbot-sandbox:py312" in report.failure_summary()


def test_preflight_accepts_runtime_default_chat_model_and_latest_alias(monkeypatch, tmp_path: Path):
    for env_name in (
        "LLM_PROVIDER",
        "EMBEDDINGS_PROVIDER",
        "JUDGE_PROVIDER",
        "OLLAMA_CHAT_MODEL",
        "OLLAMA_JUDGE_MODEL",
        "OLLAMA_EMBED_MODEL",
    ):
        monkeypatch.delenv(env_name, raising=False)

    class DummyOllamaResponse:
        status_code = 200

        def json(self):
            return {"models": [{"name": "nemotron-cascade-2:30b"}, {"name": "nomic-embed-text:latest"}]}

        def raise_for_status(self):
            return None

    monkeypatch.setattr("new_demo_notebook.lib.preflight.httpx.get", lambda *args, **kwargs: DummyOllamaResponse())
    monkeypatch.setattr(
        "new_demo_notebook.lib.preflight._check_database",
        lambda settings, *, repo_root: PreflightCheck(name="database", ok=True, detail="localhost:5432 reachable"),
    )
    monkeypatch.setattr(
        "new_demo_notebook.lib.preflight._check_docker",
        lambda: PreflightCheck(name="docker", ok=True, detail="docker daemon reachable"),
    )
    monkeypatch.setattr(
        "new_demo_notebook.lib.preflight._check_sandbox_image",
        lambda settings: PreflightCheck(name="sandbox_image", ok=True, detail="sandbox ready"),
    )

    report = run_preflight(
        repo_root=tmp_path,
        runtime_root=tmp_path / "runtime",
        workspace_root=tmp_path / "workspaces",
        memory_root=tmp_path / "memory",
    )

    rows = {row["name"]: row for row in report.to_rows()}
    assert report.ready is True
    assert rows["chat_provider_model"]["ok"] is True
    assert rows["judge_provider_model"]["ok"] is True
    assert rows["embeddings_provider_model"]["ok"] is True


def test_preflight_database_diagnostics_include_local_service_hints(monkeypatch, tmp_path: Path):
    from new_demo_notebook.lib.preflight import _check_database

    settings = SimpleNamespace(
        pg_dsn="postgresql://raguser:ragpass@localhost:5432/ragdb",
    )
    monkeypatch.setattr(
        "new_demo_notebook.lib.preflight._docker_compose_service_status",
        lambda repo_root, service: "docker compose service `rag-postgres` is not running",
    )
    monkeypatch.setattr(
        "new_demo_notebook.lib.preflight._langfuse_postgres_hint",
        lambda: "langfuse-postgres is listening on localhost:5433, but that is the observability database rather than the app DB",
    )
    monkeypatch.setattr(
        "new_demo_notebook.lib.preflight.socket.create_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionRefusedError("boom")),
    )

    check = _check_database(settings, repo_root=tmp_path)

    assert check.ok is False
    assert "5432" in check.detail
    assert "rag-postgres" in check.detail
    assert "5433" in check.detail
    assert "docker compose up -d rag-postgres" in check.hint


def test_bootstrap_local_dependencies_starts_missing_local_postgres(monkeypatch, tmp_path: Path):
    settings = SimpleNamespace(
        pg_dsn="postgresql://raguser:ragpass@localhost:5432/ragdb",
        llm_provider="azure",
        embeddings_provider="azure",
        judge_provider="azure",
        ollama_base_url="http://localhost:11434",
        sandbox_docker_image="agentic-chatbot-sandbox:py312",
    )
    monkeypatch.setattr("new_demo_notebook.lib.preflight._load_runtime_settings", lambda repo_root: settings)
    seen = {}

    def fake_bootstrap_service(repo_root, *, service, settings, wait, compose_args=None):
        seen["service"] = service
        seen["compose_args"] = compose_args
        return BootstrapAction(
            name=service,
            ok=True,
            command="docker compose up -d rag-postgres",
            detail="localhost:5432 became reachable",
        )

    monkeypatch.setattr("new_demo_notebook.lib.preflight._bootstrap_service", fake_bootstrap_service)

    report = type(
        "Report",
        (),
        {
            "ready": False,
            "checks": [
                PreflightCheck(name="database", ok=False, detail="localhost:5432 unreachable"),
                PreflightCheck(name="docker", ok=True, detail="docker daemon reachable"),
            ],
        },
    )()

    actions = bootstrap_local_dependencies(repo_root=tmp_path, report=report)

    assert [action.name for action in actions] == ["rag-postgres"]
    assert seen["service"] == "rag-postgres"
    assert seen["compose_args"] is None


def test_bootstrap_local_dependencies_skips_remote_database(monkeypatch, tmp_path: Path):
    settings = SimpleNamespace(
        pg_dsn="postgresql://raguser:ragpass@db.example.com:5432/ragdb",
        llm_provider="azure",
        embeddings_provider="azure",
        judge_provider="azure",
        ollama_base_url="http://localhost:11434",
        sandbox_docker_image="agentic-chatbot-sandbox:py312",
    )
    monkeypatch.setattr("new_demo_notebook.lib.preflight._load_runtime_settings", lambda repo_root: settings)
    monkeypatch.setattr(
        "new_demo_notebook.lib.preflight._bootstrap_service",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bootstrap should not run for remote DB")),
    )

    report = type(
        "Report",
        (),
        {
            "ready": False,
            "checks": [
                PreflightCheck(name="database", ok=False, detail="db.example.com:5432 unreachable"),
                PreflightCheck(name="docker", ok=True, detail="docker daemon reachable"),
            ],
        },
    )()

    assert bootstrap_local_dependencies(repo_root=tmp_path, report=report) == []


def test_bootstrap_local_dependencies_builds_missing_sandbox_image(monkeypatch, tmp_path: Path):
    settings = SimpleNamespace(
        pg_dsn="postgresql://raguser:ragpass@localhost:5432/ragdb",
        llm_provider="azure",
        embeddings_provider="azure",
        judge_provider="azure",
        ollama_base_url="http://localhost:11434",
        sandbox_docker_image="agentic-chatbot-sandbox:py312",
    )
    monkeypatch.setattr("new_demo_notebook.lib.preflight._load_runtime_settings", lambda repo_root: settings)
    monkeypatch.setattr(
        "agentic_chatbot_next.sandbox.build_sandbox_image",
        lambda repo_root, image: SimpleNamespace(
            ok=True,
            detail=f"Built {image}",
            command="python run.py build-sandbox-image",
        ),
    )

    report = type(
        "Report",
        (),
        {
            "ready": False,
            "checks": [
                PreflightCheck(name="database", ok=True, detail="localhost:5432 reachable"),
                PreflightCheck(name="docker", ok=True, detail="docker daemon reachable"),
                PreflightCheck(name="sandbox_image", ok=False, detail="sandbox image missing"),
            ],
        },
    )()

    actions = bootstrap_local_dependencies(repo_root=tmp_path, report=report)

    assert [action.name for action in actions] == ["sandbox_image"]
    assert actions[0].ok is True
    assert "build-sandbox-image" in actions[0].command


def test_bootstrap_local_dependencies_reports_legacy_sandbox_config(monkeypatch, tmp_path: Path):
    settings = SimpleNamespace(
        pg_dsn="postgresql://raguser:ragpass@localhost:5432/ragdb",
        llm_provider="azure",
        embeddings_provider="azure",
        judge_provider="azure",
        ollama_base_url="http://localhost:11434",
        sandbox_docker_image="python:3.12-slim",
    )
    monkeypatch.setattr("new_demo_notebook.lib.preflight._load_runtime_settings", lambda repo_root: settings)
    monkeypatch.setattr(
        "agentic_chatbot_next.sandbox.build_sandbox_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy config should not trigger a build")),
    )

    report = type(
        "Report",
        (),
        {
            "ready": False,
            "checks": [
                PreflightCheck(name="database", ok=True, detail="localhost:5432 reachable"),
                PreflightCheck(name="docker", ok=True, detail="docker daemon reachable"),
                PreflightCheck(name="sandbox_image", ok=False, detail="legacy sandbox image configured"),
            ],
        },
    )()

    actions = bootstrap_local_dependencies(repo_root=tmp_path, report=report)

    assert [action.name for action in actions] == ["sandbox_image"]
    assert actions[0].ok is False
    assert "legacy/stale" in actions[0].detail
    assert "SANDBOX_DOCKER_IMAGE=agentic-chatbot-sandbox:py312" in actions[0].command


def test_narrative_showcase_notebook_has_required_sections_and_tags() -> None:
    notebook = json.loads((REPO_ROOT / "new_demo_notebook" / "agentic_system_showcase.ipynb").read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    sources = ["".join(cell.get("source", [])) for cell in cells]
    all_tags = {
        tag
        for cell in cells
        for tag in cell.get("metadata", {}).get("tags", [])
    }

    expected_headings = [
        "# Agentic System Showcase",
        "## How The Notebook Driver Works",
        "## Basic Chat Demo",
        "## General Agent Tool-And-Skill Demo",
        "## Delegated Grounded RAG Demo",
        "## Direct Grounded RAG Demo",
        "## Workbook-Aware RAG Demo",
        "## Data Analyst Sandbox Demo",
        "## Coordinator Orchestration Demo",
        "## Long-Form Writing Demo",
        "## Repo Skill-Pack Overview",
        "## Create Runtime Skill Draft",
        "## Preview Runtime Skill Resolution",
        "## Activate Runtime Skill",
        "## Inspect Runtime Skill",
        "## Update Runtime Skill Version",
        "## Deactivate Updated Runtime Skill",
        "## Roll Back Runtime Skill",
        "## Optional Defense Corpus Appendix",
    ]
    for heading in expected_headings:
        assert any(heading in source for source in sources), heading

    expected_tags = {
        "setup",
        "driver-walkthrough",
        "basic-demo",
        "general-demo",
        "rag-demo",
        "data-analyst-demo",
        "coordinator-demo",
        "long-form-demo",
        "skills-demo",
        "defense-optional",
        "cleanup",
    }
    assert expected_tags.issubset(all_tags)
    assert "memory-demo" not in all_tags
    assert "streaming-primer" not in all_tags
    assert "scenario-appendix" not in all_tags
    assert not any("## Streaming Primer Scenario" in source for source in sources)
    assert not any("## Runtime Skill CRUD Demo" in source for source in sources)

    for cell in cells:
        source = "".join(cell.get("source", []))
        assert "runner.run_scenario(" not in source
        assert "RUN_SCENARIO_APPENDIX" not in source

    chat_demo_headings = [
        "## Basic Chat Demo",
        "## General Agent Tool-And-Skill Demo",
        "## Delegated Grounded RAG Demo",
        "## Direct Grounded RAG Demo",
        "## Workbook-Aware RAG Demo",
        "## Data Analyst Sandbox Demo",
        "## Coordinator Orchestration Demo",
        "## Long-Form Writing Demo",
    ]
    skill_action_headings = [
        "## Repo Skill-Pack Overview",
        "## Create Runtime Skill Draft",
        "## Preview Runtime Skill Resolution",
        "## Activate Runtime Skill",
        "## Inspect Runtime Skill",
        "## Update Runtime Skill Version",
        "## Deactivate Updated Runtime Skill",
        "## Roll Back Runtime Skill",
    ]
    heading_to_code_source = {}
    for heading in chat_demo_headings + skill_action_headings:
        for idx, cell in enumerate(cells):
            if cell.get("cell_type") == "markdown" and heading in "".join(cell.get("source", [])):
                assert idx + 1 < len(cells)
                assert cells[idx + 1].get("cell_type") == "code"
                heading_to_code_source[heading] = "".join(cells[idx + 1].get("source", []))
                break
        else:
            raise AssertionError(f"Could not locate scenario heading {heading!r}")

    for heading in chat_demo_headings:
        source = heading_to_code_source[heading]
        assert source.count("run_notebook_scenario(") == 1, heading

    skill_methods = [
        "create_skill",
        "preview_skill_search",
        "activate_skill",
        "get_skill",
        "update_skill",
        "deactivate_skill",
        "rollback_skill",
    ]
    for heading in skill_action_headings[1:]:
        source = heading_to_code_source[heading]
        counts = {name: source.count(f"client.{name}(") for name in skill_methods}
        assert sum(counts.values()) == 1, (heading, counts)
        assert "run_notebook_scenario(" not in source

    for method in skill_methods:
        total = sum(source.count(f"client.{method}(") for source in heading_to_code_source.values())
        assert total == 1, (method, total)

    assert any("preflight_helper_path" in source for source in sources)
    assert any("notebook_path" in source for source in sources)
    assert any("failure_summary()" in source for source in sources)
    assert any("run_notebook_scenario(" in source for source in sources)


def test_notebook_readme_documents_source_notebook_and_bootstrap_behavior() -> None:
    readme = (REPO_ROOT / "new_demo_notebook" / "README.md").read_text(encoding="utf-8")

    assert "agentic_system_showcase.ipynb" in readme
    assert "one query or one skill action per runnable code" in readme
    assert "scenario-first main path" in readme
    assert "sync long-form writing showcase" in readme
    assert "create runtime skill draft" in readme
    assert "roll back runtime skill" in readme
    assert "logs plus summaries" in readme
    assert "bootstrap missing local Docker dependencies such as `rag-postgres`" in readme
    assert "agentic-chatbot-sandbox:py312" in readme
    assert "MEMORY_ENABLED=false" in readme
