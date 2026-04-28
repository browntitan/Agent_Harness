from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_chatbot_next.agents.registry import AgentRegistry
from agentic_chatbot_next.authz import AccessSnapshot
from agentic_chatbot_next.contracts.jobs import JobRecord
from agentic_chatbot_next.contracts.messages import RuntimeMessage, SessionState
from agentic_chatbot_next.persistence.postgres.skills import SkillPackRecord
from agentic_chatbot_next.observability.events import RuntimeEvent
from agentic_chatbot_next.runtime.artifacts import register_handoff_artifact
from agentic_chatbot_next.runtime.context import RuntimePaths
from agentic_chatbot_next.runtime.job_manager import RuntimeJobManager
from agentic_chatbot_next.runtime.kernel import RuntimeKernel
from agentic_chatbot_next.runtime.task_plan import TaskExecutionState, build_fallback_plan
from agentic_chatbot_next.runtime.turn_contracts import resolve_turn_intent
from agentic_chatbot_next.runtime.transcript_store import RuntimeTranscriptStore
from agentic_chatbot_next.skills.telemetry import SkillTelemetryEvent
from agentic_chatbot_next.tools.base import ToolContext
from agentic_chatbot_next.session import ChatSession


def _repo_agents_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "agents"


def _make_runtime_settings(tmp_path: Path, *, agents_dir: Path | None = None) -> SimpleNamespace:
    repo_root = Path(__file__).resolve().parents[1]
    return SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        workspace_dir=tmp_path / "workspaces",
        memory_dir=tmp_path / "memory",
        max_worker_concurrency=2,
        agents_dir=agents_dir or _repo_agents_dir(),
        skills_dir=repo_root / "data" / "skills",
        runtime_events_enabled=True,
        enable_coordinator_mode=False,
        planner_max_tasks=4,
        max_revision_rounds=4,
        session_hydrate_window_messages=40,
        session_transcript_page_size=100,
        default_tenant_id="local-dev",
        default_user_id="local-cli",
        default_conversation_id="local-session",
    )


def test_transcript_store_round_trips_session_state_and_transcript(tmp_path: Path):
    store = RuntimeTranscriptStore(RuntimePaths.from_settings(_make_runtime_settings(tmp_path)))
    session = SessionState(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id="conv-a",
    )
    session.append_message("user", "Hello runtime")
    session.scratchpad["note"] = "persist me"

    store.persist_session_state(session)
    store.append_session_transcript(session.session_id, {"kind": "message", "content": "Hello runtime"})

    loaded = store.load_session_state(session.session_id)
    transcript = store.load_session_transcript(session.session_id)

    assert loaded is not None
    assert loaded.session_id == session.session_id
    assert loaded.messages[0].content == "Hello runtime"
    assert loaded.scratchpad == {"note": "persist me"}
    assert transcript == [{"kind": "message", "content": "Hello runtime"}]


def test_transcript_store_keeps_jsonl_events_valid_under_parallel_appends(tmp_path: Path):
    store = RuntimeTranscriptStore(RuntimePaths.from_settings(_make_runtime_settings(tmp_path)))
    session_id = "tenant:user:parallel"

    def worker(worker_id: int) -> None:
        for event_index in range(15):
            store.append_session_event(
                RuntimeEvent(
                    event_type="tool_end",
                    session_id=session_id,
                    payload={"worker_id": worker_id, "event_index": event_index},
                )
            )

    threads = [threading.Thread(target=worker, args=(worker_id,)) for worker_id in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    events = store.load_session_events(session_id)
    assert len(events) == 90
    assert all(row["event_type"] == "tool_end" for row in events)


def test_transcript_store_persists_compact_window_and_history_metadata(tmp_path: Path):
    settings = _make_runtime_settings(tmp_path)
    settings.session_hydrate_window_messages = 4
    settings.session_transcript_page_size = 3
    store = RuntimeTranscriptStore(
        RuntimePaths.from_settings(settings),
        session_hydrate_window_messages=settings.session_hydrate_window_messages,
        session_transcript_page_size=settings.session_transcript_page_size,
    )
    session = SessionState(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id="conv-a",
    )
    for index in range(8):
        role = "user" if index % 2 == 0 else "assistant"
        session.append_message(role, f"message {index}")

    store.ensure_session_transcript_seeded(session.session_id, session.messages)
    store.persist_session_state(session)

    raw_state = json.loads((store.session_dir(session.session_id) / "state.json").read_text(encoding="utf-8"))
    recent = store.load_recent_session_messages(session.session_id, limit=4)

    assert [item["content"] for item in raw_state["messages"]] == [
        "message 4",
        "message 5",
        "message 6",
        "message 7",
    ]
    assert raw_state["metadata"]["history_total_messages"] == 8
    assert raw_state["metadata"]["history_stored_window_messages"] == 4
    assert raw_state["metadata"]["has_earlier_history"] is True
    assert len(store.load_session_transcript(session.session_id)) == 8
    assert [message.content for message in recent] == [
        "message 4",
        "message 5",
        "message 6",
        "message 7",
    ]


def test_transcript_store_pages_older_messages_without_duplication(tmp_path: Path):
    settings = _make_runtime_settings(tmp_path)
    settings.session_transcript_page_size = 3
    store = RuntimeTranscriptStore(
        RuntimePaths.from_settings(settings),
        session_hydrate_window_messages=settings.session_hydrate_window_messages,
        session_transcript_page_size=settings.session_transcript_page_size,
    )
    session = SessionState(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id="conv-a",
    )
    for index in range(8):
        session.append_message("assistant" if index % 2 else "user", f"message {index}")

    store.ensure_session_transcript_seeded(session.session_id, session.messages)

    page_one = store.load_session_message_page(session.session_id)
    page_two = store.load_session_message_page(
        session.session_id,
        before_message_index=page_one["next_before_message_index"],
    )
    page_three = store.load_session_message_page(
        session.session_id,
        before_message_index=page_two["next_before_message_index"],
    )

    assert [message.content for message in page_one["messages"]] == ["message 5", "message 6", "message 7"]
    assert [message.content for message in page_two["messages"]] == ["message 2", "message 3", "message 4"]
    assert [message.content for message in page_three["messages"]] == ["message 0", "message 1"]
    assert page_one["total_messages"] == 8
    assert page_three["next_before_message_index"] is None
    assert not {
        message.message_id for message in page_one["messages"]
    }.intersection({message.message_id for message in page_two["messages"]})


def test_transcript_store_loads_legacy_full_history_state_and_rewrites_compact_snapshot(tmp_path: Path):
    settings = _make_runtime_settings(tmp_path)
    settings.session_hydrate_window_messages = 3
    store = RuntimeTranscriptStore(
        RuntimePaths.from_settings(settings),
        session_hydrate_window_messages=settings.session_hydrate_window_messages,
        session_transcript_page_size=settings.session_transcript_page_size,
    )
    session = SessionState(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id="conv-a",
    )
    for index in range(6):
        session.append_message("assistant" if index % 2 else "user", f"legacy {index}")
    store.ensure_session_transcript_seeded(session.session_id, session.messages)

    state_path = store.session_dir(session.session_id) / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = store.load_session_state(session.session_id)
    assert loaded is not None
    assert [message.content for message in loaded.messages] == ["legacy 3", "legacy 4", "legacy 5"]
    assert loaded.metadata["history_total_messages"] == 6
    assert loaded.metadata["history_stored_window_messages"] == 3
    assert loaded.metadata["has_earlier_history"] is True

    store.persist_session_state(loaded)
    rewritten = json.loads(state_path.read_text(encoding="utf-8"))
    assert [item["content"] for item in rewritten["messages"]] == ["legacy 3", "legacy 4", "legacy 5"]
    assert rewritten["metadata"]["history_total_messages"] == 6


def test_job_manager_resumes_waiting_job_after_reinstantiation(tmp_path: Path):
    store = RuntimeTranscriptStore(RuntimePaths.from_settings(_make_runtime_settings(tmp_path)))
    manager_a = RuntimeJobManager(store)
    job = manager_a.create_job(
        agent_name="utility",
        prompt="original prompt",
        session_id="tenant:user:conv",
        description="resume test",
        metadata={"session_state": {"tenant_id": "tenant", "user_id": "user", "conversation_id": "conv"}},
    )
    job.status = "waiting_message"
    store.persist_job_state(job)

    queued = manager_a.enqueue_message(job.job_id, "continue with this")
    assert queued is not None

    manager_b = RuntimeJobManager(store)

    def runner(record):
        mailbox = manager_b.drain_mailbox(record.job_id)
        text = " | ".join(message.content for message in mailbox)
        return f"{record.prompt} :: {text}"

    resumed = manager_b.continue_job(job.job_id, runner)
    assert resumed is not None

    deadline = time.time() + 5
    completed = None
    while time.time() < deadline:
        completed = manager_b.get_job(job.job_id)
        if completed and completed.status == "completed":
            break
        time.sleep(0.05)

    assert completed is not None
    assert completed.status == "completed"
    assert completed.result_summary == "original prompt :: continue with this"
    assert Path(completed.artifact_dir).is_dir()


def test_agent_registry_applies_file_overrides(tmp_path: Path):
    definitions_dir = tmp_path / "agents"
    definitions_dir.mkdir()
    (definitions_dir / "general.md").write_text(
        """---
name: general
mode: react
description: overridden general agent
prompt_file: general_agent.md
skill_scope: general
allowed_tools: ["calculator"]
allowed_worker_agents: []
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 3
max_tool_calls: 4
allow_background_jobs: false
metadata: {"role_kind": "top_level"}
---
override
""",
        encoding="utf-8",
    )

    registry = AgentRegistry(definitions_dir)
    general = registry.get("general")

    assert general is not None
    assert general.description == "overridden general agent"
    assert general.allowed_tools == ["calculator"]
    assert general.max_steps == 3


def test_runtime_agent_matrix_matches_alignment_expectations(tmp_path: Path):
    registry = AgentRegistry(_make_runtime_settings(tmp_path).agents_dir)

    general = registry.get("general")
    coordinator = registry.get("coordinator")
    verifier = registry.get("verifier")
    memory_maintainer = registry.get("memory_maintainer")

    assert general is not None
    assert coordinator is not None
    assert verifier is not None
    assert memory_maintainer is not None

    assert "coordinator" in general.allowed_worker_agents
    assert "memory_maintainer" in general.allowed_worker_agents
    assert coordinator.mode == "coordinator"
    assert "planner" in coordinator.allowed_worker_agents
    assert "graph_manager" in coordinator.allowed_worker_agents
    assert verifier.prompt_file == "verifier_agent.md"
    assert memory_maintainer.metadata["role_kind"] == "maintenance"


def test_job_runner_persists_updated_worker_session_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    job = kernel.job_manager.create_job(
        agent_name="utility",
        prompt="original worker prompt",
        session_id="tenant:user:conv",
        description="worker state persistence",
        metadata={"session_state": {"tenant_id": "tenant", "user_id": "user", "conversation_id": "conv"}},
    )

    def fake_run_agent(agent, session_state, *, user_text, callbacks, task_payload=None):
        return SimpleNamespace(
            text="worker complete",
            messages=[
                RuntimeMessage(role="user", content=user_text),
                RuntimeMessage(role="assistant", content="worker complete"),
            ],
            metadata={},
        )

    monkeypatch.setattr(kernel, "run_agent", fake_run_agent)

    kernel._job_runner(job)

    refreshed = kernel.job_manager.get_job(job.job_id)
    assert refreshed is not None
    persisted = refreshed.metadata["session_state"]
    assert [item["content"] for item in persisted["messages"]] == [
        "original worker prompt",
        "worker complete",
    ]


def test_job_runner_re_resolves_access_snapshot_before_running_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class _AuthzService:
        def apply_access_snapshot(self, session_state, *, tenant_id, user_id, user_email, session_upload_collection_id="", display_name=""):
            del display_name
            snapshot = AccessSnapshot(
                tenant_id=tenant_id,
                user_id=user_id,
                user_email=user_email,
                auth_provider="email",
                principal_id=f"principal:{user_email}",
                role_ids=("role:finance",),
                session_upload_collection_id=session_upload_collection_id,
                authz_enabled=True,
                resources={
                    "collection": {"use": ["default"], "manage": [], "use_all": False, "manage_all": False},
                    "graph": {"use": [], "manage": [], "use_all": False, "manage_all": False},
                    "tool": {"use": ["calculator"], "manage": [], "use_all": False, "manage_all": False},
                    "skill_family": {"use": [], "manage": [], "use_all": False, "manage_all": False},
                },
            )
            summary = snapshot.to_summary()
            session_state.metadata = {
                **dict(session_state.metadata or {}),
                "user_email": user_email,
                "auth_provider": "email",
                "principal_id": snapshot.principal_id,
                "role_ids": ["role:finance"],
                "access_summary": summary,
            }
            session_state.user_email = user_email
            session_state.auth_provider = "email"
            session_state.principal_id = snapshot.principal_id
            session_state.access_summary = summary
            return snapshot

    settings = _make_runtime_settings(tmp_path)
    settings.default_collection_id = "default"
    kernel = RuntimeKernel(
        settings,
        providers=SimpleNamespace(),
        stores=SimpleNamespace(authorization_service=_AuthzService()),
    )
    job = kernel.job_manager.create_job(
        agent_name="utility",
        prompt="worker prompt",
        session_id="tenant:user:conv",
        description="worker access refresh",
        metadata={
            "session_state": {
                "tenant_id": "tenant",
                "user_id": "user",
                "conversation_id": "conv",
                "user_email": "analyst@example.com",
                "metadata": {},
            }
        },
    )
    seen: dict[str, object] = {}

    def fake_run_agent(agent, session_state, *, user_text, callbacks, task_payload=None):
        del agent, callbacks, task_payload
        seen["user_email"] = session_state.user_email
        seen["access_summary"] = dict(session_state.metadata.get("access_summary") or {})
        return SimpleNamespace(
            text="worker complete",
            messages=[
                RuntimeMessage(role="user", content=user_text),
                RuntimeMessage(role="assistant", content="worker complete"),
            ],
            metadata={},
        )

    monkeypatch.setattr(kernel, "run_agent", fake_run_agent)

    kernel._job_runner(job)

    assert seen["user_email"] == "analyst@example.com"
    assert seen["access_summary"]["authz_enabled"] is True
    assert seen["access_summary"]["resources"]["tool"]["use"] == ["calculator"]


def test_coordinator_prepares_and_consumes_typed_handoff_artifacts(tmp_path: Path) -> None:
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    result = kernel.coordinator_controller.build_task_result(
        SimpleNamespace(status="completed", last_error="", result_summary="Finance approval patterns and escalation paths", output_path="", metadata={}),
        kernel.coordinator_controller.build_worker_request(
            task={
                "id": "task_1",
                "title": "Analyze data",
                "executor": "data_analyst",
                "input": "Analyze workbook",
                "produces_artifacts": ["analysis_summary", "entity_candidates", "keyword_windows"],
                "handoff_schema": "analysis_to_rag",
            },
            user_request="Analyze workbook then search the KB for similar patterns.",
            session_state=session,
        ),
    )
    task = {
        "id": "task_1",
        "title": "Analyze data",
        "executor": "data_analyst",
        "input": "Analyze workbook",
        "produces_artifacts": ["analysis_summary", "entity_candidates", "keyword_windows"],
        "handoff_schema": "analysis_to_rag",
    }
    artifacts = kernel.coordinator_controller._prepare_handoff_artifacts(session_state=session, task=task, result=result)
    assert {artifact["artifact_type"] for artifact in artifacts} == {
        "analysis_summary",
        "entity_candidates",
        "keyword_windows",
    }

    rag_task = {
        "id": "task_2",
        "title": "Search KB",
        "executor": "rag_worker",
        "input": "Search the KB",
        "consumes_artifacts": ["analysis_summary", "entity_candidates"],
        "handoff_schema": "analysis_to_rag",
    }
    resolved = kernel.coordinator_controller._resolve_task_handoffs(
        task=rag_task,
        session_state=session,
        consumer_agent="rag_worker",
    )
    assert {artifact["artifact_type"] for artifact in resolved} >= {"analysis_summary", "entity_candidates"}


def test_coordinator_filters_invalid_handoff_payloads(tmp_path: Path) -> None:
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    register_handoff_artifact(
        session,
        artifact_type="analysis_summary",
        handoff_schema="analysis_to_rag",
        producer_task_id="task_1",
        producer_agent="data_analyst",
        data={},
        allowed_consumers=["rag_worker"],
    )

    resolved = kernel.coordinator_controller._resolve_task_handoffs(
        task={
            "id": "task_2",
            "title": "Search KB",
            "executor": "rag_worker",
            "consumes_artifacts": ["analysis_summary"],
            "handoff_schema": "analysis_to_rag",
        },
        session_state=session,
        consumer_agent="rag_worker",
    )

    assert resolved == []


def test_process_agent_turn_syncs_user_message_back_to_chat_session_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    session = ChatSession(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
    )

    def fail_run_agent(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(kernel, "run_agent", fail_run_agent)

    with pytest.raises(RuntimeError, match="boom"):
        kernel.process_agent_turn(session, user_text="persist this turn")

    assert [getattr(message, "content", "") for message in session.messages] == ["persist this turn"]
    stored = kernel.transcript_store.load_session_state(session.session_id)
    assert stored is not None
    assert [message.content for message in stored.messages] == ["persist this turn"]


def test_process_agent_turn_records_skill_telemetry_and_flags_low_quality_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class _TelemetrySkillStore:
        def __init__(self) -> None:
            self.records = {
                "skill-a-v1": SkillPackRecord(
                    skill_id="skill-a-v1",
                    version_parent="skill-a",
                    name="Skill A",
                    agent_scope="general",
                    checksum="a1",
                    tenant_id="tenant",
                    status="active",
                    enabled=True,
                )
            }
            self.events: list[dict[str, object]] = []

        def get_skill_pack(self, skill_id: str, *, tenant_id: str = "local-dev", owner_user_id: str = ""):
            del owner_user_id
            record = self.records.get(skill_id)
            if record is None or record.tenant_id != tenant_id:
                return None
            return record

        def append_skill_telemetry_event(self, event) -> None:
            self.events.append(event.to_dict())

        def list_skill_telemetry_events(
            self,
            *,
            tenant_id: str = "local-dev",
            skill_family_id: str = "",
            skill_id: str = "",
            session_id: str = "",
            limit: int = 200,
        ):
            rows = [
                row
                for row in self.events
                if row.get("tenant_id") == tenant_id
                and (not skill_family_id or row.get("skill_family_id") == skill_family_id)
                and (not skill_id or row.get("skill_id") == skill_id)
                and (not session_id or row.get("session_id") == session_id)
            ]
            return list(reversed(rows))[:limit]

    store = _TelemetrySkillStore()
    for index in range(9):
        store.append_skill_telemetry_event(
            SkillTelemetryEvent.build(
                tenant_id="tenant",
                skill_id="skill-a-v1",
                skill_family_id="skill-a",
                query=f"query-{index}",
                answer_quality="pass" if index < 7 else "revise",
                agent_name="general",
                session_id="tenant:user:historic",
            )
        )

    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(skill_store=store),
    )
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="conv")

    def fake_run_agent(agent, session_state, *, user_text, callbacks, task_payload=None, chat_max_output_tokens=None):
        del callbacks, task_payload, chat_max_output_tokens
        return SimpleNamespace(
            text="Needs revision",
            messages=list(session_state.messages)
            + [RuntimeMessage(role="assistant", content="Needs revision")],
            metadata={
                "verification": {
                    "status": "revise",
                    "summary": "needs work",
                    "issues": ["missing support"],
                    "feedback": "Add support.",
                },
                "skill_resolution": {
                    "matches": [
                        {
                            "skill_id": "skill-a-v1",
                            "skill_family_id": "skill-a",
                            "name": "Skill A",
                            "agent_scope": agent.name,
                            "chunk_index": 0,
                            "score": 1.0,
                        }
                    ],
                    "resolved_skill_families": ["skill-a"],
                    "warnings": [],
                },
            },
        )

    monkeypatch.setattr(kernel, "run_agent", fake_run_agent)

    text = kernel.process_agent_turn(session, user_text="check the answer")

    assert text == "Needs revision"
    assert store.events[-1]["skill_id"] == "skill-a-v1"
    assert store.events[-1]["skill_family_id"] == "skill-a"
    assert store.events[-1]["answer_quality"] == "revise"
    events = kernel.transcript_store.load_session_events(session.session_id)
    assert any(row["event_type"] == "skill_review_flagged" for row in events)


def test_process_agent_turn_merges_pending_clarification_into_resolved_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="conv")
    original_request = (
        "Across the whole default collection, identify the major subsystems in this codebase, "
        "list supporting documents for each subsystem, and verify that the final answer does not overclaim."
    )
    session.metadata = {
        "pending_clarification": {
            "question": "Which specific codebase or repository within the default collection should be examined?",
            "reason": "scope_selection",
            "options": ["all documents within the repository"],
        },
        "resolved_turn_intent": resolve_turn_intent(original_request, {"kb_collection_id": "default"}).to_dict(),
    }
    captured: dict[str, object] = {}

    def fake_run_agent(agent, session_state, *, user_text, callbacks, task_payload=None, chat_max_output_tokens=None):
        del callbacks, task_payload, chat_max_output_tokens
        captured["user_text"] = user_text
        captured["resolved_turn_intent"] = dict(session_state.metadata.get("resolved_turn_intent") or {})
        return SimpleNamespace(
            text="grounded answer",
            messages=list(session_state.messages) + [RuntimeMessage(role="assistant", content="grounded answer")],
            metadata={},
        )

    monkeypatch.setattr(kernel, "run_agent", fake_run_agent)

    kernel.process_agent_turn(session, user_text="all documents within the repository")

    assert "Across the whole default collection" in str(captured["user_text"])
    assert "Clarification resolution" in str(captured["user_text"])
    assert "all documents within the repository" in str(captured["user_text"])
    resolved_intent = dict(captured["resolved_turn_intent"] or {})
    assert resolved_intent["clarification_response"] == "all documents within the repository"
    assert resolved_intent["answer_contract"]["kind"] == "grounded_synthesis"


def test_openwebui_helper_turn_preserves_pending_graph_clarification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="graph-clarify")
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
    clarification = {
        "question": "What output format should I use?",
        "reason": "answer_format_selection",
        "options": ["Textual synthesis", "Diagram", "Table", "Mixed"],
        "source_agent": "graph_manager",
        "blocking": True,
    }
    session.metadata = {
        "route_context": {"semantic_routing": semantic_routing, "suggested_agent": "graph_manager"},
        "resolved_turn_intent": resolved,
        "pending_clarification": {
            **clarification,
            "resolved_turn_intent": resolved,
            "semantic_routing": semantic_routing,
            "selected_agent": "graph_manager",
        },
    }
    session.messages = [
        RuntimeMessage(role="user", content=original_request),
        RuntimeMessage(
            role="assistant",
            content="What output format should I use?",
            metadata={
                "agent_name": "graph_manager",
                "turn_outcome": "clarification_request",
                "clarification": clarification,
            },
        ),
    ]
    monkeypatch.setattr("agentic_chatbot_next.basic_chat.run_basic_chat", lambda *args, **kwargs: '{"title":"Graph request"}')

    kernel.process_basic_turn(
        session,
        user_text="Generate a concise, 3-5 word title for this chat.",
        system_prompt="helper",
        chat_llm=object(),
        route_metadata={"openwebui_helper_task_type": "title"},
        user_message_metadata={"openwebui_helper_task_type": "title", "openwebui_internal": True},
        assistant_message_metadata={"openwebui_helper_task_type": "title", "openwebui_internal": True},
        skip_post_turn_memory=True,
    )

    assert session.metadata["pending_clarification"]["reason"] == "answer_format_selection"
    assert session.metadata["pending_clarification"]["selected_agent"] == "graph_manager"
    assert session.metadata["resolved_turn_intent"]["normalized_user_objective"] == original_request


def test_process_agent_turn_merges_kb_collection_selection_clarification_into_resolved_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="conv")
    original_request = "Summarize the key risks in the knowledge base and cite the supporting documents."
    session.metadata = {
        "pending_clarification": {
            "question": "Which knowledge base collection should I use?",
            "reason": "kb_collection_selection",
            "options": ["default", "rfp-corpus"],
        },
        "resolved_turn_intent": resolve_turn_intent(original_request, {"kb_collection_id": "default"}).to_dict(),
    }
    captured: dict[str, object] = {}

    def fake_run_agent(agent, session_state, *, user_text, callbacks, task_payload=None, chat_max_output_tokens=None):
        del callbacks, task_payload, chat_max_output_tokens
        captured["user_text"] = user_text
        captured["resolved_turn_intent"] = dict(session_state.metadata.get("resolved_turn_intent") or {})
        return SimpleNamespace(
            text="grounded answer",
            messages=list(session_state.messages) + [RuntimeMessage(role="assistant", content="grounded answer")],
            metadata={},
        )

    monkeypatch.setattr(kernel, "run_agent", fake_run_agent)

    kernel.process_agent_turn(session, user_text="rfp-corpus")

    assert "Use knowledge base collection `rfp-corpus`." in str(captured["user_text"])
    resolved_intent = dict(captured["resolved_turn_intent"] or {})
    assert resolved_intent["clarification_response"] == "rfp-corpus"
    assert resolved_intent["requested_scope"]["collection_id"] == "rfp-corpus"
    assert resolved_intent["requested_scope"]["requested_kb_collection_id"] == "rfp-corpus"
    assert resolved_intent["requested_scope"]["kb_collection_confirmed"] is True


def test_process_agent_turn_merges_namespace_selection_clarification_into_resolved_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="conv")
    original_request = "Summarize the key risks in the selected namespace set and cite the supporting documents."
    session.metadata = {
        "pending_clarification": {
            "question": "I found multiple visible namespaces matching `rfp-corpus`. Tell me which ones to use.",
            "reason": "namespace_scope_selection",
            "options": ["rfp-corpus", "rfp_corpus", "collections only", "graphs only", "use all"],
        },
        "pending_namespace_candidates": {
            "namespace_query": "rfp-corpus",
            "collections": [
                {"namespace_id": "rfp-corpus", "display_name": "rfp-corpus", "score": 1.0, "exactness": "exact"}
            ],
            "graphs": [
                {
                    "namespace_id": "rfp_corpus",
                    "graph_id": "rfp_corpus",
                    "display_name": "RFP Corpus Graph",
                    "collection_id": "rfp-corpus",
                    "score": 0.96,
                    "exactness": "normalized_exact",
                }
            ],
        },
        "resolved_turn_intent": resolve_turn_intent(original_request, {"kb_collection_id": "default"}).to_dict(),
    }
    captured: dict[str, object] = {}

    def fake_run_agent(agent, session_state, *, user_text, callbacks, task_payload=None, chat_max_output_tokens=None):
        del callbacks, task_payload, chat_max_output_tokens
        captured["user_text"] = user_text
        captured["resolved_turn_intent"] = dict(session_state.metadata.get("resolved_turn_intent") or {})
        return SimpleNamespace(
            text="grounded answer",
            messages=list(session_state.messages) + [RuntimeMessage(role="assistant", content="grounded answer")],
            metadata={},
        )

    monkeypatch.setattr(kernel, "run_agent", fake_run_agent)

    kernel.process_agent_turn(session, user_text="use all")

    assert "Use knowledge base collections `rfp-corpus`." in str(captured["user_text"])
    assert "Use graphs `rfp_corpus`." in str(captured["user_text"])
    resolved_intent = dict(captured["resolved_turn_intent"] or {})
    assert resolved_intent["clarification_response"] == "use all"
    assert resolved_intent["requested_scope"]["search_collection_ids"] == ["rfp-corpus"]
    assert resolved_intent["requested_scope"]["graph_ids"] == ["rfp_corpus"]


def test_coordinator_recent_context_summary_excludes_openwebui_helper_messages(tmp_path: Path):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    session_state = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    session_state.append_message("user", "Actual user question")
    session_state.append_message(
        "assistant",
        "Generate a concise title for the conversation.",
        metadata={"openwebui_helper_task_type": "title", "openwebui_internal": True},
    )
    session_state.append_message("assistant", "Meaningful assistant response")

    summary = kernel.coordinator_controller.recent_context_summary(session_state, limit=4)

    assert "Actual user question" in summary
    assert "Meaningful assistant response" in summary
    assert "Generate a concise title" not in summary


def test_coordinator_runs_planner_workers_finalizer_and_verifier_with_scoped_worker_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    coordinator = kernel.registry.get("coordinator")
    assert coordinator is not None

    session_state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
    )
    session_state.append_message("assistant", "prior parent context")
    worker_histories = []

    def fake_run_agent(agent, session_state, *, user_text, callbacks, task_payload=None):
        if agent.mode == "planner":
            return SimpleNamespace(
                text="",
                messages=list(session_state.messages),
                metadata={
                    "planner_payload": {
                        "summary": "Run one worker task.",
                        "tasks": [
                            {
                                "id": "task_1",
                                "title": "Gather evidence",
                                "executor": "rag_worker",
                                "mode": "sequential",
                                "depends_on": [],
                                "input": "Find the relevant evidence.",
                                "doc_scope": ["contract-a"],
                                "skill_queries": ["citation hygiene"],
                            }
                        ],
                    }
                },
            )
        if agent.name == "rag_worker":
            worker_histories.append([message.content for message in session_state.messages])
            worker_request = dict((task_payload or {}).get("worker_request") or {})
            return SimpleNamespace(
                text=f"Evidence for {worker_request.get('task_id')}",
                messages=[RuntimeMessage(role="assistant", content=f"Evidence for {worker_request.get('task_id')}")],
                metadata={},
            )
        if agent.mode == "finalizer":
            assert task_payload is not None
            assert task_payload["execution_digest"]["task_summaries"][0]["output_excerpt"] == "Evidence for task_1"
            return SimpleNamespace(
                text="Final synthesized answer",
                messages=list(session_state.messages),
                metadata={},
            )
        if agent.mode == "verifier":
            return SimpleNamespace(
                text='{"status":"pass","summary":"verified","issues":[],"feedback":""}',
                messages=list(session_state.messages),
                metadata={"verification": {"status": "pass", "summary": "verified", "issues": [], "feedback": ""}},
            )
        raise AssertionError(f"Unexpected agent {agent.name}")

    monkeypatch.setattr(kernel, "run_agent", fake_run_agent)

    result = kernel._run_coordinator(
        coordinator,
        session_state,
        user_text="Compare the contracts and synthesize the answer.",
        callbacks=[],
    )

    assert result.text == "Final synthesized answer"
    assert worker_histories == [[]]


def test_coordinator_passes_planner_input_packet_to_planner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    coordinator = kernel.registry.get("coordinator")
    assert coordinator is not None

    query = (
        "Look through the document I uploaded and extract all clauses and associated redlines, "
        "then loop through each clause/redline and search the internal policy guidance collection. "
        "Return recommended buyer actions to write back to the supplier."
    )
    session_state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
        metadata={
            "uploaded_doc_ids": ["UPLOAD_123"],
            "requested_kb_collection_id": "internal policy guidance",
            "effective_capabilities": {
                "enabled_agents": ["coordinator", "planner", "general", "rag_worker", "finalizer", "verifier"],
                "enabled_tools": ["read_indexed_doc"],
                "enabled_skill_pack_ids": ["policy_guidance"],
                "permission_mode": "plan",
            },
        },
    )
    planner_payloads = []

    def fake_run_agent(agent, session_state, *, user_text, callbacks, task_payload=None):
        assert agent.mode == "planner"
        planner_payloads.append(dict(task_payload or {}))
        return SimpleNamespace(
            text="",
            messages=list(session_state.messages),
            metadata={"planner_payload": {"summary": "Preview only.", "tasks": []}},
        )

    monkeypatch.setattr(kernel, "run_agent", fake_run_agent)

    result = kernel._run_coordinator(
        coordinator,
        session_state,
        user_text=query,
        callbacks=[],
    )

    assert "Plan mode is enabled" in result.text
    packet = planner_payloads[0]["planner_input_packet"]
    assert packet["attachments"] == ["UPLOAD_123"]
    assert packet["selected_kb_collections"] == ["internal policy guidance"]
    assert packet["permission_mode"] == "plan"
    assert set(packet["available_agents"]) == {"coordinator", "general", "rag_worker", "planner", "finalizer", "verifier"}
    assert packet["available_tools"] == ["read_indexed_doc"]
    assert packet["available_skill_packs"] == ["policy_guidance"]
    assert "mixed_evidence_scopes" in packet["risk_flags"]
    assert "requires_per_item_loop" in packet["risk_flags"]


def test_spawn_worker_tool_blocks_background_launch_for_agents_that_disallow_it(tmp_path: Path) -> None:
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    coordinator = kernel.registry.get("coordinator")
    assert coordinator is not None
    session_state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
    )
    tool_context = ToolContext(
        settings=kernel.settings,
        providers=None,
        stores=None,
        session=session_state,
        paths=kernel.paths,
        active_definition=coordinator,
    )

    result = kernel.spawn_worker_from_tool(
        tool_context,
        prompt="Create a plan",
        agent_name="planner",
        description="planner task",
        run_in_background=True,
    )

    assert "does not allow background jobs" in str(result.get("error") or "")


class _ExecutableSkillStore:
    def __init__(self, record: SkillPackRecord) -> None:
        self.record = record

    def get_skill_pack(self, skill_id: str, **kwargs):
        del kwargs
        return self.record if skill_id == self.record.skill_id else None


def test_execute_skill_inline_returns_rendered_prompt(tmp_path: Path) -> None:
    settings = _make_runtime_settings(tmp_path)
    settings.executable_skills_enabled = True
    record = SkillPackRecord(
        skill_id="inline-skill",
        name="Inline Skill",
        agent_scope="utility",
        checksum="checksum",
        tenant_id="tenant",
        status="active",
        enabled=True,
        kind="executable",
        body_markdown="Handle {{input}} with {{ARGUMENTS_JSON}}.",
        execution_config={"context": "inline", "allowed_tools": [], "effort": "low"},
    )
    kernel = RuntimeKernel(
        settings,
        providers=SimpleNamespace(),
        stores=SimpleNamespace(skill_store=_ExecutableSkillStore(record)),
    )
    utility = kernel.registry.get("utility")
    assert utility is not None
    utility = replace(utility, allowed_tools=[*utility.allowed_tools, "execute_skill"])
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    tool_context = ToolContext(
        settings=settings,
        providers=None,
        stores=kernel.stores,
        session=session,
        paths=kernel.paths,
        kernel=kernel,
        active_definition=utility,
    )

    result = kernel.execute_skill_from_tool(
        tool_context,
        skill_id="inline-skill",
        input_text="the request",
        arguments={"mode": "quick"},
    )

    assert result["status"] == "inline_ready"
    assert result["context"] == "inline"
    assert "the request" in result["result"]
    assert "quick" in result["result"]
    assert result["effort"] == "low"


class _FakeSkillJobManager:
    def __init__(self) -> None:
        self.created: JobRecord | None = None

    def create_job(self, **kwargs) -> JobRecord:
        job = JobRecord(
            job_id="job_skill",
            session_id=str(kwargs.get("session_id") or ""),
            agent_name=str(kwargs.get("agent_name") or ""),
            status="queued",
            prompt=str(kwargs.get("prompt") or ""),
            tenant_id=str(kwargs.get("tenant_id") or ""),
            user_id=str(kwargs.get("user_id") or ""),
            priority=str(kwargs.get("priority") or "interactive"),
            queue_class=str(kwargs.get("queue_class") or "interactive"),
            description=str(kwargs.get("description") or ""),
            parent_job_id=str(kwargs.get("parent_job_id") or ""),
            session_state=dict(kwargs.get("session_state") or {}),
            metadata=dict(kwargs.get("metadata") or {}),
        )
        self.created = job
        return job

    def run_job_inline(self, job: JobRecord, runner) -> str:
        del runner
        job.status = "completed"
        self.created = job
        return "fork result"

    def get_job(self, job_id: str) -> JobRecord | None:
        if self.created is not None and self.created.job_id == job_id:
            return self.created
        return None


def test_execute_skill_fork_creates_synchronous_clipped_job(tmp_path: Path) -> None:
    settings = _make_runtime_settings(tmp_path)
    settings.executable_skills_enabled = True
    record = SkillPackRecord(
        skill_id="fork-skill",
        name="Fork Skill",
        agent_scope="general",
        checksum="checksum",
        tenant_id="tenant",
        status="active",
        enabled=True,
        kind="executable",
        body_markdown="Run the forked workflow for {{input}}.",
        execution_config={
            "context": "fork",
            "agent": "utility",
            "allowed_tools": ["calculator"],
            "model": "gpt-test",
            "effort": "medium",
            "max_steps": 2,
            "max_tool_calls": 3,
        },
    )
    kernel = RuntimeKernel(
        settings,
        providers=SimpleNamespace(),
        stores=SimpleNamespace(skill_store=_ExecutableSkillStore(record)),
    )
    kernel.job_manager = _FakeSkillJobManager()
    general = kernel.registry.get("general")
    assert general is not None
    general = replace(general, allowed_tools=[*general.allowed_tools, "execute_skill", "calculator"])
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    tool_context = ToolContext(
        settings=settings,
        providers=None,
        stores=kernel.stores,
        session=session,
        paths=kernel.paths,
        kernel=kernel,
        active_definition=general,
    )

    result = kernel.execute_skill_from_tool(
        tool_context,
        skill_id="fork-skill",
        input_text="the request",
    )

    assert result["status"] == "completed"
    assert result["context"] == "fork"
    assert result["job_id"] == "job_skill"
    assert result["result"] == "fork result"
    assert kernel.job_manager.created is not None
    payload = kernel.job_manager.created.metadata["skill_execution"]
    assert payload["allowed_tools"] == ["calculator"]
    assert payload["max_steps"] == 2
    assert payload["max_tool_calls"] == 3
    assert payload["model"] == "gpt-test"


def test_coordinator_revises_final_answer_when_verifier_requests_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    coordinator = kernel.registry.get("coordinator")
    assert coordinator is not None

    session_state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
    )
    finalizer_calls = []
    verifier_calls = []

    def fake_run_agent(agent, session_state, *, user_text, callbacks, task_payload=None):
        if agent.mode == "planner":
            return SimpleNamespace(
                text="",
                messages=list(session_state.messages),
                metadata={
                    "planner_payload": {
                        "summary": "Run one worker task.",
                        "tasks": [
                            {
                                "id": "task_1",
                                "title": "Compute answer",
                                "executor": "general",
                                "mode": "sequential",
                                "depends_on": [],
                                "input": "Produce the main answer.",
                                "doc_scope": [],
                                "skill_queries": [],
                            }
                        ],
                    }
                },
            )
        if agent.name == "general":
            return SimpleNamespace(
                text="Draft worker output",
                messages=[RuntimeMessage(role="assistant", content="Draft worker output")],
                metadata={},
            )
        if agent.mode == "finalizer":
            finalizer_calls.append(dict(task_payload or {}))
            answer = "Initial answer" if len(finalizer_calls) == 1 else "Revised answer"
            return SimpleNamespace(text=answer, messages=list(session_state.messages), metadata={})
        if agent.mode == "verifier":
            verifier_calls.append(dict(task_payload or {}))
            status = "revise" if len(verifier_calls) == 1 else "pass"
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "status": status,
                        "summary": "needs caveat" if status == "revise" else "verified",
                        "issues": ["missing caveat"] if status == "revise" else [],
                        "feedback": "Add the missing caveat." if status == "revise" else "",
                    }
                ),
                messages=list(session_state.messages),
                metadata={
                    "verification": {
                        "status": status,
                        "summary": "needs caveat" if status == "revise" else "verified",
                        "issues": ["missing caveat"] if status == "revise" else [],
                        "feedback": "Add the missing caveat." if status == "revise" else "",
                    }
                },
            )
        raise AssertionError(f"Unexpected agent {agent.name}")

    monkeypatch.setattr(kernel, "run_agent", fake_run_agent)

    result = kernel._run_coordinator(
        coordinator,
        session_state,
        user_text="Handle this carefully.",
        callbacks=[],
    )

    assert result.text == "Revised answer"
    assert len(finalizer_calls) == 2
    assert finalizer_calls[-1]["verification"]["status"] == "revise"
    assert result.metadata["verification"]["status"] == "pass"


def test_coordinator_passes_compact_execution_digest_to_finalizer_and_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    coordinator = kernel.registry.get("coordinator")
    assert coordinator is not None

    session_state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
        metadata={
            "resolved_turn_intent": resolve_turn_intent(
                "Across the whole default collection, identify the major subsystems in this repo.",
                {"kb_collection_id": "default"},
            ).to_dict()
        },
    )
    finalizer_payloads: list[dict[str, object]] = []
    verifier_payloads: list[dict[str, object]] = []

    def fake_run_agent(agent, session_state, *, user_text, callbacks, task_payload=None):
        del callbacks, user_text
        if agent.mode == "planner":
            return SimpleNamespace(
                text="",
                messages=list(session_state.messages),
                metadata={
                    "planner_payload": {
                        "summary": "Run one worker task.",
                        "tasks": [
                            {
                                "id": "task_1",
                                "title": "Gather evidence",
                                "executor": "general",
                                "mode": "sequential",
                                "depends_on": [],
                                "input": "Produce the main answer.",
                                "doc_scope": [],
                                "skill_queries": [],
                            }
                        ],
                    }
                },
            )
        if agent.name == "general":
            return SimpleNamespace(
                text="Worker output with grounded support",
                messages=[RuntimeMessage(role="assistant", content="Worker output with grounded support")],
                metadata={},
            )
        if agent.mode == "finalizer":
            finalizer_payloads.append(dict(task_payload or {}))
            return SimpleNamespace(text="Final answer", messages=list(session_state.messages), metadata={})
        if agent.mode == "verifier":
            verifier_payloads.append(dict(task_payload or {}))
            return SimpleNamespace(
                text='{"status":"pass","summary":"verified","issues":[],"feedback":""}',
                messages=list(session_state.messages),
                metadata={"verification": {"status": "pass", "summary": "verified", "issues": [], "feedback": ""}},
            )
        raise AssertionError(f"Unexpected agent {agent.name}")

    monkeypatch.setattr(kernel, "run_agent", fake_run_agent)

    result = kernel._run_coordinator(
        coordinator,
        session_state,
        user_text="Across the whole default collection, identify the major subsystems in this repo.",
        callbacks=[],
    )

    assert result.text == "Final answer"
    assert "execution_digest" in finalizer_payloads[0]
    assert "execution_digest" in verifier_payloads[0]
    assert "task_results" not in finalizer_payloads[0]
    assert finalizer_payloads[0]["execution_digest"]["task_summaries"][0]["title"] == "Gather evidence"
    assert verifier_payloads[0]["execution_digest"]["answer_contract"]["kind"] == "grounded_synthesis"


def test_coordinator_stops_revision_loop_when_verifier_output_is_unstructured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    coordinator = kernel.registry.get("coordinator")
    assert coordinator is not None

    session_state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
    )
    finalizer_calls: list[dict[str, object]] = []

    def fake_run_agent(agent, session_state, *, user_text, callbacks, task_payload=None):
        del callbacks, user_text
        if agent.mode == "planner":
            return SimpleNamespace(
                text="",
                messages=list(session_state.messages),
                metadata={
                    "planner_payload": {
                        "summary": "Run one worker task.",
                        "tasks": [
                            {
                                "id": "task_1",
                                "title": "Compute answer",
                                "executor": "general",
                                "mode": "sequential",
                                "depends_on": [],
                                "input": "Produce the answer.",
                                "doc_scope": [],
                                "skill_queries": [],
                            }
                        ],
                    }
                },
            )
        if agent.name == "general":
            return SimpleNamespace(
                text="Draft worker output",
                messages=[RuntimeMessage(role="assistant", content="Draft worker output")],
                metadata={},
            )
        if agent.mode == "finalizer":
            finalizer_calls.append(dict(task_payload or {}))
            return SimpleNamespace(text="Initial answer", messages=list(session_state.messages), metadata={})
        if agent.mode == "verifier":
            return SimpleNamespace(
                text="this is not valid json",
                messages=list(session_state.messages),
                metadata={},
            )
        raise AssertionError(f"Unexpected agent {agent.name}")

    monkeypatch.setattr(kernel, "run_agent", fake_run_agent)

    result = kernel._run_coordinator(
        coordinator,
        session_state,
        user_text="Handle this carefully.",
        callbacks=[],
    )

    assert result.text == "Initial answer"
    assert len(finalizer_calls) == 1
    assert result.metadata["verification"]["parse_failed"] is True
    assert result.metadata["revision_stop_reason"] == "verifier_parse_failed"


def test_coordinator_stops_when_verifier_feedback_repeats_without_material_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    coordinator = kernel.registry.get("coordinator")
    assert coordinator is not None

    session_state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
    )
    finalizer_calls: list[dict[str, object]] = []

    def fake_run_agent(agent, session_state, *, user_text, callbacks, task_payload=None):
        del callbacks, user_text
        if agent.mode == "planner":
            return SimpleNamespace(
                text="",
                messages=list(session_state.messages),
                metadata={
                    "planner_payload": {
                        "summary": "Run one worker task.",
                        "tasks": [
                            {
                                "id": "task_1",
                                "title": "Compute answer",
                                "executor": "general",
                                "mode": "sequential",
                                "depends_on": [],
                                "input": "Produce the answer.",
                                "doc_scope": [],
                                "skill_queries": [],
                            }
                        ],
                    }
                },
            )
        if agent.name == "general":
            return SimpleNamespace(
                text="Draft worker output",
                messages=[RuntimeMessage(role="assistant", content="Draft worker output")],
                metadata={},
            )
        if agent.mode == "finalizer":
            finalizer_calls.append(dict(task_payload or {}))
            return SimpleNamespace(text=f"Revision {len(finalizer_calls)}", messages=list(session_state.messages), metadata={})
        if agent.mode == "verifier":
            return SimpleNamespace(
                text='{"status":"revise","summary":"needs work","issues":["missing support"],"feedback":"Add more support."}',
                messages=list(session_state.messages),
                metadata={
                    "verification": {
                        "status": "revise",
                        "summary": "needs work",
                        "issues": ["missing support"],
                        "feedback": "Add more support.",
                    }
                },
            )
        raise AssertionError(f"Unexpected agent {agent.name}")

    monkeypatch.setattr(kernel, "run_agent", fake_run_agent)

    result = kernel._run_coordinator(
        coordinator,
        session_state,
        user_text="Handle this carefully.",
        callbacks=[],
    )

    assert result.text == "Revision 2"
    assert len(finalizer_calls) == 2
    assert result.metadata["revision_stop_reason"] == "repeated_verifier_feedback"


def test_coordinator_stops_revisions_at_max_revision_rounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    settings = _make_runtime_settings(tmp_path)
    settings.max_revision_rounds = 3
    kernel = RuntimeKernel(
        settings,
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    coordinator = kernel.registry.get("coordinator")
    assert coordinator is not None

    session_state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
    )
    finalizer_calls = []
    planner_calls = []
    worker_calls = []
    verifier_calls = []

    def fake_run_agent(agent, session_state, *, user_text, callbacks, task_payload=None):
        if agent.mode == "planner":
            planner_calls.append(user_text)
            return SimpleNamespace(
                text="",
                messages=list(session_state.messages),
                metadata={
                    "planner_payload": {
                        "summary": "Run one worker task.",
                        "tasks": [
                            {
                                "id": "task_1",
                                "title": "Compute answer",
                                "executor": "general",
                                "mode": "sequential",
                                "depends_on": [],
                                "input": "Produce the main answer.",
                                "doc_scope": [],
                                "skill_queries": [],
                            }
                        ],
                    }
                },
            )
        if agent.name == "general":
            worker_calls.append(user_text)
            return SimpleNamespace(
                text="Draft worker output",
                messages=[RuntimeMessage(role="assistant", content="Draft worker output")],
                metadata={},
            )
        if agent.mode == "finalizer":
            finalizer_calls.append(dict(task_payload or {}))
            return SimpleNamespace(
                text=f"Revision {len(finalizer_calls)}",
                messages=list(session_state.messages),
                metadata={},
            )
        if agent.mode == "verifier":
            verifier_calls.append(dict(task_payload or {}))
            feedback = f"Add more support round {len(verifier_calls)}."
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "status": "revise",
                        "summary": "needs work",
                        "issues": ["missing support"],
                        "feedback": feedback,
                    }
                ),
                messages=list(session_state.messages),
                metadata={
                    "verification": {
                        "status": "revise",
                        "summary": "needs work",
                        "issues": ["missing support"],
                        "feedback": feedback,
                    }
                },
            )
        raise AssertionError(f"Unexpected agent {agent.name}")

    monkeypatch.setattr(kernel, "run_agent", fake_run_agent)

    result = kernel._run_coordinator(
        coordinator,
        session_state,
        user_text="Handle this carefully.",
        callbacks=[],
    )

    assert result.text == "Revision 3"
    assert len(planner_calls) == 1
    assert len(worker_calls) == 1
    assert len(finalizer_calls) == 3
    assert len(verifier_calls) == 3
    assert result.metadata["verification"]["revision_limit_reached"] is True
    assert result.metadata["revision_rounds_used"] == 3
    events = kernel.transcript_store.load_session_events(session_state.session_id)
    assert any(row["event_type"] == "coordinator_revision_limit_reached" for row in events)


def test_build_worker_request_preserves_structured_rag_hints(tmp_path: Path):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    session_state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
    )

    worker_request = kernel._build_worker_request(
        task={
            "id": "task_1",
            "title": "Discover workflow docs",
            "executor": "rag_worker",
            "mode": "parallel",
            "input": "Identify all documents with process flows.",
            "doc_scope": [],
            "skill_queries": ["corpus discovery", "process flow identification"],
            "research_profile": "corpus_discovery",
            "coverage_goal": "corpus_wide",
            "result_mode": "inventory",
            "answer_mode": "evidence_only",
            "controller_hints": {
                "prefer_inventory_output": True,
                "round_budget": 2,
                "retrieval_strategies": ["hybrid", "keyword"],
            },
        },
        user_request="Identify all documents with process flows.",
        session_state=session_state,
        artifact_refs=[],
    )

    assert worker_request.research_profile == "corpus_discovery"
    assert worker_request.coverage_goal == "corpus_wide"
    assert worker_request.result_mode == "inventory"
    assert worker_request.answer_mode == "evidence_only"
    assert worker_request.controller_hints == {
        "prefer_inventory_output": True,
        "round_budget": 2,
        "retrieval_strategies": ["hybrid", "keyword"],
    }
    assert worker_request.semantic_query == "Identify all documents with process flows."
    assert worker_request.instruction_prompt == worker_request.prompt
    assert worker_request.context_summary == ""
    assert worker_request.metadata["answer_mode"] == "evidence_only"
    assert worker_request.metadata["semantic_query"] == "Identify all documents with process flows."
    assert worker_request.metadata["instruction_prompt"] == worker_request.prompt
    assert worker_request.metadata["rag_search_task"]["round_budget"] == 2
    assert worker_request.metadata["rag_search_task"]["strategies"] == ["hybrid", "keyword"]


def test_build_worker_request_adds_strict_full_read_prompt_for_active_doc_focus_summary(tmp_path: Path):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    session_state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
    )

    worker_request = kernel._build_worker_request(
        task={
            "id": "task_1",
            "title": "Digest ARCHITECTURE.md",
            "executor": "general",
            "mode": "parallel",
            "input": "Inspect the exact scoped document and produce a subsystem digest.",
            "doc_scope": ["KB_architecture"],
            "skill_queries": [],
            "research_profile": "",
            "coverage_goal": "",
            "result_mode": "answer",
            "answer_mode": "answer",
            "controller_hints": {
                "summary_scope": "active_doc_focus",
                "strict_doc_focus": True,
                "doc_read_depth": "full",
                "final_output_mode": "detailed_subsystem_summary",
            },
        },
        user_request="Summarize the candidate docs in detail.",
        session_state=session_state,
        artifact_refs=[],
    )

    assert "STRICT_SCOPE_RULE" in worker_request.prompt
    assert "READ_DEPTH_RULE" in worker_request.prompt
    assert 'read_indexed_doc(mode="full")' in worker_request.prompt


def test_coordinator_expands_research_facets_into_parallel_rag_tasks(tmp_path: Path):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    session_state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
        metadata={"kb_collection_id": "default"},
    )
    register_handoff_artifact(
        session_state,
        artifact_type="research_facets",
        handoff_schema="research_inventory",
        producer_task_id="task_2",
        producer_agent="general",
        summary="seed facets",
        allowed_consumers=["coordinator", "finalizer", "rag_worker", "general"],
        data={
            "facets": [
                {
                    "name": "Runtime service",
                    "aliases": ["service", "RuntimeService"],
                    "rationale": "Top-level service boundary",
                    "seed_doc_ids": ["KB_architecture"],
                },
                {
                    "name": "Router",
                    "aliases": ["routing", "route_turn"],
                    "rationale": "Entry-point routing logic",
                    "seed_doc_ids": ["KB_router"],
                },
            ],
            "seed_documents": [{"doc_id": "KB_architecture", "title": "ARCHITECTURE.md"}],
            "scope_collection_id": "default",
        },
    )
    execution_state = TaskExecutionState(
        user_request="Identify documents that describe the major subsystems in this repo.",
        planner_summary="research campaign",
        task_plan=[
            {
                "id": "task_1",
                "title": "Seed corpus scan",
                "executor": "rag_worker",
                "mode": "sequential",
                "answer_mode": "evidence_only",
            },
            {
                "id": "task_2",
                "title": "Inspect seed docs",
                "executor": "general",
                "mode": "sequential",
                "depends_on": ["task_1"],
            },
            {
                "id": "task_3",
                "title": "Expand facet searches",
                "executor": "general",
                "mode": "sequential",
                "depends_on": ["task_2"],
                "consumes_artifacts": ["research_facets"],
                "handoff_schema": "research_inventory",
                "controller_hints": {
                    "dynamic_facet_fanout": True,
                    "max_parallel_facets": 4,
                    "retrieval_scope_mode": "kb_only",
                    "final_output_mode": "document_titles_only",
                },
            },
        ],
    )

    result = kernel.coordinator_controller._expand_research_facet_fanout(
        execution_state=execution_state,
        session_state=session_state,
        task=execution_state.task_plan[2],
    )

    generated = execution_state.task_plan[3:]
    assert result.status == "completed"
    assert len(generated) == 2
    assert all(task["executor"] == "rag_worker" for task in generated)
    assert all(task["mode"] == "parallel" for task in generated)
    assert all(task["answer_mode"] == "evidence_only" for task in generated)
    assert all(task["controller_hints"]["search_collection_ids"] == ["default"] for task in generated)


def test_coordinator_expands_subsystem_backfill_into_parallel_rag_tasks(tmp_path: Path):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    session_state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
        metadata={"kb_collection_id": "default"},
    )
    register_handoff_artifact(
        session_state,
        artifact_type="subsystem_inventory",
        handoff_schema="active_doc_focus_summary",
        producer_task_id="task_4",
        producer_agent="general",
        summary="subsystem inventory",
        allowed_consumers=["coordinator", "finalizer", "rag_worker", "general"],
        data={
            "subsystems": [
                {
                    "name": "Runtime kernel",
                    "aliases": ["RuntimeKernel"],
                    "description": "Coordinates service and query loop execution.",
                    "supporting_documents": [{"doc_id": "KB_architecture", "title": "ARCHITECTURE.md"}],
                    "supporting_citation_ids": ["KB_architecture#chunk0001"],
                    "coverage": "thin",
                },
                {
                    "name": "Gateway",
                    "aliases": ["OpenAI gateway"],
                    "description": "Exposes API entrypoints.",
                    "supporting_documents": [
                        {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md"},
                        {"doc_id": "KB_gateway", "title": "OPENAI_GATEWAY.md"},
                    ],
                    "supporting_citation_ids": ["KB_architecture#chunk0002", "KB_gateway#chunk0001"],
                    "coverage": "strong",
                },
            ],
            "source_documents": [
                {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md"},
                {"doc_id": "KB_gateway", "title": "OPENAI_GATEWAY.md"},
            ],
            "scope_collection_id": "default",
        },
    )
    execution_state = TaskExecutionState(
        user_request="Summarize the major subsystems from the candidate docs.",
        planner_summary="active doc focus summary",
        task_plan=[
            {"id": "task_1", "title": "Digest ARCHITECTURE.md", "executor": "general", "mode": "parallel"},
            {"id": "task_2", "title": "Digest OPENAI_GATEWAY.md", "executor": "general", "mode": "parallel"},
            {"id": "task_3", "title": "Digest CONTROL_FLOW.md", "executor": "general", "mode": "parallel"},
            {"id": "task_4", "title": "Consolidate subsystem inventory", "executor": "general", "mode": "sequential"},
            {
                "id": "task_5",
                "title": "Expand subsystem evidence backfill",
                "executor": "general",
                "mode": "sequential",
                "depends_on": ["task_4"],
                "consumes_artifacts": ["subsystem_inventory"],
                "handoff_schema": "active_doc_focus_summary",
                "controller_hints": {
                    "dynamic_subsystem_backfill": True,
                    "max_parallel_subsystems": 4,
                    "retrieval_scope_mode": "kb_only",
                    "strict_kb_scope": True,
                    "summary_scope": "active_doc_focus",
                    "final_output_mode": "detailed_subsystem_summary",
                    "search_collection_ids": ["default"],
                },
            },
        ],
    )

    result = kernel.coordinator_controller._expand_subsystem_backfill(
        execution_state=execution_state,
        session_state=session_state,
        task=execution_state.task_plan[4],
    )

    generated = execution_state.task_plan[5:]
    assert result.status == "completed"
    assert len(generated) == 1
    assert generated[0]["executor"] == "rag_worker"
    assert generated[0]["answer_mode"] == "evidence_only"
    assert generated[0]["doc_scope"] == ["KB_architecture", "KB_gateway"]
    assert generated[0]["controller_hints"]["subsystem_name"] == "Runtime kernel"
    assert generated[0]["controller_hints"]["summary_scope"] == "active_doc_focus"


def test_coordinator_expands_doc_review_into_ranked_parallel_general_tasks(tmp_path: Path):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    session_state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
        metadata={"kb_collection_id": "default"},
    )
    register_handoff_artifact(
        session_state,
        artifact_type="title_candidates",
        handoff_schema="research_inventory",
        producer_task_id="task_1",
        producer_agent="general",
        summary="title candidates",
        allowed_consumers=["coordinator", "finalizer", "rag_worker", "general"],
        data={
            "documents": [
                {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md", "source_path": "/docs/ARCHITECTURE.md", "match_reason": "metadata_title", "score": 0.95},
                {"doc_id": "KB_control", "title": "CONTROL_FLOW.md", "source_path": "/docs/CONTROL_FLOW.md", "match_reason": "metadata_title", "score": 0.92},
                {"doc_id": "KB_gateway", "title": "OPENAI_GATEWAY.md", "source_path": "/docs/OPENAI_GATEWAY.md", "match_reason": "metadata_title", "score": 0.90},
                {"doc_id": "KB_router", "title": "ROUTER_RUBRIC.md", "source_path": "/docs/ROUTER_RUBRIC.md", "match_reason": "metadata_title", "score": 0.89},
                {"doc_id": "KB_jobs", "title": "JOBS.md", "source_path": "/docs/JOBS.md", "match_reason": "metadata_title", "score": 0.86},
                {"doc_id": "KB_appendix", "title": "APPENDIX.md", "source_path": "/docs/APPENDIX.md", "match_reason": "metadata_title", "score": 0.85},
            ],
            "query_variants": ["major subsystems", "runtime architecture"],
            "scope_collection_id": "default",
        },
    )
    register_handoff_artifact(
        session_state,
        artifact_type="doc_focus",
        handoff_schema="research_inventory",
        producer_task_id="task_2",
        producer_agent="rag_worker",
        summary="seed docs",
        allowed_consumers=["coordinator", "finalizer", "rag_worker", "general"],
        data={
            "documents": [
                {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md"},
                {"doc_id": "KB_control", "title": "CONTROL_FLOW.md"},
                {"doc_id": "KB_gateway", "title": "OPENAI_GATEWAY.md"},
            ]
        },
    )
    register_handoff_artifact(
        session_state,
        artifact_type="research_facets",
        handoff_schema="research_inventory",
        producer_task_id="task_3",
        producer_agent="general",
        summary="facets",
        allowed_consumers=["coordinator", "finalizer", "rag_worker", "general"],
        data={
            "facets": [
                {"name": "runtime", "aliases": ["execution"], "rationale": "core runtime flow", "seed_doc_ids": ["KB_architecture", "KB_control", "KB_gateway"]},
                {"name": "routing", "aliases": ["router"], "rationale": "routing policy", "seed_doc_ids": ["KB_architecture", "KB_router"]},
                {"name": "jobs", "aliases": ["background jobs"], "rationale": "job lifecycle", "seed_doc_ids": ["KB_jobs"]},
            ],
            "seed_documents": [
                {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md"},
                {"doc_id": "KB_control", "title": "CONTROL_FLOW.md"},
                {"doc_id": "KB_gateway", "title": "OPENAI_GATEWAY.md"},
            ],
            "review_documents": [
                {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md"},
                {"doc_id": "KB_control", "title": "CONTROL_FLOW.md"},
                {"doc_id": "KB_gateway", "title": "OPENAI_GATEWAY.md"},
                {"doc_id": "KB_router", "title": "ROUTER_RUBRIC.md"},
                {"doc_id": "KB_jobs", "title": "JOBS.md"},
                {"doc_id": "KB_appendix", "title": "APPENDIX.md"},
            ],
            "scope_collection_id": "default",
        },
    )
    for task_id, facet_name, documents in (
        ("task_4_runtime", "runtime", ["KB_architecture", "KB_control", "KB_gateway"]),
        ("task_4_routing", "routing", ["KB_architecture", "KB_router"]),
        ("task_4_jobs", "jobs", ["KB_jobs"]),
    ):
        register_handoff_artifact(
            session_state,
            artifact_type="facet_matches",
            handoff_schema="research_inventory",
            producer_task_id=task_id,
            producer_agent="rag_worker",
            summary=f"{facet_name} docs",
            allowed_consumers=["coordinator", "finalizer", "general"],
            data={
                "facet": facet_name,
                "documents": [{"doc_id": doc_id, "title": doc_id.replace("KB_", "").replace("_", " ") + ".md"} for doc_id in documents],
                "rationale": f"{facet_name} facet",
                "supporting_citation_ids": [f"{documents[0]}#chunk0001"],
            },
        )
    execution_state = TaskExecutionState(
        user_request="Look through the candidate docs and give me a detailed architecture summary.",
        planner_summary="research campaign",
        task_plan=[
            {"id": "task_1", "title": "Scan title and path candidates", "executor": "general", "mode": "sequential"},
            {"id": "task_2", "title": "Seed corpus scan", "executor": "rag_worker", "mode": "sequential"},
            {"id": "task_3", "title": "Inspect seed documents and extract research facets", "executor": "general", "mode": "sequential"},
            {"id": "task_4_runtime", "title": "Search facet: runtime", "executor": "rag_worker", "mode": "parallel"},
            {"id": "task_4_routing", "title": "Search facet: routing", "executor": "rag_worker", "mode": "parallel"},
            {"id": "task_4_jobs", "title": "Search facet: jobs", "executor": "rag_worker", "mode": "parallel"},
            {
                "id": "task_5",
                "title": "Expand document review",
                "executor": "general",
                "mode": "sequential",
                "depends_on": ["task_4_runtime", "task_4_routing", "task_4_jobs"],
                "consumes_artifacts": ["title_candidates", "doc_focus", "research_facets", "facet_matches"],
                "handoff_schema": "research_inventory",
                "controller_hints": {
                    "dynamic_doc_review_fanout": True,
                    "max_parallel_doc_reviews": 4,
                    "max_optional_doc_reviews": 2,
                    "retrieval_scope_mode": "kb_only",
                    "strict_kb_scope": True,
                    "kb_collection_id": "default",
                    "search_collection_ids": ["default"],
                },
            },
        ],
    )

    result = kernel.coordinator_controller._expand_doc_review_fanout(
        execution_state=execution_state,
        session_state=session_state,
        task=execution_state.task_plan[6],
    )

    generated = execution_state.task_plan[7:]
    generated_titles = [task["title"] for task in generated]
    assert result.status == "completed"
    assert len(generated) == 5
    assert all(task["executor"] == "general" for task in generated)
    assert all(task["mode"] == "parallel" for task in generated)
    assert all(task["produces_artifacts"] == ["doc_digest"] for task in generated)
    assert all(task["controller_hints"]["strict_doc_focus"] is True for task in generated)
    assert all(task["controller_hints"]["doc_read_depth"] == "full" for task in generated)
    assert "Review ARCHITECTURE.md" in generated_titles
    assert "Review JOBS.md" in generated_titles
    assert "Review APPENDIX.md" not in generated_titles


def test_kernel_syncs_active_doc_focus_from_latest_assistant_metadata(tmp_path: Path):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
    )
    state.append_message(
        "assistant",
        "Candidate docs ready.",
        metadata={
            "doc_focus_result": {
                "collection_id": "default",
                "documents": [
                    {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md"},
                    {"doc_id": "KB_gateway", "title": "OPENAI_GATEWAY.md"},
                ],
                "source_query": "Identify candidate docs",
                "result_mode": "inventory",
            }
        },
    )

    kernel._sync_active_doc_focus(state)

    assert state.metadata["active_doc_focus"]["collection_id"] == "default"
    assert [item["doc_id"] for item in state.metadata["active_doc_focus"]["documents"]] == [
        "KB_architecture",
        "KB_gateway",
    ]
    assert state.metadata["active_doc_focus"]["message_id"] == state.messages[-1].message_id


def test_coordinator_doc_focus_result_prefers_documents_rendered_in_final_answer(tmp_path: Path):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    session_state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
        metadata={"kb_collection_id": "default"},
    )
    register_handoff_artifact(
        session_state,
        artifact_type="facet_matches",
        handoff_schema="research_inventory",
        producer_task_id="task_3_router",
        producer_agent="rag_worker",
        summary="routing docs",
        allowed_consumers=["coordinator", "finalizer", "general"],
        data={
            "facet": "routing",
            "documents": [
                {"doc_id": "KB_router", "title": "ROUTER_RUBRIC.md"},
                {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md"},
                {"doc_id": "KB_noise", "title": "TEST_QUERIES.md"},
            ],
            "rationale": "routing facet",
            "supporting_citation_ids": ["KB_router#chunk0001"],
        },
    )
    register_handoff_artifact(
        session_state,
        artifact_type="facet_matches",
        handoff_schema="research_inventory",
        producer_task_id="task_3_runtime",
        producer_agent="rag_worker",
        summary="runtime docs",
        allowed_consumers=["coordinator", "finalizer", "general"],
        data={
            "facet": "runtime",
            "documents": [
                {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md"},
                {"doc_id": "KB_control", "title": "CONTROL_FLOW.md"},
            ],
            "rationale": "runtime facet",
            "supporting_citation_ids": ["KB_control#chunk0001"],
        },
    )
    execution_state = TaskExecutionState(
        user_request="Identify subsystem docs.",
        planner_summary="research campaign",
        task_plan=[
            {
                "id": "task_3_router",
                "controller_hints": {"final_output_mode": "document_titles_only"},
            },
            {
                "id": "task_3_runtime",
                "controller_hints": {"final_output_mode": "document_titles_only"},
            },
        ],
    )

    payload = kernel.coordinator_controller._coordinator_doc_focus_result(
        session_state=session_state,
        execution_state=execution_state,
        source_query="Identify subsystem docs.",
        rendered_answer=(
            "Potential KB documents\n"
            "1. ARCHITECTURE.md (KB_architecture)\n"
            "2. CONTROL_FLOW.md (KB_control)\n"
        ),
    )

    assert payload is not None
    assert [item["doc_id"] for item in payload["documents"]] == ["KB_architecture", "KB_control"]


def test_coordinator_doc_focus_result_excludes_irrelevant_reviewed_documents(tmp_path: Path):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    session_state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
        metadata={"kb_collection_id": "default"},
    )
    register_handoff_artifact(
        session_state,
        artifact_type="title_candidates",
        handoff_schema="research_inventory",
        producer_task_id="task_1",
        producer_agent="general",
        summary="title candidates",
        allowed_consumers=["coordinator", "finalizer", "rag_worker", "general"],
        data={
            "documents": [
                {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md", "source_path": "/docs/ARCHITECTURE.md", "match_reason": "metadata_title", "score": 0.95},
                {"doc_id": "KB_noise", "title": "TEST_QUERIES.md", "source_path": "/docs/TEST_QUERIES.md", "match_reason": "metadata_title", "score": 0.90},
            ],
            "query_variants": ["major subsystems"],
            "scope_collection_id": "default",
        },
    )
    register_handoff_artifact(
        session_state,
        artifact_type="doc_digest",
        handoff_schema="research_inventory",
        producer_task_id="task_5_architecture",
        producer_agent="general",
        summary="architecture review",
        allowed_consumers=["coordinator", "finalizer", "general"],
        data={
            "document": {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md"},
            "document_summary": "Primary architecture overview.",
            "subsystems": [],
            "relevance": "relevant",
            "relevance_rationale": "Directly explains the subsystem layout.",
            "matched_facets": ["runtime", "routing"],
            "coverage": "primary",
            "used_citation_ids": ["KB_architecture#chunk0001"],
        },
    )
    register_handoff_artifact(
        session_state,
        artifact_type="doc_digest",
        handoff_schema="research_inventory",
        producer_task_id="task_5_noise",
        producer_agent="general",
        summary="noise review",
        allowed_consumers=["coordinator", "finalizer", "general"],
        data={
            "document": {"doc_id": "KB_noise", "title": "TEST_QUERIES.md"},
            "document_summary": "Examples only.",
            "subsystems": [],
            "relevance": "irrelevant",
            "relevance_rationale": "Does not describe the architecture.",
            "matched_facets": [],
            "coverage": "thin",
            "used_citation_ids": ["KB_noise#chunk0001"],
        },
    )
    execution_state = TaskExecutionState(
        user_request="Look through the candidate docs and give me a detailed architecture summary.",
        planner_summary="research campaign",
        task_plan=[
            {"id": "task_1", "controller_hints": {"final_output_mode": "detailed_subsystem_summary"}},
            {"id": "task_5_architecture", "controller_hints": {"final_output_mode": "detailed_subsystem_summary"}},
            {"id": "task_5_noise", "controller_hints": {"final_output_mode": "detailed_subsystem_summary"}},
        ],
    )

    payload = kernel.coordinator_controller._coordinator_doc_focus_result(
        session_state=session_state,
        execution_state=execution_state,
        source_query="Look through the candidate docs and give me a detailed architecture summary.",
        rendered_answer="Architecture summary grounded in ARCHITECTURE.md.",
    )

    assert payload is not None
    assert [item["doc_id"] for item in payload["documents"]] == ["KB_architecture"]


def test_research_coverage_ledger_demotes_meta_documents_and_flags_thin_coverage(tmp_path: Path):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    query = "Investigate the major subsystems in this repo and give me a concise architectural walkthrough with citations."
    intent = resolve_turn_intent(query, {"kb_collection_id": "default"})
    session_state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
        metadata={
            "kb_collection_id": "default",
            "resolved_turn_intent": intent.to_dict(),
        },
    )
    register_handoff_artifact(
        session_state,
        artifact_type="title_candidates",
        handoff_schema="research_inventory",
        producer_task_id="task_1",
        producer_agent="general",
        summary="title candidates",
        allowed_consumers=["coordinator", "finalizer", "rag_worker", "general"],
        data={
            "documents": [
                {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md", "source_path": "/docs/ARCHITECTURE.md", "match_reason": "metadata_title", "score": 0.95},
                {"doc_id": "KB_tests", "title": "TEST_QUERIES.md", "source_path": "/docs/TEST_QUERIES.md", "match_reason": "exact_prompt_match", "score": 0.99},
            ],
            "query_variants": ["major subsystems"],
            "scope_collection_id": "default",
        },
    )
    register_handoff_artifact(
        session_state,
        artifact_type="doc_digest",
        handoff_schema="research_inventory",
        producer_task_id="task_5_architecture",
        producer_agent="general",
        summary="architecture review",
        allowed_consumers=["coordinator", "finalizer", "general"],
        data={
            "document": {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md", "source_path": "/docs/ARCHITECTURE.md"},
            "document_summary": "Primary architecture overview.",
            "subsystems": [],
            "relevance": "relevant",
            "relevance_rationale": "Directly explains the subsystem layout.",
            "matched_facets": ["runtime"],
            "coverage": "primary",
            "used_citation_ids": ["KB_architecture#chunk0001"],
        },
    )
    register_handoff_artifact(
        session_state,
        artifact_type="doc_digest",
        handoff_schema="research_inventory",
        producer_task_id="task_5_tests",
        producer_agent="general",
        summary="test query review",
        allowed_consumers=["coordinator", "finalizer", "general"],
        data={
            "document": {"doc_id": "KB_tests", "title": "TEST_QUERIES.md", "source_path": "/docs/TEST_QUERIES.md"},
            "document_summary": "Prompt catalog with expected paths.",
            "subsystems": [],
            "relevance": "relevant",
            "relevance_rationale": "Mentions the user prompt but is not architecture evidence.",
            "matched_facets": ["runtime"],
            "coverage": "thin",
            "used_citation_ids": ["KB_tests#chunk0006"],
        },
    )
    execution_state = TaskExecutionState(
        user_request=query,
        planner_summary="research campaign",
        task_plan=[
            {"id": "task_1", "controller_hints": {"final_output_mode": "detailed_subsystem_summary"}},
            {"id": "task_5_architecture", "controller_hints": {"final_output_mode": "detailed_subsystem_summary"}},
            {"id": "task_5_tests", "controller_hints": {"final_output_mode": "detailed_subsystem_summary"}},
        ],
    )

    ledger = kernel.coordinator_controller._build_research_coverage_ledger(
        session_state=session_state,
        execution_state=execution_state,
    )
    issues = kernel.coordinator_controller._coverage_gate_issues(
        session_state=session_state,
        execution_state=execution_state,
        ledger=ledger,
    )

    assert ledger["primary_source_count"] == 1
    assert ledger["meta_source_count"] == 1
    assert ledger["coverage_state"] == "insufficient"
    assert any(item["title"] == "TEST_QUERIES.md" and item["is_meta_document"] for item in ledger["candidate_documents"])
    assert "primary source documents" in " ".join(issues)
    assert "Meta/test prompt documents" in " ".join(issues)


def test_doc_review_fanout_demotes_meta_documents_when_primary_candidates_exist(tmp_path: Path):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    session_state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
        metadata={"kb_collection_id": "default"},
    )
    register_handoff_artifact(
        session_state,
        artifact_type="title_candidates",
        handoff_schema="research_inventory",
        producer_task_id="task_1",
        producer_agent="general",
        summary="title candidates",
        allowed_consumers=["coordinator", "finalizer", "rag_worker", "general"],
        data={
            "documents": [
                {"doc_id": "KB_tests", "title": "TEST_QUERIES.md", "source_path": "/docs/TEST_QUERIES.md", "match_reason": "exact_prompt_match", "score": 0.99},
                {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md", "source_path": "/docs/ARCHITECTURE.md", "match_reason": "metadata_title", "score": 0.80},
                {"doc_id": "KB_control", "title": "CONTROL_FLOW.md", "source_path": "/docs/CONTROL_FLOW.md", "match_reason": "metadata_title", "score": 0.78},
            ],
            "query_variants": ["major subsystems"],
            "scope_collection_id": "default",
        },
    )
    execution_state = TaskExecutionState(
        user_request="Investigate the major subsystems in this repo and give me a concise architectural walkthrough with citations.",
        planner_summary="research campaign",
        task_plan=[
            {"id": "task_1", "title": "Scan title and path candidates", "executor": "general", "mode": "sequential"},
            {
                "id": "task_5",
                "title": "Expand document review",
                "executor": "general",
                "mode": "sequential",
                "depends_on": ["task_1"],
                "consumes_artifacts": ["title_candidates"],
                "handoff_schema": "research_inventory",
                "controller_hints": {
                    "dynamic_doc_review_fanout": True,
                    "max_parallel_doc_reviews": 2,
                    "max_optional_doc_reviews": 0,
                    "retrieval_scope_mode": "kb_only",
                    "strict_kb_scope": True,
                    "kb_collection_id": "default",
                    "search_collection_ids": ["default"],
                },
            },
        ],
    )

    result = kernel.coordinator_controller._expand_doc_review_fanout(
        execution_state=execution_state,
        session_state=session_state,
        task=execution_state.task_plan[1],
    )

    generated_titles = [task["title"] for task in execution_state.task_plan[2:]]
    assert result.status == "completed"
    assert "Review TEST_QUERIES.md" not in generated_titles
    assert "Review ARCHITECTURE.md" in generated_titles
    assert "Review CONTROL_FLOW.md" in generated_titles


def test_coordinator_uses_structured_fallback_for_shallow_detailed_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    kernel = RuntimeKernel(
        _make_runtime_settings(tmp_path),
        providers=SimpleNamespace(),
        stores=SimpleNamespace(),
    )
    coordinator = kernel.registry.get("coordinator")
    assert coordinator is not None

    session_state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
        metadata={
            "kb_collection_id": "default",
            "active_doc_focus": {
                "collection_id": "default",
                "documents": [
                    {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md"},
                    {"doc_id": "KB_control", "title": "CONTROL_FLOW.md"},
                ],
            },
        },
    )
    finalizer_calls: list[dict[str, object]] = []

    def fake_run_agent(agent, scoped_state, *, user_text, callbacks, task_payload=None):
        del callbacks
        if agent.mode == "planner":
            return SimpleNamespace(
                text="",
                messages=list(scoped_state.messages),
                metadata={
                    "planner_payload": {
                        "summary": "Summarize stored candidate docs.",
                        "tasks": build_fallback_plan(
                            user_text,
                            session_metadata=dict(scoped_state.metadata or {}),
                        ),
                    }
                },
            )
        if agent.name == "general":
            worker_request = dict((task_payload or {}).get("worker_request") or {})
            title = str(worker_request.get("title") or "")
            if title == "Digest ARCHITECTURE.md":
                return SimpleNamespace(
                    text=json.dumps(
                        {
                            "document": {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md"},
                            "document_summary": "High-level runtime architecture.",
                            "subsystems": [
                                {
                                    "name": "Runtime kernel",
                                    "aliases": ["RuntimeKernel"],
                                    "description": "Coordinates service and query loop execution.",
                                    "responsibilities": ["coordinates execution"],
                                    "interfaces": ["service layer", "query loop"],
                                    "supporting_citation_ids": ["KB_architecture#chunk0001"],
                                }
                            ],
                            "responsibilities": ["coordinates execution"],
                            "interfaces": ["service layer", "query loop"],
                            "used_citation_ids": ["KB_architecture#chunk0001"],
                        }
                    ),
                    messages=[RuntimeMessage(role="assistant", content="digest architecture")],
                    metadata={},
                )
            if title == "Digest CONTROL_FLOW.md":
                return SimpleNamespace(
                    text=json.dumps(
                        {
                            "document": {"doc_id": "KB_control", "title": "CONTROL_FLOW.md"},
                            "document_summary": "Control-flow orchestration details.",
                            "subsystems": [
                                {
                                    "name": "Router",
                                    "aliases": ["route_turn"],
                                    "description": "Routes requests into runtime agents.",
                                    "responsibilities": ["classifies turns"],
                                    "interfaces": ["gateway", "agent selection"],
                                    "supporting_citation_ids": ["KB_control#chunk0002"],
                                }
                            ],
                            "responsibilities": ["classifies turns"],
                            "interfaces": ["gateway", "agent selection"],
                            "used_citation_ids": ["KB_control#chunk0002"],
                        }
                    ),
                    messages=[RuntimeMessage(role="assistant", content="digest control flow")],
                    metadata={},
                )
            if title == "Consolidate subsystem inventory":
                return SimpleNamespace(
                    text=json.dumps(
                        {
                            "subsystems": [
                                {
                                    "name": "Runtime kernel",
                                    "aliases": ["RuntimeKernel"],
                                    "description": "Coordinates service and query loop execution.",
                                    "responsibilities": ["coordinates execution"],
                                    "interfaces": ["service layer", "query loop"],
                                    "supporting_documents": [{"doc_id": "KB_architecture", "title": "ARCHITECTURE.md"}],
                                    "supporting_citation_ids": ["KB_architecture#chunk0001"],
                                    "coverage": "strong",
                                },
                                {
                                    "name": "Router",
                                    "aliases": ["route_turn"],
                                    "description": "Routes requests into runtime agents.",
                                    "responsibilities": ["classifies turns"],
                                    "interfaces": ["gateway", "agent selection"],
                                    "supporting_documents": [{"doc_id": "KB_control", "title": "CONTROL_FLOW.md"}],
                                    "supporting_citation_ids": ["KB_control#chunk0002"],
                                    "coverage": "strong",
                                },
                            ],
                            "source_documents": [
                                {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md"},
                                {"doc_id": "KB_control", "title": "CONTROL_FLOW.md"},
                            ],
                            "scope_collection_id": "default",
                        }
                    ),
                    messages=[RuntimeMessage(role="assistant", content="inventory")],
                    metadata={},
                )
            return SimpleNamespace(
                text="",
                messages=[RuntimeMessage(role="assistant", content="")],
                metadata={},
            )
        if agent.mode == "finalizer":
            finalizer_calls.append(dict(task_payload or {}))
            return SimpleNamespace(
                text="Documents with grounded evidence relevant to the request:\n\nARCHITECTURE.md",
                messages=list(scoped_state.messages),
                metadata={},
            )
        if agent.name == "rag_worker":
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "task_id": "backfill_router",
                        "evidence_entries": [],
                        "candidate_docs": [],
                        "graded_chunks": [],
                        "warnings": [],
                        "doc_focus": [
                            {"doc_id": "KB_control", "title": "CONTROL_FLOW.md"},
                        ],
                    }
                ),
                messages=[RuntimeMessage(role="assistant", content="backfill router")],
                metadata={
                    "rag_search_result": {
                        "task_id": "backfill_router",
                        "evidence_entries": [],
                        "candidate_docs": [],
                        "graded_chunks": [],
                        "warnings": [],
                        "doc_focus": [
                            {"doc_id": "KB_control", "title": "CONTROL_FLOW.md"},
                        ],
                    }
                },
            )
        if agent.mode == "verifier":
            return SimpleNamespace(
                text='{"status":"pass","summary":"verified","issues":[],"feedback":""}',
                messages=list(scoped_state.messages),
                metadata={"verification": {"status": "pass", "summary": "verified", "issues": [], "feedback": ""}},
            )
        raise AssertionError(f"Unexpected agent {agent.name}")

    monkeypatch.setattr(kernel, "run_agent", fake_run_agent)

    result = kernel._run_coordinator(
        coordinator,
        session_state,
        user_text="Can you look through the candidate documents you provided and give me a detailed summary of the major subsystems involved?",
        callbacks=[],
    )

    assert len(finalizer_calls) == 2
    assert "## Overall Architecture" in result.text
    assert "### Runtime kernel" in result.text
    assert "### Router" in result.text
