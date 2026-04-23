from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from agentic_chatbot_next.app.service import AppContext, RuntimeService
from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.runtime.long_output import LongOutputComposer, LongOutputOptions
from agentic_chatbot_next.session import ChatSession


class _SequenceLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def invoke(self, messages, config=None):
        del messages, config
        if not self.responses:
            raise AssertionError("No more fake LLM responses are available.")
        response = self.responses.pop(0)
        if isinstance(response, AIMessage):
            return response
        return AIMessage(content=str(response))


def _service_settings(tmp_path: Path) -> SimpleNamespace:
    repo_root = Path(__file__).resolve().parents[1]
    return SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        workspace_dir=tmp_path / "workspaces",
        memory_dir=tmp_path / "memory",
        agents_dir=repo_root / "data" / "agents",
        skills_dir=repo_root / "data" / "skills",
        llm_provider="azure",
        judge_provider="azure",
        runtime_events_enabled=True,
        max_worker_concurrency=2,
        session_hydrate_window_messages=40,
        session_transcript_page_size=100,
        enable_coordinator_mode=False,
        planner_max_tasks=4,
        rag_top_k_vector=2,
        rag_top_k_keyword=2,
        rag_max_retries=1,
        chat_max_output_tokens=None,
        llm_router_enabled=False,
        llm_router_confidence_threshold=0.70,
        clear_scratchpad_per_turn=False,
        agent_runtime_mode="planner_executor",
        default_tenant_id="local-dev",
        default_user_id="local-cli",
        default_conversation_id="local-session",
        default_collection_id="default",
        control_panel_agent_overlays_dir=None,
        control_panel_prompt_overlays_dir=None,
    )


def test_long_output_composer_persists_manifest_output_and_artifact(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
        workspace_root=str(workspace_root),
    )
    settings = SimpleNamespace(workspace_dir=tmp_path / "workspaces")
    agent = AgentDefinition(name="general", mode="react")
    llm = _SequenceLLM(
        [
            json.dumps(
                {
                    "title": "Expansion Plan",
                    "executive_summary": "A durable sectioned report.",
                    "sections": [
                        {"heading": "Overview", "brief": "Explain the overall strategy.", "target_words": 220},
                        {"heading": "Implementation", "brief": "Describe the backend changes.", "target_words": 240},
                    ],
                }
            ),
            "This overview explains why multiple calls plus persistent artifacts are the safest pattern.",
            "Explains the multi-call generation strategy and persistence model.",
            "Implementation details cover orchestration, manifests, and downloadable artifacts.",
            "Describes the concrete backend implementation and artifact handoff.",
            "I finished the long-form draft and attached the full Markdown artifact.",
        ]
    )
    composer = LongOutputComposer(
        settings=settings,
        chat_llm=llm,
        agent=agent,
        system_prompt="Write clearly.",
        session_or_state=state,
    )

    result = composer.compose(
        user_text="Write a detailed implementation plan for long-form generation.",
        options=LongOutputOptions(enabled=True, target_words=900, target_sections=2, delivery_mode="hybrid"),
    )

    output_path = workspace_root / result.output_filename
    manifest_path = workspace_root / result.manifest_filename

    assert output_path.exists()
    assert manifest_path.exists()
    assert "## Overview" in output_path.read_text(encoding="utf-8")
    assert "## Implementation" in output_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert [item["status"] for item in manifest["sections"]] == ["completed", "completed"]
    assert result.artifact["download_id"] in state.metadata["downloads"]
    assert "attached the full Markdown artifact" in result.summary_text


def test_long_output_composer_continues_truncated_sections_without_losing_content(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
        workspace_root=str(workspace_root),
    )
    settings = SimpleNamespace(workspace_dir=tmp_path / "workspaces")
    agent = AgentDefinition(name="general", mode="react")
    llm = _SequenceLLM(
        [
            json.dumps(
                {
                    "title": "Truncation Recovery",
                    "executive_summary": "Recover section content when the model hits a token cap.",
                    "sections": [
                        {"heading": "Recovery", "brief": "Show continuation behavior.", "target_words": 220},
                    ],
                }
            ),
            AIMessage(content="First half of the section.", additional_kwargs={"finish_reason": "length"}),
            "Second half of the section.",
            "Summarizes the recovered section.",
            "Completed the recovered draft and attached it.",
        ]
    )
    composer = LongOutputComposer(
        settings=settings,
        chat_llm=llm,
        agent=agent,
        system_prompt="Write clearly.",
        session_or_state=state,
    )

    result = composer.compose(
        user_text="Produce one long section and recover if the model truncates it.",
        options=LongOutputOptions(enabled=True, target_words=500, target_sections=1, delivery_mode="hybrid"),
    )

    output_text = (workspace_root / result.output_filename).read_text(encoding="utf-8")
    manifest = json.loads((workspace_root / result.manifest_filename).read_text(encoding="utf-8"))

    assert "First half of the section." in output_text
    assert "Second half of the section." in output_text
    assert manifest["sections"][0]["retries"] == 1


def test_runtime_service_long_output_background_returns_job_ack_and_queues_job(tmp_path: Path, monkeypatch) -> None:
    settings = _service_settings(tmp_path)
    providers = SimpleNamespace(chat=object(), judge=object(), embeddings=object())
    stores = SimpleNamespace()

    monkeypatch.setattr("agentic_chatbot_next.app.service.load_basic_chat_skills", lambda settings: "Be concise.")
    monkeypatch.setattr("agentic_chatbot_next.app.service.ensure_kb_indexed", lambda *args, **kwargs: None)

    class _NoopSkillIndexSync:
        def __init__(self, *args, **kwargs):
            pass

        def sync(self, tenant_id: str) -> None:
            del tenant_id
            return None

    monkeypatch.setattr("agentic_chatbot_next.app.service.SkillIndexSync", _NoopSkillIndexSync)
    monkeypatch.setattr(
        "agentic_chatbot_next.app.service.route_turn",
        lambda *args, **kwargs: SimpleNamespace(
            route="AGENT",
            confidence=0.95,
            reasons=["long_form_request"],
            router_method="deterministic",
            suggested_agent="general",
        ),
    )

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="background-long-output")
    captured: dict[str, object] = {}

    def _fake_start_background_job(job, runner):
        captured["job"] = job
        captured["runner"] = runner
        return job

    monkeypatch.setattr(app.kernel.job_manager, "start_background_job", _fake_start_background_job)

    text = app.process_turn(
        session,
        user_text="Write a 4000 word technical migration report.",
        request_metadata={"long_output": {"enabled": True, "target_words": 4000, "background_ok": True}},
    )

    assert "Job ID:" in text
    job = captured["job"]
    assert job is not None
    assert job.metadata["long_output"]["enabled"] is True
    assert job.metadata["long_output"]["target_words"] == 4000
    assert session.messages[-1].additional_kwargs["job_id"] == job.job_id


def test_runtime_service_long_output_preserves_request_chat_output_cap_in_job_metadata(tmp_path: Path, monkeypatch) -> None:
    settings = _service_settings(tmp_path)
    providers = SimpleNamespace(chat=object(), judge=object(), embeddings=object())
    stores = SimpleNamespace()

    monkeypatch.setattr("agentic_chatbot_next.app.service.load_basic_chat_skills", lambda settings: "Be concise.")
    monkeypatch.setattr("agentic_chatbot_next.app.service.ensure_kb_indexed", lambda *args, **kwargs: None)

    class _NoopSkillIndexSync:
        def __init__(self, *args, **kwargs):
            pass

        def sync(self, tenant_id: str) -> None:
            del tenant_id
            return None

    monkeypatch.setattr("agentic_chatbot_next.app.service.SkillIndexSync", _NoopSkillIndexSync)
    monkeypatch.setattr(
        "agentic_chatbot_next.app.service.route_turn",
        lambda *args, **kwargs: SimpleNamespace(
            route="AGENT",
            confidence=0.95,
            reasons=["long_form_request"],
            router_method="deterministic",
            suggested_agent="general",
        ),
    )

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="background-long-output-cap")
    captured: dict[str, object] = {}

    def _fake_start_background_job(job, runner):
        captured["job"] = job
        captured["runner"] = runner
        return job

    monkeypatch.setattr(app.kernel.job_manager, "start_background_job", _fake_start_background_job)

    text = app.process_turn(
        session,
        user_text="Write a 4000 word technical migration report.",
        request_metadata={
            "chat_max_output_tokens": 5000,
            "long_output": {"enabled": True, "target_words": 4000, "background_ok": True},
        },
    )

    assert "Job ID:" in text
    job = captured["job"]
    assert job is not None
    assert job.metadata["long_output"]["chat_max_output_tokens"] == 5000


def test_runtime_service_passes_request_chat_output_cap_into_basic_provider_resolution(tmp_path: Path, monkeypatch) -> None:
    settings = _service_settings(tmp_path)
    providers = SimpleNamespace(chat=object(), judge=object(), embeddings=object())
    stores = SimpleNamespace()

    monkeypatch.setattr("agentic_chatbot_next.app.service.load_basic_chat_skills", lambda settings: "Be concise.")
    monkeypatch.setattr("agentic_chatbot_next.app.service.ensure_kb_indexed", lambda *args, **kwargs: None)

    class _NoopSkillIndexSync:
        def __init__(self, *args, **kwargs):
            pass

        def sync(self, tenant_id: str) -> None:
            del tenant_id
            return None

    monkeypatch.setattr("agentic_chatbot_next.app.service.SkillIndexSync", _NoopSkillIndexSync)
    monkeypatch.setattr(
        "agentic_chatbot_next.app.service.route_turn",
        lambda *args, **kwargs: SimpleNamespace(
            route="BASIC",
            confidence=0.99,
            reasons=["simple_chat"],
            router_method="deterministic",
            suggested_agent="basic",
        ),
    )

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="basic-output-cap")
    captured: dict[str, object] = {}

    def _fake_resolve(agent_name: str, *, chat_max_output_tokens=None):
        captured["agent_name"] = agent_name
        captured["chat_max_output_tokens"] = chat_max_output_tokens
        return SimpleNamespace(chat=object(), judge=object(), embeddings=object())

    monkeypatch.setattr(app.kernel, "resolve_providers_for_agent", _fake_resolve)
    monkeypatch.setattr(app.kernel, "process_basic_turn", lambda *args, **kwargs: "basic ok")

    text = app.process_turn(
        session,
        user_text="Hello there",
        request_metadata={"chat_max_output_tokens": 3072},
    )

    assert text == "basic ok"
    assert captured["agent_name"] == "basic"
    assert captured["chat_max_output_tokens"] == 3072
