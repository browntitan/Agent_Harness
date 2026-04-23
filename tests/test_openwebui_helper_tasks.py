from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from agentic_chatbot_next.app.service import AppContext, RuntimeService, _summarise_history
from agentic_chatbot_next.contracts.messages import RuntimeMessage, SessionState
from agentic_chatbot_next.runtime.query_loop import _recent_conversation_context
from agentic_chatbot_next.session import ChatSession


def _settings(tmp_path: Path) -> SimpleNamespace:
    repo_root = Path(__file__).resolve().parents[1]
    return SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        agents_dir=repo_root / "data" / "agents",
        max_worker_concurrency=2,
        enable_coordinator_mode=False,
        rag_top_k_vector=2,
        rag_top_k_keyword=2,
        rag_max_retries=1,
        llm_router_enabled=False,
        llm_router_confidence_threshold=0.70,
        workspace_dir=None,
        clear_scratchpad_per_turn=False,
        default_tenant_id="local-dev",
        default_user_id="local-cli",
        default_conversation_id="local-session",
        default_collection_id="default",
        memory_enabled=False,
    )


def test_runtime_service_bypasses_router_for_openwebui_helper_task(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
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
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("route_turn should not run for helper tasks")),
    )

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="helper-conv")
    captured: dict[str, object] = {}

    def fake_process_basic_turn(session_obj, **kwargs):
        del session_obj
        captured.update(kwargs)
        return "Architecture Overview"

    monkeypatch.setattr(app.kernel, "process_basic_turn", fake_process_basic_turn)

    result = app.process_turn(
        session,
        user_text="Generate a concise, 3-5 word title for this chat.",
        request_metadata={"openwebui_helper_task_type": "title"},
    )

    assert result == "Architecture Overview"
    assert "internal Open WebUI helper task" in str(captured["system_prompt"])
    assert captured["skip_post_turn_memory"] is True
    assert captured["user_message_metadata"] == {
        "openwebui_helper_task_type": "title",
        "openwebui_internal": True,
    }
    assert captured["assistant_message_metadata"] == {
        "openwebui_helper_task_type": "title",
        "openwebui_internal": True,
    }
    assert captured["route_metadata"]["openwebui_helper_task_type"] == "title"


def test_runtime_service_infers_openwebui_helper_task_from_text_for_openwebui_clients(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
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
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("route_turn should not run for inferred helper tasks")),
    )

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="helper-conv")
    captured: dict[str, object] = {}

    def fake_process_basic_turn(session_obj, **kwargs):
        del session_obj
        captured.update(kwargs)
        return '["architecture","runtime"]'

    monkeypatch.setattr(app.kernel, "process_basic_turn", fake_process_basic_turn)

    result = app.process_turn(
        session,
        user_text="Generate 1-3 broad tags for this chat.",
        request_metadata={"openwebui_client": True},
    )

    assert result == '["architecture","runtime"]'
    assert "internal Open WebUI helper task" in str(captured["system_prompt"])
    assert captured["user_message_metadata"] == {
        "openwebui_helper_task_type": "tags",
        "openwebui_internal": True,
    }


def test_runtime_service_infers_search_query_helper_task_from_text_for_openwebui_clients(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
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
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("route_turn should not run for search-query helper tasks")),
    )

    app = RuntimeService(AppContext(settings=settings, providers=providers, stores=stores))
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="helper-conv")
    captured: dict[str, object] = {}

    def fake_process_basic_turn(session_obj, **kwargs):
        del session_obj
        captured.update(kwargs)
        return '{"queries":["customer review sentiment analysis best practices"]}'

    monkeypatch.setattr(app.kernel, "process_basic_turn", fake_process_basic_turn)

    result = app.process_turn(
        session,
        user_text=(
            "### Task:\n"
            "Analyze the chat history to determine the necessity of generating search queries, in the given language.\n"
            "Respond EXCLUSIVELY with a JSON object in the form {\"queries\": [\"query1\"]}."
        ),
        request_metadata={"openwebui_client": True},
    )

    assert result == '{"queries":["customer review sentiment analysis best practices"]}'
    assert "internal Open WebUI helper task" in str(captured["system_prompt"])
    assert captured["user_message_metadata"] == {
        "openwebui_helper_task_type": "search_queries",
        "openwebui_internal": True,
    }


def test_helper_messages_are_filtered_from_history_and_context() -> None:
    history = [
        HumanMessage(content="Real user question"),
        AIMessage(content="Real answer"),
        HumanMessage(
            content="Generate a concise, 3-5 word title for this chat.",
            additional_kwargs={"openwebui_helper_task_type": "title"},
        ),
        AIMessage(
            content="Architecture Overview",
            additional_kwargs={"openwebui_helper_task_type": "title"},
        ),
    ]

    summary = _summarise_history(history, n=2)

    assert "Real user question" in summary
    assert "Real answer" in summary
    assert "Architecture Overview" not in summary

    state = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    state.messages = [
        RuntimeMessage(role="user", content="How does the runtime work?"),
        RuntimeMessage(role="assistant", content="It routes turns through the runtime kernel."),
        RuntimeMessage(
            role="user",
            content="Suggest 3-5 relevant follow-up questions.",
            metadata={"openwebui_helper_task_type": "follow_ups"},
        ),
        RuntimeMessage(
            role="assistant",
            content='["What handles retrieval?"]',
            metadata={"openwebui_helper_task_type": "follow_ups"},
        ),
    ]

    context = _recent_conversation_context(state, limit=4)

    assert "How does the runtime work?" in context
    assert "routes turns through the runtime kernel" in context
    assert "follow-up questions" not in context
