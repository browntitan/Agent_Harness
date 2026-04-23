from __future__ import annotations

import json
from types import SimpleNamespace

from langchain_core.messages import HumanMessage, ToolMessage

from agentic_chatbot_next.contracts.messages import RuntimeMessage, SessionState
from agentic_chatbot_next.runtime.context import RuntimePaths
from agentic_chatbot_next.runtime.context_budget import (
    ContextBudgetManager,
    ContextSection,
    estimate_text_tokens,
)
from agentic_chatbot_next.runtime.transcript_store import RuntimeTranscriptStore


def _settings(tmp_path, **overrides):
    payload = {
        "runtime_dir": tmp_path / "runtime",
        "workspace_dir": tmp_path / "workspaces",
        "memory_dir": tmp_path / "memory",
        "context_budget_enabled": True,
        "context_window_tokens": 1000,
        "context_target_ratio": 0.6,
        "context_autocompact_threshold": 0.5,
        "context_tool_result_max_tokens": 40,
        "context_tool_results_total_tokens": 80,
        "context_microcompact_target_tokens": 48,
        "context_compact_recent_messages": 2,
        "context_restore_recent_files": 4,
        "context_restore_recent_skills": 4,
        "tiktoken_enabled": False,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


class _FakeResponse:
    content = "Model compact summary: keep the user goal, file handles, and unresolved decisions."


class _FakeJudge:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        self.messages = messages
        return _FakeResponse()


def test_token_estimator_uses_deterministic_fallback_when_tiktoken_disabled(tmp_path) -> None:
    settings = _settings(tmp_path, tiktoken_enabled=False)

    assert estimate_text_tokens("abcd", settings) == 1
    assert estimate_text_tokens("abcde", settings) == 2


def test_budget_tool_message_preserves_full_result_in_sidecar(tmp_path) -> None:
    settings = _settings(tmp_path, context_tool_result_max_tokens=12)
    store = RuntimeTranscriptStore(RuntimePaths.from_settings(settings))
    manager = ContextBudgetManager(settings, transcript_store=store)
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    tool_context = SimpleNamespace(
        transcript_store=store,
        session=session,
        active_agent="general",
        metadata={"job_id": "job_1"},
    )
    original = json.dumps(
        {
            "answer": "A" * 1200,
            "filename": "analysis.csv",
            "warnings": ["large output"],
        }
    )

    budgeted = manager.budget_tool_message(
        ToolMessage(content=original, tool_call_id="call_1", name="large_tool"),
        tool_context=tool_context,
    )
    payload = json.loads(str(budgeted.content))
    sidecars = store.load_session_tool_results(session.session_id)

    assert payload["object"] == "budgeted_tool_result"
    assert payload["tool_name"] == "large_tool"
    assert payload["full_result_ref"].startswith(f"session:{session.session_id}:tool_result:")
    assert sidecars[0]["content"] == original
    assert sidecars[0]["tool_name"] == "large_tool"


def test_autocompact_persists_boundary_and_restore_snapshot(tmp_path) -> None:
    settings = _settings(tmp_path)
    store = RuntimeTranscriptStore(RuntimePaths.from_settings(settings))
    manager = ContextBudgetManager(settings, transcript_store=store)
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    messages = []
    for index in range(8):
        metadata = {}
        if index == 2:
            metadata = {
                "artifacts": [{"filename": "research_notes.md", "artifact_ref": "artifact:notes"}],
                "skill_resolution": {
                    "matches": [
                        {
                            "skill_id": "skill_research",
                            "skill_family_id": "family_research",
                            "name": "Research",
                            "agent_scope": "general",
                        }
                    ]
                },
            }
        message = RuntimeMessage(
            role="assistant" if index % 2 else "user",
            content=f"message {index} " + ("x" * 420),
            metadata=metadata,
        )
        messages.append(message)
    session.messages = list(messages)
    store.ensure_session_transcript_seeded(session.session_id, messages)
    judge = _FakeJudge()

    budgeted = manager.prepare_turn(
        agent_name="general",
        session_state=session,
        user_text="continue",
        sections=[ContextSection(name="base_prompt", content="You are helpful.", preserve=True, priority=100)],
        history_messages=messages,
        providers=SimpleNamespace(judge=judge),
        transcript_store=store,
    )

    boundary = session.metadata["context_compact_boundary"]
    restore = session.metadata["context_restore_snapshot"]
    compactions = store.load_session_compactions(session.session_id)

    assert judge.calls == 1
    assert len(budgeted.history_messages) == 2
    assert boundary["summary"].startswith("Model compact summary")
    assert len(boundary["covered_message_ids"]) == 6
    assert restore["recent_files"][0]["filename"] == "research_notes.md"
    assert restore["recent_skills"][0]["skill_id"] == "skill_research"
    assert compactions[0]["boundary"]["boundary_id"] == boundary["boundary_id"]


def test_microcompact_rewrites_current_turn_tool_messages(tmp_path) -> None:
    settings = _settings(tmp_path, context_tool_results_total_tokens=20, context_microcompact_target_tokens=16)
    manager = ContextBudgetManager(settings)
    messages = [
        HumanMessage(content="run tools"),
        ToolMessage(content="first " + ("x" * 1600), tool_call_id="call_1", name="alpha"),
        ToolMessage(content="second " + ("y" * 1600), tool_call_id="call_2", name="beta"),
    ]

    compacted = manager.microcompact_messages(messages)

    assert compacted[0] is messages[0]
    assert json.loads(str(compacted[1].content))["object"] == "budgeted_tool_result"
    assert json.loads(str(compacted[2].content))["microcompact"] is True
    assert compacted[1].tool_call_id == "call_1"
