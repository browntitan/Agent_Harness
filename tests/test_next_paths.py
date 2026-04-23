from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agentic_chatbot_next.runtime.context import RuntimePaths, filesystem_key


def test_runtime_paths_use_consistent_filesystem_keys(tmp_path: Path) -> None:
    settings = SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        workspace_dir=tmp_path / "workspaces",
        memory_dir=tmp_path / "memory",
    )
    paths = RuntimePaths.from_settings(settings)
    session_id = "tenant:user/conversation with spaces"
    tenant_id = "tenant"
    user_id = "user@example.com"
    conversation_id = "conversation/42"

    session_key = paths.session_key(session_id)
    assert session_key == filesystem_key(session_id)
    assert paths.session_dir(session_id) == tmp_path / "runtime" / "sessions" / session_key
    assert paths.workspace_dir(session_id) == tmp_path / "workspaces" / session_key
    assert paths.session_state_path(session_id).name == "state.json"
    assert paths.session_events_path(session_id).name == "events.jsonl"

    conversation_dir = paths.conversation_memory_dir(tenant_id, user_id, conversation_id)
    profile_dir = paths.user_profile_dir(tenant_id, user_id)
    assert conversation_dir.parts[-1] == filesystem_key(conversation_id)
    assert profile_dir.parts[-1] == "profile"
    assert filesystem_key("unsafe///value") != "unsafe///value"
