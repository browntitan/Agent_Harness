from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.runtime.context import RuntimePaths, filesystem_key
from agentic_chatbot_next.tools.base import ToolContext


def _paths(tmp_path: Path) -> RuntimePaths:
    return RuntimePaths(
        runtime_root=tmp_path / "runtime",
        workspace_root=tmp_path / "workspaces",
        memory_root=tmp_path / "memory",
    )


def test_session_handle_exposes_session_like_properties(tmp_path: Path):
    session = SessionState(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id="conv-a",
        request_id="req-a",
        uploaded_doc_ids=["doc-1"],
        demo_mode=True,
        workspace_root="/tmp/runtime-workspace",
    )
    session.scratchpad["note"] = "remember this"
    paths = _paths(tmp_path)
    context = ToolContext(
        settings=SimpleNamespace(workspace_dir=paths.workspace_root),
        providers=object(),
        stores=object(),
        session=session,
        paths=paths,
    )
    handle = context.session_handle

    assert handle.session_id == "tenant-a:user-a:conv-a"
    assert handle.tenant_id == "tenant-a"
    assert handle.user_id == "user-a"
    assert handle.conversation_id == "conv-a"
    assert handle.request_id == "req-a"
    assert handle.uploaded_doc_ids == ["doc-1"]
    assert handle.scratchpad == {"note": "remember this"}
    assert handle.demo_mode is True
    assert handle.workspace is not None
    assert handle.workspace.root == paths.workspace_root / filesystem_key(session.session_id)
    assert session.workspace_root == str(handle.workspace.root)


def test_tool_context_exposes_workspace_root_from_session_state(tmp_path: Path):
    workspace_root = tmp_path / "workspaces" / "tenant-user-conv"
    workspace_root.mkdir(parents=True)
    context = ToolContext(
        settings=SimpleNamespace(workspace_dir=None),
        providers=object(),
        stores=object(),
        session=SessionState(
            tenant_id="tenant",
            user_id="user",
            conversation_id="conv",
            workspace_root=str(workspace_root),
        ),
        paths=_paths(tmp_path),
    )

    assert context.workspace_root == workspace_root
