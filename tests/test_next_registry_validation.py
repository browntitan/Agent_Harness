from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_chatbot_next.runtime.kernel import RuntimeKernel
from agentic_chatbot_next.runtime.registry_diagnostics import build_runtime_error_payload
from agentic_chatbot_next.tools.registry import build_tool_definitions


def _settings(tmp_path: Path, *, agents_dir: Path, skills_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        workspace_dir=tmp_path / "workspaces",
        memory_dir=tmp_path / "memory",
        agents_dir=agents_dir,
        skills_dir=skills_dir,
        runtime_events_enabled=True,
        max_worker_concurrency=2,
    )


def _write_agent(path: Path, *, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_runtime_kernel_rejects_unknown_tool_reference(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agents"
    skills_dir = tmp_path / "skills"
    agents_dir.mkdir()
    skills_dir.mkdir()
    (skills_dir / "general_agent.md").write_text("general prompt", encoding="utf-8")
    _write_agent(
        agents_dir / "general.md",
        body="""---
name: general
mode: react
description: bad tool config
prompt_file: general_agent.md
skill_scope: general
allowed_tools: ["does_not_exist"]
allowed_worker_agents: []
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 3
max_tool_calls: 3
allow_background_jobs: false
metadata: {}
---
bad
""",
    )

    with pytest.raises(ValueError, match="unknown tool"):
        RuntimeKernel(_settings(tmp_path, agents_dir=agents_dir, skills_dir=skills_dir), providers=None, stores=None)


def test_runtime_kernel_rejects_invalid_worker_and_memory_scope(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agents"
    skills_dir = tmp_path / "skills"
    agents_dir.mkdir()
    skills_dir.mkdir()
    (skills_dir / "general_agent.md").write_text("general prompt", encoding="utf-8")
    _write_agent(
        agents_dir / "general.md",
        body="""---
name: general
mode: react
description: bad worker config
prompt_file: general_agent.md
skill_scope: general
allowed_tools: ["calculator"]
allowed_worker_agents: ["missing_worker"]
preload_skill_packs: []
memory_scopes: ["invalid_scope"]
max_steps: 3
max_tool_calls: 3
allow_background_jobs: false
metadata: {}
---
bad
""",
    )

    with pytest.raises(ValueError, match="unknown worker|invalid memory scope"):
        RuntimeKernel(_settings(tmp_path, agents_dir=agents_dir, skills_dir=skills_dir), providers=None, stores=None)


def test_runtime_registry_includes_worker_request_tools() -> None:
    definitions = build_tool_definitions(None)

    assert "list_worker_requests" in definitions
    assert "respond_worker_request" in definitions
    assert definitions["list_worker_requests"].group == "orchestration"
    assert definitions["respond_worker_request"].group == "orchestration"


def test_default_agent_registry_validates_worker_request_tools(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    settings = _settings(
        tmp_path,
        agents_dir=repo_root / "data" / "agents",
        skills_dir=repo_root / "data" / "skills",
    )

    RuntimeKernel(settings, providers=None, stores=None).validate_registry()


def test_runtime_registry_error_payload_names_missing_tools() -> None:
    payload = build_runtime_error_payload(
        ValueError(
            "Invalid next-runtime agent configuration:\n"
            "- agent 'coordinator' references unknown tool 'list_worker_requests'\n"
            "- agent 'general' references unknown tool 'respond_worker_request'"
        )
    )

    assert payload["error_code"] == "runtime_registry_invalid"
    assert payload["affected_agents"] == ["coordinator", "general"]
    assert payload["missing_tools"] == [
        {"agent": "coordinator", "tool": "list_worker_requests"},
        {"agent": "general", "tool": "respond_worker_request"},
    ]
