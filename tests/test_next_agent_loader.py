from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_chatbot_next.agents.loader import load_agent_markdown
from agentic_chatbot_next.agents.registry import AgentRegistry
from agentic_chatbot_next.app.service import RuntimeService


def test_agent_registry_loads_markdown_definitions_from_repo() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    registry = AgentRegistry(repo_root / "data" / "agents")
    names = [agent.name for agent in registry.list()]
    assert names == sorted(names)
    assert set(names) >= {
        "basic",
        "general",
        "coordinator",
        "research_coordinator",
        "utility",
        "data_analyst",
        "rag_worker",
        "rag_researcher",
        "graph_manager",
        "planner",
        "finalizer",
        "verifier",
        "memory_maintainer",
    }
    general = registry.get("general")
    assert general is not None
    assert general.prompt_file == "general_agent.md"
    assert "calculator" in general.allowed_tools
    assert "resolve_indexed_docs" in general.allowed_tools
    assert "search_indexed_docs" in general.allowed_tools
    assert "compare_indexed_docs" in general.allowed_tools
    assert "list_graph_indexes" in general.allowed_tools
    assert "search_graph_index" not in general.allowed_tools
    assert "explain_source_plan" not in general.allowed_tools
    assert "invoke_agent" not in general.allowed_tools
    assert general.max_steps == 10
    assert general.max_tool_calls == 12
    graph_manager = registry.get("graph_manager")
    assert graph_manager is not None
    assert "search_graph_index" in graph_manager.allowed_tools
    assert graph_manager.metadata["role_kind"] == "top_level_or_worker"
    assert graph_manager.metadata["entry_path"] == "router_fast_path_or_delegated"
    rag_researcher = registry.get("rag_researcher")
    assert rag_researcher is not None
    assert rag_researcher.mode == "react"
    assert "rag_agent_tool" in rag_researcher.allowed_tools
    assert "plan_rag_queries" in rag_researcher.allowed_tools
    assert "search_corpus_chunks" in rag_researcher.allowed_tools
    assert "grep_corpus_chunks" in rag_researcher.allowed_tools
    assert "fetch_chunk_window" in rag_researcher.allowed_tools
    assert "inspect_document_structure" in rag_researcher.allowed_tools
    assert "search_document_sections" in rag_researcher.allowed_tools
    assert "filter_indexed_docs" in rag_researcher.allowed_tools
    assert "grade_evidence_candidates" in rag_researcher.allowed_tools
    assert "prune_evidence_candidates" in rag_researcher.allowed_tools
    assert "validate_evidence_plan" in rag_researcher.allowed_tools
    assert "build_rag_controller_hints" in rag_researcher.allowed_tools
    assert "search_graph_index" in rag_researcher.allowed_tools
    assert rag_researcher.metadata["entry_path"] == "manual_or_delegated"
    assert rag_researcher.metadata["manual_override_allowed"] is True
    assert "rag_researcher" not in {agent.name for agent in registry.list_routable()}
    coordinator = registry.get("coordinator")
    assert coordinator is not None
    assert list(coordinator.allowed_tools) == [
        "spawn_worker",
        "message_worker",
        "list_worker_requests",
        "respond_worker_request",
        "create_team_channel",
        "post_team_message",
        "list_team_messages",
        "claim_team_messages",
        "respond_team_message",
        "list_jobs",
        "stop_job",
    ]
    assert coordinator.max_steps == 12
    assert coordinator.max_tool_calls == 14
    research_coordinator = registry.get("research_coordinator")
    assert research_coordinator is not None
    assert research_coordinator.mode == "coordinator"
    assert research_coordinator.metadata["role_kind"] == "manager"
    assert research_coordinator.metadata["entry_path"] == "router_fast_path_or_delegated"
    assert research_coordinator.metadata["research_campaign_agent"] is True
    assert set(research_coordinator.allowed_worker_agents) == {
        "planner",
        "rag_worker",
        "rag_researcher",
        "general",
        "graph_manager",
        "finalizer",
        "verifier",
    }
    assert "execute_code" not in research_coordinator.allowed_tools


def test_manual_override_list_includes_non_routable_rag_researcher() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    registry = AgentRegistry(repo_root / "data" / "agents")
    service = RuntimeService.__new__(RuntimeService)
    service.kernel = SimpleNamespace(registry=registry)
    service.ctx = SimpleNamespace(settings=SimpleNamespace(memory_enabled=True))

    overrides = service.list_requested_agent_overrides()

    assert "rag_researcher" in overrides
    assert "rag_researcher" not in {agent.name for agent in registry.list_routable()}


def test_repo_agents_directory_has_no_live_json_definitions() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    json_files = sorted((repo_root / "data" / "agents").glob("*.json"))
    assert json_files == []


def test_agent_loader_rejects_missing_required_frontmatter(tmp_path: Path) -> None:
    path = tmp_path / "broken.md"
    path.write_text(
        "---\n"
        "name: broken\n"
        "mode: react\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prompt_file"):
        load_agent_markdown(path)


def test_agent_loader_rejects_invalid_mode_with_path_qualified_error(tmp_path: Path) -> None:
    path = tmp_path / "broken_mode.md"
    path.write_text(
        "---\n"
        "name: broken\n"
        "mode: orchestrator\n"
        "description: bad mode\n"
        "prompt_file: broken.md\n"
        "skill_scope: general\n"
        "allowed_tools: []\n"
        "allowed_worker_agents: []\n"
        "preload_skill_packs: []\n"
        "memory_scopes: [\"conversation\"]\n"
        "max_steps: 3\n"
        "max_tool_calls: 1\n"
        "allow_background_jobs: false\n"
        "metadata: {}\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        load_agent_markdown(path)
    assert "broken_mode.md" in str(excinfo.value)
    assert "- mode:" in str(excinfo.value)


def test_agent_loader_rejects_non_array_allowed_tools(tmp_path: Path) -> None:
    path = tmp_path / "broken_tools.md"
    path.write_text(
        "---\n"
        "name: broken\n"
        "mode: react\n"
        "description: bad tools\n"
        "prompt_file: broken.md\n"
        "skill_scope: general\n"
        "allowed_tools: calculator\n"
        "allowed_worker_agents: []\n"
        "preload_skill_packs: []\n"
        "memory_scopes: [\"conversation\"]\n"
        "max_steps: 3\n"
        "max_tool_calls: 1\n"
        "allow_background_jobs: false\n"
        "metadata: {}\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        load_agent_markdown(path)
    assert "broken_tools.md" in str(excinfo.value)
    assert "- allowed_tools:" in str(excinfo.value)


def test_agent_loader_accepts_zero_max_tool_calls_for_basic_style_agents(tmp_path: Path) -> None:
    path = tmp_path / "basic.md"
    path.write_text(
        "---\n"
        "name: basic\n"
        "mode: basic\n"
        "description: direct chat\n"
        "prompt_file: basic_chat.md\n"
        "skill_scope: basic\n"
        "allowed_tools: []\n"
        "allowed_worker_agents: []\n"
        "preload_skill_packs: []\n"
        "memory_scopes: [\"conversation\"]\n"
        "max_steps: 1\n"
        "max_tool_calls: 0\n"
        "allow_background_jobs: false\n"
        "metadata: {}\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )

    loaded = load_agent_markdown(path)

    assert loaded.definition.max_tool_calls == 0


def test_agent_registry_overlay_definitions_override_base_agents(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agents"
    overlay_dir = tmp_path / "overlays"
    agents_dir.mkdir()
    overlay_dir.mkdir()
    (agents_dir / "general.md").write_text(
        "---\n"
        "name: general\n"
        "mode: react\n"
        "description: base description\n"
        "prompt_file: general_agent.md\n"
        "skill_scope: general\n"
        'allowed_tools: ["calculator"]\n'
        "allowed_worker_agents: []\n"
        "preload_skill_packs: []\n"
        'memory_scopes: ["conversation"]\n'
        "max_steps: 3\n"
        "max_tool_calls: 2\n"
        "allow_background_jobs: false\n"
        "metadata: {}\n"
        "---\n"
        "Base body\n",
        encoding="utf-8",
    )
    (overlay_dir / "general.md").write_text(
        "---\n"
        "name: general\n"
        "mode: react\n"
        "description: overlay description\n"
        "prompt_file: general_agent.md\n"
        "skill_scope: general\n"
        'allowed_tools: ["calculator", "list_documents"]\n'
        "allowed_worker_agents: []\n"
        'preload_skill_packs: ["skill-a"]\n'
        'memory_scopes: ["conversation"]\n'
        "max_steps: 5\n"
        "max_tool_calls: 4\n"
        "allow_background_jobs: true\n"
        'metadata: {"source":"overlay"}\n'
        "---\n"
        "Overlay body\n",
        encoding="utf-8",
    )

    registry = AgentRegistry(agents_dir, overlay_dir=overlay_dir)
    loaded = registry.get_loaded_file("general")

    assert loaded is not None
    assert loaded.definition.description == "overlay description"
    assert loaded.definition.max_steps == 5
    assert list(loaded.definition.preload_skill_packs) == ["skill-a"]
    assert loaded.body == "Overlay body"
