from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from agentic_chatbot_next.agents.loader import LoadedAgentFile, load_agent_markdown
from agentic_chatbot_next.contracts.agents import AgentDefinition


class AgentRegistry:
    def __init__(self, agents_dir: Path, overlay_dir: Path | None = None):
        self.agents_dir = agents_dir
        self.overlay_dir = overlay_dir
        self._definitions: Dict[str, AgentDefinition] = {}
        self._loaded_files: Dict[str, LoadedAgentFile] = {}
        self.reload()

    def reload(self) -> None:
        definitions: Dict[str, AgentDefinition] = {}
        loaded_files: Dict[str, LoadedAgentFile] = {}
        source_dirs = [self.agents_dir]
        if self.overlay_dir is not None:
            source_dirs.append(self.overlay_dir)

        for directory in source_dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.md")):
                loaded = load_agent_markdown(path)
                definitions[loaded.definition.name] = loaded.definition
                loaded_files[loaded.definition.name] = loaded
        self._definitions = definitions
        self._loaded_files = loaded_files

    def get(self, name: str) -> Optional[AgentDefinition]:
        return self._definitions.get(name)

    def list(self) -> List[AgentDefinition]:
        return list(self._definitions.values())

    def get_loaded_file(self, name: str) -> Optional[LoadedAgentFile]:
        return self._loaded_files.get(name)

    @staticmethod
    def _role_kind(agent: AgentDefinition) -> str:
        return str(agent.metadata.get("role_kind") or "").strip().lower()

    @staticmethod
    def _entry_path(agent: AgentDefinition) -> str:
        return str(agent.metadata.get("entry_path") or "").strip().lower()

    @staticmethod
    def _expected_output(agent: AgentDefinition) -> str:
        return str(agent.metadata.get("expected_output") or "").strip().lower()

    def is_routable(self, agent: AgentDefinition) -> bool:
        role_kind = self._role_kind(agent)
        entry_path = self._entry_path(agent)
        return role_kind in {"top_level", "top_level_or_worker", "manager"} or entry_path in {
            "default",
            "router_basic",
            "router_fast_path_or_delegated",
            "router_or_delegated",
        }

    def list_routable(self) -> List[AgentDefinition]:
        return [definition for definition in self.list() if self.is_routable(definition)]

    def get_default_agent_name(self) -> str:
        for agent in self.list_routable():
            if self._entry_path(agent) == "default":
                return agent.name
        for agent in self.list_routable():
            if self._role_kind(agent) in {"top_level", "top_level_or_worker"} and agent.mode != "basic":
                return agent.name
        return "general"

    def get_basic_agent_name(self) -> str:
        for agent in self.list_routable():
            if agent.mode == "basic":
                return agent.name
        return "basic"

    def get_manager_agent_name(self) -> str:
        for agent in self.list_routable():
            if self._role_kind(agent) == "manager" or agent.mode == "coordinator":
                return agent.name
        return "coordinator"

    def get_data_analyst_agent_name(self) -> str:
        for agent in self.list_routable():
            tools = set(agent.allowed_tools)
            if {"load_dataset", "execute_code"}.issubset(tools):
                return agent.name
        return "data_analyst"

    def get_rag_agent_name(self) -> str:
        for agent in self.list_routable():
            if agent.mode == "rag" or self._expected_output(agent) == "rag_contract":
                return agent.name
        return "rag_worker"

    def get_graph_agent_name(self) -> str:
        for agent in self.list_routable():
            tools = set(agent.allowed_tools)
            if {"inspect_graph_index", "search_graph_index"}.issubset(tools):
                return agent.name
        return "graph_manager"
