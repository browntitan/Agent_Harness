from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentDefinition:
    name: str
    mode: str
    description: str = ""
    prompt_file: str = ""
    skill_scope: str = ""
    allowed_tools: List[str] = field(default_factory=list)
    allowed_worker_agents: List[str] = field(default_factory=list)
    preload_skill_packs: List[str] = field(default_factory=list)
    memory_scopes: List[str] = field(default_factory=list)
    max_steps: int = 10
    max_tool_calls: int = 12
    allow_background_jobs: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AgentDefinition":
        return cls(
            name=str(raw.get("name") or ""),
            mode=str(raw.get("mode") or ""),
            description=str(raw.get("description") or ""),
            prompt_file=str(raw.get("prompt_file") or ""),
            skill_scope=str(raw.get("skill_scope") or ""),
            allowed_tools=[str(item) for item in (raw.get("allowed_tools") or []) if str(item)],
            allowed_worker_agents=[str(item) for item in (raw.get("allowed_worker_agents") or []) if str(item)],
            preload_skill_packs=[str(item) for item in (raw.get("preload_skill_packs") or []) if str(item)],
            memory_scopes=[str(item) for item in (raw.get("memory_scopes") or []) if str(item)],
            max_steps=int(raw.get("max_steps", 10)),
            max_tool_calls=int(raw.get("max_tool_calls", 12)),
            allow_background_jobs=bool(raw.get("allow_background_jobs", False)),
            metadata=dict(raw.get("metadata") or {}),
        )
