from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agentic_chatbot_next.contracts.agents import AgentDefinition

_LIVE_AGENT_MODES = (
    "basic",
    "react",
    "rag",
    "planner",
    "finalizer",
    "verifier",
    "coordinator",
    "memory_maintainer",
)


class AgentDefinitionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    mode: Literal[
        "basic",
        "react",
        "rag",
        "planner",
        "finalizer",
        "verifier",
        "coordinator",
        "memory_maintainer",
    ]
    description: str = ""
    prompt_file: str = Field(..., min_length=1)
    skill_scope: str = ""
    allowed_tools: List[str] = Field(default_factory=list)
    allowed_worker_agents: List[str] = Field(default_factory=list)
    preload_skill_packs: List[str] = Field(default_factory=list)
    memory_scopes: List[str] = Field(default_factory=list)
    max_steps: int = Field(default=10, gt=0)
    max_tool_calls: int = Field(default=12, ge=0)
    allow_background_jobs: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "name",
        "description",
        "prompt_file",
        "skill_scope",
        mode="before",
    )
    @classmethod
    def _coerce_string(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator(
        "allowed_tools",
        "allowed_worker_agents",
        "preload_skill_packs",
        "memory_scopes",
        mode="before",
    )
    @classmethod
    def _require_string_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("must be a JSON array of strings")
        return value

    @field_validator(
        "allowed_tools",
        "allowed_worker_agents",
        "preload_skill_packs",
        "memory_scopes",
    )
    @classmethod
    def _normalise_string_list(cls, value: List[Any]) -> List[str]:
        cleaned: List[str] = []
        for item in value:
            text = str(item or "").strip()
            if not text:
                raise ValueError("must not contain empty values")
            cleaned.append(text)
        return cleaned

    @field_validator("prompt_file")
    @classmethod
    def _validate_prompt_file(cls, value: str) -> str:
        clean = value.strip()
        if not clean.endswith(".md"):
            raise ValueError("must be a non-empty markdown filename ending in '.md'")
        return clean


def format_agent_validation_error(path: Any, error: ValidationError) -> ValueError:
    lines = [f"Agent file {path} is invalid:"]
    for item in error.errors():
        location = ".".join(str(part) for part in item.get("loc", ())) or "frontmatter"
        message = str(item.get("msg") or "invalid value")
        lines.append(f"- {location}: {message}")
    return ValueError("\n".join(lines))


def validate_agent_frontmatter(raw: Dict[str, Any], *, path: Any) -> AgentDefinition:
    try:
        validated = AgentDefinitionModel.model_validate(raw)
    except ValidationError as exc:
        raise format_agent_validation_error(path, exc) from exc
    return AgentDefinition.from_dict(validated.model_dump())


__all__ = [
    "AgentDefinitionModel",
    "format_agent_validation_error",
    "validate_agent_frontmatter",
    "_LIVE_AGENT_MODES",
]
