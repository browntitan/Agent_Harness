from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class ToolDefinition:
    name: str
    group: str
    builder: Any = ""
    description: str = ""
    args_schema: Dict[str, Any] = field(default_factory=dict)
    when_to_use: str = ""
    avoid_when: str = ""
    output_description: str = ""
    examples: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    read_only: bool = False
    destructive: bool = False
    background_safe: bool = False
    concurrency_key: str = ""
    requires_workspace: bool = False
    serializer: str = "default"
    should_defer: bool = False
    search_hint: str = ""
    defer_reason: str = ""
    defer_priority: int = 50
    eager_for_agents: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if callable(self.builder):
            payload["builder"] = getattr(self.builder, "__name__", repr(self.builder))
        return payload

    def render_tool_card(self) -> str:
        lines = [self.description.strip()]
        if self.when_to_use.strip():
            lines.append(f"When to use: {self.when_to_use.strip()}")
        if self.avoid_when.strip():
            lines.append(f"Avoid when: {self.avoid_when.strip()}")
        if self.output_description.strip():
            lines.append(f"Returns: {self.output_description.strip()}")
        if self.examples:
            lines.append("Examples: " + "; ".join(str(item).strip() for item in self.examples if str(item).strip()))
        return "\n".join(line for line in lines if line.strip()).strip()

    def validate_metadata(self) -> list[str]:
        errors: list[str] = []
        if not self.description.strip():
            errors.append("missing description")
        if not isinstance(self.args_schema, dict) or not self.args_schema:
            errors.append("missing args_schema")
        if not self.when_to_use.strip():
            errors.append("missing when_to_use")
        if not self.output_description.strip():
            errors.append("missing output_description")
        return errors

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ToolDefinition":
        return cls(
            name=str(raw.get("name") or ""),
            group=str(raw.get("group") or ""),
            builder=raw.get("builder", ""),
            description=str(raw.get("description") or ""),
            args_schema=dict(raw.get("args_schema") or {}),
            when_to_use=str(raw.get("when_to_use") or ""),
            avoid_when=str(raw.get("avoid_when") or ""),
            output_description=str(raw.get("output_description") or ""),
            examples=[str(item) for item in (raw.get("examples") or []) if str(item)],
            keywords=[str(item) for item in (raw.get("keywords") or []) if str(item)],
            read_only=bool(raw.get("read_only", False)),
            destructive=bool(raw.get("destructive", False)),
            background_safe=bool(raw.get("background_safe", False)),
            concurrency_key=str(raw.get("concurrency_key") or ""),
            requires_workspace=bool(raw.get("requires_workspace", False)),
            serializer=str(raw.get("serializer") or "default"),
            should_defer=bool(raw.get("should_defer", False)),
            search_hint=str(raw.get("search_hint") or ""),
            defer_reason=str(raw.get("defer_reason") or ""),
            defer_priority=int(raw.get("defer_priority", 50) or 50),
            eager_for_agents=[str(item) for item in (raw.get("eager_for_agents") or []) if str(item)],
            metadata=dict(raw.get("metadata") or {}),
        )
