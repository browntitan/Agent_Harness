from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from agentic_chatbot_next.prompting import render_template


EXECUTABLE_SKILL_KINDS = {"executable", "hybrid"}
VALID_SKILL_CONTEXTS = {"inline", "fork"}
VALID_SKILL_EFFORTS = {"", "low", "medium", "high", "xhigh"}


def _string_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(item).strip() for item in raw if str(item).strip()]
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip() for part in text.split(",") if part.strip()]


def _positive_int(raw: Any) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


@dataclass(frozen=True)
class SkillExecutionConfig:
    allowed_tools: List[str] = field(default_factory=list)
    context: str = "inline"
    model: str = ""
    effort: str = ""
    agent: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    max_steps: int | None = None
    max_tool_calls: int | None = None

    @classmethod
    def from_raw(cls, raw: Dict[str, Any] | None) -> "SkillExecutionConfig":
        payload = dict(raw or {})
        context = str(payload.get("context") or "inline").strip().lower()
        if context not in VALID_SKILL_CONTEXTS:
            context = "inline"
        effort = str(payload.get("effort") or "").strip().lower()
        if effort not in VALID_SKILL_EFFORTS:
            effort = ""
        input_schema = payload.get("input_schema") if isinstance(payload.get("input_schema"), dict) else {}
        return cls(
            allowed_tools=_string_list(payload.get("allowed_tools")),
            context=context,
            model=str(payload.get("model") or "").strip(),
            effort=effort,
            agent=str(payload.get("agent") or "").strip(),
            input_schema=dict(input_schema or {}),
            max_steps=_positive_int(payload.get("max_steps")),
            max_tool_calls=_positive_int(payload.get("max_tool_calls")),
        )

    @classmethod
    def from_record(cls, record: Any) -> "SkillExecutionConfig":
        return cls.from_raw(dict(getattr(record, "execution_config", {}) or {}))

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return {
            key: value
            for key, value in payload.items()
            if value not in ("", None) and not (key == "input_schema" and value == {})
        }


@dataclass(frozen=True)
class SkillExecutionPreview:
    skill_id: str
    name: str
    kind: str
    context: str
    rendered_prompt: str
    allowed_tools: List[str] = field(default_factory=list)
    model: str = ""
    effort: str = ""
    agent: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkillExecutionResult:
    skill_id: str
    context: str
    status: str
    result: str = ""
    job_id: str = ""
    allowed_tools: List[str] = field(default_factory=list)
    model: str = ""
    effort: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object": "skill.execution_result",
            **asdict(self),
        }


def executable_skill_enabled(record: Any) -> bool:
    return str(getattr(record, "kind", "retrievable") or "retrievable").strip().lower() in EXECUTABLE_SKILL_KINDS


def _has_template_token(text: str) -> bool:
    return "{{" in text and "}}" in text


def render_skill_execution_prompt(
    record: Any,
    *,
    input_text: str = "",
    arguments: Dict[str, Any] | None = None,
    config: SkillExecutionConfig | None = None,
) -> str:
    active_config = config or SkillExecutionConfig.from_record(record)
    args = dict(arguments or {})
    body = str(getattr(record, "body_markdown", "") or "").strip()
    if not body:
        body = str(getattr(record, "description", "") or "").strip()
    values = {
        "input": input_text,
        "INPUT": input_text,
        "arguments": args,
        "ARGUMENTS": args,
        "arguments_json": json.dumps(args, ensure_ascii=False, indent=2),
        "ARGUMENTS_JSON": json.dumps(args, ensure_ascii=False, indent=2),
        "skill_id": str(getattr(record, "skill_id", "") or ""),
        "name": str(getattr(record, "name", "") or ""),
    }
    if _has_template_token(body):
        rendered = render_template(body, values).strip()
    else:
        task_lines = ["## Skill Invocation"]
        if input_text.strip():
            task_lines.extend(["Input:", input_text.strip()])
        if args:
            task_lines.extend(["Arguments JSON:", json.dumps(args, ensure_ascii=False, indent=2)])
        if active_config.effort:
            task_lines.append(f"Reasoning effort: {active_config.effort}")
        rendered = "\n\n".join(
            [
                "\n".join(task_lines).strip(),
                "## Skill Instructions",
                body,
            ]
        ).strip()
    return rendered


def build_skill_execution_preview(
    record: Any,
    *,
    input_text: str = "",
    arguments: Dict[str, Any] | None = None,
) -> SkillExecutionPreview:
    config = SkillExecutionConfig.from_record(record)
    prompt = render_skill_execution_prompt(
        record,
        input_text=input_text,
        arguments=arguments,
        config=config,
    )
    return SkillExecutionPreview(
        skill_id=str(getattr(record, "skill_id", "") or ""),
        name=str(getattr(record, "name", "") or ""),
        kind=str(getattr(record, "kind", "retrievable") or "retrievable"),
        context=config.context,
        rendered_prompt=prompt,
        allowed_tools=list(config.allowed_tools),
        model=config.model,
        effort=config.effort,
        agent=config.agent,
    )
