from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from agentic_chatbot_next.agents.definitions import REQUIRED_AGENT_FIELDS
from agentic_chatbot_next.agents.validation import validate_agent_frontmatter
from agentic_chatbot_next.contracts.agents import AgentDefinition


@dataclass
class LoadedAgentFile:
    definition: AgentDefinition
    body: str
    source_path: Path


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    if text.startswith("[") or text.startswith("{") or text.startswith("\""):
        return json.loads(text)
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() == "null":
        return None
    if text.lstrip("-").isdigit():
        return int(text)
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        return text[1:-1]
    return text


def _parse_frontmatter(content: str) -> tuple[Dict[str, Any], str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("Agent file is missing opening frontmatter delimiter '---'.")

    closing_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing_index = index
            break
    if closing_index is None:
        raise ValueError("Agent file is missing closing frontmatter delimiter '---'.")

    data: Dict[str, Any] = {}
    for line in lines[1:closing_index]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Invalid frontmatter line: {line!r}")
        key, raw_value = line.split(":", 1)
        data[key.strip()] = _parse_scalar(raw_value)

    body = "\n".join(lines[closing_index + 1 :]).strip()
    return data, body


def load_agent_markdown(path: Path) -> LoadedAgentFile:
    raw = path.read_text(encoding="utf-8")
    return load_agent_markdown_text(raw, source_path=path)


def load_agent_markdown_text(raw: str, *, source_path: Path | str) -> LoadedAgentFile:
    frontmatter, body = _parse_frontmatter(raw)
    missing = sorted(field for field in REQUIRED_AGENT_FIELDS if not frontmatter.get(field))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Agent file {source_path} is missing required frontmatter field(s): {joined}")
    definition = validate_agent_frontmatter(frontmatter, path=source_path)
    return LoadedAgentFile(definition=definition, body=body, source_path=Path(source_path))
