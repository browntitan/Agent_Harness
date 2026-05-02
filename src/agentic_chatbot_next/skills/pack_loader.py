from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_META_PATTERN = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_-]*):\s*(.+?)\s*$")
_FRONTMATTER_DELIMITER = "---"
_COMMON_REQUIRED_METADATA_FIELDS = ("agent_scope", "description", "version", "enabled")
_RETRIEVABLE_REQUIRED_METADATA_FIELDS = ("tool_tags", "task_tags")
_VALID_SKILL_KINDS = {"retrievable", "executable", "hybrid"}
_VALID_EXECUTION_CONTEXTS = {"inline", "fork"}
_VALID_EXECUTION_EFFORTS = {"", "low", "medium", "high", "xhigh"}
_LEGACY_TOOL_TAG_ALIASES = {
    "compare_clauses": "compare_indexed_docs",
    "diff_documents": "compare_indexed_docs",
    "extract_clauses": "read_indexed_doc",
    "extract_requirements": "rag_agent_tool",
    "fetch_chunk_window": "read_indexed_doc",
    "fetch_document_outline": "read_indexed_doc",
    "list_collections": "list_indexed_docs",
    "resolve_document": "resolve_indexed_docs",
    "search_all_documents": "rag_agent_tool",
    "search_collection": "search_indexed_docs",
    "search_document": "read_indexed_doc",
}


@dataclass
class SkillPackFile:
    skill_id: str
    name: str
    agent_scope: str
    body: str
    chunks: List[str]
    checksum: str
    graph_id: str = ""
    collection_id: str = ""
    tool_tags: List[str] = field(default_factory=list)
    task_tags: List[str] = field(default_factory=list)
    version: str = "1"
    enabled: bool = True
    source_path: str = ""
    description: str = ""
    retrieval_profile: str = ""
    controller_hints: Dict[str, Any] = field(default_factory=dict)
    coverage_goal: str = ""
    result_mode: str = ""
    body_markdown: str = ""
    owner_user_id: str = ""
    visibility: str = "global"
    status: str = "active"
    version_parent: str = ""
    keywords: List[str] = field(default_factory=list)
    when_to_apply: str = ""
    avoid_when: str = ""
    examples: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    kind: str = "retrievable"
    execution_config: Dict[str, Any] = field(default_factory=dict)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "skill"


def _split_tags(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(part).strip() for part in raw if str(part).strip()]
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [str(part).strip() for part in parsed if str(part).strip()]
    return [part.strip() for part in text.split(",") if part.strip()]


def _metadata_key(raw: Any) -> str:
    return str(raw or "").strip().lower().replace("-", "_")


def _normalize_metadata(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        _metadata_key(key): value
        for key, value in dict(raw or {}).items()
        if _metadata_key(key)
    }


def _parse_json_object(raw: Any, *, field_name: str) -> Dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items() if str(key).strip()}
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be a JSON object.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return {str(key): value for key, value in parsed.items() if str(key).strip()}


def _parse_optional_positive_int(raw: Any, *, field_name: str) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer.") from exc
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _parse_frontmatter(raw_text: str) -> tuple[Dict[str, str], str]:
    raw = str(raw_text or "")
    lines = raw.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_DELIMITER:
        return {}, raw
    metadata: Dict[str, str] = {}
    closing_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == _FRONTMATTER_DELIMITER:
            closing_index = index
            break
        match = _META_PATTERN.match(lines[index])
        if match:
            metadata[_metadata_key(match.group(1))] = match.group(2).strip()
    if closing_index is None:
        raise ValueError("Skill pack frontmatter is missing its closing '---' delimiter.")
    body = "\n".join(lines[closing_index + 1 :]).strip()
    return metadata, body


def _normalize_tool_tags(tags: List[str]) -> tuple[List[str], List[str]]:
    normalized: List[str] = []
    warnings: List[str] = []
    seen: set[str] = set()
    for tag in tags:
        clean = str(tag or "").strip()
        if not clean:
            continue
        replacement = _LEGACY_TOOL_TAG_ALIASES.get(clean, clean)
        if replacement != clean:
            warnings.append(f"Normalized legacy tool tag '{clean}' to '{replacement}'.")
        if replacement in seen:
            continue
        seen.add(replacement)
        normalized.append(replacement)
    return normalized, warnings


def _metadata_value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return bool(str(value).strip())


def _normalize_skill_kind(raw: Any) -> str:
    kind = str(raw or "retrievable").strip().lower()
    if kind not in _VALID_SKILL_KINDS:
        raise ValueError(
            f"Skill kind must be one of {', '.join(sorted(_VALID_SKILL_KINDS))}."
        )
    return kind


def _normalize_execution_config(metadata: Dict[str, Any], *, kind: str) -> Dict[str, Any]:
    raw_config = _parse_json_object(metadata.get("execution_config"), field_name="execution_config")
    config = _normalize_metadata(raw_config)
    explicit_execution_fields = False
    for key in (
        "allowed_tools",
        "context",
        "model",
        "effort",
        "agent",
        "input_schema",
        "max_steps",
        "max_tool_calls",
    ):
        if key in metadata and metadata.get(key) not in (None, ""):
            explicit_execution_fields = True
            config[key] = metadata.get(key)

    if kind == "retrievable" and not config and not explicit_execution_fields:
        return {}

    allowed_tools = _split_tags(config.get("allowed_tools"))
    context = str(config.get("context") or ("inline" if kind == "executable" else "")).strip().lower()
    if context and context not in _VALID_EXECUTION_CONTEXTS:
        raise ValueError("execution context must be either 'inline' or 'fork'.")
    effort = str(config.get("effort") or "").strip().lower()
    if effort not in _VALID_EXECUTION_EFFORTS:
        raise ValueError("execution effort must be one of low, medium, high, or xhigh.")
    input_schema = _parse_json_object(config.get("input_schema"), field_name="input_schema")
    max_steps = _parse_optional_positive_int(config.get("max_steps"), field_name="max_steps")
    max_tool_calls = _parse_optional_positive_int(config.get("max_tool_calls"), field_name="max_tool_calls")
    normalized = {
        "allowed_tools": allowed_tools,
        "context": context or "inline",
        "model": str(config.get("model") or "").strip(),
        "effort": effort,
        "agent": str(config.get("agent") or "").strip(),
        "input_schema": input_schema,
    }
    if max_steps is not None:
        normalized["max_steps"] = max_steps
    if max_tool_calls is not None:
        normalized["max_tool_calls"] = max_tool_calls
    return normalized


def _validate_metadata(metadata: Dict[str, Any], *, source_path: str, kind: str) -> None:
    required = list(_COMMON_REQUIRED_METADATA_FIELDS)
    if kind in {"retrievable", "hybrid"}:
        required.extend(_RETRIEVABLE_REQUIRED_METADATA_FIELDS)
    missing = [field for field in required if not _metadata_value_present(metadata.get(field))]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Skill pack {source_path or '<memory>'} is missing required metadata field(s): {joined}")


def _coerce_hint_scalar(raw: str) -> Any:
    value = raw.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_controller_hints(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items() if str(key).strip()}
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return {str(key): value for key, value in parsed.items() if str(key).strip()}

    hints: Dict[str, Any] = {}
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            key = key.strip()
            if key:
                hints[key] = _coerce_hint_scalar(value)
            continue
        hints[item] = True
    return hints


def _coerce_enabled(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw or "true").strip().lower() not in {"0", "false", "no"}


def _chunk_skill_body(text: str, *, target_chars: int = 900) -> List[str]:
    sections: List[str] = []
    current: List[str] = []
    current_len = 0
    for line in text.splitlines():
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = [line]
            current_len = len(line)
            continue
        current.append(line)
        current_len += len(line) + 1
        if current_len >= target_chars:
            sections.append("\n".join(current).strip())
            current = []
            current_len = 0
    if current:
        sections.append("\n".join(current).strip())
    return [section for section in sections if section]


def _build_executable_discovery_card(
    *,
    name: str,
    skill_id: str,
    metadata: Dict[str, Any],
    execution_config: Dict[str, Any],
) -> str:
    lines = [
        f"# {name}",
        f"skill_id: {skill_id}",
        f"kind: {metadata.get('kind') or 'executable'}",
        f"agent_scope: {metadata.get('agent_scope') or ''}",
    ]
    description = str(metadata.get("description") or "").strip()
    if description:
        lines.append(f"description: {description}")
    when_to_apply = str(metadata.get("when_to_apply") or "").strip()
    if when_to_apply:
        lines.append(f"when_to_apply: {when_to_apply}")
    allowed_tools = list(execution_config.get("allowed_tools") or [])
    if allowed_tools:
        lines.append("allowed_tools: " + ", ".join(allowed_tools))
    context = str(execution_config.get("context") or "inline").strip()
    if context:
        lines.append(f"context: {context}")
    return "\n".join(line for line in lines if line.strip()).strip()


def load_skill_pack_from_text(
    raw_text: str,
    *,
    source_path: str = "",
    root: Optional[Path] = None,
    metadata_defaults: Optional[Dict[str, Any]] = None,
) -> SkillPackFile:
    raw = str(raw_text or "").strip()
    checksum = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()
    frontmatter, frontmatter_body = _parse_frontmatter(raw)
    lines = frontmatter_body.splitlines() if frontmatter else raw.splitlines()

    source_name = Path(source_path).stem if source_path else "skill"
    name = source_name.replace("_", " ").replace("-", " ").title()
    metadata: Dict[str, Any] = _normalize_metadata(dict(metadata_defaults or {}))
    metadata.update(_normalize_metadata(frontmatter))
    body_start = 0

    for index, line in enumerate(lines):
        if index == 0 and line.startswith("# "):
            name = line[2:].strip() or name
            continue
        if frontmatter and not line.strip():
            continue
        match = _META_PATTERN.match(line)
        if match:
            metadata[_metadata_key(match.group(1))] = match.group(2).strip()
            continue
        if not line.strip():
            continue
        body_start = index
        break

    body = "\n".join(lines[body_start:]).strip()
    if metadata.get("name"):
        name = str(metadata.get("name") or "").strip() or name
    relative = Path(source_path).name if source_path else "skill.md"
    if root is not None and source_path:
        try:
            relative = str(Path(source_path).resolve().relative_to(root.resolve()))
        except Exception:
            relative = Path(source_path).name
    kind = _normalize_skill_kind(metadata.get("kind", "retrievable"))
    execution_config = _normalize_execution_config(metadata, kind=kind)
    _validate_metadata(metadata, source_path=source_path, kind=kind)
    skill_id = metadata.get("skill_id") or _slugify(relative.replace("/", "-"))
    version_parent = metadata.get("version_parent") or skill_id
    normalized_tool_tags, warnings = _normalize_tool_tags(_split_tags(metadata.get("tool_tags", "")))
    keywords = _split_tags(metadata.get("keywords", ""))
    examples = _split_tags(metadata.get("examples", ""))
    application_notes: List[str] = []
    when_to_apply = str(metadata.get("when_to_apply") or "").strip()
    avoid_when = str(metadata.get("avoid_when") or "").strip()
    if keywords:
        application_notes.append("Keywords: " + ", ".join(keywords))
    if when_to_apply:
        application_notes.append("When to apply: " + when_to_apply)
    if avoid_when:
        application_notes.append("Avoid when: " + avoid_when)
    if examples:
        application_notes.append("Examples: " + "; ".join(examples))
    rendered_body = body or raw
    if application_notes:
        rendered_body = "\n\n".join(
            [
                rendered_body,
                "## Application Notes",
                "\n".join(f"- {line}" for line in application_notes),
            ]
        ).strip()
    chunks = (
        _chunk_skill_body(
            _build_executable_discovery_card(
                name=name,
                skill_id=skill_id,
                metadata={**metadata, "kind": kind},
                execution_config=execution_config,
            )
        )
        if kind == "executable"
        else _chunk_skill_body(rendered_body)
    )
    return SkillPackFile(
        skill_id=skill_id,
        name=name,
        agent_scope=metadata.get("agent_scope", "rag"),
        body=rendered_body,
        chunks=chunks,
        checksum=checksum,
        graph_id=metadata.get("graph_id", ""),
        collection_id=metadata.get("collection_id", ""),
        tool_tags=normalized_tool_tags,
        task_tags=_split_tags(metadata.get("task_tags", "")),
        version=metadata.get("version", "1"),
        enabled=_coerce_enabled(metadata.get("enabled", "true")),
        source_path=str(source_path),
        description=metadata.get("description", ""),
        retrieval_profile=metadata.get("retrieval_profile", ""),
        controller_hints=_parse_controller_hints(metadata.get("controller_hints", "")),
        coverage_goal=metadata.get("coverage_goal", ""),
        result_mode=metadata.get("result_mode", ""),
        body_markdown=raw,
        owner_user_id=metadata.get("owner_user_id", ""),
        visibility=metadata.get("visibility", "global"),
        status=metadata.get("status", "active"),
        version_parent=version_parent,
        keywords=keywords,
        when_to_apply=when_to_apply,
        avoid_when=avoid_when,
        examples=examples,
        warnings=warnings,
        kind=kind,
        execution_config=execution_config,
    )


def load_skill_pack_from_file(path: Path, *, root: Optional[Path] = None) -> SkillPackFile:
    return load_skill_pack_from_text(
        path.read_text(encoding="utf-8"),
        source_path=str(path),
        root=root,
    )
