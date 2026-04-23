from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from agentic_chatbot_next.config import Settings
from agentic_chatbot_next.prompt_fallbacks import fallback_prompt_for_key, fallback_shared_prompt

logger = logging.getLogger(__name__)

_DEFAULTS: Dict[str, str] = {
    "shared": fallback_shared_prompt(),
    "general_agent": fallback_prompt_for_key("general_agent"),
    "rag_agent": fallback_prompt_for_key("rag_agent"),
    "supervisor_agent": fallback_prompt_for_key("supervisor_agent"),
    "utility_agent": fallback_prompt_for_key("utility_agent"),
    "basic_chat": fallback_prompt_for_key("basic_chat"),
    "planner_agent": fallback_prompt_for_key("planner_agent"),
    "finalizer_agent": fallback_prompt_for_key("finalizer_agent"),
    "verifier_agent": fallback_prompt_for_key("verifier_agent"),
    "data_analyst_agent": fallback_prompt_for_key("data_analyst_agent"),
}

_REQUIRED_SECTIONS: Dict[str, list[str]] = {
    "general_agent": [
        "mission",
        "capabilities and limits",
        "task intake and clarification rules",
        "tool and delegation policy",
        "failure recovery",
        "output shaping",
        "anti-patterns and avoid rules",
    ],
    "basic_chat": [
        "mission",
        "capabilities and limits",
        "task intake and clarification rules",
        "output shaping",
    ],
    "utility_agent": [
        "mission",
        "capabilities and limits",
        "tool and delegation policy",
        "output shaping",
    ],
    "data_analyst_agent": [
        "mission",
        "capabilities and limits",
        "tool and delegation policy",
        "failure recovery",
    ],
    "graph_manager_agent": [
        "mission",
        "capabilities and limits",
        "tool and delegation policy",
    ],
    "planner_agent": [
        "mission",
        "capabilities and limits",
        "task intake and clarification rules",
        "output shaping",
    ],
    "finalizer_agent": [
        "mission",
        "capabilities and limits",
        "output shaping",
        "anti-patterns and avoid rules",
    ],
    "verifier_agent": [
        "mission",
        "capabilities and limits",
        "output shaping",
    ],
    "supervisor_agent": [
        "mission",
        "capabilities and limits",
        "tool and delegation policy",
        "failure recovery",
    ],
    "rag_agent": [
        "mission",
        "capabilities and limits",
        "task intake and clarification rules",
        "output shaping",
    ],
}


@dataclass
class _CacheEntry:
    content: str
    mtime: float
    path: Path


class SkillsLoader:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache: Dict[str, _CacheEntry] = {}

    def load(
        self,
        agent_key: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        shared = self._load_file("shared") or ""
        specific = self._load_file(agent_key)
        body = specific if specific is not None else _DEFAULTS.get(agent_key, "")
        prompt = f"{shared}\n\n---\n\n{body}".strip() if shared else body.strip()
        if context:
            from agentic_chatbot_next.prompting import render_template

            prompt = render_template(prompt, context)
        return prompt

    def invalidate(self, agent_key: Optional[str] = None) -> None:
        if agent_key is None:
            self._cache.clear()
        else:
            self._cache.pop(agent_key, None)

    def _load_file(self, agent_key: str) -> Optional[str]:
        path = self._get_path(agent_key)
        if path is None:
            return None
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            self._cache.pop(agent_key, None)
            return None
        except OSError as exc:
            logger.warning("Could not stat skill file %s: %s", path, exc)
            return None

        cached = self._cache.get(agent_key)
        if cached is not None and cached.mtime == mtime:
            return cached.content

        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Could not read skill file %s: %s", path, exc)
            return None
        if not content:
            self._cache.pop(agent_key, None)
            return None

        self._validate_sections(agent_key, content, path)
        self._cache[agent_key] = _CacheEntry(content=content, mtime=mtime, path=path)
        return content

    def _get_path(self, agent_key: str) -> Optional[Path]:
        settings = self._settings
        skills_dir = getattr(settings, "skills_dir", None)
        if not isinstance(skills_dir, Path):
            skills_dir = None
        verifier_path = getattr(settings, "verifier_agent_skills_path", None)
        if not isinstance(verifier_path, Path):
            verifier_path = None
        mapping: Dict[str, Optional[Path]] = {
            "shared": getattr(settings, "shared_skills_path", None),
            "general_agent": getattr(settings, "general_agent_skills_path", None),
            "rag_agent": getattr(settings, "rag_agent_skills_path", None),
            "supervisor_agent": getattr(settings, "supervisor_agent_skills_path", None),
            "utility_agent": getattr(settings, "utility_agent_skills_path", None),
            "basic_chat": getattr(settings, "basic_chat_skills_path", None),
            "planner_agent": getattr(settings, "planner_agent_skills_path", None),
            "finalizer_agent": getattr(settings, "finalizer_agent_skills_path", None),
            "data_analyst_agent": getattr(settings, "data_analyst_skills_path", None),
            "verifier_agent": verifier_path or (skills_dir / "verifier_agent.md" if skills_dir is not None else None),
        }
        return mapping.get(agent_key)

    def _validate_sections(self, agent_key: str, content: str, path: Path) -> None:
        for section in _REQUIRED_SECTIONS.get(agent_key, []):
            if section.lower() not in content.lower():
                logger.warning(
                    "Skill file %s (agent=%r) is missing expected section %r.",
                    path,
                    agent_key,
                    section,
                )


_LOADER_CACHE: Dict[int, SkillsLoader] = {}


def get_skills_loader(settings: Settings) -> SkillsLoader:
    key = id(settings)
    if key not in _LOADER_CACHE:
        _LOADER_CACHE[key] = SkillsLoader(settings)
    return _LOADER_CACHE[key]


def load_shared_skills(settings: Settings) -> str:
    loader = get_skills_loader(settings)
    return loader._load_file("shared") or ""  # noqa: SLF001


def _load(settings: Settings, key: str, *, context: Optional[Dict[str, Any]] = None) -> str:
    return get_skills_loader(settings).load(key, context=context)


def load_general_agent_skills(settings: Settings, *, context: Optional[Dict[str, Any]] = None) -> str:
    return _load(settings, "general_agent", context=context)


def load_rag_agent_skills(settings: Settings, *, context: Optional[Dict[str, Any]] = None) -> str:
    return _load(settings, "rag_agent", context=context)


def load_supervisor_skills(settings: Settings, *, context: Optional[Dict[str, Any]] = None) -> str:
    return _load(settings, "supervisor_agent", context=context)


def load_utility_agent_skills(settings: Settings, *, context: Optional[Dict[str, Any]] = None) -> str:
    return _load(settings, "utility_agent", context=context)


def load_basic_chat_skills(settings: Settings, *, context: Optional[Dict[str, Any]] = None) -> str:
    return _load(settings, "basic_chat", context=context)


def load_data_analyst_skills(settings: Settings, *, context: Optional[Dict[str, Any]] = None) -> str:
    return _load(settings, "data_analyst_agent", context=context)


def load_planner_agent_skills(settings: Settings, *, context: Optional[Dict[str, Any]] = None) -> str:
    return _load(settings, "planner_agent", context=context)


def load_finalizer_agent_skills(settings: Settings, *, context: Optional[Dict[str, Any]] = None) -> str:
    return _load(settings, "finalizer_agent", context=context)


def load_verifier_agent_skills(settings: Settings, *, context: Optional[Dict[str, Any]] = None) -> str:
    return _load(settings, "verifier_agent", context=context)
