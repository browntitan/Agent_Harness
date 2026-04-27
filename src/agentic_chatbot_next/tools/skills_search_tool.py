from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Tuple

from langchain_core.tools import tool

from agentic_chatbot_next.authz import access_summary_allowed_ids, access_summary_authz_enabled
from agentic_chatbot_next.capabilities import coerce_effective_capabilities
from agentic_chatbot_next.skills.base_loader import SkillsLoader
from agentic_chatbot_next.skills.indexer import SkillContextResolver

logger = logging.getLogger(__name__)

_ALL_SKILL_KEYS = [
    "rag_agent",
    "utility_agent",
    "data_analyst_agent",
    "supervisor_agent",
    "general_agent",
    "planner_agent",
    "finalizer_agent",
    "verifier_agent",
    "basic_chat",
    "shared",
]

_SKIP_HEADINGS = {"tools", "tool", "tools available", "tools section", "rules", "overview"}


@dataclass
class _Section:
    agent_key: str
    heading: str
    content: str
    tokens: set = field(default_factory=set)

    @property
    def key(self) -> str:
        return f"{self.agent_key} > {self.heading}"


def _tokenise(text: str) -> set:
    stop_words = {
        "a", "an", "the", "to", "of", "in", "is", "it", "for", "and", "or",
        "with", "on", "be", "use", "if", "not", "that", "this", "as", "by",
        "at", "all", "any", "when", "how", "what", "which", "do", "should",
    }
    words = re.findall(r"[a-z0-9_]+", text.lower())
    return {word for word in words if len(word) > 2 and word not in stop_words}


def _parse_sections(agent_key: str, content: str) -> List[_Section]:
    sections: List[_Section] = []
    current_heading = agent_key
    current_lines: List[str] = []
    for line in content.splitlines():
        heading_match = re.match(r"^#{2,3}\s+(.+)", line)
        if heading_match:
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body and current_heading.lower() not in _SKIP_HEADINGS:
                    section = _Section(agent_key=agent_key, heading=current_heading, content=body)
                    section.tokens = _tokenise(current_heading + " " + body)
                    sections.append(section)
            current_heading = heading_match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        body = "\n".join(current_lines).strip()
        if body and current_heading.lower() not in _SKIP_HEADINGS:
            section = _Section(agent_key=agent_key, heading=current_heading, content=body)
            section.tokens = _tokenise(current_heading + " " + body)
            sections.append(section)
    return sections


def _score_section(section: _Section, query_tokens: set) -> float:
    title_tokens = _tokenise(section.heading)
    title_hits = len(query_tokens & title_tokens)
    content_hits = len(query_tokens & (section.tokens - title_tokens))
    return title_hits * 3.0 + content_hits * 1.0


def _build_index(settings: object) -> List[_Section]:
    loader = SkillsLoader(settings)
    sections: List[_Section] = []
    for key in _ALL_SKILL_KEYS:
        try:
            raw = loader.load(key)
        except Exception as exc:
            logger.debug("Could not load skill %s: %s", key, exc)
            continue
        if not raw or not raw.strip():
            continue
        sections.extend(_parse_sections(key, raw))
    return sections


def _agent_scope_from_filter(agent_filter: str) -> str:
    mapping = {
        "rag_agent": "rag",
        "utility_agent": "utility",
        "data_analyst_agent": "data_analyst",
        "general_agent": "general",
        "planner_agent": "planner",
        "finalizer_agent": "finalizer",
        "verifier_agent": "verifier",
    }
    return mapping.get(agent_filter.strip(), agent_filter.strip())


def make_skills_search_tool(settings: object, *, stores: Any | None = None, session: Any | None = None):
    index = _build_index(settings)
    resolver = SkillContextResolver(settings, stores) if stores is not None else None

    @tool
    def search_skills(query: str, agent_filter: str = "", top_k: int = 3) -> str:
        """Search the skills library for operational guidance."""

        bounded_top_k = min(max(1, top_k), 5)
        session_metadata = dict(getattr(session, "metadata", {}) or {})
        effective_capabilities = coerce_effective_capabilities(session_metadata.get("effective_capabilities"))
        if resolver is not None:
            try:
                access_summary = session_metadata.get("access_summary") or {}
                accessible_skill_family_ids = (
                    list(access_summary_allowed_ids(access_summary, "skill_family", action="use"))
                    if access_summary_authz_enabled(access_summary)
                    else None
                )
                context = resolver.resolve(
                    query=query,
                    tenant_id=getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev")),
                    owner_user_id=getattr(session, "user_id", getattr(settings, "default_user_id", "")),
                    agent_scope=_agent_scope_from_filter(agent_filter),
                    top_k=bounded_top_k,
                    accessible_skill_family_ids=accessible_skill_family_ids,
                )
                if context.matches:
                    parts = []
                    for match in context.matches:
                        family_id = str(getattr(match, "version_parent", "") or getattr(match, "skill_id", "") or "")
                        if effective_capabilities is not None and not effective_capabilities.allows_skill(
                            match.skill_id,
                            family_id=family_id,
                        ):
                            continue
                        body = match.content
                        if len(body) > 800:
                            body = body[:800] + "\n...[truncated]"
                        parts.append(f"[{match.name} | {match.skill_id} | {match.agent_scope}]\n{body}")
                        if len(parts) >= bounded_top_k:
                            break
                    if parts:
                        return "\n\n---\n\n".join(parts)
                    return "No matching skill sections found in the enabled skill packs."
            except Exception as exc:
                logger.warning("DB-backed search_skills failed, falling back to lexical index: %s", exc)

        query_tokens = _tokenise(query)
        if not query_tokens:
            return "No results: query produced no searchable tokens."

        candidates = [
            section
            for section in index
            if (not agent_filter or section.agent_key == agent_filter.strip())
            and (
                effective_capabilities is None
                or effective_capabilities.allows_skill(section.agent_key, family_id=section.agent_key)
            )
        ]
        if not candidates:
            return f"No sections found for agent_filter={agent_filter!r}."

        scored: List[Tuple[float, _Section]] = [
            (_score_section(section, query_tokens), section) for section in candidates
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        top = [(score, section) for score, section in scored[:bounded_top_k] if score > 0]
        if not top:
            query_lower = query.lower()
            fallback = [
                (1.0, section)
                for section in candidates
                if any(word in section.content.lower() or word in section.heading.lower() for word in query_lower.split())
            ][:bounded_top_k]
            if fallback:
                top = fallback
            else:
                return f"No matching skill sections found for query: {query!r}. Try broader terms."

        parts: List[str] = []
        for _, section in top:
            body = section.content
            if len(body) > 800:
                body = body[:800] + "\n...[truncated]"
            parts.append(f"[{section.key}]\n{body}")
        return "\n\n---\n\n".join(parts)

    return search_skills
