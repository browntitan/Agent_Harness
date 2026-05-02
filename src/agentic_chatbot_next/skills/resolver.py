from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from agentic_chatbot_next.skills.indexer import SkillContextResolver as NextSkillContextResolver


@dataclass
class SkillMatch:
    skill_id: str
    skill_family_id: str
    name: str
    agent_scope: str
    content: str
    chunk_index: int
    score: float
    graph_id: str = ""
    collection_id: str = ""
    retrieval_profile: str = ""
    controller_hints: dict[str, Any] = field(default_factory=dict)
    coverage_goal: str = ""
    result_mode: str = ""
    source_path: str = ""
    checksum: str = ""
    description: str = ""
    kind: str = "retrievable"


@dataclass
class ResolvedSkillContext:
    text: str
    matches: List[SkillMatch] = field(default_factory=list)
    resolved_skill_families: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "matches": [
                {
                    "skill_id": match.skill_id,
                    "skill_family_id": match.skill_family_id,
                    "name": match.name,
                    "agent_scope": match.agent_scope,
                    "chunk_index": match.chunk_index,
                    "score": match.score,
                    "graph_id": match.graph_id,
                    "collection_id": match.collection_id,
                    "retrieval_profile": match.retrieval_profile,
                    "coverage_goal": match.coverage_goal,
                    "result_mode": match.result_mode,
                    "source_path": match.source_path,
                    "checksum": match.checksum,
                    "description": match.description,
                    "kind": match.kind,
                }
                for match in self.matches
            ],
            "resolved_skill_families": list(self.resolved_skill_families),
            "warnings": list(self.warnings),
        }


class SkillResolver:
    def __init__(self, settings: Any, stores: Any) -> None:
        self._resolver = NextSkillContextResolver(settings, stores)

    def resolve(
        self,
        *,
        query: str,
        tenant_id: str,
        agent_scope: str,
        tool_tags: List[str] | None = None,
        task_tags: List[str] | None = None,
        owner_user_id: str | None = None,
        graph_ids: List[str] | None = None,
        collection_ids: List[str] | None = None,
        top_k: int | None = None,
        max_chars: int | None = None,
        pinned_skill_ids: List[str] | None = None,
        accessible_skill_family_ids: List[str] | None = None,
    ) -> ResolvedSkillContext:
        result = self._resolver.resolve(
            query=query,
            tenant_id=tenant_id,
            agent_scope=agent_scope,
            tool_tags=tool_tags,
            task_tags=task_tags,
            owner_user_id=str(owner_user_id or ""),
            graph_ids=list(graph_ids or []),
            collection_ids=list(collection_ids) if collection_ids is not None else None,
            top_k=top_k,
            max_chars=max_chars,
            pinned_skill_ids=list(pinned_skill_ids or []),
            accessible_skill_family_ids=list(accessible_skill_family_ids or []) if accessible_skill_family_ids is not None else None,
        )
        return ResolvedSkillContext(
            text=result.text,
            matches=[
                SkillMatch(
                    skill_id=match.skill_id,
                    skill_family_id=str(match.version_parent or match.skill_id),
                    name=match.name,
                    agent_scope=match.agent_scope,
                    content=match.content,
                    chunk_index=match.chunk_index,
                    score=match.score,
                    graph_id=match.graph_id,
                    collection_id=match.collection_id,
                    retrieval_profile=match.retrieval_profile,
                    controller_hints=dict(match.controller_hints),
                    coverage_goal=match.coverage_goal,
                    result_mode=match.result_mode,
                    source_path=getattr(match, "source_path", ""),
                    checksum=getattr(match, "checksum", ""),
                    description=getattr(match, "description", ""),
                    kind=getattr(match, "kind", "retrievable"),
                )
                for match in result.matches
            ],
            resolved_skill_families=list(result.resolved_skill_families),
            warnings=list(result.warnings),
        )
