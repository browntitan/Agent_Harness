from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agentic_chatbot_next.config import Settings
from agentic_chatbot_next.persistence.postgres.skills import SkillChunkMatch, SkillPackRecord
from agentic_chatbot_next.persistence.postgres.skills import SkillStore
from agentic_chatbot_next.rag.stores import KnowledgeStores
from agentic_chatbot_next.skills.dependency_graph import build_skill_dependency_graph
from agentic_chatbot_next.skills.pack_loader import load_skill_pack_from_file


@dataclass
class SkillContext:
    text: str
    matches: List[SkillChunkMatch] = field(default_factory=list)
    resolved_skill_families: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class SkillIndexSync:
    def __init__(self, settings: Settings, stores: KnowledgeStores) -> None:
        self.settings = settings
        self.stores = stores

    def _skill_packs_root(self) -> Path:
        value = getattr(self.settings, "skill_packs_dir", None)
        return Path(value) if value is not None else Path("data") / "skill_packs"

    def iter_skill_files(self) -> Iterable[Path]:
        root = self._skill_packs_root()
        if not root.exists():
            return []
        return sorted(path for path in root.rglob("*.md") if path.is_file())

    def sync(self, *, tenant_id: str) -> Dict[str, Any]:
        indexed: List[Dict[str, Any]] = []
        root = self._skill_packs_root()
        for path in self.iter_skill_files():
            pack = load_skill_pack_from_file(path, root=root)
            self.stores.skill_store.upsert_skill_pack(
                SkillPackRecord(
                    skill_id=pack.skill_id,
                    name=pack.name,
                    agent_scope=pack.agent_scope,
                    checksum=pack.checksum,
                    tenant_id=tenant_id,
                    graph_id=pack.graph_id,
                    tool_tags=pack.tool_tags,
                    task_tags=pack.task_tags,
                    version=pack.version,
                    enabled=pack.enabled,
                    source_path=pack.source_path,
                    description=pack.description,
                    retrieval_profile=pack.retrieval_profile,
                    controller_hints=dict(pack.controller_hints),
                    coverage_goal=pack.coverage_goal,
                    result_mode=pack.result_mode,
                    body_markdown=pack.body_markdown,
                    owner_user_id=pack.owner_user_id,
                    visibility=pack.visibility,
                    status=pack.status,
                    version_parent=pack.version_parent or pack.skill_id,
                    kind=pack.kind,
                    execution_config=dict(pack.execution_config),
                ),
                pack.chunks,
            )
            indexed.append(
                {
                    "skill_id": pack.skill_id,
                    "name": pack.name,
                    "agent_scope": pack.agent_scope,
                    "chunks": len(pack.chunks),
                    "source_path": pack.source_path,
                    "kind": pack.kind,
                }
            )
        records = self.stores.skill_store.list_skill_packs(tenant_id=tenant_id)
        dependency_graph = build_skill_dependency_graph(records)
        summary = dependency_graph.summary()
        return {
            "indexed": indexed,
            "count": len(indexed),
            "dependency_graph": summary.to_dict(),
            "valid": summary.valid,
        }

    def sync_changed(self, *, tenant_id: str) -> Dict[str, Any]:
        changed: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        root = self._skill_packs_root()
        for path in self.iter_skill_files():
            pack = load_skill_pack_from_file(path, root=root)
            existing = self.stores.skill_store.get_skill_pack(
                pack.skill_id,
                tenant_id=tenant_id,
            )
            if existing is not None and str(existing.checksum or "") == pack.checksum:
                skipped.append({"skill_id": pack.skill_id, "source_path": pack.source_path})
                continue
            self.stores.skill_store.upsert_skill_pack(
                SkillPackRecord(
                    skill_id=pack.skill_id,
                    name=pack.name,
                    agent_scope=pack.agent_scope,
                    checksum=pack.checksum,
                    tenant_id=tenant_id,
                    graph_id=pack.graph_id,
                    tool_tags=pack.tool_tags,
                    task_tags=pack.task_tags,
                    version=pack.version,
                    enabled=pack.enabled,
                    source_path=pack.source_path,
                    description=pack.description,
                    retrieval_profile=pack.retrieval_profile,
                    controller_hints=dict(pack.controller_hints),
                    coverage_goal=pack.coverage_goal,
                    result_mode=pack.result_mode,
                    body_markdown=pack.body_markdown,
                    owner_user_id=pack.owner_user_id,
                    visibility=pack.visibility,
                    status=pack.status,
                    version_parent=pack.version_parent or pack.skill_id,
                    kind=pack.kind,
                    execution_config=dict(pack.execution_config),
                ),
                pack.chunks,
            )
            changed.append(
                {
                    "skill_id": pack.skill_id,
                    "name": pack.name,
                    "agent_scope": pack.agent_scope,
                    "chunks": len(pack.chunks),
                    "source_path": pack.source_path,
                    "kind": pack.kind,
                }
            )
        records = self.stores.skill_store.list_skill_packs(tenant_id=tenant_id)
        summary = build_skill_dependency_graph(records).summary()
        return {
            "changed": changed,
            "skipped": skipped,
            "changed_count": len(changed),
            "skipped_count": len(skipped),
            "dependency_graph": summary.to_dict(),
            "valid": summary.valid,
        }


class SkillContextResolver:
    def __init__(self, settings: Settings, stores: KnowledgeStores) -> None:
        self.settings = settings
        self.stores = stores

    def resolve(
        self,
        *,
        query: str,
        tenant_id: str,
        agent_scope: str,
        tool_tags: Optional[List[str]] = None,
        task_tags: Optional[List[str]] = None,
        owner_user_id: str = "",
        graph_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        max_chars: Optional[int] = None,
        pinned_skill_ids: Optional[List[str]] = None,
        accessible_skill_family_ids: Optional[List[str]] = None,
    ) -> SkillContext:
        list_skill_packs = getattr(self.stores.skill_store, "list_skill_packs", None)
        if callable(list_skill_packs):
            try:
                visible_records = list_skill_packs(
                    tenant_id=tenant_id,
                    agent_scope=agent_scope,
                    owner_user_id=owner_user_id,
                    accessible_skill_family_ids=accessible_skill_family_ids,
                )
            except TypeError:
                visible_records = list_skill_packs(
                    tenant_id=tenant_id,
                    agent_scope=agent_scope,
                )
        else:
            visible_records = []
        dependency_graph = build_skill_dependency_graph(visible_records)
        try:
            matches = self.stores.skill_store.vector_search(
                query,
                tenant_id=tenant_id,
                top_k=top_k or int(getattr(self.settings, "skill_search_top_k", 4)),
                agent_scope=agent_scope,
                tool_tags=tool_tags,
                task_tags=task_tags,
                enabled_only=True,
                owner_user_id=owner_user_id,
                graph_ids=graph_ids,
                accessible_skill_family_ids=accessible_skill_family_ids,
            )
        except TypeError:
            matches = self.stores.skill_store.vector_search(
                query,
                tenant_id=tenant_id,
                top_k=top_k or int(getattr(self.settings, "skill_search_top_k", 4)),
                agent_scope=agent_scope,
                tool_tags=tool_tags,
                task_tags=task_tags,
                enabled_only=True,
                owner_user_id=owner_user_id,
                graph_ids=graph_ids,
            )
        limit = max_chars or int(getattr(self.settings, "skill_context_max_chars", 3000))
        parts: List[str] = []
        consumed = 0
        resolved_matches: List[SkillChunkMatch] = []
        seen_skill_ids: set[str] = set()
        resolved_skill_families: List[str] = []
        warnings: List[str] = []

        for pinned_identifier in list(pinned_skill_ids or []):
            record = dependency_graph.active_record_for_identifier(str(pinned_identifier))
            if record is None:
                warnings.append(f"Pinned skill '{pinned_identifier}' could not be resolved to an active skill.")
                continue
            family_id = str(record.version_parent or record.skill_id or "")
            if family_id in dependency_graph.invalid_families:
                warnings.append(
                    f"Pinned skill family '{family_id}' was skipped because its active dependency graph is invalid."
                )
                continue
            body = str(record.body_markdown or "").strip()
            if not body:
                chunk_rows = self.stores.skill_store.get_skill_chunks(record.skill_id, tenant_id=tenant_id)
                body = "\n\n".join(str(item.get("content") or "").strip() for item in chunk_rows if str(item.get("content") or "").strip())
            if not body:
                continue
            block = f"[Pinned: {record.name} | {record.skill_id}]\n{body}"
            if consumed + len(block) > limit and parts:
                break
            parts.append(block)
            consumed += len(block) + 2
            resolved_matches.append(
                SkillChunkMatch(
                    skill_id=record.skill_id,
                    name=record.name,
                    agent_scope=record.agent_scope,
                    content=body,
                    chunk_index=0,
                    score=1.0,
                    tool_tags=list(record.tool_tags),
                    task_tags=list(record.task_tags),
                    retrieval_profile=record.retrieval_profile,
                    controller_hints=dict(record.controller_hints),
                    coverage_goal=record.coverage_goal,
                    result_mode=record.result_mode,
                    owner_user_id=record.owner_user_id,
                    visibility=record.visibility,
                    status=record.status,
                    version_parent=record.version_parent or record.skill_id,
                    kind=record.kind,
                    execution_config=dict(record.execution_config),
                )
            )
            seen_skill_ids.add(record.skill_id)
            if family_id and family_id not in resolved_skill_families:
                resolved_skill_families.append(family_id)

        for match in matches:
            if match.skill_id in seen_skill_ids:
                continue
            family_id = str(match.version_parent or match.skill_id or "")
            if family_id in dependency_graph.invalid_families:
                warnings.append(
                    f"Retrieved skill family '{family_id}' was skipped because its active dependency graph is invalid."
                )
                continue
            block = f"[{match.name} | {match.skill_id}]\n{match.content.strip()}"
            if consumed + len(block) > limit and parts:
                break
            parts.append(block)
            consumed += len(block) + 2
            resolved_matches.append(match)
            if family_id and family_id not in resolved_skill_families:
                resolved_skill_families.append(family_id)
        return SkillContext(
            text="\n\n---\n\n".join(parts).strip(),
            matches=resolved_matches,
            resolved_skill_families=resolved_skill_families,
            warnings=warnings,
        )
