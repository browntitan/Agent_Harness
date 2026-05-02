from __future__ import annotations

import logging
from typing import Any, Dict, List

from agentic_chatbot_next.authz import access_summary_allowed_ids, access_summary_allows, access_summary_authz_enabled
from agentic_chatbot_next.agents.prompt_builder import PromptBuilder
from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.prompt_fallbacks import compose_fallback_prompt
from agentic_chatbot_next.skills.query_builder import build_skill_resolver_query
from agentic_chatbot_next.skills.resolver import ResolvedSkillContext, SkillResolver

logger = logging.getLogger(__name__)


class SkillRuntime:
    def __init__(self, settings: Any, stores: Any, prompt_builder: PromptBuilder) -> None:
        self.settings = settings
        self.stores = stores
        self.prompt_builder = prompt_builder
        self.resolver = SkillResolver(settings, stores)

    def resolve_context(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        task_payload: Dict[str, Any] | None = None,
    ) -> ResolvedSkillContext:
        if not agent.skill_scope:
            return ResolvedSkillContext(text="")
        payload = dict(task_payload or {})
        worker_request = dict(payload.get("worker_request") or {})
        skill_queries = [
            str(item).strip()
            for item in (payload.get("skill_queries") or [])
            if str(item).strip()
        ]
        skill_queries.extend(
            str(item).strip()
            for item in (worker_request.get("skill_queries") or [])
            if str(item).strip()
        )
        graph_ids = self._active_graph_ids(session_state, payload)
        collection_ids = self._active_collection_ids(session_state, payload, graph_ids=graph_ids)
        access_summary = dict(session_state.metadata.get("access_summary") or {})
        accessible_skill_family_ids = (
            list(access_summary_allowed_ids(access_summary, "skill_family", action="use"))
            if access_summary_authz_enabled(access_summary)
            else None
        )
        pinned_skill_ids = [
            *list(agent.preload_skill_packs),
            *self._graph_skill_ids(session_state.tenant_id, session_state.user_id, graph_ids, access_summary=access_summary),
            *self._restored_skill_ids(session_state),
        ]
        base_query = str(worker_request.get("semantic_query") or "").strip() or user_text.strip()
        resolver_query = build_skill_resolver_query(
            self.settings,
            base_query=base_query,
            skill_queries=skill_queries,
        )
        if not resolver_query:
            return ResolvedSkillContext(text="")
        try:
            resolved = self.resolver.resolve(
                query=resolver_query,
                tenant_id=session_state.tenant_id,
                owner_user_id=session_state.user_id,
                agent_scope=agent.skill_scope,
                tool_tags=list(agent.allowed_tools),
                graph_ids=graph_ids,
                collection_ids=collection_ids,
                pinned_skill_ids=self._dedupe(pinned_skill_ids),
                accessible_skill_family_ids=accessible_skill_family_ids,
            )
            return resolved
        except Exception as exc:
            logger.warning("Skill resolution failed for %s: %s", agent.name, exc)
            return ResolvedSkillContext(
                text="",
                warnings=[f"Skill resolution failed for {agent.name}: {exc}"],
            )

    def build_prompt(self, agent: AgentDefinition) -> str:
        shared = self.prompt_builder.load_shared_prompt().strip()
        specific = ""
        try:
            specific = self.prompt_builder.load_prompt(agent.prompt_file).strip()
        except FileNotFoundError:
            specific = ""
        prompt = "\n\n---\n\n".join(part for part in [shared, specific] if part)
        return prompt.strip() or compose_fallback_prompt(agent.prompt_file)

    def _dedupe(self, items: List[str]) -> List[str]:
        seen: set[str] = set()
        result: List[str] = []
        for item in items:
            clean = str(item or "").strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            result.append(clean)
        return result

    def _active_graph_ids(self, session_state: SessionState, payload: Dict[str, Any]) -> List[str]:
        worker_request = dict(payload.get("worker_request") or {})
        controller_hints = dict(payload.get("controller_hints") or {})
        worker_hints = dict(worker_request.get("controller_hints") or {})
        graph_ids = [
            *[str(item) for item in (session_state.metadata.get("active_graph_ids") or []) if str(item).strip()],
            *[str(item) for item in (controller_hints.get("graph_ids") or []) if str(item).strip()],
            *[str(item) for item in (controller_hints.get("planned_graph_ids") or []) if str(item).strip()],
            *[str(item) for item in (worker_hints.get("graph_ids") or []) if str(item).strip()],
            *[str(item) for item in (worker_hints.get("planned_graph_ids") or []) if str(item).strip()],
        ]
        return self._dedupe(graph_ids)

    def _active_collection_ids(
        self,
        session_state: SessionState,
        payload: Dict[str, Any],
        *,
        graph_ids: List[str],
    ) -> List[str]:
        worker_request = dict(payload.get("worker_request") or {})
        controller_hints = dict(payload.get("controller_hints") or {})
        worker_hints = dict(worker_request.get("controller_hints") or {})
        metadata = dict(session_state.metadata or {})
        collection_ids: List[str] = []
        for source in (metadata, controller_hints, worker_hints, payload, worker_request):
            for key in ("active_collection_ids", "search_collection_ids", "collection_ids", "kb_collection_ids"):
                raw = source.get(key)
                if isinstance(raw, str):
                    collection_ids.extend(part.strip() for part in raw.split(",") if part.strip())
                else:
                    collection_ids.extend(str(item) for item in (raw or []) if str(item).strip())
            for key in ("collection_id", "kb_collection_id", "requested_kb_collection_id", "upload_collection_id"):
                value = str(source.get(key) or "").strip()
                if value:
                    collection_ids.append(value)
        graph_store = getattr(self.stores, "graph_index_store", None)
        if graph_store is not None:
            for graph_id in graph_ids:
                try:
                    record = graph_store.get_index(graph_id, session_state.tenant_id, user_id="*")
                except TypeError:
                    try:
                        record = graph_store.get_index(graph_id, session_state.tenant_id)
                    except Exception:
                        record = None
                except Exception:
                    record = None
                value = str(getattr(record, "collection_id", "") or "").strip() if record is not None else ""
                if value:
                    collection_ids.append(value)
        return self._dedupe(collection_ids)

    def _graph_skill_ids(
        self,
        tenant_id: str,
        user_id: str,
        graph_ids: List[str],
        *,
        access_summary: Dict[str, Any] | None = None,
    ) -> List[str]:
        graph_store = getattr(self.stores, "graph_index_store", None)
        skill_store = getattr(self.stores, "skill_store", None)
        skill_ids: List[str] = []
        scoped_access_summary = dict(access_summary or {})
        accessible_skill_family_ids = (
            list(access_summary_allowed_ids(scoped_access_summary, "skill_family", action="use"))
            if access_summary_authz_enabled(scoped_access_summary)
            else None
        )
        for graph_id in graph_ids:
            if access_summary_authz_enabled(scoped_access_summary) and not access_summary_allows(
                scoped_access_summary,
                "graph",
                graph_id,
                action="use",
            ):
                continue
            if graph_store is not None:
                try:
                    record = graph_store.get_index(graph_id, tenant_id, user_id="*")
                except TypeError:
                    record = graph_store.get_index(graph_id, tenant_id)
                except Exception:
                    record = None
                if record is not None:
                    if access_summary_authz_enabled(scoped_access_summary) and not access_summary_allows(
                        scoped_access_summary,
                        "collection",
                        str(getattr(record, "collection_id", "") or ""),
                        action="use",
                        implicit_resource_id=str(scoped_access_summary.get("session_upload_collection_id") or ""),
                    ):
                        continue
                    skill_ids.extend(str(item) for item in (getattr(record, "graph_skill_ids", []) or []) if str(item).strip())
            if skill_store is not None:
                try:
                    records = skill_store.list_skill_packs(
                        tenant_id=tenant_id,
                        owner_user_id=user_id,
                        graph_id=graph_id,
                        accessible_skill_family_ids=accessible_skill_family_ids,
                    )
                except TypeError:
                    try:
                        records = skill_store.list_skill_packs(
                            tenant_id=tenant_id,
                            owner_user_id=user_id,
                            graph_id=graph_id,
                        )
                    except Exception:
                        records = []
                except Exception:
                    records = []
                skill_ids.extend(
                    str(getattr(record, "skill_id", "") or "")
                    for record in records
                    if str(getattr(record, "skill_id", "") or "").strip()
                )
        return self._dedupe(skill_ids)

    def _restored_skill_ids(self, session_state: SessionState) -> List[str]:
        metadata = dict(session_state.metadata or {})
        restore = dict(metadata.get("context_restore_snapshot") or {})
        boundary = dict(metadata.get("context_compact_boundary") or {})
        if not restore:
            restore = dict(boundary.get("restore_snapshot") or {})
        ids: List[str] = []
        for item in restore.get("recent_skills") or []:
            if not isinstance(item, dict):
                continue
            skill_id = str(item.get("skill_id") or "").strip()
            if skill_id:
                ids.append(skill_id)
        return self._dedupe(ids)
