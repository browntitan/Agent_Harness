from __future__ import annotations

from typing import Any, Sequence

from agentic_chatbot_next.authz import access_summary_allowed_ids, access_summary_allows, access_summary_authz_enabled
from agentic_chatbot_next.rag.hints import (
    RagExecutionHints,
    apply_bounded_synthesis_override,
    coerce_rag_execution_hints,
    infer_rag_execution_hints,
    merge_controller_hints,
    merge_rag_execution_hints,
)
from agentic_chatbot_next.skills.query_builder import build_skill_resolver_query
from agentic_chatbot_next.skills.resolver import SkillResolver


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in values:
        clean = str(item or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result


def _graph_skill_ids(stores: Any, *, tenant_id: str, user_id: str, graph_ids: Sequence[str]) -> list[str]:
    return _graph_skill_ids_with_access(
        stores,
        tenant_id=tenant_id,
        user_id=user_id,
        graph_ids=graph_ids,
        access_summary={},
    )


def _graph_skill_ids_with_access(
    stores: Any,
    *,
    tenant_id: str,
    user_id: str,
    graph_ids: Sequence[str],
    access_summary: dict[str, Any],
) -> list[str]:
    graph_store = getattr(stores, "graph_index_store", None)
    skill_store = getattr(stores, "skill_store", None)
    if graph_store is None:
        graph_store = None
    skill_ids: list[str] = []
    accessible_skill_family_ids = (
        list(access_summary_allowed_ids(access_summary, "skill_family", action="use"))
        if access_summary_authz_enabled(access_summary)
        else None
    )
    for graph_id in graph_ids:
        if access_summary_authz_enabled(access_summary) and not access_summary_allows(
            access_summary,
            "graph",
            str(graph_id),
            action="use",
        ):
            continue
        if graph_store is not None:
            try:
                record = graph_store.get_index(str(graph_id), tenant_id, user_id="*")
            except TypeError:
                record = graph_store.get_index(str(graph_id), tenant_id)
            except Exception:
                record = None
            if record is not None:
                if access_summary_authz_enabled(access_summary) and not access_summary_allows(
                    access_summary,
                    "collection",
                    str(getattr(record, "collection_id", "") or ""),
                    action="use",
                    implicit_resource_id=str(access_summary.get("session_upload_collection_id") or ""),
                ):
                    continue
                skill_ids.extend(str(item) for item in (getattr(record, "graph_skill_ids", []) or []) if str(item).strip())
        if skill_store is not None:
            try:
                records = skill_store.list_skill_packs(
                    tenant_id=tenant_id,
                    owner_user_id=user_id,
                    graph_id=str(graph_id),
                    accessible_skill_family_ids=accessible_skill_family_ids,
                )
            except TypeError:
                try:
                    records = skill_store.list_skill_packs(
                        tenant_id=tenant_id,
                        owner_user_id=user_id,
                        graph_id=str(graph_id),
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
    return _dedupe(skill_ids)


def resolve_rag_execution_hints(
    settings: Any,
    stores: Any,
    *,
    session: Any,
    query: str,
    pinned_skill_ids: Sequence[str] | None = None,
    skill_queries: Sequence[str] | None = None,
    research_profile: str = "",
    coverage_goal: str = "",
    result_mode: str = "",
    controller_hints: dict[str, Any] | None = None,
) -> RagExecutionHints:
    heuristic = infer_rag_execution_hints(query, skill_queries=skill_queries)
    explicit = coerce_rag_execution_hints(
        research_profile=research_profile,
        coverage_goal=coverage_goal,
        result_mode=result_mode,
        controller_hints=controller_hints,
    )
    has_explicit_override = bool(
        str(research_profile or "").strip()
        or str(coverage_goal or "").strip()
        or str(result_mode or "").strip()
        or dict(controller_hints or {})
    )

    if settings is None or stores is None or session is None or not hasattr(stores, "skill_store"):
        merged = merge_rag_execution_hints(explicit, heuristic)
        if has_explicit_override:
            return merged
        return apply_bounded_synthesis_override(merged, query=query, skill_queries=skill_queries)

    skill_query = build_skill_resolver_query(
        settings,
        base_query=str(query or ""),
        skill_queries=[str(item) for item in (skill_queries or []) if str(item).strip()],
    )
    if not skill_query.strip():
        merged = merge_rag_execution_hints(explicit, heuristic)
        if has_explicit_override:
            return merged
        return apply_bounded_synthesis_override(merged, query=query, skill_queries=skill_queries)

    try:
        access_summary = dict((getattr(session, "metadata", {}) or {}).get("access_summary") or {})
        active_graph_ids = _dedupe(
            [
                *[str(item) for item in (getattr(session, "metadata", {}) or {}).get("active_graph_ids", []) if str(item).strip()],
                *[str(item) for item in (controller_hints or {}).get("graph_ids", []) if str(item).strip()],
                *[str(item) for item in (controller_hints or {}).get("planned_graph_ids", []) if str(item).strip()],
            ]
        )
        effective_pinned_skill_ids = _dedupe(
            [
                *[str(item) for item in (pinned_skill_ids or []) if str(item).strip()],
                *_graph_skill_ids_with_access(
                    stores,
                    tenant_id=getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev")),
                    user_id=getattr(session, "user_id", getattr(settings, "default_user_id", "")),
                    graph_ids=active_graph_ids,
                    access_summary=access_summary,
                ),
            ]
        )
        resolved = SkillResolver(settings, stores).resolve(
            query=skill_query,
            tenant_id=getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev")),
            owner_user_id=getattr(session, "user_id", getattr(settings, "default_user_id", "")),
            agent_scope="rag",
            top_k=4,
            graph_ids=active_graph_ids,
            pinned_skill_ids=effective_pinned_skill_ids,
            accessible_skill_family_ids=(
                list(access_summary_allowed_ids(access_summary, "skill_family", action="use"))
                if access_summary_authz_enabled(access_summary)
                else None
            ),
        )
    except Exception:
        merged = merge_rag_execution_hints(explicit, heuristic)
        if has_explicit_override:
            return merged
        return apply_bounded_synthesis_override(merged, query=query, skill_queries=skill_queries)

    skill_hints = RagExecutionHints(
        matched_skill_ids=[match.skill_id for match in resolved.matches],
        matched_skill_names=[match.name for match in resolved.matches],
    )
    for match in resolved.matches:
        if not skill_hints.research_profile and match.retrieval_profile:
            skill_hints.research_profile = str(match.retrieval_profile)
        if not skill_hints.coverage_goal and match.coverage_goal:
            skill_hints.coverage_goal = str(match.coverage_goal)
        if not skill_hints.result_mode and match.result_mode:
            skill_hints.result_mode = str(match.result_mode)
        skill_hints.controller_hints = merge_controller_hints(
            skill_hints.controller_hints,
            dict(match.controller_hints or {}),
        )

    merged = merge_rag_execution_hints(explicit, skill_hints, heuristic)
    if has_explicit_override:
        return merged
    return apply_bounded_synthesis_override(merged, query=query, skill_queries=skill_queries)


__all__ = ["resolve_rag_execution_hints"]
