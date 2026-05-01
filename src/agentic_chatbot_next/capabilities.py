from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agentic_chatbot_next.authz import (
    access_summary_allowed_ids,
    access_summary_authz_enabled,
)
from agentic_chatbot_next.runtime.context import RuntimePaths

logger = logging.getLogger(__name__)

PERMISSION_MODES = {"default", "plan", "restricted", "bypass"}
FAST_PATH_POLICIES = {"inventory_plus_simple", "inventory_only", "minimal"}


def _clean_strings(values: Iterable[Any] | None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in list(values or []):
        clean = str(value or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def _normalise_permission_mode(value: Any) -> str:
    clean = str(value or "default").strip().lower()
    return clean if clean in PERMISSION_MODES else "default"


def _normalise_fast_path_policy(value: Any) -> str:
    clean = str(value or "inventory_plus_simple").strip().lower()
    return clean if clean in FAST_PATH_POLICIES else "inventory_plus_simple"


def _selector_matches(selector: str, value: str) -> bool:
    selector = str(selector or "").strip()
    value = str(value or "").strip()
    if not selector or not value:
        return False
    if selector == "*":
        return True
    if selector.endswith("*"):
        return value.startswith(selector[:-1])
    return selector == value


def _any_selector_matches(selectors: Sequence[str], value: str) -> bool:
    return any(_selector_matches(selector, value) for selector in selectors)


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


@dataclass
class CapabilityProfile:
    enabled_tools: list[str] = field(default_factory=list)
    disabled_tools: list[str] = field(default_factory=list)
    enabled_tool_groups: list[str] = field(default_factory=list)
    enabled_skill_pack_ids: list[str] = field(default_factory=list)
    disabled_skill_pack_ids: list[str] = field(default_factory=list)
    enabled_mcp_tool_ids: list[str] = field(default_factory=list)
    enabled_agents: list[str] = field(default_factory=list)
    enabled_collections: list[str] = field(default_factory=list)
    enabled_plugins: list[str] = field(default_factory=list)
    permission_mode: str = "default"
    fast_path_policy: str = "inventory_plus_simple"
    plugin_preferences: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "CapabilityProfile":
        payload = dict(raw or {})
        return cls(
            enabled_tools=_clean_strings(payload.get("enabled_tools")),
            disabled_tools=_clean_strings(payload.get("disabled_tools")),
            enabled_tool_groups=_clean_strings(payload.get("enabled_tool_groups")),
            enabled_skill_pack_ids=_clean_strings(payload.get("enabled_skill_pack_ids")),
            disabled_skill_pack_ids=_clean_strings(payload.get("disabled_skill_pack_ids")),
            enabled_mcp_tool_ids=_clean_strings(payload.get("enabled_mcp_tool_ids")),
            enabled_agents=_clean_strings(payload.get("enabled_agents")),
            enabled_collections=_clean_strings(payload.get("enabled_collections")),
            enabled_plugins=_clean_strings(payload.get("enabled_plugins")),
            permission_mode=_normalise_permission_mode(payload.get("permission_mode")),
            fast_path_policy=_normalise_fast_path_policy(payload.get("fast_path_policy")),
            plugin_preferences={
                str(key): bool(value)
                for key, value in dict(payload.get("plugin_preferences") or {}).items()
                if str(key).strip()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["permission_mode"] = _normalise_permission_mode(self.permission_mode)
        payload["fast_path_policy"] = _normalise_fast_path_policy(self.fast_path_policy)
        return payload


@dataclass
class EffectiveCapabilities(CapabilityProfile):
    hidden_unavailable: list[str] = field(default_factory=list)
    source: str = "defaults"
    collection_access_limited: bool = False
    agent_access_limited: bool = False
    skill_access_limited: bool = False

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "EffectiveCapabilities":
        payload = _coerce_mapping(raw)
        base = CapabilityProfile.from_dict(payload)
        return cls(
            **base.to_dict(),
            hidden_unavailable=_clean_strings(payload.get("hidden_unavailable")),
            source=str(payload.get("source") or "defaults"),
            collection_access_limited=bool(payload.get("collection_access_limited", False)),
            agent_access_limited=bool(payload.get("agent_access_limited", False)),
            skill_access_limited=bool(payload.get("skill_access_limited", False)),
        )

    @classmethod
    def from_profile(
        cls,
        profile: CapabilityProfile | Mapping[str, Any] | None,
        *,
        tenant_available_agents: Iterable[str] | None = None,
        tenant_available_collections: Iterable[str] | None = None,
        access_summary: Mapping[str, Any] | None = None,
        source: str = "profile",
    ) -> "EffectiveCapabilities":
        if isinstance(profile, CapabilityProfile):
            base = profile
        else:
            base = CapabilityProfile.from_dict(_coerce_mapping(profile))

        available_agents = set(_clean_strings(tenant_available_agents))
        enabled_agents = list(base.enabled_agents)
        hidden: list[str] = []
        if available_agents and enabled_agents:
            hidden = [agent for agent in enabled_agents if agent not in available_agents]
            enabled_agents = [agent for agent in enabled_agents if agent in available_agents]

        enabled_collections = list(base.enabled_collections)
        available_collections = set(_clean_strings(tenant_available_collections))
        if available_collections and enabled_collections:
            hidden.extend([collection for collection in enabled_collections if collection not in available_collections])
            enabled_collections = [collection for collection in enabled_collections if collection in available_collections]

        collection_access_limited = False
        agent_access_limited = False
        skill_access_limited = False
        if access_summary_authz_enabled(access_summary):
            agent_resource = dict(dict(dict(access_summary or {}).get("resources") or {}).get("agent") or {})
            agent_use_all = bool(agent_resource.get("use_all"))
            authz_agent_ids = list(access_summary_allowed_ids(access_summary, "agent", action="use"))
            authz_agent_groups = list(access_summary_allowed_ids(access_summary, "agent_group", action="use"))
            if not agent_use_all and (authz_agent_ids or authz_agent_groups):
                allowed_agents = set(authz_agent_ids)
                for group in authz_agent_groups:
                    clean_group = str(group or "").strip().lower()
                    if clean_group == "*":
                        agent_use_all = True
                        break
                    for agent in available_agents:
                        lowered = agent.lower()
                        if clean_group and (clean_group in lowered or (clean_group in {"coordinator", "planner"} and lowered in {"coordinator", "research_coordinator", "planner"})):
                            allowed_agents.add(agent)
                if not agent_use_all:
                    if enabled_agents:
                        hidden.extend([agent for agent in enabled_agents if agent not in allowed_agents])
                        enabled_agents = [agent for agent in enabled_agents if agent in allowed_agents]
                    else:
                        enabled_agents = sorted(allowed_agents)
                    agent_access_limited = True

            authz_collection_ids = list(access_summary_allowed_ids(access_summary, "collection", action="use"))
            collection_resource = dict(dict(dict(access_summary or {}).get("resources") or {}).get("collection") or {})
            collection_use_all = bool(collection_resource.get("use_all"))
            if authz_collection_ids and "*" not in authz_collection_ids:
                allowed_collections = set(authz_collection_ids)
                if enabled_collections:
                    hidden.extend([collection for collection in enabled_collections if collection not in allowed_collections])
                    enabled_collections = [collection for collection in enabled_collections if collection in allowed_collections]
                elif not collection_use_all:
                    enabled_collections = list(authz_collection_ids)
                    collection_access_limited = True
            elif not collection_use_all and "*" not in authz_collection_ids:
                collection_access_limited = True

            authz_tool_ids = list(access_summary_allowed_ids(access_summary, "tool", action="use"))
            if authz_tool_ids and "*" not in authz_tool_ids:
                enabled_tools = [tool for tool in base.enabled_tools if tool in set(authz_tool_ids)]
            else:
                enabled_tools = list(base.enabled_tools)

            authz_tool_groups = list(access_summary_allowed_ids(access_summary, "tool_group", action="use"))
            if authz_tool_groups and "*" not in authz_tool_groups:
                enabled_tool_groups = [group for group in base.enabled_tool_groups if group in set(authz_tool_groups)]
                if not enabled_tool_groups:
                    enabled_tool_groups = list(authz_tool_groups)
            else:
                enabled_tool_groups = list(base.enabled_tool_groups)

            authz_skill_ids = [
                *list(access_summary_allowed_ids(access_summary, "skill", action="use")),
                *list(access_summary_allowed_ids(access_summary, "skill_family", action="use")),
            ]
            skill_resource = dict(dict(dict(access_summary or {}).get("resources") or {}).get("skill") or {})
            skill_family_resource = dict(dict(dict(access_summary or {}).get("resources") or {}).get("skill_family") or {})
            skill_use_all = bool(skill_resource.get("use_all") or skill_family_resource.get("use_all"))
            if authz_skill_ids and "*" not in authz_skill_ids:
                allowed_skills = set(authz_skill_ids)
                if base.enabled_skill_pack_ids:
                    enabled_skill_pack_ids = [skill for skill in base.enabled_skill_pack_ids if skill in allowed_skills]
                else:
                    enabled_skill_pack_ids = list(authz_skill_ids)
                skill_access_limited = True
            else:
                enabled_skill_pack_ids = list(base.enabled_skill_pack_ids)
                skill_access_limited = not skill_use_all and bool(authz_skill_ids)
        else:
            enabled_tools = list(base.enabled_tools)
            enabled_tool_groups = list(base.enabled_tool_groups)
            enabled_skill_pack_ids = list(base.enabled_skill_pack_ids)

        return cls(
            enabled_tools=enabled_tools,
            disabled_tools=list(base.disabled_tools),
            enabled_tool_groups=enabled_tool_groups,
            enabled_skill_pack_ids=enabled_skill_pack_ids,
            disabled_skill_pack_ids=list(base.disabled_skill_pack_ids),
            enabled_mcp_tool_ids=list(base.enabled_mcp_tool_ids),
            enabled_agents=enabled_agents,
            enabled_collections=enabled_collections,
            enabled_plugins=list(base.enabled_plugins),
            permission_mode=_normalise_permission_mode(base.permission_mode),
            fast_path_policy=_normalise_fast_path_policy(base.fast_path_policy),
            plugin_preferences=dict(base.plugin_preferences),
            hidden_unavailable=_clean_strings(hidden),
            source=source,
            collection_access_limited=collection_access_limited,
            agent_access_limited=agent_access_limited,
            skill_access_limited=skill_access_limited,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "EffectiveCapabilities":
        base = CapabilityProfile.from_dict(raw)
        payload = dict(raw or {})
        return cls(
            **base.to_dict(),
            hidden_unavailable=_clean_strings(payload.get("hidden_unavailable")),
            source=str(payload.get("source") or "metadata"),
            collection_access_limited=bool(payload.get("collection_access_limited", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["hidden_unavailable"] = list(self.hidden_unavailable)
        payload["source"] = self.source
        payload["collection_access_limited"] = self.collection_access_limited
        payload["agent_access_limited"] = self.agent_access_limited
        payload["skill_access_limited"] = self.skill_access_limited
        return payload

    def allows_agent(self, agent_name: str) -> bool:
        clean = str(agent_name or "").strip()
        if self.agent_access_limited and not self.enabled_agents:
            return False
        return not self.enabled_agents or clean in set(self.enabled_agents)

    def allows_collection(self, collection_id: str) -> bool:
        clean = str(collection_id or "").strip()
        if self.enabled_collections:
            return clean in set(self.enabled_collections)
        return not self.collection_access_limited

    def allows_skill(self, skill_id: str, *, family_id: str = "") -> bool:
        clean_skill_id = str(skill_id or "").strip()
        clean_family_id = str(family_id or "").strip()
        candidates = [item for item in (clean_skill_id, clean_family_id) if item]
        if not candidates:
            return False
        if any(_any_selector_matches(self.disabled_skill_pack_ids, candidate) for candidate in candidates):
            return False
        if self.enabled_skill_pack_ids:
            return any(_any_selector_matches(self.enabled_skill_pack_ids, candidate) for candidate in candidates)
        if self.skill_access_limited:
            return False
        return True

    def allows_tool(
        self,
        tool_name: str,
        *,
        group: str = "",
        read_only: bool = False,
        destructive: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        clean_tool = str(tool_name or "").strip()
        clean_group = str(group or "").strip()
        if not clean_tool:
            return False
        if self.permission_mode == "restricted" and not read_only:
            return False
        if _any_selector_matches(self.disabled_tools, clean_tool):
            return False
        if self.enabled_tool_groups and clean_group not in set(self.enabled_tool_groups):
            if not _any_selector_matches(self.enabled_tools, clean_tool):
                return False
        if self.enabled_tools and not _any_selector_matches(self.enabled_tools, clean_tool):
            return False
        if clean_group == "mcp" and self.enabled_mcp_tool_ids:
            meta = dict(metadata or {})
            candidates = [
                clean_tool,
                str(meta.get("tool_id") or "").strip(),
                str(meta.get("mcp_tool_id") or "").strip(),
                str(meta.get("qualified_tool_id") or "").strip(),
            ]
            if not any(candidate and candidate in set(self.enabled_mcp_tool_ids) for candidate in candidates):
                return False
        return True


def coerce_effective_capabilities(value: Any) -> EffectiveCapabilities | None:
    if isinstance(value, EffectiveCapabilities):
        return value
    payload = _coerce_mapping(value)
    if not payload:
        return None
    return EffectiveCapabilities.from_dict(payload)


class CapabilityProfileStore:
    def get_profile(self, tenant_id: str, user_id: str) -> CapabilityProfile:
        raise NotImplementedError

    def save_profile(self, tenant_id: str, user_id: str, profile: CapabilityProfile) -> CapabilityProfile:
        raise NotImplementedError


class FileCapabilityProfileStore(CapabilityProfileStore):
    def __init__(self, paths: RuntimePaths):
        self.paths = paths

    def _path(self, tenant_id: str, user_id: str) -> Path:
        return self.paths.user_profile_dir(tenant_id, user_id) / "capability_profile.json"

    def get_profile(self, tenant_id: str, user_id: str) -> CapabilityProfile:
        path = self._path(tenant_id, user_id)
        if not path.exists():
            return CapabilityProfile()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read capability profile %s: %s", path, exc)
            return CapabilityProfile()
        return CapabilityProfile.from_dict(raw)

    def save_profile(self, tenant_id: str, user_id: str, profile: CapabilityProfile) -> CapabilityProfile:
        path = self._path(tenant_id, user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return profile


def _load_profile_from_store(
    stores: Any,
    *,
    tenant_id: str,
    user_id: str,
    paths: RuntimePaths,
) -> tuple[CapabilityProfile, str]:
    store = getattr(stores, "capability_profile_store", None)
    if store is not None and hasattr(store, "get_profile"):
        try:
            return store.get_profile(tenant_id, user_id), "postgres"
        except Exception as exc:
            logger.warning("Could not load Postgres capability profile; using file fallback: %s", exc)
    try:
        return FileCapabilityProfileStore(paths).get_profile(tenant_id, user_id), "file"
    except Exception as exc:
        logger.warning("Could not load file capability profile; using defaults: %s", exc)
        return CapabilityProfile(), "defaults"


def save_capability_profile(
    *,
    settings: Any,
    stores: Any,
    tenant_id: str,
    user_id: str,
    profile: CapabilityProfile,
) -> CapabilityProfile:
    paths = RuntimePaths.from_settings(settings)
    store = getattr(stores, "capability_profile_store", None)
    if store is not None and hasattr(store, "save_profile"):
        try:
            return store.save_profile(tenant_id, user_id, profile)
        except Exception as exc:
            logger.warning("Could not save Postgres capability profile; using file fallback: %s", exc)
    return FileCapabilityProfileStore(paths).save_profile(tenant_id, user_id, profile)


def resolve_effective_capabilities(
    *,
    settings: Any,
    stores: Any,
    session: Any,
    registry: Any | None = None,
    profile: CapabilityProfile | Mapping[str, Any] | None = None,
    access_summary: Mapping[str, Any] | None = None,
) -> EffectiveCapabilities:
    metadata = dict(getattr(session, "metadata", {}) or {})
    if profile is None:
        profile = metadata.get("capability_profile")
    if profile is None and metadata.get("effective_capabilities"):
        profile = metadata.get("effective_capabilities")

    tenant_id = str(getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev")) or "local-dev")
    user_id = str(getattr(session, "user_id", getattr(settings, "default_user_id", "default")) or "default")
    paths = RuntimePaths.from_settings(settings)
    source = "metadata"
    if profile is None:
        profile, source = _load_profile_from_store(stores, tenant_id=tenant_id, user_id=user_id, paths=paths)
    elif not isinstance(profile, CapabilityProfile):
        profile = CapabilityProfile.from_dict(_coerce_mapping(profile))

    available_agents: list[str] = []
    if registry is not None and (hasattr(registry, "list") or hasattr(registry, "list_routable")):
        try:
            agent_iterable = registry.list() if hasattr(registry, "list") else registry.list_routable()
            available_agents = [
                str(getattr(agent, "name", "") or "").strip()
                for agent in agent_iterable
                if str(getattr(agent, "name", "") or "").strip()
            ]
        except Exception:
            available_agents = []

    available_collections: list[str] = []
    collection_store = getattr(stores, "collection_store", None)
    if collection_store is not None and hasattr(collection_store, "list_collections"):
        try:
            available_collections = [
                str(getattr(item, "collection_id", "") or "").strip()
                for item in collection_store.list_collections(tenant_id=tenant_id)
                if str(getattr(item, "collection_id", "") or "").strip()
            ]
        except Exception:
            available_collections = []

    effective = EffectiveCapabilities.from_profile(
        profile,
        tenant_available_agents=available_agents,
        tenant_available_collections=available_collections,
        access_summary=access_summary or metadata.get("access_summary") or getattr(session, "access_summary", {}) or {},
        source=source,
    )
    return effective


def build_capability_catalog(
    *,
    settings: Any,
    stores: Any,
    session: Any,
    registry: Any | None = None,
    tool_definitions: Mapping[str, Any] | None = None,
    effective: EffectiveCapabilities | None = None,
) -> dict[str, Any]:
    effective = effective or resolve_effective_capabilities(
        settings=settings,
        stores=stores,
        session=session,
        registry=registry,
    )
    agents: list[dict[str, Any]] = []
    if registry is not None and hasattr(registry, "list_routable"):
        try:
            for agent in registry.list_routable():
                name = str(getattr(agent, "name", "") or "").strip()
                if not name or not effective.allows_agent(name):
                    continue
                agents.append(
                    {
                        "name": name,
                        "mode": str(getattr(agent, "mode", "") or ""),
                        "enabled": True,
                        "allow_background_jobs": bool(getattr(agent, "allow_background_jobs", False)),
                    }
                )
        except Exception:
            agents = []

    tools: list[dict[str, Any]] = []
    for name, definition in dict(tool_definitions or {}).items():
        tool_name = str(getattr(definition, "name", "") or name)
        if not effective.allows_tool(
            tool_name,
            group=str(getattr(definition, "group", "") or ""),
            read_only=bool(getattr(definition, "read_only", False)),
            destructive=bool(getattr(definition, "destructive", False)),
            metadata=dict(getattr(definition, "metadata", {}) or {}),
        ):
            continue
        tools.append(
            {
                "name": tool_name,
                "group": str(getattr(definition, "group", "") or ""),
                "read_only": bool(getattr(definition, "read_only", False)),
                "destructive": bool(getattr(definition, "destructive", False)),
                "deferred": bool(getattr(definition, "should_defer", False)),
                "description": str(getattr(definition, "description", "") or ""),
            }
        )

    tenant_id = str(getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev")) or "local-dev")
    collections: list[dict[str, Any]] = []
    collection_store = getattr(stores, "collection_store", None)
    if collection_store is not None and hasattr(collection_store, "list_collections"):
        try:
            for collection in collection_store.list_collections(tenant_id=tenant_id):
                collection_id = str(getattr(collection, "collection_id", "") or "").strip()
                if collection_id and effective.allows_collection(collection_id):
                    collections.append(
                        {
                            "collection_id": collection_id,
                            "maintenance_policy": str(getattr(collection, "maintenance_policy", "") or ""),
                            "enabled": True,
                        }
                    )
        except Exception:
            collections = []

    skill_packs: list[dict[str, Any]] = []
    skill_store = getattr(stores, "skill_store", None)
    if skill_store is not None and hasattr(skill_store, "list_skill_packs"):
        try:
            for record in skill_store.list_skill_packs(
                tenant_id=tenant_id,
                enabled_only=True,
                owner_user_id=str(getattr(session, "user_id", "") or ""),
                status="active",
            ):
                family_id = str(getattr(record, "version_parent", "") or getattr(record, "skill_id", "") or "")
                skill_id = str(getattr(record, "skill_id", "") or "")
                if skill_id and effective.allows_skill(skill_id, family_id=family_id):
                    skill_packs.append(
                        {
                            "skill_id": skill_id,
                            "family_id": family_id,
                            "name": str(getattr(record, "name", "") or skill_id),
                            "agent_scope": str(getattr(record, "agent_scope", "") or ""),
                            "kind": str(getattr(record, "kind", "") or "retrievable"),
                            "enabled": True,
                        }
                    )
        except Exception:
            skill_packs = []

    return {
        "object": "capabilities.catalog",
        "effective_capabilities": effective.to_dict(),
        "agents": agents,
        "tools": tools,
        "skill_packs": skill_packs,
        "collections": collections,
        "permission_modes": sorted(PERMISSION_MODES),
        "fast_path_policies": sorted(FAST_PATH_POLICIES),
    }
