from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

from agentic_chatbot_next.contracts.messages import utc_now_iso

RESOURCE_TYPES = (
    "agent",
    "agent_group",
    "collection",
    "graph",
    "skill",
    "skill_family",
    "tool",
    "tool_group",
    "worker_request",
)
RESOURCE_ACTIONS = ("use", "manage", "approve", "delete")


def normalize_user_email(value: Any) -> str:
    return str(value or "").strip().casefold()


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    items: list[str] = []
    for value in values:
        clean = str(value or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        items.append(clean)
    return tuple(items)


def access_summary_authz_enabled(summary: Dict[str, Any] | None) -> bool:
    return bool(dict(summary or {}).get("authz_enabled"))


def access_summary_resource(
    summary: Dict[str, Any] | None,
    resource_type: str,
) -> Dict[str, Any]:
    resources = dict(dict(summary or {}).get("resources") or {})
    payload = dict(resources.get(str(resource_type or "").strip()) or {})
    return {
        **{
            action: [str(item) for item in list(payload.get(action) or []) if str(item).strip()]
            for action in RESOURCE_ACTIONS
        },
        **{f"{action}_all": bool(payload.get(f"{action}_all", False)) for action in RESOURCE_ACTIONS},
    }


def access_summary_allowed_ids(
    summary: Dict[str, Any] | None,
    resource_type: str,
    *,
    action: str = "use",
) -> tuple[str, ...]:
    resource = access_summary_resource(summary, resource_type)
    return _dedupe(resource.get(action) or [])


def access_summary_allows(
    summary: Dict[str, Any] | None,
    resource_type: str,
    resource_id: str,
    *,
    action: str = "use",
    implicit_resource_id: str = "",
) -> bool:
    payload = access_summary_resource(summary, resource_type)
    if bool(payload.get(f"{action}_all", False)):
        return True
    normalized_resource_id = str(resource_id or "").strip()
    if normalized_resource_id and normalized_resource_id in set(payload.get(action) or []):
        return True
    normalized_implicit = str(implicit_resource_id or "").strip()
    if normalized_resource_id and normalized_implicit and normalized_resource_id == normalized_implicit:
        return True
    return False


@dataclass(frozen=True)
class AccessSnapshot:
    tenant_id: str
    user_id: str
    user_email: str = ""
    auth_provider: str = ""
    principal_id: str = ""
    principal_ids: tuple[str, ...] = ()
    role_ids: tuple[str, ...] = ()
    session_upload_collection_id: str = ""
    authz_enabled: bool = False
    resources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    resolved_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def disabled(
        cls,
        *,
        tenant_id: str,
        user_id: str,
        user_email: str = "",
        session_upload_collection_id: str = "",
    ) -> "AccessSnapshot":
        return cls(
            tenant_id=tenant_id,
            user_id=user_id,
            user_email=normalize_user_email(user_email),
            auth_provider="email" if normalize_user_email(user_email) else "",
            session_upload_collection_id=str(session_upload_collection_id or "").strip(),
            authz_enabled=False,
            resources={},
        )

    def to_summary(self) -> Dict[str, Any]:
        payload_resources: Dict[str, Dict[str, Any]] = {}
        for resource_type in RESOURCE_TYPES:
            resource = dict(self.resources.get(resource_type) or {})
            payload_resources[resource_type] = {
                **{
                    action: [str(item) for item in list(resource.get(action) or []) if str(item).strip()]
                    for action in RESOURCE_ACTIONS
                },
                **{f"{action}_all": bool(resource.get(f"{action}_all", False)) for action in RESOURCE_ACTIONS},
            }
        return {
            "authz_enabled": bool(self.authz_enabled),
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "user_email": self.user_email,
            "auth_provider": self.auth_provider,
            "principal_id": self.principal_id,
            "principal_ids": list(self.principal_ids),
            "role_ids": list(self.role_ids),
            "session_upload_collection_id": self.session_upload_collection_id,
            "resolved_at": self.resolved_at,
            "resources": payload_resources,
        }


class AuthorizationService:
    """Resolve effective RBAC access for a user request from a trusted email identity."""

    def __init__(self, settings: Any, store: Any | None) -> None:
        self.settings = settings
        self.store = store

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.settings, "authz_enabled", False)) and self.store is not None

    def resolve_access_snapshot(
        self,
        *,
        tenant_id: str,
        user_id: str,
        user_email: str,
        session_upload_collection_id: str = "",
        display_name: str = "",
    ) -> AccessSnapshot:
        normalized_email = normalize_user_email(user_email)
        if not self.enabled:
            return AccessSnapshot.disabled(
                tenant_id=tenant_id,
                user_id=user_id,
                user_email=normalized_email,
                session_upload_collection_id=session_upload_collection_id,
            )

        resources: Dict[str, Dict[str, Any]] = {
            resource_type: {
                **{action: [] for action in RESOURCE_ACTIONS},
                **{f"{action}_all": False for action in RESOURCE_ACTIONS},
            }
            for resource_type in RESOURCE_TYPES
        }
        if not normalized_email:
            return AccessSnapshot(
                tenant_id=tenant_id,
                user_id=user_id,
                authz_enabled=True,
                session_upload_collection_id=str(session_upload_collection_id or "").strip(),
                resources=resources,
            )

        principal = self.store.ensure_email_principal(
            tenant_id=tenant_id,
            email_normalized=normalized_email,
            display_name=display_name or user_id or normalized_email,
        )
        principal_ids = self.store.resolve_effective_principal_ids(
            tenant_id=tenant_id,
            principal_id=str(getattr(principal, "principal_id", "") or ""),
        )
        bindings = self.store.list_role_bindings(
            tenant_id=tenant_id,
            principal_ids=list(principal_ids),
            include_disabled=False,
        )
        role_ids = _dedupe(str(getattr(binding, "role_id", "") or "") for binding in bindings)
        permissions = self.store.list_role_permissions(
            tenant_id=tenant_id,
            role_ids=list(role_ids),
        )
        for permission in permissions:
            resource_type = str(getattr(permission, "resource_type", "") or "").strip().lower()
            action = str(getattr(permission, "action", "") or "").strip().lower()
            selector = str(getattr(permission, "resource_selector", "") or "").strip()
            if resource_type not in resources or action not in RESOURCE_ACTIONS:
                continue
            if selector == "*":
                resources[resource_type][f"{action}_all"] = True
                continue
            resources[resource_type][action] = list(
                _dedupe([*list(resources[resource_type].get(action) or []), selector])
            )

        return AccessSnapshot(
            tenant_id=tenant_id,
            user_id=user_id,
            user_email=normalized_email,
            auth_provider="email",
            principal_id=str(getattr(principal, "principal_id", "") or ""),
            principal_ids=tuple(principal_ids),
            role_ids=tuple(role_ids),
            session_upload_collection_id=str(session_upload_collection_id or "").strip(),
            authz_enabled=True,
            resources=resources,
        )

    def apply_access_snapshot(
        self,
        session_or_state: Any,
        *,
        tenant_id: str,
        user_id: str,
        user_email: str,
        session_upload_collection_id: str = "",
        display_name: str = "",
    ) -> AccessSnapshot:
        snapshot = self.resolve_access_snapshot(
            tenant_id=tenant_id,
            user_id=user_id,
            user_email=user_email,
            session_upload_collection_id=session_upload_collection_id,
            display_name=display_name,
        )
        metadata = dict(getattr(session_or_state, "metadata", {}) or {})
        metadata["user_email"] = snapshot.user_email
        metadata["auth_provider"] = snapshot.auth_provider
        metadata["principal_id"] = snapshot.principal_id
        metadata["role_ids"] = list(snapshot.role_ids)
        metadata["access_summary"] = snapshot.to_summary()
        setattr(session_or_state, "metadata", metadata)
        if hasattr(session_or_state, "user_email"):
            session_or_state.user_email = snapshot.user_email
        if hasattr(session_or_state, "auth_provider"):
            session_or_state.auth_provider = snapshot.auth_provider
        if hasattr(session_or_state, "principal_id"):
            session_or_state.principal_id = snapshot.principal_id
        if hasattr(session_or_state, "access_summary"):
            session_or_state.access_summary = snapshot.to_summary()
        return snapshot

    def preview_effective_access(
        self,
        *,
        tenant_id: str,
        email: str,
    ) -> Dict[str, Any]:
        return self.resolve_access_snapshot(
            tenant_id=tenant_id,
            user_id="preview",
            user_email=email,
            session_upload_collection_id="",
            display_name=email,
        ).to_summary()
