from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import psycopg2.extras

from agentic_chatbot_next.persistence.postgres.connection import get_conn

VALID_PRINCIPAL_TYPES = {"user", "group"}
VALID_PROVIDERS = {"email", "entra", "system"}
VALID_RESOURCE_TYPES = {"collection", "graph", "tool", "skill_family"}
VALID_ACTIONS = {"use", "manage"}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def _normalize_principal_type(value: str) -> str:
    normalized = str(value or "user").strip().lower()
    return normalized if normalized in VALID_PRINCIPAL_TYPES else "user"


def _normalize_provider(value: str) -> str:
    normalized = str(value or "email").strip().lower()
    return normalized if normalized in VALID_PROVIDERS else "email"


def _normalize_action(value: str) -> str:
    normalized = str(value or "use").strip().lower()
    return normalized if normalized in VALID_ACTIONS else "use"


def _normalize_resource_type(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in VALID_RESOURCE_TYPES:
        raise ValueError(f"Unsupported resource_type={value!r}")
    return normalized


def _normalize_selector(value: str) -> str:
    return str(value or "").strip() or "*"


@dataclass
class AuthPrincipalRecord:
    principal_id: str
    tenant_id: str = "local-dev"
    principal_type: str = "user"
    provider: str = "email"
    external_id: str = ""
    email_normalized: str = ""
    display_name: str = ""
    metadata_json: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    created_at: str = ""
    updated_at: str = ""


@dataclass
class AuthPrincipalMembershipRecord:
    membership_id: str
    tenant_id: str = "local-dev"
    parent_principal_id: str = ""
    child_principal_id: str = ""
    created_at: str = ""


@dataclass
class AuthRoleRecord:
    role_id: str
    tenant_id: str = "local-dev"
    name: str = ""
    description: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass
class AuthRoleBindingRecord:
    binding_id: str
    tenant_id: str = "local-dev"
    role_id: str = ""
    principal_id: str = ""
    created_at: str = ""
    disabled_at: str = ""


@dataclass
class AuthRolePermissionRecord:
    permission_id: str
    tenant_id: str = "local-dev"
    role_id: str = ""
    resource_type: str = ""
    action: str = "use"
    resource_selector: str = "*"
    created_at: str = ""


def _row_to_principal(row: Dict[str, Any]) -> AuthPrincipalRecord:
    return AuthPrincipalRecord(
        principal_id=str(row.get("principal_id") or ""),
        tenant_id=str(row.get("tenant_id") or "local-dev"),
        principal_type=_normalize_principal_type(str(row.get("principal_type") or "user")),
        provider=_normalize_provider(str(row.get("provider") or "email")),
        external_id=str(row.get("external_id") or ""),
        email_normalized=str(row.get("email_normalized") or ""),
        display_name=str(row.get("display_name") or ""),
        metadata_json=dict(row.get("metadata_json") or {}),
        active=bool(row.get("active", True)),
        created_at=str(row.get("created_at") or ""),
        updated_at=str(row.get("updated_at") or ""),
    )


def _row_to_membership(row: Dict[str, Any]) -> AuthPrincipalMembershipRecord:
    return AuthPrincipalMembershipRecord(
        membership_id=str(row.get("membership_id") or ""),
        tenant_id=str(row.get("tenant_id") or "local-dev"),
        parent_principal_id=str(row.get("parent_principal_id") or ""),
        child_principal_id=str(row.get("child_principal_id") or ""),
        created_at=str(row.get("created_at") or ""),
    )


def _row_to_role(row: Dict[str, Any]) -> AuthRoleRecord:
    return AuthRoleRecord(
        role_id=str(row.get("role_id") or ""),
        tenant_id=str(row.get("tenant_id") or "local-dev"),
        name=str(row.get("name") or ""),
        description=str(row.get("description") or ""),
        created_at=str(row.get("created_at") or ""),
        updated_at=str(row.get("updated_at") or ""),
    )


def _row_to_binding(row: Dict[str, Any]) -> AuthRoleBindingRecord:
    return AuthRoleBindingRecord(
        binding_id=str(row.get("binding_id") or ""),
        tenant_id=str(row.get("tenant_id") or "local-dev"),
        role_id=str(row.get("role_id") or ""),
        principal_id=str(row.get("principal_id") or ""),
        created_at=str(row.get("created_at") or ""),
        disabled_at=str(row.get("disabled_at") or ""),
    )


def _row_to_permission(row: Dict[str, Any]) -> AuthRolePermissionRecord:
    return AuthRolePermissionRecord(
        permission_id=str(row.get("permission_id") or ""),
        tenant_id=str(row.get("tenant_id") or "local-dev"),
        role_id=str(row.get("role_id") or ""),
        resource_type=str(row.get("resource_type") or ""),
        action=_normalize_action(str(row.get("action") or "use")),
        resource_selector=str(row.get("resource_selector") or "*"),
        created_at=str(row.get("created_at") or ""),
    )


class AccessControlStore:
    def ensure_email_principal(
        self,
        *,
        tenant_id: str,
        email_normalized: str,
        display_name: str = "",
    ) -> AuthPrincipalRecord:
        email = str(email_normalized or "").strip().casefold()
        if not email:
            raise ValueError("email_normalized is required")
        now = _utc_now()
        principal_id = f"user_email_{uuid.uuid5(uuid.NAMESPACE_DNS, f'{tenant_id}:{email}').hex[:24]}"
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO auth_principals
                        (principal_id, tenant_id, principal_type, provider, email_normalized, display_name, metadata_json, active, created_at, updated_at)
                    VALUES (%s, %s, 'user', 'email', %s, %s, %s, TRUE, %s, %s)
                    ON CONFLICT (tenant_id, provider, email_normalized) DO UPDATE SET
                        display_name = CASE
                            WHEN NULLIF(EXCLUDED.display_name, '') IS NOT NULL THEN EXCLUDED.display_name
                            ELSE auth_principals.display_name
                        END,
                        active = TRUE,
                        updated_at = EXCLUDED.updated_at
                    RETURNING *
                    """,
                    (
                        principal_id,
                        tenant_id,
                        email,
                        str(display_name or email),
                        psycopg2.extras.Json({}),
                        now,
                        now,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return _row_to_principal(dict(row or {}))

    def upsert_principal(self, record: AuthPrincipalRecord) -> AuthPrincipalRecord:
        now = _utc_now()
        principal_id = str(record.principal_id or _new_id("principal"))
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO auth_principals
                        (principal_id, tenant_id, principal_type, provider, external_id, email_normalized,
                         display_name, metadata_json, active, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (principal_id) DO UPDATE SET
                        principal_type = EXCLUDED.principal_type,
                        provider = EXCLUDED.provider,
                        external_id = EXCLUDED.external_id,
                        email_normalized = EXCLUDED.email_normalized,
                        display_name = EXCLUDED.display_name,
                        metadata_json = EXCLUDED.metadata_json,
                        active = EXCLUDED.active,
                        updated_at = EXCLUDED.updated_at
                    RETURNING *
                    """,
                    (
                        principal_id,
                        record.tenant_id,
                        _normalize_principal_type(record.principal_type),
                        _normalize_provider(record.provider),
                        str(record.external_id or ""),
                        str(record.email_normalized or "").strip().casefold(),
                        str(record.display_name or ""),
                        psycopg2.extras.Json(dict(record.metadata_json or {})),
                        bool(record.active),
                        record.created_at or now,
                        now,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return _row_to_principal(dict(row or {}))

    def list_principals(
        self,
        *,
        tenant_id: str = "local-dev",
        principal_type: str = "",
        query: str = "",
        limit: int = 200,
    ) -> List[AuthPrincipalRecord]:
        sql = ["SELECT * FROM auth_principals WHERE tenant_id = %s"]
        params: List[Any] = [tenant_id]
        if principal_type:
            sql.append("AND principal_type = %s")
            params.append(_normalize_principal_type(principal_type))
        if query:
            sql.append(
                "AND (email_normalized ILIKE %s OR display_name ILIKE %s OR principal_id ILIKE %s OR external_id ILIKE %s)"
            )
            needle = f"%{str(query).strip()}%"
            params.extend([needle, needle, needle, needle])
        sql.append("ORDER BY principal_type ASC, COALESCE(email_normalized, display_name, principal_id) ASC LIMIT %s")
        params.append(max(1, int(limit)))
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(" ".join(sql), params)
                rows = cur.fetchall()
        return [_row_to_principal(dict(row)) for row in rows]

    def get_principal(self, principal_id: str, *, tenant_id: str = "local-dev") -> Optional[AuthPrincipalRecord]:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM auth_principals WHERE tenant_id = %s AND principal_id = %s LIMIT 1",
                    (tenant_id, principal_id),
                )
                row = cur.fetchone()
        return _row_to_principal(dict(row)) if row else None

    def upsert_membership(self, record: AuthPrincipalMembershipRecord) -> AuthPrincipalMembershipRecord:
        membership_id = str(record.membership_id or _new_id("membership"))
        now = _utc_now()
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO auth_principal_memberships
                        (membership_id, tenant_id, parent_principal_id, child_principal_id, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (tenant_id, parent_principal_id, child_principal_id) DO UPDATE SET
                        created_at = auth_principal_memberships.created_at
                    RETURNING *
                    """,
                    (
                        membership_id,
                        record.tenant_id,
                        str(record.parent_principal_id or ""),
                        str(record.child_principal_id or ""),
                        record.created_at or now,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return _row_to_membership(dict(row or {}))

    def list_memberships(
        self,
        *,
        tenant_id: str = "local-dev",
        parent_principal_id: str = "",
        child_principal_id: str = "",
    ) -> List[AuthPrincipalMembershipRecord]:
        sql = ["SELECT * FROM auth_principal_memberships WHERE tenant_id = %s"]
        params: List[Any] = [tenant_id]
        if parent_principal_id:
            sql.append("AND parent_principal_id = %s")
            params.append(str(parent_principal_id))
        if child_principal_id:
            sql.append("AND child_principal_id = %s")
            params.append(str(child_principal_id))
        sql.append("ORDER BY created_at ASC")
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(" ".join(sql), params)
                rows = cur.fetchall()
        return [_row_to_membership(dict(row)) for row in rows]

    def delete_membership(self, membership_id: str, *, tenant_id: str = "local-dev") -> bool:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM auth_principal_memberships WHERE tenant_id = %s AND membership_id = %s",
                    (tenant_id, membership_id),
                )
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted

    def resolve_effective_principal_ids(self, *, tenant_id: str, principal_id: str) -> tuple[str, ...]:
        if not str(principal_id or "").strip():
            return ()
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH RECURSIVE principal_tree AS (
                        SELECT parent_principal_id
                        FROM auth_principal_memberships
                        WHERE tenant_id = %s
                          AND child_principal_id = %s
                        UNION
                        SELECT membership.parent_principal_id
                        FROM auth_principal_memberships AS membership
                        JOIN principal_tree AS tree
                          ON tree.parent_principal_id = membership.child_principal_id
                        WHERE membership.tenant_id = %s
                    )
                    SELECT principal_id
                    FROM (
                        SELECT %s AS principal_id
                        UNION
                        SELECT parent_principal_id AS principal_id FROM principal_tree
                    ) AS resolved
                    """,
                    (tenant_id, principal_id, tenant_id, principal_id),
                )
                rows = cur.fetchall()
        return tuple(
            str(row[0] or "").strip()
            for row in rows
            if row and str(row[0] or "").strip()
        )

    def upsert_role(self, record: AuthRoleRecord) -> AuthRoleRecord:
        role_id = str(record.role_id or _new_id("role"))
        now = _utc_now()
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO auth_roles (role_id, tenant_id, name, description, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (role_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        updated_at = EXCLUDED.updated_at
                    RETURNING *
                    """,
                    (
                        role_id,
                        record.tenant_id,
                        str(record.name or ""),
                        str(record.description or ""),
                        record.created_at or now,
                        now,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return _row_to_role(dict(row or {}))

    def list_roles(self, *, tenant_id: str = "local-dev") -> List[AuthRoleRecord]:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM auth_roles WHERE tenant_id = %s ORDER BY name ASC, role_id ASC",
                    (tenant_id,),
                )
                rows = cur.fetchall()
        return [_row_to_role(dict(row)) for row in rows]

    def get_role(self, role_id: str, *, tenant_id: str = "local-dev") -> Optional[AuthRoleRecord]:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM auth_roles WHERE tenant_id = %s AND role_id = %s LIMIT 1",
                    (tenant_id, role_id),
                )
                row = cur.fetchone()
        return _row_to_role(dict(row)) if row else None

    def delete_role(self, role_id: str, *, tenant_id: str = "local-dev") -> bool:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM auth_roles WHERE tenant_id = %s AND role_id = %s", (tenant_id, role_id))
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted

    def upsert_role_binding(self, record: AuthRoleBindingRecord) -> AuthRoleBindingRecord:
        binding_id = str(record.binding_id or _new_id("binding"))
        now = _utc_now()
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO auth_role_bindings
                        (binding_id, tenant_id, role_id, principal_id, created_at, disabled_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (tenant_id, role_id, principal_id) DO UPDATE SET
                        disabled_at = EXCLUDED.disabled_at
                    RETURNING *
                    """,
                    (
                        binding_id,
                        record.tenant_id,
                        str(record.role_id or ""),
                        str(record.principal_id or ""),
                        record.created_at or now,
                        str(record.disabled_at or "") or None,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return _row_to_binding(dict(row or {}))

    def list_role_bindings(
        self,
        *,
        tenant_id: str = "local-dev",
        role_id: str = "",
        principal_id: str = "",
        principal_ids: Sequence[str] | None = None,
        include_disabled: bool = True,
    ) -> List[AuthRoleBindingRecord]:
        sql = ["SELECT * FROM auth_role_bindings WHERE tenant_id = %s"]
        params: List[Any] = [tenant_id]
        if role_id:
            sql.append("AND role_id = %s")
            params.append(str(role_id))
        if principal_id:
            sql.append("AND principal_id = %s")
            params.append(str(principal_id))
        scoped_principal_ids = [str(item).strip() for item in (principal_ids or []) if str(item).strip()]
        if scoped_principal_ids:
            sql.append("AND principal_id = ANY(%s)")
            params.append(scoped_principal_ids)
        if not include_disabled:
            sql.append("AND disabled_at IS NULL")
        sql.append("ORDER BY created_at ASC")
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(" ".join(sql), params)
                rows = cur.fetchall()
        return [_row_to_binding(dict(row)) for row in rows]

    def delete_role_binding(self, binding_id: str, *, tenant_id: str = "local-dev") -> bool:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM auth_role_bindings WHERE tenant_id = %s AND binding_id = %s", (tenant_id, binding_id))
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted

    def upsert_role_permission(self, record: AuthRolePermissionRecord) -> AuthRolePermissionRecord:
        permission_id = str(record.permission_id or _new_id("perm"))
        now = _utc_now()
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO auth_role_permissions
                        (permission_id, tenant_id, role_id, resource_type, action, resource_selector, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (tenant_id, role_id, resource_type, action, resource_selector) DO UPDATE SET
                        created_at = auth_role_permissions.created_at
                    RETURNING *
                    """,
                    (
                        permission_id,
                        record.tenant_id,
                        str(record.role_id or ""),
                        _normalize_resource_type(record.resource_type),
                        _normalize_action(record.action),
                        _normalize_selector(record.resource_selector),
                        record.created_at or now,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return _row_to_permission(dict(row or {}))

    def list_role_permissions(
        self,
        *,
        tenant_id: str = "local-dev",
        role_id: str = "",
        role_ids: Sequence[str] | None = None,
        resource_type: str = "",
    ) -> List[AuthRolePermissionRecord]:
        sql = ["SELECT * FROM auth_role_permissions WHERE tenant_id = %s"]
        params: List[Any] = [tenant_id]
        if role_id:
            sql.append("AND role_id = %s")
            params.append(str(role_id))
        scoped_role_ids = [str(item).strip() for item in (role_ids or []) if str(item).strip()]
        if scoped_role_ids:
            sql.append("AND role_id = ANY(%s)")
            params.append(scoped_role_ids)
        if resource_type:
            sql.append("AND resource_type = %s")
            params.append(_normalize_resource_type(resource_type))
        sql.append("ORDER BY role_id ASC, resource_type ASC, action ASC, resource_selector ASC")
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(" ".join(sql), params)
                rows = cur.fetchall()
        return [_row_to_permission(dict(row)) for row in rows]

    def delete_role_permission(self, permission_id: str, *, tenant_id: str = "local-dev") -> bool:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM auth_role_permissions WHERE tenant_id = %s AND permission_id = %s",
                    (tenant_id, permission_id),
                )
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted
