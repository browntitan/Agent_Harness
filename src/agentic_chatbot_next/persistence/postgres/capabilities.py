from __future__ import annotations

import datetime as dt
from typing import Any

import psycopg2.extras

from agentic_chatbot_next.capabilities import CapabilityProfile, CapabilityProfileStore
from agentic_chatbot_next.persistence.postgres.connection import get_conn


class PostgresCapabilityProfileStore(CapabilityProfileStore):
    def get_profile(self, tenant_id: str, user_id: str) -> CapabilityProfile:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT profile_json
                    FROM capability_profiles
                    WHERE tenant_id = %s AND user_id = %s
                    LIMIT 1
                    """,
                    (tenant_id, user_id),
                )
                row = cur.fetchone()
        if row is None:
            return CapabilityProfile()
        return CapabilityProfile.from_dict(dict(row).get("profile_json") or {})

    def save_profile(self, tenant_id: str, user_id: str, profile: CapabilityProfile) -> CapabilityProfile:
        updated_at = dt.datetime.utcnow().isoformat() + "Z"
        payload: dict[str, Any] = profile.to_dict()
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO capability_profiles (tenant_id, user_id, profile_json, updated_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (tenant_id, user_id) DO UPDATE SET
                        profile_json = EXCLUDED.profile_json,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (tenant_id, user_id, psycopg2.extras.Json(payload), updated_at),
                )
            conn.commit()
        return profile
