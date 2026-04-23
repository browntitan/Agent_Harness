from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from agentic_chatbot_next.contracts.messages import utc_now_iso

SKILL_TELEMETRY_MIN_SCORED_USES = 10
SKILL_TELEMETRY_REVIEW_WINDOW = 20
SKILL_TELEMETRY_SUCCESS_SLO = 0.80


@dataclass
class SkillTelemetryEvent:
    event_id: str
    tenant_id: str
    skill_id: str
    skill_family_id: str
    query: str
    answer_quality: str
    agent_name: str = ""
    session_id: str = ""
    created_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def build(
        cls,
        *,
        tenant_id: str,
        skill_id: str,
        skill_family_id: str,
        query: str,
        answer_quality: str,
        agent_name: str = "",
        session_id: str = "",
    ) -> "SkillTelemetryEvent":
        return cls(
            event_id=f"ste_{uuid.uuid4().hex[:16]}",
            tenant_id=str(tenant_id or ""),
            skill_id=str(skill_id or ""),
            skill_family_id=str(skill_family_id or ""),
            query=str(query or ""),
            answer_quality=coerce_answer_quality(answer_quality),
            agent_name=str(agent_name or ""),
            session_id=str(session_id or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "tenant_id": self.tenant_id,
            "skill_id": self.skill_id,
            "skill_family_id": self.skill_family_id,
            "query": self.query,
            "answer_quality": self.answer_quality,
            "agent_name": self.agent_name,
            "session_id": self.session_id,
            "created_at": self.created_at,
        }


@dataclass
class SkillHealthSummary:
    skill_family_id: str
    scored_uses: int = 0
    success_rate: float | None = None
    last_scored_at: str = ""
    review_status: str = "insufficient_data"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_family_id": self.skill_family_id,
            "scored_uses": int(self.scored_uses),
            "success_rate": self.success_rate,
            "last_scored_at": self.last_scored_at,
            "review_status": self.review_status,
        }


def coerce_answer_quality(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"pass", "revise"}:
        return normalized
    return ""


def is_scored_answer_quality(value: Any) -> bool:
    return coerce_answer_quality(value) in {"pass", "revise"}


def answer_quality_to_success(value: Any) -> bool | None:
    normalized = coerce_answer_quality(value)
    if normalized == "pass":
        return True
    if normalized == "revise":
        return False
    return None


def _telemetry_field(item: Any, field_name: str) -> Any:
    getter = getattr(item, "get", None)
    if callable(getter):
        try:
            return getter(field_name)
        except TypeError:
            pass
    return getattr(item, field_name, "")


def _normalize_skill_telemetry_event(item: Any) -> SkillTelemetryEvent | None:
    if item is None:
        return None
    if isinstance(item, SkillTelemetryEvent):
        return item
    return SkillTelemetryEvent(
        event_id=str(_telemetry_field(item, "event_id") or ""),
        tenant_id=str(_telemetry_field(item, "tenant_id") or ""),
        skill_id=str(_telemetry_field(item, "skill_id") or ""),
        skill_family_id=str(_telemetry_field(item, "skill_family_id") or ""),
        query=str(_telemetry_field(item, "query") or ""),
        answer_quality=coerce_answer_quality(_telemetry_field(item, "answer_quality")),
        agent_name=str(_telemetry_field(item, "agent_name") or ""),
        session_id=str(_telemetry_field(item, "session_id") or ""),
        created_at=str(_telemetry_field(item, "created_at") or ""),
    )


def compute_skill_health(
    events: Sequence[SkillTelemetryEvent | Mapping[str, Any]],
    *,
    min_scored_uses: int = SKILL_TELEMETRY_MIN_SCORED_USES,
    review_window: int = SKILL_TELEMETRY_REVIEW_WINDOW,
    success_slo: float = SKILL_TELEMETRY_SUCCESS_SLO,
) -> SkillHealthSummary:
    rows: List[SkillTelemetryEvent] = []
    for item in events:
        normalized = _normalize_skill_telemetry_event(item)
        if normalized is not None:
            rows.append(normalized)
    rows = [
        row
        for row in rows
        if row.skill_family_id and is_scored_answer_quality(row.answer_quality)
    ]
    rows.sort(key=lambda row: str(row.created_at or ""), reverse=True)
    if not rows:
        return SkillHealthSummary(skill_family_id="")

    family_id = rows[0].skill_family_id
    window = rows[: max(1, int(review_window))]
    successes = [
        answer_quality_to_success(row.answer_quality)
        for row in window
        if answer_quality_to_success(row.answer_quality) is not None
    ]
    if not successes:
        return SkillHealthSummary(
            skill_family_id=family_id,
            scored_uses=len(rows),
            last_scored_at=str(rows[0].created_at or ""),
        )

    success_rate = sum(1 for item in successes if item) / len(successes)
    review_status = "insufficient_data"
    if len(rows) >= int(min_scored_uses):
        review_status = "flagged" if success_rate < float(success_slo) else "ok"

    return SkillHealthSummary(
        skill_family_id=family_id,
        scored_uses=len(rows),
        success_rate=round(success_rate, 4),
        last_scored_at=str(rows[0].created_at or ""),
        review_status=review_status,
    )


def compute_skill_health_by_family(
    events: Iterable[SkillTelemetryEvent | Mapping[str, Any]],
    *,
    min_scored_uses: int = SKILL_TELEMETRY_MIN_SCORED_USES,
    review_window: int = SKILL_TELEMETRY_REVIEW_WINDOW,
    success_slo: float = SKILL_TELEMETRY_SUCCESS_SLO,
) -> Dict[str, SkillHealthSummary]:
    grouped: Dict[str, List[SkillTelemetryEvent | Mapping[str, Any]]] = {}
    for item in events:
        normalized = _normalize_skill_telemetry_event(item)
        if normalized is None or not normalized.skill_family_id:
            continue
        grouped.setdefault(normalized.skill_family_id, []).append(normalized)
    return {
        family_id: compute_skill_health(
            family_events,
            min_scored_uses=min_scored_uses,
            review_window=review_window,
            success_slo=success_slo,
        )
        for family_id, family_events in grouped.items()
    }


__all__ = [
    "SKILL_TELEMETRY_MIN_SCORED_USES",
    "SKILL_TELEMETRY_REVIEW_WINDOW",
    "SKILL_TELEMETRY_SUCCESS_SLO",
    "SkillHealthSummary",
    "SkillTelemetryEvent",
    "answer_quality_to_success",
    "coerce_answer_quality",
    "compute_skill_health",
    "compute_skill_health_by_family",
    "is_scored_answer_quality",
]
