from __future__ import annotations

from agentic_chatbot_next.persistence.postgres.skills import SkillTelemetryEventRecord
from agentic_chatbot_next.skills.telemetry import compute_skill_health_by_family


def test_compute_skill_health_by_family_accepts_persisted_event_records() -> None:
    summaries = compute_skill_health_by_family(
        [
            SkillTelemetryEventRecord(
                event_id="evt-1",
                tenant_id="tenant-a",
                skill_id="workflow-skill-v1",
                skill_family_id="workflow-skill",
                query="find workflow docs",
                answer_quality="pass",
                agent_name="general",
                session_id="session-1",
                created_at="2026-04-19T14:00:00Z",
            ),
            SkillTelemetryEventRecord(
                event_id="evt-2",
                tenant_id="tenant-a",
                skill_id="workflow-skill-v2",
                skill_family_id="workflow-skill",
                query="find escalation docs",
                answer_quality="revise",
                agent_name="general",
                session_id="session-2",
                created_at="2026-04-19T14:05:00Z",
            ),
        ],
        min_scored_uses=1,
    )

    assert list(summaries) == ["workflow-skill"]
    summary = summaries["workflow-skill"]
    assert summary.skill_family_id == "workflow-skill"
    assert summary.scored_uses == 2
    assert summary.success_rate == 0.5
    assert summary.review_status == "flagged"
    assert summary.last_scored_at == "2026-04-19T14:05:00Z"
