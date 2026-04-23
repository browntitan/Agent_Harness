from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.router.feedback_loop import (
    OUTCOME_NEGATIVE,
    OUTCOME_POSITIVE,
    RouterFeedbackLoop,
)
from agentic_chatbot_next.runtime.context import RuntimePaths


def _write_patterns(path: Path) -> None:
    payload = {
        "tool_or_multistep_intent": {"phrases": ["help me"], "regexes": []},
        "data_analysis_intent": {"phrases": ["analyze"], "regexes": []},
        "citation_grounding_intent": {"phrases": ["cite"], "regexes": []},
        "high_stakes_intent": {"phrases": ["legal"], "regexes": []},
        "kb_inventory_intent": {"phrases": ["what docs"], "regexes": []},
        "coordinator_campaign_intent": {"phrases": ["coordinate"], "regexes": []},
        "rag_grounding_intent": {"phrases": ["source"], "regexes": []},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _settings(tmp_path: Path, **overrides: object) -> SimpleNamespace:
    runtime_dir = tmp_path / "runtime"
    patterns_path = tmp_path / "router" / "intent_patterns.json"
    _write_patterns(patterns_path)
    payload = {
        "runtime_dir": runtime_dir,
        "workspace_dir": tmp_path / "workspaces",
        "memory_dir": tmp_path / "memory",
        "router_patterns_path": patterns_path,
        "router_feedback_enabled": True,
        "router_feedback_rephrase_window_seconds": 600,
        "router_feedback_neutral_sample_rate": 1.0,
        "router_feedback_tenant_daily_review_cap": 5,
        "router_retrain_governance": "human_reviewed",
        "llm_router_confidence_threshold": 0.7,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _session() -> SimpleNamespace:
    return SimpleNamespace(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id="conv-a",
        request_id="req-a",
        session_id="tenant-a:user-a:conv-a",
    )


def test_router_feedback_scores_manual_override_as_negative_and_samples_it(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    feedback = RouterFeedbackLoop(RuntimePaths.from_settings(settings), settings)
    session = _session()
    state = SessionState.from_session(session)

    record = feedback.register_decision(
        session,
        user_text="Please cite the source for this answer",
        route="AGENT",
        confidence=0.82,
        reasons=["document_grounding_intent"],
        router_method="hybrid",
        suggested_agent="rag_worker",
        force_agent=False,
        has_attachments=False,
    )
    state.metadata["pending_router_feedback_id"] = record.router_decision_id

    outcome = feedback.observe_turn_result(
        state,
        metadata={},
        route_context={
            "router_decision_id": record.router_decision_id,
            "requested_agent_override": "general",
            "requested_agent_override_applied": True,
        },
    )

    assert outcome is not None
    assert outcome.outcome_label == OUTCOME_NEGATIVE
    assert "manual_agent_override" in outcome.evidence_signals
    review_samples = feedback.list_review_samples(limit=10)
    assert len(review_samples) == 1
    assert review_samples[0].router_decision_id == record.router_decision_id


def test_router_feedback_scores_verifier_pass_as_positive_after_stale_finalization(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    feedback = RouterFeedbackLoop(RuntimePaths.from_settings(settings), settings)
    session = _session()
    state = SessionState.from_session(session)

    record = feedback.register_decision(
        session,
        user_text="Summarize the policy memo",
        route="AGENT",
        confidence=0.78,
        reasons=["tool_or_multistep_intent"],
        router_method="hybrid",
        suggested_agent="general",
        force_agent=False,
        has_attachments=False,
    )
    state.metadata["pending_router_feedback_id"] = record.router_decision_id
    feedback.observe_turn_result(
        state,
        metadata={"verification": {"status": "pass"}},
        route_context={"router_decision_id": record.router_decision_id},
    )

    resolved = feedback.finalize_stale_decisions(now="2030-04-14T12:30:00+00:00")

    assert len(resolved) == 1
    assert resolved[0].outcome_label == OUTCOME_POSITIVE
    assert "verifier_pass" in resolved[0].evidence_signals
    assert feedback.list_review_samples(limit=10) == []


def test_router_feedback_marks_rephrase_like_followups_negative_and_respects_daily_cap(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        router_feedback_tenant_daily_review_cap=1,
        router_feedback_neutral_sample_rate=1.0,
    )
    feedback = RouterFeedbackLoop(RuntimePaths.from_settings(settings), settings)
    session = _session()

    first_state = SessionState.from_session(session)
    first_record = feedback.register_decision(
        session,
        user_text="Explain the incident timeline",
        route="AGENT",
        confidence=0.8,
        reasons=["tool_or_multistep_intent"],
        router_method="hybrid",
        suggested_agent="general",
        force_agent=False,
        has_attachments=False,
    )
    first_state.metadata["pending_router_feedback_id"] = first_record.router_decision_id
    first_outcome = feedback.observe_followup_user_turn(
        first_state,
        user_text="That missed the point, try again and explain the incident timeline",
    )

    second_state = SessionState.from_session(session)
    second_record = feedback.register_decision(
        session,
        user_text="Rewrite the answer with citations",
        route="AGENT",
        confidence=0.76,
        reasons=["document_grounding_intent"],
        router_method="hybrid",
        suggested_agent="rag_worker",
        force_agent=False,
        has_attachments=False,
    )
    second_state.metadata["pending_router_feedback_id"] = second_record.router_decision_id
    second_outcome = feedback.observe_followup_user_turn(
        second_state,
        user_text="wrong answer, rewrite the answer with citations",
    )

    assert first_outcome is not None
    assert first_outcome.outcome_label == OUTCOME_NEGATIVE
    assert "rephrase_like_followup" in first_outcome.evidence_signals
    assert second_outcome is not None
    assert second_outcome.outcome_label == OUTCOME_NEGATIVE
    review_samples = feedback.list_review_samples(limit=10)
    assert len(review_samples) == 1
    assert review_samples[0].router_decision_id == first_record.router_decision_id


def test_router_feedback_generates_quarterly_artifacts_without_mutating_live_patterns(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    feedback = RouterFeedbackLoop(RuntimePaths.from_settings(settings), settings)
    session = _session()
    original_patterns = settings.router_patterns_path.read_text(encoding="utf-8")

    for index in range(3):
        record = feedback.register_decision(
            session,
            user_text=f"Help me with request {index}",
            route="AGENT",
            confidence=0.75,
            reasons=["tool_or_multistep_intent"],
            router_method="hybrid",
            suggested_agent="general" if index < 2 else "rag_worker",
            force_agent=False,
            has_attachments=False,
        )
        state = SessionState.from_session(session)
        state.metadata["pending_router_feedback_id"] = record.router_decision_id
        if index < 2:
            feedback.observe_turn_result(
                state,
                metadata={"verification": {"status": "revise"}},
                route_context={"router_decision_id": record.router_decision_id},
            )
        else:
            feedback.observe_turn_result(
                state,
                metadata={"verification": {"status": "pass"}},
                route_context={"router_decision_id": record.router_decision_id},
            )
            feedback.finalize_stale_decisions(now="2030-04-14T12:30:00+00:00")

    report = feedback.generate_quarterly_retrain_artifacts(force=True, now="2030-04-14T12:45:00+00:00")
    report_dir = Path(str(report["report_dir"]))

    assert report_dir.exists()
    assert (report_dir / "evaluation_report.json").exists()
    assert (report_dir / "intent_patterns_candidate.json").exists()
    assert (report_dir / "threshold_recommendation.json").exists()
    assert settings.router_patterns_path.read_text(encoding="utf-8") == original_patterns
