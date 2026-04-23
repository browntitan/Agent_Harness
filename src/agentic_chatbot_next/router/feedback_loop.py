from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

from agentic_chatbot_next.contracts.messages import SessionState, utc_now_iso
from agentic_chatbot_next.runtime.context import RuntimePaths
from agentic_chatbot_next.router.patterns import patterns_path_from_settings

OUTCOME_POSITIVE = "positive"
OUTCOME_NEUTRAL = "neutral"
OUTCOME_NEGATIVE = "negative"
REVIEW_STATUS_PENDING = "pending"
DEFAULT_NEUTRAL_SAMPLE_RATE = 0.10
DEFAULT_TENANT_DAILY_REVIEW_CAP = 25
_WORD_RE = re.compile(r"[A-Za-z0-9_]{3,}")
_DISSATISFACTION_RE = re.compile(
    r"\b("
    r"retry|try\s+again|again|wrong|not\s+what\s+i\s+asked|"
    r"that'?s\s+not|you\s+missed|missed\s+the\s+point|"
    r"rephrase|rewrite|paraphrase|clarify|different\s+answer"
    r")\b",
    flags=re.IGNORECASE,
)


def build_router_decision_id() -> str:
    return f"rtd_{uuid.uuid4().hex[:16]}"


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    return json.loads(raw)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _parse_dt(value: Any) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        return datetime.now(timezone.utc)
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return datetime.now(timezone.utc)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _normalize_signal_list(values: Iterable[Any]) -> List[str]:
    seen: set[str] = set()
    normalized: List[str] = []
    for item in values:
        clean = str(item or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)
    return normalized


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(str(text or ""))}


def _jaccard_similarity(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return len(overlap) / max(1, len(union))


def _is_rephrase_like(previous_text: str, new_text: str) -> bool:
    old = str(previous_text or "").strip().lower()
    new = str(new_text or "").strip().lower()
    if not old or not new:
        return False
    if _DISSATISFACTION_RE.search(new):
        return True
    similarity = _jaccard_similarity(old, new)
    if similarity >= 0.45:
        return True
    if len(new) >= 24 and (new in old or old in new):
        return True
    return False


def _quarter_id(now: datetime) -> str:
    quarter = ((now.month - 1) // 3) + 1
    return f"{now.year}-Q{quarter}"


@dataclass
class RouterDecisionRecord:
    router_decision_id: str
    session_id: str
    tenant_id: str
    user_id: str
    conversation_id: str
    request_id: str
    route: str
    confidence: float
    reasons: List[str] = field(default_factory=list)
    suggested_agent: str = ""
    router_method: str = "deterministic"
    user_text: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    force_agent: bool = False
    has_attachments: bool = False
    requested_agent_override: str = ""
    requested_agent_override_applied: bool = False
    router_evidence: Dict[str, Any] = field(default_factory=dict)
    verification_status: str = ""
    evidence_signals: List[str] = field(default_factory=list)
    outcome_label: str = ""
    scored_at: str = ""
    sampled_for_review: bool = False
    review_sample_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RouterDecisionRecord":
        return cls(
            router_decision_id=str(raw.get("router_decision_id") or build_router_decision_id()),
            session_id=str(raw.get("session_id") or ""),
            tenant_id=str(raw.get("tenant_id") or ""),
            user_id=str(raw.get("user_id") or ""),
            conversation_id=str(raw.get("conversation_id") or ""),
            request_id=str(raw.get("request_id") or ""),
            route=str(raw.get("route") or ""),
            confidence=float(raw.get("confidence") or 0.0),
            reasons=[str(item) for item in (raw.get("reasons") or []) if str(item)],
            suggested_agent=str(raw.get("suggested_agent") or ""),
            router_method=str(raw.get("router_method") or "deterministic"),
            user_text=str(raw.get("user_text") or ""),
            created_at=str(raw.get("created_at") or utc_now_iso()),
            force_agent=_coerce_bool(raw.get("force_agent")),
            has_attachments=_coerce_bool(raw.get("has_attachments")),
            requested_agent_override=str(raw.get("requested_agent_override") or ""),
            requested_agent_override_applied=_coerce_bool(raw.get("requested_agent_override_applied")),
            router_evidence=dict(raw.get("router_evidence") or {}),
            verification_status=str(raw.get("verification_status") or ""),
            evidence_signals=_normalize_signal_list(raw.get("evidence_signals") or []),
            outcome_label=str(raw.get("outcome_label") or ""),
            scored_at=str(raw.get("scored_at") or ""),
            sampled_for_review=_coerce_bool(raw.get("sampled_for_review")),
            review_sample_id=str(raw.get("review_sample_id") or ""),
        )

    @property
    def is_scored(self) -> bool:
        return bool(self.outcome_label and self.scored_at)


@dataclass
class RouterOutcomeRecord:
    router_decision_id: str
    session_id: str
    tenant_id: str
    user_id: str
    conversation_id: str
    request_id: str
    route: str
    router_method: str
    suggested_agent: str
    outcome_label: str
    evidence_signals: List[str] = field(default_factory=list)
    scored_at: str = field(default_factory=utc_now_iso)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RouterOutcomeRecord":
        return cls(
            router_decision_id=str(raw.get("router_decision_id") or ""),
            session_id=str(raw.get("session_id") or ""),
            tenant_id=str(raw.get("tenant_id") or ""),
            user_id=str(raw.get("user_id") or ""),
            conversation_id=str(raw.get("conversation_id") or ""),
            request_id=str(raw.get("request_id") or ""),
            route=str(raw.get("route") or ""),
            router_method=str(raw.get("router_method") or ""),
            suggested_agent=str(raw.get("suggested_agent") or ""),
            outcome_label=str(raw.get("outcome_label") or ""),
            evidence_signals=_normalize_signal_list(raw.get("evidence_signals") or []),
            scored_at=str(raw.get("scored_at") or utc_now_iso()),
            created_at=str(raw.get("created_at") or ""),
        )


@dataclass
class RouterReviewSample:
    sample_id: str
    router_decision_id: str
    tenant_id: str
    session_id: str
    route: str
    router_method: str
    suggested_agent: str
    outcome_label: str
    evidence_signals: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    review_status: str = REVIEW_STATUS_PENDING

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RouterReviewSample":
        return cls(
            sample_id=str(raw.get("sample_id") or f"rrs_{uuid.uuid4().hex[:16]}"),
            router_decision_id=str(raw.get("router_decision_id") or ""),
            tenant_id=str(raw.get("tenant_id") or ""),
            session_id=str(raw.get("session_id") or ""),
            route=str(raw.get("route") or ""),
            router_method=str(raw.get("router_method") or ""),
            suggested_agent=str(raw.get("suggested_agent") or ""),
            outcome_label=str(raw.get("outcome_label") or ""),
            evidence_signals=_normalize_signal_list(raw.get("evidence_signals") or []),
            created_at=str(raw.get("created_at") or utc_now_iso()),
            review_status=str(raw.get("review_status") or REVIEW_STATUS_PENDING),
        )


class RouterFeedbackLoop:
    def __init__(
        self,
        paths: RuntimePaths,
        settings: Any,
        *,
        emit_event: Callable[[str, str, Dict[str, Any]], None] | None = None,
    ) -> None:
        self.paths = paths
        self.settings = settings
        self.emit_event = emit_event
        self.enabled = bool(getattr(settings, "router_feedback_enabled", True))
        self.rephrase_window_seconds = max(
            60,
            int(getattr(settings, "router_feedback_rephrase_window_seconds", 600)),
        )
        self.neutral_sample_rate = min(
            1.0,
            max(0.0, float(getattr(settings, "router_feedback_neutral_sample_rate", DEFAULT_NEUTRAL_SAMPLE_RATE))),
        )
        self.tenant_daily_review_cap = max(
            1,
            int(getattr(settings, "router_feedback_tenant_daily_review_cap", DEFAULT_TENANT_DAILY_REVIEW_CAP)),
        )
        self.retrain_governance = str(
            getattr(settings, "router_retrain_governance", "human_reviewed") or "human_reviewed"
        ).strip().lower()
        self.patterns_path = patterns_path_from_settings(settings)
        self.root_dir = self.paths.runtime_root / "router_feedback"
        self.decisions_dir = self.root_dir / "decisions"
        self.outcomes_path = self.root_dir / "outcomes.jsonl"
        self.review_pool_path = self.root_dir / "review_pool.jsonl"
        self.reports_dir = self.root_dir / "reports"
        self.latest_report_path = self.reports_dir / "latest.json"
        self.decisions_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def decision_path(self, router_decision_id: str) -> Path:
        return self.decisions_dir / f"{router_decision_id}.json"

    def load_decision(self, router_decision_id: str) -> RouterDecisionRecord | None:
        path = self.decision_path(router_decision_id)
        if not path.exists():
            return None
        return RouterDecisionRecord.from_dict(_read_json(path))

    def save_decision(self, record: RouterDecisionRecord) -> RouterDecisionRecord:
        _write_json(self.decision_path(record.router_decision_id), record.to_dict())
        return record

    def list_decisions(self, *, pending_only: bool = False) -> List[RouterDecisionRecord]:
        rows: List[RouterDecisionRecord] = []
        for path in sorted(self.decisions_dir.glob("*.json")):
            try:
                record = RouterDecisionRecord.from_dict(_read_json(path))
            except Exception:
                continue
            if pending_only and record.is_scored:
                continue
            rows.append(record)
        rows.sort(key=lambda item: item.created_at)
        return rows

    def list_recent_outcomes(self, *, limit: int = 200) -> List[RouterOutcomeRecord]:
        rows = [RouterOutcomeRecord.from_dict(item) for item in _read_jsonl(self.outcomes_path)]
        rows.sort(key=lambda item: item.scored_at, reverse=True)
        return rows[: max(1, int(limit))]

    def list_review_samples(self, *, limit: int = 200) -> List[RouterReviewSample]:
        rows = [RouterReviewSample.from_dict(item) for item in _read_jsonl(self.review_pool_path)]
        rows.sort(key=lambda item: item.created_at, reverse=True)
        return rows[: max(1, int(limit))]

    def get_last_retrain_report_metadata(self) -> Dict[str, Any]:
        return _read_json(self.latest_report_path)

    def register_decision(
        self,
        session: Any,
        *,
        user_text: str,
        route: str,
        confidence: float,
        reasons: Sequence[str],
        router_method: str,
        suggested_agent: str,
        force_agent: bool,
        has_attachments: bool,
        requested_agent_override: str = "",
        requested_agent_override_applied: bool = False,
        router_decision_id: str = "",
        router_evidence: Dict[str, Any] | None = None,
    ) -> RouterDecisionRecord:
        record = RouterDecisionRecord(
            router_decision_id=str(router_decision_id or build_router_decision_id()),
            session_id=str(getattr(session, "session_id", "") or ""),
            tenant_id=str(getattr(session, "tenant_id", "") or ""),
            user_id=str(getattr(session, "user_id", "") or ""),
            conversation_id=str(getattr(session, "conversation_id", "") or ""),
            request_id=str(getattr(session, "request_id", "") or ""),
            route=str(route or ""),
            confidence=float(confidence or 0.0),
            reasons=_normalize_signal_list(reasons),
            suggested_agent=str(suggested_agent or ""),
            router_method=str(router_method or "deterministic"),
            user_text=str(user_text or ""),
            force_agent=bool(force_agent),
            has_attachments=bool(has_attachments),
            requested_agent_override=str(requested_agent_override or ""),
            requested_agent_override_applied=bool(requested_agent_override_applied),
            router_evidence=dict(router_evidence or {}),
        )
        if not self.enabled:
            return record
        self.save_decision(record)
        return record

    def observe_followup_user_turn(self, session_state: SessionState, *, user_text: str) -> RouterOutcomeRecord | None:
        if not self.enabled:
            session_state.metadata.pop("pending_router_feedback_id", None)
            return None
        router_decision_id = str((session_state.metadata or {}).get("pending_router_feedback_id") or "").strip()
        if not router_decision_id:
            return None
        record = self.load_decision(router_decision_id)
        if record is None:
            session_state.metadata.pop("pending_router_feedback_id", None)
            return None
        if record.is_scored:
            session_state.metadata.pop("pending_router_feedback_id", None)
            return None
        now = utc_now_iso()
        if (
            _parse_dt(now) - _parse_dt(record.created_at)
        ) <= timedelta(seconds=self.rephrase_window_seconds) and _is_rephrase_like(record.user_text, user_text):
            outcome = self._finalize_record(
                record,
                outcome_label=OUTCOME_NEGATIVE,
                evidence_signals=["rephrase_like_followup"],
                scored_at=now,
            )
        else:
            outcome = self._finalize_pending_record(record, scored_at=now)
        session_state.metadata.pop("pending_router_feedback_id", None)
        return outcome

    def observe_turn_result(
        self,
        session_state: SessionState,
        *,
        metadata: Dict[str, Any],
        route_context: Dict[str, Any] | None = None,
    ) -> RouterOutcomeRecord | None:
        if not self.enabled:
            return None
        context = dict(route_context or {})
        router_decision_id = str(
            context.get("router_decision_id")
            or (session_state.metadata or {}).get("pending_router_feedback_id")
            or ""
        ).strip()
        if not router_decision_id:
            return None
        record = self.load_decision(router_decision_id)
        if record is None or record.is_scored:
            return None

        negative_signals: List[str] = []
        verification = dict(metadata.get("verification") or {})
        verification_status = str(verification.get("status") or "").strip().lower()
        if verification_status in {"pass", "revise"}:
            record.verification_status = verification_status
        if _coerce_bool(context.get("requested_agent_override_applied")) and str(
            context.get("requested_agent_override") or ""
        ).strip():
            negative_signals.append("manual_agent_override")
            record.requested_agent_override = str(context.get("requested_agent_override") or "")
            record.requested_agent_override_applied = True
        if _coerce_bool(context.get("degraded_service")) or _coerce_bool(context.get("degraded_from_agent")):
            negative_signals.append("degraded_fallback")
        if verification_status == "revise":
            negative_signals.append("verifier_revise")

        record.evidence_signals = _normalize_signal_list([*record.evidence_signals, *negative_signals])
        self.save_decision(record)
        if negative_signals:
            outcome = self._finalize_record(
                record,
                outcome_label=OUTCOME_NEGATIVE,
                evidence_signals=negative_signals,
            )
            session_state.metadata.pop("pending_router_feedback_id", None)
            return outcome
        return None

    def finalize_stale_decisions(self, *, now: str | None = None, limit: int = 500) -> List[RouterOutcomeRecord]:
        if not self.enabled:
            return []
        cutoff = _parse_dt(now or utc_now_iso())
        resolved: List[RouterOutcomeRecord] = []
        pending = self.list_decisions(pending_only=True)
        for record in pending[: max(1, int(limit))]:
            if record.is_scored:
                continue
            age = cutoff - _parse_dt(record.created_at)
            if age < timedelta(seconds=self.rephrase_window_seconds):
                continue
            resolved.append(self._finalize_pending_record(record, scored_at=cutoff.isoformat()))
        return resolved

    def generate_quarterly_retrain_artifacts(
        self,
        *,
        force: bool = False,
        now: str | None = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "disabled": True,
                "generated_at": _parse_dt(now or utc_now_iso()).isoformat(),
                "governance": self.retrain_governance,
            }
        timestamp = _parse_dt(now or utc_now_iso())
        quarter = _quarter_id(timestamp)
        destination = self.reports_dir / quarter
        metadata = _read_json(destination / "report.json")
        if metadata and not force:
            _write_json(self.latest_report_path, metadata)
            return metadata

        self.finalize_stale_decisions(now=timestamp.isoformat())
        outcomes = self.list_recent_outcomes(limit=5000)
        current_patterns = _read_json(self.patterns_path)
        negative = [item for item in outcomes if item.outcome_label == OUTCOME_NEGATIVE]
        clusters: Dict[str, int] = Counter(
            f"{item.route}|{item.router_method}|{item.suggested_agent or 'default'}"
            for item in negative
        )
        negative_rate = (len(negative) / len(outcomes)) if outcomes else 0.0
        current_threshold = float(getattr(self.settings, "llm_router_confidence_threshold", 0.70))
        threshold_delta = 0.05 if negative_rate >= 0.35 else (-0.05 if negative_rate <= 0.10 else 0.0)
        recommended_threshold = round(min(0.95, max(0.35, current_threshold + threshold_delta)), 2)
        phrase_suggestions: Dict[str, List[str]] = defaultdict(list)
        decision_index = {decision.router_decision_id: decision for decision in self.list_decisions()}
        for outcome in negative[:100]:
            decision = decision_index.get(outcome.router_decision_id)
            if decision is None:
                continue
            tokens = [token for token in _tokenize(decision.user_text) if len(token) >= 5]
            if not tokens:
                continue
            route_key = outcome.suggested_agent or outcome.route.lower() or "default"
            phrase_suggestions[route_key].extend(tokens[:6])
        candidate_updates = {
            key: [phrase for phrase, _ in Counter(values).most_common(12)]
            for key, values in phrase_suggestions.items()
            if values
        }
        evaluation_report = {
            "quarter": quarter,
            "generated_at": timestamp.isoformat(),
            "governance": self.retrain_governance,
            "decision_count": len(decision_index),
            "outcome_count": len(outcomes),
            "negative_rate": round(negative_rate, 4),
            "top_failure_clusters": [
                {"cluster": cluster, "count": count}
                for cluster, count in clusters.most_common(12)
            ],
        }
        candidate_patterns = {
            "base_patterns_path": str(self.patterns_path),
            "generated_at": timestamp.isoformat(),
            "current_patterns": current_patterns,
            "candidate_updates": candidate_updates,
        }
        threshold_report = {
            "current_threshold": current_threshold,
            "recommended_threshold": recommended_threshold,
            "generated_at": timestamp.isoformat(),
            "basis": {
                "negative_rate": round(negative_rate, 4),
                "outcome_count": len(outcomes),
            },
        }
        report_metadata = {
            "quarter": quarter,
            "generated_at": timestamp.isoformat(),
            "governance": self.retrain_governance,
            "report_dir": str(destination),
            "outcome_count": len(outcomes),
            "negative_rate": round(negative_rate, 4),
            "recommended_threshold": recommended_threshold,
        }
        destination.mkdir(parents=True, exist_ok=True)
        _write_json(destination / "evaluation_report.json", evaluation_report)
        _write_json(destination / "intent_patterns_candidate.json", candidate_patterns)
        _write_json(destination / "threshold_recommendation.json", threshold_report)
        _write_json(destination / "failure_clusters.json", {"clusters": evaluation_report["top_failure_clusters"]})
        _write_json(destination / "report.json", report_metadata)
        _write_json(self.latest_report_path, report_metadata)
        return report_metadata

    def _finalize_pending_record(self, record: RouterDecisionRecord, *, scored_at: str) -> RouterOutcomeRecord:
        label = OUTCOME_POSITIVE if record.verification_status == "pass" else OUTCOME_NEUTRAL
        signals = ["verifier_pass"] if label == OUTCOME_POSITIVE else ["no_strong_signal"]
        return self._finalize_record(record, outcome_label=label, evidence_signals=signals, scored_at=scored_at)

    def _finalize_record(
        self,
        record: RouterDecisionRecord,
        *,
        outcome_label: str,
        evidence_signals: Sequence[str],
        scored_at: str | None = None,
    ) -> RouterOutcomeRecord:
        if record.is_scored:
            return RouterOutcomeRecord(
                router_decision_id=record.router_decision_id,
                session_id=record.session_id,
                tenant_id=record.tenant_id,
                user_id=record.user_id,
                conversation_id=record.conversation_id,
                request_id=record.request_id,
                route=record.route,
                router_method=record.router_method,
                suggested_agent=record.suggested_agent,
                outcome_label=record.outcome_label,
                evidence_signals=list(record.evidence_signals),
                scored_at=record.scored_at,
                created_at=record.created_at,
            )
        record.outcome_label = str(outcome_label or OUTCOME_NEUTRAL)
        record.scored_at = str(scored_at or utc_now_iso())
        record.evidence_signals = _normalize_signal_list([*record.evidence_signals, *evidence_signals])
        self.save_decision(record)
        outcome = RouterOutcomeRecord(
            router_decision_id=record.router_decision_id,
            session_id=record.session_id,
            tenant_id=record.tenant_id,
            user_id=record.user_id,
            conversation_id=record.conversation_id,
            request_id=record.request_id,
            route=record.route,
            router_method=record.router_method,
            suggested_agent=record.suggested_agent,
            outcome_label=record.outcome_label,
            evidence_signals=list(record.evidence_signals),
            scored_at=record.scored_at,
            created_at=record.created_at,
        )
        _append_jsonl(self.outcomes_path, outcome.to_dict())
        self._emit(
            "router_outcome",
            record.session_id,
            {
                "router_decision_id": record.router_decision_id,
                "route": record.route,
                "router_method": record.router_method,
                "suggested_agent": record.suggested_agent,
                "outcome_label": record.outcome_label,
                "evidence_signals": list(record.evidence_signals),
                "tenant_id": record.tenant_id,
                "user_id": record.user_id,
                "conversation_id": record.conversation_id,
                "request_id": record.request_id,
                "scored_at": record.scored_at,
            },
        )
        self._maybe_sample_for_review(record, outcome)
        return outcome

    def _maybe_sample_for_review(
        self,
        record: RouterDecisionRecord,
        outcome: RouterOutcomeRecord,
    ) -> RouterReviewSample | None:
        if record.sampled_for_review:
            return None
        if outcome.outcome_label == OUTCOME_NEGATIVE:
            should_sample = True
        elif outcome.outcome_label == OUTCOME_NEUTRAL:
            digest = int(hashlib.sha1(record.router_decision_id.encode("utf-8")).hexdigest()[:8], 16)
            should_sample = (digest / 0xFFFFFFFF) <= self.neutral_sample_rate
        else:
            should_sample = False
        if not should_sample or not self._tenant_within_review_cap(record.tenant_id, outcome.scored_at):
            return None
        sample = RouterReviewSample(
            sample_id=f"rrs_{uuid.uuid4().hex[:16]}",
            router_decision_id=record.router_decision_id,
            tenant_id=record.tenant_id,
            session_id=record.session_id,
            route=record.route,
            router_method=record.router_method,
            suggested_agent=record.suggested_agent,
            outcome_label=outcome.outcome_label,
            evidence_signals=list(outcome.evidence_signals),
        )
        _append_jsonl(self.review_pool_path, sample.to_dict())
        record.sampled_for_review = True
        record.review_sample_id = sample.sample_id
        self.save_decision(record)
        self._emit(
            "router_mispick_sampled",
            record.session_id,
            {
                "router_decision_id": record.router_decision_id,
                "sample_id": sample.sample_id,
                "outcome_label": outcome.outcome_label,
                "tenant_id": record.tenant_id,
                "route": record.route,
                "router_method": record.router_method,
            },
        )
        return sample

    def _tenant_within_review_cap(self, tenant_id: str, scored_at: str) -> bool:
        day = _parse_dt(scored_at).date().isoformat()
        count = 0
        for row in _read_jsonl(self.review_pool_path):
            if str(row.get("tenant_id") or "") != str(tenant_id or ""):
                continue
            created_at = str(row.get("created_at") or "")
            if created_at and _parse_dt(created_at).date().isoformat() == day:
                count += 1
        return count < self.tenant_daily_review_cap

    def _emit(self, event_type: str, session_id: str, payload: Dict[str, Any]) -> None:
        if self.emit_event is None or not session_id:
            return
        self.emit_event(event_type, session_id, payload)


def summarize_router_outcomes(
    outcomes: Iterable[RouterOutcomeRecord | Mapping[str, Any]],
) -> Dict[str, Any]:
    normalized: List[RouterOutcomeRecord] = []
    for item in outcomes:
        if isinstance(item, RouterOutcomeRecord):
            normalized.append(item)
        else:
            normalized.append(RouterOutcomeRecord.from_dict(item))
    outcome_counts: Counter[str] = Counter()
    by_route: Dict[str, Counter[str]] = defaultdict(Counter)
    by_method: Dict[str, Counter[str]] = defaultdict(Counter)
    for outcome in normalized:
        outcome_counts[outcome.outcome_label] += 1
        by_route[outcome.route][outcome.outcome_label] += 1
        by_method[outcome.router_method][outcome.outcome_label] += 1
    negative_rate_by_route = {
        route: round(counts.get(OUTCOME_NEGATIVE, 0) / max(1, sum(counts.values())), 4)
        for route, counts in by_route.items()
    }
    negative_rate_by_router_method = {
        method: round(counts.get(OUTCOME_NEGATIVE, 0) / max(1, sum(counts.values())), 4)
        for method, counts in by_method.items()
    }
    return {
        "outcome_counts": dict(outcome_counts),
        "negative_rate_by_route": negative_rate_by_route,
        "negative_rate_by_router_method": negative_rate_by_router_method,
    }


__all__ = [
    "DEFAULT_NEUTRAL_SAMPLE_RATE",
    "DEFAULT_TENANT_DAILY_REVIEW_CAP",
    "OUTCOME_NEGATIVE",
    "OUTCOME_NEUTRAL",
    "OUTCOME_POSITIVE",
    "RouterDecisionRecord",
    "RouterFeedbackLoop",
    "RouterOutcomeRecord",
    "RouterReviewSample",
    "REVIEW_STATUS_PENDING",
    "build_router_decision_id",
    "summarize_router_outcomes",
]
