from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

PHASE_STARTING = "starting"
PHASE_UPLOADING = "uploading_inputs"
PHASE_GRAPH_CATALOG = "inspecting_graph_catalog"
PHASE_SEARCHING = "searching_knowledge_base"
PHASE_SYNTHESIZING = "synthesizing_answer"
PHASE_READY = "answer_ready"
PHASE_FAILED = "failed"

STATUS_HEARTBEAT_SECONDS = 1.0

_PHASE_LABELS = {
    PHASE_STARTING: "Starting",
    PHASE_UPLOADING: "Uploading inputs",
    PHASE_GRAPH_CATALOG: "Inspecting graph catalog",
    PHASE_SEARCHING: "Searching knowledge base",
    PHASE_SYNTHESIZING: "Synthesizing answer",
    PHASE_READY: "Answer ready",
    PHASE_FAILED: "Failed",
}
_KNOWN_PHASES = set(_PHASE_LABELS)

_SUMMARY_PHASES = {PHASE_UPLOADING, PHASE_GRAPH_CATALOG, PHASE_SEARCHING, PHASE_SYNTHESIZING}
_TERMINAL_PHASES = {PHASE_READY, PHASE_FAILED}
_TOOL_EVENT_TYPES = {"tool_intent", "tool_call", "tool_result", "tool_error"}
_WORKER_EVENT_TYPES = {"worker_start", "worker_end"}

_PHASE_CHIP_LABELS = {
    PHASE_STARTING: "Thinking",
    PHASE_UPLOADING: "Uploads",
    PHASE_GRAPH_CATALOG: "Graph catalog",
    PHASE_SEARCHING: "Research",
    PHASE_SYNTHESIZING: "Writing",
    PHASE_READY: "Ready",
    PHASE_FAILED: "Failed",
}

_FRIENDLY_AGENT_LABELS = {
    "basic": "Basic Agent",
    "clarify": "Clarifier",
    "coordinator": "Coordinator",
    "data_analyst": "Data Analyst",
    "evaluator": "Evaluator",
    "finalizer": "Finalizer",
    "general": "General Agent",
    "general_agent": "General Agent",
    "graph_manager": "Graph Manager",
    "memory_maintainer": "Memory Maintainer",
    "parallel_planner": "Parallel Planner",
    "planner": "Planner",
    "planner_agent": "Planner",
    "rag_agent": "RAG Agent",
    "rag_synthesizer": "RAG Synthesizer",
    "rag_worker": "RAG Worker",
    "supervisor": "Supervisor",
    "utility": "Utility Agent",
    "utility_agent": "Utility Agent",
    "verifier": "Verifier",
}

_UPPERCASE_TOKENS = {"api", "kb", "llm", "rag", "ui"}

_UPLOAD_TOKENS = ("upload", "multipart")
_SYNTH_TOKENS = (
    "synthesiz",
    "grounding final response",
    "grounded answer ready",
    "final synthesis",
    "final response",
    "final answer",
    "verification",
    "verifier",
    "finalizer",
)
_GRAPH_TOKENS = (
    "graph catalog",
    "graph index",
    "graph indexes",
    "knowledge graph",
    "list_graph_indexes",
    "inspect_graph_index",
    "search_graph_index",
    "graph_manager",
)
_SEARCH_TOKENS = (
    "search",
    "retriev",
    "evidence",
    "reviewing candidate",
    "reading document",
    "expanding context",
    "planning search",
    "planning tasks",
    "dispatching workers",
    "hybrid search",
    "keyword search",
    "vector search",
    "worker",
    "handoff",
    "document",
    "extracting requirements",
    "requirements extraction",
    "tool",
)


def format_elapsed_ms(elapsed_ms: int) -> str:
    total_seconds = max(0, int(elapsed_ms / 1000))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def phase_label(phase: str) -> str:
    return _PHASE_LABELS.get(phase, str(phase or "").replace("_", " ").title())


def agent_from_progress(progress_event: Mapping[str, Any]) -> str:
    for key in ("agent", "active_agent", "selected_agent", "node"):
        agent = str(progress_event.get(key) or "").strip()
        if agent:
            return agent
    label = str(progress_event.get("label") or "").strip()
    for prefix in ("Routed to ", "Running "):
        if label.startswith(prefix):
            return label[len(prefix) :].strip()
    return ""


def infer_phase_from_progress(progress_event: Mapping[str, Any]) -> str:
    explicit_phase = str(progress_event.get("phase") or "").strip()
    if explicit_phase:
        return explicit_phase if explicit_phase in _KNOWN_PHASES else explicit_phase
    event_type = str(progress_event.get("type") or progress_event.get("event_type") or "").strip().lower()
    label = str(progress_event.get("label") or "").strip().lower()
    detail = str(progress_event.get("detail") or "").strip().lower()
    text = f"{event_type} {label} {detail}".strip()

    if not text:
        return ""
    if any(token in text for token in _UPLOAD_TOKENS):
        return PHASE_UPLOADING
    if any(token in text for token in _GRAPH_TOKENS):
        return PHASE_GRAPH_CATALOG
    if any(token in text for token in _SYNTH_TOKENS):
        return PHASE_SYNTHESIZING
    if event_type in {"worker_start", "worker_end", "decision_point", "doc_focus", "tool_intent", "tool_call", "tool_result", "tool_error"}:
        return PHASE_SEARCHING
    if any(token in text for token in _SEARCH_TOKENS):
        return PHASE_SEARCHING
    return ""


def _snapshot_signature(payload: Mapping[str, Any]) -> str:
    parts = [
        str(payload.get("status_id") or ""),
        str(payload.get("status_key") or ""),
        str(payload.get("description") or ""),
        str(payload.get("status") or ""),
        str(bool(payload.get("done"))),
        str(bool(payload.get("hidden"))),
        str(payload.get("elapsed_ms")),
        str(payload.get("delta_ms")),
        str(payload.get("phase_elapsed_ms")),
        str(payload.get("status_elapsed_ms")),
        str(payload.get("agent") or ""),
        str(payload.get("selected_agent") or ""),
        str(payload.get("phase") or ""),
        str(payload.get("phase_label") or ""),
        str(payload.get("label") or ""),
        str(payload.get("detail") or ""),
        str(payload.get("source_event_type") or ""),
        str(payload.get("job_id") or ""),
        str(payload.get("task_id") or ""),
        str(payload.get("why") or ""),
        str(payload.get("waiting_on") or ""),
        _agentic_status_signature(payload.get("agentic_status")),
    ]
    return "\u241f".join(parts)


def _agentic_status_signature(value: Any) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return str(value)


def _collapse_whitespace(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _truncate_text(value: str, *, limit: int = 140) -> str:
    text = _collapse_whitespace(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return max(0, int(numeric))


def _friendly_token(token: str) -> str:
    lowered = token.lower()
    if lowered in _UPPERCASE_TOKENS:
        return lowered.upper()
    if lowered == "id":
        return "ID"
    return lowered.title()


def _friendly_agent_label(agent: Any) -> str:
    normalized = _collapse_whitespace(agent)
    if not normalized:
        return ""
    lowered = normalized.casefold().replace(" ", "_")
    if lowered in _FRIENDLY_AGENT_LABELS:
        return _FRIENDLY_AGENT_LABELS[lowered]
    tokens = [token for token in re.split(r"[_\-\s]+", normalized) if token]
    if not tokens:
        return normalized
    return " ".join(_friendly_token(token) for token in tokens)


def _phase_chip_label(phase: Any) -> str:
    normalized = _collapse_whitespace(phase)
    if not normalized:
        return ""
    return _PHASE_CHIP_LABELS.get(normalized, phase_label(normalized))


def _agentic_is_terminal(payload: Mapping[str, Any]) -> bool:
    phase = _collapse_whitespace(payload.get("phase"))
    if phase in _TERMINAL_PHASES:
        return True
    status = _collapse_whitespace(payload.get("status")).lower()
    return bool(payload.get("done")) and status in {"complete", "error"}


def _agentic_is_clarification(payload: Mapping[str, Any]) -> bool:
    return str(payload.get("turn_outcome") or "").strip().lower() == "clarification_request"


def _agentic_state(payload: Mapping[str, Any]) -> str:
    phase = _collapse_whitespace(payload.get("phase"))
    status = _collapse_whitespace(payload.get("status")).lower()
    if status == "error" or phase == PHASE_FAILED:
        return "error"
    if status == "complete" or _agentic_is_terminal(payload):
        return "complete"
    return "active"


def _agentic_kind(payload: Mapping[str, Any], state: str) -> str:
    phase = _collapse_whitespace(payload.get("phase"))
    source_event_type = _collapse_whitespace(payload.get("source_event_type")).lower()
    if state == "error":
        return "failed"
    if _agentic_is_terminal(payload) and _agentic_is_clarification(payload):
        return "clarification"
    if _agentic_is_terminal(payload) and phase == PHASE_READY:
        return "ready"
    if source_event_type in _WORKER_EVENT_TYPES:
        return "worker"
    if source_event_type in {"route_decision", "agent_selected"}:
        return "routing"
    if phase == PHASE_SYNTHESIZING:
        return "synthesizing"
    if phase == PHASE_STARTING:
        return "thinking"
    return "research"


def _agentic_title(payload: Mapping[str, Any], state: str) -> str:
    phase = _collapse_whitespace(payload.get("phase"))
    source_event_type = _collapse_whitespace(payload.get("source_event_type")).lower()
    agent_label = _friendly_agent_label(payload.get("agent") or payload.get("selected_agent"))
    if state == "error":
        return "Run failed"
    if _agentic_is_terminal(payload) and _agentic_is_clarification(payload):
        return "Needs input"
    if _agentic_is_terminal(payload) and phase == PHASE_READY:
        return "Answer ready"
    if source_event_type == "route_decision":
        return "Routing request"
    if source_event_type == "agent_selected":
        return f"Running {agent_label}" if agent_label else "Running agent"
    if phase == PHASE_STARTING:
        return "Thinking"
    if phase == PHASE_SYNTHESIZING:
        return "Writing answer"
    if phase == PHASE_UPLOADING:
        return "Preparing files"
    if phase == PHASE_GRAPH_CATALOG:
        return "Inspecting graph catalog"
    if source_event_type in (_TOOL_EVENT_TYPES | _WORKER_EVENT_TYPES | {"doc_focus", "heartbeat"}):
        return "Researching evidence"
    if phase == PHASE_SEARCHING:
        return "Researching evidence"
    return phase_label(phase or PHASE_STARTING)


def _agentic_subtitle(payload: Mapping[str, Any], *, state: str, kind: str) -> str:
    phase = _collapse_whitespace(payload.get("phase"))
    source_event_type = _collapse_whitespace(payload.get("source_event_type")).lower()
    detail = _collapse_whitespace(payload.get("detail"))
    why = _collapse_whitespace(payload.get("why"))
    waiting_on = _collapse_whitespace(payload.get("waiting_on"))
    active_agent = _friendly_agent_label(payload.get("agent"))
    routed_agent = _friendly_agent_label(payload.get("selected_agent"))
    if state == "error":
        return detail or why or "The run stopped before the final answer was ready."
    if _agentic_is_terminal(payload) and _agentic_is_clarification(payload):
        clarification = payload.get("clarification") if isinstance(payload.get("clarification"), Mapping) else {}
        question = _collapse_whitespace(clarification.get("question")) if isinstance(clarification, Mapping) else ""
        return question or "A clarification is needed before continuing."
    if _agentic_is_terminal(payload) and phase == PHASE_READY:
        return "Grounded response is ready."
    if source_event_type == "route_decision":
        return why or "Choosing the best specialist for this request."
    if source_event_type == "agent_selected":
        if active_agent:
            return f"{active_agent} is handling this request."
        return "The selected specialist is now active."
    if waiting_on:
        return f"Waiting on {waiting_on}"
    if phase == PHASE_UPLOADING:
        return detail or "Sending files into the agent workspace."
    if phase == PHASE_GRAPH_CATALOG:
        return detail or "Reviewing available graph indexes."
    if kind == "synthesizing":
        return detail or "Grounding the final response."
    if source_event_type in _TOOL_EVENT_TYPES:
        return "Gathering and checking grounded evidence."
    if detail:
        return detail
    if why:
        return why
    if kind in {"research", "worker"}:
        if active_agent and routed_agent and active_agent != routed_agent:
            return f"{active_agent} is collecting evidence for {routed_agent}."
        if active_agent:
            return f"{active_agent} is working through grounded evidence."
        return "Gathering and checking grounded evidence."
    if phase == PHASE_STARTING:
        return "Preparing the agent workflow."
    return ""


def _agentic_chips(payload: Mapping[str, Any]) -> List[str]:
    chips: List[str] = []
    phase_chip = _phase_chip_label(payload.get("phase"))
    active_agent = _friendly_agent_label(payload.get("agent"))
    routed_agent = _friendly_agent_label(payload.get("selected_agent"))
    waiting_on = _collapse_whitespace(payload.get("waiting_on"))
    source_event_type = _collapse_whitespace(payload.get("source_event_type")).lower()
    for chip in (
        phase_chip,
        active_agent,
        f"Route: {routed_agent}" if routed_agent and routed_agent != active_agent else "",
        "Waiting" if waiting_on else "",
        "Live" if source_event_type == "heartbeat" else "",
    ):
        normalized = _truncate_text(chip, limit=48)
        if normalized and normalized not in chips:
            chips.append(normalized)
        if len(chips) >= 4:
            break
    return chips


def _build_agentic_timing_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    total_elapsed_ms = _as_int(payload.get("elapsed_ms")) or 0
    status_elapsed_ms = _as_int(payload.get("status_elapsed_ms"))
    phase_elapsed_ms = _as_int(payload.get("phase_elapsed_ms"))
    snapshot_timestamp_ms = _as_int(payload.get("timestamp")) or 0
    terminal = _agentic_is_terminal(payload)
    live = not terminal and _collapse_whitespace(payload.get("status")).lower() == "in_progress"
    timing_elapsed_ms = total_elapsed_ms if terminal else (
        status_elapsed_ms if status_elapsed_ms is not None else (
            phase_elapsed_ms if phase_elapsed_ms is not None else total_elapsed_ms
        )
    )
    return {
        "kind": "total" if terminal else "stage",
        "live": live,
        "elapsed_ms": timing_elapsed_ms,
        "status_elapsed_ms": status_elapsed_ms if status_elapsed_ms is not None else timing_elapsed_ms,
        "phase_elapsed_ms": phase_elapsed_ms if phase_elapsed_ms is not None else total_elapsed_ms,
        "total_elapsed_ms": total_elapsed_ms,
        "snapshot_timestamp_ms": snapshot_timestamp_ms,
    }


def _build_agentic_status_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    state = _agentic_state(payload)
    kind = _agentic_kind(payload, state)
    return {
        "version": 1,
        "state": state,
        "kind": kind,
        "title": _agentic_title(payload, state),
        "subtitle": _truncate_text(_agentic_subtitle(payload, state=state, kind=kind)),
        "chips": _agentic_chips(payload),
        "timing": _build_agentic_timing_payload(payload),
    }


@dataclass
class TurnStatusTracker:
    turn_started_at: float
    active_phase: str = PHASE_STARTING
    active_phase_started_at: float | None = None
    current_status_id: str = ""
    current_status_key: str = ""
    current_status_started_at: float | None = None
    current_agent: str = ""
    selected_agent: str = ""
    latest_label: str = ""
    latest_detail: str = ""
    latest_why: str = ""
    latest_waiting_on: str = ""
    latest_job_id: str = ""
    latest_task_id: str = ""
    latest_source_event_type: str = ""
    last_emit_at: float = 0.0
    last_emitted_elapsed_ms: int | None = None
    last_signature: str = ""
    next_status_seq: int = 1
    next_status_instance_id: int = 1

    def __post_init__(self) -> None:
        if self.active_phase_started_at is None:
            self.active_phase_started_at = self.turn_started_at
        if self.current_status_started_at is None:
            self.current_status_started_at = self.turn_started_at

    def seconds_until_next_heartbeat(self, now: float, *, interval_seconds: float = STATUS_HEARTBEAT_SECONDS) -> float | None:
        if self.active_phase in _TERMINAL_PHASES or self.active_phase_started_at is None:
            return None
        if not self.last_emit_at:
            return 0.0
        due_at = self.last_emit_at + float(interval_seconds)
        return max(0.0, due_at - now)

    def start_snapshots(self, now: float) -> List[Dict[str, Any]]:
        return self._in_progress_snapshots(now, source_event_type="turn_started", force=True)

    def progress_snapshots(self, progress_event: Mapping[str, Any], now: float) -> List[Dict[str, Any]]:
        raw = dict(progress_event or {})
        if not raw:
            return []

        event_type = str(raw.get("type") or raw.get("event_type") or "").strip().lower()
        label_present = "label" in raw
        detail_present = "detail" in raw
        why_present = "why" in raw
        waiting_present = "waiting_on" in raw
        job_present = "job_id" in raw
        task_present = "task_id" in raw
        selected_agent_present = "selected_agent" in raw

        label = str(raw.get("label") or "").strip() if label_present else None
        detail = str(raw.get("detail") or "").strip() if detail_present else None
        why = str(raw.get("why") or "").strip() if why_present else None
        waiting_on = str(raw.get("waiting_on") or "").strip() if waiting_present else None
        job_id = str(raw.get("job_id") or "").strip() if job_present else None
        task_id = str(raw.get("task_id") or "").strip() if task_present else None
        selected_agent = str(raw.get("selected_agent") or "").strip() if selected_agent_present else None
        agent = agent_from_progress(raw)

        self.latest_source_event_type = event_type or self.latest_source_event_type
        if label is not None:
            self.latest_label = label
        if detail is not None:
            self.latest_detail = detail
        if why is not None:
            self.latest_why = why
        if waiting_on is not None:
            self.latest_waiting_on = waiting_on
        if job_id is not None:
            self.latest_job_id = job_id
        if task_id is not None:
            self.latest_task_id = task_id
        if agent:
            self.current_agent = agent
        if selected_agent is not None:
            self.selected_agent = selected_agent
        elif event_type in {"route_decision", "agent_selected"} and agent:
            self.selected_agent = agent

        phase = infer_phase_from_progress(raw)
        if phase and phase != self.active_phase:
            return self.transition_phase(
                phase,
                now=now,
                source_event_type=event_type or "progress",
                label=label,
                detail=detail,
                why=why,
                waiting_on=waiting_on,
                agent=agent or None,
                selected_agent=selected_agent,
                job_id=job_id,
                task_id=task_id,
            )
        return self._in_progress_snapshots(now, source_event_type=event_type or "progress", force=False)

    def transition_phase(
        self,
        phase: str,
        *,
        now: float,
        source_event_type: str,
        label: str | None = None,
        detail: str | None = None,
        why: str | None = None,
        waiting_on: str | None = None,
        agent: str | None = None,
        selected_agent: str | None = None,
        job_id: str | None = None,
        task_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        snapshots: List[Dict[str, Any]] = []
        if phase == self.active_phase:
            if label is not None:
                self.latest_label = label
            if detail is not None:
                self.latest_detail = detail
            if why is not None:
                self.latest_why = why
            if waiting_on is not None:
                self.latest_waiting_on = waiting_on
            if job_id is not None:
                self.latest_job_id = job_id
            if task_id is not None:
                self.latest_task_id = task_id
            if agent is not None:
                self.current_agent = agent
            if selected_agent is not None:
                self.selected_agent = selected_agent
            self.latest_source_event_type = source_event_type or self.latest_source_event_type
            return self._in_progress_snapshots(now, source_event_type=source_event_type, force=True)

        if self.active_phase in _SUMMARY_PHASES and self.active_phase_started_at is not None:
            summary = self._phase_summary_snapshot(now, source_event_type=source_event_type)
            if summary is not None:
                snapshots.append(summary)

        if label is not None:
            self.latest_label = label
        if detail is not None:
            self.latest_detail = detail
        if why is not None:
            self.latest_why = why
        if waiting_on is not None:
            self.latest_waiting_on = waiting_on
        if job_id is not None:
            self.latest_job_id = job_id
        if task_id is not None:
            self.latest_task_id = task_id
        if agent is not None:
            self.current_agent = agent
        if selected_agent is not None:
            self.selected_agent = selected_agent
        self.latest_source_event_type = source_event_type or self.latest_source_event_type
        self.active_phase = phase
        self.active_phase_started_at = now
        self._reset_status_segment(now)
        current_phase_label = phase_label(phase)
        if label is None:
            self.latest_label = current_phase_label
        snapshots.extend(self._in_progress_snapshots(now, source_event_type=source_event_type, force=True))
        return snapshots

    def heartbeat_snapshot(self, now: float, *, force: bool = False) -> Dict[str, Any] | None:
        snapshots = self._in_progress_snapshots(now, source_event_type="heartbeat", force=force)
        return snapshots[0] if snapshots else None

    def completion_snapshots(self, now: float, *, metadata: Mapping[str, Any] | None = None) -> List[Dict[str, Any]]:
        metadata = dict(metadata or {})
        snapshots: List[Dict[str, Any]] = []
        if self.active_phase in _SUMMARY_PHASES and self.active_phase_started_at is not None:
            summary = self._phase_summary_snapshot(now, source_event_type="turn_completed")
            if summary is not None:
                snapshots.append(summary)
        self.active_phase = PHASE_READY
        self.active_phase_started_at = now
        self._reset_status_segment(now)
        payload = self._base_payload(
            now,
            status="complete",
            done=True,
            phase=PHASE_READY,
            phase_elapsed_ms=max(0, int((now - self.turn_started_at) * 1000)),
            delta_ms=self._next_delta_ms(now),
            source_event_type="turn_completed",
            label="Needs input" if _agentic_is_clarification(metadata) else phase_label(PHASE_READY),
            description=self._complete_description(now, metadata=metadata),
        )
        if metadata:
            payload.update(metadata)
            payload["agentic_status"] = _build_agentic_status_payload(payload)
        snapshot = self._maybe_emit(payload, now, force=True)
        if snapshot is not None:
            snapshots.append(snapshot)
        return snapshots

    def failure_snapshot(self, now: float) -> Dict[str, Any] | None:
        current_phase = self.active_phase
        phase_duration_ms = (
            max(0, int((now - self.active_phase_started_at) * 1000))
            if self.active_phase_started_at is not None
            else max(0, int((now - self.turn_started_at) * 1000))
        )
        description = self._failure_description(now, current_phase, phase_duration_ms)
        self.active_phase = PHASE_FAILED
        self.active_phase_started_at = now
        self._reset_status_segment(now)
        payload = self._base_payload(
            now,
            status="error",
            done=True,
            phase=PHASE_FAILED,
            phase_elapsed_ms=phase_duration_ms,
            delta_ms=phase_duration_ms,
            source_event_type="turn_failed",
            label=phase_label(PHASE_FAILED),
            description=description,
        )
        return self._maybe_emit(payload, now, force=True)

    def _in_progress_snapshots(self, now: float, *, source_event_type: str, force: bool) -> List[Dict[str, Any]]:
        key_changed = self._ensure_status_segment(now)
        phase_elapsed_ms = (
            max(0, int((now - self.active_phase_started_at) * 1000))
            if self.active_phase_started_at is not None
            else max(0, int((now - self.turn_started_at) * 1000))
        )
        payload = self._base_payload(
            now,
            status="in_progress",
            done=False,
            phase=self.active_phase,
            phase_elapsed_ms=phase_elapsed_ms,
            delta_ms=self._next_delta_ms(now),
            source_event_type=source_event_type,
            description=self._active_description(now),
        )
        snapshot = self._maybe_emit(payload, now, force=force or key_changed)
        return [snapshot] if snapshot is not None else []

    def _phase_summary_snapshot(self, now: float, *, source_event_type: str) -> Dict[str, Any] | None:
        phase_duration_ms = (
            max(0, int((now - self.active_phase_started_at) * 1000))
            if self.active_phase_started_at is not None
            else 0
        )
        payload = self._base_payload(
            now,
            status="complete",
            done=False,
            phase=self.active_phase,
            phase_elapsed_ms=phase_duration_ms,
            delta_ms=phase_duration_ms,
            source_event_type=source_event_type,
            label=phase_label(self.active_phase),
            description=self._summary_description(now, phase_duration_ms),
        )
        return self._maybe_emit(payload, now, force=True)

    def _base_payload(
        self,
        now: float,
        *,
        status: str,
        done: bool,
        phase: str,
        phase_elapsed_ms: int,
        delta_ms: int | None,
        source_event_type: str,
        description: str,
        label: str | None = None,
    ) -> Dict[str, Any]:
        total_elapsed_ms = max(0, int((now - self.turn_started_at) * 1000))
        self._ensure_status_segment(now)
        status_elapsed_ms = (
            max(0, int((now - self.current_status_started_at) * 1000))
            if self.current_status_started_at is not None
            else total_elapsed_ms
        )
        payload = {
            "status_id": self.current_status_id,
            "status_key": self.current_status_key,
            "description": description,
            "status": status,
            "done": done,
            "hidden": False,
            "elapsed_ms": total_elapsed_ms,
            "delta_ms": delta_ms,
            "status_elapsed_ms": status_elapsed_ms,
            "agent": self.current_agent,
            "selected_agent": self.selected_agent,
            "phase": phase,
            "phase_label": phase_label(phase),
            "phase_elapsed_ms": phase_elapsed_ms,
            "source_event_type": source_event_type,
            "label": label if label is not None else (self.latest_label or phase_label(phase)),
            "detail": self.latest_detail,
            "job_id": self.latest_job_id,
            "task_id": self.latest_task_id,
            "why": self.latest_why,
            "waiting_on": self.latest_waiting_on,
            "timestamp": int(time.time() * 1000),
        }
        payload["agentic_status"] = _build_agentic_status_payload(payload)
        return payload

    def _next_delta_ms(self, now: float) -> int | None:
        total_elapsed_ms = max(0, int((now - self.turn_started_at) * 1000))
        if self.last_emitted_elapsed_ms is None:
            return None
        return max(0, total_elapsed_ms - self.last_emitted_elapsed_ms)

    def _maybe_emit(self, payload: Dict[str, Any], now: float, *, force: bool) -> Dict[str, Any] | None:
        signature = _snapshot_signature(payload)
        if not force and signature == self.last_signature:
            return None
        payload["status_seq"] = self.next_status_seq
        self.next_status_seq += 1
        self.last_signature = signature
        self.last_emit_at = now
        self.last_emitted_elapsed_ms = int(payload.get("elapsed_ms") or 0)
        return payload

    def _reset_status_segment(self, now: float) -> None:
        self.current_status_id = ""
        self.current_status_key = ""
        self.current_status_started_at = now

    def _status_key(self) -> str:
        display_agent = self._display_agent()
        return "\u241f".join(
            [
                str(self.active_phase or ""),
                display_agent,
                "done" if self.active_phase in _TERMINAL_PHASES else "active",
            ]
        )

    def _ensure_status_segment(self, now: float) -> bool:
        key = self._status_key()
        if self.current_status_id and self.current_status_key == key:
            if self.current_status_started_at is None:
                self.current_status_started_at = now
            return False
        self.current_status_key = key
        self.current_status_id = f"status-{self.next_status_instance_id}"
        self.next_status_instance_id += 1
        self.current_status_started_at = now
        return True

    def _display_agent(self) -> str:
        return self.current_agent or self.selected_agent

    def _active_description(self, now: float) -> str:
        total_elapsed_ms = max(0, int((now - self.turn_started_at) * 1000))
        parts = [phase_label(self.active_phase)]
        display_agent = self._display_agent()
        if display_agent:
            parts.append(display_agent)
        parts.append(f"{format_elapsed_ms(total_elapsed_ms)} elapsed")
        return " • ".join(parts)

    def _summary_description(self, now: float, phase_duration_ms: int) -> str:
        total_elapsed_ms = max(0, int((now - self.turn_started_at) * 1000))
        parts = [phase_label(self.active_phase)]
        display_agent = self._display_agent()
        if display_agent:
            parts.append(display_agent)
        parts.append(
            f"+{format_elapsed_ms(phase_duration_ms)} (total {format_elapsed_ms(total_elapsed_ms)})"
        )
        return " • ".join(parts)

    def _complete_description(self, now: float, *, metadata: Mapping[str, Any] | None = None) -> str:
        total_elapsed_ms = max(0, int((now - self.turn_started_at) * 1000))
        metadata = dict(metadata or {})
        if _agentic_is_clarification(metadata):
            parts = ["Needs input"]
        else:
            parts = [phase_label(PHASE_READY)]
        display_agent = self._display_agent()
        if display_agent:
            parts.append(display_agent)
        if _agentic_is_clarification(metadata):
            parts.append("Waiting for clarification")
        else:
            parts.append(f"Completed in {format_elapsed_ms(total_elapsed_ms)}")
        return " • ".join(parts)

    def _failure_description(self, now: float, current_phase: str, phase_duration_ms: int) -> str:
        total_elapsed_ms = max(0, int((now - self.turn_started_at) * 1000))
        display_agent = self._display_agent()
        if current_phase not in _SUMMARY_PHASES or self.active_phase_started_at is None:
            parts = [phase_label(PHASE_FAILED)]
            if display_agent:
                parts.append(display_agent)
            parts.append(f"after {format_elapsed_ms(total_elapsed_ms)}")
            return " • ".join(parts)
        parts = [f"{phase_label(PHASE_FAILED)} during {phase_label(current_phase)}"]
        if display_agent:
            parts.append(display_agent)
        parts.append(f"+{format_elapsed_ms(phase_duration_ms)} (total {format_elapsed_ms(total_elapsed_ms)})")
        return " • ".join(parts)


__all__ = [
    "PHASE_FAILED",
    "PHASE_READY",
    "PHASE_SEARCHING",
    "PHASE_STARTING",
    "PHASE_SYNTHESIZING",
    "PHASE_UPLOADING",
    "STATUS_HEARTBEAT_SECONDS",
    "TurnStatusTracker",
    "agent_from_progress",
    "format_elapsed_ms",
    "infer_phase_from_progress",
    "phase_label",
]
