import asyncio
import contextlib
import inspect
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class UploadHandoffError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str = "",
        detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = str(error_code or "").strip()
        self.detail = detail


class BackendCompletionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str = "",
        detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = str(error_code or "").strip()
        self.detail = detail


def _compact_response_text(text: str, *, limit: int = 500) -> str:
    compact = " ".join(str(text or "").strip().split())
    return compact[:limit].rstrip()


def _response_status_code(response: Any) -> int:
    try:
        return int(getattr(response, "status_code", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _backend_failure_payload(response: Any) -> dict[str, Any]:
    status_code = getattr(response, "status_code", None)
    raw_text = str(getattr(response, "text", "") or "")
    parsed: Any = {}
    try:
        parsed = response.json()
    except Exception:
        try:
            parsed = json.loads(raw_text) if raw_text.strip() else {}
        except Exception:
            parsed = {}
    if isinstance(parsed, dict) and "detail" not in parsed:
        payload = dict(parsed)
    else:
        detail = parsed.get("detail") if isinstance(parsed, dict) else parsed
        if isinstance(detail, dict):
            payload = dict(detail)
        else:
            payload = {"detail": str(detail or raw_text or "").strip()}
    if not isinstance(payload, dict):
        payload = {"detail": str(detail or raw_text or "").strip()}
    if status_code is not None:
        payload["status_code"] = int(status_code)
    return payload


def _runtime_registry_failure_message(payload: dict[str, Any], *, prefix: str) -> str:
    tools: list[str] = []
    for item in list(payload.get("missing_tools") or []):
        if isinstance(item, dict):
            tool_name = str(item.get("tool") or "").strip()
            if tool_name and tool_name not in tools:
                tools.append(tool_name)
    agents = [
        str(item).strip()
        for item in list(payload.get("affected_agents") or [])
        if str(item).strip()
    ]
    tool_text = ", ".join(tools) if tools else "one or more configured tools"
    agent_text = ", ".join(agents) if agents else "one or more agents"
    remediation = str(payload.get("remediation") or "").strip()
    return (
        f"{prefix} because the backend runtime registry is invalid: "
        f"{agent_text} references missing tool(s) {tool_text}."
        + (f" {remediation}" if remediation else "")
    )


def _backend_upload_failure_payload(response: Any) -> dict[str, Any]:
    return _backend_failure_payload(response)


def _backend_upload_failure_message(response: Any) -> str:
    payload = _backend_upload_failure_payload(response)
    status_code = int(payload.get("status_code") or _response_status_code(response))
    error_code = str(payload.get("error_code") or "").strip()
    if error_code == "runtime_registry_invalid":
        return _runtime_registry_failure_message(payload, prefix="File upload handoff failed")
    detail = _compact_response_text(
        str(payload.get("detail") or payload.get("message") or getattr(response, "text", "") or "")
    )
    if status_code == 401 or status_code == 403:
        return "File upload handoff failed because the backend rejected the upload credentials or permissions."
    if status_code == 413:
        return "File upload handoff failed because the backend rejected the file as too large."
    if status_code == 422:
        return f"File upload handoff failed because the backend rejected the upload request: {detail or 'validation failed'}."
    if status_code >= 500:
        return f"File upload handoff failed because the backend upload service is unavailable: {detail or f'HTTP {status_code}'}."
    fallback_status = f"HTTP {status_code}" if status_code else "HTTP unknown"
    return f"File upload handoff failed before agent execution while sending files to the agent workspace: {detail or fallback_status}."


def _backend_completion_failure_message(response: Any) -> str:
    payload = _backend_failure_payload(response)
    status_code = int(payload.get("status_code") or _response_status_code(response))
    error_code = str(payload.get("error_code") or "").strip()
    if error_code == "runtime_registry_invalid":
        return _runtime_registry_failure_message(payload, prefix="Agent request failed")
    detail = _compact_response_text(
        str(payload.get("detail") or payload.get("message") or getattr(response, "text", "") or "")
    )
    if status_code == 401 or status_code == 403:
        return "Agent request failed because the backend rejected the gateway credentials or permissions."
    if status_code == 413:
        return "Agent request failed because the backend rejected the request as too large."
    if status_code == 422:
        return f"Agent request failed because the backend rejected the chat request: {detail or 'validation failed'}."
    if status_code >= 500:
        return f"Agent request failed because the backend service is unavailable: {detail or f'HTTP {status_code}'}."
    fallback_status = f"HTTP {status_code}" if status_code else "HTTP unknown"
    return f"Agent request failed before completion: {detail or fallback_status}."


async def _raise_for_backend_completion_status(response: Any) -> None:
    status_code = _response_status_code(response)
    if status_code >= 400:
        aread = getattr(response, "aread", None)
        if callable(aread):
            with contextlib.suppress(Exception):
                await aread()
        payload = _backend_failure_payload(response)
        raise BackendCompletionError(
            _backend_completion_failure_message(response),
            status_code=status_code,
            error_code=str(payload.get("error_code") or "").strip(),
            detail=payload,
        )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        error_response = exc.response
        payload = _backend_failure_payload(error_response)
        raise BackendCompletionError(
            _backend_completion_failure_message(error_response),
            status_code=_response_status_code(error_response),
            error_code=str(payload.get("error_code") or "").strip(),
            detail=payload,
        ) from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


_STRUCTURED_SECTION_PREFIX_RE = re.compile(
    r"^\s*(?:goal|context|deliverable|objective|task|request|constraints?|output)\s*:\s*",
    re.IGNORECASE,
)
_OPENWEBUI_CONTEXT_BLOCK_RE = re.compile(r"<context\b[^>]*>.*?</context>", re.IGNORECASE | re.DOTALL)
_OPENWEBUI_QUERY_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?:#{1,6}\s*)?(?:user\s+query|user\s+question|query|question)\s*:\s*",
    re.IGNORECASE,
)
_OPENWEBUI_SOURCE_SECTION_RE = re.compile(
    r"(?ims)(?:^|\n)\s*(?:#{1,6}\s*)?(?:sources?|citations?|references?)\s*:\s*\n(?:\s*(?:[-*]|\d+[.)]|\[[^\]]+\]).*\n?)+"
)
_OPENWEBUI_FILE_CONTEXT_RE = re.compile(
    r"(?ims)(?:^|\n)\s*(?:#{1,6}\s*)?(?:uploaded|attached|provided)?\s*(?:file|document)\s+context\s*:\s*\n.*?(?=\n\s*(?:#{1,6}\s*)?(?:user\s+(?:query|question)|query|question)\s*:|\Z)"
)
_OPENWEBUI_ATTACHMENT_LINE_RE = re.compile(
    r"(?im)^\s*(?:\[(?:source|file|attachment|citation)[^\]]*\]|(?:source|file|attachment|citation)\s*:\s+\S.*)\s*$"
)
_OPENWEBUI_OUTPUT_SECTION_RE = re.compile(
    r"(?ims)(?:^|\n)\s*#{1,6}\s*output\s*:?\s*(?P<body>.*)$"
)
_OPENWEBUI_WRAPPER_HINT_RE = re.compile(
    r"(?is)#{1,6}\s*task\s*:.*?(?:provided\s+context|user\s+query|chat\s+history)"
)
_OPENWEBUI_WRAPPER_HEADING_RE = re.compile(
    r"(?im)^\s*#{1,6}\s*(?:task|guidelines?|output|chat\s+history|context)\s*:?\s*$"
)
_OPENWEBUI_CHAT_HISTORY_USER_RE = re.compile(r"(?im)^\s*USER\s*:\s*(.+?)\s*$")
_OPENWEBUI_INSTRUCTION_LINE_RE = re.compile(
    r"(?i)^\s*(?:[-*]\s*)?(?:"
    r"provide\s+(?:a\s+)?(?:clear|concise|direct)|"
    r"respond\s+to\s+the\s+user|"
    r"use\s+the\s+provided\s+context|"
    r"if\s+(?:you\s+)?(?:don'?t|do\s+not)\s+know|"
    r"if\s+the\s+answer\s+(?:isn'?t|is\s+not)\s+present|"
    r"do\s+not\s+(?:make|invent|fabricate)|"
    r"strictly\s+return|"
    r"return\s+only|"
    r"only\s+(?:include|use|return)|"
    r"cite\s+|"
    r"include\s+citations?|"
    r"answer\s+the\s+question"
    r")\b"
)

try:
    from agentic_chatbot_next.api.status_tracker import (
        PHASE_FAILED,
        PHASE_GRAPH_CATALOG,
        PHASE_READY,
        PHASE_SEARCHING,
        PHASE_STARTING,
        PHASE_SYNTHESIZING,
        PHASE_UPLOADING,
        STATUS_HEARTBEAT_SECONDS,
        TurnStatusTracker,
    )
except Exception:
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
    _SUMMARY_PHASES = {PHASE_UPLOADING, PHASE_GRAPH_CATALOG, PHASE_SEARCHING, PHASE_SYNTHESIZING}
    _TERMINAL_PHASES = {PHASE_READY, PHASE_FAILED}
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

    def _phase_label(phase: str) -> str:
        return _PHASE_LABELS.get(phase, str(phase or "").replace("_", " ").title())

    def _format_elapsed_ms(elapsed_ms: int) -> str:
        total_seconds = max(0, int(elapsed_ms / 1000))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

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

    def _agent_from_progress(progress_event: Mapping[str, Any]) -> str:
        for key in ("agent", "active_agent", "selected_agent", "node"):
            agent = str(progress_event.get(key) or "").strip()
            if agent:
                return agent
        label = str(progress_event.get("label") or "").strip()
        for prefix in ("Routed to ", "Running "):
            if label.startswith(prefix):
                return label[len(prefix) :].strip()
        return ""

    def _infer_phase_from_progress(progress_event: Mapping[str, Any]) -> str:
        explicit_phase = str(progress_event.get("phase") or "").strip()
        if explicit_phase:
            return explicit_phase
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
        if event_type in {"worker_start", "worker_end", "doc_focus", "tool_intent", "tool_call", "tool_result", "tool_error"}:
            return PHASE_SEARCHING
        if any(token in text for token in _SEARCH_TOKENS):
            return PHASE_SEARCHING
        return ""

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
            label = str(raw.get("label") or "").strip() if "label" in raw else None
            detail = str(raw.get("detail") or "").strip() if "detail" in raw else None
            why = str(raw.get("why") or "").strip() if "why" in raw else None
            waiting_on = str(raw.get("waiting_on") or "").strip() if "waiting_on" in raw else None
            job_id = str(raw.get("job_id") or "").strip() if "job_id" in raw else None
            task_id = str(raw.get("task_id") or "").strip() if "task_id" in raw else None
            selected_agent = str(raw.get("selected_agent") or "").strip() if "selected_agent" in raw else None
            agent = _agent_from_progress(raw)
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
            phase = _infer_phase_from_progress(raw)
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
            if label is None:
                self.latest_label = _phase_label(phase)
            snapshots.extend(self._in_progress_snapshots(now, source_event_type=source_event_type, force=True))
            return snapshots

        def heartbeat_snapshot(self, now: float, *, force: bool = False) -> Dict[str, Any] | None:
            snapshots = self._in_progress_snapshots(now, source_event_type="heartbeat", force=force)
            return snapshots[0] if snapshots else None

        def completion_snapshots(self, now: float) -> List[Dict[str, Any]]:
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
                label=_phase_label(PHASE_READY),
                description=self._complete_description(now),
            )
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
                label=_phase_label(PHASE_FAILED),
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
                label=_phase_label(self.active_phase),
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
                "phase_label": _phase_label(phase),
                "phase_elapsed_ms": phase_elapsed_ms,
                "source_event_type": source_event_type,
                "label": label if label is not None else (self.latest_label or _phase_label(phase)),
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
            parts = [_phase_label(self.active_phase)]
            display_agent = self._display_agent()
            if display_agent:
                parts.append(display_agent)
            parts.append(f"{_format_elapsed_ms(total_elapsed_ms)} elapsed")
            return " • ".join(parts)

        def _summary_description(self, now: float, phase_duration_ms: int) -> str:
            total_elapsed_ms = max(0, int((now - self.turn_started_at) * 1000))
            parts = [_phase_label(self.active_phase)]
            display_agent = self._display_agent()
            if display_agent:
                parts.append(display_agent)
            parts.append(f"+{_format_elapsed_ms(phase_duration_ms)} (total {_format_elapsed_ms(total_elapsed_ms)})")
            return " • ".join(parts)

        def _complete_description(self, now: float) -> str:
            total_elapsed_ms = max(0, int((now - self.turn_started_at) * 1000))
            parts = [_phase_label(PHASE_READY)]
            display_agent = self._display_agent()
            if display_agent:
                parts.append(display_agent)
            parts.append(f"Completed in {_format_elapsed_ms(total_elapsed_ms)}")
            return " • ".join(parts)

        def _failure_description(self, now: float, current_phase: str, phase_duration_ms: int) -> str:
            total_elapsed_ms = max(0, int((now - self.turn_started_at) * 1000))
            display_agent = self._display_agent()
            if current_phase not in _SUMMARY_PHASES or self.active_phase_started_at is None:
                parts = [_phase_label(PHASE_FAILED)]
                if display_agent:
                    parts.append(display_agent)
                parts.append(f"after {_format_elapsed_ms(total_elapsed_ms)}")
                return " • ".join(parts)
            parts = [f"{_phase_label(PHASE_FAILED)} during {_phase_label(current_phase)}"]
            if display_agent:
                parts.append(display_agent)
            parts.append(f"+{_format_elapsed_ms(phase_duration_ms)} (total {_format_elapsed_ms(total_elapsed_ms)})")
            return " • ".join(parts)


_FOLLOW_UP_PATTERNS = (
    re.compile(r"suggest\s+3-5\s+relevant\s+follow-?up\s+questions", re.IGNORECASE),
    re.compile(r"suggest\s+follow-?up\s+questions", re.IGNORECASE),
    re.compile(r"follow-?up\s+questions?\s+(?:the\s+user\s+)?could\s+ask", re.IGNORECASE),
)
_TITLE_PATTERNS = (
    re.compile(r"generate\s+a\s+concise.*title", re.IGNORECASE),
    re.compile(r"\b3-5\s+word\s+title\b", re.IGNORECASE),
)
_TAG_PATTERNS = (
    re.compile(r"generate\s+1-3\s+broad\s+tags", re.IGNORECASE),
    re.compile(r"\bbroad\s+tags\b", re.IGNORECASE),
)
_SEARCH_QUERY_PATTERNS = (
    re.compile(
        r"analyze\s+the\s+chat\s+history\s+to\s+determine\s+the\s+necessity\s+of\s+generating\s+search\s+queries",
        re.IGNORECASE,
    ),
    re.compile(
        r"prioritize\s+generating\s+1-3\s+broad\s+and\s+relevant\s+search\s+queries",
        re.IGNORECASE,
    ),
    re.compile(r"respond\s+\*+exclusively\*+\s+with\s+a\s+json\s+object", re.IGNORECASE),
    re.compile(r"\"queries\"\s*:\s*\[", re.IGNORECASE),
)
_TERMINAL_PHASES = {PHASE_READY, PHASE_FAILED}
_PHASE_RANKS = {
    PHASE_STARTING: 0,
    PHASE_UPLOADING: 1,
    PHASE_GRAPH_CATALOG: 2,
    PHASE_SEARCHING: 3,
    PHASE_SYNTHESIZING: 4,
    PHASE_READY: 5,
    PHASE_FAILED: 5,
}

_DISPLAY_PHASE_LABELS = {
    PHASE_STARTING: "Starting",
    PHASE_UPLOADING: "Uploading inputs",
    PHASE_GRAPH_CATALOG: "Inspecting graph catalog",
    PHASE_SEARCHING: "Searching knowledge base",
    PHASE_SYNTHESIZING: "Synthesizing answer",
    PHASE_READY: "Answer ready",
    PHASE_FAILED: "Failed",
}

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
_TOOL_EVENT_TYPES = {"tool_intent", "tool_call", "tool_result", "tool_error"}
_WORKER_EVENT_TYPES = {"worker_start", "worker_end"}


def _display_phase_label(phase: Any) -> str:
    normalized = str(phase or "").strip()
    if not normalized:
        return ""
    return _DISPLAY_PHASE_LABELS.get(normalized, normalized.replace("_", " ").title())


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
    return _PHASE_CHIP_LABELS.get(normalized, _display_phase_label(normalized))


def _agentic_is_terminal(payload: Mapping[str, Any]) -> bool:
    phase = _collapse_whitespace(payload.get("phase"))
    if phase in _TERMINAL_PHASES:
        return True
    status = _collapse_whitespace(payload.get("status")).lower()
    return bool(payload.get("done")) and status in {"complete", "error"}


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
    return _display_phase_label(phase or PHASE_STARTING)


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


def _ensure_agentic_status_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload or {})
    if not normalized:
        return normalized
    structured = normalized.get("agentic_status")
    if isinstance(structured, dict) and structured.get("version") == 1:
        return normalized
    normalized["agentic_status"] = _build_agentic_status_payload(normalized)
    return normalized


def _status_signature(payload: Mapping[str, Any]) -> str:
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
        _agentic_status_signature(payload.get("agentic_agent_activity")),
        _agentic_status_signature(payload.get("agentic_parallel_group")),
        _agentic_status_signature(payload.get("agentic_tool_call")),
    ]
    return "\u241f".join(parts)


@dataclass
class _TurnStatusState:
    tracker: TurnStatusTracker
    emit_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    backend_status_seen: bool = False
    last_forwarded_status_signature: str = ""
    last_forwarded_status_seq: int = -1


class Pipe:
    class Valves(BaseModel):
        AGENT_BASE_URL: str = Field(
            default_factory=lambda: os.getenv("AGENT_BASE_URL", "http://app:8000/v1")
        )
        AGENT_API_KEY: str = Field(default_factory=lambda: os.getenv("AGENT_API_KEY", ""))
        AGENT_MODEL_ID: str = Field(default_factory=lambda: os.getenv("AGENT_MODEL_ID", "enterprise-agent"))
        KB_COLLECTION_ID: str = Field(default_factory=lambda: os.getenv("KB_COLLECTION_ID", "default"))
        PUBLIC_AGENT_API_BASE_URL: str = Field(
            default_factory=lambda: os.getenv("PUBLIC_AGENT_API_BASE_URL", "http://localhost:18000")
        )
        REQUEST_TIMEOUT_SECONDS: int = Field(
            default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT_SECONDS", "600") or "600")
        )
        COLLECTION_PREFIX: str = Field(default_factory=lambda: os.getenv("COLLECTION_PREFIX", "owui-chat-"))
        OPENWEBUI_THIN_MODE: bool = Field(default_factory=lambda: _env_bool("OPENWEBUI_THIN_MODE", True))
        OPENWEBUI_ENABLE_HELPERS: bool = Field(default_factory=lambda: _env_bool("OPENWEBUI_ENABLE_HELPERS", False))
        OPENWEBUI_ALLOW_FILE_BYTE_TRANSPORT: bool = Field(
            default_factory=lambda: _env_bool("OPENWEBUI_ALLOW_FILE_BYTE_TRANSPORT", True)
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.name = ""

    def pipes(self) -> list[dict[str, str]]:
        return [{"id": "owui_enterprise_agent", "name": "Enterprise Agent"}]

    def _now(self) -> float:
        return time.monotonic()

    def _agent_headers(self, *, chat_id: str, user_id: str, message_id: str, user_email: str = "") -> dict[str, str]:
        headers = {
            "X-Conversation-ID": chat_id,
            "X-User-ID": user_id,
            "X-Request-ID": message_id,
        }
        if str(user_email or "").strip():
            headers["X-User-Email"] = str(user_email or "").strip()
        api_key = str(self.valves.AGENT_API_KEY or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _openwebui_auth_headers(self, request: Any) -> dict[str, str]:
        authorization = str(getattr(getattr(request, "headers", {}), "get", lambda *_: "")("authorization") or "").strip()
        return {"Authorization": authorization} if authorization else {}

    def _chat_scope(self, user: dict[str, Any], metadata: dict[str, Any]) -> tuple[str, str, str, str]:
        chat_id = str(metadata.get("chat_id") or metadata.get("session_id") or "openwebui-chat").strip()
        user_id = (
            str(user.get("id") or "").strip()
            or str(user.get("email") or "").strip()
            or str(user.get("name") or "").strip()
            or "openwebui-user"
        )
        user_email = str(user.get("email") or metadata.get("user_email") or "").strip().casefold()
        message_id = str(metadata.get("message_id") or "").strip() or f"{chat_id}-message"
        return chat_id, user_id, user_email, message_id

    def _normalize_files(self, files: list[Any]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        for item in files:
            if not isinstance(item, dict):
                continue
            file_id = (
                str(item.get("id") or "").strip()
                or str(item.get("file_id") or "").strip()
                or str((item.get("file") or {}).get("id") or "").strip()
            )
            if not file_id or file_id in seen_ids:
                continue
            seen_ids.add(file_id)
            filename = (
                str(item.get("filename") or "").strip()
                or str(item.get("name") or "").strip()
                or str((item.get("file") or {}).get("filename") or "").strip()
                or f"{file_id}.bin"
            )
            normalized.append({"id": file_id, "filename": filename})
        return normalized

    async def _download_openwebui_file(
        self,
        client: httpx.AsyncClient,
        request: Any,
        file_id: str,
    ) -> httpx.Response:
        if request is None:
            raise RuntimeError("Open WebUI request context is required to fetch uploaded files.")
        url = f"http://127.0.0.1:8080/api/v1/files/{file_id}/content"
        try:
            response = await client.get(
                url,
                headers=self._openwebui_auth_headers(request),
                cookies=dict(getattr(request, "cookies", {}) or {}),
            )
            response.raise_for_status()
            return response
        except Exception:
            logger.exception("file_download_failed file_id=%s", file_id)
            raise

    def _upload_files_to_agent_sync(
        self,
        *,
        upload_items: list[dict[str, Any]],
        chat_id: str,
        user_id: str,
        user_email: str = "",
        message_id: str,
        collection_id: str,
    ) -> dict[str, Any]:
        upload_results: list[dict[str, Any]] = []
        active_uploaded_doc_ids: list[str] = []
        uploaded_doc_ids: list[str] = []

        def remember_doc_ids(values: Any, target: list[str], seen: set[str]) -> None:
            for value in list(values or []):
                text = str(value or "").strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                target.append(text)

        try:
            # httpx 0.28.x builds a sync multipart request stream for files=...
            # even on AsyncClient, so we isolate upload handoff onto a sync
            # client inside asyncio.to_thread().
            timeout = httpx.Timeout(self.valves.REQUEST_TIMEOUT_SECONDS)
            seen_active_doc_ids: set[str] = set()
            seen_uploaded_doc_ids: set[str] = set()
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                for item in upload_items:
                    headers = self._agent_headers(
                        chat_id=chat_id,
                        user_id=user_id,
                        user_email=user_email,
                        message_id=message_id,
                    )
                    source_id = str(item.get("source_id") or "").strip()
                    if source_id:
                        headers["X-Upload-Source-Ids"] = json.dumps([source_id])
                    response = client.post(
                        f"{self.valves.AGENT_BASE_URL.rstrip('/')}/upload",
                        params={"source_type": "upload", "collection_id": collection_id},
                        headers=headers,
                        files={
                            "files": (
                                str(item.get("filename") or "upload.bin"),
                                bytes(item.get("file_bytes") or b""),
                                str(item.get("content_type") or "application/octet-stream"),
                            )
                        },
                    )
                    is_error = bool(getattr(response, "is_error", False))
                    status_code = getattr(response, "status_code", None)
                    if status_code is not None and int(status_code) >= 400:
                        is_error = True
                    if is_error:
                        body_preview = str(getattr(response, "text", "") or "").strip().replace("\n", " ")[:200]
                        failure_payload = _backend_upload_failure_payload(response)
                        logger.error(
                            "backend_upload_failed_response chat_id=%s collection_id=%s filename=%s status_code=%s body=%s",
                            chat_id,
                            collection_id,
                            str(item.get("filename") or ""),
                            status_code if status_code is not None else "unknown",
                            body_preview or "<empty>",
                        )
                        raise UploadHandoffError(
                            _backend_upload_failure_message(response),
                            status_code=int(status_code) if status_code is not None else None,
                            error_code=str(failure_payload.get("error_code") or ""),
                            detail=failure_payload,
                        )
                    response.raise_for_status()
                    try:
                        parsed = response.json()
                    except Exception:
                        parsed = {}
                    if isinstance(parsed, dict):
                        upload_results.append(parsed)
                        remember_doc_ids(parsed.get("active_uploaded_doc_ids"), active_uploaded_doc_ids, seen_active_doc_ids)
                        remember_doc_ids(parsed.get("uploaded_doc_ids"), active_uploaded_doc_ids, seen_active_doc_ids)
                        remember_doc_ids(parsed.get("doc_ids"), active_uploaded_doc_ids, seen_active_doc_ids)
                        remember_doc_ids(parsed.get("doc_ids"), uploaded_doc_ids, seen_uploaded_doc_ids)
        except UploadHandoffError:
            logger.exception(
                "backend_upload_failed chat_id=%s collection_id=%s filenames=%s",
                chat_id,
                collection_id,
                [str(item.get("filename") or "") for item in upload_items],
            )
            raise
        except httpx.HTTPError as exc:
            logger.exception(
                "backend_upload_failed chat_id=%s collection_id=%s filenames=%s",
                chat_id,
                collection_id,
                [str(item.get("filename") or "") for item in upload_items],
            )
            raise UploadHandoffError(
                "File upload handoff failed because the backend upload service could not be reached.",
                detail=str(exc),
            ) from exc
        except Exception:
            logger.exception(
                "backend_upload_failed chat_id=%s collection_id=%s filenames=%s",
                chat_id,
                collection_id,
                [str(item.get("filename") or "") for item in upload_items],
            )
            raise
        return {
            "object": "openwebui.upload_handoff",
            "collection_id": collection_id,
            "active_uploaded_doc_ids": active_uploaded_doc_ids,
            "doc_ids": uploaded_doc_ids,
            "upload_results": upload_results,
        }

    def _render_artifacts(self, artifacts: list[dict[str, Any]]) -> str:
        base_url = str(self.valves.PUBLIC_AGENT_API_BASE_URL or "").rstrip("/")
        lines: list[str] = []
        for item in artifacts:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("filename") or "download").strip() or "download"
            href = str(item.get("download_url") or "").strip()
            if not href:
                continue
            if href.startswith("/"):
                href = f"{base_url}{href}" if base_url else href
            lines.append(f"- [{label}]({href})")
        if not lines:
            return ""
        return "Returned files:\n" + "\n".join(lines)

    def _coerce_text_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            return "\n".join(parts)
        return str(content or "")

    def _sanitize_openwebui_text(self, text: str) -> str:
        sanitized = str(text or "").strip()
        if not sanitized:
            return ""

        wrapped = self._extract_openwebui_wrapped_user_query(sanitized)
        if wrapped:
            sanitized = wrapped

        marker_matches = list(_OPENWEBUI_QUERY_MARKER_RE.finditer(sanitized))
        if marker_matches:
            candidate = sanitized[marker_matches[-1].end() :].strip()
            if candidate:
                sanitized = candidate
        sanitized = _OPENWEBUI_CONTEXT_BLOCK_RE.sub("", sanitized).strip()
        sanitized = _OPENWEBUI_FILE_CONTEXT_RE.sub("", sanitized).strip()
        sanitized = _OPENWEBUI_SOURCE_SECTION_RE.sub("\n", sanitized).strip()
        sanitized = _OPENWEBUI_ATTACHMENT_LINE_RE.sub("", sanitized).strip()

        lines: list[str] = []
        for raw_line in sanitized.splitlines():
            line = str(raw_line or "").strip()
            if not line:
                continue
            line = _STRUCTURED_SECTION_PREFIX_RE.sub("", line).strip()
            if line:
                lines.append(line)
        return "\n".join(lines).strip() if lines else sanitized

    def _extract_openwebui_wrapped_user_query(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw or not _OPENWEBUI_WRAPPER_HINT_RE.search(raw):
            return ""
        output_matches = list(_OPENWEBUI_OUTPUT_SECTION_RE.finditer(raw))
        if output_matches:
            tail = output_matches[-1].group("body").strip()
            tail = _OPENWEBUI_CONTEXT_BLOCK_RE.sub("", tail).strip()
            candidate = self._last_non_instruction_block(tail)
            if candidate:
                return candidate
        chat_users = [
            str(match.group(1) or "").strip()
            for match in _OPENWEBUI_CHAT_HISTORY_USER_RE.finditer(raw)
        ]
        chat_users = [item for item in chat_users if item]
        return chat_users[-1] if chat_users else ""

    def _last_non_instruction_block(self, text: str) -> str:
        paragraphs = [item.strip() for item in re.split(r"\n\s*\n", str(text or "")) if item.strip()]
        for paragraph in reversed(paragraphs):
            lines: list[str] = []
            for raw_line in paragraph.splitlines():
                line = str(raw_line or "").strip()
                if not line or _OPENWEBUI_WRAPPER_HEADING_RE.match(line):
                    continue
                line = _STRUCTURED_SECTION_PREFIX_RE.sub("", line).strip()
                if not line or _OPENWEBUI_INSTRUCTION_LINE_RE.match(line):
                    continue
                lines.append(line)
            candidate = "\n".join(lines).strip()
            if candidate:
                return candidate
        return ""

    def _sanitize_messages(self, body: dict[str, Any]) -> list[dict[str, str]]:
        sanitized_messages: list[dict[str, str]] = []
        for message in list((body or {}).get("messages") or []):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = self._sanitize_openwebui_text(self._coerce_text_content(message.get("content")))
            if not content:
                continue
            if role == "user" and self._helper_task_type_from_text(content):
                continue
            sanitized_messages.append({"role": role, "content": content})
        if sanitized_messages and sanitized_messages[-1]["role"] == "user":
            return sanitized_messages
        last_user_text = self._sanitize_openwebui_text(self._last_user_text(body))
        if last_user_text:
            sanitized_messages = [item for item in sanitized_messages if item["role"] != "user" or item["content"] != last_user_text]
            sanitized_messages.append({"role": "user", "content": last_user_text})
        return sanitized_messages

    def _build_chat_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        if not self.valves.OPENWEBUI_THIN_MODE:
            return dict(body or {})
        sanitized_messages = self._sanitize_messages(dict(body or {}))
        if not sanitized_messages:
            sanitized_messages = [{"role": "user", "content": self._sanitize_openwebui_text(self._last_user_text(dict(body or {})))}]
        return {"messages": [item for item in sanitized_messages if str(item.get("content") or "").strip()]}

    def _last_user_text(self, body: dict[str, Any]) -> str:
        for message in reversed(list(body.get("messages") or [])):
            if not isinstance(message, dict):
                continue
            if str(message.get("role") or "").strip().lower() != "user":
                continue
            return self._coerce_text_content(message.get("content")).strip()
        return ""

    def _helper_task_type_from_text(self, text: str) -> str:
        if not text:
            return ""
        if any(pattern.search(text) for pattern in _FOLLOW_UP_PATTERNS):
            return "follow_ups"
        if any(pattern.search(text) for pattern in _TITLE_PATTERNS):
            return "title"
        if any(pattern.search(text) for pattern in _TAG_PATTERNS):
            return "tags"
        if any(pattern.search(text) for pattern in _SEARCH_QUERY_PATTERNS):
            return "search_queries"
        return ""

    def _detect_helper_task_type(self, body: dict[str, Any]) -> str:
        text = self._last_user_text(body)
        return self._helper_task_type_from_text(text) or self._helper_task_type_from_text(
            self._sanitize_openwebui_text(text)
        )

    def _local_helper_response(self, helper_task_type: str) -> str:
        normalized = str(helper_task_type or "").strip().lower()
        if normalized == "title":
            return "Enterprise Agent"
        if normalized == "tags":
            return "enterprise-agent"
        if normalized == "follow_ups":
            return "[]"
        if normalized == "search_queries":
            return '{"queries":[]}'
        return ""

    async def _emit_status(
        self,
        event_emitter: Any,
        status_payload: dict[str, Any],
    ) -> None:
        if event_emitter is None:
            return
        normalized_status = _ensure_agentic_status_payload(status_payload)
        payload = {
            "type": "status",
            "data": normalized_status,
        }
        result = event_emitter(payload)
        if inspect.isawaitable(result):
            await result

    async def _emit_status_snapshots(
        self,
        event_emitter: Any,
        state: _TurnStatusState | None,
        snapshots: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> None:
        if state is None or event_emitter is None:
            return
        if state.backend_status_seen:
            return
        for snapshot in list(snapshots or []):
            if snapshot:
                await self._emit_status(event_emitter, snapshot)

    def _sync_tracker_from_status_payload(
        self,
        state: _TurnStatusState,
        status_payload: Mapping[str, Any],
        *,
        now: float,
    ) -> None:
        tracker = state.tracker
        phase = str(status_payload.get("phase") or "").strip()
        if phase:
            tracker.active_phase = phase
        phase_elapsed_ms = status_payload.get("phase_elapsed_ms")
        if isinstance(phase_elapsed_ms, (int, float)):
            tracker.active_phase_started_at = now - (max(0.0, float(phase_elapsed_ms)) / 1000.0)
        elif tracker.active_phase_started_at is None:
            tracker.active_phase_started_at = now
        for key, attr in (
            ("agent", "current_agent"),
            ("selected_agent", "selected_agent"),
            ("label", "latest_label"),
            ("detail", "latest_detail"),
            ("why", "latest_why"),
            ("waiting_on", "latest_waiting_on"),
            ("job_id", "latest_job_id"),
            ("task_id", "latest_task_id"),
            ("source_event_type", "latest_source_event_type"),
        ):
            value = str(status_payload.get(key) or "").strip()
            if value:
                setattr(tracker, attr, value)
        status_id = str(status_payload.get("status_id") or "").strip()
        if status_id:
            tracker.current_status_id = status_id
        status_key = str(status_payload.get("status_key") or "").strip()
        if status_key:
            tracker.current_status_key = status_key
        status_elapsed_ms = status_payload.get("status_elapsed_ms")
        if isinstance(status_elapsed_ms, (int, float)):
            tracker.current_status_started_at = now - (max(0.0, float(status_elapsed_ms)) / 1000.0)
        elif tracker.current_status_started_at is None:
            tracker.current_status_started_at = now
        elapsed_ms = status_payload.get("elapsed_ms")
        if isinstance(elapsed_ms, (int, float)):
            tracker.last_emitted_elapsed_ms = int(elapsed_ms)
        status_seq = status_payload.get("status_seq")
        if isinstance(status_seq, (int, float)):
            tracker.next_status_seq = max(tracker.next_status_seq, int(status_seq) + 1)
        tracker.last_emit_at = now

    async def _emit_backend_status_snapshot(
        self,
        event_emitter: Any,
        state: _TurnStatusState | None,
        status_payload: Mapping[str, Any],
    ) -> None:
        if event_emitter is None:
            return
        payload = _ensure_agentic_status_payload(status_payload)
        if not payload:
            return
        if state is not None:
            now = self._now()
            async with state.emit_lock:
                incoming_phase = str(payload.get("phase") or "").strip()
                current_phase = str(state.tracker.active_phase or "").strip()
                if (
                    incoming_phase
                    and not state.backend_status_seen
                    and _PHASE_RANKS.get(incoming_phase, -1) < _PHASE_RANKS.get(current_phase, -1)
                ):
                    return
                state.backend_status_seen = True
                self._sync_tracker_from_status_payload(state, payload, now=now)
                status_seq = payload.get("status_seq")
                if isinstance(status_seq, (int, float)) and int(status_seq) <= state.last_forwarded_status_seq:
                    return
                signature = _status_signature(payload)
                if signature in {state.last_forwarded_status_signature, state.tracker.last_signature}:
                    state.last_forwarded_status_signature = signature
                    state.tracker.last_signature = signature
                    return
                state.last_forwarded_status_signature = signature
                if isinstance(status_seq, (int, float)):
                    state.last_forwarded_status_seq = int(status_seq)
                state.tracker.last_signature = signature
        await self._emit_status(event_emitter, payload)

    async def _emit_tracker_progress(
        self,
        event_emitter: Any,
        state: _TurnStatusState | None,
        progress_event: dict[str, Any],
    ) -> None:
        if state is None or event_emitter is None:
            return
        if state.backend_status_seen:
            return
        async with state.emit_lock:
            snapshots = state.tracker.progress_snapshots(progress_event, self._now())
        await self._emit_status_snapshots(event_emitter, state, snapshots)

    async def _transition_phase(
        self,
        event_emitter: Any,
        state: _TurnStatusState | None,
        next_phase: str,
        *,
        source_event_type: str,
        label: str | None = None,
        detail: str | None = "",
    ) -> None:
        if state is None:
            return
        if state.backend_status_seen:
            return
        async with state.emit_lock:
            snapshots = state.tracker.transition_phase(
                next_phase,
                now=self._now(),
                source_event_type=source_event_type,
                label=label,
                detail=detail,
            )
        await self._emit_status_snapshots(event_emitter, state, snapshots)

    async def _emit_completion_status(
        self,
        event_emitter: Any,
        state: _TurnStatusState | None,
    ) -> None:
        if state is None:
            return
        if state.backend_status_seen and state.tracker.active_phase in _TERMINAL_PHASES:
            return
        async with state.emit_lock:
            snapshots = state.tracker.completion_snapshots(self._now())
        await self._emit_status_snapshots(event_emitter, state, snapshots)

    async def _emit_failure_status(
        self,
        event_emitter: Any,
        state: _TurnStatusState | None,
    ) -> None:
        if state is None or event_emitter is None:
            return
        if state.backend_status_seen and state.tracker.active_phase in _TERMINAL_PHASES:
            return
        async with state.emit_lock:
            snapshot = state.tracker.failure_snapshot(self._now())
        await self._emit_status_snapshots(event_emitter, state, [snapshot] if snapshot else [])

    async def _timer_heartbeat(
        self,
        event_emitter: Any,
        state: _TurnStatusState,
        stop_event: asyncio.Event,
    ) -> None:
        while True:
            if state.backend_status_seen:
                return
            delay = state.tracker.seconds_until_next_heartbeat(self._now(), interval_seconds=STATUS_HEARTBEAT_SECONDS)
            if delay is None:
                return
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=delay)
                return
            except asyncio.TimeoutError:
                async with state.emit_lock:
                    snapshot = state.tracker.heartbeat_snapshot(self._now())
                await self._emit_status_snapshots(event_emitter, state, [snapshot] if snapshot else [])

    async def _iter_sse_events(self, response: httpx.Response):
        event_name = ""
        data_lines: list[str] = []
        async for raw_line in response.aiter_lines():
            if raw_line is None:
                continue
            line = str(raw_line).rstrip("\r")
            if not line:
                if data_lines:
                    yield (event_name or "message"), "\n".join(data_lines)
                event_name = ""
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
                continue
            if line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].lstrip())
        if data_lines:
            yield (event_name or "message"), "\n".join(data_lines)

    async def _call_agent_completion_once(
        self,
        client: httpx.AsyncClient,
        *,
        payload: dict[str, Any],
        chat_id: str,
        user_id: str,
        user_email: str = "",
        message_id: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        response = await client.post(
            f"{self.valves.AGENT_BASE_URL.rstrip('/')}/chat/completions",
            headers={
                "Content-Type": "application/json",
                **self._agent_headers(chat_id=chat_id, user_id=user_id, user_email=user_email, message_id=message_id),
            },
            json=payload,
        )
        await _raise_for_backend_completion_status(response)
        result = response.json()
        content = (
            str((((result.get("choices") or [{}])[0].get("message") or {}).get("content") or "")).strip()
            or "The agent returned an empty response."
        )
        return content, list(result.get("artifacts") or [])

    async def _call_agent_completion_stream(
        self,
        client: httpx.AsyncClient,
        *,
        payload: dict[str, Any],
        chat_id: str,
        user_id: str,
        user_email: str = "",
        message_id: str,
        event_emitter: Any,
        timer_state: _TurnStatusState | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        content_parts: list[str] = []
        artifacts: list[dict[str, Any]] = []
        async with client.stream(
            "POST",
            f"{self.valves.AGENT_BASE_URL.rstrip('/')}/chat/completions",
            headers={
                "Content-Type": "application/json",
                **self._agent_headers(chat_id=chat_id, user_id=user_id, user_email=user_email, message_id=message_id),
            },
            json=payload,
        ) as response:
            await _raise_for_backend_completion_status(response)
            async for event_name, data in self._iter_sse_events(response):
                if data == "[DONE]":
                    break
                if event_name == "status":
                    try:
                        status_payload = json.loads(data)
                    except json.JSONDecodeError:
                        status_payload = {}
                    if isinstance(status_payload, dict):
                        await self._emit_backend_status_snapshot(event_emitter, timer_state, status_payload)
                    continue
                if event_name == "progress":
                    try:
                        progress_event = json.loads(data)
                    except json.JSONDecodeError:
                        progress_event = {}
                    await self._emit_tracker_progress(event_emitter, timer_state, progress_event)
                    continue
                if event_name == "artifacts":
                    try:
                        parsed_artifacts = json.loads(data)
                    except json.JSONDecodeError:
                        parsed_artifacts = []
                    if isinstance(parsed_artifacts, list):
                        artifacts = [item for item in parsed_artifacts if isinstance(item, dict)]
                    continue
                if event_name == "metadata":
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (((chunk.get("choices") or [{}])[0].get("delta") or {}))
                part = str(delta.get("content") or "")
                if part:
                    if (
                        timer_state is not None
                        and not timer_state.backend_status_seen
                        and timer_state.tracker.active_phase != PHASE_SYNTHESIZING
                    ):
                        await self._transition_phase(
                            event_emitter,
                            timer_state,
                            PHASE_SYNTHESIZING,
                            source_event_type="content_delta",
                            label="Synthesizing answer",
                            detail="Grounding final response",
                        )
                    content_parts.append(part)
        content = "".join(content_parts).strip() or "The agent returned an empty response."
        return content, artifacts

    async def _fallback_completion(
        self,
        client: httpx.AsyncClient,
        *,
        payload: dict[str, Any],
        chat_id: str,
        user_id: str,
        user_email: str = "",
        message_id: str,
        event_emitter: Any,
        timer_state: _TurnStatusState | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        return await self._call_agent_completion_once(
            client,
            payload=payload,
            chat_id=chat_id,
            user_id=user_id,
            user_email=user_email,
            message_id=message_id,
        )

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
        __metadata__: dict[str, Any] | None = None,
        __files__: list[Any] | None = None,
        __request__: Any = None,
        __event_emitter__: Any = None,
    ) -> str:
        metadata = dict(__metadata__ or {})
        chat_id, user_id, user_email, message_id = self._chat_scope(dict(__user__ or {}), metadata)
        collection_id = f"{self.valves.COLLECTION_PREFIX}{chat_id}"
        helper_task_type = self._detect_helper_task_type(dict(body or {}))
        if helper_task_type and not self.valves.OPENWEBUI_ENABLE_HELPERS:
            return self._local_helper_response(helper_task_type)
        timer_state: _TurnStatusState | None = None
        timer_stop: asyncio.Event | None = None
        timer_task: asyncio.Task[Any] | None = None

        normalized_files = self._normalize_files(list(__files__ or []))
        timeout = httpx.Timeout(self.valves.REQUEST_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            try:
                if not helper_task_type:
                    timer_state = _TurnStatusState(tracker=TurnStatusTracker(turn_started_at=self._now()))
                    timer_stop = asyncio.Event()
                    await self._emit_status_snapshots(
                        __event_emitter__,
                        timer_state,
                        timer_state.tracker.start_snapshots(self._now()),
                    )
                    timer_task = asyncio.create_task(self._timer_heartbeat(__event_emitter__, timer_state, timer_stop))

                upload_items: list[dict[str, Any]] = []
                should_handoff_files = bool(normalized_files) and not helper_task_type
                if should_handoff_files:
                    await self._transition_phase(
                        __event_emitter__,
                        timer_state,
                        PHASE_UPLOADING,
                        source_event_type="upload_started",
                        label="Uploading inputs",
                        detail="Sending files to the agent workspace",
                    )
                if normalized_files and not self.valves.OPENWEBUI_ALLOW_FILE_BYTE_TRANSPORT:
                    raise RuntimeError("OpenWebUI file byte transport is disabled for this agent pipe.")
                for item in normalized_files if should_handoff_files else []:
                    try:
                        file_response = await self._download_openwebui_file(client, __request__, item["id"])
                    except Exception as exc:
                        raise RuntimeError(
                            "File upload handoff failed before agent execution while fetching files from OpenWebUI."
                        ) from exc
                    upload_items.append(
                        {
                            "file_bytes": file_response.content,
                            "filename": item["filename"],
                            "content_type": str(
                                file_response.headers.get("content-type") or "application/octet-stream"
                            ),
                            "source_id": item["id"],
                        }
                    )
                upload_handoff: dict[str, Any] = {}
                if upload_items:
                    try:
                        upload_handoff = await asyncio.to_thread(
                            self._upload_files_to_agent_sync,
                            upload_items=upload_items,
                            chat_id=chat_id,
                            user_id=user_id,
                            user_email=user_email,
                            message_id=message_id,
                            collection_id=collection_id,
                        )
                    except UploadHandoffError:
                        raise
                    except Exception as exc:
                        raise RuntimeError(
                            "File upload handoff failed before agent execution while sending files to the agent workspace."
                        ) from exc
                if not helper_task_type:
                    await self._transition_phase(
                        __event_emitter__,
                        timer_state,
                        PHASE_SEARCHING,
                        source_event_type="chat_request_started",
                        label="Searching knowledge base",
                        detail="Waiting for the gateway to stream runtime progress",
                    )

                payload = self._build_chat_payload(dict(body or {}))
                request_metadata = dict(payload.get("metadata") or {})
                request_metadata["openwebui_client"] = True
                request_metadata["openwebui_thin_mode"] = bool(self.valves.OPENWEBUI_THIN_MODE)
                request_metadata["document_source_policy"] = "agent_repository_only"
                request_metadata["collection_id"] = collection_id
                request_metadata["upload_collection_id"] = collection_id
                request_metadata["kb_collection_id"] = self.valves.KB_COLLECTION_ID
                request_metadata["kb_collection_confirmed"] = False
                active_uploaded_doc_ids = [
                    str(item).strip()
                    for item in list(upload_handoff.get("active_uploaded_doc_ids") or [])
                    if str(item).strip()
                ]
                if active_uploaded_doc_ids:
                    request_metadata["uploaded_doc_ids"] = active_uploaded_doc_ids
                if upload_handoff:
                    request_metadata["upload_manifest"] = {
                        "object": "openwebui.upload_manifest",
                        "collection_id": collection_id,
                        "active_uploaded_doc_ids": active_uploaded_doc_ids,
                        "doc_ids": [
                            str(item).strip()
                            for item in list(upload_handoff.get("doc_ids") or [])
                            if str(item).strip()
                        ],
                        "uploaded_file_count": len(upload_items),
                        "result_count": len(list(upload_handoff.get("upload_results") or [])),
                    }
                if user_email:
                    request_metadata["user_email"] = user_email
                if helper_task_type:
                    request_metadata["openwebui_helper_task_type"] = helper_task_type
                payload["metadata"] = request_metadata
                payload["model"] = self.valves.AGENT_MODEL_ID

                if helper_task_type:
                    payload["stream"] = False
                    content, artifacts = await self._call_agent_completion_once(
                        client,
                        payload=payload,
                        chat_id=chat_id,
                        user_id=user_id,
                        user_email=user_email,
                        message_id=message_id,
                    )
                else:
                    payload["stream"] = True
                    try:
                        content, artifacts = await self._call_agent_completion_stream(
                            client,
                            payload=payload,
                            chat_id=chat_id,
                            user_id=user_id,
                            user_email=user_email,
                            message_id=message_id,
                            event_emitter=__event_emitter__,
                            timer_state=timer_state,
                        )
                    except BackendCompletionError:
                        raise
                    except Exception:
                        logger.exception(
                            "Agent streamed completion failed for chat_id=%s message_id=%s; retrying without stream",
                            chat_id,
                            message_id,
                        )
                        payload["stream"] = False
                        content, artifacts = await self._fallback_completion(
                            client,
                            payload=payload,
                            chat_id=chat_id,
                            user_id=user_id,
                            user_email=user_email,
                            message_id=message_id,
                            event_emitter=__event_emitter__,
                            timer_state=timer_state,
                        )

                rendered_artifacts = self._render_artifacts(artifacts)
                if rendered_artifacts:
                    content = f"{content}\n\n{rendered_artifacts}".strip()
                if not helper_task_type:
                    await self._emit_completion_status(__event_emitter__, timer_state)
                return content
            except UploadHandoffError as exc:
                logger.exception(
                    "OpenWebUI upload handoff failed for chat_id=%s message_id=%s",
                    chat_id,
                    message_id,
                )
                if not helper_task_type:
                    await self._emit_failure_status(__event_emitter__, timer_state)
                return str(exc)
            except BackendCompletionError as exc:
                logger.warning(
                    "OpenWebUI backend completion failed for chat_id=%s message_id=%s status_code=%s error_code=%s detail=%s",
                    chat_id,
                    message_id,
                    exc.status_code if exc.status_code is not None else "unknown",
                    exc.error_code or "-",
                    _compact_response_text(str(exc.detail or "")),
                )
                if not helper_task_type:
                    await self._emit_failure_status(__event_emitter__, timer_state)
                return str(exc)
            except Exception:
                logger.exception(
                    "OpenWebUI pipe request failed for chat_id=%s message_id=%s",
                    chat_id,
                    message_id,
                )
                if not helper_task_type:
                    await self._emit_failure_status(__event_emitter__, timer_state)
                raise
            finally:
                if timer_stop is not None:
                    timer_stop.set()
                if timer_task is not None:
                    timer_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await timer_task
