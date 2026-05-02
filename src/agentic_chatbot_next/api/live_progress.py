from __future__ import annotations

import queue
import time
from typing import Any, Dict, Iterable, List

from agentic_chatbot_next.observability.events import RuntimeEvent
from agentic_chatbot_next.runtime.event_sink import RuntimeEventSink
from agentic_chatbot_next.runtime.frontend_events import FrontendEventPolicy

_PHASE_LABELS = {
    "basic_turn_started": "Composing response",
    "basic_turn_completed": "Response ready",
    "agent_turn_started": "Running agent",
    "agent_turn_completed": "Agent completed",
    "coordinator_planning_started": "Planning tasks",
    "coordinator_planning_completed": "Task plan ready",
    "coordinator_batch_started": "Dispatching workers",
    "coordinator_finalizer_started": "Running finalizer",
    "coordinator_finalizer_completed": "Final synthesis complete",
    "coordinator_verifier_started": "Running verifier",
    "coordinator_verifier_completed": "Verification complete",
}
_TOOL_STATUS_PHASE = "searching_knowledge_base"
_AUDIT_EVENT_TYPES = {
    "agent_run_started",
    "agent_run_completed",
    "agent_turn_started",
    "agent_turn_completed",
    "basic_turn_started",
    "basic_turn_completed",
    "coordinator_planning_started",
    "coordinator_planning_completed",
    "coordinator_finalizer_started",
    "coordinator_finalizer_completed",
    "coordinator_verifier_started",
    "coordinator_verifier_completed",
    "worker_agent_started",
    "worker_agent_completed",
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _duration_ms(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0, int(value))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _audit_status(status: str) -> str:
    normalized = _clean(status).lower()
    if normalized in {"completed", "complete", "succeeded", "success"}:
        return "completed"
    if normalized in {"error", "failed", "failure"}:
        return "error"
    if normalized in {"queued", "waiting"}:
        return normalized
    return "running"


def _status_label(status: str) -> str:
    normalized = _audit_status(status)
    if normalized == "completed":
        return "Completed"
    if normalized == "error":
        return "Failed"
    if normalized == "queued":
        return "Queued"
    if normalized == "waiting":
        return "Waiting"
    return "Running"


def _ui_status(status: str) -> str:
    normalized = _audit_status(status)
    if normalized == "error":
        return "error"
    if normalized == "completed":
        return "complete"
    return "in_progress"


def _agentic_state(status: str) -> str:
    normalized = _audit_status(status)
    if normalized == "error":
        return "error"
    if normalized == "completed":
        return "complete"
    return "active"


def _status_title(subject: str, status: str) -> str:
    normalized = _audit_status(status)
    if normalized == "completed":
        return f"{subject} completed"
    if normalized == "error":
        return f"{subject} failed"
    if normalized == "queued":
        return f"{subject} queued"
    if normalized == "waiting":
        return f"{subject} waiting"
    return f"{subject} running"


def _audit_timing(status: str, elapsed_ms: int, timestamp: int) -> Dict[str, Any]:
    done = _audit_status(status) in {"completed", "error"}
    return {
        "kind": "total" if done else "stage",
        "live": not done,
        "elapsed_ms": elapsed_ms,
        "status_elapsed_ms": elapsed_ms,
        "phase_elapsed_ms": elapsed_ms,
        "total_elapsed_ms": elapsed_ms,
        "snapshot_timestamp_ms": timestamp,
    }


def _agentic_status(
    *,
    status: str,
    kind: str,
    title: str,
    subtitle: str,
    chips: Iterable[Any],
    elapsed_ms: int,
    timestamp: int,
) -> Dict[str, Any]:
    deduped_chips: List[str] = []
    for chip in chips:
        text = _clean(chip)
        if text and text not in deduped_chips:
            deduped_chips.append(text)
        if len(deduped_chips) >= 5:
            break
    return {
        "version": 1,
        "state": _agentic_state(status),
        "kind": kind,
        "title": title,
        "subtitle": subtitle,
        "chips": deduped_chips,
        "timing": _audit_timing(status, elapsed_ms, timestamp),
    }


def _agent_activity_payload(
    *,
    activity_id: str,
    agent_name: str,
    role: str,
    status: str,
    title: str,
    description: str,
    parent_agent: str = "",
    task_id: str = "",
    job_id: str = "",
    parallel_group_id: str = "",
    started_at: str = "",
    completed_at: str = "",
    duration_ms: int | None = None,
) -> Dict[str, Any]:
    return {
        "version": 1,
        "activity_id": activity_id,
        "agent_name": agent_name,
        "role": role,
        "status": _audit_status(status),
        "title": title,
        "description": description,
        "parent_agent": parent_agent,
        "task_id": task_id,
        "job_id": job_id,
        "parallel_group_id": parallel_group_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_ms": duration_ms,
    }


def _parallel_group_payload(
    *,
    group_id: str,
    group_kind: str,
    status: str,
    execution_mode: str,
    size: int,
    members: Iterable[Any],
    reason: str,
    started_at: str = "",
    completed_at: str = "",
    duration_ms: int | None = None,
) -> Dict[str, Any]:
    return {
        "version": 1,
        "group_id": group_id,
        "group_kind": group_kind,
        "status": _audit_status(status),
        "execution_mode": execution_mode,
        "size": max(0, int(size or 0)),
        "members": [dict(item) for item in members if isinstance(item, dict)],
        "reason": reason,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_ms": duration_ms,
    }


def _task_summaries(raw_tasks: Any, *, limit: int = 12) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for raw in list(raw_tasks or [])[:limit]:
        if not isinstance(raw, dict):
            continue
        dependencies = raw.get("depends_on") or raw.get("dependencies") or []
        if isinstance(dependencies, str):
            dependencies = [dependencies] if dependencies else []
        tasks.append(
            {
                "id": _clean(raw.get("id") or raw.get("task_id")),
                "title": _clean(raw.get("title") or raw.get("description"))[:240],
                "executor": _clean(raw.get("executor") or raw.get("agent_name")),
                "mode": _clean(raw.get("mode")) or "sequential",
                "dependencies": [_clean(item) for item in list(dependencies or []) if _clean(item)][:8],
                "handoff_schema": _clean(raw.get("handoff_schema")),
            }
        )
    return tasks


def _preview_value(value: Any, *, fallback: str = "", limit: int = 240) -> str:
    text = _clean(value) if isinstance(value, str) else ""
    if not text and value not in (None, "", [], {}):
        try:
            import json

            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
    text = text or fallback
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _audit_item_payload(
    *,
    item_id: str,
    kind: str,
    status: str,
    title: str,
    subtitle: str = "",
    actor: str = "",
    target: str = "",
    parent_id: str = "",
    group_id: str = "",
    chips: Iterable[Any] = (),
    started_at: str = "",
    completed_at: str = "",
    duration_ms: int | None = None,
    tasks: Iterable[Dict[str, Any]] | None = None,
    members: Iterable[Dict[str, Any]] | None = None,
    input: Any = None,
    output: Any = None,
    error: Any = None,
    notices: Iterable[Any] = (),
    redacted_fields: Iterable[Any] = (),
    payload_limit_chars: int | None = None,
    visible: bool = True,
) -> Dict[str, Any]:
    deduped_chips: List[str] = []
    for chip in chips:
        text = _clean(chip)
        if text and text not in deduped_chips:
            deduped_chips.append(text)
        if len(deduped_chips) >= 8:
            break
    item: Dict[str, Any] = {
        "version": 1,
        "item_id": item_id,
        "kind": kind,
        "status": _audit_status(status),
        "title": title,
        "subtitle": subtitle,
        "actor": actor,
        "target": target,
        "parent_id": parent_id,
        "group_id": group_id,
        "chips": deduped_chips,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_ms": duration_ms,
        "visible": visible,
    }
    if tasks is not None:
        item["tasks"] = [dict(task) for task in tasks if isinstance(task, dict)]
    if members is not None:
        item["members"] = [dict(member) for member in members if isinstance(member, dict)]
    if input is not None:
        item["input"] = input
    if output is not None:
        item["output"] = output
    if error is not None:
        item["error"] = error
    notices_list = [_clean(notice) for notice in notices if _clean(notice)]
    if notices_list:
        item["notices"] = notices_list
    redacted_list = [_clean(field) for field in redacted_fields if _clean(field)]
    if redacted_list:
        item["redacted_fields"] = redacted_list
    if payload_limit_chars:
        item["payload_limit_chars"] = int(payload_limit_chars)
    return item


def _attach_audit_item(base: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    base["agentic_audit_item"] = item
    return base


def _audit_kind_for_role(role: str) -> str:
    normalized = _clean(role).lower()
    if normalized == "planner":
        return "plan"
    if normalized == "verifier":
        return "verify"
    if normalized == "finalizer":
        return "synthesize"
    if normalized == "worker":
        return "handoff"
    return "agent"


def _agent_audit_title(*, role: str, status: str, agent_name: str, parent_agent: str, task_id: str, title: str) -> str:
    normalized_status = _audit_status(status)
    if role == "worker":
        if normalized_status == "completed":
            return f"{agent_name} completed {task_id or 'task'}"
        actor = parent_agent or "Coordinator"
        return f"{actor} handed off {task_id or 'work'} to {agent_name}"
    if role == "planner" and normalized_status == "running":
        return "Coordinator is planning tasks"
    if role == "verifier":
        return title.replace("Running verifier", "Verifier checking") if title else "Verifier checking answer"
    return title


def _status_envelope(
    base: Dict[str, Any],
    *,
    status_id: str,
    status: str,
    phase: str,
    title: str,
    subtitle: str,
    kind: str,
    chips: Iterable[Any],
    elapsed_ms: int,
    timestamp: int,
) -> Dict[str, Any]:
    ui_status = _ui_status(status)
    base.update(
        {
            "status_id": status_id,
            "status_key": status_id,
            "description": title,
            "status": ui_status,
            "done": ui_status in {"complete", "error"},
            "hidden": False,
            "elapsed_ms": elapsed_ms,
            "phase": phase,
            "phase_label": "Searching knowledge base" if phase == _TOOL_STATUS_PHASE else str(phase or ""),
            "phase_elapsed_ms": elapsed_ms,
            "status_elapsed_ms": elapsed_ms,
            "agentic_status": _agentic_status(
                status=status,
                kind=kind,
                title=title,
                subtitle=subtitle,
                chips=chips,
                elapsed_ms=elapsed_ms,
                timestamp=timestamp,
            ),
        }
    )
    return base


def _with_agent_activity(
    base: Dict[str, Any],
    *,
    event: RuntimeEvent,
    payload: Dict[str, Any],
    role: str,
    status: str,
    title: str,
    description: str,
    agent_name: str,
    parent_agent: str = "",
    task_id: str = "",
    job_id: str = "",
    parallel_group_id: str = "",
    timestamp: int,
    phase: str = _TOOL_STATUS_PHASE,
) -> Dict[str, Any]:
    activity_id = _clean(payload.get("activity_id")) or "-".join(
        item for item in ("agent", role, agent_name, task_id, job_id or event.job_id) if item
    )
    duration = _duration_ms(payload.get("duration_ms")) or 0
    base.setdefault("source_event_type", event.event_type)
    base.setdefault("timestamp", timestamp)
    base.setdefault("agent", agent_name)
    base.setdefault("selected_agent", _clean(parent_agent) or agent_name)
    base.setdefault("job_id", job_id)
    base.setdefault("task_id", task_id)
    base.setdefault("label", title)
    base.setdefault("detail", description)
    status_id = f"agent-{activity_id}"
    _status_envelope(
        base,
        status_id=status_id,
        status=status,
        phase=phase,
        title=title,
        subtitle=description,
        kind="agent",
        chips=[role, agent_name, parent_agent, task_id, job_id],
        elapsed_ms=duration,
        timestamp=timestamp,
    )
    base["agentic_agent_activity"] = _agent_activity_payload(
        activity_id=activity_id,
        agent_name=agent_name,
        role=role,
        status=status,
        title=title,
        description=description,
        parent_agent=parent_agent,
        task_id=task_id,
        job_id=job_id,
        parallel_group_id=parallel_group_id,
        started_at=_clean(payload.get("started_at")),
        completed_at=_clean(payload.get("completed_at")),
        duration_ms=_duration_ms(payload.get("duration_ms")),
    )
    actor = _clean(parent_agent) or _clean(base.get("selected_agent")) or ""
    target = agent_name if actor != agent_name else ""
    return _attach_audit_item(
        base,
        _audit_item_payload(
            item_id=status_id,
            kind=_audit_kind_for_role(role),
            status=status,
            title=_agent_audit_title(
                role=role,
                status=status,
                agent_name=agent_name,
                parent_agent=actor,
                task_id=task_id,
                title=title,
            ),
            subtitle=description,
            actor=actor if target else "",
            target=target or agent_name,
            parent_id=parent_agent,
            group_id=parallel_group_id,
            chips=[role, agent_name, task_id, job_id, _status_label(status)],
            started_at=_clean(payload.get("started_at")),
            completed_at=_clean(payload.get("completed_at")),
            duration_ms=_duration_ms(payload.get("duration_ms")),
        ),
    )


def _parallel_group_status_from_runtime_event(
    event: RuntimeEvent,
    payload: Dict[str, Any],
    *,
    timestamp: int,
) -> Dict[str, Any]:
    group_id = _clean(payload.get("group_id") or event.event_id)
    group_kind = _clean(payload.get("group_kind")) or "parallel_group"
    status = _audit_status(payload.get("status") or ("completed" if event.event_type.endswith("_completed") else "running"))
    execution_mode = _clean(payload.get("execution_mode")) or "parallel"
    members = [dict(item) for item in list(payload.get("members") or []) if isinstance(item, dict)]
    size = int(payload.get("size") or len(members) or 0)
    duration = _duration_ms(payload.get("duration_ms")) or 0
    is_parallel = execution_mode == "parallel" and size > 1
    is_tool_wave = group_kind == "tool_wave"
    subject = (
        f"Coordinator handed off {size} task(s) to worker agents"
        if group_kind == "worker_batch"
        else f"Ran {size} independent tools in parallel"
        if is_tool_wave and is_parallel
        else "Sequenced tool calls"
        if is_tool_wave and size > 1
        else "Tool ran"
    )
    title = _status_title(subject, status) if status != "running" else subject
    subtitle = _clean(payload.get("reason"))
    if is_tool_wave and size <= 1:
        subtitle = "One tool call was scheduled; details are shown on the tool row."
    base = {
        "type": "parallel_group_trace",
        "label": title,
        "detail": subtitle,
        "agent": event.agent_name,
        "selected_agent": event.agent_name,
        "job_id": event.job_id,
        "task_id": "",
        "why": subtitle,
        "waiting_on": "",
        "source_event_type": event.event_type,
        "timestamp": timestamp,
    }
    _status_envelope(
        base,
        status_id=f"group-{group_id}",
        status=status,
        phase=_TOOL_STATUS_PHASE,
        title=title,
        subtitle=subtitle,
        kind="parallel_group",
        chips=[execution_mode, group_kind.replace("_", " "), f"{size} item(s)", event.agent_name],
        elapsed_ms=duration,
        timestamp=timestamp,
    )
    base["agentic_parallel_group"] = _parallel_group_payload(
        group_id=group_id,
        group_kind=group_kind,
        status=status,
        execution_mode=execution_mode,
        size=size,
        members=members,
        reason=subtitle,
        started_at=_clean(payload.get("started_at")),
        completed_at=_clean(payload.get("completed_at")),
        duration_ms=_duration_ms(payload.get("duration_ms")),
    )
    return _attach_audit_item(
        base,
        _audit_item_payload(
            item_id=f"group-{group_id}",
            kind="parallel" if is_parallel else ("sequence" if is_tool_wave and size > 1 else "group"),
            status=status,
            title=title,
            subtitle=subtitle,
            actor=event.agent_name,
            target="tools" if is_tool_wave else "worker agents",
            group_id=group_id,
            chips=[
                "parallel" if is_parallel else execution_mode,
                group_kind.replace("_", " "),
                f"{size} item(s)",
                _status_label(status),
            ],
            started_at=_clean(payload.get("started_at")),
            completed_at=_clean(payload.get("completed_at")),
            duration_ms=_duration_ms(payload.get("duration_ms")),
            members=members,
            notices=[subtitle] if is_tool_wave and size <= 1 else [],
            visible=not (is_tool_wave and size <= 1),
        ),
    )


class LiveProgressSink(RuntimeEventSink):
    def __init__(self, settings: Any | None = None, *, policy: FrontendEventPolicy | None = None) -> None:
        self.events: queue.Queue = queue.Queue()
        self.policy = policy or FrontendEventPolicy.from_settings(settings)

    def mark_done(self) -> None:
        self.events.put(None)

    def emit_progress(self, event_type: str, **payload: Any) -> None:
        event = {"type": str(event_type), "timestamp": int(payload.pop("timestamp", _now_ms())), **payload}
        if self.policy.allows_progress_event(event):
            self.events.put(event)

    def emit(self, event: RuntimeEvent) -> None:
        translated = self._translate_runtime_event(event)
        if translated is not None and self.policy.allows_translated_event(translated):
            self.events.put(translated)

    def _translate_runtime_event(self, event: RuntimeEvent) -> Dict[str, Any] | None:
        payload = dict(event.payload or {})
        timestamp = _now_ms()

        if event.event_type in {"tool_start", "tool_end", "tool_error"}:
            return _tool_status_from_runtime_event(event, payload, timestamp=timestamp)

        if event.event_type in {
            "coordinator_worker_batch_started",
            "coordinator_worker_batch_completed",
            "tool_parallel_group_started",
            "tool_parallel_group_completed",
        }:
            return _parallel_group_status_from_runtime_event(event, payload, timestamp=timestamp)

        if event.event_type == "router_decision":
            reasons = [str(item) for item in (payload.get("reasons") or []) if str(item)]
            detail = reasons[0] if reasons else f"Route {payload.get('route') or ''}".strip()
            target = str(payload.get("suggested_agent") or payload.get("route") or "agent")
            base = {
                "type": "route_decision",
                "status_id": f"audit-route-{event.event_id}",
                "status_key": f"audit-route-{event.event_id}",
                "label": f"Routed to {target}",
                "detail": detail,
                "agent": str(payload.get("suggested_agent") or ""),
                "selected_agent": str(payload.get("suggested_agent") or ""),
                "status": str(payload.get("route") or ""),
                "done": True,
                "hidden": False,
                "why": detail,
                "counts": {
                    "confidence": float(payload.get("confidence") or 0.0),
                },
                "timestamp": timestamp,
            }
            return _attach_audit_item(
                base,
                _audit_item_payload(
                    item_id=f"route-{event.event_id}",
                    kind="route",
                    status="completed",
                    title=f"Router selected {target}",
                    subtitle=detail,
                    actor="router",
                    target=target,
                    chips=[
                        str(payload.get("route") or ""),
                        f"{float(payload.get('confidence') or 0.0):.0%} confidence",
                        str(payload.get("router_method") or ""),
                    ],
                ),
            )

        if event.event_type == "agent_run_started":
            base = {
                "type": "agent_selected",
                "label": f"Running {event.agent_name or 'agent'}",
                "detail": str(payload.get("mode") or ""),
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "status": "running",
                "why": str(payload.get("why") or ""),
                "timestamp": timestamp,
            }
            return _with_agent_activity(
                base,
                event=event,
                payload=payload,
                role="top_level",
                status="running",
                title=f"Running {event.agent_name or 'agent'}",
                description=f"{event.agent_name or 'Agent'} is handling this request.",
                agent_name=event.agent_name,
                timestamp=timestamp,
                phase="starting",
            )

        if event.event_type == "agent_run_completed":
            status = str(payload.get("status") or "completed")
            base = {
                "type": "summary",
                "label": f"Completed {event.agent_name or 'agent'}",
                "detail": str(payload.get("detail") or payload.get("output_preview") or ""),
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "status": status,
                "why": str(payload.get("why") or ""),
                "timestamp": timestamp,
            }
            return _with_agent_activity(
                base,
                event=event,
                payload=payload,
                role="top_level",
                status=status,
                title=f"{event.agent_name or 'Agent'} completed",
                description=str(payload.get("detail") or "The selected agent finished this turn."),
                agent_name=event.agent_name,
                timestamp=timestamp,
                phase="answer_ready",
            )

        if event.event_type == "coordinator_planning_started":
            planner_agent = str(payload.get("planner_agent") or event.agent_name or "")
            base = {
                "type": "phase_start",
                "label": _PHASE_LABELS[event.event_type],
                "detail": str(payload.get("detail") or ""),
                "agent": planner_agent,
                "selected_agent": event.agent_name,
                "status": "running",
                "why": str(payload.get("why") or ""),
                "timestamp": timestamp,
            }
            return _with_agent_activity(
                base,
                event=event,
                payload=payload,
                role="planner",
                status="running",
                title=f"Planner {planner_agent} running",
                description="Planner is decomposing the request into executable tasks.",
                agent_name=planner_agent,
                parent_agent=event.agent_name,
                timestamp=timestamp,
            )

        if event.event_type == "coordinator_planning_completed":
            planner_agent = str(payload.get("planner_agent") or event.agent_name or "")
            tasks = _task_summaries(payload.get("tasks") or [])
            task_count = _safe_int(payload.get("task_count"), len(tasks))
            task_count = task_count or len(tasks)
            detail = f"{task_count} task(s)"
            base = {
                "type": "phase_end",
                "label": _PHASE_LABELS[event.event_type],
                "detail": detail,
                "agent": planner_agent,
                "selected_agent": event.agent_name,
                "job_id": event.job_id,
                "status": "completed",
                "why": str(payload.get("why") or ""),
                "waiting_on": str(payload.get("waiting_on") or ""),
                "timestamp": timestamp,
            }
            status_id = f"agent-agent-planner-{planner_agent}-{event.job_id}".rstrip("-")
            _status_envelope(
                base,
                status_id=status_id,
                status="completed",
                phase=_TOOL_STATUS_PHASE,
                title=f"Coordinator planned {task_count} task(s)",
                subtitle=str(payload.get("summary") or detail),
                kind="plan",
                chips=["planner", planner_agent, f"{task_count} task(s)"],
                elapsed_ms=0,
                timestamp=timestamp,
            )
            base["agentic_agent_activity"] = _agent_activity_payload(
                activity_id=status_id.replace("agent-", "", 1),
                agent_name=planner_agent,
                role="planner",
                status="completed",
                title="Task plan ready",
                description=detail,
                parent_agent=event.agent_name if planner_agent != event.agent_name else "",
                job_id=event.job_id,
            )
            return _attach_audit_item(
                base,
                _audit_item_payload(
                    item_id=status_id,
                    kind="plan",
                    status="completed",
                    title=f"Coordinator planned {task_count} task(s)",
                    subtitle=str(payload.get("summary") or detail),
                    actor=event.agent_name,
                    target=planner_agent,
                    chips=["plan", f"{task_count} task(s)", planner_agent],
                    tasks=tasks,
                ),
            )

        if event.event_type == "coordinator_batch_started":
            task_ids = [str(item) for item in (payload.get("task_ids") or []) if str(item)]
            title = f"Coordinator queued {len(task_ids)} task(s)"
            base = {
                "type": "phase_start",
                "status_id": f"audit-batch-{event.event_id}",
                "status_key": f"audit-batch-{event.event_id}",
                "label": title,
                "detail": ", ".join(task_ids[:4]),
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "job_id": event.job_id,
                "status": "running",
                "done": False,
                "hidden": False,
                "why": str(payload.get("why") or ""),
                "waiting_on": str(payload.get("waiting_on") or ""),
                "timestamp": timestamp,
            }
            return _attach_audit_item(
                base,
                _audit_item_payload(
                    item_id=f"batch-{event.event_id}",
                    kind="handoff",
                    status="running",
                    title=title,
                    subtitle=", ".join(task_ids[:6]),
                    actor=event.agent_name,
                    target="worker agents",
                    chips=[f"{len(task_ids)} task(s)", *task_ids[:4]],
                ),
            )

        if event.event_type == "coordinator_plan_repaired":
            raw_count = _safe_int(payload.get("raw_task_count"))
            normalized_count = _safe_int(payload.get("normalized_task_count"))
            title = "Coordinator normalized the task plan"
            base = {
                "type": "plan_repair",
                "status_id": f"audit-plan-repair-{event.event_id}",
                "status_key": f"audit-plan-repair-{event.event_id}",
                "label": title,
                "detail": f"{raw_count} raw task(s) -> {normalized_count} executable task(s)",
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "status": "completed",
                "done": True,
                "hidden": False,
                "timestamp": timestamp,
            }
            return _attach_audit_item(
                base,
                _audit_item_payload(
                    item_id=f"plan-repair-{event.event_id}",
                    kind="plan",
                    status="completed",
                    title=title,
                    subtitle=base["detail"],
                    actor=event.agent_name,
                    target=str(payload.get("planner_agent") or "planner"),
                    chips=[f"{raw_count} raw", f"{normalized_count} executable"],
                ),
            )

        if event.event_type == "coordinator_finalizer_started":
            finalizer_agent = str(payload.get("finalizer_agent") or event.agent_name or "")
            detail = _runtime_event_detail(event.event_type, payload)
            base = {
                "type": "phase_start",
                "label": _PHASE_LABELS[event.event_type],
                "detail": detail,
                "agent": finalizer_agent,
                "selected_agent": event.agent_name,
                "status": "running",
                "why": str(payload.get("why") or ""),
                "timestamp": timestamp,
            }
            return _with_agent_activity(
                base,
                event=event,
                payload=payload,
                role="finalizer",
                status="running",
                title=f"Finalizer {finalizer_agent} running",
                description=detail or "Finalizer is synthesizing worker outputs into the answer.",
                agent_name=finalizer_agent,
                parent_agent=event.agent_name,
                timestamp=timestamp,
                phase="synthesizing_answer",
            )

        if event.event_type == "coordinator_verifier_started":
            verifier_agent = str(payload.get("verifier_agent") or event.agent_name or "")
            detail = _runtime_event_detail(event.event_type, payload)
            base = {
                "type": "phase_start",
                "label": _PHASE_LABELS[event.event_type],
                "detail": detail,
                "agent": verifier_agent,
                "selected_agent": event.agent_name,
                "status": "running",
                "why": str(payload.get("why") or ""),
                "timestamp": timestamp,
            }
            return _with_agent_activity(
                base,
                event=event,
                payload=payload,
                role="verifier",
                status="running",
                title=f"Verifier {verifier_agent} running",
                description=detail or "Verifier is checking the answer against the gathered evidence.",
                agent_name=verifier_agent,
                parent_agent=event.agent_name,
                timestamp=timestamp,
                phase="synthesizing_answer",
            )

        if event.event_type in _PHASE_LABELS:
            phase_type = "phase_end" if event.event_type.endswith("_completed") else "phase_start"
            translated_agent = _runtime_event_agent(event.event_type, event.agent_name, payload)
            status = "completed" if phase_type == "phase_end" else "running"
            base = {
                "type": phase_type,
                "label": _PHASE_LABELS[event.event_type],
                "detail": _runtime_event_detail(event.event_type, payload),
                "agent": translated_agent,
                "selected_agent": event.agent_name,
                "job_id": event.job_id,
                "status": status,
                "why": str(payload.get("why") or ""),
                "waiting_on": str(payload.get("waiting_on") or ""),
                "timestamp": timestamp,
            }
            if event.event_type in _AUDIT_EVENT_TYPES:
                role = (
                    "planner" if "planning" in event.event_type else
                    "finalizer" if "finalizer" in event.event_type else
                    "verifier" if "verifier" in event.event_type else
                    "basic" if "basic" in event.event_type else
                    "top_level"
                )
                return _with_agent_activity(
                    base,
                    event=event,
                    payload=payload,
                    role=role,
                    status=status,
                    title=_PHASE_LABELS[event.event_type],
                    description=_runtime_event_detail(event.event_type, payload) or _PHASE_LABELS[event.event_type],
                    agent_name=translated_agent,
                    parent_agent=event.agent_name if translated_agent != event.agent_name else "",
                    job_id=event.job_id,
                    timestamp=timestamp,
                    phase="synthesizing_answer" if role in {"finalizer", "verifier"} else _TOOL_STATUS_PHASE,
                )
            return base

        if event.event_type == "worker_agent_started":
            job_id = str(payload.get("job_id") or event.job_id or "")
            task_id = str(payload.get("task_id") or "")
            title = str(payload.get("title") or task_id or "Worker started")
            detail = str(payload.get("detail") or "")
            base = {
                "type": "worker_start",
                "label": title,
                "detail": detail,
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "job_id": job_id,
                "task_id": task_id,
                "status": "running",
                "docs": _normalize_docs(payload.get("docs") or payload.get("doc_scope") or []),
                "why": str(payload.get("why") or ""),
                "waiting_on": str(payload.get("waiting_on") or ""),
                "timestamp": timestamp,
            }
            return _with_agent_activity(
                base,
                event=event,
                payload=payload,
                role="worker",
                status="running",
                title=f"{event.agent_name} working on {task_id or title}",
                description=detail or title,
                agent_name=event.agent_name,
                parent_agent=str(payload.get("parent_agent") or payload.get("suggested_agent") or ""),
                task_id=task_id,
                job_id=job_id,
                parallel_group_id=str(payload.get("parallel_group_id") or ""),
                timestamp=timestamp,
            )

        if event.event_type == "worker_agent_completed":
            job_id = str(payload.get("job_id") or event.job_id or "")
            task_id = str(payload.get("task_id") or "")
            title = str(payload.get("title") or task_id or "Worker completed")
            detail = str(payload.get("detail") or "")
            status = str(payload.get("status") or "completed")
            base = {
                "type": "worker_end",
                "label": title,
                "detail": detail,
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "job_id": job_id,
                "task_id": task_id,
                "status": status,
                "docs": _normalize_docs(payload.get("docs") or payload.get("doc_scope") or []),
                "why": str(payload.get("why") or ""),
                "timestamp": timestamp,
            }
            return _with_agent_activity(
                base,
                event=event,
                payload=payload,
                role="worker",
                status=status,
                title=f"{event.agent_name} completed {task_id or title}",
                description=detail or title,
                agent_name=event.agent_name,
                parent_agent=str(payload.get("parent_agent") or payload.get("suggested_agent") or ""),
                task_id=task_id,
                job_id=job_id,
                parallel_group_id=str(payload.get("parallel_group_id") or ""),
                timestamp=timestamp,
            )

        if event.event_type == "worker_agent_waiting":
            message = dict(payload.get("message") or {})
            base = {
                "type": "decision_point",
                "status_id": f"audit-worker-waiting-{event.event_id}",
                "status_key": f"audit-worker-waiting-{event.event_id}",
                "label": "Worker waiting",
                "detail": str(message.get("subject") or message.get("content") or ""),
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "job_id": event.job_id,
                "status": "waiting",
                "done": False,
                "hidden": False,
                "why": str(dict(message.get("payload") or {}).get("reason") or ""),
                "waiting_on": "worker mailbox response",
                "timestamp": timestamp,
            }
            return _attach_audit_item(
                base,
                _audit_item_payload(
                    item_id=f"worker-waiting-{event.event_id}",
                    kind="handoff",
                    status="waiting",
                    title="Worker is waiting for a handoff response",
                    subtitle=base["detail"],
                    actor=event.agent_name,
                    chips=["waiting", "team mailbox"],
                ),
            )

        if event.event_type == "worker_mailbox_request_resolved":
            request = dict(payload.get("request") or {})
            base = {
                "type": "decision_point",
                "status_id": f"audit-worker-resolved-{event.event_id}",
                "status_key": f"audit-worker-resolved-{event.event_id}",
                "label": "Worker request resolved",
                "detail": str(request.get("subject") or request.get("content") or ""),
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "job_id": event.job_id,
                "status": "running",
                "done": False,
                "hidden": False,
                "why": "",
                "waiting_on": "",
                "timestamp": timestamp,
            }
            return _attach_audit_item(
                base,
                _audit_item_payload(
                    item_id=f"worker-resolved-{event.event_id}",
                    kind="handoff",
                    status="completed",
                    title="Worker handoff request resolved",
                    subtitle=base["detail"],
                    actor=event.agent_name,
                    chips=["resolved", "team mailbox"],
                ),
            )

        if event.event_type == "team_mailbox_message_posted":
            message = dict(payload.get("message") or {})
            is_handoff = message.get("message_type") == "handoff"
            base = {
                "type": "handoff_prepared" if message.get("message_type") == "handoff" else "decision_point",
                "status_id": f"audit-team-message-{event.event_id}",
                "status_key": f"audit-team-message-{event.event_id}",
                "label": "Team mailbox message",
                "detail": str(message.get("subject") or message.get("content") or ""),
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "job_id": event.job_id,
                "status": str(message.get("status") or "open"),
                "done": False,
                "hidden": False,
                "why": str(message.get("message_type") or ""),
                "waiting_on": "team mailbox" if message.get("requires_response") else "",
                "timestamp": timestamp,
            }
            return _attach_audit_item(
                base,
                _audit_item_payload(
                    item_id=f"team-message-{event.event_id}",
                    kind="handoff" if is_handoff else "decision",
                    status="running",
                    title="Team handoff posted" if is_handoff else "Team message posted",
                    subtitle=base["detail"],
                    actor=event.agent_name,
                    target=", ".join(str(item) for item in (message.get("target_agents") or []) if str(item)),
                    chips=[str(message.get("message_type") or ""), str(message.get("status") or "open")],
                ),
            )

        if event.event_type == "team_mailbox_message_resolved":
            request = dict(payload.get("request") or {})
            base = {
                "type": "decision_point",
                "status_id": f"audit-team-resolved-{event.event_id}",
                "status_key": f"audit-team-resolved-{event.event_id}",
                "label": "Team mailbox request resolved",
                "detail": str(request.get("subject") or request.get("content") or ""),
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "job_id": event.job_id,
                "status": str(request.get("status") or "resolved"),
                "done": True,
                "hidden": False,
                "why": "",
                "waiting_on": "",
                "timestamp": timestamp,
            }
            return _attach_audit_item(
                base,
                _audit_item_payload(
                    item_id=f"team-resolved-{event.event_id}",
                    kind="handoff",
                    status="completed",
                    title="Team handoff resolved",
                    subtitle=base["detail"],
                    actor=event.agent_name,
                    chips=[str(request.get("status") or "resolved")],
                ),
            )

        if event.event_type == "worker_handoff_prepared":
            artifact_type = _clean(payload.get("artifact_type") or "handoff")
            schema = _clean(payload.get("handoff_schema"))
            base = {
                "type": "handoff_prepared",
                "status_id": f"audit-handoff-prepared-{event.event_id}",
                "status_key": f"audit-handoff-prepared-{event.event_id}",
                "label": f"Prepared {artifact_type}",
                "detail": schema,
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "task_id": str(payload.get("task_id") or ""),
                "status": "completed",
                "done": True,
                "hidden": False,
                "why": "A worker packaged structured output for another worker to consume.",
                "timestamp": timestamp,
            }
            return _attach_audit_item(
                base,
                _audit_item_payload(
                    item_id=f"handoff-prepared-{event.event_id}",
                    kind="handoff",
                    status="completed",
                    title=f"{event.agent_name} prepared {artifact_type}",
                    subtitle=schema,
                    actor=event.agent_name,
                    chips=[artifact_type, schema, str(payload.get("task_id") or "")],
                    output={
                        "artifact_type": artifact_type,
                        "artifact_id": _clean(payload.get("artifact_id")),
                        "handoff_schema": schema,
                    },
                ),
            )

        if event.event_type == "worker_handoff_consumed":
            artifact_types = [str(item) for item in (payload.get("artifact_types") or []) if str(item)]
            base = {
                "type": "handoff_consumed",
                "status_id": f"audit-handoff-consumed-{event.event_id}",
                "status_key": f"audit-handoff-consumed-{event.event_id}",
                "label": "Consumed handoff artifacts",
                "detail": ", ".join(artifact_types)[:200],
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "task_id": str(payload.get("task_id") or ""),
                "status": "running",
                "done": False,
                "hidden": False,
                "why": "The worker is using structured results from an earlier task.",
                "timestamp": timestamp,
            }
            return _attach_audit_item(
                base,
                _audit_item_payload(
                    item_id=f"handoff-consumed-{event.event_id}",
                    kind="handoff",
                    status="running",
                    title=f"{event.agent_name} consumed handoff artifacts",
                    subtitle=base["detail"],
                    actor=event.agent_name,
                    chips=[*artifact_types[:4], str(payload.get("task_id") or "")],
                    input={
                        "artifact_ids": [str(item) for item in (payload.get("artifact_ids") or []) if str(item)],
                        "artifact_types": artifact_types,
                        "handoff_schema": _clean(payload.get("handoff_schema")),
                    },
                ),
            )

        if event.event_type == "peer_agent_dispatch":
            reused = bool(payload.get("reused_existing_job"))
            target_agent = str(payload.get("target_agent") or event.agent_name or "worker")
            description = str(payload.get("description") or "").strip()
            label = f"{'Continued' if reused else 'Queued'} {target_agent}"
            base = {
                "type": "peer_dispatch",
                "status_id": f"audit-peer-dispatch-{event.event_id}",
                "status_key": f"audit-peer-dispatch-{event.event_id}",
                "label": label,
                "detail": description,
                "agent": target_agent,
                "selected_agent": event.agent_name,
                "job_id": event.job_id,
                "status": "queued",
                "done": False,
                "hidden": False,
                "why": "An agent delegated a bounded follow-up to another specialist in the same session.",
                "timestamp": timestamp,
            }
            return _attach_audit_item(
                base,
                _audit_item_payload(
                    item_id=f"peer-dispatch-{event.event_id}",
                    kind="handoff",
                    status="queued",
                    title=f"{event.agent_name or 'Agent'} handed off to {target_agent}",
                    subtitle=description,
                    actor=event.agent_name,
                    target=target_agent,
                    chips=["reused" if reused else "queued", target_agent],
                ),
            )

        if event.event_type == "agent_context_loaded":
            prompt_docs = [dict(item) for item in list(payload.get("prompt_docs") or []) if isinstance(item, dict)]
            skill_docs = [dict(item) for item in list(payload.get("skill_docs") or []) if isinstance(item, dict)]
            context_sections = [
                dict(item) for item in list(payload.get("context_sections") or []) if isinstance(item, dict)
            ]
            memory_context = payload.get("memory_context") if isinstance(payload.get("memory_context"), dict) else {}
            redacted_fields = [str(item) for item in list(payload.get("redacted_fields") or []) if str(item)]
            title_agent = event.agent_name or str(payload.get("agent_name") or "agent")
            title = str(payload.get("title") or f"{title_agent} loaded prompt, skills, and context")
            detail = str(
                payload.get("detail")
                or f"{len(prompt_docs)} prompt doc(s), {len(skill_docs)} skill doc(s), {len(context_sections)} context section(s)"
            )
            audit_input = {
                "prompt_docs": prompt_docs,
                "skill_docs": skill_docs,
                "context_sections": context_sections,
                "memory_context": memory_context,
            }
            base = {
                "type": "context_trace",
                "status_id": f"context-{event.event_id}",
                "status_key": f"context-{event.event_id}",
                "description": title,
                "status": "complete",
                "done": True,
                "hidden": False,
                "elapsed_ms": 0,
                "agent": title_agent,
                "selected_agent": title_agent,
                "phase": "starting",
                "phase_label": "Starting",
                "phase_elapsed_ms": 0,
                "status_elapsed_ms": 0,
                "source_event_type": event.event_type,
                "label": title,
                "detail": detail,
                "job_id": event.job_id,
                "task_id": str(payload.get("task_id") or ""),
                "why": "The runtime assembled agent guidance before calling the model.",
                "waiting_on": "",
                "timestamp": timestamp,
                "agentic_status": _agentic_status(
                    status="completed",
                    kind="context",
                    title=title,
                    subtitle=detail,
                    chips=[
                        title_agent,
                        f"{len(prompt_docs)} prompt doc(s)",
                        f"{len(skill_docs)} skill doc(s)",
                        str(payload.get("detail_level") or ""),
                    ],
                    elapsed_ms=0,
                    timestamp=timestamp,
                ),
            }
            return _attach_audit_item(
                base,
                _audit_item_payload(
                    item_id=f"context-{event.event_id}",
                    kind="context",
                    status="completed",
                    title=title,
                    subtitle=detail,
                    actor=title_agent,
                    chips=[
                        title_agent,
                        f"{len(prompt_docs)} prompt doc(s)",
                        f"{len(skill_docs)} skill(s)",
                        f"{len(context_sections)} section(s)",
                        str(payload.get("detail_level") or ""),
                    ],
                    input=audit_input,
                    notices=[str(item) for item in list(payload.get("notices") or []) if str(item)],
                    redacted_fields=redacted_fields,
                    payload_limit_chars=int(payload.get("payload_limit_chars") or 0) or None,
                ),
            )

        if event.event_type in {
            "coverage_gate_failed",
            "coverage_backfill_triggered",
            "coordinator_revision_round_started",
            "coordinator_revision_stopped",
            "coordinator_revision_limit_reached",
        }:
            revision_round = _safe_int(payload.get("revision_round"))
            max_rounds = _safe_int(payload.get("max_revision_rounds"))
            if event.event_type == "coverage_gate_failed":
                issues = [str(item) for item in (payload.get("issues") or []) if str(item)]
                title = "Coordinator found coverage gaps"
                subtitle = "; ".join(issues[:3])
                kind = "plan"
                status = "running"
                chips = ["coverage", f"{len(issues)} issue(s)"]
                item_id = f"coverage-gate-{event.event_id}"
            elif event.event_type == "coverage_backfill_triggered":
                task_ids = [str(item) for item in (payload.get("generated_task_ids") or []) if str(item)]
                title = f"Coordinator added {len(task_ids)} backfill task(s)"
                subtitle = ", ".join(task_ids[:6])
                kind = "plan"
                status = "running"
                chips = ["backfill", *task_ids[:4]]
                item_id = f"coverage-backfill-{event.event_id}"
            elif event.event_type == "coordinator_revision_round_started":
                title = f"Verifier requested revision round {revision_round}"
                subtitle = _clean(payload.get("feedback"))
                kind = "revise"
                status = "running"
                chips = ["revise", f"round {revision_round}/{max_rounds}" if max_rounds else f"round {revision_round}"]
                item_id = f"revision-round-{event.event_id}"
            elif event.event_type == "coordinator_revision_limit_reached":
                title = "Revision limit reached"
                subtitle = _clean(payload.get("feedback"))
                kind = "revise"
                status = "completed"
                chips = ["limit reached", f"round {revision_round}/{max_rounds}" if max_rounds else f"round {revision_round}"]
                item_id = f"revision-limit-{event.event_id}"
            else:
                title = "Coordinator stopped revision loop"
                subtitle = _clean(payload.get("reason"))
                kind = "revise"
                status = "completed"
                chips = ["stopped", subtitle, f"round {revision_round}/{max_rounds}" if max_rounds else f"round {revision_round}"]
                item_id = f"revision-stopped-{event.event_id}"
            base = {
                "type": "revision" if kind == "revise" else "coverage",
                "status_id": f"audit-{item_id}",
                "status_key": f"audit-{item_id}",
                "label": title,
                "detail": subtitle,
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "status": status,
                "done": _audit_status(status) == "completed",
                "hidden": False,
                "why": subtitle,
                "timestamp": timestamp,
            }
            return _attach_audit_item(
                base,
                _audit_item_payload(
                    item_id=item_id,
                    kind=kind,
                    status=status,
                    title=title,
                    subtitle=subtitle,
                    actor=event.agent_name,
                    chips=chips,
                    notices=[subtitle] if subtitle else [],
                ),
            )

        if event.event_type == "turn_completed":
            base = {
                "type": "summary",
                "status_id": f"audit-turn-completed-{event.event_id}",
                "status_key": f"audit-turn-completed-{event.event_id}",
                "label": "Answer ready",
                "detail": "",
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "status": "completed",
                "done": True,
                "hidden": False,
                "timestamp": timestamp,
            }
            return _attach_audit_item(
                base,
                _audit_item_payload(
                    item_id=f"turn-completed-{event.event_id}",
                    kind="complete",
                    status="completed",
                    title="Answer ready",
                    actor=event.agent_name,
                    chips=["ready"],
                ),
            )

        return None


def _tool_status_from_runtime_event(
    event: RuntimeEvent,
    payload: Dict[str, Any],
    *,
    timestamp: int,
) -> Dict[str, Any]:
    tool_call_id = str(payload.get("tool_call_id") or payload.get("run_id") or event.event_id or "").strip()
    tool_name = str(payload.get("tool_name") or event.tool_name or "tool").strip() or "tool"
    status = str(payload.get("status") or "").strip().lower()
    if event.event_type == "tool_start":
        status = "running"
    elif event.event_type == "tool_error":
        status = "error"
    elif not status:
        status = "completed"
    done = status in {"completed", "error"}
    state = "error" if status == "error" else ("complete" if done else "active")
    title = f"{tool_name} {'failed' if status == 'error' else ('completed' if done else 'running')}"
    duration_ms = payload.get("duration_ms")
    if not isinstance(duration_ms, (int, float)):
        duration_ms = None
    elapsed_ms = int(duration_ms or 0)
    tool_payload = {
        "version": 1,
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "agent_name": str(event.agent_name or payload.get("agent_name") or ""),
        "parent_agent": str(payload.get("parent_agent") or ""),
        "parallel_group_id": str(payload.get("parallel_group_id") or ""),
        "job_id": str(event.job_id or payload.get("job_id") or ""),
        "status": status,
        "started_at": str(payload.get("started_at") or ""),
        "completed_at": str(payload.get("completed_at") or ""),
        "duration_ms": elapsed_ms if duration_ms is not None else None,
        "input_preview": str(payload.get("input_preview") or ""),
        "output_preview": str(payload.get("output_preview") or ""),
        "input": payload.get("input"),
        "output": payload.get("output"),
        "error": payload.get("error"),
        "error_preview": str(payload.get("error_preview") or ""),
        "truncated": bool(payload.get("truncated")),
        "truncated_fields": [str(item) for item in list(payload.get("truncated_fields") or []) if str(item)],
        "redacted_fields": [str(item) for item in list(payload.get("redacted_fields") or []) if str(item)],
        "payload_limit_chars": int(payload.get("payload_limit_chars") or 0),
        "source_event_id": event.event_id,
    }
    status_id = f"tool-{tool_call_id or event.event_id}"
    subtitle = tool_payload["output_preview"] or tool_payload["error_preview"] or tool_payload["input_preview"]
    notices: List[str] = []
    if tool_payload["truncated"]:
        notices.append(f"Payload limited to {tool_payload['payload_limit_chars']} characters.")
    if tool_payload["redacted_fields"]:
        notices.append("Sensitive fields were redacted.")
    base = {
        "type": "tool_trace",
        "status_id": status_id,
        "status_key": status_id,
        "description": title,
        "status": "error" if status == "error" else ("complete" if done else "in_progress"),
        "done": done,
        "hidden": False,
        "elapsed_ms": elapsed_ms,
        "agent": tool_payload["agent_name"],
        "selected_agent": tool_payload["agent_name"],
        "phase": _TOOL_STATUS_PHASE,
        "phase_label": "Searching knowledge base",
        "phase_elapsed_ms": elapsed_ms,
        "status_elapsed_ms": elapsed_ms,
        "source_event_type": event.event_type,
        "label": title,
        "detail": subtitle,
        "job_id": tool_payload["job_id"],
        "task_id": "",
        "why": "A runtime agent invoked a tool.",
        "waiting_on": "",
        "timestamp": timestamp,
        "agentic_tool_call": tool_payload,
        "agentic_status": {
            "version": 1,
            "state": state,
            "kind": "tool",
            "title": title,
            "subtitle": subtitle,
            "chips": [chip for chip in (tool_payload["agent_name"], tool_name, status.title()) if chip],
            "timing": {
                "kind": "total" if done else "stage",
                "live": not done,
                "elapsed_ms": elapsed_ms,
                "status_elapsed_ms": elapsed_ms,
                "phase_elapsed_ms": elapsed_ms,
                "total_elapsed_ms": elapsed_ms,
                "snapshot_timestamp_ms": timestamp,
            },
        },
    }
    return _attach_audit_item(
        base,
        _audit_item_payload(
            item_id=status_id,
            kind="tool",
            status=status,
            title=f"{tool_name} {'failed' if status == 'error' else ('ran' if done else 'running')}",
            subtitle=subtitle,
            actor=tool_payload["parent_agent"] or tool_payload["agent_name"],
            target=tool_name,
            group_id=tool_payload["parallel_group_id"],
            chips=[
                tool_payload["agent_name"],
                tool_name,
                _status_label(status),
                f"{elapsed_ms}ms" if duration_ms is not None else "",
            ],
            started_at=tool_payload["started_at"],
            completed_at=tool_payload["completed_at"],
            duration_ms=tool_payload["duration_ms"],
            input=tool_payload["input"],
            output=tool_payload["output"],
            error=tool_payload["error"],
            notices=notices,
            redacted_fields=tool_payload["redacted_fields"],
            payload_limit_chars=tool_payload["payload_limit_chars"] or None,
        ),
    )


def _runtime_event_detail(event_type: str, payload: Dict[str, Any]) -> str:
    if event_type == "coordinator_planning_completed":
        return f"{int(payload.get('task_count') or 0)} task(s)"
    if event_type == "coordinator_batch_started":
        task_ids = [str(item) for item in (payload.get("task_ids") or []) if str(item)]
        return ", ".join(task_ids[:4])
    if event_type == "coordinator_finalizer_started":
        revision_round = int(payload.get("revision_round") or 0)
        return f"Revision round {revision_round}" if revision_round else ""
    if event_type == "coordinator_verifier_started":
        revision_round = int(payload.get("revision_round") or 0)
        return f"Revision round {revision_round}" if revision_round else ""
    if event_type == "coordinator_verifier_completed":
        status = str(payload.get("status") or "")
        return status if status else ""
    return ""


def _runtime_event_agent(event_type: str, fallback_agent: str, payload: Dict[str, Any]) -> str:
    if event_type in {"coordinator_planning_started", "coordinator_planning_completed"}:
        return str(payload.get("planner_agent") or fallback_agent or "")
    if event_type in {"coordinator_finalizer_started", "coordinator_finalizer_completed"}:
        return str(payload.get("finalizer_agent") or fallback_agent or "")
    if event_type in {"coordinator_verifier_started", "coordinator_verifier_completed"}:
        return str(payload.get("verifier_agent") or fallback_agent or "")
    return str(fallback_agent or "")


def _normalize_docs(raw_docs: Iterable[Any]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for item in raw_docs or []:
        if isinstance(item, dict):
            docs.append(
                {
                    "doc_id": str(item.get("doc_id") or ""),
                    "title": str(item.get("title") or ""),
                    "source_path": str(item.get("source_path") or ""),
                    "source_type": str(item.get("source_type") or ""),
                }
            )
            continue
        value = str(item or "").strip()
        if value:
            docs.append({"doc_id": value, "title": value, "source_path": "", "source_type": ""})
    return docs
