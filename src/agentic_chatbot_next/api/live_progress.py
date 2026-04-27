from __future__ import annotations

import queue
import time
from typing import Any, Dict, Iterable, List

from agentic_chatbot_next.observability.events import RuntimeEvent
from agentic_chatbot_next.runtime.event_sink import RuntimeEventSink

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


def _now_ms() -> int:
    return int(time.time() * 1000)


class LiveProgressSink(RuntimeEventSink):
    def __init__(self) -> None:
        self.events: queue.Queue = queue.Queue()

    def mark_done(self) -> None:
        self.events.put(None)

    def emit_progress(self, event_type: str, **payload: Any) -> None:
        event = {"type": str(event_type), "timestamp": int(payload.pop("timestamp", _now_ms())), **payload}
        self.events.put(event)

    def emit(self, event: RuntimeEvent) -> None:
        translated = self._translate_runtime_event(event)
        if translated is not None:
            self.events.put(translated)

    def _translate_runtime_event(self, event: RuntimeEvent) -> Dict[str, Any] | None:
        payload = dict(event.payload or {})
        timestamp = _now_ms()

        if event.event_type in {"tool_start", "tool_end", "tool_error"}:
            return _tool_status_from_runtime_event(event, payload, timestamp=timestamp)

        if event.event_type == "router_decision":
            reasons = [str(item) for item in (payload.get("reasons") or []) if str(item)]
            detail = reasons[0] if reasons else f"Route {payload.get('route') or ''}".strip()
            return {
                "type": "route_decision",
                "label": f"Routed to {payload.get('suggested_agent') or payload.get('route') or 'agent'}",
                "detail": detail,
                "agent": str(payload.get("suggested_agent") or ""),
                "selected_agent": str(payload.get("suggested_agent") or ""),
                "status": str(payload.get("route") or ""),
                "why": detail,
                "counts": {
                    "confidence": float(payload.get("confidence") or 0.0),
                },
                "timestamp": timestamp,
            }

        if event.event_type == "agent_run_started":
            return {
                "type": "agent_selected",
                "label": f"Running {event.agent_name or 'agent'}",
                "detail": str(payload.get("mode") or ""),
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "status": "running",
                "why": str(payload.get("why") or ""),
                "timestamp": timestamp,
            }

        if event.event_type == "coordinator_planning_started":
            planner_agent = str(payload.get("planner_agent") or event.agent_name or "")
            return {
                "type": "phase_start",
                "label": _PHASE_LABELS[event.event_type],
                "detail": str(payload.get("detail") or ""),
                "agent": planner_agent,
                "selected_agent": event.agent_name,
                "status": "running",
                "why": str(payload.get("why") or ""),
                "timestamp": timestamp,
            }

        if event.event_type == "coordinator_finalizer_started":
            finalizer_agent = str(payload.get("finalizer_agent") or event.agent_name or "")
            return {
                "type": "phase_start",
                "label": _PHASE_LABELS[event.event_type],
                "detail": _runtime_event_detail(event.event_type, payload),
                "agent": finalizer_agent,
                "selected_agent": event.agent_name,
                "status": "running",
                "why": str(payload.get("why") or ""),
                "timestamp": timestamp,
            }

        if event.event_type == "coordinator_verifier_started":
            verifier_agent = str(payload.get("verifier_agent") or event.agent_name or "")
            return {
                "type": "phase_start",
                "label": _PHASE_LABELS[event.event_type],
                "detail": _runtime_event_detail(event.event_type, payload),
                "agent": verifier_agent,
                "selected_agent": event.agent_name,
                "status": "running",
                "why": str(payload.get("why") or ""),
                "timestamp": timestamp,
            }

        if event.event_type in _PHASE_LABELS:
            phase_type = "phase_end" if event.event_type.endswith("_completed") else "phase_start"
            translated_agent = _runtime_event_agent(event.event_type, event.agent_name, payload)
            return {
                "type": phase_type,
                "label": _PHASE_LABELS[event.event_type],
                "detail": _runtime_event_detail(event.event_type, payload),
                "agent": translated_agent,
                "selected_agent": event.agent_name,
                "job_id": event.job_id,
                "status": "completed" if phase_type == "phase_end" else "running",
                "why": str(payload.get("why") or ""),
                "waiting_on": str(payload.get("waiting_on") or ""),
                "timestamp": timestamp,
            }

        if event.event_type == "worker_agent_started":
            return {
                "type": "worker_start",
                "label": str(payload.get("title") or payload.get("task_id") or "Worker started"),
                "detail": str(payload.get("detail") or ""),
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "job_id": str(payload.get("job_id") or event.job_id or ""),
                "task_id": str(payload.get("task_id") or ""),
                "status": "running",
                "docs": _normalize_docs(payload.get("docs") or payload.get("doc_scope") or []),
                "why": str(payload.get("why") or ""),
                "waiting_on": str(payload.get("waiting_on") or ""),
                "timestamp": timestamp,
            }

        if event.event_type == "worker_agent_completed":
            return {
                "type": "worker_end",
                "label": str(payload.get("title") or payload.get("task_id") or "Worker completed"),
                "detail": str(payload.get("detail") or ""),
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "job_id": str(payload.get("job_id") or event.job_id or ""),
                "task_id": str(payload.get("task_id") or ""),
                "status": str(payload.get("status") or "completed"),
                "docs": _normalize_docs(payload.get("docs") or payload.get("doc_scope") or []),
                "why": str(payload.get("why") or ""),
                "timestamp": timestamp,
            }

        if event.event_type == "worker_agent_waiting":
            message = dict(payload.get("message") or {})
            return {
                "type": "decision_point",
                "label": "Worker waiting",
                "detail": str(message.get("subject") or message.get("content") or ""),
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "job_id": event.job_id,
                "status": "waiting",
                "why": str(dict(message.get("payload") or {}).get("reason") or ""),
                "waiting_on": "worker mailbox response",
                "timestamp": timestamp,
            }

        if event.event_type == "worker_mailbox_request_resolved":
            request = dict(payload.get("request") or {})
            return {
                "type": "decision_point",
                "label": "Worker request resolved",
                "detail": str(request.get("subject") or request.get("content") or ""),
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "job_id": event.job_id,
                "status": "running",
                "why": "",
                "waiting_on": "",
                "timestamp": timestamp,
            }

        if event.event_type == "team_mailbox_message_posted":
            message = dict(payload.get("message") or {})
            return {
                "type": "handoff_prepared" if message.get("message_type") == "handoff" else "decision_point",
                "label": "Team mailbox message",
                "detail": str(message.get("subject") or message.get("content") or ""),
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "job_id": event.job_id,
                "status": str(message.get("status") or "open"),
                "why": str(message.get("message_type") or ""),
                "waiting_on": "team mailbox" if message.get("requires_response") else "",
                "timestamp": timestamp,
            }

        if event.event_type == "team_mailbox_message_resolved":
            request = dict(payload.get("request") or {})
            return {
                "type": "decision_point",
                "label": "Team mailbox request resolved",
                "detail": str(request.get("subject") or request.get("content") or ""),
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "job_id": event.job_id,
                "status": str(request.get("status") or "resolved"),
                "why": "",
                "waiting_on": "",
                "timestamp": timestamp,
            }

        if event.event_type == "worker_handoff_prepared":
            return {
                "type": "handoff_prepared",
                "label": f"Prepared {payload.get('artifact_type') or 'handoff'}",
                "detail": str(payload.get("handoff_schema") or ""),
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "task_id": str(payload.get("task_id") or ""),
                "status": "completed",
                "why": "A worker packaged structured output for another worker to consume.",
                "timestamp": timestamp,
            }

        if event.event_type == "worker_handoff_consumed":
            return {
                "type": "handoff_consumed",
                "label": "Consumed handoff artifacts",
                "detail": ", ".join(str(item) for item in (payload.get("artifact_types") or []) if str(item))[:200],
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "task_id": str(payload.get("task_id") or ""),
                "status": "running",
                "why": "The worker is using structured results from an earlier task.",
                "timestamp": timestamp,
            }

        if event.event_type == "peer_agent_dispatch":
            reused = bool(payload.get("reused_existing_job"))
            target_agent = str(payload.get("target_agent") or event.agent_name or "worker")
            description = str(payload.get("description") or "").strip()
            label = f"{'Continued' if reused else 'Queued'} {target_agent}"
            return {
                "type": "peer_dispatch",
                "label": label,
                "detail": description,
                "agent": target_agent,
                "selected_agent": event.agent_name,
                "job_id": event.job_id,
                "status": "queued",
                "why": "An agent delegated a bounded follow-up to another specialist in the same session.",
                "timestamp": timestamp,
            }

        if event.event_type == "turn_completed":
            return {
                "type": "summary",
                "label": "Answer ready",
                "detail": "",
                "agent": event.agent_name,
                "selected_agent": event.agent_name,
                "status": "completed",
                "timestamp": timestamp,
            }

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
        "truncated": bool(payload.get("truncated")),
        "truncated_fields": [str(item) for item in list(payload.get("truncated_fields") or []) if str(item)],
        "source_event_id": event.event_id,
    }
    status_id = f"tool-{tool_call_id or event.event_id}"
    subtitle = tool_payload["output_preview"] or tool_payload["input_preview"]
    return {
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
