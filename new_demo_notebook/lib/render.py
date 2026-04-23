from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from new_demo_notebook.lib.client import GatewayStreamEvent, GatewayStreamResult
from new_demo_notebook.lib.scenario_runner import ScenarioDefinition, ScenarioResult
from new_demo_notebook.lib.trace_reader import TraceBundle, extract_latest_router_decision


def _display_markdown(text: str) -> None:
    try:
        from IPython.display import Markdown, display

        display(Markdown(text))
    except Exception:
        print(text)


def _display_pretty(value: Any) -> None:
    try:
        from IPython.display import JSON, display

        display(JSON(value))
    except Exception:
        print(json.dumps(value, indent=2, ensure_ascii=False))


def build_event_rows(bundle: TraceBundle, trace_focus: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    rows = list(bundle.event_rows + bundle.job_events)
    if not trace_focus:
        return rows
    focus = {item.lower() for item in trace_focus}
    filtered = []
    for row in rows:
        agent_name = str(row.get("agent_name") or "").lower()
        tool_name = str(row.get("tool_name") or "").lower()
        event_type = str(row.get("event_type") or "").lower()
        if agent_name in focus or tool_name in focus or event_type in focus:
            filtered.append(row)
    return filtered


def build_job_rows(bundle: TraceBundle) -> List[Dict[str, Any]]:
    return [
        {
            "job_id": str(job.get("job_id") or ""),
            "agent_name": str(job.get("agent_name") or ""),
            "status": str(job.get("status") or ""),
            "description": str(job.get("description") or ""),
            "result_summary": str(job.get("result_summary") or ""),
        }
        for job in bundle.jobs
    ]


def build_progress_rows(progress_events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for event in progress_events:
        event_type = str(event.get("type") or event.get("event_type") or "")
        if event_type in {"tool_call", "tool_result", "tool_error"}:
            continue
        rows.append(
            {
                "type": event_type,
                "label": str(event.get("label") or ""),
                "detail": str(event.get("detail") or ""),
                "agent": str(event.get("agent") or ""),
                "task_id": str(event.get("task_id") or ""),
                "job_id": str(event.get("job_id") or ""),
                "status": str(event.get("status") or ""),
                "why": str(event.get("why") or ""),
                "waiting_on": str(event.get("waiting_on") or ""),
                "docs": list(event.get("docs") or []),
            }
        )
    return rows


def build_tool_rows(progress_events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for event in progress_events:
        event_type = str(event.get("type") or event.get("event_type") or "")
        if event_type not in {"tool_intent", "tool_call", "tool_result", "tool_error"}:
            continue
        rows.append(
            {
                "type": event_type,
                "tool": str(event.get("tool") or ""),
                "label": str(event.get("label") or ""),
                "detail": str(event.get("detail") or ""),
                "why": str(event.get("why") or ""),
                "input": event.get("input"),
                "output": event.get("output"),
                "error": str(event.get("error") or ""),
                "duration_ms": event.get("duration_ms"),
            }
        )
    return rows


def build_artifact_rows(artifacts: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "filename": str(artifact.get("filename") or ""),
            "label": str(artifact.get("label") or ""),
            "download_url": str(artifact.get("download_url") or ""),
            "content_type": str(artifact.get("content_type") or ""),
            "size_bytes": artifact.get("size_bytes"),
        }
        for artifact in artifacts
        if isinstance(artifact, dict)
    ]


def build_metadata_rows(metadata: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = dict(metadata or {})
    return {
        "job_id": str(payload.get("job_id") or ""),
        "long_output": dict(payload.get("long_output") or {}),
    }


def build_workspace_preview(bundle: TraceBundle, filename: str, *, max_chars: int = 1200) -> Dict[str, Any]:
    wanted = str(filename or "").strip()
    if not wanted:
        return {}
    for session_id, root in bundle.workspace_roots.items():
        candidate = Path(root) / wanted
        if not candidate.exists() or not candidate.is_file():
            continue
        raw_text = candidate.read_text(encoding="utf-8")
        preview = raw_text[:max_chars]
        if len(raw_text) > max_chars:
            preview += "\n..."
        return {
            "session_id": session_id,
            "filename": wanted,
            "path": str(candidate),
            "preview": preview,
            "size_bytes": candidate.stat().st_size,
        }
    return {}


def stream_event_console_line(event: GatewayStreamEvent) -> str:
    if event.kind == "progress" and isinstance(event.payload, dict):
        payload = dict(event.payload)
        event_type = str(payload.get("type") or payload.get("event_type") or "progress")
        label = str(payload.get("label") or event_type)
        detail = str(payload.get("detail") or "")
        why = str(payload.get("why") or "")
        waiting_on = str(payload.get("waiting_on") or "")
        extras = " | ".join(item for item in [detail, why, waiting_on] if item)
        return f"[progress:{event_type}] {label}" + (f" -- {extras}" if extras else "")
    if event.kind == "content":
        return event.text_delta
    if event.kind == "artifacts":
        count = len(event.payload) if isinstance(event.payload, list) else 0
        return f"[artifacts] {count} file(s)"
    if event.kind == "metadata" and isinstance(event.payload, dict):
        payload = dict(event.payload)
        long_output = dict(payload.get("long_output") or {})
        output_filename = str(long_output.get("output_filename") or "")
        section_count = long_output.get("section_count")
        detail = ", ".join(
            item
            for item in [
                f"job_id={payload.get('job_id')}" if payload.get("job_id") else "",
                f"output={output_filename}" if output_filename else "",
                f"sections={section_count}" if section_count else "",
            ]
            if item
        )
        return f"[metadata] {detail}" if detail else "[metadata]"
    if event.kind == "done":
        return "[done]"
    return f"[{event.kind}] {event.payload}"


def display_stream_result(
    result: GatewayStreamResult,
    *,
    title: str = "Streamed Demo",
    trace_bundle: Optional[TraceBundle] = None,
    trace_focus: Optional[List[str]] = None,
) -> None:
    _display_markdown(f"## {title}")
    _display_markdown("### Final Assistant Output")
    _display_markdown(result.text or "_No assistant text returned._")
    if result.progress_events:
        _display_markdown("### Live Thought Timeline")
        _display_pretty(build_progress_rows(result.progress_events))
    if build_tool_rows(result.progress_events):
        _display_markdown("### Tool Calls And Logs")
        _display_pretty(build_tool_rows(result.progress_events))
    if result.artifacts:
        _display_markdown("### Returned Artifacts")
        _display_pretty(build_artifact_rows(result.artifacts))
    if result.metadata:
        _display_markdown("### Response Metadata")
        _display_pretty(build_metadata_rows(result.metadata))
    if trace_bundle is not None:
        display_trace_bundle(trace_bundle, trace_focus=trace_focus)


def display_trace_bundle(bundle: TraceBundle, *, trace_focus: Optional[List[str]] = None) -> None:
    _display_markdown(f"### Trace Summary: `{bundle.conversation_id}`")
    _display_pretty(
        {
            "conversation_id": bundle.conversation_id,
            "session_ids": bundle.session_ids,
            "event_count": len(bundle.event_rows) + len(bundle.job_events),
            "job_count": len(bundle.jobs),
            "notification_count": len(bundle.notifications),
            "workspace_files": bundle.workspace_files,
        }
    )
    _display_markdown("#### Event Timeline")
    _display_pretty(build_event_rows(bundle, trace_focus=trace_focus))
    _display_markdown("#### Jobs")
    _display_pretty(build_job_rows(bundle))
    _display_markdown("#### Transcript Excerpts")
    _display_pretty(bundle.transcript_rows[-6:])


def display_scenario_result(result: ScenarioResult) -> None:
    latest = result.attempts[-1]
    router_decision = extract_latest_router_decision(result.bundle)
    _display_markdown(
        f"## {result.scenario.title}\n"
        f"- success: `{latest.success}`\n"
        f"- observed route: `{latest.observed_route}`\n"
        f"- observed agents: `{', '.join(latest.observed_agents)}`"
    )
    if any(
        router_decision.get(key)
        for key in ("route", "router_method", "suggested_agent")
    ) or router_decision.get("reasons"):
        _display_markdown("### Latest Router Decision")
        _display_pretty(router_decision)
    if latest.validation_errors:
        _display_markdown("### Validation Errors")
        _display_pretty(latest.validation_errors)
    _display_markdown("### Assistant Outputs")
    _display_pretty(latest.outputs)
    display_trace_bundle(result.bundle, trace_focus=result.scenario.trace_focus)


def display_coverage_matrix(scenarios: Iterable[ScenarioDefinition]) -> None:
    rows = [
        {
            "scenario_id": scenario.id,
            "title": scenario.title,
            "expected_agents": scenario.expected_agents,
            "expected_route": scenario.expected_route,
        }
        for scenario in scenarios
    ]
    _display_markdown("## Agent Coverage Matrix")
    _display_pretty(rows)
