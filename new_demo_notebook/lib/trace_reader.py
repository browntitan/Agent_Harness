from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    return dict(json.loads(raw)) if raw else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(dict(json.loads(line)))
    return rows


def _conversation_from_session_id(session_id: str) -> str:
    if ":" not in session_id:
        return session_id
    return session_id.split(":", 2)[-1]


@dataclass
class TraceBundle:
    conversation_id: str
    session_ids: List[str] = field(default_factory=list)
    session_states: List[Dict[str, Any]] = field(default_factory=list)
    transcript_rows: List[Dict[str, Any]] = field(default_factory=list)
    event_rows: List[Dict[str, Any]] = field(default_factory=list)
    notifications: List[Dict[str, Any]] = field(default_factory=list)
    jobs: List[Dict[str, Any]] = field(default_factory=list)
    job_events: List[Dict[str, Any]] = field(default_factory=list)
    job_transcripts: List[Dict[str, Any]] = field(default_factory=list)
    workspace_roots: Dict[str, str] = field(default_factory=dict)
    workspace_files: Dict[str, List[str]] = field(default_factory=dict)


def matching_session_dirs(runtime_root: Path, conversation_id: str) -> List[Path]:
    sessions_root = runtime_root / "sessions"
    matches: List[Path] = []
    if not sessions_root.exists():
        return matches
    for state_path in sorted(sessions_root.glob("*/state.json")):
        state = _read_json(state_path)
        state_conversation_id = str(state.get("conversation_id") or "")
        state_session_id = str(state.get("session_id") or "")
        if state_conversation_id == conversation_id or _conversation_from_session_id(state_session_id) == conversation_id:
            matches.append(state_path.parent)
    return matches


def matching_job_dirs(runtime_root: Path, session_ids: List[str], conversation_id: str) -> List[Path]:
    jobs_root = runtime_root / "jobs"
    matches: List[Path] = []
    if not jobs_root.exists():
        return matches
    session_id_set = set(session_ids)
    for state_path in sorted(jobs_root.glob("*/state.json")):
        state = _read_json(state_path)
        state_session_id = str(state.get("session_id") or "")
        session_payload = dict(state.get("metadata", {}).get("session_state", {}) or {})
        state_conversation_id = str(session_payload.get("conversation_id") or "")
        if state_session_id in session_id_set or state_conversation_id == conversation_id:
            matches.append(state_path.parent)
    return matches


def normalize_event_row(raw: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(raw.get("payload") or {})
    return {
        "created_at": str(raw.get("created_at") or ""),
        "session_id": str(raw.get("session_id") or ""),
        "conversation_id": str(payload.get("conversation_id") or ""),
        "event_type": str(raw.get("event_type") or ""),
        "route": str(payload.get("route") or ""),
        "confidence": payload.get("confidence"),
        "reasons": list(payload.get("reasons") or []),
        "router_method": str(payload.get("router_method") or ""),
        "suggested_agent": str(payload.get("suggested_agent") or ""),
        "agent_name": str(raw.get("agent_name") or ""),
        "job_id": str(raw.get("job_id") or ""),
        "tool_name": str(raw.get("tool_name") or ""),
        "payload": payload,
    }


def collect_trace_bundle(runtime_root: Path, workspace_root: Path, conversation_id: str) -> TraceBundle:
    session_dirs = matching_session_dirs(runtime_root, conversation_id)
    session_ids: List[str] = []
    session_states: List[Dict[str, Any]] = []
    transcript_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    notifications: List[Dict[str, Any]] = []
    workspace_roots: Dict[str, str] = {}
    workspace_files: Dict[str, List[str]] = {}

    for session_dir in session_dirs:
        state = _read_json(session_dir / "state.json")
        if state:
            session_id = str(state.get("session_id") or session_dir.name)
            session_ids.append(session_id)
            session_states.append(state)
            workspace_root_raw = str(state.get("workspace_root") or "")
            if workspace_root_raw:
                workspace_roots[session_id] = workspace_root_raw
                workspace_path = Path(workspace_root_raw)
                if workspace_path.exists():
                    workspace_files[session_id] = sorted(
                        item.name
                        for item in workspace_path.iterdir()
                        if item.is_file() and item.name != ".meta"
                    )
        transcript_rows.extend(_read_jsonl(session_dir / "transcript.jsonl"))
        event_rows.extend(normalize_event_row(row) for row in _read_jsonl(session_dir / "events.jsonl"))
        notifications.extend(_read_jsonl(session_dir / "notifications.jsonl"))

    if workspace_root.exists():
        for workspace_dir in workspace_root.iterdir():
            if not workspace_dir.is_dir():
                continue
            meta = _read_json(workspace_dir / ".meta")
            session_id = str(meta.get("session_id") or "")
            if session_id and session_id not in workspace_files and _conversation_from_session_id(session_id) == conversation_id:
                workspace_roots[session_id] = str(workspace_dir)
                workspace_files[session_id] = sorted(
                    item.name
                    for item in workspace_dir.iterdir()
                    if item.is_file() and item.name != ".meta"
                )

    job_dirs = matching_job_dirs(runtime_root, session_ids, conversation_id)
    jobs: List[Dict[str, Any]] = []
    job_events: List[Dict[str, Any]] = []
    job_transcripts: List[Dict[str, Any]] = []
    for job_dir in job_dirs:
        state = _read_json(job_dir / "state.json")
        if state:
            jobs.append(state)
        job_events.extend(normalize_event_row(row) for row in _read_jsonl(job_dir / "events.jsonl"))
        job_transcripts.extend(_read_jsonl(job_dir / "transcript.jsonl"))

    event_rows.sort(key=lambda row: row["created_at"])
    job_events.sort(key=lambda row: row["created_at"])
    jobs.sort(key=lambda row: str(row.get("created_at") or ""))

    return TraceBundle(
        conversation_id=conversation_id,
        session_ids=session_ids,
        session_states=session_states,
        transcript_rows=transcript_rows,
        event_rows=event_rows,
        notifications=notifications,
        jobs=jobs,
        job_events=job_events,
        job_transcripts=job_transcripts,
        workspace_roots=workspace_roots,
        workspace_files=workspace_files,
    )


def extract_observed_agents(bundle: TraceBundle) -> List[str]:
    agents = {
        str(row.get("agent_name") or "")
        for row in bundle.event_rows + bundle.job_events
        if str(row.get("agent_name") or "")
    }
    for row in bundle.event_rows + bundle.job_events:
        payload = dict(row.get("payload") or {})
        for key in ("planner_agent", "finalizer_agent", "verifier_agent"):
            value = str(payload.get(key) or "")
            if value:
                agents.add(value)
    agents.update(
        str(job.get("agent_name") or "")
        for job in bundle.jobs
        if str(job.get("agent_name") or "")
    )
    return sorted(agents)


def extract_observed_event_types(bundle: TraceBundle) -> List[str]:
    return sorted({str(row.get("event_type") or "") for row in bundle.event_rows + bundle.job_events if str(row.get("event_type") or "")})


def extract_observed_route(bundle: TraceBundle) -> str:
    for row in reversed(bundle.event_rows):
        if row["event_type"] == "router_decision":
            return row["route"]
    return ""


def extract_latest_router_decision(bundle: TraceBundle) -> Dict[str, Any]:
    for row in reversed(bundle.event_rows):
        if row["event_type"] != "router_decision":
            continue
        return {
            "route": str(row.get("route") or ""),
            "confidence": row.get("confidence"),
            "reasons": list(row.get("reasons") or []),
            "router_method": str(row.get("router_method") or ""),
            "suggested_agent": str(row.get("suggested_agent") or ""),
        }
    return {
        "route": "",
        "confidence": None,
        "reasons": [],
        "router_method": "",
        "suggested_agent": "",
    }


def cleanup_conversation_artifacts(runtime_root: Path, workspace_root: Path, conversation_id: str) -> None:
    session_dirs = matching_session_dirs(runtime_root, conversation_id)
    session_ids = [str(_read_json(path / "state.json").get("session_id") or path.name) for path in session_dirs]
    for session_dir in session_dirs:
        shutil.rmtree(session_dir, ignore_errors=True)

    for job_dir in matching_job_dirs(runtime_root, session_ids, conversation_id):
        shutil.rmtree(job_dir, ignore_errors=True)

    if workspace_root.exists():
        for workspace_dir in workspace_root.iterdir():
            if not workspace_dir.is_dir():
                continue
            meta = _read_json(workspace_dir / ".meta")
            session_id = str(meta.get("session_id") or workspace_dir.name)
            if _conversation_from_session_id(session_id) == conversation_id:
                shutil.rmtree(workspace_dir, ignore_errors=True)
