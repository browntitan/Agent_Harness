from __future__ import annotations

import mimetypes
import uuid
from pathlib import Path
from typing import Any, Dict, List

from agentic_chatbot_next.contracts.messages import utc_now_iso


def _metadata(session: Any) -> Dict[str, Any]:
    raw = getattr(session, "metadata", None)
    if isinstance(raw, dict):
        return raw
    value: Dict[str, Any] = {}
    session.metadata = value
    return value


def normalize_artifact(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "download_id": str(raw.get("download_id") or ""),
        "artifact_ref": str(raw.get("artifact_ref") or ""),
        "filename": str(raw.get("filename") or ""),
        "label": str(raw.get("label") or ""),
        "download_url": str(raw.get("download_url") or ""),
        "content_type": str(raw.get("content_type") or "application/octet-stream"),
        "size_bytes": int(raw.get("size_bytes") or 0),
        "session_id": str(raw.get("session_id") or ""),
        "conversation_id": str(raw.get("conversation_id") or ""),
    }


def normalize_handoff_artifact(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "artifact_id": str(raw.get("artifact_id") or ""),
        "artifact_ref": str(raw.get("artifact_ref") or ""),
        "artifact_type": str(raw.get("artifact_type") or ""),
        "handoff_schema": str(raw.get("handoff_schema") or ""),
        "producer_task_id": str(raw.get("producer_task_id") or ""),
        "producer_agent": str(raw.get("producer_agent") or ""),
        "summary": str(raw.get("summary") or ""),
        "created_at": str(raw.get("created_at") or utc_now_iso()),
        "allowed_consumers": [str(item) for item in (raw.get("allowed_consumers") or []) if str(item)],
        "source_artifact_ids": [str(item) for item in (raw.get("source_artifact_ids") or []) if str(item)],
        "data": dict(raw.get("data") or {}),
    }


def register_workspace_artifact(session: Any, *, filename: str, label: str = "") -> Dict[str, Any]:
    workspace = getattr(session, "workspace", None)
    if workspace is None:
        workspace_root = str(getattr(session, "workspace_root", "") or "").strip()
        session_id = str(getattr(session, "session_id", "") or "")
        if workspace_root and session_id:
            from agentic_chatbot_next.sandbox.workspace import SessionWorkspace

            workspace = SessionWorkspace(session_id=session_id, root=Path(workspace_root))
            workspace.open()
        else:
            raise ValueError("No session workspace is available.")
    if not workspace.exists(filename):
        raise FileNotFoundError(f"Workspace file not found: {filename!r}")

    path = workspace.root / filename
    download_id = f"dl_{uuid.uuid4().hex[:16]}"
    artifact = normalize_artifact(
        {
            "download_id": download_id,
            "artifact_ref": f"download://{download_id}",
            "filename": filename,
            "label": label.strip() or filename,
            "download_url": f"/v1/files/{download_id}?conversation_id={getattr(session, 'conversation_id', '')}",
            "content_type": mimetypes.guess_type(filename)[0] or "application/octet-stream",
            "size_bytes": path.stat().st_size,
            "session_id": str(getattr(session, "session_id", "") or ""),
            "conversation_id": str(getattr(session, "conversation_id", "") or ""),
        }
    )
    metadata = _metadata(session)
    downloads = {
        str(key): normalize_artifact(value)
        for key, value in dict(metadata.get("downloads") or {}).items()
        if isinstance(value, dict)
    }
    downloads[download_id] = artifact
    pending = [
        normalize_artifact(value)
        for value in list(metadata.get("pending_artifacts") or [])
        if isinstance(value, dict)
    ]
    pending.append(artifact)
    metadata["downloads"] = downloads
    metadata["pending_artifacts"] = pending
    return artifact


def pop_pending_artifacts(state_or_session: Any) -> List[Dict[str, Any]]:
    metadata = _metadata(state_or_session)
    artifacts = [
        normalize_artifact(value)
        for value in list(metadata.get("pending_artifacts") or [])
        if isinstance(value, dict)
    ]
    metadata["pending_artifacts"] = []
    return artifacts


def register_handoff_artifact(
    state_or_session: Any,
    *,
    artifact_type: str,
    handoff_schema: str,
    producer_task_id: str,
    producer_agent: str,
    data: Dict[str, Any],
    summary: str = "",
    allowed_consumers: List[str] | None = None,
    source_artifact_ids: List[str] | None = None,
) -> Dict[str, Any]:
    artifact_id = f"handoff_{uuid.uuid4().hex[:16]}"
    artifact = normalize_handoff_artifact(
        {
            "artifact_id": artifact_id,
            "artifact_ref": f"handoff://{artifact_id}",
            "artifact_type": artifact_type,
            "handoff_schema": handoff_schema,
            "producer_task_id": producer_task_id,
            "producer_agent": producer_agent,
            "summary": summary.strip() or artifact_type,
            "created_at": utc_now_iso(),
            "allowed_consumers": list(allowed_consumers or []),
            "source_artifact_ids": list(source_artifact_ids or []),
            "data": dict(data or {}),
        }
    )
    metadata = _metadata(state_or_session)
    existing = {
        str(key): normalize_handoff_artifact(value)
        for key, value in dict(metadata.get("handoff_artifacts") or {}).items()
        if isinstance(value, dict)
    }
    existing[artifact_id] = artifact
    metadata["handoff_artifacts"] = existing
    return artifact


def list_handoff_artifacts(
    state_or_session: Any,
    *,
    artifact_ids: List[str] | None = None,
    artifact_types: List[str] | None = None,
) -> List[Dict[str, Any]]:
    metadata = _metadata(state_or_session)
    records = [
        normalize_handoff_artifact(value)
        for value in dict(metadata.get("handoff_artifacts") or {}).values()
        if isinstance(value, dict)
    ]
    allowed_ids = {str(item) for item in (artifact_ids or []) if str(item)}
    allowed_types = {str(item) for item in (artifact_types or []) if str(item)}
    filtered: List[Dict[str, Any]] = []
    for record in records:
        if allowed_ids and str(record.get("artifact_id") or "") not in allowed_ids:
            continue
        if allowed_types and str(record.get("artifact_type") or "") not in allowed_types:
            continue
        filtered.append(record)
    return filtered


def get_handoff_artifact(state_or_session: Any, artifact_id: str) -> Dict[str, Any] | None:
    matches = list_handoff_artifacts(state_or_session, artifact_ids=[artifact_id])
    return matches[0] if matches else None


def latest_assistant_artifacts(messages: List[Any]) -> List[Dict[str, Any]]:
    for message in reversed(list(messages or [])):
        if getattr(message, "type", "") != "ai":
            continue
        kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
        artifacts = kwargs.get("artifacts")
        if not isinstance(artifacts, list):
            return []
        return [normalize_artifact(item) for item in artifacts if isinstance(item, dict)]
    return []
