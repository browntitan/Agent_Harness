from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Any, Dict, Iterable, List, Optional

from agentic_chatbot_next.contracts.jobs import (
    JobRecord,
    TaskNotification,
    TeamMailboxChannel,
    TeamMailboxMessage,
    WorkerMailboxMessage,
)
from agentic_chatbot_next.contracts.messages import RuntimeMessage, SessionState
from agentic_chatbot_next.observability.events import RuntimeEvent
from agentic_chatbot_next.runtime.context import RuntimePaths
from agentic_chatbot_next.utils.json_utils import make_json_compatible

_PATH_LOCKS: Dict[Path, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _path_lock(path: Path) -> threading.Lock:
    resolved = path.resolve()
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(resolved)
        if lock is None:
            lock = threading.Lock()
            _PATH_LOCKS[resolved] = lock
        return lock


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with _path_lock(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(make_json_compatible(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with _path_lock(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(make_json_compatible(payload), ensure_ascii=False) + "\n")


def _read_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return dict(default or {})
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return dict(default or {})
    return json.loads(raw)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _iter_jsonl_lines_reversed(path: Path, *, chunk_size: int = 8192) -> Iterable[str]:
    if not path.exists():
        return
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        pointer = handle.tell()
        buffer = b""
        while pointer > 0:
            read_size = min(chunk_size, pointer)
            pointer -= read_size
            handle.seek(pointer)
            buffer = handle.read(read_size) + buffer
            parts = buffer.split(b"\n")
            buffer = parts[0]
            for line in reversed(parts[1:]):
                if line.strip():
                    yield line.decode("utf-8")
        if buffer.strip():
            yield buffer.decode("utf-8")


def _transcript_row_to_runtime_message(row: Dict[str, Any]) -> RuntimeMessage | None:
    kind = str(row.get("kind") or "").strip().lower()
    if kind == "message":
        payload = row.get("message")
        if isinstance(payload, dict):
            return RuntimeMessage.from_dict(payload)
        if str(row.get("content") or "").strip():
            return RuntimeMessage(
                role=str(row.get("role") or "assistant"),
                content=str(row.get("content") or ""),
            )
        return None
    if kind == "notification" and isinstance(row.get("notification"), dict):
        return TaskNotification.from_dict(dict(row.get("notification") or {})).to_runtime_message()
    return None


class RuntimeTranscriptStore:
    def __init__(
        self,
        paths: RuntimePaths,
        *,
        session_hydrate_window_messages: int = 40,
        session_transcript_page_size: int = 100,
    ):
        self.paths = paths
        self.session_hydrate_window_messages = _coerce_positive_int(session_hydrate_window_messages, 40)
        self.session_transcript_page_size = _coerce_positive_int(session_transcript_page_size, 100)
        (self.paths.runtime_root / "sessions").mkdir(parents=True, exist_ok=True)
        (self.paths.runtime_root / "jobs").mkdir(parents=True, exist_ok=True)

    def session_dir(self, session_id: str) -> Path:
        return self.paths.session_dir(session_id)

    def job_dir(self, job_id: str) -> Path:
        return self.paths.job_dir(job_id)

    def persist_session_state(self, session: SessionState) -> None:
        window = max(1, self.session_hydrate_window_messages)
        current_messages = list(session.messages or [])
        current_count = len(current_messages)
        metadata = dict(session.metadata or {})
        previous_total = int(metadata.get("history_total_messages") or 0)
        previous_window = int(metadata.get("history_stored_window_messages") or 0)
        if previous_total > previous_window:
            total_messages = max(current_count, previous_total + max(0, current_count - previous_window))
        else:
            total_messages = current_count
        tail = current_messages[-window:]
        snapshot_metadata = {
            **metadata,
            "history_total_messages": total_messages,
            "history_stored_window_messages": len(tail),
            "has_earlier_history": total_messages > len(tail),
        }
        payload = {
            "tenant_id": session.tenant_id,
            "user_id": session.user_id,
            "conversation_id": session.conversation_id,
            "request_id": session.request_id,
            "session_id": session.session_id,
            "messages": [message.to_dict() for message in tail],
            "uploaded_doc_ids": list(session.uploaded_doc_ids),
            "scratchpad": dict(session.scratchpad),
            "demo_mode": bool(session.demo_mode),
            "workspace_root": str(session.workspace_root or ""),
            "pending_notifications": [
                item.to_dict() if isinstance(item, TaskNotification) else dict(item)
                for item in list(session.pending_notifications or [])
            ],
            "active_agent": str(session.active_agent or ""),
            "metadata": snapshot_metadata,
        }
        _write_json(self.session_dir(session.session_id) / "state.json", payload)

    def load_session_state(self, session_id: str) -> Optional[SessionState]:
        path = self.session_dir(session_id) / "state.json"
        if not path.exists():
            return None
        raw = _read_json(path)
        state = SessionState.from_dict(raw)
        metadata = dict(state.metadata or {})
        window = max(1, self.session_hydrate_window_messages)
        raw_message_count = len(list(raw.get("messages") or []))
        if raw_message_count > window:
            state.messages = state.messages[-window:]
        stored_window = len(state.messages)
        total_messages = int(metadata.get("history_total_messages") or 0)
        if total_messages <= 0:
            total_messages = max(raw_message_count, stored_window)
        metadata["history_total_messages"] = total_messages
        metadata["history_stored_window_messages"] = stored_window
        metadata["has_earlier_history"] = total_messages > stored_window
        state.metadata = metadata
        return state

    def append_session_transcript(self, session_id: str, row: Dict[str, Any]) -> None:
        _append_jsonl(self.session_dir(session_id) / "transcript.jsonl", row)

    def load_session_transcript(self, session_id: str) -> List[Dict[str, Any]]:
        return _read_jsonl(self.session_dir(session_id) / "transcript.jsonl")

    def append_session_compaction(self, session_id: str, boundary: Dict[str, Any]) -> None:
        _append_jsonl(
            self.session_dir(session_id) / "compactions.jsonl",
            {
                "kind": "context_compaction",
                "boundary": make_json_compatible(dict(boundary or {})),
            },
        )

    def load_session_compactions(self, session_id: str) -> List[Dict[str, Any]]:
        return _read_jsonl(self.session_dir(session_id) / "compactions.jsonl")

    def append_session_tool_result(self, session_id: str, payload: Dict[str, Any]) -> None:
        _append_jsonl(
            self.session_dir(session_id) / "tool_results.jsonl",
            {
                "kind": "tool_result_full",
                **make_json_compatible(dict(payload or {})),
            },
        )

    def load_session_tool_results(self, session_id: str) -> List[Dict[str, Any]]:
        return _read_jsonl(self.session_dir(session_id) / "tool_results.jsonl")

    def _team_mailbox_dir(self, session_id: str) -> Path:
        path = self.session_dir(session_id) / "team_mailbox"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def append_team_mailbox_audit(self, session_id: str, row: Dict[str, Any]) -> None:
        _append_jsonl(
            self._team_mailbox_dir(session_id) / "audit.jsonl",
            make_json_compatible(dict(row or {})),
        )

    def load_team_mailbox_audit(self, session_id: str) -> List[Dict[str, Any]]:
        return _read_jsonl(self._team_mailbox_dir(session_id) / "audit.jsonl")

    def load_team_channels(self, session_id: str) -> List[TeamMailboxChannel]:
        return [
            TeamMailboxChannel.from_dict(row)
            for row in _read_jsonl(self._team_mailbox_dir(session_id) / "channels.jsonl")
        ]

    def overwrite_team_channels(self, session_id: str, channels: Iterable[TeamMailboxChannel]) -> None:
        path = self._team_mailbox_dir(session_id) / "channels.jsonl"
        with _path_lock(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for channel in channels:
                    handle.write(json.dumps(channel.to_dict(), ensure_ascii=False) + "\n")

    def append_team_channel(self, channel: TeamMailboxChannel) -> None:
        _append_jsonl(self._team_mailbox_dir(channel.session_id) / "channels.jsonl", channel.to_dict())

    def load_team_channel(self, session_id: str, channel_id: str) -> TeamMailboxChannel | None:
        clean_channel_id = str(channel_id or "").strip()
        for channel in self.load_team_channels(session_id):
            if channel.channel_id == clean_channel_id:
                return channel
        return None

    def append_team_message(self, message: TeamMailboxMessage) -> None:
        _append_jsonl(
            self._team_mailbox_dir(message.session_id) / f"{message.channel_id}.jsonl",
            message.to_dict(),
        )

    def load_team_messages(self, session_id: str, channel_id: str) -> List[TeamMailboxMessage]:
        return [
            TeamMailboxMessage.from_dict(row)
            for row in _read_jsonl(self._team_mailbox_dir(session_id) / f"{channel_id}.jsonl")
        ]

    def overwrite_team_messages(
        self,
        session_id: str,
        channel_id: str,
        messages: Iterable[TeamMailboxMessage],
    ) -> None:
        path = self._team_mailbox_dir(session_id) / f"{channel_id}.jsonl"
        with _path_lock(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for message in messages:
                    handle.write(json.dumps(message.to_dict(), ensure_ascii=False) + "\n")

    def ensure_session_transcript_seeded(self, session_id: str, messages: Iterable[RuntimeMessage]) -> None:
        path = self.session_dir(session_id) / "transcript.jsonl"
        with _path_lock(path):
            if path.exists() and path.stat().st_size > 0:
                return
            rows = [{"kind": "message", "message": message.to_dict()} for message in list(messages or [])]
            if not rows:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(make_json_compatible(row), ensure_ascii=False) + "\n")

    def load_recent_session_messages(self, session_id: str, *, limit: int | None = None) -> List[RuntimeMessage]:
        target = _coerce_positive_int(limit, self.session_hydrate_window_messages)
        path = self.session_dir(session_id) / "transcript.jsonl"
        recent: List[RuntimeMessage] = []
        for line in _iter_jsonl_lines_reversed(path):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = _transcript_row_to_runtime_message(row)
            if message is None:
                continue
            recent.append(message)
            if len(recent) >= target:
                break
        recent.reverse()
        return recent

    def load_session_message_page(
        self,
        session_id: str,
        *,
        before_message_index: int | None = None,
        page_size: int | None = None,
    ) -> Dict[str, Any]:
        size = _coerce_positive_int(page_size, self.session_transcript_page_size)
        messages = [
            message
            for row in self.load_session_transcript(session_id)
            for message in [_transcript_row_to_runtime_message(row)]
            if message is not None
        ]
        total_messages = len(messages)
        end = total_messages if before_message_index is None else max(0, min(int(before_message_index), total_messages))
        start = max(0, end - size)
        next_before = start if start > 0 else None
        return {
            "messages": messages[start:end],
            "next_before_message_index": next_before,
            "total_messages": total_messages,
            "page_size": size,
        }

    def append_session_event(self, event: RuntimeEvent) -> None:
        _append_jsonl(self.session_dir(event.session_id) / "events.jsonl", event.to_dict())

    def load_session_events(self, session_id: str) -> List[Dict[str, Any]]:
        return _read_jsonl(self.session_dir(session_id) / "events.jsonl")

    def append_notification(self, session_id: str, notification: TaskNotification) -> None:
        _append_jsonl(self.session_dir(session_id) / "notifications.jsonl", notification.to_dict())

    def drain_notifications(self, session_id: str) -> List[TaskNotification]:
        path = self.session_dir(session_id) / "notifications.jsonl"
        with _path_lock(path):
            notifications = [TaskNotification.from_dict(row) for row in _read_jsonl(path)]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        return notifications

    def persist_job_state(self, job: JobRecord) -> None:
        job_dir = self.job_dir(job.job_id)
        artifacts_dir = job_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        if not job.artifact_dir:
            job.artifact_dir = str(artifacts_dir)
        _write_json(job_dir / "state.json", job.to_dict())

    def load_job_state(self, job_id: str) -> Optional[JobRecord]:
        path = self.job_dir(job_id) / "state.json"
        if not path.exists():
            return None
        return JobRecord.from_dict(_read_json(path))

    def list_job_states(self, *, session_id: str = "") -> List[JobRecord]:
        jobs: List[JobRecord] = []
        jobs_root = self.paths.runtime_root / "jobs"
        if not jobs_root.exists():
            return []
        for path in sorted(jobs_root.glob("*/state.json")):
            raw = _read_json(path)
            if session_id and str(raw.get("session_id")) != session_id:
                continue
            jobs.append(JobRecord.from_dict(raw))
        jobs.sort(key=lambda job: job.created_at)
        return jobs

    def append_job_transcript(self, job_id: str, row: Dict[str, Any]) -> None:
        _append_jsonl(self.job_dir(job_id) / "transcript.jsonl", row)

    def load_job_transcript(self, job_id: str) -> List[Dict[str, Any]]:
        return _read_jsonl(self.job_dir(job_id) / "transcript.jsonl")

    def append_job_event(self, event: RuntimeEvent) -> None:
        _append_jsonl(self.job_dir(event.job_id) / "events.jsonl", event.to_dict())

    def load_job_events(self, job_id: str) -> List[Dict[str, Any]]:
        return _read_jsonl(self.job_dir(job_id) / "events.jsonl")

    def append_mailbox_message(self, message: WorkerMailboxMessage) -> None:
        _append_jsonl(self.job_dir(message.job_id) / "mailbox.jsonl", message.to_dict())

    def load_mailbox_messages(self, job_id: str) -> List[WorkerMailboxMessage]:
        return [
            WorkerMailboxMessage.from_dict(row)
            for row in _read_jsonl(self.job_dir(job_id) / "mailbox.jsonl")
        ]

    def overwrite_mailbox(self, job_id: str, messages: Iterable[WorkerMailboxMessage]) -> None:
        path = self.job_dir(job_id) / "mailbox.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for message in messages:
                handle.write(json.dumps(message.to_dict(), ensure_ascii=False) + "\n")

    def artifact_path(self, job_id: str, filename: str) -> Path:
        safe_name = filename.replace("/", "_")
        artifacts_dir = self.job_dir(job_id) / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        return artifacts_dir / safe_name
