from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def filesystem_key(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "id-empty"
    safe = _UNSAFE_PATH_CHARS.sub("-", raw).strip("._-")
    safe = safe[:80] if safe else "id"
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{safe}-{digest}"


@dataclass(frozen=True)
class RuntimePaths:
    runtime_root: Path
    workspace_root: Path
    memory_root: Path

    @classmethod
    def from_settings(cls, settings: Any) -> "RuntimePaths":
        def _resolve_path(attr: str, default: Path) -> Path:
            value = getattr(settings, attr, None)
            return Path(value) if value is not None else default

        runtime_root = _resolve_path("runtime_dir", Path("data") / "runtime")
        workspace_root = _resolve_path("workspace_dir", Path("data") / "workspaces")
        memory_root = _resolve_path("memory_dir", Path("data") / "memory")
        return cls(
            runtime_root=runtime_root,
            workspace_root=workspace_root,
            memory_root=memory_root,
        )

    def session_key(self, session_id: str) -> str:
        return filesystem_key(session_id)

    def job_key(self, job_id: str) -> str:
        return filesystem_key(job_id)

    def tenant_key(self, tenant_id: str) -> str:
        return filesystem_key(tenant_id)

    def user_key(self, user_id: str) -> str:
        return filesystem_key(user_id)

    def conversation_key(self, conversation_id: str) -> str:
        return filesystem_key(conversation_id)

    def session_dir(self, session_id: str) -> Path:
        return self.runtime_root / "sessions" / self.session_key(session_id)

    def job_dir(self, job_id: str) -> Path:
        return self.runtime_root / "jobs" / self.job_key(job_id)

    def workspace_dir(self, session_id: str) -> Path:
        return self.workspace_root / self.session_key(session_id)

    def user_profile_dir(self, tenant_id: str, user_id: str) -> Path:
        return (
            self.memory_root
            / "tenants"
            / self.tenant_key(tenant_id)
            / "users"
            / self.user_key(user_id)
            / "profile"
        )

    def conversation_memory_dir(self, tenant_id: str, user_id: str, conversation_id: str) -> Path:
        return (
            self.memory_root
            / "tenants"
            / self.tenant_key(tenant_id)
            / "users"
            / self.user_key(user_id)
            / "conversations"
            / self.conversation_key(conversation_id)
        )

    def session_state_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "state.json"

    def session_transcript_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "transcript.jsonl"

    def session_events_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "events.jsonl"

    def session_notifications_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "notifications.jsonl"

    def job_state_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "state.json"

    def job_transcript_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "transcript.jsonl"

    def job_events_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "events.jsonl"

    def job_mailbox_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "mailbox.jsonl"

    def job_artifacts_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "artifacts"
