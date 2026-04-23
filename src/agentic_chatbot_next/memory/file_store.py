from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from agentic_chatbot_next.contracts.messages import utc_now_iso
from agentic_chatbot_next.memory.scope import MemoryScope
from agentic_chatbot_next.runtime.context import RuntimePaths, filesystem_key

_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


@dataclass
class MemoryEntry:
    key: str
    value: str
    updated_at: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "key": self.key,
            "value": self.value,
            "updated_at": self.updated_at,
        }


class FileMemoryStore:
    """Simple file-backed memory store keyed by user or conversation scope."""

    def __init__(self, paths: RuntimePaths):
        self.paths = paths

    def save(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        scope: str,
        key: str,
        value: str,
    ) -> None:
        path = self._index_path(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            scope=scope,
        )
        with self._path_lock(path):
            payload = self._read_scope_payload(
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                scope=scope,
            )
            entries = dict(payload.get("entries") or {})
            entries[key] = MemoryEntry(key=key, value=value, updated_at=utc_now_iso()).to_dict()
            payload["entries"] = entries
            self._write_scope_payload(
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                scope=scope,
                payload=payload,
            )

    def get(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        scope: str,
        key: str,
    ) -> Optional[str]:
        payload = self._read_scope_payload(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            scope=scope,
        )
        entry = dict((payload.get("entries") or {}).get(key) or {})
        value = entry.get("value")
        return str(value) if value is not None else None

    def list_keys(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        scope: str,
    ) -> List[str]:
        payload = self._read_scope_payload(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            scope=scope,
        )
        return sorted(str(key) for key in (payload.get("entries") or {}).keys())

    def list_entries(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        scope: str,
    ) -> List[MemoryEntry]:
        payload = self._read_scope_payload(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            scope=scope,
        )
        entries = dict(payload.get("entries") or {})
        return [
            MemoryEntry(
                key=str(key),
                value=str(dict(value).get("value") or ""),
                updated_at=str(dict(value).get("updated_at") or ""),
            )
            for key, value in sorted(entries.items(), key=lambda item: str(item[0]))
        ]

    def _scope_dir(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        scope: str,
    ) -> Path:
        scope_value = MemoryScope(scope)
        if scope_value == MemoryScope.user:
            return self.paths.user_profile_dir(tenant_id, user_id)
        return self.paths.conversation_memory_dir(tenant_id, user_id, conversation_id)

    def _index_path(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        scope: str,
    ) -> Path:
        return self._scope_dir(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            scope=scope,
        ) / "index.json"

    def _read_scope_payload(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        scope: str,
    ) -> Dict[str, object]:
        path = self._index_path(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            scope=scope,
        )
        if not path.exists():
            return {
                "scope": scope,
                "entries": {},
            }
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return {
                "scope": scope,
                "entries": {},
            }
        return dict(json.loads(raw))

    def _write_scope_payload(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        scope: str,
        payload: Dict[str, object],
    ) -> None:
        path = self._index_path(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            scope=scope,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
        self._write_derived_files(path.parent, payload)

    @staticmethod
    def _path_lock(path: Path) -> threading.Lock:
        key = str(path)
        with _PATH_LOCKS_GUARD:
            lock = _PATH_LOCKS.get(key)
            if lock is None:
                lock = threading.Lock()
                _PATH_LOCKS[key] = lock
        return lock

    def _write_derived_files(self, scope_dir: Path, payload: Dict[str, object]) -> None:
        entries = dict(payload.get("entries") or {})
        memory_lines = [
            "# Memory",
            "",
            f"Scope: {payload.get('scope', '')}",
            "",
        ]
        topics_dir = scope_dir / "topics"
        topics_dir.mkdir(parents=True, exist_ok=True)
        for key, raw_entry in sorted(entries.items(), key=lambda item: str(item[0]).lower()):
            entry = dict(raw_entry or {})
            value = str(entry.get("value") or "")
            updated_at = str(entry.get("updated_at") or "")
            memory_lines.extend(
                [
                    f"## {key}",
                    "",
                    value,
                    "",
                    f"Updated: {updated_at}",
                    "",
                ]
            )
            topic_path = topics_dir / f"{filesystem_key(str(key))}.md"
            topic_path.write_text(
                "\n".join(
                    [
                        f"# {key}",
                        "",
                        value,
                        "",
                        f"Updated: {updated_at}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        (scope_dir / "MEMORY.md").write_text("\n".join(memory_lines).rstrip() + "\n", encoding="utf-8")
