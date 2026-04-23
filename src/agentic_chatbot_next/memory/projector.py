from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from agentic_chatbot_next.memory.scope import MemoryScope
from agentic_chatbot_next.memory.store import ManagedMemoryRecord, MemoryStore
from agentic_chatbot_next.runtime.context import RuntimePaths, filesystem_key


_TYPE_TITLES = {
    "profile_preference": "Profile Preferences",
    "task_state": "Task State",
    "decision": "Decisions",
    "constraint": "Constraints",
    "open_loop": "Open Loops",
}


class MemoryProjector:
    def __init__(self, store: MemoryStore, paths: RuntimePaths) -> None:
        self.store = store
        self.paths = paths

    def project_session(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        session_id: str,
    ) -> None:
        for scope in (MemoryScope.user.value, MemoryScope.conversation.value):
            records = self.store.list_records(
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                session_id=session_id,
                scope=scope,
                active_only=True,
                limit=200,
            )
            self._write_scope_projection(
                scope_dir=self._scope_dir(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    scope=scope,
                ),
                scope=scope,
                records=records,
            )

    def _scope_dir(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        scope: str,
    ) -> Path:
        if scope == MemoryScope.user.value:
            return self.paths.user_profile_dir(tenant_id, user_id)
        return self.paths.conversation_memory_dir(tenant_id, user_id, conversation_id)

    def _write_scope_projection(
        self,
        *,
        scope_dir: Path,
        scope: str,
        records: List[ManagedMemoryRecord],
    ) -> None:
        scope_dir.mkdir(parents=True, exist_ok=True)
        topics_dir = scope_dir / "topics"
        groups_dir = scope_dir / "groups"
        topics_dir.mkdir(parents=True, exist_ok=True)
        groups_dir.mkdir(parents=True, exist_ok=True)

        index_payload = {
            "scope": scope,
            "entries": {
                record.key or record.memory_id: {
                    "memory_id": record.memory_id,
                    "key": record.key,
                    "title": record.title,
                    "value": record.canonical_text,
                    "memory_type": record.memory_type,
                    "importance": record.importance,
                    "confidence": record.confidence,
                    "updated_at": record.updated_at,
                }
                for record in records
            },
        }
        (scope_dir / "index.json").write_text(
            json.dumps(index_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        grouped: Dict[str, List[ManagedMemoryRecord]] = {}
        for record in records:
            grouped.setdefault(record.memory_type, []).append(record)

        memory_lines = [
            "# Managed Memory",
            "",
            f"Scope: {scope}",
            "",
        ]
        for memory_type, title in _TYPE_TITLES.items():
            entries = grouped.get(memory_type, [])
            if not entries:
                continue
            memory_lines.extend([f"## {title}", ""])
            type_lines = [f"# {title}", "", f"Scope: {scope}", ""]
            for record in entries:
                heading = record.title or record.key or record.memory_id
                entry_lines = [
                    f"### {heading}",
                    "",
                    record.canonical_text,
                    "",
                    f"- key: {record.key or '(none)'}",
                    f"- importance: {record.importance:.2f}",
                    f"- confidence: {record.confidence:.2f}",
                    f"- updated_at: {record.updated_at}",
                    "",
                ]
                memory_lines.extend(entry_lines)
                type_lines.extend(entry_lines)
                topic_name = filesystem_key(record.key or record.title or record.memory_id) + ".md"
                (topics_dir / topic_name).write_text(
                    "\n".join(
                        [
                            f"# {heading}",
                            "",
                            f"Scope: {scope}",
                            f"Type: {record.memory_type}",
                            "",
                            record.canonical_text,
                            "",
                            f"Key: {record.key or '(none)'}",
                            f"Importance: {record.importance:.2f}",
                            f"Confidence: {record.confidence:.2f}",
                            f"Updated: {record.updated_at}",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
            (groups_dir / f"{memory_type}.md").write_text("\n".join(type_lines).rstrip() + "\n", encoding="utf-8")

        (scope_dir / "MEMORY.md").write_text("\n".join(memory_lines).rstrip() + "\n", encoding="utf-8")
