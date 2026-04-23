from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

from agentic_chatbot_next.contracts.messages import RuntimeMessage
from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.memory.file_store import FileMemoryStore
from agentic_chatbot_next.memory.scope import MemoryScope

_MEMORY_LINE = re.compile(r"^\s*(?:-|\*)?\s*([A-Za-z0-9_./ -]{2,80})\s*:\s*(.+?)\s*$")
_MEMORY_ASSIGNMENT = re.compile(r"([A-Za-z][A-Za-z0-9_./ -]{1,80})\s*=\s*([^;\n]+)")
_REMEMBER_THAT = re.compile(r"remember that\s+(.+?)\s+is\s+(.+?)(?:[.!?]|$)", re.IGNORECASE)
_EXPLICIT_MEMORY_INTENT = re.compile(
    r"\b(remember|save|store|note|keep\s+track\s+of|keep\s+in\s+mind)\b",
    re.IGNORECASE,
)


class MemoryExtractor:
    def __init__(self, store: FileMemoryStore) -> None:
        self.store = store

    @staticmethod
    def _normalize_key(key: str) -> str:
        clean = key.strip().lower()
        for prefix in ("remember ", "save ", "store ", "note "):
            if clean.startswith(prefix):
                clean = clean[len(prefix):].strip()
                break
        return clean.replace(" ", "_")

    def extract_entries(self, text: str) -> List[Tuple[str, str]]:
        entries: List[Tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for line in text.splitlines():
            for match in _MEMORY_ASSIGNMENT.finditer(line):
                key = self._normalize_key(match.group(1))
                value = match.group(2).strip().rstrip(".,;")
                item = (key, value)
                if item not in seen:
                    seen.add(item)
                    entries.append(item)
        if entries:
            return entries

        for line in text.splitlines():
            match = _MEMORY_LINE.match(line)
            if match:
                key = self._normalize_key(match.group(1))
                value = match.group(2).strip()
                item = (key, value)
                if item not in seen:
                    seen.add(item)
                    entries.append(item)
        if entries:
            return entries
        for match in _REMEMBER_THAT.finditer(text):
            key = self._normalize_key(match.group(1))
            value = match.group(2).strip()
            item = (key, value)
            if item not in seen:
                seen.add(item)
                entries.append(item)
        return entries

    def has_explicit_memory_intent(self, text: str) -> bool:
        return bool(_EXPLICIT_MEMORY_INTENT.search(text or ""))

    def save_entries(
        self,
        session_state: SessionState,
        *,
        scope: str,
        entries: Iterable[Tuple[str, str]],
    ) -> int:
        saved = 0
        for key, value in entries:
            self.store.save(
                tenant_id=session_state.tenant_id,
                user_id=session_state.user_id,
                conversation_id=session_state.conversation_id,
                scope=MemoryScope(scope).value,
                key=key,
                value=value,
            )
            saved += 1
        return saved

    def apply_from_text(self, session_state: SessionState, text: str, *, scopes: Iterable[str]) -> int:
        entries = self.extract_entries(text)
        if not entries:
            return 0
        total = 0
        for scope in scopes:
            total += self.save_entries(session_state, scope=scope, entries=entries)
        return total

    def apply_from_messages(
        self,
        session_state: SessionState,
        messages: Sequence[RuntimeMessage],
        *,
        scopes: Iterable[str],
    ) -> int:
        parts: List[str] = []
        for message in messages:
            if message.role not in {"user", "assistant"}:
                continue
            content = str(message.content or "").strip()
            if content:
                parts.append(content)
        if not parts:
            return 0
        return self.apply_from_text(session_state, "\n\n".join(parts), scopes=scopes)
