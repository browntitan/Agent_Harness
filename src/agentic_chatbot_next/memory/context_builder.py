from __future__ import annotations

from typing import Any, Iterable, List, Sequence

from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.memory.manager import MemoryCandidateRetriever, MemorySelector
from agentic_chatbot_next.memory.file_store import FileMemoryStore
from agentic_chatbot_next.memory.scope import MemoryScope
from agentic_chatbot_next.memory.store import ManagedMemoryRecord


class MemoryContextBuilder:
    def __init__(
        self,
        store: Any,
        *,
        fallback_store: FileMemoryStore | None = None,
        candidate_retriever: MemoryCandidateRetriever | None = None,
        selector: MemorySelector | None = None,
        settings: Any | None = None,
        max_chars: int = 2000,
    ) -> None:
        self.store = store
        self.fallback_store = fallback_store
        self.candidate_retriever = candidate_retriever
        self.selector = selector
        self.settings = settings
        self.max_chars = max_chars

    def build_for_agent(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str = "",
        providers: Any | None = None,
    ) -> str:
        scopes = self._normalise_scopes(agent.memory_scopes)
        managed = self._build_managed_context(agent, session_state, scopes=scopes, user_text=user_text, providers=providers)
        if managed:
            return managed
        file_store = self.fallback_store if self.fallback_store is not None else (self.store if isinstance(self.store, FileMemoryStore) else None)
        if file_store is not None:
            return self._build_legacy_context(file_store, session_state, scopes=scopes)
        if hasattr(self.store, "list_records"):
            return self._build_record_summary(session_state, scopes=scopes)
        return ""

    def _build_legacy_context(self, store: FileMemoryStore, session_state: SessionState, *, scopes: Sequence[str]) -> str:
        parts: List[str] = []
        consumed = 0
        for scope in scopes:
            entries = store.list_entries(
                tenant_id=session_state.tenant_id,
                user_id=session_state.user_id,
                conversation_id=session_state.conversation_id,
                scope=scope,
            )
            if not entries:
                continue
            lines = [f"[{scope} memory]"]
            for entry in entries:
                line = f"- {entry.key}: {entry.value}"
                if consumed + len(line) > self.max_chars and parts:
                    break
                lines.append(line)
                consumed += len(line) + 1
            block = "\n".join(lines)
            if len(lines) > 1 and block.strip():
                parts.append(block)
            if consumed >= self.max_chars:
                break
        return "\n\n".join(parts).strip()

    def _build_managed_context(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        scopes: Sequence[str],
        user_text: str,
        providers: Any | None,
    ) -> str:
        if self.selector is None or self.candidate_retriever is None or self.settings is None:
            return ""
        mode = str(getattr(self.settings, "memory_manager_mode", "shadow") or "shadow").strip().lower()
        if bool(getattr(self.settings, "memory_shadow_mode", False)) or mode not in {"selector", "live"}:
            return ""
        if hasattr(self.store, "import_legacy_for_session"):
            file_entries_by_scope = self._legacy_entries(session_state)
            try:
                self.store.import_legacy_for_session(
                    tenant_id=session_state.tenant_id,
                    user_id=session_state.user_id,
                    conversation_id=session_state.conversation_id,
                    session_id=session_state.session_id,
                    file_entries_by_scope=file_entries_by_scope,
                )
            except Exception:
                pass
        candidates = self.candidate_retriever.retrieve(
            session_state=session_state,
            query=user_text or self._latest_user_text(session_state),
            scopes=scopes,
        )
        selection = self.selector.select(
            session_state=session_state,
            agent=agent,
            user_text=user_text or self._latest_user_text(session_state),
            providers=providers,
            candidates=candidates,
        )
        if not selection.brief.strip():
            return ""
        return f"[managed memory brief]\n{selection.brief.strip()}".strip()

    def _legacy_entries(self, session_state: SessionState) -> dict[str, list[tuple[str, str]]]:
        store = self.fallback_store if self.fallback_store is not None else (self.store if isinstance(self.store, FileMemoryStore) else None)
        if store is None:
            return {}
        payload: dict[str, list[tuple[str, str]]] = {}
        for scope in (MemoryScope.user.value, MemoryScope.conversation.value):
            entries = store.list_entries(
                tenant_id=session_state.tenant_id,
                user_id=session_state.user_id,
                conversation_id=session_state.conversation_id,
                scope=scope,
            )
            payload[scope] = [(entry.key, entry.value) for entry in entries]
        return payload

    def _build_record_summary(self, session_state: SessionState, *, scopes: Sequence[str]) -> str:
        parts: List[str] = []
        consumed = 0
        for scope in scopes:
            records = self.store.list_records(
                tenant_id=session_state.tenant_id,
                user_id=session_state.user_id,
                conversation_id=session_state.conversation_id,
                session_id=session_state.session_id,
                scope=scope,
                active_only=True,
                limit=20,
            )
            if not records:
                continue
            lines = [f"[{scope} memory]"]
            for record in records:
                heading = self._record_heading(record)
                line = f"- {heading}: {record.canonical_text}"
                if consumed + len(line) > self.max_chars and parts:
                    break
                lines.append(line)
                consumed += len(line) + 1
            block = "\n".join(lines)
            if len(lines) > 1 and block.strip():
                parts.append(block)
            if consumed >= self.max_chars:
                break
        return "\n\n".join(parts).strip()

    @staticmethod
    def _record_heading(record: ManagedMemoryRecord) -> str:
        return record.key or record.title or record.memory_id

    @staticmethod
    def _latest_user_text(session_state: SessionState) -> str:
        for message in reversed(session_state.messages):
            if message.role == "user" and str(message.content or "").strip():
                return str(message.content or "")
        return ""

    def _normalise_scopes(self, scopes: Iterable[str]) -> List[str]:
        clean = []
        for scope in scopes:
            try:
                clean.append(MemoryScope(scope).value)
            except ValueError:
                continue
        return clean or [MemoryScope.conversation.value]
