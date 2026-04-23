from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agentic_chatbot_next.sandbox.workspace import SessionWorkspace
from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.messages import RuntimeMessage, SessionState
from agentic_chatbot_next.memory.file_store import FileMemoryStore
from agentic_chatbot_next.runtime.context import RuntimePaths


class SessionStateAdapter:
    """Session-like view over SessionState for reused live tools and agents."""

    def __init__(self, state: SessionState, *, paths: RuntimePaths, settings: Any) -> None:
        self._state = state
        self._paths = paths
        self._settings = settings
        self.tenant_id = state.tenant_id
        self.user_id = state.user_id
        self.conversation_id = state.conversation_id
        self.user_email = state.user_email
        self.auth_provider = state.auth_provider
        self.principal_id = state.principal_id
        self.access_summary = dict(state.access_summary or {})
        self.request_id = state.request_id
        self.session_id = state.session_id
        self.uploaded_doc_ids = state.uploaded_doc_ids
        self.scratchpad = state.scratchpad
        self.demo_mode = bool(state.demo_mode)
        self.active_agent = state.active_agent
        self.metadata = state.metadata
        self.workspace = self._build_workspace()

    @property
    def messages(self) -> List[Any]:
        return [message.to_langchain() for message in self._state.messages]

    @messages.setter
    def messages(self, value: List[Any]) -> None:
        self._state.messages = [
            item if isinstance(item, RuntimeMessage) else RuntimeMessage.from_langchain(item)
            for item in list(value or [])
        ]

    def _build_workspace(self) -> Optional[SessionWorkspace]:
        if getattr(self._settings, "workspace_dir", None) is None:
            return None
        workspace = SessionWorkspace.for_session(self.session_id, Path(self._settings.workspace_dir))
        workspace.open()
        self._state.workspace_root = str(workspace.root)
        return workspace


ToolBuilder = Callable[..., List[Any]]


@dataclass
class ToolContext:
    settings: Any
    providers: Any
    stores: Any
    session: SessionState
    paths: RuntimePaths
    callbacks: List[Any] = field(default_factory=list)
    transcript_store: Any = None
    job_manager: Any = None
    event_sink: Any = None
    kernel: Any = None
    active_agent: str = ""
    active_definition: Optional[AgentDefinition] = None
    skill_context: str = ""
    skill_resolution: Any = None
    file_memory_store: Optional[FileMemoryStore] = None
    memory_store: Any = None
    progress_emitter: Any = None
    rag_runtime_bridge: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _session_adapter: Optional[SessionStateAdapter] = field(default=None, init=False, repr=False)

    @property
    def session_handle(self) -> SessionStateAdapter:
        if self._session_adapter is None:
            self._session_adapter = SessionStateAdapter(self.session, paths=self.paths, settings=self.settings)
        return self._session_adapter

    @property
    def workspace_root(self) -> Optional[Path]:
        if not self.session.workspace_root:
            return None
        return Path(self.session.workspace_root)

    def refresh_from_session_handle(self) -> None:
        if self._session_adapter is not None:
            self.session.demo_mode = self._session_adapter.demo_mode
            self.session.active_agent = self._session_adapter.active_agent
            self.session.metadata = dict(self._session_adapter.metadata)
            self.session.user_email = str(getattr(self._session_adapter, "user_email", "") or "")
            self.session.auth_provider = str(getattr(self._session_adapter, "auth_provider", "") or "")
            self.session.principal_id = str(getattr(self._session_adapter, "principal_id", "") or "")
            self.session.access_summary = dict(getattr(self._session_adapter, "access_summary", {}) or {})
