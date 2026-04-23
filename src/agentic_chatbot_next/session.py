from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agentic_chatbot_next.context import RequestContext

if TYPE_CHECKING:
    from agentic_chatbot_next.sandbox.workspace import SessionWorkspace


@dataclass
class ChatSession:
    """Conversation state for the next runtime service."""

    tenant_id: str = "local-dev"
    user_id: str = "local-cli"
    conversation_id: str = "local-session"
    user_email: str = ""
    auth_provider: str = ""
    principal_id: str = ""
    access_summary: Dict[str, Any] = field(default_factory=dict)
    request_id: str = ""
    session_id: str = ""
    messages: List[Any] = field(default_factory=list)
    uploaded_doc_ids: List[str] = field(default_factory=list)
    scratchpad: Dict[str, str] = field(default_factory=dict)
    demo_mode: bool = False
    workspace: Optional["SessionWorkspace"] = field(default=None, repr=False)
    active_agent: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = f"{self.tenant_id}:{self.user_id}:{self.conversation_id}"

    @classmethod
    def from_context(cls, ctx: RequestContext, *, messages: Optional[List[Any]] = None) -> "ChatSession":
        return cls(
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            conversation_id=ctx.conversation_id,
            user_email=str(ctx.user_email or "").strip(),
            auth_provider=str(ctx.auth_provider or "").strip(),
            principal_id=str(ctx.principal_id or "").strip(),
            access_summary=dict(ctx.access_summary or {}),
            request_id=ctx.request_id,
            session_id=ctx.session_id,
            messages=list(messages or []),
        )

    def clear_scratchpad(self) -> None:
        self.scratchpad.clear()
