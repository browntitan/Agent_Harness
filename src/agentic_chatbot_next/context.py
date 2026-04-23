from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agentic_chatbot_next.config import Settings


@dataclass(frozen=True)
class RequestContext:
    """Request identity and scope for the next runtime."""

    tenant_id: str
    user_id: str
    conversation_id: str
    user_email: str = ""
    auth_provider: str = ""
    principal_id: str = ""
    access_summary: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""

    @property
    def session_id(self) -> str:
        return f"{self.tenant_id}:{self.user_id}:{self.conversation_id}"


def build_local_context(
    settings: Settings,
    *,
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    user_email: Optional[str] = None,
    auth_provider: str = "",
    principal_id: str = "",
    access_summary: Optional[dict[str, Any]] = None,
    request_id: str = "",
) -> RequestContext:
    return RequestContext(
        tenant_id=tenant_id or settings.default_tenant_id,
        user_id=user_id or settings.default_user_id,
        conversation_id=conversation_id or settings.default_conversation_id,
        user_email=str(user_email or "").strip().casefold(),
        auth_provider=str(auth_provider or "").strip(),
        principal_id=str(principal_id or "").strip(),
        access_summary=dict(access_summary or {}),
        request_id=request_id,
    )
