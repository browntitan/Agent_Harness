from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from agentic_chatbot_next.utils.json_utils import make_json_compatible

if TYPE_CHECKING:
    from agentic_chatbot_next.contracts.jobs import TaskNotification


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:16]}"


@dataclass
class RuntimeMessage:
    message_id: str = field(default_factory=new_message_id)
    role: str = "user"
    content: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    name: str = ""
    tool_call_id: str = ""
    artifact_refs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_langchain(self) -> Any:
        kwargs = {
            "message_id": self.message_id,
            "created_at": self.created_at,
            "artifact_refs": list(self.artifact_refs),
            **dict(self.metadata or {}),
        }
        if self.role == "system":
            return SystemMessage(content=self.content, additional_kwargs=kwargs)
        if self.role == "assistant":
            return AIMessage(content=self.content, additional_kwargs=kwargs)
        if self.role == "tool":
            return ToolMessage(
                content=self.content,
                tool_call_id=self.tool_call_id or self.name or self.message_id,
                additional_kwargs=kwargs,
            )
        return HumanMessage(content=self.content, additional_kwargs=kwargs)

    @classmethod
    def from_langchain(cls, message: Any) -> "RuntimeMessage":
        role_map = {
            "human": "user",
            "system": "system",
            "ai": "assistant",
            "tool": "tool",
        }
        role = role_map.get(getattr(message, "type", ""), "assistant")
        content = getattr(message, "content", "")
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
        metadata = dict(additional_kwargs)
        message_id = str(metadata.pop("message_id", "") or new_message_id())
        created_at = str(metadata.pop("created_at", "") or utc_now_iso())
        artifact_refs = metadata.pop("artifact_refs", []) or []
        return cls(
            message_id=message_id,
            role=role,
            content=str(content or ""),
            created_at=created_at,
            name=str(getattr(message, "name", "") or ""),
            tool_call_id=str(getattr(message, "tool_call_id", "") or ""),
            artifact_refs=[str(item) for item in artifact_refs if str(item)],
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return make_json_compatible(asdict(self))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RuntimeMessage":
        return cls(
            message_id=str(raw.get("message_id") or new_message_id()),
            role=str(raw.get("role") or "user"),
            content=str(raw.get("content") or ""),
            created_at=str(raw.get("created_at") or utc_now_iso()),
            name=str(raw.get("name") or ""),
            tool_call_id=str(raw.get("tool_call_id") or ""),
            artifact_refs=[str(item) for item in (raw.get("artifact_refs") or []) if str(item)],
            metadata=dict(raw.get("metadata") or {}),
        )


@dataclass
class SessionState:
    tenant_id: str
    user_id: str
    conversation_id: str
    user_email: str = ""
    auth_provider: str = ""
    principal_id: str = ""
    access_summary: Dict[str, Any] = field(default_factory=dict)
    request_id: str = ""
    session_id: str = ""
    messages: List[RuntimeMessage] = field(default_factory=list)
    uploaded_doc_ids: List[str] = field(default_factory=list)
    scratchpad: Dict[str, str] = field(default_factory=dict)
    demo_mode: bool = False
    workspace_root: str = ""
    pending_notifications: List["TaskNotification"] = field(default_factory=list)
    active_agent: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = f"{self.tenant_id}:{self.user_id}:{self.conversation_id}"

    def append_message(self, role: str, content: str, **kwargs: Any) -> RuntimeMessage:
        message = RuntimeMessage(role=role, content=content, **kwargs)
        self.messages.append(message)
        return message

    def add_notification(self, notification: "TaskNotification") -> None:
        self.pending_notifications.append(notification)
        self.messages.append(notification.to_runtime_message())

    def to_dict(self) -> Dict[str, Any]:
        from agentic_chatbot_next.contracts.jobs import TaskNotification

        payload = asdict(self)
        payload["messages"] = [message.to_dict() for message in self.messages]
        payload["pending_notifications"] = [
            item.to_dict() if isinstance(item, TaskNotification) else dict(item)
            for item in self.pending_notifications
        ]
        return make_json_compatible(payload)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SessionState":
        from agentic_chatbot_next.contracts.jobs import TaskNotification

        state = cls(
            tenant_id=str(raw.get("tenant_id") or "local-dev"),
            user_id=str(raw.get("user_id") or "local-user"),
            conversation_id=str(raw.get("conversation_id") or "local-conversation"),
            user_email=str(raw.get("user_email") or ""),
            auth_provider=str(raw.get("auth_provider") or ""),
            principal_id=str(raw.get("principal_id") or ""),
            access_summary=dict(raw.get("access_summary") or {}),
            request_id=str(raw.get("request_id") or ""),
            session_id=str(raw.get("session_id") or ""),
            uploaded_doc_ids=[str(item) for item in (raw.get("uploaded_doc_ids") or []) if str(item)],
            scratchpad={str(k): str(v) for k, v in dict(raw.get("scratchpad") or {}).items()},
            demo_mode=bool(raw.get("demo_mode", False)),
            workspace_root=str(raw.get("workspace_root") or ""),
            active_agent=str(raw.get("active_agent") or ""),
            metadata=dict(raw.get("metadata") or {}),
        )
        state.messages = [
            RuntimeMessage.from_dict(item)
            for item in (raw.get("messages") or [])
            if isinstance(item, dict)
        ]
        state.pending_notifications = [
            TaskNotification.from_dict(item)
            for item in (raw.get("pending_notifications") or [])
            if isinstance(item, dict)
        ]
        return state

    @classmethod
    def from_session(cls, session: Any) -> "SessionState":
        workspace = getattr(session, "workspace", None)
        state = cls(
            tenant_id=str(getattr(session, "tenant_id", "local-dev") or "local-dev"),
            user_id=str(getattr(session, "user_id", "local-user") or "local-user"),
            conversation_id=str(getattr(session, "conversation_id", "local-conversation") or "local-conversation"),
            user_email=str(getattr(session, "user_email", "") or ""),
            auth_provider=str(getattr(session, "auth_provider", "") or ""),
            principal_id=str(getattr(session, "principal_id", "") or ""),
            access_summary=dict(getattr(session, "access_summary", {}) or {}),
            request_id=str(getattr(session, "request_id", "") or ""),
            session_id=str(getattr(session, "session_id", "") or ""),
            uploaded_doc_ids=[str(item) for item in list(getattr(session, "uploaded_doc_ids", []) or []) if str(item)],
            scratchpad={str(k): str(v) for k, v in dict(getattr(session, "scratchpad", {}) or {}).items()},
            demo_mode=bool(getattr(session, "demo_mode", False)),
            workspace_root=str(getattr(workspace, "root", "") or ""),
            active_agent=str(getattr(session, "active_agent", "") or ""),
            metadata=dict(getattr(session, "metadata", {}) or {}),
        )
        state.messages = [
            message if isinstance(message, RuntimeMessage) else RuntimeMessage.from_langchain(message)
            for message in list(getattr(session, "messages", []) or [])
        ]
        return state

    def sync_to_session(self, session: Any) -> None:
        session.tenant_id = self.tenant_id
        session.user_id = self.user_id
        session.conversation_id = self.conversation_id
        session.user_email = self.user_email
        session.auth_provider = self.auth_provider
        session.principal_id = self.principal_id
        session.access_summary = dict(self.access_summary)
        session.request_id = self.request_id
        session.session_id = self.session_id
        session.messages = [message.to_langchain() for message in self.messages]
        session.uploaded_doc_ids = list(self.uploaded_doc_ids)
        session.scratchpad = dict(self.scratchpad)
        session.demo_mode = self.demo_mode
        session.active_agent = self.active_agent
        session.metadata = dict(self.metadata)
