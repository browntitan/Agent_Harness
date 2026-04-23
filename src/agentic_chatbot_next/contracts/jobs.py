from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from agentic_chatbot_next.contracts.messages import RuntimeMessage, utc_now_iso
from agentic_chatbot_next.utils.json_utils import make_json_compatible


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [part.strip() for part in value.split(",")]
    else:
        try:
            raw_items = list(value)
        except TypeError:
            raw_items = [value]
    return [str(item).strip() for item in raw_items if str(item).strip()]


@dataclass
class TaskNotification:
    job_id: str
    status: str
    summary: str
    output_path: str = ""
    result_path: str = ""
    result: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_runtime_message(self) -> RuntimeMessage:
        body = (
            f"<task-notification id=\"{self.job_id}\" status=\"{self.status}\">\n"
            f"{self.summary}\n"
            f"</task-notification>"
        )
        return RuntimeMessage(
            role="system",
            content=body,
            metadata={"notification": self.to_dict()},
        )

    def to_dict(self) -> Dict[str, Any]:
        return make_json_compatible(asdict(self))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TaskNotification":
        return cls(
            job_id=str(raw.get("job_id") or ""),
            status=str(raw.get("status") or ""),
            summary=str(raw.get("summary") or ""),
            output_path=str(raw.get("output_path") or ""),
            result_path=str(raw.get("result_path") or ""),
            result=str(raw.get("result") or ""),
            created_at=str(raw.get("created_at") or utc_now_iso()),
            metadata=dict(raw.get("metadata") or {}),
        )


@dataclass
class WorkerMailboxMessage:
    job_id: str
    content: str
    sender: str = "parent"
    created_at: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:16]}")
    message_type: str = "message"
    direction: str = "to_worker"
    status: str = "queued"
    requires_response: bool = False
    response_to: str = ""
    correlation_id: str = ""
    subject: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    resolved_at: str = ""
    resolved_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return make_json_compatible(asdict(self))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "WorkerMailboxMessage":
        message_type = str(raw.get("message_type") or "message").strip() or "message"
        direction = str(raw.get("direction") or "to_worker").strip() or "to_worker"
        status = str(raw.get("status") or "queued").strip() or "queued"
        requires_response = bool(raw.get("requires_response", message_type in {"question_request", "approval_request"}))
        return cls(
            job_id=str(raw.get("job_id") or ""),
            content=str(raw.get("content") or ""),
            sender=str(raw.get("sender") or "parent"),
            created_at=str(raw.get("created_at") or utc_now_iso()),
            metadata=dict(raw.get("metadata") or {}),
            message_id=str(raw.get("message_id") or f"msg_{uuid.uuid4().hex[:16]}"),
            message_type=message_type,
            direction=direction,
            status=status,
            requires_response=requires_response,
            response_to=str(raw.get("response_to") or ""),
            correlation_id=str(raw.get("correlation_id") or raw.get("message_id") or ""),
            subject=str(raw.get("subject") or ""),
            payload=dict(raw.get("payload") or {}),
            resolved_at=str(raw.get("resolved_at") or ""),
            resolved_by=str(raw.get("resolved_by") or ""),
        )


@dataclass
class TeamMailboxChannel:
    session_id: str
    name: str
    purpose: str = ""
    created_by_job_id: str = ""
    member_agents: List[str] = field(default_factory=list)
    member_job_ids: List[str] = field(default_factory=list)
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)
    channel_id: str = field(default_factory=lambda: f"tmc_{uuid.uuid4().hex[:12]}")
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return make_json_compatible(asdict(self))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TeamMailboxChannel":
        return cls(
            session_id=str(raw.get("session_id") or ""),
            name=str(raw.get("name") or ""),
            purpose=str(raw.get("purpose") or ""),
            created_by_job_id=str(raw.get("created_by_job_id") or ""),
            member_agents=_string_list(raw.get("member_agents")),
            member_job_ids=_string_list(raw.get("member_job_ids")),
            status=str(raw.get("status") or "active"),
            metadata=dict(raw.get("metadata") or {}),
            channel_id=str(raw.get("channel_id") or f"tmc_{uuid.uuid4().hex[:12]}"),
            created_at=str(raw.get("created_at") or utc_now_iso()),
            updated_at=str(raw.get("updated_at") or raw.get("created_at") or utc_now_iso()),
        )


@dataclass
class TeamMailboxMessage:
    channel_id: str
    session_id: str
    content: str
    source_agent: str = ""
    source_job_id: str = ""
    target_agents: List[str] = field(default_factory=list)
    target_job_ids: List[str] = field(default_factory=list)
    message_type: str = "message"
    status: str = "open"
    subject: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    requires_response: bool = False
    response_to: str = ""
    thread_id: str = ""
    claimed_by: str = ""
    resolved_by: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: f"tmm_{uuid.uuid4().hex[:16]}")
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    resolved_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return make_json_compatible(asdict(self))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TeamMailboxMessage":
        message_type = str(raw.get("message_type") or "message").strip() or "message"
        return cls(
            channel_id=str(raw.get("channel_id") or ""),
            session_id=str(raw.get("session_id") or ""),
            content=str(raw.get("content") or ""),
            source_agent=str(raw.get("source_agent") or ""),
            source_job_id=str(raw.get("source_job_id") or ""),
            target_agents=_string_list(raw.get("target_agents")),
            target_job_ids=_string_list(raw.get("target_job_ids")),
            message_type=message_type,
            status=str(raw.get("status") or "open"),
            subject=str(raw.get("subject") or ""),
            payload=dict(raw.get("payload") or {}),
            requires_response=bool(raw.get("requires_response", message_type in {"question_request", "approval_request"})),
            response_to=str(raw.get("response_to") or ""),
            thread_id=str(raw.get("thread_id") or raw.get("response_to") or raw.get("message_id") or ""),
            claimed_by=str(raw.get("claimed_by") or ""),
            resolved_by=str(raw.get("resolved_by") or ""),
            metadata=dict(raw.get("metadata") or {}),
            message_id=str(raw.get("message_id") or f"tmm_{uuid.uuid4().hex[:16]}"),
            created_at=str(raw.get("created_at") or utc_now_iso()),
            updated_at=str(raw.get("updated_at") or raw.get("created_at") or utc_now_iso()),
            resolved_at=str(raw.get("resolved_at") or ""),
        )


@dataclass
class JobRecord:
    job_id: str
    session_id: str
    agent_name: str
    status: str
    prompt: str
    tenant_id: str = ""
    user_id: str = ""
    priority: str = "interactive"
    queue_class: str = "interactive"
    description: str = ""
    parent_job_id: str = ""
    artifact_dir: str = ""
    output_path: str = ""
    result_path: str = ""
    result_summary: str = ""
    last_error: str = ""
    enqueued_at: str = field(default_factory=utc_now_iso)
    started_at: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    scheduler_state: str = "queued"
    estimated_token_cost: int = 0
    actual_token_cost: int = 0
    budget_block_reason: str = ""
    session_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return make_json_compatible(asdict(self))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "JobRecord":
        return cls(
            job_id=str(raw.get("job_id") or ""),
            session_id=str(raw.get("session_id") or ""),
            agent_name=str(raw.get("agent_name") or ""),
            status=str(raw.get("status") or "queued"),
            prompt=str(raw.get("prompt") or ""),
            tenant_id=str(raw.get("tenant_id") or ""),
            user_id=str(raw.get("user_id") or ""),
            priority=str(raw.get("priority") or "interactive"),
            queue_class=str(raw.get("queue_class") or "interactive"),
            description=str(raw.get("description") or ""),
            parent_job_id=str(raw.get("parent_job_id") or ""),
            artifact_dir=str(raw.get("artifact_dir") or ""),
            output_path=str(raw.get("output_path") or ""),
            result_path=str(raw.get("result_path") or ""),
            result_summary=str(raw.get("result_summary") or ""),
            last_error=str(raw.get("last_error") or ""),
            enqueued_at=str(raw.get("enqueued_at") or raw.get("created_at") or utc_now_iso()),
            started_at=str(raw.get("started_at") or ""),
            created_at=str(raw.get("created_at") or utc_now_iso()),
            updated_at=str(raw.get("updated_at") or utc_now_iso()),
            scheduler_state=str(raw.get("scheduler_state") or raw.get("status") or "queued"),
            estimated_token_cost=_coerce_int(raw.get("estimated_token_cost"), 0),
            actual_token_cost=_coerce_int(raw.get("actual_token_cost"), 0),
            budget_block_reason=str(raw.get("budget_block_reason") or ""),
            session_state=dict(raw.get("session_state") or {}),
            metadata=dict(raw.get("metadata") or {}),
        )
