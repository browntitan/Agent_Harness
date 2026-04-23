from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from agentic_chatbot_next.contracts.jobs import (
    JobRecord,
    TaskNotification,
    TeamMailboxChannel,
    TeamMailboxMessage,
    WorkerMailboxMessage,
)
from agentic_chatbot_next.contracts.messages import utc_now_iso
from agentic_chatbot_next.observability.events import RuntimeEvent
from agentic_chatbot_next.observability.token_usage import extract_token_usage
from agentic_chatbot_next.runtime.event_sink import RuntimeEventSink
from agentic_chatbot_next.runtime.transcript_store import RuntimeTranscriptStore
from agentic_chatbot_next.runtime.worker_scheduler import (
    WorkerScheduler,
    normalize_priority,
    normalize_queue_class,
)

logger = logging.getLogger(__name__)

JobRunner = Callable[[JobRecord], str]
TERMINAL_JOB_STATUSES = {"completed", "failed", "stopped"}
ACTIVE_JOB_STATUSES = {"queued", "running", "waiting_message", "budget_blocked"}
MAILBOX_REQUEST_TYPES = {"question_request", "approval_request"}
MAILBOX_RESPONSE_TYPES = {"question_response", "approval_response"}
MAILBOX_TO_WORKER_TYPES = {"message", *MAILBOX_RESPONSE_TYPES}
MAILBOX_OPEN_STATUSES = {"open"}
MAILBOX_RESOLVED_STATUSES = {"answered", "approved", "denied", "cancelled"}
TEAM_MAILBOX_REQUEST_TYPES = {"question_request", "approval_request"}
TEAM_MAILBOX_RESPONSE_TYPES = {"question_response", "approval_response"}
TEAM_MAILBOX_MESSAGE_TYPES = {
    "message",
    "status_update",
    "handoff",
    *TEAM_MAILBOX_REQUEST_TYPES,
    *TEAM_MAILBOX_RESPONSE_TYPES,
}
TEAM_MAILBOX_RESOLVED_STATUSES = {"answered", "approved", "denied", "resolved", "cancelled"}


@dataclass
class AgentDispatchOutcome:
    job: JobRecord
    reused_existing_job: bool
    queued: bool
    notification: TaskNotification
    mailbox_message: WorkerMailboxMessage | None = None


def _coerce_nonnegative_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _dedupe_strings(items: List[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        clean = str(item or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


class RuntimeJobManager:
    def __init__(
        self,
        transcript_store: RuntimeTranscriptStore,
        *,
        settings: Any | None = None,
        event_sink: Optional[RuntimeEventSink] = None,
        max_worker_concurrency: int = 4,
    ) -> None:
        self.transcript_store = transcript_store
        self.settings = settings
        self.event_sink = event_sink
        self.max_worker_concurrency = max(1, int(max_worker_concurrency))
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self.scheduler = WorkerScheduler(
            enabled=bool(getattr(settings, "worker_scheduler_enabled", True)),
            max_concurrency=self.max_worker_concurrency,
            urgent_reserved_slots=max(0, int(getattr(settings, "worker_scheduler_urgent_reserved_slots", 1))),
            tenant_budget_tokens_per_minute=max(
                0,
                int(getattr(settings, "worker_scheduler_tenant_budget_tokens_per_minute", 24000)),
            ),
            tenant_budget_burst_tokens=max(
                0,
                int(getattr(settings, "worker_scheduler_tenant_budget_burst_tokens", 48000)),
            ),
            emit_event=self._emit,
            persist_job=self.transcript_store.persist_job_state,
        )

    def create_job(
        self,
        *,
        agent_name: str,
        prompt: str,
        session_id: str,
        description: str = "",
        session_state: Optional[Dict[str, object]] = None,
        metadata: Optional[Dict[str, object]] = None,
        parent_job_id: str = "",
        tenant_id: str = "",
        user_id: str = "",
        priority: str = "interactive",
        queue_class: str = "",
        estimated_token_cost: int | None = None,
    ) -> JobRecord:
        timestamp = utc_now_iso()
        normalized_queue_class = normalize_queue_class(queue_class or priority or "interactive")
        normalized_priority = normalize_priority(priority or normalized_queue_class, default=normalized_queue_class)
        job = JobRecord(
            job_id=f"job_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            agent_name=agent_name,
            status="queued",
            prompt=prompt,
            tenant_id=str(tenant_id or ""),
            user_id=str(user_id or ""),
            priority=normalized_priority,
            queue_class=normalized_queue_class,
            description=description,
            parent_job_id=parent_job_id,
            created_at=timestamp,
            updated_at=timestamp,
            enqueued_at=timestamp,
            scheduler_state="queued",
            session_state=dict(session_state or {}),
            metadata=dict(metadata or {}),
        )
        if int(estimated_token_cost or 0) > 0:
            job.estimated_token_cost = int(estimated_token_cost or 0)
        else:
            job.estimated_token_cost = self.scheduler.estimate_job_token_cost(job, settings=self.settings)
        if not job.tenant_id:
            session_payload = dict(job.metadata.get("session_state") or job.session_state or {})
            job.tenant_id = str(session_payload.get("tenant_id") or "")
        if not job.user_id:
            session_payload = dict(job.metadata.get("session_state") or job.session_state or {})
            job.user_id = str(session_payload.get("user_id") or "")
        if "delegation_depth" not in job.metadata:
            parent_depth = 0
            if job.parent_job_id:
                parent_job = self.get_job(job.parent_job_id)
                if parent_job is not None:
                    parent_depth = _coerce_nonnegative_int(
                        getattr(parent_job, "metadata", {}).get("delegation_depth"),
                        0,
                    ) + 1
            job.metadata["delegation_depth"] = parent_depth
        self.transcript_store.persist_job_state(job)
        self._emit(
            "job_created",
            job,
            {
                "description": description,
                "queue_class": job.queue_class,
                "priority": job.priority,
                "estimated_token_cost": job.estimated_token_cost,
                "tenant_id": job.tenant_id,
                "user_id": job.user_id,
            },
        )
        return job

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        return self.transcript_store.load_job_state(job_id)

    def list_jobs(self, *, session_id: str = "") -> List[JobRecord]:
        return self.transcript_store.list_job_states(session_id=session_id)

    def scheduler_snapshot(self) -> Dict[str, Any]:
        return self.scheduler.snapshot()

    def start_background_job(self, job: JobRecord, runner: JobRunner) -> JobRecord:
        with self._lock:
            existing = self._threads.get(job.job_id)
            if existing is not None and existing.is_alive():
                return job
            thread = threading.Thread(target=self._run_job, args=(job.job_id, runner), daemon=True)
            self._threads[job.job_id] = thread
        thread.start()
        return job

    def continue_job(self, job_id: str, runner: JobRunner) -> Optional[JobRecord]:
        job = self.get_job(job_id)
        if job is None:
            return None
        with self._lock:
            existing = self._threads.get(job.job_id)
            if existing is not None and existing.is_alive():
                return job
        if job.status == "running":
            return job
        return self.start_background_job(job, runner)

    def run_job_inline(self, job: JobRecord, runner: JobRunner) -> str:
        return self._run_job(job.job_id, runner)

    def enqueue_message(
        self,
        job_id: str,
        content: str,
        *,
        sender: str = "parent",
        metadata: Optional[Dict[str, object]] = None,
    ) -> Optional[WorkerMailboxMessage]:
        job = self.get_job(job_id)
        if job is None:
            return None
        message = WorkerMailboxMessage(
            job_id=job_id,
            content=content,
            sender=sender,
            metadata=dict(metadata or {}),
            message_type="message",
            direction="to_worker",
            status="queued",
            requires_response=False,
        )
        self.transcript_store.append_mailbox_message(message)
        self._append_mailbox_audit(job.job_id, "mailbox_delivered", message)
        self._emit("mailbox_enqueued", job, {"sender": sender, "message_id": message.message_id})
        self._queue_job_for_mailbox_delivery(job)
        return message

    def open_worker_request(
        self,
        job_id: str,
        *,
        request_type: str,
        content: str,
        sender: str,
        subject: str = "",
        payload: Optional[Dict[str, object]] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Optional[WorkerMailboxMessage]:
        job = self.get_job(job_id)
        if job is None:
            return None
        clean_type = str(request_type or "").strip()
        if clean_type not in MAILBOX_REQUEST_TYPES:
            raise ValueError(f"Unsupported mailbox request type '{clean_type}'.")
        open_limit = max(1, int(getattr(self.settings, "worker_mailbox_open_request_limit", 3) or 3))
        open_requests = [
            item
            for item in self.transcript_store.load_mailbox_messages(job_id)
            if item.message_type in MAILBOX_REQUEST_TYPES and item.status == "open"
        ]
        if len(open_requests) >= open_limit:
            raise ValueError(f"Worker job already has {len(open_requests)} open mailbox request(s).")

        message = WorkerMailboxMessage(
            job_id=job_id,
            content=str(content or "").strip(),
            sender=str(sender or job.agent_name or "worker").strip() or "worker",
            metadata=dict(metadata or {}),
            message_id=f"msg_{uuid.uuid4().hex[:16]}",
            message_type=clean_type,
            direction="from_worker",
            status="open",
            requires_response=True,
            correlation_id=f"corr_{uuid.uuid4().hex[:16]}",
            subject=str(subject or "").strip(),
            payload=dict(payload or {}),
        )
        self.transcript_store.append_mailbox_message(message)
        self._append_mailbox_audit(job.job_id, "mailbox_request_opened", message)
        now = utc_now_iso()
        job.status = "waiting_message"
        job.scheduler_state = "waiting_message"
        job.updated_at = now
        job.result_summary = message.content[:2000]
        job.metadata["pending_mailbox_request"] = message.to_dict()
        self.transcript_store.persist_job_state(job)
        self._emit("worker_mailbox_request_opened", job, {"message": message.to_dict()})
        self._emit("worker_agent_waiting", job, {"message": message.to_dict()})
        self.append_session_notification(
            job.session_id,
            TaskNotification(
                job_id=job.job_id,
                status="waiting_message",
                summary=message.content or message.subject or f"{job.agent_name} is waiting for input.",
                metadata={
                    "agent_name": job.agent_name,
                    "mailbox_request": message.to_dict(),
                },
            ),
        )
        return message

    def respond_to_request(
        self,
        job_id: str,
        request_id: str,
        *,
        response: str,
        responder: str = "parent",
        decision: str = "",
        allow_approval: bool = False,
        metadata: Optional[Dict[str, object]] = None,
    ) -> tuple[WorkerMailboxMessage, WorkerMailboxMessage] | None:
        job = self.get_job(job_id)
        if job is None:
            return None
        clean_request_id = str(request_id or "").strip()
        messages = self.transcript_store.load_mailbox_messages(job_id)
        request: WorkerMailboxMessage | None = None
        for item in messages:
            if item.message_id == clean_request_id:
                request = item
                break
        if request is None:
            raise ValueError(f"Mailbox request '{clean_request_id}' was not found.")
        if request.message_type not in MAILBOX_REQUEST_TYPES:
            raise ValueError(f"Mailbox message '{clean_request_id}' is not a request.")
        if request.status != "open":
            raise ValueError(f"Mailbox request '{clean_request_id}' is already {request.status}.")

        clean_response = str(response or "").strip()
        clean_decision = str(decision or "").strip().lower()
        if request.message_type == "approval_request":
            if not allow_approval:
                raise PermissionError("Approval requests require operator/API approval.")
            if clean_decision not in {"approved", "denied"}:
                raise ValueError("Approval responses require decision='approved' or decision='denied'.")
            response_type = "approval_response"
            resolved_status = clean_decision
            if not clean_response:
                clean_response = "Approved." if clean_decision == "approved" else "Denied."
        else:
            if clean_decision:
                raise ValueError("Question responses must not include an approval decision.")
            if not clean_response:
                raise ValueError("Question responses require response text.")
            response_type = "question_response"
            resolved_status = "answered"

        now = utc_now_iso()
        request.status = resolved_status
        request.resolved_at = now
        request.resolved_by = str(responder or "parent").strip() or "parent"
        request.payload = {
            **dict(request.payload or {}),
            "response": clean_response,
            "decision": clean_decision,
        }
        response_message = WorkerMailboxMessage(
            job_id=job_id,
            content=clean_response,
            sender=str(responder or "parent").strip() or "parent",
            metadata=dict(metadata or {}),
            message_id=f"msg_{uuid.uuid4().hex[:16]}",
            message_type=response_type,
            direction="to_worker",
            status="queued",
            requires_response=False,
            response_to=request.message_id,
            correlation_id=request.correlation_id or request.message_id,
            subject=request.subject,
            payload={
                "request": request.to_dict(),
                "decision": clean_decision,
            },
        )
        rewritten: List[WorkerMailboxMessage] = []
        replaced = False
        for item in messages:
            if item.message_id == request.message_id:
                rewritten.append(request)
                replaced = True
            else:
                rewritten.append(item)
        if not replaced:
            rewritten.append(request)
        rewritten.append(response_message)
        self.transcript_store.overwrite_mailbox(job_id, rewritten)
        self._append_mailbox_audit(job_id, "mailbox_request_resolved", request, extra={"response": response_message.to_dict()})
        self._append_mailbox_audit(job_id, "mailbox_delivered", response_message)
        self._emit(
            "worker_mailbox_request_resolved",
            job,
            {
                "request": request.to_dict(),
                "response": response_message.to_dict(),
                "decision": clean_decision,
            },
        )
        self._queue_job_for_mailbox_delivery(job)
        return request, response_message

    def list_mailbox_messages(
        self,
        job_id: str,
        *,
        status_filter: str = "",
        request_type: str = "",
    ) -> List[WorkerMailboxMessage]:
        messages = self.transcript_store.load_mailbox_messages(job_id)
        clean_status = str(status_filter or "").strip()
        clean_type = str(request_type or "").strip()
        rows: List[WorkerMailboxMessage] = []
        for item in messages:
            if clean_status and item.status != clean_status:
                continue
            if clean_type:
                if clean_type == "request" and item.message_type not in MAILBOX_REQUEST_TYPES:
                    continue
                if clean_type != "request" and item.message_type != clean_type:
                    continue
            rows.append(item)
        return rows

    def list_mailbox_requests(
        self,
        job_id: str,
        *,
        status_filter: str = "open",
        request_type: str = "",
    ) -> List[WorkerMailboxMessage]:
        clean_type = str(request_type or "").strip()
        messages = self.list_mailbox_messages(job_id, status_filter=status_filter, request_type="")
        return [
            item
            for item in messages
            if item.message_type in MAILBOX_REQUEST_TYPES
            and (not clean_type or item.message_type == clean_type)
        ]

    def mailbox_summary(self, job_id: str) -> Dict[str, object]:
        open_requests = self.list_mailbox_requests(job_id, status_filter="open")
        latest = open_requests[-1].to_dict() if open_requests else {}
        return {
            "pending_question_count": sum(1 for item in open_requests if item.message_type == "question_request"),
            "pending_approval_count": sum(1 for item in open_requests if item.message_type == "approval_request"),
            "latest_open_request": latest,
        }

    def create_team_channel(
        self,
        *,
        session_id: str,
        name: str,
        purpose: str = "",
        created_by_job_id: str = "",
        member_agents: Optional[List[str]] = None,
        member_job_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> TeamMailboxChannel:
        clean_session_id = str(session_id or "").strip()
        clean_name = str(name or "").strip()
        if not clean_session_id:
            raise ValueError("session_id is required for team mailbox channels.")
        if not clean_name:
            raise ValueError("name is required for team mailbox channels.")
        channels = self.transcript_store.load_team_channels(clean_session_id)
        active_count = sum(1 for item in channels if item.status == "active")
        limit = max(1, int(getattr(self.settings, "team_mailbox_max_channels_per_session", 8) or 8))
        if active_count >= limit:
            raise ValueError(f"Session already has {active_count} active team mailbox channel(s).")
        channel = TeamMailboxChannel(
            session_id=clean_session_id,
            name=clean_name,
            purpose=str(purpose or "").strip(),
            created_by_job_id=str(created_by_job_id or "").strip(),
            member_agents=_dedupe_strings(list(member_agents or [])),
            member_job_ids=_dedupe_strings(list(member_job_ids or [])),
            metadata=dict(metadata or {}),
        )
        self.transcript_store.append_team_channel(channel)
        self._append_team_mailbox_audit(clean_session_id, "team_channel_created", {"channel": channel.to_dict()})
        self._emit_team_event(
            "team_mailbox_channel_created",
            session_id=clean_session_id,
            job_id=channel.created_by_job_id,
            agent_name=str(channel.metadata.get("created_by_agent") or ""),
            payload={"channel": channel.to_dict()},
        )
        return channel

    def list_team_channels(self, session_id: str, *, status_filter: str = "") -> List[TeamMailboxChannel]:
        clean_status = str(status_filter or "").strip()
        channels = self.transcript_store.load_team_channels(str(session_id or "").strip())
        if clean_status:
            channels = [item for item in channels if item.status == clean_status]
        return channels

    def archive_team_channel(self, session_id: str, channel_id: str, *, archived_by: str = "") -> TeamMailboxChannel | None:
        clean_session_id = str(session_id or "").strip()
        clean_channel_id = str(channel_id or "").strip()
        channels = self.transcript_store.load_team_channels(clean_session_id)
        archived: TeamMailboxChannel | None = None
        for channel in channels:
            if channel.channel_id == clean_channel_id:
                channel.status = "archived"
                channel.updated_at = utc_now_iso()
                channel.metadata = {**dict(channel.metadata or {}), "archived_by": str(archived_by or "").strip()}
                archived = channel
                break
        if archived is None:
            return None
        self.transcript_store.overwrite_team_channels(clean_session_id, channels)
        self._append_team_mailbox_audit(clean_session_id, "team_channel_archived", {"channel": archived.to_dict()})
        return archived

    def post_team_message(
        self,
        *,
        session_id: str,
        channel_id: str,
        content: str,
        source_agent: str = "",
        source_job_id: str = "",
        target_agents: Optional[List[str]] = None,
        target_job_ids: Optional[List[str]] = None,
        message_type: str = "message",
        subject: str = "",
        payload: Optional[Dict[str, object]] = None,
        metadata: Optional[Dict[str, object]] = None,
        response_to: str = "",
        thread_id: str = "",
    ) -> TeamMailboxMessage:
        clean_session_id = str(session_id or "").strip()
        channel = self.transcript_store.load_team_channel(clean_session_id, channel_id)
        if channel is None:
            raise ValueError(f"Team mailbox channel '{channel_id}' was not found.")
        if channel.status != "active":
            raise ValueError(f"Team mailbox channel '{channel_id}' is {channel.status}.")
        clean_type = str(message_type or "message").strip() or "message"
        if clean_type not in TEAM_MAILBOX_MESSAGE_TYPES:
            raise ValueError(f"Unsupported team mailbox message type '{clean_type}'.")
        clean_content = str(content or "").strip()
        if not clean_content:
            raise ValueError("content is required for team mailbox messages.")
        open_limit = max(1, int(getattr(self.settings, "team_mailbox_max_open_messages_per_channel", 50) or 50))
        open_messages = [
            item
            for item in self.transcript_store.load_team_messages(clean_session_id, channel.channel_id)
            if item.status == "open"
        ]
        if len(open_messages) >= open_limit:
            raise ValueError(f"Team mailbox channel already has {len(open_messages)} open message(s).")
        message = TeamMailboxMessage(
            channel_id=channel.channel_id,
            session_id=clean_session_id,
            content=clean_content,
            source_agent=str(source_agent or "").strip(),
            source_job_id=str(source_job_id or "").strip(),
            target_agents=_dedupe_strings(list(target_agents or [])),
            target_job_ids=_dedupe_strings(list(target_job_ids or [])),
            message_type=clean_type,
            status="open",
            subject=str(subject or "").strip(),
            payload=dict(payload or {}),
            requires_response=clean_type in TEAM_MAILBOX_REQUEST_TYPES,
            response_to=str(response_to or "").strip(),
            thread_id=str(thread_id or response_to or "").strip(),
            metadata=dict(metadata or {}),
        )
        if not message.thread_id:
            message.thread_id = message.message_id
        self.transcript_store.append_team_message(message)
        self._append_team_mailbox_audit(clean_session_id, "team_message_posted", {"message": message.to_dict()})
        self._emit_team_event(
            "team_mailbox_message_posted",
            session_id=clean_session_id,
            job_id=message.source_job_id,
            agent_name=message.source_agent,
            payload={"message": message.to_dict()},
        )
        return message

    def list_team_messages(
        self,
        session_id: str,
        *,
        channel_id: str = "",
        message_type: str = "",
        status_filter: str = "open",
        limit: int = 20,
    ) -> List[TeamMailboxMessage]:
        clean_session_id = str(session_id or "").strip()
        clean_channel_id = str(channel_id or "").strip()
        clean_type = str(message_type or "").strip()
        clean_status = str(status_filter or "").strip()
        max_rows = max(1, min(int(limit or 20), 200))
        channels = (
            [self.transcript_store.load_team_channel(clean_session_id, clean_channel_id)]
            if clean_channel_id
            else self.list_team_channels(clean_session_id, status_filter="")
        )
        rows: List[TeamMailboxMessage] = []
        for channel in channels:
            if channel is None:
                continue
            for item in self.transcript_store.load_team_messages(clean_session_id, channel.channel_id):
                if clean_type and item.message_type != clean_type:
                    continue
                if clean_status and item.status != clean_status:
                    continue
                rows.append(item)
        rows.sort(key=lambda item: item.created_at)
        return rows[-max_rows:]

    def claim_team_messages(
        self,
        session_id: str,
        channel_id: str,
        *,
        claimant_agent: str = "",
        claimant_job_id: str = "",
        limit: int = 0,
        message_type: str = "",
    ) -> List[TeamMailboxMessage]:
        clean_session_id = str(session_id or "").strip()
        clean_channel_id = str(channel_id or "").strip()
        channel = self.transcript_store.load_team_channel(clean_session_id, clean_channel_id)
        if channel is None:
            raise ValueError(f"Team mailbox channel '{clean_channel_id}' was not found.")
        max_claim = max(1, int(getattr(self.settings, "team_mailbox_claim_limit", 8) or 8))
        claim_limit = max(1, min(int(limit or max_claim), max_claim))
        clean_type = str(message_type or "").strip()
        clean_agent = str(claimant_agent or "").strip()
        clean_job_id = str(claimant_job_id or "").strip()
        messages = self.transcript_store.load_team_messages(clean_session_id, clean_channel_id)
        claimed: List[TeamMailboxMessage] = []
        now = utc_now_iso()
        for item in messages:
            if len(claimed) >= claim_limit:
                break
            if item.status not in {"open", "claimed"}:
                continue
            if clean_type and item.message_type != clean_type:
                continue
            if item.claimed_by and item.claimed_by not in {clean_agent, clean_job_id}:
                continue
            if item.source_job_id and clean_job_id and item.source_job_id == clean_job_id:
                continue
            if item.source_agent and clean_agent and item.source_agent == clean_agent and not item.target_job_ids:
                continue
            if item.target_agents and clean_agent not in set(item.target_agents):
                continue
            if item.target_job_ids and clean_job_id not in set(item.target_job_ids):
                continue
            item.claimed_by = clean_job_id or clean_agent
            item.updated_at = now
            if item.message_type not in TEAM_MAILBOX_REQUEST_TYPES:
                item.status = "claimed"
            claimed.append(item)
        if claimed:
            self.transcript_store.overwrite_team_messages(clean_session_id, clean_channel_id, messages)
            for item in claimed:
                self._append_team_mailbox_audit(
                    clean_session_id,
                    "team_message_claimed",
                    {"message": item.to_dict(), "claimed_by": item.claimed_by},
                )
            self._emit_team_event(
                "team_mailbox_message_claimed",
                session_id=clean_session_id,
                job_id=clean_job_id,
                agent_name=clean_agent,
                payload={"channel_id": clean_channel_id, "messages": [item.to_dict() for item in claimed]},
            )
        return claimed

    def respond_team_message(
        self,
        session_id: str,
        channel_id: str,
        message_id: str,
        *,
        response: str,
        responder_agent: str = "",
        responder_job_id: str = "",
        decision: str = "",
        allow_approval: bool = False,
        resolve: bool = True,
        metadata: Optional[Dict[str, object]] = None,
    ) -> tuple[TeamMailboxMessage, TeamMailboxMessage] | None:
        clean_session_id = str(session_id or "").strip()
        clean_channel_id = str(channel_id or "").strip()
        messages = self.transcript_store.load_team_messages(clean_session_id, clean_channel_id)
        request = next((item for item in messages if item.message_id == str(message_id or "").strip()), None)
        if request is None:
            return None
        if request.message_type not in TEAM_MAILBOX_REQUEST_TYPES:
            raise ValueError("Team mailbox responses require an open question_request or approval_request.")
        if request.status not in {"open", "claimed"}:
            raise ValueError(f"Team mailbox request is already {request.status}.")
        clean_response = str(response or "").strip()
        clean_decision = str(decision or "").strip().lower()
        if request.message_type == "approval_request":
            if not allow_approval:
                raise PermissionError("Approval requests require operator/API approval.")
            if clean_decision not in {"approved", "denied"}:
                raise ValueError("Approval responses require decision='approved' or decision='denied'.")
            response_type = "approval_response"
            resolved_status = clean_decision
            if not clean_response:
                clean_response = "Approved." if clean_decision == "approved" else "Denied."
        else:
            if clean_decision:
                raise ValueError("Question responses must not include an approval decision.")
            if not clean_response:
                raise ValueError("Question responses require response text.")
            response_type = "question_response"
            resolved_status = "answered"
        now = utc_now_iso()
        if resolve:
            request.status = resolved_status
            request.resolved_at = now
            request.resolved_by = str(responder_job_id or responder_agent or "").strip()
        request.updated_at = now
        request.payload = {
            **dict(request.payload or {}),
            "response": clean_response,
            "decision": clean_decision,
        }
        response_message = TeamMailboxMessage(
            channel_id=clean_channel_id,
            session_id=clean_session_id,
            content=clean_response,
            source_agent=str(responder_agent or "").strip(),
            source_job_id=str(responder_job_id or "").strip(),
            target_agents=[request.source_agent] if request.source_agent else [],
            target_job_ids=[request.source_job_id] if request.source_job_id else [],
            message_type=response_type,
            status="open",
            subject=request.subject,
            payload={"request": request.to_dict(), "decision": clean_decision},
            requires_response=False,
            response_to=request.message_id,
            thread_id=request.thread_id or request.message_id,
            metadata=dict(metadata or {}),
        )
        rewritten: List[TeamMailboxMessage] = []
        for item in messages:
            rewritten.append(request if item.message_id == request.message_id else item)
        rewritten.append(response_message)
        self.transcript_store.overwrite_team_messages(clean_session_id, clean_channel_id, rewritten)
        self._append_team_mailbox_audit(
            clean_session_id,
            "team_message_resolved",
            {"request": request.to_dict(), "response": response_message.to_dict()},
        )
        self._emit_team_event(
            "team_mailbox_message_resolved",
            session_id=clean_session_id,
            job_id=str(responder_job_id or request.source_job_id or ""),
            agent_name=str(responder_agent or request.source_agent or ""),
            payload={"request": request.to_dict(), "response": response_message.to_dict(), "decision": clean_decision},
        )
        return request, response_message

    def team_mailbox_summary(self, session_id: str, *, channel_id: str = "") -> Dict[str, object]:
        messages = self.list_team_messages(
            session_id,
            channel_id=channel_id,
            status_filter="open",
            limit=500,
        )
        channels = self.list_team_channels(session_id, status_filter="active")
        latest = messages[-1].to_dict() if messages else {}
        summary = {
            "active_channel_count": len(channels),
            "open_message_count": len(messages),
            "pending_question_count": sum(1 for item in messages if item.message_type == "question_request"),
            "pending_approval_count": sum(1 for item in messages if item.message_type == "approval_request"),
            "open_handoff_count": sum(1 for item in messages if item.message_type == "handoff"),
            "latest_open_message": latest,
        }
        self._emit_team_event(
            "team_mailbox_digest_created",
            session_id=str(session_id or "").strip(),
            payload=summary,
        )
        return summary

    def claim_mailbox_messages(self, job_id: str) -> List[WorkerMailboxMessage]:
        messages = self.transcript_store.load_mailbox_messages(job_id)
        claimed: List[WorkerMailboxMessage] = []
        remaining: List[WorkerMailboxMessage] = []
        now = utc_now_iso()
        for item in messages:
            if item.direction == "to_worker" and item.status == "queued":
                item.status = "delivered"
                item.resolved_at = now
                claimed.append(item)
                self._append_mailbox_audit(job_id, "mailbox_claimed", item)
                continue
            if item.message_type in MAILBOX_REQUEST_TYPES and item.status == "open":
                remaining.append(item)
                continue
            if item.direction == "to_worker" and item.status == "queued":
                remaining.append(item)
                continue
        self.transcript_store.overwrite_mailbox(job_id, remaining)
        return claimed

    @staticmethod
    def render_mailbox_prompt(messages: List[WorkerMailboxMessage]) -> str:
        rendered: List[str] = []
        for item in messages:
            if item.message_type == "message":
                if item.content.strip():
                    rendered.append(item.content.strip())
                continue
            if item.message_type == "question_response":
                rendered.append(
                    (
                        f"<worker-mailbox-response type=\"question\" request_id=\"{item.response_to}\">\n"
                        f"{item.content.strip()}\n"
                        "</worker-mailbox-response>"
                    ).strip()
                )
                continue
            if item.message_type == "approval_response":
                decision = str(dict(item.payload or {}).get("decision") or "").strip()
                rendered.append(
                    (
                        f"<worker-mailbox-response type=\"approval\" request_id=\"{item.response_to}\" decision=\"{decision}\">\n"
                        f"{item.content.strip()}\n"
                        "Continue only within your existing allowed tools and scope.\n"
                        "</worker-mailbox-response>"
                    ).strip()
                )
        return "\n\n".join(part for part in rendered if part.strip()).strip()

    def _queue_job_for_mailbox_delivery(self, job: JobRecord) -> None:
        if job.status in {"waiting_message", "failed", "queued", "budget_blocked", "completed"}:
            job.status = "queued"
            job.scheduler_state = "queued"
            job.updated_at = utc_now_iso()
            job.budget_block_reason = ""
            job.metadata.pop("pending_mailbox_request", None)
            self.transcript_store.persist_job_state(job)

    def _append_mailbox_audit(
        self,
        job_id: str,
        kind: str,
        message: WorkerMailboxMessage,
        *,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        row = {
            "kind": kind,
            "created_at": utc_now_iso(),
            "message": message.to_dict(),
        }
        if extra:
            row.update(dict(extra))
        self.transcript_store.append_job_transcript(job_id, row)

    def enqueue_agent_message(
        self,
        *,
        session_id: str,
        source_agent: str,
        target_agent: str,
        content: str,
        create_job: Callable[[], JobRecord],
        description: str = "",
        allowed_target_agents: Optional[List[str]] = None,
        source_job_id: str = "",
        target_job_id: str = "",
        reuse_running_job: bool = True,
        max_delegation_depth: int = 3,
        source_delegation_depth: int = 0,
    ) -> AgentDispatchOutcome:
        clean_session_id = str(session_id or "").strip()
        clean_source_agent = str(source_agent or "").strip()
        clean_target_agent = str(target_agent or "").strip()
        clean_content = str(content or "").strip()
        clean_description = str(description or "").strip()
        if not clean_session_id:
            raise ValueError("session_id is required for agent-to-agent dispatch.")
        if not clean_target_agent:
            raise ValueError("target_agent is required for agent-to-agent dispatch.")
        if not clean_content:
            raise ValueError("content is required for agent-to-agent dispatch.")
        if clean_source_agent and clean_source_agent == clean_target_agent:
            raise ValueError("Agents cannot dispatch peer messages to themselves.")
        if allowed_target_agents is not None and clean_target_agent not in {
            str(item).strip() for item in allowed_target_agents if str(item).strip()
        }:
            raise ValueError(f"Agent '{clean_target_agent}' is not allowed.")

        next_depth = _coerce_nonnegative_int(source_delegation_depth, 0) + 1
        if next_depth > max(1, int(max_delegation_depth or 1)):
            raise ValueError(
                f"Peer delegation depth exceeded the maximum of {max(1, int(max_delegation_depth or 1))}."
            )

        if source_job_id:
            source_job = self.get_job(source_job_id)
            if source_job is not None and source_job.session_id != clean_session_id:
                raise ValueError("Peer agent dispatch must stay within the current session.")

        target_job: JobRecord | None = None
        reused_existing_job = False
        if target_job_id:
            candidate = self.get_job(target_job_id)
            if candidate is None:
                raise ValueError(f"Target job '{target_job_id}' was not found.")
            if candidate.session_id != clean_session_id:
                raise ValueError("Target job is not part of the current session.")
            if candidate.agent_name != clean_target_agent:
                raise ValueError(
                    f"Target job '{target_job_id}' belongs to agent '{candidate.agent_name}', not '{clean_target_agent}'."
                )
            if candidate.status == "stopped":
                raise ValueError(f"Target job '{target_job_id}' is stopped and cannot be resumed.")
            target_job = candidate
            reused_existing_job = True
        elif reuse_running_job:
            candidates = [
                job
                for job in self.list_jobs(session_id=clean_session_id)
                if job.agent_name == clean_target_agent and job.status in ACTIVE_JOB_STATUSES
            ]
            if len(candidates) == 1:
                target_job = candidates[0]
                reused_existing_job = True

        mailbox_message: WorkerMailboxMessage | None = None
        if target_job is None:
            target_job = create_job()
            if target_job.session_id != clean_session_id:
                raise ValueError("Created peer job is not bound to the current session.")
            if target_job.agent_name != clean_target_agent:
                raise ValueError("Created peer job target agent did not match the dispatch target.")
            target_job.metadata["delegation_depth"] = next_depth
            self.transcript_store.persist_job_state(target_job)
        else:
            mailbox_message = self.enqueue_message(
                target_job.job_id,
                clean_content,
                sender=clean_source_agent or "peer",
                metadata={
                    "peer_dispatch": True,
                    "source_agent": clean_source_agent,
                    "source_job_id": source_job_id,
                    "description": clean_description,
                    "delegation_depth": next_depth,
                },
            )
            if mailbox_message is None:
                raise ValueError(f"Target job '{target_job.job_id}' was not found.")

        audit_row = {
            "kind": "peer_dispatch",
            "created_at": utc_now_iso(),
            "source_agent": clean_source_agent,
            "source_job_id": source_job_id,
            "target_agent": clean_target_agent,
            "target_job_id": target_job.job_id,
            "description": clean_description,
            "content": clean_content,
            "reused_existing_job": reused_existing_job,
            "delegation_depth": next_depth,
        }
        self.transcript_store.append_job_transcript(target_job.job_id, audit_row)

        notification = TaskNotification(
            job_id=target_job.job_id,
            status="queued",
            summary=self._peer_dispatch_summary(
                source_agent=clean_source_agent,
                target_agent=clean_target_agent,
                description=clean_description,
                content=clean_content,
                reused_existing_job=reused_existing_job,
            ),
            metadata={
                "agent_name": clean_target_agent,
                "source_agent": clean_source_agent,
                "source_job_id": source_job_id,
                "peer_dispatch": True,
                "reused_existing_job": reused_existing_job,
                "description": clean_description,
            },
        )
        self.append_session_notification(clean_session_id, notification)
        self._emit(
            "peer_agent_dispatch",
            target_job,
            {
                "source_agent": clean_source_agent,
                "source_job_id": source_job_id,
                "target_agent": clean_target_agent,
                "description": clean_description,
                "reused_existing_job": reused_existing_job,
                "delegation_depth": next_depth,
            },
        )
        return AgentDispatchOutcome(
            job=target_job,
            reused_existing_job=reused_existing_job,
            queued=True,
            notification=notification,
            mailbox_message=mailbox_message,
        )

    def drain_mailbox(self, job_id: str) -> List[WorkerMailboxMessage]:
        return self.claim_mailbox_messages(job_id)

    def stop_job(self, job_id: str) -> Optional[JobRecord]:
        job = self.get_job(job_id)
        if job is None:
            return None
        job.status = "stopped"
        job.scheduler_state = "stopped"
        job.updated_at = utc_now_iso()
        job.budget_block_reason = ""
        self.transcript_store.persist_job_state(job)
        self.scheduler.cancel(job)
        self._emit("job_stopped", job, {})
        return job

    def build_notification(self, job: JobRecord) -> TaskNotification:
        return TaskNotification(
            job_id=job.job_id,
            status=job.status,
            summary=job.result_summary or job.description or f"{job.agent_name} {job.status}",
            output_path=job.output_path,
            result_path=job.result_path,
            result=job.result_summary,
            metadata={"agent_name": job.agent_name},
        )

    def append_session_notification(self, session_id: str, notification: TaskNotification) -> None:
        self.transcript_store.append_session_transcript(
            session_id,
            {"kind": "notification", "notification": notification.to_dict()},
        )
        self.transcript_store.append_notification(session_id, notification)
        if self.event_sink is None:
            return
        self.event_sink.emit(
            RuntimeEvent(
                event_type="notification_appended",
                session_id=session_id,
                job_id=notification.job_id,
                agent_name=str(notification.metadata.get("agent_name") or ""),
                payload={"job_id": notification.job_id, "status": notification.status},
            )
        )

    def _run_job(self, job_id: str, runner: JobRunner) -> str:
        job = self.get_job(job_id)
        if job is None:
            raise ValueError(f"Job {job_id!r} was not found.")
        if job.status == "stopped":
            return ""
        actual_token_cost = 0
        try:
            if not self.scheduler.acquire(job):
                refreshed = self.get_job(job_id) or job
                self.transcript_store.persist_job_state(refreshed)
                return ""
            running_job = self.get_job(job_id) or job
            if running_job.status != "running":
                timestamp = utc_now_iso()
                running_job.status = "running"
                running_job.started_at = running_job.started_at or timestamp
                running_job.updated_at = timestamp
                running_job.scheduler_state = "running"
                running_job.budget_block_reason = ""
            self.transcript_store.persist_job_state(running_job)
            self._emit(
                "job_started",
                running_job,
                {
                    "queue_class": running_job.queue_class,
                    "priority": running_job.priority,
                    "estimated_token_cost": running_job.estimated_token_cost,
                },
            )
            result = runner(running_job)
            actual_token_cost = self._extract_actual_token_cost(job_id)
            refreshed = self.get_job(job_id) or running_job
            refreshed.actual_token_cost = max(int(refreshed.actual_token_cost or 0), actual_token_cost)
            if refreshed.status == "stopped":
                refreshed.scheduler_state = "stopped"
                refreshed.updated_at = utc_now_iso()
                self.transcript_store.persist_job_state(refreshed)
                return ""
            if refreshed.status == "waiting_message":
                refreshed.scheduler_state = "waiting_message"
                refreshed.updated_at = utc_now_iso()
                refreshed.result_summary = refreshed.result_summary or result[:2000]
                self.transcript_store.persist_job_state(refreshed)
                self._emit(
                    "job_waiting_message",
                    refreshed,
                    {
                        "result_preview": result[:500],
                        "actual_token_cost": refreshed.actual_token_cost,
                    },
                )
                return result
            refreshed.status = "completed"
            refreshed.scheduler_state = "completed"
            refreshed.updated_at = utc_now_iso()
            refreshed.result_summary = result[:2000]
            if not refreshed.output_path:
                output_path = self.transcript_store.artifact_path(job_id, "output.md")
                output_path.write_text(result, encoding="utf-8")
                refreshed.output_path = str(output_path)
            if not refreshed.result_path:
                result_path = self.transcript_store.artifact_path(job_id, "result.json")
                result_path.write_text(json.dumps({"result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
                refreshed.result_path = str(result_path)
            self.transcript_store.persist_job_state(refreshed)
            self._emit(
                "job_completed",
                refreshed,
                {
                    "result_preview": result[:500],
                    "actual_token_cost": refreshed.actual_token_cost,
                },
            )
            return result
        except Exception as exc:
            logger.exception("Background job %s failed", job_id)
            actual_token_cost = max(actual_token_cost, self._extract_actual_token_cost(job_id))
            failed = self.get_job(job_id) or job
            failed.status = "failed"
            failed.scheduler_state = "failed"
            failed.updated_at = utc_now_iso()
            failed.last_error = str(exc)
            failed.result_summary = str(exc)
            failed.actual_token_cost = max(int(failed.actual_token_cost or 0), actual_token_cost)
            self.transcript_store.persist_job_state(failed)
            self._emit(
                "job_failed",
                failed,
                {"error": str(exc), "actual_token_cost": failed.actual_token_cost},
            )
            return ""
        finally:
            finalized = self.get_job(job_id) or job
            final_actual_tokens = max(
                actual_token_cost,
                int(getattr(finalized, "actual_token_cost", 0) or 0),
            )
            finalized.actual_token_cost = final_actual_tokens
            self.transcript_store.persist_job_state(finalized)
            self.scheduler.complete(finalized, actual_token_cost=final_actual_tokens)
            with self._lock:
                current = self._threads.get(job_id)
                if current is not None and current is threading.current_thread():
                    del self._threads[job_id]

    def _extract_actual_token_cost(self, job_id: str) -> int:
        total_tokens = 0
        for event in self.transcript_store.load_job_events(job_id):
            if str(event.get("event_type") or "") != "model_end":
                continue
            payload = dict(event.get("payload") or {})
            usage = extract_token_usage(payload.get("token_usage"))
            if not usage.get("total_tokens"):
                usage = extract_token_usage(payload.get("llm_output"))
            if not usage.get("total_tokens"):
                usage = extract_token_usage(payload)
            total_tokens += int(usage.get("total_tokens") or 0)
        return total_tokens

    def _emit(self, event_type: str, job: JobRecord, payload: Dict[str, object]) -> None:
        if self.event_sink is None:
            return
        self.event_sink.emit(
            RuntimeEvent(
                event_type=event_type,
                session_id=job.session_id,
                job_id=job.job_id,
                agent_name=job.agent_name,
                payload=dict(payload),
            )
        )

    def _emit_team_event(
        self,
        event_type: str,
        *,
        session_id: str,
        job_id: str = "",
        agent_name: str = "",
        payload: Optional[Dict[str, object]] = None,
    ) -> None:
        if self.event_sink is None:
            return
        self.event_sink.emit(
            RuntimeEvent(
                event_type=event_type,
                session_id=str(session_id or ""),
                job_id=str(job_id or ""),
                agent_name=str(agent_name or ""),
                payload=dict(payload or {}),
            )
        )

    def _append_team_mailbox_audit(self, session_id: str, kind: str, payload: Dict[str, object]) -> None:
        self.transcript_store.append_team_mailbox_audit(
            session_id,
            {
                "kind": kind,
                "created_at": utc_now_iso(),
                **dict(payload or {}),
            },
        )

    @staticmethod
    def _peer_dispatch_summary(
        *,
        source_agent: str,
        target_agent: str,
        description: str,
        content: str,
        reused_existing_job: bool,
    ) -> str:
        detail = description.strip() or content.strip()
        detail = " ".join(detail.split())
        if len(detail) > 140:
            detail = detail[:137].rstrip() + "..."
        action = "continued" if reused_existing_job else "queued"
        if source_agent:
            return f"{source_agent} {action} {target_agent}: {detail}".strip()
        return f"{target_agent} {action}: {detail}".strip()
