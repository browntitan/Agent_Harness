from __future__ import annotations

import re
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List

from agentic_chatbot_next.contracts.jobs import JobRecord
from agentic_chatbot_next.contracts.messages import utc_now_iso

QUEUE_CLASSES = ("urgent", "interactive", "background")
QUEUE_CLASS_WEIGHTS = {
    "urgent": 8,
    "interactive": 3,
    "background": 1,
}
_TOKEN_RE = re.compile(r"\S+")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: Any) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        return _now()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return _now()
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _estimate_text_tokens(text: str) -> int:
    token_count = len(_TOKEN_RE.findall(str(text or "")))
    if token_count:
        return max(1, int(token_count * 1.35))
    return max(1, len(str(text or "")) // 4)


def normalize_queue_class(value: Any, *, default: str = "interactive") -> str:
    clean = str(value or "").strip().lower()
    return clean if clean in set(QUEUE_CLASSES) else default


def normalize_priority(value: Any, *, default: str = "interactive") -> str:
    return normalize_queue_class(value, default=default)


@dataclass
class TenantBudgetState:
    tenant_id: str
    available_tokens: float
    last_refill_at: datetime


@dataclass
class SchedulerEntry:
    job_id: str
    session_id: str
    agent_name: str
    tenant_id: str
    user_id: str
    queue_class: str
    priority: str
    estimated_token_cost: int
    enqueued_at: str
    reserved_tokens: int = 0
    budget_block_reason: str = ""
    state: str = "queued"
    job: JobRecord | None = field(default=None, repr=False)

    @property
    def ready_for_dispatch(self) -> bool:
        return self.state == "queued"


class WorkerScheduler:
    def __init__(
        self,
        *,
        enabled: bool,
        max_concurrency: int,
        urgent_reserved_slots: int,
        tenant_budget_tokens_per_minute: int,
        tenant_budget_burst_tokens: int,
        emit_event: Callable[[str, JobRecord, Dict[str, object]], None] | None = None,
        persist_job: Callable[[JobRecord], None] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_concurrency = max(1, int(max_concurrency))
        self.urgent_reserved_slots = max(0, min(self.max_concurrency - 1, int(urgent_reserved_slots)))
        self.tenant_budget_tokens_per_minute = max(0, int(tenant_budget_tokens_per_minute))
        self.tenant_budget_burst_tokens = max(
            self.tenant_budget_tokens_per_minute,
            int(tenant_budget_burst_tokens or self.tenant_budget_tokens_per_minute),
        )
        self.emit_event = emit_event
        self.persist_job = persist_job
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._entries: Dict[str, SchedulerEntry] = {}
        self._queues: Dict[str, Dict[str, Deque[str]]] = {
            queue_class: defaultdict(deque)
            for queue_class in QUEUE_CLASSES
        }
        self._tenant_order: Dict[str, Deque[str]] = {
            queue_class: deque()
            for queue_class in QUEUE_CLASSES
        }
        self._running_jobs: Dict[str, SchedulerEntry] = {}
        self._tenant_budgets: Dict[str, TenantBudgetState] = {}
        self._class_cycle: List[str] = [
            queue_class
            for queue_class, weight in QUEUE_CLASS_WEIGHTS.items()
            for _ in range(weight)
        ]
        self._class_index = 0

    def estimate_job_token_cost(self, job: JobRecord, *, settings: Any | None = None) -> int:
        if int(getattr(job, "estimated_token_cost", 0) or 0) > 0:
            return int(job.estimated_token_cost)
        prompt_tokens = _estimate_text_tokens(job.prompt)
        long_output = dict((job.metadata or {}).get("long_output") or {})
        worker_request = dict((job.metadata or {}).get("worker_request") or {})
        worker_metadata = dict(worker_request.get("metadata") or {})
        requested_max = 0
        for candidate in (
            long_output.get("chat_max_output_tokens"),
            worker_metadata.get("chat_max_output_tokens"),
            (getattr(settings, "chat_max_output_tokens", None) if settings is not None else None),
        ):
            try:
                requested_max = max(requested_max, int(candidate or 0))
            except (TypeError, ValueError):
                continue
        completion_budget = requested_max if requested_max > 0 else max(256, prompt_tokens)
        return max(1, prompt_tokens + completion_budget)

    def acquire(self, job: JobRecord) -> bool:
        if not self.enabled:
            return True
        with self._condition:
            if str(getattr(job, "status", "") or "") == "stopped":
                return False
            entry = self._entries.get(job.job_id)
            if entry is None:
                entry = self._enqueue_entry_locked(job)
            while True:
                if str(getattr(job, "status", "") or "") == "stopped":
                    self._cancel_waiting_entry_locked(job.job_id)
                    return False
                self._refresh_budget_blocks_locked()
                selected = self._select_next_entry_locked()
                if selected is not None and selected.job_id == job.job_id:
                    self._dispatch_entry_locked(job, selected)
                    return True
                self._condition.wait(timeout=0.25)

    def complete(self, job: JobRecord, *, actual_token_cost: int = 0) -> None:
        if not self.enabled:
            return
        with self._condition:
            entry = self._running_jobs.pop(job.job_id, None) or self._entries.pop(job.job_id, None)
            if entry is None:
                self._condition.notify_all()
                return
            self._reconcile_budget_locked(job, entry, actual_token_cost=actual_token_cost)
            self._condition.notify_all()

    def cancel(self, job: JobRecord) -> None:
        if not self.enabled:
            return
        with self._condition:
            if job.job_id in self._running_jobs:
                self._condition.notify_all()
                return
            self._cancel_waiting_entry_locked(job.job_id)
            self._condition.notify_all()

    def snapshot(self) -> Dict[str, Any]:
        with self._condition:
            now = _now()
            queue_depths = {
                queue_class: sum(
                    1
                    for entry in self._entries.values()
                    if entry.queue_class == queue_class and entry.job_id not in self._running_jobs
                )
                for queue_class in QUEUE_CLASSES
            }
            oldest_wait_seconds = {}
            for queue_class in QUEUE_CLASSES:
                waits = [
                    max(0.0, (now - _parse_dt(entry.enqueued_at)).total_seconds())
                    for entry in self._entries.values()
                    if entry.queue_class == queue_class and entry.job_id not in self._running_jobs
                ]
                oldest_wait_seconds[queue_class] = round(max(waits) if waits else 0.0, 3)
            tenant_rows: List[Dict[str, Any]] = []
            tenant_ids = {
                entry.tenant_id
                for entry in self._entries.values()
                if entry.tenant_id
            } | set(self._tenant_budgets.keys())
            for tenant_id in sorted(tenant_ids):
                waiting = [
                    entry for entry in self._entries.values()
                    if entry.tenant_id == tenant_id and entry.job_id not in self._running_jobs
                ]
                running = [
                    entry for entry in self._running_jobs.values()
                    if entry.tenant_id == tenant_id
                ]
                budget = self._tenant_budgets.get(tenant_id)
                tenant_rows.append(
                    {
                        "tenant_id": tenant_id,
                        "queued_jobs": len(waiting),
                        "running_jobs": len(running),
                        "budget_blocked_jobs": sum(1 for entry in waiting if entry.state == "budget_blocked"),
                        "available_tokens": round(float(getattr(budget, "available_tokens", self.tenant_budget_burst_tokens)), 2),
                    }
                )
            return {
                "enabled": self.enabled,
                "max_concurrency": self.max_concurrency,
                "running_jobs": len(self._running_jobs),
                "available_slots": max(0, self.max_concurrency - len(self._running_jobs)),
                "reserved_urgent_slots": self.urgent_reserved_slots,
                "urgent_backlog": bool(queue_depths["urgent"]),
                "queue_depths": queue_depths,
                "oldest_wait_seconds": oldest_wait_seconds,
                "budget_blocked_jobs": sum(1 for entry in self._entries.values() if entry.state == "budget_blocked"),
                "tenant_budget_health": tenant_rows[:12],
            }

    def _enqueue_entry_locked(self, job: JobRecord) -> SchedulerEntry:
        queue_class = normalize_queue_class(getattr(job, "queue_class", "interactive"))
        priority = normalize_priority(getattr(job, "priority", queue_class), default=queue_class)
        tenant_id = str(getattr(job, "tenant_id", "") or "")
        entry = SchedulerEntry(
            job_id=job.job_id,
            session_id=str(job.session_id or ""),
            agent_name=str(job.agent_name or ""),
            tenant_id=tenant_id,
            user_id=str(getattr(job, "user_id", "") or ""),
            queue_class=queue_class,
            priority=priority,
            estimated_token_cost=max(0, int(getattr(job, "estimated_token_cost", 0) or 0)),
            enqueued_at=str(getattr(job, "enqueued_at", "") or job.created_at or utc_now_iso()),
            job=job,
        )
        if not entry.estimated_token_cost:
            entry.estimated_token_cost = max(1, _estimate_text_tokens(job.prompt) + 256)
            job.estimated_token_cost = entry.estimated_token_cost
        self._entries[job.job_id] = entry
        tenant_queue = self._queues[queue_class][tenant_id]
        tenant_queue.append(job.job_id)
        if tenant_id not in self._tenant_order[queue_class]:
            self._tenant_order[queue_class].append(tenant_id)
        if self._try_reserve_budget_locked(job, entry):
            entry.state = "queued"
            job.scheduler_state = "queued"
            job.budget_block_reason = ""
        else:
            entry.state = "budget_blocked"
            job.scheduler_state = "budget_blocked"
            job.budget_block_reason = entry.budget_block_reason
            self._emit(
                "scheduler_budget_blocked",
                job,
                {
                    "queue_class": queue_class,
                    "priority": priority,
                    "estimated_token_cost": entry.estimated_token_cost,
                    "reason": entry.budget_block_reason,
                },
            )
        job.queue_class = queue_class
        job.priority = priority
        job.enqueued_at = entry.enqueued_at
        self._emit(
            "scheduler_job_enqueued",
            job,
            {
                "queue_class": queue_class,
                "priority": priority,
                "estimated_token_cost": entry.estimated_token_cost,
                "scheduler_state": job.scheduler_state,
            },
        )
        self._persist(job)
        return entry

    def _dispatch_entry_locked(self, job: JobRecord, entry: SchedulerEntry) -> None:
        self._running_jobs[job.job_id] = entry
        self._entries.pop(job.job_id, None)
        tenant_queue = self._queues[entry.queue_class].get(entry.tenant_id)
        if tenant_queue:
            if tenant_queue and tenant_queue[0] == job.job_id:
                tenant_queue.popleft()
            else:
                try:
                    tenant_queue.remove(job.job_id)
                except ValueError:
                    pass
            if tenant_queue:
                self._tenant_order[entry.queue_class].rotate(-1)
            else:
                self._queues[entry.queue_class].pop(entry.tenant_id, None)
                try:
                    self._tenant_order[entry.queue_class].remove(entry.tenant_id)
                except ValueError:
                    pass
        started_at = utc_now_iso()
        job.status = "running"
        job.started_at = started_at
        job.updated_at = started_at
        job.scheduler_state = "running"
        job.budget_block_reason = ""
        self._persist(job)
        self._emit(
            "scheduler_job_dispatched",
            job,
            {
                "queue_class": entry.queue_class,
                "priority": entry.priority,
                "estimated_token_cost": entry.estimated_token_cost,
            },
        )
        self._condition.notify_all()

    def _select_next_entry_locked(self) -> SchedulerEntry | None:
        if len(self._running_jobs) >= self.max_concurrency:
            return None
        urgent_ready = self._has_ready_entries_locked("urgent")
        remaining_slots = self.max_concurrency - len(self._running_jobs)
        if urgent_ready and remaining_slots <= 0:
            return None
        reserved_urgent_slots = max(0, self.urgent_reserved_slots)
        for offset in range(len(self._class_cycle)):
            queue_class = self._class_cycle[(self._class_index + offset) % len(self._class_cycle)]
            if queue_class != "urgent":
                remaining_slots = self.max_concurrency - len(self._running_jobs)
                if urgent_ready and reserved_urgent_slots > 0 and remaining_slots <= reserved_urgent_slots:
                    continue
            entry = self._peek_next_entry_for_class_locked(queue_class)
            if entry is None:
                continue
            self._class_index = (self._class_index + offset + 1) % len(self._class_cycle)
            return entry
        return None

    def _peek_next_entry_for_class_locked(self, queue_class: str) -> SchedulerEntry | None:
        order = self._tenant_order[queue_class]
        if not order:
            return None
        inspected = 0
        while inspected < len(order):
            tenant_id = order[0]
            queue = self._queues[queue_class].get(tenant_id)
            if not queue:
                order.popleft()
                continue
            while queue:
                entry = self._entries.get(queue[0])
                if entry is None:
                    queue.popleft()
                    continue
                if entry.ready_for_dispatch:
                    return entry
                break
            order.rotate(-1)
            inspected += 1
        return None

    def _has_ready_entries_locked(self, queue_class: str) -> bool:
        return self._peek_next_entry_for_class_locked(queue_class) is not None

    def _cancel_waiting_entry_locked(self, job_id: str) -> None:
        entry = self._entries.pop(job_id, None)
        if entry is None:
            return
        queue = self._queues[entry.queue_class].get(entry.tenant_id)
        if queue is not None:
            try:
                queue.remove(job_id)
            except ValueError:
                pass
            if not queue:
                self._queues[entry.queue_class].pop(entry.tenant_id, None)
                try:
                    self._tenant_order[entry.queue_class].remove(entry.tenant_id)
                except ValueError:
                    pass
        if entry.reserved_tokens > 0 and entry.tenant_id:
            budget = self._budget_state_locked(entry.tenant_id)
            budget.available_tokens = min(
                float(self.tenant_budget_burst_tokens),
                budget.available_tokens + float(entry.reserved_tokens),
            )
        if entry.job is not None:
            entry.job.updated_at = utc_now_iso()
            entry.job.budget_block_reason = ""
            self._persist(entry.job)

    def _refresh_budget_blocks_locked(self) -> None:
        if self.tenant_budget_tokens_per_minute <= 0:
            return
        for entry in sorted(self._entries.values(), key=lambda item: item.enqueued_at):
            if entry.state != "budget_blocked":
                continue
            job = entry.job
            if job is None:
                continue
            if self._try_reserve_budget_locked(job, entry):
                entry.state = "queued"
                job.scheduler_state = "queued"
                job.budget_block_reason = ""
                job.updated_at = utc_now_iso()
                self._persist(job)

    def _budget_state_locked(self, tenant_id: str) -> TenantBudgetState:
        state = self._tenant_budgets.get(tenant_id)
        if state is None:
            state = TenantBudgetState(
                tenant_id=tenant_id,
                available_tokens=float(self.tenant_budget_burst_tokens),
                last_refill_at=_now(),
            )
            self._tenant_budgets[tenant_id] = state
        return state

    def _try_reserve_budget_locked(self, job: JobRecord, entry: SchedulerEntry) -> bool:
        if self.tenant_budget_tokens_per_minute <= 0 or not entry.tenant_id or entry.estimated_token_cost <= 0:
            entry.reserved_tokens = entry.estimated_token_cost
            entry.budget_block_reason = ""
            return True
        budget = self._budget_state_locked(entry.tenant_id)
        self._refill_budget_locked(budget)
        if budget.available_tokens >= float(entry.estimated_token_cost):
            budget.available_tokens -= float(entry.estimated_token_cost)
            entry.reserved_tokens = entry.estimated_token_cost
            entry.budget_block_reason = ""
            job.budget_block_reason = ""
            return True
        entry.reserved_tokens = 0
        entry.budget_block_reason = "tenant_token_budget_exhausted"
        job.budget_block_reason = entry.budget_block_reason
        return False

    def _reconcile_budget_locked(
        self,
        job: JobRecord,
        entry: SchedulerEntry,
        *,
        actual_token_cost: int,
    ) -> None:
        if self.tenant_budget_tokens_per_minute <= 0 or not entry.tenant_id:
            return
        estimated = max(0, int(entry.reserved_tokens or entry.estimated_token_cost))
        actual = max(0, int(actual_token_cost or estimated))
        delta = actual - estimated
        budget = self._budget_state_locked(entry.tenant_id)
        self._refill_budget_locked(budget)
        if delta < 0:
            budget.available_tokens = min(
                float(self.tenant_budget_burst_tokens),
                budget.available_tokens + float(abs(delta)),
            )
        elif delta > 0:
            budget.available_tokens -= float(delta)
        self._emit(
            "scheduler_budget_reconciled",
            job,
            {
                "tenant_id": entry.tenant_id,
                "estimated_token_cost": estimated,
                "actual_token_cost": actual,
                "delta_tokens": delta,
                "available_tokens": round(budget.available_tokens, 2),
            },
        )

    def _refill_budget_locked(self, budget: TenantBudgetState) -> None:
        if self.tenant_budget_tokens_per_minute <= 0:
            return
        now = _now()
        elapsed_seconds = max(0.0, (now - budget.last_refill_at).total_seconds())
        if elapsed_seconds <= 0:
            return
        refill_amount = (float(self.tenant_budget_tokens_per_minute) / 60.0) * elapsed_seconds
        budget.available_tokens = min(
            float(self.tenant_budget_burst_tokens),
            budget.available_tokens + refill_amount,
        )
        budget.last_refill_at = now

    def _emit(self, event_type: str, job: JobRecord, payload: Dict[str, object]) -> None:
        if self.emit_event is None:
            return
        self.emit_event(event_type, job, payload)

    def _persist(self, job: JobRecord) -> None:
        if self.persist_job is None:
            return
        self.persist_job(job)


__all__ = [
    "QUEUE_CLASSES",
    "QUEUE_CLASS_WEIGHTS",
    "WorkerScheduler",
    "normalize_priority",
    "normalize_queue_class",
]
