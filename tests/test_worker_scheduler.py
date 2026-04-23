from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from agentic_chatbot_next.contracts.jobs import JobRecord
from agentic_chatbot_next.contracts.messages import utc_now_iso
from agentic_chatbot_next.runtime.context import RuntimePaths
from agentic_chatbot_next.runtime.job_manager import RuntimeJobManager
from agentic_chatbot_next.runtime.transcript_store import RuntimeTranscriptStore
from agentic_chatbot_next.runtime.worker_scheduler import WorkerScheduler


def _job(
    job_id: str,
    *,
    tenant_id: str,
    queue_class: str = "interactive",
    priority: str | None = None,
    estimated_token_cost: int = 32,
) -> JobRecord:
    now = utc_now_iso()
    return JobRecord(
        job_id=job_id,
        session_id=f"{tenant_id}:user:conv",
        agent_name="worker",
        status="queued",
        prompt=f"prompt for {job_id}",
        tenant_id=tenant_id,
        user_id="user",
        priority=priority or queue_class,
        queue_class=queue_class,
        enqueued_at=now,
        created_at=now,
        updated_at=now,
        estimated_token_cost=estimated_token_cost,
    )


def _paths(tmp_path: Path) -> RuntimePaths:
    settings = SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        workspace_dir=tmp_path / "workspaces",
        memory_dir=tmp_path / "memory",
    )
    return RuntimePaths.from_settings(settings)


def test_worker_scheduler_prioritizes_urgent_jobs_over_background_jobs() -> None:
    scheduler = WorkerScheduler(
        enabled=True,
        max_concurrency=1,
        urgent_reserved_slots=1,
        tenant_budget_tokens_per_minute=0,
        tenant_budget_burst_tokens=0,
    )
    running = _job("job-running", tenant_id="tenant-a", queue_class="background")
    assert scheduler.acquire(running) is True

    order: list[str] = []
    released = threading.Event()

    def wait_for_slot(job: JobRecord) -> None:
        if scheduler.acquire(job):
            order.append(job.job_id)
            released.wait(timeout=2)
            scheduler.complete(job)

    background = _job("job-background", tenant_id="tenant-a", queue_class="background")
    urgent = _job("job-urgent", tenant_id="tenant-b", queue_class="urgent")
    background_thread = threading.Thread(target=wait_for_slot, args=(background,), daemon=True)
    urgent_thread = threading.Thread(target=wait_for_slot, args=(urgent,), daemon=True)
    background_thread.start()
    urgent_thread.start()

    time.sleep(0.2)
    scheduler.complete(running)

    for _ in range(20):
        if order:
            break
        time.sleep(0.05)

    released.set()
    background_thread.join(timeout=2)
    urgent_thread.join(timeout=2)

    assert order[0] == "job-urgent"


def test_worker_scheduler_budget_blocks_then_unblocks_when_tokens_are_available() -> None:
    scheduler = WorkerScheduler(
        enabled=True,
        max_concurrency=1,
        urgent_reserved_slots=0,
        tenant_budget_tokens_per_minute=60,
        tenant_budget_burst_tokens=200,
    )
    blocked_job = _job("job-budget", tenant_id="tenant-a", estimated_token_cost=140)
    acquired: list[bool] = []

    with scheduler._condition:  # noqa: SLF001 - unit test for scheduler behavior
        budget = scheduler._budget_state_locked("tenant-a")  # noqa: SLF001
        budget.available_tokens = 100.0

    def wait_for_budget() -> None:
        acquired.append(scheduler.acquire(blocked_job))
        if acquired[-1]:
            scheduler.complete(blocked_job, actual_token_cost=140)

    thread = threading.Thread(target=wait_for_budget, daemon=True)
    thread.start()
    time.sleep(0.2)
    snapshot = scheduler.snapshot()
    assert snapshot["budget_blocked_jobs"] == 1

    with scheduler._condition:  # noqa: SLF001 - unit test for scheduler behavior
        budget = scheduler._budget_state_locked("tenant-a")  # noqa: SLF001
        budget.available_tokens = 200.0
        scheduler._refresh_budget_blocks_locked()  # noqa: SLF001
        scheduler._condition.notify_all()

    thread.join(timeout=2)
    assert acquired == [True]


def test_worker_scheduler_reconciles_actual_token_cost_back_into_tenant_budget() -> None:
    scheduler = WorkerScheduler(
        enabled=True,
        max_concurrency=1,
        urgent_reserved_slots=0,
        tenant_budget_tokens_per_minute=60,
        tenant_budget_burst_tokens=100,
    )
    job = _job("job-reconcile", tenant_id="tenant-a", estimated_token_cost=80)

    assert scheduler.acquire(job) is True
    scheduler.complete(job, actual_token_cost=40)

    snapshot = scheduler.snapshot()
    tenant_row = snapshot["tenant_budget_health"][0]
    assert round(float(tenant_row["available_tokens"]), 2) == 60.0


def test_runtime_job_manager_inline_jobs_wait_for_scheduler_capacity(tmp_path: Path) -> None:
    settings = SimpleNamespace(
        worker_scheduler_enabled=True,
        worker_scheduler_urgent_reserved_slots=0,
        worker_scheduler_tenant_budget_tokens_per_minute=0,
        worker_scheduler_tenant_budget_burst_tokens=0,
        max_worker_concurrency=1,
        chat_max_output_tokens=256,
    )
    transcript_store = RuntimeTranscriptStore(_paths(tmp_path))
    manager = RuntimeJobManager(
        transcript_store,
        settings=settings,
        max_worker_concurrency=1,
    )

    first_job = manager.create_job(
        agent_name="worker",
        prompt="first",
        session_id="tenant-a:user:conv",
        tenant_id="tenant-a",
        user_id="user",
    )
    second_job = manager.create_job(
        agent_name="worker",
        prompt="second",
        session_id="tenant-b:user:conv",
        tenant_id="tenant-b",
        user_id="user",
    )

    release_first = threading.Event()
    second_completed = threading.Event()

    def first_runner(current_job: JobRecord) -> str:
        release_first.wait(timeout=2)
        return f"done:{current_job.prompt}"

    def second_runner(current_job: JobRecord) -> str:
        second_completed.set()
        return f"done:{current_job.prompt}"

    manager.start_background_job(first_job, first_runner)
    time.sleep(0.2)

    inline_thread = threading.Thread(
        target=lambda: manager.run_job_inline(second_job, second_runner),
        daemon=True,
    )
    inline_thread.start()
    time.sleep(0.2)

    queued_second = manager.get_job(second_job.job_id)
    assert queued_second is not None
    assert queued_second.status == "queued"
    assert not second_completed.is_set()

    release_first.set()
    inline_thread.join(timeout=3)

    finished_second = manager.get_job(second_job.job_id)
    assert finished_second is not None
    assert finished_second.status == "completed"
    assert second_completed.is_set()
