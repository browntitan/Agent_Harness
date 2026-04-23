from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_chatbot_next.contracts.jobs import WorkerMailboxMessage
from agentic_chatbot_next.runtime.context import RuntimePaths
from agentic_chatbot_next.runtime.job_manager import RuntimeJobManager
from agentic_chatbot_next.runtime.notification_store import NotificationStore
from agentic_chatbot_next.runtime.transcript_store import RuntimeTranscriptStore


def _paths(tmp_path: Path) -> RuntimePaths:
    settings = SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        workspace_dir=tmp_path / "workspaces",
        memory_dir=tmp_path / "memory",
    )
    return RuntimePaths.from_settings(settings)


def test_job_creation_mailbox_drain_and_notification_persistence(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    transcript_store = RuntimeTranscriptStore(paths)
    job_manager = RuntimeJobManager(transcript_store)
    notifications = NotificationStore(transcript_store)

    job = job_manager.create_job(
        agent_name="utility",
        prompt="compute",
        session_id="tenant:user:conversation",
        description="utility worker",
        metadata={"test": True},
    )
    assert transcript_store.load_job_state(job.job_id) is not None

    mailbox = job_manager.enqueue_message(job.job_id, "follow up", sender="parent")
    assert mailbox is not None
    drained = job_manager.drain_mailbox(job.job_id)
    assert [item.content for item in drained] == ["follow up"]
    assert job_manager.drain_mailbox(job.job_id) == []

    result = job_manager.run_job_inline(job, lambda current_job: f"done:{current_job.prompt}")
    assert result == "done:compute"

    persisted_job = transcript_store.load_job_state(job.job_id)
    assert persisted_job is not None
    assert persisted_job.status == "completed"
    assert Path(persisted_job.output_path).read_text(encoding="utf-8") == "done:compute"
    assert json.loads(Path(persisted_job.result_path).read_text(encoding="utf-8")) == {"result": "done:compute"}

    notification = job_manager.build_notification(persisted_job)
    notifications.append(persisted_job.session_id, notification)
    drained_notifications = notifications.drain(persisted_job.session_id)
    assert [item.job_id for item in drained_notifications] == [persisted_job.job_id]
    session_notifications_path = paths.session_notifications_path(persisted_job.session_id)
    assert session_notifications_path.read_text(encoding="utf-8") == ""


def test_worker_mailbox_message_hydrates_legacy_rows() -> None:
    message = WorkerMailboxMessage.from_dict(
        {
            "job_id": "job_123",
            "content": "legacy follow-up",
            "sender": "parent",
        }
    )

    assert message.job_id == "job_123"
    assert message.content == "legacy follow-up"
    assert message.message_id.startswith("msg_")
    assert message.message_type == "message"
    assert message.direction == "to_worker"
    assert message.status == "queued"
    assert message.requires_response is False


def test_typed_mailbox_request_response_and_claim_flow(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    transcript_store = RuntimeTranscriptStore(paths)
    job_manager = RuntimeJobManager(transcript_store)
    job = job_manager.create_job(
        agent_name="utility",
        prompt="compute",
        session_id="tenant:user:conversation",
        description="utility worker",
    )

    request = job_manager.open_worker_request(
        job.job_id,
        request_type="question_request",
        content="Which repo should I inspect?",
        sender="utility",
        payload={"options": ["api", "ui"]},
    )

    assert request is not None
    assert request.status == "open"
    assert request.requires_response is True
    waiting = transcript_store.load_job_state(job.job_id)
    assert waiting is not None
    assert waiting.status == "waiting_message"
    assert job_manager.mailbox_summary(job.job_id)["pending_question_count"] == 1

    resolved = job_manager.respond_to_request(
        job.job_id,
        request.message_id,
        response="Inspect the API repo.",
        responder="coordinator",
    )

    assert resolved is not None
    resolved_request, response = resolved
    assert resolved_request.status == "answered"
    assert response.message_type == "question_response"
    assert response.response_to == request.message_id
    resumed = transcript_store.load_job_state(job.job_id)
    assert resumed is not None
    assert resumed.status == "queued"

    claimed = job_manager.claim_mailbox_messages(job.job_id)
    assert [item.content for item in claimed] == ["Inspect the API repo."]
    assert job_manager.list_mailbox_messages(job.job_id) == []


def test_approval_request_requires_operator_response(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    transcript_store = RuntimeTranscriptStore(paths)
    job_manager = RuntimeJobManager(transcript_store)
    job = job_manager.create_job(
        agent_name="utility",
        prompt="clean up",
        session_id="tenant:user:conversation",
    )

    request = job_manager.open_worker_request(
        job.job_id,
        request_type="approval_request",
        content="Delete generated export.csv",
        sender="utility",
        payload={"action": "Delete generated export.csv"},
    )
    assert request is not None

    with pytest.raises(PermissionError):
        job_manager.respond_to_request(
            job.job_id,
            request.message_id,
            response="approved",
            responder="coordinator",
            decision="approved",
            allow_approval=False,
        )

    resolved = job_manager.respond_to_request(
        job.job_id,
        request.message_id,
        response="Approved for generated export only.",
        responder="operator",
        decision="approved",
        allow_approval=True,
    )
    assert resolved is not None
    resolved_request, response = resolved
    assert resolved_request.status == "approved"
    assert response.message_type == "approval_response"
    assert response.payload["decision"] == "approved"


def test_run_job_preserves_waiting_message_status(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    transcript_store = RuntimeTranscriptStore(paths)
    job_manager = RuntimeJobManager(transcript_store)
    job = job_manager.create_job(
        agent_name="utility",
        prompt="compute",
        session_id="tenant:user:conversation",
    )

    def runner(record):
        job_manager.open_worker_request(
            record.job_id,
            request_type="question_request",
            content="Need an answer.",
            sender="utility",
        )
        return "waiting for answer"

    result = job_manager.run_job_inline(job, runner)
    persisted = transcript_store.load_job_state(job.job_id)

    assert result == "waiting for answer"
    assert persisted is not None
    assert persisted.status == "waiting_message"
    assert persisted.output_path == ""


def test_enqueue_agent_message_creates_scoped_peer_job_and_persists_audit_rows(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    transcript_store = RuntimeTranscriptStore(paths)
    job_manager = RuntimeJobManager(transcript_store)
    session_id = "tenant:user:conversation"
    source_job = job_manager.create_job(
        agent_name="general",
        prompt="root task",
        session_id=session_id,
        description="root",
    )

    outcome = job_manager.enqueue_agent_message(
        session_id=session_id,
        source_agent="general",
        target_agent="utility",
        content="Check the indexed docs and summarize the relevant ones.",
        description="document inventory follow-up",
        allowed_target_agents=["utility"],
        source_job_id=source_job.job_id,
        source_delegation_depth=0,
        create_job=lambda: job_manager.create_job(
            agent_name="utility",
            prompt="Check the indexed docs and summarize the relevant ones.",
            session_id=session_id,
            description="document inventory follow-up",
            parent_job_id=source_job.job_id,
        ),
    )

    assert outcome.reused_existing_job is False
    assert outcome.job.agent_name == "utility"
    assert outcome.job.parent_job_id == source_job.job_id
    persisted_target = transcript_store.load_job_state(outcome.job.job_id)
    assert persisted_target is not None
    assert persisted_target.metadata["delegation_depth"] == 1

    transcript_rows = transcript_store.load_job_transcript(outcome.job.job_id)
    assert transcript_rows[-1]["kind"] == "peer_dispatch"
    assert transcript_rows[-1]["source_agent"] == "general"
    assert transcript_rows[-1]["target_agent"] == "utility"

    session_transcript = transcript_store.load_session_transcript(session_id)
    assert any(row.get("kind") == "notification" for row in session_transcript)
    notifications = NotificationStore(transcript_store).drain(session_id)
    assert notifications[0].job_id == outcome.job.job_id
    assert notifications[0].metadata["peer_dispatch"] is True


def test_enqueue_agent_message_reuses_existing_active_job_and_persists_mailbox(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    transcript_store = RuntimeTranscriptStore(paths)
    job_manager = RuntimeJobManager(transcript_store)
    session_id = "tenant:user:conversation"
    source_job = job_manager.create_job(
        agent_name="general",
        prompt="root task",
        session_id=session_id,
        description="root",
    )
    target_job = job_manager.create_job(
        agent_name="utility",
        prompt="original utility task",
        session_id=session_id,
        description="utility",
    )
    target_job.status = "waiting_message"
    transcript_store.persist_job_state(target_job)

    outcome = job_manager.enqueue_agent_message(
        session_id=session_id,
        source_agent="general",
        target_agent="utility",
        content="Continue with the narrowed inventory request.",
        description="continue utility task",
        allowed_target_agents=["utility"],
        source_job_id=source_job.job_id,
        reuse_running_job=True,
        create_job=lambda: job_manager.create_job(
            agent_name="utility",
            prompt="should not be used",
            session_id=session_id,
            description="unused",
        ),
    )

    assert outcome.reused_existing_job is True
    assert outcome.job.job_id == target_job.job_id
    mailbox = transcript_store.load_mailbox_messages(target_job.job_id)
    assert [item.content for item in mailbox] == ["Continue with the narrowed inventory request."]
    assert mailbox[0].metadata["source_agent"] == "general"
    assert mailbox[0].metadata["peer_dispatch"] is True
    refreshed = transcript_store.load_job_state(target_job.job_id)
    assert refreshed is not None
    assert refreshed.status == "queued"


def test_team_mailbox_channel_post_claim_and_response_flow(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    transcript_store = RuntimeTranscriptStore(paths)
    job_manager = RuntimeJobManager(
        transcript_store,
        settings=SimpleNamespace(
            team_mailbox_max_channels_per_session=8,
            team_mailbox_max_open_messages_per_channel=50,
            team_mailbox_claim_limit=8,
        ),
    )
    session_id = "tenant:user:conversation"
    source_job = job_manager.create_job(agent_name="general", prompt="root", session_id=session_id)
    target_job = job_manager.create_job(agent_name="utility", prompt="worker", session_id=session_id)

    channel = job_manager.create_team_channel(
        session_id=session_id,
        name="research",
        purpose="coordinate",
        created_by_job_id=source_job.job_id,
        member_agents=["general", "utility"],
        member_job_ids=[source_job.job_id, target_job.job_id],
    )
    handoff = job_manager.post_team_message(
        session_id=session_id,
        channel_id=channel.channel_id,
        content="Review the inventory.",
        source_agent="general",
        source_job_id=source_job.job_id,
        target_agents=["utility"],
        target_job_ids=[target_job.job_id],
        message_type="handoff",
    )
    question = job_manager.post_team_message(
        session_id=session_id,
        channel_id=channel.channel_id,
        content="Which collection?",
        source_agent="utility",
        source_job_id=target_job.job_id,
        target_agents=["general"],
        target_job_ids=[source_job.job_id],
        message_type="question_request",
    )

    claimed = job_manager.claim_team_messages(
        session_id,
        channel.channel_id,
        claimant_agent="utility",
        claimant_job_id=target_job.job_id,
    )
    assert [item.message_id for item in claimed] == [handoff.message_id]
    assert transcript_store.load_team_messages(session_id, channel.channel_id)[0].status == "claimed"

    request, response = job_manager.respond_team_message(
        session_id,
        channel.channel_id,
        question.message_id,
        response="Use default.",
        responder_agent="general",
        responder_job_id=source_job.job_id,
    )
    assert request.status == "answered"
    assert response.message_type == "question_response"
    summary = job_manager.team_mailbox_summary(session_id, channel_id=channel.channel_id)
    assert summary["pending_question_count"] == 0
    assert transcript_store.load_team_mailbox_audit(session_id)[-1]["kind"] == "team_message_resolved"


def test_team_mailbox_approval_requires_operator_authority(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    transcript_store = RuntimeTranscriptStore(paths)
    job_manager = RuntimeJobManager(transcript_store)
    session_id = "tenant:user:conversation"
    channel = job_manager.create_team_channel(session_id=session_id, name="approvals")
    request = job_manager.post_team_message(
        session_id=session_id,
        channel_id=channel.channel_id,
        content="Approve export?",
        source_agent="utility",
        message_type="approval_request",
    )

    with pytest.raises(PermissionError):
        job_manager.respond_team_message(
            session_id,
            channel.channel_id,
            request.message_id,
            response="yes",
            decision="approved",
            allow_approval=False,
        )

    resolved, _ = job_manager.respond_team_message(
        session_id,
        channel.channel_id,
        request.message_id,
        response="approved with constraints",
        decision="approved",
        allow_approval=True,
    )
    assert resolved.status == "approved"


@pytest.mark.parametrize(
    ("kwargs", "message_fragment"),
    [
        (
            {
                "source_agent": "utility",
                "target_agent": "utility",
                "allowed_target_agents": ["utility"],
            },
            "cannot dispatch peer messages to themselves",
        ),
        (
            {
                "source_agent": "general",
                "target_agent": "utility",
                "allowed_target_agents": ["data_analyst"],
            },
            "not allowed",
        ),
        (
            {
                "source_agent": "general",
                "target_agent": "utility",
                "allowed_target_agents": ["utility"],
                "source_delegation_depth": 3,
            },
            "depth exceeded",
        ),
    ],
)
def test_enqueue_agent_message_rejects_invalid_peer_dispatches(
    tmp_path: Path,
    kwargs: dict[str, object],
    message_fragment: str,
) -> None:
    paths = _paths(tmp_path)
    transcript_store = RuntimeTranscriptStore(paths)
    job_manager = RuntimeJobManager(transcript_store)
    session_id = "tenant:user:conversation"

    with pytest.raises(ValueError, match=message_fragment):
        job_manager.enqueue_agent_message(
            session_id=session_id,
            content="follow-up",
            create_job=lambda: job_manager.create_job(
                agent_name="utility",
                prompt="follow-up",
                session_id=session_id,
                description="unused",
            ),
            **kwargs,
        )
