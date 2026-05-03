"""Tests for the OpenAI-compatible FastAPI gateway."""
from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone
import json
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage

from agentic_chatbot_next.api import main as api_main
from agentic_chatbot_next.authz import AccessSnapshot, normalize_user_email
from agentic_chatbot_next.contracts.jobs import TeamMailboxChannel, TeamMailboxMessage, WorkerMailboxMessage
from agentic_chatbot_next.contracts.messages import RuntimeMessage, SessionState
from agentic_chatbot_next.persistence.postgres.skills import SkillChunkMatch, SkillPackRecord, SkillTelemetryEventRecord
from agentic_chatbot_next.observability.events import RuntimeEvent
from agentic_chatbot_next.rag.ingest import KBCoverageStatus
from agentic_chatbot_next.runtime.kernel import RuntimeKernel
from agentic_chatbot_next.runtime.context import filesystem_key


class _FakeTranscriptStore:
    def __init__(self):
        self.states: dict[str, SessionState] = {}

    def load_session_state(self, session_id: str):
        return self.states.get(session_id)

    def persist_session_state(self, session: SessionState) -> None:
        self.states[session.session_id] = session


class _FakeJobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, object] = {}
        self.mailboxes: dict[str, list[WorkerMailboxMessage]] = {}
        self.team_channels: dict[str, list[TeamMailboxChannel]] = {}
        self.team_messages: dict[tuple[str, str], list[TeamMailboxMessage]] = {}
        self.continued: list[str] = []

    def get_job(self, job_id: str):
        return self.jobs.get(job_id)

    def mailbox_summary(self, job_id: str):
        open_requests = [
            item
            for item in self.mailboxes.get(job_id, [])
            if item.status == "open" and item.message_type in {"question_request", "approval_request"}
        ]
        return {
            "pending_question_count": sum(1 for item in open_requests if item.message_type == "question_request"),
            "pending_approval_count": sum(1 for item in open_requests if item.message_type == "approval_request"),
            "latest_open_request": open_requests[-1].to_dict() if open_requests else {},
        }

    def list_mailbox_messages(self, job_id: str, *, status_filter: str = "", request_type: str = ""):
        rows = list(self.mailboxes.get(job_id, []))
        if status_filter:
            rows = [item for item in rows if item.status == status_filter]
        if request_type:
            rows = [item for item in rows if item.message_type == request_type]
        return rows

    def respond_to_request(self, job_id: str, request_id: str, *, response: str, responder: str = "operator", decision: str = "", allow_approval: bool = False, metadata=None):
        messages = self.mailboxes.get(job_id, [])
        request = next((item for item in messages if item.message_id == request_id), None)
        if request is None:
            return None
        if request.message_type == "approval_request" and not allow_approval:
            raise PermissionError("Approval requests require operator/API approval.")
        request.status = decision or "answered"
        request.resolved_by = responder
        response_message = WorkerMailboxMessage(
            job_id=job_id,
            content=response,
            sender=responder,
            message_type="approval_response" if request.message_type == "approval_request" else "question_response",
            direction="to_worker",
            status="queued",
            response_to=request.message_id,
            payload={"decision": decision},
        )
        messages.append(response_message)
        return request, response_message

    def continue_job(self, job_id: str, runner):
        self.continued.append(job_id)
        return self.jobs.get(job_id)

    def create_team_channel(self, *, session_id: str, name: str, purpose: str = "", created_by_job_id: str = "", member_agents=None, member_job_ids=None, metadata=None):
        channel = TeamMailboxChannel(
            session_id=session_id,
            name=name,
            purpose=purpose,
            created_by_job_id=created_by_job_id,
            member_agents=list(member_agents or []),
            member_job_ids=list(member_job_ids or []),
            metadata=dict(metadata or {}),
        )
        self.team_channels.setdefault(session_id, []).append(channel)
        return channel

    def list_team_channels(self, session_id: str, *, status_filter: str = ""):
        channels = list(self.team_channels.get(session_id, []))
        if status_filter:
            channels = [item for item in channels if item.status == status_filter]
        return channels

    def post_team_message(self, *, session_id: str, channel_id: str, content: str, source_agent: str = "", source_job_id: str = "", target_agents=None, target_job_ids=None, message_type: str = "message", subject: str = "", payload=None, metadata=None, response_to: str = "", thread_id: str = ""):
        if not any(item.channel_id == channel_id for item in self.team_channels.get(session_id, [])):
            raise ValueError("channel not found")
        message = TeamMailboxMessage(
            session_id=session_id,
            channel_id=channel_id,
            content=content,
            source_agent=source_agent,
            source_job_id=source_job_id,
            target_agents=list(target_agents or []),
            target_job_ids=list(target_job_ids or []),
            message_type=message_type,
            subject=subject,
            payload=dict(payload or {}),
            metadata=dict(metadata or {}),
            response_to=response_to,
            thread_id=thread_id,
            requires_response=message_type in {"question_request", "approval_request"},
        )
        self.team_messages.setdefault((session_id, channel_id), []).append(message)
        return message

    def list_team_messages(self, session_id: str, *, channel_id: str = "", message_type: str = "", status_filter: str = "open", limit: int = 20):
        rows = []
        channel_ids = [channel_id] if channel_id else [item.channel_id for item in self.team_channels.get(session_id, [])]
        for cid in channel_ids:
            rows.extend(self.team_messages.get((session_id, cid), []))
        if message_type:
            rows = [item for item in rows if item.message_type == message_type]
        if status_filter:
            rows = [item for item in rows if item.status == status_filter]
        return rows[-limit:]

    def respond_team_message(self, session_id: str, channel_id: str, message_id: str, *, response: str, responder_agent: str = "", responder_job_id: str = "", decision: str = "", allow_approval: bool = False, resolve: bool = True, metadata=None):
        messages = self.team_messages.get((session_id, channel_id), [])
        request = next((item for item in messages if item.message_id == message_id), None)
        if request is None:
            return None
        if request.message_type == "approval_request" and not allow_approval:
            raise PermissionError("Approval requests require operator/API approval.")
        request.status = decision or "answered"
        request.resolved_by = responder_job_id or responder_agent
        response_message = TeamMailboxMessage(
            session_id=session_id,
            channel_id=channel_id,
            content=response,
            source_agent=responder_agent,
            source_job_id=responder_job_id,
            target_agents=[request.source_agent] if request.source_agent else [],
            target_job_ids=[request.source_job_id] if request.source_job_id else [],
            message_type="approval_response" if request.message_type == "approval_request" else "question_response",
            response_to=request.message_id,
            payload={"decision": decision},
            metadata=dict(metadata or {}),
        )
        messages.append(response_message)
        return request, response_message

    def team_mailbox_summary(self, session_id: str, *, channel_id: str = ""):
        messages = self.list_team_messages(session_id, channel_id=channel_id, status_filter="open", limit=500)
        return {
            "active_channel_count": len(self.list_team_channels(session_id, status_filter="active")),
            "open_message_count": len(messages),
            "pending_question_count": sum(1 for item in messages if item.message_type == "question_request"),
            "pending_approval_count": sum(1 for item in messages if item.message_type == "approval_request"),
            "open_handoff_count": sum(1 for item in messages if item.message_type == "handoff"),
            "latest_open_message": messages[-1].to_dict() if messages else {},
        }


class DummyBot:
    def __init__(
        self,
        answer: str = "stubbed answer",
        *,
        assistant_artifacts: list[dict[str, object]] | None = None,
        assistant_metadata: dict[str, object] | None = None,
    ):
        self.answer = answer
        self.assistant_artifacts = list(assistant_artifacts or [])
        self.assistant_metadata = dict(assistant_metadata or {})
        self.calls: list[dict[str, object]] = []
        self.ctx = SimpleNamespace(stores=object())
        self.kernel = SimpleNamespace(
            transcript_store=_FakeTranscriptStore(),
            job_manager=_FakeJobManager(),
            _sync_pending_worker_request_for_session=lambda session_id: None,
            _job_runner=lambda job: "",
        )
        self.kb_status = KBCoverageStatus(
            tenant_id="local-dev",
            collection_id="default",
            configured_source_paths=(),
            missing_source_paths=(),
            indexed_source_paths=(),
            indexed_doc_count=0,
        )

    def process_turn(
        self,
        session,
        *,
        user_text,
        upload_paths=None,
        force_agent=False,
        requested_agent="",
        extra_callbacks=None,
        progress_sink=None,
        request_metadata=None,
    ):
        self.calls.append(
            {
                "session": session,
                "user_text": user_text,
                "upload_paths": list(upload_paths or []),
                "force_agent": force_agent,
                "requested_agent": requested_agent,
                "extra_callbacks": list(extra_callbacks or []),
                "progress_sink": progress_sink,
                "request_metadata": dict(request_metadata or {}),
            }
        )
        if self.assistant_artifacts or self.assistant_metadata:
            additional_kwargs = dict(self.assistant_metadata)
            if self.assistant_artifacts:
                additional_kwargs["artifacts"] = list(self.assistant_artifacts)
            session.messages.append(
                AIMessage(content=self.answer, additional_kwargs=additional_kwargs)
            )
        return self.answer

    def list_requested_agent_overrides(self):
        return ["coordinator", "data_analyst", "general", "rag_worker"]

    def get_kb_status(self, tenant_id=None, *, refresh=False, attempt_sync=False):
        del tenant_id, refresh, attempt_sync
        return self.kb_status


class _SourceDocStore:
    def __init__(self, records):
        self.records = {str(record.doc_id): record for record in records}

    def get_document(self, doc_id: str, tenant_id: str = "local-dev"):
        record = self.records.get(str(doc_id))
        if record is None:
            return None
        if str(getattr(record, "tenant_id", tenant_id) or tenant_id) != str(tenant_id):
            return None
        return record


class _StaticAuthorizationService:
    def __init__(self, grants_by_email: dict[str, dict[str, dict[str, object]]]):
        self.grants_by_email = {
            normalize_user_email(email): dict(resources or {})
            for email, resources in grants_by_email.items()
        }

    def _snapshot(self, *, tenant_id: str, user_id: str, user_email: str, session_upload_collection_id: str = "") -> AccessSnapshot:
        normalized_email = normalize_user_email(user_email)
        configured_resources = self.grants_by_email.get(normalized_email, {})
        resources = {
            resource_type: {
                "use": list(dict(configured_resources.get(resource_type) or {}).get("use") or []),
                "manage": list(dict(configured_resources.get(resource_type) or {}).get("manage") or []),
                "use_all": bool(dict(configured_resources.get(resource_type) or {}).get("use_all", False)),
                "manage_all": bool(dict(configured_resources.get(resource_type) or {}).get("manage_all", False)),
            }
            for resource_type in ("collection", "graph", "tool", "skill_family")
        }
        return AccessSnapshot(
            tenant_id=tenant_id,
            user_id=user_id,
            user_email=normalized_email,
            auth_provider="email" if normalized_email else "",
            principal_id=f"principal:{normalized_email}" if normalized_email else "",
            role_ids=("role:test",) if configured_resources else (),
            session_upload_collection_id=session_upload_collection_id,
            authz_enabled=True,
            resources=resources,
        )

    def resolve_access_snapshot(self, *, tenant_id: str, user_id: str, user_email: str, session_upload_collection_id: str = "", display_name: str = "") -> AccessSnapshot:
        del display_name
        return self._snapshot(
            tenant_id=tenant_id,
            user_id=user_id,
            user_email=user_email,
            session_upload_collection_id=session_upload_collection_id,
        )

    def apply_access_snapshot(self, session_or_state, *, tenant_id: str, user_id: str, user_email: str, session_upload_collection_id: str = "", display_name: str = "") -> AccessSnapshot:
        del display_name
        snapshot = self._snapshot(
            tenant_id=tenant_id,
            user_id=user_id,
            user_email=user_email,
            session_upload_collection_id=session_upload_collection_id,
        )
        summary = snapshot.to_summary()
        session_or_state.metadata = {
            **dict(getattr(session_or_state, "metadata", {}) or {}),
            "user_email": snapshot.user_email,
            "auth_provider": snapshot.auth_provider,
            "principal_id": snapshot.principal_id,
            "role_ids": list(snapshot.role_ids),
            "access_summary": summary,
        }
        if hasattr(session_or_state, "user_email"):
            session_or_state.user_email = snapshot.user_email
        if hasattr(session_or_state, "auth_provider"):
            session_or_state.auth_provider = snapshot.auth_provider
        if hasattr(session_or_state, "principal_id"):
            session_or_state.principal_id = snapshot.principal_id
        if hasattr(session_or_state, "access_summary"):
            session_or_state.access_summary = summary
        return snapshot


class _InMemorySkillStore:
    def __init__(self) -> None:
        self.records: dict[str, SkillPackRecord] = {}
        self.chunks: dict[str, list[dict[str, object]]] = {}
        self.telemetry_events: list[dict[str, object]] = []

    def upsert_skill_pack(self, record: SkillPackRecord, chunks: list[str]) -> None:
        self.records[record.skill_id] = record
        self.chunks[record.skill_id] = [
            {"skill_chunk_id": f"{record.skill_id}#chunk{i:04d}", "skill_id": record.skill_id, "chunk_index": i, "content": content}
            for i, content in enumerate(chunks)
        ]

    def list_skill_packs(
        self,
        *,
        tenant_id="local-dev",
        agent_scope="",
        enabled_only=False,
        owner_user_id="",
        visibility="",
        status="",
        accessible_skill_family_ids=None,
    ):
        rows = []
        allowed_families = set(accessible_skill_family_ids or [])
        for record in self.records.values():
            if record.tenant_id != tenant_id:
                continue
            if agent_scope and record.agent_scope != agent_scope:
                continue
            if enabled_only and not record.enabled:
                continue
            if visibility and record.visibility != visibility:
                continue
            if status and record.status != status:
                continue
            if record.visibility == "private" and owner_user_id and record.owner_user_id != owner_user_id:
                continue
            family_id = record.version_parent or record.skill_id
            if accessible_skill_family_ids is not None and family_id not in allowed_families:
                continue
            rows.append(record)
        return sorted(rows, key=lambda item: item.skill_id)

    def get_skill_pack(self, skill_id, *, tenant_id="local-dev", owner_user_id="", accessible_skill_family_ids=None):
        record = self.records.get(skill_id)
        if record is None or record.tenant_id != tenant_id:
            return None
        if record.visibility == "private" and owner_user_id and record.owner_user_id != owner_user_id:
            return None
        family_id = record.version_parent or record.skill_id
        if accessible_skill_family_ids is not None and family_id not in set(accessible_skill_family_ids):
            return None
        return record

    def get_skill_chunks(self, skill_id, *, tenant_id="local-dev"):
        record = self.records.get(skill_id)
        if record is None or record.tenant_id != tenant_id:
            return []
        return list(self.chunks.get(skill_id, []))

    def set_skill_status(self, skill_id, *, tenant_id="local-dev", status, enabled=None):
        record = self.records[skill_id]
        if record.tenant_id != tenant_id:
            return
        record.status = status
        if enabled is not None:
            record.enabled = enabled

    def vector_search(
        self,
        query,
        *,
        tenant_id,
        top_k,
        agent_scope,
        tool_tags=None,
        task_tags=None,
        enabled_only=True,
        owner_user_id="",
        accessible_skill_family_ids=None,
    ):
        del tool_tags, task_tags
        matches: list[SkillChunkMatch] = []
        allowed_families = set(accessible_skill_family_ids or [])
        for record in self.records.values():
            if record.tenant_id != tenant_id:
                continue
            if agent_scope and record.agent_scope != agent_scope:
                continue
            if enabled_only and (not record.enabled or record.status != "active"):
                continue
            if record.visibility == "private" and owner_user_id and record.owner_user_id != owner_user_id:
                continue
            family_id = record.version_parent or record.skill_id
            if accessible_skill_family_ids is not None and family_id not in allowed_families:
                continue
            chunks = self.chunks.get(record.skill_id, [])
            for chunk in chunks[:1]:
                content = str(chunk["content"])
                if query.lower() not in content.lower() and query.lower() not in record.name.lower():
                    continue
                matches.append(
                    SkillChunkMatch(
                        skill_id=record.skill_id,
                        name=record.name,
                        agent_scope=record.agent_scope,
                        content=content,
                        chunk_index=int(chunk["chunk_index"]),
                        score=0.9,
                        retrieval_profile=record.retrieval_profile,
                        controller_hints=dict(record.controller_hints),
                        coverage_goal=record.coverage_goal,
                        result_mode=record.result_mode,
                        owner_user_id=record.owner_user_id,
                        visibility=record.visibility,
                        status=record.status,
                        version_parent=record.version_parent or record.skill_id,
                    )
                )
        return matches[:top_k]

    def append_skill_telemetry_event(self, event):
        payload = event.to_dict() if hasattr(event, "to_dict") else dict(getattr(event, "__dict__", {}) or {})
        self.telemetry_events.append(payload)

    def list_skill_telemetry_events(self, *, tenant_id="local-dev", skill_family_id="", skill_id="", session_id="", limit=200):
        rows = [
            event
            for event in self.telemetry_events
            if event.get("tenant_id") == tenant_id
            and (not skill_family_id or event.get("skill_family_id") == skill_family_id)
            and (not skill_id or event.get("skill_id") == skill_id)
            and (not session_id or event.get("session_id") == session_id)
        ]
        return list(reversed(rows))[:limit]


class _FakeSkillBuilderChat:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = []

    def invoke(self, messages, config=None):
        del config
        self.messages.append(messages)
        return AIMessage(content=self.content)


def _make_settings(tmp_path: Path):
    workspace_dir = tmp_path / "workspaces"
    workspace_dir.mkdir(exist_ok=True)
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir(exist_ok=True)
    return SimpleNamespace(
        gateway_model_id="enterprise-agent",
        gateway_shared_bearer_token="",
        download_url_secret="",
        download_url_ttl_seconds=900,
        connector_secret_api_key="connector-secret",
        connector_publishable_api_key="",
        connector_allowed_origins=(),
        connector_publishable_rate_limit_per_minute=60,
        authz_enabled=False,
        control_panel_enabled=True,
        control_panel_admin_token="admin-secret",
        default_tenant_id="local-dev",
        default_user_id="local-cli",
        default_conversation_id="local-session",
        default_collection_id="default",
        workspace_dir=workspace_dir,
        uploads_dir=uploads_dir,
        team_mailbox_enabled=False,
        team_mailbox_max_channels_per_session=8,
        team_mailbox_max_open_messages_per_channel=50,
        team_mailbox_claim_limit=8,
        agent_chat_model_overrides={"general": "nemotron-cascade-2:30b"},
        agent_judge_model_overrides={"general": "nemotron-cascade-2:30b"},
    )


def _make_client(tmp_path: Path):
    settings = _make_settings(tmp_path)
    bot = DummyBot()
    runtime = api_main.Runtime(settings=settings, bot=bot)
    api_main.app.dependency_overrides[api_main.get_runtime_readiness] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_runtime_or_503] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_upload_runtime_or_503] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    return client, bot, settings


def _make_skill_client(tmp_path: Path):
    settings = _make_settings(tmp_path)
    bot = DummyBot()
    bot.ctx = SimpleNamespace(stores=SimpleNamespace(skill_store=_InMemorySkillStore()))
    runtime = api_main.Runtime(settings=settings, bot=bot)
    api_main.app.dependency_overrides[api_main.get_runtime_or_503] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    return client, bot, settings


def _clear_overrides() -> None:
    api_main.app.dependency_overrides.clear()


def _connector_stream_parts(response_text: str):
    parts = []
    for block in response_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        data_lines = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
        if not data_lines:
            continue
        payload = "\n".join(data_lines)
        if payload == "[DONE]":
            parts.append(payload)
            continue
        parts.append(json.loads(payload))
    return parts


def _sse_events(response_text: str):
    events = []
    for block in response_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_name = "message"
        data_lines = []
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line[7:]
            elif line.startswith("data: "):
                data_lines.append(line[6:])
        if not data_lines:
            continue
        payload = "\n".join(data_lines)
        if payload == "[DONE]":
            events.append({"event": event_name, "data": payload})
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            parsed = payload
        events.append({"event": event_name, "data": parsed})
    return events


def test_runtime_dependency_returns_structured_registry_error(monkeypatch):
    def fail_runtime():
        raise ValueError(
            "Invalid next-runtime agent configuration:\n"
            "- agent 'coordinator' references unknown tool 'list_worker_requests'"
        )

    monkeypatch.setattr(api_main, "get_runtime", fail_runtime)

    with pytest.raises(api_main.HTTPException) as excinfo:
        api_main.get_runtime_or_503()

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail["error_code"] == "runtime_registry_invalid"
    assert excinfo.value.detail["missing_tools"] == [
        {"agent": "coordinator", "tool": "list_worker_requests"}
    ]


@pytest.mark.asyncio
async def test_list_models_returns_gateway_model(tmp_path):
    client, _, _ = _make_client(tmp_path)
    try:
        response = await client.get("/v1/models")
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == "enterprise-agent"


@pytest.mark.asyncio
async def test_list_models_requires_bearer_token_when_configured(tmp_path):
    settings = _make_settings(tmp_path)
    settings.gateway_shared_bearer_token = "shared-secret"
    bot = DummyBot()
    runtime = api_main.Runtime(settings=settings, bot=bot)
    api_main.app.dependency_overrides[api_main.get_runtime_or_503] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    try:
        unauthorized = await client.get("/v1/models")
        authorized = await client.get(
            "/v1/models",
            headers={"Authorization": "Bearer shared-secret"},
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


@pytest.mark.asyncio
async def test_list_agents_returns_registry_metadata(tmp_path):
    client, bot, _ = _make_client(tmp_path)
    bot.kernel.registry = SimpleNamespace(
        list=lambda: [
            SimpleNamespace(
                name="rag_worker",
                mode="rag",
                description="Grounded document worker",
                metadata={"display_name": "RAG Worker"},
            ),
            SimpleNamespace(
                name="finalizer",
                mode="finalizer",
                description="Final synthesis agent",
                metadata={},
            ),
        ]
    )
    try:
        response = await client.get("/v1/agents")
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"] == [
        {
            "id": "finalizer",
            "display_name": "finalizer",
            "mode": "finalizer",
            "description": "Final synthesis agent",
        },
        {
            "id": "rag_worker",
            "display_name": "RAG Worker",
            "mode": "rag",
            "description": "Grounded document worker",
        },
    ]


@pytest.mark.asyncio
async def test_chat_completions_passes_collection_id_into_session_metadata(tmp_path):
    client, bot, _ = _make_client(tmp_path)
    try:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Collection-ID": "defense-rag-test"},
            json={
                "model": "enterprise-agent",
                "messages": [{"role": "user", "content": "What is the approved CDR date?"}],
                "metadata": {"collection_id": "defense-rag-test"},
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert bot.calls
    session = bot.calls[0]["session"]
    assert session.metadata["collection_id"] == "defense-rag-test"
    assert session.metadata["upload_collection_id"] == "defense-rag-test"
    assert session.metadata["kb_collection_id"] == "defense-rag-test"
    assert session.metadata["available_kb_collection_ids"] == ["defense-rag-test"]
    assert session.metadata["kb_collection_confirmed"] is True


@pytest.mark.asyncio
async def test_chat_completions_preserves_openwebui_upload_and_kb_collections(tmp_path):
    client, bot, _ = _make_client(tmp_path)
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "enterprise-agent",
                "messages": [{"role": "user", "content": "Compare my upload to the KB."}],
                "metadata": {
                    "collection_id": "owui-chat-42",
                    "upload_collection_id": "owui-chat-42",
                    "kb_collection_id": "default",
                    "kb_collection_confirmed": False,
                },
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    session = bot.calls[0]["session"]
    assert session.metadata["collection_id"] == "owui-chat-42"
    assert session.metadata["upload_collection_id"] == "owui-chat-42"
    assert session.metadata["kb_collection_id"] == "default"
    assert session.metadata["kb_collection_confirmed"] is False
    assert session.metadata["available_kb_collection_ids"] == ["default"]


@pytest.mark.asyncio
async def test_chat_completions_accepts_openwebui_header_aliases(tmp_path):
    client, bot, _ = _make_client(tmp_path)
    try:
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "X-OpenWebUI-Chat-Id": "owui-chat-42",
                "X-OpenWebUI-Message-Id": "owui-msg-7",
                "X-OpenWebUI-User-Id": "owui-user-5",
                "X-OpenWebUI-User-Email": "Analyst@Example.com",
            },
            json={
                "model": "enterprise-agent",
                "messages": [{"role": "user", "content": "Hello from Open WebUI"}],
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert bot.calls
    session = bot.calls[0]["session"]
    assert session.conversation_id == "owui-chat-42"
    assert session.request_id == "owui-msg-7"
    assert session.user_id == "owui-user-5"
    assert session.user_email == "analyst@example.com"
    assert session.metadata["user_email"] == "analyst@example.com"


@pytest.mark.asyncio
async def test_chat_completions_infers_openwebui_helper_task_from_headers_and_text(tmp_path):
    client, bot, _ = _make_client(tmp_path)
    try:
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "X-OpenWebUI-Chat-Id": "owui-chat-99",
                "X-OpenWebUI-Message-Id": "owui-msg-8",
                "X-OpenWebUI-User-Id": "owui-user-8",
            },
            json={
                "model": "enterprise-agent",
                "messages": [{"role": "user", "content": "Generate 1-3 broad tags for this chat."}],
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert bot.calls
    metadata = bot.calls[0]["request_metadata"]
    assert metadata["openwebui_client"] is True
    assert metadata["openwebui_helper_task_type"] == "tags"


@pytest.mark.asyncio
async def test_chat_completions_infers_openwebui_search_query_helper_task_from_text(tmp_path):
    client, bot, _ = _make_client(tmp_path)
    try:
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "X-OpenWebUI-Chat-Id": "owui-chat-100",
                "X-OpenWebUI-Message-Id": "owui-msg-9",
                "X-OpenWebUI-User-Id": "owui-user-9",
            },
            json={
                "model": "enterprise-agent",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "### Task:\n"
                            "Analyze the chat history to determine the necessity of generating search queries.\n"
                            "Respond EXCLUSIVELY with a JSON object in the form {\"queries\": [\"query1\"]}."
                        ),
                    }
                ],
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert bot.calls
    metadata = bot.calls[0]["request_metadata"]
    assert metadata["openwebui_client"] is True
    assert metadata["openwebui_helper_task_type"] == "search_queries"


@pytest.mark.asyncio
async def test_list_graphs_rejects_forbidden_collection_when_authz_enabled(tmp_path):
    client, bot, settings = _make_client(tmp_path)
    settings.authz_enabled = True
    bot.ctx = SimpleNamespace(
        stores=SimpleNamespace(
            authorization_service=_StaticAuthorizationService(
                {"analyst@example.com": {"collection": {"use": ["default"]}}}
            )
        )
    )
    try:
        response = await client.get(
            "/v1/graphs",
            headers={"X-User-Email": "analyst@example.com"},
            params={"collection_id": "finance"},
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 403
    assert "finance" in response.json()["detail"]


@pytest.mark.asyncio
async def test_graph_query_rejects_forbidden_graph_when_authz_enabled(tmp_path):
    client, bot, settings = _make_client(tmp_path)
    settings.authz_enabled = True
    bot.ctx = SimpleNamespace(
        stores=SimpleNamespace(
            authorization_service=_StaticAuthorizationService(
                {"analyst@example.com": {"collection": {"use": ["default"]}}}
            )
        )
    )
    try:
        response = await client.post(
            "/v1/graphs/query",
            headers={"X-User-Email": "analyst@example.com"},
            json={
                "graph_id": "finance-graph",
                "query": "show risks",
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 403
    assert "finance-graph" in response.json()["detail"]


@pytest.mark.asyncio
async def test_skill_api_supports_create_update_activate_and_preview(tmp_path):
    client, bot, _ = _make_skill_client(tmp_path)
    try:
        create = await client.post(
            "/v1/skills",
            headers={"X-Tenant-ID": "tenant-a", "X-User-ID": "user-a", "X-Admin-Token": "admin-secret"},
            json={
                "skill_id": "process-flow-skill",
                "body_markdown": (
                    "# Process Flow Search\n"
                    "agent_scope: rag\n"
                    "retrieval_profile: process_flow_identification\n"
                    "coverage_goal: corpus_wide\n"
                    "result_mode: inventory\n\n"
                    "Find workflow and approval flow documents.\n"
                ),
            },
        )
        assert create.status_code == 200
        created = create.json()["data"]
        assert created["skill_id"] == "process-flow-skill"
        assert created["visibility"] == "private"
        assert created["status"] == "draft"
        assert "dependency_validation" in created
        assert "skill_health" in created

        activate = await client.post(
            "/v1/skills/process-flow-skill/activate",
            headers={"X-Tenant-ID": "tenant-a", "X-User-ID": "user-a", "X-Admin-Token": "admin-secret"},
        )
        assert activate.status_code == 200
        assert activate.json()["data"]["status"] == "active"
        assert activate.json()["data"]["dependency_validation"]["is_valid"] is True

        update = await client.put(
            "/v1/skills/process-flow-skill",
            headers={"X-Tenant-ID": "tenant-a", "X-User-ID": "user-a", "X-Admin-Token": "admin-secret"},
            json={
                "body_markdown": (
                    "# Process Flow Search\n"
                    "agent_scope: rag\n"
                    "retrieval_profile: process_flow_identification\n"
                    "coverage_goal: corpus_wide\n"
                    "result_mode: inventory\n\n"
                    "Find workflow, approval flow, and escalation path documents.\n"
                ),
                "status": "active",
            },
        )
        assert update.status_code == 200
        updated = update.json()["data"]
        assert updated["skill_id"] != "process-flow-skill"
        assert updated["version_parent"] == "process-flow-skill"

        preview = await client.post(
            "/v1/skills/preview",
            headers={"X-Tenant-ID": "tenant-a", "X-User-ID": "user-a", "X-Admin-Token": "admin-secret"},
            json={"query": "workflow", "agent_scope": "rag", "top_k": 2},
        )
        assert preview.status_code == 200
        matches = preview.json()["matches"]
        assert matches
        assert matches[0]["agent_scope"] == "rag"
        assert matches[0]["version_parent"] == "process-flow-skill"
        assert "dependency_validation" in matches[0]
        assert "skill_health" in matches[0]

        listing = await client.get(
            "/v1/skills",
            headers={"X-Tenant-ID": "tenant-a", "X-User-ID": "user-a", "X-Admin-Token": "admin-secret"},
        )
        assert listing.status_code == 200
    finally:
        await client.aclose()
        _clear_overrides()

    assert len(listing.json()["data"]) == 2
    assert "dependency_validation" in listing.json()["data"][0]
    assert "skill_health" in listing.json()["data"][0]


@pytest.mark.asyncio
async def test_skill_api_build_draft_uses_llm_without_persisting(tmp_path):
    client, bot, _ = _make_skill_client(tmp_path)
    store = bot.ctx.stores.skill_store
    chat = _FakeSkillBuilderChat(
        json.dumps(
            {
                "name": "Routing Triage Skill",
                "description": "Guide routing issue triage.",
                "when_to_apply": "Use when an operator is investigating routing outcomes.",
                "tool_tags": ["search_skills"],
                "task_tags": ["workflow", "routing"],
                "workflow": [
                    "Review the user task and current route decision.",
                    "Compare the decision with available routing evidence.",
                    "Summarize the recommended route and confidence.",
                ],
                "examples": ["User asks why a routing decision selected the basic agent."],
                "warnings": ["Do not invent missing routing evidence."],
            }
        )
    )
    bot.kernel = SimpleNamespace(resolve_base_providers=lambda: SimpleNamespace(chat=chat))
    try:
        response = await client.post(
            "/v1/skills/build-draft",
            headers={"X-Tenant-ID": "tenant-a", "X-User-ID": "user-a", "X-Admin-Token": "admin-secret"},
            json={
                "context": "Operators need a repeatable workflow for inspecting routing decisions.",
                "examples": "- User asks why a routing decision selected the basic agent.",
                "name": "Routing Helper",
                "agent_scope": "general",
                "target_agent": "general",
                "tool_tags": ["search_skills"],
                "task_tags": ["workflow"],
                "description": "Draft routing helper.",
                "when_to_apply": "Use for routing review.",
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    payload = response.json()
    draft = payload["draft"]
    assert payload["object"] == "skill.build_draft"
    assert draft["name"] == "Routing Triage Skill"
    assert draft["agent_scope"] == "general"
    assert draft["tool_tags"] == ["search_skills"]
    assert "kind: retrievable" in draft["body_markdown"]
    assert "## Examples" in draft["body_markdown"]
    assert "routing decision selected the basic agent" in draft["body_markdown"]
    assert store.records == {}
    assert chat.messages


@pytest.mark.asyncio
async def test_skill_api_build_draft_reports_invalid_llm_json(tmp_path):
    client, bot, _ = _make_skill_client(tmp_path)
    store = bot.ctx.stores.skill_store
    bot.kernel = SimpleNamespace(resolve_base_providers=lambda: SimpleNamespace(chat=_FakeSkillBuilderChat("not json")))
    try:
        response = await client.post(
            "/v1/skills/build-draft",
            headers={"X-Tenant-ID": "tenant-a", "X-User-ID": "user-a", "X-Admin-Token": "admin-secret"},
            json={
                "context": "Build a skill for routing review.",
                "agent_scope": "general",
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 502
    assert "invalid JSON" in response.json()["detail"]
    assert store.records == {}


@pytest.mark.asyncio
async def test_skill_api_supports_executable_skill_preview(tmp_path):
    client, _, settings = _make_skill_client(tmp_path)
    settings.executable_skills_enabled = True
    try:
        create = await client.post(
            "/v1/skills",
            headers={"X-Tenant-ID": "tenant-a", "X-User-ID": "user-a", "X-Admin-Token": "admin-secret"},
            json={
                "skill_id": "review-skill",
                "kind": "executable",
                "execution_config": {
                    "context": "fork",
                    "agent": "utility",
                    "allowed_tools": ["calculator"],
                    "model": "gpt-test",
                    "effort": "high",
                },
                "body_markdown": (
                    "# Review Skill\n"
                    "agent_scope: general\n\n"
                    "Review {{input}} with {{ARGUMENTS_JSON}}.\n"
                ),
            },
        )
        assert create.status_code == 200
        created = create.json()["data"]
        assert created["kind"] == "executable"
        assert created["execution_config"]["context"] == "fork"
        assert created["execution_config"]["allowed_tools"] == ["calculator"]

        preview = await client.post(
            "/v1/skills/review-skill/preview-execution",
            headers={"X-Tenant-ID": "tenant-a", "X-User-ID": "user-a", "X-Admin-Token": "admin-secret"},
            json={"input": "the uploaded contract", "arguments": {"focus": "termination"}},
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert preview.status_code == 200
    payload = preview.json()["data"]
    assert payload["skill_id"] == "review-skill"
    assert payload["context"] == "fork"
    assert "the uploaded contract" in payload["rendered_prompt"]
    assert "termination" in payload["rendered_prompt"]


@pytest.mark.asyncio
async def test_skill_api_list_accepts_persisted_telemetry_records(tmp_path):
    client, bot, _ = _make_skill_client(tmp_path)
    store = bot.ctx.stores.skill_store
    store.upsert_skill_pack(
        SkillPackRecord(
            skill_id="telemetry-skill",
            name="Telemetry Skill",
            agent_scope="general",
            checksum="checksum",
            tenant_id="tenant-a",
            owner_user_id="user-a",
            visibility="private",
            status="active",
            version_parent="telemetry-skill",
            body_markdown="# Telemetry Skill\nagent_scope: general\n",
        ),
        ["Find workflow documents."],
    )
    store.list_skill_telemetry_events = lambda **kwargs: [
        SkillTelemetryEventRecord(
            event_id="evt-1",
            tenant_id="tenant-a",
            skill_id="telemetry-skill",
            skill_family_id="telemetry-skill",
            query="workflow",
            answer_quality="pass",
            agent_name="general",
            session_id="session-1",
            created_at="2026-04-19T14:00:00Z",
        )
    ]
    try:
        response = await client.get(
            "/v1/skills",
            headers={"X-Tenant-ID": "tenant-a", "X-User-ID": "user-a", "X-Admin-Token": "admin-secret"},
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    payload = response.json()["data"]
    assert len(payload) == 1
    assert payload[0]["skill_id"] == "telemetry-skill"
    assert payload[0]["skill_health"]["skill_family_id"] == "telemetry-skill"
    assert payload[0]["skill_health"]["scored_uses"] == 1


@pytest.mark.asyncio
async def test_skill_mutation_requires_admin_token_or_manage_permission(tmp_path):
    client, bot, settings = _make_skill_client(tmp_path)
    settings.authz_enabled = True
    bot.ctx = SimpleNamespace(
        stores=SimpleNamespace(
            skill_store=bot.ctx.stores.skill_store,
            authorization_service=_StaticAuthorizationService(
                {"viewer@example.com": {"skill_family": {"use": []}}}
            ),
        )
    )
    try:
        denied = await client.post(
            "/v1/skills",
            headers={"X-User-Email": "viewer@example.com"},
            json={"body_markdown": "# Viewer Skill\nagent_scope: general\n"},
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert denied.status_code == 403

    client, bot, settings = _make_skill_client(tmp_path)
    settings.authz_enabled = True
    bot.ctx = SimpleNamespace(
        stores=SimpleNamespace(
            skill_store=bot.ctx.stores.skill_store,
            authorization_service=_StaticAuthorizationService(
                {
                    "editor@example.com": {
                        "skill_family": {
                            "use_all": True,
                            "manage_all": True,
                        }
                    }
                }
            ),
        )
    )
    try:
        allowed = await client.post(
            "/v1/skills",
            headers={"X-User-Email": "editor@example.com"},
            json={"body_markdown": "# Editor Skill\nagent_scope: general\n"},
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert allowed.status_code == 200
    assert allowed.json()["data"]["name"] == "Editor Skill"


@pytest.mark.asyncio
async def test_skill_activation_blocks_missing_dependencies(tmp_path):
    client, _, _ = _make_skill_client(tmp_path)
    try:
        create = await client.post(
            "/v1/skills",
            headers={"X-Tenant-ID": "tenant-a", "X-User-ID": "user-a", "X-Admin-Token": "admin-secret"},
            json={
                "skill_id": "skill-b",
                "body_markdown": (
                    "# Dependent Skill\n"
                    "agent_scope: general\n"
                    'controller_hints: {"depends_on_skills": ["skill-a"]}\n\n'
                    "Use the parent workflow first.\n"
                ),
            },
        )
        assert create.status_code == 200

        activate = await client.post(
            "/v1/skills/skill-b/activate",
            headers={"X-Tenant-ID": "tenant-a", "X-User-ID": "user-a", "X-Admin-Token": "admin-secret"},
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert activate.status_code == 409
    payload = activate.json()["detail"]
    assert payload["action"] == "activate"
    assert payload["dependency_validation"]["missing_dependencies"] == ["skill-a"]
    assert payload["dependency_validation"]["dependency_state"] == "warning"


@pytest.mark.asyncio
async def test_skill_deactivation_blocks_active_dependents(tmp_path):
    client, _, _ = _make_skill_client(tmp_path)
    try:
        parent = await client.post(
            "/v1/skills",
            headers={"X-Tenant-ID": "tenant-a", "X-User-ID": "user-a", "X-Admin-Token": "admin-secret"},
            json={
                "skill_id": "skill-a",
                "body_markdown": "# Parent Skill\nagent_scope: general\n\nParent workflow.\n",
            },
        )
        assert parent.status_code == 200
        activate_parent = await client.post(
            "/v1/skills/skill-a/activate",
            headers={"X-Tenant-ID": "tenant-a", "X-User-ID": "user-a", "X-Admin-Token": "admin-secret"},
        )
        assert activate_parent.status_code == 200
        child = await client.post(
            "/v1/skills",
            headers={"X-Tenant-ID": "tenant-a", "X-User-ID": "user-a", "X-Admin-Token": "admin-secret"},
            json={
                "skill_id": "skill-b",
                "body_markdown": (
                    "# Child Skill\n"
                    "agent_scope: general\n"
                    'controller_hints: {"depends_on_skills": ["skill-a"]}\n\n'
                    "Child workflow.\n"
                ),
            },
        )
        assert child.status_code == 200
        activate_child = await client.post(
            "/v1/skills/skill-b/activate",
            headers={"X-Tenant-ID": "tenant-a", "X-User-ID": "user-a", "X-Admin-Token": "admin-secret"},
        )
        assert activate_child.status_code == 200

        deactivate_parent = await client.post(
            "/v1/skills/skill-a/deactivate",
            headers={"X-Tenant-ID": "tenant-a", "X-User-ID": "user-a", "X-Admin-Token": "admin-secret"},
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert deactivate_parent.status_code == 409
    payload = deactivate_parent.json()["detail"]
    blocked_dependents = payload["dependency_validation"]["blocked_dependents"]
    assert payload["action"] == "deactivate"
    assert blocked_dependents
    assert blocked_dependents[0]["skill_family_id"] == "skill-b"


@pytest.mark.asyncio
async def test_health_ready_returns_200_when_kb_is_ready(tmp_path):
    client, _, _ = _make_client(tmp_path)
    try:
        response = await client.get("/health/ready")
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["model"] == "enterprise-agent"
    assert payload["capability_status"]["memory"]["configured"] is True
    assert "analyst_sandbox" in payload["capability_status"]


@pytest.mark.asyncio
async def test_ingest_documents_expands_directories_and_passes_collection_id(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path)
    nested = tmp_path / "incoming" / "nested"
    nested.mkdir(parents=True)
    file_path = nested / "notes.txt"
    file_path.write_text("hello world", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_ingest_paths(settings, stores, paths, *, source_type, tenant_id, collection_id=None, **kwargs):
        del settings, stores
        captured["paths"] = [Path(item) for item in paths]
        captured["source_type"] = source_type
        captured["tenant_id"] = tenant_id
        captured["collection_id"] = collection_id
        return ["doc-1"]

    monkeypatch.setattr(api_main, "ingest_paths", fake_ingest_paths)
    try:
        response = await client.post(
            "/v1/ingest/documents",
            headers={"X-Collection-ID": "defense-rag-test"},
            json={
                "paths": [str(tmp_path / "incoming")],
                "source_type": "kb",
                "collection_id": "defense-rag-test",
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    payload = response.json()
    assert payload["collection_id"] == "defense-rag-test"
    assert captured["collection_id"] == "defense-rag-test"
    assert captured["paths"] == [file_path.resolve()]


@pytest.mark.asyncio
async def test_ingest_documents_preview_returns_metadata_without_ingesting(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path)
    source = tmp_path / "preview.md"
    source.write_text("# Preview\n\nREQ-001 The gateway shall authenticate users.", encoding="utf-8")

    def fail_ingest_paths(*args, **kwargs):
        raise AssertionError("index preview should not call ingest_paths")

    monkeypatch.setattr(api_main, "ingest_paths", fail_ingest_paths)
    try:
        response = await client.post(
            "/v1/ingest/documents",
            json={
                "paths": [str(source)],
                "collection_id": "preview-collection",
                "metadata_profile": "auto",
                "index_preview": True,
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "ingest.preview"
    assert payload["preview"] is True
    assert payload["ingested_count"] == 0
    assert payload["files"][0]["outcome"] == "previewed"
    assert payload["metadata_summary"]["document_count"] == 1


@pytest.mark.asyncio
async def test_health_ready_returns_503_when_kb_coverage_is_missing(tmp_path):
    client, bot, _ = _make_client(tmp_path)
    bot.kb_status = KBCoverageStatus(
        tenant_id="local-dev",
        collection_id="default",
        configured_source_paths=("/tmp/docs/ARCHITECTURE.md",),
        missing_source_paths=("/tmp/docs/ARCHITECTURE.md",),
        indexed_source_paths=(),
        indexed_doc_count=0,
        sync_attempted=True,
    )
    try:
        response = await client.get("/health/ready")
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert payload["reason"] == "kb_coverage_missing"
    assert payload["collection_id"] == "default"
    assert payload["suggested_fix"] == "python run.py sync-kb --collection-id default"


@pytest.mark.asyncio
async def test_health_ready_returns_structured_503_when_runtime_registry_is_invalid(tmp_path):
    settings = _make_settings(tmp_path)
    api_main.app.dependency_overrides[api_main.get_runtime_readiness] = lambda: {
        "status": "not_ready",
        "registry_valid": False,
        "error_code": "runtime_registry_invalid",
        "message": "Invalid next-runtime agent configuration.",
        "detail": "agent 'coordinator' references unknown tool 'create_team_channel'",
        "missing_tools": [{"agent": "coordinator", "tool": "create_team_channel"}],
        "missing_workers": [],
        "affected_agents": ["coordinator"],
        "remediation": "Rebuild/recreate the app image.",
    }
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    try:
        response = await client.get("/health/ready")
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert payload["registry_valid"] is False
    assert payload["error_code"] == "runtime_registry_invalid"
    assert payload["missing_tools"] == [{"agent": "coordinator", "tool": "create_team_channel"}]
    assert payload["affected_agents"] == ["coordinator"]
    assert payload["remediation"] == "Rebuild/recreate the app image."


@pytest.mark.asyncio
async def test_list_models_ignores_agent_specific_runtime_model_overrides(tmp_path):
    client, _, _ = _make_client(tmp_path)
    try:
        response = await client.get("/v1/models")
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "enterprise-agent"


@pytest.mark.asyncio
async def test_chat_completions_uses_client_history_and_conversation_scope(tmp_path):
    client, bot, _ = _make_client(tmp_path)
    try:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Conversation-ID": "chat-001"},
            json={
                "model": "enterprise-agent",
                "messages": [
                    {"role": "system", "content": "You are concise."},
                    {"role": "assistant", "content": "How can I help?"},
                    {"role": "user", "content": [{"type": "text", "text": "Summarize the auth doc."}]},
                ],
                "metadata": {"force_agent": True},
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "stubbed answer"

    call = bot.calls[0]
    session = call["session"]
    assert call["user_text"] == "Summarize the auth doc."
    assert call["upload_paths"] == []
    assert call["force_agent"] is True
    assert session.conversation_id == "chat-001"
    assert session.session_id == "local-dev:local-cli:chat-001"
    assert [msg.content for msg in session.messages] == [
        "You are concise.",
        "How can I help?",
    ]


@pytest.mark.asyncio
async def test_chat_completions_passes_requested_agent_override(tmp_path):
    client, bot, _ = _make_client(tmp_path)
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "enterprise-agent",
                "messages": [{"role": "user", "content": "Explain the release-note changes with citations."}],
                "metadata": {"force_agent": True, "requested_agent": "general"},
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert bot.calls[0]["requested_agent"] == "general"
    assert bot.calls[0]["force_agent"] is True


@pytest.mark.asyncio
async def test_chat_completions_passes_max_tokens_through_request_metadata(tmp_path):
    client, bot, _ = _make_client(tmp_path)
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "enterprise-agent",
                "messages": [{"role": "user", "content": "Explain the release-note changes with citations."}],
                "max_tokens": 3072,
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert bot.calls[0]["request_metadata"]["chat_max_output_tokens"] == 3072


@pytest.mark.asyncio
async def test_chat_completions_rejects_invalid_requested_agent_override(tmp_path):
    client, _, _ = _make_client(tmp_path)
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "enterprise-agent",
                "messages": [{"role": "user", "content": "Explain the release-note changes with citations."}],
                "metadata": {"requested_agent": "utility"},
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 400
    assert "requested_agent='utility'" in response.json()["detail"]
    assert "general" in response.json()["detail"]
    assert "rag_worker" in response.json()["detail"]


@pytest.mark.asyncio
async def test_chat_completions_reuses_same_session_scope_for_repeated_conversation_id(tmp_path):
    client, bot, _ = _make_client(tmp_path)
    try:
        for question in ("First turn", "Second turn"):
            response = await client.post(
                "/v1/chat/completions",
                headers={"X-Conversation-ID": "stable-scope"},
                json={
                    "model": "enterprise-agent",
                    "messages": [{"role": "user", "content": question}],
                },
            )
            assert response.status_code == 200
    finally:
        await client.aclose()
        _clear_overrides()

    session_ids = [call["session"].session_id for call in bot.calls]
    assert session_ids == [
        "local-dev:local-cli:stable-scope",
        "local-dev:local-cli:stable-scope",
    ]


@pytest.mark.asyncio
async def test_chat_completions_persists_active_doc_focus_across_turns(tmp_path, monkeypatch):
    settings = _make_settings(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    settings.runtime_dir = tmp_path / "runtime"
    settings.memory_dir = tmp_path / "memory"
    settings.skills_dir = repo_root / "data" / "skills"
    settings.agents_dir = repo_root / "data" / "agents"
    settings.runtime_events_enabled = False
    settings.enable_coordinator_mode = False
    settings.planner_max_tasks = 8
    settings.max_revision_rounds = 3
    settings.session_hydrate_window_messages = 40
    settings.session_transcript_page_size = 100

    class _KernelBackedBot:
        def __init__(self, kernel_settings):
            self.ctx = SimpleNamespace(stores=SimpleNamespace())
            self.kernel = RuntimeKernel(kernel_settings, providers=SimpleNamespace(), stores=SimpleNamespace())

        def process_turn(
            self,
            session,
            *,
            user_text,
            upload_paths=None,
            force_agent=False,
            requested_agent="",
            extra_callbacks=None,
            progress_sink=None,
            request_metadata=None,
        ):
            del upload_paths, force_agent, requested_agent, extra_callbacks, progress_sink, request_metadata
            return self.kernel.process_agent_turn(
                session,
                user_text=user_text,
                agent_name="coordinator",
            )

        def list_requested_agent_overrides(self):
            return ["coordinator", "data_analyst", "general", "rag_worker"]

        def get_kb_status(self, tenant_id=None, *, refresh=False, attempt_sync=False):
            del tenant_id, refresh, attempt_sync
            return KBCoverageStatus(
                tenant_id="local-dev",
                collection_id="default",
                configured_source_paths=(),
                missing_source_paths=(),
                indexed_source_paths=(),
                indexed_doc_count=0,
            )

    bot = _KernelBackedBot(settings)

    def fake_run_agent(agent, session_state, *, user_text, callbacks, task_payload=None, chat_max_output_tokens=None):
        del callbacks, task_payload, chat_max_output_tokens
        if agent.name != "coordinator":
            raise AssertionError(f"Unexpected agent {agent.name}")
        if not dict(session_state.metadata.get("active_doc_focus") or {}):
            doc_focus_result = {
                "collection_id": "default",
                "documents": [
                    {"doc_id": "KB_architecture", "title": "ARCHITECTURE.md"},
                    {"doc_id": "KB_control", "title": "CONTROL_FLOW.md"},
                ],
                "source_query": user_text,
                "result_mode": "inventory",
            }
            answer = "1. ARCHITECTURE.md (KB_architecture)\n2. CONTROL_FLOW.md (KB_control)"
            return SimpleNamespace(
                text=answer,
                messages=list(session_state.messages)
                + [RuntimeMessage(role="assistant", content=answer, metadata={"doc_focus_result": doc_focus_result})],
                metadata={"doc_focus_result": doc_focus_result},
            )

        active_doc_focus = dict(session_state.metadata.get("active_doc_focus") or {})
        assert [item.get("doc_id") for item in (active_doc_focus.get("documents") or [])] == [
            "KB_architecture",
            "KB_control",
        ]
        answer = "Scoped docs: ARCHITECTURE.md, CONTROL_FLOW.md"
        return SimpleNamespace(
            text=answer,
            messages=list(session_state.messages) + [RuntimeMessage(role="assistant", content=answer, metadata={})],
            metadata={},
        )

    monkeypatch.setattr(bot.kernel, "run_agent", fake_run_agent)

    runtime = api_main.Runtime(settings=settings, bot=bot)
    api_main.app.dependency_overrides[api_main.get_runtime_or_503] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    try:
        first = await client.post(
            "/v1/chat/completions",
            headers={"X-Conversation-ID": "doc-focus-conv"},
            json={
                "model": "enterprise-agent",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Goal: Investigate the major subsystems in this repo and provide me a list of documents "
                            "that have information about the major sub systems"
                        ),
                    }
                ],
            },
        )
        second = await client.post(
            "/v1/chat/completions",
            headers={"X-Conversation-ID": "doc-focus-conv"},
            json={
                "model": "enterprise-agent",
                "messages": [
                    {
                        "role": "user",
                        "content": "Can you look through the candidate documents you provided and give me a detailed summary of the major subsystems involved?",
                    }
                ],
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["choices"][0]["message"]["content"] == "Scoped docs: ARCHITECTURE.md, CONTROL_FLOW.md"


@pytest.mark.asyncio
async def test_chat_completions_returns_artifacts_in_non_stream_payload(tmp_path):
    settings = _make_settings(tmp_path)
    bot = DummyBot(
        assistant_artifacts=[
            {
                "download_id": "dl_123",
                "artifact_ref": "download://dl_123",
                "filename": "analysis.xlsx",
                "label": "analysis.xlsx",
                "download_url": "/v1/files/dl_123?conversation_id=artifacts-conv",
                "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "size_bytes": 128,
                "session_id": "local-dev:local-cli:artifacts-conv",
                "conversation_id": "artifacts-conv",
            }
        ]
    )
    runtime = api_main.Runtime(settings=settings, bot=bot)
    api_main.app.dependency_overrides[api_main.get_runtime_or_503] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    try:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Conversation-ID": "artifacts-conv"},
            json={
                "model": "enterprise-agent",
                "messages": [{"role": "user", "content": "Return the file"}],
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifacts"][0]["filename"] == "analysis.xlsx"


@pytest.mark.asyncio
async def test_chat_completions_returns_long_output_metadata_and_passes_request_metadata(tmp_path):
    settings = _make_settings(tmp_path)
    bot = DummyBot(
        assistant_metadata={
            "job_id": "job_long_1",
            "long_output": {
                "background": True,
                "status": "queued",
                "delivery_mode": "hybrid",
            },
        }
    )
    runtime = api_main.Runtime(settings=settings, bot=bot)
    api_main.app.dependency_overrides[api_main.get_runtime_or_503] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "enterprise-agent",
                "messages": [{"role": "user", "content": "Draft a very long report."}],
                "metadata": {
                    "long_output": {
                        "enabled": True,
                        "target_words": 4000,
                        "background_ok": True,
                    }
                },
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata"]["job_id"] == "job_long_1"
    assert payload["metadata"]["long_output"]["background"] is True
    assert bot.calls[0]["request_metadata"]["long_output"]["enabled"] is True


@pytest.mark.asyncio
async def test_chat_completions_returns_clarification_metadata(tmp_path):
    settings = _make_settings(tmp_path)
    bot = DummyBot(
        assistant_metadata={
            "turn_outcome": "clarification_request",
            "clarification": {
                "question": "Should I use uploaded files only, the knowledge base only, both, or neither?",
                "reason": "retrieval_scope_ambiguous",
                "options": ["uploaded files only", "knowledge base only", "both", "neither"],
                "blocking": True,
            },
        }
    )
    runtime = api_main.Runtime(settings=settings, bot=bot)
    api_main.app.dependency_overrides[api_main.get_runtime_or_503] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "enterprise-agent",
                "messages": [{"role": "user", "content": "Compare the docs."}],
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata"]["turn_outcome"] == "clarification_request"
    assert payload["metadata"]["clarification"]["reason"] == "retrieval_scope_ambiguous"
    assert payload["metadata"]["clarification"]["options"] == [
        "uploaded files only",
        "knowledge base only",
        "both",
        "neither",
    ]


@pytest.mark.asyncio
async def test_chat_completions_streaming_returns_sse_chunks(tmp_path):
    client, bot, _ = _make_client(tmp_path)
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "enterprise-agent",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"object": "chat.completion.chunk"' in response.text
    assert "event: status" in response.text
    assert "data: [DONE]" in response.text
    assert len(bot.calls) == 1
    assert len(bot.calls[0]["extra_callbacks"]) == 1
    events = _sse_events(response.text)
    status_events = [event["data"] for event in events if event["event"] == "status"]
    assert status_events[0]["description"] == "Starting • 00:00 elapsed"
    assert status_events[-1]["phase"] == "answer_ready"
    assert status_events[-1]["done"] is True


@pytest.mark.asyncio
async def test_chat_completions_streaming_emits_artifacts_event(tmp_path):
    settings = _make_settings(tmp_path)
    bot = DummyBot(
        assistant_artifacts=[
            {
                "download_id": "dl_stream",
                "artifact_ref": "download://dl_stream",
                "filename": "report.csv",
                "label": "report.csv",
                "download_url": "/v1/files/dl_stream?conversation_id=stream-artifacts",
                "content_type": "text/csv",
                "size_bytes": 42,
                "session_id": "local-dev:local-cli:stream-artifacts",
                "conversation_id": "stream-artifacts",
            }
        ]
    )
    runtime = api_main.Runtime(settings=settings, bot=bot)
    api_main.app.dependency_overrides[api_main.get_runtime_or_503] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    try:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Conversation-ID": "stream-artifacts"},
            json={
                "model": "enterprise-agent",
                "messages": [{"role": "user", "content": "Return the file"}],
                "stream": True,
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert "event: artifacts" in response.text
    assert "report.csv" in response.text


@pytest.mark.asyncio
async def test_connector_chat_streams_ai_sdk_protocol_and_artifacts(tmp_path):
    settings = _make_settings(tmp_path)
    settings.gateway_shared_bearer_token = "gateway-secret"
    settings.connector_secret_api_key = "connector-secret"
    bot = DummyBot(
        answer="connector answer",
        assistant_artifacts=[
            {
                "download_id": "dl_connector",
                "artifact_ref": "download://dl_connector",
                "filename": "connector.csv",
                "label": "connector.csv",
                "download_url": "/v1/files/dl_connector?conversation_id=connector-chat",
                "content_type": "text/csv",
                "size_bytes": 64,
                "session_id": "local-dev:local-cli:connector-chat",
                "conversation_id": "connector-chat",
            }
        ],
        assistant_metadata={"job_id": "job_connector_1"},
    )
    runtime = api_main.Runtime(settings=settings, bot=bot)
    api_main.app.dependency_overrides[api_main.get_runtime_or_503] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    try:
        response = await client.post(
            "/v1/connector/chat",
            headers={"Authorization": "Bearer connector-secret"},
            json={
                "id": "connector-chat",
                "messages": [
                    {
                        "id": "user-msg-1",
                        "role": "user",
                        "parts": [{"type": "text", "text": "Hello from connector client"}],
                    }
                ],
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert response.headers["x-vercel-ai-ui-message-stream"] == "v1"
    parts = _connector_stream_parts(response.text)
    assert parts[0]["type"] == "start"
    assert any(part.get("type") == "text-start" for part in parts if isinstance(part, dict))
    text = "".join(part["delta"] for part in parts if isinstance(part, dict) and part.get("type") == "text-delta")
    assert "connector answer" in text
    assert any(part.get("type") == "data-artifact" for part in parts if isinstance(part, dict))
    assert any(part.get("type") == "file" for part in parts if isinstance(part, dict))
    assert any(part.get("type") == "finish" for part in parts if isinstance(part, dict))
    assert parts[-1] == "[DONE]"
    assert bot.calls[0]["request_metadata"]["connector_client"] is True
    assert bot.calls[0]["request_metadata"]["connector_token_type"] == "secret"


@pytest.mark.asyncio
async def test_connector_chat_uses_async_client_stream_bridge_without_manual_send(tmp_path, monkeypatch):
    settings = _make_settings(tmp_path)
    settings.gateway_shared_bearer_token = "gateway-secret"
    settings.connector_secret_api_key = "connector-secret"
    bot = DummyBot()
    runtime = api_main.Runtime(settings=settings, bot=bot)
    api_main.app.dependency_overrides[api_main.get_runtime_or_503] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )

    captured: dict[str, object] = {}

    class _StreamResponse:
        status_code = 200

        async def aread(self) -> bytes:
            return b""

        async def aiter_lines(self):
            lines = [
                'data: {"choices":[{"delta":{"content":"Bridge stream ok."},"finish_reason":null}]}',
                "",
                "data: [DONE]",
                "",
            ]
            for item in lines:
                yield item

    class _StreamContext:
        async def __aenter__(self):
            return _StreamResponse()

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _InternalAsyncClient:
        def __init__(self, *args, **kwargs):
            del args
            captured["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def stream(self, method, url, *, headers=None, json=None, **kwargs):
            del kwargs
            captured["stream"] = {
                "method": method,
                "url": url,
                "headers": headers or {},
                "json": json or {},
            }
            return _StreamContext()

    monkeypatch.setattr(api_main.httpx, "AsyncClient", _InternalAsyncClient)

    try:
        response = await client.post(
            "/v1/connector/chat",
            headers={"Authorization": "Bearer connector-secret"},
            json={
                "id": "connector-chat-stream-bridge",
                "messages": [
                    {
                        "id": "user-msg-1",
                        "role": "user",
                        "parts": [{"type": "text", "text": "Hello from connector client"}],
                    }
                ],
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    parts = _connector_stream_parts(response.text)
    text = "".join(part["delta"] for part in parts if isinstance(part, dict) and part.get("type") == "text-delta")
    assert text == "Bridge stream ok."
    assert captured["stream"]["method"] == "POST"
    assert captured["stream"]["url"] == "/v1/chat/completions"
    assert captured["stream"]["headers"]["Content-Type"] == "application/json"
    assert captured["stream"]["json"]["stream"] is True
    assert captured["stream"]["json"]["metadata"]["connector_client"] is True


def test_initial_connector_status_uses_graph_catalog_phase_for_graph_inventory():
    phase, label = api_main._initial_connector_status("what graphs do i have access to")

    assert phase == api_main.PHASE_GRAPH_CATALOG
    assert label == "Inspecting graph catalog"


def test_connector_status_parts_include_structured_agent_fields_and_one_second_heartbeats():
    tracker = api_main.TurnStatusTracker(turn_started_at=100.0)

    start = tracker.start_snapshots(100.0)[0]
    searching = tracker.transition_phase(
        api_main.PHASE_SEARCHING,
        now=100.0,
        source_event_type="chat_request_started",
        label="Searching knowledge base",
        detail="Waiting for the gateway to stream runtime progress",
    )[0]
    routed = tracker.progress_snapshots(
        {"type": "route_decision", "label": "Routed to general", "agent": "general"},
        100.0,
    )[0]
    worker = tracker.progress_snapshots(
        {
            "type": "worker_start",
            "label": "Search KB set",
            "agent": "rag_worker",
            "task_id": "task_1",
            "detail": "Collecting evidence",
        },
        101.0,
    )[0]
    assert tracker.seconds_until_next_heartbeat(102.0, interval_seconds=api_main.STATUS_HEARTBEAT_SECONDS) == 0.0
    heartbeat = tracker.heartbeat_snapshot(102.0)
    synth_snapshots = tracker.transition_phase(
        api_main.PHASE_SYNTHESIZING,
        now=103.0,
        source_event_type="content_delta",
        label="Synthesizing answer",
        detail="Grounding final response",
    )
    completed = tracker.completion_snapshots(104.0)[-1]

    status_parts = [
        _connector_stream_parts(api_main._connector_status_part("status_test", payload))[0]
        for payload in [start, searching, routed, worker, heartbeat, *synth_snapshots, completed]
        if payload is not None
    ]

    assert status_parts[0]["type"] == "data-status"
    assert status_parts[0]["data"]["description"] == "Starting • 00:00 elapsed"
    assert status_parts[0]["data"]["agentic_status"]["version"] == 1
    assert status_parts[0]["data"]["agentic_status"]["state"] == "active"
    assert status_parts[0]["data"]["agentic_status"]["kind"] == "thinking"
    assert status_parts[0]["data"]["agentic_status"]["title"] == "Thinking"
    assert status_parts[0]["data"]["agentic_status"]["subtitle"] == "Preparing the agent workflow."
    assert status_parts[0]["data"]["agentic_status"]["chips"] == ["Thinking"]
    assert status_parts[0]["data"]["agentic_status"]["timing"] == {
        "kind": "stage",
        "live": True,
        "elapsed_ms": 0,
        "status_elapsed_ms": 0,
        "phase_elapsed_ms": 0,
        "total_elapsed_ms": 0,
        "snapshot_timestamp_ms": status_parts[0]["data"]["timestamp"],
    }
    assert status_parts[2]["data"]["selected_agent"] == "general"
    assert status_parts[2]["data"]["agent"] == "general"
    assert status_parts[2]["data"]["agentic_status"]["title"] == "Routing request"
    assert status_parts[2]["data"]["agentic_status"]["kind"] == "routing"
    assert status_parts[2]["data"]["agentic_status"]["subtitle"] == "Choosing the best specialist for this request."
    assert status_parts[2]["data"]["agentic_status"]["chips"] == ["Research", "General Agent"]
    assert status_parts[2]["data"]["agentic_status"]["timing"] == {
        "kind": "stage",
        "live": True,
        "elapsed_ms": 0,
        "status_elapsed_ms": 0,
        "phase_elapsed_ms": 0,
        "total_elapsed_ms": 0,
        "snapshot_timestamp_ms": status_parts[2]["data"]["timestamp"],
    }
    assert status_parts[3]["data"]["selected_agent"] == "general"
    assert status_parts[3]["data"]["agent"] == "rag_worker"
    assert status_parts[2]["data"]["status_id"] != status_parts[3]["data"]["status_id"]
    assert status_parts[3]["data"]["status_key"] == "searching_knowledge_base\u241frag_worker\u241factive"
    assert status_parts[3]["data"]["agentic_status"]["title"] == "Researching evidence"
    assert status_parts[3]["data"]["agentic_status"]["kind"] == "worker"
    assert status_parts[3]["data"]["agentic_status"]["subtitle"] == "Collecting evidence"
    assert status_parts[3]["data"]["agentic_status"]["chips"] == [
        "Research",
        "RAG Worker",
        "Route: General Agent",
    ]
    assert status_parts[3]["data"]["agentic_status"]["timing"] == {
        "kind": "stage",
        "live": True,
        "elapsed_ms": 0,
        "status_elapsed_ms": 0,
        "phase_elapsed_ms": 1000,
        "total_elapsed_ms": 1000,
        "snapshot_timestamp_ms": status_parts[3]["data"]["timestamp"],
    }
    assert heartbeat is not None
    assert status_parts[4]["data"]["description"] == "Searching knowledge base • rag_worker • 00:02 elapsed"
    assert status_parts[4]["data"]["phase_elapsed_ms"] == 2000
    assert status_parts[4]["data"]["status_elapsed_ms"] == 1000
    assert status_parts[4]["data"]["status_id"] == status_parts[3]["data"]["status_id"]
    assert status_parts[4]["data"]["status_seq"] > status_parts[3]["data"]["status_seq"]
    assert status_parts[4]["data"]["task_id"] == "task_1"
    assert status_parts[4]["data"]["source_event_type"] == "heartbeat"
    assert status_parts[4]["data"]["agentic_status"]["chips"] == [
        "Research",
        "RAG Worker",
        "Route: General Agent",
        "Live",
    ]
    assert status_parts[4]["data"]["agentic_status"]["timing"] == {
        "kind": "stage",
        "live": True,
        "elapsed_ms": 1000,
        "status_elapsed_ms": 1000,
        "phase_elapsed_ms": 2000,
        "total_elapsed_ms": 2000,
        "snapshot_timestamp_ms": status_parts[4]["data"]["timestamp"],
    }
    assert status_parts[5]["data"]["phase"] == "searching_knowledge_base"
    assert status_parts[5]["data"]["status"] == "complete"
    assert status_parts[5]["data"]["done"] is False
    assert status_parts[5]["data"]["status_id"] == status_parts[4]["data"]["status_id"]
    assert status_parts[5]["data"]["agentic_status"]["title"] == "Researching evidence"
    assert status_parts[5]["data"]["agentic_status"]["kind"] == "research"
    assert status_parts[5]["data"]["agentic_status"]["timing"] == {
        "kind": "stage",
        "live": False,
        "elapsed_ms": 2000,
        "status_elapsed_ms": 2000,
        "phase_elapsed_ms": 3000,
        "total_elapsed_ms": 3000,
        "snapshot_timestamp_ms": status_parts[5]["data"]["timestamp"],
    }
    assert status_parts[6]["data"]["phase"] == "synthesizing_answer"
    assert status_parts[6]["data"]["agentic_status"]["title"] == "Writing answer"
    assert status_parts[6]["data"]["agentic_status"]["timing"] == {
        "kind": "stage",
        "live": True,
        "elapsed_ms": 0,
        "status_elapsed_ms": 0,
        "phase_elapsed_ms": 0,
        "total_elapsed_ms": 3000,
        "snapshot_timestamp_ms": status_parts[6]["data"]["timestamp"],
    }
    assert status_parts[-1]["data"]["phase"] == "answer_ready"
    assert status_parts[-1]["data"]["done"] is True
    assert status_parts[-1]["data"]["status_id"] != status_parts[-2]["data"]["status_id"]
    assert status_parts[-1]["data"]["agentic_status"]["version"] == 1
    assert status_parts[-1]["data"]["agentic_status"]["state"] == "complete"
    assert status_parts[-1]["data"]["agentic_status"]["kind"] == "ready"
    assert status_parts[-1]["data"]["agentic_status"]["title"] == "Answer ready"
    assert status_parts[-1]["data"]["agentic_status"]["subtitle"] == "Grounded response is ready."
    assert status_parts[-1]["data"]["agentic_status"]["chips"] == ["Ready", "RAG Worker", "Route: General Agent"]
    assert status_parts[-1]["data"]["agentic_status"]["timing"] == {
        "kind": "total",
        "live": False,
        "elapsed_ms": 4000,
        "status_elapsed_ms": 0,
        "phase_elapsed_ms": 4000,
        "total_elapsed_ms": 4000,
        "snapshot_timestamp_ms": status_parts[-1]["data"]["timestamp"],
    }
    assert all("phase_elapsed_ms" in part["data"] for part in status_parts)
    assert all("status_id" in part["data"] for part in status_parts)
    assert all("status_key" in part["data"] for part in status_parts)
    assert all("status_seq" in part["data"] for part in status_parts)
    assert all("status_elapsed_ms" in part["data"] for part in status_parts)
    assert all(part["data"]["agentic_status"]["version"] == 1 for part in status_parts)


def test_turn_status_tracker_reuses_status_id_for_heartbeats_and_rotates_on_actor_change():
    tracker = api_main.TurnStatusTracker(turn_started_at=10.0)

    start = tracker.start_snapshots(10.0)[0]
    searching = tracker.transition_phase(
        api_main.PHASE_SEARCHING,
        now=10.0,
        source_event_type="phase_start",
        label="Searching knowledge base",
    )[0]
    routed = tracker.progress_snapshots(
        {"type": "route_decision", "label": "Routed to general", "agent": "general"},
        11.0,
    )[0]
    heartbeat = tracker.heartbeat_snapshot(12.0)
    assert heartbeat is not None
    synth = tracker.transition_phase(
        api_main.PHASE_SYNTHESIZING,
        now=13.0,
        source_event_type="content_delta",
        label="Synthesizing answer",
    )[-1]

    assert start["status_id"] != searching["status_id"]
    assert searching["status_id"] != routed["status_id"]
    assert routed["status_id"] == heartbeat["status_id"]
    assert heartbeat["status_elapsed_ms"] == 1000
    assert synth["status_id"] != heartbeat["status_id"]
    assert synth["status_seq"] > heartbeat["status_seq"]
    assert routed["agentic_status"]["chips"] == ["Research", "General Agent"]
    assert routed["agentic_status"]["timing"]["kind"] == "stage"
    assert routed["agentic_status"]["timing"]["live"] is True
    assert routed["agentic_status"]["timing"]["elapsed_ms"] == 0
    assert heartbeat["agentic_status"]["chips"] == ["Research", "General Agent", "Live"]
    assert heartbeat["agentic_status"]["timing"]["kind"] == "stage"
    assert heartbeat["agentic_status"]["timing"]["live"] is True
    assert heartbeat["agentic_status"]["timing"]["elapsed_ms"] == 1000
    assert synth["agentic_status"]["title"] == "Writing answer"
    assert synth["agentic_status"]["kind"] == "synthesizing"
    assert synth["agentic_status"]["timing"]["kind"] == "stage"
    assert synth["agentic_status"]["timing"]["live"] is True
    assert synth["agentic_status"]["timing"]["elapsed_ms"] == 0


@pytest.mark.asyncio
async def test_connector_chat_bridges_ai_sdk_file_parts_into_upload_endpoint(tmp_path, monkeypatch):
    settings = _make_settings(tmp_path)
    settings.gateway_shared_bearer_token = "gateway-secret"
    settings.connector_secret_api_key = "connector-secret"
    bot = DummyBot(answer="file analyzed")
    runtime = api_main.Runtime(settings=settings, bot=bot)
    api_main.app.dependency_overrides[api_main.get_runtime_or_503] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings

    captured: dict[str, object] = {}

    def fake_ingest_paths(settings, stores, paths, *, source_type, tenant_id, collection_id=None, **kwargs):
        del settings, stores, source_type
        resolved = [Path(item) for item in paths]
        captured["paths"] = resolved
        captured["tenant_id"] = tenant_id
        captured["collection_id"] = collection_id
        return ["doc-upload-1"]

    monkeypatch.setattr(api_main, "ingest_paths", fake_ingest_paths)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    data_url = "data:text/csv;base64," + base64.b64encode(b"region,revenue\nNA,120\n").decode("ascii")
    try:
        response = await client.post(
            "/v1/connector/chat",
            headers={"Authorization": "Bearer connector-secret"},
            json={
                "id": "connector-file-chat",
                "messages": [
                    {
                        "id": "user-file-msg",
                        "role": "user",
                        "parts": [
                            {"type": "text", "text": "Analyze the attached CSV."},
                            {
                                "type": "file",
                                "id": "file-part-1",
                                "filename": "sales.csv",
                                "mediaType": "text/csv",
                                "url": data_url,
                            },
                        ],
                    }
                ],
                "metadata": {
                    "upload_collection_id": "owui-chat-connector-file-chat",
                    "kb_collection_id": "default",
                },
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert captured["collection_id"] == "owui-chat-connector-file-chat"
    saved_paths = captured["paths"]
    assert isinstance(saved_paths, list)
    assert saved_paths
    assert saved_paths[0].name == "sales.csv"
    assert saved_paths[0].read_text(encoding="utf-8") == "region,revenue\nNA,120\n"
    session_id = "local-dev:local-cli:connector-file-chat"
    state = bot.kernel.transcript_store.states[session_id]
    assert state.uploaded_doc_ids == ["doc-upload-1"]
    assert state.metadata["upload_collection_id"] == "owui-chat-connector-file-chat"
    assert (settings.workspace_dir / filesystem_key(session_id) / "sales.csv").exists()
    assert bot.calls[0]["request_metadata"]["upload_collection_id"] == "owui-chat-connector-file-chat"
    assert bot.calls[0]["request_metadata"]["kb_collection_id"] == "default"


@pytest.mark.asyncio
async def test_connector_chat_accepts_multipart_payload_and_files(tmp_path, monkeypatch):
    settings = _make_settings(tmp_path)
    settings.gateway_shared_bearer_token = "gateway-secret"
    settings.connector_secret_api_key = "connector-secret"
    bot = DummyBot(answer="multipart analyzed")
    runtime = api_main.Runtime(settings=settings, bot=bot)
    api_main.app.dependency_overrides[api_main.get_runtime_or_503] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings

    captured: dict[str, object] = {}

    def fake_ingest_paths(settings, stores, paths, *, source_type, tenant_id, collection_id=None, **kwargs):
        del settings, stores, source_type
        captured["paths"] = [Path(item) for item in paths]
        captured["tenant_id"] = tenant_id
        captured["collection_id"] = collection_id
        return ["doc-upload-2"]

    monkeypatch.setattr(api_main, "ingest_paths", fake_ingest_paths)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    try:
        response = await client.post(
            "/v1/connector/chat",
            headers={"Authorization": "Bearer connector-secret"},
            data={
                "payload": json.dumps(
                    {
                        "id": "connector-multipart-chat",
                        "messages": [
                            {
                                "id": "user-multipart-msg",
                                "role": "user",
                                "parts": [{"type": "text", "text": "Analyze the attached text file."}],
                            }
                        ],
                        "metadata": {
                            "upload_collection_id": "owui-chat-connector-multipart-chat",
                            "kb_collection_id": "default",
                        },
                    }
                )
            },
            files={"files": ("notes.txt", b"hello connector", "text/plain")},
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert captured["collection_id"] == "owui-chat-connector-multipart-chat"
    saved_paths = captured["paths"]
    assert isinstance(saved_paths, list)
    assert saved_paths[0].name == "notes.txt"
    assert saved_paths[0].read_text(encoding="utf-8") == "hello connector"
    session_id = "local-dev:local-cli:connector-multipart-chat"
    assert (settings.workspace_dir / filesystem_key(session_id) / "notes.txt").exists()
    assert bot.calls[0]["request_metadata"]["upload_collection_id"] == "owui-chat-connector-multipart-chat"


@pytest.mark.asyncio
async def test_connector_chat_allows_publishable_browser_key_only_for_configured_origin(tmp_path):
    settings = _make_settings(tmp_path)
    settings.connector_secret_api_key = ""
    settings.connector_publishable_api_key = "pk-live-demo"
    settings.connector_allowed_origins = ("https://chat.example.com",)
    bot = DummyBot()
    runtime = api_main.Runtime(settings=settings, bot=bot)
    api_main.app.dependency_overrides[api_main.get_runtime_or_503] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    try:
        forbidden = await client.post(
            "/v1/connector/chat",
            headers={"Authorization": "Bearer pk-live-demo"},
            json={
                "id": "browser-chat",
                "messages": [{"id": "user-browser-1", "role": "user", "parts": [{"type": "text", "text": "Hello"}]}],
            },
        )
        allowed = await client.post(
            "/v1/connector/chat",
            headers={
                "Authorization": "Bearer pk-live-demo",
                "Origin": "https://chat.example.com",
            },
            json={
                "id": "browser-chat",
                "messages": [{"id": "user-browser-2", "role": "user", "parts": [{"type": "text", "text": "Hello"}]}],
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert forbidden.status_code == 403
    assert allowed.status_code == 200
    assert bot.calls[0]["request_metadata"]["connector_token_type"] == "publishable"


@pytest.mark.asyncio
async def test_get_job_status_returns_artifacts_for_session_scoped_job(tmp_path):
    settings = _make_settings(tmp_path)
    bot = DummyBot()
    bot.kernel.job_manager.jobs["job_long_1"] = SimpleNamespace(
        job_id="job_long_1",
        session_id="local-dev:local-cli:job-conv",
        agent_name="general",
        status="completed",
        tenant_id="local-dev",
        user_id="local-cli",
        priority="background",
        queue_class="background",
        scheduler_state="completed",
        description="Long-form draft",
        result_summary="Completed long-form draft.",
        output_path=str(tmp_path / "workspaces" / "job-conv-report.md"),
        result_path=str(tmp_path / "runtime" / "jobs" / "job_long_1" / "result.json"),
        last_error="",
        enqueued_at="2026-04-08T11:58:00+00:00",
        started_at="2026-04-08T11:59:00+00:00",
        estimated_token_cost=2048,
        actual_token_cost=1822,
        budget_block_reason="",
        updated_at="2026-04-08T12:00:00+00:00",
        metadata={
            "artifacts": [
                {
                    "download_id": "dl_job_1",
                    "artifact_ref": "download://dl_job_1",
                    "filename": "job-conv-report.md",
                    "label": "job-conv-report.md",
                    "download_url": "/v1/files/dl_job_1?conversation_id=job-conv",
                    "content_type": "text/markdown",
                    "size_bytes": 512,
                    "session_id": "local-dev:local-cli:job-conv",
                    "conversation_id": "job-conv",
                }
            ],
            "long_output_result": {"title": "Quarterly report", "section_count": 6},
        },
    )
    runtime = api_main.Runtime(settings=settings, bot=bot)
    api_main.app.dependency_overrides[api_main.get_runtime_or_503] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    try:
        response = await client.get(
            "/v1/jobs/job_long_1",
            headers={"X-Conversation-ID": "job-conv"},
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "job_long_1"
    assert payload["queue_class"] == "background"
    assert payload["scheduler_state"] == "completed"
    assert payload["estimated_token_cost"] == 2048
    assert payload["actual_token_cost"] == 1822
    assert payload["artifacts"][0]["filename"] == "job-conv-report.md"
    assert payload["metadata"]["long_output_result"]["title"] == "Quarterly report"
    assert payload["mailbox"]["pending_question_count"] == 0
    assert payload["mailbox"]["pending_approval_count"] == 0


@pytest.mark.asyncio
async def test_job_mailbox_endpoints_list_and_answer_question_request(tmp_path):
    settings = _make_settings(tmp_path)
    bot = DummyBot()
    bot.kernel.job_manager.jobs["job_waiting_1"] = SimpleNamespace(
        job_id="job_waiting_1",
        session_id="local-dev:local-cli:mailbox-conv",
        agent_name="utility",
        status="waiting_message",
    )
    request = WorkerMailboxMessage(
        job_id="job_waiting_1",
        content="Which repo should I inspect?",
        sender="utility",
        message_id="msg_question_1",
        message_type="question_request",
        direction="from_worker",
        status="open",
        requires_response=True,
    )
    bot.kernel.job_manager.mailboxes["job_waiting_1"] = [request]
    runtime = api_main.Runtime(settings=settings, bot=bot)
    api_main.app.dependency_overrides[api_main.get_runtime_or_503] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    try:
        listed = await client.get(
            "/v1/jobs/job_waiting_1/mailbox",
            headers={"X-Conversation-ID": "mailbox-conv"},
        )
        answered = await client.post(
            "/v1/jobs/job_waiting_1/mailbox/msg_question_1/respond",
            json={"response": "Inspect the API repo.", "resume": True},
            headers={"X-Conversation-ID": "mailbox-conv"},
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert listed.status_code == 200
    assert listed.json()["mailbox"]["pending_question_count"] == 1
    assert listed.json()["data"][0]["message_id"] == "msg_question_1"
    assert answered.status_code == 200
    assert answered.json()["response"]["message_type"] == "question_response"
    assert bot.kernel.job_manager.continued == ["job_waiting_1"]


@pytest.mark.asyncio
async def test_job_mailbox_approval_requires_admin_or_permission(tmp_path):
    settings = _make_settings(tmp_path)
    settings.gateway_shared_bearer_token = "gateway-secret"
    bot = DummyBot()
    bot.kernel.job_manager.jobs["job_waiting_approval"] = SimpleNamespace(
        job_id="job_waiting_approval",
        session_id="local-dev:local-cli:mailbox-conv",
        agent_name="utility",
        status="waiting_message",
    )
    request = WorkerMailboxMessage(
        job_id="job_waiting_approval",
        content="Delete generated export.csv",
        sender="utility",
        message_id="msg_approval_1",
        message_type="approval_request",
        direction="from_worker",
        status="open",
        requires_response=True,
    )
    bot.kernel.job_manager.mailboxes["job_waiting_approval"] = [request]
    runtime = api_main.Runtime(settings=settings, bot=bot)
    api_main.app.dependency_overrides[api_main.get_runtime_or_503] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    try:
        forbidden = await client.post(
            "/v1/jobs/job_waiting_approval/mailbox/msg_approval_1/respond",
            json={"response": "Approved.", "decision": "approved"},
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Conversation-ID": "mailbox-conv",
            },
        )
        allowed = await client.post(
            "/v1/jobs/job_waiting_approval/mailbox/msg_approval_1/respond",
            json={"response": "Approved for generated file only.", "decision": "approved"},
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Admin-Token": "admin-secret",
                "X-Conversation-ID": "mailbox-conv",
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert forbidden.status_code == 403
    assert allowed.status_code == 200
    assert allowed.json()["request"]["status"] == "approved"


@pytest.mark.asyncio
async def test_team_mailbox_api_create_post_list_and_answer_question(tmp_path):
    settings = _make_settings(tmp_path)
    settings.team_mailbox_enabled = True
    bot = DummyBot()
    runtime = api_main.Runtime(settings=settings, bot=bot)
    api_main.app.dependency_overrides[api_main.get_runtime_or_503] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    try:
        created = await client.post(
            "/v1/sessions/local-dev:local-cli:team-conv/team-mailbox/channels",
            json={"name": "research", "member_agents": ["general", "utility"]},
            headers={"X-Conversation-ID": "team-conv"},
        )
        channel_id = created.json()["channel"]["channel_id"]
        posted = await client.post(
            f"/v1/sessions/local-dev:local-cli:team-conv/team-mailbox/channels/{channel_id}/messages",
            json={"content": "Which collection?", "message_type": "question_request", "target_agents": ["general"]},
            headers={"X-Conversation-ID": "team-conv"},
        )
        message_id = posted.json()["message"]["message_id"]
        listed = await client.get(
            "/v1/sessions/local-dev:local-cli:team-conv/team-mailbox/messages",
            params={"channel_id": channel_id},
            headers={"X-Conversation-ID": "team-conv"},
        )
        answered = await client.post(
            f"/v1/sessions/local-dev:local-cli:team-conv/team-mailbox/channels/{channel_id}/messages/{message_id}/respond",
            json={"response": "Use default."},
            headers={"X-Conversation-ID": "team-conv"},
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert created.status_code == 200
    assert posted.status_code == 200
    assert listed.status_code == 200
    assert listed.json()["summary"]["pending_question_count"] == 1
    assert answered.status_code == 200
    assert answered.json()["request"]["status"] == "answered"
    assert answered.json()["response"]["message_type"] == "question_response"


@pytest.mark.asyncio
async def test_team_mailbox_api_approval_requires_admin(tmp_path):
    settings = _make_settings(tmp_path)
    settings.team_mailbox_enabled = True
    settings.gateway_shared_bearer_token = "gateway-secret"
    bot = DummyBot()
    runtime = api_main.Runtime(settings=settings, bot=bot)
    api_main.app.dependency_overrides[api_main.get_runtime_or_503] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    try:
        created = await client.post(
            "/v1/sessions/local-dev:local-cli:team-conv/team-mailbox/channels",
            json={"name": "approvals"},
            headers={"Authorization": "Bearer gateway-secret", "X-Conversation-ID": "team-conv"},
        )
        channel_id = created.json()["channel"]["channel_id"]
        posted = await client.post(
            f"/v1/sessions/local-dev:local-cli:team-conv/team-mailbox/channels/{channel_id}/messages",
            json={"content": "Approve export?", "message_type": "approval_request"},
            headers={"Authorization": "Bearer gateway-secret", "X-Conversation-ID": "team-conv"},
        )
        message_id = posted.json()["message"]["message_id"]
        forbidden = await client.post(
            f"/v1/sessions/local-dev:local-cli:team-conv/team-mailbox/channels/{channel_id}/messages/{message_id}/respond",
            json={"response": "Approved.", "decision": "approved"},
            headers={"Authorization": "Bearer gateway-secret", "X-Conversation-ID": "team-conv"},
        )
        allowed = await client.post(
            f"/v1/sessions/local-dev:local-cli:team-conv/team-mailbox/channels/{channel_id}/messages/{message_id}/respond",
            json={"response": "Approved with constraints.", "decision": "approved"},
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Admin-Token": "admin-secret",
                "X-Conversation-ID": "team-conv",
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert forbidden.status_code == 403
    assert allowed.status_code == 200
    assert allowed.json()["request"]["status"] == "approved"


@pytest.mark.asyncio
async def test_team_mailbox_api_validates_job_targets_stay_in_session_and_channel(tmp_path):
    settings = _make_settings(tmp_path)
    settings.team_mailbox_enabled = True
    bot = DummyBot()
    session_id = "local-dev:local-cli:team-conv"
    bot.kernel.job_manager.jobs["job_member"] = SimpleNamespace(session_id=session_id)
    bot.kernel.job_manager.jobs["job_same_session"] = SimpleNamespace(session_id=session_id)
    bot.kernel.job_manager.jobs["job_other_session"] = SimpleNamespace(session_id="other-session")
    runtime = api_main.Runtime(settings=settings, bot=bot)
    api_main.app.dependency_overrides[api_main.get_runtime_or_503] = lambda: runtime
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    try:
        bad_create = await client.post(
            f"/v1/sessions/{session_id}/team-mailbox/channels",
            json={"name": "bad", "member_job_ids": ["job_other_session"]},
            headers={"X-Conversation-ID": "team-conv"},
        )
        created = await client.post(
            f"/v1/sessions/{session_id}/team-mailbox/channels",
            json={"name": "research", "member_job_ids": ["job_member"]},
            headers={"X-Conversation-ID": "team-conv"},
        )
        channel_id = created.json()["channel"]["channel_id"]
        bad_post = await client.post(
            f"/v1/sessions/{session_id}/team-mailbox/channels/{channel_id}/messages",
            json={"content": "wrong target", "target_job_ids": ["job_same_session"]},
            headers={"X-Conversation-ID": "team-conv"},
        )
        good_post = await client.post(
            f"/v1/sessions/{session_id}/team-mailbox/channels/{channel_id}/messages",
            json={"content": "member target", "target_job_ids": ["job_member"]},
            headers={"X-Conversation-ID": "team-conv"},
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert bad_create.status_code == 400
    assert "not in this session" in bad_create.json()["detail"]
    assert created.status_code == 200
    assert bad_post.status_code == 400
    assert "not members" in bad_post.json()["detail"]
    assert good_post.status_code == 200
    assert good_post.json()["message"]["target_job_ids"] == ["job_member"]


def test_stream_with_progress_waits_for_slow_basic_turn_without_progress_events(monkeypatch):
    real_thread_cls = threading.Thread

    class _Events:
        def __init__(self, callback):
            self._callback = callback

        def get(self, timeout=None):
            del timeout
            if self._callback.done:
                return None
            raise queue.Empty

    class _FakeProgressCallback:
        def __init__(self):
            self.done = False
            self.events = _Events(self)

        def mark_done(self):
            self.done = True

    class _JoinlessThread:
        def __init__(self, target, daemon=False):
            self._thread = real_thread_cls(target=target, daemon=daemon)

        def start(self):
            self._thread.start()

        def is_alive(self):
            return self._thread.is_alive()

        def join(self, timeout=None):
            del timeout
            return None

    class _SlowBot:
        def process_turn(self, session, *, user_text, upload_paths=None, force_agent=False, requested_agent="", extra_callbacks=None):
            del session, user_text, upload_paths, force_agent, requested_agent, extra_callbacks
            threading.Event().wait(0.05)
            return "Late answer"

    monkeypatch.setattr(api_main, "ProgressCallback", _FakeProgressCallback)
    monkeypatch.setattr(api_main.threading, "Thread", _JoinlessThread)

    payload = "".join(
        api_main._stream_with_progress(
            model="enterprise-agent",
            session=object(),
            user_text="Hello",
            bot=_SlowBot(),
            force_agent=False,
            requested_agent="",
            prompt_tokens=1,
        )
    )

    assert '"content": "Late answer"' in payload
    assert "data: [DONE]" in payload


def test_stream_with_progress_forwards_rich_progress_events_in_order():
    class _ProgressBot:
        def process_turn(self, session, *, user_text, upload_paths=None, force_agent=False, requested_agent="", extra_callbacks=None, progress_sink=None):
            del session, user_text, upload_paths, force_agent, requested_agent, extra_callbacks
            assert progress_sink is not None
            progress_sink.emit_progress(
                "route_decision",
                label="Routed to rag_worker",
                detail="Grounded answer requested",
                agent="rag_worker",
            )
            progress_sink.emit_progress(
                "agent_selected",
                label="Running rag_worker",
                agent="rag_worker",
            )
            progress_sink.emit_progress(
                "worker_start",
                label="Search document group 1",
                agent="rag_worker",
                task_id="task_1",
                job_id="job_1",
                docs=[{"doc_id": "doc-1", "title": "policy.md"}],
                status="running",
            )
            progress_sink.emit_progress(
                "doc_focus",
                label="Reviewing candidate documents",
                docs=[{"doc_id": "doc-1", "title": "policy.md"}],
            )
            return "Streamed answer"

    payload = "".join(
        api_main._stream_with_progress(
            model="enterprise-agent",
            session=object(),
            user_text="Hello",
            bot=_ProgressBot(),
            force_agent=False,
            requested_agent="",
            prompt_tokens=1,
        )
    )

    route_pos = payload.find('"type": "route_decision"')
    agent_pos = payload.find('"type": "agent_selected"')
    worker_pos = payload.find('"type": "worker_start"')
    focus_pos = payload.find('"type": "doc_focus"')

    assert -1 not in {route_pos, agent_pos, worker_pos, focus_pos}
    assert route_pos < agent_pos < worker_pos < focus_pos
    assert '"content": "Streamed answer"' in payload


def test_stream_with_progress_forwards_agent_context_loaded_status_event():
    class _ProgressBot:
        def process_turn(
            self,
            session,
            *,
            user_text,
            upload_paths=None,
            force_agent=False,
            requested_agent="",
            extra_callbacks=None,
            progress_sink=None,
        ):
            del session, user_text, upload_paths, force_agent, requested_agent, extra_callbacks
            assert progress_sink is not None
            progress_sink.emit(
                RuntimeEvent(
                    event_id="evt-context",
                    event_type="agent_context_loaded",
                    session_id="tenant:user:conv",
                    agent_name="general",
                    payload={
                        "title": "general loaded prompt, skills, and context",
                        "detail": "1 prompt doc(s), 0 skill doc(s), 1 context section(s)",
                        "detail_level": "safe_preview",
                        "prompt_docs": [{"kind": "agent_definition", "source_path": "/tmp/general.md"}],
                        "skill_docs": [],
                        "context_sections": [{"name": "base_prompt"}],
                        "memory_context": {},
                    },
                )
            )
            return "Context streamed answer"

    payload = "".join(
        api_main._stream_with_progress(
            model="enterprise-agent",
            session=object(),
            user_text="Hello",
            bot=_ProgressBot(),
            force_agent=False,
            requested_agent="",
            prompt_tokens=1,
        )
    )

    assert "event: status" in payload
    assert '"type": "context_trace"' in payload
    assert '"source_event_type": "agent_context_loaded"' in payload
    assert '"kind": "context"' in payload
    assert '"content": "Context streamed answer"' in payload


def test_stream_with_progress_emits_structured_status_events_for_agents():
    class _ProgressBot:
        def process_turn(self, session, *, user_text, upload_paths=None, force_agent=False, requested_agent="", extra_callbacks=None, progress_sink=None):
            del session, user_text, upload_paths, force_agent, requested_agent, extra_callbacks
            assert progress_sink is not None
            progress_sink.emit_progress(
                "route_decision",
                label="Routed to general",
                agent="general",
                selected_agent="general",
            )
            progress_sink.emit_progress(
                "worker_start",
                label="Search evidence set",
                detail="Collecting evidence",
                agent="rag_worker",
                selected_agent="general",
                task_id="task_7",
                phase=api_main.PHASE_SEARCHING,
            )
            return "Streamed answer"

    payload = "".join(
        api_main._stream_with_progress(
            model="enterprise-agent",
            session=object(),
            user_text="Hello",
            bot=_ProgressBot(),
            force_agent=False,
            requested_agent="",
            prompt_tokens=1,
        )
    )

    events = _sse_events(payload)
    status_events = [event["data"] for event in events if event["event"] == "status"]
    progress_events = [event["data"] for event in events if event["event"] == "progress"]

    assert progress_events[0]["type"] == "route_decision"
    assert status_events[0]["description"] == "Starting • 00:00 elapsed"
    assert any(item["agent"] == "general" and item["selected_agent"] == "general" for item in status_events)
    assert any(item["agent"] == "rag_worker" and item["selected_agent"] == "general" for item in status_events)
    assert any(item["phase"] == "searching_knowledge_base" for item in status_events)
    assert status_events[0]["agentic_status"]["title"] == "Thinking"
    assert any(item["agentic_status"]["title"] == "Routing request" for item in status_events)
    assert any(item["agentic_status"]["title"] == "Researching evidence" for item in status_events)
    assert status_events[-1]["phase"] == "answer_ready"
    assert status_events[-1]["done"] is True
    assert status_events[-1]["agentic_status"]["title"] == "Answer ready"


def test_stream_with_progress_serializes_datetime_progress_payloads():
    seen_at = datetime(2026, 4, 20, 15, 0, 0, tzinfo=timezone.utc)

    class _ProgressBot:
        def process_turn(self, session, *, user_text, upload_paths=None, force_agent=False, requested_agent="", extra_callbacks=None, progress_sink=None):
            del session, user_text, upload_paths, force_agent, requested_agent, extra_callbacks
            assert progress_sink is not None
            progress_sink.emit_progress(
                "route_decision",
                label="Routed to general",
                agent="general",
                selected_agent="general",
                seen_at=seen_at,
            )
            return "Streamed answer"

    payload = "".join(
        api_main._stream_with_progress(
            model="enterprise-agent",
            session=object(),
            user_text="Hello",
            bot=_ProgressBot(),
            force_agent=False,
            requested_agent="",
            prompt_tokens=1,
        )
    )

    events = _sse_events(payload)
    progress_events = [event["data"] for event in events if event["event"] == "progress"]

    assert progress_events[0]["seen_at"] == seen_at.isoformat()


def test_stream_with_progress_emits_heartbeat_status_events(monkeypatch):
    class _SlowProgressBot:
        def process_turn(self, session, *, user_text, upload_paths=None, force_agent=False, requested_agent="", extra_callbacks=None, progress_sink=None):
            del session, user_text, upload_paths, force_agent, requested_agent, extra_callbacks
            assert progress_sink is not None
            progress_sink.emit_progress(
                "route_decision",
                label="Routed to general",
                agent="general",
                selected_agent="general",
            )
            threading.Event().wait(0.03)
            return "Delayed answer"

    monkeypatch.setattr(api_main, "STATUS_HEARTBEAT_SECONDS", 0.01)

    payload = "".join(
        api_main._stream_with_progress(
            model="enterprise-agent",
            session=object(),
            user_text="Hello",
            bot=_SlowProgressBot(),
            force_agent=False,
            requested_agent="",
            prompt_tokens=1,
        )
    )

    events = _sse_events(payload)
    status_events = [event["data"] for event in events if event["event"] == "status"]

    assert any(item["source_event_type"] == "heartbeat" for item in status_events)
    assert status_events[-1]["phase"] == "answer_ready"


@pytest.mark.asyncio
async def test_chat_completions_rejects_unknown_model(tmp_path):
    client, _, _ = _make_client(tmp_path)
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "wrong-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported model: wrong-model"


@pytest.mark.asyncio
async def test_chat_completions_requires_last_message_to_be_user(tmp_path):
    client, _, _ = _make_client(tmp_path)
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "enterprise-agent",
                "messages": [{"role": "assistant", "content": "Hello"}],
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 400
    assert response.json()["detail"] == "last message must have role='user'"


@pytest.mark.asyncio
async def test_ingest_documents_indexes_files_and_copies_them_into_existing_workspace(tmp_path, monkeypatch):
    client, bot, settings = _make_client(tmp_path)
    src = tmp_path / "sales.csv"
    src.write_text("region,revenue\nNA,100\n")
    session_workspace = settings.workspace_dir / filesystem_key("local-dev:local-cli:conv-007")

    captured: dict[str, object] = {}

    def fake_ingest_paths(settings_arg, stores_arg, paths, *, source_type, tenant_id, collection_id=None, **kwargs):
        captured["settings"] = settings_arg
        captured["stores"] = stores_arg
        captured["paths"] = [str(path) for path in paths]
        captured["source_type"] = source_type
        captured["tenant_id"] = tenant_id
        captured["collection_id"] = collection_id
        return ["doc-upload-1"]

    monkeypatch.setattr(api_main, "ingest_paths", fake_ingest_paths)

    try:
        response = await client.post(
            "/v1/ingest/documents",
            headers={"X-Conversation-ID": "conv-007"},
            json={
                "paths": [str(src)],
                "source_type": "upload",
            },
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    payload = response.json()
    assert payload["doc_ids"] == ["doc-upload-1"]
    assert payload["workspace_copies"] == ["sales.csv"]
    assert (session_workspace / "sales.csv").read_text() == "region,revenue\nNA,100\n"
    assert captured == {
        "settings": settings,
        "stores": bot.ctx.stores,
        "paths": [str(src)],
        "source_type": "upload",
        "tenant_id": "local-dev",
        "collection_id": "default",
    }


@pytest.mark.asyncio
async def test_upload_files_copies_into_canonical_session_workspace(tmp_path, monkeypatch):
    client, _, settings = _make_client(tmp_path)
    captured: dict[str, object] = {}

    def fake_ingest_paths(settings_arg, stores_arg, paths, *, source_type, tenant_id, collection_id=None, **kwargs):
        del settings_arg, stores_arg
        captured["paths"] = [str(path) for path in paths]
        captured["source_type"] = source_type
        captured["tenant_id"] = tenant_id
        captured["collection_id"] = collection_id
        return ["doc-upload-2"]

    monkeypatch.setattr(api_main, "ingest_paths", fake_ingest_paths)

    try:
        response = await client.post(
            "/v1/upload",
            headers={"X-Conversation-ID": "conv-008"},
            files={"files": ("sales.csv", b"region,revenue\nNA,100\n", "text/csv")},
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    payload = response.json()
    assert payload["workspace_copies"] == ["sales.csv"]
    workspace_file = settings.workspace_dir / filesystem_key("local-dev:local-cli:conv-008") / "sales.csv"
    assert workspace_file.read_text() == "region,revenue\nNA,100\n"
    assert captured["source_type"] == "upload"
    assert captured["tenant_id"] == "local-dev"


@pytest.mark.asyncio
async def test_upload_files_preview_returns_metadata_without_ingesting(tmp_path, monkeypatch):
    client, _, _ = _make_client(tmp_path)

    def fail_ingest_paths(*args, **kwargs):
        raise AssertionError("upload preview should not call ingest_paths")

    monkeypatch.setattr(api_main, "ingest_paths", fail_ingest_paths)

    try:
        response = await client.post(
            "/v1/upload?index_preview=true",
            files={"files": ("preview.md", b"# Preview\n\nREQ-001 The gateway shall authenticate users.", "text/markdown")},
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "upload.preview"
    assert payload["preview"] is True
    assert payload["ingested_count"] == 0
    assert payload["files"][0]["outcome"] == "previewed"
    assert payload["metadata_summary"]["document_count"] == 1


@pytest.mark.asyncio
async def test_upload_files_skips_already_seen_openwebui_source_ids(tmp_path, monkeypatch):
    client, bot, settings = _make_client(tmp_path)
    session_id = "local-dev:local-cli:conv-009"
    bot.kernel.transcript_store.states[session_id] = SessionState(
        tenant_id="local-dev",
        user_id="local-cli",
        conversation_id="conv-009",
        session_id=session_id,
        uploaded_doc_ids=["existing-upload-doc"],
        metadata={"source_upload_ids": ["owui-file-1"]},
    )
    ingest_calls: list[list[str]] = []

    def fake_ingest_paths(settings_arg, stores_arg, paths, *, source_type, tenant_id, collection_id=None, **kwargs):
        del settings_arg, stores_arg, source_type, tenant_id, collection_id
        ingest_calls.append([str(path) for path in paths])
        return ["doc-upload-3"]

    monkeypatch.setattr(api_main, "ingest_paths", fake_ingest_paths)

    try:
        response = await client.post(
            "/v1/upload",
            headers={"X-Conversation-ID": "conv-009"},
            data={"source_ids": ["owui-file-1", "owui-file-2"]},
            files=[
                ("files", ("already.csv", b"a,b\n1,2\n", "text/csv")),
                ("files", ("fresh.csv", b"c,d\n3,4\n", "text/csv")),
            ],
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    payload = response.json()
    assert payload["skipped_source_ids"] == ["owui-file-1"]
    assert payload["workspace_copies"] == ["fresh.csv"]
    assert payload["doc_ids"] == ["doc-upload-3"]
    assert payload["active_uploaded_doc_ids"] == ["existing-upload-doc", "doc-upload-3"]
    assert payload["upload_manifest"]["active_uploaded_doc_ids"] == ["existing-upload-doc", "doc-upload-3"]
    assert payload["document_source_policy"] == "agent_repository_only"
    assert ingest_calls and ingest_calls[0][0].endswith("fresh.csv")
    state = bot.kernel.transcript_store.states[session_id]
    assert state.metadata["source_upload_ids"] == ["owui-file-1", "owui-file-2"]
    assert state.metadata["uploaded_doc_ids"] == ["existing-upload-doc", "doc-upload-3"]
    workspace_file = settings.workspace_dir / filesystem_key(session_id) / "fresh.csv"
    assert workspace_file.exists()


@pytest.mark.asyncio
async def test_upload_files_uses_repository_runtime_when_chat_registry_is_invalid(tmp_path, monkeypatch):
    settings = _make_settings(tmp_path)
    bot = DummyBot()
    runtime = api_main.Runtime(
        settings=settings,
        bot=bot,
        diagnostics={
            "error_code": "runtime_registry_invalid",
            "missing_tools": [{"agent": "coordinator", "tool": "list_worker_requests"}],
            "affected_agents": ["coordinator"],
            "upload_runtime_mode": "repository_ingest_only",
        },
    )

    def fail_runtime():
        raise ValueError(
            "Invalid next-runtime agent configuration:\n"
            "- agent 'coordinator' references unknown tool 'list_worker_requests'"
        )

    def fake_upload_runtime():
        try:
            fail_runtime()
        except ValueError as exc:
            assert "list_worker_requests" in str(exc)
        return runtime

    api_main.app.dependency_overrides[api_main.get_upload_runtime_or_503] = fake_upload_runtime
    api_main.app.dependency_overrides[api_main.get_settings] = lambda: settings

    def fake_ingest_paths(settings_arg, stores_arg, paths, *, source_type, tenant_id, collection_id=None, **kwargs):
        del settings_arg, stores_arg, paths, source_type, tenant_id, collection_id
        return ["doc-upload-degraded"]

    monkeypatch.setattr(api_main, "ingest_paths", fake_ingest_paths)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    try:
        response = await client.post(
            "/v1/upload",
            headers={"X-Conversation-ID": "conv-degraded"},
            files={"files": ("requirements.txt", b"shall do x\n", "text/plain")},
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    payload = response.json()
    assert payload["active_uploaded_doc_ids"] == ["doc-upload-degraded"]
    assert payload["runtime_diagnostics"]["error_code"] == "runtime_registry_invalid"
    assert payload["upload_manifest"]["runtime_diagnostics"]["upload_runtime_mode"] == "repository_ingest_only"
    assert payload["upload_manifest"]["warnings"]


@pytest.mark.asyncio
async def test_download_file_serves_registered_workspace_artifact(tmp_path):
    client, bot, settings = _make_client(tmp_path)
    workspace_root = settings.workspace_dir / filesystem_key("local-dev:local-cli:download-conv")
    workspace_root.mkdir(parents=True, exist_ok=True)
    target = workspace_root / "analysis.csv"
    target.write_text("a,b\n1,2\n")
    state = SessionState(
        tenant_id="local-dev",
        user_id="local-cli",
        conversation_id="download-conv",
        session_id="local-dev:local-cli:download-conv",
        workspace_root=str(workspace_root),
        metadata={
            "downloads": {
                "dl_abc": {
                    "download_id": "dl_abc",
                    "artifact_ref": "download://dl_abc",
                    "filename": "analysis.csv",
                    "label": "analysis.csv",
                    "download_url": "/v1/files/dl_abc?conversation_id=download-conv",
                    "content_type": "text/csv",
                    "size_bytes": target.stat().st_size,
                    "session_id": "local-dev:local-cli:download-conv",
                    "conversation_id": "download-conv",
                }
            }
        },
    )
    bot.kernel.transcript_store.states[state.session_id] = state

    try:
        response = await client.get("/v1/files/dl_abc?conversation_id=download-conv")
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert response.text == "a,b\n1,2\n"


@pytest.mark.asyncio
async def test_download_file_accepts_valid_signed_url_without_bearer_token(tmp_path):
    client, bot, settings = _make_client(tmp_path)
    settings.gateway_shared_bearer_token = "shared-secret"
    settings.download_url_secret = "download-secret"
    workspace_root = settings.workspace_dir / filesystem_key("tenant-a:user-a:download-conv")
    workspace_root.mkdir(parents=True, exist_ok=True)
    target = workspace_root / "analysis.csv"
    target.write_text("a,b\n1,2\n")
    state = SessionState(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id="download-conv",
        session_id="tenant-a:user-a:download-conv",
        workspace_root=str(workspace_root),
        metadata={
            "downloads": {
                "dl_signed": {
                    "download_id": "dl_signed",
                    "artifact_ref": "download://dl_signed",
                    "filename": "analysis.csv",
                    "label": "analysis.csv",
                    "download_url": "/v1/files/dl_signed",
                    "content_type": "text/csv",
                    "size_bytes": target.stat().st_size,
                    "session_id": "tenant-a:user-a:download-conv",
                    "conversation_id": "download-conv",
                }
            }
        },
    )
    bot.kernel.transcript_store.states[state.session_id] = state
    signed_url = api_main.build_signed_download_url(
        download_id="dl_signed",
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id="download-conv",
        secret=settings.download_url_secret,
        ttl_seconds=settings.download_url_ttl_seconds,
        path="/v1/files/dl_signed",
        now=1_700_000_000,
    )

    original_time = api_main.time.time
    api_main.time.time = lambda: 1_700_000_100
    try:
        response = await client.get(signed_url)
    finally:
        api_main.time.time = original_time
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert response.text == "a,b\n1,2\n"


@pytest.mark.asyncio
async def test_document_source_file_serves_repository_source_with_bearer(tmp_path):
    client, bot, settings = _make_client(tmp_path)
    settings.gateway_shared_bearer_token = "shared-secret"
    target = settings.uploads_dir / "source.txt"
    target.write_text("source body\n", encoding="utf-8")
    bot.ctx = SimpleNamespace(
        stores=SimpleNamespace(
            doc_store=_SourceDocStore(
                [
                    SimpleNamespace(
                        doc_id="DOC-1",
                        tenant_id="local-dev",
                        collection_id="default",
                        title="source.txt",
                        source_type="upload",
                        source_path=str(target),
                        source_display_path="source.txt",
                        source_metadata={"mime_type": "text/plain", "original_filename": "source.txt"},
                    )
                ]
            )
        )
    )

    try:
        response = await client.get(
            "/v1/documents/DOC-1/source",
            headers={"Authorization": "Bearer shared-secret", "X-Conversation-ID": "source-conv"},
        )
        inline_response = await client.get(
            "/v1/documents/DOC-1/source?disposition=inline",
            headers={"Authorization": "Bearer shared-secret", "X-Conversation-ID": "source-conv"},
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert response.text == "source body\n"
    assert response.headers["content-type"].startswith("text/plain")
    assert response.headers["content-disposition"].startswith("attachment;")
    assert inline_response.status_code == 200
    assert inline_response.text == "source body\n"
    assert inline_response.headers["content-disposition"].startswith("inline;")


@pytest.mark.asyncio
async def test_document_source_file_streams_remote_blob_source(tmp_path, monkeypatch):
    client, bot, settings = _make_client(tmp_path)
    settings.gateway_shared_bearer_token = "shared-secret"
    bot.ctx = SimpleNamespace(
        stores=SimpleNamespace(
            doc_store=_SourceDocStore(
                [
                    SimpleNamespace(
                        doc_id="DOC-REMOTE",
                        tenant_id="local-dev",
                        collection_id="default",
                        title="remote.txt",
                        source_type="upload",
                        source_path="s3://agentic-uploads/uploads/remote.txt",
                        source_display_path="remote.txt",
                        source_metadata={
                            "mime_type": "text/plain",
                            "original_filename": "remote.txt",
                            "blob_ref": {
                                "backend": "s3",
                                "uri": "s3://agentic-uploads/uploads/remote.txt",
                                "bucket": "agentic-uploads",
                                "key": "uploads/remote.txt",
                                "content_type": "text/plain",
                            },
                        },
                    )
                ]
            )
        )
    )

    class _RemoteBlobStore:
        def exists(self, ref):
            return ref.key == "uploads/remote.txt"

        def iter_bytes(self, ref):
            del ref
            yield b"remote source\n"

    monkeypatch.setattr(api_main, "build_blob_store", lambda settings_arg: _RemoteBlobStore())

    try:
        response = await client.get(
            "/v1/documents/DOC-REMOTE/source",
            headers={"Authorization": "Bearer shared-secret", "X-Conversation-ID": "source-conv"},
        )
        inline_response = await client.get(
            "/v1/documents/DOC-REMOTE/source?disposition=inline",
            headers={"Authorization": "Bearer shared-secret", "X-Conversation-ID": "source-conv"},
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert response.text == "remote source\n"
    assert response.headers["content-type"].startswith("text/plain")
    assert response.headers["content-disposition"].startswith("attachment;")
    assert inline_response.status_code == 200
    assert inline_response.text == "remote source\n"
    assert inline_response.headers["content-disposition"].startswith("inline;")


@pytest.mark.asyncio
async def test_document_source_file_accepts_valid_signed_url_without_bearer(tmp_path):
    client, bot, settings = _make_client(tmp_path)
    settings.gateway_shared_bearer_token = "shared-secret"
    settings.download_url_secret = "download-secret"
    target = settings.uploads_dir / "signed-source.txt"
    target.write_text("signed body\n", encoding="utf-8")
    bot.ctx = SimpleNamespace(
        stores=SimpleNamespace(
            doc_store=_SourceDocStore(
                [
                    SimpleNamespace(
                        doc_id="DOC-SIGNED",
                        tenant_id="tenant-a",
                        collection_id="default",
                        title="signed-source.txt",
                        source_type="upload",
                        source_path=str(target),
                        source_display_path="signed-source.txt",
                        source_metadata={"mime_type": "text/plain"},
                    )
                ]
            )
        )
    )
    signed_url = api_main.build_signed_download_url(
        download_id="DOC-SIGNED",
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id="source-conv",
        secret=settings.download_url_secret,
        ttl_seconds=settings.download_url_ttl_seconds,
        path="/v1/documents/DOC-SIGNED/source?disposition=inline",
    )

    try:
        response = await client.get(signed_url)
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 200
    assert response.text == "signed body\n"
    assert response.headers["content-disposition"].startswith("inline;")


@pytest.mark.asyncio
async def test_document_source_file_rejects_path_outside_source_roots(tmp_path):
    client, bot, settings = _make_client(tmp_path)
    settings.gateway_shared_bearer_token = "shared-secret"
    outside = tmp_path / "outside.txt"
    outside.write_text("nope\n", encoding="utf-8")
    bot.ctx = SimpleNamespace(
        stores=SimpleNamespace(
            doc_store=_SourceDocStore(
                [
                    SimpleNamespace(
                        doc_id="DOC-OUTSIDE",
                        tenant_id="local-dev",
                        collection_id="default",
                        title="outside.txt",
                        source_type="upload",
                        source_path=str(outside),
                        source_display_path="outside.txt",
                        source_metadata={"mime_type": "text/plain"},
                    )
                ]
            )
        )
    )

    try:
        response = await client.get(
            "/v1/documents/DOC-OUTSIDE/source",
            headers={"Authorization": "Bearer shared-secret", "X-Conversation-ID": "source-conv"},
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_graph_endpoints_list_and_query_indexes(tmp_path, monkeypatch: pytest.MonkeyPatch):
    client, _, settings = _make_client(tmp_path)
    settings.gateway_shared_bearer_token = "shared-secret"

    class _FakeGraphService:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def list_indexes(self, *, collection_id="", limit=20):
            del limit
            return [
                {
                    "graph_id": "release_graph",
                    "display_name": "Release Graph",
                    "collection_id": collection_id or "default",
                }
            ]

        def query_across_graphs(self, query, *, collection_id="", graph_ids=None, methods=None, limit=8, top_k_graphs=3, doc_ids=None):
            del graph_ids, methods, limit, top_k_graphs, doc_ids
            return {
                "query": query,
                "graph_shortlist": [{"graph_id": "release_graph", "collection_id": collection_id or "default"}],
                "results": [{"graph_id": "release_graph", "doc_id": "DOC-1", "score": 0.91}],
            }

    monkeypatch.setattr(api_main, "GraphService", _FakeGraphService)

    try:
        list_response = await client.get(
            "/v1/graphs",
            headers={"Authorization": "Bearer shared-secret", "X-Collection-ID": "default"},
            params={"collection_id": "default"},
        )
        query_response = await client.post(
            "/v1/graphs/query",
            headers={"Authorization": "Bearer shared-secret"},
            json={"query": "Which release graph should I use?", "collection_id": "default"},
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert list_response.status_code == 200
    assert list_response.json()["graphs"][0]["graph_id"] == "release_graph"
    assert query_response.status_code == 200
    assert query_response.json()["graph_shortlist"][0]["graph_id"] == "release_graph"
    assert query_response.json()["results"][0]["doc_id"] == "DOC-1"


@pytest.mark.asyncio
async def test_graph_index_endpoint_is_blocked_for_public_chat_mutation(tmp_path, monkeypatch: pytest.MonkeyPatch):
    client, _, settings = _make_client(tmp_path)
    settings.gateway_shared_bearer_token = "shared-secret"
    del monkeypatch

    try:
        response = await client.post(
            "/v1/graphs/index",
            headers={"Authorization": "Bearer shared-secret"},
            json={"graph_id": "empty-graph", "collection_id": "default"},
        )
    finally:
        await client.aclose()
        _clear_overrides()

    assert response.status_code == 403
    assert response.json()["detail"] == "Graph creation and refresh are admin-managed in the control panel."
