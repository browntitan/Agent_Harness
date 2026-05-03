from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document

from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.jobs import JobRecord, TaskNotification, TeamMailboxChannel, TeamMailboxMessage
from agentic_chatbot_next.contracts.messages import RuntimeMessage, SessionState
from agentic_chatbot_next.contracts.rag import Citation, RagContract, RetrievalSummary
from agentic_chatbot_next.contracts.tools import ToolDefinition
from agentic_chatbot_next.rag.citations import build_citations
from agentic_chatbot_next.rag.engine import render_rag_contract
from agentic_chatbot_next.rag.source_links import build_document_source_url
from agentic_chatbot_next.runtime.turn_contracts import resolve_turn_intent


def test_contract_round_trips_preserve_fields() -> None:
    message = RuntimeMessage(
        role="assistant",
        content="hello",
        name="helper",
        tool_call_id="tool-1",
        artifact_refs=["artifact://one"],
        metadata={"agent": "general"},
    )
    restored_message = RuntimeMessage.from_dict(message.to_dict())
    assert restored_message == message

    session = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conversation",
        request_id="request",
        session_id="session",
        messages=[message],
        uploaded_doc_ids=["doc-1"],
        scratchpad={"a": "b"},
        workspace_root="/tmp/workspace",
        active_agent="general",
        metadata={"runtime_kind": "next"},
    )
    notification = TaskNotification(job_id="job-1", status="completed", summary="done")
    session.pending_notifications.append(notification)
    restored_session = SessionState.from_dict(session.to_dict())
    assert restored_session.to_dict() == session.to_dict()

    agent = AgentDefinition(
        name="general",
        mode="react",
        description="desc",
        prompt_file="general_agent.md",
        skill_scope="general",
        allowed_tools=["calculator"],
        allowed_worker_agents=["utility"],
        preload_skill_packs=["base"],
        memory_scopes=["conversation", "user"],
        max_steps=10,
        max_tool_calls=12,
        allow_background_jobs=True,
        metadata={"role_kind": "top_level"},
    )
    assert AgentDefinition.from_dict(agent.to_dict()) == agent

    tool = ToolDefinition(
        name="calculator",
        group="utility",
        description="math",
        args_schema={"type": "object"},
        when_to_use="Use for arithmetic.",
        avoid_when="Avoid for grounded retrieval.",
        output_description="Returns a numeric result.",
        examples=["calculator(expression='1+1')"],
        keywords=["math"],
        read_only=True,
        background_safe=True,
        concurrency_key="utility",
        serializer="default",
        should_defer=True,
        search_hint="Use for deterministic math.",
        defer_reason="Example deferred metadata.",
        defer_priority=12,
        eager_for_agents=["utility"],
        metadata={"kind": "tool"},
    )
    assert ToolDefinition.from_dict(tool.to_dict()) == tool

    channel = TeamMailboxChannel(
        session_id="session",
        name="research",
        purpose="coordinate workers",
        created_by_job_id="job-parent",
        member_agents=["general", "utility"],
        member_job_ids=["job-parent"],
        metadata={"source": "test"},
    )
    assert TeamMailboxChannel.from_dict(channel.to_dict()) == channel

    team_message = TeamMailboxMessage(
        channel_id=channel.channel_id,
        session_id="session",
        source_agent="general",
        source_job_id="job-parent",
        target_agents=["utility"],
        target_job_ids=["job-worker"],
        message_type="question_request",
        status="open",
        subject="Need input",
        content="Which files matter?",
        payload={"options": ["a", "b"]},
        requires_response=True,
        thread_id="thread-1",
    )
    assert TeamMailboxMessage.from_dict(team_message.to_dict()) == team_message

    job = JobRecord(
        job_id="job-1",
        session_id="session",
        agent_name="utility",
        status="completed",
        prompt="do work",
        description="worker",
        artifact_dir="/tmp/artifacts",
        output_path="/tmp/artifacts/output.md",
        result_path="/tmp/artifacts/result.json",
        result_summary="done",
        session_state={"messages": 2},
        metadata={"background": True},
    )
    assert JobRecord.from_dict(job.to_dict()) == job

    rag = RagContract(
        answer="answer",
        citations=[
            Citation(
                citation_id="c1",
                doc_id="doc-1",
                title="Doc",
                source_type="pdf",
                location="p.1",
                snippet="snippet",
                collection_id="default",
                url="/v1/documents/doc-1/source?conversation_id=conversation",
                source_path="/kb/Doc.pdf",
            )
        ],
        used_citation_ids=["c1"],
        confidence=0.7,
        retrieval_summary=RetrievalSummary(
            query_used="query",
            steps=2,
            tool_calls_used=3,
            tool_call_log=["search"],
            citations_found=1,
            search_mode="deep",
            rounds=2,
            strategies_used=["hybrid", "keyword", "window"],
            candidate_counts={"unique_chunks": 5, "selected_docs": 2},
            parallel_workers_used=False,
            decomposition={"canonical_entities": [{"canonical_name": "Vendor Acme"}]},
            claim_ledger={"claims": [{"claim_id": "claim_1"}], "supported_claim_ids": ["claim_1"]},
            verified_hops=["Vendor Acme -> Finance Approval"],
            retrieval_verification={"status": "pass", "issues": []},
        ),
        followups=["next"],
        warnings=["warn"],
    )
    assert RagContract.from_dict(rag.to_dict()).to_dict() == rag.to_dict()


def test_rag_citations_preserve_and_render_collection_id() -> None:
    doc = Document(
        page_content="Rate limit policy text.",
        metadata={
            "chunk_id": "doc-rate#chunk0001",
            "doc_id": "doc-rate",
            "title": "api_rate_limits.md",
            "source_type": "kb",
            "collection_id": "default",
            "source_url": "/v1/documents/doc-rate/source?conversation_id=conv",
            "source_path": "/kb/api_rate_limits.md",
            "chunk_index": 1,
        },
    )

    citations = build_citations([doc])
    assert citations[0].collection_id == "default"

    rendered = render_rag_contract(
        RagContract(
            answer="answer",
            citations=citations,
            used_citation_ids=["doc-rate#chunk0001"],
        )
    )

    assert "KB Collection: default" in rendered
    assert "[api_rate_limits.md](/v1/documents/doc-rate/source?conversation_id=conv)" in rendered


def test_rag_renderer_humanizes_inline_and_block_citation_ids() -> None:
    citation_id = "COLLECTION_UPLOAD_935ee5d6d9#chunk0001"
    rendered = render_rag_contract(
        RagContract(
            answer=f"The approved change drove the cost delta ({citation_id}).",
            citations=[
                Citation(
                    citation_id=citation_id,
                    doc_id="doc-asterion",
                    title="asterion_issue_digest_draft.txt",
                    source_type="upload",
                    location="chunk 1",
                    snippet="Cost commentary text.",
                    collection_id="defense-rag-test",
                    url="http://localhost:18000/v1/documents/doc-asterion/source?disposition=inline",
                    source_path="/uploads/asterion_issue_digest_draft.txt",
                )
            ],
            used_citation_ids=[citation_id],
        )
    )

    assert citation_id not in rendered
    assert "([asterion_issue_digest_draft.txt](http://localhost:18000/v1/documents/doc-asterion/source?disposition=inline))" in rendered
    assert "- [asterion_issue_digest_draft.txt](http://localhost:18000/v1/documents/doc-asterion/source?disposition=inline)" in rendered


def test_document_source_url_uses_gateway_public_base_and_inline_disposition() -> None:
    settings = SimpleNamespace(
        gateway_public_base_url="https://agent.example.com",
        download_url_secret="download-secret",
        download_url_ttl_seconds=900,
        default_tenant_id="tenant-default",
        default_user_id="user-default",
        default_conversation_id="conversation-default",
    )
    session = SimpleNamespace(tenant_id="tenant-a", user_id="user-a", conversation_id="conversation-a")

    url = build_document_source_url(settings, session, "doc-asterion")

    assert url.startswith("https://agent.example.com/v1/documents/doc-asterion/source?")
    assert "tenant_id=tenant-a" in url
    assert "user_id=user-a" in url
    assert "conversation_id=conversation-a" in url
    assert "sig=" in url
    assert "disposition=inline" in url


def test_document_source_url_falls_back_to_public_agent_base(monkeypatch) -> None:
    monkeypatch.setenv("PUBLIC_AGENT_API_BASE_URL", "https://public-agent.example.com")
    settings = SimpleNamespace(
        gateway_public_base_url="",
        download_url_secret="",
        download_url_ttl_seconds=900,
        default_tenant_id="tenant-default",
        default_user_id="user-default",
        default_conversation_id="conversation-default",
    )
    session = SimpleNamespace(conversation_id="conversation-a")

    url = build_document_source_url(settings, session, "doc-asterion")

    assert (
        url
        == "https://public-agent.example.com/v1/documents/doc-asterion/source"
        "?conversation_id=conversation-a&disposition=inline"
    )


def test_runtime_message_langchain_conversion_preserves_identity_fields() -> None:
    message = RuntimeMessage(
        role="assistant",
        content="hello",
        artifact_refs=["artifact://one"],
        metadata={"agent": "general"},
    )
    restored = RuntimeMessage.from_langchain(message.to_langchain())
    assert restored.message_id == message.message_id
    assert restored.created_at == message.created_at
    assert restored.artifact_refs == message.artifact_refs
    assert restored.metadata["agent"] == "general"


def test_requirements_extraction_turn_contract_uses_upload_scope_and_workflow() -> None:
    intent = resolve_turn_intent(
        "extract all requirements/ shall statements from the uploaded document",
        {"uploaded_doc_ids": ["UPLOAD_0dc122b0e8"], "upload_collection_id": "owui-chat-1"},
    )

    assert intent.answer_contract.kind == "requirements_extraction"
    assert intent.answer_contract.requires_authoritative_inventory is True
    assert intent.requested_scope["scope_kind"] == "uploads"
    assert intent.requested_scope["workflow"] == "requirements_extraction"


def test_mermaid_turn_contract_records_diagram_output_mode() -> None:
    intent = resolve_turn_intent(
        "Use the default KB, research how skills are indexed, then produce a Mermaid flowchart with citations below it.",
        {"kb_collection_id": "default"},
    )

    assert intent.answer_contract.kind == "grounded_synthesis"
    assert intent.answer_contract.final_output_mode == "grounded_mermaid_diagram"
    assert intent.presentation_preferences.diagram_policy == "require_mermaid"


def test_requirements_document_selection_clarification_resumes_original_request() -> None:
    original = resolve_turn_intent(
        "extract all shall statements from the uploaded document",
        {"uploaded_doc_ids": ["doc-a", "doc-b"], "upload_collection_id": "uploads"},
    )
    resumed = resolve_turn_intent(
        "first one",
        {
            "pending_clarification": {
                "reason": "requirements_document_selection",
                "options": ["SPEC_A.docx", "SPEC_B.docx"],
                "resolved_turn_intent": original.to_dict(),
            },
            "uploaded_doc_ids": ["doc-a", "doc-b"],
            "upload_collection_id": "uploads",
            "requirements_candidate_documents": [
                {"doc_id": "doc-a", "title": "SPEC_A.docx"},
                {"doc_id": "doc-b", "title": "SPEC_B.docx"},
            ],
        },
    )

    assert resumed.answer_contract.kind == "requirements_extraction"
    assert resumed.requested_scope["document_names"] == ["SPEC_A.docx"]
    assert resumed.requested_scope["document_ids"] == ["doc-a"]
    assert resumed.requested_scope["selected_doc_ids"] == ["doc-a"]
    assert "SPEC_A.docx" in resumed.effective_user_text
