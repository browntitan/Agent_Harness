from __future__ import annotations

from langchain_core.documents import Document

from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.jobs import JobRecord, TaskNotification, TeamMailboxChannel, TeamMailboxMessage
from agentic_chatbot_next.contracts.messages import RuntimeMessage, SessionState
from agentic_chatbot_next.contracts.rag import Citation, RagContract, RetrievalSummary
from agentic_chatbot_next.contracts.tools import ToolDefinition
from agentic_chatbot_next.rag.citations import build_citations
from agentic_chatbot_next.rag.engine import render_rag_contract
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
        },
    )

    assert resumed.answer_contract.kind == "requirements_extraction"
    assert resumed.requested_scope["document_names"] == ["SPEC_A.docx"]
    assert "SPEC_A.docx" in resumed.effective_user_text
