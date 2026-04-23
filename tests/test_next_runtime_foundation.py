from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.agents.registry import AgentRegistry
from agentic_chatbot_next.persistence.postgres.graphs import GraphIndexRecord
from agentic_chatbot_next.session import ChatSession
from agentic_chatbot_next.runtime.context import RuntimePaths
from agentic_chatbot_next.runtime.clarification import parse_clarification_request
from agentic_chatbot_next.runtime.kernel import RuntimeKernel
from agentic_chatbot_next.runtime.query_loop import QueryLoop
from agentic_chatbot_next.runtime.turn_contracts import resolve_turn_intent
from agentic_chatbot_next.tools.base import ToolContext


def _settings(tmp_path: Path) -> SimpleNamespace:
    repo_root = Path(__file__).resolve().parents[1]
    return SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        workspace_dir=tmp_path / "workspaces",
        memory_dir=tmp_path / "memory",
        agents_dir=repo_root / "data" / "agents",
        llm_provider="ollama",
        judge_provider="ollama",
        runtime_events_enabled=True,
        max_worker_concurrency=2,
        session_hydrate_window_messages=40,
        session_transcript_page_size=100,
    )


def test_parse_clarification_request_accepts_nested_xml_fields() -> None:
    text = (
        "<clarification_request>\n"
        "  <question>Could you let me know what format you’d like?</question>\n"
        "  <reason>Choosing the output format will guide the graph summary.</reason>\n"
        "  <options>[\"Textual synthesis\", \"Diagram\", \"Table\", \"Mixed\"]</options>\n"
        "</clarification_request>"
    )

    visible, request = parse_clarification_request(text, source_agent="graph_manager")

    assert visible == "Could you let me know what format you’d like?"
    assert request is not None
    assert request.reason == "answer_format_selection"
    assert request.options == ("Textual synthesis", "Diagram", "Table", "Mixed")
    assert request.source_agent == "graph_manager"


def test_stub_kernel_persists_turn_state_transcript_and_events(tmp_path: Path) -> None:
    session = ChatSession(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conversation",
    )
    providers = SimpleNamespace(
        chat=FakeListChatModel(responses=["hello from next runtime"]),
        judge=FakeListChatModel(responses=["unused"]),
    )
    kernel = RuntimeKernel(_settings(tmp_path), providers=providers, stores=None)
    result = kernel.process_turn(session, user_text="hello foundation")

    assert result == "hello from next runtime"
    paths = RuntimePaths.from_settings(_settings(tmp_path))
    session_dir = paths.session_dir(session.session_id)
    state = json.loads((session_dir / "state.json").read_text(encoding="utf-8"))
    transcript_rows = [
        json.loads(line)
        for line in (session_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_rows = [
        json.loads(line)
        for line in (session_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert state["active_agent"] == "general"
    assert state["workspace_root"] == str(paths.workspace_dir(session.session_id))
    assert [row["message"]["role"] for row in transcript_rows] == ["user", "assistant"]
    assert {row["event_type"] for row in event_rows} >= {
        "turn_accepted",
        "agent_run_started",
        "agent_run_completed",
        "turn_completed",
    }
    assert [message.type for message in session.messages] == ["system", "human", "ai"]


class ExplodingQueryLoop(QueryLoop):
    def run(self, agent, session_state, *, user_text: str, **kwargs):  # type: ignore[override]
        raise RuntimeError(f"boom:{agent.name}:{user_text}")


def test_kernel_persists_user_turn_before_executor_failure(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    registry = AgentRegistry(settings.agents_dir)
    kernel = RuntimeKernel(settings, registry=registry, query_loop=ExplodingQueryLoop())
    session = ChatSession(
        tenant_id="tenant",
        user_id="user",
        conversation_id="failure-case",
    )

    with pytest.raises(RuntimeError, match="boom:general:fail now"):
        kernel.process_turn(session, user_text="fail now")

    paths = RuntimePaths.from_settings(settings)
    session_dir = paths.session_dir(session.session_id)
    state = json.loads((session_dir / "state.json").read_text(encoding="utf-8"))
    transcript_rows = [
        json.loads(line)
        for line in (session_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_rows = [
        json.loads(line)
        for line in (session_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(state["messages"]) == 1
    assert transcript_rows[0]["message"]["content"] == "fail now"
    assert "turn_failed" in {row["event_type"] for row in event_rows}


def test_team_mailbox_tool_listing_only_summarizes_visible_channels(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.team_mailbox_enabled = True
    kernel = RuntimeKernel(settings, providers=None, stores=None)
    session = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="team-visible",
        session_id="tenant:user:team-visible",
    )
    visible = kernel.job_manager.create_team_channel(
        session_id=session.session_id,
        name="visible",
        member_agents=["utility"],
    )
    hidden = kernel.job_manager.create_team_channel(
        session_id=session.session_id,
        name="hidden",
        member_agents=["general"],
    )
    kernel.job_manager.post_team_message(
        session_id=session.session_id,
        channel_id=visible.channel_id,
        content="visible handoff",
        source_agent="general",
        target_agents=["utility"],
        message_type="handoff",
    )
    kernel.job_manager.post_team_message(
        session_id=session.session_id,
        channel_id=hidden.channel_id,
        content="hidden question",
        source_agent="general",
        target_agents=["general"],
        message_type="question_request",
    )
    context = ToolContext(
        settings=settings,
        providers=None,
        stores=None,
        session=session,
        paths=RuntimePaths.from_settings(settings),
        transcript_store=kernel.transcript_store,
        job_manager=kernel.job_manager,
        kernel=kernel,
        active_agent="utility",
        active_definition=AgentDefinition(
            name="utility",
            mode="react",
            allowed_worker_agents=["general"],
            metadata={"role_kind": "worker"},
        ),
    )

    result = kernel.list_team_messages_from_tool(context)

    assert [item["content"] for item in result["data"]] == ["visible handoff"]
    assert result["summary"]["open_message_count"] == 1
    assert result["summary"]["pending_question_count"] == 0


def test_kernel_compacts_large_incoming_history_after_seeding_transcript(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.session_hydrate_window_messages = 3
    settings.session_transcript_page_size = 2
    kernel = RuntimeKernel(
        settings,
        providers=SimpleNamespace(
            chat=FakeListChatModel(responses=["unused"]),
            judge=FakeListChatModel(responses=["unused"]),
        ),
        stores=None,
    )
    session = ChatSession(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conversation",
        messages=[
            HumanMessage(content="message 0"),
            AIMessage(content="message 1"),
            HumanMessage(content="message 2"),
            AIMessage(content="message 3"),
            HumanMessage(content="message 4"),
            AIMessage(content="message 5"),
        ],
    )

    state = kernel.hydrate_session_state(session)
    kernel._persist_state(state)

    paths = RuntimePaths.from_settings(settings)
    session_dir = paths.session_dir(session.session_id)
    transcript_rows = [
        json.loads(line)
        for line in (session_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    persisted = json.loads((session_dir / "state.json").read_text(encoding="utf-8"))

    assert [message.content for message in state.messages] == ["message 3", "message 4", "message 5"]
    assert state.metadata["history_total_messages"] == 6
    assert state.metadata["history_stored_window_messages"] == 3
    assert state.metadata["has_earlier_history"] is True
    assert [row["message"]["content"] for row in transcript_rows] == [
        "message 0",
        "message 1",
        "message 2",
        "message 3",
        "message 4",
        "message 5",
    ]
    assert [item["content"] for item in persisted["messages"]] == ["message 3", "message 4", "message 5"]


def test_kernel_serializes_parallel_worker_batches_for_local_ollama(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    kernel = RuntimeKernel(
        settings,
        providers=SimpleNamespace(
            chat=FakeListChatModel(responses=["unused"]),
            judge=FakeListChatModel(responses=["unused"]),
        ),
        stores=None,
    )

    should_run_parallel = kernel._should_run_task_batch_in_parallel(
        batch=[
            {"mode": "parallel", "executor": "general"},
            {"mode": "parallel", "executor": "memory_maintainer"},
        ],
        real_jobs=[SimpleNamespace(job_id="job_1"), SimpleNamespace(job_id="job_2")],
    )

    assert should_run_parallel is False


def test_kernel_keeps_parallel_worker_batches_for_non_ollama_providers(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.llm_provider = "azure"
    settings.judge_provider = "azure"
    kernel = RuntimeKernel(
        settings,
        providers=SimpleNamespace(
            chat=FakeListChatModel(responses=["unused"]),
            judge=FakeListChatModel(responses=["unused"]),
        ),
        stores=None,
    )

    should_run_parallel = kernel._should_run_task_batch_in_parallel(
        batch=[
            {"mode": "parallel", "executor": "general"},
            {"mode": "parallel", "executor": "memory_maintainer"},
        ],
        real_jobs=[SimpleNamespace(job_id="job_1"), SimpleNamespace(job_id="job_2")],
    )

    assert should_run_parallel is True


def test_kernel_blocks_parallel_worker_batches_for_agents_that_disallow_background_jobs(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.llm_provider = "azure"
    settings.judge_provider = "azure"
    kernel = RuntimeKernel(
        settings,
        providers=SimpleNamespace(
            chat=FakeListChatModel(responses=["unused"]),
            judge=FakeListChatModel(responses=["unused"]),
        ),
        stores=None,
    )

    should_run_parallel = kernel._should_run_task_batch_in_parallel(
        batch=[
            {"mode": "parallel", "executor": "planner"},
            {"mode": "parallel", "executor": "verifier"},
        ],
        real_jobs=[SimpleNamespace(job_id="job_1"), SimpleNamespace(job_id="job_2")],
    )

    assert should_run_parallel is False


def test_kernel_serializes_parallel_worker_batches_when_provider_objects_are_ollama(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.llm_provider = ""
    settings.judge_provider = ""

    class FakeOllamaChat:
        pass

    FakeOllamaChat.__module__ = "langchain_ollama.chat_models"

    kernel = RuntimeKernel(
        settings,
        providers=SimpleNamespace(
            chat=FakeOllamaChat(),
            judge=FakeOllamaChat(),
        ),
        stores=None,
    )

    should_run_parallel = kernel._should_run_task_batch_in_parallel(
        batch=[{"mode": "parallel"}, {"mode": "parallel"}],
        real_jobs=[SimpleNamespace(job_id="job_1"), SimpleNamespace(job_id="job_2")],
    )

    assert should_run_parallel is False


def test_query_loop_builds_skill_context_only_once() -> None:
    class FakeSkillRuntime:
        def build_prompt(self, agent):
            del agent
            return "Base prompt"

    loop = QueryLoop(settings=SimpleNamespace(clarification_sensitivity=50), skill_runtime=FakeSkillRuntime())
    agent = AgentDefinition(name="general", mode="react", prompt_file="general_agent.md")
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conversation")

    prompt = loop._build_system_prompt(agent, session, skill_context="Use citation hygiene.")

    assert prompt.count("## Skill Context") == 1
    assert prompt.count("Use citation hygiene.") == 1
    assert "## Clarification Policy" in prompt
    assert "Clarification sensitivity is 50/100 (balanced)." in prompt


def test_query_loop_includes_pending_clarification_context() -> None:
    class FakeSkillRuntime:
        def build_prompt(self, agent):
            del agent
            return "Base prompt"

    loop = QueryLoop(settings=SimpleNamespace(clarification_sensitivity=90), skill_runtime=FakeSkillRuntime())
    agent = AgentDefinition(name="general", mode="react", prompt_file="general_agent.md")
    session = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conversation",
        metadata={
            "pending_clarification": {
                "question": "Should I use uploaded files, the knowledge base, both, or neither?",
                "reason": "retrieval_scope_ambiguous",
                "options": ["uploaded files only", "knowledge base only", "both"],
            }
        },
    )

    prompt = loop._build_system_prompt(agent, session, skill_context="")

    assert "## Pending Clarification" in prompt
    assert "Should I use uploaded files, the knowledge base, both, or neither?" in prompt
    assert "uploaded files only, knowledge base only, both" in prompt


def test_query_loop_includes_resolved_turn_intent_context() -> None:
    class FakeSkillRuntime:
        def build_prompt(self, agent):
            del agent
            return "Base prompt"

    loop = QueryLoop(settings=SimpleNamespace(clarification_sensitivity=50), skill_runtime=FakeSkillRuntime())
    agent = AgentDefinition(name="general", mode="react", prompt_file="general_agent.md")
    session = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conversation",
        metadata={
            "resolved_turn_intent": resolve_turn_intent(
                "Across the whole default collection, identify the major subsystems in this codebase.",
                {"kb_collection_id": "default"},
            ).to_dict()
        },
    )

    prompt = loop._build_system_prompt(agent, session, skill_context="")

    assert "## Resolved Turn Intent" in prompt
    assert "answer_contract" in prompt
    assert "presentation_preferences" in prompt


def test_resolve_turn_intent_prefers_semantic_routing_collection_over_default_scope() -> None:
    intent = resolve_turn_intent(
        "What is the approved current CDR date for Asterion?",
        {
            "kb_collection_id": "default",
            "available_kb_collection_ids": ["default", "rfp-corpus"],
            "semantic_routing": {
                "route": "AGENT",
                "requires_external_evidence": True,
                "answer_origin": "retrieval",
                "requested_scope_kind": "knowledge_base",
                "requested_collection_id": "rfp-corpus",
                "confidence": 0.94,
                "reasoning": "named KB collection requires grounded retrieval",
            },
        },
    )

    assert intent.requested_scope["scope_kind"] == "knowledge_base"
    assert intent.requested_scope["collection_id"] == "rfp-corpus"
    assert intent.answer_contract.kind == "grounded_synthesis"
    assert intent.evidence_contract.collection_id == "rfp-corpus"
    assert intent.evidence_contract.grounding_required is True


def test_resolve_turn_intent_does_not_infer_kb_scope_from_default_collection_alone() -> None:
    intent = resolve_turn_intent(
        "Please explain what this assistant can help with.",
        {"kb_collection_id": "default"},
    )

    assert intent.requested_scope["scope_kind"] == "auto"
    assert intent.requested_scope["collection_id"] == ""


def test_resolve_turn_intent_keeps_graph_inventory_off_grounded_retrieval_path() -> None:
    intent = resolve_turn_intent(
        "what graphs do i have access to",
        {
            "semantic_routing": {
                "route": "AGENT",
                "requires_external_evidence": False,
                "answer_origin": "parametric",
                "requested_scope_kind": "graph_indexes",
                "confidence": 0.90,
                "reasoning": "graph inventory intent",
            }
        },
    )

    assert intent.requested_scope["scope_kind"] == "graph_indexes"
    assert intent.answer_contract.kind == "inventory"
    assert intent.answer_contract.requires_supporting_evidence is False
    assert intent.evidence_contract.source_scope == "graph_indexes"
    assert intent.evidence_contract.grounding_required is False
    assert intent.requested_scope["inventory_query_type"] == "graph_index_inventory"


def test_resolve_turn_intent_keeps_kb_inventory_off_grounded_retrieval_path() -> None:
    intent = resolve_turn_intent(
        "what knowledge bases do i have access to",
        {
            "available_kb_collection_ids": ["default", "rfp-corpus"],
            "semantic_routing": {
                "route": "AGENT",
                "requires_external_evidence": True,
                "answer_origin": "retrieval",
                "requested_scope_kind": "knowledge_base",
                "confidence": 0.90,
                "reasoning": "stale KB inventory evidence flag",
            },
        },
    )

    assert intent.requested_scope["scope_kind"] == "knowledge_base"
    assert intent.answer_contract.kind == "inventory"
    assert intent.answer_contract.requires_supporting_evidence is False
    assert intent.answer_contract.requires_authoritative_inventory is True
    assert intent.presentation_preferences.preferred_structure == "inventory"
    assert intent.evidence_contract.source_scope == "knowledge_base"
    assert intent.evidence_contract.grounding_required is False
    assert intent.requested_scope["inventory_query_type"] == "kb_collection_access_inventory"


def test_kernel_dispatches_authoritative_kb_access_inventory_before_query_loop(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    providers = SimpleNamespace(
        chat=FakeListChatModel(responses=["should not run"]),
        judge=FakeListChatModel(responses=["unused"]),
    )

    def list_documents(source_type="", tenant_id="tenant", collection_id=""):
        del tenant_id
        records = [
            SimpleNamespace(
                doc_id="doc-default",
                title="ARCHITECTURE.md",
                source_type="kb",
                source_path="/repo/docs/ARCHITECTURE.md",
                collection_id="default",
                file_type="md",
                doc_structure_type="general",
                num_chunks=12,
            ),
            SimpleNamespace(
                doc_id="doc-rfp",
                title="RFP Overview.docx",
                source_type="host_path",
                source_path="/repo/docs/RFP Overview.docx",
                collection_id="rfp-corpus",
                file_type="docx",
                doc_structure_type="general",
                num_chunks=9,
            ),
        ]
        return [
            record
            for record in records
            if (not source_type or record.source_type == source_type)
            and (not collection_id or record.collection_id == collection_id)
        ]

    stores = SimpleNamespace(
        doc_store=SimpleNamespace(list_documents=list_documents),
        graph_index_store=SimpleNamespace(
            list_indexes=lambda **kwargs: [
                GraphIndexRecord(
                    graph_id="rfp_corpus",
                    tenant_id="tenant",
                    collection_id="rfp-corpus",
                    display_name="RFP Corpus Graph",
                    status="ready",
                    query_ready=True,
                    domain_summary="Graph index for cross-document RFP entity and requirement analysis",
                    source_doc_ids=["doc-rfp"],
                )
            ]
        ),
    )
    kernel = RuntimeKernel(settings, providers=providers, stores=stores)
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="inventory-bypass")
    session.metadata = {"kb_collection_id": "default"}

    monkeypatch.setattr(
        "agentic_chatbot_next.rag.inventory.get_collection_readiness_status",
        lambda *args, **kwargs: SimpleNamespace(ready=True, maintenance_policy=""),
    )
    monkeypatch.setattr(
        kernel.query_loop,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("query loop should not run")),
    )

    text = kernel.process_agent_turn(session, user_text="what knowledge bases do i have access to")

    assert "Knowledge base collections available to this chat:" in text
    assert "default" in text
    assert "rfp-corpus" in text
    assert "Knowledge graphs available to this chat:" in text
    assert "RFP Corpus Graph (`rfp_corpus`)" in text

    paths = RuntimePaths.from_settings(settings)
    session_dir = paths.session_dir(session.session_id)
    state = json.loads((session_dir / "state.json").read_text(encoding="utf-8"))
    event_rows = [
        json.loads(line)
        for line in (session_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert state["messages"][-1]["metadata"]["provenance"] == "authoritative_inventory"
    assert state["messages"][-1]["metadata"]["inventory_query_type"] == "kb_collection_access_inventory"
    assert state["messages"][-1]["metadata"]["inventory_summary"]["collection_count"] == 2
    assert state["messages"][-1]["metadata"]["inventory_summary"]["graph_count"] == 1
    assert "inventory_payload" not in state["messages"][-1]["metadata"]
    assert "authoritative_inventory_dispatched" in {row["event_type"] for row in event_rows}


def test_kernel_process_agent_turn_serializes_datetime_collection_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _DocStore:
        def list_documents(self, *, source_type="", tenant_id="tenant", collection_id=""):
            del tenant_id
            if collection_id != "default":
                return []
            return [
                SimpleNamespace(
                    doc_id="doc-arch",
                    title="Architecture Overview.md",
                    source_type=source_type or "kb",
                    collection_id="default",
                    tenant_id="tenant",
                    num_chunks=12,
                    file_type="md",
                    doc_structure_type="general",
                    source_path="/repo/docs/ARCHITECTURE.md",
                )
            ]

        def list_collections(self, tenant_id="tenant"):
            del tenant_id
            return [
                {
                    "collection_id": "default",
                    "document_count": 1,
                    "latest_ingested_at": datetime(2026, 4, 20, 14, 5, 41, tzinfo=timezone.utc),
                    "source_type_counts": {"kb": 1},
                }
            ]

    settings = _settings(tmp_path)
    providers = SimpleNamespace(
        chat=FakeListChatModel(responses=["unused"]),
        judge=FakeListChatModel(responses=["unused"]),
    )
    stores = SimpleNamespace(
        doc_store=_DocStore(),
        graph_index_store=SimpleNamespace(list_indexes=lambda **kwargs: []),
    )
    kernel = RuntimeKernel(settings, providers=providers, stores=stores)
    session = ChatSession(tenant_id="tenant", user_id="user", conversation_id="inventory-datetime")
    session.metadata = {"kb_collection_id": "default"}

    monkeypatch.setattr(
        "agentic_chatbot_next.rag.inventory.get_collection_readiness_status",
        lambda *args, **kwargs: SimpleNamespace(ready=True, maintenance_policy=""),
    )
    monkeypatch.setattr(
        kernel.query_loop,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("query loop should not run")),
    )

    text = kernel.process_agent_turn(session, user_text="what knowledge bases do i have access to")

    assert "Knowledge base collections available to this chat:" in text
    paths = RuntimePaths.from_settings(settings)
    state = json.loads((paths.session_dir(session.session_id) / "state.json").read_text(encoding="utf-8"))
    assert state["messages"][-1]["metadata"]["inventory_summary"]["collection_ids"] == ["default"]
    assert state["messages"][-1]["metadata"]["inventory_summary"]["graph_count"] == 0


def test_query_loop_uses_worker_semantic_query_for_rag_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSkillRuntime:
        def build_prompt(self, agent):
            del agent
            return "Base prompt"

    settings = _settings(tmp_path)
    settings.clarification_sensitivity = 50
    settings.rag_top_k_vector = 4
    settings.rag_top_k_keyword = 4
    settings.rag_max_retries = 1
    settings.max_rag_agent_steps = 4
    loop = QueryLoop(settings=settings, stores=SimpleNamespace(), skill_runtime=FakeSkillRuntime())
    agent = AgentDefinition(name="rag_worker", mode="rag", prompt_file="rag_worker.md")
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conversation")
    captured: dict[str, str] = {}

    def fake_resolve_rag_execution_hints(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            research_profile="",
            coverage_goal="targeted",
            result_mode="answer",
            controller_hints={},
            to_dict=lambda: {
                "research_profile": "",
                "coverage_goal": "targeted",
                "result_mode": "answer",
                "controller_hints": {},
            },
        )

    def fake_run_rag_contract(*args, **kwargs):
        del args
        captured["query"] = kwargs["query"]
        captured["task_context"] = kwargs["task_context"]
        return SimpleNamespace(
            answer="Grounded answer",
            followups=[],
            warnings=[],
            citations=[],
            to_dict=lambda: {"answer": "Grounded answer"},
        )

    monkeypatch.setattr("agentic_chatbot_next.runtime.query_loop.resolve_rag_execution_hints", fake_resolve_rag_execution_hints)
    monkeypatch.setattr("agentic_chatbot_next.runtime.query_loop.run_rag_contract", fake_run_rag_contract)
    monkeypatch.setattr("agentic_chatbot_next.runtime.query_loop.render_rag_contract", lambda contract: contract.answer)
    monkeypatch.setattr(loop, "_maybe_delegate_rag_peer", lambda **kwargs: None)

    result = loop._run_rag(
        agent,
        session,
        user_text=(
            "You are executing a scoped task delegated by a coordinator.\n\n"
            "ORIGINAL_USER_REQUEST:\nAcross the whole default collection, identify the major subsystems.\n\n"
            "TASK_INPUT:\nFind subsystem evidence."
        ),
        skill_context="",
        callbacks=[],
        providers=SimpleNamespace(chat=object()),
        task_payload={
            "worker_request": {
                "task_id": "task_1",
                "title": "Find subsystem evidence",
                "prompt": "You are executing a scoped task delegated by a coordinator.",
                "instruction_prompt": "You are executing a scoped task delegated by a coordinator.",
                "semantic_query": "Find subsystem evidence.",
                "metadata": {},
            },
            "controller_hints": {},
            "skill_queries": [],
        },
    )

    assert result.text == "Grounded answer"
    assert captured["query"] == "Find subsystem evidence."
    assert captured["task_context"] == "You are executing a scoped task delegated by a coordinator."
