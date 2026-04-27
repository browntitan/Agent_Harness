from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from agentic_chatbot_next.providers import ProviderBundle
from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.providers import factory as provider_factory
from agentic_chatbot_next.providers import llm_factory
from agentic_chatbot_next.rag.retrieval import GradedChunk
from agentic_chatbot_next.runtime.kernel import RuntimeKernel
from agentic_chatbot_next.runtime.query_loop import QueryLoop, QueryLoopResult
from agentic_chatbot_next.tools.rag_agent_tool import make_rag_agent_tool


class RecordingChatModel:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict[str, object]] = []

    def invoke(self, messages, config=None):
        self.calls.append({"messages": list(messages), "config": dict(config or {})})
        return SimpleNamespace(content=self.response_text)


class FakeRagContract:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "citations": [],
            "used_citation_ids": [],
            "warnings": [],
            "followups": [],
        }


def _provider_bundle(*, chat_text: str, judge_text: str = "judge") -> ProviderBundle:
    return ProviderBundle(
        chat=RecordingChatModel(chat_text),
        judge=RecordingChatModel(judge_text),
        embeddings=object(),
    )


def _runtime_settings(tmp_path: Path) -> SimpleNamespace:
    repo_root = Path(__file__).resolve().parents[1]
    return SimpleNamespace(
        default_collection_id="default",
        default_tenant_id="tenant",
        runtime_dir=tmp_path / "runtime",
        workspace_dir=tmp_path / "workspaces",
        memory_dir=tmp_path / "memory",
        skills_dir=repo_root / "data" / "skills",
        agents_dir=repo_root / "data" / "agents",
        llm_provider="ollama",
        judge_provider="ollama",
        ollama_chat_model="base-chat",
        ollama_judge_model="base-judge",
        runtime_events_enabled=False,
        max_worker_concurrency=2,
        planner_max_tasks=4,
        rag_top_k_vector=2,
        rag_top_k_keyword=2,
        rag_max_retries=1,
        rag_min_evidence_chunks=1,
        clarification_sensitivity=50,
        enable_coordinator_mode=False,
        chat_max_output_tokens=None,
        demo_chat_max_output_tokens=None,
        judge_max_output_tokens=None,
        ollama_num_predict=None,
        demo_ollama_num_predict=None,
        prompts_backend="local",
        judge_grading_prompt_path=Path("missing"),
        grounded_answer_prompt_path=Path("missing"),
        agent_chat_model_overrides={},
        agent_judge_model_overrides={},
        agent_chat_max_output_tokens={},
    )


def test_agent_provider_resolver_returns_base_bundle_without_override(tmp_path: Path) -> None:
    settings = _runtime_settings(tmp_path)
    base_providers = _provider_bundle(chat_text="base")

    resolver = provider_factory.AgentProviderResolver(settings, base_providers)

    assert resolver.for_agent("general") is base_providers


def test_agent_provider_resolver_reuses_cached_bundle_for_identical_override_tuples(tmp_path: Path, monkeypatch) -> None:
    settings = _runtime_settings(tmp_path)
    settings.agent_chat_model_overrides = {
        "general": "gpt-oss:20b",
        "utility": "gpt-oss:20b",
    }
    settings.agent_judge_model_overrides = {
        "general": "gpt-oss:20b",
        "utility": "gpt-oss:20b",
    }
    base_providers = _provider_bundle(chat_text="base")
    calls: list[tuple[str | None, str | None, object]] = []

    def fake_build_providers(
        settings_arg,
        *,
        embeddings=None,
        chat_model_override=None,
        judge_model_override=None,
        chat_max_output_tokens=None,
        judge_max_output_tokens=None,
    ):
        del settings_arg
        calls.append((chat_model_override, judge_model_override, embeddings, chat_max_output_tokens, judge_max_output_tokens))
        return ProviderBundle(chat=object(), judge=object(), embeddings=embeddings)

    monkeypatch.setattr(provider_factory, "build_providers", fake_build_providers)

    resolver = provider_factory.AgentProviderResolver(settings, base_providers)
    general_bundle = resolver.for_agent("general")
    utility_bundle = resolver.for_agent("utility")

    assert general_bundle is utility_bundle
    assert calls == [("gpt-oss:20b", "gpt-oss:20b", base_providers.embeddings, None, None)]


def test_agent_provider_resolver_includes_output_caps_in_bundle_identity(tmp_path: Path, monkeypatch) -> None:
    settings = _runtime_settings(tmp_path)
    settings.chat_max_output_tokens = 2048
    settings.judge_max_output_tokens = 512
    settings.agent_chat_max_output_tokens = {"general": 4096}
    base_providers = _provider_bundle(chat_text="base")
    calls: list[tuple[str | None, str | None, int | None, int | None]] = []

    def fake_build_providers(
        settings_arg,
        *,
        embeddings=None,
        chat_model_override=None,
        judge_model_override=None,
        chat_max_output_tokens=None,
        judge_max_output_tokens=None,
    ):
        del settings_arg, embeddings
        calls.append((chat_model_override, judge_model_override, chat_max_output_tokens, judge_max_output_tokens))
        return ProviderBundle(chat=object(), judge=object(), embeddings=object())

    monkeypatch.setattr(provider_factory, "build_providers", fake_build_providers)

    resolver = provider_factory.AgentProviderResolver(settings, base_providers)

    general_bundle = resolver.for_agent("general")
    request_override_bundle = resolver.for_agent("general", chat_max_output_tokens=6000)
    repeated_request_override = resolver.for_agent("general", chat_max_output_tokens=6000)

    assert general_bundle is not base_providers
    assert request_override_bundle is repeated_request_override
    assert calls == [
        ("base-chat", "base-judge", 4096, 512),
        ("base-chat", "base-judge", 6000, 512),
    ]


def test_runtime_kernel_passes_agent_specific_providers_into_tool_context_and_query_loop(tmp_path: Path, monkeypatch) -> None:
    settings = _runtime_settings(tmp_path)
    base_providers = _provider_bundle(chat_text="base")
    override_providers = _provider_bundle(chat_text="override")
    kernel = RuntimeKernel(settings, providers=base_providers, stores=SimpleNamespace())
    session_state = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    agent = kernel.registry.get("general")
    assert agent is not None
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        kernel,
        "resolve_providers_for_agent",
        lambda agent_name, chat_max_output_tokens=None: override_providers if agent_name == "general" else base_providers,
    )
    monkeypatch.setattr(
        kernel,
        "_build_tools",
        lambda agent_arg, tool_context: captured.setdefault("tool_context_providers", tool_context.providers) or [],
    )

    def fake_run(agent_arg, session_state_arg, *, user_text, providers=None, tool_context=None, tools=None, task_payload=None):
        del agent_arg, user_text, tools, task_payload
        captured["loop_providers"] = providers
        captured["loop_tool_context_providers"] = getattr(tool_context, "providers", None)
        return QueryLoopResult(text="override result", messages=list(session_state_arg.messages), metadata={})

    monkeypatch.setattr(kernel.query_loop, "run", fake_run)

    result = kernel.run_agent(agent, session_state, user_text="hello", callbacks=[])

    assert result.text == "override result"
    assert captured["tool_context_providers"] is override_providers
    assert captured["loop_providers"] is override_providers
    assert captured["loop_tool_context_providers"] is override_providers


def test_query_loop_uses_override_providers_for_planner_and_finalizer_and_verifier(tmp_path: Path) -> None:
    settings = _runtime_settings(tmp_path)
    base_providers = _provider_bundle(chat_text='{"summary":"base","tasks":[]}')
    override_providers = _provider_bundle(
        chat_text='{"status":"revise","summary":"override verifier","issues":["needs work"],"feedback":"fix it"}'
    )
    loop = QueryLoop(settings=settings, providers=base_providers, stores=SimpleNamespace())
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    planner = AgentDefinition(name="planner", mode="planner", prompt_file="planner_agent.md")
    finalizer = AgentDefinition(name="finalizer", mode="finalizer", prompt_file="finalizer_agent.md")
    verifier = AgentDefinition(name="verifier", mode="verifier", prompt_file="verifier_agent.md")

    planner_override = _provider_bundle(chat_text='{"summary":"override planner","tasks":[]}')
    planner_result = loop.run(planner, session, user_text="plan it", providers=planner_override)
    assert json.loads(planner_result.text)["summary"] == "override planner"

    finalizer_override = _provider_bundle(chat_text="override finalizer")
    finalizer_result = loop.run(
        finalizer,
        session,
        user_text="finalize it",
        providers=finalizer_override,
        task_payload={"partial_answer": "fallback"},
    )
    assert finalizer_result.text == "override finalizer"

    verifier_result = loop.run(
        verifier,
        session,
        user_text="verify it",
        providers=override_providers,
        task_payload={"partial_answer": "candidate"},
    )
    verification = json.loads(verifier_result.text)
    assert verification["summary"] == "override verifier"
    assert verification["status"] == "revise"


def test_query_loop_verifier_passes_through_clarification_requests(tmp_path: Path) -> None:
    settings = _runtime_settings(tmp_path)
    clarification_text = (
        '<clarification_request>{"question":"Should I verify the uploaded files only or the knowledge base only?",'
        '"reason":"retrieval_scope_ambiguous","options":["uploaded files only","knowledge base only"]}</clarification_request>'
    )
    providers = _provider_bundle(chat_text=clarification_text)
    loop = QueryLoop(settings=settings, providers=providers, stores=SimpleNamespace())
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    verifier = AgentDefinition(name="verifier", mode="verifier", prompt_file="verifier_agent.md")

    verifier_result = loop.run(
        verifier,
        session,
        user_text="verify it",
        providers=providers,
        task_payload={"partial_answer": "candidate"},
    )

    assert verifier_result.text == "Should I verify the uploaded files only or the knowledge base only?"
    assert verifier_result.metadata["turn_outcome"] == "clarification_request"
    assert verifier_result.metadata["clarification"]["reason"] == "retrieval_scope_ambiguous"
    assert verifier_result.metadata["clarification"]["options"] == [
        "uploaded files only",
        "knowledge base only",
    ]


def test_query_loop_uses_override_providers_for_rag(tmp_path: Path, monkeypatch) -> None:
    settings = _runtime_settings(tmp_path)
    base_providers = _provider_bundle(chat_text="base")
    override_providers = _provider_bundle(chat_text="override", judge_text="override judge")
    loop = QueryLoop(settings=settings, providers=base_providers, stores=SimpleNamespace())
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    agent = AgentDefinition(name="rag_worker", mode="rag", prompt_file="rag_agent.md")
    captured: dict[str, object] = {}

    def fake_run_rag_contract(
        settings_arg,
        stores_arg,
        *,
        providers,
        session,
        query,
        conversation_context,
        preferred_doc_ids,
        must_include_uploads,
        top_k_vector,
        top_k_keyword,
        max_retries,
        callbacks,
        skill_context,
        task_context,
        search_mode,
        max_search_rounds,
        research_profile="",
        coverage_goal="",
        result_mode="",
        controller_hints=None,
        **kwargs,
    ):
        del settings_arg, stores_arg, session, query, conversation_context, preferred_doc_ids
        del must_include_uploads, top_k_vector, top_k_keyword, max_retries, callbacks
        del skill_context, task_context, search_mode, max_search_rounds, kwargs
        del research_profile, coverage_goal, result_mode, controller_hints
        captured["providers"] = providers
        return FakeRagContract("override rag answer")

    monkeypatch.setattr("agentic_chatbot_next.runtime.query_loop.run_rag_contract", fake_run_rag_contract)
    monkeypatch.setattr(
        "agentic_chatbot_next.runtime.query_loop.render_rag_contract",
        lambda contract: contract.to_dict()["answer"],
    )

    result = loop.run(agent, session, user_text="cite docs", providers=override_providers)

    assert captured["providers"] is override_providers
    assert "override rag answer" in result.text


def test_httpx_client_uses_configured_timeout_settings(tmp_path: Path) -> None:
    settings = _runtime_settings(tmp_path)
    settings.http2_enabled = True
    settings.ssl_verify = True
    settings.ssl_cert_file = None
    settings.llm_http_timeout_seconds = 123
    settings.llm_http_connect_timeout_seconds = 17

    client = llm_factory._build_httpx_client(settings)

    assert client.timeout.connect == 17
    assert client.timeout.read == 123
    assert client.timeout.write == 123
    assert client.timeout.pool == 123
    client.close()


def test_query_loop_forwards_structured_rag_hints(tmp_path: Path, monkeypatch) -> None:
    settings = _runtime_settings(tmp_path)
    providers = _provider_bundle(chat_text="base", judge_text="judge")
    loop = QueryLoop(settings=settings, providers=providers, stores=SimpleNamespace())
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    agent = AgentDefinition(name="rag_worker", mode="rag", prompt_file="rag_agent.md")
    captured: dict[str, object] = {}

    def fake_run_rag_contract(
        settings_arg,
        stores_arg,
        *,
        providers,
        session,
        query,
        conversation_context,
        preferred_doc_ids,
        must_include_uploads,
        top_k_vector,
        top_k_keyword,
        max_retries,
        callbacks,
        skill_context,
        task_context,
        search_mode,
        max_search_rounds,
        research_profile="",
        coverage_goal="",
        result_mode="",
        controller_hints=None,
        **kwargs,
    ):
        del settings_arg, stores_arg, providers, session, query, conversation_context
        del preferred_doc_ids, must_include_uploads, top_k_vector, top_k_keyword, max_retries, callbacks
        del skill_context, task_context, search_mode, max_search_rounds, kwargs
        captured["research_profile"] = research_profile
        captured["coverage_goal"] = coverage_goal
        captured["result_mode"] = result_mode
        captured["controller_hints"] = controller_hints
        return FakeRagContract("hinted answer")

    monkeypatch.setattr("agentic_chatbot_next.runtime.query_loop.run_rag_contract", fake_run_rag_contract)
    monkeypatch.setattr(
        "agentic_chatbot_next.runtime.query_loop.render_rag_contract",
        lambda contract: contract.to_dict()["answer"],
    )

    result = loop.run(
        agent,
        session,
        user_text="Which documents contain workflow steps?",
        providers=providers,
        task_payload={
            "worker_request": {
                "task_id": "task_1",
                "skill_queries": ["corpus discovery", "process flow identification"],
                "research_profile": "corpus_discovery",
                "coverage_goal": "corpus_wide",
                "result_mode": "inventory",
                "controller_hints": {"prefer_inventory_output": True},
            }
        },
    )

    assert result.text == "hinted answer"
    assert captured["research_profile"] == "corpus_discovery"
    assert captured["coverage_goal"] == "corpus_wide"
    assert captured["result_mode"] == "inventory"
    assert captured["controller_hints"]["prefer_inventory_output"] is True
    assert captured["controller_hints"]["prefer_process_flow_docs"] is True


def test_query_loop_uses_max_rag_agent_steps_for_direct_rag_budget(tmp_path: Path, monkeypatch) -> None:
    settings = _runtime_settings(tmp_path)
    settings.max_rag_agent_steps = 7
    providers = _provider_bundle(chat_text="base", judge_text='{"action":"answer"}')
    loop = QueryLoop(settings=settings, providers=providers, stores=SimpleNamespace())
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    agent = AgentDefinition(name="rag_worker", mode="rag", prompt_file="rag_agent.md")
    captured: dict[str, object] = {}

    def fake_run_rag_contract(
        settings_arg,
        stores_arg,
        *,
        max_search_rounds,
        **kwargs,
    ):
        del settings_arg, stores_arg, kwargs
        captured["max_search_rounds"] = max_search_rounds
        return FakeRagContract("budget answer")

    monkeypatch.setattr("agentic_chatbot_next.runtime.query_loop.run_rag_contract", fake_run_rag_contract)
    monkeypatch.setattr(
        "agentic_chatbot_next.runtime.query_loop.render_rag_contract",
        lambda contract: contract.to_dict()["answer"],
    )

    result = loop.run(agent, session, user_text="cite docs", providers=providers)

    assert captured["max_search_rounds"] == 7
    assert result.text == "budget answer"


def test_query_loop_rag_worker_can_queue_peer_follow_up(tmp_path: Path, monkeypatch) -> None:
    settings = _runtime_settings(tmp_path)
    providers = _provider_bundle(
        chat_text="unused",
        judge_text=(
            '{"action":"invoke_agent","agent_name":"data_analyst","description":"analyze the evidence",'
            '"message":"Review the current evidence and look for recurring quantitative patterns.",'
            '"rationale":"Needs specialist follow-up"}'
        ),
    )
    loop = QueryLoop(settings=settings, providers=providers, stores=SimpleNamespace())
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    agent = AgentDefinition(
        name="rag_worker",
        mode="rag",
        prompt_file="rag_agent.md",
        allowed_worker_agents=["data_analyst"],
    )
    captured: dict[str, object] = {}

    class _Kernel:
        def invoke_agent_from_tool(self, tool_context, **kwargs):
            captured["tool_context"] = tool_context
            captured["kwargs"] = kwargs
            return {
                "job_id": "job_peer_123",
                "target_agent": kwargs["agent_name"],
                "status": "queued",
                "reused_existing_job": False,
                "queued": True,
            }

    monkeypatch.setattr(
        "agentic_chatbot_next.runtime.query_loop.run_rag_contract",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run_rag_contract should not execute")),
    )

    tool_context = SimpleNamespace(
        kernel=_Kernel(),
        active_definition=agent,
        active_agent="rag_worker",
        callbacks=[],
        progress_emitter=None,
        rag_runtime_bridge=None,
    )

    result = loop.run(
        agent,
        session,
        user_text="Investigate the evidence more deeply.",
        providers=providers,
        tool_context=tool_context,
    )

    assert captured["kwargs"]["agent_name"] == "data_analyst"
    assert "background `data_analyst` follow-up" in result.text
    assert result.metadata["peer_dispatch"]["job_id"] == "job_peer_123"
    assert result.metadata["turn_outcome"] == "background_delegated"


def test_query_loop_applies_no_confirmed_match_policy_for_worker_request_inventory_search(tmp_path: Path, monkeypatch) -> None:
    settings = _runtime_settings(tmp_path)
    providers = _provider_bundle(chat_text="base", judge_text="judge")
    loop = QueryLoop(settings=settings, providers=providers, stores=SimpleNamespace())
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    agent = AgentDefinition(name="rag_worker", mode="rag", prompt_file="rag_agent.md")
    weak_doc = Document(
        page_content="Workflow description for routing approval handoffs between teams.",
        metadata={
            "doc_id": "doc-weak",
            "chunk_id": "doc-weak#chunk0001",
            "title": "TOOLS_AND_TOOL_CALLING.md",
            "source_type": "kb",
            "chunk_index": 1,
        },
    )

    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.run_retrieval_controller",
        lambda *args, **kwargs: SimpleNamespace(
            selected_docs=[weak_doc],
            candidate_docs=[weak_doc],
            graded=[GradedChunk(doc=weak_doc, relevance=2, reason="test")],
            query_used="Which documents contain onboarding workflows?",
            search_mode="deep",
            rounds=2,
            tool_calls_used=4,
            tool_call_log=["round1:search_corpus[hybrid]:workflow"],
            strategies_used=["hybrid", "keyword"],
            candidate_counts={"unique_docs": 1, "selected_docs": 1},
            evidence_ledger={"round_summaries": [{"queries": ["keyword:workflow approval handoff"]}], "entries": []},
            parallel_workers_used=False,
            retrieval_verification={},
            to_summary=lambda citations_found: SimpleNamespace(
                to_dict=lambda: {},
                query_used="Which documents contain onboarding workflows?",
                steps=4,
                tool_calls_used=4,
                tool_call_log=["round1:search_corpus[hybrid]:workflow"],
                citations_found=citations_found,
                search_mode="deep",
                rounds=2,
                strategies_used=["hybrid", "keyword"],
                candidate_counts={"unique_docs": 1, "selected_docs": 1, "confirmed_match_count": 0},
                parallel_workers_used=False,
                retrieval_verification={
                    "downgraded_to_negative_evidence": True,
                    "downgrade_reason": "no_confirmed_topic_matches",
                },
            ),
        ),
    )
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.engine.get_kb_coverage_status",
        lambda *args, **kwargs: SimpleNamespace(ready=True),
    )

    result = loop.run(
        agent,
        session,
        user_text="Which documents in the knowledge base contain onboarding workflows?",
        providers=providers,
        task_payload={
            "worker_request": {
                "task_id": "task_1",
                "skill_queries": ["corpus discovery", "process flow identification"],
                "research_profile": "corpus_discovery",
                "coverage_goal": "corpus_wide",
                "result_mode": "inventory",
                "controller_hints": {"prefer_inventory_output": True},
            }
        },
    )

    assert "No confirmed knowledge-base documents were found that explicitly mention onboarding workflows" in result.text
    assert "TOOLS_AND_TOOL_CALLING.md" not in result.text
    assert "Warnings: INSUFFICIENT_CORPUS_EVIDENCE" in result.text


def test_query_loop_does_not_hard_scope_rag_to_uploaded_doc_ids(tmp_path: Path, monkeypatch) -> None:
    settings = _runtime_settings(tmp_path)
    providers = _provider_bundle(chat_text="base", judge_text="judge")
    loop = QueryLoop(settings=settings, providers=providers, stores=SimpleNamespace())
    session = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
        uploaded_doc_ids=["doc-upload-1"],
    )
    agent = AgentDefinition(name="rag_worker", mode="rag", prompt_file="rag_agent.md")
    captured: dict[str, object] = {}

    def fake_run_rag_contract(
        settings_arg,
        stores_arg,
        *,
        providers,
        session,
        query,
        conversation_context,
        preferred_doc_ids,
        must_include_uploads,
        top_k_vector,
        top_k_keyword,
        max_retries,
        callbacks,
        skill_context,
        task_context,
        search_mode,
        max_search_rounds,
        research_profile="",
        coverage_goal="",
        result_mode="",
        controller_hints=None,
        **kwargs,
    ):
        del settings_arg, stores_arg, providers, session, query, conversation_context
        del top_k_vector, top_k_keyword, max_retries, callbacks, skill_context, task_context
        del search_mode, max_search_rounds, research_profile, coverage_goal, result_mode
        del controller_hints, kwargs
        captured["preferred_doc_ids"] = preferred_doc_ids
        captured["must_include_uploads"] = must_include_uploads
        return FakeRagContract("scope answer")

    monkeypatch.setattr("agentic_chatbot_next.runtime.query_loop.run_rag_contract", fake_run_rag_contract)
    monkeypatch.setattr(
        "agentic_chatbot_next.runtime.query_loop.render_rag_contract",
        lambda contract: contract.to_dict()["answer"],
    )

    result = loop.run(
        agent,
        session,
        user_text="Use both the uploaded file and the docs.",
        providers=providers,
    )

    assert captured["preferred_doc_ids"] == []
    assert captured["must_include_uploads"] is True
    assert result.text == "scope answer"


def test_query_loop_rag_worker_returns_evidence_only_payload_for_internal_search_tasks(tmp_path: Path, monkeypatch) -> None:
    settings = _runtime_settings(tmp_path)
    providers = _provider_bundle(chat_text="unused", judge_text="unused judge")
    loop = QueryLoop(settings=settings, providers=providers, stores=SimpleNamespace())
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    agent = AgentDefinition(name="rag_worker", mode="rag", prompt_file="rag_agent.md")

    worker_doc = Document(
        page_content="Parallel evidence chunk",
        metadata={
            "doc_id": "doc-1",
            "chunk_id": "doc-1#chunk0001",
            "title": "policy.md",
            "source_type": "kb",
        },
    )

    def fake_run_retrieval_controller(
        settings_arg,
        stores_arg,
        *,
        providers,
        session,
        query,
        conversation_context,
        preferred_doc_ids,
        must_include_uploads,
        top_k_vector,
        top_k_keyword,
        max_retries,
        callbacks,
        search_mode,
        max_search_rounds,
        allow_internal_fanout,
        **kwargs,
    ):
        del settings_arg, stores_arg, providers, session, query, conversation_context, preferred_doc_ids
        del must_include_uploads, top_k_vector, top_k_keyword, max_retries, callbacks, search_mode, max_search_rounds, kwargs
        assert allow_internal_fanout is False
        return SimpleNamespace(
            evidence_ledger={
                "entries": [
                    {
                        "chunk_id": "doc-1#chunk0001",
                        "doc_id": "doc-1",
                        "title": "policy.md",
                        "query": "Find workflow evidence",
                        "strategy": "worker",
                        "rationale": "parallel",
                        "score": 0.88,
                        "relevance": 3,
                        "coverage_state": "strong",
                        "grade_reason": "strong evidence",
                    }
                ]
            },
            candidate_docs=[worker_doc],
            graded=[GradedChunk(doc=worker_doc, relevance=3, reason="strong evidence")],
            selected_docs=[worker_doc],
        )

    monkeypatch.setattr("agentic_chatbot_next.runtime.query_loop.run_retrieval_controller", fake_run_retrieval_controller)

    result = loop.run(
        agent,
        session,
        user_text="Find workflow evidence",
        providers=providers,
        task_payload={
            "worker_request": {
                "task_id": "task_1",
                "metadata": {
                    "answer_mode": "evidence_only",
                    "rag_search_task": {
                        "task_id": "task_1",
                        "title": "Search policy",
                        "query": "Find workflow evidence",
                        "doc_scope": ["doc-1"],
                        "round_budget": 1,
                        "answer_mode": "evidence_only",
                    },
                },
            }
        },
    )

    payload = json.loads(result.text)
    assert payload["task_id"] == "task_1"
    assert payload["candidate_docs"][0]["metadata"]["doc_id"] == "doc-1"
    assert result.metadata["rag_search_result"]["task_id"] == "task_1"


def test_query_loop_drops_unresolved_collection_scope_from_rag_search_task(tmp_path: Path, monkeypatch) -> None:
    settings = _runtime_settings(tmp_path)
    providers = _provider_bundle(chat_text="unused", judge_text="unused judge")
    loop = QueryLoop(settings=settings, providers=providers, stores=SimpleNamespace())
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")
    agent = AgentDefinition(name="rag_worker", mode="rag", prompt_file="rag_agent.md")
    captured: dict[str, object] = {}

    def fake_run_rag_contract(
        settings_arg,
        stores_arg,
        *,
        providers,
        session,
        query,
        conversation_context,
        preferred_doc_ids,
        must_include_uploads,
        top_k_vector,
        top_k_keyword,
        max_retries,
        callbacks,
        skill_context,
        task_context,
        search_mode,
        max_search_rounds,
        research_profile="",
        coverage_goal="",
        result_mode="",
        controller_hints=None,
        **kwargs,
    ):
        del settings_arg, stores_arg, providers, session, query, conversation_context
        del must_include_uploads, top_k_vector, top_k_keyword, max_retries, callbacks
        del skill_context, task_context, search_mode, max_search_rounds, controller_hints, kwargs
        captured["preferred_doc_ids"] = preferred_doc_ids
        captured["research_profile"] = research_profile
        captured["coverage_goal"] = coverage_goal
        captured["result_mode"] = result_mode
        return FakeRagContract("scope answer")

    monkeypatch.setattr("agentic_chatbot_next.runtime.query_loop.run_rag_contract", fake_run_rag_contract)
    monkeypatch.setattr(
        "agentic_chatbot_next.runtime.query_loop.render_rag_contract",
        lambda contract: contract.to_dict()["answer"],
    )

    result = loop.run(
        agent,
        session,
        user_text="""
        Goal: Investigate the major subsystems in this repo and provide me a list of documents that have information about the major sub systems
        Context: The documents i want you to pull this information from is provided to you internally in the knowledge base we have access to in the "default" collection, conduct your search accross the documents that are in there
        Deliverable: you will provide me only a list of potential documents that have information about major subsystems
        """,
        providers=providers,
        task_payload={
            "worker_request": {
                "task_id": "task_1",
                "metadata": {
                    "rag_search_task": {
                        "task_id": "task_1",
                        "title": "Find relevant documents",
                        "query": "Identify documents that discuss the major subsystems in this repo.",
                        "doc_scope": ["default"],
                        "round_budget": 2,
                        "answer_mode": "answer",
                    }
                },
            }
        },
    )

    assert captured["preferred_doc_ids"] == []
    assert captured["research_profile"] == "corpus_discovery"
    assert captured["coverage_goal"] == "corpus_wide"
    assert captured["result_mode"] == "inventory"
    assert result.text == "scope answer"


def test_query_loop_evidence_only_respects_kb_only_scope_when_uploads_exist(tmp_path: Path, monkeypatch) -> None:
    settings = _runtime_settings(tmp_path)
    providers = _provider_bundle(chat_text="unused", judge_text="unused judge")
    loop = QueryLoop(settings=settings, providers=providers, stores=SimpleNamespace())
    session = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
        uploaded_doc_ids=["upload-1"],
        metadata={"kb_collection_id": "default"},
    )
    agent = AgentDefinition(name="rag_worker", mode="rag", prompt_file="rag_agent.md")
    captured: dict[str, object] = {}
    worker_doc = Document(
        page_content="runtime service",
        metadata={"doc_id": "doc-1", "title": "ARCHITECTURE.md", "source_type": "kb"},
    )

    def fake_run_retrieval_controller(
        settings_arg,
        stores_arg,
        *,
        providers,
        session,
        query,
        conversation_context,
        preferred_doc_ids,
        must_include_uploads,
        top_k_vector,
        top_k_keyword,
        max_retries,
        callbacks,
        search_mode,
        max_search_rounds,
        allow_internal_fanout,
        **kwargs,
    ):
        del settings_arg, stores_arg, providers, session, query, conversation_context, preferred_doc_ids
        del top_k_vector, top_k_keyword, max_retries, callbacks, search_mode, max_search_rounds, allow_internal_fanout
        captured["must_include_uploads"] = must_include_uploads
        captured["controller_hints"] = dict(kwargs.get("controller_hints") or {})
        return SimpleNamespace(
            evidence_ledger={"entries": []},
            candidate_docs=[worker_doc],
            graded=[GradedChunk(doc=worker_doc, relevance=3, reason="strong evidence")],
            selected_docs=[worker_doc],
        )

    monkeypatch.setattr("agentic_chatbot_next.runtime.query_loop.run_retrieval_controller", fake_run_retrieval_controller)

    result = loop.run(
        agent,
        session,
        user_text="Find subsystem docs",
        providers=providers,
        task_payload={
            "worker_request": {
                "task_id": "task_1",
                "metadata": {
                    "answer_mode": "evidence_only",
                    "rag_search_task": {
                        "task_id": "task_1",
                        "title": "Seed corpus scan",
                        "query": "Find subsystem docs",
                        "doc_scope": [],
                        "round_budget": 2,
                        "answer_mode": "evidence_only",
                        "controller_hints": {
                            "retrieval_scope_mode": "kb_only",
                            "search_collection_ids": ["default"],
                        },
                    },
                },
            }
        },
    )

    assert captured["must_include_uploads"] is False
    assert captured["controller_hints"]["retrieval_scope_mode"] == "kb_only"
    assert json.loads(result.text)["task_id"] == "task_1"


def test_job_runner_resolves_providers_for_worker_agent_name(tmp_path: Path, monkeypatch) -> None:
    settings = _runtime_settings(tmp_path)
    base_providers = _provider_bundle(chat_text="base")
    utility_providers = _provider_bundle(chat_text="utility override")
    kernel = RuntimeKernel(settings, providers=base_providers, stores=SimpleNamespace())
    seen: list[str] = []

    def fake_resolve(agent_name: str, chat_max_output_tokens=None):
        del chat_max_output_tokens
        seen.append(agent_name)
        if agent_name == "utility":
            return utility_providers
        return base_providers

    monkeypatch.setattr(kernel, "resolve_providers_for_agent", fake_resolve)
    monkeypatch.setattr(
        kernel.query_loop,
        "run",
        lambda agent, session_state, *, user_text, providers=None, tool_context=None, tools=None, task_payload=None: QueryLoopResult(
            text=f"worker:{agent.name}",
            messages=list(session_state.messages),
            metadata={},
        ),
    )

    job = kernel.job_manager.create_job(
        agent_name="utility",
        prompt="worker prompt",
        session_id="tenant:user:conv",
        description="worker",
        metadata={
            "session_state": {
                "tenant_id": "tenant",
                "user_id": "user",
                "conversation_id": "conv",
            },
            "worker_request": {
                "task_id": "task_1",
                "skill_queries": [],
            },
        },
    )

    result = kernel._job_runner(job)

    assert result == "worker:utility"
    assert seen == ["utility"]


def test_rag_agent_tool_forwards_search_mode_and_max_rounds(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_rag_contract(
        settings_arg,
        stores_arg,
        *,
        providers,
        session,
        query,
        conversation_context,
        preferred_doc_ids,
        must_include_uploads,
        top_k_vector,
        top_k_keyword,
        max_retries,
        callbacks,
        search_mode,
        max_search_rounds,
        research_profile="",
        coverage_goal="",
        result_mode="",
        controller_hints=None,
        **kwargs,
    ):
        del settings_arg, stores_arg, providers, session, query, conversation_context
        del preferred_doc_ids, must_include_uploads, max_retries, callbacks, kwargs
        del research_profile, coverage_goal, result_mode, controller_hints
        captured["search_mode"] = search_mode
        captured["max_search_rounds"] = max_search_rounds
        captured["top_k_vector"] = top_k_vector
        captured["top_k_keyword"] = top_k_keyword
        return FakeRagContract("adaptive answer")

    monkeypatch.setattr("agentic_chatbot_next.tools.rag_agent_tool.run_rag_contract", fake_run_rag_contract)

    tool = make_rag_agent_tool(
        SimpleNamespace(),
        SimpleNamespace(),
        providers=SimpleNamespace(),
        session=SimpleNamespace(scratchpad={}),
    )

    result = tool.invoke({"query": "find workflows", "search_mode": "deep", "max_search_rounds": 4})

    assert result["answer"] == "adaptive answer"
    assert captured == {
        "search_mode": "deep",
        "max_search_rounds": 4,
        "top_k_vector": 15,
        "top_k_keyword": 15,
    }
