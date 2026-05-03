from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from agentic_chatbot_next.agents.prompt_builder import PromptBuilder
from agentic_chatbot_next.agents.registry import AgentRegistry
from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.contracts.rag import RagContract, RetrievalSummary
from agentic_chatbot_next.rag.engine import _answer_context
from agentic_chatbot_next.runtime.query_loop import QueryLoop
from agentic_chatbot_next.skills.runtime import SkillRuntime
from agentic_chatbot_next.skills.resolver import ResolvedSkillContext
from agentic_chatbot_next.tools.registry import build_tool_definitions


_LEGACY_TOOL_NAMES = re.compile(
    r"\b(resolve_document|search_document|search_all_documents|diff_documents|compare_clauses|"
    r"extract_clauses|extract_requirements|fetch_chunk_window|fetch_document_outline|search_collection|list_collections)\b"
)
_DEMO_HINT_TERMS = re.compile(
    r"\b("
    r"rfp-corpus|requirements-extraction-pack|defense_rag_test_corpus|"
    r"asterion|blue mica|ember reach|iron vale|trident echo|raven crest|"
    r"customer_reviews|sales_performance|support_tickets|marketing_leads|"
    r"request-for-proposal|primary corporate knowledge base"
    r")\b"
    r"|defense[-_\s]+(?:corpus|graph|repository|program)",
    re.IGNORECASE,
)
_TERSE_DEFAULT_TERMS = re.compile(
    r"concise by default|Be concise and direct|clearly and concisely|"
    r"smallest answer shape|one short synthesis paragraph|simplest shape",
    re.IGNORECASE,
)


def test_tool_definitions_have_rich_metadata() -> None:
    definitions = build_tool_definitions(None)
    assert definitions
    for name, definition in definitions.items():
        assert not definition.validate_metadata(), name
        assert definition.render_tool_card()
        assert definition.args_schema


def test_prompt_files_have_required_sections_and_no_placeholder_or_legacy_terms() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    role_prompts = repo_root / "data" / "skills"
    required_sections = {
        "general_agent.md": [
            "## Mission",
            "## Capabilities And Limits",
            "## Task Intake And Clarification Rules",
            "## Tool And Delegation Policy",
            "## Failure Recovery",
            "## Output Shaping",
            "## Anti-Patterns And Avoid Rules",
        ],
        "basic_chat.md": [
            "## Mission",
            "## Capabilities And Limits",
            "## Task Intake And Clarification Rules",
            "## Output Shaping",
        ],
        "utility_agent.md": [
            "## Mission",
            "## Capabilities And Limits",
            "## Task Intake And Clarification Rules",
            "## Tool And Delegation Policy",
            "## Output Shaping",
        ],
        "data_analyst_agent.md": [
            "## Mission",
            "## Capabilities And Limits",
            "## Task Intake And Clarification Rules",
            "## Tool And Delegation Policy",
            "## Failure Recovery",
        ],
        "graph_manager_agent.md": [
            "## Mission",
            "## Capabilities And Limits",
            "## Task Intake And Clarification Rules",
            "## Tool And Delegation Policy",
            "## Output Shaping",
        ],
        "planner_agent.md": [
            "## Mission",
            "## Capabilities And Limits",
            "## Task Intake And Clarification Rules",
            "## Output Shaping",
        ],
        "finalizer_agent.md": [
            "## Mission",
            "## Capabilities And Limits",
            "## Task Intake And Clarification Rules",
            "## Output Shaping",
            "## Anti-Patterns And Avoid Rules",
        ],
        "verifier_agent.md": [
            "## Mission",
            "## Capabilities And Limits",
            "## Task Intake And Clarification Rules",
            "## Output Shaping",
        ],
        "supervisor_agent.md": [
            "## Mission",
            "## Capabilities And Limits",
            "## Task Intake And Clarification Rules",
            "## Tool And Delegation Policy",
            "## Failure Recovery",
            "## Output Shaping",
        ],
        "rag_agent.md": [
            "## Mission",
            "## Capabilities And Limits",
            "## Task Intake And Clarification Rules",
            "## Output Shaping",
            "## Anti-Patterns And Avoid Rules",
        ],
    }
    for filename, sections in required_sections.items():
        text = (role_prompts / filename).read_text(encoding="utf-8")
        assert "placeholder" not in text.lower(), filename
        assert not _LEGACY_TOOL_NAMES.search(text), filename
        for section in sections:
            assert section in text, f"{filename} missing {section}"

    for path in sorted((repo_root / "data" / "skill_packs").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        assert "placeholder" not in text.lower(), path
        assert not _LEGACY_TOOL_NAMES.search(text), path

    for path in sorted((repo_root / "data" / "prompts").glob("*.md")):
        assert "placeholder" not in path.read_text(encoding="utf-8").lower(), path


def test_response_depth_guidance_prefers_balanced_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runtime_paths = [
        repo_root / "data" / "skills" / "skills.md",
        repo_root / "data" / "skills" / "general_agent.md",
        repo_root / "data" / "skills" / "basic_chat.md",
        repo_root / "data" / "skills" / "rag_agent.md",
        repo_root / "data" / "skills" / "finalizer_agent.md",
        repo_root / "data" / "prompts" / "grounded_answer.txt",
        repo_root / "data" / "prompts" / "rag_synthesis.txt",
        repo_root / "src" / "agentic_chatbot_next" / "general_agent.py",
        repo_root / "src" / "agentic_chatbot_next" / "prompt_fallbacks.py",
        repo_root / "src" / "agentic_chatbot_next" / "prompting.py",
        repo_root / "src" / "agentic_chatbot_next" / "rag" / "engine.py",
    ]
    for path in runtime_paths:
        text = path.read_text(encoding="utf-8")
        assert not _TERSE_DEFAULT_TERMS.search(text), path

    shared = (repo_root / "data" / "skills" / "skills.md").read_text(encoding="utf-8")
    general = (repo_root / "data" / "skills" / "general_agent.md").read_text(encoding="utf-8")
    grounded = (repo_root / "data" / "prompts" / "grounded_answer.txt").read_text(encoding="utf-8")
    assert "Default to substantial but not verbose" in shared
    assert "Default to substantial but not verbose" in general
    assert "Default to substantial but not verbose" in grounded


def test_rag_prompt_templates_and_fallbacks_use_balanced_answer_shapes() -> None:
    from agentic_chatbot_next.prompt_fallbacks import fallback_prompt_for_key, fallback_shared_prompt
    from agentic_chatbot_next.prompting import load_grounded_answer_prompt, load_rag_synthesis_prompt

    repo_root = Path(__file__).resolve().parents[1]
    settings = SimpleNamespace(
        prompts_backend="local",
        grounded_answer_prompt_path=repo_root / "data" / "prompts" / "grounded_answer.txt",
        rag_synthesis_prompt_path=repo_root / "data" / "prompts" / "rag_synthesis.txt",
        control_panel_prompt_overlays_dir=None,
    )
    grounded = load_grounded_answer_prompt(settings)
    synthesis = load_rag_synthesis_prompt(settings)
    assert "substantial but not verbose" in grounded
    assert "do not collapse rich findings into a vague paragraph" in synthesis

    missing_prompt_settings = SimpleNamespace(
        prompts_backend="local",
        grounded_answer_prompt_path=Path("missing-grounded.txt"),
        rag_synthesis_prompt_path=Path("missing-synthesis.txt"),
        control_panel_prompt_overlays_dir=None,
    )
    assert "substantial but not verbose" in load_grounded_answer_prompt(missing_prompt_settings)
    assert "do not collapse rich findings into a vague paragraph" in load_rag_synthesis_prompt(missing_prompt_settings)
    assert "substantial but not verbose" in fallback_shared_prompt()
    for key in ("general_agent", "basic_chat", "rag_agent", "finalizer_agent"):
        assert not _TERSE_DEFAULT_TERMS.search(fallback_prompt_for_key(key)), key


def test_general_agent_recovery_synthesis_asks_for_enough_detail() -> None:
    from agentic_chatbot_next.general_agent import _synthesize_tool_results

    class CapturingLLM:
        def __init__(self) -> None:
            self.messages = []

        def invoke(self, messages, config=None):  # noqa: ANN001, ANN202
            self.messages = list(messages)
            return SimpleNamespace(content="Recovered answer.")

    llm = CapturingLLM()
    text = _synthesize_tool_results(
        llm,
        user_text="Summarize the tool result.",
        tool_results=[{"tool": "example", "result": {"answer": "Useful evidence"}}],
        callbacks=[],
        system_prompt="",
        recovery_reason="unit_test",
    )
    assert text == "Recovered answer."
    system_prompt = llm.messages[0].content
    assert "enough detail to satisfy the request" in system_prompt
    assert "clearly and concisely" not in system_prompt


def test_runtime_prompt_surfaces_do_not_include_demo_corpus_hints() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runtime_paths = [
        *sorted((repo_root / "data" / "agents").glob("*.md")),
        *sorted((repo_root / "data" / "prompts").glob("*.md")),
        *sorted((repo_root / "data" / "prompts").glob("*.txt")),
        *sorted((repo_root / "data" / "skill_packs").rglob("*.md")),
        repo_root / "src" / "agentic_chatbot_next" / "tools" / "registry.py",
        repo_root / "src" / "agentic_chatbot_next" / "rag" / "inventory.py",
        repo_root / "control_panel" / "src" / "App.tsx",
    ]

    for path in runtime_paths:
        text = path.read_text(encoding="utf-8")
        assert not _DEMO_HINT_TERMS.search(text), path


def test_prompt_assembly_includes_shared_charter_for_all_prompt_backed_roles() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    registry = AgentRegistry(repo_root / "data" / "agents")
    runtime = SkillRuntime(
        settings=SimpleNamespace(),
        stores=None,
        prompt_builder=PromptBuilder(repo_root / "data" / "skills"),
    )
    for agent in registry.list():
        if agent.mode in {"rag", "memory_maintainer"}:
            continue
        prompt = runtime.build_prompt(agent)
        assert "Shared Charter" in prompt, agent.name
        assert "## Mission" in prompt, agent.name


def test_direct_rag_answer_context_includes_base_guidance() -> None:
    context = _answer_context(
        "How does auth work?",
        "user asked about auth",
        {},
        base_guidance="## Mission\nGround every claim in retrieved evidence.",
        task_context="focus on authentication",
    )

    assert "Base guidance:" in context
    assert "Ground every claim in retrieved evidence." in context
    assert "Task focus:" in context


def test_query_loop_passes_base_guidance_to_direct_rag_contract(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class _FakeExecutionHints(SimpleNamespace):
        def to_dict(self) -> dict[str, object]:
            return {
                "research_profile": self.research_profile,
                "coverage_goal": self.coverage_goal,
                "result_mode": self.result_mode,
                "controller_hints": self.controller_hints,
            }

    class FakeSkillRuntime:
        def build_prompt(self, agent):
            del agent
            return "## Shared Charter\nShared.\n\n---\n\n# RAG Agent\n## Mission\nUse grounded evidence."

    loop = QueryLoop(
        settings=SimpleNamespace(
            rag_top_k_vector=8,
            rag_top_k_keyword=8,
            rag_max_retries=1,
            max_rag_agent_steps=4,
        ),
        skill_runtime=FakeSkillRuntime(),
    )
    agent = AgentDefinition(name="rag_worker", mode="rag", prompt_file="rag_agent.md")
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conversation")

    monkeypatch.setattr(
        "agentic_chatbot_next.runtime.query_loop.resolve_rag_execution_hints",
        lambda *args, **kwargs: _FakeExecutionHints(
            research_profile="",
            coverage_goal="",
            result_mode="answer",
            controller_hints={},
        ),
    )
    monkeypatch.setattr(loop, "_maybe_delegate_rag_peer", lambda **kwargs: None)

    def fake_run_rag_contract(settings, stores, **kwargs):
        del settings, stores
        captured["base_guidance"] = kwargs.get("base_guidance", "")
        return RagContract(
            answer="Grounded answer.",
            citations=[],
            used_citation_ids=[],
            confidence=0.7,
            retrieval_summary=RetrievalSummary(
                query_used="How does auth work?",
                steps=1,
                tool_calls_used=1,
                tool_call_log=[],
                citations_found=0,
            ),
            followups=[],
            warnings=[],
        )

    monkeypatch.setattr("agentic_chatbot_next.runtime.query_loop.run_rag_contract", fake_run_rag_contract)

    result = loop._run_rag(
        agent,
        session,
        user_text="How does auth work?",
        skill_context="",
        callbacks=[],
        providers=SimpleNamespace(chat=object()),
        tool_context=None,
        task_payload={},
    )

    assert "Shared Charter" in captured["base_guidance"]
    assert "Use grounded evidence." in captured["base_guidance"]
    assert result.text


def test_skill_runtime_uses_bounded_worker_semantic_query_for_resolution() -> None:
    runtime = SkillRuntime(
        settings=SimpleNamespace(skill_context_max_chars=80),
        stores=SimpleNamespace(),
        prompt_builder=SimpleNamespace(),
    )
    captured: dict[str, str] = {}

    def fake_resolve(**kwargs):
        captured["query"] = kwargs["query"]
        return ResolvedSkillContext(text="")

    runtime.resolver = SimpleNamespace(resolve=fake_resolve)
    agent = AgentDefinition(name="general", mode="agent", skill_scope="general")
    session = SessionState(tenant_id="tenant", user_id="user", conversation_id="conversation")
    long_prompt = (
        "You are executing a scoped task delegated by a coordinator.\n"
        "ORIGINAL_USER_REQUEST:\n"
        + ("major subsystem summary\n" * 20)
    )

    runtime.resolve_context(
        agent,
        session,
        user_text=long_prompt,
        task_payload={
            "worker_request": {
                "semantic_query": "what knowledge bases do i have access to",
                "skill_queries": ["inventory lookup", "knowledge base access"],
            }
        },
    )

    assert captured["query"].startswith("what knowledge bases do i have access to")
    assert "You are executing a scoped task delegated by a coordinator." not in captured["query"]
    assert len(captured["query"]) <= 80
