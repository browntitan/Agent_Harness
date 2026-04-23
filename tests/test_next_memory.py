from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.memory.context_builder import MemoryContextBuilder
from agentic_chatbot_next.memory.extractor import MemoryExtractor
from agentic_chatbot_next.memory.file_store import FileMemoryStore
from agentic_chatbot_next.memory.store import MemoryCandidate, MemorySelection
from agentic_chatbot_next.runtime.kernel import RuntimeKernel
from agentic_chatbot_next.runtime.context import RuntimePaths
from agentic_chatbot_next.tools.base import ToolContext
from agentic_chatbot_next.tools.groups.memory import build_memory_tools


def _paths(tmp_path: Path) -> RuntimePaths:
    settings = SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        workspace_dir=tmp_path / "workspaces",
        memory_dir=tmp_path / "memory",
    )
    return RuntimePaths.from_settings(settings)


def _session() -> SessionState:
    return SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conversation",
    )


def _kernel_settings(tmp_path: Path, *, memory_enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        workspace_dir=tmp_path / "workspaces",
        memory_dir=tmp_path / "memory",
        agents_dir=Path(__file__).resolve().parents[1] / "data" / "agents",
        skills_dir=Path(__file__).resolve().parents[1] / "data" / "skills",
        runtime_events_enabled=True,
        max_worker_concurrency=2,
        memory_enabled=memory_enabled,
    )


def test_file_memory_store_writes_authoritative_and_derived_files(tmp_path: Path) -> None:
    store = FileMemoryStore(_paths(tmp_path))
    store.save(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conversation",
        scope="conversation",
        key="preferred_name",
        value="Shiv",
    )

    memory_dir = _paths(tmp_path).conversation_memory_dir("tenant", "user", "conversation")
    index_payload = json.loads((memory_dir / "index.json").read_text(encoding="utf-8"))
    assert index_payload["entries"]["preferred_name"]["value"] == "Shiv"
    assert (memory_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert list((memory_dir / "topics").glob("*.md"))


def test_memory_context_builder_and_extractor_use_scopes(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    store = FileMemoryStore(paths)
    extractor = MemoryExtractor(store)
    session = _session()

    saved = extractor.apply_from_text(
        session,
        "preferred_name: Shiv\nfavorite_editor: Neovim",
        scopes=["user"],
    )
    assert saved == 2

    agent = AgentDefinition(
        name="general",
        mode="react",
        prompt_file="general_agent.md",
        memory_scopes=["user"],
    )
    context = MemoryContextBuilder(store, max_chars=500).build_for_agent(agent, session)
    assert "preferred_name" in context
    assert "favorite_editor" in context
    assert "conversation memory" not in context


def test_memory_extractor_can_apply_from_recent_messages(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    store = FileMemoryStore(paths)
    extractor = MemoryExtractor(store)
    session = _session()
    session.append_message("user", "remember that favorite_language is Python.")
    session.append_message("assistant", "Noted.")

    saved = extractor.apply_from_messages(session, session.messages, scopes=["user"])

    assert saved == 1
    assert (
        store.get(
            tenant_id="tenant",
            user_id="user",
            conversation_id="conversation",
            scope="user",
            key="favorite_language",
        )
        == "Python"
    )


def test_memory_extractor_parses_key_value_assignments_from_sentences(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    store = FileMemoryStore(paths)
    extractor = MemoryExtractor(store)
    session = _session()

    saved = extractor.apply_from_text(
        session,
        "Remember these exact values: risk_reserve_monthly_usd=40250; target_jurisdiction=England and Wales.",
        scopes=["user"],
    )

    assert saved == 2
    assert (
        store.get(
            tenant_id="tenant",
            user_id="user",
            conversation_id="conversation",
            scope="user",
            key="risk_reserve_monthly_usd",
        )
        == "40250"
    )
    assert (
        store.get(
            tenant_id="tenant",
            user_id="user",
            conversation_id="conversation",
            scope="user",
            key="target_jurisdiction",
        )
        == "England and Wales"
    )


def test_memory_context_builder_respects_scope_order_and_char_budget(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    store = FileMemoryStore(paths)
    session = _session()
    store.save(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conversation",
        scope="conversation",
        key="project_status",
        value="active",
    )
    store.save(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conversation",
        scope="user",
        key="preferred_name",
        value="Shiv",
    )
    agent = AgentDefinition(
        name="general",
        mode="react",
        prompt_file="general_agent.md",
        memory_scopes=["conversation", "user"],
    )

    context = MemoryContextBuilder(store, max_chars=35).build_for_agent(agent, session)

    assert context.startswith("[conversation memory]")
    assert "project_status" in context
    assert "preferred_name" not in context


def test_memory_context_builder_can_render_managed_selector_brief(tmp_path: Path) -> None:
    class _FakeStore:
        def __init__(self) -> None:
            self.import_calls = 0

        def import_legacy_for_session(self, **_: object) -> int:
            self.import_calls += 1
            return 0

    class _FakeRetriever:
        def __init__(self) -> None:
            self.last_query = ""
            self.last_scopes: list[str] = []

        def retrieve(self, *, session_state: SessionState, query: str, scopes: list[str]) -> list[MemoryCandidate]:
            del session_state
            self.last_query = query
            self.last_scopes = list(scopes)
            return [
                MemoryCandidate(
                    candidate_id="mem_1",
                    candidate_kind="record",
                    score=0.95,
                    reason="active_decision",
                    text="The current implementation should preserve active task state and user preferences.",
                    scope="conversation",
                    memory_type="decision",
                )
            ]

    class _FakeSelector:
        def __init__(self) -> None:
            self.last_user_text = ""

        def select(self, *, session_state: SessionState, agent: AgentDefinition, user_text: str, providers: object | None, candidates: list[MemoryCandidate]) -> MemorySelection:
            del session_state, agent, providers, candidates
            self.last_user_text = user_text
            return MemorySelection(
                selected_ids=["mem_1"],
                brief="- [active_decision] Preserve active task state and user preferences for the next answer.",
                confidence=0.93,
                mode="llm",
            )

    store = _FakeStore()
    retriever = _FakeRetriever()
    selector = _FakeSelector()
    session = _session()
    session.append_message("user", "What should the assistant remember for the next turn?")
    agent = AgentDefinition(
        name="general",
        mode="react",
        prompt_file="general_agent.md",
        memory_scopes=["conversation"],
    )

    context = MemoryContextBuilder(
        store,
        candidate_retriever=retriever,
        selector=selector,
        settings=SimpleNamespace(memory_manager_mode="selector", memory_shadow_mode=False),
    ).build_for_agent(
        agent,
        session,
        user_text="What should the assistant remember for the next turn?",
    )

    assert context.startswith("[managed memory brief]")
    assert "Preserve active task state and user preferences" in context
    assert retriever.last_query == "What should the assistant remember for the next turn?"
    assert retriever.last_scopes == ["conversation"]
    assert selector.last_user_text == "What should the assistant remember for the next turn?"
    assert store.import_calls == 1


def test_file_memory_store_serializes_parallel_writes_to_same_scope(tmp_path: Path) -> None:
    store = FileMemoryStore(_paths(tmp_path))

    def save_pair(index: int) -> None:
        store.save(
            tenant_id="tenant",
            user_id="user",
            conversation_id="conversation",
            scope="user",
            key=f"key_{index}",
            value=f"value_{index}",
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(save_pair, range(8)))

    keys = store.list_keys(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conversation",
        scope="user",
    )
    assert keys == [f"key_{index}" for index in range(8)]


def test_runtime_memory_maintenance_uses_heuristic_conversation_scope_by_default(tmp_path: Path) -> None:
    settings = SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        workspace_dir=tmp_path / "workspaces",
        memory_dir=tmp_path / "memory",
        agents_dir=Path(__file__).resolve().parents[1] / "data" / "agents",
        skills_dir=Path(__file__).resolve().parents[1] / "data" / "skills",
        runtime_events_enabled=True,
        max_worker_concurrency=2,
    )
    kernel = RuntimeKernel(settings, providers=None, stores=None)
    session = _session()

    kernel._run_post_turn_memory_maintenance(session, latest_text="project_status=active")

    assert (
        kernel.file_memory_store.get(
            tenant_id="tenant",
            user_id="user",
            conversation_id="conversation",
            scope="conversation",
            key="project_status",
        )
        == "active"
    )
    assert (
        kernel.file_memory_store.get(
            tenant_id="tenant",
            user_id="user",
            conversation_id="conversation",
            scope="user",
            key="project_status",
        )
        is None
    )


def test_runtime_memory_maintenance_requires_explicit_intent_for_user_scope(tmp_path: Path) -> None:
    settings = SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        workspace_dir=tmp_path / "workspaces",
        memory_dir=tmp_path / "memory",
        agents_dir=Path(__file__).resolve().parents[1] / "data" / "agents",
        skills_dir=Path(__file__).resolve().parents[1] / "data" / "skills",
        runtime_events_enabled=True,
        max_worker_concurrency=2,
    )
    kernel = RuntimeKernel(settings, providers=None, stores=None)
    session = _session()

    kernel._run_post_turn_memory_maintenance(session, latest_text="Remember favorite_editor=Neovim")

    assert (
        kernel.file_memory_store.get(
            tenant_id="tenant",
            user_id="user",
            conversation_id="conversation",
            scope="conversation",
            key="favorite_editor",
        )
        == "Neovim"
    )


def test_memory_disabled_turns_off_runtime_memory_maintenance(tmp_path: Path) -> None:
    settings = _kernel_settings(tmp_path, memory_enabled=False)
    kernel = RuntimeKernel(settings, providers=None, stores=None)
    session = _session()

    kernel._run_post_turn_memory_maintenance(session, latest_text="Remember favorite_editor=Neovim")

    assert kernel.file_memory_store is None
    assert kernel.memory_extractor is None
    assert kernel.transcript_store.load_session_events(session.session_id) == []


def test_memory_tool_builder_returns_no_tools_when_feature_disabled(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    session = _session()
    context = ToolContext(
        settings=SimpleNamespace(memory_enabled=False),
        providers=None,
        stores=None,
        session=session,
        paths=paths,
        file_memory_store=FileMemoryStore(paths),
    )

    assert build_memory_tools(context) == []


def test_memory_tools_use_managed_store_when_available(tmp_path: Path) -> None:
    class _FakeManagedStore:
        def __init__(self) -> None:
            self.values: dict[tuple[str, str], str] = {}
            self.saved_scopes: list[str] = []

        def save_explicit(
            self,
            *,
            tenant_id: str,
            user_id: str,
            conversation_id: str,
            session_id: str,
            scope: str,
            key: str,
            value: str,
            source: str = "",
            evidence_turn_ids: list[str] | None = None,
        ) -> SimpleNamespace:
            del tenant_id, user_id, conversation_id, session_id, source, evidence_turn_ids
            self.values[(scope, key)] = value
            self.saved_scopes.append(scope)
            return SimpleNamespace(memory_id="mem_1")

        def load_value(
            self,
            *,
            tenant_id: str,
            user_id: str,
            conversation_id: str,
            session_id: str,
            scope: str,
            key: str,
        ) -> str | None:
            del tenant_id, user_id, conversation_id, session_id
            return self.values.get((scope, key))

        def list_keys(
            self,
            *,
            tenant_id: str,
            user_id: str,
            conversation_id: str,
            session_id: str,
            scope: str,
        ) -> list[str]:
            del tenant_id, user_id, conversation_id, session_id
            return sorted(key for stored_scope, key in self.values if stored_scope == scope)

    paths = _paths(tmp_path)
    session = _session()
    session.append_message("user", "Remember preferred_name=Shiv")
    managed_store = _FakeManagedStore()
    file_store = FileMemoryStore(paths)
    context = ToolContext(
        settings=SimpleNamespace(memory_enabled=True),
        providers=None,
        stores=None,
        session=session,
        paths=paths,
        file_memory_store=file_store,
        memory_store=managed_store,
    )
    tool_map = {tool.name: tool for tool in build_memory_tools(context)}

    save_result = tool_map["memory_save"].invoke({"key": "preferred_name", "value": "Shiv", "scope": "user"})
    load_result = tool_map["memory_load"].invoke({"key": "preferred_name", "scope": "user"})
    list_result = tool_map["memory_list"].invoke({"scope": "user"})

    assert "Saved memory in user scope" in save_result
    assert load_result == "Shiv"
    assert list_result == "preferred_name"
    assert managed_store.saved_scopes == ["user"]
    assert (
        file_store.get(
            tenant_id="tenant",
            user_id="user",
            conversation_id="conversation",
            scope="user",
            key="preferred_name",
        )
        is None
    )


def test_spawn_worker_rejects_memory_maintainer_when_memory_is_disabled(tmp_path: Path) -> None:
    settings = _kernel_settings(tmp_path, memory_enabled=False)
    kernel = RuntimeKernel(settings, providers=None, stores=None)
    session = _session()
    general = kernel.registry.get("general")
    assert general is not None
    tool_context = ToolContext(
        settings=settings,
        providers=None,
        stores=None,
        session=session,
        paths=kernel.paths,
        kernel=kernel,
        active_definition=general,
    )

    result = kernel.spawn_worker_from_tool(
        tool_context,
        prompt="Store a few facts in memory.",
        agent_name="memory_maintainer",
        run_in_background=True,
    )

    assert "MEMORY_ENABLED" in str(result.get("error") or "")
