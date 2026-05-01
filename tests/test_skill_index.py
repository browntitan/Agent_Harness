from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agentic_chatbot_next.persistence.postgres.skills import SkillChunkMatch, SkillPackRecord
from agentic_chatbot_next.skills.dependency_graph import build_skill_dependency_graph
from agentic_chatbot_next.skills.indexer import SkillContextResolver, SkillIndexSync
from agentic_chatbot_next.skills.pack_loader import load_skill_pack_from_file
from agentic_chatbot_next.tools.skills_search_tool import make_skills_search_tool


def test_load_skill_pack_from_file_parses_metadata_and_chunks(tmp_path: Path):
    skill_file = tmp_path / "comparison.md"
    skill_file.write_text(
        "# Comparison Workflow\n"
        "agent_scope: rag\n"
        "tool_tags: diff_documents, compare_clauses\n"
        "task_tags: comparison, diff\n"
        "version: 2\n"
        "enabled: true\n"
        "description: Compare documents carefully.\n\n"
        'retrieval_profile: comparison_campaign\n'
        'controller_hints: {"compare_across_documents": true}\n'
        "coverage_goal: cross_document\n"
        "result_mode: comparison\n\n"
        "## Workflow\n"
        "Analyze each document independently before synthesizing.\n"
    )

    pack = load_skill_pack_from_file(skill_file, root=tmp_path)

    assert pack.name == "Comparison Workflow"
    assert pack.agent_scope == "rag"
    assert pack.tool_tags == ["compare_indexed_docs"]
    assert pack.task_tags == ["comparison", "diff"]
    assert pack.version == "2"
    assert pack.enabled is True
    assert pack.retrieval_profile == "comparison_campaign"
    assert pack.controller_hints == {"compare_across_documents": True}
    assert pack.coverage_goal == "cross_document"
    assert pack.result_mode == "comparison"
    assert "## Workflow" in pack.body_markdown
    assert pack.version_parent == pack.skill_id
    assert pack.chunks


def test_load_skill_pack_supports_frontmatter_and_normalizes_legacy_tool_tags(tmp_path: Path):
    skill_file = tmp_path / "legacy.md"
    skill_file.write_text(
        "---\n"
        "name: Legacy Resolution\n"
        "agent_scope: rag\n"
        "tool_tags: resolve_document, search_document\n"
        "task_tags: recovery, lookup\n"
        "version: 2\n"
        "enabled: true\n"
        "description: Recover legacy tool naming.\n"
        "keywords: legacy, aliases\n"
        "when_to_apply: Use when old tool names appear.\n"
        "avoid_when: Avoid leaving them unnormalized.\n"
        "examples: legacy lookup\n"
        "---\n"
        "# Legacy Resolution\n\n"
        "Use the modern tool vocabulary.\n",
        encoding="utf-8",
    )

    pack = load_skill_pack_from_file(skill_file, root=tmp_path)

    assert pack.tool_tags == ["resolve_indexed_docs", "read_indexed_doc"]
    assert pack.keywords == ["legacy", "aliases"]
    assert pack.when_to_apply == "Use when old tool names appear."
    assert pack.examples == ["legacy lookup"]
    assert pack.warnings


def test_load_executable_skill_parses_execution_frontmatter_and_indexes_discovery_card(tmp_path: Path):
    skill_file = tmp_path / "review.md"
    skill_file.write_text(
        "---\n"
        "name: Review Skill\n"
        "kind: executable\n"
        "agent_scope: general\n"
        "version: 1\n"
        "enabled: true\n"
        "description: Run a focused review.\n"
        "allowed-tools: calculator, search_skills\n"
        "context: fork\n"
        "agent: utility\n"
        "model: gpt-test\n"
        "effort: high\n"
        "max_steps: 3\n"
        "max_tool_calls: 4\n"
        "input_schema: {\"type\":\"object\"}\n"
        "---\n"
        "# Review Skill\n\n"
        "Use {{input}} and {{ARGUMENTS_JSON}}.\n",
        encoding="utf-8",
    )

    pack = load_skill_pack_from_file(skill_file, root=tmp_path)

    assert pack.kind == "executable"
    assert pack.tool_tags == []
    assert pack.execution_config["allowed_tools"] == ["calculator", "search_skills"]
    assert pack.execution_config["context"] == "fork"
    assert pack.execution_config["agent"] == "utility"
    assert pack.execution_config["model"] == "gpt-test"
    assert pack.execution_config["effort"] == "high"
    assert pack.execution_config["max_steps"] == 3
    assert pack.execution_config["max_tool_calls"] == 4
    assert "Run a focused review" in pack.chunks[0]
    assert "Use {{input}}" not in pack.chunks[0]


class _FakeSkillStore:
    def __init__(self) -> None:
        self.records = {
            "pinned-workflow": SkillPackRecord(
                skill_id="pinned-workflow",
                name="Pinned Workflow",
                agent_scope="general",
                checksum="abc123",
                tenant_id="local-dev",
                body_markdown="Always run the pinned checklist first.",
                status="active",
                enabled=True,
                version_parent="pinned-workflow",
            )
        }

    def list_skill_packs(self, *, tenant_id="local-dev", agent_scope="", enabled_only=False, owner_user_id="", visibility="", status="", graph_id=""):
        del owner_user_id, visibility, graph_id
        rows = []
        for record in self.records.values():
            if record.tenant_id != tenant_id:
                continue
            if agent_scope and record.agent_scope != agent_scope:
                continue
            if enabled_only and (not record.enabled or record.status != "active"):
                continue
            if status and record.status != status:
                continue
            rows.append(record)
        return rows

    def vector_search(self, query, *, tenant_id, top_k, agent_scope, tool_tags=None, task_tags=None, enabled_only=True, owner_user_id="", graph_ids=None):
        del graph_ids
        return [
            SkillChunkMatch(
                skill_id="rag-comparison",
                name="Comparison Workflow",
                agent_scope=agent_scope or "rag",
                content="Use diff_documents before compare_clauses.",
                chunk_index=0,
                score=0.95,
                tool_tags=["diff_documents"],
                task_tags=["comparison"],
                retrieval_profile="comparison_campaign",
                controller_hints={"compare_across_documents": True},
                coverage_goal="cross_document",
                result_mode="comparison",
                version_parent="rag-comparison",
            )
        ]

    def get_skill_packs_by_ids(self, skill_ids, *, tenant_id="local-dev", owner_user_id=""):
        del owner_user_id
        return [
            record
            for skill_id in skill_ids
            if (record := self.records.get(skill_id)) is not None and record.tenant_id == tenant_id
        ]

    def get_skill_chunks(self, skill_id, *, tenant_id="local-dev"):
        record = self.records.get(skill_id)
        if record is None or record.tenant_id != tenant_id:
            return []
        return [
            {
                "skill_chunk_id": f"{skill_id}#chunk0000",
                "skill_id": skill_id,
                "chunk_index": 0,
                "content": record.body_markdown,
            }
        ]


class _SyncSkillStore:
    def __init__(self, records: list[SkillPackRecord]) -> None:
        self.records = {record.skill_id: record for record in records}
        self.chunks: dict[str, list[str]] = {}

    def upsert_skill_pack(self, record: SkillPackRecord, chunks: list[str]) -> None:
        self.records[record.skill_id] = record
        self.chunks[record.skill_id] = list(chunks)

    def get_skill_pack(self, skill_id: str, *, tenant_id: str = "local-dev") -> SkillPackRecord | None:
        record = self.records.get(skill_id)
        if record is None or record.tenant_id != tenant_id:
            return None
        return record

    def list_skill_packs(
        self,
        *,
        tenant_id: str = "local-dev",
        agent_scope: str = "",
        enabled_only: bool = False,
        owner_user_id: str = "",
        visibility: str = "",
        status: str = "",
        graph_id: str = "",
    ) -> list[SkillPackRecord]:
        del owner_user_id, visibility
        rows: list[SkillPackRecord] = []
        for record in self.records.values():
            if record.tenant_id != tenant_id:
                continue
            if agent_scope and record.agent_scope != agent_scope:
                continue
            if graph_id and record.graph_id != graph_id:
                continue
            if enabled_only and (not record.enabled or record.status != "active"):
                continue
            if status and record.status != status:
                continue
            rows.append(record)
        return rows

    def set_skill_status(
        self,
        skill_id: str,
        *,
        tenant_id: str = "local-dev",
        status: str,
        enabled: bool | None = None,
    ) -> None:
        record = self.records.get(skill_id)
        if record is None or record.tenant_id != tenant_id:
            return
        record.status = status
        if enabled is not None:
            record.enabled = bool(enabled)


def _write_seeded_skill(root: Path, relative_path: str, *, name: str = "Current Guidance") -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"name: {name}\n"
        "skill_id: rag-current-guidance\n"
        "agent_scope: rag\n"
        "tool_tags: search_indexed_docs, rag_agent_tool\n"
        "task_tags: retrieval, grounding\n"
        "version: 1\n"
        "enabled: true\n"
        "description: General seeded retrieval guidance.\n"
        "keywords: retrieval, grounding\n"
        "when_to_apply: Use for grounded lookup.\n"
        "avoid_when: Avoid when no evidence is needed.\n"
        "examples: scoped lookup\n"
        "---\n"
        "# Current Guidance\n\n"
        "Search the indexed evidence before answering.\n",
        encoding="utf-8",
    )
    return path


def test_skill_index_sync_archives_removed_seeded_files_without_touching_operator_skills(tmp_path: Path) -> None:
    for method_name in ("sync", "sync_changed"):
        root = tmp_path / method_name / "skill_packs"
        _write_seeded_skill(root, "rag/current_guidance.md")
        store = _SyncSkillStore(
            [
                SkillPackRecord(
                    skill_id="rag-removed-guidance",
                    name="Removed Seeded Guidance",
                    agent_scope="rag",
                    checksum="old",
                    tenant_id="local-dev",
                    source_path="rag/removed_guidance.md",
                    status="active",
                    enabled=True,
                ),
                SkillPackRecord(
                    skill_id="operator-api-skill",
                    name="Operator API Skill",
                    agent_scope="rag",
                    checksum="operator",
                    tenant_id="local-dev",
                    source_path="api://skills/operator-api-skill.md",
                    status="active",
                    enabled=True,
                ),
                SkillPackRecord(
                    skill_id="operator-graph-skill",
                    name="Operator Graph Skill",
                    agent_scope="rag",
                    checksum="graph",
                    tenant_id="local-dev",
                    graph_id="operator_graph",
                    source_path="",
                    status="active",
                    enabled=True,
                ),
            ]
        )
        syncer = SkillIndexSync(
            SimpleNamespace(skill_packs_dir=root),
            SimpleNamespace(skill_store=store),
        )

        result = getattr(syncer, method_name)(tenant_id="local-dev")

        assert result["archived_removed_seeded"] == [
            {"skill_id": "rag-removed-guidance", "source_path": "rag/removed_guidance.md"}
        ]
        assert result["archived_removed_seeded_count"] == 1
        assert store.records["rag-removed-guidance"].status == "archived"
        assert store.records["rag-removed-guidance"].enabled is False
        assert store.records["operator-api-skill"].status == "active"
        assert store.records["operator-api-skill"].enabled is True
        assert store.records["operator-graph-skill"].status == "active"
        assert store.records["operator-graph-skill"].enabled is True
        assert "rag-current-guidance" in store.records


def test_search_skills_prefers_db_backed_matches():
    settings = SimpleNamespace(
        default_tenant_id="local-dev",
        skill_search_top_k=4,
        skill_context_max_chars=4000,
    )
    stores = SimpleNamespace(skill_store=_FakeSkillStore())
    tool = make_skills_search_tool(settings, stores=stores)

    result = tool.invoke({"query": "how to compare two documents", "agent_filter": "rag_agent", "top_k": 1})

    assert "Comparison Workflow" in result
    assert "diff_documents" in result


def test_skill_context_resolver_prepends_pinned_skill_packs():
    settings = SimpleNamespace(
        skill_search_top_k=4,
        skill_context_max_chars=4000,
    )
    stores = SimpleNamespace(skill_store=_FakeSkillStore())
    resolver = SkillContextResolver(settings, stores)

    context = resolver.resolve(
        query="how to compare two documents",
        tenant_id="local-dev",
        agent_scope="general",
        pinned_skill_ids=["pinned-workflow"],
    )

    assert context.matches[0].skill_id == "pinned-workflow"
    assert context.text.startswith("[Pinned: Pinned Workflow | pinned-workflow]")
    assert "Comparison Workflow" in context.text


def test_dependency_graph_detects_missing_dependencies_and_dependents():
    records = [
        SkillPackRecord(
            skill_id="skill-a",
            version_parent="skill-a",
            name="Parent",
            agent_scope="general",
            checksum="a",
            tenant_id="local-dev",
            status="active",
            enabled=True,
        ),
        SkillPackRecord(
            skill_id="skill-b",
            version_parent="skill-b",
            name="Child",
            agent_scope="general",
            checksum="b",
            tenant_id="local-dev",
            status="active",
            enabled=True,
            controller_hints={"depends_on_skills": ["missing-family"]},
        ),
        SkillPackRecord(
            skill_id="skill-c",
            version_parent="skill-c",
            name="Grandchild",
            agent_scope="general",
            checksum="c",
            tenant_id="local-dev",
            status="active",
            enabled=True,
            controller_hints={"depends_on_skills": ["skill-b"]},
        ),
    ]

    graph = build_skill_dependency_graph(records)
    summary = graph.summary().to_dict()

    assert summary["valid"] is False
    assert summary["missing_dependencies"]["skill-b"] == ["missing-family"]
    assert "skill-c" in summary["blocked_families"]


def test_repo_skill_packs_cover_runtime_agent_scopes():
    root = Path(__file__).resolve().parents[1] / "data" / "skill_packs"
    scopes = {
        load_skill_pack_from_file(path, root=root).agent_scope
        for path in root.rglob("*.md")
    }

    assert {"rag", "general", "utility", "data_analyst", "planner", "finalizer", "verifier"} <= scopes
