from __future__ import annotations

import builtins
import json
import os
import subprocess
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from agentic_chatbot_next.graph.artifacts import load_artifact_bundle
from agentic_chatbot_next.graph.backend import GraphOperationResult, GraphQueryHit, MicrosoftGraphRagBackend
from agentic_chatbot_next.graph.community_report_recovery import (
    analyze_community_report_inputs,
    generate_fallback_community_reports,
)
import agentic_chatbot_next.graph.prompt_tuning as prompt_tuning_module
from agentic_chatbot_next.graph.prompt_tuning import COMMON_GRAPHRAG_PROMPT_TARGETS, GraphPromptTuningService
from agentic_chatbot_next.graph.service import GraphService
from agentic_chatbot_next.persistence.postgres.graphs import (
    GraphIndexRecord,
    GraphIndexRunRecord,
    GraphIndexSourceRecord,
)


class _FakeDocStore:
    def __init__(self, records):
        self.records = list(records)

    def get_document(self, doc_id, tenant_id="local-dev"):
        del tenant_id
        for record in self.records:
            if record.doc_id == doc_id:
                return record
        return None

    def list_documents(self, *, tenant_id="local-dev", collection_id="", source_type=""):
        del tenant_id, source_type
        if collection_id:
            return [record for record in self.records if record.collection_id == collection_id]
        return list(self.records)

    def fuzzy_search_title(self, query, tenant_id, *, limit=6, collection_id=""):
        del tenant_id
        lowered = str(query).lower()
        terms = [term for term in lowered.split() if len(term) > 2]
        matches = []
        for record in self.list_documents(collection_id=collection_id):
            title = record.title.lower()
            if lowered in title or any(term in title for term in terms):
                matches.append(
                    {
                        "doc_id": record.doc_id,
                        "title": record.title,
                        "source_type": record.source_type,
                        "doc_structure_type": getattr(record, "doc_structure_type", ""),
                        "score": 0.9,
                    }
                )
            if len(matches) >= limit:
                break
        return matches


class _FakeGraphIndexStore:
    def __init__(self):
        self.records: dict[str, GraphIndexRecord] = {}

    def upsert_index(self, record: GraphIndexRecord) -> None:
        self.records[record.graph_id] = replace(record)

    def _visible(self, record: GraphIndexRecord, user_id: str = "") -> bool:
        if record.visibility == "private":
            return bool(str(user_id or "").strip()) and record.owner_admin_user_id == str(user_id or "").strip()
        return True

    def get_index(self, graph_id: str, tenant_id: str, user_id: str = ""):
        record = self.records.get(graph_id)
        if record is None or record.tenant_id != tenant_id or not self._visible(record, user_id):
            return None
        return replace(record)

    def list_indexes(self, *, tenant_id: str, user_id: str = "", collection_id: str = "", limit: int = 100):
        rows = [
            replace(record)
            for record in self.records.values()
            if (
                record.tenant_id == tenant_id
                and self._visible(record, user_id)
                and (not collection_id or record.collection_id == collection_id)
            )
        ]
        return rows[:limit]

    def search_indexes(self, query: str, *, tenant_id: str, user_id: str = "", collection_id: str = "", limit: int = 6):
        lowered = str(query).lower()
        query_terms = [term for term in lowered.split() if len(term) > 2]
        rows = []
        for record in self.records.values():
            if record.tenant_id != tenant_id:
                continue
            if not self._visible(record, user_id):
                continue
            if collection_id and record.collection_id != collection_id:
                continue
            haystack = " ".join(
                [
                    record.graph_id,
                    record.display_name,
                    record.domain_summary,
                    " ".join(record.entity_samples),
                    " ".join(record.relationship_samples),
                ]
            ).lower()
            if lowered and lowered not in haystack and not any(term in haystack for term in query_terms):
                continue
            rows.append(replace(record))
            if len(rows) >= limit:
                break
        return rows

    def update_index_status(self, graph_id: str, tenant_id: str, *, status: str, health=None):
        record = self.records.get(graph_id)
        if record is None or record.tenant_id != tenant_id:
            return False
        self.records[graph_id] = replace(record, status=status, health=dict(health or record.health))
        return True

    def delete_index(self, graph_id: str, tenant_id: str):
        record = self.records.get(graph_id)
        if record is None or record.tenant_id != tenant_id:
            return {"graph_indexes": 0}
        del self.records[graph_id]
        return {
            "entity_mentions": 0,
            "canonical_entities": 0,
            "skills": 0,
            "auth_role_permissions": 0,
            "graph_indexes": 1,
        }


class _FakeGraphIndexSourceStore:
    def __init__(self):
        self.records: dict[tuple[str, str], list[GraphIndexSourceRecord]] = {}

    def replace_sources(self, graph_id: str, *, tenant_id: str, sources):
        self.records[(tenant_id, graph_id)] = [replace(item) for item in sources]

    def list_sources(self, graph_id: str, *, tenant_id: str = "local-dev", limit: int = 100):
        return list(self.records.get((tenant_id, graph_id), []))[:limit]


class _FakeGraphIndexRunStore:
    def __init__(self):
        self.records: dict[str, list[GraphIndexRunRecord]] = {}

    def upsert_run(self, record: GraphIndexRunRecord) -> None:
        if not str(record.started_at or "").strip():
            record = replace(record, started_at=datetime.now(timezone.utc).isoformat())
        bucket = self.records.setdefault(record.graph_id, [])
        for index, existing in enumerate(bucket):
            if existing.run_id == record.run_id:
                bucket[index] = replace(record)
                return
        bucket.append(replace(record))

    def list_runs(self, graph_id: str, *, tenant_id: str = "local-dev", limit: int = 20):
        rows = [replace(item) for item in self.records.get(graph_id, []) if item.tenant_id == tenant_id]
        rows.sort(key=lambda item: str(item.started_at or ""), reverse=True)
        return rows[:limit]

    def list_runs_by_status(
        self,
        *,
        tenant_id: str = "local-dev",
        status: str = "",
        graph_id: str = "",
        limit: int = 100,
    ):
        rows = [
            replace(item)
            for runs in self.records.values()
            for item in runs
            if item.tenant_id == tenant_id
            and (not status or item.status == status)
            and (not graph_id or item.graph_id == graph_id)
        ]
        rows.sort(key=lambda item: str(item.started_at or ""), reverse=True)
        return rows[:limit]

    def delete_run(self, run_id: str, *, tenant_id: str = "local-dev", graph_id: str = "", statuses=None) -> int:
        allowed = set(statuses or [])
        deleted = 0
        for key, runs in list(self.records.items()):
            kept = []
            for run in runs:
                matches = (
                    run.tenant_id == tenant_id
                    and run.run_id == run_id
                    and (not graph_id or run.graph_id == graph_id)
                    and (not allowed or run.status in allowed)
                )
                if matches:
                    deleted += 1
                else:
                    kept.append(run)
            self.records[key] = kept
        return deleted


class _FakeGraphQueryCacheStore:
    def __init__(self):
        self.payloads = {}

    def get_cached(self, *, graph_id: str, tenant_id: str, query_text: str, query_method: str):
        return self.payloads.get((tenant_id, graph_id, query_text, query_method))

    def put_cached(self, *, graph_id: str, tenant_id: str, query_text: str, query_method: str, response_json, ttl_seconds: int):
        del ttl_seconds
        self.payloads[(tenant_id, graph_id, query_text, query_method)] = SimpleNamespace(response_json=dict(response_json))


class _FakeGraphStore:
    def local_search(self, query, *, tenant_id, limit=8, doc_ids=None):
        del tenant_id, limit
        return [
            SimpleNamespace(
                doc_id="DOC-1",
                chunk_ids=["chunk-1"],
                score=0.88,
                title=f"{query} evidence",
                source_path="/tmp/roadmap.md",
                source_type="kb",
                relationship_path=["Roadmap", "Release"],
                summary="Linked roadmap evidence",
                metadata={"backend": "neo4j"},
            )
        ] if "release" in str(query).lower() and (not doc_ids or "DOC-1" in set(doc_ids)) else []

    def global_search(self, query, *, tenant_id, limit=8, doc_ids=None):
        return self.local_search(query, tenant_id=tenant_id, limit=limit, doc_ids=doc_ids)


class _FakeBackend:
    backend_name = "microsoft_graphrag"
    supported_query_methods = ("local", "global", "drift")

    def index_documents(self, graph_id: str, root_path: Path, *, refresh: bool = False):
        del graph_id, refresh
        return GraphOperationResult(
            status="ready",
            detail="Indexed successfully.",
            capabilities=["catalog", "graphrag_cli"],
            supported_query_methods=["local", "global", "drift"],
            artifact_path=str(root_path / "output"),
        )

    def import_existing_graph(self, graph_id: str, root_path: Path, *, artifact_path: str = "", metadata=None):
        del graph_id, root_path, metadata
        return GraphOperationResult(
            status="ready",
            detail="Imported successfully.",
            capabilities=["catalog", "artifact_registration"],
            supported_query_methods=["local", "global"],
            artifact_path=artifact_path,
        )

    def query_index(self, graph_id: str, root_path: Path, *, query: str, method: str, limit: int, doc_ids=None):
        del graph_id, root_path, query, method, limit, doc_ids
        return []


class _AsyncBackend(_FakeBackend):
    def __init__(self, *, final_status: str = "ready") -> None:
        self.final_status = final_status
        self.sync_calls = 0

    def launch_index_process(self, graph_id: str, root_path: Path, *, refresh: bool = False, run_id: str = ""):
        del graph_id, refresh
        logs_dir = root_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        state_path = logs_dir / f"{run_id or 'graph'}_job_state.json"
        stream_log_path = logs_dir / f"{run_id or 'graph'}_job_stream.log"
        runner_log_path = logs_dir / f"{run_id or 'graph'}_runner.log"
        return GraphOperationResult(
            status="running",
            detail="Started GraphRAG index in the background.",
            capabilities=["catalog", "graphrag_cli"],
            supported_query_methods=["local", "global", "drift"],
            artifact_path=str(root_path),
            metadata={
                "run_mode": "background",
                "active_pid": 4321,
                "active_process_group_id": 4321,
                "state_path": str(state_path),
                "stream_log_path": str(stream_log_path),
                "runner_log_path": str(runner_log_path),
                "command": ["graphrag", "index", "--root", str(root_path)],
            },
        )

    def collect_job_result(self, graph_id: str, root_path: Path, *, state: dict[str, object]):
        del graph_id, state
        return GraphOperationResult(
            status=self.final_status,
            detail="Background GraphRAG build finished.",
            capabilities=["catalog", "graph_store_fallback", "graphrag_cli"],
            supported_query_methods=["local", "global", "drift"],
            artifact_path=str(root_path / "output"),
            query_ready=self.final_status == "ready",
            query_backend="graphrag_artifacts" if self.final_status == "ready" else "",
            artifact_tables=["documents", "text_units"] if self.final_status == "ready" else [],
            metadata={"run_mode": "background", "returncode": 0},
        )

    def index_documents(self, graph_id: str, root_path: Path, *, refresh: bool = False):
        del graph_id, root_path, refresh
        self.sync_calls += 1
        raise AssertionError("background backend should not fall back to synchronous indexing")


class _FakeSkillStore:
    def __init__(self, records=None):
        self.records = list(records or [])

    def list_skill_packs(self, *, tenant_id="local-dev", owner_user_id="", graph_id="", **kwargs):
        del kwargs
        matches = []
        for record in self.records:
            if str(getattr(record, "tenant_id", "local-dev") or "local-dev") != tenant_id:
                continue
            if graph_id and str(getattr(record, "graph_id", "") or "") != graph_id:
                continue
            visibility = str(getattr(record, "visibility", "tenant") or "tenant")
            if visibility == "private" and str(getattr(record, "owner_user_id", "") or "") != str(owner_user_id or ""):
                continue
            matches.append(record)
        return matches


def _make_settings(tmp_path: Path):
    data_dir = tmp_path / "data"
    return SimpleNamespace(
        data_dir=data_dir,
        graphrag_projects_dir=data_dir / "graphrag" / "projects",
        default_tenant_id="local-dev",
        default_collection_id="default",
        graph_backend="microsoft_graphrag",
        graph_search_enabled=True,
        graph_source_planning_enabled=True,
        graph_sql_enabled=True,
        graph_sql_allowed_views=("documents", "graph_indexes"),
        graphrag_llm_provider="openai",
        graphrag_base_url="http://localhost:11434/v1",
        graphrag_api_key="ollama",
        graphrag_chat_model="nemotron-cascade-2:30b",
        graphrag_index_chat_model="nemotron-cascade-2:30b",
        graphrag_community_report_mode="text",
        graphrag_community_report_chat_model="nemotron-cascade-2:30b",
        graphrag_embed_model="nomic-embed-text:latest",
        graphrag_concurrency=1,
        graphrag_request_timeout_seconds=7200,
        graphrag_index_request_timeout_seconds=900,
        graphrag_community_report_request_timeout_seconds=300,
        graphrag_community_report_max_input_length=4000,
        graphrag_community_report_max_length=1200,
        graphrag_job_timeout_seconds=21600,
        graphrag_timeout_seconds=7200,
        graphrag_stale_run_after_seconds=1800,
        graphrag_default_query_method="local",
        graph_query_cache_ttl_seconds=900,
    )


def _make_doc(doc_id: str, title: str, *, collection_id: str = "default", source_path: str = "/tmp/doc.md"):
    return SimpleNamespace(
        doc_id=doc_id,
        title=title,
        collection_id=collection_id,
        source_path=source_path,
        source_type="kb",
        doc_structure_type="text",
    )


def _make_service(tmp_path: Path, *, user_id: str = "user", skill_store=None) -> GraphService:
    settings = _make_settings(tmp_path)
    docs = [
        _make_doc("DOC-1", "Release Readiness", source_path="/tmp/release.md"),
        _make_doc("DOC-2", "Dependency Map", source_path="/tmp/dependency.md"),
    ]
    stores = SimpleNamespace(
        doc_store=_FakeDocStore(docs),
        graph_index_store=_FakeGraphIndexStore(),
        graph_index_source_store=_FakeGraphIndexSourceStore(),
        graph_index_run_store=_FakeGraphIndexRunStore(),
        graph_query_cache_store=_FakeGraphQueryCacheStore(),
        graph_store=_FakeGraphStore(),
        skill_store=skill_store,
    )
    service = GraphService(
        settings,
        stores,
        session=SimpleNamespace(tenant_id="local-dev", conversation_id="conv", user_id=user_id, metadata={}),
    )
    service._backend_for = lambda backend_name: _FakeBackend()
    return service


def _make_tuning_services(tmp_path: Path):
    settings = _make_settings(tmp_path)
    release_path = tmp_path / "release.md"
    dependency_path = tmp_path / "dependency.md"
    release_path.write_text(
        "Release Readiness requires approval by the Change Advisory Board. "
        "The Deployment Service depends on the Notification API and the Incident Runbook. "
        "Security exceptions must be owned by the Release Manager.",
        encoding="utf-8",
    )
    dependency_path.write_text(
        "The Dependency Map lists vendors, internal teams, APIs, and escalation owners. "
        "Payments API uses the Identity Service. The Vendor Risk policy governs suppliers.",
        encoding="utf-8",
    )
    docs = [
        _make_doc("DOC-1", "Release Readiness", source_path=str(release_path)),
        _make_doc("DOC-2", "Dependency Map", source_path=str(dependency_path)),
    ]
    stores = SimpleNamespace(
        doc_store=_FakeDocStore(docs),
        graph_index_store=_FakeGraphIndexStore(),
        graph_index_source_store=_FakeGraphIndexSourceStore(),
        graph_index_run_store=_FakeGraphIndexRunStore(),
        graph_query_cache_store=_FakeGraphQueryCacheStore(),
        graph_store=_FakeGraphStore(),
        skill_store=None,
    )
    session = SimpleNamespace(tenant_id="local-dev", conversation_id="conv", user_id="user", metadata={})
    graph_service = GraphService(settings, stores, session=session)
    graph_service._backend_for = lambda backend_name: _FakeBackend()
    tuning_service = GraphPromptTuningService(settings, stores, session=session)
    tuning_service.graph_service._backend_for = lambda backend_name: _FakeBackend()
    return graph_service, tuning_service, stores


def _write_phase_1_orphan_artifacts(root: Path) -> None:
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "id": ["ent-1"],
                "title": ["Release Entity"],
                "type": ["organization"],
                "description": ["Release dependency owner"],
                "text_unit_ids": [["tu-1"]],
                "frequency": [1],
                "degree": [3],
            }
        ),
        output_dir / "entities.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "id": ["tu-1"],
                "human_readable_id": [1],
                "text": ["Vendor Acme depends on finance approval for release."],
                "n_tokens": [24],
                "document_id": ["DOC-1"],
                "entity_ids": [["ent-1"]],
                "relationship_ids": [[]],
                "covariate_ids": [[]],
            }
        ),
        output_dir / "text_units.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "id": ["community-1", "community-2"],
                "human_readable_id": [1, 2],
                "community": [1, 2],
                "level": [0, 1],
                "parent": [None, None],
                "children": [[], []],
                "title": ["Community 1", "Community 2"],
                "entity_ids": [["ent-1"], []],
                "relationship_ids": [[], []],
                "text_unit_ids": [["tu-1"], ["tu-1"]],
                "period": ["2026Q2", "2026Q2"],
                "size": [1, 1],
            }
        ),
        output_dir / "communities.parquet",
    )


def test_graph_service_indexes_documents_into_catalog(tmp_path: Path):
    service = _make_service(tmp_path)

    payload = service.index_corpus(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1", "DOC-2"],
    )

    assert payload["status"] == "ready"
    assert payload["graph_id"] == "release_graph"
    assert (service._graph_settings_path("release_graph")).exists()
    assert sorted(path.name for path in service._graph_input_dir("release_graph").glob("*.txt")) == [
        "001_release.txt",
        "002_dependency.txt",
    ]
    inspected = service.inspect_index("release_graph")
    assert inspected["graph"]["display_name"] == "Release Graph"
    assert inspected["graph"]["backend"] == "microsoft_graphrag"
    assert inspected["graph"]["supported_query_methods"] == ["local", "global", "drift"]
    assert {item["source_doc_id"] for item in inspected["sources"]} == {"DOC-1", "DOC-2"}
    assert inspected["runs"][0]["status"] == "ready"


def test_graph_service_renders_local_ollama_graphrag_settings(tmp_path: Path):
    service = _make_service(tmp_path)

    settings_path = service._render_settings_yaml("release_graph")
    payload = yaml.safe_load(Path(settings_path).read_text(encoding="utf-8"))

    assert payload["concurrent_requests"] == 1
    assert payload["completion_models"]["default_completion_model"]["model"] == "nemotron-cascade-2:30b"
    assert payload["completion_models"]["index_completion_model"]["model"] == "nemotron-cascade-2:30b"
    assert payload["completion_models"]["community_report_completion_model"]["model"] == "nemotron-cascade-2:30b"
    assert payload["completion_models"]["default_completion_model"]["call_args"]["temperature"] == 0
    assert payload["completion_models"]["index_completion_model"]["call_args"]["timeout"] == 900
    assert payload["completion_models"]["community_report_completion_model"]["call_args"]["timeout"] == 300
    assert payload["embedding_models"]["default_embedding_model"]["call_args"]["timeout"] == 7200
    assert payload["embedding_models"]["default_embedding_model"]["model"] == "nomic-embed-text:latest"
    assert payload["chunking"]["size"] == 800
    assert payload["chunking"]["overlap"] == 80
    assert payload["vector_store"]["vector_size"] == 768
    assert payload["extract_graph"]["completion_model_id"] == "index_completion_model"
    assert payload["community_reports"]["completion_model_id"] == "community_report_completion_model"
    assert payload["community_reports"]["max_input_length"] == 4000
    assert payload["community_reports"]["max_length"] == 1200
    assert payload["workflows"] == [
        "load_input_documents",
        "create_base_text_units",
        "create_final_documents",
        "extract_graph",
        "finalize_graph",
        "extract_covariates",
        "create_communities",
        "create_final_text_units",
        "create_community_reports_text",
        "generate_text_embeddings",
    ]


def test_graph_service_keeps_gpt_oss_reasoning_low_for_local_ollama(tmp_path: Path):
    service = _make_service(tmp_path)
    service.settings.graphrag_index_chat_model = "gpt-oss:120b"

    settings_path = service._render_settings_yaml("release_graph")
    payload = yaml.safe_load(Path(settings_path).read_text(encoding="utf-8"))

    assert payload["completion_models"]["index_completion_model"]["call_args"]["reasoning_effort"] == "low"
    assert payload["completion_models"]["index_completion_model"]["call_args"]["reasoning"] == {"effort": "low"}


def test_graph_service_graph_mode_preserves_standard_workflow_list(tmp_path: Path):
    service = _make_service(tmp_path)
    service.settings.graphrag_community_report_mode = "graph"

    settings_path = service._render_settings_yaml("release_graph")
    payload = yaml.safe_load(Path(settings_path).read_text(encoding="utf-8"))

    assert "workflows" not in payload
    assert payload["community_reports"]["completion_model_id"] == "community_report_completion_model"


def test_graph_service_query_uses_catalog_shortlist_and_graph_store_fallback(tmp_path: Path):
    service = _make_service(tmp_path)
    service.index_corpus(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )

    payload = service.query_across_graphs(
        "Which release dependencies should we review?",
        collection_id="default",
        limit=5,
        top_k_graphs=2,
    )

    assert payload["graph_shortlist"]
    assert payload["graph_shortlist"][0]["graph_id"] == "release_graph"
    assert payload["results"][0]["doc_id"] == "DOC-1"
    assert payload["results"][0]["query_method"] == "local"


def test_graph_service_explain_source_plan_shortlists_graph_and_sql(tmp_path: Path):
    service = _make_service(tmp_path)
    service.index_corpus(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )

    plan = service.explain_source_plan(
        "What graphs exist for release relationships and which release document should I open first?",
        collection_id="default",
    )

    assert "graph" in plan["sources_chosen"]
    assert "sql" in plan["sources_chosen"]
    assert plan["graph_ids"] == ["release_graph"]
    assert "documents" in plan["sql_views_used"]


def test_graph_service_catalog_fallback_returns_doc_candidates_without_live_graph_store(tmp_path: Path):
    service = _make_service(tmp_path)
    service.stores.graph_store = None
    service.index_corpus(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )

    payload = service.query_index(
        "release_graph",
        query="release readiness",
        methods=["local"],
        limit=3,
    )

    assert payload["results"]
    assert payload["results"][0]["backend"] == "catalog"
    assert payload["results"][0]["doc_id"] == "DOC-1"
    assert payload["results"][0]["metadata"]["evidence_kind"] == "source_candidate"
    assert payload["evidence_status"] == "source_candidates_only"
    assert payload["requires_source_read"] is True
    assert "source candidates only" in payload["warnings"][0]
    assert payload["results"][0]["citation_ids"] == ["DOC-1#graph"]
    assert payload["citations"][0]["title"] == "Release Readiness"
    assert payload["citations"][0]["catalog_only"] is True
    assert "/v1/documents/DOC-1/source?" in payload["citations"][0]["url"]
    assert "disposition=inline" in payload["citations"][0]["url"]


def test_graph_service_normalizes_graph_method_alias_and_rejects_non_graph_methods(tmp_path: Path):
    service = _make_service(tmp_path)
    service.stores.graph_store = None
    service.index_corpus(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )
    record = service.stores.graph_index_store.records["release_graph"]
    service.stores.graph_index_store.records["release_graph"] = replace(
        record,
        query_ready=True,
        supported_query_methods=["local", "global", "drift"],
    )

    class _RecordingBackend(_FakeBackend):
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def query_index(self, graph_id: str, root_path: Path, *, query: str, method: str, limit: int, doc_ids=None):
            del graph_id, root_path, limit, doc_ids
            self.calls.append((method, query))
            return []

    backend = _RecordingBackend()
    service._backend_for = lambda backend_name: backend

    payload = service.query_index(
        "release_graph",
        query="release readiness",
        methods=["graph"],
        limit=3,
    )

    assert payload["methods"] == ["local", "global"]
    assert payload["method_aliases"] == {"graph": ["local", "global"]}
    assert [method for method, _query in backend.calls] == ["local", "global"]
    assert payload["evidence_status"] == "source_candidates_only"

    invalid = service.query_index(
        "release_graph",
        query="release readiness",
        methods=["vector"],
        limit=3,
    )

    assert invalid["results"] == []
    assert invalid["evidence_status"] == "method_error"
    assert "Unsupported graph query method(s): vector" in invalid["error"]
    assert invalid["supported_query_methods"] == ["local", "global", "drift"]


def test_graph_service_does_not_inject_hidden_corpus_specific_query_variants(tmp_path: Path):
    service = _make_service(tmp_path)
    service.stores.graph_store = None
    service.index_corpus(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )
    record = service.stores.graph_index_store.records["release_graph"]
    service.stores.graph_index_store.records["release_graph"] = replace(
        record,
        query_ready=True,
        supported_query_methods=["local"],
    )

    class _ExpansionBackend(_FakeBackend):
        def __init__(self) -> None:
            self.queries: list[str] = []

        def query_index(self, graph_id: str, root_path: Path, *, query: str, method: str, limit: int, doc_ids=None):
            del graph_id, root_path, method, limit, doc_ids
            self.queries.append(query)
            return []

    backend = _ExpansionBackend()
    service._backend_for = lambda backend_name: backend
    question = (
        "If someone says Blue Mica Wave 2 slipped because the hardware was bad, "
        "what is the better evidence-based answer?"
    )

    payload = service.query_index("release_graph", query=question, methods=["local"], limit=3)

    assert payload["expanded_queries"] == []
    assert backend.queries == [question]


def test_graph_service_auto_plans_rewrites_reranks_and_requires_source_read_for_hard_queries(tmp_path: Path):
    service = _make_service(tmp_path)
    service.stores.graph_store = None
    service.index_corpus(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )
    record = service.stores.graph_index_store.records["release_graph"]
    service.stores.graph_index_store.records["release_graph"] = replace(
        record,
        query_ready=True,
        supported_query_methods=["local", "global", "drift"],
        graph_context_summary={"entity_samples": ["release", "change", "risk"]},
    )

    class _PlanningBackend(_FakeBackend):
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def query_index(self, graph_id: str, root_path: Path, *, query: str, method: str, limit: int, doc_ids=None):
            del root_path, limit, doc_ids
            self.calls.append((method, query))
            if method != "global":
                return []
            return [
                GraphQueryHit(
                    graph_id=graph_id,
                    backend="graphrag_artifacts",
                    query_method=method,
                    doc_id="DOC-1",
                    chunk_ids=["chunk-final"],
                    score=0.42,
                    title="Release Change Approval",
                    source_path="/tmp/release.md",
                    source_type="kb",
                    relationship_path=["Approved change", "Cost increase", "Residual risk"],
                    summary=(
                        "The approved change directly supports one cost increase, while a separate "
                        "concern remains tracked as a residual risk pending source-text confirmation."
                    ),
                )
            ]

    backend = _PlanningBackend()
    service._backend_for = lambda backend_name: backend
    question = (
        "Which release cost increase is directly attributable to the approved change, "
        "and which concern remained a separate residual risk?"
    )

    payload = service.query_index("release_graph", query=question, methods=[], limit=3)

    assert payload["search_plan"]["planner"] == "heuristic"
    assert payload["methods"] == ["global", "local"]
    assert len(payload["expanded_queries"]) >= 2
    assert {method for method, _query in backend.calls} == {"global", "local"}
    assert payload["rerank"]["status"] == "disabled"
    assert payload["evidence_status"] == "grounded_graph_evidence"
    assert payload["requires_source_read"] is True
    assert payload["source_read_plan"]["required"] is True
    assert payload["source_read_plan"]["preferred_doc_ids"] == ["DOC-1"]
    assert set(payload["source_read_plan"]["missing_claim_slots"]) >= {
        "exact_amount_or_cost_component",
        "causal_support",
        "distinction_or_residual_risk",
    }


def test_graph_service_does_not_force_source_read_for_plain_relationship_queries(tmp_path: Path):
    service = _make_service(tmp_path)
    service.stores.graph_store = None
    service.index_corpus(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )
    record = service.stores.graph_index_store.records["release_graph"]
    service.stores.graph_index_store.records["release_graph"] = replace(
        record,
        query_ready=True,
        supported_query_methods=["local", "global"],
    )

    class _RelationshipBackend(_FakeBackend):
        def query_index(self, graph_id: str, root_path: Path, *, query: str, method: str, limit: int, doc_ids=None):
            del root_path, query, limit, doc_ids
            if method != "local":
                return []
            return [
                GraphQueryHit(
                    graph_id=graph_id,
                    backend="graphrag_artifacts",
                    query_method=method,
                    doc_id="DOC-1",
                    chunk_ids=["chunk-relationship"],
                    score=0.71,
                    title="Release Dependency Map",
                    relationship_path=["Release", "depends on", "Notification API"],
                    summary=(
                        "The graph relationship links the release to the Notification API dependency "
                        "and identifies it as the service dependency to review for readiness."
                    ),
                )
            ]

    service._backend_for = lambda backend_name: _RelationshipBackend()

    payload = service.query_index(
        "release_graph",
        query="Which services does the release depend on?",
        methods=[],
        limit=3,
    )

    assert payload["evidence_status"] == "grounded_graph_evidence"
    assert payload["requires_source_read"] is False
    assert "source_read_plan" not in payload


def test_graph_service_query_across_graphs_propagates_child_source_read_plan(tmp_path: Path):
    service = _make_service(tmp_path)
    service.stores.graph_store = None
    service.index_corpus(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )

    payload = service.query_across_graphs(
        "Which release cost increase was approved and which risk remains?",
        collection_id="default",
        limit=5,
        top_k_graphs=2,
    )

    assert payload["evidence_status"] == "source_candidates_only"
    assert payload["requires_source_read"] is True
    assert payload["source_read_plan"]["required"] is True
    assert payload["source_read_plan"]["preferred_doc_ids"] == ["DOC-1"]


def test_graph_service_enforces_private_visibility_across_list_inspect_and_query(tmp_path: Path):
    owner = _make_service(tmp_path, user_id="owner")
    created = owner.create_admin_graph(
        graph_id="private-release",
        display_name="Private Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
        visibility="private",
    )
    assert created["created"] is True

    viewer = GraphService(
        owner.settings,
        owner.stores,
        session=SimpleNamespace(tenant_id="local-dev", conversation_id="conv-2", user_id="viewer", metadata={}),
    )
    viewer._backend_for = lambda backend_name: _FakeBackend()

    assert [item["graph_id"] for item in owner.list_indexes()] == ["private_release"]
    assert viewer.list_indexes() == []
    assert viewer.inspect_index("private_release")["error"].startswith("Graph 'private_release'")
    assert viewer.query_index("private_release", query="release", methods=["local"])["error"].startswith("Graph 'private_release'")


def test_graph_prompt_tuning_generates_durable_drafts_without_mutating_graph_prompts(tmp_path: Path):
    graph_service, tuning_service, stores = _make_tuning_services(tmp_path)
    graph_service.create_admin_graph(
        graph_id="release-risk",
        display_name="Release Risk",
        collection_id="default",
        source_doc_ids=["DOC-1", "DOC-2"],
    )

    payload = tuning_service.start_tuning_run(
        "release_risk",
        guidance="Focus on approvals, dependencies, policy controls, and owners.",
        target_prompt_files=["extract_graph.txt", "missing_prompt.txt"],
    )

    assert payload["status"] == "completed"
    assert payload["coverage"]["digested_doc_count"] == 2
    assert "extract_graph.txt" in payload["prompt_drafts"]
    assert "missing_prompt.txt" not in payload["prompt_drafts"]
    assert any("missing_prompt.txt" in warning for warning in payload["warnings"])

    artifact_dir = Path(payload["artifact_dir"])
    assert (artifact_dir / "manifest.json").exists()
    assert (artifact_dir / "scratchpad.md").exists()
    assert (artifact_dir / "doc_digests.jsonl").exists()
    assert (artifact_dir / "corpus_profile.json").exists()
    assert (artifact_dir / "prompt_drafts.json").exists()
    assert (artifact_dir / "prompt_diffs.json").exists()

    graph_record = stores.graph_index_store.records["release_risk"]
    assert graph_record.prompt_overrides_json == {}
    assert [run.operation for run in stores.graph_index_run_store.records["release_risk"]].count("research_tune") == 1


def test_graph_prompt_tuning_loads_current_graphrag_prompt_baselines(tmp_path: Path):
    graph_service, tuning_service, stores = _make_tuning_services(tmp_path)
    graph_service.create_admin_graph(
        graph_id="release-risk",
        display_name="Release Risk",
        collection_id="default",
        source_doc_ids=["DOC-1", "DOC-2"],
    )
    target_prompt_files = [
        "summarize_descriptions.txt",
        "community_report_text.txt",
        "community_report_graph.txt",
        "global_search_map_system_prompt.txt",
    ]

    payload = tuning_service.start_tuning_run(
        "release_risk",
        guidance="Keep query prompts grounded in source-provided facts.",
        target_prompt_files=target_prompt_files,
    )

    assert payload["status"] == "completed"
    assert set(target_prompt_files).issubset(set(payload["prompt_drafts"]))
    assert not any("No baseline prompt was available" in warning for warning in payload["warnings"])
    assert all(payload["prompt_drafts"][filename]["baseline_source"] for filename in target_prompt_files)
    assert [run.operation for run in stores.graph_index_run_store.records["release_risk"]].count("research_tune") == 1


def test_graph_prompt_tuning_generates_all_common_targets_from_local_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    graph_service, tuning_service, stores = _make_tuning_services(tmp_path)
    graph_service.create_admin_graph(
        graph_id="release-risk",
        display_name="Release Risk",
        collection_id="default",
        source_doc_ids=["DOC-1", "DOC-2"],
    )

    def _missing_module(*args, **kwargs):
        del args, kwargs
        raise ModuleNotFoundError("graphrag prompt module hidden in test")

    monkeypatch.setattr(prompt_tuning_module.importlib, "import_module", _missing_module)

    payload = tuning_service.start_tuning_run(
        "release_risk",
        guidance="Use local fallback baselines when GraphRAG prompt constants are unavailable.",
        target_prompt_files=COMMON_GRAPHRAG_PROMPT_TARGETS,
    )

    assert payload["status"] == "completed"
    assert set(COMMON_GRAPHRAG_PROMPT_TARGETS) == set(payload["prompt_drafts"])
    assert not any("No baseline prompt was available" in warning for warning in payload["warnings"])
    assert {
        draft["baseline_source"]
        for draft in payload["prompt_drafts"].values()
    } == {"agentic_chatbot_local_fallback"}
    assert [run.operation for run in stores.graph_index_run_store.records["release_risk"]].count("research_tune") == 1


def test_graph_prompt_tuning_apply_writes_only_selected_valid_drafts(tmp_path: Path):
    graph_service, tuning_service, stores = _make_tuning_services(tmp_path)
    graph_service.create_admin_graph(
        graph_id="release-risk",
        display_name="Release Risk",
        collection_id="default",
        source_doc_ids=["DOC-1", "DOC-2"],
        prompt_overrides={"local_search_system_prompt.txt": "Existing local prompt."},
    )
    tuned = tuning_service.start_tuning_run(
        "release_risk",
        guidance="Prefer release ownership and supplier policy relationships.",
        target_prompt_files=["extract_graph.txt"],
    )

    applied = tuning_service.apply_tuning_run(
        "release_risk",
        tuned["run_id"],
        prompt_files=["extract_graph.txt"],
        actor="tester",
    )

    assert applied["applied"] is True
    assert applied["applied_prompt_files"] == ["extract_graph.txt"]
    graph_record = stores.graph_index_store.records["release_risk"]
    assert graph_record.prompt_overrides_json["local_search_system_prompt.txt"] == "Existing local prompt."
    assert "extract_graph.txt" in graph_record.prompt_overrides_json
    assert "Dataset-Specific Curation Guidance" in graph_record.prompt_overrides_json["extract_graph.txt"]
    assert "research_tune_apply" in [run.operation for run in stores.graph_index_run_store.records["release_risk"]]


def test_graph_service_requires_both_graph_and_collection_grants_when_authz_enabled(tmp_path: Path):
    service = _make_service(tmp_path)
    service.create_admin_graph(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
        visibility="tenant",
    )
    service.session.metadata["access_summary"] = {
        "authz_enabled": True,
        "session_upload_collection_id": "",
        "resources": {
            "collection": {"use": [], "manage": [], "use_all": False, "manage_all": False},
            "graph": {"use": ["release_graph"], "manage": [], "use_all": False, "manage_all": False},
            "tool": {"use": [], "manage": [], "use_all": False, "manage_all": False},
            "skill_family": {"use": [], "manage": [], "use_all": False, "manage_all": False},
        },
    }

    assert service.list_indexes() == []
    assert service.inspect_index("release_graph")["error"].startswith("Graph 'release_graph'")

    service.session.metadata["access_summary"]["resources"]["collection"]["use"] = ["default"]

    assert [item["graph_id"] for item in service.list_indexes()] == ["release_graph"]
    assert service.inspect_index("release_graph")["graph"]["graph_id"] == "release_graph"


def test_graph_service_persists_active_graph_ids_after_query(tmp_path: Path):
    service = _make_service(tmp_path)
    service.index_corpus(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )
    service.session.metadata.clear()

    payload = service.query_across_graphs(
        "Which release dependencies should we review?",
        collection_id="default",
        limit=5,
        top_k_graphs=2,
    )

    assert payload["graph_shortlist"][0]["graph_id"] == "release_graph"
    assert service.session.metadata["active_graph_ids"] == ["release_graph"]


def test_graph_service_prefers_live_run_state_over_stale_failed_graph_status(tmp_path: Path):
    service = _make_service(tmp_path)
    service.index_corpus(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )
    store = service._index_store()
    stale = store.get_index("release_graph", "local-dev", user_id="user")
    store.upsert_index(
        replace(
            stale,
            status="failed",
            health={"warnings": ["GRAPHRAG_INDEX_FAILED"]},
        )
    )
    service._run_result(
        graph_id="release_graph",
        operation="refresh",
        status="running",
        detail="Starting graph refresh.",
    )

    listed = service.list_indexes(collection_id="default")
    inspected = service.inspect_index("release_graph")

    assert listed[0]["status"] == "running"
    assert listed[0]["active_run"]["operation"] == "refresh"
    assert listed[0]["health"]["warnings"] == []
    assert listed[0]["health"]["previous_warnings"] == ["GRAPHRAG_INDEX_FAILED"]
    assert inspected["graph"]["status"] == "running"
    assert inspected["graph"]["status_detail"] == "Starting graph refresh."


def test_graph_service_prefers_active_build_over_later_validation_run(tmp_path: Path):
    service = _make_service(tmp_path)
    service.create_admin_graph(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )
    build_started_at = datetime(2026, 4, 17, 12, 0, tzinfo=timezone.utc)
    validate_started_at = build_started_at + timedelta(minutes=3)
    service._run_store().upsert_run(
        GraphIndexRunRecord(
            run_id="grun_build1234",
            graph_id="release_graph",
            tenant_id="local-dev",
            operation="build",
            status="running",
            detail="Building graph artifacts.",
            metadata={
                "runner_pid": 321,
                "child_pid": 654,
                "last_heartbeat_at": (build_started_at + timedelta(minutes=2)).isoformat(),
            },
            started_at=build_started_at.isoformat(),
            completed_at="",
        )
    )
    service._run_store().upsert_run(
        GraphIndexRunRecord(
            run_id="grun_validate1234",
            graph_id="release_graph",
            tenant_id="local-dev",
            operation="validate",
            status="completed",
            detail="Validation finished with status 'warning'.",
            metadata={"payload": {"status": "warning", "ok": True}},
            started_at=validate_started_at.isoformat(),
            completed_at=(validate_started_at + timedelta(seconds=5)).isoformat(),
        )
    )

    inspected = service.inspect_index("release_graph")

    assert inspected["graph"]["status"] == "running"
    assert inspected["graph"]["latest_run"]["run_id"] == "grun_build1234"
    assert inspected["graph"]["active_run"]["operation"] == "build"
    assert inspected["runs"][0]["run_id"] == "grun_build1234"
    assert any(item["run_id"] == "grun_validate1234" for item in inspected["runs"][1:])


def test_graph_service_marks_stale_running_runs_failed_when_process_disappears(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service = _make_service(tmp_path)
    service.create_admin_graph(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )
    store = service._index_store()
    current = store.get_index("release_graph", "local-dev", user_id="user")
    store.upsert_index(replace(current, status="running", health={"warnings": ["GRAPHRAG_INDEX_FAILED"]}))
    service._run_store().records["release_graph"] = []

    old_started_at = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    service._run_store().upsert_run(
        GraphIndexRunRecord(
            run_id="grun_stale1234",
            graph_id="release_graph",
            tenant_id="local-dev",
            operation="refresh",
            status="running",
            detail="Starting graph refresh.",
            metadata={},
            started_at=old_started_at.isoformat(),
            completed_at="",
        )
    )
    monkeypatch.setattr(service, "_latest_graph_log_activity_at", lambda graph_id: old_started_at)
    monkeypatch.setattr(service, "_is_graphrag_process_active", lambda graph_id, *, root_path: False)
    monkeypatch.setattr(service, "_now_utc", lambda: old_started_at + timedelta(hours=2))

    listed = service.list_indexes(collection_id="default")
    inspected = service.inspect_index("release_graph")

    assert listed[0]["status"] == "failed"
    assert listed[0]["status_detail"].startswith("GraphRAG refresh run appears stalled")
    assert "GRAPHRAG_RUN_STALE" in listed[0]["health"]["warnings"]
    assert listed[0]["failure_mode"] == "stale_no_output"
    assert inspected["graph"]["status"] == "failed"
    assert inspected["runs"][0]["status"] == "failed"
    assert inspected["runs"][0]["metadata"]["stale_run_recovered"] is True
    assert inspected["runs"][0]["metadata"]["failure_mode"] == "stale_no_output"


def test_graph_service_full_rebuild_clears_partial_project_state(tmp_path: Path):
    service = _make_service(tmp_path)
    service.index_corpus(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )
    root = service._graph_root("release_graph")
    (root / "output").mkdir(parents=True, exist_ok=True)
    (root / "output" / "stale.parquet").write_text("stale", encoding="utf-8")
    (root / "cache").mkdir(parents=True, exist_ok=True)
    (root / "cache" / "stale.json").write_text("stale", encoding="utf-8")
    (root / "input" / "999_stale.txt").write_text("stale", encoding="utf-8")

    seen = {}

    class _TrackingBackend(_FakeBackend):
        def index_documents(self, graph_id: str, root_path: Path, *, refresh: bool = False):
            seen["refresh"] = refresh
            seen["output_exists"] = (root_path / "output" / "stale.parquet").exists()
            seen["cache_exists"] = (root_path / "cache" / "stale.json").exists()
            seen["input_files"] = sorted(path.name for path in (root_path / "input").glob("*.txt"))
            return super().index_documents(graph_id, root_path, refresh=refresh)

    service._backend_for = lambda backend_name: _TrackingBackend()

    payload = service.build_admin_graph("release_graph")

    assert seen["refresh"] is False
    assert seen["output_exists"] is False
    assert seen["cache_exists"] is False
    assert seen["input_files"] == ["001_release.txt"]
    assert payload["refresh_mode"] == "full_rebuild"
    assert sorted(Path(path).name for path in payload["cleared_state"]["removed_dirs"]) == ["cache", "input", "output"]


def test_graph_service_auto_binds_graph_scoped_skills(tmp_path: Path):
    skill_store = _FakeSkillStore(
        [
            SimpleNamespace(
                skill_id="release-graph-guidance",
                tenant_id="local-dev",
                graph_id="release_graph",
                visibility="tenant",
                owner_user_id="user",
            )
        ]
    )
    service = _make_service(tmp_path, skill_store=skill_store)

    payload = service.index_corpus(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )

    assert "release-graph-guidance" in payload["graph"]["graph_skill_ids"]


def test_graph_service_replaces_stale_running_builds(tmp_path: Path):
    service = _make_service(tmp_path)
    service.create_admin_graph(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )
    stale_run_id = service._run_result(
        graph_id="release_graph",
        operation="build",
        status="running",
        detail="Starting graph build.",
    )

    payload = service.build_admin_graph("release_graph")

    runs = payload["progress"]["latest_run"]
    assert payload["operation_status"] == "already_running"
    assert payload["active_run"]["run_id"] == stale_run_id
    assert runs["status"] == "running"


def test_graph_service_background_build_returns_running_payload(tmp_path: Path):
    service = _make_service(tmp_path)
    backend = _AsyncBackend()
    service._backend_for = lambda backend_name: backend
    service.create_admin_graph(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )

    payload = service.build_admin_graph("release_graph")

    assert payload["status"] == "running"
    assert payload["graph"]["status"] == "running"
    assert payload["graph"]["run_mode"] == "background"
    assert payload["graph"]["active_pid"] == 4321
    assert payload["runs"][0]["status"] == "running"
    assert payload["runs"][0]["metadata"]["run_mode"] == "background"
    assert payload["runs"][0]["metadata"]["runner_log_path"].endswith("_runner.log")
    assert backend.sync_calls == 0


def test_graph_service_inspect_finalizes_completed_background_run(tmp_path: Path):
    service = _make_service(tmp_path)
    backend = _AsyncBackend()
    service._backend_for = lambda backend_name: backend
    service.create_admin_graph(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )

    payload = service.build_admin_graph("release_graph")
    run = payload["runs"][0]
    state_path = Path(run["metadata"]["state_path"])
    output_dir = service._graph_root("release_graph") / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        '{"status":"completed","returncode":0,"timed_out":false,"state_path":"%s","stream_log_path":"","stream_tail":"done"}'
        % str(state_path),
        encoding="utf-8",
    )

    inspected = service.inspect_index("release_graph")

    assert inspected["graph"]["status"] == "ready"
    assert inspected["graph"]["query_ready"] is True
    assert inspected["runs"][0]["status"] == "ready"
    assert inspected["runs"][0]["metadata"]["run_mode"] == "background"


def test_graph_service_surfaces_log_progress_for_running_background_build(tmp_path: Path):
    service = _make_service(tmp_path)
    backend = _AsyncBackend()
    service._backend_for = lambda backend_name: backend
    service.create_admin_graph(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )
    payload = service.build_admin_graph("release_graph")
    log_path = service._graph_log_dir("release_graph") / "indexing-engine.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "Workflow started: extract_graph\nextract graph progress: 6/42\n",
        encoding="utf-8",
    )

    inspected = service.inspect_index("release_graph")

    assert inspected["graph"]["status"] == "running"
    assert inspected["graph"]["progress"]["current"] == 6
    assert inspected["graph"]["progress"]["total"] == 42
    assert inspected["graph"]["run_mode"] == "background"
    assert inspected["graph"]["last_log_activity_at"]


def test_graph_service_inspect_surfaces_background_heartbeat_metadata(tmp_path: Path):
    service = _make_service(tmp_path)
    backend = _AsyncBackend()
    service._backend_for = lambda backend_name: backend
    service.create_admin_graph(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )

    payload = service.build_admin_graph("release_graph")
    run = payload["runs"][0]
    state_path = Path(run["metadata"]["state_path"])
    state_path.write_text(
        json.dumps(
            {
                "status": "running",
                "runner_pid": 3333,
                "child_pid": 4444,
                "last_heartbeat_at": "2026-04-16T12:00:10+00:00",
                "last_output_at": "2026-04-16T12:00:12+00:00",
                "runner_log_path": str(state_path.with_name(state_path.name.replace("_job_state.json", "_runner.log"))),
                "stream_log_path": str(state_path.with_name(state_path.name.replace("_job_state.json", "_job_stream.log"))),
            }
        ),
        encoding="utf-8",
    )

    inspected = service.inspect_index("release_graph")

    assert inspected["graph"]["status"] == "running"
    assert inspected["graph"]["runner_pid"] == 3333
    assert inspected["graph"]["child_pid"] == 4444
    assert inspected["graph"]["last_heartbeat_at"] == "2026-04-16T12:00:10+00:00"
    assert inspected["graph"]["last_output_at"] == "2026-04-16T12:00:12+00:00"
    assert inspected["runs"][0]["metadata"]["runner_pid"] == 3333
    assert inspected["runs"][0]["metadata"]["child_pid"] == 4444


def test_graph_service_inspect_surfaces_phased_build_metadata(tmp_path: Path):
    service = _make_service(tmp_path)
    backend = _AsyncBackend()
    service._backend_for = lambda backend_name: backend
    service.create_admin_graph(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )

    payload = service.build_admin_graph("release_graph")
    run = payload["runs"][0]
    state_path = Path(run["metadata"]["state_path"])
    state_path.write_text(
        json.dumps(
            {
                "status": "running",
                "runner_pid": 3333,
                "child_pid": 4444,
                "last_heartbeat_at": "2026-04-16T12:05:10+00:00",
                "last_output_at": "2026-04-16T12:05:12+00:00",
                "build_phase": "phase_2_reports",
                "fallback_used": True,
                "repair_summary": {
                    "orphan_membership_count": 2,
                    "affected_community_ids": [56, 88],
                },
                "runner_log_path": str(state_path.with_name(state_path.name.replace("_job_state.json", "_runner.log"))),
                "stream_log_path": str(state_path.with_name(state_path.name.replace("_job_state.json", "_job_stream.log"))),
            }
        ),
        encoding="utf-8",
    )

    inspected = service.inspect_index("release_graph")

    assert inspected["graph"]["status"] == "running"
    assert inspected["graph"]["build_phase"] == "phase_2_reports"
    assert inspected["graph"]["fallback_used"] is True
    assert inspected["graph"]["repair_summary"]["affected_community_ids"] == [56, 88]
    assert inspected["runs"][0]["metadata"]["build_phase"] == "phase_2_reports"
    assert inspected["runs"][0]["metadata"]["fallback_used"] is True
    assert inspected["runs"][0]["metadata"]["repair_summary"]["orphan_membership_count"] == 2


def test_graph_service_validate_admin_graph_includes_extract_graph_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    service = _make_service(tmp_path)
    service.create_admin_graph(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )
    monkeypatch.setattr(
        service,
        "_sample_extraction_probe_text",
        lambda resolved_docs: {
            "doc_id": "DOC-1",
            "title": "Release Readiness",
            "text": "Vendor onboarding depends on finance approval and security review. " * 6,
        },
    )

    def _fake_get(url: str, **kwargs):
        del kwargs
        assert url.endswith("/models")
        return httpx.Response(
            200,
                json={
                    "data": [
                        {"id": "nemotron-cascade-2:30b"},
                        {"id": "nemotron-cascade-2:30b"},
                        {"id": "nomic-embed-text:latest"},
                    ]
                },
            )

    def _fake_post(url: str, **kwargs):
        assert url.endswith("/chat/completions")
        assert kwargs["json"]["model"] == "nemotron-cascade-2:30b"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "(\"entity\"<|>\"Vendor\"<|>\"organization\"<|>\"Supplier\")##<|COMPLETE|>"
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "get", _fake_get)
    monkeypatch.setattr(httpx, "post", _fake_post)

    payload = service.validate_admin_graph("release_graph")

    assert payload["ok"] is True
    assert payload["profile"]["vector_size"] == 768
    assert payload["profile"]["chunk_size"] == 800
    assert payload["profile"]["chunk_overlap"] == 80
    assert payload["profile"]["index_chat_model"] == "nemotron-cascade-2:30b"
    assert payload["profile"]["index_timeout_seconds"] == 900
    assert payload["extraction_preflight"]["status"] == "ready"
    assert payload["extraction_preflight"]["ok"] is True


def test_graph_service_extract_graph_preflight_uses_builtin_prompt_when_graphrag_import_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    service = _make_service(tmp_path)
    service.create_admin_graph(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )
    store = service._index_store()
    record = store.get_index("release_graph", "local-dev", user_id="user")
    original_import = builtins.__import__

    def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "graphrag.prompts.index.extract_graph":
            raise ModuleNotFoundError("No module named 'graphrag'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _patched_import)

    prompt = service._extract_graph_prompt_template(record)

    assert "Given a text document" in prompt
    assert "Entity_types: {entity_types}" in prompt
    assert "<|COMPLETE|>" in prompt


def test_graph_service_validate_admin_graph_accepts_complete_only_preflight_as_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    service = _make_service(tmp_path)
    service.create_admin_graph(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )
    monkeypatch.setattr(
        service,
        "_sample_extraction_probe_text",
        lambda resolved_docs: {
            "doc_id": "DOC-1",
            "title": "Sparse Spreadsheet Slice",
            "text": "Line item owner amount date line item owner amount date " * 20,
        },
    )

    def _fake_get(url: str, **kwargs):
        del kwargs
        assert url.endswith("/models")
        return httpx.Response(
            200,
            json={"data": [{"id": "nemotron-cascade-2:30b"}, {"id": "nemotron-cascade-2:30b"}, {"id": "nomic-embed-text:latest"}]},
        )

    def _fake_post(url: str, **kwargs):
        assert url.endswith("/chat/completions")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "<|COMPLETE|>"}}]},
        )

    monkeypatch.setattr(httpx, "get", _fake_get)
    monkeypatch.setattr(httpx, "post", _fake_post)

    payload = service.validate_admin_graph("release_graph")

    assert payload["ok"] is True
    assert payload["status"] == "warning"
    assert payload["extraction_preflight"]["ok"] is True
    assert payload["extraction_preflight"]["status"] == "warning"
    assert payload["profile"]["community_report_mode"] == "text"
    assert payload["profile"]["community_report_chat_model"] == "nemotron-cascade-2:30b"
    assert payload["profile"]["community_report_timeout_seconds"] == 300


def test_community_report_input_repair_removes_orphan_memberships(tmp_path: Path):
    project_root = tmp_path / "graph_project"
    _write_phase_1_orphan_artifacts(project_root)

    dry_run = analyze_community_report_inputs(project_root, dry_run=True)
    applied = analyze_community_report_inputs(project_root, dry_run=False)

    assert dry_run["status"] == "warning"
    assert dry_run["orphan_membership_count"] == 1
    assert dry_run["affected_community_ids"] == [2]
    assert dry_run["emptied_community_ids"] == [2]
    assert dry_run["native_phase2_safe"] is False

    repaired = pq.read_table(project_root / "output" / "communities.parquet").to_pandas()
    text_unit_ids = {
        int(row["community"]): (
            []
            if row["text_unit_ids"] is None or (hasattr(row["text_unit_ids"], "size") and row["text_unit_ids"].size == 0)
            else list(row["text_unit_ids"])
        )
        for row in repaired.to_dict(orient="records")
    }
    assert applied["dropped_tuple_count"] == 1
    assert text_unit_ids[1] == ["tu-1"]
    assert text_unit_ids[2] == []


def test_graph_service_validate_admin_graph_includes_community_report_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    service = _make_service(tmp_path)
    service.create_admin_graph(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )
    _write_phase_1_orphan_artifacts(service._graph_root("release_graph"))
    monkeypatch.setattr(
        service,
        "_validate_graphrag_connectivity",
        lambda profile: {"ok": True, "status": "ready", "detail": "ok"},
    )
    monkeypatch.setattr(
        service,
        "_validate_extract_graph_preflight",
        lambda **kwargs: {"ok": True, "status": "ready", "detail": "ok"},
    )

    payload = service.validate_admin_graph("release_graph")

    assert payload["status"] == "warning"
    assert payload["ok"] is True
    assert payload["community_report_preflight"]["status"] == "warning"
    assert payload["community_report_preflight"]["orphan_membership_count"] == 1
    assert payload["community_report_preflight"]["affected_community_ids"] == [2]
    assert payload["community_report_preflight"]["emptied_community_ids"] == [2]
    assert payload["community_report_preflight"]["native_phase2_safe"] is False


def test_fallback_community_reports_writer_produces_queryable_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    service = _make_service(tmp_path)
    project_root = tmp_path / "graph_project"
    _write_phase_1_orphan_artifacts(project_root)
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("llm offline")))

    payload = generate_fallback_community_reports(project_root, settings=service.settings)
    bundle = load_artifact_bundle(project_root)
    reports = pq.read_table(project_root / "output" / "community_reports.parquet").to_pandas()

    assert payload["ok"] is True
    assert payload["fallback_used"] is True
    assert payload["deterministic_generated_count"] == 2
    assert bundle.query_ready is True
    assert "community_reports" in bundle.artifact_tables
    assert {
        "id",
        "human_readable_id",
        "community",
        "level",
        "parent",
        "children",
        "title",
        "summary",
        "full_content",
        "rank",
        "rating_explanation",
        "findings",
        "full_content_json",
        "period",
        "size",
    }.issubset(set(reports.columns))
    assert {"doc_ids", "text_unit_ids"}.issubset(set(reports.columns))


def test_graph_service_ignores_stale_progress_logs_from_prior_runs(tmp_path: Path):
    service = _make_service(tmp_path)
    backend = _AsyncBackend()
    service._backend_for = lambda backend_name: backend
    service.create_admin_graph(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )
    log_path = service._graph_log_dir("release_graph") / "indexing-engine.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "Workflow started: extract_graph\nextract graph progress: 6/42\n",
        encoding="utf-8",
    )
    old_time = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc).timestamp()
    os.utime(log_path, (old_time, old_time))

    payload = service.build_admin_graph("release_graph")

    assert payload["graph"]["status"] == "running"
    assert "progress" not in payload["graph"]
    assert payload["runs"][0]["metadata"].get("progress") is None


def test_graph_service_progress_payload_reports_live_stages_and_logs(tmp_path: Path):
    service = _make_service(tmp_path)
    backend = _AsyncBackend()
    service._backend_for = lambda backend_name: backend
    service.create_admin_graph(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )

    service.build_admin_graph("release_graph")
    log_path = service._graph_log_dir("release_graph") / "indexing-engine.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "Workflow started: extract_graph\nextract graph progress: 6/42\n",
        encoding="utf-8",
    )

    progress = service.graph_progress("release_graph")

    assert progress["active"] is True
    assert progress["workflow"] == "extract_graph"
    assert progress["percent"] > 0
    assert any(stage["state"] == "active" and stage["id"] == "extract_graph" for stage in progress["stages"])
    assert "extract graph progress" in progress["log_tail"]


def test_graph_service_cancel_and_delete_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service = _make_service(tmp_path)
    backend = _AsyncBackend()
    service._backend_for = lambda backend_name: backend
    service.create_admin_graph(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )
    service.build_admin_graph("release_graph")
    run_id = service.list_graph_runs("release_graph")[0]["run_id"]
    terminated: list[str] = []
    monkeypatch.setattr(service, "_terminate_owned_run_process", lambda run: terminated.append(run.run_id))

    blocked = service.delete_admin_graph("release_graph")
    cancelled = service.cancel_admin_graph_run("release_graph", run_id=run_id, actor="tester")
    deleted = service.delete_admin_graph("release_graph")

    assert "active build" in blocked["error"]
    assert cancelled["status"] == "cancelled"
    assert terminated == [run_id]
    assert deleted["deleted"] is True
    assert service.inspect_index("release_graph")["error"].startswith("Graph")


def test_graph_service_lists_and_deletes_failed_runs_only(tmp_path: Path):
    service = _make_service(tmp_path)
    service.create_admin_graph(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )
    service.create_admin_graph(
        graph_id="ops-graph",
        display_name="Ops Graph",
        collection_id="default",
        source_doc_ids=["DOC-2"],
    )
    failed_run_id = service._run_result(
        graph_id="release_graph",
        operation="build",
        status="failed",
        detail="Build failed.",
        run_id="failed-run-1",
    )
    ready_run_id = service._run_result(
        graph_id="release_graph",
        operation="validate",
        status="ready",
        detail="Validation passed.",
        run_id="ready-run-1",
    )
    running_run_id = service._run_result(
        graph_id="ops_graph",
        operation="build",
        status="running",
        detail="Build running.",
        run_id="running-run-1",
    )
    second_failed_run_id = service._run_result(
        graph_id="ops_graph",
        operation="refresh",
        status="failed",
        detail="Refresh failed.",
        run_id="failed-run-2",
    )

    failed_runs = service.list_graph_runs_by_status(status="failed")
    assert {item["run_id"] for item in failed_runs} == {failed_run_id, second_failed_run_id}

    blocked_active = service.delete_admin_graph_run("ops_graph", run_id=running_run_id)
    blocked_ready = service.delete_admin_graph_run("release_graph", run_id=ready_run_id)
    deleted = service.delete_admin_graph_run("release_graph", run_id=failed_run_id)
    cleanup = service.cleanup_admin_graph_runs(status="failed")

    assert "Active graph runs" in blocked_active["error"]
    assert "Only failed graph runs" in blocked_ready["error"]
    assert deleted["deleted"] is True
    assert cleanup["deleted_count"] == 1
    remaining_run_ids = {
        item["run_id"]
        for item in service.list_graph_runs_by_status(status="", limit=20)
    }
    assert failed_run_id not in remaining_run_ids
    assert second_failed_run_id not in remaining_run_ids
    assert ready_run_id in remaining_run_ids
    assert running_run_id in remaining_run_ids


def test_graph_service_terminates_superseded_background_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service = _make_service(tmp_path)
    service.create_admin_graph(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )
    stale_run_id = service._run_result(
        graph_id="release_graph",
        operation="build",
        status="running",
        detail="Starting graph build.",
        metadata={"run_mode": "background", "active_pid": 111, "active_process_group_id": 111},
    )
    terminated: list[str] = []
    monkeypatch.setattr(
        service,
        "_terminate_owned_run_process",
        lambda run: terminated.append(str(run.run_id)),
    )

    payload = service.build_admin_graph("release_graph", cancel_existing=True)

    stale = next(item for item in payload["runs"] if item["run_id"] == stale_run_id)
    assert terminated == [stale_run_id]
    assert stale["status"] == "failed"


def test_graph_service_refresh_falls_back_to_full_rebuild_after_failed_partial_output(tmp_path: Path):
    service = _make_service(tmp_path)
    service.create_admin_graph(
        graph_id="release-graph",
        display_name="Release Graph",
        collection_id="default",
        source_doc_ids=["DOC-1"],
    )
    output_dir = service._graph_root("release_graph") / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "documents.parquet").write_text("partial", encoding="utf-8")
    (output_dir / "text_units.parquet").write_text("partial", encoding="utf-8")

    store = service._index_store()
    record = store.get_index("release_graph", "local-dev", user_id="user")
    store.upsert_index(replace(record, status="failed", query_ready=False))

    seen = {}

    class _TrackingBackend(_FakeBackend):
        def index_documents(self, graph_id: str, root_path: Path, *, refresh: bool = False):
            seen["refresh"] = refresh
            return super().index_documents(graph_id, root_path, refresh=refresh)

    service._backend_for = lambda backend_name: _TrackingBackend()

    payload = service.build_admin_graph("release_graph", refresh=True)

    assert seen["refresh"] is False
    assert payload["refresh_mode"] == "full_rebuild"


def test_microsoft_graphrag_backend_detects_global_only_artifacts(tmp_path: Path):
    root = tmp_path / "graph_project"
    output_dir = root / "output"
    output_dir.mkdir(parents=True)
    pq.write_table(pa.table({"id": ["ent-1"], "title": ["Release Dependency"], "description": ["Tracks release blockers"]}), output_dir / "entities.parquet")
    pq.write_table(pa.table({"id": ["community-1"], "level": [1]}), output_dir / "communities.parquet")
    pq.write_table(pa.table({"id": ["report-1"], "title": ["Release Status"], "summary": ["Release dependency overview"], "doc_ids": [["DOC-1"]]}), output_dir / "community_reports.parquet")

    backend = MicrosoftGraphRagBackend(
        SimpleNamespace(
            graphrag_use_container=False,
            graphrag_cli_command="graphrag",
            graphrag_artifact_cache_ttl_seconds=300,
        )
    )

    result = backend.import_existing_graph("release_graph", root, artifact_path=str(output_dir))

    assert result.query_ready is True
    assert result.query_backend == "graphrag_python_api_preferred"
    assert result.supported_query_methods == ["global"]
    assert set(result.artifact_tables) == {"communities", "community_reports", "entities"}


def test_microsoft_graphrag_backend_queries_local_artifacts_without_live_api(tmp_path: Path):
    root = tmp_path / "graph_project"
    output_dir = root / "output"
    output_dir.mkdir(parents=True)
    pq.write_table(pa.table({"id": ["ent-1"], "title": ["Vendor Acme"], "description": ["Vendor entity"], "text_unit_ids": [["doc-1#chunk-1"]]}), output_dir / "entities.parquet")
    pq.write_table(pa.table({"id": ["community-1"], "level": [1]}), output_dir / "communities.parquet")
    pq.write_table(pa.table({"id": ["report-1"], "title": ["Vendor Risk"], "summary": ["Vendor Acme requires approval"], "doc_ids": [["DOC-1"]]}), output_dir / "community_reports.parquet")
    pq.write_table(pa.table({"id": ["rel-1"], "source": ["Vendor Acme"], "target": ["Finance Approval"], "doc_ids": [["DOC-1"]], "chunk_ids": [["doc-1#chunk-1"]]}), output_dir / "relationships.parquet")
    pq.write_table(pa.table({"id": ["doc-1#chunk-1"], "doc_id": ["DOC-1"], "chunk_id": ["doc-1#chunk-1"], "text": ["Vendor Acme depends on Finance approval for renewal."]}), output_dir / "text_units.parquet")

    backend = MicrosoftGraphRagBackend(
        SimpleNamespace(
            graphrag_use_container=False,
            graphrag_cli_command="graphrag",
            graphrag_artifact_cache_ttl_seconds=300,
        )
    )

    hits = backend.query_index(
        "release_graph",
        root,
        query="Which vendor depends on finance approval?",
        method="local",
        limit=4,
        doc_ids=["DOC-1"],
    )

    assert hits
    assert hits[0].backend in {"graphrag_api", "graphrag_artifacts"}
    assert hits[0].doc_id == "DOC-1"
    assert "doc-1#chunk-1" in hits[0].chunk_ids


def test_microsoft_graphrag_backend_resolves_manifest_sources_when_doc_id_is_internal(tmp_path: Path):
    root = tmp_path / "graph_project"
    output_dir = root / "output"
    output_dir.mkdir(parents=True)
    (root / "graph_manifest.json").write_text(
        json.dumps(
            {
                "materialized_sources": [
                    {
                        "doc_id": "DOC-1",
                        "title": "Asterion Planning Draft",
                        "source_path": "/kb/asterion.md",
                        "source_type": "kb",
                        "collection_id": "default",
                        "materialized_path": str(root / "input" / "001_asterion.txt"),
                        "materialized_filename": "001_asterion.txt",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(pa.table({"id": ["ent-1"], "title": ["Asterion"], "description": ["Program"], "text_unit_ids": [["tu-1"]]}), output_dir / "entities.parquet")
    pq.write_table(pa.table({"id": ["community-1"], "level": [1]}), output_dir / "communities.parquet")
    pq.write_table(pa.table({"id": ["report-1"], "title": ["Asterion Risk"], "summary": ["Asterion schedule risk"], "doc_ids": [["001_asterion.txt"]]}), output_dir / "community_reports.parquet")
    pq.write_table(pa.table({"id": ["rel-1"], "source": ["Asterion"], "target": ["North Coast"], "doc_ids": [["001_asterion.txt"]], "chunk_ids": [["tu-1"]]}), output_dir / "relationships.parquet")
    pq.write_table(pa.table({"id": ["tu-1"], "doc_id": ["001_asterion.txt"], "chunk_id": ["tu-1"], "text": ["Asterion depends on North Coast delivery."]}), output_dir / "text_units.parquet")

    backend = MicrosoftGraphRagBackend(
        SimpleNamespace(
            graphrag_use_container=False,
            graphrag_cli_command="graphrag",
            graphrag_artifact_cache_ttl_seconds=300,
        )
    )

    hits = backend.query_index(
        "defense_graph",
        root,
        query="Asterion North Coast schedule risk",
        method="local",
        limit=4,
        doc_ids=["DOC-1"],
    )

    assert hits
    assert hits[0].doc_id == "DOC-1"
    assert hits[0].source_path == "/kb/asterion.md"
    assert hits[0].metadata["source"]["title"] == "Asterion Planning Draft"


def test_microsoft_graphrag_backend_reports_cli_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    backend = MicrosoftGraphRagBackend(
        SimpleNamespace(
            graphrag_use_container=False,
            graphrag_cli_command="graphrag",
            graphrag_timeout_seconds=90,
            graphrag_artifact_cache_ttl_seconds=300,
        )
    )

    monkeypatch.setattr(backend, "_cli_prefix", lambda: ["graphrag"])

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=kwargs.get("args") or args[0], timeout=90, output="started", stderr="")

    monkeypatch.setattr("agentic_chatbot_next.graph.backend.subprocess.run", _raise_timeout)

    result = backend.index_documents("release_graph", tmp_path / "graph_project")

    assert result.status == "failed"
    assert "timed out after 90 seconds" in result.detail
    assert result.warnings == ["GRAPHRAG_INDEX_TIMEOUT"]
