from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List

import httpx
import pytest

from agentic_chatbot_next.agents.registry import AgentRegistry
from agentic_chatbot_next.api import main as api_main
from agentic_chatbot_next.control_panel import routes as control_panel_routes
from agentic_chatbot_next.control_panel.overlay_store import OverlayStore
from agentic_chatbot_next.control_panel.runtime_manager import get_runtime_manager
from agentic_chatbot_next.graph.service import GraphService
from agentic_chatbot_next.persistence.postgres.chunks import ChunkRecord
from agentic_chatbot_next.persistence.postgres.documents import DocumentRecord
from agentic_chatbot_next.persistence.postgres.graphs import GraphIndexRecord, GraphIndexRunRecord, GraphIndexSourceRecord
from agentic_chatbot_next.persistence.postgres.skills import SkillPackRecord
from agentic_chatbot_next.prompting import load_grounded_answer_prompt
from agentic_chatbot_next.runtime.context import RuntimePaths


def _agent_markdown(
    *,
    name: str = "general",
    mode: str = "react",
    description: str = "general agent",
    prompt_file: str = "general_agent.md",
    allowed_tools: str = '["calculator"]',
    allowed_worker_agents: str = "[]",
    preload_skill_packs: str = "[]",
    memory_scopes: str = '["conversation"]',
    metadata: str = '{"role_kind": "top_level", "entry_path": "default"}',
) -> str:
    return (
        "---\n"
        f"name: {name}\n"
        f"mode: {mode}\n"
        f"description: {description}\n"
        f"prompt_file: {prompt_file}\n"
        "skill_scope: general\n"
        f"allowed_tools: {allowed_tools}\n"
        f"allowed_worker_agents: {allowed_worker_agents}\n"
        f"preload_skill_packs: {preload_skill_packs}\n"
        f"memory_scopes: {memory_scopes}\n"
        "max_steps: 3\n"
        "max_tool_calls: 3\n"
        "allow_background_jobs: false\n"
        f"metadata: {metadata}\n"
        "---\n"
        "General body\n"
    )


def _control_panel_settings(tmp_path: Path, *, enabled: bool = True, token: str = "admin-token") -> SimpleNamespace:
    agents_dir = tmp_path / "agents"
    skills_dir = tmp_path / "skills"
    prompts_dir = tmp_path / "prompts"
    kb_dir = tmp_path / "kb"
    docs_dir = tmp_path / "docs"
    uploads_dir = tmp_path / "uploads"
    workspace_dir = tmp_path / "workspace"
    runtime_dir = tmp_path / "runtime"
    overlay_root = tmp_path / "control_panel" / "overlays"
    prompt_overlay_dir = overlay_root / "prompts"
    agent_overlay_dir = overlay_root / "agents"
    audit_log_path = tmp_path / "control_panel" / "audit" / "events.jsonl"

    agents_dir.mkdir(parents=True)
    skills_dir.mkdir(parents=True)
    prompts_dir.mkdir(parents=True)
    kb_dir.mkdir(parents=True)
    docs_dir.mkdir(parents=True)
    uploads_dir.mkdir(parents=True)
    workspace_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    prompt_overlay_dir.mkdir(parents=True)
    agent_overlay_dir.mkdir(parents=True)
    audit_log_path.parent.mkdir(parents=True)

    (skills_dir / "general_agent.md").write_text("Base general prompt", encoding="utf-8")
    (prompts_dir / "grounded_answer.txt").write_text("Base grounded answer", encoding="utf-8")
    (agents_dir / "general.md").write_text(_agent_markdown(), encoding="utf-8")

    return SimpleNamespace(
        control_panel_enabled=enabled,
        control_panel_admin_token=token,
        control_panel_overlay_dir=overlay_root,
        control_panel_runtime_env_path=overlay_root / "runtime.env",
        control_panel_prompt_overlays_dir=prompt_overlay_dir,
        control_panel_agent_overlays_dir=agent_overlay_dir,
        control_panel_audit_log_path=audit_log_path,
        control_panel_static_dir=tmp_path / "dist",
        control_panel_source_allowed_roots=(kb_dir, docs_dir),
        agents_dir=agents_dir,
        skills_dir=skills_dir,
        prompts_dir=prompts_dir,
        kb_dir=kb_dir,
        kb_extra_dirs=(docs_dir,),
        uploads_dir=uploads_dir,
        workspace_dir=workspace_dir,
        runtime_dir=runtime_dir,
        prompts_backend="local",
        grounded_answer_prompt_path=prompts_dir / "grounded_answer.txt",
        default_tenant_id="tenant-1",
        default_user_id="user-1",
        default_conversation_id="conv-1",
        default_collection_id="default",
        gateway_model_id="gateway-local",
        llm_provider="ollama",
        judge_provider="ollama",
        embeddings_provider="ollama",
        ollama_base_url="http://ollama:11434",
        ollama_chat_model="gpt-oss:20b",
        ollama_judge_model="gpt-oss:20b",
        ollama_embed_model="nomic-embed-text",
        graphrag_projects_dir=tmp_path / "graphrag" / "projects",
        graph_backend="microsoft_graphrag",
        graphrag_llm_provider="openai",
        graphrag_base_url="",
        graphrag_api_key="ollama",
        graphrag_chat_model="gpt-oss:20b",
        graphrag_index_chat_model="gpt-oss:20b",
        graphrag_embed_model="nomic-embed-text",
        graphrag_concurrency=2,
        graphrag_request_timeout_seconds=60,
        graphrag_index_request_timeout_seconds=120,
        graphrag_job_timeout_seconds=600,
        graphrag_timeout_seconds=60,
        graphrag_default_query_method="local",
        graphrag_artifact_cache_ttl_seconds=60,
        graph_query_cache_ttl_seconds=60,
        graphrag_use_container=False,
        graphrag_cli_command="graphrag",
        tavily_api_key="super-secret-1234",
        max_agent_steps=6,
        clarification_sensitivity=50,
        agent_chat_model_overrides={},
        agent_judge_model_overrides={},
        llm_router_enabled=True,
        llm_router_mode="hybrid",
        llm_router_confidence_threshold=0.7,
        enable_coordinator_mode=False,
    )


class _FakeDocStore:
    def __init__(self) -> None:
        self.records: Dict[str, DocumentRecord] = {}

    def upsert_document(self, record: DocumentRecord) -> None:
        self.records[record.doc_id] = record

    def document_exists(
        self,
        doc_id: str,
        content_hash: str,
        tenant_id: str,
        *,
        collection_id: str = "",
        source_type: str = "",
        title: str = "",
    ) -> bool:
        record = self.records.get(doc_id)
        if record and record.content_hash == content_hash and record.tenant_id == tenant_id:
            return True
        if not collection_id or not source_type or not title:
            return False
        return any(
            existing.tenant_id == tenant_id
            and existing.content_hash == content_hash
            and existing.collection_id == collection_id
            and existing.source_type == source_type
            and existing.title == title
            for existing in self.records.values()
        )

    def get_document(self, doc_id: str, tenant_id: str) -> DocumentRecord | None:
        record = self.records.get(doc_id)
        if record is None or record.tenant_id != tenant_id:
            return None
        return record

    def list_documents(
        self,
        source_type: str = "",
        tenant_id: str = "local-dev",
        collection_id: str = "",
    ) -> List[DocumentRecord]:
        matches = []
        for record in self.records.values():
            if record.tenant_id != tenant_id:
                continue
            if source_type and record.source_type != source_type:
                continue
            if collection_id and record.collection_id != collection_id:
                continue
            matches.append(record)
        return sorted(matches, key=lambda item: (item.ingested_at, item.title))

    def delete_document(self, doc_id: str, tenant_id: str) -> None:
        record = self.records.get(doc_id)
        if record is not None and record.tenant_id == tenant_id:
            self.records.pop(doc_id, None)

    def search_by_metadata(
        self,
        *,
        tenant_id: str = "local-dev",
        collection_id: str = "",
        source_type: str = "",
        file_type: str = "",
        doc_structure_type: str = "",
        title_contains: str = "",
        limit: int = 100,
    ) -> List[DocumentRecord]:
        matches = []
        for record in self.records.values():
            if record.tenant_id != tenant_id:
                continue
            if collection_id and record.collection_id != collection_id:
                continue
            if source_type and record.source_type != source_type:
                continue
            if file_type and record.file_type != file_type:
                continue
            if doc_structure_type and record.doc_structure_type != doc_structure_type:
                continue
            if title_contains and title_contains.lower() not in record.title.lower():
                continue
            matches.append(record)
        return sorted(matches, key=lambda item: (item.ingested_at, item.title), reverse=True)[:limit]

    def list_collections(self, tenant_id: str = "local-dev") -> List[Dict[str, Any]]:
        grouped: Dict[str, List[DocumentRecord]] = {}
        for record in self.records.values():
            if record.tenant_id != tenant_id:
                continue
            grouped.setdefault(record.collection_id, []).append(record)
        payload = []
        for collection_id, docs in grouped.items():
            source_type_counts: Dict[str, int] = {}
            for doc in docs:
                source_type_counts[doc.source_type] = source_type_counts.get(doc.source_type, 0) + 1
            payload.append(
                {
                    "collection_id": collection_id,
                    "document_count": len(docs),
                    "latest_ingested_at": max(doc.ingested_at for doc in docs) if docs else "",
                    "source_type_counts": source_type_counts,
                }
            )
        return sorted(payload, key=lambda item: str(item["collection_id"]))

    def get_collection_summary(self, collection_id: str, tenant_id: str) -> Dict[str, Any] | None:
        return next(
            (
                dict(item)
                for item in self.list_collections(tenant_id=tenant_id)
                if str(item["collection_id"]) == collection_id
            ),
            None,
        )


class _FakeCollectionStore:
    def __init__(self) -> None:
        self.records: Dict[tuple[str, str], Dict[str, str]] = {}

    def ensure_collection(
        self,
        *,
        tenant_id: str,
        collection_id: str,
        maintenance_policy: str = "",
    ) -> SimpleNamespace:
        key = (tenant_id, collection_id)
        existing = self.records.get(key)
        if existing is None:
            existing = {
                "tenant_id": tenant_id,
                "collection_id": collection_id,
                "maintenance_policy": maintenance_policy or "indexed_documents",
                "created_at": "2026-04-08T10:00:00Z",
                "updated_at": "2026-04-08T10:00:00Z",
            }
        else:
            existing = {
                **existing,
                "maintenance_policy": maintenance_policy or str(existing.get("maintenance_policy") or "indexed_documents"),
                "updated_at": "2026-04-08T10:05:00Z",
            }
        self.records[key] = existing
        return SimpleNamespace(**existing)

    def get_collection(self, collection_id: str, *, tenant_id: str) -> SimpleNamespace | None:
        record = self.records.get((tenant_id, collection_id))
        return SimpleNamespace(**record) if record is not None else None

    def list_collections(self, *, tenant_id: str = "local-dev") -> List[SimpleNamespace]:
        return [
            SimpleNamespace(**record)
            for (record_tenant_id, _), record in sorted(self.records.items())
            if record_tenant_id == tenant_id
        ]

    def delete_collection(self, collection_id: str, *, tenant_id: str) -> bool:
        return self.records.pop((tenant_id, collection_id), None) is not None


class _FakeChunkStore:
    def __init__(self) -> None:
        self.records: Dict[str, List[ChunkRecord]] = {}

    def set_document_chunks(self, doc_id: str, chunks: List[ChunkRecord]) -> None:
        self.records[doc_id] = list(chunks)

    def list_document_chunks(self, doc_id: str, tenant_id: str) -> List[ChunkRecord]:
        del tenant_id
        return sorted(self.records.get(doc_id, []), key=lambda item: item.chunk_index)

    def delete_doc_chunks(self, doc_id: str, tenant_id: str) -> None:
        del tenant_id
        self.records.pop(doc_id, None)


class _FakeSkillStore:
    def __init__(self, records: List[SkillPackRecord]) -> None:
        self.records = {record.skill_id: record for record in records}
        self.chunks = {record.skill_id: ["chunk-1"] for record in records}

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
    ) -> List[SkillPackRecord]:
        del visibility
        matches = []
        for record in self.records.values():
            if record.tenant_id != tenant_id:
                continue
            if agent_scope and record.agent_scope != agent_scope:
                continue
            if enabled_only and not record.enabled:
                continue
            if owner_user_id and record.owner_user_id and record.owner_user_id != owner_user_id:
                continue
            if status and record.status != status:
                continue
            if graph_id and record.graph_id != graph_id:
                continue
            matches.append(record)
        return sorted(matches, key=lambda item: item.name)

    def get_skill_pack(self, skill_id: str, *, tenant_id: str, owner_user_id: str = "") -> SkillPackRecord | None:
        record = self.records.get(skill_id)
        if record is None or record.tenant_id != tenant_id:
            return None
        if owner_user_id and record.owner_user_id and record.owner_user_id != owner_user_id:
            return None
        return record

    def get_skill_chunks(self, skill_id: str, *, tenant_id: str) -> List[str]:
        del tenant_id
        return list(self.chunks.get(skill_id, []))

    def set_skill_status(self, skill_id: str, *, tenant_id: str, status: str, enabled: bool) -> None:
        record = self.records[skill_id]
        self.records[skill_id] = SkillPackRecord(
            **{
                **record.__dict__,
                "tenant_id": tenant_id,
                "status": status,
                "enabled": enabled,
            }
        )

    def get_skill_packs_by_ids(self, skill_ids: List[str], *, tenant_id: str, owner_user_id: str = "") -> List[SkillPackRecord]:
        results = []
        for skill_id in skill_ids:
            record = self.get_skill_pack(skill_id, tenant_id=tenant_id, owner_user_id=owner_user_id)
            if record is not None:
                results.append(record)
        return results

    def upsert_skill_pack(self, record: SkillPackRecord, chunks: List[str]) -> None:
        self.records[record.skill_id] = replace(record)
        self.chunks[record.skill_id] = list(chunks)


class _FakeGraphIndexStore:
    def __init__(self) -> None:
        self.records: Dict[str, GraphIndexRecord] = {}

    def upsert_index(self, record: GraphIndexRecord) -> None:
        self.records[record.graph_id] = replace(record)

    def get_index(self, graph_id: str, tenant_id: str) -> GraphIndexRecord | None:
        record = self.records.get(graph_id)
        if record is None or record.tenant_id != tenant_id:
            return None
        return replace(record)

    def list_indexes(
        self,
        *,
        tenant_id: str = "local-dev",
        collection_id: str = "",
        status: str = "",
        backend: str = "",
        limit: int = 100,
    ) -> List[GraphIndexRecord]:
        matches = []
        for record in self.records.values():
            if record.tenant_id != tenant_id:
                continue
            if collection_id and record.collection_id != collection_id:
                continue
            if status and record.status != status:
                continue
            if backend and record.backend != backend:
                continue
            matches.append(replace(record))
        matches.sort(key=lambda item: ((item.last_indexed_at or item.created_at), item.display_name), reverse=True)
        return matches[:limit]

    def search_indexes(
        self,
        query: str,
        *,
        tenant_id: str = "local-dev",
        collection_id: str = "",
        limit: int = 6,
    ) -> List[GraphIndexRecord]:
        lowered = str(query or "").lower()
        terms = [term for term in lowered.split() if len(term) > 2]
        matches = []
        for record in self.list_indexes(tenant_id=tenant_id, collection_id=collection_id, limit=250):
            haystack = " ".join(
                [
                    record.graph_id,
                    record.display_name,
                    record.domain_summary,
                    " ".join(record.entity_samples),
                    " ".join(record.relationship_samples),
                ]
            ).lower()
            if lowered and lowered not in haystack and not any(term in haystack for term in terms):
                continue
            matches.append(replace(record))
            if len(matches) >= limit:
                break
        return matches

    def update_index_status(self, graph_id: str, tenant_id: str, *, status: str, health=None) -> bool:
        record = self.records.get(graph_id)
        if record is None or record.tenant_id != tenant_id:
            return False
        self.records[graph_id] = replace(record, status=status, health=dict(health or record.health))
        return True

    def delete_index(self, graph_id: str, tenant_id: str) -> Dict[str, int]:
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
    def __init__(self) -> None:
        self.records: Dict[tuple[str, str], List[GraphIndexSourceRecord]] = {}

    def replace_sources(
        self,
        graph_id: str,
        *,
        tenant_id: str = "local-dev",
        sources: List[GraphIndexSourceRecord] | None = None,
    ) -> None:
        self.records[(tenant_id, graph_id)] = [replace(item) for item in (sources or [])]

    def list_sources(self, graph_id: str, *, tenant_id: str = "local-dev", limit: int = 100) -> List[GraphIndexSourceRecord]:
        return [replace(item) for item in self.records.get((tenant_id, graph_id), [])][:limit]


class _FakeGraphIndexRunStore:
    def __init__(self) -> None:
        self.records: Dict[str, List[GraphIndexRunRecord]] = {}

    def upsert_run(self, record: GraphIndexRunRecord) -> None:
        bucket = self.records.setdefault(record.graph_id, [])
        for index, existing in enumerate(bucket):
            if existing.run_id == record.run_id:
                bucket[index] = replace(record)
                return
        bucket.append(replace(record))

    def list_runs(self, graph_id: str, *, tenant_id: str = "local-dev", limit: int = 20) -> List[GraphIndexRunRecord]:
        return [replace(item) for item in self.records.get(graph_id, []) if item.tenant_id == tenant_id][:limit]


class _FakeGraphQueryCacheStore:
    def __init__(self) -> None:
        self.payloads: Dict[tuple[str, str, str, str], Any] = {}

    def get_cached(self, *, graph_id: str, tenant_id: str, query_text: str, query_method: str, now: Any | None = None) -> Any | None:
        del now
        return self.payloads.get((tenant_id, graph_id, query_text, query_method))

    def put_cached(
        self,
        *,
        graph_id: str,
        tenant_id: str,
        query_text: str,
        query_method: str,
        response_json: Dict[str, Any],
        ttl_seconds: int,
    ) -> None:
        del ttl_seconds
        self.payloads[(tenant_id, graph_id, query_text, query_method)] = SimpleNamespace(response_json=dict(response_json))


class _FakePromptBuilder:
    def __init__(self, settings: SimpleNamespace, overlay_store: OverlayStore) -> None:
        self._settings = settings
        self._overlay_store = overlay_store

    def load_prompt(self, prompt_file: str) -> str:
        overlay = self._overlay_store.read_prompt_overlay(prompt_file)
        if overlay:
            return overlay
        for base_dir in (Path(self._settings.skills_dir), Path(self._settings.prompts_dir)):
            path = base_dir / prompt_file
            if path.exists():
                return path.read_text(encoding="utf-8")
        return ""


class _FakeJobManager:
    def list_jobs(self) -> List[Any]:
        return [
            SimpleNamespace(
                job_id="job-1",
                session_id="tenant-1:user-1:conv-1",
                agent_name="general",
                status="complete",
                scheduler_state="completed",
                priority="interactive",
                queue_class="interactive",
                tenant_id="tenant-1",
                user_id="user-1",
                description="Smoke job",
                estimated_token_cost=320,
                actual_token_cost=288,
                budget_block_reason="",
                enqueued_at="2026-04-08T09:59:00Z",
                started_at="2026-04-08T10:00:00Z",
                updated_at="2026-04-08T10:00:00Z",
                output_path="/tmp/output.txt",
            )
        ]

    def scheduler_snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "max_concurrency": 4,
            "running_jobs": 1,
            "available_slots": 3,
            "reserved_urgent_slots": 1,
            "urgent_backlog": False,
            "queue_depths": {"urgent": 0, "interactive": 1, "background": 0},
            "oldest_wait_seconds": {"urgent": 0.0, "interactive": 3.5, "background": 0.0},
            "budget_blocked_jobs": 0,
            "tenant_budget_health": [
                {
                    "tenant_id": "tenant-1",
                    "queued_jobs": 1,
                    "running_jobs": 1,
                    "budget_blocked_jobs": 0,
                    "available_tokens": 42000.0,
                }
            ],
        }


class _FakeOutcome:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = dict(payload)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._payload)


class _FakeReviewSample(_FakeOutcome):
    pass


class _FakeRouterFeedback:
    def finalize_stale_decisions(self) -> List[Any]:
        return []

    def list_recent_outcomes(self, *, limit: int = 200) -> List[Any]:
        del limit
        return [
            _FakeOutcome(
                {
                    "router_decision_id": "rtd_1",
                    "route": "AGENT",
                    "router_method": "hybrid",
                    "suggested_agent": "rag_worker",
                    "outcome_label": "negative",
                    "evidence_signals": ["manual_agent_override"],
                    "scored_at": "2026-04-08T10:06:00Z",
                }
            ),
            _FakeOutcome(
                {
                    "router_decision_id": "rtd_2",
                    "route": "BASIC",
                    "router_method": "deterministic",
                    "suggested_agent": "basic",
                    "outcome_label": "positive",
                    "evidence_signals": ["verifier_pass"],
                    "scored_at": "2026-04-08T10:07:00Z",
                }
            ),
        ]

    def list_review_samples(self, *, limit: int = 200) -> List[Any]:
        del limit
        return [
            _FakeReviewSample(
                {
                    "sample_id": "rrs_1",
                    "router_decision_id": "rtd_1",
                    "route": "AGENT",
                    "router_method": "hybrid",
                    "suggested_agent": "rag_worker",
                    "outcome_label": "negative",
                    "evidence_signals": ["manual_agent_override"],
                    "review_status": "pending",
                    "created_at": "2026-04-08T10:06:30Z",
                }
            )
        ]

    def get_last_retrain_report_metadata(self) -> Dict[str, Any]:
        return {
            "quarter": "2026-Q2",
            "generated_at": "2026-04-08T10:08:00Z",
            "recommended_threshold": 0.75,
        }


class _FakeKernel:
    def __init__(self, settings: SimpleNamespace, registry: AgentRegistry, prompt_builder: _FakePromptBuilder) -> None:
        self.paths = RuntimePaths.from_settings(settings)
        self.registry = registry
        self.prompt_builder = prompt_builder
        self.job_manager = _FakeJobManager()
        self.router_feedback = _FakeRouterFeedback()

    def export_langgraph_react_graph(self, agent_name: str = "") -> Dict[str, Any]:
        selected_agent = agent_name or self.registry.get_default_agent_name()
        return {
            "status": "available",
            "generated_at": "2026-04-08T10:00:00Z",
            "agent_name": selected_agent,
            "mermaid": "graph TD\n  __start__ --> agent\n  agent --> tools\n  tools --> agent\n  agent --> __end__",
            "nodes": [
                {"id": "__start__", "name": "__start__", "data_type": "RunnableCallable", "metadata": {}},
                {"id": "agent", "name": "agent", "data_type": "RunnableCallable", "metadata": {}},
                {"id": "tools", "name": "tools", "data_type": "PolicyAwareToolNode", "metadata": {}},
                {"id": "__end__", "name": "__end__", "data_type": "RunnableCallable", "metadata": {}},
            ],
            "edges": [
                {"id": "langgraph-edge-1", "source": "__start__", "target": "agent", "conditional": False, "data": None},
                {"id": "langgraph-edge-2", "source": "agent", "target": "tools", "conditional": True, "data": "tools_condition"},
                {"id": "langgraph-edge-3", "source": "tools", "target": "agent", "conditional": False, "data": None},
            ],
            "warnings": [],
        }


ENV_ATTR_MAP = {
    "MAX_AGENT_STEPS": "max_agent_steps",
    "OLLAMA_CHAT_MODEL": "ollama_chat_model",
    "CLARIFICATION_SENSITIVITY": "clarification_sensitivity",
}


def _merge_settings(settings: SimpleNamespace, env_overrides: Dict[str, str | None]) -> SimpleNamespace:
    payload = dict(vars(settings))
    for env_name, value in env_overrides.items():
        attr = ENV_ATTR_MAP.get(env_name, env_name.lower())
        if value is None:
            if isinstance(payload.get(attr), bool):
                payload[attr] = False
            elif isinstance(payload.get(attr), int):
                payload[attr] = 0
            else:
                payload[attr] = ""
            continue
        current = payload.get(attr)
        if isinstance(current, bool):
            payload[attr] = str(value).strip().lower() == "true"
        elif isinstance(current, int) and not isinstance(current, bool):
            payload[attr] = int(value)
        else:
            payload[attr] = value
    return SimpleNamespace(**payload)


def _build_runtime(settings: SimpleNamespace) -> SimpleNamespace:
    overlay_store = OverlayStore.from_settings(settings)
    registry = AgentRegistry(Path(settings.agents_dir), overlay_dir=Path(settings.control_panel_agent_overlays_dir))
    doc_store = _FakeDocStore()
    chunk_store = _FakeChunkStore()
    skill_store = _FakeSkillStore(
        [
            SkillPackRecord(
                skill_id="skill-1",
                name="Pinned Skill",
                agent_scope="general",
                checksum="checksum-1",
                tenant_id=settings.default_tenant_id,
                body_markdown="# Pinned Skill\nagent_scope: general\n",
                owner_user_id=settings.default_user_id,
                version_parent="skill-1",
            )
        ]
    )
    stores = SimpleNamespace(
        doc_store=doc_store,
        chunk_store=chunk_store,
        collection_store=_FakeCollectionStore(),
        skill_store=skill_store,
        graph_index_store=_FakeGraphIndexStore(),
        graph_index_source_store=_FakeGraphIndexSourceStore(),
        graph_index_run_store=_FakeGraphIndexRunStore(),
        graph_query_cache_store=_FakeGraphQueryCacheStore(),
    )
    prompt_builder = _FakePromptBuilder(settings, overlay_store)
    kernel = _FakeKernel(settings, registry, prompt_builder)
    bot = SimpleNamespace(ctx=SimpleNamespace(stores=stores), kernel=kernel)
    return SimpleNamespace(settings=settings, bot=bot)


class _FakeManager:
    def __init__(self, settings: SimpleNamespace, runtime: SimpleNamespace | None = None) -> None:
        self._settings = settings
        self._runtime = runtime or _build_runtime(settings)
        self._last_reload: Dict[str, Any] = {
            "status": "success",
            "timestamp": "2026-04-08T10:00:00Z",
            "reason": "startup",
            "actor": "system",
            "changed_keys": [],
            "error": "",
        }
        self.preview_error: str = ""
        self.reload_error: str = ""

    def get_settings(self) -> SimpleNamespace:
        return self._settings

    def get_snapshot(self) -> SimpleNamespace:
        return self._runtime

    def get_overlay_store(self) -> OverlayStore:
        return OverlayStore.from_settings(self._settings)

    def preview_snapshot(self, *, env_overrides: Dict[str, str | None] | None = None) -> SimpleNamespace:
        if self.preview_error:
            raise RuntimeError(self.preview_error)
        preview_settings = _merge_settings(self._settings, env_overrides or {})
        return SimpleNamespace(settings=preview_settings, bot=self._runtime.bot)

    def reload_runtime(self, *, reason: str, actor: str = "control-panel", changed_keys: List[str] | None = None) -> Dict[str, Any]:
        if self.reload_error:
            self._last_reload = {
                "status": "failed",
                "timestamp": "2026-04-08T10:01:00Z",
                "reason": reason,
                "actor": actor,
                "changed_keys": list(changed_keys or []),
                "error": self.reload_error,
            }
            return dict(self._last_reload)
        overlay_values = self.get_overlay_store().read_runtime_env()
        self._settings = _merge_settings(self._settings, overlay_values)
        self._runtime.settings = self._settings
        self._last_reload = {
            "status": "success",
            "timestamp": "2026-04-08T10:01:00Z",
            "reason": reason,
            "actor": actor,
            "changed_keys": list(changed_keys or []),
            "error": "",
        }
        return dict(self._last_reload)

    def reload_agents(self, *, actor: str = "control-panel", changed_keys: List[str] | None = None) -> Dict[str, Any]:
        try:
            self._runtime.bot.kernel.registry = AgentRegistry(
                Path(self._settings.agents_dir),
                overlay_dir=Path(self._settings.control_panel_agent_overlays_dir),
            )
        except Exception as exc:
            self._last_reload = {
                "status": "failed",
                "timestamp": "2026-04-08T10:02:00Z",
                "reason": "agent_reload",
                "actor": actor,
                "changed_keys": list(changed_keys or []),
                "error": str(exc),
            }
            return dict(self._last_reload)
        self._last_reload = {
            "status": "success",
            "timestamp": "2026-04-08T10:02:00Z",
            "reason": "agent_reload",
            "actor": actor,
            "changed_keys": list(changed_keys or []),
            "error": "",
        }
        return dict(self._last_reload)

    def last_reload_summary(self) -> Dict[str, Any]:
        return dict(self._last_reload)


@asynccontextmanager
async def _admin_client(manager: _FakeManager) -> AsyncIterator[httpx.AsyncClient]:
    api_main.app.dependency_overrides[get_runtime_manager] = lambda: manager
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://testserver",
    )
    try:
        yield client
    finally:
        await client.aclose()
        api_main.app.dependency_overrides.clear()


def _admin_headers(token: str = "admin-token") -> Dict[str, str]:
    return {"X-Admin-Token": token}


@pytest.mark.asyncio
async def test_admin_overview_includes_runtime_diagnostics(tmp_path: Path) -> None:
    settings = _control_panel_settings(tmp_path)
    manager = _FakeManager(settings)

    async with _admin_client(manager) as client:
        response = await client.get("/v1/admin/overview", headers=_admin_headers())

    payload = response.json()

    assert response.status_code == 200
    assert payload["models"]["graphrag_chat_model"] == settings.graphrag_chat_model
    assert payload["models"]["graphrag_index_chat_model"] == settings.graphrag_index_chat_model
    assert payload["runtime_diagnostics"]["settings_fingerprint"]
    assert payload["runtime_diagnostics"]["loaded_overlay_env_path"].endswith("runtime.env")
    assert payload["runtime_diagnostics"]["process_started_at"]


def _fake_kb_status(runtime: SimpleNamespace, collection_id: str) -> Any:
    docs = runtime.bot.ctx.stores.doc_store.search_by_metadata(
        tenant_id=runtime.settings.default_tenant_id,
        collection_id=collection_id,
        limit=500,
    )
    return SimpleNamespace(
        ready=bool(docs),
        reason="indexed" if docs else "empty",
        collection_id=collection_id,
        missing_source_paths=[],
        indexed_doc_count=len(docs),
        suggested_fix="" if docs else f"python run.py sync-kb --collection-id {collection_id}",
    )


def _make_fake_ingest() -> Any:
    def fake_ingest_paths(
        settings: Any,
        stores: Any,
        paths: List[Path],
        *,
        source_type: str,
        tenant_id: str,
        collection_id: str | None = None,
        source_display_paths: Dict[str, str] | None = None,
        source_identities: Dict[str, str] | None = None,
        source_metadata_by_path: Dict[str, Dict[str, Any]] | None = None,
        metadata_profile: str = "auto",
        metadata_enrichment: str = "deterministic",
    ) -> List[str]:
        doc_ids: List[str] = []
        effective_collection = str(collection_id or settings.default_collection_id)
        stores.collection_store.ensure_collection(tenant_id=tenant_id, collection_id=effective_collection)
        for path in paths:
            title = Path(path).name
            content = Path(path).read_text(encoding="utf-8", errors="replace")
            doc_id = f"{effective_collection}-{title.replace('.', '-')}"
            record = DocumentRecord(
                doc_id=doc_id,
                tenant_id=tenant_id,
                collection_id=effective_collection,
                title=title,
                source_type=source_type,
                content_hash=f"hash-{title}",
                source_path=str(path),
                num_chunks=1,
                ingested_at="2026-04-08T10:03:00Z",
                file_type=Path(path).suffix.lstrip("."),
                doc_structure_type="general",
                source_display_path=(source_display_paths or {}).get(str(path), title),
                source_identity=(source_identities or {}).get(str(path), f"path:{path}"),
                source_metadata={
                    **dict((source_metadata_by_path or {}).get(str(path)) or {}),
                    "metadata_profile": metadata_profile,
                    "metadata_enrichment": metadata_enrichment,
                    "index_metadata": {
                        "extractor_version": "document_index_metadata_v1",
                        "metadata_profile": metadata_profile,
                        "metadata_enrichment": metadata_enrichment,
                        "doc_structure_type": "general",
                        "tags": ["general"],
                        "parser_chain": ["fake"],
                        "warnings": [],
                    },
                },
            )
            stores.doc_store.upsert_document(record)
            stores.chunk_store.set_document_chunks(
                doc_id,
                [
                    ChunkRecord(
                        chunk_id=f"{doc_id}#0",
                        doc_id=doc_id,
                        tenant_id=tenant_id,
                        collection_id=effective_collection,
                        chunk_index=0,
                        content=content,
                    )
                ],
            )
            doc_ids.append(doc_id)
        return doc_ids

    return fake_ingest_paths


def _write_session_events(settings: SimpleNamespace, session_id: str, events: List[Dict[str, Any]]) -> None:
    paths = RuntimePaths.from_settings(settings)
    event_path = paths.session_events_path(session_id)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(event, ensure_ascii=False) for event in events]
    event_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def test_overlay_store_lists_template_prompt_files_and_timestamps(tmp_path: Path) -> None:
    settings = _control_panel_settings(tmp_path)
    store = OverlayStore.from_settings(settings)

    store.write_prompt_overlay("grounded_answer.txt", "Override grounded answer")
    store.write_prompt_overlay("general_agent.md", "Override agent prompt")
    store.append_audit_event(action="prompt_overlay_write", actor="tester", details={"prompt_file": "grounded_answer.txt"})

    assert store.list_prompt_overlays() == ["general_agent.md", "grounded_answer.txt"]
    event = store.read_audit_events(limit=1)[0]
    assert event["action"] == "prompt_overlay_write"
    assert event["timestamp"]


def test_prompt_loader_prefers_control_panel_overlay_for_template_prompts(tmp_path: Path) -> None:
    settings = _control_panel_settings(tmp_path)
    overlay_path = Path(settings.control_panel_prompt_overlays_dir) / "grounded_answer.txt"
    overlay_path.write_text("Overlay grounded answer", encoding="utf-8")

    assert load_grounded_answer_prompt(settings) == "Overlay grounded answer"


@pytest.mark.asyncio
async def test_admin_auth_enforces_missing_invalid_disabled_and_unconfigured_states(tmp_path: Path) -> None:
    enabled_manager = _FakeManager(_control_panel_settings(tmp_path / "enabled"))
    async with _admin_client(enabled_manager) as client:
        unauthorized = await client.get("/v1/admin/config/effective")
        wrong_token = await client.get("/v1/admin/config/effective", headers=_admin_headers("wrong-token"))

    disabled_manager = _FakeManager(_control_panel_settings(tmp_path / "disabled", enabled=False))
    async with _admin_client(disabled_manager) as client:
        disabled = await client.get("/v1/admin/config/effective", headers=_admin_headers())

    unconfigured_manager = _FakeManager(_control_panel_settings(tmp_path / "unconfigured", token=""))
    async with _admin_client(unconfigured_manager) as client:
        unconfigured = await client.get("/v1/admin/config/effective", headers=_admin_headers())

    assert unauthorized.status_code == 401
    assert wrong_token.status_code == 401
    assert disabled.status_code == 404
    assert unconfigured.status_code == 503


@pytest.mark.asyncio
async def test_capabilities_route_reports_full_support_and_missing_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _control_panel_settings(tmp_path)
    manager = _FakeManager(settings)

    async with _admin_client(manager) as client:
        response = await client.get("/v1/admin/capabilities", headers=_admin_headers())

        full_payload = response.json()
        assert response.status_code == 200
        assert full_payload["schema_version"] == "1"
        assert full_payload["contract_version"] == "control-panel-v1"
        assert full_payload["compatible"] is True
        assert set(full_payload["sections"]) == {
            "dashboard",
            "architecture",
            "config",
            "agents",
            "prompts",
            "collections",
            "uploads",
            "graphs",
            "skills",
            "access",
            "mcp",
            "operations",
        }
        assert full_payload["sections"]["architecture"]["supported"] is True
        assert full_payload["sections"]["architecture"]["missing_routes"] == []

        required = {
            route
            for routes in control_panel_routes.CONTROL_PANEL_REQUIRED_ROUTES.values()
            for route in routes
        }
        degraded_paths = required - {
            "/v1/admin/architecture",
            "/v1/admin/architecture/activity",
            "/v1/admin/capabilities",
        }
        monkeypatch.setattr(control_panel_routes, "_registered_route_paths", lambda _request: degraded_paths)

        degraded_response = await client.get("/v1/admin/capabilities", headers=_admin_headers())

    degraded_payload = degraded_response.json()
    assert degraded_response.status_code == 200
    assert degraded_payload["compatible"] is False
    assert degraded_payload["sections"]["dashboard"]["supported"] is True
    assert degraded_payload["sections"]["architecture"]["supported"] is False
    assert degraded_payload["sections"]["architecture"]["missing_routes"] == [
        "/v1/admin/architecture",
        "/v1/admin/architecture/activity",
    ]
    assert degraded_payload["sections"]["architecture"]["reason"]


def test_tool_catalog_serializes_deferred_metadata() -> None:
    catalog = control_panel_routes._serialize_tool_catalog()
    by_name = {item["name"]: item for item in catalog}

    assert by_name["search_graph_index"]["should_defer"] is True
    assert "relationship" in by_name["search_graph_index"]["search_hint"]
    assert by_name["search_graph_index"]["eager_for_agents"] == ["graph_manager"]
    assert by_name["list_graph_indexes"]["should_defer"] is False


@pytest.mark.asyncio
async def test_admin_config_validate_apply_and_failed_reload_restore_previous_overlay(tmp_path: Path) -> None:
    settings = _control_panel_settings(tmp_path)
    manager = _FakeManager(settings)
    manager.get_overlay_store().write_runtime_env({"OLLAMA_CHAT_MODEL": "granite-3.1:8b"})

    async with _admin_client(manager) as client:
        schema = await client.get("/v1/admin/config/schema", headers=_admin_headers())
        effective = await client.get("/v1/admin/config/effective", headers=_admin_headers())
        validated = await client.post(
            "/v1/admin/config/validate",
            headers=_admin_headers(),
            json={
                "changes": {
                    "MAX_AGENT_STEPS": "9",
                    "CHAT_MAX_OUTPUT_TOKENS": "",
                    "CLARIFICATION_SENSITIVITY": "75",
                }
            },
        )
        runtime_env_after_validate = manager.get_overlay_store().read_runtime_env()

        applied = await client.post(
            "/v1/admin/config/apply",
            headers=_admin_headers(),
            json={"changes": {"MAX_AGENT_STEPS": "9", "CLARIFICATION_SENSITIVITY": "75"}},
        )

        manager.reload_error = "Provider initialization failed."
        failed = await client.post(
            "/v1/admin/config/apply",
            headers=_admin_headers(),
            json={"changes": {"MAX_AGENT_STEPS": "11"}},
        )

    schema_payload = schema.json()
    effective_payload = effective.json()
    validated_payload = validated.json()
    applied_payload = applied.json()
    failed_payload = failed.json()

    assert schema.status_code == 200
    field_names = {field["env_name"] for field in schema_payload["fields"]}
    assert "CHAT_MAX_OUTPUT_TOKENS" in field_names
    assert "AGENT_GENERAL_MAX_OUTPUT_TOKENS" in field_names
    assert validated_payload["runtime_diagnostics"]["settings_fingerprint"]
    assert "WORKER_JOB_WAIT_TIMEOUT_SECONDS" in field_names
    assert "LLM_HTTP_TIMEOUT_SECONDS" in field_names
    assert "LLM_HTTP_CONNECT_TIMEOUT_SECONDS" in field_names
    clarification_field = next(field for field in schema_payload["fields"] if field["env_name"] == "CLARIFICATION_SENSITIVITY")
    assert clarification_field["ui_control"] == "slider"
    assert clarification_field["min_value"] == 0
    assert clarification_field["max_value"] == 100
    assert clarification_field["step"] == 5
    assert effective.status_code == 200
    assert effective_payload["overlay_values"]["OLLAMA_CHAT_MODEL"] == "granite-3.1:8b"
    assert effective_payload["values"]["TAVILY_API_KEY"] != settings.tavily_api_key
    assert effective_payload["values"]["TAVILY_API_KEY"].endswith("1234")
    assert validated_payload["valid"] is True
    assert validated_payload["normalized_changes"]["CHAT_MAX_OUTPUT_TOKENS"] is None
    assert validated_payload["preview_diff"]["MAX_AGENT_STEPS"]["after"] == "9"
    assert validated_payload["preview_diff"]["CLARIFICATION_SENSITIVITY"]["after"] == "75"
    assert runtime_env_after_validate == {"OLLAMA_CHAT_MODEL": "granite-3.1:8b"}
    assert applied_payload["applied"] is True
    assert manager.get_overlay_store().read_runtime_env()["MAX_AGENT_STEPS"] == "9"
    assert manager.get_overlay_store().read_runtime_env()["CLARIFICATION_SENSITIVITY"] == "75"
    assert failed_payload["valid"] is False
    assert failed_payload["errors"]["runtime"] == "Provider initialization failed."
    assert manager.get_overlay_store().read_runtime_env()["MAX_AGENT_STEPS"] == "9"


@pytest.mark.asyncio
async def test_admin_config_validate_rejects_invalid_clarification_sensitivity(tmp_path: Path) -> None:
    settings = _control_panel_settings(tmp_path)
    manager = _FakeManager(settings)

    async with _admin_client(manager) as client:
        below = await client.post(
            "/v1/admin/config/validate",
            headers=_admin_headers(),
            json={"changes": {"CLARIFICATION_SENSITIVITY": "-1"}},
        )
        above = await client.post(
            "/v1/admin/config/validate",
            headers=_admin_headers(),
            json={"changes": {"CLARIFICATION_SENSITIVITY": "101"}},
        )
        invalid = await client.post(
            "/v1/admin/config/validate",
            headers=_admin_headers(),
            json={"changes": {"CLARIFICATION_SENSITIVITY": "high"}},
        )

    assert below.json()["valid"] is False
    assert below.json()["errors"]["CLARIFICATION_SENSITIVITY"] == "must be greater than or equal to 0"
    assert above.json()["valid"] is False
    assert above.json()["errors"]["CLARIFICATION_SENSITIVITY"] == "must be less than or equal to 100"
    assert invalid.json()["valid"] is False
    assert "invalid literal for int()" in invalid.json()["errors"]["CLARIFICATION_SENSITIVITY"]


@pytest.mark.asyncio
async def test_prompt_routes_list_update_and_reset(tmp_path: Path) -> None:
    settings = _control_panel_settings(tmp_path)
    manager = _FakeManager(settings)

    async with _admin_client(manager) as client:
        listed = await client.get("/v1/admin/prompts", headers=_admin_headers())
        fetched = await client.get("/v1/admin/prompts/general_agent.md", headers=_admin_headers())
        updated = await client.put(
            "/v1/admin/prompts/general_agent.md",
            headers=_admin_headers(),
            json={"content": "Updated prompt"},
        )
        reset = await client.delete("/v1/admin/prompts/general_agent.md", headers=_admin_headers())
        fetched_again = await client.get("/v1/admin/prompts/general_agent.md", headers=_admin_headers())

    assert listed.status_code == 200
    assert any(item["prompt_file"] == "general_agent.md" for item in listed.json()["prompts"])
    assert fetched.json()["effective_content"] == "Base general prompt"
    assert updated.json()["saved"] is True
    assert reset.json()["removed"] is True
    assert fetched_again.json()["effective_content"] == "Base general prompt"


@pytest.mark.asyncio
async def test_agent_routes_round_trip_overlay_reload_and_pinned_skills(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _control_panel_settings(tmp_path)
    runtime = _build_runtime(settings)
    manager = _FakeManager(settings, runtime=runtime)

    monkeypatch.setattr(
        control_panel_routes,
        "build_tool_definitions",
        lambda _providers: {
            "calculator": SimpleNamespace(
                name="calculator",
                group="core",
                description="Simple math",
                read_only=True,
                destructive=False,
                background_safe=True,
                requires_workspace=False,
                concurrency_key="",
                serializer="json",
                metadata={},
            )
        },
    )

    async with _admin_client(manager) as client:
        listed = await client.get("/v1/admin/agents", headers=_admin_headers())
        updated = await client.put(
            "/v1/admin/agents/general",
            headers=_admin_headers(),
            json={
                "description": "overlaid general agent",
                "preload_skill_packs": ["skill-1"],
                "body": "Overlay body",
            },
        )
        reloaded = await client.post("/v1/admin/agents/reload", headers=_admin_headers(), json={"changes": {}})
        fetched = await client.get("/v1/admin/agents/general", headers=_admin_headers())
        reset = await client.delete("/v1/admin/agents/general", headers=_admin_headers())
        reloaded_reset = await client.post("/v1/admin/agents/reload", headers=_admin_headers(), json={"changes": {}})
        fetched_reset = await client.get("/v1/admin/agents/general", headers=_admin_headers())

    assert listed.status_code == 200
    assert listed.json()["tools"][0]["name"] == "calculator"
    assert updated.json()["saved"] is True
    assert updated.json()["pending_reload"] is True
    assert reloaded.json()["status"] == "success"
    assert fetched.json()["description"] == "overlaid general agent"
    assert fetched.json()["preload_skill_packs"] == ["skill-1"]
    assert fetched.json()["pinned_skills"][0]["skill_id"] == "skill-1"
    assert reset.json()["removed"] is True
    assert reloaded_reset.json()["status"] == "success"
    assert fetched_reset.json()["description"] == "general agent"
    assert fetched_reset.json()["preload_skill_packs"] == []


@pytest.mark.asyncio
async def test_architecture_routes_reflect_live_registry_and_activity(tmp_path: Path) -> None:
    settings = _control_panel_settings(tmp_path)
    (Path(settings.agents_dir) / "basic.md").write_text(
        _agent_markdown(
            name="basic",
            mode="basic",
            description="basic chat path",
            prompt_file="general_agent.md",
            metadata='{"role_kind": "top_level", "entry_path": "router_basic"}',
        ),
        encoding="utf-8",
    )
    (Path(settings.agents_dir) / "coordinator.md").write_text(
        _agent_markdown(
            name="coordinator",
            mode="coordinator",
            description="manager path",
            prompt_file="general_agent.md",
            allowed_worker_agents='["memory_maintainer"]',
            metadata='{"role_kind": "manager", "entry_path": "router_or_delegated"}',
        ),
        encoding="utf-8",
    )
    (Path(settings.agents_dir) / "data_analyst.md").write_text(
        _agent_markdown(
            name="data_analyst",
            description="data specialist",
            prompt_file="general_agent.md",
            allowed_tools='["load_dataset", "execute_code", "return_file"]',
            metadata='{"role_kind": "top_level", "entry_path": "router_fast_path_or_delegated"}',
        ),
        encoding="utf-8",
    )
    (Path(settings.agents_dir) / "rag_worker.md").write_text(
        _agent_markdown(
            name="rag_worker",
            mode="rag",
            description="grounded lookup",
            prompt_file="general_agent.md",
            allowed_tools='["list_indexed_docs"]',
            metadata='{"role_kind": "top_level_or_worker", "expected_output": "rag_contract"}',
        ),
        encoding="utf-8",
    )
    (Path(settings.agents_dir) / "memory_maintainer.md").write_text(
        _agent_markdown(
            name="memory_maintainer",
            description="memory worker",
            prompt_file="general_agent.md",
            allowed_tools='["write_memory"]',
            memory_scopes='["conversation", "user"]',
            metadata='{"role_kind": "worker"}',
        ),
        encoding="utf-8",
    )

    runtime = _build_runtime(settings)
    manager = _FakeManager(settings, runtime=runtime)

    _write_session_events(
        settings,
        "tenant-1:user-1:conv-1",
        [
            {
                "event_type": "router_decision",
                "session_id": "tenant-1:user-1:conv-1",
                "created_at": "2026-04-08T10:00:00Z",
                "agent_name": "router",
                "payload": {
                    "conversation_id": "conv-1",
                    "route": "AGENT",
                    "router_method": "hybrid",
                    "reasons": ["document_grounding_intent"],
                    "suggested_agent": "rag_worker",
                },
            },
            {
                "event_type": "agent_turn_started",
                "session_id": "tenant-1:user-1:conv-1",
                "created_at": "2026-04-08T10:00:01Z",
                "agent_name": "general",
                "payload": {
                    "conversation_id": "conv-1",
                    "route": "AGENT",
                    "router_method": "hybrid",
                    "router_reasons": ["document_grounding_intent"],
                    "suggested_agent": "rag_worker",
                    "user_text": "sensitive user prompt that should not leak",
                },
            },
            {
                "event_type": "worker_agent_started",
                "session_id": "tenant-1:user-1:conv-1",
                "created_at": "2026-04-08T10:00:02Z",
                "agent_name": "memory_maintainer",
                "payload": {
                    "conversation_id": "conv-1",
                    "job_id": "job-1",
                    "route": "AGENT",
                },
            },
            {
                "event_type": "router_degraded_to_deterministic",
                "session_id": "tenant-1:user-1:conv-1",
                "created_at": "2026-04-08T10:00:03Z",
                "agent_name": "router",
                "payload": {
                    "conversation_id": "conv-1",
                    "router_method": "llm_fallback",
                    "reasons": ["llm_router_failed"],
                },
            },
        ],
    )
    _write_session_events(
        settings,
        "tenant-1:user-1:conv-2",
        [
            {
                "event_type": "basic_turn_started",
                "session_id": "tenant-1:user-1:conv-2",
                "created_at": "2026-04-08T10:05:00Z",
                "agent_name": "basic",
                "payload": {
                    "conversation_id": "conv-2",
                    "route": "BASIC",
                    "router_method": "deterministic",
                    "router_reasons": ["general_knowledge_or_small_talk"],
                    "user_text": "hello there",
                },
            }
        ],
    )

    async with _admin_client(manager) as client:
        snapshot = await client.get("/v1/admin/architecture", headers=_admin_headers())
        activity = await client.get("/v1/admin/architecture/activity", headers=_admin_headers())

        manager._settings = SimpleNamespace(**{**vars(manager._settings), "llm_router_mode": "llm_only"})
        manager.get_snapshot().settings = manager._settings
        snapshot_llm_only = await client.get("/v1/admin/architecture", headers=_admin_headers())

        (Path(settings.agents_dir) / "planner.md").write_text(
            _agent_markdown(
                name="planner",
                description="planner agent",
                prompt_file="general_agent.md",
                allowed_worker_agents='["memory_maintainer"]',
                preload_skill_packs='["skill-1"]',
                metadata='{"role_kind": "top_level", "entry_path": "default"}',
            ),
            encoding="utf-8",
        )
        reloaded = await client.post("/v1/admin/agents/reload", headers=_admin_headers(), json={"changes": {}})
        snapshot_after_reload = await client.get("/v1/admin/architecture", headers=_admin_headers())

    assert snapshot.status_code == 200
    snapshot_payload = snapshot.json()
    assert snapshot_payload["router"]["default_agent"] == "general"
    assert snapshot_payload["router"]["data_analyst_agent"] == "data_analyst"
    assert snapshot_payload["router"]["rag_agent"] == "rag_worker"
    node_labels = {node["label"] for node in snapshot_payload["nodes"]}
    assert {"User", "API Gateway", "Router", "general", "basic", "data_analyst", "rag_worker"} <= node_labels
    service_labels = {node["label"] for node in snapshot_payload["nodes"] if node["kind"] == "service"}
    assert {"Knowledge Base", "Job Manager", "Memory Store", "Python Sandbox"} <= service_labels
    edge_ids = {edge["id"] for edge in snapshot_payload["edges"]}
    assert "edge-router-basic-basic" in edge_ids
    assert "edge-router-rag-rag_worker" in edge_ids
    assert "edge-delegate-coordinator-memory_maintainer" in edge_ids
    assert any(path["id"] == "grounded-lookup" and path["target_agent"] == "rag_worker" for path in snapshot_payload["canonical_paths"])
    assert snapshot_payload["langgraph"]["status"] == "available"
    assert snapshot_payload["langgraph"]["agent_name"] == "general"
    assert "graph TD" in snapshot_payload["langgraph"]["mermaid"]
    assert snapshot_payload["langgraph"]["nodes"]
    assert snapshot_payload["langgraph"]["edges"]

    assert activity.status_code == 200
    activity_payload = activity.json()
    assert activity_payload["route_counts"] == {"AGENT": 1, "BASIC": 1}
    assert activity_payload["router_method_counts"] == {"hybrid": 1, "deterministic": 1}
    assert activity_payload["start_agent_counts"] == {"general": 1, "basic": 1}
    assert activity_payload["delegation_counts"] == {"memory_maintainer": 1}
    assert activity_payload["outcome_counts"] == {"negative": 1, "positive": 1}
    assert activity_payload["negative_rate_by_route"] == {"AGENT": 1.0, "BASIC": 0.0}
    assert activity_payload["review_backlog"]["pending"] == 1
    assert activity_payload["recent_mispicks"][0]["router_decision_id"] == "rtd_1"
    assert activity_payload["last_retrain_report"]["quarter"] == "2026-Q2"
    serialized_activity = json.dumps(activity_payload)
    assert "sensitive user prompt that should not leak" not in serialized_activity
    assert "hello there" not in serialized_activity

    assert snapshot_llm_only.json()["router"]["mode_label"] == "LLM primary"
    assert reloaded.json()["status"] == "success"
    reloaded_labels = {node["label"] for node in snapshot_after_reload.json()["nodes"]}
    assert "planner" in reloaded_labels
    assert "edge-delegate-planner-memory_maintainer" in {edge["id"] for edge in snapshot_after_reload.json()["edges"]}


@pytest.mark.asyncio
async def test_architecture_langgraph_export_failure_preserves_snapshot(tmp_path: Path) -> None:
    settings = _control_panel_settings(tmp_path)
    runtime = _build_runtime(settings)

    def fail_export(_agent_name: str = "") -> Dict[str, Any]:
        raise RuntimeError("langgraph export failed")

    runtime.bot.kernel.export_langgraph_react_graph = fail_export
    manager = _FakeManager(settings, runtime=runtime)

    async with _admin_client(manager) as client:
        snapshot = await client.get("/v1/admin/architecture", headers=_admin_headers())

    payload = snapshot.json()
    assert snapshot.status_code == 200
    assert {node["label"] for node in payload["nodes"]} >= {"User", "API Gateway", "Router", "general"}
    assert payload["edges"]
    assert payload["langgraph"]["status"] == "unavailable"
    assert payload["langgraph"]["warnings"] == ["langgraph export failed"]


@pytest.mark.asyncio
async def test_operations_route_exposes_scheduler_snapshot_and_enriched_job_metadata(tmp_path: Path) -> None:
    settings = _control_panel_settings(tmp_path)
    runtime = _build_runtime(settings)
    manager = _FakeManager(settings, runtime=runtime)

    async with _admin_client(manager) as client:
        response = await client.get("/v1/admin/operations", headers=_admin_headers())

    payload = response.json()
    assert response.status_code == 200
    assert payload["scheduler"]["enabled"] is True
    assert payload["scheduler"]["queue_depths"]["interactive"] == 1
    assert payload["jobs"][0]["scheduler_state"] == "completed"
    assert payload["jobs"][0]["estimated_token_cost"] == 320
    assert payload["jobs"][0]["queue_class"] == "interactive"


@pytest.mark.asyncio
async def test_collection_routes_ingest_upload_reindex_delete_and_keep_empty_collection_until_explicit_delete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _control_panel_settings(tmp_path)
    runtime = _build_runtime(settings)
    manager = _FakeManager(settings, runtime=runtime)
    path_file = tmp_path / "regional_spend.csv"
    kb_file = tmp_path / "kb-sync.md"
    path_file.write_text("regional spend content", encoding="utf-8")
    kb_file.write_text("kb sync content", encoding="utf-8")

    monkeypatch.setattr(control_panel_routes, "ingest_paths", _make_fake_ingest())
    monkeypatch.setattr(control_panel_routes, "iter_kb_source_paths", lambda _settings: [kb_file])
    monkeypatch.setattr(
        control_panel_routes,
        "get_kb_coverage_status",
        lambda settings_arg, stores_arg, tenant_id, collection_id: _fake_kb_status(runtime, collection_id)
    )

    async with _admin_client(manager) as client:
        initial = await client.get("/v1/admin/collections", headers=_admin_headers())
        ingested = await client.post(
            "/v1/admin/collections/smoke-control-panel/ingest-paths",
            headers=_admin_headers(),
            json={"paths": [str(path_file)], "source_type": "host_path"},
        )
        uploaded = await client.post(
            "/v1/admin/collections/smoke-control-panel/upload",
            headers=_admin_headers(),
            files=[("files", ("regional_controls.csv", b"regional controls content", "text/csv"))],
        )
        synced = await client.post(
            "/v1/admin/collections/smoke-sync/sync",
            headers=_admin_headers(),
            json={"changes": {}},
        )
        collections = await client.get("/v1/admin/collections", headers=_admin_headers())
        documents = await client.get("/v1/admin/collections/smoke-control-panel/documents", headers=_admin_headers())
        uploads = await client.get("/v1/admin/uploads", headers=_admin_headers())
        first_doc_id = documents.json()["documents"][0]["doc_id"]
        detail = await client.get(
            f"/v1/admin/collections/smoke-control-panel/documents/{first_doc_id}",
            headers=_admin_headers(),
        )
        reindexed = await client.post(
            f"/v1/admin/collections/smoke-control-panel/documents/{first_doc_id}/reindex",
            headers=_admin_headers(),
            json={"changes": {}},
        )
        documents_after_reindex = await client.get("/v1/admin/collections/smoke-control-panel/documents", headers=_admin_headers())
        for doc in list(documents_after_reindex.json()["documents"]):
            await client.delete(
                f"/v1/admin/collections/smoke-control-panel/documents/{doc['doc_id']}",
                headers=_admin_headers(),
            )
        final_collections = await client.get("/v1/admin/collections", headers=_admin_headers())
        deleted = await client.delete("/v1/admin/collections/smoke-control-panel", headers=_admin_headers())
        collections_after_delete = await client.get("/v1/admin/collections", headers=_admin_headers())

    assert initial.json()["collections"] == []
    assert ingested.json()["status"] == "success"
    assert ingested.json()["ingested_count"] == 1
    assert ingested.json()["files"][0]["source_type"] == "host_path"
    assert ingested.json()["files"][0]["display_path"] == "regional_spend.csv"
    assert uploaded.json()["status"] == "success"
    assert uploaded.json()["ingested_count"] == 1
    assert uploaded.json()["files"][0]["display_path"] == "regional_controls.csv"
    assert synced.json()["status"] == "success"
    assert synced.json()["ingested_count"] == 1
    listed_ids = [item["collection_id"] for item in collections.json()["collections"]]
    assert "smoke-control-panel" in listed_ids
    assert "smoke-sync" in listed_ids
    assert len(documents.json()["documents"]) == 2
    assert {item["source_type"] for item in documents.json()["documents"]} == {"host_path", "collection_upload"}
    assert any(item["title"] == "regional_controls.csv" for item in documents.json()["documents"])
    upload_payload = uploads.json()["uploads"]
    assert upload_payload == []
    detail_payload = detail.json()
    assert detail_payload["extracted_content"]["content"]
    assert detail_payload["raw_source"]["content"]
    assert reindexed.json()["ingested_doc_ids"] == [first_doc_id]
    final_ids = [item["collection_id"] for item in final_collections.json()["collections"]]
    assert "smoke-control-panel" in final_ids
    assert deleted.json()["deleted"] is True
    final_after_delete_ids = [item["collection_id"] for item in collections_after_delete.json()["collections"]]
    assert "smoke-control-panel" not in final_after_delete_ids
    assert "smoke-sync" in final_ids


@pytest.mark.asyncio
async def test_source_routes_scan_register_and_incremental_refresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _control_panel_settings(tmp_path)
    runtime = _build_runtime(settings)
    manager = _FakeManager(settings, runtime=runtime)
    source_root = settings.kb_dir / "repo-source"
    docs_dir = source_root / "docs"
    docs_dir.mkdir(parents=True)
    file_a = docs_dir / "a.md"
    file_b = docs_dir / "b.md"
    file_a.write_text("alpha graph content", encoding="utf-8")
    file_b.write_text("beta graph content", encoding="utf-8")
    (source_root / "ignored.tmp").write_text("ignored", encoding="utf-8")
    outside_path = tmp_path / "private-not-allowed.md"
    outside_path.write_text("secret", encoding="utf-8")

    monkeypatch.setattr(control_panel_routes, "ingest_paths", _make_fake_ingest())

    async with _admin_client(manager) as client:
        blocked = await client.post(
            "/v1/admin/sources/scan",
            headers=_admin_headers(),
            json={
                "paths": [str(outside_path)],
                "source_kind": "local_folder",
                "collection_id": "source-corpus",
            },
        )
        scan = await client.post(
            "/v1/admin/sources/scan",
            headers=_admin_headers(),
            json={
                "paths": [str(source_root)],
                "source_kind": "local_folder",
                "collection_id": "source-corpus",
                "include_globs": ["docs/**"],
            },
        )
        registered = await client.post(
            "/v1/admin/sources/register",
            headers=_admin_headers(),
            json={
                "paths": [str(source_root)],
                "source_kind": "local_folder",
                "collection_id": "source-corpus",
                "include_globs": ["docs/**"],
            },
        )
        source_id = registered.json()["source"]["source_id"]
        sources = await client.get("/v1/admin/sources", headers=_admin_headers())
        first_refresh = await client.post(
            f"/v1/admin/sources/{source_id}/refresh",
            headers=_admin_headers(),
            json={"index_preview": False},
        )
        first_docs = await client.get("/v1/admin/collections/source-corpus/documents", headers=_admin_headers())

        file_a.write_text("alpha graph content with a changed relationship", encoding="utf-8")
        file_b.unlink()
        file_c = docs_dir / "c.md"
        file_c.write_text("gamma graph content", encoding="utf-8")
        second_refresh = await client.post(
            f"/v1/admin/sources/{source_id}/refresh",
            headers=_admin_headers(),
            json={"index_preview": False},
        )
        second_docs = await client.get("/v1/admin/collections/source-corpus/documents", headers=_admin_headers())

    assert blocked.status_code == 200
    assert blocked.json()["summary"]["blocked_count"] == 1
    assert blocked.json()["blocked_paths"] == [str(outside_path.resolve())]
    assert scan.status_code == 200
    assert scan.json()["summary"]["supported_count"] == 2
    assert [item["display_path"] for item in scan.json()["supported_files"]] == ["docs/a.md", "docs/b.md"]
    assert registered.status_code == 200
    assert registered.json()["source"]["collection_id"] == "source-corpus"
    assert sources.json()["sources"][0]["source_id"] == source_id
    assert first_refresh.status_code == 200
    assert first_refresh.json()["ingested_count"] == 2
    assert first_refresh.json()["files"][0]["source_type"] == "local_folder"
    assert {doc["source_type"] for doc in first_docs.json()["documents"]} == {"local_folder"}
    assert second_refresh.status_code == 200
    assert second_refresh.json()["changes"]["changed_count"] == 1
    assert second_refresh.json()["changes"]["added_count"] == 1
    assert second_refresh.json()["changes"]["deleted_count"] == 1
    assert second_refresh.json()["deleted_doc_ids"] == ["source-corpus-b-md"]
    assert {doc["title"] for doc in second_docs.json()["documents"]} == {"a.md", "c.md"}


@pytest.mark.asyncio
async def test_uploaded_file_routes_manage_uploads_without_collection_listing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _control_panel_settings(tmp_path)
    runtime = _build_runtime(settings)
    manager = _FakeManager(settings, runtime=runtime)

    monkeypatch.setattr(control_panel_routes, "ingest_paths", _make_fake_ingest())

    async with _admin_client(manager) as client:
        uploaded = await client.post(
            "/v1/admin/uploads",
            headers=_admin_headers(),
            files=[("files", ("chat-context.txt", b"chat upload content", "text/plain"))],
        )
        collections = await client.get("/v1/admin/collections", headers=_admin_headers())
        uploads = await client.get("/v1/admin/uploads", headers=_admin_headers())
        doc_id = uploads.json()["uploads"][0]["doc_id"]
        detail = await client.get(f"/v1/admin/uploads/{doc_id}", headers=_admin_headers())
        reindexed = await client.post(
            f"/v1/admin/uploads/{doc_id}/reindex",
            headers=_admin_headers(),
            json={"changes": {}},
        )
        deleted = await client.delete(f"/v1/admin/uploads/{doc_id}", headers=_admin_headers())
        uploads_after_delete = await client.get("/v1/admin/uploads", headers=_admin_headers())

    assert uploaded.status_code == 200
    assert uploaded.json()["status"] == "success"
    assert uploaded.json()["collection_id"] == "control-panel-uploads"
    assert uploaded.json()["metadata_summary"]["document_count"] == 1
    assert collections.json()["collections"] == []
    assert uploads.json()["uploads"][0]["title"] == "chat-context.txt"
    assert detail.json()["document"]["source_type"] == "upload"
    assert detail.json()["metadata_summary"]["metadata_profile"] == "auto"
    assert reindexed.json()["ingested_doc_ids"] == [doc_id]
    assert deleted.json()["deleted"] is True
    assert uploads_after_delete.json()["uploads"] == []


@pytest.mark.asyncio
async def test_collection_upload_returns_partial_result_when_one_file_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _control_panel_settings(tmp_path)
    runtime = _build_runtime(settings)
    manager = _FakeManager(settings, runtime=runtime)

    def fake_ingest_with_failure(
        settings: Any,
        stores: Any,
        paths: List[Path],
        *,
        source_type: str,
        tenant_id: str,
        collection_id: str | None = None,
        source_display_paths: Dict[str, str] | None = None,
        source_identities: Dict[str, str] | None = None,
        source_metadata_by_path: Dict[str, Dict[str, Any]] | None = None,
        metadata_profile: str = "auto",
        metadata_enrichment: str = "deterministic",
    ) -> List[str]:
        target = paths[0]
        if target.name == "bad.docx":
            raise RuntimeError("parser failed")
        return _make_fake_ingest()(
            settings,
            stores,
            paths,
            source_type=source_type,
            tenant_id=tenant_id,
            collection_id=collection_id,
            source_display_paths=source_display_paths,
            source_identities=source_identities,
            source_metadata_by_path=source_metadata_by_path,
            metadata_profile=metadata_profile,
            metadata_enrichment=metadata_enrichment,
        )

    monkeypatch.setattr(control_panel_routes, "ingest_paths", fake_ingest_with_failure)

    async with _admin_client(manager) as client:
        response = await client.post(
            "/v1/admin/collections/partial-upload/upload",
            headers=_admin_headers(),
            files=[
                ("files", ("good.txt", b"good upload", "text/plain")),
                ("files", ("bad.docx", b"bad upload", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
            ],
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "partial"
    assert payload["ingested_count"] == 1
    assert payload["failed_count"] == 1
    assert any(item["display_path"] == "good.txt" and item["outcome"] == "ingested" for item in payload["files"])
    assert any(item["display_path"] == "bad.docx" and item["outcome"] == "failed" for item in payload["files"])
    assert "parser failed" in payload["errors"][0]


@pytest.mark.asyncio
async def test_collection_upload_preserves_browser_relative_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _control_panel_settings(tmp_path)
    runtime = _build_runtime(settings)
    manager = _FakeManager(settings, runtime=runtime)

    monkeypatch.setattr(control_panel_routes, "ingest_paths", _make_fake_ingest())

    async with _admin_client(manager) as client:
        response = await client.post(
            "/v1/admin/collections/folder-upload/upload",
            headers=_admin_headers(),
            files=[
                ("relative_paths", (None, "alpha/same.txt")),
                ("relative_paths", (None, "beta/same.txt")),
                ("files", ("same.txt", b"alpha upload", "text/plain")),
                ("files", ("same.txt", b"beta upload", "text/plain")),
            ],
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "success"
    assert payload["display_paths"] == ["alpha/same.txt", "beta/same.txt"]
    assert payload["files"][0]["source_path"].endswith("alpha/same.txt")
    assert payload["files"][1]["source_path"].endswith("beta/same.txt")
    assert all(item["outcome"] == "ingested" for item in payload["files"])


@pytest.mark.asyncio
async def test_collection_path_ingest_ignores_hidden_system_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _control_panel_settings(tmp_path)
    runtime = _build_runtime(settings)
    manager = _FakeManager(settings, runtime=runtime)
    corpus_dir = tmp_path / "mixed-folder"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "notes.txt").write_text("notes", encoding="utf-8")
    (corpus_dir / ".DS_Store").write_text("junk", encoding="utf-8")

    monkeypatch.setattr(control_panel_routes, "ingest_paths", _make_fake_ingest())

    async with _admin_client(manager) as client:
        response = await client.post(
            "/v1/admin/collections/hidden-filter/ingest-paths",
            headers=_admin_headers(),
            json={"paths": [str(corpus_dir)]},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "partial"
    assert payload["ingested_count"] == 1
    assert payload["skipped_count"] == 1
    assert any(item["display_path"] == "notes.txt" and item["outcome"] == "ingested" for item in payload["files"])
    assert any(item["display_path"] == ".DS_Store" and item["outcome"] == "skipped" for item in payload["files"])


@pytest.mark.asyncio
async def test_collection_path_ingest_preview_returns_metadata_without_writing_docs(tmp_path: Path) -> None:
    settings = _control_panel_settings(tmp_path)
    runtime = _build_runtime(settings)
    manager = _FakeManager(settings, runtime=runtime)
    doc_path = tmp_path / "preview_spec.md"
    doc_path.write_text("# Preview Spec\n\nREQ-001 The gateway shall authenticate users.", encoding="utf-8")

    async with _admin_client(manager) as client:
        response = await client.post(
            "/v1/admin/collections/preview/ingest-paths",
            headers=_admin_headers(),
            json={"paths": [str(doc_path)], "metadata_profile": "auto", "index_preview": True},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["preview"] is True
    assert payload["files"][0]["outcome"] == "previewed"
    assert payload["metadata_summary"]["document_count"] == 1
    assert runtime.bot.ctx.stores.doc_store.records == {}


@pytest.mark.asyncio
async def test_collection_health_and_repair_prune_stale_duplicates(tmp_path: Path) -> None:
    settings = _control_panel_settings(tmp_path)
    runtime = _build_runtime(settings)
    manager = _FakeManager(settings, runtime=runtime)
    docs_dir = Path(settings.kb_extra_dirs[0])
    architecture_path = docs_dir / "ARCHITECTURE.md"
    architecture_text = "# Architecture\n\nLatest runtime design."
    architecture_path.write_text(architecture_text, encoding="utf-8")
    latest_hash = hashlib.sha1(architecture_text.encode("utf-8")).hexdigest()

    runtime.bot.ctx.stores.doc_store.upsert_document(
        DocumentRecord(
            doc_id="doc-arch-old",
            tenant_id=settings.default_tenant_id,
            collection_id="default",
            title="ARCHITECTURE.md",
            source_type="kb",
            content_hash="old-hash",
            source_path=str(architecture_path),
            num_chunks=2,
            ingested_at="2026-04-09T02:00:00Z",
            file_type="md",
            doc_structure_type="process_flow_doc",
        )
    )
    runtime.bot.ctx.stores.doc_store.upsert_document(
        DocumentRecord(
            doc_id="doc-arch-new",
            tenant_id=settings.default_tenant_id,
            collection_id="default",
            title="ARCHITECTURE.md",
            source_type="kb",
            content_hash=latest_hash,
            source_path=str(architecture_path),
            num_chunks=3,
            ingested_at="2026-04-09T03:00:00Z",
            file_type="md",
            doc_structure_type="process_flow_doc",
        )
    )

    async with _admin_client(manager) as client:
        health_before = await client.get("/v1/admin/collections/default/health", headers=_admin_headers())
        repaired = await client.post(
            "/v1/admin/collections/default/repair",
            headers=_admin_headers(),
            json={"changes": {}},
        )
        health_after = await client.get("/v1/admin/collections/default/health", headers=_admin_headers())

    assert health_before.status_code == 200
    assert health_before.json()["duplicate_group_count"] == 1
    assert health_before.json()["duplicate_groups"][0]["active_doc_id"] == "doc-arch-new"
    assert set(health_before.json()["duplicate_groups"][0]["duplicate_doc_ids"]) == {"doc-arch-old"}

    assert repaired.status_code == 200
    assert repaired.json()["deleted_doc_ids"] == ["doc-arch-old"]
    assert repaired.json()["health_after"]["duplicate_group_count"] == 0

    assert health_after.status_code == 200
    assert health_after.json()["duplicate_group_count"] == 0


@pytest.mark.asyncio
async def test_collection_health_and_repair_prune_canonical_upload_duplicates(tmp_path: Path) -> None:
    settings = _control_panel_settings(tmp_path)
    runtime = _build_runtime(settings)
    manager = _FakeManager(settings, runtime=runtime)
    repo_root = Path(__file__).resolve().parents[1]
    repo_path = repo_root / "defense_rag_test_corpus" / "documents" / "rfp-sample.md"
    container_path = Path("/app/defense_rag_test_corpus/documents/rfp-sample.md")

    runtime.bot.ctx.stores.collection_store.ensure_collection(
        tenant_id=settings.default_tenant_id,
        collection_id="rfp-corpus",
        maintenance_policy="indexed_documents",
    )
    runtime.bot.ctx.stores.doc_store.upsert_document(
        DocumentRecord(
            doc_id="doc-rfp-host",
            tenant_id=settings.default_tenant_id,
            collection_id="rfp-corpus",
            title="rfp-sample.md",
            source_type="upload",
            content_hash="hash-rfp",
            source_path=str(repo_path),
            source_identity=f"path:{repo_path}",
            num_chunks=1,
            ingested_at="2026-04-09T02:00:00Z",
            file_type="md",
            doc_structure_type="general",
        )
    )
    runtime.bot.ctx.stores.doc_store.upsert_document(
        DocumentRecord(
            doc_id="doc-rfp-container",
            tenant_id=settings.default_tenant_id,
            collection_id="rfp-corpus",
            title="rfp-sample.md",
            source_type="upload",
            content_hash="hash-rfp",
            source_path=str(container_path),
            source_identity=f"path:{container_path}",
            num_chunks=1,
            ingested_at="2026-04-09T03:00:00Z",
            file_type="md",
            doc_structure_type="general",
        )
    )

    async with _admin_client(manager) as client:
        health_before = await client.get("/v1/admin/collections/rfp-corpus/health", headers=_admin_headers())
        repaired = await client.post(
            "/v1/admin/collections/rfp-corpus/repair",
            headers=_admin_headers(),
            json={"changes": {}},
        )
        health_after = await client.get("/v1/admin/collections/rfp-corpus/health", headers=_admin_headers())
        uploads_after = await client.get("/v1/admin/uploads", headers=_admin_headers())

    assert health_before.status_code == 200
    assert health_before.json()["maintenance_policy"] == "indexed_documents"
    assert health_before.json()["duplicate_group_count"] == 0
    assert health_before.json()["duplicate_groups"] == []

    assert repaired.status_code == 200
    assert repaired.json()["deleted_doc_ids"] == ["doc-rfp-host"]
    assert repaired.json()["health_after"]["duplicate_group_count"] == 0

    assert health_after.status_code == 200
    assert health_after.json()["duplicate_group_count"] == 0
    assert [item["doc_id"] for item in uploads_after.json()["uploads"]] == ["doc-rfp-container"]


@pytest.mark.asyncio
async def test_admin_graph_routes_create_validate_build_and_bind_graph_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _control_panel_settings(tmp_path)
    runtime = _build_runtime(settings)
    manager = _FakeManager(settings, runtime=runtime)

    runtime.bot.ctx.stores.doc_store.upsert_document(
        DocumentRecord(
            doc_id="graph-doc-1",
            tenant_id=settings.default_tenant_id,
            collection_id="graph-collection",
            title="Release Readiness.md",
            source_type="upload",
            content_hash="hash-release-readiness",
            source_path=str(tmp_path / "uploads" / "release-readiness.md"),
            num_chunks=1,
            ingested_at="2026-04-09T04:00:00Z",
            file_type="md",
            doc_structure_type="general",
        )
    )
    runtime.bot.ctx.stores.chunk_store.set_document_chunks(
        "graph-doc-1",
        [
            ChunkRecord(
                chunk_id="graph-doc-1#0",
                doc_id="graph-doc-1",
                tenant_id=settings.default_tenant_id,
                collection_id="graph-collection",
                chunk_index=0,
                content="Release readiness depends on Finance approval and vendor onboarding.",
                section_title="Overview",
            )
        ],
    )

    class _FakeGraphBackend:
        backend_name = "microsoft_graphrag"
        supported_query_methods = ("local", "global", "drift")

        def __init__(self, runtime_settings: SimpleNamespace) -> None:
            self.settings = runtime_settings

        def validate_runtime(self) -> Dict[str, Any]:
            return {
                "ok": True,
                "provider": "openai",
                "chat_model": "gpt-oss:20b",
                "embed_model": "nomic-embed-text",
                "cli_available": True,
                "warnings": [],
            }

        def init_project(self, root_path: Path, *, chat_model: str, embed_model: str, force: bool = False) -> SimpleNamespace:
            del chat_model, embed_model, force
            (root_path / "output").mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(
                status="ready",
                detail="Initialized GraphRAG project.",
                warnings=[],
                capabilities=["catalog", "graphrag_cli"],
                supported_query_methods=["local", "global", "drift"],
                artifact_path=str(root_path / "output"),
                metadata={},
            )

        def index_documents(self, graph_id: str, root_path: Path, *, refresh: bool = False) -> SimpleNamespace:
            del graph_id, refresh
            output_dir = root_path / "output"
            log_dir = root_path / "logs"
            output_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "index.log").write_text("Indexed graph successfully.\n", encoding="utf-8")
            return SimpleNamespace(
                status="ready",
                detail="Indexed graph successfully.",
                warnings=[],
                capabilities=["catalog", "graphrag_cli", "admin_managed"],
                supported_query_methods=["local", "global", "drift"],
                artifact_path=str(output_dir),
                query_ready=True,
                query_backend="graphrag_python_api_preferred",
                artifact_tables=["entities", "relationships", "text_units"],
                artifact_mtime="2026-04-09T04:30:00Z",
                graph_context_summary={"communities": 2, "entities": 5},
                metadata={"indexer": "fake-test-backend"},
            )

    monkeypatch.setattr(GraphService, "_backend_for", lambda self, backend_name: _FakeGraphBackend(self.settings))

    async with _admin_client(manager) as client:
        initial = await client.get("/v1/admin/graphs", headers=_admin_headers())
        created = await client.post(
            "/v1/admin/graphs",
            headers=_admin_headers(),
            json={
                "graph_id": "release-risk",
                "display_name": "Release Risk",
                "collection_id": "graph-collection",
                "source_doc_ids": ["graph-doc-1"],
                "prompt_overrides": {"extract_graph.txt": "Entity_types: {entity_types}\nText: {input_text}\nOutput:"},
                "config_overrides": {"extract_graph": {"entity_types": ["vendor", "approval"]}},
            },
        )
        tuned = await client.post(
            "/v1/admin/graphs/release_risk/research-tune",
            headers=_admin_headers(),
            json={
                "guidance": "Focus on release approvals, vendor onboarding, and finance dependencies.",
                "target_prompt_files": ["extract_graph.txt"],
                "actor": "tester",
            },
        )
        tune_run_id = tuned.json().get("run_id", "")
        tune_detail = await client.get(
            f"/v1/admin/graphs/release_risk/research-tune/{tune_run_id}",
            headers=_admin_headers(),
        )
        before_apply = await client.get("/v1/admin/graphs/release_risk", headers=_admin_headers())
        tune_apply = await client.post(
            f"/v1/admin/graphs/release_risk/research-tune/{tune_run_id}/apply",
            headers=_admin_headers(),
            json={"prompt_files": ["extract_graph.txt"], "actor": "tester"},
        )
        validated = await client.post(
            "/v1/admin/graphs/release_risk/validate",
            headers=_admin_headers(),
            json={"actor": "tester"},
        )
        built = await client.post(
            "/v1/admin/graphs/release_risk/build",
            headers=_admin_headers(),
            json={"actor": "tester"},
        )
        prompts = await client.put(
            "/v1/admin/graphs/release_risk/prompts",
            headers=_admin_headers(),
            json={"prompt_overrides": {"local_search_system_prompt.txt": "Stay grounded in section evidence."}},
        )
        skills = await client.put(
            "/v1/admin/graphs/release_risk/skills",
            headers=_admin_headers(),
            json={
                "skill_ids": ["skill-1"],
                "overlay_markdown": "# Release Risk Overlay\nagent_scope: rag\n\n## Workflow\n\n- Prefer approval-chain terminology.\n",
                "overlay_skill_name": "Release Risk Overlay",
            },
        )
        detail = await client.get("/v1/admin/graphs/release_risk", headers=_admin_headers())
        runs = await client.get("/v1/admin/graphs/release_risk/runs", headers=_admin_headers())
        progress = await client.get("/v1/admin/graphs/release_risk/progress", headers=_admin_headers())
        deleted = await client.request(
            "DELETE",
            "/v1/admin/graphs/release_risk",
            headers=_admin_headers(),
            json={"delete_artifacts": False, "actor": "tester"},
        )
        after_delete = await client.get("/v1/admin/graphs", headers=_admin_headers())

    assert initial.status_code == 200
    assert initial.json()["graphs"] == []

    created_payload = created.json()
    assert created.status_code == 200
    assert created_payload["graph_id"] == "release_risk"
    assert created_payload["graph"]["status"] == "draft"

    assert tuned.status_code == 200
    assert tuned.json()["status"] == "completed"
    assert tuned.json()["coverage"]["digested_doc_count"] == 1
    assert "extract_graph.txt" in tuned.json()["prompt_drafts"]
    assert tune_detail.status_code == 200
    assert "Release readiness depends on Finance approval" in tune_detail.json()["scratchpad_preview"]
    assert before_apply.json()["graph"]["prompt_overrides_json"]["extract_graph.txt"] == "Entity_types: {entity_types}\nText: {input_text}\nOutput:"
    assert tune_apply.status_code == 200
    assert tune_apply.json()["applied_prompt_files"] == ["extract_graph.txt"]
    assert "Dataset-Specific Curation Guidance" in tune_apply.json()["graph"]["prompt_overrides_json"]["extract_graph.txt"]

    assert validated.status_code == 200
    assert validated.json()["ok"] is True
    assert "extraction_preflight" in validated.json()

    built_payload = built.json()
    assert built.status_code == 200
    assert built_payload["status"] == "ready"
    assert built_payload["graph"]["query_ready"] is True
    assert built_payload["graph"]["query_backend"] == "graphrag_python_api_preferred"

    graph_root = Path(built_payload["graph"]["root_path"])
    assert (graph_root / "settings.yaml").exists()
    assert (graph_root / "prompts" / "extract_graph.txt").exists()
    assert (graph_root / "logs" / "index.log").exists()

    assert prompts.status_code == 200
    assert prompts.json()["graph"]["prompt_overrides_json"]["local_search_system_prompt.txt"] == "Stay grounded in section evidence."

    skills_payload = skills.json()
    assert skills.status_code == 200
    assert "skill-1" in skills_payload["graph"]["graph_skill_ids"]
    assert any(item["graph_id"] == "release_risk" for item in skills_payload["skills"])

    detail_payload = detail.json()
    assert detail.status_code == 200
    assert detail_payload["graph"]["display_name"] == "Release Risk"
    assert any(item["graph_id"] == "release_risk" for item in detail_payload["skills"])
    assert detail_payload["logs"][0]["name"] == "index.log"

    runs_payload = runs.json()
    assert runs.status_code == 200
    operations = [item["operation"] for item in runs_payload["runs"]]
    assert "create" in operations
    assert "validate" in operations
    assert "build" in operations
    assert "update" in operations
    assert "research_tune" in operations
    assert "research_tune_apply" in operations

    assert progress.status_code == 200
    assert progress.json()["graph_id"] == "release_risk"
    assert progress.json()["percent"] == 100.0
    assert progress.json()["logs"][0]["name"] == "index.log"

    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert deleted.json()["artifact_deleted"] is False
    assert graph_root.exists()
    assert after_delete.json()["graphs"] == []
