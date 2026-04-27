from __future__ import annotations

import datetime as dt
import json
import os
import re
import signal
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import httpx
import yaml

from agentic_chatbot_next.authz import access_summary_allows, access_summary_authz_enabled
from agentic_chatbot_next.graph.artifacts import load_artifact_bundle
from agentic_chatbot_next.graph.backend import (
    GraphOperationResult,
    GraphQueryHit,
    MicrosoftGraphRagBackend,
    Neo4jGraphImportBackend,
)
from agentic_chatbot_next.graph.community_report_recovery import (
    analyze_community_report_inputs,
)
from agentic_chatbot_next.graph.planner import plan_sources
from agentic_chatbot_next.persistence.postgres.entities import (
    CanonicalEntityRecord,
    EntityAliasRecord,
    EntityMentionRecord,
    make_alias_id,
    make_entity_id,
    make_mention_id,
)
from agentic_chatbot_next.persistence.postgres.graphs import (
    GraphIndexRecord,
    GraphIndexRunRecord,
    GraphIndexSourceRecord,
)
from agentic_chatbot_next.rag.ingest import canonicalize_local_source_path
from agentic_chatbot_next.rag.source_links import build_document_source_url
from agentic_chatbot_next.storage import blob_ref_from_record, build_blob_store


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_") or "graph"


def _dedupe(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        clean = str(item or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result


_GRAPH_QUERY_METHOD_ALIASES: dict[str, tuple[str, ...]] = {
    "graph": ("local", "global"),
    "graphrag": ("local", "global"),
    "knowledge_graph": ("local", "global"),
    "knowledge-graph": ("local", "global"),
    "relationship": ("local",),
    "relationships": ("local",),
    "multihop": ("local", "global"),
    "multi-hop": ("local", "global"),
    "multi_hop": ("local", "global"),
}


def _normalize_graph_query_methods(
    requested: Sequence[str] | None,
    *,
    supported: Sequence[str] | None,
    default_method: str = "local",
) -> tuple[List[str], List[str], Dict[str, List[str]]]:
    supported_methods = _dedupe(str(item).strip().lower() for item in (supported or []) if str(item).strip())
    if not supported_methods:
        supported_methods = _dedupe([str(default_method or "local").strip().lower() or "local"])
    raw_requested = [str(item).strip().lower() for item in (requested or []) if str(item).strip()]
    if not raw_requested:
        return list(supported_methods), [], {}

    normalized: List[str] = []
    invalid: List[str] = []
    aliases: Dict[str, List[str]] = {}
    for method in raw_requested:
        expanded = list(_GRAPH_QUERY_METHOD_ALIASES.get(method, (method,)))
        accepted = [item for item in expanded if item in supported_methods]
        if not accepted:
            invalid.append(method)
            continue
        aliases[method] = accepted
        normalized.extend(accepted)
    return _dedupe(normalized), _dedupe(invalid), aliases


def _expanded_graph_queries(query: str) -> List[str]:
    base = str(query or "").strip()
    if not base:
        return []
    return [base]


def _source_record_id(graph_id: str, source_doc_id: str, source_path: str) -> str:
    seed = f"{graph_id}:{source_doc_id}:{canonicalize_local_source_path(source_path)}"
    return f"graphsrc_{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex[:20]}"


def _source_lookup_keys(value: Any) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    keys = {text, text.lower()}
    try:
        path = Path(text)
        if path.name:
            keys.add(path.name)
            keys.add(path.name.lower())
        if path.stem:
            keys.add(path.stem)
            keys.add(path.stem.lower())
    except Exception:
        pass
    return [key for key in keys if key]


_WORKFLOW_STARTED_RE = re.compile(r"Workflow started: (?P<workflow>[A-Za-z0-9_]+)")
_PROGRESS_RE = re.compile(
    r"(?P<label>[A-Za-z0-9_ ]+?)\s+progress:\s*(?P<current>\d+)\s*/\s*(?P<total>\d+)",
    re.IGNORECASE,
)

_DEFAULT_EXTRACT_GRAPH_PREFLIGHT_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are clearly related to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation of why the source and target entities are related
- relationship_strength: a numeric score indicating relationship strength
Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use ## as the list delimiter.
4. When finished, output <|COMPLETE|>

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
""".strip()


class GraphService:
    def __init__(self, settings: Any, stores: Any, *, session: Any | None = None) -> None:
        self.settings = settings
        self.stores = stores
        self.session = session
        self.tenant_id = str(
            getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev"))
            or getattr(settings, "default_tenant_id", "local-dev")
            or "local-dev"
        )
        self.user_id = str(
            getattr(session, "user_id", getattr(settings, "default_user_id", ""))
            or getattr(settings, "default_user_id", "")
            or ""
        )

    def _backend_for(self, backend_name: str):
        resolved = str(backend_name or getattr(self.settings, "graph_backend", "microsoft_graphrag") or "microsoft_graphrag").strip().lower()
        if resolved == "neo4j":
            return Neo4jGraphImportBackend(self.settings)
        return MicrosoftGraphRagBackend(self.settings)

    def _graph_root(self, graph_id: str) -> Path:
        return Path(self.settings.graphrag_projects_dir) / self.tenant_id / graph_id

    def _session_access_summary(self) -> Dict[str, Any]:
        metadata = dict(getattr(self.session, "metadata", {}) or {}) if self.session is not None else {}
        return dict(metadata.get("access_summary") or {})

    def _authz_bypass(self) -> bool:
        metadata = dict(getattr(self.session, "metadata", {}) or {}) if self.session is not None else {}
        return bool(metadata.get("authz_bypass", False))

    def _authz_enabled(self) -> bool:
        return access_summary_authz_enabled(self._session_access_summary())

    def _graph_store_user_id(self) -> str:
        if self._authz_enabled() or self._authz_bypass():
            return "*"
        return self.user_id

    def _graph_allowed(self, record: GraphIndexRecord | None) -> bool:
        if record is None:
            return False
        if self._authz_bypass():
            return True
        access_summary = self._session_access_summary()
        if not access_summary_authz_enabled(access_summary):
            return True
        return access_summary_allows(
            access_summary,
            "graph",
            str(record.graph_id or ""),
            action="use",
        ) and access_summary_allows(
            access_summary,
            "collection",
            str(record.collection_id or ""),
            action="use",
            implicit_resource_id=str(access_summary.get("session_upload_collection_id") or ""),
        )

    def _get_index_record(self, store: Any, graph_id: str) -> GraphIndexRecord | None:
        try:
            record = store.get_index(graph_id, self.tenant_id, user_id=self._graph_store_user_id())
        except TypeError:
            record = store.get_index(graph_id, self.tenant_id)
        return record if self._graph_allowed(record) else None

    def _list_index_records(
        self,
        store: Any,
        *,
        collection_id: str = "",
        status: str = "",
        backend: str = "",
        limit: int = 100,
    ) -> List[GraphIndexRecord]:
        try:
            records = list(
                store.list_indexes(
                    tenant_id=self.tenant_id,
                    user_id=self._graph_store_user_id(),
                    collection_id=collection_id,
                    status=status,
                    backend=backend,
                    limit=limit,
                )
            )
        except TypeError:
            try:
                records = list(
                    store.list_indexes(
                        tenant_id=self.tenant_id,
                        user_id=self._graph_store_user_id(),
                        collection_id=collection_id,
                        limit=limit,
                    )
                )
            except TypeError:
                try:
                    records = list(
                        store.list_indexes(
                            tenant_id=self.tenant_id,
                            collection_id=collection_id,
                            status=status,
                            backend=backend,
                            limit=limit,
                        )
                    )
                except TypeError:
                    records = list(
                        store.list_indexes(
                            tenant_id=self.tenant_id,
                            collection_id=collection_id,
                            limit=limit,
                        )
                    )
        return [record for record in records if self._graph_allowed(record)]

    def _search_index_records(
        self,
        store: Any,
        query: str,
        *,
        collection_id: str = "",
        limit: int = 6,
    ) -> List[GraphIndexRecord]:
        try:
            records = list(
                store.search_indexes(
                    query,
                    tenant_id=self.tenant_id,
                    user_id=self._graph_store_user_id(),
                    collection_id=collection_id,
                    limit=limit,
                )
            )
        except TypeError:
            records = list(
                store.search_indexes(
                    query,
                    tenant_id=self.tenant_id,
                    collection_id=collection_id,
                    limit=limit,
                )
            )
        return [record for record in records if self._graph_allowed(record)]

    def _graph_store(self) -> Any | None:
        return getattr(self.stores, "graph_store", None)

    def _index_store(self):
        return getattr(self.stores, "graph_index_store", None)

    def _source_store(self):
        return getattr(self.stores, "graph_index_source_store", None)

    def _run_store(self):
        return getattr(self.stores, "graph_index_run_store", None)

    def _query_cache_store(self):
        return getattr(self.stores, "graph_query_cache_store", None)

    def _entity_store(self):
        return getattr(self.stores, "entity_store", None)

    def _skill_store(self):
        return getattr(self.stores, "skill_store", None)

    def _chunk_store(self):
        return getattr(self.stores, "chunk_store", None)

    def _graph_input_dir(self, graph_id: str) -> Path:
        return self._graph_root(graph_id) / "input"

    def _graph_prompts_dir(self, graph_id: str) -> Path:
        return self._graph_root(graph_id) / "prompts"

    def _graph_settings_path(self, graph_id: str) -> Path:
        return self._graph_root(graph_id) / "settings.yaml"

    def _graph_env_path(self, graph_id: str) -> Path:
        return self._graph_root(graph_id) / ".env"

    def _graph_log_dir(self, graph_id: str) -> Path:
        return self._graph_root(graph_id) / "logs"

    def _graph_runs(self, graph_id: str, *, limit: int = 20) -> List[GraphIndexRunRecord]:
        run_store = self._run_store()
        if run_store is None:
            return []
        return list(run_store.list_runs(graph_id, tenant_id=self.tenant_id, limit=limit))

    def _now_utc(self) -> dt.datetime:
        return dt.datetime.now(dt.timezone.utc)

    def _parse_iso_datetime(self, value: Any) -> dt.datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = dt.datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)

    def _graphrag_stale_run_after_seconds(self) -> int:
        return max(300, int(getattr(self.settings, "graphrag_stale_run_after_seconds", 1800) or 1800))

    def _latest_graph_log_activity_at(self, graph_id: str) -> dt.datetime | None:
        log_dir = self._graph_log_dir(graph_id)
        if not log_dir.exists():
            return None
        timestamps: List[float] = []
        for path in log_dir.rglob("*"):
            if not path.is_file():
                continue
            try:
                timestamps.append(path.stat().st_mtime)
            except OSError:
                continue
        if not timestamps:
            return None
        return dt.datetime.fromtimestamp(max(timestamps), tz=dt.timezone.utc)

    def _graph_log_progress(self, graph_id: str) -> Dict[str, Any]:
        log_path = self._graph_log_dir(graph_id) / "indexing-engine.log"
        if not log_path.exists():
            return {}
        try:
            excerpt = log_path.read_text(encoding="utf-8", errors="ignore")[-64000:]
        except Exception:
            return {}
        latest_workflow = ""
        for match in _WORKFLOW_STARTED_RE.finditer(excerpt):
            latest_workflow = str(match.group("workflow") or "").strip()
        latest_progress: Dict[str, Any] = {}
        for match in _PROGRESS_RE.finditer(excerpt):
            label = " ".join(str(match.group("label") or "").strip().split()).lower()
            current = int(match.group("current") or 0)
            total = int(match.group("total") or 0)
            latest_progress = {
                "label": label,
                "current": current,
                "total": total,
                "percent": round((current / total) * 100.0, 2) if total > 0 else 0.0,
            }
        payload: Dict[str, Any] = {}
        if latest_workflow:
            payload["workflow"] = latest_workflow
        if latest_progress:
            payload["progress"] = latest_progress
        activity = self._latest_graph_log_activity_at(graph_id)
        if activity is not None:
            payload["last_log_activity_at"] = activity.isoformat()
        return payload

    def _read_background_state(self, path: str) -> Dict[str, Any]:
        clean_path = str(path or "").strip()
        if not clean_path:
            return {}
        state_path = Path(clean_path)
        if not state_path.exists():
            return {}
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        payload.setdefault("state_path", str(state_path))
        return payload

    def _background_state_updates(self, state: Dict[str, Any]) -> Dict[str, Any]:
        updates: Dict[str, Any] = {}
        for key in (
            "runner_pid",
            "child_pid",
            "last_heartbeat_at",
            "last_output_at",
            "updated_at",
            "returncode",
            "failure_mode",
            "runner_log_path",
            "stream_log_path",
            "state_path",
            "timeout_seconds",
            "timed_out",
            "build_phase",
            "fallback_used",
        ):
            value = state.get(key)
            if value in (None, "", []):
                continue
            updates[key] = value
        if state.get("repair_summary"):
            updates["repair_summary"] = dict(state.get("repair_summary") or {})
        if state.get("fallback_summary"):
            updates["fallback_summary"] = dict(state.get("fallback_summary") or {})
        return updates

    def _pid_is_alive(self, pid: int) -> bool | None:
        if pid <= 0:
            return None
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return None
        return True

    def _run_process_active(self, run: GraphIndexRunRecord, *, root_path: Path) -> bool | None:
        metadata = dict(run.metadata or {})
        pid_candidates = [
            int(metadata.get("child_pid") or 0),
            int(metadata.get("active_pid") or 0),
            int(metadata.get("runner_pid") or 0),
        ]
        pid_states = [self._pid_is_alive(pid) for pid in pid_candidates if pid > 0]
        pid_active = None
        if any(state is True for state in pid_states):
            pid_active = True
        elif pid_states and all(state is False for state in pid_states):
            pid_active = False
        scan_active = self._is_graphrag_process_active(run.graph_id, root_path=root_path)
        if pid_active is True or scan_active is True:
            return True
        if pid_active is False and scan_active is False:
            return False
        return pid_active if pid_active is not None else scan_active

    def _terminate_owned_run_process(self, run: GraphIndexRunRecord) -> None:
        metadata = dict(run.metadata or {})
        pgid = int(metadata.get("active_process_group_id") or 0)
        pid = int(metadata.get("active_pid") or 0)
        if pgid > 0:
            try:
                os.killpg(pgid, signal.SIGTERM)
                return
            except OSError:
                pass
        if pid > 0:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass

    def _materialized_sources_from_health(self, record: GraphIndexRecord, resolved_docs: Sequence[Any]) -> List[Dict[str, Any]]:
        health = dict(record.health or {})
        paths = [str(item) for item in (health.get("materialized_source_paths") or []) if str(item).strip()]
        materialized: List[Dict[str, Any]] = []
        for index, item in enumerate(resolved_docs):
            payload = {
                "doc_id": str(getattr(item, "doc_id", "") or ""),
                "title": str(getattr(item, "title", "") or ""),
                "source_path": str(getattr(item, "source_path", "") or ""),
                "source_display_path": str(getattr(item, "source_display_path", "") or ""),
                "source_type": str(getattr(item, "source_type", "") or ""),
                "collection_id": str(getattr(item, "collection_id", "") or ""),
                "source_metadata": dict(getattr(item, "source_metadata", {}) or {}),
                "materialized_path": paths[index] if index < len(paths) else "",
                "materialized_filename": Path(paths[index]).name if index < len(paths) and paths[index] else "",
            }
            materialized.append(payload)
        return materialized

    def _validation_warnings(
        self,
        runtime_validation: Dict[str, Any],
        connectivity: Dict[str, Any],
        extraction_preflight: Dict[str, Any] | None = None,
        community_report_preflight: Dict[str, Any] | None = None,
    ) -> List[str]:
        preflight = dict(extraction_preflight or {})
        report_preflight = dict(community_report_preflight or {})
        return _dedupe(
            [
                *[str(item) for item in (runtime_validation.get("issues") or []) if str(item).strip()],
                *[str(item) for item in (runtime_validation.get("warnings") or []) if str(item).strip()],
                str(connectivity.get("detail") or "").strip()
                if str(connectivity.get("status") or "").strip().lower() in {"warning", "error"}
                else "",
                str(preflight.get("detail") or "").strip()
                if str(preflight.get("status") or "").strip().lower() in {"warning", "error"}
                else "",
                str(report_preflight.get("detail") or "").strip()
                if str(report_preflight.get("status") or "").strip().lower() in {"warning", "error"}
                else "",
            ]
        )

    def _is_graphrag_process_active(self, graph_id: str, *, root_path: Path) -> bool | None:
        try:
            result = subprocess.run(
                ["ps", "-axo", "command="],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        graph_token = str(graph_id or "").strip().lower()
        root_token = str(root_path.resolve()).lower()
        for line in result.stdout.splitlines():
            command = str(line or "").strip()
            if not command:
                continue
            lowered = command.lower()
            if ("graphrag" not in lowered and "graph-refresh" not in lowered) or "ps -axo" in lowered:
                continue
            if root_token in lowered or graph_token in lowered:
                return True
        return False

    def _mark_run_stale(
        self,
        record: GraphIndexRecord,
        run: GraphIndexRunRecord,
        *,
        inactivity_seconds: int,
        process_check: str,
    ) -> GraphIndexRunRecord:
        detected_at = self._now_utc()
        detail = (
            f"GraphRAG {str(run.operation or 'build').strip() or 'build'} run appears stalled: "
            f"no active process was found and graph logs have been idle for {inactivity_seconds} seconds."
        )
        stale_run = replace(
            run,
            status="failed",
            detail=detail,
            completed_at=detected_at.isoformat(),
            metadata={
                **dict(run.metadata or {}),
                "stale_run_recovered": True,
                "stale_run_detected_at": detected_at.isoformat(),
                "stale_run_inactivity_seconds": inactivity_seconds,
                "stale_run_process_check": process_check,
                "failure_mode": "stale_no_output",
            },
        )
        run_store = self._run_store()
        if run_store is not None:
            run_store.upsert_run(stale_run)

        health = dict(record.health or {})
        warnings = [str(item) for item in (health.get("warnings") or []) if str(item).strip()]
        health["warnings"] = _dedupe([*warnings, "GRAPHRAG_RUN_STALE"])
        health["latest_run"] = asdict(stale_run)
        health["status_detail"] = detail
        health["status_reason"] = "A previously reported GraphRAG run stalled and was automatically marked failed."
        if process_check:
            health["stale_run_process_check"] = process_check
        updated_record = replace(record, status="failed", health=health)
        if self._index_store() is not None:
            self._index_store().upsert_index(updated_record)
        return stale_run

    def _prioritize_runs_for_display(
        self,
        runs: Sequence[GraphIndexRunRecord],
    ) -> List[GraphIndexRunRecord]:
        graph_runs = list(runs or [])
        if not graph_runs:
            return []

        def _is_active(run: GraphIndexRunRecord) -> bool:
            return str(run.status or "").strip().lower() in {"queued", "running"}

        def _is_build_like(run: GraphIndexRunRecord) -> bool:
            return str(run.operation or "").strip().lower() in {"build", "refresh"}

        prioritized: List[GraphIndexRunRecord] = []
        seen: set[str] = set()
        for candidate in graph_runs:
            if _is_active(candidate) and _is_build_like(candidate):
                prioritized.append(candidate)
                seen.add(candidate.run_id)
        for candidate in graph_runs:
            if candidate.run_id in seen:
                continue
            if _is_active(candidate):
                prioritized.append(candidate)
                seen.add(candidate.run_id)
        for candidate in graph_runs:
            if candidate.run_id in seen:
                continue
            prioritized.append(candidate)
        return prioritized

    def _replace_run_in_sequence(
        self,
        runs: Sequence[GraphIndexRunRecord],
        updated: GraphIndexRunRecord,
    ) -> List[GraphIndexRunRecord]:
        replaced = [
            updated if str(item.run_id or "") == str(updated.run_id or "") else item
            for item in list(runs or [])
        ]
        return self._prioritize_runs_for_display(replaced)

    def _reconcile_live_run_state(
        self,
        record: GraphIndexRecord,
        *,
        runs: Sequence[GraphIndexRunRecord] | None = None,
    ) -> tuple[GraphIndexRecord, List[GraphIndexRunRecord]]:
        graph_runs = self._prioritize_runs_for_display(runs or self._graph_runs(record.graph_id, limit=6))
        latest = graph_runs[0] if graph_runs else None
        if latest is None:
            return record, graph_runs

        health = dict(record.health or {})
        health["latest_run"] = asdict(latest)
        if str(latest.status or "").strip().lower() not in {"queued", "running"}:
            return replace(record, health=health), graph_runs

        root_path = Path(record.root_path or self._graph_root(record.graph_id))
        progress_state = self._graph_log_progress(record.graph_id)
        run_started_at = self._parse_iso_datetime(latest.started_at)
        progress_activity_at = self._parse_iso_datetime(progress_state.get("last_log_activity_at"))
        if (
            run_started_at is not None
            and progress_activity_at is not None
            and progress_activity_at < run_started_at
        ):
            progress_state = {}
        progress_updates: Dict[str, Any] = {}
        if progress_state.get("last_log_activity_at"):
            progress_updates["last_log_activity_at"] = str(progress_state.get("last_log_activity_at") or "")
        if progress_state.get("progress"):
            progress_updates["progress"] = dict(progress_state.get("progress") or {})
        if progress_state.get("workflow"):
            progress_updates["workflow"] = str(progress_state.get("workflow") or "")
        if progress_updates:
            next_metadata = {**dict(latest.metadata or {}), **progress_updates}
            if next_metadata != dict(latest.metadata or {}):
                latest = replace(latest, metadata=next_metadata)
                run_store = self._run_store()
                if run_store is not None:
                    run_store.upsert_run(latest)
                graph_runs = self._replace_run_in_sequence(graph_runs, latest)

        state = self._read_background_state(str(dict(latest.metadata or {}).get("state_path") or ""))
        if state:
            state_updates = self._background_state_updates(state)
            if state_updates:
                next_metadata = {**dict(latest.metadata or {}), **state_updates}
                if next_metadata != dict(latest.metadata or {}):
                    latest = replace(latest, metadata=next_metadata)
                    run_store = self._run_store()
                    if run_store is not None:
                        run_store.upsert_run(latest)
                    graph_runs = self._replace_run_in_sequence(graph_runs, latest)
        if state and str(state.get("status") or "").strip().lower() in {"completed", "failed"}:
            self._complete_background_graph_run(record, latest, state=state)
            refreshed = self._get_index_record(self._index_store(), record.graph_id) if self._index_store() is not None else None
            refreshed_runs = self._graph_runs(record.graph_id, limit=6)
            effective = refreshed if refreshed is not None else record
            return self._reconcile_live_run_state(effective, runs=refreshed_runs)

        log_activity = self._latest_graph_log_activity_at(record.graph_id)
        latest_activity = max(
            [
                timestamp
                for timestamp in [
                    self._parse_iso_datetime(latest.started_at),
                    self._parse_iso_datetime(latest.completed_at),
                    self._parse_iso_datetime(dict(latest.metadata or {}).get("last_log_activity_at")),
                    self._parse_iso_datetime(dict(latest.metadata or {}).get("last_output_at")),
                    self._parse_iso_datetime(dict(latest.metadata or {}).get("last_heartbeat_at")),
                    self._parse_iso_datetime(dict(latest.metadata or {}).get("updated_at")),
                    log_activity,
                ]
                if timestamp is not None
            ],
            default=None,
        )
        process_active = self._run_process_active(latest, root_path=root_path)
        inactivity_seconds = None
        if latest_activity is not None:
            inactivity_seconds = int(max(0.0, (self._now_utc() - latest_activity).total_seconds()))
        if process_active is False and inactivity_seconds is not None and inactivity_seconds >= self._graphrag_stale_run_after_seconds():
            runner_pid = int(dict(latest.metadata or {}).get("runner_pid") or dict(latest.metadata or {}).get("active_pid") or 0)
            child_pid = int(dict(latest.metadata or {}).get("child_pid") or 0)
            process_check = (
                f"runner_pid={runner_pid or 'unknown'}, child_pid={child_pid or 'unknown'}; "
                f"ps scan found no active GraphRAG process for {record.graph_id}."
            )
            latest = self._mark_run_stale(
                record,
                latest,
                inactivity_seconds=inactivity_seconds,
                process_check=process_check,
            )
            graph_runs = self._replace_run_in_sequence(graph_runs, latest)
            refreshed = self._get_index_record(self._index_store(), record.graph_id) if self._index_store() is not None else None
            effective = refreshed if refreshed is not None else replace(record, status="failed")
            return self._reconcile_live_run_state(effective, runs=graph_runs)

        prior_warnings = [str(item) for item in (health.get("warnings") or []) if str(item).strip()]
        if prior_warnings:
            health["previous_warnings"] = prior_warnings
        health["warnings"] = []
        health["active_run"] = asdict(latest)
        health["active_operation"] = str(latest.operation or "")
        health["status_detail"] = str(latest.detail or "")
        health["status_reason"] = (
            "A managed graph build is currently in progress; prior failure warnings may be stale until this run finishes."
        )
        if progress_state.get("workflow"):
            health["active_workflow"] = str(progress_state.get("workflow") or "")
        if progress_state.get("progress"):
            health["progress"] = dict(progress_state.get("progress") or {})
        if dict(latest.metadata or {}).get("last_output_at"):
            health["last_output_at"] = str(dict(latest.metadata or {}).get("last_output_at") or "")
        if dict(latest.metadata or {}).get("last_heartbeat_at"):
            health["last_heartbeat_at"] = str(dict(latest.metadata or {}).get("last_heartbeat_at") or "")
        if dict(latest.metadata or {}).get("build_phase"):
            health["build_phase"] = str(dict(latest.metadata or {}).get("build_phase") or "")
        if "fallback_used" in dict(latest.metadata or {}):
            health["fallback_used"] = bool(dict(latest.metadata or {}).get("fallback_used", False))
        if dict(latest.metadata or {}).get("repair_summary"):
            health["repair_summary"] = dict(dict(latest.metadata or {}).get("repair_summary") or {})
        if dict(latest.metadata or {}).get("failure_mode"):
            health["failure_mode"] = str(dict(latest.metadata or {}).get("failure_mode") or "")
        if process_active is False and inactivity_seconds is not None:
            health["last_activity_seconds_ago"] = inactivity_seconds
        return replace(record, status=str(latest.status or "running"), health=health), graph_runs

    def _record_with_live_run_state(
        self,
        record: GraphIndexRecord,
        *,
        runs: Sequence[GraphIndexRunRecord] | None = None,
    ) -> GraphIndexRecord:
        effective, _ = self._reconcile_live_run_state(record, runs=runs)
        return effective

    def _graph_output_payload(
        self,
        record: GraphIndexRecord,
        *,
        runs: Sequence[GraphIndexRunRecord] | None = None,
    ) -> Dict[str, Any]:
        effective, effective_runs = self._reconcile_live_run_state(record, runs=runs)
        payload = asdict(effective)
        latest = effective_runs[:1]
        if latest:
            payload["latest_run"] = asdict(latest[0])
            metadata = dict(latest[0].metadata or {})
            if metadata.get("progress"):
                payload["progress"] = dict(metadata.get("progress") or {})
            if metadata.get("last_log_activity_at"):
                payload["last_log_activity_at"] = str(metadata.get("last_log_activity_at") or "")
            if metadata.get("last_output_at"):
                payload["last_output_at"] = str(metadata.get("last_output_at") or "")
            if metadata.get("last_heartbeat_at"):
                payload["last_heartbeat_at"] = str(metadata.get("last_heartbeat_at") or "")
            if metadata.get("active_pid"):
                payload["active_pid"] = int(metadata.get("active_pid") or 0)
            if metadata.get("child_pid"):
                payload["child_pid"] = int(metadata.get("child_pid") or 0)
            if metadata.get("runner_pid"):
                payload["runner_pid"] = int(metadata.get("runner_pid") or 0)
            if metadata.get("run_mode"):
                payload["run_mode"] = str(metadata.get("run_mode") or "")
            if metadata.get("failure_mode"):
                payload["failure_mode"] = str(metadata.get("failure_mode") or "")
            if metadata.get("build_phase"):
                payload["build_phase"] = str(metadata.get("build_phase") or "")
            if "fallback_used" in metadata:
                payload["fallback_used"] = bool(metadata.get("fallback_used", False))
            if metadata.get("repair_summary"):
                payload["repair_summary"] = dict(metadata.get("repair_summary") or {})
            if str(latest[0].status or "").strip().lower() in {"queued", "running"}:
                payload["active_run"] = asdict(latest[0])
                payload["status_detail"] = str(latest[0].detail or "")
            else:
                payload["status_detail"] = str(latest[0].detail or "")
        return payload

    def _resolve_graph_reference(self, graph_ref: str) -> GraphIndexRecord | None:
        store = self._index_store()
        reference = str(graph_ref or "").strip()
        if store is None or not reference:
            return None
        direct = self._get_index_record(store, reference)
        if direct is not None:
            return direct
        for record in self._list_index_records(store, limit=250):
            if str(record.display_name or "").strip().casefold() == reference.casefold():
                return record
        return None

    def _admin_error(self) -> Dict[str, Any]:
        return {"error": "Graph creation and refresh are admin-managed in the control panel."}

    def _sanitize_prompt_overrides(self, overrides: Dict[str, Any] | None) -> Dict[str, str]:
        payload: Dict[str, str] = {}
        for key, value in dict(overrides or {}).items():
            filename = str(key or "").strip()
            if not filename:
                continue
            payload[filename] = str(value or "")
        return payload

    def _sanitize_graph_skills(self, skill_ids: Sequence[str] | None) -> List[str]:
        return _dedupe(str(item) for item in (skill_ids or []) if str(item).strip())

    def _session_metadata(self) -> Dict[str, Any]:
        if self.session is None:
            return {}
        metadata = getattr(self.session, "metadata", None)
        if isinstance(metadata, dict):
            return metadata
        metadata = {}
        try:
            setattr(self.session, "metadata", metadata)
        except Exception:
            return {}
        return metadata

    def _remember_active_graphs(self, graph_ids: Sequence[str] | None) -> List[str]:
        active = _dedupe(str(item) for item in (graph_ids or []) if str(item).strip())
        if not active:
            return []
        metadata = self._session_metadata()
        if not metadata and self.session is None:
            return active
        prior = [str(item) for item in (metadata.get("active_graph_ids") or []) if str(item).strip()]
        merged = _dedupe([*active, *prior])[:8]
        metadata["active_graph_ids"] = merged
        return merged

    def _discovered_graph_skill_ids(self, graph_id: str) -> List[str]:
        skill_store = self._skill_store()
        if skill_store is None or not str(graph_id or "").strip():
            return []
        try:
            records = skill_store.list_skill_packs(
                tenant_id=self.tenant_id,
                owner_user_id=self.user_id,
                graph_id=str(graph_id).strip(),
            )
        except TypeError:
            records = []
        except Exception:
            records = []
        return _dedupe(
            str(getattr(record, "skill_id", "") or "")
            for record in records
            if str(getattr(record, "skill_id", "") or "").strip()
        )

    def _bound_graph_skill_ids(self, graph_id: str, graph_skill_ids: Sequence[str] | None = None) -> List[str]:
        return _dedupe(
            [
                *self._sanitize_graph_skills(graph_skill_ids),
                *self._discovered_graph_skill_ids(graph_id),
            ]
        )

    def _graph_visibility(self, value: str) -> str:
        normalized = str(value or "tenant").strip().lower()
        return normalized if normalized in {"tenant", "private", "global"} else "tenant"

    def _graph_record_payload(
        self,
        *,
        graph_id: str,
        display_name: str,
        collection_id: str,
        backend: str,
        status: str,
        root_path: Path,
        artifact_path: str,
        summary: Dict[str, Any],
        source_doc_ids: Sequence[str],
        owner_admin_user_id: str,
        visibility: str,
        config_json: Dict[str, Any] | None = None,
        prompt_overrides_json: Dict[str, Any] | None = None,
        graph_skill_ids: Sequence[str] | None = None,
        capabilities: Sequence[str] | None = None,
        supported_query_methods: Sequence[str] | None = None,
        query_ready: bool = False,
        query_backend: str = "",
        artifact_tables: Sequence[str] | None = None,
        artifact_mtime: str = "",
        graph_context_summary: Dict[str, Any] | None = None,
        health: Dict[str, Any] | None = None,
        freshness_score: float = 0.0,
    ) -> GraphIndexRecord:
        return GraphIndexRecord(
            graph_id=graph_id,
            tenant_id=self.tenant_id,
            collection_id=collection_id,
            display_name=display_name,
            owner_admin_user_id=owner_admin_user_id,
            visibility=self._graph_visibility(visibility),
            backend=backend,
            status=status,
            root_path=str(root_path),
            artifact_path=artifact_path or str(root_path),
            domain_summary=str(summary.get("domain_summary") or ""),
            entity_samples=[str(item) for item in (summary.get("entity_samples") or []) if str(item)],
            relationship_samples=[str(item) for item in (summary.get("relationship_samples") or []) if str(item)],
            source_doc_ids=[str(item) for item in source_doc_ids if str(item)],
            capabilities=[str(item) for item in (capabilities or []) if str(item)],
            supported_query_methods=[str(item) for item in (supported_query_methods or []) if str(item)],
            query_ready=bool(query_ready),
            query_backend=str(query_backend or ""),
            artifact_tables=[str(item) for item in (artifact_tables or []) if str(item)],
            artifact_mtime=str(artifact_mtime or ""),
            graph_context_summary=dict(graph_context_summary or {}),
            config_json=dict(config_json or {}),
            prompt_overrides_json=dict(prompt_overrides_json or {}),
            graph_skill_ids=self._bound_graph_skill_ids(graph_id, graph_skill_ids),
            health=dict(health or {}),
            freshness_score=float(freshness_score or 0.0),
        )

    def _reconstruct_document_text(self, record: Any) -> str:
        chunk_store = self._chunk_store()
        doc_id = str(getattr(record, "doc_id", "") or "")
        title = str(getattr(record, "title", "") or doc_id or "Document")
        if chunk_store is not None and doc_id:
            try:
                chunks = chunk_store.list_document_chunks(doc_id, tenant_id=self.tenant_id)
            except Exception:
                chunks = []
            if chunks:
                body = "\n\n".join(str(getattr(chunk, "content", "") or "").strip() for chunk in chunks if str(getattr(chunk, "content", "") or "").strip())
                if body.strip():
                    return f"# {title}\n\n{body.strip()}\n"
        source_path = Path(str(getattr(record, "source_path", "") or "")).expanduser()
        if not source_path.exists():
            blob_ref = blob_ref_from_record(record)
            if blob_ref is not None:
                try:
                    source_path = build_blob_store(self.settings).materialize_to_path(blob_ref)
                except Exception:
                    source_path = Path("__missing_remote_source__")
        if source_path.exists():
            try:
                raw = source_path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                raw = ""
            if raw:
                return raw + "\n"
        return f"# {title}\n\nNo extracted text is currently available for this source.\n"

    def _materialize_source_documents(self, graph_id: str, records: Sequence[Any]) -> List[Dict[str, Any]]:
        input_dir = self._graph_input_dir(graph_id)
        if input_dir.exists():
            shutil.rmtree(input_dir, ignore_errors=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        materialized: List[Dict[str, Any]] = []
        for index, record in enumerate(records, start=1):
            doc_id = str(getattr(record, "doc_id", "") or f"doc_{index}")
            title = str(getattr(record, "title", "") or doc_id)
            stem = _slugify(Path(str(getattr(record, "source_path", "") or title)).stem or title)
            filename = f"{index:03d}_{stem}.txt"
            content = self._reconstruct_document_text(record)
            destination = input_dir / filename
            destination.write_text(content, encoding="utf-8")
            materialized.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "source_path": str(getattr(record, "source_path", "") or ""),
                    "source_display_path": str(getattr(record, "source_display_path", "") or ""),
                    "source_type": str(getattr(record, "source_type", "") or ""),
                    "collection_id": str(getattr(record, "collection_id", "") or ""),
                    "source_metadata": dict(getattr(record, "source_metadata", {}) or {}),
                    "materialized_path": str(destination),
                    "materialized_filename": filename,
                }
            )
        return materialized

    def _load_project_settings(self, graph_id: str) -> Dict[str, Any]:
        settings_path = self._graph_settings_path(graph_id)
        if not settings_path.exists():
            return {}
        try:
            payload = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _graphrag_chunking_defaults(self) -> Dict[str, int]:
        return {"size": 800, "overlap": 80}

    def _graphrag_vector_size(self, profile: Dict[str, Any]) -> int:
        embed_model = str(profile.get("embed_model") or "").strip().lower()
        if "nomic-embed-text" in embed_model:
            return 768
        return max(1, int(getattr(self.settings, "embedding_dim", 768) or 768))

    def _graphrag_profile(self) -> Dict[str, Any]:
        chunking = self._graphrag_chunking_defaults()
        community_report_mode = str(
            getattr(self.settings, "graphrag_community_report_mode", "text") or "text"
        ).strip().lower()
        if community_report_mode not in {"text", "graph"}:
            community_report_mode = "text"
        chat_model = str(getattr(self.settings, "graphrag_chat_model", "") or "").strip()
        index_chat_model = str(
            getattr(self.settings, "graphrag_index_chat_model", chat_model)
            or chat_model
            or ""
        ).strip()
        return {
            "model_provider": str(getattr(self.settings, "graphrag_llm_provider", "") or "").strip() or "openai",
            "api_base": str(getattr(self.settings, "graphrag_base_url", "") or "").strip(),
            "api_key": str(getattr(self.settings, "graphrag_api_key", "") or "").strip(),
            "chat_model": chat_model,
            "index_chat_model": index_chat_model,
            "community_report_mode": community_report_mode,
            "community_report_chat_model": str(
                getattr(self.settings, "graphrag_community_report_chat_model", index_chat_model)
                or index_chat_model
                or chat_model
                or ""
            ).strip(),
            "embed_model": str(getattr(self.settings, "graphrag_embed_model", "") or "").strip(),
            "concurrency": max(1, int(getattr(self.settings, "graphrag_concurrency", 4) or 4)),
            "chunk_size": int(chunking["size"]),
            "chunk_overlap": int(chunking["overlap"]),
            "timeout_seconds": max(
                30,
                int(
                    getattr(
                        self.settings,
                        "graphrag_request_timeout_seconds",
                        getattr(self.settings, "graphrag_timeout_seconds", 180),
                    )
                    or getattr(self.settings, "graphrag_timeout_seconds", 180)
                    or 180
                ),
            ),
            "index_timeout_seconds": max(
                30,
                int(
                    getattr(
                        self.settings,
                        "graphrag_index_request_timeout_seconds",
                        getattr(
                            self.settings,
                            "graphrag_request_timeout_seconds",
                            getattr(self.settings, "graphrag_timeout_seconds", 180),
                        ),
                    )
                    or getattr(
                        self.settings,
                        "graphrag_request_timeout_seconds",
                        getattr(self.settings, "graphrag_timeout_seconds", 180),
                    )
                    or 180
                ),
            ),
            "community_report_timeout_seconds": max(
                30,
                int(
                    getattr(
                        self.settings,
                        "graphrag_community_report_request_timeout_seconds",
                        getattr(
                            self.settings,
                            "graphrag_index_request_timeout_seconds",
                            getattr(
                                self.settings,
                                "graphrag_request_timeout_seconds",
                                getattr(self.settings, "graphrag_timeout_seconds", 180),
                            ),
                        ),
                    )
                    or getattr(
                        self.settings,
                        "graphrag_index_request_timeout_seconds",
                        getattr(
                            self.settings,
                            "graphrag_request_timeout_seconds",
                            getattr(self.settings, "graphrag_timeout_seconds", 180),
                        ),
                    )
                    or 180
                ),
            ),
            "community_report_max_input_length": max(
                500,
                int(getattr(self.settings, "graphrag_community_report_max_input_length", 4000) or 4000),
            ),
            "community_report_max_length": max(
                200,
                int(getattr(self.settings, "graphrag_community_report_max_length", 1200) or 1200),
            ),
        }

    def _is_ollama_profile(self, profile: Dict[str, Any]) -> bool:
        api_base = str(profile.get("api_base") or "").strip().lower()
        return "11434" in api_base or "ollama" in api_base

    def _completion_call_args(
        self,
        profile: Dict[str, Any],
        *,
        model_key: str = "chat_model",
        timeout_key: str = "timeout_seconds",
    ) -> Dict[str, Any]:
        call_args: Dict[str, Any] = {
            "temperature": 0,
            "timeout": int(profile.get(timeout_key) or 180),
        }
        chat_model = str(profile.get(model_key) or "").strip().lower()
        if self._is_ollama_profile(profile) and "gpt-oss" in chat_model:
            # GPT-OSS on Ollama emits a thinking trace by default; keep it on the
            # shortest setting so GraphRAG extraction stays predictable and tractable.
            call_args["reasoning_effort"] = "low"
            call_args["reasoning"] = {"effort": "low"}
        return call_args

    def _embedding_call_args(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        return {"timeout": int(profile.get("timeout_seconds") or 180)}

    def _extract_graph_prompt_template(self, record: GraphIndexRecord) -> str:
        override = str(dict(record.prompt_overrides_json or {}).get("extract_graph.txt") or "").strip()
        if override:
            return override
        try:
            from graphrag.prompts.index.extract_graph import GRAPH_EXTRACTION_PROMPT

            template = str(GRAPH_EXTRACTION_PROMPT or "").strip()
            if template:
                return template
        except Exception:
            pass
        return _DEFAULT_EXTRACT_GRAPH_PREFLIGHT_PROMPT

    def _extract_graph_entity_types(self, record: GraphIndexRecord) -> List[str]:
        extract_graph_config = dict(dict(record.config_json or {}).get("extract_graph") or {})
        entity_types = [str(item).strip() for item in (extract_graph_config.get("entity_types") or []) if str(item).strip()]
        return entity_types or ["organization", "person", "geo", "event"]

    def _sample_extraction_probe_text(self, resolved_docs: Sequence[Any]) -> Dict[str, Any]:
        for record in resolved_docs:
            raw_text = self._reconstruct_document_text(record)
            normalized = re.sub(r"\s+", " ", str(raw_text or "")).strip()
            if not normalized:
                continue
            excerpt = normalized[:1400].strip()
            if len(excerpt) < 80:
                continue
            return {
                "doc_id": str(getattr(record, "doc_id", "") or ""),
                "title": str(getattr(record, "title", "") or ""),
                "text": excerpt,
            }
        return {"doc_id": "", "title": "", "text": ""}

    def _validate_extract_graph_preflight(
        self,
        *,
        record: GraphIndexRecord,
        profile: Dict[str, Any],
        resolved_docs: Sequence[Any],
    ) -> Dict[str, Any]:
        base_url = str(profile.get("api_base") or "").strip().rstrip("/")
        if not base_url:
            return {
                "ok": True,
                "status": "skipped",
                "detail": "No GraphRAG base URL configured; skipping extraction preflight.",
            }
        if not str(profile.get("chat_model") or "").strip():
            return {
                "ok": False,
                "status": "error",
                "detail": "GRAPHRAG_CHAT_MODEL is required for extraction preflight.",
            }
        index_chat_model = str(profile.get("index_chat_model") or profile.get("chat_model") or "").strip()
        if not index_chat_model:
            return {
                "ok": False,
                "status": "error",
                "detail": "A GraphRAG indexing chat model is required for extraction preflight.",
            }

        sample = self._sample_extraction_probe_text(resolved_docs)
        sample_text = str(sample.get("text") or "").strip()
        if not sample_text:
            return {
                "ok": True,
                "status": "skipped",
                "detail": "No sample chunk was available for extraction preflight.",
            }

        prompt_template = self._extract_graph_prompt_template(record)
        if not prompt_template:
            return {
                "ok": False,
                "status": "error",
                "detail": "The extract_graph prompt template could not be loaded for preflight validation.",
            }

        entity_types = self._extract_graph_entity_types(record)
        try:
            prompt_text = prompt_template.format(
                input_text=sample_text,
                entity_types=",".join(entity_types),
            )
        except Exception as exc:
            return {
                "ok": False,
                "status": "error",
                "detail": f"Could not materialize the extract_graph prompt template: {exc}",
            }

        headers: Dict[str, str] = {}
        api_key = str(profile.get("api_key") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{base_url}/chat/completions",
                headers=headers,
                timeout=max(30, int(profile.get("index_timeout_seconds") or profile.get("timeout_seconds") or 30)),
                json={
                    "model": index_chat_model,
                    "messages": [{"role": "user", "content": prompt_text}],
                    **self._completion_call_args(
                        profile,
                        model_key="index_chat_model",
                        timeout_key="index_timeout_seconds",
                    ),
                },
            )
        except Exception as exc:
            return {
                "ok": False,
                "status": "error",
                "detail": f"GraphRAG extraction preflight failed to reach the chat completion endpoint: {exc}",
                "sample": {"doc_id": sample["doc_id"], "title": sample["title"]},
            }
        latency_seconds = round(time.perf_counter() - started, 3)

        if response.status_code >= 400:
            return {
                "ok": False,
                "status": "error",
                "detail": f"GraphRAG extraction preflight returned HTTP {response.status_code}.",
                "latency_seconds": latency_seconds,
                "sample": {"doc_id": sample["doc_id"], "title": sample["title"]},
            }

        try:
            payload = response.json()
        except Exception as exc:
            return {
                "ok": False,
                "status": "error",
                "detail": f"GraphRAG extraction preflight returned a non-JSON response envelope: {exc}",
                "latency_seconds": latency_seconds,
                "sample": {"doc_id": sample["doc_id"], "title": sample["title"]},
            }

        content = str(
            (((payload.get("choices") or [{}])[0] or {}).get("message") or {}).get("content") or ""
        ).strip()
        if content == "<|COMPLETE|>":
            return {
                "ok": True,
                "status": "warning",
                "detail": (
                    "GraphRAG extraction preflight completed without tuple output for the sampled chunk. "
                    "The model responded in the expected protocol, but the sample may not contain extractable entities."
                ),
                "latency_seconds": latency_seconds,
                "timeout_seconds": max(30, int(profile.get("timeout_seconds") or 30)),
                "sample": {"doc_id": sample["doc_id"], "title": sample["title"]},
                "entity_types": entity_types,
                "response_preview": content,
            }
        usable = bool(content) and "<|>" in content and ("##" in content or "<|COMPLETE|>" in content)
        if not usable:
            return {
                "ok": False,
                "status": "error",
                "detail": "GraphRAG extraction preflight returned output that does not match the expected tuple-delimited extract_graph format.",
                "latency_seconds": latency_seconds,
                "sample": {"doc_id": sample["doc_id"], "title": sample["title"]},
                "response_preview": content[:500],
            }

        timeout_seconds = max(30, int(profile.get("timeout_seconds") or 30))
        status = "ready"
        detail = "GraphRAG extraction preflight succeeded."
        if latency_seconds >= round(timeout_seconds * 0.75, 3):
            status = "warning"
            detail = (
                "GraphRAG extraction preflight succeeded but is close to the configured request timeout. "
                "Expect slower local builds on Nemotron."
            )
        return {
            "ok": True,
            "status": status,
            "detail": detail,
            "latency_seconds": latency_seconds,
            "timeout_seconds": timeout_seconds,
            "sample": {"doc_id": sample["doc_id"], "title": sample["title"]},
            "entity_types": entity_types,
            "response_preview": content[:500],
        }

    def _validate_community_report_preflight(
        self,
        *,
        record: GraphIndexRecord,
        profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        del profile
        root_path = Path(record.root_path or self._graph_root(record.graph_id))
        result = analyze_community_report_inputs(root_path, dry_run=True)
        payload = {
            "ok": bool(result.get("ok", False)),
            "status": str(result.get("status") or "skipped"),
            "detail": str(result.get("detail") or ""),
            "native_phase2_safe": bool(result.get("native_phase2_safe", True)),
            "orphan_membership_count": int(result.get("orphan_membership_count") or 0),
            "remaining_orphan_membership_count": int(result.get("remaining_orphan_membership_count") or 0),
            "dropped_tuple_count": int(result.get("dropped_tuple_count") or 0),
            "affected_community_ids": [
                int(item) for item in (result.get("affected_community_ids") or []) if str(item).strip()
            ],
            "emptied_community_ids": [
                int(item) for item in (result.get("emptied_community_ids") or []) if str(item).strip()
            ],
            "output_dir": str(result.get("output_dir") or ""),
        }
        if payload["status"] == "skipped":
            payload["ok"] = True
        return payload

    def _write_project_env(self, graph_id: str, profile: Dict[str, Any]) -> str:
        env_path = self._graph_env_path(graph_id)
        api_key = str(profile.get("api_key") or "ollama").strip() or "ollama"
        env_path.write_text(f"GRAPHRAG_API_KEY={api_key}\n", encoding="utf-8")
        return str(env_path)

    def _reset_project_for_full_rebuild(self, root_path: Path) -> Dict[str, List[str]]:
        removed_dirs: List[str] = []
        for name in ("output", "cache", "input"):
            target = root_path / name
            if not target.exists():
                continue
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
            removed_dirs.append(str(target))
        return {"removed_dirs": removed_dirs}

    def _persist_running_graph_state(
        self,
        *,
        record: GraphIndexRecord,
        resolved_docs: Sequence[Any],
        source_records: Sequence[GraphIndexSourceRecord],
        backend_result: GraphOperationResult,
        runtime_validation: Dict[str, Any],
        connectivity: Dict[str, Any],
        init_result: GraphOperationResult,
        materialized_sources: Sequence[Dict[str, Any]],
        written_settings: str,
        written_prompts: Sequence[str],
        manifest_path: str,
        refresh_mode: str,
        cleared_state: Dict[str, Any],
        run_id: str,
        operation: str,
    ) -> Dict[str, Any]:
        graph_store = self._index_store()
        source_store = self._source_store()
        if graph_store is None or source_store is None:
            return {"error": "Graph catalog stores are unavailable."}
        summary = self._summarize_records(
            resolved_docs,
            display_name=record.display_name or record.graph_id,
            collection_id=record.collection_id,
        )
        validation_warnings = self._validation_warnings(runtime_validation, connectivity)
        retained_query_ready = bool(refresh_mode == "incremental_update" and record.query_ready)
        retained_health = {
            key: value
            for key, value in dict(record.health or {}).items()
            if key
            not in {
                "warnings",
                "latest_run",
                "active_run",
                "active_operation",
                "active_workflow",
                "progress",
                "last_log_activity_at",
                "last_activity_seconds_ago",
                "status_detail",
                "status_reason",
                "previous_warnings",
                "returncode",
                "stale_run_process_check",
            }
        }
        updated_record = self._graph_record_payload(
            graph_id=record.graph_id,
            display_name=record.display_name or record.graph_id,
            collection_id=record.collection_id,
            backend=record.backend,
            status="running",
            root_path=Path(record.root_path or self._graph_root(record.graph_id)),
            artifact_path=record.artifact_path or record.root_path or str(self._graph_root(record.graph_id)),
            summary=summary,
            source_doc_ids=[str(getattr(item, "doc_id", "") or "") for item in resolved_docs],
            owner_admin_user_id=record.owner_admin_user_id,
            visibility=record.visibility,
            config_json=dict(record.config_json or {}),
            prompt_overrides_json=dict(record.prompt_overrides_json or {}),
            graph_skill_ids=list(record.graph_skill_ids),
            capabilities=_dedupe([*record.capabilities, *backend_result.capabilities, "admin_managed", "graphrag_project"]),
            supported_query_methods=list(backend_result.supported_query_methods or record.supported_query_methods),
            query_ready=retained_query_ready,
            query_backend=record.query_backend if retained_query_ready else "",
            artifact_tables=list(record.artifact_tables if retained_query_ready else []),
            artifact_mtime=str(record.artifact_mtime if retained_query_ready else ""),
            graph_context_summary=dict(record.graph_context_summary if retained_query_ready else {}),
            health={
                **retained_health,
                "warnings": validation_warnings,
                "materialized_source_count": len(materialized_sources),
                "materialized_source_paths": [str(item.get("materialized_path") or "") for item in materialized_sources],
                "settings_path": written_settings,
                "prompt_files": list(written_prompts),
                "manifest_path": manifest_path,
                "init_status": init_result.status,
                "init_detail": init_result.detail,
                "refresh_mode": refresh_mode,
                "cleared_state": dict(cleared_state or {}),
                "runtime_validation": dict(runtime_validation or {}),
                "connectivity": dict(connectivity or {}),
                **dict(backend_result.metadata or {}),
            },
            freshness_score=float(record.freshness_score or 0.0),
        )
        updated_record.last_indexed_at = record.last_indexed_at
        graph_store.upsert_index(updated_record)
        source_store.replace_sources(record.graph_id, tenant_id=self.tenant_id, sources=list(source_records))
        self._run_result(
            graph_id=record.graph_id,
            operation=operation,
            status="running",
            detail=backend_result.detail or f"Starting graph {operation}.",
            metadata={
                "warnings": list(backend_result.warnings),
                "artifact_path": updated_record.artifact_path,
                "manifest_path": manifest_path,
                **dict(backend_result.metadata or {}),
            },
            run_id=run_id,
        )
        payload = self._admin_graph_payload(updated_record)
        payload.update(
            {
                "status": "running",
                "detail": backend_result.detail,
                "warnings": list(backend_result.warnings),
                "manifest_path": manifest_path,
                "settings_path": written_settings,
                "prompt_files": list(written_prompts),
                "materialized_sources": list(materialized_sources),
                "refresh_mode": refresh_mode,
                "cleared_state": dict(cleared_state or {}),
                "runtime_validation": dict(runtime_validation or {}),
                "connectivity": dict(connectivity or {}),
                "run_id": run_id,
            }
        )
        return payload

    def _persist_completed_graph_state(
        self,
        *,
        record: GraphIndexRecord,
        resolved_docs: Sequence[Any],
        source_records: Sequence[GraphIndexSourceRecord],
        backend_result: GraphOperationResult,
        run_id: str,
        operation: str,
    ) -> Dict[str, Any]:
        graph_store = self._index_store()
        source_store = self._source_store()
        if graph_store is None or source_store is None:
            return {"error": "Graph catalog stores are unavailable."}
        health = dict(record.health or {})
        runtime_validation = dict(health.get("runtime_validation") or {})
        connectivity = dict(health.get("connectivity") or {})
        materialized_sources = self._materialized_sources_from_health(record, resolved_docs)
        written_settings = str(health.get("settings_path") or self._graph_settings_path(record.graph_id))
        written_prompts = [str(item) for item in (health.get("prompt_files") or []) if str(item).strip()]
        manifest_path = str(health.get("manifest_path") or "")
        init_status = str(health.get("init_status") or "ready")
        init_detail = str(health.get("init_detail") or "Existing GraphRAG project reused.")
        refresh_mode = str(health.get("refresh_mode") or "full_rebuild")
        cleared_state = dict(health.get("cleared_state") or {"removed_dirs": []})
        summary = self._summarize_records(
            resolved_docs,
            display_name=record.display_name or record.graph_id,
            collection_id=record.collection_id,
        )
        validation_warnings = self._validation_warnings(runtime_validation, connectivity)
        updated_record = self._graph_record_payload(
            graph_id=record.graph_id,
            display_name=record.display_name or record.graph_id,
            collection_id=record.collection_id,
            backend=record.backend,
            status=backend_result.status,
            root_path=Path(record.root_path or self._graph_root(record.graph_id)),
            artifact_path=backend_result.artifact_path or record.artifact_path or str(self._graph_root(record.graph_id)),
            summary=summary,
            source_doc_ids=[str(getattr(item, "doc_id", "") or "") for item in resolved_docs],
            owner_admin_user_id=record.owner_admin_user_id,
            visibility=record.visibility,
            config_json=dict(record.config_json or {}),
            prompt_overrides_json=dict(record.prompt_overrides_json or {}),
            graph_skill_ids=list(record.graph_skill_ids),
            capabilities=_dedupe([*record.capabilities, *backend_result.capabilities, "admin_managed", "graphrag_project"]),
            supported_query_methods=list(backend_result.supported_query_methods or record.supported_query_methods),
            query_ready=bool(backend_result.query_ready),
            query_backend=str(backend_result.query_backend or record.query_backend),
            artifact_tables=list(backend_result.artifact_tables or record.artifact_tables),
            artifact_mtime=str(backend_result.artifact_mtime or ""),
            graph_context_summary=dict(backend_result.graph_context_summary or {}),
            health={
                **{key: value for key, value in health.items() if key != "warnings"},
                "warnings": _dedupe([*[str(item) for item in backend_result.warnings if str(item).strip()], *validation_warnings]),
                "materialized_source_count": len(materialized_sources),
                "materialized_source_paths": [str(item.get("materialized_path") or "") for item in materialized_sources],
                "settings_path": written_settings,
                "prompt_files": written_prompts,
                "manifest_path": manifest_path,
                "init_status": init_status,
                "init_detail": init_detail,
                "refresh_mode": refresh_mode,
                "cleared_state": cleared_state,
                "runtime_validation": runtime_validation,
                "connectivity": connectivity,
                **dict(backend_result.metadata or {}),
            },
            freshness_score=1.0 if backend_result.status == "ready" else float(record.freshness_score or 0.0),
        )
        if backend_result.status == "ready":
            updated_record.last_indexed_at = self._now_utc().isoformat()
        else:
            updated_record.last_indexed_at = record.last_indexed_at
        graph_store.upsert_index(updated_record)
        source_store.replace_sources(record.graph_id, tenant_id=self.tenant_id, sources=list(source_records))
        self._sync_entities_from_artifacts(graph_record=updated_record)
        self._run_result(
            graph_id=record.graph_id,
            operation=operation,
            status=backend_result.status,
            detail=backend_result.detail,
            metadata={
                "warnings": list(backend_result.warnings),
                "artifact_path": updated_record.artifact_path,
                "manifest_path": manifest_path,
                **dict(backend_result.metadata or {}),
            },
            run_id=run_id,
        )
        payload = self._admin_graph_payload(updated_record)
        payload.update(
            {
                "status": backend_result.status,
                "detail": backend_result.detail,
                "warnings": list(backend_result.warnings),
                "manifest_path": manifest_path,
                "settings_path": written_settings,
                "prompt_files": written_prompts,
                "materialized_sources": materialized_sources,
                "refresh_mode": refresh_mode,
                "cleared_state": cleared_state,
                "runtime_validation": runtime_validation,
                "connectivity": connectivity,
                "run_id": run_id,
            }
        )
        return payload

    def _complete_background_graph_run(
        self,
        record: GraphIndexRecord,
        run: GraphIndexRunRecord,
        *,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        backend = self._backend_for(record.backend)
        if not hasattr(backend, "collect_job_result"):
            return {}
        resolved_docs = self._resolved_records_for_graph(record)
        source_records = self._build_source_records(record.graph_id, resolved_docs)
        backend_result = backend.collect_job_result(record.graph_id, Path(record.root_path or self._graph_root(record.graph_id)), state=state)
        return self._persist_completed_graph_state(
            record=record,
            resolved_docs=resolved_docs,
            source_records=source_records,
            backend_result=backend_result,
            run_id=run.run_id,
            operation=str(run.operation or "build"),
        )

    def _render_settings_yaml(
        self,
        graph_id: str,
        *,
        config_overrides: Dict[str, Any] | None = None,
    ) -> str:
        profile = self._graphrag_profile()
        vector_size = self._graphrag_vector_size(profile)
        current = self._load_project_settings(graph_id)
        completion_model = {
            "model_provider": profile["model_provider"],
            "model": profile["chat_model"],
            "auth_method": "api_key",
            "api_key": "${GRAPHRAG_API_KEY}",
            "retry": {"type": "exponential_backoff"},
        }
        embedding_model = {
            "model_provider": profile["model_provider"],
            "model": profile["embed_model"],
            "auth_method": "api_key",
            "api_key": "${GRAPHRAG_API_KEY}",
            "retry": {"type": "exponential_backoff"},
        }
        if profile["api_base"]:
            completion_model["api_base"] = profile["api_base"]
            embedding_model["api_base"] = profile["api_base"]
        completion_model["call_args"] = self._completion_call_args(profile)
        index_completion_model = {
            **completion_model,
            "model": str(profile.get("index_chat_model") or profile["chat_model"]),
            "call_args": self._completion_call_args(
                profile,
                model_key="index_chat_model",
                timeout_key="index_timeout_seconds",
            ),
        }
        community_report_completion_model = {
            **completion_model,
            "model": str(
                profile.get("community_report_chat_model")
                or profile.get("index_chat_model")
                or profile["chat_model"]
            ),
            "call_args": self._completion_call_args(
                profile,
                model_key="community_report_chat_model",
                timeout_key="community_report_timeout_seconds",
            ),
        }
        embedding_model["call_args"] = self._embedding_call_args(profile)

        payload: Dict[str, Any] = {
            **current,
            "completion_models": {
                "default_completion_model": completion_model,
                "index_completion_model": index_completion_model,
                "community_report_completion_model": community_report_completion_model,
            },
            "embedding_models": {"default_embedding_model": embedding_model},
            "concurrent_requests": profile["concurrency"],
            "input": {"type": "text"},
            "chunking": {
                "type": "tokens",
                "size": int(profile.get("chunk_size") or 800),
                "overlap": int(profile.get("chunk_overlap") or 80),
                "encoding_model": "o200k_base",
            },
            "input_storage": {"type": "file", "base_dir": "input"},
            "output_storage": {"type": "file", "base_dir": "output"},
            "reporting": {"type": "file", "base_dir": "logs"},
            "cache": {"type": "json", "storage": {"type": "file", "base_dir": "cache"}},
            "vector_store": {"type": "lancedb", "db_uri": "output/lancedb", "vector_size": vector_size},
            "embed_text": {"embedding_model_id": "default_embedding_model"},
            "extract_graph": {
                "completion_model_id": "index_completion_model",
                "prompt": "prompts/extract_graph.txt",
                "entity_types": ["organization", "person", "geo", "event"],
                "max_gleanings": 1,
            },
            "summarize_descriptions": {
                "completion_model_id": "index_completion_model",
                "prompt": "prompts/summarize_descriptions.txt",
                "max_length": 500,
            },
            "community_reports": {
                "completion_model_id": "community_report_completion_model",
                "graph_prompt": "prompts/community_report_graph.txt",
                "text_prompt": "prompts/community_report_text.txt",
                "max_length": int(profile.get("community_report_max_length") or 1200),
                "max_input_length": int(profile.get("community_report_max_input_length") or 4000),
            },
            "local_search": {
                "completion_model_id": "default_completion_model",
                "embedding_model_id": "default_embedding_model",
                "prompt": "prompts/local_search_system_prompt.txt",
            },
            "global_search": {
                "completion_model_id": "default_completion_model",
                "map_prompt": "prompts/global_search_map_system_prompt.txt",
                "reduce_prompt": "prompts/global_search_reduce_system_prompt.txt",
                "knowledge_prompt": "prompts/global_search_knowledge_system_prompt.txt",
            },
            "drift_search": {
                "completion_model_id": "default_completion_model",
                "embedding_model_id": "default_embedding_model",
                "prompt": "prompts/drift_search_system_prompt.txt",
                "reduce_prompt": "prompts/drift_reduce_prompt.txt",
            },
            "basic_search": {
                "completion_model_id": "default_completion_model",
                "embedding_model_id": "default_embedding_model",
                "prompt": "prompts/basic_search_system_prompt.txt",
            },
        }
        if str(profile.get("community_report_mode") or "text").strip().lower() == "text":
            payload["workflows"] = [
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
        else:
            payload.pop("workflows", None)
        for key, value in dict(config_overrides or {}).items():
            if not str(key).strip():
                continue
            payload[str(key)] = value
        settings_path = self._graph_settings_path(graph_id)
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        self._write_project_env(graph_id, profile)
        return str(settings_path)

    def _write_prompt_overrides(self, graph_id: str, prompt_overrides: Dict[str, Any] | None) -> List[str]:
        prompt_dir = self._graph_prompts_dir(graph_id)
        prompt_dir.mkdir(parents=True, exist_ok=True)
        written: List[str] = []
        for filename, content in self._sanitize_prompt_overrides(prompt_overrides).items():
            path = prompt_dir / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(content or ""), encoding="utf-8")
            written.append(str(path))
        return written

    def _collect_graph_logs(self, graph_id: str, *, limit: int = 8) -> List[Dict[str, Any]]:
        log_dir = self._graph_log_dir(graph_id)
        if not log_dir.exists():
            return []
        entries: List[Dict[str, Any]] = []
        for path in sorted((item for item in log_dir.rglob("*") if item.is_file()), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
            try:
                preview = path.read_text(encoding="utf-8", errors="ignore")[-2000:]
            except Exception:
                preview = ""
            entries.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "modified_at": dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc).isoformat(),
                    "preview": preview,
                }
            )
        return entries

    def _validate_graphrag_connectivity(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        base_url = str(profile.get("api_base") or "").strip().rstrip("/")
        if not base_url:
            return {"ok": True, "status": "skipped", "detail": "No GraphRAG base URL configured; skipping remote validation."}
        url = f"{base_url}/models"
        headers = {}
        api_key = str(profile.get("api_key") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            response = httpx.get(url, headers=headers, timeout=max(5, min(30, int(profile.get("timeout_seconds") or 30))))
            detail = {"status_code": response.status_code}
            if response.status_code >= 400:
                return {"ok": False, "status": "error", "detail": f"GraphRAG model endpoint returned HTTP {response.status_code}.", "response": detail}
            try:
                payload = response.json()
            except Exception:
                payload = {}
            models = [
                str(item.get("id") or "")
                for item in (payload.get("data") or [])
                if isinstance(item, dict) and str(item.get("id") or "").strip()
            ]
            configured_models = _dedupe(
                [
                    str(profile.get("chat_model") or ""),
                    str(profile.get("index_chat_model") or ""),
                    str(profile.get("community_report_chat_model") or ""),
                    str(profile.get("embed_model") or ""),
                ]
            )
            missing = [
                model_name
                for model_name in configured_models
                if model_name and model_name not in models
            ]
            return {
                "ok": not missing,
                "status": "ready" if not missing else "warning",
                "detail": "Validated GraphRAG model endpoint." if not missing else "Model endpoint is reachable but one or more configured models were not advertised.",
                "models": models[:50],
                "missing_models": missing,
            }
        except Exception as exc:
            return {"ok": False, "status": "error", "detail": f"Could not reach the configured GraphRAG model endpoint: {exc}"}

    def _resolve_source_documents(
        self,
        *,
        collection_id: str,
        source_doc_ids: Sequence[str] | None = None,
        source_paths: Sequence[str] | None = None,
    ) -> List[Any]:
        doc_store = getattr(self.stores, "doc_store", None)
        if doc_store is None:
            return []

        explicit_doc_ids = _dedupe(source_doc_ids or [])
        if explicit_doc_ids:
            records: List[Any] = []
            for doc_id in explicit_doc_ids:
                try:
                    record = doc_store.get_document(doc_id, self.tenant_id)
                except Exception:
                    record = None
                if record is not None:
                    records.append(record)
            return records

        resolved_paths = {
            canonicalize_local_source_path(path)
            for path in (source_paths or [])
            if str(path).strip()
        }
        try:
            records = doc_store.list_documents(tenant_id=self.tenant_id, collection_id=collection_id)
        except Exception:
            records = []
        if not resolved_paths:
            return list(records)
        return [
            record
            for record in records
            if canonicalize_local_source_path(str(getattr(record, "source_path", "") or "")) in resolved_paths
        ]

    def _build_source_records(self, graph_id: str, records: Sequence[Any]) -> List[GraphIndexSourceRecord]:
        sources: List[GraphIndexSourceRecord] = []
        for record in records:
            source_doc_id = str(getattr(record, "doc_id", "") or "")
            source_path = str(getattr(record, "source_path", "") or "")
            sources.append(
                GraphIndexSourceRecord(
                    graph_source_id=_source_record_id(graph_id, source_doc_id, source_path),
                    graph_id=graph_id,
                    tenant_id=self.tenant_id,
                    source_doc_id=source_doc_id,
                    source_path=source_path,
                    source_title=str(getattr(record, "title", "") or source_doc_id),
                    source_type=str(getattr(record, "source_type", "") or ""),
                )
            )
        return sources

    def _summarize_records(self, records: Sequence[Any], *, display_name: str, collection_id: str) -> Dict[str, Any]:
        titles = [str(getattr(record, "title", "") or "") for record in records if str(getattr(record, "title", "") or "").strip()]
        entity_samples: List[str] = []
        seen_entities: set[str] = set()
        for record in records:
            haystack = " ".join(
                [
                    str(getattr(record, "title", "") or ""),
                    str(Path(str(getattr(record, "source_path", "") or "")).stem),
                ]
            )
            for entity in re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", haystack):
                if entity in seen_entities:
                    continue
                seen_entities.add(entity)
                entity_samples.append(entity)
                if len(entity_samples) >= 12:
                    break
            if len(entity_samples) >= 12:
                break

        relationship_samples = [
            f"{str(getattr(record, 'title', '') or getattr(record, 'doc_id', 'document'))} -> collection:{collection_id}"
            for record in records[:6]
        ]
        doc_count = len(records)
        title_preview = ", ".join(title for title in titles[:4] if title)
        if len(titles) > 4:
            title_preview += ", ..."
        domain_summary = (
            f"{display_name or 'Knowledge graph'} covers {doc_count} indexed document(s)"
            f" in collection '{collection_id}'."
        )
        if title_preview:
            domain_summary += f" Representative sources: {title_preview}."
        return {
            "domain_summary": domain_summary,
            "entity_samples": entity_samples,
            "relationship_samples": relationship_samples,
        }

    def _write_manifest(self, root_path: Path, payload: Dict[str, Any]) -> str:
        root_path.mkdir(parents=True, exist_ok=True)
        manifest_path = root_path / "graph_manifest.json"
        manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(manifest_path)

    def _sync_entities_from_artifacts(self, *, graph_record: GraphIndexRecord) -> None:
        entity_store = self._entity_store()
        if entity_store is None:
            return
        artifact_root = Path(graph_record.artifact_path or graph_record.root_path or self._graph_root(graph_record.graph_id))
        try:
            bundle = load_artifact_bundle(
                artifact_root,
                ttl_seconds=int(getattr(self.settings, "graphrag_artifact_cache_ttl_seconds", 300) or 300),
            )
        except Exception:
            return
        entity_rows = bundle.table_rows("entities")
        if not entity_rows:
            return

        text_units = {
            str(row.get("id") or row.get("text_unit_id") or ""): row
            for row in bundle.table_rows("text_units")
            if str(row.get("id") or row.get("text_unit_id") or "").strip()
        }
        entities: List[CanonicalEntityRecord] = []
        aliases: List[EntityAliasRecord] = []
        mentions: List[EntityMentionRecord] = []
        seen_alias_keys: set[tuple[str, str]] = set()

        for row in entity_rows:
            canonical_name = str(row.get("title") or row.get("name") or row.get("human_readable_id") or row.get("id") or "").strip()
            if not canonical_name:
                continue
            entity_id = make_entity_id(
                tenant_id=self.tenant_id,
                collection_id=graph_record.collection_id,
                canonical_name=canonical_name,
            )
            entities.append(
                CanonicalEntityRecord(
                    entity_id=entity_id,
                    tenant_id=self.tenant_id,
                    collection_id=graph_record.collection_id,
                    canonical_name=canonical_name,
                    entity_type=str(row.get("type") or row.get("entity_type") or ""),
                    description=str(row.get("description") or row.get("summary") or ""),
                    graph_id=graph_record.graph_id,
                    metadata={"graph_id": graph_record.graph_id, "raw_row": dict(row)},
                )
            )

            alias_values = _dedupe(
                [
                    canonical_name,
                    str(row.get("human_readable_id") or ""),
                    str(row.get("short_id") or ""),
                    *[str(item) for item in (row.get("aliases") or []) if str(item)],
                ]
            )
            for alias in alias_values:
                alias_key = (entity_id, alias)
                if alias_key in seen_alias_keys:
                    continue
                seen_alias_keys.add(alias_key)
                aliases.append(
                    EntityAliasRecord(
                        alias_id=make_alias_id(entity_id=entity_id, alias=alias),
                        entity_id=entity_id,
                        tenant_id=self.tenant_id,
                        collection_id=graph_record.collection_id,
                        alias=alias,
                        source="graphrag_artifact",
                    )
                )

            text_unit_ids = [
                str(item)
                for item in (
                    row.get("text_unit_ids")
                    or row.get("source_chunk_ids")
                    or row.get("chunk_ids")
                    or []
                )
                if str(item)
            ]
            for text_unit_id in text_unit_ids[:24]:
                unit = text_units.get(text_unit_id)
                if unit is None:
                    continue
                doc_id = str(unit.get("doc_id") or unit.get("document_id") or "")
                chunk_id = str(unit.get("chunk_id") or unit.get("id") or "")
                mention_text = str(unit.get("text") or canonical_name)
                mentions.append(
                    EntityMentionRecord(
                        mention_id=make_mention_id(
                            entity_id=entity_id,
                            doc_id=doc_id,
                            chunk_id=chunk_id,
                            graph_id=graph_record.graph_id,
                            mention_text=mention_text,
                        ),
                        entity_id=entity_id,
                        tenant_id=self.tenant_id,
                        collection_id=graph_record.collection_id,
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        graph_id=graph_record.graph_id,
                        mention_text=mention_text[:500],
                        mention_type="text_unit",
                        metadata={"text_unit_id": text_unit_id},
                    )
                )
        if entities:
            entity_store.upsert_entities(
                entities=entities,
                aliases=aliases,
                mentions=mentions,
                replace_graph_scope=graph_record.graph_id,
                tenant_id=self.tenant_id,
            )

    def _source_records_for_graph(self, record: GraphIndexRecord) -> List[GraphIndexSourceRecord]:
        source_store = self._source_store()
        if source_store is None:
            return []
        return list(source_store.list_sources(record.graph_id, tenant_id=self.tenant_id))

    def _resolved_records_for_graph(self, record: GraphIndexRecord) -> List[Any]:
        sources = self._source_records_for_graph(record)
        return self._resolve_source_documents(
            collection_id=record.collection_id,
            source_doc_ids=[item.source_doc_id for item in sources if str(item.source_doc_id).strip()],
            source_paths=[item.source_path for item in sources if str(item.source_path).strip()],
        )

    def _write_graph_manifest(
        self,
        *,
        graph_record: GraphIndexRecord,
        source_records: Sequence[GraphIndexSourceRecord],
        materialized_sources: Sequence[Dict[str, Any]] | None = None,
    ) -> str:
        root_path = Path(graph_record.root_path or self._graph_root(graph_record.graph_id))
        root_path.mkdir(parents=True, exist_ok=True)
        source_manifest_path = root_path / "source_manifest.json"
        source_manifest_path.write_text(
            json.dumps(
                {
                    "graph_id": graph_record.graph_id,
                    "tenant_id": graph_record.tenant_id,
                    "collection_id": graph_record.collection_id,
                    "sources": [dict(item) for item in (materialized_sources or [])],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        payload = {
            "graph_id": graph_record.graph_id,
            "display_name": graph_record.display_name,
            "tenant_id": graph_record.tenant_id,
            "collection_id": graph_record.collection_id,
            "backend": graph_record.backend,
            "status": graph_record.status,
            "source_doc_ids": [str(item.source_doc_id or "") for item in source_records if str(item.source_doc_id or "").strip()],
            "source_paths": [str(item.source_path or "") for item in source_records if str(item.source_path or "").strip()],
            "summary": {
                "domain_summary": graph_record.domain_summary,
                "entity_samples": list(graph_record.entity_samples),
                "relationship_samples": list(graph_record.relationship_samples),
            },
            "config_json": dict(graph_record.config_json or {}),
            "prompt_overrides_json": dict(graph_record.prompt_overrides_json or {}),
            "graph_skill_ids": list(graph_record.graph_skill_ids),
            "materialized_sources": [dict(item) for item in (materialized_sources or [])],
            "source_manifest_path": str(source_manifest_path),
        }
        return self._write_manifest(root_path, payload)

    def _safe_project_probe(self, root_path: Path) -> Dict[str, Any]:
        root_path.mkdir(parents=True, exist_ok=True)
        probe = root_path / ".write_probe"
        try:
            probe.write_text("ok\n", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return {"ok": True, "detail": "Graph project path is writable.", "root_path": str(root_path)}
        except Exception as exc:
            return {"ok": False, "detail": f"Graph project path is not writable: {exc}", "root_path": str(root_path)}

    def _can_incremental_refresh(self, record: GraphIndexRecord) -> bool:
        if str(record.status or "").strip().lower() != "ready" or not bool(record.query_ready):
            return False
        root_path = Path(record.root_path or self._graph_root(record.graph_id))
        artifact_root = Path(record.artifact_path or root_path)
        candidate_sets = [
            [
                artifact_root / "documents.parquet",
                artifact_root / "text_units.parquet",
                artifact_root / "entities.parquet",
                artifact_root / "relationships.parquet",
            ],
            [
                artifact_root / "output" / "documents.parquet",
                artifact_root / "output" / "text_units.parquet",
                artifact_root / "output" / "entities.parquet",
                artifact_root / "output" / "relationships.parquet",
            ],
            [
                root_path / "output" / "documents.parquet",
                root_path / "output" / "text_units.parquet",
                root_path / "output" / "entities.parquet",
                root_path / "output" / "relationships.parquet",
            ],
        ]
        return any(all(path.exists() for path in candidates) for candidates in candidate_sets)

    def _admin_graph_payload(
        self,
        record: GraphIndexRecord,
        *,
        include_runs: bool = True,
        include_logs: bool = True,
    ) -> Dict[str, Any]:
        payload = self.inspect_index(record.graph_id)
        if include_runs or include_logs:
            payload.setdefault("graph", asdict(record))
        if include_runs:
            payload.setdefault("runs", self.list_graph_runs(record.graph_id))
        if include_logs:
            payload.setdefault("logs", self._collect_graph_logs(record.graph_id))
        return payload

    def create_admin_graph(
        self,
        *,
        graph_id: str = "",
        display_name: str = "",
        collection_id: str = "",
        source_doc_ids: Sequence[str] | None = None,
        source_paths: Sequence[str] | None = None,
        backend: str = "",
        owner_admin_user_id: str = "",
        visibility: str = "tenant",
        config_overrides: Dict[str, Any] | None = None,
        prompt_overrides: Dict[str, Any] | None = None,
        graph_skill_ids: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        graph_store = self._index_store()
        source_store = self._source_store()
        if graph_store is None or source_store is None:
            return {"error": "Graph catalog stores are unavailable."}

        effective_collection = str(collection_id or getattr(self.settings, "default_collection_id", "default") or "default").strip() or "default"
        effective_name = str(display_name or graph_id or "Managed Graph").strip() or "Managed Graph"
        effective_graph_id = _slugify(graph_id or effective_name)
        owner_id = str(owner_admin_user_id or getattr(self.session, "user_id", "") or getattr(self.settings, "default_user_id", "admin"))
        backend_name = str(backend or getattr(self.settings, "graph_backend", "microsoft_graphrag") or "microsoft_graphrag").strip() or "microsoft_graphrag"
        backend_impl = self._backend_for(backend_name)
        root_path = self._graph_root(effective_graph_id)
        resolved_docs = self._resolve_source_documents(
            collection_id=effective_collection,
            source_doc_ids=source_doc_ids,
            source_paths=source_paths,
        )
        source_records = self._build_source_records(effective_graph_id, resolved_docs)
        summary = self._summarize_records(resolved_docs, display_name=effective_name, collection_id=effective_collection)
        graph_record = self._graph_record_payload(
            graph_id=effective_graph_id,
            display_name=effective_name,
            collection_id=effective_collection,
            backend=backend_name,
            status="draft",
            root_path=root_path,
            artifact_path=str(root_path),
            summary=summary,
            source_doc_ids=[str(getattr(item, "doc_id", "") or "") for item in resolved_docs],
            owner_admin_user_id=owner_id,
            visibility=visibility,
            config_json=dict(config_overrides or {}),
            prompt_overrides_json=self._sanitize_prompt_overrides(prompt_overrides),
            graph_skill_ids=self._sanitize_graph_skills(graph_skill_ids),
            capabilities=["admin_managed", "graphrag_project"],
            supported_query_methods=list(getattr(backend_impl, "supported_query_methods", ()) or ()),
            health={"admin_managed": True, "source_count": len(source_records)},
            freshness_score=0.0,
        )
        graph_store.upsert_index(graph_record)
        source_store.replace_sources(effective_graph_id, tenant_id=self.tenant_id, sources=source_records)
        self._write_graph_manifest(graph_record=graph_record, source_records=source_records)
        self._run_result(
            graph_id=effective_graph_id,
            operation="create",
            status="completed",
            detail="Created admin-managed graph draft.",
            metadata={"source_count": len(source_records), "collection_id": effective_collection},
        )
        return {
            "created": True,
            "graph_id": effective_graph_id,
            "graph": asdict(graph_record),
            "sources": [asdict(item) for item in source_records],
        }

    def update_admin_graph(
        self,
        graph_ref: str,
        *,
        display_name: str | None = None,
        collection_id: str | None = None,
        source_doc_ids: Sequence[str] | None = None,
        source_paths: Sequence[str] | None = None,
        backend: str | None = None,
        owner_admin_user_id: str = "",
        visibility: str | None = None,
        config_overrides: Dict[str, Any] | None = None,
        prompt_overrides: Dict[str, Any] | None = None,
        graph_skill_ids: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        graph_store = self._index_store()
        source_store = self._source_store()
        record = self._resolve_graph_reference(graph_ref)
        if graph_store is None or source_store is None:
            return {"error": "Graph catalog stores are unavailable."}
        if record is None:
            return {"error": f"Graph '{graph_ref}' was not found."}

        next_collection_id = str(collection_id if collection_id is not None else record.collection_id).strip() or record.collection_id
        next_display_name = str(display_name if display_name is not None else record.display_name).strip() or record.display_name
        next_backend = str(backend if backend is not None else record.backend).strip() or record.backend
        if source_doc_ids is None and source_paths is None:
            resolved_docs = self._resolved_records_for_graph(record)
        else:
            resolved_docs = self._resolve_source_documents(
                collection_id=next_collection_id,
                source_doc_ids=source_doc_ids,
                source_paths=source_paths,
            )
        source_records = self._build_source_records(record.graph_id, resolved_docs)
        summary = self._summarize_records(resolved_docs, display_name=next_display_name, collection_id=next_collection_id)
        updated_record = self._graph_record_payload(
            graph_id=record.graph_id,
            display_name=next_display_name,
            collection_id=next_collection_id,
            backend=next_backend,
            status=record.status or "draft",
            root_path=Path(record.root_path or self._graph_root(record.graph_id)),
            artifact_path=record.artifact_path or record.root_path or str(self._graph_root(record.graph_id)),
            summary=summary,
            source_doc_ids=[str(getattr(item, "doc_id", "") or "") for item in resolved_docs],
            owner_admin_user_id=str(owner_admin_user_id or record.owner_admin_user_id or getattr(self.session, "user_id", "")),
            visibility=visibility or record.visibility,
            config_json=dict(config_overrides if config_overrides is not None else record.config_json),
            prompt_overrides_json=self._sanitize_prompt_overrides(
                prompt_overrides if prompt_overrides is not None else record.prompt_overrides_json
            ),
            graph_skill_ids=self._sanitize_graph_skills(
                graph_skill_ids if graph_skill_ids is not None else record.graph_skill_ids
            ),
            capabilities=list(record.capabilities),
            supported_query_methods=list(record.supported_query_methods),
            query_ready=bool(record.query_ready),
            query_backend=record.query_backend,
            artifact_tables=list(record.artifact_tables),
            artifact_mtime=record.artifact_mtime,
            graph_context_summary=dict(record.graph_context_summary or {}),
            health=dict(record.health or {}),
            freshness_score=float(record.freshness_score or 0.0),
        )
        updated_record.last_indexed_at = record.last_indexed_at
        graph_store.upsert_index(updated_record)
        source_store.replace_sources(record.graph_id, tenant_id=self.tenant_id, sources=source_records)
        self._write_graph_manifest(graph_record=updated_record, source_records=source_records)
        self._run_result(
            graph_id=record.graph_id,
            operation="update",
            status="completed",
            detail="Updated admin-managed graph configuration.",
            metadata={"source_count": len(source_records), "collection_id": next_collection_id},
        )
        return self._admin_graph_payload(updated_record)

    def validate_admin_graph(self, graph_ref: str) -> Dict[str, Any]:
        record = self._resolve_graph_reference(graph_ref)
        if record is None:
            return {"error": f"Graph '{graph_ref}' was not found."}
        backend = self._backend_for(record.backend)
        profile = self._graphrag_profile()
        runtime_validation = (
            backend.validate_runtime()
            if hasattr(backend, "validate_runtime")
            else {"ok": True, "detail": "Backend does not expose explicit runtime validation."}
        )
        connectivity = self._validate_graphrag_connectivity(profile)
        probe = self._safe_project_probe(Path(record.root_path or self._graph_root(record.graph_id)))
        source_records = self._source_records_for_graph(record)
        resolved_docs = self._resolved_records_for_graph(record)
        extraction_preflight = self._validate_extract_graph_preflight(
            record=record,
            profile=profile,
            resolved_docs=resolved_docs,
        )
        community_report_preflight = self._validate_community_report_preflight(
            record=record,
            profile=profile,
        )
        resolved_doc_ids = {str(getattr(item, "doc_id", "") or "") for item in resolved_docs}
        missing_doc_ids = [
            str(item.source_doc_id)
            for item in source_records
            if str(item.source_doc_id or "").strip() and str(item.source_doc_id or "") not in resolved_doc_ids
        ]
        profile_summary = {
            "provider": profile["model_provider"],
            "api_base": profile["api_base"],
            "chat_model": profile["chat_model"],
            "index_chat_model": str(profile.get("index_chat_model") or profile["chat_model"]),
            "community_report_mode": str(profile.get("community_report_mode") or "text"),
            "community_report_chat_model": str(
                profile.get("community_report_chat_model") or profile.get("index_chat_model") or profile["chat_model"]
            ),
            "embed_model": profile["embed_model"],
            "concurrency": profile["concurrency"],
            "timeout_seconds": profile["timeout_seconds"],
            "index_timeout_seconds": int(profile.get("index_timeout_seconds") or profile["timeout_seconds"]),
            "community_report_timeout_seconds": int(
                profile.get("community_report_timeout_seconds")
                or profile.get("index_timeout_seconds")
                or profile["timeout_seconds"]
            ),
            "community_report_max_input_length": int(profile.get("community_report_max_input_length") or 4000),
            "community_report_max_length": int(profile.get("community_report_max_length") or 1200),
            "chunk_size": profile["chunk_size"],
            "chunk_overlap": profile["chunk_overlap"],
            "vector_size": self._graphrag_vector_size(profile),
            "api_key_configured": bool(str(profile["api_key"] or "").strip()),
        }
        status = "ready"
        ok = (
            bool(runtime_validation.get("ok", False))
            and bool(connectivity.get("ok", True))
            and bool(probe.get("ok", False))
            and bool(extraction_preflight.get("ok", False))
            and (
                str(community_report_preflight.get("status") or "").strip().lower() == "skipped"
                or bool(community_report_preflight.get("ok", False))
            )
        )
        if not ok:
            status = "error"
        elif (
            missing_doc_ids
            or connectivity.get("status") == "warning"
            or extraction_preflight.get("status") == "warning"
            or community_report_preflight.get("status") == "warning"
        ):
            status = "warning"
        payload = {
            "graph_id": record.graph_id,
            "display_name": record.display_name,
            "status": status,
            "ok": ok and not missing_doc_ids,
            "runtime": runtime_validation,
            "connectivity": connectivity,
            "extraction_preflight": extraction_preflight,
            "community_report_preflight": community_report_preflight,
            "project_probe": probe,
            "profile": profile_summary,
            "source_count": len(source_records),
            "resolved_source_count": len(resolved_docs),
            "missing_source_doc_ids": missing_doc_ids,
            "logs": self._collect_graph_logs(record.graph_id),
        }
        self._run_result(
            graph_id=record.graph_id,
            operation="validate",
            status="completed" if payload["ok"] else "failed",
            detail=f"Validation finished with status '{status}'.",
            metadata={"payload": payload},
        )
        return payload

    def build_admin_graph(self, graph_ref: str, *, refresh: bool = False) -> Dict[str, Any]:
        graph_store = self._index_store()
        source_store = self._source_store()
        record = self._resolve_graph_reference(graph_ref)
        if graph_store is None or source_store is None:
            return {"error": "Graph catalog stores are unavailable."}
        if record is None:
            return {"error": f"Graph '{graph_ref}' was not found."}

        source_records = self._source_records_for_graph(record)
        resolved_docs = self._resolved_records_for_graph(record)
        if not resolved_docs:
            return {"error": "No indexed source documents were resolved for this graph."}

        root_path = Path(record.root_path or self._graph_root(record.graph_id))
        backend = self._backend_for(record.backend)
        profile = self._graphrag_profile()
        runtime_validation = (
            backend.validate_runtime()
            if hasattr(backend, "validate_runtime")
            else {"ok": True, "detail": "Backend does not expose explicit runtime validation."}
        )
        connectivity = self._validate_graphrag_connectivity(profile)
        settings_path = self._graph_settings_path(record.graph_id)
        init_result = GraphOperationResult(status="ready", detail="Existing GraphRAG project reused.")
        if not settings_path.exists():
            init_result = (
                backend.init_project(
                    root_path,
                    chat_model=str(profile["chat_model"] or ""),
                    embed_model=str(profile["embed_model"] or ""),
                )
                if hasattr(backend, "init_project")
                else GraphOperationResult(status="ready", detail="Backend does not require explicit project initialization.")
            )
            if init_result.status == "failed":
                self._run_result(
                    graph_id=record.graph_id,
                    operation="build_init",
                    status="failed",
                    detail=init_result.detail,
                    metadata={"warnings": list(init_result.warnings)},
                )
                return {"error": init_result.detail, "warnings": list(init_result.warnings)}

        use_incremental_refresh = bool(refresh and self._can_incremental_refresh(record))
        cleared_state: Dict[str, List[str]] = {"removed_dirs": []}
        if not use_incremental_refresh:
            cleared_state = self._reset_project_for_full_rebuild(root_path)
        materialized_sources = self._materialize_source_documents(record.graph_id, resolved_docs)
        written_settings = self._render_settings_yaml(record.graph_id, config_overrides=dict(record.config_json or {}))
        written_prompts = self._write_prompt_overrides(record.graph_id, dict(record.prompt_overrides_json or {}))
        source_records = self._build_source_records(record.graph_id, resolved_docs)
        manifest_path = self._write_graph_manifest(
            graph_record=record,
            source_records=source_records,
            materialized_sources=materialized_sources,
        )
        operation = "refresh" if refresh else "build"
        refresh_mode = "incremental_update" if use_incremental_refresh else "full_rebuild"
        prepared_record = replace(
            record,
            health={
                **{key: value for key, value in dict(record.health or {}).items() if key != "warnings"},
                "materialized_source_count": len(materialized_sources),
                "materialized_source_paths": [str(item.get("materialized_path") or "") for item in materialized_sources],
                "settings_path": written_settings,
                "prompt_files": written_prompts,
                "manifest_path": manifest_path,
                "init_status": init_result.status,
                "init_detail": init_result.detail,
                "refresh_mode": refresh_mode,
                "cleared_state": cleared_state,
                "runtime_validation": runtime_validation,
                "connectivity": connectivity,
            },
        )
        self._expire_inflight_runs(record.graph_id, replacement_operation=operation)
        run_id = f"grun_{uuid.uuid4().hex[:16]}"

        launch_result = None
        if hasattr(backend, "launch_index_process"):
            try:
                launch_result = backend.launch_index_process(
                    record.graph_id,
                    root_path,
                    refresh=use_incremental_refresh,
                    run_id=run_id,
                )
            except NotImplementedError:
                launch_result = None
            except Exception as exc:
                launch_result = GraphOperationResult(
                    status="failed",
                    detail=f"GraphRAG background build failed unexpectedly: {type(exc).__name__}: {exc}",
                    warnings=["GRAPHRAG_INDEX_EXCEPTION"],
                    capabilities=list(record.capabilities),
                    supported_query_methods=list(record.supported_query_methods),
                    artifact_path=str(root_path),
                    metadata={"exception_type": type(exc).__name__},
                )
        if launch_result is not None:
            if str(launch_result.status or "").strip().lower() in {"queued", "running"}:
                return self._persist_running_graph_state(
                    record=prepared_record,
                    resolved_docs=resolved_docs,
                    source_records=source_records,
                    backend_result=launch_result,
                    runtime_validation=runtime_validation,
                    connectivity=connectivity,
                    init_result=init_result,
                    materialized_sources=materialized_sources,
                    written_settings=written_settings,
                    written_prompts=written_prompts,
                    manifest_path=manifest_path,
                    refresh_mode=refresh_mode,
                    cleared_state=cleared_state,
                    run_id=run_id,
                    operation=operation,
                )
            return self._persist_completed_graph_state(
                record=prepared_record,
                resolved_docs=resolved_docs,
                source_records=source_records,
                backend_result=launch_result,
                run_id=run_id,
                operation=operation,
            )

        try:
            backend_result = backend.index_documents(record.graph_id, root_path, refresh=use_incremental_refresh)
        except Exception as exc:
            backend_result = GraphOperationResult(
                status="failed",
                detail=f"GraphRAG build failed unexpectedly: {type(exc).__name__}: {exc}",
                warnings=["GRAPHRAG_INDEX_EXCEPTION"],
                capabilities=list(record.capabilities),
                supported_query_methods=list(record.supported_query_methods),
                artifact_path=str(root_path),
                metadata={"exception_type": type(exc).__name__},
            )
        return self._persist_completed_graph_state(
            record=prepared_record,
            resolved_docs=resolved_docs,
            source_records=source_records,
            backend_result=backend_result,
            run_id=run_id,
            operation=operation,
        )

    def refresh_admin_graph(self, graph_ref: str) -> Dict[str, Any]:
        return self.build_admin_graph(graph_ref, refresh=True)

    def update_graph_prompts(
        self,
        graph_ref: str,
        *,
        prompt_overrides: Dict[str, Any] | None = None,
        owner_admin_user_id: str = "",
    ) -> Dict[str, Any]:
        record = self._resolve_graph_reference(graph_ref)
        if record is None:
            return {"error": f"Graph '{graph_ref}' was not found."}
        return self.update_admin_graph(
            record.graph_id,
            prompt_overrides=self._sanitize_prompt_overrides(prompt_overrides),
            owner_admin_user_id=owner_admin_user_id or record.owner_admin_user_id,
        )

    def update_graph_skills(
        self,
        graph_ref: str,
        *,
        graph_skill_ids: Sequence[str] | None = None,
        owner_admin_user_id: str = "",
    ) -> Dict[str, Any]:
        record = self._resolve_graph_reference(graph_ref)
        if record is None:
            return {"error": f"Graph '{graph_ref}' was not found."}
        return self.update_admin_graph(
            record.graph_id,
            graph_skill_ids=self._sanitize_graph_skills(graph_skill_ids),
            owner_admin_user_id=owner_admin_user_id or record.owner_admin_user_id,
        )

    def list_graph_runs(self, graph_ref: str) -> List[Dict[str, Any]]:
        record = self._resolve_graph_reference(graph_ref)
        if record is None or self._run_store() is None:
            return []
        runs = self._prioritize_runs_for_display(self._graph_runs(record.graph_id))
        return [asdict(item) for item in runs]

    def list_indexes(self, *, collection_id: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        store = self._index_store()
        if store is None:
            return []
        return [
            self._graph_output_payload(item)
            for item in self._list_index_records(store, collection_id=collection_id, limit=limit)
        ]

    def inspect_index(self, graph_id: str) -> Dict[str, Any]:
        store = self._index_store()
        if store is None:
            return {"error": "Graph catalog is unavailable."}
        record = self._resolve_graph_reference(graph_id)
        if record is None:
            return {"error": f"Graph '{graph_id}' was not found."}
        self._remember_active_graphs([record.graph_id])
        sources = self._source_store().list_sources(record.graph_id, tenant_id=self.tenant_id) if self._source_store() is not None else []
        runs = self._prioritize_runs_for_display(self._graph_runs(record.graph_id))
        graph_payload = self._graph_output_payload(record, runs=runs)
        runs = self._prioritize_runs_for_display(self._graph_runs(record.graph_id))
        return {
            "graph": graph_payload,
            "sources": [asdict(item) for item in sources],
            "runs": [asdict(item) for item in runs],
            "logs": self._collect_graph_logs(record.graph_id),
        }

    def search_indexes(self, query: str, *, collection_id: str = "", limit: int = 6) -> List[Dict[str, Any]]:
        store = self._index_store()
        if store is None:
            return []
        return [
            self._graph_output_payload(item)
            for item in self._search_index_records(store, query, collection_id=collection_id, limit=limit)
        ]

    def _run_result(
        self,
        *,
        graph_id: str,
        operation: str,
        status: str,
        detail: str,
        metadata: Dict[str, Any] | None = None,
        run_id: str = "",
    ) -> str:
        record = GraphIndexRunRecord(
            run_id=run_id or f"grun_{uuid.uuid4().hex[:16]}",
            graph_id=graph_id,
            tenant_id=self.tenant_id,
            operation=operation,
            status=status,
            detail=detail,
            metadata=dict(metadata or {}),
            completed_at="",
        )
        if status not in {"queued", "running"}:
            record.completed_at = str(
                record.metadata.get("completed_at") or dt.datetime.now(dt.timezone.utc).isoformat()
            )
        if self._run_store() is not None:
            self._run_store().upsert_run(record)
        return record.run_id

    def _expire_inflight_runs(self, graph_id: str, *, replacement_operation: str) -> None:
        run_store = self._run_store()
        if run_store is None:
            return
        for existing in run_store.list_runs(graph_id, tenant_id=self.tenant_id):
            if str(existing.status or "").strip().lower() not in {"queued", "running"}:
                continue
            self._terminate_owned_run_process(existing)
            stale = replace(
                existing,
                status="failed",
                detail=f"{existing.detail} Superseded by a newer {replacement_operation} attempt.".strip(),
                completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                metadata={
                    **dict(existing.metadata or {}),
                    "superseded_by_operation": replacement_operation,
                },
            )
            run_store.upsert_run(stale)

    def index_corpus(
        self,
        *,
        graph_id: str = "",
        display_name: str = "",
        collection_id: str = "",
        source_doc_ids: Sequence[str] | None = None,
        source_paths: Sequence[str] | None = None,
        refresh: bool = False,
        backend: str = "",
    ) -> Dict[str, Any]:
        if self._index_store() is None or self._source_store() is None:
            return {"error": "Graph catalog stores are unavailable."}

        effective_collection = str(collection_id or getattr(self.settings, "default_collection_id", "default") or "default")
        effective_name = str(display_name or graph_id or "Managed Graph Index").strip()
        effective_graph_id = _slugify(graph_id or display_name or f"{effective_collection}_graph")
        records = self._resolve_source_documents(
            collection_id=effective_collection,
            source_doc_ids=source_doc_ids,
            source_paths=source_paths,
        )
        if not records:
            return {"error": "No indexed source documents were resolved for this graph."}
        existing = self._resolve_graph_reference(effective_graph_id)
        if existing is None:
            created = self.create_admin_graph(
                graph_id=effective_graph_id,
                display_name=effective_name,
                collection_id=effective_collection,
                source_doc_ids=[str(getattr(item, "doc_id", "") or "") for item in records],
                source_paths=[str(getattr(item, "source_path", "") or "") for item in records],
                backend=backend,
                owner_admin_user_id=self.user_id,
                visibility="tenant",
            )
            if created.get("error"):
                return created
        else:
            updated = self.update_admin_graph(
                existing.graph_id,
                display_name=effective_name or None,
                collection_id=effective_collection,
                source_doc_ids=[str(getattr(item, "doc_id", "") or "") for item in records],
                source_paths=[str(getattr(item, "source_path", "") or "") for item in records],
                backend=(backend or existing.backend),
                owner_admin_user_id=existing.owner_admin_user_id or self.user_id,
            )
            if updated.get("error"):
                return updated

        built = self.build_admin_graph(effective_graph_id, refresh=refresh)
        if built.get("error"):
            return built
        graph = dict(built.get("graph") or {})
        return {
            **built,
            "graph_id": str(graph.get("graph_id") or effective_graph_id),
            "display_name": str(graph.get("display_name") or effective_name),
            "status": str(built.get("status") or graph.get("status") or ""),
            "detail": str(built.get("detail") or ""),
            "warnings": [str(item) for item in (built.get("warnings") or []) if str(item).strip()],
            "source_doc_ids": [str(item) for item in (graph.get("source_doc_ids") or []) if str(item)],
            "manifest_path": str(built.get("manifest_path") or ""),
            "artifact_path": str(graph.get("artifact_path") or ""),
            "backend": str(graph.get("backend") or backend or getattr(self.settings, "graph_backend", "microsoft_graphrag")),
        }

    def import_existing_graph(
        self,
        *,
        graph_id: str = "",
        display_name: str = "",
        collection_id: str = "",
        import_backend: str = "",
        artifact_path: str = "",
        source_doc_ids: Sequence[str] | None = None,
        source_paths: Sequence[str] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        graph_store = self._index_store()
        source_store = self._source_store()
        if graph_store is None or source_store is None:
            return {"error": "Graph catalog stores are unavailable."}

        effective_collection = str(collection_id or getattr(self.settings, "default_collection_id", "default") or "default")
        effective_name = str(display_name or graph_id or "Imported Graph").strip()
        effective_graph_id = _slugify(graph_id or display_name or f"{effective_collection}_imported_graph")
        records = self._resolve_source_documents(
            collection_id=effective_collection,
            source_doc_ids=source_doc_ids,
            source_paths=source_paths,
        )
        source_records = self._build_source_records(effective_graph_id, records)
        summary = self._summarize_records(records, display_name=effective_name, collection_id=effective_collection)
        root_path = self._graph_root(effective_graph_id)
        manifest_path = self._write_manifest(
            root_path,
            {
                "graph_id": effective_graph_id,
                "display_name": effective_name,
                "collection_id": effective_collection,
                "tenant_id": self.tenant_id,
                "artifact_path": artifact_path,
                "import_backend": import_backend or "neo4j",
                "summary": summary,
                "metadata": dict(metadata or {}),
            },
        )
        backend_impl = self._backend_for(import_backend or "neo4j")
        self._expire_inflight_runs(effective_graph_id, replacement_operation="import")
        run_id = self._run_result(
            graph_id=effective_graph_id,
            operation="import",
            status="running",
            detail="Starting graph import",
            metadata={"manifest_path": manifest_path, "artifact_path": artifact_path},
        )
        backend_result = backend_impl.import_existing_graph(
            effective_graph_id,
            root_path,
            artifact_path=artifact_path,
            metadata=metadata,
        )
        graph_record = GraphIndexRecord(
            graph_id=effective_graph_id,
            tenant_id=self.tenant_id,
            collection_id=effective_collection,
            display_name=effective_name,
            backend=backend_impl.backend_name,
            status=backend_result.status,
            root_path=str(root_path),
            artifact_path=artifact_path or backend_result.artifact_path or str(root_path),
            domain_summary=str(summary.get("domain_summary") or ""),
            entity_samples=[str(item) for item in (summary.get("entity_samples") or []) if str(item)],
            relationship_samples=[str(item) for item in (summary.get("relationship_samples") or []) if str(item)],
            source_doc_ids=[str(getattr(item, "doc_id", "") or "") for item in records if str(getattr(item, "doc_id", "") or "")],
            capabilities=list(backend_result.capabilities),
            supported_query_methods=list(backend_result.supported_query_methods),
            query_ready=bool(backend_result.query_ready),
            query_backend=str(backend_result.query_backend or ""),
            artifact_tables=list(backend_result.artifact_tables),
            artifact_mtime=str(backend_result.artifact_mtime or ""),
            graph_context_summary=dict(backend_result.graph_context_summary or {}),
            health={"warnings": list(backend_result.warnings), **dict(backend_result.metadata or {})},
            freshness_score=1.0,
        )
        graph_store.upsert_index(graph_record)
        source_store.replace_sources(effective_graph_id, tenant_id=self.tenant_id, sources=source_records)
        self._sync_entities_from_artifacts(graph_record=graph_record)
        self._run_result(
            graph_id=effective_graph_id,
            operation="import",
            status=backend_result.status,
            detail=backend_result.detail,
            metadata={"warnings": list(backend_result.warnings), "artifact_path": graph_record.artifact_path},
            run_id=run_id,
        )
        return {
            "graph_id": effective_graph_id,
            "display_name": effective_name,
            "status": backend_result.status,
            "detail": backend_result.detail,
            "warnings": list(backend_result.warnings),
            "artifact_path": graph_record.artifact_path,
            "manifest_path": manifest_path,
            "backend": backend_impl.backend_name,
        }

    def refresh_graph_index(self, graph_id: str) -> Dict[str, Any]:
        record = self._resolve_graph_reference(graph_id)
        if record is None:
            return {"error": f"Graph '{graph_id}' was not found."}
        return self.build_admin_graph(record.graph_id, refresh=True)

    def _query_graph_store(
        self,
        *,
        graph_id: str,
        query: str,
        methods: Sequence[str],
        limit: int,
        doc_ids: Sequence[str],
    ) -> List[GraphQueryHit]:
        graph_store = self._graph_store()
        if graph_store is None or not bool(getattr(self.settings, "graph_search_enabled", False)):
            return []
        results: List[GraphQueryHit] = []
        scoped_doc_ids = [str(item) for item in doc_ids if str(item)]
        for method in methods:
            try:
                if method == "global":
                    hits = graph_store.global_search(
                        query,
                        tenant_id=self.tenant_id,
                        limit=max(1, int(limit)),
                        doc_ids=scoped_doc_ids,
                    )
                else:
                    hits = graph_store.local_search(
                        query,
                        tenant_id=self.tenant_id,
                        limit=max(1, int(limit)),
                        doc_ids=scoped_doc_ids,
                    )
            except Exception:
                hits = []
            for hit in hits:
                results.append(
                    GraphQueryHit(
                        graph_id=graph_id,
                        backend=str(getattr(hit, "metadata", {}).get("backend") or "neo4j"),
                        query_method=method,
                        doc_id=str(getattr(hit, "doc_id", "") or ""),
                        chunk_ids=[str(item) for item in (getattr(hit, "chunk_ids", []) or []) if str(item)],
                        score=float(getattr(hit, "score", 0.0) or 0.0),
                        title=str(getattr(hit, "title", "") or ""),
                        source_path=str(getattr(hit, "source_path", "") or ""),
                        source_type=str(getattr(hit, "source_type", "") or ""),
                        relationship_path=[str(item) for item in (getattr(hit, "relationship_path", []) or []) if str(item)],
                        summary=str(getattr(hit, "summary", "") or ""),
                        metadata={"graph_id": graph_id},
                    )
                )
        return results

    def _query_catalog_sources(
        self,
        *,
        graph_id: str,
        query: str,
        methods: Sequence[str],
        limit: int,
        source_records: Sequence[GraphIndexSourceRecord],
    ) -> List[GraphQueryHit]:
        if not source_records:
            return []
        doc_store = getattr(self.stores, "doc_store", None)
        lower_query = str(query or "").strip().lower()
        query_terms = [term for term in re.findall(r"[a-z0-9_-]+", lower_query) if len(term) > 2]
        preferred_method = str((methods or [getattr(self.settings, "graphrag_default_query_method", "local")])[0] or "local")
        results: List[GraphQueryHit] = []
        seen_doc_ids: set[str] = set()

        for source in source_records:
            doc_id = str(source.source_doc_id or "")
            source_path = str(source.source_path or "")
            title = str(source.source_title or doc_id or Path(source_path).name)
            source_type = str(source.source_type or "")
            if doc_store is not None and doc_id:
                try:
                    record = doc_store.get_document(doc_id, self.tenant_id)
                except Exception:
                    record = None
                if record is not None:
                    title = str(getattr(record, "title", "") or title)
                    source_path = str(getattr(record, "source_path", "") or source_path)
                    source_type = str(getattr(record, "source_type", "") or source_type)

            haystack = " ".join([title, Path(source_path).name if source_path else "", source_path]).lower()
            overlap = len([term for term in query_terms if term in haystack])
            if lower_query and lower_query not in haystack and overlap <= 0:
                continue
            if doc_id and doc_id in seen_doc_ids:
                continue
            if doc_id:
                seen_doc_ids.add(doc_id)
            score = 0.32 + min(0.45, overlap * 0.11)
            if lower_query and lower_query in haystack:
                score += 0.15
            results.append(
                GraphQueryHit(
                    graph_id=graph_id,
                    backend="catalog",
                    query_method=preferred_method,
                    doc_id=doc_id,
                    score=score,
                    title=title,
                    source_path=source_path,
                    source_type=source_type,
                    summary=(
                        f"Catalog source candidate from managed graph '{graph_id}'. "
                        "Read the source text before using this as evidence."
                    ),
                    metadata={
                        "graph_id": graph_id,
                        "fallback": "catalog",
                        "catalog_only": True,
                        "evidence_kind": "source_candidate",
                    },
                )
            )
            if len(results) >= max(1, int(limit)):
                break
        return results

    def _source_catalog_lookup(self, source_records: Sequence[GraphIndexSourceRecord]) -> Dict[str, Dict[str, str]]:
        lookup: Dict[str, Dict[str, str]] = {}
        for source in source_records:
            info = {
                "doc_id": str(source.source_doc_id or ""),
                "title": str(source.source_title or source.source_doc_id or ""),
                "source_path": str(source.source_path or ""),
                "source_type": str(source.source_type or ""),
            }
            for value in [info["doc_id"], info["title"], info["source_path"]]:
                for key in _source_lookup_keys(value):
                    lookup[key] = dict(info)
        return lookup

    def _resolve_graph_hit_source(
        self,
        hit: Dict[str, Any],
        *,
        source_lookup: Dict[str, Dict[str, str]],
    ) -> Dict[str, Any]:
        candidates = [
            str(hit.get("doc_id") or ""),
            str(hit.get("title") or ""),
            str(hit.get("source_path") or ""),
        ]
        candidates.extend(str(item) for item in (hit.get("chunk_ids") or []) if str(item))
        metadata = dict(hit.get("metadata") or {})
        source_info = dict(metadata.get("source") or {})
        candidates.extend(
            [
                str(source_info.get("doc_id") or ""),
                str(source_info.get("title") or ""),
                str(source_info.get("source_path") or ""),
                str(source_info.get("materialized_path") or ""),
                str(source_info.get("materialized_filename") or ""),
            ]
        )
        resolved: Dict[str, Any] = dict(source_info)
        for candidate in candidates:
            for key in _source_lookup_keys(candidate):
                if key in source_lookup:
                    resolved = {**{k: v for k, v in resolved.items() if v}, **source_lookup[key]}
                    break
            if resolved.get("doc_id"):
                break
        doc_id = str(resolved.get("doc_id") or hit.get("doc_id") or "").strip()
        record = None
        doc_store = getattr(self.stores, "doc_store", None)
        if doc_store is not None and doc_id:
            try:
                record = doc_store.get_document(doc_id, self.tenant_id)
            except Exception:
                record = None
        if record is not None:
            resolved.update(
                {
                    "doc_id": str(getattr(record, "doc_id", "") or doc_id),
                    "title": str(getattr(record, "title", "") or resolved.get("title") or doc_id),
                    "source_path": str(getattr(record, "source_path", "") or resolved.get("source_path") or ""),
                    "source_type": str(getattr(record, "source_type", "") or resolved.get("source_type") or ""),
                    "collection_id": str(getattr(record, "collection_id", "") or ""),
                    "source_display_path": str(getattr(record, "source_display_path", "") or ""),
                }
            )
        if doc_id:
            resolved.setdefault("doc_id", doc_id)
        return resolved

    def _graph_citation_id(self, graph_id: str, doc_id: str, hit: Dict[str, Any]) -> str:
        for chunk_id in hit.get("chunk_ids") or []:
            text = str(chunk_id or "").strip()
            if text:
                return text
        if doc_id:
            return f"{doc_id}#graph"
        return f"{graph_id}#graph"

    def _attach_graph_citations(
        self,
        payload: Dict[str, Any],
        *,
        source_records: Sequence[GraphIndexSourceRecord],
    ) -> Dict[str, Any]:
        source_lookup = self._source_catalog_lookup(source_records)
        citations_by_id: Dict[str, Dict[str, Any]] = {}
        enriched_results: List[Dict[str, Any]] = []
        graph_id = str(payload.get("graph_id") or "")
        for raw_hit in payload.get("results") or []:
            if not isinstance(raw_hit, dict):
                continue
            hit = dict(raw_hit)
            source = self._resolve_graph_hit_source(hit, source_lookup=source_lookup)
            doc_id = str(source.get("doc_id") or hit.get("doc_id") or "").strip()
            if doc_id:
                hit["doc_id"] = doc_id
            if source.get("source_path") and not str(hit.get("source_path") or "").strip():
                hit["source_path"] = str(source.get("source_path") or "")
            if source.get("source_type") and not str(hit.get("source_type") or "").strip():
                hit["source_type"] = str(source.get("source_type") or "")
            metadata = dict(hit.get("metadata") or {})
            metadata["source"] = {**dict(metadata.get("source") or {}), **source}
            hit["metadata"] = metadata

            if doc_id:
                citation_id = self._graph_citation_id(graph_id, doc_id, hit)
                hit["citation_ids"] = [citation_id]
                citation = {
                    "citation_id": citation_id,
                    "doc_id": doc_id,
                    "title": str(source.get("title") or hit.get("title") or doc_id),
                    "source_type": str(source.get("source_type") or hit.get("source_type") or ""),
                    "location": str(hit.get("query_method") or ""),
                    "snippet": str(hit.get("summary") or "")[:320],
                    "collection_id": str(source.get("collection_id") or ""),
                    "url": build_document_source_url(self.settings, self.session or self, doc_id),
                    "source_path": str(source.get("source_path") or hit.get("source_path") or ""),
                    "evidence_kind": str(metadata.get("evidence_kind") or ""),
                    "catalog_only": bool(metadata.get("catalog_only") or metadata.get("fallback") == "catalog"),
                }
                citations_by_id.setdefault(citation_id, citation)
            else:
                hit.setdefault("citation_ids", [])
            enriched_results.append(hit)
        payload["results"] = enriched_results
        payload["citations"] = list(citations_by_id.values())
        return payload

    def query_index(
        self,
        graph_id: str,
        *,
        query: str,
        methods: Sequence[str] | None = None,
        limit: int = 8,
        doc_ids: Sequence[str] | None = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        store = self._index_store()
        if store is None:
            return {"error": "Graph catalog is unavailable.", "results": []}
        record = self._resolve_graph_reference(graph_id)
        if record is None:
            return {"error": f"Graph '{graph_id}' was not found.", "results": []}
        source_records = self._source_store().list_sources(graph_id, tenant_id=self.tenant_id) if self._source_store() is not None else []
        scoped_doc_ids = _dedupe(
            [*(doc_ids or []), *[item.source_doc_id for item in source_records if item.source_doc_id], *record.source_doc_ids]
        )
        requested_methods, invalid_methods, method_aliases = _normalize_graph_query_methods(
            methods,
            supported=record.supported_query_methods,
            default_method=str(getattr(self.settings, "graphrag_default_query_method", "local") or "local"),
        )
        if invalid_methods:
            supported_methods = _dedupe(record.supported_query_methods or [getattr(self.settings, "graphrag_default_query_method", "local")])
            return {
                "error": (
                    "Unsupported graph query method(s): "
                    + ", ".join(invalid_methods)
                    + ". Supported methods for this graph are: "
                    + ", ".join(supported_methods)
                    + ". Use methods_csv='' for defaults, or 'local', 'global', 'drift'."
                ),
                "graph_id": graph_id,
                "display_name": record.display_name or graph_id,
                "backend": record.backend,
                "query_ready": bool(record.query_ready),
                "supported_query_methods": supported_methods,
                "requested_methods": [str(item) for item in (methods or []) if str(item).strip()],
                "method_aliases": method_aliases,
                "results": [],
                "citations": [],
                "evidence_status": "method_error",
            }
        cache_store = self._query_cache_store()
        if use_cache and cache_store is not None and len(requested_methods) == 1:
            cached = cache_store.get_cached(
                graph_id=graph_id,
                tenant_id=self.tenant_id,
                query_text=query,
                query_method=requested_methods[0],
            )
            if cached is not None:
                self._remember_active_graphs([record.graph_id])
                return self._attach_graph_citations(
                    dict(cached.response_json),
                    source_records=source_records,
                )

        backend = self._backend_for(record.backend)
        results: List[GraphQueryHit] = []
        effective_queries = _expanded_graph_queries(query) or [query]
        if bool(record.query_ready):
            for method in requested_methods:
                for effective_query in effective_queries:
                    try:
                        results.extend(
                            backend.query_index(
                                graph_id,
                                Path(record.artifact_path or record.root_path or self._graph_root(graph_id)),
                                query=effective_query,
                                method=method,
                                limit=max(1, int(limit)),
                                doc_ids=scoped_doc_ids,
                            )
                        )
                    except Exception:
                        continue
        if not results and str(record.backend or "").strip().lower() == "neo4j":
            results = self._query_graph_store(
                graph_id=graph_id,
                query=query,
                methods=requested_methods,
                limit=limit,
                doc_ids=scoped_doc_ids,
            )
        if not results:
            results = self._query_catalog_sources(
                graph_id=graph_id,
                query=query,
                methods=requested_methods,
                limit=limit,
                source_records=source_records,
            )
        payload = {
            "graph_id": graph_id,
            "display_name": record.display_name or graph_id,
            "backend": record.backend,
            "query_ready": bool(record.query_ready),
            "query_backend": record.query_backend,
            "artifact_tables": list(record.artifact_tables),
            "artifact_mtime": record.artifact_mtime,
            "graph_context_summary": dict(record.graph_context_summary or {}),
            "query": query,
            "methods": requested_methods,
            "method_aliases": method_aliases,
            "expanded_queries": effective_queries if len(effective_queries) > 1 else [],
            "results": [asdict(item) for item in results[: max(1, int(limit))]],
        }
        payload = self._attach_graph_citations(payload, source_records=source_records)
        catalog_only = bool(payload["results"]) and all(
            str((item.get("metadata") or {}).get("fallback") or "").strip() == "catalog"
            or str(item.get("backend") or "").strip().lower() == "catalog"
            for item in payload["results"]
            if isinstance(item, dict)
        )
        if catalog_only:
            payload["evidence_status"] = "source_candidates_only"
            payload["requires_source_read"] = True
            payload["warnings"] = [
                "Graph search returned catalog source candidates only; read source text or run grounded RAG before answering."
            ]
        elif payload["results"]:
            payload["evidence_status"] = "grounded_graph_evidence"
            payload["requires_source_read"] = False
        else:
            payload["evidence_status"] = "no_results"
            payload["requires_source_read"] = True
        if use_cache and cache_store is not None and len(requested_methods) == 1:
            cache_store.put_cached(
                graph_id=graph_id,
                tenant_id=self.tenant_id,
                query_text=query,
                query_method=requested_methods[0],
                response_json=payload,
                ttl_seconds=int(getattr(self.settings, "graph_query_cache_ttl_seconds", 900) or 900),
            )
        self._remember_active_graphs([record.graph_id])
        return payload

    def query_across_graphs(
        self,
        query: str,
        *,
        collection_id: str = "",
        graph_ids: Sequence[str] | None = None,
        methods: Sequence[str] | None = None,
        limit: int = 8,
        top_k_graphs: int = 3,
        doc_ids: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        shortlist = (
            [self.inspect_index(graph_id).get("graph", {}) for graph_id in _dedupe(graph_ids or [])]
            if graph_ids
            else self.search_indexes(query, collection_id=collection_id, limit=top_k_graphs)
        )
        shortlist_graph_ids = [str(item.get("graph_id") or "") for item in shortlist if str(item.get("graph_id") or "").strip()]
        self._remember_active_graphs(shortlist_graph_ids)
        aggregated: List[Dict[str, Any]] = []
        citations_by_id: Dict[str, Dict[str, Any]] = {}
        for item in shortlist[: max(1, int(top_k_graphs))]:
            graph_id = str(item.get("graph_id") or "")
            if not graph_id:
                continue
            response = self.query_index(
                graph_id,
                query=query,
                methods=methods,
                limit=limit,
                doc_ids=doc_ids,
            )
            for citation in response.get("citations", []) or []:
                if isinstance(citation, dict):
                    citation_id = str(citation.get("citation_id") or "").strip()
                    if citation_id:
                        citations_by_id.setdefault(citation_id, citation)
            for hit in response.get("results", []) or []:
                if isinstance(hit, dict):
                    aggregated.append(hit)
        aggregated.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        visible_results = aggregated[: max(1, int(limit))]
        catalog_only = bool(visible_results) and all(
            str((item.get("metadata") or {}).get("fallback") or "").strip() == "catalog"
            or str(item.get("backend") or "").strip().lower() == "catalog"
            for item in visible_results
            if isinstance(item, dict)
        )
        payload = {
            "query": query,
            "graph_shortlist": shortlist,
            "results": visible_results,
            "citations": list(citations_by_id.values()),
        }
        if catalog_only:
            payload["evidence_status"] = "source_candidates_only"
            payload["requires_source_read"] = True
            payload["warnings"] = [
                "Graph search returned catalog source candidates only; read source text or run grounded RAG before answering."
            ]
        elif visible_results:
            payload["evidence_status"] = "grounded_graph_evidence"
            payload["requires_source_read"] = False
        else:
            payload["evidence_status"] = "no_results"
            payload["requires_source_read"] = True
        return payload

    def explain_source_plan(
        self,
        query: str,
        *,
        collection_id: str = "",
        controller_hints: Dict[str, Any] | None = None,
        preferred_doc_ids: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        plan = plan_sources(
            query,
            settings=self.settings,
            stores=self.stores,
            session=self.session or type("Session", (), {"tenant_id": self.tenant_id})(),
            controller_hints=controller_hints,
            collection_id=collection_id,
            preferred_doc_ids=preferred_doc_ids,
        )
        return plan.to_dict()


__all__ = ["GraphService"]
