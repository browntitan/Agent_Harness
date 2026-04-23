from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from agentic_chatbot_next.graph.artifacts import (
    load_artifact_bundle,
    normalize_context_hits,
    search_artifact_rows,
    try_graphrag_query,
)


@dataclass
class GraphQueryHit:
    graph_id: str
    backend: str
    query_method: str
    doc_id: str = ""
    chunk_ids: List[str] = field(default_factory=list)
    score: float = 0.0
    title: str = ""
    source_path: str = ""
    source_type: str = ""
    relationship_path: List[str] = field(default_factory=list)
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphOperationResult:
    status: str
    detail: str = ""
    warnings: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    supported_query_methods: List[str] = field(default_factory=list)
    artifact_path: str = ""
    query_ready: bool = False
    query_backend: str = ""
    artifact_tables: List[str] = field(default_factory=list)
    artifact_mtime: str = ""
    graph_context_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphBackendBase:
    backend_name = "base"
    supported_query_methods: tuple[str, ...] = ()

    def __init__(self, settings: Any) -> None:
        self.settings = settings

    def index_documents(self, graph_id: str, root_path: Path, *, refresh: bool = False) -> GraphOperationResult:
        raise NotImplementedError

    def launch_index_process(
        self,
        graph_id: str,
        root_path: Path,
        *,
        refresh: bool = False,
        run_id: str = "",
    ) -> GraphOperationResult:
        raise NotImplementedError

    def import_existing_graph(
        self,
        graph_id: str,
        root_path: Path,
        *,
        artifact_path: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> GraphOperationResult:
        raise NotImplementedError

    def query_index(
        self,
        graph_id: str,
        root_path: Path,
        *,
        query: str,
        method: str,
        limit: int,
        doc_ids: Sequence[str] | None = None,
    ) -> List[GraphQueryHit]:
        return []


class MicrosoftGraphRagBackend(GraphBackendBase):
    backend_name = "microsoft_graphrag"
    supported_query_methods = ("local", "global", "drift")

    def _community_report_mode(self) -> str:
        mode = str(getattr(self.settings, "graphrag_community_report_mode", "text") or "text").strip().lower()
        return mode if mode in {"text", "graph"} else "text"

    def _use_repo_owned_text_build(self) -> bool:
        return self._community_report_mode() == "text"

    def _text_build_command(self, *, root_path: Path, refresh: bool) -> list[str]:
        runner_script = Path(__file__).resolve().with_name("phased_build.py")
        command = [sys.executable, str(runner_script), "--root", str(root_path)]
        if refresh:
            command.append("--refresh")
        return command

    def _artifact_state(self, root_path: Path) -> Dict[str, Any]:
        ttl_seconds = int(getattr(self.settings, "graphrag_artifact_cache_ttl_seconds", 300) or 300)
        try:
            bundle = load_artifact_bundle(root_path, ttl_seconds=ttl_seconds)
            query_ready = bundle.query_ready
            artifact_tables = list(bundle.artifact_tables)
            artifact_mtime = bundle.artifact_mtime
            graph_context_summary = bundle.graph_context_summary()
        except Exception:
            bundle = None
            query_ready = False
            artifact_tables = []
            artifact_mtime = ""
            graph_context_summary = {}
        query_backend = "graphrag_artifacts"
        if query_ready:
            query_backend = "graphrag_python_api_preferred"
        return {
            "bundle": bundle,
            "query_ready": query_ready,
            "query_backend": query_backend,
            "artifact_tables": artifact_tables,
            "artifact_mtime": artifact_mtime,
            "graph_context_summary": graph_context_summary,
            "supported_query_methods": list(bundle.supported_query_methods if bundle is not None else self.supported_query_methods),
            "artifact_path": str(bundle.artifact_dir) if bundle is not None else str(root_path),
        }

    def _cli_prefix(self) -> List[str] | None:
        if bool(getattr(self.settings, "graphrag_use_container", False)):
            image = str(getattr(self.settings, "graphrag_container_image", "") or "").strip()
            if not image or shutil.which("docker") is None:
                return None
            return ["docker", "run", "--rm", "-v"]
        cli_command = str(getattr(self.settings, "graphrag_cli_command", "graphrag") or "graphrag").strip()
        binary = cli_command.split()[0]
        if shutil.which(binary) is None:
            return None
        return cli_command.split()

    def cli_available(self) -> bool:
        return self._cli_prefix() is not None

    def _request_timeout_seconds(self) -> int:
        return max(
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
        )

    def _job_timeout_seconds(self) -> int:
        raw = getattr(
            self.settings,
            "graphrag_job_timeout_seconds",
            getattr(self.settings, "graphrag_timeout_seconds", 0),
        )
        try:
            timeout_seconds = int(raw or 0)
        except (TypeError, ValueError):
            timeout_seconds = 0
        return max(0, timeout_seconds)

    def _build_cli_command(self, args: List[str], *, root_path: Path) -> List[str] | None:
        prefix = self._cli_prefix()
        if prefix is None:
            return None
        if prefix[:3] == ["docker", "run", "--rm"]:
            image = str(getattr(self.settings, "graphrag_container_image", "graphrag:latest") or "graphrag:latest")
            mount = f"{root_path.resolve()}:/workspace"
            return [*prefix, mount, image, *args]
        return [*prefix, *args]

    def validate_runtime(self) -> Dict[str, Any]:
        issues: List[str] = []
        warnings: List[str] = []
        provider = str(getattr(self.settings, "graphrag_llm_provider", "") or "").strip()
        base_url = str(getattr(self.settings, "graphrag_base_url", "") or "").strip()
        chat_model = str(getattr(self.settings, "graphrag_chat_model", "") or "").strip()
        embed_model = str(getattr(self.settings, "graphrag_embed_model", "") or "").strip()
        if not self.cli_available():
            issues.append("GraphRAG CLI or configured container runtime is unavailable.")
        if not provider:
            issues.append("GRAPHRAG_LLM_PROVIDER is required.")
        if not chat_model:
            issues.append("GRAPHRAG_CHAT_MODEL is required.")
        if not embed_model:
            issues.append("GRAPHRAG_EMBED_MODEL is required.")
        if not base_url:
            warnings.append("GRAPHRAG_BASE_URL is blank; GraphRAG will use the provider defaults.")
        return {
            "ok": not issues,
            "provider": provider,
            "base_url": base_url,
            "chat_model": chat_model,
            "embed_model": embed_model,
            "issues": issues,
            "warnings": warnings,
            "cli_available": self.cli_available(),
            "containerized": bool(getattr(self.settings, "graphrag_use_container", False)),
        }

    def _run_cli(self, args: List[str], *, root_path: Path) -> subprocess.CompletedProcess[str] | None:
        command = self._build_cli_command(args, root_path=root_path)
        if command is None:
            return None
        timeout_seconds = self._job_timeout_seconds()
        try:
            run_kwargs: Dict[str, Any] = {
                "cwd": str(root_path),
                "capture_output": True,
                "text": True,
                "check": False,
            }
            if timeout_seconds > 0:
                run_kwargs["timeout"] = timeout_seconds
            return subprocess.run(command, **run_kwargs)
        except subprocess.TimeoutExpired as exc:
            stdout = str(exc.stdout or "")
            stderr = str(exc.stderr or "")
            detail_lines = [
                f"GraphRAG CLI timed out after {timeout_seconds} seconds while running: {' '.join(command)}",
                "Increase GRAPHRAG_JOB_TIMEOUT_SECONDS or lower GRAPHRAG_CONCURRENCY for slower local models.",
            ]
            if stderr.strip():
                detail_lines.append(stderr.strip())
            elif stdout.strip():
                detail_lines.append(stdout.strip()[-2000:])
            return subprocess.CompletedProcess(
                command,
                returncode=124,
                stdout=stdout,
                stderr="\n".join(line for line in detail_lines if line).strip(),
            )

    def init_project(
        self,
        root_path: Path,
        *,
        chat_model: str,
        embed_model: str,
        force: bool = False,
    ) -> GraphOperationResult:
        root_path.mkdir(parents=True, exist_ok=True)
        args = ["init", "--root", str(root_path), "--model", chat_model, "--embedding", embed_model]
        if force:
            args.append("--force")
        result = self._run_cli(args, root_path=root_path)
        if result is None:
            return GraphOperationResult(
                status="catalog_only",
                detail="GraphRAG CLI is unavailable; project scaffolding could not be initialized automatically.",
                warnings=["GRAPHRAG_CLI_UNAVAILABLE"],
                capabilities=["catalog"],
                supported_query_methods=list(self.supported_query_methods),
                artifact_path=str(root_path),
                metadata={"command_available": False},
            )
        if result.returncode != 0:
            return GraphOperationResult(
                status="failed",
                detail=(result.stderr or result.stdout or "").strip()[:4000],
                warnings=["GRAPHRAG_INIT_TIMEOUT" if result.returncode == 124 else "GRAPHRAG_INIT_FAILED"],
                capabilities=["catalog"],
                supported_query_methods=list(self.supported_query_methods),
                artifact_path=str(root_path),
                metadata={"command_available": True, "returncode": result.returncode},
            )
        return GraphOperationResult(
            status="ready",
            detail=(result.stdout or "Initialized GraphRAG project.").strip()[:4000],
            capabilities=["catalog", "graphrag_cli"],
            supported_query_methods=list(self.supported_query_methods),
            artifact_path=str(root_path),
            metadata={"command_available": True, "returncode": result.returncode},
        )

    def launch_index_process(
        self,
        graph_id: str,
        root_path: Path,
        *,
        refresh: bool = False,
        run_id: str = "",
    ) -> GraphOperationResult:
        action = "update" if refresh else "index"
        if self._use_repo_owned_text_build():
            command = self._text_build_command(root_path=root_path, refresh=refresh)
            detail = f"Started phased GraphRAG {action} in the background."
            metadata = {
                "command_available": True,
                "run_mode": "background",
                "build_phase": "phase_1_index",
                "fallback_used": False,
            }
        else:
            command = self._build_cli_command([action, "--root", str(root_path)], root_path=root_path)
            detail = f"Started GraphRAG {action} in the background."
            metadata = {
                "command_available": True,
                "run_mode": "background",
            }
        if command is None:
            artifact_state = self._artifact_state(root_path)
            return GraphOperationResult(
                status="catalog_only",
                detail="GraphRAG CLI is unavailable; registered graph metadata without executing the external indexer.",
                warnings=["GRAPHRAG_CLI_UNAVAILABLE"],
                capabilities=["catalog", "graph_store_fallback"],
                supported_query_methods=list(artifact_state["supported_query_methods"]),
                artifact_path=str(artifact_state["artifact_path"] or root_path),
                query_ready=bool(artifact_state["query_ready"]),
                query_backend=str(artifact_state["query_backend"] or ""),
                artifact_tables=list(artifact_state["artifact_tables"]),
                artifact_mtime=str(artifact_state["artifact_mtime"] or ""),
                graph_context_summary=dict(artifact_state["graph_context_summary"] or {}),
                metadata={"command_available": False},
            )
        logs_dir = root_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        state_path = logs_dir / f"{run_id or graph_id}_job_state.json"
        stream_log_path = logs_dir / f"{run_id or graph_id}_job_stream.log"
        runner_log_path = logs_dir / f"{run_id or graph_id}_runner.log"
        runner_script = Path(__file__).resolve().with_name("job_runner.py")
        timeout_seconds = self._job_timeout_seconds()
        runner_command = [
            sys.executable,
            str(runner_script),
            "--state-path",
            str(state_path),
            "--stream-log-path",
            str(stream_log_path),
            "--runner-log-path",
            str(runner_log_path),
            "--cwd",
            str(root_path),
            "--timeout-seconds",
            str(timeout_seconds),
            "--",
            *command,
        ]
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        with runner_log_path.open("a", encoding="utf-8") as runner_log:
            process = subprocess.Popen(
                runner_command,
                cwd=str(root_path),
                env=env,
                stdout=runner_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return GraphOperationResult(
            status="running",
            detail=detail,
            capabilities=["catalog", "graphrag_cli"],
            supported_query_methods=list(self.supported_query_methods),
            artifact_path=str(root_path),
            metadata={
                **metadata,
                "active_pid": process.pid,
                "active_process_group_id": process.pid,
                "job_timeout_seconds": timeout_seconds,
                "state_path": str(state_path),
                "stream_log_path": str(stream_log_path),
                "runner_log_path": str(runner_log_path),
                "command": command,
            },
        )

    def collect_job_result(
        self,
        graph_id: str,
        root_path: Path,
        *,
        state: Dict[str, Any],
    ) -> GraphOperationResult:
        del graph_id
        returncode = int(state.get("returncode") or 0)
        timed_out = bool(state.get("timed_out", False))
        stream_tail = str(state.get("stream_tail") or "")
        command = " ".join(str(item) for item in (state.get("command") or []) if str(item).strip())
        background_metadata = {
            "command_available": True,
            "run_mode": "background",
            "returncode": returncode,
            "state_path": str(state.get("state_path") or ""),
            "stream_log_path": str(state.get("stream_log_path") or ""),
            "runner_log_path": str(state.get("runner_log_path") or ""),
            "runner_pid": int(state.get("runner_pid") or 0) or None,
            "child_pid": int(state.get("child_pid") or 0) or None,
            "last_heartbeat_at": str(state.get("last_heartbeat_at") or ""),
            "last_output_at": str(state.get("last_output_at") or ""),
            "failure_mode": str(state.get("failure_mode") or ""),
            "build_phase": str(state.get("build_phase") or ""),
            "fallback_used": bool(state.get("fallback_used", False)),
        }
        if state.get("repair_summary"):
            background_metadata["repair_summary"] = dict(state.get("repair_summary") or {})
        if timed_out:
            timeout_seconds = int(state.get("timeout_seconds") or 0)
            detail_lines = [
                f"GraphRAG background job timed out after {timeout_seconds} seconds while running: {command}".strip(),
                "Increase GRAPHRAG_JOB_TIMEOUT_SECONDS or lower GRAPHRAG_CONCURRENCY for slower local models.",
            ]
            if stream_tail.strip():
                detail_lines.append(stream_tail.strip()[-3000:])
            return GraphOperationResult(
                status="failed",
                detail="\n".join(line for line in detail_lines if line).strip()[:4000],
                warnings=["GRAPHRAG_JOB_TIMEOUT"],
                capabilities=["catalog"],
                supported_query_methods=list(self.supported_query_methods),
                artifact_path=str(root_path),
                metadata={**background_metadata, "returncode": returncode or 124, "failure_mode": "timeout"},
            )
        if returncode != 0:
            detail = stream_tail.strip() or str(state.get("detail") or "")
            return GraphOperationResult(
                status="failed",
                detail=detail[:4000],
                warnings=["GRAPHRAG_INDEX_FAILED"],
                capabilities=["catalog"],
                supported_query_methods=list(self.supported_query_methods),
                artifact_path=str(root_path),
                metadata={**background_metadata, "failure_mode": str(state.get("failure_mode") or "nonzero_exit")},
            )
        artifact_state = self._artifact_state(root_path)
        return GraphOperationResult(
            status="ready",
            detail=stream_tail.strip()[:4000],
            capabilities=["catalog", "graph_store_fallback", "graphrag_cli"],
            supported_query_methods=list(artifact_state["supported_query_methods"]),
            artifact_path=str(artifact_state["artifact_path"] or root_path),
            query_ready=bool(artifact_state["query_ready"]),
            query_backend=str(artifact_state["query_backend"] or ""),
            artifact_tables=list(artifact_state["artifact_tables"]),
            artifact_mtime=str(artifact_state["artifact_mtime"] or ""),
            graph_context_summary=dict(artifact_state["graph_context_summary"] or {}),
            metadata=background_metadata,
        )

    def index_documents(self, graph_id: str, root_path: Path, *, refresh: bool = False) -> GraphOperationResult:
        action = "update" if refresh else "index"
        if self._use_repo_owned_text_build():
            command = self._text_build_command(root_path=root_path, refresh=refresh)
            timeout_seconds = self._job_timeout_seconds()
            try:
                run_kwargs: Dict[str, Any] = {
                    "cwd": str(root_path),
                    "capture_output": True,
                    "text": True,
                    "check": False,
                }
                if timeout_seconds > 0:
                    run_kwargs["timeout"] = timeout_seconds
                result = subprocess.run(command, **run_kwargs)
            except subprocess.TimeoutExpired as exc:
                stdout = str(exc.stdout or "")
                stderr = str(exc.stderr or "")
                detail_lines = [
                    f"GraphRAG CLI timed out after {timeout_seconds} seconds while running: {' '.join(command)}",
                    "Increase GRAPHRAG_JOB_TIMEOUT_SECONDS or lower GRAPHRAG_CONCURRENCY for slower local models.",
                ]
                if stderr.strip():
                    detail_lines.append(stderr.strip())
                elif stdout.strip():
                    detail_lines.append(stdout.strip()[-2000:])
                result = subprocess.CompletedProcess(
                    command,
                    returncode=124,
                    stdout=stdout,
                    stderr="\n".join(line for line in detail_lines if line).strip(),
                )
        else:
            result = self._run_cli([action, "--root", str(root_path)], root_path=root_path)
        if result is None:
            artifact_state = self._artifact_state(root_path)
            return GraphOperationResult(
                status="catalog_only",
                detail="GraphRAG CLI is unavailable; registered graph metadata without executing the external indexer.",
                warnings=["GRAPHRAG_CLI_UNAVAILABLE"],
                capabilities=["catalog", "graph_store_fallback"],
                supported_query_methods=list(artifact_state["supported_query_methods"]),
                artifact_path=str(artifact_state["artifact_path"] or root_path),
                query_ready=bool(artifact_state["query_ready"]),
                query_backend=str(artifact_state["query_backend"] or ""),
                artifact_tables=list(artifact_state["artifact_tables"]),
                artifact_mtime=str(artifact_state["artifact_mtime"] or ""),
                graph_context_summary=dict(artifact_state["graph_context_summary"] or {}),
                metadata={"command_available": False},
            )
        if result.returncode != 0:
            return GraphOperationResult(
                status="failed",
                detail=(result.stderr or result.stdout or "").strip()[:4000],
                warnings=["GRAPHRAG_INDEX_TIMEOUT" if result.returncode == 124 else "GRAPHRAG_INDEX_FAILED"],
                capabilities=["catalog"],
                supported_query_methods=list(self.supported_query_methods),
                artifact_path=str(root_path),
                metadata={"command_available": True, "returncode": result.returncode},
            )
        artifact_state = self._artifact_state(root_path)
        return GraphOperationResult(
            status="ready",
            detail=(result.stdout or "").strip()[:4000],
            capabilities=["catalog", "graph_store_fallback", "graphrag_cli"],
            supported_query_methods=list(artifact_state["supported_query_methods"]),
            artifact_path=str(artifact_state["artifact_path"] or root_path),
            query_ready=bool(artifact_state["query_ready"]),
            query_backend=str(artifact_state["query_backend"] or ""),
            artifact_tables=list(artifact_state["artifact_tables"]),
            artifact_mtime=str(artifact_state["artifact_mtime"] or ""),
            graph_context_summary=dict(artifact_state["graph_context_summary"] or {}),
            metadata={"command_available": True, "returncode": result.returncode},
        )

    def import_existing_graph(
        self,
        graph_id: str,
        root_path: Path,
        *,
        artifact_path: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> GraphOperationResult:
        del graph_id, metadata
        artifact_state = self._artifact_state(Path(artifact_path or root_path))
        return GraphOperationResult(
            status="ready",
            detail="Registered an existing GraphRAG-compatible graph artifact for managed access.",
            capabilities=["catalog", "artifact_registration"],
            supported_query_methods=list(artifact_state["supported_query_methods"]),
            artifact_path=artifact_path or str(artifact_state["artifact_path"] or root_path),
            query_ready=bool(artifact_state["query_ready"]),
            query_backend=str(artifact_state["query_backend"] or ""),
            artifact_tables=list(artifact_state["artifact_tables"]),
            artifact_mtime=str(artifact_state["artifact_mtime"] or ""),
            graph_context_summary=dict(artifact_state["graph_context_summary"] or {}),
            metadata={"command_available": self._cli_prefix() is not None},
        )

    def query_index(
        self,
        graph_id: str,
        root_path: Path,
        *,
        query: str,
        method: str,
        limit: int,
        doc_ids: Sequence[str] | None = None,
    ) -> List[GraphQueryHit]:
        artifact_state = self._artifact_state(root_path)
        bundle = artifact_state["bundle"]
        if bundle is None:
            return []
        if method not in set(bundle.supported_query_methods):
            return []

        api_result = try_graphrag_query(
            bundle=bundle,
            method=method,
            query=query,
        )
        results = []
        if not api_result.get("error"):
            results = normalize_context_hits(
                graph_id=graph_id,
                method=method,
                response=api_result.get("response"),
                context=api_result.get("context"),
                limit=limit,
                doc_ids=doc_ids,
                text_unit_doc_map=bundle.text_unit_doc_map(),
            )
        if not results:
            results = search_artifact_rows(
                graph_id=graph_id,
                bundle=bundle,
                query=query,
                method=method,
                limit=limit,
                doc_ids=doc_ids,
            )
        return [
            GraphQueryHit(
                graph_id=graph_id,
                backend=str(item.get("backend") or "graphrag_artifacts"),
                query_method=str(item.get("query_method") or method),
                doc_id=str(item.get("doc_id") or ""),
                chunk_ids=[str(entry) for entry in (item.get("chunk_ids") or []) if str(entry)],
                score=float(item.get("score") or 0.0),
                title=str(item.get("title") or ""),
                source_path=str(item.get("source_path") or ""),
                source_type=str(item.get("source_type") or ""),
                relationship_path=[str(entry) for entry in (item.get("relationship_path") or []) if str(entry)],
                summary=str(item.get("summary") or ""),
                metadata=dict(item.get("metadata") or {}),
            )
            for item in results[: max(1, int(limit))]
        ]


class Neo4jGraphImportBackend(GraphBackendBase):
    backend_name = "neo4j"
    supported_query_methods = ("local", "global")

    def index_documents(self, graph_id: str, root_path: Path, *, refresh: bool = False) -> GraphOperationResult:
        del graph_id, refresh
        return GraphOperationResult(
            status="catalog_only",
            detail="Neo4j imports rely on the live graph store; no separate project indexing was executed.",
            capabilities=["catalog", "neo4j_import", "graph_store_fallback"],
            supported_query_methods=list(self.supported_query_methods),
            artifact_path=str(root_path),
        )

    def import_existing_graph(
        self,
        graph_id: str,
        root_path: Path,
        *,
        artifact_path: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> GraphOperationResult:
        del graph_id, root_path, metadata
        uri = str(getattr(self.settings, "neo4j_uri", "") or "").strip()
        status = "ready" if uri else "catalog_only"
        detail = "Registered an existing Neo4j graph for managed access."
        if not uri:
            detail = "Registered Neo4j graph metadata, but NEO4J_URI is not configured for live querying."
        return GraphOperationResult(
            status=status,
            detail=detail,
            capabilities=["catalog", "neo4j_import", "graph_store_fallback"],
            supported_query_methods=list(self.supported_query_methods),
            artifact_path=artifact_path,
            metadata={"neo4j_uri_configured": bool(uri)},
        )


__all__ = [
    "GraphBackendBase",
    "GraphOperationResult",
    "GraphQueryHit",
    "MicrosoftGraphRagBackend",
    "Neo4jGraphImportBackend",
]
