from __future__ import annotations

import asyncio
import concurrent.futures
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:  # pragma: no cover - optional dependency surface
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency surface
    pd = None


REQUIRED_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "global": ("entities", "communities", "community_reports"),
    "local": ("entities", "communities", "community_reports", "relationships", "text_units"),
    "drift": ("entities", "communities", "community_reports", "relationships", "text_units"),
}

_CACHE: dict[str, "GraphRagArtifactBundle"] = {}


def _safe_rows(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        try:
            return list(value.to_dict(orient="records"))
        except Exception:
            pass
    if hasattr(value, "to_pylist"):
        try:
            return list(value.to_pylist())
        except Exception:
            pass
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [part.strip(" '\"") for part in text.split(",") if part.strip(" '\"")]


def _term_overlap(query: str, haystack: str) -> float:
    query_terms = {term for term in str(query or "").lower().split() if len(term) > 2}
    hay_terms = {term for term in str(haystack or "").lower().split() if len(term) > 2}
    if not query_terms or not hay_terms:
        return 0.0
    overlap = len(query_terms & hay_terms)
    return overlap / max(1, len(query_terms))


def _to_iso(mtime: float) -> str:
    return dt.datetime.fromtimestamp(mtime, tz=dt.timezone.utc).isoformat()


def _find_project_root(root_path: Path) -> Path:
    root = root_path.expanduser().resolve()
    if root.is_file():
        root = root.parent
    return root


def _discover_artifacts(project_root: Path) -> tuple[Path, Dict[str, Path]]:
    direct = {path.stem: path for path in project_root.glob("*.parquet")}
    if direct:
        return project_root, direct
    candidates = [
        candidate
        for candidate in [project_root / "output", project_root / "artifacts", project_root / "data"]
        if candidate.exists()
    ]
    candidates.extend(
        candidate
        for candidate in project_root.rglob("*")
        if candidate.is_dir() and any(path.suffix == ".parquet" for path in candidate.iterdir())
    )
    best_dir = project_root
    best_tables: Dict[str, Path] = {}
    best_score = -1
    for candidate in candidates:
        table_map = {path.stem: path for path in candidate.glob("*.parquet")}
        score = len(table_map)
        if score > best_score:
            best_dir = candidate
            best_tables = table_map
            best_score = score
    return best_dir, best_tables


def _community_level(communities: Sequence[Dict[str, Any]]) -> int:
    levels = []
    for row in communities:
        try:
            levels.append(int(row.get("level") or row.get("community_level") or 0))
        except Exception:
            continue
    return max(levels) if levels else 1


@dataclass
class GraphRagArtifactBundle:
    project_root: Path
    artifact_dir: Path
    table_paths: Dict[str, Path]
    rows: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    loaded_at: str = ""
    artifact_mtime: str = ""

    @property
    def artifact_tables(self) -> List[str]:
        return sorted(self.table_paths)

    @property
    def supported_query_methods(self) -> List[str]:
        methods: List[str] = []
        for method, required in REQUIRED_ARTIFACTS.items():
            if all(name in self.table_paths for name in required):
                methods.append(method)
        return methods

    @property
    def query_ready(self) -> bool:
        return bool(self.supported_query_methods)

    def table_rows(self, name: str) -> List[Dict[str, Any]]:
        if name not in self.rows and name in self.table_paths:
            try:
                import pyarrow.parquet as pq
            except Exception as exc:
                raise RuntimeError("pyarrow is required to read GraphRAG parquet artifacts.") from exc
            table = pq.read_table(self.table_paths[name])
            self.rows[name] = list(table.to_pylist())
        return list(self.rows.get(name) or [])

    def graph_context_summary(self) -> Dict[str, Any]:
        entities = self.table_rows("entities")
        relationships = self.table_rows("relationships")
        reports = self.table_rows("community_reports")
        summary = {
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "community_report_count": len(reports),
            "entity_samples": [
                str(row.get("title") or row.get("name") or row.get("id") or "")
                for row in entities[:8]
                if str(row.get("title") or row.get("name") or row.get("id") or "").strip()
            ],
            "relationship_samples": [
                " -> ".join(
                    part
                    for part in [
                        str(row.get("source") or row.get("source_name") or ""),
                        str(row.get("target") or row.get("target_name") or ""),
                    ]
                    if part.strip()
                )
                for row in relationships[:8]
                if row
            ],
            "community_levels": sorted(
                {
                    int(row.get("level") or row.get("community_level") or 0)
                    for row in self.table_rows("communities")
                    if row.get("level") is not None or row.get("community_level") is not None
                }
            ),
        }
        return summary

    def text_unit_doc_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for row in self.table_rows("text_units"):
            doc_id = str(row.get("doc_id") or row.get("document_id") or "").strip()
            if not doc_id:
                continue
            for key in [
                str(row.get("id") or "").strip(),
                str(row.get("chunk_id") or "").strip(),
                *_coerce_list(row.get("chunk_ids")),
            ]:
                if key:
                    mapping[key] = doc_id
        return mapping

    def to_pandas(self) -> Dict[str, Any]:
        if pd is None:
            raise RuntimeError("pandas is required for GraphRAG Python query execution.")
        return {
            name: pd.DataFrame(self.table_rows(name))
            for name in self.table_paths
        }


def load_artifact_bundle(root_path: Path, *, ttl_seconds: int = 300) -> GraphRagArtifactBundle:
    project_root = _find_project_root(root_path)
    artifact_dir, table_paths = _discover_artifacts(project_root)
    cache_key = str(artifact_dir)
    latest_mtime = max((path.stat().st_mtime for path in table_paths.values()), default=0.0)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        try:
            age_seconds = (
                dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(cached.loaded_at)
            ).total_seconds()
        except Exception:
            age_seconds = float(ttl_seconds + 1)
        if age_seconds <= max(1, int(ttl_seconds)) and cached.artifact_mtime == _to_iso(latest_mtime):
            return cached

    bundle = GraphRagArtifactBundle(
        project_root=project_root,
        artifact_dir=artifact_dir,
        table_paths=table_paths,
        loaded_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        artifact_mtime=_to_iso(latest_mtime) if latest_mtime else "",
    )
    _CACHE[cache_key] = bundle
    return bundle


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result(timeout=120)


def try_graphrag_query(
    *,
    bundle: GraphRagArtifactBundle,
    method: str,
    query: str,
    response_type: str = "Evidence-backed summary",
) -> Dict[str, Any]:
    try:  # pragma: no cover - optional dependency surface
        from graphrag import api
        from graphrag.config.load_config import load_config
    except Exception as exc:  # pragma: no cover - optional dependency surface
        return {"error": f"GRAPHRAG_IMPORT_FAILED: {exc}"}

    try:  # pragma: no cover - optional dependency surface
        tables = bundle.to_pandas()
        config = load_config(bundle.project_root)
        entities = tables["entities"]
        communities = tables["communities"]
        reports = tables["community_reports"]
        community_level = _community_level(bundle.table_rows("communities"))
        if method == "global":
            response, context = _run_async(
                api.global_search(
                    config=config,
                    entities=entities,
                    communities=communities,
                    community_reports=reports,
                    community_level=community_level,
                    dynamic_community_selection=False,
                    response_type=response_type,
                    query=query,
                )
            )
        elif method == "drift":
            response, context = _run_async(
                api.drift_search(
                    config=config,
                    entities=entities,
                    communities=communities,
                    community_reports=reports,
                    text_units=tables["text_units"],
                    relationships=tables["relationships"],
                    community_level=community_level,
                    response_type=response_type,
                    query=query,
                )
            )
        else:
            response, context = _run_async(
                api.local_search(
                    config=config,
                    entities=entities,
                    communities=communities,
                    community_reports=reports,
                    text_units=tables["text_units"],
                    relationships=tables["relationships"],
                    covariates=tables.get("covariates"),
                    community_level=community_level,
                    response_type=response_type,
                    query=query,
                )
            )
        return {"response": response, "context": context}
    except Exception as exc:  # pragma: no cover - optional dependency surface
        return {"error": f"GRAPHRAG_QUERY_FAILED: {exc}"}


def _row_text(row: Dict[str, Any], keys: Iterable[str]) -> str:
    return " ".join(str(row.get(key) or "") for key in keys if str(row.get(key) or "").strip())


def _candidate_doc_ids(row: Dict[str, Any], *, text_unit_doc_map: Dict[str, str] | None = None) -> List[str]:
    doc_ids = _coerce_list(
        row.get("doc_ids")
        or row.get("document_ids")
        or row.get("source_doc_ids")
        or row.get("documents")
    )
    if not doc_ids and text_unit_doc_map:
        for chunk_id in _candidate_chunk_ids(row):
            mapped = str(text_unit_doc_map.get(chunk_id) or "").strip()
            if mapped and mapped not in doc_ids:
                doc_ids.append(mapped)
    return [doc_id for doc_id in doc_ids if doc_id]


def _candidate_chunk_ids(row: Dict[str, Any]) -> List[str]:
    return [
        chunk_id
        for chunk_id in _coerce_list(
            row.get("chunk_ids")
            or row.get("text_unit_ids")
            or row.get("source_chunk_ids")
            or row.get("chunk_id")
            or row.get("id")
        )
        if chunk_id
    ]


def _build_hit(
    *,
    graph_id: str,
    method: str,
    backend: str,
    score: float,
    title: str,
    summary: str,
    row: Dict[str, Any],
    text_unit_doc_map: Dict[str, str] | None = None,
    extra_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    chunk_ids = _candidate_chunk_ids(row)
    doc_ids = _candidate_doc_ids(row, text_unit_doc_map=text_unit_doc_map)
    return {
        "graph_id": graph_id,
        "backend": backend,
        "query_method": method,
        "doc_id": (doc_ids or [""])[0],
        "chunk_ids": chunk_ids,
        "score": score,
        "title": title,
        "source_path": str(row.get("source_path") or ""),
        "source_type": str(row.get("source_type") or ""),
        "relationship_path": [
            item
            for item in [
                str(row.get("source") or row.get("source_name") or ""),
                str(row.get("target") or row.get("target_name") or ""),
            ]
            if item
        ],
        "summary": summary,
        "metadata": {**dict(extra_metadata or {}), "raw_row": dict(row)},
    }


def normalize_context_hits(
    *,
    graph_id: str,
    method: str,
    response: Any,
    context: Any,
    limit: int,
    doc_ids: Sequence[str] | None = None,
    text_unit_doc_map: Dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    scoped_doc_ids = {str(item) for item in (doc_ids or []) if str(item)}
    context_rows: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(context, dict):
        context_rows = {str(name): _safe_rows(value) for name, value in context.items()}
    elif isinstance(context, list):
        context_rows = {"context": _safe_rows(context)}
    else:
        context_rows = {"context": []}

    for name, rows in context_rows.items():
        for row in rows:
            title = str(row.get("title") or row.get("name") or row.get("short_id") or row.get("id") or name)
            summary = _row_text(row, ("summary", "description", "text", "full_content", "full_content_json"))
            score = max(0.2, _term_overlap(str(response or ""), title + " " + summary))
            hit = _build_hit(
                graph_id=graph_id,
                method=method,
                backend="graphrag_api",
                score=score,
                title=title,
                summary=summary or str(response or "")[:280],
                row=row,
                text_unit_doc_map=text_unit_doc_map,
                extra_metadata={"context_table": name, "api_response": str(response or "")[:2000]},
            )
            if scoped_doc_ids and hit["doc_id"] and hit["doc_id"] not in scoped_doc_ids:
                continue
            results.append(hit)
    if results:
        results.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        return results[: max(1, int(limit))]
    return []


def search_artifact_rows(
    *,
    graph_id: str,
    bundle: GraphRagArtifactBundle,
    query: str,
    method: str,
    limit: int,
    doc_ids: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    query_text = str(query or "")
    scoped_doc_ids = {str(item) for item in (doc_ids or []) if str(item)}
    candidates: List[Dict[str, Any]] = []
    text_unit_doc_map = bundle.text_unit_doc_map()
    table_order = ["text_units", "relationships", "entities", "community_reports", "communities"]
    if method == "global":
        table_order = ["community_reports", "communities", "entities"]
    elif method == "drift":
        table_order = ["community_reports", "relationships", "text_units", "entities"]

    for table_name in table_order:
        for row in bundle.table_rows(table_name):
            title = str(row.get("title") or row.get("name") or row.get("short_id") or row.get("id") or table_name)
            summary = _row_text(
                row,
                (
                    "summary",
                    "description",
                    "text",
                    "full_content",
                    "full_content_json",
                    "source",
                    "source_name",
                    "target",
                    "target_name",
                ),
            )
            score = _term_overlap(query_text, title + " " + summary)
            if score <= 0:
                continue
            hit = _build_hit(
                graph_id=graph_id,
                method=method,
                backend="graphrag_artifacts",
                score=score + (0.15 if table_name in {"text_units", "community_reports"} else 0.0),
                title=title,
                summary=summary[:500],
                row=row,
                text_unit_doc_map=text_unit_doc_map,
                extra_metadata={"context_table": table_name},
            )
            if scoped_doc_ids and hit["doc_id"] and hit["doc_id"] not in scoped_doc_ids:
                continue
            candidates.append(hit)
    candidates.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return candidates[: max(1, int(limit))]


__all__ = [
    "GraphRagArtifactBundle",
    "REQUIRED_ARTIFACTS",
    "load_artifact_bundle",
    "normalize_context_hits",
    "search_artifact_rows",
    "try_graphrag_query",
]
