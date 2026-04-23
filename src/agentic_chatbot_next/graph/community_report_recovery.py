from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import graphrag.data_model.schemas as schemas
from graphrag.index.operations.finalize_community_reports import (
    finalize_community_reports,
)
from graphrag.index.operations.summarize_communities.explode_communities import (
    explode_communities,
)
from graphrag.index.operations.summarize_communities.text_unit_context.prep_text_units import (
    prep_text_units,
)


TEXT_MODE_PHASE_1_WORKFLOWS = [
    "load_input_documents",
    "create_base_text_units",
    "create_final_documents",
    "extract_graph",
    "finalize_graph",
    "extract_covariates",
    "create_communities",
    "create_final_text_units",
]
TEXT_MODE_PHASE_2_REPORT_WORKFLOWS = ["create_community_reports_text"]
TEXT_MODE_PHASE_2_EMBED_WORKFLOWS = ["generate_text_embeddings"]

_DEFAULT_LLM_SUMMARY_PROMPT = """
You are writing a concise community report for a knowledge-graph build.

Return plain text with this structure:
Title: <short title>
Summary: <2-3 sentence summary>
Findings:
- <finding 1>
- <finding 2>
- <finding 3>

Community metadata:
- community_id: {community_id}
- community_level: {community_level}
- community_size: {community_size}
- source_doc_ids: {doc_ids}

Use only the evidence below.

Evidence:
{context}
""".strip()


def _project_output_dir(project_root: Path) -> Path:
    root = Path(project_root).expanduser().resolve()
    candidate = root / "output"
    return candidate if candidate.exists() or candidate.parent.exists() else root


def _read_parquet_df(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def _write_parquet_df(path: Path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(dataframe, preserve_index=False)
    pq.write_table(table, path)


def _normalize_listlike(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if converted is None:
            return []
        if isinstance(converted, list):
            return converted
        if isinstance(converted, tuple):
            return list(converted)
        return [converted]
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _workflow_payload(settings_path: Path, workflows: Sequence[str]) -> dict[str, Any]:
    payload = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        payload = {}
    payload["workflows"] = [str(item) for item in workflows if str(item).strip()]
    return payload


def rewrite_project_workflows(settings_path: Path, workflows: Sequence[str]) -> None:
    payload = _workflow_payload(settings_path, workflows)
    settings_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def run_graphrag_cli_phase(
    *,
    root_path: Path,
    command_prefix: Sequence[str],
    action: str = "index",
    emit_log: Callable[[str], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    log = emit_log or (lambda message: None)
    command = [*command_prefix, str(action or "index"), "--root", str(root_path)]
    log(f"[graph-phase-command] {' '.join(command)}")
    result = subprocess.run(
        command,
        cwd=str(root_path),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = str(result.stdout or "").strip()
    stderr = str(result.stderr or "").strip()
    if stdout:
        log(stdout)
    if stderr:
        log(stderr)
    return result


def build_graphrag_command_prefix(settings: Any) -> list[str]:
    use_container = bool(getattr(settings, "graphrag_use_container", False))
    if use_container:
        image = str(getattr(settings, "graphrag_container_image", "") or "").strip()
        if not image:
            raise RuntimeError("GRAPHRAG_CONTAINER_IMAGE is required when GRAPHRAG_USE_CONTAINER=true.")
        if not shutil.which("docker"):
            raise RuntimeError("Docker is required for containerized GraphRAG execution.")
        raise RuntimeError("Containerized phased GraphRAG builds are not yet supported in the repo-owned recovery path.")
    cli_command = str(getattr(settings, "graphrag_cli_command", "graphrag") or "graphrag").strip()
    command = cli_command.split()
    if not command:
        raise RuntimeError("GRAPHRAG_CLI_COMMAND is empty.")
    return command


def _artifact_inputs_exist(project_root: Path) -> bool:
    output_dir = _project_output_dir(project_root)
    required = [
        output_dir / "communities.parquet",
        output_dir / "entities.parquet",
        output_dir / "text_units.parquet",
    ]
    return all(path.exists() for path in required)


def _load_phase_1_tables(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = _project_output_dir(project_root)
    return (
        _read_parquet_df(output_dir / "communities.parquet"),
        _read_parquet_df(output_dir / "entities.parquet"),
        _read_parquet_df(output_dir / "text_units.parquet"),
    )


def analyze_community_report_inputs(
    project_root: Path,
    *,
    dry_run: bool = True,
) -> Dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    output_dir = _project_output_dir(root)
    if not _artifact_inputs_exist(root):
        return {
            "ok": True,
            "status": "skipped",
            "detail": "Phase-1 GraphRAG artifacts are not available yet.",
            "output_dir": str(output_dir),
            "native_phase2_safe": True,
            "orphan_membership_count": 0,
            "dropped_tuple_count": 0,
            "remaining_orphan_membership_count": 0,
            "affected_community_ids": [],
            "emptied_community_ids": [],
            "dry_run": dry_run,
        }

    communities, entities, text_units = _load_phase_1_tables(root)
    nodes = explode_communities(communities.copy(), entities.copy())
    prepped = prep_text_units(text_units.copy(), nodes.copy())
    valid_pairs = {
        (int(row[schemas.COMMUNITY_ID]), str(row[schemas.ID]))
        for _, row in prepped.dropna(subset=[schemas.COMMUNITY_ID, schemas.ID]).iterrows()
    }
    exploded = communities.loc[
        :, [schemas.COMMUNITY_ID, schemas.COMMUNITY_LEVEL, schemas.TEXT_UNIT_IDS]
    ].explode(schemas.TEXT_UNIT_IDS)
    exploded[schemas.TEXT_UNIT_IDS] = exploded[schemas.TEXT_UNIT_IDS].apply(
        lambda value: str(value or "").strip()
    )
    orphan_rows = exploded[
        exploded.apply(
            lambda row: (
                bool(str(row.get(schemas.TEXT_UNIT_IDS) or "").strip())
                and (int(row[schemas.COMMUNITY_ID]), str(row[schemas.TEXT_UNIT_IDS])) not in valid_pairs
            ),
            axis=1,
        )
    ]

    repaired = communities.copy()
    affected_ids: list[int] = []
    emptied_ids: list[int] = []
    dropped_tuple_count = 0
    if not orphan_rows.empty:
        for index, row in repaired.iterrows():
            community_id = int(row[schemas.COMMUNITY_ID])
            original_ids = [
                str(item).strip()
                for item in _normalize_listlike(row.get(schemas.TEXT_UNIT_IDS))
                if str(item).strip()
            ]
            kept_ids = [
                text_unit_id
                for text_unit_id in original_ids
                if (community_id, text_unit_id) in valid_pairs
            ]
            if len(kept_ids) == len(original_ids):
                continue
            affected_ids.append(community_id)
            dropped_tuple_count += len(original_ids) - len(kept_ids)
            repaired.at[index, schemas.TEXT_UNIT_IDS] = kept_ids
            if original_ids and not kept_ids:
                emptied_ids.append(community_id)

    if not dry_run and dropped_tuple_count > 0:
        _write_parquet_df(output_dir / "communities.parquet", repaired)

    remaining = repaired.loc[
        :, [schemas.COMMUNITY_ID, schemas.COMMUNITY_LEVEL, schemas.TEXT_UNIT_IDS]
    ].explode(schemas.TEXT_UNIT_IDS)
    remaining[schemas.TEXT_UNIT_IDS] = remaining[schemas.TEXT_UNIT_IDS].apply(
        lambda value: str(value or "").strip()
    )
    remaining_orphans = remaining[
        remaining.apply(
            lambda row: (
                bool(str(row.get(schemas.TEXT_UNIT_IDS) or "").strip())
                and (int(row[schemas.COMMUNITY_ID]), str(row[schemas.TEXT_UNIT_IDS])) not in valid_pairs
            ),
            axis=1,
        )
    ]

    repair_needed = not orphan_rows.empty or dropped_tuple_count > 0
    native_safe = not emptied_ids and remaining_orphans.empty
    status = "warning" if repair_needed or not native_safe else "ready"
    if not repair_needed and native_safe:
        detail = "Community-report inputs are consistent with native text report generation."
    elif native_safe:
        detail = "Community-report inputs contain orphan memberships that can be repaired before native text report generation."
    else:
        detail = "Community-report inputs contain orphan memberships that require repair or fallback generation."
    return {
        "ok": True,
        "status": status,
        "detail": detail,
        "output_dir": str(output_dir),
        "native_phase2_safe": native_safe,
        "orphan_membership_count": int(len(orphan_rows)),
        "dropped_tuple_count": int(dropped_tuple_count),
        "remaining_orphan_membership_count": int(len(remaining_orphans)),
        "affected_community_ids": sorted({int(item) for item in affected_ids}),
        "emptied_community_ids": sorted({int(item) for item in emptied_ids}),
        "dry_run": dry_run,
    }


def _profile_from_settings(settings: Any) -> Dict[str, Any]:
    chat_model = str(getattr(settings, "graphrag_chat_model", "") or "").strip()
    index_chat_model = str(getattr(settings, "graphrag_index_chat_model", chat_model) or chat_model or "").strip()
    report_model = str(
        getattr(settings, "graphrag_community_report_chat_model", index_chat_model)
        or index_chat_model
        or chat_model
        or ""
    ).strip()
    timeout_seconds = max(
        30,
        int(
            getattr(
                settings,
                "graphrag_community_report_request_timeout_seconds",
                getattr(settings, "graphrag_index_request_timeout_seconds", getattr(settings, "graphrag_request_timeout_seconds", 300)),
            )
            or 300
        ),
    )
    return {
        "api_base": str(getattr(settings, "graphrag_base_url", "") or "").strip().rstrip("/"),
        "api_key": str(getattr(settings, "graphrag_api_key", "") or "").strip(),
        "chat_model": report_model,
        "timeout_seconds": timeout_seconds,
        "max_input_length": max(
            500,
            int(getattr(settings, "graphrag_community_report_max_input_length", 4000) or 4000),
        ),
        "max_length": max(
            200,
            int(getattr(settings, "graphrag_community_report_max_length", 1200) or 1200),
        ),
    }


def _trim_sentences(text: str, *, max_sentences: int = 2) -> str:
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if segment.strip()]
    return " ".join(sentences[:max_sentences]).strip()


def _context_records_for_reports(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    communities, entities, text_units = _load_phase_1_tables(project_root)
    nodes = explode_communities(communities.copy(), entities.copy())
    prepped = prep_text_units(text_units.copy(), nodes.copy()).rename(
        columns={schemas.ID: schemas.TEXT_UNIT_IDS, schemas.COMMUNITY_ID: schemas.COMMUNITY_ID}
    )
    context = communities.loc[
        :, [schemas.COMMUNITY_ID, schemas.COMMUNITY_LEVEL, schemas.TEXT_UNIT_IDS]
    ].explode(schemas.TEXT_UNIT_IDS)
    context[schemas.TEXT_UNIT_IDS] = context[schemas.TEXT_UNIT_IDS].apply(
        lambda value: str(value or "").strip()
    )
    context = context.merge(
        prepped,
        on=[schemas.TEXT_UNIT_IDS, schemas.COMMUNITY_ID],
        how="left",
    )
    context = context.dropna(subset=[schemas.ALL_DETAILS])
    return communities, text_units, context


def _context_text(details: Sequence[Dict[str, Any]], *, max_chars: int) -> str:
    snippets: list[str] = []
    remaining = max_chars
    for detail in details:
        unit_id = str(detail.get("id") or "").strip()
        degree = detail.get("entity_degree")
        text = re.sub(r"\s+", " ", str(detail.get("text") or "")).strip()
        if not text:
            continue
        snippet = f"[{unit_id or 'unit'} | degree={degree}] {text}"
        if len(snippet) > remaining and snippets:
            break
        if len(snippet) > remaining:
            snippet = snippet[:remaining].rstrip()
        snippets.append(snippet)
        remaining -= len(snippet) + 2
        if remaining <= 0:
            break
    return "\n\n".join(snippets).strip()


def _llm_summary(
    *,
    profile: Dict[str, Any],
    community_id: int,
    community_level: int,
    community_size: int,
    doc_ids: Sequence[str],
    context: str,
) -> Dict[str, str] | None:
    if not str(profile.get("api_base") or "").strip():
        return None
    if not str(profile.get("chat_model") or "").strip():
        return None
    prompt = _DEFAULT_LLM_SUMMARY_PROMPT.format(
        community_id=community_id,
        community_level=community_level,
        community_size=community_size,
        doc_ids=", ".join(str(item) for item in doc_ids if str(item).strip()) or "none",
        context=context,
    )
    headers: Dict[str, str] = {}
    if str(profile.get("api_key") or "").strip():
        headers["Authorization"] = f"Bearer {profile['api_key']}"
    payload = {
        "model": str(profile["chat_model"]),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "timeout": int(profile.get("timeout_seconds") or 300),
    }
    if "gpt-oss" in str(profile["chat_model"]).lower():
        payload["reasoning_effort"] = "low"
        payload["reasoning"] = {"effort": "low"}
    try:
        response = httpx.post(
            f"{str(profile['api_base']).rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=int(profile.get("timeout_seconds") or 300),
        )
        response.raise_for_status()
        content = str(
            (((response.json().get("choices") or [{}])[0] or {}).get("message") or {}).get("content") or ""
        ).strip()
    except Exception:
        return None
    if not content:
        return None
    title = ""
    summary = ""
    findings: List[str] = []
    for raw_line in content.splitlines():
        line = str(raw_line or "").strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("title:"):
            title = line.split(":", 1)[1].strip()
        elif lowered.startswith("summary:"):
            summary = line.split(":", 1)[1].strip()
        elif line.startswith("- "):
            findings.append(line[2:].strip())
    if not summary:
        summary = _trim_sentences(content, max_sentences=2)
    return {
        "title": title.strip(),
        "summary": summary.strip(),
        "full_content": content,
        "findings": findings,
    }


def _deterministic_summary(
    *,
    community_title: str,
    context_details: Sequence[Dict[str, Any]],
    doc_ids: Sequence[str],
) -> Dict[str, str]:
    snippets = [
        re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
        for item in context_details[:3]
        if str(item.get("text") or "").strip()
    ]
    findings = [_trim_sentences(text, max_sentences=1) for text in snippets if text]
    summary = (
        " ".join(findings[:2]).strip()
        or f"{community_title or 'Community'} draws from {len(doc_ids)} source document(s)."
    )
    full_sections = [
        f"# {community_title or 'Community Report'}",
        "",
        "Summary",
        summary,
        "",
        "Key Findings",
    ]
    if findings:
        full_sections.extend(f"- {item}" for item in findings[:5])
    else:
        full_sections.append("- No detailed findings were extractable from the repaired context.")
    if doc_ids:
        full_sections.extend(["", "Source Documents", *[f"- {item}" for item in doc_ids[:8]]])
    return {
        "title": community_title or "Community Report",
        "summary": summary,
        "full_content": "\n".join(full_sections).strip(),
        "findings": findings[:5],
    }


def generate_fallback_community_reports(
    project_root: Path,
    *,
    settings: Any,
    emit_log: Callable[[str], None] | None = None,
) -> Dict[str, Any]:
    log = emit_log or (lambda message: None)
    repair = analyze_community_report_inputs(project_root, dry_run=False)
    communities, text_units, context = _context_records_for_reports(project_root)
    text_unit_doc_map = {
        str(row.get("id") or ""): str(row.get("document_id") or row.get("doc_id") or "")
        for row in text_units.to_dict(orient="records")
        if str(row.get("id") or "").strip()
    }
    grouped_context: Dict[int, List[Dict[str, Any]]] = {}
    for row in context.to_dict(orient="records"):
        details = row.get(schemas.ALL_DETAILS)
        if not isinstance(details, dict):
            continue
        community_id = int(row[schemas.COMMUNITY_ID])
        grouped_context.setdefault(community_id, []).append(
            {
                "id": str(row.get(schemas.TEXT_UNIT_IDS) or ""),
                "text": str(details.get(schemas.TEXT) or ""),
                "entity_degree": float(details.get(schemas.ENTITY_DEGREE) or 0.0),
            }
        )

    profile = _profile_from_settings(settings)
    max_chars = max(1000, int(profile["max_input_length"]) * 4)
    report_rows: list[dict[str, Any]] = []
    llm_count = 0
    deterministic_count = 0
    for community in communities.to_dict(orient="records"):
        community_id = int(community[schemas.COMMUNITY_ID])
        context_details = sorted(
            grouped_context.get(community_id, []),
            key=lambda item: float(item.get("entity_degree") or 0.0),
            reverse=True,
        )
        text_unit_ids = [str(item.get("id") or "") for item in context_details if str(item.get("id") or "").strip()]
        doc_ids = sorted(
            {
                text_unit_doc_map[text_unit_id]
                for text_unit_id in text_unit_ids
                if text_unit_doc_map.get(text_unit_id)
            }
        )
        context_text = _context_text(context_details, max_chars=max_chars)
        llm_payload = _llm_summary(
            profile=profile,
            community_id=community_id,
            community_level=int(community.get(schemas.COMMUNITY_LEVEL) or 0),
            community_size=int(community.get("size") or len(text_unit_ids) or 0),
            doc_ids=doc_ids,
            context=context_text,
        )
        if llm_payload is not None:
            llm_count += 1
            content = llm_payload
            generator = "llm"
        else:
            deterministic_count += 1
            content = _deterministic_summary(
                community_title=str(community.get("title") or f"Community {community_id}"),
                context_details=context_details,
                doc_ids=doc_ids,
            )
            generator = "deterministic"
        report_rows.append(
            {
                "community": community_id,
                "level": int(community.get(schemas.COMMUNITY_LEVEL) or 0),
                "title": content["title"] or str(community.get("title") or f"Community {community_id}"),
                "summary": content["summary"],
                "full_content": content["full_content"],
                "rank": float(community.get("size") or len(text_unit_ids) or 1),
                "rating_explanation": (
                    "Generated by repo-owned fallback after repairing community/text-unit context."
                ),
                "findings": json.dumps(content.get("findings") or [], ensure_ascii=False),
                "full_content_json": json.dumps(
                    {
                        "generator": generator,
                        "community": community_id,
                        "level": int(community.get(schemas.COMMUNITY_LEVEL) or 0),
                        "doc_ids": doc_ids,
                        "text_unit_ids": text_unit_ids,
                    },
                    ensure_ascii=False,
                ),
                "doc_ids": doc_ids,
                "text_unit_ids": text_unit_ids,
            }
        )
    reports_df = pd.DataFrame(report_rows)
    finalized = finalize_community_reports(
        reports_df.loc[
            :,
            [
                "community",
                "level",
                "title",
                "summary",
                "full_content",
                "rank",
                "rating_explanation",
                "findings",
                "full_content_json",
            ],
        ],
        communities,
    )
    extras = reports_df.loc[:, ["community", "doc_ids", "text_unit_ids"]]
    finalized = finalized.merge(extras, on="community", how="left")
    output_path = _project_output_dir(project_root) / "community_reports.parquet"
    _write_parquet_df(output_path, finalized)
    summary = {
        "ok": True,
        "status": "ready",
        "detail": "Generated fallback community_reports.parquet from repaired phase-1 artifacts.",
        "fallback_used": True,
        "output_path": str(output_path),
        "report_count": int(len(finalized)),
        "llm_generated_count": int(llm_count),
        "deterministic_generated_count": int(deterministic_count),
        "repair_summary": repair,
    }
    log(f"[graph-repair-summary] {json.dumps(repair, ensure_ascii=False, sort_keys=True)}")
    log(f"[graph-fallback-summary] {json.dumps(summary, ensure_ascii=False, sort_keys=True)}")
    return summary
