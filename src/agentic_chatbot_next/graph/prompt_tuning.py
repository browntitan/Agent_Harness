from __future__ import annotations

import datetime as dt
import difflib
import importlib
import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from agentic_chatbot_next.graph.service import GraphService, _DEFAULT_EXTRACT_GRAPH_PREFLIGHT_PROMPT


COMMON_GRAPHRAG_PROMPT_TARGETS = [
    "extract_graph.txt",
    "summarize_descriptions.txt",
    "community_report_text.txt",
    "community_report_graph.txt",
    "local_search_system_prompt.txt",
    "global_search_map_system_prompt.txt",
    "global_search_reduce_system_prompt.txt",
    "global_search_knowledge_system_prompt.txt",
    "drift_search_system_prompt.txt",
    "drift_reduce_prompt.txt",
    "basic_search_system_prompt.txt",
]

_SAFE_PROMPT_FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,160}$")
_PLACEHOLDER_RE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "between",
    "could",
    "every",
    "from",
    "have",
    "into",
    "more",
    "must",
    "only",
    "other",
    "over",
    "should",
    "than",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "under",
    "using",
    "when",
    "where",
    "which",
    "while",
    "with",
    "within",
    "would",
}

_PROMPT_MODULE_CANDIDATES: Dict[str, List[tuple[str, str]]] = {
    "extract_graph.txt": [
        ("graphrag.prompts.index.extract_graph", "GRAPH_EXTRACTION_PROMPT"),
        ("graphrag.prompts.index.extract_graph", "PROMPT"),
    ],
    "summarize_descriptions.txt": [
        ("graphrag.prompts.index.summarize_descriptions", "SUMMARIZE_DESCRIPTIONS_PROMPT"),
        ("graphrag.prompts.index.summarize_descriptions", "PROMPT"),
    ],
    "community_report_text.txt": [
        ("graphrag.prompts.index.community_report_text", "COMMUNITY_REPORT_TEXT_PROMPT"),
        ("graphrag.prompts.index.community_report_text", "PROMPT"),
    ],
    "community_report_graph.txt": [
        ("graphrag.prompts.index.community_report_graph", "COMMUNITY_REPORT_GRAPH_PROMPT"),
        ("graphrag.prompts.index.community_report_graph", "PROMPT"),
    ],
    "local_search_system_prompt.txt": [
        ("graphrag.prompts.query.local_search_system_prompt", "LOCAL_SEARCH_SYSTEM_PROMPT"),
        ("graphrag.prompts.query.local_search_system_prompt", "PROMPT"),
    ],
    "global_search_map_system_prompt.txt": [
        ("graphrag.prompts.query.global_search_map_system_prompt", "GLOBAL_SEARCH_MAP_SYSTEM_PROMPT"),
        ("graphrag.prompts.query.global_search_map_system_prompt", "PROMPT"),
    ],
    "global_search_reduce_system_prompt.txt": [
        ("graphrag.prompts.query.global_search_reduce_system_prompt", "GLOBAL_SEARCH_REDUCE_SYSTEM_PROMPT"),
        ("graphrag.prompts.query.global_search_reduce_system_prompt", "PROMPT"),
    ],
    "global_search_knowledge_system_prompt.txt": [
        ("graphrag.prompts.query.global_search_knowledge_system_prompt", "GLOBAL_SEARCH_KNOWLEDGE_SYSTEM_PROMPT"),
        ("graphrag.prompts.query.global_search_knowledge_system_prompt", "PROMPT"),
    ],
    "drift_search_system_prompt.txt": [
        ("graphrag.prompts.query.drift_search_system_prompt", "DRIFT_SEARCH_SYSTEM_PROMPT"),
        ("graphrag.prompts.query.drift_search_system_prompt", "PROMPT"),
    ],
    "drift_reduce_prompt.txt": [
        ("graphrag.prompts.query.drift_reduce_prompt", "DRIFT_REDUCE_PROMPT"),
        ("graphrag.prompts.query.drift_reduce_prompt", "PROMPT"),
    ],
    "basic_search_system_prompt.txt": [
        ("graphrag.prompts.query.basic_search_system_prompt", "BASIC_SEARCH_SYSTEM_PROMPT"),
        ("graphrag.prompts.query.basic_search_system_prompt", "PROMPT"),
    ],
}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _dedupe(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    values: List[str] = []
    for item in items:
        value = str(item or "").strip()
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
    return values


def _truncate(text: str, *, limit: int) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 1)].rstrip() + "..."


def _sentences(text: str, *, limit: int = 8) -> List[str]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return []
    values = []
    for sentence in _SENTENCE_RE.split(normalized):
        clean = sentence.strip()
        if len(clean) < 24:
            continue
        values.append(_truncate(clean, limit=360))
        if len(values) >= limit:
            break
    if values:
        return values
    return [_truncate(normalized, limit=360)] if normalized else []


def _top_terms(text: str, *, limit: int = 20) -> List[str]:
    counts: Dict[str, int] = {}
    originals: Dict[str, str] = {}
    for match in _WORD_RE.finditer(str(text or "")):
        word = match.group(0).strip()
        lowered = word.casefold()
        if lowered in _STOPWORDS or len(lowered) < 4:
            continue
        counts[lowered] = counts.get(lowered, 0) + 1
        originals.setdefault(lowered, word)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [originals[key] for key, _ in ranked[:limit]]


def _title_terms(text: str, *, limit: int = 20) -> List[str]:
    matches = re.findall(r"\b(?:[A-Z][A-Za-z0-9_-]{2,})(?:\s+[A-Z][A-Za-z0-9_-]{2,}){0,3}\b", str(text or ""))
    return _dedupe(matches)[:limit]


def _safe_prompt_file(filename: str) -> bool:
    clean = str(filename or "").strip()
    return bool(clean and _SAFE_PROMPT_FILE_RE.fullmatch(clean) and Path(clean).name == clean and ".." not in clean)


def _json_dump(path: Path, payload: Dict[str, Any] | List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _json_load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


class GraphPromptTuningService:
    """Opt-in pre-build corpus research and GraphRAG prompt drafting.

    This service intentionally does not mutate graph prompt overrides while drafting.
    The only mutation path is apply_tuning_run(...), which writes selected drafts
    through GraphService.update_graph_prompts.
    """

    def __init__(self, settings: Any, stores: Any, *, session: Any | None = None) -> None:
        self.settings = settings
        self.stores = stores
        self.session = session
        self.graph_service = GraphService(settings, stores, session=session)
        self.tenant_id = self.graph_service.tenant_id
        self.user_id = self.graph_service.user_id

    def start_tuning_run(
        self,
        graph_ref: str,
        *,
        guidance: str = "",
        target_prompt_files: Sequence[str] | None = None,
        actor: str = "control-panel",
    ) -> Dict[str, Any]:
        record = self.graph_service._resolve_graph_reference(graph_ref)
        if record is None:
            return {"error": f"Graph '{graph_ref}' was not found."}

        targets, target_warnings = self._normalize_targets(target_prompt_files)
        run_id = f"gtune_{uuid.uuid4().hex[:16]}"
        artifact_dir = self._tuning_dir(record, run_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        started_at = _now_iso()
        self.graph_service._run_result(
            graph_id=record.graph_id,
            operation="research_tune",
            status="running",
            detail="Research & Tune started.",
            run_id=run_id,
            metadata={
                "artifact_dir": str(artifact_dir),
                "guidance": str(guidance or ""),
                "target_prompt_files": list(targets),
                "warnings": list(target_warnings),
                "actor": actor,
                "started_at": started_at,
            },
        )

        try:
            payload = self._execute_tuning_run(
                record=record,
                run_id=run_id,
                artifact_dir=artifact_dir,
                guidance=str(guidance or ""),
                target_prompt_files=targets,
                initial_warnings=target_warnings,
                actor=actor,
                started_at=started_at,
            )
        except Exception as exc:
            failure_manifest = {
                "version": 1,
                "run_id": run_id,
                "graph_id": record.graph_id,
                "status": "failed",
                "detail": f"Research & Tune failed: {type(exc).__name__}: {exc}",
                "warnings": [*target_warnings, "RESEARCH_TUNE_EXCEPTION"],
                "created_at": started_at,
                "completed_at": _now_iso(),
            }
            _json_dump(artifact_dir / "manifest.json", failure_manifest)
            self.graph_service._run_result(
                graph_id=record.graph_id,
                operation="research_tune",
                status="failed",
                detail=failure_manifest["detail"],
                run_id=run_id,
                metadata={
                    "artifact_dir": str(artifact_dir),
                    "manifest_path": str(artifact_dir / "manifest.json"),
                    "warnings": failure_manifest["warnings"],
                    "actor": actor,
                    "completed_at": failure_manifest["completed_at"],
                },
            )
            return self.get_tuning_run(record.graph_id, run_id)

        self.graph_service._run_result(
            graph_id=record.graph_id,
            operation="research_tune",
            status="completed",
            detail="Research & Tune completed. Review generated prompt drafts before applying.",
            run_id=run_id,
            metadata={
                "artifact_dir": str(artifact_dir),
                "manifest_path": str(artifact_dir / "manifest.json"),
                "warnings": list(payload.get("warnings") or []),
                "coverage": dict(payload.get("coverage") or {}),
                "prompt_files": sorted(dict(payload.get("prompt_drafts") or {}).keys()),
                "actor": actor,
                "completed_at": payload.get("completed_at") or _now_iso(),
            },
        )
        return self.get_tuning_run(record.graph_id, run_id)

    def get_tuning_run(self, graph_ref: str, run_id: str) -> Dict[str, Any]:
        record = self.graph_service._resolve_graph_reference(graph_ref)
        if record is None:
            return {"error": f"Graph '{graph_ref}' was not found."}
        clean_run_id = str(run_id or "").strip()
        if not clean_run_id or "/" in clean_run_id or ".." in clean_run_id:
            return {"error": "Invalid tuning run id."}
        artifact_dir = self._tuning_dir(record, clean_run_id)
        manifest_path = artifact_dir / "manifest.json"
        if not manifest_path.exists():
            return {"error": f"Research & Tune run '{clean_run_id}' was not found."}

        manifest = dict(_json_load(manifest_path, {}))
        corpus_profile = dict(_json_load(artifact_dir / "corpus_profile.json", {}))
        prompt_drafts = dict(_json_load(artifact_dir / "prompt_drafts.json", {}))
        prompt_diffs = dict(_json_load(artifact_dir / "prompt_diffs.json", {}))
        doc_digests = self._read_jsonl(artifact_dir / "doc_digests.jsonl")
        scratchpad_text = ""
        scratchpad_path = artifact_dir / "scratchpad.md"
        if scratchpad_path.exists():
            scratchpad_text = scratchpad_path.read_text(encoding="utf-8", errors="ignore")
        run = self._find_run(record.graph_id, clean_run_id)
        status = str(getattr(run, "status", "") or manifest.get("status") or "completed")
        detail = str(getattr(run, "detail", "") or manifest.get("detail") or "")
        return {
            "run_id": clean_run_id,
            "graph_id": record.graph_id,
            "status": status,
            "detail": detail,
            "artifact_dir": str(artifact_dir),
            "manifest_path": str(manifest_path),
            "scratchpad_path": str(scratchpad_path),
            "scratchpad_preview": scratchpad_text[:12000],
            "manifest": manifest,
            "coverage": dict(manifest.get("coverage") or {}),
            "warnings": [str(item) for item in (manifest.get("warnings") or []) if str(item)],
            "corpus_profile": corpus_profile,
            "doc_digests": doc_digests[:100],
            "prompt_drafts": prompt_drafts,
            "prompt_diffs": prompt_diffs,
        }

    def apply_tuning_run(
        self,
        graph_ref: str,
        run_id: str,
        *,
        prompt_files: Sequence[str] | None = None,
        actor: str = "control-panel",
    ) -> Dict[str, Any]:
        record = self.graph_service._resolve_graph_reference(graph_ref)
        if record is None:
            return {"error": f"Graph '{graph_ref}' was not found."}
        payload = self.get_tuning_run(record.graph_id, run_id)
        if payload.get("error"):
            return payload
        if str(payload.get("status") or "").strip().lower() not in {"completed", "ready"}:
            return {"error": "Research & Tune drafts can only be applied after a completed run.", "run": payload}

        drafts = dict(payload.get("prompt_drafts") or {})
        requested = _dedupe(prompt_files or drafts.keys())
        selected: Dict[str, str] = {}
        skipped: List[str] = []
        for filename in requested:
            draft = dict(drafts.get(filename) or {})
            content = str(draft.get("content") or "")
            validation = dict(draft.get("validation") or {})
            if not draft or not content.strip() or validation.get("ok") is False:
                skipped.append(filename)
                continue
            selected[filename] = content
        if not selected:
            return {
                "error": "No valid prompt drafts were selected for apply.",
                "requested_prompt_files": requested,
                "skipped_prompt_files": skipped,
            }

        merged = {
            **dict(record.prompt_overrides_json or {}),
            **selected,
        }
        graph_payload = self.graph_service.update_graph_prompts(
            record.graph_id,
            prompt_overrides=merged,
            owner_admin_user_id=self.user_id or record.owner_admin_user_id,
        )
        if graph_payload.get("error"):
            return graph_payload

        artifact_dir = self._tuning_dir(record, str(run_id))
        manifest_path = artifact_dir / "manifest.json"
        manifest = dict(_json_load(manifest_path, {}))
        apply_record = {
            "applied_at": _now_iso(),
            "applied_by": actor,
            "applied_prompt_files": sorted(selected.keys()),
            "skipped_prompt_files": skipped,
        }
        manifest["apply"] = apply_record
        manifest["applied_prompt_files"] = sorted(
            _dedupe([*[str(item) for item in (manifest.get("applied_prompt_files") or [])], *selected.keys()])
        )
        _json_dump(manifest_path, manifest)
        self.graph_service._run_result(
            graph_id=record.graph_id,
            operation="research_tune_apply",
            status="completed",
            detail=f"Applied {len(selected)} Research & Tune prompt draft(s).",
            metadata={
                "research_tune_run_id": str(run_id),
                "applied_prompt_files": sorted(selected.keys()),
                "skipped_prompt_files": skipped,
                "actor": actor,
            },
        )
        return {
            "applied": True,
            "run_id": str(run_id),
            "graph_id": record.graph_id,
            "applied_prompt_files": sorted(selected.keys()),
            "skipped_prompt_files": skipped,
            **graph_payload,
        }

    def _execute_tuning_run(
        self,
        *,
        record: Any,
        run_id: str,
        artifact_dir: Path,
        guidance: str,
        target_prompt_files: Sequence[str],
        initial_warnings: Sequence[str],
        actor: str,
        started_at: str,
    ) -> Dict[str, Any]:
        source_records = self.graph_service._source_records_for_graph(record)
        resolved_docs = self.graph_service._resolved_records_for_graph(record)
        resolved_doc_ids = {str(getattr(item, "doc_id", "") or "") for item in resolved_docs}
        missing_doc_ids = [
            str(item.source_doc_id)
            for item in source_records
            if str(item.source_doc_id or "").strip() and str(item.source_doc_id or "") not in resolved_doc_ids
        ]
        if not resolved_docs:
            raise ValueError("No indexed source documents were resolved for this graph.")

        digests = self._digest_documents(resolved_docs, guidance=guidance)
        corpus_profile = self._build_corpus_profile(
            record=record,
            digests=digests,
            guidance=guidance,
            missing_doc_ids=missing_doc_ids,
        )
        prompt_drafts, prompt_diffs, prompt_warnings = self._draft_prompt_overrides(
            record=record,
            corpus_profile=corpus_profile,
            target_prompt_files=target_prompt_files,
            guidance=guidance,
        )
        coverage = {
            "source_count": len(source_records),
            "resolved_source_count": len(resolved_docs),
            "digested_doc_count": len(digests),
            "missing_source_doc_ids": missing_doc_ids,
            "coverage_state": "complete" if not missing_doc_ids and len(digests) == len(resolved_docs) else "partial",
        }
        warnings = _dedupe([*initial_warnings, *prompt_warnings])
        completed_at = _now_iso()
        manifest = {
            "version": 1,
            "run_id": run_id,
            "graph_id": record.graph_id,
            "tenant_id": self.tenant_id,
            "status": "completed",
            "detail": "Research & Tune completed. Review generated prompt drafts before applying.",
            "actor": actor,
            "guidance": guidance,
            "target_prompt_files": list(target_prompt_files),
            "generated_prompt_files": sorted(prompt_drafts.keys()),
            "coverage": coverage,
            "warnings": warnings,
            "created_at": started_at,
            "completed_at": completed_at,
            "artifacts": {
                "scratchpad": str(artifact_dir / "scratchpad.md"),
                "doc_digests": str(artifact_dir / "doc_digests.jsonl"),
                "corpus_profile": str(artifact_dir / "corpus_profile.json"),
                "prompt_drafts": str(artifact_dir / "prompt_drafts.json"),
                "prompt_diffs": str(artifact_dir / "prompt_diffs.json"),
            },
        }
        self._write_scratchpad(
            artifact_dir / "scratchpad.md",
            record=record,
            corpus_profile=corpus_profile,
            digests=digests,
            prompt_drafts=prompt_drafts,
            warnings=warnings,
        )
        self._write_jsonl(artifact_dir / "doc_digests.jsonl", digests)
        _json_dump(artifact_dir / "corpus_profile.json", corpus_profile)
        _json_dump(artifact_dir / "prompt_drafts.json", prompt_drafts)
        _json_dump(artifact_dir / "prompt_diffs.json", prompt_diffs)
        _json_dump(artifact_dir / "manifest.json", manifest)
        return {
            "run_id": run_id,
            "graph_id": record.graph_id,
            "coverage": coverage,
            "warnings": warnings,
            "prompt_drafts": prompt_drafts,
            "completed_at": completed_at,
        }

    def _digest_documents(self, records: Sequence[Any], *, guidance: str) -> List[Dict[str, Any]]:
        max_workers = max(
            1,
            min(
                len(records),
                int(getattr(self.settings, "max_worker_concurrency", 0) or getattr(self.settings, "graphrag_concurrency", 1) or 1),
            ),
        )
        if max_workers <= 1:
            return [self._digest_document(index, record, guidance=guidance) for index, record in enumerate(records)]

        results: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._digest_document, index, record, guidance=guidance): index
                for index, record in enumerate(records)
            }
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
        return [results[index] for index in sorted(results)]

    def _digest_document(self, index: int, record: Any, *, guidance: str) -> Dict[str, Any]:
        raw_text = self.graph_service._reconstruct_document_text(record)
        text = raw_text[:60000]
        title = str(getattr(record, "title", "") or getattr(record, "doc_id", "") or f"Document {index + 1}")
        doc_id = str(getattr(record, "doc_id", "") or "")
        source_path = str(getattr(record, "source_path", "") or "")
        summary_sentences = _sentences(text, limit=3)
        considerations = [
            sentence
            for sentence in _sentences(text, limit=16)
            if re.search(r"\b(must|shall|should|required|risk|approval|security|privacy|incident|dependency|control|exception|owner|timeline)\b", sentence, re.I)
        ][:8]
        if not considerations:
            considerations = summary_sentences[:2]
        vocabulary = _dedupe([*_title_terms(f"{title}\n{text}", limit=16), *_top_terms(text, limit=18)])[:24]
        entity_types = self._infer_entity_types(f"{title}\n{text}")
        relationship_types = self._infer_relationship_types(text)
        hazards = self._infer_hazards(text)
        prompt_implications = self._prompt_implications(
            title=title,
            vocabulary=vocabulary,
            entity_types=entity_types,
            relationship_types=relationship_types,
            hazards=hazards,
            guidance=guidance,
        )
        return {
            "doc_index": index,
            "document": {
                "doc_id": doc_id,
                "title": title,
                "source_path": source_path,
                "source_type": str(getattr(record, "source_type", "") or ""),
                "collection_id": str(getattr(record, "collection_id", "") or ""),
            },
            "summary": " ".join(summary_sentences).strip() or _truncate(text, limit=700),
            "important_considerations": considerations,
            "candidate_entity_types": entity_types,
            "candidate_relationship_types": relationship_types,
            "domain_vocabulary": vocabulary,
            "aliases": self._infer_aliases(text),
            "extraction_hazards": hazards,
            "prompt_implications": prompt_implications,
            "citations": [doc_id or title],
        }

    def _infer_entity_types(self, text: str) -> List[str]:
        signals = {
            "organization": r"\b(company|vendor|supplier|partner|customer|organization|team|agency|department)\b",
            "person": r"\b(owner|approver|manager|operator|engineer|analyst|contact|person)\b",
            "geo": r"\b(region|country|state|city|location|jurisdiction)\b",
            "event": r"\b(incident|release|deployment|meeting|audit|review|handover|milestone)\b",
            "system": r"\b(system|service|platform|application|api|database|pipeline|integration|workflow)\b",
            "policy": r"\b(policy|standard|control|requirement|procedure|runbook|agreement|contract)\b",
            "risk": r"\b(risk|issue|exception|dependency|vulnerability|failure|breach)\b",
            "metric": r"\b(metric|kpi|score|rate|threshold|sla|slo|latency|cost|budget)\b",
        }
        inferred = [name for name, pattern in signals.items() if re.search(pattern, text, re.I)]
        return inferred or ["organization", "person", "geo", "event"]

    def _infer_relationship_types(self, text: str) -> List[str]:
        signals = {
            "owns": r"\b(owns?|owner|responsible|accountable)\b",
            "depends_on": r"\b(depends? on|dependency|requires?|blocked by)\b",
            "uses": r"\b(uses?|calls?|integrates? with|connects? to)\b",
            "governs": r"\b(governs?|controls?|requires?|policy|standard)\b",
            "approves": r"\b(approves?|approval|sign[- ]?off|authorized)\b",
            "mitigates": r"\b(mitigates?|reduces?|remediates?|prevents?)\b",
            "reports_to": r"\b(reports? to|escalates? to|notifies?)\b",
            "process_step": r"\b(before|after|then|next|handoff|workflow|process)\b",
        }
        inferred = [name for name, pattern in signals.items() if re.search(pattern, text, re.I)]
        return inferred or ["related_to", "mentions"]

    def _infer_hazards(self, text: str) -> List[str]:
        hazards = []
        if re.search(r"\b(v1|v2|version|revision|updated|legacy|deprecated)\b", text, re.I):
            hazards.append("Versioned or legacy terminology may describe similar entities with different names.")
        if re.search(r"\b(exception|unless|except|optional|may)\b", text, re.I):
            hazards.append("Conditional language should be captured without turning exceptions into universal rules.")
        if re.search(r"\b(acronym|aka|also known as|formerly|alias)\b", text, re.I):
            hazards.append("Aliases and acronyms may need entity resolution rather than separate nodes.")
        if re.search(r"\b(table|spreadsheet|column|row|csv)\b", text, re.I):
            hazards.append("Tabular evidence may encode relationships in headings or repeated rows.")
        return hazards

    def _infer_aliases(self, text: str) -> List[Dict[str, str]]:
        aliases = []
        for match in re.finditer(r"\b([A-Z][A-Za-z0-9 _/-]{2,60})\s+\(([A-Z0-9]{2,12})\)", text):
            aliases.append({"name": match.group(1).strip(), "alias": match.group(2).strip()})
            if len(aliases) >= 12:
                break
        return aliases

    def _prompt_implications(
        self,
        *,
        title: str,
        vocabulary: Sequence[str],
        entity_types: Sequence[str],
        relationship_types: Sequence[str],
        hazards: Sequence[str],
        guidance: str,
    ) -> List[str]:
        implications = [
            f"For sources like '{title}', prioritize entity types: {', '.join(entity_types[:8])}.",
            f"Capture relationship types such as: {', '.join(relationship_types[:8])}.",
        ]
        if vocabulary:
            implications.append("Treat repeated domain terms as candidate entity aliases: " + ", ".join(vocabulary[:10]) + ".")
        if hazards:
            implications.append("Preserve caveats during extraction: " + hazards[0])
        if guidance.strip():
            implications.append("Honor operator guidance while preserving grounded extraction: " + _truncate(guidance, limit=240))
        return implications

    def _build_corpus_profile(
        self,
        *,
        record: Any,
        digests: Sequence[Dict[str, Any]],
        guidance: str,
        missing_doc_ids: Sequence[str],
    ) -> Dict[str, Any]:
        entity_types = _dedupe(
            item
            for digest in digests
            for item in list(digest.get("candidate_entity_types") or [])
        )
        relationship_types = _dedupe(
            item
            for digest in digests
            for item in list(digest.get("candidate_relationship_types") or [])
        )
        vocabulary = _dedupe(
            item
            for digest in digests
            for item in list(digest.get("domain_vocabulary") or [])
        )[:48]
        hazards = _dedupe(
            item
            for digest in digests
            for item in list(digest.get("extraction_hazards") or [])
        )[:16]
        considerations = _dedupe(
            item
            for digest in digests
            for item in list(digest.get("important_considerations") or [])
        )[:24]
        summaries = [str(digest.get("summary") or "").strip() for digest in digests if str(digest.get("summary") or "").strip()]
        return {
            "graph_id": record.graph_id,
            "display_name": record.display_name,
            "collection_id": record.collection_id,
            "document_count": len(digests),
            "missing_source_doc_ids": list(missing_doc_ids),
            "operator_guidance": guidance,
            "corpus_summary": _truncate(" ".join(summaries[:8]), limit=2200),
            "candidate_entity_types": entity_types,
            "candidate_relationship_types": relationship_types,
            "domain_vocabulary": vocabulary,
            "important_considerations": considerations,
            "extraction_hazards": hazards,
            "prompt_strategy": self._profile_strategy(entity_types, relationship_types, vocabulary, hazards, guidance),
        }

    def _profile_strategy(
        self,
        entity_types: Sequence[str],
        relationship_types: Sequence[str],
        vocabulary: Sequence[str],
        hazards: Sequence[str],
        guidance: str,
    ) -> List[str]:
        strategy = [
            "Bias extraction toward the corpus-specific entity and relationship types listed in the profile.",
            "Model explicit relationships only when the text supports the edge; do not infer hidden dependencies.",
        ]
        if entity_types:
            strategy.append("Entity type priorities: " + ", ".join(entity_types[:10]) + ".")
        if relationship_types:
            strategy.append("Relationship type priorities: " + ", ".join(relationship_types[:10]) + ".")
        if vocabulary:
            strategy.append("Use domain vocabulary and aliases for entity normalization: " + ", ".join(vocabulary[:18]) + ".")
        if hazards:
            strategy.append("Extraction hazards to guard against: " + " ".join(hazards[:4]))
        if guidance.strip():
            strategy.append("Operator guidance: " + _truncate(guidance, limit=500))
        return strategy

    def _draft_prompt_overrides(
        self,
        *,
        record: Any,
        corpus_profile: Dict[str, Any],
        target_prompt_files: Sequence[str],
        guidance: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any], List[str]]:
        drafts: Dict[str, Any] = {}
        diffs: Dict[str, Any] = {}
        warnings: List[str] = []
        for filename in target_prompt_files:
            if not _safe_prompt_file(filename):
                warnings.append(f"Skipped unsafe prompt target: {filename}")
                continue
            baseline, baseline_source = self._load_baseline_prompt(record, filename)
            if not baseline.strip():
                warnings.append(f"No baseline prompt was available for {filename}; skipped.")
                continue
            content = self._compose_prompt_draft(
                filename=filename,
                baseline=baseline,
                corpus_profile=corpus_profile,
                guidance=guidance,
            )
            validation = self._validate_prompt_draft(
                filename=filename,
                baseline=baseline,
                content=content,
                corpus_profile=corpus_profile,
            )
            draft_warnings = [str(item) for item in validation.get("warnings", []) if str(item)]
            warnings.extend(f"{filename}: {warning}" for warning in draft_warnings)
            drafts[filename] = {
                "prompt_file": filename,
                "content": content,
                "baseline_source": baseline_source,
                "validation": validation,
                "warnings": draft_warnings,
                "summary": f"Dataset-tailored draft for {filename}",
            }
            diffs[filename] = {
                "prompt_file": filename,
                "diff": "\n".join(
                    difflib.unified_diff(
                        baseline.splitlines(),
                        content.splitlines(),
                        fromfile=f"{filename}:baseline",
                        tofile=f"{filename}:draft",
                        lineterm="",
                    )
                ),
            }
        return drafts, diffs, _dedupe(warnings)

    def _load_baseline_prompt(self, record: Any, filename: str) -> tuple[str, str]:
        override = str(dict(record.prompt_overrides_json or {}).get(filename) or "")
        if override.strip():
            return override, "graph_prompt_override"
        for module_name, attr_name in _PROMPT_MODULE_CANDIDATES.get(filename, []):
            try:
                module = importlib.import_module(module_name)
                value = str(getattr(module, attr_name, "") or "").strip()
            except Exception:
                value = ""
            if value:
                return value, f"{module_name}.{attr_name}"
        if filename == "extract_graph.txt":
            return _DEFAULT_EXTRACT_GRAPH_PREFLIGHT_PROMPT, "agentic_chatbot_default_extract_graph"
        return "", ""

    def _compose_prompt_draft(
        self,
        *,
        filename: str,
        baseline: str,
        corpus_profile: Dict[str, Any],
        guidance: str,
    ) -> str:
        strategy = "\n".join(f"- {item}" for item in (corpus_profile.get("prompt_strategy") or []) if str(item).strip())
        entity_types = ", ".join([str(item) for item in (corpus_profile.get("candidate_entity_types") or [])][:16])
        relationship_types = ", ".join([str(item) for item in (corpus_profile.get("candidate_relationship_types") or [])][:16])
        vocabulary = ", ".join([str(item) for item in (corpus_profile.get("domain_vocabulary") or [])][:24])
        hazards = "\n".join(f"- {item}" for item in (corpus_profile.get("extraction_hazards") or [])[:10])
        guidance_block = _truncate(guidance, limit=1200) if guidance.strip() else "No additional operator guidance was provided."
        curation = f"""

######################
-Dataset-Specific Curation Guidance-
######################
Target prompt file: {filename}
Corpus: {corpus_profile.get("display_name") or corpus_profile.get("graph_id")} ({corpus_profile.get("document_count", 0)} source document(s))

Operator guidance:
{guidance_block}

Corpus summary:
{corpus_profile.get("corpus_summary") or "No corpus summary was available."}

Entity type priorities:
{entity_types or "Use GraphRAG defaults, but preserve corpus-specific vocabulary."}

Relationship type priorities:
{relationship_types or "Use GraphRAG defaults, and prefer explicit source-supported relationships."}

Domain vocabulary and aliases:
{vocabulary or "No repeated domain vocabulary was detected."}

Extraction hazards:
{hazards or "- No major extraction hazards were detected."}

Prompt strategy:
{strategy or "- Preserve the baseline prompt behavior and add only corpus-specific grounding preferences."}
######################
""".rstrip()
        return baseline.rstrip() + "\n\n" + curation + "\n"

    def _validate_prompt_draft(
        self,
        *,
        filename: str,
        baseline: str,
        content: str,
        corpus_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        warnings: List[str] = []
        ok = True
        if not content.strip():
            ok = False
            warnings.append("Prompt draft is empty.")
        if not _safe_prompt_file(filename):
            ok = False
            warnings.append("Prompt filename is unsafe.")
        baseline_placeholders = set(_PLACEHOLDER_RE.findall(baseline))
        draft_placeholders = set(_PLACEHOLDER_RE.findall(content))
        missing = sorted(baseline_placeholders - draft_placeholders)
        if missing:
            ok = False
            warnings.append("Prompt draft removed baseline placeholders: " + ", ".join(missing))
        preflight: Dict[str, Any] = {"attempted": False, "status": "skipped"}
        if filename == "extract_graph.txt":
            preflight = {"attempted": True, "status": "passed"}
            for required in ("{entity_types}", "{input_text}"):
                if required not in draft_placeholders:
                    ok = False
                    preflight["status"] = "failed"
                    warnings.append(f"extract_graph prompt is missing required placeholder {required}.")
            if preflight["status"] != "failed":
                try:
                    format_values = {
                        placeholder[1:-1]: f"<{placeholder[1:-1]}>"
                        for placeholder in draft_placeholders
                    }
                    format_values.update(
                        {
                            "entity_types": ", ".join([str(item) for item in (corpus_profile.get("candidate_entity_types") or [])])
                            or "organization, person, geo, event",
                            "input_text": "Research & Tune extraction preflight sample.",
                        }
                    )
                    content.format(**format_values)
                except Exception as exc:
                    ok = False
                    preflight["status"] = "failed"
                    preflight["detail"] = f"Format preflight failed: {type(exc).__name__}: {exc}"
                    warnings.append(str(preflight["detail"]))
        return {
            "ok": ok,
            "warnings": warnings,
            "baseline_placeholders": sorted(baseline_placeholders),
            "draft_placeholders": sorted(draft_placeholders),
            "missing_placeholders": missing,
            "extract_graph_preflight": preflight,
        }

    def _write_scratchpad(
        self,
        path: Path,
        *,
        record: Any,
        corpus_profile: Dict[str, Any],
        digests: Sequence[Dict[str, Any]],
        prompt_drafts: Dict[str, Any],
        warnings: Sequence[str],
    ) -> None:
        lines = [
            f"# Research & Tune Scratchpad: {record.display_name or record.graph_id}",
            "",
            f"- Graph ID: {record.graph_id}",
            f"- Collection: {record.collection_id}",
            f"- Source documents reviewed: {len(digests)}",
            f"- Generated prompt drafts: {', '.join(sorted(prompt_drafts.keys())) or 'None'}",
            "",
            "## Corpus Profile",
            "",
            str(corpus_profile.get("corpus_summary") or "No corpus summary was available."),
            "",
            "## Prompt Strategy",
            "",
            *[f"- {item}" for item in (corpus_profile.get("prompt_strategy") or [])],
            "",
            "## Warnings",
            "",
            *([f"- {item}" for item in warnings] if warnings else ["- None"]),
            "",
            "## Document Digests",
            "",
        ]
        for digest in digests:
            doc = dict(digest.get("document") or {})
            lines.extend(
                [
                    f"### {doc.get('title') or doc.get('doc_id') or 'Document'}",
                    "",
                    f"- Doc ID: {doc.get('doc_id') or ''}",
                    f"- Source: {doc.get('source_path') or ''}",
                    f"- Candidate entity types: {', '.join(digest.get('candidate_entity_types') or [])}",
                    f"- Candidate relationship types: {', '.join(digest.get('candidate_relationship_types') or [])}",
                    "",
                    str(digest.get("summary") or ""),
                    "",
                    "Important considerations:",
                    *[f"- {item}" for item in (digest.get("important_considerations") or [])],
                    "",
                    "Prompt implications:",
                    *[f"- {item}" for item in (digest.get("prompt_implications") or [])],
                    "",
                ]
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    def _normalize_targets(self, target_prompt_files: Sequence[str] | None) -> tuple[List[str], List[str]]:
        raw_targets = list(target_prompt_files or COMMON_GRAPHRAG_PROMPT_TARGETS)
        targets: List[str] = []
        warnings: List[str] = []
        for item in _dedupe(raw_targets):
            if not _safe_prompt_file(item):
                warnings.append(f"Skipped unsafe prompt target: {item}")
                continue
            targets.append(item)
        if not targets:
            targets = ["extract_graph.txt"]
        return targets, warnings

    def _tuning_dir(self, record: Any, run_id: str) -> Path:
        root = Path(str(getattr(record, "root_path", "") or self.graph_service._graph_root(record.graph_id)))
        return root / "tuning" / str(run_id)

    def _find_run(self, graph_id: str, run_id: str) -> Any | None:
        run_store = self.graph_service._run_store()
        if run_store is None:
            return None
        for run in run_store.list_runs(graph_id, tenant_id=self.tenant_id, limit=100):
            if str(getattr(run, "run_id", "") or "") == str(run_id):
                return run
        return None

    def _write_jsonl(self, path: Path, rows: Sequence[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows


__all__ = ["COMMON_GRAPHRAG_PROMPT_TARGETS", "GraphPromptTuningService"]
