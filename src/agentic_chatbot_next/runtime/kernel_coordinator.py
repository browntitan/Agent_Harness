from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from agentic_chatbot_next.capabilities import coerce_effective_capabilities
from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.jobs import JobRecord
from agentic_chatbot_next.contracts.messages import RuntimeMessage, SessionState
from agentic_chatbot_next.runtime.artifacts import get_handoff_artifact, list_handoff_artifacts, register_handoff_artifact
from agentic_chatbot_next.runtime.clarification import is_clarification_turn
from agentic_chatbot_next.runtime.doc_focus import build_doc_focus_result
from agentic_chatbot_next.runtime.task_plan import (
    TERMINAL_TASK_STATUSES,
    TaskExecutionState,
    TaskResult,
    TaskSpec,
    VerificationResult,
    WorkerExecutionRequest,
    select_execution_batch,
)
from agentic_chatbot_next.runtime.task_decomposition import build_planner_input_packet
from agentic_chatbot_next.runtime.turn_contracts import (
    build_execution_digest,
    filter_context_messages,
    infer_result_provenance,
    resolve_turn_intent,
    resolved_turn_intent_from_metadata,
)
from agentic_chatbot_next.utils.json_utils import extract_json

if TYPE_CHECKING:
    from agentic_chatbot_next.runtime.kernel import RuntimeKernel


_HANDOFF_ALLOWED_CONSUMERS = {
    "analysis_summary": {"general", "rag_worker", "finalizer", "coordinator"},
    "entity_candidates": {"rag_worker", "general", "coordinator"},
    "keyword_windows": {"rag_worker", "general", "coordinator"},
    "title_candidates": {"rag_worker", "general", "finalizer", "coordinator"},
    "doc_focus": {"rag_worker", "general", "finalizer", "coordinator", "data_analyst"},
    "research_facets": {"rag_worker", "general", "finalizer", "coordinator"},
    "facet_matches": {"general", "finalizer", "coordinator"},
    "doc_digest": {"general", "finalizer", "coordinator"},
    "subsystem_inventory": {"rag_worker", "general", "finalizer", "coordinator"},
    "subsystem_evidence": {"general", "finalizer", "coordinator"},
    "research_coverage_ledger": {"general", "finalizer", "verifier", "coordinator"},
    "evidence_request": {"rag_worker", "general", "coordinator"},
    "evidence_response": {"finalizer", "general", "coordinator"},
    "clause_redline_inventory": {"rag_worker", "general", "verifier", "finalizer", "coordinator"},
    "policy_guidance_matches": {"general", "verifier", "finalizer", "coordinator"},
    "policy_coverage_verification": {"general", "finalizer", "coordinator"},
    "buyer_recommendation_table": {"finalizer", "general", "coordinator"},
}

_HANDOFF_SCHEMA_KEYS = {
    "analysis_summary": {"summary"},
    "entity_candidates": {"entities"},
    "keyword_windows": {"keywords", "windows"},
    "title_candidates": {"documents", "query_variants", "scope_collection_id"},
    "doc_focus": {"documents"},
    "research_facets": {"facets", "seed_documents", "scope_collection_id"},
    "facet_matches": {"facet", "documents", "rationale", "supporting_citation_ids"},
    "doc_digest": {"document", "document_summary", "subsystems"},
    "subsystem_inventory": {"subsystems", "source_documents", "scope_collection_id"},
    "subsystem_evidence": {"subsystem", "documents", "supporting_citation_ids"},
    "research_coverage_ledger": {
        "scope",
        "candidate_documents",
        "reviewed_documents",
        "facets",
        "primary_source_count",
        "meta_source_count",
        "coverage_state",
        "warnings",
    },
    "evidence_request": {"query"},
    "evidence_response": {"summary"},
    "clause_redline_inventory": {"clauses"},
    "policy_guidance_matches": {"matches"},
    "policy_coverage_verification": {"verdict"},
    "buyer_recommendation_table": {"recommendations"},
}

_META_DOCUMENT_RE = re.compile(
    r"("
    r"test_queries|rfp_corpus_test_prompts|prompt\s*catalog|query\s*pack|"
    r"/tests?/|\\tests?\\|fixtures?|scenarios?|acceptance|"
    r"(?:^|[/\\])test_[^/\\]+|[^/\\]+_test\."
    r")",
    flags=re.I,
)


class KernelCoordinatorController:
    def __init__(self, kernel: "RuntimeKernel") -> None:
        self.kernel = kernel

    @staticmethod
    def _extract_doc_focus_documents(result: TaskResult) -> List[Dict[str, Any]]:
        payload = dict(result.metadata.get("rag_search_result") or {})
        documents = payload.get("doc_focus") or []
        if isinstance(documents, list) and documents:
            return [dict(item) for item in documents if isinstance(item, dict)]
        candidates = payload.get("candidate_docs") or []
        extracted: List[Dict[str, Any]] = []
        for item in candidates:
            metadata = dict((item or {}).get("metadata") or {})
            doc_id = str(metadata.get("doc_id") or "")
            title = str(metadata.get("title") or "")
            if not doc_id and not title:
                continue
            extracted.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "source_path": str(metadata.get("source_path") or ""),
                    "source_type": str(metadata.get("source_type") or ""),
                }
            )
        return extracted

    @staticmethod
    def _extract_seed_documents(payload: Dict[str, Any]) -> List[Dict[str, str]]:
        documents = []
        for item in (payload.get("seed_documents") or []):
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("doc_id") or "").strip()
            title = str(item.get("title") or "").strip()
            if not doc_id and not title:
                continue
            documents.append({"doc_id": doc_id, "title": title})
        return documents

    @staticmethod
    def _normalize_document_brief(raw: Dict[str, Any]) -> Dict[str, str]:
        return {
            "doc_id": str(raw.get("doc_id") or "").strip(),
            "title": str(raw.get("title") or "").strip(),
            "source_path": str(raw.get("source_path") or "").strip(),
            "source_type": str(raw.get("source_type") or "").strip(),
        }

    @staticmethod
    def _is_meta_document(raw: Dict[str, Any]) -> bool:
        doc = KernelCoordinatorController._normalize_document_brief(raw)
        searchable = " ".join(
            str(doc.get(field) or "")
            for field in ("doc_id", "title", "source_path", "source_type")
        )
        return bool(_META_DOCUMENT_RE.search(searchable))

    @staticmethod
    def _document_identity_key(raw: Dict[str, Any]) -> tuple[str, str]:
        doc = KernelCoordinatorController._normalize_document_brief(raw)
        doc_id = str(doc.get("doc_id") or "").strip()
        if doc_id:
            return (doc_id, "")
        return ("", str(doc.get("title") or "").strip().casefold())

    @staticmethod
    def _normalize_string_list(raw_items: Any) -> List[str]:
        seen: set[str] = set()
        values: List[str] = []
        for item in raw_items or []:
            value = str(item).strip()
            if not value:
                continue
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            values.append(value)
        return values

    @staticmethod
    def _reviewed_relevance_rank(value: str) -> int:
        normalized = str(value or "").strip().lower()
        if normalized == "relevant":
            return 3
        if normalized == "partial":
            return 2
        if normalized == "irrelevant":
            return 0
        return 1

    def _ranked_document_rows(
        self,
        artifacts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rows: Dict[tuple[str, str], Dict[str, Any]] = {}

        def _ensure_row(raw: Dict[str, Any]) -> Dict[str, Any] | None:
            doc = self._normalize_document_brief(raw)
            key = self._document_identity_key(doc)
            if key == ("", ""):
                return None
            existing = rows.get(key)
            if existing is None:
                existing = {
                    **doc,
                    "is_meta_document": self._is_meta_document(doc),
                    "match_reason": "",
                    "title_path_score": 0.0,
                    "seed_hits": 0,
                    "strong_evidence_count": 0,
                    "review_preferred": False,
                    "reviewed": False,
                    "reviewed_relevance": "",
                    "relevance_rationale": "",
                    "coverage": "",
                    "matched_facets": set(),
                    "used_citation_ids": set(),
                }
                rows[key] = existing
            else:
                for field in ("doc_id", "title", "source_path", "source_type"):
                    if not existing.get(field) and doc.get(field):
                        existing[field] = doc[field]
                existing["is_meta_document"] = bool(existing.get("is_meta_document")) or self._is_meta_document(doc)
            return existing

        for artifact in artifacts:
            artifact_type = str(artifact.get("artifact_type") or "").strip()
            payload = dict(artifact.get("data") or {})
            if artifact_type == "title_candidates":
                for item in payload.get("documents") or []:
                    if not isinstance(item, dict):
                        continue
                    row = _ensure_row(item)
                    if row is None:
                        continue
                    try:
                        score = float(item.get("score") or 0.0)
                    except (TypeError, ValueError):
                        score = 0.0
                    row["title_path_score"] = max(float(row.get("title_path_score") or 0.0), score)
                    if not row.get("match_reason"):
                        row["match_reason"] = str(item.get("match_reason") or "").strip()
            elif artifact_type == "doc_focus":
                for item in payload.get("documents") or []:
                    if not isinstance(item, dict):
                        continue
                    row = _ensure_row(item)
                    if row is None:
                        continue
                    row["seed_hits"] += 1
                    row["strong_evidence_count"] += 1
            elif artifact_type == "research_facets":
                for item in payload.get("seed_documents") or []:
                    if not isinstance(item, dict):
                        continue
                    row = _ensure_row(item)
                    if row is None:
                        continue
                    row["seed_hits"] += 1
                for item in payload.get("review_documents") or []:
                    if not isinstance(item, dict):
                        continue
                    row = _ensure_row(item)
                    if row is None:
                        continue
                    row["review_preferred"] = True
                for facet in payload.get("facets") or []:
                    if not isinstance(facet, dict):
                        continue
                    facet_name = str(facet.get("name") or "").strip()
                    if not facet_name:
                        continue
                    for doc_id in facet.get("seed_doc_ids") or []:
                        row = _ensure_row({"doc_id": str(doc_id).strip(), "title": str(doc_id).strip()})
                        if row is None:
                            continue
                        row["matched_facets"].add(facet_name)
                        row["seed_hits"] += 1
            elif artifact_type == "facet_matches":
                facet_name = str(payload.get("facet") or "").strip()
                evidence_count = max(
                    1,
                    len(
                        [
                            str(item).strip()
                            for item in (payload.get("supporting_citation_ids") or [])
                            if str(item).strip()
                        ]
                    ),
                )
                for item in payload.get("documents") or []:
                    if not isinstance(item, dict):
                        continue
                    row = _ensure_row(item)
                    if row is None:
                        continue
                    if facet_name:
                        row["matched_facets"].add(facet_name)
                    row["strong_evidence_count"] += evidence_count
            elif artifact_type == "doc_digest":
                document = payload.get("document")
                if not isinstance(document, dict):
                    continue
                row = _ensure_row(document)
                if row is None:
                    continue
                relevance = str(payload.get("relevance") or "").strip().lower()
                if relevance and (
                    not str(row.get("reviewed_relevance") or "").strip()
                    or self._reviewed_relevance_rank(relevance)
                    >= self._reviewed_relevance_rank(str(row.get("reviewed_relevance") or ""))
                ):
                    row["reviewed_relevance"] = relevance
                rationale = str(payload.get("relevance_rationale") or "").strip()
                if rationale and not row.get("relevance_rationale"):
                    row["relevance_rationale"] = rationale
                coverage = str(payload.get("coverage") or "").strip().lower()
                if coverage:
                    row["coverage"] = coverage
                row["reviewed"] = True
                row["matched_facets"].update(self._normalize_string_list(payload.get("matched_facets") or []))
                row["used_citation_ids"].update(
                    self._normalize_string_list(payload.get("used_citation_ids") or [])
                )
                row["strong_evidence_count"] += len(
                    self._normalize_string_list(payload.get("used_citation_ids") or [])
                )

        ranked_rows: List[Dict[str, Any]] = []
        for row in rows.values():
            ranked_rows.append(
                {
                    **row,
                    "is_meta_document": bool(row.get("is_meta_document")),
                    "matched_facets": sorted(str(item) for item in row.get("matched_facets") or [] if str(item).strip()),
                    "used_citation_ids": sorted(
                        str(item) for item in row.get("used_citation_ids") or [] if str(item).strip()
                    ),
                }
            )
        ranked_rows.sort(
            key=lambda row: (
                0 if bool(row.get("is_meta_document")) else 1,
                self._reviewed_relevance_rank(str(row.get("reviewed_relevance") or "")),
                len(row.get("matched_facets") or []),
                int(row.get("strong_evidence_count") or 0),
                float(row.get("title_path_score") or 0.0),
                1 if row.get("review_preferred") else 0,
                int(row.get("seed_hits") or 0),
                str(row.get("title") or row.get("doc_id") or "").casefold(),
            ),
            reverse=True,
        )
        return ranked_rows

    @staticmethod
    def _select_ranked_review_rows(
        rows: List[Dict[str, Any]],
        *,
        primary_limit: int,
        optional_limit: int,
    ) -> List[Dict[str, Any]]:
        if not rows:
            return []
        non_meta_rows = [row for row in rows if not bool(row.get("is_meta_document"))]
        review_rows = non_meta_rows or rows
        selected: List[Dict[str, Any]] = list(review_rows[: max(0, primary_limit)])
        covered_facets: set[str] = {
            str(item).strip()
            for row in selected
            for item in (row.get("matched_facets") or [])
            if str(item).strip()
        }
        if optional_limit <= 0:
            return selected
        for row in review_rows[max(0, primary_limit) :]:
            if len(selected) >= primary_limit + optional_limit:
                break
            row_facets = {
                str(item).strip()
                for item in (row.get("matched_facets") or [])
                if str(item).strip()
            }
            if not row_facets.difference(covered_facets):
                continue
            selected.append(row)
            covered_facets.update(row_facets)
        return selected

    def _documents_for_synthesis(
        self,
        artifacts: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        rows = self._ranked_document_rows(artifacts)
        if not rows:
            return []
        has_reviewed = any(bool(row.get("reviewed")) for row in rows)
        documents: List[Dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        covered_facets: set[str] = set()
        has_primary_documents = any(not bool(row.get("is_meta_document")) for row in rows)
        for row in rows:
            if has_primary_documents and bool(row.get("is_meta_document")):
                continue
            relevance = str(row.get("reviewed_relevance") or "").strip().lower()
            if relevance == "irrelevant":
                continue
            row_facets = {
                str(item).strip()
                for item in (row.get("matched_facets") or [])
                if str(item).strip()
            }
            if has_reviewed and relevance == "partial" and row_facets and not row_facets.difference(covered_facets):
                continue
            if has_reviewed and relevance == "partial" and not row_facets:
                continue
            doc = self._normalize_document_brief(row)
            key = self._document_identity_key(doc)
            if key == ("", "") or key in seen:
                continue
            seen.add(key)
            documents.append(doc)
            covered_facets.update(row_facets)
        if documents or not has_reviewed:
            return documents
        return [self._normalize_document_brief(row) for row in rows if str(row.get("reviewed_relevance") or "").strip().lower() != "irrelevant"]

    @staticmethod
    def _ordered_documents_from_rendered_answer(
        rendered_answer: str,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        answer = str(rendered_answer or "").strip()
        if not answer:
            return []
        lowered = answer.casefold()
        ordered: List[tuple[int, int, Dict[str, str]]] = []
        for index, raw in enumerate(documents):
            doc = KernelCoordinatorController._normalize_document_brief(raw)
            tokens = [
                str(doc.get("doc_id") or "").strip(),
                str(doc.get("title") or "").strip(),
            ]
            positions = [lowered.find(token.casefold()) for token in tokens if token and lowered.find(token.casefold()) >= 0]
            if not positions:
                continue
            ordered.append((min(positions), index, doc))
        ordered.sort(key=lambda item: (item[0], item[1]))
        return [doc for _, _, doc in ordered]

    @staticmethod
    def _normalize_subsystem_items(raw_items: Any) -> List[Dict[str, Any]]:
        subsystems: List[Dict[str, Any]] = []
        for raw in (raw_items or []):
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            if not name:
                continue
            subsystems.append(
                {
                    "name": name,
                    "aliases": [str(item).strip() for item in (raw.get("aliases") or []) if str(item).strip()],
                    "description": str(raw.get("description") or "").strip(),
                    "responsibilities": [str(item).strip() for item in (raw.get("responsibilities") or []) if str(item).strip()],
                    "interfaces": [str(item).strip() for item in (raw.get("interfaces") or []) if str(item).strip()],
                    "supporting_citation_ids": [
                        str(item).strip()
                        for item in (raw.get("supporting_citation_ids") or [])
                        if str(item).strip()
                    ],
                }
            )
        return subsystems

    @staticmethod
    def _current_task_ids(execution_state: TaskExecutionState) -> set[str]:
        return {
            str(item.get("id") or "")
            for item in (execution_state.task_plan or [])
            if str(item.get("id") or "")
        }

    def _current_workflow_handoff_artifacts(
        self,
        *,
        session_state: SessionState,
        execution_state: TaskExecutionState,
        artifact_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        current_task_ids = self._current_task_ids(execution_state)
        artifacts = list_handoff_artifacts(session_state, artifact_types=artifact_types)
        if not current_task_ids:
            return artifacts
        return [
            artifact
            for artifact in artifacts
            if str(artifact.get("producer_task_id") or "") in current_task_ids
        ]

    def _collect_documents_from_handoffs(
        self,
        artifacts: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        documents: List[Dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def _add_document(raw: Dict[str, Any]) -> None:
            doc = self._normalize_document_brief(raw)
            key = self._document_identity_key(doc)
            if key == ("", "") or key in seen:
                return
            seen.add(key)
            documents.append(doc)

        for artifact in artifacts:
            payload = dict(artifact.get("data") or {})
            for doc in payload.get("documents") or []:
                if isinstance(doc, dict):
                    _add_document(doc)
            for doc in payload.get("review_documents") or []:
                if isinstance(doc, dict):
                    _add_document(doc)
            document = payload.get("document")
            if isinstance(document, dict):
                _add_document(document)
            for doc in payload.get("source_documents") or []:
                if isinstance(doc, dict):
                    _add_document(doc)
            for subsystem in payload.get("subsystems") or []:
                if not isinstance(subsystem, dict):
                    continue
                for doc in subsystem.get("supporting_documents") or []:
                    if isinstance(doc, dict):
                        _add_document(doc)
        return documents

    def _latest_current_artifact_payload(
        self,
        *,
        session_state: SessionState,
        execution_state: TaskExecutionState,
        artifact_type: str,
    ) -> Dict[str, Any]:
        artifacts = self._current_workflow_handoff_artifacts(
            session_state=session_state,
            execution_state=execution_state,
            artifact_types=[artifact_type],
        )
        if not artifacts:
            return {}
        return dict(artifacts[-1].get("data") or {})

    def _current_subsystem_evidence_payloads(
        self,
        *,
        session_state: SessionState,
        execution_state: TaskExecutionState,
    ) -> List[Dict[str, Any]]:
        artifacts = self._current_workflow_handoff_artifacts(
            session_state=session_state,
            execution_state=execution_state,
            artifact_types=["subsystem_evidence"],
        )
        return [dict(artifact.get("data") or {}) for artifact in artifacts]

    @staticmethod
    def _coverage_doc_payload(row: Dict[str, Any]) -> Dict[str, Any]:
        doc = KernelCoordinatorController._normalize_document_brief(row)
        return {
            **doc,
            "is_meta_document": bool(row.get("is_meta_document")),
            "reviewed": bool(row.get("reviewed")),
            "reviewed_relevance": str(row.get("reviewed_relevance") or "").strip(),
            "matched_facets": [
                str(item).strip()
                for item in (row.get("matched_facets") or [])
                if str(item).strip()
            ],
            "coverage": str(row.get("coverage") or "").strip(),
            "used_citation_ids": [
                str(item).strip()
                for item in (row.get("used_citation_ids") or [])
                if str(item).strip()
            ],
        }

    def _build_research_coverage_ledger(
        self,
        *,
        session_state: SessionState,
        execution_state: TaskExecutionState,
    ) -> Dict[str, Any]:
        artifacts = self._current_workflow_handoff_artifacts(
            session_state=session_state,
            execution_state=execution_state,
            artifact_types=[
                "title_candidates",
                "doc_focus",
                "research_facets",
                "facet_matches",
                "doc_digest",
                "subsystem_inventory",
                "subsystem_evidence",
            ],
        )
        rows = self._ranked_document_rows(artifacts)
        reviewed_rows = [
            row
            for row in rows
            if bool(row.get("reviewed"))
            or str(row.get("reviewed_relevance") or "").strip()
            or [item for item in (row.get("used_citation_ids") or []) if str(item).strip()]
        ]
        source_rows = reviewed_rows or rows
        primary_documents = [
            self._coverage_doc_payload(row)
            for row in source_rows
            if not bool(row.get("is_meta_document"))
            and str(row.get("reviewed_relevance") or "").strip().lower() != "irrelevant"
        ]
        meta_documents = [
            self._coverage_doc_payload(row)
            for row in source_rows
            if bool(row.get("is_meta_document"))
            and str(row.get("reviewed_relevance") or "").strip().lower() != "irrelevant"
        ]

        facets: List[Dict[str, Any]] = []
        seen_facets: set[str] = set()
        for artifact in artifacts:
            artifact_type = str(artifact.get("artifact_type") or "")
            payload = dict(artifact.get("data") or {})
            if artifact_type == "research_facets":
                for item in payload.get("facets") or []:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or "").strip()
                    if not name or name.casefold() in seen_facets:
                        continue
                    seen_facets.add(name.casefold())
                    facets.append(
                        {
                            "name": name,
                            "aliases": [
                                str(alias).strip()
                                for alias in (item.get("aliases") or [])
                                if str(alias).strip()
                            ],
                            "source": "research_facets",
                        }
                    )
            elif artifact_type == "facet_matches":
                name = str(payload.get("facet") or "").strip()
                if not name or name.casefold() in seen_facets:
                    continue
                seen_facets.add(name.casefold())
                facets.append(
                    {
                        "name": name,
                        "aliases": [],
                        "source": "facet_matches",
                        "supporting_citation_ids": [
                            str(item).strip()
                            for item in (payload.get("supporting_citation_ids") or [])
                            if str(item).strip()
                        ],
                    }
                )

        source_documents = self._latest_current_artifact_payload(
            session_state=session_state,
            execution_state=execution_state,
            artifact_type="subsystem_inventory",
        ).get("source_documents") or []
        for item in source_documents:
            if not isinstance(item, dict):
                continue
            doc = self._normalize_document_brief(item)
            if self._document_identity_key(doc) == ("", ""):
                continue
            if self._is_meta_document(doc):
                if not any(self._document_identity_key(doc) == self._document_identity_key(existing) for existing in meta_documents):
                    meta_documents.append({**doc, "is_meta_document": True, "reviewed": False, "matched_facets": []})
            elif not any(self._document_identity_key(doc) == self._document_identity_key(existing) for existing in primary_documents):
                primary_documents.append({**doc, "is_meta_document": False, "reviewed": False, "matched_facets": []})

        warnings: List[str] = []
        if meta_documents:
            warnings.append(
                "Meta or prompt/test documents were found during discovery and were not counted as primary architecture evidence."
            )
        if len(primary_documents) < 3:
            warnings.append("Primary source coverage is thin for a holistic repository architecture request.")
        if len(facets) < 2:
            warnings.append("Facet coverage is thin; the research may not cover all major subsystem surfaces.")
        coverage_state = "strong"
        if warnings:
            coverage_state = "thin"
        if len(primary_documents) < 2 or len(facets) < 1:
            coverage_state = "insufficient"

        collection_id = str(
            session_state.metadata.get("kb_collection_id")
            or session_state.metadata.get("collection_id")
            or "default"
        ).strip() or "default"
        return {
            "scope": {
                "collection_id": collection_id,
                "source_query": execution_state.user_request,
                "coverage_profile": str(
                    (
                        resolved_turn_intent_from_metadata(dict(session_state.metadata or {}))
                        or resolve_turn_intent(execution_state.user_request, dict(session_state.metadata or {}))
                    ).answer_contract.coverage_profile
                    or ""
                ),
            },
            "candidate_documents": [self._coverage_doc_payload(row) for row in rows],
            "reviewed_documents": [self._coverage_doc_payload(row) for row in reviewed_rows],
            "facets": facets,
            "primary_source_count": len(primary_documents),
            "meta_source_count": len(meta_documents),
            "coverage_state": coverage_state,
            "warnings": warnings,
        }

    def _coverage_gate_issues(
        self,
        *,
        session_state: SessionState,
        execution_state: TaskExecutionState,
        ledger: Dict[str, Any],
    ) -> List[str]:
        intent = resolved_turn_intent_from_metadata(dict(session_state.metadata or {})) or resolve_turn_intent(
            execution_state.user_request,
            dict(session_state.metadata or {}),
        )
        if str(getattr(intent.answer_contract, "coverage_profile", "") or "").strip() != "holistic_repository":
            return []
        issues: List[str] = []
        primary_source_count = int(ledger.get("primary_source_count") or 0)
        meta_source_count = int(ledger.get("meta_source_count") or 0)
        facet_count = len([item for item in (ledger.get("facets") or []) if isinstance(item, dict)])
        reviewed_primary_count = len(
            [
                item
                for item in (ledger.get("reviewed_documents") or [])
                if isinstance(item, dict) and not bool(item.get("is_meta_document"))
            ]
        )
        if primary_source_count < 3:
            issues.append("Holistic repository research needs at least three primary source documents before final synthesis.")
        if reviewed_primary_count < 2:
            issues.append("Holistic repository research needs multiple reviewed primary documents, not only retrieval hits.")
        if facet_count < 2:
            issues.append("Holistic repository research needs multiple subsystem or architecture facets.")
        if meta_source_count and primary_source_count < 4:
            issues.append("Meta/test prompt documents were discovered and must not substitute for primary architecture evidence.")
        return issues

    def _register_research_coverage_ledger(
        self,
        *,
        session_state: SessionState,
        execution_state: TaskExecutionState,
    ) -> Dict[str, Any]:
        ledger = self._build_research_coverage_ledger(
            session_state=session_state,
            execution_state=execution_state,
        )
        producer_task_id = ""
        for task in reversed(execution_state.task_plan):
            producer_task_id = str(task.get("id") or "").strip()
            if producer_task_id:
                break
        if producer_task_id:
            artifact = register_handoff_artifact(
                session_state,
                artifact_type="research_coverage_ledger",
                handoff_schema="research_coverage_ledger",
                producer_task_id=producer_task_id,
                producer_agent="coordinator",
                data=ledger,
                summary=f"Coverage {ledger.get('coverage_state')}",
                allowed_consumers=sorted(_HANDOFF_ALLOWED_CONSUMERS["research_coverage_ledger"]),
            )
            ledger["artifact_id"] = str(artifact.get("artifact_id") or "")
        self.kernel._emit(
            "research_coverage_ledger_updated",
            session_state.session_id,
            agent_name="coordinator",
            payload={
                "conversation_id": session_state.conversation_id,
                "coverage_state": ledger.get("coverage_state"),
                "primary_source_count": ledger.get("primary_source_count"),
                "meta_source_count": ledger.get("meta_source_count"),
                "facet_count": len(ledger.get("facets") or []),
            },
        )
        meta_documents = [
            item
            for item in (ledger.get("candidate_documents") or [])
            if isinstance(item, dict) and bool(item.get("is_meta_document"))
        ]
        if meta_documents:
            self.kernel._emit(
                "meta_document_demoted",
                session_state.session_id,
                agent_name="coordinator",
                payload={
                    "conversation_id": session_state.conversation_id,
                    "documents": [
                        {
                            "doc_id": str(item.get("doc_id") or ""),
                            "title": str(item.get("title") or ""),
                            "source_path": str(item.get("source_path") or ""),
                        }
                        for item in meta_documents[:10]
                    ],
                },
            )
        return ledger

    def _needs_detailed_summary_repair(
        self,
        final_text: str,
        *,
        inventory_payload: Dict[str, Any],
    ) -> bool:
        answer = str(final_text or "").strip()
        if not answer:
            return True
        subsystems = [dict(item) for item in (inventory_payload.get("subsystems") or []) if isinstance(item, dict)]
        if not subsystems:
            return False
        lowered = answer.casefold()
        if lowered.startswith("documents with grounded evidence relevant to the request"):
            return True
        mentioned = 0
        for item in subsystems:
            name = str(item.get("name") or "").strip()
            if name and name.casefold() in lowered:
                mentioned += 1
        minimum_mentions = min(2, len(subsystems))
        if mentioned < minimum_mentions:
            return True
        if len(answer.split()) < 140:
            return True
        return False

    def _render_detailed_subsystem_fallback(
        self,
        *,
        session_state: SessionState,
        execution_state: TaskExecutionState,
    ) -> str:
        inventory_payload = self._latest_current_artifact_payload(
            session_state=session_state,
            execution_state=execution_state,
            artifact_type="subsystem_inventory",
        )
        subsystems = [dict(item) for item in (inventory_payload.get("subsystems") or []) if isinstance(item, dict)]
        if not subsystems:
            return str(execution_state.partial_answer or "").strip()

        collection_id = str(
            inventory_payload.get("scope_collection_id")
            or session_state.metadata.get("kb_collection_id")
            or session_state.metadata.get("collection_id")
            or "default"
        ).strip() or "default"
        source_documents = [
            self._normalize_document_brief(item)
            for item in (inventory_payload.get("source_documents") or [])
            if isinstance(item, dict)
        ]
        unique_source_titles: List[str] = []
        seen_titles: set[str] = set()
        for item in source_documents:
            title = str(item.get("title") or item.get("doc_id") or "").strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_source_titles.append(title)

        evidence_payloads = self._current_subsystem_evidence_payloads(
            session_state=session_state,
            execution_state=execution_state,
        )
        evidence_by_name = {
            str(item.get("subsystem") or "").strip().casefold(): dict(item)
            for item in evidence_payloads
            if str(item.get("subsystem") or "").strip()
        }

        lines: List[str] = []
        lines.append("## Overall Architecture")
        overview_names = [str(item.get("name") or "").strip() for item in subsystems if str(item.get("name") or "").strip()]
        if overview_names:
            lines.append(
                "The scoped candidate documents in the "
                f"`{collection_id}` collection describe "
                f"{len(overview_names)} major subsystem areas: {', '.join(overview_names)}."
            )
        if unique_source_titles:
            lines.append("Primary supporting documents: " + ", ".join(unique_source_titles) + ".")

        lines.append("## Subsystem Breakdown")
        thin_items: List[str] = []
        for item in subsystems:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            description = str(item.get("description") or "").strip()
            responsibilities = [str(value).strip() for value in (item.get("responsibilities") or []) if str(value).strip()]
            interfaces = [str(value).strip() for value in (item.get("interfaces") or []) if str(value).strip()]
            supporting_documents = [
                str(doc.get("title") or doc.get("doc_id") or "").strip()
                for doc in (item.get("supporting_documents") or [])
                if isinstance(doc, dict) and str(doc.get("title") or doc.get("doc_id") or "").strip()
            ]
            evidence_summary = str(
                evidence_by_name.get(name.casefold(), {}).get("summary") or ""
            ).strip()
            coverage = str(item.get("coverage") or "").strip().lower()
            if coverage in {"thin", "weak", "sparse"}:
                thin_items.append(name)

            lines.append(f"### {name}")
            if description:
                lines.append(description)
            if responsibilities:
                lines.append("Responsibilities: " + "; ".join(responsibilities) + ".")
            if interfaces:
                lines.append("Interfaces and touchpoints: " + "; ".join(interfaces) + ".")
            if supporting_documents:
                lines.append("Supported by: " + ", ".join(supporting_documents) + ".")
            if evidence_summary:
                lines.append("Additional grounded evidence: " + evidence_summary)

        lines.append("## Cross-Cutting Systems")
        cross_cutting = [
            str(item.get("name") or "").strip()
            for item in subsystems
            if re.search(r"\b(router|routing|observability|memory|tool|job|gateway)\b", str(item.get("name") or ""), flags=re.I)
        ]
        if cross_cutting:
            lines.append(
                "The cross-cutting subsystems called out across the scoped documents include "
                + ", ".join(cross_cutting)
                + "."
            )
        else:
            lines.append(
                "The scoped documents primarily emphasize execution flow, service orchestration, and component boundaries rather than a separate cross-cutting subsystem taxonomy."
            )

        lines.append("## Thin Or Conflicting Evidence")
        if thin_items:
            lines.append(
                "The following subsystems have thinner support and were supplemented with targeted backfill: "
                + ", ".join(thin_items)
                + "."
            )
        else:
            lines.append("No major subsystem category was flagged as thin after consolidation.")

        if unique_source_titles:
            lines.append("Sources: " + ", ".join(unique_source_titles))
        return "\n\n".join(line for line in lines if str(line).strip()).strip()

    @staticmethod
    def _extract_keywords(text: str, *, limit: int = 8) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9_]{4,}", str(text or ""))
        seen: set[str] = set()
        keywords: List[str] = []
        for token in tokens:
            lowered = token.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            keywords.append(token)
            if len(keywords) >= limit:
                break
        return keywords

    def _handoff_payload(
        self,
        *,
        artifact_type: str,
        task: Dict[str, Any],
        result: TaskResult,
    ) -> Dict[str, Any]:
        output = str(result.output or "").strip()
        doc_scope = [str(item) for item in (task.get("doc_scope") or []) if str(item)]
        controller_hints = dict(task.get("controller_hints") or {})
        keywords = self._extract_keywords(output or str(task.get("input") or ""))
        if artifact_type == "title_candidates":
            payload = extract_json(output or "") or {}
            documents: List[Dict[str, Any]] = []
            for item in (payload.get("documents") or []):
                if not isinstance(item, dict):
                    continue
                normalized = self._normalize_document_brief(item)
                if not normalized.get("doc_id") and not normalized.get("title"):
                    continue
                try:
                    score = round(float(item.get("score") or 0.0), 4)
                except (TypeError, ValueError):
                    score = 0.0
                documents.append(
                    {
                        **normalized,
                        "match_reason": str(item.get("match_reason") or "").strip(),
                        "score": score,
                    }
                )
            if not documents:
                documents = [
                    {
                        "doc_id": item,
                        "title": item,
                        "source_path": "",
                        "source_type": "",
                        "match_reason": "task_doc_scope",
                        "score": 0.0,
                    }
                    for item in doc_scope
                ]
            query_variants = self._normalize_string_list(payload.get("query_variants") or [])
            if not query_variants and str(task.get("input") or "").strip():
                query_variants = [str(task.get("input") or "").strip()]
            return {
                "documents": documents,
                "query_variants": query_variants[:8],
                "scope_collection_id": str(
                    payload.get("scope_collection_id")
                    or controller_hints.get("requested_kb_collection_id")
                    or controller_hints.get("kb_collection_id")
                    or "default"
                ).strip()
                or "default",
                "summary": output[:1200],
            }
        if artifact_type == "analysis_summary":
            return {
                "summary": output[:4000],
                "highlights": [line.strip("- ").strip() for line in output.splitlines() if line.strip()][:8],
                "keywords": keywords[:6],
            }
        if artifact_type == "entity_candidates":
            entities = re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", output)
            deduped: List[str] = []
            seen: set[str] = set()
            for entity in entities:
                if entity in seen:
                    continue
                seen.add(entity)
                deduped.append(entity)
                if len(deduped) >= 12:
                    break
            return {"entities": deduped or keywords[:8], "summary": output[:1200]}
        if artifact_type == "keyword_windows":
            return {
                "keywords": keywords[:8],
                "windows": [{"keyword": keyword, "excerpt": output[:600]} for keyword in keywords[:4]],
            }
        if artifact_type == "doc_focus":
            documents = self._extract_doc_focus_documents(result)
            if not documents:
                documents = [{"doc_id": item, "title": item, "source_path": "", "source_type": ""} for item in doc_scope]
            return {
                "documents": documents,
                "summary": output[:1200],
            }
        if artifact_type == "doc_digest":
            payload = extract_json(output or "") or {}
            document = dict(payload.get("document") or {})
            if not document and doc_scope:
                first_scope = str(doc_scope[0] or "").strip()
                document = {"doc_id": first_scope, "title": first_scope}
            return {
                "document": self._normalize_document_brief(document),
                "document_summary": str(payload.get("document_summary") or "").strip(),
                "subsystems": self._normalize_subsystem_items(payload.get("subsystems") or []),
                "responsibilities": [str(item).strip() for item in (payload.get("responsibilities") or []) if str(item).strip()],
                "interfaces": [str(item).strip() for item in (payload.get("interfaces") or []) if str(item).strip()],
                "used_citation_ids": [str(item).strip() for item in (payload.get("used_citation_ids") or []) if str(item).strip()],
                "relevance": str(payload.get("relevance") or "").strip().lower(),
                "relevance_rationale": str(payload.get("relevance_rationale") or "").strip(),
                "matched_facets": self._normalize_string_list(payload.get("matched_facets") or []),
                "coverage": str(payload.get("coverage") or "").strip().lower(),
            }
        if artifact_type == "research_facets":
            payload = extract_json(output or "") or {}
            scope_collection_id = str(
                payload.get("scope_collection_id")
                or controller_hints.get("requested_kb_collection_id")
                or controller_hints.get("kb_collection_id")
                or "default"
            ).strip()
            facets: List[Dict[str, Any]] = []
            for item in (payload.get("facets") or []):
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                if not name:
                    continue
                facets.append(
                    {
                        "name": name,
                        "aliases": [str(alias) for alias in (item.get("aliases") or []) if str(alias).strip()],
                        "rationale": str(item.get("rationale") or "").strip(),
                        "seed_doc_ids": [str(doc_id) for doc_id in (item.get("seed_doc_ids") or []) if str(doc_id).strip()],
                    }
                )
            seed_documents = self._extract_seed_documents(payload)
            if not seed_documents:
                for item in self._extract_doc_focus_documents(result):
                    seed_doc_id = str(item.get("doc_id") or "").strip()
                    seed_title = str(item.get("title") or "").strip()
                    if seed_doc_id or seed_title:
                        seed_documents.append({"doc_id": seed_doc_id, "title": seed_title})
            return {
                "facets": facets,
                "seed_documents": seed_documents,
                "review_documents": [
                    self._normalize_document_brief(item)
                    for item in (payload.get("review_documents") or [])
                    if isinstance(item, dict)
                ],
                "unresolved_questions": [
                    str(item).strip()
                    for item in (payload.get("unresolved_questions") or [])
                    if str(item).strip()
                ],
                "scope_collection_id": scope_collection_id or "default",
                "summary": output[:1200],
            }
        if artifact_type == "subsystem_inventory":
            payload = extract_json(output or "") or {}
            source_documents = [
                self._normalize_document_brief(item)
                for item in (payload.get("source_documents") or controller_hints.get("active_doc_focus_documents") or [])
                if isinstance(item, dict)
            ]
            subsystems: List[Dict[str, Any]] = []
            for item in (payload.get("subsystems") or []):
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                if not name:
                    continue
                supporting_documents = [
                    self._normalize_document_brief(doc)
                    for doc in (item.get("supporting_documents") or [])
                    if isinstance(doc, dict)
                ]
                subsystems.append(
                    {
                        "name": name,
                        "aliases": [str(alias).strip() for alias in (item.get("aliases") or []) if str(alias).strip()],
                        "description": str(item.get("description") or "").strip(),
                        "responsibilities": [str(value).strip() for value in (item.get("responsibilities") or []) if str(value).strip()],
                        "interfaces": [str(value).strip() for value in (item.get("interfaces") or []) if str(value).strip()],
                        "supporting_documents": supporting_documents,
                        "supporting_citation_ids": [
                            str(value).strip()
                            for value in (item.get("supporting_citation_ids") or [])
                            if str(value).strip()
                        ],
                        "coverage": str(item.get("coverage") or "").strip().lower() or "thin",
                    }
                )
            return {
                "subsystems": subsystems,
                "source_documents": source_documents,
                "scope_collection_id": str(
                    payload.get("scope_collection_id")
                    or controller_hints.get("requested_kb_collection_id")
                    or controller_hints.get("kb_collection_id")
                    or "default"
                ).strip()
                or "default",
            }
        if artifact_type == "facet_matches":
            search_payload = dict(result.metadata.get("rag_search_result") or {})
            documents = self._extract_doc_focus_documents(result)
            evidence_entries = [dict(item) for item in (search_payload.get("evidence_entries") or []) if isinstance(item, dict)]
            facet_name = str(controller_hints.get("facet_name") or task.get("title") or "").strip()
            return {
                "facet": facet_name,
                "documents": documents,
                "rationale": str(
                    controller_hints.get("facet_rationale")
                    or f"Documents retrieved for facet '{facet_name}'"
                ).strip(),
                "supporting_citation_ids": [
                    str(item.get("chunk_id") or "").strip()
                    for item in evidence_entries[:8]
                    if str(item.get("chunk_id") or "").strip()
                ],
            }
        if artifact_type == "subsystem_evidence":
            search_payload = dict(result.metadata.get("rag_search_result") or {})
            documents = self._extract_doc_focus_documents(result)
            evidence_entries = [dict(item) for item in (search_payload.get("evidence_entries") or []) if isinstance(item, dict)]
            subsystem_name = str(controller_hints.get("subsystem_name") or task.get("title") or "").strip()
            return {
                "subsystem": subsystem_name,
                "documents": documents,
                "rationale": str(
                    controller_hints.get("subsystem_rationale")
                    or f"Additional evidence gathered for subsystem '{subsystem_name}'"
                ).strip(),
                "supporting_citation_ids": [
                    str(item.get("chunk_id") or "").strip()
                    for item in evidence_entries[:10]
                    if str(item.get("chunk_id") or "").strip()
                ],
                "summary": output[:1600],
            }
        if artifact_type == "evidence_request":
            return {
                "query": str(task.get("input") or task.get("title") or ""),
                "doc_scope": doc_scope,
                "summary": output[:1200],
            }
        if artifact_type == "evidence_response":
            return {
                "summary": output[:4000],
                "doc_scope": doc_scope,
                "artifact_ref": result.artifact_ref,
            }
        if artifact_type == "clause_redline_inventory":
            payload = extract_json(output or "") or {}
            clauses: List[Dict[str, Any]] = []
            for index, item in enumerate(payload.get("clauses") or [], start=1):
                if not isinstance(item, dict):
                    continue
                clause_text = str(item.get("clause_text") or "").strip()
                redline_text = str(item.get("redline_text") or "").strip()
                if not clause_text and not redline_text:
                    continue
                clauses.append(
                    {
                        "clause_id": str(item.get("clause_id") or f"clause_{index}").strip(),
                        "clause_text": clause_text,
                        "redline_text": redline_text,
                        "redline_type": str(item.get("redline_type") or "unknown").strip(),
                        "source_doc_id": str(item.get("source_doc_id") or "").strip(),
                        "location": str(item.get("location") or "").strip(),
                        "confidence": item.get("confidence", 0.0),
                    }
                )
            return {
                "clauses": clauses,
                "warnings": [str(item).strip() for item in (payload.get("warnings") or []) if str(item).strip()],
                "summary": output[:1600],
            }
        if artifact_type == "policy_guidance_matches":
            payload = extract_json(output or "") or {}
            matches = payload.get("matches") or payload.get("policy_guidance_matches") or []
            if not isinstance(matches, list):
                matches = []
            return {
                "matches": [dict(item) for item in matches if isinstance(item, dict)],
                "summary": output[:2400],
                "doc_scope": doc_scope,
            }
        if artifact_type == "policy_coverage_verification":
            payload = extract_json(output or "") or {}
            verdict = str(payload.get("verdict") or payload.get("status") or "").strip().upper()
            return {
                "verdict": verdict if verdict in {"PASS", "FAIL", "PARTIAL"} else "PARTIAL",
                "missing_clause_ids": [
                    str(item).strip()
                    for item in (payload.get("missing_clause_ids") or [])
                    if str(item).strip()
                ],
                "risks": [str(item).strip() for item in (payload.get("risks") or []) if str(item).strip()],
                "summary": output[:1600],
            }
        if artifact_type == "buyer_recommendation_table":
            payload = extract_json(output or "") or {}
            recommendations = payload.get("recommendations") or payload.get("rows") or []
            if not isinstance(recommendations, list):
                recommendations = []
            return {
                "recommendations": [dict(item) for item in recommendations if isinstance(item, dict)],
                "summary": output[:2400],
            }
        return {"summary": output[:2000], "doc_scope": doc_scope}

    def _is_valid_handoff_payload(self, artifact_type: str, payload: Dict[str, Any]) -> bool:
        required = set(_HANDOFF_SCHEMA_KEYS.get(artifact_type) or set())
        return isinstance(payload, dict) and required.issubset({str(key) for key in payload})

    def _prepare_handoff_artifacts(
        self,
        *,
        session_state: SessionState,
        task: Dict[str, Any],
        result: TaskResult,
    ) -> List[Dict[str, Any]]:
        produced_types = [str(item) for item in (task.get("produces_artifacts") or []) if str(item)]
        handoff_schema = str(task.get("handoff_schema") or "")
        if not produced_types:
            return []
        artifacts: List[Dict[str, Any]] = []
        for artifact_type in produced_types:
            payload = self._handoff_payload(artifact_type=artifact_type, task=task, result=result)
            payload.setdefault(
                "provenance_kind",
                str(result.metadata.get("evidence_provenance") or infer_result_provenance(result.to_dict())),
            )
            if not self._is_valid_handoff_payload(artifact_type, payload):
                result.warnings.append(f"Invalid handoff payload for artifact type '{artifact_type}'.")
                continue
            allowed_consumers = sorted(_HANDOFF_ALLOWED_CONSUMERS.get(artifact_type) or {"coordinator"})
            artifact = register_handoff_artifact(
                session_state,
                artifact_type=artifact_type,
                handoff_schema=handoff_schema or artifact_type,
                producer_task_id=result.task_id,
                producer_agent=result.executor,
                data=payload,
                summary=result.title or artifact_type,
                allowed_consumers=allowed_consumers,
                source_artifact_ids=list(result.handoff_artifact_ids),
            )
            artifacts.append(artifact)
            result.handoff_artifact_ids.append(str(artifact.get("artifact_id") or ""))
            self.kernel._emit(
                "worker_handoff_prepared",
                session_state.session_id,
                agent_name=result.executor,
                payload={
                    "conversation_id": session_state.conversation_id,
                    "task_id": result.task_id,
                    "artifact_type": artifact_type,
                    "artifact_id": artifact.get("artifact_id"),
                    "handoff_schema": artifact.get("handoff_schema"),
                    "allowed_consumers": allowed_consumers,
                },
            )
        return artifacts

    def _resolve_task_handoffs(
        self,
        *,
        task: Dict[str, Any],
        session_state: SessionState,
        consumer_agent: str,
    ) -> List[Dict[str, Any]]:
        explicit_ids = [str(item) for item in (task.get("input_artifact_ids") or []) if str(item)]
        desired_types = [str(item) for item in (task.get("consumes_artifacts") or []) if str(item)]
        desired_schema = str(task.get("handoff_schema") or "")
        records = (
            [artifact for artifact_id in explicit_ids if (artifact := get_handoff_artifact(session_state, artifact_id)) is not None]
            if explicit_ids
            else list_handoff_artifacts(session_state, artifact_types=desired_types)
        )
        validated: List[Dict[str, Any]] = []
        for artifact in records:
            allowed_consumers = {str(item) for item in (artifact.get("allowed_consumers") or []) if str(item)}
            if allowed_consumers and consumer_agent not in allowed_consumers:
                continue
            if desired_schema and str(artifact.get("handoff_schema") or "") != desired_schema:
                continue
            artifact_type = str(artifact.get("artifact_type") or "")
            payload = dict(artifact.get("data") or {})
            if not self._is_valid_handoff_payload(artifact_type, payload):
                continue
            validated.append(artifact)
        return validated

    def _resolve_document_records(
        self,
        *,
        session_state: SessionState,
        doc_hints: List[str],
    ) -> List[Any]:
        tenant_id = session_state.tenant_id
        raw_collection_ids = session_state.metadata.get("search_collection_ids")
        collection_ids = (
            [str(item) for item in raw_collection_ids if str(item)]
            if isinstance(raw_collection_ids, (list, tuple))
            else []
        )
        if not collection_ids:
            collection_ids = [
                str(item)
                for item in (
                    session_state.metadata.get("collection_id"),
                    session_state.metadata.get("upload_collection_id"),
                    session_state.metadata.get("kb_collection_id"),
                )
                if str(item or "").strip()
            ]
        collection_ids = list(dict.fromkeys(collection_ids))
        resolved: List[Any] = []
        seen: set[str] = set()
        cached_records: List[Any] | None = None

        for hint in doc_hints:
            hint_text = str(hint or "").strip()
            if not hint_text:
                continue
            record = self.kernel.stores.doc_store.get_document(hint_text, tenant_id=tenant_id)
            if record is None:
                if cached_records is None:
                    cached_records = []
                    if collection_ids:
                        for collection_id in collection_ids:
                            cached_records.extend(
                                self.kernel.stores.doc_store.list_documents(
                                    tenant_id=tenant_id,
                                    collection_id=collection_id,
                                )
                            )
                    else:
                        cached_records = self.kernel.stores.doc_store.list_documents(
                            tenant_id=tenant_id,
                        )
                lowered_hint = hint_text.lower()
                for candidate in cached_records:
                    title = str(getattr(candidate, "title", "") or "").lower()
                    source_name = Path(str(getattr(candidate, "source_path", "") or "")).name.lower()
                    if lowered_hint == title or lowered_hint == source_name or lowered_hint in title:
                        record = candidate
                        break
            if record is None:
                continue
            doc_id = str(getattr(record, "doc_id", "") or "")
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            resolved.append(record)
        return resolved

    def _materialize_workspace_sources(
        self,
        *,
        session_state: SessionState,
        task: Dict[str, Any],
        handoff_artifacts: List[Dict[str, Any]],
        worker_request: WorkerExecutionRequest,
    ) -> List[str]:
        if worker_request.agent_name != "data_analyst":
            return []

        workspace_root_raw = str(session_state.workspace_root or "").strip()
        if not workspace_root_raw:
            return []
        workspace_root = Path(workspace_root_raw).expanduser()
        workspace_root.mkdir(parents=True, exist_ok=True)

        doc_hints = [str(item) for item in (task.get("doc_scope") or []) if str(item)]
        for artifact in handoff_artifacts:
            if str(artifact.get("artifact_type") or "") != "doc_focus":
                continue
            for doc in artifact.get("data", {}).get("documents", []) or []:
                if not isinstance(doc, dict):
                    continue
                doc_id = str(doc.get("doc_id") or "")
                title = str(doc.get("title") or "")
                if doc_id:
                    doc_hints.append(doc_id)
                elif title:
                    doc_hints.append(title)

        records = self._resolve_document_records(session_state=session_state, doc_hints=doc_hints)
        copied_files: List[str] = []
        for record in records:
            source_path = Path(str(getattr(record, "source_path", "") or "")).expanduser()
            if not source_path.exists() or not source_path.is_file():
                continue
            destination = workspace_root / source_path.name
            if destination.exists():
                destination = workspace_root / f"{getattr(record, 'doc_id', 'doc')}_{source_path.name}"
            try:
                shutil.copy2(source_path, destination)
            except Exception:
                continue
            copied_files.append(destination.name)
        return copied_files

    @staticmethod
    def _is_dynamic_facet_fanout_task(task: Dict[str, Any]) -> bool:
        return bool(dict(task.get("controller_hints") or {}).get("dynamic_facet_fanout"))

    @staticmethod
    def _is_dynamic_doc_review_fanout_task(task: Dict[str, Any]) -> bool:
        return bool(dict(task.get("controller_hints") or {}).get("dynamic_doc_review_fanout"))

    @staticmethod
    def _is_dynamic_subsystem_backfill_task(task: Dict[str, Any]) -> bool:
        return bool(dict(task.get("controller_hints") or {}).get("dynamic_subsystem_backfill"))

    @staticmethod
    def _determine_final_output_mode(task_plan: List[Dict[str, Any]]) -> str:
        for task in task_plan:
            controller_hints = dict(task.get("controller_hints") or {})
            mode = str(controller_hints.get("final_output_mode") or "").strip()
            if mode:
                return mode
        return ""

    def _validate_task_graph(
        self,
        *,
        agent: AgentDefinition,
        session_state: SessionState,
        task_plan: List[Dict[str, Any]],
    ) -> List[str]:
        issues: List[str] = []
        effective = coerce_effective_capabilities(
            dict(session_state.metadata or {}).get("effective_capabilities")
            or dict(dict(session_state.metadata or {}).get("route_context") or {}).get("effective_capabilities")
        )
        known_ids = {str(task.get("id") or "").strip() for task in task_plan if str(task.get("id") or "").strip()}
        all_produced_artifacts = {
            str(item).strip()
            for task in task_plan
            for item in (task.get("produces_artifacts") or [])
            if str(item).strip()
        }
        allowed_workers = set(agent.allowed_worker_agents or [])
        for task in task_plan:
            task_id = str(task.get("id") or "").strip() or "<unknown>"
            executor = str(task.get("executor") or "").strip()
            if not executor:
                issues.append(f"Task {task_id} has no executor.")
                continue
            if executor not in allowed_workers:
                issues.append(f"Task {task_id} requests worker '{executor}', which coordinator cannot dispatch.")
            try:
                self.kernel._resolve_agent(executor)
            except Exception:
                issues.append(f"Task {task_id} requests unknown worker '{executor}'.")
            if effective is not None and not effective.allows_agent(executor):
                issues.append(f"Task {task_id} requests disabled worker '{executor}'.")
            for dependency in [str(item).strip() for item in (task.get("depends_on") or []) if str(item).strip()]:
                if dependency not in known_ids:
                    issues.append(f"Task {task_id} depends on unknown task '{dependency}'.")
            for artifact in [str(item).strip() for item in (task.get("consumes_artifacts") or []) if str(item).strip()]:
                if artifact not in all_produced_artifacts:
                    issues.append(f"Task {task_id} consumes artifact '{artifact}' that no task produces.")
            capability_requirements = dict(task.get("capability_requirements") or {})
            if effective is not None:
                for required_agent in [
                    str(item).strip()
                    for item in (capability_requirements.get("agents") or [])
                    if str(item).strip()
                ]:
                    if not effective.allows_agent(required_agent):
                        issues.append(f"Task {task_id} requires disabled agent '{required_agent}'.")
                for collection_id in [
                    str(item).strip()
                    for item in (capability_requirements.get("collections") or [])
                    if str(item).strip()
                ]:
                    if not effective.allows_collection(collection_id):
                        issues.append(f"Task {task_id} requires disabled collection '{collection_id}'.")
                for tool_name in [
                    str(item).strip()
                    for item in (capability_requirements.get("tools") or [])
                    if str(item).strip()
                ]:
                    definition = dict(getattr(self.kernel, "tool_definitions", {}) or {}).get(tool_name)
                    if definition is not None and not effective.allows_tool(
                        tool_name,
                        group=str(getattr(definition, "group", "") or ""),
                        read_only=bool(getattr(definition, "read_only", False)),
                        destructive=bool(getattr(definition, "destructive", False)),
                        metadata=dict(getattr(definition, "metadata", {}) or {}),
                    ):
                        issues.append(f"Task {task_id} requires disabled tool '{tool_name}'.")
        return issues

    @staticmethod
    def _safe_task_slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
        return slug[:40] or "facet"

    def _collection_scope_for_worker(
        self,
        *,
        session_state: SessionState,
        controller_hints: Dict[str, Any],
    ) -> str:
        explicit = str(
            controller_hints.get("requested_kb_collection_id")
            or controller_hints.get("kb_collection_id")
            or ""
        ).strip()
        if explicit:
            return explicit
        return str(
            session_state.metadata.get("kb_collection_id")
            or session_state.metadata.get("collection_id")
            or "default"
        ).strip() or "default"

    @staticmethod
    def _subsystem_support_is_thin(item: Dict[str, Any]) -> bool:
        coverage = str(item.get("coverage") or "").strip().lower()
        if coverage in {"thin", "weak", "sparse"}:
            return True
        supporting_documents = [dict(doc) for doc in (item.get("supporting_documents") or []) if isinstance(doc, dict)]
        supporting_citation_ids = [str(citation).strip() for citation in (item.get("supporting_citation_ids") or []) if str(citation).strip()]
        return len(supporting_documents) < 2 or len(supporting_citation_ids) < 2

    def _apply_worker_scope_overrides(
        self,
        *,
        scoped_state: SessionState,
        session_state: SessionState,
        worker_request: WorkerExecutionRequest,
    ) -> None:
        controller_hints = dict(worker_request.controller_hints or {})
        retrieval_scope_mode = str(controller_hints.get("retrieval_scope_mode") or "").strip().lower()
        if retrieval_scope_mode != "kb_only" and not bool(controller_hints.get("strict_kb_scope")):
            return
        collection_id = self._collection_scope_for_worker(
            session_state=session_state,
            controller_hints=controller_hints,
        )
        scoped_state.uploaded_doc_ids = []
        scoped_state.metadata = {
            **dict(scoped_state.metadata or {}),
            "kb_collection_id": collection_id,
            "collection_id": collection_id,
            "search_collection_ids": [collection_id],
            "retrieval_scope_mode": "kb_only",
            "retrieval_scope_reason": "coordinator_kb_only_research_workflow",
            "has_uploads": False,
            "source_upload_ids": [],
        }

    def _build_facet_search_tasks(
        self,
        *,
        session_state: SessionState,
        placeholder_task: Dict[str, Any],
        execution_state: TaskExecutionState,
        facets_payload: Dict[str, Any],
    ) -> List[TaskSpec]:
        controller_hints = dict(placeholder_task.get("controller_hints") or {})
        max_parallel_facets = max(1, min(4, int(controller_hints.get("max_parallel_facets") or 4)))
        collection_id = str(facets_payload.get("scope_collection_id") or "").strip() or self._collection_scope_for_worker(
            session_state=session_state,
            controller_hints=controller_hints,
        )
        existing_ids = {str(task.get("id") or "") for task in execution_state.task_plan}
        seen_names: set[str] = set()
        tasks: List[TaskSpec] = []

        for item in (facets_payload.get("facets") or []):
            if not isinstance(item, dict):
                continue
            facet_name = str(item.get("name") or "").strip()
            if not facet_name:
                continue
            facet_key = facet_name.casefold()
            if facet_key in seen_names:
                continue
            seen_names.add(facet_key)
            aliases = [str(alias).strip() for alias in (item.get("aliases") or []) if str(alias).strip()]
            seed_doc_ids = [str(doc_id).strip() for doc_id in (item.get("seed_doc_ids") or []) if str(doc_id).strip()]
            facet_query_parts = [
                f"Search only the knowledge base collection '{collection_id}' for documents that discuss the subsystem or architectural facet '{facet_name}'.",
            ]
            if aliases:
                facet_query_parts.append("Use these aliases and related terms: " + ", ".join(aliases[:8]) + ".")
            facet_query_parts.append("Search broadly across the KB and return evidence only for coordinator compilation.")
            facet_query_parts.append(f"OVERALL_REQUEST:\n{execution_state.user_request}")
            task_id = f"{placeholder_task.get('id', 'task')}_{self._safe_task_slug(facet_name)}"
            if task_id in existing_ids:
                continue
            existing_ids.add(task_id)
            tasks.append(
                TaskSpec(
                    id=task_id,
                    title=f"Search facet: {facet_name}",
                    executor="rag_worker",
                    mode="parallel",
                    depends_on=[str(placeholder_task.get("id") or "")],
                    input="\n\n".join(facet_query_parts),
                    doc_scope=[],
                    skill_queries=[
                        "corpus discovery",
                        "cross document inventory",
                        "windowed keyword followup",
                    ],
                    research_profile="corpus_discovery",
                    coverage_goal="corpus_wide",
                    result_mode="inventory",
                    answer_mode="evidence_only",
                    controller_hints={
                        **controller_hints,
                        "workflow_phase": "facet_search",
                        "dynamic_facet_fanout": False,
                        "facet_name": facet_name,
                        "facet_aliases": aliases,
                        "facet_rationale": str(item.get("rationale") or "").strip(),
                        "seed_doc_ids": seed_doc_ids,
                        "requested_kb_collection_id": collection_id,
                        "kb_collection_id": collection_id,
                        "search_collection_ids": [collection_id],
                        "round_budget": 2,
                        "retrieval_strategies": ["hybrid", "keyword"],
                    },
                    produces_artifacts=["facet_matches"],
                    handoff_schema=str(placeholder_task.get("handoff_schema") or "research_inventory"),
                )
            )
            if len(tasks) >= max_parallel_facets:
                break
        return tasks

    def _build_doc_review_tasks(
        self,
        *,
        session_state: SessionState,
        placeholder_task: Dict[str, Any],
        execution_state: TaskExecutionState,
        handoff_artifacts: List[Dict[str, Any]],
    ) -> List[TaskSpec]:
        controller_hints = dict(placeholder_task.get("controller_hints") or {})
        max_parallel = max(1, min(6, int(controller_hints.get("max_parallel_doc_reviews") or 4)))
        max_optional = max(0, min(2, int(controller_hints.get("max_optional_doc_reviews") or 2)))
        existing_ids = {str(task.get("id") or "") for task in execution_state.task_plan}
        ranked_rows = [
            row
            for row in self._ranked_document_rows(handoff_artifacts)
            if str(row.get("reviewed_relevance") or "").strip().lower() != "irrelevant"
        ]
        selected_rows = self._select_ranked_review_rows(
            ranked_rows,
            primary_limit=max_parallel,
            optional_limit=max_optional,
        )
        if not selected_rows:
            return []

        collection_id = ""
        for artifact in reversed(handoff_artifacts):
            payload = dict(artifact.get("data") or {})
            collection_id = str(payload.get("scope_collection_id") or "").strip()
            if collection_id:
                break
        if not collection_id:
            collection_id = self._collection_scope_for_worker(
                session_state=session_state,
                controller_hints=controller_hints,
            )

        tasks: List[TaskSpec] = []
        for row in selected_rows:
            doc_id = str(row.get("doc_id") or "").strip()
            title = str(row.get("title") or doc_id or "candidate_document").strip()
            task_id = f"{placeholder_task.get('id', 'task')}_{self._safe_task_slug(doc_id or title)}"
            if task_id in existing_ids:
                continue
            existing_ids.add(task_id)
            matched_facets = [str(item).strip() for item in (row.get("matched_facets") or []) if str(item).strip()]
            task_brief = [
                f"Review the exact indexed document '{title}' and judge whether it is relevant to the overall request.",
                "Stay strictly inside this document. Use read_indexed_doc(mode=\"full\") with cursor pagination until the document has been covered to completion or every major outline section has been inspected.",
                "Prefer section-first exploration only as a prelude to full coverage; do not rely on a single overview read.",
                "Return JSON only for coordinator synthesis.",
                f"OVERALL_REQUEST:\n{execution_state.user_request}",
            ]
            if matched_facets:
                task_brief.append("PRIORITIZED_FACETS:\n- " + "\n- ".join(matched_facets[:8]))
            if str(row.get("match_reason") or "").strip():
                task_brief.append(
                    f"TITLE_PATH_SIGNAL:\n- reason: {row.get('match_reason')}\n- score: {float(row.get('title_path_score') or 0.0):.4f}"
                )
            task_brief.append(
                "Return JSON only with this schema:\n"
                "{"
                '"document":{"doc_id":"...", "title":"...", "source_path":"...", "source_type":"..."}, '
                '"document_summary":"...", '
                '"relevance":"relevant|partial|irrelevant", '
                '"relevance_rationale":"...", '
                '"matched_facets":["..."], '
                '"coverage":"primary|supplemental|thin", '
                '"subsystems":[{"name":"...", "aliases":["..."], "description":"...", '
                '"responsibilities":["..."], "interfaces":["..."], "supporting_citation_ids":["..."]}], '
                '"responsibilities":["..."], '
                '"interfaces":["..."], '
                '"used_citation_ids":["..."]'
                "}\n"
                "Do not include prose outside JSON."
            )
            tasks.append(
                TaskSpec(
                    id=task_id,
                    title=f"Review {title}",
                    executor="general",
                    mode="parallel",
                    depends_on=[str(placeholder_task.get("id") or "")],
                    input="\n\n".join(task_brief),
                    doc_scope=[doc_id] if doc_id else [title],
                    consumes_artifacts=["title_candidates", "doc_focus", "research_facets", "facet_matches"],
                    produces_artifacts=["doc_digest"],
                    handoff_schema=str(placeholder_task.get("handoff_schema") or "research_inventory"),
                    controller_hints={
                        **controller_hints,
                        "workflow_phase": "doc_review",
                        "dynamic_doc_review_fanout": False,
                        "strict_doc_focus": True,
                        "doc_read_depth": "full",
                        "force_deep_search": True,
                        "prefer_full_reads": True,
                        "prefer_section_first": True,
                        "max_reflection_rounds": 2,
                        "requested_kb_collection_id": collection_id,
                        "kb_collection_id": collection_id,
                        "search_collection_ids": [collection_id],
                        "reviewed_doc_id": doc_id,
                        "reviewed_doc_title": title,
                        "matched_facets": matched_facets,
                        "title_path_score": float(row.get("title_path_score") or 0.0),
                    },
                )
            )
        return tasks

    def _build_subsystem_backfill_tasks(
        self,
        *,
        session_state: SessionState,
        placeholder_task: Dict[str, Any],
        execution_state: TaskExecutionState,
        inventory_payload: Dict[str, Any],
    ) -> List[TaskSpec]:
        controller_hints = dict(placeholder_task.get("controller_hints") or {})
        max_parallel = max(1, min(4, int(controller_hints.get("max_parallel_subsystems") or 4)))
        collection_id = str(inventory_payload.get("scope_collection_id") or "").strip() or self._collection_scope_for_worker(
            session_state=session_state,
            controller_hints=controller_hints,
        )
        source_documents = [
            self._normalize_document_brief(item)
            for item in (inventory_payload.get("source_documents") or controller_hints.get("active_doc_focus_documents") or [])
            if isinstance(item, dict)
        ]
        source_doc_ids = [str(item.get("doc_id") or "").strip() for item in source_documents if str(item.get("doc_id") or "").strip()]
        existing_ids = {str(task.get("id") or "") for task in execution_state.task_plan}
        tasks: List[TaskSpec] = []

        for item in (inventory_payload.get("subsystems") or []):
            if not isinstance(item, dict):
                continue
            subsystem_name = str(item.get("name") or "").strip()
            if not subsystem_name or not self._subsystem_support_is_thin(item):
                continue
            task_id = f"{placeholder_task.get('id', 'task')}_{self._safe_task_slug(subsystem_name)}"
            if task_id in existing_ids:
                continue
            existing_ids.add(task_id)
            aliases = [str(alias).strip() for alias in (item.get("aliases") or []) if str(alias).strip()]
            tasks.append(
                TaskSpec(
                    id=task_id,
                    title=f"Backfill evidence: {subsystem_name}",
                    executor="rag_worker",
                    mode="parallel",
                    depends_on=[str(placeholder_task.get("id") or "")],
                    input=(
                        f"Using only the scoped indexed documents, gather additional grounded evidence about the subsystem '{subsystem_name}'. "
                        "Focus on its responsibilities, interfaces, and how it fits into the repo architecture. Return evidence only for coordinator compilation."
                    ),
                    doc_scope=list(source_doc_ids),
                    skill_queries=[
                        "windowed keyword followup",
                        "cross document inventory",
                    ],
                    research_profile="",
                    coverage_goal="cross_document",
                    result_mode="answer",
                    answer_mode="evidence_only",
                    controller_hints={
                        **controller_hints,
                        "workflow_phase": "subsystem_evidence_backfill",
                        "dynamic_subsystem_backfill": False,
                        "subsystem_name": subsystem_name,
                        "subsystem_aliases": aliases,
                        "subsystem_rationale": str(item.get("description") or "").strip(),
                        "requested_kb_collection_id": collection_id,
                        "kb_collection_id": collection_id,
                        "search_collection_ids": [collection_id],
                        "retrieval_strategies": ["hybrid", "keyword"],
                        "round_budget": 2,
                    },
                    produces_artifacts=["subsystem_evidence"],
                    handoff_schema=str(placeholder_task.get("handoff_schema") or "active_doc_focus_summary"),
                )
            )
            if len(tasks) >= max_parallel:
                break
        return tasks

    def _build_coverage_backfill_tasks(
        self,
        *,
        session_state: SessionState,
        execution_state: TaskExecutionState,
        ledger: Dict[str, Any],
        issues: List[str],
    ) -> List[TaskSpec]:
        existing_ids = self._current_task_ids(execution_state)
        if any(bool(dict(task.get("controller_hints") or {}).get("coverage_backfill")) for task in execution_state.task_plan):
            return []
        collection_id = str(
            session_state.metadata.get("kb_collection_id")
            or session_state.metadata.get("collection_id")
            or "default"
        ).strip() or "default"
        candidate_titles = [
            str(item.get("title") or item.get("doc_id") or "").strip()
            for item in (ledger.get("candidate_documents") or [])
            if isinstance(item, dict) and not bool(item.get("is_meta_document"))
        ]
        backfill_id = "task_coverage_backfill"
        refresh_id = "task_coverage_inventory_refresh"
        suffix = 2
        while backfill_id in existing_ids or refresh_id in existing_ids:
            backfill_id = f"task_coverage_backfill_{suffix}"
            refresh_id = f"task_coverage_inventory_refresh_{suffix}"
            suffix += 1
        search_input_parts = [
            f"Search only the knowledge base collection '{collection_id}' for primary source documents that improve holistic repository architecture coverage.",
            "Prioritize documents about agents, routing, coordinator execution, tools, skills, RAG/retrieval, persistence, observability, gateway/API, workspace, and runtime control flow.",
            "Do not count prompt catalogs, test query packs, fixtures, or acceptance scenarios as primary architecture evidence unless the user explicitly asked about tests.",
            "Return evidence only for coordinator compilation.",
            "COVERAGE_GATE_ISSUES:\n- " + "\n- ".join(issues),
            f"ORIGINAL_REQUEST:\n{execution_state.user_request}",
        ]
        if candidate_titles:
            search_input_parts.append("CURRENT_PRIMARY_CANDIDATES:\n- " + "\n- ".join(candidate_titles[:12]))
        refresh_input = (
            "Use the existing research handoffs plus the coverage-backfill evidence to refresh the subsystem inventory. "
            "Keep only primary architecture/source documents as strong support. Mark any subsystem thin when it still has sparse support. "
            "Return JSON only with this schema:\n"
            "{"
            '"subsystems":[{"name":"...", "aliases":["..."], "description":"...", '
            '"responsibilities":["..."], "interfaces":["..."], '
            '"supporting_documents":[{"doc_id":"...", "title":"...", "source_path":"...", "source_type":"..."}], '
            '"supporting_citation_ids":["..."], "coverage":"strong|thin"}], '
            '"source_documents":[{"doc_id":"...", "title":"...", "source_path":"...", "source_type":"..."}], '
            f'"scope_collection_id":"{collection_id}"'
            "}\n"
            "Do not include prose outside JSON."
        )
        return [
            TaskSpec(
                id=backfill_id,
                title="Backfill holistic repository coverage",
                executor="rag_worker",
                mode="sequential",
                depends_on=[],
                input="\n\n".join(search_input_parts),
                doc_scope=[],
                skill_queries=[
                    "corpus discovery",
                    "cross document inventory",
                    "coverage sufficiency audit",
                    "windowed keyword followup",
                ],
                research_profile="corpus_discovery",
                coverage_goal="corpus_wide",
                result_mode="inventory",
                answer_mode="evidence_only",
                controller_hints={
                    "coverage_backfill": True,
                    "workflow_phase": "coverage_backfill",
                    "retrieval_scope_mode": "kb_only",
                    "strict_kb_scope": True,
                    "requested_kb_collection_id": collection_id,
                    "kb_collection_id": collection_id,
                    "search_collection_ids": [collection_id],
                    "retrieval_strategies": ["hybrid", "keyword"],
                    "round_budget": 2,
                    "force_deep_search": True,
                    "prefer_full_reads": True,
                    "final_output_mode": "detailed_subsystem_summary",
                },
                produces_artifacts=["doc_focus"],
                handoff_schema="research_inventory",
            ),
            TaskSpec(
                id=refresh_id,
                title="Refresh subsystem inventory after coverage backfill",
                executor="general",
                mode="sequential",
                depends_on=[backfill_id],
                input=refresh_input,
                consumes_artifacts=[
                    "title_candidates",
                    "doc_focus",
                    "research_facets",
                    "facet_matches",
                    "doc_digest",
                    "subsystem_inventory",
                    "subsystem_evidence",
                ],
                produces_artifacts=["subsystem_inventory"],
                handoff_schema="research_inventory",
                controller_hints={
                    "coverage_backfill": True,
                    "workflow_phase": "coverage_inventory_refresh",
                    "retrieval_scope_mode": "kb_only",
                    "strict_kb_scope": True,
                    "requested_kb_collection_id": collection_id,
                    "kb_collection_id": collection_id,
                    "search_collection_ids": [collection_id],
                    "prefer_structured_final_answer": True,
                    "final_output_mode": "detailed_subsystem_summary",
                },
            ),
        ]

    def _expand_research_facet_fanout(
        self,
        *,
        execution_state: TaskExecutionState,
        session_state: SessionState,
        task: Dict[str, Any],
    ) -> TaskResult:
        handoff_artifacts = self._resolve_task_handoffs(
            task=task,
            session_state=session_state,
            consumer_agent="coordinator",
        )
        latest_payload = {}
        if handoff_artifacts:
            latest_payload = dict(handoff_artifacts[-1].get("data") or {})
        generated_tasks = self._build_facet_search_tasks(
            session_state=session_state,
            placeholder_task=task,
            execution_state=execution_state,
            facets_payload=latest_payload,
        )
        if generated_tasks:
            insertion_index = next(
                (index for index, item in enumerate(execution_state.task_plan) if str(item.get("id") or "") == str(task.get("id") or "")),
                len(execution_state.task_plan),
            )
            execution_state.task_plan[insertion_index + 1:insertion_index + 1] = [item.to_dict() for item in generated_tasks]
            self.kernel._emit(
                "coordinator_dynamic_fanout_prepared",
                session_state.session_id,
                agent_name="coordinator",
                payload={
                    "conversation_id": session_state.conversation_id,
                    "placeholder_task_id": str(task.get("id") or ""),
                    "generated_task_ids": [item.id for item in generated_tasks],
                },
            )
            output = "Expanded facet searches: " + ", ".join(item.title for item in generated_tasks)
        else:
            output = "No research facets were available to expand into follow-up searches."
        return TaskResult(
            task_id=str(task.get("id") or ""),
            title=str(task.get("title") or "Expand facet searches"),
            executor="coordinator",
            status="completed",
            output=output,
            artifact_ref=f"task:{task.get('id', '')}",
            warnings=[] if generated_tasks else ["No research facets available for dynamic fanout."],
            metadata={
                "dynamic_fanout": {
                    "generated_task_ids": [item.id for item in generated_tasks],
                    "facet_count": len(latest_payload.get("facets") or []),
                }
            },
        )

    def _expand_doc_review_fanout(
        self,
        *,
        execution_state: TaskExecutionState,
        session_state: SessionState,
        task: Dict[str, Any],
    ) -> TaskResult:
        handoff_artifacts = self._resolve_task_handoffs(
            task=task,
            session_state=session_state,
            consumer_agent="coordinator",
        )
        generated_tasks = self._build_doc_review_tasks(
            session_state=session_state,
            placeholder_task=task,
            execution_state=execution_state,
            handoff_artifacts=handoff_artifacts,
        )
        if generated_tasks:
            insertion_index = next(
                (index for index, item in enumerate(execution_state.task_plan) if str(item.get("id") or "") == str(task.get("id") or "")),
                len(execution_state.task_plan),
            )
            execution_state.task_plan[insertion_index + 1:insertion_index + 1] = [item.to_dict() for item in generated_tasks]
            self.kernel._emit(
                "coordinator_dynamic_doc_review_prepared",
                session_state.session_id,
                agent_name="coordinator",
                payload={
                    "conversation_id": session_state.conversation_id,
                    "placeholder_task_id": str(task.get("id") or ""),
                    "generated_task_ids": [item.id for item in generated_tasks],
                },
            )
            output = "Expanded document review: " + ", ".join(item.title for item in generated_tasks)
        else:
            output = "No candidate documents met the threshold for expanded review."
        return TaskResult(
            task_id=str(task.get("id") or ""),
            title=str(task.get("title") or "Expand document review"),
            executor="coordinator",
            status="completed",
            output=output,
            artifact_ref=f"task:{task.get('id', '')}",
            warnings=[] if generated_tasks else ["No document review tasks were generated."],
            metadata={
                "dynamic_doc_review": {
                    "generated_task_ids": [item.id for item in generated_tasks],
                    "candidate_count": len(self._ranked_document_rows(handoff_artifacts)),
                }
            },
        )

    def _expand_subsystem_backfill(
        self,
        *,
        execution_state: TaskExecutionState,
        session_state: SessionState,
        task: Dict[str, Any],
    ) -> TaskResult:
        handoff_artifacts = self._resolve_task_handoffs(
            task=task,
            session_state=session_state,
            consumer_agent="coordinator",
        )
        latest_payload = {}
        if handoff_artifacts:
            latest_payload = dict(handoff_artifacts[-1].get("data") or {})
        generated_tasks = self._build_subsystem_backfill_tasks(
            session_state=session_state,
            placeholder_task=task,
            execution_state=execution_state,
            inventory_payload=latest_payload,
        )
        if generated_tasks:
            insertion_index = next(
                (index for index, item in enumerate(execution_state.task_plan) if str(item.get("id") or "") == str(task.get("id") or "")),
                len(execution_state.task_plan),
            )
            execution_state.task_plan[insertion_index + 1:insertion_index + 1] = [item.to_dict() for item in generated_tasks]
            self.kernel._emit(
                "coordinator_dynamic_backfill_prepared",
                session_state.session_id,
                agent_name="coordinator",
                payload={
                    "conversation_id": session_state.conversation_id,
                    "placeholder_task_id": str(task.get("id") or ""),
                    "generated_task_ids": [item.id for item in generated_tasks],
                },
            )
            output = "Expanded subsystem evidence backfill: " + ", ".join(item.title for item in generated_tasks)
        else:
            output = "No thin-support subsystems required targeted evidence backfill."
        return TaskResult(
            task_id=str(task.get("id") or ""),
            title=str(task.get("title") or "Expand subsystem evidence backfill"),
            executor="coordinator",
            status="completed",
            output=output,
            artifact_ref=f"task:{task.get('id', '')}",
            warnings=[] if generated_tasks else ["No subsystem evidence backfill tasks were needed."],
            metadata={
                "dynamic_backfill": {
                    "generated_task_ids": [item.id for item in generated_tasks],
                    "subsystem_count": len(latest_payload.get("subsystems") or []),
                }
            },
        )

    def run(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        callbacks: List[Any],
    ) -> Any:
        planner_name = str(agent.metadata.get("planner_agent") or "planner")
        finalizer_name = str(agent.metadata.get("finalizer_agent") or "finalizer")
        verifier_name = str(agent.metadata.get("verifier_agent") or "verifier")
        verify_outputs = bool(agent.metadata.get("verify_outputs", False))
        configured_max_revision_rounds = max(1, int(getattr(self.kernel.settings, "max_revision_rounds", 4)))

        planner = self.kernel._resolve_agent(planner_name)
        finalizer = self.kernel._resolve_agent(finalizer_name)
        effective_capabilities = coerce_effective_capabilities(
            dict(session_state.metadata or {}).get("effective_capabilities")
            or dict(dict(session_state.metadata or {}).get("route_context") or {}).get("effective_capabilities")
        )
        registry_agents = list(self.kernel.registry.list())
        available_agents = [
            definition.name
            for definition in registry_agents
            if effective_capabilities is None or effective_capabilities.allows_agent(definition.name)
        ]
        available_tools = sorted(
            {
                str(tool_name)
                for definition in registry_agents
                for tool_name in list(getattr(definition, "allowed_tools", []) or [])
                if str(tool_name).strip()
                and (effective_capabilities is None or effective_capabilities.allows_tool(str(tool_name)))
            }
        )
        planner_input_packet = build_planner_input_packet(
            user_text,
            session_metadata=dict(session_state.metadata or {}),
            available_agents=available_agents,
            available_tools=available_tools,
        )
        planner_context_summary = {
            "attachment_count": len(planner_input_packet.get("attachments") or []),
            "selected_kb_collections": list(planner_input_packet.get("selected_kb_collections") or []),
            "permission_mode": str(planner_input_packet.get("permission_mode") or ""),
            "risk_flags": list(planner_input_packet.get("risk_flags") or []),
            "available_agent_count": len(planner_input_packet.get("available_agents") or []),
            "available_tool_count": len(planner_input_packet.get("available_tools") or []),
            "available_skill_pack_count": len(planner_input_packet.get("available_skill_packs") or []),
        }

        self.kernel._emit(
            "coordinator_planning_started",
            session_state.session_id,
            agent_name=agent.name,
            payload={
                "conversation_id": session_state.conversation_id,
                "planner_agent": planner.name,
                "planner_context": planner_context_summary,
                **dict(session_state.metadata.get("route_context") or {}),
            },
        )
        planner_result = self.kernel.run_agent(
            planner,
            session_state,
            user_text=user_text,
            callbacks=callbacks,
            task_payload={"planner_input_packet": planner_input_packet},
        )
        if is_clarification_turn(planner_result.metadata):
            from agentic_chatbot_next.runtime.kernel import AgentRunResult

            clarification_message = RuntimeMessage(
                role="assistant",
                content=planner_result.text,
                metadata=dict(planner_result.metadata or {}),
            )
            return AgentRunResult(
                text=planner_result.text,
                messages=list(session_state.messages) + [clarification_message],
                metadata=dict(planner_result.metadata or {}),
            )
        planner_skill_resolution = dict(planner_result.metadata.get("skill_resolution") or {})
        planner_rag_execution_hints = dict(planner_result.metadata.get("rag_execution_hints") or {})
        planner_payload = dict(planner_result.metadata.get("planner_payload") or {})
        if not planner_payload:
            planner_payload = extract_json(planner_result.text or "") or {}
        self.kernel._emit(
            "coordinator_planning_completed",
            session_state.session_id,
            agent_name=agent.name,
            payload={
                "conversation_id": session_state.conversation_id,
                "task_count": len(planner_payload.get("tasks") or []),
                "planner_agent": planner.name,
                "planner_context": planner_context_summary,
                **dict(session_state.metadata.get("route_context") or {}),
            },
        )
        if bool(planner_result.metadata.get("plan_repair_applied")):
            self.kernel._emit(
                "coordinator_plan_repaired",
                session_state.session_id,
                agent_name=agent.name,
                payload={
                    "conversation_id": session_state.conversation_id,
                    "planner_agent": planner.name,
                    "raw_task_count": int(planner_result.metadata.get("planner_raw_task_count") or 0),
                    "normalized_task_count": int(planner_result.metadata.get("planner_normalized_task_count") or len(planner_payload.get("tasks") or [])),
                    **dict(session_state.metadata.get("route_context") or {}),
                },
            )

        execution_state = TaskExecutionState(
            user_request=user_text,
            planner_summary=str(planner_payload.get("summary") or ""),
            task_plan=list(planner_payload.get("tasks") or []),
        )
        validation_issues = self._validate_task_graph(
            agent=agent,
            session_state=session_state,
            task_plan=execution_state.task_plan,
        )
        if validation_issues:
            from agentic_chatbot_next.runtime.kernel import AgentRunResult

            text = (
                "I could not execute the coordinator plan because the active capability profile blocks part of it:\n"
                + "\n".join(f"- {issue}" for issue in validation_issues)
            )
            assistant_message = RuntimeMessage(
                role="assistant",
                content=text,
                metadata={
                    "agent_name": agent.name,
                    "turn_outcome": "capability_plan_validation_failed",
                    "plan_validation_issues": validation_issues,
                },
            )
            return AgentRunResult(
                text=text,
                messages=list(session_state.messages) + [assistant_message],
                metadata=dict(assistant_message.metadata or {}),
            )
        effective_capabilities = coerce_effective_capabilities(
            dict(session_state.metadata or {}).get("effective_capabilities")
            or dict(dict(session_state.metadata or {}).get("route_context") or {}).get("effective_capabilities")
        )
        if effective_capabilities is not None and effective_capabilities.permission_mode == "plan":
            from agentic_chatbot_next.runtime.kernel import AgentRunResult

            task_lines = [
                f"- {task.get('id')}: {task.get('title')} [{task.get('executor')}]"
                for task in execution_state.task_plan
            ]
            text = (
                "Plan mode is enabled, so I prepared the coordinator task graph and stopped before execution.\n\n"
                + "\n".join(task_lines)
                + "\n\nApprove execution by switching permission mode back to `default` or `bypass` for this turn."
            )
            assistant_message = RuntimeMessage(
                role="assistant",
                content=text,
                metadata={
                    "agent_name": agent.name,
                    "turn_outcome": "plan_mode_preview",
                    "task_graph": execution_state.task_plan,
                },
            )
            return AgentRunResult(
                text=text,
                messages=list(session_state.messages) + [assistant_message],
                metadata=dict(assistant_message.metadata or {}),
            )

        task_results: List[Dict[str, Any]] = []

        def _execute_available_batches() -> Optional[Any]:
            while True:
                batch = select_execution_batch(execution_state.task_plan, task_results)
                if not batch:
                    return None
                if len(batch) == 1 and self._is_dynamic_facet_fanout_task(batch[0]):
                    expansion_result = self._expand_research_facet_fanout(
                        execution_state=execution_state,
                        session_state=session_state,
                        task=batch[0],
                    )
                    task_results.append(expansion_result.to_dict())
                    continue
                if len(batch) == 1 and self._is_dynamic_doc_review_fanout_task(batch[0]):
                    expansion_result = self._expand_doc_review_fanout(
                        execution_state=execution_state,
                        session_state=session_state,
                        task=batch[0],
                    )
                    task_results.append(expansion_result.to_dict())
                    continue
                if len(batch) == 1 and self._is_dynamic_subsystem_backfill_task(batch[0]):
                    expansion_result = self._expand_subsystem_backfill(
                        execution_state=execution_state,
                        session_state=session_state,
                        task=batch[0],
                    )
                    task_results.append(expansion_result.to_dict())
                    continue
                self.kernel._emit(
                    "coordinator_batch_started",
                    session_state.session_id,
                    agent_name=agent.name,
                    payload={
                        "conversation_id": session_state.conversation_id,
                        "task_ids": [str(task.get("id") or "") for task in batch],
                        **dict(session_state.metadata.get("route_context") or {}),
                    },
                )
                task_results.extend(
                    self.run_task_batch(
                        agent=agent,
                        session_state=session_state,
                        user_request=user_text,
                        callbacks=callbacks,
                        batch=batch,
                    )
                )
                clarification_result = self._first_task_clarification(task_results)
                if clarification_result is not None:
                    from agentic_chatbot_next.runtime.kernel import AgentRunResult

                    return AgentRunResult(
                        text=str(clarification_result.get("output") or "").strip(),
                        messages=list(session_state.messages)
                        + [
                            RuntimeMessage(
                                role="assistant",
                                content=str(clarification_result.get("output") or "").strip(),
                                metadata=dict(clarification_result.get("metadata") or {}),
                            )
                        ],
                        metadata=dict(clarification_result.get("metadata") or {}),
                    )

        clarification_response = _execute_available_batches()
        if clarification_response is not None:
            return clarification_response

        coverage_ledger = self._register_research_coverage_ledger(
            session_state=session_state,
            execution_state=execution_state,
        )
        coverage_issues = self._coverage_gate_issues(
            session_state=session_state,
            execution_state=execution_state,
            ledger=coverage_ledger,
        )
        if coverage_issues:
            self.kernel._emit(
                "coverage_gate_failed",
                session_state.session_id,
                agent_name=agent.name,
                payload={
                    "conversation_id": session_state.conversation_id,
                    "issues": coverage_issues,
                    "coverage_state": coverage_ledger.get("coverage_state"),
                    **dict(session_state.metadata.get("route_context") or {}),
                },
            )
            backfill_tasks = self._build_coverage_backfill_tasks(
                session_state=session_state,
                execution_state=execution_state,
                ledger=coverage_ledger,
                issues=coverage_issues,
            )
            if backfill_tasks:
                execution_state.task_plan.extend([task.to_dict() for task in backfill_tasks])
                self.kernel._emit(
                    "coverage_backfill_triggered",
                    session_state.session_id,
                    agent_name=agent.name,
                    payload={
                        "conversation_id": session_state.conversation_id,
                        "issues": coverage_issues,
                        "generated_task_ids": [task.id for task in backfill_tasks],
                        **dict(session_state.metadata.get("route_context") or {}),
                    },
                )
                clarification_response = _execute_available_batches()
                if clarification_response is not None:
                    return clarification_response
                coverage_ledger = self._register_research_coverage_ledger(
                    session_state=session_state,
                    execution_state=execution_state,
                )

        execution_state.task_results = task_results
        execution_state.partial_answer = self.build_partial_answer(task_results)
        execution_payload = execution_state.to_dict()
        execution_payload["research_coverage_ledger"] = dict(coverage_ledger or {})
        execution_payload["skill_queries"] = self.coordinator_skill_queries(execution_state.task_plan)
        final_output_mode = self._determine_final_output_mode(execution_state.task_plan)
        if final_output_mode:
            execution_payload["final_output_mode"] = final_output_mode
        max_revision_rounds = self._effective_revision_limit(
            session_state=session_state,
            execution_state=execution_state,
            configured_limit=configured_max_revision_rounds,
        )

        verifier = self.kernel.registry.get(verifier_name) if verify_outputs else None
        verification_payload: Dict[str, Any] = VerificationResult().to_dict()
        revision_round = 0
        final_text = ""
        detailed_summary_mode = final_output_mode == "detailed_subsystem_summary"
        finalizer_skill_resolution: Dict[str, Any] = {}
        finalizer_rag_execution_hints: Dict[str, Any] = {}
        verifier_skill_resolution: Dict[str, Any] = {}
        verifier_rag_execution_hints: Dict[str, Any] = {}
        previous_feedback_signature = ""
        repeated_feedback_rounds = 0
        previous_final_answer_signature = ""
        revision_stop_reason = ""
        while True:
            revision_round += 1
            finalizer_attempt = 0
            while True:
                finalizer_attempt += 1
                execution_digest = build_execution_digest(
                    execution_payload,
                    metadata=dict(session_state.metadata or {}),
                    artifacts=self._current_workflow_handoff_artifacts(
                        session_state=session_state,
                        execution_state=execution_state,
                    ),
                    revision_feedback=str(execution_payload.get("revision_feedback") or ""),
                )
                finalizer_payload = {
                    "execution_digest": execution_digest.to_dict(),
                    "verification": dict(execution_payload.get("verification") or {}),
                    "revision_feedback": str(execution_payload.get("revision_feedback") or ""),
                }
                if detailed_summary_mode and finalizer_attempt > 1:
                    finalizer_payload["prefer_structured_final_answer"] = True
                    finalizer_payload["finalizer_retry_reason"] = (
                        "The previous answer was too shallow or drifted away from the exact candidate-document scope. "
                        "Rebuild a detailed subsystem-organized synthesis using only the structured task outputs and the scoped candidate documents."
                    )
                self.kernel._emit(
                    "coordinator_finalizer_started",
                    session_state.session_id,
                    agent_name=agent.name,
                    payload={
                        "conversation_id": session_state.conversation_id,
                        "finalizer_agent": finalizer.name,
                        "revision_round": revision_round,
                        "max_revision_rounds": max_revision_rounds,
                        **dict(session_state.metadata.get("route_context") or {}),
                    },
                )
                finalizer_result = self.kernel.run_agent(
                    finalizer,
                    session_state,
                    user_text=user_text,
                    callbacks=callbacks,
                    task_payload=finalizer_payload,
                )
                if is_clarification_turn(finalizer_result.metadata):
                    from agentic_chatbot_next.runtime.kernel import AgentRunResult

                    return AgentRunResult(
                        text=finalizer_result.text,
                        messages=list(session_state.messages)
                        + [
                            RuntimeMessage(
                                role="assistant",
                                content=finalizer_result.text,
                                metadata=dict(finalizer_result.metadata or {}),
                            )
                        ],
                        metadata=dict(finalizer_result.metadata or {}),
                    )
                finalizer_skill_resolution = dict(finalizer_result.metadata.get("skill_resolution") or {})
                finalizer_rag_execution_hints = dict(finalizer_result.metadata.get("rag_execution_hints") or {})
                final_text = finalizer_result.text
                if not detailed_summary_mode:
                    break
                inventory_payload = self._latest_current_artifact_payload(
                    session_state=session_state,
                    execution_state=execution_state,
                    artifact_type="subsystem_inventory",
                )
                if not self._needs_detailed_summary_repair(final_text, inventory_payload=inventory_payload):
                    break
                if finalizer_attempt >= 2:
                    fallback_text = self._render_detailed_subsystem_fallback(
                        session_state=session_state,
                        execution_state=execution_state,
                    )
                    if fallback_text.strip():
                        final_text = fallback_text
                    break
            execution_state.final_answer = final_text
            execution_payload["final_answer"] = final_text
            self.kernel._emit(
                "coordinator_finalizer_completed",
                session_state.session_id,
                agent_name=agent.name,
                payload={
                    "conversation_id": session_state.conversation_id,
                    "finalizer_agent": finalizer.name,
                    "revision_round": revision_round,
                    "max_revision_rounds": max_revision_rounds,
                    **dict(session_state.metadata.get("route_context") or {}),
                },
            )

            if verifier is None:
                break

            self.kernel._emit(
                "coordinator_verifier_started",
                session_state.session_id,
                agent_name=agent.name,
                payload={
                    "conversation_id": session_state.conversation_id,
                    "verifier_agent": verifier.name,
                    "revision_round": revision_round,
                    "max_revision_rounds": max_revision_rounds,
                    **dict(session_state.metadata.get("route_context") or {}),
                },
            )
            verifier_result = self.kernel.run_agent(
                verifier,
                session_state,
                user_text=user_text,
                callbacks=callbacks,
                task_payload={
                    "execution_digest": build_execution_digest(
                        execution_payload,
                        metadata=dict(session_state.metadata or {}),
                        artifacts=self._current_workflow_handoff_artifacts(
                            session_state=session_state,
                            execution_state=execution_state,
                        ),
                        revision_feedback=str(execution_payload.get("revision_feedback") or ""),
                    ).to_dict(),
                    "verification": dict(execution_payload.get("verification") or {}),
                    "revision_feedback": str(execution_payload.get("revision_feedback") or ""),
                },
            )
            if is_clarification_turn(verifier_result.metadata):
                from agentic_chatbot_next.runtime.kernel import AgentRunResult

                return AgentRunResult(
                    text=verifier_result.text,
                    messages=list(session_state.messages)
                    + [
                        RuntimeMessage(
                            role="assistant",
                            content=verifier_result.text,
                            metadata=dict(verifier_result.metadata or {}),
                        )
                    ],
                        metadata=dict(verifier_result.metadata or {}),
                    )
            verifier_skill_resolution = dict(verifier_result.metadata.get("skill_resolution") or {})
            verifier_rag_execution_hints = dict(verifier_result.metadata.get("rag_execution_hints") or {})
            verification = self.parse_verification_result(verifier_result)
            verification_payload = {
                **verification.to_dict(),
                "revision_round": revision_round,
                "max_revision_rounds": max_revision_rounds,
                "configured_max_revision_rounds": configured_max_revision_rounds,
            }
            execution_state.verification = dict(verification_payload)
            execution_payload["verification"] = dict(verification_payload)
            self.kernel._emit(
                "coordinator_verifier_completed",
                session_state.session_id,
                agent_name=agent.name,
                payload={
                    "conversation_id": session_state.conversation_id,
                    "verifier_agent": verifier.name,
                    "status": verification.status,
                    "verdict": verification.verdict,
                    "issues": verification.issues,
                    "revision_round": revision_round,
                    "max_revision_rounds": max_revision_rounds,
                    **dict(session_state.metadata.get("route_context") or {}),
                },
            )
            if verification.parse_failed:
                revision_stop_reason = "verifier_parse_failed"
                verification_payload["revision_stop_reason"] = revision_stop_reason
                execution_state.verification = dict(verification_payload)
                execution_payload["verification"] = dict(verification_payload)
                break
            if verification.status != "revise" or not verification.feedback:
                break
            feedback_signature = self._feedback_signature(verification.feedback)
            if feedback_signature and feedback_signature == previous_feedback_signature:
                repeated_feedback_rounds += 1
            else:
                repeated_feedback_rounds = 0
            previous_feedback_signature = feedback_signature
            if repeated_feedback_rounds >= 1:
                revision_stop_reason = "repeated_verifier_feedback"
                verification_payload["revision_stop_reason"] = revision_stop_reason
                execution_state.verification = dict(verification_payload)
                execution_payload["verification"] = dict(verification_payload)
                break
            answer_signature = self._answer_signature(final_text)
            if previous_final_answer_signature and answer_signature == previous_final_answer_signature:
                revision_stop_reason = "no_material_final_answer_delta"
                verification_payload["revision_stop_reason"] = revision_stop_reason
                execution_state.verification = dict(verification_payload)
                execution_payload["verification"] = dict(verification_payload)
                break
            previous_final_answer_signature = answer_signature
            if revision_round >= max_revision_rounds:
                verification_payload["revision_limit_reached"] = True
                revision_stop_reason = "revision_limit_reached"
                verification_payload["revision_stop_reason"] = revision_stop_reason
                execution_state.verification = dict(verification_payload)
                execution_payload["verification"] = dict(verification_payload)
                self.kernel._emit(
                    "coordinator_revision_limit_reached",
                    session_state.session_id,
                    agent_name=agent.name,
                    payload={
                        "conversation_id": session_state.conversation_id,
                        "revision_round": revision_round,
                        "max_revision_rounds": max_revision_rounds,
                        "feedback": verification.feedback,
                        **dict(session_state.metadata.get("route_context") or {}),
                    },
                )
                break
            self.kernel._emit(
                "coordinator_revision_round_started",
                session_state.session_id,
                agent_name=agent.name,
                payload={
                    "conversation_id": session_state.conversation_id,
                    "revision_round": revision_round + 1,
                    "max_revision_rounds": max_revision_rounds,
                    "feedback": verification.feedback,
                    **dict(session_state.metadata.get("route_context") or {}),
                },
            )
            execution_state.partial_answer = final_text
            execution_payload["partial_answer"] = final_text
            execution_payload["revision_feedback"] = verification.feedback

        from agentic_chatbot_next.runtime.kernel import AgentRunResult

        if revision_stop_reason and revision_stop_reason != "revision_limit_reached":
            self.kernel._emit(
                "coordinator_revision_stopped",
                session_state.session_id,
                agent_name=agent.name,
                payload={
                    "conversation_id": session_state.conversation_id,
                    "revision_round": revision_round,
                    "reason": revision_stop_reason,
                    "max_revision_rounds": max_revision_rounds,
                    **dict(session_state.metadata.get("route_context") or {}),
                },
            )

        doc_focus_result = self._coordinator_doc_focus_result(
            session_state=session_state,
            execution_state=execution_state,
            source_query=user_text,
            rendered_answer=final_text,
        )
        assistant_metadata = {
            "agent_name": agent.name,
            "planner_payload": planner_payload,
            "planner_input_packet": planner_input_packet,
            "task_execution_state": execution_state.to_dict(),
            "verification": verification_payload,
            "revision_rounds_used": revision_round,
            "max_revision_rounds": max_revision_rounds,
            "configured_max_revision_rounds": configured_max_revision_rounds,
            "research_coverage_ledger": dict(coverage_ledger or {}),
        }
        if revision_stop_reason:
            assistant_metadata["revision_stop_reason"] = revision_stop_reason
        if planner_skill_resolution:
            assistant_metadata["planner_skill_resolution"] = planner_skill_resolution
        if planner_rag_execution_hints:
            assistant_metadata["planner_rag_execution_hints"] = planner_rag_execution_hints
        if finalizer_skill_resolution:
            assistant_metadata["finalizer_skill_resolution"] = finalizer_skill_resolution
        if finalizer_rag_execution_hints:
            assistant_metadata["finalizer_rag_execution_hints"] = finalizer_rag_execution_hints
        if verifier_skill_resolution:
            assistant_metadata["verifier_skill_resolution"] = verifier_skill_resolution
        if verifier_rag_execution_hints:
            assistant_metadata["verifier_rag_execution_hints"] = verifier_rag_execution_hints
        if doc_focus_result is not None:
            assistant_metadata["doc_focus_result"] = doc_focus_result

        return AgentRunResult(
            text=final_text,
            messages=list(session_state.messages) + [RuntimeMessage(role="assistant", content=final_text, metadata=assistant_metadata)],
            metadata=dict(assistant_metadata),
        )

    def coordinator_skill_queries(self, task_plan: List[Dict[str, Any]]) -> List[str]:
        queries: List[str] = []
        for task in task_plan:
            if str(task.get("executor") or "") != "rag_worker":
                continue
            result_mode = str(task.get("result_mode") or "").strip().lower()
            coverage_goal = str(task.get("coverage_goal") or "").strip().lower()
            research_profile = str(task.get("research_profile") or "").strip().lower()
            if result_mode == "inventory" and "document campaign synthesis" not in queries:
                queries.append("document campaign synthesis")
            if coverage_goal in {"corpus_wide", "exhaustive"} and "corpus coverage and overclaim check" not in queries:
                queries.append("corpus coverage and overclaim check")
            if research_profile == "comparison_campaign" and "document campaign synthesis" not in queries:
                queries.append("document campaign synthesis")
        return queries

    def _coordinator_doc_focus_result(
        self,
        *,
        session_state: SessionState,
        execution_state: TaskExecutionState,
        source_query: str,
        rendered_answer: str = "",
    ) -> Dict[str, Any] | None:
        artifacts = self._current_workflow_handoff_artifacts(
            session_state=session_state,
            execution_state=execution_state,
            artifact_types=[
                "title_candidates",
                "doc_focus",
                "research_facets",
                "facet_matches",
                "doc_digest",
                "subsystem_inventory",
                "subsystem_evidence",
            ],
        )
        final_output_mode = self._determine_final_output_mode(execution_state.task_plan)
        if final_output_mode == "detailed_subsystem_summary":
            existing_focus = dict(session_state.metadata.get("active_doc_focus") or {})
            preserved = build_doc_focus_result(
                collection_id=str(
                    existing_focus.get("collection_id")
                    or session_state.metadata.get("kb_collection_id")
                    or session_state.metadata.get("collection_id")
                    or "default"
                ),
                documents=existing_focus.get("documents") or [],
                source_query=source_query,
                result_mode="answer",
            )
            if preserved is not None:
                return preserved
        documents = self._documents_for_synthesis(artifacts) or self._collect_documents_from_handoffs(artifacts)
        if final_output_mode == "document_titles_only":
            ordered = self._ordered_documents_from_rendered_answer(rendered_answer, documents)
            if ordered:
                documents = ordered
        if not documents:
            return None
        collection_id = str(
            session_state.metadata.get("kb_collection_id")
            or session_state.metadata.get("collection_id")
            or "default"
        ).strip() or "default"
        result_mode = "inventory" if final_output_mode == "document_titles_only" else "answer"
        return build_doc_focus_result(
            collection_id=collection_id,
            documents=documents,
            source_query=source_query,
            result_mode=result_mode,
        )

    def build_scoped_worker_state(self, parent: SessionState, *, agent_name: str) -> SessionState:
        return SessionState(
            tenant_id=parent.tenant_id,
            user_id=parent.user_id,
            conversation_id=parent.conversation_id,
            request_id=parent.request_id,
            session_id=parent.session_id,
            uploaded_doc_ids=list(parent.uploaded_doc_ids),
            demo_mode=parent.demo_mode,
            workspace_root=parent.workspace_root,
            metadata={
                **dict(parent.metadata or {}),
                "scoped_worker": True,
                "agent_name": agent_name,
                "parent_session_id": parent.session_id,
            },
        )

    def _effective_revision_limit(
        self,
        *,
        session_state: SessionState,
        execution_state: TaskExecutionState,
        configured_limit: int,
    ) -> int:
        intent = resolved_turn_intent_from_metadata(dict(session_state.metadata or {})) or resolve_turn_intent(
            execution_state.user_request,
            dict(session_state.metadata or {}),
        )
        limit = max(1, int(configured_limit))
        if intent.answer_contract.kind == "inventory":
            limit = min(limit, 2)
        elif intent.answer_contract.kind == "grounded_synthesis" and intent.answer_contract.depth == "deep":
            limit = min(limit, 4)
        else:
            limit = min(limit, 3)
        digest = build_execution_digest(
            execution_state.to_dict(),
            metadata=dict(session_state.metadata or {}),
            artifacts=self._current_workflow_handoff_artifacts(
                session_state=session_state,
                execution_state=execution_state,
            ),
        )
        if digest.estimated_chars >= 24000:
            limit = min(limit, 2)
        return max(1, limit)

    @staticmethod
    def _feedback_signature(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip()).casefold()

    @staticmethod
    def _answer_signature(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip()).casefold()

    def recent_context_summary(self, session_state: SessionState, limit: int = 4) -> str:
        rows: List[str] = []
        for message in filter_context_messages(session_state.messages[-limit:]):
            if message.role not in {"user", "assistant"} or not message.content.strip():
                continue
            rows.append(f"{message.role}: {message.content[:300]}")
        summary = "\n".join(rows)
        manager = getattr(self.kernel, "context_budget_manager", None)
        if manager is not None:
            return manager.budget_text_block("coordinator_recent_context", summary, max_tokens=600)
        return summary

    def build_worker_request(
        self,
        *,
        task: Dict[str, Any],
        user_request: str,
        session_state: SessionState,
        artifact_refs: Optional[List[str]] = None,
        handoff_artifacts: Optional[List[Dict[str, Any]]] = None,
    ) -> WorkerExecutionRequest:
        doc_scope = [str(item) for item in (task.get("doc_scope") or []) if str(item)]
        skill_queries = [str(item) for item in (task.get("skill_queries") or []) if str(item)]
        research_profile = str(task.get("research_profile") or "").strip()
        coverage_goal = str(task.get("coverage_goal") or "").strip()
        result_mode = str(task.get("result_mode") or "").strip()
        answer_mode = str(task.get("answer_mode") or "answer").strip().lower() or "answer"
        controller_hints = dict(task.get("controller_hints") or {})
        capability_requirements = dict(task.get("capability_requirements") or {})
        evidence_scope = dict(task.get("evidence_scope") or {})
        acceptance_criteria = [str(item) for item in (task.get("acceptance_criteria") or []) if str(item)]
        produces_artifacts = [str(item) for item in (task.get("produces_artifacts") or []) if str(item)]
        consumes_artifacts = [str(item) for item in (task.get("consumes_artifacts") or []) if str(item)]
        handoff_schema = str(task.get("handoff_schema") or "")
        input_artifact_ids = [str(item) for item in (task.get("input_artifact_ids") or []) if str(item)]
        context_summary = self.recent_context_summary(session_state)
        resolved_turn_intent = resolved_turn_intent_from_metadata(dict(session_state.metadata or {}))
        parts = [
            "You are executing a scoped task delegated by a coordinator.",
            "Work only from the task brief below. Do not assume you have the full parent conversation.",
            f"ORIGINAL_USER_REQUEST:\n{user_request}",
            f"TASK_ID: {task.get('id', '')}",
            f"TASK_TITLE: {task.get('title', '')}",
            f"TASK_INPUT:\n{task.get('input', '')}",
        ]
        if doc_scope:
            parts.append("DOCUMENT_SCOPE:\n- " + "\n- ".join(doc_scope))
        if skill_queries:
            parts.append("SKILL_HINTS:\n- " + "\n- ".join(skill_queries))
        if research_profile:
            parts.append(f"RESEARCH_PROFILE:\n{research_profile}")
        if coverage_goal:
            parts.append(f"COVERAGE_GOAL:\n{coverage_goal}")
        if result_mode:
            parts.append(f"RESULT_MODE:\n{result_mode}")
        if answer_mode:
            parts.append(f"ANSWER_MODE:\n{answer_mode}")
        if controller_hints:
            parts.append("CONTROLLER_HINTS:\n" + json.dumps(controller_hints, ensure_ascii=False, indent=2))
        if capability_requirements:
            parts.append("CAPABILITY_REQUIREMENTS:\n" + json.dumps(capability_requirements, ensure_ascii=False, indent=2))
        if evidence_scope:
            parts.append("EVIDENCE_SCOPE:\n" + json.dumps(evidence_scope, ensure_ascii=False, indent=2))
        if str(task.get("loop_over_artifact") or "").strip():
            parts.append(f"LOOP_OVER_ARTIFACT:\n{str(task.get('loop_over_artifact') or '').strip()}")
        if acceptance_criteria:
            parts.append("ACCEPTANCE_CRITERIA:\n- " + "\n- ".join(acceptance_criteria))
        if bool(controller_hints.get("strict_doc_focus")):
            parts.append(
                "STRICT_SCOPE_RULE:\n"
                "Only inspect and reason from the exact DOCUMENT_SCOPE. Do not broaden to other knowledge-base documents."
            )
        if str(controller_hints.get("doc_read_depth") or "").strip().lower() == "full":
            parts.append(
                "READ_DEPTH_RULE:\n"
                "Use read_indexed_doc(mode=\"full\") with cursor pagination until `has_more` is false or every major outline section has been covered. "
                "Do not rely on `overview` mode alone for this task."
            )
        if artifact_refs:
            parts.append("AVAILABLE_ARTIFACTS:\n- " + "\n- ".join(artifact_refs))
        if handoff_artifacts:
            parts.append(
                "STRUCTURED_HANDOFFS:\n"
                + "\n".join(
                    f"- {artifact.get('artifact_type')}: {json.dumps(dict(artifact.get('data') or {}), ensure_ascii=False)[:1000]}"
                    for artifact in handoff_artifacts
                )
            )
        if context_summary:
            parts.append(f"RECENT_PARENT_CONTEXT:\n{context_summary}")
        parts.append("Return a focused result for this task only.")
        prompt = "\n\n".join(part for part in parts if part.strip())
        metadata: Dict[str, Any] = {
            "task_spec": dict(task),
            "handoff_artifacts": [dict(item) for item in (handoff_artifacts or [])],
            "answer_mode": answer_mode,
            "semantic_query": str(task.get("input") or ""),
            "instruction_prompt": prompt,
        }
        if resolved_turn_intent is not None:
            metadata["resolved_turn_intent"] = resolved_turn_intent.to_dict()
        if str(task.get("executor") or "") == "rag_worker":
            metadata["rag_search_task"] = {
                "task_id": str(task.get("id") or ""),
                "title": str(task.get("title") or ""),
                "query": str(task.get("input") or ""),
                "doc_scope": list(doc_scope),
                "strategies": [str(item) for item in (controller_hints.get("retrieval_strategies") or []) if str(item)],
                "round_budget": max(1, int(controller_hints.get("round_budget") or 1)),
                "answer_mode": answer_mode,
                "research_profile": research_profile,
                "coverage_goal": coverage_goal,
                "result_mode": result_mode,
                "controller_hints": dict(controller_hints),
            }
        return WorkerExecutionRequest(
            agent_name=str(task.get("executor") or "general"),
            task_id=str(task.get("id") or ""),
            title=str(task.get("title") or ""),
            prompt=prompt,
            instruction_prompt=prompt,
            semantic_query=str(task.get("input") or ""),
            context_summary=context_summary,
            description=str(task.get("title") or task.get("input") or "")[:120],
            doc_scope=doc_scope,
            skill_queries=skill_queries,
            research_profile=research_profile,
            coverage_goal=coverage_goal,
            result_mode=result_mode,
            answer_mode=answer_mode,
            controller_hints=controller_hints,
            artifact_refs=list(artifact_refs or []),
            produces_artifacts=produces_artifacts,
            consumes_artifacts=consumes_artifacts,
            handoff_schema=handoff_schema,
            input_artifact_ids=input_artifact_ids,
            metadata=metadata,
        )

    def run_task_batch(
        self,
        *,
        agent: AgentDefinition,
        session_state: SessionState,
        user_request: str,
        callbacks: List[Any],
        batch: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        del callbacks
        artifact_refs: List[str] = []
        jobs: List[tuple[str, WorkerExecutionRequest, Any]] = []
        team_channel_id = ""
        if bool(getattr(self.kernel.settings, "team_mailbox_enabled", False)) and len(batch) > 1:
            member_agents = [
                str(task.get("executor") or "").strip()
                for task in batch
                if str(task.get("executor") or "").strip()
            ]
            try:
                channel = self.kernel.job_manager.create_team_channel(
                    session_id=session_state.session_id,
                    name="coordinator-campaign",
                    purpose=user_request[:240],
                    created_by_job_id="",
                    member_agents=member_agents,
                    member_job_ids=[],
                    metadata={"created_by_agent": agent.name, "source": "coordinator"},
                )
                team_channel_id = channel.channel_id
            except ValueError:
                team_channel_id = ""
        for task in batch:
            handoff_artifacts = self._resolve_task_handoffs(
                task=task,
                session_state=session_state,
                consumer_agent=str(task.get("executor") or "general"),
            )
            worker_request = self.build_worker_request(
                task=task,
                user_request=user_request,
                session_state=session_state,
                artifact_refs=artifact_refs,
                handoff_artifacts=handoff_artifacts,
            )
            staged_workspace_files = self._materialize_workspace_sources(
                session_state=session_state,
                task=task,
                handoff_artifacts=handoff_artifacts,
                worker_request=worker_request,
            )
            if staged_workspace_files:
                worker_request.prompt = (
                    worker_request.prompt
                    + "\n\nSESSION_WORKSPACE_FILES:\n- "
                    + "\n- ".join(staged_workspace_files)
                    + "\nUse files from /workspace when you need structured extraction."
                )
                worker_request.instruction_prompt = worker_request.prompt
                worker_request.metadata["workspace_files"] = list(staged_workspace_files)
            if team_channel_id:
                worker_request.metadata["team_channel_id"] = team_channel_id
                worker_request.controller_hints["team_channel_id"] = team_channel_id
            if handoff_artifacts:
                self.kernel._emit(
                    "worker_handoff_consumed",
                    session_state.session_id,
                    agent_name=worker_request.agent_name,
                    payload={
                        "conversation_id": session_state.conversation_id,
                        "task_id": worker_request.task_id,
                        "artifact_ids": [str(item.get("artifact_id") or "") for item in handoff_artifacts],
                        "artifact_types": [str(item.get("artifact_type") or "") for item in handoff_artifacts],
                        "handoff_schema": worker_request.handoff_schema,
                    },
                )
            if worker_request.agent_name not in set(agent.allowed_worker_agents):
                result = TaskResult(
                    task_id=worker_request.task_id,
                    title=worker_request.title,
                    executor=worker_request.agent_name,
                    status="failed",
                    output=f"Worker '{worker_request.agent_name}' is not allowed for coordinator execution.",
                    artifact_ref=str(task.get("artifact_ref") or f"task:{worker_request.task_id}"),
                    warnings=[f"Agent '{worker_request.agent_name}' is not allowed."],
                )
                artifact_refs.append(result.artifact_ref)
                jobs.append(("synthetic", worker_request, result))
                continue
            scoped_state = self.build_scoped_worker_state(session_state, agent_name=worker_request.agent_name)
            self._apply_worker_scope_overrides(
                scoped_state=scoped_state,
                session_state=session_state,
                worker_request=worker_request,
            )
            job = self.kernel.job_manager.create_job(
                agent_name=worker_request.agent_name,
                prompt=worker_request.prompt,
                session_id=session_state.session_id,
                description=worker_request.description,
                tenant_id=session_state.tenant_id,
                user_id=session_state.user_id,
                priority=str(worker_request.controller_hints.get("priority") or "interactive"),
                queue_class=str(worker_request.controller_hints.get("queue_class") or worker_request.controller_hints.get("priority") or "interactive"),
                session_state=scoped_state.to_dict(),
                metadata={
                    "session_state": scoped_state.to_dict(),
                    "worker_request": worker_request.to_dict(),
                    "team_channel_id": team_channel_id,
                    "route_context": dict(session_state.metadata.get("route_context") or {}),
                },
            )
            if team_channel_id:
                channel = self.kernel.job_manager.transcript_store.load_team_channel(session_state.session_id, team_channel_id)
                if channel is not None and job.job_id not in set(channel.member_job_ids):
                    channels = self.kernel.job_manager.transcript_store.load_team_channels(session_state.session_id)
                    for existing in channels:
                        if existing.channel_id == team_channel_id:
                            existing.member_job_ids.append(job.job_id)
                            existing.member_job_ids = list(dict.fromkeys(existing.member_job_ids))
                            break
                    self.kernel.job_manager.transcript_store.overwrite_team_channels(session_state.session_id, channels)
                self.kernel.job_manager.post_team_message(
                    session_id=session_state.session_id,
                    channel_id=team_channel_id,
                    content=worker_request.description or worker_request.title,
                    source_agent=agent.name,
                    source_job_id="",
                    target_agents=[worker_request.agent_name],
                    target_job_ids=[job.job_id],
                    message_type="handoff",
                    subject=worker_request.title,
                    payload={"task_id": worker_request.task_id, "job_id": job.job_id},
                    metadata={"source": "coordinator"},
                )
            jobs.append((job.job_id, worker_request, job))

        real_jobs = [job for job_id, _, job in jobs if job_id != "synthetic"]
        run_parallel = self.should_run_task_batch_in_parallel(batch=batch, real_jobs=real_jobs)
        if run_parallel:
            for job in real_jobs:
                self.kernel.job_manager.start_background_job(job, self.kernel._job_runner)
            self.wait_for_jobs([job.job_id for job in real_jobs])
        else:
            for job in real_jobs:
                self.kernel.job_manager.run_job_inline(job, self.kernel._job_runner)

        results: List[Dict[str, Any]] = []
        for job_id, worker_request, record in jobs:
            if job_id == "synthetic":
                results.append(record.to_dict())
                continue
            job = self.kernel.job_manager.get_job(job_id) or record
            result = self.build_task_result(job, worker_request)
            self._prepare_handoff_artifacts(
                session_state=session_state,
                task=dict(worker_request.metadata.get("task_spec") or {}),
                result=result,
            )
            artifact_refs.append(result.artifact_ref)
            results.append(result.to_dict())
        return results

    def should_run_task_batch_in_parallel(self, *, batch: List[Dict[str, Any]], real_jobs: List[Any]) -> bool:
        if len(real_jobs) <= 1:
            return False
        if not all(str(task.get("mode", "sequential")) == "parallel" for task in batch):
            return False
        for task in batch:
            executor = str(task.get("executor") or "").strip()
            if not executor:
                return False
            try:
                if not self.kernel._resolve_agent(executor).allow_background_jobs:
                    return False
            except ValueError:
                return False
        if self.kernel._uses_local_ollama_workers():
            return False
        return self.kernel._can_identify_non_ollama_worker_runtime()

    def wait_for_jobs(self, job_ids: List[str], *, timeout_seconds: float = 300.0) -> None:
        if timeout_seconds == 300.0:
            timeout_seconds = float(getattr(self.kernel.settings, "worker_job_wait_timeout_seconds", 600.0))
        deadline = time.monotonic() + timeout_seconds
        pending = set(job_ids)
        while pending:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for jobs: {sorted(pending)}")
            completed = set()
            for job_id in pending:
                job = self.kernel.job_manager.get_job(job_id)
                if job is not None and job.status in TERMINAL_TASK_STATUSES:
                    completed.add(job_id)
            pending -= completed
            if pending:
                time.sleep(0.02)

    def build_task_result(self, job: JobRecord, worker_request: WorkerExecutionRequest) -> TaskResult:
        warnings: List[str] = []
        if job.last_error:
            warnings.append(str(job.last_error))
        output = str(job.result_summary or "")
        if job.output_path:
            try:
                output = Path(job.output_path).read_text(encoding="utf-8")
            except Exception:
                output = output or str(job.result_summary or "")
        metadata = dict(job.metadata.get("result_metadata") or {})
        if job.status == "waiting_message":
            metadata["worker_mailbox_request"] = dict(job.metadata.get("pending_mailbox_request") or {})
            metadata["turn_outcome"] = "worker_mailbox_request"
        metadata.setdefault("worker_request", worker_request.to_dict())
        metadata.setdefault(
            "evidence_provenance",
            infer_result_provenance(
                {
                    "metadata": metadata,
                    "output": output,
                }
            ),
        )
        return TaskResult(
            task_id=worker_request.task_id,
            title=worker_request.title,
            executor=worker_request.agent_name,
            status=str(job.status or "failed"),
            output=output,
            artifact_ref=job.output_path or f"task:{worker_request.task_id}",
            warnings=warnings,
            metadata=metadata,
        )

    def build_partial_answer(self, task_results: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for result in task_results:
            title = str(result.get("title") or result.get("task_id") or "Task")
            output = str(result.get("output") or "").strip()
            if not output:
                continue
            parts.append(f"{title}:\n{output}")
        return "\n\n".join(parts).strip()

    def _first_task_clarification(self, task_results: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        for result in task_results:
            metadata = dict(result.get("metadata") or {})
            if is_clarification_turn(metadata):
                return result
            if metadata.get("worker_mailbox_request"):
                return result
        return None

    def parse_verification_result(self, result: Any) -> VerificationResult:
        payload = dict(result.metadata.get("verification") or {})
        parse_failed = False
        if not payload:
            payload = extract_json(result.text or "") or {}
            parse_failed = not bool(payload) and bool(str(getattr(result, "text", "") or "").strip())
        raw_status = str(payload.get("verdict") or payload.get("status") or "pass").strip()
        normalized_status = raw_status.lower()
        verdict = raw_status.upper()
        if verdict in {"PASS", "FAIL", "PARTIAL"}:
            status = "pass" if verdict == "PASS" else "revise"
        else:
            if normalized_status not in {"pass", "revise"}:
                normalized_status = "pass"
            status = normalized_status
            verdict = "PASS" if status == "pass" else "PARTIAL"
        summary = str(payload.get("summary") or result.text or "").strip()
        issues = [str(item) for item in (payload.get("issues") or []) if str(item)]
        feedback = str(payload.get("feedback") or "\n".join(issues) or summary).strip()
        return VerificationResult(
            status=status,
            verdict=verdict,
            summary=summary,
            issues=issues,
            feedback=feedback,
            parse_failed=bool(payload.get("parse_failed")) or parse_failed,
        )


__all__ = ["KernelCoordinatorController"]
