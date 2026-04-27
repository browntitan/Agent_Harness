from __future__ import annotations

import fnmatch
import json
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from agentic_chatbot_next.documents.consolidation_models import (
    CampaignDocumentRecord,
    ConsolidationCluster,
    DocumentConsolidationCampaignResult,
    DocumentSimilarityEdge,
    ProcessFlowMatch,
    SectorSummary,
)
from agentic_chatbot_next.documents.extractors import (
    DocumentExtractionService,
    DocumentResolutionError,
    SUPPORTED_FILE_TYPES,
)
from agentic_chatbot_next.documents.models import DocumentExtractResult
from agentic_chatbot_next.documents.similarity import (
    build_idf,
    build_suppressed_terms,
    compare_fingerprints,
    fingerprint_document,
)
from agentic_chatbot_next.runtime.artifacts import register_handoff_artifact, register_workspace_artifact


_GENERIC_PATH_SEGMENTS = {
    "data",
    "doc",
    "docs",
    "document",
    "documents",
    "kb",
    "knowledge-base",
    "knowledge_base",
    "policy",
    "policies",
    "procedure",
    "procedures",
    "upload",
    "uploads",
    "workspace",
    "workspaces",
}
_CAMPAIGN_ARTIFACT_MARKERS = {
    "__document_extract__",
    "__document_compare",
    "changed_obligations",
    "consolidation_manifest",
    "document_similarity_edges",
    "process_flow_matches",
    "consolidation_clusters",
    "sector_summary",
    "document_consolidation_report",
    "document_consolidation_checkpoint",
}
_TITLE_PREFIX_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 &/]{1,48})\s*(?:[-_:|]|--)\s+")


@dataclass(frozen=True)
class _CandidateDocument:
    ref: str
    source_scope: str
    source_path: str = ""
    source_type: str = ""
    collection_id: str = ""
    title: str = ""


def _tenant_id(settings: object, session: object) -> str:
    return str(getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev")) or "local-dev")


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalise_scope(value: str) -> str:
    scope = str(value or "auto").strip().lower()
    return scope if scope in {"auto", "uploads", "kb", "workspace"} else "auto"


def _normalise_sector_mode(value: str) -> str:
    mode = str(value or "infer").strip().lower()
    allowed = {"infer", "collection_id", "path_segment", "title_prefix", "explicit_map", "none"}
    return mode if mode in allowed else "infer"


def _normalise_cross_sector_mode(value: str, *, allow_cross_sector_comparisons: bool) -> str:
    mode = str(value or "blocked").strip().lower()
    if mode not in {"blocked", "report_only", "allowed"}:
        mode = "blocked"
    if allow_cross_sector_comparisons and mode == "blocked":
        return "allowed"
    return mode


def _normalise_focus(value: str) -> str:
    focus = str(value or "auto").strip().lower()
    allowed = {"auto", "process_flows", "policies", "tables", "requirements", "full_text"}
    return focus if focus in allowed else "auto"


def _record_attr(record: Any, name: str) -> str:
    return _clean(getattr(record, name, "") if record is not None else "")


def _source_scope_for_record(record: Any, fallback: str) -> str:
    source_type = _record_attr(record, "source_type").casefold()
    if source_type == "upload":
        return "uploads"
    if source_type == "kb":
        return "kb"
    return fallback if fallback in {"uploads", "kb"} else "auto"


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _artifact_write(session: object, filename: str, content: str) -> Dict[str, Any]:
    workspace = getattr(session, "workspace", None)
    if workspace is None:
        workspace_root = _clean(getattr(session, "workspace_root", ""))
        session_id = _clean(getattr(session, "session_id", ""))
        if not workspace_root or not session_id:
            raise ValueError("No session workspace is available.")
        from agentic_chatbot_next.sandbox.workspace import SessionWorkspace

        workspace = SessionWorkspace(session_id=session_id, root=Path(workspace_root))
        workspace.open()
        session.workspace = workspace
    workspace.write_text(filename, content)
    return register_workspace_artifact(session, filename=filename, label=filename)


class DocumentConsolidationCampaignService:
    """Corpus-level read-only consolidation candidate discovery."""

    def __init__(
        self,
        settings: object,
        stores: object,
        session: object,
        *,
        event_sink: object | None = None,
    ) -> None:
        self.settings = settings
        self.stores = stores
        self.session = session
        self.event_sink = event_sink
        self.tenant_id = _tenant_id(settings, session)

    def run(
        self,
        *,
        query: str = "",
        source_scope: str = "auto",
        collection_id: str = "",
        document_refs: Sequence[str] | None = None,
        sector_mode: str = "infer",
        sector_map: Dict[str, str] | None = None,
        allow_cross_sector_comparisons: bool = False,
        cross_sector_mode: str = "blocked",
        similarity_focus: str = "auto",
        min_similarity_score: float = 0.72,
        max_documents: int = 100,
        max_candidate_pairs: int = 150,
        export: bool = True,
        campaign_id: str = "",
    ) -> DocumentConsolidationCampaignResult:
        campaign_id = campaign_id or f"doc_consolidation_{uuid.uuid4().hex[:12]}"
        scope = _normalise_scope(source_scope)
        mode = _normalise_sector_mode(sector_mode)
        cross_mode = _normalise_cross_sector_mode(
            cross_sector_mode,
            allow_cross_sector_comparisons=allow_cross_sector_comparisons,
        )
        focus = _normalise_focus(similarity_focus)
        min_score = max(0.0, min(1.0, _safe_float(min_similarity_score, 0.72)))
        max_docs = _safe_int(max_documents, 100, minimum=1, maximum=1000)
        max_pairs = _safe_int(max_candidate_pairs, 150, minimum=1, maximum=5000)
        warnings: list[str] = []

        candidates = self._select_candidates(
            source_scope=scope,
            collection_id=_clean(collection_id),
            document_refs=document_refs,
            max_documents=max_docs,
            warnings=warnings,
        )
        if len(candidates) >= max_docs:
            warnings.append(f"Document selection was capped at max_documents={max_docs}.")

        extracted: list[DocumentExtractResult] = []
        manifest_by_doc_id: dict[str, CampaignDocumentRecord] = {}
        extractor = DocumentExtractionService(self.settings, self.stores, self.session)
        for candidate in candidates:
            try:
                result = extractor.extract(
                    document_ref=candidate.ref,
                    source_scope=candidate.source_scope,
                    collection_id=candidate.collection_id or _clean(collection_id),
                    include_tables=True,
                    include_figures=False,
                    include_metadata=True,
                    include_hierarchy=True,
                    max_elements=800,
                )
                sector, sector_source = self._infer_sector(
                    result,
                    candidate=candidate,
                    sector_mode=mode,
                    sector_map=dict(sector_map or {}),
                    requested_collection_id=_clean(collection_id),
                )
                record = CampaignDocumentRecord(
                    doc_id=result.document.doc_id or candidate.ref,
                    title=result.document.title or candidate.title or candidate.ref,
                    sector=sector,
                    sector_source=sector_source,
                    source_type=result.document.source_type or candidate.source_type,
                    source_path=result.document.source_path or candidate.source_path,
                    collection_id=result.document.collection_id or candidate.collection_id or _clean(collection_id),
                    file_type=result.document.file_type,
                    content_hash=result.document.content_hash,
                    extraction_status="extracted",
                    warnings=list(result.warnings),
                )
                extracted.append(result)
                manifest_by_doc_id[record.doc_id] = record
            except DocumentResolutionError as exc:
                payload = dict(exc.payload)
                warnings.append(_clean(payload.get("error") or f"Could not resolve {candidate.ref}."))
                manifest_by_doc_id[candidate.ref] = CampaignDocumentRecord(
                    doc_id=candidate.ref,
                    title=candidate.title or candidate.ref,
                    sector="unknown" if mode != "none" else "all",
                    sector_source="resolution_error",
                    source_type=candidate.source_type,
                    source_path=candidate.source_path,
                    collection_id=candidate.collection_id or _clean(collection_id),
                    extraction_status="failed",
                    warnings=[json.dumps(payload, sort_keys=True)],
                )
            except Exception as exc:
                warnings.append(f"Extraction failed for {candidate.ref}: {exc}")
                manifest_by_doc_id[candidate.ref] = CampaignDocumentRecord(
                    doc_id=candidate.ref,
                    title=candidate.title or candidate.ref,
                    sector="unknown" if mode != "none" else "all",
                    sector_source="extraction_error",
                    source_type=candidate.source_type,
                    source_path=candidate.source_path,
                    collection_id=candidate.collection_id or _clean(collection_id),
                    extraction_status="failed",
                    warnings=[str(exc)],
                )

        manifest = list(manifest_by_doc_id.values())
        sector_summary = self._sector_summary(manifest, cross_mode)
        if not extracted:
            warnings.append("No documents were successfully extracted for consolidation analysis.")

        edges, process_matches = self._score_candidates(
            extracted,
            manifest_by_doc_id=manifest_by_doc_id,
            cross_sector_mode=cross_mode,
            similarity_focus=focus,
            min_similarity_score=min_score,
            max_candidate_pairs=max_pairs,
        )
        clusters = self._build_clusters(edges, manifest_by_doc_id)
        result = DocumentConsolidationCampaignResult(
            campaign_id=campaign_id,
            status="completed",
            query=_clean(query),
            selected_document_count=len(manifest),
            manifest=manifest,
            sector_summary=sector_summary,
            similarity_edges=edges,
            process_flow_matches=process_matches,
            consolidation_clusters=clusters,
            warnings=warnings,
        )
        if export:
            result.artifacts = self._write_artifacts(result)
            self._register_handoff_artifacts(result)
        return result

    def _select_candidates(
        self,
        *,
        source_scope: str,
        collection_id: str,
        document_refs: Sequence[str] | None,
        max_documents: int,
        warnings: list[str],
    ) -> list[_CandidateDocument]:
        explicit_refs = [_clean(item) for item in list(document_refs or []) if _clean(item)]
        if explicit_refs:
            return [
                _CandidateDocument(
                    ref=ref,
                    source_scope=source_scope,
                    collection_id=collection_id,
                    title=Path(ref).name,
                )
                for ref in explicit_refs[:max_documents]
            ]

        if source_scope in {"auto", "workspace"}:
            workspace_candidates = self._workspace_candidates()
            if workspace_candidates:
                return workspace_candidates[:max_documents]
            if source_scope == "workspace":
                warnings.append("No supported workspace documents were found for consolidation analysis.")
                return []

        records = self._list_indexed_records(source_scope=source_scope, collection_id=collection_id)
        if not records:
            warnings.append("No indexed documents were found for consolidation analysis.")
            return []
        candidates: list[_CandidateDocument] = []
        for record in records[:max_documents]:
            doc_id = _record_attr(record, "doc_id")
            source_path = _record_attr(record, "source_path")
            title = _record_attr(record, "title") or Path(source_path).name or doc_id
            candidates.append(
                _CandidateDocument(
                    ref=doc_id or title,
                    source_scope=_source_scope_for_record(record, source_scope),
                    source_path=source_path,
                    source_type=_record_attr(record, "source_type"),
                    collection_id=_record_attr(record, "collection_id") or collection_id,
                    title=title,
                )
            )
        return candidates

    def _workspace_candidates(self) -> list[_CandidateDocument]:
        workspace = getattr(self.session, "workspace", None)
        root = Path(getattr(workspace, "root", "")) if workspace is not None else None
        if root is None or not root.exists():
            root_text = _clean(getattr(self.session, "workspace_root", ""))
            root = Path(root_text) if root_text else None
        if root is None or not root.exists():
            return []
        try:
            filenames = sorted(item.name for item in root.iterdir() if item.is_file() and not item.name.startswith("."))
        except Exception:
            return []
        candidates: list[_CandidateDocument] = []
        for filename in filenames:
            if self._is_generated_artifact(filename):
                continue
            file_type = Path(filename).suffix.lower().lstrip(".")
            if file_type not in SUPPORTED_FILE_TYPES:
                continue
            path = root / filename
            candidates.append(
                _CandidateDocument(
                    ref=filename,
                    source_scope="workspace",
                    source_path=str(path),
                    source_type="workspace",
                    title=filename,
                )
            )
        return candidates

    def _is_generated_artifact(self, filename: str) -> bool:
        lowered = filename.casefold()
        return any(marker in lowered for marker in _CAMPAIGN_ARTIFACT_MARKERS)

    def _list_indexed_records(self, *, source_scope: str, collection_id: str) -> list[Any]:
        doc_store = getattr(self.stores, "doc_store", None)
        list_documents = getattr(doc_store, "list_documents", None)
        if list_documents is None:
            return []
        source_type = ""
        if source_scope == "uploads":
            source_type = "upload"
        elif source_scope == "kb":
            source_type = "kb"

        attempts = [
            {"tenant_id": self.tenant_id, "collection_id": collection_id, "source_type": source_type},
            {"tenant_id": self.tenant_id, "collection_id": collection_id},
            {"tenant_id": self.tenant_id, "source_type": source_type},
            {"tenant_id": self.tenant_id},
        ]
        records: list[Any] = []
        for kwargs in attempts:
            clean_kwargs = {key: value for key, value in kwargs.items() if value or key == "tenant_id"}
            try:
                records = list(list_documents(**clean_kwargs) or [])
                break
            except TypeError:
                continue
            except Exception:
                return []
        if not records:
            try:
                records = list(list_documents(self.tenant_id, collection_id, source_type) or [])
            except Exception:
                records = []
        filtered: list[Any] = []
        for record in records:
            if collection_id and _record_attr(record, "collection_id") != collection_id:
                continue
            if source_type and _record_attr(record, "source_type").casefold() != source_type:
                continue
            filtered.append(record)
        return filtered

    def _infer_sector(
        self,
        result: DocumentExtractResult,
        *,
        candidate: _CandidateDocument,
        sector_mode: str,
        sector_map: Dict[str, str],
        requested_collection_id: str,
    ) -> tuple[str, str]:
        if sector_mode == "none":
            return "all", "none"

        explicit = self._sector_from_explicit_map(result, candidate, sector_map)
        if explicit:
            return explicit, "explicit_map"
        if sector_mode == "explicit_map":
            return "unknown", "explicit_map_missing"

        metadata_sector = self._sector_from_metadata(result)
        collection_sector = _clean(requested_collection_id or result.document.collection_id or candidate.collection_id)
        path_sector = self._sector_from_path(result.document.source_path or candidate.source_path)
        title_sector = self._sector_from_title(result.document.title or candidate.title)

        if sector_mode == "collection_id":
            return (collection_sector or "unknown", "collection_id" if collection_sector else "unknown")
        if sector_mode == "path_segment":
            return (path_sector or "unknown", "path_segment" if path_sector else "unknown")
        if sector_mode == "title_prefix":
            return (title_sector or "unknown", "title_prefix" if title_sector else "unknown")

        if metadata_sector:
            return metadata_sector, "metadata"
        if collection_sector:
            return collection_sector, "collection_id"
        if path_sector:
            return path_sector, "path_segment"
        if title_sector:
            return title_sector, "title_prefix"
        return "unknown", "unknown"

    def _sector_from_explicit_map(
        self,
        result: DocumentExtractResult,
        candidate: _CandidateDocument,
        sector_map: Dict[str, str],
    ) -> str:
        if not sector_map:
            return ""
        fields = {
            candidate.ref,
            candidate.title,
            candidate.source_path,
            Path(candidate.source_path).name if candidate.source_path else "",
            result.document.doc_id,
            result.document.title,
            result.document.source_path,
            Path(result.document.source_path).name if result.document.source_path else "",
        }
        values = {item.casefold() for item in fields if item}
        for pattern, sector in sector_map.items():
            clean_sector = _clean(sector)
            clean_pattern = _clean(pattern)
            if not clean_sector or not clean_pattern:
                continue
            lowered = clean_pattern.casefold()
            if lowered in values:
                return clean_sector
            if any(fnmatch.fnmatch(value, lowered) for value in values):
                return clean_sector
        return ""

    def _sector_from_metadata(self, result: DocumentExtractResult) -> str:
        for key in ("sector", "business_unit", "division", "department", "organization", "organisation"):
            value = _clean(result.metadata.get(key))
            if value:
                return value
        return ""

    def _sector_from_path(self, source_path: str) -> str:
        if not source_path:
            return ""
        path = Path(source_path)
        for segment in reversed(path.parts[:-1]):
            clean = _clean(segment)
            if clean and clean.casefold() not in _GENERIC_PATH_SEGMENTS and not clean.startswith("."):
                return clean
        return ""

    def _sector_from_title(self, title: str) -> str:
        match = _TITLE_PREFIX_RE.match(title or "")
        if not match:
            return ""
        value = _clean(match.group(1))
        if value.casefold() in _GENERIC_PATH_SEGMENTS:
            return ""
        return value

    def _sector_summary(self, manifest: Sequence[CampaignDocumentRecord], cross_sector_mode: str) -> SectorSummary:
        counts: dict[str, int] = defaultdict(int)
        unknown: list[str] = []
        for record in manifest:
            sector = record.sector or "unknown"
            counts[sector] += 1
            if sector == "unknown":
                unknown.append(record.title or record.doc_id)
        return SectorSummary(
            sectors=dict(sorted(counts.items())),
            unknown_documents=sorted(unknown),
            cross_sector_mode=cross_sector_mode,
        )

    def _score_candidates(
        self,
        results: Sequence[DocumentExtractResult],
        *,
        manifest_by_doc_id: Dict[str, CampaignDocumentRecord],
        cross_sector_mode: str,
        similarity_focus: str,
        min_similarity_score: float,
        max_candidate_pairs: int,
    ) -> tuple[list[DocumentSimilarityEdge], list[ProcessFlowMatch]]:
        if len(results) < 2:
            return [], []
        records = [
            manifest_by_doc_id.get(result.document.doc_id)
            or manifest_by_doc_id.get(result.document.title)
            or CampaignDocumentRecord(
                doc_id=result.document.doc_id,
                title=result.document.title,
                sector="unknown",
                sector_source="missing_manifest",
            )
            for result in results
        ]
        extra_terms = self._configured_boilerplate_terms()
        extra_terms.extend(record.sector for record in records if record.sector)
        suppressed_terms = build_suppressed_terms(results, extra_terms=extra_terms)
        idf = build_idf(results, suppressed_terms=suppressed_terms)
        fingerprints = {
            result.document.doc_id: fingerprint_document(
                result,
                sector=(manifest_by_doc_id.get(result.document.doc_id).sector if manifest_by_doc_id.get(result.document.doc_id) else "unknown"),
                suppressed_terms=suppressed_terms,
            )
            for result in results
        }

        edges: list[DocumentSimilarityEdge] = []
        process_matches: list[ProcessFlowMatch] = []
        for index, (left_result, right_result) in enumerate(combinations(results, 2), start=1):
            left_record = manifest_by_doc_id.get(left_result.document.doc_id)
            right_record = manifest_by_doc_id.get(right_result.document.doc_id)
            if left_record is None or right_record is None:
                continue
            cross_sector = left_record.sector != right_record.sector
            if cross_sector and cross_sector_mode == "blocked":
                continue
            score = compare_fingerprints(
                fingerprints[left_result.document.doc_id],
                fingerprints[right_result.document.doc_id],
                idf=idf,
                similarity_focus=similarity_focus,
                cross_sector=cross_sector,
            )
            if score.consolidation_score < min_similarity_score:
                continue
            edge_id = f"edge_{index:05d}"
            advisory = bool(cross_sector and cross_sector_mode == "report_only")
            edge = DocumentSimilarityEdge(
                edge_id=edge_id,
                left_doc_id=left_record.doc_id,
                right_doc_id=right_record.doc_id,
                left_title=left_record.title,
                right_title=right_record.title,
                left_sector=left_record.sector,
                right_sector=right_record.sector,
                consolidation_score=score.consolidation_score,
                content_overlap_score=score.content_overlap_score,
                process_flow_score=score.process_flow_score,
                section_structure_score=score.section_structure_score,
                table_schema_score=score.table_schema_score,
                obligation_overlap_score=score.obligation_overlap_score,
                metadata_title_score=score.metadata_title_score,
                reason_codes=score.reason_codes,
                shared_terms=score.shared_terms,
                cross_sector=cross_sector,
                cross_sector_advisory=advisory,
            )
            edges.append(edge)
            if score.matched_left_steps and score.matched_right_steps and score.process_flow_score >= 0.52:
                process_matches.append(
                    ProcessFlowMatch(
                        match_id=f"flow_{len(process_matches) + 1:05d}",
                        left_doc_id=left_record.doc_id,
                        right_doc_id=right_record.doc_id,
                        left_title=left_record.title,
                        right_title=right_record.title,
                        left_sector=left_record.sector,
                        right_sector=right_record.sector,
                        score=score.process_flow_score,
                        matched_left_steps=score.matched_left_steps,
                        matched_right_steps=score.matched_right_steps,
                        cross_sector_advisory=advisory,
                    )
                )

        edges.sort(key=lambda item: (item.consolidation_score, item.process_flow_score), reverse=True)
        process_matches.sort(key=lambda item: item.score, reverse=True)
        return edges[:max_candidate_pairs], process_matches[:max_candidate_pairs]

    def _configured_boilerplate_terms(self) -> list[str]:
        value = getattr(self.settings, "document_consolidation_boilerplate_terms", "")
        if isinstance(value, str):
            return [_clean(item).casefold() for item in value.split(",") if _clean(item)]
        if isinstance(value, Iterable):
            return [_clean(item).casefold() for item in value if _clean(item)]
        return []

    def _build_clusters(
        self,
        edges: Sequence[DocumentSimilarityEdge],
        manifest_by_doc_id: Dict[str, CampaignDocumentRecord],
    ) -> list[ConsolidationCluster]:
        usable_edges = [edge for edge in edges if not edge.cross_sector_advisory]
        parent: dict[str, str] = {}

        def find(value: str) -> str:
            parent.setdefault(value, value)
            if parent[value] != value:
                parent[value] = find(parent[value])
            return parent[value]

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for edge in usable_edges:
            union(edge.left_doc_id, edge.right_doc_id)

        groups: dict[str, set[str]] = defaultdict(set)
        for edge in usable_edges:
            root = find(edge.left_doc_id)
            groups[root].update({edge.left_doc_id, edge.right_doc_id})

        clusters: list[ConsolidationCluster] = []
        for group_index, doc_ids in enumerate(groups.values(), start=1):
            if len(doc_ids) < 2:
                continue
            group_edges = [
                edge
                for edge in usable_edges
                if edge.left_doc_id in doc_ids and edge.right_doc_id in doc_ids
            ]
            docs = sorted(
                [manifest_by_doc_id[doc_id] for doc_id in doc_ids if doc_id in manifest_by_doc_id],
                key=lambda item: (item.sector, item.title),
            )
            if len(docs) < 2:
                continue
            sector_names = sorted({doc.sector or "unknown" for doc in docs})
            reason_counter: Counter[str] = Counter()
            for edge in group_edges:
                reason_counter.update(edge.reason_codes)
            score = max((edge.consolidation_score for edge in group_edges), default=0.0)
            reasons = [reason for reason, _ in reason_counter.most_common()]
            clusters.append(
                ConsolidationCluster(
                    cluster_id=f"cluster_{group_index:04d}",
                    documents=docs,
                    sectors=sector_names,
                    edge_ids=[edge.edge_id for edge in group_edges],
                    consolidation_score=round(score, 4),
                    reason_codes=reasons,
                    recommendation=self._cluster_recommendation(reasons, docs),
                    cross_sector=len(sector_names) > 1,
                )
            )
        clusters.sort(key=lambda item: (item.consolidation_score, len(item.documents)), reverse=True)
        return clusters

    def _cluster_recommendation(self, reasons: Sequence[str], docs: Sequence[CampaignDocumentRecord]) -> str:
        if "near_duplicate" in reasons or "same_version_family" in reasons:
            return "Review as a likely duplicate or version family before retiring superseded copies."
        if "shared_process_flow" in reasons:
            return "Review with process owners as a candidate for a single canonical procedure or harmonized workflow."
        if "same_table_schema" in reasons:
            return "Review recurring tables and schemas for a shared controlled template."
        if "overlapping_requirements" in reasons:
            return "Review duplicated obligations and align authoritative ownership before consolidation."
        titles = ", ".join(doc.title for doc in list(docs)[:3])
        return f"Review overlapping operational content across {titles}."

    def _write_artifacts(self, result: DocumentConsolidationCampaignResult) -> list[Dict[str, Any]]:
        artifacts: list[Dict[str, Any]] = []
        artifacts.append(
            _artifact_write(
                self.session,
                "consolidation_manifest.json",
                json.dumps([item.to_dict() for item in result.manifest], indent=2, sort_keys=True),
            )
        )
        artifacts.append(
            _artifact_write(
                self.session,
                "document_similarity_edges.jsonl",
                "\n".join(json.dumps(item.to_dict(), sort_keys=True) for item in result.similarity_edges) + "\n",
            )
        )
        artifacts.append(
            _artifact_write(
                self.session,
                "process_flow_matches.jsonl",
                "\n".join(json.dumps(item.to_dict(), sort_keys=True) for item in result.process_flow_matches) + "\n",
            )
        )
        artifacts.append(
            _artifact_write(
                self.session,
                "consolidation_clusters.json",
                json.dumps([item.to_dict() for item in result.consolidation_clusters], indent=2, sort_keys=True),
            )
        )
        artifacts.append(
            _artifact_write(
                self.session,
                "sector_summary.json",
                json.dumps(result.sector_summary.to_dict(), indent=2, sort_keys=True),
            )
        )
        artifacts.append(
            _artifact_write(self.session, "document_consolidation_report.md", self._render_report(result))
        )
        checkpoint = {
            "campaign_id": result.campaign_id,
            "status": result.status,
            "selected_document_count": result.selected_document_count,
            "similarity_edge_count": len(result.similarity_edges),
            "process_flow_match_count": len(result.process_flow_matches),
            "candidate_cluster_count": len(result.consolidation_clusters),
        }
        artifacts.append(
            _artifact_write(
                self.session,
                "document_consolidation_checkpoint.json",
                json.dumps(checkpoint, indent=2, sort_keys=True),
            )
        )
        return artifacts

    def _register_handoff_artifacts(self, result: DocumentConsolidationCampaignResult) -> None:
        try:
            register_handoff_artifact(
                self.session,
                artifact_type="document_consolidation_manifest",
                handoff_schema="document_consolidation_manifest.v1",
                producer_task_id=result.campaign_id,
                producer_agent="document_consolidation_campaign",
                summary=f"{result.selected_document_count} selected documents across {len(result.sector_summary.sectors)} sectors.",
                data={"manifest": [item.to_dict() for item in result.manifest]},
            )
            register_handoff_artifact(
                self.session,
                artifact_type="document_similarity_candidates",
                handoff_schema="document_similarity_candidates.v1",
                producer_task_id=result.campaign_id,
                producer_agent="document_consolidation_campaign",
                summary=f"{len(result.similarity_edges)} scored candidate document pairs.",
                data={"similarity_edges": [item.to_dict() for item in result.similarity_edges[:250]]},
            )
            register_handoff_artifact(
                self.session,
                artifact_type="consolidation_clusters",
                handoff_schema="consolidation_clusters.v1",
                producer_task_id=result.campaign_id,
                producer_agent="document_consolidation_campaign",
                summary=f"{len(result.consolidation_clusters)} recommended consolidation clusters.",
                data={"clusters": [item.to_dict() for item in result.consolidation_clusters]},
            )
            register_handoff_artifact(
                self.session,
                artifact_type="process_flow_overlap",
                handoff_schema="process_flow_overlap.v1",
                producer_task_id=result.campaign_id,
                producer_agent="document_consolidation_campaign",
                summary=f"{len(result.process_flow_matches)} process-flow overlaps found.",
                data={"process_flow_matches": [item.to_dict() for item in result.process_flow_matches[:250]]},
            )
        except Exception:
            return

    def _render_report(self, result: DocumentConsolidationCampaignResult) -> str:
        lines = [
            "# Document Consolidation Campaign",
            "",
            "## Executive Summary",
            f"- Campaign ID: {result.campaign_id}",
            f"- Status: {result.status}",
            f"- Objective: {result.query or 'identify consolidation candidates'}",
            f"- Selected documents: {result.selected_document_count}",
            f"- Candidate clusters: {len(result.consolidation_clusters)}",
            f"- Similarity edges: {len(result.similarity_edges)}",
            f"- Process-flow matches: {len(result.process_flow_matches)}",
            "- Rewrite performed: no",
            "",
            "## Sector Summary",
        ]
        for sector, count in result.sector_summary.sectors.items():
            lines.append(f"- {sector}: {count} document(s)")
        if result.sector_summary.unknown_documents:
            lines.extend(["", "## Unknown Sector Documents"])
            lines.extend(f"- {title}" for title in result.sector_summary.unknown_documents[:50])
        if result.consolidation_clusters:
            lines.extend(["", "## Top Consolidation Recommendations"])
            for cluster in result.consolidation_clusters[:25]:
                lines.append(f"### {cluster.cluster_id}: score {cluster.consolidation_score:.2f}")
                lines.append(f"- Sectors: {', '.join(cluster.sectors)}")
                lines.append(f"- Reason codes: {', '.join(cluster.reason_codes)}")
                lines.append(f"- Recommendation: {cluster.recommendation}")
                lines.append("- Documents:")
                for doc in cluster.documents:
                    lines.append(f"  - {doc.title} ({doc.doc_id})")
        advisory_edges = [edge for edge in result.similarity_edges if edge.cross_sector_advisory]
        if advisory_edges:
            lines.extend(["", "## Cross-Sector Advisory Matches"])
            for edge in advisory_edges[:25]:
                lines.append(
                    f"- {edge.left_title} [{edge.left_sector}] <-> {edge.right_title} "
                    f"[{edge.right_sector}]: {edge.consolidation_score:.2f} "
                    f"({', '.join(edge.reason_codes)})"
                )
        if result.warnings:
            lines.extend(["", "## Warnings"])
            lines.extend(f"- {warning}" for warning in result.warnings)
        lines.extend(
            [
                "",
                "## Next Actions",
                "- Validate candidate clusters with sector process owners.",
                "- Decide authoritative document ownership before any rewrite or retirement.",
                "- Run document_compare only on high-confidence candidate pairs that need detailed deltas.",
            ]
        )
        return "\n".join(lines).rstrip() + "\n"


__all__ = ["DocumentConsolidationCampaignService"]
