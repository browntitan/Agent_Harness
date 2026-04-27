from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List

from langchain.tools import tool

from agentic_chatbot_next.documents.compare import DocumentComparisonService
from agentic_chatbot_next.documents.consolidation import DocumentConsolidationCampaignService
from agentic_chatbot_next.documents.evidence import EvidenceBinderService
from agentic_chatbot_next.documents.extractors import DocumentResolutionError
from agentic_chatbot_next.documents.extractors import DocumentExtractionService
from agentic_chatbot_next.documents.serializers import (
    compact_compare_payload,
    compact_extract_payload,
    write_compare_artifacts,
    write_extract_artifacts,
)
from agentic_chatbot_next.documents.templates import TemplateTransformService


_INLINE_CAMPAIGN_DOCUMENT_THRESHOLD = 12
_INLINE_TRANSFORM_DOCUMENT_THRESHOLD = 4
_INLINE_BINDER_DOCUMENT_THRESHOLD = 6
_INLINE_BINDER_ARTIFACT_THRESHOLD = 10


def _normalize_output(output: str) -> str:
    normalized = str(output or "summary").strip().lower()
    return normalized if normalized in {"summary", "json", "markdown"} else "summary"


def _should_queue_campaign(document_refs: List[str] | None, max_documents: int, run_in_background: bool) -> bool:
    if not run_in_background:
        return False
    refs = [str(item).strip() for item in list(document_refs or []) if str(item).strip()]
    if refs:
        return len(refs) > _INLINE_CAMPAIGN_DOCUMENT_THRESHOLD
    try:
        selected_cap = int(max_documents or 100)
    except (TypeError, ValueError):
        selected_cap = 100
    return selected_cap > _INLINE_CAMPAIGN_DOCUMENT_THRESHOLD


def _job_manager_from_context(tool_context: object | None) -> object | None:
    if tool_context is None:
        return None
    job_manager = getattr(tool_context, "job_manager", None)
    if job_manager is not None:
        return job_manager
    kernel = getattr(tool_context, "kernel", None)
    return getattr(kernel, "job_manager", None) if kernel is not None else None


def _should_queue_transform(document_refs: List[str] | None, max_source_elements: int, run_in_background: bool) -> bool:
    if not run_in_background:
        return False
    refs = [str(item).strip() for item in list(document_refs or []) if str(item).strip()]
    if len(refs) > _INLINE_TRANSFORM_DOCUMENT_THRESHOLD:
        return True
    try:
        element_cap = int(max_source_elements or 1200)
    except (TypeError, ValueError):
        element_cap = 1200
    return element_cap > 1200


def _should_queue_binder(
    document_refs: List[str] | None,
    artifact_refs: List[str] | None,
    max_source_excerpts: int,
    run_in_background: bool,
) -> bool:
    if not run_in_background:
        return False
    refs = [str(item).strip() for item in list(document_refs or []) if str(item).strip()]
    artifacts = [str(item).strip() for item in list(artifact_refs or []) if str(item).strip()]
    if len(refs) > _INLINE_BINDER_DOCUMENT_THRESHOLD:
        return True
    if len(artifacts) > _INLINE_BINDER_ARTIFACT_THRESHOLD:
        return True
    try:
        excerpt_cap = int(max_source_excerpts or 300)
    except (TypeError, ValueError):
        excerpt_cap = 300
    return excerpt_cap > 300


def make_document_tools(
    settings: object,
    stores: object,
    session: object,
    *,
    providers: object | None = None,
    event_sink: object | None = None,
    tool_context: object | None = None,
) -> List[Any]:
    @tool
    def document_extract(
        document_ref: str,
        source_scope: str = "auto",
        collection_id: str = "",
        include_tables: bool = True,
        include_figures: bool = True,
        include_metadata: bool = True,
        include_hierarchy: bool = True,
        output: str = "summary",
        max_elements: int = 200,
        export: bool = True,
    ) -> Dict[str, Any]:
        """Extract structured layout, hierarchy, tables, figures, and metadata from one document."""

        try:
            result = DocumentExtractionService(settings, stores, session).extract(
                document_ref=document_ref,
                source_scope=source_scope,
                collection_id=collection_id,
                include_tables=include_tables,
                include_figures=include_figures,
                include_metadata=include_metadata,
                include_hierarchy=include_hierarchy,
                max_elements=max_elements,
            )
            artifacts = write_extract_artifacts(session, result) if export else []
            return compact_extract_payload(result, artifacts=artifacts, output=_normalize_output(output))
        except DocumentResolutionError as exc:
            return exc.payload
        except Exception as exc:
            return {
                "error": "Document extraction failed.",
                "document_ref": str(document_ref or ""),
                "detail": str(exc),
            }

    @tool
    def document_compare(
        left_document_ref: str,
        right_document_ref: str,
        source_scope: str = "auto",
        collection_id: str = "",
        compare_mode: str = "auto",
        focus: str = "",
        include_changed_obligations: bool = True,
        export: bool = True,
    ) -> Dict[str, Any]:
        """Compare two documents and return redline summaries, clause deltas, and changed obligations."""

        try:
            result = DocumentComparisonService(settings, stores, session).compare(
                left_document_ref=left_document_ref,
                right_document_ref=right_document_ref,
                source_scope=source_scope,
                collection_id=collection_id,
                compare_mode=compare_mode,
                focus=focus,
                include_changed_obligations=include_changed_obligations,
            )
            artifacts = write_compare_artifacts(session, result) if export else []
            return compact_compare_payload(result, artifacts=artifacts)
        except DocumentResolutionError as exc:
            return exc.payload
        except Exception as exc:
            return {
                "error": "Document comparison failed.",
                "left_document_ref": str(left_document_ref or ""),
                "right_document_ref": str(right_document_ref or ""),
                "detail": str(exc),
            }

    @tool
    def document_consolidation_campaign(
        query: str = "",
        source_scope: str = "auto",
        collection_id: str = "",
        document_refs: List[str] | None = None,
        sector_mode: str = "infer",
        sector_map: Dict[str, str] | None = None,
        allow_cross_sector_comparisons: bool = False,
        cross_sector_mode: str = "blocked",
        similarity_focus: str = "auto",
        min_similarity_score: float = 0.72,
        max_documents: int = 100,
        max_candidate_pairs: int = 150,
        run_in_background: bool = True,
        export: bool = True,
    ) -> Dict[str, Any]:
        """Run a read-only corpus campaign that recommends document consolidation candidates."""

        params = {
            "query": query,
            "source_scope": source_scope,
            "collection_id": collection_id,
            "document_refs": document_refs or [],
            "sector_mode": sector_mode,
            "sector_map": dict(sector_map or {}),
            "allow_cross_sector_comparisons": allow_cross_sector_comparisons,
            "cross_sector_mode": cross_sector_mode,
            "similarity_focus": similarity_focus,
            "min_similarity_score": min_similarity_score,
            "max_documents": max_documents,
            "max_candidate_pairs": max_candidate_pairs,
            "export": export,
        }
        try:
            job_manager = _job_manager_from_context(tool_context)
            if _should_queue_campaign(document_refs, max_documents, run_in_background) and job_manager is not None:
                campaign_id = f"doc_consolidation_{uuid.uuid4().hex[:12]}"
                prompt = (
                    "Run document_consolidation_campaign as a read-only background analysis. "
                    f"Campaign ID: {campaign_id}. Parameters: {json.dumps(params, sort_keys=True, default=str)}"
                )
                job = job_manager.create_job(
                    agent_name="document_consolidation_campaign",
                    prompt=prompt,
                    session_id=str(getattr(session, "session_id", "") or ""),
                    description="Document consolidation campaign",
                    session_state={
                        "tenant_id": str(getattr(session, "tenant_id", "") or ""),
                        "conversation_id": str(getattr(session, "conversation_id", "") or ""),
                    },
                    metadata={
                        "tool_name": "document_consolidation_campaign",
                        "campaign_id": campaign_id,
                        "parameters": params,
                    },
                    tenant_id=str(getattr(session, "tenant_id", "") or ""),
                    user_id=str(getattr(session, "user_id", "") or ""),
                    priority="background",
                    queue_class="background",
                    estimated_token_cost=0,
                )

                def runner(_running_job: object) -> str:
                    result = DocumentConsolidationCampaignService(
                        settings,
                        stores,
                        session,
                        event_sink=event_sink,
                    ).run(campaign_id=campaign_id, **params)
                    return json.dumps(result.compact(max_clusters=12), indent=2, sort_keys=True)

                job_manager.start_background_job(job, runner)
                return {
                    "campaign_id": campaign_id,
                    "status": "queued",
                    "background_job_id": str(getattr(job, "job_id", "") or ""),
                    "selected_document_count": len(document_refs or []) or int(max_documents or 0),
                    "candidate_cluster_count": 0,
                    "warnings": [
                        "Campaign queued in the background because the requested corpus is above the inline threshold."
                    ],
                    "artifacts": [],
                }

            result = DocumentConsolidationCampaignService(
                settings,
                stores,
                session,
                event_sink=event_sink,
            ).run(**params)
            return result.compact(max_clusters=8)
        except Exception as exc:
            return {
                "error": "Document consolidation campaign failed.",
                "query": str(query or ""),
                "detail": str(exc),
            }

    @tool
    def template_transform(
        document_refs: List[str],
        template_type: str,
        source_scope: str = "auto",
        collection_id: str = "",
        focus: str = "",
        audience: str = "",
        output_format: str = "auto",
        template_parameters: Dict[str, Any] | None = None,
        include_source_trace: bool = True,
        drafting_mode: str = "grounded",
        max_source_elements: int = 1200,
        run_in_background: bool = True,
        export: bool = True,
    ) -> Dict[str, Any]:
        """Convert source documents into a grounded memo, matrix, tracker, brief, or test-report artifact."""

        params = {
            "document_refs": document_refs or [],
            "template_type": template_type,
            "source_scope": source_scope,
            "collection_id": collection_id,
            "focus": focus,
            "audience": audience,
            "output_format": output_format,
            "template_parameters": dict(template_parameters or {}),
            "include_source_trace": include_source_trace,
            "drafting_mode": drafting_mode,
            "max_source_elements": max_source_elements,
            "export": export,
        }
        try:
            job_manager = _job_manager_from_context(tool_context)
            if _should_queue_transform(document_refs, max_source_elements, run_in_background) and job_manager is not None:
                transform_id = f"template_transform_{uuid.uuid4().hex[:12]}"
                prompt = (
                    "Run template_transform as a read-only background document transform. "
                    f"Transform ID: {transform_id}. Parameters: {json.dumps(params, sort_keys=True, default=str)}"
                )
                job = job_manager.create_job(
                    agent_name="template_transform",
                    prompt=prompt,
                    session_id=str(getattr(session, "session_id", "") or ""),
                    description="Template transform",
                    session_state={
                        "tenant_id": str(getattr(session, "tenant_id", "") or ""),
                        "conversation_id": str(getattr(session, "conversation_id", "") or ""),
                    },
                    metadata={
                        "tool_name": "template_transform",
                        "transform_id": transform_id,
                        "parameters": params,
                    },
                    tenant_id=str(getattr(session, "tenant_id", "") or ""),
                    user_id=str(getattr(session, "user_id", "") or ""),
                    priority="background",
                    queue_class="background",
                    estimated_token_cost=0,
                )

                def runner(_running_job: object) -> str:
                    result = TemplateTransformService(
                        settings,
                        stores,
                        session,
                        providers=providers,
                        event_sink=event_sink,
                    ).transform(transform_id=transform_id, **params)
                    return json.dumps(result.compact(), indent=2, sort_keys=True)

                job_manager.start_background_job(job, runner)
                return {
                    "transform_id": transform_id,
                    "status": "queued",
                    "background_job_id": str(getattr(job, "job_id", "") or ""),
                    "template_type": str(template_type or ""),
                    "output_format": str(output_format or "auto"),
                    "selected_document_count": len(document_refs or []),
                    "generated_artifacts": [],
                    "source_trace_artifacts": [],
                    "warnings": [
                        "Template transform queued in the background because the requested source set is above the inline threshold."
                    ],
                }

            result = TemplateTransformService(
                settings,
                stores,
                session,
                providers=providers,
                event_sink=event_sink,
            ).transform(**params)
            return result.compact()
        except Exception as exc:
            return {
                "error": "Template transform failed.",
                "template_type": str(template_type or ""),
                "detail": str(exc),
            }

    @tool
    def evidence_binder(
        binder_title: str = "Evidence Binder",
        objective: str = "",
        document_refs: List[str] | None = None,
        artifact_refs: List[str] | None = None,
        handoff_artifact_ids: List[str] | None = None,
        handoff_artifact_types: List[str] | None = None,
        source_scope: str = "auto",
        collection_id: str = "",
        include_latest_artifacts: bool = True,
        include_generated_artifacts: bool = True,
        include_original_sources: bool = False,
        citation_policy: str = "warn_and_include",
        max_source_excerpts: int = 300,
        run_in_background: bool = True,
        export: bool = True,
    ) -> Dict[str, Any]:
        """Package source excerpts, generated artifacts, citations, and provenance into an auditable binder."""

        params = {
            "binder_title": binder_title,
            "objective": objective,
            "document_refs": document_refs or [],
            "artifact_refs": artifact_refs or [],
            "handoff_artifact_ids": handoff_artifact_ids or [],
            "handoff_artifact_types": handoff_artifact_types or [],
            "source_scope": source_scope,
            "collection_id": collection_id,
            "include_latest_artifacts": include_latest_artifacts,
            "include_generated_artifacts": include_generated_artifacts,
            "include_original_sources": include_original_sources,
            "citation_policy": citation_policy,
            "max_source_excerpts": max_source_excerpts,
            "export": export,
        }
        try:
            job_manager = _job_manager_from_context(tool_context)
            if _should_queue_binder(document_refs, artifact_refs, max_source_excerpts, run_in_background) and job_manager is not None:
                binder_id = f"evidence_binder_{uuid.uuid4().hex[:12]}"
                prompt = (
                    "Run evidence_binder as a read-only background evidence packaging job. "
                    f"Binder ID: {binder_id}. Parameters: {json.dumps(params, sort_keys=True, default=str)}"
                )
                job = job_manager.create_job(
                    agent_name="evidence_binder",
                    prompt=prompt,
                    session_id=str(getattr(session, "session_id", "") or ""),
                    description="Evidence binder",
                    session_state={
                        "tenant_id": str(getattr(session, "tenant_id", "") or ""),
                        "conversation_id": str(getattr(session, "conversation_id", "") or ""),
                    },
                    metadata={
                        "tool_name": "evidence_binder",
                        "binder_id": binder_id,
                        "parameters": params,
                    },
                    tenant_id=str(getattr(session, "tenant_id", "") or ""),
                    user_id=str(getattr(session, "user_id", "") or ""),
                    priority="background",
                    queue_class="background",
                    estimated_token_cost=0,
                )

                def runner(_running_job: object) -> str:
                    result = EvidenceBinderService(settings, stores, session).build(binder_id=binder_id, **params)
                    return json.dumps(result.compact(), indent=2, sort_keys=True)

                job_manager.start_background_job(job, runner)
                return {
                    "binder_id": binder_id,
                    "status": "queued",
                    "background_job_id": str(getattr(job, "job_id", "") or ""),
                    "source_document_count": len(document_refs or []),
                    "evidence_row_count": 0,
                    "included_artifact_count": len(artifact_refs or []),
                    "missing_citation_warning_count": 0,
                    "binder_artifacts": [],
                    "warnings": [
                        "Evidence binder queued in the background because the requested evidence set is above the inline threshold."
                    ],
                }

            result = EvidenceBinderService(settings, stores, session).build(**params)
            return result.compact()
        except Exception as exc:
            return {
                "error": "Evidence binder failed.",
                "binder_title": str(binder_title or ""),
                "detail": str(exc),
            }

    return [
        document_extract,
        document_compare,
        document_consolidation_campaign,
        template_transform,
        evidence_binder,
    ]


__all__ = ["make_document_tools"]
