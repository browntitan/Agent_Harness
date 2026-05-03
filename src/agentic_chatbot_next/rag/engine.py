from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List

from agentic_chatbot_next.contracts.rag import Citation, RagContract, RetrievalSummary
from agentic_chatbot_next.rag.hints import (
    answer_contract_allows_inventory,
    answer_contract_kind,
    coerce_controller_hints,
    normalize_coverage_goal,
    normalize_research_profile,
    normalize_result_mode,
    prefers_bounded_synthesis,
)
from agentic_chatbot_next.rag.adaptive import CorpusRetrievalAdapter, run_retrieval_controller
from agentic_chatbot_next.rag.citations import (
    build_citations,
    citation_display_label,
    replace_inline_citation_ids,
)
from agentic_chatbot_next.rag.source_links import make_document_source_url_resolver
from agentic_chatbot_next.rag.collection_selection import (
    apply_selection_to_session,
    select_collection_for_query,
    selection_answer,
)
from agentic_chatbot_next.rag.discovery_precision import (
    discovery_topic_label,
    document_has_explicit_topic_support,
)
from agentic_chatbot_next.rag.doc_targets import IndexedDocResolution, resolve_query_document_targets
from agentic_chatbot_next.rag.fanout import RagRuntimeBridge
from agentic_chatbot_next.rag.ingest import (
    CollectionReadinessStatus,
    KBCoverageStatus,
    get_collection_readiness_status,
    get_kb_coverage_status,
)
from agentic_chatbot_next.rag.inventory import (
    INVENTORY_QUERY_GRAPH_INDEXES,
    INVENTORY_QUERY_GRAPH_FILE,
    INVENTORY_QUERY_KB_FILE,
    INVENTORY_QUERY_KB_COLLECTIONS,
    INVENTORY_QUERY_NONE,
    INVENTORY_QUERY_SESSION_ACCESS,
    classify_inventory_query,
    dispatch_authoritative_inventory,
    inventory_query_requests_grounded_analysis,
    sync_session_kb_collection_state,
)
from agentic_chatbot_next.rag.query_normalization import normalize_retrieval_question
from agentic_chatbot_next.rag.retrieval import GradedChunk
from agentic_chatbot_next.rag.retrieval_scope import (
    RetrievalScopeDecision,
    decide_retrieval_scope,
    has_upload_evidence,
    resolve_kb_collection_id,
)
from agentic_chatbot_next.rag.synthesis import build_extractive_grounded_answer, generate_grounded_answer
from agentic_chatbot_next.rag.tabular import (
    deterministic_status_evidence_results,
    plan_tabular_evidence_tasks,
    tabular_evidence_results_to_documents,
)
from agentic_chatbot_next.runtime.deep_rag import (
    deep_rag_controller_hints,
    deep_rag_search_mode,
)


def _tenant_id(settings: Any, session: Any) -> str:
    return str(
        getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev"))
        or getattr(settings, "default_tenant_id", "local-dev")
        or "local-dev"
    )


def _mapping_or_attr(value: Any, key: str, default: Any = "") -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _requires_mermaid_output(
    question: str,
    *,
    answer_contract: Any | None = None,
    presentation_preferences: Dict[str, Any] | None = None,
) -> bool:
    final_output_mode = str(_mapping_or_attr(answer_contract, "final_output_mode", "") or "").strip().lower()
    diagram_policy = str((presentation_preferences or {}).get("diagram_policy") or "").strip().lower()
    if "mermaid" in final_output_mode or diagram_policy == "require_mermaid":
        return True
    return bool(re.search(r"\bmermaid\b", str(question or ""), flags=re.IGNORECASE))


def _prefers_mermaid_output(
    question: str,
    *,
    answer_contract: Any | None = None,
    presentation_preferences: Dict[str, Any] | None = None,
) -> bool:
    if _requires_mermaid_output(
        question,
        answer_contract=answer_contract,
        presentation_preferences=presentation_preferences,
    ):
        return True
    diagram_policy = str((presentation_preferences or {}).get("diagram_policy") or "").strip().lower()
    final_output_mode = str(_mapping_or_attr(answer_contract, "final_output_mode", "") or "").strip().lower()
    return bool(diagram_policy == "auto_mermaid" or "diagram" in final_output_mode)


def _early_contract(query: str, answer_payload: Dict[str, Any], *, search_mode: str = "none") -> RagContract:
    return RagContract(
        answer=str(answer_payload.get("answer") or ""),
        citations=[],
        used_citation_ids=[],
        confidence=float(answer_payload.get("confidence_hint") or 0.0),
        retrieval_summary=RetrievalSummary(query_used=query, search_mode=search_mode),
        followups=[str(item) for item in (answer_payload.get("followups") or []) if str(item)],
        warnings=[str(item) for item in (answer_payload.get("warnings") or []) if str(item)],
    )


def _title_overlap_score(question: str, doc: Any) -> int:
    title = str((getattr(doc, "metadata", {}) or {}).get("title") or "").lower()
    if not title:
        return 0
    q_terms = set(re.findall(r"[A-Za-z0-9_]{3,}", question.lower()))
    t_terms = set(re.findall(r"[A-Za-z0-9_]{3,}", title.replace("_", " ")))
    overlap = len(q_terms & t_terms)
    if "architecture" in q_terms and "architecture" in t_terms:
        overlap += 2
    return overlap


def _select_evidence_docs(question: str, graded: list[GradedChunk], min_chunks: int) -> list[Any]:
    target = max(1, int(min_chunks))
    strong = [item.doc for item in graded if item.relevance >= 2]
    strong.sort(key=lambda doc: _title_overlap_score(question, doc), reverse=True)
    if len(strong) >= target:
        return strong

    supplemental = [item.doc for item in graded if item.relevance == 1]
    supplemental.sort(key=lambda doc: _title_overlap_score(question, doc), reverse=True)
    return (strong + supplemental)[:target]


def _summary_snippet(text: str, *, limit: int = 180) -> str:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return ""
    return normalized[:limit].rstrip()


def _kb_not_ready_answer(status: CollectionReadinessStatus | KBCoverageStatus) -> Dict[str, Any]:
    reason = str(getattr(status, "reason", "") or "").strip().lower()
    maintenance_policy = str(getattr(status, "maintenance_policy", "") or "").strip().lower()
    if status.sync_error:
        detail = (
            f"Startup KB sync failed for collection '{status.collection_id}': {status.sync_error}. "
            f"Run `{status.suggested_fix}` and retry the request."
        )
        warning = "KB_SYNC_FAILED"
    elif reason == "empty_collection" and maintenance_policy == "indexed_documents":
        detail = (
            f"Collection '{status.collection_id}' does not have indexed documents yet. "
            f"{status.suggested_fix}"
        )
        warning = "KB_COLLECTION_EMPTY"
    else:
        detail = (
            f"The configured knowledge base is not indexed for collection '{status.collection_id}'. "
            f"Run `{status.suggested_fix}` and retry the request."
        )
        warning = "KB_COVERAGE_MISSING"

    if status.missing_source_paths:
        preview = ", ".join(status.missing_source_paths[:3])
        if len(status.missing_source_paths) > 3:
            preview += ", ..."
        detail = f"{detail} Missing sources: {preview}"

    return {
        "answer": detail,
        "used_citation_ids": [],
        "followups": [],
        "warnings": [warning],
        "confidence_hint": 0.0,
    }


def _ambiguous_scope_answer(decision: RetrievalScopeDecision) -> Dict[str, Any]:
    detail = (
        "I can answer this using the uploaded files, the knowledge base, both, or neither, "
        "but your request does not say which one to use."
    )
    if decision.has_uploads and decision.kb_available:
        detail = (
            "I have both the same-chat uploads and the shared knowledge base available, "
            "but your request does not say which one to use."
        )
    return {
        "answer": (
            f"{detail} Reply with one of: "
            "`uploaded files only`, `knowledge base only`, `both`, or `neither`."
        ),
        "used_citation_ids": [],
        "followups": [
            "uploaded files only",
            "knowledge base only",
            "both",
            "neither",
        ],
        "warnings": ["RETRIEVAL_SCOPE_AMBIGUOUS"],
        "confidence_hint": 0.0,
    }


def _ambiguous_kb_collection_answer(collection_ids: list[str]) -> Dict[str, Any]:
    options = [str(item).strip() for item in collection_ids if str(item).strip()]
    quoted_options = ", ".join(f"`{item}`" for item in options)
    return {
        "answer": (
            "I can search multiple knowledge base collections in this chat, "
            "but your request does not say which collection to use. "
            f"Reply with one of: {quoted_options}."
        ),
        "used_citation_ids": [],
        "followups": options,
        "warnings": ["KB_COLLECTION_SELECTION_REQUIRED"],
        "confidence_hint": 0.0,
    }


def _uploads_missing_answer(*, allow_kb_fallback: bool) -> Dict[str, Any]:
    suffix = " Upload a file and retry the request."
    if allow_kb_fallback:
        suffix = " Upload a file, or ask me to use the knowledge base instead."
    return {
        "answer": f"There are no uploaded files attached to this chat yet.{suffix}",
        "used_citation_ids": [],
        "followups": [],
        "warnings": ["NO_CHAT_UPLOADS_AVAILABLE"],
        "confidence_hint": 0.0,
    }


def _retrieval_disabled_answer() -> Dict[str, Any]:
    return {
        "answer": (
            "I’m not consulting the knowledge base or uploaded files for this turn. "
            "Tell me to use `uploaded files only`, `knowledge base only`, `both`, or `neither` "
            "if you want a grounded answer."
        ),
        "used_citation_ids": [],
        "followups": [],
        "warnings": ["RETRIEVAL_SKIPPED_BY_INTENT"],
        "confidence_hint": 0.0,
    }


def _build_inventory_answer(docs: list[Any]) -> Dict[str, Any]:
    lines = ["Documents with grounded evidence relevant to the request:"]
    used_citation_ids: List[str] = []
    seen_doc_ids: set[str] = set()
    for doc in docs:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        doc_id = str(metadata.get("doc_id") or "")
        if doc_id and doc_id in seen_doc_ids:
            continue
        if doc_id:
            seen_doc_ids.add(doc_id)
        title = str(metadata.get("title") or doc_id or "Untitled document")
        citation_id = str(metadata.get("chunk_id") or "")
        snippet = _summary_snippet(getattr(doc, "page_content", ""))
        suffix = f" ({citation_id})" if citation_id else ""
        if snippet:
            lines.append(f"- {title}: {snippet}{suffix}")
        else:
            lines.append(f"- {title}{suffix}")
        if citation_id:
            used_citation_ids.append(citation_id)
    if len(lines) == 1:
        lines.append("- No matching documents were identified from the selected evidence set.")
    return {
        "answer": "\n".join(lines),
        "used_citation_ids": used_citation_ids,
        "followups": [],
        "warnings": [],
        "confidence_hint": 0.68 if used_citation_ids else 0.4,
    }


def _is_corpus_discovery_inventory(
    *,
    query: str,
    research_profile: str,
    coverage_goal: str,
    result_mode: str,
    controller_hints: Dict[str, Any],
) -> bool:
    if not (result_mode == "inventory" or bool(controller_hints.get("prefer_inventory_output"))):
        return False
    if coverage_goal in {"corpus_wide", "exhaustive"}:
        return True
    if research_profile in {"corpus_discovery", "process_flow_identification"}:
        return True
    return bool(re.search(r"\b(which|identify|list|find)\s+(?:all\s+)?(?:documents|files)\b", query, re.IGNORECASE))


def _filter_confirmed_discovery_docs(query: str, docs: list[Any]) -> tuple[list[Any], list[Dict[str, Any]]]:
    confirmed: List[Any] = []
    rejected: List[Dict[str, Any]] = []
    for doc in docs:
        details = document_has_explicit_topic_support(query, doc)
        if details.get("matches", True):
            confirmed.append(doc)
            continue
        rejected.append(
            {
                "doc_id": str(details.get("doc_id") or ""),
                "title": str(details.get("title") or ""),
                "reason": "missing_topic_anchor",
                "required_anchors": list(details.get("required_anchors") or []),
            }
        )
    return confirmed, rejected


def _build_no_confirmed_discovery_matches_answer(
    query: str,
    retrieval_run: Any,
    *,
    rejected_candidates: list[Dict[str, Any]] | None = None,
    verification_failed: bool = False,
) -> Dict[str, Any]:
    label = discovery_topic_label(query)
    target = "knowledge-base documents" if re.search(r"\bknowledge\s*base\b|\bKB\b", query, re.IGNORECASE) else "indexed documents"
    lines = [f"No confirmed {target} were found that explicitly mention {label}."]
    if rejected_candidates:
        lines.append(
            f"I reviewed broader candidate matches across {int((retrieval_run.candidate_counts or {}).get('unique_docs', 0))} document(s), "
            f"but the grounded snippets did not explicitly mention {label}."
        )
    if verification_failed:
        lines.append("The remaining evidence also could not be verified strongly enough to return a document list.")
    return {
        "answer": " ".join(lines),
        "used_citation_ids": [],
        "followups": [
            "If you want, I can broaden this to near matches or search for an exact phrase like `onboarding workflow` or `new hire workflow`."
        ],
        "warnings": ["INSUFFICIENT_CORPUS_EVIDENCE"],
        "confidence_hint": 0.18,
    }


def _build_negative_evidence_answer(query: str, retrieval_run: Any) -> Dict[str, Any]:
    strategies = ", ".join(retrieval_run.strategies_used or ["hybrid"])
    doc_count = int((retrieval_run.candidate_counts or {}).get("unique_docs", 0))
    round_count = max(1, int(getattr(retrieval_run, "rounds", 0) or 0))
    answer = (
        f"I could not find enough grounded evidence to answer this request confidently. "
        f"I searched for '{query}' across {doc_count} candidate document(s) using {strategies} "
        f"over {round_count} retrieval round(s)."
    )
    return {
        "answer": answer,
        "used_citation_ids": [],
        "followups": [
            "Try naming a specific workflow, process name, or exact term to narrow the search."
        ],
        "warnings": ["INSUFFICIENT_CORPUS_EVIDENCE"],
        "confidence_hint": 0.15,
    }


def _build_cited_evidence_fallback_answer(query: str, docs: list[Any]) -> Dict[str, Any]:
    lines = [
        "I found relevant evidence, but could not produce a fully cited synthesis. "
        "Here is the grounded evidence I can cite directly:"
    ]
    used_citation_ids: List[str] = []
    seen: set[str] = set()
    for doc in docs:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        citation_id = str(metadata.get("chunk_id") or "").strip()
        if not citation_id or citation_id in seen:
            continue
        seen.add(citation_id)
        title = str(metadata.get("title") or metadata.get("doc_id") or "Retrieved evidence").strip()
        snippet = _summary_snippet(getattr(doc, "page_content", ""), limit=220)
        if not snippet:
            continue
        lines.append(f"- {title}: {snippet} ({citation_id})")
        used_citation_ids.append(citation_id)
        if len(used_citation_ids) >= 4:
            break
    if not used_citation_ids:
        return {
            "answer": (
                "I could not produce a safely cited answer from the retrieved evidence. "
                "Please narrow the request or provide more source material."
            ),
            "used_citation_ids": [],
            "followups": [],
            "warnings": ["MISSING_VALID_CITATIONS", "NO_CITABLE_EVIDENCE"],
            "confidence_hint": 0.15,
        }
    return {
        "answer": "\n".join(lines),
        "used_citation_ids": used_citation_ids,
        "followups": [],
        "warnings": ["MISSING_VALID_CITATIONS_FALLBACK"],
        "confidence_hint": 0.35,
    }


_CITATION_AUGMENT_STOPWORDS = {
    "about",
    "after",
    "answer",
    "briefly",
    "citation",
    "citations",
    "does",
    "from",
    "knowledge",
    "search",
    "source",
    "sources",
    "that",
    "this",
    "what",
    "when",
    "which",
    "with",
}


def _content_terms(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9_]{2,}", str(value or "").casefold())
        if token not in _CITATION_AUGMENT_STOPWORDS
    }


_STRUCTURED_LABEL_ALIASES: Dict[str, tuple[str, ...]] = {
    "approval": ("approval", "approver", "approved by", "approval owner"),
    "cost": ("cost", "amount", "price", "budget", "spend", "revenue", "value"),
    "customer": ("customer", "client", "account"),
    "date": ("date", "due", "deadline", "milestone", "target date"),
    "issue": ("issue", "problem", "defect", "finding"),
    "owner": ("owner", "assignee", "responsible", "lead", "poc"),
    "region": ("region", "area", "territory"),
    "risk": ("risk", "hazard", "threat"),
    "status": ("status", "state", "disposition", "outcome"),
    "supplier": ("supplier", "vendor", "provider", "manufacturer"),
}
_STRUCTURED_CONTEXT_LABEL_TERMS = {
    "comment",
    "comments",
    "context",
    "evidence",
    "finding",
    "findings",
    "note",
    "notes",
    "question",
    "ref",
    "refs",
    "source",
    "sources",
}


def _doc_is_binding_structured_evidence(doc: Any) -> bool:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    source_type = str(metadata.get("source_type") or "").strip().lower()
    chunk_type = str(metadata.get("chunk_type") or "").strip().lower()
    return bool(
        metadata.get("is_synthetic_evidence")
        or source_type in {"tabular_analysis", "tool_result"}
        or chunk_type in {"tabular_analysis", "tool_result"}
        or metadata.get("sheet_name")
        or metadata.get("cell_range")
        or metadata.get("row_start") is not None
    )


def _structured_label_terms(question: str) -> set[str]:
    lowered = str(question or "").casefold()
    requested: set[str] = set()
    for canonical, aliases in _STRUCTURED_LABEL_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias.casefold())}s?\b", lowered):
                requested.add(canonical)
                requested.update(alias.casefold().split())
                break
    return requested


def _structured_label_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _structured_label_match_score(label: str, requested: set[str]) -> float:
    if not requested:
        return 0.0
    lowered = str(label or "").casefold()
    label_tokens = _structured_label_tokens(lowered)
    if not label_tokens:
        return 0.0
    score = 0.0
    for canonical, aliases in _STRUCTURED_LABEL_ALIASES.items():
        canonical_requested = canonical in requested
        for alias in aliases:
            alias_tokens = _structured_label_tokens(alias)
            if not alias_tokens:
                continue
            alias_requested = bool(alias_tokens & requested)
            if alias_requested and label_tokens == alias_tokens:
                score = max(score, 6.0)
            elif alias_requested and alias_tokens <= label_tokens:
                score = max(score, 5.0)
            elif alias_requested and alias in lowered:
                score = max(score, 4.0)
            elif canonical_requested and canonical in label_tokens:
                score = max(score, 4.0)
            elif canonical_requested and alias_tokens <= label_tokens:
                score = max(score, 1.5)
    if label_tokens & _STRUCTURED_CONTEXT_LABEL_TERMS and len(label_tokens) > 1:
        score = max(0.0, score - 3.0)
    return score


def _label_matches_requested(label: str, requested: set[str]) -> bool:
    return _structured_label_match_score(label, requested) > 0.0


def _extract_labeled_values(text: str) -> list[Dict[str, str]]:
    values: list[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in re.finditer(
        r"\b([A-Za-z][A-Za-z0-9_ /-]{1,42})\s*:\s*([^|;\n.]{1,120})",
        str(text or ""),
    ):
        label = re.sub(r"\s+", " ", match.group(1)).strip(" -")
        value = re.sub(r"\s+", " ", match.group(2)).strip(" ,")
        if not label or not value:
            continue
        if len(value) < 2 or value.casefold() in {"none", "n/a", "unknown"}:
            continue
        key = (label.casefold(), value.casefold())
        if key in seen:
            continue
        seen.add(key)
        values.append({"label": label, "value": value})
        if len(values) >= 20:
            break
    return values


def _structured_candidate_score(question: str, doc: Any) -> float:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    content = str(getattr(doc, "page_content", "") or "")
    query_terms = _content_terms(question)
    evidence_terms = _content_terms(
        " ".join(
            [
                str(metadata.get("title") or ""),
                str(metadata.get("source_path") or ""),
                str(metadata.get("sheet_name") or ""),
                str(metadata.get("cell_range") or ""),
                content[:1800],
            ]
        )
    )
    score = float(len(query_terms & evidence_terms))
    score += float(metadata.get("_adaptive_score") or 0.0) * 0.05
    score += max(0.0, float(metadata.get("tabular_confidence") or 0.0)) * 3.0
    if bool(metadata.get("is_synthetic_evidence")):
        score += 8.0
    if metadata.get("sheet_name") or metadata.get("cell_range") or metadata.get("row_start") is not None:
        score += 3.0
    return score


def _binding_evidence_candidates(question: str, docs: list[Any], *, limit: int = 6) -> list[Dict[str, Any]]:
    requested_labels = _structured_label_terms(question)
    candidates: list[Dict[str, Any]] = []
    for doc in docs:
        if not _doc_is_binding_structured_evidence(doc):
            continue
        metadata = dict(getattr(doc, "metadata", {}) or {})
        content = str(getattr(doc, "page_content", "") or "")
        labeled_values = _extract_labeled_values(content)
        requested_values: list[Dict[str, str]] = []
        requested_label_score = 0.0
        for item in labeled_values:
            label_score = _structured_label_match_score(str(item.get("label") or ""), requested_labels)
            if label_score >= 3.0:
                requested_values.append({**item, "label_score": round(label_score, 3)})
                requested_label_score += label_score
        candidate_values = requested_values or labeled_values[:5]
        citation_id = str(metadata.get("chunk_id") or "").strip()
        if not citation_id and not candidate_values:
            continue
        candidates.append(
            {
                "citation_id": citation_id,
                "doc_id": str(metadata.get("doc_id") or ""),
                "title": str(metadata.get("title") or ""),
                "source_path": str(metadata.get("source_path") or ""),
                "sheet_name": str(metadata.get("sheet_name") or ""),
                "cell_range": str(metadata.get("cell_range") or ""),
                "row_start": metadata.get("row_start"),
                "row_end": metadata.get("row_end"),
                "score": round(_structured_candidate_score(question, doc), 3),
                "requested_label_score": round(requested_label_score, 3),
                "requested_labeled_values": requested_values[:6],
                "labeled_values": candidate_values[:6],
                "snippet": _summary_snippet(content, limit=360),
            }
        )
    candidates.sort(
        key=lambda item: (
            float(item.get("requested_label_score") or 0.0),
            len(item.get("requested_labeled_values") or []),
            float(item.get("score") or 0.0),
        ),
        reverse=True,
    )
    return candidates[: max(1, int(limit))]


def _value_present_in_answer(value: str, answer: str) -> bool:
    clean = re.sub(r"\s+", " ", str(value or "")).strip()
    if not clean:
        return False
    answer_text = re.sub(r"\s+", " ", str(answer or "")).casefold()
    if clean.casefold() in answer_text:
        return True
    terms = [term for term in _content_terms(clean) if len(term) >= 3]
    if len(terms) >= 2:
        return all(term in answer_text for term in terms[:4])
    return bool(terms and terms[0] in answer_text)


def _question_expects_keyed_structured_value(question: str) -> bool:
    text = str(question or "").casefold()
    return bool(
        _structured_label_terms(question)
        or re.search(r"\b(?:what|which|who|when|where|how much|how many)\b", text)
    )


def _build_structured_value_fallback_payload(
    query: str,
    candidate: Dict[str, Any],
    required_values: list[Dict[str, str]],
) -> Dict[str, Any]:
    citation_id = str(candidate.get("citation_id") or "").strip()
    value_parts = [
        f"{str(item.get('label') or 'Value').strip()}: {str(item.get('value') or '').strip()}"
        for item in required_values
        if str(item.get("value") or "").strip()
    ]
    if not value_parts:
        return {}
    location_parts: list[str] = []
    if candidate.get("sheet_name"):
        location_parts.append(f"sheet {candidate['sheet_name']}")
    row_start = candidate.get("row_start")
    row_end = candidate.get("row_end")
    if row_start is not None:
        if row_end is not None and str(row_end) != str(row_start):
            location_parts.append(f"rows {row_start}-{row_end}")
        else:
            location_parts.append(f"row {row_start}")
    if candidate.get("cell_range"):
        location_parts.append(f"cells {candidate['cell_range']}")
    title = str(candidate.get("title") or candidate.get("doc_id") or "the structured evidence").strip()
    location = f" ({'; '.join(location_parts)})" if location_parts else ""
    suffix = f" ({citation_id})" if citation_id else ""
    answer = f"The structured evidence in {title}{location} supports: {'; '.join(value_parts)}.{suffix}"
    return {
        "answer": answer,
        "used_citation_ids": [citation_id] if citation_id else [],
        "followups": [],
        "warnings": ["STRUCTURED_EVIDENCE_VERIFIER_FALLBACK"],
        "confidence_hint": 0.62,
    }


def _structured_evidence_verifier_feedback(candidate: Dict[str, Any], missing_values: list[Dict[str, str]]) -> str:
    citation_id = str(candidate.get("citation_id") or "").strip()
    value_parts = [
        f"{str(item.get('label') or 'Value').strip()}: {str(item.get('value') or '').strip()}"
        for item in missing_values
        if str(item.get("value") or "").strip()
    ]
    source_parts = [str(candidate.get("title") or candidate.get("doc_id") or "structured evidence").strip()]
    if candidate.get("sheet_name"):
        source_parts.append(f"sheet={candidate['sheet_name']}")
    if candidate.get("cell_range"):
        source_parts.append(f"cells={candidate['cell_range']}")
    if citation_id:
        source_parts.append(f"citation_id={citation_id}")
    return (
        "Evidence verifier feedback: the prior draft omitted binding structured evidence. "
        f"Use these exact value(s) when answering: {'; '.join(value_parts)}. "
        f"Ground the answer in {'; '.join(part for part in source_parts if part)}."
    )


def _apply_structured_evidence_arbitration(
    answer_payload: Dict[str, Any],
    *,
    query: str,
    selected_docs: list[Any],
    retrieval_run: Any,
    regenerate_answer: Callable[[Dict[str, Any], list[Dict[str, str]]], Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    candidates = _binding_evidence_candidates(query, selected_docs)
    verification: Dict[str, Any] = {
        "attempted": True,
        "binding_candidate_count": len(candidates),
        "binding_candidates": [
            {
                key: value
                for key, value in candidate.items()
                if key
                in {
                    "citation_id",
                    "doc_id",
                    "title",
                    "sheet_name",
                    "cell_range",
                    "row_start",
                    "row_end",
                    "score",
                    "requested_label_score",
                    "requested_labeled_values",
                }
            }
            for candidate in candidates[:4]
        ],
        "conflicts": [],
        "action": "accepted",
    }
    retrieval_run.retrieval_verification = {
        **dict(getattr(retrieval_run, "retrieval_verification", {}) or {}),
        "evidence_verification": verification,
    }
    if not candidates or not _question_expects_keyed_structured_value(query):
        return answer_payload

    answer = str(answer_payload.get("answer") or "")
    requested_labels = _structured_label_terms(query)
    requested_candidates = [candidate for candidate in candidates if candidate.get("requested_labeled_values")]
    top = requested_candidates[0] if requested_labels and requested_candidates else candidates[0]
    required_values = list(top.get("requested_labeled_values") or []) or list(top.get("labeled_values") or [])[:3]
    missing_values = [
        item
        for item in required_values
        if not _value_present_in_answer(str(item.get("value") or ""), answer)
    ]
    if not missing_values:
        return answer_payload

    verification["conflicts"] = [
        {
            "reason": "answer_omitted_binding_labeled_value",
            "label": str(item.get("label") or ""),
            "expected_value": str(item.get("value") or ""),
            "citation_id": str(top.get("citation_id") or ""),
        }
        for item in missing_values[:6]
    ]

    if regenerate_answer is not None:
        try:
            regenerated_payload = regenerate_answer(top, missing_values) or {}
        except Exception as exc:
            regenerated_payload = {"warnings": [f"STRUCTURED_EVIDENCE_REGENERATION_FAILED: {exc}"]}
        regenerated_answer = str(regenerated_payload.get("answer") or "")
        if regenerated_answer and not any(
            not _value_present_in_answer(str(item.get("value") or ""), regenerated_answer)
            for item in missing_values[:2]
        ):
            verification["action"] = "regenerated_with_verifier_feedback"
            retrieval_run.retrieval_verification = {
                **dict(getattr(retrieval_run, "retrieval_verification", {}) or {}),
                "evidence_verification": verification,
            }
            warnings = [str(item) for item in (regenerated_payload.get("warnings") or []) if str(item)]
            if "STRUCTURED_EVIDENCE_VERIFIER_REGENERATED" not in warnings:
                warnings.append("STRUCTURED_EVIDENCE_VERIFIER_REGENERATED")
            for warning in (answer_payload.get("warnings") or []):
                if str(warning) and str(warning) not in warnings:
                    warnings.append(str(warning))
            return {
                **answer_payload,
                **regenerated_payload,
                "warnings": warnings,
                "confidence_hint": max(
                    float(answer_payload.get("confidence_hint") or 0.0),
                    float(regenerated_payload.get("confidence_hint") or 0.0),
                ),
            }

    fallback_payload = _build_structured_value_fallback_payload(query, top, missing_values) or build_extractive_grounded_answer(
        query,
        [
            doc
            for doc in selected_docs
            if str((getattr(doc, "metadata", {}) or {}).get("chunk_id") or "") == str(top.get("citation_id") or "")
        ]
        or [doc for doc in selected_docs if _doc_is_binding_structured_evidence(doc)],
        warning="STRUCTURED_EVIDENCE_VERIFIER_FALLBACK",
    )
    fallback_answer = str(fallback_payload.get("answer") or "")
    if not fallback_answer or any(
        not _value_present_in_answer(str(item.get("value") or ""), fallback_answer)
        for item in missing_values[:2]
    ):
        return answer_payload

    verification["action"] = "replaced_with_structured_extractive_answer"
    retrieval_run.retrieval_verification = {
        **dict(getattr(retrieval_run, "retrieval_verification", {}) or {}),
        "evidence_verification": verification,
    }
    warnings = [str(item) for item in (fallback_payload.get("warnings") or []) if str(item)]
    for warning in (answer_payload.get("warnings") or []):
        if str(warning) and str(warning) not in warnings:
            warnings.append(str(warning))
    return {
        **answer_payload,
        **fallback_payload,
        "warnings": warnings,
        "confidence_hint": max(float(answer_payload.get("confidence_hint") or 0.0), float(fallback_payload.get("confidence_hint") or 0.0)),
    }


def _augment_used_citation_ids(
    *,
    query: str,
    answer: str,
    docs: list[Any],
    used_citation_ids: list[str],
    max_extra: int = 2,
) -> list[str]:
    if not used_citation_ids:
        return used_citation_ids
    query_terms = _content_terms(query)
    answer_terms = _content_terms(answer)
    if not query_terms or not answer_terms:
        return used_citation_ids
    used = list(dict.fromkeys(str(item) for item in used_citation_ids if str(item)))
    used_set = set(used)
    candidates: list[tuple[float, str]] = []
    for doc in docs:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        citation_id = str(metadata.get("chunk_id") or "").strip()
        if not citation_id or citation_id in used_set:
            continue
        title_source = " ".join(
            [
                str(metadata.get("title") or ""),
                str(metadata.get("source_path") or ""),
                str(metadata.get("section_title") or ""),
                str(metadata.get("sheet_name") or ""),
                str(metadata.get("cell_range") or ""),
            ]
        )
        content = str(getattr(doc, "page_content", "") or "")
        evidence_terms = _content_terms(title_source + " " + content[:1200])
        title_terms = _content_terms(title_source)
        query_overlap = len(query_terms & evidence_terms)
        answer_overlap = len(answer_terms & evidence_terms)
        title_overlap = len(query_terms & title_terms)
        if query_overlap < 2 or answer_overlap < 1:
            continue
        score = (query_overlap * 2.0) + answer_overlap + (title_overlap * 1.5)
        score += float(metadata.get("_adaptive_score") or 0.0) * 0.05
        candidates.append((score, citation_id))
    for _score, citation_id in sorted(candidates, reverse=True)[: max(0, int(max_extra))]:
        used.append(citation_id)
    return used


def _emit_progress(progress_emitter: Any | None, event_type: str, **payload: Any) -> None:
    if progress_emitter is None or not hasattr(progress_emitter, "emit_progress"):
        return
    progress_emitter.emit_progress(event_type, **payload)


def _record_stage_timing(retrieval_run: Any, stage: str, elapsed_ms: float) -> None:
    timings = dict(getattr(retrieval_run, "stage_timings_ms", {}) or {})
    name = str(stage or "unknown").strip() or "unknown"
    timings[name] = round(float(timings.get(name, 0.0) or 0.0) + float(elapsed_ms or 0.0), 3)
    try:
        retrieval_run.stage_timings_ms = timings
        threshold = 5_000.0
        retrieval_run.slow_stages = [
            item
            for item, value in sorted(timings.items(), key=lambda entry: entry[1], reverse=True)
            if value >= threshold
        ][:5]
    except Exception:
        pass


def _augment_with_tabular_evidence(
    *,
    settings: Any,
    query: str,
    selected_docs: List[Any],
    retrieval_run: Any,
    runtime_bridge: RagRuntimeBridge | None,
    progress_emitter: Any | None = None,
) -> tuple[List[Any], List[str]]:
    max_tasks = int(getattr(settings, "rag_tabular_handoff_max_tasks", 2) or 2)
    planning_docs = _merge_unique_documents(selected_docs, list(getattr(retrieval_run, "candidate_docs", []) or []))
    tasks = plan_tabular_evidence_tasks(query, planning_docs, max_tasks=max_tasks)
    if not tasks:
        return selected_docs, []

    if progress_emitter is not None and hasattr(progress_emitter, "emit_progress"):
        progress_emitter.emit_progress(
            "phase_start",
            label="Analyzing spreadsheet evidence",
            detail=f"{len(tasks)} tabular source(s)",
            agent="rag_worker",
            waiting_on="status_extractor",
            docs=[
                {
                    "doc_id": task.doc_id,
                    "title": task.title,
                    "source_path": task.source_path,
                    "source_type": "tabular",
                }
                for task in tasks
            ],
            counts={"tasks": len(tasks)},
        )

    started = time.perf_counter()
    handoff_record: Dict[str, Any] = {
        "attempted": True,
        "task_count": len(tasks),
        "tasks": [task.to_dict() for task in tasks],
    }
    warnings: List[str] = []
    status_started = time.perf_counter()
    status_results, status_warnings, status_stats = deterministic_status_evidence_results(query, tasks)
    status_docs = tabular_evidence_results_to_documents(status_results, tasks)
    _record_stage_timing(retrieval_run, "status_workbook_extractor", (time.perf_counter() - status_started) * 1000.0)
    if status_stats.get("attempted"):
        handoff_record["status_workbook"] = status_stats
        try:
            retrieval_run.status_extractors = dict(status_stats)
        except Exception:
            pass
    if status_docs:
        selected_docs = _merge_unique_documents(status_docs, selected_docs)
        candidate_counts = dict(getattr(retrieval_run, "candidate_counts", {}) or {})
        candidate_counts["status_workbook_evidence_docs"] = len(status_docs)
        candidate_counts["status_workbook_records"] = int(status_stats.get("record_count") or 0)
        retrieval_run.candidate_counts = candidate_counts
        strategies = list(getattr(retrieval_run, "strategies_used", []) or [])
        if "status_workbook_extractor" not in strategies:
            strategies.append("status_workbook_extractor")
        retrieval_run.strategies_used = strategies
        sources_used = list(getattr(retrieval_run, "sources_used", []) or [])
        for task in tasks:
            if task.title and task.title not in sources_used:
                sources_used.append(task.title)
        retrieval_run.sources_used = sources_used
        handoff_record.update(
            {
                "status": "ok",
                "source": "deterministic_status_extractor",
                "result_count": len(status_results),
                "synthetic_doc_count": len(status_docs),
                "warnings": list(status_warnings[:8]),
            }
        )
        retrieval_run.retrieval_verification = {
            **dict(getattr(retrieval_run, "retrieval_verification", {}) or {}),
            "tabular_handoff": handoff_record,
        }
        return selected_docs, list(dict.fromkeys([*warnings, *status_warnings]))

    if runtime_bridge is None or not hasattr(runtime_bridge, "run_tabular_evidence_tasks"):
        if status_warnings:
            warnings.append("STATUS_WORKBOOK_EXTRACTOR_WARNINGS")
        handoff_record.update(
            {
                "status": "no_structured_evidence",
                "source": "deterministic_status_extractor",
                "warnings": list(status_warnings[:8]),
            }
        )
        retrieval_run.retrieval_verification = {
            **dict(getattr(retrieval_run, "retrieval_verification", {}) or {}),
            "tabular_handoff": handoff_record,
        }
        return selected_docs, list(dict.fromkeys(warnings))

    handoff_record["fallback_to_worker"] = True
    started = time.perf_counter()
    try:
        batch = runtime_bridge.run_tabular_evidence_tasks(tasks)
    except Exception as exc:
        _record_stage_timing(retrieval_run, "tabular_handoff", (time.perf_counter() - started) * 1000.0)
        handoff_record.update({"status": "failed", "error": str(exc)})
        retrieval_run.retrieval_verification = {
            **dict(getattr(retrieval_run, "retrieval_verification", {}) or {}),
            "tabular_handoff": handoff_record,
        }
        return selected_docs, ["TABULAR_HANDOFF_FAILED"]

    batch_results = list(getattr(batch, "results", []) or [])
    batch_warnings = [str(item) for item in (getattr(batch, "warnings", []) or []) if str(item)]
    tabular_docs = tabular_evidence_results_to_documents(batch_results, tasks)
    _record_stage_timing(retrieval_run, "tabular_handoff", (time.perf_counter() - started) * 1000.0)
    if tabular_docs:
        selected_docs = _merge_unique_documents(tabular_docs, selected_docs)
        candidate_counts = dict(getattr(retrieval_run, "candidate_counts", {}) or {})
        candidate_counts["tabular_evidence_docs"] = len(tabular_docs)
        retrieval_run.candidate_counts = candidate_counts
        strategies = list(getattr(retrieval_run, "strategies_used", []) or [])
        if "tabular_analyst" not in strategies:
            strategies.append("tabular_analyst")
        retrieval_run.strategies_used = strategies
        sources_used = list(getattr(retrieval_run, "sources_used", []) or [])
        for task in tasks:
            if task.title and task.title not in sources_used:
                sources_used.append(task.title)
        retrieval_run.sources_used = sources_used
        handoff_record.update(
            {
                "status": "ok",
                "result_count": len(batch_results),
                "synthetic_doc_count": len(tabular_docs),
                "warnings": list(batch_warnings[:8]),
            }
        )
    else:
        handoff_record.update(
            {
                "status": "no_structured_evidence",
                "result_count": len(batch_results),
                "warnings": list(batch_warnings[:8]),
            }
        )
        warnings.append("TABULAR_HANDOFF_NO_STRUCTURED_EVIDENCE")

    if batch_warnings:
        warnings.append("TABULAR_HANDOFF_WARNINGS")
    retrieval_run.retrieval_verification = {
        **dict(getattr(retrieval_run, "retrieval_verification", {}) or {}),
        "tabular_handoff": handoff_record,
    }
    return selected_docs, list(dict.fromkeys(warnings))


def _budget_requires_extractive_fallback(retrieval_run: Any, settings: Any) -> bool:
    if not bool(getattr(settings, "rag_extractive_fallback_enabled", True)):
        return False
    if bool(getattr(retrieval_run, "budget_exhausted", False)):
        return True
    budget_ms = int(getattr(retrieval_run, "budget_ms", 0) or 0)
    if budget_ms <= 0:
        return False
    reserve_ms = int(getattr(settings, "rag_budget_synthesis_reserve_ms", 30_000) or 30_000)
    elapsed_ms = sum(float(value or 0.0) for value in dict(getattr(retrieval_run, "stage_timings_ms", {}) or {}).values())
    if elapsed_ms + reserve_ms >= budget_ms:
        try:
            retrieval_run.budget_exhausted = True
        except Exception:
            pass
        return True
    return False


def _format_ambiguous_doc_candidate(candidate: Any) -> str:
    title = str(getattr(candidate, "title", "") or "").strip() or str(
        getattr(candidate, "match_name", "") or "document"
    ).strip() or "document"
    details: List[str] = []
    source_type = str(getattr(candidate, "source_type", "") or "").strip()
    collection_id = str(getattr(candidate, "collection_id", "") or "").strip()
    source_path = str(getattr(candidate, "source_path", "") or "").strip()
    if source_type:
        details.append(f"scope {source_type}")
    if collection_id:
        details.append(f"collection {collection_id}")
    if source_path:
        details.append(f"path {source_path}")
    if not details:
        return title
    return f"{title} ({'; '.join(details)})"


def _format_ambiguous_doc_request(match: Any) -> str:
    candidates = list(getattr(match, "candidates", ()) or [])
    candidate_text = ", ".join(
        _format_ambiguous_doc_candidate(candidate)
        for candidate in candidates[:5]
    )
    if len(candidates) > 5:
        candidate_text = f"{candidate_text}, +{len(candidates) - 5} more"
    requested_name = str(getattr(match, "requested_name", "") or "document").strip() or "document"
    if not candidate_text:
        return requested_name
    return f"{requested_name} [{candidate_text}]"


def _requested_doc_resolution_answer(resolution: IndexedDocResolution) -> Dict[str, Any]:
    missing = ", ".join(item.requested_name for item in resolution.missing)
    ambiguous_names = ", ".join(item.requested_name for item in resolution.ambiguous)
    ambiguous_details = " ".join(
        _format_ambiguous_doc_request(item) for item in resolution.ambiguous
    ).strip()
    parts: List[str] = []
    warnings: List[str] = []
    if missing:
        parts.append(
            f"I could not find indexed documents matching: {missing}. "
            "Those files may not be indexed in the current collection yet."
        )
        warnings.append("REQUESTED_DOCS_NOT_INDEXED")
    if ambiguous_names:
        parts.append(
            f"These document names were ambiguous in the current index: {ambiguous_names}. "
            f"Candidates: {ambiguous_details}. Use an exact title or a more specific path/basename."
        )
        warnings.append("REQUESTED_DOCS_AMBIGUOUS")
    return {
        "answer": " ".join(parts).strip(),
        "used_citation_ids": [],
        "followups": [],
        "warnings": warnings,
        "confidence_hint": 0.0,
    }


def _apply_requested_doc_resolution_note(
    answer_payload: Dict[str, Any],
    *,
    resolution: IndexedDocResolution,
) -> Dict[str, Any]:
    if not resolution.missing and not resolution.ambiguous:
        return answer_payload
    prefix_parts: List[str] = []
    warnings = [str(item) for item in (answer_payload.get("warnings") or []) if str(item)]
    if resolution.missing:
        prefix_parts.append(
            "Missing indexed docs: " + ", ".join(item.requested_name for item in resolution.missing) + "."
        )
        if "REQUESTED_DOCS_NOT_INDEXED" not in warnings:
            warnings.append("REQUESTED_DOCS_NOT_INDEXED")
    if resolution.ambiguous:
        prefix_parts.append(
            "Ambiguous doc names: " + ", ".join(item.requested_name for item in resolution.ambiguous) + "."
        )
        if "REQUESTED_DOCS_AMBIGUOUS" not in warnings:
            warnings.append("REQUESTED_DOCS_AMBIGUOUS")
    answer = str(answer_payload.get("answer") or "").strip()
    combined = " ".join(prefix_parts).strip()
    if answer:
        combined = f"{combined}\n\n{answer}".strip()
    return {
        **answer_payload,
        "answer": combined,
        "warnings": warnings,
    }


def _merge_unique_documents(primary: List[Any], secondary: List[Any]) -> List[Any]:
    merged: List[Any] = []
    seen: set[str] = set()
    for doc in [*primary, *secondary]:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        chunk_id = str(metadata.get("chunk_id") or "")
        key = chunk_id or f"{metadata.get('doc_id')}#{metadata.get('chunk_index')}"
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(doc)
    return merged


def _ensure_target_doc_coverage(
    *,
    settings: Any,
    stores: Any,
    session: Any,
    question: str,
    selected_docs: List[Any],
    graded: List[GradedChunk],
    resolution: IndexedDocResolution,
) -> List[Any]:
    resolved_doc_ids = resolution.resolved_doc_ids
    if not resolved_doc_ids:
        return selected_docs

    per_doc_candidates: Dict[str, List[Any]] = {doc_id: [] for doc_id in resolved_doc_ids}
    for item in graded:
        doc_id = str((item.doc.metadata or {}).get("doc_id") or "")
        if doc_id in per_doc_candidates and item.relevance >= 1:
            per_doc_candidates[doc_id].append(item.doc)

    adapter = CorpusRetrievalAdapter(stores, settings=settings, session=session)
    guaranteed: List[Any] = []
    for doc_id in resolved_doc_ids:
        candidates = per_doc_candidates.get(doc_id) or []
        if candidates:
            guaranteed.extend(candidates[:2])
            continue
        fallback_docs = adapter.read_document(doc_id, focus=question, max_chunks=2)
        guaranteed.extend(fallback_docs[:2])
    return _merge_unique_documents(guaranteed, selected_docs)


def run_rag_contract(
    settings: Any,
    stores: Any,
    *,
    providers: Any,
    session: Any,
    query: str,
    conversation_context: str,
    preferred_doc_ids: list[str],
    must_include_uploads: bool,
    top_k_vector: int,
    top_k_keyword: int,
    max_retries: int,
    callbacks: list[Any] | None = None,
    base_guidance: str = "",
    skill_context: str = "",
    task_context: str = "",
    search_mode: str = "auto",
    max_search_rounds: int = 0,
    research_profile: str = "",
    coverage_goal: str = "",
    result_mode: str = "",
    controller_hints: Dict[str, Any] | None = None,
    runtime_bridge: RagRuntimeBridge | None = None,
    progress_emitter: Any | None = None,
    event_sink: Any | None = None,
    allow_internal_fanout: bool = True,
    answer_contract: Any | None = None,
) -> RagContract:
    original_query = str(query or "")
    retrieval_query = normalize_retrieval_question(original_query)
    kb_scope = sync_session_kb_collection_state(settings, stores, session, query=retrieval_query)
    resolved_research_profile = normalize_research_profile(research_profile)
    resolved_coverage_goal = normalize_coverage_goal(coverage_goal)
    resolved_result_mode = normalize_result_mode(result_mode)
    resolved_controller_hints = coerce_controller_hints(controller_hints)
    session_metadata = dict(getattr(session, "metadata", {}) or {})
    resolved_turn_intent = (
        dict(session_metadata.get("resolved_turn_intent") or {})
        if isinstance(session_metadata.get("resolved_turn_intent"), dict)
        else {}
    )
    if answer_contract is None:
        maybe_contract = resolved_turn_intent.get("answer_contract")
        answer_contract = dict(maybe_contract) if isinstance(maybe_contract, dict) else maybe_contract
    presentation_preferences = (
        dict(resolved_turn_intent.get("presentation_preferences") or {})
        if isinstance(resolved_turn_intent.get("presentation_preferences"), dict)
        else {}
    )
    if answer_contract is not None and not answer_contract_allows_inventory(answer_contract):
        if resolved_result_mode == "inventory" or bool(resolved_controller_hints.get("prefer_inventory_output")):
            resolved_result_mode = "comparison" if answer_contract_kind(answer_contract) == "comparison" else "answer"
            for key in (
                "prefer_inventory_output",
                "prefer_session_access_inventory",
                "inventory_query_type",
                "graph_inventory_only",
            ):
                resolved_controller_hints.pop(key, None)
    tenant_id = _tenant_id(settings, session)
    route_context = dict(session_metadata.get("route_context") or {})
    resolved_controller_hints = {
        **deep_rag_controller_hints(route_context),
        **dict(resolved_controller_hints or {}),
    }
    effective_search_mode = deep_rag_search_mode(route_context, default=search_mode or "auto")
    has_uploads = has_upload_evidence(session)
    inventory_query_type = classify_inventory_query(original_query)
    if not inventory_query_type or inventory_query_type == INVENTORY_QUERY_NONE:
        inventory_query_type = classify_inventory_query(retrieval_query)
    if inventory_query_type in {
        INVENTORY_QUERY_SESSION_ACCESS,
        INVENTORY_QUERY_KB_COLLECTIONS,
        INVENTORY_QUERY_KB_FILE,
        INVENTORY_QUERY_GRAPH_FILE,
        INVENTORY_QUERY_GRAPH_INDEXES,
    } and not inventory_query_requests_grounded_analysis(original_query, query_type=inventory_query_type):
        dispatched = dispatch_authoritative_inventory(
            settings,
            stores,
            session,
            query=original_query,
            query_type=inventory_query_type,
        )
        if bool(dispatched.get("handled")):
            return _early_contract(
                original_query,
                dict(dispatched.get("answer") or {}),
                search_mode="metadata_inventory",
            )

    kb_collection_id = str(kb_scope.get("kb_collection_id") or resolve_kb_collection_id(settings, session))
    available_kb_collection_ids = [
        str(item).strip()
        for item in (kb_scope.get("available_kb_collection_ids") or [])
        if str(item).strip()
    ]
    kb_collection_confirmed = bool(kb_scope.get("kb_collection_confirmed"))
    preliminary_scope = decide_retrieval_scope(
        settings,
        session,
        query=retrieval_query,
        kb_available=False,
        has_uploads=has_uploads,
    )
    if preliminary_scope.mode in {"kb_only", "both"} and len(available_kb_collection_ids) > 1 and not kb_collection_confirmed:
        return _early_contract(
            query,
            _ambiguous_kb_collection_answer(available_kb_collection_ids),
            search_mode="none",
        )
    kb_status = None
    should_check_kb = preliminary_scope.mode not in {"none"} and preliminary_scope.reason not in {
        "explicit_uploads_only",
    }
    if should_check_kb:
        kb_status = get_collection_readiness_status(
            settings,
            stores,
            tenant_id=tenant_id,
            collection_id=kb_collection_id,
        )
    scope_decision = decide_retrieval_scope(
        settings,
        session,
        query=retrieval_query,
        kb_available=bool(kb_status.ready) if kb_status is not None else False,
        has_uploads=has_uploads,
    )
    session.metadata = {
        **dict(getattr(session, "metadata", {}) or {}),
        **scope_decision.to_metadata(),
    }
    resolved_controller_hints = {
        **dict(resolved_controller_hints or {}),
        **scope_decision.to_metadata(),
    }
    active_graph_ids = [
        str(item).strip()
        for item in (dict(getattr(session, "metadata", {}) or {}).get("active_graph_ids") or [])
        if str(item).strip()
    ]
    if active_graph_ids:
        resolved_controller_hints = {
            **dict(resolved_controller_hints or {}),
            "graph_ids": list(active_graph_ids),
            "planned_graph_ids": list(active_graph_ids),
        }
    effective_preferred_doc_ids = list(preferred_doc_ids or [])

    if scope_decision.mode == "ambiguous":
        return _early_contract(query, _ambiguous_scope_answer(scope_decision), search_mode="none")

    if scope_decision.mode == "none":
        return _early_contract(query, _retrieval_disabled_answer(), search_mode="none")

    if scope_decision.mode in {"uploads_only", "both"} and not has_uploads:
        return _early_contract(
            query,
            _uploads_missing_answer(allow_kb_fallback=scope_decision.mode == "uploads_only"),
            search_mode="none",
        )

    if scope_decision.mode in {"kb_only", "both"} and kb_status is not None and not kb_status.ready:
        return _early_contract(query, _kb_not_ready_answer(kb_status), search_mode="none")

    requested_doc_resolution = IndexedDocResolution()
    if scope_decision.mode in {"kb_only", "both"}:
        requested_doc_resolution = resolve_query_document_targets(
            settings,
            stores,
            session,
            query=retrieval_query,
        )
        if requested_doc_resolution.requested_names:
            _emit_progress(
                progress_emitter,
                "doc_focus",
                label="Resolving named documents",
                detail=", ".join(requested_doc_resolution.requested_names[:4]),
                agent="rag_worker",
                docs=[
                    {
                        "doc_id": item.doc_id,
                        "title": item.title,
                        "source_path": item.source_path,
                        "source_type": item.source_type,
                    }
                    for item in requested_doc_resolution.resolved[:6]
                ],
                why="The request named specific indexed files, so retrieval is being scoped to those docs first.",
            )
            if not requested_doc_resolution.resolved_doc_ids:
                return _early_contract(
                    query,
                    _requested_doc_resolution_answer(requested_doc_resolution),
                    search_mode="none",
                )
            resolved_controller_hints = {
                **dict(resolved_controller_hints or {}),
                "prefer_doc_focus": True,
                "explicit_doc_targets": list(requested_doc_resolution.requested_names),
                "resolved_doc_ids": list(requested_doc_resolution.resolved_doc_ids),
            }

    if scope_decision.mode in {"kb_only", "both"} and not requested_doc_resolution.resolved_doc_ids:
        requested_collection_id = str(
            resolved_controller_hints.get("requested_kb_collection_id")
            or dict(getattr(session, "metadata", {}) or {}).get("requested_kb_collection_id")
            or ""
        ).strip()
        collection_selection = select_collection_for_query(
            stores,
            settings,
            session,
            retrieval_query,
            source_type="all" if scope_decision.mode == "both" else "kb",
            explicit_collection_id=requested_collection_id,
            event_sink=event_sink,
        )
        resolved_controller_hints = {
            **dict(resolved_controller_hints or {}),
            "collection_selection": collection_selection.to_dict(),
        }
        if collection_selection.resolved:
            apply_selection_to_session(session, collection_selection)
            resolved_controller_hints = {
                **dict(resolved_controller_hints or {}),
                "requested_kb_collection_id": collection_selection.selected_collection_id,
                "kb_collection_id": collection_selection.selected_collection_id,
                "search_collection_ids": [collection_selection.selected_collection_id],
            }
        elif scope_decision.mode == "kb_only":
            return _early_contract(query, selection_answer(collection_selection), search_mode="none")

    if scope_decision.mode == "uploads_only":
        effective_preferred_doc_ids = list(getattr(session, "uploaded_doc_ids", []) or [])
    elif requested_doc_resolution.resolved_doc_ids:
        effective_preferred_doc_ids = list(requested_doc_resolution.resolved_doc_ids)

    retrieval_run = run_retrieval_controller(
        settings,
        stores,
        providers=providers,
        session=session,
        query=retrieval_query,
        conversation_context=conversation_context,
        preferred_doc_ids=effective_preferred_doc_ids,
        must_include_uploads=must_include_uploads,
        top_k_vector=top_k_vector,
        top_k_keyword=top_k_keyword,
        max_retries=max_retries,
        callbacks=callbacks or [],
        search_mode=effective_search_mode,
        max_search_rounds=max_search_rounds,
        research_profile=research_profile,
        coverage_goal=resolved_coverage_goal,
        result_mode=resolved_result_mode,
        controller_hints=resolved_controller_hints,
        runtime_bridge=runtime_bridge,
        progress_emitter=progress_emitter,
        event_sink=event_sink,
        allow_internal_fanout=allow_internal_fanout,
    )
    selected_docs = list(retrieval_run.selected_docs)

    if hasattr(session, "scratchpad") and isinstance(getattr(session, "scratchpad", None), dict):
        try:
            session.scratchpad["rag_last_evidence_ledger"] = json.dumps(
                retrieval_run.evidence_ledger,
                ensure_ascii=False,
            )
            session.scratchpad["rag_last_search_mode"] = str(retrieval_run.search_mode)
            session.scratchpad["rag_last_requested_doc_resolution"] = json.dumps(
                {
                    "mode": (
                        "explicit_doc_targets"
                        if requested_doc_resolution.resolved_doc_ids
                        else "generic_retrieval"
                    ),
                    **requested_doc_resolution.to_dict(),
                },
                ensure_ascii=False,
            )
            session.scratchpad["rag_last_retrieval_verification"] = json.dumps(
                dict(getattr(retrieval_run, "retrieval_verification", {}) or {}),
                ensure_ascii=False,
            )
        except Exception:
            pass

    selected_docs = _ensure_target_doc_coverage(
        settings=settings,
        stores=stores,
        session=session,
        question=retrieval_query,
        selected_docs=selected_docs,
        graded=list(retrieval_run.graded),
        resolution=requested_doc_resolution,
    )
    selected_docs, tabular_warnings = _augment_with_tabular_evidence(
        settings=settings,
        query=retrieval_query,
        selected_docs=selected_docs,
        retrieval_run=retrieval_run,
        runtime_bridge=runtime_bridge,
        progress_emitter=progress_emitter,
    )

    inventory_mode = False
    if not selected_docs:
        if scope_decision.mode in {"kb_only", "both"} and kb_status is not None and not kb_status.ready:
            answer_payload = _kb_not_ready_answer(kb_status)
        else:
            negative_reporting = (
                resolved_coverage_goal in {"corpus_wide", "exhaustive"}
                or bool(resolved_controller_hints.get("prefer_negative_evidence_reporting"))
            )
            if negative_reporting:
                answer_payload = _build_negative_evidence_answer(query, retrieval_run)
            else:
                if progress_emitter is not None and hasattr(progress_emitter, "emit_progress"):
                    progress_emitter.emit_progress(
                        "phase_start",
                        label="Synthesizing answer",
                        detail="Grounding final response",
                        agent="rag_worker",
                    )
                synthesis_started = time.perf_counter()
                answer_payload = generate_grounded_answer(
                    providers.chat,
                    settings=settings,
                    question=query,
                    conversation_context=_answer_context(
                        query,
                        conversation_context,
                        retrieval_run.evidence_ledger,
                        base_guidance=base_guidance,
                        skill_context=skill_context,
                        task_context=task_context,
                        coverage_goal=resolved_coverage_goal,
                        result_mode=resolved_result_mode,
                        controller_hints=resolved_controller_hints,
                        answer_contract=answer_contract,
                        presentation_preferences=presentation_preferences,
                    ),
                    evidence_docs=selected_docs,
                    callbacks=callbacks or [],
                )
                _record_stage_timing(retrieval_run, "synthesis", (time.perf_counter() - synthesis_started) * 1000.0)
    else:
        inventory_mode = resolved_result_mode == "inventory" or bool(resolved_controller_hints.get("prefer_inventory_output"))
        discovery_inventory_mode = _is_corpus_discovery_inventory(
            query=query,
            research_profile=resolved_research_profile,
            coverage_goal=resolved_coverage_goal,
            result_mode=resolved_result_mode,
            controller_hints=resolved_controller_hints,
        )
        rejected_discovery_candidates: List[Dict[str, Any]] = []
        downgraded_inventory_reason = ""
        if inventory_mode and discovery_inventory_mode:
            confirmed_docs, rejected_discovery_candidates = _filter_confirmed_discovery_docs(query, selected_docs)
            retrieval_run.candidate_counts["confirmed_match_count"] = len(confirmed_docs)
            retrieval_verification = dict(getattr(retrieval_run, "retrieval_verification", {}) or {})
            if rejected_discovery_candidates:
                retrieval_verification["rejected_candidate_reasons"] = rejected_discovery_candidates
            retrieval_verification["confirmed_match_count"] = len(confirmed_docs)
            generated_subqueries = [
                str(item)
                for summary in (retrieval_run.evidence_ledger or {}).get("round_summaries", [])
                for item in (summary.get("queries") or [])
                if str(item)
            ]
            if generated_subqueries:
                retrieval_verification["generated_subqueries"] = generated_subqueries
            retrieval_run.retrieval_verification = retrieval_verification
            selected_docs = confirmed_docs
            if not selected_docs:
                downgraded_inventory_reason = "no_confirmed_topic_matches"
                retrieval_run.retrieval_verification = {
                    **dict(getattr(retrieval_run, "retrieval_verification", {}) or {}),
                    "downgraded_to_negative_evidence": True,
                    "downgrade_reason": downgraded_inventory_reason,
                }

        if inventory_mode:
            if downgraded_inventory_reason:
                answer_payload = _build_no_confirmed_discovery_matches_answer(
                    query,
                    retrieval_run,
                    rejected_candidates=rejected_discovery_candidates,
                )
            else:
                answer_payload = _build_inventory_answer(selected_docs)
        else:
            if progress_emitter is not None and hasattr(progress_emitter, "emit_progress"):
                progress_emitter.emit_progress(
                    "phase_start",
                    label="Synthesizing answer",
                    detail="Grounding final response",
                    agent="rag_worker",
                )
            if _budget_requires_extractive_fallback(retrieval_run, settings):
                answer_payload = build_extractive_grounded_answer(
                    query,
                    selected_docs,
                    warning="BUDGET_EXTRACTIVE_FALLBACK",
                )
            else:
                synthesis_started = time.perf_counter()
                answer_payload = generate_grounded_answer(
                    providers.chat,
                    settings=settings,
                    question=query,
                    conversation_context=_answer_context(
                        query,
                        conversation_context,
                        retrieval_run.evidence_ledger,
                        base_guidance=base_guidance,
                        skill_context=skill_context,
                        task_context=task_context,
                        coverage_goal=resolved_coverage_goal,
                        result_mode=resolved_result_mode,
                        controller_hints=resolved_controller_hints,
                        answer_contract=answer_contract,
                        presentation_preferences=presentation_preferences,
                    ),
                    evidence_docs=selected_docs,
                    callbacks=callbacks or [],
                )
                _record_stage_timing(retrieval_run, "synthesis", (time.perf_counter() - synthesis_started) * 1000.0)
        if inventory_mode and discovery_inventory_mode:
            retrieval_verification = dict(getattr(retrieval_run, "retrieval_verification", {}) or {})
            if str(retrieval_verification.get("status") or "").lower() == "revise":
                retrieval_run.retrieval_verification = {
                    **retrieval_verification,
                    "downgraded_to_negative_evidence": True,
                    "downgrade_reason": "retrieval_verification_failed",
                }
                selected_docs = []
                answer_payload = _build_no_confirmed_discovery_matches_answer(
                    query,
                    retrieval_run,
                    rejected_candidates=rejected_discovery_candidates,
                    verification_failed=True,
                )
    structured_regenerate_answer = None
    if not inventory_mode and selected_docs and not _budget_requires_extractive_fallback(retrieval_run, settings):
        def structured_regenerate_answer(candidate: Dict[str, Any], missing_values: list[Dict[str, str]]) -> Dict[str, Any]:
            feedback = _structured_evidence_verifier_feedback(candidate, missing_values)
            regeneration_started = time.perf_counter()
            payload = generate_grounded_answer(
                providers.chat,
                settings=settings,
                question=query,
                conversation_context=(
                    _answer_context(
                        query,
                        conversation_context,
                        retrieval_run.evidence_ledger,
                        base_guidance=base_guidance,
                        skill_context=skill_context,
                        task_context=task_context,
                        coverage_goal=resolved_coverage_goal,
                        result_mode=resolved_result_mode,
                        controller_hints=resolved_controller_hints,
                        answer_contract=answer_contract,
                        presentation_preferences=presentation_preferences,
                    )
                    + "\n\n"
                    + feedback
                ),
                evidence_docs=selected_docs,
                callbacks=callbacks or [],
            )
            _record_stage_timing(
                retrieval_run,
                "structured_evidence_regeneration",
                (time.perf_counter() - regeneration_started) * 1000.0,
            )
            return payload

    answer_payload = _apply_structured_evidence_arbitration(
        answer_payload,
        query=query,
        selected_docs=selected_docs,
        retrieval_run=retrieval_run,
        regenerate_answer=structured_regenerate_answer,
    )
    answer_payload = _apply_requested_doc_resolution_note(
        answer_payload,
        resolution=requested_doc_resolution,
    )
    if tabular_warnings:
        existing_warnings = [str(item) for item in (answer_payload.get("warnings") or []) if str(item)]
        for warning in tabular_warnings:
            if warning not in existing_warnings:
                existing_warnings.append(warning)
        answer_payload = {
            **answer_payload,
            "warnings": existing_warnings,
        }
    citation_started = time.perf_counter()
    citations = build_citations(
        selected_docs,
        url_resolver=make_document_source_url_resolver(settings, session),
    )
    _record_stage_timing(retrieval_run, "citation_building", (time.perf_counter() - citation_started) * 1000.0)
    inventory_mode = resolved_result_mode == "inventory" or bool(resolved_controller_hints.get("prefer_inventory_output"))
    used_citation_ids = [
        citation_id
        for citation_id in answer_payload.get("used_citation_ids", [])
        if citation_id in {citation.citation_id for citation in citations}
        ]
    if not used_citation_ids:
        if inventory_mode:
            used_citation_ids = [citation.citation_id for citation in citations[: min(4, len(citations))]]
        elif citations:
            answer_payload = _build_cited_evidence_fallback_answer(query, selected_docs)
            used_citation_ids = [
                citation_id
                for citation_id in answer_payload.get("used_citation_ids", [])
                if citation_id in {citation.citation_id for citation in citations}
            ]
    elif not inventory_mode:
        used_citation_ids = _augment_used_citation_ids(
            query=query,
            answer=str(answer_payload.get("answer") or ""),
            docs=selected_docs,
            used_citation_ids=used_citation_ids,
        )

    if progress_emitter is not None and hasattr(progress_emitter, "emit_progress"):
        progress_emitter.emit_progress(
            "summary",
            label="Grounded answer ready",
            detail=f"{len(used_citation_ids)} citation(s) selected",
            agent="rag_worker",
            docs=[
                {
                    "doc_id": citation.doc_id,
                    "title": citation.title,
                    "source_path": "",
                    "source_type": citation.source_type,
                    "collection_id": citation.collection_id,
                }
                for citation in citations[:6]
            ],
            counts={"citations": len(citations)},
        )

    warnings = [str(item) for item in (answer_payload.get("warnings") or []) if str(item)]
    retrieval_verification = dict(getattr(retrieval_run, "retrieval_verification", {}) or {})
    suppress_verification_warning = bool(retrieval_verification.get("downgraded_to_negative_evidence"))
    if str(retrieval_verification.get("status") or "").lower() == "revise" and not suppress_verification_warning:
        if "RETRIEVAL_VERIFICATION_ISSUES" not in warnings:
            warnings.append("RETRIEVAL_VERIFICATION_ISSUES")

    return RagContract(
        answer=str(answer_payload.get("answer") or ""),
        citations=citations,
        used_citation_ids=used_citation_ids,
        confidence=float(answer_payload.get("confidence_hint") or 0.0),
        retrieval_summary=retrieval_run.to_summary(citations_found=len(citations)),
        followups=[str(item) for item in (answer_payload.get("followups") or []) if str(item)],
        warnings=warnings,
    )


def _answer_context(
    question: str,
    conversation_context: str,
    evidence_ledger: Dict[str, Any],
    *,
    base_guidance: str = "",
    skill_context: str = "",
    task_context: str = "",
    coverage_goal: str = "",
    result_mode: str = "",
    controller_hints: Dict[str, Any] | None = None,
    answer_contract: Any | None = None,
    presentation_preferences: Dict[str, Any] | None = None,
) -> str:
    context = str(conversation_context or "").strip()
    normalized_controller_hints = coerce_controller_hints(controller_hints)
    detailed_doc_focus_mode = (
        str(normalized_controller_hints.get("summary_scope") or "").strip().lower() == "active_doc_focus"
        or bool(normalized_controller_hints.get("prefer_detailed_synthesis"))
    )
    if prefers_bounded_synthesis(question) and not detailed_doc_focus_mode:
        context = (
            f"{context}\n\n"
            "Formatting directive: answer with a direct synthesis, then supporting bullets or a compact table "
            "covering the main implementation details when they help, and end with a `Sources:` line naming the "
            "most relevant documents used. Do not switch into a per-document inventory unless the "
            "user explicitly asked to list or identify documents."
        ).strip()
    if detailed_doc_focus_mode:
        context = (
            f"{context}\n\n"
            "Formatting directive: provide a detailed subsystem-organized synthesis grounded in the scoped documents. "
            "Merge overlapping evidence across documents, name which documents support each subsystem, and call out any thin or conflicting support instead of omitting it."
        ).strip()
    if re.search(r"\b(identify|list|which|find)\s+(?:all\s+)?(?:documents|files)\b", question, re.IGNORECASE):
        extra = (
            "Retrieval directive: this is a corpus-discovery request. "
            "Prefer a per-document answer that names relevant titles or file names with short grounded justifications."
        )
        context = f"{context}\n\n{extra}".strip()
    if normalize_result_mode(result_mode) == "inventory" or bool(normalized_controller_hints.get("prefer_inventory_output")):
        context = (
            f"{context}\n\n"
            "Formatting directive: prefer a per-document inventory with file titles and short grounded evidence."
        ).strip()
    if _requires_mermaid_output(
        question,
        answer_contract=answer_contract,
        presentation_preferences=presentation_preferences,
    ):
        context = (
            f"{context}\n\n"
            "Formatting directive: produce the final answer as exactly one fenced ```mermaid code block, "
            "using simple Mermaid-safe node labels derived from the evidence. Put citations, source notes, "
            "warnings, and uncertainty outside the code block under a short Grounding notes section; do not place "
            "citation IDs or markdown links inside Mermaid node labels."
        ).strip()
    elif _prefers_mermaid_output(
        question,
        answer_contract=answer_contract,
        presentation_preferences=presentation_preferences,
    ):
        context = (
            f"{context}\n\n"
            "Formatting directive: include a Mermaid diagram when it helps communicate the grounded workflow or architecture. "
            "Keep citations and grounding notes outside the Mermaid code block."
        ).strip()
    if normalize_coverage_goal(coverage_goal) in {"corpus_wide", "exhaustive"}:
        context = (
            f"{context}\n\n"
            "Coverage directive: avoid overclaiming corpus-wide completeness unless the evidence clearly supports it."
        ).strip()
    if base_guidance:
        context = f"{context}\n\nBase guidance:\n{base_guidance}".strip()
    if task_context:
        context = f"{context}\n\nTask focus:\n{task_context}".strip()
    if skill_context:
        context = f"{context}\n\nSkill guidance:\n{skill_context}".strip()
    rounds = list((evidence_ledger or {}).get("round_summaries") or [])
    if rounds:
        tried = []
        for round_summary in rounds[:3]:
            for item in round_summary.get("queries") or []:
                if str(item) and str(item) not in tried:
                    tried.append(str(item))
        if tried:
            context = f"{context}\n\nQueries tried: {'; '.join(tried[:6])}".strip()
    claim_ledger = dict(evidence_ledger or {})
    supported_claim_ids = [str(item) for item in (claim_ledger.get("supported_claim_ids") or []) if str(item)]
    unverified_hops = [str(item) for item in (claim_ledger.get("unverified_hops") or []) if str(item)]
    if supported_claim_ids:
        context = f"{context}\n\nSupported claims: {', '.join(supported_claim_ids[:6])}".strip()
    if unverified_hops:
        context = f"{context}\n\nUnverified hops: {', '.join(unverified_hops[:4])}".strip()
    facet_coverage = dict(claim_ledger.get("facet_coverage") or {})
    if facet_coverage:
        coverage_items = []
        for value in list(facet_coverage.values())[:6]:
            label = str(value.get("label") or "").strip()
            state = str(value.get("state") or "").strip()
            if label and state:
                coverage_items.append(f"{label}: {state}")
        if coverage_items:
            context = f"{context}\n\nFacet coverage: {'; '.join(coverage_items)}".strip()
    return context


def coerce_rag_contract(contract: Dict[str, Any]) -> RagContract:
    return RagContract(
        answer=str(contract.get("answer") or ""),
        citations=[
            Citation.from_dict(dict(item))
            for item in (contract.get("citations") or [])
            if isinstance(item, dict)
        ],
        used_citation_ids=[str(item) for item in (contract.get("used_citation_ids") or []) if str(item)],
        confidence=float(contract.get("confidence") or 0.0),
        retrieval_summary=RetrievalSummary.from_dict(dict(contract.get("retrieval_summary") or {})),
        followups=[str(item) for item in (contract.get("followups") or []) if str(item)],
        warnings=[str(item) for item in (contract.get("warnings") or []) if str(item)],
    )


def _markdown_link(label: str, url: str) -> str:
    clean_url = str(url or "").strip()
    clean_label = str(label or clean_url or "source").replace("\n", " ").strip()
    if not clean_url:
        return clean_label
    escaped_label = clean_label.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
    escaped_url = clean_url.replace(" ", "%20").replace(")", "%29")
    return f"[{escaped_label}]({escaped_url})"


def render_rag_contract(contract: RagContract | Dict[str, Any]) -> str:
    raw = contract.to_dict() if isinstance(contract, RagContract) else dict(contract)
    answer = raw.get("answer", "")
    citations = raw.get("citations", [])
    used = set(raw.get("used_citation_ids", []))
    warnings = raw.get("warnings", [])
    followups = raw.get("followups", [])

    lines = [
        replace_inline_citation_ids(
            str(answer).strip(),
            citations,
            used_citation_ids=list(used),
            link_renderer=_markdown_link,
        )
    ]
    if citations:
        lines.append("\nCitations:")
        for citation in citations:
            item = citation.to_dict() if isinstance(citation, Citation) else dict(citation)
            citation_id = item.get("citation_id", "")
            if used and citation_id not in used:
                continue
            location = str(item.get("location") or "").strip()
            collection_id = str(item.get("collection_id") or "").strip()
            source_type = str(item.get("source_type") or "").strip().lower()
            details = []
            if location:
                details.append(location)
            if collection_id:
                collection_label = "KB Collection" if source_type == "kb" else "Collection"
                details.append(f"{collection_label}: {collection_id}")
            suffix = f" ({'; '.join(details)})" if details else ""
            rendered_title = _markdown_link(citation_display_label(item), str(item.get("url") or ""))
            lines.append(f"- {rendered_title}{suffix}")
    if warnings:
        lines.append("\nWarnings: " + ", ".join(str(item) for item in warnings))
    if followups:
        lines.append("\nFollow-ups:")
        for followup in followups:
            lines.append(f"- {followup}")
    return "\n".join(lines).strip()
