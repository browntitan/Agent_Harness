from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from agentic_chatbot_next.contracts.rag import Citation
from agentic_chatbot_next.rag.discovery_precision import match_discovery_topic_anchors


def _significant_terms(query: str) -> List[str]:
    stopwords = {
        "about",
        "across",
        "and",
        "for",
        "from",
        "into",
        "that",
        "the",
        "this",
        "what",
        "which",
        "with",
    }
    return [
        term
        for term in re.findall(r"[A-Za-z0-9_]{4,}", str(query or "").lower())
        if term not in stopwords
    ][:12]


def _is_document_discovery_query(query: str) -> bool:
    return bool(
        re.search(
            r"\b(which|identify|list|find)\s+(?:all\s+)?(?:documents|files)\b",
            str(query or ""),
            flags=re.IGNORECASE,
        )
    )


def verify_retrieval_quality(
    *,
    settings: Any,
    stores: Any,
    session: Any,
    query: str,
    retrieval_run: Any,
    citations: Sequence[Citation] | None = None,
    controller_hints: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    hints = dict(controller_hints or {})
    selected_docs = list(getattr(retrieval_run, "selected_docs", []) or [])
    claim_ledger = dict(getattr(retrieval_run, "claim_ledger", {}) or {})
    source_plan = dict(getattr(retrieval_run, "source_plan", {}) or {})
    issues: List[Dict[str, Any]] = []
    retry_focus: Dict[str, Any] = {"doc_ids": [], "queries": [], "sections": []}

    selected_doc_ids = {
        str((getattr(doc, "metadata", {}) or {}).get("doc_id") or "")
        for doc in selected_docs
        if str((getattr(doc, "metadata", {}) or {}).get("doc_id") or "")
    }
    requested_doc_ids = [
        str(item)
        for item in (
            hints.get("resolved_doc_ids")
            or source_plan.get("preferred_doc_ids")
            or []
        )
        if str(item)
    ]
    missing_doc_ids = [doc_id for doc_id in requested_doc_ids if doc_id not in selected_doc_ids]
    if missing_doc_ids:
        issues.append(
            {
                "check": "missed_doc_detection",
                "severity": "high",
                "detail": f"Requested or shortlisted documents were not confirmed: {', '.join(missing_doc_ids[:6])}.",
            }
        )
        retry_focus["doc_ids"] = missing_doc_ids[:6]

    unverified_hops = [str(item) for item in (claim_ledger.get("unverified_hops") or []) if str(item)]
    if unverified_hops:
        discovery_mode = _is_document_discovery_query(query)
        issues.append(
            {
                "check": "unsupported_hop_detection",
                "severity": "high",
                "detail": (
                    "The current snippets do not explicitly verify that the returned documents cover the requested topic."
                    if discovery_mode
                    else f"Relationship hops are still unverified in text evidence: {', '.join(unverified_hops[:4])}."
                ),
            }
        )
        retry_focus["queries"].extend(unverified_hops[:4])

    citations = list(citations or [])
    citation_haystack = " ".join(str(citation.snippet or "") for citation in citations)
    anchor_match = match_discovery_topic_anchors(query, citation_haystack)
    if citations and anchor_match["required_anchors"] and not anchor_match["matches"]:
        issues.append(
            {
                "check": "citation_topic_mismatch",
                "severity": "high",
                "detail": "The cited snippets do not explicitly mention the requested topic.",
            }
        )
        retry_focus["queries"].append(query)
    significant_terms = _significant_terms(query)
    if citations and significant_terms and anchor_match["matches"]:
        matched_terms = [term for term in significant_terms if term in citation_haystack.lower()]
        if len(matched_terms) < max(1, min(2, len(significant_terms))):
            issues.append(
                {
                    "check": "citation_text_mismatch",
                    "severity": "medium",
                    "detail": "The cited snippets do not explicitly cover enough of the requested topic.",
                }
            )
            retry_focus["queries"].append(query)

    graphs_considered = [str(item) for item in (getattr(retrieval_run, "graphs_considered", []) or []) if str(item)]
    graph_index_store = getattr(stores, "graph_index_store", None)
    doc_store = getattr(stores, "doc_store", None)
    tenant_id = str(getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev")) or "local-dev")
    user_id = str(getattr(session, "user_id", getattr(settings, "default_user_id", "")) or "")
    stale_graphs: List[str] = []
    if graph_index_store is not None and doc_store is not None:
        for graph_id in graphs_considered[:6]:
            try:
                graph_record = graph_index_store.get_index(graph_id, tenant_id, user_id=user_id)
            except TypeError:
                graph_record = graph_index_store.get_index(graph_id, tenant_id)
            except Exception:
                graph_record = None
            if graph_record is None:
                continue
            graph_time = str(getattr(graph_record, "last_indexed_at", "") or "")
            source_doc_ids = [str(item) for item in (getattr(graph_record, "source_doc_ids", []) or []) if str(item)]
            for doc_id in source_doc_ids[:12]:
                try:
                    record = doc_store.get_document(doc_id, tenant_id)
                except Exception:
                    record = None
                if record is None:
                    continue
                ingested_at = str(getattr(record, "ingested_at", "") or "")
                if ingested_at and graph_time and ingested_at > graph_time:
                    stale_graphs.append(graph_id)
                    break
    if stale_graphs:
        issues.append(
            {
                "check": "stale_graph_detection",
                "severity": "medium",
                "detail": f"Graph indexes may be stale relative to their source documents: {', '.join(sorted(set(stale_graphs))[:4])}.",
            }
        )

    prioritized_sections = [dict(item) for item in (source_plan.get("prioritized_sections") or []) if isinstance(item, dict)]
    if prioritized_sections and (missing_doc_ids or unverified_hops):
        retry_focus["sections"] = prioritized_sections[:6]

    return {
        "status": "revise" if issues else "pass",
        "issues": issues,
        "retryable": bool(retry_focus["doc_ids"] or retry_focus["queries"] or retry_focus["sections"]),
        "retry_focus": retry_focus,
    }


__all__ = ["verify_retrieval_quality"]
