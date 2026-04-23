from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from agentic_chatbot_next.prompting import load_judge_grading_prompt, render_template
from agentic_chatbot_next.persistence.postgres.chunks import ScoredChunk
from agentic_chatbot_next.rag.stores import KnowledgeStores
from agentic_chatbot_next.utils.json_utils import coerce_int, extract_json

__all__ = [
    "ScoredChunk",
    "GradedChunk",
    "keyword_search",
    "merge_dedupe",
    "retrieve_candidates",
    "vector_search",
    "grade_chunks",
]


@dataclass
class GradedChunk:
    doc: Document
    relevance: int
    reason: str


def _doc_key(doc: Document) -> str:
    metadata = doc.metadata or {}
    return str(metadata.get("chunk_id") or f"{metadata.get('doc_id')}#{metadata.get('chunk_index')}")


def vector_search(
    stores: KnowledgeStores,
    query: str,
    *,
    top_k: int,
    tenant_id: str,
    doc_id_filter: Optional[str] = None,
    collection_id_filter: Optional[str] = None,
    collection_ids_filter: Optional[Sequence[str]] = None,
) -> List[ScoredChunk]:
    normalized_collection_ids = _normalize_collection_ids(collection_ids_filter, collection_id_filter)
    if doc_id_filter or not normalized_collection_ids:
        return stores.chunk_store.vector_search(
            query,
            top_k=top_k,
            doc_id_filter=doc_id_filter,
            collection_id_filter=collection_id_filter,
            tenant_id=tenant_id,
        )

    results: List[ScoredChunk] = []
    for candidate_collection_id in normalized_collection_ids:
        results.extend(
            stores.chunk_store.vector_search(
                query,
                top_k=top_k,
                doc_id_filter=doc_id_filter,
                collection_id_filter=candidate_collection_id,
                tenant_id=tenant_id,
            )
        )
    return merge_dedupe(results)


def keyword_search(
    stores: KnowledgeStores,
    query: str,
    *,
    top_k: int,
    tenant_id: str,
    doc_id_filter: Optional[str] = None,
    collection_id_filter: Optional[str] = None,
    collection_ids_filter: Optional[Sequence[str]] = None,
) -> List[ScoredChunk]:
    normalized_collection_ids = _normalize_collection_ids(collection_ids_filter, collection_id_filter)
    if doc_id_filter or not normalized_collection_ids:
        return stores.chunk_store.keyword_search(
            query,
            top_k=top_k,
            doc_id_filter=doc_id_filter,
            collection_id_filter=collection_id_filter,
            tenant_id=tenant_id,
        )

    results: List[ScoredChunk] = []
    for candidate_collection_id in normalized_collection_ids:
        results.extend(
            stores.chunk_store.keyword_search(
                query,
                top_k=top_k,
                doc_id_filter=doc_id_filter,
                collection_id_filter=candidate_collection_id,
                tenant_id=tenant_id,
            )
        )
    return merge_dedupe(results)


def merge_dedupe(chunks: Sequence[ScoredChunk]) -> List[ScoredChunk]:
    by_key: Dict[str, ScoredChunk] = {}
    for chunk in chunks:
        key = _doc_key(chunk.doc)
        if key not in by_key or chunk.score > by_key[key].score:
            by_key[key] = chunk
    return sorted(by_key.values(), key=lambda chunk: chunk.score, reverse=True)


def _title_matched_doc_ids(
    stores: KnowledgeStores,
    query: str,
    *,
    tenant_id: str,
    preferred_doc_ids: Sequence[str],
    collection_id_filter: Optional[str],
    collection_ids_filter: Optional[Sequence[str]] = None,
    limit: int = 3,
) -> List[str]:
    normalized_collection_ids = _normalize_collection_ids(collection_ids_filter, collection_id_filter)
    try:
        matches: List[Dict[str, Any]] = []
        if normalized_collection_ids:
            for candidate_collection_id in normalized_collection_ids:
                matches.extend(
                    stores.doc_store.fuzzy_search_title(
                        query,
                        tenant_id=tenant_id,
                        limit=max(1, limit * 2),
                        collection_id=candidate_collection_id,
                    )
                )
        else:
            matches = stores.doc_store.fuzzy_search_title(
                query,
                tenant_id=tenant_id,
                limit=max(1, limit * 2),
                collection_id=collection_id_filter or "",
            )
    except Exception:
        return []

    allowed = set(preferred_doc_ids)
    doc_ids: List[str] = []
    seen: set[str] = set()
    for item in matches:
        doc_id = str(item.get("doc_id") or "")
        if not doc_id or doc_id in seen:
            continue
        if allowed and doc_id not in allowed:
            continue
        seen.add(doc_id)
        doc_ids.append(doc_id)
        if len(doc_ids) >= limit:
            break
    return doc_ids


def _normalize_collection_ids(
    collection_ids_filter: Optional[Sequence[str]],
    collection_id_filter: Optional[str],
) -> List[str]:
    seen: set[str] = set()
    normalized: List[str] = []
    for raw_value in list(collection_ids_filter or []) + [collection_id_filter]:
        value = str(raw_value or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _boost_title_matches(chunks: Sequence[ScoredChunk], title_matched_doc_ids: Sequence[str]) -> List[ScoredChunk]:
    if not title_matched_doc_ids:
        return list(chunks)

    boosts = {
        doc_id: max(0.05, 0.18 - (index * 0.04))
        for index, doc_id in enumerate(title_matched_doc_ids)
    }
    boosted: List[ScoredChunk] = []
    for chunk in chunks:
        doc_id = str((chunk.doc.metadata or {}).get("doc_id") or "")
        boost = boosts.get(doc_id, 0.0)
        if boost <= 0.0:
            boosted.append(chunk)
            continue
        boosted.append(ScoredChunk(doc=chunk.doc, score=chunk.score + boost, method=chunk.method))
    return boosted


def retrieve_candidates(
    stores: KnowledgeStores,
    query: str,
    *,
    tenant_id: str,
    preferred_doc_ids: List[str],
    must_include_uploads: bool,
    top_k_vector: int,
    top_k_keyword: int,
    doc_id_filter: Optional[str] = None,
    collection_id_filter: Optional[str] = None,
    collection_ids_filter: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    effective_filter = doc_id_filter
    title_matched_doc_ids: List[str] = []
    if not effective_filter:
        title_matched_doc_ids = _title_matched_doc_ids(
            stores,
            query,
            tenant_id=tenant_id,
            preferred_doc_ids=preferred_doc_ids,
            collection_id_filter=collection_id_filter,
            collection_ids_filter=collection_ids_filter,
        )

    vector_hits = vector_search(
        stores,
        query,
        top_k=top_k_vector,
        tenant_id=tenant_id,
        doc_id_filter=effective_filter,
        collection_id_filter=collection_id_filter if not effective_filter else None,
        collection_ids_filter=collection_ids_filter if not effective_filter else None,
    )
    if title_matched_doc_ids:
        for matched_doc_id in title_matched_doc_ids:
            vector_hits.extend(
                vector_search(
                    stores,
                    query,
                    top_k=max(1, min(3, top_k_vector)),
                    tenant_id=tenant_id,
                    doc_id_filter=matched_doc_id,
                )
            )
    keyword_hits = keyword_search(
        stores,
        query,
        top_k=top_k_keyword,
        tenant_id=tenant_id,
        doc_id_filter=effective_filter,
        collection_id_filter=collection_id_filter if not effective_filter else None,
        collection_ids_filter=collection_ids_filter if not effective_filter else None,
    )
    if title_matched_doc_ids:
        for matched_doc_id in title_matched_doc_ids:
            keyword_hits.extend(
                keyword_search(
                    stores,
                    query,
                    top_k=max(1, min(2, top_k_keyword)),
                    tenant_id=tenant_id,
                    doc_id_filter=matched_doc_id,
                )
            )
    if not effective_filter and preferred_doc_ids:
        vector_hits = [chunk for chunk in vector_hits if (chunk.doc.metadata or {}).get("doc_id") in preferred_doc_ids]
        keyword_hits = [chunk for chunk in keyword_hits if (chunk.doc.metadata or {}).get("doc_id") in preferred_doc_ids]

    if must_include_uploads:
        boosted_vector: List[ScoredChunk] = []
        for chunk in vector_hits:
            if (chunk.doc.metadata or {}).get("source_type") == "upload":
                boosted_vector.append(ScoredChunk(doc=chunk.doc, score=chunk.score + 0.01, method=chunk.method))
            else:
                boosted_vector.append(chunk)
        vector_hits = boosted_vector

    vector_hits = _boost_title_matches(vector_hits, title_matched_doc_ids)
    keyword_hits = _boost_title_matches(keyword_hits, title_matched_doc_ids)

    merged = merge_dedupe(vector_hits + keyword_hits)
    return {"vector": vector_hits, "keyword": keyword_hits, "merged": merged}


def _heuristic_relevance(question: str, text: str) -> int:
    q_terms = set(re.findall(r"[A-Za-z0-9_]{3,}", question.lower()))
    t_terms = set(re.findall(r"[A-Za-z0-9_]{3,}", text.lower()))
    overlap = len(q_terms & t_terms)
    if overlap >= 10:
        return 3
    if overlap >= 5:
        return 2
    if overlap >= 2:
        return 1
    return 0


def _title_hint_relevance(question: str, metadata: Dict[str, Any]) -> int:
    title = str(metadata.get("title") or "")
    if not title:
        return 0
    q_terms = set(re.findall(r"[A-Za-z0-9_]{3,}", question.lower()))
    t_terms = set(re.findall(r"[A-Za-z0-9_]{3,}", title.lower()))
    overlap = len(q_terms & t_terms)
    if overlap >= 2:
        return 2
    if overlap >= 1 and q_terms & {"architecture", "contract", "policy", "requirement", "runbook", "playbook", "agreement"}:
        return 2
    return 0


def _normalize_for_match(value: str) -> str:
    parts = re.findall(r"[A-Za-z0-9_]+", value.lower())
    normalized: list[str] = []
    for part in parts:
        normalized.extend(piece for piece in part.replace("_", " ").split() if piece)
    return " ".join(normalized)


def _question_echo_penalty(question: str, chunk: Document) -> int:
    metadata = chunk.metadata or {}
    title = _normalize_for_match(str(metadata.get("title") or ""))
    text = _normalize_for_match(chunk.page_content)
    normalized_question = _normalize_for_match(question)
    if len(normalized_question) < 24:
        return 0

    meta_title_terms = (
        "test queries",
        "prompt",
        "prompts",
        "example query",
        "example queries",
        "sample query",
        "sample queries",
    )
    if not any(term in title for term in meta_title_terms):
        return 0
    if normalized_question in text:
        return 2
    return 0


def _meta_catalog_penalty(question: str, metadata: Dict[str, Any]) -> int:
    title = _normalize_for_match(str(metadata.get("title") or ""))
    if not title:
        return 0

    meta_title_terms = (
        "test queries",
        "prompt",
        "prompts",
        "example",
        "examples",
        "sample query",
        "sample queries",
    )
    if not any(term in title for term in meta_title_terms):
        return 0

    q_terms = set(re.findall(r"[A-Za-z0-9_]{3,}", question.lower()))
    t_terms = set(re.findall(r"[A-Za-z0-9_]{3,}", title.lower()))
    overlap = len(q_terms & t_terms)
    if overlap == 0:
        return 2
    if overlap == 1:
        return 1
    return 0


def _operational_runbook_penalty(question: str, metadata: Dict[str, Any]) -> int:
    title = _normalize_for_match(str(metadata.get("title") or ""))
    if not title:
        return 0

    smoke_title_terms = (
        "local docker stack",
        "quick start",
        "smoke test",
        "acceptance test",
    )
    if not any(term in title for term in smoke_title_terms):
        return 0

    q_terms = set(re.findall(r"[A-Za-z0-9_]{3,}", question.lower()))
    if "architecture" not in q_terms:
        return 0
    if q_terms & {"docker", "compose", "stack", "startup", "deploy", "local", "runbook"}:
        return 0
    return 2


def grade_chunks(
    judge_llm: Any,
    *,
    settings: Any,
    question: str,
    chunks: Sequence[Document],
    max_chunks: int = 12,
    callbacks=None,
) -> List[GradedChunk]:
    selected = list(chunks)[:max_chunks]
    items = []
    for chunk in selected:
        metadata = chunk.metadata or {}
        chunk_id = metadata.get("chunk_id") or f"{metadata.get('doc_id')}#chunk{metadata.get('chunk_index')}"
        title = metadata.get("title", "")
        location = "page " + str(metadata.get("page")) if "page" in metadata else f"chunk {metadata.get('chunk_index')}"
        snippet = chunk.page_content[:800] + ("..." if len(chunk.page_content) > 800 else "")
        items.append({"chunk_id": str(chunk_id), "title": str(title), "location": str(location), "text": snippet})

    prompt = render_template(
        load_judge_grading_prompt(settings),
        {"QUESTION": question, "CHUNKS_JSON": items},
    )
    callbacks = callbacks or []
    try:
        response = judge_llm.invoke(prompt, config={"callbacks": callbacks})
        text = getattr(response, "content", None) or str(response)
        obj = extract_json(text)
        if obj and isinstance(obj.get("grades"), list):
            grade_map: Dict[str, Tuple[int, str]] = {}
            for grade in obj["grades"]:
                if not isinstance(grade, dict):
                    continue
                chunk_id = str(grade.get("chunk_id", ""))
                relevance = coerce_int(grade.get("relevance"), default=0)
                relevance = max(0, min(3, relevance))
                reason = str(grade.get("reason", ""))[:200]
                if chunk_id:
                    grade_map[chunk_id] = (relevance, reason)
            graded: List[GradedChunk] = []
            for chunk in selected:
                chunk_id = str((chunk.metadata or {}).get("chunk_id") or "")
                relevance, reason = grade_map.get(
                    chunk_id,
                    (_heuristic_relevance(question, chunk.page_content), "heuristic"),
                )
                title_relevance = _title_hint_relevance(question, chunk.metadata or {})
                if title_relevance > relevance:
                    relevance = title_relevance
                    reason = "title_hint"
                echo_penalty = _question_echo_penalty(question, chunk)
                meta_penalty = _meta_catalog_penalty(question, chunk.metadata or {})
                runbook_penalty = _operational_runbook_penalty(question, chunk.metadata or {})
                if echo_penalty or meta_penalty or runbook_penalty:
                    if echo_penalty >= meta_penalty and echo_penalty >= runbook_penalty:
                        relevance = max(0, relevance - echo_penalty)
                        reason = "question_echo"
                    elif meta_penalty >= runbook_penalty:
                        relevance = max(0, relevance - meta_penalty)
                        reason = "meta_catalog"
                    else:
                        relevance = max(0, relevance - runbook_penalty)
                        reason = "operational_runbook"
                graded.append(GradedChunk(doc=chunk, relevance=relevance, reason=reason))
            return graded
    except Exception:
        pass

    graded: List[GradedChunk] = []
    for chunk in selected:
        relevance = max(
            _heuristic_relevance(question, chunk.page_content),
            _title_hint_relevance(question, chunk.metadata or {}),
        )
        echo_penalty = _question_echo_penalty(question, chunk)
        meta_penalty = _meta_catalog_penalty(question, chunk.metadata or {})
        runbook_penalty = _operational_runbook_penalty(question, chunk.metadata or {})
        if echo_penalty or meta_penalty or runbook_penalty:
            if echo_penalty >= meta_penalty and echo_penalty >= runbook_penalty:
                relevance = max(0, relevance - echo_penalty)
                reason = "question_echo"
            elif meta_penalty >= runbook_penalty:
                relevance = max(0, relevance - meta_penalty)
                reason = "meta_catalog"
            else:
                relevance = max(0, relevance - runbook_penalty)
                reason = "operational_runbook"
        else:
            reason = "heuristic"
        graded.append(GradedChunk(doc=chunk, relevance=relevance, reason=reason))
    return graded
