from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    "rank_fuse_dedupe",
    "retrieve_candidates",
    "vector_search",
    "grade_chunks",
]

_GRADE_CACHE_MAX = 256
_GRADE_CACHE: "OrderedDict[tuple[str, tuple[tuple[str, str, int, str], ...]], List[GradedChunk]]" = OrderedDict()


@contextmanager
def _optional_stage(stage_timer: Any, name: str):
    if stage_timer is not None and hasattr(stage_timer, "measure"):
        with stage_timer.measure(name):
            yield
        return
    yield


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


def rank_fuse_dedupe(
    lanes: Mapping[str, Sequence[ScoredChunk]],
    *,
    rrf_k: int = 60,
) -> List[ScoredChunk]:
    fused_scores: Dict[str, float] = {}
    lane_details: Dict[str, Dict[str, Dict[str, float]]] = {}
    best_by_key: Dict[str, ScoredChunk] = {}
    best_source_score: Dict[str, float] = {}

    for lane_name, chunks in lanes.items():
        clean_lane = str(lane_name or "retrieval").strip() or "retrieval"
        for rank, chunk in enumerate(chunks, start=1):
            key = _doc_key(chunk.doc)
            fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (max(1, int(rrf_k)) + rank))
            lane_details.setdefault(key, {})[clean_lane] = {
                "rank": float(rank),
                "score": float(chunk.score),
            }
            if key not in best_by_key or float(chunk.score) > best_source_score.get(key, float("-inf")):
                best_by_key[key] = chunk
                best_source_score[key] = float(chunk.score)

    fused: List[ScoredChunk] = []
    for key, chunk in best_by_key.items():
        metadata = dict(chunk.doc.metadata or {})
        metadata["_retrieval_lanes"] = lane_details.get(key, {})
        metadata["_rrf_score"] = fused_scores.get(key, 0.0)
        doc = Document(page_content=chunk.doc.page_content, metadata=metadata)
        fused.append(
            ScoredChunk(
                doc=doc,
                score=float(fused_scores.get(key, 0.0)),
                method="+".join(sorted(lane_details.get(key, {}) or {chunk.method: {}})),
            )
        )
    return sorted(fused, key=lambda chunk: chunk.score, reverse=True)


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
    stage_timer: Any | None = None,
) -> Dict[str, Any]:
    effective_filter = doc_id_filter
    title_matched_doc_ids: List[str] = []
    if not effective_filter:
        with _optional_stage(stage_timer, "source_planning"):
            title_matched_doc_ids = _title_matched_doc_ids(
                stores,
                query,
                tenant_id=tenant_id,
                preferred_doc_ids=preferred_doc_ids,
                collection_id_filter=collection_id_filter,
                collection_ids_filter=collection_ids_filter,
            )

    with _optional_stage(stage_timer, "vector_search"):
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
    with _optional_stage(stage_timer, "bm25_search"):
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

    merged = rank_fuse_dedupe({"vector": vector_hits, "keyword": keyword_hits})
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


def _apply_relevance_adjustments(question: str, chunk: Document, relevance: int, reason: str) -> tuple[int, str]:
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
    return max(0, min(3, int(relevance))), reason


def _heuristic_grade_selected(question: str, selected: Sequence[Document], *, reason: str = "heuristic_pregrade") -> List[GradedChunk]:
    graded: List[GradedChunk] = []
    for chunk in selected:
        relevance = max(
            _heuristic_relevance(question, chunk.page_content),
            _title_hint_relevance(question, chunk.metadata or {}),
        )
        adjusted_relevance, adjusted_reason = _apply_relevance_adjustments(question, chunk, relevance, reason)
        if adjusted_reason == reason and adjusted_relevance == 0:
            adjusted_reason = "heuristic"
        graded.append(GradedChunk(doc=chunk, relevance=adjusted_relevance, reason=adjusted_reason))
    return graded


def _chunk_cache_signature(chunk: Document) -> tuple[str, str, int, str]:
    metadata = chunk.metadata or {}
    chunk_id = str(metadata.get("chunk_id") or f"{metadata.get('doc_id')}#chunk{metadata.get('chunk_index')}")
    version = str(
        metadata.get("chunk_version")
        or metadata.get("content_hash")
        or metadata.get("updated_at")
        or metadata.get("ingested_at")
        or ""
    )
    text = str(chunk.page_content or "")
    text_fingerprint = f"{text[:64]}|{text[-64:]}" if text else ""
    return chunk_id, version, len(text), text_fingerprint


def _grade_cache_key(question: str, selected: Sequence[Document]) -> tuple[str, tuple[tuple[str, str, int, str], ...]]:
    normalized_question = _normalize_for_match(question)
    return normalized_question, tuple(_chunk_cache_signature(chunk) for chunk in selected)


def _get_cached_grades(key: tuple[str, tuple[tuple[str, str, int, str], ...]]) -> List[GradedChunk] | None:
    cached = _GRADE_CACHE.get(key)
    if cached is None:
        return None
    _GRADE_CACHE.move_to_end(key)
    return list(cached)


def _put_cached_grades(key: tuple[str, tuple[tuple[str, str, int, str], ...]], graded: Sequence[GradedChunk]) -> None:
    _GRADE_CACHE[key] = list(graded)
    _GRADE_CACHE.move_to_end(key)
    while len(_GRADE_CACHE) > _GRADE_CACHE_MAX:
        _GRADE_CACHE.popitem(last=False)


def _heuristic_grades_are_decisive(settings: Any, selected: Sequence[Document], graded: Sequence[GradedChunk]) -> bool:
    if not bool(getattr(settings, "rag_heuristic_grading_enabled", True)):
        return False
    if not selected:
        return True
    strong = sum(1 for item in graded if item.relevance >= 2)
    if strong <= 0:
        return False
    min_evidence = max(1, int(getattr(settings, "rag_min_evidence_chunks", 2) or 2))
    if len(selected) <= 6:
        return strong >= min_evidence
    return strong >= max(3, min_evidence) and strong >= max(1, len(selected) // 5)


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
    cache_key = _grade_cache_key(question, selected)
    cached = _get_cached_grades(cache_key)
    if cached is not None:
        return cached

    pregraded = _heuristic_grade_selected(question, selected)
    if judge_llm is None or _heuristic_grades_are_decisive(settings, selected, pregraded):
        _put_cached_grades(cache_key, pregraded)
        return list(pregraded)

    judge_window = min(
        len(selected),
        max(1, int(getattr(settings, "rag_judge_grade_max_chunks", max_chunks) or max_chunks)),
    )
    judge_selected = selected[:judge_window]
    items = []
    for chunk in judge_selected:
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
            pregrade_by_id = {
                str((item.doc.metadata or {}).get("chunk_id") or f"{(item.doc.metadata or {}).get('doc_id')}#chunk{(item.doc.metadata or {}).get('chunk_index')}"): item
                for item in pregraded
            }
            for chunk in selected:
                metadata = chunk.metadata or {}
                chunk_id = str(metadata.get("chunk_id") or f"{metadata.get('doc_id')}#chunk{metadata.get('chunk_index')}")
                fallback = pregrade_by_id.get(chunk_id)
                relevance, reason = grade_map.get(chunk_id, (fallback.relevance, fallback.reason) if fallback else (0, "heuristic"))
                relevance, reason = _apply_relevance_adjustments(question, chunk, relevance, reason)
                graded.append(GradedChunk(doc=chunk, relevance=relevance, reason=reason))
            _put_cached_grades(cache_key, graded)
            return graded
    except Exception:
        pass

    fallback = _heuristic_grade_selected(question, selected, reason="heuristic_timeout_or_unavailable")
    _put_cached_grades(cache_key, fallback)
    return fallback
