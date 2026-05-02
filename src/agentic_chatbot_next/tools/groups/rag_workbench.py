from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Iterable, List, Sequence

from langchain.tools import tool

from agentic_chatbot_next.rag.adaptive import CorpusRetrievalAdapter, SearchFilters
from agentic_chatbot_next.utils.json_utils import extract_json


def _parse_csv(raw: str) -> List[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _cap_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, min(parsed, maximum))


def _normalize_text(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _query_terms(query: str) -> set[str]:
    stopwords = {
        "about",
        "after",
        "also",
        "answer",
        "because",
        "before",
        "between",
        "cite",
        "cited",
        "could",
        "does",
        "from",
        "have",
        "into",
        "only",
        "search",
        "should",
        "source",
        "that",
        "their",
        "there",
        "these",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
    }
    return {term for term in _normalize_text(query).split() if len(term) >= 3 and term not in stopwords}


def _extract_entities(query: str, *, limit: int = 8) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,4}\b", str(query or "")):
        value = match.group(0).strip(" .,;:")
        if len(value) >= 3 and value.lower() not in {"what", "which", "when", "where", "why", "how"}:
            candidates.append(value)
    for match in re.finditer(r"\b[A-Z]{2,}[-_A-Z0-9]*\d+[A-Z0-9-]*\b|\b[A-Za-z]+-\d+[A-Za-z0-9-]*\b", str(query or "")):
        candidates.append(match.group(0))
    return _dedupe(candidates)[:limit]


def _dedupe(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = str(value or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result


def _parse_json_payload(raw: Any, *, expected: str = "object_or_array") -> tuple[Any, list[str]]:
    if isinstance(raw, (dict, list)):
        return raw, []
    text = str(raw or "").strip()
    if not text:
        return [] if expected == "array" else {}, ["empty_json_input"]
    try:
        payload = json.loads(text)
        if isinstance(payload, (dict, list)):
            return payload, []
        return {}, ["json_input_not_object_or_array"]
    except Exception:
        parsed = extract_json(text)
        if parsed is not None:
            return parsed, []
    return [] if expected == "array" else {}, ["invalid_json_input"]


def _serialize_document(doc: Any) -> dict[str, Any]:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    return {
        "chunk_id": str(metadata.get("chunk_id") or ""),
        "doc_id": str(metadata.get("doc_id") or ""),
        "title": str(metadata.get("title") or ""),
        "source_type": str(metadata.get("source_type") or ""),
        "source_path": str(metadata.get("source_path") or ""),
        "collection_id": str(metadata.get("collection_id") or ""),
        "chunk_index": int(metadata.get("chunk_index") or 0),
        "chunk_type": str(metadata.get("chunk_type") or ""),
        "section_title": str(metadata.get("section_title") or ""),
        "clause_number": str(metadata.get("clause_number") or ""),
        "page_number": metadata.get("page"),
        "sheet_name": str(metadata.get("sheet_name") or ""),
        "cell_range": str(metadata.get("cell_range") or ""),
        "citation_id": str(metadata.get("chunk_id") or ""),
        "content": str(getattr(doc, "page_content", "") or ""),
    }


def _serialize_scored_chunk(item: Any) -> dict[str, Any]:
    return {
        **_serialize_document(getattr(item, "doc", None)),
        "score": round(float(getattr(item, "score", 0.0) or 0.0), 4),
        "method": str(getattr(item, "method", "") or ""),
    }


def _serialize_document_record(record: Any) -> dict[str, Any]:
    return {
        "doc_id": str(getattr(record, "doc_id", "") or (record.get("doc_id") if isinstance(record, dict) else "") or ""),
        "title": str(getattr(record, "title", "") or (record.get("title") if isinstance(record, dict) else "") or ""),
        "source_type": str(getattr(record, "source_type", "") or (record.get("source_type") if isinstance(record, dict) else "") or ""),
        "source_path": str(getattr(record, "source_path", "") or (record.get("source_path") if isinstance(record, dict) else "") or ""),
        "file_type": str(getattr(record, "file_type", "") or (record.get("file_type") if isinstance(record, dict) else "") or ""),
        "doc_structure_type": str(
            getattr(record, "doc_structure_type", "")
            or (record.get("doc_structure_type") if isinstance(record, dict) else "")
            or ""
        ),
        "collection_id": str(getattr(record, "collection_id", "") or (record.get("collection_id") if isinstance(record, dict) else "") or ""),
        "num_chunks": int(getattr(record, "num_chunks", 0) or (record.get("num_chunks") if isinstance(record, dict) else 0) or 0),
    }


def _candidate_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = []
        for key in ("results", "candidates", "chunks", "grades", "kept", "selected_candidates", "evidence", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                items = value
                break
    else:
        items = []
    result: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            result.append(dict(item))
    return result


def _candidate_id(candidate: dict[str, Any], *, fallback_index: int = 0) -> str:
    for key in ("chunk_id", "citation_id", "id"):
        value = str(candidate.get(key) or "").strip()
        if value:
            return value
    doc_id = str(candidate.get("doc_id") or "").strip()
    chunk_index = str(candidate.get("chunk_index") or "").strip()
    if doc_id and chunk_index:
        return f"{doc_id}#chunk{chunk_index}"
    if doc_id:
        return doc_id
    return f"candidate-{fallback_index}"


def _candidate_text(candidate: dict[str, Any]) -> str:
    return " ".join(
        str(candidate.get(key) or "")
        for key in (
            "title",
            "section_title",
            "clause_number",
            "sheet_name",
            "cell_range",
            "content",
            "snippet",
            "text",
            "reason",
            "missing_evidence",
        )
    )


def _candidate_score(candidate: dict[str, Any]) -> float:
    grade = str(candidate.get("grade") or candidate.get("label") or "").strip().lower()
    grade_weight = {"strong": 3.0, "partial": 2.0, "weak": 1.0, "off_topic": 0.0}.get(grade, 0.0)
    try:
        score = float(candidate.get("score") or 0.0)
    except Exception:
        score = 0.0
    return grade_weight + max(0.0, min(score, 1.0))


def _overlap_score(query: str, text: str) -> float:
    terms = _query_terms(query)
    if not terms:
        return 0.0
    haystack = set(_normalize_text(text).split())
    return len(terms & haystack) / max(1, len(terms))


def _heuristic_query_plan(query: str, *, collection_id: str, preferred_doc_ids: Sequence[str], max_queries: int) -> list[dict[str, Any]]:
    clean = str(query or "").strip()
    entities = _extract_entities(clean)
    quoted = re.findall(r'"([^"]+)"|`([^`]+)`', clean)
    exact_terms = [left or right for left, right in quoted if (left or right)]
    identifiers = re.findall(r"\b[A-Z]{2,}[-_A-Z0-9]*\d+[A-Z0-9-]*\b|\b[A-Za-z]+-\d+[A-Za-z0-9-]*\b", clean)
    date_terms = re.findall(r"\b(?:\d{1,2}\s+[A-Z][a-z]{2,8}\s+\d{4}|\d{4}-\d{2}-\d{2}|current|latest|approved|final|draft|revision|rev)\b", clean)
    facets = [
        {
            "facet": "semantic",
            "query": clean,
            "strategy": "hybrid",
            "rationale": "Broad semantic pass for the user question.",
        },
        {
            "facet": "exact_terms",
            "query": " ".join(_dedupe([*exact_terms, *identifiers, *entities])) or clean,
            "strategy": "keyword",
            "rationale": "Exact names, dates, identifiers, or quoted terms.",
        },
        {
            "facet": "entity",
            "query": " ".join(entities) or clean,
            "strategy": "hybrid",
            "rationale": "Entity-focused search for document leads and disambiguation.",
        },
        {
            "facet": "date_version",
            "query": " ".join(_dedupe([*entities[:4], *date_terms, "current approved final draft"])) or clean,
            "strategy": "keyword",
            "rationale": "Authority, date, current/final/draft, and version checks.",
        },
        {
            "facet": "source_discovery",
            "query": " ".join(_dedupe([*entities[:4], *list(_query_terms(clean))[:6]])) or clean,
            "strategy": "hybrid",
            "rationale": "Find candidate documents before scoped reads.",
        },
        {
            "facet": "contradiction",
            "query": f"{clean} conflicting contrary outdated superseded exception",
            "strategy": "hybrid",
            "rationale": "Look for conflicting, superseded, or exception evidence.",
        },
    ]
    for facet in facets:
        facet["collection_id"] = str(collection_id or "").strip()
        facet["preferred_doc_ids"] = list(preferred_doc_ids)
    return facets[: max(1, int(max_queries))]


def _invoke_judge_json(judge_llm: Any, prompt: str, callbacks: Sequence[Any] | None = None) -> dict[str, Any]:
    if judge_llm is None:
        return {}
    try:
        response = judge_llm.invoke(prompt, config={"callbacks": list(callbacks or [])})
    except TypeError:
        try:
            response = judge_llm.invoke(prompt)
        except Exception:
            return {}
    except Exception:
        return {}
    text = getattr(response, "content", None) or str(response)
    return extract_json(text) or {}


def _heuristic_grade_candidate(query: str, candidate: dict[str, Any]) -> dict[str, Any]:
    text = _candidate_text(candidate)
    overlap = _overlap_score(query, text)
    normalized_query = _normalize_text(query)
    normalized_text = _normalize_text(text)
    if normalized_query and normalized_query in normalized_text:
        overlap = max(overlap, 0.85)
    if overlap >= 0.45:
        grade = "strong"
        reason = "Candidate contains several query terms and is likely directly relevant."
        missing = ""
    elif overlap >= 0.20:
        grade = "partial"
        reason = "Candidate overlaps with the query but may need neighboring or corroborating evidence."
        missing = "Confirm the exact claim with a scoped read or final RAG synthesis."
    elif overlap > 0.0:
        grade = "weak"
        reason = "Candidate has limited term overlap with the query."
        missing = "Find stronger evidence or try a rewrite."
    else:
        grade = "off_topic"
        reason = "Candidate does not visibly overlap with the query."
        missing = "Use a different query or source scope."
    return {
        **candidate,
        "grade": grade,
        "reason": reason,
        "missing_evidence": missing,
        "overlap_score": round(overlap, 3),
    }


def _is_conflict_candidate(candidate: dict[str, Any]) -> bool:
    if bool(candidate.get("conflict") or candidate.get("contradiction")):
        return True
    text = _normalize_text(_candidate_text(candidate))
    return any(term in text for term in ("conflict", "contradict", "contrary", "superseded", "obsolete", "draft", "current", "approved"))


def build_rag_workbench_tools(ctx: Any) -> List[Any]:
    session = ctx.session_handle
    adapter = CorpusRetrievalAdapter(ctx.stores, settings=ctx.settings, session=session)
    judge_llm = getattr(getattr(ctx, "providers", None), "judge", None)
    callbacks = list(getattr(ctx, "callbacks", []) or [])
    tenant_id = str(getattr(session, "tenant_id", getattr(ctx.settings, "default_tenant_id", "local-dev")) or "local-dev")

    @tool
    def plan_rag_queries(
        query: str,
        collection_id: str = "",
        preferred_doc_ids_csv: str = "",
        max_queries: int = 6,
    ) -> dict[str, Any]:
        """Plan focused query facets for autonomous RAG evidence exploration."""

        preferred_doc_ids = _parse_csv(preferred_doc_ids_csv)
        capped = _cap_int(max_queries, default=6, minimum=1, maximum=8)
        fallback_queries = _heuristic_query_plan(
            query,
            collection_id=str(collection_id or "").strip(),
            preferred_doc_ids=preferred_doc_ids,
            max_queries=capped,
        )
        prompt = (
            "You are planning retrieval queries for a RAG specialist. Do not answer the question. "
            "Return JSON only with key queries, a list of objects with keys facet, query, strategy, rationale. "
            "Use these facets when useful: semantic, exact_terms, entity, date_version, source_discovery, contradiction. "
            "Strategies must be hybrid, keyword, or vector. Keep queries short and grounded in visible wording.\n"
            f"QUESTION: {query}\n"
            f"COLLECTION_ID: {collection_id}\n"
            f"PREFERRED_DOC_IDS: {preferred_doc_ids}\n"
            f"MAX_QUERIES: {capped}\n"
        )
        payload = _invoke_judge_json(judge_llm, prompt, callbacks)
        planned: list[dict[str, Any]] = []
        if isinstance(payload.get("queries"), list):
            for item in payload["queries"]:
                if not isinstance(item, dict):
                    continue
                strategy = str(item.get("strategy") or "hybrid").strip().lower()
                if strategy not in {"hybrid", "keyword", "vector"}:
                    strategy = "hybrid"
                planned.append(
                    {
                        "facet": str(item.get("facet") or "semantic").strip() or "semantic",
                        "query": str(item.get("query") or query).strip() or str(query or ""),
                        "strategy": strategy,
                        "rationale": str(item.get("rationale") or "").strip(),
                        "collection_id": str(collection_id or "").strip(),
                        "preferred_doc_ids": preferred_doc_ids,
                    }
                )
                if len(planned) >= capped:
                    break
        warnings = [] if planned else ["judge_query_planner_unavailable_or_invalid; used heuristic plan"]
        return {
            "query": str(query or ""),
            "collection_id": str(collection_id or "").strip(),
            "preferred_doc_ids": preferred_doc_ids,
            "queries": planned or fallback_queries,
            "warnings": warnings,
        }

    @tool
    def search_corpus_chunks(
        query: str,
        strategy: str = "hybrid",
        preferred_doc_ids_csv: str = "",
        collection_id: str = "",
        limit: int = 12,
    ) -> dict[str, Any]:
        """Search corpus chunks with vector, keyword, or hybrid retrieval for evidence exploration."""

        normalized_strategy = str(strategy or "hybrid").strip().lower()
        if normalized_strategy not in {"hybrid", "vector", "keyword"}:
            return {
                "error": "strategy must be one of: hybrid, vector, keyword",
                "results": [],
            }
        preferred_doc_ids = _parse_csv(preferred_doc_ids_csv)
        capped_limit = max(1, min(int(limit or 12), 50))
        hits = adapter.search_corpus(
            query,
            filters=SearchFilters(doc_ids=preferred_doc_ids, collection_id=str(collection_id or "").strip()),
            strategy=normalized_strategy,
            limit=capped_limit,
            preferred_doc_ids=preferred_doc_ids,
        )
        return {
            "query": str(query or ""),
            "strategy": normalized_strategy,
            "collection_id": str(collection_id or "").strip(),
            "preferred_doc_ids": preferred_doc_ids,
            "results": [_serialize_scored_chunk(item) for item in hits[:capped_limit]],
        }

    @tool
    def grep_corpus_chunks(
        query: str,
        preferred_doc_ids_csv: str = "",
        collection_id: str = "",
        limit: int = 12,
    ) -> dict[str, Any]:
        """Run keyword-first corpus retrieval for exact terms, identifiers, clauses, and names."""

        preferred_doc_ids = _parse_csv(preferred_doc_ids_csv)
        capped_limit = max(1, min(int(limit or 12), 50))
        hits = adapter.grep_corpus(
            query,
            filters=SearchFilters(doc_ids=preferred_doc_ids, collection_id=str(collection_id or "").strip()),
            limit=capped_limit,
        )
        return {
            "query": str(query or ""),
            "collection_id": str(collection_id or "").strip(),
            "preferred_doc_ids": preferred_doc_ids,
            "results": [_serialize_scored_chunk(item) for item in hits[:capped_limit]],
        }

    @tool
    def fetch_chunk_window(chunk_id: str, before: int = 1, after: int = 1) -> dict[str, Any]:
        """Fetch neighboring chunks around one candidate chunk id for local context inspection."""

        before_count = max(0, min(int(before or 0), 5))
        after_count = max(0, min(int(after or 0), 5))
        docs = adapter.fetch_chunk_window(chunk_id, before=before_count, after=after_count)
        return {
            "chunk_id": str(chunk_id or ""),
            "before": before_count,
            "after": after_count,
            "chunks": [_serialize_document(doc) for doc in docs],
        }

    @tool
    def inspect_document_structure(doc_id: str, max_items: int = 80) -> dict[str, Any]:
        """Inspect section, clause, sheet, or table structure for one indexed document."""

        capped = _cap_int(max_items, default=80, minimum=1, maximum=200)
        chunk_store = getattr(ctx.stores, "chunk_store", None)
        outline: list[dict[str, Any]] = []
        warnings: list[str] = []
        if chunk_store is None or not hasattr(chunk_store, "get_structure_outline"):
            warnings.append("chunk_store_structure_outline_unavailable")
        else:
            try:
                outline = [dict(item) for item in chunk_store.get_structure_outline(str(doc_id or ""), tenant_id=tenant_id)]
            except TypeError:
                try:
                    outline = [dict(item) for item in chunk_store.get_structure_outline(str(doc_id or ""), tenant_id)]
                except Exception:
                    outline = []
                    warnings.append("structure_outline_failed")
            except Exception:
                warnings.append("structure_outline_failed")
        samples: list[dict[str, Any]] = []
        if not outline and hasattr(adapter, "outline_scan"):
            samples = [_serialize_document(doc) for doc in adapter.outline_scan(str(doc_id or ""), max_chunks=min(capped, 12))]
        return {
            "doc_id": str(doc_id or ""),
            "outline": outline[:capped],
            "outline_count": len(outline),
            "truncated": len(outline) > capped,
            "sample_chunks": samples,
            "warnings": warnings,
        }

    @tool
    def search_document_sections(
        query: str,
        doc_ids_csv: str,
        prioritized_sections_json: str = "",
        limit: int = 12,
    ) -> dict[str, Any]:
        """Search section, clause, sheet, or table scopes inside selected documents."""

        doc_ids = _parse_csv(doc_ids_csv)
        capped = _cap_int(limit, default=12, minimum=1, maximum=50)
        prioritized_payload, warnings = _parse_json_payload(prioritized_sections_json, expected="array") if prioritized_sections_json else ([], [])
        prioritized_sections = _candidate_list(prioritized_payload) if isinstance(prioritized_payload, dict) else (
            [item for item in prioritized_payload if isinstance(item, dict)] if isinstance(prioritized_payload, list) else []
        )
        docs = adapter.search_section_scope(
            str(query or ""),
            doc_ids=doc_ids,
            prioritized_sections=prioritized_sections,
            limit=capped,
        )
        return {
            "query": str(query or ""),
            "doc_ids": doc_ids,
            "prioritized_sections": prioritized_sections,
            "results": [_serialize_document(doc) for doc in docs[:capped]],
            "warnings": warnings,
        }

    @tool
    def filter_indexed_docs(
        collection_id: str = "",
        source_type: str = "",
        file_type: str = "",
        title_contains: str = "",
        doc_structure_type: str = "",
        limit: int = 100,
    ) -> dict[str, Any]:
        """Filter indexed document metadata before expensive retrieval."""

        capped = _cap_int(limit, default=100, minimum=1, maximum=200)
        doc_store = getattr(ctx.stores, "doc_store", None)
        if doc_store is None:
            return {"documents": [], "count": 0, "warnings": ["doc_store_unavailable"]}
        warnings: list[str] = []
        try:
            if hasattr(doc_store, "search_by_metadata"):
                records = doc_store.search_by_metadata(
                    tenant_id=tenant_id,
                    collection_id=str(collection_id or "").strip(),
                    source_type=str(source_type or "").strip(),
                    file_type=str(file_type or "").strip(),
                    doc_structure_type=str(doc_structure_type or "").strip(),
                    title_contains=str(title_contains or "").strip(),
                    limit=capped,
                )
            else:
                records = doc_store.list_documents(
                    source_type=str(source_type or "").strip(),
                    tenant_id=tenant_id,
                    collection_id=str(collection_id or "").strip(),
                )
                title_filter = str(title_contains or "").strip().lower()
                file_filter = str(file_type or "").strip().lower()
                structure_filter = str(doc_structure_type or "").strip().lower()
                records = [
                    record
                    for record in records
                    if (not title_filter or title_filter in str(getattr(record, "title", "") or "").lower())
                    and (not file_filter or file_filter == str(getattr(record, "file_type", "") or "").lower())
                    and (not structure_filter or structure_filter == str(getattr(record, "doc_structure_type", "") or "").lower())
                ][:capped]
        except TypeError:
            try:
                records = doc_store.list_documents(tenant_id=tenant_id, collection_id=str(collection_id or ""))
            except Exception:
                records = []
                warnings.append("document_filter_failed")
        except Exception:
            records = []
            warnings.append("document_filter_failed")
        documents = [_serialize_document_record(record) for record in list(records)[:capped]]
        return {
            "filters": {
                "collection_id": str(collection_id or "").strip(),
                "source_type": str(source_type or "").strip(),
                "file_type": str(file_type or "").strip(),
                "title_contains": str(title_contains or "").strip(),
                "doc_structure_type": str(doc_structure_type or "").strip(),
            },
            "documents": documents,
            "count": len(documents),
            "warnings": warnings,
        }

    @tool
    def grade_evidence_candidates(
        query: str,
        candidates_json: str,
        max_candidates: int = 20,
    ) -> dict[str, Any]:
        """Grade exploratory candidate chunks for relevance before final RAG synthesis."""

        payload, warnings = _parse_json_payload(candidates_json)
        candidates = _candidate_list(payload)[: _cap_int(max_candidates, default=20, minimum=1, maximum=50)]
        if not candidates:
            return {"query": str(query or ""), "grades": [], "warnings": [*warnings, "no_candidates_to_grade"]}
        compact = [
            {
                "candidate_id": _candidate_id(candidate, fallback_index=index),
                "doc_id": str(candidate.get("doc_id") or ""),
                "title": str(candidate.get("title") or ""),
                "section_title": str(candidate.get("section_title") or ""),
                "text": str(candidate.get("content") or candidate.get("snippet") or candidate.get("text") or "")[:900],
            }
            for index, candidate in enumerate(candidates)
        ]
        prompt = (
            "Grade RAG evidence candidates for the question. Return JSON only with key grades, a list of objects "
            "with keys candidate_id, grade, reason, missing_evidence. Grade must be strong, partial, weak, or off_topic. "
            "Do not answer the question.\n"
            f"QUESTION: {query}\n"
            f"CANDIDATES_JSON: {json.dumps(compact, ensure_ascii=False)}\n"
        )
        payload = _invoke_judge_json(judge_llm, prompt, callbacks)
        by_id: dict[str, dict[str, Any]] = {}
        if isinstance(payload.get("grades"), list):
            for grade in payload["grades"]:
                if not isinstance(grade, dict):
                    continue
                candidate_id = str(grade.get("candidate_id") or "").strip()
                label = str(grade.get("grade") or "").strip().lower()
                if label not in {"strong", "partial", "weak", "off_topic"}:
                    label = "weak"
                if candidate_id:
                    by_id[candidate_id] = {
                        "grade": label,
                        "reason": str(grade.get("reason") or "").strip(),
                        "missing_evidence": str(grade.get("missing_evidence") or "").strip(),
                    }
        if not by_id:
            warnings.append("judge_evidence_grader_unavailable_or_invalid; used heuristic grades")
        grades: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            candidate_id = _candidate_id(candidate, fallback_index=index)
            graded = {**candidate, "candidate_id": candidate_id}
            if candidate_id in by_id:
                graded.update(by_id[candidate_id])
                graded["overlap_score"] = round(_overlap_score(query, _candidate_text(candidate)), 3)
            else:
                graded = _heuristic_grade_candidate(query, graded)
                graded["candidate_id"] = candidate_id
            grades.append(graded)
        return {"query": str(query or ""), "grades": grades, "warnings": warnings}

    @tool
    def prune_evidence_candidates(
        query: str,
        candidates_json: str,
        keep: int = 12,
        max_per_doc: int = 3,
    ) -> dict[str, Any]:
        """Deduplicate and balance evidence candidates across documents."""

        payload, warnings = _parse_json_payload(candidates_json)
        candidates = _candidate_list(payload)
        keep_count = _cap_int(keep, default=12, minimum=1, maximum=50)
        per_doc_limit = _cap_int(max_per_doc, default=3, minimum=1, maximum=10)
        if not candidates:
            return {"kept": [], "pruned_candidate_ids": [], "warnings": [*warnings, "no_candidates_to_prune"]}
        by_id: dict[str, dict[str, Any]] = {}
        for index, candidate in enumerate(candidates):
            candidate_id = _candidate_id(candidate, fallback_index=index)
            normalized = {**candidate, "candidate_id": str(candidate.get("candidate_id") or candidate_id)}
            existing = by_id.get(candidate_id)
            if existing is None or (_candidate_score(normalized), _overlap_score(query, _candidate_text(normalized))) > (
                _candidate_score(existing),
                _overlap_score(query, _candidate_text(existing)),
            ):
                by_id[candidate_id] = normalized
        ordered = sorted(
            by_id.values(),
            key=lambda item: (
                _candidate_score(item),
                _overlap_score(query, _candidate_text(item)),
                1 if _is_conflict_candidate(item) else 0,
            ),
            reverse=True,
        )
        kept: list[dict[str, Any]] = []
        pruned_ids: list[str] = []
        per_doc_counts: dict[str, int] = defaultdict(int)
        for candidate in ordered:
            doc_id = str(candidate.get("doc_id") or "")
            if doc_id and per_doc_counts[doc_id] >= per_doc_limit and not _is_conflict_candidate(candidate):
                pruned_ids.append(str(candidate.get("candidate_id") or ""))
                continue
            if len(kept) < keep_count:
                kept.append(candidate)
                if doc_id:
                    per_doc_counts[doc_id] += 1
            else:
                pruned_ids.append(str(candidate.get("candidate_id") or ""))
        if len(kept) < keep_count:
            kept_ids = {str(item.get("candidate_id") or "") for item in kept}
            for candidate in ordered:
                candidate_id = str(candidate.get("candidate_id") or "")
                if candidate_id in kept_ids or not _is_conflict_candidate(candidate):
                    continue
                kept.append(candidate)
                kept_ids.add(candidate_id)
                if len(kept) >= keep_count:
                    break
        return {
            "query": str(query or ""),
            "kept": kept[:keep_count],
            "kept_doc_ids": _dedupe(item.get("doc_id") for item in kept[:keep_count]),
            "pruned_candidate_ids": [item for item in pruned_ids if item],
            "doc_counts": dict(per_doc_counts),
            "warnings": warnings,
        }

    @tool
    def validate_evidence_plan(
        query: str,
        selected_candidates_json: str,
        expected_scope: str = "auto",
    ) -> dict[str, Any]:
        """Check whether selected evidence appears sufficient before final RAG synthesis."""

        payload, warnings = _parse_json_payload(selected_candidates_json)
        candidates = _candidate_list(payload)
        issues: list[dict[str, Any]] = []
        suggestions: list[str] = []
        doc_ids = _dedupe(candidate.get("doc_id") for candidate in candidates)
        strong_count = sum(1 for candidate in candidates if str(candidate.get("grade") or "").lower() == "strong")
        partial_count = sum(1 for candidate in candidates if str(candidate.get("grade") or "").lower() == "partial")
        combined_text = " ".join(_candidate_text(candidate) for candidate in candidates)
        overlap = _overlap_score(query, combined_text)
        expected = str(expected_scope or "auto").strip().lower()
        authority_requested = bool(re.search(r"\b(current|latest|approved|final|draft|revision|rev)\b", str(query or ""), re.I))
        authority_seen = bool(re.search(r"\b(current|latest|approved|final|draft|revision|rev)\b", combined_text, re.I))
        exhaustive_requested = expected in {"corpus_wide", "exhaustive"} or bool(
            re.search(r"\b(all|every|no evidence|not found|absence|only|any documents)\b", str(query or ""), re.I)
        )
        if not candidates:
            issues.append({"severity": "high", "check": "empty_evidence", "detail": "No selected candidates were provided."})
            suggestions.append("Run broader corpus search and keyword follow-up before final RAG synthesis.")
        if candidates and overlap < 0.18:
            issues.append({"severity": "medium", "check": "low_query_overlap", "detail": "Selected evidence has weak visible overlap with the question."})
            suggestions.append("Try a query rewrite or exact-term grep for named entities, dates, and identifiers.")
        if authority_requested and not authority_seen:
            issues.append({"severity": "medium", "check": "authority_signal_missing", "detail": "The question asks for authority/version/date resolution but selected evidence lacks visible authority terms."})
            suggestions.append("Search for current/latest/approved/final/draft/version evidence before final synthesis.")
        if exhaustive_requested and len(doc_ids) < 2:
            issues.append({"severity": "medium", "check": "thin_scope_for_exhaustive_claim", "detail": "A corpus-wide or negative-evidence claim should usually inspect multiple documents."})
            suggestions.append("Use source discovery and negative-evidence-friendly controller hints.")
        if candidates and strong_count == 0 and partial_count < 2:
            issues.append({"severity": "medium", "check": "weak_relevance", "detail": "No strong evidence candidates were selected."})
            suggestions.append("Grade more candidates or use section-scoped reads around promising documents.")
        status = "sufficient" if not any(item["severity"] == "high" for item in issues) and (strong_count or partial_count >= 2 or overlap >= 0.35) else "needs_more_evidence"
        return {
            "query": str(query or ""),
            "expected_scope": expected,
            "status": status,
            "selected_doc_ids": doc_ids,
            "strong_count": strong_count,
            "partial_count": partial_count,
            "query_overlap": round(overlap, 3),
            "issues": issues,
            "suggestions": suggestions,
            "warnings": warnings,
        }

    @tool
    def build_rag_controller_hints(
        query: str,
        selected_doc_ids_csv: str = "",
        selected_chunks_json: str = "",
        research_profile: str = "",
        coverage_goal: str = "",
        result_mode: str = "",
    ) -> dict[str, Any]:
        """Build safe controller_hints_json for a final rag_agent_tool call."""

        selected_doc_ids = _parse_csv(selected_doc_ids_csv)
        payload, warnings = _parse_json_payload(selected_chunks_json) if selected_chunks_json else ({}, [])
        candidates = _candidate_list(payload)
        candidate_doc_ids = _dedupe(candidate.get("doc_id") for candidate in candidates)
        selected_chunk_ids = _dedupe(_candidate_id(candidate, fallback_index=index) for index, candidate in enumerate(candidates))
        doc_ids = _dedupe([*selected_doc_ids, *candidate_doc_ids])
        clean_profile = str(research_profile or "").strip()
        clean_coverage = str(coverage_goal or "").strip()
        clean_mode = str(result_mode or "").strip()
        hints: dict[str, Any] = {
            "preferred_doc_ids": doc_ids,
            "resolved_doc_ids": doc_ids,
            "researcher_selected_chunk_ids": selected_chunk_ids[:24],
            "researcher_evidence_plan": [
                {
                    "chunk_id": _candidate_id(candidate, fallback_index=index),
                    "doc_id": str(candidate.get("doc_id") or ""),
                    "title": str(candidate.get("title") or ""),
                    "grade": str(candidate.get("grade") or ""),
                    "reason": str(candidate.get("reason") or "")[:240],
                }
                for index, candidate in enumerate(candidates[:24])
            ],
        }
        if clean_profile:
            hints["research_profile"] = clean_profile
        if clean_coverage:
            hints["coverage_goal"] = clean_coverage
        if clean_mode:
            hints["result_mode"] = clean_mode
        if clean_coverage in {"corpus_wide", "exhaustive", "cross_document"} or len(doc_ids) > 1:
            hints["force_deep_search"] = True
            hints["prefer_parallel_docs"] = True
        if clean_mode == "inventory":
            hints["prefer_inventory_output"] = True
        if clean_mode == "comparison":
            hints["prefer_parallel_docs"] = True
        if clean_coverage == "exhaustive" or re.search(r"\b(no evidence|not found|absence|all|every|any documents)\b", str(query or ""), re.I):
            hints["prefer_negative_evidence_reporting"] = True
        if re.search(r"\b(current|latest|approved|final|draft|revision|rev)\b", str(query or ""), re.I):
            hints["authority_version_check"] = True
        return {
            "query": str(query or ""),
            "selected_doc_ids": doc_ids,
            "selected_chunk_ids": selected_chunk_ids,
            "controller_hints": hints,
            "controller_hints_json": json.dumps(hints, sort_keys=True),
            "warnings": warnings,
        }

    return [
        plan_rag_queries,
        search_corpus_chunks,
        grep_corpus_chunks,
        fetch_chunk_window,
        inspect_document_structure,
        search_document_sections,
        filter_indexed_docs,
        grade_evidence_candidates,
        prune_evidence_candidates,
        validate_evidence_plan,
        build_rag_controller_hints,
    ]
