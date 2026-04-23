from __future__ import annotations

import concurrent.futures
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from agentic_chatbot_next.observability.events import RuntimeEvent
from agentic_chatbot_next.rag.retrieval_scope import (
    resolve_collection_ids_for_source,
    resolve_requested_kb_collection_id,
    resolve_upload_collection_id,
)


_STOPWORDS = {
    "about",
    "all",
    "and",
    "base",
    "collection",
    "corpus",
    "doc",
    "docs",
    "document",
    "documents",
    "find",
    "for",
    "from",
    "in",
    "indexed",
    "knowledge",
    "lookup",
    "of",
    "on",
    "policy",
    "search",
    "the",
    "to",
    "use",
    "what",
    "which",
    "with",
}


@dataclass(frozen=True)
class CollectionScore:
    collection_id: str
    score: float = 0.0
    metadata_score: float = 0.0
    keyword_score: float = 0.0
    vector_score: float = 0.0
    metadata_hits: int = 0
    keyword_hits: int = 0
    vector_hits: int = 0
    top_documents: List[Dict[str, Any]] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    @property
    def total_hits(self) -> int:
        return int(self.metadata_hits + self.keyword_hits + self.vector_hits)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collection_id": self.collection_id,
            "score": round(float(self.score), 4),
            "metadata_score": round(float(self.metadata_score), 4),
            "keyword_score": round(float(self.keyword_score), 4),
            "vector_score": round(float(self.vector_score), 4),
            "metadata_hits": int(self.metadata_hits),
            "keyword_hits": int(self.keyword_hits),
            "vector_hits": int(self.vector_hits),
            "total_hits": self.total_hits,
            "top_documents": [dict(item) for item in self.top_documents[:5]],
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class CollectionSelection:
    status: str
    selected_collection_id: str = ""
    ranked_collections: List[CollectionScore] = field(default_factory=list)
    searched_methods: List[str] = field(default_factory=list)
    reason: str = ""
    clarification_options: List[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.status in {"explicit", "single", "selected"} and bool(self.selected_collection_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "selected_collection_id": self.selected_collection_id,
            "ranked_collections": [item.to_dict() for item in self.ranked_collections],
            "searched_methods": list(self.searched_methods),
            "reason": self.reason,
            "clarification_options": list(self.clarification_options),
        }


def select_collection_for_query(
    stores: Any,
    settings: Any,
    session: Any,
    query: str,
    source_type: str = "kb",
    explicit_collection_id: str = "",
    *,
    event_sink: Any | None = None,
) -> CollectionSelection:
    normalized_source = str(source_type or "kb").strip().lower()
    explicit = str(explicit_collection_id or "").strip()
    if not explicit and normalized_source == "kb":
        explicit = resolve_requested_kb_collection_id(session)

    candidate_ids = _candidate_collection_ids(
        stores,
        settings,
        session,
        source_type=normalized_source,
        explicit_collection_id=explicit,
    )
    _emit_selection_event(
        event_sink,
        session,
        "collection_selection_started",
        {
            "query": str(query or ""),
            "source_type": normalized_source or "kb",
            "explicit_collection_id": explicit,
            "candidate_collection_ids": list(candidate_ids),
        },
    )

    if explicit:
        selection = CollectionSelection(
            status="explicit",
            selected_collection_id=explicit,
            ranked_collections=[CollectionScore(collection_id=explicit, score=1.0, reasons=["explicit_collection"])],
            searched_methods=[],
            reason="explicit_collection_requested",
        )
        _emit_completed(event_sink, session, selection, source_type=normalized_source)
        return selection

    if not candidate_ids:
        selection = CollectionSelection(
            status="no_match",
            searched_methods=[],
            reason="no_accessible_collections",
        )
        _emit_completed(event_sink, session, selection, source_type=normalized_source)
        return selection

    if len(candidate_ids) == 1:
        selection = CollectionSelection(
            status="single",
            selected_collection_id=candidate_ids[0],
            ranked_collections=[
                CollectionScore(collection_id=candidate_ids[0], score=1.0, reasons=["single_accessible_collection"])
            ],
            searched_methods=[],
            reason="single_accessible_collection",
        )
        _emit_completed(event_sink, session, selection, source_type=normalized_source)
        return selection

    max_collections = max(1, int(getattr(settings, "max_collection_discovery_collections", 25) or 25))
    narrowed_ids = list(candidate_ids)
    if len(narrowed_ids) > max_collections:
        narrowed_ids = _narrow_large_candidate_set(
            stores,
            query,
            candidate_ids=narrowed_ids,
            tenant_id=_tenant_id(settings, session),
            source_type=normalized_source,
            limit=max_collections,
        )
        if not narrowed_ids:
            options = list(candidate_ids[: min(len(candidate_ids), 10)])
            selection = CollectionSelection(
                status="ambiguous",
                searched_methods=["metadata"],
                reason="too_many_collections_without_metadata_signal",
                clarification_options=options,
            )
            _emit_completed(event_sink, session, selection, source_type=normalized_source)
            return selection

    ranked = _probe_collections(
        stores,
        settings,
        session,
        query,
        collection_ids=narrowed_ids[:max_collections],
        source_type=normalized_source,
    )
    selection = _choose_collection(ranked)
    _emit_completed(event_sink, session, selection, source_type=normalized_source)
    return selection


def selection_answer(selection: CollectionSelection) -> Dict[str, Any]:
    options = list(selection.clarification_options or [])
    if not options:
        options = [item.collection_id for item in selection.ranked_collections[:5] if item.collection_id]
    if selection.status == "no_match":
        message = (
            "I could not find evidence in any accessible collection for this lookup. "
            "Please name the collection you want me to search, or rephrase the lookup terms."
        )
    else:
        option_text = ", ".join(f"`{item}`" for item in options[:5]) or "the available collections"
        message = (
            "I found multiple possible knowledge-base collections for this lookup and need you to choose one: "
            f"{option_text}."
        )
    return {
        "needs_clarification": True,
        "answer": message,
        "message": message,
        "confidence_hint": 0.0,
        "collection_selection": selection.to_dict(),
        "warnings": ["COLLECTION_SELECTION_AMBIGUOUS" if selection.status == "ambiguous" else "COLLECTION_SELECTION_NO_MATCH"],
    }


def apply_selection_to_session(session: Any, selection: CollectionSelection) -> None:
    if not selection.resolved:
        return
    metadata = dict(getattr(session, "metadata", {}) or {})
    metadata.update(
        {
            "kb_collection_id": selection.selected_collection_id,
            "selected_kb_collection_id": selection.selected_collection_id,
            "search_collection_ids": [selection.selected_collection_id],
            "collection_selection": selection.to_dict(),
        }
    )
    try:
        session.metadata = metadata
    except Exception:
        pass


def _candidate_collection_ids(
    stores: Any,
    settings: Any,
    session: Any,
    *,
    source_type: str,
    explicit_collection_id: str,
) -> List[str]:
    ids = list(
        resolve_collection_ids_for_source(
            settings,
            session,
            source_type=source_type,
            explicit_collection_id=explicit_collection_id,
        )
    )
    if explicit_collection_id:
        return _dedupe([explicit_collection_id])

    normalized_source = str(source_type or "kb").strip().lower()
    if normalized_source == "kb" and not _access_summary_authz_enabled(session):
        upload_collection_id = resolve_upload_collection_id(settings, session)
        for item in _catalog_collection_ids(stores, tenant_id=_tenant_id(settings, session)):
            collection_id = str(item.get("collection_id") or "").strip()
            if not collection_id or collection_id == upload_collection_id:
                continue
            source_counts = dict(item.get("source_type_counts") or {})
            if source_counts and int(source_counts.get("kb") or 0) <= 0:
                continue
            ids.append(collection_id)
    return _dedupe(ids)


def _catalog_collection_ids(stores: Any, *, tenant_id: str) -> List[Dict[str, Any]]:
    doc_store = getattr(stores, "doc_store", None)
    list_collections = getattr(doc_store, "list_collections", None)
    if not callable(list_collections):
        return []
    try:
        rows = list_collections(tenant_id=tenant_id)
    except TypeError:
        try:
            rows = list_collections(tenant_id)
        except Exception:
            return []
    except Exception:
        return []
    payload: List[Dict[str, Any]] = []
    for row in rows or []:
        if isinstance(row, dict):
            payload.append(dict(row))
        else:
            payload.append(
                {
                    "collection_id": str(getattr(row, "collection_id", "") or ""),
                    "source_type_counts": dict(getattr(row, "source_type_counts", {}) or {}),
                }
            )
    return payload


def _probe_collections(
    stores: Any,
    settings: Any,
    session: Any,
    query: str,
    *,
    collection_ids: Sequence[str],
    source_type: str,
) -> List[CollectionScore]:
    tenant_id = _tenant_id(settings, session)
    max_parallel = max(1, int(getattr(settings, "max_parallel_collection_probes", 4) or 4))
    searched_methods = _source_methods(stores)
    if not collection_ids:
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_parallel, len(collection_ids))) as executor:
        futures = [
            executor.submit(
                _probe_collection,
                stores,
                query,
                tenant_id=tenant_id,
                collection_id=collection_id,
                source_type=source_type,
                searched_methods=searched_methods,
            )
            for collection_id in collection_ids
        ]
        scores: List[CollectionScore] = []
        for future in concurrent.futures.as_completed(futures):
            try:
                scores.append(future.result())
            except Exception:
                continue
    return sorted(scores, key=lambda item: (item.score, item.total_hits, item.collection_id), reverse=True)


def _probe_collection(
    stores: Any,
    query: str,
    *,
    tenant_id: str,
    collection_id: str,
    source_type: str,
    searched_methods: Sequence[str],
) -> CollectionScore:
    source_filter = "" if source_type in {"all", "*", "any", ""} else source_type
    metadata_score, metadata_hits, metadata_docs = _metadata_probe(
        stores,
        query,
        tenant_id=tenant_id,
        collection_id=collection_id,
        source_type=source_filter,
    )
    keyword_score, keyword_hits, keyword_docs = _chunk_probe(
        stores,
        query,
        tenant_id=tenant_id,
        collection_id=collection_id,
        method="keyword",
    )
    vector_score, vector_hits, vector_docs = _chunk_probe(
        stores,
        query,
        tenant_id=tenant_id,
        collection_id=collection_id,
        method="vector",
    )
    reasons: List[str] = []
    if metadata_hits:
        reasons.append("metadata")
    if keyword_hits:
        reasons.append("keyword")
    if vector_hits:
        reasons.append("vector")
    top_documents = _dedupe_documents([*metadata_docs, *keyword_docs, *vector_docs])
    total_score = metadata_score + keyword_score + vector_score
    return CollectionScore(
        collection_id=collection_id,
        score=total_score,
        metadata_score=metadata_score,
        keyword_score=keyword_score,
        vector_score=vector_score,
        metadata_hits=metadata_hits,
        keyword_hits=keyword_hits,
        vector_hits=vector_hits,
        top_documents=top_documents,
        reasons=[item for item in reasons if item in searched_methods],
    )


def _metadata_probe(
    stores: Any,
    query: str,
    *,
    tenant_id: str,
    collection_id: str,
    source_type: str,
) -> tuple[float, int, List[Dict[str, Any]]]:
    doc_store = getattr(stores, "doc_store", None)
    if doc_store is None:
        return (0.0, 0, [])
    best_score = 0.0
    hits = 0
    docs: List[Dict[str, Any]] = []
    fuzzy_search_title = getattr(doc_store, "fuzzy_search_title", None)
    if callable(fuzzy_search_title):
        try:
            fuzzy_rows = fuzzy_search_title(query, tenant_id=tenant_id, limit=5, collection_id=collection_id)
        except TypeError:
            try:
                fuzzy_rows = fuzzy_search_title(query, tenant_id, 5, collection_id)
            except Exception:
                fuzzy_rows = []
        except Exception:
            fuzzy_rows = []
        for row in fuzzy_rows or []:
            try:
                score = float(row.get("score") or 0.0)
            except Exception:
                score = 0.0
            doc_id = str(row.get("doc_id") or "")
            if doc_id:
                docs.append(
                    {
                        "doc_id": doc_id,
                        "title": str(row.get("title") or doc_id),
                        "collection_id": collection_id,
                        "method": "metadata",
                    }
                )
                hits += 1
                best_score = max(best_score, score)

    for record in _list_documents(doc_store, tenant_id=tenant_id, collection_id=collection_id, source_type=source_type):
        score = _metadata_match_score(query, record, collection_id=collection_id)
        if score <= 0.0:
            continue
        hits += 1
        best_score = max(best_score, score)
        docs.append(_record_document(record, method="metadata", fallback_collection_id=collection_id))
    return (min(2.0, best_score + (hits * 0.08)), hits, docs)


def _chunk_probe(
    stores: Any,
    query: str,
    *,
    tenant_id: str,
    collection_id: str,
    method: str,
) -> tuple[float, int, List[Dict[str, Any]]]:
    chunk_store = getattr(stores, "chunk_store", None)
    search = getattr(chunk_store, f"{method}_search", None)
    if not callable(search):
        return (0.0, 0, [])
    try:
        hits = search(
            query,
            top_k=4,
            collection_id_filter=collection_id,
            tenant_id=tenant_id,
        )
    except Exception:
        return (0.0, 0, [])
    docs: List[Dict[str, Any]] = []
    best = 0.0
    count = 0
    for hit in hits or []:
        try:
            score = float(getattr(hit, "score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        best = max(best, score)
        count += 1
        docs.append(_chunk_document(hit, method=method, fallback_collection_id=collection_id))
    multiplier = 1.15 if method == "keyword" else 1.0
    return (min(2.5, (best * multiplier) + (count * 0.12)), count, docs)


def _choose_collection(ranked: Sequence[CollectionScore]) -> CollectionSelection:
    ordered = list(ranked)
    searched_methods = _searched_methods_from_scores(ordered)
    options = [item.collection_id for item in ordered[:5] if item.collection_id]
    if not ordered or ordered[0].score <= 0.0 or ordered[0].total_hits <= 0:
        return CollectionSelection(
            status="no_match",
            ranked_collections=ordered,
            searched_methods=searched_methods,
            reason="no_collection_evidence",
            clarification_options=options,
        )
    if len(ordered) == 1:
        return CollectionSelection(
            status="selected",
            selected_collection_id=ordered[0].collection_id,
            ranked_collections=ordered,
            searched_methods=searched_methods,
            reason="only_collection_with_evidence",
        )
    top = ordered[0]
    second = ordered[1]
    margin = top.score - second.score
    required_margin = max(0.25, second.score * 0.2)
    if second.score <= 0.0 or margin >= required_margin:
        return CollectionSelection(
            status="selected",
            selected_collection_id=top.collection_id,
            ranked_collections=ordered,
            searched_methods=searched_methods,
            reason="clear_collection_match",
        )
    return CollectionSelection(
        status="ambiguous",
        ranked_collections=ordered,
        searched_methods=searched_methods,
        reason="collection_candidates_tied",
        clarification_options=options,
    )


def _narrow_large_candidate_set(
    stores: Any,
    query: str,
    *,
    candidate_ids: Sequence[str],
    tenant_id: str,
    source_type: str,
    limit: int,
) -> List[str]:
    scored: List[tuple[float, str]] = []
    source_filter = "" if source_type in {"all", "*", "any", ""} else source_type
    doc_store = getattr(stores, "doc_store", None)
    if doc_store is None:
        return []
    for collection_id in candidate_ids:
        score = 0.0
        records = _list_documents(
            doc_store,
            tenant_id=tenant_id,
            collection_id=collection_id,
            source_type=source_filter,
        )[:25]
        for record in records:
            score = max(score, _metadata_match_score(query, record, collection_id=collection_id))
        if score > 0.0:
            scored.append((score, collection_id))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [collection_id for _score, collection_id in scored[:limit]]


def _list_documents(
    doc_store: Any,
    *,
    tenant_id: str,
    collection_id: str,
    source_type: str,
) -> List[Any]:
    list_documents = getattr(doc_store, "list_documents", None)
    if not callable(list_documents):
        return []
    try:
        return list(
            list_documents(
                tenant_id=tenant_id,
                collection_id=collection_id,
                source_type=source_type,
            )
            or []
        )
    except TypeError:
        try:
            rows = list_documents(tenant_id=tenant_id, collection_id=collection_id) or []
        except TypeError:
            try:
                rows = list_documents(tenant_id, collection_id) or []
            except Exception:
                return []
        except Exception:
            return []
        if not source_type:
            return list(rows)
        return [
            row
            for row in rows
            if str(getattr(row, "source_type", "") or "").strip().lower() == source_type
        ]
    except Exception:
        return []


def _metadata_match_score(query: str, record: Any, *, collection_id: str) -> float:
    terms = _query_terms(query)
    if not terms:
        return 0.0
    title = str(getattr(record, "title", "") or "")
    source_path = str(getattr(record, "source_path", "") or "")
    source_name = Path(source_path).name if source_path else ""
    haystacks = [
        str(collection_id or ""),
        title,
        source_name,
        source_path,
        str(getattr(record, "doc_structure_type", "") or ""),
    ]
    best = 0.0
    for haystack in haystacks:
        haystack_terms = _query_terms(haystack)
        overlap = len(terms & haystack_terms)
        if not overlap:
            continue
        best = max(best, min(1.25, 0.2 + (overlap * 0.22)))
    normalized_query = " ".join(sorted(terms))
    normalized_title = " ".join(sorted(_query_terms(title)))
    if normalized_query and normalized_query in normalized_title:
        best = max(best, 1.4)
    return best


def _query_terms(value: str) -> set[str]:
    normalized = str(value or "").lower().replace("_", " ").replace("-", " ")
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]{3,}", normalized)
        if token not in _STOPWORDS and not token.isdigit()
    }


def _record_document(record: Any, *, method: str, fallback_collection_id: str = "") -> Dict[str, Any]:
    return {
        "doc_id": str(getattr(record, "doc_id", "") or ""),
        "title": str(getattr(record, "title", "") or ""),
        "source_path": str(getattr(record, "source_path", "") or ""),
        "collection_id": str(getattr(record, "collection_id", "") or fallback_collection_id or ""),
        "method": method,
    }


def _chunk_document(hit: Any, *, method: str, fallback_collection_id: str = "") -> Dict[str, Any]:
    doc = getattr(hit, "doc", None)
    metadata = dict(getattr(doc, "metadata", {}) or {})
    return {
        "doc_id": str(metadata.get("doc_id") or ""),
        "title": str(metadata.get("title") or ""),
        "source_path": str(metadata.get("source_path") or ""),
        "collection_id": str(metadata.get("collection_id") or fallback_collection_id or ""),
        "method": method,
    }


def _dedupe_documents(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    payload: List[Dict[str, Any]] = []
    for item in items:
        doc_id = str(item.get("doc_id") or "")
        title = str(item.get("title") or "")
        key = doc_id or title
        if not key or key in seen:
            continue
        seen.add(key)
        payload.append(dict(item))
    return payload


def _searched_methods_from_scores(scores: Sequence[CollectionScore]) -> List[str]:
    methods: List[str] = []
    for score in scores:
        if score.metadata_hits and "metadata" not in methods:
            methods.append("metadata")
        if score.keyword_hits and "keyword" not in methods:
            methods.append("keyword")
        if score.vector_hits and "vector" not in methods:
            methods.append("vector")
    return methods or ["metadata", "keyword", "vector"]


def _source_methods(stores: Any) -> List[str]:
    methods = ["metadata"]
    chunk_store = getattr(stores, "chunk_store", None)
    if callable(getattr(chunk_store, "keyword_search", None)):
        methods.append("keyword")
    if callable(getattr(chunk_store, "vector_search", None)):
        methods.append("vector")
    return methods


def _tenant_id(settings: Any, session: Any) -> str:
    return str(getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev")) or "local-dev")


def _dedupe(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    payload: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        payload.append(text)
    return payload


def _access_summary_authz_enabled(session: Any) -> bool:
    metadata = dict(getattr(session, "metadata", {}) or {})
    access_summary = dict(metadata.get("access_summary") or getattr(session, "access_summary", {}) or {})
    return bool(access_summary.get("authz_enabled") or access_summary.get("enabled"))


def _emit_completed(
    event_sink: Any | None,
    session: Any,
    selection: CollectionSelection,
    *,
    source_type: str,
) -> None:
    _emit_selection_event(
        event_sink,
        session,
        "collection_selection_completed",
        {
            "source_type": source_type or "kb",
            **selection.to_dict(),
        },
    )


def _emit_selection_event(
    event_sink: Any | None,
    session: Any,
    event_type: str,
    payload: Dict[str, Any],
) -> None:
    sink = event_sink or getattr(session, "event_sink", None)
    if sink is None or not hasattr(sink, "emit"):
        return
    session_id = str(getattr(session, "session_id", "") or "")
    if not session_id:
        return
    try:
        sink.emit(
            RuntimeEvent(
                event_type=event_type,
                session_id=session_id,
                agent_name=str(getattr(session, "active_agent", "") or ""),
                payload=dict(payload or {}),
            )
        )
    except Exception:
        return


__all__ = [
    "CollectionScore",
    "CollectionSelection",
    "apply_selection_to_session",
    "select_collection_for_query",
    "selection_answer",
]
