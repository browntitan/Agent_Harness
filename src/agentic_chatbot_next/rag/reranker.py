from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Tuple, TypeVar

import httpx


DEFAULT_RERANK_MODEL = "rjmalagon/mxbai-rerank-large-v2:1.5b-fp16"


@dataclass
class RerankDecision:
    status: str
    provider: str = ""
    model: str = ""
    top_n: int = 0
    scores: Dict[str, float] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "status": self.status,
            "provider": self.provider,
            "model": self.model,
            "top_n": self.top_n,
            "scores": dict(self.scores),
        }
        if self.error:
            payload["error"] = self.error
        return payload


T = TypeVar("T")


def _enabled(settings: Any) -> bool:
    return bool(getattr(settings, "rerank_enabled", False))


def _provider(settings: Any) -> str:
    return str(getattr(settings, "rerank_provider", "ollama") or "ollama").strip().lower()


def _model(settings: Any) -> str:
    return str(getattr(settings, "rerank_model", DEFAULT_RERANK_MODEL) or DEFAULT_RERANK_MODEL).strip()


def _top_n(settings: Any, count: int) -> int:
    configured = int(getattr(settings, "rerank_top_n", 12) or 12)
    return max(1, min(max(1, configured), max(1, int(count))))


def _timeout(settings: Any) -> int:
    return max(1, int(getattr(settings, "rerank_timeout_seconds", 30) or 30))


def _candidate_id(index: int, candidate: Any) -> str:
    if isinstance(candidate, dict):
        explicit = str(candidate.get("candidate_id") or candidate.get("id") or "").strip()
        if explicit:
            return explicit
        citation_ids = candidate.get("citation_ids")
        if isinstance(citation_ids, list) and citation_ids:
            return str(citation_ids[0])
        chunk_ids = candidate.get("chunk_ids")
        if isinstance(chunk_ids, list) and chunk_ids:
            return str(chunk_ids[0])
        doc_id = str(candidate.get("doc_id") or "").strip()
        method = str(candidate.get("query_method") or "").strip()
        if doc_id:
            return f"{doc_id}:{method or index}"
    return f"candidate-{index}"


def _dict_candidate_text(candidate: Dict[str, Any]) -> str:
    metadata = dict(candidate.get("metadata") or {})
    source = dict(metadata.get("source") or {})
    pieces = [
        str(candidate.get("title") or ""),
        str(candidate.get("source_path") or source.get("source_path") or ""),
        str(candidate.get("source_type") or source.get("source_type") or ""),
        str(candidate.get("query_method") or ""),
        " ".join(str(item) for item in (candidate.get("relationship_path") or []) if str(item).strip()),
        str(candidate.get("summary") or ""),
    ]
    text = " ".join(part.strip() for part in pieces if part and str(part).strip())
    return " ".join(text.split())[:1600]


def _chunk_candidate_text(chunk: Any) -> str:
    doc = getattr(chunk, "doc", None)
    metadata = dict(getattr(doc, "metadata", {}) or {}) if doc is not None else {}
    pieces = [
        str(metadata.get("title") or ""),
        str(metadata.get("source_path") or ""),
        str(metadata.get("section_title") or ""),
        str(metadata.get("sheet_name") or ""),
        str(metadata.get("file_type") or ""),
        str(getattr(doc, "page_content", "") or ""),
    ]
    text = " ".join(part.strip() for part in pieces if part and str(part).strip())
    return " ".join(text.split())[:1600]


def _extract_json_payload(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass
    starts = [pos for pos in [raw.find("{"), raw.find("[")] if pos >= 0]
    ends = [pos for pos in [raw.rfind("}"), raw.rfind("]")] if pos >= 0]
    if not starts or not ends:
        return None
    start = min(starts)
    end = max(ends)
    if end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None


def _coerce_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return score


def _score_map_from_payload(payload: Any, ids: Sequence[str]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    if isinstance(payload, dict):
        if isinstance(payload.get("scores"), list):
            payload = payload.get("scores")
        elif isinstance(payload.get("results"), list):
            payload = payload.get("results")
        elif all(key in payload for key in ids):
            for candidate_id in ids:
                score = _coerce_score(payload.get(candidate_id))
                if score is not None:
                    scores[candidate_id] = score
            return scores
    if isinstance(payload, list):
        if all(not isinstance(item, dict) for item in payload):
            for candidate_id, raw_score in zip(ids, payload):
                score = _coerce_score(raw_score)
                if score is not None:
                    scores[candidate_id] = score
            return scores
        for item in payload:
            if not isinstance(item, dict):
                continue
            candidate_id = str(
                item.get("id")
                or item.get("candidate_id")
                or item.get("corpus_id")
                or item.get("index")
                or ""
            ).strip()
            if candidate_id.isdigit():
                idx = int(candidate_id)
                candidate_id = ids[idx] if 0 <= idx < len(ids) else candidate_id
            score = _coerce_score(
                item.get("score")
                if "score" in item
                else item.get("relevance_score")
                if "relevance_score" in item
                else item.get("relevance")
            )
            if candidate_id and score is not None:
                scores[candidate_id] = score
    return scores


def _post_ollama_rerank(
    *,
    base_url: str,
    model: str,
    query: str,
    candidates: Sequence[Tuple[str, str]],
    timeout_seconds: int,
) -> Dict[str, float]:
    candidate_payload = [
        {"id": candidate_id, "text": text}
        for candidate_id, text in candidates
        if str(candidate_id).strip() and str(text).strip()
    ]
    prompt = (
        "You are a document reranker. Score each candidate for relevance to the query.\n"
        "Return ONLY valid JSON in this exact shape: "
        '{"scores":[{"id":"candidate id","score":0.0}]}.\n'
        "Scores must be numbers; higher means more relevant.\n\n"
        f"QUERY:\n{query}\n\n"
        "CANDIDATES:\n"
        + json.dumps(candidate_payload, ensure_ascii=False)
    )
    url = str(base_url or "").rstrip("/") + "/api/chat"
    with httpx.Client(timeout=httpx.Timeout(timeout_seconds)) as client:
        response = client.post(
            url,
            json={
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": "Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "format": "json",
                "options": {"temperature": 0},
            },
        )
        response.raise_for_status()
        data = response.json()
    content = ""
    message = data.get("message") if isinstance(data, dict) else None
    if isinstance(message, dict):
        content = str(message.get("content") or "")
    if not content and isinstance(data, dict):
        content = str(data.get("response") or "")
    parsed = _extract_json_payload(content)
    return _score_map_from_payload(parsed, [candidate_id for candidate_id, _text in candidates])


def rerank_items(
    settings: Any,
    *,
    query: str,
    items: Sequence[T],
    text_fn: Callable[[T], str],
    id_fn: Callable[[int, T], str] | None = None,
) -> tuple[List[T], RerankDecision]:
    original = list(items)
    if len(original) <= 1:
        return original, RerankDecision(status="skipped_single_candidate")
    if not _enabled(settings):
        return original, RerankDecision(status="disabled")
    provider = _provider(settings)
    model = _model(settings)
    if provider != "ollama":
        return original, RerankDecision(status="unsupported_provider", provider=provider, model=model)
    base_url = str(getattr(settings, "ollama_base_url", "") or "").strip()
    if not base_url or not model:
        return original, RerankDecision(status="missing_config", provider=provider, model=model)

    top_n = _top_n(settings, len(original))
    head = original[:top_n]
    tail = original[top_n:]
    indexed_head: List[Tuple[str, int, T]] = [
        (
            id_fn(index, item) if id_fn is not None else f"candidate-{index}",
            index,
            item,
        )
        for index, item in enumerate(head)
    ]
    ids = [candidate_id for candidate_id, _index, _item in indexed_head]
    candidates: List[Tuple[str, str]] = []
    for candidate_id, _index, item in indexed_head:
        text = text_fn(item)
        if str(text).strip():
            candidates.append((candidate_id, text))
    if not candidates:
        return original, RerankDecision(status="no_candidate_text", provider=provider, model=model, top_n=top_n)

    try:
        scores = _post_ollama_rerank(
            base_url=base_url,
            model=model,
            query=query,
            candidates=candidates,
            timeout_seconds=_timeout(settings),
        )
    except Exception as exc:
        if bool(getattr(settings, "rerank_fallback_to_heuristics", True)):
            return original, RerankDecision(
                status="fallback",
                provider=provider,
                model=model,
                top_n=top_n,
                error=f"{type(exc).__name__}: {exc}",
            )
        raise

    if not scores:
        return original, RerankDecision(
            status="fallback",
            provider=provider,
            model=model,
            top_n=top_n,
            error="Reranker returned no parseable scores.",
        )

    reranked_indexed_head = sorted(
        indexed_head,
        key=lambda item: (
            -scores.get(item[0], float("-inf")),
            item[1],
        ),
    )
    reranked_head = [item for _candidate_id, _index, item in reranked_indexed_head]
    return reranked_head + tail, RerankDecision(
        status="reranked",
        provider=provider,
        model=model,
        top_n=top_n,
        scores=scores,
    )


def rerank_graph_candidates(
    settings: Any,
    *,
    query: str,
    candidates: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    original = [dict(item) for item in candidates]
    candidate_ids = [_candidate_id(index, candidate) for index, candidate in enumerate(original)]
    object_candidate_ids = {
        id(candidate): candidate_ids[index]
        for index, candidate in enumerate(original)
    }

    def id_fn(index: int, candidate: Dict[str, Any]) -> str:
        del candidate
        if 0 <= index < len(candidate_ids):
            return candidate_ids[index]
        return f"candidate-{index}"

    reranked, decision = rerank_items(
        settings,
        query=query,
        items=original,
        id_fn=id_fn,
        text_fn=_dict_candidate_text,
    )
    scores = decision.scores
    if scores:
        for index, candidate in enumerate(reranked):
            candidate_id = object_candidate_ids.get(id(candidate), _candidate_id(index, candidate))
            if candidate_id in scores:
                candidate["rerank_score"] = scores[candidate_id]
                metadata = dict(candidate.get("metadata") or {})
                metadata["rerank"] = {
                    "provider": decision.provider,
                    "model": decision.model,
                    "score": scores[candidate_id],
                }
                candidate["metadata"] = metadata
    return reranked, decision.to_dict()


def rerank_scored_chunks(settings: Any, *, query: str, chunks: Sequence[T]) -> tuple[List[T], Dict[str, Any]]:
    reranked, decision = rerank_items(
        settings,
        query=query,
        items=list(chunks),
        text_fn=_chunk_candidate_text,
        id_fn=lambda index, _item: f"chunk-{index}",
    )
    return reranked, decision.to_dict()


__all__ = [
    "DEFAULT_RERANK_MODEL",
    "RerankDecision",
    "rerank_graph_candidates",
    "rerank_items",
    "rerank_scored_chunks",
]
