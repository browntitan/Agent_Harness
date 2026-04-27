from __future__ import annotations

import re
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence

from langchain_core.documents import Document

from agentic_chatbot_next.contracts.rag import RetrievalSummary
from agentic_chatbot_next.persistence.postgres.chunks import ChunkRecord, ScoredChunk
from agentic_chatbot_next.prompting import load_judge_rewrite_prompt, render_template
from agentic_chatbot_next.rag.entity_linking import resolve_query_entities
from agentic_chatbot_next.rag.fanout import (
    RagRuntimeBridge,
    RagSearchTask,
    RagSearchTaskResult,
    deserialize_document,
    deserialize_graded_chunk,
)
from agentic_chatbot_next.rag.discovery_precision import (
    document_has_explicit_topic_support,
    workflow_topic_seed_terms,
)

if TYPE_CHECKING:  # pragma: no cover - import-only typing surface
    from agentic_chatbot_next.graph.planner import SourcePlan


def _new_source_plan(query: str) -> "SourcePlan":
    from agentic_chatbot_next.graph.planner import SourcePlan

    return SourcePlan(query=query)


def _plan_sources(*args: Any, **kwargs: Any) -> "SourcePlan":
    from agentic_chatbot_next.graph.planner import plan_sources

    return plan_sources(*args, **kwargs)
from agentic_chatbot_next.rag.hints import (
    coerce_controller_hints,
    normalize_coverage_goal,
    normalize_research_profile,
    normalize_result_mode,
)
from agentic_chatbot_next.rag.collection_selection import (
    apply_selection_to_session,
    select_collection_for_query,
)
from agentic_chatbot_next.rag.retrieval_scope import resolve_collection_ids_for_source
from agentic_chatbot_next.rag.retrieval import GradedChunk, grade_chunks, merge_dedupe, rank_fuse_dedupe, retrieve_candidates
from agentic_chatbot_next.rag.verification import verify_retrieval_quality
from agentic_chatbot_next.utils.json_utils import extract_json

_STOPWORDS = {
    "about",
    "across",
    "after",
    "agent",
    "all",
    "also",
    "an",
    "and",
    "answer",
    "any",
    "are",
    "around",
    "based",
    "be",
    "can",
    "cite",
    "cited",
    "corpus",
    "document",
    "documents",
    "does",
    "each",
    "evidence",
    "file",
    "files",
    "find",
    "for",
    "from",
    "give",
    "grounded",
    "how",
    "identify",
    "in",
    "include",
    "information",
    "into",
    "is",
    "it",
    "its",
    "key",
    "list",
    "many",
    "me",
    "needle",
    "of",
    "on",
    "or",
    "out",
    "over",
    "process",
    "provide",
    "question",
    "response",
    "return",
    "search",
    "should",
    "show",
    "that",
    "the",
    "their",
    "them",
    "these",
    "this",
    "through",
    "use",
    "uses",
    "using",
    "what",
    "which",
    "with",
    "workflow",
}

_DISCOVERY_HINTS = re.compile(
    r"\b("
    r"identify\s+all\s+documents|which\s+documents|list\s+(?:all\s+)?(?:documents|files)|"
    r"across\s+(?:the\s+)?(?:corpus|documents)|find\s+all|every\s+document|"
    r"process\s+flows?|workflows?|flowcharts?|procedure\s+steps"
    r")\b",
    re.IGNORECASE,
)

_COMPLEX_HINTS = re.compile(
    r"\b("
    r"compare|difference|differences|cross[-\s]?reference|multi[-\s]?step|"
    r"step\s+by\s+step|verify|investigate|filter|narrow|broad|reason|"
    r"needle|haystack|parallel|across|sufficient|exhaustive"
    r")\b",
    re.IGNORECASE,
)

_EXACT_MATCH_HINTS = re.compile(
    r"\"[^\"]+\"|\b(?:clause|section|article)\s+[a-z0-9.]+\b|\bREQ-\d+\b",
    re.IGNORECASE,
)

_PROCESS_FLOW_HINTS = re.compile(
    r"\b(process\s+flows?|workflows?|flowcharts?|handoff|approval\s+flows?|escalation)\b",
    re.IGNORECASE,
)

_COMPARISON_HINTS = re.compile(r"\b(compare|difference|differences|versus|vs\.?|contrast)\b", re.IGNORECASE)
_LATEST_HINTS = re.compile(r"\b(current|latest|approved|authoritative|final|most recent)\b", re.IGNORECASE)
_DRAFT_HINTS = re.compile(r"\b(draft|earlier|history|historical|emerg|preliminary|initial)\b", re.IGNORECASE)
_WORKBOOK_FOCUS_HINTS = re.compile(
    r"\b(budget|schedule|staffing|scorecard|kpi|bom|cost|costs|training|spares|milestone|milestones|variance|supplier|suppliers|price|procurement|resource|deployment|deploy|rollout|ims|risks?)\b",
    re.IGNORECASE,
)
_CAUSAL_REWRITE_HINTS = re.compile(
    r"\b(because|caused|cause|reason|why|driver|drove|driven|factor|issue|root\s+cause|attributed|resulted)\b",
    re.IGNORECASE,
)
_REFUTATION_REWRITE_HINTS = re.compile(
    r"\b(says|claim|claims|claimed|because|whether|true|false|instead|rather|not|better\s+evidence)\b",
    re.IGNORECASE,
)
_STATUS_OUTCOME_REWRITE_HINTS = re.compile(
    r"\b(status|date|deadline|milestone|schedule|slip|slipped|delay|delayed|move|moved|outcome|result|final|approved|ready|readiness)\b|\b\d{4}\b",
    re.IGNORECASE,
)


def _resolve_collection_id(settings: Any, session: Any) -> str:
    collection_ids = _resolve_collection_ids(settings, session)
    if collection_ids:
        return str(collection_ids[0])
    return str(getattr(settings, "default_collection_id", "default") or "default")


def _resolve_collection_ids(settings: Any, session: Any, *, explicit_collection_id: str = "") -> List[str]:
    explicit = str(explicit_collection_id or "").strip()
    if explicit:
        return [explicit]
    return [
        str(item)
        for item in resolve_collection_ids_for_source(
            settings,
            session,
            source_type="kb",
        )
        if str(item).strip()
    ]


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.casefold().split())


def _query_terms(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]{3,}", _normalize_text(value)))


def _augment_query_with_aliases(query: str) -> str:
    return query


def _emit_progress(progress_emitter: Any | None, event_type: str, **payload: Any) -> None:
    if progress_emitter is None or not hasattr(progress_emitter, "emit_progress"):
        return
    progress_emitter.emit_progress(event_type, **payload)


class RetrievalBudget:
    def __init__(self, *, budget_ms: int = 0, synthesis_reserve_ms: int = 30_000) -> None:
        self.budget_ms = max(0, int(budget_ms or 0))
        self.synthesis_reserve_ms = max(0, int(synthesis_reserve_ms or 0))
        self.started_at = time.perf_counter()
        self.stage_timings_ms: Dict[str, float] = {}
        self.budget_exhausted = False

    @contextmanager
    def measure(self, stage: str):
        name = str(stage or "unknown").strip() or "unknown"
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            self.stage_timings_ms[name] = round(self.stage_timings_ms.get(name, 0.0) + elapsed, 3)
            if self.budget_ms and self.elapsed_ms() >= self.budget_ms:
                self.budget_exhausted = True

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000.0

    def remaining_ms(self) -> float:
        if not self.budget_ms:
            return float("inf")
        return max(0.0, float(self.budget_ms) - self.elapsed_ms())

    def should_cutoff(self, *, reserve_ms: int | None = None) -> bool:
        if not self.budget_ms:
            return False
        reserve = self.synthesis_reserve_ms if reserve_ms is None else max(0, int(reserve_ms))
        if self.remaining_ms() <= reserve:
            self.budget_exhausted = True
            return True
        return False

    def slow_stages(self, *, threshold_ms: int = 5_000, limit: int = 5) -> List[str]:
        ranked = sorted(self.stage_timings_ms.items(), key=lambda item: item[1], reverse=True)
        return [name for name, elapsed in ranked if elapsed >= threshold_ms][:limit]

    def to_fields(self) -> Dict[str, Any]:
        return {
            "stage_timings_ms": dict(self.stage_timings_ms),
            "budget_ms": int(self.budget_ms),
            "budget_exhausted": bool(self.budget_exhausted),
            "slow_stages": self.slow_stages(),
        }


def _new_retrieval_budget(settings: Any, controller_hints: Dict[str, Any]) -> RetrievalBudget:
    raw_budget = (
        controller_hints.get("rag_budget_ms")
        or controller_hints.get("budget_ms")
        or getattr(settings, "rag_budget_ms", 0)
        or 0
    )
    raw_reserve = (
        controller_hints.get("rag_budget_synthesis_reserve_ms")
        or getattr(settings, "rag_budget_synthesis_reserve_ms", 30_000)
        or 30_000
    )
    return RetrievalBudget(budget_ms=int(raw_budget or 0), synthesis_reserve_ms=int(raw_reserve or 0))


@contextmanager
def _measure_stage(stage_timer: RetrievalBudget | None, stage: str):
    if stage_timer is None:
        yield
        return
    with stage_timer.measure(stage):
        yield


def _chunk_id(doc: Document) -> str:
    return str((doc.metadata or {}).get("chunk_id") or "")


def _doc_id(doc: Document) -> str:
    return str((doc.metadata or {}).get("doc_id") or "")


def _title(doc: Document) -> str:
    return str((doc.metadata or {}).get("title") or "")


def _adaptive_score(doc: Document) -> float:
    try:
        return float((doc.metadata or {}).get("_adaptive_score") or 0.0)
    except Exception:
        return 0.0


def _with_adaptive_metadata(doc: Document, *, score: float, strategy: str, query: str, rationale: str) -> Document:
    metadata = dict(doc.metadata or {})
    metadata["_adaptive_score"] = max(float(score), float(metadata.get("_adaptive_score") or 0.0))
    metadata["_adaptive_strategy"] = strategy
    metadata["_adaptive_query"] = query
    metadata["_adaptive_rationale"] = rationale
    return Document(page_content=doc.page_content, metadata=metadata)


def _title_overlap_score(question: str, doc: Document) -> int:
    title = _title(doc).lower()
    if not title:
        return 0
    q_terms = set(re.findall(r"[A-Za-z0-9_]{3,}", question.lower()))
    t_terms = set(re.findall(r"[A-Za-z0-9_]{3,}", title.replace("_", " ")))
    overlap = len(q_terms & t_terms)
    if "architecture" in q_terms and "architecture" in t_terms:
        overlap += 2
    if "workflow" in q_terms and "workflow" in t_terms:
        overlap += 2
    return overlap


def _select_evidence_docs(question: str, graded: Sequence[GradedChunk], min_chunks: int) -> List[Document]:
    target = max(1, int(min_chunks))
    strong = [item.doc for item in graded if item.relevance >= 2]
    strong.sort(key=lambda doc: (_adaptive_score(doc), _title_overlap_score(question, doc)), reverse=True)
    if len(strong) >= target:
        return strong[:target]

    supplemental = [item.doc for item in graded if item.relevance == 1]
    supplemental.sort(key=lambda doc: (_adaptive_score(doc), _title_overlap_score(question, doc)), reverse=True)
    return (strong + supplemental)[:target]


def _normalize_search_mode(value: str) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized not in {"fast", "auto", "deep"}:
        return "auto"
    return normalized


def _is_discovery_query(question: str) -> bool:
    return bool(_DISCOVERY_HINTS.search(question))


def _is_complex_query(question: str) -> bool:
    return len(question.split()) >= 18 or bool(_COMPLEX_HINTS.search(question) or _DISCOVERY_HINTS.search(question))


def _looks_simple_fact_query(question: str) -> bool:
    lowered = str(question or "").strip().lower()
    if not lowered:
        return False
    if _COMPARISON_HINTS.search(lowered) or _DISCOVERY_HINTS.search(lowered):
        return False
    if re.search(r"\b(compare|contrast|extract|summarize|all|every|each|across|relationship|dependencies|inventory|list)\b", lowered):
        return False
    return bool(re.search(r"\b(what|when|where|who|which|how many|does|is|are)\b", lowered)) and len(lowered.split()) <= 24


def _is_comparison_query(question: str) -> bool:
    return bool(_COMPARISON_HINTS.search(question))


def _is_exact_match_query(question: str) -> bool:
    return bool(_EXACT_MATCH_HINTS.search(question))


def _prefer_process_flow(question: str, *, research_profile: str, controller_hints: Dict[str, Any]) -> bool:
    return bool(
        _PROCESS_FLOW_HINTS.search(question)
        or research_profile == "process_flow_identification"
        or controller_hints.get("prefer_process_flow_docs")
    )


def _discovery_mode(
    question: str,
    *,
    research_profile: str,
    coverage_goal: str,
    result_mode: str,
    controller_hints: Dict[str, Any],
) -> bool:
    return bool(
        _is_discovery_query(question)
        or research_profile in {"corpus_discovery", "process_flow_identification"}
        or coverage_goal in {"corpus_wide", "exhaustive"}
        or result_mode == "inventory"
        or controller_hints.get("prefer_inventory_output")
    )


def _keyword_projection(question: str) -> str:
    quoted = [item.strip() for item in re.findall(r'"([^"]+)"', question) if item.strip()]
    if quoted:
        return " ".join(quoted[:2])

    tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9_]{3,}", question.lower())
        if token not in _STOPWORDS and not token.isdigit()
    ]
    if not tokens:
        return question.strip()

    seen: set[str] = set()
    projected: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        projected.append(token)
        if len(projected) >= 6:
            break
    return " ".join(projected) or question.strip()


def _compact_visible_projection(value: str, *, limit: int = 8) -> str:
    terms: List[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9_]{3,}", str(value or "").lower()):
        if token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= max(1, int(limit)):
            break
    return " ".join(terms)


def _visible_claim_from_question(question: str) -> str:
    text = str(question or "").strip()
    quoted = [item.strip() for item in re.findall(r'"([^"]+)"', text) if item.strip()]
    if quoted:
        return quoted[0]
    patterns = (
        r"\b(?:someone|they|the\s+source|the\s+document)\s+says?\s+(.+?)(?:,\s*(?:what|which|how|is)\b|\?\s*$|$)",
        r"\bclaim(?:s|ed)?\s+(?:that\s+)?(.+?)(?:,\s*(?:what|which|how|is)\b|\?\s*$|$)",
        r"\bif\s+(.+?)(?:,\s*(?:what|which|how|is)\b|\?\s*$|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            claim = re.sub(r"\s+", " ", match.group(1)).strip(" .,:;")
            if claim:
                return claim
    return ""


def _source_title_projection(conversation_context: str) -> str:
    title_fragments: List[str] = []
    for line in str(conversation_context or "").splitlines():
        if "doc_id:" not in line and "candidate" not in line.lower():
            continue
        title_fragments.append(line)
    if not title_fragments:
        return ""
    return _compact_visible_projection(" ".join(title_fragments), limit=8)


def _generic_rewrite_candidates(
    question: str,
    *,
    conversation_context: str,
    round_index: int,
) -> List[tuple[str, str, str]]:
    base_terms = _compact_visible_projection(question, limit=8)
    if not base_terms:
        return []
    variants: List[tuple[str, str, str]] = []
    claim = _visible_claim_from_question(question)
    if claim:
        claim_terms = _compact_visible_projection(claim, limit=8)
        if claim_terms:
            variants.append((f"{claim_terms} evidence support refute", "keyword", "claim_focused_rewrite"))
    variants.append((f"{base_terms} evidence support", "hybrid", "entity_focused_rewrite"))
    if _REFUTATION_REWRITE_HINTS.search(question):
        variants.append((f"{base_terms} support refute contradiction evidence", "keyword", "refutation_focused_rewrite"))
    if _CAUSAL_REWRITE_HINTS.search(question):
        variants.append((f"{base_terms} cause reason driver factor evidence", "keyword", "causal_factor_rewrite"))
    if _STATUS_OUTCOME_REWRITE_HINTS.search(question):
        variants.append((f"{base_terms} status outcome date milestone evidence", "keyword", "status_outcome_rewrite"))
    if round_index > 1:
        title_terms = _source_title_projection(conversation_context)
        if title_terms:
            variants.append((f"{base_terms} {title_terms}", "hybrid", "source_metadata_rewrite"))
    return variants[:5]


def _terms_from_text(value: str) -> List[str]:
    return [token for token in re.findall(r"[A-Za-z0-9_]{4,}", str(value or "").lower()) if token not in _STOPWORDS]


def _facet_terms(value: str) -> List[str]:
    return [token for token in re.findall(r"[A-Za-z0-9_]{2,}", str(value or "").lower()) if token not in _STOPWORDS]


_FACET_SPAN_RE = re.compile(
    r"\b(?:about|covering|including|includes?|regarding|concerning|related\s+to|focused\s+on|on)\b\s+(?P<span>[^.?!;]+)",
    re.IGNORECASE,
)
_FACET_INSTRUCTION_RE = re.compile(
    r"\b(?:cite|include|provide|show|return)\b.*\b(?:citations?|sources?|source\s+links?)\b.*$",
    re.IGNORECASE,
)
_FACET_LEADING_VERB_RE = re.compile(
    r"^\s*(?:answer|compose|create|design|draft|explain|generate|summari[sz]e|describe|write)\b\s*",
    re.IGNORECASE,
)


def _clean_facet_label(value: str) -> str:
    text = str(value or "").strip().strip("`'\"“”‘’[](){}")
    text = _FACET_INSTRUCTION_RE.sub("", text).strip()
    text = _FACET_LEADING_VERB_RE.sub("", text).strip()
    text = re.sub(r"^\b(?:a|an|the|and|or|plus|also)\b\s+", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s+", " ", text).strip(" ,.;:")
    return text


def _extract_query_facets(question: str, *, max_facets: int = 6) -> List[Dict[str, Any]]:
    text = str(question or "").strip()
    if not text:
        return []
    spans = [match.group("span") for match in _FACET_SPAN_RE.finditer(text)]
    candidate_text = spans[-1] if spans else text
    candidate_text = _FACET_INSTRUCTION_RE.sub("", candidate_text).strip()
    if not re.search(r"[,;/]|\b(?:and|or|plus)\b", candidate_text, flags=re.IGNORECASE):
        return []
    normalized = re.sub(r"\b(?:as\s+well\s+as|along\s+with|plus)\b", ",", candidate_text, flags=re.IGNORECASE)
    normalized = re.sub(r"\s*(?:,|;|/)\s*", ",", normalized)
    normalized = re.sub(r"\s+\b(?:and|or)\b\s+", ",", normalized, flags=re.IGNORECASE)
    labels: List[str] = []
    seen: set[str] = set()
    for raw_part in normalized.split(","):
        label = _clean_facet_label(raw_part)
        if not label:
            continue
        terms = _facet_terms(label)
        if not terms:
            continue
        if len(label.split()) > 8 or len(label) > 90:
            continue
        key = " ".join(terms)
        if key in seen:
            continue
        seen.add(key)
        labels.append(label)
        if len(labels) >= max(2, int(max_facets)):
            break
    if len(labels) < 2:
        return []
    return [
        {
            "facet_id": f"facet_{index + 1}",
            "label": label,
            "query": f"{question} {label}",
            "terms": _facet_terms(label),
        }
        for index, label in enumerate(labels)
    ]


def _doc_matches_facet(doc: Document, facet: Dict[str, Any]) -> bool:
    terms = [str(item).lower() for item in (facet.get("terms") or []) if str(item)]
    if not terms:
        return False
    metadata = dict(doc.metadata or {})
    haystack = _normalize_text(
        " ".join(
            [
                str(metadata.get("title") or ""),
                str(metadata.get("source_path") or ""),
                str(metadata.get("section_title") or ""),
                doc.page_content,
            ]
        )
    )
    label = _normalize_text(str(facet.get("label") or ""))
    if label and label in haystack:
        return True
    matches = [term for term in terms if term in haystack]
    threshold = 1 if len(terms) <= 2 else 2
    return len(matches) >= threshold


def _select_facet_aware_evidence_docs(
    question: str,
    graded: Sequence[GradedChunk],
    min_chunks: int,
    facets: Sequence[Dict[str, Any]] | None,
) -> List[Document]:
    selected: List[Document] = []
    selected_keys: set[str] = set()
    facet_list = [dict(item) for item in (facets or []) if isinstance(item, dict)]
    target = max(int(min_chunks), min(len(facet_list), 6) if facet_list else int(min_chunks))

    for facet in facet_list:
        matches = [
            item
            for item in graded
            if item.relevance >= 1 and _doc_matches_facet(item.doc, facet)
        ]
        matches.sort(
            key=lambda item: (
                item.relevance,
                _adaptive_score(item.doc),
                _title_overlap_score(question, item.doc),
            ),
            reverse=True,
        )
        if not matches:
            continue
        doc = matches[0].doc
        key = _chunk_id(doc) or f"{_doc_id(doc)}#{len(selected)}"
        if key in selected_keys:
            continue
        selected.append(doc)
        selected_keys.add(key)
        if len(selected) >= target:
            break

    for doc in _select_evidence_docs(question, graded, target):
        key = _chunk_id(doc) or f"{_doc_id(doc)}#{len(selected)}"
        if key in selected_keys:
            continue
        selected.append(doc)
        selected_keys.add(key)
        if len(selected) >= target:
            break
    return selected


def _phase_sequence(max_search_rounds: int) -> List[str]:
    phases = [
        "entity_discovery",
        "relationship_discovery",
        "source_confirmation",
        "synthesis_ready",
    ]
    return phases[: max(1, int(max_search_rounds))]


def _desired_evidence_budget(question: str, settings: Any) -> int:
    base = max(1, int(getattr(settings, "rag_min_evidence_chunks", 1)))
    if _is_discovery_query(question):
        return max(base + 2, 4)
    if _is_complex_query(question):
        return max(base + 1, 3)
    return base


def _doc_focus_from_documents(documents: Sequence[Document], *, limit: int = 6) -> List[Dict[str, str]]:
    seen: set[str] = set()
    docs: List[Dict[str, str]] = []
    for doc in documents or []:
        metadata = dict(doc.metadata or {})
        doc_id = str(metadata.get("doc_id") or "")
        key = doc_id or str(metadata.get("title") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        docs.append(
            {
                "doc_id": doc_id,
                "title": str(metadata.get("title") or ""),
                "source_path": str(metadata.get("source_path") or ""),
                "source_type": str(metadata.get("source_type") or ""),
            }
        )
        if len(docs) >= limit:
            break
    return docs


def _rewrite_query(judge_llm: Any, *, settings: Any, question: str, conversation_context: str, attempt: int, callbacks: List[Any]) -> str:
    try:
        prompt = render_template(
            load_judge_rewrite_prompt(settings),
            {
                "ATTEMPT": attempt,
                "QUESTION": question,
                "CONVERSATION_CONTEXT": conversation_context,
            },
        )
        response = judge_llm.invoke(prompt, config={"callbacks": callbacks})
        text = getattr(response, "content", None) or str(response)
        payload = extract_json(text) or {}
        rewritten = str(payload.get("rewritten_query") or "").strip()
        return rewritten or question
    except Exception:
        return question


def _matches_process_flow(doc: Document) -> bool:
    metadata = doc.metadata or {}
    haystack = " ".join(
        [
            str(metadata.get("chunk_type") or ""),
            str(metadata.get("section_title") or ""),
            _title(doc),
            doc.page_content[:400],
        ]
    )
    return bool(_PROCESS_FLOW_HINTS.search(haystack))


def _chunk_to_document(chunk: ChunkRecord, *, extra_metadata: Dict[str, Any] | None = None) -> Document:
    metadata = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "tenant_id": chunk.tenant_id,
        "collection_id": chunk.collection_id,
        "chunk_index": chunk.chunk_index,
        "chunk_type": chunk.chunk_type,
        "page": chunk.page_number,
        "clause_number": chunk.clause_number,
        "section_title": chunk.section_title,
        "sheet_name": chunk.sheet_name,
        "row_start": chunk.row_start,
        "row_end": chunk.row_end,
        "cell_range": chunk.cell_range,
    }
    metadata.update(dict(extra_metadata or {}))
    return Document(
        page_content=chunk.content,
        metadata=metadata,
    )


@dataclass
class SearchFilters:
    doc_ids: List[str] = field(default_factory=list)
    source_types: List[str] = field(default_factory=list)
    file_types: List[str] = field(default_factory=list)
    doc_structure_types: List[str] = field(default_factory=list)
    title_contains: str = ""
    collection_id: str = ""


@dataclass
class EvidenceEntry:
    chunk_id: str
    doc_id: str
    title: str
    query: str
    strategy: str
    rationale: str
    score: float
    round_index: int
    snippet: str
    relevance: int = 0
    coverage_state: str = "candidate"
    grade_reason: str = ""
    retrieval_lanes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceLedger:
    question: str
    entries: Dict[str, EvidenceEntry] = field(default_factory=dict)
    documents: Dict[str, Document] = field(default_factory=dict)
    facets: List[Dict[str, Any]] = field(default_factory=list)
    facet_coverage: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    round_summaries: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_subquestions: List[str] = field(default_factory=list)
    pruned_chunk_ids: List[str] = field(default_factory=list)
    claims: List[Dict[str, Any]] = field(default_factory=list)
    supported_claim_ids: List[str] = field(default_factory=list)
    unsupported_claim_ids: List[str] = field(default_factory=list)
    verified_hops: List[str] = field(default_factory=list)
    unverified_hops: List[str] = field(default_factory=list)
    source_confirmation_state: Dict[str, Any] = field(default_factory=dict)

    def add_documents(
        self,
        docs: Sequence[Document],
        *,
        score: float,
        query: str,
        strategy: str,
        rationale: str,
        round_index: int,
    ) -> int:
        added = 0
        for doc in docs:
            chunk_id = _chunk_id(doc)
            if not chunk_id:
                continue
            existing = self.entries.get(chunk_id)
            candidate = _with_adaptive_metadata(
                doc,
                score=score,
                strategy=strategy,
                query=query,
                rationale=rationale,
            )
            if existing is None:
                self.entries[chunk_id] = EvidenceEntry(
                    chunk_id=chunk_id,
                    doc_id=_doc_id(candidate),
                    title=_title(candidate),
                    query=query,
                    strategy=strategy,
                    rationale=rationale,
                    score=float(score),
                    round_index=round_index,
                    snippet=candidate.page_content[:260],
                    retrieval_lanes=dict((candidate.metadata or {}).get("_retrieval_lanes") or {}),
                )
                self.documents[chunk_id] = candidate
                added += 1
                continue
            if float(score) > existing.score:
                existing.score = float(score)
                existing.query = query
                existing.strategy = strategy
                existing.rationale = rationale
                existing.round_index = round_index
                existing.snippet = candidate.page_content[:260]
                existing.retrieval_lanes = dict((candidate.metadata or {}).get("_retrieval_lanes") or existing.retrieval_lanes or {})
                self.documents[chunk_id] = candidate
        return added

    def add_scored_chunks(
        self,
        chunks: Sequence[ScoredChunk],
        *,
        query: str,
        strategy: str,
        rationale: str,
        round_index: int,
    ) -> int:
        added = 0
        for chunk in chunks:
            added += self.add_documents(
                [
                    _with_adaptive_metadata(
                        chunk.doc,
                        score=chunk.score,
                        strategy=strategy,
                        query=query,
                        rationale=rationale,
                    )
                ],
                score=chunk.score if chunk.score > 0 else 0.01,
                query=query,
                strategy=strategy,
                rationale=rationale,
                round_index=round_index,
            )
        return added

    def apply_grades(self, graded: Sequence[GradedChunk]) -> None:
        for item in graded:
            entry = self.entries.get(_chunk_id(item.doc))
            if entry is None:
                continue
            entry.relevance = int(item.relevance)
            entry.grade_reason = str(item.reason or "")
            if item.relevance >= 2:
                entry.coverage_state = "strong"
            elif item.relevance == 1:
                entry.coverage_state = "supporting"
            else:
                entry.coverage_state = "weak"
            self.documents[entry.chunk_id] = _with_adaptive_metadata(
                item.doc,
                score=max(entry.score, float(item.relevance)),
                strategy=entry.strategy,
                query=entry.query,
                rationale=entry.rationale,
            )

    def materialize_documents(self, *, max_chunks: int = 18) -> List[Document]:
        docs = [
            self.documents[chunk_id]
            for chunk_id in self.entries
            if chunk_id in self.documents
        ]
        docs.sort(
            key=lambda doc: (
                int(self.entries.get(_chunk_id(doc)).relevance if self.entries.get(_chunk_id(doc)) else 0),
                _adaptive_score(doc),
                _title_overlap_score(self.question, doc),
            ),
            reverse=True,
        )
        return docs[:max_chunks]

    def best_doc_ids(self, *, limit: int = 6) -> List[str]:
        scores: Dict[str, float] = {}
        for entry in self.entries.values():
            if not entry.doc_id:
                continue
            scores[entry.doc_id] = max(scores.get(entry.doc_id, 0.0), entry.score + (entry.relevance * 0.15))
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:limit]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "entries": [
                {
                    "chunk_id": entry.chunk_id,
                    "doc_id": entry.doc_id,
                    "title": entry.title,
                    "query": entry.query,
                    "strategy": entry.strategy,
                    "rationale": entry.rationale,
                    "score": round(entry.score, 4),
                    "relevance": entry.relevance,
                    "coverage_state": entry.coverage_state,
                    "grade_reason": entry.grade_reason,
                    "retrieval_lanes": dict(entry.retrieval_lanes or {}),
                    "round_index": entry.round_index,
                    "snippet": entry.snippet,
                }
                for entry in sorted(self.entries.values(), key=lambda item: (item.round_index, -item.score))
            ],
            "facets": [dict(item) for item in self.facets],
            "facet_coverage": {str(key): dict(value) for key, value in self.facet_coverage.items()},
            "round_summaries": list(self.round_summaries),
            "unresolved_subquestions": list(self.unresolved_subquestions),
            "pruned_chunk_ids": list(self.pruned_chunk_ids),
            "claims": [dict(item) for item in self.claims],
            "supported_claim_ids": list(self.supported_claim_ids),
            "unsupported_claim_ids": list(self.unsupported_claim_ids),
            "verified_hops": list(self.verified_hops),
            "unverified_hops": list(self.unverified_hops),
            "source_confirmation_state": dict(self.source_confirmation_state),
        }


@dataclass
class RetrievalRun:
    selected_docs: List[Document]
    candidate_docs: List[Document]
    graded: List[GradedChunk]
    query_used: str
    search_mode: str
    rounds: int
    tool_calls_used: int
    tool_call_log: List[str]
    strategies_used: List[str]
    candidate_counts: Dict[str, int]
    evidence_ledger: Dict[str, Any] = field(default_factory=dict)
    parallel_workers_used: bool = False
    source_plan: Dict[str, Any] = field(default_factory=dict)
    sources_used: List[str] = field(default_factory=list)
    graphs_considered: List[str] = field(default_factory=list)
    graph_methods_used: List[str] = field(default_factory=list)
    sql_sources_used: List[str] = field(default_factory=list)
    resolution_stats: Dict[str, int] = field(default_factory=dict)
    decomposition: Dict[str, Any] = field(default_factory=dict)
    claim_ledger: Dict[str, Any] = field(default_factory=dict)
    verified_hops: List[str] = field(default_factory=list)
    retrieval_verification: Dict[str, Any] = field(default_factory=dict)
    stage_timings_ms: Dict[str, float] = field(default_factory=dict)
    budget_ms: int = 0
    budget_exhausted: bool = False
    slow_stages: List[str] = field(default_factory=list)

    def to_summary(self, *, citations_found: int) -> RetrievalSummary:
        return RetrievalSummary(
            query_used=self.query_used,
            steps=max(self.rounds + 2, 1),
            tool_calls_used=self.tool_calls_used,
            tool_call_log=list(self.tool_call_log),
            citations_found=int(citations_found),
            search_mode=self.search_mode,
            rounds=self.rounds,
            strategies_used=list(self.strategies_used),
            candidate_counts=dict(self.candidate_counts),
            parallel_workers_used=self.parallel_workers_used,
            sources_used=list(self.sources_used),
            source_plan=dict(self.source_plan),
            graphs_considered=list(self.graphs_considered),
            graph_methods_used=list(self.graph_methods_used),
            sql_sources_used=list(self.sql_sources_used),
            resolution_stats=dict(self.resolution_stats),
            decomposition=dict(self.decomposition),
            claim_ledger=dict(self.claim_ledger),
            verified_hops=list(self.verified_hops),
            retrieval_verification=dict(self.retrieval_verification),
            stage_timings_ms=dict(self.stage_timings_ms),
            budget_ms=int(self.budget_ms),
            budget_exhausted=bool(self.budget_exhausted),
            slow_stages=list(self.slow_stages),
        )


class CorpusRetrievalAdapter:
    def __init__(
        self,
        stores: Any,
        *,
        settings: Any,
        session: Any,
        source_plan: SourcePlan | None = None,
        stage_timer: RetrievalBudget | None = None,
    ) -> None:
        self.stores = stores
        self.settings = settings
        self.session = session
        self.source_plan = source_plan or _new_source_plan("")
        self.tenant_id = getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev"))
        self._read_cache: Dict[tuple[str, str, str, int, int], List[Document]] = {}
        self.stage_timer = stage_timer

    def _document_metadata(self, doc_id: str) -> Dict[str, Any]:
        try:
            record = self.stores.doc_store.get_document(doc_id, self.tenant_id)
        except Exception:
            record = None
        if record is None:
            return {}
        return {
            "title": str(getattr(record, "title", "") or ""),
            "source_type": str(getattr(record, "source_type", "") or ""),
            "source_path": str(getattr(record, "source_path", "") or ""),
            "file_type": str(getattr(record, "file_type", "") or ""),
            "doc_structure_type": str(getattr(record, "doc_structure_type", "") or ""),
        }

    def _graph_store(self) -> Any | None:
        store = getattr(self.stores, "graph_store", None)
        if store is None or not bool(getattr(self.settings, "graph_search_enabled", False)):
            return None
        if hasattr(store, "available") and not bool(getattr(store, "available")):
            return None
        return store

    def _graph_service(self) -> Any | None:
        if not bool(getattr(self.settings, "graph_search_enabled", False)):
            return None
        if getattr(self.stores, "graph_index_store", None) is None:
            return None
        try:
            from agentic_chatbot_next.graph.service import GraphService

            return GraphService(self.settings, self.stores, session=self.session)
        except Exception:
            return None

    def _graph_capable_query(self, query: str) -> bool:
        lower = str(query or "").lower()
        return bool(
            re.search(
                r"\b("
                r"knowledge\s+graph|graph|entity\s+neighbou?rhood|relationship|relationships|"
                r"depends\s+on|dependency|dependencies|connected\s+to|connects?|path\s+between|"
                r"requires?\s+.+\s+approval|approval\s+(?:from|by|dependency)|references?|cites?|"
                r"source\s+list|graph\s+sources?|graph\s+inventory|cross-document"
                r")\b",
                lower,
            )
        )

    def _graph_methods(self) -> List[str]:
        methods = [str(item) for item in (self.source_plan.graph_methods or []) if str(item)]
        if methods:
            return methods
        default_method = str(getattr(self.settings, "graphrag_default_query_method", "local") or "local")
        return [default_method]

    def _graph_ids(self) -> List[str]:
        return [str(item) for item in (self.source_plan.graph_ids or []) if str(item)]

    def _should_consult_graph(self, query: str) -> bool:
        explicit_graph_scope = bool(self._graph_ids() or self.source_plan.graph_shortlist)
        if explicit_graph_scope and self._graph_capable_query(query):
            return True
        return self._graph_capable_query(query)

    def _resolve_graph_hits(self, hits: Sequence[Any]) -> List[ScoredChunk]:
        resolved: List[ScoredChunk] = []
        seen_chunk_ids: set[str] = set()
        for hit in hits or []:
            chunk_ids = [str(item) for item in (getattr(hit, "chunk_ids", None) or getattr(hit, "chunk_ids", []) or []) if str(item)]
            for chunk_id in chunk_ids[:8]:
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                try:
                    chunk = self.stores.chunk_store.get_chunk_by_id(chunk_id, self.tenant_id)
                except Exception:
                    chunk = None
                if chunk is None:
                    continue
                metadata = self._document_metadata(chunk.doc_id)
                doc = _chunk_to_document(chunk, extra_metadata=metadata)
                resolved.append(
                    ScoredChunk(
                        doc=doc,
                        score=max(float(getattr(hit, "score", 0.0) or 0.0), 0.2),
                        method="graph",
                    )
                )
            if resolved:
                continue
            doc_id = str(getattr(hit, "doc_id", "") or "")
            if not doc_id:
                continue
            fallback_query = str(getattr(hit, "summary", "") or doc_id)
            try:
                resolved.extend(
                    self.stores.chunk_store.vector_search(
                        fallback_query,
                        top_k=2,
                        doc_id_filter=doc_id,
                        tenant_id=self.tenant_id,
                    )
                )
            except Exception:
                pass
            try:
                resolved.extend(
                    self.stores.chunk_store.keyword_search(
                        fallback_query,
                        top_k=2,
                        doc_id_filter=doc_id,
                        tenant_id=self.tenant_id,
                    )
                )
            except Exception:
                pass
        return resolved

    def _matching_doc_ids(self, filters: SearchFilters | None) -> List[str]:
        if filters is None:
            return []
        doc_id_filter = {item for item in filters.doc_ids if item}
        if not any(
            [
                filters.source_types,
                filters.file_types,
                filters.doc_structure_types,
                filters.title_contains,
            ]
        ):
            return sorted(doc_id_filter)

        collection_ids = _resolve_collection_ids(
            self.settings,
            self.session,
            explicit_collection_id=filters.collection_id,
        )
        try:
            records = []
            if collection_ids:
                for collection_id in collection_ids:
                    records.extend(
                        self.stores.doc_store.list_documents(
                            tenant_id=self.tenant_id,
                            collection_id=collection_id,
                        )
                    )
            else:
                records = self.stores.doc_store.list_documents(tenant_id=self.tenant_id)
        except Exception:
            return sorted(doc_id_filter)

        matched: set[str] = set()
        title_contains = filters.title_contains.lower().strip()
        for record in records:
            if filters.source_types and record.source_type not in set(filters.source_types):
                continue
            if filters.file_types and record.file_type not in set(filters.file_types):
                continue
            if filters.doc_structure_types and record.doc_structure_type not in set(filters.doc_structure_types):
                continue
            if title_contains and title_contains not in record.title.lower():
                continue
            matched.add(str(record.doc_id))
        if doc_id_filter:
            matched &= doc_id_filter
        return sorted(matched)

    def _rerank_with_query_heuristics(self, query: str, chunks: Sequence[ScoredChunk]) -> List[ScoredChunk]:
        normalized_query = _normalize_text(query)
        query_tokens = _query_terms(query)
        wants_latest = bool(_LATEST_HINTS.search(query))
        wants_draft = bool(_DRAFT_HINTS.search(query))
        workbook_focus = bool(_WORKBOOK_FOCUS_HINTS.search(query))

        boosted: List[ScoredChunk] = []
        for chunk in chunks:
            metadata = dict(chunk.doc.metadata or {})
            haystack = " ".join(
                [
                    str(metadata.get("title") or ""),
                    str(metadata.get("source_path") or ""),
                    str(metadata.get("section_title") or ""),
                    str(metadata.get("sheet_name") or ""),
                    str(metadata.get("cell_range") or ""),
                    str(metadata.get("file_type") or ""),
                    str(metadata.get("doc_structure_type") or ""),
                    chunk.doc.page_content[:500],
                ]
            )
            normalized_haystack = _normalize_text(haystack)
            haystack_tokens = _query_terms(haystack)
            score = float(chunk.score)

            if workbook_focus:
                if str(metadata.get("file_type") or "").lower() in {"xlsx", "xls"}:
                    score += 0.16
                if query_tokens & _query_terms(str(metadata.get("sheet_name") or "")):
                    score += 0.12

            if wants_latest:
                if any(token in normalized_haystack for token in ("final", "approved", "authoritative", "current")):
                    score += 0.12
                if re.search(r"\brev[\s._-]*[bcdef]\b", normalized_haystack):
                    score += 0.08
                if "draft" in normalized_haystack:
                    score -= 0.12
            elif wants_draft and "draft" in normalized_haystack:
                score += 0.14

            if query_tokens and haystack_tokens:
                overlap = len(query_tokens & haystack_tokens)
                if overlap >= 3:
                    score += min(0.12, overlap * 0.02)

            boosted.append(ScoredChunk(doc=chunk.doc, score=score, method=chunk.method))
        return merge_dedupe(boosted)

    def _structured_neighbor_chunks(
        self,
        chunks: Sequence[ScoredChunk],
        *,
        max_seed_chunks: int = 8,
    ) -> List[ScoredChunk]:
        expanded: List[ScoredChunk] = []
        seen = {_chunk_id(chunk.doc) for chunk in chunks if _chunk_id(chunk.doc)}
        structured_file_types = {"xlsx", "xls", "pdf", "docx", "doc"}
        for chunk in list(chunks)[: max(1, int(max_seed_chunks))]:
            metadata = dict(chunk.doc.metadata or {})
            file_type = str(metadata.get("file_type") or "").strip().lower()
            if file_type not in structured_file_types:
                continue
            doc_id = _doc_id(chunk.doc)
            try:
                chunk_index = int(metadata.get("chunk_index"))
            except Exception:
                continue
            if not doc_id:
                continue
            window = 2 if file_type in {"xlsx", "xls"} else 1
            for neighbor in self._read_chunk_range(doc_id, chunk_index - window, chunk_index + window):
                neighbor_id = _chunk_id(neighbor)
                if not neighbor_id or neighbor_id in seen:
                    continue
                seen.add(neighbor_id)
                try:
                    neighbor_index = int((neighbor.metadata or {}).get("chunk_index"))
                except Exception:
                    neighbor_index = chunk_index
                distance = abs(neighbor_index - chunk_index)
                expanded.append(
                    ScoredChunk(
                        doc=neighbor,
                        score=max(0.01, float(chunk.score) - (0.02 * max(1, distance))),
                        method="structured_neighbor",
                    )
                )
        return expanded

    def search_corpus(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        exclude_chunk_ids: Iterable[str] | None = None,
        strategy: str = "hybrid",
        limit: int = 24,
        top_k_vector: int = 0,
        top_k_keyword: int = 0,
        preferred_doc_ids: Sequence[str] | None = None,
        must_include_uploads: bool = False,
    ) -> List[ScoredChunk]:
        strategy = str(strategy or "hybrid").strip().lower()
        query = _augment_query_with_aliases(query)
        top_k_vector = max(1, int(top_k_vector or getattr(self.settings, "rag_top_k_vector", 15)))
        top_k_keyword = max(1, int(top_k_keyword or getattr(self.settings, "rag_top_k_keyword", 15)))
        collection_id = ""
        if filters is not None:
            collection_id = filters.collection_id
        collection_ids = _resolve_collection_ids(
            self.settings,
            self.session,
            explicit_collection_id=collection_id,
        )

        excluded = {str(item) for item in (exclude_chunk_ids or []) if str(item)}
        allowed_doc_ids = self._matching_doc_ids(filters)
        if preferred_doc_ids:
            preferred = {str(item) for item in preferred_doc_ids if str(item)}
            allowed_doc_ids = sorted(set(allowed_doc_ids) & preferred) if allowed_doc_ids else sorted(preferred)

        vector_results: List[ScoredChunk] = []
        keyword_results: List[ScoredChunk] = []
        fanout_limit = 10
        if allowed_doc_ids and len(allowed_doc_ids) <= fanout_limit:
            if strategy in {"vector", "hybrid"}:
                with _measure_stage(self.stage_timer, "vector_search"):
                    for doc_id in allowed_doc_ids:
                        vector_results.extend(
                            self.stores.chunk_store.vector_search(
                                query,
                                top_k=max(2, min(6, top_k_vector)),
                                doc_id_filter=doc_id,
                                tenant_id=self.tenant_id,
                            )
                        )
            if strategy in {"keyword", "hybrid"}:
                with _measure_stage(self.stage_timer, "bm25_search"):
                    for doc_id in allowed_doc_ids:
                        keyword_results.extend(
                            self.stores.chunk_store.keyword_search(
                                query,
                                top_k=max(2, min(6, top_k_keyword)),
                                doc_id_filter=doc_id,
                                tenant_id=self.tenant_id,
                            )
                        )
        elif strategy == "hybrid":
            retrieval = retrieve_candidates(
                self.stores,
                query,
                tenant_id=self.tenant_id,
                preferred_doc_ids=allowed_doc_ids or list(preferred_doc_ids or []),
                must_include_uploads=must_include_uploads,
                top_k_vector=top_k_vector,
                top_k_keyword=top_k_keyword,
                collection_id_filter=collection_id,
                collection_ids_filter=collection_ids,
                stage_timer=self.stage_timer,
            )
            vector_results = list(retrieval.get("vector") or [])
            keyword_results = list(retrieval.get("keyword") or [])
        else:
            active_collection_ids = collection_ids or ([collection_id] if collection_id else [""])
            if strategy in {"vector", "hybrid"}:
                with _measure_stage(self.stage_timer, "vector_search"):
                    for active_collection_id in active_collection_ids:
                        vector_results.extend(
                            self.stores.chunk_store.vector_search(
                                query,
                                top_k=top_k_vector,
                                collection_id_filter=active_collection_id or None,
                                tenant_id=self.tenant_id,
                            )
                        )
            if strategy in {"keyword", "hybrid"}:
                with _measure_stage(self.stage_timer, "bm25_search"):
                    for active_collection_id in active_collection_ids:
                        keyword_results.extend(
                            self.stores.chunk_store.keyword_search(
                                query,
                                top_k=top_k_keyword,
                                collection_id_filter=active_collection_id or None,
                                tenant_id=self.tenant_id,
                            )
                        )

        merged = rank_fuse_dedupe(
            {
                "vector": vector_results,
                "keyword": keyword_results,
            }
        )
        if allowed_doc_ids:
            allowed = set(allowed_doc_ids)
            merged = [item for item in merged if _doc_id(item.doc) in allowed]
        if self._should_consult_graph(query):
            graph_service = self._graph_service()
            scoped_doc_ids = allowed_doc_ids or list(preferred_doc_ids or [])
            graph_hits: List[Any] = []
            with _measure_stage(self.stage_timer, "graph_search"):
                if graph_service is not None:
                    try:
                        graph_payload = graph_service.query_across_graphs(
                            query,
                            collection_id=collection_id,
                            graph_ids=self._graph_ids(),
                            methods=self._graph_methods(),
                            limit=max(4, int(limit)),
                            top_k_graphs=max(1, len(self._graph_ids()) or 3),
                            doc_ids=scoped_doc_ids,
                        )
                        graph_hits = [item for item in (graph_payload.get("results") or []) if isinstance(item, dict)]
                    except Exception:
                        graph_hits = []
            if graph_hits:
                resolved_hits = [
                    type(
                        "GraphCandidate",
                        (),
                        {
                            "doc_id": str(item.get("doc_id") or ""),
                            "chunk_ids": [str(entry) for entry in (item.get("chunk_ids") or []) if str(entry)],
                            "score": float(item.get("score") or 0.0),
                            "title": str(item.get("title") or ""),
                            "source_path": str(item.get("source_path") or ""),
                            "source_type": str(item.get("source_type") or ""),
                            "relationship_path": [str(entry) for entry in (item.get("relationship_path") or []) if str(entry)],
                            "summary": str(item.get("summary") or ""),
                            "metadata": dict(item.get("metadata") or {}),
                        },
                    )()
                    for item in graph_hits
                ]
                graph_chunks = self._resolve_graph_hits(resolved_hits)
                merged = rank_fuse_dedupe({"text": merged, "graph": graph_chunks})
            elif graph_service is None or str(getattr(self.settings, "graph_backend", "microsoft_graphrag") or "microsoft_graphrag").strip().lower() == "neo4j":
                graph_store = self._graph_store()
                if graph_store is not None and self._graph_capable_query(query):
                    with _measure_stage(self.stage_timer, "graph_search"):
                        try:
                            store_hits = graph_store.local_search(
                                query,
                                tenant_id=self.tenant_id,
                                limit=max(4, int(limit)),
                                doc_ids=scoped_doc_ids,
                            )
                            if len(store_hits) < 2:
                                store_hits.extend(
                                    graph_store.global_search(
                                        query,
                                        tenant_id=self.tenant_id,
                                        limit=max(4, int(limit)),
                                        doc_ids=scoped_doc_ids,
                                    )
                                )
                            graph_chunks = self._resolve_graph_hits(store_hits)
                            merged = rank_fuse_dedupe({"text": merged, "graph": graph_chunks})
                        except Exception:
                            pass
        if excluded:
            merged = [item for item in merged if _chunk_id(item.doc) not in excluded]
        with _measure_stage(self.stage_timer, "neighbor_expansion"):
            neighbor_chunks = self._structured_neighbor_chunks(
                merged,
                max_seed_chunks=min(8, max(1, int(limit))),
            )
        if neighbor_chunks:
            merged = rank_fuse_dedupe({"retrieval": merged, "structured_neighbor": neighbor_chunks})
        merged = self._rerank_with_query_heuristics(query, merged)
        return merged[: max(1, int(limit))]

    def grep_corpus(
        self,
        pattern: str,
        *,
        filters: SearchFilters | None = None,
        exclude_chunk_ids: Iterable[str] | None = None,
        limit: int = 20,
    ) -> List[ScoredChunk]:
        return self.search_corpus(
            pattern,
            filters=filters,
            exclude_chunk_ids=exclude_chunk_ids,
            strategy="keyword",
            limit=limit,
            top_k_keyword=max(4, int(getattr(self.settings, "rag_top_k_keyword", 15))),
        )

    def document_chunk_count(self, doc_id: str) -> int:
        try:
            return max(0, int(self.stores.chunk_store.chunk_count(doc_id=doc_id, tenant_id=self.tenant_id)))
        except Exception:
            return 0

    def _read_chunk_range(self, doc_id: str, start_index: int, end_index: int) -> List[Document]:
        if end_index < start_index:
            return []
        raw = self.stores.chunk_store.get_chunks_by_index_range(
            doc_id,
            max(0, int(start_index)),
            max(0, int(end_index)),
            self.tenant_id,
        )
        extra = self._document_metadata(doc_id)
        return [_chunk_to_document(chunk, extra_metadata=extra) for chunk in raw]

    def outline_scan(self, doc_id: str, *, max_chunks: int = 8) -> List[Document]:
        count = self.document_chunk_count(doc_id)
        if count <= 0:
            return []
        sample_points = sorted(
            {
                0,
                max(0, count // 4),
                max(0, count // 2),
                max(0, (count * 3) // 4),
                max(0, count - 1),
            }
        )
        docs: List[Document] = []
        seen: set[str] = set()
        for index in sample_points:
            for doc in self._read_chunk_range(doc_id, index, index):
                chunk_id = _chunk_id(doc)
                if not chunk_id or chunk_id in seen:
                    continue
                seen.add(chunk_id)
                docs.append(doc)
                if len(docs) >= max(1, int(max_chunks)):
                    return docs
        return docs

    def _sample_full_document(self, doc_id: str, *, max_chunks: int, total_chunks: int) -> List[Document]:
        if total_chunks <= 0 or max_chunks <= 0:
            return []
        if total_chunks <= max_chunks:
            docs: List[Document] = []
            page_size = 6
            for start in range(0, total_chunks, page_size):
                docs.extend(
                    self._read_chunk_range(
                        doc_id,
                        start,
                        min(total_chunks - 1, start + page_size - 1),
                    )
                )
            return docs

        sample_indexes = sorted(
            {
                min(total_chunks - 1, (index * total_chunks) // max_chunks)
                for index in range(max_chunks)
            }
        )
        docs: List[Document] = []
        seen: set[str] = set()
        for index in sample_indexes:
            for doc in self._read_chunk_range(doc_id, index, index):
                chunk_id = _chunk_id(doc)
                if not chunk_id or chunk_id in seen:
                    continue
                seen.add(chunk_id)
                docs.append(doc)
        return docs

    def read_document(
        self,
        doc_id: str,
        *,
        focus: str = "",
        max_chunks: int = 6,
        read_depth: str = "focused",
        full_read_chunk_threshold: int = 24,
    ) -> List[Document]:
        focus_query = focus.strip() or "document overview key topics summary"
        normalized_depth = str(read_depth or "focused").strip().lower()
        cache_key = (
            str(doc_id),
            focus_query,
            normalized_depth,
            max(1, int(max_chunks)),
            max(1, int(full_read_chunk_threshold or 24)),
        )
        cached = self._read_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        with _measure_stage(self.stage_timer, "document_reads"):
            docs: List[Document] = []
            count = self.document_chunk_count(doc_id)
            if normalized_depth == "outline":
                docs = self.outline_scan(doc_id, max_chunks=max_chunks)
            elif normalized_depth == "full":
                effective_limit = min(
                    count,
                    max(
                        max(1, int(max_chunks)),
                        max(1, int(full_read_chunk_threshold or 24)),
                    ),
                )
                docs = self._sample_full_document(
                    doc_id,
                    max_chunks=effective_limit,
                    total_chunks=count,
                )
            else:
                results = self.search_corpus(
                    focus_query,
                    filters=SearchFilters(doc_ids=[doc_id]),
                    strategy="hybrid",
                    limit=max_chunks,
                    preferred_doc_ids=[doc_id],
                )
                docs = [item.doc for item in results]
                if not docs and count > 0:
                    docs = self._read_chunk_range(
                        doc_id,
                        0,
                        max(0, min(count, max_chunks) - 1),
                    )

        self._read_cache[cache_key] = list(docs)
        return list(docs)

    def fetch_chunk_window(self, chunk_id: str, *, before: int = 1, after: int = 1) -> List[Document]:
        try:
            chunk = self.stores.chunk_store.get_chunk_by_id(chunk_id, self.tenant_id)
        except Exception:
            chunk = None
        if chunk is None:
            return []
        neighbours = self.stores.chunk_store.get_chunks_by_index_range(
            chunk.doc_id,
            max(0, int(chunk.chunk_index) - max(0, int(before))),
            int(chunk.chunk_index) + max(0, int(after)),
            self.tenant_id,
        )
        extra = self._document_metadata(chunk.doc_id)
        return [_chunk_to_document(item, extra_metadata=extra) for item in neighbours]

    def section_candidates(
        self,
        doc_id: str,
        *,
        query: str,
        prioritized_sections: Sequence[Dict[str, Any]] | None = None,
        limit: int = 6,
    ) -> List[Document]:
        chunk_store = getattr(self.stores, "chunk_store", None)
        if chunk_store is None:
            return []
        clause_numbers = [
            str(item.get("value") or "")
            for item in (prioritized_sections or [])
            if str(item.get("match_type") or "") == "clause_number" and str(item.get("value") or "")
        ]
        sheet_names = [
            str(item.get("value") or "")
            for item in (prioritized_sections or [])
            if str(item.get("match_type") or "") == "sheet_name" and str(item.get("value") or "")
        ]
        try:
            rows = chunk_store.search_sections(
                doc_id,
                tenant_id=self.tenant_id,
                section_query=str(query or ""),
                clause_numbers=clause_numbers,
                sheet_names=sheet_names,
                limit=limit,
            )
        except Exception:
            rows = []
        extra = self._document_metadata(doc_id)
        return [_chunk_to_document(item, extra_metadata=extra) for item in rows]

    def search_section_scope(
        self,
        query: str,
        *,
        doc_ids: Sequence[str],
        prioritized_sections: Sequence[Dict[str, Any]] | None = None,
        limit: int = 8,
    ) -> List[Document]:
        results: List[Document] = []
        seen: set[str] = set()
        for doc_id in [str(item) for item in doc_ids if str(item)]:
            for doc in self.section_candidates(
                doc_id,
                query=query,
                prioritized_sections=prioritized_sections,
                limit=limit,
            ):
                chunk_id = _chunk_id(doc)
                if not chunk_id or chunk_id in seen:
                    continue
                seen.add(chunk_id)
                results.append(doc)
                if len(results) >= max(1, int(limit)):
                    return results
        return results

    def prune_chunks(
        self,
        docs: Sequence[Document],
        *,
        keep: int = 18,
        max_per_doc: int = 3,
    ) -> tuple[List[Document], List[str]]:
        by_chunk: Dict[str, Document] = {}
        for doc in docs:
            chunk_id = _chunk_id(doc)
            if not chunk_id:
                continue
            existing = by_chunk.get(chunk_id)
            if existing is None or _adaptive_score(doc) > _adaptive_score(existing):
                by_chunk[chunk_id] = doc

        ordered = sorted(
            by_chunk.values(),
            key=lambda doc: (_adaptive_score(doc), _title_overlap_score("", doc)),
            reverse=True,
        )
        kept: List[Document] = []
        pruned: List[str] = []
        per_doc_counts: Dict[str, int] = {}
        for doc in ordered:
            doc_id = _doc_id(doc)
            if len(kept) >= keep or per_doc_counts.get(doc_id, 0) >= max_per_doc:
                pruned.append(_chunk_id(doc))
                continue
            kept.append(doc)
            per_doc_counts[doc_id] = per_doc_counts.get(doc_id, 0) + 1
        return kept, pruned


def _fast_candidate_counts(
    retrieval: Dict[str, Any],
    graded: Sequence[GradedChunk],
    selected_docs: Sequence[Document],
    *,
    graph_hits: int = 0,
    sql_doc_hints: int = 0,
) -> Dict[str, int]:
    return {
        "vector_hits": len(retrieval.get("vector") or []),
        "keyword_hits": len(retrieval.get("keyword") or []),
        "merged_hits": len(retrieval.get("merged") or []),
        "graded_chunks": len(graded),
        "selected_docs": len(selected_docs),
        "strong_chunks": sum(1 for item in graded if item.relevance >= 2),
        "unique_docs": len({_doc_id(item.doc) for item in graded if _doc_id(item.doc)}),
        "graph_hits": int(graph_hits),
        "sql_doc_hints": int(sql_doc_hints),
    }


def _build_decomposition(
    *,
    query: str,
    settings: Any,
    stores: Any,
    session: Any,
    controller_hints: Dict[str, Any],
    source_plan: SourcePlan,
) -> Dict[str, Any]:
    canonical_entities = resolve_query_entities(
        query=query,
        stores=stores,
        settings=settings,
        session=session,
        collection_id=_resolve_collection_id(settings, session),
        controller_hints=controller_hints,
    )
    if not canonical_entities:
        canonical_entities = list(source_plan.decomposition.get("canonical_entities") or [])
    query_facets = _extract_query_facets(query)
    relationship_questions = [str(query or "")]
    if len(canonical_entities) >= 2:
        relationship_questions.extend(
            [
                f"What connects {canonical_entities[index]['canonical_name']} and {canonical_entities[index + 1]['canonical_name']}?"
                for index in range(0, min(len(canonical_entities) - 1, 3))
            ]
        )
    source_confirmation_questions = [str(query or "")]
    for facet in query_facets[:6]:
        source_confirmation_questions.append(str(facet.get("query") or facet.get("label") or query))
    for entity in canonical_entities[:4]:
        source_confirmation_questions.append(
            f"Find grounded evidence mentioning {entity.get('canonical_name') or entity.get('matched_alias') or ''}."
        )
    claim_checklist = list(source_plan.claim_checklist or [])
    if not claim_checklist:
        claim_checklist = [
            {
                "claim_id": f"claim_{index + 1}",
                "question": str(query or ""),
                "entity": str(entity.get("canonical_name") or entity.get("matched_alias") or ""),
                "priority": "high" if index == 0 else "medium",
            }
            for index, entity in enumerate(canonical_entities[:4])
        ]
    existing_claim_questions = {str(item.get("question") or "").casefold() for item in claim_checklist if isinstance(item, dict)}
    for facet in query_facets:
        facet_query = str(facet.get("query") or facet.get("label") or query)
        if facet_query.casefold() in existing_claim_questions:
            continue
        claim_checklist.append(
            {
                "claim_id": str(facet.get("facet_id") or f"facet_{len(claim_checklist) + 1}"),
                "question": facet_query,
                "entity": str(facet.get("label") or ""),
                "priority": "high",
            }
        )
        existing_claim_questions.add(facet_query.casefold())
    return {
        "canonical_entities": canonical_entities,
        "relationship_questions": relationship_questions[:6],
        "source_confirmation_questions": source_confirmation_questions[:6],
        "query_facets": query_facets,
        "answer_goal": str(query or ""),
        "preferred_sources": list(source_plan.sources_chosen),
        "required_hops": list(source_plan.required_hops or []),
        "claim_checklist": claim_checklist,
        "prioritized_sections": list(source_plan.prioritized_sections or []),
        "phases": _phase_sequence(4),
    }


def _initialize_claim_ledger(ledger: EvidenceLedger, decomposition: Dict[str, Any]) -> None:
    ledger.claims = [dict(item) for item in (decomposition.get("claim_checklist") or []) if isinstance(item, dict)]
    ledger.facets = [dict(item) for item in (decomposition.get("query_facets") or []) if isinstance(item, dict)]
    ledger.facet_coverage = {
        str(facet.get("facet_id") or f"facet_{index + 1}"): {
            "label": str(facet.get("label") or ""),
            "query": str(facet.get("query") or ""),
            "state": "missing",
            "supporting_chunk_ids": [],
            "supporting_doc_ids": [],
        }
        for index, facet in enumerate(ledger.facets)
    }
    ledger.supported_claim_ids = []
    ledger.unsupported_claim_ids = []
    ledger.verified_hops = []
    ledger.unverified_hops = [str(item) for item in (decomposition.get("required_hops") or []) if str(item)]
    ledger.source_confirmation_state = {
        "phase": "entity_discovery",
        "prioritized_sections": [dict(item) for item in (decomposition.get("prioritized_sections") or []) if isinstance(item, dict)],
    }


def _refresh_facet_coverage(ledger: EvidenceLedger, docs: Sequence[Document]) -> None:
    if not ledger.facets:
        return
    coverage: Dict[str, Dict[str, Any]] = {}
    for index, facet in enumerate(ledger.facets):
        facet_id = str(facet.get("facet_id") or f"facet_{index + 1}")
        supporting_chunks: List[str] = []
        supporting_docs: List[str] = []
        for doc in docs:
            if not _doc_matches_facet(doc, facet):
                continue
            chunk_id = _chunk_id(doc)
            doc_id = _doc_id(doc)
            if chunk_id and chunk_id not in supporting_chunks:
                supporting_chunks.append(chunk_id)
            if doc_id and doc_id not in supporting_docs:
                supporting_docs.append(doc_id)
        coverage[facet_id] = {
            "label": str(facet.get("label") or ""),
            "query": str(facet.get("query") or ""),
            "state": "covered" if supporting_chunks or supporting_docs else "missing",
            "supporting_chunk_ids": supporting_chunks[:4],
            "supporting_doc_ids": supporting_docs[:4],
        }
    ledger.facet_coverage = coverage


def _refresh_claim_ledger(
    ledger: EvidenceLedger,
    *,
    decomposition: Dict[str, Any],
    strong_docs: Sequence[Document],
    phase: str,
) -> None:
    strong_text = " ".join(doc.page_content for doc in strong_docs[:8]).lower()
    supported: List[str] = []
    unsupported: List[str] = []
    for claim in ledger.claims:
        claim_id = str(claim.get("claim_id") or "")
        if not claim_id:
            continue
        entity_name = str(claim.get("entity") or "").lower()
        question_terms = _terms_from_text(str(claim.get("question") or ""))
        matched = False
        if entity_name and entity_name in strong_text:
            matched = True
        elif question_terms and len([term for term in question_terms[:4] if term in strong_text]) >= 2:
            matched = True
        if matched:
            supported.append(claim_id)
        else:
            unsupported.append(claim_id)
    ledger.supported_claim_ids = supported
    ledger.unsupported_claim_ids = unsupported
    _refresh_facet_coverage(ledger, strong_docs)

    verified_hops: List[str] = []
    unverified_hops: List[str] = []
    for hop in [str(item) for item in (decomposition.get("required_hops") or []) if str(item)]:
        left, _sep, right = hop.partition("->")
        left_text = left.strip().lower()
        right_text = right.strip().lower()
        if left_text and right_text and left_text in strong_text and right_text in strong_text:
            verified_hops.append(hop)
        else:
            unverified_hops.append(hop)
    ledger.verified_hops = verified_hops
    ledger.unverified_hops = unverified_hops
    ledger.source_confirmation_state = {
        **dict(ledger.source_confirmation_state or {}),
        "phase": phase,
        "supported_claims": len(supported),
        "unsupported_claims": len(unsupported),
        "verified_hops": len(verified_hops),
        "unverified_hops": len(unverified_hops),
    }


def _phase_queries(
    phase: str,
    *,
    query: str,
    decomposition: Dict[str, Any],
    fallback_queries: Sequence[tuple[str, str, str]],
) -> List[tuple[str, str, str]]:
    def _dedupe(items: Sequence[tuple[str, str, str]], *, limit: int) -> List[tuple[str, str, str]]:
        seen: set[tuple[str, str]] = set()
        materialized: List[tuple[str, str, str]] = []
        for query_text, strategy, rationale in items:
            normalized = str(query_text or "").strip()
            strategy_name = str(strategy or "").strip().lower()
            key = (strategy_name, normalized.casefold())
            if not normalized or key in seen:
                continue
            seen.add(key)
            materialized.append((normalized, strategy, rationale))
            if len(materialized) >= limit:
                break
        return materialized

    if phase == "entity_discovery":
        queries = [
            (
                str(entity.get("canonical_name") or entity.get("matched_alias") or query),
                "hybrid",
                "entity_discovery",
            )
            for entity in (decomposition.get("canonical_entities") or [])[:4]
        ]
        queries.extend(list(fallback_queries[:3]))
        return _dedupe(queries, limit=4) or list(fallback_queries[:1])
    if phase == "relationship_discovery":
        queries = [(item, "hybrid", "relationship_discovery") for item in (decomposition.get("relationship_questions") or [])[:4]]
        queries.extend(list(fallback_queries[:3]))
        return _dedupe(queries, limit=4) or list(fallback_queries[:2])
    if phase == "source_confirmation":
        queries = [
            (item, "keyword", "source_confirmation")
            for item in (decomposition.get("source_confirmation_questions") or [])[:4]
        ]
        queries.extend([item for item in fallback_queries if str(item[1] or "").strip().lower() == "keyword"][:3])
        if not any(str(item[1] or "").strip().lower() == "keyword" for item in queries) and fallback_queries:
            queries.append((str(fallback_queries[0][0] or query), "keyword", "source_confirmation"))
        queries.extend(list(fallback_queries[:2]))
        return _dedupe(queries, limit=4) or list(fallback_queries[:2])
    return list(fallback_queries[:1])


def _lane_candidate_counts(chunks: Sequence[ScoredChunk]) -> Dict[str, int]:
    counts = {
        "vector": 0,
        "keyword": 0,
        "graph": 0,
        "structured_neighbor": 0,
    }
    for chunk in chunks:
        metadata = dict(chunk.doc.metadata or {})
        lanes = dict(metadata.get("_retrieval_lanes") or {})
        method = str(chunk.method or "").lower()
        if "vector" in lanes or "vector" in method:
            counts["vector"] += 1
        if "keyword" in lanes or "bm25" in method or "keyword" in method:
            counts["keyword"] += 1
        if ("retrieval" in lanes or method == "retrieval") and "vector" not in lanes and "keyword" not in lanes:
            counts["vector"] += 1
            counts["keyword"] += 1
        if "graph" in lanes or "graph" in method:
            counts["graph"] += 1
        if "structured_neighbor" in lanes or "structured_neighbor" in method:
            counts["structured_neighbor"] += 1
    return counts


def _build_fast_run(
    settings: Any,
    stores: Any,
    *,
    providers: Any,
    session: Any,
    query: str,
    preferred_doc_ids: List[str],
    must_include_uploads: bool,
    top_k_vector: int,
    top_k_keyword: int,
    callbacks: List[Any],
    source_plan: SourcePlan | None = None,
    stage_timer: RetrievalBudget | None = None,
) -> RetrievalRun:
    collection_id = _resolve_collection_id(settings, session)
    collection_ids = _resolve_collection_ids(settings, session, explicit_collection_id=collection_id)
    search_query = _augment_query_with_aliases(query)
    plan = source_plan or _new_source_plan(query)
    preferred_doc_ids = list(dict.fromkeys([*preferred_doc_ids, *plan.preferred_doc_ids]))
    del collection_ids
    adapter = CorpusRetrievalAdapter(stores, settings=settings, session=session, source_plan=plan, stage_timer=stage_timer)
    with _measure_stage(stage_timer, "hybrid_retrieval"):
        merged = adapter.search_corpus(
            query,
            filters=SearchFilters(doc_ids=list(preferred_doc_ids), collection_id=collection_id),
            strategy="hybrid",
            limit=max(12, top_k_vector + top_k_keyword),
            top_k_vector=top_k_vector,
            top_k_keyword=top_k_keyword,
            preferred_doc_ids=list(preferred_doc_ids),
            must_include_uploads=must_include_uploads,
        )
    with _measure_stage(stage_timer, "chunk_grading"):
        graded = grade_chunks(
            providers.judge,
            settings=settings,
            question=query,
            chunks=[_with_adaptive_metadata(chunk.doc, score=chunk.score, strategy=chunk.method, query=query, rationale="fast_path") for chunk in merged],
            max_chunks=max(12, min(len(merged), top_k_vector + top_k_keyword)),
            callbacks=callbacks,
        )
    query_facets = _extract_query_facets(query)
    selected_docs = _select_facet_aware_evidence_docs(
        query,
        graded,
        _desired_evidence_budget(query, settings),
        query_facets,
    )
    ledger = EvidenceLedger(question=query)
    ledger.facets = list(query_facets)
    for chunk in merged:
        ledger.add_scored_chunks(
            [chunk],
            query=query,
            strategy=chunk.method,
            rationale="fast_path_seed",
            round_index=0,
        )
    ledger.apply_grades(graded)
    _refresh_facet_coverage(ledger, [item.doc for item in graded if item.relevance >= 2] or selected_docs)
    facet_coverage = dict(ledger.facet_coverage or {})
    lane_counts = _lane_candidate_counts(merged)
    timing_fields = stage_timer.to_fields() if stage_timer is not None else {}
    return RetrievalRun(
        selected_docs=list(selected_docs),
        candidate_docs=[item.doc for item in graded],
        graded=list(graded),
        query_used=search_query,
        search_mode="fast",
        rounds=1,
        tool_calls_used=3,
        tool_call_log=[
            f"fast:vector:{lane_counts['vector']}",
            f"fast:keyword:{lane_counts['keyword']}",
            f"fast:graph:{lane_counts['graph']}",
            f"fast:graded:{len(graded)}",
        ],
        strategies_used=sorted({chunk.method for chunk in merged} | {"grade"}),
        candidate_counts=_fast_candidate_counts(
            {"vector": [None] * lane_counts["vector"], "keyword": [None] * lane_counts["keyword"], "merged": merged},
            graded,
            selected_docs,
            graph_hits=lane_counts["graph"],
            sql_doc_hints=len(plan.preferred_doc_ids),
        )
        | {
            "structured_neighbor_hits": lane_counts["structured_neighbor"],
            "query_facets": len(query_facets),
            "covered_facets": sum(1 for item in facet_coverage.values() if str(item.get("state")) == "covered"),
            "missing_facets": sum(1 for item in facet_coverage.values() if str(item.get("state")) == "missing"),
        },
        evidence_ledger=ledger.to_dict(),
        source_plan=plan.to_dict(),
        sources_used=list(plan.sources_chosen),
        graphs_considered=[str(item.get("graph_id") or "") for item in plan.graph_shortlist if str(item.get("graph_id") or "")],
        graph_methods_used=list(plan.graph_methods),
        sql_sources_used=list(plan.sql_views_used),
        resolution_stats={
            "graph_hits_resolved": sum(1 for item in merged if item.method == "graph"),
            "sql_doc_hints": len(plan.preferred_doc_ids),
        },
        decomposition=dict(plan.decomposition or {}),
        claim_ledger=ledger.to_dict(),
        verified_hops=list(ledger.verified_hops),
        stage_timings_ms=dict(timing_fields.get("stage_timings_ms") or {}),
        budget_ms=int(timing_fields.get("budget_ms") or 0),
        budget_exhausted=bool(timing_fields.get("budget_exhausted", False)),
        slow_stages=[str(item) for item in (timing_fields.get("slow_stages") or []) if str(item)],
    )


def _should_escalate(
    question: str,
    fast_run: RetrievalRun,
    settings: Any,
    *,
    research_profile: str = "",
    coverage_goal: str = "",
    result_mode: str = "",
    controller_hints: Dict[str, Any] | None = None,
) -> bool:
    resolved_hints = coerce_controller_hints(controller_hints)
    min_evidence = max(1, int(getattr(settings, "rag_min_evidence_chunks", 1)))
    facet_coverage = dict((fast_run.evidence_ledger or {}).get("facet_coverage") or {})
    strong_chunks = int(fast_run.candidate_counts.get("strong_chunks", 0))
    if (
        _looks_simple_fact_query(question)
        and not _is_discovery_query(question)
        and result_mode not in {"inventory", "comparison"}
        and not bool(resolved_hints.get("force_deep_search"))
        and not any(str(item.get("state") or "") == "missing" for item in facet_coverage.values())
        and len(fast_run.selected_docs) >= min_evidence
        and strong_chunks >= min_evidence
    ):
        return False
    if (
        _is_discovery_query(question)
        or _is_complex_query(question)
        or research_profile in {"corpus_discovery", "process_flow_identification", "comparison_campaign"}
        or coverage_goal in {"corpus_wide", "exhaustive"}
        or result_mode in {"inventory", "comparison"}
        or bool(resolved_hints.get("force_deep_search"))
    ):
        return True
    if any(str(item.get("state") or "") == "missing" for item in facet_coverage.values()):
        return True
    if len(fast_run.selected_docs) < min_evidence:
        return True
    if strong_chunks < min_evidence:
        return True
    if "LLM" in " ".join(fast_run.tool_call_log):
        return True
    return False


def _discovery_filters(
    question: str,
    settings: Any,
    session: Any,
    preferred_doc_ids: Sequence[str],
    *,
    prefer_process_flow: bool = False,
) -> SearchFilters:
    filters = SearchFilters(
        doc_ids=[str(item) for item in preferred_doc_ids if str(item)],
    )
    if prefer_process_flow:
        filters.doc_structure_types = ["process_flow_doc"]
    return filters


def _build_round_queries(
    question: str,
    *,
    settings: Any,
    providers: Any,
    conversation_context: str,
    callbacks: List[Any],
    round_index: int,
    discovery: bool,
    prefer_process_flow: bool,
    controller_hints: Dict[str, Any],
    seen_queries: set[tuple[str, str]],
) -> List[tuple[str, str, str]]:
    candidates: List[tuple[str, str, str]] = []
    topic_seed_terms = workflow_topic_seed_terms(question)
    if round_index == 1:
        candidates.append((question, "hybrid", "original"))
    rewritten = _rewrite_query(
        providers.judge,
        settings=settings,
        question=question,
        conversation_context=conversation_context,
        attempt=round_index,
        callbacks=callbacks,
    ).strip()
    if rewritten and rewritten.lower() != question.lower():
        candidates.append((rewritten, "hybrid", "rewritten"))

    keyword_query = _keyword_projection(question)
    if keyword_query and keyword_query.lower() != question.lower():
        candidates.append((keyword_query, "keyword", "keyword_projection"))
        if not _is_exact_match_query(question):
            candidates.append((keyword_query, "vector", "semantic_projection"))

    candidates.extend(
        _generic_rewrite_candidates(
            question,
            conversation_context=conversation_context,
            round_index=round_index,
        )
    )

    if discovery and prefer_process_flow:
        if topic_seed_terms:
            candidates.append(
                (
                    " ".join([*topic_seed_terms, "workflow", "process flow", "procedure steps", "approval handoff"]),
                    "keyword",
                    "process_flow_fallback",
                )
            )
        else:
            candidates.append(("process flow workflow approval flow handoff escalation path", "keyword", "process_flow_fallback"))
    if discovery and controller_hints.get("prefer_inventory_output"):
        if topic_seed_terms:
            candidates.append(
                (
                    " ".join([*topic_seed_terms, "document title", "workflow", "procedure steps", "approval handoff"]),
                    "keyword",
                    "inventory_fallback",
                )
            )
        else:
            candidates.append(("document title file workflow procedure steps approval handoff", "keyword", "inventory_fallback"))

    materialized: List[tuple[str, str, str]] = []
    for query_text, strategy, rationale in candidates:
        key = (strategy, query_text.strip().lower())
        if not query_text.strip() or key in seen_queries:
            continue
        materialized.append((query_text.strip(), strategy, rationale))
    return materialized


def _select_discovery_docs(question: str, ledger: EvidenceLedger, graded: Sequence[GradedChunk], settings: Any) -> List[Document]:
    desired = _desired_evidence_budget(question, settings)
    best_by_doc: Dict[str, Document] = {}
    for item in graded:
        doc_id = _doc_id(item.doc)
        if not doc_id:
            continue
        current = best_by_doc.get(doc_id)
        if current is None or (_adaptive_score(item.doc), item.relevance) > (_adaptive_score(current), int((current.metadata or {}).get("_adaptive_relevance") or 0)):
            metadata = dict(item.doc.metadata or {})
            metadata["_adaptive_relevance"] = item.relevance
            best_by_doc[doc_id] = Document(page_content=item.doc.page_content, metadata=metadata)

    selected: List[Document] = []
    for doc_id in ledger.best_doc_ids(limit=max(desired + 2, 6)):
        candidate = best_by_doc.get(doc_id)
        if candidate is None:
            continue
        if _PROCESS_FLOW_HINTS.search(question) and not _matches_process_flow(candidate):
            continue
        if not document_has_explicit_topic_support(question, candidate).get("matches", True):
            continue
        selected.append(candidate)
        if len(selected) >= desired:
            break
    return selected or _select_evidence_docs(question, graded, desired)


def _group_doc_ids(doc_ids: Sequence[str], *, group_size: int) -> List[List[str]]:
    groups: List[List[str]] = []
    current: List[str] = []
    for doc_id in doc_ids:
        if not doc_id:
            continue
        current.append(doc_id)
        if len(current) >= max(1, int(group_size)):
            groups.append(list(current))
            current = []
    if current:
        groups.append(list(current))
    return groups


def _covers_requested_docs(selected_docs: Sequence[Document], controller_hints: Dict[str, Any]) -> bool:
    requested = [str(item) for item in (controller_hints.get("resolved_doc_ids") or []) if str(item).strip()]
    if not requested:
        return False
    selected_ids = {_doc_id(doc) for doc in selected_docs if _doc_id(doc)}
    return all(doc_id in selected_ids for doc_id in requested)


def _plan_parallel_search_tasks(
    question: str,
    *,
    ledger: EvidenceLedger,
    round_queries: Sequence[tuple[str, str, str]],
    max_search_rounds: int,
    research_profile: str,
    coverage_goal: str,
    result_mode: str,
    controller_hints: Dict[str, Any],
) -> List[RagSearchTask]:
    candidate_doc_ids = ledger.best_doc_ids(limit=8)
    if len(candidate_doc_ids) < 2:
        return []
    max_parallel_lanes = max(1, int(controller_hints.get("max_parallel_lanes") or 3))

    tasks: List[RagSearchTask] = []
    if _is_discovery_query(question):
        group_size = 1 if len(candidate_doc_ids) <= 3 else 2
        for index, doc_group in enumerate(_group_doc_ids(candidate_doc_ids[:6], group_size=group_size), start=1):
            tasks.append(
                RagSearchTask(
                    task_id=f"search_group_{index}",
                    title=f"Search document group {index}",
                    query=question,
                    doc_scope=list(doc_group),
                    strategies=["hybrid", "keyword"],
                    round_budget=max(1, min(2, int(max_search_rounds) - 1 or 1)),
                    research_profile=research_profile,
                    coverage_goal=coverage_goal,
                    result_mode=result_mode,
                    controller_hints=dict(controller_hints),
                )
            )
    elif _is_comparison_query(question):
        for index, doc_id in enumerate(candidate_doc_ids[:4], start=1):
            tasks.append(
                RagSearchTask(
                    task_id=f"compare_doc_{index}",
                    title=f"Inspect {doc_id}",
                    query=question,
                    doc_scope=[doc_id],
                    strategies=["hybrid"],
                    round_budget=1,
                    research_profile=research_profile,
                    coverage_goal=coverage_goal,
                    result_mode=result_mode,
                    controller_hints=dict(controller_hints),
                )
            )
    elif len(candidate_doc_ids) >= 4 and len(round_queries) >= 2:
        scoped_doc_ids = candidate_doc_ids[:4]
        for index, (query_text, strategy, _rationale) in enumerate(round_queries[:3], start=1):
            tasks.append(
                RagSearchTask(
                    task_id=f"facet_{index}",
                    title=f"{strategy.title()} search focus {index}",
                    query=query_text,
                    doc_scope=list(scoped_doc_ids),
                    strategies=[strategy],
                    round_budget=1,
                    research_profile=research_profile,
                    coverage_goal=coverage_goal,
                    result_mode=result_mode,
                    controller_hints=dict(controller_hints),
                )
            )
    tasks = tasks[:max_parallel_lanes]
    return tasks if len(tasks) >= 2 else []


def _merge_worker_result(
    ledger: EvidenceLedger,
    result: RagSearchTaskResult,
    *,
    round_index: int,
) -> int:
    added = 0
    candidate_docs = [deserialize_document(item) for item in result.candidate_docs]
    if candidate_docs:
        added += ledger.add_documents(
            candidate_docs,
            score=0.72,
            query=result.task_id,
            strategy="worker",
            rationale=f"worker:{result.task_id}",
            round_index=round_index,
        )

    graded = [deserialize_graded_chunk(item) for item in result.graded_chunks]
    if graded:
        added += ledger.add_documents(
            [item.doc for item in graded],
            score=0.78,
            query=result.task_id,
            strategy="worker_grade",
            rationale=f"worker_grade:{result.task_id}",
            round_index=round_index,
        )
        ledger.apply_grades(graded)

    for raw in result.evidence_entries:
        chunk_id = str(raw.get("chunk_id") or "")
        if not chunk_id:
            continue
        entry = ledger.entries.get(chunk_id)
        if entry is None:
            continue
        entry.coverage_state = str(raw.get("coverage_state") or entry.coverage_state)
        entry.grade_reason = str(raw.get("grade_reason") or entry.grade_reason)
        entry.relevance = max(int(raw.get("relevance") or 0), entry.relevance)
        entry.score = max(float(raw.get("score") or 0.0), entry.score)
        entry.title = str(raw.get("title") or entry.title)
        entry.doc_id = str(raw.get("doc_id") or entry.doc_id)
        entry.query = str(raw.get("query") or entry.query)
        entry.strategy = str(raw.get("strategy") or entry.strategy)
        entry.rationale = str(raw.get("rationale") or entry.rationale)
        entry.round_index = max(entry.round_index, round_index)
    return added


@dataclass
class RetrievalReflection:
    action: str = "stop"
    followup_queries: List[str] = field(default_factory=list)
    prefer_doc_ids: List[str] = field(default_factory=list)
    full_read_doc_ids: List[str] = field(default_factory=list)
    rationale: str = ""


def _reflect_on_retrieval(
    *,
    providers: Any,
    question: str,
    conversation_context: str,
    ledger: EvidenceLedger,
    selected_docs: Sequence[Document],
    graded: Sequence[GradedChunk],
    round_index: int,
    max_rounds: int,
    controller_hints: Dict[str, Any],
) -> RetrievalReflection:
    strong_chunks = sum(1 for item in graded if item.relevance >= 2)
    selected_doc_ids = [_doc_id(doc) for doc in selected_docs if _doc_id(doc)]
    best_doc_ids = ledger.best_doc_ids(limit=4)
    judge_model = getattr(providers, "judge", None)
    if judge_model is not None:
        try:
            prompt = (
                "Decide whether the retrieval controller should stop or run one more focused evidence wave.\n"
                "Return JSON only with keys: action, followup_queries, prefer_doc_ids, full_read_doc_ids, rationale.\n"
                f"QUESTION: {question}\n"
                f"STRONG_CHUNKS: {strong_chunks}\n"
                f"SELECTED_DOC_IDS: {selected_doc_ids}\n"
                f"BEST_DOC_IDS: {best_doc_ids}\n"
                f"UNSUPPORTED_CLAIMS: {ledger.unsupported_claim_ids}\n"
                f"UNVERIFIED_HOPS: {ledger.unverified_hops}\n"
                f"FACET_COVERAGE: {ledger.facet_coverage}\n"
                f"CONTEXT: {conversation_context[:1000]}\n"
            )
            response = judge_model.invoke(prompt, config={"callbacks": []})
            text = getattr(response, "content", None) or str(response)
            payload = extract_json(text) or {}
            if isinstance(payload, dict):
                action = str(payload.get("action") or "").strip().lower()
                if action in {"stop", "retry"}:
                    return RetrievalReflection(
                        action=action,
                        followup_queries=[str(item) for item in (payload.get("followup_queries") or []) if str(item).strip()][:2],
                        prefer_doc_ids=[str(item) for item in (payload.get("prefer_doc_ids") or []) if str(item).strip()][:3],
                        full_read_doc_ids=[str(item) for item in (payload.get("full_read_doc_ids") or []) if str(item).strip()][:2],
                        rationale=str(payload.get("rationale") or "").strip(),
                    )
        except Exception:
            pass
    if strong_chunks >= 3 and selected_doc_ids:
        return RetrievalReflection(action="stop", rationale="sufficient strong evidence")
    if round_index >= max_rounds:
        return RetrievalReflection(action="stop", rationale="round budget exhausted")

    if best_doc_ids:
        followup_queries: List[str] = []
        if not strong_chunks:
            followup_queries.append(_keyword_projection(question))
        for doc_id in best_doc_ids[:2]:
            entry = next((item for item in ledger.entries.values() if item.doc_id == doc_id and item.title), None)
            if entry is not None:
                followup_queries.append(f"{question} {entry.title}")
        deduped_queries: List[str] = []
        seen: set[str] = set()
        for query_text in followup_queries:
            clean = str(query_text or "").strip()
            if not clean or clean.lower() in seen:
                continue
            seen.add(clean.lower())
            deduped_queries.append(clean)
        return RetrievalReflection(
            action="retry",
            followup_queries=deduped_queries[:2],
            prefer_doc_ids=best_doc_ids[:3],
            full_read_doc_ids=best_doc_ids[:2] if bool(controller_hints.get("prefer_full_reads") or selected_doc_ids) else best_doc_ids[:1],
            rationale="evidence is still thin; retry with doc-focused and full-read follow-up",
        )
    return RetrievalReflection(action="stop", rationale="no better evidence targets available")


def _deep_path_sufficient(
    question: str,
    *,
    ledger: EvidenceLedger,
    decomposition: Dict[str, Any],
    selected_docs: Sequence[Document],
    graded: Sequence[GradedChunk],
    new_chunks: int,
    round_index: int,
    max_rounds: int,
    settings: Any,
    discovery: bool = False,
    controller_hints: Dict[str, Any] | None = None,
) -> bool:
    desired = _desired_evidence_budget(question, settings)
    strong = sum(1 for item in graded if item.relevance >= 2)
    resolved_hints = coerce_controller_hints(controller_hints)
    high_priority_claims = [
        str(item.get("claim_id") or "")
        for item in ledger.claims
        if str(item.get("priority") or "").lower() == "high"
    ]
    if ledger.claims:
        all_claims_supported = len(ledger.supported_claim_ids) >= len(ledger.claims)
        no_high_priority_gap = all(
            claim_id in set(ledger.supported_claim_ids)
            for claim_id in high_priority_claims
            if claim_id
        )
        if all_claims_supported and not ledger.unverified_hops and no_high_priority_gap:
            return True
    if ledger.facets:
        missing_facets = [
            key
            for key, value in dict(ledger.facet_coverage or {}).items()
            if str(value.get("state") or "") != "covered"
        ]
        if not missing_facets and selected_docs:
            return True
        if missing_facets and round_index < max_rounds:
            return False
    if _covers_requested_docs(selected_docs, resolved_hints):
        if strong >= min(desired, len(selected_docs)) or round_index >= max_rounds or new_chunks == 0:
            return True
    if discovery or _is_discovery_query(question):
        if len({_doc_id(doc) for doc in selected_docs if _doc_id(doc)}) >= desired:
            return True
        if round_index >= max_rounds and selected_docs:
            return True
        return new_chunks == 0 and bool(selected_docs)
    if strong >= desired and len(selected_docs) >= min(desired, strong):
        return True
    return round_index >= max_rounds and bool(selected_docs)


def _run_deep_path(
    settings: Any,
    stores: Any,
    *,
    providers: Any,
    session: Any,
    query: str,
    conversation_context: str,
    preferred_doc_ids: List[str],
    must_include_uploads: bool,
    top_k_vector: int,
    top_k_keyword: int,
    max_search_rounds: int,
    callbacks: List[Any],
    seed_run: RetrievalRun | None,
    research_profile: str,
    coverage_goal: str,
    result_mode: str,
    controller_hints: Dict[str, Any],
    runtime_bridge: RagRuntimeBridge | None,
    progress_emitter: Any | None,
    allow_internal_fanout: bool,
    source_plan: SourcePlan | None = None,
    stage_timer: RetrievalBudget | None = None,
) -> RetrievalRun:
    plan = source_plan or _new_source_plan(query)
    preferred_doc_ids = list(dict.fromkeys([*preferred_doc_ids, *plan.preferred_doc_ids]))
    adapter = CorpusRetrievalAdapter(stores, settings=settings, session=session, source_plan=plan, stage_timer=stage_timer)
    ledger = EvidenceLedger(question=query)
    with _measure_stage(stage_timer, "source_planning"):
        decomposition = _build_decomposition(
            query=query,
            settings=settings,
            stores=stores,
            session=session,
            controller_hints=controller_hints,
            source_plan=plan,
        )
    _initialize_claim_ledger(ledger, decomposition)
    tool_calls = 0
    tool_call_log: List[str] = []
    strategies: set[str] = set()
    parallel_workers_used = False
    fanout_attempted = False
    discovery = _discovery_mode(
        query,
        research_profile=research_profile,
        coverage_goal=coverage_goal,
        result_mode=result_mode,
        controller_hints=controller_hints,
    )
    prefer_process_flow = _prefer_process_flow(
        query,
        research_profile=research_profile,
        controller_hints=controller_hints,
    )
    filters = _discovery_filters(
        query,
        settings,
        session,
        preferred_doc_ids,
        prefer_process_flow=prefer_process_flow,
    )
    max_parallel_lanes = max(1, int(controller_hints.get("max_parallel_lanes") or 3))
    max_reflection_rounds = max(0, int(controller_hints.get("max_reflection_rounds") or 0))
    full_read_chunk_threshold = max(6, int(controller_hints.get("full_read_chunk_threshold") or 24))
    default_doc_read_depth = str(controller_hints.get("doc_read_depth") or "").strip().lower()
    forced_full_read_doc_ids: set[str] = set()
    reflection_rounds_used = 0
    reflection_followup_queries: List[str] = []
    _emit_progress(
        progress_emitter,
        "phase_start",
        label="Planning search",
        detail="Deep retrieval controller",
        agent="rag_worker",
        why="The query needs more evidence coverage than the fast path can safely provide.",
    )

    if seed_run is not None:
        added = ledger.add_documents(
            seed_run.candidate_docs,
            score=0.8,
            query=query,
            strategy="seed",
            rationale="fast_path_seed",
            round_index=0,
        )
        ledger.round_summaries.append({"round": 0, "new_chunks": added, "mode": "seed"})
        if seed_run.claim_ledger:
            ledger.claims = [dict(item) for item in (seed_run.claim_ledger.get("claims") or ledger.claims)]
            ledger.supported_claim_ids = [str(item) for item in (seed_run.claim_ledger.get("supported_claim_ids") or []) if str(item)]
            ledger.unsupported_claim_ids = [str(item) for item in (seed_run.claim_ledger.get("unsupported_claim_ids") or []) if str(item)]
            ledger.verified_hops = [str(item) for item in (seed_run.claim_ledger.get("verified_hops") or []) if str(item)]
            ledger.unverified_hops = [str(item) for item in (seed_run.claim_ledger.get("unverified_hops") or []) if str(item)]
            ledger.source_confirmation_state = dict(seed_run.claim_ledger.get("source_confirmation_state") or {})

    resolved_doc_ids = [str(item) for item in (controller_hints.get("resolved_doc_ids") or []) if str(item).strip()]
    if resolved_doc_ids:
        direct_read_added = 0
        for doc_id in resolved_doc_ids[:4]:
            docs = adapter.read_document(doc_id, focus=query, max_chunks=4)
            if not docs:
                continue
            direct_read_added += ledger.add_documents(
                docs,
                score=0.88,
                query=query,
                strategy="read_document",
                rationale="explicit_doc_target",
                round_index=0,
            )
            tool_calls += 1
            strategies.add("read_document")
            tool_call_log.append(f"round0:read_document:{doc_id}")
            _emit_progress(
                progress_emitter,
                "doc_focus",
                label="Reading named document",
                detail=doc_id,
                agent="rag_worker",
                docs=_doc_focus_from_documents(docs),
            )
        if direct_read_added:
            ledger.round_summaries.append({"round": 0, "new_chunks": direct_read_added, "mode": "explicit_doc_target"})
        if bool(controller_hints.get("prefer_full_reads")):
            forced_full_read_doc_ids.update(resolved_doc_ids[: max_parallel_lanes])

    seen_queries: set[tuple[str, str]] = set()
    excluded_chunk_ids = set(ledger.entries.keys())
    latest_graded = list(seed_run.graded) if seed_run is not None else []
    latest_selected = list(seed_run.selected_docs) if seed_run is not None else []
    latest_candidates = ledger.materialize_documents(max_chunks=max(12, top_k_vector + top_k_keyword))

    phase_sequence = _phase_sequence(max_search_rounds)
    for round_index, phase in enumerate(phase_sequence, start=1):
        if stage_timer is not None and stage_timer.should_cutoff():
            ledger.round_summaries.append(
                {
                    "round": round_index,
                    "phase": phase,
                    "new_chunks": 0,
                    "mode": "budget_cutoff",
                }
            )
            break
        _emit_progress(
            progress_emitter,
            "decision_point",
            label=f"{phase.replace('_', ' ').title()}",
            detail="Choosing the next retrieval strategy",
            agent="rag_worker",
            why="The controller is deciding whether to broaden, focus, or reread the evidence set.",
            counts={"round": round_index, "phase": phase},
        )
        _emit_progress(
            progress_emitter,
            "phase_update",
            label=f"{phase.replace('_', ' ').title()}",
            detail="Expanding evidence",
            agent="rag_worker",
            counts={"round": round_index, "phase": phase},
        )
        with _measure_stage(stage_timer, "source_planning"):
            round_queries = _build_round_queries(
                query,
                settings=settings,
                providers=providers,
                conversation_context=conversation_context,
                callbacks=callbacks,
                round_index=round_index,
                discovery=discovery,
                prefer_process_flow=prefer_process_flow,
                controller_hints=controller_hints,
                seen_queries=seen_queries,
            )
        round_queries = _phase_queries(
            phase,
            query=query,
            decomposition=decomposition,
            fallback_queries=round_queries,
        )
        if reflection_followup_queries:
            reflected_queries = [
                (item, "hybrid", "reflection_retry")
                for item in reflection_followup_queries
                if str(item).strip()
            ]
            round_queries = [*reflected_queries, *round_queries]
            reflection_followup_queries = []
        if not round_queries:
            break

        round_new_chunks = 0
        round_queries_used: List[str] = []
        round_top_k_vector = top_k_vector * (2 if discovery else 1)
        round_top_k_keyword = top_k_keyword * (2 if discovery else 1)

        prioritized_sections = decomposition.get("prioritized_sections") or plan.prioritized_sections
        early_section_confirmation = bool(prioritized_sections) or bool(resolved_doc_ids)
        if (
            bool(getattr(settings, "section_first_retrieval_enabled", False))
            and (phase == "source_confirmation" or (round_index == 1 and early_section_confirmation))
        ):
            section_docs = adapter.search_section_scope(
                query,
                doc_ids=(resolved_doc_ids or preferred_doc_ids or ledger.best_doc_ids(limit=6))[:6],
                prioritized_sections=prioritized_sections,
                limit=max(4, round_top_k_keyword),
            )
            if section_docs:
                tool_calls += 1
                strategies.add("section_scope")
                tool_call_log.append(f"round{round_index}:search_sections:{query}")
                round_new_chunks += ledger.add_documents(
                    section_docs,
                    score=0.9,
                    query=query,
                    strategy="section_scope",
                    rationale="section_first_confirmation",
                    round_index=round_index,
                )
                _emit_progress(
                    progress_emitter,
                    "doc_focus",
                    label="Narrowing to likely sections",
                    detail=query[:120],
                    agent="rag_worker",
                    docs=_doc_focus_from_documents(section_docs),
                )

        doc_read_targets = list(
            dict.fromkeys(
                [
                    *list(forced_full_read_doc_ids),
                    *(
                        resolved_doc_ids[:max_parallel_lanes]
                        if bool(controller_hints.get("prefer_full_reads"))
                        else []
                    ),
                    *(
                        ledger.best_doc_ids(limit=max_parallel_lanes)
                        if discovery or bool(controller_hints.get("prefer_full_reads"))
                        else []
                    ),
                ]
            )
        )[:max_parallel_lanes]
        if doc_read_targets:
            for doc_id in doc_read_targets:
                read_depth = "full" if doc_id in forced_full_read_doc_ids else (default_doc_read_depth or "focused")
                if read_depth not in {"focused", "outline", "full"}:
                    read_depth = "focused"
                docs = adapter.read_document(
                    doc_id,
                    focus=query,
                    max_chunks=full_read_chunk_threshold if read_depth == "full" else 6,
                    read_depth=read_depth,
                    full_read_chunk_threshold=full_read_chunk_threshold,
                )
                if not docs:
                    continue
                tool_calls += 1
                strategies.add("document_read")
                tool_call_log.append(f"round{round_index}:document_read[{read_depth}]:{doc_id}")
                round_new_chunks += ledger.add_documents(
                    docs,
                    score=0.91 if read_depth == "full" else 0.84,
                    query=query,
                    strategy="document_read",
                    rationale=f"doc_read_lane:{read_depth}",
                    round_index=round_index,
                )
                _emit_progress(
                    progress_emitter,
                    "doc_focus",
                    label="Reading candidate document",
                    detail=f"{doc_id} ({read_depth})",
                    agent="rag_worker",
                    docs=_doc_focus_from_documents(docs),
                )
            forced_full_read_doc_ids.clear()

        for query_text, strategy, rationale in round_queries:
            seen_queries.add((strategy, query_text.lower()))
            round_queries_used.append(f"{strategy}:{query_text}")
            _emit_progress(
                progress_emitter,
                "phase_update",
                label=f"Searching via {strategy}",
                detail=query_text[:120],
                agent="rag_worker",
                counts={"round": round_index},
            )
            if strategy == "keyword":
                hits = adapter.grep_corpus(
                    query_text,
                    filters=filters,
                    exclude_chunk_ids=excluded_chunk_ids,
                    limit=max(6, round_top_k_keyword),
                )
                tool_call_log.append(f"round{round_index}:grep_corpus:{query_text}")
            else:
                hits = adapter.search_corpus(
                    query_text,
                    filters=filters,
                    exclude_chunk_ids=excluded_chunk_ids,
                    strategy=strategy,
                    limit=max(8, round_top_k_vector + round_top_k_keyword),
                    top_k_vector=round_top_k_vector,
                    top_k_keyword=round_top_k_keyword,
                    preferred_doc_ids=preferred_doc_ids,
                    must_include_uploads=must_include_uploads,
                )
                tool_call_log.append(f"round{round_index}:search_corpus[{strategy}]:{query_text}")
            tool_calls += 1
            strategies.add(strategy)
            round_new_chunks += ledger.add_scored_chunks(
                hits,
                query=query_text,
                strategy=strategy,
                rationale=rationale,
                round_index=round_index,
            )
            if hits:
                _emit_progress(
                    progress_emitter,
                    "doc_focus",
                    label="Reviewing candidate documents",
                    detail=query_text[:120],
                    agent="rag_worker",
                    docs=_doc_focus_from_documents([hit.doc for hit in hits]),
                    counts={"hits": len(hits), "round": round_index},
                )

            window_limit = 4 if controller_hints.get("prefer_windowed_keyword_followup") else 2
            for hit in hits[:window_limit]:
                window_docs = adapter.fetch_chunk_window(_chunk_id(hit.doc), before=1, after=1)
                if not window_docs:
                    continue
                tool_calls += 1
                strategies.add("window")
                tool_call_log.append(f"round{round_index}:fetch_chunk_window:{_chunk_id(hit.doc)}")
                round_new_chunks += ledger.add_documents(
                    window_docs,
                    score=max(hit.score - 0.02, 0.01),
                    query=query_text,
                    strategy="window",
                    rationale="context_window",
                    round_index=round_index,
                )
                _emit_progress(
                    progress_emitter,
                    "doc_focus",
                    label="Expanding context window",
                    detail=_chunk_id(hit.doc),
                    agent="rag_worker",
                    docs=_doc_focus_from_documents(window_docs),
                )

            if discovery:
                seen_doc_ids: set[str] = set()
                for hit in hits:
                    hit_doc_id = _doc_id(hit.doc)
                    if not hit_doc_id or hit_doc_id in seen_doc_ids:
                        continue
                    seen_doc_ids.add(hit_doc_id)
                    docs = adapter.read_document(
                        hit_doc_id,
                        focus=query_text,
                        max_chunks=4,
                        read_depth="full" if bool(controller_hints.get("prefer_full_reads")) else "focused",
                        full_read_chunk_threshold=full_read_chunk_threshold,
                    )
                    if not docs:
                        continue
                    tool_calls += 1
                    strategies.add("read_document")
                    tool_call_log.append(f"round{round_index}:read_document:{hit_doc_id}")
                    round_new_chunks += ledger.add_documents(
                        docs,
                        score=max(hit.score - 0.01, 0.01),
                        query=query_text,
                        strategy="read_document",
                        rationale="focused_read",
                        round_index=round_index,
                    )
                    _emit_progress(
                        progress_emitter,
                        "doc_focus",
                        label="Reading document",
                        detail=hit_doc_id,
                        agent="rag_worker",
                        docs=_doc_focus_from_documents(docs),
                    )
                    if len(seen_doc_ids) >= 3:
                        break

            excluded_chunk_ids = set(ledger.entries.keys()) | set(ledger.pruned_chunk_ids)

        latest_candidates = ledger.materialize_documents(
            max_chunks=max(12, (top_k_vector + top_k_keyword) * 2),
        )
        with _measure_stage(stage_timer, "chunk_grading"):
            latest_graded = grade_chunks(
                providers.judge,
                settings=settings,
                question=query,
                chunks=latest_candidates,
                max_chunks=max(12, min(len(latest_candidates), max(18, top_k_vector + top_k_keyword))),
                callbacks=callbacks,
            )
        tool_calls += 1
        strategies.add("grade")
        tool_call_log.append(f"round{round_index}:grade_chunks:{len(latest_candidates)}")
        ledger.apply_grades(latest_graded)
        _emit_progress(
            progress_emitter,
            "evidence_status",
            label="Graded candidate evidence",
            detail=f"{sum(1 for item in latest_graded if item.relevance >= 2)} strong chunk(s)",
            agent="rag_worker",
            counts={
                "round": round_index,
                "graded_chunks": len(latest_graded),
                "strong_chunks": sum(1 for item in latest_graded if item.relevance >= 2),
            },
        )

        if discovery:
            latest_selected = _select_discovery_docs(query, ledger, latest_graded, settings)
        else:
            latest_selected = _select_facet_aware_evidence_docs(
                query,
                latest_graded,
                _desired_evidence_budget(query, settings),
                decomposition.get("query_facets") or [],
            )
        _refresh_claim_ledger(
            ledger,
            decomposition=decomposition,
            strong_docs=[item.doc for item in latest_graded if item.relevance >= 2] or latest_selected,
            phase=phase,
        )

        kept, pruned = adapter.prune_chunks(
            ledger.materialize_documents(max_chunks=max(18, (top_k_vector + top_k_keyword) * 2)),
            keep=max(18, top_k_vector + top_k_keyword),
            max_per_doc=3 if discovery else 2,
        )
        ledger.pruned_chunk_ids = list(pruned)
        excluded_chunk_ids = set(ledger.entries.keys()) | set(pruned)
        latest_candidates = kept

        if (
            not fanout_attempted
            and allow_internal_fanout
            and runtime_bridge is not None
        ):
            planned_tasks = _plan_parallel_search_tasks(
                query,
                ledger=ledger,
                round_queries=round_queries,
                max_search_rounds=max_search_rounds,
                research_profile=research_profile,
                coverage_goal=coverage_goal,
                result_mode=result_mode,
                controller_hints=controller_hints,
            )
            if planned_tasks and bool(runtime_bridge.can_run_parallel(task_count=len(planned_tasks))):
                fanout_attempted = True
                _emit_progress(
                    progress_emitter,
                    "task_plan",
                    label=f"Spawning {len(planned_tasks)} evidence workers",
                    detail="Parallel evidence gathering",
                    agent="rag_worker",
                    waiting_on="evidence workers",
                    counts={"tasks": len(planned_tasks), "round": round_index},
                    docs=_doc_focus_from_documents(latest_selected or latest_candidates),
                )
                batch_result = runtime_bridge.run_search_tasks(planned_tasks)
                parallel_workers_used = parallel_workers_used or bool(batch_result.parallel_workers_used)
                if batch_result.warnings:
                    ledger.unresolved_subquestions.extend(batch_result.warnings)
                successful_worker_results = 0
                for worker_result in batch_result.results:
                    round_new_chunks += _merge_worker_result(
                        ledger,
                        worker_result,
                        round_index=round_index,
                    )
                    if worker_result.candidate_docs or worker_result.graded_chunks or worker_result.evidence_entries:
                        successful_worker_results += 1
                        tool_call_log.append(f"round{round_index}:worker:{worker_result.task_id}")
                        strategies.add("worker")
                if successful_worker_results:
                    latest_candidates = ledger.materialize_documents(
                        max_chunks=max(12, (top_k_vector + top_k_keyword) * 2),
                    )
                    with _measure_stage(stage_timer, "chunk_grading"):
                        latest_graded = grade_chunks(
                            providers.judge,
                            settings=settings,
                            question=query,
                            chunks=latest_candidates,
                            max_chunks=max(12, min(len(latest_candidates), max(18, top_k_vector + top_k_keyword))),
                            callbacks=callbacks,
                        )
                    tool_calls += 1
                    strategies.add("grade")
                    tool_call_log.append(f"round{round_index}:grade_chunks:merged_workers")
                    ledger.apply_grades(latest_graded)
                    _emit_progress(
                        progress_emitter,
                        "evidence_status",
                        label="Merged worker evidence",
                        detail=f"{successful_worker_results} worker result(s)",
                        agent="rag_worker",
                        counts={
                            "round": round_index,
                            "graded_chunks": len(latest_graded),
                            "worker_results": successful_worker_results,
                        },
                    )
                    if discovery:
                        latest_selected = _select_discovery_docs(query, ledger, latest_graded, settings)
                    else:
                        latest_selected = _select_facet_aware_evidence_docs(
                            query,
                            latest_graded,
                            _desired_evidence_budget(query, settings),
                            decomposition.get("query_facets") or [],
                        )
                    _refresh_claim_ledger(
                        ledger,
                        decomposition=decomposition,
                        strong_docs=[item.doc for item in latest_graded if item.relevance >= 2] or latest_selected,
                        phase=phase,
                    )
                    _emit_progress(
                        progress_emitter,
                        "phase_update",
                        label="Merging evidence",
                        detail=f"{successful_worker_results} worker result(s)",
                        agent="rag_worker",
                        counts={"round": round_index},
                    )

        ledger.round_summaries.append(
            {
                "round": round_index,
                "phase": phase,
                "queries": round_queries_used,
                "new_chunks": round_new_chunks,
                "selected_docs": len(latest_selected),
                "strong_chunks": sum(1 for item in latest_graded if item.relevance >= 2),
            }
        )
        sufficient = _deep_path_sufficient(
            query,
            ledger=ledger,
            decomposition=decomposition,
            selected_docs=latest_selected,
            graded=latest_graded,
            new_chunks=round_new_chunks,
            round_index=round_index,
            max_rounds=max_search_rounds,
            settings=settings,
            discovery=discovery,
            controller_hints=controller_hints,
        )
        if not sufficient and reflection_rounds_used < max_reflection_rounds and not (stage_timer and stage_timer.should_cutoff()):
            reflection = _reflect_on_retrieval(
                providers=providers,
                question=query,
                conversation_context=conversation_context,
                ledger=ledger,
                selected_docs=latest_selected,
                graded=latest_graded,
                round_index=round_index,
                max_rounds=max_search_rounds,
                controller_hints=controller_hints,
            )
            if reflection.action == "retry":
                reflection_rounds_used += 1
                if reflection.followup_queries:
                    reflection_followup_queries = list(reflection.followup_queries)
                if reflection.prefer_doc_ids:
                    preferred_doc_ids = list(dict.fromkeys([*reflection.prefer_doc_ids, *preferred_doc_ids]))
                if reflection.full_read_doc_ids:
                    forced_full_read_doc_ids.update(reflection.full_read_doc_ids[:max_parallel_lanes])
                tool_call_log.append(f"round{round_index}:reflection:{reflection.rationale or 'retry'}")
                strategies.add("reflection")
                _emit_progress(
                    progress_emitter,
                    "decision_point",
                    label="Reflecting on coverage",
                    detail=reflection.rationale or "Retrying with refined evidence targets",
                    agent="rag_worker",
                    why="The first evidence wave did not cover enough of the question to answer confidently.",
                    docs=_doc_focus_from_documents(latest_selected or latest_candidates),
                )
                continue
        if sufficient:
            break

    candidate_counts = {
        "unique_chunks": len(ledger.entries),
        "unique_docs": len({_doc_id(doc) for doc in latest_candidates if _doc_id(doc)}),
        "graded_chunks": len(latest_graded),
        "selected_docs": len(latest_selected),
        "strong_chunks": sum(1 for item in latest_graded if item.relevance >= 2),
        "worker_tasks": len([item for item in tool_call_log if ":worker:" in item]),
        "graph_hits": sum(1 for item in ledger.entries.values() if item.strategy == "graph"),
        "sql_doc_hints": len(plan.preferred_doc_ids),
        "query_facets": len(ledger.facets),
        "covered_facets": sum(1 for item in ledger.facet_coverage.values() if str(item.get("state")) == "covered"),
        "missing_facets": sum(1 for item in ledger.facet_coverage.values() if str(item.get("state")) == "missing"),
    }
    _emit_progress(
        progress_emitter,
        "phase_end",
        label="Search complete",
        detail=f"{len(latest_selected)} evidence document(s)",
        agent="rag_worker",
        docs=_doc_focus_from_documents(latest_selected or latest_candidates),
        counts=dict(candidate_counts),
    )
    timing_fields = stage_timer.to_fields() if stage_timer is not None else {}
    return RetrievalRun(
        selected_docs=list(latest_selected),
        candidate_docs=list(latest_candidates or ledger.materialize_documents()),
        graded=list(latest_graded),
        query_used=query,
        search_mode="deep",
        rounds=max(1, len(ledger.round_summaries)),
        tool_calls_used=tool_calls,
        tool_call_log=tool_call_log,
        strategies_used=sorted(strategies),
        candidate_counts=candidate_counts,
        evidence_ledger=ledger.to_dict(),
        parallel_workers_used=parallel_workers_used,
        source_plan=plan.to_dict(),
        sources_used=list(plan.sources_chosen),
        graphs_considered=[str(item.get("graph_id") or "") for item in plan.graph_shortlist if str(item.get("graph_id") or "")],
        graph_methods_used=list(plan.graph_methods),
        sql_sources_used=list(plan.sql_views_used),
        resolution_stats={
            "graph_hits_resolved": sum(1 for item in ledger.entries.values() if item.strategy == "graph" and item.doc_id),
            "explicit_doc_targets": len(resolved_doc_ids),
            "sql_doc_hints": len(plan.preferred_doc_ids),
        },
        decomposition=decomposition,
        claim_ledger=ledger.to_dict(),
        verified_hops=list(ledger.verified_hops),
        stage_timings_ms=dict(timing_fields.get("stage_timings_ms") or {}),
        budget_ms=int(timing_fields.get("budget_ms") or 0),
        budget_exhausted=bool(timing_fields.get("budget_exhausted", False)),
        slow_stages=[str(item) for item in (timing_fields.get("slow_stages") or []) if str(item)],
    )


def run_retrieval_controller(
    settings: Any,
    stores: Any,
    *,
    providers: Any,
    session: Any,
    query: str,
    conversation_context: str,
    preferred_doc_ids: List[str],
    must_include_uploads: bool,
    top_k_vector: int,
    top_k_keyword: int,
    max_retries: int,
    callbacks: List[Any] | None = None,
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
) -> RetrievalRun:
    callbacks = list(callbacks or [])
    normalized_mode = _normalize_search_mode(search_mode)
    normalized_research_profile = normalize_research_profile(research_profile)
    normalized_coverage_goal = normalize_coverage_goal(coverage_goal)
    normalized_result_mode = normalize_result_mode(result_mode)
    resolved_controller_hints = coerce_controller_hints(controller_hints)
    stage_timer = _new_retrieval_budget(settings, resolved_controller_hints)
    effective_rounds = max(1, int(max_search_rounds or max(2, min(4, int(max_retries or 1) + 1))))
    existing_selection = dict(resolved_controller_hints.get("collection_selection") or {})
    explicit_collection_id = str(
        resolved_controller_hints.get("requested_kb_collection_id")
        or dict(getattr(session, "metadata", {}) or {}).get("requested_kb_collection_id")
        or ""
    ).strip()
    if not preferred_doc_ids and not str(existing_selection.get("selected_collection_id") or "").strip():
        with stage_timer.measure("collection_scope"):
            collection_selection = select_collection_for_query(
                stores,
                settings,
                session,
                query,
                source_type="kb",
                explicit_collection_id=explicit_collection_id,
                event_sink=event_sink,
            )
        if collection_selection.resolved:
            apply_selection_to_session(session, collection_selection)
        resolved_controller_hints = {
            **dict(resolved_controller_hints),
            "collection_selection": collection_selection.to_dict(),
        }
    elif str(existing_selection.get("selected_collection_id") or "").strip():
        metadata = dict(getattr(session, "metadata", {}) or {})
        selected_collection_id = str(existing_selection.get("selected_collection_id") or "").strip()
        metadata.update(
            {
                "kb_collection_id": selected_collection_id,
                "selected_kb_collection_id": selected_collection_id,
                "search_collection_ids": [selected_collection_id],
            }
        )
        try:
            session.metadata = metadata
        except Exception:
            pass
    with stage_timer.measure("source_planning"):
        source_plan = _plan_sources(
            query,
            settings=settings,
            stores=stores,
            session=session,
            controller_hints=resolved_controller_hints,
            collection_id=_resolve_collection_id(settings, session),
            preferred_doc_ids=preferred_doc_ids,
        )
    merged_controller_hints = {
        **dict(resolved_controller_hints),
        **dict(source_plan.to_controller_hints()),
    }
    preferred_doc_ids = list(dict.fromkeys([*preferred_doc_ids, *source_plan.preferred_doc_ids]))
    _emit_progress(
        progress_emitter,
        "phase_update",
        label="Planning sources",
        detail=", ".join(source_plan.sources_chosen) or "vector, keyword",
        agent="rag_worker",
        sources=list(source_plan.sources_chosen),
        graph_shortlist=[dict(item) for item in source_plan.graph_shortlist[:3]],
        graph_methods=list(source_plan.graph_methods),
        sql_views=list(source_plan.sql_views_used),
    )
    _emit_progress(
        progress_emitter,
        "phase_start",
        label="Searching knowledge base",
        detail=f"{normalized_mode} mode",
        agent="rag_worker",
        why="Starting with the fast path, then escalating only if the evidence is weak or the query is broad.",
    )

    fast_run = _build_fast_run(
        settings,
        stores,
        providers=providers,
        session=session,
        query=query,
        preferred_doc_ids=preferred_doc_ids,
        must_include_uploads=must_include_uploads,
        top_k_vector=top_k_vector,
        top_k_keyword=top_k_keyword,
        callbacks=callbacks,
        source_plan=source_plan,
        stage_timer=stage_timer,
    )
    if normalized_mode == "fast":
        _emit_progress(
            progress_emitter,
            "phase_end",
            label="Search complete",
            detail="Fast retrieval path",
            agent="rag_worker",
            docs=_doc_focus_from_documents(fast_run.selected_docs or fast_run.candidate_docs),
            counts=dict(fast_run.candidate_counts),
        )
        return fast_run
    if stage_timer.should_cutoff():
        _emit_progress(
            progress_emitter,
            "phase_end",
            label="Search complete",
            detail="Budget cutoff after fast path",
            agent="rag_worker",
            docs=_doc_focus_from_documents(fast_run.selected_docs or fast_run.candidate_docs),
            counts=dict(fast_run.candidate_counts),
        )
        fast_run.budget_exhausted = True
        fast_run.budget_ms = int(stage_timer.budget_ms)
        fast_run.stage_timings_ms = dict(stage_timer.stage_timings_ms)
        fast_run.slow_stages = stage_timer.slow_stages()
        return fast_run
    if normalized_mode == "deep" or _should_escalate(
        query,
        fast_run,
        settings,
        research_profile=normalized_research_profile,
        coverage_goal=normalized_coverage_goal,
        result_mode=normalized_result_mode,
        controller_hints=merged_controller_hints,
    ):
        _emit_progress(
            progress_emitter,
            "decision_point",
            label="Escalating search",
            detail="Evidence was weak or the query is complex",
            agent="rag_worker",
            why="A deeper search is safer than answering from the initial evidence set.",
        )
        _emit_progress(
            progress_emitter,
            "phase_update",
            label="Escalating search",
            detail="Evidence was weak or the query is complex",
            agent="rag_worker",
        )
        deep_run = _run_deep_path(
            settings,
            stores,
            providers=providers,
            session=session,
            query=query,
            conversation_context=conversation_context,
            preferred_doc_ids=preferred_doc_ids,
            must_include_uploads=must_include_uploads,
            top_k_vector=top_k_vector,
            top_k_keyword=top_k_keyword,
            max_search_rounds=effective_rounds,
            callbacks=callbacks,
            seed_run=fast_run,
            research_profile=normalized_research_profile,
            coverage_goal=normalized_coverage_goal,
            result_mode=normalized_result_mode,
            controller_hints=merged_controller_hints,
            runtime_bridge=runtime_bridge,
            progress_emitter=progress_emitter,
            allow_internal_fanout=allow_internal_fanout,
            source_plan=source_plan,
            stage_timer=stage_timer,
        )
        if bool(getattr(settings, "retrieval_quality_verifier_enabled", False)):
            from agentic_chatbot_next.rag.citations import build_citations

            with stage_timer.measure("citation_building"):
                verification_citations = build_citations(deep_run.selected_docs)
            with stage_timer.measure("retrieval_verification"):
                verification = verify_retrieval_quality(
                    settings=settings,
                    stores=stores,
                    session=session,
                    query=query,
                    retrieval_run=deep_run,
                    citations=verification_citations,
                    controller_hints=merged_controller_hints,
                )
            deep_run.retrieval_verification = verification
            deep_run.claim_ledger = deep_run.claim_ledger or dict(deep_run.evidence_ledger or {})
            if verification.get("retryable") and not merged_controller_hints.get("retrieval_verification_retry_used"):
                retry_focus = dict(verification.get("retry_focus") or {})
                retry_doc_ids = [str(item) for item in (retry_focus.get("doc_ids") or []) if str(item)]
                retry_queries = [str(item) for item in (retry_focus.get("queries") or []) if str(item)]
                retry_controller_hints = {
                    **dict(merged_controller_hints),
                    "retrieval_verification_retry_used": True,
                    "resolved_doc_ids": retry_doc_ids or merged_controller_hints.get("resolved_doc_ids") or [],
                    "prefer_doc_focus": bool(retry_doc_ids) or bool(merged_controller_hints.get("prefer_doc_focus")),
                    "entity_candidates": [
                        *[str(item.get("canonical_name") or "") for item in (deep_run.decomposition.get("canonical_entities") or []) if str(item.get("canonical_name") or "")],
                        *retry_queries,
                    ],
                }
                deep_run = _run_deep_path(
                    settings,
                    stores,
                    providers=providers,
                    session=session,
                    query=retry_queries[0] if retry_queries else query,
                    conversation_context=conversation_context,
                    preferred_doc_ids=retry_doc_ids or preferred_doc_ids,
                    must_include_uploads=must_include_uploads,
                    top_k_vector=top_k_vector,
                    top_k_keyword=top_k_keyword,
                    max_search_rounds=1,
                    callbacks=callbacks,
                    seed_run=deep_run,
                    research_profile=normalized_research_profile,
                    coverage_goal=normalized_coverage_goal,
                    result_mode=normalized_result_mode,
                    controller_hints=retry_controller_hints,
                    runtime_bridge=runtime_bridge,
                    progress_emitter=progress_emitter,
                    allow_internal_fanout=False,
                    source_plan=source_plan,
                    stage_timer=stage_timer,
                )
                with stage_timer.measure("citation_building"):
                    retry_citations = build_citations(deep_run.selected_docs)
                with stage_timer.measure("retrieval_verification"):
                    deep_run.retrieval_verification = verify_retrieval_quality(
                        settings=settings,
                        stores=stores,
                        session=session,
                        query=query,
                        retrieval_run=deep_run,
                        citations=retry_citations,
                        controller_hints=retry_controller_hints,
                    )
        timing_fields = stage_timer.to_fields()
        deep_run.stage_timings_ms = dict(timing_fields.get("stage_timings_ms") or {})
        deep_run.budget_ms = int(timing_fields.get("budget_ms") or 0)
        deep_run.budget_exhausted = bool(timing_fields.get("budget_exhausted", False))
        deep_run.slow_stages = [str(item) for item in (timing_fields.get("slow_stages") or []) if str(item)]
        return deep_run
    _emit_progress(
        progress_emitter,
        "phase_end",
        label="Search complete",
        detail="Adaptive fast path",
        agent="rag_worker",
        docs=_doc_focus_from_documents(fast_run.selected_docs or fast_run.candidate_docs),
        counts=dict(fast_run.candidate_counts),
    )
    return fast_run


__all__ = [
    "CorpusRetrievalAdapter",
    "EvidenceLedger",
    "RetrievalRun",
    "SearchFilters",
    "run_retrieval_controller",
]
