from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from agentic_chatbot_next.documents.models import DocumentExtractResult

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - rapidfuzz is a declared dependency, but keep fallback deterministic.
    fuzz = None


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_NUMBERED_STEP_RE = re.compile(r"^\s*(?:\d+[.)]|\([a-zA-Z0-9]+\)|[-*])\s+")
_OBLIGATION_RE = re.compile(
    r"\b(shall|must|required|responsible\s+for|agrees?\s+to|will|may\s+not|prohibited)\b",
    re.IGNORECASE,
)
_PROCESS_RE = re.compile(
    r"\b("
    r"approv\w*|approval|review\w*|submit\w*|rout\w*|routing|escalat\w*|handoff|hand\s*off|"
    r"notif\w*|validat\w*|verif\w*|receiv\w*|initiat\w*|complet\w*|clos\w*|archiv\w*|retain\w*|"
    r"intake|screen\w*|ticket\w*|packet\w*|handover|transfer\w*|assign\w*|"
    r"workflow|process|procedure|step|before|after|then|next|decision|"
    r"owner|responsible|raci|input|output|form|ticket|request"
    r")\b",
    re.IGNORECASE,
)
_CORPORATE_BOILERPLATE_TERMS = {
    "company",
    "corporate",
    "confidential",
    "document",
    "documents",
    "policy",
    "procedure",
    "procedures",
    "standard",
    "standards",
    "business",
    "operations",
    "operational",
    "organization",
    "department",
    "division",
    "sector",
    "enterprise",
    "shall",
    "must",
    "will",
    "may",
    "required",
    "page",
    "section",
    "revision",
    "version",
}
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "are",
    "was",
    "were",
    "been",
    "have",
    "has",
    "not",
    "into",
    "upon",
    "when",
    "where",
    "which",
    "their",
    "there",
    "these",
    "those",
    "each",
    "such",
    "within",
    "without",
    "through",
    "between",
}


@dataclass
class DocumentFingerprint:
    doc_id: str
    title: str
    sector: str
    content_terms: Set[str] = field(default_factory=set)
    section_terms: Set[str] = field(default_factory=set)
    table_terms: Set[str] = field(default_factory=set)
    obligation_terms: Set[str] = field(default_factory=set)
    title_terms: Set[str] = field(default_factory=set)
    process_steps: List[str] = field(default_factory=list)
    process_terms: Set[str] = field(default_factory=set)


@dataclass
class SimilarityScore:
    content_overlap_score: float
    process_flow_score: float
    section_structure_score: float
    table_schema_score: float
    obligation_overlap_score: float
    metadata_title_score: float
    consolidation_score: float
    reason_codes: List[str]
    shared_terms: List[str]
    matched_left_steps: List[str] = field(default_factory=list)
    matched_right_steps: List[str] = field(default_factory=list)


def _tokens(text: str, *, suppressed_terms: set[str] | None = None) -> set[str]:
    suppressed = set(suppressed_terms or set())
    values = {
        token.casefold().replace("_", "-")
        for token in _TOKEN_RE.findall(str(text or ""))
    }
    return {
        token
        for token in values
        if token not in _STOPWORDS
        and token not in _CORPORATE_BOILERPLATE_TERMS
        and token not in suppressed
        and not token.isdigit()
    }


def _ngram_terms(tokens: Iterable[str], *, sizes: tuple[int, ...] = (2, 3)) -> set[str]:
    ordered = [token for token in tokens if token]
    terms = set(ordered)
    for size in sizes:
        for index in range(0, max(0, len(ordered) - size + 1)):
            terms.add(" ".join(ordered[index : index + size]))
    return terms


def _weighted_jaccard(left: set[str], right: set[str], idf: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    intersection = left & right
    numerator = sum(float(idf.get(term, 1.0)) for term in intersection)
    denominator = sum(float(idf.get(term, 1.0)) for term in union)
    return round(numerator / denominator, 4) if denominator else 0.0


def _ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    left_text = " ".join(sorted(_tokens(left)))
    right_text = " ".join(sorted(_tokens(right)))
    if not left_text or not right_text:
        return 0.0
    if fuzz is not None:
        return round(float(fuzz.token_set_ratio(left_text, right_text)) / 100.0, 4)
    left_terms = set(left_text.split())
    right_terms = set(right_text.split())
    return _weighted_jaccard(left_terms, right_terms, {})


def _process_step_texts(result: DocumentExtractResult) -> list[str]:
    rows: list[str] = []
    pending_marker = False
    for element in result.elements:
        haystack = " ".join([element.element_type, element.section_title, element.text])
        text = str(element.text or "").strip()
        if re.fullmatch(r"\d+[.)]", text):
            pending_marker = True
            continue
        if _NUMBERED_STEP_RE.search(text) or pending_marker or _PROCESS_RE.search(haystack):
            rows.append(element.text)
            pending_marker = False
    for table in result.tables:
        headers = " ".join(table.columns)
        if _PROCESS_RE.search(headers) or _PROCESS_RE.search(table.title):
            for row in table.rows[:80]:
                text = "; ".join(str(cell) for cell in row if str(cell).strip())
                if text:
                    rows.append(text)
    return rows[:80]


def build_suppressed_terms(
    results: Sequence[DocumentExtractResult],
    *,
    extra_terms: Iterable[str] = (),
    common_threshold: float = 0.72,
) -> set[str]:
    doc_count = max(1, len(results))
    frequencies: Counter[str] = Counter()
    for result in results:
        doc_terms = set()
        for element in result.elements:
            doc_terms.update(_tokens(element.text, suppressed_terms=set()))
        doc_terms.update(_tokens(result.document.title, suppressed_terms=set()))
        frequencies.update(doc_terms)
    suppressed = set(_CORPORATE_BOILERPLATE_TERMS)
    suppressed.update(token.casefold() for token in extra_terms if str(token).strip())
    for term, count in frequencies.items():
        if count / doc_count >= float(common_threshold):
            suppressed.add(term)
    return suppressed


def build_idf(results: Sequence[DocumentExtractResult], *, suppressed_terms: set[str]) -> Dict[str, float]:
    doc_count = max(1, len(results))
    frequencies: Counter[str] = Counter()
    for result in results:
        doc_terms: set[str] = set()
        for element in result.elements:
            doc_terms.update(_tokens(element.text, suppressed_terms=suppressed_terms))
        for section in result.sections:
            doc_terms.update(_tokens(section.title, suppressed_terms=suppressed_terms))
        frequencies.update(doc_terms)
    return {
        term: 1.0 + math.log((doc_count + 1.0) / (count + 1.0))
        for term, count in frequencies.items()
        if term not in suppressed_terms
    }


def fingerprint_document(
    result: DocumentExtractResult,
    *,
    sector: str,
    suppressed_terms: set[str],
) -> DocumentFingerprint:
    content_tokens: list[str] = []
    obligation_tokens: set[str] = set()
    for element in result.elements:
        tokens = _tokens(element.text, suppressed_terms=suppressed_terms)
        content_tokens.extend(sorted(tokens))
        if _OBLIGATION_RE.search(element.text):
            obligation_tokens.update(tokens)
    section_terms = set()
    for section in result.sections:
        section_terms.update(_tokens(section.title, suppressed_terms=suppressed_terms))
    table_terms = set()
    for table in result.tables:
        table_terms.update(_tokens(" ".join([table.title, " ".join(table.columns)]), suppressed_terms=suppressed_terms))
    process_steps = _process_step_texts(result)
    process_terms = set()
    for step in process_steps:
        process_terms.update(_tokens(step, suppressed_terms=suppressed_terms))
    title_terms = _tokens(result.document.title, suppressed_terms=suppressed_terms)
    return DocumentFingerprint(
        doc_id=result.document.doc_id,
        title=result.document.title,
        sector=sector,
        content_terms=_ngram_terms(content_tokens),
        section_terms=section_terms,
        table_terms=table_terms,
        obligation_terms=obligation_tokens,
        title_terms=title_terms,
        process_steps=process_steps,
        process_terms=_ngram_terms(sorted(process_terms), sizes=(2,)),
    )


def compare_fingerprints(
    left: DocumentFingerprint,
    right: DocumentFingerprint,
    *,
    idf: Dict[str, float],
    similarity_focus: str = "auto",
    cross_sector: bool = False,
) -> SimilarityScore:
    content = _weighted_jaccard(left.content_terms, right.content_terms, idf)
    process = _weighted_jaccard(left.process_terms, right.process_terms, idf)
    section = _weighted_jaccard(left.section_terms, right.section_terms, idf)
    table = _weighted_jaccard(left.table_terms, right.table_terms, idf)
    obligations = _weighted_jaccard(left.obligation_terms, right.obligation_terms, idf)
    title = _ratio(left.title, right.title)
    matched_left, matched_right, step_ratio = _matched_process_steps(left.process_steps, right.process_steps)
    process = max(process, step_ratio)
    weights = _weights_for_focus(similarity_focus)
    score = (
        content * weights["content"]
        + process * weights["process"]
        + section * weights["section"]
        + table * weights["table"]
        + obligations * weights["obligations"]
        + title * weights["title"]
    )
    if process >= 0.72:
        score = max(score, (process * 0.9) + (max(content, section, obligations) * 0.1))
    if content >= 0.78:
        score = max(score, (content * 0.82) + (section * 0.12) + (title * 0.06))
    if table >= 0.75:
        score = max(score, (table * 0.72) + (content * 0.18) + (section * 0.1))
    if cross_sector and score < 0.9:
        score *= 0.97 if process >= 0.8 else 0.93
    reason_codes = _reason_codes(
        content=content,
        process=process,
        section=section,
        table=table,
        obligations=obligations,
        title=title,
    )
    shared_terms = sorted(
        (left.content_terms & right.content_terms) | (left.process_terms & right.process_terms),
        key=lambda term: (-float(idf.get(term, 1.0)), term),
    )[:20]
    return SimilarityScore(
        content_overlap_score=round(content, 4),
        process_flow_score=round(process, 4),
        section_structure_score=round(section, 4),
        table_schema_score=round(table, 4),
        obligation_overlap_score=round(obligations, 4),
        metadata_title_score=round(title, 4),
        consolidation_score=round(score, 4),
        reason_codes=reason_codes,
        shared_terms=shared_terms,
        matched_left_steps=matched_left,
        matched_right_steps=matched_right,
    )


def _matched_process_steps(left_steps: Sequence[str], right_steps: Sequence[str]) -> tuple[list[str], list[str], float]:
    if not left_steps or not right_steps:
        return [], [], 0.0
    pairs: list[tuple[float, str, str]] = []
    for left in left_steps:
        for right in right_steps:
            score = _ratio(left, right)
            if score >= 0.58:
                pairs.append((score, left, right))
    pairs.sort(reverse=True, key=lambda item: item[0])
    used_left: set[str] = set()
    used_right: set[str] = set()
    matched_left: list[str] = []
    matched_right: list[str] = []
    for score, left, right in pairs:
        del score
        if left in used_left or right in used_right:
            continue
        used_left.add(left)
        used_right.add(right)
        matched_left.append(left)
        matched_right.append(right)
        if len(matched_left) >= 8:
            break
    denominator = max(len(left_steps), len(right_steps), 1)
    coverage = len(matched_left) / denominator
    best_average = sum(item[0] for item in pairs[: max(1, len(matched_left))]) / max(1, len(matched_left)) if matched_left else 0.0
    return matched_left, matched_right, round(min(1.0, (coverage * 0.55) + (best_average * 0.45)), 4)


def _weights_for_focus(similarity_focus: str) -> Dict[str, float]:
    normalized = str(similarity_focus or "auto").strip().lower()
    if normalized == "process_flows":
        return {"content": 0.2, "process": 0.45, "section": 0.15, "table": 0.1, "obligations": 0.05, "title": 0.05}
    if normalized == "tables":
        return {"content": 0.2, "process": 0.15, "section": 0.1, "table": 0.4, "obligations": 0.05, "title": 0.1}
    if normalized == "requirements":
        return {"content": 0.25, "process": 0.15, "section": 0.1, "table": 0.05, "obligations": 0.4, "title": 0.05}
    if normalized == "policies":
        return {"content": 0.42, "process": 0.15, "section": 0.2, "table": 0.05, "obligations": 0.12, "title": 0.06}
    if normalized == "full_text":
        return {"content": 0.5, "process": 0.15, "section": 0.15, "table": 0.05, "obligations": 0.1, "title": 0.05}
    return {"content": 0.32, "process": 0.28, "section": 0.14, "table": 0.08, "obligations": 0.1, "title": 0.08}


def _reason_codes(
    *,
    content: float,
    process: float,
    section: float,
    table: float,
    obligations: float,
    title: float,
) -> List[str]:
    reasons: list[str] = []
    if process >= 0.52:
        reasons.append("shared_process_flow")
    if content >= 0.72:
        reasons.append("duplicated_policy_text")
    if table >= 0.58:
        reasons.append("same_table_schema")
    if obligations >= 0.55:
        reasons.append("overlapping_requirements")
    if content >= 0.82 and section >= 0.45:
        reasons.append("near_duplicate")
    if title >= 0.8 and content >= 0.35:
        reasons.append("same_version_family")
    return reasons or ["general_operational_overlap"]


__all__ = [
    "DocumentFingerprint",
    "SimilarityScore",
    "build_idf",
    "build_suppressed_terms",
    "compare_fingerprints",
    "fingerprint_document",
]
