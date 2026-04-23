from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

from agentic_chatbot_next.contracts.rag import RagContract

DEFENSE_COLLECTION_ID = "defense-rag-test"

_AUTHORITY_HINTS = re.compile(r"\b(current|latest|approved|authoritative|final|draft)\b", re.IGNORECASE)
_ENTITY_CONFUSION_HINTS = (
    ("north coast systems llc", "northcoast signal labs"),
    ("halcyon foundry", "halcyon microdevices"),
)


@dataclass(frozen=True)
class DefenseBenchmarkQuestion:
    question_id: str
    difficulty: str
    question_text: str
    expected_answer: str
    source_documents: tuple[str, ...]
    supporting_references: tuple[str, ...]
    rationale: str = ""


@dataclass
class DefenseBenchmarkResult:
    question_id: str
    difficulty: str
    answer_correct: bool
    citation_present: bool
    citation_source_match: bool
    authority_version_correct: bool
    diagnostics: List[str] = field(default_factory=list)
    answer: str = ""
    expected_answer: str = ""
    matched_citation_titles: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "difficulty": self.difficulty,
            "answer_correct": self.answer_correct,
            "citation_present": self.citation_present,
            "citation_source_match": self.citation_source_match,
            "authority_version_correct": self.authority_version_correct,
            "diagnostics": list(self.diagnostics),
            "answer": self.answer,
            "expected_answer": self.expected_answer,
            "matched_citation_titles": list(self.matched_citation_titles),
        }


@dataclass
class DefenseBenchmarkReport:
    collection_id: str
    total_questions: int
    results: List[DefenseBenchmarkResult] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        by_difficulty: Dict[str, Dict[str, int]] = {}
        for result in self.results:
            bucket = by_difficulty.setdefault(
                result.difficulty,
                {"total": 0, "answer_correct": 0, "citation_source_match": 0},
            )
            bucket["total"] += 1
            bucket["answer_correct"] += int(result.answer_correct)
            bucket["citation_source_match"] += int(result.citation_source_match)
        return {
            "collection_id": self.collection_id,
            "total_questions": self.total_questions,
            "answered_correctly": sum(int(item.answer_correct) for item in self.results),
            "with_source_match": sum(int(item.citation_source_match) for item in self.results),
            "with_citations": sum(int(item.citation_present) for item in self.results),
            "by_difficulty": by_difficulty,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "results": [item.to_dict() for item in self.results],
        }


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    stripped = re.sub(r"[^a-z0-9]+", " ", ascii_text.casefold())
    return " ".join(stripped.split())


def _basename(value: str) -> str:
    return Path(str(value or "")).name


def _normalize_documents(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        parts = [item.strip() for item in raw.split("|") if item.strip()]
    else:
        parts = [str(item).strip() for item in (raw or []) if str(item).strip()]
    return tuple(parts)


def _normalize_references(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        parts = [item.strip() for item in raw.split("|") if item.strip()]
    else:
        parts = [str(item).strip() for item in (raw or []) if str(item).strip()]
    return tuple(parts)


def load_defense_answer_key(path: Path) -> List[DefenseBenchmarkQuestion]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Defense answer key must be a list of questions.")
    questions: List[DefenseBenchmarkQuestion] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        questions.append(
            DefenseBenchmarkQuestion(
                question_id=str(item.get("question_id") or ""),
                difficulty=str(item.get("difficulty") or ""),
                question_text=str(item.get("question_text") or ""),
                expected_answer=str(item.get("expected_answer") or ""),
                source_documents=_normalize_documents(item.get("source_documents") or []),
                supporting_references=_normalize_references(item.get("supporting_references") or []),
                rationale=str(item.get("rationale") or ""),
            )
        )
    return questions


def _answer_matches_expected(answer: str, expected: str) -> bool:
    normalized_answer = _normalize_text(answer)
    normalized_expected = _normalize_text(expected)
    if not normalized_answer or not normalized_expected:
        return False
    if normalized_expected in normalized_answer:
        return True
    expected_tokens = [token for token in normalized_expected.split() if len(token) >= 3 or token.isdigit()]
    return bool(expected_tokens) and all(token in normalized_answer for token in expected_tokens)


def _expected_source_names(question: DefenseBenchmarkQuestion) -> set[str]:
    return {_normalize_text(_basename(item)) for item in question.source_documents}


def _citation_titles(contract: RagContract) -> List[str]:
    return [str(item.title or "") for item in contract.citations]


def _matching_citation_titles(question: DefenseBenchmarkQuestion, contract: RagContract) -> List[str]:
    expected = _expected_source_names(question)
    matched: List[str] = []
    for title in _citation_titles(contract):
        if _normalize_text(_basename(title)) in expected:
            matched.append(title)
    return matched


def _authority_sensitive(question: DefenseBenchmarkQuestion) -> bool:
    return bool(_AUTHORITY_HINTS.search(question.question_text))


def _workbook_expected(question: DefenseBenchmarkQuestion) -> bool:
    return any(str(item).lower().endswith((".xlsx", ".xls")) for item in question.source_documents)


def _entity_confusion_possible(question: DefenseBenchmarkQuestion) -> bool:
    normalized = _normalize_text(question.question_text)
    return any(any(alias in normalized for alias in group) for group in _ENTITY_CONFUSION_HINTS)


def evaluate_defense_contract(question: DefenseBenchmarkQuestion, contract: RagContract) -> DefenseBenchmarkResult:
    answer_correct = _answer_matches_expected(contract.answer, question.expected_answer)
    citation_present = bool(contract.citations)
    matched_titles = _matching_citation_titles(question, contract)
    citation_source_match = bool(matched_titles)
    authority_version_correct = citation_source_match if _authority_sensitive(question) else True

    diagnostics: List[str] = []
    if not citation_present:
        diagnostics.append("missing_citations")
    if _workbook_expected(question) and not any(
        _normalize_text(_basename(title)).endswith(("xlsx", "xls"))
        for title in matched_titles
    ):
        diagnostics.append("workbook_not_retrieved")
    if _authority_sensitive(question) and not authority_version_correct:
        diagnostics.append("wrong_source_authority")
    elif citation_present and not citation_source_match:
        diagnostics.append("wrong_source_document_match")
    if not answer_correct:
        if _entity_confusion_possible(question):
            diagnostics.append("entity_confusion")
        elif len(question.source_documents) >= 2:
            diagnostics.append("insufficient_multi_hop_synthesis")
        else:
            diagnostics.append("relationship_missed")

    return DefenseBenchmarkResult(
        question_id=question.question_id,
        difficulty=question.difficulty,
        answer_correct=answer_correct,
        citation_present=citation_present,
        citation_source_match=citation_source_match,
        authority_version_correct=authority_version_correct,
        diagnostics=diagnostics,
        answer=contract.answer,
        expected_answer=question.expected_answer,
        matched_citation_titles=matched_titles,
    )


def run_defense_benchmark(
    questions: Sequence[DefenseBenchmarkQuestion],
    *,
    answer_fn: Callable[[DefenseBenchmarkQuestion], RagContract],
    collection_id: str = DEFENSE_COLLECTION_ID,
) -> DefenseBenchmarkReport:
    results = [evaluate_defense_contract(question, answer_fn(question)) for question in questions]
    return DefenseBenchmarkReport(
        collection_id=collection_id,
        total_questions=len(list(questions)),
        results=results,
    )


__all__ = [
    "DEFENSE_COLLECTION_ID",
    "DefenseBenchmarkQuestion",
    "DefenseBenchmarkReport",
    "DefenseBenchmarkResult",
    "evaluate_defense_contract",
    "load_defense_answer_key",
    "run_defense_benchmark",
]
