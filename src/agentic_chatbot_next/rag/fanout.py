from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Sequence

from langchain_core.documents import Document

from agentic_chatbot_next.rag.retrieval import GradedChunk


def serialize_document(doc: Document) -> Dict[str, Any]:
    return {
        "page_content": str(doc.page_content or ""),
        "metadata": dict(doc.metadata or {}),
    }


def deserialize_document(raw: Dict[str, Any]) -> Document:
    return Document(
        page_content=str(raw.get("page_content") or ""),
        metadata=dict(raw.get("metadata") or {}),
    )


def serialize_graded_chunk(item: GradedChunk) -> Dict[str, Any]:
    return {
        "doc": serialize_document(item.doc),
        "relevance": int(item.relevance),
        "reason": str(item.reason or ""),
    }


def deserialize_graded_chunk(raw: Dict[str, Any]) -> GradedChunk:
    return GradedChunk(
        doc=deserialize_document(dict(raw.get("doc") or {})),
        relevance=int(raw.get("relevance") or 0),
        reason=str(raw.get("reason") or ""),
    )


@dataclass
class RagSearchTask:
    task_id: str
    title: str
    query: str
    doc_scope: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    strategies: List[str] = field(default_factory=list)
    round_budget: int = 1
    answer_mode: str = "evidence_only"
    research_profile: str = ""
    coverage_goal: str = ""
    result_mode: str = ""
    controller_hints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "query": self.query,
            "doc_scope": list(self.doc_scope),
            "filters": dict(self.filters),
            "strategies": list(self.strategies),
            "round_budget": int(self.round_budget),
            "answer_mode": self.answer_mode,
            "research_profile": self.research_profile,
            "coverage_goal": self.coverage_goal,
            "result_mode": self.result_mode,
            "controller_hints": dict(self.controller_hints),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RagSearchTask":
        return cls(
            task_id=str(raw.get("task_id") or raw.get("id") or ""),
            title=str(raw.get("title") or ""),
            query=str(raw.get("query") or ""),
            doc_scope=[str(item) for item in (raw.get("doc_scope") or []) if str(item)],
            filters=dict(raw.get("filters") or {}),
            strategies=[str(item) for item in (raw.get("strategies") or []) if str(item)],
            round_budget=max(1, int(raw.get("round_budget") or 1)),
            answer_mode=str(raw.get("answer_mode") or "evidence_only"),
            research_profile=str(raw.get("research_profile") or ""),
            coverage_goal=str(raw.get("coverage_goal") or ""),
            result_mode=str(raw.get("result_mode") or ""),
            controller_hints=dict(raw.get("controller_hints") or {}),
        )


@dataclass
class RagSearchTaskResult:
    task_id: str
    evidence_entries: List[Dict[str, Any]] = field(default_factory=list)
    candidate_docs: List[Dict[str, Any]] = field(default_factory=list)
    graded_chunks: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    doc_focus: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "evidence_entries": [dict(item) for item in self.evidence_entries],
            "candidate_docs": [dict(item) for item in self.candidate_docs],
            "graded_chunks": [dict(item) for item in self.graded_chunks],
            "warnings": list(self.warnings),
            "doc_focus": [dict(item) for item in self.doc_focus],
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RagSearchTaskResult":
        return cls(
            task_id=str(raw.get("task_id") or ""),
            evidence_entries=[dict(item) for item in (raw.get("evidence_entries") or []) if isinstance(item, dict)],
            candidate_docs=[dict(item) for item in (raw.get("candidate_docs") or []) if isinstance(item, dict)],
            graded_chunks=[dict(item) for item in (raw.get("graded_chunks") or []) if isinstance(item, dict)],
            warnings=[str(item) for item in (raw.get("warnings") or []) if str(item)],
            doc_focus=[dict(item) for item in (raw.get("doc_focus") or []) if isinstance(item, dict)],
        )


@dataclass
class RagSearchBatchResult:
    results: List[RagSearchTaskResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    parallel_workers_used: bool = False


class RagRuntimeBridge(Protocol):
    def can_run_parallel(self, *, task_count: int) -> bool:
        ...

    def run_search_tasks(self, tasks: Sequence[RagSearchTask]) -> RagSearchBatchResult:
        ...
