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


@dataclass
class TabularEvidenceTask:
    task_id: str
    query: str
    doc_id: str
    title: str
    source_path: str = ""
    file_type: str = ""
    sheet_hints: List[str] = field(default_factory=list)
    cell_ranges: List[str] = field(default_factory=list)
    row_hints: List[Dict[str, Any]] = field(default_factory=list)
    requested_operations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "query": self.query,
            "doc_id": self.doc_id,
            "title": self.title,
            "source_path": self.source_path,
            "file_type": self.file_type,
            "sheet_hints": list(self.sheet_hints),
            "cell_ranges": list(self.cell_ranges),
            "row_hints": [dict(item) for item in self.row_hints],
            "requested_operations": list(self.requested_operations),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TabularEvidenceTask":
        return cls(
            task_id=str(raw.get("task_id") or ""),
            query=str(raw.get("query") or ""),
            doc_id=str(raw.get("doc_id") or ""),
            title=str(raw.get("title") or ""),
            source_path=str(raw.get("source_path") or ""),
            file_type=str(raw.get("file_type") or ""),
            sheet_hints=[str(item) for item in (raw.get("sheet_hints") or []) if str(item)],
            cell_ranges=[str(item) for item in (raw.get("cell_ranges") or []) if str(item)],
            row_hints=[dict(item) for item in (raw.get("row_hints") or []) if isinstance(item, dict)],
            requested_operations=[str(item) for item in (raw.get("requested_operations") or []) if str(item)],
        )


@dataclass
class TabularEvidenceResult:
    task_id: str
    status: str = "ok"
    summary: str = ""
    findings: List[Dict[str, Any]] = field(default_factory=list)
    source_refs: List[Dict[str, Any]] = field(default_factory=list)
    operations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "summary": self.summary,
            "findings": [dict(item) for item in self.findings],
            "source_refs": [dict(item) for item in self.source_refs],
            "operations": list(self.operations),
            "warnings": list(self.warnings),
            "confidence": float(self.confidence),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TabularEvidenceResult":
        try:
            confidence = float(raw.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        findings: List[Dict[str, Any]] = []
        for item in raw.get("findings") or []:
            if isinstance(item, dict):
                findings.append(dict(item))
            elif str(item).strip():
                findings.append({"text": str(item).strip()})
        return cls(
            task_id=str(raw.get("task_id") or ""),
            status=str(raw.get("status") or "ok"),
            summary=str(raw.get("summary") or ""),
            findings=findings,
            source_refs=[dict(item) for item in (raw.get("source_refs") or []) if isinstance(item, dict)],
            operations=[str(item) for item in (raw.get("operations") or []) if str(item)],
            warnings=[str(item) for item in (raw.get("warnings") or []) if str(item)],
            confidence=confidence,
        )


@dataclass
class TabularEvidenceBatchResult:
    results: List[TabularEvidenceResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class RagRuntimeBridge(Protocol):
    def can_run_parallel(self, *, task_count: int) -> bool:
        ...

    def run_search_tasks(self, tasks: Sequence[RagSearchTask]) -> RagSearchBatchResult:
        ...

    def run_tabular_evidence_tasks(self, tasks: Sequence[TabularEvidenceTask]) -> TabularEvidenceBatchResult:
        ...
