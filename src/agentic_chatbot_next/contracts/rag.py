from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class Citation:
    citation_id: str
    doc_id: str
    title: str
    source_type: str
    location: str
    snippet: str
    collection_id: str = ""
    url: str = ""
    source_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Citation":
        return cls(
            citation_id=str(raw.get("citation_id") or ""),
            doc_id=str(raw.get("doc_id") or ""),
            title=str(raw.get("title") or ""),
            source_type=str(raw.get("source_type") or ""),
            location=str(raw.get("location") or ""),
            snippet=str(raw.get("snippet") or ""),
            collection_id=str(raw.get("collection_id") or ""),
            url=str(raw.get("url") or ""),
            source_path=str(raw.get("source_path") or ""),
        )


@dataclass
class RetrievalSummary:
    query_used: str
    steps: int = 0
    tool_calls_used: int = 0
    tool_call_log: List[str] = field(default_factory=list)
    citations_found: int = 0
    search_mode: str = "fast"
    rounds: int = 0
    strategies_used: List[str] = field(default_factory=list)
    candidate_counts: Dict[str, int] = field(default_factory=dict)
    parallel_workers_used: bool = False
    sources_used: List[str] = field(default_factory=list)
    source_plan: Dict[str, Any] = field(default_factory=dict)
    graphs_considered: List[str] = field(default_factory=list)
    graph_methods_used: List[str] = field(default_factory=list)
    sql_sources_used: List[str] = field(default_factory=list)
    resolution_stats: Dict[str, int] = field(default_factory=dict)
    decomposition: Dict[str, Any] = field(default_factory=dict)
    claim_ledger: Dict[str, Any] = field(default_factory=dict)
    verified_hops: List[str] = field(default_factory=list)
    retrieval_verification: Dict[str, Any] = field(default_factory=dict)
    workbook_profiles: Dict[str, Any] = field(default_factory=dict)
    status_extractors: Dict[str, Any] = field(default_factory=dict)
    stage_timings_ms: Dict[str, float] = field(default_factory=dict)
    budget_ms: int = 0
    budget_exhausted: bool = False
    slow_stages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RetrievalSummary":
        return cls(
            query_used=str(raw.get("query_used") or ""),
            steps=int(raw.get("steps") or 0),
            tool_calls_used=int(raw.get("tool_calls_used") or 0),
            tool_call_log=[str(item) for item in (raw.get("tool_call_log") or []) if str(item)],
            citations_found=int(raw.get("citations_found") or 0),
            search_mode=str(raw.get("search_mode") or "fast"),
            rounds=int(raw.get("rounds") or 0),
            strategies_used=[str(item) for item in (raw.get("strategies_used") or []) if str(item)],
            candidate_counts={
                str(key): int(value or 0)
                for key, value in dict(raw.get("candidate_counts") or {}).items()
                if str(key)
            },
            parallel_workers_used=bool(raw.get("parallel_workers_used", False)),
            sources_used=[str(item) for item in (raw.get("sources_used") or []) if str(item)],
            source_plan=dict(raw.get("source_plan") or {}),
            graphs_considered=[str(item) for item in (raw.get("graphs_considered") or []) if str(item)],
            graph_methods_used=[str(item) for item in (raw.get("graph_methods_used") or []) if str(item)],
            sql_sources_used=[str(item) for item in (raw.get("sql_sources_used") or []) if str(item)],
            resolution_stats={
                str(key): int(value or 0)
                for key, value in dict(raw.get("resolution_stats") or {}).items()
                if str(key)
            },
            decomposition=dict(raw.get("decomposition") or {}),
            claim_ledger=dict(raw.get("claim_ledger") or {}),
            verified_hops=[str(item) for item in (raw.get("verified_hops") or []) if str(item)],
            retrieval_verification=dict(raw.get("retrieval_verification") or {}),
            workbook_profiles=dict(raw.get("workbook_profiles") or {}),
            status_extractors=dict(raw.get("status_extractors") or {}),
            stage_timings_ms={
                str(key): float(value or 0.0)
                for key, value in dict(raw.get("stage_timings_ms") or {}).items()
                if str(key)
            },
            budget_ms=int(raw.get("budget_ms") or 0),
            budget_exhausted=bool(raw.get("budget_exhausted", False)),
            slow_stages=[str(item) for item in (raw.get("slow_stages") or []) if str(item)],
        )


@dataclass
class RagContract:
    answer: str
    citations: List[Citation] = field(default_factory=list)
    used_citation_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    retrieval_summary: RetrievalSummary = field(default_factory=lambda: RetrievalSummary(query_used=""))
    followups: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [item.to_dict() for item in self.citations],
            "used_citation_ids": list(self.used_citation_ids),
            "confidence": float(self.confidence),
            "retrieval_summary": self.retrieval_summary.to_dict(),
            "followups": list(self.followups),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RagContract":
        return cls(
            answer=str(raw.get("answer") or ""),
            citations=[Citation.from_dict(dict(item)) for item in (raw.get("citations") or []) if isinstance(item, dict)],
            used_citation_ids=[str(item) for item in (raw.get("used_citation_ids") or []) if str(item)],
            confidence=float(raw.get("confidence") or 0.0),
            retrieval_summary=RetrievalSummary.from_dict(dict(raw.get("retrieval_summary") or {})),
            followups=[str(item) for item in (raw.get("followups") or []) if str(item)],
            warnings=[str(item) for item in (raw.get("warnings") or []) if str(item)],
        )
