from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class BinderSourceDocument:
    doc_id: str
    title: str
    source_type: str = ""
    source_path: str = ""
    collection_id: str = ""
    file_type: str = ""
    content_hash: str = ""
    parser_path: str = ""
    warnings: List[str] = field(default_factory=list)
    classification_markings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BinderArtifact:
    artifact_ref: str
    filename: str
    label: str = ""
    content_type: str = ""
    size_bytes: int = 0
    content_hash: str = ""
    source: str = ""
    included_in_zip: bool = False
    missing_file: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BinderHandoffArtifact:
    artifact_id: str
    artifact_ref: str
    artifact_type: str
    handoff_schema: str = ""
    producer_task_id: str = ""
    producer_agent: str = ""
    summary: str = ""
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceRow:
    evidence_id: str
    claim: str
    source_excerpt: str = ""
    source_document: str = ""
    source_location: str = ""
    source_hash: str = ""
    producing_tool: str = ""
    artifact_ref: str = ""
    artifact_filename: str = ""
    citation_status: str = "missing"
    review_status: str = "needs_review"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BinderOpenIssue:
    issue_id: str
    severity: str
    message: str
    related_artifact: str = ""
    related_evidence_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceBinderResult:
    binder_id: str
    status: str
    binder_title: str
    objective: str = ""
    generated_at: str = ""
    source_documents: List[BinderSourceDocument] = field(default_factory=list)
    artifacts: List[BinderArtifact] = field(default_factory=list)
    handoff_artifacts: List[BinderHandoffArtifact] = field(default_factory=list)
    evidence_rows: List[EvidenceRow] = field(default_factory=list)
    open_issues: List[BinderOpenIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    binder_artifacts: List[Dict[str, Any]] = field(default_factory=list)
    background_job_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "binder_id": self.binder_id,
            "status": self.status,
            "binder_title": self.binder_title,
            "objective": self.objective,
            "generated_at": self.generated_at,
            "source_documents": [item.to_dict() for item in self.source_documents],
            "artifacts": [item.to_dict() for item in self.artifacts],
            "handoff_artifacts": [item.to_dict() for item in self.handoff_artifacts],
            "evidence_rows": [item.to_dict() for item in self.evidence_rows],
            "open_issues": [item.to_dict() for item in self.open_issues],
            "warnings": list(self.warnings),
            "binder_artifacts": list(self.binder_artifacts),
            "background_job_id": self.background_job_id,
        }

    def compact(self) -> Dict[str, Any]:
        return {
            "binder_id": self.binder_id,
            "status": self.status,
            "binder_title": self.binder_title,
            "objective": self.objective,
            "source_document_count": len(self.source_documents),
            "evidence_row_count": len(self.evidence_rows),
            "included_artifact_count": sum(1 for item in self.artifacts if item.included_in_zip),
            "missing_citation_warning_count": len(self.open_issues),
            "binder_artifacts": list(self.binder_artifacts),
            "evidence_preview": [item.to_dict() for item in self.evidence_rows[:8]],
            "open_issues": [item.to_dict() for item in self.open_issues[:12]],
            "warnings": list(self.warnings),
            "background_job_id": self.background_job_id,
        }


__all__ = [
    "BinderArtifact",
    "BinderHandoffArtifact",
    "BinderOpenIssue",
    "BinderSourceDocument",
    "EvidenceBinderResult",
    "EvidenceRow",
]
