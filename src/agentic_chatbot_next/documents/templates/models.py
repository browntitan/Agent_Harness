from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class SourceTraceEntry:
    source_id: str
    document_id: str
    document_title: str
    element_id: str = ""
    element_type: str = ""
    source_location: Dict[str, Any] = field(default_factory=dict)
    text_excerpt: str = ""
    parser_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SourceDocumentPacket:
    document_id: str
    title: str
    source_type: str = ""
    source_path: str = ""
    collection_id: str = ""
    file_type: str = ""
    parser_path: str = ""
    content_hash: str = ""
    counts: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    classification_markings: List[str] = field(default_factory=list)
    trace_entries: List[SourceTraceEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "collection_id": self.collection_id,
            "file_type": self.file_type,
            "parser_path": self.parser_path,
            "content_hash": self.content_hash,
            "counts": dict(self.counts),
            "warnings": list(self.warnings),
            "classification_markings": list(self.classification_markings),
            "trace_entries": [item.to_dict() for item in self.trace_entries],
        }


@dataclass
class TemplateSection:
    title: str
    body: str = ""
    bullets: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemplateTable:
    name: str
    columns: List[str]
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "columns": list(self.columns),
            "rows": [dict(row) for row in self.rows],
        }


@dataclass
class TemplateSlide:
    title: str
    bullets: List[str] = field(default_factory=list)
    speaker_notes: str = ""
    source_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratedArtifact:
    filename: str
    artifact_kind: str
    artifact_ref: str = ""
    content_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UnsupportedClaimWarning:
    message: str
    location: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemplateTransformResult:
    transform_id: str
    status: str
    template_type: str
    output_format: str
    selected_document_count: int
    source_documents: List[SourceDocumentPacket] = field(default_factory=list)
    sections: List[TemplateSection] = field(default_factory=list)
    tables: List[TemplateTable] = field(default_factory=list)
    slides: List[TemplateSlide] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    unsupported_claims: List[UnsupportedClaimWarning] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    source_trace_artifacts: List[Dict[str, Any]] = field(default_factory=list)
    background_job_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "status": self.status,
            "template_type": self.template_type,
            "output_format": self.output_format,
            "selected_document_count": self.selected_document_count,
            "source_documents": [item.to_dict() for item in self.source_documents],
            "sections": [item.to_dict() for item in self.sections],
            "tables": [item.to_dict() for item in self.tables],
            "slides": [item.to_dict() for item in self.slides],
            "warnings": list(self.warnings),
            "unsupported_claims": [item.to_dict() for item in self.unsupported_claims],
            "artifacts": list(self.artifacts),
            "source_trace_artifacts": list(self.source_trace_artifacts),
            "background_job_id": self.background_job_id,
        }

    def compact(self) -> Dict[str, Any]:
        table_preview = []
        for table in self.tables[:2]:
            table_preview.append(
                {
                    "name": table.name,
                    "columns": list(table.columns),
                    "rows": [dict(row) for row in table.rows[:8]],
                    "row_count": len(table.rows),
                }
            )
        return {
            "transform_id": self.transform_id,
            "status": self.status,
            "template_type": self.template_type,
            "output_format": self.output_format,
            "selected_document_count": self.selected_document_count,
            "parser_warnings": [
                warning
                for packet in self.source_documents
                for warning in packet.warnings
            ][:20],
            "generated_artifacts": list(self.artifacts),
            "source_trace_artifacts": list(self.source_trace_artifacts),
            "unsupported_content_warnings": [item.to_dict() for item in self.unsupported_claims[:20]],
            "warnings": list(self.warnings),
            "preview": {
                "sections": [item.to_dict() for item in self.sections[:5]],
                "tables": table_preview,
                "slides": [item.to_dict() for item in self.slides[:5]],
            },
            "background_job_id": self.background_job_id,
        }


__all__ = [
    "GeneratedArtifact",
    "SourceDocumentPacket",
    "SourceTraceEntry",
    "TemplateSection",
    "TemplateSlide",
    "TemplateTable",
    "TemplateTransformResult",
    "UnsupportedClaimWarning",
]
