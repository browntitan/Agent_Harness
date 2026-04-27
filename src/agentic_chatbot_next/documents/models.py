from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DocumentIdentity:
    doc_id: str
    title: str
    source_type: str = ""
    source_path: str = ""
    collection_id: str = ""
    file_type: str = ""
    content_hash: str = ""
    parser_path: str = ""
    source_scope: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentSection:
    section_id: str
    title: str
    level: int = 1
    parent_id: str = ""
    order: int = 0
    location: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentTable:
    table_id: str
    title: str = ""
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    sheet_name: str = ""
    cell_range: str = ""
    columns: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentFigure:
    figure_id: str
    title: str = ""
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentElement:
    element_id: str
    element_type: str
    text: str
    order: int = 0
    section_id: str = ""
    section_title: str = ""
    section_path: List[str] = field(default_factory=list)
    clause_number: str = ""
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    sheet_name: str = ""
    row_start: Optional[int] = None
    row_end: Optional[int] = None
    cell_range: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentExtractResult:
    document: DocumentIdentity
    metadata: Dict[str, Any] = field(default_factory=dict)
    sections: List[DocumentSection] = field(default_factory=list)
    elements: List[DocumentElement] = field(default_factory=list)
    tables: List[DocumentTable] = field(default_factory=list)
    figures: List[DocumentFigure] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    truncated: bool = False

    def counts(self) -> Dict[str, int]:
        return {
            "sections": len(self.sections),
            "elements": len(self.elements),
            "tables": len(self.tables),
            "figures": len(self.figures),
            "metadata_fields": len(self.metadata),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "metadata": dict(self.metadata),
            "sections": [item.to_dict() for item in self.sections],
            "elements": [item.to_dict() for item in self.elements],
            "tables": [item.to_dict() for item in self.tables],
            "figures": [item.to_dict() for item in self.figures],
            "warnings": list(self.warnings),
            "truncated": self.truncated,
            "counts": self.counts(),
        }


@dataclass
class DocumentDelta:
    delta_id: str
    change_type: str
    summary: str
    left_element_id: str = ""
    right_element_id: str = ""
    left_text: str = ""
    right_text: str = ""
    location: Dict[str, Any] = field(default_factory=dict)
    similarity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChangedObligation:
    obligation_id: str
    change_type: str
    modality: str
    severity: str
    before_text: str = ""
    after_text: str = ""
    location: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentCompareResult:
    left_document: DocumentIdentity
    right_document: DocumentIdentity
    compare_mode: str
    focus: str = ""
    deltas: List[DocumentDelta] = field(default_factory=list)
    changed_obligations: List[ChangedObligation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        counts = {"added": 0, "removed": 0, "modified": 0, "unchanged": 0}
        for delta in self.deltas:
            key = delta.change_type if delta.change_type in counts else "modified"
            counts[key] += 1
        counts["changed_obligations"] = len(self.changed_obligations)
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "left_document": self.left_document.to_dict(),
            "right_document": self.right_document.to_dict(),
            "compare_mode": self.compare_mode,
            "focus": self.focus,
            "counts": self.counts(),
            "deltas": [item.to_dict() for item in self.deltas],
            "changed_obligations": [item.to_dict() for item in self.changed_obligations],
            "warnings": list(self.warnings),
        }
