from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class CampaignDocumentRecord:
    doc_id: str
    title: str
    sector: str
    sector_source: str
    source_type: str = ""
    source_path: str = ""
    collection_id: str = ""
    file_type: str = ""
    content_hash: str = ""
    extraction_status: str = "pending"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessFlowMatch:
    match_id: str
    left_doc_id: str
    right_doc_id: str
    left_title: str
    right_title: str
    left_sector: str
    right_sector: str
    score: float
    matched_left_steps: List[str] = field(default_factory=list)
    matched_right_steps: List[str] = field(default_factory=list)
    cross_sector_advisory: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentSimilarityEdge:
    edge_id: str
    left_doc_id: str
    right_doc_id: str
    left_title: str
    right_title: str
    left_sector: str
    right_sector: str
    consolidation_score: float
    content_overlap_score: float = 0.0
    process_flow_score: float = 0.0
    section_structure_score: float = 0.0
    table_schema_score: float = 0.0
    obligation_overlap_score: float = 0.0
    metadata_title_score: float = 0.0
    reason_codes: List[str] = field(default_factory=list)
    shared_terms: List[str] = field(default_factory=list)
    cross_sector: bool = False
    cross_sector_advisory: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConsolidationCluster:
    cluster_id: str
    documents: List[CampaignDocumentRecord]
    sectors: List[str]
    edge_ids: List[str]
    consolidation_score: float
    reason_codes: List[str] = field(default_factory=list)
    recommendation: str = ""
    cross_sector: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["documents"] = [doc.to_dict() for doc in self.documents]
        return payload


@dataclass
class SectorSummary:
    sectors: Dict[str, int] = field(default_factory=dict)
    unknown_documents: List[str] = field(default_factory=list)
    cross_sector_mode: str = "blocked"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentConsolidationCampaignResult:
    campaign_id: str
    status: str
    query: str
    selected_document_count: int
    manifest: List[CampaignDocumentRecord] = field(default_factory=list)
    sector_summary: SectorSummary = field(default_factory=SectorSummary)
    similarity_edges: List[DocumentSimilarityEdge] = field(default_factory=list)
    process_flow_matches: List[ProcessFlowMatch] = field(default_factory=list)
    consolidation_clusters: List[ConsolidationCluster] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    background_job_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "status": self.status,
            "query": self.query,
            "selected_document_count": self.selected_document_count,
            "manifest": [item.to_dict() for item in self.manifest],
            "sector_summary": self.sector_summary.to_dict(),
            "similarity_edges": [item.to_dict() for item in self.similarity_edges],
            "process_flow_matches": [item.to_dict() for item in self.process_flow_matches],
            "consolidation_clusters": [item.to_dict() for item in self.consolidation_clusters],
            "warnings": list(self.warnings),
            "artifacts": list(self.artifacts),
            "background_job_id": self.background_job_id,
        }

    def compact(self, *, max_clusters: int = 8) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "status": self.status,
            "query": self.query,
            "selected_document_count": self.selected_document_count,
            "sector_summary": self.sector_summary.to_dict(),
            "candidate_cluster_count": len(self.consolidation_clusters),
            "similarity_edge_count": len(self.similarity_edges),
            "process_flow_match_count": len(self.process_flow_matches),
            "top_consolidation_clusters": [
                item.to_dict()
                for item in self.consolidation_clusters[: max(1, int(max_clusters))]
            ],
            "warnings": list(self.warnings),
            "artifacts": list(self.artifacts),
            "background_job_id": self.background_job_id,
        }


__all__ = [
    "CampaignDocumentRecord",
    "ConsolidationCluster",
    "DocumentConsolidationCampaignResult",
    "DocumentSimilarityEdge",
    "ProcessFlowMatch",
    "SectorSummary",
]
