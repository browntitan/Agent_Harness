from __future__ import annotations

import json
import re
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_chatbot_next.documents.extractors import DocumentExtractionService, DocumentResolutionError, element_location
from agentic_chatbot_next.documents.models import DocumentExtractResult
from agentic_chatbot_next.documents.templates.models import (
    SourceDocumentPacket,
    SourceTraceEntry,
    TemplateSection,
    TemplateSlide,
    TemplateTable,
    TemplateTransformResult,
    UnsupportedClaimWarning,
)
from agentic_chatbot_next.documents.templates.renderers import (
    csv_for_table,
    markdown_for_result,
    render_docx_bytes,
    render_pptx_bytes,
    render_xlsx_bytes,
    safe_stem,
    sidecar_json,
    source_trace_json,
    write_binary_artifact,
    write_text_artifact,
)
from agentic_chatbot_next.rag.requirements import BROAD_REQUIREMENT_MODE, build_requirement_inventory
from agentic_chatbot_next.utils.json_utils import extract_json


TEMPLATE_TYPES = {"memo", "rfp_compliance_matrix", "cdrl_sdrl_tracker", "executive_brief", "test_report"}
OUTPUT_FORMATS = {"auto", "docx", "xlsx", "pptx", "markdown", "json"}
DRAFTING_MODES = {"grounded", "extractive", "deterministic"}
MATRIX_COLUMNS = [
    "Requirement ID",
    "Source Document",
    "Clause/Section",
    "Requirement Text",
    "Compliance Response",
    "Evidence/Source",
    "Owner",
    "Status",
    "Risk/Notes",
]
TRACKER_COLUMNS = [
    "Deliverable ID",
    "Deliverable Title",
    "CDRL/SDRL Type",
    "Source Clause",
    "Frequency/Event",
    "Due Date/Trigger",
    "Format",
    "Owner",
    "Status",
    "Notes",
]

_PRIMARY_FORMAT_BY_TEMPLATE = {
    "memo": "docx",
    "rfp_compliance_matrix": "xlsx",
    "cdrl_sdrl_tracker": "xlsx",
    "executive_brief": "pptx",
    "test_report": "docx",
}
_CLASSIFICATION_RE = re.compile(
    r"\b(?:UNCLASSIFIED|CUI|CONTROLLED\s+UNCLASSIFIED|SECRET|TOP\s+SECRET|NOFORN|EXPORT\s+CONTROL|DISTRIBUTION\s+STATEMENT)\b",
    re.IGNORECASE,
)
_DELIVERABLE_RE = re.compile(
    r"\b(?:CDRL|SDRL|deliverable|data\s+item|DID|report|plan|manual|drawing|submission|submit|"
    r"due|monthly|weekly|quarterly|days?\s+after|calendar\s+days|business\s+days|format)\b",
    re.IGNORECASE,
)
_TEST_RE = re.compile(r"\b(?:test|verification|setup|method|result|anomaly|defect|pass|fail|procedure|campaign)\b", re.IGNORECASE)


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalise_template_type(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in TEMPLATE_TYPES:
        raise ValueError(f"Unsupported template_type '{value}'.")
    return normalized


def _normalise_output_format(value: str) -> str:
    normalized = str(value or "auto").strip().lower()
    return normalized if normalized in OUTPUT_FORMATS else "auto"


def _normalise_drafting_mode(value: str) -> str:
    normalized = str(value or "grounded").strip().lower()
    return normalized if normalized in DRAFTING_MODES else "grounded"


def _primary_format(template_type: str) -> str:
    return _PRIMARY_FORMAT_BY_TEMPLATE.get(template_type, "markdown")


def _source_id(doc_index: int, element_index: int) -> str:
    return f"S{doc_index:02d}.E{element_index:04d}"


def _short(text: str, limit: int = 360) -> str:
    clean = _clean(text)
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _location_text(entry: SourceTraceEntry) -> str:
    parts = []
    for key in ("section_title", "clause_number", "page_number", "slide_number", "sheet_name", "cell_range"):
        value = entry.source_location.get(key)
        if value not in ("", None):
            parts.append(f"{key}={value}")
    return "; ".join(parts)


class TemplateTransformService:
    def __init__(
        self,
        settings: object,
        stores: object,
        session: object,
        *,
        providers: object | None = None,
        event_sink: object | None = None,
    ) -> None:
        self.settings = settings
        self.stores = stores
        self.session = session
        self.providers = providers
        self.event_sink = event_sink

    def transform(
        self,
        *,
        document_refs: Sequence[str],
        template_type: str,
        source_scope: str = "auto",
        collection_id: str = "",
        focus: str = "",
        audience: str = "",
        output_format: str = "auto",
        template_parameters: Dict[str, Any] | None = None,
        include_source_trace: bool = True,
        drafting_mode: str = "grounded",
        max_source_elements: int = 1200,
        export: bool = True,
        transform_id: str = "",
    ) -> TemplateTransformResult:
        template = _normalise_template_type(template_type)
        requested_output = _normalise_output_format(output_format)
        drafting = _normalise_drafting_mode(drafting_mode)
        transform_id = transform_id or f"template_transform_{uuid.uuid4().hex[:12]}"
        params = dict(template_parameters or {})
        warnings: list[str] = [
            "Generated artifacts require human review before external sharing; classification marking and redaction review are not automatic in V1."
        ]
        refs = [_clean(item) for item in list(document_refs or []) if _clean(item)]
        if not refs:
            raise ValueError("template_transform requires at least one document_ref.")

        packets, traces, results = self._build_source_packets(
            refs,
            source_scope=source_scope,
            collection_id=collection_id,
            max_source_elements=max_source_elements,
            warnings=warnings,
        )
        if not packets:
            raise ValueError("No source documents could be extracted for template_transform.")

        sections: list[TemplateSection] = []
        tables: list[TemplateTable] = []
        slides: list[TemplateSlide] = []
        unsupported: list[UnsupportedClaimWarning] = []

        if template == "rfp_compliance_matrix":
            tables.append(self._build_compliance_matrix(results, traces, params=params))
        elif template == "cdrl_sdrl_tracker":
            tables.append(self._build_cdrl_sdrl_tracker(results, traces, params=params))
        elif template == "executive_brief":
            slides = self._build_executive_brief(traces, packets, focus=focus, audience=audience, params=params)
            sections = [TemplateSection(title=slide.title, bullets=slide.bullets, source_ids=slide.source_ids) for slide in slides]
        elif template == "test_report":
            sections = self._build_test_report(traces, packets, focus=focus, audience=audience, params=params)
        else:
            sections = self._build_memo(traces, packets, focus=focus, audience=audience, params=params)

        if drafting == "grounded" and template in {"memo", "executive_brief", "test_report"}:
            drafted = self._try_grounded_model_draft(template, traces, focus=focus, audience=audience)
            if drafted:
                if template == "executive_brief":
                    slides = drafted["slides"]
                    sections = [TemplateSection(title=slide.title, bullets=slide.bullets, source_ids=slide.source_ids) for slide in slides]
                else:
                    sections = drafted["sections"]
            else:
                warnings.append("Grounded model drafting was unavailable or invalid; used deterministic source-backed drafting.")
        elif drafting == "extractive":
            warnings.append("Extractive drafting mode used source excerpts without model-generated prose.")
        elif drafting == "deterministic":
            warnings.append("Deterministic drafting mode used built-in template rules only.")

        result = TemplateTransformResult(
            transform_id=transform_id,
            status="completed",
            template_type=template,
            output_format=_primary_format(template) if requested_output == "auto" else requested_output,
            selected_document_count=len(packets),
            source_documents=packets,
            sections=sections,
            tables=tables,
            slides=slides,
            warnings=warnings,
            unsupported_claims=unsupported,
        )
        if export:
            result.artifacts, result.source_trace_artifacts = self._write_artifacts(
                result,
                requested_output=requested_output,
                include_source_trace=include_source_trace,
            )
        return result

    def _build_source_packets(
        self,
        refs: Sequence[str],
        *,
        source_scope: str,
        collection_id: str,
        max_source_elements: int,
        warnings: list[str],
    ) -> tuple[list[SourceDocumentPacket], dict[str, SourceTraceEntry], list[DocumentExtractResult]]:
        extractor = DocumentExtractionService(self.settings, self.stores, self.session)
        packets: list[SourceDocumentPacket] = []
        traces: dict[str, SourceTraceEntry] = {}
        results: list[DocumentExtractResult] = []
        remaining = max(1, int(max_source_elements or 1200))
        for doc_index, ref in enumerate(refs, start=1):
            if remaining <= 0:
                warnings.append(f"Source extraction truncated at max_source_elements={max_source_elements}.")
                break
            try:
                result = extractor.extract(
                    document_ref=ref,
                    source_scope=source_scope,
                    collection_id=collection_id,
                    include_tables=True,
                    include_figures=True,
                    include_metadata=True,
                    include_hierarchy=True,
                    max_elements=remaining,
                )
            except DocumentResolutionError as exc:
                warnings.append(f"Could not resolve source document {ref}: {exc.payload.get('error')}")
                continue
            results.append(result)
            packet_entries: list[SourceTraceEntry] = []
            for element_index, element in enumerate(result.elements[:remaining], start=1):
                source_id = _source_id(doc_index, element_index)
                entry = SourceTraceEntry(
                    source_id=source_id,
                    document_id=result.document.doc_id,
                    document_title=result.document.title,
                    element_id=element.element_id,
                    element_type=element.element_type,
                    source_location=element_location(element),
                    text_excerpt=_short(element.text, 700),
                    parser_path=result.document.parser_path,
                )
                traces[source_id] = entry
                packet_entries.append(entry)
            remaining -= len(packet_entries)
            packets.append(
                SourceDocumentPacket(
                    document_id=result.document.doc_id,
                    title=result.document.title,
                    source_type=result.document.source_type,
                    source_path=result.document.source_path,
                    collection_id=result.document.collection_id,
                    file_type=result.document.file_type,
                    parser_path=result.document.parser_path,
                    content_hash=result.document.content_hash,
                    counts=result.counts(),
                    warnings=list(result.warnings),
                    classification_markings=self._classification_markings(result),
                    trace_entries=packet_entries,
                )
            )
        return packets, traces, results

    def _classification_markings(self, result: DocumentExtractResult) -> list[str]:
        markings: list[str] = []
        for element in result.elements[:20]:
            text = _short(element.text, 240)
            if _CLASSIFICATION_RE.search(text):
                markings.append(text)
        return list(dict.fromkeys(markings))[:10]

    def _build_compliance_matrix(
        self,
        results: Sequence[DocumentExtractResult],
        traces: dict[str, SourceTraceEntry],
        *,
        params: dict[str, Any],
    ) -> TemplateTable:
        rows: list[dict[str, Any]] = []
        default_owner = _clean(params.get("owner")) or "TBD"
        default_status = _clean(params.get("status")) or "Not Started"
        for result in results:
            trace_by_element_id = {
                entry.element_id: entry
                for entry in traces.values()
                if entry.document_id == result.document.doc_id
            }
            chunks = []
            for index, element in enumerate(result.elements):
                chunks.append(
                    SimpleNamespace(
                        content=element.text,
                        chunk_id=element.element_id,
                        chunk_index=index,
                        chunk_type=element.element_type,
                        page_number=element.page_number,
                        clause_number=element.clause_number,
                        section_title=element.section_title,
                    )
                )
            inventory = build_requirement_inventory(
                SimpleNamespace(
                    doc_id=result.document.doc_id,
                    tenant_id=str(getattr(self.session, "tenant_id", "local-dev") or "local-dev"),
                    collection_id=result.document.collection_id or "workspace",
                    source_type=result.document.source_type,
                    title=result.document.title,
                    file_type=result.document.file_type,
                ),
                chunks,
                mode=BROAD_REQUIREMENT_MODE,
            )
            for record in inventory.records:
                trace = trace_by_element_id.get(str(getattr(record, "chunk_id", "") or ""))
                source_id = trace.source_id if trace else ""
                rows.append(
                    {
                        "Requirement ID": str(getattr(record, "requirement_id", "") or f"REQ-{len(rows) + 1:04d}"),
                        "Source Document": result.document.title,
                        "Clause/Section": str(getattr(record, "clause_number", "") or getattr(record, "section_title", "") or ""),
                        "Requirement Text": str(getattr(record, "statement_text", "") or ""),
                        "Compliance Response": "",
                        "Evidence/Source": source_id,
                        "Owner": default_owner,
                        "Status": default_status,
                        "Risk/Notes": str(getattr(record, "risk_label", "") or ""),
                    }
                )
        if not rows:
            rows.append({column: "" for column in MATRIX_COLUMNS})
            rows[0]["Risk/Notes"] = "No requirement-like statements were detected."
        return TemplateTable(name="Compliance Matrix", columns=MATRIX_COLUMNS, rows=rows)

    def _build_cdrl_sdrl_tracker(
        self,
        results: Sequence[DocumentExtractResult],
        traces: dict[str, SourceTraceEntry],
        *,
        params: dict[str, Any],
    ) -> TemplateTable:
        default_owner = _clean(params.get("owner")) or "TBD"
        default_status = _clean(params.get("status")) or "Not Started"
        rows: list[dict[str, Any]] = []
        for entry in traces.values():
            if not _DELIVERABLE_RE.search(entry.text_excerpt):
                continue
            text = entry.text_excerpt
            rows.append(
                {
                    "Deliverable ID": self._deliverable_id(text, len(rows) + 1),
                    "Deliverable Title": self._deliverable_title(text),
                    "CDRL/SDRL Type": self._deliverable_type(text),
                    "Source Clause": _location_text(entry),
                    "Frequency/Event": self._frequency(text),
                    "Due Date/Trigger": self._due_trigger(text),
                    "Format": self._format_hint(text),
                    "Owner": default_owner,
                    "Status": default_status,
                    "Notes": f"Source {entry.source_id}: {_short(text, 160)}",
                }
            )
        if not rows:
            rows.append({column: "" for column in TRACKER_COLUMNS})
            rows[0]["Notes"] = "No deliverable-like statements were detected."
        return TemplateTable(name="CDRL SDRL Tracker", columns=TRACKER_COLUMNS, rows=rows)

    def _build_memo(
        self,
        traces: dict[str, SourceTraceEntry],
        packets: Sequence[SourceDocumentPacket],
        *,
        focus: str,
        audience: str,
        params: dict[str, Any],
    ) -> list[TemplateSection]:
        selected = self._ranked_traces(traces, focus=focus)
        title = _clean(params.get("title")) or "Source Document Memo"
        source_ids = [entry.source_id for entry in selected[:5]]
        return [
            TemplateSection("Header", body=f"{title}. Audience: {_clean(audience) or 'general enterprise users'}.", source_ids=source_ids[:1]),
            TemplateSection("Purpose", body=_clean(focus) or "Convert the cited source documents into a decision-ready memo.", source_ids=source_ids[:1]),
            TemplateSection("Executive Summary", bullets=[_short(entry.text_excerpt, 240) for entry in selected[:3]], source_ids=source_ids[:3]),
            TemplateSection("Background", bullets=[f"{packet.title} ({packet.file_type or 'document'}, parser={packet.parser_path})" for packet in packets], source_ids=source_ids[:3]),
            TemplateSection("Key Findings", bullets=[_short(entry.text_excerpt, 260) for entry in selected[:6]], source_ids=source_ids),
            TemplateSection("Recommendations", bullets=["Review cited findings with the accountable document owner before release or downstream reuse."], source_ids=source_ids[:3]),
            TemplateSection("Source Notes", bullets=[f"{entry.source_id}: {entry.document_title} {_location_text(entry)}" for entry in selected[:8]], source_ids=source_ids),
        ]

    def _build_executive_brief(
        self,
        traces: dict[str, SourceTraceEntry],
        packets: Sequence[SourceDocumentPacket],
        *,
        focus: str,
        audience: str,
        params: dict[str, Any],
    ) -> list[TemplateSlide]:
        selected = self._ranked_traces(traces, focus=focus)
        title = _clean(params.get("title")) or "Executive Brief"
        source_ids = [entry.source_id for entry in selected[:8]]
        return [
            TemplateSlide(title, bullets=[_clean(focus) or "Grounded summary from source documents.", f"Audience: {_clean(audience) or 'executive'}"], source_ids=source_ids[:2]),
            TemplateSlide("Situation", bullets=[f"{len(packets)} source document(s) reviewed.", *[_short(entry.text_excerpt, 180) for entry in selected[:2]]], source_ids=source_ids[:2]),
            TemplateSlide("Key Findings", bullets=[_short(entry.text_excerpt, 190) for entry in selected[:5]], source_ids=source_ids[:5]),
            TemplateSlide("Risks / Issues", bullets=self._bullets_matching(selected, r"\b(?:risk|issue|delay|late|fail|defect|noncompliance|shall|must)\b", limit=5), source_ids=source_ids[:5]),
            TemplateSlide("Decisions Needed", bullets=["Confirm owners, disposition, and next-step actions for cited findings."], source_ids=source_ids[:3]),
            TemplateSlide("Next Steps", bullets=["Validate source interpretation.", "Assign action owners.", "Run classification/redaction review before sharing."], source_ids=source_ids[:3]),
            TemplateSlide("Source Appendix", bullets=[f"{entry.source_id}: {entry.document_title}" for entry in selected[:8]], source_ids=source_ids),
        ]

    def _build_test_report(
        self,
        traces: dict[str, SourceTraceEntry],
        packets: Sequence[SourceDocumentPacket],
        *,
        focus: str,
        audience: str,
        params: dict[str, Any],
    ) -> list[TemplateSection]:
        selected = self._ranked_traces(traces, focus=focus)
        test_traces = [entry for entry in selected if _TEST_RE.search(entry.text_excerpt)] or selected
        source_ids = [entry.source_id for entry in test_traces[:8]]
        title = _clean(params.get("title") or params.get("test_campaign_name")) or "Test Report"
        return [
            TemplateSection("Header", body=f"{title}. Audience: {_clean(audience) or 'engineering and program management'}.", source_ids=source_ids[:1]),
            TemplateSection("Objective", bullets=[_short(entry.text_excerpt, 240) for entry in test_traces[:2]], source_ids=source_ids[:2]),
            TemplateSection("Scope", bullets=[f"{packet.title} ({packet.file_type or 'document'})" for packet in packets], source_ids=source_ids[:3]),
            TemplateSection("Test Method", bullets=self._bullets_matching(test_traces, r"\b(?:method|procedure|setup|configuration|verify|verification)\b", limit=4), source_ids=source_ids[:4]),
            TemplateSection("Test Setup", bullets=self._bullets_matching(test_traces, r"\b(?:setup|equipment|configuration|environment|fixture)\b", limit=4), source_ids=source_ids[:4]),
            TemplateSection("Results", bullets=self._bullets_matching(test_traces, r"\b(?:result|pass|fail|measured|observed|completed)\b", limit=5), source_ids=source_ids[:5]),
            TemplateSection("Anomalies", bullets=self._bullets_matching(test_traces, r"\b(?:anomal|defect|issue|fail|deviation|waiver)\b", limit=4), source_ids=source_ids[:4]),
            TemplateSection("Conclusion", bullets=["The conclusions above are source-backed excerpts and require test lead review before release."], source_ids=source_ids[:3]),
            TemplateSection("Evidence Appendix", bullets=[f"{entry.source_id}: {entry.document_title} {_location_text(entry)}" for entry in test_traces[:10]], source_ids=source_ids),
        ]

    def _try_grounded_model_draft(
        self,
        template: str,
        traces: dict[str, SourceTraceEntry],
        *,
        focus: str,
        audience: str,
    ) -> dict[str, Any] | None:
        model = None
        if self.providers is not None:
            model = getattr(self.providers, "chat", None) or getattr(self.providers, "judge", None)
        if model is None:
            return None
        selected = self._ranked_traces(traces, focus=focus)[:24]
        source_ids = {entry.source_id for entry in selected}
        source_block = "\n".join(f"{entry.source_id}: {entry.text_excerpt}" for entry in selected)
        prompt = (
            "Return strict JSON only. Use only the provided source ids. "
            "Every bullet must include source_ids with at least one valid id. "
            "Schema for memo/test_report: {\"sections\":[{\"title\":\"...\",\"body\":\"...\",\"bullets\":[\"...\"],\"source_ids\":[\"S01.E0001\"]}]}. "
            "Schema for executive_brief: {\"slides\":[{\"title\":\"...\",\"bullets\":[\"...\"],\"source_ids\":[\"S01.E0001\"]}]}.\n\n"
            f"TEMPLATE: {template}\nFOCUS: {focus}\nAUDIENCE: {audience}\nSOURCES:\n{source_block}"
        )
        try:
            response = model.invoke(
                [
                    SystemMessage(content="You produce grounded enterprise document templates from cited source excerpts."),
                    HumanMessage(content=prompt),
                ],
                config={"metadata": {"session_id": str(getattr(self.session, "session_id", "") or "")}},
            )
            payload = extract_json(getattr(response, "content", None) or str(response)) or {}
        except Exception:
            return None
        if template == "executive_brief":
            slides = []
            for raw in list(payload.get("slides") or [])[:10]:
                ids = [str(item) for item in list(raw.get("source_ids") or []) if str(item) in source_ids]
                bullets = [_clean(item) for item in list(raw.get("bullets") or []) if _clean(item)]
                title = _clean(raw.get("title"))
                if title and bullets and ids:
                    slides.append(TemplateSlide(title=title, bullets=bullets[:6], source_ids=ids[:8]))
            return {"slides": slides} if slides else None
        sections = []
        for raw in list(payload.get("sections") or [])[:12]:
            ids = [str(item) for item in list(raw.get("source_ids") or []) if str(item) in source_ids]
            bullets = [_clean(item) for item in list(raw.get("bullets") or []) if _clean(item)]
            title = _clean(raw.get("title"))
            body = _clean(raw.get("body"))
            if title and (body or bullets) and ids:
                sections.append(TemplateSection(title=title, body=body, bullets=bullets[:8], source_ids=ids[:8]))
        return {"sections": sections} if sections else None

    def _write_artifacts(
        self,
        result: TemplateTransformResult,
        *,
        requested_output: str,
        include_source_trace: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        stem = safe_stem(result.template_type)
        primary = _primary_format(result.template_type)
        requested = requested_output if requested_output != "auto" else primary
        artifacts: list[dict[str, Any]] = []
        trace_artifacts: list[dict[str, Any]] = []
        formats = self._artifact_formats(result.template_type, requested)
        if requested != primary and requested not in formats:
            result.warnings.append(f"Requested output_format={requested} is not native for {result.template_type}; emitted default artifacts instead.")
        for fmt in formats:
            filename = f"{stem}__{result.transform_id}.{fmt}"
            if fmt == "docx":
                artifacts.append(write_binary_artifact(self.session, filename, render_docx_bytes(result)))
            elif fmt == "xlsx":
                artifacts.append(write_binary_artifact(self.session, filename, render_xlsx_bytes(result)))
            elif fmt == "pptx":
                artifacts.append(write_binary_artifact(self.session, filename, render_pptx_bytes(result)))
            elif fmt == "md":
                artifacts.append(write_text_artifact(self.session, filename, markdown_for_result(result)))
            elif fmt == "csv":
                artifacts.append(write_text_artifact(self.session, filename, csv_for_table(result)))
            elif fmt == "json":
                artifacts.append(write_text_artifact(self.session, filename, sidecar_json(result)))
        if include_source_trace:
            trace_artifacts.append(
                write_text_artifact(
                    self.session,
                    f"{stem}__{result.transform_id}__source_trace.json",
                    source_trace_json(result),
                )
            )
        return artifacts, trace_artifacts

    def _artifact_formats(self, template_type: str, requested: str) -> list[str]:
        defaults = {
            "memo": ["docx", "md", "json"],
            "rfp_compliance_matrix": ["xlsx", "csv", "json"],
            "cdrl_sdrl_tracker": ["xlsx", "csv", "json"],
            "executive_brief": ["pptx", "md", "json"],
            "test_report": ["docx", "md", "json"],
        }
        if requested == "json":
            return ["json"]
        if requested == "markdown":
            return ["md", "json"]
        if requested == _primary_format(template_type):
            return defaults[template_type]
        return defaults[template_type]

    def _ranked_traces(self, traces: dict[str, SourceTraceEntry], *, focus: str) -> list[SourceTraceEntry]:
        focus_terms = {term.casefold() for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", focus or "")}
        entries = [
            entry
            for entry in traces.values()
            if len(entry.text_excerpt) >= 12 and entry.element_type not in {"heading", "slide_title"}
        ]
        if not focus_terms:
            return entries[:80]
        return sorted(
            entries,
            key=lambda entry: len(focus_terms & {term.casefold() for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", entry.text_excerpt)}),
            reverse=True,
        )[:80]

    def _bullets_matching(self, entries: Sequence[SourceTraceEntry], pattern: str, *, limit: int) -> list[str]:
        regex = re.compile(pattern, re.IGNORECASE)
        matches = [_short(entry.text_excerpt, 220) for entry in entries if regex.search(entry.text_excerpt)]
        return matches[:limit] or [_short(entry.text_excerpt, 220) for entry in entries[: min(limit, len(entries))]]

    def _deliverable_id(self, text: str, index: int) -> str:
        match = re.search(r"\b(?:CDRL|SDRL)[-\s_]*([A-Za-z0-9.-]+)\b", text, re.IGNORECASE)
        if match:
            prefix = "CDRL" if "cdrl" in match.group(0).casefold() else "SDRL"
            return f"{prefix}-{match.group(1)}"
        return f"DEL-{index:04d}"

    def _deliverable_title(self, text: str) -> str:
        clean = _short(text, 110)
        return clean.rstrip(".")

    def _deliverable_type(self, text: str) -> str:
        lowered = text.casefold()
        if "sdrl" in lowered:
            return "SDRL"
        if "cdrl" in lowered:
            return "CDRL"
        return "Deliverable"

    def _frequency(self, text: str) -> str:
        match = re.search(r"\b(monthly|weekly|quarterly|annually|daily|one[-\s]?time|as[-\s]?required)\b", text, re.IGNORECASE)
        return match.group(1) if match else ""

    def _due_trigger(self, text: str) -> str:
        match = re.search(r"\b(?:within|not later than|no later than)\s+[^.;,]{1,80}", text, re.IGNORECASE)
        return _clean(match.group(0)) if match else ""

    def _format_hint(self, text: str) -> str:
        match = re.search(r"\b(?:PDF|DOCX|XLSX|Excel|PowerPoint|PPTX|DI-[A-Z0-9-]+)\b", text, re.IGNORECASE)
        return match.group(0) if match else ""


__all__ = [
    "DRAFTING_MODES",
    "OUTPUT_FORMATS",
    "TEMPLATE_TYPES",
    "TemplateTransformService",
]
