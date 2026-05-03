from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from agentic_chatbot_next.utils.json_utils import extract_json

CORPUS_METADATA_VERSION = "corpus_metadata_v1"

_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
_CDRL_RE = re.compile(r"\b(?:CDRL[\s:-]*[A-Z0-9_.-]+|A\d{3}(?:[-.]\d+)?)\b", re.IGNORECASE)
_WBS_RE = re.compile(r"\b(?:WBS[\s:-]*)?\d{1,2}(?:\.\d{1,3}){1,5}\b", re.IGNORECASE)
_REQ_ID_RE = re.compile(r"\b(?:REQ|RQT|SRD|IRS|SRS|FRD|NFR)[-_ ]?[A-Z0-9]+(?:[-_.][A-Z0-9]+)*\b", re.IGNORECASE)
_REQ_VERB_RE = re.compile(r"\b(?:shall|must|required to|requirement)\b", re.IGNORECASE)
_REVISION_RE = re.compile(r"\b(?:rev(?:ision)?|version|ver\.?)\s*[:#-]?\s*([A-Z0-9][A-Z0-9_.-]{0,24})\b", re.IGNORECASE)
_AUTHOR_RE = re.compile(r"\b(?:author|prepared by|owner)\s*[:\-]\s*([A-Z][A-Za-z0-9 .,'&/-]{1,80})", re.IGNORECASE)
_STATUS_RE = re.compile(
    r"\b(?:draft|in review|under review|approved|released|submitted|accepted|rejected|"
    r"closed|open|complete|completed|overdue|late|on track|blocked|superseded|archived)\b",
    re.IGNORECASE,
)
_ENTITY_RE = re.compile(r"\b[A-Z][A-Z0-9&/-]{2,}(?:\s+[A-Z][A-Z0-9&/-]{2,}){0,4}\b")

_DOC_TYPE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cdrl", ("cdrl", "contract data requirements list")),
    ("requirements", ("shall", "requirement", "requirements", "srs", "srd")),
    ("wbs", ("wbs", "work breakdown structure")),
    ("schedule", ("schedule", "ims", "milestone")),
    ("status_report", ("status report", "monthly status", "weekly status", "program status")),
    ("test_report", ("test report", "verification", "validation", "test procedure")),
    ("design", ("design", "architecture", "interface control", "icd")),
    ("presentation", ("slide", "presentation", "briefing")),
    ("spreadsheet", ("workbook", "worksheet", "sheet:", "columns:")),
)
_LIFECYCLE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("archived", ("archived", "retired")),
    ("superseded", ("superseded", "obsolete")),
    ("released", ("released", "baselined")),
    ("approved", ("approved", "accepted")),
    ("review", ("in review", "under review", "redline", "review")),
    ("draft", ("draft", "preliminary", "working copy")),
)


def enrich_corpus_metadata(
    *,
    path: Path,
    raw_docs: Iterable[Document],
    source_metadata: dict[str, Any] | None = None,
    document_metadata: dict[str, Any] | None = None,
    providers: object | None = None,
    metadata_enrichment: str = "llm",
) -> dict[str, Any]:
    docs = list(raw_docs)
    source = dict(source_metadata or {})
    document_index = dict(document_metadata or {})
    full_text = "\n\n".join(str(doc.page_content or "") for doc in docs)
    sample = full_text[:8000]
    deterministic = _deterministic_metadata(
        path=path,
        full_text=full_text,
        raw_docs=docs,
        source_metadata=source,
        document_metadata=document_index,
    )
    warnings = list(deterministic.get("warnings") or [])
    mode = str(metadata_enrichment or "deterministic").strip().lower() or "deterministic"
    llm_payload: dict[str, Any] = {}
    llm_used = False
    llm_error = ""

    if mode in {"llm", "auto", "model"}:
        model = _select_model(providers)
        if model is None:
            warnings.append("LLM metadata enrichment requested but no runtime provider was available; deterministic metadata used.")
            deterministic["metadata_confidence"] = min(float(deterministic.get("metadata_confidence") or 0.5), 0.49)
        else:
            try:
                llm_payload = _call_llm_metadata(model, path=path, text_sample=sample, deterministic=deterministic)
                if llm_payload:
                    llm_used = True
                    deterministic = _merge_llm_metadata(deterministic, llm_payload)
                else:
                    llm_error = "LLM returned no parseable JSON."
            except Exception as exc:
                llm_error = str(exc)
            if llm_error:
                warnings.append(f"LLM metadata enrichment failed; deterministic metadata used: {llm_error}")
                deterministic["metadata_confidence"] = min(float(deterministic.get("metadata_confidence") or 0.5), 0.45)

    deterministic["warnings"] = _unique(warnings)
    deterministic["metadata_enrichment"] = "llm" if llm_used else "deterministic"
    deterministic["llm_enrichment"] = {
        "requested": mode in {"llm", "auto", "model"},
        "used": llm_used,
        "error": llm_error,
    }
    deterministic["signal_summary"] = _signal_summary(deterministic.get("signals"))
    return deterministic


def _deterministic_metadata(
    *,
    path: Path,
    full_text: str,
    raw_docs: list[Document],
    source_metadata: dict[str, Any],
    document_metadata: dict[str, Any],
) -> dict[str, Any]:
    haystack = f"{path.name}\n{path.parent}\n{full_text[:20000]}"
    lower = haystack.casefold()
    props = _document_properties(raw_docs)
    dates = _matches(_DATE_RE, full_text, label="date", limit=20)
    authors = _unique(
        [
            *(_string_list(source_metadata.get("authors") or source_metadata.get("author"))),
            *(_string_list(props.get("author"))),
            *[match["value"].strip(" .") for match in _matches(_AUTHOR_RE, full_text, label="author", group=1, limit=10)],
        ]
    )
    revision = str(source_metadata.get("revision") or props.get("revision") or props.get("version") or "").strip()
    if not revision:
        match = _REVISION_RE.search(full_text[:12000])
        if match:
            revision = match.group(1)
    signals = {
        "cdrl": _matches(_CDRL_RE, full_text, label="cdrl", limit=20),
        "wbs": _matches(_WBS_RE, full_text, label="wbs", limit=20),
        "requirements": _requirement_signals(full_text),
        "status": _matches(_STATUS_RE, full_text, label="status", limit=20),
    }
    doc_type = _first_rule(lower, _DOC_TYPE_RULES) or _doc_type_from_suffix(path)
    lifecycle_phase = _first_rule(lower, _LIFECYCLE_RULES) or "unknown"
    program_entities = _unique(
        [
            *(_string_list(source_metadata.get("program_entities"))),
            *[match.group(0).strip() for match in _ENTITY_RE.finditer(haystack[:20000])],
            *[str(item) for item in list(document_metadata.get("entities") or [])[:20]],
        ]
    )[:30]
    confidence = 0.35
    if doc_type != "unknown":
        confidence += 0.12
    if lifecycle_phase != "unknown":
        confidence += 0.10
    if program_entities:
        confidence += 0.08
    if authors or revision or dates:
        confidence += 0.08
    if any(signals.values()):
        confidence += 0.15
    if document_metadata.get("confidence"):
        confidence += min(0.1, max(0.0, float(document_metadata.get("confidence") or 0) * 0.1))
    warnings: list[str] = []
    if confidence < 0.55:
        warnings.append("Low-confidence metadata: deterministic signals were sparse.")
    return {
        "schema_version": CORPUS_METADATA_VERSION,
        "doc_type": doc_type,
        "lifecycle_phase": lifecycle_phase,
        "program_entities": program_entities,
        "dates": dates,
        "authors": authors,
        "revision": revision,
        "signals": signals,
        "metadata_confidence": round(min(confidence, 0.95), 2),
        "evidence": {
            "source_path": str(path),
            "filename": path.name,
            "document_property_keys": sorted(props.keys()),
        },
        "warnings": warnings,
    }


def _call_llm_metadata(model: Any, *, path: Path, text_sample: str, deterministic: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        "Return strict JSON only. Extract metadata for corpus governance using only the source text. "
        "Schema: {\"doc_type\":\"...\",\"lifecycle_phase\":\"draft|review|approved|released|superseded|archived|unknown\","
        "\"program_entities\":[\"...\"],\"authors\":[\"...\"],\"dates\":[{\"value\":\"...\",\"type\":\"...\",\"evidence\":\"...\",\"confidence\":0.0}],"
        "\"revision\":\"...\",\"signals\":{\"cdrl\":[{\"value\":\"...\",\"evidence\":\"...\",\"confidence\":0.0}],"
        "\"wbs\":[{\"value\":\"...\",\"evidence\":\"...\",\"confidence\":0.0}],"
        "\"requirements\":[{\"value\":\"...\",\"evidence\":\"...\",\"confidence\":0.0}],"
        "\"status\":[{\"value\":\"...\",\"evidence\":\"...\",\"confidence\":0.0}]},\"metadata_confidence\":0.0}. "
        "Use short evidence snippets and omit guesses.\n\n"
        f"FILE: {path.name}\nDETERMINISTIC PRIOR: {deterministic}\nTEXT SAMPLE:\n{text_sample}"
    )
    response = model.invoke(
        [
            SystemMessage(content="You extract constrained document metadata for a RAG corpus ingestion pipeline."),
            HumanMessage(content=prompt),
        ]
    )
    payload = extract_json(getattr(response, "content", None) or str(response)) or {}
    return payload if isinstance(payload, dict) else {}


def _merge_llm_metadata(base: dict[str, Any], llm_payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key in ("doc_type", "lifecycle_phase", "revision"):
        value = str(llm_payload.get(key) or "").strip()
        if value:
            merged[key] = value[:120]
    for key in ("program_entities", "authors"):
        merged[key] = _unique([*list(merged.get(key) or []), *_string_list(llm_payload.get(key))])[:40]
    if isinstance(llm_payload.get("dates"), list):
        merged["dates"] = _merge_signal_lists(list(merged.get("dates") or []), llm_payload.get("dates"), label="date")
    if isinstance(llm_payload.get("signals"), dict):
        base_signals = dict(merged.get("signals") or {})
        for signal_key in ("cdrl", "wbs", "requirements", "status"):
            base_signals[signal_key] = _merge_signal_lists(
                list(base_signals.get(signal_key) or []),
                llm_payload["signals"].get(signal_key),
                label=signal_key,
            )
        merged["signals"] = base_signals
    try:
        llm_confidence = float(llm_payload.get("metadata_confidence"))
    except Exception:
        llm_confidence = 0.0
    merged["metadata_confidence"] = round(min(0.98, max(float(merged.get("metadata_confidence") or 0.5), llm_confidence)), 2)
    return merged


def _merge_signal_lists(existing: list[Any], incoming: Any, *, label: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*existing, *list(incoming or [])]:
        if isinstance(item, dict):
            value = str(item.get("value") or item.get("text") or "").strip()
            evidence = str(item.get("evidence") or "").strip()
            confidence = item.get("confidence", 0.65)
        else:
            value = str(item or "").strip()
            evidence = value
            confidence = 0.6
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        try:
            score = float(confidence)
        except Exception:
            score = 0.6
        normalized.append(
            {
                "value": value[:160],
                "type": label,
                "evidence": evidence[:240] if evidence else value[:160],
                "confidence": round(max(0.0, min(1.0, score)), 2),
            }
        )
    return normalized[:30]


def _matches(
    pattern: re.Pattern[str],
    text: str,
    *,
    label: str,
    group: int = 0,
    limit: int = 20,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in pattern.finditer(text or ""):
        value = match.group(group).strip() if group else match.group(0).strip()
        key = value.casefold()
        if not value or key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "value": value[:160],
                "type": label,
                "evidence": _snippet(text, match.start(), match.end()),
                "confidence": 0.72,
            }
        )
        if len(items) >= limit:
            break
    return items


def _requirement_signals(text: str) -> list[dict[str, Any]]:
    items = _matches(_REQ_ID_RE, text, label="requirement", limit=20)
    for match in _REQ_VERB_RE.finditer(text or ""):
        sentence = _sentence_snippet(text, match.start())
        if not sentence:
            continue
        key = sentence.casefold()
        if any(str(item.get("value") or "").casefold() == key for item in items):
            continue
        items.append({"value": sentence[:160], "type": "requirement", "evidence": sentence[:240], "confidence": 0.62})
        if len(items) >= 30:
            break
    return items


def _signal_summary(signals: Any) -> dict[str, Any]:
    payload = dict(signals or {}) if isinstance(signals, dict) else {}
    return {
        key: {
            "count": len(list(value or [])),
            "sample": [str(item.get("value") if isinstance(item, dict) else item) for item in list(value or [])[:5]],
        }
        for key, value in payload.items()
    }


def _select_model(providers: object | None) -> Any | None:
    if providers is None:
        return None
    return getattr(providers, "chat", None) or getattr(providers, "judge", None)


def _document_properties(raw_docs: list[Document]) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for doc in raw_docs:
        metadata = dict(doc.metadata or {})
        doc_props = metadata.get("document_properties")
        if isinstance(doc_props, dict):
            props.update({str(key): value for key, value in doc_props.items() if value not in ("", None)})
    return props


def _doc_type_from_suffix(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"pptx", "ppt"}:
        return "presentation"
    if suffix in {"xlsx", "xls", "csv", "tsv"}:
        return "spreadsheet"
    if suffix in {"pdf", "docx", "txt", "md"}:
        return "document"
    return "unknown"


def _first_rule(text: str, rules: tuple[tuple[str, tuple[str, ...]], ...]) -> str:
    for label, needles in rules:
        if any(needle in text for needle in needles):
            return label
    return ""


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"[,;]\s*", text) if part.strip()]


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        result.append(text)
    return result


def _snippet(text: str, start: int, end: int, *, radius: int = 90) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return " ".join(text[left:right].split())[:240]


def _sentence_snippet(text: str, index: int) -> str:
    left = max(text.rfind(".", 0, index), text.rfind("\n", 0, index))
    right_candidates = [candidate for candidate in (text.find(".", index), text.find("\n", index)) if candidate >= 0]
    right = min(right_candidates) if right_candidates else min(len(text), index + 220)
    return " ".join(text[left + 1 : right].split())
