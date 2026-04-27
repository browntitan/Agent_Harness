from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence

from agentic_chatbot_next.persistence.postgres.chunks import ChunkRecord
from agentic_chatbot_next.persistence.postgres.documents import DocumentRecord
from agentic_chatbot_next.persistence.postgres.requirements import RequirementStatementRecord

REQUIREMENT_EXTRACTOR_VERSION = "requirements_v2"
SUPPORTED_REQUIREMENTS_FILE_TYPES = {"pdf", "docx", "md", "txt"}
STRICT_SHALL_MODE = "strict_shall"
MANDATORY_MODE = "mandatory"
LEGAL_CLAUSE_MODE = "legal_clause"
BROAD_REQUIREMENT_MODE = "broad_requirement"
SUPPORTED_REQUIREMENT_MODES = {
    BROAD_REQUIREMENT_MODE,
    MANDATORY_MODE,
    STRICT_SHALL_MODE,
    LEGAL_CLAUSE_MODE,
}

_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|[A-Za-z][.)])\s+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=(?:[A-Z0-9(]|[-*•]))")
_SEMICOLON_BOUNDARY_RE = re.compile(r";\s+(?=(?:[A-Z0-9(]|[-*•]))")
_TOC_FRAGMENT_RE = re.compile(r"\.{4,}\s*\d+\s*$")
_HEADING_LIKE_RE = re.compile(r"^[A-Z][A-Za-z0-9 /&()_:-]{0,100}$")
_EXAMPLE_PREFIX_RE = re.compile(r"^\s*(?:example|note|notes|e\.g\.)[:\s-]", re.IGNORECASE)
_PIPE_ROW_RE = re.compile(r"^[^|\n]+(?:\|[^|\n]+)+$")
_EXTRACTION_META_RE = re.compile(
    r"\b(?:requirements?\s+extraction|extracted\s+shall\s+statements?|synthetic\s+test\s+document|"
    r"exercise\s+requirements?\s+extraction\s+workflows?|realistic\s+engineering\s+prose)\b",
    re.IGNORECASE,
)
_POSITIVE_SECTION_HINT_RE = re.compile(
    r"\b(?:requirement|requirements|statement\s+of\s+work|performance|interface|deliverable|task|"
    r"obligation|compliance|applicable\s+documents|quality|inspection|packaging|support|operations)\b",
    re.IGNORECASE,
)
_NEGATIVE_SECTION_HINT_RE = re.compile(
    r"\b(?:verification|revision\s+history|table\s+of\s+contents|contents|references|bibliography|appendix)\b",
    re.IGNORECASE,
)
_ACTOR_RE = re.compile(
    r"\b(?:the\s+)?(?:contractor|subcontractor|offeror|supplier|vendor|provider|government|agency|"
    r"system|subsystem|platform|service|operator|administrator|user|equipment|software|hardware)\b",
    re.IGNORECASE,
)
_IMPERATIVE_RE = re.compile(
    r"^\s*(?:provide|maintain|support|submit|identify|document|record|protect|enable|ensure|verify|"
    r"archive|retain|display|route|reject|fabricate|coordinate|perform|preserve|establish|"
    r"implement|monitor|report|use|apply)\b",
    re.IGNORECASE,
)
_THRESHOLD_RE = re.compile(
    r"\b(?:at\s+least|not\s+less\s+than|no\s+more\s+than|within|greater\s+than|less\s+than|minimum|"
    r"maximum|threshold|seconds?|minutes?|hours?|days?|percent|availability|latency|throughput)\b",
    re.IGNORECASE,
)
_REQ_ID_RE = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+){1,}\b")

_EXPLICIT_MODALITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("shall_not", re.compile(r"\bshall\s+not\b", re.IGNORECASE)),
    ("must_not", re.compile(r"\bmust\s+not\b", re.IGNORECASE)),
    ("required_to", re.compile(r"\b(?:is|are)\s+required\s+to\b", re.IGNORECASE)),
    (
        "prohibited",
        re.compile(
            r"\b(?:is|are)\s+prohibited\s+from\b"
            r"|\b(?:is|are)\s+not\s+permitted\s+to\b"
            r"|\bmay\s+not\b",
            re.IGNORECASE,
        ),
    ),
    ("shall", re.compile(r"\bshall\b", re.IGNORECASE)),
    ("must", re.compile(r"\bmust\b", re.IGNORECASE)),
    ("required", re.compile(r"\b(?:is|are|be)\s+required\b", re.IGNORECASE)),
    ("responsible_for", re.compile(r"\b(?:is|are|be)\s+responsible\s+for\b", re.IGNORECASE)),
    ("agrees_to", re.compile(r"\b(?:agrees|agree)\s+to\b", re.IGNORECASE)),
    (
        "will",
        re.compile(
            r"\b(?:contractor|subcontractor|offeror|supplier|vendor|provider|agency|government|system|service)\s+will\b",
            re.IGNORECASE,
        ),
    ),
    ("permitted", re.compile(r"\b(?:is|are)\s+permitted\s+to\b", re.IGNORECASE)),
    ("may", re.compile(r"\bmay\b", re.IGNORECASE)),
    ("can", re.compile(r"\bcan\b", re.IGNORECASE)),
)

_STRICT_MODALITIES = {"shall", "shall_not"}
_MANDATORY_MODALITIES = {"shall", "shall_not", "must", "must_not", "required_to", "prohibited"}
_LEGAL_MODALITIES = {
    *_MANDATORY_MODALITIES,
    "required",
    "responsible_for",
    "agrees_to",
    "will",
}
_INFERRED_MODALITIES = {"imperative", "constraint", "table_row"}
_BROAD_MODALITIES = tuple(
    dict.fromkeys([name for name, _ in _EXPLICIT_MODALITY_PATTERNS] + sorted(_INFERRED_MODALITIES))
)
_PROHIBITIVE_MODALITIES = {"shall_not", "must_not", "prohibited"}
_PERMISSIVE_MODALITIES = {"may", "can", "permitted"}


@dataclass(frozen=True)
class TextSlice:
    text: str
    start: int
    end: int

    def trimmed(self) -> "TextSlice":
        raw = self.text
        if not raw:
            return self
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        start = self.start + leading
        end = self.end - trailing
        return TextSlice(text=raw.strip(), start=max(self.start, start), end=max(start, end))


@dataclass(frozen=True)
class RequirementCandidate:
    text: str
    source_excerpt: str
    normalized_text: str
    start: int
    end: int
    modality: str
    source_structure: str
    chunk_id: str
    chunk_index: int
    chunk_type: str
    page_number: Optional[int]
    clause_number: str
    section_title: str
    multi_requirement: bool
    merged_chunk_ids: tuple[str, ...] = ()
    merged_source_locations: tuple[str, ...] = ()


@dataclass(frozen=True)
class RequirementAssessment:
    keep: bool
    modality: str
    binding_strength: str
    confidence: float
    risk_label: str
    risk_rationale: str
    requirement_text: str
    source_structure: str


@dataclass(frozen=True)
class RequirementInventoryBuild:
    records: List[RequirementStatementRecord]
    candidate_count: int
    dedupe_count: int
    kept_count: int
    dropped_count: int


RequirementClassifier = Callable[[RequirementCandidate, RequirementAssessment], RequirementAssessment | None]


def normalize_requirement_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    return normalized if normalized in SUPPORTED_REQUIREMENT_MODES else BROAD_REQUIREMENT_MODE


def requirement_modalities_for_mode(mode: str) -> tuple[str, ...]:
    normalized = normalize_requirement_mode(mode)
    if normalized == STRICT_SHALL_MODE:
        return tuple(sorted(_STRICT_MODALITIES))
    if normalized == LEGAL_CLAUSE_MODE:
        return tuple(sorted(_LEGAL_MODALITIES))
    if normalized == MANDATORY_MODE:
        return tuple(sorted(_MANDATORY_MODALITIES))
    return tuple(_BROAD_MODALITIES)


def supports_requirements_extraction(file_type: str) -> bool:
    return str(file_type or "").strip().lower() in SUPPORTED_REQUIREMENTS_FILE_TYPES


def normalize_requirement_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def format_requirement_location(record: RequirementStatementRecord | RequirementCandidate | dict[str, Any]) -> str:
    payload = record if isinstance(record, dict) else record.__dict__
    parts: List[str] = []
    page_number = payload.get("page_number")
    clause_number = str(payload.get("clause_number") or "").strip()
    section_title = str(payload.get("section_title") or "").strip()
    chunk_index = int(payload.get("chunk_index") or 0)
    char_start = int(payload.get("char_start", payload.get("start", 0)) or 0)
    if page_number:
        parts.append(f"Page {page_number}")
    if clause_number:
        parts.append(f"Clause {clause_number}")
    if section_title:
        parts.append(section_title)
    if not parts:
        parts.append(f"Chunk {chunk_index}")
    parts.append(f"Offset {char_start}")
    return " · ".join(parts)


def build_requirement_statement_records(
    document: DocumentRecord,
    chunks: Sequence[ChunkRecord],
    *,
    mode: str = BROAD_REQUIREMENT_MODE,
    classifier: RequirementClassifier | None = None,
) -> List[RequirementStatementRecord]:
    return build_requirement_inventory(document, chunks, mode=mode, classifier=classifier).records


def build_requirement_inventory(
    document: DocumentRecord,
    chunks: Sequence[ChunkRecord],
    *,
    mode: str = BROAD_REQUIREMENT_MODE,
    classifier: RequirementClassifier | None = None,
) -> RequirementInventoryBuild:
    normalized_mode = normalize_requirement_mode(mode)
    raw_candidates: List[RequirementCandidate] = []
    for chunk in sorted(chunks, key=lambda item: int(getattr(item, "chunk_index", 0) or 0)):
        raw_candidates.extend(_extract_chunk_candidates(chunk))

    deduped_candidates = _dedupe_candidates(raw_candidates)
    records: List[RequirementStatementRecord] = []
    next_index = 0
    for candidate in deduped_candidates:
        deterministic = _assess_candidate_deterministically(candidate)
        resolved = _merge_assessments(
            candidate,
            deterministic,
            classifier(candidate, deterministic) if classifier is not None else None,
        )
        if not resolved.keep:
            continue
        if not _mode_allows_modality(resolved.modality, mode=normalized_mode):
            continue
        requirement_id = f"{document.doc_id}#req{next_index:05d}"
        merged_chunk_ids = candidate.merged_chunk_ids or (candidate.chunk_id,)
        merged_source_locations = candidate.merged_source_locations or (format_requirement_location(candidate),)
        records.append(
            RequirementStatementRecord(
                requirement_id=requirement_id,
                doc_id=document.doc_id,
                tenant_id=document.tenant_id,
                collection_id=document.collection_id,
                source_type=document.source_type,
                document_title=document.title,
                statement_index=next_index,
                chunk_id=candidate.chunk_id,
                chunk_index=candidate.chunk_index,
                statement_text=resolved.requirement_text,
                normalized_statement_text=normalize_requirement_text(resolved.requirement_text).casefold(),
                modality=resolved.modality,
                page_number=candidate.page_number,
                clause_number=candidate.clause_number,
                section_title=candidate.section_title,
                char_start=int(candidate.start),
                char_end=int(candidate.end),
                multi_requirement=bool(candidate.multi_requirement),
                source_excerpt=candidate.source_excerpt,
                source_structure=resolved.source_structure,
                binding_strength=resolved.binding_strength,
                confidence=float(resolved.confidence),
                risk_label=resolved.risk_label,
                risk_rationale=resolved.risk_rationale,
                dedupe_group_id=requirement_id,
                duplicate_count=max(1, len(merged_source_locations)),
                merged_chunk_ids="; ".join(dict.fromkeys(str(item) for item in merged_chunk_ids if str(item))),
                merged_source_locations=" | ".join(
                    dict.fromkeys(str(item) for item in merged_source_locations if str(item))
                ),
                extractor_version=REQUIREMENT_EXTRACTOR_VERSION,
                extractor_mode=BROAD_REQUIREMENT_MODE,
            )
        )
        next_index += 1

    return RequirementInventoryBuild(
        records=records,
        candidate_count=len(raw_candidates),
        dedupe_count=max(0, len(raw_candidates) - len(deduped_candidates)),
        kept_count=len(records),
        dropped_count=max(0, len(deduped_candidates) - len(records)),
    )


def _extract_chunk_candidates(chunk: ChunkRecord) -> List[RequirementCandidate]:
    content = str(chunk.content or "")
    if not content.strip():
        return []
    extracted: List[RequirementCandidate] = []
    seen: set[tuple[int, int, str]] = set()
    for block in _split_blocks(content):
        trimmed_block = block.trimmed()
        if not trimmed_block.text:
            continue
        if _is_table_like_text(trimmed_block.text):
            for row in _extract_table_row_slices(trimmed_block):
                _append_candidate(extracted, seen, row, chunk, source_structure="table_row")
            continue
        units = _split_line_units(trimmed_block) or _split_list_units(trimmed_block)
        if not units:
            units = [trimmed_block]
        for unit in units:
            trimmed_unit = unit.trimmed()
            if not trimmed_unit.text:
                continue
            if _is_table_like_text(trimmed_unit.text):
                for row in _extract_table_row_slices(trimmed_unit):
                    _append_candidate(extracted, seen, row, chunk, source_structure="table_row")
                continue
            structure = "list_item" if _LIST_ITEM_RE.match(trimmed_unit.text) else "paragraph"
            fragments = _split_semicolon_fragments(trimmed_unit)
            sentences = _split_sentences(trimmed_unit)
            append_parent = not fragments and len(sentences) <= 1
            if append_parent:
                _append_candidate(extracted, seen, trimmed_unit, chunk, source_structure=structure)
            for fragment in fragments:
                _append_candidate(extracted, seen, fragment, chunk, source_structure="fragment")
            for sentence in sentences:
                _append_candidate(extracted, seen, sentence, chunk, source_structure="sentence")
    return extracted


def _append_candidate(
    extracted: List[RequirementCandidate],
    seen: set[tuple[int, int, str]],
    raw_slice: TextSlice,
    chunk: ChunkRecord,
    *,
    source_structure: str,
) -> None:
    candidate = raw_slice.trimmed()
    text = normalize_requirement_text(candidate.text)
    if not text:
        return
    key = (candidate.start, candidate.end, source_structure)
    if key in seen:
        return
    seen.add(key)
    location = format_requirement_location(
        {
            "page_number": chunk.page_number,
            "clause_number": str(chunk.clause_number or ""),
            "section_title": str(chunk.section_title or ""),
            "chunk_index": int(chunk.chunk_index or 0),
            "char_start": int(candidate.start),
        }
    )
    extracted.append(
        RequirementCandidate(
            text=text,
            source_excerpt=text,
            normalized_text=text.casefold(),
            start=int(candidate.start),
            end=int(candidate.end),
            modality=_detect_modality(text, mode=BROAD_REQUIREMENT_MODE),
            source_structure=source_structure,
            chunk_id=str(chunk.chunk_id or ""),
            chunk_index=int(chunk.chunk_index or 0),
            chunk_type=str(chunk.chunk_type or ""),
            page_number=chunk.page_number,
            clause_number=str(chunk.clause_number or ""),
            section_title=str(chunk.section_title or ""),
            multi_requirement=_count_mandatory_operators(text, mode=BROAD_REQUIREMENT_MODE) > 1,
            merged_chunk_ids=(str(chunk.chunk_id or ""),),
            merged_source_locations=(location,),
        )
    )


def _dedupe_candidates(candidates: Sequence[RequirementCandidate]) -> List[RequirementCandidate]:
    deduped: List[RequirementCandidate] = []
    for candidate in candidates:
        merged = False
        for index, existing in enumerate(deduped):
            if not _can_merge_candidate(existing, candidate):
                continue
            deduped[index] = _merge_candidates(existing, candidate)
            merged = True
            break
        if not merged:
            deduped.append(candidate)
    return deduped


def _can_merge_candidate(left: RequirementCandidate, right: RequirementCandidate) -> bool:
    if left.normalized_text != right.normalized_text:
        return False
    if left.clause_number and right.clause_number and left.clause_number != right.clause_number:
        return False
    if left.section_title and right.section_title and left.section_title.casefold() != right.section_title.casefold():
        return False
    if left.page_number and right.page_number and left.page_number != right.page_number:
        return False
    if abs(int(left.chunk_index or 0) - int(right.chunk_index or 0)) > 2:
        return False
    return True


def _merge_candidates(left: RequirementCandidate, right: RequirementCandidate) -> RequirementCandidate:
    merged_chunk_ids = tuple(
        dict.fromkeys(
            [*left.merged_chunk_ids, *right.merged_chunk_ids]
            or [left.chunk_id, right.chunk_id]
        )
    )
    merged_locations = tuple(
        dict.fromkeys(
            [*left.merged_source_locations, *right.merged_source_locations]
            or [format_requirement_location(left), format_requirement_location(right)]
        )
    )
    preferred = left if _candidate_priority(left) >= _candidate_priority(right) else right
    return RequirementCandidate(
        text=preferred.text,
        source_excerpt=preferred.source_excerpt,
        normalized_text=preferred.normalized_text,
        start=preferred.start,
        end=preferred.end,
        modality=preferred.modality or left.modality or right.modality,
        source_structure=preferred.source_structure,
        chunk_id=preferred.chunk_id,
        chunk_index=preferred.chunk_index,
        chunk_type=preferred.chunk_type,
        page_number=preferred.page_number,
        clause_number=preferred.clause_number,
        section_title=preferred.section_title,
        multi_requirement=bool(left.multi_requirement or right.multi_requirement),
        merged_chunk_ids=merged_chunk_ids,
        merged_source_locations=merged_locations,
    )


def _candidate_priority(candidate: RequirementCandidate) -> int:
    structure_rank = {
        "table_row": 4,
        "list_item": 3,
        "paragraph": 2,
        "sentence": 1,
        "fragment": 0,
    }
    explicit_rank = 5 if candidate.modality else 0
    return explicit_rank + structure_rank.get(candidate.source_structure, 0)


def _assess_candidate_deterministically(candidate: RequirementCandidate) -> RequirementAssessment:
    text = normalize_requirement_text(candidate.text)
    modality = candidate.modality or _detect_modality(text, mode=BROAD_REQUIREMENT_MODE)
    source_structure = candidate.source_structure
    requirement_like = _is_requirement_like_candidate(candidate, text)
    negative_context = _NEGATIVE_SECTION_HINT_RE.search(str(candidate.section_title or "")) is not None
    if _should_skip_candidate(text):
        return RequirementAssessment(
            keep=False,
            modality=modality or "",
            binding_strength="",
            confidence=0.1,
            risk_label="possible narrative false positive",
            risk_rationale="This fragment looks like scaffolding, heading text, or explanatory prose rather than a requirement.",
            requirement_text=text,
            source_structure=source_structure,
        )
    if _EXTRACTION_META_RE.search(text):
        return RequirementAssessment(
            keep=False,
            modality=modality or "",
            binding_strength="",
            confidence=0.08,
            risk_label="possible narrative false positive",
            risk_rationale="This fragment discusses the extraction workflow itself instead of stating a document requirement.",
            requirement_text=text,
            source_structure=source_structure,
        )

    keep = False
    binding_strength = ""
    risk_label = "ambiguous wording for requirement"
    risk_rationale = "The wording is requirement-like, but it depends on surrounding context or weaker structural cues."

    if modality in _PROHIBITIVE_MODALITIES:
        keep = True
        binding_strength = "prohibitive"
        risk_label = "clear prohibition with bounded scope"
        risk_rationale = "The statement uses explicit prohibitive language that reads as a bounded requirement."
    elif modality in _MANDATORY_MODALITIES or modality in {"required", "responsible_for", "agrees_to", "will"}:
        keep = True
        binding_strength = "mandatory"
        risk_label = "clear, direct statement with minimal dependencies"
        risk_rationale = "The statement uses explicit mandatory language and stands alone as a requirement candidate."
    elif modality in _PERMISSIVE_MODALITIES:
        keep = requirement_like
        binding_strength = "permissive"
        risk_label = "permissive or discretionary language; possible requirement candidate"
        risk_rationale = "The wording is discretionary, so it may reflect policy, permission, or a weak requirement depending on context."
    elif source_structure == "table_row" and _row_has_requirement_signal(text):
        keep = requirement_like
        modality = "table_row"
        binding_strength = "contextual"
        risk_label = "table/list fragment reconstructed from structured content"
        risk_rationale = "This row was reconstructed from structured content and may encode a requirement even without explicit modal verbs."
    elif _IMPERATIVE_RE.match(text) and requirement_like:
        keep = True
        modality = "imperative"
        binding_strength = "mandatory"
        risk_label = "context-dependent requirement that relies on surrounding text"
        risk_rationale = "The requirement is expressed as an imperative fragment and depends on nearby document structure for full interpretation."
    elif _THRESHOLD_RE.search(text) and requirement_like and (_ACTOR_RE.search(text) or _REQ_ID_RE.search(text)):
        keep = True
        modality = "constraint"
        binding_strength = "mandatory"
        risk_label = "context-dependent requirement that relies on surrounding text"
        risk_rationale = "The text expresses a threshold or bounded constraint that looks requirement-like in context."

    if keep and candidate.multi_requirement:
        risk_label = "compound requirement with multiple obligations"
        risk_rationale = "The extracted text contains more than one obligation and may need downstream splitting for clean traceability."

    if keep and source_structure in {"table_row", "list_item"} and modality in {"table_row", "imperative", "constraint"}:
        risk_label = "table/list fragment reconstructed from structured content"
        risk_rationale = "The extracted candidate came from structured content and may depend on table/list context for a complete reading."

    if keep and negative_context and modality in _PERMISSIVE_MODALITIES | {"imperative", "constraint", "table_row"}:
        risk_label = "context-dependent requirement that relies on surrounding text"
        risk_rationale = "The surrounding section is not requirement-first, so this candidate should be reviewed in context."

    confidence = _confidence_for_candidate(
        modality=modality,
        binding_strength=binding_strength,
        source_structure=source_structure,
        multi_requirement=bool(candidate.multi_requirement),
        negative_context=negative_context,
        keep=keep,
    )
    return RequirementAssessment(
        keep=keep,
        modality=modality or "",
        binding_strength=binding_strength,
        confidence=confidence,
        risk_label=risk_label,
        risk_rationale=risk_rationale,
        requirement_text=text,
        source_structure=source_structure,
    )


def _merge_assessments(
    candidate: RequirementCandidate,
    deterministic: RequirementAssessment,
    model_result: RequirementAssessment | None,
) -> RequirementAssessment:
    if model_result is None:
        return deterministic
    if deterministic.keep and not model_result.keep:
        return RequirementAssessment(
            keep=True,
            modality=deterministic.modality,
            binding_strength=deterministic.binding_strength,
            confidence=max(deterministic.confidence, min(model_result.confidence, 0.75)),
            risk_label=deterministic.risk_label,
            risk_rationale=deterministic.risk_rationale,
            requirement_text=deterministic.requirement_text,
            source_structure=deterministic.source_structure,
        )
    if model_result.keep:
        return RequirementAssessment(
            keep=True,
            modality=model_result.modality or deterministic.modality,
            binding_strength=model_result.binding_strength or deterministic.binding_strength,
            confidence=max(deterministic.confidence, model_result.confidence)
            if deterministic.keep
            else model_result.confidence,
            risk_label=model_result.risk_label or deterministic.risk_label,
            risk_rationale=model_result.risk_rationale or deterministic.risk_rationale,
            requirement_text=model_result.requirement_text or deterministic.requirement_text or candidate.text,
            source_structure=model_result.source_structure or deterministic.source_structure or candidate.source_structure,
        )
    return deterministic


def _confidence_for_candidate(
    *,
    modality: str,
    binding_strength: str,
    source_structure: str,
    multi_requirement: bool,
    negative_context: bool,
    keep: bool,
) -> float:
    if not keep:
        return 0.12
    if modality in _PROHIBITIVE_MODALITIES:
        score = 0.95
    elif modality in _MANDATORY_MODALITIES:
        score = 0.92
    elif modality in {"required", "responsible_for", "agrees_to", "will"}:
        score = 0.82
    elif modality in _PERMISSIVE_MODALITIES:
        score = 0.58
    elif modality == "imperative":
        score = 0.72
    elif modality in {"constraint", "table_row"}:
        score = 0.66
    else:
        score = 0.55 if binding_strength else 0.42
    if source_structure == "table_row":
        score -= 0.05
    if multi_requirement:
        score -= 0.06
    if negative_context and modality in _PERMISSIVE_MODALITIES | {"imperative", "constraint", "table_row"}:
        score -= 0.08
    return max(0.05, min(0.99, round(score, 2)))


def _is_requirement_like_candidate(candidate: RequirementCandidate, text: str) -> bool:
    if candidate.chunk_type in {"requirement", "clause"}:
        return True
    section_title = str(candidate.section_title or "")
    if section_title and _POSITIVE_SECTION_HINT_RE.search(section_title):
        return True
    if source_hint := candidate.source_structure:
        if source_hint in {"list_item", "table_row"} and (_ACTOR_RE.search(text) or _THRESHOLD_RE.search(text)):
            return True
    if _ACTOR_RE.search(text) and (_THRESHOLD_RE.search(text) or _IMPERATIVE_RE.match(text) or candidate.modality):
        return True
    return bool(_REQ_ID_RE.search(text) and (_THRESHOLD_RE.search(text) or candidate.modality))


def _row_has_requirement_signal(text: str) -> bool:
    if _ACTOR_RE.search(text):
        return True
    if _THRESHOLD_RE.search(text):
        return True
    if _IMPERATIVE_RE.match(text):
        return True
    return bool(_REQ_ID_RE.search(text))


def _mode_allows_modality(modality: str, *, mode: str) -> bool:
    allowed = set(requirement_modalities_for_mode(mode))
    if mode == BROAD_REQUIREMENT_MODE:
        return True
    return modality in allowed


def _split_blocks(text: str) -> List[TextSlice]:
    cursor = 0
    blocks: List[TextSlice] = []
    for match in re.finditer(r"\n\s*\n+", text):
        if match.start() > cursor:
            blocks.append(TextSlice(text=text[cursor:match.start()], start=cursor, end=match.start()))
        cursor = match.end()
    if cursor < len(text):
        blocks.append(TextSlice(text=text[cursor:], start=cursor, end=len(text)))
    return blocks


def _split_list_units(block: TextSlice) -> List[TextSlice]:
    lines = block.text.splitlines(keepends=True)
    if not lines:
        return []
    units: List[TextSlice] = []
    unit_start = 0
    cursor = 0
    saw_list = False
    for index, line in enumerate(lines):
        if index > 0 and _LIST_ITEM_RE.match(line):
            saw_list = True
            units.append(
                TextSlice(
                    text=block.text[unit_start:cursor],
                    start=block.start + unit_start,
                    end=block.start + cursor,
                )
            )
            unit_start = cursor
        cursor += len(line)
    if cursor > unit_start:
        units.append(
            TextSlice(
                text=block.text[unit_start:cursor],
                start=block.start + unit_start,
                end=block.start + cursor,
            )
        )
    return units if saw_list else []


def _split_line_units(block: TextSlice) -> List[TextSlice]:
    lines = block.text.splitlines(keepends=True)
    non_empty_lines = [line for line in lines if line.strip()]
    if len(non_empty_lines) <= 1:
        return []
    structured_lines = sum(
        1
        for line in non_empty_lines
        if _LIST_ITEM_RE.match(line) or _is_table_like_text(line.rstrip("\n"))
    )
    if structured_lines == 0 and len(non_empty_lines) < 3:
        return []
    units: List[TextSlice] = []
    cursor = 0
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if line.strip():
            units.append(
                TextSlice(
                    text=line,
                    start=block.start + cursor,
                    end=block.start + cursor + len(line),
                ).trimmed()
            )
        cursor += len(raw_line)
    return [item for item in units if item.text]


def _split_sentences(unit: TextSlice) -> List[TextSlice]:
    boundaries = [0]
    for match in _SENTENCE_BOUNDARY_RE.finditer(unit.text):
        boundaries.append(match.start())
    boundaries.append(len(unit.text))
    slices: List[TextSlice] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        slices.append(
            TextSlice(
                text=unit.text[start:end],
                start=unit.start + start,
                end=unit.start + end,
            ).trimmed()
        )
    return [item for item in slices if item.text]


def _split_semicolon_fragments(unit: TextSlice) -> List[TextSlice]:
    if "|" in unit.text or ";" not in unit.text:
        return []
    boundaries = [0]
    for match in _SEMICOLON_BOUNDARY_RE.finditer(unit.text):
        boundaries.append(match.end())
    boundaries.append(len(unit.text))
    slices: List[TextSlice] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        slices.append(
            TextSlice(
                text=unit.text[start:end],
                start=unit.start + start,
                end=unit.start + end,
            ).trimmed()
        )
    return [item for item in slices if item.text]


def _extract_table_row_slices(block: TextSlice) -> List[TextSlice]:
    rows: List[TextSlice] = []
    cursor = 0
    for raw_line in block.text.splitlines(keepends=True):
        clean = raw_line.rstrip("\n")
        if _is_table_like_text(clean):
            rows.append(
                TextSlice(
                    text=clean,
                    start=block.start + cursor,
                    end=block.start + cursor + len(clean),
                ).trimmed()
            )
        cursor += len(raw_line)
    if rows:
        return [row for row in rows if row.text]
    if _is_table_like_text(block.text):
        return [block.trimmed()]
    return []


def _should_skip_candidate(text: str) -> bool:
    normalized = normalize_requirement_text(text)
    if not normalized:
        return True
    lowered = normalized.casefold()
    if _TOC_FRAGMENT_RE.search(normalized):
        return True
    if _EXAMPLE_PREFIX_RE.match(normalized):
        return True
    if len(normalized) <= 80 and _HEADING_LIKE_RE.match(normalized) and not any(char in normalized for char in ".;:!?|"):
        return True
    if lowered.startswith(("appendix ", "table ", "figure ", "references", "bibliography")) and not _has_any_modality(normalized):
        return True
    return False


def _is_table_like_text(text: str) -> bool:
    raw = str(text or "")
    if "|" not in raw:
        return False
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) > 1:
        return all(_PIPE_ROW_RE.match(normalize_requirement_text(line)) is not None for line in lines)
    normalized = normalize_requirement_text(raw)
    return _PIPE_ROW_RE.match(normalized) is not None


def _has_any_modality(text: str) -> bool:
    return any(pattern.search(text) for _, pattern in _EXPLICIT_MODALITY_PATTERNS)


def _patterns_for_mode(mode: str) -> tuple[tuple[str, re.Pattern[str]], ...]:
    del mode
    return _EXPLICIT_MODALITY_PATTERNS


def _detect_modality(text: str, *, mode: str) -> str:
    allowed = set(requirement_modalities_for_mode(mode))
    if mode == BROAD_REQUIREMENT_MODE:
        allowed = {name for name, _ in _EXPLICIT_MODALITY_PATTERNS}
    best_name = ""
    best_start: int | None = None
    for name, pattern in _patterns_for_mode(mode):
        if name not in allowed:
            continue
        match = pattern.search(text)
        if match is None:
            continue
        if best_start is None or match.start() < best_start:
            best_name = name
            best_start = match.start()
    return best_name


def _count_mandatory_operators(text: str, *, mode: str = BROAD_REQUIREMENT_MODE) -> int:
    del mode
    count = 0
    for _, pattern in _EXPLICIT_MODALITY_PATTERNS:
        count += len(list(pattern.finditer(text)))
    return count


__all__ = [
    "BROAD_REQUIREMENT_MODE",
    "MANDATORY_MODE",
    "LEGAL_CLAUSE_MODE",
    "REQUIREMENT_EXTRACTOR_VERSION",
    "RequirementAssessment",
    "RequirementCandidate",
    "RequirementInventoryBuild",
    "STRICT_SHALL_MODE",
    "SUPPORTED_REQUIREMENT_MODES",
    "SUPPORTED_REQUIREMENTS_FILE_TYPES",
    "build_requirement_inventory",
    "build_requirement_statement_records",
    "format_requirement_location",
    "normalize_requirement_mode",
    "normalize_requirement_text",
    "requirement_modalities_for_mode",
    "supports_requirements_extraction",
]
