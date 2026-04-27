from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

from agentic_chatbot_next.documents.extractors import DocumentExtractionService, element_location
from agentic_chatbot_next.documents.models import (
    ChangedObligation,
    DocumentCompareResult,
    DocumentDelta,
    DocumentElement,
    DocumentExtractResult,
)
from agentic_chatbot_next.rag.requirements import BROAD_REQUIREMENT_MODE, requirement_modalities_for_mode


_NORMALIZE_RE = re.compile(r"\s+")
_CLAUSE_RE = re.compile(r"^\s*(?P<num>(?:\d+\.)+\d*|\d+)\b")
_OBLIGATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("shall_not", re.compile(r"\bshall\s+not\b", re.IGNORECASE)),
    ("must_not", re.compile(r"\bmust\s+not\b", re.IGNORECASE)),
    ("required_to", re.compile(r"\b(?:is|are)\s+required\s+to\b", re.IGNORECASE)),
    ("prohibited", re.compile(r"\b(?:prohibited\s+from|not\s+permitted\s+to|may\s+not)\b", re.IGNORECASE)),
    ("shall", re.compile(r"\bshall\b", re.IGNORECASE)),
    ("must", re.compile(r"\bmust\b", re.IGNORECASE)),
    ("required", re.compile(r"\brequired\b", re.IGNORECASE)),
    ("responsible_for", re.compile(r"\bresponsible\s+for\b", re.IGNORECASE)),
    ("agrees_to", re.compile(r"\bagrees?\s+to\b", re.IGNORECASE)),
    ("will", re.compile(r"\b(?:contractor|subcontractor|offeror|supplier|vendor|provider|agency|government|system|service)\s+will\b", re.IGNORECASE)),
    ("permitted", re.compile(r"\bpermitted\s+to\b", re.IGNORECASE)),
    ("may", re.compile(r"\bmay\b", re.IGNORECASE)),
)
_MANDATORY = {"shall", "shall_not", "must", "must_not", "required_to", "prohibited", "required", "responsible_for", "agrees_to", "will"}
_PROHIBITIVE = {"shall_not", "must_not", "prohibited"}
_PERMISSIVE = {"may", "permitted"}


@dataclass
class _ComparableElement:
    element: DocumentElement
    key: str
    text: str
    normalized: str


def _normalize_text(text: str) -> str:
    return _NORMALIZE_RE.sub(" ", str(text or "").strip().casefold())


def _sentence_summary(change_type: str, left: str, right: str) -> str:
    if change_type == "added":
        return f"Added: {right[:240]}"
    if change_type == "removed":
        return f"Removed: {left[:240]}"
    diff = list(difflib.ndiff(left.split(), right.split()))
    removed = [token[2:] for token in diff if token.startswith("- ")][:16]
    added = [token[2:] for token in diff if token.startswith("+ ")][:16]
    parts = []
    if removed:
        parts.append("removed " + " ".join(removed))
    if added:
        parts.append("added " + " ".join(added))
    if not parts:
        return "Modified wording with no concise token-level summary."
    return "Modified: " + "; ".join(parts)


def _stable_key(element: DocumentElement) -> str:
    if element.clause_number:
        return f"clause:{element.clause_number}"
    if element.sheet_name and element.cell_range:
        return f"sheet:{element.sheet_name}:{element.cell_range}"
    if element.sheet_name and element.row_start:
        return f"sheet:{element.sheet_name}:row:{element.row_start}"
    if element.slide_number:
        title = element.section_title or element.text[:80]
        return f"slide:{element.slide_number}:{element.element_type}:{_normalize_text(title)[:80]}"
    clause_match = _CLAUSE_RE.match(element.text)
    if clause_match:
        return f"clause:{clause_match.group('num')}"
    if element.section_path:
        return f"section:{' > '.join(element.section_path)}:{element.element_type}:{element.order}"
    return f"order:{element.element_type}:{element.order}"


def _comparable_elements(result: DocumentExtractResult, *, focus: str = "") -> List[_ComparableElement]:
    focus_terms = [term for term in re.findall(r"[A-Za-z0-9_]{3,}", str(focus or "").casefold()) if term]
    items: List[_ComparableElement] = []
    for element in result.elements:
        normalized = _normalize_text(element.text)
        if not normalized:
            continue
        if focus_terms and not any(term in normalized for term in focus_terms):
            continue
        items.append(_ComparableElement(element=element, key=_stable_key(element), text=element.text, normalized=normalized))
    return items


def _match_by_key(left: Sequence[_ComparableElement], right: Sequence[_ComparableElement]) -> tuple[list[tuple[_ComparableElement, _ComparableElement]], list[_ComparableElement], list[_ComparableElement]]:
    right_by_key: Dict[str, List[_ComparableElement]] = {}
    for item in right:
        right_by_key.setdefault(item.key, []).append(item)
    pairs: list[tuple[_ComparableElement, _ComparableElement]] = []
    unmatched_left: list[_ComparableElement] = []
    used_right_ids: set[str] = set()
    for left_item in left:
        candidates = [item for item in right_by_key.get(left_item.key, []) if item.element.element_id not in used_right_ids]
        if not candidates:
            unmatched_left.append(left_item)
            continue
        best = max(candidates, key=lambda item: difflib.SequenceMatcher(None, left_item.normalized, item.normalized).ratio())
        used_right_ids.add(best.element.element_id)
        pairs.append((left_item, best))
    unmatched_right = [item for item in right if item.element.element_id not in used_right_ids]
    return pairs, unmatched_left, unmatched_right


def _match_fuzzy(left: Sequence[_ComparableElement], right: Sequence[_ComparableElement]) -> tuple[list[tuple[_ComparableElement, _ComparableElement]], list[_ComparableElement], list[_ComparableElement]]:
    pairs: list[tuple[_ComparableElement, _ComparableElement]] = []
    unmatched_right = list(right)
    unmatched_left: list[_ComparableElement] = []
    for left_item in left:
        scored = [
            (difflib.SequenceMatcher(None, left_item.normalized, right_item.normalized).ratio(), right_item)
            for right_item in unmatched_right
        ]
        if not scored:
            unmatched_left.append(left_item)
            continue
        score, best = max(scored, key=lambda item: item[0])
        if score < 0.72:
            unmatched_left.append(left_item)
            continue
        unmatched_right = [item for item in unmatched_right if item.element.element_id != best.element.element_id]
        pairs.append((left_item, best))
    return pairs, unmatched_left, unmatched_right


def _obligation_modality(text: str) -> str:
    allowed = set(requirement_modalities_for_mode(BROAD_REQUIREMENT_MODE))
    for modality, pattern in _OBLIGATION_PATTERNS:
        if modality in allowed and pattern.search(text):
            return modality
    return ""


def _binding_strength(modality: str) -> int:
    if modality in _PROHIBITIVE:
        return 3
    if modality in _MANDATORY:
        return 2
    if modality in _PERMISSIVE:
        return 1
    return 0


def _changed_obligation(
    *,
    obligation_id: str,
    change_type: str,
    left: DocumentElement | None,
    right: DocumentElement | None,
) -> ChangedObligation | None:
    before = left.text if left is not None else ""
    after = right.text if right is not None else ""
    before_modality = _obligation_modality(before)
    after_modality = _obligation_modality(after)
    if not before_modality and not after_modality:
        return None
    modality = after_modality or before_modality
    severity = "medium"
    rationale = ""
    if change_type in {"added", "removed"} and modality in _MANDATORY:
        severity = "high"
        rationale = f"Mandatory obligation was {change_type}."
    elif change_type == "modified":
        before_strength = _binding_strength(before_modality)
        after_strength = _binding_strength(after_modality)
        if before_strength != after_strength:
            severity = "high"
            if after_strength > before_strength:
                change_type = "strengthened"
                rationale = "Binding language became stronger."
            else:
                change_type = "weakened"
                rationale = "Binding language became weaker."
        else:
            rationale = "Obligation-bearing language changed."
    location = element_location(right or left) if (right or left) else {}
    return ChangedObligation(
        obligation_id=obligation_id,
        change_type=change_type,
        modality=modality,
        severity=severity,
        before_text=before,
        after_text=after,
        location=location,
        rationale=rationale,
    )


class DocumentComparisonService:
    def __init__(self, settings: object, stores: object, session: object) -> None:
        self.extractor = DocumentExtractionService(settings, stores, session)

    def compare(
        self,
        *,
        left_document_ref: str,
        right_document_ref: str,
        source_scope: str = "auto",
        collection_id: str = "",
        compare_mode: str = "auto",
        focus: str = "",
        include_changed_obligations: bool = True,
        max_elements: int = 1500,
    ) -> DocumentCompareResult:
        left = self.extractor.extract(
            document_ref=left_document_ref,
            source_scope=source_scope,
            collection_id=collection_id,
            max_elements=max_elements,
        )
        right = self.extractor.extract(
            document_ref=right_document_ref,
            source_scope=source_scope,
            collection_id=collection_id,
            max_elements=max_elements,
        )
        mode = self._normalize_mode(compare_mode)
        left_items = _comparable_elements(left, focus=focus)
        right_items = _comparable_elements(right, focus=focus)
        key_pairs, unmatched_left, unmatched_right = _match_by_key(left_items, right_items)
        fuzzy_pairs, removed, added = _match_fuzzy(unmatched_left, unmatched_right)
        pairs = [*key_pairs, *fuzzy_pairs]

        deltas: List[DocumentDelta] = []
        obligations: List[ChangedObligation] = []
        next_delta = 1
        for left_item, right_item in pairs:
            similarity = difflib.SequenceMatcher(None, left_item.normalized, right_item.normalized).ratio()
            if similarity >= 0.985:
                change_type = "unchanged"
                summary = "Unchanged."
            else:
                change_type = "modified"
                summary = _sentence_summary(change_type, left_item.text, right_item.text)
            delta = DocumentDelta(
                delta_id=f"delta_{next_delta:05d}",
                change_type=change_type,
                summary=summary,
                left_element_id=left_item.element.element_id,
                right_element_id=right_item.element.element_id,
                left_text=left_item.text,
                right_text=right_item.text,
                location=element_location(right_item.element),
                similarity=round(similarity, 4),
            )
            deltas.append(delta)
            if include_changed_obligations and change_type != "unchanged":
                obligation = _changed_obligation(
                    obligation_id=f"obl_{len(obligations) + 1:05d}",
                    change_type=change_type,
                    left=left_item.element,
                    right=right_item.element,
                )
                if obligation is not None:
                    obligations.append(obligation)
            next_delta += 1
        for left_item in removed:
            delta = DocumentDelta(
                delta_id=f"delta_{next_delta:05d}",
                change_type="removed",
                summary=_sentence_summary("removed", left_item.text, ""),
                left_element_id=left_item.element.element_id,
                left_text=left_item.text,
                location=element_location(left_item.element),
            )
            deltas.append(delta)
            if include_changed_obligations:
                obligation = _changed_obligation(
                    obligation_id=f"obl_{len(obligations) + 1:05d}",
                    change_type="removed",
                    left=left_item.element,
                    right=None,
                )
                if obligation is not None:
                    obligations.append(obligation)
            next_delta += 1
        for right_item in added:
            delta = DocumentDelta(
                delta_id=f"delta_{next_delta:05d}",
                change_type="added",
                summary=_sentence_summary("added", "", right_item.text),
                right_element_id=right_item.element.element_id,
                right_text=right_item.text,
                location=element_location(right_item.element),
            )
            deltas.append(delta)
            if include_changed_obligations:
                obligation = _changed_obligation(
                    obligation_id=f"obl_{len(obligations) + 1:05d}",
                    change_type="added",
                    left=None,
                    right=right_item.element,
                )
                if obligation is not None:
                    obligations.append(obligation)
            next_delta += 1

        warnings = [*left.warnings, *right.warnings]
        if focus and not left_items and not right_items:
            warnings.append("No comparable elements matched the requested focus; comparison output is empty.")
        return DocumentCompareResult(
            left_document=left.document,
            right_document=right.document,
            compare_mode=mode,
            focus=str(focus or ""),
            deltas=self._shape_deltas_for_mode(deltas, mode),
            changed_obligations=obligations if include_changed_obligations else [],
            warnings=list(dict.fromkeys(item for item in warnings if item)),
        )

    def _normalize_mode(self, mode: str) -> str:
        normalized = str(mode or "auto").strip().lower()
        return normalized if normalized in {"auto", "redline", "clause", "version_delta", "obligations"} else "auto"

    def _shape_deltas_for_mode(self, deltas: Iterable[DocumentDelta], mode: str) -> List[DocumentDelta]:
        ordered = list(deltas)
        if mode == "obligations":
            return [item for item in ordered if _obligation_modality(item.left_text) or _obligation_modality(item.right_text)]
        if mode == "clause":
            clause_deltas = [
                item
                for item in ordered
                if item.location.get("clause_number")
                or re.match(r"^\s*(?:\d+\.)+\d*", item.left_text or item.right_text)
            ]
            return clause_deltas or ordered
        if mode == "version_delta":
            return [item for item in ordered if item.change_type != "unchanged"] or ordered
        return ordered


__all__ = ["DocumentComparisonService"]
