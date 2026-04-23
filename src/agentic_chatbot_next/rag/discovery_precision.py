from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Sequence

_GENERIC_DISCOVERY_TERMS = {
    "about",
    "across",
    "all",
    "and",
    "answer",
    "approval",
    "approvals",
    "are",
    "can",
    "available",
    "base",
    "cite",
    "collection",
    "collections",
    "contain",
    "contains",
    "descriptions",
    "describe",
    "describes",
    "describing",
    "document",
    "documents",
    "docs",
    "evidence",
    "explicit",
    "explicitly",
    "file",
    "files",
    "find",
    "flow",
    "flows",
    "for",
    "grounded",
    "has",
    "have",
    "handoff",
    "handoffs",
    "identify",
    "include",
    "includes",
    "including",
    "in",
    "indexed",
    "inventory",
    "kb",
    "knowledge",
    "list",
    "match",
    "matches",
    "me",
    "mention",
    "mentions",
    "outlined",
    "outline",
    "out",
    "please",
    "process",
    "procedures",
    "procedure",
    "request",
    "relevant",
    "results",
    "return",
    "search",
    "show",
    "steps",
    "that",
    "the",
    "their",
    "these",
    "this",
    "tell",
    "them",
    "those",
    "use",
    "we",
    "what",
    "which",
    "with",
    "workflow",
    "workflows",
    "you",
}

_TOPIC_EQUIVALENTS = {
    "onboarding": ("onboarding", "new hire", "newhire"),
}

_DISCOVERY_TOPIC_PATTERNS = (
    re.compile(r"\bcontain\s+(?P<label>.+?)(?:[?.!]|$)", re.IGNORECASE),
    re.compile(r"\bmention\s+(?P<label>.+?)(?:[?.!]|$)", re.IGNORECASE),
    re.compile(r"\babout\s+(?P<label>.+?)(?:[?.!]|$)", re.IGNORECASE),
    re.compile(r"\bfor\s+(?P<label>.+?)(?:[?.!]|$)", re.IGNORECASE),
)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.casefold().split())


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = _normalize_text(phrase)
    if not normalized_phrase:
        return False
    pattern = r"\b" + re.escape(normalized_phrase).replace(r"\ ", r"\s+") + r"\b"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def extract_discovery_topic_anchors(query: str) -> List[List[str]]:
    normalized = _normalize_text(query)
    anchors: List[List[str]] = []
    seen_tokens: set[str] = set()

    for canonical, aliases in _TOPIC_EQUIVALENTS.items():
        phrases = [canonical, *aliases]
        if any(_contains_phrase(normalized, phrase) for phrase in phrases):
            anchors.append(list(dict.fromkeys(phrases)))
            for phrase in phrases:
                seen_tokens.update(re.findall(r"[a-z0-9_]{3,}", _normalize_text(phrase)))

    for term in re.findall(r"[a-z0-9_]{3,}", normalized):
        if term in _GENERIC_DISCOVERY_TERMS or term in seen_tokens:
            continue
        anchors.append([term])
        seen_tokens.add(term)

    return anchors


def workflow_topic_seed_terms(query: str) -> List[str]:
    phrases: List[str] = []
    seen: set[str] = set()
    for group in extract_discovery_topic_anchors(query):
        canonical = str(group[0] or "").strip()
        if not canonical:
            continue
        candidates: Sequence[str] = group
        if canonical == "onboarding":
            candidates = ("onboarding", "new hire")
        for candidate in candidates:
            clean = str(candidate or "").strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            phrases.append(clean)
        if len(phrases) >= 4:
            break
    return phrases[:4]


def match_discovery_topic_anchors(query: str, text: str) -> Dict[str, Any]:
    anchor_groups = extract_discovery_topic_anchors(query)
    if not anchor_groups:
        return {
            "matches": True,
            "required_anchors": [],
            "matched_anchors": [],
        }

    haystack = _normalize_text(text)
    matched: List[str] = []
    for group in anchor_groups:
        if any(_contains_phrase(haystack, phrase) for phrase in group):
            matched.append(str(group[0]))
    return {
        "matches": bool(matched),
        "required_anchors": [str(group[0]) for group in anchor_groups if group],
        "matched_anchors": matched,
    }


def document_has_explicit_topic_support(query: str, doc: Any) -> Dict[str, Any]:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    text = "\n".join(
        part
        for part in (
            str(metadata.get("title") or ""),
            str(metadata.get("section_title") or ""),
            str(getattr(doc, "page_content", "") or ""),
        )
        if part
    )
    details = match_discovery_topic_anchors(query, text)
    return {
        **details,
        "doc_id": str(metadata.get("doc_id") or ""),
        "title": str(metadata.get("title") or ""),
    }


def discovery_topic_label(query: str) -> str:
    text = str(query or "").strip()
    for pattern in _DISCOVERY_TOPIC_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        label = str(match.group("label") or "").strip(" .?!")
        label = re.sub(r"\bin\s+the\s+knowledge\s+base\b", "", label, flags=re.IGNORECASE).strip(" ,")
        if label:
            return label
    anchors = [group[0] for group in extract_discovery_topic_anchors(query) if group]
    if anchors:
        return " ".join(anchors[:3]).strip()
    return "the requested topic"


__all__ = [
    "discovery_topic_label",
    "document_has_explicit_topic_support",
    "extract_discovery_topic_anchors",
    "match_discovery_topic_anchors",
    "workflow_topic_seed_terms",
]
