from __future__ import annotations

import re

_ROUTING_PREFIX_HINTS = re.compile(
    r"\b("
    r"search|look\s+up|consult|use|query|find|retrieve|check|"
    r"knowledge\s*base|kb|knowledge\s+graph|graph|inventory|indexed|"
    r"answer|respond|briefly|citations?|sources?|fact|policy|requirement"
    r")\b",
    re.IGNORECASE,
)
_ROUTING_ACTION_HINTS = re.compile(
    r"\b("
    r"search|look\s+up|consult|use|query|find|retrieve|check|"
    r"knowledge\s*base|kb|knowledge\s+graph|graph|inventory|indexed|"
    r"answer|respond|briefly|citations?|sources?"
    r")\b",
    re.IGNORECASE,
)
_QUESTION_START = re.compile(
    r"^\s*(?:what|which|who|when|where|why|how|does|do|is|are|can|should|list|show|name|identify)\b",
    re.IGNORECASE,
)
_LEADING_ROUTING_PHRASES = (
    re.compile(
        r"^\s*(?:please\s+)?(?:search|look\s+up|consult|query|check)\s+"
        r"(?:the\s+)?(?:default\s+|[\w.-]+\s+)?(?:knowledge\s*base|kb|knowledge\s+graph|graph)\s+"
        r"(?:for|about|on)\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:please\s+)?(?:answer|respond)\s+(?:briefly\s+)?(?:with\s+citations?\s*)?(?:to\s+)?",
        re.IGNORECASE,
    ),
)


def normalize_retrieval_question(query: str) -> str:
    """Return the content-bearing retrieval query without UI/tool-routing wrapper text."""
    text = " ".join(str(query or "").strip().split())
    if not text:
        return ""

    if ":" in text:
        prefix, suffix = text.split(":", 1)
        suffix = suffix.strip()
        if (
            suffix
            and len(prefix) <= 220
            and _ROUTING_PREFIX_HINTS.search(prefix)
            and _ROUTING_ACTION_HINTS.search(prefix)
            and (_QUESTION_START.search(suffix) or len(suffix.split()) >= 4)
        ):
            return suffix

    cleaned = text
    changed = True
    while changed:
        changed = False
        for pattern in _LEADING_ROUTING_PHRASES:
            next_value = pattern.sub("", cleaned).strip()
            if next_value != cleaned and next_value:
                cleaned = next_value
                changed = True
    return cleaned or text


__all__ = ["normalize_retrieval_question"]
