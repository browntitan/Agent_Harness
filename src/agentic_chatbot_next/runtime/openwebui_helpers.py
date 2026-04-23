from __future__ import annotations

import re
from typing import Any

OPENWEBUI_HELPER_TASK_TYPES = {"follow_ups", "title", "tags", "search_queries"}

_FOLLOW_UP_PATTERNS = (
    re.compile(r"suggest\s+3-5\s+relevant\s+follow-?up\s+questions", re.IGNORECASE),
    re.compile(r"suggest\s+follow-?up\s+questions", re.IGNORECASE),
    re.compile(r"follow-?up\s+questions?\s+(?:the\s+user\s+)?could\s+ask", re.IGNORECASE),
)
_TITLE_PATTERNS = (
    re.compile(r"generate\s+a\s+concise.*title", re.IGNORECASE),
    re.compile(r"\b3-5\s+word\s+title\b", re.IGNORECASE),
)
_TAG_PATTERNS = (
    re.compile(r"generate\s+1-3\s+broad\s+tags", re.IGNORECASE),
    re.compile(r"\bbroad\s+tags\b", re.IGNORECASE),
)
_SEARCH_QUERY_PATTERNS = (
    re.compile(r"analyze\s+the\s+chat\s+history\s+to\s+determine\s+the\s+necessity\s+of\s+generating\s+search\s+queries", re.IGNORECASE),
    re.compile(r"prioritize\s+generating\s+1-3\s+broad\s+and\s+relevant\s+search\s+queries", re.IGNORECASE),
    re.compile(r"respond\s+\*+exclusively\*+\s+with\s+a\s+json\s+object", re.IGNORECASE),
    re.compile(r"\"queries\"\s*:\s*\[", re.IGNORECASE),
)


def normalize_openwebui_helper_task_type(value: Any) -> str:
    task_type = str(value or "").strip().lower()
    return task_type if task_type in OPENWEBUI_HELPER_TASK_TYPES else ""


def infer_openwebui_helper_task_type(text: Any) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return ""
    if any(pattern.search(normalized) for pattern in _FOLLOW_UP_PATTERNS):
        return "follow_ups"
    if any(pattern.search(normalized) for pattern in _TITLE_PATTERNS):
        return "title"
    if any(pattern.search(normalized) for pattern in _TAG_PATTERNS):
        return "tags"
    if any(pattern.search(normalized) for pattern in _SEARCH_QUERY_PATTERNS):
        return "search_queries"
    return ""


def is_openwebui_helper_message(message: Any) -> bool:
    metadata = {}
    if hasattr(message, "metadata"):
        metadata = dict(getattr(message, "metadata", {}) or {})
    else:
        metadata = dict(getattr(message, "additional_kwargs", {}) or {})
    return bool(normalize_openwebui_helper_task_type(metadata.get("openwebui_helper_task_type")))


def openwebui_helper_system_prompt(task_type: str) -> str:
    normalized = normalize_openwebui_helper_task_type(task_type)
    base = (
        "You are handling an internal Open WebUI helper task.\n"
        "Treat the input as UI metadata generation, not as a user question.\n"
        "Do not search the knowledge base.\n"
        "Do not cite sources.\n"
        "Do not mention retrieval, routing, or the underlying system.\n"
        "Return only the requested payload with no markdown fences, preamble, or explanation."
    )
    if normalized == "follow_ups":
        return (
            f"{base}\n"
            "Return only a JSON array of 3 to 5 short follow-up questions as strings."
        )
    if normalized == "title":
        return (
            f"{base}\n"
            "Return only the title text as plain text, ideally 3 to 5 words, with no quotes."
        )
    if normalized == "tags":
        return (
            f"{base}\n"
            "Return only a JSON array of 1 to 3 broad lowercase tags as strings."
        )
    if normalized == "search_queries":
        return (
            f"{base}\n"
            "Return only a JSON object in the form {\"queries\": [\"query1\", \"query2\"]}."
        )
    return base


__all__ = [
    "OPENWEBUI_HELPER_TASK_TYPES",
    "infer_openwebui_helper_task_type",
    "is_openwebui_helper_message",
    "normalize_openwebui_helper_task_type",
    "openwebui_helper_system_prompt",
]
