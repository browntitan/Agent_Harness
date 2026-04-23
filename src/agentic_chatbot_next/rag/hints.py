from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Sequence

from agentic_chatbot_next.rag.inventory import (
    INVENTORY_QUERY_NONE,
    INVENTORY_QUERY_SESSION_ACCESS,
    classify_inventory_query,
)

_STRUCTURED_SECTION_PREFIX = re.compile(
    r"^\s*(?:goal|context|deliverable|objective|task|request|constraints?|output)\s*:\s*",
    re.IGNORECASE,
)
_OPENWEBUI_CONTEXT_BLOCK_RE = re.compile(r"<context\b[^>]*>.*?</context>", re.IGNORECASE | re.DOTALL)
_OPENWEBUI_QUERY_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?:#{1,6}\s*)?(?:user\s+query|user\s+question|query|question)\s*:\s*",
    re.IGNORECASE,
)
_OPENWEBUI_OUTPUT_SECTION_RE = re.compile(
    r"(?ims)(?:^|\n)\s*#{1,6}\s*output\s*:?\s*(?P<body>.*)$"
)
_OPENWEBUI_WRAPPER_HINT_RE = re.compile(
    r"(?is)#{1,6}\s*task\s*:.*?(?:provided\s+context|user\s+query|chat\s+history)"
)
_OPENWEBUI_WRAPPER_HEADING_RE = re.compile(
    r"(?im)^\s*#{1,6}\s*(?:task|guidelines?|output|chat\s+history|context)\s*:?\s*$"
)
_OPENWEBUI_CHAT_HISTORY_USER_RE = re.compile(r"(?im)^\s*USER\s*:\s*(.+?)\s*$")
_OPENWEBUI_INSTRUCTION_LINE_RE = re.compile(
    r"(?i)^\s*(?:[-*]\s*)?(?:"
    r"provide\s+(?:a\s+)?(?:clear|concise|direct)|"
    r"respond\s+to\s+the\s+user|"
    r"use\s+the\s+provided\s+context|"
    r"if\s+(?:you\s+)?(?:don'?t|do\s+not)\s+know|"
    r"if\s+the\s+answer\s+(?:isn'?t|is\s+not)\s+present|"
    r"do\s+not\s+(?:make|invent|fabricate)|"
    r"strictly\s+return|"
    r"return\s+only|"
    r"only\s+(?:include|use|return)|"
    r"cite\s+|"
    r"include\s+citations?|"
    r"answer\s+the\s+question"
    r")\b"
)
_DISCOVERY_HINTS = re.compile(
    r"\b("
    r"identify\s+all\s+documents|which\s+documents|list\s+(?:all\s+)?(?:documents|files)|"
    r"across\s+(?:the\s+)?(?:corpus|documents|policies|sops)|every\s+document|"
    r"find\s+all|inventory|exhaustive|all\s+sops?|"
    r"(?:provide|give|return)\s+(?:me\s+)?(?:only\s+)?(?:a\s+)?list\s+of\s+(?:potential\s+)?(?:documents|files)|"
    r"potential\s+(?:documents|files)\s+(?:about|for)|"
    r"(?:documents|files)\s+that\s+(?:have|contain)\s+information\s+about|"
    r"(?:documents|files)\s+that\s+(?:discuss|describe|cover|contain)|"
    r"search\s+across\s+(?:the\s+)?documents"
    r")\b",
    re.IGNORECASE,
)
_PROCESS_FLOW_HINTS = re.compile(
    r"\b(process\s+flows?|workflows?|flowcharts?|approval\s+flows?|handoff|escalation)\b",
    re.IGNORECASE,
)
_COMPARISON_HINTS = re.compile(
    r"\b(compare|difference|differences|versus|vs\.?|contrast)\b",
    re.IGNORECASE,
)
_BOUNDED_SYNTHESIS_HINTS = re.compile(
    r"\b("
    r"what\s+are\s+the\s+key\s+implementation\s+details|key\s+implementation\s+details|"
    r"summari[sz]e|explain|walk\s+me\s+through|walk\s+through|overview|how\s+does|how\s+do"
    r")\b",
    re.IGNORECASE,
)
_EXPANSIVE_OUTPUT_HINTS = re.compile(
    r"\b(detailed|detail|verbose|comprehensive|thorough|in-depth|in depth|longer|deep dive)\b",
    re.IGNORECASE,
)


@dataclass
class RagExecutionHints:
    research_profile: str = ""
    coverage_goal: str = ""
    result_mode: str = ""
    controller_hints: Dict[str, Any] = field(default_factory=dict)
    matched_skill_ids: list[str] = field(default_factory=list)
    matched_skill_names: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "research_profile": self.research_profile,
            "coverage_goal": self.coverage_goal,
            "result_mode": self.result_mode,
            "controller_hints": dict(self.controller_hints),
            "matched_skill_ids": list(self.matched_skill_ids),
            "matched_skill_names": list(self.matched_skill_names),
        }


def normalize_structured_query(query: str) -> str:
    text = str(query or "").strip()
    if not text:
        return ""

    wrapped = _extract_openwebui_wrapped_user_query(text)
    if wrapped:
        text = wrapped

    marker_matches = list(_OPENWEBUI_QUERY_MARKER_RE.finditer(text))
    if marker_matches:
        candidate = text[marker_matches[-1].end() :].strip()
        if candidate:
            text = candidate
    elif "<context" in text.casefold() and "</context>" in text.casefold():
        parts = re.split(r"</context>", text, flags=re.IGNORECASE)
        candidate = parts[-1].strip() if parts else ""
        if candidate:
            text = candidate
        else:
            text = _OPENWEBUI_CONTEXT_BLOCK_RE.sub("", text).strip()
            text = re.sub(r"(?is)^###\s*task\s*:.*?(?:\n\s*\n|$)", "", text).strip()

    lines: list[str] = []
    for raw_line in text.splitlines():
        line = str(raw_line or "").strip()
        if not line:
            continue
        line = _STRUCTURED_SECTION_PREFIX.sub("", line).strip()
        if line:
            lines.append(line)

    if not lines:
        return text
    return "\n".join(lines)


def _extract_openwebui_wrapped_user_query(text: str) -> str:
    raw = str(text or "").strip()
    if not raw or not _OPENWEBUI_WRAPPER_HINT_RE.search(raw):
        return ""

    output_matches = list(_OPENWEBUI_OUTPUT_SECTION_RE.finditer(raw))
    if output_matches:
        tail = output_matches[-1].group("body").strip()
        tail = _OPENWEBUI_CONTEXT_BLOCK_RE.sub("", tail).strip()
        candidate = _last_non_instruction_block(tail)
        if candidate:
            return candidate

    chat_users = [str(match.group(1) or "").strip() for match in _OPENWEBUI_CHAT_HISTORY_USER_RE.finditer(raw)]
    chat_users = [item for item in chat_users if item]
    return chat_users[-1] if chat_users else ""


def _last_non_instruction_block(text: str) -> str:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", str(text or "")) if item.strip()]
    for paragraph in reversed(paragraphs):
        lines: list[str] = []
        for raw_line in paragraph.splitlines():
            line = str(raw_line or "").strip()
            if not line or _OPENWEBUI_WRAPPER_HEADING_RE.match(line):
                continue
            line = _STRUCTURED_SECTION_PREFIX.sub("", line).strip()
            if not line or _OPENWEBUI_INSTRUCTION_LINE_RE.match(line):
                continue
            lines.append(line)
        candidate = "\n".join(lines).strip()
        if candidate:
            return candidate
    return ""


def normalize_research_profile(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "corpus": "corpus_discovery",
        "corpus_inventory": "corpus_discovery",
        "document_inventory": "corpus_discovery",
        "inventory": "corpus_discovery",
        "process_flow": "process_flow_identification",
        "workflow_identification": "process_flow_identification",
        "comparison": "comparison_campaign",
    }
    return mapping.get(normalized, normalized)


def normalize_coverage_goal(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "corpus": "corpus_wide",
        "corpuswide": "corpus_wide",
        "broad": "corpus_wide",
        "exhaustive_search": "exhaustive",
        "target": "targeted",
        "comparison": "cross_document",
    }
    return mapping.get(normalized, normalized)


def normalize_result_mode(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "per_document_inventory": "inventory",
        "per_doc_inventory": "inventory",
        "file_list": "inventory",
        "list": "inventory",
        "default": "answer",
        "grounded_answer": "answer",
        "comparison_notes": "comparison",
    }
    return mapping.get(normalized, normalized)


def coerce_controller_hints(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items() if str(key).strip()}
    if raw is None:
        return {}
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return {str(key): value for key, value in parsed.items() if str(key).strip()}
        hints: Dict[str, Any] = {}
        for part in text.split(","):
            item = part.strip()
            if not item:
                continue
            if "=" in item:
                key, value = item.split("=", 1)
                hints[key.strip()] = value.strip()
            else:
                hints[item] = True
        return hints
    return {}


def merge_controller_hints(*maps: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for mapping in maps:
        for key, value in dict(mapping or {}).items():
            if not str(key).strip():
                continue
            merged[str(key)] = value
    return merged


def merge_rag_execution_hints(*items: RagExecutionHints) -> RagExecutionHints:
    merged = RagExecutionHints()
    for item in items:
        if not merged.research_profile and item.research_profile:
            merged.research_profile = item.research_profile
        if not merged.coverage_goal and item.coverage_goal:
            merged.coverage_goal = item.coverage_goal
        if not merged.result_mode and item.result_mode:
            merged.result_mode = item.result_mode
        merged.controller_hints = merge_controller_hints(merged.controller_hints, item.controller_hints)
        merged.matched_skill_ids.extend(skill_id for skill_id in item.matched_skill_ids if skill_id not in merged.matched_skill_ids)
        merged.matched_skill_names.extend(name for name in item.matched_skill_names if name not in merged.matched_skill_names)
    return merged


def is_metadata_inventory_query(query: str) -> bool:
    normalized_query = normalize_structured_query(query) or str(query or "")
    return classify_inventory_query(normalized_query) != INVENTORY_QUERY_NONE


def infer_rag_execution_hints(query: str, *, skill_queries: Sequence[str] | None = None) -> RagExecutionHints:
    normalized_query = normalize_structured_query(query) or str(query or "")
    text = "\n".join([normalized_query, *[str(item) for item in (skill_queries or []) if str(item)]])
    hints = RagExecutionHints()
    inventory_query_type = classify_inventory_query(normalized_query)
    if inventory_query_type != INVENTORY_QUERY_NONE:
        inventory_hints: Dict[str, Any] = {
            "force_deep_search": True,
            "prefer_parallel_docs": True,
            "prefer_inventory_output": True,
            "inventory_query_type": inventory_query_type,
        }
        if inventory_query_type == INVENTORY_QUERY_SESSION_ACCESS:
            inventory_hints["prefer_session_access_inventory"] = True
        hints.research_profile = "corpus_discovery"
        hints.coverage_goal = "corpus_wide"
        hints.result_mode = "inventory"
        hints.controller_hints = merge_controller_hints(
            hints.controller_hints,
            inventory_hints,
        )
    if _DISCOVERY_HINTS.search(text):
        hints.research_profile = "corpus_discovery"
        hints.coverage_goal = "corpus_wide"
        hints.result_mode = "inventory"
        hints.controller_hints = merge_controller_hints(
            hints.controller_hints,
            {
                "force_deep_search": True,
                "prefer_parallel_docs": True,
                "prefer_inventory_output": True,
            },
        )
    if _PROCESS_FLOW_HINTS.search(text):
        if not hints.research_profile:
            hints.research_profile = "process_flow_identification"
        if not hints.coverage_goal:
            hints.coverage_goal = "corpus_wide" if _DISCOVERY_HINTS.search(text) else "targeted"
        if not hints.result_mode and _DISCOVERY_HINTS.search(text):
            hints.result_mode = "inventory"
        hints.controller_hints = merge_controller_hints(
            hints.controller_hints,
            {
                "prefer_process_flow_docs": True,
                "prefer_windowed_keyword_followup": True,
            },
        )
    if _COMPARISON_HINTS.search(text):
        hints.research_profile = "comparison_campaign"
        if not hints.coverage_goal:
            hints.coverage_goal = "cross_document"
        if not hints.result_mode:
            hints.result_mode = "comparison"
        hints.controller_hints = merge_controller_hints(
            hints.controller_hints,
            {
                "force_deep_search": True,
                "prefer_parallel_docs": True,
                "compare_across_documents": True,
            },
        )
    lowered = text.lower()
    if "coverage sufficiency" in lowered or "sufficiency" in lowered:
        hints.controller_hints = merge_controller_hints(
            hints.controller_hints,
            {"enforce_sufficiency_check": True},
        )
    if "negative evidence" in lowered or "not found" in lowered or "insufficient evidence" in lowered:
        hints.controller_hints = merge_controller_hints(
            hints.controller_hints,
            {"prefer_negative_evidence_reporting": True},
        )
    if "windowed keyword" in lowered or "keyword followup" in lowered:
        hints.controller_hints = merge_controller_hints(
            hints.controller_hints,
            {"prefer_windowed_keyword_followup": True},
        )
    if "inventory" in lowered and not hints.result_mode:
        hints.result_mode = "inventory"
    if not hints.coverage_goal:
        hints.coverage_goal = "targeted"
    if not hints.result_mode:
        hints.result_mode = "answer"
    return hints


def prefers_bounded_synthesis(query: str, *, skill_queries: Sequence[str] | None = None) -> bool:
    normalized_query = normalize_structured_query(query) or str(query or "")
    text = "\n".join([normalized_query, *[str(item) for item in (skill_queries or []) if str(item)]])
    if not _BOUNDED_SYNTHESIS_HINTS.search(text):
        return False
    if _EXPANSIVE_OUTPUT_HINTS.search(text):
        return False
    if _DISCOVERY_HINTS.search(text) or _PROCESS_FLOW_HINTS.search(text) or _COMPARISON_HINTS.search(text):
        return False
    return True


def apply_bounded_synthesis_override(
    hints: RagExecutionHints,
    *,
    query: str,
    skill_queries: Sequence[str] | None = None,
) -> RagExecutionHints:
    if not prefers_bounded_synthesis(query, skill_queries=skill_queries):
        return hints

    suppressed_keys = {
        "force_deep_search",
        "prefer_inventory_output",
        "prefer_parallel_docs",
        "compare_across_documents",
        "prefer_process_flow_docs",
        "prefer_windowed_keyword_followup",
    }
    filtered_controller_hints = {
        str(key): value
        for key, value in dict(hints.controller_hints or {}).items()
        if str(key) not in suppressed_keys
    }
    return RagExecutionHints(
        research_profile="",
        coverage_goal="targeted",
        result_mode="answer",
        controller_hints=filtered_controller_hints,
        matched_skill_ids=list(hints.matched_skill_ids),
        matched_skill_names=list(hints.matched_skill_names),
    )


def coerce_rag_execution_hints(
    *,
    research_profile: str = "",
    coverage_goal: str = "",
    result_mode: str = "",
    controller_hints: Dict[str, Any] | None = None,
    matched_skill_ids: Iterable[str] | None = None,
    matched_skill_names: Iterable[str] | None = None,
) -> RagExecutionHints:
    return RagExecutionHints(
        research_profile=normalize_research_profile(research_profile),
        coverage_goal=normalize_coverage_goal(coverage_goal),
        result_mode=normalize_result_mode(result_mode),
        controller_hints=coerce_controller_hints(controller_hints),
        matched_skill_ids=[str(item) for item in (matched_skill_ids or []) if str(item)],
        matched_skill_names=[str(item) for item in (matched_skill_names or []) if str(item)],
    )


__all__ = [
    "apply_bounded_synthesis_override",
    "RagExecutionHints",
    "coerce_controller_hints",
    "coerce_rag_execution_hints",
    "infer_rag_execution_hints",
    "is_metadata_inventory_query",
    "merge_controller_hints",
    "merge_rag_execution_hints",
    "normalize_coverage_goal",
    "normalize_research_profile",
    "normalize_result_mode",
    "normalize_structured_query",
    "prefers_bounded_synthesis",
]
