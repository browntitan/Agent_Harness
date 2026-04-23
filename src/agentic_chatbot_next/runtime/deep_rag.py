from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict

from pydantic import BaseModel, Field, field_validator

from agentic_chatbot_next.utils.json_utils import extract_json

_ALLOWED_MODES = {"off", "auto", "force"}
_ALLOWED_SEARCH_MODES = {"fast", "auto", "deep"}
_ALLOWED_AGENTS = {"", "rag_worker", "coordinator"}

_BROAD_RESEARCH_HINTS = re.compile(
    r"\b("
    r"investigate|across\s+the\s+default\s+collection|across\s+the\s+documents|"
    r"identify\s+all\s+documents|provide\s+(?:me\s+)?a\s+list\s+of\s+documents|"
    r"be\s+exhaustive|corpus|comprehensive|deep\s+research|major\s+subsystems|"
    r"full\s+lifecycle|end-to-end|cross-cutting|compare\s+many|long-form|long form"
    r")\b",
    re.IGNORECASE,
)
_DETAILED_HINTS = re.compile(
    r"\b(detailed|comprehensive|thorough|verbose|deep\s+dive|deep dive|exhaustive)\b",
    re.IGNORECASE,
)
_DOC_READ_HINTS = re.compile(
    r"\b("
    r"read\s+the\s+document|look\s+through|inspect\s+the\s+docs|"
    r"entire\s+document|full\s+document|full\s+text|read\s+directly"
    r")\b",
    re.IGNORECASE,
)
_DIRECT_DOC_HINTS = re.compile(
    r"\b[a-z0-9._/-]+\.(?:md|pdf|docx|txt|csv|xlsx|xls)\b",
    re.IGNORECASE,
)
_FOLLOWUP_SUMMARY_HINTS = re.compile(
    r"\b("
    r"those\s+docs|those\s+documents|candidate\s+documents|documents\s+you\s+provided|"
    r"docs\s+above|summari[sz]e\s+those|summari[sz]e\s+the\s+docs|"
    r"explain\s+the\s+docs|look\s+through\s+them"
    r")\b",
    re.IGNORECASE,
)
_BACKGROUND_HINTS = re.compile(
    r"\b(report|handbook|research\s+report|deep\s+research|write\s+up|write-up|comprehensive)\b",
    re.IGNORECASE,
)


def _is_authoritative_inventory_query(user_text: str) -> bool:
    try:
        from agentic_chatbot_next.rag.inventory import (
            classify_inventory_query,
            inventory_query_requests_grounded_analysis,
            is_authoritative_inventory_query_type,
        )
    except Exception:
        return False
    query_type = classify_inventory_query(user_text)
    return is_authoritative_inventory_query_type(query_type) and not inventory_query_requests_grounded_analysis(
        user_text,
        query_type=query_type,
    )

_POLICY_SYSTEM_PROMPT = """\
You are a retrieval policy planner for an enterprise document assistant.

Choose whether the current AGENT turn should stay in fast retrieval, escalate to deep staged retrieval,
or start in the coordinator.

Important rules:
- Hard blockers like missing scope or missing uploads are handled elsewhere; do not reason about them here.
- Prefer `search_mode="deep"` when the query is complex, broad, evidence-sensitive, or asks for detailed synthesis.
- Prefer `preferred_agent="coordinator"` for broad research campaigns, compare-many tasks, or exhaustive corpus work.
- Prefer `preferred_agent="rag_worker"` for focused grounded questions that still need deep staged retrieval.
- Prefer `prefer_full_reads=true` when the user asks to inspect, look through, or directly read documents.
- Prefer `prefer_section_first=true` for long or structured documents before paginating through the full document.
- Recommend background work only when the request is clearly large enough to benefit from long-form execution.

Return JSON only.
"""

_POLICY_HUMAN_TEMPLATE = """\
Current route: {route}
Router suggested agent: {suggested_agent}
Has attachments: {has_attachments}
Requested deep_rag mode: {requested_mode}
Background allowed: {background_ok}
Max reflection rounds override: {max_reflection_rounds}

Research packet:
{research_packet}

Current user message:
{user_text}
"""


def normalize_deep_rag_mode(value: Any, *, default: str = "auto") -> str:
    clean = str(value or "").strip().lower()
    if clean in _ALLOWED_MODES:
        return clean
    return default


def normalize_deep_rag_search_mode(value: Any, *, default: str = "auto") -> str:
    clean = str(value or "").strip().lower()
    if clean in _ALLOWED_SEARCH_MODES:
        return clean
    return default


def _as_optional_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    clean = str(value).strip().lower()
    if clean in {"1", "true", "yes", "on"}:
        return True
    if clean in {"0", "false", "no", "off"}:
        return False
    return default


class _DeepRagPolicyOutput(BaseModel):
    search_mode: str = Field(default="auto")
    preferred_agent: str = Field(default="")
    prefer_full_reads: bool = Field(default=False)
    prefer_section_first: bool = Field(default=True)
    background_recommended: bool = Field(default=False)
    max_reflection_rounds: int | None = Field(default=None)
    reasoning: str = Field(default="")

    @field_validator("search_mode")
    @classmethod
    def _validate_search_mode(cls, value: str) -> str:
        return normalize_deep_rag_search_mode(value)

    @field_validator("preferred_agent")
    @classmethod
    def _validate_preferred_agent(cls, value: str) -> str:
        clean = str(value or "").strip().lower()
        return clean if clean in _ALLOWED_AGENTS else ""


@dataclass(frozen=True)
class DeepRagRequest:
    mode: str = "auto"
    background_ok: bool = False
    max_reflection_rounds: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "mode": self.mode,
            "background_ok": self.background_ok,
        }
        if self.max_reflection_rounds is not None:
            payload["max_reflection_rounds"] = self.max_reflection_rounds
        return payload


@dataclass(frozen=True)
class DeepRagPolicyDecision:
    mode: str = "auto"
    search_mode: str = "auto"
    preferred_agent: str = ""
    prefer_full_reads: bool = False
    prefer_section_first: bool = True
    max_reflection_rounds: int = 1
    max_parallel_lanes: int = 3
    full_read_chunk_threshold: int = 24
    background_recommended: bool = False
    reasoning: str = ""
    complexity_score: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "search_mode": self.search_mode,
            "preferred_agent": self.preferred_agent,
            "prefer_full_reads": self.prefer_full_reads,
            "prefer_section_first": self.prefer_section_first,
            "max_reflection_rounds": self.max_reflection_rounds,
            "max_parallel_lanes": self.max_parallel_lanes,
            "full_read_chunk_threshold": self.full_read_chunk_threshold,
            "background_recommended": self.background_recommended,
            "reasoning": self.reasoning,
            "complexity_score": self.complexity_score,
        }


def parse_deep_rag_request(settings: Any, raw: Any) -> DeepRagRequest:
    payload = dict(raw or {}) if isinstance(raw, dict) else {}
    return DeepRagRequest(
        mode=normalize_deep_rag_mode(
            payload.get("mode"),
            default=str(getattr(settings, "deep_rag_default_mode", "auto") or "auto"),
        ),
        background_ok=_coerce_bool(payload.get("background_ok"), default=False),
        max_reflection_rounds=_as_optional_positive_int(payload.get("max_reflection_rounds")),
    )


def deep_rag_controller_hints(route_context: Dict[str, Any] | None) -> Dict[str, Any]:
    deep_rag = dict((route_context or {}).get("deep_rag") or {})
    if not deep_rag:
        return {}
    hints: Dict[str, Any] = {}
    if normalize_deep_rag_search_mode(deep_rag.get("search_mode")) == "deep":
        hints["force_deep_search"] = True
    if _coerce_bool(deep_rag.get("prefer_full_reads")):
        hints["prefer_full_reads"] = True
        hints.setdefault("doc_read_depth", "full")
    if _coerce_bool(deep_rag.get("prefer_section_first"), default=True):
        hints["prefer_section_first"] = True
    max_parallel_lanes = _as_optional_positive_int(deep_rag.get("max_parallel_lanes"))
    if max_parallel_lanes is not None:
        hints["max_parallel_lanes"] = max_parallel_lanes
    max_reflection_rounds = _as_optional_positive_int(deep_rag.get("max_reflection_rounds"))
    if max_reflection_rounds is not None:
        hints["max_reflection_rounds"] = max_reflection_rounds
    full_read_chunk_threshold = _as_optional_positive_int(deep_rag.get("full_read_chunk_threshold"))
    if full_read_chunk_threshold is not None:
        hints["full_read_chunk_threshold"] = full_read_chunk_threshold
    if _coerce_bool(deep_rag.get("background_recommended")):
        hints["background_recommended"] = True
    return hints


def deep_rag_search_mode(route_context: Dict[str, Any] | None, *, default: str = "auto") -> str:
    normalized_default = normalize_deep_rag_search_mode(default, default="auto")
    if normalized_default in {"fast", "deep"}:
        return normalized_default
    deep_rag = dict((route_context or {}).get("deep_rag") or {})
    return normalize_deep_rag_search_mode(deep_rag.get("search_mode"), default=normalized_default)


def _complexity_score(
    *,
    user_text: str,
    route: str,
    suggested_agent: str,
    has_attachments: bool,
    session_metadata: Dict[str, Any],
) -> int:
    text = str(user_text or "")
    lowered = text.lower()
    score = 0
    if route == "AGENT":
        score += 1
    if len(text.split()) >= 24:
        score += 1
    if len(text) >= 500:
        score += 1
    if _BROAD_RESEARCH_HINTS.search(text):
        score += 2
    if _DETAILED_HINTS.search(text):
        score += 1
    if "cite" in lowered or "citations" in lowered:
        score += 1
    if _DIRECT_DOC_HINTS.search(text):
        score += 1
    if _DOC_READ_HINTS.search(text):
        score += 1
    if has_attachments:
        score += 1
    if suggested_agent == "coordinator":
        score += 2
    if _FOLLOWUP_SUMMARY_HINTS.search(text) and bool(dict(session_metadata or {}).get("active_doc_focus")):
        score += 2
    return score


def _heuristic_policy(
    settings: Any,
    *,
    request: DeepRagRequest,
    user_text: str,
    route: str,
    suggested_agent: str,
    has_attachments: bool,
    session_metadata: Dict[str, Any],
) -> DeepRagPolicyDecision:
    default_parallel = max(1, int(getattr(settings, "deep_rag_max_parallel_lanes", 3) or 3))
    default_threshold = max(6, int(getattr(settings, "deep_rag_full_read_chunk_threshold", 24) or 24))
    default_reflections = max(1, int(getattr(settings, "deep_rag_sync_reflection_rounds", 1) or 1))
    background_threshold = max(2, int(getattr(settings, "deep_rag_background_threshold", 4) or 4))
    score = _complexity_score(
        user_text=user_text,
        route=route,
        suggested_agent=suggested_agent,
        has_attachments=has_attachments,
        session_metadata=session_metadata,
    )
    semantic_routing = dict(session_metadata or {}).get("semantic_routing") or {}
    if not isinstance(semantic_routing, dict):
        semantic_routing = {}
    scope_kind = str(semantic_routing.get("requested_scope_kind") or "").strip().lower()
    if _is_authoritative_inventory_query(user_text) or scope_kind in {"graph_indexes", "session_access"}:
        return DeepRagPolicyDecision(
            mode=request.mode,
            search_mode="fast",
            preferred_agent="",
            prefer_full_reads=False,
            prefer_section_first=True,
            max_reflection_rounds=max(1, request.max_reflection_rounds or default_reflections),
            max_parallel_lanes=default_parallel,
            full_read_chunk_threshold=default_threshold,
            background_recommended=False,
            reasoning="authoritative inventory request should stay on the lightweight catalog path",
            complexity_score=0,
        )
    active_doc_focus = bool(dict(session_metadata or {}).get("active_doc_focus"))
    followup_doc_summary = active_doc_focus and bool(_FOLLOWUP_SUMMARY_HINTS.search(user_text))
    broad_research = bool(_BROAD_RESEARCH_HINTS.search(user_text))
    direct_doc_read = bool(_DOC_READ_HINTS.search(user_text) or _DIRECT_DOC_HINTS.search(user_text))

    if request.mode == "off":
        return DeepRagPolicyDecision(
            mode="off",
            search_mode="auto",
            preferred_agent="",
            prefer_full_reads=False,
            prefer_section_first=True,
            max_reflection_rounds=max(1, request.max_reflection_rounds or default_reflections),
            max_parallel_lanes=default_parallel,
            full_read_chunk_threshold=default_threshold,
            background_recommended=False,
            reasoning="deep_rag disabled by request metadata",
            complexity_score=score,
        )

    search_mode = "deep" if request.mode == "force" or score >= 3 or followup_doc_summary else "auto"
    preferred_agent = ""
    if route == "AGENT":
        if followup_doc_summary or broad_research or suggested_agent == "coordinator":
            preferred_agent = "coordinator"
        elif search_mode == "deep":
            preferred_agent = "rag_worker"
    prefer_full_reads = bool(followup_doc_summary or direct_doc_read or request.mode == "force")
    prefer_section_first = True if prefer_full_reads or score >= 2 else True
    background_recommended = bool(
        request.background_ok and (score >= background_threshold or (broad_research and _BACKGROUND_HINTS.search(user_text)))
    )
    max_reflection_rounds = request.max_reflection_rounds or default_reflections
    if request.mode == "force":
        max_reflection_rounds = max(max_reflection_rounds, default_reflections + 1)

    reasoning_parts = []
    if broad_research:
        reasoning_parts.append("broad research prompt")
    if followup_doc_summary:
        reasoning_parts.append("follow-up synthesis over prior doc focus")
    if direct_doc_read:
        reasoning_parts.append("direct document inspection requested")
    if score >= 3 and not reasoning_parts:
        reasoning_parts.append("complex evidence-sensitive query")
    if not reasoning_parts:
        reasoning_parts.append("grounded query can start in staged retrieval")

    return DeepRagPolicyDecision(
        mode=request.mode,
        search_mode=search_mode,
        preferred_agent=preferred_agent,
        prefer_full_reads=prefer_full_reads,
        prefer_section_first=prefer_section_first,
        max_reflection_rounds=max(1, int(max_reflection_rounds)),
        max_parallel_lanes=default_parallel,
        full_read_chunk_threshold=default_threshold,
        background_recommended=background_recommended,
        reasoning=", ".join(reasoning_parts),
        complexity_score=score,
    )


def _call_policy_llm(
    judge_llm: Any,
    *,
    request: DeepRagRequest,
    user_text: str,
    research_packet: str,
    route: str,
    suggested_agent: str,
    has_attachments: bool,
) -> _DeepRagPolicyOutput | None:
    if judge_llm is None:
        return None

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception:
        return None

    messages = [
        SystemMessage(content=_POLICY_SYSTEM_PROMPT),
        HumanMessage(
            content=_POLICY_HUMAN_TEMPLATE.format(
                route=route or "AGENT",
                suggested_agent=suggested_agent or "(default)",
                has_attachments=bool(has_attachments),
                requested_mode=request.mode,
                background_ok=request.background_ok,
                max_reflection_rounds=request.max_reflection_rounds or "",
                research_packet=research_packet or "(no prior context)",
                user_text=user_text,
            )
        ),
    ]

    try:
        structured = judge_llm.with_structured_output(_DeepRagPolicyOutput)
        result = structured.invoke(messages)
        if isinstance(result, _DeepRagPolicyOutput):
            return result
    except Exception:
        pass

    try:
        response = judge_llm.invoke(messages)
        text = getattr(response, "content", None) or str(response)
        payload = extract_json(text) or {}
        if not isinstance(payload, dict):
            return None
        return _DeepRagPolicyOutput.model_validate(payload)
    except Exception:
        return None


def decide_deep_rag_policy(
    settings: Any,
    judge_llm: Any,
    *,
    user_text: str,
    route: str,
    suggested_agent: str,
    has_attachments: bool,
    research_packet: str,
    session_metadata: Dict[str, Any] | None = None,
    request_metadata: Dict[str, Any] | None = None,
) -> DeepRagPolicyDecision:
    metadata = dict(request_metadata or {})
    request = parse_deep_rag_request(settings, metadata.get("deep_rag"))
    heuristic = _heuristic_policy(
        settings,
        request=request,
        user_text=user_text,
        route=route,
        suggested_agent=suggested_agent,
        has_attachments=has_attachments,
        session_metadata=dict(session_metadata or {}),
    )
    semantic_routing = dict(session_metadata or {}).get("semantic_routing") or {}
    if not isinstance(semantic_routing, dict):
        semantic_routing = {}
    scope_kind = str(semantic_routing.get("requested_scope_kind") or "").strip().lower()
    if route != "AGENT" or request.mode == "off":
        return heuristic
    if _is_authoritative_inventory_query(user_text) or scope_kind in {"graph_indexes", "session_access"}:
        return heuristic

    llm_policy = _call_policy_llm(
        judge_llm,
        request=request,
        user_text=user_text,
        research_packet=research_packet,
        route=route,
        suggested_agent=suggested_agent,
        has_attachments=has_attachments,
    )
    if llm_policy is None:
        return heuristic

    return DeepRagPolicyDecision(
        mode=request.mode,
        search_mode=normalize_deep_rag_search_mode(
            llm_policy.search_mode,
            default=heuristic.search_mode,
        ),
        preferred_agent=str(llm_policy.preferred_agent or heuristic.preferred_agent),
        prefer_full_reads=bool(llm_policy.prefer_full_reads or heuristic.prefer_full_reads),
        prefer_section_first=bool(llm_policy.prefer_section_first),
        max_reflection_rounds=max(
            1,
            int(
                llm_policy.max_reflection_rounds
                or request.max_reflection_rounds
                or heuristic.max_reflection_rounds
            ),
        ),
        max_parallel_lanes=heuristic.max_parallel_lanes,
        full_read_chunk_threshold=heuristic.full_read_chunk_threshold,
        background_recommended=bool(llm_policy.background_recommended and request.background_ok),
        reasoning=str(llm_policy.reasoning or heuristic.reasoning),
        complexity_score=heuristic.complexity_score,
    )


__all__ = [
    "DeepRagPolicyDecision",
    "DeepRagRequest",
    "decide_deep_rag_policy",
    "deep_rag_controller_hints",
    "deep_rag_search_mode",
    "normalize_deep_rag_mode",
    "normalize_deep_rag_search_mode",
    "parse_deep_rag_request",
]
