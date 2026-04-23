from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


DEFAULT_CLARIFICATION_SENSITIVITY = 50
MIN_CLARIFICATION_SENSITIVITY = 0
MAX_CLARIFICATION_SENSITIVITY = 100

_CLARIFICATION_JSON_TAG_RE = re.compile(
    r"<clarification_request>\s*(\{.*?\})\s*</clarification_request>",
    re.IGNORECASE | re.DOTALL,
)
_CLARIFICATION_BLOCK_RE = re.compile(
    r"<clarification_request>\s*(.*?)\s*</clarification_request>",
    re.IGNORECASE | re.DOTALL,
)


def _extract_xml_field(body: str, field: str) -> str:
    match = re.search(
        rf"<{re.escape(field)}>\s*(.*?)\s*</{re.escape(field)}>",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def _parse_options(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return ()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in raw.split(",")]
    else:
        parsed = value
    return tuple(str(item).strip() for item in (parsed or []) if str(item).strip())


def _normalize_reason(reason: str, question: str, options: Sequence[str]) -> str:
    clean = str(reason or "").strip()
    haystack = " ".join([clean, question, " ".join(options)]).casefold()
    if (
        "output format" in haystack
        or "format" in haystack
        and any(option.casefold() in {"textual synthesis", "diagram", "table", "mixed"} for option in options)
    ):
        return "answer_format_selection"
    return clean


@dataclass(frozen=True)
class ClarificationRequest:
    question: str
    reason: str = ""
    options: tuple[str, ...] = ()
    source_agent: str = ""
    blocking: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "reason": self.reason,
            "options": list(self.options),
            "source_agent": self.source_agent,
            "blocking": self.blocking,
        }


def normalize_clarification_sensitivity(
    value: Any,
    *,
    default: int = DEFAULT_CLARIFICATION_SENSITIVITY,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(MIN_CLARIFICATION_SENSITIVITY, min(MAX_CLARIFICATION_SENSITIVITY, parsed))


def clarification_policy_bucket(value: Any) -> str:
    score = normalize_clarification_sensitivity(value)
    if score <= 24:
        return "low"
    if score >= 75:
        return "high"
    return "balanced"


def build_clarification_policy_text(value: Any) -> str:
    score = normalize_clarification_sensitivity(value)
    bucket = clarification_policy_bucket(score)
    bucket_rules = {
        "low": (
            "For soft ambiguity, proceed with a reasonable assumption unless the ambiguity would "
            "materially change the tools, sources, or answer shape."
        ),
        "balanced": (
            "For soft ambiguity, ask a clarification question when the ambiguity would materially "
            "change execution, evidence scope, or the answer shape."
        ),
        "high": (
            "For soft ambiguity, ask a clarification question whenever multiple plausible "
            "interpretations could change the work performed."
        ),
    }
    return (
        "Clarification sensitivity is "
        f"{score}/100 ({bucket}).\n"
        "Always ask for clarification on hard blockers such as missing required user input, "
        "ambiguous named-document resolution, required source-scope choice, or unsafe ambiguity "
        "that would make the result misleading.\n"
        f"{bucket_rules[bucket]}\n"
        "When you must ask for clarification, ask exactly one blocking question and format the "
        "response as "
        '<clarification_request>{"question":"...","reason":"...","options":["..."]}</clarification_request>.'
    )


def parse_clarification_request(
    text: str,
    *,
    source_agent: str = "",
    blocking_default: bool = True,
) -> tuple[str, ClarificationRequest | None]:
    raw_text = str(text or "").strip()
    match = _CLARIFICATION_BLOCK_RE.search(raw_text)
    if match is None:
        return raw_text, None
    body = match.group(1).strip()
    payload: Dict[str, Any] = {}
    json_match = _CLARIFICATION_JSON_TAG_RE.search(raw_text)
    if json_match is not None:
        try:
            payload = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            payload = {}
    if not payload:
        question = _extract_xml_field(body, "question")
        reason = _extract_xml_field(body, "reason")
        options = _parse_options(_extract_xml_field(body, "options"))
        blocking_text = _extract_xml_field(body, "blocking").casefold()
        blocking = blocking_default if not blocking_text else blocking_text not in {"false", "0", "no"}
        payload = {
            "question": question,
            "reason": reason,
            "options": list(options),
            "blocking": blocking,
        }
    try:
        question = str(payload.get("question") or "").strip()
    except AttributeError:
        return raw_text, None
    if not question:
        return raw_text, None
    options = _parse_options(payload.get("options"))
    reason = _normalize_reason(str(payload.get("reason") or "").strip(), question, options)
    blocking = bool(payload.get("blocking", blocking_default))
    visible_text = _CLARIFICATION_BLOCK_RE.sub("", raw_text).strip() or question
    return (
        visible_text,
        ClarificationRequest(
            question=question,
            reason=reason,
            options=options,
            source_agent=source_agent,
            blocking=blocking,
        ),
    )


def clarification_turn_metadata(
    request: ClarificationRequest | None,
    *,
    agent_name: str,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata = dict(extra or {})
    metadata["agent_name"] = agent_name
    if request is None:
        metadata.setdefault("turn_outcome", "final_answer")
        metadata.pop("clarification", None)
        return metadata
    metadata["turn_outcome"] = "clarification_request"
    metadata["clarification"] = request.to_dict()
    return metadata


def is_clarification_turn(metadata: Dict[str, Any] | None) -> bool:
    payload = dict(metadata or {})
    return (
        str(payload.get("turn_outcome") or "").strip().lower() == "clarification_request"
        and isinstance(payload.get("clarification"), dict)
    )


def clarification_from_metadata(metadata: Dict[str, Any] | None) -> ClarificationRequest | None:
    payload = dict(metadata or {})
    raw = dict(payload.get("clarification") or {})
    question = str(raw.get("question") or "").strip()
    if not question:
        return None
    options = tuple(str(item).strip() for item in (raw.get("options") or []) if str(item).strip())
    return ClarificationRequest(
        question=question,
        reason=str(raw.get("reason") or "").strip(),
        options=options,
        source_agent=str(raw.get("source_agent") or payload.get("agent_name") or "").strip(),
        blocking=bool(raw.get("blocking", True)),
    )


def contract_clarification_request(
    *,
    answer: str,
    followups: Sequence[str] | None = None,
    warnings: Sequence[str] | None = None,
    source_agent: str,
) -> ClarificationRequest | None:
    warning_set = {str(item).strip() for item in (warnings or []) if str(item).strip()}
    if "KB_COLLECTION_SELECTION_REQUIRED" in warning_set:
        return ClarificationRequest(
            question=str(answer or "").strip(),
            reason="kb_collection_selection",
            options=tuple(str(item).strip() for item in (followups or []) if str(item).strip()),
            source_agent=source_agent,
            blocking=True,
        )
    if "RETRIEVAL_SCOPE_AMBIGUOUS" in warning_set:
        return ClarificationRequest(
            question=str(answer or "").strip(),
            reason="retrieval_scope_ambiguous",
            options=tuple(str(item).strip() for item in (followups or []) if str(item).strip()),
            source_agent=source_agent,
            blocking=True,
        )
    if "REQUESTED_DOCS_AMBIGUOUS" in warning_set:
        return ClarificationRequest(
            question=(
                "I found multiple indexed documents that could match your request. "
                "Which exact title or path should I use?"
            ),
            reason="requested_docs_ambiguous",
            options=tuple(str(item).strip() for item in (followups or []) if str(item).strip()),
            source_agent=source_agent,
            blocking=True,
        )
    if "SOFT_QUERY_AMBIGUITY" in warning_set:
        return ClarificationRequest(
            question=str(answer or "").strip(),
            reason="soft_query_ambiguity",
            options=tuple(str(item).strip() for item in (followups or []) if str(item).strip()),
            source_agent=source_agent,
            blocking=True,
        )
    if "NAMESPACE_SCOPE_SELECTION_REQUIRED" in warning_set:
        return ClarificationRequest(
            question=str(answer or "").strip(),
            reason="namespace_scope_selection",
            options=tuple(str(item).strip() for item in (followups or []) if str(item).strip()),
            source_agent=source_agent,
            blocking=True,
        )
    return None


def append_clarification_policy_context(context: str, *, sensitivity: Any) -> str:
    policy = build_clarification_policy_text(sensitivity)
    base = str(context or "").strip()
    if not base:
        return f"Clarification policy:\n{policy}"
    return f"{base}\n\nClarification policy:\n{policy}"


def clarification_options_preview(options: Sequence[str] | None, *, limit: int = 3) -> List[str]:
    return [
        str(item).strip()
        for item in list(options or [])[:limit]
        if str(item).strip()
    ]


def pending_clarification_prompt_block(metadata: Dict[str, Any] | None) -> str:
    pending = dict((metadata or {}).get("pending_clarification") or {})
    question = str(pending.get("question") or "").strip()
    if not question:
        return ""
    lines = [
        "## Pending Clarification",
        "A clarification request is still pending from the prior assistant turn.",
        f"Previous question: {question}",
    ]
    reason = str(pending.get("reason") or "").strip()
    if reason:
        lines.append(f"Reason: {reason}")
    options = clarification_options_preview(pending.get("options"))
    if options:
        lines.append("Suggested options: " + ", ".join(options))
    lines.append(
        "Treat the latest user message as the likely answer to that question unless the user clearly changed topic."
    )
    return "\n".join(lines)
