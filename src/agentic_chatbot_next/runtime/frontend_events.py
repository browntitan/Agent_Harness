from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


DETAIL_METADATA = "metadata"
DETAIL_SAFE_PREVIEW = "safe_preview"
SUPPORTED_DETAIL_LEVELS = {DETAIL_METADATA, DETAIL_SAFE_PREVIEW}

TOOL_PROGRESS_EVENT_TYPES = {"tool_intent", "tool_call", "tool_result", "tool_error"}
AGENT_PROGRESS_EVENT_TYPES = {
    "agent_start",
    "agent_selected",
    "decision_point",
    "route_decision",
    "worker_start",
    "worker_end",
}


def normalize_frontend_event_detail_level(raw: Any) -> str:
    value = str(raw or DETAIL_SAFE_PREVIEW).strip().lower()
    return value if value in SUPPORTED_DETAIL_LEVELS else DETAIL_SAFE_PREVIEW


def compact_preview(value: Any, *, limit: int) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class FrontendEventPolicy:
    enabled: bool = True
    show_status: bool = True
    show_agents: bool = True
    show_tools: bool = True
    show_parallel_groups: bool = True
    show_guidance: bool = True
    show_skills: bool = True
    show_context: bool = True
    show_memory_context: bool = False
    detail_level: str = DETAIL_SAFE_PREVIEW
    preview_chars: int = 480

    @classmethod
    def from_settings(cls, settings: Any | None) -> "FrontendEventPolicy":
        return cls(
            enabled=_coerce_bool(getattr(settings, "frontend_events_enabled", True), True),
            show_status=_coerce_bool(getattr(settings, "frontend_events_show_status", True), True),
            show_agents=_coerce_bool(getattr(settings, "frontend_events_show_agents", True), True),
            show_tools=_coerce_bool(getattr(settings, "frontend_events_show_tools", True), True),
            show_parallel_groups=_coerce_bool(getattr(settings, "frontend_events_show_parallel_groups", True), True),
            show_guidance=_coerce_bool(getattr(settings, "frontend_events_show_guidance", True), True),
            show_skills=_coerce_bool(getattr(settings, "frontend_events_show_skills", True), True),
            show_context=_coerce_bool(getattr(settings, "frontend_events_show_context", True), True),
            show_memory_context=_coerce_bool(getattr(settings, "frontend_events_show_memory_context", False), False),
            detail_level=normalize_frontend_event_detail_level(
                getattr(settings, "frontend_events_detail_level", DETAIL_SAFE_PREVIEW)
            ),
            preview_chars=max(0, _coerce_int(getattr(settings, "frontend_events_preview_chars", 480), 480)),
        )

    @property
    def safe_preview_enabled(self) -> bool:
        return self.detail_level == DETAIL_SAFE_PREVIEW

    @property
    def show_context_trace(self) -> bool:
        return bool(self.show_guidance or self.show_skills or self.show_context or self.show_memory_context)

    def preview(self, value: Any) -> str:
        if not self.safe_preview_enabled:
            return ""
        return compact_preview(value, limit=self.preview_chars)

    def allows_status_snapshot(self, payload: Mapping[str, Any] | None) -> bool:
        if not self.enabled:
            return False
        payload = dict(payload or {})
        if _is_tool_payload(payload):
            return self.show_tools
        if _is_parallel_payload(payload):
            return self.show_parallel_groups
        if _is_agent_payload(payload):
            return self.show_agents
        if _is_context_payload(payload):
            return self.show_context_trace
        return self.show_status

    def allows_progress_event(self, payload: Mapping[str, Any] | None) -> bool:
        if not self.enabled:
            return False
        event_type = str((payload or {}).get("type") or (payload or {}).get("event_type") or "").strip()
        if event_type in TOOL_PROGRESS_EVENT_TYPES:
            return self.show_tools
        if event_type in AGENT_PROGRESS_EVENT_TYPES:
            return self.show_agents
        return self.show_status

    def allows_translated_event(self, payload: Mapping[str, Any] | None) -> bool:
        return self.allows_status_snapshot(payload)


def _audit_kind(payload: Mapping[str, Any]) -> str:
    item = payload.get("agentic_audit_item")
    if isinstance(item, Mapping):
        return str(item.get("kind") or "").strip().lower()
    return ""


def _is_tool_payload(payload: Mapping[str, Any]) -> bool:
    return isinstance(payload.get("agentic_tool_call"), Mapping) or _audit_kind(payload) == "tool"


def _is_parallel_payload(payload: Mapping[str, Any]) -> bool:
    kind = _audit_kind(payload)
    return isinstance(payload.get("agentic_parallel_group"), Mapping) or kind in {"parallel", "sequence", "group"}


def _is_agent_payload(payload: Mapping[str, Any]) -> bool:
    kind = _audit_kind(payload)
    return isinstance(payload.get("agentic_agent_activity"), Mapping) or kind in {
        "agent",
        "handoff",
        "plan",
        "verify",
        "synthesize",
        "route",
        "decision",
        "revise",
    }


def _is_context_payload(payload: Mapping[str, Any]) -> bool:
    return str(payload.get("source_event_type") or "").strip() == "agent_context_loaded" or _audit_kind(payload) == "context"
