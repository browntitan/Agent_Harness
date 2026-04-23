from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, List, Sequence

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agentic_chatbot_next.contracts.messages import RuntimeMessage, utc_now_iso
from agentic_chatbot_next.observability.events import RuntimeEvent
from agentic_chatbot_next.utils.json_utils import extract_json, make_json_compatible


_FILE_KEYS = {
    "artifact_ref",
    "filename",
    "file",
    "file_path",
    "path",
    "source_path",
    "written_file",
}
_DOC_KEYS = {"doc_id", "document_id"}
_SKILL_METADATA_KEYS = {
    "skill_resolution",
    "planner_skill_resolution",
    "finalizer_skill_resolution",
    "verifier_skill_resolution",
}


def _safe_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _safe_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _clip_text(text: str, max_chars: int) -> str:
    clean = str(text or "")
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean
    if max_chars < 80:
        return clean[:max_chars].rstrip()
    head = max_chars // 2
    tail = max_chars - head - 34
    return f"{clean[:head].rstrip()}\n...[truncated]...\n{clean[-tail:].lstrip()}"


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def estimate_text_tokens(text: str, settings: Any | None = None) -> int:
    clean = str(text or "")
    if not clean:
        return 0
    if bool(getattr(settings, "tiktoken_enabled", True)):
        try:
            import tiktoken  # type: ignore

            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(clean))
        except Exception:
            pass
    return max(1, math.ceil(len(clean) / 4))


def estimate_message_tokens(messages: Iterable[Any], settings: Any | None = None) -> int:
    total = 0
    for message in messages:
        content = getattr(message, "content", "")
        if isinstance(message, RuntimeMessage):
            content = message.content
        total += estimate_text_tokens(str(content or ""), settings) + 4
    return total


@dataclass(frozen=True)
class ContextBudgetConfig:
    enabled: bool = False
    window_tokens: int = 32768
    target_ratio: float = 0.72
    autocompact_threshold: float = 0.85
    tool_result_max_tokens: int = 2000
    tool_results_total_tokens: int = 8000
    microcompact_target_tokens: int = 2400
    compact_recent_messages: int = 12
    restore_recent_files: int = 10
    restore_recent_skills: int = 6

    @classmethod
    def from_settings(cls, settings: Any | None) -> "ContextBudgetConfig":
        return cls(
            enabled=bool(getattr(settings, "context_budget_enabled", False)),
            window_tokens=max(1024, _safe_int(getattr(settings, "context_window_tokens", None), 32768)),
            target_ratio=min(0.95, max(0.25, _safe_float(getattr(settings, "context_target_ratio", None), 0.72))),
            autocompact_threshold=min(
                0.98,
                max(0.40, _safe_float(getattr(settings, "context_autocompact_threshold", None), 0.85)),
            ),
            tool_result_max_tokens=max(
                128,
                _safe_int(getattr(settings, "context_tool_result_max_tokens", None), 2000),
            ),
            tool_results_total_tokens=max(
                512,
                _safe_int(getattr(settings, "context_tool_results_total_tokens", None), 8000),
            ),
            microcompact_target_tokens=max(
                256,
                _safe_int(getattr(settings, "context_microcompact_target_tokens", None), 2400),
            ),
            compact_recent_messages=max(
                2,
                _safe_int(getattr(settings, "context_compact_recent_messages", None), 12),
            ),
            restore_recent_files=max(
                0,
                _safe_int(getattr(settings, "context_restore_recent_files", None), 10),
            ),
            restore_recent_skills=max(
                0,
                _safe_int(getattr(settings, "context_restore_recent_skills", None), 6),
            ),
        )

    @property
    def target_tokens(self) -> int:
        return max(1, int(self.window_tokens * self.target_ratio))

    @property
    def autocompact_tokens(self) -> int:
        return max(1, int(self.window_tokens * self.autocompact_threshold))


@dataclass
class ContextSection:
    name: str
    content: str
    title: str = ""
    priority: int = 50
    preserve: bool = False

    def render(self) -> str:
        body = str(self.content or "").strip()
        if not body:
            return ""
        title = str(self.title or "").strip()
        return f"## {title}\n{body}" if title else body


@dataclass
class ContextLedger:
    enabled: bool
    window_tokens: int
    target_tokens: int
    autocompact_tokens: int
    estimated_input_tokens: int = 0
    history_tokens: int = 0
    user_tokens: int = 0
    sections: List[dict[str, Any]] = field(default_factory=list)
    actions: List[dict[str, Any]] = field(default_factory=list)

    def add_section(self, name: str, original_tokens: int, kept_tokens: int, *, clipped: bool = False) -> None:
        self.sections.append(
            {
                "name": name,
                "original_tokens": int(original_tokens),
                "kept_tokens": int(kept_tokens),
                "clipped": bool(clipped),
            }
        )

    def add_action(self, action: str, **payload: Any) -> None:
        self.actions.append({"action": action, **make_json_compatible(payload)})

    def to_dict(self) -> dict[str, Any]:
        return make_json_compatible(asdict(self))


@dataclass
class RestoreSnapshot:
    recent_files: List[dict[str, str]] = field(default_factory=list)
    recent_skills: List[dict[str, str]] = field(default_factory=list)
    active_doc_focus: dict[str, Any] = field(default_factory=dict)
    latest_artifact_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return make_json_compatible(asdict(self))

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "RestoreSnapshot":
        payload = dict(raw or {})
        return cls(
            recent_files=[
                {str(k): str(v) for k, v in dict(item).items() if str(v)}
                for item in (payload.get("recent_files") or [])
                if isinstance(item, dict)
            ],
            recent_skills=[
                {str(k): str(v) for k, v in dict(item).items() if str(v)}
                for item in (payload.get("recent_skills") or [])
                if isinstance(item, dict)
            ],
            active_doc_focus=dict(payload.get("active_doc_focus") or {}),
            latest_artifact_refs=[str(item) for item in (payload.get("latest_artifact_refs") or []) if str(item)],
        )

    def merge(self, other: "RestoreSnapshot") -> "RestoreSnapshot":
        return RestoreSnapshot(
            recent_files=_dedupe_dicts([*self.recent_files, *other.recent_files], keys=("ref", "path", "doc_id")),
            recent_skills=_dedupe_dicts([*self.recent_skills, *other.recent_skills], keys=("skill_id", "skill_family_id")),
            active_doc_focus=dict(other.active_doc_focus or self.active_doc_focus or {}),
            latest_artifact_refs=_dedupe_strings([*self.latest_artifact_refs, *other.latest_artifact_refs]),
        )

    def render_prompt_block(self) -> str:
        lines: List[str] = []
        if self.recent_files:
            lines.append("Recent files and document handles:")
            for item in self.recent_files:
                label = item.get("label") or item.get("filename") or item.get("title") or item.get("path") or item.get("doc_id")
                details = {
                    key: value
                    for key, value in item.items()
                    if key in {"artifact_ref", "filename", "path", "source_path", "doc_id", "source_type"} and value
                }
                lines.append(f"- {label}: {json.dumps(details, ensure_ascii=False)}")
        if self.active_doc_focus:
            documents = self.active_doc_focus.get("documents") or []
            if documents:
                lines.append("Active document focus:")
                for doc in documents[:6]:
                    if isinstance(doc, dict):
                        lines.append(
                            "- "
                            + json.dumps(
                                {
                                    "doc_id": str(doc.get("doc_id") or ""),
                                    "title": str(doc.get("title") or ""),
                                    "source_path": str(doc.get("source_path") or ""),
                                },
                                ensure_ascii=False,
                            )
                        )
        if self.recent_skills:
            lines.append("Recent skills:")
            for item in self.recent_skills:
                lines.append(
                    "- "
                    + json.dumps(
                        {
                            "skill_id": item.get("skill_id", ""),
                            "skill_family_id": item.get("skill_family_id", ""),
                            "name": item.get("name", ""),
                            "agent_scope": item.get("agent_scope", ""),
                        },
                        ensure_ascii=False,
                    )
                )
        if self.latest_artifact_refs:
            lines.append("Latest artifact refs: " + ", ".join(self.latest_artifact_refs[:8]))
        if not lines:
            return ""
        lines.append("Reopen file or skill content with the available tools before relying on details not shown here.")
        return "\n".join(lines).strip()


@dataclass
class CompactBoundary:
    boundary_id: str
    created_at: str
    summary: str
    covered_message_ids: List[str]
    covered_until_message_id: str = ""
    covered_transcript_index: int = 0
    token_estimates: dict[str, int] = field(default_factory=dict)
    source_model: str = "extractive"
    restore_snapshot: RestoreSnapshot = field(default_factory=RestoreSnapshot)
    reason: str = "autocompact"
    recent_message_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["restore_snapshot"] = self.restore_snapshot.to_dict()
        return make_json_compatible(payload)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "CompactBoundary | None":
        payload = dict(raw or {})
        boundary_id = str(payload.get("boundary_id") or "").strip()
        summary = str(payload.get("summary") or "").strip()
        if not boundary_id or not summary:
            return None
        return cls(
            boundary_id=boundary_id,
            created_at=str(payload.get("created_at") or utc_now_iso()),
            summary=summary,
            covered_message_ids=[str(item) for item in (payload.get("covered_message_ids") or []) if str(item)],
            covered_until_message_id=str(payload.get("covered_until_message_id") or ""),
            covered_transcript_index=int(payload.get("covered_transcript_index") or 0),
            token_estimates={str(k): int(v) for k, v in dict(payload.get("token_estimates") or {}).items()},
            source_model=str(payload.get("source_model") or "extractive"),
            restore_snapshot=RestoreSnapshot.from_dict(payload.get("restore_snapshot")),
            reason=str(payload.get("reason") or "autocompact"),
            recent_message_count=int(payload.get("recent_message_count") or 0),
        )

    def render_prompt_block(self) -> str:
        return (
            f"Compact boundary `{self.boundary_id}` covers earlier conversation up to "
            f"{len(self.covered_message_ids)} message(s). Preserve its decisions and unresolved constraints.\n\n"
            f"{self.summary.strip()}"
        ).strip()


@dataclass
class BudgetedTurn:
    system_prompt: str
    history_messages: List[RuntimeMessage]
    ledger: ContextLedger
    compact_boundary: CompactBoundary | None = None
    restore_snapshot: RestoreSnapshot = field(default_factory=RestoreSnapshot)


def _dedupe_strings(items: Iterable[str], limit: int | None = None) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for item in items:
        clean = str(item or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
        if limit is not None and len(result) >= limit:
            break
    return result


def _dedupe_dicts(items: Iterable[dict[str, Any]], *, keys: Sequence[str], limit: int | None = None) -> List[dict[str, str]]:
    result: List[dict[str, str]] = []
    seen: set[str] = set()
    for raw in items:
        item = {str(k): str(v) for k, v in dict(raw or {}).items() if str(v)}
        identity = ""
        for key in keys:
            if item.get(key):
                identity = f"{key}:{item[key]}"
                break
        if not identity:
            identity = json.dumps(item, sort_keys=True)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(item)
        if limit is not None and len(result) >= limit:
            break
    return result


class ContextBudgetManager:
    def __init__(self, settings: Any | None = None, *, transcript_store: Any | None = None, event_sink: Any | None = None) -> None:
        self.settings = settings
        self.config = ContextBudgetConfig.from_settings(settings)
        self.transcript_store = transcript_store
        self.event_sink = event_sink

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def bind_runtime(self, *, transcript_store: Any | None = None, event_sink: Any | None = None) -> None:
        if transcript_store is not None:
            self.transcript_store = transcript_store
        if event_sink is not None:
            self.event_sink = event_sink

    def estimate_text(self, text: str) -> int:
        return estimate_text_tokens(text, self.settings)

    def estimate_messages(self, messages: Iterable[Any]) -> int:
        return estimate_message_tokens(messages, self.settings)

    def prepare_turn(
        self,
        *,
        agent_name: str,
        session_state: SessionState,
        user_text: str,
        sections: Sequence[ContextSection],
        history_messages: Sequence[RuntimeMessage],
        providers: Any | None = None,
        transcript_store: Any | None = None,
        event_sink: Any | None = None,
        job_id: str = "",
    ) -> BudgetedTurn:
        ledger = self._new_ledger()
        original_history = list(history_messages or [])
        if transcript_store is not None or event_sink is not None:
            self.bind_runtime(transcript_store=transcript_store, event_sink=event_sink)
        if not self.enabled:
            prompt = self._render_sections(sections)
            ledger.history_tokens = self.estimate_messages(original_history)
            ledger.user_tokens = self.estimate_text(user_text)
            ledger.estimated_input_tokens = ledger.history_tokens + ledger.user_tokens + self.estimate_text(prompt)
            for section in sections:
                tokens = self.estimate_text(section.render())
                ledger.add_section(section.name, tokens, tokens, clipped=False)
            return BudgetedTurn(system_prompt=prompt, history_messages=original_history, ledger=ledger)

        self._emit(
            "context_budget_estimated",
            session_state.session_id,
            agent_name=agent_name,
            job_id=job_id,
            payload={
                "phase": "before_budget",
                "history_tokens": self.estimate_messages(original_history),
                "section_tokens": self.estimate_text(self._render_sections(sections)),
                "user_tokens": self.estimate_text(user_text),
                "target_tokens": self.config.target_tokens,
                "autocompact_tokens": self.config.autocompact_tokens,
            },
        )

        boundary = CompactBoundary.from_dict(dict(session_state.metadata or {}).get("context_compact_boundary"))
        current_restore = self.build_restore_snapshot(session_state, original_history)
        restore = RestoreSnapshot.from_dict(dict(session_state.metadata or {}).get("context_restore_snapshot")).merge(current_restore)
        history = list(original_history)
        prompt_sections = self._augment_sections(sections, boundary=boundary, restore=restore)
        estimated = self.estimate_text(self._render_sections(prompt_sections)) + self.estimate_messages(history) + self.estimate_text(user_text)

        if estimated > self.config.autocompact_tokens and len(history) > self.config.compact_recent_messages:
            self._emit(
                "autocompact_started",
                session_state.session_id,
                agent_name=agent_name,
                job_id=job_id,
                payload={"estimated_tokens": estimated, "message_count": len(history)},
            )
            boundary = self.create_compact_boundary(
                session_state=session_state,
                messages=history,
                providers=providers,
                reason="autocompact",
                restore_snapshot=restore,
            )
            history = history[-self.config.compact_recent_messages :]
            restore = boundary.restore_snapshot.merge(self.build_restore_snapshot(session_state, history))
            prompt_sections = self._augment_sections(sections, boundary=boundary, restore=restore)
            session_state.metadata = {
                **dict(session_state.metadata or {}),
                "context_compact_boundary": boundary.to_dict(),
                "context_restore_snapshot": restore.to_dict(),
            }
            self._append_compaction(session_state.session_id, boundary)
            ledger.add_action(
                "autocompact",
                boundary_id=boundary.boundary_id,
                covered_messages=len(boundary.covered_message_ids),
                estimated_tokens_before=estimated,
            )
            self._emit(
                "autocompact_completed",
                session_state.session_id,
                agent_name=agent_name,
                job_id=job_id,
                payload={"boundary": boundary.to_dict(), "remaining_history_messages": len(history)},
            )

        history = self._trim_history_to_budget(
            history,
            budget=max(256, self.config.target_tokens - self.estimate_text(user_text) - self.estimate_text(self._render_sections(prompt_sections))),
            ledger=ledger,
        )
        prompt, final_sections = self._fit_sections(
            prompt_sections,
            max_tokens=max(512, self.config.target_tokens - self.estimate_messages(history) - self.estimate_text(user_text)),
            ledger=ledger,
        )
        ledger.history_tokens = self.estimate_messages(history)
        ledger.user_tokens = self.estimate_text(user_text)
        ledger.estimated_input_tokens = ledger.history_tokens + ledger.user_tokens + self.estimate_text(prompt)
        self._emit(
            "context_budget_estimated",
            session_state.session_id,
            agent_name=agent_name,
            job_id=job_id,
            payload={
                "phase": "after_budget",
                "ledger": ledger.to_dict(),
                "sections": [section.name for section in final_sections],
            },
        )
        if restore.render_prompt_block():
            self._emit(
                "context_restore_applied",
                session_state.session_id,
                agent_name=agent_name,
                job_id=job_id,
                payload=restore.to_dict(),
            )
        return BudgetedTurn(
            system_prompt=prompt,
            history_messages=history,
            ledger=ledger,
            compact_boundary=boundary,
            restore_snapshot=restore,
        )

    def manual_compact_session(
        self,
        *,
        session_state: SessionState,
        messages: Sequence[RuntimeMessage],
        providers: Any | None = None,
        preview: bool = False,
        reason: str = "manual",
    ) -> dict[str, Any]:
        restore = self.build_restore_snapshot(session_state, messages)
        boundary = self.create_compact_boundary(
            session_state=session_state,
            messages=list(messages),
            providers=providers,
            reason=reason,
            restore_snapshot=restore,
        )
        if not preview:
            session_state.metadata = {
                **dict(session_state.metadata or {}),
                "context_compact_boundary": boundary.to_dict(),
                "context_restore_snapshot": boundary.restore_snapshot.to_dict(),
            }
            if len(session_state.messages) > self.config.compact_recent_messages:
                session_state.messages = session_state.messages[-self.config.compact_recent_messages :]
            self._append_compaction(session_state.session_id, boundary)
            self._emit(
                "compact_boundary_created",
                session_state.session_id,
                agent_name=str(session_state.active_agent or ""),
                payload={"boundary": boundary.to_dict(), "preview": False},
            )
        return {
            "object": "context_compaction",
            "preview": bool(preview),
            "boundary": boundary.to_dict(),
            "recent_message_count": min(len(messages), self.config.compact_recent_messages),
        }

    def create_compact_boundary(
        self,
        *,
        session_state: SessionState,
        messages: Sequence[RuntimeMessage],
        providers: Any | None = None,
        reason: str = "autocompact",
        restore_snapshot: RestoreSnapshot | None = None,
    ) -> CompactBoundary:
        recent_count = self.config.compact_recent_messages
        covered = list(messages[:-recent_count] if len(messages) > recent_count else messages)
        if not covered:
            covered = list(messages)
        summary, source_model = self._summarize_messages(covered, providers=providers)
        restore = (restore_snapshot or self.build_restore_snapshot(session_state, messages)).merge(
            self.build_restore_snapshot(session_state, covered)
        )
        return CompactBoundary(
            boundary_id=f"cmp_{uuid.uuid4().hex[:16]}",
            created_at=utc_now_iso(),
            summary=summary,
            covered_message_ids=[message.message_id for message in covered],
            covered_until_message_id=str(covered[-1].message_id if covered else ""),
            covered_transcript_index=max(0, len(covered) - 1),
            token_estimates={
                "covered_tokens": self.estimate_messages(covered),
                "summary_tokens": self.estimate_text(summary),
            },
            source_model=source_model,
            restore_snapshot=restore,
            reason=reason,
            recent_message_count=min(len(messages), recent_count),
        )

    def build_restore_snapshot(
        self,
        session_state: SessionState,
        messages: Sequence[RuntimeMessage],
    ) -> RestoreSnapshot:
        files: List[dict[str, Any]] = []
        skills: List[dict[str, Any]] = []
        artifact_refs: List[str] = []
        active_doc_focus = dict((session_state.metadata or {}).get("active_doc_focus") or {})
        for message in reversed(list(messages or [])):
            artifact_refs.extend(str(item) for item in (message.artifact_refs or []) if str(item))
            metadata = dict(message.metadata or {})
            files.extend(self._extract_file_refs(metadata))
            skills.extend(self._extract_skill_refs(metadata))
            parsed = extract_json(message.content)
            if isinstance(parsed, dict):
                files.extend(self._extract_file_refs(parsed))
                skills.extend(self._extract_skill_refs(parsed))
        files.extend(self._extract_file_refs(dict(session_state.metadata or {})))
        skills.extend(self._extract_skill_refs(dict(session_state.metadata or {})))
        if active_doc_focus:
            files.extend(self._extract_file_refs(active_doc_focus))
        return RestoreSnapshot(
            recent_files=_dedupe_dicts(
                files,
                keys=("artifact_ref", "path", "source_path", "filename", "doc_id"),
                limit=self.config.restore_recent_files,
            ),
            recent_skills=_dedupe_dicts(
                skills,
                keys=("skill_id", "skill_family_id"),
                limit=self.config.restore_recent_skills,
            ),
            active_doc_focus=active_doc_focus,
            latest_artifact_refs=_dedupe_strings(artifact_refs, limit=8),
        )

    def budget_text_block(self, name: str, text: str, *, max_tokens: int | None = None) -> str:
        if not self.enabled:
            return str(text or "")
        limit = max_tokens or self.config.microcompact_target_tokens
        if self.estimate_text(text) <= limit:
            return str(text or "")
        return _clip_text(str(text or ""), limit * 4)

    def budget_tool_message(self, message: ToolMessage, *, tool_context: Any | None = None) -> ToolMessage:
        if not self.enabled:
            return message
        content = str(getattr(message, "content", "") or "")
        if self._is_budgeted_tool_result(content):
            return message
        original_tokens = self.estimate_text(content)
        if original_tokens <= self.config.tool_result_max_tokens:
            return message
        tool_name = str(getattr(message, "name", "") or "")
        full_ref = self._append_tool_result(tool_context=tool_context, tool_name=tool_name, content=content)
        budgeted = self.budget_tool_content(
            content,
            tool_name=tool_name,
            max_tokens=self.config.tool_result_max_tokens,
            full_result_ref=full_ref,
        )
        self._emit_tool_budgeted(tool_context, tool_name=tool_name, original_tokens=original_tokens, budgeted_content=budgeted, full_ref=full_ref)
        return self._copy_tool_message(message, content=budgeted)

    def microcompact_messages(
        self,
        messages: Sequence[Any],
        *,
        providers: Any | None = None,
        tool_context: Any | None = None,
    ) -> List[Any]:
        del providers
        items = list(messages or [])
        if not self.enabled:
            return items
        last_human_index = -1
        for index, message in enumerate(items):
            if isinstance(message, HumanMessage):
                last_human_index = index
        current = items[last_human_index + 1 :] if last_human_index >= 0 else items
        tool_messages = [message for message in current if isinstance(message, ToolMessage)]
        if not tool_messages:
            return items
        total = sum(self.estimate_text(str(getattr(message, "content", "") or "")) for message in tool_messages)
        threshold = min(self.config.tool_results_total_tokens, max(self.config.microcompact_target_tokens, self.config.tool_result_max_tokens))
        if total <= threshold:
            return items
        per_tool = max(128, self.config.microcompact_target_tokens // max(1, len(tool_messages)))
        changed = False
        compacted: List[Any] = []
        for message in items:
            if isinstance(message, ToolMessage) and message in tool_messages:
                content = str(getattr(message, "content", "") or "")
                if self._is_budgeted_tool_result(content) and self.estimate_text(content) <= per_tool:
                    compacted.append(message)
                    continue
                compacted_content = self.budget_tool_content(
                    content,
                    tool_name=str(getattr(message, "name", "") or ""),
                    max_tokens=per_tool,
                    full_result_ref=self._budgeted_ref_from_content(content),
                    microcompact=True,
                )
                compacted.append(self._copy_tool_message(message, content=compacted_content))
                changed = True
            else:
                compacted.append(message)
        if changed:
            session_id = str(getattr(getattr(tool_context, "session", None), "session_id", "") or "")
            self._emit(
                "microcompact_created",
                session_id,
                agent_name=str(getattr(tool_context, "active_agent", "") or ""),
                job_id=str((getattr(tool_context, "metadata", {}) or {}).get("job_id") or ""),
                payload={
                    "original_tool_tokens": total,
                    "target_tool_tokens": self.config.microcompact_target_tokens,
                    "tool_message_count": len(tool_messages),
                },
            )
        return compacted

    def budget_tool_content(
        self,
        content: str,
        *,
        tool_name: str = "",
        max_tokens: int | None = None,
        full_result_ref: str = "",
        microcompact: bool = False,
    ) -> str:
        limit = max(64, int(max_tokens or self.config.tool_result_max_tokens))
        original = str(content or "")
        original_tokens = self.estimate_text(original)
        parsed = extract_json(original)
        preview = self._tool_preview(original, parsed, max_chars=max(160, limit * 3))
        payload: dict[str, Any] = {
            "object": "budgeted_tool_result",
            "tool_name": tool_name,
            "budgeted": True,
            "microcompact": bool(microcompact),
            "original_tokens": original_tokens,
            "budget_tokens": min(original_tokens, limit),
            "preview": preview,
            "warnings": [
                "Tool output was reduced for model context. Use referenced files, document ids, or artifacts to reopen details."
            ],
        }
        if full_result_ref:
            payload["full_result_ref"] = full_result_ref
        key_fields = self._extract_key_fields(parsed)
        if key_fields:
            payload["key_fields"] = key_fields
        return json.dumps(make_json_compatible(payload), ensure_ascii=False)

    def _new_ledger(self) -> ContextLedger:
        return ContextLedger(
            enabled=self.enabled,
            window_tokens=self.config.window_tokens,
            target_tokens=self.config.target_tokens,
            autocompact_tokens=self.config.autocompact_tokens,
        )

    def _render_sections(self, sections: Sequence[ContextSection]) -> str:
        return "\n\n".join(section.render() for section in sections if section.render()).strip()

    def _augment_sections(
        self,
        sections: Sequence[ContextSection],
        *,
        boundary: CompactBoundary | None,
        restore: RestoreSnapshot,
    ) -> List[ContextSection]:
        result: List[ContextSection] = []
        inserted = False
        for section in sections:
            result.append(section)
            if not inserted:
                if boundary is not None:
                    result.append(
                        ContextSection(
                            name="compact_boundary",
                            title="Earlier Conversation Summary",
                            content=boundary.render_prompt_block(),
                            priority=95,
                            preserve=True,
                        )
                    )
                restore_block = restore.render_prompt_block()
                if restore_block:
                    result.append(
                        ContextSection(
                            name="restore_snapshot",
                            title="Restored Recent Files And Skills",
                            content=restore_block,
                            priority=90,
                            preserve=True,
                        )
                    )
                inserted = True
        return result

    def _fit_sections(
        self,
        sections: Sequence[ContextSection],
        *,
        max_tokens: int,
        ledger: ContextLedger,
    ) -> tuple[str, List[ContextSection]]:
        rendered = [(section, section.render(), self.estimate_text(section.render())) for section in sections]
        total = sum(tokens for _, _, tokens in rendered)
        if total <= max_tokens:
            for section, _, tokens in rendered:
                ledger.add_section(section.name, tokens, tokens, clipped=False)
            return self._render_sections(sections), list(sections)
        budget_left = max_tokens
        fitted: List[ContextSection] = []
        for section, text, tokens in sorted(rendered, key=lambda item: item[0].priority, reverse=True):
            if not text:
                continue
            if tokens <= budget_left or section.preserve:
                fitted.append(section)
                kept = tokens
                budget_left -= tokens
                ledger.add_section(section.name, tokens, kept, clipped=False)
                continue
            if budget_left <= 128:
                ledger.add_section(section.name, tokens, 0, clipped=True)
                ledger.add_action("section_dropped", section=section.name, original_tokens=tokens)
                continue
            clipped_content = _clip_text(section.content, max(256, budget_left * 4))
            clipped = ContextSection(
                name=section.name,
                title=section.title,
                content=clipped_content,
                priority=section.priority,
                preserve=section.preserve,
            )
            clipped_tokens = self.estimate_text(clipped.render())
            fitted.append(clipped)
            budget_left -= clipped_tokens
            ledger.add_section(section.name, tokens, clipped_tokens, clipped=True)
            ledger.add_action("section_clipped", section=section.name, original_tokens=tokens, kept_tokens=clipped_tokens)
        order = {id(section): index for index, section in enumerate(sections)}
        fitted.sort(key=lambda section: order.get(id(section), len(order)))
        return self._render_sections(fitted), fitted

    def _trim_history_to_budget(self, messages: List[RuntimeMessage], *, budget: int, ledger: ContextLedger) -> List[RuntimeMessage]:
        if budget <= 0:
            kept = messages[-2:]
            ledger.add_action("history_trimmed", original_messages=len(messages), kept_messages=len(kept), budget_tokens=budget)
            return kept
        kept: List[RuntimeMessage] = []
        total = 0
        for message in reversed(messages):
            tokens = self.estimate_text(message.content) + 4
            if kept and total + tokens > budget:
                break
            kept.append(message)
            total += tokens
        kept.reverse()
        if len(kept) < len(messages):
            ledger.add_action("history_trimmed", original_messages=len(messages), kept_messages=len(kept), budget_tokens=budget)
        return kept

    def _summarize_messages(self, messages: Sequence[RuntimeMessage], *, providers: Any | None = None) -> tuple[str, str]:
        extractive = self._extractive_summary(messages)
        judge = getattr(providers, "judge", None) if providers is not None else None
        if judge is None:
            return extractive, "extractive"
        prompt = (
            "Summarize the earlier conversation for restoring agent context after compaction. "
            "Preserve user goals, explicit constraints, decisions, unresolved questions, file/doc handles, skill names, "
            "tool outcomes, and promised next steps. Be concise and factual.\n\n"
            f"EARLIER_MESSAGES:\n{_clip_text(self._messages_digest(messages), 18000)}"
        )
        try:
            response = judge.invoke(
                [
                    SystemMessage(content="You write compact, durable conversation summaries for agent context."),
                    HumanMessage(content=prompt),
                ]
            )
            text = str(getattr(response, "content", None) or response).strip()
            if text:
                return _clip_text(text, 6000), self._model_identity(judge)
        except Exception:
            pass
        return extractive, "extractive"

    def _extractive_summary(self, messages: Sequence[RuntimeMessage]) -> str:
        lines = ["Extractive compact summary of earlier conversation:"]
        for message in list(messages)[-24:]:
            content = _compact_whitespace(message.content)
            if not content:
                continue
            lines.append(f"- {message.role}: {_clip_text(content, 360)}")
        return "\n".join(lines).strip()

    def _messages_digest(self, messages: Sequence[RuntimeMessage]) -> str:
        lines: List[str] = []
        for message in messages:
            label = message.role
            name = f" ({message.name})" if message.name else ""
            lines.append(f"{label}{name}: {_clip_text(message.content, 1200)}")
        return "\n\n".join(lines)

    def _model_identity(self, model: Any) -> str:
        for attr in ("model_name", "model", "deployment_name", "name"):
            value = getattr(model, attr, None)
            if value:
                return str(value)
        return model.__class__.__name__

    def _tool_preview(self, original: str, parsed: Any, *, max_chars: int) -> str:
        if isinstance(parsed, dict):
            key_fields = self._extract_key_fields(parsed)
            if key_fields:
                return _clip_text(json.dumps(key_fields, ensure_ascii=False), max_chars)
        if isinstance(parsed, list):
            preview = {"item_count": len(parsed), "first_items": parsed[:3]}
            return _clip_text(json.dumps(preview, ensure_ascii=False), max_chars)
        return _clip_text(original, max_chars)

    def _extract_key_fields(self, parsed: Any) -> dict[str, Any]:
        if not isinstance(parsed, dict):
            return {}
        keep: dict[str, Any] = {}
        for key, value in parsed.items():
            key_text = str(key)
            if key_text in {
                "answer",
                "error",
                "status",
                "success",
                "warnings",
                "filename",
                "written_file",
                "artifact",
                "artifact_ref",
                "doc_id",
                "title",
                "source_path",
                "citations",
                "followups",
                "summary",
                "summary_text",
                "processed_rows",
                "result_counts",
            }:
                keep[key_text] = self._preview_value(value)
        for key, value in parsed.items():
            if len(keep) >= 12:
                break
            if isinstance(value, (str, int, float, bool)) or value is None:
                keep.setdefault(str(key), value)
        return keep

    def _preview_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return _clip_text(value, 800)
        if isinstance(value, list):
            return {"count": len(value), "items": [self._preview_value(item) for item in value[:5]]}
        if isinstance(value, dict):
            return {str(k): self._preview_value(v) for k, v in list(value.items())[:10]}
        return value

    def _extract_file_refs(self, payload: Any) -> List[dict[str, Any]]:
        refs: List[dict[str, Any]] = []
        if isinstance(payload, dict):
            current: dict[str, Any] = {}
            for key, value in payload.items():
                key_text = str(key)
                if key_text in _FILE_KEYS and str(value or "").strip():
                    current[key_text] = str(value)
                elif key_text in _DOC_KEYS and str(value or "").strip():
                    current["doc_id"] = str(value)
                elif key_text in {"title", "label", "source_type"} and str(value or "").strip():
                    current[key_text] = str(value)
            if current:
                refs.append(current)
            for value in payload.values():
                refs.extend(self._extract_file_refs(value))
        elif isinstance(payload, list):
            for item in payload:
                refs.extend(self._extract_file_refs(item))
        return refs

    def _extract_skill_refs(self, payload: Any) -> List[dict[str, Any]]:
        refs: List[dict[str, Any]] = []
        if isinstance(payload, dict):
            if any(key in payload for key in ("skill_id", "skill_family_id", "name", "agent_scope")):
                item = {
                    "skill_id": str(payload.get("skill_id") or ""),
                    "skill_family_id": str(payload.get("skill_family_id") or payload.get("version_parent") or ""),
                    "name": str(payload.get("name") or ""),
                    "agent_scope": str(payload.get("agent_scope") or ""),
                }
                if item["skill_id"] or item["skill_family_id"]:
                    refs.append(item)
            for key in _SKILL_METADATA_KEYS:
                value = payload.get(key)
                if isinstance(value, dict):
                    refs.extend(self._extract_skill_refs(value.get("matches") or []))
            if "matches" in payload:
                refs.extend(self._extract_skill_refs(payload.get("matches")))
            for value in payload.values():
                if isinstance(value, (dict, list)):
                    refs.extend(self._extract_skill_refs(value))
        elif isinstance(payload, list):
            for item in payload:
                refs.extend(self._extract_skill_refs(item))
        return refs

    def _copy_tool_message(self, message: ToolMessage, *, content: str) -> ToolMessage:
        kwargs = {
            "content": content,
            "tool_call_id": str(getattr(message, "tool_call_id", "") or ""),
            "name": str(getattr(message, "name", "") or ""),
            "additional_kwargs": dict(getattr(message, "additional_kwargs", {}) or {}),
        }
        status = getattr(message, "status", None)
        if status:
            kwargs["status"] = status
        try:
            return ToolMessage(**kwargs)
        except TypeError:
            kwargs.pop("status", None)
            return ToolMessage(**kwargs)

    def _is_budgeted_tool_result(self, content: str) -> bool:
        parsed = extract_json(content)
        return isinstance(parsed, dict) and parsed.get("object") == "budgeted_tool_result"

    def _budgeted_ref_from_content(self, content: str) -> str:
        parsed = extract_json(content)
        if isinstance(parsed, dict):
            return str(parsed.get("full_result_ref") or "")
        return ""

    def _append_tool_result(self, *, tool_context: Any | None, tool_name: str, content: str) -> str:
        transcript_store = getattr(tool_context, "transcript_store", None) or self.transcript_store
        session = getattr(tool_context, "session", None)
        session_id = str(getattr(session, "session_id", "") or "")
        if transcript_store is None or not session_id:
            return ""
        result_id = f"toolres_{uuid.uuid4().hex[:16]}"
        payload = {
            "tool_result_id": result_id,
            "created_at": utc_now_iso(),
            "session_id": session_id,
            "job_id": str((getattr(tool_context, "metadata", {}) or {}).get("job_id") or ""),
            "agent_name": str(getattr(tool_context, "active_agent", "") or ""),
            "tool_name": tool_name,
            "content": content,
            "estimated_tokens": self.estimate_text(content),
        }
        try:
            if hasattr(transcript_store, "append_session_tool_result"):
                transcript_store.append_session_tool_result(session_id, payload)
            else:
                transcript_store.append_session_transcript(session_id, {"kind": "tool_result_full", **payload})
        except Exception:
            return ""
        return f"session:{session_id}:tool_result:{result_id}"

    def _append_compaction(self, session_id: str, boundary: CompactBoundary) -> None:
        store = self.transcript_store
        if store is None or not session_id:
            return
        try:
            if hasattr(store, "append_session_compaction"):
                store.append_session_compaction(session_id, boundary.to_dict())
        except Exception:
            return

    def _emit_tool_budgeted(
        self,
        tool_context: Any | None,
        *,
        tool_name: str,
        original_tokens: int,
        budgeted_content: str,
        full_ref: str,
    ) -> None:
        session_id = str(getattr(getattr(tool_context, "session", None), "session_id", "") or "")
        self._emit(
            "tool_result_budgeted",
            session_id,
            agent_name=str(getattr(tool_context, "active_agent", "") or ""),
            tool_name=tool_name,
            job_id=str((getattr(tool_context, "metadata", {}) or {}).get("job_id") or ""),
            payload={
                "original_tokens": original_tokens,
                "budgeted_tokens": self.estimate_text(budgeted_content),
                "full_result_ref": full_ref,
            },
        )

    def _emit(
        self,
        event_type: str,
        session_id: str,
        *,
        agent_name: str = "",
        tool_name: str = "",
        job_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not session_id or self.event_sink is None:
            return
        try:
            self.event_sink.emit(
                RuntimeEvent(
                    event_type=event_type,
                    session_id=session_id,
                    agent_name=agent_name,
                    tool_name=tool_name,
                    job_id=job_id,
                    payload=dict(payload or {}),
                )
            )
        except Exception:
            return


def build_microcompact_hook(
    manager: ContextBudgetManager | None,
    *,
    providers: Any | None = None,
    tool_context: Any | None = None,
):
    if manager is None or not manager.enabled:
        return None

    def _hook(state: dict[str, Any]) -> dict[str, Any]:
        messages = list((state or {}).get("messages") or [])
        return {
            "llm_input_messages": manager.microcompact_messages(
                messages,
                providers=providers,
                tool_context=tool_context,
            )
        }

    return _hook


__all__ = [
    "BudgetedTurn",
    "CompactBoundary",
    "ContextBudgetConfig",
    "ContextBudgetManager",
    "ContextLedger",
    "ContextSection",
    "RestoreSnapshot",
    "build_microcompact_hook",
    "estimate_message_tokens",
    "estimate_text_tokens",
]
