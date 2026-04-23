from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.messages import RuntimeMessage, SessionState
from agentic_chatbot_next.memory.projector import MemoryProjector
from agentic_chatbot_next.memory.scope import MemoryScope
from agentic_chatbot_next.memory.store import (
    MEMORY_TYPES,
    ManagedMemoryRecord,
    MemoryCandidate,
    MemoryEpisode,
    MemorySelection,
    MemoryStore,
    MemoryWriteOperation,
    MemoryWriteResult,
)
from agentic_chatbot_next.providers.llm_factory import build_providers
from agentic_chatbot_next.utils.json_utils import extract_json

logger = logging.getLogger(__name__)

_PROFILE_SIGNAL_RE = re.compile(
    r"\b(remember|save|store|prefer|i prefer|my preference|call me|my name is|i like|i always|use .* for me)\b",
    re.IGNORECASE,
)
_EXPLICIT_MEMORY_INTENT_RE = re.compile(r"\b(remember|save|store|note|keep track of|keep in mind)\b", re.IGNORECASE)


def _recent_messages(session_state: SessionState, *, limit: int = 8) -> List[RuntimeMessage]:
    rows: List[RuntimeMessage] = []
    for message in session_state.messages[-max(1, int(limit)) :]:
        if message.role in {"user", "assistant"} and str(message.content or "").strip():
            rows.append(message)
    return rows


def _active_doc_focus(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [dict(item) for item in dict(metadata or {}).get("active_doc_focus", {}).get("documents", []) if isinstance(item, dict)]


def _selector_live(settings: Any) -> bool:
    mode = str(getattr(settings, "memory_manager_mode", "shadow") or "shadow").strip().lower()
    if bool(getattr(settings, "memory_shadow_mode", False)):
        return False
    return mode in {"selector", "live"}


def _writer_live(settings: Any) -> bool:
    mode = str(getattr(settings, "memory_manager_mode", "shadow") or "shadow").strip().lower()
    if bool(getattr(settings, "memory_shadow_mode", False)):
        return False
    return mode == "live"


def _candidate_limit(settings: Any) -> int:
    try:
        return max(4, int(getattr(settings, "memory_candidate_top_k", 16) or 16))
    except (TypeError, ValueError):
        return 16


def _context_char_budget(settings: Any) -> int:
    try:
        return max(400, int(getattr(settings, "memory_context_token_budget", 1600) or 1600))
    except (TypeError, ValueError):
        return 1600


def _scope_priority(scope: str) -> int:
    return 2 if str(scope or "") == MemoryScope.conversation.value else 1


def _type_priority(memory_type: str) -> int:
    order = {
        "decision": 5,
        "constraint": 4,
        "open_loop": 3,
        "task_state": 2,
        "profile_preference": 1,
    }
    return order.get(str(memory_type or ""), 0)


class _SelectorOutput(BaseModel):
    selected_ids: List[str] = Field(default_factory=list)
    brief: str = Field(default="")
    must_include: List[str] = Field(default_factory=list)
    stale_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class _WriterOpOutput(BaseModel):
    operation: str = Field(default="ignore")
    scope: str = Field(default=MemoryScope.conversation.value)
    memory_type: str = Field(default="task_state")
    title: str = Field(default="")
    canonical_text: str = Field(default="")
    key: str = Field(default="")
    structured_payload: Dict[str, Any] = Field(default_factory=dict)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_turn_ids: List[str] = Field(default_factory=list)
    supersedes_ids: List[str] = Field(default_factory=list)
    ttl_hint: str = Field(default="")
    note: str = Field(default="")

    @field_validator("operation")
    @classmethod
    def _validate_operation(cls, value: str) -> str:
        normalized = str(value or "ignore").strip().lower()
        return normalized if normalized in {"ignore", "create", "reinforce", "update", "supersede"} else "ignore"

    @field_validator("scope")
    @classmethod
    def _validate_scope(cls, value: str) -> str:
        try:
            return MemoryScope(str(value or MemoryScope.conversation.value)).value
        except ValueError:
            return MemoryScope.conversation.value

    @field_validator("memory_type")
    @classmethod
    def _validate_memory_type(cls, value: str) -> str:
        normalized = str(value or "task_state").strip().lower()
        return normalized if normalized in MEMORY_TYPES else "task_state"


class _WriterOutput(BaseModel):
    operations: List[_WriterOpOutput] = Field(default_factory=list)


class MemoryCandidateRetriever:
    def __init__(self, store: MemoryStore, settings: Any) -> None:
        self.store = store
        self.settings = settings

    def retrieve(
        self,
        *,
        session_state: SessionState,
        query: str,
        scopes: Sequence[str],
    ) -> List[MemoryCandidate]:
        limit = _candidate_limit(self.settings)
        candidates: List[MemoryCandidate] = []
        seen: set[str] = set()

        def add(candidate: MemoryCandidate) -> None:
            if candidate.candidate_id in seen:
                return
            seen.add(candidate.candidate_id)
            candidates.append(candidate)

        recent = self.store.list_records(
            tenant_id=session_state.tenant_id,
            user_id=session_state.user_id,
            conversation_id=session_state.conversation_id,
            session_id=session_state.session_id,
            active_only=True,
            limit=limit * 2,
        )
        for record in recent[:limit]:
            add(
                MemoryCandidate(
                    candidate_id=record.memory_id,
                    candidate_kind="record",
                    score=(record.importance * 0.75) + (_type_priority(record.memory_type) * 0.03) + (_scope_priority(record.scope) * 0.02),
                    reason="recent_high_importance",
                    text=record.canonical_text,
                    scope=record.scope,
                    memory_type=record.memory_type,
                    updated_at=record.updated_at,
                    record=record,
                )
            )
        for memory_type, reason in (("decision", "active_decision"), ("open_loop", "open_loop"), ("constraint", "constraint")):
            rows = self.store.list_records(
                tenant_id=session_state.tenant_id,
                user_id=session_state.user_id,
                conversation_id=session_state.conversation_id,
                session_id=session_state.session_id,
                active_only=True,
                memory_types=[memory_type],
                limit=max(4, limit // 2),
            )
            for record in rows:
                add(
                    MemoryCandidate(
                        candidate_id=record.memory_id,
                        candidate_kind="record",
                        score=max(record.importance, 0.8 if memory_type == "decision" else 0.7),
                        reason=reason,
                        text=record.canonical_text,
                        scope=record.scope,
                        memory_type=record.memory_type,
                        updated_at=record.updated_at,
                        record=record,
                    )
                )
        for candidate in self.store.search_candidates(
            tenant_id=session_state.tenant_id,
            user_id=session_state.user_id,
            conversation_id=session_state.conversation_id,
            session_id=session_state.session_id,
            query=query,
            scopes=scopes,
            limit=limit,
        ):
            add(candidate)
        for candidate in self.store.latest_episode_candidates(
            tenant_id=session_state.tenant_id,
            user_id=session_state.user_id,
            conversation_id=session_state.conversation_id,
            session_id=session_state.session_id,
            query=query,
            limit=max(3, limit // 3),
        ):
            add(candidate)
        candidates.sort(
            key=lambda item: (
                item.score,
                _type_priority(item.memory_type),
                _scope_priority(item.scope),
            ),
            reverse=True,
        )
        return candidates[:limit]


class MemorySelector:
    def __init__(self, store: MemoryStore, settings: Any) -> None:
        self.store = store
        self.settings = settings
        self._cached_models: Dict[str, Any] = {}

    def _resolve_model(self, providers: Any | None, *, role: str) -> Any | None:
        if providers is None:
            return None
        override = str(
            getattr(self.settings, "memory_selector_model" if role == "selector" else "memory_writer_model", "") or ""
        ).strip()
        cache_key = f"{role}:{override or '<default>'}"
        if cache_key in self._cached_models:
            return self._cached_models[cache_key]
        if not override:
            model = getattr(providers, "judge", None) or getattr(providers, "chat", None)
            self._cached_models[cache_key] = model
            return model
        try:
            resolved = build_providers(
                self.settings,
                embeddings=getattr(providers, "embeddings", None),
                judge_model_override=override,
            )
            model = getattr(resolved, "judge", None)
        except Exception as exc:
            logger.warning("Failed to resolve memory %s model override %r: %s", role, override, exc)
            model = getattr(providers, "judge", None) or getattr(providers, "chat", None)
        self._cached_models[cache_key] = model
        return model

    def select(
        self,
        *,
        session_state: SessionState,
        agent: AgentDefinition,
        user_text: str,
        providers: Any | None,
        candidates: Sequence[MemoryCandidate],
    ) -> MemorySelection:
        if not candidates:
            return MemorySelection(mode="deterministic")
        if not _selector_live(self.settings):
            return self._deterministic_selection(candidates)
        model = self._resolve_model(providers, role="selector")
        if model is None:
            return self._deterministic_selection(candidates)
        messages = [
            SystemMessage(
                content=(
                    "You are the memory context selector for a conversational AI runtime.\n"
                    "Choose only the memory candidates that materially improve the answer to the user's next request.\n"
                    "Favor current decisions, constraints, active task state, and durable preferences when relevant.\n"
                    "Return concise JSON only."
                )
            ),
            HumanMessage(
                content=(
                    "Return JSON with keys selected_ids, brief, must_include, stale_ids, confidence.\n\n"
                    f"ACTIVE_AGENT: {agent.name}\n"
                    f"USER_TEXT: {user_text}\n"
                    f"DOC_FOCUS: {str(_active_doc_focus(dict(session_state.metadata or {})))[:1000]}\n"
                    f"RECENT_MESSAGES: {self._message_digest(session_state)}\n"
                    f"CONTEXT_BUDGET_CHARS: {_context_char_budget(self.settings)}\n"
                    f"CANDIDATES: {self._serialize_candidates(candidates)}"
                )
            ),
        ]
        parsed: _SelectorOutput | None = None
        try:
            structured = model.with_structured_output(_SelectorOutput)
            result = structured.invoke(messages)
            if isinstance(result, _SelectorOutput):
                parsed = result
        except Exception:
            parsed = None
        if parsed is None:
            try:
                response = model.invoke(messages)
                payload = extract_json(getattr(response, "content", None) or str(response)) or {}
                if isinstance(payload, dict):
                    parsed = _SelectorOutput.model_validate(payload)
            except Exception:
                parsed = None
        if parsed is None:
            return self._deterministic_selection(candidates)
        if parsed.selected_ids:
            self.store.touch_records(parsed.selected_ids)
        return MemorySelection(
            selected_ids=list(parsed.selected_ids),
            brief=str(parsed.brief or "").strip(),
            must_include=list(parsed.must_include),
            stale_ids=list(parsed.stale_ids),
            confidence=float(parsed.confidence),
            mode="llm",
        )

    def _message_digest(self, session_state: SessionState) -> str:
        return "\n".join(
            f"{message.role}: {str(message.content or '')[:220]}"
            for message in _recent_messages(session_state, limit=6)
        )

    def _serialize_candidates(self, candidates: Sequence[MemoryCandidate]) -> str:
        rows: List[str] = []
        for item in candidates:
            rows.append(
                str(
                    {
                        "id": item.candidate_id,
                        "kind": item.candidate_kind,
                        "reason": item.reason,
                        "score": round(item.score, 3),
                        "scope": item.scope,
                        "memory_type": item.memory_type,
                        "updated_at": item.updated_at,
                        "text": item.text[:280],
                    }
                )
            )
        return "\n".join(rows)

    def _deterministic_selection(self, candidates: Sequence[MemoryCandidate]) -> MemorySelection:
        chosen = list(candidates[: min(len(candidates), 6)])
        selected_ids = [candidate.candidate_id for candidate in chosen]
        record_ids = [candidate.record.memory_id for candidate in chosen if candidate.record is not None]
        if record_ids:
            self.store.touch_records(record_ids)
        lines: List[str] = []
        consumed = 0
        budget = _context_char_budget(self.settings)
        for candidate in chosen:
            line = f"- [{candidate.reason}] {candidate.text.strip()}"
            if consumed + len(line) > budget and lines:
                break
            lines.append(line)
            consumed += len(line) + 1
        return MemorySelection(
            selected_ids=selected_ids,
            brief="\n".join(lines).strip(),
            must_include=[],
            stale_ids=[],
            confidence=0.45,
            mode="deterministic",
        )


class MemoryWriteManager:
    def __init__(
        self,
        store: MemoryStore,
        settings: Any,
        *,
        selector: MemorySelector,
        projector: MemoryProjector | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.selector = selector
        self.projector = projector

    def process_turn(
        self,
        *,
        session_state: SessionState,
        latest_user_text: str,
        providers: Any | None,
    ) -> MemoryWriteResult:
        shadow = bool(getattr(self.settings, "memory_shadow_mode", False)) or not _writer_live(self.settings)
        operations = self._extract_operations(
            session_state=session_state,
            latest_user_text=latest_user_text,
            providers=providers,
        )
        result = self.store.apply_operations(
            tenant_id=session_state.tenant_id,
            user_id=session_state.user_id,
            conversation_id=session_state.conversation_id,
            session_id=session_state.session_id,
            operations=operations,
            source="memory_manager",
            shadow=shadow,
        )
        if self._should_refresh_episode(session_state):
            episode = self._build_episode_summary(session_state=session_state, providers=providers)
            self.store.upsert_episode(episode, shadow=shadow)
        if not shadow and self.projector is not None:
            self.projector.project_session(
                tenant_id=session_state.tenant_id,
                user_id=session_state.user_id,
                conversation_id=session_state.conversation_id,
                session_id=session_state.session_id,
            )
        return result

    def _extract_operations(
        self,
        *,
        session_state: SessionState,
        latest_user_text: str,
        providers: Any | None,
    ) -> List[MemoryWriteOperation]:
        model = self.selector._resolve_model(providers, role="writer")
        message_ids = [message.message_id for message in _recent_messages(session_state, limit=4)]
        existing = self.store.list_records(
            tenant_id=session_state.tenant_id,
            user_id=session_state.user_id,
            conversation_id=session_state.conversation_id,
            session_id=session_state.session_id,
            active_only=True,
            limit=12,
        )
        if model is None:
            return self._heuristic_operations(latest_user_text=latest_user_text, message_ids=message_ids)
        messages = [
            SystemMessage(
                content=(
                    "You are the memory write manager for a conversational AI runtime.\n"
                    "Return JSON only with an operations array.\n"
                    "Allowed operations: ignore, create, reinforce, update, supersede.\n"
                    "Never invent durable user profile memory without explicit evidence."
                )
            ),
            HumanMessage(
                content=(
                    "Return JSON only with {\"operations\": [...]}.\n\n"
                    f"LATEST_USER_TEXT: {latest_user_text}\n"
                    f"RECENT_MESSAGES: {self.selector._message_digest(session_state)}\n"
                    f"EXISTING_ACTIVE_MEMORIES: {self._serialize_records(existing)}\n"
                    "Focus on profile preferences, active task state, decisions, constraints, and open loops."
                )
            ),
        ]
        parsed: _WriterOutput | None = None
        try:
            structured = model.with_structured_output(_WriterOutput)
            result = structured.invoke(messages)
            if isinstance(result, _WriterOutput):
                parsed = result
        except Exception:
            parsed = None
        if parsed is None:
            try:
                response = model.invoke(messages)
                payload = extract_json(getattr(response, "content", None) or str(response)) or {}
                if isinstance(payload, dict):
                    parsed = _WriterOutput.model_validate(payload)
            except Exception:
                parsed = None
        if parsed is None:
            return self._heuristic_operations(latest_user_text=latest_user_text, message_ids=message_ids)
        return self._validate_operations(
            raw_ops=list(parsed.operations or []),
            latest_user_text=latest_user_text,
            existing=existing,
            default_evidence_ids=message_ids,
        )

    def _serialize_records(self, records: Sequence[ManagedMemoryRecord]) -> str:
        rows: List[str] = []
        for record in records:
            rows.append(
                str(
                    {
                        "memory_id": record.memory_id,
                        "scope": record.scope,
                        "memory_type": record.memory_type,
                        "key": record.key,
                        "title": record.title,
                        "canonical_text": record.canonical_text[:240],
                        "importance": round(record.importance, 3),
                        "confidence": round(record.confidence, 3),
                    }
                )
            )
        return "\n".join(rows)

    def _heuristic_operations(self, *, latest_user_text: str, message_ids: Sequence[str]) -> List[MemoryWriteOperation]:
        text = str(latest_user_text or "").strip()
        if not text:
            return []
        operations: List[MemoryWriteOperation] = []
        if _PROFILE_SIGNAL_RE.search(text):
            operations.append(
                MemoryWriteOperation(
                    operation="create",
                    scope=MemoryScope.user.value,
                    memory_type="profile_preference",
                    title="user preference",
                    canonical_text=text[:240],
                    key="user_preference",
                    importance=0.8,
                    confidence=0.55,
                    evidence_turn_ids=list(message_ids),
                    note="Heuristic fallback profile preference capture.",
                )
            )
        elif _EXPLICIT_MEMORY_INTENT_RE.search(text):
            operations.append(
                MemoryWriteOperation(
                    operation="create",
                    scope=MemoryScope.conversation.value,
                    memory_type="task_state",
                    title="conversation note",
                    canonical_text=text[:240],
                    key="conversation_note",
                    importance=0.6,
                    confidence=0.5,
                    evidence_turn_ids=list(message_ids),
                    note="Heuristic fallback conversation note capture.",
                )
            )
        return operations

    def _validate_operations(
        self,
        *,
        raw_ops: Sequence[_WriterOpOutput],
        latest_user_text: str,
        existing: Sequence[ManagedMemoryRecord],
        default_evidence_ids: Sequence[str],
    ) -> List[MemoryWriteOperation]:
        valid: List[MemoryWriteOperation] = []
        existing_ids = {record.memory_id for record in existing}
        for raw in raw_ops:
            op = str(raw.operation or "").strip().lower()
            if op == "ignore":
                continue
            scope = MemoryScope.user.value if str(raw.scope) == MemoryScope.user.value else MemoryScope.conversation.value
            memory_type = str(raw.memory_type or "").strip().lower()
            if scope == MemoryScope.user.value and memory_type not in {"profile_preference", "constraint", "decision"}:
                continue
            if scope == MemoryScope.user.value and memory_type == "profile_preference":
                repeated_existing = any(
                    record.scope == MemoryScope.user.value and record.memory_type == "profile_preference"
                    for record in existing
                )
                if not _PROFILE_SIGNAL_RE.search(latest_user_text) and not repeated_existing:
                    continue
            canonical_text = str(raw.canonical_text or "").strip()
            if not canonical_text:
                continue
            supersedes_ids = [str(item) for item in raw.supersedes_ids if str(item) in existing_ids]
            evidence_turn_ids = [str(item) for item in raw.evidence_turn_ids if str(item)] or list(default_evidence_ids)
            valid.append(
                MemoryWriteOperation(
                    operation=op,
                    scope=scope,
                    memory_type=memory_type if memory_type in MEMORY_TYPES else "task_state",
                    title=str(raw.title or raw.key or memory_type).strip(),
                    canonical_text=canonical_text[:1200],
                    key=str(raw.key or "").strip(),
                    structured_payload=dict(raw.structured_payload or {}),
                    importance=max(0.0, min(float(raw.importance), 1.0)),
                    confidence=max(0.0, min(float(raw.confidence), 1.0)),
                    evidence_turn_ids=evidence_turn_ids,
                    supersedes_ids=supersedes_ids,
                    ttl_hint=str(raw.ttl_hint or ""),
                    note=str(raw.note or ""),
                )
            )
        return valid

    def _should_refresh_episode(self, session_state: SessionState) -> bool:
        message_count = len(_recent_messages(session_state, limit=12))
        if message_count >= 8:
            return True
        latest_user = next((msg for msg in reversed(session_state.messages) if msg.role == "user"), None)
        text = str(getattr(latest_user, "content", "") or "")
        return bool(re.search(r"\b(plan|status|decision|constraint|next step|follow up)\b", text, re.IGNORECASE))

    def _build_episode_summary(self, *, session_state: SessionState, providers: Any | None) -> MemoryEpisode:
        messages = _recent_messages(session_state, limit=10)
        digest = "\n".join(f"{message.role}: {str(message.content or '')[:240]}" for message in messages)
        summary_text = digest[:900]
        topic_hint = "conversation episode"
        model = self.selector._resolve_model(providers, role="writer")
        if model is not None:
            try:
                response = model.invoke(
                    [
                        SystemMessage(
                            content=(
                                "Summarize the recent transcript window for long-horizon memory retrieval.\n"
                                "Return JSON only with keys summary_text, topic_hint, importance."
                            )
                        ),
                        HumanMessage(content=f"RECENT_MESSAGES:\n{digest}"),
                    ]
                )
                payload = extract_json(getattr(response, "content", None) or str(response)) or {}
                if isinstance(payload, dict):
                    summary_text = str(payload.get("summary_text") or summary_text)[:1200]
                    topic_hint = str(payload.get("topic_hint") or topic_hint)[:240]
                    importance = float(payload.get("importance") or 0.6)
                else:
                    importance = 0.6
            except Exception:
                importance = 0.6
        else:
            importance = 0.6
        total_messages = int(dict(session_state.metadata or {}).get("history_total_messages") or len(session_state.messages))
        start_index = max(0, total_messages - len(messages))
        return MemoryEpisode(
            episode_id="",
            tenant_id=session_state.tenant_id,
            user_id=session_state.user_id,
            conversation_id=session_state.conversation_id,
            session_id=session_state.session_id,
            summary_text=summary_text.strip(),
            topic_hint=topic_hint.strip(),
            start_turn_index=start_index,
            end_turn_index=total_messages,
            message_ids=[message.message_id for message in messages if str(message.message_id or "")],
            importance=max(0.0, min(float(importance), 1.0)),
        )
