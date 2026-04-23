from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence


MEMORY_TYPES = {
    "profile_preference",
    "task_state",
    "decision",
    "constraint",
    "open_loop",
}


@dataclass
class ManagedMemoryRecord:
    memory_id: str
    tenant_id: str
    user_id: str
    conversation_id: str
    session_id: str
    scope: str
    memory_type: str
    key: str
    title: str
    canonical_text: str
    structured_payload: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    confidence: float = 0.5
    active: bool = True
    superseded_by: str = ""
    provenance_turn_ids: List[str] = field(default_factory=list)
    last_used_at: str = ""
    created_at: str = ""
    updated_at: str = ""
    source: str = ""
    ttl_hint: str = ""


@dataclass
class MemoryObservation:
    observation_id: str
    memory_id: str
    tenant_id: str
    user_id: str
    conversation_id: str
    session_id: str
    operation: str
    evidence_turn_ids: List[str] = field(default_factory=list)
    note: str = ""
    raw_payload: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    created_at: str = ""


@dataclass
class MemoryEpisode:
    episode_id: str
    tenant_id: str
    user_id: str
    conversation_id: str
    session_id: str
    summary_text: str
    topic_hint: str = ""
    start_turn_index: int = 0
    end_turn_index: int = 0
    message_ids: List[str] = field(default_factory=list)
    importance: float = 0.5
    created_at: str = ""
    updated_at: str = ""


@dataclass
class MemoryCandidate:
    candidate_id: str
    candidate_kind: str
    score: float
    reason: str
    text: str
    scope: str = ""
    memory_type: str = ""
    updated_at: str = ""
    record: Optional[ManagedMemoryRecord] = None
    episode: Optional[MemoryEpisode] = None


@dataclass
class MemorySelection:
    selected_ids: List[str] = field(default_factory=list)
    brief: str = ""
    must_include: List[str] = field(default_factory=list)
    stale_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    mode: str = "deterministic"


@dataclass
class MemoryWriteOperation:
    operation: str
    scope: str
    memory_type: str
    title: str
    canonical_text: str
    key: str = ""
    structured_payload: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    confidence: float = 0.5
    evidence_turn_ids: List[str] = field(default_factory=list)
    supersedes_ids: List[str] = field(default_factory=list)
    ttl_hint: str = ""
    note: str = ""


@dataclass
class MemoryWriteResult:
    operations: List[MemoryWriteOperation] = field(default_factory=list)
    applied_count: int = 0
    skipped_count: int = 0
    shadow: bool = False
    errors: List[str] = field(default_factory=list)
    mode: str = ""


class MemoryStore(Protocol):
    def save_explicit(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        session_id: str,
        scope: str,
        key: str,
        value: str,
        source: str = "",
        evidence_turn_ids: Sequence[str] | None = None,
    ) -> ManagedMemoryRecord:
        ...

    def load_value(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        session_id: str,
        scope: str,
        key: str,
    ) -> str | None:
        ...

    def list_keys(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        session_id: str,
        scope: str,
    ) -> List[str]:
        ...

    def list_records(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        session_id: str,
        scope: str = "",
        active_only: bool = True,
        memory_types: Sequence[str] | None = None,
        limit: int = 50,
    ) -> List[ManagedMemoryRecord]:
        ...

    def search_candidates(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        session_id: str,
        query: str,
        scopes: Sequence[str],
        limit: int,
    ) -> List[MemoryCandidate]:
        ...

    def latest_episode_candidates(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        session_id: str,
        query: str,
        limit: int,
    ) -> List[MemoryCandidate]:
        ...

    def apply_operations(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        session_id: str,
        operations: Sequence[MemoryWriteOperation],
        source: str,
        shadow: bool,
    ) -> MemoryWriteResult:
        ...

    def upsert_episode(
        self,
        episode: MemoryEpisode,
        *,
        shadow: bool = False,
    ) -> MemoryEpisode | None:
        ...

    def touch_records(self, record_ids: Sequence[str]) -> None:
        ...

    def import_legacy_for_session(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        session_id: str,
        file_entries_by_scope: Dict[str, Sequence[tuple[str, str]]] | None = None,
    ) -> int:
        ...
