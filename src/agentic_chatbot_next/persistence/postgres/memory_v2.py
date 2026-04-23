from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import replace
from typing import Any, Callable, Dict, List, Sequence

import psycopg2.extras

from agentic_chatbot_next.memory.scope import MemoryScope
from agentic_chatbot_next.memory.store import (
    MEMORY_TYPES,
    ManagedMemoryRecord,
    MemoryCandidate,
    MemoryEpisode,
    MemoryObservation,
    MemorySelection,
    MemoryStore,
    MemoryWriteOperation,
    MemoryWriteResult,
)
from agentic_chatbot_next.persistence.postgres.connection import get_conn


def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_scope(value: str) -> str:
    try:
        return MemoryScope(str(value or "").strip() or MemoryScope.conversation.value).value
    except ValueError:
        return MemoryScope.conversation.value


def _normalize_memory_type(value: str, *, scope: str) -> str:
    clean = str(value or "").strip().lower()
    if clean in MEMORY_TYPES:
        return clean
    return "profile_preference" if scope == MemoryScope.user.value else "task_state"


def _record_from_row(row: Dict[str, Any]) -> ManagedMemoryRecord:
    return ManagedMemoryRecord(
        memory_id=str(row.get("memory_id") or ""),
        tenant_id=str(row.get("tenant_id") or "local-dev"),
        user_id=str(row.get("user_id") or ""),
        conversation_id=str(row.get("conversation_id") or ""),
        session_id=str(row.get("session_id") or ""),
        scope=str(row.get("scope") or MemoryScope.conversation.value),
        memory_type=str(row.get("memory_type") or "task_state"),
        key=str(row.get("memory_key") or ""),
        title=str(row.get("title") or ""),
        canonical_text=str(row.get("canonical_text") or ""),
        structured_payload=dict(row.get("structured_payload") or {}),
        importance=_coerce_float(row.get("importance"), 0.5),
        confidence=_coerce_float(row.get("confidence"), 0.5),
        active=bool(row.get("active", True)),
        superseded_by=str(row.get("superseded_by") or ""),
        provenance_turn_ids=[str(item) for item in (row.get("provenance_turn_ids") or []) if str(item)],
        last_used_at=str(row.get("last_used_at") or ""),
        created_at=str(row.get("created_at") or ""),
        updated_at=str(row.get("updated_at") or ""),
        source=str(row.get("source") or ""),
        ttl_hint=str(row.get("ttl_hint") or ""),
    )


def _episode_from_row(row: Dict[str, Any]) -> MemoryEpisode:
    return MemoryEpisode(
        episode_id=str(row.get("episode_id") or ""),
        tenant_id=str(row.get("tenant_id") or "local-dev"),
        user_id=str(row.get("user_id") or ""),
        conversation_id=str(row.get("conversation_id") or ""),
        session_id=str(row.get("session_id") or ""),
        summary_text=str(row.get("summary_text") or ""),
        topic_hint=str(row.get("topic_hint") or ""),
        start_turn_index=int(row.get("start_turn_index") or 0),
        end_turn_index=int(row.get("end_turn_index") or 0),
        message_ids=[str(item) for item in (row.get("message_ids") or []) if str(item)],
        importance=_coerce_float(row.get("importance"), 0.5),
        created_at=str(row.get("created_at") or ""),
        updated_at=str(row.get("updated_at") or ""),
    )


class PostgresMemoryStore(MemoryStore):
    def __init__(self, embed_fn: Callable[[str], List[float]], embedding_dim: int = 768) -> None:
        self._embed = embed_fn
        self.embedding_dim = embedding_dim
        self._imported_sessions: set[tuple[str, str, str, str]] = set()

    def _latest_active_by_key(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        session_id: str,
        scope: str,
        key: str,
    ) -> ManagedMemoryRecord | None:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM memory_records
                    WHERE tenant_id = %s
                      AND user_id = %s
                      AND conversation_id = %s
                      AND session_id = %s
                      AND scope = %s
                      AND memory_key = %s
                      AND active = TRUE
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (tenant_id, user_id, conversation_id, session_id, scope, key),
                )
                row = cur.fetchone()
        return _record_from_row(dict(row)) if row is not None else None

    def _insert_record(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        session_id: str,
        scope: str,
        memory_type: str,
        key: str,
        title: str,
        canonical_text: str,
        structured_payload: Dict[str, Any],
        importance: float,
        confidence: float,
        provenance_turn_ids: Sequence[str],
        source: str,
        ttl_hint: str,
    ) -> ManagedMemoryRecord:
        memory_id = _new_id("mem")
        now = _now_iso()
        embedding = self._embed(canonical_text or title or key)
        with get_conn() as conn:
            from pgvector.psycopg2 import register_vector

            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_records
                        (memory_id, tenant_id, user_id, conversation_id, session_id, scope, memory_type,
                         memory_key, title, canonical_text, structured_payload, importance, confidence,
                         active, superseded_by, provenance_turn_ids, last_used_at, created_at, updated_at,
                         source, ttl_hint, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE, '', %s, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    (
                        memory_id,
                        tenant_id,
                        user_id,
                        conversation_id,
                        session_id,
                        scope,
                        memory_type,
                        key,
                        title,
                        canonical_text,
                        psycopg2.extras.Json(structured_payload or {}),
                        max(0.0, min(float(importance), 1.0)),
                        max(0.0, min(float(confidence), 1.0)),
                        list(provenance_turn_ids or []),
                        now,
                        now,
                        now,
                        source,
                        ttl_hint,
                        embedding,
                    ),
                )
            conn.commit()
        return ManagedMemoryRecord(
            memory_id=memory_id,
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            session_id=session_id,
            scope=scope,
            memory_type=memory_type,
            key=key,
            title=title,
            canonical_text=canonical_text,
            structured_payload=dict(structured_payload or {}),
            importance=max(0.0, min(float(importance), 1.0)),
            confidence=max(0.0, min(float(confidence), 1.0)),
            active=True,
            superseded_by="",
            provenance_turn_ids=[str(item) for item in provenance_turn_ids or [] if str(item)],
            last_used_at=now,
            created_at=now,
            updated_at=now,
            source=source,
            ttl_hint=ttl_hint,
        )

    def _supersede(self, *, memory_ids: Sequence[str], replacement_id: str) -> None:
        clean_ids = [str(item) for item in memory_ids if str(item)]
        if not clean_ids:
            return
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE memory_records
                    SET active = FALSE,
                        superseded_by = %s,
                        updated_at = now()
                    WHERE memory_id = ANY(%s)
                    """,
                    (replacement_id, clean_ids),
                )
            conn.commit()

    def _add_observation(
        self,
        *,
        memory_id: str,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        session_id: str,
        operation: str,
        evidence_turn_ids: Sequence[str],
        note: str,
        raw_payload: Dict[str, Any],
        confidence: float,
    ) -> None:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_observations
                        (observation_id, memory_id, tenant_id, user_id, conversation_id, session_id,
                         operation, evidence_turn_ids, note, raw_payload, confidence, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    """,
                    (
                        _new_id("obs"),
                        memory_id,
                        tenant_id,
                        user_id,
                        conversation_id,
                        session_id,
                        operation,
                        list(evidence_turn_ids or []),
                        note,
                        psycopg2.extras.Json(raw_payload or {}),
                        max(0.0, min(float(confidence), 1.0)),
                    ),
                )
            conn.commit()

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
        clean_scope = _normalize_scope(scope)
        clean_key = str(key or "").strip()
        current = self._latest_active_by_key(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            session_id=session_id,
            scope=clean_scope,
            key=clean_key,
        )
        if current is not None and current.canonical_text == str(value or ""):
            self.touch_records([current.memory_id])
            self._add_observation(
                memory_id=current.memory_id,
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                session_id=session_id,
                operation="reinforce",
                evidence_turn_ids=evidence_turn_ids or [],
                note="Explicit tool save matched existing memory.",
                raw_payload={"key": clean_key, "value": str(value or ""), "source": source or "tool"},
                confidence=1.0,
            )
            return current
        record = self._insert_record(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            session_id=session_id,
            scope=clean_scope,
            memory_type="profile_preference" if clean_scope == MemoryScope.user.value else "task_state",
            key=clean_key,
            title=clean_key.replace("_", " ").strip() or clean_key,
            canonical_text=str(value or ""),
            structured_payload={},
            importance=0.95 if clean_scope == MemoryScope.user.value else 0.85,
            confidence=1.0,
            provenance_turn_ids=evidence_turn_ids or [],
            source=source or "tool",
            ttl_hint="",
        )
        if current is not None:
            self._supersede(memory_ids=[current.memory_id], replacement_id=record.memory_id)
        self._add_observation(
            memory_id=record.memory_id,
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            session_id=session_id,
            operation="create" if current is None else "update",
            evidence_turn_ids=evidence_turn_ids or [],
            note="Explicit tool memory save.",
            raw_payload={"key": clean_key, "value": str(value or ""), "source": source or "tool"},
            confidence=1.0,
        )
        return record

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
        current = self._latest_active_by_key(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            session_id=session_id,
            scope=_normalize_scope(scope),
            key=str(key or "").strip(),
        )
        if current is None:
            return None
        self.touch_records([current.memory_id])
        return current.canonical_text

    def list_keys(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        session_id: str,
        scope: str,
    ) -> List[str]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT memory_key
                    FROM memory_records
                    WHERE tenant_id = %s
                      AND user_id = %s
                      AND conversation_id = %s
                      AND session_id = %s
                      AND scope = %s
                      AND active = TRUE
                      AND memory_key <> ''
                    ORDER BY memory_key ASC
                    """,
                    (tenant_id, user_id, conversation_id, session_id, _normalize_scope(scope)),
                )
                rows = cur.fetchall()
        return [str(row[0]) for row in rows if row and str(row[0])]

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
        sql = [
            "SELECT * FROM memory_records WHERE tenant_id = %s AND user_id = %s AND conversation_id = %s AND session_id = %s"
        ]
        params: List[Any] = [tenant_id, user_id, conversation_id, session_id]
        if scope:
            sql.append("AND scope = %s")
            params.append(_normalize_scope(scope))
        if active_only:
            sql.append("AND active = TRUE")
        clean_types = [str(item).strip() for item in (memory_types or []) if str(item).strip()]
        if clean_types:
            sql.append("AND memory_type = ANY(%s)")
            params.append(clean_types)
        sql.append("ORDER BY importance DESC, updated_at DESC LIMIT %s")
        params.append(max(1, int(limit)))
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(" ".join(sql), params)
                rows = cur.fetchall()
        return [_record_from_row(dict(row)) for row in rows]

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
        clean_query = str(query or "").strip()
        clean_scopes = [_normalize_scope(item) for item in scopes if str(item)]
        if not clean_query or not clean_scopes:
            return []
        candidates: List[MemoryCandidate] = []
        seen: set[str] = set()
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT *,
                           ts_rank_cd(search_vector, plainto_tsquery('english', %s)) AS kw_score
                    FROM memory_records
                    WHERE tenant_id = %s
                      AND user_id = %s
                      AND conversation_id = %s
                      AND session_id = %s
                      AND scope = ANY(%s)
                      AND active = TRUE
                      AND search_vector @@ plainto_tsquery('english', %s)
                    ORDER BY kw_score DESC, importance DESC, updated_at DESC
                    LIMIT %s
                    """,
                    (clean_query, tenant_id, user_id, conversation_id, session_id, clean_scopes, clean_query, max(1, int(limit))),
                )
                for row in cur.fetchall():
                    record = _record_from_row(dict(row))
                    if record.memory_id in seen:
                        continue
                    seen.add(record.memory_id)
                    candidates.append(
                        MemoryCandidate(
                            candidate_id=record.memory_id,
                            candidate_kind="record",
                            score=_coerce_float(row.get("kw_score"), 0.0),
                            reason="keyword_match",
                            text=record.canonical_text,
                            scope=record.scope,
                            memory_type=record.memory_type,
                            updated_at=record.updated_at,
                            record=record,
                        )
                    )
        if clean_query:
            embedding = self._embed(clean_query)
            with get_conn() as conn:
                from pgvector.psycopg2 import register_vector

                register_vector(conn)
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT *,
                               1 - (embedding <=> %s::vector) AS vector_score
                        FROM memory_records
                        WHERE tenant_id = %s
                          AND user_id = %s
                          AND conversation_id = %s
                          AND session_id = %s
                          AND scope = ANY(%s)
                          AND active = TRUE
                          AND embedding IS NOT NULL
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (
                            embedding,
                            tenant_id,
                            user_id,
                            conversation_id,
                            session_id,
                            clean_scopes,
                            embedding,
                            max(1, int(limit)),
                        ),
                    )
                    for row in cur.fetchall():
                        record = _record_from_row(dict(row))
                        if record.memory_id in seen:
                            continue
                        seen.add(record.memory_id)
                        candidates.append(
                            MemoryCandidate(
                                candidate_id=record.memory_id,
                                candidate_kind="record",
                                score=_coerce_float(row.get("vector_score"), 0.0),
                                reason="vector_match",
                                text=record.canonical_text,
                                scope=record.scope,
                                memory_type=record.memory_type,
                                updated_at=record.updated_at,
                                record=record,
                            )
                        )
        candidates.sort(key=lambda item: (item.score, item.record.importance if item.record is not None else 0.0), reverse=True)
        return candidates[: max(1, int(limit))]

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
        clean_query = str(query or "").strip()
        rows: List[dict[str, Any]] = []
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if clean_query:
                    cur.execute(
                        """
                        SELECT *,
                               ts_rank_cd(search_vector, plainto_tsquery('english', %s)) AS kw_score
                        FROM memory_episodes
                        WHERE tenant_id = %s
                          AND user_id = %s
                          AND conversation_id = %s
                          AND session_id = %s
                          AND search_vector @@ plainto_tsquery('english', %s)
                        ORDER BY kw_score DESC, importance DESC, updated_at DESC
                        LIMIT %s
                        """,
                        (clean_query, tenant_id, user_id, conversation_id, session_id, clean_query, max(1, int(limit))),
                    )
                    rows.extend(dict(row) for row in cur.fetchall())
                cur.execute(
                    """
                    SELECT *
                    FROM memory_episodes
                    WHERE tenant_id = %s
                      AND user_id = %s
                      AND conversation_id = %s
                      AND session_id = %s
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (tenant_id, user_id, conversation_id, session_id, max(1, int(limit))),
                )
                rows.extend(dict(row) for row in cur.fetchall())
        candidates: List[MemoryCandidate] = []
        seen: set[str] = set()
        for row in rows:
            episode = _episode_from_row(row)
            if episode.episode_id in seen:
                continue
            seen.add(episode.episode_id)
            candidates.append(
                MemoryCandidate(
                    candidate_id=episode.episode_id,
                    candidate_kind="episode",
                    score=max(_coerce_float(row.get("kw_score"), 0.0), episode.importance),
                    reason="episode_summary",
                    text=episode.summary_text,
                    updated_at=episode.updated_at,
                    episode=episode,
                )
            )
        return candidates[: max(1, int(limit))]

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
        result = MemoryWriteResult(shadow=bool(shadow), mode="shadow" if shadow else "live")
        if shadow:
            result.operations = list(operations or [])
            result.skipped_count = len(result.operations)
            return result
        for op in operations or []:
            if str(op.operation or "").strip().lower() == "ignore":
                result.skipped_count += 1
                continue
            replacement = self._insert_record(
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                session_id=session_id,
                scope=_normalize_scope(op.scope),
                memory_type=_normalize_memory_type(op.memory_type, scope=_normalize_scope(op.scope)),
                key=str(op.key or "").strip(),
                title=str(op.title or "").strip() or str(op.key or "").strip(),
                canonical_text=str(op.canonical_text or "").strip(),
                structured_payload=dict(op.structured_payload or {}),
                importance=max(0.0, min(op.importance, 1.0)),
                confidence=max(0.0, min(op.confidence, 1.0)),
                provenance_turn_ids=list(op.evidence_turn_ids or []),
                source=source,
                ttl_hint=str(op.ttl_hint or ""),
            )
            if op.operation in {"update", "supersede"} and op.supersedes_ids:
                self._supersede(memory_ids=op.supersedes_ids, replacement_id=replacement.memory_id)
            elif op.operation == "reinforce":
                self._add_observation(
                    memory_id=replacement.memory_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    session_id=session_id,
                    operation="reinforce",
                    evidence_turn_ids=op.evidence_turn_ids,
                    note=str(op.note or ""),
                    raw_payload={"operation": op.operation, "title": op.title, "canonical_text": op.canonical_text},
                    confidence=op.confidence,
                )
            self._add_observation(
                memory_id=replacement.memory_id,
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                session_id=session_id,
                operation=op.operation,
                evidence_turn_ids=op.evidence_turn_ids,
                note=str(op.note or ""),
                raw_payload={
                    "scope": op.scope,
                    "memory_type": op.memory_type,
                    "title": op.title,
                    "canonical_text": op.canonical_text,
                    "structured_payload": dict(op.structured_payload or {}),
                    "supersedes_ids": list(op.supersedes_ids or []),
                },
                confidence=op.confidence,
            )
            result.operations.append(op)
            result.applied_count += 1
        return result

    def upsert_episode(
        self,
        episode: MemoryEpisode,
        *,
        shadow: bool = False,
    ) -> MemoryEpisode | None:
        if shadow:
            return None
        effective = replace(
            episode,
            episode_id=episode.episode_id or _new_id("ep"),
            created_at=episode.created_at or _now_iso(),
            updated_at=_now_iso(),
        )
        embedding = self._embed(effective.summary_text or effective.topic_hint or "memory episode")
        with get_conn() as conn:
            from pgvector.psycopg2 import register_vector

            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_episodes
                        (episode_id, tenant_id, user_id, conversation_id, session_id, summary_text,
                         topic_hint, start_turn_index, end_turn_index, message_ids, importance,
                         created_at, updated_at, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (episode_id) DO UPDATE SET
                        summary_text = EXCLUDED.summary_text,
                        topic_hint = EXCLUDED.topic_hint,
                        start_turn_index = EXCLUDED.start_turn_index,
                        end_turn_index = EXCLUDED.end_turn_index,
                        message_ids = EXCLUDED.message_ids,
                        importance = EXCLUDED.importance,
                        updated_at = EXCLUDED.updated_at,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        effective.episode_id,
                        effective.tenant_id,
                        effective.user_id,
                        effective.conversation_id,
                        effective.session_id,
                        effective.summary_text,
                        effective.topic_hint,
                        int(effective.start_turn_index),
                        int(effective.end_turn_index),
                        list(effective.message_ids or []),
                        max(0.0, min(effective.importance, 1.0)),
                        effective.created_at,
                        effective.updated_at,
                        embedding,
                    ),
                )
            conn.commit()
        return effective

    def touch_records(self, record_ids: Sequence[str]) -> None:
        clean_ids = [str(item) for item in record_ids if str(item)]
        if not clean_ids:
            return
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE memory_records
                    SET last_used_at = now()
                    WHERE memory_id = ANY(%s)
                    """,
                    (clean_ids,),
                )
            conn.commit()

    def import_legacy_for_session(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        session_id: str,
        file_entries_by_scope: Dict[str, Sequence[tuple[str, str]]] | None = None,
    ) -> int:
        cache_key = (tenant_id, user_id, conversation_id, session_id)
        if cache_key in self._imported_sessions:
            return 0
        imported = 0
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT key, value
                    FROM memory
                    WHERE tenant_id = %s
                      AND session_id = %s
                    ORDER BY updated_at DESC, created_at DESC
                    """,
                    (tenant_id, session_id),
                )
                rows = cur.fetchall()
        for row in rows:
            key = str(row.get("key") or "").strip()
            value = str(row.get("value") or "").strip()
            if not key or not value:
                continue
            if self.load_value(
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                session_id=session_id,
                scope=MemoryScope.conversation.value,
                key=key,
            ) is not None:
                continue
            self.save_explicit(
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                session_id=session_id,
                scope=MemoryScope.conversation.value,
                key=key,
                value=value,
                source="legacy_table_import",
            )
            imported += 1
        for scope, entries in dict(file_entries_by_scope or {}).items():
            for key, value in entries:
                clean_key = str(key or "").strip()
                clean_value = str(value or "").strip()
                if not clean_key or not clean_value:
                    continue
                if self.load_value(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    session_id=session_id,
                    scope=scope,
                    key=clean_key,
                ) is not None:
                    continue
                self.save_explicit(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    session_id=session_id,
                    scope=scope,
                    key=clean_key,
                    value=clean_value,
                    source="legacy_file_import",
                )
                imported += 1
        self._imported_sessions.add(cache_key)
        return imported
