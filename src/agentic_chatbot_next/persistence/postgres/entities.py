from __future__ import annotations

import datetime as dt
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

import psycopg2.extras

from agentic_chatbot_next.persistence.postgres.connection import get_conn


def normalize_entity_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-z0-9]+", " ", ascii_text.casefold())
    return " ".join(cleaned.split())


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


@dataclass
class CanonicalEntityRecord:
    entity_id: str
    canonical_name: str
    tenant_id: str = "local-dev"
    collection_id: str = "default"
    normalized_name: str = ""
    entity_type: str = ""
    description: str = ""
    graph_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class EntityAliasRecord:
    alias_id: str
    entity_id: str
    alias: str
    tenant_id: str = "local-dev"
    collection_id: str = "default"
    normalized_alias: str = ""
    source: str = "graph"
    created_at: str = ""
    updated_at: str = ""


@dataclass
class EntityMentionRecord:
    mention_id: str
    entity_id: str
    tenant_id: str = "local-dev"
    collection_id: str = "default"
    doc_id: str = ""
    chunk_id: str = ""
    graph_id: str = ""
    mention_text: str = ""
    mention_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


def make_entity_id(*, tenant_id: str, collection_id: str, canonical_name: str) -> str:
    normalized = normalize_entity_text(canonical_name) or canonical_name.strip().casefold()
    seed = f"{tenant_id}:{collection_id}:{normalized}"
    return f"ent_{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex[:24]}"


def make_alias_id(*, entity_id: str, alias: str) -> str:
    seed = f"{entity_id}:{normalize_entity_text(alias)}"
    return f"alias_{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex[:24]}"


def make_mention_id(
    *,
    entity_id: str,
    doc_id: str,
    chunk_id: str,
    graph_id: str,
    mention_text: str,
) -> str:
    seed = f"{entity_id}:{doc_id}:{chunk_id}:{graph_id}:{normalize_entity_text(mention_text)}"
    return f"mention_{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex[:24]}"


class CanonicalEntityStore:
    def upsert_entities(
        self,
        *,
        entities: List[CanonicalEntityRecord],
        aliases: List[EntityAliasRecord] | None = None,
        mentions: List[EntityMentionRecord] | None = None,
        replace_graph_scope: str = "",
        tenant_id: str = "local-dev",
    ) -> None:
        entity_rows = list(entities or [])
        alias_rows = list(aliases or [])
        mention_rows = list(mentions or [])
        now = _now_iso()
        with get_conn() as conn:
            with conn.cursor() as cur:
                for entity in entity_rows:
                    normalized_name = entity.normalized_name or normalize_entity_text(entity.canonical_name)
                    cur.execute(
                        """
                        INSERT INTO canonical_entities
                            (entity_id, tenant_id, collection_id, canonical_name, normalized_name, entity_type, description, graph_id, metadata, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (entity_id) DO UPDATE SET
                            canonical_name = EXCLUDED.canonical_name,
                            normalized_name = EXCLUDED.normalized_name,
                            entity_type = EXCLUDED.entity_type,
                            description = EXCLUDED.description,
                            graph_id = EXCLUDED.graph_id,
                            metadata = EXCLUDED.metadata,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (
                            entity.entity_id,
                            entity.tenant_id,
                            entity.collection_id,
                            entity.canonical_name,
                            normalized_name,
                            entity.entity_type,
                            entity.description,
                            entity.graph_id,
                            psycopg2.extras.Json(dict(entity.metadata or {})),
                            entity.created_at or now,
                            entity.updated_at or now,
                        ),
                    )
                for alias in alias_rows:
                    normalized_alias = alias.normalized_alias or normalize_entity_text(alias.alias)
                    cur.execute(
                        """
                        INSERT INTO entity_aliases
                            (alias_id, entity_id, tenant_id, collection_id, alias, normalized_alias, source, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (alias_id) DO UPDATE SET
                            alias = EXCLUDED.alias,
                            normalized_alias = EXCLUDED.normalized_alias,
                            source = EXCLUDED.source,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (
                            alias.alias_id,
                            alias.entity_id,
                            alias.tenant_id,
                            alias.collection_id,
                            alias.alias,
                            normalized_alias,
                            alias.source,
                            alias.created_at or now,
                            alias.updated_at or now,
                        ),
                    )
                if replace_graph_scope:
                    cur.execute(
                        "DELETE FROM entity_mentions WHERE tenant_id = %s AND graph_id = %s",
                        (tenant_id, replace_graph_scope),
                    )
                for mention in mention_rows:
                    cur.execute(
                        """
                        INSERT INTO entity_mentions
                            (mention_id, entity_id, tenant_id, collection_id, doc_id, chunk_id, graph_id, mention_text, mention_type, metadata, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (mention_id) DO UPDATE SET
                            doc_id = EXCLUDED.doc_id,
                            chunk_id = EXCLUDED.chunk_id,
                            graph_id = EXCLUDED.graph_id,
                            mention_text = EXCLUDED.mention_text,
                            mention_type = EXCLUDED.mention_type,
                            metadata = EXCLUDED.metadata,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (
                            mention.mention_id,
                            mention.entity_id,
                            mention.tenant_id,
                            mention.collection_id,
                            mention.doc_id,
                            mention.chunk_id,
                            mention.graph_id,
                            mention.mention_text,
                            mention.mention_type,
                            psycopg2.extras.Json(dict(mention.metadata or {})),
                            mention.created_at or now,
                            mention.updated_at or now,
                        ),
                    )
            conn.commit()

    def resolve_aliases(
        self,
        query: str,
        *,
        tenant_id: str = "local-dev",
        collection_id: str = "",
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        normalized_query = normalize_entity_text(query)
        if not normalized_query:
            return []
        query_terms = [term for term in normalized_query.split() if len(term) > 2]
        if not query_terms:
            return []

        sql = [
            """
            SELECT
                ce.entity_id,
                ce.canonical_name,
                ce.normalized_name,
                ce.entity_type,
                ce.description,
                ce.graph_id,
                ea.alias,
                ea.normalized_alias
            FROM entity_aliases ea
            JOIN canonical_entities ce ON ce.entity_id = ea.entity_id
            WHERE ea.tenant_id = %s
            """
        ]
        params: List[Any] = [tenant_id]
        if collection_id:
            sql.append("AND ea.collection_id = %s")
            params.append(collection_id)
        sql.append("ORDER BY ce.canonical_name ASC")

        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(" ".join(sql), params)
                rows = cur.fetchall()

        matches: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            normalized_alias = str(row.get("normalized_alias") or "")
            normalized_name = str(row.get("normalized_name") or "")
            overlap = len({term for term in query_terms if term in normalized_alias or term in normalized_name})
            if normalized_alias and normalized_alias in normalized_query:
                overlap += 2
            if normalized_name and normalized_name in normalized_query:
                overlap += 2
            if overlap <= 0:
                continue
            entity_id = str(row.get("entity_id") or "")
            if not entity_id or entity_id in seen:
                continue
            seen.add(entity_id)
            matches.append(
                {
                    "entity_id": entity_id,
                    "canonical_name": str(row.get("canonical_name") or ""),
                    "entity_type": str(row.get("entity_type") or ""),
                    "description": str(row.get("description") or ""),
                    "graph_id": str(row.get("graph_id") or ""),
                    "matched_alias": str(row.get("alias") or ""),
                    "score": overlap,
                }
            )
        matches.sort(key=lambda item: (float(item.get("score") or 0.0), item.get("canonical_name") or ""), reverse=True)
        return matches[: max(1, int(limit))]

    def mentions_for_entity_ids(
        self,
        entity_ids: List[str],
        *,
        tenant_id: str = "local-dev",
        collection_id: str = "",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        scoped_ids = [str(item) for item in entity_ids if str(item)]
        if not scoped_ids:
            return []
        sql = [
            """
            SELECT *
            FROM entity_mentions
            WHERE tenant_id = %s
              AND entity_id = ANY(%s)
            """
        ]
        params: List[Any] = [tenant_id, scoped_ids]
        if collection_id:
            sql.append("AND collection_id = %s")
            params.append(collection_id)
        sql.append("ORDER BY updated_at DESC, mention_text ASC LIMIT %s")
        params.append(max(1, int(limit)))
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(" ".join(sql), params)
                rows = cur.fetchall()
        return [dict(row) for row in rows]


__all__ = [
    "CanonicalEntityRecord",
    "CanonicalEntityStore",
    "EntityAliasRecord",
    "EntityMentionRecord",
    "make_alias_id",
    "make_entity_id",
    "make_mention_id",
    "normalize_entity_text",
]
