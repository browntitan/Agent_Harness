from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

from agentic_chatbot_next.persistence.postgres.chunks import ChunkRecord
from agentic_chatbot_next.persistence.postgres.documents import DocumentRecord

logger = logging.getLogger(__name__)


@dataclass
class GraphSearchHit:
    doc_id: str
    chunk_ids: List[str] = field(default_factory=list)
    score: float = 0.0
    title: str = ""
    source_path: str = ""
    source_type: str = ""
    relationship_path: List[str] = field(default_factory=list)
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def _graph_terms(text: str) -> List[str]:
    seen: set[str] = set()
    terms: List[str] = []
    for token in re.findall(r"[A-Za-z0-9_-]{3,}", str(text or "").lower()):
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= 12:
            break
    return terms


def _extract_entities(text: str) -> List[str]:
    seen: set[str] = set()
    entities: List[str] = []
    for entity in re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", str(text or "")):
        if entity in seen:
            continue
        seen.add(entity)
        entities.append(entity)
        if len(entities) >= 16:
            break
    return entities


class Neo4jGraphStore:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.enabled = bool(getattr(settings, "graph_search_enabled", False) or getattr(settings, "graph_ingest_enabled", False))
        self.uri = str(getattr(settings, "neo4j_uri", "") or "")
        self.username = str(getattr(settings, "neo4j_username", "") or "")
        self.password = str(getattr(settings, "neo4j_password", "") or "")
        self.database = str(getattr(settings, "neo4j_database", "neo4j") or "neo4j")
        self.timeout_seconds = int(getattr(settings, "neo4j_timeout_seconds", 15) or 15)
        self._graph: Any | None = None
        self._driver: Any | None = None
        self._available = False
        if self.enabled and self.uri and self.username and self.password:
            self._init_backend()

    @property
    def available(self) -> bool:
        return bool(self._available)

    def _init_backend(self) -> None:
        try:
            from langchain_neo4j import Neo4jGraph  # type: ignore

            self._graph = Neo4jGraph(
                url=self.uri,
                username=self.username,
                password=self.password,
                database=self.database,
            )
            self._available = True
            return
        except Exception as exc:
            logger.debug("LangChain Neo4jGraph not available, falling back to Neo4j driver: %s", exc)
        try:
            from neo4j import GraphDatabase  # type: ignore

            self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            self._available = True
        except Exception as exc:
            logger.warning("Could not initialize Neo4j graph store: %s", exc)
            self._available = False

    def _query(self, cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self._graph is not None:
            rows = self._graph.query(cypher, params=params)
            return [dict(row) for row in rows or []]
        if self._driver is None:
            return []
        with self._driver.session(database=self.database) as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]

    def healthcheck(self) -> bool:
        if not self.available:
            return False
        try:
            self._query("RETURN 1 AS ok", {})
            return True
        except Exception as exc:
            logger.warning("Neo4j graph healthcheck failed: %s", exc)
            return False

    def ingest_document(
        self,
        document: DocumentRecord,
        chunks: Sequence[ChunkRecord],
        *,
        tenant_id: str,
    ) -> None:
        if not self.available or not getattr(self.settings, "graph_ingest_enabled", False):
            return
        try:
            self._query(
                """
                MERGE (d:Document {doc_id: $doc_id, tenant_id: $tenant_id})
                SET d.title = $title,
                    d.source_path = $source_path,
                    d.source_type = $source_type,
                    d.collection_id = $collection_id,
                    d.doc_structure_type = $doc_structure_type
                """,
                {
                    "doc_id": document.doc_id,
                    "tenant_id": tenant_id,
                    "title": document.title,
                    "source_path": document.source_path,
                    "source_type": document.source_type,
                    "collection_id": document.collection_id,
                    "doc_structure_type": document.doc_structure_type,
                },
            )
            for chunk in chunks:
                self._query(
                    """
                    MERGE (c:Chunk {chunk_id: $chunk_id, tenant_id: $tenant_id})
                    SET c.doc_id = $doc_id,
                        c.text = $text,
                        c.chunk_type = $chunk_type,
                        c.section_title = $section_title,
                        c.clause_number = $clause_number
                    WITH c
                    MATCH (d:Document {doc_id: $doc_id, tenant_id: $tenant_id})
                    MERGE (d)-[:HAS_CHUNK]->(c)
                    """,
                    {
                        "chunk_id": chunk.chunk_id,
                        "tenant_id": tenant_id,
                        "doc_id": chunk.doc_id,
                        "text": chunk.content[:6000],
                        "chunk_type": chunk.chunk_type,
                        "section_title": chunk.section_title or "",
                        "clause_number": chunk.clause_number or "",
                    },
                )
                entities = _extract_entities(" ".join([document.title, chunk.section_title or "", chunk.content[:1000]]))
                for entity in entities:
                    self._query(
                        """
                        MERGE (e:Entity {name: $entity, tenant_id: $tenant_id})
                        WITH e
                        MATCH (c:Chunk {chunk_id: $chunk_id, tenant_id: $tenant_id})
                        MERGE (c)-[:MENTIONS {source_chunk_id: $chunk_id}]->(e)
                        """,
                        {
                            "entity": entity,
                            "tenant_id": tenant_id,
                            "chunk_id": chunk.chunk_id,
                        },
                    )
        except Exception as exc:
            logger.warning("Neo4j graph ingest failed for %s: %s", document.doc_id, exc)

    def local_search(
        self,
        query: str,
        *,
        tenant_id: str,
        limit: int = 8,
        doc_ids: Sequence[str] | None = None,
    ) -> List[GraphSearchHit]:
        if not self.available or not getattr(self.settings, "graph_search_enabled", False):
            return []
        terms = _graph_terms(query)
        if not terms:
            return []
        rows = self._query(
            """
            MATCH (d:Document {tenant_id: $tenant_id})-[:HAS_CHUNK]->(c:Chunk {tenant_id: $tenant_id})
            OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity {tenant_id: $tenant_id})
            WHERE ($doc_ids_empty OR d.doc_id IN $doc_ids)
              AND (
                ANY(term IN $terms WHERE toLower(d.title) CONTAINS term)
                OR ANY(term IN $terms WHERE toLower(coalesce(c.text, '')) CONTAINS term)
                OR ANY(term IN $terms WHERE toLower(coalesce(e.name, '')) CONTAINS term)
              )
            WITH d, collect(DISTINCT c.chunk_id)[0..8] AS chunk_ids,
                 collect(DISTINCT e.name)[0..8] AS entities,
                 count(DISTINCT e) + count(DISTINCT c) AS raw_score
            RETURN d.doc_id AS doc_id,
                   d.title AS title,
                   d.source_path AS source_path,
                   d.source_type AS source_type,
                   chunk_ids,
                   entities,
                   raw_score
            ORDER BY raw_score DESC, title ASC
            LIMIT $limit
            """,
            {
                "tenant_id": tenant_id,
                "terms": terms,
                "doc_ids": list(doc_ids or []),
                "doc_ids_empty": not bool(doc_ids),
                "limit": max(1, int(limit)),
            },
        )
        return [
            GraphSearchHit(
                doc_id=str(row.get("doc_id") or ""),
                title=str(row.get("title") or ""),
                source_path=str(row.get("source_path") or ""),
                source_type=str(row.get("source_type") or ""),
                chunk_ids=[str(item) for item in (row.get("chunk_ids") or []) if str(item)],
                relationship_path=[str(item) for item in (row.get("entities") or []) if str(item)],
                summary=f"Matched graph entities: {', '.join(str(item) for item in (row.get('entities') or [])[:4])}",
                score=float(row.get("raw_score") or 0.0),
            )
            for row in rows
            if str(row.get("doc_id") or "")
        ]

    def global_search(
        self,
        query: str,
        *,
        tenant_id: str,
        limit: int = 8,
        doc_ids: Sequence[str] | None = None,
    ) -> List[GraphSearchHit]:
        if not self.available or not getattr(self.settings, "graph_search_enabled", False):
            return []
        terms = _graph_terms(query)
        if not terms:
            return []
        rows = self._query(
            """
            MATCH (d:Document {tenant_id: $tenant_id})-[:HAS_CHUNK]->(:Chunk {tenant_id: $tenant_id})-[:MENTIONS]->(e:Entity {tenant_id: $tenant_id})
            WHERE ($doc_ids_empty OR d.doc_id IN $doc_ids)
              AND ANY(term IN $terms WHERE toLower(e.name) CONTAINS term OR toLower(d.title) CONTAINS term)
            WITH d, collect(DISTINCT e.name)[0..8] AS entities, count(DISTINCT e) AS raw_score
            RETURN d.doc_id AS doc_id,
                   d.title AS title,
                   d.source_path AS source_path,
                   d.source_type AS source_type,
                   entities,
                   raw_score
            ORDER BY raw_score DESC, title ASC
            LIMIT $limit
            """,
            {
                "tenant_id": tenant_id,
                "terms": terms,
                "doc_ids": list(doc_ids or []),
                "doc_ids_empty": not bool(doc_ids),
                "limit": max(1, int(limit)),
            },
        )
        return [
            GraphSearchHit(
                doc_id=str(row.get("doc_id") or ""),
                title=str(row.get("title") or ""),
                source_path=str(row.get("source_path") or ""),
                source_type=str(row.get("source_type") or ""),
                relationship_path=[str(item) for item in (row.get("entities") or []) if str(item)],
                summary=f"Cross-document graph match: {', '.join(str(item) for item in (row.get('entities') or [])[:4])}",
                score=float(row.get("raw_score") or 0.0),
            )
            for row in rows
            if str(row.get("doc_id") or "")
        ]


def build_graph_store(settings: Any) -> Neo4jGraphStore | None:
    store = Neo4jGraphStore(settings)
    return store if store.available else None


__all__ = ["GraphSearchHit", "Neo4jGraphStore", "build_graph_store"]
