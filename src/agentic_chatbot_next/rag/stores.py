from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, List

from agentic_chatbot_next.authz.service import AuthorizationService
from agentic_chatbot_next.config import Settings
from agentic_chatbot_next.persistence.postgres.access import AccessControlStore
from agentic_chatbot_next.persistence.postgres.chunks import ChunkStore
from agentic_chatbot_next.persistence.postgres.collections import CollectionStore
from agentic_chatbot_next.persistence.postgres.connection import apply_schema, init_pool
from agentic_chatbot_next.persistence.postgres.documents import DocumentStore
from agentic_chatbot_next.persistence.postgres.entities import CanonicalEntityStore
from agentic_chatbot_next.persistence.postgres.graphs import (
    GraphIndexRunStore,
    GraphIndexSourceStore,
    GraphIndexStore,
    GraphQueryCacheStore,
)
from agentic_chatbot_next.persistence.postgres.memory_v2 import PostgresMemoryStore
from agentic_chatbot_next.persistence.postgres.mcp import McpConnectionStore
from agentic_chatbot_next.persistence.postgres.requirements import RequirementStatementStore
from agentic_chatbot_next.persistence.postgres.skills import SkillStore
from agentic_chatbot_next.rag.graph_store import build_graph_store


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def make_doc_id(
    source_type: str,
    source_identity: str,
    content_hash: str,
    tenant_id: str,
    *,
    collection_id: str = "",
) -> str:
    key = f"{tenant_id}:{collection_id}:{source_type}:{source_identity}:{content_hash}"
    return f"{source_type.upper()}_{_sha1(key)[:10]}"


@dataclass
class KnowledgeStores:
    chunk_store: ChunkStore
    doc_store: DocumentStore
    collection_store: CollectionStore | None
    memory_store: object | None
    requirement_store: RequirementStatementStore | None
    skill_store: SkillStore
    access_store: AccessControlStore | None = None
    authorization_service: AuthorizationService | None = None
    graph_store: object | None = None
    graph_index_store: GraphIndexStore | None = None
    graph_index_source_store: GraphIndexSourceStore | None = None
    graph_index_run_store: GraphIndexRunStore | None = None
    graph_query_cache_store: GraphQueryCacheStore | None = None
    entity_store: CanonicalEntityStore | None = None
    mcp_connection_store: McpConnectionStore | None = None


def load_stores(settings: Settings, embeddings: object) -> KnowledgeStores:
    if settings.database_backend != "postgres":
        raise NotImplementedError(
            f"DATABASE_BACKEND={settings.database_backend!r} is not implemented. Supported: postgres."
        )
    if settings.vector_store_backend != "pgvector":
        raise NotImplementedError(
            f"VECTOR_STORE_BACKEND={settings.vector_store_backend!r} is not implemented. Supported: pgvector."
        )

    init_pool(settings)
    apply_schema(settings)
    embed_fn: Callable[[str], List[float]] = lambda text: embeddings.embed_query(text)  # type: ignore[attr-defined]
    stores = KnowledgeStores(
        chunk_store=ChunkStore(embed_fn=embed_fn, embedding_dim=settings.embedding_dim),
        doc_store=DocumentStore(),
        collection_store=CollectionStore(),
        memory_store=PostgresMemoryStore(embed_fn=embed_fn, embedding_dim=settings.embedding_dim)
        if bool(getattr(settings, "memory_enabled", True))
        else None,
        requirement_store=RequirementStatementStore(),
        skill_store=SkillStore(embed_fn=embed_fn, embedding_dim=settings.embedding_dim),
        access_store=AccessControlStore(),
        graph_store=build_graph_store(settings),
        graph_index_store=GraphIndexStore(embed_fn=embed_fn, embedding_dim=settings.embedding_dim),
        graph_index_source_store=GraphIndexSourceStore(),
        graph_index_run_store=GraphIndexRunStore(),
        graph_query_cache_store=GraphQueryCacheStore(),
        entity_store=CanonicalEntityStore(),
        mcp_connection_store=McpConnectionStore(),
    )
    stores.authorization_service = AuthorizationService(settings, stores.access_store)
    return stores
