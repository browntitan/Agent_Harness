-- ============================================================
-- Agentic RAG Chatbot — PostgreSQL Schema
-- Run once: psql -d ragdb -f schema.sql
-- Requires: pgvector >= 0.5 (HNSW), pg_trgm
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ------------------------------------------------------------
-- collections: persistent admin-managed collection catalog
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS collections (
    tenant_id      TEXT NOT NULL DEFAULT 'local-dev',
    collection_id  TEXT NOT NULL,
    maintenance_policy TEXT NOT NULL DEFAULT 'indexed_documents',
    created_at     TIMESTAMPTZ DEFAULT now(),
    updated_at     TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (tenant_id, collection_id)
);

CREATE INDEX IF NOT EXISTS collections_updated_idx
    ON collections(tenant_id, updated_at DESC, collection_id);

-- ------------------------------------------------------------
-- auth_principals: user and group identities for RBAC
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS auth_principals (
    principal_id      TEXT PRIMARY KEY,
    tenant_id         TEXT NOT NULL DEFAULT 'local-dev',
    principal_type    TEXT NOT NULL DEFAULT 'user',
    provider          TEXT NOT NULL DEFAULT 'email',
    external_id       TEXT DEFAULT '',
    email_normalized  TEXT DEFAULT '',
    display_name      TEXT DEFAULT '',
    metadata_json     JSONB NOT NULL DEFAULT '{}'::jsonb,
    active            BOOLEAN NOT NULL DEFAULT TRUE,
    created_at        TIMESTAMPTZ DEFAULT now(),
    updated_at        TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS auth_principals_email_idx
    ON auth_principals(tenant_id, provider, email_normalized)
    WHERE email_normalized IS NOT NULL AND email_normalized <> '';

CREATE INDEX IF NOT EXISTS auth_principals_lookup_idx
    ON auth_principals(tenant_id, principal_type, display_name, principal_id);

-- ------------------------------------------------------------
-- auth_principal_memberships: group nesting and user membership
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS auth_principal_memberships (
    membership_id         TEXT PRIMARY KEY,
    tenant_id             TEXT NOT NULL DEFAULT 'local-dev',
    parent_principal_id   TEXT NOT NULL REFERENCES auth_principals(principal_id) ON DELETE CASCADE,
    child_principal_id    TEXT NOT NULL REFERENCES auth_principals(principal_id) ON DELETE CASCADE,
    created_at            TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS auth_principal_memberships_dedupe_idx
    ON auth_principal_memberships(tenant_id, parent_principal_id, child_principal_id);

CREATE INDEX IF NOT EXISTS auth_principal_memberships_child_idx
    ON auth_principal_memberships(tenant_id, child_principal_id);

-- ------------------------------------------------------------
-- auth_roles: reusable RBAC roles
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS auth_roles (
    role_id         TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL DEFAULT 'local-dev',
    name            TEXT NOT NULL,
    description     TEXT DEFAULT '',
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS auth_roles_name_idx
    ON auth_roles(tenant_id, name);

-- ------------------------------------------------------------
-- auth_role_bindings: principal -> role assignments
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS auth_role_bindings (
    binding_id      TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL DEFAULT 'local-dev',
    role_id         TEXT NOT NULL REFERENCES auth_roles(role_id) ON DELETE CASCADE,
    principal_id    TEXT NOT NULL REFERENCES auth_principals(principal_id) ON DELETE CASCADE,
    created_at      TIMESTAMPTZ DEFAULT now(),
    disabled_at     TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS auth_role_bindings_dedupe_idx
    ON auth_role_bindings(tenant_id, role_id, principal_id);

CREATE INDEX IF NOT EXISTS auth_role_bindings_principal_idx
    ON auth_role_bindings(tenant_id, principal_id, disabled_at);

-- ------------------------------------------------------------
-- auth_role_permissions: resource selectors granted to each role
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS auth_role_permissions (
    permission_id      TEXT PRIMARY KEY,
    tenant_id          TEXT NOT NULL DEFAULT 'local-dev',
    role_id            TEXT NOT NULL REFERENCES auth_roles(role_id) ON DELETE CASCADE,
    resource_type      TEXT NOT NULL,
    action             TEXT NOT NULL DEFAULT 'use',
    resource_selector  TEXT NOT NULL DEFAULT '*',
    created_at         TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS auth_role_permissions_dedupe_idx
    ON auth_role_permissions(tenant_id, role_id, resource_type, action, resource_selector);

CREATE INDEX IF NOT EXISTS auth_role_permissions_role_idx
    ON auth_role_permissions(tenant_id, role_id, resource_type, action);

-- ------------------------------------------------------------
-- mcp_connections: user-owned Streamable HTTP MCP server profiles
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mcp_connections (
    connection_id    TEXT PRIMARY KEY,
    tenant_id        TEXT NOT NULL DEFAULT 'local-dev',
    owner_user_id    TEXT NOT NULL DEFAULT 'local-cli',
    display_name     TEXT NOT NULL,
    connection_slug  TEXT NOT NULL,
    server_url       TEXT NOT NULL,
    auth_type        TEXT NOT NULL DEFAULT 'none',
    encrypted_secret TEXT NOT NULL DEFAULT '',
    status           TEXT NOT NULL DEFAULT 'active',
    allowed_agents   JSONB NOT NULL DEFAULT '["general"]'::jsonb,
    visibility       TEXT NOT NULL DEFAULT 'private',
    health           JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata_json    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ DEFAULT now(),
    updated_at       TIMESTAMPTZ DEFAULT now(),
    last_tested_at   TIMESTAMPTZ,
    last_refreshed_at TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS mcp_connections_owner_slug_idx
    ON mcp_connections(tenant_id, owner_user_id, connection_slug);

CREATE INDEX IF NOT EXISTS mcp_connections_owner_status_idx
    ON mcp_connections(tenant_id, owner_user_id, status, updated_at DESC);

-- ------------------------------------------------------------
-- mcp_tool_catalog: cached tools discovered from MCP connections
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mcp_tool_catalog (
    tool_id          TEXT PRIMARY KEY,
    connection_id    TEXT NOT NULL REFERENCES mcp_connections(connection_id) ON DELETE CASCADE,
    tenant_id        TEXT NOT NULL DEFAULT 'local-dev',
    owner_user_id    TEXT NOT NULL DEFAULT 'local-cli',
    raw_tool_name    TEXT NOT NULL,
    registry_name    TEXT NOT NULL,
    tool_slug        TEXT NOT NULL,
    description      TEXT NOT NULL DEFAULT '',
    input_schema     JSONB NOT NULL DEFAULT '{}'::jsonb,
    read_only        BOOLEAN NOT NULL DEFAULT FALSE,
    destructive      BOOLEAN NOT NULL DEFAULT TRUE,
    background_safe  BOOLEAN NOT NULL DEFAULT FALSE,
    should_defer     BOOLEAN NOT NULL DEFAULT TRUE,
    search_hint      TEXT NOT NULL DEFAULT '',
    defer_priority   INTEGER NOT NULL DEFAULT 50,
    enabled          BOOLEAN NOT NULL DEFAULT TRUE,
    status           TEXT NOT NULL DEFAULT 'active',
    checksum         TEXT NOT NULL DEFAULT '',
    metadata_json    JSONB NOT NULL DEFAULT '{}'::jsonb,
    first_seen_at    TIMESTAMPTZ DEFAULT now(),
    last_seen_at     TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS mcp_tool_catalog_connection_raw_idx
    ON mcp_tool_catalog(connection_id, raw_tool_name);

CREATE UNIQUE INDEX IF NOT EXISTS mcp_tool_catalog_registry_idx
    ON mcp_tool_catalog(tenant_id, owner_user_id, registry_name);

CREATE INDEX IF NOT EXISTS mcp_tool_catalog_connection_status_idx
    ON mcp_tool_catalog(connection_id, enabled, status, registry_name);

-- ------------------------------------------------------------
-- documents: one row per ingested file
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    doc_id             TEXT PRIMARY KEY,
    tenant_id          TEXT NOT NULL DEFAULT 'local-dev',
    collection_id      TEXT NOT NULL DEFAULT 'default',
    title              TEXT NOT NULL,
    source_type        TEXT NOT NULL,           -- 'kb' | 'upload'
    source_path        TEXT,
    content_hash       TEXT NOT NULL,
    num_chunks         INTEGER DEFAULT 0,
    ingested_at        TIMESTAMPTZ DEFAULT now(),
    file_type          TEXT,                    -- 'pdf' | 'txt' | 'md' | 'docx'
    doc_structure_type TEXT DEFAULT 'general',  -- see chunk_type values below
    source_display_path TEXT DEFAULT '',
    source_identity    TEXT DEFAULT ''
);

-- Backfill / migrate existing databases created before tenant_id.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS tenant_id TEXT;
UPDATE documents SET tenant_id = 'local-dev' WHERE tenant_id IS NULL OR tenant_id = '';
ALTER TABLE documents ALTER COLUMN tenant_id SET DEFAULT 'local-dev';
ALTER TABLE documents ALTER COLUMN tenant_id SET NOT NULL;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS collection_id TEXT;
UPDATE documents SET collection_id = 'default' WHERE collection_id IS NULL OR collection_id = '';
ALTER TABLE documents ALTER COLUMN collection_id SET DEFAULT 'default';
ALTER TABLE documents ALTER COLUMN collection_id SET NOT NULL;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_display_path TEXT;
UPDATE documents
SET source_display_path = COALESCE(NULLIF(source_path, ''), title)
WHERE source_display_path IS NULL OR source_display_path = '';
ALTER TABLE documents ALTER COLUMN source_display_path SET DEFAULT '';
ALTER TABLE documents ALTER COLUMN source_display_path SET NOT NULL;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_identity TEXT;
UPDATE documents
SET source_identity = COALESCE(NULLIF(source_path, ''), title)
WHERE source_identity IS NULL OR source_identity = '';
ALTER TABLE documents ALTER COLUMN source_identity SET DEFAULT '';
ALTER TABLE documents ALTER COLUMN source_identity SET NOT NULL;

ALTER TABLE collections ADD COLUMN IF NOT EXISTS maintenance_policy TEXT;
UPDATE collections
SET maintenance_policy = 'configured_kb_sources'
WHERE collection_id = 'default'
  AND (maintenance_policy IS NULL OR maintenance_policy = '');
UPDATE collections AS c
SET maintenance_policy = CASE
    WHEN EXISTS (
        SELECT 1
        FROM documents AS d
        WHERE d.tenant_id = c.tenant_id
          AND d.collection_id = c.collection_id
    )
    AND NOT EXISTS (
        SELECT 1
        FROM documents AS d
        WHERE d.tenant_id = c.tenant_id
          AND d.collection_id = c.collection_id
          AND COALESCE(d.source_type, '') <> 'kb'
    )
    THEN 'configured_kb_sources'
    ELSE 'indexed_documents'
END
WHERE maintenance_policy IS NULL OR maintenance_policy = '';
ALTER TABLE collections ALTER COLUMN maintenance_policy SET DEFAULT 'indexed_documents';
UPDATE collections
SET maintenance_policy = 'indexed_documents'
WHERE maintenance_policy IS NULL OR maintenance_policy = '';
ALTER TABLE collections ALTER COLUMN maintenance_policy SET NOT NULL;

CREATE INDEX IF NOT EXISTS documents_tenant_idx
    ON documents(tenant_id);

CREATE INDEX IF NOT EXISTS documents_tenant_source_idx
    ON documents(tenant_id, source_type);

CREATE INDEX IF NOT EXISTS documents_tenant_collection_idx
    ON documents(tenant_id, collection_id);

CREATE INDEX IF NOT EXISTS documents_tenant_collection_source_identity_idx
    ON documents(tenant_id, collection_id, source_identity);

INSERT INTO collections (tenant_id, collection_id, maintenance_policy, created_at, updated_at)
SELECT
    tenant_id,
    collection_id,
    CASE
        WHEN collection_id = 'default' OR bool_and(COALESCE(source_type, '') = 'kb')
        THEN 'configured_kb_sources'
        ELSE 'indexed_documents'
    END,
    COALESCE(MIN(ingested_at), now()),
    COALESCE(MAX(ingested_at), now())
FROM documents
GROUP BY tenant_id, collection_id
ON CONFLICT (tenant_id, collection_id) DO UPDATE SET
    maintenance_policy = COALESCE(NULLIF(collections.maintenance_policy, ''), EXCLUDED.maintenance_policy),
    created_at = LEAST(collections.created_at, EXCLUDED.created_at),
    updated_at = GREATEST(collections.updated_at, EXCLUDED.updated_at);

-- ------------------------------------------------------------
-- chunks: one row per document chunk
--
-- embedding dimension is injected from Settings.EMBEDDING_DIM when schema is applied.
-- Existing databases may still require `python run.py migrate-embedding-dim --yes`
-- to realign and reindex.
--
-- chunk_type values:
--   'general'     – plain prose (no detected structure)
--   'clause'      – numbered clause / article
--   'section'     – section heading block
--   'requirement' – contains shall/must/REQ-NNN language
--   'header'      – document title / heading only
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       TEXT PRIMARY KEY,
    doc_id         TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    tenant_id      TEXT NOT NULL DEFAULT 'local-dev',
    collection_id  TEXT NOT NULL DEFAULT 'default',
    chunk_index    INTEGER NOT NULL,
    page_number    INTEGER,
    clause_number  TEXT,          -- e.g. '3', '3.2', '10.1.4'
    section_title  TEXT,          -- heading text extracted from the boundary line
    sheet_name     TEXT,
    row_start      INTEGER,
    row_end        INTEGER,
    cell_range     TEXT,
    content        TEXT NOT NULL,
    embedding      vector(__EMBEDDING_DIM__),
    ts             tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    chunk_type     TEXT DEFAULT 'general'
);

-- Backfill / migrate existing databases created before tenant_id.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tenant_id TEXT;
UPDATE chunks c
SET tenant_id = COALESCE((SELECT d.tenant_id FROM documents d WHERE d.doc_id = c.doc_id), 'local-dev')
WHERE c.tenant_id IS NULL OR c.tenant_id = '';
ALTER TABLE chunks ALTER COLUMN tenant_id SET DEFAULT 'local-dev';
ALTER TABLE chunks ALTER COLUMN tenant_id SET NOT NULL;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS collection_id TEXT;
UPDATE chunks c
SET collection_id = COALESCE((SELECT d.collection_id FROM documents d WHERE d.doc_id = c.doc_id), 'default')
WHERE c.collection_id IS NULL OR c.collection_id = '';
ALTER TABLE chunks ALTER COLUMN collection_id SET DEFAULT 'default';
ALTER TABLE chunks ALTER COLUMN collection_id SET NOT NULL;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS sheet_name TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS row_start INTEGER;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS row_end INTEGER;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS cell_range TEXT;

-- Indexes
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx
    ON chunks(doc_id);

CREATE INDEX IF NOT EXISTS chunks_tenant_doc_idx
    ON chunks(tenant_id, doc_id);

CREATE INDEX IF NOT EXISTS chunks_tenant_collection_idx
    ON chunks(tenant_id, collection_id);

-- HNSW can be created on an empty table (unlike IVFFlat)
-- m=16, ef_construction=64 are conservative defaults; tune for your dataset size
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_ts_gin_idx
    ON chunks USING GIN(ts);

CREATE INDEX IF NOT EXISTS chunks_chunk_type_idx
    ON chunks(chunk_type);

CREATE INDEX IF NOT EXISTS chunks_clause_number_idx
    ON chunks(tenant_id, doc_id, clause_number)
    WHERE clause_number IS NOT NULL;

-- ------------------------------------------------------------
-- requirement_statements: persisted statement-level requirement inventory
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS requirement_statements (
    requirement_id             TEXT PRIMARY KEY,
    doc_id                     TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    tenant_id                  TEXT NOT NULL DEFAULT 'local-dev',
    collection_id              TEXT NOT NULL DEFAULT 'default',
    source_type                TEXT NOT NULL DEFAULT '',
    document_title             TEXT NOT NULL DEFAULT '',
    statement_index            INTEGER NOT NULL DEFAULT 0,
    chunk_id                   TEXT NOT NULL DEFAULT '',
    chunk_index                INTEGER NOT NULL DEFAULT 0,
    statement_text             TEXT NOT NULL,
    normalized_statement_text  TEXT NOT NULL DEFAULT '',
    modality                   TEXT NOT NULL DEFAULT '',
    page_number                INTEGER,
    clause_number              TEXT DEFAULT '',
    section_title              TEXT DEFAULT '',
    char_start                 INTEGER NOT NULL DEFAULT 0,
    char_end                   INTEGER NOT NULL DEFAULT 0,
    multi_requirement          BOOLEAN NOT NULL DEFAULT FALSE,
    extractor_version          TEXT NOT NULL DEFAULT 'requirements_v1',
    extractor_mode             TEXT NOT NULL DEFAULT 'mandatory',
    created_at                 TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS requirement_statements_tenant_doc_idx
    ON requirement_statements(tenant_id, doc_id, statement_index);

CREATE INDEX IF NOT EXISTS requirement_statements_tenant_collection_idx
    ON requirement_statements(tenant_id, collection_id, source_type);

CREATE INDEX IF NOT EXISTS requirement_statements_modality_idx
    ON requirement_statements(tenant_id, collection_id, modality);

-- ------------------------------------------------------------
-- graph_indexes: managed graph catalog for GraphRAG and imports
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS graph_indexes (
    graph_id               TEXT PRIMARY KEY,
    tenant_id              TEXT NOT NULL DEFAULT 'local-dev',
    collection_id          TEXT NOT NULL DEFAULT 'default',
    display_name           TEXT NOT NULL,
    owner_admin_user_id    TEXT DEFAULT '',
    visibility             TEXT NOT NULL DEFAULT 'tenant',
    backend                TEXT NOT NULL DEFAULT 'microsoft_graphrag',
    status                 TEXT NOT NULL DEFAULT 'draft',
    root_path              TEXT,
    artifact_path          TEXT,
    domain_summary         TEXT DEFAULT '',
    entity_samples         JSONB NOT NULL DEFAULT '[]'::jsonb,
    relationship_samples   JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_doc_ids         TEXT[] DEFAULT '{}'::TEXT[],
    capabilities           TEXT[] DEFAULT '{}'::TEXT[],
    supported_query_methods TEXT[] DEFAULT '{}'::TEXT[],
    query_ready            BOOLEAN NOT NULL DEFAULT FALSE,
    query_backend          TEXT NOT NULL DEFAULT '',
    artifact_tables        TEXT[] DEFAULT '{}'::TEXT[],
    artifact_mtime         TIMESTAMPTZ,
    graph_context_summary  JSONB NOT NULL DEFAULT '{}'::jsonb,
    config_json            JSONB NOT NULL DEFAULT '{}'::jsonb,
    prompt_overrides_json  JSONB NOT NULL DEFAULT '{}'::jsonb,
    graph_skill_ids        TEXT[] DEFAULT '{}'::TEXT[],
    health                 JSONB NOT NULL DEFAULT '{}'::jsonb,
    freshness_score        DOUBLE PRECISION NOT NULL DEFAULT 0,
    last_indexed_at        TIMESTAMPTZ,
    created_at             TIMESTAMPTZ DEFAULT now(),
    updated_at             TIMESTAMPTZ DEFAULT now(),
    summary_embedding      vector(__EMBEDDING_DIM__)
);

ALTER TABLE graph_indexes ADD COLUMN IF NOT EXISTS query_ready BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE graph_indexes ADD COLUMN IF NOT EXISTS query_backend TEXT NOT NULL DEFAULT '';
ALTER TABLE graph_indexes ADD COLUMN IF NOT EXISTS artifact_tables TEXT[] DEFAULT '{}'::TEXT[];
ALTER TABLE graph_indexes ADD COLUMN IF NOT EXISTS artifact_mtime TIMESTAMPTZ;
ALTER TABLE graph_indexes ADD COLUMN IF NOT EXISTS graph_context_summary JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE graph_indexes ADD COLUMN IF NOT EXISTS owner_admin_user_id TEXT DEFAULT '';
ALTER TABLE graph_indexes ADD COLUMN IF NOT EXISTS visibility TEXT NOT NULL DEFAULT 'tenant';
ALTER TABLE graph_indexes ADD COLUMN IF NOT EXISTS config_json JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE graph_indexes ADD COLUMN IF NOT EXISTS prompt_overrides_json JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE graph_indexes ADD COLUMN IF NOT EXISTS graph_skill_ids TEXT[] DEFAULT '{}'::TEXT[];

CREATE INDEX IF NOT EXISTS graph_indexes_tenant_collection_idx
    ON graph_indexes(tenant_id, collection_id);

CREATE INDEX IF NOT EXISTS graph_indexes_tenant_status_idx
    ON graph_indexes(tenant_id, status);

CREATE INDEX IF NOT EXISTS graph_indexes_visibility_idx
    ON graph_indexes(tenant_id, visibility, owner_admin_user_id);

CREATE INDEX IF NOT EXISTS graph_indexes_backend_idx
    ON graph_indexes(tenant_id, backend);

CREATE INDEX IF NOT EXISTS graph_indexes_query_ready_idx
    ON graph_indexes(tenant_id, query_ready, query_backend);

CREATE INDEX IF NOT EXISTS graph_indexes_embedding_hnsw_idx
    ON graph_indexes USING hnsw (summary_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

INSERT INTO collections (tenant_id, collection_id, created_at, updated_at)
SELECT
    tenant_id,
    collection_id,
    COALESCE(MIN(created_at), now()),
    COALESCE(MAX(COALESCE(updated_at, created_at)), now())
FROM graph_indexes
GROUP BY tenant_id, collection_id
ON CONFLICT (tenant_id, collection_id) DO UPDATE SET
    created_at = LEAST(collections.created_at, EXCLUDED.created_at),
    updated_at = GREATEST(collections.updated_at, EXCLUDED.updated_at);

-- ------------------------------------------------------------
-- graph_index_sources: source lineage for each graph
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS graph_index_sources (
    graph_source_id  TEXT PRIMARY KEY,
    graph_id         TEXT NOT NULL REFERENCES graph_indexes(graph_id) ON DELETE CASCADE,
    tenant_id        TEXT NOT NULL DEFAULT 'local-dev',
    source_doc_id    TEXT DEFAULT '',
    source_path      TEXT DEFAULT '',
    source_title     TEXT DEFAULT '',
    source_type      TEXT DEFAULT '',
    created_at       TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS graph_index_sources_dedupe_idx
    ON graph_index_sources(tenant_id, graph_id, source_doc_id, source_path);

CREATE INDEX IF NOT EXISTS graph_index_sources_graph_idx
    ON graph_index_sources(tenant_id, graph_id);

-- ------------------------------------------------------------
-- graph_index_runs: job and status history for graph lifecycle operations
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS graph_index_runs (
    run_id         TEXT PRIMARY KEY,
    tenant_id      TEXT NOT NULL DEFAULT 'local-dev',
    graph_id       TEXT NOT NULL REFERENCES graph_indexes(graph_id) ON DELETE CASCADE,
    operation      TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'queued',
    detail         TEXT DEFAULT '',
    metadata       JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at     TIMESTAMPTZ DEFAULT now(),
    completed_at   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS graph_index_runs_graph_idx
    ON graph_index_runs(tenant_id, graph_id, started_at DESC);

-- ------------------------------------------------------------
-- graph_query_cache: lightweight cache of graph query results
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS graph_query_cache (
    cache_id        TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL DEFAULT 'local-dev',
    graph_id        TEXT NOT NULL REFERENCES graph_indexes(graph_id) ON DELETE CASCADE,
    query_text      TEXT NOT NULL,
    query_method    TEXT NOT NULL DEFAULT 'local',
    response_json   JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ DEFAULT now(),
    expires_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS graph_query_cache_lookup_idx
    ON graph_query_cache(tenant_id, graph_id, query_method, expires_at DESC);

-- ------------------------------------------------------------
-- canonical_entities: shared identity layer across graph, chunks, and metadata
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS canonical_entities (
    entity_id          TEXT PRIMARY KEY,
    tenant_id          TEXT NOT NULL DEFAULT 'local-dev',
    collection_id      TEXT NOT NULL DEFAULT 'default',
    canonical_name     TEXT NOT NULL,
    normalized_name    TEXT NOT NULL,
    entity_type        TEXT DEFAULT '',
    description        TEXT DEFAULT '',
    graph_id           TEXT DEFAULT '',
    metadata           JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at         TIMESTAMPTZ DEFAULT now(),
    updated_at         TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS canonical_entities_name_idx
    ON canonical_entities(tenant_id, collection_id, normalized_name);

CREATE INDEX IF NOT EXISTS canonical_entities_graph_idx
    ON canonical_entities(tenant_id, collection_id, graph_id);

-- ------------------------------------------------------------
-- entity_aliases: exact alias lookup for canonical entities
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entity_aliases (
    alias_id           TEXT PRIMARY KEY,
    entity_id          TEXT NOT NULL REFERENCES canonical_entities(entity_id) ON DELETE CASCADE,
    tenant_id          TEXT NOT NULL DEFAULT 'local-dev',
    collection_id      TEXT NOT NULL DEFAULT 'default',
    alias              TEXT NOT NULL,
    normalized_alias   TEXT NOT NULL,
    source             TEXT DEFAULT 'graph',
    created_at         TIMESTAMPTZ DEFAULT now(),
    updated_at         TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS entity_aliases_lookup_idx
    ON entity_aliases(tenant_id, collection_id, normalized_alias, entity_id);

CREATE INDEX IF NOT EXISTS entity_aliases_entity_idx
    ON entity_aliases(entity_id);

-- ------------------------------------------------------------
-- entity_mentions: doc/chunk-level links back to canonical entities
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entity_mentions (
    mention_id         TEXT PRIMARY KEY,
    entity_id          TEXT NOT NULL REFERENCES canonical_entities(entity_id) ON DELETE CASCADE,
    tenant_id          TEXT NOT NULL DEFAULT 'local-dev',
    collection_id      TEXT NOT NULL DEFAULT 'default',
    doc_id             TEXT DEFAULT '',
    chunk_id           TEXT DEFAULT '',
    graph_id           TEXT DEFAULT '',
    mention_text       TEXT DEFAULT '',
    mention_type       TEXT DEFAULT '',
    metadata           JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at         TIMESTAMPTZ DEFAULT now(),
    updated_at         TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS entity_mentions_lookup_idx
    ON entity_mentions(tenant_id, entity_id, doc_id, chunk_id, graph_id, mention_text);

CREATE INDEX IF NOT EXISTS entity_mentions_doc_idx
    ON entity_mentions(tenant_id, collection_id, doc_id, chunk_id);

-- ------------------------------------------------------------
-- skills: skill-pack metadata indexed separately from KB docs
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS skills (
    skill_id      TEXT PRIMARY KEY,
    tenant_id     TEXT NOT NULL DEFAULT 'local-dev',
    owner_user_id TEXT DEFAULT '',
    graph_id      TEXT DEFAULT '',
    name          TEXT NOT NULL,
    agent_scope   TEXT NOT NULL,
    tool_tags     TEXT[] DEFAULT '{}'::TEXT[],
    task_tags     TEXT[] DEFAULT '{}'::TEXT[],
    version       TEXT NOT NULL DEFAULT '1',
    enabled       BOOLEAN NOT NULL DEFAULT TRUE,
    visibility    TEXT NOT NULL DEFAULT 'global',
    status        TEXT NOT NULL DEFAULT 'active',
    version_parent TEXT DEFAULT '',
    source_path   TEXT,
    checksum      TEXT NOT NULL,
    body_markdown TEXT DEFAULT '',
    kind TEXT NOT NULL DEFAULT 'retrievable',
    execution_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    description   TEXT DEFAULT '',
    retrieval_profile TEXT DEFAULT '',
    controller_hints JSONB NOT NULL DEFAULT '{}'::jsonb,
    coverage_goal TEXT DEFAULT '',
    result_mode TEXT DEFAULT '',
    updated_at    TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE skills ADD COLUMN IF NOT EXISTS retrieval_profile TEXT DEFAULT '';
ALTER TABLE skills ADD COLUMN IF NOT EXISTS controller_hints JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE skills ADD COLUMN IF NOT EXISTS coverage_goal TEXT DEFAULT '';
ALTER TABLE skills ADD COLUMN IF NOT EXISTS result_mode TEXT DEFAULT '';
ALTER TABLE skills ADD COLUMN IF NOT EXISTS owner_user_id TEXT DEFAULT '';
ALTER TABLE skills ADD COLUMN IF NOT EXISTS graph_id TEXT DEFAULT '';
ALTER TABLE skills ADD COLUMN IF NOT EXISTS visibility TEXT NOT NULL DEFAULT 'global';
ALTER TABLE skills ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';
ALTER TABLE skills ADD COLUMN IF NOT EXISTS version_parent TEXT DEFAULT '';
ALTER TABLE skills ADD COLUMN IF NOT EXISTS body_markdown TEXT DEFAULT '';
ALTER TABLE skills ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'retrievable';
ALTER TABLE skills ADD COLUMN IF NOT EXISTS execution_config JSONB NOT NULL DEFAULT '{}'::jsonb;
UPDATE skills SET version_parent = skill_id WHERE version_parent IS NULL OR version_parent = '';

CREATE INDEX IF NOT EXISTS skills_tenant_scope_idx
    ON skills(tenant_id, agent_scope);

CREATE INDEX IF NOT EXISTS skills_enabled_idx
    ON skills(tenant_id, enabled);

CREATE INDEX IF NOT EXISTS skills_scope_visibility_idx
    ON skills(tenant_id, owner_user_id, visibility, status, agent_scope);

CREATE INDEX IF NOT EXISTS skills_version_parent_idx
    ON skills(tenant_id, version_parent);

CREATE INDEX IF NOT EXISTS skills_graph_idx
    ON skills(tenant_id, graph_id, agent_scope, status);

CREATE INDEX IF NOT EXISTS skills_kind_idx
    ON skills(tenant_id, kind, status);

-- ------------------------------------------------------------
-- skill_chunks: retrieval surface for skill-pack chunks
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS skill_chunks (
    skill_chunk_id TEXT PRIMARY KEY,
    skill_id       TEXT NOT NULL REFERENCES skills(skill_id) ON DELETE CASCADE,
    tenant_id      TEXT NOT NULL DEFAULT 'local-dev',
    chunk_index    INTEGER NOT NULL,
    content        TEXT NOT NULL,
    embedding      vector(__EMBEDDING_DIM__)
);

CREATE INDEX IF NOT EXISTS skill_chunks_skill_idx
    ON skill_chunks(skill_id);

CREATE INDEX IF NOT EXISTS skill_chunks_tenant_idx
    ON skill_chunks(tenant_id);

CREATE INDEX IF NOT EXISTS skill_chunks_embedding_hnsw_idx
    ON skill_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ------------------------------------------------------------
-- skill_telemetry_events: append-only verifier-backed scoring
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS skill_telemetry_events (
    event_id         TEXT PRIMARY KEY,
    tenant_id        TEXT NOT NULL DEFAULT 'local-dev',
    skill_id         TEXT NOT NULL,
    skill_family_id  TEXT NOT NULL,
    query            TEXT DEFAULT '',
    answer_quality   TEXT NOT NULL DEFAULT '',
    agent_name       TEXT DEFAULT '',
    session_id       TEXT DEFAULT '',
    created_at       TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS skill_telemetry_family_idx
    ON skill_telemetry_events(tenant_id, skill_family_id, created_at DESC);

CREATE INDEX IF NOT EXISTS skill_telemetry_skill_idx
    ON skill_telemetry_events(tenant_id, skill_id, created_at DESC);

CREATE INDEX IF NOT EXISTS skill_telemetry_session_idx
    ON skill_telemetry_events(tenant_id, session_id, created_at DESC);

-- ------------------------------------------------------------
-- memory: persistent cross-turn key-value store per tenant+session
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memory (
    id          SERIAL PRIMARY KEY,
    tenant_id   TEXT NOT NULL DEFAULT 'local-dev',
    session_id  TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

-- Backfill / migrate existing databases created before tenant_id.
ALTER TABLE memory ADD COLUMN IF NOT EXISTS tenant_id TEXT;
UPDATE memory SET tenant_id = 'local-dev' WHERE tenant_id IS NULL OR tenant_id = '';
ALTER TABLE memory ALTER COLUMN tenant_id SET DEFAULT 'local-dev';
ALTER TABLE memory ALTER COLUMN tenant_id SET NOT NULL;

-- Keep old uniqueness for backward compatibility if it exists, and add the
-- tenant-aware unique index required by ON CONFLICT (tenant_id, session_id, key).
CREATE UNIQUE INDEX IF NOT EXISTS memory_tenant_session_key_uniq
    ON memory(tenant_id, session_id, key);

CREATE INDEX IF NOT EXISTS memory_tenant_session_idx
    ON memory(tenant_id, session_id);

-- ------------------------------------------------------------
-- managed memory v2: typed hybrid memory records, observations, episodes
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memory_records (
    memory_id            TEXT PRIMARY KEY,
    tenant_id            TEXT NOT NULL DEFAULT 'local-dev',
    user_id              TEXT NOT NULL DEFAULT '',
    conversation_id      TEXT NOT NULL DEFAULT '',
    session_id           TEXT NOT NULL DEFAULT '',
    scope                TEXT NOT NULL DEFAULT 'conversation',
    memory_type          TEXT NOT NULL DEFAULT 'task_state',
    memory_key           TEXT NOT NULL DEFAULT '',
    title                TEXT NOT NULL DEFAULT '',
    canonical_text       TEXT NOT NULL DEFAULT '',
    structured_payload   JSONB NOT NULL DEFAULT '{}'::jsonb,
    importance           DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    confidence           DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    active               BOOLEAN NOT NULL DEFAULT TRUE,
    superseded_by        TEXT NOT NULL DEFAULT '',
    provenance_turn_ids  JSONB NOT NULL DEFAULT '[]'::jsonb,
    last_used_at         TIMESTAMPTZ,
    created_at           TIMESTAMPTZ DEFAULT now(),
    updated_at           TIMESTAMPTZ DEFAULT now(),
    source               TEXT NOT NULL DEFAULT '',
    ttl_hint             TEXT NOT NULL DEFAULT '',
    embedding            vector(__EMBEDDING_DIM__),
    search_vector        tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(memory_key, '') || ' ' || coalesce(canonical_text, ''))
    ) STORED
);

CREATE INDEX IF NOT EXISTS memory_records_scope_idx
    ON memory_records(tenant_id, user_id, conversation_id, session_id, scope, active, updated_at DESC);

CREATE INDEX IF NOT EXISTS memory_records_type_idx
    ON memory_records(tenant_id, user_id, conversation_id, session_id, memory_type, active, updated_at DESC);

CREATE INDEX IF NOT EXISTS memory_records_key_idx
    ON memory_records(tenant_id, user_id, conversation_id, session_id, scope, memory_key, updated_at DESC);

CREATE INDEX IF NOT EXISTS memory_records_search_idx
    ON memory_records USING GIN(search_vector);

CREATE INDEX IF NOT EXISTS memory_records_embedding_hnsw_idx
    ON memory_records USING hnsw(embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS memory_observations (
    observation_id       TEXT PRIMARY KEY,
    memory_id            TEXT NOT NULL REFERENCES memory_records(memory_id) ON DELETE CASCADE,
    tenant_id            TEXT NOT NULL DEFAULT 'local-dev',
    user_id              TEXT NOT NULL DEFAULT '',
    conversation_id      TEXT NOT NULL DEFAULT '',
    session_id           TEXT NOT NULL DEFAULT '',
    operation            TEXT NOT NULL DEFAULT 'create',
    evidence_turn_ids    JSONB NOT NULL DEFAULT '[]'::jsonb,
    note                 TEXT NOT NULL DEFAULT '',
    raw_payload          JSONB NOT NULL DEFAULT '{}'::jsonb,
    confidence           DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    created_at           TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS memory_observations_memory_idx
    ON memory_observations(memory_id, created_at DESC);

CREATE INDEX IF NOT EXISTS memory_observations_session_idx
    ON memory_observations(tenant_id, user_id, conversation_id, session_id, created_at DESC);

CREATE TABLE IF NOT EXISTS memory_episodes (
    episode_id           TEXT PRIMARY KEY,
    tenant_id            TEXT NOT NULL DEFAULT 'local-dev',
    user_id              TEXT NOT NULL DEFAULT '',
    conversation_id      TEXT NOT NULL DEFAULT '',
    session_id           TEXT NOT NULL DEFAULT '',
    summary_text         TEXT NOT NULL DEFAULT '',
    topic_hint           TEXT NOT NULL DEFAULT '',
    start_turn_index     INTEGER NOT NULL DEFAULT 0,
    end_turn_index       INTEGER NOT NULL DEFAULT 0,
    message_ids          JSONB NOT NULL DEFAULT '[]'::jsonb,
    importance           DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    created_at           TIMESTAMPTZ DEFAULT now(),
    updated_at           TIMESTAMPTZ DEFAULT now(),
    embedding            vector(__EMBEDDING_DIM__),
    search_vector        tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(topic_hint, '') || ' ' || coalesce(summary_text, ''))
    ) STORED
);

CREATE INDEX IF NOT EXISTS memory_episodes_session_idx
    ON memory_episodes(tenant_id, user_id, conversation_id, session_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS memory_episodes_search_idx
    ON memory_episodes USING GIN(search_vector);

CREATE INDEX IF NOT EXISTS memory_episodes_embedding_hnsw_idx
    ON memory_episodes USING hnsw(embedding vector_cosine_ops);
