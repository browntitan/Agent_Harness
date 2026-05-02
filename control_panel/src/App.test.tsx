import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { ToastProvider } from './components/ui'
import { ThemeProvider } from './theme/ThemeProvider'
import { DensityProvider } from './theme/DensityProvider'

function renderApp(initialPath: string = '/') {
  return render(
    <ThemeProvider initialTheme="dark">
      <DensityProvider initialDensity="comfortable">
        <ToastProvider>
          <MemoryRouter initialEntries={[initialPath]}>
            <App />
          </MemoryRouter>
        </ToastProvider>
      </DensityProvider>
    </ThemeProvider>,
  )
}

type DocState = {
  doc_id: string
  title: string
  collection_id: string
  source_path: string
  source_display_path: string
  source_type: string
  ingested_at: string
  file_type: string
  doc_structure_type: string
  extracted: string
  raw: string
}

type CollectionCatalogState = {
  created_at: string
  updated_at: string
  maintenance_policy: string
}

type SkillState = {
  skill_id: string
  name: string
  agent_scope: string
  graph_id?: string
  collection_id?: string
  body_markdown: string
  enabled: boolean
  status: string
  version: string
  version_parent: string
  updated_at: string
}

type GraphState = {
  graph_id: string
  display_name: string
  collection_id: string
  backend: string
  status: string
  query_ready: boolean
  query_backend: string
  domain_summary: string
  source_doc_ids: string[]
  graph_skill_ids: string[]
  prompt_overrides_json: Record<string, unknown>
  config_json: Record<string, unknown>
  logs: Array<Record<string, unknown>>
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function requestPath(input: RequestInfo | URL): string {
  if (typeof input === 'string') return new URL(input, 'http://test.local').pathname
  if (input instanceof URL) return input.pathname
  return new URL(input.url, 'http://test.local').pathname
}

function requestMethod(input: RequestInfo | URL, init?: RequestInit): string {
  if (init?.method) return init.method.toUpperCase()
  if (typeof input === 'string' || input instanceof URL) return 'GET'
  return input.method.toUpperCase()
}

function readJsonBody(init?: RequestInit): Record<string, unknown> {
  if (!init?.body || typeof init.body !== 'string') return {}
  return JSON.parse(init.body)
}

function slugify(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'item'
}

function createDoc(
  collectionId: string,
  title: string,
  content: string,
  options: {
    docId?: string
    sourcePath?: string
    sourceDisplayPath?: string
    sourceType?: string
    ingestedAt?: string
  } = {},
): DocState {
  return {
    doc_id: options.docId ?? `${collectionId}-${slugify(title)}`,
    title,
    collection_id: collectionId,
    source_path: options.sourcePath ?? `/tmp/${title}`,
    source_display_path: options.sourceDisplayPath ?? title,
    source_type: options.sourceType ?? 'upload',
    ingested_at: options.ingestedAt ?? '2026-04-08T10:00:00Z',
    file_type: title.split('.').pop() ?? 'txt',
    doc_structure_type: 'general',
    extracted: content,
    raw: content,
  }
}

function parseSkillName(bodyMarkdown: string): string {
  const firstLine = bodyMarkdown.split('\n')[0]?.trim() ?? ''
  return firstLine.startsWith('# ') ? firstLine.slice(2).trim() || 'New Skill' : 'New Skill'
}

function parseSkillMetadata(bodyMarkdown: string): Record<string, string> {
  const metadata: Record<string, string> = {}
  for (const line of bodyMarkdown.split('\n')) {
    const match = /^([a-zA-Z_][a-zA-Z0-9_-]*):\s*(.+?)\s*$/.exec(line.trim())
    if (match) metadata[match[1].toLowerCase().replace(/-/g, '_')] = match[2]
  }
  return metadata
}

function createFetchMock(options: {
  operationsError?: string
  compatibilityMode?: 'full' | 'no-endpoint' | 'partial-architecture'
  blockSkillDeactivate?: boolean
  architectureSnapshotFailureCalls?: number[]
  architectureActivityFailureCalls?: number[]
  langGraphUnavailable?: boolean
} = {}) {
  const state = {
    maxAgentSteps: '6',
    clarificationSensitivity: '50',
    ollamaChatModel: 'gpt-oss:20b',
    promptBase: 'Base general prompt',
    promptOverlay: '',
    agent: {
      name: 'general',
      mode: 'react',
      description: 'general agent',
      prompt_file: 'general_agent.md',
      skill_scope: 'general',
      allowed_tools: ['calculator'] as string[],
      allowed_worker_agents: [] as string[],
      preload_skill_packs: [] as string[],
      memory_scopes: ['conversation'] as string[],
      max_steps: 3,
      max_tool_calls: 3,
      allow_background_jobs: false,
      metadata: {},
      source_path: '/tmp/general.md',
      overlay_active: false,
      body: 'General body',
    },
    collectionCatalog: {
      default: {
        created_at: '2026-04-08T09:00:00Z',
        updated_at: '2026-04-08T10:00:00Z',
        maintenance_policy: 'configured_kb_sources',
      },
    } as Record<string, CollectionCatalogState>,
    collections: {
      default: [
        createDoc('default', 'default-notes.md', 'default collection note', {
          sourceType: 'kb',
          sourceDisplayPath: 'knowledge_base/default-notes.md',
        }),
      ],
    } as Record<string, DocState[]>,
    nextDocIndex: 1,
    skills: {
      'skill-existing': {
        skill_id: 'skill-existing',
        name: 'Existing Skill',
        agent_scope: 'general',
        graph_id: '',
        collection_id: '',
        body_markdown: '# Existing Skill\nagent_scope: general\n\n## Workflow\n\n- Existing step.\n',
        enabled: true,
        status: 'active',
        version: '1',
        version_parent: 'skill-existing',
        updated_at: '2026-04-08T10:00:00Z',
      },
    } as Record<string, SkillState>,
    nextSkillIndex: 1,
    graphs: {} as Record<string, GraphState>,
    graphRuns: {} as Record<string, Array<Record<string, unknown>>>,
    graphTuneRuns: {} as Record<string, Record<string, Record<string, unknown>>>,
    registeredSources: {} as Record<string, Record<string, unknown>>,
    sourceRuns: [] as Array<Record<string, unknown>>,
    accessPrincipals: [
      {
        principal_id: 'principal-user-alex',
        tenant_id: 'openwebui',
        principal_type: 'user',
        provider: 'email',
        external_id: '',
        email_normalized: 'alex@example.com',
        display_name: 'alex@example.com',
        metadata_json: {},
        active: true,
        created_at: '2026-04-08T10:00:00Z',
        updated_at: '2026-04-08T10:00:00Z',
      },
      {
        principal_id: 'principal-group-finance',
        tenant_id: 'openwebui',
        principal_type: 'group',
        provider: 'system',
        external_id: '',
        email_normalized: '',
        display_name: 'Finance Analysts',
        metadata_json: {},
        active: true,
        created_at: '2026-04-08T10:00:00Z',
        updated_at: '2026-04-08T10:00:00Z',
      },
    ] as Array<Record<string, unknown>>,
    accessMemberships: [] as Array<Record<string, unknown>>,
    accessRoles: [
      {
        role_id: 'role-finance',
        tenant_id: 'openwebui',
        name: 'Finance Analyst',
        description: 'Access to finance resources',
        created_at: '2026-04-08T10:00:00Z',
        updated_at: '2026-04-08T10:00:00Z',
      },
    ] as Array<Record<string, unknown>>,
    accessBindings: [] as Array<Record<string, unknown>>,
    accessPermissions: [] as Array<Record<string, unknown>>,
    lastReload: {
      status: 'success',
      timestamp: '2026-04-08T10:00:00Z',
      reason: 'startup',
      actor: 'system',
      changed_keys: [] as string[],
      error: '',
    },
    auditEvents: [] as Array<Record<string, unknown>>,
    collectionHealth: {
      default: {
        status: 'not_ready',
        reason: 'kb_duplicate_docs',
        tenant_id: 'openwebui',
        collection_id: 'default',
        maintenance_policy: 'configured_kb_sources',
        configured_source_count: 2,
        indexed_doc_count: 3,
        active_doc_count: 2,
        missing_sources: [] as string[],
        duplicate_group_count: 1,
        content_drift_count: 0,
        duplicate_groups: [
          {
            source_identity: 'path:/tmp/ARCHITECTURE.md',
            title: 'ARCHITECTURE.md',
            source_type: 'kb',
            collection_id: 'default',
            configured_source_path: '/tmp/ARCHITECTURE.md',
            active_doc_id: 'doc-arch-new',
            active_content_hash: 'hash-new',
            active_ingested_at: '2026-04-08T10:00:00Z',
            active_source_path: '/tmp/ARCHITECTURE.md',
            current_file_hash: 'hash-new',
            source_exists: true,
            content_drift: false,
            duplicate_doc_ids: ['doc-arch-old'],
            status: 'duplicate',
            records: [
              {
                doc_id: 'doc-arch-new',
                title: 'ARCHITECTURE.md',
                source_type: 'kb',
                source_path: '/tmp/ARCHITECTURE.md',
                collection_id: 'default',
                content_hash: 'hash-new',
                ingested_at: '2026-04-08T10:00:00Z',
                num_chunks: 4,
                file_type: 'md',
                doc_structure_type: 'process_flow_doc',
                active: true,
              },
              {
                doc_id: 'doc-arch-old',
                title: 'ARCHITECTURE.md',
                source_type: 'kb',
                source_path: '/tmp/ARCHITECTURE.md',
                collection_id: 'default',
                content_hash: 'hash-old',
                ingested_at: '2026-04-07T10:00:00Z',
                num_chunks: 3,
                file_type: 'md',
                doc_structure_type: 'process_flow_doc',
                active: false,
              },
            ],
          },
        ],
        drifted_groups: [] as Array<Record<string, unknown>>,
        source_groups: [] as Array<Record<string, unknown>>,
        sync_error: '',
        suggested_fix: 'python run.py repair-kb --collection-id default',
      },
    } as Record<string, Record<string, unknown>>,
  }

  state.collectionHealth.default.source_groups = state.collectionHealth.default.duplicate_groups as Array<Record<string, unknown>>

  const compatibilityMode = options.compatibilityMode ?? 'full'
  let architectureSnapshotCalls = 0
  let architectureActivityCalls = 0

  function ensureCollection(collectionId: string): CollectionCatalogState {
    const existing = state.collectionCatalog[collectionId]
    if (existing) {
      state.collections[collectionId] = state.collections[collectionId] ?? []
      return existing
    }
    const created = {
      created_at: '2026-04-08T10:00:00Z',
      updated_at: '2026-04-08T10:00:00Z',
      maintenance_policy: 'indexed_documents',
    }
    state.collectionCatalog[collectionId] = created
    state.collections[collectionId] = state.collections[collectionId] ?? []
    return created
  }

  function touchCollection(collectionId: string, updatedAt = '2026-04-08T10:00:00Z') {
    const collection = ensureCollection(collectionId)
    collection.updated_at = updatedAt
  }

  function collectionIds(): string[] {
    return Array.from(new Set([
      ...Object.keys(state.collectionCatalog),
      ...Object.keys(state.collections),
      ...Object.values(state.graphs).map(graph => graph.collection_id),
    ]))
      .filter(collectionId => {
        const docs = state.collections[collectionId] ?? []
        const hasNonUploadDocs = docs.some(doc => doc.source_type !== 'upload')
        const hasGraphs = graphIdsForCollection(collectionId).length > 0
        return hasNonUploadDocs || hasGraphs || docs.length === 0
      })
      .sort()
  }

  function graphIdsForCollection(collectionId: string): string[] {
    return Object.values(state.graphs)
      .filter(graph => graph.collection_id === collectionId)
      .map(graph => graph.graph_id)
  }

  function sourceTypeCounts(collectionId: string): Record<string, number> {
    return (state.collections[collectionId] ?? []).filter(doc => doc.source_type !== 'upload').reduce<Record<string, number>>((counts, doc) => {
      counts[doc.source_type] = (counts[doc.source_type] ?? 0) + 1
      return counts
    }, {})
  }

  function storageProfile(collectionId: string) {
    const graphIds = graphIdsForCollection(collectionId)
    return {
      vector_store_backend: 'pgvector',
      tables: graphIds.length > 0 ? ['documents', 'chunks', 'graph_indexes'] : ['documents', 'chunks'],
      embeddings_provider: 'openai',
      embedding_model: 'text-embedding-3-large',
      graph_embedding_model: graphIds.length > 0 ? 'text-embedding-3-small' : '',
      configured_embedding_dim: 3072,
      actual_embedding_dims: collectionId === 'default'
        ? { chunks: 1536, ...(graphIds.length > 0 ? { graph_indexes: 1536 } : {}) }
        : graphIds.length > 0
          ? { chunks: 3072, graph_indexes: 3072 }
          : { chunks: 3072 },
      mismatch_warnings: collectionId === 'default'
        ? ['chunks uses 1536 dimensions while the active embedding model expects 3072.']
        : [],
    }
  }

  function collectionSummaryPayload(collectionId: string) {
    const catalog = ensureCollection(collectionId)
    const docs = (state.collections[collectionId] ?? []).filter(doc => doc.source_type !== 'upload')
    const graphIds = graphIdsForCollection(collectionId)
    return {
      collection_id: collectionId,
      created_at: catalog.created_at,
      updated_at: catalog.updated_at,
      maintenance_policy: catalog.maintenance_policy,
      document_count: docs.length,
      source_type_counts: sourceTypeCounts(collectionId),
      latest_ingested_at: docs[docs.length - 1]?.ingested_at ?? '',
      graph_count: graphIds.length,
      graph_ids: graphIds,
      storage_profile: storageProfile(collectionId),
      status: {
        ready: true,
        reason: docs.length > 0 ? 'indexed' : 'empty_collection',
        collection_id: collectionId,
        missing_sources: [],
        indexed_doc_count: docs.length,
        active_doc_count: docs.length,
        duplicate_group_count: 0,
        content_drift_count: 0,
        suggested_fix: docs.length > 0 ? '' : `Upload or sync content into ${collectionId}.`,
      },
    }
  }

  function collectionsPayload() {
    return {
      collections: collectionIds().map(collectionSummaryPayload),
    }
  }

  function documentsPayload(collectionId: string) {
    return {
      documents: (state.collections[collectionId] ?? []).filter(doc => doc.source_type !== 'upload').map(doc => ({
        doc_id: doc.doc_id,
        title: doc.title,
        source_type: doc.source_type,
        source_path: doc.source_path,
        source_display_path: doc.source_display_path,
        collection_id: doc.collection_id,
        num_chunks: 1,
        ingested_at: doc.ingested_at,
        file_type: doc.file_type,
        doc_structure_type: doc.doc_structure_type,
      })),
    }
  }

  function collectionHealthPayload(collectionId: string) {
    const catalog = ensureCollection(collectionId)
    return state.collectionHealth[collectionId] ?? {
      status: 'ready',
      reason: 'ready',
      tenant_id: 'openwebui',
      collection_id: collectionId,
      maintenance_policy: catalog.maintenance_policy,
      configured_source_count: 0,
      indexed_doc_count: (state.collections[collectionId] ?? []).filter(doc => doc.source_type !== 'upload').length,
      active_doc_count: (state.collections[collectionId] ?? []).filter(doc => doc.source_type !== 'upload').length,
      missing_sources: [],
      duplicate_group_count: 0,
      content_drift_count: 0,
      duplicate_groups: [],
      drifted_groups: [],
      source_groups: [],
      sync_error: '',
      suggested_fix: `python run.py repair-kb --collection-id ${collectionId}`,
    }
  }

  function collectionOperationPayload(
    collectionId: string,
    files: Array<{
      displayPath: string
      sourceType: string
      outcome?: 'ingested' | 'already_indexed' | 'skipped' | 'failed'
      error?: string
      docIds?: string[]
      sourcePath?: string
    }>,
    options: {
      missingPaths?: string[]
    } = {},
  ) {
    const normalizedFiles = files.map(file => ({
      display_path: file.displayPath,
      filename: file.displayPath.split('/').pop() || file.displayPath,
      source_type: file.sourceType,
      source_path: file.sourcePath ?? '',
      outcome: file.outcome ?? 'ingested',
      error: file.error ?? '',
      doc_ids: file.docIds ?? [],
    }))
    const docIds = normalizedFiles.flatMap(file => file.doc_ids)
    const alreadyIndexedCount = normalizedFiles.filter(file => file.outcome === 'already_indexed').length
    const skippedCount = normalizedFiles.filter(file => file.outcome === 'skipped').length
    const failedCount = normalizedFiles.filter(file => file.outcome === 'failed').length
    const missingPaths = options.missingPaths ?? []
    const status = failedCount > 0 && docIds.length === 0
      ? 'failed'
      : failedCount > 0 || skippedCount > 0 || missingPaths.length > 0
        ? 'partial'
        : 'success'
    return {
      collection_id: collectionId,
      status,
      summary: {
        resolved_count: normalizedFiles.length,
        ingested_count: docIds.length,
        already_indexed_count: alreadyIndexedCount,
        skipped_count: skippedCount,
        failed_count: failedCount,
        missing_count: missingPaths.length,
      },
      resolved_count: normalizedFiles.length,
      ingested_count: docIds.length,
      already_indexed_count: alreadyIndexedCount,
      skipped_count: skippedCount,
      failed_count: failedCount,
      doc_ids: docIds,
      missing_paths: missingPaths,
      errors: normalizedFiles.filter(file => file.error).map(file => file.error),
      files: normalizedFiles,
      filenames: normalizedFiles.map(file => file.filename),
      display_paths: normalizedFiles.map(file => file.display_path),
      workspace_copies: normalizedFiles.map(file => file.display_path),
    }
  }

  function getDoc(collectionId: string, docId: string): DocState | undefined {
    return (state.collections[collectionId] ?? []).find(doc => doc.doc_id === docId)
  }

  function getUploadDoc(docId: string): DocState | undefined {
    return Object.values(state.collections).flat().find(doc => doc.doc_id === docId && doc.source_type === 'upload')
  }

  function uploadedFilesPayload() {
    return {
      uploads: Object.values(state.collections).flat()
        .filter(doc => doc.source_type === 'upload')
        .sort((left, right) => right.ingested_at.localeCompare(left.ingested_at) || left.title.localeCompare(right.title))
        .map(doc => ({
          doc_id: doc.doc_id,
          title: doc.title,
          source_type: doc.source_type,
          source_path: doc.source_path,
          source_display_path: doc.source_display_path,
          collection_id: doc.collection_id,
          num_chunks: 1,
          ingested_at: doc.ingested_at,
          file_type: doc.file_type,
          doc_structure_type: doc.doc_structure_type,
        })),
    }
  }

  function upsertCollectionDoc(
    collectionId: string,
    title: string,
    content: string,
    options: {
      sourcePath?: string
      sourceDisplayPath?: string
      sourceType?: string
    } = {},
  ) {
    ensureCollection(collectionId)
    const docs = state.collections[collectionId] ?? []
    const docId = `${collectionId}-${slugify(title)}-${state.nextDocIndex++}`
    const nextDoc = createDoc(collectionId, title, content, {
      docId,
      sourcePath: options.sourcePath,
      sourceDisplayPath: options.sourceDisplayPath,
      sourceType: options.sourceType,
    })
    state.collections[collectionId] = [...docs, nextDoc]
    touchCollection(collectionId, nextDoc.ingested_at)
    return nextDoc
  }

  function listSkillsPayload() {
    return {
      object: 'list',
      data: Object.values(state.skills).sort((left, right) => left.name.localeCompare(right.name)),
    }
  }

  function listGraphsPayload() {
    return {
      graphs: Object.values(state.graphs).sort((left, right) => left.display_name.localeCompare(right.display_name)),
    }
  }

  function graphDetailPayload(graphId: string) {
    const graph = state.graphs[graphId]
    if (!graph) return null
    const skills = Object.values(state.skills)
      .filter(skill => graph.graph_skill_ids.includes(skill.skill_id) || skill.graph_id === graphId)
      .map(skill => ({
        skill_id: skill.skill_id,
        name: skill.name,
        agent_scope: skill.agent_scope,
        graph_id: skill.graph_id ?? '',
        collection_id: skill.collection_id ?? '',
        version: skill.version,
        enabled: skill.enabled,
        status: skill.status,
        visibility: 'tenant',
        version_parent: skill.version_parent,
      }))
    const sources = graph.source_doc_ids.map((docId, index) => {
      const doc = getDoc(graph.collection_id, docId)
      return {
        graph_source_id: `${graphId}-source-${index + 1}`,
        graph_id: graphId,
        source_doc_id: docId,
        source_path: doc?.source_path ?? '',
        source_title: doc?.title ?? docId,
        source_type: doc?.source_type ?? 'upload',
      }
    })
    return {
      graph: {
        ...graph,
        tenant_id: 'openwebui',
        visibility: 'tenant',
        artifact_tables: graph.query_ready ? ['entities', 'relationships', 'text_units'] : [],
        graph_context_summary: { communities: graph.query_ready ? 3 : 0 },
      },
      sources,
      runs: state.graphRuns[graphId] ?? [],
      logs: graph.logs,
      skills,
    }
  }

  function graphResearchTunePayload(
    graphId: string,
    runId: string,
    guidance: string,
    targetPromptFiles: string[],
  ) {
    const graph = state.graphs[graphId]
    const docs = graph?.source_doc_ids ?? []
    const promptDrafts: Record<string, Record<string, unknown>> = {}
    const promptDiffs: Record<string, Record<string, unknown>> = {}
    for (const promptFile of targetPromptFiles.length > 0 ? targetPromptFiles : ['extract_graph.txt']) {
      promptDrafts[promptFile] = {
        prompt_file: promptFile,
        baseline_source: graph?.prompt_overrides_json?.[promptFile] ? 'graph_prompt_override' : 'graphrag_default',
        summary: `Dataset-tailored draft for ${promptFile}.`,
        warnings: [],
        validation: { ok: true, warnings: [] },
        content: [
          `Tuned ${promptFile} for vendor risk knowledge graph extraction.`,
          '',
          'Dataset-Specific Curation Guidance',
          `- ${guidance || 'Model vendor ownership, approval, control, and exception relationships.'}`,
          '- Preserve GraphRAG placeholders such as {entity_types} and {input_text}.',
        ].join('\n'),
      }
      promptDiffs[promptFile] = {
        prompt_file: promptFile,
        diff: [
          '--- baseline',
          '+++ draft',
          '@@',
          '-Use vendor-centric extraction.',
          '+Tuned vendor-risk extraction with approval and control relationships.',
        ].join('\n'),
      }
    }
    return {
      run_id: runId,
      graph_id: graphId,
      status: 'completed',
      detail: 'Research & Tune completed. Review generated prompt drafts before applying.',
      artifact_dir: `/tmp/${graphId}/tuning/${runId}`,
      manifest_path: `/tmp/${graphId}/tuning/${runId}/manifest.json`,
      scratchpad_path: `/tmp/${graphId}/tuning/${runId}/scratchpad.md`,
      scratchpad_preview: [
        '# Research & Tune Scratchpad',
        '',
        'Vendor risk documents emphasize supplier ownership, financial approval, critical controls, exceptions, and escalation paths.',
      ].join('\n'),
      coverage: {
        digested_doc_count: docs.length,
        resolved_source_count: docs.length,
        coverage_state: 'complete',
      },
      warnings: [],
      corpus_profile: {
        corpus_summary: 'Vendor risk corpus summary with controls, approvers, exceptions, and supplier aliases.',
        candidate_entity_types: ['Vendor', 'Risk', 'Control', 'Approver'],
        candidate_relationship_types: ['OWNS', 'APPROVES', 'MITIGATES', 'ESCALATES_TO'],
      },
      doc_digests: docs.map(docId => ({
        doc_id: docId,
        summary: `Digest for ${docId}.`,
        prompt_implications: ['Track approval and exception relationships.'],
      })),
      prompt_drafts: promptDrafts,
      prompt_diffs: promptDiffs,
      manifest: {
        run_id: runId,
        graph_id: graphId,
        status: 'completed',
        target_prompt_files: targetPromptFiles,
        applied_prompt_files: [],
      },
    }
  }

  function architecturePayload() {
    const skillStoreEnabled = state.agent.preload_skill_packs.length > 0
    const nodes = [
      {
        id: 'entry-user',
        label: 'User',
        kind: 'entry',
        layer: 'entry',
        description: 'Incoming chat request.',
        status: 'active',
        badges: ['Requests'],
      },
      {
        id: 'entry-api-gateway',
        label: 'API Gateway',
        kind: 'gateway',
        layer: 'entry',
        description: 'HTTP and session entry point.',
        status: 'configured',
        badges: ['gateway-local'],
      },
      {
        id: 'router-core',
        label: 'Router',
        kind: 'router',
        layer: 'routing',
        description: 'Chooses BASIC vs AGENT.',
        status: 'configured',
        badges: ['Hybrid'],
      },
      {
        id: 'agent-basic',
        label: 'basic',
        kind: 'agent',
        layer: 'top_level',
        description: 'Basic chat path.',
        status: 'configured',
        mode: 'basic',
        role_kind: 'top_level',
        entry_path: 'router_basic',
        prompt_file: 'basic_chat.md',
        overlay_active: false,
        allowed_tools: [],
        allowed_worker_agents: [],
        preload_skill_packs: [],
        memory_scopes: [],
        badges: ['Basic'],
      },
      {
        id: 'agent-general',
        label: state.agent.name,
        kind: 'agent',
        layer: 'top_level',
        description: state.agent.description,
        status: state.agent.overlay_active ? 'overlay active' : 'configured',
        mode: state.agent.mode,
        role_kind: 'top_level',
        entry_path: 'default',
        prompt_file: state.agent.prompt_file,
        overlay_active: state.agent.overlay_active,
        allowed_tools: state.agent.allowed_tools,
        allowed_worker_agents: state.agent.allowed_worker_agents,
        preload_skill_packs: state.agent.preload_skill_packs,
        memory_scopes: state.agent.memory_scopes,
        badges: [
          ...(state.agent.overlay_active ? ['Overlay active'] : []),
          ...(state.agent.allowed_worker_agents.length > 0 ? ['Worker-capable'] : []),
        ],
      },
      {
        id: 'agent-rag-worker',
        label: 'rag_worker',
        kind: 'agent',
        layer: 'top_level',
        description: 'Grounded retrieval specialist.',
        status: 'configured',
        mode: 'rag',
        role_kind: 'top_level_or_worker',
        entry_path: '',
        prompt_file: 'grounded_answer.txt',
        overlay_active: false,
        allowed_tools: ['list_indexed_docs'],
        allowed_worker_agents: [],
        preload_skill_packs: [],
        memory_scopes: [],
        badges: ['RAG'],
      },
      {
        id: 'service-knowledge-base',
        label: 'Knowledge Base',
        kind: 'service',
        layer: 'services',
        description: 'Grounded document storage and retrieval.',
        status: 'active',
        badges: ['Grounded'],
      },
      {
        id: 'service-skill-store',
        label: 'Skill Store',
        kind: 'service',
        layer: 'services',
        description: 'Pinned and retrieved skills.',
        status: 'active',
        badges: ['Pinned skills'],
      },
    ]
    const edges = [
      { id: 'edge-entry-user-api', source: 'entry-user', target: 'entry-api-gateway', kind: 'request_flow', label: 'send request', emphasis: 'high' },
      { id: 'edge-entry-api-router', source: 'entry-api-gateway', target: 'router-core', kind: 'request_flow', label: 'route turn', emphasis: 'high' },
      { id: 'edge-router-basic-basic', source: 'router-core', target: 'agent-basic', kind: 'routing_path', label: 'BASIC', emphasis: 'high' },
      { id: 'edge-router-default-general', source: 'router-core', target: 'agent-general', kind: 'routing_path', label: 'Default AGENT', emphasis: 'high' },
      { id: 'edge-router-rag-rag_worker', source: 'router-core', target: 'agent-rag-worker', kind: 'routing_path', label: 'Grounded lookup', emphasis: 'normal' },
      { id: 'edge-agent-general-service-skill-store', source: 'agent-general', target: 'service-skill-store', kind: 'service_dependency', label: 'Skill Store', emphasis: 'normal' },
      { id: 'edge-agent-rag-worker-service-knowledge-base', source: 'agent-rag-worker', target: 'service-knowledge-base', kind: 'service_dependency', label: 'Knowledge Base', emphasis: 'normal' },
      ...(skillStoreEnabled
        ? []
        : []),
    ]
    const canonical_paths = [
      {
        id: 'basic',
        label: 'Basic response',
        route: 'BASIC',
        summary: 'General questions stay on the lightweight path.',
        when: 'No tool or grounding signals are present.',
        target_agent: 'basic',
        badges: ['Fast path'],
        node_ids: ['entry-user', 'entry-api-gateway', 'router-core', 'agent-basic'],
        edge_ids: ['edge-entry-user-api', 'edge-entry-api-router', 'edge-router-basic-basic'],
      },
      {
        id: 'default-agent',
        label: 'Default agent',
        route: 'AGENT',
        summary: 'The standard agent path for normal tool-capable work.',
        when: 'The router chooses AGENT without a specialist hint.',
        target_agent: state.agent.name,
        badges: ['Default start'],
        node_ids: ['entry-user', 'entry-api-gateway', 'router-core', 'agent-general'],
        edge_ids: ['edge-entry-user-api', 'edge-entry-api-router', 'edge-router-default-general'],
      },
      {
        id: 'grounded-lookup',
        label: 'Grounded lookup',
        route: 'AGENT',
        summary: 'Focused citation and retrieval requests start in the grounded specialist.',
        when: 'The router sees document-grounding intent.',
        target_agent: 'rag_worker',
        badges: ['RAG', 'Documents'],
        node_ids: ['entry-user', 'entry-api-gateway', 'router-core', 'agent-rag-worker', 'service-knowledge-base'],
        edge_ids: ['edge-entry-user-api', 'edge-entry-api-router', 'edge-router-rag-rag_worker', 'edge-agent-rag-worker-service-knowledge-base'],
      },
    ]
    return {
      generated_at: '2026-04-08T10:00:00Z',
      system: {
        gateway_model_id: 'gateway-local',
        counts: { agents: 3, services: 2, edges: edges.length, overlays: state.agent.overlay_active ? 1 : 0 },
      },
      router: {
        mode_label: 'Hybrid',
        default_agent: state.agent.name,
        basic_agent: 'basic',
        coordinator_agent: 'coordinator',
        data_analyst_agent: 'data_analyst',
        rag_agent: 'rag_worker',
      },
      nodes,
      edges,
      canonical_paths,
      langgraph: options.langGraphUnavailable ? {
        status: 'unavailable',
        generated_at: '2026-04-08T10:00:00Z',
        agent_name: state.agent.name,
        mermaid: '',
        nodes: [],
        edges: [],
        warnings: ['No chat provider available for graph export.'],
      } : {
        status: 'available',
        generated_at: '2026-04-08T10:00:00Z',
        agent_name: state.agent.name,
        mermaid: 'graph TD\n  __start__ --> agent\n  agent --> tools\n  tools --> agent\n  agent --> __end__',
        nodes: [
          { id: '__start__', name: '__start__', data_type: 'RunnableCallable', metadata: {} },
          { id: 'agent', name: 'agent', data_type: 'RunnableCallable', metadata: {} },
          { id: 'tools', name: 'tools', data_type: 'PolicyAwareToolNode', metadata: {} },
          { id: '__end__', name: '__end__', data_type: 'RunnableCallable', metadata: {} },
        ],
        edges: [
          { id: 'langgraph-edge-1', source: '__start__', target: 'agent', conditional: false, data: null },
          { id: 'langgraph-edge-2', source: 'agent', target: 'tools', conditional: true, data: 'tools_condition' },
          { id: 'langgraph-edge-3', source: 'tools', target: 'agent', conditional: false, data: null },
        ],
        warnings: [],
      },
    }
  }

  function architectureActivityPayload() {
    return {
      route_counts: { BASIC: 1, AGENT: 2 },
      router_method_counts: { hybrid: 2, deterministic: 1 },
      start_agent_counts: { basic: 1, general: 1, rag_worker: 1 },
      delegation_counts: state.agent.allowed_worker_agents.length > 0 ? { [state.agent.allowed_worker_agents[0]]: 1 } : {},
      outcome_counts: { positive: 1, negative: 1, neutral: 1 },
      negative_rate_by_route: { AGENT: 0.5, BASIC: 0.0 },
      negative_rate_by_router_method: { hybrid: 0.5, deterministic: 0.0 },
      recent_mispicks: [
        {
          sample_id: 'rrs_1',
          router_decision_id: 'rtd_1',
          route: 'AGENT',
          router_method: 'hybrid',
          suggested_agent: 'rag_worker',
          outcome_label: 'negative',
          evidence_signals: ['manual_agent_override'],
          created_at: '2026-04-08T10:05:00Z',
        },
      ],
      review_backlog: { pending: 1, total_samples: 1, negative_samples: 1, neutral_samples: 0 },
      last_retrain_report: { quarter: '2026-Q2', generated_at: '2026-04-08T10:06:00Z', recommended_threshold: 0.75 },
      recent_flows: [
        {
          session_id: 'tenant-1:user-1:conv-1',
          conversation_id: 'conv-1',
          route: 'AGENT',
          router_method: 'hybrid',
          start_agent: state.agent.name,
          suggested_agent: '',
          reasons: ['tool_or_multistep_intent'],
          worker_agents: state.agent.allowed_worker_agents,
          degraded: false,
          degraded_events: [],
          updated_at: '2026-04-08T10:02:00Z',
        },
        {
          session_id: 'tenant-1:user-1:conv-2',
          conversation_id: 'conv-2',
          route: 'AGENT',
          router_method: 'hybrid',
          start_agent: 'rag_worker',
          suggested_agent: 'rag_worker',
          reasons: ['document_grounding_intent'],
          worker_agents: [],
          degraded: true,
          degraded_events: ['router_degraded_to_deterministic'],
          updated_at: '2026-04-08T10:04:00Z',
        },
      ],
      updated_at: '2026-04-08T10:04:00Z',
    }
  }

  function openApiPathsPayload() {
    const paths: Record<string, unknown> = {
      '/v1/admin/overview': {},
      '/v1/admin/operations': {},
      '/v1/admin/config/schema': {},
      '/v1/admin/config/effective': {},
      '/v1/admin/agents': {},
      '/v1/admin/prompts': {},
      '/v1/admin/collections': {},
      '/v1/admin/uploads': {},
      '/v1/admin/graphs': {},
      '/v1/admin/graphs/{graph_id}': {},
      '/v1/admin/access/principals': {},
      '/v1/admin/access/roles': {},
      '/v1/admin/access/effective-access': {},
      '/v1/admin/mcp/connections': {},
      '/v1/skills': {},
    }
    if (compatibilityMode !== 'partial-architecture') {
      paths['/v1/admin/architecture'] = {}
      paths['/v1/admin/architecture/activity'] = {}
    }
    if (compatibilityMode !== 'no-endpoint') {
      paths['/v1/admin/capabilities'] = {}
    }
    return { paths }
  }

  function capabilitiesPayload() {
    const architectureSupported = compatibilityMode !== 'partial-architecture'
    return {
      schema_version: '1',
      contract_version: 'control-panel-v1',
      compatible: architectureSupported,
      generated_at: '2026-04-08T10:00:00Z',
      sections: {
        dashboard: { supported: true, required_routes: ['/v1/admin/overview'], missing_routes: [], reason: '' },
        architecture: {
          supported: architectureSupported,
          required_routes: ['/v1/admin/architecture', '/v1/admin/architecture/activity'],
          missing_routes: architectureSupported ? [] : ['/v1/admin/architecture', '/v1/admin/architecture/activity'],
          reason: architectureSupported ? '' : 'Running backend is missing one or more required routes for this section.',
        },
        config: { supported: true, required_routes: ['/v1/admin/config/schema', '/v1/admin/config/effective'], missing_routes: [], reason: '' },
        agents: { supported: true, required_routes: ['/v1/admin/agents'], missing_routes: [], reason: '' },
        prompts: { supported: true, required_routes: ['/v1/admin/prompts'], missing_routes: [], reason: '' },
        collections: { supported: true, required_routes: ['/v1/admin/collections'], missing_routes: [], reason: '' },
        uploads: { supported: true, required_routes: ['/v1/admin/uploads'], missing_routes: [], reason: '' },
        graphs: { supported: true, required_routes: ['/v1/admin/graphs', '/v1/admin/graphs/{graph_id}'], missing_routes: [], reason: '' },
        skills: { supported: true, required_routes: ['/v1/skills'], missing_routes: [], reason: '' },
        access: { supported: true, required_routes: ['/v1/admin/access/principals', '/v1/admin/access/roles', '/v1/admin/access/effective-access'], missing_routes: [], reason: '' },
        mcp: { supported: true, required_routes: ['/v1/admin/mcp/connections'], missing_routes: [], reason: '' },
        operations: { supported: true, required_routes: ['/v1/admin/operations'], missing_routes: [], reason: '' },
      },
    }
  }

  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = requestPath(input)
    const method = requestMethod(input, init)

    if (path === '/openapi.json' && method === 'GET') {
      return jsonResponse(openApiPathsPayload())
    }

    if (path === '/v1/admin/capabilities' && method === 'GET') {
      if (compatibilityMode === 'no-endpoint') {
        return jsonResponse({ detail: 'Not Found' }, 404)
      }
      return jsonResponse(capabilitiesPayload())
    }

    if (path === '/v1/admin/overview' && method === 'GET') {
      return jsonResponse({
        status: 'ok',
        gateway_model_id: 'gateway-local',
        providers: {
          llm_provider: 'ollama',
          judge_provider: 'ollama',
          embeddings_provider: 'ollama',
        },
        models: {
          ollama_chat_model: 'gpt-oss:20b',
          ollama_judge_model: 'gpt-oss:20b',
          ollama_embed_model: 'nomic-embed-text',
        },
        counts: {
          collections: collectionsPayload().collections.length,
          agents: 1,
          skills: Object.keys(state.skills).length,
          tools: 1,
          jobs: 0,
        },
        collections: collectionsPayload().collections,
        agents: [{ name: state.agent.name, mode: state.agent.mode, prompt_file: state.agent.prompt_file, overlay_active: state.agent.overlay_active }],
        jobs: [],
        last_reload: state.lastReload,
        audit_events: state.auditEvents,
      })
    }

    if (path === '/v1/admin/operations' && method === 'GET') {
      if (options.operationsError) {
        return jsonResponse({ detail: options.operationsError }, 503)
      }
      return jsonResponse({
        last_reload: state.lastReload,
        scheduler: {
          enabled: true,
          running_jobs: 1,
          available_slots: 3,
          reserved_urgent_slots: 1,
          urgent_backlog: false,
          queue_depths: { urgent: 0, interactive: 1, background: 0 },
          oldest_wait_seconds: { urgent: 0, interactive: 4, background: 0 },
          budget_blocked_jobs: 0,
          tenant_budget_health: [
            { tenant_id: 'tenant-1', queued_jobs: 1, running_jobs: 1, budget_blocked_jobs: 0, available_tokens: 42000 },
          ],
        },
        jobs: [
          {
            job_id: 'job-1',
            agent_name: 'general',
            status: 'completed',
            scheduler_state: 'completed',
            priority: 'interactive',
            queue_class: 'interactive',
            estimated_token_cost: 320,
            actual_token_cost: 288,
            description: 'Smoke job',
            updated_at: '2026-04-08T10:00:00Z',
          },
        ],
        audit_events: state.auditEvents,
      })
    }

    if (path === '/v1/admin/access/principals' && method === 'GET') {
      return jsonResponse({ principals: state.accessPrincipals })
    }

    if (path === '/v1/admin/access/principals' && method === 'POST') {
      const body = readJsonBody(init)
      const principalType = String(body.principal_type ?? 'user')
      const email = String(body.email_normalized ?? '')
      const displayName = String(body.display_name ?? email)
      const principalId = String(body.principal_id ?? (
        principalType === 'group'
          ? `principal-group-${slugify(displayName)}`
          : `principal-user-${slugify(email || displayName)}`
      ))
      const principal = {
        principal_id: principalId,
        tenant_id: 'openwebui',
        principal_type: principalType,
        provider: String(body.provider ?? (principalType === 'group' ? 'system' : 'email')),
        external_id: String(body.external_id ?? ''),
        email_normalized: email,
        display_name: displayName,
        metadata_json: (body.metadata_json ?? {}) as Record<string, unknown>,
        active: body.active !== false,
        created_at: '2026-04-08T10:00:00Z',
        updated_at: '2026-04-08T10:00:00Z',
      }
      state.accessPrincipals = state.accessPrincipals.filter(existing => existing.principal_id !== principalId)
      state.accessPrincipals.push(principal)
      return jsonResponse({ principal })
    }

    if (path === '/v1/admin/access/memberships' && method === 'GET') {
      return jsonResponse({ memberships: state.accessMemberships })
    }

    if (path === '/v1/admin/access/memberships' && method === 'POST') {
      const body = readJsonBody(init)
      const membership = {
        membership_id: String(body.membership_id ?? `membership-${slugify(String(body.parent_principal_id ?? 'group'))}-${slugify(String(body.child_principal_id ?? 'member'))}-${state.accessMemberships.length + 1}`),
        tenant_id: 'openwebui',
        parent_principal_id: String(body.parent_principal_id ?? ''),
        child_principal_id: String(body.child_principal_id ?? ''),
        created_at: '2026-04-08T10:00:00Z',
      }
      state.accessMemberships = state.accessMemberships.filter(existing => existing.membership_id !== membership.membership_id)
      state.accessMemberships.push(membership)
      return jsonResponse({ membership })
    }

    if (path.startsWith('/v1/admin/access/memberships/') && method === 'DELETE') {
      const membershipId = path.split('/').pop() ?? ''
      state.accessMemberships = state.accessMemberships.filter(membership => membership.membership_id !== membershipId)
      return jsonResponse({ deleted: true, membership_id: membershipId })
    }

    if (path === '/v1/admin/access/roles' && method === 'GET') {
      return jsonResponse({ roles: state.accessRoles })
    }

    if (path === '/v1/admin/access/roles' && method === 'POST') {
      const body = readJsonBody(init)
      const name = String(body.name ?? 'Role')
      const role = {
        role_id: String(body.role_id ?? `role-${slugify(name)}-${state.accessRoles.length + 1}`),
        tenant_id: 'openwebui',
        name,
        description: String(body.description ?? ''),
        created_at: '2026-04-08T10:00:00Z',
        updated_at: '2026-04-08T10:00:00Z',
      }
      state.accessRoles = state.accessRoles.filter(existing => existing.role_id !== role.role_id)
      state.accessRoles.push(role)
      return jsonResponse({ role })
    }

    if (path.startsWith('/v1/admin/access/roles/') && method === 'DELETE') {
      const roleId = path.split('/').pop() ?? ''
      state.accessRoles = state.accessRoles.filter(role => role.role_id !== roleId)
      state.accessBindings = state.accessBindings.filter(binding => binding.role_id !== roleId)
      state.accessPermissions = state.accessPermissions.filter(permission => permission.role_id !== roleId)
      return jsonResponse({ deleted: true, role_id: roleId })
    }

    if (path === '/v1/admin/access/bindings' && method === 'GET') {
      return jsonResponse({ bindings: state.accessBindings })
    }

    if (path === '/v1/admin/access/bindings' && method === 'POST') {
      const body = readJsonBody(init)
      const binding = {
        binding_id: String(body.binding_id ?? `binding-${slugify(String(body.role_id ?? 'role'))}-${slugify(String(body.principal_id ?? 'principal'))}-${state.accessBindings.length + 1}`),
        tenant_id: 'openwebui',
        role_id: String(body.role_id ?? ''),
        principal_id: String(body.principal_id ?? ''),
        created_at: '2026-04-08T10:00:00Z',
        disabled_at: body.disabled ? '2026-04-08T10:00:00Z' : '',
        disabled: Boolean(body.disabled),
      }
      state.accessBindings = state.accessBindings.filter(existing => existing.binding_id !== binding.binding_id)
      state.accessBindings.push(binding)
      return jsonResponse({ binding })
    }

    if (path.startsWith('/v1/admin/access/bindings/') && method === 'DELETE') {
      const bindingId = path.split('/').pop() ?? ''
      state.accessBindings = state.accessBindings.filter(binding => binding.binding_id !== bindingId)
      return jsonResponse({ deleted: true, binding_id: bindingId })
    }

    if (path === '/v1/admin/access/permissions' && method === 'GET') {
      return jsonResponse({ permissions: state.accessPermissions })
    }

    if (path === '/v1/admin/access/permissions' && method === 'POST') {
      const body = readJsonBody(init)
      const permission = {
        permission_id: String(body.permission_id ?? `permission-${slugify(String(body.role_id ?? 'role'))}-${slugify(String(body.resource_type ?? 'resource'))}-${slugify(String(body.action ?? 'use'))}-${state.accessPermissions.length + 1}`),
        tenant_id: 'openwebui',
        role_id: String(body.role_id ?? ''),
        resource_type: String(body.resource_type ?? 'collection'),
        action: String(body.action ?? 'use'),
        resource_selector: String(body.resource_selector ?? '*'),
        created_at: '2026-04-08T10:00:00Z',
      }
      state.accessPermissions = state.accessPermissions.filter(existing => existing.permission_id !== permission.permission_id)
      state.accessPermissions.push(permission)
      return jsonResponse({ permission })
    }

    if (path.startsWith('/v1/admin/access/permissions/') && method === 'DELETE') {
      const permissionId = path.split('/').pop() ?? ''
      state.accessPermissions = state.accessPermissions.filter(permission => permission.permission_id !== permissionId)
      return jsonResponse({ deleted: true, permission_id: permissionId })
    }

    if (path.startsWith('/v1/admin/access/effective-access') && method === 'GET') {
      const email = new URL(`http://test.local${path}`).searchParams.get('email') ?? ''
      return jsonResponse({
        email,
        access: {
          authz_enabled: true,
          user_email: email,
          role_ids: state.accessRoles.map(role => String(role.role_id)),
          resources: {
            collection: { use: ['default'], manage: [], use_all: false, manage_all: false },
            graph: { use: [], manage: [], use_all: false, manage_all: false },
            tool: { use: [], manage: [], use_all: false, manage_all: false },
            skill_family: { use: [], manage: [], use_all: false, manage_all: false },
          },
        },
      })
    }

    if (path === '/v1/admin/architecture' && method === 'GET') {
      architectureSnapshotCalls += 1
      if (options.architectureSnapshotFailureCalls?.includes(architectureSnapshotCalls)) {
        return jsonResponse({ detail: 'HTTP 500' }, 500)
      }
      if (compatibilityMode === 'partial-architecture' || compatibilityMode === 'no-endpoint') {
        return jsonResponse({ detail: 'Not Found' }, 404)
      }
      return jsonResponse(architecturePayload())
    }

    if (path === '/v1/admin/architecture/activity' && method === 'GET') {
      architectureActivityCalls += 1
      if (options.architectureActivityFailureCalls?.includes(architectureActivityCalls)) {
        return jsonResponse({ detail: 'HTTP 500' }, 500)
      }
      if (compatibilityMode === 'partial-architecture' || compatibilityMode === 'no-endpoint') {
        return jsonResponse({ detail: 'Not Found' }, 404)
      }
      return jsonResponse(architectureActivityPayload())
    }

    if (path === '/v1/admin/config/schema' && method === 'GET') {
      return jsonResponse({
        fields: [
          {
            env_name: 'MAX_AGENT_STEPS',
            label: 'Max Agent Steps',
            group: 'Runtime',
            description: 'Global default max steps.',
            kind: 'int',
            choices: [],
            secret: false,
            readonly: false,
            reload_scope: 'runtime_swap',
            value: state.maxAgentSteps,
            is_configured: true,
          },
          {
            env_name: 'CLARIFICATION_SENSITIVITY',
            label: 'Clarification Sensitivity',
            group: 'Runtime',
            description: 'How readily the runtime asks clarifying questions for soft ambiguity.',
            kind: 'int',
            choices: [],
            secret: false,
            readonly: false,
            reload_scope: 'runtime_swap',
            ui_control: 'slider',
            min_value: 0,
            max_value: 100,
            step: 5,
            value: state.clarificationSensitivity,
            is_configured: true,
          },
          {
            env_name: 'OLLAMA_CHAT_MODEL',
            label: 'Ollama Chat Model',
            group: 'Providers',
            description: 'Primary Ollama chat model used by the runtime.',
            kind: 'enum',
            choices: ['gpt-oss:20b', 'gpt-oss:120b'],
            secret: false,
            readonly: false,
            reload_scope: 'runtime_swap',
            value: state.ollamaChatModel,
            is_configured: true,
          },
        ],
      })
    }

    if (path === '/v1/admin/config/effective' && method === 'GET') {
      return jsonResponse({
        values: {
          MAX_AGENT_STEPS: state.maxAgentSteps,
          CLARIFICATION_SENSITIVITY: state.clarificationSensitivity,
          OLLAMA_CHAT_MODEL: state.ollamaChatModel,
        },
        overlay_values: {},
      })
    }

    if (path === '/v1/admin/config/validate' && method === 'POST') {
      const body = readJsonBody(init)
      const changes = (body.changes ?? {}) as Record<string, string>
      const normalizedChanges: Record<string, string> = {}
      const previewDiff: Record<string, { before: string; after: string }> = {}
      if (changes.MAX_AGENT_STEPS !== undefined) {
        const after = String(changes.MAX_AGENT_STEPS)
        normalizedChanges.MAX_AGENT_STEPS = after
        previewDiff.MAX_AGENT_STEPS = {
          before: state.maxAgentSteps,
          after,
        }
      }
      if (changes.CLARIFICATION_SENSITIVITY !== undefined) {
        const after = String(changes.CLARIFICATION_SENSITIVITY)
        normalizedChanges.CLARIFICATION_SENSITIVITY = after
        previewDiff.CLARIFICATION_SENSITIVITY = {
          before: state.clarificationSensitivity,
          after,
        }
      }
      return jsonResponse({
        valid: true,
        normalized_changes: normalizedChanges,
        preview_diff: previewDiff,
        reload_scope: 'runtime_swap',
      })
    }

    if (path === '/v1/admin/config/apply' && method === 'POST') {
      const body = readJsonBody(init)
      const changes = (body.changes ?? {}) as Record<string, string>
      const normalizedChanges: Record<string, string> = {}
      const previewDiff: Record<string, { before: string; after: string }> = {}
      const changedKeys: string[] = []
      if (changes.MAX_AGENT_STEPS !== undefined) {
        const before = state.maxAgentSteps
        state.maxAgentSteps = String(changes.MAX_AGENT_STEPS)
        normalizedChanges.MAX_AGENT_STEPS = state.maxAgentSteps
        previewDiff.MAX_AGENT_STEPS = {
          before,
          after: state.maxAgentSteps,
        }
        changedKeys.push('MAX_AGENT_STEPS')
      }
      if (changes.CLARIFICATION_SENSITIVITY !== undefined) {
        const before = state.clarificationSensitivity
        state.clarificationSensitivity = String(changes.CLARIFICATION_SENSITIVITY)
        normalizedChanges.CLARIFICATION_SENSITIVITY = state.clarificationSensitivity
        previewDiff.CLARIFICATION_SENSITIVITY = {
          before,
          after: state.clarificationSensitivity,
        }
        changedKeys.push('CLARIFICATION_SENSITIVITY')
      }
      state.lastReload = {
        ...state.lastReload,
        reason: 'config_apply',
        changed_keys: changedKeys,
      }
      return jsonResponse({
        valid: true,
        applied: true,
        normalized_changes: normalizedChanges,
        preview_diff: previewDiff,
        reload_scope: 'runtime_swap',
        reload: state.lastReload,
      })
    }

    if (path === '/v1/admin/agents' && method === 'GET') {
      return jsonResponse({
        agents: [{
          name: state.agent.name,
          mode: state.agent.mode,
          prompt_file: state.agent.prompt_file,
          overlay_active: state.agent.overlay_active,
        }],
        tools: [{
          name: 'calculator',
          group: 'core',
          description: 'Simple math',
          read_only: true,
          destructive: false,
          background_safe: true,
          requires_workspace: false,
          concurrency_key: '',
          serializer: 'json',
          metadata: {},
        }],
      })
    }

    if (path === '/v1/admin/agents/general' && method === 'GET') {
      return jsonResponse({
        ...state.agent,
        pinned_skills: state.agent.preload_skill_packs.map(skillId => state.skills[skillId]).filter(Boolean),
        overlay_markdown: state.agent.overlay_active ? 'overlay' : '',
      })
    }

    if (path === '/v1/admin/agents/general' && method === 'PUT') {
      const body = readJsonBody(init)
      state.agent = {
        ...state.agent,
        description: String(body.description ?? state.agent.description),
        prompt_file: String(body.prompt_file ?? state.agent.prompt_file),
        skill_scope: String(body.skill_scope ?? state.agent.skill_scope),
        allowed_tools: Array.isArray(body.allowed_tools) ? body.allowed_tools.map(String) : state.agent.allowed_tools,
        allowed_worker_agents: Array.isArray(body.allowed_worker_agents) ? body.allowed_worker_agents.map(String) : state.agent.allowed_worker_agents,
        preload_skill_packs: Array.isArray(body.preload_skill_packs) ? body.preload_skill_packs.map(String) : state.agent.preload_skill_packs,
        memory_scopes: Array.isArray(body.memory_scopes) ? body.memory_scopes.map(String) : state.agent.memory_scopes,
        max_steps: Number(body.max_steps ?? state.agent.max_steps),
        max_tool_calls: Number(body.max_tool_calls ?? state.agent.max_tool_calls),
        body: String(body.body ?? state.agent.body),
        overlay_active: true,
      }
      return jsonResponse({ saved: true, pending_reload: true })
    }

    if (path === '/v1/admin/agents/reload' && method === 'POST') {
      state.lastReload = {
        ...state.lastReload,
        reason: 'agent_reload',
        changed_keys: ['agent_overlay'],
      }
      return jsonResponse(state.lastReload)
    }

    if (path === '/v1/admin/prompts' && method === 'GET') {
      return jsonResponse({
        prompts: [{ prompt_file: 'general_agent.md', overlay_active: Boolean(state.promptOverlay) }],
      })
    }

    if (path === '/v1/admin/prompts/general_agent.md' && method === 'GET') {
      return jsonResponse({
        prompt_file: 'general_agent.md',
        kind: 'agent_prompt',
        base_content: state.promptBase,
        overlay_content: state.promptOverlay,
        effective_content: state.promptOverlay || state.promptBase,
        overlay_active: Boolean(state.promptOverlay),
      })
    }

    if (path === '/v1/admin/prompts/general_agent.md' && method === 'PUT') {
      const body = readJsonBody(init)
      state.promptOverlay = String(body.content ?? '')
      return jsonResponse({ saved: true })
    }

    if (path === '/v1/admin/prompts/general_agent.md' && method === 'DELETE') {
      state.promptOverlay = ''
      return jsonResponse({ removed: true })
    }

    if (path === '/v1/admin/collections' && method === 'POST') {
      const body = readJsonBody(init)
      const collectionId = String(body.collection_id ?? '').trim()
      ensureCollection(collectionId)
      return jsonResponse({ created: true, collection: collectionSummaryPayload(collectionId) })
    }

    if (path === '/v1/admin/collections' && method === 'GET') {
      return jsonResponse(collectionsPayload())
    }

    if (path === '/v1/admin/sources' && method === 'GET') {
      return jsonResponse({
        sources: Object.values(state.registeredSources),
        runs: state.sourceRuns,
        allowed_roots: ['/tmp', '/workspace'],
      })
    }

    if (path === '/v1/admin/uploads' && method === 'GET') {
      return jsonResponse(uploadedFilesPayload())
    }

    if (path === '/v1/admin/uploads' && method === 'POST') {
      const form = init?.body as FormData
      const files = form ? form.getAll('files') as File[] : []
      const relativePaths = form ? form.getAll('relative_paths').map(value => String(value)) : []
      const collectionId = String(form?.get('collection_id') ?? 'control-panel-uploads')
      const operationFiles = files.map((file, index) => {
        const displayPath = relativePaths[index] || file.name
        const doc = upsertCollectionDoc(collectionId, file.name, `uploaded content for ${file.name}`, {
          sourcePath: `/uploads/${displayPath}`,
          sourceDisplayPath: displayPath,
          sourceType: 'upload',
        })
        return {
          displayPath,
          sourcePath: `/uploads/${displayPath}`,
          sourceType: 'upload',
          docIds: [doc.doc_id],
        }
      })
      return jsonResponse(collectionOperationPayload(collectionId, operationFiles))
    }

    const uploadDetailMatch = path.match(/^\/v1\/admin\/uploads\/([^/]+)$/)
    if (uploadDetailMatch && method === 'GET') {
      const [, docId] = uploadDetailMatch
      const doc = getUploadDoc(docId)
      if (!doc) return jsonResponse({ detail: 'Uploaded file not found.' }, 404)
      return jsonResponse({
        document: {
          doc_id: doc.doc_id,
          title: doc.title,
          source_type: doc.source_type,
          source_path: doc.source_path,
          source_display_path: doc.source_display_path,
          collection_id: doc.collection_id,
          num_chunks: 1,
          ingested_at: doc.ingested_at,
          file_type: doc.file_type,
          doc_structure_type: doc.doc_structure_type,
        },
        extracted_content: {
          content: doc.extracted,
          truncated: false,
          chunk_count: 1,
        },
        raw_source: {
          path: doc.source_path,
          content: doc.raw,
          truncated: false,
        },
        chunks: [{
          chunk_id: `${doc.doc_id}#0`,
          chunk_index: 0,
          chunk_type: 'general',
          page_number: null,
          section_title: null,
          clause_number: null,
          sheet_name: null,
          content: doc.extracted,
        }],
      })
    }

    const uploadReindexMatch = path.match(/^\/v1\/admin\/uploads\/([^/]+)\/reindex$/)
    if (uploadReindexMatch && method === 'POST') {
      const [, docId] = uploadReindexMatch
      const doc = getUploadDoc(docId)
      if (doc) {
        doc.extracted = `${doc.extracted} (reindexed)`
        doc.raw = `${doc.raw} (reindexed)`
      }
      return jsonResponse({ collection_id: doc?.collection_id ?? 'control-panel-uploads', deleted_doc_id: docId, ingested_doc_ids: [docId] })
    }

    if (uploadDetailMatch && method === 'DELETE') {
      const [, docId] = uploadDetailMatch
      const doc = getUploadDoc(docId)
      if (!doc) return jsonResponse({ detail: 'Uploaded file not found.' }, 404)
      state.collections[doc.collection_id] = (state.collections[doc.collection_id] ?? []).filter(item => item.doc_id !== docId)
      touchCollection(doc.collection_id)
      return jsonResponse({ deleted: true, doc_id: docId, collection_id: doc.collection_id })
    }

    const collectionCatalogMatch = path.match(/^\/v1\/admin\/collections\/([^/]+)$/)
    if (collectionCatalogMatch && method === 'GET') {
      const [, collectionId] = collectionCatalogMatch
      if (!collectionIds().includes(collectionId)) return jsonResponse({ detail: 'Collection not found.' }, 404)
      return jsonResponse({ collection: collectionSummaryPayload(collectionId) })
    }

    if (collectionCatalogMatch && method === 'DELETE') {
      const [, collectionId] = collectionCatalogMatch
      const hasDocs = (state.collections[collectionId] ?? []).some(doc => doc.source_type !== 'upload')
      const hasGraphs = graphIdsForCollection(collectionId).length > 0
      if (hasDocs || hasGraphs) {
        return jsonResponse({ detail: 'Collection is not empty.' }, 409)
      }
      delete state.collectionCatalog[collectionId]
      delete state.collections[collectionId]
      delete state.collectionHealth[collectionId]
      return jsonResponse({ deleted: true, collection_id: collectionId })
    }

    const collectionMatch = path.match(/^\/v1\/admin\/collections\/([^/]+)\/documents$/)
    if (collectionMatch && method === 'GET') {
      return jsonResponse(documentsPayload(collectionMatch[1]))
    }

    const collectionHealthMatch = path.match(/^\/v1\/admin\/collections\/([^/]+)\/health$/)
    if (collectionHealthMatch && method === 'GET') {
      return jsonResponse(collectionHealthPayload(collectionHealthMatch[1]))
    }

    const collectionRepairMatch = path.match(/^\/v1\/admin\/collections\/([^/]+)\/repair$/)
    if (collectionRepairMatch && method === 'POST') {
      const [, collectionId] = collectionRepairMatch
      state.collectionHealth[collectionId] = {
        ...collectionHealthPayload(collectionId),
        status: 'ready',
        reason: 'ready',
        indexed_doc_count: 2,
        active_doc_count: 2,
        duplicate_group_count: 0,
        content_drift_count: 0,
        duplicate_groups: [],
        drifted_groups: [],
        source_groups: [],
      }
      return jsonResponse({
        collection_id: collectionId,
        deleted_doc_ids: ['doc-arch-old'],
        reindexed_doc_ids: [],
        ingested_missing_doc_ids: [],
        unresolved_paths: [],
        health_after: state.collectionHealth[collectionId],
      })
    }

    const collectionDetailMatch = path.match(/^\/v1\/admin\/collections\/([^/]+)\/documents\/([^/]+)$/)
    if (collectionDetailMatch && method === 'GET') {
      const [, collectionId, docId] = collectionDetailMatch
      const doc = getDoc(collectionId, docId)
      return jsonResponse({
        document: {
          doc_id: doc?.doc_id,
          title: doc?.title,
          source_type: doc?.source_type,
          source_path: doc?.source_path,
          source_display_path: doc?.source_display_path,
          collection_id: doc?.collection_id,
          num_chunks: 1,
          ingested_at: doc?.ingested_at,
          file_type: doc?.file_type,
          doc_structure_type: doc?.doc_structure_type,
        },
        extracted_content: {
          content: doc?.extracted ?? '',
          truncated: false,
          chunk_count: 1,
        },
        raw_source: {
          path: doc?.source_path ?? '',
          content: doc?.raw ?? '',
          truncated: false,
        },
        chunks: [{
          chunk_id: `${doc?.doc_id}#0`,
          chunk_index: 0,
          chunk_type: 'general',
          page_number: null,
          section_title: null,
          clause_number: null,
          sheet_name: null,
          content: doc?.extracted ?? '',
        }],
      })
    }

    const ingestMatch = path.match(/^\/v1\/admin\/collections\/([^/]+)\/ingest-paths$/)
    if (ingestMatch && method === 'POST') {
      const [, collectionId] = ingestMatch
      const body = readJsonBody(init)
      const paths = Array.isArray(body.paths) ? body.paths.map(String) : []
      const operationFiles = paths.map(filePath => {
        const title = filePath.split('/').pop() || 'uploaded.txt'
        const doc = upsertCollectionDoc(collectionId, title, `ingested content for ${title}`, {
          sourcePath: filePath,
          sourceDisplayPath: title,
          sourceType: 'host_path',
        })
        return {
          displayPath: title,
          sourcePath: filePath,
          sourceType: 'host_path',
          docIds: [doc.doc_id],
        }
      })
      return jsonResponse(collectionOperationPayload(collectionId, operationFiles))
    }

    const uploadMatch = path.match(/^\/v1\/admin\/collections\/([^/]+)\/upload$/)
    if (uploadMatch && method === 'POST') {
      const [, collectionId] = uploadMatch
      const form = init?.body as FormData
      const files = form ? form.getAll('files') as File[] : []
      const relativePaths = form ? form.getAll('relative_paths').map(value => String(value)) : []
      const sourceType = form ? String(form.get('source_type') ?? 'collection_upload') : 'collection_upload'
      const operationFiles = files.map((file, index) => {
        const displayPath = relativePaths[index] || file.name
        const doc = upsertCollectionDoc(collectionId, file.name, `uploaded content for ${file.name}`, {
          sourcePath: `/uploads/${displayPath}`,
          sourceDisplayPath: displayPath,
          sourceType,
        })
        return {
          displayPath,
          sourcePath: `/uploads/${displayPath}`,
          sourceType,
          docIds: [doc.doc_id],
        }
      })
      return jsonResponse(collectionOperationPayload(collectionId, operationFiles))
    }

    const syncMatch = path.match(/^\/v1\/admin\/collections\/([^/]+)\/sync$/)
    if (syncMatch && method === 'POST') {
      const [, collectionId] = syncMatch
      ensureCollection(collectionId).maintenance_policy = 'configured_kb_sources'
      const doc = upsertCollectionDoc(collectionId, 'kb-sync.md', 'kb synced content', {
        sourcePath: '/kb/kb-sync.md',
        sourceDisplayPath: 'knowledge_base/kb-sync.md',
        sourceType: 'kb',
      })
      return jsonResponse(collectionOperationPayload(collectionId, [{
        displayPath: 'knowledge_base/kb-sync.md',
        sourcePath: '/kb/kb-sync.md',
        sourceType: 'kb',
        docIds: [doc.doc_id],
      }]))
    }

    const reindexMatch = path.match(/^\/v1\/admin\/collections\/([^/]+)\/documents\/([^/]+)\/reindex$/)
    if (reindexMatch && method === 'POST') {
      const [, collectionId, docId] = reindexMatch
      const doc = getDoc(collectionId, docId)
      if (doc) {
        doc.extracted = `${doc.extracted} (reindexed)`
        doc.raw = `${doc.raw} (reindexed)`
      }
      return jsonResponse({ collection_id: collectionId, deleted_doc_id: docId, ingested_doc_ids: [docId] })
    }

    if (collectionDetailMatch && method === 'DELETE') {
      const [, collectionId, docId] = collectionDetailMatch
      state.collections[collectionId] = (state.collections[collectionId] ?? []).filter(doc => doc.doc_id !== docId)
      touchCollection(collectionId)
      return jsonResponse({ deleted: true, doc_id: docId, collection_id: collectionId })
    }

    if (path === '/v1/admin/graphs' && method === 'GET') {
      return jsonResponse(listGraphsPayload())
    }

    if (path === '/v1/admin/graphs/assistant/suggest' && method === 'POST') {
      const body = readJsonBody(init)
      const collectionId = String(body.collection_id ?? 'default')
      const intent = String(body.intent ?? 'general')
      const graphId = slugify(`${collectionId}-${intent}-graph`)
      return jsonResponse({
        graph_id: graphId,
        display_name: `${collectionId} ${intent} Graph`,
        collection_id: collectionId,
        source_doc_ids: Array.isArray(body.source_doc_ids) ? body.source_doc_ids.map(String) : [],
        source_count: (state.collections[collectionId] ?? []).length,
        config_overrides: { extract_graph: { entity_types: ['entity', 'relationship'] } },
        prompt_overrides: { 'extract_graph.txt': 'Extract graph facts from {input_text}.' },
        friendly: { status: 'ready', headline: 'Suggested defaults' },
      })
    }

    if (path === '/v1/admin/graphs' && method === 'POST') {
      const body = readJsonBody(init)
      const displayName = String(body.display_name ?? 'New Graph')
      const requestedGraphId = String(body.graph_id ?? '').trim()
      const graphId = requestedGraphId || slugify(displayName)
      const collectionId = String(body.collection_id ?? 'default')
      ensureCollection(collectionId)
      const requestedSourceDocIds = Array.isArray(body.source_doc_ids) ? body.source_doc_ids.map(String) : []
      const sourceDocIds = requestedSourceDocIds.length > 0
        ? requestedSourceDocIds
        : (state.collections[collectionId] ?? []).map(doc => doc.doc_id)
      state.graphs[graphId] = {
        graph_id: graphId,
        display_name: displayName,
        collection_id: collectionId,
        backend: 'microsoft_graphrag',
        status: 'draft',
        query_ready: false,
        query_backend: '',
        domain_summary: `${displayName} draft for ${collectionId}.`,
        source_doc_ids: sourceDocIds,
        graph_skill_ids: Array.isArray(body.graph_skill_ids) ? body.graph_skill_ids.map(String) : [],
        prompt_overrides_json: (body.prompt_overrides ?? {}) as Record<string, unknown>,
        config_json: (body.config_overrides ?? {}) as Record<string, unknown>,
        logs: [],
      }
      state.graphRuns[graphId] = [{
        run_id: `${graphId}-create`,
        graph_id: graphId,
        operation: 'create',
        status: 'completed',
        detail: 'Created admin-managed graph draft.',
        started_at: '2026-04-08T10:00:00Z',
        completed_at: '2026-04-08T10:00:00Z',
      }]
      return jsonResponse({ created: true, graph_id: graphId, graph: state.graphs[graphId], sources: [] })
    }

    if (path === '/v1/admin/graphs/runs' && method === 'GET') {
      const requestUrl = input instanceof URL ? input.href : typeof input === 'string' ? input : input.url
      const searchParams = new URL(requestUrl, 'http://test.local').searchParams
      const status = searchParams.get('status') ?? ''
      const limit = Number(searchParams.get('limit') ?? 100)
      const runs = Object.values(state.graphRuns)
        .flat()
        .filter(run => !status || String(run.status) === status)
        .slice(0, limit)
      return jsonResponse({ runs })
    }

    if (path === '/v1/admin/graphs/runs/cleanup' && method === 'POST') {
      const body = readJsonBody(init)
      const status = String(body.status ?? 'failed')
      let deletedCount = 0
      for (const graphId of Object.keys(state.graphRuns)) {
        const runs = state.graphRuns[graphId] ?? []
        const kept = runs.filter(run => {
          const shouldDelete = String(run.status) === status
          if (shouldDelete) deletedCount += 1
          return !shouldDelete
        })
        state.graphRuns[graphId] = kept
      }
      return jsonResponse({ deleted: deletedCount > 0, deleted_count: deletedCount, status, runs: [] })
    }

    const graphMatch = path.match(/^\/v1\/admin\/graphs\/([^/]+)$/)
    if (graphMatch && method === 'GET') {
      const detail = graphDetailPayload(graphMatch[1])
      if (!detail) return jsonResponse({ detail: 'Graph not found.' }, 404)
      return jsonResponse(detail)
    }

    const graphProgressMatch = path.match(/^\/v1\/admin\/graphs\/([^/]+)\/progress$/)
    if (graphProgressMatch && method === 'GET') {
      const [, graphId] = graphProgressMatch
      const graph = state.graphs[graphId]
      if (!graph) return jsonResponse({ detail: 'Graph not found.' }, 404)
      const latestRun = (state.graphRuns[graphId] ?? [])[0] ?? null
      const active = ['queued', 'running'].includes(String(latestRun?.status ?? '').toLowerCase())
      return jsonResponse({
        graph_id: graphId,
        status: graph.status,
        active,
        active_run: active ? latestRun : null,
        latest_run: latestRun,
        workflow: active ? 'extract_graph' : 'finalize',
        task_progress: active ? { label: 'extract graph', current: 2, total: 4, percent: 50 } : {},
        stages: [
          { id: 'prepare', label: 'Prepare Project', state: active ? 'completed' : 'completed' },
          { id: 'extract_graph', label: 'Extract Graph', state: active ? 'active' : 'completed', workflow: 'extract_graph' },
          { id: 'finalize', label: 'Finalize Index', state: active ? 'pending' : 'completed' },
        ],
        percent: active ? 50 : 100,
        updated_at: '2026-04-08T10:02:00Z',
        logs: graph.logs,
        log_tail: String(graph.logs[0]?.preview ?? ''),
        cursor: '2026-04-08T10:02:00Z:420',
      })
    }

    const graphValidateMatch = path.match(/^\/v1\/admin\/graphs\/([^/]+)\/validate$/)
    if (graphValidateMatch && method === 'POST') {
      const [, graphId] = graphValidateMatch
      if (!state.graphs[graphId]) return jsonResponse({ detail: 'Graph not found.' }, 404)
      const payload = {
        graph_id: graphId,
        status: 'ready',
        ok: true,
        runtime: { ok: true, cli_available: true, provider: 'openai' },
        connectivity: { ok: true, status: 'ready', models: ['nemotron-cascade-2:30b', 'nomic-embed-text:latest'] },
        project_probe: { ok: true, detail: 'Graph project path is writable.' },
      }
      state.graphRuns[graphId] = [
        {
          run_id: `${graphId}-validate-${(state.graphRuns[graphId] ?? []).length + 1}`,
          graph_id: graphId,
          operation: 'validate',
          status: 'completed',
          detail: 'Validation finished with status ready.',
          started_at: '2026-04-08T10:01:00Z',
          completed_at: '2026-04-08T10:01:00Z',
        },
        ...(state.graphRuns[graphId] ?? []),
      ]
      return jsonResponse(payload)
    }

    const graphAssistantPreflightMatch = path.match(/^\/v1\/admin\/graphs\/([^/]+)\/assistant\/preflight$/)
    if (graphAssistantPreflightMatch && method === 'POST') {
      const [, graphId] = graphAssistantPreflightMatch
      const graph = state.graphs[graphId]
      if (!graph) return jsonResponse({ detail: 'Graph not found.' }, 404)
      const validation = {
        graph_id: graphId,
        status: 'ready',
        ok: true,
        runtime: { ok: true, cli_available: true, provider: 'openai' },
        connectivity: { ok: true, status: 'ready', models: ['nemotron-cascade-2:30b', 'nomic-embed-text:latest'] },
        extraction_preflight: { ok: true, status: 'ready', detail: 'Extraction sample succeeded.' },
        community_report_preflight: { ok: true, status: 'ready', detail: 'Community report prompt is available.' },
        resolved_source_count: graph.source_doc_ids.length,
      }
      return jsonResponse({
        graph_id: graphId,
        validation,
        friendly: {
          ready: true,
          status: 'ready',
          headline: 'Ready to build',
          source_count: graph.source_doc_ids.length,
          runtime_ok: true,
          model_endpoint_status: 'ready',
          extraction_status: 'ready',
          community_report_status: 'ready',
          blockers: [],
          warnings: [],
        },
      })
    }

    const graphBuildMatch = path.match(/^\/v1\/admin\/graphs\/([^/]+)\/(build|refresh)$/)
    if (graphBuildMatch && method === 'POST') {
      const [, graphId, action] = graphBuildMatch
      const graph = state.graphs[graphId]
      if (!graph) return jsonResponse({ detail: 'Graph not found.' }, 404)
      if (graphId.includes('failed')) {
        graph.status = 'failed'
        graph.query_ready = false
        state.graphRuns[graphId] = [
          {
            run_id: `${graphId}-${action}-failed`,
            graph_id: graphId,
            operation: action,
            status: 'failed',
            detail: `${action} failed during indexing.`,
            metadata: { failure_mode: 'test_failure' },
            started_at: '2026-04-08T10:02:00Z',
            completed_at: '2026-04-08T10:02:30Z',
          },
          ...(state.graphRuns[graphId] ?? []),
        ]
        return jsonResponse({
          ...graphDetailPayload(graphId),
          status: 'failed',
          detail: `${action} failed during indexing.`,
        })
      }
      graph.status = 'ready'
      graph.query_ready = true
      graph.query_backend = 'graphrag_python_api_preferred'
      graph.domain_summary = `${graph.display_name} is ready for graph-backed retrieval.`
      graph.logs = [
        {
          path: `/tmp/${graphId}.log`,
          name: `${graphId}.log`,
          size_bytes: 420,
          modified_at: '2026-04-08T10:02:00Z',
          preview: `${action} completed successfully for ${graph.display_name}.`,
        },
      ]
      state.graphRuns[graphId] = [
        {
          run_id: `${graphId}-${action}-${(state.graphRuns[graphId] ?? []).length + 1}`,
          graph_id: graphId,
          operation: action,
          status: 'ready',
          detail: `${action} completed successfully.`,
          started_at: '2026-04-08T10:02:00Z',
          completed_at: '2026-04-08T10:02:00Z',
        },
        ...(state.graphRuns[graphId] ?? []),
      ]
      return jsonResponse({
        ...graphDetailPayload(graphId),
        status: 'ready',
        detail: `${action} completed successfully.`,
        logs: graph.logs,
      })
    }

    const graphCancelMatch = path.match(/^\/v1\/admin\/graphs\/([^/]+)\/runs\/([^/]+)\/cancel$/)
    if (graphCancelMatch && method === 'POST') {
      const [, graphId, runId] = graphCancelMatch
      const runs = state.graphRuns[graphId] ?? []
      state.graphRuns[graphId] = runs.map(run => run.run_id === runId ? { ...run, status: 'cancelled', completed_at: '2026-04-08T10:03:00Z' } : run)
      if (state.graphs[graphId]) state.graphs[graphId].status = 'failed'
      return jsonResponse({ graph_id: graphId, status: 'cancelled', run: state.graphRuns[graphId][0] })
    }

    const graphRunDeleteMatch = path.match(/^\/v1\/admin\/graphs\/([^/]+)\/runs\/([^/]+)$/)
    if (graphRunDeleteMatch && method === 'DELETE') {
      const [, graphId, runId] = graphRunDeleteMatch
      const runs = state.graphRuns[graphId] ?? []
      const target = runs.find(run => run.run_id === runId)
      if (!target) return jsonResponse({ detail: 'Graph run not found.' }, 404)
      if (String(target.status) !== 'failed') return jsonResponse({ detail: 'Only failed graph runs can be deleted through this cleanup action.' }, 400)
      state.graphRuns[graphId] = runs.filter(run => run.run_id !== runId)
      return jsonResponse({ deleted: true, deleted_count: 1, graph_id: graphId, run_id: runId, status: 'failed', run: target })
    }

    if (graphMatch && method === 'DELETE') {
      const [, graphId] = graphMatch
      if (!state.graphs[graphId]) return jsonResponse({ detail: 'Graph not found.' }, 404)
      delete state.graphs[graphId]
      delete state.graphRuns[graphId]
      return jsonResponse({ deleted: true, graph_id: graphId, cleanup: { graph_indexes: 1 } })
    }

    const graphSmokeMatch = path.match(/^\/v1\/admin\/graphs\/([^/]+)\/assistant\/smoke-test$/)
    if (graphSmokeMatch && method === 'POST') {
      const [, graphId] = graphSmokeMatch
      const graph = state.graphs[graphId]
      if (!graph) return jsonResponse({ detail: 'Graph not found.' }, 404)
      return jsonResponse({
        graph_id: graphId,
        query: 'What are the main entities and relationships in this graph?',
        friendly: {
          status: graph.query_ready ? 'grounded' : 'source_candidates_only',
          query_ready: graph.query_ready,
          result_count: 1,
          citation_count: graph.query_ready ? 1 : 0,
          message: graph.query_ready ? 'Graph returned cited evidence.' : 'Graph returned source candidates only; use RAG grounding before answering.',
        },
        result: {
          query_ready: graph.query_ready,
          evidence_status: graph.query_ready ? 'grounded_graph_evidence' : 'source_candidates_only',
          results: [{ text: 'Vendor Risk Graph evidence' }],
          citations: graph.query_ready ? [{ citation_id: 'C1', doc_id: graph.source_doc_ids[0] ?? 'doc-1' }] : [],
        },
      })
    }

    const graphRunsMatch = path.match(/^\/v1\/admin\/graphs\/([^/]+)\/runs$/)
    if (graphRunsMatch && method === 'GET') {
      const [, graphId] = graphRunsMatch
      return jsonResponse({ runs: state.graphRuns[graphId] ?? [] })
    }

    const graphTuneStartMatch = path.match(/^\/v1\/admin\/graphs\/([^/]+)\/research-tune$/)
    if (graphTuneStartMatch && method === 'POST') {
      const [, graphId] = graphTuneStartMatch
      const graph = state.graphs[graphId]
      if (!graph) return jsonResponse({ detail: 'Graph not found.' }, 404)
      const body = readJsonBody(init)
      const targetPromptFiles = Array.isArray(body.target_prompt_files) ? body.target_prompt_files.map(String) : ['extract_graph.txt']
      const runId = `${graphId}-tune-${Object.keys(state.graphTuneRuns[graphId] ?? {}).length + 1}`
      const run = graphResearchTunePayload(graphId, runId, String(body.guidance ?? ''), targetPromptFiles)
      state.graphTuneRuns[graphId] = {
        ...(state.graphTuneRuns[graphId] ?? {}),
        [runId]: run,
      }
      state.graphRuns[graphId] = [
        {
          run_id: runId,
          graph_id: graphId,
          operation: 'research_tune',
          status: 'completed',
          detail: String(run.detail),
          metadata: {
            artifact_dir: String(run.artifact_dir),
            target_prompt_files: targetPromptFiles,
          },
          started_at: '2026-04-08T10:01:30Z',
          completed_at: '2026-04-08T10:01:31Z',
        },
        ...(state.graphRuns[graphId] ?? []),
      ]
      return jsonResponse(run)
    }

    const graphTuneApplyMatch = path.match(/^\/v1\/admin\/graphs\/([^/]+)\/research-tune\/([^/]+)\/apply$/)
    if (graphTuneApplyMatch && method === 'POST') {
      const [, graphId, runId] = graphTuneApplyMatch
      const graph = state.graphs[graphId]
      const run = state.graphTuneRuns[graphId]?.[runId]
      if (!graph || !run) return jsonResponse({ detail: 'Research & Tune run not found.' }, 404)
      const body = readJsonBody(init)
      const promptFiles = Array.isArray(body.prompt_files) ? body.prompt_files.map(String) : []
      const promptDrafts = run.prompt_drafts as Record<string, Record<string, unknown>>
      const appliedPromptFiles = promptFiles.filter(promptFile => promptDrafts[promptFile]?.content)
      for (const promptFile of appliedPromptFiles) {
        graph.prompt_overrides_json[promptFile] = String(promptDrafts[promptFile]?.content ?? '')
      }
      run.status = 'applied'
      run.manifest = {
        ...(run.manifest as Record<string, unknown>),
        status: 'applied',
        applied_prompt_files: appliedPromptFiles,
      }
      state.graphRuns[graphId] = [
        {
          run_id: `${runId}-apply`,
          graph_id: graphId,
          operation: 'research_tune_apply',
          status: 'completed',
          detail: `Applied ${appliedPromptFiles.length} Research & Tune prompt draft(s).`,
          metadata: { applied_prompt_files: appliedPromptFiles },
          started_at: '2026-04-08T10:01:32Z',
          completed_at: '2026-04-08T10:01:32Z',
        },
        ...(state.graphRuns[graphId] ?? []),
      ]
      return jsonResponse({
        ...graphDetailPayload(graphId),
        applied: true,
        applied_prompt_files: appliedPromptFiles,
        tuning_run: run,
      })
    }

    const graphTuneRunMatch = path.match(/^\/v1\/admin\/graphs\/([^/]+)\/research-tune\/([^/]+)$/)
    if (graphTuneRunMatch && method === 'GET') {
      const [, graphId, runId] = graphTuneRunMatch
      const run = state.graphTuneRuns[graphId]?.[runId]
      if (!run) return jsonResponse({ detail: 'Research & Tune run not found.' }, 404)
      return jsonResponse(run)
    }

    const graphPromptsMatch = path.match(/^\/v1\/admin\/graphs\/([^/]+)\/prompts$/)
    if (graphPromptsMatch && method === 'PUT') {
      const [, graphId] = graphPromptsMatch
      const body = readJsonBody(init)
      state.graphs[graphId].prompt_overrides_json = (body.prompt_overrides ?? {}) as Record<string, unknown>
      return jsonResponse(graphDetailPayload(graphId))
    }

    const graphSkillsMatch = path.match(/^\/v1\/admin\/graphs\/([^/]+)\/skills$/)
    if (graphSkillsMatch && method === 'PUT') {
      const [, graphId] = graphSkillsMatch
      const body = readJsonBody(init)
      const overlayMarkdown = String(body.overlay_markdown ?? '')
      const skillIds = Array.isArray(body.skill_ids) ? body.skill_ids.map(String) : []
      if (overlayMarkdown) {
        const overlaySkillId = `graph-${graphId}-overlay`
        state.skills[overlaySkillId] = {
          skill_id: overlaySkillId,
          name: String(body.overlay_skill_name ?? `${graphId} overlay`),
          agent_scope: 'rag',
          graph_id: graphId,
          body_markdown: overlayMarkdown,
          enabled: true,
          status: 'active',
          version: '1',
          version_parent: overlaySkillId,
          updated_at: '2026-04-08T10:03:00Z',
        }
        if (!skillIds.includes(overlaySkillId)) skillIds.push(overlaySkillId)
      }
      state.graphs[graphId].graph_skill_ids = skillIds
      return jsonResponse(graphDetailPayload(graphId))
    }

    const collectionSkillDraftMatch = path.match(/^\/v1\/admin\/collections\/([^/]+)\/skill-drafts$/)
    if (collectionSkillDraftMatch && method === 'POST') {
      const [, collectionId] = collectionSkillDraftMatch
      const body = readJsonBody(init)
      const graphId = String(body.graph_id ?? '')
      const collectionSkillId = `collection-${collectionId.replace(/[^a-zA-Z0-9_-]+/g, '-')}-rag-skill`
      const drafts: Array<Record<string, unknown>> = [
        {
          draft_type: 'collection_rag',
          label: 'Collection RAG skill',
          skill_id: collectionSkillId,
          name: `${collectionId} RAG Skill`,
          agent_scope: 'rag',
          collection_id: collectionId,
          graph_id: '',
          body_markdown: [
            `# ${collectionId} RAG Skill`,
            `skill_id: ${collectionSkillId}`,
            'agent_scope: rag',
            `collection_id: ${collectionId}`,
            'description: Collection scoped test skill.',
            'version: 1',
            'enabled: true',
            'tool_tags: search_indexed_docs',
            'task_tags: collection_research',
            '',
            '## Workflow',
            '',
            '- Stay inside the collection.',
          ].join('\n'),
          selected: true,
        },
      ]
      if (graphId) {
        const graphSkillId = `graph-${graphId}-manager-skill`
        drafts.push({
          draft_type: 'graph_manager',
          label: 'Graph skill',
          skill_id: graphSkillId,
          name: `${graphId} Graph Skill`,
          agent_scope: 'graph_manager',
          collection_id: collectionId,
          graph_id: graphId,
          body_markdown: [
            `# ${graphId} Graph Skill`,
            `skill_id: ${graphSkillId}`,
            'agent_scope: graph_manager',
            `graph_id: ${graphId}`,
            `collection_id: ${collectionId}`,
            'description: Graph scoped test skill.',
            'version: 1',
            'enabled: true',
            'tool_tags: list_graph_indexes, inspect_graph_index',
            'task_tags: graph_research',
            '',
            '## Workflow',
            '',
            '- Prefer source-backed graph relationships.',
          ].join('\n'),
          selected: true,
        })
      }
      return jsonResponse({
        object: 'collection.skill_draft.list',
        collection_id: collectionId,
        graph_id: graphId,
        drafts,
        mutated: false,
      })
    }

    const collectionSkillDraftApplyMatch = path.match(/^\/v1\/admin\/collections\/([^/]+)\/skill-drafts\/apply$/)
    if (collectionSkillDraftApplyMatch && method === 'POST') {
      const [, collectionId] = collectionSkillDraftApplyMatch
      const body = readJsonBody(init)
      const graphId = String(body.graph_id ?? '')
      const drafts = Array.isArray(body.drafts) ? body.drafts as Array<Record<string, unknown>> : []
      const appliedSkillIds: string[] = []
      const graphBoundSkillIds: string[] = []
      for (const draft of drafts) {
        const markdown = String(draft.body_markdown ?? draft.markdown ?? '')
        const metadata = parseSkillMetadata(markdown)
        const skillId = String(metadata.skill_id ?? draft.skill_id ?? `skill-new-${state.nextSkillIndex++}`)
        const agentScope = String(metadata.agent_scope ?? draft.agent_scope ?? 'rag')
        const draftGraphId = String(metadata.graph_id ?? draft.graph_id ?? graphId)
        state.skills[skillId] = {
          skill_id: skillId,
          name: parseSkillName(markdown),
          agent_scope: agentScope,
          graph_id: draftGraphId,
          collection_id: collectionId,
          body_markdown: markdown,
          enabled: true,
          status: 'active',
          version: '1',
          version_parent: skillId,
          updated_at: '2026-04-08T10:03:00Z',
        }
        appliedSkillIds.push(skillId)
        if (agentScope === 'graph_manager' && draftGraphId && state.graphs[draftGraphId]) {
          if (!state.graphs[draftGraphId].graph_skill_ids.includes(skillId)) {
            state.graphs[draftGraphId].graph_skill_ids.push(skillId)
          }
          graphBoundSkillIds.push(skillId)
        }
      }
      return jsonResponse({
        object: 'collection.skill_draft.apply',
        collection_id: collectionId,
        graph_id: graphId,
        applied_skill_ids: appliedSkillIds,
        graph_bound_skill_ids: graphBoundSkillIds,
        skills: appliedSkillIds.map(skillId => state.skills[skillId]),
      })
    }

    if (path === '/v1/skills' && method === 'GET') {
      return jsonResponse(listSkillsPayload())
    }

    if (path === '/v1/skills/preview' && method === 'POST') {
      return jsonResponse({
        object: 'skill.preview',
        matches: Object.values(state.skills)
          .filter(skill => skill.enabled)
          .slice(0, 1)
          .map(skill => ({
            skill_id: skill.skill_id,
            name: skill.name,
            agent_scope: skill.agent_scope,
            score: 0.93,
          })),
      })
    }

    const skillMatch = path.match(/^\/v1\/skills\/([^/]+)$/)
    if (skillMatch && method === 'GET') {
      return jsonResponse({
        ...state.skills[skillMatch[1]],
        chunks: ['chunk-1'],
        chunk_count: 1,
      })
    }

    if (path === '/v1/skills' && method === 'POST') {
      const body = readJsonBody(init)
      const bodyMarkdown = String(body.body_markdown ?? '')
      const name = parseSkillName(bodyMarkdown)
      const skillId = `skill-new-${state.nextSkillIndex++}`
      state.skills[skillId] = {
        skill_id: skillId,
        name,
        agent_scope: 'general',
        graph_id: String(body.graph_id ?? ''),
        collection_id: String(body.collection_id ?? ''),
        body_markdown: bodyMarkdown,
        enabled: true,
        status: 'active',
        version: '1',
        version_parent: skillId,
        updated_at: '2026-04-08T10:00:00Z',
      }
      return jsonResponse({ object: 'skill', data: state.skills[skillId] })
    }

    if (skillMatch && method === 'PUT') {
      const skillId = skillMatch[1]
      const body = readJsonBody(init)
      const bodyMarkdown = String(body.body_markdown ?? '')
      state.skills[skillId] = {
        ...state.skills[skillId],
        name: parseSkillName(bodyMarkdown),
        graph_id: String(body.graph_id ?? state.skills[skillId]?.graph_id ?? ''),
        collection_id: String(body.collection_id ?? state.skills[skillId]?.collection_id ?? ''),
        body_markdown: bodyMarkdown,
      }
      return jsonResponse({ object: 'skill', data: state.skills[skillId] })
    }

    const skillStatusMatch = path.match(/^\/v1\/skills\/([^/]+)\/(activate|deactivate)$/)
    if (skillStatusMatch && method === 'POST') {
      const [, skillId, action] = skillStatusMatch
      if (action === 'deactivate' && options.blockSkillDeactivate) {
        return jsonResponse({
          detail: {
            message: 'Cannot deactivate this skill because active dependents would break.',
            action: 'deactivate',
            dependency_validation: {
              skill_id: skillId,
              skill_family_id: state.skills[skillId]?.version_parent ?? skillId,
              dependency_state: 'blocked',
              is_valid: false,
              missing_dependencies: [],
              cycles: [],
              blocked_dependents: [
                {
                  skill_id: 'skill-dependent-v1',
                  skill_family_id: 'skill-dependent',
                  name: 'Dependent Skill',
                },
              ],
            },
          },
        }, 409)
      }
      state.skills[skillId] = {
        ...state.skills[skillId],
        enabled: action === 'activate',
        status: action === 'activate' ? 'active' : 'archived',
      }
      return jsonResponse({ object: 'skill', data: state.skills[skillId] })
    }

    return jsonResponse({ detail: `Unhandled ${method} ${path}` }, 404)
  })

  return { fetchMock }
}

function getSection(name: string): HTMLElement {
  return screen.getByRole('heading', { name, level: 3 }).closest('section') as HTMLElement
}

const navLabelAliases: Record<string, string> = {
  Dashboard: 'Overview',
  Config: 'Settings',
  Collections: 'Knowledge',
  'Uploaded Files': 'Uploads',
  MCP: 'Tools',
}

function navLabel(name: string): string {
  return navLabelAliases[name] ?? name
}

function navButton(name: string): HTMLElement {
  return screen.getByRole('button', { name: new RegExp(`^${navLabel(name)}\\b`) })
}

function openSection(name: string) {
  fireEvent.click(navButton(name))
}

describe('App', () => {
  beforeEach(() => {
    sessionStorage.clear()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('shows the login card, loads the dashboard, and locks again', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    expect(screen.getByText('Agent Control Panel')).toBeInTheDocument()

    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))

    expect(sessionStorage.getItem('control-panel-token')).toBe('token')
    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()
    expect(navButton('Dashboard')).toHaveAttribute('aria-current', 'page')

    fireEvent.click(screen.getByRole('button', { name: 'Lock' }))
    expect(screen.getByText('Agent Control Panel')).toBeInTheDocument()
    expect(sessionStorage.getItem('control-panel-token')).toBe('')
  })

  it('surfaces API failures in the error banner', async () => {
    const { fetchMock } = createFetchMock({ operationsError: 'Operations unavailable' })
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))

    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()
    openSection('Operations')

    expect(await screen.findByText('Operations unavailable')).toBeInTheDocument()
  })

  it('organizes access controls into guided tabs and readable effective access', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))

    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()
    openSection('Access')

    expect(await screen.findByRole('heading', { name: 'Access Control' })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Overview' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('button', { name: 'Access Setup Wizard' })).toBeInTheDocument()
    expect(screen.getByText('Group-first access keeps policy readable as the runtime grows.')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('tab', { name: 'Users' }))
    expect(await screen.findByRole('button', { name: 'Manage User' })).toBeInTheDocument()
    expect(screen.getAllByText('alex@example.com').length).toBeGreaterThan(0)

    fireEvent.click(screen.getByRole('tab', { name: 'Groups' }))
    expect(await screen.findByRole('button', { name: 'Create Group' })).toBeInTheDocument()
    expect(screen.getByText('Finance Analysts')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('tab', { name: 'Effective Access' }))
    fireEvent.change(screen.getByLabelText('User Email'), { target: { value: 'alex@example.com' } })
    fireEvent.click(screen.getByRole('button', { name: 'Preview Access' }))
    expect(await screen.findByText('Allowed Collections')).toBeInTheDocument()
    expect(screen.getAllByText('Blocked / No Grant').length).toBeGreaterThan(0)

    fireEvent.click(screen.getByRole('tab', { name: 'Matrix / Audit' }))
    expect(await screen.findByRole('heading', { name: 'Access Matrix', level: 3 })).toBeInTheDocument()
  })

  it('creates users, groups, and resource grants through access wizards', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))

    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()
    openSection('Access')
    expect(await screen.findByRole('heading', { name: 'Access Control' })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('tab', { name: 'Users' }))
    fireEvent.click(screen.getByRole('button', { name: 'Manage User' }))
    let dialog = await screen.findByRole('dialog', { name: 'Manage User' })
    fireEvent.change(within(dialog).getByLabelText('User Email'), { target: { value: 'taylor@example.com' } })
    fireEvent.change(within(dialog).getByLabelText('Display Name'), { target: { value: 'Taylor Admin' } })
    fireEvent.click(within(dialog).getByRole('radio', { name: 'Admin' }))
    fireEvent.click(within(dialog).getByRole('button', { name: 'Next' }))
    fireEvent.click(within(dialog).getByLabelText(/Finance Analysts/))
    fireEvent.click(within(dialog).getByRole('button', { name: 'Next' }))
    fireEvent.click(within(dialog).getByRole('button', { name: 'Next' }))
    fireEvent.click(within(dialog).getByRole('button', { name: 'Apply' }))
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Manage User' })).not.toBeInTheDocument())
    expect(await screen.findByText('Taylor Admin')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Grant Resource Access' }))
    dialog = await screen.findByRole('dialog', { name: 'Grant Resource Access' })
    expect(within(dialog).getByText('All Collections')).toBeInTheDocument()
    fireEvent.click(within(dialog).getByRole('button', { name: 'Next' }))
    expect(within(dialog).getByLabelText(/Finance Analysts/)).toBeChecked()
    fireEvent.click(within(dialog).getByRole('button', { name: 'Next' }))
    expect(within(dialog).getByLabelText(/Use/)).toBeChecked()
    fireEvent.click(within(dialog).getByRole('button', { name: 'Next' }))
    fireEvent.click(within(dialog).getByRole('button', { name: 'Apply' }))
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Grant Resource Access' })).not.toBeInTheDocument())
    await waitFor(() => expect(screen.getAllByText('All Collections').length).toBeGreaterThan(0))
    expect(screen.getByText('Finance Analysts')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('tab', { name: 'Groups' }))
    fireEvent.click(screen.getByRole('button', { name: 'Create Group' }))
    dialog = await screen.findByRole('dialog', { name: 'Create Group' })
    fireEvent.change(within(dialog).getByLabelText('Group Name'), { target: { value: 'Research Team' } })
    fireEvent.click(within(dialog).getByRole('radio', { name: 'Team / Sharing Group' }))
    fireEvent.click(within(dialog).getByRole('button', { name: 'Next' }))
    fireEvent.click(within(dialog).getByRole('button', { name: 'Next' }))
    fireEvent.click(within(dialog).getByLabelText(/Taylor Admin/))
    fireEvent.click(within(dialog).getByRole('button', { name: 'Next' }))
    fireEvent.click(within(dialog).getByRole('button', { name: 'Apply' }))
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Create Group' })).not.toBeInTheDocument())
    await waitFor(() => expect(screen.getAllByText('Research Team').length).toBeGreaterThan(0))

    expect(fetchMock.mock.calls.some(([input, init]) => requestPath(input) === '/v1/admin/access/principals' && requestMethod(input, init) === 'POST')).toBe(true)
    expect(fetchMock.mock.calls.some(([input, init]) => requestPath(input) === '/v1/admin/access/memberships' && requestMethod(input, init) === 'POST')).toBe(true)
    expect(fetchMock.mock.calls.some(([input, init]) => requestPath(input) === '/v1/admin/access/permissions' && requestMethod(input, init) === 'POST')).toBe(true)
  })

  it('loads the architecture section, updates the inspector, and shows routing and traffic views', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))

    await screen.findByRole('heading', { name: 'Runtime', level: 3 })
    openSection('Architecture')

    expect(await screen.findByRole('heading', { name: 'System Overview', level: 3 })).toBeInTheDocument()
    expect(navButton('Architecture')).toHaveAttribute('aria-current', 'page')
    expect(await screen.findByRole('button', { name: /Router/i })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /^general$/i }))
    await waitFor(() => expect(within(getSection('Node Inspector')).getAllByText('general').length).toBeGreaterThan(0))
    await waitFor(() => {
      expect(document.querySelector('[data-edge-id="edge-router-default-general"]')).toHaveAttribute('data-edge-layer', 'highlighted')
      expect(document.querySelector('[data-edge-id="edge-agent-general-service-skill-store"]')).toHaveAttribute('data-edge-layer', 'highlighted')
      expect(document.querySelector('[data-edge-id="edge-router-basic-basic"]')).toHaveAttribute('data-edge-layer', 'dimmed')
      expect(document.querySelector('[data-edge-id="edge-router-rag-rag_worker"]')).toHaveAttribute('data-edge-layer', 'dimmed')
    })

    fireEvent.click(screen.getByRole('tab', { name: 'Agent Graph' }))
    expect(await screen.findByRole('heading', { name: 'Agent Graph Overview', level: 3 })).toBeInTheDocument()
    expect(await screen.findByRole('heading', { name: 'LangGraph Export', level: 3 })).toBeInTheDocument()
    expect(screen.getByText('LangGraph Nodes')).toBeInTheDocument()
    expect(screen.getByText('PolicyAwareToolNode')).toBeInTheDocument()
    fireEvent.click(document.querySelector('[data-edge-id="edge-router-default-general"]')!)
    expect(await screen.findByRole('heading', { name: 'Edge Inspector', level: 3 })).toBeInTheDocument()
    expect(within(getSection('Edge Inspector')).getByText('Default AGENT')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('tab', { name: 'Routing Paths' }))
    expect(await screen.findByRole('heading', { name: 'Grounded lookup', level: 3 })).toBeInTheDocument()
    fireEvent.click(within(getSection('Grounded lookup')).getByRole('button', { name: 'Trace On Map' }))
    expect(await screen.findByRole('heading', { name: 'Path Inspector', level: 3 })).toBeInTheDocument()
    expect(within(getSection('Path Inspector')).getByText('Grounded lookup')).toBeInTheDocument()
    await waitFor(() => {
      expect(document.querySelector('[data-edge-id="edge-router-rag-rag_worker"]')).toHaveAttribute('data-edge-layer', 'highlighted')
      expect(document.querySelector('[data-edge-id="edge-router-default-general"]')).toHaveAttribute('data-edge-layer', 'dimmed')
    })

    fireEvent.click(screen.getByRole('tab', { name: 'Live Traffic' }))
    expect(await screen.findByRole('heading', { name: 'Recent Flows', level: 3 })).toBeInTheDocument()
    expect(await screen.findByRole('heading', { name: 'Router Quality', level: 3 })).toBeInTheDocument()
    expect(await screen.findByRole('heading', { name: 'Recent Mispicks', level: 3 })).toBeInTheDocument()
    expect(within(getSection('Recent Flows')).getAllByText(/rag_worker|general/).length).toBeGreaterThan(0)

    const architectureCalls = fetchMock.mock.calls.filter(([input]) => requestPath(input) === '/v1/admin/architecture')
    const activityCalls = fetchMock.mock.calls.filter(([input]) => requestPath(input) === '/v1/admin/architecture/activity')
    expect(architectureCalls.length).toBeGreaterThan(0)
    expect(activityCalls.length).toBeGreaterThan(0)
  })

  it('keeps the native agent graph visible when LangGraph export is unavailable', async () => {
    const { fetchMock } = createFetchMock({ langGraphUnavailable: true })
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))

    await screen.findByRole('heading', { name: 'Runtime', level: 3 })
    openSection('Architecture')
    expect(await screen.findByRole('heading', { name: 'System Overview', level: 3 })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('tab', { name: 'Agent Graph' }))
    expect(await screen.findByRole('heading', { name: 'Agent Graph Overview', level: 3 })).toBeInTheDocument()
    expect(await screen.findByText('LangGraph export unavailable')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^general$/i })).toBeInTheDocument()
  })

  it('recovers the architecture section after an initial snapshot failure when retry succeeds', async () => {
    const { fetchMock } = createFetchMock({ architectureSnapshotFailureCalls: [1] })
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))

    await screen.findByRole('heading', { name: 'Runtime', level: 3 })
    openSection('Architecture')

    expect(await screen.findByText('HTTP 500')).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'System Overview', level: 3 })).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))

    expect(await screen.findByRole('heading', { name: 'System Overview', level: 3 })).toBeInTheDocument()
    await waitFor(() => expect(screen.queryByText('HTTP 500')).not.toBeInTheDocument())
  })

  it('keeps the last good architecture data visible when a later refresh partially fails', async () => {
    const { fetchMock } = createFetchMock({ architectureActivityFailureCalls: [2] })
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))

    await screen.findByRole('heading', { name: 'Runtime', level: 3 })
    openSection('Architecture')

    expect(await screen.findByRole('heading', { name: 'System Overview', level: 3 })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))

    expect(await screen.findByText('HTTP 500')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'System Overview', level: 3 })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('tab', { name: 'Live Traffic' }))
    expect(await screen.findByRole('heading', { name: 'Recent Flows', level: 3 })).toBeInTheDocument()
    expect(within(getSection('Recent Flows')).getAllByText(/rag_worker|general/).length).toBeGreaterThan(0)
  })

  it('warns about a partial backend and renders a local unsupported state for architecture', async () => {
    const { fetchMock } = createFetchMock({ compatibilityMode: 'partial-architecture' })
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))

    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()
    expect(await screen.findByText(/partially compatible/i)).toBeInTheDocument()

    openSection('Architecture')

    expect(await screen.findByRole('heading', { name: 'Architecture is unavailable on this backend', level: 3 })).toBeInTheDocument()
    expect(screen.getByText(/Missing routes: \/v1\/admin\/architecture, \/v1\/admin\/architecture\/activity/i)).toBeInTheDocument()
    expect(navButton('Architecture')).toHaveAttribute('aria-current', 'page')
    expect(navButton('Architecture')).toHaveClass('owui-nav-item-warning')
    expect(within(navButton('Architecture')).getByLabelText('Unsupported')).toBeInTheDocument()

    const architectureCalls = fetchMock.mock.calls.filter(([input]) => requestPath(input) === '/v1/admin/architecture')
    const activityCalls = fetchMock.mock.calls.filter(([input]) => requestPath(input) === '/v1/admin/architecture/activity')
    expect(architectureCalls.length).toBe(0)
    expect(activityCalls.length).toBe(0)
  })

  it('handles config, agent, prompt, and operations workflows', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))
    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()

    openSection('Config')
    expect(navButton('Config')).toHaveAttribute('aria-current', 'page')
    const configTabs = await screen.findByRole('tablist', { name: 'Config groups' })
    expect(within(configTabs).getByRole('tab', { name: 'All' })).toHaveAttribute('aria-selected', 'true')
    expect(within(configTabs).getByRole('tab', { name: 'Runtime' })).toHaveAttribute('aria-selected', 'false')
    expect(within(configTabs).getByRole('tab', { name: 'Providers' })).toHaveAttribute('aria-selected', 'false')
    expect(await screen.findByLabelText('Ollama Chat Model')).toBeInTheDocument()
    const clarificationSlider = await screen.findByLabelText('Clarification Sensitivity')
    expect(clarificationSlider).toHaveAttribute('type', 'range')
    expect(within(getSection('Runtime')).getByText('Proceed')).toBeInTheDocument()
    expect(within(getSection('Runtime')).getByText('Balanced')).toBeInTheDocument()
    expect(within(getSection('Runtime')).getByText('Ask early')).toBeInTheDocument()
    fireEvent.change(clarificationSlider, { target: { value: '75' } })
    expect(within(getSection('Runtime')).getByText('75')).toBeInTheDocument()
    const configField = await screen.findByLabelText('Max Agent Steps')
    fireEvent.change(configField, { target: { value: '9' } })
    fireEvent.click(screen.getByRole('button', { name: 'Validate' }))
    await screen.findByRole('heading', { name: 'Preview', level: 3 })
    expect(within(getSection('Preview')).getByText('Valid')).toBeInTheDocument()
    expect(within(getSection('Preview')).getByText('runtime_swap')).toBeInTheDocument()
    expect(within(getSection('Preview')).getByText('Max Agent Steps')).toBeInTheDocument()
    expect(within(getSection('Preview')).getByText('Clarification Sensitivity')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
    await waitFor(() => expect(within(getSection('Preview')).getByText('Applied')).toBeInTheDocument())

    openSection('Agents')
    const descriptionField = await screen.findByLabelText('description')
    fireEvent.change(descriptionField, { target: { value: 'smoke test agent' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save Overlay' }))
    fireEvent.click(screen.getByRole('button', { name: 'Reload Agents' }))
    await waitFor(() => expect(within(getSection('Agent Inspector')).getAllByText(/smoke test agent/).length).toBeGreaterThan(0))
    fireEvent.click(screen.getByRole('tab', { name: 'Tool Catalog' }))
    expect(await screen.findByRole('heading', { name: 'Tool Catalog', level: 3 })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('tab', { name: 'Workspace' }))

    openSection('Prompts')
    const promptEditor = await within(getSection('Prompt Editor')).findByLabelText('Prompt Editor')
    fireEvent.change(promptEditor, { target: { value: 'Updated prompt from test' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save Overlay' }))
    await waitFor(() => expect(promptEditor).toHaveValue('Updated prompt from test'))
    fireEvent.click(screen.getByRole('button', { name: 'Reset Overlay' }))
    await waitFor(() => expect(promptEditor).toHaveValue('Base general prompt'))
    fireEvent.click(screen.getByRole('tab', { name: 'Compare' }))
    expect(await screen.findByRole('heading', { name: 'Prompt Summary', level: 3 })).toBeInTheDocument()
    expect(await screen.findByRole('heading', { name: 'Prompt Snapshot', level: 3 })).toBeInTheDocument()

    openSection('Operations')
    expect(await screen.findByRole('heading', { name: 'Last Reload', level: 3 })).toBeInTheDocument()
    expect(within(getSection('Last Reload')).getAllByText(/config_apply|agent_reload/).length).toBeGreaterThan(0)
    fireEvent.click(screen.getByRole('tab', { name: 'Jobs' }))
    expect(await screen.findByRole('heading', { name: 'Scheduler Health', level: 3 })).toBeInTheDocument()
    expect(await screen.findByRole('heading', { name: 'Background Jobs', level: 3 })).toBeInTheDocument()
    expect(within(getSection('Background Jobs')).getAllByText('interactive').length).toBeGreaterThan(0)
    fireEvent.click(screen.getByRole('tab', { name: 'Audit' }))
    expect(await screen.findByRole('heading', { name: 'Audit Stream', level: 3 })).toBeInTheDocument()

    const architectureCalls = fetchMock.mock.calls.filter(([input]) => requestPath(input) === '/v1/admin/architecture')
    expect(architectureCalls.length).toBeGreaterThanOrEqual(2)
  })

  it('shows all config groups by default and switches fields with the group selector', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))

    await screen.findByRole('heading', { name: 'Runtime', level: 3 })
    openSection('Config')

    const groupTabs = await screen.findByRole('tablist', { name: 'Config groups' })
    const allTab = within(groupTabs).getByRole('tab', { name: 'All' })
    const runtimeTab = within(groupTabs).getByRole('tab', { name: 'Runtime' })
    const providersTab = within(groupTabs).getByRole('tab', { name: 'Providers' })

    expect(allTab).toHaveAttribute('aria-selected', 'true')
    expect(runtimeTab).toHaveAttribute('aria-selected', 'false')
    expect(providersTab).toHaveAttribute('aria-selected', 'false')
    expect(await screen.findByLabelText('Max Agent Steps')).toBeInTheDocument()
    expect(await screen.findByLabelText('Ollama Chat Model')).toBeInTheDocument()

    fireEvent.click(providersTab)

    expect(providersTab).toHaveAttribute('aria-selected', 'true')
    expect(allTab).toHaveAttribute('aria-selected', 'false')
    expect(runtimeTab).toHaveAttribute('aria-selected', 'false')
    expect(await screen.findByLabelText('Ollama Chat Model')).toBeInTheDocument()
    expect(screen.queryByLabelText('Max Agent Steps')).not.toBeInTheDocument()

    fireEvent.click(allTab)

    expect(allTab).toHaveAttribute('aria-selected', 'true')
    expect(await screen.findByLabelText('Max Agent Steps')).toBeInTheDocument()
    expect(await screen.findByLabelText('Ollama Chat Model')).toBeInTheDocument()
  })

  it('filters settings groups and workspace resources from search inputs', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))

    await screen.findByRole('heading', { name: 'Runtime', level: 3 })
    openSection('Config')

    const groupTabs = await screen.findByRole('tablist', { name: 'Config groups' })
    expect(within(groupTabs).getByRole('tab', { name: 'Runtime' })).toBeInTheDocument()
    expect(within(groupTabs).getByRole('tab', { name: 'Providers' })).toBeInTheDocument()

    fireEvent.change(screen.getByPlaceholderText('Search settings'), { target: { value: 'ollama' } })

    expect(within(groupTabs).getByRole('tab', { name: 'All' })).toHaveAttribute('aria-selected', 'true')
    expect(within(groupTabs).queryByRole('tab', { name: 'Runtime' })).not.toBeInTheDocument()
    expect(within(groupTabs).getByRole('tab', { name: 'Providers' })).toHaveAttribute('aria-selected', 'false')
    expect(await screen.findByLabelText('Ollama Chat Model')).toBeInTheDocument()
    expect(screen.queryByLabelText('Max Agent Steps')).not.toBeInTheDocument()

    openSection('Agents')
    await screen.findByRole('heading', { name: 'Available Agents', level: 3 })
    expect(within(getSection('Available Agents')).getByRole('button', { name: /^general\b/i })).toBeInTheDocument()
    fireEvent.change(screen.getByPlaceholderText('Search agents'), { target: { value: 'missing-agent' } })
    expect(within(getSection('Available Agents')).queryByRole('button', { name: /^general\b/i })).not.toBeInTheDocument()
    expect(within(getSection('Available Agents')).getByText('No agents are registered in the current runtime.')).toBeInTheDocument()

    openSection('Collections')
    await screen.findByRole('heading', { name: 'Collections', level: 3 })
    expect(screen.getByRole('option', { name: 'default' })).toBeInTheDocument()
    fireEvent.change(screen.getByPlaceholderText('Search knowledge'), { target: { value: 'missing-collection' } })
    expect(screen.queryByRole('option', { name: 'default' })).not.toBeInTheDocument()
  })

  it('keeps technical details collapsed until requested', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))

    await screen.findByRole('heading', { name: 'Runtime', level: 3 })
    const runtimeSection = getSection('Runtime')
    const summary = within(runtimeSection).getByText('Technical details')
    const inspector = summary.closest('details') as HTMLDetailsElement

    expect(inspector).not.toHaveAttribute('open')
    fireEvent.click(summary)
    expect(inspector).toHaveAttribute('open')
  })

  it('renders studio selection rails with readable secondary metadata', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))

    await screen.findByRole('heading', { name: 'Runtime', level: 3 })

    openSection('Agents')
    await screen.findByRole('heading', { name: 'Available Agents', level: 3 })
    expect(within(getSection('Available Agents')).getByText('general_agent.md')).toBeInTheDocument()

    openSection('Prompts')
    await screen.findByRole('heading', { name: 'Prompt Files', level: 3 })
    expect(within(getSection('Prompt Files')).getByText('Using base prompt')).toBeInTheDocument()

    openSection('Skills')
    await screen.findByRole('heading', { name: 'Skill Library', level: 3 })
    expect(within(getSection('Skill Library')).getByRole('button', { name: 'New Skill' })).toBeInTheDocument()
    expect(within(getSection('Skill Library')).getByText('skill-existing')).toBeInTheDocument()
  })

  it('creates collections via a typed namespace and manages documents', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))
    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()

    openSection('Collections')
    const collectionIdField = await screen.findByLabelText('Collection ID')
    fireEvent.change(collectionIdField, { target: { value: 'smoke-control-panel' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create Collection' }))

    expect(await screen.findByRole('button', { name: /^smoke-control-panel\b/ })).toBeInTheDocument()
    const inspector = getSection('Collection Inspector')
    fireEvent.click(within(inspector).getByRole('button', { name: /Expand/ }))
    expect(await within(inspector).findAllByText('text-embedding-3-large')).toHaveLength(2)

    fireEvent.change(screen.getByLabelText('Collection Upload Files Input'), {
      target: { files: [new File(['regional spend'], 'regional_spend.csv', { type: 'text/csv' })] },
    })

    await waitFor(() => expect(within(getSection('Documents')).getByRole('button', { name: /^regional_spend\.csv\b/ })).toBeInTheDocument())
    fireEvent.click(within(getSection('Documents')).getByRole('button', { name: /^regional_spend\.csv\b/ }))
    await waitFor(() => expect(within(getSection('Document Viewer')).getAllByText(/uploaded content for regional_spend.csv/).length).toBeGreaterThan(0))

    fireEvent.click(screen.getByRole('button', { name: 'Reindex' }))
    await waitFor(() => expect(within(getSection('Document Viewer')).getAllByText(/reindexed/).length).toBeGreaterThan(0))
    fireEvent.click(within(getSection('Document Viewer')).getByRole('tab', { name: 'Raw' }))
    expect(within(getSection('Document Viewer')).getAllByText(/uploaded content for regional_spend.csv/).length).toBeGreaterThan(0)

    fireEvent.click(screen.getByRole('button', { name: 'Delete' }))
    fireEvent.click(within(await screen.findByRole('dialog', { name: 'Delete this document?' })).getByRole('button', { name: 'Delete' }))
    await waitFor(() => expect(within(getSection('Documents')).queryByRole('button', { name: /^regional_spend\.csv\b/ })).not.toBeInTheDocument())
    await waitFor(() => expect(screen.getByRole('button', { name: /^smoke-control-panel\b/ })).toBeInTheDocument())
    expect(collectionIdField).toHaveValue('smoke-control-panel')

    fireEvent.click(screen.getByRole('button', { name: 'Delete Empty' }))
    fireEvent.click(within(await screen.findByRole('dialog', { name: 'Delete this collection?' })).getByRole('button', { name: 'Delete' }))
    await waitFor(() => expect(screen.queryByRole('button', { name: /^smoke-control-panel\b/ })).not.toBeInTheDocument())
  })

  it('shows one collection action form at a time in the compact workspace', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))
    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()

    openSection('Collections')

    expect(await screen.findByRole('heading', { name: 'Knowledge Builder', level: 3 })).toBeInTheDocument()
    expect(screen.getByLabelText('Supported document types')).toHaveTextContent('.docx')
    expect(screen.getByLabelText('Supported document types')).toHaveTextContent('.xlsx')
    expect(await screen.findByRole('button', { name: 'Upload Files' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Upload Folder' })).toBeInTheDocument()
    expect(screen.queryByLabelText('Local Paths')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('tab', { name: 'Configured Sync' }))
    expect(await screen.findByRole('button', { name: 'Sync Configured Sources' })).toBeInTheDocument()
    expect(screen.queryByLabelText('Local Paths')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('tab', { name: 'Local Path' }))
    expect(await screen.findByLabelText('Local Paths')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Preview Scan' })).toBeInTheDocument()
  })

  it('manages uploaded files in their own workspace', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))
    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()

    openSection('Uploaded Files')
    expect(await screen.findByRole('heading', { name: 'Uploaded Files', level: 3 })).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('Upload Files Input'), {
      target: {
        files: [new File(['regional controls content'], 'regional_controls.csv', { type: 'text/csv' })],
      },
    })

    expect(await screen.findByRole('button', { name: /^regional_controls\.csv\b/ })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /^regional_controls\.csv\b/ }))
    await waitFor(() => expect(within(getSection('Uploaded File Viewer')).getAllByText(/uploaded content for regional_controls.csv/).length).toBeGreaterThan(0))

    openSection('Collections')
    await waitFor(() => expect(within(getSection('Documents')).queryByRole('button', { name: /^regional_controls\.csv\b/ })).not.toBeInTheDocument())

    openSection('Uploaded Files')
    fireEvent.click(screen.getByRole('button', { name: /^regional_controls\.csv\b/ }))
    fireEvent.click(within(getSection('Uploaded File Viewer')).getByRole('button', { name: 'Reindex' }))
    await waitFor(() => expect(within(getSection('Uploaded File Viewer')).getAllByText(/reindexed/).length).toBeGreaterThan(0))

    fireEvent.click(within(getSection('Uploaded File Viewer')).getByRole('button', { name: 'Delete' }))
    fireEvent.click(within(await screen.findByRole('dialog', { name: 'Delete this uploaded file?' })).getByRole('button', { name: 'Delete' }))
    await waitFor(() => expect(screen.queryByRole('button', { name: /^regional_controls\.csv\b/ })).not.toBeInTheDocument())
  })

  it('accepts partial upload API responses without collapsing into a generic error state', async () => {
    const { fetchMock } = createFetchMock()
    const wrappedFetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(input)
      const method = requestMethod(input, init)
      if (path === '/v1/admin/uploads' && method === 'POST') {
        await fetchMock(input, init)
        return jsonResponse({
          collection_id: 'control-panel-uploads',
          status: 'partial',
          summary: {
            resolved_count: 2,
            ingested_count: 1,
            skipped_count: 0,
            failed_count: 1,
            missing_count: 0,
          },
          resolved_count: 2,
          ingested_count: 1,
          skipped_count: 0,
          failed_count: 1,
          doc_ids: ['control-panel-uploads-alpha-1'],
          missing_paths: [],
          errors: ['beta.docx: parser failed'],
          files: [
            {
              display_path: 'alpha.txt',
              filename: 'alpha.txt',
              source_type: 'upload',
              source_path: '/uploads/alpha.txt',
              outcome: 'ingested',
              error: '',
              doc_ids: ['control-panel-uploads-alpha-1'],
            },
            {
              display_path: 'beta.docx',
              filename: 'beta.docx',
              source_type: 'upload',
              source_path: '/uploads/beta.docx',
              outcome: 'failed',
              error: 'beta.docx: parser failed',
              doc_ids: [],
            },
          ],
          filenames: ['alpha.txt', 'beta.docx'],
          display_paths: ['alpha.txt', 'beta.docx'],
          workspace_copies: ['alpha.txt'],
        })
      }
      return fetchMock(input, init)
    })
    vi.stubGlobal('fetch', wrappedFetch)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))
    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()

    openSection('Uploaded Files')
    expect(await screen.findByRole('heading', { name: 'Add Uploaded Files', level: 3 })).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('Upload Files Input'), {
      target: {
        files: [
          new File(['alpha'], 'alpha.txt', { type: 'text/plain' }),
          new File(['beta'], 'beta.docx', { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' }),
        ],
      },
    })

    expect(await screen.findByRole('button', { name: /^alpha\.txt\b/ })).toBeInTheDocument()
    expect(screen.queryByText('Unknown error')).not.toBeInTheDocument()
  })

  it('shows collection health issues and repairs duplicate KB rows', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))
    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()

    openSection('Collections')
    fireEvent.change(await screen.findByLabelText('Available Collections'), { target: { value: 'default' } })
    fireEvent.click(screen.getByRole('button', { name: 'Load Workspace' }))

    expect(await screen.findByRole('heading', { name: 'Collection Health', level: 3 })).toBeInTheDocument()
    expect(await within(getSection('Collection Health')).findByText('1 duplicate group')).toBeInTheDocument()
    expect(within(getSection('Collection Health')).getByText('1 duplicate group')).toBeInTheDocument()
    expect(await screen.findByText('ARCHITECTURE.md')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Repair Collection' }))

    await waitFor(() => expect(within(getSection('Collection Health')).getByText('Healthy')).toBeInTheDocument())
    expect(within(getSection('Collection Health')).getByText('0 duplicate groups')).toBeInTheDocument()
    expect(await screen.findByText('No duplicate or drift issues')).toBeInTheDocument()
  })

  it('preserves uploaded folder-relative paths for duplicate filenames and sends ordered relative paths', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))
    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()

    openSection('Uploaded Files')
    expect(await screen.findByRole('heading', { name: 'Add Uploaded Files', level: 3 })).toBeInTheDocument()

    const fileA = new File(['alpha'], 'same.txt', { type: 'text/plain' })
    Object.defineProperty(fileA, 'webkitRelativePath', { value: 'alpha/same.txt' })
    const fileB = new File(['beta'], 'same.txt', { type: 'text/plain' })
    Object.defineProperty(fileB, 'webkitRelativePath', { value: 'beta/same.txt' })

    fireEvent.change(screen.getByLabelText('Upload Folder Input'), {
      target: { files: [fileA, fileB] },
    })

    await waitFor(() => expect(screen.getAllByRole('button', { name: /^same\.txt\b/ })).toHaveLength(2))
    expect(screen.getAllByText('alpha/same.txt').length).toBeGreaterThan(0)
    expect(screen.getAllByText('beta/same.txt').length).toBeGreaterThan(0)

    await waitFor(() => expect(fetchMock.mock.calls.some(([, init]) => init?.body instanceof FormData)).toBe(true))
    const uploadCall = fetchMock.mock.calls.find(([input, init]) => requestPath(input) === '/v1/admin/uploads' && init?.body instanceof FormData)
    const uploadForm = uploadCall?.[1]?.body as FormData
    expect(uploadForm.getAll('relative_paths')).toEqual(['alpha/same.txt', 'beta/same.txt'])
  })

  it('includes collection uploads in graph collection document selection', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)
    const legacyUpload = new FormData()
    legacyUpload.append('files', new File(['legacy upload'], 'legacy-upload.txt', { type: 'text/plain' }))
    legacyUpload.append('relative_paths', 'legacy-upload.txt')
    await fetchMock('/v1/admin/collections/default/upload', { method: 'POST', body: legacyUpload })

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))
    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()

    openSection('Graphs')
    expect(await screen.findByRole('heading', { name: 'Graph Workspace', level: 3 })).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Graph Collection'), { target: { value: 'default' } })

    const graphSection = getSection('Graph Workspace')
    expect(screen.queryByLabelText('Graph Source Folder Input')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Upload Files' })).not.toBeInTheDocument()
    fireEvent.click(within(graphSection).getByRole('tab', { name: 'Choose Documents' }))
    expect(await within(graphSection).findByText('knowledge_base/default-notes.md')).toBeInTheDocument()
    expect((await within(graphSection).findAllByText('legacy-upload.txt')).length).toBeGreaterThan(0)
  })

  it('keeps brand new empty collections available in the graph workspace dropdown', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))
    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()

    openSection('Collections')
    fireEvent.change(await screen.findByLabelText('Collection ID'), { target: { value: 'graph-empty' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create Collection' }))
    expect(await screen.findByRole('button', { name: /^graph-empty\b/ })).toBeInTheDocument()

    openSection('Graphs')
    expect(await screen.findByRole('heading', { name: 'Graph Workspace', level: 3 })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'graph-empty' })).toBeInTheDocument()
  })

  it('explains ingestion wizard source modes and can tune prompts before building', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))
    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()

    openSection('Collections')
    fireEvent.click(await screen.findByRole('button', { name: 'Ingestion Wizard' }))
    const dialog = await screen.findByRole('dialog', { name: 'Ingestion Wizard' })
    fireEvent.click(within(dialog).getByRole('button', { name: 'Source' }))

    expect(within(dialog).getByText('Upload files or a folder from this browser')).toBeInTheDocument()
    expect(within(dialog).queryByLabelText('Wizard Server-Readable Local Paths')).not.toBeInTheDocument()

    fireEvent.click(within(dialog).getByRole('tab', { name: 'Local Source' }))
    expect(within(dialog).getByLabelText('Wizard Server-Readable Local Paths')).toBeInTheDocument()
    expect(within(dialog).getByText('Allowed Roots')).toBeInTheDocument()

    fireEvent.click(within(dialog).getByRole('tab', { name: 'Registered Source' }))
    expect(within(dialog).getByText('Refresh a saved folder or repository source')).toBeInTheDocument()
    expect(within(dialog).queryByLabelText('Wizard Server-Readable Local Paths')).not.toBeInTheDocument()

    fireEvent.click(within(dialog).getByRole('tab', { name: 'Sync Existing' }))
    expect(within(dialog).getByText('Sync runtime-configured KB sources')).toBeInTheDocument()
    expect(within(dialog).queryByLabelText('Wizard Server-Readable Local Paths')).not.toBeInTheDocument()

    fireEvent.click(within(dialog).getByRole('button', { name: 'Graph' }))
    fireEvent.click(within(dialog).getByLabelText(/Start the graph build after creating the draft/))
    fireEvent.click(within(dialog).getByRole('button', { name: 'Tuning' }))
    fireEvent.click(within(dialog).getByLabelText(/Run prompt tuning before the graph build/))
    fireEvent.change(within(dialog).getByLabelText('Wizard Prompt Tuning Guidance'), {
      target: { value: 'Prioritize supplier ownership and approval chains.' },
    })
    fireEvent.click(within(dialog).getByRole('button', { name: 'Run Research & Tune' }))

    await waitFor(() => expect(within(dialog).getByText('Prompt tuning result')).toBeInTheDocument())
    fireEvent.click(within(dialog).getByLabelText(/Apply selected prompt drafts before build/))
    fireEvent.click(within(dialog).getByRole('button', { name: 'Skills' }))
    expect(within(dialog).getByText('Preview first')).toBeInTheDocument()
    fireEvent.click(within(dialog).getByRole('button', { name: 'Generate Skill Drafts' }))
    await waitFor(() => expect(within(dialog).getByText('Collection RAG skill')).toBeInTheDocument())
    expect(within(dialog).getByText('Graph skill')).toBeInTheDocument()
    fireEvent.click(within(dialog).getByRole('button', { name: 'Apply Selected Drafts' }))
    await waitFor(() => expect(within(dialog).getByText(/Applied 2 skill draft/)).toBeInTheDocument())
    fireEvent.click(within(dialog).getByRole('button', { name: 'Review' }))
    expect(within(dialog).getByText('Run and apply selected drafts')).toBeInTheDocument()
    expect(within(dialog).getByText(/Applied 2 skill draft/)).toBeInTheDocument()
    fireEvent.click(within(dialog).getByRole('button', { name: 'Finish' }))

    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Ingestion Wizard' })).not.toBeInTheDocument())
    const skillApplyIndex = fetchMock.mock.calls.findIndex(([input]) => requestPath(input) === '/v1/admin/collections/default/skill-drafts/apply')
    const applyIndex = fetchMock.mock.calls.findIndex(([input]) => requestPath(input).includes('/research-tune/default_graph-tune-1/apply'))
    const buildIndex = fetchMock.mock.calls.findIndex(([input]) => requestPath(input) === '/v1/admin/graphs/default_graph/build')
    expect(skillApplyIndex).toBeGreaterThanOrEqual(0)
    expect(applyIndex).toBeGreaterThanOrEqual(0)
    expect(buildIndex).toBeGreaterThan(skillApplyIndex)
    expect(buildIndex).toBeGreaterThan(applyIndex)
  })

  it('creates a named graph from the wizard after creating a new collection and ingesting local paths', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))
    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()

    openSection('Collections')
    fireEvent.click(await screen.findByRole('button', { name: 'Ingestion Wizard' }))
    const dialog = await screen.findByRole('dialog', { name: 'Ingestion Wizard' })

    fireEvent.click(within(dialog).getByRole('radio', { name: 'Create New Collection' }))
    fireEvent.change(within(dialog).getByLabelText('Wizard Collection ID'), {
      target: { value: 'defense-rag-test-v2' },
    })
    fireEvent.click(within(dialog).getByRole('button', { name: 'Source' }))
    fireEvent.click(within(dialog).getByRole('tab', { name: 'Local Source' }))
    fireEvent.change(within(dialog).getByLabelText('Wizard Server-Readable Local Paths'), {
      target: { value: '/tmp/defense-rag/doc-one.md' },
    })
    fireEvent.click(within(dialog).getByRole('button', { name: 'Ingest Paths' }))
    await waitFor(() => {
      expect(fetchMock.mock.calls.some(([input]) => requestPath(input) === '/v1/admin/collections/defense-rag-test-v2/ingest-paths')).toBe(true)
    })

    fireEvent.click(within(dialog).getByRole('button', { name: 'Graph' }))
    fireEvent.change(within(dialog).getByLabelText('Wizard Graph Display Name'), {
      target: { value: 'defense rag test graph v2' },
    })
    expect(within(dialog).getByLabelText('Wizard Graph ID')).toHaveValue('defense_rag_test_graph_v2')
    fireEvent.click(within(dialog).getByRole('button', { name: 'Review' }))
    expect(within(dialog).getByText('defense rag test graph v2 (defense_rag_test_graph_v2)')).toBeInTheDocument()
    fireEvent.click(within(dialog).getByRole('button', { name: 'Finish' }))

    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Ingestion Wizard' })).not.toBeInTheDocument())
    const createCollectionIndex = fetchMock.mock.calls.findIndex(([input, init]) => (
      requestPath(input) === '/v1/admin/collections' && requestMethod(input, init) === 'POST'
    ))
    const ingestIndex = fetchMock.mock.calls.findIndex(([input]) => requestPath(input) === '/v1/admin/collections/defense-rag-test-v2/ingest-paths')
    const graphCreateCall = fetchMock.mock.calls.find(([input, init]) => (
      requestPath(input) === '/v1/admin/graphs' && requestMethod(input, init) === 'POST'
      && readJsonBody(init).graph_id === 'defense_rag_test_graph_v2'
    ))
    expect(createCollectionIndex).toBeGreaterThanOrEqual(0)
    expect(ingestIndex).toBeGreaterThan(createCollectionIndex)
    expect(readJsonBody(graphCreateCall?.[1]).display_name).toBe('defense rag test graph v2')
    expect(readJsonBody(graphCreateCall?.[1]).collection_id).toBe('defense-rag-test-v2')
  })

  it('renders collection inspector metadata including storage tables, dims, and mismatch warnings', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))
    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()

    openSection('Collections')
    fireEvent.change(await screen.findByLabelText('Available Collections'), { target: { value: 'default' } })

    const inspector = getSection('Collection Inspector')
    fireEvent.click(within(inspector).getByRole('button', { name: /Expand/ }))
    expect(await within(inspector).findAllByText('text-embedding-3-large')).toHaveLength(2)
    expect(within(inspector).getByText('documents, chunks')).toBeInTheDocument()
    expect(within(inspector).getByText('chunks: 1536')).toBeInTheDocument()
    expect(within(inspector).getByText('Vector dimension mismatch detected.')).toBeInTheDocument()
  })

  it('previews, creates, and toggles skills', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))
    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()

    openSection('Skills')
    fireEvent.click(screen.getByRole('tab', { name: 'Preview' }))
    const previewInput = await screen.findByLabelText('Preview Query')
    fireEvent.change(previewInput, { target: { value: 'help with routing' } })
    fireEvent.click(screen.getByRole('button', { name: 'Preview Match' }))
    await waitFor(() => expect(within(getSection('Skill Preview')).getAllByText(/skill-existing/).length).toBeGreaterThan(0))

    fireEvent.click(screen.getByRole('tab', { name: 'Editor' }))
    fireEvent.click(screen.getByRole('button', { name: 'New Skill' }))
    const skillEditor = within(getSection('Skill Editor')).getByRole('textbox')
    fireEvent.change(skillEditor, {
      target: {
        value: '# Smoke Skill\nagent_scope: general\n\ndescription: Temp skill.\n\n## Workflow\n\n- Do the smoke test.\n',
      },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Create Skill' }))

    expect(await screen.findByRole('button', { name: /^Smoke Skill\b/ })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Deactivate' }))
    await waitFor(() => expect(within(getSection('Skill Status')).getAllByText('archived').length).toBeGreaterThan(0))
    fireEvent.click(screen.getByRole('button', { name: 'Activate' }))
    await waitFor(() => expect(within(getSection('Skill Status')).getAllByText('active').length).toBeGreaterThan(0))
  })

  it('shows dependency blockers when a skill status change is rejected', async () => {
    const { fetchMock } = createFetchMock({ blockSkillDeactivate: true })
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))
    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()

    openSection('Skills')
    fireEvent.click(await screen.findByRole('button', { name: /^Existing Skill\b/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Deactivate' }))

    await waitFor(() => {
      expect(within(getSection('Skill Status')).getByText('Cannot deactivate this skill because active dependents would break.')).toBeInTheDocument()
    })
    expect(within(getSection('Skill Status')).getByText('Impacted dependents: Dependent Skill')).toBeInTheDocument()
  })

  it('creates, validates, builds, and updates a graph from the control-panel workspace', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))
    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()

    openSection('Graphs')
    expect(await screen.findByRole('heading', { name: 'Graph Workspace', level: 3 })).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('Graph Display Name'), { target: { value: 'Vendor Risk Graph' } })
    fireEvent.change(screen.getByLabelText('Graph ID'), { target: { value: 'vendor-risk' } })
    fireEvent.change(screen.getByLabelText('Graph Collection'), { target: { value: 'default' } })
    const graphWorkspace = getSection('Graph Workspace')
    expect(within(graphWorkspace).getByRole('tab', { name: 'Use Entire Collection' })).toHaveAttribute('aria-selected', 'true')
    await waitFor(() => expect(within(graphWorkspace).getByText('Build graph from 1 indexed documents')).toBeInTheDocument())
    fireEvent.click(within(graphWorkspace).getByRole('button', { name: 'Show Advanced GraphRAG JSON' }))

    fireEvent.change(screen.getByLabelText('Graph Prompt Overrides'), {
      target: { value: '{"extract_graph.txt":"Use vendor-centric extraction."}' },
    })
    fireEvent.change(screen.getByLabelText('Graph Config Overrides'), {
      target: { value: '{"extract_graph":{"entity_types":["vendor","risk"]}}' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Save Draft' }))

    expect(await screen.findByRole('button', { name: /^Vendor Risk Graph\b/ })).toBeInTheDocument()
    const graphCreateCall = fetchMock.mock.calls.find(
      ([input, init]) => requestPath(input) === '/v1/admin/graphs' && requestMethod(input, init) === 'POST',
    )
    expect(readJsonBody(graphCreateCall?.[1]).source_doc_ids).toEqual([])

    const tuneSection = getSection('Research & Tune')
    fireEvent.change(within(tuneSection).getByLabelText('Research Tune Guidance'), {
      target: { value: 'Prioritize supplier ownership, approval chains, and mitigation controls.' },
    })
    fireEvent.click(within(tuneSection).getByRole('button', { name: 'Run Research & Tune' }))
    await waitFor(() => expect(within(tuneSection).getByText('Scratchpad Preview')).toBeInTheDocument())
    expect(within(tuneSection).getByText('Vendor risk corpus summary with controls, approvers, exceptions, and supplier aliases.')).toBeInTheDocument()
    expect(within(tuneSection).getAllByText('Valid').length).toBeGreaterThan(0)
    expect((screen.getByLabelText('Graph Prompt Overrides') as HTMLTextAreaElement).value).toContain('Use vendor-centric extraction.')
    expect((screen.getByLabelText('Graph Prompt Overrides') as HTMLTextAreaElement).value).not.toContain('Dataset-Specific Curation Guidance')
    expect(fetchMock.mock.calls.some(([input]) => requestPath(input) === '/v1/admin/graphs/vendor-risk/research-tune/vendor-risk-tune-1/apply')).toBe(false)

    fireEvent.click(within(tuneSection).getByRole('button', { name: 'Apply Selected Prompts' }))
    await waitFor(() => {
      expect((screen.getByLabelText('Graph Prompt Overrides') as HTMLTextAreaElement).value).toContain('Dataset-Specific Curation Guidance')
    })
    expect(fetchMock.mock.calls.some(([input]) => requestPath(input) === '/v1/admin/graphs/vendor-risk/research-tune')).toBe(true)
    expect(fetchMock.mock.calls.some(([input]) => requestPath(input) === '/v1/admin/graphs/vendor-risk/research-tune/vendor-risk-tune-1/apply')).toBe(true)

    fireEvent.click(screen.getByRole('button', { name: 'Run Preflight' }))
    await waitFor(() => expect(within(graphWorkspace).getByText('Ready to build')).toBeInTheDocument())

    fireEvent.click(screen.getByRole('button', { name: 'Build' }))
    await waitFor(() => expect(within(getSection('Graph Inspector')).getByText('Query Ready')).toBeInTheDocument())

    fireEvent.change(screen.getByLabelText('Bound Graph Skill IDs'), { target: { value: 'skill-existing' } })
    fireEvent.change(screen.getByLabelText('Graph Overlay Skill Markdown'), {
      target: {
        value: '# Vendor Graph Overlay\nagent_scope: rag\n\n## Workflow\n\n- Prefer approval-chain language when this graph is selected.\n',
      },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Save Skill Overlay' }))

    await waitFor(() => expect(within(getSection('Graph Inspector')).getAllByText(/Vendor Risk Graph|Query Ready/).length).toBeGreaterThan(0))
    expect(fetchMock.mock.calls.some(([input]) => requestPath(input) === '/v1/admin/graphs/vendor-risk/build')).toBe(true)
    expect(fetchMock.mock.calls.some(([input]) => requestPath(input) === '/v1/admin/graphs/vendor-risk/skills')).toBe(true)
  })

  it('bulk-selects graph documents and prompt targets, and blocks empty manual graph drafts', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))
    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()

    openSection('Graphs')
    const graphWorkspace = await screen.findByRole('heading', { name: 'Graph Workspace', level: 3 }).then(() => getSection('Graph Workspace'))
    fireEvent.change(within(graphWorkspace).getByLabelText('Graph Display Name'), { target: { value: 'Manual Graph' } })
    fireEvent.change(within(graphWorkspace).getByLabelText('Graph ID'), { target: { value: 'manual-graph' } })
    fireEvent.change(within(graphWorkspace).getByLabelText('Graph Collection'), { target: { value: 'default' } })
    fireEvent.click(within(graphWorkspace).getByRole('tab', { name: 'Choose Documents' }))
    await waitFor(() => expect(within(graphWorkspace).getByRole('button', { name: 'Select All' })).toBeInTheDocument())

    expect(within(graphWorkspace).getByRole('button', { name: 'Save Draft' })).toBeDisabled()
    fireEvent.click(within(graphWorkspace).getByRole('button', { name: 'Select All' }))
    expect(within(graphWorkspace).getByRole('button', { name: 'Save Draft' })).not.toBeDisabled()
    fireEvent.click(within(graphWorkspace).getByRole('button', { name: 'Clear' }))
    expect(within(graphWorkspace).getByRole('button', { name: 'Save Draft' })).toBeDisabled()

    const tuneSection = getSection('Research & Tune')
    fireEvent.click(within(tuneSection).getByRole('button', { name: 'Clear' }))
    fireEvent.click(within(tuneSection).getByRole('button', { name: 'Select All' }))
    expect(within(tuneSection).getByRole('button', { name: 'Run Research & Tune' })).toBeDisabled()
  })

  it('shows and deletes failed graph runs from the failed-runs tab', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))
    expect(await screen.findByRole('heading', { name: 'Runtime', level: 3 })).toBeInTheDocument()

    openSection('Graphs')
    const graphWorkspace = await screen.findByRole('heading', { name: 'Graph Workspace', level: 3 }).then(() => getSection('Graph Workspace'))
    fireEvent.change(within(graphWorkspace).getByLabelText('Graph Display Name'), { target: { value: 'Failed Graph' } })
    fireEvent.change(within(graphWorkspace).getByLabelText('Graph ID'), { target: { value: 'failed-graph' } })
    fireEvent.change(within(graphWorkspace).getByLabelText('Graph Collection'), { target: { value: 'default' } })
    fireEvent.click(within(graphWorkspace).getByRole('button', { name: 'Save Draft' }))
    expect(await screen.findByRole('button', { name: /^Failed Graph\b/ })).toBeInTheDocument()
    fireEvent.click(within(graphWorkspace).getByRole('button', { name: 'Build' }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => requestPath(input) === '/v1/admin/graphs/failed-graph/build')).toBe(true))

    fireEvent.click(screen.getByRole('tab', { name: 'Failed Runs' }))
    expect(await screen.findByText('build failed during indexing.')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Delete Failed Run' }))
    fireEvent.click(within(await screen.findByRole('dialog', { name: 'Delete failed run?' })).getByRole('button', { name: 'Delete Failed Run' }))
    await waitFor(() => expect(screen.queryByText('build failed during indexing.')).not.toBeInTheDocument())
    expect(fetchMock.mock.calls.some(([input, init]) => (
      requestPath(input) === '/v1/admin/graphs/failed-graph/runs/failed-graph-build-failed'
      && requestMethod(input, init) === 'DELETE'
    ))).toBe(true)
  })

  it('remembers sub-tabs and collapsible panels within the browser session', async () => {
    const { fetchMock } = createFetchMock()
    vi.stubGlobal('fetch', fetchMock)

    const firstRender = renderApp()
    fireEvent.change(screen.getByPlaceholderText('Admin token'), { target: { value: 'token' } })
    fireEvent.click(screen.getByText('Unlock'))
    await screen.findByRole('heading', { name: 'Runtime', level: 3 })

    openSection('Agents')
    await screen.findByRole('heading', { name: 'Agent Inspector', level: 3 })
    fireEvent.click(within(getSection('Agent Inspector')).getByRole('button', { name: /Collapse/ }))

    openSection('Prompts')
    await screen.findByRole('heading', { name: 'Prompt Editor', level: 3 })
    fireEvent.click(screen.getByRole('tab', { name: 'Compare' }))
    expect(screen.getByRole('tab', { name: 'Compare' })).toHaveAttribute('aria-selected', 'true')

    firstRender.unmount()

    renderApp()
    await screen.findByRole('heading', { name: 'Runtime', level: 3 })

    openSection('Prompts')
    expect(screen.getByRole('tab', { name: 'Compare' })).toHaveAttribute('aria-selected', 'true')
    expect(await screen.findByRole('heading', { name: 'Prompt Summary', level: 3 })).toBeInTheDocument()

    openSection('Agents')
    await screen.findByRole('heading', { name: 'Agent Inspector', level: 3 })
    expect(within(getSection('Agent Inspector')).getByRole('button', { name: /Expand/ })).toBeInTheDocument()
  })
})
