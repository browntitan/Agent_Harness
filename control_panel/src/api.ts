import type {
  AccessMembership,
  AccessPrincipal,
  AccessRole,
  AccessRoleBinding,
  AccessRolePermission,
  AdminField,
  AdminOverview,
  ArchitectureActivity,
  ArchitectureSnapshot,
  CapabilitySectionStatus,
  CollectionHealthReport,
  CollectionOperationResult,
  CollectionSummary,
  ConfigValidationResult,
  ControlPanelCapabilities,
  EffectiveAccessPayload,
  GraphDetailPayload,
  GraphIndexRecord,
  GraphIndexRunRecord,
  GraphResearchTunePayload,
  McpConnectionRecord,
  McpToolCatalogRecord,
  UploadedFileSummary,
} from './types'

class ApiError extends Error {
  status?: number
  data?: unknown

  constructor(message: string, status?: number, data?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.data = data
  }
}

const SECTION_ROUTE_MAP = {
  dashboard: ['/v1/admin/overview'],
  architecture: ['/v1/admin/architecture', '/v1/admin/architecture/activity'],
  config: ['/v1/admin/config/schema', '/v1/admin/config/effective'],
  agents: ['/v1/admin/agents'],
  prompts: ['/v1/admin/prompts'],
  collections: ['/v1/admin/collections'],
  uploads: ['/v1/admin/uploads'],
  graphs: ['/v1/admin/graphs', '/v1/admin/graphs/{graph_id}'],
  skills: ['/v1/skills'],
  access: ['/v1/admin/access/principals', '/v1/admin/access/roles', '/v1/admin/access/effective-access'],
  mcp: ['/v1/admin/mcp/connections'],
  operations: ['/v1/admin/operations'],
} as const

export type CompatibilitySource = 'capabilities' | 'openapi'

export interface CompatibilityResult {
  source: CompatibilitySource
  capabilities: ControlPanelCapabilities
}

function headers(token: string, extra?: Record<string, string>): HeadersInit {
  return {
    ...(extra ?? {}),
    ...(token ? { 'X-Admin-Token': token } : {}),
  }
}

async function readError(res: Response): Promise<{ message: string; data?: unknown }> {
  const contentType = res.headers.get('content-type') ?? ''
  if (contentType.includes('application/json')) {
    try {
      const payload = await res.json()
      if (typeof payload?.detail === 'string') return { message: payload.detail, data: payload }
      if (typeof payload?.message === 'string') return { message: payload.message, data: payload }
      if (payload && typeof payload === 'object' && typeof payload?.detail?.message === 'string') {
        return { message: payload.detail.message, data: payload }
      }
      return { message: `HTTP ${res.status}`, data: payload }
    } catch {
      // Ignore malformed JSON payloads.
    }
  }
  return { message: (await res.text()).trim() || `HTTP ${res.status}` }
}

function operatorGuidance(path: string, status: number, message: string): string {
  const normalized = message.trim()

  if (
    status === 404 &&
    normalized === 'Not Found' &&
    path.startsWith('/v1/admin/architecture')
  ) {
    return 'Connected backend is stale and architecture routes are unavailable. Restart the API from this repo.'
  }
  if (
    status === 404 &&
    normalized === 'Not Found' &&
    (path.startsWith('/v1/admin/') || path.startsWith('/v1/skills'))
  ) {
    return 'Connected backend does not expose control-panel routes. Restart the API from this repo.'
  }
  if (status === 404 && normalized === 'Control panel is disabled.') {
    return 'Control panel is disabled on the backend. Set CONTROL_PANEL_ENABLED=true and restart the API.'
  }
  if (status === 503 && normalized === 'Control panel admin token is not configured.') {
    return 'Set CONTROL_PANEL_ADMIN_TOKEN in .env and restart the API.'
  }
  if (status === 503 && normalized.includes('not found in Ollama')) {
    return 'The configured Ollama model is missing. Pull the model or update the OLLAMA_* settings in .env, then restart the API.'
  }
  if (status === 401 && normalized === 'Invalid admin token.') {
    return 'The control-panel token does not match the backend. Re-enter the token or restart with the updated .env.'
  }

  return message
}

async function apiFetch<T>(path: string, token: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: headers(token, init.headers as Record<string, string> | undefined),
  })
  if (!res.ok) {
    const { message, data } = await readError(res)
    throw new ApiError(operatorGuidance(path, res.status, message), res.status, data)
  }
  return res.json() as Promise<T>
}

async function fetchOpenApiPaths(): Promise<Set<string>> {
  const res = await fetch('/openapi.json')
  if (!res.ok) {
    const { message, data } = await readError(res)
    throw new ApiError(message, res.status, data)
  }
  const payload = (await res.json()) as { paths?: Record<string, unknown> }
  return new Set(Object.keys(payload.paths ?? {}))
}

function buildFallbackCapabilities(paths: Set<string>): ControlPanelCapabilities {
  const sections = Object.fromEntries(
    Object.entries(SECTION_ROUTE_MAP).map(([sectionName, requiredRoutes]) => {
      const missingRoutes = requiredRoutes.filter(route => !paths.has(route))
      const supported = missingRoutes.length === 0
      const status: CapabilitySectionStatus = {
        supported,
        required_routes: [...requiredRoutes],
        missing_routes: missingRoutes,
        reason: supported
          ? 'Derived from openapi.json only because the compatibility endpoint is unavailable.'
          : 'Running backend is missing one or more required routes for this section.',
      }
      return [sectionName, status]
    }),
  )
  return {
    schema_version: 'fallback-openapi',
    contract_version: 'control-panel-v1',
    compatible: Object.values(sections).every(section => section.supported),
    generated_at: new Date().toISOString(),
    sections,
  }
}

export const api = {
  getOverview(token: string) {
    return apiFetch<AdminOverview>('/v1/admin/overview', token)
  },
  getOperations(token: string) {
    return apiFetch<Record<string, unknown>>('/v1/admin/operations', token)
  },
  getCapabilities(token: string) {
    return apiFetch<ControlPanelCapabilities>('/v1/admin/capabilities', token)
  },
  async inspectCompatibility(token: string): Promise<CompatibilityResult> {
    try {
      const capabilities = await apiFetch<ControlPanelCapabilities>('/v1/admin/capabilities', token)
      return { source: 'capabilities', capabilities }
    } catch (error) {
      if (isApiError(error) && error.status === 404) {
        const openApiPaths = await fetchOpenApiPaths()
        return {
          source: 'openapi',
          capabilities: buildFallbackCapabilities(openApiPaths),
        }
      }
      throw error
    }
  },
  getArchitecture(token: string) {
    return apiFetch<ArchitectureSnapshot>('/v1/admin/architecture', token)
  },
  getArchitectureActivity(token: string) {
    return apiFetch<ArchitectureActivity>('/v1/admin/architecture/activity', token)
  },
  getConfigSchema(token: string) {
    return apiFetch<{ fields: AdminField[] }>('/v1/admin/config/schema', token)
  },
  getEffectiveConfig(token: string) {
    return apiFetch<{ values: Record<string, string>; overlay_values: Record<string, string> }>(
      '/v1/admin/config/effective',
      token,
    )
  },
  validateConfig(token: string, changes: Record<string, unknown>) {
    return apiFetch<ConfigValidationResult>('/v1/admin/config/validate', token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ changes }),
    })
  },
  applyConfig(token: string, changes: Record<string, unknown>) {
    return apiFetch<ConfigValidationResult>('/v1/admin/config/apply', token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ changes }),
    })
  },
  listAgents(token: string) {
    return apiFetch<{ agents: Array<Record<string, unknown>>; tools: Array<Record<string, unknown>> }>(
      '/v1/admin/agents',
      token,
    )
  },
  getAgent(token: string, agentName: string) {
    return apiFetch<Record<string, unknown>>(`/v1/admin/agents/${agentName}`, token)
  },
  updateAgent(token: string, agentName: string, payload: Record<string, unknown>) {
    return apiFetch<Record<string, unknown>>(`/v1/admin/agents/${agentName}`, token, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  },
  reloadAgents(token: string) {
    return apiFetch<Record<string, unknown>>('/v1/admin/agents/reload', token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ changes: {} }),
    })
  },
  listPrompts(token: string) {
    return apiFetch<{ prompts: Array<Record<string, unknown>> }>('/v1/admin/prompts', token)
  },
  getPrompt(token: string, promptFile: string) {
    return apiFetch<Record<string, unknown>>(`/v1/admin/prompts/${promptFile}`, token)
  },
  updatePrompt(token: string, promptFile: string, content: string) {
    return apiFetch<Record<string, unknown>>(`/v1/admin/prompts/${promptFile}`, token, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content }),
    })
  },
  resetPrompt(token: string, promptFile: string) {
    return apiFetch<Record<string, unknown>>(`/v1/admin/prompts/${promptFile}`, token, {
      method: 'DELETE',
    })
  },
  listCollections(token: string) {
    return apiFetch<{ collections: CollectionSummary[] }>('/v1/admin/collections', token)
  },
  createCollection(token: string, collectionId: string) {
    return apiFetch<{ created: boolean; collection: CollectionSummary }>('/v1/admin/collections', token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ collection_id: collectionId }),
    })
  },
  getCollection(token: string, collectionId: string) {
    return apiFetch<{ collection: CollectionSummary }>(`/v1/admin/collections/${collectionId}`, token)
  },
  deleteCollection(token: string, collectionId: string) {
    return apiFetch<Record<string, unknown>>(`/v1/admin/collections/${collectionId}`, token, {
      method: 'DELETE',
    })
  },
  syncCollection(token: string, collectionId: string) {
    return apiFetch<CollectionOperationResult>(`/v1/admin/collections/${collectionId}/sync`, token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ changes: {} }),
    })
  },
  ingestPaths(token: string, collectionId: string, paths: string[], metadataProfile = 'auto', indexPreview = false) {
    return apiFetch<CollectionOperationResult>(`/v1/admin/collections/${collectionId}/ingest-paths`, token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        paths,
        source_type: 'host_path',
        metadata_profile: metadataProfile,
        index_preview: indexPreview,
      }),
    })
  },
  uploadFiles(token: string, collectionId: string, files: File[], relativePaths: string[] = [], metadataProfile = 'auto', indexPreview = false) {
    const form = new FormData()
    files.forEach((file, index) => {
      form.append('files', file)
      form.append('relative_paths', relativePaths[index] ?? '')
    })
    form.append('metadata_profile', metadataProfile)
    form.append('index_preview', String(indexPreview))
    return apiFetch<CollectionOperationResult>(`/v1/admin/collections/${collectionId}/upload`, token, {
      method: 'POST',
      body: form,
    })
  },
  listUploadedFiles(token: string) {
    return apiFetch<{ uploads: UploadedFileSummary[] }>('/v1/admin/uploads', token)
  },
  uploadUploadedFiles(token: string, files: File[], relativePaths: string[] = [], collectionId = '', metadataProfile = 'auto', indexPreview = false) {
    const form = new FormData()
    files.forEach((file, index) => {
      form.append('files', file)
      form.append('relative_paths', relativePaths[index] ?? '')
    })
    if (collectionId.trim()) form.append('collection_id', collectionId.trim())
    form.append('metadata_profile', metadataProfile)
    form.append('index_preview', String(indexPreview))
    return apiFetch<CollectionOperationResult>('/v1/admin/uploads', token, {
      method: 'POST',
      body: form,
    })
  },
  getUploadedFile(token: string, docId: string) {
    return apiFetch<Record<string, unknown>>(`/v1/admin/uploads/${docId}`, token)
  },
  reindexUploadedFile(token: string, docId: string) {
    return apiFetch<Record<string, unknown>>(
      `/v1/admin/uploads/${docId}/reindex`,
      token,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ changes: {} }),
      },
    )
  },
  deleteUploadedFile(token: string, docId: string) {
    return apiFetch<Record<string, unknown>>(
      `/v1/admin/uploads/${docId}`,
      token,
      { method: 'DELETE' },
    )
  },
  listCollectionDocuments(token: string, collectionId: string) {
    return apiFetch<{ documents: Array<Record<string, unknown>> }>(
      `/v1/admin/collections/${collectionId}/documents`,
      token,
    )
  },
  getCollectionHealth(token: string, collectionId: string) {
    return apiFetch<CollectionHealthReport>(
      `/v1/admin/collections/${collectionId}/health`,
      token,
    )
  },
  repairCollection(token: string, collectionId: string) {
    return apiFetch<Record<string, unknown>>(
      `/v1/admin/collections/${collectionId}/repair`,
      token,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ changes: {} }),
      },
    )
  },
  getCollectionDocument(token: string, collectionId: string, docId: string) {
    return apiFetch<Record<string, unknown>>(
      `/v1/admin/collections/${collectionId}/documents/${docId}`,
      token,
    )
  },
  reindexDocument(token: string, collectionId: string, docId: string) {
    return apiFetch<Record<string, unknown>>(
      `/v1/admin/collections/${collectionId}/documents/${docId}/reindex`,
      token,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ changes: {} }),
      },
    )
  },
  deleteDocument(token: string, collectionId: string, docId: string) {
    return apiFetch<Record<string, unknown>>(
      `/v1/admin/collections/${collectionId}/documents/${docId}`,
      token,
      { method: 'DELETE' },
    )
  },
  listGraphs(token: string, collectionId = '') {
    const suffix = collectionId ? `?collection_id=${encodeURIComponent(collectionId)}` : ''
    return apiFetch<{ graphs: GraphIndexRecord[] }>(`/v1/admin/graphs${suffix}`, token)
  },
  getGraph(token: string, graphId: string) {
    return apiFetch<GraphDetailPayload>(`/v1/admin/graphs/${graphId}`, token)
  },
  createGraph(token: string, payload: Record<string, unknown>) {
    return apiFetch<Record<string, unknown>>('/v1/admin/graphs', token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  },
  validateGraph(token: string, graphId: string) {
    return apiFetch<Record<string, unknown>>(`/v1/admin/graphs/${graphId}/validate`, token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    })
  },
  buildGraph(token: string, graphId: string) {
    return apiFetch<Record<string, unknown>>(`/v1/admin/graphs/${graphId}/build`, token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    })
  },
  refreshGraph(token: string, graphId: string) {
    return apiFetch<Record<string, unknown>>(`/v1/admin/graphs/${graphId}/refresh`, token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    })
  },
  getGraphRuns(token: string, graphId: string) {
    return apiFetch<{ runs: GraphIndexRunRecord[] }>(`/v1/admin/graphs/${graphId}/runs`, token)
  },
  startGraphResearchTune(
    token: string,
    graphId: string,
    payload: { guidance?: string; target_prompt_files?: string[] },
  ) {
    return apiFetch<GraphResearchTunePayload>(`/v1/admin/graphs/${graphId}/research-tune`, token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  },
  getGraphResearchTune(token: string, graphId: string, runId: string) {
    return apiFetch<GraphResearchTunePayload>(`/v1/admin/graphs/${graphId}/research-tune/${runId}`, token)
  },
  applyGraphResearchTune(token: string, graphId: string, runId: string, promptFiles: string[]) {
    return apiFetch<GraphDetailPayload & Record<string, unknown>>(
      `/v1/admin/graphs/${graphId}/research-tune/${runId}/apply`,
      token,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt_files: promptFiles }),
      },
    )
  },
  updateGraphPrompts(token: string, graphId: string, promptOverrides: Record<string, unknown>) {
    return apiFetch<Record<string, unknown>>(`/v1/admin/graphs/${graphId}/prompts`, token, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt_overrides: promptOverrides }),
    })
  },
  updateGraphSkills(
    token: string,
    graphId: string,
    payload: { skill_ids: string[]; overlay_markdown?: string; overlay_skill_name?: string },
  ) {
    return apiFetch<Record<string, unknown>>(`/v1/admin/graphs/${graphId}/skills`, token, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  },
  listSkills(token: string) {
    return apiFetch<{ object: string; data: Array<Record<string, unknown>> }>('/v1/skills', token)
  },
  getSkill(token: string, skillId: string) {
    return apiFetch<Record<string, unknown>>(`/v1/skills/${skillId}`, token)
  },
  createSkill(token: string, payload: Record<string, unknown>) {
    return apiFetch<Record<string, unknown>>('/v1/skills', token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  },
  updateSkill(token: string, skillId: string, payload: Record<string, unknown>) {
    return apiFetch<Record<string, unknown>>(`/v1/skills/${skillId}`, token, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  },
  activateSkill(token: string, skillId: string) {
    return apiFetch<Record<string, unknown>>(`/v1/skills/${skillId}/activate`, token, { method: 'POST' })
  },
  deactivateSkill(token: string, skillId: string) {
    return apiFetch<Record<string, unknown>>(`/v1/skills/${skillId}/deactivate`, token, { method: 'POST' })
  },
  previewSkill(token: string, query: string, agentScope: string) {
    return apiFetch<Record<string, unknown>>('/v1/skills/preview', token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, agent_scope: agentScope, top_k: 5 }),
    })
  },
  listAccessPrincipals(token: string, params: { principalType?: string; query?: string } = {}) {
    const search = new URLSearchParams()
    if (params.principalType) search.set('principal_type', params.principalType)
    if (params.query) search.set('query', params.query)
    const suffix = search.toString() ? `?${search.toString()}` : ''
    return apiFetch<{ principals: AccessPrincipal[] }>(`/v1/admin/access/principals${suffix}`, token)
  },
  createAccessPrincipal(token: string, payload: Record<string, unknown>) {
    return apiFetch<{ principal: AccessPrincipal }>('/v1/admin/access/principals', token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  },
  listAccessMemberships(token: string) {
    return apiFetch<{ memberships: AccessMembership[] }>('/v1/admin/access/memberships', token)
  },
  createAccessMembership(token: string, payload: Record<string, unknown>) {
    return apiFetch<{ membership: AccessMembership }>('/v1/admin/access/memberships', token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  },
  deleteAccessMembership(token: string, membershipId: string) {
    return apiFetch<Record<string, unknown>>(`/v1/admin/access/memberships/${membershipId}`, token, {
      method: 'DELETE',
    })
  },
  listAccessRoles(token: string) {
    return apiFetch<{ roles: AccessRole[] }>('/v1/admin/access/roles', token)
  },
  createAccessRole(token: string, payload: Record<string, unknown>) {
    return apiFetch<{ role: AccessRole }>('/v1/admin/access/roles', token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  },
  deleteAccessRole(token: string, roleId: string) {
    return apiFetch<Record<string, unknown>>(`/v1/admin/access/roles/${roleId}`, token, {
      method: 'DELETE',
    })
  },
  listAccessBindings(token: string) {
    return apiFetch<{ bindings: AccessRoleBinding[] }>('/v1/admin/access/bindings', token)
  },
  createAccessBinding(token: string, payload: Record<string, unknown>) {
    return apiFetch<{ binding: AccessRoleBinding }>('/v1/admin/access/bindings', token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  },
  deleteAccessBinding(token: string, bindingId: string) {
    return apiFetch<Record<string, unknown>>(`/v1/admin/access/bindings/${bindingId}`, token, {
      method: 'DELETE',
    })
  },
  listAccessPermissions(token: string) {
    return apiFetch<{ permissions: AccessRolePermission[] }>('/v1/admin/access/permissions', token)
  },
  createAccessPermission(token: string, payload: Record<string, unknown>) {
    return apiFetch<{ permission: AccessRolePermission }>('/v1/admin/access/permissions', token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  },
  deleteAccessPermission(token: string, permissionId: string) {
    return apiFetch<Record<string, unknown>>(`/v1/admin/access/permissions/${permissionId}`, token, {
      method: 'DELETE',
    })
  },
  getEffectiveAccess(token: string, email: string) {
    return apiFetch<EffectiveAccessPayload>(`/v1/admin/access/effective-access?email=${encodeURIComponent(email)}`, token)
  },
  listMcpConnections(token: string) {
    return apiFetch<{ enabled: boolean; connections: McpConnectionRecord[] }>('/v1/admin/mcp/connections', token)
  },
  createMcpConnection(token: string, payload: Record<string, unknown>) {
    return apiFetch<{ connection: McpConnectionRecord }>('/v1/admin/mcp/connections', token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  },
  updateMcpConnection(token: string, connectionId: string, payload: Record<string, unknown>) {
    return apiFetch<{ connection: McpConnectionRecord }>(`/v1/admin/mcp/connections/${connectionId}`, token, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  },
  disableMcpConnection(token: string, connectionId: string) {
    return apiFetch<{ connection: McpConnectionRecord }>(`/v1/admin/mcp/connections/${connectionId}/disable`, token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ changes: {} }),
    })
  },
  testMcpConnection(token: string, connectionId: string) {
    return apiFetch<{ health: Record<string, unknown> }>(`/v1/admin/mcp/connections/${connectionId}/test`, token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ changes: {} }),
    })
  },
  refreshMcpTools(token: string, connectionId: string) {
    return apiFetch<{ tools: McpToolCatalogRecord[] }>(`/v1/admin/mcp/connections/${connectionId}/refresh-tools`, token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ changes: {} }),
    })
  },
  updateMcpTool(token: string, connectionId: string, toolId: string, payload: Record<string, unknown>) {
    return apiFetch<{ tool: McpToolCatalogRecord }>(`/v1/admin/mcp/connections/${connectionId}/tools/${toolId}`, token, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  },
}

export function isApiError(error: unknown): error is ApiError {
  return error instanceof ApiError
}
