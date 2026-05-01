import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { Search } from 'lucide-react'
import { api, isApiError } from './api'
import { buildArchitectureMapLayout } from './architectureLayout'
import {
  ActionBar,
  ActionButton,
  AppShell,
  CollapsibleSurfaceCard,
  ConfirmDialog,
  DataTable,
  DetailTabs,
  Dialog,
  EmptyState,
  FilterChip,
  EntityList,
  JsonInspector,
  Kbd,
  Popover,
  SectionHeader,
  SectionTabs,
  SegmentedControl,
  Skeleton,
  StatCard,
  StatusBadge,
  SurfaceCard,
  Tooltip,
  useToast,
} from './components/ui'
import { useTheme } from './theme/ThemeProvider'
import { useDensity } from './theme/DensityProvider'
import { statusHelp } from './statusCopy'
import { RbacMatrix } from './sections/access/RbacMatrix'
import { CommandPalette, type PaletteCommand } from './components/CommandPalette'
import { ControlPanelTopNav } from './components/ControlPanelTopNav'
import { ResourceSearch } from './components/ResourceSearch'
import {
  SECTION_META,
  parseSectionFromPath,
  sectionToPath,
  type Section,
} from './navigation'
import type {
  AccessMembership,
  AccessPrincipal,
  AccessRole,
  AccessRoleBinding,
  AccessRolePermission,
  AdminField,
  AdminOverview,
  ArchitectureActivity,
  ArchitectureEdge,
  ArchitectureNode,
  ArchitectureSnapshot,
  CapabilitySectionStatus,
  CanonicalRoutingPath,
  CollectionHealthReport,
  CollectionOperationResult,
  CollectionSummary,
  ConfigValidationResult,
  ControlPanelCapabilities,
  EffectiveAccessPayload,
  GraphAssistantPayload,
  GraphDetailPayload,
  GraphIndexRecord,
  GraphProgressPayload,
  GraphIndexRunRecord,
  GraphResearchTunePayload,
  LangGraphExport,
  McpConnectionRecord,
  RegisteredSource,
  SourceRefreshRun,
  SourceScanPayload,
  UploadedFileSummary,
} from './types'

function useSectionRoute(): [Section, (next: Section) => void] {
  const location = useLocation()
  const navigate = useNavigate()
  const active = parseSectionFromPath(location.pathname)
  const setActive = useCallback(
    (next: Section) => {
      if (next === parseSectionFromPath(location.pathname)) return
      navigate(sectionToPath(next))
    },
    [navigate, location.pathname],
  )
  return [active, setActive]
}

const NEW_SKILL_TEMPLATE = `# New Skill
agent_scope: general
description: Temporary skill created from the control panel.

## Workflow

- Describe the reusable steps here.
`

const AGENT_EDITOR_FIELDS = [
  'description',
  'prompt_file',
  'skill_scope',
  'allowed_tools',
  'allowed_worker_agents',
  'preload_skill_packs',
  'memory_scopes',
  'max_steps',
  'max_tool_calls',
] as const

const ARCHITECTURE_LAYERS = [
  { id: 'entry', label: 'Entry' },
  { id: 'routing', label: 'Routing' },
  { id: 'top_level', label: 'Top-Level Agents' },
  { id: 'workers', label: 'Workers' },
  { id: 'services', label: 'Runtime Services' },
] as const

type ArchitectureTab = 'map' | 'agent-graph' | 'paths' | 'traffic'
type AgentsTab = 'workspace' | 'catalog'
type PromptsTab = 'edit' | 'compare'
type CollectionsTab = 'workspace'
type CollectionActionMode = 'upload' | 'local' | 'registered' | 'sync'
type GraphsTab = 'workspace' | 'runs'
type GraphSourceMode = 'collection' | 'manual'
type KnowledgeSourceKind = 'local_folder' | 'local_repo'
type SkillsTab = 'editor' | 'preview'
type AccessResourceType = 'agent' | 'agent_group' | 'collection' | 'graph' | 'skill' | 'skill_family' | 'tool' | 'tool_group' | 'worker_request'
type AccessAction = 'use' | 'manage' | 'approve' | 'delete'
type AccessTab = 'overview' | 'users' | 'groups' | 'roles' | 'grants' | 'effective' | 'advanced'
type AccessWizardMode = 'setup' | 'grant' | 'user' | 'group' | null
type AccessGroupPurpose = 'permission' | 'team' | 'admin'
type AccessPresetId = 'kb_reader' | 'graph_builder' | 'agent_operator' | 'approval_manager' | 'custom'
type IngestionWizardStep = 'collection' | 'source' | 'graph' | 'tuning' | 'review'
type OperationsTab = 'reloads' | 'jobs' | 'audit'

const COLLECTION_ACTION_MODES: CollectionActionMode[] = ['upload', 'local', 'registered', 'sync']
const INGESTION_WIZARD_STEPS: IngestionWizardStep[] = ['collection', 'source', 'graph', 'tuning', 'review']
const ACCESS_TABS: Array<{ id: AccessTab; label: string }> = [
  { id: 'overview', label: 'Overview' },
  { id: 'users', label: 'Users' },
  { id: 'groups', label: 'Groups' },
  { id: 'roles', label: 'Roles / Presets' },
  { id: 'grants', label: 'Resource Grants' },
  { id: 'effective', label: 'Effective Access' },
  { id: 'advanced', label: 'Matrix / Audit' },
]
const ACCESS_RESOURCE_TYPES: Array<{ key: AccessResourceType; label: string }> = [
  { key: 'agent', label: 'Agent' },
  { key: 'agent_group', label: 'Agent Group' },
  { key: 'collection', label: 'Collection' },
  { key: 'graph', label: 'Graph' },
  { key: 'skill', label: 'Skill' },
  { key: 'skill_family', label: 'Skill Family' },
  { key: 'tool', label: 'Tool' },
  { key: 'tool_group', label: 'Tool Group' },
  { key: 'worker_request', label: 'Worker Request' },
]
const ACCESS_ACTIONS: Array<{ key: AccessAction; label: string }> = [
  { key: 'use', label: 'Use' },
  { key: 'manage', label: 'Manage' },
  { key: 'approve', label: 'Approve' },
  { key: 'delete', label: 'Delete' },
]
const ACCESS_GROUP_PURPOSES: Array<{ key: AccessGroupPurpose; label: string; description: string }> = [
  { key: 'permission', label: 'Permission Group', description: 'Best for capability and resource access.' },
  { key: 'team', label: 'Team / Sharing Group', description: 'Best for organizing people and shared work.' },
  { key: 'admin', label: 'Admin Group', description: 'Best for operators who manage policy.' },
]
const ACCESS_PRESETS: Array<{
  id: AccessPresetId
  label: string
  description: string
  resourceType: AccessResourceType
  actions: AccessAction[]
}> = [
  { id: 'kb_reader', label: 'KB Reader', description: 'Use selected knowledge base collections.', resourceType: 'collection', actions: ['use'] },
  { id: 'graph_builder', label: 'Graph Builder', description: 'Use and manage selected GraphRAG graphs.', resourceType: 'graph', actions: ['use', 'manage'] },
  { id: 'agent_operator', label: 'Agent Operator', description: 'Use selected agents or agent groups.', resourceType: 'agent', actions: ['use'] },
  { id: 'approval_manager', label: 'Approval Manager', description: 'Approve worker-request queues.', resourceType: 'worker_request', actions: ['approve'] },
  { id: 'custom', label: 'Custom', description: 'Choose resource type, actions, and selectors manually.', resourceType: 'collection', actions: ['use'] },
]
const SUPPORTED_DOCUMENT_TYPES = ['.txt', '.md', '.csv', '.pdf', '.docx', '.xls', '.xlsx', 'OCR images'] as const
const GRAPH_RESEARCH_TUNE_TARGETS = [
  'extract_graph.txt',
  'summarize_descriptions.txt',
  'community_report_text.txt',
  'community_report_graph.txt',
  'local_search_system_prompt.txt',
  'global_search_map_system_prompt.txt',
  'global_search_reduce_system_prompt.txt',
  'global_search_knowledge_system_prompt.txt',
  'drift_search_system_prompt.txt',
  'drift_reduce_prompt.txt',
  'basic_search_system_prompt.txt',
] as const

type ResourceGrantSummary = {
  roleNames: string[]
  principalNames: string[]
}

function normalizeCollectionId(value: string): string {
  return value.trim()
}

function getMessage(error: unknown): string {
  if (isApiError(error)) return error.message
  if (error instanceof Error) return error.message
  return 'Unknown error'
}

function formatArchitectureRefreshError(errorMessages: string[]): string {
  if (errorMessages.length === 0) return ''
  if (errorMessages.length === 1) return errorMessages[0]
  return `Architecture refresh partially failed: ${errorMessages.join(' | ')}`
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

function asString(value: unknown, fallback = ''): string {
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return fallback
}

function configFieldName(field: AdminField): string {
  const record = field as unknown as Record<string, unknown>
  return asString(field.env_name || record.key || field.label, 'setting')
}

function matchesTextQuery(query: string, ...parts: unknown[]): boolean {
  const normalized = query.trim().toLowerCase()
  if (!normalized) return true
  return parts.some(part => asString(part).toLowerCase().includes(normalized))
}

function asNumber(value: unknown): number | null {
  if (typeof value === 'number') return value
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

function asArray<T = unknown>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : []
}

function extractSkillDependencyError(error: unknown): Record<string, unknown> | null {
  if (!isApiError(error)) return null
  const payload = asRecord(error.data)
  const detail = asRecord(payload?.detail) ?? payload
  return detail && asRecord(detail.dependency_validation) ? detail : null
}

function humanizeKey(value: string): string {
  return value
    .toLowerCase()
    .replace(/_/g, ' ')
    .replace(/\b\w/g, character => character.toUpperCase())
}

function shortList(items: string[]): string {
  if (items.length === 0) return 'None'
  return items.join(', ')
}

function formatTimestamp(value: unknown): string {
  const text = asString(value)
  if (!text) return 'No recent activity'
  const parsed = Date.parse(text)
  if (Number.isNaN(parsed)) return text
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(new Date(parsed))
}

function formatPercent(value: unknown): string {
  const parsed = asNumber(value)
  if (parsed === null) return 'Unscored'
  return `${Math.round(parsed * 100)}%`
}

function formatWholeNumber(value: unknown): string {
  const parsed = asNumber(value)
  if (parsed === null) return '0'
  return Math.round(parsed).toLocaleString()
}

function toneForStatus(value: unknown): 'neutral' | 'ok' | 'warning' | 'danger' | 'accent' {
  const normalized = asString(value).toLowerCase()
  if (['ok', 'active', 'success', 'configured', 'ready'].includes(normalized)) return 'ok'
  if (['warning', 'pending', 'runtime_swap', 'insufficient_data', 'partial'].includes(normalized)) return 'warning'
  if (['failed', 'error', 'archived', 'disabled', 'blocked', 'flagged'].includes(normalized)) return 'danger'
  if (['overlay active', 'overlay', 'live', 'preview'].includes(normalized)) return 'accent'
  return 'neutral'
}

function boolLabel(value: boolean, truthy: string, falsy: string): string {
  return value ? truthy : falsy
}

function maskSecret(field: AdminField, value: string): string {
  if (!field.secret) return value || 'Not set'
  if (value) return 'configured'
  return field.is_configured ? 'configured' : 'Not set'
}

function groupFields(fields: AdminField[]): Array<[string, AdminField[]]> {
  const groups = new Map<string, AdminField[]>()
  for (const field of fields) {
    const list = groups.get(field.group) ?? []
    list.push(field)
    groups.set(field.group, list)
  }
  return Array.from(groups.entries())
}

function shortId(value: string): string {
  return value.length > 18 ? `${value.slice(0, 8)}…${value.slice(-6)}` : value
}

function uniqueList(items: string[]): string[] {
  return Array.from(new Set(items.filter(Boolean)))
}

function normalizeToolTag(value: unknown): string {
  return asString(value).trim().replace(/\s+/g, ' ')
}

function collectToolTags(tool: Record<string, unknown>): string[] {
  const metadata = asRecord(tool.metadata) ?? {}
  const group = normalizeToolTag(tool.group || 'general')
  const tags = [
    group,
    ...asArray(tool.tags),
    ...asArray(tool.tool_tags),
    ...asArray(tool.keywords),
    ...asArray(metadata.tags),
    ...asArray(metadata.tool_tags),
    ...asArray(metadata.keywords),
    Boolean(tool.read_only) ? 'read only' : Boolean(tool.destructive) ? 'destructive' : 'mutable',
    Boolean(tool.background_safe) ? 'background safe' : 'foreground only',
    Boolean(tool.requires_workspace) ? 'needs workspace' : 'no workspace',
    Boolean(tool.should_defer || tool.deferred) ? 'deferred' : '',
  ]
  return uniqueList(tags.map(normalizeToolTag)).sort((left, right) => left.localeCompare(right))
}

function primaryToolTag(tool: Record<string, unknown>): string {
  return normalizeToolTag(tool.group || 'general') || 'general'
}

const TOOL_TAG_ACRONYMS = new Set(['api', 'csv', 'docx', 'json', 'kb', 'llm', 'mcp', 'nlp', 'ocr', 'pdf', 'rag', 'sql', 'xls', 'xlsx'])

function toolTagLabel(tag: string): string {
  const normalized = tag.replace(/[_-]/g, ' ')
  return normalized
    .split(' ')
    .filter(Boolean)
    .map(word => TOOL_TAG_ACRONYMS.has(word.toLowerCase()) ? word.toUpperCase() : humanizeKey(word))
    .join(' ')
}

function multilineList(value: string): string[] {
  return uniqueList(value.split('\n').map(item => item.trim()))
}

function accessResourceLabel(resourceType: string): string {
  const labels: Record<string, string> = {
    agent: 'Agent',
    agent_group: 'Agent Group',
    collection: 'Collection',
    graph: 'Graph',
    tool: 'Tool',
    tool_group: 'Tool Group',
    skill: 'Skill',
    skill_family: 'Skill Family',
    worker_request: 'Worker Request',
  }
  return labels[resourceType] ?? humanizeKey(resourceType)
}

function principalLabel(principal: AccessPrincipal | null | undefined): string {
  if (!principal) return 'Unknown principal'
  return principal.display_name || principal.email_normalized || principal.principal_id
}

function principalSystemRole(principal: AccessPrincipal | null | undefined): string {
  return asString(principal?.metadata_json?.system_role, principal?.principal_type === 'group' ? 'group' : 'user')
}

function groupPurpose(principal: AccessPrincipal | null | undefined): AccessGroupPurpose {
  const raw = asString(principal?.metadata_json?.purpose, 'permission').toLowerCase()
  if (raw === 'admin') return 'admin'
  if (raw === 'team') return 'team'
  return 'permission'
}

function groupPurposeLabel(value: string): string {
  return ACCESS_GROUP_PURPOSES.find(purpose => purpose.key === value)?.label ?? humanizeKey(value)
}

function computeResourceGrantSummary(options: {
  resourceType: string
  resourceId: string
  permissions: AccessRolePermission[]
  bindings: AccessRoleBinding[]
  roles: AccessRole[]
  principals: AccessPrincipal[]
}): ResourceGrantSummary {
  const resourceId = options.resourceId.trim()
  if (!resourceId) return { roleNames: [], principalNames: [] }
  const matchingPermissions = options.permissions.filter(permission => (
    permission.resource_type === options.resourceType
    && permission.action === 'use'
    && (permission.resource_selector === '*' || permission.resource_selector === resourceId)
  ))
  const roleById = new Map(options.roles.map(role => [role.role_id, role]))
  const principalById = new Map(options.principals.map(principal => [principal.principal_id, principal]))
  const roleNames = uniqueList(
    matchingPermissions
      .map(permission => roleById.get(permission.role_id)?.name || permission.role_id)
      .filter(Boolean),
  )
  const matchedRoleIds = new Set(matchingPermissions.map(permission => permission.role_id))
  const principalNames = uniqueList(
    options.bindings
      .filter(binding => matchedRoleIds.has(binding.role_id) && !binding.disabled)
      .map(binding => {
        const principal = principalById.get(binding.principal_id)
        return principal?.display_name || principal?.email_normalized || binding.principal_id
      })
      .filter(Boolean),
  )
  return { roleNames, principalNames }
}

function parseJsonObject(text: string): Record<string, unknown> {
  const trimmed = text.trim()
  if (!trimmed) return {}
  const parsed = JSON.parse(trimmed)
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Expected a JSON object.')
  }
  return parsed as Record<string, unknown>
}

function summarizeRouterMode(router: Record<string, unknown> | undefined): string {
  return asString(router?.mode_label, 'Deterministic only')
}

function nodeTone(node: ArchitectureNode | null): 'neutral' | 'ok' | 'warning' | 'danger' | 'accent' {
  if (!node) return 'neutral'
  if (node.kind === 'router') return 'accent'
  if (node.kind === 'service') return 'ok'
  if (node.kind === 'agent' && node.overlay_active) return 'accent'
  if (node.kind === 'agent' && node.mode === 'basic') return 'warning'
  return 'neutral'
}

function edgeTone(edge: ArchitectureEdge | null): 'neutral' | 'ok' | 'warning' | 'danger' | 'accent' {
  if (!edge) return 'neutral'
  if (edge.kind === 'routing_path') return 'accent'
  if (edge.kind === 'delegation') return 'warning'
  if (edge.kind === 'service_dependency') return 'ok'
  return 'neutral'
}

function langGraphStatusTone(exportPayload: LangGraphExport | null | undefined): 'neutral' | 'ok' | 'warning' | 'danger' | 'accent' {
  const status = asString(exportPayload?.status, 'unavailable').toLowerCase()
  if (status === 'available') return 'ok'
  if (status === 'error' || status === 'failed') return 'danger'
  return 'warning'
}

function langGraphItemLabel(item: Record<string, unknown>): string {
  return asString(item.name ?? item.id, 'node')
}

function formatSectionList(sectionIds: string[]): string {
  return sectionIds.map(humanizeKey).join(', ')
}

const EMPTY_COLLECTION_STORAGE_PROFILE = {
  vector_store_backend: 'pgvector',
  tables: ['documents', 'chunks'],
  embeddings_provider: '',
  embedding_model: '',
  graph_embedding_model: '',
  configured_embedding_dim: 0,
  actual_embedding_dims: {} as Record<string, number>,
  mismatch_warnings: [] as string[],
}

function collectionStorageProfile(collection: CollectionSummary | null | undefined) {
  const profile = asRecord(collection?.storage_profile)
  if (!profile) return EMPTY_COLLECTION_STORAGE_PROFILE
  return {
    vector_store_backend: asString(profile.vector_store_backend ?? profile.vector_store, EMPTY_COLLECTION_STORAGE_PROFILE.vector_store_backend),
    tables: asArray<string>(profile.tables ?? profile.table_names),
    embeddings_provider: asString(profile.embeddings_provider ?? profile.embedding_provider),
    embedding_model: asString(profile.embedding_model),
    graph_embedding_model: asString(profile.graph_embedding_model),
    configured_embedding_dim: asNumber(profile.configured_embedding_dim ?? profile.expected_embedding_dims) ?? 0,
    actual_embedding_dims: asRecord(profile.actual_embedding_dims) as Record<string, number> ?? {},
    mismatch_warnings: asArray<string>(profile.mismatch_warnings),
  }
}

function collectionStatusSummary(collection: CollectionSummary | null | undefined) {
  const status = asRecord(collection?.status)
  return {
    ready: Boolean(status?.ready),
    reason: asString(status?.reason, 'unknown'),
  }
}

function graphQualityIssues(detail: GraphDetailPayload | null): string[] {
  if (!detail?.graph) return []
  const graphRecord = detail.graph as unknown as Record<string, unknown>
  const health = asRecord((detail as unknown as Record<string, unknown>).health) ?? asRecord(graphRecord.health) ?? {}
  const issues: string[] = []
  const sourceCount = asArray<string>(graphRecord.source_doc_ids).length
  if (sourceCount > 0 && sourceCount < 3) {
    issues.push('Very small source set can produce sparse communities and weak relationships.')
  }
  if (!Boolean(graphRecord.query_ready)) {
    issues.push('Query-ready artifacts are not available yet.')
  }
  const entityCount = asNumber(health.entity_count ?? graphRecord.entity_count)
  const relationshipCount = asNumber(health.relationship_count ?? graphRecord.relationship_count)
  if (entityCount !== null && entityCount === 0) {
    issues.push('No entities were detected in the latest graph artifacts.')
  }
  if (relationshipCount !== null && relationshipCount === 0) {
    issues.push('No relationships were detected in the latest graph artifacts.')
  }
  if (Boolean(health.community_report_fallback ?? graphRecord.community_report_fallback)) {
    issues.push('Community reports used a fallback path, so summaries may be shallow.')
  }
  if (Boolean(health.stale_sources ?? graphRecord.stale_sources)) {
    issues.push('The graph may be stale relative to the source collection.')
  }
  for (const warning of asArray<string>(health.warnings ?? graphRecord.warnings).slice(0, 4)) {
    if (warning) issues.push(warning)
  }
  return uniqueList(issues)
}

function buildCompatibilityBanner(
  compatibility: ControlPanelCapabilities | null,
  source: 'capabilities' | 'openapi' | null,
): string {
  if (!compatibility || !source) return ''
  const unsupported = Object.entries(compatibility.sections)
    .filter(([, section]) => !section.supported)
    .map(([sectionName]) => sectionName)
  if (unsupported.length > 0) {
    return `Connected backend is partially compatible. Unsupported sections: ${formatSectionList(unsupported)}. Restart the API from this repo to load the full control-panel contract.`
  }
  if (source === 'openapi') {
    return 'Connected backend does not expose the compatibility contract yet. Section support is being inferred from openapi.json, so restarting the API from this repo is still recommended.'
  }
  return ''
}

function buildUnsupportedSectionMessage(
  section: Section,
  status: CapabilitySectionStatus | null,
  source: 'capabilities' | 'openapi' | null,
): { title: string; body: string } {
  const missingRoutes = status?.missing_routes ?? []
  const missingList = missingRoutes.length > 0 ? missingRoutes.join(', ') : 'the required routes'
  if (section === 'architecture') {
    return {
      title: 'Architecture is unavailable on this backend',
      body: `This backend cannot serve the live architecture views right now. Missing routes: ${missingList}. Restart the API from this repo so the architecture contract is available.`,
    }
  }
  return {
    title: `${humanizeKey(section)} is unavailable on this backend`,
    body: source === 'openapi'
      ? `This section is unsupported on the connected backend, based on the currently advertised openapi.json route set. Missing routes: ${missingList}. Restart the API from this repo to restore the shipped control-panel contract.`
      : `This section is unsupported on the connected backend. Missing routes: ${missingList}. Restart the API from this repo to restore the shipped control-panel contract.`,
  }
}

function useSessionToken(): [string, (token: string) => void] {
  const [token, setTokenState] = useState(() => sessionStorage.getItem('control-panel-token') ?? '')
  const setToken = (nextToken: string) => {
    sessionStorage.setItem('control-panel-token', nextToken)
    setTokenState(nextToken)
  }
  return [token, setToken]
}

function isCompactViewport(): boolean {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false
  return window.matchMedia('(max-width: 1180px)').matches
}

function useSessionStringState<T extends string>(key: string, initialValue: T): [T, (nextValue: T) => void] {
  const [value, setValueState] = useState<T>(() => (sessionStorage.getItem(key) as T | null) ?? initialValue)
  const setValue = (nextValue: T) => {
    sessionStorage.setItem(key, nextValue)
    setValueState(nextValue)
  }
  return [value, setValue]
}

function useSessionBooleanState(key: string, initialValue: boolean): [boolean, (nextValue: boolean | ((current: boolean) => boolean)) => void] {
  const [value, setValueState] = useState<boolean>(() => {
    const stored = sessionStorage.getItem(key)
    if (stored === 'true') return true
    if (stored === 'false') return false
    return initialValue
  })
  const setValue = (nextValue: boolean | ((current: boolean) => boolean)) => {
    setValueState(current => {
      const resolved = typeof nextValue === 'function' ? nextValue(current) : nextValue
      sessionStorage.setItem(key, String(resolved))
      return resolved
    })
  }
  return [value, setValue]
}

export default function App() {
  const [token, setToken] = useSessionToken()
  const [draftToken, setDraftToken] = useState(token)
  const [active, setActive] = useSectionRoute()
  const { theme, toggleTheme } = useTheme()
  const { density, toggleDensity } = useDensity()
  const toast = useToast()
  const notifyOk = useCallback((title: string, body?: string) => { toast.push({ title, body, tone: 'ok' }) }, [toast])
  const notifyError = useCallback((title: string, err: unknown) => { toast.push({ title, body: getMessage(err), tone: 'danger' }) }, [toast])
  const [pendingConfirm, setPendingConfirm] = useState<{
    title: string
    description?: string
    confirmLabel?: string
    run: () => void | Promise<void>
  } | null>(null)
  const [confirmLoading, setConfirmLoading] = useState(false)
  const askConfirm = useCallback((spec: {
    title: string
    description?: string
    confirmLabel?: string
    run: () => void | Promise<void>
  }) => { setPendingConfirm(spec) }, [])
  const handleConfirmRun = useCallback(async () => {
    if (!pendingConfirm) return
    setConfirmLoading(true)
    try { await pendingConfirm.run() }
    finally { setConfirmLoading(false); setPendingConfirm(null) }
  }, [pendingConfirm])
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [shortcutsOpen, setShortcutsOpen] = useState(false)
  const [configDiffOpen, setConfigDiffOpen] = useState(false)
  const [auditRange, setAuditRange] = useState<'24h' | '7d' | '30d' | 'all'>('all')

  const paletteCommands: PaletteCommand[] = useMemo(() => {
    const cmds: PaletteCommand[] = []
    for (const section of SECTION_META) {
      cmds.push({
        id: `nav:${section.id}`,
        label: `Go to ${section.label}`,
        hint: section.eyebrow,
        group: 'Navigate',
        keywords: [section.label, section.eyebrow, section.id],
        run: () => setActive(section.id),
      })
    }
    cmds.push({
      id: 'action:lock',
      label: 'Lock session',
      group: 'Actions',
      keywords: ['logout', 'sign out', 'token'],
      run: () => setToken(''),
    })
    cmds.push({
      id: 'action:theme',
      label: theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme',
      group: 'Actions',
      keywords: ['theme', 'dark', 'light', 'appearance'],
      run: () => toggleTheme(),
    })
    cmds.push({
      id: 'action:density',
      label: density === 'comfortable' ? 'Switch to compact density' : 'Switch to comfortable density',
      group: 'Actions',
      keywords: ['density', 'compact', 'spacing'],
      run: () => toggleDensity(),
    })
    return cmds
  }, [setActive, setToken, toggleTheme, toggleDensity, theme, density])
  const [error, setError] = useState('')
  const [compatibilityChecked, setCompatibilityChecked] = useState(false)
  const [compatibility, setCompatibility] = useState<ControlPanelCapabilities | null>(null)
  const [compatibilitySource, setCompatibilitySource] = useState<'capabilities' | 'openapi' | null>(null)
  const [overview, setOverview] = useState<AdminOverview | null>(null)
  const [operations, setOperations] = useState<Record<string, unknown> | null>(null)
  const [architecture, setArchitecture] = useState<ArchitectureSnapshot | null>(null)
  const [architectureActivity, setArchitectureActivity] = useState<ArchitectureActivity | null>(null)
  const [architectureRefreshing, setArchitectureRefreshing] = useState(false)
  const [architectureTab, setArchitectureTab] = useSessionStringState<ArchitectureTab>('control-panel-ui-architecture-tab', 'map')
  const [selectedArchitectureNodeId, setSelectedArchitectureNodeId] = useState('')
  const [selectedArchitectureEdgeId, setSelectedArchitectureEdgeId] = useState('')
  const [selectedArchitecturePathId, setSelectedArchitecturePathId] = useState('')
  const [configFields, setConfigFields] = useState<AdminField[]>([])
  const [configEffective, setConfigEffective] = useState<Record<string, string>>({})
  const [configChanges, setConfigChanges] = useState<Record<string, string>>({})
  const [configPreview, setConfigPreview] = useState<ConfigValidationResult | null>(null)
  const [activeConfigGroup, setActiveConfigGroup] = useState('')
  const [settingsSearch, setSettingsSearch] = useSessionStringState<string>('control-panel-ui-settings-search', '')
  const [agentsPayload, setAgentsPayload] = useState<{ agents: Array<Record<string, unknown>>; tools: Array<Record<string, unknown>> } | null>(null)
  const [selectedAgent, setSelectedAgent] = useState('')
  const [agentSearch, setAgentSearch] = useSessionStringState<string>('control-panel-ui-agent-search', '')
  const [toolTagFilter, setToolTagFilter] = useSessionStringState<string>('control-panel-ui-tool-tag-filter', '')
  const [agentDetail, setAgentDetail] = useState<Record<string, unknown> | null>(null)
  const [agentForm, setAgentForm] = useState<Record<string, unknown>>({})
  const [prompts, setPrompts] = useState<Array<Record<string, unknown>>>([])
  const [selectedPrompt, setSelectedPrompt] = useState('')
  const [promptSearch, setPromptSearch] = useSessionStringState<string>('control-panel-ui-prompt-search', '')
  const [promptDetail, setPromptDetail] = useState<Record<string, unknown> | null>(null)
  const [promptDraft, setPromptDraft] = useState('')
  const [collections, setCollections] = useState<CollectionSummary[]>([])
  const [selectedCollection, setSelectedCollection] = useState('')
  const [collectionSearch, setCollectionSearch] = useSessionStringState<string>('control-panel-ui-collection-search', '')
  const [collectionDraft, setCollectionDraft] = useState('')
  const [collectionDetail, setCollectionDetail] = useState<CollectionSummary | null>(null)
  const [collectionActivity, setCollectionActivity] = useState<CollectionOperationResult | Record<string, unknown> | null>(null)
  const [collectionDocs, setCollectionDocs] = useState<Array<Record<string, unknown>>>([])
  const [documentSearch, setDocumentSearch] = useState('')
  const [documentSourceFilter, setDocumentSourceFilter] = useState('all')
  const [selectedDoc, setSelectedDoc] = useState('')
  const [docDetail, setDocDetail] = useState<Record<string, unknown> | null>(null)
  const [collectionHealth, setCollectionHealth] = useState<CollectionHealthReport | null>(null)
  const [pathDraft, setPathDraft] = useState('')
  const [knowledgeSourceKind, setKnowledgeSourceKind] = useSessionStringState<KnowledgeSourceKind>('control-panel-ui-knowledge-source-kind', 'local_folder')
  const [sourceIncludeGlobs, setSourceIncludeGlobs] = useState('')
  const [sourceExcludeGlobs, setSourceExcludeGlobs] = useState('node_modules/**\n.git/**\ndist/**\nbuild/**')
  const [sourceScan, setSourceScan] = useState<SourceScanPayload | null>(null)
  const [registeredSources, setRegisteredSources] = useState<RegisteredSource[]>([])
  const [sourceRuns, setSourceRuns] = useState<SourceRefreshRun[]>([])
  const [allowedSourceRoots, setAllowedSourceRoots] = useState<string[]>([])
  const [selectedSourceId, setSelectedSourceId] = useState('')
  const [metadataProfile, setMetadataProfile] = useSessionStringState<string>('control-panel-ui-metadata-profile', 'auto')
  const [indexPreview, setIndexPreview] = useSessionBooleanState('control-panel-ui-index-preview', false)
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFileSummary[]>([])
  const [uploadSearch, setUploadSearch] = useSessionStringState<string>('control-panel-ui-upload-search', '')
  const [selectedUploadDoc, setSelectedUploadDoc] = useState('')
  const [uploadDocDetail, setUploadDocDetail] = useState<Record<string, unknown> | null>(null)
  const [uploadActivity, setUploadActivity] = useState<CollectionOperationResult | Record<string, unknown> | null>(null)
  const [graphs, setGraphs] = useState<GraphIndexRecord[]>([])
  const [selectedGraph, setSelectedGraph] = useState('')
  const [graphSearch, setGraphSearch] = useSessionStringState<string>('control-panel-ui-graph-search', '')
  const [graphDetail, setGraphDetail] = useState<GraphDetailPayload | null>(null)
  const [graphValidation, setGraphValidation] = useState<Record<string, unknown> | null>(null)
  const [graphRuns, setGraphRuns] = useState<GraphIndexRunRecord[]>([])
  const [graphProgress, setGraphProgress] = useState<GraphProgressPayload | null>(null)
  const [deleteGraphArtifacts, setDeleteGraphArtifacts] = useState(false)
  const [graphLifecycleBusy, setGraphLifecycleBusy] = useState(false)
  const [graphCollectionId, setGraphCollectionId] = useState('')
  const [graphCollectionDocs, setGraphCollectionDocs] = useState<Array<Record<string, unknown>>>([])
  const [graphSelectedDocIds, setGraphSelectedDocIds] = useState<string[]>([])
  const [graphDraftId, setGraphDraftId] = useState('')
  const [graphDisplayNameDraft, setGraphDisplayNameDraft] = useState('')
  const [graphPromptDraft, setGraphPromptDraft] = useState('{}')
  const [graphConfigDraft, setGraphConfigDraft] = useState('{}')
  const [graphSkillIdsDraft, setGraphSkillIdsDraft] = useState('')
  const [graphSkillOverlayDraft, setGraphSkillOverlayDraft] = useState('')
  const [graphIntent, setGraphIntent] = useSessionStringState<string>('control-panel-ui-graph-intent', 'general')
  const [graphAssistantPreflight, setGraphAssistantPreflight] = useState<GraphAssistantPayload | null>(null)
  const [graphSmokeQuery, setGraphSmokeQuery] = useState('What are the main entities and relationships in this graph?')
  const [graphSmokeResult, setGraphSmokeResult] = useState<GraphAssistantPayload | null>(null)
  const [graphAdvancedOpen, setGraphAdvancedOpen] = useSessionBooleanState('control-panel-ui-graph-advanced-open', false)
  const [graphTuneGuidance, setGraphTuneGuidance] = useState('')
  const [graphTuneTargets, setGraphTuneTargets] = useState<string[]>(['extract_graph.txt'])
  const [graphTuneResult, setGraphTuneResult] = useState<GraphResearchTunePayload | null>(null)
  const [graphTuneSelectedPrompts, setGraphTuneSelectedPrompts] = useState<string[]>([])
  const [graphTuneRunning, setGraphTuneRunning] = useState(false)
  const [ingestionWizardOpen, setIngestionWizardOpen] = useState(false)
  const [ingestionWizardStep, setIngestionWizardStep] = useState<IngestionWizardStep>('collection')
  const [wizardCollectionId, setWizardCollectionId] = useState('')
  const [wizardCreateGraph, setWizardCreateGraph] = useState(true)
  const [wizardStartBuild, setWizardStartBuild] = useState(false)
  const [wizardRunTune, setWizardRunTune] = useState(false)
  const [wizardApplyTune, setWizardApplyTune] = useState(false)
  const [wizardRequireTuneBeforeBuild, setWizardRequireTuneBeforeBuild] = useState(false)
  const [skills, setSkills] = useState<Array<Record<string, unknown>>>([])
  const [selectedSkill, setSelectedSkill] = useState('')
  const [skillSearch, setSkillSearch] = useSessionStringState<string>('control-panel-ui-skill-search', '')
  const [skillDetail, setSkillDetail] = useState<Record<string, unknown> | null>(null)
  const [skillEditor, setSkillEditor] = useState('')
  const [skillPreviewQuery, setSkillPreviewQuery] = useState('')
  const [skillPreviewResult, setSkillPreviewResult] = useState<Record<string, unknown> | null>(null)
  const [creatingSkill, setCreatingSkill] = useState(false)
  const [skillActionDetail, setSkillActionDetail] = useState<Record<string, unknown> | null>(null)
  const [accessPrincipals, setAccessPrincipals] = useState<AccessPrincipal[]>([])
  const [accessMemberships, setAccessMemberships] = useState<AccessMembership[]>([])
  const [accessRoles, setAccessRoles] = useState<AccessRole[]>([])
  const [accessBindings, setAccessBindings] = useState<AccessRoleBinding[]>([])
  const [accessPermissions, setAccessPermissions] = useState<AccessRolePermission[]>([])
  const [accessPreviewEmail, setAccessPreviewEmail] = useState('')
  const [effectiveAccess, setEffectiveAccess] = useState<EffectiveAccessPayload | null>(null)
  const [accessWizardPreview, setAccessWizardPreview] = useState<EffectiveAccessPayload | null>(null)
  const [mcpConnections, setMcpConnections] = useState<McpConnectionRecord[]>([])
  const [selectedMcpConnection, setSelectedMcpConnection] = useState('')
  const [mcpSearch, setMcpSearch] = useSessionStringState<string>('control-panel-ui-mcp-search', '')
  const [mcpDraftName, setMcpDraftName] = useState('')
  const [mcpDraftUrl, setMcpDraftUrl] = useState('')
  const [mcpDraftSecret, setMcpDraftSecret] = useState('')
  const [mcpDraftAgents, setMcpDraftAgents] = useState('general')
  const [mcpDraftVisibility, setMcpDraftVisibility] = useState('private')
  const [principalDraftType, setPrincipalDraftType] = useState<'user' | 'group'>('user')
  const [principalDraftValue, setPrincipalDraftValue] = useState('')
  const [principalDraftProvider, setPrincipalDraftProvider] = useState('email')
  const [roleDraftName, setRoleDraftName] = useState('')
  const [roleDraftDescription, setRoleDraftDescription] = useState('')
  const [bindingRoleId, setBindingRoleId] = useState('')
  const [bindingPrincipalId, setBindingPrincipalId] = useState('')
  const [permissionRoleId, setPermissionRoleId] = useState('')
  const [permissionResourceType, setPermissionResourceType] = useState<AccessResourceType>('collection')
  const [permissionAction, setPermissionAction] = useState<'use' | 'manage' | 'approve' | 'delete'>('use')
  const [permissionSelector, setPermissionSelector] = useState('')
  const [membershipParentId, setMembershipParentId] = useState('')
  const [membershipChildId, setMembershipChildId] = useState('')
  const [accessTab, setAccessTab] = useSessionStringState<AccessTab>('control-panel-ui-access-tab', 'overview')
  const [accessUserSearch, setAccessUserSearch] = useSessionStringState<string>('control-panel-ui-access-user-search', '')
  const [accessGroupSearch, setAccessGroupSearch] = useSessionStringState<string>('control-panel-ui-access-group-search', '')
  const [accessGrantSearch, setAccessGrantSearch] = useSessionStringState<string>('control-panel-ui-access-grant-search', '')
  const [accessWizard, setAccessWizard] = useState<AccessWizardMode>(null)
  const [accessWizardStep, setAccessWizardStep] = useState('')
  const [setupGroupMode, setSetupGroupMode] = useState<'existing' | 'create'>('existing')
  const [setupGroupId, setSetupGroupId] = useState('')
  const [setupGroupName, setSetupGroupName] = useState('')
  const [setupGroupPurpose, setSetupGroupPurpose] = useState<AccessGroupPurpose>('permission')
  const [setupPresetId, setSetupPresetId] = useState<AccessPresetId>('kb_reader')
  const [setupMemberIds, setSetupMemberIds] = useState<string[]>([])
  const [setupResourceType, setSetupResourceType] = useState<AccessResourceType>('collection')
  const [setupResourceSelectors, setSetupResourceSelectors] = useState<string[]>([])
  const [setupActions, setSetupActions] = useState<AccessAction[]>(['use'])
  const [grantResourceType, setGrantResourceType] = useState<AccessResourceType>('collection')
  const [grantResourceSelectors, setGrantResourceSelectors] = useState<string[]>([])
  const [grantPrincipalIds, setGrantPrincipalIds] = useState<string[]>([])
  const [grantActions, setGrantActions] = useState<AccessAction[]>(['use'])
  const [manageUserEmail, setManageUserEmail] = useState('')
  const [manageUserDisplayName, setManageUserDisplayName] = useState('')
  const [manageUserSystemRole, setManageUserSystemRole] = useState<'admin' | 'user' | 'pending'>('user')
  const [manageUserGroupIds, setManageUserGroupIds] = useState<string[]>([])
  const [createGroupName, setCreateGroupName] = useState('')
  const [createGroupPurpose, setCreateGroupPurpose] = useState<AccessGroupPurpose>('permission')
  const [createGroupRoleId, setCreateGroupRoleId] = useState('')
  const [createGroupMemberIds, setCreateGroupMemberIds] = useState<string[]>([])
  const [agentsTab, setAgentsTab] = useSessionStringState<AgentsTab>('control-panel-ui-agents-tab', 'workspace')
  const [promptsTab, setPromptsTab] = useSessionStringState<PromptsTab>('control-panel-ui-prompts-tab', 'edit')
  const [collectionsTab, setCollectionsTab] = useSessionStringState<CollectionsTab>('control-panel-ui-collections-tab', 'workspace')
  const [collectionAction, setCollectionAction] = useSessionStringState<CollectionActionMode>('control-panel-ui-collection-action', 'upload')
  const [graphsTab, setGraphsTab] = useSessionStringState<GraphsTab>('control-panel-ui-graphs-tab', 'workspace')
  const [graphSourceMode, setGraphSourceMode] = useSessionStringState<GraphSourceMode>('control-panel-ui-graph-source-mode', 'collection')
  const [skillsTab, setSkillsTab] = useSessionStringState<SkillsTab>('control-panel-ui-skills-tab', 'editor')
  const [operationsTab, setOperationsTab] = useSessionStringState<OperationsTab>('control-panel-ui-operations-tab', 'reloads')
  const [dashboardReloadOpen, setDashboardReloadOpen] = useSessionBooleanState('control-panel-ui-dashboard-reload-open', true)
  const [dashboardActivityOpen, setDashboardActivityOpen] = useSessionBooleanState('control-panel-ui-dashboard-activity-open', true)
  const [architectureInspectorOpen, setArchitectureInspectorOpen] = useSessionBooleanState('control-panel-ui-architecture-inspector-open', !isCompactViewport())
  const [configPreviewOpen, setConfigPreviewOpen] = useSessionBooleanState('control-panel-ui-config-preview-open', false)
  const [agentEditorOpen, setAgentEditorOpen] = useSessionBooleanState('control-panel-ui-agent-editor-open', true)
  const [agentInspectorOpen, setAgentInspectorOpen] = useSessionBooleanState('control-panel-ui-agent-inspector-open', !isCompactViewport())
  const [promptSummaryOpen, setPromptSummaryOpen] = useSessionBooleanState('control-panel-ui-prompt-summary-open', !isCompactViewport())
  const [collectionInspectorOpen, setCollectionInspectorOpen] = useSessionBooleanState('control-panel-ui-collection-inspector-open', false)
  const [collectionViewerOpen, setCollectionViewerOpen] = useSessionBooleanState('control-panel-ui-collection-viewer-open', !isCompactViewport())
  const [uploadViewerOpen, setUploadViewerOpen] = useSessionBooleanState('control-panel-ui-upload-viewer-open', !isCompactViewport())
  const [graphInspectorOpen, setGraphInspectorOpen] = useSessionBooleanState('control-panel-ui-graph-inspector-open', !isCompactViewport())
  const [skillSummaryOpen, setSkillSummaryOpen] = useSessionBooleanState('control-panel-ui-skill-summary-open', !isCompactViewport())
  const collectionFilesInputRef = useRef<HTMLInputElement | null>(null)
  const collectionFolderInputRef = useRef<HTMLInputElement | null>(null)
  const wizardFilesInputRef = useRef<HTMLInputElement | null>(null)
  const wizardFolderInputRef = useRef<HTMLInputElement | null>(null)
  const uploadFilesInputRef = useRef<HTMLInputElement | null>(null)
  const uploadFolderInputRef = useRef<HTMLInputElement | null>(null)

  const groupedConfigFields = useMemo(() => groupFields(configFields), [configFields])
  const filteredConfigGroups = useMemo(() => {
    const query = settingsSearch.trim().toLowerCase()
    if (!query) return groupedConfigFields
    return groupedConfigFields
      .map(([group, fields]) => {
        const groupMatches = group.toLowerCase().includes(query)
        const matchingFields = fields.filter(field => {
          return groupMatches
            || asString(field.label).toLowerCase().includes(query)
            || configFieldName(field).toLowerCase().includes(query)
            || asString(field.description).toLowerCase().includes(query)
        })
        return [group, matchingFields] as [string, AdminField[]]
      })
      .filter(([, fields]) => fields.length > 0)
  }, [groupedConfigFields, settingsSearch])
  const activeMeta = SECTION_META.find(item => item.id === active) ?? SECTION_META[0]
  const activeSectionSupport = compatibility?.sections?.[active] ?? null
  const activeSectionSupported = activeSectionSupport?.supported ?? true
  const unsupportedSectionIds = compatibility
    ? Object.entries(compatibility.sections)
        .filter(([, section]) => !section.supported)
        .map(([sectionName]) => sectionName as Section)
    : []
  const compatibilityBanner = buildCompatibilityBanner(compatibility, compatibilitySource)
  const unsupportedSectionMessage = !activeSectionSupported
    ? buildUnsupportedSectionMessage(active, activeSectionSupport, compatibilitySource)
    : null
  const visibleConfigGroup = filteredConfigGroups.find(([group]) => group === activeConfigGroup) ?? filteredConfigGroups[0] ?? null
  const architectureNodeMap = useMemo(() => new Map((architecture?.nodes ?? []).map(node => [node.id, node])), [architecture])
  const architectureEdgeMap = useMemo(() => new Map((architecture?.edges ?? []).map(edge => [edge.id, edge])), [architecture])
  const selectedArchitectureNode = architectureNodeMap.get(selectedArchitectureNodeId) ?? null
  const selectedArchitectureEdge = architectureEdgeMap.get(selectedArchitectureEdgeId) ?? null
  const selectedArchitecturePath = (architecture?.canonical_paths ?? []).find(path => path.id === selectedArchitecturePathId) ?? null
  const langGraphExport = architecture?.langgraph ?? null
  const langGraphNodes = asArray<Record<string, unknown>>(langGraphExport?.nodes)
  const langGraphEdges = asArray<Record<string, unknown>>(langGraphExport?.edges)
  const langGraphWarnings = asArray<string>(langGraphExport?.warnings)
  const architectureGraphStats = useMemo(() => {
    const nodes = architecture?.nodes ?? []
    const edges = architecture?.edges ?? []
    return {
      agents: nodes.filter(node => node.kind === 'agent').length,
      services: nodes.filter(node => node.kind === 'service').length,
      routingEdges: edges.filter(edge => edge.kind === 'routing_path').length,
      delegations: edges.filter(edge => edge.kind === 'delegation').length,
      serviceLinks: edges.filter(edge => edge.kind === 'service_dependency').length,
      totalEdges: edges.length,
    }
  }, [architecture])
  const highlightedArchitectureEdgeIds = useMemo(() => {
    if (selectedArchitecturePath?.edge_ids?.length) return new Set(selectedArchitecturePath.edge_ids)
    if (!selectedArchitectureNodeId) return new Set<string>()
    return new Set(
      (architecture?.edges ?? [])
        .filter(edge => edge.source === selectedArchitectureNodeId || edge.target === selectedArchitectureNodeId)
        .map(edge => edge.id),
    )
  }, [architecture, selectedArchitectureNodeId, selectedArchitecturePath])
  const highlightedArchitectureNodeIds = useMemo(() => {
    if (selectedArchitecturePath?.node_ids?.length) return new Set(selectedArchitecturePath.node_ids)
    if (!selectedArchitectureNodeId) return new Set<string>()
    return new Set([selectedArchitectureNodeId])
  }, [selectedArchitectureNodeId, selectedArchitecturePath])
  const lastReload = (operations?.last_reload as Record<string, unknown> | undefined)
    ?? (overview?.last_reload as Record<string, unknown> | undefined)
    ?? null
  const environmentLabel = (import.meta.env.VITE_ENVIRONMENT as string | undefined)?.toLowerCase() || 'local'
  const environmentTone: 'neutral' | 'info' | 'warn' | 'danger' =
    environmentLabel === 'prod' || environmentLabel === 'production' ? 'danger'
    : environmentLabel === 'staging' ? 'warn'
    : environmentLabel === 'dev' || environmentLabel === 'development' ? 'info'
    : 'neutral'
  const reloadStatusValue = asString(lastReload?.status, '').toLowerCase()
  const healthPulseTone: 'ok' | 'warn' | 'danger' =
    reloadStatusValue === 'failed' || reloadStatusValue === 'error' || reloadStatusValue === 'offline' ? 'danger'
    : reloadStatusValue === 'degraded' || reloadStatusValue === 'warning' || reloadStatusValue === 'healthy_with_overrides' ? 'warn'
    : 'ok'
  const healthPulseLabel = healthPulseTone === 'ok' ? 'healthy' : healthPulseTone === 'warn' ? 'degraded' : 'offline'
  const healthPulseTooltip = lastReload?.timestamp
    ? `Backend ${healthPulseLabel} • last checked ${formatTimestamp(lastReload.timestamp)}`
    : `Backend ${healthPulseLabel}`
  const isMacPlatform = typeof navigator !== 'undefined' && /Mac|iPod|iPhone|iPad/.test(navigator.platform)
  const auditEvents = asArray<Record<string, unknown>>(operations?.audit_events ?? overview?.audit_events)
  const jobs = asArray<Record<string, unknown>>(operations?.jobs ?? overview?.jobs)
  const schedulerSummary = asRecord(operations?.scheduler)
  const reviewBacklog = asRecord(architectureActivity?.review_backlog)
  const lastRetrainReport = asRecord(architectureActivity?.last_retrain_report)
  const recentMispicks = asArray<Record<string, unknown>>(architectureActivity?.recent_mispicks)
  const configDiffEntries = Object.entries(configPreview?.preview_diff ?? {})
  const selectedCollectionStatus = collections.find(collection => asString(collection.collection_id) === selectedCollection)
  const selectedCollectionMeta = collectionDetail ?? selectedCollectionStatus ?? null
  const selectedCollectionStorage = collectionStorageProfile(selectedCollectionMeta)
  const duplicateGroups = collectionHealth?.duplicate_groups ?? []
  const driftedGroups = collectionHealth?.drifted_groups ?? []
  const promptOverlayActive = Boolean(promptDetail?.overlay_active)
  const documentRecord = (docDetail?.document as Record<string, unknown> | undefined) ?? null
  const extractedContent = asString((docDetail?.extracted_content as Record<string, unknown> | undefined)?.content)
  const rawContent = asString((docDetail?.raw_source as Record<string, unknown> | undefined)?.content)
  const docMetadataSummary = asRecord(docDetail?.metadata_summary) ?? {}
  const uploadDocumentRecord = (uploadDocDetail?.document as Record<string, unknown> | undefined) ?? null
  const uploadExtractedContent = asString((uploadDocDetail?.extracted_content as Record<string, unknown> | undefined)?.content)
  const uploadRawContent = asString((uploadDocDetail?.raw_source as Record<string, unknown> | undefined)?.content)
  const uploadDocMetadataSummary = asRecord(uploadDocDetail?.metadata_summary) ?? {}
  const documentSourceTypes = useMemo(
    () => uniqueList(collectionDocs.map(doc => asString(doc.source_type)).filter(Boolean)),
    [collectionDocs],
  )
  const filteredCollectionDocs = useMemo(() => {
    return collectionDocs.filter(doc => {
      const matchesSearch = !documentSearch.trim()
        || asString(doc.title).toLowerCase().includes(documentSearch.trim().toLowerCase())
        || asString(doc.source_display_path || doc.source_path).toLowerCase().includes(documentSearch.trim().toLowerCase())
      const matchesSource = documentSourceFilter === 'all' || asString(doc.source_type) === documentSourceFilter
      return matchesSearch && matchesSource
    })
  }, [collectionDocs, documentSearch, documentSourceFilter])
  const filteredAgents = useMemo(() => {
    return (agentsPayload?.agents ?? []).filter(agent => matchesTextQuery(
      agentSearch,
      agent.name,
      agent.prompt_file,
      agent.description,
      agent.mode,
    ))
  }, [agentsPayload, agentSearch])
  const filteredPrompts = useMemo(() => {
    return prompts.filter(prompt => matchesTextQuery(
      promptSearch,
      prompt.prompt_file,
      prompt.kind,
      Boolean(prompt.overlay_active) ? 'overlay active' : 'base only',
    ))
  }, [prompts, promptSearch])
  const filteredCollections = useMemo(() => {
    return collections.filter(collection => matchesTextQuery(
      collectionSearch,
      collection.collection_id,
      collection.status?.reason,
      collection.maintenance_policy,
    ))
  }, [collections, collectionSearch])
  const filteredUploadedFiles = useMemo(() => {
    return uploadedFiles.filter(file => matchesTextQuery(
      uploadSearch,
      file.title,
      file.source_display_path,
      file.source_path,
      file.collection_id,
    ))
  }, [uploadedFiles, uploadSearch])
  const filteredGraphs = useMemo(() => {
    return graphs.filter(graph => matchesTextQuery(
      graphSearch,
      graph.graph_id,
      graph.display_name,
      graph.collection_id,
      graph.status,
    ))
  }, [graphs, graphSearch])
  const filteredSkills = useMemo(() => {
    return skills.filter(skill => matchesTextQuery(
      skillSearch,
      skill.skill_id,
      skill.name,
      skill.description,
      skill.status,
      skill.agent_scope,
    ))
  }, [skills, skillSearch])
  const filteredMcpConnections = useMemo(() => {
    return mcpConnections.filter(connection => matchesTextQuery(
      mcpSearch,
      connection.display_name,
      connection.connection_slug,
      connection.server_url,
      connection.status,
      connection.visibility,
    ))
  }, [mcpConnections, mcpSearch])
  const collectionActivityRecord = asRecord(collectionActivity)
  const collectionActivitySummary = asRecord(collectionActivityRecord?.summary)
  const collectionMetadataSummary = asRecord(collectionActivityRecord?.metadata_summary) ?? {}
  const collectionActivityFiles = asArray<Record<string, unknown>>(collectionActivityRecord?.files)
  const collectionActivityExceptions = collectionActivityFiles.filter(item => asString(item.outcome) !== 'ingested')
  const collectionActivityStatus = asString(collectionActivityRecord?.status)
  const activeCollectionAction = COLLECTION_ACTION_MODES.includes(collectionAction) ? collectionAction : 'local'
  const sourceScanSummary = sourceScan?.summary
  const filteredRegisteredSources = useMemo(() => {
    const collectionId = normalizeCollectionId(selectedCollection || collectionDraft)
    return registeredSources.filter(source => !collectionId || source.collection_id === collectionId)
  }, [registeredSources, selectedCollection, collectionDraft])
  const visibleRegisteredSource = filteredRegisteredSources.find(source => source.source_id === selectedSourceId)
    ?? filteredRegisteredSources[0]
    ?? null
  const selectedCollectionCanBuildGraph = (
    Boolean(normalizeCollectionId(selectedCollection || collectionDraft))
    && (
      Number(selectedCollectionMeta?.document_count ?? 0) > 0
      || collectionDocs.length > 0
      || Number(collectionActivitySummary?.ingested_count ?? collectionActivityRecord?.ingested_count ?? 0) > 0
      || Number(collectionActivitySummary?.already_indexed_count ?? collectionActivityRecord?.already_indexed_count ?? 0) > 0
    )
  )
  const selectedSourceRuns = sourceRuns.filter(run => run.source_id === (selectedSourceId || visibleRegisteredSource?.source_id))
  const uploadActivityRecord = asRecord(uploadActivity)
  const uploadActivitySummary = asRecord(uploadActivityRecord?.summary)
  const uploadMetadataSummary = asRecord(uploadActivityRecord?.metadata_summary) ?? {}
  const uploadActivityFiles = asArray<Record<string, unknown>>(uploadActivityRecord?.files)
  const uploadActivityExceptions = uploadActivityFiles.filter(item => asString(item.outcome) !== 'ingested')
  const uploadActivityStatus = asString(uploadActivityRecord?.status)
  const graphBuildDocCount = graphSourceMode === 'collection' ? graphCollectionDocs.length : graphSelectedDocIds.length
  const selectedGraphRecord = graphs.find(graph => graph.graph_id === selectedGraph) ?? null
  const activeGraphRun = graphProgress?.active_run
    ?? graphRuns.find(run => ['queued', 'running'].includes(asString(run.status).toLowerCase()))
    ?? null
  const graphBuildRunning = Boolean(
    selectedGraph
    && (
      graphProgress?.active
      || activeGraphRun
      || ['queued', 'running'].includes(asString(graphDetail?.graph.status || selectedGraphRecord?.status).toLowerCase())
    ),
  )
  const graphAssistantFriendly = asRecord(graphAssistantPreflight?.friendly)
  const graphSmokeFriendly = asRecord(graphSmokeResult?.friendly)
  const graphQualityWarnings = useMemo(() => graphQualityIssues(graphDetail), [graphDetail])
  const graphQualityHealth = asRecord(graphDetail?.graph.health)
  const graphTuneCoverage = asRecord(graphTuneResult?.coverage)
  const graphTuneCorpusProfile = asRecord(graphTuneResult?.corpus_profile)
  const graphTunePromptDraftEntries = Object.entries(graphTuneResult?.prompt_drafts ?? {})
  const graphTuneWarnings = asArray<string>(graphTuneResult?.warnings)
  const selectedSkillStatus = asString(skillDetail?.status, 'unknown')
  const skillDependencyValidation = asRecord(skillDetail?.dependency_validation)
  const skillHealth = asRecord(skillDetail?.skill_health)
  const skillActionValidation = asRecord(skillActionDetail?.dependency_validation)
  const selectedCollectionGrantSummary = useMemo(() => {
    return computeResourceGrantSummary({
      resourceType: 'collection',
      resourceId: asString(selectedCollectionMeta?.collection_id),
      permissions: accessPermissions,
      bindings: accessBindings,
      roles: accessRoles,
      principals: accessPrincipals,
    })
  }, [selectedCollectionMeta, accessPermissions, accessBindings, accessRoles, accessPrincipals])
  const selectedGraphGrantSummary = useMemo(() => {
    return computeResourceGrantSummary({
      resourceType: 'graph',
      resourceId: asString(graphDetail?.graph?.graph_id),
      permissions: accessPermissions,
      bindings: accessBindings,
      roles: accessRoles,
      principals: accessPrincipals,
    })
  }, [graphDetail, accessPermissions, accessBindings, accessRoles, accessPrincipals])
  const selectedSkillGrantSummary = useMemo(() => {
    return computeResourceGrantSummary({
      resourceType: 'skill_family',
      resourceId: asString(skillDetail?.version_parent || skillDetail?.skill_id),
      permissions: accessPermissions,
      bindings: accessBindings,
      roles: accessRoles,
      principals: accessPrincipals,
    })
  }, [skillDetail, accessPermissions, accessBindings, accessRoles, accessPrincipals])
  const toolCatalog = asArray<Record<string, unknown>>(agentsPayload?.tools)
  const toolCatalogItems = useMemo(() => {
    return toolCatalog.map(tool => ({
      tool,
      primaryTag: primaryToolTag(tool),
      tags: collectToolTags(tool),
    }))
  }, [toolCatalog])
  const toolTagOptions = useMemo(() => {
    const counts = new Map<string, number>()
    for (const item of toolCatalogItems) {
      for (const tag of item.tags) counts.set(tag, (counts.get(tag) ?? 0) + 1)
    }
    return Array.from(counts.entries())
      .map(([tag, count]) => ({ tag, count }))
      .sort((left, right) => {
        if (right.count !== left.count) return right.count - left.count
        return toolTagLabel(left.tag).localeCompare(toolTagLabel(right.tag))
      })
  }, [toolCatalogItems])
  const activeToolTag = toolTagOptions.some(option => option.tag === toolTagFilter) ? toolTagFilter : ''
  const filteredToolCatalogItems = useMemo(() => {
    if (!activeToolTag) return toolCatalogItems
    return toolCatalogItems.filter(item => item.tags.includes(activeToolTag))
  }, [activeToolTag, toolCatalogItems])
  const groupedToolCatalog = useMemo(() => {
    const groups = new Map<string, typeof toolCatalogItems>()
    for (const item of filteredToolCatalogItems) {
      const groupTag = activeToolTag || item.primaryTag
      const current = groups.get(groupTag) ?? []
      current.push(item)
      groups.set(groupTag, current)
    }
    return Array.from(groups.entries())
      .map(([tag, items]) => ({ tag, items }))
      .sort((left, right) => toolTagLabel(left.tag).localeCompare(toolTagLabel(right.tag)))
  }, [activeToolTag, filteredToolCatalogItems])
  const accessUsers = useMemo(() => {
    return accessPrincipals
      .filter(principal => principal.principal_type === 'user')
      .filter(principal => matchesTextQuery(accessUserSearch, principal.display_name, principal.email_normalized, principal.provider, principalSystemRole(principal)))
  }, [accessPrincipals, accessUserSearch])
  const accessGroups = useMemo(() => {
    return accessPrincipals
      .filter(principal => principal.principal_type === 'group')
      .filter(principal => matchesTextQuery(accessGroupSearch, principal.display_name, principal.principal_id, groupPurposeLabel(groupPurpose(principal))))
  }, [accessPrincipals, accessGroupSearch])
  const accessPrincipalById = useMemo(() => new Map(accessPrincipals.map(principal => [principal.principal_id, principal])), [accessPrincipals])
  const accessRoleById = useMemo(() => new Map(accessRoles.map(role => [role.role_id, role])), [accessRoles])
  const accessMembershipsByGroup = useMemo(() => {
    const map = new Map<string, AccessMembership[]>()
    for (const membership of accessMemberships) {
      const list = map.get(membership.parent_principal_id) ?? []
      list.push(membership)
      map.set(membership.parent_principal_id, list)
    }
    return map
  }, [accessMemberships])
  const accessBindingsByRole = useMemo(() => {
    const map = new Map<string, AccessRoleBinding[]>()
    for (const binding of accessBindings.filter(binding => !binding.disabled)) {
      const list = map.get(binding.role_id) ?? []
      list.push(binding)
      map.set(binding.role_id, list)
    }
    return map
  }, [accessBindings])
  const accessResourceOptions = useMemo(() => {
    const options: Record<AccessResourceType, Array<{ id: string; label: string; description: string }>> = {
      agent: (agentsPayload?.agents ?? []).map(agent => ({
        id: asString(agent.name),
        label: asString(agent.name, 'agent'),
        description: asString(agent.description, 'Agent runtime profile'),
      })).filter(option => option.id),
      agent_group: [
        { id: 'coordinator', label: 'Coordinator / Planner Workflow', description: 'Planner and coordinator dispatch path' },
        { id: 'rag', label: 'RAG Agents', description: 'Retrieval-oriented agents' },
        { id: 'data', label: 'Data Agents', description: 'Data and analysis workers' },
      ],
      collection: collections.map(collection => ({
        id: collection.collection_id,
        label: collection.collection_id,
        description: `${formatWholeNumber(collection.document_count)} documents`,
      })),
      graph: graphs.map(graph => ({
        id: graph.graph_id,
        label: graph.display_name || graph.graph_id,
        description: graph.collection_id,
      })),
      skill: skills.map(skill => ({
        id: asString(skill.skill_id),
        label: asString(skill.name, asString(skill.skill_id)),
        description: asString(skill.agent_scope, 'general'),
      })).filter(option => option.id),
      skill_family: uniqueList(skills.map(skill => asString(skill.version_parent || skill.skill_id)).filter(Boolean))
        .map(familyId => {
          const skill = skills.find(candidate => asString(candidate.version_parent || candidate.skill_id) === familyId)
          return {
            id: familyId,
            label: asString(skill?.name, familyId),
            description: 'Skill family',
          }
        }),
      tool: toolCatalog.map(tool => ({
        id: asString(tool.name),
        label: asString(tool.name, 'tool'),
        description: asString(tool.agent, asString(tool.description, 'Tool')),
      })).filter(option => option.id),
      tool_group: ['rag', 'orchestration', 'mcp', 'code', 'memory'].map(group => ({
        id: group,
        label: humanizeKey(group),
        description: 'Tool group',
      })),
      worker_request: [
        { id: '*', label: 'All Worker Requests', description: 'Every worker request queue' },
        { id: 'approval', label: 'Approval Queue', description: 'Human approval workflow' },
      ],
    }
    for (const key of Object.keys(options) as AccessResourceType[]) {
      options[key] = [{ id: '*', label: `All ${accessResourceLabel(key)}s`, description: 'Wildcard selector' }, ...options[key]]
    }
    return options
  }, [agentsPayload, collections, graphs, skills, toolCatalog])
  const accessResourceLabelByTypeAndId = useMemo(() => {
    const map = new Map<string, string>()
    for (const resourceType of Object.keys(accessResourceOptions) as AccessResourceType[]) {
      for (const option of accessResourceOptions[resourceType]) {
        map.set(`${resourceType}:${option.id}`, option.label)
      }
    }
    return map
  }, [accessResourceOptions])
  const accessGrantRows = useMemo(() => {
    const rows = accessPermissions.map(permission => {
      const bindings = accessBindingsByRole.get(permission.role_id) ?? []
      const principals = uniqueList(bindings.map(binding => principalLabel(accessPrincipalById.get(binding.principal_id))))
      const roleName = accessRoleById.get(permission.role_id)?.name || permission.role_id
      const selector = permission.resource_selector || '*'
      const selectorLabel = accessResourceLabelByTypeAndId.get(`${permission.resource_type}:${selector}`) || selector
      return {
        key: permission.permission_id,
        roleName,
        resourceType: permission.resource_type,
        resourceLabel: accessResourceLabel(permission.resource_type),
        selector,
        selectorLabel,
        action: permission.action,
        principalNames: principals,
      }
    })
    return rows.filter(row => matchesTextQuery(
      accessGrantSearch,
      row.roleName,
      row.resourceLabel,
      row.selector,
      row.selectorLabel,
      row.action,
      row.principalNames.join(' '),
    ))
  }, [accessPermissions, accessBindingsByRole, accessPrincipalById, accessRoleById, accessResourceLabelByTypeAndId, accessGrantSearch])
  const effectiveAccessRows = useMemo(() => {
    const resources = asRecord(asRecord(effectiveAccess?.access)?.resources) ?? {}
    return ACCESS_RESOURCE_TYPES.map(resourceType => {
      const payload = asRecord(resources[resourceType.key]) ?? {}
      const useIds = asArray<string>(payload.use)
      const manageIds = asArray<string>(payload.manage)
      const approveIds = asArray<string>(payload.approve)
      const deleteIds = asArray<string>(payload.delete)
      const allFlags = [
        Boolean(payload.use_all) ? 'use *' : '',
        Boolean(payload.manage_all) ? 'manage *' : '',
        Boolean(payload.approve_all) ? 'approve *' : '',
        Boolean(payload.delete_all) ? 'delete *' : '',
      ].filter(Boolean)
      const labels = uniqueList([...useIds, ...manageIds, ...approveIds, ...deleteIds].map(id => (
        accessResourceLabelByTypeAndId.get(`${resourceType.key}:${id}`) || id
      )))
      const why = accessPermissions
        .filter(permission => permission.resource_type === resourceType.key)
        .filter(permission => {
          if (allFlags.length > 0) return true
          return labels.length === 0
            ? false
            : permission.resource_selector === '*' || labels.includes(accessResourceLabelByTypeAndId.get(`${resourceType.key}:${permission.resource_selector}`) || permission.resource_selector)
        })
        .map(permission => `${permission.action} ${permission.resource_selector || '*'} via ${accessRoleById.get(permission.role_id)?.name || permission.role_id}`)
      return {
        key: resourceType.key,
        label: `Allowed ${resourceType.label}s`,
        allowed: [...allFlags, ...labels],
        why: uniqueList(why),
      }
    })
  }, [effectiveAccess, accessPermissions, accessRoleById, accessResourceLabelByTypeAndId])
  const selectedMcpRecord = mcpConnections.find(connection => connection.connection_id === selectedMcpConnection) ?? null
  const architectureSupported = compatibility?.sections?.architecture?.supported ?? true
  const architectureMapLayout = useMemo(() => {
    return buildArchitectureMapLayout({
      layers: ARCHITECTURE_LAYERS,
      nodes: architecture?.nodes ?? [],
      edges: architecture?.edges ?? [],
      highlightedEdgeIds: highlightedArchitectureEdgeIds,
      highlightedNodeIds: highlightedArchitectureNodeIds,
    })
  }, [architecture, highlightedArchitectureEdgeIds, highlightedArchitectureNodeIds])
  const architectureFullGraphLayout = useMemo(() => {
    return buildArchitectureMapLayout({
      layers: ARCHITECTURE_LAYERS,
      nodes: architecture?.nodes ?? [],
      edges: architecture?.edges ?? [],
      highlightedEdgeIds: new Set<string>(),
      highlightedNodeIds: new Set<string>(),
    })
  }, [architecture])

  function collectionOutcomeTone(outcome: string): 'neutral' | 'ok' | 'warning' | 'danger' | 'accent' {
    if (outcome === 'ingested') return 'ok'
    if (outcome === 'already_indexed') return 'neutral'
    if (outcome === 'previewed') return 'accent'
    if (outcome === 'skipped') return 'warning'
    if (outcome === 'failed') return 'danger'
    return 'neutral'
  }

  function applyCollectionSelection(nextCollectionId: string) {
    const normalized = normalizeCollectionId(nextCollectionId)
    setSelectedCollection(normalized)
    setCollectionDraft(normalized)
    setSelectedDoc('')
    setDocDetail(null)
    setCollectionHealth(null)
    setCollectionDetail(null)
    setCollectionActivity(null)
  }

  function applyCollections(nextCollections: CollectionSummary[], preferredCollectionId = ''): string {
    setCollections(nextCollections)
    const candidate = normalizeCollectionId(preferredCollectionId || selectedCollection || collectionDraft)
    const nextCollectionId = candidate || asString(nextCollections[0]?.collection_id)
    setSelectedCollection(nextCollectionId)
    setCollectionDraft(nextCollectionId)
    if (!nextCollectionId) {
      setCollectionDocs([])
      setSelectedDoc('')
      setDocDetail(null)
      setCollectionHealth(null)
      setCollectionDetail(null)
    }
    return nextCollectionId
  }

  function applyCollectionDocuments(nextDocuments: Array<Record<string, unknown>>, preferredDocId = ''): string {
    setCollectionDocs(nextDocuments)
    const candidate = asString(preferredDocId || selectedDoc)
    const nextDocId = nextDocuments.some(document => asString(document.doc_id) === candidate)
      ? candidate
      : asString(nextDocuments[0]?.doc_id)
    setSelectedDoc(nextDocId)
    if (!nextDocId) setDocDetail(null)
    return nextDocId
  }

  function applyUploadedFiles(nextUploads: UploadedFileSummary[], preferredDocId = ''): string {
    setUploadedFiles(nextUploads)
    const candidate = asString(preferredDocId || selectedUploadDoc)
    const nextDocId = nextUploads.some(document => document.doc_id === candidate)
      ? candidate
      : asString(nextUploads[0]?.doc_id)
    setSelectedUploadDoc(nextDocId)
    if (!nextDocId) setUploadDocDetail(null)
    return nextDocId
  }

  async function refreshCollections(preferredCollectionId = ''): Promise<string> {
    const payload = await api.listCollections(token)
    return applyCollections(payload.collections, preferredCollectionId)
  }

  async function refreshUploadedFiles(preferredDocId = ''): Promise<string> {
    const payload = await api.listUploadedFiles(token)
    return applyUploadedFiles(payload.uploads, preferredDocId)
  }

  async function refreshSources(preferredSourceId = ''): Promise<string> {
    const payload = await api.listSources(token)
    setRegisteredSources(payload.sources)
    setSourceRuns(payload.runs)
    setAllowedSourceRoots(payload.allowed_roots)
    const candidate = preferredSourceId || selectedSourceId
    const nextSourceId = payload.sources.some(source => source.source_id === candidate)
      ? candidate
      : payload.sources[0]?.source_id ?? ''
    setSelectedSourceId(nextSourceId)
    return nextSourceId
  }

  async function refreshCollectionDetail(collectionId: string): Promise<CollectionSummary | null> {
    if (!collectionId) {
      setCollectionDetail(null)
      return null
    }
    const payload = await api.getCollection(token, collectionId)
    setCollectionDetail(payload.collection)
    return payload.collection
  }

  async function refreshCollectionDocuments(collectionId: string, preferredDocId = ''): Promise<string> {
    const payload = await api.listCollectionDocuments(token, collectionId)
    return applyCollectionDocuments(payload.documents, preferredDocId)
  }

  async function refreshCollectionHealth(collectionId: string): Promise<CollectionHealthReport> {
    const payload = await api.getCollectionHealth(token, collectionId)
    setCollectionHealth(payload)
    return payload
  }

  async function refreshCollectionWorkspace(collectionId: string, preferredDocId = ''): Promise<string> {
    const [, , nextDocId] = await Promise.all([
      refreshCollections(collectionId),
      refreshCollectionDetail(collectionId),
      refreshCollectionDocuments(collectionId, preferredDocId),
      refreshSources(),
    ])
    await refreshCollectionHealth(collectionId)
    return nextDocId
  }

  async function refreshGraphs(preferredGraphId = ''): Promise<string> {
    const payload = await api.listGraphs(token)
    setGraphs(payload.graphs)
    const candidate = preferredGraphId || selectedGraph
    const nextGraphId = payload.graphs.some(graph => graph.graph_id === candidate)
      ? candidate
      : payload.graphs[0]?.graph_id ?? ''
    setSelectedGraph(nextGraphId)
    if (!nextGraphId) {
      setGraphDetail(null)
      setGraphRuns([])
    }
    return nextGraphId
  }

  async function refreshSelectedGraph(graphId: string): Promise<GraphDetailPayload | null> {
    if (!graphId) {
      setGraphDetail(null)
      setGraphRuns([])
      setGraphProgress(null)
      return null
    }
    const [detail, runsPayload, progressPayload] = await Promise.all([
      api.getGraph(token, graphId),
      api.getGraphRuns(token, graphId),
      api.getGraphProgress(token, graphId),
    ])
    setGraphDetail(detail)
    hydrateGraphForm(detail)
    setGraphRuns(runsPayload.runs)
    setGraphProgress(progressPayload)
    return detail
  }

  async function refreshGraphCollectionDocs(collectionId: string): Promise<Array<Record<string, unknown>>> {
    if (!collectionId) {
      setGraphCollectionDocs([])
      return []
    }
    const payload = await api.listCollectionDocuments(token, collectionId)
    setGraphCollectionDocs(payload.documents)
    return payload.documents
  }

  function toggleGraphDocSelection(docId: string) {
    setGraphSelectedDocIds(current => (
      current.includes(docId)
        ? current.filter(item => item !== docId)
        : [...current, docId]
    ))
  }

  function hydrateGraphForm(detail: GraphDetailPayload) {
    const graph = detail.graph
    setGraphCollectionId(asString(graph.collection_id))
    setGraphSelectedDocIds(asArray<string>(graph.source_doc_ids))
    setGraphDraftId(asString(graph.graph_id))
    setGraphDisplayNameDraft(asString(graph.display_name))
    setGraphPromptDraft(JSON.stringify(graph.prompt_overrides_json ?? {}, null, 2))
    setGraphConfigDraft(JSON.stringify(graph.config_json ?? {}, null, 2))
    setGraphSkillIdsDraft(asArray<string>(graph.graph_skill_ids).join(', '))
  }

  function startNewSkillDraft() {
    setCreatingSkill(true)
    setSelectedSkill('')
    setSkillDetail(null)
    setSkillPreviewResult(null)
    setSkillActionDetail(null)
    setSkillEditor(NEW_SKILL_TEMPLATE)
  }

  function applyArchitectureSnapshot(snapshot: ArchitectureSnapshot) {
    setArchitecture(snapshot)
    const nodeIds = new Set(snapshot.nodes.map(node => node.id))
    const edgeIds = new Set(snapshot.edges.map(edge => edge.id))
    const pathIds = new Set(snapshot.canonical_paths.map(path => path.id))
    setSelectedArchitectureNodeId(current => (current && nodeIds.has(current) ? current : snapshot.nodes[0]?.id ?? ''))
    setSelectedArchitectureEdgeId(current => (current && edgeIds.has(current) ? current : ''))
    setSelectedArchitecturePathId(current => (current && pathIds.has(current) ? current : snapshot.canonical_paths[0]?.id ?? ''))
  }

  async function refreshArchitectureSnapshot() {
    const snapshot = await api.getArchitecture(token)
    applyArchitectureSnapshot(snapshot)
    return snapshot
  }

  async function refreshArchitectureActivity() {
    const activity = await api.getArchitectureActivity(token)
    setArchitectureActivity(activity)
    return activity
  }

  async function refreshArchitectureData() {
    if (!token) return
    setArchitectureRefreshing(true)
    try {
      const [snapshotResult, activityResult] = await Promise.allSettled([
        api.getArchitecture(token),
        api.getArchitectureActivity(token),
      ])
      const errorMessages: string[] = []
      let allSucceeded = true

      if (snapshotResult.status === 'fulfilled') {
        applyArchitectureSnapshot(snapshotResult.value)
      } else {
        allSucceeded = false
        errorMessages.push(getMessage(snapshotResult.reason))
      }

      if (activityResult.status === 'fulfilled') {
        setArchitectureActivity(activityResult.value)
      } else {
        allSucceeded = false
        errorMessages.push(getMessage(activityResult.reason))
      }

      if (allSucceeded) {
        setError('')
        return
      }

      setError(formatArchitectureRefreshError(errorMessages))
    } finally {
      setArchitectureRefreshing(false)
    }
  }

  async function refreshCompatibility() {
    const result = await api.inspectCompatibility(token)
    setCompatibility(result.capabilities)
    setCompatibilitySource(result.source)
    setCompatibilityChecked(true)
    return result
  }

  async function refreshAccessData() {
    const [principalsPayload, membershipsPayload, rolesPayload, bindingsPayload, permissionsPayload] = await Promise.all([
      api.listAccessPrincipals(token),
      api.listAccessMemberships(token),
      api.listAccessRoles(token),
      api.listAccessBindings(token),
      api.listAccessPermissions(token),
    ])
    setAccessPrincipals(principalsPayload.principals)
    setAccessMemberships(membershipsPayload.memberships)
    setAccessRoles(rolesPayload.roles)
    setAccessBindings(bindingsPayload.bindings)
    setAccessPermissions(permissionsPayload.permissions)
    setBindingRoleId(current => current || rolesPayload.roles[0]?.role_id || '')
    setPermissionRoleId(current => current || rolesPayload.roles[0]?.role_id || '')
    setBindingPrincipalId(current => current || principalsPayload.principals[0]?.principal_id || '')
    setMembershipParentId(current => current || principalsPayload.principals.find(principal => principal.principal_type === 'group')?.principal_id || '')
    setMembershipChildId(current => current || principalsPayload.principals.find(principal => principal.principal_type === 'user')?.principal_id || '')
  }

  async function refreshMcpData(preferredConnectionId = '') {
    const payload = await api.listMcpConnections(token)
    setMcpConnections(payload.connections)
    const candidate = preferredConnectionId || selectedMcpConnection
    const nextConnectionId = payload.connections.some(connection => connection.connection_id === candidate)
      ? candidate
      : payload.connections[0]?.connection_id ?? ''
    setSelectedMcpConnection(nextConnectionId)
    return nextConnectionId
  }

  function clearSectionData(section: Section) {
    if (section === 'architecture') {
      setArchitecture(null)
      setArchitectureActivity(null)
    }
  }

  useEffect(() => {
    if (!token) {
      setCompatibilityChecked(false)
      setCompatibility(null)
      setCompatibilitySource(null)
      return
    }
    setCompatibilityChecked(false)
    void refreshCompatibility().catch(err => {
      setCompatibilityChecked(true)
      setError(getMessage(err))
    })
  }, [token])

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setPaletteOpen(prev => !prev)
        return
      }
      if (e.key === '?' && !e.metaKey && !e.ctrlKey && !e.altKey) {
        const target = e.target as HTMLElement | null
        const tag = target?.tagName
        const editable = target?.isContentEditable
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || editable) return
        e.preventDefault()
        setShortcutsOpen(prev => !prev)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => {
    setError('')
    if (!token) return
    if (!compatibilityChecked) return
    if (activeSectionSupport && !activeSectionSupported) {
      clearSectionData(active)
      return
    }
    if (active === 'dashboard') {
      void api.getOverview(token).then(setOverview).catch(err => setError(getMessage(err)))
    }
    if (active === 'operations') {
      void api.getOperations(token).then(setOperations).catch(err => setError(getMessage(err)))
    }
    if (active === 'architecture') {
      void refreshArchitectureData()
    }
    if (active === 'config') {
      void Promise.all([api.getConfigSchema(token), api.getEffectiveConfig(token)])
        .then(([schema, effective]) => {
          setConfigFields(schema.fields)
          setConfigEffective(effective.values)
          setConfigPreview(null)
        })
        .catch(err => setError(getMessage(err)))
    }
    if (active === 'agents') {
      void api.listAgents(token).then(payload => {
        setAgentsPayload(payload)
        const keepSelected = payload.agents.some(agent => asString(agent.name) === selectedAgent)
        const first = asString(payload.agents[0]?.name)
        setSelectedAgent(keepSelected ? selectedAgent : first)
      }).catch(err => setError(getMessage(err)))
    }
    if (active === 'prompts') {
      void api.listPrompts(token).then(payload => {
        setPrompts(payload.prompts)
        const keepSelected = payload.prompts.some(prompt => asString(prompt.prompt_file) === selectedPrompt)
        const first = asString(payload.prompts[0]?.prompt_file)
        setSelectedPrompt(keepSelected ? selectedPrompt : first)
      }).catch(err => setError(getMessage(err)))
    }
    if (active === 'collections') {
      void Promise.all([api.listCollections(token), api.listSources(token)])
        .then(([collectionsPayload, sourcesPayload]) => {
          applyCollections(collectionsPayload.collections)
          setRegisteredSources(sourcesPayload.sources)
          setSourceRuns(sourcesPayload.runs)
          setAllowedSourceRoots(sourcesPayload.allowed_roots)
        })
        .catch(err => setError(getMessage(err)))
    }
    if (active === 'uploads') {
      void api.listUploadedFiles(token).then(payload => {
        applyUploadedFiles(payload.uploads)
      }).catch(err => setError(getMessage(err)))
    }
    if (active === 'graphs') {
      void Promise.all([api.listCollections(token), api.listGraphs(token), api.listSkills(token), api.listSources(token)])
        .then(([collectionsPayload, graphsPayload, skillsPayload, sourcesPayload]) => {
          applyCollections(collectionsPayload.collections)
          setGraphs(graphsPayload.graphs)
          setSkills(skillsPayload.data)
          setRegisteredSources(sourcesPayload.sources)
          setSourceRuns(sourcesPayload.runs)
          setAllowedSourceRoots(sourcesPayload.allowed_roots)
          const keepSelected = graphsPayload.graphs.some(graph => graph.graph_id === selectedGraph)
          const first = graphsPayload.graphs[0]?.graph_id ?? ''
          setSelectedGraph(keepSelected ? selectedGraph : first)
          if (!graphCollectionId) {
            setGraphCollectionId(asString(collectionsPayload.collections[0]?.collection_id))
          }
        })
        .catch(err => setError(getMessage(err)))
    }
    if (active === 'skills') {
      void api.listSkills(token).then(payload => {
        setSkills(payload.data)
        if (creatingSkill) return
        const keepSelected = payload.data.some(skill => asString(skill.skill_id) === selectedSkill)
        const first = asString(payload.data[0]?.skill_id)
        setSelectedSkill(keepSelected ? selectedSkill : first)
      }).catch(err => setError(getMessage(err)))
    }
    if (active === 'access') {
      void Promise.all([api.listCollections(token), api.listGraphs(token), api.listSkills(token), api.listAgents(token)])
        .then(([collectionsPayload, graphsPayload, skillsPayload, agentsPayload]) => {
          setCollections(collectionsPayload.collections)
          setGraphs(graphsPayload.graphs)
          setSkills(skillsPayload.data)
          setAgentsPayload(agentsPayload)
        })
        .catch(err => setError(getMessage(err)))
    }
    if (active === 'mcp') {
      void refreshMcpData().catch(err => setError(getMessage(err)))
    }
    if (active === 'access' || active === 'collections' || active === 'graphs' || active === 'skills') {
      void refreshAccessData().catch(err => setError(getMessage(err)))
    }
  }, [active, activeSectionSupport, activeSectionSupported, compatibilityChecked, token])

  useEffect(() => {
    if (!token || active !== 'agents' || !selectedAgent) return
    void api.getAgent(token, selectedAgent).then(detail => {
      setAgentDetail(detail)
      setAgentForm({
        description: detail.description ?? '',
        prompt_file: detail.prompt_file ?? '',
        skill_scope: detail.skill_scope ?? '',
        allowed_tools: asArray<string>(detail.allowed_tools).join(', '),
        allowed_worker_agents: asArray<string>(detail.allowed_worker_agents).join(', '),
        preload_skill_packs: asArray<string>(detail.preload_skill_packs).join(', '),
        memory_scopes: asArray<string>(detail.memory_scopes).join(', '),
        max_steps: asString(detail.max_steps),
        max_tool_calls: asString(detail.max_tool_calls),
        body: asString(detail.body),
      })
    }).catch(err => setError(getMessage(err)))
  }, [active, selectedAgent, token])

  useEffect(() => {
    if (!token || active !== 'prompts' || !selectedPrompt) return
    void api.getPrompt(token, selectedPrompt).then(detail => {
      setPromptDetail(detail)
      setPromptDraft(asString(detail.overlay_content || detail.effective_content))
    }).catch(err => setError(getMessage(err)))
  }, [active, selectedPrompt, token])

  useEffect(() => {
    if (!token || !selectedCollection) {
      setCollectionDocs([])
      setSelectedDoc('')
      setDocDetail(null)
      setCollectionDetail(null)
      setCollectionHealth(null)
      return
    }
    if (active !== 'collections') return
    void Promise.all([
      refreshCollectionDetail(selectedCollection),
      api.listCollectionDocuments(token, selectedCollection),
      refreshCollectionHealth(selectedCollection),
    ])
      .then(([, payload]) => {
        applyCollectionDocuments(payload.documents)
      })
      .catch(err => setError(getMessage(err)))
  }, [active, selectedCollection, token])

  useEffect(() => {
    if (!token || active !== 'collections' || !selectedCollection || !selectedDoc) return
    void api.getCollectionDocument(token, selectedCollection, selectedDoc).then(setDocDetail).catch(err => setError(getMessage(err)))
  }, [active, selectedCollection, selectedDoc, token])

  useEffect(() => {
    if (selectedDoc) setCollectionViewerOpen(true)
  }, [selectedDoc, setCollectionViewerOpen])

  useEffect(() => {
    if (!token || active !== 'uploads' || !selectedUploadDoc) return
    void api.getUploadedFile(token, selectedUploadDoc).then(setUploadDocDetail).catch(err => setError(getMessage(err)))
  }, [active, selectedUploadDoc, token])

  useEffect(() => {
    if (selectedUploadDoc) setUploadViewerOpen(true)
  }, [selectedUploadDoc, setUploadViewerOpen])

  useEffect(() => {
    if (!token || active !== 'skills' || !selectedSkill) return
    void api.getSkill(token, selectedSkill).then(detail => {
      setSkillDetail(detail)
      setSkillEditor(asString(detail.body_markdown))
      setCreatingSkill(false)
    }).catch(err => setError(getMessage(err)))
  }, [active, selectedSkill, token])

  useEffect(() => {
    if (!token || active !== 'graphs' || !selectedGraph) return
    void refreshSelectedGraph(selectedGraph)
      .then(detail => {
        if (!detail) return undefined
        const boundSkills = asArray<Record<string, unknown>>(detail.skills)
        const overlaySkill = boundSkills.find(skill => asString(skill.graph_id) === selectedGraph)
        if (overlaySkill) {
          return api.getSkill(token, asString(overlaySkill.skill_id)).then(skill => {
            setGraphSkillOverlayDraft(asString(skill.body_markdown))
          })
        }
        setGraphSkillOverlayDraft('')
        return undefined
      })
      .catch(err => setError(getMessage(err)))
  }, [active, selectedGraph, token])

  useEffect(() => {
    if (!token || active !== 'graphs' || !selectedGraph || !graphBuildRunning) return
    let cancelled = false
    const poll = async () => {
      try {
        const progressPayload = await api.getGraphProgress(token, selectedGraph)
        if (cancelled) return
        setGraphProgress(progressPayload)
        if (!progressPayload.active) {
          await refreshSelectedGraph(selectedGraph)
          await refreshGraphs(selectedGraph)
        }
      } catch (err) {
        if (!cancelled) setError(getMessage(err))
      }
    }
    void poll()
    const intervalId = window.setInterval(() => {
      void poll()
    }, 60000)
    return () => {
      cancelled = true
      window.clearInterval(intervalId)
    }
  }, [active, selectedGraph, token, graphBuildRunning])

  useEffect(() => {
    if (!token || active !== 'graphs' || !graphCollectionId) {
      if (active !== 'graphs') setGraphCollectionDocs([])
      return
    }
    void refreshGraphCollectionDocs(graphCollectionId).catch(err => setError(getMessage(err)))
  }, [active, graphCollectionId, token])

  useEffect(() => {
    if (active !== 'config') return
    const firstGroup = groupedConfigFields[0]?.[0] ?? ''
    if (!firstGroup) {
      if (activeConfigGroup) setActiveConfigGroup('')
      return
    }
    const hasActiveGroup = groupedConfigFields.some(([group]) => group === activeConfigGroup)
    if (!hasActiveGroup) setActiveConfigGroup(firstGroup)
  }, [active, activeConfigGroup, groupedConfigFields])

  useEffect(() => {
    if (!token || active !== 'architecture' || !architectureSupported) return
    const intervalId = window.setInterval(() => {
      void refreshArchitectureData()
    }, 15000)
    return () => window.clearInterval(intervalId)
  }, [active, architectureSupported, token])

  useEffect(() => {
    if (accessPreviewEmail) return
    const firstEmailPrincipal = accessPrincipals.find(principal => principal.email_normalized)?.email_normalized ?? ''
    if (firstEmailPrincipal) setAccessPreviewEmail(firstEmailPrincipal)
  }, [accessPreviewEmail, accessPrincipals])

  async function handleConfigValidate() {
    try {
      const result = await api.validateConfig(token, configChanges)
      setConfigPreview(result)
      setConfigPreviewOpen(true)
      setError('')
    } catch (err) {
      setError(getMessage(err))
    }
  }

  async function handleConfigApply() {
    try {
      const result = await api.applyConfig(token, configChanges)
      setConfigPreview(result)
      setConfigPreviewOpen(true)
      const effective = await api.getEffectiveConfig(token)
      setConfigEffective(effective.values)
      const refreshedOverview = await api.getOverview(token)
      setOverview(refreshedOverview)
      const nextCompatibility = await refreshCompatibility()
      if (nextCompatibility.capabilities.sections.architecture?.supported ?? true) {
        await refreshArchitectureSnapshot()
      }
      setConfigChanges({})
      setError('')
      notifyOk('Config applied', 'Runtime picked up the new values.')
    } catch (err) {
      setError(getMessage(err))
      notifyError('Config apply failed', err)
    }
  }

  async function handleAgentSave() {
    try {
      await api.updateAgent(token, selectedAgent, {
        description: agentForm.description,
        prompt_file: agentForm.prompt_file,
        skill_scope: agentForm.skill_scope,
        allowed_tools: asString(agentForm.allowed_tools).split(',').map(item => item.trim()).filter(Boolean),
        allowed_worker_agents: asString(agentForm.allowed_worker_agents).split(',').map(item => item.trim()).filter(Boolean),
        preload_skill_packs: asString(agentForm.preload_skill_packs).split(',').map(item => item.trim()).filter(Boolean),
        memory_scopes: asString(agentForm.memory_scopes).split(',').map(item => item.trim()).filter(Boolean),
        max_steps: Number(agentForm.max_steps || 0),
        max_tool_calls: Number(agentForm.max_tool_calls || 0),
        body: agentForm.body,
      })
      const detail = await api.getAgent(token, selectedAgent)
      setAgentDetail(detail)
      setError('')
      notifyOk('Agent saved', selectedAgent)
    } catch (err) {
      setError(getMessage(err))
      notifyError('Agent save failed', err)
    }
  }

  async function handleAgentReload() {
    try {
      const result = await api.reloadAgents(token)
      setOperations(current => ({ ...(current ?? {}), last_reload: result }))
      const refreshedOverview = await api.getOverview(token)
      setOverview(refreshedOverview)
      const detail = await api.getAgent(token, selectedAgent)
      setAgentDetail(detail)
      const nextCompatibility = await refreshCompatibility()
      if (nextCompatibility.capabilities.sections.architecture?.supported ?? true) {
        await refreshArchitectureSnapshot()
      }
      setError('')
      notifyOk('Agents reloaded')
    } catch (err) {
      setError(getMessage(err))
      notifyError('Reload failed', err)
    }
  }

  async function handlePromptSave() {
    try {
      await api.updatePrompt(token, selectedPrompt, promptDraft)
      const detail = await api.getPrompt(token, selectedPrompt)
      setPromptDetail(detail)
      setError('')
      notifyOk('Prompt saved', selectedPrompt)
    } catch (err) {
      setError(getMessage(err))
      notifyError('Prompt save failed', err)
    }
  }

  async function handlePromptReset() {
    try {
      await api.resetPrompt(token, selectedPrompt)
      const detail = await api.getPrompt(token, selectedPrompt)
      setPromptDetail(detail)
      setPromptDraft(asString(detail.effective_content))
      setError('')
      notifyOk('Prompt reset to default', selectedPrompt)
    } catch (err) {
      setError(getMessage(err))
      notifyError('Prompt reset failed', err)
    }
  }

  async function handleUseCollection() {
    const collectionId = normalizeCollectionId(collectionDraft)
    if (!collectionId) {
      setError('Choose a collection ID first.')
      return
    }
    applyCollectionSelection(collectionId)
    try {
      await refreshCollectionWorkspace(collectionId)
      setError('')
    } catch (err) {
      setError(getMessage(err))
    }
  }

  async function handleCreateCollection() {
    const collectionId = normalizeCollectionId(collectionDraft)
    if (!collectionId) {
      setError('Choose a collection ID first.')
      return
    }
    try {
      const result = await api.createCollection(token, collectionId)
      applyCollectionSelection(collectionId)
      await refreshCollectionWorkspace(collectionId)
      setCollectionActivity(result)
      setError('')
      notifyOk('Collection created', collectionId)
    } catch (err) {
      setError(getMessage(err))
      notifyError('Create collection failed', err)
    }
  }

  async function handleSyncCollection() {
    const collectionId = normalizeCollectionId(selectedCollection || collectionDraft)
    if (!collectionId) {
      setError('Choose a collection ID first.')
      return
    }
    try {
      const result = await api.syncCollection(token, collectionId)
      applyCollectionSelection(collectionId)
      await refreshCollectionWorkspace(collectionId)
      setCollectionActivity(result)
      setError('')
    } catch (err) {
      setError(getMessage(err))
    }
  }

  async function handlePathIngest() {
    const collectionId = normalizeCollectionId(selectedCollection || collectionDraft)
    if (!collectionId) {
      setError('Choose a collection ID first.')
      return
    }
    try {
      const result = await api.ingestPaths(
        token,
        collectionId,
        pathDraft.split('\n').map(item => item.trim()).filter(Boolean),
        metadataProfile,
        indexPreview,
        knowledgeSourceKind,
      )
      applyCollectionSelection(collectionId)
      await refreshCollectionWorkspace(collectionId)
      setCollectionActivity(result)
      setError('')
    } catch (err) {
      setError(getMessage(err))
    }
  }

  async function handleCollectionFilesUpload(files: FileList | File[] | null) {
    const collectionId = normalizeCollectionId(selectedCollection || collectionDraft)
    if (!collectionId) {
      setError('Choose a collection ID first.')
      return
    }
    if (!files || files.length === 0) return
    const fileArray = Array.isArray(files) ? files : Array.from(files)
    const relativePaths = fileArray.map(file => {
      const relativePath = asString((file as File & { webkitRelativePath?: string }).webkitRelativePath)
      return relativePath || file.name
    })
    try {
      const result = await api.uploadFiles(token, collectionId, fileArray, relativePaths, metadataProfile, indexPreview, 'collection_upload')
      applyCollectionSelection(collectionId)
      const preferredDocId = asString(asArray<string>(result.doc_ids)[0])
      await refreshCollectionWorkspace(collectionId, preferredDocId)
      setCollectionActivity(result)
      setError('')
      notifyOk(indexPreview ? 'Upload preview ready' : 'Documents indexed', collectionId)
    } catch (err) {
      setError(getMessage(err))
      notifyError('Collection upload failed', err)
    }
  }

  function sourceGlobs(value: string): string[] {
    return multilineList(value)
  }

  async function handleSourceScan() {
    const collectionId = normalizeCollectionId(selectedCollection || collectionDraft)
    if (!collectionId) {
      setError('Choose a collection ID first.')
      return
    }
    const paths = multilineList(pathDraft)
    if (paths.length === 0) {
      setError('Add at least one local folder or repository path first.')
      return
    }
    try {
      const scan = await api.scanSource(token, {
        paths,
        source_kind: knowledgeSourceKind,
        collection_id: collectionId,
        include_globs: sourceGlobs(sourceIncludeGlobs),
        exclude_globs: sourceGlobs(sourceExcludeGlobs),
        metadata_profile: metadataProfile,
      })
      setSourceScan(scan)
      setError('')
      notifyOk('Source preview ready', `${scan.summary.supported_count} supported`)
    } catch (err) {
      setError(getMessage(err))
      notifyError('Source preview failed', err)
    }
  }

  async function handleIndexLocalSource() {
    const collectionId = normalizeCollectionId(selectedCollection || collectionDraft)
    if (!collectionId) {
      setError('Choose a collection ID first.')
      return
    }
    const paths = multilineList(pathDraft)
    if (paths.length === 0) {
      setError('Add at least one local folder or repository path first.')
      return
    }
    try {
      const registered = await api.registerSource(token, {
        paths,
        source_kind: knowledgeSourceKind,
        collection_id: collectionId,
        include_globs: sourceGlobs(sourceIncludeGlobs),
        exclude_globs: sourceGlobs(sourceExcludeGlobs),
        metadata_profile: metadataProfile,
      })
      setSourceScan(registered.scan)
      setSelectedSourceId(registered.source.source_id)
      await refreshSources(registered.source.source_id)
      const result = await api.refreshSource(token, registered.source.source_id, {
        metadata_profile: metadataProfile,
        index_preview: indexPreview,
        background: false,
      })
      const scan = asRecord(result.scan) as SourceScanPayload | null
      if (scan) setSourceScan(scan)
      applyCollectionSelection(collectionId)
      await refreshCollectionWorkspace(collectionId)
      await refreshSources(registered.source.source_id)
      setCollectionActivity(result)
      setError('')
      notifyOk(indexPreview ? 'Source index preview ready' : 'Source indexed', registered.source.display_name)
    } catch (err) {
      setError(getMessage(err))
      notifyError('Source index failed', err)
    }
  }

  async function handleRegisterSource() {
    const collectionId = normalizeCollectionId(selectedCollection || collectionDraft)
    if (!collectionId) {
      setError('Choose a collection ID first.')
      return
    }
    const paths = multilineList(pathDraft)
    if (paths.length === 0) {
      setError('Add at least one local folder or repository path first.')
      return
    }
    try {
      const result = await api.registerSource(token, {
        paths,
        source_kind: knowledgeSourceKind,
        collection_id: collectionId,
        include_globs: sourceGlobs(sourceIncludeGlobs),
        exclude_globs: sourceGlobs(sourceExcludeGlobs),
        metadata_profile: metadataProfile,
      })
      setSourceScan(result.scan)
      await refreshSources(result.source.source_id)
      setSelectedSourceId(result.source.source_id)
      setError('')
      notifyOk('Source registered', result.source.display_name)
    } catch (err) {
      setError(getMessage(err))
      notifyError('Source registration failed', err)
    }
  }

  async function handleRefreshSource(previewOnly = false, background = false) {
    const sourceId = selectedSourceId || filteredRegisteredSources[0]?.source_id
    if (!sourceId) {
      setError('Choose a registered source first.')
      return
    }
    try {
      const result = await api.refreshSource(token, sourceId, {
        metadata_profile: metadataProfile,
        index_preview: previewOnly || indexPreview,
        background,
      })
      const run = asRecord(result.run)
      if (run) {
        await refreshSources(sourceId)
        setCollectionActivity(result)
        setError('')
        notifyOk('Source refresh queued', sourceId)
        return
      }
      const scan = asRecord(result.scan) as SourceScanPayload | null
      if (scan) setSourceScan(scan)
      const collectionId = asString(result.collection_id, selectedCollection || collectionDraft)
      if (collectionId) {
        applyCollectionSelection(collectionId)
        await refreshCollectionWorkspace(collectionId)
      }
      await refreshSources(sourceId)
      setCollectionActivity(result)
      setError('')
      notifyOk(previewOnly || indexPreview ? 'Source refresh preview ready' : 'Registered source refreshed', sourceId)
    } catch (err) {
      setError(getMessage(err))
      notifyError('Source refresh failed', err)
    }
  }

  function openGraphBuilderForCollection(collectionId: string) {
    const normalized = normalizeCollectionId(collectionId)
    if (!normalized) return
    setGraphCollectionId(normalized)
    setGraphSourceMode('collection')
    setGraphSelectedDocIds([])
    setGraphDraftId('')
    setGraphDisplayNameDraft('')
    setGraphPromptDraft('{}')
    setGraphConfigDraft('{}')
    setGraphAssistantPreflight(null)
    setGraphSmokeResult(null)
    setActive('graphs')
  }

  async function handleGraphSuggest() {
    const collectionId = normalizeCollectionId(graphCollectionId || selectedCollection || collectionDraft)
    if (!collectionId) {
      setError('Choose a collection before generating graph defaults.')
      return
    }
    try {
      const payload = await api.suggestGraph(token, {
        collection_id: collectionId,
        intent: graphIntent,
        source_doc_ids: graphSourceMode === 'manual' ? graphSelectedDocIds : [],
      })
      setGraphCollectionId(collectionId)
      setGraphDraftId(asString(payload.graph_id, graphDraftId))
      setGraphDisplayNameDraft(asString(payload.display_name, graphDisplayNameDraft))
      setGraphConfigDraft(JSON.stringify(payload.config_overrides ?? {}, null, 2))
      setGraphPromptDraft(JSON.stringify(payload.prompt_overrides ?? {}, null, 2))
      if (graphSourceMode === 'manual') {
        setGraphSelectedDocIds(asArray<string>(payload.source_doc_ids))
      }
      setGraphAssistantPreflight(null)
      setGraphSmokeResult(null)
      setError('')
      notifyOk('Graph defaults suggested', `${formatWholeNumber(payload.source_count)} source documents`)
    } catch (err) {
      setError(getMessage(err))
      notifyError('Graph suggestion failed', err)
    }
  }

  async function handleUploadedFilesUpload(files: FileList | File[] | null) {
    if (!files || files.length === 0) return
    const fileArray = Array.isArray(files) ? files : Array.from(files)
    const relativePaths = fileArray.map(file => {
      const relativePath = asString((file as File & { webkitRelativePath?: string }).webkitRelativePath)
      return relativePath || file.name
    })
    try {
      const result = await api.uploadUploadedFiles(token, fileArray, relativePaths, '', metadataProfile, indexPreview)
      const preferredDocId = asString(asArray<string>(result.doc_ids)[0])
      await refreshUploadedFiles(preferredDocId)
      setUploadActivity(result)
      setError('')
    } catch (err) {
      setError(getMessage(err))
    }
  }

  async function handleReindexUploadedFile() {
    if (!selectedUploadDoc) return
    try {
      const result = await api.reindexUploadedFile(token, selectedUploadDoc)
      const ingestedIds = Array.isArray(result.ingested_doc_ids) ? result.ingested_doc_ids : []
      const preferredDocId = asString(ingestedIds[0])
      const nextDocId = await refreshUploadedFiles(preferredDocId)
      if (nextDocId) {
        const detail = await api.getUploadedFile(token, nextDocId)
        setUploadDocDetail(detail)
      } else {
        setUploadDocDetail(null)
      }
      setError('')
      notifyOk('Uploaded file reindexed', selectedUploadDoc)
    } catch (err) {
      setError(getMessage(err))
      notifyError('Upload reindex failed', err)
    }
  }

  async function handleDeleteUploadedFile() {
    if (!selectedUploadDoc) return
    try {
      await api.deleteUploadedFile(token, selectedUploadDoc)
      const nextDocId = await refreshUploadedFiles()
      if (nextDocId) {
        const detail = await api.getUploadedFile(token, nextDocId)
        setUploadDocDetail(detail)
      } else {
        setUploadDocDetail(null)
      }
      setError('')
      notifyOk('Uploaded file deleted')
    } catch (err) {
      setError(getMessage(err))
      notifyError('Upload delete failed', err)
    }
  }

  async function handleReindexDocument() {
    if (!selectedCollection || !selectedDoc) return
    try {
      const result = await api.reindexDocument(token, selectedCollection, selectedDoc)
      const ingestedIds = Array.isArray(result.ingested_doc_ids) ? result.ingested_doc_ids : []
      const preferredDocId = asString(ingestedIds[0])
      await refreshCollectionDetail(selectedCollection)
      await refreshCollections(selectedCollection)
      const nextDocId = await refreshCollectionDocuments(selectedCollection, preferredDocId)
      await refreshCollectionHealth(selectedCollection)
      if (nextDocId) {
        const detail = await api.getCollectionDocument(token, selectedCollection, nextDocId)
        setDocDetail(detail)
      } else {
        setDocDetail(null)
      }
      setError('')
      notifyOk('Document reindexed', selectedDoc)
    } catch (err) {
      setError(getMessage(err))
      notifyError('Reindex failed', err)
    }
  }

  async function handleDeleteDocument() {
    if (!selectedCollection || !selectedDoc) return
    try {
      await api.deleteDocument(token, selectedCollection, selectedDoc)
      await refreshCollectionDetail(selectedCollection)
      await refreshCollections(selectedCollection)
      const nextDocId = await refreshCollectionDocuments(selectedCollection)
      await refreshCollectionHealth(selectedCollection)
      if (nextDocId) {
        const detail = await api.getCollectionDocument(token, selectedCollection, nextDocId)
        setDocDetail(detail)
      } else {
        setDocDetail(null)
      }
      setError('')
      notifyOk('Document deleted')
    } catch (err) {
      setError(getMessage(err))
      notifyError('Delete failed', err)
    }
  }

  async function handleRepairCollection() {
    const collectionId = normalizeCollectionId(selectedCollection || collectionDraft)
    if (!collectionId) {
      setError('Choose a collection ID first.')
      return
    }
    try {
      const result = await api.repairCollection(token, collectionId)
      applyCollectionSelection(collectionId)
      await refreshCollectionWorkspace(collectionId)
      setCollectionActivity(result)
      setError('')
    } catch (err) {
      setError(getMessage(err))
    }
  }

  async function handleDeleteCollection() {
    const collectionId = normalizeCollectionId(selectedCollection || collectionDraft)
    if (!collectionId) {
      setError('Choose a collection ID first.')
      return
    }
    try {
      const result = await api.deleteCollection(token, collectionId)
      const nextCollectionId = await refreshCollections()
      setCollectionDraft(nextCollectionId)
      if (nextCollectionId) {
        await refreshCollectionWorkspace(nextCollectionId)
      } else {
        setCollectionDetail(null)
        setCollectionDocs([])
        setCollectionHealth(null)
        setSelectedDoc('')
        setDocDetail(null)
      }
      setCollectionActivity(result)
      setError('')
      notifyOk('Collection deleted', collectionId)
    } catch (err) {
      setError(getMessage(err))
      notifyError('Delete collection failed', err)
    }
  }

  async function handleGraphCreate() {
    const collectionId = normalizeCollectionId(graphCollectionId)
    if (!collectionId) {
      setError('Choose a collection ID for the graph.')
      return
    }
    try {
      const payload = await api.createGraph(token, {
        graph_id: graphDraftId,
        display_name: graphDisplayNameDraft,
        collection_id: collectionId,
        source_doc_ids: graphSourceMode === 'manual' ? graphSelectedDocIds : [],
        backend: 'microsoft_graphrag',
        visibility: 'tenant',
        config_overrides: parseJsonObject(graphConfigDraft),
        prompt_overrides: parseJsonObject(graphPromptDraft),
        graph_skill_ids: uniqueList(graphSkillIdsDraft.split(',').map(item => item.trim())),
      })
      const createdGraphId = asString(payload.graph_id)
      if (createdGraphId) {
        setSelectedGraph(createdGraphId)
        setGraphTuneResult(null)
        setGraphTuneSelectedPrompts([])
        setGraphAssistantPreflight(null)
        setGraphSmokeResult(null)
        await refreshGraphs(createdGraphId)
        const detail = await api.getGraph(token, createdGraphId)
        setGraphDetail(detail)
        hydrateGraphForm(detail)
      }
      setError('')
      notifyOk('Graph created', createdGraphId || graphDraftId)
    } catch (err) {
      setError(getMessage(err))
      notifyError('Create graph failed', err)
    }
  }

  function openIngestionWizard(collectionId = '') {
    const normalized = normalizeCollectionId(collectionId || selectedCollection || collectionDraft || graphCollectionId)
    setWizardCollectionId(normalized)
    if (normalized) {
      setCollectionDraft(normalized)
      setGraphCollectionId(normalized)
    }
    setIngestionWizardStep('collection')
    setIngestionWizardOpen(true)
  }

  async function ensureWizardCollection(collectionId: string): Promise<void> {
    const collectionExists = collections.some(collection => collection.collection_id === collectionId)
    if (!collectionExists) {
      await api.createCollection(token, collectionId)
    }
    applyCollectionSelection(collectionId)
    await refreshCollectionWorkspace(collectionId)
    setGraphCollectionId(collectionId)
  }

  async function ensureWizardGraphDraft(collectionId: string): Promise<string> {
    const currentGraphId = asString(graphDraftId || selectedGraph)
    if (currentGraphId) {
      try {
        const existing = await api.getGraph(token, currentGraphId)
        if (asString(existing.graph.collection_id) === collectionId) {
          setSelectedGraph(currentGraphId)
          setGraphDetail(existing)
          hydrateGraphForm(existing)
          return currentGraphId
        }
      } catch (err) {
        if (!isApiError(err) || err.status !== 404) throw err
      }
    }

    const suggestion = await api.suggestGraph(token, {
      collection_id: collectionId,
      intent: graphIntent,
      source_doc_ids: [],
    })
    const graphId = asString(suggestion.graph_id, graphDraftId)
    const displayName = asString(suggestion.display_name, graphDisplayNameDraft || graphId)
    setGraphDraftId(graphId)
    setGraphDisplayNameDraft(displayName)
    setGraphConfigDraft(JSON.stringify(suggestion.config_overrides ?? {}, null, 2))
    setGraphPromptDraft(JSON.stringify(suggestion.prompt_overrides ?? {}, null, 2))

    try {
      const existing = await api.getGraph(token, graphId)
      if (asString(existing.graph.collection_id) === collectionId) {
        setSelectedGraph(graphId)
        setGraphDetail(existing)
        hydrateGraphForm(existing)
        return graphId
      }
    } catch (err) {
      if (!isApiError(err) || err.status !== 404) throw err
    }

    const created = await api.createGraph(token, {
      graph_id: graphId,
      display_name: displayName,
      collection_id: collectionId,
      source_doc_ids: [],
      backend: 'microsoft_graphrag',
      visibility: 'tenant',
      config_overrides: suggestion.config_overrides ?? {},
      prompt_overrides: suggestion.prompt_overrides ?? {},
      graph_skill_ids: [],
    })
    const createdGraphId = asString(created.graph_id, graphId)
    setSelectedGraph(createdGraphId)
    await refreshGraphs(createdGraphId)
    await refreshSelectedGraph(createdGraphId)
    return createdGraphId
  }

  async function runGraphTuneForGraph(graphId: string): Promise<GraphResearchTunePayload> {
    if (graphTuneTargets.length === 0) {
      throw new Error('Choose at least one prompt target for Research & Tune.')
    }
    const result = await api.startGraphResearchTune(token, graphId, {
      guidance: graphTuneGuidance,
      target_prompt_files: graphTuneTargets,
    })
    setGraphTuneResult(result)
    const draftFiles = Object.entries(result.prompt_drafts ?? {})
      .filter(([, draft]) => {
        const draftRecord = asRecord(draft)
        const validation = asRecord(draftRecord?.validation)
        return validation?.ok !== false
      })
      .map(([promptFile]) => promptFile)
    setGraphTuneSelectedPrompts(draftFiles)
    const runsPayload = await api.getGraphRuns(token, graphId)
    setGraphRuns(runsPayload.runs)
    return result
  }

  async function handleWizardRunTune() {
    const collectionId = normalizeCollectionId(wizardCollectionId || selectedCollection || collectionDraft)
    if (!collectionId) {
      setError('Choose a collection before running prompt tuning.')
      return
    }
    if (!wizardCreateGraph) {
      setError('Enable graph draft creation before running prompt tuning.')
      return
    }
    setGraphTuneRunning(true)
    try {
      await ensureWizardCollection(collectionId)
      const graphId = await ensureWizardGraphDraft(collectionId)
      await runGraphTuneForGraph(graphId)
      setWizardRunTune(true)
      setError('')
      notifyOk('Prompt tuning run complete', graphId)
    } catch (err) {
      setError(getMessage(err))
      notifyError('Prompt tuning failed', err)
    } finally {
      setGraphTuneRunning(false)
    }
  }

  async function handleWizardFinish() {
    const collectionId = normalizeCollectionId(wizardCollectionId || selectedCollection || collectionDraft)
    if (!collectionId) {
      setError('Choose a collection ID before finishing the wizard.')
      return
    }
    try {
      await ensureWizardCollection(collectionId)
      if (collectionAction === 'sync') {
        const result = await api.syncCollection(token, collectionId)
        setCollectionActivity(result)
        await refreshCollectionWorkspace(collectionId)
      }
      if (wizardCreateGraph) {
        const createdGraphId = await ensureWizardGraphDraft(collectionId)
        let tuneResult = graphTuneResult
        if (wizardRunTune && asString(tuneResult?.graph_id) !== createdGraphId) {
          try {
            tuneResult = await runGraphTuneForGraph(createdGraphId)
          } catch (err) {
            tuneResult = null
            if (wizardRequireTuneBeforeBuild && wizardStartBuild) throw err
            setError(getMessage(err))
          }
        }
        if (wizardRunTune && wizardApplyTune && tuneResult?.run_id && graphTuneSelectedPrompts.length > 0) {
          await api.applyGraphResearchTune(token, createdGraphId, tuneResult.run_id, graphTuneSelectedPrompts)
          await refreshSelectedGraph(createdGraphId)
        }
        if (wizardStartBuild) {
          const response = await api.buildGraph(token, createdGraphId)
          setGraphValidation(response)
          await refreshSelectedGraph(createdGraphId)
        }
        setActive('graphs')
      } else {
        setActive('collections')
      }
      setIngestionWizardOpen(false)
      setError('')
      notifyOk('Ingestion wizard complete', collectionId)
    } catch (err) {
      setError(getMessage(err))
      notifyError('Wizard failed', err)
    }
  }

  async function handleGraphValidate() {
    if (!selectedGraph) {
      setError('Create or select a graph first.')
      return
    }
    try {
      const payload = await api.graphAssistantPreflight(token, selectedGraph)
      setGraphAssistantPreflight(payload)
      setGraphValidation(payload.validation ?? (payload as unknown as Record<string, unknown>))
      setError('')
      notifyOk('Graph preflight complete', asString(asRecord(payload.friendly)?.headline, selectedGraph))
    } catch (err) {
      setError(getMessage(err))
      notifyError('Graph preflight failed', err)
    }
  }

  async function handleGraphBuild(refresh = false) {
    if (!selectedGraph) {
      setError('Create or select a graph first.')
      return
    }
    if (graphBuildRunning) {
      setError('This graph already has a build or refresh run in progress.')
      return
    }
    setGraphLifecycleBusy(true)
    try {
      const response = refresh
        ? await api.refreshGraph(token, selectedGraph)
        : await api.buildGraph(token, selectedGraph)
      await refreshGraphs(selectedGraph)
      await refreshSelectedGraph(selectedGraph)
      setGraphValidation(response)
      if (asString(response.operation_status) !== 'already_running') try {
        const smoke = await api.graphSmokeTest(token, selectedGraph, {
          query: graphSmokeQuery,
          methods: [],
          limit: 6,
        })
        setGraphSmokeResult(smoke)
      } catch (smokeErr) {
        setGraphSmokeResult({
          graph_id: selectedGraph,
          query: graphSmokeQuery,
          friendly: {
            status: 'no_results',
            query_ready: false,
            result_count: 0,
            citation_count: 0,
            message: getMessage(smokeErr),
          },
        })
      }
      setError('')
      notifyOk(refresh ? 'Graph refresh started' : 'Graph build started', selectedGraph)
    } catch (err) {
      setError(getMessage(err))
      notifyError(refresh ? 'Graph refresh failed' : 'Graph build failed', err)
    } finally {
      setGraphLifecycleBusy(false)
    }
  }

  async function handleGraphCancelRun() {
    if (!selectedGraph || !activeGraphRun?.run_id) {
      setError('No active graph run is available to cancel.')
      return
    }
    setGraphLifecycleBusy(true)
    try {
      const payload = await api.cancelGraphRun(token, selectedGraph, activeGraphRun.run_id)
      setGraphValidation(payload)
      await refreshGraphs(selectedGraph)
      await refreshSelectedGraph(selectedGraph)
      setError('')
      notifyOk('Graph run cancelled', activeGraphRun.run_id)
    } catch (err) {
      setError(getMessage(err))
      notifyError('Cancel graph run failed', err)
    } finally {
      setGraphLifecycleBusy(false)
    }
  }

  async function handleDeleteGraph() {
    if (!selectedGraph) {
      setError('Select a graph before deleting it.')
      return
    }
    if (graphBuildRunning) {
      setError('Cancel the active graph run before deleting this graph.')
      return
    }
    setGraphLifecycleBusy(true)
    try {
      const payload = await api.deleteGraph(token, selectedGraph, { delete_artifacts: deleteGraphArtifacts })
      const deletedGraphId = selectedGraph
      const nextGraphId = await refreshGraphs()
      if (nextGraphId) {
        await refreshSelectedGraph(nextGraphId)
      } else {
        setGraphDetail(null)
        setGraphRuns([])
        setGraphProgress(null)
      }
      setGraphValidation(payload)
      setDeleteGraphArtifacts(false)
      setError('')
      notifyOk('Graph deleted', deletedGraphId)
    } catch (err) {
      setError(getMessage(err))
      notifyError('Delete graph failed', err)
    } finally {
      setGraphLifecycleBusy(false)
    }
  }

  async function handleGraphSmokeTest() {
    if (!selectedGraph) {
      setError('Create or select a graph first.')
      return
    }
    try {
      const payload = await api.graphSmokeTest(token, selectedGraph, {
        query: graphSmokeQuery,
        methods: [],
        limit: 6,
      })
      setGraphSmokeResult(payload)
      setError('')
      notifyOk('Smoke test complete', asString(asRecord(payload.friendly)?.message, selectedGraph))
    } catch (err) {
      setError(getMessage(err))
      notifyError('Smoke test failed', err)
    }
  }

  async function handleGraphSavePrompts() {
    if (!selectedGraph) {
      setError('Create or select a graph first.')
      return
    }
    try {
      await api.updateGraphPrompts(token, selectedGraph, parseJsonObject(graphPromptDraft))
      const detail = await api.getGraph(token, selectedGraph)
      setGraphDetail(detail)
      hydrateGraphForm(detail)
      setError('')
    } catch (err) {
      setError(getMessage(err))
    }
  }

  function toggleGraphTuneTarget(promptFile: string) {
    setGraphTuneTargets(current => (
      current.includes(promptFile)
        ? current.filter(item => item !== promptFile)
        : [...current, promptFile]
    ))
  }

  function toggleGraphTuneSelectedPrompt(promptFile: string) {
    setGraphTuneSelectedPrompts(current => (
      current.includes(promptFile)
        ? current.filter(item => item !== promptFile)
        : [...current, promptFile]
    ))
  }

  async function handleGraphResearchTune() {
    if (!selectedGraph) {
      setError('Create or select a graph first.')
      return
    }
    if (graphTuneTargets.length === 0) {
      setError('Choose at least one prompt target for Research & Tune.')
      return
    }
    setGraphTuneRunning(true)
    try {
      await runGraphTuneForGraph(selectedGraph)
      setError('')
    } catch (err) {
      setError(getMessage(err))
    } finally {
      setGraphTuneRunning(false)
    }
  }

  async function handleGraphResearchTuneApply() {
    if (!selectedGraph || !graphTuneResult?.run_id) {
      setError('Run Research & Tune before applying prompt drafts.')
      return
    }
    if (graphTuneSelectedPrompts.length === 0) {
      setError('Choose at least one generated prompt draft to apply.')
      return
    }
    try {
      const payload = await api.applyGraphResearchTune(token, selectedGraph, graphTuneResult.run_id, graphTuneSelectedPrompts)
      if (payload.graph) {
        setGraphDetail(payload)
        hydrateGraphForm(payload)
      } else {
        const detail = await api.getGraph(token, selectedGraph)
        setGraphDetail(detail)
        hydrateGraphForm(detail)
      }
      const refreshed = await api.getGraphResearchTune(token, selectedGraph, graphTuneResult.run_id)
      setGraphTuneResult(refreshed)
      const runsPayload = await api.getGraphRuns(token, selectedGraph)
      setGraphRuns(runsPayload.runs)
      setError('')
    } catch (err) {
      setError(getMessage(err))
    }
  }

  async function handleGraphSaveSkills() {
    if (!selectedGraph) {
      setError('Create or select a graph first.')
      return
    }
    try {
      await api.updateGraphSkills(token, selectedGraph, {
        skill_ids: uniqueList(graphSkillIdsDraft.split(',').map(item => item.trim())),
        overlay_markdown: graphSkillOverlayDraft,
        overlay_skill_name: graphDisplayNameDraft ? `${graphDisplayNameDraft} Overlay` : '',
      })
      const detail = await api.getGraph(token, selectedGraph)
      setGraphDetail(detail)
      hydrateGraphForm(detail)
      setError('')
    } catch (err) {
      setError(getMessage(err))
    }
  }

  async function handleSkillSave() {
    try {
      const response = selectedSkill
        ? await api.updateSkill(token, selectedSkill, { body_markdown: skillEditor })
        : await api.createSkill(token, { body_markdown: skillEditor, agent_scope: 'general' })
      const savedSkill = response.data as Record<string, unknown> | undefined
      const savedSkillId = asString(savedSkill?.skill_id)
      if (savedSkillId) {
        const detail = await api.getSkill(token, savedSkillId)
        setSkillDetail(detail)
        setSkillEditor(asString(detail.body_markdown))
        setSelectedSkill(savedSkillId)
      }
      setCreatingSkill(false)
      setSkillActionDetail(null)
      const list = await api.listSkills(token)
      setSkills(list.data)
      setError('')
      notifyOk(selectedSkill ? 'Skill updated' : 'Skill created')
    } catch (err) {
      setSkillActionDetail(extractSkillDependencyError(err))
      setError(getMessage(err))
      notifyError('Skill save failed', err)
    }
  }

  async function handleSkillStatus(nextStatus: 'active' | 'archived') {
    if (!selectedSkill) return
    try {
      if (nextStatus === 'active') {
        await api.activateSkill(token, selectedSkill)
      } else {
        await api.deactivateSkill(token, selectedSkill)
      }
      const detail = await api.getSkill(token, selectedSkill)
      setSkillDetail(detail)
      const list = await api.listSkills(token)
      setSkills(list.data)
      setSkillActionDetail(null)
      setError('')
      notifyOk(nextStatus === 'active' ? 'Skill activated' : 'Skill deactivated', selectedSkill)
    } catch (err) {
      setSkillActionDetail(extractSkillDependencyError(err))
      setError(getMessage(err))
      notifyError(nextStatus === 'active' ? 'Activate failed' : 'Deactivate failed', err)
    }
  }

  async function handleSkillPreview() {
    try {
      const result = await api.previewSkill(token, skillPreviewQuery, 'general')
      setSkillPreviewResult(result)
      setError('')
    } catch (err) {
      setError(getMessage(err))
    }
  }

  async function handleCreatePrincipal() {
    const value = principalDraftValue.trim()
    if (!value) {
      setError('Provide an email address or group name first.')
      return
    }
    try {
      await api.createAccessPrincipal(token, {
        principal_type: principalDraftType,
        provider: principalDraftType === 'user' ? principalDraftProvider : 'system',
        email_normalized: principalDraftType === 'user' ? value : '',
        display_name: principalDraftType === 'group' ? value : value,
      })
      setPrincipalDraftValue('')
      await refreshAccessData()
      setError('')
      notifyOk('Principal created', value)
    } catch (err) {
      setError(getMessage(err))
      notifyError('Create principal failed', err)
    }
  }

  async function handleCreateRole() {
    const name = roleDraftName.trim()
    if (!name) {
      setError('Provide a role name first.')
      return
    }
    try {
      await api.createAccessRole(token, {
        name,
        description: roleDraftDescription,
      })
      setRoleDraftName('')
      setRoleDraftDescription('')
      await refreshAccessData()
      setError('')
      notifyOk('Role created', roleDraftName)
    } catch (err) {
      setError(getMessage(err))
      notifyError('Create role failed', err)
    }
  }

  async function handleCreateBinding() {
    if (!bindingRoleId || !bindingPrincipalId) {
      setError('Choose both a role and a principal first.')
      return
    }
    try {
      await api.createAccessBinding(token, {
        role_id: bindingRoleId,
        principal_id: bindingPrincipalId,
      })
      await refreshAccessData()
      setError('')
      notifyOk('Binding created')
    } catch (err) {
      setError(getMessage(err))
      notifyError('Create binding failed', err)
    }
  }

  async function handleCreatePermission() {
    if (!permissionRoleId) {
      setError('Choose a role first.')
      return
    }
    const selector = permissionSelector.trim() || '*'
    try {
      await api.createAccessPermission(token, {
        role_id: permissionRoleId,
        resource_type: permissionResourceType,
        action: permissionAction,
        resource_selector: selector,
      })
      setPermissionSelector('')
      await refreshAccessData()
      setError('')
      notifyOk('Permission added')
    } catch (err) {
      setError(getMessage(err))
      notifyError('Add permission failed', err)
    }
  }

  async function handleCreateMembership() {
    if (!membershipParentId || !membershipChildId) {
      setError('Choose a group and a member first.')
      return
    }
    try {
      await api.createAccessMembership(token, {
        parent_principal_id: membershipParentId,
        child_principal_id: membershipChildId,
      })
      await refreshAccessData()
      setError('')
      notifyOk('Membership added')
    } catch (err) {
      setError(getMessage(err))
      notifyError('Add membership failed', err)
    }
  }

  async function handleLoadEffectiveAccess() {
    const email = accessPreviewEmail.trim()
    if (!email) {
      setError('Choose an email to preview.')
      return
    }
    try {
      const payload = await api.getEffectiveAccess(token, email)
      setEffectiveAccess(payload)
      setError('')
    } catch (err) {
      setError(getMessage(err))
    }
  }

  function setAccessWizardDefaults(mode: Exclude<AccessWizardMode, null>) {
    setAccessWizard(mode)
    setAccessWizardPreview(null)
    if (mode === 'setup') {
      const firstGroup = accessGroups[0]?.principal_id ?? ''
      const preset = ACCESS_PRESETS.find(item => item.id === setupPresetId) ?? ACCESS_PRESETS[0]
      setAccessWizardStep('group')
      setSetupGroupMode(firstGroup ? 'existing' : 'create')
      setSetupGroupId(current => current || firstGroup)
      setSetupResourceType(preset.resourceType)
      setSetupActions(preset.actions)
      setSetupResourceSelectors(current => current.length > 0 ? current : ['*'])
      return
    }
    if (mode === 'grant') {
      setAccessWizardStep('resource')
      setGrantPrincipalIds(current => current.length > 0 ? current : accessGroups[0] ? [accessGroups[0].principal_id] : [])
      setGrantResourceSelectors(current => current.length > 0 ? current : ['*'])
      return
    }
    if (mode === 'user') {
      setAccessWizardStep('profile')
      return
    }
    setAccessWizardStep('details')
  }

  function accessWizardSteps(): string[] {
    if (accessWizard === 'setup') return ['group', 'preset', 'members', 'resources', 'preview', 'review']
    if (accessWizard === 'grant') return ['resource', 'principals', 'actions', 'review']
    if (accessWizard === 'user') return ['profile', 'groups', 'preview', 'review']
    if (accessWizard === 'group') return ['details', 'role', 'members', 'review']
    return []
  }

  function toggleSetupMember(memberId: string) {
    setSetupMemberIds(current => current.includes(memberId)
      ? current.filter(id => id !== memberId)
      : [...current, memberId])
  }

  function toggleSetupSelector(selector: string) {
    setSetupResourceSelectors(current => current.includes(selector)
      ? current.filter(id => id !== selector)
      : [...current, selector])
  }

  function toggleSetupAction(action: AccessAction) {
    setSetupActions(current => current.includes(action)
      ? current.filter(item => item !== action)
      : [...current, action])
  }

  function toggleGrantSelector(selector: string) {
    setGrantResourceSelectors(current => current.includes(selector)
      ? current.filter(id => id !== selector)
      : [...current, selector])
  }

  function toggleGrantPrincipal(principalId: string) {
    setGrantPrincipalIds(current => current.includes(principalId)
      ? current.filter(id => id !== principalId)
      : [...current, principalId])
  }

  function toggleGrantAction(action: AccessAction) {
    setGrantActions(current => current.includes(action)
      ? current.filter(item => item !== action)
      : [...current, action])
  }

  function toggleManageUserGroup(groupId: string) {
    setManageUserGroupIds(current => current.includes(groupId)
      ? current.filter(id => id !== groupId)
      : [...current, groupId])
  }

  function toggleCreateGroupMember(memberId: string) {
    setCreateGroupMemberIds(current => current.includes(memberId)
      ? current.filter(id => id !== memberId)
      : [...current, memberId])
  }

  function normalizedAccessSelectors(selectors: string[]): string[] {
    const cleaned = uniqueList(selectors.filter(Boolean))
    return cleaned.length > 0 ? cleaned : ['*']
  }

  async function createAccessRoleWithPermissions(options: {
    name: string
    description: string
    resourceType: AccessResourceType
    selectors: string[]
    actions: AccessAction[]
  }): Promise<string> {
    const rolePayload = await api.createAccessRole(token, {
      name: options.name,
      description: options.description,
    })
    const roleId = rolePayload.role.role_id
    const selectors = normalizedAccessSelectors(options.selectors)
    const actions = options.actions.length > 0 ? options.actions : ['use']
    for (const selector of selectors) {
      for (const action of actions) {
        await api.createAccessPermission(token, {
          role_id: roleId,
          resource_type: options.resourceType,
          action,
          resource_selector: selector,
        })
      }
    }
    return roleId
  }

  async function handleAccessWizardPreview() {
    const selectedUserId = accessWizard === 'user' ? '' : setupMemberIds[0]
    const selectedUser = accessPrincipalById.get(selectedUserId)
    const email = accessWizard === 'user' ? manageUserEmail.trim() : asString(selectedUser?.email_normalized)
    if (!email) {
      setError('Choose or enter a user email before previewing access.')
      return
    }
    try {
      const payload = await api.getEffectiveAccess(token, email)
      setAccessWizardPreview(payload)
      setError('')
    } catch (err) {
      setError(getMessage(err))
      notifyError('Preview access failed', err)
    }
  }

  async function handleFinishAccessWizard() {
    try {
      if (accessWizard === 'setup') {
        const preset = ACCESS_PRESETS.find(item => item.id === setupPresetId) ?? ACCESS_PRESETS[0]
        let groupId = setupGroupId
        let groupName = principalLabel(accessPrincipalById.get(groupId))
        if (setupGroupMode === 'create') {
          const name = setupGroupName.trim()
          if (!name) throw new Error('Provide a group name before applying setup.')
          const payload = await api.createAccessPrincipal(token, {
            principal_type: 'group',
            provider: 'system',
            display_name: name,
            metadata_json: { purpose: setupGroupPurpose },
            active: true,
          })
          groupId = payload.principal.principal_id
          groupName = principalLabel(payload.principal)
        }
        if (!groupId) throw new Error('Choose or create a group before applying setup.')
        const roleId = await createAccessRoleWithPermissions({
          name: `${groupName} ${preset.label}`,
          description: `${preset.description} Created from the Access Setup Wizard.`,
          resourceType: setupResourceType,
          selectors: setupResourceSelectors,
          actions: setupActions,
        })
        await api.createAccessBinding(token, { role_id: roleId, principal_id: groupId })
        for (const memberId of setupMemberIds) {
          await api.createAccessMembership(token, {
            parent_principal_id: groupId,
            child_principal_id: memberId,
          })
        }
        await refreshAccessData()
        setAccessTab('groups')
        notifyOk('Access setup applied', groupName)
      } else if (accessWizard === 'grant') {
        if (grantPrincipalIds.length === 0) throw new Error('Choose at least one user or group.')
        const selectorLabel = grantResourceSelectors.length === 1
          ? (accessResourceLabelByTypeAndId.get(`${grantResourceType}:${grantResourceSelectors[0]}`) || grantResourceSelectors[0])
          : `${grantResourceSelectors.length || 1} resources`
        const roleId = await createAccessRoleWithPermissions({
          name: `${accessResourceLabel(grantResourceType)} ${shortId(selectorLabel)} Grant`,
          description: `Resource-centric grant created from the Grant Resource Access Wizard.`,
          resourceType: grantResourceType,
          selectors: grantResourceSelectors,
          actions: grantActions,
        })
        for (const principalId of grantPrincipalIds) {
          await api.createAccessBinding(token, { role_id: roleId, principal_id: principalId })
        }
        await refreshAccessData()
        setAccessTab('grants')
        notifyOk('Resource access granted')
      } else if (accessWizard === 'user') {
        const email = manageUserEmail.trim()
        if (!email) throw new Error('Provide a user email before saving.')
        const payload = await api.createAccessPrincipal(token, {
          principal_type: 'user',
          provider: 'email',
          email_normalized: email,
          display_name: manageUserDisplayName.trim() || email,
          metadata_json: { system_role: manageUserSystemRole },
          active: manageUserSystemRole !== 'pending',
        })
        for (const groupId of manageUserGroupIds) {
          await api.createAccessMembership(token, {
            parent_principal_id: groupId,
            child_principal_id: payload.principal.principal_id,
          })
        }
        setAccessPreviewEmail(email)
        await refreshAccessData()
        setAccessTab('users')
        notifyOk('User saved', email)
      } else if (accessWizard === 'group') {
        const name = createGroupName.trim()
        if (!name) throw new Error('Provide a group name before saving.')
        const payload = await api.createAccessPrincipal(token, {
          principal_type: 'group',
          provider: 'system',
          display_name: name,
          metadata_json: { purpose: createGroupPurpose },
          active: true,
        })
        if (createGroupRoleId) {
          await api.createAccessBinding(token, {
            role_id: createGroupRoleId,
            principal_id: payload.principal.principal_id,
          })
        }
        for (const memberId of createGroupMemberIds) {
          await api.createAccessMembership(token, {
            parent_principal_id: payload.principal.principal_id,
            child_principal_id: memberId,
          })
        }
        await refreshAccessData()
        setAccessTab('groups')
        notifyOk('Group created', name)
      }
      setAccessWizard(null)
      setAccessWizardStep('')
      setAccessWizardPreview(null)
      setError('')
    } catch (err) {
      setError(getMessage(err))
      notifyError('Access wizard failed', err)
    }
  }

  async function handleCreateMcpConnection() {
    if (!mcpDraftName.trim() || !mcpDraftUrl.trim()) {
      setError('Provide an MCP display name and Streamable HTTP URL first.')
      return
    }
    try {
      const payload = await api.createMcpConnection(token, {
        display_name: mcpDraftName.trim(),
        server_url: mcpDraftUrl.trim(),
        auth_type: mcpDraftSecret.trim() ? 'bearer' : 'none',
        secret: mcpDraftSecret,
        allowed_agents: uniqueList(mcpDraftAgents.split(',').map(item => item.trim())).filter(Boolean),
        visibility: mcpDraftVisibility,
      })
      setMcpDraftName('')
      setMcpDraftUrl('')
      setMcpDraftSecret('')
      const nextId = payload.connection.connection_id
      await refreshMcpData(nextId)
      setError('')
      notifyOk('MCP connection created', payload.connection.display_name)
    } catch (err) {
      setError(getMessage(err))
      notifyError('MCP create failed', err)
    }
  }

  async function handleTestMcpConnection(connectionId: string) {
    try {
      await api.testMcpConnection(token, connectionId)
      await refreshMcpData(connectionId)
      setError('')
      notifyOk('MCP connection tested')
    } catch (err) {
      setError(getMessage(err))
      notifyError('MCP test failed', err)
    }
  }

  async function handleRefreshMcpTools(connectionId: string) {
    try {
      await api.refreshMcpTools(token, connectionId)
      await refreshMcpData(connectionId)
      setError('')
      notifyOk('MCP tools refreshed')
    } catch (err) {
      setError(getMessage(err))
      notifyError('MCP refresh failed', err)
    }
  }

  async function handleDisableMcpConnection(connectionId: string) {
    try {
      await api.disableMcpConnection(token, connectionId)
      await refreshMcpData(connectionId)
      setError('')
      notifyOk('MCP connection disabled')
    } catch (err) {
      setError(getMessage(err))
      notifyError('MCP disable failed', err)
    }
  }

  async function handleToggleMcpTool(connectionId: string, toolId: string, enabled: boolean) {
    try {
      await api.updateMcpTool(token, connectionId, toolId, { enabled })
      await refreshMcpData(connectionId)
      setError('')
    } catch (err) {
      setError(getMessage(err))
      notifyError('MCP tool update failed', err)
    }
  }

  async function handleToggleMcpToolReadOnly(connectionId: string, toolId: string, readOnly: boolean) {
    try {
      await api.updateMcpTool(token, connectionId, toolId, {
        read_only: readOnly,
        destructive: !readOnly,
      })
      await refreshMcpData(connectionId)
      setError('')
    } catch (err) {
      setError(getMessage(err))
      notifyError('MCP tool safety update failed', err)
    }
  }

  const sectionCounts = useMemo(() => {
    const counts = new Map<Section, number>()
    counts.set('agents', agentsPayload?.agents.length ?? 0)
    counts.set('prompts', prompts.length)
    counts.set('collections', collections.length)
    counts.set('uploads', uploadedFiles.length)
    counts.set('graphs', graphs.length)
    counts.set('skills', skills.length)
    counts.set('access', accessPrincipals.length)
    counts.set('mcp', mcpConnections.length)
    counts.set('operations', jobs.length)
    return counts
  }, [
    accessPrincipals.length,
    agentsPayload?.agents.length,
    collections.length,
    graphs.length,
    jobs.length,
    mcpConnections.length,
    prompts.length,
    skills.length,
    uploadedFiles.length,
  ])

  const primaryNav = (
    <ControlPanelTopNav
      active={active}
      onSelect={setActive}
      sectionCounts={sectionCounts}
      unsupportedSectionIds={unsupportedSectionIds}
      theme={theme}
      density={density}
      onOpenPalette={() => setPaletteOpen(true)}
      onToggleTheme={toggleTheme}
      onToggleDensity={toggleDensity}
      onLock={() => setToken('')}
    />
  )

  const sectionToolbar = active === 'config' && groupedConfigFields.length > 0 ? (
    <div className="settings-toolbar">
      <label className="settings-search" aria-label="Search settings">
        <Search size={14} strokeWidth={2} aria-hidden="true" />
        <input
          value={settingsSearch}
          onChange={event => setSettingsSearch(event.target.value)}
          placeholder="Search settings"
        />
        {settingsSearch && (
          <button type="button" onClick={() => setSettingsSearch('')} aria-label="Clear settings search">
            ×
          </button>
        )}
      </label>
      <SectionTabs
        tabs={filteredConfigGroups.map(([group]) => ({ id: group, label: group }))}
        active={visibleConfigGroup?.[0] ?? filteredConfigGroups[0]?.[0] ?? ''}
        onChange={group => setActiveConfigGroup(group)}
        ariaLabel="Config groups"
      />
    </div>
  ) : active === 'architecture' ? (
    <SectionTabs
      tabs={[
        { id: 'map', label: 'Map' },
        { id: 'agent-graph', label: 'Agent Graph' },
        { id: 'paths', label: 'Routing Paths' },
        { id: 'traffic', label: 'Live Traffic' },
      ]}
      active={architectureTab}
      onChange={tab => setArchitectureTab(tab as ArchitectureTab)}
      ariaLabel="Architecture views"
    />
  ) : active === 'agents' ? (
    <SectionTabs
      tabs={[
        { id: 'workspace', label: 'Workspace' },
        { id: 'catalog', label: 'Tool Catalog' },
      ]}
      active={agentsTab}
      onChange={tab => setAgentsTab(tab as AgentsTab)}
      ariaLabel="Agent views"
    />
  ) : active === 'prompts' ? (
    <SectionTabs
      tabs={[
        { id: 'edit', label: 'Edit' },
        { id: 'compare', label: 'Compare' },
      ]}
      active={promptsTab}
      onChange={tab => setPromptsTab(tab as PromptsTab)}
      ariaLabel="Prompt views"
    />
  ) : active === 'collections' ? (
    <SectionTabs
      tabs={[
        { id: 'workspace', label: 'Workspace' },
      ]}
      active={collectionsTab}
      onChange={tab => setCollectionsTab(tab as CollectionsTab)}
      ariaLabel="Collection views"
    />
  ) : active === 'graphs' ? (
    <SectionTabs
      tabs={[
        { id: 'workspace', label: 'Workspace' },
        { id: 'runs', label: 'Runs' },
      ]}
      active={graphsTab}
      onChange={tab => setGraphsTab(tab as GraphsTab)}
      ariaLabel="Graph views"
    />
  ) : active === 'skills' ? (
    <SectionTabs
      tabs={[
        { id: 'editor', label: 'Editor' },
        { id: 'preview', label: 'Preview' },
      ]}
      active={skillsTab}
      onChange={tab => setSkillsTab(tab as SkillsTab)}
      ariaLabel="Skill views"
    />
  ) : active === 'operations' ? (
    <SectionTabs
      tabs={[
        { id: 'reloads', label: 'Reloads' },
        { id: 'jobs', label: 'Jobs' },
        { id: 'audit', label: 'Audit' },
      ]}
      active={operationsTab}
      onChange={tab => setOperationsTab(tab as OperationsTab)}
      ariaLabel="Operations views"
    />
  ) : undefined

  if (!token) {
    return (
      <div className="login-shell">
        <div className="login-noise" aria-hidden="true" />
        <section className="login-card">
          <span className="section-eyebrow">Executive Console</span>
          <h1>Agent Control Panel</h1>
          <p className="login-copy">
            Unlock the control plane for local admin workflows. The token is stored only in this browser session and should
            match <code>CONTROL_PANEL_ADMIN_TOKEN</code> in the repo root <code>.env</code>.
          </p>
          <div className="login-highlights">
            <StatusBadge tone="accent">Runtime edits</StatusBadge>
            <StatusBadge tone="warning">Live reloads</StatusBadge>
            <StatusBadge tone="neutral">Overlay-safe</StatusBadge>
          </div>
          <label className="field">
            <span>Admin Token</span>
            <input
              value={draftToken}
              onChange={event => setDraftToken(event.target.value)}
              placeholder="Admin token"
              type="password"
            />
          </label>
          <ActionBar>
            <ActionButton tone="primary" onClick={() => setToken(draftToken)}>Unlock</ActionButton>
          </ActionBar>
        </section>
      </div>
    )
  }

  return (
    <AppShell
      className={active === 'agents' || active === 'prompts' || active === 'collections' || active === 'uploads' || active === 'graphs' || active === 'skills' || active === 'access' || active === 'mcp' ? 'app-shell-studio' : undefined}
      topNav={primaryNav}
      toolbar={sectionToolbar}
      header={(
        <SectionHeader
          eyebrow={activeMeta.eyebrow}
          title={activeMeta.label}
          description={activeMeta.description}
          actions={(
            <>
              {active === 'architecture' && activeSectionSupported && (
                <ActionButton
                  tone="secondary"
                  onClick={() => void refreshArchitectureData()}
                  disabled={architectureRefreshing}
                >
                  {architectureRefreshing ? 'Retrying...' : 'Retry'}
                </ActionButton>
              )}
              <Tooltip content={`Environment: ${environmentLabel}`}>
                <span className={`env-pill env-pill-${environmentTone}`} aria-label={`Environment ${environmentLabel}`}>
                  <span className="env-pill-dot" aria-hidden="true" />
                  {environmentLabel}
                </span>
              </Tooltip>
              <Tooltip content={healthPulseTooltip}>
                <span className={`health-pulse health-pulse-${healthPulseTone}`} aria-label={`Backend status ${healthPulseLabel}`} role="status">
                  <span className="health-pulse-core" aria-hidden="true" />
                </span>
              </Tooltip>
              <Tooltip content="Open command palette">
                <button
                  type="button"
                  className="cmdk-hint"
                  onClick={() => setPaletteOpen(true)}
                  aria-label="Open command palette"
                >
                  <Kbd>{isMacPlatform ? '⌘' : 'Ctrl'}</Kbd>
                  <Kbd>K</Kbd>
                </button>
              </Tooltip>
              {overview?.status && <StatusBadge tone={toneForStatus(overview.status)}>{asString(overview.status)}</StatusBadge>}
              {lastReload?.reason && <StatusBadge tone="neutral">{asString(lastReload.reason)}</StatusBadge>}
              {lastReload?.timestamp && <StatusBadge tone="accent">{formatTimestamp(lastReload.timestamp)}</StatusBadge>}
            </>
          )}
        />
      )}
    >
      {compatibilityBanner && <div className="compatibility-banner">{compatibilityBanner}</div>}
      {error && <div className="error-banner">{error}</div>}

      {!activeSectionSupported && unsupportedSectionMessage ? (
        <SurfaceCard
          title={unsupportedSectionMessage.title}
          subtitle="The control panel is keeping the healthy sections usable while calling out the missing backend contract for this section."
        >
          <EmptyState title={unsupportedSectionMessage.title} body={unsupportedSectionMessage.body} />
          {activeSectionSupport && (
            <div className="field-stack">
              {activeSectionSupport.missing_routes.length > 0 && (
                <div className="inline-alert inline-alert-warning">
                  <span>Missing routes</span>
                  <strong>{activeSectionSupport.missing_routes.join(', ')}</strong>
                </div>
              )}
              <div className="summary-row">
                <span>Compatibility source</span>
                <strong>{compatibilitySource === 'openapi' ? 'openapi fallback' : 'capabilities contract'}</strong>
              </div>
              {activeSectionSupport.reason && (
                <div className="summary-row">
                  <span>Reason</span>
                  <strong>{activeSectionSupport.reason}</strong>
                </div>
              )}
            </div>
          )}
        </SurfaceCard>
      ) : active === 'dashboard' && !overview ? (
        <div className="content-stack">
          <SurfaceCard title="Runtime" subtitle="High-signal overview of the live environment, providers, models, and resource counts.">
            <div className="stat-grid">
              {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="stat-card">
                  <Skeleton width="40%" height={12} />
                  <Skeleton width="60%" height={24} />
                  <Skeleton width="80%" height={10} />
                </div>
              ))}
            </div>
          </SurfaceCard>
        </div>
      ) : active === 'dashboard' && overview && (
        <div className="content-stack">
          <SurfaceCard title="Runtime" subtitle="High-signal overview of the live environment, providers, models, and resource counts.">
            <div className="stat-grid">
              <StatCard label="Collections" value={overview.counts.collections ?? 0} caption="Namespaces with indexed documents" />
              <StatCard label="Agents" value={overview.counts.agents ?? 0} caption="Registered agent definitions" />
              <StatCard label="Skills" value={overview.counts.skills ?? 0} caption="Available reusable skill packs" />
              <StatCard label="Tools" value={overview.counts.tools ?? 0} caption="Tool contracts available to agents" />
              <StatCard label="Jobs" value={overview.counts.jobs ?? 0} caption="Currently tracked background jobs" />
              <StatCard label="Gateway" value={overview.gateway_model_id || 'local'} caption="Active model gateway id" tone="accent" />
            </div>
            <div className="chip-grid">
              {Object.entries(overview.providers).map(([key, value]) => (
                <div key={key} className="meta-chip">
                  <span>{humanizeKey(key)}</span>
                  <strong>{value}</strong>
                </div>
              ))}
              {Object.entries(overview.models).map(([key, value]) => (
                <div key={key} className="meta-chip meta-chip-strong">
                  <span>{humanizeKey(key)}</span>
                  <strong>{value}</strong>
                </div>
              ))}
            </div>
            <JsonInspector label="Technical details" value={overview} />
          </SurfaceCard>

          {(() => {
            const failedJobs = jobs.filter(j => {
              const state = asString(j.scheduler_state || j.status).toLowerCase()
              return state === 'failed' || state === 'error'
            })
            const reloadFailed = asString(lastReload?.status, '').toLowerCase() === 'failed'
            const attentionItems: Array<{ key: string; title: string; detail: string; tone: 'danger' | 'warn'; onClick?: () => void }> = []
            if (reloadFailed) {
              attentionItems.push({
                key: 'reload',
                title: 'Last reload failed',
                detail: asString(lastReload?.error, asString(lastReload?.reason, 'See operations for details')),
                tone: 'danger',
                onClick: () => setActive('operations'),
              })
            }
            failedJobs.slice(0, 4).forEach((job, i) => {
              attentionItems.push({
                key: `job-${i}`,
                title: `Job failed: ${asString(job.agent_name, asString(job.job_id, 'job'))}`,
                detail: asString(job.error, 'Check job details in Operations'),
                tone: 'danger',
                onClick: () => setActive('operations'),
              })
            })
            if (attentionItems.length === 0) return null
            return (
              <SurfaceCard title="Needs attention" subtitle="Failures and degraded states surfaced here while they persist.">
                <ul className="attention-list">
                  {attentionItems.map(item => (
                    <li key={item.key} className={`attention-item attention-item-${item.tone}`}>
                      <div>
                        <strong>{item.title}</strong>
                        <p className="muted-copy">{item.detail}</p>
                      </div>
                      {item.onClick && (
                        <ActionButton tone="ghost" onClick={item.onClick}>Investigate</ActionButton>
                      )}
                    </li>
                  ))}
                </ul>
              </SurfaceCard>
            )
          })()}

          <div className="dashboard-grid">
            <CollapsibleSurfaceCard
              title="Reload Summary"
              subtitle="What changed most recently and whether it completed cleanly."
              open={dashboardReloadOpen}
              onToggle={() => setDashboardReloadOpen(open => !open)}
            >
              {overview.last_reload ? (
                <div className="timeline-card">
                  <div className="summary-row">
                    <span>Status</span>
                    <StatusBadge tone={toneForStatus(overview.last_reload.status)}>{asString(overview.last_reload.status, 'unknown')}</StatusBadge>
                  </div>
                  <div className="summary-row">
                    <span>Reason</span>
                    <strong>{asString(overview.last_reload.reason, 'startup')}</strong>
                  </div>
                  <div className="summary-row">
                    <span>Changed Keys</span>
                    <strong>{shortList(asArray<string>(overview.last_reload.changed_keys))}</strong>
                  </div>
                  <div className="summary-row">
                    <span>Timestamp</span>
                    <strong>{formatTimestamp(overview.last_reload.timestamp)}</strong>
                  </div>
                  {asString(overview.last_reload.error) && (
                    <div className="inline-alert">
                      <span>Error</span>
                      <strong>{asString(overview.last_reload.error)}</strong>
                    </div>
                  )}
                </div>
              ) : (
                <EmptyState title="No reload data yet" body="Reload information will appear here after startup and each config or agent reload." />
              )}
            </CollapsibleSurfaceCard>

            <CollapsibleSurfaceCard
              title="Activity Feed"
              subtitle="Recent audit events and operator-visible changes across the control plane."
              open={dashboardActivityOpen}
              onToggle={() => setDashboardActivityOpen(open => !open)}
            >
              {auditEvents.length > 0 ? (
                <div className="timeline-list">
                  {auditEvents.slice(0, 6).map((event, index) => (
                    <article key={`${asString(event.action)}-${index}`} className="timeline-item">
                      <div className="timeline-dot" aria-hidden="true" />
                      <div>
                        <strong>{asString(event.action, 'event')}</strong>
                        <p>{asString(event.actor, 'system')} • {formatTimestamp(event.timestamp)}</p>
                        <span>{shortList(asArray<string>(event.changed_keys))}</span>
                      </div>
                    </article>
                  ))}
                </div>
              ) : (
                <EmptyState title="No audit events yet" body="Once admins validate, apply, reload, or ingest content, the recent timeline will appear here." />
              )}
            </CollapsibleSurfaceCard>
          </div>
        </div>
      )}

      {active === 'architecture' && architecture && (
        <div className="content-stack architecture-shell">
          <SurfaceCard
            title="System Overview"
            subtitle="Live topology generated from the runtime registry, router config, and current overlays after the latest successful reload."
          >
            <div className="stat-grid">
              <StatCard label="Agents" value={asString((architecture.system.counts as Record<string, unknown> | undefined)?.agents, '0')} caption="Live registry nodes" />
              <StatCard label="Services" value={asString((architecture.system.counts as Record<string, unknown> | undefined)?.services, '0')} caption="Runtime capabilities shown on the map" />
              <StatCard label="Edges" value={asString((architecture.system.counts as Record<string, unknown> | undefined)?.edges, '0')} caption="Request, routing, delegation, and service links" />
              <StatCard label="Router" value={summarizeRouterMode(architecture.router)} caption="Current routing posture" tone="accent" />
            </div>
            <div className="chip-grid">
              <div className="meta-chip meta-chip-strong">
                <span>Default Agent</span>
                <strong>{asString(architecture.router.default_agent, 'general')}</strong>
              </div>
              <div className="meta-chip">
                <span>Basic Path</span>
                <strong>{asString(architecture.router.basic_agent, 'basic')}</strong>
              </div>
              <div className="meta-chip">
                <span>Coordinator</span>
                <strong>{asString(architecture.router.coordinator_agent, 'coordinator')}</strong>
              </div>
              <div className="meta-chip">
                <span>Data Analyst</span>
                <strong>{asString(architecture.router.data_analyst_agent, 'data_analyst')}</strong>
              </div>
              <div className="meta-chip">
                <span>Grounded Worker</span>
                <strong>{asString(architecture.router.rag_agent, 'rag_worker')}</strong>
              </div>
            </div>
          </SurfaceCard>

          {architectureTab === 'map' && (
            <div className={architectureInspectorOpen ? 'architecture-layout' : 'architecture-layout architecture-layout-collapsed'}>
              <SurfaceCard
                className="architecture-map-card"
                title="Live System Map"
                subtitle="Click a node to inspect it. Click a routing path card to trace that path directly on the diagram."
              >
                <div className="architecture-legend">
                  <StatusBadge tone="accent">Router</StatusBadge>
                  <span>Routing control point</span>
                  <StatusBadge tone="neutral">Agent</StatusBadge>
                  <span>Runtime role</span>
                  <StatusBadge tone="ok">Service</StatusBadge>
                  <span>Shared capability</span>
                </div>
                <div className="architecture-scroll">
                  <div
                    className="architecture-map-canvas"
                    style={{ width: `${architectureMapLayout.width}px`, height: `${architectureMapLayout.height}px` }}
                  >
                    <svg
                      className="architecture-map-svg"
                      viewBox={`0 0 ${architectureMapLayout.width} ${architectureMapLayout.height}`}
                      aria-hidden="true"
                    >
                      {architectureMapLayout.lanes.map(lane => {
                        return (
                          <g key={lane.id}>
                            <rect
                              x={lane.x}
                              y={lane.y}
                              width={lane.width}
                              height={lane.height}
                              rx={10}
                              className="architecture-lane"
                            />
                            <text x={lane.x + 18} y={40} className="architecture-lane-label">
                              {lane.label}
                            </text>
                          </g>
                        )
                      })}
                      {(['dimmed', 'normal', 'highlighted'] as const).map(layer => (
                        <g key={layer} data-edge-group={layer}>
                          {architectureMapLayout.edges
                            .filter(edge => edge.layer === layer)
                            .map(edge => (
                              <path
                                key={edge.id}
                                d={edge.path}
                                data-edge-id={edge.id}
                                data-edge-layer={edge.layer}
                                data-edge-highlighted={edge.highlighted ? 'true' : 'false'}
                                className={[
                                  'architecture-edge',
                                  edge.kind === 'routing_path' ? 'architecture-edge-routing' : '',
                                  edge.highlighted ? 'architecture-edge-highlighted' : '',
                                  edge.dimmed ? 'architecture-edge-dimmed' : '',
                                ].filter(Boolean).join(' ')}
                              />
                            ))}
                        </g>
                      ))}
                    </svg>
                    {architectureMapLayout.nodes.map(node => {
                      const selected = highlightedArchitectureNodeIds.has(node.id)
                      const dimmed = (highlightedArchitectureNodeIds.size > 0 || highlightedArchitectureEdgeIds.size > 0) && !selected
                      return (
                        <button
                          key={node.id}
                          type="button"
                          aria-label={node.label}
                          className={[
                            'architecture-node',
                            `architecture-node-${node.kind}`,
                            selected ? 'architecture-node-selected' : '',
                            dimmed ? 'architecture-node-dimmed' : '',
                          ].filter(Boolean).join(' ')}
                          style={{
                            left: `${node.x}px`,
                            top: `${node.y}px`,
                            width: `${node.width}px`,
                            height: `${node.height}px`,
                          }}
                          onClick={() => {
                            setSelectedArchitectureNodeId(node.id)
                            setSelectedArchitectureEdgeId('')
                            setSelectedArchitecturePathId('')
                          }}
                        >
                          <span className="architecture-node-label">{node.label}</span>
                          <span className="architecture-node-meta">
                            {node.kind === 'agent' ? (node.mode || node.role_kind || 'agent') : node.kind}
                          </span>
                        </button>
                      )
                    })}
                  </div>
                </div>
              </SurfaceCard>

              <CollapsibleSurfaceCard
                className="architecture-inspector"
                title={selectedArchitectureNode ? 'Node Inspector' : selectedArchitecturePath ? 'Path Inspector' : 'Architecture Notes'}
                subtitle="A plain-language explanation of the selected node or path, plus the live capabilities attached to it."
                open={architectureInspectorOpen}
                onToggle={() => setArchitectureInspectorOpen(open => !open)}
              >
                {selectedArchitectureNode ? (
                  <>
                    <div className="badge-cluster">
                      <StatusBadge tone={nodeTone(selectedArchitectureNode)}>{selectedArchitectureNode.kind}</StatusBadge>
                      {asArray<string>(selectedArchitectureNode.badges).map(badge => (
                        <StatusBadge key={badge} tone="neutral">{badge}</StatusBadge>
                      ))}
                    </div>
                    <div className="summary-list">
                      <div className="summary-row">
                        <span>Label</span>
                        <strong>{selectedArchitectureNode.label}</strong>
                      </div>
                      <div className="summary-row">
                        <span>Status</span>
                        <StatusBadge tone={toneForStatus(selectedArchitectureNode.status)}>{selectedArchitectureNode.status}</StatusBadge>
                      </div>
                      <div className="summary-row">
                        <span>Description</span>
                        <strong>{selectedArchitectureNode.description || 'No description provided.'}</strong>
                      </div>
                      {selectedArchitectureNode.mode && (
                        <div className="summary-row">
                          <span>Mode</span>
                          <strong>{selectedArchitectureNode.mode}</strong>
                        </div>
                      )}
                      {selectedArchitectureNode.prompt_file && (
                        <div className="summary-row">
                          <span>Prompt File</span>
                          <strong>{selectedArchitectureNode.prompt_file}</strong>
                        </div>
                      )}
                      {selectedArchitectureNode.entry_path && (
                        <div className="summary-row">
                          <span>Entry Path</span>
                          <strong>{selectedArchitectureNode.entry_path}</strong>
                        </div>
                      )}
                      {selectedArchitectureNode.allowed_tools && (
                        <div className="summary-row">
                          <span>Tool Access</span>
                          <strong>{shortList(selectedArchitectureNode.allowed_tools)}</strong>
                        </div>
                      )}
                      {selectedArchitectureNode.allowed_worker_agents && (
                        <div className="summary-row">
                          <span>Worker Agents</span>
                          <strong>{shortList(selectedArchitectureNode.allowed_worker_agents)}</strong>
                        </div>
                      )}
                      {selectedArchitectureNode.preload_skill_packs && (
                        <div className="summary-row">
                          <span>Pinned Skills</span>
                          <strong>{shortList(selectedArchitectureNode.preload_skill_packs)}</strong>
                        </div>
                      )}
                    </div>
                  </>
                ) : selectedArchitecturePath ? (
                  <>
                    <div className="badge-cluster">
                      <StatusBadge tone={selectedArchitecturePath.route === 'BASIC' ? 'warning' : 'accent'}>
                        {selectedArchitecturePath.route}
                      </StatusBadge>
                      {asArray<string>(selectedArchitecturePath.badges).map(badge => (
                        <StatusBadge key={badge} tone="neutral">{badge}</StatusBadge>
                      ))}
                    </div>
                    <div className="summary-list">
                      <div className="summary-row">
                        <span>Path</span>
                        <strong>{selectedArchitecturePath.label}</strong>
                      </div>
                      <div className="summary-row">
                        <span>Target Agent</span>
                        <strong>{selectedArchitecturePath.target_agent || 'Default runtime selection'}</strong>
                      </div>
                      <div className="summary-row">
                        <span>When It Happens</span>
                        <strong>{selectedArchitecturePath.when || 'See router summary for details.'}</strong>
                      </div>
                    </div>
                    <p className="muted-copy architecture-path-summary">{selectedArchitecturePath.summary}</p>
                  </>
                ) : (
                  <EmptyState title="Select a node or path" body="Choose a node from the map or open Routing Paths to trace a canonical route and inspect it here." />
                )}
                {selectedArchitectureNode && <JsonInspector label="Technical details" value={selectedArchitectureNode} />}
                {!selectedArchitectureNode && selectedArchitecturePath && <JsonInspector label="Technical details" value={selectedArchitecturePath} />}
              </CollapsibleSurfaceCard>
            </div>
          )}

          {architectureTab === 'agent-graph' && (
            <div className="architecture-agent-graph">
              <SurfaceCard
                className="architecture-map-card architecture-agent-graph-card"
                title="Agent Graph Overview"
                subtitle="All runtime nodes and edges are shown together. Select a node or edge to inspect the registry metadata behind it."
              >
                <div className="architecture-agent-graph-stats">
                  <div className="meta-chip">
                    <span>Agents</span>
                    <strong>{formatWholeNumber(architectureGraphStats.agents)}</strong>
                  </div>
                  <div className="meta-chip">
                    <span>Services</span>
                    <strong>{formatWholeNumber(architectureGraphStats.services)}</strong>
                  </div>
                  <div className="meta-chip">
                    <span>Routing Edges</span>
                    <strong>{formatWholeNumber(architectureGraphStats.routingEdges)}</strong>
                  </div>
                  <div className="meta-chip">
                    <span>Delegations</span>
                    <strong>{formatWholeNumber(architectureGraphStats.delegations)}</strong>
                  </div>
                  <div className="meta-chip">
                    <span>Service Links</span>
                    <strong>{formatWholeNumber(architectureGraphStats.serviceLinks)}</strong>
                  </div>
                </div>
                <div className="architecture-legend">
                  <StatusBadge tone="accent">Route</StatusBadge>
                  <span>Router to agent</span>
                  <StatusBadge tone="warning">Delegate</StatusBadge>
                  <span>Agent handoff</span>
                  <StatusBadge tone="ok">Service</StatusBadge>
                  <span>Shared runtime dependency</span>
                </div>
                <div className="architecture-scroll">
                  <div
                    className="architecture-map-canvas architecture-agent-graph-canvas"
                    style={{ width: `${architectureFullGraphLayout.width}px`, height: `${architectureFullGraphLayout.height}px` }}
                  >
                    <svg
                      className="architecture-map-svg"
                      viewBox={`0 0 ${architectureFullGraphLayout.width} ${architectureFullGraphLayout.height}`}
                      aria-label="Agent graph edges"
                    >
                      {architectureFullGraphLayout.lanes.map(lane => (
                        <g key={lane.id}>
                          <rect
                            x={lane.x}
                            y={lane.y}
                            width={lane.width}
                            height={lane.height}
                            rx={10}
                            className="architecture-lane"
                          />
                          <text x={lane.x + 18} y={40} className="architecture-lane-label">
                            {lane.label}
                          </text>
                        </g>
                      ))}
                      {architectureFullGraphLayout.edges.map(edge => {
                        const selected = selectedArchitectureEdgeId === edge.id
                        return (
                          <path
                            key={edge.id}
                            d={edge.path}
                            role="button"
                            tabIndex={0}
                            aria-label={`Inspect edge ${edge.label || edge.id}`}
                            data-edge-id={edge.id}
                            data-edge-layer={edge.layer}
                            data-edge-highlighted={selected ? 'true' : 'false'}
                            className={[
                              'architecture-edge',
                              edge.kind === 'routing_path' ? 'architecture-edge-routing' : '',
                              selected ? 'architecture-edge-selected' : '',
                            ].filter(Boolean).join(' ')}
                            onClick={() => {
                              setSelectedArchitectureEdgeId(edge.id)
                              setSelectedArchitectureNodeId('')
                              setSelectedArchitecturePathId('')
                            }}
                            onKeyDown={event => {
                              if (event.key === 'Enter' || event.key === ' ') {
                                event.preventDefault()
                                setSelectedArchitectureEdgeId(edge.id)
                                setSelectedArchitectureNodeId('')
                                setSelectedArchitecturePathId('')
                              }
                            }}
                          />
                        )
                      })}
                    </svg>
                    {architectureFullGraphLayout.nodes.map(node => {
                      const selected = selectedArchitectureNodeId === node.id
                      return (
                        <button
                          key={node.id}
                          type="button"
                          aria-label={node.label}
                          className={[
                            'architecture-node',
                            `architecture-node-${node.kind}`,
                            selected ? 'architecture-node-selected' : '',
                          ].filter(Boolean).join(' ')}
                          style={{
                            left: `${node.x}px`,
                            top: `${node.y}px`,
                            width: `${node.width}px`,
                            height: `${node.height}px`,
                          }}
                          onClick={() => {
                            setSelectedArchitectureNodeId(node.id)
                            setSelectedArchitectureEdgeId('')
                            setSelectedArchitecturePathId('')
                          }}
                        >
                          <span className="architecture-node-label">{node.label}</span>
                          <span className="architecture-node-meta">
                            {node.kind === 'agent' ? (node.mode || node.role_kind || 'agent') : node.kind}
                          </span>
                        </button>
                      )
                    })}
                  </div>
                </div>
              </SurfaceCard>

              <div className="content-stack architecture-agent-graph-side">
                <SurfaceCard
                  title={selectedArchitectureEdge ? 'Edge Inspector' : selectedArchitectureNode ? 'Node Inspector' : 'Graph Inspector'}
                  subtitle="Focused metadata for the selected graph element."
                >
                  {selectedArchitectureEdge ? (
                    <>
                      <div className="badge-cluster">
                        <StatusBadge tone={edgeTone(selectedArchitectureEdge)}>{humanizeKey(selectedArchitectureEdge.kind)}</StatusBadge>
                        {selectedArchitectureEdge.emphasis && (
                          <StatusBadge tone={selectedArchitectureEdge.emphasis === 'high' ? 'accent' : 'neutral'}>
                            {humanizeKey(selectedArchitectureEdge.emphasis)}
                          </StatusBadge>
                        )}
                      </div>
                      <div className="summary-list">
                        <div className="summary-row">
                          <span>Label</span>
                          <strong>{selectedArchitectureEdge.label || selectedArchitectureEdge.id}</strong>
                        </div>
                        <div className="summary-row">
                          <span>Source</span>
                          <strong>{architectureNodeMap.get(selectedArchitectureEdge.source)?.label || selectedArchitectureEdge.source}</strong>
                        </div>
                        <div className="summary-row">
                          <span>Target</span>
                          <strong>{architectureNodeMap.get(selectedArchitectureEdge.target)?.label || selectedArchitectureEdge.target}</strong>
                        </div>
                      </div>
                      <JsonInspector label="Edge details" value={selectedArchitectureEdge} />
                    </>
                  ) : selectedArchitectureNode ? (
                    <>
                      <div className="badge-cluster">
                        <StatusBadge tone={nodeTone(selectedArchitectureNode)}>{selectedArchitectureNode.kind}</StatusBadge>
                        {asArray<string>(selectedArchitectureNode.badges).map(badge => (
                          <StatusBadge key={badge} tone="neutral">{badge}</StatusBadge>
                        ))}
                      </div>
                      <div className="summary-list">
                        <div className="summary-row">
                          <span>Label</span>
                          <strong>{selectedArchitectureNode.label}</strong>
                        </div>
                        <div className="summary-row">
                          <span>Status</span>
                          <StatusBadge tone={toneForStatus(selectedArchitectureNode.status)}>{selectedArchitectureNode.status}</StatusBadge>
                        </div>
                        <div className="summary-row">
                          <span>Description</span>
                          <strong>{selectedArchitectureNode.description || 'No description provided.'}</strong>
                        </div>
                        {selectedArchitectureNode.allowed_tools && (
                          <div className="summary-row">
                            <span>Tool Access</span>
                            <strong>{shortList(selectedArchitectureNode.allowed_tools)}</strong>
                          </div>
                        )}
                        {selectedArchitectureNode.allowed_worker_agents && (
                          <div className="summary-row">
                            <span>Worker Agents</span>
                            <strong>{shortList(selectedArchitectureNode.allowed_worker_agents)}</strong>
                          </div>
                        )}
                      </div>
                      <JsonInspector label="Node details" value={selectedArchitectureNode} />
                    </>
                  ) : (
                    <EmptyState title="Select a graph element" body="Choose a node or edge in the graph to inspect its live registry metadata." />
                  )}
                </SurfaceCard>

                <SurfaceCard
                  title="LangGraph Export"
                  subtitle="Compiled ReAct graph metadata for the selected default agent, exported without invoking the agent."
                  actions={(
                    <StatusBadge tone={langGraphStatusTone(langGraphExport)}>
                      {humanizeKey(asString(langGraphExport?.status, 'unavailable'))}
                    </StatusBadge>
                  )}
                >
                  {asString(langGraphExport?.status).toLowerCase() === 'available' ? (
                    <>
                      <div className="summary-list">
                        <div className="summary-row">
                          <span>Agent</span>
                          <strong>{asString(langGraphExport?.agent_name, 'default')}</strong>
                        </div>
                        <div className="summary-row">
                          <span>LangGraph Nodes</span>
                          <strong>{formatWholeNumber(langGraphNodes.length)}</strong>
                        </div>
                        <div className="summary-row">
                          <span>LangGraph Edges</span>
                          <strong>{formatWholeNumber(langGraphEdges.length)}</strong>
                        </div>
                        {langGraphExport?.generated_at && (
                          <div className="summary-row">
                            <span>Generated</span>
                            <strong>{formatTimestamp(langGraphExport.generated_at)}</strong>
                          </div>
                        )}
                      </div>

                      {langGraphWarnings.length > 0 && (
                        <div className="inline-alert inline-alert-warning">
                          <span>Export notes</span>
                          <strong>{langGraphWarnings.join(' | ')}</strong>
                        </div>
                      )}

                      <DetailTabs
                        tabs={[
                          {
                            id: 'nodes',
                            label: 'Nodes',
                            content: (
                              <div className="langgraph-mini-list">
                                {langGraphNodes.map((node, index) => (
                                  <div key={`${asString(node.id, 'node')}-${index}`} className="meta-chip">
                                    <span>{langGraphItemLabel(node)}</span>
                                    <strong>{asString(node.data_type, 'node')}</strong>
                                  </div>
                                ))}
                              </div>
                            ),
                          },
                          {
                            id: 'edges',
                            label: 'Edges',
                            content: (
                              <div className="langgraph-mini-list">
                                {langGraphEdges.map((edge, index) => (
                                  <div key={`${asString(edge.id, 'edge')}-${index}`} className="meta-chip">
                                    <span>{asString(edge.source, 'source')} {'->'} {asString(edge.target, 'target')}</span>
                                    <strong>{Boolean(edge.conditional) ? 'Conditional' : 'Direct'}</strong>
                                  </div>
                                ))}
                              </div>
                            ),
                          },
                          {
                            id: 'mermaid',
                            label: 'Mermaid',
                            content: <div className="code-panel langgraph-mermaid-source">{asString(langGraphExport?.mermaid, 'No Mermaid source returned.')}</div>,
                          },
                        ]}
                      />
                    </>
                  ) : (
                    <>
                      <EmptyState
                        title="LangGraph export unavailable"
                        body={langGraphWarnings[0] || 'The native graph still reflects the live registry, but this runtime could not export a compiled LangGraph view.'}
                      />
                      {langGraphExport && <JsonInspector label="Export payload" value={langGraphExport} />}
                    </>
                  )}
                </SurfaceCard>
              </div>
            </div>
          )}

          {architectureTab === 'paths' && (
            <div className="architecture-path-grid">
              {(architecture.canonical_paths ?? []).map(path => (
                <SurfaceCard
                  key={path.id}
                  className={selectedArchitecturePathId === path.id ? 'path-card path-card-active' : 'path-card'}
                  title={path.label}
                  subtitle={path.summary}
                  actions={<StatusBadge tone={path.route === 'BASIC' ? 'warning' : 'accent'}>{path.route}</StatusBadge>}
                >
                  <div className="badge-cluster">
                    {asArray<string>(path.badges).map(badge => (
                      <StatusBadge key={badge} tone="neutral">{badge}</StatusBadge>
                    ))}
                    {path.target_agent && <StatusBadge tone="ok">{path.target_agent}</StatusBadge>}
                  </div>
                  <p className="muted-copy">{path.when}</p>
                  <ActionBar>
                    <ActionButton
                      tone="secondary"
                      onClick={() => {
                        setSelectedArchitecturePathId(path.id)
                        setSelectedArchitectureNodeId('')
                        setSelectedArchitectureEdgeId('')
                        setArchitectureTab('map')
                      }}
                    >
                      Trace On Map
                    </ActionButton>
                  </ActionBar>
                </SurfaceCard>
              ))}
            </div>
          )}

          {architectureTab === 'traffic' && (
            <div className="content-stack">
              <div className="stat-grid">
                {Object.entries(architectureActivity?.route_counts ?? {}).map(([route, count]) => (
                  <StatCard key={route} label={route} value={count} caption="Observed starting routes" tone={route === 'BASIC' ? 'warning' : 'accent'} />
                ))}
                {Object.entries(architectureActivity?.router_method_counts ?? {}).map(([method, count]) => (
                  <StatCard key={method} label={humanizeKey(method)} value={count} caption="Router method hits" />
                ))}
                {Object.entries(architectureActivity?.outcome_counts ?? {}).map(([label, count]) => (
                  <StatCard
                    key={`outcome-${label}`}
                    label={humanizeKey(label)}
                    value={count}
                    caption="Router outcomes"
                    tone={label === 'negative' ? 'warning' : label === 'positive' ? 'accent' : 'default'}
                  />
                ))}
              </div>
              <div className="architecture-traffic-grid">
                <SurfaceCard title="Hot Start Agents" subtitle="Which agents are actually starting turns most often in the latest observed runtime activity.">
                  {Object.keys(architectureActivity?.start_agent_counts ?? {}).length > 0 ? (
                    <div className="field-stack">
                      {Object.entries(architectureActivity?.start_agent_counts ?? {}).map(([agentName, count]) => (
                        <div key={agentName} className="summary-row">
                          <span>{agentName}</span>
                          <strong>{count}</strong>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <EmptyState title="No traffic yet" body="Once users run turns through the system, start-agent counts will appear here." />
                  )}
                </SurfaceCard>

                <SurfaceCard title="Worker Handoffs" subtitle="Recent worker usage aggregated from runtime worker-start events.">
                  {Object.keys(architectureActivity?.delegation_counts ?? {}).length > 0 ? (
                    <div className="field-stack">
                      {Object.entries(architectureActivity?.delegation_counts ?? {}).map(([agentName, count]) => (
                        <div key={agentName} className="summary-row">
                          <span>{agentName}</span>
                          <strong>{count}</strong>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <EmptyState title="No delegations yet" body="Worker handoffs will show up here when top-level agents start delegating background or scoped work." />
                  )}
                </SurfaceCard>
              </div>
              <div className="architecture-traffic-grid">
                <SurfaceCard title="Router Quality" subtitle="Negative-rate rollups from the router feedback loop, based on verifier outcomes, retries, overrides, and degraded fallbacks.">
                  {Object.keys(architectureActivity?.negative_rate_by_route ?? {}).length > 0 ? (
                    <div className="field-stack">
                      {Object.entries(architectureActivity?.negative_rate_by_route ?? {}).map(([route, rate]) => (
                        <div key={route} className="summary-row">
                          <span>{route}</span>
                          <strong>{formatPercent(rate)}</strong>
                        </div>
                      ))}
                      {Object.entries(architectureActivity?.negative_rate_by_router_method ?? {}).map(([method, rate]) => (
                        <div key={`method-${method}`} className="summary-row">
                          <span>{humanizeKey(method)}</span>
                          <strong>{formatPercent(rate)}</strong>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <EmptyState title="No scored router outcomes yet" body="Once routed turns accumulate outcome signals, quality rates will appear here." />
                  )}
                </SurfaceCard>

                <SurfaceCard title="Review Backlog" subtitle="Sampled mis-picks queued for human review, plus the latest retrain artifact metadata.">
                  {reviewBacklog ? (
                    <div className="field-stack">
                      <div className="summary-row">
                        <span>Pending reviews</span>
                        <strong>{formatWholeNumber(reviewBacklog.pending)}</strong>
                      </div>
                      <div className="summary-row">
                        <span>Total samples</span>
                        <strong>{formatWholeNumber(reviewBacklog.total_samples)}</strong>
                      </div>
                      <div className="summary-row">
                        <span>Negative samples</span>
                        <strong>{formatWholeNumber(reviewBacklog.negative_samples)}</strong>
                      </div>
                      <div className="summary-row">
                        <span>Neutral samples</span>
                        <strong>{formatWholeNumber(reviewBacklog.neutral_samples)}</strong>
                      </div>
                      {lastRetrainReport && Object.keys(lastRetrainReport).length > 0 && (
                        <>
                          <div className="summary-row">
                            <span>Last report</span>
                            <strong>{formatTimestamp(lastRetrainReport.generated_at)}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Quarter</span>
                            <strong>{asString(lastRetrainReport.quarter, 'Current')}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Recommended threshold</span>
                            <strong>{asString(lastRetrainReport.recommended_threshold, 'n/a')}</strong>
                          </div>
                        </>
                      )}
                    </div>
                  ) : (
                    <EmptyState title="No review metadata yet" body="Review backlog and retrain artifact details will populate once router outcomes are sampled." />
                  )}
                </SurfaceCard>
              </div>
              <SurfaceCard title="Recent Mispicks" subtitle="Sampled router decisions that should be reviewed before future pattern or threshold updates.">
                {recentMispicks.length > 0 ? (
                  <div className="traffic-flow-list">
                    {recentMispicks.map(sample => (
                      <article key={asString(sample.sample_id)} className="traffic-flow-card">
                        <div className="tool-card-head">
                          <strong>{asString(sample.route, 'route')}</strong>
                          <StatusBadge
                            tone={
                              asString(sample.outcome_label) === 'negative'
                                ? 'danger'
                                : asString(sample.outcome_label) === 'positive'
                                  ? 'ok'
                                  : 'warning'
                            }
                          >
                            {humanizeKey(asString(sample.outcome_label, 'pending'))}
                          </StatusBadge>
                        </div>
                        <div className="summary-list">
                          <div className="summary-row">
                            <span>Decision</span>
                            <strong>{shortId(asString(sample.router_decision_id, 'unknown'))}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Router Method</span>
                            <strong>{asString(sample.router_method, 'deterministic')}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Suggested Agent</span>
                            <strong>{asString(sample.suggested_agent, 'Default')}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Created</span>
                            <strong>{formatTimestamp(sample.created_at)}</strong>
                          </div>
                        </div>
                        {asArray<string>(sample.evidence_signals).length > 0 && (
                          <div className="badge-cluster">
                            {uniqueList(asArray<string>(sample.evidence_signals)).map(signal => (
                              <StatusBadge key={signal} tone="warning">{humanizeKey(signal)}</StatusBadge>
                            ))}
                          </div>
                        )}
                      </article>
                    ))}
                  </div>
                ) : (
                  <EmptyState title="No sampled mis-picks yet" body="Negative outcomes and a small slice of neutral outcomes will appear here once the feedback loop has enough traffic." />
                )}
              </SurfaceCard>
              <SurfaceCard title="Recent Flows" subtitle="Privacy-safe summaries of recent routing decisions, specialist starts, fallback conditions, and worker handoffs.">
                {asArray(architectureActivity?.recent_flows).length > 0 ? (
                  <div className="traffic-flow-list">
                    {asArray<Record<string, unknown>>(architectureActivity?.recent_flows).map(flow => (
                      <article key={asString(flow.session_id)} className="traffic-flow-card">
                        <div className="tool-card-head">
                          <strong>{asString(flow.start_agent || flow.route || 'flow')}</strong>
                          <StatusBadge tone={Boolean(flow.degraded) ? 'warning' : 'ok'}>
                            {Boolean(flow.degraded) ? 'Degraded' : asString(flow.route, 'Observed')}
                          </StatusBadge>
                        </div>
                        <div className="summary-list">
                          <div className="summary-row">
                            <span>Session</span>
                            <strong>{shortId(asString(flow.session_id, 'unknown-session'))}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Router Method</span>
                            <strong>{asString(flow.router_method, 'deterministic')}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Suggested Agent</span>
                            <strong>{asString(flow.suggested_agent, 'Default')}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Workers</span>
                            <strong>{shortList(asArray<string>(flow.worker_agents))}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Updated</span>
                            <strong>{formatTimestamp(flow.updated_at)}</strong>
                          </div>
                        </div>
                        {asArray<string>(flow.reasons).length > 0 && (
                          <div className="badge-cluster">
                            {uniqueList(asArray<string>(flow.reasons)).map(reason => (
                              <StatusBadge key={reason} tone="neutral">{reason}</StatusBadge>
                            ))}
                          </div>
                        )}
                        {asArray<string>(flow.degraded_events).length > 0 && (
                          <div className="inline-alert">
                            <span>Fallback</span>
                            <strong>{shortList(asArray<string>(flow.degraded_events))}</strong>
                          </div>
                        )}
                      </article>
                    ))}
                  </div>
                ) : (
                  <EmptyState title="No live traffic yet" body="This view starts empty on a fresh runtime. As users send requests, the panel will summarize actual route starts and worker handoffs here." />
                )}
                {architectureActivity && <JsonInspector label="Technical details" value={architectureActivity} />}
              </SurfaceCard>
            </div>
          )}
        </div>
      )}

      {active === 'config' && (
        <div className="content-stack">
          {Object.keys(configChanges).filter(k => configChanges[k] !== '').length > 0 && (
            <div className="config-ribbon" role="status" aria-live="polite">
              <div className="config-ribbon-body">
                <strong>{Object.keys(configChanges).filter(k => configChanges[k] !== '').length} unsaved {Object.keys(configChanges).filter(k => configChanges[k] !== '').length === 1 ? 'change' : 'changes'}</strong>
                <span className="muted-copy">Validate or apply to update the runtime config.</span>
              </div>
              <div className="config-ribbon-actions">
                <ActionButton tone="ghost" onClick={() => setConfigDiffOpen(true)}>Preview diff</ActionButton>
                <ActionButton tone="ghost" onClick={() => askConfirm({
                  title: 'Discard unsaved changes?',
                  description: 'All drafted values will be cleared. Applied values in the runtime are not affected.',
                  confirmLabel: 'Discard',
                  run: () => { setConfigChanges({}); notifyOk('Unsaved changes discarded') },
                })}>Discard</ActionButton>
                <ActionButton tone="primary" onClick={() => void handleConfigApply()}>Apply now</ActionButton>
              </div>
            </div>
          )}
          <div className="card-grid">
            {visibleConfigGroup ? (
              <SurfaceCard key={visibleConfigGroup[0]} title={visibleConfigGroup[0]} subtitle="Review the live value, make a draft change, then validate before applying.">
                <div className="field-stack">
                  {visibleConfigGroup[1].map(field => {
                    const fieldName = configFieldName(field)
                    const draftValue = configChanges[fieldName] ?? ''
                    const currentValue = configEffective[fieldName] ?? field.value
                    const changed = draftValue !== ''
                    const sliderDraftValue = changed
                      ? draftValue
                      : asString(currentValue, asString(field.min_value ?? 0))
                    return (
                      <div
                        key={fieldName}
                        className={[
                          'config-row',
                          changed ? 'config-row-changed' : '',
                          field.readonly ? 'config-row-readonly' : '',
                        ].filter(Boolean).join(' ')}
                      >
                        <div className="config-head">
                          <div>
                            <label className="config-label" htmlFor={fieldName}>{field.label}</label>
                            <p className="muted-copy">{field.description}</p>
                          </div>
                          <div className="badge-cluster">
                            {field.readonly && (
                              <Tooltip content="Surfaced here for reference; must be set via env var or host config.">
                                <StatusBadge tone="warning">Read only</StatusBadge>
                              </Tooltip>
                            )}
                            {field.secret && (
                              <Tooltip content="Value is masked in the UI and redacted from audit logs.">
                                <StatusBadge tone="neutral">Secret</StatusBadge>
                              </Tooltip>
                            )}
                            {!field.readonly && changed && (
                              <Tooltip content="Change staged locally but not yet applied to the runtime.">
                                <StatusBadge tone="accent">Drafted</StatusBadge>
                              </Tooltip>
                            )}
                            <Tooltip content={field.reload_scope === 'live' ? 'Change takes effect immediately on save.' : 'Change requires a service restart before it takes effect.'}>
                              <StatusBadge tone="neutral">{field.reload_scope}</StatusBadge>
                            </Tooltip>
                          </div>
                        </div>
                        <div className="config-values">
                          <div className="value-panel">
                            <span>Current</span>
                            <code>{maskSecret(field, asString(currentValue))}</code>
                          </div>
                          <div className="value-panel">
                            <span>Draft</span>
                            {field.readonly ? (
                              <div className="static-pill">Managed at startup only</div>
                            ) : field.ui_control === 'slider' ? (
                              <div className="slider-field">
                                <div className="slider-value-row">
                                  <strong>{sliderDraftValue}</strong>
                                  <span className="muted-copy">0-100</span>
                                </div>
                                <input
                                  id={fieldName}
                                  aria-label={field.label}
                                  type="range"
                                  min={field.min_value ?? 0}
                                  max={field.max_value ?? 100}
                                  step={field.step ?? 1}
                                  value={asNumber(sliderDraftValue) ?? field.min_value ?? 0}
                                  onChange={event => setConfigChanges(current => ({ ...current, [fieldName]: event.target.value }))}
                                />
                                <div className="slider-scale" aria-hidden="true">
                                  <span>0</span>
                                  <span>50</span>
                                  <span>100</span>
                                </div>
                                <div className="slider-cues" aria-hidden="true">
                                  <span>Proceed</span>
                                  <span>Balanced</span>
                                  <span>Ask early</span>
                                </div>
                              </div>
                            ) : field.kind === 'enum' ? (
                              <select
                                id={fieldName}
                                aria-label={field.label}
                                value={draftValue}
                                onChange={event => setConfigChanges(current => ({ ...current, [fieldName]: event.target.value }))}
                              >
                                <option value="">No change</option>
                                {field.choices.map(choice => <option key={choice} value={choice}>{choice}</option>)}
                              </select>
                            ) : field.kind === 'bool' ? (
                              <select
                                id={fieldName}
                                aria-label={field.label}
                                value={draftValue}
                                onChange={event => setConfigChanges(current => ({ ...current, [fieldName]: event.target.value }))}
                              >
                                <option value="">No change</option>
                                <option value="true">true</option>
                                <option value="false">false</option>
                              </select>
                            ) : (
                              <input
                                id={fieldName}
                                aria-label={field.label}
                                type={field.secret ? 'password' : 'text'}
                                value={draftValue}
                                onChange={event => setConfigChanges(current => ({ ...current, [fieldName]: event.target.value }))}
                                placeholder={field.secret && field.is_configured ? 'configured' : 'Set new value'}
                              />
                            )}
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </SurfaceCard>
            ) : (
              <SurfaceCard title="No settings match" subtitle="Try another setting name, environment variable, provider, or capability.">
                <EmptyState title="No settings found" body="The current search filters out every settings group." />
              </SurfaceCard>
            )}
          </div>

          <ActionBar sticky>
            <ActionButton tone="secondary" onClick={() => void handleConfigValidate()}>Validate</ActionButton>
            <ActionButton tone="primary" onClick={() => void handleConfigApply()}>Apply</ActionButton>
          </ActionBar>

          <CollapsibleSurfaceCard
            title="Preview"
            subtitle="Readable validation output with before and after values, plus reload scope."
            open={configPreviewOpen}
            onToggle={() => setConfigPreviewOpen(open => !open)}
          >
            {configPreview ? (
              <>
              <div className="badge-cluster">
                <StatusBadge tone={configPreview.valid ? 'ok' : 'danger'}>{configPreview.valid ? 'Valid' : 'Invalid'}</StatusBadge>
                {configPreview.applied && <StatusBadge tone="accent">Applied</StatusBadge>}
                {configPreview.reload_scope && <StatusBadge tone="warning">{configPreview.reload_scope}</StatusBadge>}
              </div>
              {configDiffEntries.length > 0 ? (
                <div className="diff-list">
                  {configDiffEntries.map(([key, diff]) => (
                    <div key={key} className="diff-row">
                      <div>
                        <strong>{humanizeKey(key)}</strong>
                        <p className="muted-copy">Changes live runtime behavior after apply.</p>
                      </div>
                      <div className="diff-values">
                        <code>{diff.before}</code>
                        <span className="diff-arrow" aria-hidden="true">→</span>
                        <code>{diff.after}</code>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <EmptyState title="No pending changes" body="Validate after drafting a value to see the diff and reload scope here." />
              )}
              <JsonInspector label="Technical details" value={configPreview} />
              </>
            ) : (
              <EmptyState title="No preview yet" body="Validate or apply a draft change to open a readable diff and reload summary here." />
            )}
          </CollapsibleSurfaceCard>
          <Dialog
            open={configDiffOpen}
            onClose={() => setConfigDiffOpen(false)}
            title="Unsaved config changes"
            description="Preview the pending draft values before applying."
            size="md"
            footer={(
              <>
                <ActionButton tone="ghost" onClick={() => setConfigDiffOpen(false)}>Close</ActionButton>
                <ActionButton tone="primary" onClick={() => { setConfigDiffOpen(false); void handleConfigApply() }}>Apply now</ActionButton>
              </>
            )}
          >
            {Object.keys(configChanges).filter(k => configChanges[k] !== '').length === 0 ? (
              <EmptyState title="No unsaved changes" body="Drafted values will show up here with a before/after diff." />
            ) : (
              <div className="diff-list">
                {Object.entries(configChanges).filter(([, v]) => v !== '').map(([key, newValue]) => {
                  const before = asString(configEffective[key] ?? '', '—')
                  return (
                    <div key={key} className="diff-row">
                      <div>
                        <strong>{humanizeKey(key)}</strong>
                        <p className="muted-copy"><code>{key}</code></p>
                      </div>
                      <div className="diff-values">
                        <code>{before}</code>
                        <span className="diff-arrow" aria-hidden="true">→</span>
                        <code>{newValue}</code>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </Dialog>
        </div>
      )}

      {active === 'agents' && agentsPayload && (
        agentsTab === 'workspace' ? (
          <div className="studio-layout agent-studio">
            <SurfaceCard className="selection-rail agent-rail" title="Available Agents" subtitle="Choose which agent overlay you want to inspect or edit.">
              <ResourceSearch value={agentSearch} onChange={setAgentSearch} placeholder="Search agents" />
              <EntityList
                variant="rail"
                items={filteredAgents}
                selectedKey={selectedAgent}
                getKey={agent => asString(agent.name)}
                getLabel={agent => asString(agent.name)}
                getDescription={agent => asString(agent.prompt_file, 'No prompt file')}
                getMeta={agent => (
                  <StatusBadge tone={Boolean(agent.overlay_active) ? 'accent' : 'neutral'}>
                    {Boolean(agent.overlay_active) ? 'Overlay active' : 'Base only'}
                  </StatusBadge>
                )}
                emptyText="No agents are registered in the current runtime."
                onSelect={agent => setSelectedAgent(asString(agent.name))}
              />
            </SurfaceCard>

            <CollapsibleSurfaceCard
              className="editor-pane agent-editor-pane"
              title="Agent Editor"
              subtitle="Adjust editable overlay fields, then save the overlay before reloading agents."
              open={agentEditorOpen}
              onToggle={() => setAgentEditorOpen(open => !open)}
            >
              {agentDetail ? (
                <>
                  <div className="form-grid form-grid-compact">
                    {AGENT_EDITOR_FIELDS.map(key => (
                      <label key={key} className="field">
                        <span>{humanizeKey(key)}</span>
                        <input
                          aria-label={key}
                          value={asString(agentForm[key])}
                          onChange={event => setAgentForm(current => ({ ...current, [key]: event.target.value }))}
                        />
                      </label>
                    ))}
                  </div>
                  <div className="field-stack">
                    <label className="field">
                      <span>Body</span>
                      <textarea
                        aria-label="body"
                        rows={16}
                        value={asString(agentForm.body)}
                        onChange={event => setAgentForm(current => ({ ...current, body: event.target.value }))}
                      />
                    </label>
                  </div>
                  <ActionBar>
                    <ActionButton tone="secondary" onClick={() => void handleAgentSave()}>Save Overlay</ActionButton>
                    <ActionButton tone="primary" onClick={() => void handleAgentReload()}>Reload Agents</ActionButton>
                  </ActionBar>
                </>
              ) : (
                <EmptyState title="Choose an agent" body="The editor becomes active once an agent is selected from the left rail." />
              )}
            </CollapsibleSurfaceCard>

            <div className="content-stack studio-sidebar">
              <CollapsibleSurfaceCard
                title="Agent Inspector"
                subtitle="Operational summary for the selected agent, including tools, workers, and pinned skills."
                open={agentInspectorOpen}
                onToggle={() => setAgentInspectorOpen(open => !open)}
              >
                {agentDetail ? (
                  <>
                    <div className="summary-list">
                      <div className="summary-row">
                        <span>Description</span>
                        <strong>{asString(agentDetail.description, 'No description')}</strong>
                      </div>
                      <div className="summary-row">
                        <span>Prompt File</span>
                        <strong>{asString(agentDetail.prompt_file, 'None')}</strong>
                      </div>
                      <div className="summary-row">
                        <span>Overlay Status</span>
                        <StatusBadge tone={Boolean(agentDetail.overlay_active) ? 'accent' : 'neutral'}>
                          {Boolean(agentDetail.overlay_active) ? 'Overlay active' : 'Base only'}
                        </StatusBadge>
                      </div>
                      <div className="summary-row">
                        <span>Pinned Skills</span>
                        <strong>{shortList(asArray<string>(agentDetail.preload_skill_packs))}</strong>
                      </div>
                      <div className="summary-row">
                        <span>Tool Access</span>
                        <strong>{shortList(asArray<string>(agentDetail.allowed_tools))}</strong>
                      </div>
                      <div className="summary-row">
                        <span>Worker Agents</span>
                        <strong>{shortList(asArray<string>(agentDetail.allowed_worker_agents))}</strong>
                      </div>
                    </div>
                    <JsonInspector label="Technical details" value={agentDetail} />
                  </>
                ) : (
                  <EmptyState title="No agent selected" body="Select an agent to see its overlay status, tool access, and technical metadata." />
                )}
              </CollapsibleSurfaceCard>
            </div>
          </div>
        ) : (
          <div className="content-stack">
            <SurfaceCard title="Tool Catalog" subtitle="Current tool contracts available to the runtime, organized by catalog tags.">
              {toolCatalog.length > 0 ? (
                <div className="tool-catalog">
                  <div className="tool-catalog-toolbar">
                    <div className="tool-catalog-summary">
                      <span><strong>{filteredToolCatalogItems.length}</strong> shown</span>
                      <span><strong>{groupedToolCatalog.length}</strong> groups</span>
                      <span><strong>{toolTagOptions.length}</strong> tags</span>
                    </div>
                    <div className="tool-tag-filter-row" aria-label="Tool tag filters">
                      <button
                        type="button"
                        className="filter-chip-btn tool-tag-filter-btn"
                        aria-pressed={!activeToolTag}
                        onClick={() => setToolTagFilter('')}
                      >
                        <FilterChip label="All" count={toolCatalog.length} tone={!activeToolTag ? 'accent' : 'neutral'} />
                      </button>
                      {toolTagOptions.map(option => (
                        <button
                          key={option.tag}
                          type="button"
                          className="filter-chip-btn tool-tag-filter-btn"
                          aria-pressed={activeToolTag === option.tag}
                          onClick={() => setToolTagFilter(activeToolTag === option.tag ? '' : option.tag)}
                        >
                          <FilterChip label={toolTagLabel(option.tag)} count={option.count} tone={activeToolTag === option.tag ? 'accent' : 'neutral'} />
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="tool-tag-groups">
                    {groupedToolCatalog.map(group => (
                      <section key={group.tag} className="tool-tag-group" aria-labelledby={`tool-tag-group-${group.tag.replace(/[^a-zA-Z0-9_-]/g, '-')}`}>
                        <div className="tool-tag-group-head">
                          <h4 id={`tool-tag-group-${group.tag.replace(/[^a-zA-Z0-9_-]/g, '-')}`}>{toolTagLabel(group.tag)}</h4>
                          <StatusBadge tone="neutral">{group.items.length} tools</StatusBadge>
                        </div>
                        <div className="tool-card-grid">
                          {group.items.map(({ tool, tags }) => (
                            <div key={asString(tool.name)} className="tool-card">
                              <div className="tool-card-head">
                                <strong>{asString(tool.name)}</strong>
                                <StatusBadge tone={Boolean(tool.read_only) ? 'ok' : Boolean(tool.destructive) ? 'danger' : 'neutral'}>
                                  {Boolean(tool.read_only) ? 'Read only' : Boolean(tool.destructive) ? 'Destructive' : 'Mutable'}
                                </StatusBadge>
                              </div>
                              <p>{asString(tool.description, 'No description')}</p>
                              <div className="tool-card-tags">
                                {tags.map(tag => (
                                  <button
                                    key={tag}
                                    type="button"
                                    className="filter-chip-btn tool-tag-filter-btn"
                                    aria-pressed={activeToolTag === tag}
                                    onClick={() => setToolTagFilter(activeToolTag === tag ? '' : tag)}
                                  >
                                    <FilterChip label={toolTagLabel(tag)} tone={activeToolTag === tag ? 'accent' : 'neutral'} />
                                  </button>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                      </section>
                    ))}
                  </div>
                </div>
              ) : (
                <EmptyState title="No tools registered" body="Tool metadata will appear here after the runtime publishes its current tool catalog." />
              )}
            </SurfaceCard>
          </div>
        )
      )}

      {active === 'prompts' && (
        promptsTab === 'edit' ? (
          <div className="studio-layout prompt-edit-layout">
            <SurfaceCard className="selection-rail prompt-rail" title="Prompt Files" subtitle="Pick a prompt to edit. Overlays take precedence on the next turn without a full runtime reload.">
              <ResourceSearch value={promptSearch} onChange={setPromptSearch} placeholder="Search prompts" />
              <EntityList
                variant="rail"
                items={filteredPrompts}
                selectedKey={selectedPrompt}
                getKey={prompt => asString(prompt.prompt_file)}
                getLabel={prompt => asString(prompt.prompt_file)}
                getDescription={prompt => (Boolean(prompt.overlay_active) ? 'Custom overlay saved' : 'Using base prompt')}
                getMeta={prompt => (
                  <StatusBadge tone={Boolean(prompt.overlay_active) ? 'accent' : 'neutral'}>
                    {Boolean(prompt.overlay_active) ? 'Overlay active' : 'Base only'}
                  </StatusBadge>
                )}
                emptyText="No prompt files were returned by the backend."
                onSelect={prompt => setSelectedPrompt(asString(prompt.prompt_file))}
              />
            </SurfaceCard>

            <SurfaceCard className="editor-pane prompt-editor-pane" title="Prompt Editor" subtitle="Edit only the overlay content here. The compare view stays available in its own mode to reduce visual noise.">
              {promptDetail ? (
                <>
                  <div className="badge-cluster">
                    <StatusBadge tone={promptOverlayActive ? 'accent' : 'neutral'}>
                      {promptOverlayActive ? 'Overlay active' : 'Using base prompt'}
                    </StatusBadge>
                    <StatusBadge tone="neutral">{asString(promptDetail.kind, 'prompt')}</StatusBadge>
                    <StatusBadge tone="warning">Next turn applies changes</StatusBadge>
                  </div>
                  <label className="field">
                    <span>Prompt Editor</span>
                    <textarea
                      aria-label="Prompt Editor"
                      rows={20}
                      className="editor-textarea"
                      value={promptDraft}
                      onChange={event => setPromptDraft(event.target.value)}
                    />
                  </label>
                  <ActionBar>
                    <ActionButton tone="secondary" onClick={() => void handlePromptSave()}>Save Overlay</ActionButton>
                    <ActionButton tone="destructive" onClick={() => void handlePromptReset()}>Reset Overlay</ActionButton>
                  </ActionBar>
                </>
              ) : (
                <EmptyState title="Choose a prompt" body="Select a prompt file to compare its base content, overlay state, and effective output." />
              )}
            </SurfaceCard>
          </div>
        ) : (
          <div className="studio-layout prompt-compare-layout">
            <SurfaceCard className="selection-rail prompt-rail" title="Prompt Files" subtitle="Choose which prompt snapshot you want to compare.">
              <ResourceSearch value={promptSearch} onChange={setPromptSearch} placeholder="Search prompts" />
              <EntityList
                variant="rail"
                items={filteredPrompts}
                selectedKey={selectedPrompt}
                getKey={prompt => asString(prompt.prompt_file)}
                getLabel={prompt => asString(prompt.prompt_file)}
                getDescription={prompt => (Boolean(prompt.overlay_active) ? 'Custom overlay saved' : 'Using base prompt')}
                getMeta={prompt => (
                  <StatusBadge tone={Boolean(prompt.overlay_active) ? 'accent' : 'neutral'}>
                    {Boolean(prompt.overlay_active) ? 'Overlay active' : 'Base only'}
                  </StatusBadge>
                )}
                emptyText="No prompt files were returned by the backend."
                onSelect={prompt => setSelectedPrompt(asString(prompt.prompt_file))}
              />
            </SurfaceCard>

            <div className="content-stack studio-sidebar">
              <CollapsibleSurfaceCard
                title="Prompt Summary"
                subtitle="Quick context for the selected prompt before you dive into the compare tabs."
                open={promptSummaryOpen}
              onToggle={() => setPromptSummaryOpen(open => !open)}
              >
                {promptDetail ? (
                  <div className="summary-list">
                    <div className="summary-row">
                      <span>Prompt File</span>
                      <strong>{asString(promptDetail.prompt_file)}</strong>
                    </div>
                    <div className="summary-row">
                      <span>Overlay Status</span>
                      <StatusBadge tone={promptOverlayActive ? 'accent' : 'neutral'}>
                        {promptOverlayActive ? 'Overlay active' : 'Base only'}
                      </StatusBadge>
                    </div>
                    <div className="summary-row">
                      <span>Kind</span>
                      <strong>{asString(promptDetail.kind, 'prompt')}</strong>
                    </div>
                  </div>
                ) : (
                  <EmptyState title="No prompt selected" body="Choose a prompt to see its current file, overlay state, and prompt type." />
                )}
              </CollapsibleSurfaceCard>

              <SurfaceCard className="preview-pane prompt-snapshot-pane" title="Prompt Snapshot" subtitle="Compare the base prompt, live overlay, and effective content without reading raw JSON first.">
                {promptDetail ? (
                  <>
                    <DetailTabs
                      key={selectedPrompt}
                      tabs={[
                        {
                          id: 'effective',
                          label: 'Effective',
                          content: <div className="code-panel code-panel-scroll">{asString(promptDetail.effective_content, 'No prompt content')}</div>,
                        },
                        {
                          id: 'base',
                          label: 'Base',
                          content: <div className="code-panel code-panel-scroll">{asString(promptDetail.base_content, 'No base content')}</div>,
                        },
                        {
                          id: 'overlay',
                          label: 'Overlay',
                          content: <div className="code-panel code-panel-scroll">{asString(promptDetail.overlay_content, 'No overlay saved')}</div>,
                        },
                      ]}
                    />
                    <JsonInspector label="Technical details" value={promptDetail} />
                  </>
                ) : (
                  <EmptyState title="No prompt selected" body="The snapshot view will show how base and overlay content combine once you select a prompt." />
                )}
              </SurfaceCard>
            </div>
          </div>
        )
      )}

      {active === 'collections' && (
        <div className="workspace-grid workspace-grid-collections">
          <SurfaceCard className="selection-rail collection-rail" title="Collections" subtitle="Pick a namespace, keep empty collections visible, and switch the workspace without juggling multiple panels.">
            <ResourceSearch value={collectionSearch} onChange={setCollectionSearch} placeholder="Search knowledge" />
            <label className="field">
              <span>Available Collections</span>
              <select
                aria-label="Available Collections"
                value={selectedCollection}
                onChange={event => applyCollectionSelection(event.target.value)}
              >
                <option value="">Choose a collection</option>
                {filteredCollections.map(collection => (
                  <option key={collection.collection_id} value={collection.collection_id}>
                    {collection.collection_id}
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span>Collection ID</span>
              <input
                aria-label="Collection ID"
                value={collectionDraft}
                onChange={event => setCollectionDraft(event.target.value)}
                placeholder="collection-id"
              />
            </label>

            <ActionBar>
              <ActionButton tone="secondary" onClick={() => openIngestionWizard()}>
                Ingestion Wizard
              </ActionButton>
              <ActionButton tone="secondary" onClick={() => void handleUseCollection()}>Load Workspace</ActionButton>
              <ActionButton tone="primary" onClick={() => void handleCreateCollection()}>Create Collection</ActionButton>
              <ActionButton
                tone="secondary"
                onClick={() => {
                  const collectionId = normalizeCollectionId(selectedCollection || collectionDraft)
                  setGraphCollectionId(collectionId)
                  setGraphDraftId('')
                  setGraphDisplayNameDraft('')
                  setActive('graphs')
                  if (collectionId) void refreshGraphCollectionDocs(collectionId)
                }}
                disabled={!selectedCollectionCanBuildGraph}
              >
                Build Graph
              </ActionButton>
              <ActionButton
                tone="destructive"
                onClick={() => askConfirm({
                  title: 'Delete this collection?',
                  description: `Permanently delete "${selectedCollectionMeta?.collection_id ?? selectedCollection}". This cannot be undone.`,
                  confirmLabel: 'Delete',
                  run: handleDeleteCollection,
                })}
                disabled={!selectedCollectionMeta || selectedCollectionMeta.document_count > 0 || selectedCollectionMeta.graph_count > 0}
              >
                Delete Empty
              </ActionButton>
            </ActionBar>

            <div className="badge-cluster">
              <StatusBadge tone={selectedCollectionMeta ? 'ok' : 'neutral'}>
                {selectedCollectionMeta ? 'Cataloged' : 'Draft'}
              </StatusBadge>
              {selectedCollectionMeta && (
                <StatusBadge tone={collectionStatusSummary(selectedCollectionMeta).ready ? 'ok' : 'warning'}>
                  {collectionStatusSummary(selectedCollectionMeta).ready ? 'Ready' : humanizeKey(collectionStatusSummary(selectedCollectionMeta).reason)}
                </StatusBadge>
              )}
              {selectedCollectionMeta && <StatusBadge tone="accent">{selectedCollectionMeta.document_count} docs</StatusBadge>}
              {selectedCollectionMeta && <StatusBadge tone="neutral">{selectedCollectionMeta.graph_count} graphs</StatusBadge>}
            </div>

            {selectedCollectionMeta && (
              <div className="summary-list">
                <div className="summary-row">
                  <span>Recent Activity</span>
                  <strong>{formatTimestamp(selectedCollectionMeta.latest_ingested_at || selectedCollectionMeta.updated_at)}</strong>
                </div>
                <div className="summary-row">
                  <span>Embed Model</span>
                  <strong>{selectedCollectionStorage.embedding_model || 'Pending selection'}</strong>
                </div>
                <div className="summary-row">
                  <span>Maintenance</span>
                  <strong>{selectedCollectionMeta ? humanizeKey(selectedCollectionMeta.maintenance_policy || 'indexed_documents') : 'Indexed documents'}</strong>
                </div>
              </div>
            )}

            {collections.length > 0 ? (
              <EntityList
                variant="rail"
                items={collections}
                selectedKey={selectedCollection}
                getKey={collection => collection.collection_id}
                getLabel={collection => collection.collection_id}
                getDescription={collection => formatTimestamp(collection.latest_ingested_at || collection.updated_at)}
                getMeta={collection => (
                  <>
                    <StatusBadge tone={collectionStatusSummary(collection).ready ? 'ok' : 'warning'}>
                      {collectionStatusSummary(collection).ready ? 'Ready' : humanizeKey(collectionStatusSummary(collection).reason)}
                    </StatusBadge>
                    <span>{collection.document_count} docs</span>
                    <span>{collection.graph_count} graphs</span>
                  </>
                )}
                emptyText="Create a collection to keep an empty namespace visible before the first ingest."
                onSelect={collection => applyCollectionSelection(collection.collection_id)}
              />
            ) : (
              <EmptyState title="No collections yet" body="Create a collection, then use the workspace to ingest host paths or sync configured KB content." />
            )}
          </SurfaceCard>

          <div className="content-stack collection-main-stack">
            <SurfaceCard
              className="collection-workspace-card"
              title="Knowledge Builder"
              subtitle="Bring files, folders, or approved local repositories into a collection, then hand the corpus directly to GraphRAG."
            >
              <input
                ref={collectionFilesInputRef}
                aria-label="Collection Upload Files Input"
                type="file"
                multiple
                className="visually-hidden"
                tabIndex={-1}
                onChange={event => {
                  void handleCollectionFilesUpload(event.target.files)
                  event.currentTarget.value = ''
                }}
              />
              <input
                ref={node => {
                  collectionFolderInputRef.current = node
                  if (node) {
                    node.setAttribute('webkitdirectory', '')
                    node.setAttribute('directory', '')
                  }
                }}
                aria-label="Collection Upload Folder Input"
                type="file"
                multiple
                className="visually-hidden"
                tabIndex={-1}
                onChange={event => {
                  void handleCollectionFilesUpload(event.target.files)
                  event.currentTarget.value = ''
                }}
              />

              <div className="badge-cluster">
                <StatusBadge tone={selectedCollectionMeta ? 'accent' : 'neutral'}>
                  {selectedCollectionMeta?.collection_id || normalizeCollectionId(collectionDraft) || 'No collection selected'}
                </StatusBadge>
                {selectedCollectionMeta && (
                  <StatusBadge tone={collectionStatusSummary(selectedCollectionMeta).ready ? 'ok' : 'warning'}>
                    {collectionStatusSummary(selectedCollectionMeta).ready ? 'Ready' : humanizeKey(collectionStatusSummary(selectedCollectionMeta).reason)}
                  </StatusBadge>
                )}
                {Boolean(collectionActivityStatus) && (
                  <StatusBadge tone={toneForStatus(collectionActivityStatus)}>
                    {humanizeKey(collectionActivityStatus)}
                  </StatusBadge>
                )}
              </div>

              <div className="supported-type-strip" aria-label="Supported document types">
                <span>Supported Types</span>
                {SUPPORTED_DOCUMENT_TYPES.map(type => (
                  <StatusBadge key={type} tone="neutral">{type}</StatusBadge>
                ))}
              </div>

              <div className="collection-summary-strip">
                <div className="meta-chip">
                  <span>Docs</span>
                  <strong>{formatWholeNumber(selectedCollectionMeta?.document_count)}</strong>
                </div>
                <div className="meta-chip">
                  <span>Graphs</span>
                  <strong>{formatWholeNumber(selectedCollectionMeta?.graph_count)}</strong>
                </div>
                <div className="meta-chip">
                  <span>Embedding Model</span>
                  <strong>{selectedCollectionStorage.embedding_model || 'Pending selection'}</strong>
                </div>
                <div className="meta-chip">
                  <span>Configured Dim</span>
                  <strong>{selectedCollectionStorage.configured_embedding_dim || 'n/a'}</strong>
                </div>
                <div className="meta-chip">
                  <span>Maintenance</span>
                  <strong>{selectedCollectionMeta ? humanizeKey(selectedCollectionMeta.maintenance_policy || 'indexed_documents') : 'n/a'}</strong>
                </div>
              </div>

              <SectionTabs
                tabs={[
                  { id: 'upload', label: 'Upload' },
                  { id: 'local', label: 'Local Path' },
                  { id: 'registered', label: 'Registered Sources' },
                  { id: 'sync', label: 'Configured Sync' },
                ]}
                active={activeCollectionAction}
                onChange={value => setCollectionAction(value as CollectionActionMode)}
                ariaLabel="Collection actions"
                className="collection-action-tabs"
              />

              {activeCollectionAction === 'upload' && (
                <div className="collection-action-panel">
                  <div className="collection-action-copy">
                    <strong>Upload files or a folder</strong>
                    <p>Folder uploads preserve relative paths and index into this knowledge collection instead of upload-only evidence scope.</p>
                  </div>
                  <div className="form-grid form-grid-compact">
                    <label className="field">
                      <span>Indexing Profile</span>
                      <select value={metadataProfile} onChange={event => setMetadataProfile(event.target.value)}>
                        <option value="auto">Auto</option>
                        <option value="deterministic">Deterministic</option>
                        <option value="basic">Basic</option>
                        <option value="off">Off</option>
                      </select>
                    </label>
                    <label className="checkbox-card">
                      <input
                        type="checkbox"
                        checked={indexPreview}
                        onChange={event => setIndexPreview(event.target.checked)}
                      />
                      <span className="checkbox-copy">
                        <strong>Preview only</strong>
                        <span>Inspect metadata before writing documents.</span>
                      </span>
                    </label>
                  </div>
                  <div className="inline-action-pair">
                    <ActionButton tone="primary" onClick={() => collectionFilesInputRef.current?.click()}>Upload Files</ActionButton>
                    <ActionButton tone="secondary" onClick={() => collectionFolderInputRef.current?.click()}>Upload Folder</ActionButton>
                  </div>
                </div>
              )}

              {activeCollectionAction === 'local' && (
                <div className="collection-action-panel">
                  <div className="collection-action-copy">
                    <strong>{knowledgeSourceKind === 'local_repo' ? 'Local repository' : 'Local folder'} source</strong>
                    <p>Local paths are scanned against server-side allowed roots before any files are indexed.</p>
                  </div>
                  <SegmentedControl<KnowledgeSourceKind>
                    ariaLabel="Local source type"
                    value={knowledgeSourceKind}
                    onChange={setKnowledgeSourceKind}
                    options={[
                      { value: 'local_folder', label: 'Folder' },
                      { value: 'local_repo', label: 'Repository' },
                    ]}
                  />
                  <label className="field">
                    <span>Local Paths</span>
                    <textarea
                      aria-label="Local Paths"
                      rows={5}
                      value={pathDraft}
                      onChange={event => setPathDraft(event.target.value)}
                      placeholder={'/absolute/path/to/folder\n/absolute/path/to/repository'}
                    />
                  </label>
                  <div className="form-grid form-grid-compact">
                    <label className="field">
                      <span>Include Globs</span>
                      <textarea
                        aria-label="Include Globs"
                        rows={4}
                        value={sourceIncludeGlobs}
                        onChange={event => setSourceIncludeGlobs(event.target.value)}
                        placeholder={'docs/**\n*.md'}
                      />
                    </label>
                    <label className="field">
                      <span>Exclude Globs</span>
                      <textarea
                        aria-label="Exclude Globs"
                        rows={4}
                        value={sourceExcludeGlobs}
                        onChange={event => setSourceExcludeGlobs(event.target.value)}
                        placeholder={'node_modules/**\n.git/**'}
                      />
                    </label>
                    <label className="field">
                      <span>Indexing Profile</span>
                      <select value={metadataProfile} onChange={event => setMetadataProfile(event.target.value)}>
                        <option value="auto">Auto</option>
                        <option value="deterministic">Deterministic</option>
                        <option value="basic">Basic</option>
                        <option value="off">Off</option>
                      </select>
                    </label>
                    <label className="checkbox-card">
                      <input
                        type="checkbox"
                        checked={indexPreview}
                        onChange={event => setIndexPreview(event.target.checked)}
                      />
                      <span className="checkbox-copy">
                        <strong>Preview only</strong>
                        <span>Inspect metadata before writing documents.</span>
                      </span>
                    </label>
                  </div>
                  <div className="collection-action-copy">
                    <span className="field-label">Allowed Roots</span>
                    <div className="badge-cluster">
                      {allowedSourceRoots.length > 0 ? allowedSourceRoots.slice(0, 4).map(root => (
                        <StatusBadge key={root} tone="neutral">{root}</StatusBadge>
                      )) : (
                        <StatusBadge tone="warning">No allowed roots reported</StatusBadge>
                      )}
                      {allowedSourceRoots.length > 4 && <StatusBadge tone="neutral">+{allowedSourceRoots.length - 4} more</StatusBadge>}
                    </div>
                  </div>
                  <ActionBar>
                    <ActionButton tone="secondary" onClick={() => void handleSourceScan()}>Preview Scan</ActionButton>
                    <ActionButton tone="ghost" onClick={() => void handleRegisterSource()}>Register Source</ActionButton>
                    <ActionButton tone="primary" onClick={() => void handleIndexLocalSource()}>Index Source</ActionButton>
                  </ActionBar>
                </div>
              )}

              {activeCollectionAction === 'registered' && (
                <div className="collection-action-panel">
                  <div className="collection-action-copy">
                    <strong>Refresh a registered source</strong>
                    <p>Registered folders and repositories remember their include/exclude rules and show changed, added, and deleted files before writing.</p>
                  </div>
                  {filteredRegisteredSources.length > 0 ? (
                    <EntityList
                      items={filteredRegisteredSources}
                      selectedKey={visibleRegisteredSource?.source_id ?? selectedSourceId}
                      getKey={source => source.source_id}
                      getLabel={source => source.display_name}
                      getDescription={source => `${source.source_kind} → ${source.collection_id}`}
                      getMeta={source => (
                        <>
                          <StatusBadge tone="neutral">{source.source_kind}</StatusBadge>
                          <span>{source.last_scan?.summary.supported_count ?? 0} files</span>
                        </>
                      )}
                      onSelect={source => {
                        setSelectedSourceId(source.source_id)
                        if (source.last_scan) setSourceScan(source.last_scan)
                      }}
                    />
                  ) : (
                    <EmptyState title="No registered sources" body="Register a local folder or repository from the Local Path tab, then refresh it here without retyping paths." />
                  )}

                  {visibleRegisteredSource && (
                    <div className="summary-list">
                      <div className="summary-row">
                        <span>Paths</span>
                        <strong>{visibleRegisteredSource.paths.join(', ')}</strong>
                      </div>
                      <div className="summary-row">
                        <span>Last Scan</span>
                        <strong>{visibleRegisteredSource.last_scan ? `${visibleRegisteredSource.last_scan.summary.supported_count} supported, ${visibleRegisteredSource.last_scan.summary.skipped_count} skipped` : 'Not scanned'}</strong>
                      </div>
                      <div className="summary-row">
                        <span>Last Refresh</span>
                        <strong>{formatTimestamp(asRecord(visibleRegisteredSource.last_refresh)?.updated_at)}</strong>
                      </div>
                    </div>
                  )}

                  <ActionBar>
                    <ActionButton tone="secondary" onClick={() => void handleRefreshSource(true)} disabled={!visibleRegisteredSource}>Preview Drift</ActionButton>
                    <ActionButton tone="primary" onClick={() => void handleRefreshSource(false)} disabled={!visibleRegisteredSource}>Refresh Now</ActionButton>
                    <ActionButton tone="ghost" onClick={() => void handleRefreshSource(false, true)} disabled={!visibleRegisteredSource}>Queue Refresh</ActionButton>
                  </ActionBar>

                  {selectedSourceRuns.length > 0 && (
                    <div className="timeline-list">
                      {selectedSourceRuns.slice(0, 3).map(run => (
                        <article key={run.run_id} className="timeline-item">
                          <div className="timeline-dot" aria-hidden="true" />
                          <div>
                            <strong>{humanizeKey(run.status)}</strong>
                            <p>{run.detail || run.operation}</p>
                            <span>{formatTimestamp(run.completed_at || run.updated_at || run.started_at)}</span>
                          </div>
                        </article>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {activeCollectionAction === 'sync' && (
                <div className="collection-action-panel collection-action-panel-compact">
                  <div className="collection-action-copy">
                    <strong>Sync configured KB sources</strong>
                    <p>Use this when the collection should mirror the runtime KB roots. Uploaded folders and registered sources do not require this step.</p>
                  </div>
                  <ActionButton tone="ghost" onClick={() => void handleSyncCollection()}>Sync Configured Sources</ActionButton>
                </div>
              )}

              {sourceScan && (
                <div className="preview-card collection-result-card">
                  <div className="badge-cluster">
                    <StatusBadge tone={toneForStatus(sourceScan.status)}>{humanizeKey(sourceScan.status)}</StatusBadge>
                    <StatusBadge tone="accent">{sourceScanSummary?.supported_count ?? 0} supported</StatusBadge>
                    <StatusBadge tone={sourceScanSummary?.skipped_count ? 'warning' : 'neutral'}>{sourceScanSummary?.skipped_count ?? 0} skipped</StatusBadge>
                    <StatusBadge tone={sourceScanSummary?.blocked_count ? 'danger' : 'neutral'}>{sourceScanSummary?.blocked_count ?? 0} blocked</StatusBadge>
                    <StatusBadge tone="neutral">{sourceScanSummary?.estimated_chunks ?? 0} est. chunks</StatusBadge>
                  </div>
                  <div className="summary-list">
                    <div className="summary-row">
                      <span>Source</span>
                      <strong>{humanizeKey(sourceScan.source_kind)} into {sourceScan.collection_id}</strong>
                    </div>
                    <div className="summary-row">
                      <span>Size</span>
                      <strong>{formatWholeNumber(sourceScanSummary?.total_size_bytes)} bytes</strong>
                    </div>
                    <div className="summary-row">
                      <span>Duplicates</span>
                      <strong>
                        {sourceScan.duplicate_display_paths.length > 0
                          ? shortList(sourceScan.duplicate_display_paths.slice(0, 3))
                          : (sourceScan.duplicate_filenames ?? []).length > 0
                            ? shortList((sourceScan.duplicate_filenames ?? []).slice(0, 3))
                            : 'None'}
                      </strong>
                    </div>
                  </div>
                  {sourceScan.warnings.length > 0 && (
                    <div className="inline-alert inline-alert-warning">
                      <span>Warnings</span>
                      <strong>{shortList(sourceScan.warnings.slice(0, 4))}</strong>
                    </div>
                  )}
                  {sourceScan.supported_files.length > 0 && (
                    <div className="collection-result-list">
                      {sourceScan.supported_files.slice(0, 6).map(file => (
                        <div key={`${file.source_path}-${file.display_path}`} className="meta-chip">
                          <span>{file.display_path}</span>
                          <strong>{file.filename}</strong>
                          <span>{formatWholeNumber(file.size_bytes)} bytes</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {sourceScan.skipped_files.length > 0 && (
                    <p className="muted-copy">{sourceScan.skipped_files.length} skipped file(s), including {shortList(sourceScan.skipped_files.slice(0, 3).map(file => file.display_path))}.</p>
                  )}
                </div>
              )}

              {collectionActivity ? (
                <div className="preview-card collection-result-card">
                  <div className="badge-cluster">
                    <StatusBadge tone={toneForStatus(collectionActivityRecord?.status)}>
                      {collectionActivityStatus ? humanizeKey(collectionActivityStatus) : 'Latest result'}
                    </StatusBadge>
                    <StatusBadge tone="accent">
                      {asString(collectionActivitySummary?.ingested_count, asString(collectionActivityRecord?.ingested_count, '0'))} ingested
                    </StatusBadge>
                    {'summary' in (collectionActivityRecord ?? {}) && (
                      <>
                        <StatusBadge tone="neutral">{asString(collectionActivitySummary?.resolved_count, '0')} processed</StatusBadge>
                        {Number(collectionActivitySummary?.already_indexed_count ?? 0) > 0 && (
                          <StatusBadge tone="neutral">{asString(collectionActivitySummary?.already_indexed_count, '0')} already indexed</StatusBadge>
                        )}
                        <StatusBadge tone="warning">{asString(collectionActivitySummary?.skipped_count, '0')} skipped</StatusBadge>
                        <StatusBadge tone={asString(collectionActivitySummary?.failed_count, '0') === '0' ? 'neutral' : 'danger'}>
                          {asString(collectionActivitySummary?.failed_count, '0')} failed
                        </StatusBadge>
                      </>
                    )}
                  </div>

                  <div className="summary-list">
                    <div className="summary-row">
                      <span>Collection</span>
                      <strong>{asString(collectionActivityRecord?.collection_id, selectedCollection)}</strong>
                    </div>
                    {Object.keys(collectionMetadataSummary).length > 0 && (
                      <div className="summary-row">
                        <span>Metadata</span>
                        <strong>
                          {asString(collectionMetadataSummary.document_count, '0')} docs | {asString(collectionMetadataSummary.chunk_count, '0')} chunks | {asString(collectionMetadataSummary.metadata_profile, metadataProfile)}
                        </strong>
                      </div>
                    )}
                    {'created' in (collectionActivityRecord ?? {}) && (
                      <div className="summary-row">
                        <span>Create Result</span>
                        <strong>{Boolean(collectionActivityRecord?.created) ? 'Created now' : 'Already existed'}</strong>
                      </div>
                    )}
                    {collectionActivityRecord?.deleted === true && (
                      <div className="summary-row">
                        <span>Delete Result</span>
                        <strong>Collection removed</strong>
                      </div>
                    )}
                    {asArray<string>(collectionActivityRecord?.missing_paths).length > 0 && (
                      <div className="summary-row">
                        <span>Missing Paths</span>
                        <strong>{asArray<string>(collectionActivityRecord?.missing_paths).join(', ')}</strong>
                      </div>
                    )}
                  </div>

                  {collectionActivityExceptions.length > 0 && (
                    <div className="collection-result-list">
                      {collectionActivityExceptions.slice(0, 6).map(item => {
                        const displayPath = asString(item.display_path, asString(item.filename, 'Unknown file'))
                        return (
                          <div key={`${displayPath}-${asString(item.outcome)}-${asString(item.error)}`} className="meta-chip">
                            <span>{displayPath}</span>
                            <StatusBadge tone={collectionOutcomeTone(asString(item.outcome))}>
                              {humanizeKey(asString(item.outcome, 'unknown'))}
                            </StatusBadge>
                            {asString(item.error) && <span>{asString(item.error)}</span>}
                          </div>
                        )
                      })}
                      {collectionActivityExceptions.length > 6 && (
                        <p className="muted-copy">{collectionActivityExceptions.length - 6} more file result(s) are available in the API payload.</p>
                      )}
                    </div>
                  )}

                  {selectedCollectionCanBuildGraph && !Boolean(collectionActivityRecord?.preview) && (
                    <ActionBar>
                      <ActionButton
                        tone="primary"
                        onClick={() => openGraphBuilderForCollection(normalizeCollectionId(selectedCollection || collectionDraft))}
                      >
                        Build Graph From This Corpus
                      </ActionButton>
                    </ActionBar>
                  )}
                </div>
              ) : (
                <EmptyState title="No recent collection action" body="Create a collection or run one ingest action here, and the latest result will stay visible above the document list." />
              )}
            </SurfaceCard>

            <SurfaceCard title="Documents" subtitle="Search by title, display path, or source type once the collection has content.">
              <div className="form-grid form-grid-compact collection-document-filters">
                <label className="field">
                  <span>Search Documents</span>
                  <input
                    aria-label="Search Documents"
                    value={documentSearch}
                    onChange={event => setDocumentSearch(event.target.value)}
                    placeholder="Search by title or display path"
                  />
                </label>
                <label className="field">
                  <span>Source Filter</span>
                  <select
                    aria-label="Source Filter"
                    value={documentSourceFilter}
                    onChange={event => setDocumentSourceFilter(event.target.value)}
                  >
                    <option value="all">All sources</option>
                    {documentSourceTypes.map(sourceType => (
                      <option key={sourceType} value={sourceType}>
                        {sourceType}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <div className="badge-cluster">
                <StatusBadge tone="neutral">{filteredCollectionDocs.length} shown</StatusBadge>
                <StatusBadge tone="accent">{collectionDocs.length} total</StatusBadge>
              </div>

              {documentSourceTypes.length > 0 && (
                <div className="doc-filter-chips">
                  <button
                    type="button"
                    className="filter-chip-btn"
                    onClick={() => setDocumentSourceFilter('all')}
                    aria-pressed={documentSourceFilter === 'all'}
                  >
                    <FilterChip label="All sources" tone={documentSourceFilter === 'all' ? 'accent' : 'neutral'} />
                  </button>
                  {documentSourceTypes.map(sourceType => {
                    const count = collectionDocs.filter(d => asString(d.source_type) === sourceType).length
                    const active = documentSourceFilter === sourceType
                    return (
                      <button
                        key={sourceType}
                        type="button"
                        className="filter-chip-btn"
                        onClick={() => setDocumentSourceFilter(active ? 'all' : sourceType)}
                        aria-pressed={active}
                      >
                        <FilterChip label={sourceType} count={count} tone={active ? 'accent' : 'neutral'} />
                      </button>
                    )
                  })}
                </div>
              )}

              {filteredCollectionDocs.length > 0 ? (
                <EntityList
                  items={filteredCollectionDocs}
                  selectedKey={selectedDoc}
                  getKey={doc => asString(doc.doc_id)}
                  getLabel={doc => asString(doc.title)}
                  getDescription={doc => asString(doc.source_display_path || doc.source_path)}
                  getMeta={doc => (
                    <>
                      <StatusBadge tone="neutral">{asString(doc.source_type, 'source')}</StatusBadge>
                      <span>{asString(doc.num_chunks, '0')} chunks</span>
                    </>
                  )}
                  onSelect={doc => setSelectedDoc(asString(doc.doc_id))}
                />
              ) : (
                <EmptyState title="No documents match yet" body="Create the collection and ingest host paths or sync configured KB content to populate the document workspace." />
              )}
            </SurfaceCard>

            <CollapsibleSurfaceCard
              title="Document Viewer"
              subtitle="Extracted content stays readable, while the metadata tab keeps the logical display path and raw source path together."
              actions={(
                <>
                  <ActionButton tone="ghost" onClick={() => void handleReindexDocument()} disabled={!selectedDoc}>Reindex</ActionButton>
                  <ActionButton
                    tone="destructive"
                    onClick={() => askConfirm({
                      title: 'Delete this document?',
                      description: `Permanently remove "${asString(documentRecord?.title) || selectedDoc}" from this collection.`,
                      confirmLabel: 'Delete',
                      run: handleDeleteDocument,
                    })}
                    disabled={!selectedDoc}
                  >
                    Delete
                  </ActionButton>
                </>
              )}
              open={collectionViewerOpen}
              onToggle={() => setCollectionViewerOpen(open => !open)}
            >
              {docDetail && documentRecord ? (
                <>
                  <div className="summary-list">
                    <div className="summary-row">
                      <span>Title</span>
                      <strong>{asString(documentRecord.title, 'Untitled')}</strong>
                    </div>
                    <div className="summary-row">
                      <span>Display Path</span>
                      <strong>{asString(documentRecord.source_display_path || documentRecord.source_path, 'Unknown')}</strong>
                    </div>
                    <div className="summary-row">
                      <span>Source</span>
                      <strong>{asString(documentRecord.source_type, 'unknown')}</strong>
                    </div>
                    <div className="summary-row">
                      <span>Chunk Count</span>
                      <strong>{asString((docDetail.extracted_content as Record<string, unknown> | undefined)?.chunk_count, '0')}</strong>
                    </div>
                  </div>
                  <DetailTabs
                    key={selectedDoc}
                    tabs={[
                      {
                        id: 'extracted',
                        label: 'Extracted',
                        content: <div className="code-panel">{extractedContent || 'No extracted content available.'}</div>,
                      },
                      {
                        id: 'raw',
                        label: 'Raw',
                        content: <div className="code-panel">{rawContent || 'No raw source is available for this document.'}</div>,
                      },
                      {
                        id: 'metadata',
                        label: 'Metadata',
                        content: (
                          <div className="summary-list">
                            <div className="summary-row">
                              <span>Display Path</span>
                              <strong>{asString(documentRecord.source_display_path || documentRecord.source_path, 'Unknown')}</strong>
                            </div>
                            <div className="summary-row">
                              <span>Source Path</span>
                              <strong>{asString(documentRecord.source_path, 'Unknown')}</strong>
                            </div>
                            <div className="summary-row">
                              <span>File Type</span>
                              <strong>{asString(documentRecord.file_type, 'Unknown')}</strong>
                            </div>
                            <div className="summary-row">
                              <span>Structure Type</span>
                              <strong>{asString(documentRecord.doc_structure_type, 'Unknown')}</strong>
                            </div>
                            {Object.keys(docMetadataSummary).length > 0 && (
                              <div className="summary-row">
                                <span>Indexing Profile</span>
                                <strong>{asString(docMetadataSummary.metadata_profile, 'auto')}</strong>
                              </div>
                            )}
                            {Object.keys(docMetadataSummary).length > 0 && (
                              <div className="summary-row">
                                <span>Tags</span>
                                <strong>{asArray<string>(docMetadataSummary.tags).slice(0, 6).join(', ') || 'None'}</strong>
                              </div>
                            )}
                            <div className="summary-row">
                              <span>Collection</span>
                              <strong>{asString(documentRecord.collection_id, 'Unknown')}</strong>
                            </div>
                          </div>
                        ),
                      },
                    ]}
                  />
                  <JsonInspector label="Technical details" value={docDetail} />
                </>
              ) : (
                <EmptyState title="Choose a document" body="Selecting a document reveals extracted content, raw source, and the collection-aware metadata needed for reindex and troubleshooting." />
              )}
            </CollapsibleSurfaceCard>

            <SurfaceCard
              title="Collection Health"
              subtitle="Keep duplicates, drift, and missing files visible while you ingest or repair the current namespace."
              actions={(
                <ActionBar>
                  <ActionButton tone="ghost" onClick={() => selectedCollection && void refreshCollectionHealth(selectedCollection)} disabled={!selectedCollection}>Refresh Health</ActionButton>
                  <ActionButton tone="primary" onClick={() => void handleRepairCollection()} disabled={!selectedCollection && !collectionDraft}>Repair Collection</ActionButton>
                </ActionBar>
              )}
            >
              {collectionHealth ? (
                <>
                  <div className="badge-cluster">
                    <StatusBadge tone={collectionHealth.status === 'ready' ? 'ok' : 'warning'}>
                      {collectionHealth.status === 'ready' ? 'Healthy' : humanizeKey(collectionHealth.reason)}
                    </StatusBadge>
                    <StatusBadge tone={duplicateGroups.length > 0 ? 'danger' : 'ok'}>
                      {duplicateGroups.length} duplicate group{duplicateGroups.length === 1 ? '' : 's'}
                    </StatusBadge>
                    <StatusBadge tone={driftedGroups.length > 0 ? 'warning' : 'ok'}>
                      {driftedGroups.length} drifted group{driftedGroups.length === 1 ? '' : 's'}
                    </StatusBadge>
                  </div>
                  <div className="stats-grid">
                    <StatCard label="Active Docs" value={collectionHealth.active_doc_count} caption="Winning indexed copies used for retrieval." />
                    <StatCard label="Indexed Docs" value={collectionHealth.indexed_doc_count} caption="All document rows currently stored." />
                    <StatCard label="Missing Files" value={collectionHealth.missing_sources.length} caption="Configured KB files not present here." />
                    <StatCard label="Suggested Fix" value={collectionHealth.reason === 'ready' ? 'None' : 'Repair'} caption={collectionHealth.suggested_fix || 'No repair needed.'} />
                  </div>
                  {collectionHealth.source_groups.filter(group => group.status !== 'healthy').length > 0 ? (
                    <EntityList
                      items={collectionHealth.source_groups.filter(group => group.status !== 'healthy')}
                      selectedKey=""
                      onSelect={() => {}}
                      getKey={group => asString(group.source_identity)}
                      getLabel={group => asString(group.title, asString(group.source_identity))}
                      getDescription={group => asString(group.configured_source_path || group.active_source_path)}
                      getMeta={group => (
                        <>
                          <StatusBadge tone={asString(group.status) === 'duplicate' ? 'danger' : 'warning'}>
                            {humanizeKey(asString(group.status))}
                          </StatusBadge>
                          {asArray<string>(group.duplicate_doc_ids).length > 0 && <span>{asArray<string>(group.duplicate_doc_ids).length} stale</span>}
                        </>
                      )}
                    />
                  ) : (
                    <EmptyState title="No duplicate or drift issues" body="This collection currently has clean coverage for the sources it tracks." />
                  )}
                  <JsonInspector label="Health payload" value={collectionHealth} />
                </>
              ) : (
                <EmptyState title="No collection selected" body="Choose a collection to inspect health, then repair duplicates or drift without leaving the workspace." />
              )}
            </SurfaceCard>

            <CollapsibleSurfaceCard
              title="Collection Inspector"
              subtitle="Expand the technical view only when you need tables, dimensions, provider details, or graph metadata."
              open={collectionInspectorOpen}
              onToggle={() => setCollectionInspectorOpen(open => !open)}
            >
              {selectedCollectionMeta ? (
                <>
                  <div className="stats-grid">
                    <StatCard label="Docs" value={selectedCollectionMeta.document_count} caption="Indexed documents in this namespace." />
                    <StatCard label="Graphs" value={selectedCollectionMeta.graph_count} caption="Graph projects attached to this collection." />
                    <StatCard label="Embed Dim" value={selectedCollectionStorage.configured_embedding_dim || 'n/a'} caption="Configured embedding dimension." />
                    <StatCard label="Provider" value={selectedCollectionStorage.embeddings_provider || 'n/a'} caption={selectedCollectionStorage.embedding_model || 'No embedding model configured.'} />
                  </div>
                  <div className="summary-list">
                    <div className="summary-row">
                      <span>Vector Backend</span>
                      <strong>{selectedCollectionStorage.vector_store_backend || 'Unknown'}</strong>
                    </div>
                    <div className="summary-row">
                      <span>Tables</span>
                      <strong>{selectedCollectionStorage.tables.join(', ')}</strong>
                    </div>
                    <div className="summary-row">
                      <span>Embedding Model</span>
                      <strong>{selectedCollectionStorage.embedding_model || 'Unknown'}</strong>
                    </div>
                    {selectedCollectionStorage.graph_embedding_model && (
                      <div className="summary-row">
                        <span>Graph Embed Model</span>
                        <strong>{selectedCollectionStorage.graph_embedding_model}</strong>
                      </div>
                    )}
                    <div className="summary-row">
                      <span>Actual Vector Dims</span>
                      <strong>
                        {Object.keys(selectedCollectionStorage.actual_embedding_dims).length > 0
                          ? Object.entries(selectedCollectionStorage.actual_embedding_dims).map(([tableName, dim]) => `${tableName}: ${dim}`).join(', ')
                          : 'Not available from this runtime connection'}
                      </strong>
                    </div>
                    <div className="summary-row">
                      <span>Source Mix</span>
                      <strong>
                        {Object.keys(selectedCollectionMeta.source_type_counts).length > 0
                          ? Object.entries(selectedCollectionMeta.source_type_counts).map(([sourceType, count]) => `${sourceType}: ${count}`).join(', ')
                          : 'No documents yet'}
                      </strong>
                    </div>
                    <div className="summary-row">
                      <span>Maintenance Policy</span>
                      <strong>{humanizeKey(selectedCollectionMeta.maintenance_policy || 'indexed_documents')}</strong>
                    </div>
                    <div className="summary-row">
                      <span>Recent Ingest</span>
                      <strong>{formatTimestamp(selectedCollectionMeta.latest_ingested_at || selectedCollectionMeta.updated_at)}</strong>
                    </div>
                    <div className="summary-row">
                      <span>RBAC Access</span>
                      <strong>{selectedCollectionGrantSummary.principalNames.length > 0 ? shortList(selectedCollectionGrantSummary.principalNames) : 'No current grants'}</strong>
                    </div>
                  </div>
                  {selectedCollectionStorage.mismatch_warnings.length > 0 && (
                    <div className="inline-alert inline-alert-warning">
                      <strong>Vector dimension mismatch detected.</strong>
                      <span>{selectedCollectionStorage.mismatch_warnings.join(' ')}</span>
                    </div>
                  )}
                  <JsonInspector label="Collection payload" value={selectedCollectionMeta} />
                </>
              ) : (
                <EmptyState title="Select a collection" body="The inspector will show table names, embedding settings, graph usage, and readiness as soon as a collection is selected or created." />
              )}
            </CollapsibleSurfaceCard>
          </div>
        </div>
      )}

      {active === 'uploads' && (
        <div className="workspace-grid workspace-grid-collections">
          <SurfaceCard className="selection-rail collection-rail" title="Uploaded Files" subtitle="Chat and ad hoc uploads are kept separate from knowledge collections.">
            <ResourceSearch value={uploadSearch} onChange={setUploadSearch} placeholder="Search uploads" />
            <div className="badge-cluster">
              <StatusBadge tone="accent">{uploadedFiles.length} uploads</StatusBadge>
              <StatusBadge tone="neutral">{filteredUploadedFiles.length} shown</StatusBadge>
            </div>
            {filteredUploadedFiles.length > 0 ? (
              <EntityList
                variant="rail"
                items={filteredUploadedFiles}
                selectedKey={selectedUploadDoc}
                getKey={file => file.doc_id}
                getLabel={file => file.title}
                getDescription={file => file.source_display_path || file.source_path}
                getMeta={file => (
                  <>
                    <StatusBadge tone="neutral">{file.collection_id}</StatusBadge>
                    <span>{file.num_chunks} chunks</span>
                  </>
                )}
                onSelect={file => setSelectedUploadDoc(file.doc_id)}
              />
            ) : (
              <EmptyState title="No uploaded files" body="Files uploaded through chat or this workspace will appear here instead of in knowledge collections." />
            )}
          </SurfaceCard>

          <div className="content-stack collection-main-stack">
            <SurfaceCard
              className="collection-workspace-card"
              title="Add Uploaded Files"
              subtitle="Index files as upload-scoped evidence without adding them to a knowledge collection."
            >
              <input
                ref={uploadFilesInputRef}
                aria-label="Upload Files Input"
                type="file"
                multiple
                className="visually-hidden"
                tabIndex={-1}
                onChange={event => {
                  void handleUploadedFilesUpload(event.target.files)
                  event.currentTarget.value = ''
                }}
              />
              <input
                ref={node => {
                  uploadFolderInputRef.current = node
                  if (node) {
                    node.setAttribute('webkitdirectory', '')
                    node.setAttribute('directory', '')
                  }
                }}
                aria-label="Upload Folder Input"
                type="file"
                multiple
                className="visually-hidden"
                tabIndex={-1}
                onChange={event => {
                  void handleUploadedFilesUpload(event.target.files)
                  event.currentTarget.value = ''
                }}
              />

              <div className="badge-cluster">
                <StatusBadge tone="accent">Upload scope</StatusBadge>
                {Boolean(uploadActivityStatus) && (
                  <StatusBadge tone={toneForStatus(uploadActivityStatus)}>
                    {humanizeKey(uploadActivityStatus)}
                  </StatusBadge>
                )}
              </div>

              <div className="supported-type-strip" aria-label="Supported upload document types">
                <span>Supported Types</span>
                {SUPPORTED_DOCUMENT_TYPES.map(type => (
                  <StatusBadge key={type} tone="neutral">{type}</StatusBadge>
                ))}
              </div>

              <div className="collection-action-panel collection-action-panel-compact">
                <div className="collection-action-copy">
                  <strong>Upload evidence files</strong>
                  <p>Uploaded files stay in upload scope and no longer increase knowledge collection document counts.</p>
                </div>
                <label className="field">
                  <span>Indexing Profile</span>
                  <select value={metadataProfile} onChange={event => setMetadataProfile(event.target.value)}>
                    <option value="auto">Auto</option>
                    <option value="deterministic">Deterministic</option>
                    <option value="basic">Basic</option>
                    <option value="off">Off</option>
                  </select>
                </label>
                <label className="checkbox-card">
                  <input
                    type="checkbox"
                    checked={indexPreview}
                    onChange={event => setIndexPreview(event.target.checked)}
                  />
                  <span className="checkbox-copy">
                    <strong>Preview only</strong>
                    <span>Inspect metadata before writing documents.</span>
                  </span>
                </label>
                <div className="inline-action-pair">
                  <ActionButton tone="secondary" onClick={() => uploadFilesInputRef.current?.click()}>Upload Files</ActionButton>
                  <ActionButton tone="ghost" onClick={() => uploadFolderInputRef.current?.click()}>Upload Folder</ActionButton>
                </div>
              </div>

              {uploadActivity ? (
                <div className="preview-card collection-result-card">
                  <div className="badge-cluster">
                    <StatusBadge tone={toneForStatus(uploadActivityRecord?.status)}>
                      {uploadActivityStatus ? humanizeKey(uploadActivityStatus) : 'Latest result'}
                    </StatusBadge>
                    <StatusBadge tone="accent">
                      {asString(uploadActivitySummary?.ingested_count, asString(uploadActivityRecord?.ingested_count, '0'))} ingested
                    </StatusBadge>
                    {'summary' in (uploadActivityRecord ?? {}) && (
                      <>
                        <StatusBadge tone="neutral">{asString(uploadActivitySummary?.resolved_count, '0')} processed</StatusBadge>
                        {Number(uploadActivitySummary?.already_indexed_count ?? 0) > 0 && (
                          <StatusBadge tone="neutral">{asString(uploadActivitySummary?.already_indexed_count, '0')} already indexed</StatusBadge>
                        )}
                        <StatusBadge tone="warning">{asString(uploadActivitySummary?.skipped_count, '0')} skipped</StatusBadge>
                        <StatusBadge tone={asString(uploadActivitySummary?.failed_count, '0') === '0' ? 'neutral' : 'danger'}>
                          {asString(uploadActivitySummary?.failed_count, '0')} failed
                        </StatusBadge>
                      </>
                    )}
                  </div>

                  <div className="summary-list">
                    <div className="summary-row">
                      <span>Upload Collection</span>
                      <strong>{asString(uploadActivityRecord?.collection_id, 'control-panel-uploads')}</strong>
                    </div>
                    {Object.keys(uploadMetadataSummary).length > 0 && (
                      <div className="summary-row">
                        <span>Metadata</span>
                        <strong>
                          {asString(uploadMetadataSummary.document_count, '0')} docs | {asString(uploadMetadataSummary.chunk_count, '0')} chunks | {asString(uploadMetadataSummary.metadata_profile, metadataProfile)}
                        </strong>
                      </div>
                    )}
                  </div>

                  {uploadActivityExceptions.length > 0 && (
                    <div className="collection-result-list">
                      {uploadActivityExceptions.slice(0, 6).map(item => {
                        const displayPath = asString(item.display_path, asString(item.filename, 'Unknown file'))
                        return (
                          <div key={`${displayPath}-${asString(item.outcome)}-${asString(item.error)}`} className="meta-chip">
                            <span>{displayPath}</span>
                            <StatusBadge tone={collectionOutcomeTone(asString(item.outcome))}>
                              {humanizeKey(asString(item.outcome, 'unknown'))}
                            </StatusBadge>
                            {asString(item.error) && <span>{asString(item.error)}</span>}
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>
              ) : (
                <EmptyState title="No recent upload action" body="Upload files here and the latest result will stay visible above the viewer." />
              )}
            </SurfaceCard>

            <CollapsibleSurfaceCard
              title="Uploaded File Viewer"
              subtitle="Inspect extracted content, source metadata, and upload collection IDs without mixing these files into KB collections."
              actions={(
                <>
                  <ActionButton tone="ghost" onClick={() => void handleReindexUploadedFile()} disabled={!selectedUploadDoc}>Reindex</ActionButton>
                  <ActionButton
                    tone="destructive"
                    onClick={() => askConfirm({
                      title: 'Delete this uploaded file?',
                      description: `Permanently remove "${asString(uploadDocumentRecord?.title) || selectedUploadDoc}".`,
                      confirmLabel: 'Delete',
                      run: handleDeleteUploadedFile,
                    })}
                    disabled={!selectedUploadDoc}
                  >
                    Delete
                  </ActionButton>
                </>
              )}
              open={uploadViewerOpen}
              onToggle={() => setUploadViewerOpen(open => !open)}
            >
              {uploadDocDetail && uploadDocumentRecord ? (
                <>
                  <div className="summary-list">
                    <div className="summary-row">
                      <span>Title</span>
                      <strong>{asString(uploadDocumentRecord.title, 'Untitled')}</strong>
                    </div>
                    <div className="summary-row">
                      <span>Display Path</span>
                      <strong>{asString(uploadDocumentRecord.source_display_path || uploadDocumentRecord.source_path, 'Unknown')}</strong>
                    </div>
                    <div className="summary-row">
                      <span>Upload Collection</span>
                      <strong>{asString(uploadDocumentRecord.collection_id, 'Unknown')}</strong>
                    </div>
                    <div className="summary-row">
                      <span>Chunk Count</span>
                      <strong>{asString((uploadDocDetail.extracted_content as Record<string, unknown> | undefined)?.chunk_count, '0')}</strong>
                    </div>
                  </div>
                  <DetailTabs
                    key={selectedUploadDoc}
                    tabs={[
                      {
                        id: 'extracted',
                        label: 'Extracted',
                        content: <div className="code-panel">{uploadExtractedContent || 'No extracted content available.'}</div>,
                      },
                      {
                        id: 'raw',
                        label: 'Raw',
                        content: <div className="code-panel">{uploadRawContent || 'No raw source is available for this uploaded file.'}</div>,
                      },
                      {
                        id: 'metadata',
                        label: 'Metadata',
                        content: (
                          <div className="summary-list">
                            <div className="summary-row">
                              <span>Source Path</span>
                              <strong>{asString(uploadDocumentRecord.source_path, 'Unknown')}</strong>
                            </div>
                            <div className="summary-row">
                              <span>File Type</span>
                              <strong>{asString(uploadDocumentRecord.file_type, 'Unknown')}</strong>
                            </div>
                            <div className="summary-row">
                              <span>Structure Type</span>
                              <strong>{asString(uploadDocumentRecord.doc_structure_type, 'Unknown')}</strong>
                            </div>
                            {Object.keys(uploadDocMetadataSummary).length > 0 && (
                              <div className="summary-row">
                                <span>Indexing Profile</span>
                                <strong>{asString(uploadDocMetadataSummary.metadata_profile, 'auto')}</strong>
                              </div>
                            )}
                            {Object.keys(uploadDocMetadataSummary).length > 0 && (
                              <div className="summary-row">
                                <span>Tags</span>
                                <strong>{asArray<string>(uploadDocMetadataSummary.tags).slice(0, 6).join(', ') || 'None'}</strong>
                              </div>
                            )}
                            <div className="summary-row">
                              <span>Source</span>
                              <strong>{asString(uploadDocumentRecord.source_type, 'upload')}</strong>
                            </div>
                          </div>
                        ),
                      },
                    ]}
                  />
                  <JsonInspector label="Technical details" value={uploadDocDetail} />
                </>
              ) : (
                <EmptyState title="Choose an uploaded file" body="Selecting an upload reveals extracted content, raw source, and metadata for troubleshooting." />
              )}
            </CollapsibleSurfaceCard>
          </div>
        </div>
      )}

      {active === 'skills' && (
        skillsTab === 'editor' ? (
          <div className="studio-layout skill-editor-layout">
            <SurfaceCard
              className="selection-rail skill-rail"
              title="Skill Library"
              subtitle="Browse existing skills or start a new one without losing access to preview and status controls."
            >
              <div className="rail-actions">
                <ActionButton tone="secondary" onClick={startNewSkillDraft}>New Skill</ActionButton>
              </div>
              <ResourceSearch value={skillSearch} onChange={setSkillSearch} placeholder="Search skills" />
              {skills.length > 0 ? (
                <EntityList
                  variant="rail"
                  items={filteredSkills}
                  selectedKey={selectedSkill}
                  getKey={skill => asString(skill.skill_id)}
                  getLabel={skill => asString(skill.name, asString(skill.skill_id))}
                  getDescription={skill => asString(skill.skill_id)}
                  getMeta={skill => (
                    <StatusBadge tone={toneForStatus(skill.status)}>{asString(skill.status, 'unknown')}</StatusBadge>
                  )}
                  onSelect={skill => {
                    setCreatingSkill(false)
                    setSkillActionDetail(null)
                    setSelectedSkill(asString(skill.skill_id))
                  }}
                />
              ) : (
                <EmptyState title="No skills yet" body="Create a new skill to start building reusable workflows for the agent runtime." />
              )}
            </SurfaceCard>

            <div className="content-stack studio-sidebar">
              <CollapsibleSurfaceCard
                title="Skill Status"
                subtitle="Metadata and activation posture for the selected skill stay here so the editor can breathe."
                open={skillSummaryOpen}
                onToggle={() => setSkillSummaryOpen(open => !open)}
              >
                {creatingSkill || skillDetail ? (
                  <>
                    <div className="badge-cluster">
                      {selectedSkill && <StatusBadge tone="neutral">{selectedSkill}</StatusBadge>}
                      {!creatingSkill && <StatusBadge tone={toneForStatus(selectedSkillStatus)}>{selectedSkillStatus}</StatusBadge>}
                      {!creatingSkill && <StatusBadge tone={Boolean(skillDetail?.enabled) ? 'ok' : 'danger'}>
                        {boolLabel(Boolean(skillDetail?.enabled), 'Enabled', 'Disabled')}
                      </StatusBadge>}
                      {!creatingSkill && skillDependencyValidation && (
                        <StatusBadge tone={toneForStatus(skillDependencyValidation.dependency_state)}>
                          {humanizeKey(asString(skillDependencyValidation.dependency_state, 'healthy'))}
                        </StatusBadge>
                      )}
                      {!creatingSkill && skillHealth && (() => {
                        const raw = asString(skillHealth.review_status, 'insufficient_data')
                        const help = statusHelp(raw)
                        const badge = (
                          <StatusBadge tone={toneForStatus(skillHealth.review_status)}>
                            {humanizeKey(raw)}
                          </StatusBadge>
                        )
                        return help ? <Tooltip content={help}>{badge}</Tooltip> : badge
                      })()}
                    </div>
                    {!creatingSkill && (
                      <>
                        <div className="summary-list">
                          <div className="summary-row">
                            <span>Name</span>
                            <strong>{asString(skillDetail?.name, 'Unnamed Skill')}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Status</span>
                            <strong>{selectedSkillStatus}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Family</span>
                            <strong>{asString(skillDetail?.version_parent, asString(skillDetail?.skill_id))}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Depends On</span>
                            <strong>{shortList(asArray<string>(skillDependencyValidation?.depends_on_skills))}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Dependency State</span>
                            <strong>{humanizeKey(asString(skillDependencyValidation?.dependency_state, 'healthy'))}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Review Status</span>
                            <strong>{humanizeKey(asString(skillHealth?.review_status, 'insufficient_data'))}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Success Rate</span>
                            <strong>{formatPercent(skillHealth?.success_rate)}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Scored Uses</span>
                            <strong>{asNumber(skillHealth?.scored_uses) ?? 0}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Last Scored</span>
                            <strong>{formatTimestamp(skillHealth?.last_scored_at)}</strong>
                          </div>
                          <div className="summary-row">
                            <span>RBAC Access</span>
                            <strong>{selectedSkillGrantSummary.principalNames.length > 0 ? shortList(selectedSkillGrantSummary.principalNames) : 'No current grants'}</strong>
                          </div>
                        </div>
                        {(asArray<string>(skillDependencyValidation?.missing_dependencies).length > 0
                          || asArray<unknown>(skillDependencyValidation?.cycles).length > 0
                          || asArray<Record<string, unknown>>(skillDependencyValidation?.blocked_dependents).length > 0
                          || asArray<string>(skillDependencyValidation?.warnings).length > 0) && (
                          <div className="field-stack">
                            <strong>Dependency blockers</strong>
                            {asArray<string>(skillDependencyValidation?.missing_dependencies).length > 0 && (
                              <p>Missing families: {shortList(asArray<string>(skillDependencyValidation?.missing_dependencies))}</p>
                            )}
                            {asArray<unknown>(skillDependencyValidation?.cycles).length > 0 && (
                              <p>
                                Cycles: {asArray<Array<string>>(skillDependencyValidation?.cycles)
                                  .map(cycle => cycle.join(' → '))
                                  .join('; ')}
                              </p>
                            )}
                            {asArray<Record<string, unknown>>(skillDependencyValidation?.blocked_dependents).length > 0 && (
                              <p>
                                Impacted dependents: {asArray<Record<string, unknown>>(skillDependencyValidation?.blocked_dependents)
                                  .map(item => asString(item.name, asString(item.skill_family_id)))
                                  .join(', ')}
                              </p>
                            )}
                            {asArray<string>(skillDependencyValidation?.warnings).map(warning => (
                              <p key={warning}>{warning}</p>
                            ))}
                          </div>
                        )}
                        {skillActionValidation && (
                          <div className="field-stack">
                            <strong>{asString(skillActionDetail?.message, 'Status change blocked')}</strong>
                            {asArray<string>(skillActionValidation.missing_dependencies).length > 0 && (
                              <p>Missing families: {shortList(asArray<string>(skillActionValidation.missing_dependencies))}</p>
                            )}
                            {asArray<unknown>(skillActionValidation.cycles).length > 0 && (
                              <p>
                                Cycles: {asArray<Array<string>>(skillActionValidation.cycles)
                                  .map(cycle => cycle.join(' → '))
                                  .join('; ')}
                              </p>
                            )}
                            {asArray<Record<string, unknown>>(skillActionValidation.blocked_dependents).length > 0 && (
                              <p>
                                Impacted dependents: {asArray<Record<string, unknown>>(skillActionValidation.blocked_dependents)
                                  .map(item => asString(item.name, asString(item.skill_family_id)))
                                  .join(', ')}
                              </p>
                            )}
                          </div>
                        )}
                      </>
                    )}
                  </>
                ) : (
                  <EmptyState title="No skill selected" body="Choose a skill from the library to review its current status and metadata." />
                )}
              </CollapsibleSurfaceCard>

              <SurfaceCard className="editor-pane skill-editor-pane" title="Skill Editor" subtitle="Edit markdown, then create or update the selected skill. Activation status is surfaced separately for clarity.">
                {creatingSkill || skillDetail ? (
                  <>
                    <label className="field">
                      <span>Skill Markdown</span>
                      <textarea
                        rows={20}
                        value={skillEditor}
                        onChange={event => setSkillEditor(event.target.value)}
                      />
                    </label>
                    <ActionBar>
                      <ActionButton tone="primary" onClick={() => void handleSkillSave()}>
                        {selectedSkill ? 'Update Skill' : 'Create Skill'}
                      </ActionButton>
                      <ActionButton tone="ghost" onClick={() => void handleSkillStatus('active')} disabled={!selectedSkill}>Activate</ActionButton>
                      <ActionButton tone="destructive" onClick={() => void handleSkillStatus('archived')} disabled={!selectedSkill}>Deactivate</ActionButton>
                    </ActionBar>
                    {skillDetail && <JsonInspector label="Technical details" value={skillDetail} />}
                  </>
                ) : (
                  <EmptyState title="Select or create a skill" body="Choose an existing skill from the library or use New Skill to open a fresh markdown draft." />
                )}
              </SurfaceCard>
            </div>
          </div>
        ) : (
          <div className="content-stack">
            <SurfaceCard className="preview-pane skill-preview-pane" title="Skill Preview" subtitle="Preview the most relevant active skills for a query before attaching or editing anything.">
              <label className="field">
                <span>Preview Query</span>
                <input
                  aria-label="Preview Query"
                  value={skillPreviewQuery}
                  onChange={event => setSkillPreviewQuery(event.target.value)}
                />
              </label>
              <ActionBar>
                <ActionButton tone="secondary" onClick={() => void handleSkillPreview()}>Preview Match</ActionButton>
              </ActionBar>
              {skillPreviewResult ? (
                <>
                  <div className="field-stack">
                    {asArray<Record<string, unknown>>(skillPreviewResult.matches).map(match => (
                      <div key={asString(match.skill_id)} className="preview-card">
                        <div className="tool-card-head">
                          <strong>{asString(match.name, asString(match.skill_id))}</strong>
                          <StatusBadge tone="accent">{asNumber(match.score)?.toFixed(2) ?? '0.00'}</StatusBadge>
                        </div>
                        <p>{asString(match.skill_id)}</p>
                        <span>{asString(match.agent_scope, 'general')}</span>
                      </div>
                    ))}
                  </div>
                  <JsonInspector label="Technical details" value={skillPreviewResult} />
                </>
              ) : (
                <EmptyState title="No preview yet" body="Enter a query to see the highest-ranked active skills for the current agent scope." />
              )}
            </SurfaceCard>
          </div>
        )
      )}

      {active === 'graphs' && (
        graphsTab === 'workspace' ? (
          <div className="studio-layout graph-studio">
            <SurfaceCard
              className="selection-rail graph-rail"
              title="Named Graphs"
              subtitle="Admin-managed graphs are tenant-visible once built. Drafts stay here until you validate and index them."
            >
              <ActionBar>
                <ActionButton tone="secondary" onClick={() => openIngestionWizard(graphCollectionId)}>
                  Ingestion Wizard
                </ActionButton>
                <ActionButton tone="secondary" onClick={() => {
                  setSelectedGraph('')
                  setGraphDetail(null)
                  setGraphValidation(null)
                  setGraphRuns([])
                  setGraphProgress(null)
                  setGraphDraftId('')
                  setGraphDisplayNameDraft('')
                  setGraphPromptDraft('{}')
                  setGraphConfigDraft('{}')
                  setGraphSkillIdsDraft('')
                  setGraphSkillOverlayDraft('')
                  setGraphSelectedDocIds([])
                  setGraphTuneGuidance('')
                  setGraphTuneTargets(['extract_graph.txt'])
                  setGraphTuneResult(null)
                  setGraphTuneSelectedPrompts([])
                  setGraphAssistantPreflight(null)
                  setGraphSmokeResult(null)
                }}>
                  New Graph
                </ActionButton>
              </ActionBar>
              <ResourceSearch value={graphSearch} onChange={setGraphSearch} placeholder="Search graphs" />
              {graphs.length > 0 ? (
                <EntityList
                  variant="rail"
                  items={filteredGraphs}
                  selectedKey={selectedGraph}
                  getKey={graph => graph.graph_id}
                  getLabel={graph => graph.display_name || graph.graph_id}
                  getDescription={graph => graph.graph_id}
                  getMeta={graph => (
                    <>
                      <StatusBadge tone={toneForStatus(graph.status)}>{graph.status}</StatusBadge>
                      {graph.query_ready && <StatusBadge tone="ok">Query Ready</StatusBadge>}
                    </>
                  )}
                  onSelect={graph => {
                    setSelectedGraph(graph.graph_id)
                    setGraphTuneResult(null)
                    setGraphTuneSelectedPrompts([])
                  }}
                />
              ) : (
                <EmptyState title="No graphs yet" body="Create a graph draft here, then validate and build it once the source collection is ready." />
              )}
            </SurfaceCard>

            <div className="content-stack">
              <SurfaceCard
                title="Graph Workspace"
                subtitle="Choose a corpus, generate sensible GraphRAG defaults, run a friendly preflight, then build and smoke-test the graph."
              >
                <div className="form-grid form-grid-compact">
                  <label className="field">
                    <span>Display Name</span>
                    <input
                      aria-label="Graph Display Name"
                      value={graphDisplayNameDraft}
                      onChange={event => setGraphDisplayNameDraft(event.target.value)}
                      placeholder="Vendor Risk Graph"
                    />
                  </label>
                  <label className="field">
                    <span>Graph ID</span>
                    <input
                      aria-label="Graph ID"
                      value={graphDraftId}
                      onChange={event => setGraphDraftId(event.target.value)}
                      placeholder="vendor-risk"
                    />
                  </label>
                  <label className="field">
                    <span>Collection</span>
                    <select
                      aria-label="Graph Collection"
                      value={graphCollectionId}
                      onChange={event => setGraphCollectionId(event.target.value)}
                    >
                      <option value="">Choose a collection</option>
                      {collections.map(collection => (
                        <option key={asString(collection.collection_id)} value={asString(collection.collection_id)}>
                          {asString(collection.collection_id)}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="field">
                    <span>Graph Intent</span>
                    <select
                      aria-label="Graph Intent"
                      value={graphIntent}
                      onChange={event => setGraphIntent(event.target.value)}
                    >
                      <option value="general">General Knowledge</option>
                      <option value="vendor_risk">Vendor Risk</option>
                      <option value="requirements">Requirements</option>
                      <option value="policy">Policy & Controls</option>
                      <option value="research">Research Corpus</option>
                    </select>
                  </label>
                </div>

                <div className="field-stack">
                  <SectionTabs
                    tabs={[
                      { id: 'collection', label: 'Use Entire Collection' },
                      { id: 'manual', label: 'Choose Documents' },
                    ]}
                    active={graphSourceMode}
                    onChange={value => setGraphSourceMode(value as GraphSourceMode)}
                    ariaLabel="Graph source mode"
                  />
                  <div className="summary-row">
                    <span>Build Scope</span>
                    <strong>Build graph from {formatWholeNumber(graphBuildDocCount)} indexed documents</strong>
                  </div>
                  {graphSourceMode === 'collection' ? (
                    <div className="summary-list">
                      <div className="summary-row">
                        <span>Collection Documents</span>
                        <strong>{formatWholeNumber(graphCollectionDocs.length)}</strong>
                      </div>
                      <div className="summary-row">
                        <span>Source Mode</span>
                        <strong>Entire collection</strong>
                      </div>
                    </div>
                  ) : (
                    <div className="checkbox-list">
                      {graphCollectionDocs.length > 0 ? graphCollectionDocs.map(doc => {
                        const docId = asString(doc.doc_id)
                        const checked = graphSelectedDocIds.includes(docId)
                        const displayPath = asString(doc.source_display_path || doc.source_path || docId)
                        return (
                          <label key={docId} className={checked ? 'checkbox-card checkbox-card-active' : 'checkbox-card'}>
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => toggleGraphDocSelection(docId)}
                            />
                            <span className="checkbox-copy">
                              <strong>{asString(doc.title, docId)}</strong>
                              <span>{displayPath}</span>
                              <span>{shortId(docId)}</span>
                            </span>
                          </label>
                        )
                      }) : (
                        <EmptyState title="No documents in this collection" body="Use the Collections workspace to ingest KB documents, then come back to build the graph from indexed documents." />
                      )}
                    </div>
                    )}
                </div>

                <ActionBar>
                  <ActionButton tone="secondary" onClick={() => void handleGraphSuggest()}>Suggest Defaults</ActionButton>
                  <ActionButton tone="primary" onClick={() => void handleGraphCreate()}>Save Draft</ActionButton>
                  <ActionButton tone="secondary" onClick={() => void handleGraphValidate()} disabled={!selectedGraph}>Run Preflight</ActionButton>
                  <ActionButton tone="ghost" onClick={() => void handleGraphBuild(false)} disabled={!selectedGraph || graphBuildRunning || graphLifecycleBusy}>Build</ActionButton>
                  <ActionButton tone="ghost" onClick={() => void handleGraphBuild(true)} disabled={!selectedGraph || graphBuildRunning || graphLifecycleBusy}>Refresh</ActionButton>
                  <ActionButton
                    tone="destructive"
                    onClick={() => askConfirm({
                      title: 'Cancel active graph run?',
                      description: `Cancel ${activeGraphRun?.operation ?? 'the active run'} for "${selectedGraph}". The current GraphRAG process will be stopped.`,
                      confirmLabel: 'Cancel Run',
                      run: handleGraphCancelRun,
                    })}
                    disabled={!activeGraphRun || graphLifecycleBusy}
                  >
                    Cancel Run
                  </ActionButton>
                </ActionBar>

                {graphBuildRunning && (
                  <div className="inline-alert inline-alert-warning">
                    <span>Build in progress</span>
                    <strong>{activeGraphRun ? `${activeGraphRun.operation} ${shortId(activeGraphRun.run_id)}` : 'Waiting for progress update'}</strong>
                  </div>
                )}

                {graphAssistantFriendly && (
                  <div className="preview-card collection-result-card">
                    <div className="badge-cluster">
                      <StatusBadge tone={toneForStatus(graphAssistantFriendly.status)}>{humanizeKey(asString(graphAssistantFriendly.status, 'unknown'))}</StatusBadge>
                      <StatusBadge tone={graphAssistantFriendly.ready ? 'ok' : 'warning'}>{asString(graphAssistantFriendly.headline, 'Preflight')}</StatusBadge>
                      <StatusBadge tone="neutral">{formatWholeNumber(graphAssistantFriendly.source_count)} sources</StatusBadge>
                      <StatusBadge tone={graphAssistantFriendly.runtime_ok ? 'ok' : 'danger'}>Runtime {graphAssistantFriendly.runtime_ok ? 'ok' : 'blocked'}</StatusBadge>
                    </div>
                    <div className="summary-list">
                      <div className="summary-row">
                        <span>Model Endpoint</span>
                        <strong>{humanizeKey(asString(graphAssistantFriendly.model_endpoint_status, 'unknown'))}</strong>
                      </div>
                      <div className="summary-row">
                        <span>Extraction Sample</span>
                        <strong>{humanizeKey(asString(graphAssistantFriendly.extraction_status, 'unknown'))}</strong>
                      </div>
                      <div className="summary-row">
                        <span>Community Reports</span>
                        <strong>{humanizeKey(asString(graphAssistantFriendly.community_report_status, 'unknown'))}</strong>
                      </div>
                    </div>
                    {asArray<string>(graphAssistantFriendly.blockers).length > 0 && (
                      <div className="inline-alert">
                        <span>Blockers</span>
                        <strong>{shortList(asArray<string>(graphAssistantFriendly.blockers).slice(0, 4))}</strong>
                      </div>
                    )}
                    {asArray<string>(graphAssistantFriendly.warnings).length > 0 && (
                      <div className="inline-alert inline-alert-warning">
                        <span>Warnings</span>
                        <strong>{shortList(asArray<string>(graphAssistantFriendly.warnings).slice(0, 4))}</strong>
                      </div>
                    )}
                  </div>
                )}

                <div className="collection-action-panel collection-action-panel-compact">
                  <label className="field">
                    <span>Smoke Query</span>
                    <input
                      aria-label="Graph Smoke Query"
                      value={graphSmokeQuery}
                      onChange={event => setGraphSmokeQuery(event.target.value)}
                      placeholder="What are the main entities and relationships in this graph?"
                    />
                  </label>
                  <ActionButton tone="secondary" onClick={() => void handleGraphSmokeTest()} disabled={!selectedGraph}>Smoke Test</ActionButton>
                </div>

                {graphSmokeFriendly && (
                  <div className="inline-alert inline-alert-warning">
                    <span>{asString(graphSmokeFriendly.status, 'Smoke test')}</span>
                    <strong>{asString(graphSmokeFriendly.message, 'No smoke-test message available.')}</strong>
                  </div>
                )}

                <div className="field-stack">
                  <button
                    type="button"
                    className="collapsible-toggle"
                    aria-expanded={graphAdvancedOpen}
                    onClick={() => setGraphAdvancedOpen(open => !open)}
                  >
                    {graphAdvancedOpen ? 'Hide Advanced GraphRAG JSON' : 'Show Advanced GraphRAG JSON'}
                  </button>
                  {graphAdvancedOpen && (
                    <>
                      <div className="form-grid">
                        <label className="field">
                          <span>Prompt Overrides JSON</span>
                          <textarea
                            aria-label="Graph Prompt Overrides"
                            rows={8}
                            value={graphPromptDraft}
                            onChange={event => setGraphPromptDraft(event.target.value)}
                            placeholder='{"extract_graph.txt": "Custom extraction instructions"}'
                          />
                        </label>
                        <label className="field">
                          <span>Config Overrides JSON</span>
                          <textarea
                            aria-label="Graph Config Overrides"
                            rows={8}
                            value={graphConfigDraft}
                            onChange={event => setGraphConfigDraft(event.target.value)}
                            placeholder='{"extract_graph": {"entity_types": ["vendor", "risk"]}}'
                          />
                        </label>
                      </div>
                      <ActionBar>
                        <ActionButton tone="ghost" onClick={() => void handleGraphSavePrompts()} disabled={!selectedGraph}>Save Prompt Overrides</ActionButton>
                      </ActionBar>
                    </>
                  )}
                </div>
              </SurfaceCard>

              <SurfaceCard
                title="Research & Tune"
                subtitle="Optional pre-build research that drafts GraphRAG prompt overrides for review before they affect any build."
              >
                <label className="field">
                  <span>Research Guidance</span>
                  <textarea
                    aria-label="Research Tune Guidance"
                    rows={4}
                    value={graphTuneGuidance}
                    onChange={event => setGraphTuneGuidance(event.target.value)}
                    placeholder="Focus on policy controls, owners, exceptions, and approval relationships."
                  />
                </label>

                <div className="field-stack">
                  <span className="field-label">Prompt Targets</span>
                  <div className="checkbox-list">
                    {GRAPH_RESEARCH_TUNE_TARGETS.map(promptFile => {
                      const checked = graphTuneTargets.includes(promptFile)
                      return (
                        <label key={promptFile} className={checked ? 'checkbox-card checkbox-card-active' : 'checkbox-card'}>
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={() => toggleGraphTuneTarget(promptFile)}
                          />
                          <span className="checkbox-copy">
                            <strong>{promptFile}</strong>
                            <span>{checked ? 'Included' : 'Skipped'}</span>
                          </span>
                        </label>
                      )
                    })}
                  </div>
                </div>

                <ActionBar>
                  <ActionButton tone="primary" onClick={() => void handleGraphResearchTune()} disabled={!selectedGraph || graphTuneRunning}>
                    {graphTuneRunning ? 'Running Research & Tune' : 'Run Research & Tune'}
                  </ActionButton>
                  <ActionButton
                    tone="secondary"
                    onClick={() => void handleGraphResearchTuneApply()}
                    disabled={!selectedGraph || !graphTuneResult?.run_id || graphTuneSelectedPrompts.length === 0}
                  >
                    Apply Selected Prompts
                  </ActionButton>
                </ActionBar>

                {graphTuneResult ? (
                  <div className="field-stack">
                    <div className="badge-cluster">
                      <StatusBadge tone={toneForStatus(graphTuneResult.status)}>{asString(graphTuneResult.status, 'completed')}</StatusBadge>
                      <StatusBadge tone="neutral">{shortId(asString(graphTuneResult.run_id, 'run'))}</StatusBadge>
                      <StatusBadge tone={asString(graphTuneCoverage?.coverage_state) === 'complete' ? 'ok' : 'warning'}>
                        {humanizeKey(asString(graphTuneCoverage?.coverage_state, 'unknown'))}
                      </StatusBadge>
                    </div>

                    <div className="summary-list">
                      <div className="summary-row">
                        <span>Reviewed Docs</span>
                        <strong>{formatWholeNumber(graphTuneCoverage?.digested_doc_count)}</strong>
                      </div>
                      <div className="summary-row">
                        <span>Resolved Sources</span>
                        <strong>{formatWholeNumber(graphTuneCoverage?.resolved_source_count)}</strong>
                      </div>
                      <div className="summary-row">
                        <span>Prompt Drafts</span>
                        <strong>{formatWholeNumber(graphTunePromptDraftEntries.length)}</strong>
                      </div>
                    </div>

                    {graphTuneCorpusProfile && (
                      <p className="muted-copy">{asString(graphTuneCorpusProfile.corpus_summary, 'No corpus profile summary was produced.')}</p>
                    )}

                    {graphTuneWarnings.length > 0 && (
                      <div className="inline-alert inline-alert-warning">
                        <span>Warnings</span>
                        <strong>{shortList(graphTuneWarnings.slice(0, 4))}</strong>
                      </div>
                    )}

                    {asString(graphTuneResult.scratchpad_preview) && (
                      <details>
                        <summary>Scratchpad Preview</summary>
                        <div className="code-panel code-panel-scroll">{asString(graphTuneResult.scratchpad_preview)}</div>
                      </details>
                    )}

                    {graphTunePromptDraftEntries.length > 0 ? (
                      <div className="field-stack">
                        {graphTunePromptDraftEntries.map(([promptFile, draft]) => {
                          const draftRecord = asRecord(draft) ?? {}
                          const validation = asRecord(draftRecord.validation)
                          const diffRecord = asRecord(graphTuneResult.prompt_diffs?.[promptFile])
                          const selected = graphTuneSelectedPrompts.includes(promptFile)
                          const valid = validation?.ok !== false
                          return (
                            <div key={promptFile} className="preview-card">
                              <div className="tool-card-head">
                                <label className="inline-check">
                                  <input
                                    type="checkbox"
                                    checked={selected}
                                    disabled={!valid}
                                    onChange={() => toggleGraphTuneSelectedPrompt(promptFile)}
                                  />
                                  <strong>{promptFile}</strong>
                                </label>
                                <StatusBadge tone={valid ? 'ok' : 'danger'}>{valid ? 'Valid' : 'Blocked'}</StatusBadge>
                              </div>
                              <p>{asString(draftRecord.summary, 'Generated prompt draft')}</p>
                              <span>{asString(draftRecord.baseline_source, 'baseline')}</span>
                              {asArray<string>(draftRecord.warnings).length > 0 && (
                                <div className="inline-alert inline-alert-warning">
                                  <span>Draft warnings</span>
                                  <strong>{shortList(asArray<string>(draftRecord.warnings).slice(0, 3))}</strong>
                                </div>
                              )}
                              {asString(diffRecord?.diff) && (
                                <details>
                                  <summary>Diff</summary>
                                  <div className="code-panel code-panel-scroll">{asString(diffRecord?.diff)}</div>
                                </details>
                              )}
                              {asString(draftRecord.content) && (
                                <details>
                                  <summary>Draft Content</summary>
                                  <div className="code-panel code-panel-scroll">{asString(draftRecord.content)}</div>
                                </details>
                              )}
                            </div>
                          )
                        })}
                      </div>
                    ) : (
                      <EmptyState title="No prompt drafts" body="This run did not produce any draftable prompt overrides." />
                    )}
                  </div>
                ) : (
                  <EmptyState title="No tuning run yet" body="Drafts appear here after an operator starts Research & Tune for the selected graph." />
                )}
              </SurfaceCard>

              <SurfaceCard
                title="Graph-Bound Skills"
                subtitle="Attach existing skill ids and author one graph-specific overlay so graph-aware retrieval can inject domain rules only when this graph is selected."
              >
                <label className="field">
                  <span>Bound Skill IDs</span>
                  <input
                    aria-label="Bound Graph Skill IDs"
                    value={graphSkillIdsDraft}
                    onChange={event => setGraphSkillIdsDraft(event.target.value)}
                    placeholder="graph-vendor-risk-overlay, vendor-risk-local-tracing"
                  />
                </label>
                <div className="badge-cluster">
                  {skills.slice(0, 8).map(skill => (
                    <StatusBadge key={asString(skill.skill_id)} tone="neutral">
                      {asString(skill.name, asString(skill.skill_id))}
                    </StatusBadge>
                  ))}
                </div>
                <label className="field">
                  <span>Overlay Skill Markdown</span>
                  <textarea
                    aria-label="Graph Overlay Skill Markdown"
                    rows={10}
                    value={graphSkillOverlayDraft}
                    onChange={event => setGraphSkillOverlayDraft(event.target.value)}
                    placeholder="# Graph Overlay&#10;agent_scope: rag&#10;&#10;## Workflow&#10;&#10;- Explain ontology-specific graph cues here."
                  />
                </label>
                <ActionBar>
                  <ActionButton tone="secondary" onClick={() => void handleGraphSaveSkills()} disabled={!selectedGraph}>Save Skill Overlay</ActionButton>
                </ActionBar>
              </SurfaceCard>
            </div>

            <CollapsibleSurfaceCard
              className="graph-inspector-card"
              title={selectedGraph ? 'Graph Inspector' : 'Graph Notes'}
              subtitle="Review build state, query readiness, recent runs, logs, and any validation payload without leaving the workspace."
              open={graphInspectorOpen}
              onToggle={() => setGraphInspectorOpen(open => !open)}
            >
              {graphDetail ? (
                <>
                  <div className="badge-cluster">
                    <StatusBadge tone={toneForStatus(graphDetail.graph.status)}>{asString(graphDetail.graph.status, 'draft')}</StatusBadge>
                    <StatusBadge tone={graphDetail.graph.query_ready ? 'ok' : 'warning'}>
                      {graphDetail.graph.query_ready ? 'Query Ready' : 'Not Query Ready'}
                    </StatusBadge>
                    <StatusBadge tone="neutral">{asString(graphDetail.graph.backend, 'backend')}</StatusBadge>
                  </div>
                  <div className="summary-list">
                    <div className="summary-row">
                      <span>Collection</span>
                      <strong>{asString(graphDetail.graph.collection_id, 'Unknown')}</strong>
                    </div>
                    <div className="summary-row">
                      <span>Sources</span>
                      <strong>{asArray<string>(graphDetail.graph.source_doc_ids).length}</strong>
                    </div>
                    <div className="summary-row">
                      <span>Last Indexed</span>
                      <strong>{formatTimestamp(graphDetail.graph.last_indexed_at)}</strong>
                    </div>
                    <div className="summary-row">
                      <span>RBAC Access</span>
                      <strong>{selectedGraphGrantSummary.principalNames.length > 0 ? shortList(selectedGraphGrantSummary.principalNames) : 'No current grants'}</strong>
                    </div>
                  </div>
                  <div className="collection-action-panel collection-action-panel-compact">
                    <label className="inline-check">
                      <input
                        type="checkbox"
                        checked={deleteGraphArtifacts}
                        onChange={event => setDeleteGraphArtifacts(event.target.checked)}
                      />
                      <span>Delete on-disk GraphRAG artifacts too</span>
                    </label>
                    <ActionButton
                      tone="destructive"
                      onClick={() => askConfirm({
                        title: 'Delete this graph?',
                        description: deleteGraphArtifacts
                          ? `Delete "${graphDetail.graph.display_name || graphDetail.graph.graph_id}" and remove its on-disk GraphRAG artifacts. This cannot be undone.`
                          : `Delete "${graphDetail.graph.display_name || graphDetail.graph.graph_id}" from the catalog and keep on-disk GraphRAG artifacts.`,
                        confirmLabel: 'Delete Graph',
                        run: handleDeleteGraph,
                      })}
                      disabled={!selectedGraph || graphBuildRunning || graphLifecycleBusy}
                    >
                      Delete Graph
                    </ActionButton>
                  </div>
                  {graphProgress && (
                    <div className="graph-progress-panel">
                      <div className="tool-card-head">
                        <strong>Build Progress</strong>
                        <StatusBadge tone={graphProgress.active ? 'accent' : toneForStatus(graphProgress.status)}>
                          {graphProgress.active ? 'Live polling' : humanizeKey(asString(graphProgress.status, 'idle'))}
                        </StatusBadge>
                      </div>
                      <div className="progress-track" aria-label="Graph build progress">
                        <span style={{ width: `${Math.max(0, Math.min(100, Number(graphProgress.percent || 0)))}%` }} />
                      </div>
                      <div className="summary-row">
                        <span>{asString(graphProgress.workflow, 'Waiting for workflow')}</span>
                        <strong>{formatPercent((Number(graphProgress.percent || 0)) / 100)}</strong>
                      </div>
                      <div className="stage-list">
                        {graphProgress.stages.map(stage => (
                          <span key={stage.id} className={`stage-chip stage-chip-${asString(stage.state, 'pending')}`}>
                            {stage.label}
                          </span>
                        ))}
                      </div>
                      {asString(graphProgress.log_tail) && (
                        <div className="code-panel code-panel-scroll">{asString(graphProgress.log_tail)}</div>
                      )}
                    </div>
                  )}
                  <DetailTabs
                    tabs={[
                      {
                        id: 'summary',
                        label: 'Summary',
                        content: (
                          <div className="field-stack">
                            <p className="muted-copy">{asString(graphDetail.graph.domain_summary, 'No domain summary yet.')}</p>
                            <JsonInspector label="Technical details" value={graphDetail.graph} />
                          </div>
                        ),
                      },
                      {
                        id: 'runs',
                        label: 'Runs',
                        content: graphRuns.length > 0 ? (
                          <div className="timeline-list">
                            {graphRuns.map(run => (
                              <article key={run.run_id} className="timeline-item">
                                <div className="timeline-dot" aria-hidden="true" />
                                <div>
                                  <strong>{run.operation}</strong>
                                  <p>{run.detail}</p>
                                  <span>{formatTimestamp(run.completed_at || run.started_at)}</span>
                                </div>
                              </article>
                            ))}
                          </div>
                        ) : (
                          <EmptyState title="No graph runs yet" body="Validate or build the selected graph to populate recent run history." />
                        ),
                      },
                      {
                        id: 'logs',
                        label: 'Logs',
                        content: asArray<Record<string, unknown>>(graphProgress?.logs ?? graphDetail.logs).length > 0 ? (
                          <div className="field-stack">
                            {asArray<Record<string, unknown>>(graphProgress?.logs ?? graphDetail.logs).map(log => (
                              <div key={asString(log.path)} className="preview-card">
                                <div className="tool-card-head">
                                  <strong>{asString(log.name, 'log')}</strong>
                                  <StatusBadge tone="neutral">{formatTimestamp(log.modified_at)}</StatusBadge>
                                </div>
                                <div className="code-panel">{asString(log.preview, 'No preview available.')}</div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <EmptyState title="No logs yet" body="GraphRAG logs will appear here after validation or build operations write to the project log directory." />
                        ),
                      },
                    ]}
                  />
                  {graphValidation && <JsonInspector label="Latest validation/build payload" value={graphValidation} />}
                </>
              ) : (
                <EmptyState title="Select or create a graph" body="The inspector will show query readiness, runs, logs, and graph-bound skills once a graph is selected." />
              )}
            </CollapsibleSurfaceCard>

            <SurfaceCard
              title="Graph Quality Studio"
              subtitle="Spot coverage, freshness, and grounding signals after a build so users know when GraphRAG is stronger than normal RAG."
            >
              {graphDetail ? (
                <div className="field-stack">
                  <div className="collection-summary-strip">
                    <div className="meta-chip">
                      <span>Entities</span>
                      <strong>{formatWholeNumber(graphQualityHealth?.entity_count ?? graphDetail.graph.entity_samples?.length)}</strong>
                    </div>
                    <div className="meta-chip">
                      <span>Relationships</span>
                      <strong>{formatWholeNumber(graphQualityHealth?.relationship_count ?? graphDetail.graph.relationship_samples?.length)}</strong>
                    </div>
                    <div className="meta-chip">
                      <span>Source Coverage</span>
                      <strong>{formatWholeNumber(graphDetail.sources.length)} sources</strong>
                    </div>
                    <div className="meta-chip">
                      <span>Freshness</span>
                      <strong>{formatPercent(graphDetail.graph.freshness_score)}</strong>
                    </div>
                  </div>

                  <div className="form-grid form-grid-compact">
                    <div className="preview-card">
                      <div className="tool-card-head">
                        <strong>Entity Samples</strong>
                        <StatusBadge tone="neutral">{graphDetail.graph.entity_samples?.length ?? 0}</StatusBadge>
                      </div>
                      {graphDetail.graph.entity_samples?.length ? (
                        <p>{shortList(graphDetail.graph.entity_samples.slice(0, 6))}</p>
                      ) : (
                        <p>No entity samples available yet.</p>
                      )}
                    </div>
                    <div className="preview-card">
                      <div className="tool-card-head">
                        <strong>Relationship Samples</strong>
                        <StatusBadge tone="neutral">{graphDetail.graph.relationship_samples?.length ?? 0}</StatusBadge>
                      </div>
                      {graphDetail.graph.relationship_samples?.length ? (
                        <p>{shortList(graphDetail.graph.relationship_samples.slice(0, 6))}</p>
                      ) : (
                        <p>No relationship samples available yet.</p>
                      )}
                    </div>
                  </div>

                  {graphQualityWarnings.length > 0 ? (
                    <div className="inline-alert inline-alert-warning">
                      <span>Why this graph may be weak</span>
                      <strong>{shortList(graphQualityWarnings.slice(0, 5))}</strong>
                    </div>
                  ) : (
                    <div className="inline-alert inline-alert-warning">
                      <span>Graph / vector guidance</span>
                      <strong>Use GraphRAG for relationship-heavy questions; use normal RAG for pinpoint quotes, table lookup, and single-document evidence.</strong>
                    </div>
                  )}

                  <div className="summary-list">
                    <div className="summary-row">
                      <span>GraphRAG Best For</span>
                      <strong>Entities, relationships, communities, multi-hop summaries</strong>
                    </div>
                    <div className="summary-row">
                      <span>Normal RAG Best For</span>
                      <strong>Exact passages, narrow facts, fresh source-only retrieval</strong>
                    </div>
                  </div>
                </div>
              ) : (
                <EmptyState title="No graph selected" body="Build or select a graph to inspect quality signals." />
              )}
            </SurfaceCard>
          </div>
        ) : (
          <div className="content-stack">
            <SurfaceCard title="Graph Runs" subtitle="Operational history for the selected graph build, refresh, and validation steps.">
              {graphRuns.length > 0 ? (
                <div className="timeline-list">
                  {graphRuns.map(run => (
                    <article key={run.run_id} className="timeline-item">
                      <div className="timeline-dot" aria-hidden="true" />
                      <div>
                        <strong>{run.operation}</strong>
                        <p>{run.detail}</p>
                        <span>{formatTimestamp(run.completed_at || run.started_at)}</span>
                      </div>
                    </article>
                  ))}
                </div>
              ) : (
                <EmptyState title="No graph runs yet" body="Select a graph and validate or build it to populate run history." />
              )}
              {graphDetail && <JsonInspector label="Technical details" value={graphDetail} />}
            </SurfaceCard>
          </div>
        )
      )}

      {active === 'access' && (
        <div className="content-stack">
          <SectionHeader
            eyebrow="Open WebUI-inspired RBAC"
            title="Access Control"
            description="Manage users, groups, roles, resource grants, and effective access through guided workflows instead of one flat policy dump."
            actions={(
              <>
                <ActionButton tone="primary" onClick={() => setAccessWizardDefaults('setup')}>Access Setup Wizard</ActionButton>
                <ActionButton tone="secondary" onClick={() => setAccessWizardDefaults('grant')}>Grant Resource Access</ActionButton>
              </>
            )}
          />
          <SectionTabs
            tabs={ACCESS_TABS}
            active={accessTab}
            onChange={tab => setAccessTab(tab as AccessTab)}
            ariaLabel="Access workspace"
          />

          {accessTab === 'overview' && (
            <div className="content-stack">
              <div className="stats-grid">
                <StatCard label="Users" value={accessPrincipals.filter(principal => principal.principal_type === 'user').length} caption="Email-backed principals" />
                <StatCard label="Groups" value={accessGroups.length} caption="Preferred grant target" />
                <StatCard label="Roles" value={accessRoles.length} caption="Reusable permission bundles" />
                <StatCard label="Resource Grants" value={accessPermissions.length} caption="Additive permissions" />
              </div>
              <div className="card-grid">
                <SurfaceCard title="Recommended Workflow" subtitle="Group-first access keeps policy readable as the runtime grows.">
                  <div className="summary-list">
                    <div className="summary-row">
                      <span>1. Create groups</span>
                      <strong>Permission, team, or admin groups</strong>
                    </div>
                    <div className="summary-row">
                      <span>2. Bind presets</span>
                      <strong>KB reader, graph builder, agent operator</strong>
                    </div>
                    <div className="summary-row">
                      <span>3. Preview access</span>
                      <strong>Check effective permissions before users test</strong>
                    </div>
                  </div>
                  <ActionBar>
                    <ActionButton tone="primary" onClick={() => setAccessWizardDefaults('setup')}>Start Guided Setup</ActionButton>
                    <ActionButton tone="secondary" onClick={() => setAccessWizardDefaults('group')}>Create Group</ActionButton>
                    <ActionButton tone="secondary" onClick={() => setAccessWizardDefaults('user')}>Manage User</ActionButton>
                  </ActionBar>
                </SurfaceCard>
                <SurfaceCard title="Recent Policy Shape" subtitle="A compact readout before diving into matrix-level detail.">
                  <div className="summary-list">
                    <div className="summary-row">
                      <span>Direct user bindings</span>
                      <strong>{formatWholeNumber(accessBindings.filter(binding => accessPrincipalById.get(binding.principal_id)?.principal_type === 'user').length)}</strong>
                    </div>
                    <div className="summary-row">
                      <span>Group bindings</span>
                      <strong>{formatWholeNumber(accessBindings.filter(binding => accessPrincipalById.get(binding.principal_id)?.principal_type === 'group').length)}</strong>
                    </div>
                    <div className="summary-row">
                      <span>Memberships</span>
                      <strong>{formatWholeNumber(accessMemberships.length)}</strong>
                    </div>
                    <div className="summary-row">
                      <span>Wildcard grants</span>
                      <strong>{formatWholeNumber(accessPermissions.filter(permission => permission.resource_selector === '*').length)}</strong>
                    </div>
                  </div>
                </SurfaceCard>
              </div>
            </div>
          )}

          {accessTab === 'users' && (
            <SurfaceCard title="Users" subtitle="Create users, assign a simple system status, add them to groups, and preview effective access.">
              <ActionBar>
                <ActionButton tone="primary" onClick={() => setAccessWizardDefaults('user')}>Manage User</ActionButton>
                <ActionButton tone="secondary" onClick={() => setAccessWizardDefaults('setup')}>Add To Setup Wizard</ActionButton>
              </ActionBar>
              <ResourceSearch value={accessUserSearch} onChange={setAccessUserSearch} placeholder="Search users" />
              <DataTable<AccessPrincipal>
                ariaLabel="Access users"
                rows={accessUsers}
                rowKey={principal => principal.principal_id}
                columns={[
                  {
                    key: 'user',
                    header: 'User',
                    sortable: true,
                    accessor: principal => principalLabel(principal),
                    render: principal => (
                      <div>
                        <strong>{principalLabel(principal)}</strong>
                        <p>{principal.email_normalized || principal.principal_id}</p>
                      </div>
                    ),
                  },
                  {
                    key: 'status',
                    header: 'System Status',
                    sortable: true,
                    accessor: principal => principalSystemRole(principal),
                    render: principal => <StatusBadge tone={principalSystemRole(principal) === 'pending' ? 'warning' : principalSystemRole(principal) === 'admin' ? 'accent' : 'neutral'}>{principalSystemRole(principal)}</StatusBadge>,
                  },
                  {
                    key: 'groups',
                    header: 'Groups',
                    render: principal => shortList(accessMemberships
                      .filter(membership => membership.child_principal_id === principal.principal_id)
                      .map(membership => principalLabel(accessPrincipalById.get(membership.parent_principal_id)))),
                  },
                  {
                    key: 'active',
                    header: 'Active',
                    render: principal => <StatusBadge tone={principal.active ? 'ok' : 'warning'}>{principal.active ? 'Active' : 'Pending'}</StatusBadge>,
                  },
                ]}
                emptyState={<EmptyState title="No users" body="Create a user with the Manage User wizard." />}
              />
            </SurfaceCard>
          )}

          {accessTab === 'groups' && (
            <SurfaceCard title="Groups" subtitle="Groups are the preferred place to attach roles and resource grants. Direct user grants should stay rare.">
              <ActionBar>
                <ActionButton tone="primary" onClick={() => setAccessWizardDefaults('group')}>Create Group</ActionButton>
                <ActionButton tone="secondary" onClick={() => setAccessWizardDefaults('setup')}>Access Setup Wizard</ActionButton>
              </ActionBar>
              <ResourceSearch value={accessGroupSearch} onChange={setAccessGroupSearch} placeholder="Search groups" />
              <DataTable<AccessPrincipal>
                ariaLabel="Access groups"
                rows={accessGroups}
                rowKey={principal => principal.principal_id}
                columns={[
                  {
                    key: 'group',
                    header: 'Group',
                    sortable: true,
                    accessor: principal => principalLabel(principal),
                    render: principal => (
                      <div>
                        <strong>{principalLabel(principal)}</strong>
                        <p>{groupPurposeLabel(groupPurpose(principal))}</p>
                      </div>
                    ),
                  },
                  {
                    key: 'members',
                    header: 'Members',
                    render: principal => formatWholeNumber(accessMembershipsByGroup.get(principal.principal_id)?.length ?? 0),
                  },
                  {
                    key: 'roles',
                    header: 'Bound Roles',
                    render: principal => shortList(accessBindings
                      .filter(binding => binding.principal_id === principal.principal_id && !binding.disabled)
                      .map(binding => accessRoleById.get(binding.role_id)?.name || binding.role_id)),
                  },
                  {
                    key: 'purpose',
                    header: 'Purpose',
                    render: principal => <StatusBadge tone={groupPurpose(principal) === 'admin' ? 'accent' : groupPurpose(principal) === 'team' ? 'neutral' : 'ok'}>{groupPurposeLabel(groupPurpose(principal))}</StatusBadge>,
                  },
                ]}
                emptyState={<EmptyState title="No groups" body="Create a permission group before assigning resource access." />}
              />
            </SurfaceCard>
          )}

          {accessTab === 'roles' && (
            <div className="content-stack">
              <div className="card-grid">
                {ACCESS_PRESETS.filter(preset => preset.id !== 'custom').map(preset => (
                  <SurfaceCard key={preset.id} title={preset.label} subtitle={preset.description}>
                    <div className="badge-cluster">
                      <StatusBadge tone="neutral">{accessResourceLabel(preset.resourceType)}</StatusBadge>
                      {preset.actions.map(action => <StatusBadge key={action} tone={action === 'manage' ? 'accent' : 'neutral'}>{action}</StatusBadge>)}
                    </div>
                    <ActionBar>
                      <ActionButton
                        tone="secondary"
                        onClick={() => {
                          setSetupPresetId(preset.id)
                          setSetupResourceType(preset.resourceType)
                          setSetupActions(preset.actions)
                          setAccessWizardDefaults('setup')
                        }}
                      >
                        Use Preset
                      </ActionButton>
                    </ActionBar>
                  </SurfaceCard>
                ))}
              </div>
              <SurfaceCard title="Roles" subtitle="Reusable permission bundles still exist underneath the friendly group and grant workflows.">
                <div className="form-grid form-grid-compact">
                  <label className="field">
                    <span>Role Name</span>
                    <input value={roleDraftName} onChange={event => setRoleDraftName(event.target.value)} placeholder="finance-analyst" />
                  </label>
                  <label className="field">
                    <span>Description</span>
                    <input value={roleDraftDescription} onChange={event => setRoleDraftDescription(event.target.value)} placeholder="Access to finance KBs and graph skills" />
                  </label>
                </div>
                <ActionBar>
                  <ActionButton tone="secondary" onClick={() => void handleCreateRole()}>Save Role</ActionButton>
                </ActionBar>
                <DataTable<AccessRole>
                  ariaLabel="Access roles"
                  rows={accessRoles}
                  rowKey={role => role.role_id}
                  columns={[
                    { key: 'name', header: 'Role', sortable: true, accessor: role => role.name, render: role => <strong>{role.name}</strong> },
                    { key: 'description', header: 'Description', render: role => role.description || 'No description' },
                    { key: 'permissions', header: 'Permissions', render: role => formatWholeNumber(accessPermissions.filter(permission => permission.role_id === role.role_id).length) },
                    { key: 'bindings', header: 'Bindings', render: role => formatWholeNumber(accessBindings.filter(binding => binding.role_id === role.role_id && !binding.disabled).length) },
                  ]}
                  rowActions={role => [
                    {
                      label: 'Delete role',
                      tone: 'danger',
                      onSelect: () => askConfirm({
                        title: 'Delete this role?',
                        description: `Remove "${role.name}". Existing bindings using this role will also be removed.`,
                        confirmLabel: 'Delete',
                        run: () => api.deleteAccessRole(token, role.role_id)
                          .then(() => { notifyOk('Role deleted', role.name); return refreshAccessData() })
                          .catch(err => { notifyError('Delete role failed', err); setError(getMessage(err)) }),
                      }),
                    },
                  ]}
                  emptyState={<EmptyState title="No roles yet" body="Create a role or use a preset from the cards above." />}
                />
              </SurfaceCard>
            </div>
          )}

          {accessTab === 'grants' && (
            <div className="content-stack">
              <SurfaceCard title="Resource Grants" subtitle="Resource-centric grants read like normalized ACLs, then write through existing roles and bindings.">
                <ActionBar>
                  <ActionButton tone="primary" onClick={() => setAccessWizardDefaults('grant')}>Grant Resource Access</ActionButton>
                </ActionBar>
                <ResourceSearch value={accessGrantSearch} onChange={setAccessGrantSearch} placeholder="Search grants" />
                <DataTable<(typeof accessGrantRows)[number]>
                  ariaLabel="Resource grants"
                  rows={accessGrantRows}
                  rowKey={row => row.key}
                  columns={[
                    { key: 'resource', header: 'Resource', sortable: true, accessor: row => row.resourceLabel, render: row => <strong>{row.resourceLabel}</strong> },
                    { key: 'selector', header: 'Selector', sortable: true, accessor: row => row.selectorLabel, render: row => <code>{row.selectorLabel}</code> },
                    { key: 'action', header: 'Action', sortable: true, accessor: row => row.action, render: row => <StatusBadge tone={row.action === 'manage' ? 'accent' : row.action === 'approve' ? 'warning' : 'neutral'}>{row.action}</StatusBadge> },
                    { key: 'principals', header: 'Who', render: row => row.principalNames.length > 0 ? shortList(row.principalNames) : 'No active bindings' },
                    { key: 'role', header: 'Via Role', render: row => row.roleName },
                  ]}
                  rowActions={row => [
                    {
                      label: 'Remove permission',
                      tone: 'danger',
                      onSelect: () => askConfirm({
                        title: 'Remove this permission?',
                        description: 'The role will lose this grant. Effective access for bound principals will narrow.',
                        confirmLabel: 'Remove',
                        run: () => api.deleteAccessPermission(token, row.key)
                          .then(() => { notifyOk('Permission removed'); return refreshAccessData() })
                          .catch(err => { notifyError('Remove permission failed', err); setError(getMessage(err)) }),
                      }),
                    },
                  ]}
                  emptyState={<EmptyState title="No grants" body="Use the Grant Resource Access wizard to add a resource-centric grant." />}
                />
              </SurfaceCard>

              <SurfaceCard title="Manual Permission Entry" subtitle="Advanced escape hatch for exact role, resource, action, and selector writes.">
                <div className="form-grid form-grid-compact">
                  <label className="field">
                    <span>Role</span>
                    <select value={permissionRoleId} onChange={event => setPermissionRoleId(event.target.value)}>
                      <option value="">Choose a role</option>
                      {accessRoles.map(role => <option key={role.role_id} value={role.role_id}>{role.name}</option>)}
                    </select>
                  </label>
                  <label className="field">
                    <span>Resource Type</span>
                    <select value={permissionResourceType} onChange={event => setPermissionResourceType(event.target.value as AccessResourceType)}>
                      {ACCESS_RESOURCE_TYPES.map(resourceType => (
                        <option key={resourceType.key} value={resourceType.key}>{resourceType.label}</option>
                      ))}
                    </select>
                  </label>
                  <label className="field">
                    <span>Action</span>
                    <select value={permissionAction} onChange={event => setPermissionAction(event.target.value as AccessAction)}>
                      {ACCESS_ACTIONS.map(action => <option key={action.key} value={action.key}>{action.label}</option>)}
                    </select>
                  </label>
                  <label className="field">
                    <span>Resource Selector</span>
                    <select value={permissionSelector} onChange={event => setPermissionSelector(event.target.value)}>
                      {accessResourceOptions[permissionResourceType].map(option => (
                        <option key={option.id} value={option.id === '*' ? '' : option.id}>{option.label}</option>
                      ))}
                    </select>
                  </label>
                </div>
                <ActionBar>
                  <ActionButton tone="secondary" onClick={() => void handleCreatePermission()}>Add Permission</ActionButton>
                </ActionBar>
              </SurfaceCard>
            </div>
          )}

          {accessTab === 'effective' && (
            <SurfaceCard title="Effective Access" subtitle="Preview the resolved principal graph and explain the grants before showing raw JSON.">
              <label className="field">
                <span>User Email</span>
                <input value={accessPreviewEmail} onChange={event => setAccessPreviewEmail(event.target.value)} placeholder="alex@example.com" />
              </label>
              <ActionBar>
                <ActionButton tone="primary" onClick={() => void handleLoadEffectiveAccess()}>Preview Access</ActionButton>
              </ActionBar>
              {effectiveAccess ? (
                <div className="field-stack">
                  <div className="summary-list">
                    <div className="summary-row">
                      <span>Preview Email</span>
                      <strong>{effectiveAccess.email}</strong>
                    </div>
                    <div className="summary-row">
                      <span>Authz Enabled</span>
                      <strong>{boolLabel(Boolean(asRecord(effectiveAccess.access)?.authz_enabled), 'Yes', 'No')}</strong>
                    </div>
                  </div>
                  <div className="card-grid">
                    {effectiveAccessRows.map(row => (
                      <div key={row.key} className="preview-card">
                        <div className="tool-card-head">
                          <strong>{row.label}</strong>
                          <StatusBadge tone={row.allowed.length > 0 ? 'ok' : 'warning'}>{row.allowed.length > 0 ? 'Allowed' : 'No Grant'}</StatusBadge>
                        </div>
                        <p>{row.allowed.length > 0 ? shortList(row.allowed.slice(0, 6)) : 'Blocked / No Grant'}</p>
                        <Popover
                          ariaLabel={`${row.label} why`}
                          trigger={({ toggle, ref }) => (
                            <button type="button" ref={el => ref(el)} onClick={toggle} className="icon-btn icon-btn-ghost">Why?</button>
                          )}
                        >
                          <div className="field-stack">
                            <strong>{row.label}</strong>
                            {row.why.length > 0 ? row.why.slice(0, 8).map(reason => <p key={reason}>{reason}</p>) : <p>No matching role permission was found in the local grant table.</p>}
                          </div>
                        </Popover>
                      </div>
                    ))}
                  </div>
                  <JsonInspector label="Effective access snapshot" value={effectiveAccess.access} />
                </div>
              ) : (
                <EmptyState title="No preview yet" body="Choose a user email to inspect the exact access snapshot the runtime will resolve." />
              )}
            </SurfaceCard>
          )}

          {accessTab === 'advanced' && (
            <div className="content-stack">
              <SurfaceCard
                title="Access Matrix"
                subtitle="Advanced comparison view. Click a cell for the source of a grant."
                className="rbac-matrix-card"
              >
                <RbacMatrix
                  principals={accessPrincipals}
                  roles={accessRoles}
                  bindings={accessBindings}
                  permissions={accessPermissions}
                />
              </SurfaceCard>
              <div className="card-grid">
                <SurfaceCard title="Bindings" subtitle="Role attachments to users and groups.">
                  <DataTable<AccessRoleBinding>
                    ariaLabel="Access bindings"
                    rows={accessBindings}
                    rowKey={binding => binding.binding_id}
                    columns={[
                      { key: 'role', header: 'Role', render: binding => accessRoleById.get(binding.role_id)?.name || binding.role_id },
                      { key: 'principal', header: 'Principal', render: binding => principalLabel(accessPrincipalById.get(binding.principal_id)) },
                      { key: 'status', header: 'Status', render: binding => <StatusBadge tone={binding.disabled ? 'warning' : 'ok'}>{binding.disabled ? 'Disabled' : 'Active'}</StatusBadge> },
                    ]}
                    rowActions={binding => [
                      {
                        label: 'Remove binding',
                        tone: 'danger',
                        onSelect: () => askConfirm({
                          title: 'Remove this binding?',
                          description: 'The principal will lose access granted via this role until a new binding is created.',
                          confirmLabel: 'Remove',
                          run: () => api.deleteAccessBinding(token, binding.binding_id)
                            .then(() => { notifyOk('Binding removed'); return refreshAccessData() })
                            .catch(err => { notifyError('Remove binding failed', err); setError(getMessage(err)) }),
                        }),
                      },
                    ]}
                    emptyState={<EmptyState title="No bindings yet" body="Bindings connect principals to roles and drive effective runtime access." />}
                  />
                </SurfaceCard>
                <SurfaceCard title="Memberships" subtitle="Group membership edges.">
                  <DataTable<AccessMembership>
                    ariaLabel="Access memberships"
                    rows={accessMemberships}
                    rowKey={membership => membership.membership_id}
                    columns={[
                      { key: 'group', header: 'Group', render: membership => principalLabel(accessPrincipalById.get(membership.parent_principal_id)) },
                      { key: 'member', header: 'Member', render: membership => principalLabel(accessPrincipalById.get(membership.child_principal_id)) },
                      { key: 'created', header: 'Created', render: membership => formatTimestamp(membership.created_at) },
                    ]}
                    rowActions={membership => [
                      {
                        label: 'Remove membership',
                        tone: 'danger',
                        onSelect: () => askConfirm({
                          title: 'Remove this membership?',
                          description: 'Detaches the member from the group. Inherited access will be revoked.',
                          confirmLabel: 'Remove',
                          run: () => api.deleteAccessMembership(token, membership.membership_id)
                            .then(() => { notifyOk('Membership removed'); return refreshAccessData() })
                            .catch(err => { notifyError('Remove membership failed', err); setError(getMessage(err)) }),
                        }),
                      },
                    ]}
                    emptyState={<EmptyState title="No memberships yet" body="Use groups to assign users into policy bundles." />}
                  />
                </SurfaceCard>
              </div>
            </div>
          )}
        </div>
      )}

      <Dialog
        open={accessWizard !== null}
        onClose={() => setAccessWizard(null)}
        title={
          accessWizard === 'setup' ? 'Access Setup Wizard'
            : accessWizard === 'grant' ? 'Grant Resource Access'
              : accessWizard === 'user' ? 'Manage User'
                : 'Create Group'
        }
        description="Guided RBAC changes use the existing principals, roles, memberships, bindings, and permissions APIs."
        size="lg"
        footer={(() => {
          const steps = accessWizardSteps()
          const index = Math.max(0, steps.indexOf(accessWizardStep))
          const isLast = steps.length === 0 || index === steps.length - 1
          return (
            <>
              <ActionButton
                tone="ghost"
                disabled={index <= 0}
                onClick={() => setAccessWizardStep(steps[Math.max(0, index - 1)] ?? '')}
              >
                Back
              </ActionButton>
              {isLast ? (
                <ActionButton tone="primary" onClick={() => void handleFinishAccessWizard()}>
                  Apply
                </ActionButton>
              ) : (
                <ActionButton
                  tone="primary"
                  onClick={() => setAccessWizardStep(steps[Math.min(steps.length - 1, index + 1)] ?? '')}
                >
                  Next
                </ActionButton>
              )}
            </>
          )
        })()}
      >
        <div className="wizard-shell">
          <div className="stage-list">
            {accessWizardSteps().map(step => (
              <button
                key={step}
                type="button"
                className={`stage-chip ${accessWizardStep === step ? 'stage-chip-active' : 'stage-chip-pending'}`}
                onClick={() => setAccessWizardStep(step)}
              >
                {humanizeKey(step)}
              </button>
            ))}
          </div>

          {accessWizard === 'setup' && accessWizardStep === 'group' && (
            <div className="field-stack">
              <SegmentedControl<'existing' | 'create'>
                ariaLabel="Setup group mode"
                value={setupGroupMode}
                onChange={setSetupGroupMode}
                options={[
                  { value: 'existing', label: 'Existing Group' },
                  { value: 'create', label: 'Create Group' },
                ]}
              />
              {setupGroupMode === 'existing' ? (
                <label className="field">
                  <span>Group</span>
                  <select value={setupGroupId} onChange={event => setSetupGroupId(event.target.value)}>
                    <option value="">Choose a group</option>
                    {accessGroups.map(group => <option key={group.principal_id} value={group.principal_id}>{principalLabel(group)}</option>)}
                  </select>
                </label>
              ) : (
                <>
                  <label className="field">
                    <span>Group Name</span>
                    <input value={setupGroupName} onChange={event => setSetupGroupName(event.target.value)} placeholder="Finance Analysts" />
                  </label>
                  <SegmentedControl<AccessGroupPurpose>
                    ariaLabel="Setup group purpose"
                    value={setupGroupPurpose}
                    onChange={setSetupGroupPurpose}
                    options={ACCESS_GROUP_PURPOSES.map(purpose => ({ value: purpose.key, label: purpose.label }))}
                  />
                </>
              )}
            </div>
          )}

          {accessWizard === 'setup' && accessWizardStep === 'preset' && (
            <div className="checkbox-list">
              {ACCESS_PRESETS.map(preset => (
                <label key={preset.id} className={setupPresetId === preset.id ? 'checkbox-card checkbox-card-active' : 'checkbox-card'}>
                  <input
                    type="radio"
                    name="setup-preset"
                    checked={setupPresetId === preset.id}
                    onChange={() => {
                      setSetupPresetId(preset.id)
                      setSetupResourceType(preset.resourceType)
                      setSetupActions(preset.actions)
                    }}
                  />
                  <span className="checkbox-copy">
                    <strong>{preset.label}</strong>
                    <span>{preset.description}</span>
                  </span>
                </label>
              ))}
            </div>
          )}

          {accessWizard === 'setup' && accessWizardStep === 'members' && (
            <div className="checkbox-list">
              {accessPrincipals.filter(principal => principal.principal_type === 'user').map(user => (
                <label key={user.principal_id} className={setupMemberIds.includes(user.principal_id) ? 'checkbox-card checkbox-card-active' : 'checkbox-card'}>
                  <input type="checkbox" checked={setupMemberIds.includes(user.principal_id)} onChange={() => toggleSetupMember(user.principal_id)} />
                  <span className="checkbox-copy">
                    <strong>{principalLabel(user)}</strong>
                    <span>{user.email_normalized}</span>
                  </span>
                </label>
              ))}
              {accessUsers.length === 0 && <EmptyState title="No users yet" body="You can apply the group and add members later from Manage User." />}
            </div>
          )}

          {accessWizard === 'setup' && accessWizardStep === 'resources' && (
            <div className="field-stack">
              <label className="field">
                <span>Resource Type</span>
                <select value={setupResourceType} onChange={event => setSetupResourceType(event.target.value as AccessResourceType)}>
                  {ACCESS_RESOURCE_TYPES.map(resourceType => <option key={resourceType.key} value={resourceType.key}>{resourceType.label}</option>)}
                </select>
              </label>
              <div className="checkbox-list">
                {accessResourceOptions[setupResourceType].map(option => (
                  <label key={option.id} className={setupResourceSelectors.includes(option.id) ? 'checkbox-card checkbox-card-active' : 'checkbox-card'}>
                    <input type="checkbox" checked={setupResourceSelectors.includes(option.id)} onChange={() => toggleSetupSelector(option.id)} />
                    <span className="checkbox-copy">
                      <strong>{option.label}</strong>
                      <span>{option.description}</span>
                    </span>
                  </label>
                ))}
              </div>
              <span className="field-label">Actions</span>
              <div className="checkbox-list">
                {ACCESS_ACTIONS.map(action => (
                  <label key={action.key} className={setupActions.includes(action.key) ? 'checkbox-card checkbox-card-active' : 'checkbox-card'}>
                    <input type="checkbox" checked={setupActions.includes(action.key)} onChange={() => toggleSetupAction(action.key)} />
                    <span className="checkbox-copy">
                      <strong>{action.label}</strong>
                      <span>{action.key === 'use' ? 'Invoke or query the resource' : action.key === 'manage' ? 'Modify settings or grants' : action.key === 'approve' ? 'Approve workflow requests' : 'Delete resource state'}</span>
                    </span>
                  </label>
                ))}
              </div>
            </div>
          )}

          {accessWizard === 'setup' && accessWizardStep === 'preview' && (
            <div className="field-stack">
              <EmptyState title="Preview Current Access" body="Preview a selected member before applying changes, then compare again after the setup is saved." />
              <ActionBar>
                <ActionButton tone="secondary" onClick={() => void handleAccessWizardPreview()} disabled={setupMemberIds.length === 0}>Preview Selected User</ActionButton>
              </ActionBar>
              {accessWizardPreview && <JsonInspector label={`Current access for ${accessWizardPreview.email}`} value={accessWizardPreview.access} />}
            </div>
          )}

          {accessWizard === 'setup' && accessWizardStep === 'review' && (
            <div className="summary-list">
              <div className="summary-row"><span>Group</span><strong>{setupGroupMode === 'create' ? setupGroupName || 'New group' : principalLabel(accessPrincipalById.get(setupGroupId))}</strong></div>
              <div className="summary-row"><span>Preset</span><strong>{ACCESS_PRESETS.find(preset => preset.id === setupPresetId)?.label}</strong></div>
              <div className="summary-row"><span>Members</span><strong>{formatWholeNumber(setupMemberIds.length)}</strong></div>
              <div className="summary-row"><span>Resource</span><strong>{accessResourceLabel(setupResourceType)} / {formatWholeNumber(normalizedAccessSelectors(setupResourceSelectors).length)} selector(s)</strong></div>
              <div className="summary-row"><span>Actions</span><strong>{shortList(setupActions)}</strong></div>
            </div>
          )}

          {accessWizard === 'grant' && accessWizardStep === 'resource' && (
            <div className="field-stack">
              <label className="field">
                <span>Resource Type</span>
                <select value={grantResourceType} onChange={event => setGrantResourceType(event.target.value as AccessResourceType)}>
                  {ACCESS_RESOURCE_TYPES.map(resourceType => <option key={resourceType.key} value={resourceType.key}>{resourceType.label}</option>)}
                </select>
              </label>
              <div className="checkbox-list">
                {accessResourceOptions[grantResourceType].map(option => (
                  <label key={option.id} className={grantResourceSelectors.includes(option.id) ? 'checkbox-card checkbox-card-active' : 'checkbox-card'}>
                    <input type="checkbox" checked={grantResourceSelectors.includes(option.id)} onChange={() => toggleGrantSelector(option.id)} />
                    <span className="checkbox-copy">
                      <strong>{option.label}</strong>
                      <span>{option.description}</span>
                    </span>
                  </label>
                ))}
              </div>
            </div>
          )}

          {accessWizard === 'grant' && accessWizardStep === 'principals' && (
            <div className="checkbox-list">
              {accessPrincipals.map(principal => (
                <label key={principal.principal_id} className={grantPrincipalIds.includes(principal.principal_id) ? 'checkbox-card checkbox-card-active' : 'checkbox-card'}>
                  <input type="checkbox" checked={grantPrincipalIds.includes(principal.principal_id)} onChange={() => toggleGrantPrincipal(principal.principal_id)} />
                  <span className="checkbox-copy">
                    <strong>{principalLabel(principal)}</strong>
                    <span>{principal.principal_type === 'group' ? groupPurposeLabel(groupPurpose(principal)) : principal.email_normalized}</span>
                  </span>
                </label>
              ))}
            </div>
          )}

          {accessWizard === 'grant' && accessWizardStep === 'actions' && (
            <div className="checkbox-list">
              {ACCESS_ACTIONS.map(action => (
                <label key={action.key} className={grantActions.includes(action.key) ? 'checkbox-card checkbox-card-active' : 'checkbox-card'}>
                  <input type="checkbox" checked={grantActions.includes(action.key)} onChange={() => toggleGrantAction(action.key)} />
                  <span className="checkbox-copy">
                    <strong>{action.label}</strong>
                    <span>{action.key === 'use' ? 'Invoke or query the resource' : action.key === 'manage' ? 'Modify settings or grants' : action.key === 'approve' ? 'Approve workflow requests' : 'Delete resource state'}</span>
                  </span>
                </label>
              ))}
            </div>
          )}

          {accessWizard === 'grant' && accessWizardStep === 'review' && (
            <div className="summary-list">
              <div className="summary-row"><span>Resource Type</span><strong>{accessResourceLabel(grantResourceType)}</strong></div>
              <div className="summary-row"><span>Selectors</span><strong>{formatWholeNumber(normalizedAccessSelectors(grantResourceSelectors).length)}</strong></div>
              <div className="summary-row"><span>Principals</span><strong>{formatWholeNumber(grantPrincipalIds.length)}</strong></div>
              <div className="summary-row"><span>Actions</span><strong>{shortList(grantActions)}</strong></div>
            </div>
          )}

          {accessWizard === 'user' && accessWizardStep === 'profile' && (
            <div className="field-stack">
              <label className="field">
                <span>User Email</span>
                <input value={manageUserEmail} onChange={event => setManageUserEmail(event.target.value)} placeholder="alex@example.com" />
              </label>
              <label className="field">
                <span>Display Name</span>
                <input value={manageUserDisplayName} onChange={event => setManageUserDisplayName(event.target.value)} placeholder="Alex Analyst" />
              </label>
              <SegmentedControl<'admin' | 'user' | 'pending'>
                ariaLabel="User system status"
                value={manageUserSystemRole}
                onChange={setManageUserSystemRole}
                options={[
                  { value: 'user', label: 'User' },
                  { value: 'admin', label: 'Admin' },
                  { value: 'pending', label: 'Pending' },
                ]}
              />
            </div>
          )}

          {accessWizard === 'user' && accessWizardStep === 'groups' && (
            <div className="checkbox-list">
              {accessGroups.map(group => (
                <label key={group.principal_id} className={manageUserGroupIds.includes(group.principal_id) ? 'checkbox-card checkbox-card-active' : 'checkbox-card'}>
                  <input type="checkbox" checked={manageUserGroupIds.includes(group.principal_id)} onChange={() => toggleManageUserGroup(group.principal_id)} />
                  <span className="checkbox-copy">
                    <strong>{principalLabel(group)}</strong>
                    <span>{groupPurposeLabel(groupPurpose(group))}</span>
                  </span>
                </label>
              ))}
              {accessGroups.length === 0 && <EmptyState title="No groups yet" body="Create a group first, or save the user without group membership." />}
            </div>
          )}

          {accessWizard === 'user' && accessWizardStep === 'preview' && (
            <div className="field-stack">
              <ActionBar>
                <ActionButton tone="secondary" onClick={() => void handleAccessWizardPreview()} disabled={!manageUserEmail.trim()}>Preview Current Access</ActionButton>
              </ActionBar>
              {accessWizardPreview ? <JsonInspector label={`Current access for ${accessWizardPreview.email}`} value={accessWizardPreview.access} /> : <EmptyState title="No preview yet" body="Preview the current email before saving the user or memberships." />}
            </div>
          )}

          {accessWizard === 'user' && accessWizardStep === 'review' && (
            <div className="summary-list">
              <div className="summary-row"><span>User</span><strong>{manageUserDisplayName || manageUserEmail || 'New user'}</strong></div>
              <div className="summary-row"><span>System Status</span><strong>{manageUserSystemRole}</strong></div>
              <div className="summary-row"><span>Groups</span><strong>{formatWholeNumber(manageUserGroupIds.length)}</strong></div>
            </div>
          )}

          {accessWizard === 'group' && accessWizardStep === 'details' && (
            <div className="field-stack">
              <label className="field">
                <span>Group Name</span>
                <input value={createGroupName} onChange={event => setCreateGroupName(event.target.value)} placeholder="Finance Analysts" />
              </label>
              <SegmentedControl<AccessGroupPurpose>
                ariaLabel="Create group purpose"
                value={createGroupPurpose}
                onChange={setCreateGroupPurpose}
                options={ACCESS_GROUP_PURPOSES.map(purpose => ({ value: purpose.key, label: purpose.label }))}
              />
            </div>
          )}

          {accessWizard === 'group' && accessWizardStep === 'role' && (
            <label className="field">
              <span>Optional Starting Role</span>
              <select value={createGroupRoleId} onChange={event => setCreateGroupRoleId(event.target.value)}>
                <option value="">No role yet</option>
                {accessRoles.map(role => <option key={role.role_id} value={role.role_id}>{role.name}</option>)}
              </select>
            </label>
          )}

          {accessWizard === 'group' && accessWizardStep === 'members' && (
            <div className="checkbox-list">
              {accessPrincipals.filter(principal => principal.principal_type === 'user').map(user => (
                <label key={user.principal_id} className={createGroupMemberIds.includes(user.principal_id) ? 'checkbox-card checkbox-card-active' : 'checkbox-card'}>
                  <input type="checkbox" checked={createGroupMemberIds.includes(user.principal_id)} onChange={() => toggleCreateGroupMember(user.principal_id)} />
                  <span className="checkbox-copy">
                    <strong>{principalLabel(user)}</strong>
                    <span>{user.email_normalized}</span>
                  </span>
                </label>
              ))}
            </div>
          )}

          {accessWizard === 'group' && accessWizardStep === 'review' && (
            <div className="summary-list">
              <div className="summary-row"><span>Group</span><strong>{createGroupName || 'New group'}</strong></div>
              <div className="summary-row"><span>Purpose</span><strong>{groupPurposeLabel(createGroupPurpose)}</strong></div>
              <div className="summary-row"><span>Starting Role</span><strong>{createGroupRoleId ? accessRoleById.get(createGroupRoleId)?.name || createGroupRoleId : 'None'}</strong></div>
              <div className="summary-row"><span>Members</span><strong>{formatWholeNumber(createGroupMemberIds.length)}</strong></div>
            </div>
          )}
        </div>
      </Dialog>

      {active === 'mcp' && (
        <div className="content-stack">
          <div className="card-grid">
            <SurfaceCard title="Connections" subtitle="User-owned Streamable HTTP MCP profiles. Catalog refreshes feed the runtime tool registry on the next request.">
              <div className="form-grid form-grid-compact">
                <label className="field">
                  <span>Display Name</span>
                  <input value={mcpDraftName} onChange={event => setMcpDraftName(event.target.value)} placeholder="github-tools" />
                </label>
                <label className="field">
                  <span>Server URL</span>
                  <input value={mcpDraftUrl} onChange={event => setMcpDraftUrl(event.target.value)} placeholder="https://mcp.example.com/mcp" />
                </label>
                <label className="field">
                  <span>Bearer Token</span>
                  <input type="password" value={mcpDraftSecret} onChange={event => setMcpDraftSecret(event.target.value)} placeholder="optional" />
                </label>
                <label className="field">
                  <span>Allowed Agents</span>
                  <input value={mcpDraftAgents} onChange={event => setMcpDraftAgents(event.target.value)} placeholder="general, coordinator" />
                </label>
                <label className="field">
                  <span>Visibility</span>
                  <select value={mcpDraftVisibility} onChange={event => setMcpDraftVisibility(event.target.value)}>
                    <option value="private">Private</option>
                    <option value="tenant">Tenant</option>
                  </select>
                </label>
              </div>
              <ActionBar>
                <ActionButton tone="primary" onClick={() => void handleCreateMcpConnection()}>Add MCP</ActionButton>
                <ActionButton tone="secondary" onClick={() => void refreshMcpData(selectedMcpConnection)}>Refresh List</ActionButton>
              </ActionBar>
              <ResourceSearch value={mcpSearch} onChange={setMcpSearch} placeholder="Search tools" />

              {mcpConnections.length > 0 ? (
                <EntityList
                  items={filteredMcpConnections}
                  selectedKey={selectedMcpConnection}
                  getKey={connection => connection.connection_id}
                  getLabel={connection => connection.display_name || connection.connection_slug}
                  getDescription={connection => connection.server_url}
                  getMeta={connection => (
                    <StatusBadge tone={toneForStatus(connection.status)}>{connection.status}</StatusBadge>
                  )}
                  onSelect={connection => setSelectedMcpConnection(connection.connection_id)}
                />
              ) : (
                <EmptyState title="No MCP connections" body="Add a Streamable HTTP MCP server to begin cataloging external tools." />
              )}
            </SurfaceCard>

            <SurfaceCard title={selectedMcpRecord?.display_name || 'Connection Detail'} subtitle="Connection health, ownership, and the cached tool catalog visible to deferred discovery.">
              {selectedMcpRecord ? (
                <div className="field-stack">
                  <div className="badge-cluster">
                    <StatusBadge tone={toneForStatus(selectedMcpRecord.status)}>{selectedMcpRecord.status}</StatusBadge>
                    <StatusBadge tone={selectedMcpRecord.secret_configured ? 'ok' : 'neutral'}>
                      {selectedMcpRecord.secret_configured ? 'Secret configured' : 'No secret'}
                    </StatusBadge>
                    <StatusBadge tone="neutral">{selectedMcpRecord.visibility}</StatusBadge>
                  </div>
                  <div className="summary-list">
                    <div className="summary-row">
                      <span>Registry Prefix</span>
                      <strong>mcp__{selectedMcpRecord.connection_slug}__*</strong>
                    </div>
                    <div className="summary-row">
                      <span>Allowed Agents</span>
                      <strong>{shortList(selectedMcpRecord.allowed_agents)}</strong>
                    </div>
                    <div className="summary-row">
                      <span>Last Tested</span>
                      <strong>{formatTimestamp(selectedMcpRecord.last_tested_at)}</strong>
                    </div>
                    <div className="summary-row">
                      <span>Last Catalog Refresh</span>
                      <strong>{formatTimestamp(selectedMcpRecord.last_refreshed_at)}</strong>
                    </div>
                  </div>
                  <ActionBar>
                    <ActionButton tone="secondary" onClick={() => void handleTestMcpConnection(selectedMcpRecord.connection_id)}>Test</ActionButton>
                    <ActionButton tone="secondary" onClick={() => void handleRefreshMcpTools(selectedMcpRecord.connection_id)}>Refresh Tools</ActionButton>
                    <ActionButton
                      tone="destructive"
                      onClick={() => askConfirm({
                        title: 'Disable this MCP connection?',
                        description: 'The catalog stays for audit, but tools stop appearing in runtime discovery.',
                        confirmLabel: 'Disable',
                        run: () => handleDisableMcpConnection(selectedMcpRecord.connection_id),
                      })}
                    >
                      Disable
                    </ActionButton>
                  </ActionBar>
                  <JsonInspector label="Health" value={selectedMcpRecord.health} />
                </div>
              ) : (
                <EmptyState title="Select a connection" body="Connection detail appears after a profile is created or selected." />
              )}
            </SurfaceCard>
          </div>

          <SurfaceCard title="Tool Catalog" subtitle="External tools default to deferred, non-read-only, destructive, and background-blocked until reviewed.">
            {selectedMcpRecord && (selectedMcpRecord.tools ?? []).length > 0 ? (
              <div className="field-stack">
                {(selectedMcpRecord.tools ?? []).map(tool => (
                  <div key={tool.tool_id} className="preview-card">
                    <div className="tool-card-head">
                      <div>
                        <strong>{tool.registry_name}</strong>
                        <p>{tool.description || tool.raw_tool_name}</p>
                      </div>
                      <div className="badge-cluster">
                        <StatusBadge tone={tool.enabled ? 'ok' : 'warning'}>{tool.enabled ? 'Enabled' : 'Disabled'}</StatusBadge>
                        <StatusBadge tone={tool.read_only ? 'ok' : 'danger'}>{tool.read_only ? 'Read-only' : 'Mutating'}</StatusBadge>
                        <StatusBadge tone={tool.should_defer ? 'accent' : 'neutral'}>{tool.should_defer ? 'Deferred' : 'Eager'}</StatusBadge>
                      </div>
                    </div>
                    <div className="form-grid form-grid-compact">
                      <label className="inline-check">
                        <input
                          type="checkbox"
                          checked={tool.enabled}
                          onChange={event => void handleToggleMcpTool(selectedMcpRecord.connection_id, tool.tool_id, event.target.checked)}
                        />
                        <span>Enabled</span>
                      </label>
                      <label className="inline-check">
                        <input
                          type="checkbox"
                          checked={tool.read_only}
                          onChange={event => void handleToggleMcpToolReadOnly(selectedMcpRecord.connection_id, tool.tool_id, event.target.checked)}
                        />
                        <span>Read-only</span>
                      </label>
                    </div>
                    <JsonInspector label="Input schema" value={tool.input_schema} />
                  </div>
                ))}
              </div>
            ) : (
              <EmptyState title="No tools cataloged" body="Test and refresh the selected MCP connection to cache its tools." />
            )}
          </SurfaceCard>
        </div>
      )}

      {active === 'operations' && (
        <div className="content-stack">
          {operationsTab === 'reloads' && (
            <SurfaceCard title="Last Reload" subtitle="Most recent runtime swap, prompt reset, or agent reload event.">
              {lastReload ? (
                <div className="summary-list">
                  <div className="summary-row">
                    <span>Status</span>
                    <StatusBadge tone={toneForStatus(lastReload.status)}>{asString(lastReload.status, 'unknown')}</StatusBadge>
                  </div>
                  <div className="summary-row">
                    <span>Reason</span>
                    <strong>{asString(lastReload.reason, 'startup')}</strong>
                  </div>
                  <div className="summary-row">
                    <span>Actor</span>
                    <strong>{asString(lastReload.actor, 'system')}</strong>
                  </div>
                  <div className="summary-row">
                    <span>When</span>
                    <strong>{formatTimestamp(lastReload.timestamp)}</strong>
                  </div>
                </div>
              ) : (
                <EmptyState title="No reload activity yet" body="Reload events will appear here after startup or any control-plane apply action." />
              )}
              {operations && <JsonInspector label="Technical details" value={operations} />}
            </SurfaceCard>
          )}

          {operationsTab === 'jobs' && (
            <div className="content-stack">
              <div className="architecture-traffic-grid">
                <SurfaceCard title="Scheduler Health" subtitle="Live queue pressure, urgent-slot reservation, and budget-blocking status from the worker scheduler.">
                  {schedulerSummary ? (
                    <div className="field-stack">
                      <div className="summary-row">
                        <span>Running jobs</span>
                        <strong>{formatWholeNumber(schedulerSummary.running_jobs)}</strong>
                      </div>
                      <div className="summary-row">
                        <span>Available slots</span>
                        <strong>{formatWholeNumber(schedulerSummary.available_slots)}</strong>
                      </div>
                      <div className="summary-row">
                        <span>Reserved urgent slots</span>
                        <strong>{formatWholeNumber(schedulerSummary.reserved_urgent_slots)}</strong>
                      </div>
                      <div className="summary-row">
                        <span>Budget blocked jobs</span>
                        <strong>{formatWholeNumber(schedulerSummary.budget_blocked_jobs)}</strong>
                      </div>
                      <div className="summary-row">
                        <span>Urgent backlog</span>
                        <strong>{Boolean(schedulerSummary.urgent_backlog) ? 'Yes' : 'No'}</strong>
                      </div>
                    </div>
                  ) : (
                    <EmptyState title="No scheduler snapshot" body="The backend has not returned worker-scheduler state for this runtime yet." />
                  )}
                </SurfaceCard>

                <SurfaceCard title="Queue Depths" subtitle="Per-class queue depth and oldest wait time, useful for spotting starvation or tenant contention.">
                  {schedulerSummary && asRecord(schedulerSummary.queue_depths) ? (
                    <div className="field-stack">
                      {Object.entries(asRecord(schedulerSummary.queue_depths) ?? {}).map(([queueClass, count]) => (
                        <div key={queueClass} className="summary-row">
                          <span>{humanizeKey(queueClass)}</span>
                          <strong>
                            {formatWholeNumber(count)} queued / {formatWholeNumber(asRecord(schedulerSummary.oldest_wait_seconds)?.[queueClass])}s oldest
                          </strong>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <EmptyState title="No queue data yet" body="Queue-depth metrics will appear here once worker jobs start flowing through the scheduler." />
                  )}
                </SurfaceCard>
              </div>

              <SurfaceCard title="Background Jobs" subtitle="Active or recent jobs surfaced as concise cards instead of a single payload blob.">
                {jobs.length > 0 ? (
                  <div className="field-stack">
                    {jobs.map((job, index) => (
                      <div key={`${asString(job.job_id)}-${index}`} className="preview-card">
                        <div className="tool-card-head">
                          <strong>{asString(job.job_id, `job-${index + 1}`)}</strong>
                          {(() => {
                            const raw = asString(job.scheduler_state || job.status, 'unknown')
                            const help = statusHelp(raw)
                            const badge = (
                              <StatusBadge tone={toneForStatus(job.scheduler_state || job.status)}>
                                {raw}
                              </StatusBadge>
                            )
                            return help ? <Tooltip content={help}>{badge}</Tooltip> : badge
                          })()}
                        </div>
                        <div className="summary-list">
                          <div className="summary-row">
                            <span>Agent</span>
                            <strong>{asString(job.agent_name, 'background')}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Priority</span>
                            <strong>{asString(job.priority, 'interactive')}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Queue Class</span>
                            <strong>{asString(job.queue_class, 'interactive')}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Estimated Tokens</span>
                            <strong>{formatWholeNumber(job.estimated_token_cost)}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Actual Tokens</span>
                            <strong>{formatWholeNumber(job.actual_token_cost)}</strong>
                          </div>
                          <div className="summary-row">
                            <span>Updated</span>
                            <strong>{formatTimestamp(job.updated_at)}</strong>
                          </div>
                        </div>
                        {asString(job.description) && <p>{asString(job.description)}</p>}
                        {asString(job.budget_block_reason) && (
                          <div className="inline-alert">
                            <span>Budget block</span>
                            <strong>{humanizeKey(asString(job.budget_block_reason))}</strong>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <EmptyState title="No background jobs" body="The runtime is currently idle from the control panel perspective." />
                )}
                {schedulerSummary && asArray<Record<string, unknown>>(schedulerSummary.tenant_budget_health).length > 0 && (
                  <div className="field-stack">
                    {asArray<Record<string, unknown>>(schedulerSummary.tenant_budget_health).map(tenant => (
                      <div key={asString(tenant.tenant_id)} className="summary-row">
                        <span>{asString(tenant.tenant_id, 'tenant')}</span>
                        <strong>
                          {formatWholeNumber(tenant.queued_jobs)} queued / {formatWholeNumber(tenant.budget_blocked_jobs)} blocked / {formatWholeNumber(tenant.available_tokens)} tokens
                        </strong>
                      </div>
                    ))}
                  </div>
                )}
                {operations && <JsonInspector label="Technical details" value={operations} />}
              </SurfaceCard>
            </div>
          )}

          {operationsTab === 'audit' && (
            <SurfaceCard title="Audit Stream" subtitle="Recent operator-visible events, with reasons and changed keys called out up front.">
              <div className="audit-toolbar">
                <SegmentedControl<'24h' | '7d' | '30d' | 'all'>
                  size="sm"
                  ariaLabel="Audit date range"
                  value={auditRange}
                  onChange={setAuditRange}
                  options={[
                    { value: '24h', label: 'Last 24h' },
                    { value: '7d', label: 'Last 7d' },
                    { value: '30d', label: 'Last 30d' },
                    { value: 'all', label: 'All' },
                  ]}
                />
              </div>
              {(() => {
                const now = Date.now()
                const cutoffMs = auditRange === '24h' ? 24 * 3600 * 1000
                  : auditRange === '7d' ? 7 * 24 * 3600 * 1000
                  : auditRange === '30d' ? 30 * 24 * 3600 * 1000
                  : null
                const filtered = cutoffMs === null
                  ? auditEvents
                  : auditEvents.filter(event => {
                    const ts = Date.parse(asString(event.timestamp, ''))
                    return Number.isFinite(ts) && (now - ts) <= cutoffMs
                  })
                return (
                  <DataTable<Record<string, unknown>>
                    ariaLabel="Audit events"
                    rows={filtered}
                    rowKey={(event) => `${asString(event.action)}-${asString(event.timestamp)}-${asString(event.actor)}`}
                    pageSize={25}
                    columns={[
                      {
                        key: 'timestamp',
                        header: 'Timestamp',
                        sortable: true,
                        accessor: event => asString(event.timestamp),
                        render: event => <span className="tabular-nums">{formatTimestamp(event.timestamp)}</span>,
                      },
                      {
                        key: 'actor',
                        header: 'Actor',
                        sortable: true,
                        accessor: event => asString(event.actor, 'system'),
                        render: event => <StatusBadge tone="neutral">{asString(event.actor, 'system')}</StatusBadge>,
                      },
                      {
                        key: 'action',
                        header: 'Action',
                        sortable: true,
                        accessor: event => asString(event.action),
                        render: event => <strong>{asString(event.action, 'event')}</strong>,
                      },
                      {
                        key: 'changed_keys',
                        header: 'Changed keys',
                        render: event => <span>{shortList(asArray<string>(event.changed_keys)) || '—'}</span>,
                      },
                      {
                        key: 'details',
                        header: '',
                        width: 80,
                        align: 'right',
                        render: event => (
                          <Popover
                            ariaLabel="Audit event details"
                            trigger={({ toggle, ref }) => (
                              <button
                                type="button"
                                ref={el => ref(el)}
                                onClick={toggle}
                                className="icon-btn icon-btn-ghost"
                                aria-label="Show audit event details"
                              >
                                Details
                              </button>
                            )}
                          >
                            <div className="audit-popover">
                              <strong>{asString(event.action, 'event')}</strong>
                              <JsonInspector label="Payload" value={event} />
                            </div>
                          </Popover>
                        ),
                      },
                    ]}
                    emptyState={<EmptyState title="No audit events" body={auditRange === 'all' ? 'Apply config, reload agents, or ingest content to start filling the audit stream.' : 'Nothing matches the selected time range.'} />}
                  />
                )
              })()}
            </SurfaceCard>
          )}
        </div>
      )}
      <Dialog
        open={ingestionWizardOpen}
        onClose={() => setIngestionWizardOpen(false)}
        title="Ingestion Wizard"
        description="Create or select a collection, choose the source workflow, and optionally prepare a GraphRAG graph from the same setup."
        size="lg"
        footer={(
          <>
            <ActionButton
              tone="ghost"
              onClick={() => {
                const index = Math.max(0, INGESTION_WIZARD_STEPS.indexOf(ingestionWizardStep) - 1)
                setIngestionWizardStep(INGESTION_WIZARD_STEPS[index])
              }}
              disabled={ingestionWizardStep === 'collection'}
            >
              Back
            </ActionButton>
            {ingestionWizardStep !== 'review' ? (
              <ActionButton
                tone="primary"
                onClick={() => {
                  const index = Math.min(INGESTION_WIZARD_STEPS.length - 1, INGESTION_WIZARD_STEPS.indexOf(ingestionWizardStep) + 1)
                  setIngestionWizardStep(INGESTION_WIZARD_STEPS[index])
                }}
              >
                Next
              </ActionButton>
            ) : (
              <ActionButton tone="primary" onClick={() => void handleWizardFinish()}>
                Finish
              </ActionButton>
            )}
          </>
        )}
      >
        <div className="wizard-shell">
          <div className="stage-list">
            {INGESTION_WIZARD_STEPS.map(step => (
              <button
                key={step}
                type="button"
                className={`stage-chip ${ingestionWizardStep === step ? 'stage-chip-active' : 'stage-chip-pending'}`}
                onClick={() => setIngestionWizardStep(step)}
              >
                {humanizeKey(step)}
              </button>
            ))}
          </div>

          {ingestionWizardStep === 'collection' && (
            <div className="field-stack">
              <label className="field">
                <span>Collection ID</span>
                <input
                  aria-label="Wizard Collection ID"
                  value={wizardCollectionId}
                  onChange={event => {
                    const next = normalizeCollectionId(event.target.value)
                    setWizardCollectionId(next)
                    setCollectionDraft(next)
                    setGraphCollectionId(next)
                  }}
                  placeholder="vendor-risk"
                />
              </label>
              <div className="checkbox-list">
                {collections.slice(0, 6).map(collection => (
                  <label key={collection.collection_id} className={wizardCollectionId === collection.collection_id ? 'checkbox-card checkbox-card-active' : 'checkbox-card'}>
                    <input
                      type="radio"
                      name="wizard-collection"
                      checked={wizardCollectionId === collection.collection_id}
                      onChange={() => {
                        setWizardCollectionId(collection.collection_id)
                        setCollectionDraft(collection.collection_id)
                        setGraphCollectionId(collection.collection_id)
                      }}
                    />
                    <span className="checkbox-copy">
                      <strong>{collection.collection_id}</strong>
                      <span>{formatWholeNumber(collection.document_count)} documents, {formatWholeNumber(collection.graph_count)} graphs</span>
                    </span>
                  </label>
                ))}
              </div>
            </div>
          )}

          {ingestionWizardStep === 'source' && (
            <div className="field-stack">
              <input
                ref={wizardFilesInputRef}
                type="file"
                multiple
                hidden
                onChange={event => {
                  void handleCollectionFilesUpload(event.target.files)
                  event.currentTarget.value = ''
                }}
              />
              <input
                ref={wizardFolderInputRef}
                type="file"
                multiple
                hidden
                // @ts-expect-error webkitdirectory is still the browser-supported folder picker attribute.
                webkitdirectory="true"
                onChange={event => {
                  void handleCollectionFilesUpload(event.target.files)
                  event.currentTarget.value = ''
                }}
              />
              <SectionTabs
                tabs={[
                  { id: 'upload', label: 'Upload Files' },
                  { id: 'local', label: 'Local Source' },
                  { id: 'registered', label: 'Registered Source' },
                  { id: 'sync', label: 'Sync Existing' },
                ]}
                active={collectionAction}
                onChange={value => setCollectionAction(value as CollectionActionMode)}
                ariaLabel="Wizard source mode"
              />

              {collectionAction === 'upload' && (
                <div className="collection-action-panel collection-action-panel-compact">
                  <div className="collection-action-copy">
                    <strong>Upload files or a folder from this browser</strong>
                    <p>Use this for PDFs, Office files, markdown, text, CSVs, spreadsheets, and OCR-friendly images. No path is needed here; the browser sends selected files to the backend.</p>
                  </div>
                  <div className="form-grid form-grid-compact">
                    <label className="field">
                      <span>Indexing Profile</span>
                      <select value={metadataProfile} onChange={event => setMetadataProfile(event.target.value)}>
                        <option value="auto">Auto</option>
                        <option value="deterministic">Deterministic</option>
                        <option value="basic">Basic</option>
                        <option value="off">Off</option>
                      </select>
                    </label>
                    <label className="checkbox-card">
                      <input
                        type="checkbox"
                        checked={indexPreview}
                        onChange={event => setIndexPreview(event.target.checked)}
                      />
                      <span className="checkbox-copy">
                        <strong>Preview only</strong>
                        <span>Inspect metadata before writing documents.</span>
                      </span>
                    </label>
                  </div>
                  <ActionBar>
                    <ActionButton tone="primary" onClick={() => wizardFilesInputRef.current?.click()}>Upload Files</ActionButton>
                    <ActionButton tone="secondary" onClick={() => wizardFolderInputRef.current?.click()}>Upload Folder</ActionButton>
                  </ActionBar>
                </div>
              )}

              {collectionAction === 'local' && (
                <div className="collection-action-panel">
                  <div className="collection-action-copy">
                    <strong>Index files that already exist on the API server</strong>
                    <p>Enter absolute paths the backend process can read, one per line. These are server paths, not URLs and not necessarily paths from your browser machine.</p>
                  </div>
                  <SegmentedControl<KnowledgeSourceKind>
                    ariaLabel="Wizard local source type"
                    value={knowledgeSourceKind}
                    onChange={setKnowledgeSourceKind}
                    options={[
                      { value: 'local_folder', label: 'Folder' },
                      { value: 'local_repo', label: 'Repository' },
                    ]}
                  />
                  <label className="field">
                    <span>Server-Readable Local Paths</span>
                    <textarea
                      aria-label="Wizard Server-Readable Local Paths"
                      rows={5}
                      value={pathDraft}
                      onChange={event => setPathDraft(event.target.value)}
                      placeholder={'/Users/shivbalodi/Desktop/Rag_Research/source_docs\n/Users/shivbalodi/Desktop/Rag_Research/another_corpus'}
                    />
                  </label>
                  <div className="collection-action-copy">
                    <span className="field-label">Allowed Roots</span>
                    <div className="badge-cluster">
                      {allowedSourceRoots.length > 0 ? allowedSourceRoots.slice(0, 4).map(root => (
                        <StatusBadge key={root} tone="neutral">{root}</StatusBadge>
                      )) : (
                        <StatusBadge tone="warning">No allowed roots reported</StatusBadge>
                      )}
                      {allowedSourceRoots.length > 4 && <StatusBadge tone="neutral">+{allowedSourceRoots.length - 4} more</StatusBadge>}
                    </div>
                  </div>
                  <div className="form-grid form-grid-compact">
                    <label className="field">
                      <span>Include Globs</span>
                      <textarea
                        aria-label="Wizard Include Globs"
                        rows={4}
                        value={sourceIncludeGlobs}
                        onChange={event => setSourceIncludeGlobs(event.target.value)}
                        placeholder={'docs/**\n*.md'}
                      />
                    </label>
                    <label className="field">
                      <span>Exclude Globs</span>
                      <textarea
                        aria-label="Wizard Exclude Globs"
                        rows={4}
                        value={sourceExcludeGlobs}
                        onChange={event => setSourceExcludeGlobs(event.target.value)}
                        placeholder={'node_modules/**\n.git/**'}
                      />
                    </label>
                  </div>
                  <ActionBar>
                    <ActionButton tone="secondary" onClick={() => void handleSourceScan()}>Preview Scan</ActionButton>
                    <ActionButton tone="ghost" onClick={() => void handleRegisterSource()}>Register Source</ActionButton>
                    <ActionButton tone="primary" onClick={() => void handleIndexLocalSource()}>Index Source</ActionButton>
                  </ActionBar>
                </div>
              )}

              {collectionAction === 'registered' && (
                <div className="collection-action-panel">
                  <div className="collection-action-copy">
                    <strong>Refresh a saved folder or repository source</strong>
                    <p>Registered sources remember their server-readable paths and include/exclude rules, so users can refresh changed files without retyping paths.</p>
                  </div>
                  {filteredRegisteredSources.length > 0 ? (
                    <EntityList
                      items={filteredRegisteredSources}
                      selectedKey={visibleRegisteredSource?.source_id ?? selectedSourceId}
                      getKey={source => source.source_id}
                      getLabel={source => source.display_name}
                      getDescription={source => `${source.source_kind} -> ${source.collection_id}`}
                      getMeta={source => (
                        <>
                          <StatusBadge tone="neutral">{source.source_kind}</StatusBadge>
                          <span>{source.last_scan?.summary.supported_count ?? 0} files</span>
                        </>
                      )}
                      onSelect={source => {
                        setSelectedSourceId(source.source_id)
                        if (source.last_scan) setSourceScan(source.last_scan)
                      }}
                    />
                  ) : (
                    <EmptyState title="No registered sources" body="Register a local folder or repository first, then use this mode to preview drift or refresh it." />
                  )}
                  <ActionBar>
                    <ActionButton tone="secondary" onClick={() => void handleRefreshSource(true)} disabled={!visibleRegisteredSource}>Preview Drift</ActionButton>
                    <ActionButton tone="primary" onClick={() => void handleRefreshSource(false)} disabled={!visibleRegisteredSource}>Refresh Now</ActionButton>
                    <ActionButton tone="ghost" onClick={() => void handleRefreshSource(false, true)} disabled={!visibleRegisteredSource}>Queue Refresh</ActionButton>
                  </ActionBar>
                </div>
              )}

              {collectionAction === 'sync' && (
                <div className="collection-action-panel collection-action-panel-compact">
                  <div className="collection-action-copy">
                    <strong>Sync runtime-configured KB sources</strong>
                    <p>Use this only when the collection should mirror KB source roots configured on the backend. It does not upload arbitrary files or scan a new typed path.</p>
                  </div>
                  <ActionButton tone="primary" onClick={() => void handleSyncCollection()}>
                    Sync Configured Sources
                  </ActionButton>
                </div>
              )}
            </div>
          )}

          {ingestionWizardStep === 'graph' && (
            <div className="field-stack">
              <label className="inline-check">
                <input
                  type="checkbox"
                  checked={wizardCreateGraph}
                  onChange={event => setWizardCreateGraph(event.target.checked)}
                />
                <span>Create a graph draft for this collection</span>
              </label>
              <label className="inline-check">
                <input
                  type="checkbox"
                  checked={wizardStartBuild}
                  onChange={event => setWizardStartBuild(event.target.checked)}
                  disabled={!wizardCreateGraph}
                />
                <span>Start the graph build after creating the draft</span>
              </label>
              <label className="field">
                <span>Graph Intent</span>
                <select value={graphIntent} onChange={event => setGraphIntent(event.target.value)}>
                  <option value="general">General Knowledge</option>
                  <option value="vendor_risk">Vendor Risk</option>
                  <option value="requirements">Requirements</option>
                  <option value="policy">Policy & Controls</option>
                  <option value="research">Research Corpus</option>
                </select>
              </label>
            </div>
          )}

          {ingestionWizardStep === 'tuning' && (
            <div className="field-stack">
              {wizardCreateGraph ? (
                <>
                  <label className="inline-check">
                    <input
                      type="checkbox"
                      checked={wizardRunTune}
                      onChange={event => setWizardRunTune(event.target.checked)}
                    />
                    <span>Run prompt tuning before the graph build</span>
                  </label>
                  <label className="inline-check">
                    <input
                      type="checkbox"
                      checked={wizardApplyTune}
                      onChange={event => setWizardApplyTune(event.target.checked)}
                      disabled={!wizardRunTune || !graphTuneResult?.run_id || graphTuneSelectedPrompts.length === 0}
                    />
                    <span>Apply selected prompt drafts before build</span>
                  </label>
                  <label className="inline-check">
                    <input
                      type="checkbox"
                      checked={wizardRequireTuneBeforeBuild}
                      onChange={event => setWizardRequireTuneBeforeBuild(event.target.checked)}
                      disabled={!wizardRunTune}
                    />
                    <span>Require prompt tuning to succeed before starting build</span>
                  </label>
                  <label className="field">
                    <span>Research Guidance</span>
                    <textarea
                      aria-label="Wizard Prompt Tuning Guidance"
                      rows={4}
                      value={graphTuneGuidance}
                      onChange={event => setGraphTuneGuidance(event.target.value)}
                      placeholder="Focus extraction on policy controls, owners, exceptions, approval chains, and relationship evidence."
                    />
                  </label>
                  <div className="field-stack">
                    <span className="field-label">Prompt Targets</span>
                    <div className="checkbox-list">
                      {GRAPH_RESEARCH_TUNE_TARGETS.map(promptFile => {
                        const checked = graphTuneTargets.includes(promptFile)
                        return (
                          <label key={promptFile} className={checked ? 'checkbox-card checkbox-card-active' : 'checkbox-card'}>
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => toggleGraphTuneTarget(promptFile)}
                            />
                            <span className="checkbox-copy">
                              <strong>{promptFile}</strong>
                              <span>{checked ? 'Included' : 'Skipped'}</span>
                            </span>
                          </label>
                        )
                      })}
                    </div>
                  </div>
                  <ActionBar>
                    <ActionButton tone="primary" onClick={() => void handleWizardRunTune()} disabled={!wizardRunTune || graphTuneRunning}>
                      {graphTuneRunning ? 'Running Research & Tune' : 'Run Research & Tune'}
                    </ActionButton>
                  </ActionBar>
                  {graphTuneResult ? (
                    <div className="preview-card">
                      <div className="tool-card-head">
                        <strong>Prompt tuning result</strong>
                        <StatusBadge tone={toneForStatus(graphTuneResult.status)}>{asString(graphTuneResult.status, 'completed')}</StatusBadge>
                      </div>
                      <div className="summary-list">
                        <div className="summary-row">
                          <span>Run</span>
                          <strong>{shortId(asString(graphTuneResult.run_id, 'run'))}</strong>
                        </div>
                        <div className="summary-row">
                          <span>Valid drafts selected</span>
                          <strong>{formatWholeNumber(graphTuneSelectedPrompts.length)}</strong>
                        </div>
                        <div className="summary-row">
                          <span>Prompt drafts</span>
                          <strong>{formatWholeNumber(Object.keys(graphTuneResult.prompt_drafts ?? {}).length)}</strong>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <EmptyState title="Prompt tuning is optional" body="Enable tuning and run it here to draft GraphRAG prompt overrides before building the graph." />
                  )}
                </>
              ) : (
                <EmptyState title="Prompt tuning skipped" body="Prompt tuning requires a graph draft. Enable graph draft creation in the previous step to tune GraphRAG prompts." />
              )}
            </div>
          )}

          {ingestionWizardStep === 'review' && (
            <div className="summary-list">
              <div className="summary-row">
                <span>Collection</span>
                <strong>{wizardCollectionId || 'Not selected'}</strong>
              </div>
              <div className="summary-row">
                <span>Source Workflow</span>
                <strong>{humanizeKey(collectionAction)}</strong>
              </div>
              <div className="summary-row">
                <span>Graph Draft</span>
                <strong>{wizardCreateGraph ? humanizeKey(graphIntent) : 'Skipped'}</strong>
              </div>
              <div className="summary-row">
                <span>Build</span>
                <strong>{wizardStartBuild && wizardCreateGraph ? 'Start after draft creation' : 'Manual start later'}</strong>
              </div>
              <div className="summary-row">
                <span>Prompt Tuning</span>
                <strong>{wizardCreateGraph && wizardRunTune ? `${wizardApplyTune ? 'Run and apply selected drafts' : 'Run only'}` : 'Skipped'}</strong>
              </div>
            </div>
          )}
        </div>
      </Dialog>
      <ConfirmDialog
        open={pendingConfirm !== null}
        title={pendingConfirm?.title ?? ''}
        description={pendingConfirm?.description}
        confirmLabel={pendingConfirm?.confirmLabel ?? 'Confirm'}
        tone="destructive"
        loading={confirmLoading}
        onCancel={() => { if (!confirmLoading) setPendingConfirm(null) }}
        onConfirm={handleConfirmRun}
      />
      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        commands={paletteCommands}
      />
      <Dialog
        open={shortcutsOpen}
        onClose={() => setShortcutsOpen(false)}
        title="Keyboard shortcuts"
        description="All keyboard commands available in the control panel."
        size="md"
        footer={<ActionButton tone="ghost" onClick={() => setShortcutsOpen(false)}>Close</ActionButton>}
      >
        <div className="shortcuts-groups">
          <section>
            <h4>Navigation</h4>
            <dl className="shortcuts-list">
              <div><dt>Open command palette</dt><dd><Kbd>{isMacPlatform ? '⌘' : 'Ctrl'}</Kbd><Kbd>K</Kbd></dd></div>
              <div><dt>Keyboard shortcuts</dt><dd><Kbd>?</Kbd></dd></div>
              <div><dt>Close dialog / menu</dt><dd><Kbd>Esc</Kbd></dd></div>
            </dl>
          </section>
          <section>
            <h4>Command palette</h4>
            <dl className="shortcuts-list">
              <div><dt>Move selection</dt><dd><Kbd>↑</Kbd><Kbd>↓</Kbd></dd></div>
              <div><dt>Run selected</dt><dd><Kbd>↵</Kbd></dd></div>
              <div><dt>Close palette</dt><dd><Kbd>Esc</Kbd></dd></div>
            </dl>
          </section>
          <section>
            <h4>Tables &amp; menus</h4>
            <dl className="shortcuts-list">
              <div><dt>Move focus</dt><dd><Kbd>Tab</Kbd></dd></div>
              <div><dt>Activate control</dt><dd><Kbd>Space</Kbd> / <Kbd>↵</Kbd></dd></div>
              <div><dt>Navigate menu items</dt><dd><Kbd>↑</Kbd><Kbd>↓</Kbd></dd></div>
            </dl>
          </section>
        </div>
      </Dialog>
    </AppShell>
  )
}
