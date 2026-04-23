import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { api, isApiError } from './api'
import { buildArchitectureMapLayout } from './architectureLayout'
import {
  ActionBar,
  ActionButton,
  AppShell,
  CollapsibleSurfaceCard,
  ConfirmDialog,
  DetailTabs,
  EmptyState,
  EntityList,
  IconButton,
  JsonInspector,
  SectionHeader,
  SectionIcon,
  SectionTabs,
  SidebarNav,
  StatCard,
  StatusBadge,
  SurfaceCard,
  Tooltip,
  useToast,
} from './components/ui'
import { useTheme } from './theme/ThemeProvider'
import { useDensity } from './theme/DensityProvider'
import { statusHelp } from './statusCopy'
import type {
  AccessMembership,
  AccessPrincipal,
  AccessRole,
  AccessRoleBinding,
  AccessRolePermission,
  AdminField,
  AdminOverview,
  ArchitectureActivity,
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
  GraphDetailPayload,
  GraphIndexRecord,
  GraphIndexRunRecord,
  GraphResearchTunePayload,
  McpConnectionRecord,
} from './types'

type Section = 'dashboard' | 'architecture' | 'config' | 'agents' | 'prompts' | 'collections' | 'graphs' | 'skills' | 'access' | 'mcp' | 'operations'

const SECTION_META: Array<{
  id: Section
  label: string
  eyebrow: string
  description: string
}> = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    eyebrow: 'Executive Console',
    description: 'Monitor the runtime posture, collection inventory, and recent operational activity at a glance.',
  },
  {
    id: 'architecture',
    label: 'Architecture',
    eyebrow: 'System Map',
    description: 'See the live routing model, active agent topology, and recent pathing activity as the runtime changes.',
  },
  {
    id: 'config',
    label: 'Config',
    eyebrow: 'Runtime Controls',
    description: 'Validate and apply live-safe environment changes without losing sight of current values or reload impact.',
  },
  {
    id: 'agents',
    label: 'Agents',
    eyebrow: 'Agent Studio',
    description: 'Edit agent overlays, inspect pinned skills and tool access, then reload definitions when you are ready.',
  },
  {
    id: 'prompts',
    label: 'Prompts',
    eyebrow: 'Prompt Layers',
    description: 'Work with base prompts and live overlays in a safer editor that makes the effective prompt obvious.',
  },
  {
    id: 'collections',
    label: 'Collections',
    eyebrow: 'Knowledge Operations',
    description: 'Create namespaces, ingest source files or folders, and inspect collection readiness without leaving the workspace.',
  },
  {
    id: 'graphs',
    label: 'Graphs',
    eyebrow: 'Graph Workspace',
    description: 'Admin-managed GraphRAG control plane for named graphs, build validation, prompt overrides, and graph-bound skill overlays.',
  },
  {
    id: 'skills',
    label: 'Skills',
    eyebrow: 'Skill Library',
    description: 'Preview relevance, create reusable skills, and control whether each skill is active or archived.',
  },
  {
    id: 'access',
    label: 'Access',
    eyebrow: 'RBAC Workspace',
    description: 'Manage email users, placeholder groups, roles, bindings, and effective access for collections, graphs, tools, and skill families.',
  },
  {
    id: 'mcp',
    label: 'MCP',
    eyebrow: 'Plugin Tool Plane',
    description: 'Connect Streamable HTTP MCP servers, refresh tool catalogs, and keep external tools behind the same runtime policies.',
  },
  {
    id: 'operations',
    label: 'Operations',
    eyebrow: 'Ops Stream',
    description: 'Review reload history, background job state, and audit activity with the technical payload tucked away.',
  },
]

const SECTION_IDS: ReadonlySet<Section> = new Set(SECTION_META.map(s => s.id))
const DEFAULT_SECTION: Section = 'dashboard'

function parseSectionFromPath(pathname: string): Section {
  const first = pathname.replace(/^\/+/, '').split('/')[0] ?? ''
  return SECTION_IDS.has(first as Section) ? (first as Section) : DEFAULT_SECTION
}

function useSectionRoute(): [Section, (next: Section) => void] {
  const location = useLocation()
  const navigate = useNavigate()
  const active = parseSectionFromPath(location.pathname)
  const setActive = useCallback(
    (next: Section) => {
      if (next === parseSectionFromPath(location.pathname)) return
      navigate(`/${next}`)
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

type ArchitectureTab = 'map' | 'paths' | 'traffic'
type AgentsTab = 'workspace' | 'catalog'
type PromptsTab = 'edit' | 'compare'
type CollectionsTab = 'workspace'
type CollectionActionMode = 'host' | 'files' | 'folder' | 'sync'
type GraphsTab = 'workspace' | 'runs'
type GraphSourceMode = 'collection' | 'manual'
type SkillsTab = 'editor' | 'preview'
type AccessResourceType = 'collection' | 'graph' | 'tool' | 'skill_family'
type OperationsTab = 'reloads' | 'jobs' | 'audit'

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
  if (['overlay active', 'overlay', 'live'].includes(normalized)) return 'accent'
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

function accessResourceLabel(resourceType: string): string {
  const labels: Record<string, string> = {
    collection: 'Collection',
    graph: 'Graph',
    tool: 'Tool',
    skill_family: 'Skill Family',
  }
  return labels[resourceType] ?? humanizeKey(resourceType)
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
  return collection?.storage_profile ?? EMPTY_COLLECTION_STORAGE_PROFILE
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

function useSessionBooleanState(key: string, initialValue: boolean): [boolean, (nextValue: boolean) => void] {
  const [value, setValueState] = useState<boolean>(() => {
    const stored = sessionStorage.getItem(key)
    if (stored === 'true') return true
    if (stored === 'false') return false
    return initialValue
  })
  const setValue = (nextValue: boolean) => {
    sessionStorage.setItem(key, String(nextValue))
    setValueState(nextValue)
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
  const [selectedArchitecturePathId, setSelectedArchitecturePathId] = useState('')
  const [configFields, setConfigFields] = useState<AdminField[]>([])
  const [configEffective, setConfigEffective] = useState<Record<string, string>>({})
  const [configChanges, setConfigChanges] = useState<Record<string, string>>({})
  const [configPreview, setConfigPreview] = useState<ConfigValidationResult | null>(null)
  const [activeConfigGroup, setActiveConfigGroup] = useState('')
  const [agentsPayload, setAgentsPayload] = useState<{ agents: Array<Record<string, unknown>>; tools: Array<Record<string, unknown>> } | null>(null)
  const [selectedAgent, setSelectedAgent] = useState('')
  const [agentDetail, setAgentDetail] = useState<Record<string, unknown> | null>(null)
  const [agentForm, setAgentForm] = useState<Record<string, unknown>>({})
  const [prompts, setPrompts] = useState<Array<Record<string, unknown>>>([])
  const [selectedPrompt, setSelectedPrompt] = useState('')
  const [promptDetail, setPromptDetail] = useState<Record<string, unknown> | null>(null)
  const [promptDraft, setPromptDraft] = useState('')
  const [collections, setCollections] = useState<CollectionSummary[]>([])
  const [selectedCollection, setSelectedCollection] = useState('')
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
  const [graphs, setGraphs] = useState<GraphIndexRecord[]>([])
  const [selectedGraph, setSelectedGraph] = useState('')
  const [graphDetail, setGraphDetail] = useState<GraphDetailPayload | null>(null)
  const [graphValidation, setGraphValidation] = useState<Record<string, unknown> | null>(null)
  const [graphRuns, setGraphRuns] = useState<GraphIndexRunRecord[]>([])
  const [graphCollectionId, setGraphCollectionId] = useState('')
  const [graphCollectionDocs, setGraphCollectionDocs] = useState<Array<Record<string, unknown>>>([])
  const [graphSelectedDocIds, setGraphSelectedDocIds] = useState<string[]>([])
  const [graphDraftId, setGraphDraftId] = useState('')
  const [graphDisplayNameDraft, setGraphDisplayNameDraft] = useState('')
  const [graphPromptDraft, setGraphPromptDraft] = useState('{}')
  const [graphConfigDraft, setGraphConfigDraft] = useState('{}')
  const [graphSkillIdsDraft, setGraphSkillIdsDraft] = useState('')
  const [graphSkillOverlayDraft, setGraphSkillOverlayDraft] = useState('')
  const [graphTuneGuidance, setGraphTuneGuidance] = useState('')
  const [graphTuneTargets, setGraphTuneTargets] = useState<string[]>(['extract_graph.txt'])
  const [graphTuneResult, setGraphTuneResult] = useState<GraphResearchTunePayload | null>(null)
  const [graphTuneSelectedPrompts, setGraphTuneSelectedPrompts] = useState<string[]>([])
  const [graphTuneRunning, setGraphTuneRunning] = useState(false)
  const [skills, setSkills] = useState<Array<Record<string, unknown>>>([])
  const [selectedSkill, setSelectedSkill] = useState('')
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
  const [mcpConnections, setMcpConnections] = useState<McpConnectionRecord[]>([])
  const [selectedMcpConnection, setSelectedMcpConnection] = useState('')
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
  const [permissionAction, setPermissionAction] = useState<'use' | 'manage'>('use')
  const [permissionSelector, setPermissionSelector] = useState('')
  const [membershipParentId, setMembershipParentId] = useState('')
  const [membershipChildId, setMembershipChildId] = useState('')
  const [agentsTab, setAgentsTab] = useSessionStringState<AgentsTab>('control-panel-ui-agents-tab', 'workspace')
  const [promptsTab, setPromptsTab] = useSessionStringState<PromptsTab>('control-panel-ui-prompts-tab', 'edit')
  const [collectionsTab, setCollectionsTab] = useSessionStringState<CollectionsTab>('control-panel-ui-collections-tab', 'workspace')
  const [collectionAction, setCollectionAction] = useSessionStringState<CollectionActionMode>('control-panel-ui-collection-action', 'host')
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
  const [graphInspectorOpen, setGraphInspectorOpen] = useSessionBooleanState('control-panel-ui-graph-inspector-open', !isCompactViewport())
  const [skillSummaryOpen, setSkillSummaryOpen] = useSessionBooleanState('control-panel-ui-skill-summary-open', !isCompactViewport())
  const uploadFilesInputRef = useRef<HTMLInputElement | null>(null)
  const uploadFolderInputRef = useRef<HTMLInputElement | null>(null)
  const graphUploadFilesInputRef = useRef<HTMLInputElement | null>(null)
  const graphUploadFolderInputRef = useRef<HTMLInputElement | null>(null)

  const groupedConfigFields = useMemo(() => groupFields(configFields), [configFields])
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
  const visibleConfigGroup = groupedConfigFields.find(([group]) => group === activeConfigGroup) ?? groupedConfigFields[0] ?? null
  const architectureNodeMap = useMemo(() => new Map((architecture?.nodes ?? []).map(node => [node.id, node])), [architecture])
  const selectedArchitectureNode = architectureNodeMap.get(selectedArchitectureNodeId) ?? null
  const selectedArchitecturePath = (architecture?.canonical_paths ?? []).find(path => path.id === selectedArchitecturePathId) ?? null
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
  const collectionActivityRecord = asRecord(collectionActivity)
  const collectionActivitySummary = asRecord(collectionActivityRecord?.summary)
  const collectionActivityFiles = asArray<Record<string, unknown>>(collectionActivityRecord?.files)
  const collectionActivityExceptions = collectionActivityFiles.filter(item => asString(item.outcome) !== 'ingested')
  const collectionActivityStatus = asString(collectionActivityRecord?.status)
  const graphBuildDocCount = graphSourceMode === 'collection' ? graphCollectionDocs.length : graphSelectedDocIds.length
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

  function collectionOutcomeTone(outcome: string): 'neutral' | 'ok' | 'warning' | 'danger' | 'accent' {
    if (outcome === 'ingested') return 'ok'
    if (outcome === 'already_indexed') return 'neutral'
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

  async function refreshCollections(preferredCollectionId = ''): Promise<string> {
    const payload = await api.listCollections(token)
    return applyCollections(payload.collections, preferredCollectionId)
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
    const pathIds = new Set(snapshot.canonical_paths.map(path => path.id))
    setSelectedArchitectureNodeId(current => (current && nodeIds.has(current) ? current : snapshot.nodes[0]?.id ?? ''))
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
      void api.listCollections(token).then(payload => {
        applyCollections(payload.collections)
      }).catch(err => setError(getMessage(err)))
    }
    if (active === 'graphs') {
      void Promise.all([api.listCollections(token), api.listGraphs(token), api.listSkills(token)])
        .then(([collectionsPayload, graphsPayload, skillsPayload]) => {
          applyCollections(collectionsPayload.collections)
          setGraphs(graphsPayload.graphs)
          setSkills(skillsPayload.data)
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
    if (!token || active !== 'skills' || !selectedSkill) return
    void api.getSkill(token, selectedSkill).then(detail => {
      setSkillDetail(detail)
      setSkillEditor(asString(detail.body_markdown))
      setCreatingSkill(false)
    }).catch(err => setError(getMessage(err)))
  }, [active, selectedSkill, token])

  useEffect(() => {
    if (!token || active !== 'graphs' || !selectedGraph) return
    void Promise.all([api.getGraph(token, selectedGraph), api.getGraphRuns(token, selectedGraph)])
      .then(([detail, runsPayload]) => {
        setGraphDetail(detail)
        hydrateGraphForm(detail)
        setGraphRuns(runsPayload.runs)
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
      const result = await api.ingestPaths(token, collectionId, pathDraft.split('\n').map(item => item.trim()).filter(Boolean))
      applyCollectionSelection(collectionId)
      await refreshCollectionWorkspace(collectionId)
      setCollectionActivity(result)
      setError('')
    } catch (err) {
      setError(getMessage(err))
    }
  }

  async function handleUpload(files: FileList | File[] | null) {
    if (!files || files.length === 0) return
    const collectionId = normalizeCollectionId(selectedCollection || collectionDraft)
    if (!collectionId) {
      setError('Choose a collection ID first.')
      return
    }
    const fileArray = Array.isArray(files) ? files : Array.from(files)
    const relativePaths = fileArray.map(file => {
      const relativePath = asString((file as File & { webkitRelativePath?: string }).webkitRelativePath)
      return relativePath || file.name
    })
    try {
      const result = await api.uploadFiles(token, collectionId, fileArray, relativePaths)
      applyCollectionSelection(collectionId)
      await refreshCollectionWorkspace(collectionId)
      setCollectionActivity(result)
      setError('')
    } catch (err) {
      setError(getMessage(err))
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

  async function handleGraphUpload(files: FileList | File[] | null) {
    if (!files || files.length === 0) return
    const collectionId = normalizeCollectionId(graphCollectionId)
    if (!collectionId) {
      setError('Choose a graph collection first.')
      return
    }
    const fileArray = Array.isArray(files) ? files : Array.from(files)
    const relativePaths = fileArray.map(file => {
      const relativePath = asString((file as File & { webkitRelativePath?: string }).webkitRelativePath)
      return relativePath || file.name
    })
    try {
      const result = await api.uploadFiles(token, collectionId, fileArray, relativePaths)
      await refreshCollections(collectionId)
      await refreshGraphCollectionDocs(collectionId)
      const uploadedDocIds = asArray<string>(result.doc_ids).map(String)
      if (uploadedDocIds.length > 0) {
        setGraphSelectedDocIds(current => uniqueList([...current, ...uploadedDocIds]))
      }
      setError('')
    } catch (err) {
      setError(getMessage(err))
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

  async function handleGraphValidate() {
    if (!selectedGraph) {
      setError('Create or select a graph first.')
      return
    }
    try {
      const payload = await api.validateGraph(token, selectedGraph)
      setGraphValidation(payload)
      setError('')
    } catch (err) {
      setError(getMessage(err))
    }
  }

  async function handleGraphBuild(refresh = false) {
    if (!selectedGraph) {
      setError('Create or select a graph first.')
      return
    }
    try {
      const response = refresh
        ? await api.refreshGraph(token, selectedGraph)
        : await api.buildGraph(token, selectedGraph)
      await refreshGraphs(selectedGraph)
      const detail = await api.getGraph(token, selectedGraph)
      setGraphDetail(detail)
      hydrateGraphForm(detail)
      const runsPayload = await api.getGraphRuns(token, selectedGraph)
      setGraphRuns(runsPayload.runs)
      setGraphValidation(response)
      setError('')
      notifyOk(refresh ? 'Graph refreshed' : 'Graph built', selectedGraph)
    } catch (err) {
      setError(getMessage(err))
      notifyError(refresh ? 'Graph refresh failed' : 'Graph build failed', err)
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
      const result = await api.startGraphResearchTune(token, selectedGraph, {
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
      const runsPayload = await api.getGraphRuns(token, selectedGraph)
      setGraphRuns(runsPayload.runs)
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

  const sectionToolbar = active === 'config' && groupedConfigFields.length > 0 ? (
    <SectionTabs
      tabs={groupedConfigFields.map(([group]) => ({ id: group, label: group }))}
      active={visibleConfigGroup?.[0] ?? groupedConfigFields[0]?.[0] ?? ''}
      onChange={group => setActiveConfigGroup(group)}
      ariaLabel="Config groups"
    />
  ) : active === 'architecture' ? (
    <SectionTabs
      tabs={[
        { id: 'map', label: 'Map' },
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
      className={active === 'agents' || active === 'prompts' || active === 'collections' || active === 'graphs' || active === 'skills' || active === 'access' || active === 'mcp' ? 'app-shell-studio' : undefined}
      sidebar={(
        <SidebarNav
          brand={(
            <div className="brand-block">
              <span className="section-eyebrow">Agentic RAG</span>
              <h1>Control Panel</h1>
              <p>Local admin workspace for runtime controls, assets, and agent behavior.</p>
            </div>
          )}
          items={SECTION_META.map(section => {
            const unsupported = unsupportedSectionIds.includes(section.id)
            let count: number | undefined
            if (section.id === 'collections') count = collections.length
            else if (section.id === 'graphs') count = graphs.length
            else if (section.id === 'skills') count = skills.length
            else if (section.id === 'operations') count = jobs.length
            return {
              id: section.id,
              label: section.label,
              description: section.eyebrow,
              icon: <SectionIcon kind={section.id} />,
              warning: unsupported,
              badge: unsupported
                ? <StatusBadge tone="warning">Unsupported</StatusBadge>
                : typeof count === 'number' && count > 0
                  ? <span className="nav-count">{count.toLocaleString()}</span>
                  : undefined,
            }
          })}
          active={active}
          onSelect={id => setActive(id as Section)}
          footer={(
            <>
              <div className="sidebar-status">
                <StatusBadge tone="accent">Local admin</StatusBadge>
                {lastReload && <StatusBadge tone={toneForStatus(lastReload.status)}>{asString(lastReload.status, 'idle')}</StatusBadge>}
              </div>
              <div className="sidebar-footer-actions">
                <IconButton
                  aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
                  title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
                  onClick={toggleTheme}
                >
                  {theme === 'dark' ? (
                    <svg viewBox="0 0 20 20" width="16" height="16" fill="currentColor" aria-hidden="true">
                      <path d="M10 2.5a.75.75 0 0 1 .75.75v1.25a.75.75 0 0 1-1.5 0V3.25A.75.75 0 0 1 10 2.5m0 12a.75.75 0 0 1 .75.75v1.25a.75.75 0 0 1-1.5 0v-1.25a.75.75 0 0 1 .75-.75M3.25 10a.75.75 0 0 1 .75-.75h1.25a.75.75 0 0 1 0 1.5H4a.75.75 0 0 1-.75-.75m11.5 0a.75.75 0 0 1 .75-.75H16.75a.75.75 0 0 1 0 1.5H15.5a.75.75 0 0 1-.75-.75m-9-4.75a.75.75 0 0 1 1.06 0l.89.89a.75.75 0 0 1-1.06 1.06l-.89-.89a.75.75 0 0 1 0-1.06m8.24 8.24a.75.75 0 0 1 1.06 0l.89.89a.75.75 0 1 1-1.06 1.06l-.89-.89a.75.75 0 0 1 0-1.06m0-8.24a.75.75 0 0 1 0 1.06l-.89.89a.75.75 0 1 1-1.06-1.06l.89-.89a.75.75 0 0 1 1.06 0M5.25 13.49a.75.75 0 0 1 0 1.06l-.89.89a.75.75 0 1 1-1.06-1.06l.89-.89a.75.75 0 0 1 1.06 0M10 6.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7" />
                    </svg>
                  ) : (
                    <svg viewBox="0 0 20 20" width="16" height="16" fill="currentColor" aria-hidden="true">
                      <path d="M10.5 2.75a.75.75 0 0 0-1.18-.61 7 7 0 1 0 8.54 8.54.75.75 0 0 0-.61-1.18 5.5 5.5 0 0 1-6.75-6.75" />
                    </svg>
                  )}
                </IconButton>
                <IconButton
                  aria-label={density === 'comfortable' ? 'Switch to compact density' : 'Switch to comfortable density'}
                  title={density === 'comfortable' ? 'Switch to compact density' : 'Switch to comfortable density'}
                  aria-pressed={density === 'compact'}
                  onClick={toggleDensity}
                >
                  {density === 'comfortable' ? (
                    <svg viewBox="0 0 20 20" width="16" height="16" fill="currentColor" aria-hidden="true">
                      <path d="M3.75 4a.75.75 0 0 0 0 1.5h12.5a.75.75 0 0 0 0-1.5zm0 5a.75.75 0 0 0 0 1.5h12.5a.75.75 0 0 0 0-1.5zm0 5a.75.75 0 0 0 0 1.5h12.5a.75.75 0 0 0 0-1.5z" />
                    </svg>
                  ) : (
                    <svg viewBox="0 0 20 20" width="16" height="16" fill="currentColor" aria-hidden="true">
                      <path d="M3.75 3.5a.75.75 0 0 0 0 1.5h12.5a.75.75 0 0 0 0-1.5zm0 3a.75.75 0 0 0 0 1.5h12.5a.75.75 0 0 0 0-1.5zm0 3a.75.75 0 0 0 0 1.5h12.5a.75.75 0 0 0 0-1.5zm0 3a.75.75 0 0 0 0 1.5h12.5a.75.75 0 0 0 0-1.5zm0 3a.75.75 0 0 0 0 1.5h12.5a.75.75 0 0 0 0-1.5z" />
                    </svg>
                  )}
                </IconButton>
                <ActionButton tone="ghost" onClick={() => setToken('')}>Lock</ActionButton>
              </div>
            </>
          )}
        />
      )}
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

          <div className="dashboard-grid">
            <CollapsibleSurfaceCard
              title="Reload Summary"
              subtitle="What changed most recently and whether it completed cleanly."
              open={dashboardReloadOpen}
              onToggle={() => setDashboardReloadOpen(!dashboardReloadOpen)}
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
              onToggle={() => setDashboardActivityOpen(!dashboardActivityOpen)}
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
                onToggle={() => setArchitectureInspectorOpen(!architectureInspectorOpen)}
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
          <div className="card-grid">
            {visibleConfigGroup && (
              <SurfaceCard key={visibleConfigGroup[0]} title={visibleConfigGroup[0]} subtitle="Review the live value, make a draft change, then validate before applying.">
                <div className="field-stack">
                  {visibleConfigGroup[1].map(field => {
                    const draftValue = configChanges[field.env_name] ?? ''
                    const currentValue = configEffective[field.env_name] ?? field.value
                    const changed = draftValue !== ''
                    const sliderDraftValue = changed
                      ? draftValue
                      : asString(currentValue, asString(field.min_value ?? 0))
                    return (
                      <div
                        key={field.env_name}
                        className={[
                          'config-row',
                          changed ? 'config-row-changed' : '',
                          field.readonly ? 'config-row-readonly' : '',
                        ].filter(Boolean).join(' ')}
                      >
                        <div className="config-head">
                          <div>
                            <label className="config-label" htmlFor={field.env_name}>{field.label}</label>
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
                                  id={field.env_name}
                                  aria-label={field.label}
                                  type="range"
                                  min={field.min_value ?? 0}
                                  max={field.max_value ?? 100}
                                  step={field.step ?? 1}
                                  value={asNumber(sliderDraftValue) ?? field.min_value ?? 0}
                                  onChange={event => setConfigChanges(current => ({ ...current, [field.env_name]: event.target.value }))}
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
                                id={field.env_name}
                                aria-label={field.label}
                                value={draftValue}
                                onChange={event => setConfigChanges(current => ({ ...current, [field.env_name]: event.target.value }))}
                              >
                                <option value="">No change</option>
                                {field.choices.map(choice => <option key={choice} value={choice}>{choice}</option>)}
                              </select>
                            ) : field.kind === 'bool' ? (
                              <select
                                id={field.env_name}
                                aria-label={field.label}
                                value={draftValue}
                                onChange={event => setConfigChanges(current => ({ ...current, [field.env_name]: event.target.value }))}
                              >
                                <option value="">No change</option>
                                <option value="true">true</option>
                                <option value="false">false</option>
                              </select>
                            ) : (
                              <input
                                id={field.env_name}
                                aria-label={field.label}
                                type={field.secret ? 'password' : 'text'}
                                value={draftValue}
                                onChange={event => setConfigChanges(current => ({ ...current, [field.env_name]: event.target.value }))}
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
            onToggle={() => setConfigPreviewOpen(!configPreviewOpen)}
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
        </div>
      )}

      {active === 'agents' && agentsPayload && (
        agentsTab === 'workspace' ? (
          <div className="studio-layout agent-studio">
            <SurfaceCard className="selection-rail agent-rail" title="Available Agents" subtitle="Choose which agent overlay you want to inspect or edit.">
              <EntityList
                variant="rail"
                items={agentsPayload.agents}
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
              onToggle={() => setAgentEditorOpen(!agentEditorOpen)}
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
                onToggle={() => setAgentInspectorOpen(!agentInspectorOpen)}
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
            <SurfaceCard title="Tool Catalog" subtitle="Current tool contracts available to the runtime, grouped as compact operator-friendly cards.">
              {agentsPayload.tools.length > 0 ? (
                <div className="chip-grid">
                  {agentsPayload.tools.map(tool => (
                    <div key={asString(tool.name)} className="tool-card">
                      <div className="tool-card-head">
                        <strong>{asString(tool.name)}</strong>
                        <StatusBadge tone={Boolean(tool.read_only) ? 'ok' : Boolean(tool.destructive) ? 'danger' : 'neutral'}>
                          {Boolean(tool.read_only) ? 'Read only' : Boolean(tool.destructive) ? 'Destructive' : 'Mutable'}
                        </StatusBadge>
                      </div>
                      <p>{asString(tool.description, 'No description')}</p>
                      <div className="badge-cluster">
                        <StatusBadge tone="neutral">{asString(tool.group, 'general')}</StatusBadge>
                        <StatusBadge tone={Boolean(tool.background_safe) ? 'ok' : 'warning'}>
                          {boolLabel(Boolean(tool.background_safe), 'Background safe', 'Foreground only')}
                        </StatusBadge>
                        <StatusBadge tone={Boolean(tool.requires_workspace) ? 'warning' : 'neutral'}>
                          {boolLabel(Boolean(tool.requires_workspace), 'Needs workspace', 'No workspace')}
                        </StatusBadge>
                      </div>
                    </div>
                  ))}
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
              <EntityList
                variant="rail"
                items={prompts}
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
              <EntityList
                variant="rail"
                items={prompts}
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
                onToggle={() => setPromptSummaryOpen(!promptSummaryOpen)}
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
            <label className="field">
              <span>Available Collections</span>
              <select
                aria-label="Available Collections"
                value={selectedCollection}
                onChange={event => applyCollectionSelection(event.target.value)}
              >
                <option value="">Choose a collection</option>
                {collections.map(collection => (
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
                placeholder="rfp-corpus"
              />
            </label>

            <ActionBar>
              <ActionButton tone="secondary" onClick={() => void handleUseCollection()}>Load Workspace</ActionButton>
              <ActionButton tone="primary" onClick={() => void handleCreateCollection()}>Create Collection</ActionButton>
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
                <StatusBadge tone={selectedCollectionMeta.status.ready ? 'ok' : 'warning'}>
                  {selectedCollectionMeta.status.ready ? 'Ready' : humanizeKey(selectedCollectionMeta.status.reason)}
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
                    <StatusBadge tone={collection.status.ready ? 'ok' : 'warning'}>
                      {collection.status.ready ? 'Ready' : humanizeKey(collection.status.reason)}
                    </StatusBadge>
                    <span>{collection.document_count} docs</span>
                    <span>{collection.graph_count} graphs</span>
                  </>
                )}
                emptyText="Create a collection to keep an empty namespace visible before the first ingest."
                onSelect={collection => applyCollectionSelection(collection.collection_id)}
              />
            ) : (
              <EmptyState title="No collections yet" body="Create a collection, then use the workspace to ingest files, folders, host paths, or KB content." />
            )}
          </SurfaceCard>

          <div className="content-stack collection-main-stack">
            <SurfaceCard
              className="collection-workspace-card"
              title="Add Documents"
              subtitle="Upload files, upload a folder, ingest a host path, or sync configured sources into the selected collection."
            >
              <input
                ref={uploadFilesInputRef}
                aria-label="Upload Files Input"
                type="file"
                multiple
                className="visually-hidden"
                tabIndex={-1}
                onChange={event => {
                  void handleUpload(event.target.files)
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
                  void handleUpload(event.target.files)
                  event.currentTarget.value = ''
                }}
              />

              <div className="badge-cluster">
                <StatusBadge tone={selectedCollectionMeta ? 'accent' : 'neutral'}>
                  {selectedCollectionMeta?.collection_id || normalizeCollectionId(collectionDraft) || 'No collection selected'}
                </StatusBadge>
                {selectedCollectionMeta && (
                  <StatusBadge tone={selectedCollectionMeta.status.ready ? 'ok' : 'warning'}>
                    {selectedCollectionMeta.status.ready ? 'Ready' : humanizeKey(selectedCollectionMeta.status.reason)}
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
                  { id: 'host', label: 'Ingest Host Path' },
                  { id: 'files', label: 'Upload Files' },
                  { id: 'folder', label: 'Upload Folder' },
                  { id: 'sync', label: 'Sync Configured Sources' },
                ]}
                active={collectionAction}
                onChange={value => setCollectionAction(value as CollectionActionMode)}
                ariaLabel="Collection actions"
                className="collection-action-tabs"
              />

              {collectionAction === 'host' && (
                <div className="collection-action-panel">
                  <label className="field">
                    <span>Host Paths</span>
                    <textarea
                      aria-label="Host Paths"
                      rows={5}
                      value={pathDraft}
                      onChange={event => setPathDraft(event.target.value)}
                      placeholder={'/absolute/path/to/file.csv\n/absolute/path/to/folder'}
                    />
                  </label>
                  <div className="collection-action-copy">
                    <p>Use this for deterministic local ingest when you want the workspace to preserve folder-relative display paths.</p>
                    <ActionButton tone="primary" onClick={() => void handlePathIngest()}>Ingest Host Paths</ActionButton>
                  </div>
                </div>
              )}

              {collectionAction === 'files' && (
                <div className="collection-action-panel collection-action-panel-compact">
                  <div className="collection-action-copy">
                    <strong>Upload individual files</strong>
                    <p>Choose a set of files from your machine and index them into the selected collection.</p>
                  </div>
                  <ActionButton tone="secondary" onClick={() => uploadFilesInputRef.current?.click()}>Upload Files</ActionButton>
                </div>
              )}

              {collectionAction === 'folder' && (
                <div className="collection-action-panel collection-action-panel-compact">
                  <div className="collection-action-copy">
                    <strong>Upload a folder</strong>
                    <p>Folder-relative paths are preserved so repeated filenames stay readable in the document list.</p>
                  </div>
                  <ActionButton tone="secondary" onClick={() => uploadFolderInputRef.current?.click()}>Upload Folder</ActionButton>
                </div>
              )}

              {collectionAction === 'sync' && (
                <div className="collection-action-panel collection-action-panel-compact">
                  <div className="collection-action-copy">
                    <strong>Sync configured KB sources</strong>
                    <p>Use this when the collection should mirror the runtime KB roots. Ad hoc uploaded or host-path corpora do not require this step.</p>
                  </div>
                  <ActionButton tone="ghost" onClick={() => void handleSyncCollection()}>Sync Configured Sources</ActionButton>
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
                <EmptyState title="No documents match yet" body="Create the collection and ingest files, folders, host paths, or KB content to populate the document workspace." />
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
              onToggle={() => setCollectionViewerOpen(!collectionViewerOpen)}
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
              onToggle={() => setCollectionInspectorOpen(!collectionInspectorOpen)}
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
              {skills.length > 0 ? (
                <EntityList
                  variant="rail"
                  items={skills}
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
                onToggle={() => setSkillSummaryOpen(!skillSummaryOpen)}
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
                <ActionButton tone="secondary" onClick={() => {
                  setSelectedGraph('')
                  setGraphDetail(null)
                  setGraphValidation(null)
                  setGraphRuns([])
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
                }}>
                  New Graph
                </ActionButton>
              </ActionBar>
              {graphs.length > 0 ? (
                <EntityList
                  variant="rail"
                  items={graphs}
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
                subtitle="Select a collection, upload or choose source documents, then save the draft before validating and building the GraphRAG project."
              >
                <input
                  ref={graphUploadFilesInputRef}
                  aria-label="Graph Source Files Input"
                  type="file"
                  multiple
                  className="visually-hidden"
                  tabIndex={-1}
                  onChange={event => {
                    void handleGraphUpload(event.target.files)
                    event.currentTarget.value = ''
                  }}
                />
                <input
                  ref={node => {
                    graphUploadFolderInputRef.current = node
                    if (node) {
                      node.setAttribute('webkitdirectory', '')
                      node.setAttribute('directory', '')
                    }
                  }}
                  aria-label="Graph Source Folder Input"
                  type="file"
                  multiple
                  className="visually-hidden"
                  tabIndex={-1}
                  onChange={event => {
                    void handleGraphUpload(event.target.files)
                    event.currentTarget.value = ''
                  }}
                />
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
                  <div className="field">
                    <span>Upload To Collection</span>
                    <div className="inline-action-pair">
                      <ActionButton
                        tone="secondary"
                        disabled={!graphCollectionId}
                        onClick={() => graphUploadFilesInputRef.current?.click()}
                      >
                        Upload Files
                      </ActionButton>
                      <ActionButton
                        tone="ghost"
                        disabled={!graphCollectionId}
                        onClick={() => graphUploadFolderInputRef.current?.click()}
                      >
                        Upload Folder
                      </ActionButton>
                    </div>
                  </div>
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
                        <EmptyState title="No documents in this collection" body="Upload files here or use the Collections workspace first, then come back to build the graph from indexed documents." />
                      )}
                    </div>
                    )}
                </div>

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
                  <ActionButton tone="primary" onClick={() => void handleGraphCreate()}>Save Draft</ActionButton>
                  <ActionButton tone="secondary" onClick={() => void handleGraphValidate()} disabled={!selectedGraph}>Validate</ActionButton>
                  <ActionButton tone="ghost" onClick={() => void handleGraphBuild(false)} disabled={!selectedGraph}>Build</ActionButton>
                  <ActionButton tone="ghost" onClick={() => void handleGraphBuild(true)} disabled={!selectedGraph}>Refresh</ActionButton>
                  <ActionButton tone="ghost" onClick={() => void handleGraphSavePrompts()} disabled={!selectedGraph}>Save Prompts</ActionButton>
                </ActionBar>
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
              title={selectedGraph ? 'Graph Inspector' : 'Graph Notes'}
              subtitle="Review build state, query readiness, recent runs, logs, and any validation payload without leaving the workspace."
              open={graphInspectorOpen}
              onToggle={() => setGraphInspectorOpen(!graphInspectorOpen)}
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
                        content: asArray<Record<string, unknown>>(graphDetail.logs).length > 0 ? (
                          <div className="field-stack">
                            {asArray<Record<string, unknown>>(graphDetail.logs).map(log => (
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
          <div className="card-grid">
            <SurfaceCard title="Principals" subtitle="Email-backed users today, placeholder groups for future IdP sync.">
              <div className="form-grid form-grid-compact">
                <label className="field">
                  <span>Principal Type</span>
                  <select value={principalDraftType} onChange={event => setPrincipalDraftType(event.target.value as 'user' | 'group')}>
                    <option value="user">User</option>
                    <option value="group">Group</option>
                  </select>
                </label>
                {principalDraftType === 'user' && (
                  <label className="field">
                    <span>Provider</span>
                    <select value={principalDraftProvider} onChange={event => setPrincipalDraftProvider(event.target.value)}>
                      <option value="email">Email</option>
                      <option value="entra">Future Entra</option>
                    </select>
                  </label>
                )}
                <label className="field">
                  <span>{principalDraftType === 'user' ? 'Email' : 'Group Name'}</span>
                  <input value={principalDraftValue} onChange={event => setPrincipalDraftValue(event.target.value)} placeholder={principalDraftType === 'user' ? 'alex@example.com' : 'Finance Analysts'} />
                </label>
              </div>
              <ActionBar>
                <ActionButton tone="secondary" onClick={() => void handleCreatePrincipal()}>Save Principal</ActionButton>
              </ActionBar>
              {accessPrincipals.length > 0 ? (
                <div className="timeline-list">
                  {accessPrincipals.map(principal => (
                    <article key={principal.principal_id} className="timeline-item">
                      <div className="timeline-dot" aria-hidden="true" />
                      <div>
                        <strong>{principal.display_name || principal.email_normalized || principal.principal_id}</strong>
                        <p>{humanizeKey(principal.principal_type)} • {principal.provider || 'system'}</p>
                        <span>{principal.email_normalized || principal.principal_id}</span>
                      </div>
                    </article>
                  ))}
                </div>
              ) : (
                <EmptyState title="No principals yet" body="Add an email user or a placeholder group to start assigning roles." />
              )}
            </SurfaceCard>

            <SurfaceCard title="Roles" subtitle="Reusable permission bundles for collections, graphs, tools, and skill families.">
              <div className="field-stack">
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
              {accessRoles.length > 0 ? (
                <div className="timeline-list">
                  {accessRoles.map(role => (
                    <article key={role.role_id} className="timeline-item">
                      <div className="timeline-dot" aria-hidden="true" />
                      <div>
                        <strong>{role.name}</strong>
                        <p>{role.description || 'No description'}</p>
                        <span>{role.role_id}</span>
                      </div>
                      <ActionButton
                        tone="destructive"
                        onClick={() => askConfirm({
                          title: 'Delete this role?',
                          description: `Remove "${role.name}". Existing bindings using this role will also be removed.`,
                          confirmLabel: 'Delete',
                          run: () => api.deleteAccessRole(token, role.role_id)
                            .then(() => { notifyOk('Role deleted', role.name); return refreshAccessData() })
                            .catch(err => { notifyError('Delete role failed', err); setError(getMessage(err)) }),
                        })}
                      >
                        Delete
                      </ActionButton>
                    </article>
                  ))}
                </div>
              ) : (
                <EmptyState title="No roles yet" body="Create roles before adding permissions or bindings." />
              )}
            </SurfaceCard>
          </div>

          <div className="card-grid">
            <SurfaceCard title="Bindings" subtitle="Attach roles to users or groups. Disabled bindings are ignored at runtime.">
              <div className="form-grid form-grid-compact">
                <label className="field">
                  <span>Role</span>
                  <select value={bindingRoleId} onChange={event => setBindingRoleId(event.target.value)}>
                    <option value="">Choose a role</option>
                    {accessRoles.map(role => <option key={role.role_id} value={role.role_id}>{role.name}</option>)}
                  </select>
                </label>
                <label className="field">
                  <span>Principal</span>
                  <select value={bindingPrincipalId} onChange={event => setBindingPrincipalId(event.target.value)}>
                    <option value="">Choose a principal</option>
                    {accessPrincipals.map(principal => (
                      <option key={principal.principal_id} value={principal.principal_id}>
                        {principal.display_name || principal.email_normalized || principal.principal_id}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <ActionBar>
                <ActionButton tone="secondary" onClick={() => void handleCreateBinding()}>Add Binding</ActionButton>
              </ActionBar>
              {accessBindings.length > 0 ? (
                <div className="timeline-list">
                  {accessBindings.map(binding => (
                    <article key={binding.binding_id} className="timeline-item">
                      <div className="timeline-dot" aria-hidden="true" />
                      <div>
                        <strong>{accessRoles.find(role => role.role_id === binding.role_id)?.name || binding.role_id}</strong>
                        <p>{accessPrincipals.find(principal => principal.principal_id === binding.principal_id)?.display_name || binding.principal_id}</p>
                        <span>{binding.disabled ? 'Disabled' : 'Active'}</span>
                      </div>
                      <ActionButton
                        tone="destructive"
                        onClick={() => askConfirm({
                          title: 'Remove this binding?',
                          description: 'The principal will lose access granted via this role until a new binding is created.',
                          confirmLabel: 'Remove',
                          run: () => api.deleteAccessBinding(token, binding.binding_id)
                            .then(() => { notifyOk('Binding removed'); return refreshAccessData() })
                            .catch(err => { notifyError('Remove binding failed', err); setError(getMessage(err)) }),
                        })}
                      >
                        Remove
                      </ActionButton>
                    </article>
                  ))}
                </div>
              ) : (
                <EmptyState title="No bindings yet" body="Bindings connect principals to roles and drive effective runtime access." />
              )}
            </SurfaceCard>

            <SurfaceCard title="Memberships" subtitle="Model group membership now so Entra group sync has a clean landing zone later.">
              <div className="form-grid form-grid-compact">
                <label className="field">
                  <span>Group</span>
                  <select value={membershipParentId} onChange={event => setMembershipParentId(event.target.value)}>
                    <option value="">Choose a group</option>
                    {accessPrincipals.filter(principal => principal.principal_type === 'group').map(principal => (
                      <option key={principal.principal_id} value={principal.principal_id}>
                        {principal.display_name || principal.principal_id}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span>Member</span>
                  <select value={membershipChildId} onChange={event => setMembershipChildId(event.target.value)}>
                    <option value="">Choose a member</option>
                    {accessPrincipals.filter(principal => principal.principal_id !== membershipParentId).map(principal => (
                      <option key={principal.principal_id} value={principal.principal_id}>
                        {principal.display_name || principal.email_normalized || principal.principal_id}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <ActionBar>
                <ActionButton tone="secondary" onClick={() => void handleCreateMembership()}>Add Membership</ActionButton>
              </ActionBar>
              {accessMemberships.length > 0 ? (
                <div className="timeline-list">
                  {accessMemberships.map(membership => (
                    <article key={membership.membership_id} className="timeline-item">
                      <div className="timeline-dot" aria-hidden="true" />
                      <div>
                        <strong>{accessPrincipals.find(principal => principal.principal_id === membership.parent_principal_id)?.display_name || membership.parent_principal_id}</strong>
                        <p>{accessPrincipals.find(principal => principal.principal_id === membership.child_principal_id)?.display_name || membership.child_principal_id}</p>
                        <span>{formatTimestamp(membership.created_at)}</span>
                      </div>
                      <ActionButton
                        tone="destructive"
                        onClick={() => askConfirm({
                          title: 'Remove this membership?',
                          description: 'Detaches the member from the group. Inherited access will be revoked.',
                          confirmLabel: 'Remove',
                          run: () => api.deleteAccessMembership(token, membership.membership_id)
                            .then(() => { notifyOk('Membership removed'); return refreshAccessData() })
                            .catch(err => { notifyError('Remove membership failed', err); setError(getMessage(err)) }),
                        })}
                      >
                        Remove
                      </ActionButton>
                    </article>
                  ))}
                </div>
              ) : (
                <EmptyState title="No memberships yet" body="Use memberships to assign users into placeholder groups now and future synced groups later." />
              )}
            </SurfaceCard>
          </div>

          <div className="card-grid">
            <SurfaceCard title="Permissions" subtitle="Grant `use` or `manage` at the role level, with exact ids or `*` for a wildcard.">
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
                    <option value="collection">Collection</option>
                    <option value="graph">Graph</option>
                    <option value="tool">Tool</option>
                    <option value="skill_family">Skill Family</option>
                  </select>
                </label>
                <label className="field">
                  <span>
                    Action{' '}
                    <Tooltip content="Use: can invoke the resource (run pipeline, query collection). Manage: can modify or delete the resource and grant others access.">
                      <span className="help-cue" aria-label="Action help">?</span>
                    </Tooltip>
                  </span>
                  <select value={permissionAction} onChange={event => setPermissionAction(event.target.value as 'use' | 'manage')}>
                    <option value="use">Use</option>
                    <option value="manage">Manage</option>
                  </select>
                </label>
                <label className="field">
                  <span>Resource Selector</span>
                  <select value={permissionSelector} onChange={event => setPermissionSelector(event.target.value)}>
                    <option value="">*</option>
                    {permissionResourceType === 'collection' && collections.map(collection => (
                      <option key={collection.collection_id} value={collection.collection_id}>{collection.collection_id}</option>
                    ))}
                    {permissionResourceType === 'graph' && graphs.map(graph => (
                      <option key={graph.graph_id} value={graph.graph_id}>{graph.display_name || graph.graph_id}</option>
                    ))}
                    {permissionResourceType === 'tool' && toolCatalog.map(tool => (
                      <option key={asString(tool.name)} value={asString(tool.name)}>{asString(tool.name)}</option>
                    ))}
                    {permissionResourceType === 'skill_family' && skills.map(skill => {
                      const familyId = asString(skill.version_parent || skill.skill_id)
                      return <option key={`${familyId}-${asString(skill.skill_id)}`} value={familyId}>{asString(skill.name, familyId)}</option>
                    })}
                  </select>
                </label>
              </div>
              <ActionBar>
                <ActionButton tone="secondary" onClick={() => void handleCreatePermission()}>Add Permission</ActionButton>
              </ActionBar>
              {accessPermissions.length > 0 ? (
                <div className="timeline-list">
                  {accessPermissions.map(permission => (
                    <article key={permission.permission_id} className="timeline-item">
                      <div className="timeline-dot" aria-hidden="true" />
                      <div>
                        <strong>{accessRoles.find(role => role.role_id === permission.role_id)?.name || permission.role_id}</strong>
                        <p>{accessResourceLabel(permission.resource_type)} • {permission.action}</p>
                        <span>{permission.resource_selector}</span>
                      </div>
                      <ActionButton
                        tone="destructive"
                        onClick={() => askConfirm({
                          title: 'Remove this permission?',
                          description: 'The role will lose this grant. Effective access for bound principals will narrow.',
                          confirmLabel: 'Remove',
                          run: () => api.deleteAccessPermission(token, permission.permission_id)
                            .then(() => { notifyOk('Permission removed'); return refreshAccessData() })
                            .catch(err => { notifyError('Remove permission failed', err); setError(getMessage(err)) }),
                        })}
                      >
                        Remove
                      </ActionButton>
                    </article>
                  ))}
                </div>
              ) : (
                <EmptyState title="No permissions yet" body="Permissions define what each role can use or manage in the runtime." />
              )}
            </SurfaceCard>

            <SurfaceCard title="Effective Access" subtitle="Preview the resolved principal graph and effective grants for any email before testing in OpenWebUI.">
              <label className="field">
                <span>User Email</span>
                <input value={accessPreviewEmail} onChange={event => setAccessPreviewEmail(event.target.value)} placeholder="alex@example.com" />
              </label>
              <ActionBar>
                <ActionButton tone="primary" onClick={() => void handleLoadEffectiveAccess()}>Preview Access</ActionButton>
              </ActionBar>
              {effectiveAccess ? (
                <>
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
                  <JsonInspector label="Effective access snapshot" value={effectiveAccess.access} />
                </>
              ) : (
                <EmptyState title="No preview yet" body="Choose a user email to inspect the exact access snapshot the runtime will resolve." />
              )}
            </SurfaceCard>
          </div>
        </div>
      )}

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

              {mcpConnections.length > 0 ? (
                <EntityList
                  items={mcpConnections}
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
              {auditEvents.length > 0 ? (
                <div className="timeline-list">
                  {auditEvents.map((event, index) => (
                    <article key={`${asString(event.action)}-${index}`} className="timeline-item">
                      <div className="timeline-dot" aria-hidden="true" />
                      <div>
                        <div className="tool-card-head">
                          <strong>{asString(event.action, 'event')}</strong>
                          <StatusBadge tone="neutral">{formatTimestamp(event.timestamp)}</StatusBadge>
                        </div>
                        <p>{asString(event.actor, 'system')}</p>
                        <span>{shortList(asArray<string>(event.changed_keys))}</span>
                      </div>
                    </article>
                  ))}
                </div>
              ) : (
                <EmptyState title="No audit events yet" body="Apply config, reload agents, or ingest content to start filling the audit stream." />
              )}
              {operations && <JsonInspector label="Technical details" value={operations} />}
            </SurfaceCard>
          )}
        </div>
      )}
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
    </AppShell>
  )
}
