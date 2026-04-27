export type Section =
  | 'dashboard'
  | 'architecture'
  | 'config'
  | 'agents'
  | 'prompts'
  | 'collections'
  | 'uploads'
  | 'graphs'
  | 'skills'
  | 'access'
  | 'mcp'
  | 'operations'

export type SectionGroup = 'admin' | 'workspace'

export interface SectionMeta {
  id: Section
  group: SectionGroup
  label: string
  shortLabel?: string
  eyebrow: string
  description: string
  route: string
  legacyRoutes: string[]
}

export const SECTION_META: SectionMeta[] = [
  {
    id: 'dashboard',
    group: 'admin',
    label: 'Overview',
    shortLabel: 'Overview',
    eyebrow: 'Admin Panel',
    description: 'Runtime posture, inventory, and recent activity.',
    route: '/admin/overview',
    legacyRoutes: ['/dashboard'],
  },
  {
    id: 'config',
    group: 'admin',
    label: 'Settings',
    shortLabel: 'Settings',
    eyebrow: 'Admin Panel',
    description: 'Search, validate, and apply runtime-safe settings.',
    route: '/admin/settings',
    legacyRoutes: ['/config'],
  },
  {
    id: 'access',
    group: 'admin',
    label: 'Access',
    shortLabel: 'Access',
    eyebrow: 'Admin Panel',
    description: 'Manage principals, roles, permissions, and effective access.',
    route: '/admin/access',
    legacyRoutes: ['/access', '/admin/users'],
  },
  {
    id: 'architecture',
    group: 'admin',
    label: 'Architecture',
    shortLabel: 'Architecture',
    eyebrow: 'Admin Panel',
    description: 'Inspect live topology, routing paths, and traffic signals.',
    route: '/admin/architecture',
    legacyRoutes: ['/architecture'],
  },
  {
    id: 'operations',
    group: 'admin',
    label: 'Operations',
    shortLabel: 'Operations',
    eyebrow: 'Admin Panel',
    description: 'Review reloads, background jobs, and audit activity.',
    route: '/admin/operations',
    legacyRoutes: ['/operations'],
  },
  {
    id: 'agents',
    group: 'workspace',
    label: 'Agents',
    shortLabel: 'Agents',
    eyebrow: 'Workspace',
    description: 'Edit agent overlays, tools, workers, and pinned skills.',
    route: '/workspace/agents',
    legacyRoutes: ['/agents', '/workspace/models'],
  },
  {
    id: 'collections',
    group: 'workspace',
    label: 'Knowledge',
    shortLabel: 'Knowledge',
    eyebrow: 'Workspace',
    description: 'Create collections, ingest files, inspect documents, and repair health.',
    route: '/workspace/knowledge',
    legacyRoutes: ['/collections'],
  },
  {
    id: 'uploads',
    group: 'workspace',
    label: 'Uploaded Files',
    shortLabel: 'Uploads',
    eyebrow: 'Workspace',
    description: 'Review chat and ad hoc uploads separately from knowledge collections.',
    route: '/workspace/uploads',
    legacyRoutes: ['/uploads', '/workspace/uploaded-files'],
  },
  {
    id: 'graphs',
    group: 'workspace',
    label: 'Graphs',
    shortLabel: 'Graphs',
    eyebrow: 'Workspace',
    description: 'Manage GraphRAG drafts, builds, prompts, skills, and runs.',
    route: '/workspace/graphs',
    legacyRoutes: ['/graphs'],
  },
  {
    id: 'prompts',
    group: 'workspace',
    label: 'Prompts',
    shortLabel: 'Prompts',
    eyebrow: 'Workspace',
    description: 'Edit overlays and compare base, overlay, and effective prompts.',
    route: '/workspace/prompts',
    legacyRoutes: ['/prompts'],
  },
  {
    id: 'skills',
    group: 'workspace',
    label: 'Skills',
    shortLabel: 'Skills',
    eyebrow: 'Workspace',
    description: 'Preview, author, activate, and archive reusable skill packs.',
    route: '/workspace/skills',
    legacyRoutes: ['/skills'],
  },
  {
    id: 'mcp',
    group: 'workspace',
    label: 'Tools',
    shortLabel: 'Tools',
    eyebrow: 'Workspace',
    description: 'Connect MCP servers and review the external tool catalog.',
    route: '/workspace/tools',
    legacyRoutes: ['/mcp', '/workspace/functions'],
  },
]

export const DEFAULT_SECTION: Section = 'dashboard'
export const SECTION_IDS: ReadonlySet<Section> = new Set(SECTION_META.map(section => section.id))

const exactRouteMap = new Map<string, Section>()

for (const section of SECTION_META) {
  exactRouteMap.set(section.route, section.id)
  for (const legacyRoute of section.legacyRoutes) {
    exactRouteMap.set(legacyRoute, section.id)
  }
}

export function getSectionMeta(section: Section): SectionMeta {
  return SECTION_META.find(item => item.id === section) ?? SECTION_META[0]
}

export function sectionToPath(section: Section): string {
  return getSectionMeta(section).route
}

export function sectionsForGroup(group: SectionGroup): SectionMeta[] {
  return SECTION_META.filter(section => section.group === group)
}

export function parseSectionFromPath(pathname: string): Section {
  const normalized = normalizePath(pathname)
  const exactMatch = exactRouteMap.get(normalized)
  if (exactMatch) return exactMatch

  const parts = normalized.replace(/^\/+/, '').split('/').filter(Boolean)
  const [first, second] = parts

  if (!first) return DEFAULT_SECTION
  if (SECTION_IDS.has(first as Section)) return first as Section

  if (first === 'admin') {
    if (!second || second === 'overview') return 'dashboard'
    if (second === 'settings') return 'config'
    if (second === 'access' || second === 'users') return 'access'
    if (second === 'architecture') return 'architecture'
    if (second === 'operations') return 'operations'
    return DEFAULT_SECTION
  }

  if (first === 'workspace') {
    if (!second || second === 'agents' || second === 'models') return 'agents'
    if (second === 'knowledge') return 'collections'
    if (second === 'uploads' || second === 'uploaded-files') return 'uploads'
    if (second === 'graphs') return 'graphs'
    if (second === 'prompts') return 'prompts'
    if (second === 'skills') return 'skills'
    if (second === 'tools' || second === 'functions') return 'mcp'
  }

  return DEFAULT_SECTION
}

function normalizePath(pathname: string): string {
  const withoutQuery = pathname.split(/[?#]/)[0] || '/'
  const trimmed = withoutQuery.replace(/\/+$/, '')
  return trimmed || '/'
}
