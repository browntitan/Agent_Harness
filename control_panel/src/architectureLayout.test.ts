import { describe, expect, it } from 'vitest'
import { buildArchitectureMapLayout } from './architectureLayout'
import type { ArchitectureEdge, ArchitectureNode } from './types'

const LAYERS = [
  { id: 'routing', label: 'Routing' },
  { id: 'top_level', label: 'Top-Level Agents' },
  { id: 'services', label: 'Runtime Services' },
] as const

const NODES: ArchitectureNode[] = [
  {
    id: 'router-core',
    label: 'Router',
    kind: 'router',
    layer: 'routing',
    description: 'Routing control point',
    status: 'configured',
  },
  {
    id: 'agent-general',
    label: 'general',
    kind: 'agent',
    layer: 'top_level',
    description: 'General agent',
    status: 'configured',
    mode: 'react',
  },
  {
    id: 'service-skill-store',
    label: 'Skill Store',
    kind: 'service',
    layer: 'services',
    description: 'Shared skill capability',
    status: 'active',
  },
]

describe('buildArchitectureMapLayout', () => {
  it('separates sibling edges so they do not share identical paths', () => {
    const edges: ArchitectureEdge[] = [
      { id: 'edge-default', source: 'router-core', target: 'agent-general', kind: 'routing_path', label: 'Default AGENT' },
      { id: 'edge-specialist', source: 'router-core', target: 'agent-general', kind: 'routing_path', label: 'Specialist AGENT' },
      { id: 'edge-skill-store', source: 'agent-general', target: 'service-skill-store', kind: 'service_dependency', label: 'Skill Store' },
    ]

    const layout = buildArchitectureMapLayout({
      layers: LAYERS,
      nodes: NODES,
      edges,
      highlightedEdgeIds: new Set(['edge-default']),
      highlightedNodeIds: new Set(['agent-general']),
    })

    const defaultEdge = layout.edges.find(edge => edge.id === 'edge-default')
    const specialistEdge = layout.edges.find(edge => edge.id === 'edge-specialist')
    expect(defaultEdge).toBeTruthy()
    expect(specialistEdge).toBeTruthy()
    expect(defaultEdge?.path).not.toBe(specialistEdge?.path)
    expect(defaultEdge?.startY).not.toBe(specialistEdge?.startY)
    expect(defaultEdge?.endY).not.toBe(specialistEdge?.endY)
  })

  it('keeps the highlighted sibling closest to the centerline and marks it for the top layer', () => {
    const edges: ArchitectureEdge[] = [
      { id: 'edge-default', source: 'router-core', target: 'agent-general', kind: 'routing_path', label: 'Default AGENT' },
      { id: 'edge-specialist', source: 'router-core', target: 'agent-general', kind: 'routing_path', label: 'Specialist AGENT' },
    ]

    const layout = buildArchitectureMapLayout({
      layers: LAYERS,
      nodes: NODES,
      edges,
      highlightedEdgeIds: new Set(['edge-default']),
      highlightedNodeIds: new Set(['agent-general']),
    })

    const router = layout.nodes.find(node => node.id === 'router-core')
    const defaultEdge = layout.edges.find(edge => edge.id === 'edge-default')
    const specialistEdge = layout.edges.find(edge => edge.id === 'edge-specialist')
    const routerCenterY = (router?.y ?? 0) + (router?.height ?? 0) / 2

    expect(defaultEdge?.layer).toBe('highlighted')
    expect(defaultEdge?.highlighted).toBe(true)
    expect(specialistEdge?.layer).toBe('dimmed')
    expect(specialistEdge?.highlighted).toBe(false)
    expect(specialistEdge?.dimmed).toBe(true)
    expect(Math.abs((defaultEdge?.startY ?? 0) - routerCenterY)).toBeLessThan(Math.abs((specialistEdge?.startY ?? 0) - routerCenterY))
  })

  it('deduplicates truly identical edges while preserving highlight intent', () => {
    const edges: ArchitectureEdge[] = [
      { id: 'edge-first', source: 'router-core', target: 'agent-general', kind: 'routing_path', label: 'Default AGENT' },
      { id: 'edge-duplicate', source: 'router-core', target: 'agent-general', kind: 'routing_path', label: 'Default AGENT' },
    ]

    const layout = buildArchitectureMapLayout({
      layers: LAYERS,
      nodes: NODES,
      edges,
      highlightedEdgeIds: new Set(['edge-duplicate']),
      highlightedNodeIds: new Set(),
    })

    expect(layout.edges).toHaveLength(1)
    expect(layout.edges[0]).toMatchObject({
      id: 'edge-first',
      highlighted: true,
      layer: 'highlighted',
    })
  })

  it('keeps every edge in the normal layer when the overview has no selection', () => {
    const edges: ArchitectureEdge[] = [
      { id: 'edge-default', source: 'router-core', target: 'agent-general', kind: 'routing_path', label: 'Default AGENT' },
      { id: 'edge-skill-store', source: 'agent-general', target: 'service-skill-store', kind: 'service_dependency', label: 'Skill Store' },
    ]

    const layout = buildArchitectureMapLayout({
      layers: LAYERS,
      nodes: NODES,
      edges,
      highlightedEdgeIds: new Set(),
      highlightedNodeIds: new Set(),
    })

    expect(layout.edges).toHaveLength(2)
    expect(layout.edges.every(edge => edge.layer === 'normal')).toBe(true)
    expect(layout.edges.every(edge => edge.dimmed === false)).toBe(true)
  })
})
