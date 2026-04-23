import type { ArchitectureEdge, ArchitectureNode } from './types'

type ArchitectureLayer = {
  id: string
  label: string
}

type ArchitectureMapLayoutOptions = {
  layers: readonly ArchitectureLayer[]
  nodes: ArchitectureNode[]
  edges: ArchitectureEdge[]
  highlightedEdgeIds: Set<string>
  highlightedNodeIds: Set<string>
}

export type PositionedArchitectureLane = ArchitectureLayer & {
  x: number
  y: number
  width: number
  height: number
}

export type PositionedArchitectureNode = ArchitectureNode & {
  x: number
  y: number
  width: number
  height: number
}

export type ArchitectureEdgeRenderLayer = 'dimmed' | 'normal' | 'highlighted'

export type PositionedArchitectureEdge = ArchitectureEdge & {
  path: string
  startX: number
  startY: number
  endX: number
  endY: number
  highlighted: boolean
  dimmed: boolean
  layer: ArchitectureEdgeRenderLayer
}

export type ArchitectureMapLayout = {
  lanes: PositionedArchitectureLane[]
  nodes: PositionedArchitectureNode[]
  edges: PositionedArchitectureEdge[]
  width: number
  height: number
}

const COLUMN_WIDTH = 220
const NODE_WIDTH = 176
const NODE_HEIGHT = 70
const GUTTER_X = 64
const GUTTER_Y = 24
const TOP_OFFSET = 56
const LEFT_OFFSET = 44
const LANE_OFFSET_X = 20
const LANE_OFFSET_Y = 14
const EDGE_LANE_GAP = 12
const EDGE_PAIR_GAP = 8

function edgeSignature(edge: ArchitectureEdge): string {
  return [edge.source, edge.target, edge.kind, edge.label ?? '', edge.emphasis ?? ''].join('|')
}

function centeredOffsets(count: number): number[] {
  const offsets: number[] = []
  if (count <= 0) return offsets
  offsets.push(0)
  for (let step = 1; offsets.length < count; step += 1) {
    offsets.push(-step)
    if (offsets.length < count) offsets.push(step)
  }
  return offsets
}

function prioritySort(edgeIds: string[], highlightedIds: Set<string>, indexMap: Map<string, number>): string[] {
  return [...edgeIds].sort((left, right) => {
    const leftHighlighted = highlightedIds.has(left) ? 1 : 0
    const rightHighlighted = highlightedIds.has(right) ? 1 : 0
    if (leftHighlighted !== rightHighlighted) return rightHighlighted - leftHighlighted
    return (indexMap.get(left) ?? 0) - (indexMap.get(right) ?? 0)
  })
}

function offsetsForGroup(edgeIds: string[], highlightedIds: Set<string>, indexMap: Map<string, number>): Map<string, number> {
  const orderedEdgeIds = prioritySort(edgeIds, highlightedIds, indexMap)
  const offsets = centeredOffsets(orderedEdgeIds.length)
  return new Map(orderedEdgeIds.map((edgeId, index) => [edgeId, offsets[index] ?? 0]))
}

export function buildArchitectureMapLayout({
  layers,
  nodes,
  edges,
  highlightedEdgeIds,
  highlightedNodeIds,
}: ArchitectureMapLayoutOptions): ArchitectureMapLayout {
  const lanes = layers.map((layer, columnIndex) => ({
    ...layer,
    x: LANE_OFFSET_X + columnIndex * (COLUMN_WIDTH + GUTTER_X),
    y: LANE_OFFSET_Y,
    width: COLUMN_WIDTH,
    height: 0,
  }))

  const positionedNodes = lanes.flatMap((lane, columnIndex) =>
    nodes
      .filter(node => node.layer === lane.id)
      .map((node, rowIndex) => ({
        ...node,
        x: LEFT_OFFSET + columnIndex * (COLUMN_WIDTH + GUTTER_X),
        y: TOP_OFFSET + rowIndex * (NODE_HEIGHT + GUTTER_Y),
        width: NODE_WIDTH,
        height: NODE_HEIGHT,
      })),
  )

  const height = Math.max(
    280,
    ...positionedNodes.map(node => node.y + node.height + 48),
  )
  const width = LANE_OFFSET_X + lanes.length * COLUMN_WIDTH + Math.max(0, lanes.length - 1) * GUTTER_X + LANE_OFFSET_X
  const finalizedLanes = lanes.map(lane => ({ ...lane, height: height - LANE_OFFSET_Y * 2 }))
  const positionedNodeMap = new Map(positionedNodes.map(node => [node.id, node]))

  const highlightedSignatures = new Set(
    edges.filter(edge => highlightedEdgeIds.has(edge.id)).map(edgeSignature),
  )
  const dedupedEdges = Array.from(
    edges.reduce<Map<string, ArchitectureEdge>>((map, edge) => {
      const signature = edgeSignature(edge)
      if (!map.has(signature)) map.set(signature, edge)
      return map
    }, new Map()).values(),
  )

  const indexedEdges = dedupedEdges
    .map((edge, index) => ({ edge, index }))
    .filter(({ edge }) => positionedNodeMap.has(edge.source) && positionedNodeMap.has(edge.target))

  const edgeIndexMap = new Map(indexedEdges.map(({ edge, index }) => [edge.id, index]))
  const effectiveHighlightedIds = new Set(
    indexedEdges
      .filter(({ edge }) => highlightedEdgeIds.has(edge.id) || highlightedSignatures.has(edgeSignature(edge)))
      .map(({ edge }) => edge.id),
  )

  const sourceGroups = indexedEdges.reduce<Map<string, string[]>>((map, { edge }) => {
    const group = map.get(edge.source) ?? []
    group.push(edge.id)
    map.set(edge.source, group)
    return map
  }, new Map())
  const targetGroups = indexedEdges.reduce<Map<string, string[]>>((map, { edge }) => {
    const group = map.get(edge.target) ?? []
    group.push(edge.id)
    map.set(edge.target, group)
    return map
  }, new Map())
  const pairGroups = indexedEdges.reduce<Map<string, string[]>>((map, { edge }) => {
    const groupKey = `${edge.source}|${edge.target}`
    const group = map.get(groupKey) ?? []
    group.push(edge.id)
    map.set(groupKey, group)
    return map
  }, new Map())

  const sourceOffsets = new Map<string, number>()
  sourceGroups.forEach(edgeIds => {
    offsetsForGroup(edgeIds, effectiveHighlightedIds, edgeIndexMap).forEach((offset, edgeId) => {
      sourceOffsets.set(edgeId, offset)
    })
  })

  const targetOffsets = new Map<string, number>()
  targetGroups.forEach(edgeIds => {
    offsetsForGroup(edgeIds, effectiveHighlightedIds, edgeIndexMap).forEach((offset, edgeId) => {
      targetOffsets.set(edgeId, offset)
    })
  })

  const pairOffsets = new Map<string, number>()
  pairGroups.forEach(edgeIds => {
    offsetsForGroup(edgeIds, effectiveHighlightedIds, edgeIndexMap).forEach((offset, edgeId) => {
      pairOffsets.set(edgeId, offset)
    })
  })

  const hasSelection = highlightedNodeIds.size > 0 || effectiveHighlightedIds.size > 0
  const positionedEdges = indexedEdges.map(({ edge }) => {
    const source = positionedNodeMap.get(edge.source)!
    const target = positionedNodeMap.get(edge.target)!
    const sourceOffset = (sourceOffsets.get(edge.id) ?? 0) * EDGE_LANE_GAP + (pairOffsets.get(edge.id) ?? 0) * EDGE_PAIR_GAP * 0.4
    const targetOffset = (targetOffsets.get(edge.id) ?? 0) * EDGE_LANE_GAP - (pairOffsets.get(edge.id) ?? 0) * EDGE_PAIR_GAP * 0.4
    const startX = source.x + source.width
    const startY = source.y + source.height / 2 + sourceOffset
    const endX = target.x
    const endY = target.y + target.height / 2 + targetOffset
    const distanceX = Math.max(80, endX - startX)
    const curveX = Math.min(180, Math.max(96, distanceX * 0.38))
    const control1X = startX + curveX
    const control2X = endX - curveX
    const control1Y = startY + (sourceOffsets.get(edge.id) ?? 0) * EDGE_LANE_GAP * 0.65 + (pairOffsets.get(edge.id) ?? 0) * EDGE_PAIR_GAP
    const control2Y = endY + (targetOffsets.get(edge.id) ?? 0) * EDGE_LANE_GAP * 0.65 - (pairOffsets.get(edge.id) ?? 0) * EDGE_PAIR_GAP
    const highlighted = effectiveHighlightedIds.has(edge.id)
    const dimmed = hasSelection && !highlighted
    const layer: ArchitectureEdgeRenderLayer = highlighted ? 'highlighted' : dimmed ? 'dimmed' : 'normal'
    return {
      ...edge,
      path: `M ${startX} ${startY} C ${control1X} ${control1Y}, ${control2X} ${control2Y}, ${endX} ${endY}`,
      startX,
      startY,
      endX,
      endY,
      highlighted,
      dimmed,
      layer,
    }
  })

  return {
    lanes: finalizedLanes,
    nodes: positionedNodes,
    edges: positionedEdges,
    width,
    height,
  }
}
