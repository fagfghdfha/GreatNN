import { Position } from "@xyflow/react"
import { topologicalSort } from "../core/graph/utils"
import { getLayerDef } from "../core/registry/layerRegistry"
import type { GraphLayoutMode, LayerType, NeuralGraph } from "../core/graph/types"
import type { CanvasNode } from "./types"

export function getPortPositions(layoutMode: GraphLayoutMode) {
  return layoutMode === "vertical"
    ? { source: Position.Bottom, target: Position.Top }
    : { source: Position.Right, target: Position.Left }
}

export function orientPosition(x: number, y: number, layoutMode: GraphLayoutMode) {
  return layoutMode === "vertical" ? { x: y, y: x } : { x, y }
}

export function applyLayoutModeToNodes(nodes: CanvasNode[], layoutMode: GraphLayoutMode): CanvasNode[] {
  const minX = Math.min(...nodes.map((node) => node.position.x))
  const minY = Math.min(...nodes.map((node) => node.position.y))
  const { source, target } = getPortPositions(layoutMode)

  return nodes.map((node) => {
    const relativeX = node.position.x - minX
    const relativeY = node.position.y - minY
    const nextPosition = { x: 140 + relativeY, y: 120 + relativeX }

    return {
      ...node,
      position: nextPosition,
      sourcePosition: source,
      targetPosition: target,
      data: {
        ...node.data,
        layoutMode,
      },
    }
  })
}

export function getNodeSize(type: LayerType) {
  const category = getLayerDef(type).category
  if (category === "activation") return { width: 112, height: 30 }
  if (category === "norm" || category === "reshape") return { width: 155, height: 46 }
  if (category === "input" || category === "output" || category === "merge") return { width: 134, height: 58 }
  if (category === "transformer" || category === "attention" || category === "custom" || category === "residual") {
    return { width: 206, height: 88 }
  }
  return { width: 164, height: 68 }
}

export function compactGraph(graph: NeuralGraph, mode: GraphLayoutMode): NeuralGraph {
  if (!graph.nodes.length) return graph
  const { orderedNodeIds, hasCycle } = topologicalSort(graph)
  if (hasCycle) return graph

  const rankById = new Map(graph.nodes.map((node) => [node.id, 0]))
  const outgoing = new Map<string, string[]>()
  for (const edge of graph.edges) {
    const targets = outgoing.get(edge.fromNodeId) ?? []
    targets.push(edge.toNodeId)
    outgoing.set(edge.fromNodeId, targets)
  }
  for (const id of orderedNodeIds) {
    const nextRank = (rankById.get(id) ?? 0) + 1
    for (const target of outgoing.get(id) ?? []) {
      rankById.set(target, Math.max(rankById.get(target) ?? 0, nextRank))
    }
  }

  const gap = 48
  const sizes = graph.nodes.map((node) => node.size ?? getNodeSize(node.layerType))
  const widest = Math.max(...sizes.map((size) => size.width))
  const tallest = Math.max(...sizes.map((size) => size.height))
  const forward = (mode === "horizontal" ? widest : tallest) + gap
  const cross = (mode === "horizontal" ? tallest : widest) + gap
  const columns = mode === "horizontal" ? 5 : 6
  const usedRows = new Map<number, number>()
  const bandHeight = new Map<number, number>()
  const locations = new Map<string, { rank: number; row: number }>()
  for (const node of [...graph.nodes].sort((a, b) => a.position.y - b.position.y)) {
    const rank = rankById.get(node.id) ?? 0
    const row = usedRows.get(rank) ?? 0
    usedRows.set(rank, row + 1)
    const band = Math.floor(rank / columns)
    bandHeight.set(band, Math.max(bandHeight.get(band) ?? 0, row + 1))
    locations.set(node.id, { rank, row })
  }

  const offsets = new Map<number, number>()
  let offset = 0
  for (const band of [...bandHeight.keys()].sort((a, b) => a - b)) {
    offsets.set(band, offset)
    offset += (bandHeight.get(band) ?? 1) * cross + 24
  }
  return {
    ...graph,
    nodes: graph.nodes.map((node) => {
      const { rank, row } = locations.get(node.id)!
      return { ...node, position: { x: (rank % columns) * forward, y: (offsets.get(Math.floor(rank / columns)) ?? 0) + row * cross } }
    }),
  }
}
