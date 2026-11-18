import { layerRegistry } from "../registry/layerRegistry"
import type { NeuralGraph } from "../graph/types"

export function serializeGraph(graph: NeuralGraph): string {
  return JSON.stringify(graph, null, 2)
}

export function parseGraphJson(raw: string): NeuralGraph {
  const parsed = JSON.parse(raw) as NeuralGraph

  if (!parsed || parsed.framework !== "pytorch" || !Array.isArray(parsed.nodes) || !Array.isArray(parsed.edges)) {
    throw new Error("无效的图 JSON 格式。")
  }

  if (parsed.nodes.some((node) =>
    !node || typeof node.id !== "string" || typeof node.name !== "string" ||
    !layerRegistry.has(node.layerType) || !node.params || typeof node.params !== "object" ||
    !Number.isFinite(node.position?.x) || !Number.isFinite(node.position?.y) ||
    (node.size && (!Number.isFinite(node.size.width) || !Number.isFinite(node.size.height) || node.size.width <= 0 || node.size.height <= 0))
  )) {
    throw new Error("图 JSON 中包含无法识别的节点，请检查模块类型、参数和坐标。")
  }

  const nodeIds = new Set(parsed.nodes.map((node) => node.id))
  if (nodeIds.size !== parsed.nodes.length || parsed.edges.some((edge) =>
    !edge || typeof edge.id !== "string" || !nodeIds.has(edge.fromNodeId) || !nodeIds.has(edge.toNodeId) ||
    typeof edge.fromPort !== "string" || typeof edge.toPort !== "string"
  )) {
    throw new Error("图 JSON 包含重复节点或无效连线。")
  }

  return parsed
}
