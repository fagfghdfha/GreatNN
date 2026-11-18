import { topologicalSort } from "./utils"
import { getLayerDef } from "../registry/layerRegistry"
import type { LayerNode, NeuralGraph } from "./types"

const reserved = new Set("False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield self torch nn F".split(" "))

export function pythonIdentifier(value: string, fallback = "tensor"): string {
  const fixed = value.trim().replace(/[^A-Za-z0-9_]+/g, "_").replace(/^_+|_+$/g, "")
  if (!fixed) return fallback
  return /^\d/.test(fixed) || reserved.has(fixed) ? `v_${fixed}` : fixed
}

export function localInputNames(node: Pick<LayerNode, "layerType" | "params">): string[] {
  const names = node.params.input_aliases
  return getLayerDef(node.layerType, node.params).inputs.map((port, index) => {
    const alias = Array.isArray(names) ? names[index] : undefined
    const implementation = Array.isArray(node.params.input_names) ? node.params.input_names[index] : undefined
    return pythonIdentifier(String(alias || implementation || (port.name === "in" ? "input_tensor" : port.name)))
  })
}

export function graphSymbols(graph: NeuralGraph) {
  const used = new Set<string>()
  const names = new Map<string, string>()
  const nodes = new Map(graph.nodes.map(node => [node.id, node]))
  const order = topologicalSort(graph).orderedNodeIds
  for (const id of order) {
    const node = nodes.get(id)!
    const fallback = node.layerType === "Input" || node.layerType === "Output" ? node.name : `${node.name}_out`
    const name = pythonIdentifier(String(node.params.output_alias || fallback))
    let unique = name
    let n = 2
    while (used.has(unique)) unique = `${name}_${n++}`
    used.add(unique)
    names.set(id, unique)
  }
  return names
}

export function graphPortBindings(graph: NeuralGraph, node: LayerNode, names = graphSymbols(graph)) {
  const localNames = localInputNames(node)
  const ports = getLayerDef(node.layerType, node.params).inputs
  return ports.map((port, index) => {
    const upstream = graph.edges.find(edge => edge.toNodeId === node.id && edge.toPort === port.name)
    const source = upstream ? names.get(upstream.fromNodeId) : undefined
    const explicit = Array.isArray(node.params.input_aliases) && Boolean(node.params.input_aliases[index])
    return { port: port.name, local: explicit ? localNames[index] : source ?? localNames[index], source }
  })
}
