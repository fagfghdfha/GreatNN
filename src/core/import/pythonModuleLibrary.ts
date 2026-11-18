import { parser } from "@lezer/python"
import { createDefaultParams } from "../registry/layerRegistry"
import type { GraphEdge, LayerNode, NeuralGraph } from "../graph/types"

type SyntaxNode = ReturnType<typeof parser.parse>["topNode"]

type PythonParameter = { name: string; optional: boolean; keywordOnly: boolean; positionalOnly: boolean }

export type PythonModule = {
  name: string
  documentation: string
  code: string
  start: number
  end: number
  inputs: PythonParameter[]
  constructorArguments: string
  requiredConstructorArguments: string
}

function readParams(source: string, method: SyntaxNode | null): PythonParameter[] {
  const params = method?.getChild("ParamList")
  if (!params) return []
  const result: PythonParameter[] = []
  let parameter: PythonParameter | undefined
  let keywordOnly = false
  let variadic = false
  for (let token = params.firstChild; token; token = token.nextSibling) {
    const text = source.slice(token.from, token.to)
    if (token.name === ",") {
      parameter = undefined
      variadic = false
    } else if (token.name === "*" || token.name === "**") {
      keywordOnly = true
      variadic = true
    } else if (token.name === "/") {
      for (const item of result) item.positionalOnly = true
    } else if (token.name === "VariableName" && !parameter && !variadic) {
      parameter = { name: text, optional: false, keywordOnly, positionalOnly: false }
      result.push(parameter)
    } else if (token.name === "AssignOp" && parameter) {
      parameter.optional = true
    }
  }
  return result.slice(1)
}

export function parsePython(source: string) {
  let tree = parser.parse(source)
  const trailingDots: number[] = []
  // Lezer reads a Python float like 10. as an unfinished attribute access.
  // Mask only that dot when parsing; the original source and offsets stay intact.
  tree.iterate({ enter(token) {
    const expression = token.node.parent
    if (!token.type.isError || expression?.name !== "MemberExpression") return
    const number = expression.firstChild
    const dot = number?.nextSibling
    if (number?.name === "Number" && dot?.name === "." && dot.from === number.to &&
      /^[0-9][0-9_]*$/.test(source.slice(number.from, number.to)) &&
      /^(?:[ \t]*[,):\]}+*/%<>=#;\n\r-]|[ \t]*$)/.test(source.slice(dot.to))) {
      trailingDots.push(dot.from)
    }
  } })
  if (trailingDots.length) {
    const pieces: string[] = []
    let start = 0
    for (const dot of new Set(trailingDots.sort((left, right) => left - right))) {
      pieces.push(source.slice(start, dot), " ")
      start = dot + 1
    }
    pieces.push(source.slice(start))
    tree = parser.parse(pieces.join(""))
  }
  return tree
}

export function pythonDiagnostics(source: string) {
  const issues: Array<{ from: number; to: number }> = []
  parsePython(source).iterate({ enter(node) {
    if (node.type.isError) issues.push({ from: node.from, to: node.to })
  } })
  return issues
}

export function readPythonModuleLibrary(source: string): PythonModule[] {
  const tree = parsePython(source)
  let firstError = -1
  tree.iterate({ enter(node) {
    if (node.type.isError && firstError < 0) firstError = node.from
  } })
  if (firstError >= 0) {
    const line = source.slice(0, firstError).split("\n").length
    throw new Error(`Python 文件存在语法错误，请检查第 ${line} 行附近。`)
  }

  const candidates = new Map<string, {
    classNode: SyntaxNode
    body: SyntaxNode
    methods: Map<string, SyntaxNode>
    bases: string[]
  }>()
  for (let statement = tree.topNode.firstChild; statement; statement = statement.nextSibling) {
    const definition = statement.name === "ClassDefinition" ? statement : statement.name === "DecoratedStatement"
      ? statement.getChild("ClassDefinition") : null
    if (!definition) continue
    const node = definition.getChild("VariableName")
    const body = definition.getChild("Body")
    if (!node || !body) continue
    const bases = (definition.getChild("ArgList")?.getChildren("MemberExpression") ?? [])
      .map((base) => source.slice(base.from, base.to))
      .concat((definition.getChild("ArgList")?.getChildren("VariableName") ?? []).map((base) => source.slice(base.from, base.to)))
    const methods = new Map<string, SyntaxNode>()
    for (let child = body.firstChild; child; child = child.nextSibling) {
      const method = child.name === "FunctionDefinition" ? child : child.name === "DecoratedStatement"
        ? child.getChild("FunctionDefinition") : null
      if (!method) continue
      const name = method.getChild("VariableName")
      if (name) methods.set(source.slice(name.from, name.to), method)
    }
    candidates.set(source.slice(node.from, node.to), { classNode: statement, body, methods, bases })
  }

  function findMethod(name: string, method: string, seen = new Set<string>()): SyntaxNode | null {
    if (seen.has(name)) return null
    seen.add(name)
    const entry = candidates.get(name)
    if (!entry) return null
    const own = entry.methods.get(method)
    if (own) return own
    for (const base of entry.bases) {
      const inherited = findMethod(base, method, seen)
      if (inherited) return inherited
    }
    return null
  }

  const result: PythonModule[] = []
  for (const [name, entry] of candidates) {
    const forward = findMethod(name, "forward")
    if (!forward) continue
    const constructor = findMethod(name, "__init__")
    const constructorParams = readParams(source, constructor)
    const paramList = constructor?.getChild("ParamList")
    let firstStatement = entry.body.firstChild?.nextSibling
    while (firstStatement?.name === "Comment") firstStatement = firstStatement.nextSibling
    const documentation = firstStatement?.getChild("String")
    result.push({
      name,
      documentation: documentation ? source.slice(documentation.from, documentation.to).replace(/^(?:"""|'''|"|')|(?:"""|'''|"|')$/g, "").trim() : "",
      code: source.slice(entry.classNode.from, entry.classNode.to),
      start: entry.classNode.from,
      end: entry.classNode.to,
      inputs: readParams(source, forward),
      constructorArguments: paramList ? source.slice(paramList.from + 1, paramList.to - 1).replace(/^\s*self\s*,?\s*/, "").trim() : "",
      requiredConstructorArguments: constructorParams.filter((parameter) => !parameter.optional)
        .map(({ name, positionalOnly }) => positionalOnly ? "None" : `${name}=None`).join(", "),
    })
  }
  return result
}

export function pythonModuleToGraph(source: string, module: PythonModule, includeOptional: boolean, sourcePath = ""): NeuralGraph {
  const inputs = module.inputs.filter((parameter) => includeOptional || !parameter.optional)
  if (inputs.length === 0) {
    throw new Error("此模块的 forward 没有显式输入参数。请选择另一个类，或勾选可选参数。")
  }
  const inputNodes: LayerNode[] = inputs.map(({ name }, index) => ({
    id: `input-${index}`,
    name,
    layerType: "Input",
    position: { x: 0, y: index * 100 },
    params: { ...createDefaultParams("Input"), shape: [] },
  }))
  const block: LayerNode = {
    id: "python-module",
    name: module.name,
    layerType: "CustomBlock",
    position: { x: 250, y: 0 },
    params: {
      ...createDefaultParams("CustomBlock"),
      input_names: inputs.map(({ name }) => name),
      input_shapes: inputs.map(() => ""),
      argument_modes: inputs.map(({ name, positionalOnly }) => positionalOnly ? "" : name),
      block_source: source,
      source_path: sourcePath,
      class_name: module.name,
      constructor_args: module.requiredConstructorArguments,
      constructor_signature: module.constructorArguments,
      documentation: module.documentation,
    },
  }
  const output: LayerNode = {
    id: "output",
    name: "output",
    layerType: "Output",
    position: { x: 500, y: 0 },
    params: createDefaultParams("Output"),
  }
  const edges: GraphEdge[] = inputNodes.map((input, index) => ({
    id: `link-${index}`,
    fromNodeId: input.id,
    fromPort: "out",
    toNodeId: block.id,
    toPort: `in${index}`,
  }))
  edges.push({ id: "result", fromNodeId: block.id, fromPort: "out", toNodeId: output.id, toPort: "in" })
  return { version: 1, framework: "pytorch", nodes: [...inputNodes, block, output], edges }
}
