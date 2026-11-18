import type { ParamValue, TensorShape } from "./types"

export const defaultCustomCode = "# x is the input tensor\nreturn x"

export function customInputNames(params: Record<string, ParamValue>): string[] {
  const names = params.input_names
  return Array.isArray(names) && names.length > 0 && names.every((name) => typeof name === "string" && /^[a-zA-Z_]\w*$/.test(name))
    ? names as string[]
    : ["x"]
}

export function parseShapeText(value: string): TensorShape {
  const trimmed = value.trim().replace(/^\[|\]$/g, "").trim()
  if (!trimmed) return []
  return trimmed.split(/[,，\s]+/).map((dim) => {
    if (dim.toUpperCase() === "B") return "B"
    if (dim === "?") return null
    const number = Number(dim)
    if (!/^\d+$/.test(dim) || !Number.isSafeInteger(number) || number <= 0) {
      throw new Error(`无效的张量维度：${dim}。请填写正整数、B 或 ?。`)
    }
    return number
  })
}

export function shapeToText(value: ParamValue): string {
  return Array.isArray(value) ? value.map((dim) => dim ?? "?").join(", ") : ""
}


export const manualClassName = "NNMindCustomBlock"

export function customClassTemplate(params: Record<string, ParamValue>, connections: string[] = [], outputs = 0, symbols?: { args: string[]; result: string }): string {
  const names = customInputNames(params)
  const body = String(params.code || defaultCustomCode).trimEnd()
  const args = symbols?.args ?? names
  const identity = symbols && body === defaultCustomCode
  const originalBindings = !identity && names.some((name, index) => name !== args[index])
    ? [`        ${names.join(", ")} = ${args.join(", ")}`] : []

  const aliases = Array.isArray(params.input_aliases) ? params.input_aliases : []
  const taken = new Set([...names, "self", "torch", "nn", "F", "True", "False", "None"])
  const bindings = names.flatMap((name, index) => {
    const alias = aliases[index]
    if (typeof alias !== "string" || !/^[a-zA-Z_]\w*$/.test(alias) || taken.has(alias) ||
      /^(?:and|as|assert|async|await|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield)$/.test(alias)) return []
    taken.add(alias)
    return [`        ${alias} = ${name}`]
  })
  return [
    `class ${manualClassName}(nn.Module):`,
    "    def __init__(self):",
    "        super().__init__()",
    "        # Configure layers here.",
    "",
    `    def forward(self, ${args.join(", ")}):`,
    ...connections.map((connection) => `        # Input: ${connection.replace(/\r?\n/g, " ")}`),
    ...(outputs ? [`        # Connected output links: ${outputs}`] : []),
    ...originalBindings,
    ...(identity
      ? [`        ${symbols.result} = ${args[0]}`, `        return ${symbols.result}`]
      : [...bindings, ...body.split("\n").map((line) => `        ${line}`)]),
  ].join("\n")
}
