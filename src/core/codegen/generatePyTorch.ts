import { modalityDefinitions } from "./modalityDefinitions"
import { graphPortBindings, graphSymbols, localInputNames, pythonIdentifier } from "../graph/graphSymbols"
import { customClassTemplate, manualClassName } from "../graph/customBlock"
import { getIncomingEdges, topologicalSort } from "../graph/utils"
import type { InferenceResult, LayerNode, NeuralGraph, ParamValue, TensorShape } from "../graph/types"
import { getLayerDef } from "../registry/layerRegistry"
import { inferGraph } from "../shape/inferShape"

function pyBool(value: ParamValue): string {
  return value ? "True" : "False"
}

function actualChannel(
  inference: InferenceResult | null,
  nodeId: string,
  paramFallback: ParamValue,
): number {
  const spec = inference?.specsByNodeId[nodeId]?.inputs[0]
  if (!spec) return Number(paramFallback)
  const dim = spec.shape[1]
  return typeof dim === "number" ? dim : Number(paramFallback)
}

function actualLastDim(
  inference: InferenceResult | null,
  nodeId: string,
  paramFallback: ParamValue,
): number {
  const spec = inference?.specsByNodeId[nodeId]?.inputs[0]
  if (!spec) return Number(paramFallback)
  const dim = spec.shape[spec.shape.length - 1]
  return typeof dim === "number" ? dim : Number(paramFallback)
}

function actualInputFeatures(
  inference: InferenceResult | null,
  nodeId: string,
  paramFallback: ParamValue,
): number {
  const spec = inference?.specsByNodeId[nodeId]?.inputs[0]
  if (!spec || spec.shape.length < 2) return Number(paramFallback)
  const dim = spec.shape[spec.shape.length - 1]
  return typeof dim === "number" ? dim : Number(paramFallback)
}

function actualNormalizedShape(
  inference: InferenceResult | null,
  nodeId: string,
  paramFallback: ParamValue,
): string {
  const spec = inference?.specsByNodeId[nodeId]?.inputs[0]
  if (!spec) {
    return Array.isArray(paramFallback) ? `(${paramFallback.join(", ")})` : String(paramFallback)
  }
  const normalizedLen = Array.isArray(paramFallback) ? paramFallback.length : 1
  const trailing = spec.shape.slice(spec.shape.length - normalizedLen)
  if (trailing.every((d) => typeof d === "number")) {
    return `(${trailing.join(", ")}${trailing.length === 1 ? "," : ""})`
  }
  return Array.isArray(paramFallback) ? `(${paramFallback.join(", ")})` : String(paramFallback)
}

function tupleToPython(value: ParamValue): string {
  if (!Array.isArray(value)) {
    return String(value)
  }

  const numeric = value.map((item) => String(item))
  return `(${numeric.join(", ")}${numeric.length === 1 ? "," : ""})`
}

function sanitizePythonName(raw: string, fallback: string): string {
  const normalized = raw
    .trim()
    .replace(/[^a-zA-Z0-9_]/g, "_")
    .replace(/_{2,}/g, "_")
    .replace(/^_+|_+$/g, "")

  if (!normalized) {
    return fallback
  }

  return /^[0-9]/.test(normalized) ? `${fallback}_${normalized}` : normalized
}

function buildUniqueName(raw: string, fallback: string, used: Set<string>): string {
  const base = sanitizePythonName(raw, fallback)

  if (!used.has(base)) {
    used.add(base)
    return base
  }

  let index = 2

  while (used.has(`${base}_${index}`)) {
    index += 1
  }

  const unique = `${base}_${index}`
  used.add(unique)
  return unique
}

function exampleTensor(shape: TensorShape, dtype: ParamValue): string {
  const dims = shape.map((dim) => (typeof dim === "number" ? dim : 2))
  const tuple = `(${dims.join(", ")}${dims.length === 1 ? "," : ""})`

  if (dtype === "int64") {
    return `torch.zeros(${tuple}, dtype=torch.long)`
  }

  return `torch.randn(${tuple})`
}

function needsModule(node: LayerNode): boolean {
  return Boolean(node.params.implementation_code) || !["Input", "Output", "Add", "Concat", "TokenPool"].includes(node.layerType)
}

function helperCode(helpers: Set<string>): string[] {
  const blocks: string[] = []

  if (helpers.has("ResidualBlock2D")) {
    blocks.push(
      [
        "class ResidualBlock2D(nn.Module):",
        "    def __init__(self, in_channels, out_channels, stride=1, use_projection=False):",
        "        super().__init__()",
        "        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)",
        "        self.bn1 = nn.BatchNorm2d(out_channels)",
        "        self.relu = nn.ReLU(inplace=True)",
        "        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)",
        "        self.bn2 = nn.BatchNorm2d(out_channels)",
        "        self.use_projection = use_projection or stride != 1 or in_channels != out_channels",
        "        self.proj = nn.Identity() if not self.use_projection else nn.Sequential(",
        "            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),",
        "            nn.BatchNorm2d(out_channels),",
        "        )",
        "",
        "    def forward(self, x):",
        "        identity = self.proj(x)",
        "        out = self.conv1(x)",
        "        out = self.bn1(out)",
        "        out = self.relu(out)",
        "        out = self.conv2(out)",
        "        out = self.bn2(out)",
        "        out = out + identity",
        "        out = self.relu(out)",
        "        return out",
      ].join("\n"),
    )
  }

  if (helpers.has("ResNetBasicBlock")) {
    blocks.push(
      [
        "class ResNetBasicBlock(nn.Module):",
        "    def __init__(self, in_channels, out_channels, stride=1, use_projection=False):",
        "        super().__init__()",
        "        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)",
        "        self.bn1 = nn.BatchNorm2d(out_channels)",
        "        self.relu = nn.ReLU(inplace=True)",
        "        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)",
        "        self.bn2 = nn.BatchNorm2d(out_channels)",
        "        self.use_projection = use_projection or stride != 1 or in_channels != out_channels",
        "        self.proj = nn.Identity() if not self.use_projection else nn.Sequential(",
        "            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),",
        "            nn.BatchNorm2d(out_channels),",
        "        )",
        "",
        "    def forward(self, x):",
        "        identity = self.proj(x)",
        "        out = self.conv1(x)",
        "        out = self.bn1(out)",
        "        out = self.relu(out)",
        "        out = self.conv2(out)",
        "        out = self.bn2(out)",
        "        out = out + identity",
        "        out = self.relu(out)",
        "        return out",
      ].join("\n"),
    )
  }

  if (helpers.has("ResNetBottleneck")) {
    blocks.push(
      [
        "class ResNetBottleneck(nn.Module):",
        "    def __init__(self, in_channels, bottleneck_channels, out_channels, stride=1, use_projection=False):",
        "        super().__init__()",
        "        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)",
        "        self.bn1 = nn.BatchNorm2d(bottleneck_channels)",
        "        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False)",
        "        self.bn2 = nn.BatchNorm2d(bottleneck_channels)",
        "        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)",
        "        self.bn3 = nn.BatchNorm2d(out_channels)",
        "        self.relu = nn.ReLU(inplace=True)",
        "        self.use_projection = use_projection or stride != 1 or in_channels != out_channels",
        "        self.proj = nn.Identity() if not self.use_projection else nn.Sequential(",
        "            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),",
        "            nn.BatchNorm2d(out_channels),",
        "        )",
        "",
        "    def forward(self, x):",
        "        identity = self.proj(x)",
        "        out = self.conv1(x)",
        "        out = self.bn1(out)",
        "        out = self.relu(out)",
        "        out = self.conv2(out)",
        "        out = self.bn2(out)",
        "        out = self.relu(out)",
        "        out = self.conv3(out)",
        "        out = self.bn3(out)",
        "        out = out + identity",
        "        out = self.relu(out)",
        "        return out",
      ].join("\n"),
    )
  }

  if (helpers.has("PatchEmbedding2D")) {
    blocks.push(
      [
        "class PatchEmbedding2D(nn.Module):",
        "    def __init__(self, in_channels, embed_dim, patch_size):",
        "        super().__init__()",
        "        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)",
        "",
        "    def forward(self, x):",
        "        x = self.proj(x)",
        "        x = x.flatten(2).transpose(1, 2)",
        "        return x",
      ].join("\n"),
    )
  }

  if (helpers.has("CausalLM")) blocks.push([
    "class NNMindCausalLM(nn.Module):",
    "    def __init__(self, path, targets, rank, alpha, precision, checkpointing):",
    "        super().__init__()",
    "        from transformers import AutoModelForCausalLM",
    "        from peft import LoraConfig, TaskType, get_peft_model",
    "        self.model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_safetensors=True, dtype=getattr(torch, precision), low_cpu_mem_usage=True)",
    "        self.model.config.use_cache = False",
    "        config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=targets or None, r=rank, lora_alpha=alpha)",
    "        self.model = get_peft_model(self.model, config)",
    "        if checkpointing:",
    "            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})",
    "",
    "    def forward(self, input_ids, attention_mask=None):",
    "        if isinstance(input_ids, dict):",
    "            attention_mask = input_ids['attention_mask']",
    "            input_ids = input_ids['input_ids']",
    "        return self.model(input_ids=input_ids.long(), attention_mask=attention_mask, use_cache=False).logits",
    "",
    "    def save_adapter(self, directory):",
    "        self.model.save_pretrained(directory, safe_serialization=True)",
  ].join("\n"))
  for (const [type, definition] of Object.entries(modalityDefinitions)) {
    if (helpers.has(type)) blocks.push(definition)
  }
  return blocks
}

function moduleLine(
  node: LayerNode,
  moduleName: string,
  helpers: Set<string>,
  inference: InferenceResult | null,
): string | null {
  switch (node.layerType) {
    case "Tokenizer":
    case "VisionEncoder":
    case "AudioEncoder":
      helpers.add(node.layerType)
      return `self.${moduleName} = NNMind${node.layerType}(${JSON.stringify(String(node.params.model_path ?? ""))}${node.layerType === "Tokenizer" ? `, ${Number(node.params.max_length || 512)}` : ""})`
    case "CausalLM": {
      helpers.add("CausalLM")
      const setting = String(node.params.lora_targets ?? "").trim()
      const targets = setting === "all-linear" ? setting : setting.split(",").map(value => value.trim()).filter(Boolean)
      return `self.${moduleName} = NNMindCausalLM(${JSON.stringify(String(node.params.model_path ?? ""))}, ${JSON.stringify(targets)}, ${Number(node.params.lora_rank)}, ${Number(node.params.lora_alpha)}, ${JSON.stringify(node.params.precision)}, ${pyBool(node.params.checkpointing)})`
    }
    case "CustomBlock":
      return node.params.block_source
        ? `self.${moduleName} = ${node.params.class_name}(${node.params.constructor_args ?? ""})`
        : null
    case "Embedding":
      return `self.${moduleName} = nn.Embedding(${node.params.num_embeddings}, ${node.params.embedding_dim})`
    case "Conv2d":
      return `self.${moduleName} = nn.Conv2d(${actualChannel(inference, node.id, node.params.in_channels)}, ${node.params.out_channels}, ${tupleToPython(node.params.kernel_size)}, ${tupleToPython(node.params.stride)}, ${tupleToPython(node.params.padding)}, dilation=${tupleToPython(node.params.dilation)}, bias=${pyBool(node.params.bias)})`
    case "ResidualBlock2d":
      helpers.add("ResidualBlock2D")
      return `self.${moduleName} = ResidualBlock2D(${actualChannel(inference, node.id, node.params.in_channels)}, ${node.params.out_channels}, stride=${node.params.stride}, use_projection=${pyBool(node.params.use_projection)})`
    case "ResNetBasicBlock":
      helpers.add("ResNetBasicBlock")
      return `self.${moduleName} = ResNetBasicBlock(${actualChannel(inference, node.id, node.params.in_channels)}, ${node.params.out_channels}, stride=${node.params.stride}, use_projection=${pyBool(node.params.use_projection)})`
    case "ResNetBottleneck":
      helpers.add("ResNetBottleneck")
      return `self.${moduleName} = ResNetBottleneck(${actualChannel(inference, node.id, node.params.in_channels)}, ${node.params.bottleneck_channels}, ${node.params.out_channels}, stride=${node.params.stride}, use_projection=${pyBool(node.params.use_projection)})`
    case "BatchNorm2d":
      return `self.${moduleName} = nn.BatchNorm2d(${actualChannel(inference, node.id, node.params.num_features)})`
    case "BatchNorm1d":
      return `self.${moduleName} = nn.BatchNorm1d(${actualLastDim(inference, node.id, node.params.num_features)})`
    case "LayerNorm":
      return `self.${moduleName} = nn.LayerNorm(${actualNormalizedShape(inference, node.id, node.params.normalized_shape)})`
    case "ReLU":
      return `self.${moduleName} = nn.ReLU(inplace=${pyBool(node.params.inplace)})`
    case "GELU":
      return `self.${moduleName} = nn.GELU()`
    case "MaxPool2d":
      return `self.${moduleName} = nn.MaxPool2d(${tupleToPython(node.params.kernel_size)}, ${tupleToPython(node.params.stride)}, ${tupleToPython(node.params.padding)})`
    case "AdaptiveAvgPool2d":
      return `self.${moduleName} = nn.AdaptiveAvgPool2d(${tupleToPython(node.params.output_size)})`
    case "PatchEmbedding":
      helpers.add("PatchEmbedding2D")
      return `self.${moduleName} = PatchEmbedding2D(${actualChannel(inference, node.id, node.params.in_channels)}, ${node.params.embed_dim}, ${tupleToPython(node.params.patch_size)})`
    case "Flatten":
      return `self.${moduleName} = nn.Flatten(start_dim=${node.params.start_dim}, end_dim=${node.params.end_dim})`
    case "Linear":
      return `self.${moduleName} = nn.Linear(${actualInputFeatures(inference, node.id, node.params.in_features)}, ${node.params.out_features}, bias=${pyBool(node.params.bias)})`
    case "Dropout":
      return `self.${moduleName} = nn.Dropout(p=${node.params.p})`
    case "LSTM":
      return `self.${moduleName} = nn.LSTM(input_size=${actualInputFeatures(inference, node.id, node.params.input_size)}, hidden_size=${node.params.hidden_size}, num_layers=${node.params.num_layers}, dropout=${node.params.dropout}, batch_first=${pyBool(node.params.batch_first)}, bidirectional=${pyBool(node.params.bidirectional)})`
    case "GRU":
      return `self.${moduleName} = nn.GRU(input_size=${actualInputFeatures(inference, node.id, node.params.input_size)}, hidden_size=${node.params.hidden_size}, num_layers=${node.params.num_layers}, dropout=${node.params.dropout}, batch_first=${pyBool(node.params.batch_first)}, bidirectional=${pyBool(node.params.bidirectional)})`
    case "SelfAttention":
      return `self.${moduleName} = nn.MultiheadAttention(embed_dim=${actualLastDim(inference, node.id, node.params.embed_dim)}, num_heads=${node.params.num_heads}, batch_first=${pyBool(node.params.batch_first)})`
    case "TransformerEncoder":
      return `self.${moduleName} = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=${actualLastDim(inference, node.id, node.params.d_model)}, nhead=${node.params.nhead}, dim_feedforward=${node.params.dim_feedforward}, dropout=${node.params.dropout}, activation="${node.params.activation}", batch_first=True), num_layers=${node.params.num_layers})`
    case "TransformerDecoder":
      return `self.${moduleName} = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=${actualLastDim(inference, node.id, node.params.d_model)}, nhead=${node.params.nhead}, dim_feedforward=${node.params.dim_feedforward}, dropout=${node.params.dropout}, activation="${node.params.activation}", batch_first=True), num_layers=${node.params.num_layers})`
    default:
      return null
  }
}

function orderedInputEdges(graph: NeuralGraph, node: LayerNode) {
  const incomingEdges = getIncomingEdges(graph, node.id)
  const layerDef = getLayerDef(node.layerType, node.params)

  if (layerDef.inputs.length <= 1) {
    return incomingEdges.slice(0, 1)
  }

  return layerDef.inputs
    .map((portDef) => incomingEdges.find((edge) => edge.toPort === portDef.name))
    .filter((edge): edge is NonNullable<typeof edge> => Boolean(edge))
}

function operationLine(node: LayerNode, moduleName: string, outputVar: string, inputVars: string[]): string | null {
  switch (node.layerType) {
    case "CustomBlock": {
      if (node.params.block_source) {
        const modes = Array.isArray(node.params.argument_modes) ? node.params.argument_modes : []
        const args = inputVars.map((value, index) => modes[index] ? `${modes[index]}=${value}` : value)
        return `${outputVar} = self.${moduleName}(${args.join(", ")})`
      }
      return `${outputVar} = self.${moduleName}(${inputVars.join(", ")})`
    }
    case "Add":
      return `${outputVar} = ${inputVars[0]} + ${inputVars[1]}`
    case "Concat":
      return `${outputVar} = torch.cat([${inputVars.join(", ")}], dim=${node.params.dim})`
    case "TokenPool":
      return node.params.mode === "cls" ? `${outputVar} = ${inputVars[0]}[:, 0]` : `${outputVar} = ${inputVars[0]}.mean(dim=1)`
    case "LSTM":
    case "GRU":
      return node.params.return_sequences
        ? `${outputVar}, _ = self.${moduleName}(${inputVars[0]})`
        : `${outputVar}, _ = self.${moduleName}(${inputVars[0]})\n${outputVar} = ${outputVar}[:, -1] if ${pyBool(node.params.batch_first)} else ${outputVar}[-1]`
    case "SelfAttention":
      return `${outputVar}, _ = self.${moduleName}(${inputVars[0]}, ${inputVars[0]}, ${inputVars[0]}, need_weights=False)`
    case "CausalLM":
      return `${outputVar} = self.${moduleName}(${inputVars.join(", ")})`
    case "TransformerDecoder":
      return `${outputVar} = self.${moduleName}(${inputVars[0]}, ${inputVars[1]})`
    default:
      return needsModule(node) ? `${outputVar} = self.${moduleName}(${inputVars[0]})` : null
  }
}

export function describeLayerCode(node: LayerNode, inputs?: string[], output?: string): { definition?: string; init?: string | null; forward?: string | null } {
  const { args, result } = implementationSymbols(node, inputs, output)
  if (node.layerType === "Input") {
    return { forward: `${result} = ${exampleTensor(Array.isArray(node.params.shape) ? node.params.shape as TensorShape : [2, 3, 224, 224], node.params.dtype)}` }
  }
  if (node.layerType === "Output") return { forward: `return ${args[0]}` }
  const helpers = new Set<string>()
  const init = moduleLine(node, "layer", helpers, null)
  const forward = operationLine(node, "layer", result, args)
  return { definition: helperCode(helpers).join("\n\n"), init, forward }
}

export const implementationClassName = "NNMindImplementation"

export function implementationTemplate(node: LayerNode, inputs?: string[], output?: string): string {
  const helpers = new Set<string>()
  const { args, result } = implementationSymbols(node, inputs, output)
  const endpoints = node.layerType === "Input" || node.layerType === "Output"
  const init = endpoints ? null : moduleLine(node, "layer", helpers, null)
  const forward = endpoints ? `${result} = ${args[0]}` : operationLine(node, "layer", result, args)
  return [
    ...helperCode(helpers).flatMap(block => [block, ""]),
    `class ${implementationClassName}(nn.Module):`,
    "    def __init__(self):",
    "        super().__init__()",
    ...(init ? [`        ${init}`] : []),
    "",
    `    def forward(self, ${args.join(", ")}):`,
    ...(forward ?? `${result} = None`).split("\n").map(line => "        " + line),
    `        return ${result}`,
    ...(node.layerType === "CausalLM" ? ["", "    def save_adapter(self, directory):", "        self.layer.save_adapter(directory)"] : []),
  ].join("\n")
}

export function implementationSymbols(node: LayerNode, inputs?: string[], output?: string) {
  const result = pythonIdentifier(output || String(node.params.output_alias || `${node.name}_out`))
  const defaults = node.layerType === "Input" || node.layerType === "Output" ? ["value"] : localInputNames(node)
  const used = new Set<string>([result])
  const args = defaults.map((name, index) => {
    const base = pythonIdentifier(inputs?.[index] || name)
    let candidate = base
    let suffix = 2
    while (used.has(candidate)) candidate = `${base}_${suffix++}`
    used.add(candidate)
    return candidate
  })
  return { args, result }
}

export function generatePyTorch(graph: NeuralGraph): string {
  const { orderedNodeIds, hasCycle } = topologicalSort(graph)

  if (hasCycle) {
    return [
      "# Cannot export a cyclic graph.",
      "# Remove loops before generating PyTorch code.",
    ].join("\n")
  }

  const nodeMap = new Map(graph.nodes.map((node) => [node.id, node]))
  const orderedNodes = orderedNodeIds.map((id) => nodeMap.get(id)).filter((node): node is LayerNode => Boolean(node))
  const inputNodes = orderedNodes.filter((node) => node.layerType === "Input")
  const outputNodes = orderedNodes.filter((node) => node.layerType === "Output")

  if (inputNodes.length === 0 || outputNodes.length === 0) {
    return [
      "# Graph must contain at least one Input and one Output node.",
    ].join("\n")
  }

  let inference: InferenceResult | null = null
  try {
    inference = inferGraph(graph)
  } catch {
    // If inference fails, fall back to node params
  }

  const usedNames = new Set<string>()
  const moduleNames = new Map<string, string>()
  const symbols = graphSymbols(graph)
  const variableNames = new Map<string, string>()
  const moduleLines: string[] = []
  const forwardLines: string[] = []
  const sourceBlocks = new Map<string, string>()
  const runtimeNames = new Map<string, string>()
  const helpers = new Set<string>()

  for (const node of orderedNodes) {
    if (needsModule(node)) {
      const moduleName = buildUniqueName("_layer_" + node.name, "_layer_" + node.layerType.toLowerCase(), usedNames)
      moduleNames.set(node.id, moduleName)
      if (typeof node.params.implementation_code === "string" && node.params.implementation_code) {
        const key = JSON.stringify([node.params.implementation_code, ""])
        const namespace = sourceBlocks.get(key) ?? `_nnmind_block_${sourceBlocks.size + 1}`
        sourceBlocks.set(key, namespace)
        moduleLines.push(`self.${moduleName} = ${namespace}.${implementationClassName}()`)
        runtimeNames.set(node.id, moduleName)
        continue
      }
      if (node.layerType === "CustomBlock") {
        const imported = typeof node.params.block_source === "string" && Boolean(node.params.block_source)
        const source = imported ? String(node.params.block_source) : String(node.params.class_code || customClassTemplate(node.params, [], 0, implementationSymbols(node, graphPortBindings(graph, node, symbols).map(binding => binding.local), symbols.get(node.id))))
        const className = imported ? String(node.params.class_name) : manualClassName
        const path = imported ? String(node.params.source_path ?? "") : ""
        const key = JSON.stringify([source, path])
        const namespace = sourceBlocks.get(key) ?? `_nnmind_block_${sourceBlocks.size + 1}`
        sourceBlocks.set(key, namespace)
        moduleLines.push(`self.${moduleName} = getattr(${namespace}, ${JSON.stringify(className)})(${imported ? node.params.constructor_args ?? "" : ""})`)
        runtimeNames.set(node.id, moduleName)
        continue
      }
      const line = moduleLine(node, moduleName, helpers, inference)

      if (line) {
        moduleLines.push(line)
        runtimeNames.set(node.id, moduleName)
      }
    }
  }

  const inputArgNames = new Map<string, string>()

  for (const node of inputNodes) {
    const argName = symbols.get(node.id)!
    inputArgNames.set(node.id, argName)
    variableNames.set(node.id, argName)
    const implementation = node.params.implementation_code && moduleNames.get(node.id)
    if (implementation) forwardLines.push(`${argName} = self.${implementation}(${argName})`)
    forwardLines.push(`if "${argName}" in _nnmind_overrides: ${argName} = _nnmind_overrides["${argName}"]`)
  }

  for (const node of orderedNodes) {
    if (node.layerType === "Input") {
      continue
    }

    const orderedEdges = orderedInputEdges(graph, node)
    const inputVars = orderedEdges.map((edge) => variableNames.get(edge.fromNodeId)).filter((value): value is string => Boolean(value))

    if (node.layerType === "Output") {
      const outputVar = inputVars[0]
      if (outputVar) {
        const alias = symbols.get(node.id)!
        const implementation = node.params.implementation_code && moduleNames.get(node.id)
        forwardLines.push(`${alias} = ${implementation ? `self.${implementation}(${outputVar})` : outputVar}`)
        variableNames.set(node.id, alias)
      }
      continue
    }

    if (inputVars.length < getLayerDef(node.layerType, node.params).inputs.length) {
      forwardLines.push(`# Skipped ${sanitizePythonName(node.name, "layer")}: missing input connection.`)
      continue
    }

    const moduleName = moduleNames.get(node.id) ?? buildUniqueName(node.name, "layer", usedNames)
    const outputVar = symbols.get(node.id)!
    const line = node.params.implementation_code
      ? `${outputVar} = self.${moduleName}(${inputVars.join(", ")})`
      : operationLine(node, moduleName, outputVar, inputVars)

    if (!line) {
      continue
    }

    forwardLines.push(`if "${outputVar}" in _nnmind_overrides:\n    ${outputVar} = _nnmind_overrides["${outputVar}"]\nelse:\n${line.split("\n").map(part => "    " + part).join("\n")}`)
    variableNames.set(node.id, outputVar)
  }

  const returnedVars = outputNodes.map((node) => variableNames.get(node.id)).filter((value): value is string => Boolean(value))
  const helperBlocks = helperCode(helpers)
  const exampleLines = inputNodes.map((node) => {
    const argName = inputArgNames.get(node.id) ?? "input_tensor"
    const inferredShape = inference?.specsByNodeId[node.id]?.outputs[0]?.shape
    const shape: TensorShape = inferredShape && inferredShape.every((d) => d !== null)
      ? (inferredShape as TensorShape)
      : Array.isArray(node.params.shape)
        ? (node.params.shape as TensorShape)
        : [2, 3, 224, 224]
    return `    ${argName} = ${exampleTensor(shape, node.params.dtype)}`
  })
  const callArgs = inputNodes.map((node) => inputArgNames.get(node.id)).filter((value): value is string => Boolean(value))
  const forwardSignature = inputNodes.map((node) => inputArgNames.get(node.id) ?? "input_tensor").join(", ")
  const contextFlag = "_nnmind_context"
  const causalNodes = orderedNodes.filter(node => node.layerType === "CausalLM").map(node => {
    const sources = orderedInputEdges(graph, node).map(edge => graph.nodes.find(item => item.id === edge.fromNodeId)!)
    return {
      output: symbols.get(node.id),
      inputs: sources.map(source => symbols.get(source.id)),
      tokenizer: String((sources[0]?.layerType === "Tokenizer" && sources[0].params.model_path) || node.params.model_path || ""),
    }
  })
  const overrides = "_nnmind_overrides"
  const frozen = orderedNodes.filter(node => node.params.training_mode === "frozen").flatMap(node => runtimeNames.get(node.id) ?? [])
  const frozenAttributes = ["__nnmind_frozen__ = " + JSON.stringify(frozen)]
  const capture = Array.from(variableNames, ([id, name]) => `${JSON.stringify(symbols.get(id))}: ${name}`).join(", ")
  const prediction = returnedVars.length === 1 ? returnedVars[0] : `(${returnedVars.join(", ")})`

  return [
    "import torch",
    "import torch.nn as nn",
    "",
    ...(sourceBlocks.size ? [
      "import pathlib, sys, types",
      "",
      "def _nnmind_source(source, filename, name):",
      "    path = pathlib.Path(filename).resolve() if filename else None",
      "    parent = path.parent if path else None",
      "    package = []",
      "    while parent and (parent / '__init__.py').is_file():",
      "        package.insert(0, parent.name)",
      "        parent = parent.parent",
      "    if parent and str(parent) not in sys.path:",
      "        sys.path.insert(0, str(parent))",
      "    qualname = '.'.join([*package, name])",
      "    module = types.ModuleType(qualname)",
      "    module.__package__ = '.'.join(package)",
      "    module.__file__ = str(path) if path else name + '.py'",
      "    module.__dict__.update(torch=torch, nn=nn, F=nn.functional)",
      "    sys.modules[qualname] = module",
      "    exec(compile(source, module.__file__, 'exec', dont_inherit=True), module.__dict__)",
      "    return module",
      "",
      ...Array.from(sourceBlocks, ([key, namespace]) => {
        const [source, path] = JSON.parse(key) as [string, string]
        return `${namespace} = _nnmind_source(${JSON.stringify(source)}, ${JSON.stringify(path)}, "${namespace}")`
      }),
      "",
    ] : []),
    ...helperBlocks.flatMap((block) => [block, ""]),
    "class GeneratedNet(nn.Module):",
    ...frozenAttributes.map(line => "    " + line),
    `    __nnmind_causal__ = ${JSON.stringify(causalNodes)}`,
    `    __nnmind_nodes__ = ${JSON.stringify(Object.fromEntries(runtimeNames))}`,
    "    def __init__(self):",
    "        super().__init__()",
    ...moduleLines.map((line) => `        ${line}`),
    "",
    "        for name in self.__nnmind_frozen__:",
    "            getattr(self, name).requires_grad_(False).eval()",
    "",
    "    def train(self, mode=True):",
    "        super().train(mode)",
    "        for name in self.__nnmind_frozen__:",
    "            getattr(self, name).eval()",
    "        return self",
    "",
    `    def forward(self, ${forwardSignature}, *, ${contextFlag}=False, ${overrides}=None):`,
    `        ${overrides} = ${overrides} or {}`,
    ...forwardLines.flatMap((line) => line.split("\n").map((part) => `        ${part}`)),
    returnedVars.length !== outputNodes.length
      ? '        raise RuntimeError("Connect all graph outputs before running the model.")'
      : `        return (${prediction}, {${capture}}) if ${contextFlag} else ${prediction}`,
    "",
    "",
    ...(sourceBlocks.size || helpers.has("CausalLM") || Object.keys(modalityDefinitions).some(type => helpers.has(type)) || inputNodes.some((node) => Array.isArray(node.params.shape) && node.params.shape.length === 0) ? [] : [
      "if __name__ == \"__main__\":",
      "    model = GeneratedNet()",
      ...exampleLines,
      `    y = model(${callArgs.join(", ")})`,
      "    if isinstance(y, tuple):",
      "        print([item.shape for item in y])",
      "    else:",
      "        print(y.shape)",
    ]),
  ].join("\n")
}
