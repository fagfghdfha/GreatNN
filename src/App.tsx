import { topologicalSort } from "./core/graph/utils"
import { tensorDTypes } from "./core/graph/types"
import { RunDialog } from "./editor/RunDialog"
import { graphPortBindings, graphSymbols } from "./core/graph/graphSymbols"
import { readBlockLibrary, writeBlockLibrary, type SavedBlock } from "./core/graph/blockLibrary"
import type { GraphDataset, InputProbeResult, ObservedInput, PythonEnvironment, TrainingParameters } from "./core/trainingContracts"
import { InputScriptDialog } from "./editor/InputScriptDialog"
import { TrainingDialog } from "./editor/TrainingDialog"
import { trainingDesktop } from "./editor/trainingDesktop"
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent as ReactKeyboardEvent } from "react"
import {
  addEdge,
  Background,
  ControlButton,
  Controls,
  type FinalConnectionState,
  type HandleType,
  MiniMap,
  MarkerType,
  ReactFlow,
  type XYPosition,
  useEdgesState,
  useNodesState,
  type Connection,
  type Edge,
  type NodeTypes,
  type ReactFlowInstance,
} from "@xyflow/react"
import { generatePyTorch, implementationTemplate } from "./core/codegen/generatePyTorch"
import { estimateNodeParamCount, formatParamCount } from "./core/graph/paramCount"
import type { GraphLayoutMode, NeuralGraph, ParamValue } from "./core/graph/types"
import { importModelSource, importOnnxBuffer } from "./core/import/importModel"
import { type PythonModule, pythonModuleToGraph, readPythonModuleLibrary } from "./core/import/pythonModuleLibrary"
import { getLayerDef, layerLibrary } from "./core/registry/layerRegistry"
import { serializeGraph, parseGraphJson } from "./core/serialize/graphJson"
import { inferGraph } from "./core/shape/inferShape"
import { validateGraph } from "./core/validate/validateGraph"
import { modelPresets } from "./examples/modelPresets"
import { ExportPanel } from "./editor/ExportPanel"
import { canvasToGraph, buildCanvasNode, graphToCanvasEdges, graphToCanvasNodes } from "./editor/graphAdapter"
import { Inspector } from "./editor/Inspector"
import { IssuesPanel } from "./editor/IssuesPanel"
import { applyLayoutModeToNodes, compactGraph } from "./editor/layout"
import { BlockCodeDialog } from "./editor/BlockCodeDialog"
import { LayerNode } from "./editor/LayerNode"
import { LayerPalette } from "./editor/LayerPalette"
import { PresetLibrary } from "./editor/PresetLibrary"
import type { CanvasNode } from "./editor/types"
import {
  type Locale,
  formatExactParamCount,
  formatLayoutMode,
  getLayerCategoryLabel,
  getLayerDescription,
  getLayerLabel,
  getUiText,
  translateMessage,
} from "./i18n"
import "./App.css"

const nodeTypes = {
  layerNode: LayerNode,
} satisfies NodeTypes

const canvasAriaLabels = {
  zh: {
    "controls.ariaLabel": "画布操作",
    "controls.zoomIn.ariaLabel": "放大",
    "controls.zoomOut.ariaLabel": "缩小",
    "controls.fitView.ariaLabel": "适应画布",
    "controls.interactive.ariaLabel": "切换画布交互",
    "minimap.ariaLabel": "缩略图",
    "handle.ariaLabel": "连接端口",
    "node.a11yDescription.default": "按 Enter 或空格选择节点，按 Delete 删除节点。",
    "edge.a11yDescription.default": "按 Enter 或空格选择连线，按 Delete 删除连线。",
  },
  en: {
    "node.a11yDescription.default": "Press Enter or Space to select a node. Press Delete to remove it.",
    "edge.a11yDescription.default": "Press Enter or Space to select an edge. Press Delete to remove it.",
  },
}

const defaultLayoutMode: GraphLayoutMode = "horizontal"
const defaultGraph = modelPresets[0].graph
const emptyNodes: CanvasNode[] = []
const emptyEdges: Edge[] = []

type WorkbenchView = "workspace" | "problems" | "terminal" | "python" | "exchange"

type OutputEntry = { time: string; message: string; error: boolean }

type PendingConnectionMenu = {
  flowPosition: XYPosition
  menuPosition: XYPosition
  fromNodeId: string
  fromHandleId: string | null
  fromHandleType: HandleType
}

function getClientPositionFromPointerEvent(event: MouseEvent | TouchEvent): XYPosition | null {
  if ("changedTouches" in event && event.changedTouches.length > 0) {
    const touch = event.changedTouches[0]
    return { x: touch.clientX, y: touch.clientY }
  }

  if ("clientX" in event && "clientY" in event) {
    return { x: event.clientX, y: event.clientY }
  }

  return null
}

function getPreferredLocale(): Locale {
  if (typeof window === "undefined") {
    return "zh"
  }

  const stored = window.localStorage.getItem("nnmind-locale")
  if (stored === "zh" || stored === "en") {
    return stored
  }

  return navigator.language.toLowerCase().startsWith("zh") ? "zh" : "en"
}

function getStoredPanelState(key: string, fallback: boolean): boolean {
  if (typeof window === "undefined") {
    return fallback
  }

  const stored = window.localStorage.getItem(key)
  return stored === null ? fallback : stored === "true"
}

export default function App() {
  const [locale, setLocale] = useState<Locale>(getPreferredLocale)
  const [layoutMode, setLayoutMode] = useState<GraphLayoutMode>(defaultLayoutMode)
  const [nodes, setNodes, onNodesChange] = useNodesState<CanvasNode>(emptyNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>(emptyEdges)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [view, setView] = useState<WorkbenchView>("workspace")
  const [outputEntries, setOutputEntries] = useState<OutputEntry[]>([])
  const [savedBlocks, setSavedBlocks] = useState(readBlockLibrary)
  const [libraryMessage, setLibraryMessage] = useState<{ nodeId: string; text: string } | null>(null)
  const [inputEditor, setInputEditor] = useState<string | null>(null)
  const [pythonEnvironment, setPythonEnvironment] = useState<PythonEnvironment | null>(null)
  const [pythonSelecting, setPythonSelecting] = useState(false)
  const inspectedInput = useRef("")
  const [runOpen, setRunOpen] = useState(false)
  const [trainingOpen, setTrainingOpen] = useState(false)
  const [runtimeWeights, setRuntimeWeights] = useState<{ model: string; layers: TrainingParameters } | null>(null)
  const handleRuntimeParameters = useCallback((model: string, layers: TrainingParameters) => setRuntimeWeights({ model, layers }), [])
  const [importError, setImportError] = useState("")
  const [pythonLibrary, setPythonLibrary] = useState<{ fileName: string; source: string; sourcePath: string; modules: PythonModule[] } | null>(null)
  const importAttempt = useRef(0)
  const [codeDialog, setCodeDialog] = useState<{ nodeId: string; edit: boolean } | null>(null)
  const [draftJson, setDraftJson] = useState("")
  const [flowInstance, setFlowInstance] = useState<ReactFlowInstance<CanvasNode, Edge> | null>(null)
  const [pendingConnectionMenu, setPendingConnectionMenu] = useState<PendingConnectionMenu | null>(null)
  const [pendingConnectionQuery, setPendingConnectionQuery] = useState("")
  const [leftSidebarCollapsed, setLeftSidebarCollapsed] = useState<boolean>(() =>
    getStoredPanelState("nnmind-left-sidebar-collapsed", false),
  )
  const [rightSidebarCollapsed, setRightSidebarCollapsed] = useState<boolean>(() =>
    getStoredPanelState("nnmind-right-sidebar-collapsed", false),
  )
  const [showMiniMap, setShowMiniMap] = useState(false)
  const [isCanvasFullscreen, setCanvasFullscreen] = useState(false)
  const [inspectorPopup, setInspectorPopup] = useState<{
    nodeId: string
    position: XYPosition
  } | null>(null)

  const text = getUiText(locale)
  const zh = locale === "zh"
  const presetMenuRef = useRef<HTMLDivElement>(null)
  const flowSurfaceRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    window.localStorage.setItem("nnmind-locale", locale)
    document.documentElement.lang = locale === "zh" ? "zh-CN" : "en"
    const desktop = window as Window & { electronAPI?: { setLocale?: (value: Locale) => void } }
    desktop.electronAPI?.setLocale?.(locale)
  }, [locale])

  useEffect(() => {
    window.localStorage.setItem("nnmind-left-sidebar-collapsed", String(leftSidebarCollapsed))
  }, [leftSidebarCollapsed])

  useEffect(() => {
    window.localStorage.setItem("nnmind-right-sidebar-collapsed", String(rightSidebarCollapsed))
  }, [rightSidebarCollapsed])

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setInspectorPopup(null)
        setPendingConnectionMenu(null)
        setPendingConnectionQuery("")
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [])

  function appendOutput(message: string, error = false) {
    const time = new Date().toLocaleTimeString()
    setOutputEntries((entries) => [...entries.slice(-199), { time, message, error }])
  }

  async function copyOutput(value: string, label: string) {
    try {
      await navigator.clipboard.writeText(value)
      appendOutput(text.copied(label))
    } catch {
      appendOutput(text.copyFailed(label), true)
      setView("terminal")
    }
  }

  const graph = useMemo(() => canvasToGraph(nodes, edges), [edges, nodes])
  const inference = useMemo(() => inferGraph(graph), [graph])

  useEffect(() => {
    let changed = false
    const nextNodes = nodes.map((node) => {
      const spec = inference.specsByNodeId[node.id]
      const inputShape = spec?.inputs[0]?.shape
      if (!inputShape || inputShape.length === 0) return node

      const nextParams = { ...node.data.params }
      const lt = node.data.layerType

      function setIfDiff(key: string, actual: unknown) {
        if (actual === null || actual === undefined) return
        if (Array.isArray(actual)) {
          const cur = node.data.params[key]
          if (!Array.isArray(cur) || cur.length !== actual.length || cur.some((v, i) => v !== actual[i])) {
            ;(nextParams as Record<string, unknown>)[key] = [...actual]
            changed = true
          }
          return
        }
        if (node.data.params[key] !== actual) {
          ;(nextParams as Record<string, unknown>)[key] = actual
          changed = true
        }
      }

      const ch = typeof inputShape[1] === "number" ? inputShape[1] : null
      const last = typeof inputShape[inputShape.length - 1] === "number" ? inputShape[inputShape.length - 1] : null
      const third = inputShape.length >= 3 && typeof inputShape[2] === "number" ? inputShape[2] : null

      switch (lt) {
        case "Conv2d":
        case "ResidualBlock2d":
        case "ResNetBasicBlock":
        case "ResNetBottleneck":
        case "PatchEmbedding":
          setIfDiff("in_channels", ch)
          break
        case "BatchNorm2d":
          setIfDiff("num_features", ch)
          break
        case "BatchNorm1d":
          setIfDiff("num_features", last)
          break
        case "Linear":
          setIfDiff("in_features", last)
          break
        case "LSTM":
        case "GRU":
          setIfDiff("input_size", last)
          break
        case "SelfAttention":
          setIfDiff("embed_dim", third)
          break
        case "TransformerEncoder":
        case "TransformerDecoder":
          setIfDiff("d_model", third)
          break
        case "LayerNorm": {
          const ns = node.data.params.normalized_shape
          const nsLen = Array.isArray(ns) ? ns.length : 1
          const trailing = inputShape.slice(inputShape.length - nsLen)
          if (trailing.every((d) => typeof d === "number")) {
            setIfDiff("normalized_shape", trailing as number[])
          }
          break
        }
      }

      if (!changed) return node
      return { ...node, data: { ...node.data, params: nextParams } }
    })

    if (changed) {
      setNodes(nextNodes)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inference])

  const issues = useMemo(() => validateGraph(graph), [graph])
  const nodeParamCounts = useMemo(
    () =>
      Object.fromEntries(graph.nodes.map((node) => [node.id, estimateNodeParamCount(node)])),
    [graph.nodes],
  )
  const totalParamCount = useMemo(
    () => graph.nodes.reduce((total, node) => total + estimateNodeParamCount(node), 0),
    [graph.nodes],
  )
  const decoratedNodes = useMemo(
    () =>
      nodes.map((node) => ({
        ...node,
        data: {
          ...node.data,
          label: getLayerLabel(node.data.layerType, node.data.label, locale),
          description: getLayerDescription(node.data.layerType, node.data.description, locale),
          layoutMode,
          locale,
          paramCount: nodeParamCounts[node.id] ?? 0,
          specs: inference.specsByNodeId[node.id] ?? { inputs: [], outputs: [] },
          issueLevel: issues.find((issue) => issue.nodeId === node.id)?.level,
        },
      })),
    [inference.specsByNodeId, issues, layoutMode, locale, nodeParamCounts, nodes],
  )

  const graphJson = useMemo(() => serializeGraph(graph), [graph])
  const pythonCode = useMemo(() => generatePyTorch(graph), [graph])
  const dataNodes = graph.nodes.filter(node => node.layerType === "Input" && typeof node.params.dataset_code === "string" && node.params.dataset_code.trim())
  const dataNode = dataNodes.find(node => node.params.dataset_active === true) ?? dataNodes[0]
  const dataset: GraphDataset | undefined = dataNode && {
    name: dataNode.name, code: String(dataNode.params.dataset_code), path: String(dataNode.params.dataset_path ?? ""),
  }
  const inputEditorNode = nodes.find(node => node.id === inputEditor)
  const currentWeights = runtimeWeights?.model === pythonCode ? runtimeWeights.layers : undefined
  const symbolNames = useMemo(() => graphSymbols(graph), [graph])
  const boundEdges = useMemo(() => {
    const named = new Set(graph.nodes.filter(node => node.params.output_alias).map(node => node.id))
    return edges.map(edge => named.has(edge.source) ? { ...edge, label: symbolNames.get(edge.source), labelShowBg: true } : edge)
  }, [edges, graph.nodes, symbolNames])
  const codeNode = decoratedNodes.find((node) => node.id === codeDialog?.nodeId)
  const selectedNode = decoratedNodes.find((node) => node.id === selectedNodeId)
  const inspectorPopupNode = inspectorPopup
    ? decoratedNodes.find((node) => node.id === inspectorPopup.nodeId)
    : null
  const connectionMenuLayers = useMemo(() => {
    if (!pendingConnectionMenu) {
      return []
    }

    const keyword = pendingConnectionQuery.trim().toLowerCase()

    return layerLibrary
      .filter((layer) =>
        pendingConnectionMenu.fromHandleType === "source"
          ? layer.inputs.length > 0
          : layer.outputs.length > 0,
      )
      .filter((layer) => {
        if (!keyword) {
          return true
        }

        return [
          layer.type,
          getLayerLabel(layer.type, layer.label, locale),
          getLayerDescription(layer.type, layer.description, locale),
          getLayerCategoryLabel(layer.category, locale),
        ]
          .join(" ")
          .toLowerCase()
          .includes(keyword)
      })
  }, [locale, pendingConnectionMenu, pendingConnectionQuery])

  function fitCanvasSoon() {
    requestAnimationFrame(() => {
      void flowInstance?.fitView({ padding: 0.12, minZoom: 0.025, maxZoom: 1, duration: 190 })
    })
  }

  function loadGraph(nextGraph: NeuralGraph, arrange = false) {
    inspectedInput.current = ""
    setPendingConnectionMenu(null)
    setPendingConnectionQuery("")
    setInspectorPopup(null)
    setCodeDialog(null)
    const loaded = arrange ? compactGraph(nextGraph, layoutMode) : nextGraph
    setNodes(graphToCanvasNodes(loaded, layoutMode))
    setEdges(graphToCanvasEdges(loaded))
    setSelectedNodeId(loaded.nodes[0]?.id ?? null)
    fitCanvasSoon()
  }

  function arrangeCanvas(mode = layoutMode) {
    const arranged = compactGraph(graph, mode)
    const positions = new Map(graphToCanvasNodes(arranged, mode).map((node) => [node.id, node.position]))
    setNodes((current) => {
      const oriented = mode === layoutMode ? current : applyLayoutModeToNodes(current, mode)
      return oriented.map((node) => ({ ...node, position: positions.get(node.id) ?? node.position }))
    })
    fitCanvasSoon()
  }

  function showBlockCode(nodeId: string, edit: boolean) {
    setInspectorPopup(null)
    setCodeDialog({ nodeId, edit })
  }

  function updateNodeParams(nodeId: string, update: (params: CanvasNode["data"]["params"]) => CanvasNode["data"]["params"]) {
    setNodes((current) => current.map((node) => node.id === nodeId ? { ...node, data: { ...node.data, params: update(node.data.params) } } : node))
  }

  function saveBlockCode(nodeId: string, params: CanvasNode["data"]["params"]) {
    updateNodeParams(nodeId, () => params)
    setCodeDialog(null)
  }

  function saveToLibrary(node: CanvasNode) {
    const name = node.data.name.trim() || (zh ? "自定义块" : "Custom Block")
    const existing = savedBlocks.find(block => block.name === name)
    const block: SavedBlock = {
      id: existing?.id ?? crypto.randomUUID(), name, params: structuredClone(node.data.params), width: node.width, height: node.height,
    }
    try {
      const next = existing ? savedBlocks.map(item => item.id === existing.id ? block : item) : [...savedBlocks, block]
      writeBlockLibrary(next)
      setSavedBlocks(next)
      setLibraryMessage({ nodeId: node.id, text: zh ? "已保存到快速添加 → 自建库。" : "Saved in Quick Add → My Blocks." })
    } catch (error) {
      setLibraryMessage({ nodeId: node.id, text: String(error) })
    }
  }

  function removeFromLibrary(id: string) {
    const next = savedBlocks.filter(block => block.id !== id)
    try {
      writeBlockLibrary(next)
      setSavedBlocks(next)
      setLibraryMessage(null)
    } catch (error) {
      appendOutput(String(error), true)
    }
  }

  function changeCustomPorts(node: CanvasNode, count: number) {
    if (edges.some(edge => edge.target === node.id && Number(edge.targetHandle?.slice(2)) >= count)) {
      setLibraryMessage({ nodeId: node.id, text: zh ? "请先断开要移除的端口。" : "Disconnect the ports before removing them." })
      return
    }
    const prior = Array.isArray(node.data.params.input_shapes) ? node.data.params.input_shapes : []
    updateNodeParams(node.id, params => ({
      ...params,
      input_names: Array.from({ length: count }, (_, index) => index ? `x${index + 1}` : "x"),
      input_shapes: Array.from({ length: count }, (_, index) => prior[index] ?? ""),
    }))
    setLibraryMessage(null)
  }

  useEffect(() => {
    if (!trainingDesktop?.getEnvironment) return
    let changed = false
    const unsubscribe = trainingDesktop.onEnvironment(value => {
      if (value) {
        changed = true
        setPythonEnvironment(value)
      }
    })
    void trainingDesktop.getEnvironment().then(value => { if (!changed) setPythonEnvironment(value) }).catch(() => undefined)
    return () => { changed = true; unsubscribe() }
  }, [])

  async function selectLocalPython() {
    if (pythonSelecting) return
    setPythonSelecting(true)
    try {
      const selected = await trainingDesktop?.selectEnvironment?.()
      if (selected) setPythonEnvironment(selected)
    } catch (error) {
      appendOutput(String(error), true)
      setView("terminal")
    } finally {
      setPythonSelecting(false)
    }
  }

  function inputParams(params: CanvasNode["data"]["params"], observed: ObservedInput, source: InputProbeResult["source"]): CanvasNode["data"]["params"] {
    const batchShape = (shape?: number[]) => (shape ?? []).map((dim, index) => index === 0 && dim > 0 ? "B" : dim)
    const dtype = tensorDTypes.find(value => value === observed.dtype)
    return {
      ...params, shape: batchShape(observed.shape), dtype: dtype ?? params.dtype,
      script_kind: observed.kind, script_sample_shape: observed.shape ?? null, script_sample_dtype: observed.dtype ?? null,
      script_source: source, script_output_shape: observed.forwarded ? batchShape(observed.forwarded.shape) : observed.forwardedKind ? [] : null,
      script_output_dtype: observed.forwarded?.dtype ?? null,
    }
  }

  async function syncScriptInputs(nextGraph: NeuralGraph, activeDataset: GraphDataset, forTraining = false) {
    if (!trainingDesktop?.inspectInputs) throw new Error(zh ? "请在桌面版检测输入。" : "Detect inputs in the desktop app.")
    if (!pythonEnvironment?.torch) throw new Error(zh ? "请在右下角选择装有 PyTorch 的 Python 环境。" : "Choose a Python environment with PyTorch in the status bar.")
    const names = graphSymbols(nextGraph)
    const byId = new Map(nextGraph.nodes.map(node => [node.id, node]))
    const ordered = topologicalSort(nextGraph).orderedNodeIds.flatMap(id => {
      const node = byId.get(id)
      const name = names.get(id)
      return node?.layerType === "Input" && name ? [{ id, name, transform: String(node.params.implementation_code ?? "") }] : []
    })
    const preferDataset = forTraining && /^def\s+build_inputs\s*\(/m.test(activeDataset.code)
    const signature = JSON.stringify([pythonEnvironment.path, activeDataset.path, activeDataset.code, ordered, preferDataset])
    if (inspectedInput.current === signature) return nextGraph
    const result = await trainingDesktop.inspectInputs({ script: activeDataset.code, path: activeDataset.path, ports: ordered, preferDataset })
    const observed = new Map(ordered.map(port => [port.id, result.inputs.find(value => value.name === port.name)]))
    const synced = {
      ...nextGraph,
      nodes: nextGraph.nodes.map(node => {
        const input = observed.get(node.id)
        return input ? { ...node, params: inputParams(node.params, input, result.source) } : node
      }),
    }
    const updated = new Map(synced.nodes.filter(node => observed.get(node.id)).map(node => [node.id, node.params]))
    setNodes(current => current.map(node => {
      const params = updated.get(node.id)
      return params ? { ...node, data: { ...node.data, params } } : node
    }))
    inspectedInput.current = signature
    return synced
  }

  async function prepareModel(forTraining = false) {
    const result = dataset ? await syncScriptInputs(graph, dataset, forTraining) : graph
    const error = validateGraph(result).find(issue => issue.level === "error")
    if (error) throw new Error(translateMessage(locale, error.message))
    return generatePyTorch(result)
  }

  function inspectorExtras(node: CanvasNode | undefined) {
    const graphNode = graph.nodes.find(item => item.id === node?.id)
    return {
      portBindings: graphNode ? graphPortBindings(graph, graphNode, symbolNames) : [],
      outputSymbol: node ? symbolNames.get(node.id) : undefined,
      onSaveBlock: () => { if (node) saveToLibrary(node) },
      onEditInput: () => { if (node) { setInspectorPopup(null); setInputEditor(node.id) } },
      onChangePorts: (count: number) => { if (node) changeCustomPorts(node, count) },
      runtimeParameters: node ? currentWeights?.[node.id] ?? (currentWeights ? [] : undefined) : undefined,
      libraryMessage: libraryMessage?.nodeId === node?.id ? libraryMessage?.text : undefined,
    }
  }

  function handleConnect(connection: Connection) {
    setPendingConnectionMenu(null)
    setPendingConnectionQuery("")
    setEdges((current) =>
      addEdge(
        {
          ...connection,
          animated: false,
        },
        current,
      ),
    )
  }

  function handleAddLayer(layerType: CanvasNode["data"]["layerType"], saved?: SavedBlock) {
    setPendingConnectionMenu(null)
    setPendingConnectionQuery("")
    const selectedNode = nodes.find((node) => node.id === selectedNodeId)
    const offset = layoutMode === "vertical"
      ? { x: 0, y: (selectedNode?.height ?? 88) + 56 }
      : { x: (selectedNode?.width ?? 206) + 56, y: 0 }
    const nextPosition = selectedNode
      ? { x: selectedNode.position.x + offset.x, y: selectedNode.position.y + offset.y }
      : undefined
    const newNode = buildCanvasNode(layerType, nodes.length, layoutMode, nextPosition)
    if (saved) {
      newNode.data.name = saved.name
      newNode.data.params = structuredClone(saved.params)
      newNode.width = saved.width ?? newNode.width
      newNode.height = saved.height ?? newNode.height
    }

    setNodes((current) => [...current.map((node) => ({ ...node, selected: false })), { ...newNode, selected: true }])
    setEdges((current) => current.map((edge) => edge.selected ? { ...edge, selected: false } : edge))
    flowSurfaceRef.current?.focus({ preventScroll: true })
    setSelectedNodeId(newNode.id)

    if (!selectedNode) {
      fitCanvasSoon()
      return
    }

    const sourceDef = getLayerDef(selectedNode.data.layerType, selectedNode.data.params)
    const targetDef = getLayerDef(layerType, newNode.data.params)

    if (sourceDef.outputs.length === 0 || targetDef.inputs.length === 0) {
      fitCanvasSoon()
      return
    }

    const outgoing = edges.filter((edge) => edge.source === selectedNode.id)
    const sourcePort = sourceDef.outputs[0].name
    const targetPort = targetDef.inputs[0].name

    if (outgoing.length === 1 && targetDef.outputs.length > 0) {
      const downstream = outgoing[0]
      const relayPort = targetDef.outputs[0].name

      setEdges((current) => [
        ...current.filter((edge) => edge.id !== downstream.id),
        {
          id: `e-${selectedNode.id}-${newNode.id}-${targetPort}`,
          source: selectedNode.id,
          sourceHandle: sourcePort,
          target: newNode.id,
          targetHandle: targetPort,
          animated: false,
        },
        {
          id: `e-${newNode.id}-${downstream.target}-${downstream.targetHandle ?? "in"}`,
          source: newNode.id,
          sourceHandle: relayPort,
          target: downstream.target,
          targetHandle: downstream.targetHandle ?? "in",
          animated: false,
        },
      ])
    } else {
      setEdges((current) => [
        ...current,
        {
          id: `e-${selectedNode.id}-${newNode.id}-${targetPort}`,
          source: selectedNode.id,
          sourceHandle: sourcePort,
          target: newNode.id,
          targetHandle: targetPort,
          animated: false,
        },
      ])
    }

    fitCanvasSoon()
  }

  function handleUpdateName(name: string) {
    if (!selectedNodeId) {
      return
    }

    setNodes((current) =>
      current.map((node) => (node.id === selectedNodeId ? { ...node, data: { ...node.data, name } } : node)),
    )
  }

  function handleUpdateParam(paramName: string, value: ParamValue) {
    if (paramName === "tokenizer_bundle" && selectedNodeId && edges.some(edge => edge.target === selectedNodeId)) {
      setLibraryMessage({ nodeId: selectedNodeId, text: zh ? "切换输入类型前请先断开旧输入连线。" : "Disconnect existing inputs before switching port types." })
      return
    }
    if (!selectedNodeId) {
      return
    }

    setNodes((current) =>
      current.map((node) =>
        node.id === selectedNodeId
          ? {
              ...node,
              data: {
                ...node.data,
                params: {
                  ...node.data.params,
                  [paramName]: value,
                },
              },
            }
          : node,
      ),
    )
  }

  function loadGraphFromJson(raw: string) {
    const parsed = parseGraphJson(raw)
    setImportError("")
    loadGraph(parsed)
    appendOutput(text.imported(text.graphJson))
  }

  function handleConnectEnd(
    event: MouseEvent | TouchEvent,
    connectionState: FinalConnectionState,
  ) {
    if (!flowInstance || !connectionState.fromNode || !connectionState.fromHandle) {
      return
    }

    if (connectionState.toNode || connectionState.toHandle) {
      return
    }

    const clientPosition = getClientPositionFromPointerEvent(event)
    if (!clientPosition) {
      return
    }

    const surfaceRect = flowSurfaceRef.current?.getBoundingClientRect()
    const menuWidth = 320
    const menuHeight = 420
    const menuPosition = surfaceRect
      ? {
          x: Math.min(
            Math.max(clientPosition.x - surfaceRect.left, 16),
            Math.max(16, surfaceRect.width - menuWidth - 16),
          ),
          y: Math.min(
            Math.max(clientPosition.y - surfaceRect.top, 16),
            Math.max(16, surfaceRect.height - menuHeight - 16),
          ),
        }
      : { x: 16, y: 16 }

    setPendingConnectionQuery("")
    setPendingConnectionMenu({
      flowPosition: flowInstance.screenToFlowPosition(clientPosition),
      menuPosition,
      fromNodeId: connectionState.fromNode.id,
      fromHandleId: connectionState.fromHandle.id ?? null,
      fromHandleType: connectionState.fromHandle.type,
    })
  }

  function handleAddLayerFromConnection(layerType: CanvasNode["data"]["layerType"]) {
    if (!pendingConnectionMenu) {
      return
    }

    const newNode = buildCanvasNode(layerType, nodes.length, layoutMode, pendingConnectionMenu.flowPosition)
    const newLayerDef = getLayerDef(layerType)

    setNodes((current) => [...current.map((node) => ({ ...node, selected: false })), { ...newNode, selected: true }])
    setEdges((current) => current.map((edge) => edge.selected ? { ...edge, selected: false } : edge))
    flowSurfaceRef.current?.focus({ preventScroll: true })
    setSelectedNodeId(newNode.id)

    if (pendingConnectionMenu.fromHandleType === "source") {
      const targetPort = newLayerDef.inputs[0]?.name

      if (targetPort) {
        setEdges((current) => [
          ...current,
          {
            id: `e-${pendingConnectionMenu.fromNodeId}-${newNode.id}-${targetPort}`,
            source: pendingConnectionMenu.fromNodeId,
            sourceHandle: pendingConnectionMenu.fromHandleId ?? undefined,
            target: newNode.id,
            targetHandle: targetPort,
            animated: false,
          },
        ])
      }
    } else {
      const sourcePort = newLayerDef.outputs[0]?.name

      if (sourcePort) {
        setEdges((current) => [
          ...current,
          {
            id: `e-${newNode.id}-${pendingConnectionMenu.fromNodeId}-${pendingConnectionMenu.fromHandleId ?? "in"}`,
            source: newNode.id,
            sourceHandle: sourcePort,
            target: pendingConnectionMenu.fromNodeId,
            targetHandle: pendingConnectionMenu.fromHandleId ?? undefined,
            animated: false,
          },
        ])
      }
    }

    setPendingConnectionMenu(null)
    setPendingConnectionQuery("")
  }

  function reportImportError(error: unknown) {
    const message = error instanceof Error ? translateMessage(locale, error.message) : text.importFailed
    setImportError(message)
    appendOutput(message, true)
  }

  async function handleImportModelFile(file: File) {
    const attempt = ++importAttempt.current
    setImportError("")
    setPythonLibrary(null)
    const lowerName = file.name.toLowerCase()
    if (/\.(pt|pth|ckpt)$/.test(lowerName)) {
      reportImportError(new Error(text.unsupportedCheckpoint))
      return
    }

    try {
      if (/\.(onnx|pb)$/.test(lowerName)) {
        const buffer = await file.arrayBuffer()
        if (attempt !== importAttempt.current) return
        loadGraph(importOnnxBuffer(buffer, "horizontal"), true)
      } else {
        const source = await file.text()
        if (attempt !== importAttempt.current) return
        if (/\.(py|pyw)$/.test(lowerName)) {
          const modules = readPythonModuleLibrary(source)
          if (modules.length) {
            setPythonLibrary({ fileName: file.name, source, modules, sourcePath: trainingDesktop?.filePath?.(file) || "" })
            return
          }
        }
        if (lowerName.endsWith(".json")) loadGraph(parseGraphJson(source))
        else loadGraph(importModelSource(source, "horizontal"), true)
      }
      setView("workspace")
      appendOutput(text.imported(file.name))
    } catch (error) {
      if (attempt === importAttempt.current) reportImportError(error)
    }
  }

  function handleImportPythonClass(name: string, includeOptional: boolean) {
    const module = pythonLibrary?.modules.find((item) => item.name === name)
    if (!module || !pythonLibrary) return
    try {
      setImportError("")
      loadGraph(pythonModuleToGraph(pythonLibrary.source, module, includeOptional, pythonLibrary.sourcePath), true)
      appendOutput(text.imported(`${pythonLibrary.fileName} · ${name}`))
      setView("workspace")
    } catch (error) {
      reportImportError(error)
    }
  }

  function handleToggleLayout() {
    const nextLayoutMode: GraphLayoutMode = layoutMode === "horizontal" ? "vertical" : "horizontal"
    setLayoutMode(nextLayoutMode)
    arrangeCanvas(nextLayoutMode)
    appendOutput(formatLayoutMode(locale, nextLayoutMode))
  }

  function deleteCanvasSelection() {
    const selectedNodes = nodes.filter((node) => node.selected)
    const selectedEdges = edges.filter((edge) => edge.selected)
    const nodeIds = new Set(selectedNodes.map((node) => node.id))
    const edgeIds = new Set(selectedEdges.map((edge) => edge.id))
    if (selectedNodes.length === 0 && selectedEdges.length === 0 && selectedNodeId) {
      nodeIds.add(selectedNodeId)
    }
    if (nodeIds.size === 0 && edgeIds.size === 0) return

    setNodes((current) => current.filter((node) => !nodeIds.has(node.id)))
    setEdges((current) => current.filter((edge) =>
      !edgeIds.has(edge.id) && !nodeIds.has(edge.source) && !nodeIds.has(edge.target),
    ))
    if (selectedNodeId && nodeIds.has(selectedNodeId)) setSelectedNodeId(null)
    setInspectorPopup(null)
    setPendingConnectionMenu(null)
    setPendingConnectionQuery("")
  }

  function handleCanvasKeyDown(event: ReactKeyboardEvent<HTMLDivElement>) {
    const target = event.target as HTMLElement
    if (event.key !== "Delete" || event.repeat || event.nativeEvent.isComposing ||
      event.altKey || event.ctrlKey || event.metaKey || event.shiftKey ||
      target.closest("input, textarea, select, [contenteditable], .inspector-popup, .connection-menu")) return
    event.preventDefault()
    deleteCanvasSelection()
  }

  function handleNodeContextMenu(
    event: React.MouseEvent,
    node: CanvasNode,
  ) {
    event.preventDefault()
    setSelectedNodeId(node.id)
    setNodes((current) => current.map((item) => ({ ...item, selected: item.id === node.id })))
    setEdges((current) => current.map((edge) => edge.selected ? { ...edge, selected: false } : edge))
    const surfaceRect = flowSurfaceRef.current?.getBoundingClientRect()
    const popupWidth = 340
    const popupHeight = 480
    const rawX = event.clientX - (surfaceRect?.left ?? 0)
    const rawY = event.clientY - (surfaceRect?.top ?? 0)
    setInspectorPopup({
      nodeId: node.id,
      position: {
        x: Math.min(Math.max(rawX, 8), Math.max(8, (surfaceRect?.width ?? window.innerWidth) - popupWidth - 8)),
        y: Math.min(Math.max(rawY, 8), Math.max(8, (surfaceRect?.height ?? 640) - popupHeight - 8)),
      },
    })
  }

  return (
    <main className={`app-shell${isCanvasFullscreen ? " is-canvas-fullscreen" : ""}`}>
      <header className="toolbar" aria-label={text.toolbar}>
        <strong className="toolbar__brand"><img src={`${import.meta.env.BASE_URL}greatnn-cat.svg`} width="25" height="25" alt="" />GreatNN</strong>
        <div className="toolbar__group">
          <button type="button" onClick={() => {
            loadGraph(defaultGraph)
            appendOutput(text.resetSample)
          }}>
            {text.resetSample}
          </button>
          <button type="button" onClick={handleToggleLayout}>
            {layoutMode === "horizontal" ? text.switchToVertical : text.switchToHorizontal}
          </button>
          <button type="button" popoverTarget="nnmind-presets">
            {text.presetTitle}
          </button>
          <div ref={presetMenuRef} id="nnmind-presets" className="preset-menu" popover="auto">
            <PresetLibrary locale={locale} presets={modelPresets} onLoadPreset={(preset) => {
              loadGraph(preset.graph, true)
              appendOutput(text.imported(preset.title))
              presetMenuRef.current?.hidePopover()
            }} />
          </div>
        </div>
        <div className="toolbar__group">
          <button type="button" aria-pressed={view === "python"} onClick={() => setView(view === "python" ? "workspace" : "python")}>
            {text.pythonPreview}
          </button>
          <button type="button" aria-pressed={view === "exchange"} onClick={() => setView(view === "exchange" ? "workspace" : "exchange")}>
            {text.exportTitle}
          </button>
          <button type="button" onClick={() => void copyOutput(graphJson, text.graphJson)}>{text.copyGraphJson}</button>
          <button type="button" onClick={() => void copyOutput(pythonCode, "PyTorch")}>{text.copyPyTorch}</button>
        </div>
        <button type="button" className="training-launch" onClick={() => setRunOpen(true)}>{zh ? "运行网络" : "Run Network"}</button>
        <button type="button" className="training-launch" onClick={() => setTrainingOpen(true)}>{zh ? "训练网络" : "Train Network"}</button>
        <label className="toolbar__locale">
          <span>{text.language}</span>
          <select value={locale} onChange={(event) => setLocale(event.target.value === "zh" ? "zh" : "en")}>
            <option value="zh">{text.chinese}</option>
            <option value="en">{text.english}</option>
          </select>
        </label>
      </header>

      <section className="workspace-shell" style={{
        "--left-sidebar-width": leftSidebarCollapsed ? "36px" : "236px",
        "--right-sidebar-width": rightSidebarCollapsed ? "36px" : "256px",
      } as CSSProperties}>
        <aside className="sidebar" aria-label={text.leftSidebar}>
          <div className="sidebar__header">
            {!leftSidebarCollapsed && <strong>{text.quickAddTitle}</strong>}
            <button type="button" className="sidebar-toggle" aria-expanded={!leftSidebarCollapsed}
              aria-label={leftSidebarCollapsed ? `${text.expand} ${text.leftSidebar}` : `${text.collapse} ${text.leftSidebar}`}
              title={leftSidebarCollapsed ? text.expand : text.collapse}
              onClick={() => {
                setLeftSidebarCollapsed(!leftSidebarCollapsed)
                fitCanvasSoon()
              }}>
              {leftSidebarCollapsed ? "»" : "«"}
            </button>
          </div>
          {!leftSidebarCollapsed && (
            <div className="sidebar__content">
              <LayerPalette locale={locale} onAddLayer={handleAddLayer} selectedNodeName={selectedNode?.data.name}
                savedBlocks={savedBlocks} onAddSaved={block => handleAddLayer("CustomBlock", block)} onRemoveSaved={removeFromLibrary} />
            </div>
          )}
        </aside>

        <div className="center-stack">
          <section className="canvas-panel">
            <div className="canvas-panel__header">
              <div className="canvas-panel__title">
                <h2>{text.canvasTitle}</h2>
                <span>{formatLayoutMode(locale, layoutMode)}</span>
              </div>
              <button type="button" onClick={() => arrangeCanvas()}>{zh ? "重新排布" : "Arrange"}</button>
              <button type="button" onClick={fitCanvasSoon}>{zh ? "适应画布" : "Fit Canvas"}</button>
              <button type="button" disabled={!selectedNode && !nodes.some((node) => node.selected) && !edges.some((edge) => edge.selected)}
                title="Delete" onClick={deleteCanvasSelection}>{text.deleteSelection}</button>
              <button
                type="button"
                className="canvas-fullscreen-toggle"
                onClick={() => {
                  setCanvasFullscreen((current) => !current)
                  fitCanvasSoon()
                }}
              >
                {isCanvasFullscreen ? text.exitFullscreen : text.fullscreen}
              </button>
            </div>
            <div className="flow-surface" ref={flowSurfaceRef} tabIndex={0} aria-label={text.canvasTitle} onKeyDown={handleCanvasKeyDown}>
              <ReactFlow
                nodes={decoratedNodes}
                edges={boundEdges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                deleteKeyCode={null}
                minZoom={0.025}
                maxZoom={2.5}
                panOnDrag={[0, 1]}
                defaultEdgeOptions={{ type: "smoothstep", markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16 } }}
                ariaLabelConfig={canvasAriaLabels[locale]}
                onConnect={handleConnect}
                onConnectEnd={handleConnectEnd}
                onNodeClick={(_, node) => {
                  setSelectedNodeId(node.id)
                  setInspectorPopup(null)
                }}
                onEdgeClick={() => {
                  setSelectedNodeId(null)
                  setInspectorPopup(null)
                }}
                onNodeContextMenu={(event, node) => handleNodeContextMenu(event, node)}
                onPaneClick={() => {
                  setSelectedNodeId(null)
                  setInspectorPopup(null)
                }}
                onInit={setFlowInstance}
                fitView
                nodeTypes={nodeTypes}
                proOptions={{ hideAttribution: true }}
              >
                <Background gap={24} size={1} color="#cbd5e1" />
                <Controls>
                  <ControlButton onClick={() => setShowMiniMap(!showMiniMap)}
                    title={zh ? "切换缩略图" : "Toggle Minimap"} aria-label={zh ? "切换缩略图" : "Toggle Minimap"} aria-pressed={showMiniMap}>▦</ControlButton>
                </Controls>
                {showMiniMap && <MiniMap pannable zoomable />}
              </ReactFlow>
              {pendingConnectionMenu ? (
                <div
                  className="connection-menu"
                  style={{
                    left: pendingConnectionMenu.menuPosition.x,
                    top: pendingConnectionMenu.menuPosition.y,
                  }}
                >
                  <div className="connection-menu__header">
                    <div>
                      <strong>
                        {pendingConnectionMenu.fromHandleType === "source"
                          ? text.connectionMenuTitleNext
                          : text.connectionMenuTitlePrevious}
                      </strong>
                      <p>
                        {pendingConnectionMenu.fromHandleType === "source"
                          ? text.connectionMenuHintNext
                          : text.connectionMenuHintPrevious}
                      </p>
                    </div>
                    <button
                      type="button"
                      className="connection-menu__close"
                      onClick={() => {
                        setPendingConnectionMenu(null)
                        setPendingConnectionQuery("")
                      }}
                    >
                      {text.cancel}
                    </button>
                  </div>
                  <label className="field connection-menu__search">
                    <input
                      value={pendingConnectionQuery}
                      onChange={(event) => setPendingConnectionQuery(event.target.value)}
                      placeholder={text.connectionMenuSearchPlaceholder}
                      autoFocus
                    />
                  </label>
                  <div className="connection-menu__list">
                    {connectionMenuLayers.length === 0 ? (
                      <div className="connection-menu__empty">{text.connectionMenuEmpty}</div>
                    ) : (
                      connectionMenuLayers.map((layer) => (
                        <button
                          key={layer.type}
                          type="button"
                          className="connection-menu__item"
                          onClick={() => handleAddLayerFromConnection(layer.type)}
                        >
                          <span>{getLayerLabel(layer.type, layer.label, locale)}</span>
                          <small>{getLayerCategoryLabel(layer.category, locale)}</small>
                        </button>
                      ))
                    )}
                  </div>
                </div>
              ) : null}
              {inspectorPopupNode ? (
                <div
                  className="inspector-popup"
                  style={{
                    left: inspectorPopup!.position.x,
                    top: inspectorPopup!.position.y,
                  }}
                >
                  <div className="inspector-popup__header">
                    <strong>{inspectorPopupNode.data.label}</strong>
                    <button
                      type="button"
                      className="inspector-popup__close"
                      aria-label={text.closePanel}
                      onClick={() => setInspectorPopup(null)}
                    >
                      ✕
                    </button>
                  </div>
                  <div className="inspector-popup__body">
                    <Inspector
                      locale={locale}
                      {...inspectorExtras(inspectorPopupNode)}
                      key={inspectorPopupNode.id}
                      node={inspectorPopupNode}
                      onViewCode={() => showBlockCode(inspectorPopupNode.id, false)}
                      onEditCode={() => showBlockCode(inspectorPopupNode.id, true)}
                      onUpdateName={handleUpdateName}
                      onUpdateParam={handleUpdateParam}
                    />
                  </div>
                </div>
              ) : null}
            </div>
          </section>

        </div>

        <aside className="sidebar sidebar--right" aria-label={text.rightSidebar}>
          <div className="sidebar__header">
            {!rightSidebarCollapsed && <strong>{text.inspectorTitle}</strong>}
            <button type="button" className="sidebar-toggle" aria-expanded={!rightSidebarCollapsed}
              aria-label={rightSidebarCollapsed ? `${text.expand} ${text.rightSidebar}` : `${text.collapse} ${text.rightSidebar}`}
              title={rightSidebarCollapsed ? text.expand : text.collapse}
              onClick={() => {
                setRightSidebarCollapsed(!rightSidebarCollapsed)
                fitCanvasSoon()
              }}>
              {rightSidebarCollapsed ? "«" : "»"}
            </button>
          </div>
          {!rightSidebarCollapsed && (
            <div className="sidebar__content">
              <Inspector {...inspectorExtras(selectedNode)} key={selectedNode?.id} locale={locale} node={selectedNode} onUpdateName={handleUpdateName} onUpdateParam={handleUpdateParam}
                onViewCode={() => selectedNode && showBlockCode(selectedNode.id, false)}
                onEditCode={() => selectedNode && showBlockCode(selectedNode.id, true)} />
            </div>
          )}
        </aside>
      </section>

      <section className="workbench" aria-label={text.bottomPanel}>
        <div className="workbench__bar">
          <div className="workbench__views" role="group" aria-label={text.bottomPanel}>
            <button type="button" aria-pressed={view === "workspace"} onClick={() => setView("workspace")}>{text.workspace}</button>
            <button type="button" aria-pressed={view === "problems"} onClick={() => setView("problems")}>
              {text.problems} <span className="workbench__count">{issues.length}</span>
            </button>
            <button type="button" aria-pressed={view === "terminal"} onClick={() => setView("terminal")}>{text.terminal}</button>
            {view === "python" && <span className="workbench__active-title">{text.pythonPreview}</span>}
            {view === "exchange" && <span className="workbench__active-title">{text.exportTitle}</span>}
          </div>
          {view !== "workspace" && (
            <button type="button" className="workbench__close" aria-label={text.closePanel} title={text.closePanel}
              onClick={() => setView("workspace")}>×</button>
          )}
        </div>
        {view !== "workspace" && (
          <div className="workbench__content">
            {view === "problems" && <IssuesPanel locale={locale} issues={issues} />}
            {view === "terminal" && (
              <div className="terminal-output">
                <div className="terminal-output__header">
                  <strong>{text.operationLog}</strong>
                  <button type="button" disabled={outputEntries.length === 0} onClick={() => setOutputEntries([])}>{text.clearLog}</button>
                </div>
                <div className="terminal-output__lines" role="log" aria-label={text.operationLog}>
                  {outputEntries.length === 0 ? <p>{text.emptyLog}</p> : outputEntries.map((entry, index) => (
                    <p key={index} className={entry.error ? "terminal-output__error" : undefined}>
                      <time>{entry.time}</time> {entry.message}
                    </p>
                  ))}
                </div>
              </div>
            )}
            {view === "python" && (
              <section className="python-preview">
                <button type="button" onClick={() => void copyOutput(pythonCode, "PyTorch")}>{text.copyPyTorch}</button>
                <textarea readOnly aria-label={text.pythonPreview} value={pythonCode} spellCheck={false} />
              </section>
            )}
            {view === "exchange" && (
              <ExportPanel locale={locale} graphJson={graphJson} draftJson={draftJson} onDraftJsonChange={setDraftJson}
                onImportJson={() => {
                  try {
                    loadGraphFromJson(draftJson)
                  } catch (error) {
                    reportImportError(error)
                  }
                }}
                onImportModelFile={handleImportModelFile}
                importError={importError}
                pythonLibrary={pythonLibrary}
                onImportPythonClass={handleImportPythonClass}
                onCopyJson={() => void copyOutput(graphJson, text.graphJson)}
              />
            )}
          </div>
        )}
        <footer className="status-bar" aria-label={text.graphStatus}>
          <span title={formatExactParamCount(locale, totalParamCount)}>{text.totalParams} <strong>{formatParamCount(totalParamCount)}</strong></span>
          <span>{text.nodes} <strong>{nodes.length}</strong></span>
          <span>{text.edges} <strong>{edges.length}</strong></span>
          <button className="status-bar__problems" type="button" onClick={() => setView("problems")}>{text.issues} <strong>{issues.length}</strong></button>
          <button className="status-bar__python" type="button" disabled={pythonSelecting} title={pythonEnvironment?.path ?? (zh ? "选择本地 Python 环境" : "Choose local Python environment")}
            onClick={() => void selectLocalPython()}>
            {pythonEnvironment ? `Python ${pythonEnvironment.version}${pythonEnvironment.torch ? "" : zh ? " · 缺少 PyTorch" : " · No PyTorch"}` :
              (zh ? "选择 Python 环境" : "Select Python Interpreter")}
          </button>
        </footer>
      </section>
      {codeDialog && codeNode && (
        <BlockCodeDialog key={codeNode.id} locale={locale} node={codeNode}
          inputSymbols={inspectorExtras(codeNode).portBindings.map(binding => binding.local)}
          connectedInputs={inspectorExtras(codeNode).portBindings.flatMap(binding => binding.source ? [`${binding.local} <- ${binding.source}`] : [])}
          outputSymbol={symbolNames.get(codeNode.id)}
          outputConnections={edges.filter(edge => edge.source === codeNode.id).length}
          onClose={() => setCodeDialog(null)}
          onSave={codeDialog.edit ? (params) => saveBlockCode(codeNode.id, params) : undefined} />
      )}
      {inputEditorNode && <InputScriptDialog key={inputEditorNode.id} zh={zh} causal={graph.nodes.some(node => node.layerType === "CausalLM")}
        initial={String(inputEditorNode.data.params.dataset_code ?? "")}
        initialPath={String(inputEditorNode.data.params.dataset_path ?? "")}
        initialTransform={String(inputEditorNode.data.params.implementation_code ?? "")}
        transformTemplate={implementationTemplate({ id: inputEditorNode.id, position: inputEditorNode.position, ...inputEditorNode.data }, undefined, symbolNames.get(inputEditorNode.id))}
        onClose={() => setInputEditor(null)}
        onSave={async (code, path, transform, detect) => {
          inspectedInput.current = ""
          const updated = graph.nodes.map(node => {
            if (node.layerType !== "Input") return node
            if (node.id !== inputEditorNode.id) {
              return node.params.dataset_active ? { ...node, params: { ...node.params, dataset_active: false } } : node
            }
            return { ...node, params: { ...node.params, dataset_code: code, dataset_path: path,
              dataset_active: Boolean(code.trim()), implementation_code: transform || null,
              script_source: null, script_kind: null, script_sample_shape: null, script_sample_dtype: null, script_output_shape: null, script_output_dtype: null } }
          })
          const next = { ...graph, nodes: updated }
          if (detect) {
            await syncScriptInputs(next, { name: inputEditorNode.data.name, code, path })
          } else {
            const params = new Map(updated.filter(node => node.layerType === "Input").map(node => [node.id, node.params]))
            setNodes(current => current.map(node => params.has(node.id) ? { ...node, data: { ...node.data, params: params.get(node.id)! } } : node))
          }
          setInputEditor(null)
        }} />}
      <RunDialog open={runOpen} zh={zh} prepareModel={prepareModel}
        dataset={dataset} environment={pythonEnvironment}
        basePath={String(graph.nodes.find(node => node.layerType === "CausalLM")?.params.model_path ?? "")}
        onClose={() => setRunOpen(false)} onParameters={handleRuntimeParameters} />
      <TrainingDialog symbols={[...symbolNames.values()]} causal={graph.nodes.some(node => node.layerType === "CausalLM")} open={trainingOpen} zh={zh} prepareModel={prepareModel} dataset={dataset} environment={pythonEnvironment}
        onClose={() => setTrainingOpen(false)} onParameters={handleRuntimeParameters} />
    </main>
  )
}
