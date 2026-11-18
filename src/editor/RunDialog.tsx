import { useEffect, useRef, useState } from "react"
import { trainingDesktop } from "./trainingDesktop"
import type { GraphDataset, PythonEnvironment, TrainingParameters, TrainingRequest } from "../core/trainingContracts"

export function RunDialog({ open, zh, prepareModel, dataset, environment, basePath, onClose, onParameters }: {
  open: boolean; zh: boolean; prepareModel: () => Promise<string>; dataset?: GraphDataset; environment: PythonEnvironment | null; basePath: string;
  onClose: () => void; onParameters: (model: string, layers: TrainingParameters) => void
}) {
  const ref = useRef<HTMLDialogElement>(null)
  const modelAtStart = useRef("")
  const [mode, setMode] = useState<"generate" | "execute">("execute")
  const [path, setPath] = useState("")
  const [adapter, setAdapter] = useState("")
  const [prompt, setPrompt] = useState("")
  const [details, setDetails] = useState<string>()
  const [tokens, setTokens] = useState(24)
  const [device, setDevice] = useState<TrainingRequest["device"]>("auto")
  const [precision, setPrecision] = useState("float32")
  const [detecting, setDetecting] = useState(false)
  const [busy, setBusy] = useState(false)
  const [output, setOutput] = useState<string>()
  const [error, setError] = useState("")
  const [log, setLog] = useState("")
  useEffect(() => {
    const dialog = ref.current
    if (!dialog) return
    if (open && !dialog.open) {
      dialog.showModal()
    }
    if (!open && dialog.open) dialog.close()
  }, [open])
  useEffect(() => trainingDesktop?.onTraining?.(event => {
    if (event.scope !== "execute") return
    if (event.type === "result") { setOutput(event.text); setDetails(event.details) }
    if (event.type === "parameters") onParameters(modelAtStart.current, event.layers)
    if (event.type === "log") setLog(previous => (previous + event.message + "\n").slice(-10000))
    if (event.type === "finished") {
      setBusy(false)
      if (event.status === "failed") setError(event.message || (zh ? "运行失败，请查看日志。" : "Execution failed. See logs."))
      if (event.status === "stopped") setError(zh ? "已停止。" : "Stopped.")
    }
  }), [onParameters, zh])

  async function run() {
    if (busy) return
    setError(""); setLog(""); setOutput(undefined); setDetails(undefined)
    if (!trainingDesktop?.startTraining) {
      setError(zh ? "请在 Windows 桌面版中运行。" : "Use the desktop application.")
      return
    }
    if (!environment?.torch) {
      setError(zh ? "请先在右下角选择装有 PyTorch 的环境。" : "Choose a Python environment with PyTorch from the status bar.")
      return
    }
    if (mode === "execute") {
      if (!dataset) { setError(zh ? "请先在输入节点保存输入脚本。" : "Save a script on an Input node first."); return }
    } else if (!(path || basePath) || !prompt.trim()) {
      setError(zh ? "请选择本地模型目录并输入提问。" : "Choose a local model and enter a prompt.")
      return
    }
    setBusy(true)
    try {
      setDetecting(mode === "execute")
      const model = mode === "execute" ? await prepareModel() : ""
      setDetecting(false)
      modelAtStart.current = model
      await trainingDesktop.startTraining({
        model, dataset: dataset?.code ?? "", datasetPath: dataset?.path ?? "", loss: "mse", lossCode: "", epochs: 1, batchSize: 1,
        learningRate: 0.001, optimizer: "adam", device, mode, canvasTokens: basePath && mode === "execute" ? tokens : undefined,
        generation: { path: path || basePath, adapter, prompt, tokens, precision },
      })
    } catch (cause) { setDetecting(false); setBusy(false); setError(String(cause)) }
  }

  function switchMode(value: typeof mode) {
    if (value === mode) return
    setMode(value)
    setError(""); setLog(""); setOutput(undefined); setDetails(undefined)
  }

  async function choose(set: (value: string) => void) {
    try { const path = await trainingDesktop?.chooseDirectory(); if (path) set(path) }
    catch (cause) { setError(String(cause)) }
  }

  return <dialog ref={ref} className="training-dialog run-dialog" aria-labelledby="run-title" onCancel={event => { event.preventDefault(); onClose() }}>
    <header><div><h2 id="run-title">{zh ? "运行网络" : "Run Network"}</h2><span>{zh ? "本地推理" : "Local inference"}</span></div>
      <button type="button" aria-label={zh ? "关闭" : "Close"} onClick={onClose}>×</button></header>
    <div className="run-content">
      <div className="palette-tabs">
        <button type="button" disabled={busy} aria-pressed={mode === "execute"} onClick={() => switchMode("execute")}>{zh ? "运行画布" : "Run Canvas"}</button>
        <button type="button" disabled={busy} aria-pressed={mode === "generate"} onClick={() => switchMode("generate")}>{zh ? "大模型问答" : "Ask a Model"}</button>
      </div>
      <fieldset className="run-options" disabled={busy}>
        {mode === "generate" && <>
          <div className="run-path"><button type="button" onClick={() => void choose(setPath)}>{zh ? "选择本地模型" : "Choose Local Model"}</button><small title={path || basePath}>{path || basePath || (zh ? "请选择包含分词器和权重的目录" : "Choose the folder containing tokenizer and weights")}</small></div>
          <div className="run-path"><button type="button" onClick={() => void choose(setAdapter)}>{zh ? "加载适配器（可选）" : "Load Adapter (Optional)"}</button><small>{adapter}</small>{adapter && <button type="button" onClick={() => setAdapter("")}>{zh ? "移除" : "Clear"}</button>}</div>
          <label>{zh ? "权重精度" : "Precision"} <select value={precision} onChange={event => setPrecision(event.target.value)}><option>float32</option><option>bfloat16</option><option>float16</option></select></label>
        </>}
        {(mode === "generate" || basePath) && <label>{zh ? "最多生成 token" : "Max new tokens"} <input type="number" min={1} max={128} value={tokens} onChange={event => setTokens(Number(event.target.value))} /></label>}
        <label>{zh ? "设备" : "Device"} <select value={device} onChange={event => setDevice(event.target.value as TrainingRequest["device"])}><option value="auto">{zh ? "自动" : "Auto"}</option><option value="cpu">CPU</option><option value="cuda">CUDA</option></select></label>
      </fieldset>
      {mode === "generate" ? <label className="run-input">
        <strong>{zh ? "提问" : "Prompt"}</strong>
        <textarea disabled={busy} value={prompt} onChange={event => setPrompt(event.target.value)} placeholder={zh ? "输入想问的内容…" : "Enter your question…"} />
      </label> : <p className="run-source">{dataset ? (zh ? `输入来源：${dataset.name} · 推理输入或验证集首批` : `Source: ${dataset.name} · inference inputs or first validation batch`)
        : (zh ? "请在输入节点的「输入设置」里保存脚本。" : "Save a script in the Input node’s settings.")}</p>}
      <section className="run-result" aria-live="polite"><strong>{zh ? "输出" : "Output"}</strong><pre>{(output === "" ? (zh ? "模型未生成可显示的文字。" : "No displayable text was generated.") : output) ?? (busy ? (detecting ? (zh ? "正在检测脚本输入…" : "Detecting script inputs…") : (zh ? "正在加载或运行模型…" : "Loading or running model…")) : "—")}</pre></section>
      {details && <details><summary>{zh ? "张量详情" : "Tensor Details"}</summary><pre className="training-log">{details}</pre></details>}
      {error && <p className="import-error" role="alert">{error}</p>}
      {log && <details><summary>{zh ? "日志" : "Logs"}</summary><pre className="training-log">{log}</pre></details>}
    </div>
    <footer>
      <button type="button" disabled={!busy || detecting} onClick={() => void trainingDesktop?.stopTraining?.().catch(cause => setError(String(cause)))}>{zh ? "停止" : "Stop"}</button>
      <button type="button" disabled={busy} className="action-primary" onClick={() => void run()}>{mode === "generate" ? (zh ? "发送" : "Send") : (zh ? "运行" : "Run")}</button>
    </footer>
  </dialog>
}
