import { useEffect, useRef, useState } from "react"
import { lossTemplate, type GraphDataset, type LossPoint, type PythonEnvironment, type TrainingParameters, type TrainingRequest } from "../core/trainingContracts"
import { pythonDiagnostics } from "../core/import/pythonModuleLibrary"
import { PythonEditor } from "./PythonEditor"
import { trainingDesktop } from "./trainingDesktop"

function LossChart({ points, zh }: { points: LossPoint[]; zh: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const element = canvas.current
    const context = element?.getContext("2d")
    if (!element || !context) return
    function draw() {
      if (!element || !context) return
      const width = element.clientWidth
      const height = element.clientHeight
      if (width === 0 || height === 0) return
      const scale = window.devicePixelRatio || 1
      element.width = width * scale
      element.height = height * scale
      context.scale(scale, scale)
      const left = 60, top = 16, chartWidth = Math.max(1, width - left - 15), chartHeight = Math.max(1, height - top - 30)
      const values = points.flatMap(point => [point.train, point.validation])
      const minimum = Math.min(0, ...values)
      const maximum = Math.max(...values, minimum === 0 ? 0.001 : 0)
      const span = maximum - minimum
      context.font = "11px Consolas"
      context.textAlign = "right"
      for (let index = 0; index <= 4; index++) {
        const y = top + chartHeight * index / 4
        context.strokeStyle = "#e1e8f2"
        context.beginPath()
        context.moveTo(left, y)
        context.lineTo(left + chartWidth, y)
        context.stroke()
        context.fillStyle = "#47617d"
        context.fillText((maximum - span * index / 4).toPrecision(3), left - 7, y + 4)
      }
      context.textAlign = "center"
      const stride = Math.max(1, Math.ceil(points.length / 8))
      points.forEach((point, index) => {
        if (index % stride === 0 || index === points.length - 1) {
          context.fillText(String(point.epoch), left + chartWidth * index / Math.max(1, points.length - 1), top + chartHeight + 16)
        }
      })
      for (const [key, color] of [["train", "#2466c6"], ["validation", "#087d6b"]] as const) {
        context.strokeStyle = color
        context.fillStyle = color
        context.lineWidth = 2
        context.beginPath()
        points.forEach((point, index) => {
          const x = left + chartWidth * index / Math.max(1, points.length - 1)
          const y = top + chartHeight * (maximum - point[key]) / span
          if (!index) context.moveTo(x, y)
          else context.lineTo(x, y)
        })
        context.stroke()
        if (points.length === 1) {
          context.beginPath()
          context.arc(left, top + chartHeight * (maximum - points[0][key]) / span, 3, 0, 2 * Math.PI)
          context.fill()
        }
      }
    }
    const resize = new ResizeObserver(draw)
    resize.observe(element)
    draw()
    return () => resize.disconnect()
  }, [points])
  return (
    <div className="loss-chart">
      <div className="loss-chart__legend">
        <strong>LOSS</strong><span className="train-swatch">{zh ? "训练集" : "Train"}</span><span className="val-swatch">{zh ? "验证集" : "Validation"}</span>
      </div>
      {points.length ? <canvas ref={canvas} aria-label={zh ? "训练与验证损失曲线" : "Training and validation loss curves"} /> :
        <div className="loss-chart__empty">{zh ? "完成第一个轮次后显示曲线" : "Curves appear after the first epoch"}</div>}
    </div>
  )
}

export function TrainingDialog({ open, zh, prepareModel, dataset, environment, symbols, causal, onClose, onParameters }: {
  open: boolean
  zh: boolean
  prepareModel: (forTraining: boolean) => Promise<string>
  dataset?: GraphDataset
  environment: PythonEnvironment | null
  symbols: string[]
  causal: boolean
  onClose: () => void
  onParameters: (model: string, layers: TrainingParameters) => void
}) {
  const dialog = useRef<HTMLDialogElement>(null)
  const runningModel = useRef("")
  const [lossSelection, setLoss] = useState<TrainingRequest["loss"] | "auto">("auto")
  const loss = lossSelection === "auto" ? (causal ? "causal" : "mse") : lossSelection
  const [insertion, setInsertion] = useState<{ text: string }>()
  const [savePath, setSavePath] = useState("")
  const [lossCode, setLossCode] = useState(lossTemplate)
  const [epochs, setEpochs] = useState("10")
  const [batchSize, setBatchSize] = useState("")
  const [learningRate, setLearningRate] = useState("0.001")
  const [optimizer, setOptimizer] = useState<TrainingRequest["optimizer"]>("adam")
  const [device, setDevice] = useState<TrainingRequest["device"]>("auto")
  const [points, setPoints] = useState<LossPoint[]>([])
  const [status, setStatus] = useState<"idle" | "running" | "completed" | "failed" | "stopped">("idle")
  const [detecting, setDetecting] = useState(false)
  const [error, setError] = useState("")
  const [logs, setLogs] = useState<string[]>([])
  const [ready, setReady] = useState<{ device: string; output: string } | null>(null)
  const [testLoss, setTestLoss] = useState<number | null>(null)
  const latest = points.at(-1)
  const busy = status === "running"

  useEffect(() => {
    if (open && !dialog.current?.open) dialog.current?.showModal()
    if (!open && dialog.current?.open) dialog.current.close()
  }, [open])
  useEffect(() => {
    return trainingDesktop?.onTraining?.(event => {
      if (event.scope === "execute") return
      if (event.type === "parameters") onParameters(runningModel.current, event.layers)
      else if (event.type === "ready") setReady(event)
      else if (event.type === "epoch") setPoints(points => [...points, event])
      else if (event.type === "test") setTestLoss(event.loss)
      else if (event.type === "log") setLogs(lines => [...lines.slice(-89), event.message])
      else if (event.type === "finished") {
        setStatus(event.status)
        if (event.status === "failed") setError(event.message || (zh ? "Python 任务失败。请查看日志。" : "Python failed. See the log."))
      }
    })
  }, [onParameters, zh])

  async function start(mode: TrainingRequest["mode"]) {
    if (busy) return
    if (!trainingDesktop?.startTraining || !environment?.torch || !dataset) {
      setError(!environment?.torch
        ? (zh ? "请在右下角选择装有 PyTorch 的环境。" : "Choose Python with PyTorch from the status bar.")
        : (zh ? "请在输入节点保存数据脚本。" : "Save a dataset script on the Input node."))
      return
    }
    if (!/^def\s+build_datasets\s*\(/m.test(dataset.code)) {
      setError(zh ? "训练需要输入脚本的 build_datasets() 提供训练、验证和测试集。" : "Training requires build_datasets() with train, validation and test splits.")
      return
    }
    if (mode === "train" && causal && !savePath) {
      setError(zh ? "请选择 LoRA 适配器保存目录。" : "Choose a folder for the LoRA adapter.")
      return
    }
    if (mode === "train" && loss === "custom" && (pythonDiagnostics(lossCode).length || !/^def\s+compute_loss\s*\(/m.test(lossCode))) {
      setError(zh ? "请修正 LOSS 脚本，并定义 compute_loss(prediction, target)。" : "Fix the LOSS script and define compute_loss(prediction, target).")
      return
    }
    setPoints([]); setError(""); setReady(null); setTestLoss(null); setLogs([]); setStatus("running")
    try {
      setDetecting(true)
      const model = await prepareModel(true)
      setDetecting(false)
      runningModel.current = model
      await trainingDesktop.startTraining({ model, dataset: dataset.code, datasetPath: dataset.path, loss, lossCode, savePath, epochs: Number(epochs),
        batchSize: Number(batchSize || (causal ? "1" : "16")), learningRate: Number(learningRate), optimizer, device, mode })
    } catch (error) { setDetecting(false); setError(String(error)); setStatus("failed") }
  }

  return (
    <dialog className="training-dialog" ref={dialog} onCancel={event => { event.preventDefault(); onClose() }}>
      <header><div><h2>{zh ? "训练网络" : "Train Network"}</h2><span>{zh ? "PyTorch 本地运行" : "Local PyTorch runtime"}</span></div>
        <button type="button" onClick={onClose} aria-label={zh ? "关闭" : "Close"}>×</button></header>
      <div className="training-grid">
        <div className="training-options">
          <fieldset disabled={busy}>
            <legend>{zh ? "运行设备" : "Compute"}</legend>
            <label>{zh ? "设备" : "Device"}<select value={device} onChange={event => setDevice(event.target.value as TrainingRequest["device"])}>
              <option value="auto">{zh ? "自动" : "Auto"}</option><option value="cpu">CPU</option><option value="cuda">CUDA</option>
            </select></label>
          </fieldset>
          <fieldset disabled={busy}>
            <legend>{zh ? "训练设置" : "Training Settings"}</legend>
            <label>LOSS<select value={loss} onChange={event => setLoss(event.target.value as TrainingRequest["loss"])}>
              <option value="causal">Causal LM</option><option value="mse">MSE</option><option value="crossEntropy">CrossEntropy</option><option value="l1">L1</option><option value="bce">BCEWithLogits</option>
              <option value="custom">{zh ? "自定义 Python" : "Custom Python"}</option>
            </select></label>
            <label>{zh ? "优化器" : "Optimizer"}<select value={optimizer} onChange={event => setOptimizer(event.target.value as TrainingRequest["optimizer"])}>
              <option value="adam">Adam</option><option value="sgd">SGD</option>
            </select></label>
            <label>{zh ? "轮次" : "Epochs"}<input type="number" min="1" max="10000" value={epochs} onChange={event => setEpochs(event.target.value)} /></label>
            <label>{zh ? "批大小" : "Batch size"}<input type="number" min="1" max="10000" value={batchSize || (causal ? "1" : "16")} onChange={event => setBatchSize(event.target.value)} /></label>
            <label>{zh ? "学习率" : "Learning rate"}<input type="number" min="0.000001" max="10" step="0.0001" value={learningRate} onChange={event => setLearningRate(event.target.value)} /></label>
            {causal && <button type="button" title={savePath} onClick={async () => {
              try { const path = await trainingDesktop?.chooseDirectory(); if (path) setSavePath(path) }
              catch (error) { setError(String(error)) }
            }}>{savePath ? (zh ? "更换适配器保存目录" : "Change Adapter Folder") : (zh ? "选择适配器保存目录" : "Choose Adapter Folder")}</button>}
            {causal && <small>{zh ? "Causal LM 使用右移标签，-100 忽略该位置。" : "Causal LM shifts labels and ignores -100."}</small>}
          </fieldset>
        </div>
        <div className="training-results">
          <div className="training-results-title"><strong>{zh ? "训练进度" : "Progress"}</strong><span>{latest ? `${latest.epoch} / ${epochs}` : "—"} {zh ? "轮" : "epochs"}</span></div>
          {loss === "custom" && !busy && <><div className="loss-symbols">
            <strong>{zh ? "插入图变量" : "Insert Graph Tensor"}</strong>
            <select aria-label={zh ? "插入图变量" : "Insert Graph Tensor"} value="" onChange={event => {
              if (event.target.value) setInsertion({ text: `tensors[${JSON.stringify(event.target.value)}]` })
            }}><option value="">{zh ? "选择变量…" : "Choose…"}</option>{symbols.map(name => <option key={name}>{name}</option>)}</select>
            <small>{zh ? "LOSS 可接收 prediction、target、tensors 与 inputs。" : "LOSS can accept prediction, target, tensors and inputs."}</small>
          </div><div className="loss-code"><PythonEditor insertion={insertion} symbols={symbols} value={lossCode} onChange={setLossCode} label={zh ? "自定义 LOSS 函数" : "Custom LOSS function"} /></div></>}
          <LossChart points={points} zh={zh} />
          <div className="training-metrics">
            <span>{zh ? "训练" : "Train"} <strong>{latest?.train.toPrecision(5) ?? "—"}</strong></span>
            <span>{zh ? "验证" : "Validation"} <strong>{latest?.validation.toPrecision(5) ?? "—"}</strong></span>
            <span>{zh ? "测试" : "Test"} <strong>{testLoss?.toPrecision(5) ?? "—"}</strong></span>
          </div>
          {ready && <p className="training-detail">{ready.device.toUpperCase()} · {zh ? "验证批次输出" : "Validation batch output"} {ready.output}</p>}
          {logs.length > 0 && <details><summary>{zh ? "运行日志" : "Runtime Log"}</summary><pre className="training-log">{logs.join("\n")}</pre></details>}
        </div>
      </div>
      {(error) && <p role="alert" className="import-error">{error}</p>}
      <footer>
        <span role="status" data-training-status={status}>{({
          idle: zh ? "就绪" : "Ready", running: detecting ? (zh ? "正在检测脚本输入…" : "Detecting script inputs…") : (zh ? "运行中，关闭窗口后仍会继续" : "Running in the background if closed"),
          completed: zh ? "已完成" : "Completed", failed: zh ? "运行失败" : "Failed", stopped: zh ? "已停止" : "Stopped",
        })[status]}</span>
        <button type="button" disabled={busy} onClick={() => void start("preview")}>{zh ? "预览验证集" : "Preview Validation"}</button>
        <button type="button" disabled={busy} className="action-primary" onClick={() => void start("train")}>{zh ? "开始训练" : "Start Training"}</button>
        <button type="button" disabled={!busy || detecting} onClick={() => void trainingDesktop?.stopTraining?.().catch(error => setError(String(error)))}>{zh ? "停止" : "Stop"}</button>
        <button type="button" onClick={onClose}>{zh ? "关闭" : "Close"}</button>
      </footer>
    </dialog>
  )
}
