import { useEffect, useRef, useState } from "react"
import { causalDatasetTemplate, datasetTemplate } from "../core/trainingContracts"
import { pythonDiagnostics, readPythonModuleLibrary } from "../core/import/pythonModuleLibrary"
import { PythonEditor } from "./PythonEditor"
import { trainingDesktop } from "./trainingDesktop"

export function InputScriptDialog({ zh, initial, initialPath, causal, initialTransform, transformTemplate, onSave, onClose }: {
  zh: boolean
  initial: string
  initialPath: string
  causal: boolean
  initialTransform: string
  transformTemplate: string
  onSave: (code: string, path: string, transform: string, detect: boolean) => Promise<void>
  onClose: () => void
}) {
  const dialog = useRef<HTMLDialogElement>(null)
  const template = causal ? causalDatasetTemplate : datasetTemplate
  const [source, setSource] = useState(initial || template)
  const [path, setPath] = useState(initialPath)
  const [tab, setTab] = useState<"script" | "transform">("script")
  const [transform, setTransform] = useState(initialTransform)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  useEffect(() => { dialog.current?.showModal() }, [])
  async function save(detect: boolean) {
    if (source.trim() && (pythonDiagnostics(source).length || !/^def\s+(?:build_datasets|build_inputs)\s*\(/m.test(source))) {
      setError(zh ? "请修正语法并定义 build_datasets() 或 build_inputs()。" : "Fix syntax and define build_datasets() or build_inputs().")
      return
    }
    if (transform && (pythonDiagnostics(transform).length ||
      !readPythonModuleLibrary(transform).some(block => block.name === "NNMindImplementation" && block.inputs.length === 1 && !block.inputs[0].keywordOnly))) {
      setError(zh ? "输入转换须保留 NNMindImplementation 类及一个位置输入。" : "Keep NNMindImplementation and one positional input.")
      return
    }
    setBusy(true)
    setError("")
    try { await onSave(source, path, transform, detect && Boolean(source.trim())) }
    catch (cause) { setError(String(cause)); setBusy(false) }
  }

  return (
    <dialog ref={dialog} className="block-code-dialog input-script-dialog" onCancel={event => { event.preventDefault(); if (!busy) onClose() }}>
      <header><h2>{zh ? "输入设置" : "Input Settings"}</h2><button type="button" disabled={busy} onClick={onClose} aria-label={zh ? "关闭" : "Close"}>×</button></header>
      <div className="palette-tabs" inert={busy}>
        <button type="button" aria-pressed={tab === "script"} onClick={() => { setTab("script"); setError("") }}>{zh ? "输入脚本" : "Data Script"}</button>
        <button type="button" aria-pressed={tab === "transform"} onClick={() => { setTab("transform"); setError("") }}>{zh ? "输入转换 · 进阶" : "Input Transform"}</button>
      </div>
      <div className="input-script-body" inert={busy}>
      {tab === "script" ? <>
      <p>{zh ? "定义 build_datasets()，返回 train、validation 和 test。每项包含 (输入, 目标)，多输入可用元组或按模型参数名组成字典。运行画布使用验证集第一批。仅需推理时，可定义 build_inputs() 直接返回输入。「保存并检测」会运行脚本读取一个样本，同步画布形状和数据类型。" :
        "Define build_datasets() and return train, validation and test. Each item is (inputs, target). Use a tuple or named dictionary for multiple inputs. Canvas uses the first validation batch, or build_inputs() for inference only. Save and Detect runs the script against one sample and synchronizes shape and dtype."}</p>
      <div className="script-actions">
        <label className="file-action">
          <input type="file" accept=".py" onChange={async event => {
            const file = event.currentTarget.files?.[0]
            event.currentTarget.value = ""
            if (!file) return
            try {
              setSource(await file.text())
              setPath(trainingDesktop?.filePath?.(file) || "")
              setError("")
            } catch (error) {
              setError(String(error))
            }
          }} /><span>{zh ? "导入 .py 脚本" : "Import .py Script"}</span>
        </label>
        <span title={path}>{path ? path.split(/[\\/]/).pop() : zh ? "本地编辑" : "Inline script"}</span>
        <button type="button" onClick={() => { setSource(template); setPath(""); setError("") }}>{zh ? "载入示例" : "Load Example"}</button>
      </div>
      <PythonEditor sourcePath={path} value={source} onChange={setSource} label={zh ? "数据 Python 脚本" : "Dataset Python script"} />
      </> : <>
        <p>{zh ? "数据脚本提供一批输入。这个可选的 forward 转换只处理当前端口的值，然后把结果传给下游节点。" : "The data script supplies a batch. This optional forward transform processes the current port before passing it downstream."}</p>
        <button type="button" onClick={() => setTransform("")}>{zh ? "恢复原样传递" : "Restore Pass-through"}</button>
        <PythonEditor key="transform" value={transform || transformTemplate} onChange={setTransform}
          imports={"import torch\nimport torch.nn as nn\n"} label={zh ? "输入转换代码" : "Input transform code"} />
      </>}
      </div>
      {error && <p role="alert" className="import-error">{error}</p>}
      <footer>
        <button type="button" disabled={busy} onClick={onClose}>{zh ? "关闭" : "Close"}</button>
        <button type="button" disabled={busy} onClick={() => void save(false)}>{zh ? "仅保存" : "Save Only"}</button>
        <button type="button" disabled={busy} className="action-primary" onClick={() => void save(true)}>
          {busy ? (zh ? "正在检测…" : "Detecting…") : (zh ? "保存并检测" : "Save and Detect")}
        </button>
      </footer>
    </dialog>
  )
}
