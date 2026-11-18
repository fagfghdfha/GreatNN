import { useEffect, useRef, useState } from "react"
import { describeLayerCode, implementationClassName, implementationSymbols, implementationTemplate } from "../core/codegen/generatePyTorch"
import { pythonDiagnostics, readPythonModuleLibrary } from "../core/import/pythonModuleLibrary"
import { customClassTemplate, customInputNames, manualClassName } from "../core/graph/customBlock"
import type { Locale } from "../i18n"
import type { ParamValue } from "../core/graph/types"
import { PythonEditor } from "./PythonEditor"
import type { CanvasNode } from "./types"

export function BlockCodeDialog({ locale, node, inputSymbols, connectedInputs, outputConnections, outputSymbol, onClose, onSave }: {
  locale: Locale
  node: CanvasNode
  inputSymbols: string[]
  connectedInputs: string[]
  outputConnections: number
  outputSymbol?: string
  onClose: () => void
  onSave?: (params: Record<string, ParamValue>) => void
}) {
  const zh = locale === "zh"
  const ref = useRef<HTMLDialogElement>(null)
  const params = node.data.params
  const source = typeof params.block_source === "string" ? params.block_source : ""
  const builtin = node.data.layerType !== "CustomBlock"
  const modelNode = { id: node.id, position: node.position, ...node.data }
  const symbols = implementationSymbols(modelNode, inputSymbols, outputSymbol)
  const className = builtin ? implementationClassName : source ? String(params.class_name) : manualClassName
  const [sourceClass] = useState(() => {
    try {
      return source ? readPythonModuleLibrary(source).find((module) => module.name === className) : undefined
    } catch { return undefined }
  })
  const [value, setValue] = useState(() => builtin ? String(params.implementation_code || (onSave ? implementationTemplate(modelNode, inputSymbols, outputSymbol) : "")) : sourceClass?.code ?? (
    source ? "" : String(params.class_code || customClassTemplate(params, connectedInputs, outputConnections, symbols))
  ))
  const [error, setError] = useState("")
  const description = onSave || source ? null : describeLayerCode({ id: node.id, position: node.position, ...node.data }, inputSymbols, outputSymbol)

  useEffect(() => { ref.current?.showModal() }, [])

  function save() {
    const code = value.replace(/\r\n?/g, "\n").trimEnd()
    if (!code.trim() || pythonDiagnostics(code).length) {
      setError(zh ? "请修正 Python 语法和缩进。" : "Fix Python syntax and indentation.")
      return
    }
    const edited = readPythonModuleLibrary(code).find(module => module.name === className)
    if (builtin) {
      const expected = symbols.args.length
      if (!edited || edited.inputs.length !== expected || edited.inputs.some(input => input.keywordOnly)) {
        setError(zh ? `请保留类名 ${className} 和 ${expected} 个位置输入。` : `Keep ${className} and ${expected} positional inputs.`)
        return
      }
      onSave?.({ ...params, implementation_code: code })
      return
    }
    const names = customInputNames(params)
    if (!source) {
      if (!edited || edited.inputs.length !== symbols.args.length || edited.inputs.some(input => input.keywordOnly)) {
        setError(zh ? `请保留 ${className} 及 ${symbols.args.length} 个位置输入。` : `Keep ${className} and ${symbols.args.length} positional inputs.`)
        return
      }
      onSave?.({ ...params, class_code: code })
      return
    }
    if (!edited || names.some((name) => !edited.inputs.some((input) => input.name === name)) ||
      edited.inputs.some((input) => !input.optional && !names.includes(input.name))) {
      setError(zh ? `请保留 ${className} 和当前端口签名 forward(self, ${names.join(", ")})。` : `Keep ${className} and the current forward ports: ${names.join(", ")}.`)
      return
    }
    if (source && sourceClass) {
      onSave?.({ ...params, block_source: source.slice(0, sourceClass.start) + code + "\n" + source.slice(sourceClass.end) })
    }
  }

  return (
    <dialog ref={ref} className="block-code-dialog" aria-labelledby="block-code-title" onCancel={(event) => {
      event.preventDefault()
      onClose()
    }}>
      <header>
        <h2 id="block-code-title">{node.data.name} · {onSave ? (zh ? "编辑块" : "Edit Block") : (zh ? "PyTorch 实现" : "PyTorch Implementation")}</h2>
        <button type="button" aria-label={zh ? "关闭" : "Close"} onClick={onClose}>×</button>
      </header>
      {(connectedInputs.length > 0 || outputSymbol) && <div className="block-binding-summary">
        {connectedInputs.map(binding => <code key={binding}>{binding}</code>)}
        {outputSymbol && <code>{zh ? "输出" : "Output"} → {outputSymbol}</code>}
      </div>}
      {onSave ? (
        <>
          <p>{zh ? "在 __init__ 中声明子层，在 forward 中编排计算。Ctrl+空格触发 Python 补全。" : "Declare layers in __init__ and compute in forward. Ctrl+Space opens Python completions."}</p>
          <PythonEditor imports={source ? source.split("\n").filter(line => /^(import|from) /.test(line)).join("\n") + "\n" : "import torch\nfrom torch import nn\nfrom torch.nn import functional as F\n"} sourcePath={String(params.source_path ?? "")} label={zh ? "块 Python 代码" : "Block Python code"} value={value} onChange={setValue} />
        </>
      ) : params.implementation_code || source || node.data.layerType === "CustomBlock" || description?.definition ? (
        <div className="code-sections">
          <section><h3>{zh ? "模块定义" : "Module Definition"}</h3><pre>{params.implementation_code || source || node.data.layerType === "CustomBlock" ? value || (zh ? "找不到所选类的定义。" : "Selected class definition unavailable.") : description?.definition}</pre></section>
        </div>
      ) : (
        <div className="code-sections">
          <section><h3>__init__</h3><pre>{description?.init || (zh ? "# 无需初始化" : "# No initialization required")}</pre></section>
          <section><h3>forward</h3><pre>{description?.forward || "pass"}</pre></section>
        </div>
      )}
      {error && <p role="alert" className="import-error">{error}</p>}
      <footer>
        <button type="button" onClick={onClose}>{zh ? "关闭" : "Close"}</button>
        {onSave && builtin && params.implementation_code && <button type="button" onClick={() => onSave({ ...params, implementation_code: null })}>{zh ? "恢复内置实现" : "Restore Built-in Implementation"}</button>}
        {onSave && <button type="button" className="action-primary" onClick={save}>{zh ? "保存块" : "Save Block"}</button>}
      </footer>
    </dialog>
  )
}
