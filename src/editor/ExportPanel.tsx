import { useState } from "react"
import type { PythonModule } from "../core/import/pythonModuleLibrary"
import { type Locale, getUiText } from "../i18n"

type ExportPanelProps = {
  locale: Locale
  graphJson: string
  draftJson: string
  onDraftJsonChange: (value: string) => void
  onImportJson: () => void
  onImportModelFile: (file: File) => void
  onCopyJson: () => void
  importError: string
  pythonLibrary: { fileName: string; modules: PythonModule[] } | null
  onImportPythonClass: (name: string, includeOptional: boolean) => void
}

export function ExportPanel({
  locale,
  graphJson,
  draftJson,
  onDraftJsonChange,
  onImportJson,
  onImportModelFile,
  onCopyJson,
  importError,
  pythonLibrary,
  onImportPythonClass,
}: ExportPanelProps) {
  const text = getUiText(locale)
  const zh = locale === "zh"

  return (
    <section className="panel export-panel">
      <div className="panel__header">
        <h2>{text.exportTitle}</h2>
        <p>{text.exportDescription}</p>
      </div>

      <div className="export-actions">
        <button type="button" onClick={onCopyJson}>
          {text.copyJson}
        </button>
        <button type="button" onClick={onImportJson}>
          {text.loadDraftJson}
        </button>
        <label className="file-action">
          <input
            type="file"
            accept=".json,.py,.txt,.onnx,.pb"
            onChange={(event) => {
              const file = event.target.files?.[0]
              if (file) {
                onImportModelFile(file)
                event.currentTarget.value = ""
              }
            }}
          />
          <span>{text.importModelFile}</span>
        </label>
      </div>

      {importError && <p className="import-error" role="alert">{importError}</p>}
      {pythonLibrary && (
        <PythonClassPicker key={pythonLibrary.fileName} locale={locale} library={pythonLibrary} onImport={onImportPythonClass} />
      )}

      <details className="graph-json-details">
      <summary>{zh ? "图结构 JSON 与导入草稿" : "Graph JSON and Import Draft"}</summary>
      <div className="code-stack">
        <label className="code-block">
          <span>{text.graphJson}</span>
          <textarea readOnly value={graphJson} />
        </label>
        <label className="code-block">
          <span>{text.importDraft}</span>
          <textarea value={draftJson} onChange={(event) => onDraftJsonChange(event.target.value)} />
        </label>
      </div>
      </details>
    </section>
  )
}

function PythonClassPicker({ locale, library, onImport }: {
  locale: Locale
  library: { fileName: string; modules: PythonModule[] }
  onImport: (name: string, includeOptional: boolean) => void
}) {
  const zh = locale === "zh"
  const [selection, setSelection] = useState("")
  const [includeOptional, setIncludeOptional] = useState(false)
  const selected = library.modules.find((module) => module.name === selection) ?? library.modules[0]
  return (
    <div className="python-class-picker">
      <label>
        <span>{zh ? "选择 PyTorch 模块类" : "Choose a PyTorch Module Class"} · {library.fileName}</span>
        <select value={selected.name} onChange={(event) => setSelection(event.target.value)}>
          {library.modules.map((module) => <option key={module.name}>{module.name}</option>)}
        </select>
      </label>
      {selected.inputs.some((input) => input.optional) && (
        <label className="checkbox-line">
          <input type="checkbox" checked={includeOptional} onChange={(event) => setIncludeOptional(event.target.checked)} />
          {zh ? "为可选参数添加输入端口" : "Add ports for optional arguments"}
        </label>
      )}
      <button type="button" className="action-primary" onClick={() => onImport(selected.name, includeOptional)}>
        {zh ? "导入所选模块" : "Import Selected Module"}
      </button>
      <small>{zh ? "按类导入为可编辑的自定义块，保留原始 Python 实现。" : "Imports the class as an editable block with its original Python implementation."}</small>
    </div>
  )
}
