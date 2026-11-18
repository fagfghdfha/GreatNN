import { trainingDesktop } from "./trainingDesktop"
import type { TrainingParameter } from "../core/trainingContracts"
import { customInputNames } from "../core/graph/customBlock"
import { layerExplanations } from "./layerExplanations"
import { ShapeField } from "./ShapeField"
import { getLayerDef } from "../core/registry/layerRegistry"
import { formatParamCount } from "../core/graph/paramCount"
import type { ParamValue } from "../core/graph/types"
import {
  type Locale,
  formatExactParamCount,
  getLayerDescription,
  getLayerLabel,
  getParamLabel,
  getUiText,
} from "../i18n"
import { formatTensorSpec } from "./format"
import type { CanvasNode } from "./types"

type InspectorProps = {
  locale: Locale
  node?: CanvasNode
  onViewCode: () => void
  onEditCode: () => void
  onSaveBlock: () => void
  onEditInput: () => void
  onChangePorts: (ports: number) => void
  runtimeParameters?: TrainingParameter[]
  libraryMessage?: string
  portBindings: { port: string; local: string; source?: string }[]
  outputSymbol?: string
  onUpdateName: (name: string) => void
  onUpdateParam: (paramName: string, value: ParamValue) => void
}

function TupleInput({
  value,
  onChange,
}: {
  value: ParamValue
  onChange: (next: ParamValue) => void
}) {
  const tuple = Array.isArray(value) ? value : []

  return (
    <div className="tuple-input">
      {tuple.map((item, index) => (
        <input
          key={index}
          value={String(item)}
          onChange={(event) => {
            const next = [...tuple]
            const raw = event.target.value.trim()
            next[index] = raw === "B" ? "B" : raw === "" ? null : Number.isNaN(Number(raw)) ? raw : Number(raw)
            onChange(next)
          }}
        />
      ))}
    </div>
  )
}

export function Inspector({ locale, node, onViewCode, onEditCode, onSaveBlock, onEditInput, onChangePorts, runtimeParameters, libraryMessage, portBindings, outputSymbol, onUpdateName, onUpdateParam }: InspectorProps) {
  const text = getUiText(locale)

  if (!node) {
    return (
      <section className="panel inspector-panel">
        <div className="panel__header">
          <p>{text.inspectorEmpty}</p>
        </div>
      </section>
    )
  }

  const zh = locale === "zh"
  const layerDef = getLayerDef(node.data.layerType, node.data.params)
  const custom = layerDef.type === "CustomBlock"
  const imported = custom && Boolean(node.data.params.block_source)
  const inputShapes = Array.isArray(node.data.params.input_shapes) ? node.data.params.input_shapes : []
  const label = getLayerLabel(node.data.layerType, node.data.label, locale)
  const description = getLayerDescription(node.data.layerType, node.data.description, locale)
  const paramCount = node.data.paramCount ?? 0

  return (
    <section className="panel inspector-panel">
      <div className="panel__header">
        <h2>{label}</h2>
        <p>{description}</p>
        <p className="inspector-explanation">{layerExplanations[layerDef.type][zh ? 0 : 1]}</p>
        {imported && Boolean(node.data.params.documentation) && <p className="source-docstring">{String(node.data.params.documentation)}</p>}
        <div className="inspector-actions">
          {layerDef.type === "Input" ? (
            <button type="button" className="action-primary" onClick={onEditInput}>{zh ? "输入设置 · 编写／导入脚本" : "Input Settings · Edit / Import Script"}</button>
          ) : <>
            <button type="button" onClick={onViewCode}>{zh ? "查看实现代码" : "View Implementation"}</button>
            <button type="button" onClick={onEditCode}>{custom ? (zh ? "编辑块" : "Edit Block") : (zh ? "编辑实现" : "Edit Implementation")}</button>
            {custom && <button type="button" onClick={onSaveBlock}>{zh ? "固化到自建库" : "Save to My Blocks"}</button>}
          </>}
        </div>
        {libraryMessage && <small role="status">{libraryMessage}</small>}
        {layerDef.type === "Input" && node.data.params.script_source && <p className="script-spec-note">
          {node.data.params.script_source === "validation" ? (zh ? "已从验证样本同步" : "Synced from validation sample") : (zh ? "已从输入脚本同步" : "Synced from input script")}
          {Array.isArray(node.data.params.script_sample_shape)
            ? ` · ${node.data.params.script_sample_dtype} [${node.data.params.script_sample_shape.join(", ")}]`
            : ` · ${node.data.params.script_kind}`}
        </p>}
        {layerDef.type === "Input" && node.data.params.dataset_code && !node.data.params.script_source && <small>{zh ? "已设置输入脚本，尚未检测输入" : "Input script configured; not yet detected"}</small>}
        <p className="panel__meta">
          {custom ? (zh ? "参数量需由实际 PyTorch 模型计算" : "Parameter count requires the PyTorch model") :
            `${formatParamCount(paramCount)} ${text.params} · ${text.exactParams} ${formatExactParamCount(locale, paramCount)}`}
        </p>
      </div>

      <label className="field">
        <span>{text.nodeName}</span>
        <input value={node.data.name} onChange={(event) => onUpdateName(event.target.value)} />
      </label>

      <section className="port-settings" aria-label={zh ? "端口" : "Ports"}>
        <h3>{zh ? "端口" : "Ports"}</h3>
        {portBindings.map((binding, index) => <label className="port-row" key={binding.port}>
          <span className="port-row__heading"><strong>{zh ? "输入" : "In"} · {binding.port}</strong>
            <small title={binding.source}>{binding.source ? `← ${binding.source}` : (zh ? "未连接" : "Disconnected")}</small></span>
          <input aria-label={zh ? `输入变量 ${binding.port}` : `Input symbol ${binding.port}`}
            key={String(Array.isArray(node.data.params.input_aliases) ? node.data.params.input_aliases[index] ?? "" : "")}
            defaultValue={binding.local} onBlur={event => {
              const value = event.target.value.trim()
              if (value === binding.local) return
              const aliases = Array.isArray(node.data.params.input_aliases) ? [...node.data.params.input_aliases] : []
              while (aliases.length <= index) aliases.push(null)
              aliases[index] = value || null
              onUpdateParam("input_aliases", aliases)
            }} />
          {binding.source && <code>{formatTensorSpec(node.data.specs?.inputs[index], locale)}</code>}
        </label>)}
        <label className="port-row">
          <span className="port-row__heading"><strong>{zh ? "输出变量" : "Output Symbol"}</strong>
            <small title={outputSymbol}>{outputSymbol}</small></span>
          <input key={String(node.data.params.output_alias ?? "")} defaultValue={String(node.data.params.output_alias ?? outputSymbol ?? "")}
            onBlur={event => {
              const value = event.target.value.trim()
              if (value !== (node.data.params.output_alias ?? outputSymbol ?? "")) onUpdateParam("output_alias", value || null)
            }} />
          {layerDef.outputs.length > 0 && <code>{formatTensorSpec(node.data.specs?.outputs[0], locale)}</code>}
        </label>
      </section>

      <div className="node-training-settings">
        {layerDef.type !== "Input" && layerDef.type !== "Output" && <label className="field">
          <span>{zh ? "训练模式" : "Training Mode"}</span>
          <select value={String(node.data.params.training_mode ?? "online")} onChange={event => onUpdateParam("training_mode", event.target.value)}>
            <option value="online">{zh ? "在线（可训练）" : "Trainable"}</option>
            <option value="frozen">{zh ? "冻结" : "Frozen"}</option>
          </select>
        </label>}
        {["CausalLM", "Tokenizer", "VisionEncoder", "AudioEncoder"].includes(layerDef.type) && <button type="button" onClick={async () => {
          const path = await trainingDesktop?.chooseDirectory?.()
          if (path) onUpdateParam("model_path", path)
        }}>{zh ? "选择本地模型目录" : "Choose Local Model Folder"}</button>}
      </div>

      {custom && (
        <div className="custom-shapes">
          <small>{zh ? "B 表示批量维，? 表示未知维，留空表示形状未声明。" : "B is batch, ? is an unknown dimension. Leave blank for an unspecified shape."}</small>
          {!imported && <label className="field"><span>{zh ? "输入端口数" : "Input Ports"}</span>
            <select value={customInputNames(node.data.params).length} disabled={Boolean(node.data.params.class_code)}
              title={node.data.params.class_code ? (zh ? "已保存块使用代码中的端口签名" : "Saved definition controls the ports") : undefined}
              onChange={event => onChangePorts(Number(event.target.value))}>
              {Array.from({ length: 8 }, (_, index) => <option key={index} value={index + 1}>{index + 1}</option>)}
            </select></label>}
          {customInputNames(node.data.params).map((name, index) => (
            <ShapeField key={name} locale={locale} label={`${text.inputPort} · ${name}`} value={inputShapes[index] ?? ""}
              onChange={(raw) => onUpdateParam("input_shapes", customInputNames(node.data.params).map((_, axis) => axis === index ? raw : inputShapes[axis] ?? ""))} />
          ))}
          <ShapeField locale={locale} label={`${text.outputPort} · out`} value={node.data.params.output_shape ?? ""}
            onChange={(raw) => onUpdateParam("output_shape", raw)} />
          {imported && (
            <label className="field constructor-field">
              <span>{zh ? "构造参数" : "Constructor Arguments"}</span>
              <input value={String(node.data.params.constructor_args ?? "")} onChange={(event) => onUpdateParam("constructor_args", event.target.value)} />
              {Boolean(node.data.params.constructor_signature) && <small>{String(node.data.params.constructor_signature)}</small>}
              {String(node.data.params.constructor_args).includes("=None") && <small>{zh ? "请按原项目配置必填构造参数。" : "Set the required constructor arguments for your project."}</small>}
            </label>
          )}
        </div>
      )}

      {!custom && layerDef.params.map((param) => {
        const value = node.data.params[param.name]
        const localLabels: Record<string, string> = {
          lora_targets: "LoRA 目标模块（可选）", tokenizer_bundle: "分词器字典输入", max_length: "最大 token 数",
          model_path: "本地模型目录", precision: "权重精度", checkpointing: "梯度检查点",
        }
        const paramLabel = zh && localLabels[param.name] ? localLabels[param.name] : getParamLabel(param.name, param.label, locale)

        if (param.name === "lora_targets") {
          const options = [
            ["", zh ? "自动匹配" : "Auto-detect"],
            ["q_proj, v_proj", "Q / V"],
            ["q_proj, k_proj, v_proj, o_proj", "Q / K / V / O"],
            ["gate_proj, up_proj, down_proj", "MLP · Gate / Up / Down"],
            ["all-linear", zh ? "全部线性层" : "All Linear Layers"],
          ]
          const existing = String(value ?? "").split(",").map(item => item.trim()).filter(Boolean).join(", ")
          if (!options.some(([option]) => option === existing)) options.push([existing, (zh ? "已保存： " : "Saved: ") + existing])
          return <label key={param.name} className="field">
            <span>{paramLabel}</span>
            <select value={existing} onChange={event => onUpdateParam(param.name, event.target.value)}>
              {options.map(([option, label]) => <option key={option} value={option}>{label}</option>)}
            </select>
          </label>
        }

        if (param.name === "lora_rank" || param.name === "lora_alpha") {
          const current = Number(value ?? param.default)
          const presets = param.name === "lora_rank" ? [2, 4, 8, 16, 32, 64, 128] : [4, 8, 16, 32, 64, 128, 256]
          const options = [...new Set([...presets, current])].sort((left, right) => left - right)
          return <label key={param.name} className="field">
            <span>{zh ? (param.name === "lora_rank" ? "LoRA 秩（r）" : "LoRA 缩放（α）") : paramLabel}</span>
            <select value={current} onChange={event => onUpdateParam(param.name, Number(event.target.value))}>
              {options.map(option => <option key={option} value={option}>{option}</option>)}
            </select>
          </label>
        }
        if (param.type === "boolean") {
          return (
            <label key={param.name} className="field field--checkbox">
              <span>{paramLabel}</span>
              <input
                type="checkbox"
                checked={Boolean(value)}
                onChange={(event) => onUpdateParam(param.name, event.target.checked)}
              />
            </label>
          )
        }

        if (param.type === "select") {
          return (
            <label key={param.name} className="field">
              <span>{paramLabel}</span>
              <select value={String(value)} onChange={(event) => onUpdateParam(param.name, event.target.value)}>
                {param.options?.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
          )
        }

        if (layerDef.type === "Input" && param.name === "shape") {
          return <ShapeField key={param.name} locale={locale} label={paramLabel} value={value}
            onChange={(_, shape) => onUpdateParam(param.name, shape)} />
        }

        if (param.type === "tuple") {
          return (
            <label key={param.name} className="field">
              <span>{paramLabel}</span>
              <TupleInput value={value} onChange={(next) => onUpdateParam(param.name, next)} />
            </label>
          )
        }

        return (
          <label key={param.name} className="field">
            <span>{paramLabel}</span>
            <input
              value={String(value ?? param.default)}
              onChange={(event) => {
                const raw = event.target.value
                onUpdateParam(param.name, param.type === "number" ? Number(raw) : raw)
              }}
            />
          </label>
        )
      })}

      <details className="runtime-parameters">
        <summary>{zh ? "实例参数" : "Instance Parameters"}</summary>
        {runtimeParameters === undefined ? <p>{zh ? "预览验证集或训练后查看该实例的实际权重。" : "Preview validation data or train to inspect live weights."}</p> :
          !runtimeParameters.length ? <p>{zh ? "此实例没有权重参数。" : "This instance has no weight parameters."}</p> :
            runtimeParameters.map(parameter => <div key={parameter.name}>
              <strong>{parameter.name}</strong>
              <small>[{parameter.shape.join(", ")}] · {parameter.count.toLocaleString()} · {parameter.trainable ? (zh ? "可训练" : "Trainable") : (zh ? "冻结" : "Frozen")}</small>
              <code title={zh ? "权重样本（最多 8 个值）" : "Weight sample (up to 8 values)"}>{parameter.sample.map(value => value == null ? "—" : value.toPrecision(3)).join(", ")}</code>
            </div>)}
      </details>
    </section>
  )
}
