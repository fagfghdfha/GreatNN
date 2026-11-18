import { useEffect } from "react"
import type { CSSProperties } from "react"
import { Handle, NodeResizer, useUpdateNodeInternals, type NodeProps } from "@xyflow/react"
import { getUiText } from "../i18n"
import { getLayerDef } from "../core/registry/layerRegistry"
import { formatTensorSpec } from "./format"
import { getNodeSize, getPortPositions } from "./layout"
import type { CanvasNode } from "./types"

function handleStyle(index: number, total: number, mode: CanvasNode["data"]["layoutMode"]): CSSProperties {
  const offset = `${((index + 1) / (total + 1)) * 100}%`
  return mode === "vertical" ? { left: offset } : { top: offset }
}

export function LayerNode({ id, data, selected }: NodeProps<CanvasNode>) {
  const layer = getLayerDef(data.layerType, data.params)
  const updateNodeInternals = useUpdateNodeInternals()
  useEffect(() => { updateNodeInternals(id) }, [id, layer.inputs.length, data.layoutMode, updateNodeInternals])
  const ports = getPortPositions(data.layoutMode)
  const locale = data.locale ?? "en"
  const text = getUiText(locale)
  const minimum = getNodeSize(data.layerType)
  const output = formatTensorSpec(data.specs?.outputs[0], locale)
  const primaryLabel = data.params.class_name ? String(data.params.class_name) : data.label

  return (
    <article className={`layer-node layer-node--${layer.category} ${selected ? "is-selected" : ""} ${data.issueLevel ? `has-${data.issueLevel}` : ""}`}
      style={{ "--accent": data.accent } as CSSProperties}
      title={`${data.name} · ${data.description}\n${text.outputPort}: ${output}`}>
      <NodeResizer isVisible={selected} minWidth={Math.min(minimum.width, 102)} minHeight={minimum.height}
        maxWidth={720} maxHeight={480} color="#285ead" />
      {layer.inputs.map((port, index) => (
        <Handle key={port.name} type="target" position={ports.target} id={port.name}
          style={handleStyle(index, layer.inputs.length, data.layoutMode)} />
      ))}
      {layer.outputs.map((port, index) => (
        <Handle key={port.name} type="source" position={ports.source} id={port.name}
          style={handleStyle(index, layer.outputs.length, data.layoutMode)} />
      ))}
      <div className="layer-node__body">
        <div className="layer-node__heading">
          <span className="layer-node__glyph" aria-hidden="true">{layer.category === "transformer" ? "✦" : layer.category === "embedding" ? "◈" : layer.category === "input" ? "→" : layer.category === "output" ? "↗" : "▪"}</span>
          <strong className="layer-node__kind">{primaryLabel}</strong>
          {data.issueLevel && <span className="layer-node__issue" title={locale === "zh" ? "请检查问题面板" : "See Problems"} aria-label={data.issueLevel} />}
        </div>
        {data.name !== primaryLabel && <span className="layer-node__name">{data.name}</span>}
        <span className="layer-node__tensor">{data.layerType === "Tokenizer" ? "input_ids · attention_mask" : output}</span>
      </div>
    </article>
  )
}
