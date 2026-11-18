import { useState } from "react"
import { parseShapeText, shapeToText } from "../core/graph/customBlock"
import type { ParamValue, TensorShape } from "../core/graph/types"
import type { Locale } from "../i18n"

export function ShapeField({ label, value, locale, onChange }: {
  label: string
  value: ParamValue
  locale: Locale
  onChange: (text: string, shape: TensorShape) => void
}) {
  const stored = typeof value === "string" ? value : shapeToText(value)
  const [draft, setDraft] = useState<string | null>(null)
  const [invalid, setInvalid] = useState(false)
  return (
    <label className="field shape-field">
      <span>{label}</span>
      <input value={draft ?? stored} placeholder="B, 3, 224, 224" aria-invalid={invalid} onChange={(event) => {
        const raw = event.target.value
        setDraft(raw)
        try {
          if (/[,，]\s*$/.test(raw)) return
          const shape = parseShapeText(raw)
          setInvalid(false)
          onChange(raw, shape)
        } catch {
          setInvalid(true)
        }
      }} onBlur={() => { if (!invalid) setDraft(null) }} />
      {invalid && <small role="alert">{locale === "zh" ? "填写正整数、B 或 ?，用逗号分隔。" : "Enter positive integers, B, or ?, separated by commas."}</small>}
    </label>
  )
}
