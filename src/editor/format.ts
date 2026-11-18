import type { TensorShape, TensorSpec } from "../core/graph/types"

export function formatTensorShape(shape: TensorShape): string {
  return `[${shape.map((dim) => String(dim ?? "?")).join(", ")}]`
}

export function formatTensorSpec(spec: TensorSpec | null | undefined, locale: "zh" | "en" = "en"): string {
  if (!spec || spec.shape.length === 0) {
    return locale === "zh" ? "形状未知" : "Unknown shape"
  }

  return `${spec.dtype} ${formatTensorShape(spec.shape)}`
}
