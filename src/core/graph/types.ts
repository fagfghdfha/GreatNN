export type TensorDim = number | "B" | null

export type TensorShape = TensorDim[]

export const tensorDTypes = ["float32", "float16", "bfloat16", "float64", "int64", "int32", "int16", "int8", "uint8", "bool"] as const
export type TensorDType = typeof tensorDTypes[number]

export type TensorSpec = {
  dtype: TensorDType
  shape: TensorShape
}

export type GraphLayoutMode = "horizontal" | "vertical"

export type LayerType =
  | "Input"
  | "Embedding"
  | "Conv2d"
  | "ResidualBlock2d"
  | "ResNetBasicBlock"
  | "ResNetBottleneck"
  | "BatchNorm2d"
  | "BatchNorm1d"
  | "LayerNorm"
  | "ReLU"
  | "GELU"
  | "MaxPool2d"
  | "AdaptiveAvgPool2d"
  | "PatchEmbedding"
  | "TokenPool"
  | "Flatten"
  | "Linear"
  | "Dropout"
  | "LSTM"
  | "GRU"
  | "SelfAttention"
  | "TransformerEncoder"
  | "TransformerDecoder"
  | "Add"
  | "Concat"
  | "Tokenizer"
  | "VisionEncoder"
  | "AudioEncoder"
  | "CausalLM"
  | "CustomBlock"
  | "Output"

export type LayerCategory =
  | "input"
  | "embedding"
  | "conv"
  | "residual"
  | "norm"
  | "activation"
  | "pool"
  | "transformer"
  | "sequence"
  | "attention"
  | "reshape"
  | "linear"
  | "merge"
  | "custom"
  | "output"

export type ParamValue =
  | string
  | number
  | boolean
  | null
  | Array<string | number | null>

export type LayerNode = {
  id: string
  name: string
  layerType: LayerType
  position: {
    x: number
    y: number
    z?: number
  }
  size?: { width: number; height: number }
  params: Record<string, ParamValue>
  inputSpecs?: TensorSpec[]
  outputSpecs?: TensorSpec[]
}

export type GraphEdge = {
  id: string
  fromNodeId: string
  fromPort: string
  toNodeId: string
  toPort: string
}

export type NeuralGraph = {
  version: number
  framework: "pytorch"
  nodes: LayerNode[]
  edges: GraphEdge[]
}

export type GraphIssue = {
  level: "error" | "warning" | "info"
  nodeId?: string
  edgeId?: string
  message: string
}

export type InferenceResult = {
  orderedNodeIds: string[]
  specsByNodeId: Record<
    string,
    {
      inputs: Array<TensorSpec | null>
      outputs: TensorSpec[]
    }
  >
  issues: GraphIssue[]
}
