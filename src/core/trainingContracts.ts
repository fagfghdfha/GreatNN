export type InputProbeRequest = {
  preferDataset?: boolean
  script: string
  path: string
  ports: Array<{ name: string; transform?: string }>
}
export type ObservedInput = { name: string; kind: string; dtype?: string; shape?: number[]; forwarded?: { dtype: string; shape: number[] }; forwardedKind?: string }
export type InputProbeResult = { source: "direct" | "validation"; inputs: ObservedInput[] }

export type PythonCompletion = { label: string; type: string; detail: string }
export type PythonCompletionRequest = { code: string; path: string; line: number; column: number }

export type GraphDataset = { name: string; code: string; path: string }

export type PythonEnvironment = { path: string; version: string; torch: boolean }
export type TrainingParameter = { name: string; shape: number[]; count: number; trainable: boolean; sample: Array<number | null> }
export type TrainingParameters = Record<string, TrainingParameter[]>
export type LossPoint = { epoch: number; train: number; validation: number }
export type TrainingEvent = ({ scope?: "execute" } & (
  | { type: "result"; text: string; details?: string }
  | { type: "ready"; device: string; output: string }
  | { type: "parameters"; layers: TrainingParameters }
  | ({ type: "epoch" } & LossPoint)
  | { type: "test"; loss: number }
  | { type: "log"; message: string }
  | { type: "finished"; status: "completed" | "failed" | "stopped"; message?: string }))

export type TrainingRequest = {
  model: string
  dataset: string
  datasetPath: string
  loss: "mse" | "crossEntropy" | "l1" | "bce" | "custom" | "causal"
  lossCode: string
  epochs: number
  batchSize: number
  learningRate: number
  optimizer: "adam" | "sgd"
  device: "auto" | "cpu" | "cuda"
  savePath?: string
  mode: "train" | "preview" | "execute" | "generate"
  inputJson?: string
  canvasTokens?: number
  generation?: { path: string; adapter: string; prompt: string; tokens: number; precision: string }
}

export const datasetTemplate = `import torch
from torch.utils.data import TensorDataset

def build_datasets():
    # Example: replace these tensors with data loaded from your project.
    # Each item contains (network_input, target).
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(120, 4, generator=generator)
    y = x.sum(dim=1, keepdim=True)
    return {
        "train": TensorDataset(x[:80], y[:80]),
        "validation": TensorDataset(x[80:100], y[80:100]),
        "test": TensorDataset(x[100:], y[100:]),
    }
`

export const lossTemplate = `import torch

def compute_loss(prediction, target, tensors, inputs):
    # tensors contains named graph values; inputs contains the input batch.
    return torch.mean((prediction - target) ** 2)
`

export const causalDatasetTemplate = `from pathlib import Path
from transformers import AutoTokenizer

MODEL_FOLDER = r"填入所选的本地模型目录"
DATA_FOLDER = Path(r"填入含 train.txt、validation.txt、test.txt 的目录")
MAX_LENGTH = 128

def build_datasets():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def read_split(name):
        texts = (DATA_FOLDER / (name + ".txt")).read_text(encoding="utf-8").splitlines()
        rows = []
        for text in texts:
            if not text.strip():
                continue
            inputs = tokenizer(text, truncation=True, padding="max_length",
                               max_length=MAX_LENGTH, return_tensors="pt")
            ids = inputs["input_ids"][0]
            mask = inputs["attention_mask"][0]
            if mask.sum().item() >= 2:
                rows.append(({"input_ids": ids, "attention_mask": mask},
                             ids.masked_fill(mask == 0, -100)))
        return rows

    return {name: read_split(name) for name in ("train", "validation", "test")}
`
