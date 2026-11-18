import type { InputProbeRequest, InputProbeResult, PythonCompletion, PythonCompletionRequest, PythonEnvironment, TrainingEvent, TrainingRequest } from "../core/trainingContracts"

type TrainingDesktop = {
  inspectInputs: (request: InputProbeRequest) => Promise<InputProbeResult>
  completePython: (request: PythonCompletionRequest) => Promise<PythonCompletion[]>
  chooseDirectory: () => Promise<string>
  getEnvironment: () => Promise<PythonEnvironment | null>
  selectEnvironment: () => Promise<PythonEnvironment | null>
  startTraining: (request: TrainingRequest) => Promise<void>
  stopTraining: () => Promise<void>
  onEnvironment: (callback: (environment: PythonEnvironment | null, error?: string) => void) => () => void
  onTraining: (callback: (event: TrainingEvent) => void) => () => void
  filePath: (file: File) => string
}

export const trainingDesktop = (window as Window & { electronAPI?: TrainingDesktop }).electronAPI
