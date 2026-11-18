import type { InputProbeRequest, PythonCompletionRequest, PythonEnvironment, TrainingEvent, TrainingRequest } from "../src/core/trainingContracts"
import { contextBridge, ipcRenderer, webUtils } from 'electron'

// Expose protected APIs to the renderer process
contextBridge.exposeInMainWorld('electronAPI', {
  platform: process.platform,
  isElectron: true,
  inspectInputs: (request: InputProbeRequest) => ipcRenderer.invoke("nnmind:inputs-inspect", request),
  completePython: (request: PythonCompletionRequest) => ipcRenderer.invoke("nnmind:complete", request),
  chooseDirectory: () => ipcRenderer.invoke("nnmind:directory"),
  filePath: (file: File) => webUtils.getPathForFile(file),
  getEnvironment: () => ipcRenderer.invoke("nnmind:python-get"),
  selectEnvironment: () => ipcRenderer.invoke("nnmind:python-select"),
  startTraining: (request: TrainingRequest) => ipcRenderer.invoke("nnmind:training-start", request),
  stopTraining: () => ipcRenderer.invoke("nnmind:training-stop"),
  onEnvironment: (callback: (environment: PythonEnvironment | null, error?: string) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, environment: PythonEnvironment | null, error?: string) => callback(environment, error)
    ipcRenderer.on("nnmind:python-selected", listener)
    return () => { ipcRenderer.removeListener("nnmind:python-selected", listener) }
  },
  onTraining: (callback: (event: TrainingEvent) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, value: TrainingEvent) => callback(value)
    ipcRenderer.on("nnmind:training-event", listener)
    return () => { ipcRenderer.removeListener("nnmind:training-event", listener) }
  },
  setLocale: (locale: 'zh' | 'en') => ipcRenderer.send('nnmind:set-locale', locale),
})
