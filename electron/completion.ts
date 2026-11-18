import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process"
import { createInterface } from "node:readline"
import type { PythonCompletion, PythonCompletionRequest } from "../src/core/trainingContracts"
import completionRunner from "./complete_python.py?raw"

type Worker = { child: ChildProcessWithoutNullStreams; path: string; pending: Map<number, (items: PythonCompletion[]) => void> }
let worker: Worker | undefined
let id = 0

export function stopCompletion() {
  const current = worker
  worker = undefined
  current?.child.kill()
  if (current) for (const resolve of current.pending.values()) resolve([])
}

export function completePython(executable: string, library: string, request: PythonCompletionRequest): Promise<PythonCompletion[]> {
  if (typeof request.code !== "string" || request.code.length > 280000 || typeof request.path !== "string" ||
    !Number.isInteger(request.line) || request.line < 1 || !Number.isInteger(request.column) || request.column < 0) {
    return Promise.resolve([])
  }
  if (!worker || worker.path !== executable) {
    stopCompletion()
    const child = spawn(executable, ["-u", "-c", completionRunner, library], { windowsHide: true, env: { ...process.env, PYTHONUTF8: "1" } })
    const instance: Worker = { child, path: executable, pending: new Map() }
    worker = instance
    child.stderr.resume()
    child.stdin.on("error", () => undefined)
    child.on("error", () => { if (worker === instance) stopCompletion() })
    child.on("exit", () => { if (worker === instance) stopCompletion() })
    createInterface({ input: child.stdout }).on("line", line => {
      try {
        const { id, matches } = JSON.parse(line) as { id: number; matches: PythonCompletion[] }
        instance.pending.get(id)?.(matches)
        instance.pending.delete(id)
      } catch { /* Ignore malformed replies. */ }
    })
  }
  const instance = worker
  const requestId = ++id
  if (instance.pending.size >= 4) return Promise.resolve([])
  return new Promise(resolve => {
    const timeout = setTimeout(() => { instance.pending.delete(requestId); resolve([]) }, 2600)
    instance.pending.set(requestId, items => { clearTimeout(timeout); resolve(items) })
    instance.child.stdin.write(JSON.stringify({ ...request, id: requestId }) + "\n")
  })
}
