import { app, dialog, ipcMain, type BrowserWindow } from "electron"
import { execFile, spawn, type ChildProcess } from "node:child_process"
import { mkdtemp, readFile, realpath, rm, stat, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { basename, isAbsolute, join } from "node:path"
import { promisify } from "node:util"
import { createInterface } from "node:readline"
import type { InputProbeRequest, InputProbeResult, PythonEnvironment, TrainingEvent, TrainingRequest } from "../src/core/trainingContracts"
import { completePython } from "./completion"
import type { PythonCompletionRequest } from "../src/core/trainingContracts"
import runner from "./train.py?raw"

const exec = promisify(execFile)
const prefix = "NNMIND_EVENT:"
let active: { child: ChildProcess; stop: boolean } | null = null
let starting = false
let cancelStarting = false

const pythonTools = () => join(app.getPath("userData"), "python-support")
let selectedExecutable = ""
const preference = () => join(app.getPath("userData"), "training-python.json")

async function probe(path: string): Promise<PythonEnvironment> {
  const resolved = await realpath(path)
  if (!isAbsolute(resolved) || !(await stat(resolved)).isFile() ||
    !/^python(?:\d+(?:\.\d+)*)?(?:\.exe)?$/i.test(basename(resolved))) {
    throw new Error("请选择 Python 解释器，例如 python.exe。")
  }
  const { stdout } = await exec(path, [
    "-I", "-c", "import importlib.util, json, sys; print(json.dumps({'version':sys.version.split()[0], 'torch':importlib.util.find_spec('torch') is not None}))",
  ], { timeout: 10000, windowsHide: true, maxBuffer: 16384 })
  return { path, ...JSON.parse(stdout.trim()) } as PythonEnvironment
}

function pythonProcessEnvironment(environment: PythonEnvironment) {
  return {
    ...process.env, PYTHONUTF8: "1", PYTHONUNBUFFERED: "1",
    ...(environment.version.startsWith("3.13.") ? {
      PYTHONPATH: [pythonTools(), process.env.PYTHONPATH].filter(Boolean).join(process.platform === "win32" ? ";" : ":"),
    } : {}),
  }
}

let inspecting = false

async function inspectInputScript(request: InputProbeRequest): Promise<InputProbeResult> {
  if (!request || typeof request.script !== "string" || !request.script.trim() || request.script.length > 6_000_000 ||
    typeof request.path !== "string" || (request.path && !isAbsolute(request.path)) ||
    !Array.isArray(request.ports) || !request.ports.length || request.ports.length > 256 ||
    request.ports.some(port => typeof port?.name !== "string" || !/^[a-zA-Z_]\w*$/.test(port.name) ||
      (port.transform !== undefined && (typeof port.transform !== "string" || port.transform.length > 500_000)))) {
    throw new Error("输入脚本或端口参数无效。")
  }
  if (active || starting || inspecting) throw new Error("已有 Python 任务正在运行。")
  inspecting = true
  let work: string | undefined
  try {
    const environment = await readEnvironment()
    if (!environment?.torch) throw new Error("请在右下角选择装有 PyTorch 的环境。")
    work = await mkdtemp(join(tmpdir(), "nnmind-input-"))
    await Promise.all([
      writeFile(join(work, "runner.py"), runner, "utf8"),
      writeFile(join(work, "data.py"), request.script, "utf8"),
      writeFile(join(work, "settings.json"), JSON.stringify({ mode: "inspect-inputs", datasetPath: request.path, ports: request.ports, preferDataset: request.preferDataset === true }), "utf8"),
    ])
    try {
      const result = await exec(environment.path, ["-u", join(work, "runner.py")], {
        cwd: work, windowsHide: true, timeout: 30000, maxBuffer: 2_000_000, env: pythonProcessEnvironment(environment),
      })
      for (const line of result.stdout.split(/\r?\n/).reverse()) {
        if (line.startsWith(prefix)) {
          const value = JSON.parse(line.slice(prefix.length))
          if (value.type === "input-specs" && Array.isArray(value.inputs)) return value
        }
      }
      throw new Error("Python 未返回输入规格。")
    } catch (cause) {
      const failure = cause as { killed?: boolean; stderr?: string; message?: string }
      if (failure.killed) throw new Error("输入检测超过 30 秒。可保存脚本，稍后重试检测。", { cause })
      throw new Error(failure.stderr?.trim().split(/\r?\n/).at(-1) || failure.message || "输入检测失败。", { cause })
    }
  } finally {
    inspecting = false
    if (work) await rm(work, { recursive: true, force: true }).catch(() => undefined)
  }
}

async function readEnvironment(): Promise<PythonEnvironment | null> {
  try {
    const saved = JSON.parse(await readFile(preference(), "utf8"))
    return typeof saved.path === "string" ? await probe(saved.path) : null
  } catch {
    return null
  }
}

export async function selectTrainingEnvironment(window: BrowserWindow, locale: "zh" | "en") {
  const zh = locale === "zh"
  const selected = await dialog.showOpenDialog(window, {
    title: zh ? "选择本地 Python 解释器" : "Choose a local Python interpreter",
    properties: ["openFile"],
    filters: process.platform === "win32" ? [{ name: "Python executable", extensions: ["exe"] }] : undefined,
  })
  if (selected.canceled || !selected.filePaths[0]) return null
  const environment = await probe(selected.filePaths[0])
  selectedExecutable = environment.path
  await writeFile(preference(), JSON.stringify(environment), "utf8")
  window.webContents.send("nnmind:python-selected", environment)
  return environment
}

function validateRequest(request: TrainingRequest) {
  const generation = request?.generation
  if (request?.mode === "generate") {
    if (!generation || typeof generation.path !== "string" || !isAbsolute(generation.path) ||
      typeof generation.adapter !== "string" || (generation.adapter && !isAbsolute(generation.adapter)) ||
      typeof generation.prompt !== "string" || !generation.prompt.trim() || generation.prompt.length > 32768 ||
      !Number.isInteger(generation.tokens) || generation.tokens < 1 || generation.tokens > 2048 ||
      !["float32", "float16", "bfloat16"].includes(generation.precision) ||
      !["auto", "cpu", "cuda"].includes(request.device)) throw new Error("请检查本地模型目录和生成参数。")
    return
  }
  if (!request || typeof request.model !== "string" || !request.model.includes("class GeneratedNet") ||
    typeof request.dataset !== "string" || (!request.dataset.trim() && !(request.mode === "execute" && request.inputJson)) || typeof request.datasetPath !== "string" ||
    typeof request.lossCode !== "string" || (request.savePath && (typeof request.savePath !== "string" || !isAbsolute(request.savePath))) ||
    [request.model, request.dataset, request.lossCode].some(source => source.length > 6_000_000) ||
    !["mse", "crossEntropy", "l1", "bce", "custom", "causal"].includes(request.loss) ||
    !["adam", "sgd"].includes(request.optimizer) || !["auto", "cpu", "cuda"].includes(request.device) ||
    !["train", "preview", "execute"].includes(request.mode) || (request.mode === "execute" && ((request.inputJson !== undefined && (typeof request.inputJson !== "string" || request.inputJson.length > 500000)) || (request.canvasTokens !== undefined && (!Number.isInteger(request.canvasTokens) || request.canvasTokens < 1 || request.canvasTokens > 128)))) ||
    !Number.isInteger(request.epochs) || request.epochs < 1 || request.epochs > 10000 ||
    !Number.isInteger(request.batchSize) || request.batchSize < 1 || request.batchSize > 10000 ||
    !Number.isFinite(request.learningRate) || request.learningRate <= 0 || request.learningRate > 10) {
    throw new Error("训练设置无效，请检查数据脚本、模型及参数。")
  }
}

export async function stopTraining() {
  if (starting) cancelStarting = true
  const job = active
  if (!job?.child.pid || job.stop) return
  job.stop = true
  if (process.platform === "win32") {
    try {
      await exec("taskkill.exe", ["/PID", String(job.child.pid), "/T", "/F"], { windowsHide: true, timeout: 8000 })
    } catch {
      job.child.kill()
    }
  } else {
    try { process.kill(-job.child.pid, "SIGKILL") } catch { job.child.kill() }
  }
}

export function registerTraining(getWindow: () => BrowserWindow | null, getLocale: () => "zh" | "en") {
  ipcMain.handle("nnmind:python-get", async (event) => {
    if (event.sender !== getWindow()?.webContents) return null
    const environment = await readEnvironment()
    selectedExecutable = environment?.path ?? ""
    return environment
  })
  ipcMain.handle("nnmind:complete", async (event, request: PythonCompletionRequest) => {
    if (!request || event.sender !== getWindow()?.webContents) return []
    if (!selectedExecutable) selectedExecutable = (await readEnvironment())?.path ?? ""
    return selectedExecutable ? completePython(selectedExecutable, pythonTools(), request) : []
  })
  ipcMain.handle("nnmind:directory", async (event) => {
    const window = getWindow()
    if (!window || event.sender !== window.webContents) return ""
    const result = await dialog.showOpenDialog(window, { properties: ["openDirectory"] })
    return result.filePaths[0] ?? ""
  })
  ipcMain.handle("nnmind:python-select", (event) => {
    const window = getWindow()
    if (!window || event.sender !== window.webContents) return null
    return selectTrainingEnvironment(window, getLocale())
  })
  ipcMain.handle("nnmind:training-stop", (event) => event.sender === getWindow()?.webContents ? stopTraining() : undefined)
  ipcMain.handle("nnmind:inputs-inspect", (event, request: InputProbeRequest) => {
    if (event.sender !== getWindow()?.webContents) throw new Error("Invalid NNMind window.")
    return inspectInputScript(request)
  })
  ipcMain.handle("nnmind:training-start", async (event, request: TrainingRequest) => {
    if (event.sender !== getWindow()?.webContents) throw new Error("Invalid training window.")
    if (active || starting || inspecting) throw new Error("已有任务正在运行。")
    validateRequest(request)
    starting = true
    cancelStarting = false
    const send = (value: TrainingEvent) => {
      if (!event.sender.isDestroyed()) event.sender.send("nnmind:training-event", request.mode === "execute" || request.mode === "generate" ? { ...value, scope: "execute" } : value)
    }
    let work: string | undefined
    try {
      const environment = await readEnvironment()
      if (!environment?.torch) throw new Error("请先选择安装了 PyTorch 的本地 Python 环境。")
      if (request.savePath && !(await stat(request.savePath)).isDirectory()) throw new Error("请选择适配器保存目录。")
      work = await mkdtemp(join(tmpdir(), "nnmind-train-"))
      const directory = work
      const { model, dataset, lossCode, ...settings } = request
      await Promise.all([
        writeFile(join(directory, "runner.py"), runner, "utf8"),
        writeFile(join(directory, "model.py"), model, "utf8"),
        writeFile(join(directory, "data.py"), dataset, "utf8"),
        writeFile(join(directory, "loss.py"), lossCode, "utf8"),
        writeFile(join(directory, "settings.json"), JSON.stringify(settings), "utf8"),
      ])
      if (cancelStarting) {
        await rm(directory, { recursive: true, force: true })
        send({ type: "finished", status: "stopped" })
        return
      }
      const child = spawn(environment.path, ["-u", join(directory, "runner.py")], {
        cwd: directory, windowsHide: true, detached: process.platform !== "win32",
        env: pythonProcessEnvironment(environment),
      })
      const job = { child, stop: false }
      active = job
      let lastError = ""
      createInterface({ input: child.stdout }).on("line", (line) => {
        if (line.startsWith(prefix)) {
          try { send(JSON.parse(line.slice(prefix.length)) as TrainingEvent) } catch { send({ type: "log", message: line }) }
        } else send({ type: "log", message: line })
      })
      createInterface({ input: child.stderr }).on("line", (line) => {
        if (line.trim()) lastError = line
        send({ type: "log", message: line })
      })
      child.on("error", (error) => { lastError = error.message })
      child.once("close", (code) => {
        if (active === job) active = null
        send({ type: "finished", status: job.stop ? "stopped" : code === 0 ? "completed" : "failed", message: lastError })
        void rm(directory, { recursive: true, force: true }).catch(() => undefined)
      })
    } catch (error) {
      if (work) await rm(work, { recursive: true, force: true }).catch(() => undefined)
      throw error
    } finally {
      starting = false
    }
  })
}
