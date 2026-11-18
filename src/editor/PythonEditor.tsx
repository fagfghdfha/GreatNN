import { trainingDesktop } from "./trainingDesktop"
import { useEffect, useRef } from "react"
import { python } from "@codemirror/lang-python"
import { autocompletion, completionKeymap, completeFromList, type CompletionContext } from "@codemirror/autocomplete"
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands"
import { bracketMatching, defaultHighlightStyle, indentOnInput, syntaxHighlighting } from "@codemirror/language"
import { linter, lintGutter } from "@codemirror/lint"
import { EditorState } from "@codemirror/state"
import { EditorView, keymap, lineNumbers, highlightActiveLine } from "@codemirror/view"
import { pythonDiagnostics } from "../core/import/pythonModuleLibrary"

const torchNames = ["Linear", "Conv2d", "BatchNorm2d", "LayerNorm", "ReLU", "GELU", "LSTM", "GRU", "Sequential", "ModuleList", "CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss", "Dropout", "Flatten", "Embedding"]
const common = completeFromList(["def", "class", "import", "from", "return", "if", "else", "for", "in", "with", "try", "except", "None", "True", "False", "self", "torch", "nn", "F", "torch.relu", "torch.cat", "torch.stack", "torch.flatten", "torch.softmax", "torch.mean", "torch.randn", "torch.zeros", "range", "len", "enumerate", "super", "TensorDataset", "DataLoader"].map(label => ({ label, type: "function" })))

function completions(context: CompletionContext) {
  const attribute = context.matchBefore(/\b(?:nn|torch|self|F)\.\w*/)
  if (attribute) {
    const dot = attribute.text.indexOf(".")
    const object = attribute.text.slice(0, dot)
    const suggestions = object === "nn" ? torchNames : object === "torch"
      ? ["Tensor", "relu", "add", "cat", "stack", "flatten", "softmax", "mean", "randn", "zeros", "ones", "no_grad"]
      : object === "F" ? ["relu", "gelu", "cross_entropy", "mse_loss", "binary_cross_entropy_with_logits", "interpolate"]
        : [...new Set([...context.state.doc.toString().matchAll(/\bself\.(\w+)\s*=/g)].map(match => match[1]))]
    return { from: attribute.from + dot + 1, options: suggestions.map(label => ({ label, type: "property" })) }
  }
  return common(context)
}

export function PythonEditor({ value, onChange, label, imports = "", sourcePath = "", symbols = [], insertion }: { value: string; onChange: (value: string) => void; label: string; imports?: string; sourcePath?: string; symbols?: string[]; insertion?: { text: string } }) {
  const div = useRef<HTMLDivElement>(null)
  const view = useRef<EditorView | null>(null)
  const contextRef = useRef({ imports, sourcePath, symbols })
  useEffect(() => { contextRef.current = { imports, sourcePath, symbols } }, [imports, sourcePath, symbols])
  async function complete(context: CompletionContext) {
    const line = context.state.doc.lineAt(context.pos)
    const token = context.matchBefore(/\w+/)
    const fallback = completions(context)
    const { imports, sourcePath, symbols } = contextRef.current
    const key = context.matchBefore(/tensors\[["']\w*/)
    if (key) return { from: key.from + 9, options: symbols.map(label => ({ label, type: "variable" })) }
    if (!context.explicit && !token && !line.text.slice(0, context.pos - line.from).endsWith(".")) return fallback
    const code = imports + context.state.doc.toString()
    try {
      const options = await trainingDesktop?.completePython?.({
        code, path: sourcePath, line: line.number + (imports.match(/\n/g)?.length ?? 0),
        column: Array.from(line.text.slice(0, context.pos - line.from)).length,
      })
      if (context.aborted || !options?.length) return fallback
      return { from: token?.from ?? context.pos, options, validFor: /^\w*$/ }
    } catch { return fallback }
  }

  const onChangeRef = useRef(onChange)
  useEffect(() => { onChangeRef.current = onChange }, [onChange])
  useEffect(() => {
    if (!div.current) return
    const editor = new EditorView({
      parent: div.current,
      state: EditorState.create({
        doc: value,
        extensions: [
          lineNumbers(), highlightActiveLine(), history(), python(), indentOnInput(), bracketMatching(),
          syntaxHighlighting(defaultHighlightStyle),
          keymap.of([...completionKeymap, ...defaultKeymap, ...historyKeymap, indentWithTab]),
          autocompletion({ override: [complete], activateOnTyping: true }),
          lintGutter(),
          linter((editorView) => pythonDiagnostics(editorView.state.doc.toString()).map(({ from, to }) => ({
            from, to, severity: "error", message: "Python syntax or indentation / Python 语法或缩进错误",
          })), { delay: 350 }),
          EditorView.contentAttributes.of({ "aria-label": label, spellcheck: "false" }),
          EditorView.updateListener.of(update => {
            if (update.docChanged) onChangeRef.current(update.state.doc.toString())
          }),
        ],
      }),
    })
    view.current = editor
    return () => { editor.destroy(); view.current = null }
    // The editor instance retains its undo stack; external changes use the effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [label])
  useEffect(() => {
    const editor = view.current
    if (editor && editor.state.doc.toString() !== value) {
      editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: value } })
    }
  }, [value])
  useEffect(() => {
    const editor = view.current
    if (!editor || !insertion) return
    editor.dispatch(editor.state.replaceSelection(insertion.text))
    editor.focus()
  }, [insertion])
  return <div className="python-editor" ref={div} />
}
