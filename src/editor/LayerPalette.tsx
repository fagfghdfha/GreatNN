import type { SavedBlock } from "../core/graph/blockLibrary"
import type { CSSProperties } from "react"
import { useMemo, useState } from "react"
import { layerLibrary } from "../core/registry/layerRegistry"
import type { LayerType } from "../core/graph/types"
import {
  type Locale,
  getLayerCategoryLabel,
  getLayerDescription,
  getLayerLabel,
  getUiText,
} from "../i18n"

const sortedLayers = [...layerLibrary].sort((left, right) => {
  const priority = (type: LayerType) => type === "Input" ? 0 : type === "Output" ? 1 : 2
  return priority(left.type) - priority(right.type) || left.type.localeCompare(right.type, "en")
})

type LayerPaletteProps = {
  locale: Locale
  onAddLayer: (layerType: LayerType) => void
  selectedNodeName?: string
  savedBlocks: SavedBlock[]
  onAddSaved: (block: SavedBlock) => void
  onRemoveSaved: (id: string) => void
}

export function LayerPalette({ locale, onAddLayer, selectedNodeName, savedBlocks, onAddSaved, onRemoveSaved }: LayerPaletteProps) {
  const [query, setQuery] = useState("")
  const [libraryOnly, setLibraryOnly] = useState(false)
  const text = getUiText(locale)
  const zh = locale === "zh"
  const visibleSaved = savedBlocks.filter(block => block.name.toLowerCase().includes(query.trim().toLowerCase()))

  const filteredLayers = useMemo(() => {
    const keyword = query.trim().toLowerCase()

    return keyword
      ? sortedLayers.filter(
          (layer) => {
            const localizedLabel = getLayerLabel(layer.type, layer.label, locale).toLowerCase()
            const localizedCategory = getLayerCategoryLabel(layer.category, locale).toLowerCase()
            const localizedDescription = getLayerDescription(layer.type, layer.description, locale).toLowerCase()

            return (
              layer.label.toLowerCase().includes(keyword) ||
              layer.type.toLowerCase().includes(keyword) ||
              layer.category.toLowerCase().includes(keyword) ||
              layer.description.toLowerCase().includes(keyword) ||
              localizedLabel.includes(keyword) ||
              localizedCategory.includes(keyword) ||
              localizedDescription.includes(keyword)
            )
          },
        )
      : sortedLayers
  }, [locale, query])

  return (
    <section className="panel quick-add-panel">
      <div className="quick-add-header">
        <div className="panel__header">
          <p>{selectedNodeName ? text.quickAddHintSelected(selectedNodeName) : text.quickAddHintIdle}</p>
        </div>
        <label className="field quick-add-search">
          <input aria-label={text.search} value={query} onChange={(event) => setQuery(event.target.value)} placeholder={text.searchPlaceholder} />
        </label>
      </div>
      <div className="palette-tabs" role="group" aria-label={zh ? "模块库" : "Block Libraries"}>
        <button type="button" aria-pressed={!libraryOnly} onClick={() => setLibraryOnly(false)}>{zh ? "内置模块" : "Built-in"}</button>
        <button type="button" aria-pressed={libraryOnly} onClick={() => setLibraryOnly(true)}>{zh ? "自建库" : "My Blocks"} <span>{savedBlocks.length}</span></button>
      </div>
      {libraryOnly ? (
        <div className="saved-blocks">
          {!visibleSaved.length && <p>{zh ? "可从自定义块检查器中固化模块。" : "Save a custom block from its inspector."}</p>}
          {visibleSaved.map(block => (
            <div key={block.id} className="saved-block">
              <button type="button" className="quick-add-card" style={{ "--accent": "#763eb2" } as CSSProperties} onClick={() => onAddSaved(block)}>{block.name}</button>
              <button type="button" aria-label={zh ? `删除 ${block.name}` : `Delete ${block.name}`} title={zh ? "从自建库删除" : "Remove from library"}
                onClick={() => { if (window.confirm(zh ? `从自建库删除「${block.name}」？` : `Remove “${block.name}” from My Blocks?`)) onRemoveSaved(block.id) }}>×</button>
            </div>
          ))}
        </div>
      ) : <div className="quick-add-row">
        {filteredLayers.length === 0 && <p>{text.connectionMenuEmpty}</p>}
        {filteredLayers.map((layer) => (
          <button
            key={layer.type}
            type="button"
            className="quick-add-card"
            onClick={() => onAddLayer(layer.type)}
            style={{ "--accent": layer.accent } as CSSProperties}
            title={getLayerDescription(layer.type, layer.description, locale)}
          >
            <span className="quick-add-card__label">{getLayerLabel(layer.type, layer.label, locale)}</span>
            <span className="quick-add-card__meta">{getLayerCategoryLabel(layer.category, locale)}</span>
          </button>
        ))}
      </div>}
    </section>
  )
}
