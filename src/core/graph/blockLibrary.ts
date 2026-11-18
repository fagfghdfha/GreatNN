import type { ParamValue } from "./types"

export type SavedBlock = {
  id: string
  name: string
  params: Record<string, ParamValue>
  width?: number
  height?: number
}

const libraryKey = "nnmind-custom-library-v1"

export function readBlockLibrary(): SavedBlock[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(libraryKey) ?? "[]")
    return Array.isArray(parsed) ? parsed.filter(item => typeof item?.id === "string" && typeof item?.name === "string" && item?.params) : []
  } catch {
    return []
  }
}

export function writeBlockLibrary(blocks: SavedBlock[]) {
  localStorage.setItem(libraryKey, JSON.stringify(blocks))
}
