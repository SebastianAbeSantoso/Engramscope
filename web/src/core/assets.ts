// Loading the built assets (web/public/assets/<revision>/), with Cache Storage so the decode index
// downloads once per browser.

import { REVISION } from './constants'

export const ASSET_BASE = `${import.meta.env.BASE_URL}assets/${REVISION}/`
const CACHE = `engramscope-${REVISION}`

/** Fetch an asset, from Cache Storage when available. `onBytes` reports download progress. */
export async function fetchAsset(name: string, onBytes?: (n: number) => void): Promise<ArrayBuffer> {
  const url = ASSET_BASE + name
  const cache = typeof caches !== 'undefined' ? await caches.open(CACHE).catch(() => null) : null
  const hit = cache ? await cache.match(url) : undefined
  if (hit) {
    const b = await hit.arrayBuffer()
    onBytes?.(b.byteLength)
    return b
  }
  const r = await fetch(url)
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${name}`)
  let buf: ArrayBuffer
  if (r.body && onBytes) {
    const reader = r.body.getReader()
    const parts: Uint8Array[] = []
    let n = 0
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      parts.push(value)
      n += value.length
      onBytes(value.length)
    }
    const all = new Uint8Array(n)
    let o = 0
    for (const p of parts) {
      all.set(p, o)
      o += p.length
    }
    buf = all.buffer
  } else {
    buf = await r.arrayBuffer()
  }
  if (cache) await cache.put(url, new Response(buf.slice(0))).catch(() => undefined)
  return buf
}

export async function fetchJson<T>(name: string): Promise<T> {
  return JSON.parse(new TextDecoder().decode(await fetchAsset(name))) as T
}
