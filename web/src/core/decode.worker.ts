// Decoding off the main thread. The worker loads the decode index chunk by chunk (reporting progress)
// and answers readout requests: for each set of columns, the top words of the sum of their
// contributions, and that sum's cosine with the full set (in the compressed space).

import { fetchAsset } from './assets'
import type { Constants } from './constants'
import { cosine, DecodeIndex, sumVectors } from './decodeIndex'
import { halfArrayToFloat32 } from './fp8'

export type WorkerRequest =
  | { type: 'load'; constants: Constants }
  | { type: 'decode'; id: number; key: string; layer: number; rows: Float32Array[]; sets: number[][]; k: number }

export interface SetResult {
  ids: number[]
  scores: number[]
  cosine: number
}

export type WorkerResponse =
  | { type: 'progress'; chunks: number; total: number; bytes: number; totalBytes: number }
  | { type: 'ready'; chunks: number }
  | { type: 'decoded'; id: number; dims: number; results: SetResult[] }
  | { type: 'error'; id?: number; message: string }

let index: DecodeIndex | null = null
const contribCache = new Map<string, { dims: number; y: Float32Array[] }>()
const resultCache = new Map<string, SetResult>()
const post = (m: WorkerResponse) => (self as unknown as Worker).postMessage(m)

async function load(c: Constants) {
  const di = c.decode_index
  index = new DecodeIndex(c.vocab, di.chunk_dims, c.n_cols * c.head_dim, di.chunks)
  const names = (k: number) => [`z${k}.bin`, `zs${k}.bin`, ...c.layer_ids.map((l) => `m${l}-${k}.bin`)]
  let totalBytes = 0
  for (let k = 0; k < di.chunks; k++) for (const n of names(k)) totalBytes += di.files[n].bytes
  let bytes = 0
  const onBytes = (n: number) => {
    bytes += n
    post({ type: 'progress', chunks: index!.chunks.length, total: di.chunks, bytes, totalBytes })
  }
  for (let k = 0; k < di.chunks; k++) {
    const [z, zs, ...ms] = await Promise.all(names(k).map((n) => fetchAsset(n, onBytes)))
    const m: Record<number, Float32Array> = {}
    c.layer_ids.forEach((l, i) => (m[l] = halfArrayToFloat32(new Uint16Array(ms[i]))))
    index.addChunk({ z: new Uint8Array(z), zs: halfArrayToFloat32(new Uint16Array(zs)), m })
    post({ type: 'progress', chunks: index.chunks.length, total: di.chunks, bytes, totalBytes })
  }
  post({ type: 'ready', chunks: index.chunks.length })
}

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data
  if (msg.type === 'load') {
    load(msg.constants).catch((err) => post({ type: 'error', message: String(err) }))
    return
  }
  try {
    if (!index || index.chunks.length === 0) throw new Error('decode index not loaded yet')
    const cacheKey = `${msg.layer}:${msg.key}`
    let entry = contribCache.get(cacheKey)
    if (!entry || entry.dims !== index.dims) {
      entry = { dims: index.dims, y: index.contributions(msg.layer, msg.rows) }
      contribCache.set(cacheKey, entry)
      if (contribCache.size > 64) contribCache.delete(contribCache.keys().next().value!)
    }
    const full = sumVectors(entry.y)
    const results = msg.sets.map((cols) => {
      const rkey = `${cacheKey}|${index!.dims}|${msg.k}|${[...cols].sort((x, y) => x - y).join(',')}`
      const hit = resultCache.get(rkey)
      if (hit) return hit
      const keep = entry!.y.map((_, c) => cols.includes(c))
      const v = sumVectors(entry!.y, keep)
      const top = cols.length ? index!.topk(v, msg.k) : { ids: [], scores: [] }
      const res = { ...top, cosine: cosine(v, full) }
      resultCache.set(rkey, res)
      if (resultCache.size > 1024) resultCache.delete(resultCache.keys().next().value!)
      return res
    })
    post({ type: 'decoded', id: msg.id, dims: index.dims, results })
  } catch (err) {
    post({ type: 'error', id: msg.id, message: String(err) })
  }
}
