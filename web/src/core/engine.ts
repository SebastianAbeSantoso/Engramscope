// The instrument, end to end: phrase → tokens → compressed ids → 24 addresses per layer → fetched rows
// → per-row contributions → decoded words, with a control of random rows (Req 2, 3, 4).

import { fetchAsset, fetchJson } from './assets'
import type { Constants } from './constants'
import type { SetResult, WorkerRequest, WorkerResponse } from './decode.worker'
import { RowFetcher, type FetchStats } from './fetcher'
import { Hasher } from './hash'
import { ModelTokenizer, phraseText } from './tokenizer'

export interface LoadProgress {
  stage: 'basics' | 'index'
  chunks: number
  total: number
  bytes: number
  totalBytes: number
}

export interface Token {
  id: number
  text: string
  compressed: number
}

export interface LayerLookup {
  layer: number
  addresses: number[]
  rows: Float32Array[]
  randomAddresses: number[]
  randomRows: Float32Array[]
}

export interface Lookup {
  phrase: string
  tokens: Token[]
  position: number
  layers: LayerLookup[]
  stats: FetchStats
  /** bytes of the Engram tables in total, for "12 KB of 189 GiB" */
  tableBytes: number
}

export interface Decoded {
  dims: number
  results: SetResult[]
}

/** A small seeded generator, so the control rows of a phrase are the same every time. */
function seeded(text: string): () => number {
  let h = 2166136261
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619)
  return () => {
    h = (h + 0x6d2b79f5) | 0
    let t = Math.imul(h ^ (h >>> 15), 1 | h)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export class Engine {
  readonly c: Constants
  readonly hasher: Hasher
  readonly tokenizer: ModelTokenizer
  readonly strings: string[]
  readonly fetcher: RowFetcher
  private readonly worker: Worker
  private nextId = 1
  private readonly pending = new Map<number, { resolve: (d: Decoded) => void; reject: (e: Error) => void }>()
  private readyChunks = 0
  private readonly readyWaiters: (() => void)[] = []
  onProgress?: (p: LoadProgress) => void

  private constructor(c: Constants, hasher: Hasher, tokenizer: ModelTokenizer, strings: string[]) {
    this.c = c
    this.hasher = hasher
    this.tokenizer = tokenizer
    this.strings = strings
    this.fetcher = new RowFetcher(c)
    this.worker = new Worker(new URL('./decode.worker.ts', import.meta.url), { type: 'module' })
    this.worker.onmessage = (e: MessageEvent<WorkerResponse>) => this.onMessage(e.data)
    this.worker.postMessage({ type: 'load', constants: c } satisfies WorkerRequest)
  }

  static async load(onProgress?: (p: LoadProgress) => void): Promise<Engine> {
    onProgress?.({ stage: 'basics', chunks: 0, total: 4, bytes: 0, totalBytes: 0 })
    const c = await fetchJson<Constants>('constants.json')
    const [map, tj, tc, strings] = await Promise.all([
      fetchAsset('token_map.bin'),
      fetchJson<object>('tokenizer.json'),
      fetchJson<object>('tokenizer_config.json'),
      fetchJson<string[]>('strings.json'),
    ])
    const engine = new Engine(c, new Hasher(c, new Uint32Array(map)), new ModelTokenizer(tj, tc), strings)
    engine.onProgress = onProgress
    return engine
  }

  private onMessage(m: WorkerResponse) {
    if (m.type === 'progress') {
      this.onProgress?.({ stage: 'index', chunks: m.chunks, total: m.total, bytes: m.bytes, totalBytes: m.totalBytes })
      if (m.chunks > this.readyChunks) {
        this.readyChunks = m.chunks
        this.readyWaiters.splice(0).forEach((r) => r())
      }
    } else if (m.type === 'ready') {
      this.readyChunks = m.chunks
      this.readyWaiters.splice(0).forEach((r) => r())
    } else if (m.type === 'decoded') {
      this.pending.get(m.id)?.resolve({ dims: m.dims, results: m.results })
      this.pending.delete(m.id)
    } else if (m.type === 'error') {
      if (m.id !== undefined) {
        this.pending.get(m.id)?.reject(new Error(m.message))
        this.pending.delete(m.id)
      } else console.error(m.message)
    }
  }

  get indexChunks(): number {
    return this.readyChunks
  }

  /** Resolves once at least one chunk of the decode index is loaded. */
  whenDecodable(): Promise<void> {
    return this.readyChunks > 0 ? Promise.resolve() : new Promise((r) => this.readyWaiters.push(r))
  }

  /** Resolves on the next change in the number of loaded chunks. */
  nextChunk(): Promise<void> {
    return new Promise((r) => this.readyWaiters.push(r))
  }

  tokens(phrase: string): Token[] {
    const ids = this.tokenizer.encode(phraseText(phrase))
    const comp = this.hasher.compress(ids)
    return ids.map((id, i) => ({ id, text: this.strings[id] ?? '', compressed: comp[i] }))
  }

  async lookup(phrase: string, position?: number): Promise<Lookup> {
    const tokens = this.tokens(phrase)
    if (tokens.length === 0) throw new Error('empty phrase')
    const pos = position === undefined ? tokens.length - 1 : Math.max(0, Math.min(tokens.length - 1, position))
    const addr = this.hasher.addresses(tokens.map((t) => t.id))[pos]
    const stats: FetchStats = { requests: 0, bytes: 0, ms: 0 }
    const t0 = performance.now()
    const rand = seeded(phrase.trim().toLowerCase())
    const layers = await Promise.all(
      this.c.layer_ids.map(async (layer, li) => {
        const randomAddresses = Array.from({ length: this.c.n_cols }, () => Math.floor(rand() * this.c.num_embeddings[li]))
        const [rows, randomRows] = await Promise.all([this.fetcher.rows(layer, addr[li], stats), this.fetcher.rows(layer, randomAddresses, stats)])
        return { layer, addresses: addr[li], rows, randomAddresses, randomRows }
      }),
    )
    stats.ms = performance.now() - t0
    const tableBytes = this.c.num_embeddings.reduce((a, n) => a + n * (this.c.head_dim + this.c.head_dim / 32), 0)
    return { phrase, tokens, position: pos, layers, stats, tableBytes }
  }

  /** Decode sets of columns of one layer's rows. Each set: the columns kept. */
  decode(key: string, layer: number, rows: Float32Array[], sets: number[][], k = 10): Promise<Decoded> {
    const id = this.nextId++
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      this.worker.postMessage({ type: 'decode', id, key, layer, rows, sets, k } satisfies WorkerRequest)
    })
  }

  word(id: number): string {
    return this.strings[id] ?? `#${id}`
  }
}

export const ALL_COLUMNS = Array.from({ length: 24 }, (_, i) => i)
export const columnsOfSize = (n: number) => ALL_COLUMNS.filter((c) => 2 + Math.floor(c / 8) === n)
export const columnsOfHead = (h: number) => ALL_COLUMNS.filter((c) => c % 8 === h)
