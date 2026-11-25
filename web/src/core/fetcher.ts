// Row access over HTTP range requests (Req 2.1, 2.4). Each shard's CDN URL is resolved once per
// session (the hub's resolver is rate-limited); ranges then go to the CDN directly, with a
// concurrency limit, back-off on 429 and 5xx, and a fresh redirect if a signed URL expires (403).
// The CDN rejects multi-range requests (416), so every range is its own request.

import type { Constants } from './constants'
import { decodeRow } from './fp8'

export interface FetchStats {
  requests: number
  bytes: number
  ms: number
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

export class RowFetcher {
  private readonly c: Constants
  private readonly cdn = new Map<string, Promise<string>>()
  private readonly cache = new Map<string, Float32Array>()
  private active = 0
  private readonly queue: (() => void)[] = []
  readonly concurrency: number
  readonly fetchImpl: typeof fetch

  constructor(c: Constants, concurrency = 32, fetchImpl: typeof fetch = (...a) => fetch(...a)) {
    this.c = c
    this.concurrency = concurrency
    this.fetchImpl = fetchImpl
  }

  private resolve(shard: string, refresh = false): Promise<string> {
    if (refresh || !this.cdn.has(shard)) {
      const p = (async () => {
        const r = await this.fetchImpl(this.c.hub + shard, { headers: { Range: 'bytes=0-0' } })
        if (r.status !== 206 && r.status !== 200) throw new Error(`HTTP ${r.status} resolving ${shard}`)
        await r.arrayBuffer()
        return r.url || this.c.hub + shard
      })()
      this.cdn.set(shard, p)
      p.catch(() => this.cdn.delete(shard))
    }
    return this.cdn.get(shard)!
  }

  private async slot<T>(fn: () => Promise<T>): Promise<T> {
    if (this.active >= this.concurrency) await new Promise<void>((r) => this.queue.push(r))
    this.active++
    try {
      return await fn()
    } finally {
      this.active--
      this.queue.shift()?.()
    }
  }

  private async range(shard: string, start: number, length: number): Promise<Uint8Array> {
    let refresh = false
    for (let attempt = 0; attempt < 8; attempt++) {
      const url = await this.resolve(shard, refresh)
      refresh = false
      let r: Response
      try {
        r = await this.fetchImpl(url, { headers: { Range: `bytes=${start}-${start + length - 1}` } })
      } catch {
        await sleep(500 * (attempt + 1))
        continue
      }
      if (r.status === 429 || r.status >= 500) {
        const after = Number(r.headers.get('Retry-After'))
        await sleep(after > 0 ? after * 1000 : 2 ** attempt * 250)
        continue
      }
      if (r.status === 403) {
        refresh = true
        continue
      }
      if (r.status !== 206) throw new Error(`HTTP ${r.status} for ${shard} bytes ${start}+${length}`)
      const b = new Uint8Array(await r.arrayBuffer())
      if (b.length !== length) throw new Error(`got ${b.length} bytes for ${shard} bytes ${start}+${length}`)
      return b
    }
    throw new Error(`giving up on ${shard} bytes ${start}+${length}`)
  }

  /** Byte ranges (values and scales) of one row. */
  rowRanges(layer: number, row: number): { shard: string; values: [number, number]; scales: [number, number] } {
    const t = this.c.tensors[String(layer)]
    if (row < 0 || row >= t.rows) throw new Error(`row ${row} outside layer ${layer}`)
    const hd = this.c.head_dim
    return { shard: t.shard, values: [t.values_start + row * hd, hd], scales: [t.scales_start + row * (hd / 32), hd / 32] }
  }

  /** Decoded rows in the order given; cached for the session. */
  async rows(layer: number, rows: number[], stats?: FetchStats): Promise<Float32Array[]> {
    const t0 = performance.now()
    const out = await Promise.all(
      rows.map(async (row) => {
        const key = `${layer}:${row}`
        const hit = this.cache.get(key)
        if (hit) return hit
        const { shard, values, scales } = this.rowRanges(layer, row)
        const [v, s] = await Promise.all([
          this.slot(() => this.range(shard, ...values)),
          this.slot(() => this.range(shard, ...scales)),
        ])
        if (stats) {
          stats.requests += 2
          stats.bytes += v.length + s.length
        }
        const decoded = decodeRow(v, s)
        this.cache.set(key, decoded)
        return decoded
      }),
    )
    if (stats) stats.ms += performance.now() - t0
    return out
  }
}
