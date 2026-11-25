// Engram addresses, bit-exact with tools/engramscope/hashing.py (and the model's inference/engram.py).
// Products and XORs stay below 2^63 (the multipliers are bounded for that), so BigInt arithmetic on
// non-negative values gives the same bits as the reference's int64.

import type { Constants } from './constants'

export class Hasher {
  private readonly mult: bigint[][]
  private readonly primes: bigint[][]
  private readonly offsets: number[][]
  private readonly n: number
  private readonly nHeads: number
  readonly pad: number
  readonly tokenMap: Uint32Array
  private readonly c: Constants

  constructor(c: Constants, tokenMap: Uint32Array) {
    if (tokenMap.length !== c.token_map_size) throw new Error(`token map has ${tokenMap.length} ids, expected ${c.token_map_size}`)
    const sums = c.primes.map((row) => row.reduce((a, b) => a + b, 0))
    if (sums.some((s, i) => s !== c.num_embeddings[i])) throw new Error(`bucket primes sum to ${sums}, expected ${c.num_embeddings}`)
    this.c = c
    this.tokenMap = tokenMap
    this.mult = c.multipliers.map((row) => row.map((m) => BigInt(m)))
    this.primes = c.primes.map((row) => row.map((p) => BigInt(p)))
    this.offsets = c.offsets
    this.n = c.max_ngram_size
    this.nHeads = c.n_heads
    this.pad = c.pad
  }

  /** Compressed ids of the tokens. */
  compress(tokenIds: number[]): number[] {
    return tokenIds.map((t) => {
      if (t < 0 || t >= this.tokenMap.length) throw new Error(`token id ${t} outside the vocabulary`)
      return this.tokenMap[t]
    })
  }

  /** [position][layer][column] row ids. */
  addresses(tokenIds: number[]): number[][][] {
    const ids = this.compress(tokenIds).map((x) => BigInt(x))
    const pad = BigInt(this.pad)
    const out: number[][][] = []
    for (let t = 0; t < ids.length; t++) {
      const window: bigint[] = []
      for (let s = 0; s < this.n; s++) window.push(t - s >= 0 ? ids[t - s] : pad)
      const perLayer: number[][] = []
      for (let li = 0; li < this.c.layer_ids.length; li++) {
        const m = this.mult[li]
        let rolling = window[0] * m[0]
        const cols: number[] = []
        for (let i = 1; i < this.n; i++) {
          rolling ^= window[i] * m[i]
          for (let h = 0; h < this.nHeads; h++) {
            const col = cols.length
            cols.push(Number(rolling % this.primes[li][col]) + this.offsets[li][col])
          }
        }
        perLayer.push(cols)
      }
      out.push(perLayer)
    }
    return out
  }

  /** The n-gram window (compressed ids, newest first, padded) behind each column at position t. */
  windows(tokenIds: number[], t: number): number[][] {
    const ids = this.compress(tokenIds)
    const win: number[] = []
    for (let s = 0; s < this.n; s++) win.push(t - s >= 0 ? ids[t - s] : this.pad)
    return Array.from({ length: this.n - 1 }, (_, i) => win.slice(0, i + 2))
  }
}
