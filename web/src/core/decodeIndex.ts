// The compressed readout (Decision 8; tools/engramscope/build_assets.py).
// scores = Z · y with y = M · x: Z is [vocab, d] at 4 bits with a per-token fp16 scale per chunk,
// M is [d, 6144] in fp16, x the 24 fetched rows. d = 1,024 in 4 chunks of 256, ordered by importance,
// so the readout works from the first chunk and sharpens as the others load.

export interface IndexChunk {
  /** packed int4, [vocab, chunkDims / 2]: low nibble = even dimension, stored as q + 8 */
  z: Uint8Array
  /** per-token scale, [vocab] */
  zs: Float32Array
  /** per layer: M rows of this chunk, [chunkDims, nIn] row-major */
  m: Record<number, Float32Array>
}

export interface TopK {
  ids: number[]
  scores: number[]
}

export class DecodeIndex {
  readonly chunks: IndexChunk[] = []
  readonly vocab: number
  readonly chunkDims: number
  readonly nIn: number
  readonly totalChunks: number

  constructor(vocab: number, chunkDims: number, nIn: number, totalChunks: number) {
    this.vocab = vocab
    this.chunkDims = chunkDims
    this.nIn = nIn
    this.totalChunks = totalChunks
  }

  addChunk(chunk: IndexChunk): void {
    if (chunk.z.length !== (this.vocab * this.chunkDims) / 2) throw new Error('chunk z has the wrong size')
    if (chunk.zs.length !== this.vocab) throw new Error('chunk scales have the wrong size')
    for (const m of Object.values(chunk.m)) if (m.length !== this.chunkDims * this.nIn) throw new Error('chunk M has the wrong size')
    this.chunks.push(chunk)
  }

  get dims(): number {
    return this.chunks.length * this.chunkDims
  }

  /** Per-column contributions in the reduced space: y_c = M[:, block_c] · row_c. rows: one per column. */
  contributions(layer: number, rows: Float32Array[]): Float32Array[] {
    const hd = this.nIn / rows.length
    const d = this.dims
    return rows.map((row, c) => {
      const y = new Float32Array(d)
      for (let k = 0; k < this.chunks.length; k++) {
        const m = this.chunks[k].m[layer]
        if (!m) throw new Error(`no M for layer ${layer}`)
        for (let j = 0; j < this.chunkDims; j++) {
          const base = j * this.nIn + c * hd
          let acc = 0
          for (let i = 0; i < hd; i++) acc += m[base + i] * row[i]
          y[k * this.chunkDims + j] = acc
        }
      }
      return y
    })
  }

  /** Token scores for a reduced vector y (length ≥ dims; only loaded chunks are used). */
  scores(y: Float32Array): Float32Array {
    const out = new Float32Array(this.vocab)
    const half = this.chunkDims / 2
    const table = new Float32Array(half * 256)
    for (let k = 0; k < this.chunks.length; k++) {
      const off = k * this.chunkDims
      for (let j = 0; j < half; j++) {
        const y0 = y[off + 2 * j]
        const y1 = y[off + 2 * j + 1]
        const tb = j * 256
        for (let b = 0; b < 256; b++) table[tb + b] = ((b & 15) - 8) * y0 + ((b >> 4) - 8) * y1
      }
      const { z, zs } = this.chunks[k]
      for (let t = 0; t < this.vocab; t++) {
        const zb = t * half
        let acc = 0
        for (let j = 0; j < half; j++) acc += table[j * 256 + z[zb + j]]
        out[t] += zs[t] * acc
      }
    }
    return out
  }

  topk(y: Float32Array, k = 10): TopK {
    return topK(this.scores(y), k)
  }
}

/** Indices of the k largest values, best first (ties: lower index first, as numpy's stable sort). */
export function topK(scores: Float32Array, k: number): TopK {
  const ids: number[] = []
  const vals: number[] = []
  for (let t = 0; t < scores.length; t++) {
    const s = scores[t]
    if (ids.length === k && s <= vals[k - 1]) continue
    let i = ids.length === k ? k - 1 : ids.length
    if (ids.length < k) {
      ids.push(t)
      vals.push(s)
    }
    while (i > 0 && vals[i - 1] < s) {
      ids[i] = ids[i - 1]
      vals[i] = vals[i - 1]
      i--
    }
    ids[i] = t
    vals[i] = s
  }
  return { ids, scores: vals }
}

export function sumVectors(vs: Float32Array[], keep?: boolean[]): Float32Array {
  const out = new Float32Array(vs[0]?.length ?? 0)
  vs.forEach((v, c) => {
    if (keep && !keep[c]) return
    for (let i = 0; i < v.length; i++) out[i] += v[i]
  })
  return out
}

export function cosine(a: Float32Array, b: Float32Array): number {
  let ab = 0
  let aa = 0
  let bb = 0
  for (let i = 0; i < a.length; i++) {
    ab += a[i] * b[i]
    aa += a[i] * a[i]
    bb += b[i] * b[i]
  }
  return aa === 0 || bb === 0 ? 0 : ab / Math.sqrt(aa * bb)
}
