// Shape of constants.json, written by tools/engramscope/build_assets.py.

export interface TensorInfo {
  shard: string
  values_start: number
  scales_start: number
  rows: number
}

export interface Constants {
  repo: string
  revision: string
  hub: string
  layer_ids: number[]
  max_ngram_size: number
  n_heads: number
  n_cols: number
  head_dim: number
  hidden: number
  vocab: number
  token_map_size: number
  compressed_vocab_size: number
  pad: number
  primes: number[][]
  offsets: number[][]
  multipliers: string[][]
  num_embeddings: number[]
  tensors: Record<string, TensorInfo>
  decode_index: {
    method: string
    d: number
    chunks: number
    chunk_dims: number
    bits: number
    qmax: number
    energy_kept: number
    files: Record<string, { bytes: number; sha256: string }>
  }
}

export const REVISION = '314113f8a7dadb388215396ba5fbee2c610a04d6'

/** n-gram size and head of column c (c = (n - 2) * n_heads + head). */
export function column(c: number, nHeads = 8): { n: number; head: number } {
  return { n: 2 + Math.floor(c / nHeads), head: c % nHeads }
}
