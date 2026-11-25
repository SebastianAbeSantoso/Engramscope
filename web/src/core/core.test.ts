// Correctness property 9 (design.md): the browser hash equals the Python reference on the shared test
// vectors, and the browser readout matches the reference's compressed readout. The vectors and assets
// come from tools/engramscope/build_assets.py; without them these tests are skipped.

import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

import { REVISION, type Constants } from './constants'
import { DecodeIndex, sumVectors, topK } from './decodeIndex'
import { decodeRow, E4M3, E8M0, halfArrayToFloat32 } from './fp8'
import { Hasher } from './hash'
import { ModelTokenizer } from './tokenizer'

const dir = join(__dirname, '../../public/assets', REVISION)
const have = existsSync(join(dir, 'test-vectors.json'))
const read = (name: string) => readFileSync(join(dir, name))
const json = (name: string) => JSON.parse(read(name).toString('utf8'))
const fromHex = (h: string) => Uint8Array.from(h.match(/../g)!.map((b) => parseInt(b, 16)))

interface Vector {
  phrase: string
  text: string
  ids: number[]
  addresses: number[][][]
  rows?: Record<string, { values_hex: string[]; scales_hex: string[]; approx_top10: number[]; exact_top10: number[] }>
}

describe('fp8', () => {
  it('decodes E4M3 and E8M0 like the reference', () => {
    expect(E4M3[0x38]).toBe(1) // exponent 7, mantissa 0
    expect(E4M3[0xb8]).toBe(-1)
    expect(E4M3[0x01]).toBe(2 ** -9) // smallest subnormal
    expect(E4M3[0x7e]).toBe(448) // largest finite
    expect(Number.isNaN(E4M3[0x7f])).toBe(true)
    expect(E8M0[127]).toBe(1)
    expect(Number.isNaN(E8M0[255])).toBe(true)
    const row = decodeRow(new Uint8Array(64).fill(0x38), Uint8Array.from([128, 126]))
    expect(row[0]).toBe(2)
    expect(row[63]).toBe(0.5)
  })
})

describe('topK', () => {
  it('returns the largest values best first, earlier index on ties', () => {
    const r = topK(Float32Array.from([1, 5, 3, 5, 2, 9]), 3)
    expect(r.ids).toEqual([5, 1, 3])
  })
})

describe.skipIf(!have)('against the Python reference (test vectors)', () => {
  const c: Constants = have ? json('constants.json') : ({} as Constants)
  const vectors: Vector[] = have ? json('test-vectors.json').phrases : []

  it('tokenizes every test phrase exactly', () => {
    const tok = new ModelTokenizer(json('tokenizer.json'), json('tokenizer_config.json'))
    for (const v of vectors) expect(tok.encode(v.text), v.phrase).toEqual(v.ids)
  })

  it('computes every address exactly', () => {
    const b = read('token_map.bin')
    const hasher = new Hasher(c, new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4))
    for (const v of vectors) expect(hasher.addresses(v.ids), v.phrase).toEqual(v.addresses)
  })

  it('matches the reference compressed readout on fetched rows', { timeout: 120_000 }, () => {
    const di = c.decode_index
    const nIn = c.n_cols * c.head_dim
    const index = new DecodeIndex(c.vocab, di.chunk_dims, nIn, di.chunks)
    for (let k = 0; k < di.chunks; k++) {
      const z = read(`z${k}.bin`)
      const zs = read(`zs${k}.bin`)
      const m: Record<number, Float32Array> = {}
      for (const layer of c.layer_ids) {
        const mb = read(`m${layer}-${k}.bin`)
        m[layer] = halfArrayToFloat32(new Uint16Array(mb.buffer, mb.byteOffset, mb.byteLength / 2))
      }
      index.addChunk({
        z: new Uint8Array(z.buffer, z.byteOffset, z.byteLength),
        zs: halfArrayToFloat32(new Uint16Array(zs.buffer, zs.byteOffset, zs.byteLength / 2)),
        m,
      })
    }
    let checked = 0
    for (const v of vectors) {
      if (!v.rows) continue
      for (const layer of c.layer_ids) {
        const r = v.rows[String(layer)]
        const rows = r.values_hex.map((h, i) => decodeRow(fromHex(h), fromHex(r.scales_hex[i])))
        const y = sumVectors(index.contributions(layer, rows))
        const got = index.topk(y, 10).ids
        const same = got.filter((t) => r.approx_top10.includes(t)).length
        expect(same, `${v.phrase} layer ${layer}: ${got} vs ${r.approx_top10}`).toBeGreaterThanOrEqual(9)
        checked++
      }
    }
    expect(checked).toBeGreaterThan(0)
  })
})
