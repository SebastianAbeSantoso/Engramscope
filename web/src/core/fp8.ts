// FP8 decoding for the model files: E4M3 values, E8M0 power-of-two scales, one scale per 32 values.
// Mirrors tools/engramscope/fp8.py.

function e4m3(code: number): number {
  const sign = code & 0x80 ? -1 : 1
  const exponent = (code >> 3) & 0xf
  const mantissa = code & 7
  if (exponent === 15 && mantissa === 7) return NaN
  if (exponent === 0) return sign * (mantissa / 8) * 2 ** -6
  return sign * (1 + mantissa / 8) * 2 ** (exponent - 7)
}

export const E4M3 = Float32Array.from({ length: 256 }, (_, c) => e4m3(c))
export const E8M0 = Float32Array.from({ length: 256 }, (_, c) => (c < 255 ? 2 ** (c - 127) : NaN))

/** One Engram row: `values` (head_dim bytes of E4M3) with one E8M0 scale per 32 values. */
export function decodeRow(values: Uint8Array, scales: Uint8Array, block = 32): Float32Array {
  if (scales.length * block !== values.length) throw new Error(`row has ${values.length} values and ${scales.length} scales`)
  const out = new Float32Array(values.length)
  for (let i = 0; i < values.length; i++) out[i] = E4M3[values[i]] * E8M0[scales[(i / block) | 0]]
  return out
}

/** IEEE half-precision bits to a number. */
export function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1
  const e = (h >> 10) & 0x1f
  const f = h & 0x3ff
  if (e === 0) return s * f * 2 ** -24
  if (e === 31) return f ? NaN : s * Infinity
  return s * (1 + f / 1024) * 2 ** (e - 15)
}

export function halfArrayToFloat32(bits: Uint16Array): Float32Array {
  const out = new Float32Array(bits.length)
  for (let i = 0; i < bits.length; i++) out[i] = halfToFloat(bits[i])
  return out
}
