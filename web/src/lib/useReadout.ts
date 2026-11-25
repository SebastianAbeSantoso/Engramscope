import { useEffect, useState } from 'react'

import type { SetResult } from '../core/decode.worker'
import { ALL_COLUMNS, type Engine, type Lookup } from '../core/engine'
import { useIndexChunks } from './EngineContext'

export interface LayerReadout {
  layer: number
  /** decoded sets, in the order requested; layers other than 1 only get the intact set */
  sets: (SetResult | undefined)[]
  control: SetResult
  dims: number
}

export interface Readout {
  lookup: Lookup | null
  layers: LayerReadout[]
  loading: boolean
  error: string | null
  decodeMs: number
}

/** Look up a phrase and decode column sets of layer 1 (other layers: the intact set only), plus the
 *  random-rows control. The first set and the control come first, the rest after; everything is
 *  decoded again as more of the index loads, so the words sharpen. */
export function useReadout(engine: Engine | null, phrase: string, position: number | undefined, sets: number[][] = [ALL_COLUMNS], layers?: number[]): Readout {
  const chunks = useIndexChunks(engine)
  const [lookup, setLookup] = useState<Lookup | null>(null)
  const [out, setOut] = useState<LayerReadout[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [decodeMs, setDecodeMs] = useState(0)
  const setsKey = JSON.stringify(sets)
  const layersKey = JSON.stringify(layers ?? null)

  useEffect(() => {
    if (!engine || !phrase.trim()) return
    let alive = true
    setLoading(true)
    setError(null)
    setLookup(null)
    setOut([])
    engine
      .lookup(phrase, position)
      .then((l) => alive && setLookup(l))
      .catch((e) => alive && (setError(String(e)), setLoading(false)))
    return () => {
      alive = false
    }
  }, [engine, phrase, position])

  useEffect(() => {
    if (!engine || !lookup || chunks === 0) return
    let alive = true
    const parsed = JSON.parse(setsKey) as number[][]
    const only = JSON.parse(layersKey) as number[] | null
    const chosen = lookup.layers.filter((l) => !only || only.includes(l.layer))
    const key = `${lookup.phrase}@${lookup.position}`
    const t0 = performance.now()
    const run = async (all: boolean) =>
      Promise.all(
        chosen.map(async (l) => {
          const want = l.layer === 1 ? (all ? parsed : parsed.slice(0, 1)) : [ALL_COLUMNS]
          const [main, ctrl] = await Promise.all([engine.decode(key, l.layer, l.rows, want), engine.decode(`${key}#control`, l.layer, l.randomRows, [ALL_COLUMNS])])
          const padded: (SetResult | undefined)[] = l.layer === 1 ? parsed.map((_, i) => main.results[i]) : [main.results[0]]
          return { layer: l.layer, sets: padded, control: ctrl.results[0], dims: main.dims }
        }),
      )
    ;(async () => {
      try {
        const first = await run(false)
        if (!alive) return
        setOut(first)
        setDecodeMs(performance.now() - t0)
        setLoading(false)
        if (parsed.length > 1) {
          const full = await run(true)
          if (alive) setOut(full)
        }
      } catch (e) {
        if (alive) {
          setError(String(e))
          setLoading(false)
        }
      }
    })()
    return () => {
      alive = false
    }
  }, [engine, lookup, chunks, setsKey, layersKey])

  return { lookup, layers: out, loading, error, decodeMs }
}
