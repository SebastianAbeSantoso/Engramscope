import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'

import { Engine, type LoadProgress } from '../core/engine'

interface EngineState {
  engine: Engine | null
  progress: LoadProgress | null
  error: string | null
}

const Ctx = createContext<EngineState>({ engine: null, progress: null, error: null })

let shared: Promise<Engine> | null = null

/** Loads the instrument once, on first need, and shares it. */
export function EngineProvider({ children, start }: { children: ReactNode; start: boolean }) {
  const [state, setState] = useState<EngineState>({ engine: null, progress: null, error: null })
  useEffect(() => {
    if (!start) return
    let alive = true
    shared ??= Engine.load((p) => alive && setState((s) => ({ ...s, progress: p })))
    shared
      .then((engine) => {
        if (!alive) return
        engine.onProgress = (p) => setState((s) => ({ ...s, progress: p }))
        setState((s) => ({ ...s, engine }))
      })
      .catch((e) => alive && setState((s) => ({ ...s, error: String(e) })))
    return () => {
      alive = false
    }
  }, [start])
  return <Ctx.Provider value={state}>{children}</Ctx.Provider>
}

export const useEngine = () => useContext(Ctx)

/** Re-render when the number of loaded index chunks changes. */
export function useIndexChunks(engine: Engine | null): number {
  const [n, setN] = useState(engine?.indexChunks ?? 0)
  useEffect(() => {
    if (!engine) return
    let alive = true
    const loop = () => {
      setN(engine.indexChunks)
      if (engine.indexChunks < engine.c.decode_index.chunks) engine.nextChunk().then(() => alive && loop())
    }
    loop()
    return () => {
      alive = false
    }
  }, [engine])
  return n
}
