import type { SetResult } from '../core/decode.worker'
import type { Engine } from '../core/engine'

/** Decoded words, fading by rank. Words are shown as the tokenizer decodes them. */
export function Words({ engine, result, label, dim, compact }: { engine: Engine; result?: SetResult; label?: string; dim?: boolean; compact?: boolean }) {
  if (!result) return <div className="words words--empty">…</div>
  if (result.ids.length === 0) return <div className="words words--empty">nothing left</div>
  const words = result.ids.map((id) => engine.word(id))
  return (
    <div className={`words${dim ? ' words--dim' : ''}${compact ? ' words--compact' : ''}`} aria-label={label ? `${label}: ${words.join(', ')}` : words.join(', ')}>
      {words.map((w, i) => (
        <span key={i} className="word" style={{ opacity: 1 - (i / words.length) * 0.6 }} title={`token ${result.ids[i]}`}>
          {w.trim() === '' ? `␣${JSON.stringify(w)}` : w}
        </span>
      ))}
    </div>
  )
}

export function LoadNote({ engine, dims }: { engine: Engine; dims?: number }) {
  const total = engine.c.decode_index.d
  if (!dims || dims >= total) return <p className="note">Approximate decoding (compressed index, {total} dimensions). Every number on the Experiments page comes from the exact readout.</p>
  return (
    <p className="note note--loading">
      Still loading the decoding index: decoded with {dims} of {total} dimensions so far. The words sharpen as the rest arrives.
    </p>
  )
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`
  return `${(n / 1024 ** 3).toFixed(0)} GiB`
}
