import { useEffect, useState } from 'react'

export const EXAMPLES = ['the Eiffel Tower', 'the Kremlin', 'the Leaning Tower of Pisa', 'the Dutch East Indies', 'Albert Einstein', 'Mount Fuji']

/** Phrase input with example chips. Single words have no readable trace (see the Experiments page). */
export function PhraseForm({ initial, onSubmit, label = 'A phrase' }: { initial: string; onSubmit: (p: string) => void; label?: string }) {
  const [value, setValue] = useState(initial)
  useEffect(() => setValue(initial), [initial])
  return (
    <div className="phrase">
      <form
        onSubmit={(e) => {
          e.preventDefault()
          if (value.trim()) onSubmit(value.trim())
        }}
      >
        <label htmlFor="phrase-input">{label}</label>
        <div className="phrase__row">
          <input id="phrase-input" value={value} onChange={(e) => setValue(e.target.value)} placeholder="the Eiffel Tower" autoComplete="off" spellCheck={false} maxLength={200} />
          <button type="submit">Look up</button>
        </div>
      </form>
      <p className="examples">
        <span>Examples:</span>
        {EXAMPLES.map((p) => (
          <button key={p} type="button" className="chip" onClick={() => onSubmit(p)}>
            {p}
          </button>
        ))}
      </p>
    </div>
  )
}
