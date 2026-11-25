import { useMemo, useState } from 'react'

import { LoadNote, Words } from '../components/Words'
import { PhraseForm } from '../components/PhraseForm'
import { column } from '../core/constants'
import { ALL_COLUMNS, columnsOfHead, columnsOfSize, type Engine } from '../core/engine'
import { useEngine } from '../lib/EngineContext'
import { go, href } from '../lib/useHash'
import { useReadout } from '../lib/useReadout'
import results from '../data/results.json'

const L1 = results.lesion.L1
const pct = (x: number) => `${Math.round(x * 100)}%`

interface Preset {
  label: string
  removed: () => number[]
  recorded?: string
}

function randomSubset(k: number): number[] {
  const a = [...ALL_COLUMNS]
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a.slice(0, k)
}

const retentionAt = (k: number) => L1.Q1.retention[L1.Q1.k.indexOf(k)]

const PRESETS: Preset[] = [
  { label: 'Intact', removed: () => [] },
  { label: 'Remove 8 at random', removed: () => randomSubset(8), recorded: `In the experiments, 8 random traces removed left the concept in ${pct(retentionAt(8))} of phrases.` },
  { label: 'Remove 20 at random', removed: () => randomSubset(20), recorded: `20 random traces removed left the concept in ${pct(retentionAt(20))} of phrases.` },
  {
    label: 'Remove the 2-word traces',
    removed: () => columnsOfSize(2),
    recorded: `Removing the 8 two-word traces left the concept in ${pct(L1.Q2.lesions['B3(2)'].retention)} of phrases, against ${pct(L1.Q2.lesions['B3(2)'].random_same_size)} for 8 random traces.`,
  },
  {
    label: 'Remove the 4-word traces',
    removed: () => columnsOfSize(4),
    recorded: `Removing the 8 four-word traces left the concept in ${pct(L1.Q2.lesions['B3(4)'].retention)} of phrases: less damaging than chance.`,
  },
  {
    label: 'Remove head 0',
    removed: () => columnsOfHead(0),
    recorded: `Removing any one head (3 traces) cost about the same as 3 random traces: ${pct(L1.Q2.lesions['B4(0)'].retention)} for head 0, ${pct(L1.Q2.lesions['B4(0)'].random_same_size)} at random.`,
  },
  {
    label: 'Keep only the 2-word traces',
    removed: () => ALL_COLUMNS.filter((c) => !columnsOfSize(2).includes(c)),
    recorded: `A 2-word cue alone retrieved the concept in ${pct(L1.Q3a.sizes['2'].retention)} of phrases (Semon's partial cue).`,
  },
]

export function Lesion({ params }: { params: URLSearchParams }) {
  const phrase = params.get('p') ?? 'the Eiffel Tower'
  const posParam = params.get('pos')
  const position = posParam === null ? undefined : Number(posParam)
  const { engine, error } = useEngine()
  return (
    <div className="lesion">
      <h1>Live lesion</h1>
      <p className="lede">
        Switch a phrase's traces off and watch what is left of the memory. This is exploration with the approximate browser decoding. The recorded, pre-registered results are on the <a href={href('experiments')}>Experiments</a> page.
      </p>
      <PhraseForm initial={phrase} onSubmit={(p) => go('lesion', { p })} />
      {error && <p className="error">{error}</p>}
      {!engine && !error && <p className="note">Loading…</p>}
      {engine && <LesionGrid key={`${phrase}@${position}`} engine={engine} phrase={phrase} position={position} />}
    </div>
  )
}

function LesionGrid({ engine, phrase, position }: { engine: Engine; phrase: string; position?: number }) {
  const [removed, setRemoved] = useState<Set<number>>(new Set())
  const [note, setNote] = useState<string | undefined>()
  const kept = useMemo(() => ALL_COLUMNS.filter((c) => !removed.has(c)), [removed])
  const sets = useMemo(() => [ALL_COLUMNS, kept], [kept])
  const r = useReadout(engine, phrase, position, sets, [1])
  const layer1 = r.layers.find((x) => x.layer === 1)
  const toggle = (c: number) => {
    const next = new Set(removed)
    if (next.has(c)) next.delete(c)
    else next.add(c)
    setRemoved(next)
    setNote(undefined)
  }
  if (r.error) return <p className="error">{r.error}</p>
  return (
    <>
      <section aria-labelledby="grid-h">
        <h2 id="grid-h">
          The 24 traces of “{phrase}” <span className="muted">(layer 1)</span>
        </h2>
        <div className="presets" role="group" aria-label="Lesion presets">
          {PRESETS.map((p) => (
            <button
              key={p.label}
              type="button"
              className="chip"
              onClick={() => {
                setRemoved(new Set(p.removed()))
                setNote(p.recorded)
              }}
            >
              {p.label}
            </button>
          ))}
        </div>
        <div className="grid" role="group" aria-label="Traces: rows are n-gram sizes, columns are hash heads">
          <div className="grid__corner" aria-hidden />
          {Array.from({ length: 8 }, (_, h) => (
            <div key={h} className="grid__label" aria-hidden>
              head {h}
            </div>
          ))}
          {[2, 3, 4].map((n) => (
            <Row key={n} n={n} removed={removed} toggle={toggle} />
          ))}
        </div>
        <p className="stats">
          {removed.size} of 24 traces removed · similarity to the intact memory: {layer1?.sets[1] ? layer1.sets[1].cosine.toFixed(2) : '…'} <span className="muted">(cosine, in the compressed space)</span>
        </p>
        {note && <p className="recorded">Recorded result: {note}</p>}
      </section>
      <section className="readout__grid">
        <div className="panel panel--main">
          <h3>What is left</h3>
          <Words engine={engine} result={layer1?.sets[1]} label="after the lesion" />
        </div>
        <div className="panel">
          <h3>Intact</h3>
          <Words engine={engine} result={layer1?.sets[0]} label="intact" dim />
          <h4>Control: random rows</h4>
          <Words engine={engine} result={layer1?.control} label="control" dim compact />
        </div>
      </section>
      <LoadNote engine={engine} dims={layer1?.dims} />
    </>
  )
}

function Row({ n, removed, toggle }: { n: number; removed: Set<number>; toggle: (c: number) => void }) {
  return (
    <>
      <div className="grid__label grid__label--row">{n}-word</div>
      {Array.from({ length: 8 }, (_, h) => {
        const c = (n - 2) * 8 + h
        const off = removed.has(c)
        const { n: size, head } = column(c)
        return (
          <button
            key={c}
            type="button"
            className={`trace${off ? ' trace--off' : ''}`}
            aria-pressed={!off}
            aria-label={`${size}-word trace, head ${head}: ${off ? 'removed' : 'kept'}`}
            onClick={() => toggle(c)}
          />
        )
      })}
    </>
  )
}
