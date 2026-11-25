import { useMemo } from 'react'

import { formatBytes, LoadNote, Words } from '../components/Words'
import { PhraseForm } from '../components/PhraseForm'
import { column } from '../core/constants'
import { ALL_COLUMNS, columnsOfSize, type Engine, type Lookup } from '../core/engine'
import { useEngine } from '../lib/EngineContext'
import { go, href } from '../lib/useHash'
import { useReadout } from '../lib/useReadout'
import results from '../data/results.json'

const pct = (x: number) => `${(x * 100).toFixed(1)}%`
const HIT10 = (name: 'layer 1' | 'layer 14') => results.bench.table[name]['10'].mean

const SIZE_SETS = [columnsOfSize(2), columnsOfSize(3), columnsOfSize(4)]
const SINGLE_SETS = ALL_COLUMNS.map((c) => [c])

export function Read({ params }: { params: URLSearchParams }) {
  const phrase = params.get('p') ?? ''
  const posParam = params.get('pos')
  const position = posParam === null ? undefined : Number(posParam)
  const { engine, error } = useEngine()
  return (
    <div className="read">
      <h1>Read a memory</h1>
      <PhraseForm initial={phrase} onSubmit={(p) => go('read', { p })} />
      {error && <p className="error">{error}</p>}
      {!engine && !error && <p className="note">Loading the tokenizer and addresses…</p>}
      {engine && phrase && <ReadResult engine={engine} phrase={phrase} position={position} />}
    </div>
  )
}

function ReadResult({ engine, phrase, position }: { engine: Engine; phrase: string; position?: number }) {
  const sets = useMemo(() => [ALL_COLUMNS, ...SIZE_SETS, ...SINGLE_SETS], [])
  const r = useReadout(engine, phrase, position, sets)
  if (r.error) return <p className="error">{r.error}</p>
  if (!r.lookup) return <p className="note">Fetching traces…</p>
  const l = r.lookup
  const layer1 = r.layers.find((x) => x.layer === 1)
  const layer14 = r.layers.find((x) => x.layer === 14)
  return (
    <>
      <Positions engine={engine} lookup={l} />
      <section className="readout" aria-labelledby="readout-h">
        <h2 id="readout-h">What the traces point at</h2>
        <div className="readout__grid">
          <div className="panel panel--main">
            <h3>Layer 1</h3>
            <Words engine={engine} result={layer1?.sets[0]} label="layer 1" />
            <h4>Control: 24 random rows of the same table</h4>
            <Words engine={engine} result={layer1?.control} label="control" dim />
          </div>
          <div className="panel">
            <h3>
              Layer 14 <span className="tag">unclear</span>
            </h3>
            <Words engine={engine} result={layer14?.sets[0]} label="layer 14" dim />
            <p className="note">Layer 14 reads as noise with this decoding: {pct(HIT10('layer 14'))} on the benchmark, against {pct(HIT10('layer 1'))} for layer 1.</p>
            <h4>Control</h4>
            <Words engine={engine} result={layer14?.control} label="layer 14 control" dim />
          </div>
        </div>
        <LoadNote engine={engine} dims={layer1?.dims} />
        <p className="stats">
          Fetched {formatBytes(l.stats.bytes)} of {formatBytes(l.tableBytes)} in {l.stats.requests} range requests, {Math.round(l.stats.ms)} ms (traces and control; repeat lookups come from memory). Decoded in {Math.round(r.decodeMs)} ms.
        </p>
        <p>
          <a className="button button--ghost" href={href('lesion', { p: phrase, pos: position })}>
            Remove some of these traces
          </a>
        </p>
      </section>

      <section aria-labelledby="inside-h">
        <h2 id="inside-h">Inside the lookup</h2>
        <ol className="pipeline">
          <li>
            <strong>Tokens</strong> → <strong>compressed ids</strong>: tokens that differ only in case, accents or spacing share an id ({engine.c.compressed_vocab_size.toLocaleString()} ids).
          </li>
          <li>
            <strong>Windows</strong>: at the chosen position, the last 2, 3 and 4 compressed ids (padded before the start).
          </li>
          <li>
            <strong>24 addresses per layer</strong>: each window is hashed by 8 heads, each with its own prime-sized range of the table.
          </li>
          <li>
            <strong>Rows</strong>: 256 FP8 numbers each, fetched by byte range from the public model files.
          </li>
          <li>
            <strong>Value vector</strong>: the sum of 24 per-row contributions. The model scales it by a gate that depends on its internal state, which Engramscope does not compute, so these are the words the rows point at, not necessarily what the model used at full strength.
          </li>
        </ol>
        <h3>By n-gram size (layer 1)</h3>
        <div className="sizes">
          {SIZE_SETS.map((_, i) => (
            <div key={i} className="panel">
              <h4>{i + 2}-word traces</h4>
              <Words engine={engine} result={layer1?.sets[1 + i]} compact />
            </div>
          ))}
        </div>
        <h3>The 24 traces (layer 1)</h3>
        <p className="note">Each trace alone, decoded. On its own a trace is mostly noise: the concept comes from their sum.</p>
        <div className="cards">
          {ALL_COLUMNS.map((c) => {
            const { n, head } = column(c)
            const addr = l.layers[0].addresses[c]
            const range = engine.fetcher.rowRanges(1, addr)
            return (
              <div key={c} className="card">
                <div className="card__head">
                  {n}-word · head {head}
                </div>
                <div className="card__addr">row {addr.toLocaleString()}</div>
                <div className="card__addr">byte {range.values[0].toLocaleString()}</div>
                <Words engine={engine} result={layer1?.sets[4 + c] ? { ...layer1.sets[4 + c]!, ids: layer1.sets[4 + c]!.ids.slice(0, 4) } : undefined} compact dim />
              </div>
            )
          })}
        </div>
      </section>
    </>
  )
}

function Positions({ engine, lookup }: { engine: Engine; lookup: Lookup }) {
  return (
    <section aria-labelledby="tokens-h">
      <h2 id="tokens-h">Tokens</h2>
      <p className="note">The memory is read at one position: by default the last token. Pick another to read there.</p>
      <div className="tokens" role="list">
        {lookup.tokens.map((t, i) => (
          <a
            key={i}
            role="listitem"
            className={`token${i === lookup.position ? ' token--on' : ''}`}
            href={href('read', { p: lookup.phrase, pos: i === lookup.tokens.length - 1 ? undefined : i })}
            aria-current={i === lookup.position ? 'true' : undefined}
          >
            <span className="token__text">{t.text.replace(/ /g, '·') || '∅'}</span>
            <span className="token__id">id {t.id}</span>
            <span className="token__id">c {t.compressed}</span>
          </a>
        ))}
      </div>
      <p className="note">{engine.hasher.windows(lookup.tokens.map((t) => t.id), lookup.position).map((w, i) => `${i + 2}-word window: [${w.join(', ')}]`).join(' · ')}</p>
    </section>
  )
}
