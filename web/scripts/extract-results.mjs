// Reads the recorded results in docs/results/ and writes the parts the app shows to
// src/data/results.json (Req 9.7: the app's figures come from the raw outputs, never typed by hand).
// Runs before dev, build and test.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const results = join(here, '../../docs/results')
const read = (name) => JSON.parse(readFileSync(join(results, name), 'utf8'))
const latest = (prefix) =>
  readdirSync(results)
    .filter((f) => f.startsWith(prefix) && f.endsWith('.json') && /\d{4}-\d{2}-\d{2}\.json$/.test(f))
    .sort()
    .at(-1)

const lesionFile = latest('lesion-')
const benchFile = latest('bench-')
const indexFile = latest('decode-index-')
const lesion = read(lesionFile)
const bench = read(benchFile)

const layer = (L) => {
  const { per_phrase: _drop, ...rest } = lesion.layers[L]
  return rest
}

const out = {
  files: { lesion: lesionFile, bench: benchFile, decodeIndex: indexFile ?? null },
  lesion: { provenance: lesion.provenance, label: lesion.label, L1: layer('L1'), L14: layer('L14') },
  bench: {
    provenance: bench.provenance,
    n_phrases: bench.n_phrases,
    table: bench.table,
    beats: bench.beats,
    layer1_beats_every_control: bench.layer1_beats_every_control,
  },
  decodeIndex: indexFile ? read(indexFile) : null,
  sideTest: (() => {
    const dir = join(here, '../../spikes/own_memory/results/scale')
    const j = (n) => JSON.parse(readFileSync(join(dir, n), 'utf8'))
    const { lines: _a, ...summary } = j('summary.json')
    const { lines: _b, ...extra } = j('extra.json')
    return { summary, extra }
  })(),
}

mkdirSync(join(here, '../src/data'), { recursive: true })
writeFileSync(join(here, '../src/data/results.json'), JSON.stringify(out))
console.log(`results.json from ${lesionFile}, ${benchFile}, ${indexFile}`)
