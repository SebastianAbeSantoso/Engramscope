import { Bars, CurveChart } from '../components/Charts'
import results from '../data/results.json'
import { href } from '../lib/useHash'
import { MODEL_NAME } from '../lib/naming'

const L1 = results.lesion.L1
const L14 = results.lesion.L14
const pct = (x: number, d = 0) => `${(x * 100).toFixed(d)}%`
const ci = (c: number[], d = 1) => `[${pct(c[0], d)}, ${pct(c[1], d)}]`
const short = (s: string) => s.slice(0, 7)

export function Experiments() {
  const q1 = L1.Q1
  const q2 = L1.Q2
  const les = q2.lesions as Record<string, { retention: number; random_same_size: number; ci: number[]; special: boolean; direction: string | null }>
  const heads = Array.from({ length: 8 }, (_, h) => les[`B4(${h})`])
  const t = results.bench.table as Record<string, Record<string, { mean: number; ci95: number[] }>>
  const prov = results.lesion.provenance
  return (
    <div className="experiments">
      <h1>The experiments</h1>
      <p className="lede">
        Three questions from the search for the engram, rerun on {MODEL_NAME}'s Engram memory. The rules that turn numbers into words were committed before any run. Every figure on this page is read from the recorded results.
      </p>
      <p className="note">
        All results are <strong>{results.lesion.label}</strong>: traces are removed from the sum the model would add to its internal state, and the rest is decoded with the exact readout. They show how a concept is spread across a phrase's 24 traces, not what the running model forgets, and they say nothing about brains. Layer 1, {L1.n_phrases} pre-registered phrases, {L1.n_intact_hits} with a readable concept when intact.
      </p>

      <section aria-labelledby="inst-h">
        <h2 id="inst-h">0. Is the instrument reading anything?</h2>
        <p>
          Before any lesion: across the {results.bench.n_phrases} phrases, layer 1 has a related word in its top 10 for <strong>{pct(t['layer 1']['10'].mean, 1)}</strong> {ci(t['layer 1']['10'].ci95)}. It beats every control:
        </p>
        <Bars
          label="Benchmark hit@10 with 95% intervals: layer 1 against its controls"
          bars={[
            { label: 'Layer 1', value: t['layer 1']['10'].mean, highlight: true },
            { label: 'Last word’s own embedding', value: t['last-token embedding']['10'].mean },
            { label: 'Layer 14', value: t['layer 14']['10'].mean },
            { label: 'Random rows', value: t['random rows L1']['10'].mean },
            { label: 'Another phrase’s rows', value: t['shuffled']['10'].mean },
          ]}
        />
      </section>

      <section aria-labelledby="q1-h">
        <h2 id="q1-h">1. Mass action: does loss grow with the amount removed?</h2>
        <p className="source">Lashley 1950, Section VII (from p. 469): rats lost more of a maze habit the more cortex was removed.</p>
        <p className="verdict">“{q1.label}”</p>
        <CurveChart
          k={q1.k}
          r={q1.retention}
          ci={q1.retention_ci95}
          label={`Share of concepts still readable after removing k random traces, with 95% intervals; the dashed line is a straight-line decline. Half are gone at ${q1.k50.toFixed(1)} traces ${`[${q1.k50_ci95[0].toFixed(1)}, ${q1.k50_ci95[1].toFixed(1)}]`}.`}
        />
        <p>
          Removing 8 random traces of 24 leaves the concept in {pct(q1.retention[q1.k.indexOf(8)], 1)} of phrases, 16 leaves {pct(q1.retention[q1.k.indexOf(16)], 1)}, 23 leaves {pct(q1.retention[q1.k.indexOf(23)], 1)}. Half the concepts are gone only when <strong>{q1.k50.toFixed(1)}</strong> traces are removed. Shape statistic S = {q1.S.toFixed(2)} [{q1.S_ci95[0].toFixed(2)}, {q1.S_ci95[1].toFixed(2)}] (0 = straight line, +0.5 = holds until the end).
        </p>
      </section>

      <section aria-labelledby="q2-h">
        <h2 id="q2-h">2. Equipotentiality: does it matter which traces go?</h2>
        <p className="source">Lashley 1950, p. 469: the parts of a functional area are “equipotential” for a memory trace.</p>
        <p className="verdict">“{q2.heads_label}”</p>
        <Bars
          label="Concept still readable after removing one hash head (3 traces); the tick is 3 random traces"
          refLabel="3 random traces"
          bars={heads.map((b, h) => ({ label: `head ${h}`, value: b.retention, reference: b.random_same_size }))}
        />
        <p className="verdict">“not equipotential across n-gram sizes”</p>
        <Bars
          label="Concept still readable after removing all 8 traces of one n-gram size; the tick is 8 random traces"
          refLabel="8 random traces"
          bars={[2, 3, 4].map((n) => ({ label: `${n}-word traces`, value: les[`B3(${n})`].retention, reference: les[`B3(${n})`].random_same_size, highlight: les[`B3(${n})`].special }))}
        />
        <p>
          Removing the 8 two-word traces leaves the concept in <strong>{pct(les['B3(2)'].retention, 1)}</strong> of phrases, against {pct(les['B3(2)'].random_same_size, 1)} for 8 random traces. Removing the four-word traces is {les['B3(4)'].direction}. A single trace is critical in {pct(q2.B1.share_single_point, 1)} of phrases; on their own, an average of {q2.B2_mean_single_traces_that_keep_hit.toFixed(1)} of the 24 traces keep the concept.
        </p>
      </section>

      <section aria-labelledby="q3-h">
        <h2 id="q3-h">3. Semon's partial cue: can part of the stimulus retrieve the whole?</h2>
        <p className="source">Semon 1921, p. 138: ecphory, “the passage of an engram from a latent to a manifest state”, which “may be released by the partial recurrence” of the original stimulus.</p>
        <Bars
          label="Concept retrieved from one n-gram size alone"
          bars={(['2', '3', '4'] as const).map((n) => ({ label: `${n}-word traces only`, value: L1.Q3a.sizes[n].retention, highlight: L1.Q3a.sizes[n].retrieves_above_chance }))}
        />
        <Bars
          label="Concept still readable when the phrase follows other words (no label, by design)"
          bars={Object.entries(L1.Q3b).map(([ctx, v]) => ({ label: `“${ctx} …”`, value: (v as { retention: number }).retention }))}
        />
        <p>
          In context, only {pct(L1.Q3b['I read about'].share_of_traces_recurring_by_size['4'])} of the four-word traces and {pct(L1.Q3b['I read about'].share_of_traces_recurring_by_size['3'])} of the three-word traces stay the same; {pct(L1.Q3b['I read about'].share_of_traces_recurring_by_size['2'])} of the two-word traces recur.
        </p>
      </section>

      <section aria-labelledby="lashley-h">
        <h2 id="lashley-h">What the rules produce</h2>
        <ul className="checks">
          <li>Prediction 1, no head is special: {L1.predictions.P1_no_head_lesion_special ? 'held' : 'did not hold'}.</li>
          <li>Prediction 2, some n-gram size is special: {L1.predictions.P2_some_size_lesion_special ? 'held' : 'did not hold'}.</li>
        </ul>
        <p>
          Both labels together parallel Lashley's conclusion (2) (1950, p. 478): limited regions may be essential, but “within such regions the parts are functionally equivalent.” Here the regions are the n-gram sizes and the parts are the hash heads.
        </p>
        <h3>Layer 14</h3>
        <p>
          Only {L14.n_intact_hits} phrases have a readable concept in layer 14, below the power guard of 30, so it gets numbers and no labels: removing 8 random traces leaves {pct(L14.Q1.retention[L14.Q1.k.indexOf(8)])} of its {L14.n_intact_hits} concepts.
        </p>
        <h3>Caveats</h3>
        <ul>
          <li>Short phrases are padded before their start, which favours the two-word traces. In phrases of 4 or more tokens, which have no padding, the two-word lesion still costs the most.</li>
          <li>Single words have no readable trace: {L1.secondary_by_length['1'].n_intact_hits} of the {L1.secondary_by_length['1'].n_phrases} single-token phrases have one (exploratory, not pre-registered).</li>
          <li>Readout, not behaviour: the model also uses the rows in its gate, and later layers can compensate.</li>
        </ul>
        <p className="provenance">
          Recorded results: <code>{results.files.lesion}</code> and <code>{results.files.bench}</code>, code commit {short(prov.code_commit)}, run {prov.run_at.slice(0, 10)}. Protocol registered at {short(prov.registered.protocol.commit)}, phrases at {short(prov.registered.phrases.commit)}, both before the run.{' '}
          <a href="https://github.com/SebastianAbeSantoso/Engramscope/blob/main/bench/lesion-protocol.md">Protocol</a> ·{' '}
          <a href="https://github.com/SebastianAbeSantoso/Engramscope/tree/main/docs/results">raw outputs</a> · <a href={href('lesion')}>try it live</a>
        </p>
      </section>
    </div>
  )
}
