import results from '../data/results.json'
import { MODEL_NAME, MODEL_OWNER } from '../lib/naming'

const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`

export function About() {
  const t = results.bench.table as Record<string, Record<string, { mean: number; ci95: number[] }>>
  const di = results.decodeIndex
  const side = results.sideTest
  const rows: [string, string][] = [
    ['layer 1', 'Engram layer 1'],
    ['layer 14', 'Engram layer 14'],
    ['random rows L1', 'random rows (layer 1)'],
    ['shuffled', 'another phrase’s rows'],
    ['last-token embedding', 'the last word’s own embedding'],
  ]
  return (
    <div className="about">
      <h1>About Engramscope</h1>

      <section>
        <h2>The question</h2>
        <p>
          Richard Semon coined <em>engram</em> in 1904 for the lasting change a stimulus writes into living substance. He called its retrieval <em>ecphory</em>: an engram becoming active again when part of the original stimulus recurs (<cite>The Mneme</cite>, 1921, p. 138). Karl Lashley spent decades looking for engrams in rats' cortex. In 1950 he concluded that “it is not possible to demonstrate the isolated localization of a memory trace anywhere within the nervous system”: the loss depended on how much was removed, and within a functional area the parts were equivalent (“In Search of the Engram”, 1950, pp. 469, 478).
        </p>
        <p>
          {MODEL_NAME} has two lookup tables its developers named Engram, at layers 1 and 14, about 384 million rows each. At every position the model fetches 24 rows per table: one per n-gram size (2, 3, 4 tokens) and hash head (8). Every trace has an address computable from the text, which was never possible in a brain. So Lashley's experiments can be rerun exactly.
        </p>
      </section>

      <section>
        <h2>What is whose</h2>
        <ul>
          <li>
            <strong>{MODEL_OWNER}'s:</strong> Engram itself, its hashing and its tables (Cheng et al., <a href="https://arxiv.org/abs/2601.07372">arXiv 2601.07372</a>; the model's reference code). MIT license.
          </li>
          <li>
            <strong>Lashley's and Semon's:</strong> the questions and the lesion method (1950; 1904/1921).
          </li>
          <li>
            <strong>Engramscope's:</strong> rerunning those experiments as pre-registered trace lesions on a released model's Engram memory, and the instrument they need: reading the rows directly, without running the model, over HTTP from a browser, with controls.
          </li>
        </ul>
        <p>Nothing here is a claim about biological memory. The experiments borrow Lashley's and Semon's questions and methods, not their conclusions.</p>
      </section>

      <section>
        <h2>How the instrument works</h2>
        <ol>
          <li>Your browser tokenizes the phrase with the model's own tokenizer and computes the model's 24 addresses per layer, bit for bit.</li>
          <li>It fetches those rows straight from the public model files with HTTP range requests: about 12 KB of a 189 GiB pair of tables. There is no Engramscope server.</li>
          <li>It projects the rows the way the model does and decodes the result into words by similarity with the model's input embeddings. That decoding is an approximation: it assumes the value vector lives in embedding space, which is plausible at layer 1 and not proven.</li>
        </ol>
      </section>

      <section>
        <h2>Benchmark (pre-registered)</h2>
        <p>
          {results.bench.n_phrases} phrases with related-word sets, committed before any readout of them. A hit means a related word, not contained in the phrase, appears in the top 10. Exact readout, Python.
        </p>
        <table>
          <thead>
            <tr>
              <th scope="col">Readout</th>
              <th scope="col">hit@1</th>
              <th scope="col">hit@5</th>
              <th scope="col">hit@10 (95% interval)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([key, label]) => (
              <tr key={key}>
                <th scope="row">{label}</th>
                <td>{pct(t[key]['1'].mean)}</td>
                <td>{pct(t[key]['5'].mean)}</td>
                <td>
                  {pct(t[key]['10'].mean)} [{pct(t[key]['10'].ci95[0])}, {pct(t[key]['10'].ci95[1])}]
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p>Layer 1 beats every control: {results.bench.layer1_beats_every_control ? 'yes' : 'no'}.</p>
      </section>

      {di && (
        <section>
          <h2>The browser's decoding is approximate</h2>
          <p>
            The exact readout needs 1.3 GB of embeddings. The browser uses a compressed index instead: {di.meta.d} dimensions at 4 bits, about 92 MB, downloaded once and cached. It loads in {di.meta.chunks} parts and decodes from the first. Measured against the exact readout (<code>{results.files.decodeIndex}</code>):
          </p>
          <table>
            <thead>
              <tr>
                <th scope="col">Layer, dimensions loaded</th>
                <th scope="col">Top-10 words shared with the exact readout</th>
                <th scope="col">Benchmark hit@10 (exact: {pct(di.layers['1'].bench_hit10.exact)})</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(di.layers['1'].dev).map(([dims, v]) => (
                <tr key={dims}>
                  <th scope="row">layer 1, {dims}</th>
                  <td>{pct(v.top10_overlap, 0)}</td>
                  <td>{pct((di.layers['1'].bench_hit10 as Record<string, number>)[dims])}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p>Layer 14 shares only {pct(di.layers['14'].dev['1024'].top10_overlap, 0)} of its words with the exact readout, which itself reads as noise for layer 14. Every number on the Experiments page uses the exact readout.</p>
        </section>
      )}

      <section>
        <h2>A side test: does this memory design help another model?</h2>
        <p>
          We also added an Engram-style memory, with the same hashing, to frozen SmolLM2 models and compared it with LoRA, under rules fixed before each run. At 135M parameters LoRA won (the pre-registered outcome: kill). At 1.7B the memory narrowly passed the rule ({side.summary.mem_val_ppl.toFixed(3)} vs {side.summary.lora_val_ppl.toFixed(3)} perplexity), but a LoRA with a lower learning rate, run afterwards as an exploratory check, beat it clearly ({side.extra.extra_lora_val_ppl.toFixed(3)}). Adding Engram-style memory to a frozen model is not new: Engram Adapter, TF-Engram, NGM, Engram-PEFT and TinyEngram do it. Details in <a href="https://github.com/SebastianAbeSantoso/Engramscope/blob/main/spikes/own_memory/RESULTS.md">the side test's results</a>.
        </p>
      </section>

      <section>
        <h2>Limitations</h2>
        <ul>
          <li>Trace lesions act on the readout, not the running model. The model also uses the rows in its gate, which depends on its internal state; later layers may compensate.</li>
          <li>The gate is not computed, so decoded words are what the rows point at, not necessarily what the model used at full strength.</li>
          <li>Single words have no readable trace in this memory.</li>
          <li>The browser's decoding is approximate (above).</li>
        </ul>
      </section>

      <section>
        <h2>Sources and credits</h2>
        <ul>
          <li>Semon, R. (1904). <cite>Die Mneme</cite>. Leipzig: Engelmann. English: <cite>The Mneme</cite> (1921), trans. L. Simon. London: Allen &amp; Unwin.</li>
          <li>Lashley, K. S. (1950). In search of the engram. <cite>Symposia of the Society for Experimental Biology</cite> 4, 454–482.</li>
          <li>Cheng et al. (2026). Conditional Memory via Scalable Lookup. <a href="https://arxiv.org/abs/2601.07372">arXiv 2601.07372</a>.</li>
          <li>Model files: {MODEL_NAME} (MIT), read at a pinned revision. Tokenizer: @huggingface/tokenizers.</li>
          <li>Related work on Engram tables: EngramEdit, CP and GateRQ (compression), remote and on-demand row access for inference. Cited, not claimed.</li>
          <li>
            Code: <a href="https://github.com/SebastianAbeSantoso/Engramscope">github.com/SebastianAbeSantoso/Engramscope</a>, MIT license. Built with Kiro for Kiroween 2026.
          </li>
        </ul>
      </section>
    </div>
  )
}
