import { PhraseForm } from '../components/PhraseForm'
import { go, href } from '../lib/useHash'
import { MODEL_NAME, MODEL_OWNER } from '../lib/naming'
import results from '../data/results.json'

export function Home() {
  const L1 = results.lesion.L1
  const k50 = Math.round(L1.Q1.k50)
  return (
    <div className="home">
      <section className="hero">
        <p className="eyebrow">Lashley's search for the engram, rerun</p>
        <h1>Where is a memory stored?</h1>
        <p className="lede">
          Karl Lashley spent about 30 years removing parts of rats' cortex to find where a learned habit was kept. In 1950 he wrote that it is not possible to demonstrate "the isolated localization of a memory trace anywhere within the nervous system."
        </p>
        <p className="lede">
          {MODEL_NAME} has a memory where every trace has an address. Its <em>Engram</em> tables, built by {MODEL_OWNER}, hold rows that the model looks up from the last few words of the text: 24 traces per phrase. Engramscope finds those traces, reads them in your browser, and removes them the way Lashley removed cortex.
        </p>
      </section>

      <section className="finding" aria-labelledby="finding-h">
        <h2 id="finding-h">What the pre-registered experiments found</h2>
        <p>
          We reran Karl Lashley's lesion experiments on a language model's memory, pre-registered. Each phrase is stored in 24 traces. Half of the stored concepts are still readable after {k50} of the 24 are removed at random. Any of the 8 hash heads can stand in for any other, but the two-word traces cannot be replaced: remove those 8 and the concept is gone in 9 of 10 phrases. That is the pattern Lashley described in rat cortex in 1950: some regions are essential, and within a region the parts are interchangeable. It describes this model's memory, not brains.
        </p>
        <p>
          <a className="button" href={href('experiments')}>
            See the experiments
          </a>{' '}
          <a className="button button--ghost" href={href('lesion', { p: 'the Eiffel Tower' })}>
            Remove traces yourself
          </a>
        </p>
      </section>

      <section aria-labelledby="try-h">
        <h2 id="try-h">Read a memory</h2>
        <p>Type a name or a set phrase. The browser computes the model's own addresses, fetches about 12 KB from the public model files, and decodes it. Nothing runs the model.</p>
        <PhraseForm initial="" onSubmit={(p) => go('read', { p })} />
        <p className="note">Single words have no readable trace in this memory; multi-word names work best.</p>
      </section>
    </div>
  )
}
