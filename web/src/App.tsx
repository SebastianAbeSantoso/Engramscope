import { EngineProvider, useEngine } from './lib/EngineContext'
import { href, useRoute } from './lib/useHash'
import { About } from './views/About'
import { Experiments } from './views/Experiments'
import { Home } from './views/Home'
import { Lesion } from './views/Lesion'
import { Read } from './views/Read'

const NAV = [
  ['home', 'Engramscope'],
  ['experiments', 'Experiments'],
  ['lesion', 'Live lesion'],
  ['read', 'Read a memory'],
  ['about', 'About'],
] as const

// Views that need the instrument (tokenizer, hash, fetcher, decode index); the others are static.
const LIVE = new Set(['home', 'lesion', 'read'])

export default function App() {
  const route = useRoute()
  return (
    <EngineProvider start={LIVE.has(route.view)}>
      <a className="skip" href="#main" onClick={(e) => (e.preventDefault(), document.getElementById('main')?.focus())}>
        Skip to content
      </a>
      <header className="top">
        <nav aria-label="Main">
          {NAV.map(([view, label]) => (
            <a key={view} href={href(view)} className={route.view === view ? 'active' : undefined} aria-current={route.view === view ? 'page' : undefined}>
              {label}
            </a>
          ))}
        </nav>
        <LoadBadge />
      </header>
      <main id="main" tabIndex={-1}>
        {route.view === 'experiments' ? (
          <Experiments />
        ) : route.view === 'lesion' ? (
          <Lesion params={route.params} />
        ) : route.view === 'read' ? (
          <Read params={route.params} />
        ) : route.view === 'about' ? (
          <About />
        ) : (
          <Home />
        )}
      </main>
      <footer className="foot">
        <p>
          A Kiroween 2026 project. Engram, its hashing and its tables are the model developers' work; Engramscope reads them. MIT license ·{' '}
          <a href="https://github.com/SebastianAbeSantoso/Engramscope">source</a>
        </p>
      </footer>
    </EngineProvider>
  )
}

function LoadBadge() {
  const { engine, progress, error } = useEngine()
  if (error) return <span className="badge badge--error">Couldn't load: {error}</span>
  if (!progress) return null
  if (progress.stage === 'basics') return <span className="badge">Loading the tokenizer…</span>
  if (engine && progress.chunks >= progress.total) return null
  const pct = progress.totalBytes ? Math.round((progress.bytes / progress.totalBytes) * 100) : 0
  return (
    <span className="badge" role="status">
      Loading the decoding index: {pct}% ({progress.chunks}/{progress.total} parts)
    </span>
  )
}
