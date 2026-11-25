// Hash routing (GitHub Pages has no server rewrites): #/view?param=value

import { useEffect, useState } from 'react'

export interface Route {
  view: string
  params: URLSearchParams
}

export function parseHash(hash: string): Route {
  const h = hash.replace(/^#\/?/, '')
  const [view, query = ''] = h.split('?')
  return { view: view || 'home', params: new URLSearchParams(query) }
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseHash(location.hash))
  useEffect(() => {
    const on = () => setRoute(parseHash(location.hash))
    addEventListener('hashchange', on)
    return () => removeEventListener('hashchange', on)
  }, [])
  return route
}

export function href(view: string, params?: Record<string, string | number | undefined>): string {
  const q = new URLSearchParams()
  for (const [k, v] of Object.entries(params ?? {})) if (v !== undefined && v !== '') q.set(k, String(v))
  const s = q.toString()
  return `#/${view}${s ? `?${s}` : ''}`
}

export function go(view: string, params?: Record<string, string | number | undefined>): void {
  location.hash = href(view, params)
}
