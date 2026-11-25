// Small SVG charts for the recorded results. Every chart has a text alternative (Req 7.5).

export function CurveChart({ k, r, ci, label }: { k: number[]; r: number[]; ci: number[][]; label: string }) {
  const W = 560
  const H = 260
  const P = { l: 44, r: 12, t: 12, b: 40 }
  const kmax = k[k.length - 1]
  const x = (v: number) => P.l + (v / kmax) * (W - P.l - P.r)
  const y = (v: number) => P.t + (1 - v) * (H - P.t - P.b)
  const line = k.map((kv, i) => `${i ? 'L' : 'M'}${x(kv)},${y(r[i])}`).join(' ')
  const band = [...k.map((kv, i) => `${i ? 'L' : 'M'}${x(kv)},${y(ci[i][1])}`), ...k.map((_, j) => k.length - 1 - j).map((i) => `L${x(k[i])},${y(ci[i][0])}`), 'Z'].join(' ')
  const straight = `M${x(0)},${y(r[0])} L${x(kmax)},${y(r[r.length - 1])}`
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line x1={P.l} x2={W - P.r} y1={y(v)} y2={y(v)} className="chart__grid" />
            <text x={P.l - 6} y={y(v) + 4} textAnchor="end" className="chart__tick">
              {Math.round(v * 100)}%
            </text>
          </g>
        ))}
        {k.map((kv) => (
          <text key={kv} x={x(kv)} y={H - P.b + 16} textAnchor="middle" className="chart__tick">
            {kv}
          </text>
        ))}
        <text x={(W + P.l) / 2} y={H - 6} textAnchor="middle" className="chart__axis">
          traces removed at random (of 24)
        </text>
        <path d={straight} className="chart__ref" />
        <path d={band} className="chart__band" />
        <path d={line} className="chart__line" />
        {k.map((kv, i) => (
          <circle key={kv} cx={x(kv)} cy={y(r[i])} r={3} className="chart__dot" />
        ))}
      </svg>
      <figcaption>{label}</figcaption>
    </figure>
  )
}

export interface Bar {
  label: string
  value: number
  ci?: number[]
  reference?: number
  highlight?: boolean
}

/** Horizontal bars (retention, 0–1) with an optional reference tick (chance at the same lesion size). */
export function Bars({ bars, label, refLabel }: { bars: Bar[]; label: string; refLabel?: string }) {
  return (
    <figure className="bars" aria-label={label}>
      <ul>
        {bars.map((b) => (
          <li key={b.label} className={b.highlight ? 'bars--hl' : undefined}>
            <span className="bars__label">{b.label}</span>
            <span className="bars__track">
              <span className="bars__fill" style={{ width: `${b.value * 100}%` }} />
              {b.reference !== undefined && <span className="bars__ref" style={{ left: `${b.reference * 100}%` }} title={refLabel} />}
            </span>
            <span className="bars__value">
              {Math.round(b.value * 100)}%{b.reference !== undefined && <span className="muted"> vs {Math.round(b.reference * 100)}%</span>}
            </span>
          </li>
        ))}
      </ul>
      <figcaption>{label}</figcaption>
    </figure>
  )
}
