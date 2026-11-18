---
inclusion: always
---

# Repository Structure

```
Engramscope/
├── .kiro/
│   ├── specs/engramscope/   requirements.md, design.md, tasks.md
│   ├── steering/            product, tech, structure, workflow, research-integrity
│   └── hooks/               post-task checks (sh and PowerShell)
├── docs/
│   ├── findings/            kill test and gate results, dated
│   ├── results/             benchmark and decode-index measurements (generated)
│   ├── theory-sources.md    what Semon, Lashley and other sources actually say
│   ├── kiroween-rules.md    Devpost rules check
│   └── submission/          deliverables tracker, running Kiro write-up
├── spikes/engram_readout/   kill-test scripts (throwaway; reference for tools/)
├── tools/engramscope/       Python reference: hashing, fetch, fp8, readout, build_assets, bench
├── bench/phrases.jsonl      pre-registered benchmark phrases (committed before readouts)
├── web/                     React + Vite app; public/assets/<revision>/ holds built assets
├── tests/                   Python tests
├── README.md
└── LICENSE                  MIT
```

## Rules

- Python package code goes in `tools/engramscope/`, its tests in `tests/`.
- Web code goes in `web/src/`; shared test vectors are generated into `web/public/assets/<revision>/` by `build_assets.py`, never edited by hand.
- Generated results go in `docs/results/` with the commit, date and inputs recorded.
- Spikes stay in `spikes/` and are not imported by `tools/` or `web/`.
