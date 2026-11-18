---
inclusion: always
---

# Tech Stack and Commands

## Stack

| Part | Technology |
| --- | --- |
| Reference toolkit and asset build | Python 3.12, `uv`, numpy, `tokenizers`, `requests`, `sympy` (primes) |
| Web app | React, Vite, TypeScript; static site on GitHub Pages; no Engramscope server |
| Browser tokenizer | `@huggingface/transformers` `AutoTokenizer` on the model's `tokenizer.json` (Gate G2), else a WebAssembly build of `tokenizers` |
| Tests | `pytest` + `hypothesis` (Python); `vitest` (web); Playwright smoke test on a recorded fixture |
| Lint and format | `ruff check`, `ruff format` (Python); `eslint`, `prettier` (web) |
| CI | GitHub Actions: Python lint and tests; web lint, tests, build |

## Model pin

- `deepseek-ai/DeepSeek-V4.1-Flash` at revision `314113f8a7dadb388215396ba5fbee2c610a04d6` (MIT). Every model URL uses `resolve/<revision>/`, never `main`.
- Hub redirects are rate-limited (3,000 resolver requests per 300 s). Resolve each file's CDN URL once per session and fetch ranges from the CDN directly.

## Commands (once Task 2 scaffolding exists)

```sh
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest -q                                   # offline tests
ENGRAMSCOPE_TEST_NETWORK=1 uv run pytest -q         # also tests against the live model files
uv run python -m engramscope.build_assets           # writes web/public/assets/<revision>/
uv run python -m engramscope.bench                  # writes docs/results/
cd web && npm ci && npm run lint && npx vitest --run && npm run build
```

Never run watch modes (`vitest` without `--run`, `npm run dev`) in the Kiro sandbox.

## Environment facts

- The Kiro sandbox is Linux with no GPU and about 30 GB of RAM. Everything except browser checks and the video runs there.
- The builder's machines: Windows laptop (RTX 4060 8 GB, 16 GB RAM) and PC (RX 6650 XT 8 GB, 32 GB RAM). Neither runs the full model; Engramscope does not need them to.
- Large intermediate files (embedding tables as `.npy`, cached rows) never go into the repository.
