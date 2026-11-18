# Implementation Plan: Engramscope

Ordered by dependency. Tasks marked **[HW]** need the builder's machine or browser. Everything else runs in the Kiro sandbox. Checkpoints stop for the builder's review.

Schedule (proposed): Task 1 by Nov 21, core app (Tasks 2–6) by Nov 28, polish and benchmark by Dec 1, video and submission by Dec 3. Deadline: December 4, 2026, 2:00 PM PT (December 5, 06:00 WITA).

## Pre-flight

- [x] P0. Kill test of the idea (done 2026-11-19 in the MaltOE repo; copied to `spikes/engram_readout/` and `docs/findings/engram-killtest.md`). _Requirements: 6_
- [x] P1. Builder: answer the open questions at the end of `requirements.md` (accepted the proposed answers, 2026-11-11).
- [ ] P2. Builder: set the repository's About description in the GitHub UI, and confirm GitHub shows the MIT license. _Requirements: 12.1_
- [ ] P3. Draft and send the trademark clarification email to Devpost (Kiro drafts, builder sends). _Requirements: 12.3_

## Tasks

- [ ] 1. Feasibility gates
  - [ ] 1.1 G1 decode index: build PCA-256 int8 embeddings and folded value matrices; measure top-10 overlap with the exact readout on 50 fixed phrases and the pilot score with both; record size. Try d = 512 or vocabulary restriction if needed. _Requirements: 5_
  - [ ] 1.2 G2 tokenizer parity: transformers.js `AutoTokenizer` against Python `tokenizers` on 1,000 strings (CJK, Indonesian, emoji, leading spaces). Fall back to a WebAssembly build of `tokenizers` if any id differs. _Requirements: 1.1_
  - [ ] 1.3 G3 browser fetch **[HW]**: a one-page prototype that resolves the CDN URLs and fetches 96 ranges; the builder opens it on their connection and reports the time. _Requirements: 2.4, 2.5_
  - [ ] 1.4 G4 benchmark set: write and commit at least 100 phrases with related-word sets before any readout (categories: places, people, events, science, culture, Indonesian topics, everyday things). _Requirements: 6.1, 6.2_
  - [ ] 1.5 G5 history sources: find and read primary or close-to-primary sources for Semon (1904) and Lashley (1950); write `docs/theory-sources.md` with what each says and where. _Requirements: 8.1_
  - [ ] 1.6 G6 prior-art re-check for direct Engram readouts and n-gram table interpretability. _Requirements: claims boundary_
  - [ ] 1.7 **Checkpoint:** the builder reviews gates G1–G6, confirms the decode index size budget and the latency target, and approves the build.

- [ ] 2. Scaffolding
  - [ ] 2.1 Python project with `uv`: `tools/engramscope/` package, `pyproject.toml` with ruff and pytest config, Python 3.12.
  - [ ] 2.2 Web app with Vite, React, TypeScript, vitest, eslint; GitHub Pages build.
  - [ ] 2.3 GitHub Actions CI: Python lint and tests; web lint, tests and build.
  - _Requirements: 11_

- [ ] 3. Reference implementation
  - [ ] 3.1 `hashing.py` with the Req 1.2 checks, ported from the spike.
  - [ ] 3.2 `fetch.py`, `fp8.py`, `readout.py`, including per-row contributions.
  - [ ] 3.3 Property tests 1–8 from design.
  - _Requirements: 1, 3.2, 11.1_

- [ ] 4. Asset build
  - [ ] 4.1 `build_assets.py`: constants JSON, token map, decode index (format from G1), test vectors, all under `web/public/assets/<revision>/`.
  - [ ] 4.2 Record the decode index's agreement numbers in `docs/results/`.
  - _Requirements: 1.4, 5_

- [ ] 5. Browser core
  - [ ] 5.1 Tokenizer (format from G2), BigInt hash, test-vector checks in vitest.
  - [ ] 5.2 Fetcher: CDN resolution once per shard, concurrency limit, 429 back-off, session cache.
  - [ ] 5.3 FP8 decode, projection, decoding against the index; per-row and per-n-gram contributions; control readout.
  - _Requirements: 1.3, 2, 3, 4.1_

- [ ] 6. Interface
  - [ ] 6.1 Landing and result views; bytes and timing shown.
  - [ ] 6.2 Inside view: pipeline and 24 row cards; position picker.
  - [ ] 6.3 About page: history, attribution, benchmark, limitations, sources.
  - [ ] 6.4 Visual design pass (Req 7.3), accessibility pass (Req 7.5), phone layout.
  - [ ] 6.5 **Checkpoint [HW]:** the builder uses the app in their browser and reviews.
  - _Requirements: 2, 3, 4, 7, 8_

- [ ] 7. Benchmark
  - [ ] 7.1 `bench.py` with all controls and bootstrap intervals; run on the committed phrase set.
  - [ ] 7.2 Write `docs/results/` and the README/About numbers from the output only.
  - [ ] 7.3 **Checkpoint:** the builder reviews the result; if layer 1 does not beat every control, the claims are revised (Req 6.5).
  - _Requirements: 6_

- [ ] 8. Stretch (only after Task 7)
  - [ ] 8.1 Lesion experiment on rented hardware (Req 9). Decide by November 30.
  - [ ] 8.2 Second model with an n-gram table (Req 10).
  - [ ] 8.3 Investigate the "Titanic" direction seen in the pilot (an attractor in layer-1 readouts).

- [ ] 9. Write-up and submission
  - [ ] 9.1 README final: problem, how it works, claims boundary, results, limitations, disclosures, how to run.
  - [ ] 9.2 Finish the Kiro write-up from the running draft.
  - [ ] 9.3 Video script and recording **[HW]**: under 3 minutes, on the builder's machine, no copyrighted music, no logos; YouTube public.
  - [ ] 9.4 Text description for the form.
  - [ ] 9.5 Final compliance check against `docs/submission/deliverables.md`; tag the submitted commit; submit before the deadline.
  - _Requirements: 12_
