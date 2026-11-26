# Implementation Plan: Engramscope

Ordered by dependency. Tasks marked **[HW]** need the builder's machine or browser. Everything else runs in the Kiro sandbox. Checkpoints stop for the builder's review.

Schedule (proposed): Task 1 by Nov 21; Python reference, benchmark and lesion run (Tasks 2–4, 7, 8.1–8.2) by Nov 25; app with experiment and lesion views (Tasks 5–6, 8.3) by Nov 30; polish by Dec 1; video and submission by Dec 3. Deadline: December 4, 2026, 2:00 PM PT (December 5, 06:00 WITA).

**Status (2026-11-25):** Task 1, the Python reference, the benchmark, the lesion run and the own-memory side test (8c) are done. The web app (Tasks 2.2–6, 8.3) hasn't started; the time went into the side test. Revised plan: app with experiment and lesion views by Dec 1; write-ups Dec 2; video and submission by Dec 3.

## Pre-flight

- [x] P0. Kill test of the idea (done 2026-11-19 in the MaltOE repo; copied to `spikes/engram_readout/` and `docs/findings/engram-killtest.md`). _Requirements: 6_
- [x] P1. Builder: answer the open questions at the end of `requirements.md` (answered 2026-11-19; see Decisions in `requirements.md`).
- [ ] P2. Builder: set the repository's About description in the GitHub UI, and confirm GitHub shows the MIT license. _Requirements: 12.1_
- [x] P3. Trademark approach decided without emailing Devpost: plain-text name in README, app and written submission; generic wording in the video (Decision 6). _Requirements: 12.3_

## Tasks

- [x] 1. Feasibility gates
  - [x] 1.1 G1 decode index: build PCA-256 int8 embeddings and folded value matrices; measure top-10 overlap with the exact readout on 50 fixed phrases and the pilot score with both; record size. Try d = 512 or vocabulary restriction if needed. _Requirements: 5_ **Done:** measured, no small index matches the exact readout well; best per size is fitted d=1024 int4 (~92 MB total, 0.60 top-10 overlap on layer 1). Options in `docs/findings/g1-decode-index.md`; the builder picks at 1.7.
  - [x] 1.2 G2 tokenizer parity: transformers.js `AutoTokenizer` against Python `tokenizers` on 1,000 strings (CJK, Indonesian, emoji, leading spaces). Fall back to a WebAssembly build of `tokenizers` if any id differs. _Requirements: 1.1_ **Done:** 0 mismatches of 1,000 for `@huggingface/tokenizers` 0.2.0 and transformers.js 4.3.2; use the former (`docs/findings/g2-tokenizer.md`).
  - [x] 1.3 G3 browser fetch **[HW]**: a one-page prototype that resolves the CDN URLs and fetches 96 ranges; the builder opens it on their connection and reports the time. _Requirements: 2.4, 2.5_ **Done:** on the builder's PC, 1.07–1.09 s per phrase at concurrency 32 (cold), 0.54 s warm; all bytes verified (`docs/findings/g3-browser-fetch.md`).
  - [x] 1.4 G4 benchmark set: write and commit at least 100 phrases with related-word sets before any readout (categories: places, people, events, science, culture, Indonesian topics, everyday things). _Requirements: 6.1, 6.2_ **Done:** 123 phrases committed in ec0e429 before any readout (`bench/README.md`).
  - [x] 1.5 G5 history sources: find and read primary or close-to-primary sources for Semon (1904) and Lashley (1950); write `docs/theory-sources.md` with what each says and where. _Requirements: 8.1_ **Done:** Semon 1904 (German first edition) and 1921 (English) and Lashley 1950 read, with page numbers, in `docs/theory-sources.md`. Lashley pp. 476–477 are missing from the scan used.
  - [x] 1.6 G6 prior-art re-check for direct Engram readouts and n-gram table interpretability. _Requirements: claims boundary_ **Done:** no prior tool or study decodes row contents (`docs/findings/g6-prior-art.md`); a newer checkpoint `-1017` exists.
  - [x] 1.6b Write and commit the lesion protocol (`bench/lesion-protocol.md`) before any lesion run. _Requirements: 9.1_
  - [x] 1.7 **Checkpoint:** the builder reviews gates G1–G6, confirms the decode index size budget and the latency target, **approves the lesion protocol's interpretation rules (Section 6)**, and approves the build. **Done 2026-11-19:** decode index fitted d=1024 int4, about 92 MB (builder); rules delegated to Kiro and finalized before any run; latency target proposed (Decision 10), to be confirmed at 6.6.

- [ ] 2. Scaffolding
  - [x] 2.1 Python project with `uv`: `tools/engramscope/` package, `pyproject.toml` with ruff and pytest config, Python 3.12.
  - [x] 2.2 Web app with Vite, React, TypeScript, vitest, oxlint; GitHub Pages build.
  - [x] 2.3 GitHub Actions CI: Python lint and tests; web lint, tests and build (`ci.yml`); Pages deploy with the asset build (`pages.yml`).
  - _Requirements: 11_

- [x] 3. Reference implementation
  - [x] 3.1 `hashing.py` with the Req 1.2 checks, ported from the spike.
  - [x] 3.2 `hub.py` (fetch), `fp8.py`, `readout.py`, including per-row contributions.
  - [x] 3.3 Property tests 1–8 and 10 from design (26 tests incl. network; `ENGRAMSCOPE_TEST_NETWORK=1 uv run pytest`).
  - _Requirements: 1, 3.2, 11.1_

- [ ] 4. Asset build
  - [x] 4.1 `build_assets.py`: constants JSON, token map, decode index (format from G1), test vectors, all under `web/public/assets/<revision>/`. One projection fitted on layer 1 and shared by both layers, so the index stays at about 92 MB.
  - [x] 4.2 Record the decode index's agreement numbers in `docs/results/` (`decode-index-2026-11-25.md`: layer 1 top-10 overlap 0.568; benchmark hit@10 0.618, equal to exact, from 512 dimensions on).
  - _Requirements: 1.4, 5_

- [ ] 5. Browser core
  - [x] 5.1 Tokenizer (format from G2), BigInt hash, test-vector checks in vitest (58 phrases: ids and every address exact).
  - [x] 5.2 Fetcher: CDN resolution once per shard, concurrency limit, 429 back-off, session cache.
  - [x] 5.3 FP8 decode, projection, decoding against the index (in a Web Worker); per-row and per-n-gram contributions; control readout. Matches the Python compressed readout on the test vectors.
  - _Requirements: 1.3, 2, 3, 4.1_

- [ ] 6. Interface
  - [x] 6.1 Landing and result views; bytes and timing shown.
  - [x] 6.2 Inside view: pipeline and 24 row cards; position picker.
  - [x] 6.3 `?generic` URL flag that replaces the model name with "an open-weight frontier model" (Req 12.3).
  - [x] 6.4 About page, framed around Lashley's question; the read views are the instrument: history, attribution, benchmark, limitations, sources.
  - [ ] 6.5 Visual design pass (Req 7.3), accessibility pass (Req 7.5), phone layout.
  - [ ] 6.6 **Checkpoint [HW]:** the builder uses the app in their browser and reviews.
  - _Requirements: 2, 3, 4, 7, 8_

- [ ] 7. Benchmark
  - [x] 7.1 `bench.py` with all controls and bootstrap intervals; run on the committed phrase set. **Done:** layer 1 hit@10 61.8% [53.7, 69.9], beats every control (`docs/results/bench-2026-11-19.md`).
  - [ ] 7.2 Write `docs/results/` and the README/About numbers from the output only.
  - [x] 7.3 **Checkpoint:** the builder reviews the result; if layer 1 does not beat every control, the claims are revised (Req 6.5). **Approved by the builder 2026-11-25** (`docs/findings/experiments-2026-11-19.md`; layer 1 beats every control).
  - _Requirements: 6_

- [ ] 8. Lesion experiments (core; after Task 7, which supplies the chance level)
  - [x] 8.1 `lesion.py` per the approved protocol, with a test that its conditions match the protocol table, and the git-hash guard.
  - [x] 8.2 Run on layers 1 and 14; write `docs/results/lesion-<date>.{json,md}` from the output only, with the rule outcomes applied mechanically. **Done:** `docs/results/lesion-2026-11-19.md`; summary for review in `docs/findings/experiments-2026-11-19.md`.
  - [x] 8.3 Experiments view in the app (reads the recorded results) and live lesion view (3 × 8 trace grid, presets for the protocol's conditions, labelled as exploration).
  - [x] 8.4 **Checkpoint:** the builder reviews the results and the wording that follows from the rules before it goes into the README, app or video. **Approved by the builder 2026-11-25**, including the draft README paragraph (section 4 of the summary).
  - _Requirements: 8, 9_

- [ ] 8b. Stretch (only after Task 8)
  - [ ] 8b.1 Behavioural lesion on rented hardware (Req 10): pre-register an addendum, then run. Decide by November 30.
  - [ ] 8b.2 Investigate the "Titanic" direction seen in the pilot (an attractor in layer-1 readouts).
  - ~~Second model with an n-gram table~~: cut (Decision 7).

- [ ] 8c. Own-memory side test (Req 13; Decisions 11–14)
  - [x] 8c.1 135M kill test: pre-register (59c1f85), LR pilots, 2M (EXTEND), 6M (KILL). `spikes/own_memory/RESULTS.md`.
  - [x] 8c.2 Bigger-table test, pre-registered (4f59d1e): H1 and H2 fail. Row-shuffle diagnostic (exploratory): the gain is in the rows.
  - [x] 8c.3 1.7B kit: pre-register (cd4f870); `scale.py`, `laptop.ps1`, `rent.sh` (background run, one-time-code transfer, isolation from the rental image's Python), with CPU smoke tests.
  - [x] 8c.4 1.7B run **[HW]** on a rented RTX 5090 (Vast.ai), started from the builder's phone: PASS, 8.868 vs 8.949 (0a68d14).
  - [x] 8c.5 Exploratory LoRA run at 3e-4 **[HW]** (rule written in f48d794 before the run): 8.235, below the memory's 8.868, so the PASS does not survive this check. The pre-registered outcome stays PASS.
  - [ ] 8c.6 Write the follow-up result into `RESULTS.md`; side-test section in the README and About page (Req 13.5).

- [ ] 9. Write-up and submission
  - [x] 9.1 README final: problem, how it works, claims boundary, results, limitations, disclosures, how to run. (2026-11-25; revisit after the video.)
  - [ ] 9.2 Finish the Kiro write-up from the running draft.
  - [ ] 9.3 Video script and recording **[HW]**: under 3 minutes, on the builder's machine, no copyrighted music, no logos, model named generically (record with `?generic`); YouTube public.
  - [ ] 9.4 Text description for the form.
  - [ ] 9.5 Final compliance check against `docs/submission/deliverables.md`; tag the submitted commit; submit before the deadline.
  - _Requirements: 12_
