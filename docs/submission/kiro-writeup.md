# How Kiro was used (running draft)

Running draft of the Kiro write-up (Req 12.6, deliverable D11). Kiro adds to it at every checkpoint; the builder edits it into the final version (Task 9.2). Only things that happened go here. Plans are marked **planned**, unverified items **check**.

Last updated: 2026-11-11.

## How the project got here (the spec gate that killed the first idea)

Engramscope is the second project of this Kiroween. The first, MaltOE ([repo](https://github.com/SebastianAbeSantoso/MaltOE)), planned an inference runtime that narrows mixture-of-experts layers under GPU memory pressure. Its Kiro spec put a feasibility round with falsification gates before any engine work. Over two days of measurements on the builder's laptop, the gates showed:
- narrower experts lose to lower-bit model files at the same size (width 75%: 1.16× perplexity; a Q3_K_M file with the same expert bytes: 1.006×);
- the space is crowded with fast engines.

The project stopped before building the engine.

Kiro then ran several idea searches with research sub-agents (about 150 candidates checked against prior art), and the builder steered towards model internals. One question from the builder ("can you run a model via HTTP range requests?") led to DeepSeek-V4.1-Flash's Engram tables. Kiro wrote a kill test in the sandbox:
- reproduced the model's hashing from its reference code, with two exact checks;
- fetched rows from Hugging Face by range request;
- tested two hypotheses. The first, "the rows predict the next token", failed. The second, "the rows encode related concepts", passed: 22/30 against 0/30 for random rows.

That became Engramscope. *(Builder: add how this compared with vibe coding.)*

## Kiro clients

- **Kiro Web, Vibe mode** for the whole process so far: research, kill test, spec, scaffolding.
- Kiro IDE, CLI and Crew: not used so far.
- Sessions in Kiro Web keep running when the browser tab is closed (per kiro.dev docs, cloud sessions).

## Spec-driven development

- `.kiro/specs/engramscope/`: requirements (with a claims boundary and open questions for the builder), design (pinned model facts, components, correctness properties), tasks (feasibility gates first, then build).
- Task 1 gates, run in the sandbox before any app code:
  - **G4:** 123 benchmark phrases with related-word sets, committed (ec0e429) before any readout of them. The authoring rules and what the author had already seen are written down in `bench/README.md`.
  - **G1:** the planned 33 MB decode index failed its measurement (20% top-10 overlap with the exact readout). Kiro then fitted the projection to the readout instead of the embeddings and tested 4-bit storage. The best option per megabyte reached 60% at about 92 MB. The result went back to the builder as a choice, instead of the plan being changed quietly.
  - **G2:** both browser tokenizers matched Python on 1,000 strings with 0 mismatches.
  - **G5:** the research-integrity rule required primary sources. Kiro read Semon's 1904 first edition (German), the 1921 English translation and Lashley's 1950 paper, and recorded page numbers. The scan of Lashley's famous sentence is missing the page where it starts, so the docs only allow quoting the verified half.
  - **G6:** prior-art re-check; it also found a newer checkpoint of the model.
- **planned:** G3 (browser latency) on the builder's machine, then the Task 1.7 checkpoint.

## Steering

| File | Purpose |
| --- | --- |
| `product.md` | the exact claim and vocabulary |
| `tech.md` | stack, model pin, rate-limit rule, commands |
| `structure.md` | layout and where code goes |
| `research-integrity.md` | no invented numbers; benchmark pre-registration; history only from `docs/theory-sources.md` |
| `workflow.md` | push at checkpoints, commit as the builder, merge on request, keep the tracker and this draft current |

These carried over from MaltOE, where the research-integrity rule caught seven claims Kiro had made from memory about two 1960s–90s papers and made it check the sources.

## Agent hooks

- `checks-after-task.json` (sh) and `checks-after-task-windows.json` + `scripts/checks.ps1` (PowerShell): after each spec task, run ruff and pytest for Python and eslint and vitest for the web app, each skipping until it exists.
- **check:** whether they fired during task execution, and whether one caught a failure.

## Property-based testing

- **planned:** `hypothesis` tests for hash parity with the reference code, address ranges, bucket sums, normalization invariance, padding, byte offsets, FP8 decoding and linearity of the per-row decomposition (design.md, Correctness properties). Shared test vectors check the browser implementation against Python.

## Sub-agents, powers, MCP

- Research sub-agents: idea and prior-art searches (MaltOE phase).
- **check:** github-cli power usage.
- MCP: none so far.
