# How Kiro was used (running draft)

Last updated: 2026-11-26.

## How the project got here (the spec gate that killed the first idea)

Engramscope is the second project of this Kiroween. The first, MaltOE ([repo](https://github.com/SebastianAbeSantoso/MaltOE)), planned an inference runtime that narrows mixture-of-experts layers under GPU memory pressure. Its Kiro spec put a feasibility round with falsification gates before any engine work. Over two days of measurements on the builder's laptop, the gates showed:
- narrower experts lose to lower-bit model files at the same size (width 75%: 1.16× perplexity; a Q3_K_M file with the same expert bytes: 1.006×);
- the space is crowded with fast engines.

This showed a negative result, i could've stopped there and built the app, but it felt unsatisfying to me. The project stopped before building the engine.

Kiro then ran several idea searches with research sub-agents (about 150 candidates checked against prior art), and the builder steered towards model internals. One question from the builder ("can you run a model via HTTP range requests?") led to DeepSeek-V4.1-Flash's Engram tables. Kiro wrote a kill test in the sandbox:
- reproduced the model's hashing from its reference code, with two exact checks;
- fetched rows from Hugging Face by range request;
- tested two hypotheses. The first, "the rows predict the next token", failed. The second, "the rows encode related concepts", passed: 22/30 against 0/30 for random rows.

That became Engramscope. It's similar to vibecoding, but it isn't coding so it's just vibing haha. Anyway, the agent here failed to confirm prior products for LoRA x n-gram hash and it dissapointed me.

## Kiro clients

- **Kiro Web, Vibe mode** for the whole process so far: research, kill test, spec, gates, Python reference, experiments and the side test. From Nov 25 the builder worked from a phone only (laptop unavailable); Kiro did all repository and sandbox work, and the builder ran the one GPU job through a phone browser.
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
  - **G3:** a one-page fetch prototype the builder opened on their own connection: 1.07–1.09 s per phrase cold, 0.54 s warm, every byte verified. The Task 1.7 checkpoint followed.
- **Scope change after the gates.** The builder asked whether the project was only dissecting DeepSeek's implementation, not implementing the old research. It was: Semon and Lashley were only framing. The spec was rewritten so that Lashley's lesion experiments (mass action, equipotentiality) and Semon's partial-cue test, rerun on the Engram traces, are the core (Req 9). The readout became the instrument, and the second-model stretch was cut. Kiro wrote the lesion protocol as a pre-registration and committed it before any lesion run. The protocol's interpretation rules wait for the builder's approval, because the research-integrity steering says pass marks come from the builder.
- **Pre-registration in practice.** The builder delegated the lesion rules to Kiro. Before any run, Kiro tightened them: paired tests with Bonferroni correction, a power guard, and a shape statistic that does not depend on the uneven lesion sizes. It also wrote two predictions down, and committed it all (9d79b32). Code guards refuse to run if the phrase file or the protocol differs from its registered commit. Kiro then built the Python reference with 26 tests (hypothesis property tests against a transcription of DeepSeek's reference hash, FP8 against `ml_dtypes`, lesion additivity) and ran the benchmark and the lesions in the sandbox (about 13 minutes). Both predictions held. The rules, applied mechanically, produced the parallel with Lashley's 1950 conclusion. A null result would have been reported the same way.
- The same steering rule caught Kiro claiming a negative prior-art result ("no lesion study found") before running that search. The search then found a whole-table removal result (GateRQ) that is now cited.

## The results, and what the rules made Kiro say (Nov 19)

The benchmark and lesions ran in the sandbox in about 13 minutes. Layer 1 found a related word for 61.8% of the 123 phrases [53.7, 69.9], against 0% for random and shuffled rows. Half of a phrase's concepts disappeared only when 19 of its 24 traces were removed. The hash heads were interchangeable, but removing the 8 two-word traces left only 10.5% of concepts, against 87.8% for 8 random traces. Kiro wrote the results up for the builder's review (`docs/findings/experiments-2026-11-19.md`) and kept them out of the README until approved, as the steering requires. It also flagged an unplanned finding as exploratory: none of the 17 single-word phrases had a readable trace.

## Looking for something bigger, and stopping (Nov 19–25)

After the results, the builder went looking for a stronger idea. In the builder's words: "I felt that Engramscope by itself was not in the spirit of a hackathon. You build something new, deploy it, and people use it. A study, in my opinion, lacks the second and third points. And it was using another well-documented technology (DeepSeek's), which imo is worse than a study using new materials." Kiro ran about 15 rounds of idea searches with research sub-agents, and checked an outside 30-idea report the builder attached. The builder sharpened the filter along the way: an old idea as the core, real use, and no existing working tool. Kiro logged each round, and why each candidate failed, on the `explore/holophone` branch (`docs/ideas/`). Two ideas (a holophone and SMS compression) got kill tests in the sandbox and passed them technically, but were dropped anyway; SMS compression, for example, already exists as a working tool (ts_sms). Nothing passed every filter, and the builder decided to finish Engramscope.

## The side test: does DeepSeek's memory help another model? (Nov 21–25)

The builder asked to add a small n-gram memory on a bigger model. Kiro proposed a kill test first, with the rules committed before any training (Req 13):
- **135M, sandbox CPU:** an Engram-style memory using Engramscope's exact hash, bolted onto frozen SmolLM2-135M, against a LoRA of similar size. The rules gave EXTEND at 2M tokens and KILL at 6M (17.798 vs 16.483). Kiro reported the KILL as the outcome, so the bigger model wasn't cleared by that rule.
- **Two follow-ups explained the KILL:** a 16× bigger table changed nothing (pre-registered H1 failed), and shuffling the trained rows erased the whole gain, so the gain is in the rows.
- **1.7B, rented GPU:** pre-registered with the table size fixed by the bigger-table rule. The outcome was PASS by 0.9% (8.868 vs 8.949, base 11.129).
- **When the builder reacted to the PASS with surprise, Kiro pointed out the weak spots first:** the small margin, one seed, and LoRA's validation getting worse mid-run. It proposed a LoRA re-run at a lower learning rate, wrote its reading down before the run, and labelled it exploratory so it couldn't change the PASS. LoRA at the lower rate reached 8.235, 7.1% below the memory, so by the reading written down before the run, the PASS does not survive. The write-up reports both: the pre-registered PASS and the follow-up that undoes it in practice.
- **Prior art:** asked how this differs from DeepSeek's and Qwen's n-gram work, Kiro searched and found four papers doing the same basic move (Engram Adapter, TF-Engram, NGM, Memory Grafting). It recommended against calling the side test new, and the spec's claims boundary now says so (Decision 14).

## Running a GPU job from a phone (Nov 25)

With the laptop out of reach, Kiro built `rent.sh` so the builder could start the 1.7B run on a rented Vast.ai GPU by pasting one line into a phone browser:
- **The run is detached** (`setsid`, `nohup`, a `flock` lock), so closing the tab doesn't stop it, and a second start just shows progress. A `watch` command shows progress again later.
- **It checks the disk first, and resumes after a stop.** Every step skips when its result exists, and training checkpoints every 15 minutes.
- **Results come back by one-time code.** `rent.sh send` packs the results and sends them with a pinned, checksummed transfer tool. Kiro receives them in the sandbox. There's no token on the rented machine and no file handling on the phone.
- **Kiro tested the whole loop in the sandbox on CPU before any money was spent.** That included start, a refused second start, watch, send and receive, and the published start line from a fresh clone.
- **A catch before renting:** reading Vast's image source, Kiro found it activates its own Python environment (with PyTorch) and turns off uv's cache. A sandbox test showed such an environment leaks its packages into the run. `rent.sh` now runs only the pinned packages on a clean Python, with its own cache.
- **Picking the machine:** the builder sent offer screenshots, and Kiro checked each against the job. It turned down an A100 because its 24.7 GB free disk was too small, and a China-hosted 5090 because the test downloads from Hugging Face. It also explained that Vast's DLPerf score and the cards' FP32 TFLOPS don't measure this workload.
- **Outcome:** one PyPI timeout on the first start, fixed by a rerun. The whole rental, including the follow-up run, cost $2.20 of the $5 credit. The extra LoRA run reused the machine's downloads, and Kiro tested its update line against a copy of the machine's state, which caught a `git pull` that would have failed.

## The app (Nov 25)

- **Asset build first, measured.** `build_assets.py` fits the decode index chosen at G1 and writes the browser assets, then measures them against the exact readout at every loading stage (`docs/results/decode-index-2026-11-25.md`). One change from the plan, recorded in the code and tasks: one projection is shared by both layers, so the download stays at about 92 MB instead of doubling.
- **Exactness checked by tests, not by eye.** Python writes test vectors; vitest checks the browser tokenizer and hash against them (58 phrases, every id and address exact) and the browser readout against Python's compressed readout.
- **Checked in a real browser.** Kiro served the built app to a headless browser and ran lookups and lesions against the live model files: "the Kremlin" decoded to Kremlin, Moscow, Putin; removing the 2-word traces of "the Eiffel Tower" erased Paris, as the recorded results predict. The phone-width layout has no horizontal overflow.

## A sandbox restore (Nov 25)

At the start of one session, Kiro checked `git status` and found that a sandbox restore had dropped 8 result files, 2 git objects and a script's executable bit. It restored them from GitHub (`git fetch --refetch`, then `git restore`) and confirmed with `git fsck` before doing anything else.

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
- The hooks are configured. I have not verified that one fired during a task, so treat their use as unverified.

## Property-based testing

- Done for Python: 26 tests, including `hypothesis` property tests for hash parity against a transcription of DeepSeek's reference code, address ranges, bucket sums, FP8 decoding against `ml_dtypes`, and lesion additivity (the per-row decomposition sums back to the full vector).
- Shared test vectors (58 phrases) check the browser tokenizer, hash and readout against Python.

## Sub-agents, powers, MCP

- Research sub-agents: idea and prior-art searches (MaltOE phase).
- The GitHub CLI power was not used; `gh` was run from the sandbox shell.
- MCP: none so far.
