# Requirements Document: Engramscope

## Introduction

Engramscope reads what a frontier language model remembers about a phrase, **without running the model**. DeepSeek-V4.1-Flash (284B parameters, MIT license) keeps two large lookup tables called *Engram*, at layers 1 and 14: about 384 million rows each, 189 GiB in total. The address of each row is a hash of the last few tokens of the text. Engramscope computes those addresses in the browser and fetches only the needed rows from the public model files with HTTP range requests: 24 rows per layer, about 6 KB of data plus scales. It then projects the rows the way the model does and decodes the result into words.

A kill test on 2026-11-19 (`docs/findings/engram-killtest.md`) showed two things:
- Read directly, the rows do **not** predict the next token.
- They **do** point at concepts related to the phrase. In 30 phrases, 22 had a related word in the top 10 for layer 1, against 0 of 30 for random rows and 8 of 30 for the phrase's own last-token embedding. For example, "the Eiffel Tower" decoded to Paris, France and Louvre.

The name and story come from neuroscience. Richard Semon coined *engram* for a memory trace in 1904. Karl Lashley spent decades looking for where engrams live in the brain and concluded in 1950 that he could not localize them. In this model an engram has an exact address computed from the text. Sources are in `docs/theory-sources.md`.

Kiroween category: **Resurrection** (proposed). Deadline: December 4, 2026, 2:00 PM PT (December 5, 06:00 WITA).

### Claims boundary

These apply to the spec, README, write-up and video.

- Engram, its hashing and its tables are DeepSeek's work ([Cheng et al., arXiv 2601.07372](https://arxiv.org/abs/2601.07372); the model's `inference/engram.py`). Engramscope claims none of it.
- Engramscope's claimed contribution is limited to: **reading a released model's Engram rows directly, without running the model, over HTTP from a browser, and showing what they encode, with controls that show the readout is not noise.**
- Related work edits the tables ([EngramEdit](https://arxiv.org/abs/2610.10533)), compresses them ([CP](https://arxiv.org/abs/2610.21810), [GateRQ](https://arxiv.org/abs/2610.35985)), fetches them remotely for inference ([remote Engram over RDMA/TCP](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma)), or reads them from disk on demand (llama.cpp n-gram read mode, DwarfStar). Engramscope cites these and does not claim remote row access or on-demand reads.
- Every quality claim comes from the benchmark in Requirement 6, with its controls. Examples shown in the UI or video are labelled as examples.

## Glossary

- **Engram table:** one of the model's two lookup tables (layers 1 and 14). Each row is 256 FP8 E4M3 values with 8 E8M0 scales (one per 32 values).
- **Address / row id:** the row index for one (n-gram size, head) pair, computed by the model's hash from the compressed ids of the last up-to-4 tokens. Each position has 24 addresses per layer: 3 n-gram sizes (2, 3, 4) × 8 heads.
- **Compressed id:** a token id mapped through the model's normalization table (NFKC, accents stripped, lowercased, whitespace collapsed), so "The", "the" and "THE" share an id. The compressed vocabulary has 99,092 ids.
- **Value vector:** the 5,120-dimensional vector the model adds to its residual stream (scaled by a gate): the value part of `wkv` applied to the 24 concatenated rows.
- **Readout:** decoding a value vector into words by cosine similarity with the model's input token embeddings.
- **Decode index:** a compressed copy of what the readout needs (token embeddings and value projection), small enough to ship to a browser.
- **Control:** a readout of something that should carry no information about the phrase (random rows, rows of an unrelated phrase), shown next to the real result.
- **Lesion:** zeroing a phrase's rows and observing what the running model can no longer recall (Lashley's method). Needs the full model.

## Requirements

### Requirement 1: Exact addressing

**User Story:** As a viewer, I want the addresses Engramscope fetches to be exactly the ones the model uses, so that what I see is the model's memory and not an approximation.

#### Acceptance Criteria

1. THE address computation SHALL be bit-exact with `inference/engram.py` at the pinned model revision: compressed token map, hash multipliers (per-layer RNG), prime bucket sizes and offsets, padding at the start of the text, and the XOR-and-modulo hash.
2. THE build SHALL check, and fail on a mismatch, that the compressed vocabulary has 99,092 ids and that the bucket primes of each layer sum to that layer's row count (384,006,168 and 384,016,682).
3. THE browser implementation SHALL produce the same addresses as the Python reference implementation for every input in a shared test-vector file, and for randomly generated token sequences in property tests.
4. ALL model file URLs SHALL pin the model revision (commit), so the app keeps working if the repository's main branch changes.

### Requirement 2: Lookup without running the model

**User Story:** As a viewer, I want to type a phrase and see what the model remembers about it in about a second, without a GPU or a download, so that I can explore the model's memory casually.

#### Acceptance Criteria

1. WHEN the user enters a phrase, THE app SHALL tokenize it with the model's tokenizer, compute the addresses for the phrase's last position, fetch the rows with HTTP range requests, and show the decoded words for each Engram layer.
2. THE app SHALL show what it fetched: the 24 row addresses per layer, the byte offsets, and the total bytes downloaded against the table size.
3. THE app SHALL run entirely in the browser, with no Engramscope server. The only network requests SHALL be to the model files and to the app's own static assets.
4. THE app SHALL resolve each model file's CDN location at most once per session per file and reuse it, and SHALL back off on HTTP 429.
5. THE time from submitting a phrase to showing results SHALL be measured and shown. A target is set by the builder from the Task 1 measurements.

### Requirement 3: Inside the lookup

**User Story:** As someone curious about how models work, I want to see each step from text to remembered words, so that the mechanism is visible and not a black box.

#### Acceptance Criteria

1. THE app SHALL show the pipeline for the chosen position: tokens → compressed ids → the 2-, 3- and 4-gram windows → 24 addresses → fetched bytes → value vector → decoded words.
2. THE app SHALL decompose the value vector into the contribution of each n-gram size, and of each of the 24 rows, and decode each contribution separately. This is possible because the value projection is linear.
3. THE app SHALL let the user pick any position in the phrase, not only the last one.
4. THE app SHALL show the gate inputs it cannot compute without running the model (the residual stream) as unknown, and SHALL NOT imply that a decoded row is what the model used at full strength.

### Requirement 4: Honest readout

**User Story:** As a skeptical viewer, I want to see that the decoded words are not noise or wishful reading, so that I can trust the demo.

#### Acceptance Criteria

1. EVERY lookup SHALL show a control readout beside the real one: the same decoding applied to random rows of the same table.
2. THE app SHALL label the readout as an approximation: decoding by nearest input embeddings assumes the value vector lives in embedding space, which is plausible at layer 1 and not proven.
3. THE app SHALL show the benchmark result (Requirement 6) with its sample size and controls on its About page.
4. WHERE the decode index is compressed, THE app SHALL state its measured agreement with the exact readout (Requirement 5.3).

### Requirement 5: Decode index

**User Story:** As a viewer on an ordinary laptop or phone, I want the app to load quickly, so that I am not downloading gigabytes before I see anything.

#### Acceptance Criteria

1. THE decode index SHALL be built offline from the pinned model files, by a script in this repository, reproducibly.
2. THE decode index SHALL be small enough to load in a browser. A size budget is proposed in design and confirmed by the builder after Task 1.
3. THE build SHALL measure and record agreement between the compressed readout and the exact readout (top-10 overlap over a fixed phrase set) and the benchmark score with each.
4. THE decode index SHALL cover the model's full vocabulary, or SHALL document which tokens were excluded and why.

### Requirement 6: Benchmark with controls

**User Story:** As a judge, I want a measured, pre-registered result, so that the central claim is not cherry-picked.

#### Acceptance Criteria

1. THE benchmark SHALL use at least 100 phrases, with their related-word sets written and committed **before** any readout of those phrases is run. The commit hash SHALL be recorded in the results.
2. EACH related-word set SHALL exclude the phrase's own words and normalized forms of them.
3. THE benchmark SHALL report hit@1, hit@5 and hit@10 with 95% bootstrap intervals for: Engram layer 1, Engram layer 14, random rows, the phrase's last-token input embedding, and the rows of a different phrase (shuffled control).
4. THE benchmark SHALL run offline in Python, from the same code path the decode index is built from, and SHALL store raw outputs in `docs/results/`.
5. IF layer 1 does not beat every control, THEN the README, app and video SHALL say so and SHALL NOT claim the readout works.

### Requirement 7: Interface and experience

**User Story:** As a viewer, I want the app to feel like looking into a memory, so that it is memorable and fits Kiroween.

#### Acceptance Criteria

1. THE interface SHALL be a complete product, not a debug page: a landing view with one input, a result view, an inside-the-lookup view, and an About page with sources and the benchmark.
2. THE app SHALL work on current desktop Chrome, Edge and Firefox, and SHALL be usable on a phone-width screen.
3. THE visual design SHALL be atmospheric and spooky in a way that serves the function (memory traces, fading, addresses), without obscuring the data.
4. THE app SHALL offer example phrases, and SHALL work for any phrase the user types.
5. THE app SHALL be accessible: keyboard operation, visible focus, sufficient contrast, and text alternatives for visualizations.

### Requirement 8: Theory and history

**User Story:** As a judge in the Resurrection category, I want the old idea to be accurate and central, not decoration.

#### Acceptance Criteria

1. EVERY statement about Semon, Lashley or the history of the engram SHALL come from `docs/theory-sources.md`, which records what each source says and where.
2. THE app and README SHALL connect the history to a working feature: an engram with a computable address that can be read (Semon's term), and, if the stretch is done, a lesion experiment (Lashley's method).
3. THE README SHALL label clearly which parts are DeepSeek's design and which are Engramscope's.

### Requirement 9: Lesion experiment (stretch)

**User Story:** As a viewer, I want to see what the model forgets when one memory is removed, the experiment Lashley ran on rats.

#### Acceptance Criteria

1. WHERE time and budget allow, THE project SHALL run the full model on rented hardware with the rows of selected phrases zeroed, and SHALL record the model's answers to prompts about those phrases with and without the lesion.
2. THE results SHALL be shown as recorded results, labelled with the hardware and date, not as a live feature.
3. IF the lesion has no measurable effect, THEN that result SHALL be reported.

### Requirement 10: Other models (stretch)

**User Story:** As a viewer, I want to compare what different models remember.

#### Acceptance Criteria

1. WHERE time allows, THE app SHALL support a second model with an n-gram lookup table (candidates: Qwen3.8-Flash-Next, LongCat-Flash-Lite), with the same exactness checks as Requirement 1.
2. IF a candidate's hashing or table layout cannot be reproduced exactly, THEN it SHALL NOT be added.

### Requirement 11: Engineering standards

**User Story:** As the builder, I want the code tested and reproducible, so that judges can run it and so that changes don't silently break exactness.

#### Acceptance Criteria

1. THE Python tooling SHALL have unit tests and property-based tests (`hypothesis`) for the hash, FP8 decoding, byte-offset computation and readout.
2. THE web app SHALL have unit tests (`vitest`) and SHALL check its hash against the shared test vectors.
3. CI SHALL run lint and tests for Python and web on every push.
4. THE repository SHALL contain all source and instructions needed to rebuild the decode index and run the app.
5. THE app SHALL be published as a static site at a public URL and SHALL stay available until January 29, 2027, 1:00 PM PT.

### Requirement 12: Kiroween submission compliance

**User Story:** As the builder, I want the submission to meet the rules, so that it is eligible.

#### Acceptance Criteria

1. THE repository SHALL be public, SHALL carry an open-source license detected by GitHub, and SHALL include `.kiro/` at the root, not listed in `.gitignore`.
2. THE submission SHALL include a URL to the working app, a text description, a public YouTube demo video under 3 minutes showing the app on the device it was built for, and a Kiro write-up.
3. THE video SHALL contain no copyrighted music and no third-party logos. How the model's name is shown follows the answer to the trademark question in `docs/kiroween-rules.md`.
4. THE submission SHALL disclose pre-existing work used: the model and its reference code (MIT), the tokenizer, libraries, the MaltOE project (where the kill test was first run, during the submission period), and the papers cited.
5. THE submission SHALL be made before December 4, 2026, 2:00 PM PT, entering Resurrection and Best Student Project: Resurrection.
6. ALL deliverables SHALL be tracked in `docs/submission/deliverables.md`, and the Kiro write-up kept as a running draft in `docs/submission/kiro-writeup.md`.

## Open Questions (proposed answers in brackets; the builder decides)

1. **Category:** Resurrection (Semon and Lashley as the old idea) or Costume Contest (the interface as the main prize)? [Resurrection; the interface is still polished per Req 7.]
2. **Layer 14:** it reads mostly as noise with this decoding. Show it with that caveat, or show layer 1 only? [Show both, with layer 14 labelled as unclear.]
3. **Fully client-side, or a small server for decoding?** [Client-side, if the compressed decode index passes Task 1; otherwise a free static-plus-serverless fallback.]
4. **Lesion stretch (Req 9):** worth about $50–150 of rented GPU time near the end? [Decide after the core is done, by November 30.]
5. **Second model (Req 10):** in scope as a stretch, or out? [Stretch, after the core.]
6. **Video naming:** email Devpost about naming "DeepSeek" on screen? [Yes, now; until answered, use the name only in plain text and no logos.]
