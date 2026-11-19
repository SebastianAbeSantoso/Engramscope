# Lesion protocol (pre-registered)

Status: **written and committed before any lesion was run** (a6caabf, 2026-11-19). **Interpretation rules finalized and approved on 2026-11-19, still before any run** (Section 6 and the change log). The builder delegated the choice of rules to Kiro. `lesion.py` records the commit of this file in every result and refuses to run on any other version. Any later change gets a dated note in the change log.

What had been seen before writing this: the kill-test pilot (30 dev phrases, intact readouts only) and gate G1. G1 decoded single rows of the 50 dev phrases to measure compression, not lesions, and never touched the benchmark phrases. No readout of `bench/phrases.jsonl` has been run.

## 1. Questions

Lashley's two principles and Semon's retrieval rule, rerun on DeepSeek-V4.1-Flash's Engram memory, where every trace has a known address:

- **Q1, mass action (Lashley 1950, Section VII).** Does the stored concept degrade gradually with the number of traces removed, or does it collapse at some lesion size?
- **Q2, equipotentiality (Lashley 1950, p. 469 and conclusion 2, p. 478).** At equal lesion size, does it matter *which* traces are removed? Is any single trace critical?
- **Q3, retrieval from a partial cue (Semon 1921, p. 138).** Does part of the cue retrieve the concept? Two forms: keeping only the 2-, 3- or 4-gram traces, and putting the phrase after different preceding words, so that only some of its traces recur.

Sources and page numbers: `docs/theory-sources.md`.

## 2. What a trace lesion is, and what it is not

At one token position and one layer, the model fetches 24 rows: 3 n-gram sizes × 8 hash heads. The value vector is a sum of per-row contributions, `v = Σᵢ cᵢ`, with `cᵢ = Wv[:, blockᵢ] · rowᵢ`. A **trace lesion** removes a set S of rows from that sum, `v_S = Σ_{i∉S} cᵢ`, and decodes `v_S` the same way as the intact `v`.

This is arithmetic on a linear sum. What it measures is how the decoded concept is spread across the 24 traces: in many redundant rows, in a few, or in one. It does **not** measure what the running model forgets. In the model, the rows also feed the gate (through the keys), and later layers can compensate. That is the **behavioural lesion**: Req 10, a stretch goal on rented hardware. Every result from this protocol is labelled "trace lesion (readout)".

## 3. Material

- **Phrases:** the 123 pre-registered phrases in `bench/phrases.jsonl` (commit ec0e429), tokenized as `" " + phrase`, last token position. The related-word sets and matching rules are those of the readout benchmark (`bench/README.md`).
- **Layers:** layer 1 is primary. Layer 14 is run with identical conditions and reported as secondary. Its intact readout reads mostly as noise (kill test, G1), so its curves are expected to sit at the floor. That expectation is stated, not tested.
- **Decoding:** the exact readout in Python: cosine against the full, row-normalized input-embedding table (129,280 tokens), top 10. The browser's compressed index is not used for any number in this protocol.
- **Contexts for Q3b:** the phrase placed after each of these 5 fixed prefixes, last token position of the phrase:
  1. `"I read about"`
  2. `"Yesterday we talked about"`
  3. `"There is a picture of"`
  4. `"Nobody knows much about"`
  5. `"Saya pernah membaca tentang"`

  The text is `prefix + " " + phrase`, so the phrase keeps its leading-space tokens.

## 4. Conditions

All lesions apply to the 24 rows of the last position, per phrase and per layer. Random subsets use `numpy.random.default_rng(20261111)`, drawn in the order phrase id, then layer, then k.

| Id | Condition | Rows removed |
| --- | --- | --- |
| A0 | intact | none |
| A(k) | random lesions of size k, for k ∈ {1, 2, 3, 4, 8, 12, 16, 20, 22, 23} | 50 random subsets of size k per phrase |
| B1(i) | single-row lesion | row i only (24 conditions) |
| B2(i) | single-row sparing | every row except i (24 conditions) |
| B3(n) | n-gram-size lesion | the 8 rows of size n, for n ∈ {2, 3, 4} |
| B4(h) | head lesion | head h at all three sizes (3 rows), for h ∈ {0..7} |
| C1(n) | partial cue: keep size n only | the 16 rows not of size n (the complement of B3(n)) |
| C2(p) | context | none; the intact readout of the phrase after prefix p |

## 5. Measures

Per phrase, layer and condition. A condition with 50 subsets is averaged per phrase first.

- **M1, concept hit:** at least one of the top-10 decoded tokens (stripped, lowercased) is in the phrase's related set, after excluding the phrase's own tokens. The same rule as the readout benchmark.
- **M2, retention:** M1 under the lesion, only over phrases with M1 = 1 when intact (the phrases the lesion can damage). This is the main measure for Q1 and Q2.
- **M3, vector similarity:** `cos(v_S, v)`.
- **M4, list overlap:** |top-10(v_S) ∩ top-10(v)| / 10.
- **Structure measures (intact, no lesion):**
  - the share of each row in the total, `‖cᵢ‖² / Σⱼ‖cⱼ‖²`;
  - the mean pairwise cosine between contributions;
  - per n-gram size, the cosine between the sum of that size's rows and `v`.
- **Q3b extras:** the number of the 24 rows whose address is unchanged between the bare phrase and phrase-after-prefix, per n-gram size. M1 for each context. M2 among phrases with a bare-phrase hit.

References shown with the results. These are not controls.
- **Orthogonal reference for M3:** `√((24 − k) / 24)`, the value expected if the 24 contributions were orthogonal with equal norm. Above it means the remaining rows carry redundant directions; below it means the removed rows carried more than their share.
- **Chance level for M1:** the random-rows control of the readout benchmark (Req 6). It runs in the same session, before the lesions.

Uncertainty: 95% percentile bootstrap over phrases, 2,000 resamples, seed 20261111. Reported for every curve point and condition.

## 6. Interpretation rules (approved 2026-11-19)

The rules apply to layer 1, measure M2, in this order. Each rule names its comparison, so the wording in the README and app follows from the numbers. Layer 14 gets the same rules if it passes the power guard.

- **Power guard.** A layer gets labels only if at least 30 phrases have an intact concept hit (M1 = 1 under A0). Otherwise its results are shown with intervals and no labels.
- **Q1, mass action.** Let R(k) be mean retention over phrases at lesion size k. R(0) = 1, then the A(k) points for k = 1 … 23. Let D = 1 − R(23) be the total drop.
  1. If R(23) ≥ 0.5, say **"redundant: most concepts survive with a single trace"**.
  2. Otherwise, if R rises between two neighbouring sizes and their 95% intervals do not overlap, say **"irregular"** and show the curve without another label.
  3. Otherwise compute the shape statistic S, the mean over k from 0 to 23 of (R(k) − L(k)) / D. Here L is the straight line from (0, 1) to (23, R(23)), R is interpolated linearly between measured sizes, and the mean uses the trapezoid rule. S = 0 is a straight line. S = +0.5 is the limit for a curve that stays at 1 and collapses at the end; S = −0.5 for one that collapses at once. The 95% interval comes from 2,000 bootstrap resamples over phrases.
     - Interval inside [−0.1, 0.1]: **"proportional, as in Lashley's mass action"**.
     - Interval entirely above 0.1: **"late collapse: the traces stand in for each other until most are gone"**.
     - Interval entirely below −0.1: **"early loss: each trace removed costs more than its share"**.
     - Anything else: no label.
  4. Always report k50, the lesion size at which R crosses 0.5 (linear interpolation), with its 95% interval.
- **Q2, equipotentiality.**
  - Each structured lesion is compared with random lesions of the same size: B3(n) (8 traces) with A(8), B4(h) (3 traces) with A(3). Per phrase, d = M2(structured) − M2(random), where the random M2 is that phrase's mean over its 50 subsets. Bootstrap the mean of d over phrases, 10,000 resamples. There are 11 comparisons, so each uses a Bonferroni-corrected two-sided interval (percentiles 0.227 and 99.773).
  - A structured lesion is **"special"** if its interval excludes 0. It is **"more damaging than chance"** if the interval is below 0, **"less damaging"** if above.
  - Say **"equipotential across heads"** if no B4 lesion is special, and **"equipotential across n-gram sizes"** if no B3 lesion is special. Otherwise name each special lesion and its direction.
  - B1 is descriptive: report the share of phrases where at least one single-trace lesion turns the hit into a miss, and the share where exactly one trace does (**"single-point"**).
- **Q3, partial cue.**
  - **Q3a:** for each n, per phrase d = M2(C1(n)) − chance, where chance is that phrase's random-rows hit (0 or 1) at the same layer, from the readout benchmark. Bootstrap the mean of d over phrases, 10,000 resamples, Bonferroni-corrected over the 3 sizes (percentiles 0.833 and 99.167). If the interval lies above 0, say **"a size-n cue alone retrieves the concept above chance"**.
  - **Q3b:** report M2 per context with its 95% interval, beside the share of the 24 traces whose address recurs. No label.
- **Secondary analysis (descriptive, no labels).** B3 and C1 split by the phrase's length in tokens (`" " + phrase`), in bins 1, 2, 3 and ≥4 tokens. The bins were chosen from the token counts before any readout: 17, 45, 39 and 22 phrases. Length matters because at the last position, windows longer than the phrase reach before its start and contain padding. For a 2-token phrase, only the 2-gram window lies inside the phrase; for a 1-token phrase, none does.

## 6b. Predictions (written before any run)

These are written down so that a confirmed prediction counts for something and a failed one is visible. The results are reported against them either way.

- **P1, heads.** No head lesion is special. The 8 heads hash the same n-gram into different tables, so they are interchangeable by construction. The model has a reason to store the same content in each: summing them averages out hash collisions.
- **P2, n-gram sizes.** At least one n-gram-size lesion is special. In the 2-token bin, removing the 2-gram traces is more damaging than chance, and removing the 3- or 4-gram traces is not. This bin is checked with its own 95% interval of d.
- **No prediction** for the shape of the mass-action curve (Q1), single traces (B1), or the partial-cue conditions (Q3).

If P1 and P2 both hold, the result would parallel Lashley's conclusion (2): limited regions may be essential, but within such a region the parts are functionally equivalent (1950, p. 478). The parallel is claimed only if the rules above produce both labels.

## 7. Output

- `docs/results/lesion-<date>.json`: per phrase, layer and condition, the decoded top-10, M1, M3 and M4, plus all structure measures.
- `docs/results/lesion-<date>.md`: the tables and curves with intervals, the applied rules from Section 6 with their outcome, the commit hash of this file, and the hash of the code.
- Everything shown in the app's experiment views is read from these files. The app's live lesion view (where a user removes rows by hand) is labelled as exploration, not as the result.

## Change log

- **2026-11-19, before any run: Section 6 finalized and approved; Section 6b added.** The builder delegated the decision ("up to you since whats the best for the project"). Changes from the proposed version (a6caabf):
  - Added the power guard.
  - Q1: the largest-step rule depended on the uneven spacing of the lesion sizes, so it was replaced by the interpolated shape statistic S with a ±0.1 band. The labels "redundant", "late collapse" and "early loss" were added, and k50 is always reported.
  - Q2 and Q3a: each now uses paired per-phrase differences with Bonferroni-corrected intervals (10,000 resamples). The proposed version compared one mean with an unpaired bootstrap range.
  - Added the secondary analysis by phrase length and the predictions.
- **Note on the seed.** 20261111 is an arbitrary fixed number. It was chosen while Kiro's session date was wrongly set to 2026-11-11 (the real date was 2026-11-19). It stays unchanged because it is part of the pre-registration.
