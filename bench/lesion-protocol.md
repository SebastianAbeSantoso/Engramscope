# Lesion protocol (pre-registered)

Status: **written and committed before any lesion was run.** The commit that adds this file is the pre-registration commit; `lesion.py` records its hash in every result. The interpretation rules in Section 6 are **proposed by Kiro and wait for the builder's approval** (research-integrity rule: pass marks come from the builder). No lesion runs until they are approved. Any change after that gets a dated note at the end of this file.

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

## 6. Interpretation rules (PROPOSED, need the builder's approval)

Applied to layer 1, measure M2, in this order. Each rule names its comparison, so the wording in the README and app follows from the numbers.

- **Q1, mass action.**
  - Say **"graded"** if mean M2 does not increase as k grows (allowing overlap of the 95% intervals between neighbouring k), and no single step between neighbouring k values accounts for more than half of the total drop from A0 to A(23).
  - Say **"threshold-like"** if one step accounts for more than half of the drop.
  - Otherwise say **"irregular"**, and show the curve.
- **Q2, equipotentiality.**
  - Compare each structured lesion, B3(n) and B4(h), with the random lesions of the same size, A(8) and A(3).
  - A structured lesion is **"special"** if its mean M2 lies outside the 2.5–97.5% range of the means of 2,000 bootstrap draws of the random condition.
  - Say **"equipotential at this level"** if no B3 and no B4 lesion is special, and **"localized to n-gram size n" / "head h"** for each special one.
  - Separately, for B1: report the share of phrases in which one single-row lesion flips the hit to a miss. Call a phrase's trace **"single-point"** if exactly one row does that.
- **Q3, partial cue.**
  - **Q3a:** for each n, report C1(n) M2 with its interval. Say "a size-n cue alone retrieves the concept" for a phrase set if the interval's lower bound is above the chance level.
  - **Q3b:** report M2 per context, and the share of rows recurring per n-gram size. No pass mark; the rule is to show both side by side.

## 7. Output

- `docs/results/lesion-<date>.json`: per phrase, layer and condition, the decoded top-10, M1, M3 and M4, plus all structure measures.
- `docs/results/lesion-<date>.md`: the tables and curves with intervals, the applied rules from Section 6 with their outcome, the commit hash of this file, and the hash of the code.
- Everything shown in the app's experiment views is read from these files. The app's live lesion view (where a user removes rows by hand) is labelled as exploration, not as the result.

## Change log

- (none yet)
