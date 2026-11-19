# First results: readout benchmark and lesion experiments (2026-11-19)

**For the builder's review (checkpoints 7.3 and 8.4).** The wording below follows the pre-registered rules. None of it goes into the README, app or video until the builder approves it.

Generated reports (all numbers here come from them):
- [`docs/results/bench-2026-11-19.md`](../results/bench-2026-11-19.md), with `.json`;
- [`docs/results/lesion-2026-11-19.md`](../results/lesion-2026-11-19.md), with `.json` and `-decoded.json.gz` (every decoded list, including each random subset).

Both ran at code commit 85a33f0. The phrases are as registered in ec0e429 and the protocol as approved in 9d79b32, all before either run.

## 1. The instrument works (readout benchmark, Req 6)

Across the 123 pre-registered phrases, layer 1 has a related word in its top 10 for **61.8%** of phrases (95% interval 53.7–69.9%). The controls:

| Control | hit@10 |
| --- | ---: |
| random rows | 0.0% |
| another phrase's rows (shuffled) | 0.0% |
| layer 14 | 2.4% |
| the last word's own embedding | 32.5% |

**Layer 1 beats every control** (paired intervals all above 0). The smallest margin is over the last word's embedding: +29.3 points [16.3, 41.5].

Examples, labelled as examples:
- "the Kremlin" → kremlin, 斯大林 (Stalin), 莫斯科 (Moscow), moscow, putin, 苏联 (USSR);
- "the Leaning Tower of Pisa" → bologna, venice, florence, italy, galileo;
- "the Dutch East Indies" → indonesia, 印尼, sumatra.

## 2. Lashley's questions (lesion experiments, Req 9)

All results are **trace lesions (readout)**: traces are removed from the value-vector sum, and what is left is decoded. They show how a concept is spread across a phrase's 24 traces, not what the running model forgets. They say nothing about brains. Layer 1, 76 phrases with an intact hit:

- **Mass action (Q1): "late collapse: the traces stand in for each other until most are gone."**
  - Removing 8 random traces of 24 leaves the concept in 87.8% of phrases, 16 leaves 66.2%, 23 leaves 15.2%.
  - Half the concepts are gone only when **19.0** traces are removed [18.0, 19.8].
  - Shape statistic S = 0.19 [0.14, 0.23].
- **Equipotentiality (Q2).**
  - **"Equipotential across heads."** Removing any one hash head (3 traces) costs the same as removing 3 random traces, within the corrected intervals of about ±6 points. Each head holds 12.3–12.8% of the vector's energy.
  - **"Not equipotential across n-gram sizes."** Removing the 8 two-word traces leaves the concept in **10.5%** of phrases, against 87.8% for 8 random traces. Removing the 8 four-word traces leaves it in 98.7%, which is less damaging than chance.
- **Single traces (B1, descriptive).** A single trace is critical in 3.9% of phrases. On their own, an average of 3.6 of the 24 traces keep the concept.
- **Semon's partial cue (Q3a).** "A 2-gram cue alone retrieves the concept above chance": 94.7% of phrases. The 3-word traces alone retrieve it in 15.8% (also above chance), and the 4-word traces alone in 0%.
- **Semon's partial cue in other words (Q3b, no label by design).** Put after five different lead-ins ("I read about …", "Saya pernah membaca tentang …"), the concept survives in 92–93% of phrases. This holds even though only 18% of the four-word traces and 50% of the three-word traces stay the same in context; 86% of the two-word traces recur.
- **Predictions written before the run:**
  - P1 (no head is special): **held**.
  - P2 (some n-gram size is special; in 2-token phrases the two-word traces are essential and the others are not): **held**.
- **The rules produce the parallel with Lashley's conclusion (2)** (1950, p. 478): limited regions may be essential, but within such a region the parts are functionally equivalent. Here the "regions" are the n-gram sizes and the "parts" are the hash heads.
- **Layer 14:** only 3 phrases have an intact hit, so the power guard applies. Numbers only, no labels.

Why it could look like this, an interpretation that has not been tested:
- Each trace on its own is mostly noise. Two traces' contributions have a mean cosine of 0.014, and only 3.6 of 24 retrieve the concept alone.
- What a phrase shares across its heads is the concept, so the 8 heads add up the signal while their noise cancels.
- The four-word tables hold far more distinct n-grams per row than the two-word tables, which would make their rows noisier.

## 3. Caveats

- **Padding.** Phrases were tokenized alone, as registered. For short phrases, the windows longer than the phrase contain padding, which favours the two-word traces. The pre-registered split by length shows the same pattern in phrases of 4 or more tokens, which have no padding: removing the two-word traces costs 50.2 points [19.2, 78.5]. There are only 12 such phrases with a hit, so that interval is wide.
- **Single words are not in this memory (exploratory, not pre-registered).** None of the 17 single-token phrases ("coffee", "Bali", "DNA", …) has a readable trace, alone or after any of the five lead-ins. For the 106 multi-token phrases, the hit rate is 71.7%. Engram's own paper presents it as a memory for multi-token patterns ([Cheng et al. 2026](https://arxiv.org/html/2601.07372), Sec. 6.5). For the app, example phrases should be multi-word names, and single words should get an honest "nothing readable here".
- **Readout, not behaviour.** The running model also uses the rows in its gate, and later layers can compensate. Whether the model forgets anything is the behavioural lesion (Req 10).
- **Decoding is an approximation** (nearest input embeddings). These numbers use the exact readout; the browser index will be approximate.

## 4. Draft wording for the README (needs approval)

> We reran Karl Lashley's lesion experiments on a language model's memory, pre-registered. Each phrase is stored in 24 traces. The stored concept survives until about 19 of them are removed. Any of the 8 hash heads can stand in for any other, but the two-word traces cannot be replaced: remove those 8 and the concept is gone in 9 of 10 phrases. That is the pattern Lashley described in rat cortex in 1950: some regions are essential, and within a region the parts are interchangeable. It describes this model's memory, not brains.
