# Gate G1: decode index (2026-11-19)

**Question.** Can the browser decode Engram readouts into words without downloading the full embedding table (129,280 × 5,120, 1.3 GB in BF16)?

**Result.** Only approximately. No index of 66 MB or less reproduces the exact top-10 words well. The best one for its size keeps 60% of the exact top-10 words for layer 1. It is a 1,024-dimension index fitted to the readout, stored at 4 bits per value, 66 MB for the embedding side plus 12.6 MB per layer. Per-row cards fare worse: 28% overlap. The builder chooses the design at the Task 1.7 checkpoint (options below).

## Method

- **Exact readout** (as in the kill test): `v = Wv · concat(24 rows)`, words = top 10 by cosine against the row-normalized input embeddings, full vocabulary.
- **Compressed readout:** `score = Z · (M · x)`, with `x` the 6,144 fetched values, `Z` (vocab × d) quantized per token with one fp16 scale, and `M` (d × 6,144) in fp16. Two ways to choose the d directions:
  - **PCA** of the embeddings (`spikes/g1_decode_index/g1.py`).
  - **Fitted:** the best rank-d approximation of the score matrix `E · Wv · x`, weighted by the second moment of `x`. The moment is estimated from 4,096 contiguous table rows per column (`g1b.py`, variant `block`). The variant with `x` assumed isotropic (`iso`) scored within 0.01 of `block` (`g1b-fitted.json`).
- **Phrases:** `bench/dev_phrases.txt` (50 phrases; the first 30 are the kill-test pilot). The benchmark set is never used here. That gives 144 token positions per layer, 1,200 single-row contributions (24 rows at each phrase's last position), and 50 random-row controls.
- **Metrics:**
  - top-10 overlap with the exact list;
  - the share of positions where the exact top-1 word is in the compressed top 10;
  - pilot hit@10 (as in the kill test);
  - recall@200: the share of the exact top-10 found in the compressed top 200, for a shortlist-and-rerank design.

## Results (layer 1)

| Index | Embedding side | M per layer | Top-10 overlap | Exact top-1 in top 10 | Pilot hit@10 (exact: 22/30) | Row-card overlap | Recall@200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PCA d=256 int8 | 33.4 MB | 3.1 MB | 0.205 | 0.396 | 17 | 0.051 | n/a |
| PCA d=512 int8 | 66.4 MB | 6.3 MB | 0.324 | 0.688 | 20 | 0.103 | n/a |
| fitted d=256 int8 | 33.4 MB | 3.1 MB | 0.286 | 0.493 | 16 | 0.087 | 0.704 |
| fitted d=512 int4 | 33.4 MB | 6.3 MB | 0.428 | 0.806 | 19 | 0.149 | 0.828 |
| fitted d=512 int8 | 66.4 MB | 6.3 MB | 0.434 | 0.819 | 20 | 0.157 | 0.829 |
| **fitted d=1024 int4** | **66.4 MB** | **12.6 MB** | **0.599** | **0.896** | **20** | **0.278** | **0.972** |
| fitted d=1024 int8 | 132.6 MB | 12.6 MB | 0.592 | 0.889 | 20 | 0.294 | 0.976 |
| fitted d=2048 int8 | 265.0 MB | 25.2 MB | 0.800 | 1.000 | 21 | 0.636 | 1.000 |

Layer 14 agrees much less. With fitted d=1024 int4, overlap is 0.217 and the exact top-1 is in the top 10 at 0.396 of positions. Its pilot score is 1–3 of 30 for every index and for the exact readout. This fits the kill-test finding that layer 14 reads as noise with this decoding.

Findings:
- **4-bit storage costs nothing measurable** compared with 8-bit at the same d (0.599 against 0.592 at d=1024), so the size budget should go into dimensions.
- **Fitting to the readout beats PCA** at equal size: at d=512, overlap is 0.434 against 0.324, and the exact top-1 is kept at 0.819 against 0.688.
- **The benchmark-style score survives compression better than the word lists do.** Pilot hit@10 is 20/30 at d=1024 against 22/30 exact. A hit only needs one related word anywhere in the top 10.
- **Per-row cards are the weak point.** A single row's contribution is a smaller, noisier vector, and its decoded words change most under compression.
- **Shortlist and rerank doesn't help in the browser.** Reranking 200 candidates exactly would need 200 embedding rows from the model files, about 2 MB per readout over 200 separate requests. The CDN refuses multi-range requests (HTTP 416, tested), so they cannot be batched.

Raw outputs: `docs/results/g1-pca.{json,log}`, `g1b-fitted.{json,log}`, `g1b-block-*-int{4,8}.{json,log}`.

## Options for the Task 1.7 checkpoint

1. **Browser only, approximate (recommended).** Use fitted d=1024 int4: 66 MB of embeddings plus 2 × 12.6 MB for M, about 92 MB downloaded once and kept in the browser cache. The app labels its words "approximate decoding". The benchmark (Python) reports the exact and the compressed scores side by side (Req 5.3). Row cards show their words with a lower-confidence style.
2. **Browser only, smaller.** Use fitted d=512 int4, about 46 MB in total. Phrase overlap is 0.43, so a weaker match to the exact list, but loading is quicker on phones.
3. **Hybrid.** Option 1 in the browser, plus exact decoding from a free Hugging Face Space. The browser would send the 5,120-number vector `v` (20 KB) and get the exact top words back. This adds a server (Decision 3 allows it as the fallback). Free Spaces go to sleep when idle and take time to wake, which matters for judges testing until Jan 29, 2027.
