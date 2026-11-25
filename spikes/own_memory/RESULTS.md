# Own-memory kill test: results (2026-11-22)

Rules: `PREREG.md` (committed in 59c1f85 before any training). Validation = WikiText-103 validation, held out. Memorisation slice = the first 200k tokens of the training stream (seen once, early).

| Model | Val PPL 0.5M | Val PPL 1M | Val PPL 2M | Memo PPL 0.5M | Memo PPL 1M | Memo PPL 2M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base (frozen SmolLM2-135M) | 21.785 | 21.785 | 21.785 | 22.217 | 22.217 | 22.217 |
| Engram-style memory, 34.2M trainable | 19.721 | 19.022 | 18.520 | 20.116 | 19.447 | 18.943 |
| LoRA r=64, 19.5M trainable | 15.937 | 15.776 | 15.725 | 14.401 | 14.771 | 15.115 |

## Rules applied (val, 2M)

- r_mem = 14.99% (gain over base); r_lora = 27.82%.
- PASS needs r_mem >= 3% and PPL_mem <= PPL_lora: 18.520 vs 15.725, so **not passed**.
- KILL clause 2: memory is behind (18.520 > 1.02 x 15.725 = 16.039), but its 1M->2M gain (2.64%) is larger than LoRA's (0.32%), so it is catching up and the clause does not fire.
- **Outcome: EXTEND.** Per the rules, both runs continue to 6M tokens; whatever doesn't pass there is killed.

## Notes (descriptive, not rules)

- Training speed on the sandbox CPU: memory 1310 tok/s, LoRA 933 tok/s.
- Seen vs unseen text: LoRA's memo/val ratio went 0.904 -> 0.961 (memorised the early slice, then partly forgot it). The memory's stayed 1.020 -> 1.023, the same as the base model's 1.020: it learned general patterns, not the specific text it saw. Untested explanation: about 65.5k rows per column is small for the number of distinct n-grams, so many share rows.
- Projection (estimate, not a measurement): at the memory's 1M->2M rate, val PPL at 6M would be about 17.75, still above LoRA. Matching LoRA's 2M value at that rate would take about 140M tokens, and the rate is still falling.

## 6M extension (2026-11-22): final outcome

Both runs resumed from their 2M checkpoints on fresh, non-repeating stream data (change log, 8b6356e).

| Model | Val PPL 2M | Val PPL 6M | Memo PPL 2M | Memo PPL 6M | Training time to 6M |
| --- | ---: | ---: | ---: | ---: | ---: |
| base (frozen) | 21.785 | 21.785 | 22.217 | 22.217 | - |
| Engram-style memory | 18.520 | 17.798 | 18.943 | 18.288 | 4668 s |
| LoRA r=64 | 15.725 | 16.483 | 15.115 | 16.533 | 6637 s |

- Rule at 6M (binary): PASS needs r_mem >= 3% and PPL_mem <= PPL_lora. r_mem = 18.30% (r_lora = 24.34%); PPL_mem 17.798 > PPL_lora 16.483.
- **Outcome: KILL.** The bolted-on memory learns, but a LoRA of similar size beats it. Per PREREG, scaling to a 3B model is not cleared.

Notes (descriptive):
- **LoRA got worse with more data:** val PPL 15.725 at 2M to 16.483 at 6M, on data it had never seen. Likely cause, untested: the constant LR of 1e-3 was picked by a 300k-token pilot (top of its grid) and is too high for long training without decay.
- **The outcome does not depend on that.** LoRA's own 2M checkpoint (15.725, after 2146 s of training) beats the memory at 6M (17.798, after 4668 s). LoRA wins at matched tokens and at matched training time.
- The memory was still improving (3.9% from 2M to 6M). Its mean gate fell to 0.525 (0.621 at 2M).
- Still no memorisation: the memory's memo/val ratio is 1.028 (base 1.020). LoRA's advantage on the early slice is gone (1.003).
- Training speed on the sandbox CPU: memory 1285 tok/s, LoRA 904 tok/s.

## Bigger-table test (2026-11-22; rules: `PREREG-bigtable.md`, committed in 4f59d1e before the run)

| Memory table | Rows | Val PPL 0.5M | Val PPL 1M | Val PPL 2M | Memo/val ratio at 2M |
| --- | ---: | ---: | ---: | ---: | ---: |
| small (main test) | 524,450 | 19.721 | 19.022 | 18.520 | 1.023 |
| 16x bigger | 8,388,916 | 19.726 | 19.049 | 18.546 | 1.026 |

- **H1, table size matters** (needs <= 17.964): 18.546, so **H1 fails**. The 16x table scores the same as the small one at every checkpoint.
- **H2, beats LoRA** (needs <= 15.725): **fails**.
- The memorisation ratio stays at the base model's level, so the bigger table does not memorise either.
- **Decision for the 1.7B test** (rule fixed in PREREG-bigtable.md): use a small table, sized to keep the 135M ratio of memory to LoRA trainable parameters.

## Exploratory diagnostic (not pre-registered): is the gain in the rows?

`diag_shuffle.py` scores the 6M small-table memory with its trained rows, with the rows randomly permuted, and with every row replaced by the mean row:

| Rows | Val PPL |
| --- | ---: |
| trained | 17.798 |
| shuffled | 21.740 |
| mean row | 21.780 |
| base model (no memory) | 21.785 |

With the wrong or the average rows, the whole gain disappears. So the memory's gain comes from n-gram-specific content in its rows, not from the dense part acting on generic rows. Combined with H1, table size is not what limits it at this data scale. A remaining candidate, not tested, is the design itself: a frozen backbone that was never trained to read the memory.

## 1.7B test (2026-11-25; rules: `PREREG-1.7B.md`, committed in cd4f870 before any 1.7B run)

Backbone: SmolLM2-1.7B, frozen, bf16. Run on a rented RTX 5090 (Vast.ai, driver 580.173.02, torch 2.14.1+cu130, transformers 5.24.0, peft 0.22.0), code b27ef71. Raw files: `results/scale/`. Memory: 126.5M trainable (1,944,242 rows); LoRA r=64: 72.4M trainable. Both at micro-batch 4 with gradient checkpointing; the gradient check passed (largest relative difference 1.6%, limit 2%).

**LR pilot** (300k tokens each; selection-slice PPL, lowest wins):

| Memory LR | 1e-3 | 3e-3 | **1e-2** | | LoRA LR | 3e-4 | **1e-3** | 3e-3 |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| sel PPL | 10.347 | 9.718 | **9.374** | | sel PPL | 8.434 | **8.371** | 9.451 |

**Main runs** (base model: val PPL 11.129, memo PPL 11.361):

| Tokens | Memory val PPL | LoRA val PPL | Memory memo PPL | LoRA memo PPL | Memory training time | LoRA training time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2M | 9.238 | 9.371 | 9.307 | 8.854 | 212 s | 387 s |
| 5M | 9.117 | 9.488 | 9.256 | 9.240 | 525 s | 972 s |
| 10M | 8.988 | 9.972 | 9.209 | 10.023 | 1051 s | 1947 s |
| 20M | **8.868** | **8.949** | 9.096 | 8.967 | 2103 s | 3899 s |

### Rule applied (val, 20M)

- r_mem = 20.31% (r_lora = 19.58%), so r_mem ≥ 3%.
- PPL_mem 8.868 ≤ PPL_lora 8.949.
- **Outcome: PASS.**

### What the PASS does and doesn't show (descriptive, not rules)

- **The margin is small:** 0.081 PPL (0.9%), one run per method, one seed. The rule had no margin requirement, so this is a PASS, but a narrow one.
- **The flip from 135M comes mostly from LoRA.** The memory's gain over base barely moved (135M: 18.3%; 1.7B: 20.3%). LoRA's fell (24.3% to 19.6%).
- **LoRA trained unevenly.** Its val PPL rose from 9.371 at 2M to 9.972 at 10M, then fell to 8.949 at 20M as the cosine schedule lowered its learning rate. That suggests its pilot-chosen rate (1e-3, picked on 300k tokens) was too high for this run. 3e-4 was close in the pilot. An exploratory LoRA run at 3e-4 was written down before it ran (`PREREG-1.7B.md`, "Exploratory follow-up"); its result goes below when it arrives.
- **The memory's rate was at the edge of its grid** (1e-2, the highest tested), so it may be undertuned too. That isn't tested.
- **The memory was ahead at every milestone on validation, and at matched training time.** For example, after about 1,000 s the memory was at 8.988 (10M tokens) and LoRA at 9.488 (5M). The memory trained about 1.9× faster per token (10,026 vs 5,078 tok/s in the speed check).
- **Still no memorisation.** The memory's memo/val ratio stayed at 1.007–1.026, around the base model's 1.021. LoRA's started at 0.945 (it memorised the early slice) and ended at 1.002. At 20M, LoRA is better on the memorisation slice (8.967 vs 9.096).
- **The memory's mean gate fell** from 0.395 at 2M to 0.168 at 20M. Why is untested.
- **A CPU-only note:** in a later sandbox code test (SmolLM2-135M on CPU), the gradient-checkpointing check failed once (differences 3.5–5.7%) and passed with 0.0 on two reruns. The GPU run's own check passed, and its result is the one recorded above.

### Exploratory follow-up: LoRA at learning rate 3e-4

**Pending** (running on the same machine, code f48d794). Reading, decided before the run: at or above 8.868 at 20M, "the PASS survives this check"; below, "the PASS does not survive this check". The pre-registered outcome stays PASS either way.

## Related work (found 2026-11-25, after the runs)

Bolting Engram-style memory onto a frozen model is not new. These papers do it, or something close, and are cited rather than competed with:
- **Engram Adapter** ([arXiv 2608.29327](https://arxiv.org/abs/2608.29327), Findings of EMNLP 2026): Engram-style memory as a post-hoc adapter for frozen Qwen3-4B and 8B, aimed at domain adaptation without losing general skills.
- **TF-Engram** ([arXiv 2607.07388](https://arxiv.org/html/2607.07388)): a train-free Engram with an SSD-backed table; on Qwen3-0.6B it reports beating the frozen model and a parameter-matched LoRA.
- **NGM** ([arXiv 2605.16893](https://arxiv.org/abs/2605.16893)): a training-free n-gram memory built from a frozen model's own token embeddings (Qwen3, Llama 3.2).
- **Memory Grafting** ([arXiv 2605.20948](https://arxiv.org/abs/2605.20948)): stores another model's hidden states as n-gram memory, during pretraining.

What this test adds is narrower: the same question asked as a pre-registered, controlled comparison against LoRA (same data, budget, schedule and machine, with pilots for both) at two sizes, reporting both outcomes (KILL at 135M, a narrow PASS at 1.7B), plus a row-shuffle control at 135M.
