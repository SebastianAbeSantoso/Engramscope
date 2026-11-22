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
