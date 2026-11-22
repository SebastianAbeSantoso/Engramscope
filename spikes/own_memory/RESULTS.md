# Own-memory kill test: results at 2M tokens (2026-11-22)

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
