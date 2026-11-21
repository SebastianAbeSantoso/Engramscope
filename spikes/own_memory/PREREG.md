# Kill test: does an Engram-style memory, bolted onto a frozen pretrained model, add anything? (pre-registered)

Written and committed on 2026-11-21 (WITA), **before any training run**. `ownmem.py` in this folder is the code that runs it. The LR pilot in step 3 runs after this commit and only picks learning rates, using a selection slice that is never used for the final numbers.

## Question

DeepSeek trains its Engram memory together with the model, from scratch. Our plan bolts a memory onto an existing small model and trains **only the memory**. Does that learn anything useful beyond plain fine-tuning on the same data?

## Setup

- **Backbone:** `HuggingFaceTB/SmolLM2-135M` (Apache-2.0), frozen, bf16, on the sandbox CPU (Xeon 8488C with AMX; about 1,390 training tokens/s measured).
- **Data:** WikiText-103 raw (CC BY-SA). Training text comes from `train` shard 0. Articles are shuffled with seed 20261121 and packed into 512-token sequences, with EOS between articles.
  - Disjoint article sets: a **selection slice** of 100k tokens for the LR pilot, a **memorisation slice** of 200k tokens drawn from the training stream, and the training stream itself.
  - Final evaluation uses the WikiText-103 **validation** split: held-out articles, never trained on.
- **Memory module (ours, Engram-style):** our exact DeepSeek-style hash (`engramscope.hashing`), applied to the SmolLM2 tokenizer.
  - Compressed token ids feed 2-gram and 3-gram windows, 4 heads each: 8 columns.
  - Each column is a prime-sized table starting above 65,536 rows, for about 524k rows in total, 64 dimensions per row (about 33.6M parameters).
  - The 8 rows are concatenated (512 dimensions). Then: key `k = W_k·e`, value `v = W_v·e`, gate `α = sigmoid(rmsnorm(h)·rmsnorm(k) / √576)`, output `h ← h + α·v`.
  - The module sits after decoder layer 2 (0-based index 1). `W_v` is zero-initialised, so the model starts exactly unchanged.
  - Trainable: the table (SparseAdam), plus `W_k`, `W_v` and the two norm scales (Adam).
- **Baseline: LoRA**, rank 64, alpha 64, on q/k/v/o/gate/up/down in all 30 layers. About 19.5M trainable parameters, similar to the memory's size. Same data, same token budget.
- **Base:** the frozen model with no training, evaluation only.
- **Training:** batch 8 × 512 tokens, 20 warm-up steps then a constant learning rate, bf16 autocast. Each method gets the learning rate picked in the pilot.

## Procedure

1. Commit this file and `ownmem.py` (the pre-registration).
2. Prepare and tokenize the data. Test that the vectorised hash equals `engramscope.hashing.Hasher` on two full training sequences (1,024 positions × 8 columns).
3. **LR pilot:** for each method, LR ∈ {3e-4, 1e-3, 3e-3} for the memory's table (with `W` at LR/5), and LR ∈ {1e-4, 3e-4, 1e-3} for LoRA. Each run trains on 300k tokens of the training stream. The learning rate with the lowest loss on the **selection slice** wins. That learning rate is then fixed.
4. **Main runs:** memory and LoRA, 2M training tokens each, with checkpoints evaluated at 0.5M, 1M and 2M tokens.
5. Evaluate on validation and on the memorisation slice.

## Measures

- **Primary:** validation perplexity (PPL) after 2M tokens.
- **Secondary:**
  - perplexity on the memorisation slice;
  - the curve over 0.5M, 1M and 2M tokens;
  - wall time per method;
  - the memory's mean gate value.

## Rules (decided now)

Let `r_mem = 1 − PPL_mem / PPL_base` and `r_lora = 1 − PPL_lora / PPL_base`, on validation at 2M tokens.

- **PASS:** `r_mem ≥ 3%` **and** `PPL_mem ≤ PPL_lora`. The bolted-on memory helps at least as much as a LoRA of similar size.
- **KILL:** either of these:
  - `r_mem < 1%` (the memory barely helps);
  - `PPL_mem > 1.02 × PPL_lora`, **and** the memory's relative PPL gain from 1M to 2M tokens is not larger than LoRA's (the memory is behind and not catching up).
- **EXTEND:** anything else. Both methods continue to 6M tokens and the same rules apply there, except that "not catching up" uses 2M→6M. After that, no more extensions: whatever doesn't pass is killed.
- A PASS here only clears the idea for scaling to a 3B model. It is not a claim that small beats big; that needs the later test against a bigger model at matched VRAM.

## Change log

- 2026-11-21, before the commit and before any training on real data: data preparation, the hash test (passed: 1,024 positions × 8 columns), and a 2-step smoke test on random token ids ran first, to check the code. No result depends on them.
