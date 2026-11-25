# 1.7B test: does the 135M result hold on a 12.7x bigger model? (pre-registered)

Written and committed on 2026-11-22 (WITA), **before any 1.7B run**. The code is `scale.py` in this folder. On the builder's Windows laptop it runs through `laptop.ps1`, and on a rented Linux GPU through `rent.sh`. The 135M kill and the bigger-table result (`RESULTS.md`) stand whatever happens here.

## Question

At 135M, an Engram-style memory bolted onto a frozen model improved it, but a LoRA of similar size beat it. Does that hold for SmolLM2-1.7B?

## Setup

- **Backbone:** `HuggingFaceTB/SmolLM2-1.7B` (Apache-2.0), frozen, in bf16, on an NVIDIA GPU. It uses the same tokenizer file as SmolLM2-135M (sha256 checked), so the hash and the compressed token map are unchanged.
- **Data:** built exactly as in the 135M test: the same files (sha256 checked at download), the same shuffle seed and the same selection, memorisation and validation slices. The training stream is the 135M stream extended to 20.6M tokens. A sandbox check confirmed identical slices and an identical stream prefix.
- **Memory:** the same design as `PREREG.md`, after decoder layer 2: 2- and 3-gram windows × 4 heads, 64-dimensional rows.
  - The table lives in CPU RAM, as in Engram's design; only the looked-up rows move to the GPU.
  - **Size, from the bigger-table rule** (H1 failed): 1.75 times LoRA's trainable parameters, the same ratio as at 135M. That means `OWNMEM_PRIME_START=242944`: 1,944,242 rows and 126.5M trainable parameters, against LoRA's 72.4M.
- **LoRA:** r=64, alpha 64, on q/k/v/o/gate/up/down in all layers (72,351,744 trainable).
- **Training:**
  - Steps of 8 × 512 tokens. Micro-batches of 4, 2 or 1, whichever fits, with gradient accumulation.
  - Gradient checkpointing (non-reentrant). In the sandbox it left the memory's gradients exactly unchanged. The check step repeats this on the run machine and stops if the difference is 2% or more.
  - bf16 autocast.
- **Learning rate (changed from 135M):** 20 warm-up steps, then cosine decay to 10% of the peak at the last step, for both methods. At 135M a constant rate let LoRA get worse after 2M tokens.
- **LR pilot:** 300k tokens per run with the same schedule shape. Memory table LR ∈ {1e-3, 3e-3, 1e-2}, with `W` at LR/5. LoRA LR ∈ {3e-4, 1e-3, 3e-3}. Each grid is centred on that method's 135M choice. The lowest selection-slice perplexity wins.
- **Token budget:** 20M tokens per method. If the speed check projects more than 7 hours for either run (training plus milestone evaluations), the budget drops to 10M, then 5M, taking the first that fits. If even 5M doesn't fit, the test moves to a rented GPU with the same code and rules.
- **Milestones:** validation and memorisation-slice evaluations at 2M, 5M, 10M and 20M tokens, as far as the budget reaches.
- **Environment:** transformers 5.24.0 and peft 0.22.0, as in the 135M runs. torch is 2.15.0 if the NVIDIA driver supports CUDA 13.2, otherwise 2.14.1 (CUDA 13.0 or 12.6). Both methods run in the same environment on the same machine, and `check.json` records it.

## Rule (decided now)

At the final budget, with validation perplexity (PPL):

- **PASS:** `r_mem = 1 − PPL_mem / PPL_base ≥ 3%` **and** `PPL_mem ≤ PPL_lora`.
- **KILL:** anything else. There is no extension.

The earlier milestones are reported for information only.

## Outputs

`results/scale/`: `check.json` (environment, speeds, chosen budget), `base.json`, `pilot-choice.json`, `main-mem.json`, `main-lora.json` and `summary.json`.

## Change log

- 2026-11-22, before the commit and before any 1.7B run: CPU code tests in the sandbox, with SmolLM2-135M and budgets of a few steps (`SCALE_SMOKE=1`). The full pipeline ran from an empty folder, the gradient check passed (difference 0.0), and the data matched the 135M test. No 1.7B result exists yet.
- 2026-11-25, before any 1.7B run: the builder chose a rented GPU. `rent.sh` changed in how it runs, not what it runs:
  - It still runs `scale.py all` with the same library versions, the same driver rule for the torch build, and the same rules.
  - It now runs in the background, so a closed browser terminal doesn't stop it.
  - It keeps its files under `/workspace`, which survives a pod restart, and checks the free disk space first.
  - It packs the results for a one-time-code transfer with runpodctl 3.1.1.
  - Tested in the sandbox with `SCALE_SMOKE=1` and SmolLM2-135M on CPU. The background run finished, a second start was refused while it ran, and the packed results arrived intact through the transfer. No 1.7B result exists yet.
- 2026-11-25, later, before any 1.7B run: the builder chose an A100 on Vast.ai. `rent.sh` again changed in how it runs, not what it runs:
  - It now ignores the rental image's own Python environment (Vast's activates one with PyTorch installed). The run gets only the pinned packages, on a clean uv-managed Python, with its own package cache under `/workspace`.
  - Each start writes the code version (git commit) to `run.log`.
  - Tested in the sandbox with Vast-like settings: an active environment holding an extra package, and uv's cache turned off. The run's Python couldn't see the extra package, and the `SCALE_SMOKE=1` run on CPU finished. No rules or versions changed. No 1.7B result exists yet.
