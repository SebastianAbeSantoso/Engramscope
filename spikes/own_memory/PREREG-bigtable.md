# Bigger-table test (pre-registered, 2026-11-22, before the run)

**Question.** Does table size hold back the bolted-on memory? At 6M tokens the memory memorised nothing: it scored the same on text it had seen as on text it hadn't. One untested explanation is that its tables are too small (about 65,500 rows per column), so many different n-grams share a row.

This is a **new test**, not a rescue. The 135M kill (`RESULTS.md`) stands regardless of the outcome.

## Setup

Identical to the 135M test in `PREREG.md` except for one thing:

- **Table size:** 16 times bigger. Rows per column start at 1,048,576 instead of 65,536 (`OWNMEM_PRIME_START=1048576`). That gives about 8.39M rows in total, or about 537M table parameters instead of about 33.6M.
- **Kept the same:** the backbone (frozen SmolLM2-135M), the data, the order of the training stream, the seed, the batch size, the warm-up and the evaluation.
- **Learning rate:** the small-table pilot's choice is reused: 3e-3 for the table and 6e-4 for `W_k`, `W_v` and the norms. There is no new pilot, to keep the test cheap. A bigger table might want a different rate, so a failure here could partly reflect an untuned rate. That is stated as a limitation.
- **Training:** 2M tokens (2,002,944 after rounding up to whole steps).
- **Code:** `ownmem.py` with additive options whose defaults are unchanged: `OWNMEM_PRIME_START`, `OWNMEM_CKPT_EVERY` and `--tag big-`. Milestone checkpoints now store parameters only. Neither change affects training.

## Rules (decided now)

The comparisons use validation perplexity (PPL) at 2M tokens against two results already measured with the same data and evaluation: the small-table memory (18.520) and LoRA (15.725).

- **H1, table size matters:** `PPL_big <= 0.97 × 18.520 = 17.964`.
- **H2, the bigger table beats LoRA:** `PPL_big <= 15.725`.
- **Secondary (descriptive):** the memorisation ratio (memorisation-slice PPL divided by validation PPL), against 1.020 for the base model and 1.023 for the small table. A ratio of 0.99 or lower is reported as "memorises the text it saw".

## What it decides for the 1.7B test

- **If H1 holds:** the 1.7B memory uses the 16× table, kept in CPU RAM as Engram's design intends. If that does not fit in the laptop's RAM according to the speed check, it uses the largest power-of-two fraction of it that fits.
- **If H1 fails:** the 1.7B memory uses a small table, sized to keep the 135M ratio of memory to LoRA trainable parameters (1.75).
