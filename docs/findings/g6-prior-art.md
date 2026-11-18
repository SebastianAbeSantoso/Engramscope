# Gate G6: prior-art re-check (2026-11-11)

**Question.** Has anyone already decoded what DeepSeek's Engram rows store, or built a tool that reads them without running the model?

**Result: nothing found that does either.** Existing work explains when the memory is used, how to store or serve it, or how to redesign it. None of it decodes the content of the rows. A search can only show what it finds, so the claim stays: "we found no prior tool or study that decodes Engram rows directly". Re-check before submission (Task 9.5).

Searches run (web search, 2026-11-11):
- Engram conditional memory and interpretability;
- probing or decoding rows without running the model;
- interpretability of hashed n-gram embeddings;
- logit lens or nearest tokens on Engram rows;
- browser tools that visualize the Engram table;
- GitHub tools that decode Engram concepts.

## What exists

| Work | What it does with Engram | Overlap with Engramscope |
| --- | --- | --- |
| [Cheng et al. 2026, "Conditional Memory via Scalable Lookup", arXiv 2601.07372](https://arxiv.org/html/2601.07372) (the Engram paper) | LogitLens and CKA on **hidden states** to argue that Engram adds effective depth (Sec. 6.1). Visualizes the **gate value** per token, which is high on multi-token entities and set phrases in English and Chinese (Sec. 6.5, Fig. 7). | Shows *when* memory is used, not *what a row contains*. We cite it as the source of the mechanism and the gate result. |
| Serving runtimes: [SGLang Engram docs](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/engram.mdx), [vLLM Engram docs](https://docs.vllm.ai/en/v0.33.0/features/engram/), [colibri](https://github.com/JustVugg/colibri/blob/main/docs/deepseek-v41.md), [lucebox](https://github.com/Luce-Org/lucebox/blob/main/server/docs/DS41.md), community DGX Spark and KTransformers recipes | Read rows from disk or NVMe row by row during inference. | Same row arithmetic (colibri notes 48 rows per token across both tables). These tools run the model; they do not interpret rows. |
| Quantization: [arXiv 2610.21810](https://arxiv.org/pdf/2610.21810.pdf) (2-bit tables), [arXiv 2610.35985](https://arxiv.org/pdf/2610.35985.pdf) (GateRQ), [arXiv 2610.31646](https://ar5iv.labs.arxiv.org/html/2610.31646) | Compress the tables. | None on content. |
| Redesigns: [Tokenizer-Agnostic Engram, arXiv 2607.29065](https://arxiv.org/pdf/2607.29065v1), [Tensorizing Engram, arXiv 2606.08347](https://arxiv.org/html/2606.08347v1), [Factorized N-gram Memory, arXiv 2609.35578](https://arxiv.org/html/2609.35578) | Change hashing or factorization. | None on content. |
| Same-name projects (ENGRAM conversational memory, arXiv 2511.12960; several GitHub "engram" memory tools) | Unrelated agent memory systems. | Name only. Engramscope's name is distinct. |

## Notes for the claims

- **Safe wording:** "Engramscope reads DeepSeek-V4.1-Flash's Engram rows over HTTP without running the model and decodes them into words. We found no earlier tool or study that decodes row contents."
- **Not safe:** "first to interpret Engram". The paper's gate visualization is a form of interpretation.
- **Lesions** (searched separately on 2026-11-11 with lesion and ablation queries):
  - **Whole-table removal exists.** GateRQ reports that removing the Engram tables from DeepSeek-V4.1-Flash costs 36.8 PopQA points ([arXiv 2610.35985, abstract](https://arxiv.org/abs/2610.35985)). That is a behavioural lesion of the entire memory. Cite it as the "all traces removed" end point that Engramscope does not measure.
  - **Component ablations while training:** the Engram paper (Sec. 6.2, Fig. 5); a collision study ([arXiv 2601.16531](https://arxiv.org/html/2601.16531v1)).
  - **"Computational lesions" of language models** are an existing method in general (e.g. [arXiv 2604.10627](https://arxiv.org/pdf/2604.10627v1.pdf)). Lashley is used as an analogy in interpretability writing ([example](https://interpreted.substack.com/p/the-limitations-of-anthropics-attribution)).
  - **Not found:** lesions of individual Engram traces (rows, heads, n-gram sizes) of one phrase, or mass-action and equipotentiality tests on them.
  - **Safe wording:** "We found no earlier study that lesions individual Engram traces or tests Lashley's mass action and equipotentiality on them."
  - Do not claim to be the first to lesion a language model, or the first to remove Engram.
- **Newer checkpoint.** DeepSeek released `DeepSeek-V4.1-Flash-1017` on 2026-10-17 ([lucataco's journey notes](https://github.com/lucataco/DeepSeek-V4.1-Flash-2x-DGX-Spark/blob/main/docs/JOURNEY.md)), and community repos mention "official 1017 Engram tables" ([x00byte](https://github.com/x00byte/DeepSeek-V4.1-Flash-1017-Dual-Spark)). Engramscope stays pinned to revision `314113f8…` of `DeepSeek-V4.1-Flash`. The README should name the pin, and comparing the two checkpoints' tables was a candidate for the second-model stretch, which is now cut (Decision 7). Not verified yet: whether the 1017 tables differ.
