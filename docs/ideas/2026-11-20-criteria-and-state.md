# Idea search: the rule set and where we are (2026-11-20)

Running record so we stop rediscovering the same dead ends. Deadline Dec 4, 2026 PT (Dec 5 WITA). ~14 days left.

## The bar (all must hold)

1. **Core = implementing an old idea.** The project's central mechanism is an old paper/idea we build. Not a tool, lens, or wrapper on top of someone else's modern system. (Killed: Engramscope = "a tool on top of DeepSeek's Engram".)
2. **Deep learning.** The builder's interest is in the learning/mechanism, not adjacent tricks.
3. **Novel = no existing working software.** A *paper or theory* with no implementation is fine; we provide the implementation. If a working **tool** already does the thing, it's out, even with no commercial product. (Killed: SMS compression, because Bellard's `ts_sms` already exists. Analogy from the builder: "making a commercial tool for training when Unsloth exists for free".)
4. **5-second rule.** A stranger watching a silent 5-second clip, with one caption line, gets it. No narration. (Killed: holophone = noisy audio, point unclear.)
5. **Expectations defied.** A sentence most people agree with, and a demo that breaks it. (Builder's examples: "a Mac can't run a 500B model" -> it does; "a machine can't make new ideas" -> it synthesises.)
6. **Real use, not research-wow.** Someone would actually want it. (Killed: split-brain / hydra LLM = only research-interesting.)
7. **Feasible:** ~10 days, browser (WebGPU/WebAudio/canvas/cam/mic) or Linux CPU sandbox. The builder's own machines can't run local AI; a few hours of rented GPU is possible, not preferred.

## The tension that makes this hard

(3) says "no existing software" and (6) says "real use". Useful + easy things get built, so they already exist. Deep learning is the most-implemented field, so most named old DL papers already have GitHub code. The surviving zone is narrow: an old idea with a real use that was **ahead of its hardware** (so nobody shipped it) and has a **visual** wow.

## Rejected so far (do not revisit)

- Engramscope (tool on DeepSeek's Engram; but it's a working fallback with real pre-registered results on branch `task1-gates`).
- Holophone (noisy, point unclear). Descartes/Hooke/Freud memory machines (need narration / no use).
- Hydra & split-brain LLM (research-wow, no use).
- Video Rewrite 1997 deepfake (builder: faking isn't the interest). Von Neumann reliable-from-unreliable; Linsker cortex-from-noise (core unclear to builder).
- SMS / message compression (ts_sms already exists).
- Optical / delay-line "LLM without hardware" (needs lab hardware; and research/tools exist).
- Earlier: MoE width runtime, super-weights, OLMoE organ, NETtalk, Shannon mind-reader, dream-unlearning, memory transplant, etc.

## Proposal to break the loop

Start from a **real use the builder has**, then find the old paper that solves it. Starting from "cool old idea" keeps failing filter 3 or 6. Candidate pains tied to the builder (Indonesia, RX 6650 XT 8 GB, student, bad local-AI support): weak/no GPU for local models; metered or patchy mobile data; study workload.

## Update, late 2026-11-20

- **Category:** Resurrection is not fixed. Frankenstein is also acceptable. Skeleton Crew and Costume Contest don't interest the builder.
- **Dasher + LLM** rejected. The Dasher project is under active development (v6). The belief "hands-free writing is slow" is false: stenography and Dasher itself already exist.
- **Belief ballot:** of 8 beliefs, the builder holds and would be wowed by only **#5, "a small AI can't beat a big one."**
- **#5 is already broken publicly, with code:**
  - TRM, "Less is More: Recursive Reasoning with Tiny Networks" ([arXiv 2510.04871](https://arxiv.org/html/2510.04871v1), Oct 2025). With 7M parameters it scores 45% on ARC-AGI-1, above DeepSeek R1, o3-mini and Gemini 2.5 Pro. A probabilistic follow-up exists ([arXiv 2605.19943](https://arxiv.org/html/2605.19943)).
  - In 2026, task-specific small models beat frontier ones on narrow jobs: Forbes, June 2026; Cactus "Needle 3"; ["How 3B beats 100B+"](https://www.edenai.co/post/tiny-llms-that-beat-giant-models-how-efficient-3b-parameter-models-compete-with).
  - Crowds of models fail because their errors correlate. In ["Nine judges, two effective votes"](https://arxiv.org/html/2605.29800) (May 2026), 9 frontier judges gave about 2 independent votes, and the best single judge matched the panel.
- **Openings left in #5 are weak:**
  - TRM has research code but no product; its wins are puzzles, which have no real use.
  - Decorrelated crowds would be research-wow with no clear use.

## Update, 2026-11-21: can Engramscope be upgraded out of "just looking at DeepSeek's design"?

Every direction that would make it *build* something on the Engram memory already exists, with code or a paper:
- **Edit or write facts into the memory:**
  - EngramEdit ([arXiv 2610.10533](https://arxiv.org/abs/2610.10533), code at [ModalityDance/EngramEdit](https://github.com/ModalityDance/EngramEdit)).
  - "User as Engram", per-user facts written as local edits ([arXiv 2606.19172](https://arxiv.org/pdf/2606.19172v1)).
  - Sequential Engram editing and its forgetting ([arXiv 2607.24805](https://arxiv.org/html/2607.24805)).
  - A "delete 2 KB to forget a fact" tool would be a narrower version of these.
- **Shrink the memory using the "heads are interchangeable" result:**
  - GateRQ cuts table memory 7.8× without retraining, for about 1 PopQA point ([arXiv 2610.35985](https://arxiv.org/abs/2610.35985)).
  - 4-bit tables save about 75% ([arXiv 2610.31646](https://ar5iv.labs.arxiv.org/html/2610.31646)).
  - Dropping heads is the same move as MaltOE's width cut, which lost to lower-bit files at matched bytes (MaltOE G3: width 75% gave 1.16× perplexity; Q3_K_M at the same bytes gave 1.006×).
- **Stream or transplant the memory:** vLLM disk-backed tables ([PR #56757](https://github.com/vllm-project/vllm/pull/56757)); colibri; XMemTransfer.
- **What's left only improves the study:**
  - A behavioural lesion: run the real model with a phrase's essential traces deleted. This needs a rented machine.
  - The app.

Builder, 2026-11-21: **MaltOE was the best idea of the whole search**, "it just had a negative result".
