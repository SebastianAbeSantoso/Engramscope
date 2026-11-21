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
