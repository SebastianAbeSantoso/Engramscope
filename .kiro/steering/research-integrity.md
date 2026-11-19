---
inclusion: always
---

# Research Integrity Rules

Engramscope is judged partly on whether its claims hold up. These rules apply to code comments, docs, the README, UI text and the video script.

## Numbers

- Never invent results, sizes, speeds or thresholds.
- Every number is one of three kinds:
  - **measured:** from a script in this repo, with its output in `docs/results/` or `docs/findings/`;
  - **cited:** from a primary source, with a link, checked before quoting;
  - **estimate:** labelled as such.
- Thresholds (latency target, decode-index size budget, any pass mark) come from the builder at a checkpoint, never invented by Kiro.

## The benchmark

- The phrase set and its related-word sets are committed before any readout of those phrases. Never edit a set after seeing its readout; add a new phrase instead and note it.
- Always report controls (random rows, last-token embedding, shuffled phrase, layer 14) next to the result, with sample size and intervals.
- If layer 1 does not beat every control, say so plainly and do not claim the readout works.
- Examples in the UI or video are labelled as examples, never presented as the result.

## The lesion experiments

- The protocol (`bench/lesion-protocol.md`) is committed before any lesion run, and its interpretation rules are approved by the builder before the first run. After that, changes only go into its dated change log, never silently.
- Conclusions use the wording the approved rules produce. No summary label where a rule says "irregular" or the result is unclear.
- Every lesion result is labelled "trace lesion (readout)". Never describe it as what the model forgets; that needs the behavioural lesion.
- Report a trivial or null result as plainly as an interesting one.

## Claims

- State the claim exactly as in `product.md`.
- Engram, its hash and its tables are DeepSeek's. Remote and on-demand row access exist (cite them). Do not imply Engramscope invented either.
- Decoding by nearest input embeddings is an approximation. Say so wherever decoded words appear.
- Mark anything not verified from a primary source as unverified, and say how it will be checked.

## History and old ideas

- Every statement about Semon, Lashley or the history of the engram must come from `docs/theory-sources.md`. Do not describe them from memory.
- If a needed claim is not there, read the source first and add it, with where it was found.

## Sources

- Prefer papers, the model's own files and code, and official docs over blogs.
- Link sources inline. Paraphrase; never quote more than a short phrase.
