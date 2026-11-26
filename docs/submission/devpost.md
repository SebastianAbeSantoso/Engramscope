# Devpost text (draft, for the builder to check)

## Project title (60 characters max)

Engramscope: where is a memory stored?

## Elevator pitch (200 characters max)

Lashley's lesion experiments rerun on a language model's memory, in your browser, with every result pre-registered.

## About the project (Markdown)

## Inspiration

In 1950 Karl Lashley ended about 30 years of removing parts of rats' cortex without finding where a learned habit was stored. Richard Semon had coined *engram* in 1904. I wanted to rerun Lashley's questions on a memory where every trace has an address, which no brain offers.

An open-weight frontier model has such a memory. Its *Engram* tables hold 24 traces per phrase, and each trace's address is a hash of the last few words. So the traces can be found, read and removed exactly.

## What it does

- **Reads a memory in the browser.** It computes the model's addresses, fetches about 12 KB of rows from the public model files with HTTP range requests, and decodes them into words, next to a random-rows control. It needs no server and no GPU.
- **Runs the lesion experiments.** Mass action, equipotentiality and Semon's partial cue, pre-registered before any run.
- **Lets you remove traces yourself** in a live 3 × 8 grid, labelled as exploration.

## Findings (layer 1, 76 phrases with a readable concept)

- **Mass action:** half the concepts are gone only after 19.0 of 24 traces are removed [18.0, 19.8].
- **Equipotentiality:** removing any one hash head costs the same as removing 3 random traces. Removing the 8 two-word traces leaves **10.5%** of concepts, against 87.8% for 8 random traces.
- **Semon's partial cue:** the two-word traces alone retrieve the concept in 94.7% of phrases.
- **The instrument:** on 123 pre-registered phrases, layer 1 gives a related word in the top 10 for 61.8% (95% interval 53.7–69.9). Random rows score 0%.

Both predictions written before the run held. The rules produce the parallel with Lashley's 1950 conclusion: some regions are essential, and within a region the parts are interchangeable.

## What this does not show

- These are **trace lesions**. They show how a concept is spread across traces. They do not show what the running model forgets, and they say nothing about brains.
- The browser's decoding is an approximation, which is labelled in the app.
- **A side test went against our expectation.** We added the same kind of memory to frozen small models and compared it with LoRA. At 135M, LoRA won. At 1.7B the memory narrowly passed its pre-registered rule, but a LoRA with a lower learning rate, run afterwards, did better. Adding such a memory to a frozen model is already done by other groups. We report it as a controlled comparison, not a new method.

## Built with

Kiro (specs, steering, hooks, property-based tests), React, Vite, TypeScript, Web Workers, @huggingface/tokenizers, Python, NumPy, Hugging Face (model files over HTTP range requests), GitHub Pages, GitHub Actions, Vitest, pytest, Hypothesis.

## Pre-existing work used

See the "Pre-existing work used" section of the README: the model and its reference code (MIT), the tokenizer and libraries listed there, the SmolLM2 and WikiText-103 data used for the side test, Semon's and Lashley's texts, and the MaltOE kill test (first run in the builder's earlier project on 2026-11-19, then copied here).

## Try it out

- App: https://sebastianabesantoso.github.io/Engramscope/
- Source: https://github.com/SebastianAbeSantoso/Engramscope
- Video: (add the YouTube link after upload)

## Testing instructions (judges only)

No login. Open the app and type a phrase, or use the examples. Live lesion: open the Live lesion page, choose "the Eiffel Tower", press **Remove the 2-word traces**. The first visit downloads a 92 MB index once; expect about 20 seconds before words appear.

## Main technology requirement: how Kiro was used

- **Specs:** `.kiro/specs/engramscope/` has requirements (with a claims boundary), design and tasks. Feasibility gates ran before any app code. One gate failed its plan, and Kiro put the choice back to the builder.
- **Steering:** `.kiro/steering/` has five files, including a research-integrity rule (no invented numbers, pre-registration, history only from read sources).
- **Hooks:** a post-task hook runs the Python and web linters and tests (`.kiro/hooks/`).
- **Property-based tests:** Hypothesis checks the Python hash against the model's reference code; Vitest checks the browser hash, tokenizer and readout against Python test vectors.
- **Sub-agents:** research and codebase sub-agents for prior-art searches and for mapping the reference code before porting it to the browser.
- **Pre-registration enforcement:** the benchmark and lesion runners refuse to run on files that differ from their registered commits.

The full account is in `docs/submission/kiro-writeup.md`.
