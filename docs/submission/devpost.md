# Devpost text (draft, for the builder to check)

## Project title (60 characters max)

Engramscope: where is a memory stored?

## Elevator pitch (200 characters max)

Lashley's lesion experiments rerun on a language model's memory, in your browser, with every result pre-registered.

## About the project (Markdown)

## Inspiration

In 1950 Karl Lashley ended about 30 years of removing parts of rats' cortex without finding where a learned habit was stored. Richard Semon had coined *engram* in 1904 for the lasting change a stimulus leaves behind. I wanted to rerun Lashley's questions on a memory where every trace has an address, which no brain offers.

An open-weight frontier model has such a memory. Its *Engram* tables hold 24 traces per phrase, and each trace's address is a hash of the last few words, so the traces can be found, read and removed exactly. This project started from a kill test in an earlier project of mine, MaltOE, which stopped on its own negative results. The Engram kill test showed the traces carry readable concepts, and that is what Engramscope was built on.

## How we built it

Built in Kiro, spec first. The spec's feasibility gates measured the browser decode index, tokenizer parity, browser fetch time and prior art before any app code.

- **Python reference:** the model's addressing, FP8 decoding and readout, checked by property-based tests against the model's reference code.
- **Pre-registration:** the 123 benchmark phrases and the lesion protocol were committed before any run. Code refuses to run on any other version.
- **Browser instrument:** a BigInt port of the hash, the model's own tokenizer, HTTP range requests for the rows, FP8 decoding, and a compressed readout in a Web Worker. Vitest checks it against Python test vectors: 58 phrases, every token id and address exact.
- **App:** React, Vite and TypeScript, deployed to GitHub Pages by a GitHub Actions workflow that builds the browser assets from the pinned model revision.

## Challenges we ran into

- **The planned decode index failed.** A 33 MB projection kept only 20% of the exact top-10 words. We fitted the projection to the readout instead and shipped a 92 MB index, which the browser downloads once.
- **Layer 14 reads as noise** with this decoding, so the app labels it unclear.
- **The browser decoding is approximate**, and the app says so next to every decoded word.
- **A side test went against our expectation.** A small Engram-style memory lost to LoRA at 135M, and at 1.7B it narrowly passed its rule while a better-tuned LoRA beat it. We report both results.
- **Limited compute.** The side test ran on a rented GPU that I started from my phone. It cost $2.20.

## Accomplishments that we're proud of

- The lesion predictions, written before the run, held: no head is special, and some n-gram size is.
- The instrument is checked, not eyeballed. Addresses match the model's reference code exactly, and layer 1 beats every control on 123 pre-registered phrases: 61.8% against 0% for random rows.
- The app runs with no server and no GPU, and the experiment figures come straight from the raw result files.
- We kept the negative side-test result in the write-up and in the app.

## What we learned

- Pre-registration is worth its cost: the rules decided the labels, including the ones we did not want.
- A negative result is still a result when the rules were fixed in advance. It shaped what the project claims.
- Trace lesions show how a concept is spread across traces. They do not show what the running model forgets, and the difference matters.
- Spec-first with gates in Kiro caught a failed plan early, before it reached the app.

## What's next for Engramscope: where is a memory stored?

- Run the behavioural lesion: zero the traces in the running model on rented hardware and check what it can no longer recall. This is the stretch goal in the spec, and it needs a pre-registered addendum first.
- Investigate layer 14, and single words, which currently have no readable trace.
- Extend the benchmark beyond 123 phrases, and add a second, independent decoding check.

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
