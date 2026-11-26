# Engramscope

**Lashley's search for the engram, rerun on a memory where every trace has an address.**

**Try it:** [sebastianabesantoso.github.io/Engramscope](https://sebastianabesantoso.github.io/Engramscope/). It runs in your browser, with no server and no GPU needed.

Kiroween 2026, **Resurrection** category. Built with [Kiro](https://kiro.dev), spec first: see [How Kiro was used](#how-kiro-was-used).

## The question

In 1950 Karl Lashley summed up about 30 years of removing parts of rats' cortex to find where a learned habit was stored. He could not localize it: "It is not possible to demonstrate the isolated localization of a memory trace anywhere within the nervous system" (1950, p. 478). The loss grew with how much cortex was removed (*mass action*), and within a functional area the parts were equivalent for the trace (*equipotentiality*). Richard Semon, who coined *engram* in 1904, said a trace is retrieved when part of the original stimulus recurs (1921, p. 138). [Sources with page numbers](docs/theory-sources.md).

DeepSeek-V4.1-Flash, a 284-billion-parameter open model (MIT), has a memory its developers named *Engram*. It consists of two lookup tables of about 384 million rows each, 189 GiB together. For every position in a text, the model fetches 24 rows per table, one for each n-gram size (2, 3 or 4 tokens) and hash head (8). Each row's address is a hash of the last few tokens. So every trace of a phrase can be found, read and removed exactly, which was never possible in a brain.

Engramscope asks Lashley's and Semon's questions of this memory:
- **Mass action:** does a phrase's stored concept fade gradually as more of its traces are removed, or collapse at some point?
- **Equipotentiality:** at equal lesion size, does it matter *which* traces go?
- **Semon's partial cue:** does part of the cue (one n-gram size, or the phrase after different words) still retrieve it?

## What we found

We reran Karl Lashley's lesion experiments on a language model's memory, pre-registered. Each phrase is stored in 24 traces. Half of the stored concepts are still readable after 19 of the 24 are removed at random. Any of the 8 hash heads can stand in for any other, but the two-word traces cannot be replaced: remove those 8 and the concept is gone in 9 of 10 phrases. That is the pattern Lashley described in rat cortex in 1950: some regions are essential, and within a region the parts are interchangeable. It describes this model's memory, not brains.

| Question | Result (layer 1, 76 phrases with a readable concept; 95% intervals) | Label from the pre-registered rules |
| --- | --- | --- |
| Mass action | Half the concepts are gone at **19.0** traces removed [18.0, 19.8]. Removing 8 at random leaves 87.8% | "late collapse: the traces stand in for each other until most are gone" |
| Equipotentiality, heads | Removing any one head (3 traces) costs the same as 3 random traces: differences −4.0 to +3.9 points, every corrected interval includes zero | "equipotential across heads" |
| Equipotentiality, n-gram sizes | Removing the 8 two-word traces leaves **10.5%**, against 87.8% for 8 random traces | "not equipotential across n-gram sizes" |
| Semon's partial cue | The two-word traces alone retrieve the concept in 94.7% of phrases. After five different lead-ins, it survives in 92–93% | above chance; context: no label by design |

- **Both predictions written before the run held:** no head is special, and some n-gram size is.
- **Pre-registration:** the protocol ([`bench/lesion-protocol.md`](bench/lesion-protocol.md)) and the 123 phrases were committed before any run, and code guards refuse to run on any other version.
- **Sources:** all numbers come from [`docs/results/lesion-2026-11-19.md`](docs/results/lesion-2026-11-19.md). The summary with caveats is in [`docs/findings/experiments-2026-11-19.md`](docs/findings/experiments-2026-11-19.md).

These are **trace lesions**: they remove rows from what the model would add to its internal state, and decode the rest. They show how a stored concept is spread across traces, not what the running model forgets.

## The instrument: reading a trace without running the model

The experiments need a way to read a trace. Engramscope computes the model's own addresses, fetches only the needed rows from the public model files with HTTP range requests (about 12 KB per phrase), projects them the way the model does, and decodes them into words. A pre-registered benchmark checks that this reads something real. For 123 phrases, related-word sets were written before any readout. A hit means a related word not contained in the phrase appears in the top 10.

| Readout | hit@10 (95% interval) |
| --- | --- |
| **Engram layer 1** | **61.8%** [53.7, 69.9] |
| the phrase's last-word embedding | 32.5% |
| Engram layer 14 | 2.4% |
| random rows | 0.0% |
| another phrase's rows | 0.0% |

Layer 1 beats every control ([`docs/results/bench-2026-11-19.md`](docs/results/bench-2026-11-19.md)). Some examples, labelled as examples:
- "the Kremlin" → Kremlin, Moscow, Putin, 斯大林 (Stalin);
- "the Leaning Tower of Pisa" → Bologna, Venice, Florence, Italy, Galileo;
- "the Dutch East Indies" → Indonesia, 印尼, Sumatra.

## The app

- **Experiments:** the recorded results, one panel per question, each naming its source and page. Every figure is read from `docs/results/`.
- **Live lesion:** a phrase's 24 traces as a 3 × 8 grid. Switch traces off, or use presets that match the protocol's conditions and show the recorded result next to what you see. Labelled as exploration.
- **Read a memory:** the decoded words for both layers next to a random-rows control, plus the whole pipeline. It shows tokens, compressed ids, windows, the 24 addresses with byte offsets, each trace decoded alone, and the bytes and time taken.
- **About:** history, attribution, the benchmark, limitations, sources.

The browser decodes with a compressed index: about 92 MB, downloaded once and cached. It loads in four parts and the words sharpen as each part arrives. On the benchmark it scores the same as the exact readout (61.8%) from half of the index on. Its word lists share 57% of their top 10 with the exact readout ([`docs/results/decode-index-2026-11-25.md`](docs/results/decode-index-2026-11-25.md)). Every number in the experiments comes from the exact readout in Python.

Single words have no readable trace in this memory, so multi-word names work best. Adding `?generic` to the URL replaces the model's name with a generic label, for recording the video.

## How it works

1. Tokenize the phrase with the model's tokenizer, and map tokens that differ only in case, accents or spacing to one compressed id, as the model does.
2. Hash the 2-, 3- and 4-token windows ending at a position, with 8 hash heads each: 24 row addresses per table. This is bit-exact with the model's reference code, checked by tests in Python and in the browser.
3. Fetch those rows (256 FP8 values plus 8 scales each) from the Hugging Face CDN with HTTP range requests, at a pinned model revision.
4. Project the rows with the model's value matrix. The result is a sum of 24 per-row contributions, which is what makes trace lesions exact arithmetic.
5. Decode the vector into the words whose input embeddings are most similar, next to a control decoded from random rows.

## What is DeepSeek's and what is ours

- **DeepSeek's:** Engram, its hashing and its tables ([Cheng et al., 2026, arXiv 2601.07372](https://arxiv.org/abs/2601.07372)); the model and its reference code (MIT).
- **Lashley's and Semon's:** the questions and the lesion method (1950; 1904/1921).
- **Ours:** rerunning those experiments as pre-registered trace lesions on a released model's Engram memory, and the instrument they need: reading the rows directly, without running the model, over HTTP from a browser, with controls.
- **Not ours, from related work:**
  - removing the whole Engram memory from the running model has been measured ([GateRQ, arXiv 2610.35985](https://arxiv.org/abs/2610.35985): −36.8 PopQA points);
  - fetching rows remotely or on demand during inference already exists ([remote Engram over RDMA/TCP](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma); llama.cpp n-gram read mode);
  - other work edits ([EngramEdit](https://arxiv.org/abs/2610.10533)) or compresses ([CP](https://arxiv.org/abs/2610.21810), [GateRQ](https://arxiv.org/abs/2610.35985)) the tables.

Nothing here is a claim about biological memory. The experiments borrow Lashley's and Semon's questions and methods, not their conclusions.

## A side test: does this memory design help another model?

We added an Engram-style memory, with the same hashing, to frozen SmolLM2 models and trained it against LoRA, under rules fixed before each run.
- **135M:** LoRA won. The pre-registered outcome was KILL.
- **1.7B:** the memory narrowly passed the rule (8.868 vs 8.949 perplexity, base 11.129). But a LoRA with a lower learning rate, run afterwards as an exploratory check, reached 8.235.

So a well-tuned LoRA does better at both sizes. Adding Engram-style memory to a frozen model isn't new either (Engram Adapter, TF-Engram, NGM, Engram-PEFT, TinyEngram). The test is reported for its controlled comparison, not as a new method. Details: [`spikes/own_memory/RESULTS.md`](spikes/own_memory/RESULTS.md).

## Limitations

- **Decoding is an approximation.** Decoding by nearest input embeddings assumes the rows' projection lives in the model's embedding space, which is plausible at layer 1 and not proven. The browser's compressed index adds a further approximation (above).
- **The gate isn't computed.** The model mixes the rows in through a gate that depends on its internal state, and Engramscope can't compute that without running the model. A decoded row shows what the rows point at, not how strongly the model used them.
- **Readout, not behaviour.** Trace lesions act on the readout, not on the running model. The model also uses the rows for its gate, and later layers may compensate.
- **Layer 14** mostly reads as noise with this decoding. Only 3 phrases have a readable concept there, so its lesion results get numbers and no labels.
- **Padding.** Short phrases are padded before their start, which favours the two-word traces. The pre-registered split by length shows the same pattern in phrases of 4 or more tokens, with a wide interval (12 phrases).

## How Kiro was used

Engramscope was built in Kiro Web, spec first. The full account is in [`docs/submission/kiro-writeup.md`](docs/submission/kiro-writeup.md). In short:
- **A spec with feasibility gates before any app code** ([`.kiro/specs/engramscope/`](.kiro/specs/engramscope/)). The gates measured the decode index, tokenizer parity, browser fetch time and prior art, and one of them failed its plan and went back to the builder as a choice.
- **Steering** ([`.kiro/steering/`](.kiro/steering/)), including a research-integrity rule: no invented numbers, pre-registration before runs, history only from read sources.
- **Hooks** that run the linters and tests after each spec task.
- **Property-based tests** of the hash against DeepSeek's reference code.
- **Pre-registration enforced by code:** the lesion and benchmark runners refuse to run on unregistered files.

## Run it yourself

```sh
uv sync && uv run pytest -q                       # Python reference: hashing, FP8, readout, lesion conditions
uv run python -m engramscope.bench                 # readout benchmark   -> docs/results/bench-<date>.*
uv run python -m engramscope.lesion                # lesion experiments  -> docs/results/lesion-<date>.*
uv run python -m engramscope.build_assets          # browser assets      -> web/public/assets/<revision>/
cd web && npm ci && npm test && npm run build      # web app (tests check the browser against the Python test vectors)
```

The benchmark, lesions and asset build download parts of the model files. The asset build needs the 1.3 GB embedding table, and it runs on a CPU with about 16 GB of RAM. On every push to `main`, [`pages.yml`](.github/workflows/pages.yml) builds the assets from the pinned revision and deploys the app. No large binaries are committed.

## Repository

- [`.kiro/`](.kiro/): spec (requirements, design, tasks), steering, hooks.
- [`tools/engramscope/`](tools/engramscope/): Python reference (hashing, model-file access, FP8, readout, benchmark, lesions, asset build); tests in [`tests/`](tests/).
- [`web/`](web/): the app (React, Vite, TypeScript); the browser instrument is in [`web/src/core/`](web/src/core/).
- [`bench/`](bench/): the pre-registered phrases, benchmark plan and lesion protocol.
- [`docs/`](docs/): results, findings, history sources, rules check, submission tracker.
- [`spikes/`](spikes/): the kill test, gate scripts and the side test.

## Pre-existing work used

- **Model:** DeepSeek-V4.1-Flash and its reference code (MIT), read at revision `314113f8`. Engramscope doesn't redistribute its weights. The browser assets are derived from the embedding and value matrices.
- **Libraries:**
  - browser: [`@huggingface/tokenizers`](https://github.com/huggingface/tokenizers.js) (Apache-2.0), React and Vite (MIT);
  - Python: numpy, `tokenizers`, `requests`, `sympy`.
- **Side test:** SmolLM2-135M and 1.7B (Apache-2.0), WikiText-103 (CC BY-SA), transformers and peft.
- **History:** Semon (1904, 1921) and Lashley (1950), cited from the editions in [`docs/theory-sources.md`](docs/theory-sources.md).
- **MaltOE:** the builder's first Kiroween project, also built during the submission period ([repo](https://github.com/SebastianAbeSantoso/MaltOE)). The Engram kill test was first run there on 2026-11-19 and copied into this repository.

## License

MIT, see [LICENSE](LICENSE).
