# Engramscope

**Read a model's memory without running it.**

DeepSeek-V4.1-Flash, a 284-billion-parameter open model, keeps two giant lookup tables called *Engram*: about 384 million rows each, 189 GiB together. Each row's address is a hash of the last few words of the text. Engramscope computes those addresses in your browser, fetches only the rows it needs from the public model files, and shows what they hold. No GPU, no download, no model run.

> **Status:** specification and feasibility phase for Kiroween 2026. The app doesn't exist yet; the [kill test](docs/findings/engram-killtest.md) does.

## What the kill test found

We reproduced the model's own hashing and read the layer-1 table directly:

| Phrase | Nearest words for the rows (layer 1) |
| --- | --- |
| the Eiffel Tower | Paris, 巴黎, France, Louvre |
| Mount Everest | Himalayas, Nepal, Bhutan, Tibet |
| Søren Kierkegaard | Nietzsche, Hegel, Heidegger, Kant, Wittgenstein |
| Yogyakarta | Jakarta, Indonesia, Bali |

These are examples. The measured result, on a 30-phrase pilot: a related word not contained in the phrase appeared in the top 10 for **22 of 30** phrases, against **0 of 30** for random rows and **8 of 30** for the phrase's own last-word embedding. Read as next-word predictions, the same rows score at chance. A pre-registered benchmark of at least 100 phrases is part of the plan.

## How it works

1. Tokenize the phrase with the model's tokenizer and normalize the tokens (case, accents, spacing) as the model does.
2. Hash the 2-, 3- and 4-word windows ending at a position, with 8 hash heads each: 24 row addresses per table.
3. Fetch those rows (256 bytes plus 8 scale bytes each) from the Hugging Face CDN with HTTP range requests: about 12 KB of a 189 GiB table.
4. Project the rows with the model's own value matrix, which gives the vector the model adds to its internal state at that layer.
5. Decode that vector into the nearest words in the model's vocabulary, next to a control decoded from random rows.

## What is DeepSeek's and what is ours

- **DeepSeek's:** Engram, its hashing and its tables ([Cheng et al., 2026, arXiv 2601.07372](https://arxiv.org/abs/2601.07372)); the model and its reference code (MIT).
- **Ours:** reading a released model's Engram rows directly, without running the model, over HTTP from a browser, and showing what they encode, with controls.
- **Not ours:** fetching rows remotely or on demand during inference already exists ([remote Engram over RDMA/TCP](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma); llama.cpp n-gram read mode). Other work edits ([EngramEdit](https://arxiv.org/abs/2610.10533)) or compresses ([CP](https://arxiv.org/abs/2610.21810), [GateRQ](https://arxiv.org/abs/2610.35985)) the tables.

## Why "engram"

Richard Semon coined *engram* for a physical memory trace in 1904. Karl Lashley spent about 30 years removing parts of rats' brains to find where a memory lives, and concluded in 1950 that he could not localize it. In this model an engram has an exact address you can compute from the text. (Sources: [`docs/theory-sources.md`](docs/theory-sources.md). This is an analogy; it says nothing about brains.)

## Limitations

- Decoding by nearest input embeddings is an approximation; it assumes the rows' projection lives in the model's embedding space, which is plausible at layer 1 and not proven.
- The model mixes each row in through a gate that depends on its internal state. Engramscope can't compute that without running the model, so a decoded row shows what is stored, not how strongly it was used.
- Layer 14 mostly reads as noise with this decoding.

## Repository

- [`.kiro/specs/engramscope/`](.kiro/specs/engramscope/): requirements, design, tasks.
- [`spikes/engram_readout/`](spikes/engram_readout/): the kill-test scripts.
- [`docs/`](docs/): findings, history sources, rules check, submission tracker.

Built with [Kiro](https://kiro.dev) for [Kiroween 2026](https://kiroween.devpost.com/).

## License

MIT, see [LICENSE](LICENSE). DeepSeek-V4.1-Flash is MIT-licensed by DeepSeek; Engramscope does not redistribute its weights.
