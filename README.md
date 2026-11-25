# Engramscope

**Lashley's search for the engram, rerun on a memory where every trace has an address.**

In 1950 Karl Lashley summed up about 30 years of removing parts of rats' cortex to find where a learned habit was stored. He could not localize it. The loss grew with how much cortex was removed (*mass action*), and within a functional area the parts were equivalent for the trace (*equipotentiality*). Richard Semon, who coined *engram* in 1904, said a trace is retrieved when part of the original stimulus recurs. ([Sources with page numbers](docs/theory-sources.md).)

DeepSeek-V4.1-Flash, a 284-billion-parameter open model, has a memory called *Engram*: two lookup tables of about 384 million rows each, 189 GiB together. For every position in a text the model fetches 24 rows per table, one per n-gram size (2, 3 or 4 tokens) and hash head (8). Each row's address is a hash of the last few tokens. So every trace of a phrase can be found, read and removed exactly, which was never possible in a brain.

Engramscope asks Lashley's questions of this memory:
- **Mass action:** does what a phrase's traces store fade gradually as more of them are removed, or collapse at some point?
- **Equipotentiality:** at equal lesion size, does it matter *which* traces go?
- **Semon's partial cue:** does part of the cue (one n-gram size, or the phrase after different words) still retrieve it?

The experiments are pre-registered in [`bench/lesion-protocol.md`](bench/lesion-protocol.md), committed before any run. They are **trace lesions**: they remove rows from what the model would add to its internal state, and decode the rest. They show how a stored concept is spread across traces, not what the running model forgets. They say nothing about brains.

> **Status (2026-11-25):** the pre-registered readout benchmark and lesion experiments have run; the results are in [`docs/results/`](docs/results/) and under the builder's review ([summary](docs/findings/experiments-2026-11-19.md)). A side test of DeepSeek's memory design on frozen SmolLM2 models, against LoRA, is in [`spikes/own_memory/RESULTS.md`](spikes/own_memory/RESULTS.md). The web app is next.

## The instrument: reading a trace without running the model

Engramscope computes the model's own addresses in the browser, fetches only the needed rows from the public model files with HTTP range requests (about 12 KB per phrase), projects them the way the model does, and decodes them into words. The kill test showed that this reads something real:

| Phrase | Nearest words for the rows (layer 1) |
| --- | --- |
| the Eiffel Tower | Paris, 巴黎, France, Louvre |
| Mount Everest | Himalayas, Nepal, Bhutan, Tibet |
| Søren Kierkegaard | Nietzsche, Hegel, Heidegger, Kant, Wittgenstein |
| Yogyakarta | Jakarta, Indonesia, Bali |

These are examples. The measured result, on a 30-phrase pilot: a related word not contained in the phrase appeared in the top 10 for **22 of 30** phrases, against **0 of 30** for random rows and **8 of 30** for the phrase's own last-word embedding. Read as next-word predictions, the same rows score at chance. A pre-registered benchmark of 123 phrases ([`bench/`](bench/)) comes before the lesion runs.

## How it works

1. Tokenize the phrase with the model's tokenizer and normalize the tokens (case, accents, spacing) as the model does.
2. Hash the 2-, 3- and 4-word windows ending at a position, with 8 hash heads each: 24 row addresses per table.
3. Fetch those rows (256 bytes plus 8 scale bytes each) from the Hugging Face CDN with HTTP range requests: about 12 KB of a 189 GiB table.
4. Project the rows with the model's own value matrix, which gives the vector the model adds to its internal state at that layer.
5. Decode that vector into the nearest words in the model's vocabulary, next to a control decoded from random rows.

## What is DeepSeek's and what is ours

- **DeepSeek's:** Engram, its hashing and its tables ([Cheng et al., 2026, arXiv 2601.07372](https://arxiv.org/abs/2601.07372)); the model and its reference code (MIT).
- **Lashley's and Semon's:** the questions and the lesion method (1950; 1904/1921).
- **Ours:** rerunning those experiments as pre-registered trace lesions on a released model's Engram memory, and the instrument they need: reading the rows directly, without running the model, over HTTP from a browser, with controls.
- **Not ours:** removing the whole Engram memory from the running model has been measured ([GateRQ, arXiv 2610.35985](https://arxiv.org/abs/2610.35985): −36.8 PopQA points).
- **Not ours:** fetching rows remotely or on demand during inference already exists ([remote Engram over RDMA/TCP](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma); llama.cpp n-gram read mode). Other work edits ([EngramEdit](https://arxiv.org/abs/2610.10533)) or compresses ([CP](https://arxiv.org/abs/2610.21810), [GateRQ](https://arxiv.org/abs/2610.35985)) the tables.

## Limitations

- Decoding by nearest input embeddings is an approximation; it assumes the rows' projection lives in the model's embedding space, which is plausible at layer 1 and not proven.
- The model mixes each row in through a gate that depends on its internal state. Engramscope can't compute that without running the model, so a decoded row shows what is stored, not how strongly it was used.
- Layer 14 mostly reads as noise with this decoding.
- Trace lesions act on the readout, not on the running model. The model also uses the rows for its gate, and later layers may compensate. A behavioural lesion on rented hardware is a stretch goal.

## Repository

- [`.kiro/specs/engramscope/`](.kiro/specs/engramscope/): requirements, design, tasks.
- [`bench/`](bench/): the pre-registered benchmark phrases and lesion protocol.
- [`spikes/`](spikes/): the kill test and feasibility-gate scripts.
- [`docs/`](docs/): findings, history sources, rules check, submission tracker.

Built with [Kiro](https://kiro.dev) for [Kiroween 2026](https://kiroween.devpost.com/).

## License

MIT, see [LICENSE](LICENSE). DeepSeek-V4.1-Flash is MIT-licensed by DeepSeek; Engramscope does not redistribute its weights.
