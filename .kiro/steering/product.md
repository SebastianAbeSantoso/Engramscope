---
inclusion: always
---

# Product: Engramscope

Engramscope reruns Karl Lashley's search for the engram (1950: mass action, equipotentiality) and Richard Semon's partial-cue retrieval (ecphory) on DeepSeek-V4.1-Flash's Engram memory, where every trace has an address. The experiments are pre-registered trace lesions (`bench/lesion-protocol.md`): remove some of a phrase's 24 traces and measure what survives.

The instrument: the browser computes the model's own hash addresses, fetches the rows (about 12 KB per phrase, both layers) from the public model files with HTTP range requests, projects them the way the model does, and decodes them into words, without running the model. Every result is shown next to a control.

**The experiments are the product; the readout is the instrument.** Lead with Lashley's question in the README, app and video.

Kiroween 2026, Resurrection category. Solo student builder. Deadline December 4, 2026, 2:00 PM PT.

## Claim (exact wording; do not widen)

Engramscope's contribution is limited to (1) Lashley's lesion experiments and Semon's partial-cue retrieval, rerun as pre-registered trace lesions on a released model's Engram memory, and (2) the instrument they need: reading the Engram rows directly, without running the model, over HTTP from a browser, and decoding them, with controls that show the readout is not noise. Trace lesions show how a stored concept is spread across traces, not what the running model forgets. Nothing here is a claim about biological memory. Engram itself, its hashing and tables are DeepSeek's (Cheng et al., arXiv 2601.07372). Remote and on-demand row access for inference already exist and are cited, not claimed.

## Side test (Req 13)

The own-memory test asks whether DeepSeek's memory design helps a frozen model it wasn't trained with, against LoRA, under pre-registered rules: KILL at 135M, a narrow PASS at 1.7B. It comes after the lesion experiments in every presentation. It is not part of the claim: adding Engram-style memory to a frozen model is prior work (Engram Adapter, TF-Engram, NGM), so never call it new or first. Its value is the controlled comparison and reporting both outcomes.

## Vocabulary

- **Engram table, row, address, compressed id, value vector, readout, decode index, control, trace, trace lesion, behavioural lesion, retention:** as defined in the requirements glossary. Use these terms consistently.
- Say "the model remembers" or "the rows point at" only for results backed by the benchmark. In the UI, label examples as examples.

## Users

- People curious about how language models work inside.
- Interpretability and architecture researchers looking at memory/lookup layers.
- Judges, who should be able to verify the claim from the app and README alone.

## What it is not

- Not an inference engine and not a chatbot.
- Not a claim that the rows are all the model knows, or that the decoded words are what the model used at full strength (the gate depends on the residual stream, which Engramscope does not compute).
