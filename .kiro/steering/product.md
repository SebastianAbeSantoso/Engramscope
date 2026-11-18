---
inclusion: always
---

# Product: Engramscope

Engramscope reads what DeepSeek-V4.1-Flash remembers about a phrase directly from its Engram lookup tables (layers 1 and 14, 189 GiB), without running the model. The browser computes the model's own hash addresses, fetches about 6 KB of rows plus scales per layer from the public model files with HTTP range requests, projects them the way the model does, and decodes them into words. Every result is shown next to a control.

Kiroween 2026, Resurrection category (proposed). Solo student builder. Deadline December 4, 2026, 2:00 PM PT.

## Claim (exact wording; do not widen)

Engramscope's contribution is limited to reading a released model's Engram rows directly, without running the model, over HTTP from a browser, and showing what they encode, with controls that show the readout is not noise. Engram itself, its hashing and tables are DeepSeek's (Cheng et al., arXiv 2601.07372). Remote and on-demand row access for inference already exist and are cited, not claimed.

## Vocabulary

- **Engram table, row, address, compressed id, value vector, readout, decode index, control, lesion:** as defined in the requirements glossary. Use these terms consistently.
- Say "the model remembers" or "the rows point at" only for results backed by the benchmark. In the UI, label examples as examples.

## Users

- People curious about how language models work inside.
- Interpretability and architecture researchers looking at memory/lookup layers.
- Judges, who should be able to verify the claim from the app and README alone.

## What it is not

- Not an inference engine and not a chatbot.
- Not a claim that the rows are all the model knows, or that the decoded words are what the model used at full strength (the gate depends on the residual stream, which Engramscope does not compute).
