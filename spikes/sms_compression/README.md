# Kill test: language-model SMS compression (2026-11-20)

The idea: both phones run the same small language model. Only the parts of a message the model can't predict are sent (arithmetic coding). This revives telegraph code books, such as [Bentley's Complete Phrase Code](https://archive.org/details/bentleyscomplete00bent), where one code word stood for a whole phrase.

`PREREG.md` (the criteria) was written before any measurement.

Data (not in the repo; download to run):
- [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) "ham" messages (CC BY 4.0).
- [IndoNLU EmoT](https://github.com/IndoNLP/indonlu) tweets (Apache-2.0 repository).
- 200 of each, seed 20261120.

Compressor tested: [ts_sms](https://bellard.org/ts_sms/) 2024-12-26.

## Results (median bits per character; one SMS carries 1,120 bits)

| Method | English SMS | Indonesian informal |
| --- | ---: | ---: |
| plain SMS (GSM-7 or UCS-2) | 7.00 | 7.00 |
| zlib | 8.00 | 5.67 |
| brotli | 8.18 | 5.80 |
| ts_sms (RWKV 169M, real bytes, round trip checked) | **2.21** | 2.85 |
| SmolLM2-360M (ideal arithmetic-coding length) | 2.54 | 2.60 |
| Qwen2.5-0.5B (ideal arithmetic-coding length) | 2.70 | **2.43** |

- ts_sms round trips: 0 failures in 400.
- Speed on the sandbox CPU: 0.15–0.26 s per message for scoring with the small models; ts_sms took 1.4–3.9 s per message for compress plus decompress.
- Share of messages that fit in one SMS:
  - English: 95% plain, 100% with any language-model method.
  - Indonesian: 30% plain, 100% with any language-model method.

All four kill rules pass:
- **K1:** English 2.21 ≤ 3.5.
- **K2:** Indonesian 2.43 ≤ 4.67.
- **K3:** the best non-AI option is plain text for English (7.00) and zlib for Indonesian (5.67). The language-model methods use well under 2/3 of those bits.
- **K4:** 0 round-trip failures.

## Novelty (searches 2026-11-20)

- **The technique exists as a tool:** [ts_sms](https://bellard.org/ts_sms/), Fabrice Bellard, Dec 2024. It is a command-line program that compresses short messages with a language model. Bellard reports identical results across CPUs and GPUs for its RWKV model. The library is closed.
- **Research:**
  - [Real-time text transmission via LLM-based entropy coding](https://arxiv.org/html/2605.01991v1) (2026).
  - [Haiku to Opus in 10 bits](https://arxiv.org/html/2604.02343v2) (2026). It reports that domain-adapted LoRA adapters roughly double the gain on LLM-generated text.
- **Products:**
  - [Apple's Messages via satellite](https://support.apple.com/en-us/125127) compresses messages with a custom protocol. Apple does not say it is a language model.
  - [Garmin inReach Messenger Plus](https://www.garmin.com/en-GB/blog/the-evolution-of-garmin-inreach-messenger/) raised its limit from 160 to 1,600 characters.
  - Meshtastic tried classic short-text compression (Unishox2) in 2023 ([issue #2833](https://github.com/meshtastic/firmware/issues/2833)). Its router code still has "TODO: Allow modules to opt into compression".
  - The closest project found is [VaaniSetu](https://github.com/shenmok/VaaniSetu/blob/main/docs/Architecture2.md), an offline mesh messenger whose design mentions "semantic message compression".
  - No messaging app was found that uses language-model compression for SMS, satellite or mesh messages.
- **Determinism:** standard WebAssembly floating point is deterministic except for NaN bit patterns ([WebAssembly design: Nondeterminism](https://github.com/WebAssembly/design/blob/main/Nondeterminism.md)). Relaxed SIMD is the exception to avoid. So a browser engine built that way can give identical probabilities on every device. Not yet tested here.
