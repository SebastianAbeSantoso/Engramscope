# Kill test: language-model message compression (written before any measurement, 2026-11-20)

Question: does compressing short everyday messages with a small language model fit enough extra text into one SMS to be worth a product?

Data (fixed seed 20261120, sampled before any compression is run):
- English: 200 random "ham" (non-spam) messages from the UCI SMS Spam Collection (real SMS; CC BY 4.0).
- Indonesian: 200 random tweets from the IndoNLU EmoT set (informal Indonesian; Apache-2.0 repo). Tweets stand in for chat messages; this is a known limitation.

Methods, per message:
- Plain SMS: 7 bits per character if every character is in the GSM-7 alphabet, otherwise 16 bits (UCS-2).
- Classical: zlib (raw deflate, level 9) and brotli (quality 11, its built-in dictionary helps short text).
- ts_sms (Bellard, 2024-12-26, RWKV 169M q8): real compressed bytes, with a full decompression check.
- Small open models (Qwen2.5-0.5B, SmolLM2-360M): ideal arithmetic-coding length = sum of -log2 p(token) + 2 bits, with the fixed context "Message: ". No tuning on the data.

Metric: bits per character (bpc) of the original message; one SMS carries 1,120 bits (140 bytes).

Kill rules:
- K1: for English, the best language-model method has median bpc above 3.5 (less than 2x a plain GSM-7 SMS) -> kill.
- K2: for Indonesian, the best language-model method has median bpc above 4.67 (less than 1.5x) -> kill the Indonesian angle.
- K3: at the median, the best language-model method needs more than 2/3 of the bits of the best classical method -> "no need for AI", kill.
- K4: any ts_sms round trip that does not reproduce the message exactly -> report; kill if above 1%.
Also reported, not used to kill: share of messages that fit in one SMS before and after, and time per message on the sandbox CPU.
