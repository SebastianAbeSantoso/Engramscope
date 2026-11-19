# Gate G3: browser fetch latency

**Status: done.** Sandbox reference and the builder's run, 2026-11-19.

Page: `spikes/g3_fetch/index.html` (version 2). It fetches 96 byte ranges per phrase (48 rows of 256 bytes plus their 8-byte scales, both Engram layers, 12,672 bytes in total) from the Hugging Face CDN, with the browser cache off. It checks every byte against SHA-256 hashes computed in Python.

## Sandbox reference (2026-11-19)

Headless Chrome 151 in the Kiro sandbox (AWS us-east-1), page loaded into an `https://example.com` origin. CDN host: `us.aws.cdn.hf.co`. Resolving each shard took 31–36 ms.

| Phrase | CDN cache | Concurrency | Total ms | Median request ms | p95 request ms | Bytes OK |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| a lighthouse keeper | cold | 8 | 1426 | 121 | 190 | yes |
| the Andromeda galaxy | cold | 16 | 702 | 114 | 196 | yes |
| Komodo National Park | cold | 32 | 875 | 114 | 603 | yes |
| a bowl of ramen | cold | 8 | 1134 | 76 | 169 | yes |
| the Trojan horse | cold | 16 | 631 | 83 | 179 | yes |
| Ki Hajar Dewantara | cold | 32 | 346 | 75 | 180 | yes |
| a thunderstorm | cold | 8 | 1398 | 84 | 189 | yes |
| the Rosetta Stone | cold | 16 | 753 | 78 | 198 | yes |
| Danau Kelimutu | cold | 32 | 378 | 72 | 175 | yes |
| a lighthouse keeper | warm (repeat) | 8 | 354 | 10 | 100 | yes |

No retries and no errors. "Cold" is approximate: `make_vectors.py` had fetched the same ranges from the same region minutes earlier to compute the hashes (see `spikes/g3_fetch/README.md`).

Reading: at concurrency 16–32, a full phrase (both layers) took 0.35–0.88 s here, and 0.35 s on a warm repeat. Concurrency 8 took 1.1–1.4 s. The CORS checks passed for a normal origin. A `null` origin (a page opened from disk) was checked with curl: the resolver answers `access-control-allow-origin: null`, and the CDN answers `*`.

## Builder's run (2026-11-19)

The builder's PC: Chrome 157 on Windows, Indonesia; the browser reported `4g`, 10 Mbps, 100 ms RTT. CDN host: `us.aws.cdn.hf.co`. Resolving each shard took 1,014 ms and 804 ms (one after the other on this page).

| Phrase | CDN cache | Concurrency | Total ms | Median request ms | p95 request ms | Bytes OK |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| a lighthouse keeper | cold | 8 | 3732 | 348 | 475 | yes |
| the Andromeda galaxy | cold | 16 | 1921 | 338 | 449 | yes |
| Komodo National Park | cold | 32 | 1077 | 360 | 451 | yes |
| a bowl of ramen | cold | 8 | 3412 | 323 | 429 | yes |
| the Trojan horse | cold | 16 | 1906 | 349 | 419 | yes |
| Ki Hajar Dewantara | cold | 32 | 1092 | 349 | 454 | yes |
| a thunderstorm | cold | 8 | 3444 | 322 | 442 | yes |
| the Rosetta Stone | cold | 16 | 1790 | 336 | 418 | yes |
| Danau Kelimutu | cold | 32 | 1071 | 364 | 450 | yes |
| a lighthouse keeper | warm (repeat) | 8 | 543 | 40 | 58 | yes |

No retries, no errors, every byte verified.

Reading:
- **Latency-bound, not bandwidth-bound.** Each cold request took about 0.32–0.36 s, and the total is close to (96 ÷ concurrency) × that: 3.4–3.7 s at 8, 1.8–1.9 s at 16, 1.07–1.09 s at 32.
- **The CDN caches near the builder.** The repeat took 40 ms per request, so a phrase someone fetched recently (an example phrase, say) loads in about half a second even at concurrency 8.
- **For the app:** resolve both shards in parallel when the page opens; send a phrase's requests with concurrency 32 or more (higher is untested); fetch layer 1 first; keep fetched rows for the session.

Latency target (Req 2.5) proposed from these numbers: Decision 10 in `requirements.md`, to be confirmed by the builder at checkpoint 6.6.
