# Gate G3: browser fetch latency

**Status: sandbox reference done; the builder's run is pending** (Task 1.3 is [HW]).

Page: `spikes/g3_fetch/index.html` (version 2). It fetches 96 byte ranges per phrase (48 rows of 256 bytes plus their 8-byte scales, both Engram layers, 12,672 bytes in total) from the Hugging Face CDN, with the browser cache off. It checks every byte against SHA-256 hashes computed in Python.

## Sandbox reference (2026-11-11)

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

## Builder's run

**Pending.** Open the page on the PC, paste the copied JSON here, and compare the numbers. Latency from Indonesia to the CDN will differ from this table.
