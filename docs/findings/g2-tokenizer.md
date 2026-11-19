# Gate G2: tokenizer parity (2026-11-19)

**Result: pass.** Both JavaScript tokenizers give exactly the same token ids as the Python reference on all 1,000 test strings.

## Setup

- Reference: Python `tokenizers` 0.23.3 (Python 3.12.15), loading `tokenizer.json` at the pinned revision `314113f8…`.
- Candidates, on Node 22.23.4:
  - `@huggingface/tokenizers` 0.2.0: pure JavaScript, about 8 kB gzipped, no dependencies;
  - `@huggingface/transformers` 4.3.2 `AutoTokenizer`: the same tokenizer code, shipped inside the full library.
- 1,000 strings (`spikes/g2_tokenizer/make_strings.py`, fixed seed, 12,171 characters, 5,671 tokens):
  - every benchmark and dev phrase, with and without a leading space (346);
  - 78 hand-written edge cases: CJK, Korean, Indonesian, emoji with skin tones and ZWJ sequences, flags, combining accents, full-width forms, ligatures, RTL scripts, code, URLs, the model's special-token strings, zero-width space, BOM, tabs and newlines;
  - 576 random strings, alternating between decoded random token ids and random characters from 13 Unicode ranges.
- Encoded with and without special tokens. The two results are identical, because this tokenizer adds no BOS or EOS on `encode`.

## Result

| Library | Mismatches (with / without special tokens) | Time for 1,000 strings |
| --- | --- | --- |
| `@huggingface/tokenizers` 0.2.0 | 0 / 0 | 59 / 18 ms |
| transformers.js 4.3.2 `AutoTokenizer` | 0 / 0 | 53 / 19 ms |

Files: `spikes/g2_tokenizer/{make_strings.py, js_ids.mjs, strings.json, py_ids.json, g2-report.json}`.

## Decision for the build

Use `@huggingface/tokenizers`, pinned to 0.2.0. Its size is negligible, while transformers.js pulls in about 740 MB of `node_modules` (ONNX runtime) that Engramscope does not need. Load `tokenizer.json` (6.4 MB) and `tokenizer_config.json` from the pinned revision. The CI test (Task 5.1) reruns this comparison against `py_ids.json`, so any tokenizer upgrade has to pass it.
