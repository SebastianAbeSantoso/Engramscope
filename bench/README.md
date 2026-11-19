# Benchmark phrase sets

## `phrases.jsonl` (pre-registered, Req 6.1)

123 phrases in 7 categories: places 20, people 20, events 15, science 15, culture 15, Indonesian topics 23, everyday things 15. Each line has `id`, `category`, `phrase` and `related` (the accepted related words).

Written and committed by Kiro **before any Engram readout of these phrases was run**. The commit that adds this file is the pre-registration commit; `bench.py` records its hash in every result. Editing a phrase or a related set after the first benchmark run invalidates the run; any change gets a new file and a note here.

What the author had seen when writing it: the 30-phrase pilot from the kill test (`spikes/engram_readout/assoc_bench.py`), including its readouts. None of the pilot phrases is in this set. The pilot showed that decoded tokens are often Chinese, which is why the sets include Chinese words (see the rules below).

### Rules used to write the related sets

1. Single words in lowercase, because a hit is a match between one decoded token (stripped, lowercased) and one set entry.
2. English words, plus common Chinese words where one exists, because the model's vocabulary is bilingual. Indonesian topics also get Indonesian words.
3. No word of the phrase itself, no normalized form of one (plural, abbreviation, alternative spelling), and no direct translation of the phrase or of its words. For example, "nasi goreng" does not accept "rice" or "fried", and "Spanish flu" does not accept "influenza".
4. At least 4 entries per set. Entries are things a reader would name as closely related: location, country, people, works, parts, or typical companions.
5. Entries that the tokenizer splits into several tokens can never match. They were not removed, because checking that would mean tuning the set against the model.

The script that generated the file asserts rule 3 for the phrase's own words (after NFKC and lowercasing), checks that sets have no duplicates and at least 4 entries, and checks that no phrase repeats or comes from the pilot. `bench.py` repeats the own-word exclusion at run time, using the phrase's decoded tokens.

## `dev_phrases.txt` (not part of the benchmark)

Phrases used for gate G1 (decode index agreement) and for development. They have no related sets and are never scored. They are disjoint from `phrases.jsonl`, so tuning the decode index cannot see the benchmark phrases.

## Tokenization

Phrases are tokenized with one leading space (`" " + phrase`), as in the pilot, so that the first word gets its in-sentence token.

## Analysis plan for `phrases.jsonl` (written 2026-11-19, before the first run)

This fixes what Req 6 leaves open. It was written after the phrase set was committed and before any readout of it. `bench.py` follows it exactly.

- **Readout:** exact. The top 10 tokens by cosine against the full, row-normalized input-embedding table (129,280 tokens), at the last position of `" " + phrase`.
- **Conditions** (per phrase):
  - **layer 1** and **layer 14:** the phrase's 24 rows in that layer.
  - **random rows, layer 1** and **random rows, layer 14:** one random row from each of the 24 sub-tables (n-gram size × head), so each block of the value projection gets a row of its own table. Drawn with `numpy.random.default_rng(20261111)` in the order phrase id, layer (1, then 14), column 0–23.
  - **last-token embedding:** the nearest input embeddings of the phrase's last token itself.
  - **shuffled:** the layer-1 rows of another phrase, scored against this phrase's set. The pairing is phrase i ↔ phrase ((i − 1 + 61) mod 123) + 1. Categories are contiguous blocks of at most 23 ids, so a shift of 61 never pairs a phrase with its own category.
- **Hit:** a top-k decoded token (stripped, lowercased) is in the phrase's related set, after removing the phrase's own decoded tokens from the set.
- **Report:** hit@1, hit@5 and hit@10 for every condition, with 95% percentile bootstrap intervals over phrases (2,000 resamples, seed 20261111). Hit@10 per category is descriptive.
- **"Beats" (Req 6.5):** layer 1 beats a control if the 95% paired bootstrap interval of the per-phrase difference in hit@10 (layer 1 minus control) lies entirely above 0. The claim needs every control beaten. That is an intersection of tests, so no multiplicity correction is needed. The controls are layer 14, random rows (layer 1), the last-token embedding and shuffled.
- **Provenance:** every result records the commit and blob hash of `phrases.jsonl`, the code commit, and the model revision. `bench.py` refuses to run if `phrases.jsonl` has uncommitted changes or was last changed by any commit other than ec0e429.
