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
