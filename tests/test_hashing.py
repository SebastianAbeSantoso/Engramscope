"""Design properties 1-5: hash parity, address ranges, bucket sums, normalization, padding."""

from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from _reference import reference_addresses
from engramscope.hashing import EngramConfig, Hasher, bucket_primes


@st.composite
def synthetic(draw):
    """A small random Engram configuration with an identity token map (token ids are compressed ids)."""
    n_layers = draw(st.integers(1, 2))
    vocab = draw(st.integers(2, 100_000))
    cfg = EngramConfig(
        layer_ids=tuple(draw(st.lists(st.integers(0, 60), min_size=n_layers, max_size=n_layers, unique=True))),
        num_embeddings=(0,) * n_layers,
        max_ngram_size=draw(st.integers(2, 4)),
        n_heads=draw(st.integers(1, 4)),
        head_dim=8,
        vocab_size=draw(st.integers(2, 50_000)),
        compressed_vocab_size=vocab,
        pad_token_id=draw(st.integers(0, vocab - 1)),
    )
    cfg = replace(cfg, num_embeddings=tuple(int(s) for s in bucket_primes(cfg).sum(axis=1)))
    hasher = Hasher(cfg, np.arange(vocab, dtype=np.int64), vocab)
    tokens = draw(st.lists(st.integers(0, vocab - 1), min_size=1, max_size=12))
    return cfg, hasher, tokens


@settings(max_examples=200, deadline=None)
@given(synthetic())
def test_parity_with_reference_transcription(case):
    cfg, hasher, tokens = case
    primes3 = hasher.primes.reshape(len(cfg.layer_ids), cfg.max_ngram_size - 1, cfg.n_heads)
    expected = reference_addresses(np.array(tokens, np.int64), hasher.pad, hasher.multipliers, primes3, hasher.offsets)
    np.testing.assert_array_equal(hasher.addresses(tokens), expected)


@settings(max_examples=200, deadline=None)
@given(synthetic())
def test_addresses_stay_in_their_column_range(case):
    cfg, hasher, tokens = case
    out = hasher.addresses(tokens)
    lo, hi = hasher.offsets, hasher.offsets + hasher.primes
    assert np.all(out >= lo[None]) and np.all(out < hi[None])
    for li in range(len(cfg.layer_ids)):  # column ranges tile the table without overlap
        assert lo[li, 0] == 0 and np.all(lo[li, 1:] == hi[li, :-1]) and hi[li, -1] == cfg.num_embeddings[li]


@settings(max_examples=200, deadline=None)
@given(synthetic(), st.data())
def test_position_depends_only_on_its_window(case, data):
    cfg, hasher, tokens = case
    t = data.draw(st.integers(0, len(tokens) - 1))
    changed = list(tokens)
    for j in range(0, max(0, t - cfg.max_ngram_size + 1)):
        changed[j] = data.draw(st.integers(0, cfg.compressed_vocab_size - 1))
    np.testing.assert_array_equal(hasher.addresses(tokens)[t], hasher.addresses(changed)[t])


@settings(max_examples=100, deadline=None)
@given(synthetic())
def test_first_position_is_padded(case):
    cfg, hasher, tokens = case
    # position 0 sees only its own token and padding, so it equals the address of a one-token text
    np.testing.assert_array_equal(hasher.addresses(tokens)[0], hasher.addresses(tokens[:1])[0])


def test_rejects_wrong_compressed_vocab():
    cfg = EngramConfig((1,), (0,), 2, 1, 8, 10, 5, 0)
    with pytest.raises(ValueError):
        Hasher(cfg, np.arange(6), 6)


@pytest.mark.network
def test_model_constants(model):
    """Req 1.2: compressed vocabulary and bucket sums of the pinned model."""
    assert model.cfg.compressed_vocab_size == 99_092
    assert int(model.hasher.token_map.max()) + 1 == 99_092
    assert tuple(int(s) for s in model.hasher.primes.sum(axis=1)) == (384_006_168, 384_016_682)


@pytest.mark.network
@pytest.mark.parametrize("word", ["music", "river", "the", "city", "water", "house", "café"])
def test_case_and_space_variants_share_addresses(model, word):
    """Design property 4: tokens that normalize alike get one compressed id, so the same addresses."""
    variants = [" " + word, " " + word.capitalize(), " " + word.upper(), word, word.capitalize()]
    single = [ids for ids in (model.encode(v) for v in variants) if len(ids) == 1]
    assert len({ids[0] for ids in single}) >= 2  # at least two different token ids to compare
    addresses = [model.hasher.addresses(ids) for ids in single]
    for a in addresses[1:]:
        np.testing.assert_array_equal(a, addresses[0])
