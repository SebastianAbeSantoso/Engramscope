"""A direct numpy transcription of `NgramHashState.forward` in the model's `inference/engram.py`.

DeepSeek-V4.1-Flash reference code, MIT License, Copyright (c) DeepSeek. Transcribed for one text-only
sequence starting at position 0 (no dead tokens), with numpy int64 arithmetic in place of torch.
Used only by the tests, to check `engramscope.hashing.Hasher` against the reference structure.
"""

import numpy as np


def reference_addresses(
    compressed: np.ndarray, pad_id: int, multipliers: np.ndarray, primes: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """compressed: [L] compressed ids; multipliers: [n_layers, max_ngram]; primes: [n_layers, max_ngram - 1, n_heads];
    offsets: [n_layers, n_cols]. Returns [L, n_layers, n_cols]."""
    seqlen = len(compressed)
    max_ngram = multipliers.shape[1]
    positions = np.arange(seqlen)
    tokens, blocked = [], np.zeros(seqlen, dtype=bool)
    for shift in range(max_ngram):
        source = compressed[np.clip(positions - shift, 0, None)]
        blocked = blocked | (positions < shift)
        tokens.append(np.where(blocked, pad_id, source))
    tokens = np.stack(tokens, axis=-1).astype(np.int64)  # [L, max_ngram]
    products = tokens[:, None, :] * multipliers[None, :, :]  # [L, n_layers, max_ngram]
    rolling, hashes = products[..., 0], []
    for i in range(1, max_ngram):
        rolling = np.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling[..., None] % primes[None, :, i - 1, :])
    return np.concatenate(hashes, axis=-1) + offsets[None]
