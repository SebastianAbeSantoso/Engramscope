"""Engram addresses, bit-exact with the model's `inference/engram.py` at the pinned revision.

At each position, the compressed ids of the current token and up to 3 previous tokens (padding
before the start of the text) are combined by a rolling XOR of `id * multiplier`. After each step
the running value is the hash of the next n-gram (2, 3, 4); each of the 8 heads takes it modulo its
own prime and adds its table offset. 3 sizes x 8 heads = 24 addresses per position and layer, in
column order c = (n - 2) * n_heads + head.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sympy import isprime
from tokenizers import Regex, Tokenizer, normalizers


@dataclass(frozen=True)
class EngramConfig:
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    max_ngram_size: int
    n_heads: int
    head_dim: int
    vocab_size: int  # `engram_vocab_size`: where the prime search starts
    compressed_vocab_size: int
    pad_token_id: int

    @property
    def n_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    def column(self, c: int) -> tuple[int, int]:
        """(n-gram size, head) of column c."""
        return 2 + c // self.n_heads, c % self.n_heads

    @classmethod
    def from_json(cls, config: dict) -> EngramConfig:
        text = config.get("text_config", config)
        return cls(
            layer_ids=tuple(text["engram_layer_ids"]),
            num_embeddings=tuple(text["engram_num_embeddings"]),
            max_ngram_size=text["engram_max_ngram_size"],
            n_heads=text["engram_n_heads"],
            head_dim=text["engram_head_dim"],
            vocab_size=text["engram_vocab_size"],
            compressed_vocab_size=text["engram_compressed_vocab_size"],
            pad_token_id=text["engram_pad_token_id"],
        )


def bucket_primes(cfg: EngramConfig) -> np.ndarray:
    """[n_layers, n_cols] prime table sizes. Primes are handed out in order and never reused."""
    seen: set[int] = set()
    out = []
    for _ in cfg.layer_ids:
        row = []
        for _ in range(cfg.max_ngram_size - 1):
            current = cfg.vocab_size - 1
            for _ in range(cfg.n_heads):
                current += 1
                while not isprime(current) or current in seen:
                    current += 1
                seen.add(current)
                row.append(current)
        out.append(row)
    return np.array(out, dtype=np.int64)


def bucket_offsets(primes: np.ndarray) -> np.ndarray:
    """Start of each column's range in its layer's table: the cumulative sum of the primes before it."""
    return np.concatenate([np.zeros((primes.shape[0], 1), np.int64), np.cumsum(primes, axis=1)[:, :-1]], axis=1)


def hash_multipliers(cfg: EngramConfig, compressed_vocab_size: int) -> np.ndarray:
    """[n_layers, max_ngram_size] odd multipliers, bounded so `id * multiplier` fits in int64."""
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
    rows = []
    for layer_id in cfg.layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        rows.append(generator.integers(0, bound, size=(cfg.max_ngram_size,), dtype=np.int64) * 2 + 1)
    return np.stack(rows)


_SENTINEL = "\ue000"
NORMALIZER = normalizers.Sequence(
    [
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), _SENTINEL),
        normalizers.Strip(),
        normalizers.Replace(_SENTINEL, " "),
    ]
)


def compressed_token_map(tokenizer: Tokenizer) -> tuple[np.ndarray, int]:
    """Token id -> compressed id; tokens that normalize alike (case, accents, spacing) share an id."""
    key_to_new: dict[str, int] = {}
    size = tokenizer.get_vocab_size(with_added_tokens=True)
    lookup = np.zeros(size, dtype=np.int64)
    for token_id in range(size):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:  # a partial UTF-8 byte token: key it by its raw form
            key = tokenizer.id_to_token(token_id)
        else:
            normalized = NORMALIZER.normalize_str(text)
            key = normalized if normalized else text
        lookup[token_id] = key_to_new.setdefault(key, len(key_to_new))
    return lookup, len(key_to_new)


class Hasher:
    """Exact Engram addresses for a token sequence, with Python integers (no overflow possible)."""

    def __init__(self, cfg: EngramConfig, token_map: np.ndarray, compressed_vocab_size: int):
        if compressed_vocab_size != cfg.compressed_vocab_size:
            raise ValueError(f"compressed vocab {compressed_vocab_size} != config {cfg.compressed_vocab_size}")
        self.cfg = cfg
        self.token_map = token_map
        self.pad = int(token_map[cfg.pad_token_id])
        self.primes = bucket_primes(cfg)
        sums = tuple(int(s) for s in self.primes.sum(axis=1))
        if sums != tuple(cfg.num_embeddings):
            raise ValueError(f"bucket primes sum to {sums}, config says {cfg.num_embeddings}")
        self.offsets = bucket_offsets(self.primes)
        self.multipliers = hash_multipliers(cfg, compressed_vocab_size)

    def addresses(self, token_ids: list[int] | np.ndarray) -> np.ndarray:
        """[positions, n_layers, n_cols] row ids."""
        cfg = self.cfg
        ids = [int(self.token_map[t]) for t in token_ids]
        n = cfg.max_ngram_size
        primes = [[int(p) for p in row] for row in self.primes]
        offsets = [[int(o) for o in row] for row in self.offsets]
        mults = [[int(m) for m in row] for row in self.multipliers]
        out = np.empty((len(ids), len(cfg.layer_ids), cfg.n_cols), dtype=np.int64)
        for t in range(len(ids)):
            window = [ids[t - s] if t - s >= 0 else self.pad for s in range(n)]
            for li in range(len(cfg.layer_ids)):
                rolling = window[0] * mults[li][0]
                col = 0
                for i in range(1, n):
                    rolling ^= window[i] * mults[li][i]
                    for _ in range(cfg.n_heads):
                        out[t, li, col] = rolling % primes[li][col] + offsets[li][col]
                        col += 1
        return out
