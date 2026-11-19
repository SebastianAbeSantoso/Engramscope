"""The exact readout: rows -> per-row contributions -> value vector -> nearest input embeddings.

The value vector the model adds (before its gate) is `Wv @ concat(rows)`, where Wv is the value part
of `wkv`. It is a sum of 24 per-row contributions `Wv[:, block_c] @ row_c`, which is what makes
trace lesions plain arithmetic. Decoding takes the tokens whose input embeddings have the highest
cosine with the vector: an approximation that assumes the vector lives in embedding space.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import sqlite3

import numpy as np
from tokenizers import Tokenizer

from . import fp8, hub
from .hashing import EngramConfig, Hasher, compressed_token_map


def contributions(wv: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Per-row contributions. wv: [hidden, n_cols * head_dim]; rows: [n_cols, head_dim] -> [n_cols, hidden]."""
    n_cols, head_dim = rows.shape
    return np.einsum("hcd,cd->ch", wv.reshape(wv.shape[0], n_cols, head_dim), rows)


def topk_ids(embeddings: np.ndarray, vectors: np.ndarray, k: int = 10, batch: int = 1024) -> np.ndarray:
    """Token ids of the k highest cosines, best first. embeddings: row-normalized [vocab, hidden]; vectors: [n, hidden]."""
    out = np.empty((vectors.shape[0], k), dtype=np.int64)
    for i in range(0, vectors.shape[0], batch):
        scores = embeddings @ vectors[i : i + batch].T  # [vocab, b]; the vector's norm does not change the ranking
        idx = np.argpartition(-scores, k, axis=0)[:k]
        order = np.argsort(-np.take_along_axis(scores, idx, axis=0), axis=0, kind="stable")
        out[i : i + batch] = np.take_along_axis(idx, order, axis=0).T
    return out


class RowStore:
    """Engram rows (256 FP8 values + 8 E8M0 scales), fetched by range request and cached in SQLite."""

    def __init__(self, cfg: EngramConfig):
        self.cfg = cfg
        self.db = sqlite3.connect(hub.cache_dir() / "rows.sqlite")
        self.db.execute(
            "create table if not exists rows (layer int, row int, v blob, s blob, primary key (layer, row))"
        )
        self.tensors = {
            layer: (hub.tensor(f"layers.{layer}.engram.embed.weight"), hub.tensor(f"layers.{layer}.engram.embed.scale"))
            for layer in cfg.layer_ids
        }
        for layer, (v, s) in self.tensors.items():
            n = cfg.num_embeddings[cfg.layer_ids.index(layer)]
            if v.shape != (n, cfg.head_dim) or s.shape != (n, cfg.head_dim // 32):
                raise ValueError(f"unexpected Engram table shapes for layer {layer}: {v.shape}, {s.shape}")

    def _fetch(self, layer: int, row: int) -> tuple[int, bytes, bytes]:
        v, s = self.tensors[layer]
        vb = hub.read_range(v.shard, *hub.row_range(v, row, self.cfg.head_dim))
        sb = hub.read_range(s.shard, *hub.row_range(s, row, self.cfg.head_dim // 32))
        return row, vb, sb

    def prefetch(self, layer: int, rows: set[int] | list[int], workers: int = 32) -> int:
        """Fetch rows not yet cached; returns how many were fetched."""
        have = set()
        wanted = sorted({int(r) for r in rows})
        for i in range(0, len(wanted), 900):
            chunk = wanted[i : i + 900]
            q = f"select row from rows where layer = ? and row in ({','.join('?' * len(chunk))})"
            have.update(r for (r,) in self.db.execute(q, [layer, *chunk]))
        missing = [r for r in wanted if r not in have]
        with cf.ThreadPoolExecutor(workers) as pool:
            for row, vb, sb in pool.map(lambda r: self._fetch(layer, r), missing):
                self.db.execute("insert or replace into rows values (?, ?, ?, ?)", (layer, row, vb, sb))
        self.db.commit()
        return len(missing)

    def get(self, layer: int, rows: np.ndarray) -> np.ndarray:
        """Decoded rows, float32 [len(rows), head_dim], in the order given."""
        self.prefetch(layer, list(rows))
        values, scales = [], []
        for r in rows:
            vb, sb = self.db.execute("select v, s from rows where layer = ? and row = ?", (layer, int(r))).fetchone()
            values.append(np.frombuffer(vb, np.uint8))
            scales.append(np.frombuffer(sb, np.uint8))
        return fp8.decode_rows(np.stack(values), np.stack(scales))


class Model:
    """Everything the exact readout needs from the pinned model files, cached locally."""

    def __init__(self):
        config = json.loads(hub.file("config.json").read_text())
        text = config.get("text_config", config)
        self.cfg = EngramConfig.from_json(config)
        self.hidden, self.hc_mult, self.vocab = text["hidden_size"], text["hc_mult"], text["vocab_size"]
        self.tokenizer = Tokenizer.from_file(str(hub.file("tokenizer.json")))
        token_map, size = self._cached_token_map()
        self.hasher = Hasher(self.cfg, token_map, size)
        self.rows = RowStore(self.cfg)
        self._wv: dict[int, np.ndarray] = {}
        self._embeddings: np.ndarray | None = None
        self._strings: list[str] | None = None

    def _cached_token_map(self) -> tuple[np.ndarray, int]:
        path = hub.cache_dir() / "token_map.npy"
        if not path.exists():
            lookup, size = compressed_token_map(self.tokenizer)
            np.save(path, np.append(lookup, size))
        stored = np.load(path)
        return stored[:-1], int(stored[-1])

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def addresses(self, token_ids: list[int]) -> np.ndarray:
        """[n_layers, n_cols] addresses at the last position."""
        return self.hasher.addresses(token_ids)[-1]

    def value_matrix(self, layer: int) -> np.ndarray:
        """Wv for one layer, float32 [hidden, n_cols * head_dim]: the value rows of `wkv`, after the keys."""
        if layer not in self._wv:
            path = hub.cache_dir() / f"wv-L{layer}.npy"
            if not path.exists():
                w = hub.tensor(f"layers.{layer}.engram.wkv.weight")
                s = hub.tensor(f"layers.{layer}.engram.wkv.scale")
                n_in = self.cfg.n_cols * self.cfg.head_dim
                if w.shape != ((self.hc_mult + 1) * self.hidden, n_in) or s.shape != (w.shape[0] // 32, n_in // 32):
                    raise ValueError(f"unexpected wkv shapes {w.shape}, {s.shape}")
                r0 = self.hc_mult * self.hidden
                raw = hub.read_tensor(w, r0 * n_in, self.hidden * n_in)
                sraw = hub.read_tensor(s, (r0 // 32) * (n_in // 32), (self.hidden // 32) * (n_in // 32))
                matrix = fp8.decode_block_scaled(
                    np.frombuffer(raw, np.uint8).reshape(self.hidden, n_in),
                    np.frombuffer(sraw, np.uint8).reshape(self.hidden // 32, n_in // 32),
                )
                np.save(path, matrix)
            self._wv[layer] = np.load(path)
        return self._wv[layer]

    def embeddings(self) -> np.ndarray:
        """Row-normalized input embeddings, float32 [vocab, hidden]."""
        if self._embeddings is None:
            path = hub.cache_dir() / "embed-normalized.npy"
            if not path.exists():
                t = hub.tensor("embed.weight")
                if t.dtype != "BF16" or t.shape != (self.vocab, self.hidden):
                    raise ValueError(f"unexpected embed.weight {t.dtype} {t.shape}")
                raw = np.frombuffer(hub.read_tensor(t), np.uint16)
                matrix = (raw.astype(np.uint32) << 16).view(np.float32).reshape(t.shape)
                matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
                np.save(path, matrix)
            self._embeddings = np.load(path)
        return self._embeddings

    def token_strings(self) -> list[str]:
        """Each token decoded alone, stripped and lowercased: what hits are matched against."""
        if self._strings is None:
            path = hub.cache_dir() / "token_strings.json"
            if not path.exists():
                strings = [self.tokenizer.decode([i]).strip().lower() for i in range(self.vocab)]
                path.write_text(json.dumps(strings, ensure_ascii=False))
            self._strings = json.loads(path.read_text())
        return self._strings

    def contributions(self, layer: int, row_ids: np.ndarray) -> np.ndarray:
        """[n_cols, hidden] per-row contributions of the given rows (one per column, in column order)."""
        return contributions(self.value_matrix(layer), self.rows.get(layer, row_ids))

    def decode(self, vectors: np.ndarray, k: int = 10) -> np.ndarray:
        return topk_ids(self.embeddings(), np.asarray(vectors, np.float32), k)
