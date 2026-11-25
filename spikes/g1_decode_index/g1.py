"""Gate G1: does a compressed decode index give the same decoded words as the exact readout?

Exact readout (as in the kill test): v = Wv @ concat(rows); words = top-k of cos(E_i, v) over the
full input-embedding table E (129,280 x 5,120).

Compressed readout: PCA basis P (d x 5,120) fitted on the row-normalized embeddings En.
  Z = (En - mean) @ P.T, stored as int8 with one float16 scale per token.
  M = P @ Wv, stored as float16 (d x 6,144 per layer).
  score_i = scale_i * (Zq_i . (M @ x)).
Ranking by En_i . v is unchanged by the mean term (it adds the same constant to every token), so
the only losses are the dropped PCA directions and the int8/fp16 rounding.

Measured on bench/dev_phrases.txt (never the benchmark set):
  - top-10 overlap with the exact readout, every token position of every phrase, both layers;
  - the same for single-row contributions (Wv[:, block_c] @ row_c), as shown on the row cards;
  - the same for random rows;
  - the pilot hit@10 (first 30 dev phrases, pilot sets) with exact and compressed readouts.

Run from a directory holding config.json, tokenizer.json and embed.npy (see spikes/engram_readout):
  python g1.py /path/to/Engramscope
"""
import sys, json, time, concurrent.futures as cf
import numpy as np
from tokenizers import Tokenizer

import killtest as K
from assoc_bench import PAIRS

REPO_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
DIMS = [128, 256, 512]
K_TOP = 10
RNG = np.random.default_rng(1234)


def topk(scores, k=K_TOP):
    # scores: (vocab, n) -> (n, k) indices, best first
    idx = np.argpartition(-scores, k, axis=0)[:k]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=0), axis=0)
    return np.take_along_axis(idx, order, axis=0).T


def overlap(a, b):
    return np.array([len(set(x) & set(y)) / K_TOP for x, y in zip(a, b)])


def main():
    t0 = time.time()
    tok = Tokenizer.from_file("tokenizer.json")
    tmap, vocab = K.token_map(tok)
    pad = int(tmap[K.CFG["engram_pad_token_id"]])
    primes, offs = K.layout()
    mult = K.multipliers(vocab)

    E = np.load("embed.npy", mmap_mode="r")
    En = np.empty(E.shape, np.float32)
    for i in range(0, E.shape[0], 8192):
        b = np.asarray(E[i:i + 8192], np.float32)
        En[i:i + 8192] = b / np.linalg.norm(b, axis=1, keepdims=True)
    mu = En.mean(0)
    C = np.zeros((En.shape[1],) * 2, np.float64)
    for i in range(0, En.shape[0], 8192):
        b = (En[i:i + 8192] - mu).astype(np.float32)
        C += b.T @ b
    w, V = np.linalg.eigh(C)
    order = np.argsort(-w)
    w, V = w[order], V[:, order]
    var = w / w.sum()
    P = V[:, :max(DIMS)].T.astype(np.float32)  # (512, 5120)
    print(f"PCA done {time.time() - t0:.0f}s; variance kept: " + ", ".join(f"d={d}: {var[:d].sum():.3f}" for d in DIMS))

    Zfull = (En - mu) @ P.T  # (vocab, 512)
    index = {}
    for d in DIMS:
        Z = Zfull[:, :d]
        s = np.abs(Z).max(1, keepdims=True) / 127
        q = np.clip(np.round(Z / s), -127, 127).astype(np.int8)
        index[d] = (q.astype(np.float32), s.astype(np.float16).astype(np.float32))

    dev = [l.strip() for l in open(f"{REPO_DIR}/bench/dev_phrases.txt", encoding="utf-8") if l.strip() and not l.startswith("#")]
    assert len(dev) == 50 and [p for p, _ in PAIRS] == [" " + p for p in dev[:30]]

    results = {"dims": DIMS, "variance_kept": {d: float(var[:d].sum()) for d in DIMS}, "layers": {}}
    for li, lid in enumerate(K.LAYERS):
        base, h = K.header(K.SHARD[lid])
        wt, st = h[f"layers.{lid}.engram.wkv.weight"], h[f"layers.{lid}.engram.wkv.scale"]
        n_in = wt["shape"][1]
        r0 = K.HC * K.DIM
        raw = K.rng(K.SHARD[lid], base + wt["data_offsets"][0] + r0 * n_in, K.DIM * n_in)
        sraw = K.rng(K.SHARD[lid], base + st["data_offsets"][0] + (r0 // 32) * (n_in // 32), (K.DIM // 32) * (n_in // 32))
        W = K.E4M3[np.frombuffer(raw, dtype=np.uint8)].reshape(K.DIM, n_in)
        W *= np.repeat(np.repeat(K.E8M0[np.frombuffer(sraw, dtype=np.uint8)].reshape(K.DIM // 32, n_in // 32), 32, 0), 32, 1)
        et, es = h[f"layers.{lid}.engram.embed.weight"], h[f"layers.{lid}.engram.embed.scale"]
        eo, so = base + et["data_offsets"][0], base + es["data_offsets"][0]

        def row(r):
            wb = K.rng(K.SHARD[lid], eo + int(r) * K.HD, K.HD)
            sb = K.rng(K.SHARD[lid], so + int(r) * (K.HD // 32), K.HD // 32)
            return (K.E4M3[np.frombuffer(wb, dtype=np.uint8)].reshape(K.HD // 32, 32) * K.E8M0[np.frombuffer(sb, dtype=np.uint8)][:, None]).reshape(K.HD)

        addr, meta = [], []  # one entry per (phrase, position)
        for pi, ph in enumerate(dev):
            ids = np.array(tok.encode(" " + ph).ids, dtype=np.int64)
            a = K.hash_ids(tmap[ids], pad, mult[li], primes[li], offs[li])
            for pos in range(len(ids)):
                addr.append(a[pos]); meta.append((pi, pos, pos == len(ids) - 1))
        rand = [RNG.integers(0, K.NUM_EMB[li], size=K.NCOLS) for _ in range(len(dev))]
        need = sorted({int(r) for a in addr + rand for r in a})
        with cf.ThreadPoolExecutor(32) as ex:
            rows = dict(zip(need, ex.map(row, need)))
        print(f"layer {lid}: {len(addr)} positions, {len(need)} rows fetched, {time.time() - t0:.0f}s")

        X = np.stack([np.concatenate([rows[int(r)] for r in a]) for a in addr], 1)  # (6144, npos)
        XR = np.stack([np.concatenate([rows[int(r)] for r in a]) for a in rand], 1)
        # per-row contributions at each phrase's last position: block c only
        last = [i for i, m in enumerate(meta) if m[2]]
        XC = np.zeros((n_in, len(last) * K.NCOLS), np.float32)
        for j, i in enumerate(last):
            for c in range(K.NCOLS):
                XC[c * K.HD:(c + 1) * K.HD, j * K.NCOLS + c] = X[c * K.HD:(c + 1) * K.HD, i]
        sets = {"positions": X, "single rows": XC, "random rows": XR}

        exact = {k: topk(En @ (W @ x)) for k, x in sets.items()}
        lr = {"n": {k: int(x.shape[1]) for k, x in sets.items()}, "overlap": {}, "top1_in_top10": {}, "pilot_hit10": {}}

        def pilot(tops):
            hits = 0
            for (ph, ok), pi in zip(PAIRS, range(30)):
                ids = tok.encode(ph).ids
                inph = {tok.decode([t]).strip().lower() for t in ids}
                acc = {a for a in ok if a not in inph}
                i = [k for k, m in enumerate(meta) if m[0] == pi and m[2]][0]
                hits += any(tok.decode([int(t)]).strip().lower() in acc for t in tops[i])
            return hits

        lr["pilot_hit10"]["exact"] = pilot(exact["positions"])
        for d in DIMS:
            q, s = index[d]
            M = (P[:d] @ W).astype(np.float16).astype(np.float32)
            for k, x in sets.items():
                ap = topk(s * (q @ (M @ x)))
                ov = overlap(exact[k], ap)
                lr["overlap"].setdefault(k, {})[d] = {"mean": float(ov.mean()), "p10": float(np.percentile(ov, 10)), "min": float(ov.min())}
                lr["top1_in_top10"].setdefault(k, {})[d] = float(np.mean([e[0] in set(a) for e, a in zip(exact[k], ap)]))
                if k == "positions":
                    lr["pilot_hit10"][d] = pilot(ap)
        results["layers"][lid] = lr
        print(json.dumps(lr, indent=1))

    vocab_n = En.shape[0]
    results["sizes_bytes"] = {d: {"embeddings_int8_plus_fp16_scale": vocab_n * (d + 2), "M_fp16_per_layer": d * 6144 * 2} for d in DIMS}
    import os; os.makedirs(f"{REPO_DIR}/docs/results", exist_ok=True)
    out = f"{REPO_DIR}/docs/results/g1-decode-index.json"
    json.dump(results, open(out, "w"), indent=1)
    print("wrote", out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
