"""Gate G1, second attempt: a projection fitted to the readout, not only to the embeddings.

Scores are s = E v (E: row-normalized embeddings, vocab x D; v = W x). The best rank-d
approximation of E v in expectation over v (Frobenius) is:
  E = Ua Sa Va^T                    (thin SVD, via eigh of E^T E)
  T = Sa Va^T Gv^{1/2},  Gv = E[v v^T] = W Gx W^T
  T = U S Wt  ->  keep U_d
  embedding side: Z = Ua U_d            (vocab x d, stored int8 + fp16 scale per token)
  query side:     M = U_d^T Sa Va^T W   (d x 6144, fp16), y = M x, score = Z y
Gx variants:
  "iso":   Gx = I (only W's own structure)
  "block": Gx block-diagonal, one 256x256 second-moment matrix per column, estimated from a
           contiguous chunk of 4,096 table rows per column (rows are placed by hash, so a chunk
           is a spread-out sample of the table).
Same measurements as g1.py on bench/dev_phrases.txt (never the benchmark set).

  python g1b.py /path/to/Engramscope [dims, default 128,256,384,512] [variants, default iso,block] [bits, default 8]
Also reports recall: the share of the exact top-10 found in the compressed top-K, for a shortlist-and-rerank design.
"""
import sys, json, time, concurrent.futures as cf
import numpy as np
from tokenizers import Tokenizer

import killtest as K
from assoc_bench import PAIRS

REPO_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
DIMS = [int(d) for d in (sys.argv[2] if len(sys.argv) > 2 else "128,256,384,512").split(",")]
BITS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
QMAX = 2 ** (BITS - 1) - 1
VARIANTS = (sys.argv[3] if len(sys.argv) > 3 else "iso,block").split(",")
K_TOP = 10
RECALL_K = [50, 100, 200, 500, 1000, 2000]
CHUNK = 4096
RNG = np.random.default_rng(1234)


def topk(scores, k=K_TOP):
    idx = np.argpartition(-scores, k, axis=0)[:k]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=0), axis=0)
    return np.take_along_axis(idx, order, axis=0).T


def overlap(a, b):
    return np.array([len(set(x) & set(y)) / K_TOP for x, y in zip(a, b)])


def psd_sqrt(G, floor=1e-10):
    w, V = np.linalg.eigh(G)
    w = np.clip(w, w.max() * floor, None)
    return (V * np.sqrt(w)) @ V.T


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
    Ge = np.zeros((En.shape[1],) * 2, np.float64)
    for i in range(0, En.shape[0], 8192):
        b = En[i:i + 8192]
        Ge += (b.T @ b).astype(np.float64)
    lam, Va = np.linalg.eigh(Ge)
    keep = lam > lam.max() * 1e-10
    lam, Va = lam[keep], Va[:, keep]
    Sa = np.sqrt(lam)
    SaVaT = (Sa[:, None] * Va.T)  # (r, D)
    Ua = En @ (Va / Sa).astype(np.float32)  # (vocab, r), orthonormal columns
    print(f"embedding Gram done, rank {keep.sum()}, {time.time() - t0:.0f}s")

    dev = [l.strip() for l in open(f"{REPO_DIR}/bench/dev_phrases.txt", encoding="utf-8") if l.strip() and not l.startswith("#")]
    assert len(dev) == 50 and [p for p, _ in PAIRS] == [" " + p for p in dev[:30]]

    results = {"method": "readout-fitted rank-d projection", "dims": DIMS, "bits": BITS, "chunk_rows_per_column": CHUNK, "layers": {}}
    for li, lid in enumerate(K.LAYERS):
        sh = K.SHARD[lid]
        base, h = K.header(sh)
        wt, st = h[f"layers.{lid}.engram.wkv.weight"], h[f"layers.{lid}.engram.wkv.scale"]
        n_in = wt["shape"][1]
        r0 = K.HC * K.DIM
        raw = K.rng(sh, base + wt["data_offsets"][0] + r0 * n_in, K.DIM * n_in)
        sraw = K.rng(sh, base + st["data_offsets"][0] + (r0 // 32) * (n_in // 32), (K.DIM // 32) * (n_in // 32))
        W = K.E4M3[np.frombuffer(raw, dtype=np.uint8)].reshape(K.DIM, n_in)
        W *= np.repeat(np.repeat(K.E8M0[np.frombuffer(sraw, dtype=np.uint8)].reshape(K.DIM // 32, n_in // 32), 32, 0), 32, 1)
        et, es = h[f"layers.{lid}.engram.embed.weight"], h[f"layers.{lid}.engram.embed.scale"]
        eo, so = base + et["data_offsets"][0], base + es["data_offsets"][0]

        def rows_block(r, n):
            wb = K.rng(sh, eo + int(r) * K.HD, n * K.HD)
            sb = K.rng(sh, so + int(r) * (K.HD // 32), n * (K.HD // 32))
            v = K.E4M3[np.frombuffer(wb, dtype=np.uint8)].reshape(n, K.HD // 32, 32) * K.E8M0[np.frombuffer(sb, dtype=np.uint8)].reshape(n, K.HD // 32, 1)
            return v.reshape(n, K.HD)

        # block second moments from contiguous chunks
        starts = [int(offs[li][c] + RNG.integers(0, int(primes[li][c]) - CHUNK)) for c in range(K.NCOLS)]
        with cf.ThreadPoolExecutor(12) as ex:
            chunks = list(ex.map(lambda s: rows_block(s, CHUNK), starts))
        Gblocks = [(c.T @ c / CHUNK).astype(np.float64) for c in chunks]
        chunk_norm = float(np.mean([np.linalg.norm(c, axis=1).mean() for c in chunks]))

        # dev rows
        addr, meta = [], []
        for pi, ph in enumerate(dev):
            ids = np.array(tok.encode(" " + ph).ids, dtype=np.int64)
            a = K.hash_ids(tmap[ids], pad, mult[li], primes[li], offs[li])
            for pos in range(len(ids)):
                addr.append(a[pos]); meta.append((pi, pos, pos == len(ids) - 1))
        rand = [RNG.integers(0, K.NUM_EMB[li], size=K.NCOLS) for _ in range(len(dev))]
        need = sorted({int(r) for a in addr + rand for r in a})
        with cf.ThreadPoolExecutor(32) as ex:
            rows = dict(zip(need, ex.map(lambda r: rows_block(r, 1)[0], need)))
        dev_norm = float(np.mean([np.linalg.norm(rows[int(r)]) for a in addr for r in a]))
        print(f"layer {lid}: chunks + {len(need)} rows fetched, {time.time() - t0:.0f}s; mean row norm chunk {chunk_norm:.3f} dev {dev_norm:.3f}")

        X = np.stack([np.concatenate([rows[int(r)] for r in a]) for a in addr], 1)
        XR = np.stack([np.concatenate([rows[int(r)] for r in a]) for a in rand], 1)
        last = [i for i, m in enumerate(meta) if m[2]]
        XC = np.zeros((n_in, len(last) * K.NCOLS), np.float32)
        for j, i in enumerate(last):
            for c in range(K.NCOLS):
                XC[c * K.HD:(c + 1) * K.HD, j * K.NCOLS + c] = X[c * K.HD:(c + 1) * K.HD, i]
        sets = {"positions": X, "single rows": XC, "random rows": XR}
        exact = {k: topk(En @ (W @ x)) for k, x in sets.items()}

        def pilot(tops):
            hits = 0
            for pi, (ph, ok) in enumerate(PAIRS):
                inph = {tok.decode([t]).strip().lower() for t in tok.encode(ph).ids}
                acc = {a for a in ok if a not in inph}
                i = [k for k, m in enumerate(meta) if m[0] == pi and m[2]][0]
                hits += any(tok.decode([int(t)]).strip().lower() in acc for t in tops[i])
            return hits

        lr = {"pilot_hit10": {"exact": pilot(exact["positions"])}, "variants": {}}
        Wd = W.astype(np.float64)
        for var in VARIANTS:
            if var == "iso":
                Gv = Wd @ Wd.T
            else:
                Gv = np.zeros((K.DIM, K.DIM))
                for c in range(K.NCOLS):
                    Wc = Wd[:, c * K.HD:(c + 1) * K.HD]
                    Gv += Wc @ Gblocks[c] @ Wc.T
            T = SaVaT @ psd_sqrt(Gv)
            U, S, _ = np.linalg.svd(T, full_matrices=False)
            vr = {}
            for d in DIMS:
                Ud = U[:, :d]
                Z = Ua @ Ud.astype(np.float32)
                s = np.abs(Z).max(1, keepdims=True) / QMAX
                q = np.clip(np.round(Z / s), -QMAX, QMAX).astype(np.int8).astype(np.float32)
                s = s.astype(np.float16).astype(np.float32)
                M = ((Ud.T @ SaVaT) @ Wd).astype(np.float16).astype(np.float32)
                r = {"energy_kept": float((S[:d] ** 2).sum() / (S ** 2).sum())}
                for k, x in sets.items():
                    big = topk(s * (q @ (M @ x)), max(RECALL_K))
                    ap = big[:, :K_TOP]
                    ov = overlap(exact[k], ap)
                    rec = {kk: round(float(np.mean([len(set(e) & set(g[:kk])) / K_TOP for e, g in zip(exact[k], big)])), 4) for kk in RECALL_K}
                    full = {kk: round(float(np.mean([set(e) <= set(g[:kk]) for e, g in zip(exact[k], big)])), 4) for kk in RECALL_K}
                    r[k] = {"overlap_mean": round(float(ov.mean()), 3), "overlap_p10": float(np.percentile(ov, 10)),
                            "top1_in_top10": round(float(np.mean([e[0] in set(a) for e, a in zip(exact[k], ap)])), 3),
                            "recall_of_exact_top10_in_topK": rec, "all_exact_top10_in_topK": full}
                    if k == "positions":
                        r["pilot_hit10"] = pilot(ap)
                vr[d] = r
                print(lid, var, d, json.dumps(r))
            lr["variants"][var] = vr
        results["layers"][lid] = lr
    n = En.shape[0]
    results["sizes_bytes"] = {d: {"embeddings_q_plus_fp16_scale": n * (d * BITS // 8 + 2), "M_fp16_per_layer": d * 6144 * 2} for d in DIMS}
    import os; os.makedirs(f"{REPO_DIR}/docs/results", exist_ok=True)
    out = f"{REPO_DIR}/docs/results/g1b-{'-'.join(VARIANTS)}-{'-'.join(map(str, DIMS))}-int{BITS}.json"
    json.dump(results, open(out, "w"), indent=1)
    print("wrote", out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
