"""Build the browser assets: constants, token map, tokenizer, decode index, test vectors (Req 1.3, 1.4, 5).

  uv run python -m engramscope.build_assets [--out web/public/assets] [--skip-agreement]

Writes `<out>/<revision>/`. The decode index is the readout-fitted projection chosen at G1
(Decision 8; `docs/findings/g1-decode-index.md`, variant `block`), d = 1,024 at 4 bits:

  scores = E v, with E the row-normalized embeddings and v = Wv x (x: the 24 fetched rows, 6,144 values).
  E = Ua Sa Vaᵀ;  Gv = Σ_c Wv_c G_c Wv_cᵀ, G_c the second moment of 4,096 contiguous rows of column c;
  T = Sa Vaᵀ Gv^½ = U S Wᵀ; keep U_d.  Embedding side Z = Ua U_d;  query side M = U_dᵀ Sa Vaᵀ Wv.

One projection serves both layers. It is fitted on layer 1 (primary); layer 14 gets its own M on
the same Z. Z ships as 4 chunks of 256 dimensions, ordered by importance, each with a per-token
fp16 scale and values in [-7, 7] packed two per byte (low nibble first, stored as q + 8). M ships
in fp16, chunked the same way. Agreement with the exact readout is measured on the shipped
format, at each loading stage, and written to `docs/results/decode-index-<date>.{json,md}`.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from . import REPO, REVISION, fp8, hub, scoring
from .readout import Model

D = 1024
CHUNKS = 4
CHUNK_DIMS = D // CHUNKS
QMAX = 7
FIT_ROWS = 4096
FIT_SEED = 1234
FIT_LAYER = 1

# Test-vector phrases beyond the dev set: scripts, accents, emoji, spacing (Req 1.3).
EXTRA_PHRASES = [
    "Saya pernah membaca tentang Candi Borobudur",
    "東京タワー",
    "北京烤鸭",
    "Ünïcödé Café crème brûlée",
    "🎃 Halloween night",
    "  double  spaces and\ttabs",
    "THE EIFFEL TOWER",
    "a",
]
READOUT_VECTORS = 8  # phrases whose rows and approximate top-10 are included for the browser readout test


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fit(model: Model, log) -> dict:
    """The fitted projection: P (hidden × D) with Z = En P, and M per layer (D × 6,144)."""
    en = model.embeddings()
    ge = np.zeros((en.shape[1],) * 2, np.float64)
    for i in range(0, en.shape[0], 8192):
        b = en[i : i + 8192]
        ge += (b.T @ b).astype(np.float64)
    lam, va = np.linalg.eigh(ge)
    keep = lam > lam.max() * 1e-10
    lam, va = lam[keep], va[:, keep]
    sa = np.sqrt(lam)
    savat = sa[:, None] * va.T  # [r, hidden]
    log(f"embedding Gram: rank {int(keep.sum())}")

    cfg = model.cfg
    w = {layer: model.value_matrix(layer).astype(np.float64) for layer in cfg.layer_ids}
    li = cfg.layer_ids.index(FIT_LAYER)
    primes, offsets = model.hasher.primes[li], model.hasher.offsets[li]
    v, s = model.rows.tensors[FIT_LAYER]
    rng = np.random.default_rng(FIT_SEED)
    hd = cfg.head_dim
    gv = np.zeros((model.hidden, model.hidden))
    for c in range(cfg.n_cols):
        start = int(offsets[c] + rng.integers(0, int(primes[c]) - FIT_ROWS))
        hub.row_range(v, start + FIT_ROWS - 1, hd)  # bounds check on the last row
        vb = hub.read_range(v.shard, v.start + start * hd, FIT_ROWS * hd)
        sb = hub.read_range(s.shard, s.start + start * (hd // 32), FIT_ROWS * (hd // 32))
        rows = fp8.decode_rows(
            np.frombuffer(vb, np.uint8).reshape(FIT_ROWS, hd), np.frombuffer(sb, np.uint8).reshape(FIT_ROWS, hd // 32)
        )
        g = rows.T.astype(np.float64) @ rows.astype(np.float64) / FIT_ROWS
        wc = w[FIT_LAYER][:, c * hd : (c + 1) * hd]
        gv += wc @ g @ wc.T
    log("value second moment: done")
    ev, evec = np.linalg.eigh(gv)
    ev = np.clip(ev, ev.max() * 1e-10, None)
    gv_half = (evec * np.sqrt(ev)) @ evec.T
    u, sv, _ = np.linalg.svd(savat @ gv_half, full_matrices=False)
    ud = u[:, :D]
    log(f"SVD: energy kept at d={D}: {float((sv[:D] ** 2).sum() / (sv**2).sum()):.4f}")
    p = ((va / sa) @ ud).astype(np.float32)  # Z = En P = Ua U_d
    m = {layer: (ud.T @ savat @ w[layer]).astype(np.float16) for layer in cfg.layer_ids}
    return {"p": p, "m": m, "energy_kept": float((sv[:D] ** 2).sum() / (sv**2).sum())}


def quantize(z: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per chunk: packed int4 [vocab, CHUNK_DIMS // 2] uint8 and fp16 scales [vocab]."""
    packed, scales = [], []
    for k in range(CHUNKS):
        zc = z[:, k * CHUNK_DIMS : (k + 1) * CHUNK_DIMS]
        s = (np.abs(zc).max(axis=1) / QMAX).astype(np.float16)
        s[s == 0] = np.float16(1)
        q = np.clip(np.round(zc / s.astype(np.float32)[:, None]), -QMAX, QMAX).astype(np.int8)
        u = (q + 8).astype(np.uint8)
        packed.append((u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8))
        scales.append(s)
    return packed, scales


def dequantize(packed: list[np.ndarray], scales: list[np.ndarray], chunks: int = CHUNKS) -> np.ndarray:
    """The embedding side as the browser sees it, float32 [vocab, chunks * CHUNK_DIMS]."""
    parts = []
    for k in range(chunks):
        b = packed[k]
        q = np.empty((b.shape[0], CHUNK_DIMS), np.float32)
        q[:, 0::2] = (b & 0xF).astype(np.float32) - 8
        q[:, 1::2] = (b >> 4).astype(np.float32) - 8
        parts.append(q * scales[k].astype(np.float32)[:, None])
    return np.concatenate(parts, axis=1)


def approx_topk(zq: np.ndarray, m: np.ndarray, x: np.ndarray, k: int = 10) -> np.ndarray:
    """Top-k token ids of the shipped readout for inputs x [n, 6144]; zq: dequantized [vocab, d'], m: [D, 6144]."""
    y = x.astype(np.float32) @ m[: zq.shape[1]].astype(np.float32).T  # [n, d']
    scores = zq @ y.T  # [vocab, n]
    idx = np.argpartition(-scores, k, axis=0)[:k]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=0), axis=0, kind="stable")
    return np.take_along_axis(idx, order, axis=0).T


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([len(set(x) & set(y)) / len(x) for x, y in zip(a, b, strict=True)]))


def _agreement(model: Model, zfull: tuple, m: dict, log) -> dict:
    """Req 5.3: shipped index against the exact readout, per layer and loading stage."""
    cfg = model.cfg
    dev = [line.strip() for line in Path("bench/dev_phrases.txt").read_text(encoding="utf-8").splitlines()]
    dev = [p for p in dev if p and not p.startswith("#")]
    bench = [
        json.loads(line)
        for line in Path("bench/phrases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    strings = model.token_strings()
    stages = {k: dequantize(*zfull, chunks=k) for k in range(1, CHUNKS + 1)}
    out: dict = {"dev_phrases": len(dev), "bench_phrases": len(bench), "layers": {}}
    for li, layer in enumerate(cfg.layer_ids):
        wv = model.value_matrix(layer)
        # dev set: every position
        xs = []
        for phrase in dev:
            ids = model.encode(" " + phrase)
            addr = model.hasher.addresses(ids)[:, li]
            for a in addr:
                xs.append(model.rows.get(layer, a).reshape(-1))
        x = np.stack(xs)
        exact = model.decode(x @ wv.T)
        dev_r = {}
        for k, zq in stages.items():
            ap = approx_topk(zq, m[layer], x)
            dev_r[f"{k * CHUNK_DIMS}"] = {
                "top10_overlap": round(_overlap(exact, ap), 3),
                "exact_top1_in_top10": round(
                    float(np.mean([e[0] in set(a) for e, a in zip(exact, ap, strict=True)])), 3
                ),
            }
        # benchmark phrases: last position, hit@10 with the exact and the shipped readout
        xb, accepts = [], []
        for item in bench:
            ids = model.encode(" " + item["phrase"])
            xb.append(model.rows.get(layer, model.hasher.addresses(ids)[-1, li]).reshape(-1))
            accepts.append(scoring.accept_set(item["related"], ids, strings))
        xb = np.stack(xb)
        ex_b = model.decode(xb @ wv.T)
        bench_r = {
            "exact": round(float(np.mean([scoring.hit(t, strings, a) for t, a in zip(ex_b, accepts, strict=True)])), 3)
        }
        for k, zq in stages.items():
            ap = approx_topk(zq, m[layer], xb)
            bench_r[f"{k * CHUNK_DIMS}"] = round(
                float(np.mean([scoring.hit(t, strings, a) for t, a in zip(ap, accepts, strict=True)])), 3
            )
        out["layers"][str(layer)] = {"dev_positions": int(x.shape[0]), "dev": dev_r, "bench_hit10": bench_r}
        log(f"agreement layer {layer}: {json.dumps(out['layers'][str(layer)])}")
    return out


def _agreement_report(a: dict, meta: dict) -> str:
    lines = [
        f"# Decode index: agreement with the exact readout ({meta['date']})",
        "",
        f"Generated by `build_assets.py` at code commit {meta['commit']}; model revision {REVISION}. Req 5.3, Task 4.2.",
        "",
        f"The shipped index: d = {D}, 4-bit, {CHUNKS} chunks of {CHUNK_DIMS} dimensions; one projection fitted on layer {FIT_LAYER}, shared by both layers.",
        "The browser decodes with the first chunk as soon as it arrives and sharpens as the rest load, so every loading stage is measured.",
        "",
        f"Top-10 overlap is |exact ∩ shipped| / 10 over all token positions of the {a['dev_phrases']} dev phrases (`bench/dev_phrases.txt`, never scored).",
        f"hit@10 is the benchmark rule on the {a['bench_phrases']} pre-registered phrases (`bench/phrases.jsonl`), last position.",
        "",
        "| Layer | Dimensions loaded | Top-10 overlap | Exact top-1 in top 10 | Benchmark hit@10 (exact: see right) | Exact hit@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for layer, r in a["layers"].items():
        for dims, d in r["dev"].items():
            lines.append(
                f"| {layer} | {dims} | {d['top10_overlap']:.3f} | {d['exact_top1_in_top10']:.3f} | {r['bench_hit10'][dims]:.3f} | {r['bench_hit10']['exact']:.3f} |"
            )
    lines += ["", "Numbers come from the JSON next to this file; nothing here is typed by hand."]
    return "\n".join(lines) + "\n"


def _test_vectors(model: Model, m: dict, zfull: tuple) -> dict:
    cfg = model.cfg
    dev = [line.strip() for line in Path("bench/dev_phrases.txt").read_text(encoding="utf-8").splitlines()]
    phrases = [p for p in dev if p and not p.startswith("#")] + EXTRA_PHRASES
    zq = dequantize(*zfull)
    items = []
    for i, phrase in enumerate(phrases):
        text = " " + phrase
        ids = model.encode(text)
        addr = model.hasher.addresses(ids)
        item = {"phrase": phrase, "text": text, "ids": [int(t) for t in ids], "addresses": addr.tolist()}
        if i < READOUT_VECTORS:
            item["rows"] = {}
            for li, layer in enumerate(cfg.layer_ids):
                model.rows.prefetch(layer, [int(r) for r in addr[-1, li]])  # parallel fetch into the row cache
                raw_v, raw_s, xs = [], [], []
                for r in addr[-1, li]:
                    q = "select v, s from rows where layer = ? and row = ?"
                    vb, sb = model.rows.db.execute(q, (layer, int(r))).fetchone()
                    raw_v.append(bytes(vb).hex())
                    raw_s.append(bytes(sb).hex())
                    xs.append(fp8.decode_rows(np.frombuffer(vb, np.uint8)[None], np.frombuffer(sb, np.uint8)[None])[0])
                x = np.concatenate(xs)[None]
                item["rows"][str(layer)] = {
                    "values_hex": raw_v,
                    "scales_hex": raw_s,
                    "approx_top10": approx_topk(zq, m[layer], x)[0].tolist(),
                    "exact_top10": model.decode(x @ model.value_matrix(layer).T)[0].tolist(),
                }
        items.append(item)
    return {"revision": REVISION, "note": "Generated by build_assets.py; do not edit.", "phrases": items}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="web/public/assets")
    ap.add_argument("--skip-agreement", action="store_true", help="skip the Req 5.3 measurement (faster rebuilds)")
    args = ap.parse_args()

    def log(msg: str) -> None:
        print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)

    out = Path(args.out) / REVISION
    out.mkdir(parents=True, exist_ok=True)
    model = Model()
    cfg = model.cfg
    log("model loaded")

    # tokenizer and token map
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(hub.file(name), out / name)
    tmap = model.hasher.token_map.astype("<u4")
    (out / "token_map.bin").write_bytes(tmap.tobytes())
    strings = [model.tokenizer.decode([i]) for i in range(model.vocab)]
    (out / "strings.json").write_text(json.dumps(strings, ensure_ascii=False), encoding="utf-8")
    log("tokenizer, token map, strings written")

    fit = _fit(model, log)
    z = model.embeddings() @ fit["p"]
    packed, scales = quantize(z)
    del z
    files: dict[str, dict] = {}

    def write(name: str, data: bytes) -> None:
        (out / name).write_bytes(data)
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    for k in range(CHUNKS):
        write(f"z{k}.bin", packed[k].tobytes())
        write(f"zs{k}.bin", scales[k].astype("<f2").tobytes())
    for layer in cfg.layer_ids:
        for k in range(CHUNKS):
            write(f"m{layer}-{k}.bin", fit["m"][layer][k * CHUNK_DIMS : (k + 1) * CHUNK_DIMS].astype("<f2").tobytes())
    log("decode index written")

    tensors = {}
    for layer in cfg.layer_ids:
        v, s = model.rows.tensors[layer]
        tensors[str(layer)] = {"shard": v.shard, "values_start": v.start, "scales_start": s.start, "rows": v.shape[0]}
    hasher = model.hasher
    constants = {
        "repo": REPO,
        "revision": REVISION,
        "hub": f"https://huggingface.co/{REPO}/resolve/{REVISION}/",
        "layer_ids": list(cfg.layer_ids),
        "max_ngram_size": cfg.max_ngram_size,
        "n_heads": cfg.n_heads,
        "n_cols": cfg.n_cols,
        "head_dim": cfg.head_dim,
        "hidden": model.hidden,
        "vocab": model.vocab,
        "token_map_size": int(tmap.shape[0]),
        "compressed_vocab_size": cfg.compressed_vocab_size,
        "pad": hasher.pad,
        "primes": hasher.primes.tolist(),
        "offsets": hasher.offsets.tolist(),
        "multipliers": [[str(int(x)) for x in row] for row in hasher.multipliers],
        "num_embeddings": list(cfg.num_embeddings),
        "tensors": tensors,
        "decode_index": {
            "method": "readout-fitted projection (G1 variant block), shared by both layers, fitted on layer 1",
            "d": D,
            "chunks": CHUNKS,
            "chunk_dims": CHUNK_DIMS,
            "bits": 4,
            "qmax": QMAX,
            "energy_kept": round(fit["energy_kept"], 4),
            "files": files,
        },
    }
    (out / "constants.json").write_text(json.dumps(constants, indent=1))
    (out / "test-vectors.json").write_text(
        json.dumps(_test_vectors(model, fit["m"], (packed, scales)), ensure_ascii=False)
    )
    log("constants and test vectors written")

    if not args.skip_agreement:
        import subprocess

        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        meta = {
            "date": datetime.date.today().isoformat(),
            "commit": commit,
            "revision": REVISION,
            "d": D,
            "chunks": CHUNKS,
        }
        a = _agreement(model, (packed, scales), fit["m"], log)
        res = Path("docs/results")
        res.mkdir(parents=True, exist_ok=True)
        (res / f"decode-index-{meta['date']}.json").write_text(json.dumps({"meta": meta, **a}, indent=1))
        (res / f"decode-index-{meta['date']}.md").write_text(_agreement_report(a, meta))
        log("agreement written")
    total = sum(f.stat().st_size for f in out.iterdir())
    log(f"done: {out} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
