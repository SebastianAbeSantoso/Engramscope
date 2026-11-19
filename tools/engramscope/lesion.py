"""Lesion experiments (Req 9), exactly as pre-registered in bench/lesion-protocol.md.

    uv run python -m engramscope.lesion

Runs after the readout benchmark at the same code commit; its random-rows control is the chance level.
Writes docs/results/lesion-<date>.json (measures and rule outcomes), lesion-<date>.md (the report) and
lesion-<date>-decoded.json.gz (every decoded top-10 list, including each random subset).

Every result here is a trace lesion (readout): traces are removed from the value-vector sum before
decoding. It shows how a stored concept is spread across traces, not what the running model forgets.
"""

from __future__ import annotations

import gzip
import json
import math

import numpy as np

from . import hub, provenance
from .readout import Model
from .scoring import accept_set, boot_indices, crossing, curve_shape, hit, mean_ci

PROTOCOL = "bench/lesion-protocol.md"
APPROVED = "9d79b32521022954761061ecefb4ea56a45be736"
PHRASES = "bench/phrases.jsonl"
REGISTERED = "ec0e429"
SEED = 20261111
K_SIZES = (1, 2, 3, 4, 8, 12, 16, 20, 22, 23)
N_SUBSETS = 50
PREFIXES = (
    "I read about",
    "Yesterday we talked about",
    "There is a picture of",
    "Nobody knows much about",
    "Saya pernah membaca tentang",
)
RESAMPLES = 2000
RESAMPLES_CORRECTED = 10000
POWER_GUARD = 30
BAND = 0.1
N_COLS, N_HEADS, SIZES = 24, 8, (2, 3, 4)


def size_cols(n: int) -> list[int]:
    return [(n - 2) * N_HEADS + h for h in range(N_HEADS)]


def head_cols(h: int) -> list[int]:
    return [(n - 2) * N_HEADS + h for n in SIZES]


def condition_names() -> list[tuple]:
    names: list[tuple] = [("A0",)]
    names += [("A", k, j) for k in K_SIZES for j in range(N_SUBSETS)]
    names += [("B1", i) for i in range(N_COLS)] + [("B2", i) for i in range(N_COLS)]
    names += [("B3", n) for n in SIZES] + [("B4", h) for h in range(N_HEADS)] + [("C1", n) for n in SIZES]
    return names


def removed_traces(name: tuple, subsets: dict[int, np.ndarray]) -> list[int]:
    kind = name[0]
    if kind == "A0":
        return []
    if kind == "A":
        return [int(c) for c in subsets[name[1]][name[2]]]
    if kind == "B1":
        return [name[1]]
    if kind == "B2":
        return [c for c in range(N_COLS) if c != name[1]]
    if kind == "B3":
        return size_cols(name[1])
    if kind == "B4":
        return head_cols(name[1])
    if kind == "C1":
        return [c for c in range(N_COLS) if c not in size_cols(name[1])]
    raise ValueError(name)


def keep_masks(names: list[tuple], subsets: dict[int, np.ndarray]) -> np.ndarray:
    masks = np.ones((len(names), N_COLS), np.float32)
    for j, name in enumerate(names):
        masks[j, removed_traces(name, subsets)] = 0
    return masks


def draw_subsets(phrase_ids: list[int], layers: tuple[int, ...]) -> dict[tuple[int, int, int], np.ndarray]:
    """Random lesion subsets, drawn in the order phrase id, then layer, then k (protocol, Section 4)."""
    rng = np.random.default_rng(SEED)
    out = {}
    for pid in phrase_ids:
        for layer in layers:
            for k in K_SIZES:
                out[(pid, layer, k)] = np.stack([rng.choice(N_COLS, size=k, replace=False) for _ in range(N_SUBSETS)])
    return out


def finite(obj):
    """NaN and infinity become None, so the JSON stays valid for the web app."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [finite(v) for v in obj]
    return obj


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b) + 1e-30)


def run_layer(
    model: Model, layer: int, phrases: list[dict], bare: list[list[int]], subsets: dict, accept: list[set]
) -> dict:
    """All conditions for one layer. Returns arrays indexed [phrase, condition]."""
    cfg, strings = model.cfg, model.token_strings()
    li = cfg.layer_ids.index(layer)
    names = condition_names()
    ctx_ids = [[model.encode(prefix + " " + p["phrase"]) for prefix in PREFIXES] for p in phrases]
    bare_addr = [model.addresses(ids)[li] for ids in bare]
    ctx_addr = [[model.addresses(ids)[li] for ids in row] for row in ctx_ids]
    need = {int(r) for a in bare_addr for r in a} | {int(r) for row in ctx_addr for a in row for r in a}
    print(f"layer {layer}: {model.rows.prefetch(layer, need)} new rows fetched of {len(need)}", flush=True)

    n, m = len(phrases), len(names)
    top = np.zeros((n, m + len(PREFIXES), 10), np.int32)
    m3 = np.zeros((n, m), np.float32)
    shares = np.zeros((n, N_COLS), np.float32)
    pair_cos = np.zeros(n, np.float32)
    size_cos = np.zeros((n, len(SIZES)), np.float32)
    recur = np.zeros((n, len(PREFIXES), len(SIZES)), np.int32)
    kept = np.zeros((n, len(PREFIXES)), bool)
    for i, p in enumerate(phrases):
        contrib = model.contributions(layer, bare_addr[i])  # [24, hidden]
        vectors = keep_masks(names, {k: subsets[(p["id"], layer, k)] for k in K_SIZES}) @ contrib
        ctx = np.stack([model.contributions(layer, a).sum(axis=0) for a in ctx_addr[i]])
        top[i] = model.decode(np.concatenate([vectors, ctx]))
        v = vectors[0]
        m3[i] = cosine(vectors, v)
        energy = (contrib.astype(np.float64) ** 2).sum(axis=1)
        shares[i] = energy / energy.sum()
        unit = contrib / np.linalg.norm(contrib, axis=1, keepdims=True)
        gram = unit @ unit.T
        pair_cos[i] = (gram.sum() - np.trace(gram)) / (N_COLS * (N_COLS - 1))
        size_cos[i] = [cosine(contrib[size_cols(s)].sum(axis=0), v) for s in SIZES]
        for q in range(len(PREFIXES)):
            recur[i, q] = [int((ctx_addr[i][q][size_cols(s)] == bare_addr[i][size_cols(s)]).sum()) for s in SIZES]
            kept[i, q] = ctx_ids[i][q][-len(bare[i]) :] == bare[i]
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"layer {layer}: {i + 1}/{n} phrases", flush=True)
    m1 = np.array([[hit(top[i, j], strings, accept[i]) for j in range(top.shape[1])] for i in range(n)])
    m4 = np.array([[len(set(top[i, j]) & set(top[i, 0])) / 10 for j in range(m)] for i in range(n)], np.float32)
    return {
        "names": names,
        "top": top,
        "m1": m1,
        "m3": m3,
        "m4": m4,
        "shares": shares,
        "pair_cos": pair_cos,
        "size_cos": size_cos,
        "recur": recur,
        "phrase_tokens_kept": kept,
    }


def ci_pct(level: float, family: int) -> tuple[float, float]:
    alpha = (1 - level) / family
    return 100 * alpha / 2, 100 * (1 - alpha / 2)


def boot_mean(values: np.ndarray, idx: np.ndarray, lo_pct: float, hi_pct: float) -> dict:
    values = np.asarray(values, np.float64)
    means = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci": [float(np.percentile(means, lo_pct)), float(np.percentile(means, hi_pct))],
    }


def analyse(work: dict, chance: np.ndarray, lengths: np.ndarray) -> dict:
    names = work["names"]
    col = {name: j for j, name in enumerate(names)}
    m1, m3, m4 = work["m1"], work["m3"], work["m4"]
    n_all = m1.shape[0]
    intact = m1[:, col[("A0",)]]
    hset = np.flatnonzero(intact)
    n_h = len(hset)
    out: dict = {"n_phrases": n_all, "n_intact_hits": n_h, "labels_allowed": n_h >= POWER_GUARD}
    if n_h < 2:
        out["note"] = "too few phrases with an intact hit for any statistic"
        return out
    idx = boot_indices(n_h, RESAMPLES, SEED)
    idx_c = boot_indices(n_h, RESAMPLES_CORRECTED, SEED)

    def retention(k: int) -> np.ndarray:  # per phrase (all phrases), mean over its 50 subsets
        return np.mean([m1[:, col[("A", k, j)]] for j in range(N_SUBSETS)], axis=0)

    ret = {k: retention(k) for k in K_SIZES}
    grid = np.array([0, *K_SIZES], float)
    per_phrase = np.stack([np.ones(n_all)] + [ret[k] for k in K_SIZES], axis=1)[hset]  # [n_h, len(grid)]
    curve = per_phrase.mean(axis=0)
    cis = []
    for g in range(len(grid)):
        _, lo, hi = mean_ci(per_phrase[:, g], idx)
        cis.append([lo, hi])
    boot_curves = per_phrase[idx].mean(axis=1)  # [R, len(grid)]
    s_boot = np.array([curve_shape(grid, c) for c in boot_curves])
    k50_boot = np.array([crossing(grid, c) for c in boot_curves])
    q1: dict = {
        "k": grid.tolist(),
        "retention": curve.tolist(),
        "retention_ci95": cis,
        "S": curve_shape(grid, curve),
        "S_ci95": [float(np.nanpercentile(s_boot, 2.5)), float(np.nanpercentile(s_boot, 97.5))],
        "k50": crossing(grid, curve),
        "k50_ci95": [float(np.nanpercentile(k50_boot, 2.5)), float(np.nanpercentile(k50_boot, 97.5))]
        if np.isfinite(k50_boot).any()
        else None,
        "k50_resamples_without_crossing": int(np.isnan(k50_boot).sum()),
        "m3_all_phrases": [1.0]
        + [float(np.mean([m3[:, col[("A", k, j)]] for j in range(N_SUBSETS)])) for k in K_SIZES],
        "m3_orthogonal_reference": [math.sqrt((N_COLS - k) / N_COLS) for k in grid],
        "m4_all_phrases": [1.0]
        + [float(np.mean([m4[:, col[("A", k, j)]] for j in range(N_SUBSETS)])) for k in K_SIZES],
    }
    rises = [
        (int(grid[g]), int(grid[g + 1]))
        for g in range(len(grid) - 1)
        if curve[g + 1] > curve[g] and cis[g + 1][0] > cis[g][1]
    ]
    q1["rises_with_disjoint_intervals"] = rises
    if curve[-1] >= 0.5:
        q1["label"] = "redundant: most concepts survive with a single trace"
    elif rises:
        q1["label"] = "irregular"
    else:
        lo, hi = q1["S_ci95"]
        if -BAND <= lo and hi <= BAND:
            q1["label"] = "proportional, as in Lashley's mass action"
        elif lo > BAND:
            q1["label"] = "late collapse: the traces stand in for each other until most are gone"
        elif hi < -BAND:
            q1["label"] = "early loss: each trace removed costs more than its share"
        else:
            q1["label"] = None
    out["Q1"] = q1

    lo_c, hi_c = ci_pct(0.95, 11)
    q2 = {"family": 11, "ci_percentiles": [lo_c, hi_c], "lesions": {}}
    for name, k in [*((("B3", s), 8) for s in SIZES), *((("B4", h), 3) for h in range(N_HEADS))]:
        d = m1[hset, col[name]].astype(float) - ret[k][hset]
        r = boot_mean(d, idx_c, lo_c, hi_c)
        lo, hi = r["ci"]
        r["retention"] = float(m1[hset, col[name]].mean())
        r["random_same_size"] = float(ret[k][hset].mean())
        r["special"] = bool(lo > 0 or hi < 0)
        r["direction"] = "more damaging than chance" if hi < 0 else "less damaging than chance" if lo > 0 else None
        q2["lesions"][f"{name[0]}({name[1]})"] = r
    special_heads = [k for k, v in q2["lesions"].items() if k.startswith("B4") and v["special"]]
    special_sizes = [k for k, v in q2["lesions"].items() if k.startswith("B3") and v["special"]]
    q2["heads_label"] = (
        "equipotential across heads" if not special_heads else f"not equipotential across heads: {special_heads}"
    )
    q2["sizes_label"] = (
        "equipotential across n-gram sizes"
        if not special_sizes
        else f"not equipotential across n-gram sizes: {special_sizes}"
    )
    flips = np.array([sum(1 - m1[i, col[("B1", c)]] for c in range(N_COLS)) for i in hset])
    q2["B1"] = {
        "share_any_single_trace_breaks_hit": float((flips >= 1).mean()),
        "share_single_point": float((flips == 1).mean()),
        "mean_traces_that_break_hit": float(flips.mean()),
    }
    spare = np.array([sum(m1[i, col[("B2", c)]] for c in range(N_COLS)) for i in hset])
    q2["B2_mean_single_traces_that_keep_hit"] = float(spare.mean())
    out["Q2"] = q2

    lo3, hi3 = ci_pct(0.95, 3)
    q3a = {"family": 3, "ci_percentiles": [lo3, hi3], "sizes": {}}
    for s in SIZES:
        d = m1[hset, col[("C1", s)]].astype(float) - chance[hset]
        r = boot_mean(d, idx_c, lo3, hi3)
        r["retention"] = float(m1[hset, col[("C1", s)]].mean())
        r["chance"] = float(chance[hset].mean())
        r["retrieves_above_chance"] = bool(r["ci"][0] > 0)
        q3a["sizes"][str(s)] = r
    out["Q3a"] = q3a
    m = len(names)
    q3b = {}
    for q, prefix in enumerate(PREFIXES):
        mean, lo, hi = mean_ci(m1[hset, m + q].astype(float), idx)
        q3b[prefix] = {
            "retention": mean,
            "ci95": [lo, hi],
            "hit_rate_all_phrases": float(m1[:, m + q].mean()),
            "share_of_traces_recurring_by_size": {
                str(s): float(work["recur"][:, q, si].mean() / N_HEADS) for si, s in enumerate(SIZES)
            },
            "phrase_tokens_kept": int(work["phrase_tokens_kept"][:, q].sum()),
        }
    out["Q3b"] = q3b

    bins = {"1": lengths == 1, "2": lengths == 2, "3": lengths == 3, ">=4": lengths >= 4}
    sec = {}
    for b, mask in bins.items():
        members = np.flatnonzero(mask[hset])
        entry: dict = {"n_phrases": int(mask.sum()), "n_intact_hits": int(len(members))}
        if len(members) >= 2:
            bidx = boot_indices(len(members), RESAMPLES, SEED)
            rows = hset[members]
            for s in SIZES:
                d3 = m1[rows, col[("B3", s)]].astype(float) - ret[8][rows]
                entry[f"B3({s})"] = boot_mean(d3, bidx, 2.5, 97.5)
                dc = m1[rows, col[("C1", s)]].astype(float) - chance[rows]
                entry[f"C1({s})"] = boot_mean(dc, bidx, 2.5, 97.5)
        sec[b] = entry
    out["secondary_by_length"] = sec

    out["structure"] = {
        "mean_share_by_column": work["shares"].mean(axis=0).tolist(),
        "mean_share_by_size": [float(work["shares"][:, size_cols(s)].sum(axis=1).mean()) for s in SIZES],
        "mean_share_by_head": [float(work["shares"][:, head_cols(h)].sum(axis=1).mean()) for h in range(N_HEADS)],
        "mean_pairwise_cosine": float(work["pair_cos"].mean()),
        "mean_cosine_of_size_sum_with_v": work["size_cos"].mean(axis=0).tolist(),
    }
    two = sec["2"]
    out["predictions"] = {
        "P1_no_head_lesion_special": not special_heads,
        "P2_some_size_lesion_special": bool(special_sizes),
        "P2_two_token_bin": None
        if "B3(2)" not in two
        else {
            "B3(2)_more_damaging": two["B3(2)"]["ci"][1] < 0,
            "B3(3)_not_more_damaging": not two["B3(3)"]["ci"][1] < 0,
            "B3(4)_not_more_damaging": not two["B3(4)"]["ci"][1] < 0,
        },
    }
    return out


def main() -> None:
    reg_protocol = provenance.require_registered(PROTOCOL, APPROVED)
    reg_phrases = provenance.require_registered(PHRASES, REGISTERED)
    head = provenance.git("rev-parse", "HEAD")
    benches = sorted((provenance.ROOT / "docs" / "results").glob("bench-*.json"))
    if not benches:
        raise SystemExit("run the readout benchmark first (it supplies the chance level)")
    bench = json.loads(benches[-1].read_text())
    if bench["provenance"]["code_commit"] != head:
        raise SystemExit(f"{benches[-1].name} was made at another code commit; rerun the benchmark first")

    phrases = [json.loads(line) for line in open(provenance.ROOT / PHRASES, encoding="utf-8")]
    model = Model()
    strings = model.token_strings()
    layers = model.cfg.layer_ids
    bare = [model.encode(" " + p["phrase"]) for p in phrases]
    accept = [accept_set(p["related"], ids, strings) for p, ids in zip(phrases, bare, strict=True)]
    lengths = np.array([len(ids) for ids in bare])
    subsets = draw_subsets([p["id"] for p in phrases], layers)

    results, decoded = {}, {}
    for layer in layers:
        work_path = hub.cache_dir() / f"lesion-work-{APPROVED[:8]}-{head[:8]}-L{layer}.npz"
        if work_path.exists():
            stored = np.load(work_path)
            work = {key: stored[key] for key in stored.files} | {"names": condition_names()}
        else:
            work = run_layer(model, layer, phrases, bare, subsets, accept)
            np.savez_compressed(work_path, **{k: v for k, v in work.items() if k != "names"})
        bench_phrases = {bp["id"]: bp for bp in bench["phrases"]}
        chance = np.array([float(bench_phrases[p["id"]]["hits"][f"random rows L{layer}"]["10"]) for p in phrases])
        results[f"L{layer}"] = analyse(work, chance, lengths)
        names = work["names"] + [("context", q) for q in range(len(PREFIXES))]
        decoded[f"L{layer}"] = {
            "conditions": ["/".join(str(x) for x in nm) for nm in names],
            "top10": {str(p["id"]): work["top"][i].tolist() for i, p in enumerate(phrases)},
        }
        results[f"L{layer}"]["per_phrase"] = [
            {
                "id": p["id"],
                "phrase": p["phrase"],
                "tokens": int(lengths[i]),
                "intact_top10": [strings[int(t)] for t in work["top"][i, 0]],
                "intact_hit": bool(work["m1"][i, 0]),
                "retention_by_k": {
                    str(k): float(
                        np.mean([work["m1"][i, 1 + K_SIZES.index(k) * N_SUBSETS + j] for j in range(N_SUBSETS)])
                    )
                    for k in K_SIZES
                },
                "structured": {
                    "/".join(str(x) for x in nm): {
                        "hit": bool(work["m1"][i, j]),
                        "m3": float(work["m3"][i, j]),
                        "m4": float(work["m4"][i, j]),
                        "top10": [strings[int(t)] for t in work["top"][i, j]],
                    }
                    for j, nm in enumerate(work["names"])
                    if nm[0] != "A"
                },
                "contexts": {
                    prefix: {
                        "hit": bool(work["m1"][i, len(work["names"]) + q]),
                        "top10": [strings[int(t)] for t in work["top"][i, len(work["names"]) + q]],
                        "recurring_traces_by_size": work["recur"][i, q].tolist(),
                    }
                    for q, prefix in enumerate(PREFIXES)
                },
            }
            for i, p in enumerate(phrases)
        ]

    stem = provenance.ROOT / "docs" / "results" / f"lesion-{provenance.today()}"
    record = provenance.record(protocol=reg_protocol, phrases=reg_phrases)
    record["chance_level_from"] = benches[-1].name
    summary = finite({"provenance": record, "label": "trace lesion (readout)", "layers": results})
    stem.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    with gzip.open(stem.parent / f"{stem.name}-decoded.json.gz", "wt", encoding="utf-8") as f:
        json.dump(
            {
                "provenance": record,
                "subsets": {f"{a}/{b}/{c}": v.tolist() for (a, b, c), v in subsets.items()},
                "layers": decoded,
            },
            f,
        )
    from .lesion_report import render

    stem.with_suffix(".md").write_text(render(summary, strings))
    print(stem.with_suffix(".md").read_text())


if __name__ == "__main__":
    main()
