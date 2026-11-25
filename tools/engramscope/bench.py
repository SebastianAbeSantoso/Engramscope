"""Readout benchmark (Req 6) on the pre-registered phrase set, following the analysis plan in bench/README.md.

    uv run python -m engramscope.bench

Writes docs/results/bench-<date>.json (every decoded list) and bench-<date>.md (the tables).
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from . import provenance
from .readout import Model
from .scoring import accept_set, boot_indices, hit, mean_ci

PHRASES = "bench/phrases.jsonl"
REGISTERED = "ec0e429"
SEED = 20261111
SHIFT = 61
RESAMPLES = 2000
KS = (1, 5, 10)
CONTROLS = ("layer 14", "random rows L1", "last-token embedding", "shuffled")


def conditions(model: Model, phrases: list[dict]) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Value vectors per condition, [n, hidden] each, plus per-phrase details."""
    cfg = model.cfg
    rng = np.random.default_rng(SEED)
    details = []
    for p in phrases:
        ids = model.encode(" " + p["phrase"])
        addr = model.addresses(ids)
        rand = np.array(
            [
                [
                    int(model.hasher.offsets[li, c]) + int(rng.integers(0, int(model.hasher.primes[li, c])))
                    for c in range(cfg.n_cols)
                ]
                for li in range(len(cfg.layer_ids))
            ],
            dtype=np.int64,
        )
        details.append({"id": p["id"], "ids": ids, "addr": addr, "rand": rand})
    for li, layer in enumerate(cfg.layer_ids):
        need = {int(r) for d in details for r in np.concatenate([d["addr"][li], d["rand"][li]])}
        print(f"layer {layer}: {model.rows.prefetch(layer, need)} rows fetched of {len(need)}", flush=True)

    n = len(phrases)
    vec = {
        name: np.zeros((n, model.hidden), np.float32)
        for name in ("layer 1", "layer 14", "random rows L1", "random rows L14", "last-token embedding", "shuffled")
    }
    emb = model.embeddings()
    for i, d in enumerate(details):
        for li, layer in enumerate(cfg.layer_ids):
            vec[f"layer {layer}"][i] = model.contributions(layer, d["addr"][li]).sum(axis=0)
            vec[f"random rows L{layer}"][i] = model.contributions(layer, d["rand"][li]).sum(axis=0)
        vec["last-token embedding"][i] = emb[d["ids"][-1]]
    partner = [((i + SHIFT) % n) for i in range(n)]
    vec["shuffled"] = vec["layer 1"][partner]
    for i, d in enumerate(details):
        d["shuffled_partner"] = phrases[partner[i]]["id"]
    return vec, details


def main() -> None:
    registered = provenance.require_registered(PHRASES, REGISTERED)
    phrases = [json.loads(line) for line in open(provenance.ROOT / PHRASES, encoding="utf-8")]
    if [p["id"] for p in phrases] != list(range(1, len(phrases) + 1)):
        raise SystemExit("phrase ids must be 1..n in file order")
    cats = [p["category"] for p in phrases]
    n = len(phrases)
    for i in range(n):
        if cats[i] == cats[(i + SHIFT) % n]:
            raise SystemExit("the shuffled pairing must never pair a phrase with its own category")

    model = Model()
    strings = model.token_strings()
    vec, details = conditions(model, phrases)
    top = {name: model.decode(v) for name, v in vec.items()}

    accept = [accept_set(p["related"], d["ids"], strings) for p, d in zip(phrases, details, strict=True)]
    hits = {
        name: {k: np.array([hit(t[i], strings, accept[i], k) for i in range(n)]) for k in KS} for name, t in top.items()
    }

    idx = boot_indices(n, RESAMPLES, SEED)
    table = {name: {k: mean_ci(h[k], idx) for k in KS} for name, h in hits.items()}
    beats = {}
    for control in CONTROLS:
        diff = hits["layer 1"][10].astype(float) - hits[control][10].astype(float)
        mean, lo, hi = mean_ci(diff, idx)
        beats[control] = {"mean_diff_hit10": mean, "ci95": [lo, hi], "beaten": lo > 0}
    by_cat = defaultdict(list)
    for i, c in enumerate(cats):
        by_cat[c].append(i)
    per_cat = {c: {name: float(hits[name][10][ix].mean()) for name in hits} for c, ix in by_cat.items()}

    per_phrase = []
    for i, (p, d) in enumerate(zip(phrases, details, strict=True)):
        per_phrase.append(
            {
                "id": p["id"],
                "phrase": p["phrase"],
                "category": p["category"],
                "token_ids": d["ids"],
                "accept": sorted(accept[i]),
                "addresses": {f"L{layer}": d["addr"][li].tolist() for li, layer in enumerate(model.cfg.layer_ids)},
                "random_rows": {f"L{layer}": d["rand"][li].tolist() for li, layer in enumerate(model.cfg.layer_ids)},
                "shuffled_partner": d["shuffled_partner"],
                "decoded": {name: [strings[int(t)] for t in top[name][i]] for name in top},
                "top10_ids": {name: top[name][i].tolist() for name in top},
                "hits": {name: {str(k): bool(hits[name][k][i]) for k in KS} for name in hits},
            }
        )
    result = {
        "provenance": provenance.record(phrases=registered),
        "plan": "bench/README.md, 'Analysis plan for phrases.jsonl'",
        "n_phrases": n,
        "table": {
            name: {str(k): {"mean": m, "ci95": [lo, hi]} for k, (m, lo, hi) in row.items()}
            for name, row in table.items()
        },
        "beats": beats,
        "layer1_beats_every_control": all(b["beaten"] for b in beats.values()),
        "hit10_by_category": per_cat,
        "phrases": per_phrase,
    }
    out = provenance.ROOT / "docs" / "results" / f"bench-{provenance.today()}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    out.with_suffix(".md").write_text(report(result, table, beats, per_cat))
    print(out.with_suffix(".md").read_text())


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def report(result: dict, table: dict, beats: dict, per_cat: dict) -> str:
    p = result["provenance"]
    lines = [
        f"# Readout benchmark, {p['run_at'][:10]}",
        "",
        f"Generated by `tools/engramscope/bench.py` at code commit `{p['code_commit'][:12]}`"
        f"{'' if p['code_clean'] else ' (with uncommitted changes)'}, on `{PHRASES}` as registered in"
        f" `{p['registered']['phrases']['commit'][:12]}`, model revision `{p['revision'][:12]}`. Exact readout, top 10,"
        f" {result['n_phrases']} phrases, 95% percentile bootstrap intervals over phrases ({RESAMPLES:,} resamples)."
        " Analysis plan: `bench/README.md`. Raw output: the `.json` file next to this one.",
        "",
        "| Condition | hit@1 | hit@5 | hit@10 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, row in table.items():
        cells = [f"{pct(m)} [{pct(lo)}, {pct(hi)}]" for m, lo, hi in (row[k] for k in KS)]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Does layer 1 beat every control? (Req 6.5, paired difference in hit@10)",
        "",
        "| Control | Layer 1 minus control | 95% interval | Beaten |",
        "| --- | ---: | ---: | --- |",
    ]
    for control, b in beats.items():
        lo, hi = b["ci95"]
        lines.append(
            f"| {control} | {pct(b['mean_diff_hit10'])} | [{pct(lo)}, {pct(hi)}] | {'yes' if b['beaten'] else 'no'} |"
        )
    verdict = "yes" if result["layer1_beats_every_control"] else "no"
    lines += ["", f"**Layer 1 beats every control: {verdict}.**", "", "## Hit@10 by category (descriptive)", ""]
    names = list(next(iter(per_cat.values())).keys())
    lines += ["| Category | " + " | ".join(names) + " |", "| --- |" + " ---: |" * len(names)]
    for c, row in per_cat.items():
        lines.append(f"| {c} | " + " | ".join(pct(row[nm]) for nm in names) + " |")
    lines += ["", "Decoding by nearest input embeddings is an approximation; see the README's limitations.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
