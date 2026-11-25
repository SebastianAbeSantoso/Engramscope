"""Markdown report of the lesion experiments. The wording comes from the approved rules only."""

from __future__ import annotations


def _p(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _pp(x: float) -> str:
    return f"{100 * x:+.1f} pts"


def _ci(ci: list[float] | None, unit: str = "%") -> str:
    if not ci:
        return "n/a"
    if unit == "pts":
        return f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"
    if unit == "raw":
        return f"[{ci[0]:.2f}, {ci[1]:.2f}]"
    return f"[{100 * ci[0]:.1f}%, {100 * ci[1]:.1f}%]"


def _layer(name: str, r: dict, primary: bool) -> list[str]:
    title = f"## {name}" + (" (primary)" if primary else " (secondary)")
    lines = [title, ""]
    lines.append(f"{r['n_phrases']} phrases; {r['n_intact_hits']} have a concept hit when intact (M1 under A0).")
    if "Q1" not in r:
        return [*lines, "", r.get("note", ""), ""]
    labels = r["labels_allowed"]
    if not labels:
        lines.append(
            "**Power guard:** fewer than 30 intact hits, so no labels are applied to this layer; numbers only."
        )
    lines.append("")

    q1 = r["Q1"]
    lines += ["### Q1, mass action: retention as traces are removed at random", ""]
    lines += [
        "| Traces removed (k) | Retention (M2) | 95% interval | Vector similarity (M3, all phrases) | Orthogonal reference | List overlap (M4) |"
    ]
    lines += ["| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for g, k in enumerate(q1["k"]):
        lines.append(
            f"| {int(k)} | {_p(q1['retention'][g])} | {_ci(q1['retention_ci95'][g])} | {q1['m3_all_phrases'][g]:.3f} |"
            f" {q1['m3_orthogonal_reference'][g]:.3f} | {_p(q1['m4_all_phrases'][g])} |"
        )
    k50 = "never (retention stays above 50%)" if q1["k50"] is None else f"{q1['k50']:.1f}"
    shape = "n/a" if q1["S"] is None else f"{q1['S']:.3f}"
    lines += [
        "",
        f"- Shape statistic S = {shape}, 95% interval {_ci(q1['S_ci95'], 'raw')} (0 = proportional; +0.5 = holds then collapses; −0.5 = collapses at once).",
        f"- k50 (traces removed when retention reaches 50%): {k50}, 95% interval {_ci(q1['k50_ci95'], 'raw')}.",
    ]
    if labels:
        lines.append(f"- **Rule outcome:** {q1['label'] or 'no label (the interval of S straddles a band edge)'}.")
    lines.append("")

    q2 = r["Q2"]
    lines += ["### Q2, equipotentiality: structured lesions against random lesions of the same size", ""]
    lines += ["| Lesion | Traces removed | Retention | Random, same size | Difference | Corrected interval | Special |"]
    lines += ["| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for key, v in q2["lesions"].items():
        size = 8 if key.startswith("B3") else 3
        what = f"all {key[3]}-gram traces" if key.startswith("B3") else f"head {key[3]} (all sizes)"
        lines.append(
            f"| {key}: {what} | {size} | {_p(v['retention'])} | {_p(v['random_same_size'])} | {_pp(v['mean'])} |"
            f" {_ci(v['ci'], 'pts')} | {v['direction'] or 'no'} |"
        )
    lo, hi = q2["ci_percentiles"]
    b1 = q2["B1"]
    lines += [
        "",
        f"Intervals are Bonferroni-corrected over {q2['family']} comparisons (percentiles {lo:.3f} and {hi:.3f}, 10,000 resamples).",
        "",
        f"- Single-trace lesions (B1): in {_p(b1['share_any_single_trace_breaks_hit'])} of phrases with a hit, removing one trace"
        f" can break it; in {_p(b1['share_single_point'])}, exactly one trace does (single-point). On average"
        f" {b1['mean_traces_that_break_hit']:.2f} of the 24 traces break the hit alone.",
        f"- Single-trace sparing (B2): on average {q2['B2_mean_single_traces_that_keep_hit']:.2f} of the 24 traces keep the hit on their own.",
    ]
    if labels:
        lines += [
            f"- **Rule outcome, heads:** {q2['heads_label']}.",
            f"- **Rule outcome, n-gram sizes:** {q2['sizes_label']}.",
        ]
    lines.append("")

    q3a = r["Q3a"]
    lines += ["### Q3a, partial cue: one n-gram size alone (all other traces removed)", ""]
    lines += ["| Cue kept | Retention | Chance (random rows) | Difference | Corrected interval | Above chance |"]
    lines += ["| --- | ---: | ---: | ---: | ---: | --- |"]
    for s, v in q3a["sizes"].items():
        lines.append(
            f"| {s}-gram traces only | {_p(v['retention'])} | {_p(v['chance'])} | {_pp(v['mean'])} | {_ci(v['ci'], 'pts')} |"
            f" {'yes' if v['retrieves_above_chance'] else 'no'} |"
        )
    if labels:
        held = [s for s, v in q3a["sizes"].items() if v["retrieves_above_chance"]]
        text = (
            "; ".join(f"a {s}-gram cue alone retrieves the concept above chance" for s in held)
            or "no single n-gram size retrieves the concept above chance"
        )
        lines += ["", f"- **Rule outcome:** {text}."]
    lines.append("")

    lines += ["### Q3b, partial cue: the phrase after other words (no label by design)", ""]
    lines += [
        "| Context | Retention | 95% interval | Hit rate, all phrases | Traces recurring: 2-gram / 3-gram / 4-gram |"
    ]
    lines += ["| --- | ---: | ---: | ---: | ---: |"]
    for prefix, v in r["Q3b"].items():
        rec = v["share_of_traces_recurring_by_size"]
        lines.append(
            f'| "{prefix} …" | {_p(v["retention"])} | {_ci(v["ci95"])} | {_p(v["hit_rate_all_phrases"])} |'
            f" {_p(rec['2'])} / {_p(rec['3'])} / {_p(rec['4'])} |"
        )
    lines.append("")

    lines += ["### Secondary: by phrase length in tokens (descriptive, 95% intervals)", ""]
    lines += [
        "| Tokens | Phrases | Intact hits | Remove 2-gram | Remove 3-gram | Remove 4-gram | Keep only 2-gram | Keep only 3-gram | Keep only 4-gram |"
    ]
    lines += ["| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for b, e in r["secondary_by_length"].items():
        cells = []
        for key in ("B3(2)", "B3(3)", "B3(4)", "C1(2)", "C1(3)", "C1(4)"):
            cells.append(f"{_pp(e[key]['mean'])} {_ci(e[key]['ci'], 'pts')}" if key in e else "n/a")
        lines.append(f"| {b} | {e['n_phrases']} | {e['n_intact_hits']} | " + " | ".join(cells) + " |")
    lines += ["", "Remove: retention minus random lesions of 8 traces. Keep only: retention minus chance.", ""]

    st = r["structure"]
    lines += ["### Structure of the intact memory (all phrases)", ""]
    lines.append(
        "- Share of the value vector's energy by n-gram size (2 / 3 / 4): "
        + " / ".join(_p(x) for x in st["mean_share_by_size"])
        + "."
    )
    lines.append("- Share by head (0–7): " + ", ".join(_p(x) for x in st["mean_share_by_head"]) + ".")
    lines.append(f"- Mean cosine between two traces' contributions: {st['mean_pairwise_cosine']:.3f}.")
    lines.append(
        "- Cosine of each size's sum with the full vector (2 / 3 / 4): "
        + " / ".join(f"{x:.3f}" for x in st["mean_cosine_of_size_sum_with_v"])
        + "."
    )
    lines.append("")
    return lines


def render(summary: dict, strings: list[str]) -> str:
    p = summary["provenance"]
    reg = p["registered"]
    lines = [
        f"# Lesion experiments, {p['run_at'][:10]}: trace lesion (readout)",
        "",
        f"Generated by `tools/engramscope/lesion.py` at code commit `{p['code_commit'][:12]}`"
        f"{'' if p['code_clean'] else ' (with uncommitted changes)'}, following `bench/lesion-protocol.md` as approved in"
        f" `{reg['protocol']['commit'][:12]}`, on `bench/phrases.jsonl` as registered in `{reg['phrases']['commit'][:12]}`."
        f" Model revision `{p['revision'][:12]}`. Exact readout. Chance level from `{p['chance_level_from']}`.",
        "",
        "**What this measures.** Traces are removed from the sum that makes the value vector, and the rest is decoded."
        " This shows how a stored concept is spread across a phrase's 24 traces. It does not show what the running model"
        " forgets (that needs the behavioural lesion), and it says nothing about brains.",
        "",
    ]
    layers = summary["layers"]
    for i, (name, r) in enumerate(layers.items()):
        lines += _layer(f"Layer {name[1:]}", r, primary=i == 0)

    l1 = layers["L1"]
    lines += ["## Predictions (written before the run, protocol Section 6b), layer 1", ""]
    if "predictions" in l1:
        pr = l1["predictions"]
        lines.append(
            f"- **P1, no head lesion is special:** {'held' if pr['P1_no_head_lesion_special'] else 'did not hold'}."
        )
        lines.append(
            f"- **P2, some n-gram-size lesion is special:** {'held' if pr['P2_some_size_lesion_special'] else 'did not hold'}."
        )
        two = pr["P2_two_token_bin"]
        if two is None:
            lines.append("- **P2, 2-token bin:** not testable (too few intact hits).")
        else:
            ok = two["B3(2)_more_damaging"] and two["B3(3)_not_more_damaging"] and two["B3(4)_not_more_damaging"]
            lines.append(
                f"- **P2, 2-token bin** (removing the 2-gram traces more damaging than chance, removing 3- or 4-gram traces not):"
                f" {'held' if ok else 'did not hold'} (2-gram: {two['B3(2)_more_damaging']}, 3-gram not: {two['B3(3)_not_more_damaging']},"
                f" 4-gram not: {two['B3(4)_not_more_damaging']})."
            )
        q2 = l1["Q2"]
        parallel = (
            l1["labels_allowed"]
            and q2["heads_label"] == "equipotential across heads"
            and q2["sizes_label"].startswith("not")
        )
        lines += [
            "",
            "**Lashley's conclusion (2) parallel** (limited regions may be essential, but within such a region the parts are"
            f" functionally equivalent): {'produced by the rules' if parallel else 'not produced by the rules'}.",
        ]
    lines.append("")
    return "\n".join(lines)
