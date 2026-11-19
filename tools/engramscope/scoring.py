"""Hit matching and bootstrap statistics shared by the benchmark and the lesion experiments."""

from __future__ import annotations

import numpy as np


def accept_set(related: list[str], phrase_ids: list[int], strings: list[str]) -> set[str]:
    """The related words minus the phrase's own decoded tokens (bench/README.md, rule 3)."""
    own = {strings[t] for t in phrase_ids}
    return {w for w in related if w not in own}


def hit(top: np.ndarray, strings: list[str], accept: set[str], k: int = 10) -> bool:
    return any(strings[int(t)] in accept for t in top[:k])


def boot_indices(n: int, resamples: int, seed: int) -> np.ndarray:
    """[resamples, n] indices for a percentile bootstrap over phrases; the same seed gives paired resamples."""
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def mean_ci(values: np.ndarray, idx: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    means = values[idx].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return float(values.mean()), float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def curve_shape(k: np.ndarray, r: np.ndarray) -> float:
    """Shape statistic S of a retention curve (lesion protocol, Q1).

    k: lesion sizes starting at 0; r: mean retention at each, r[0] = 1. S is the mean over [0, k_max]
    of (r - line) / drop, where the line joins the end points; linear interpolation, trapezoid rule.
    """
    k = np.asarray(k, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    drop = r[0] - r[-1]
    if drop <= 0:
        return float("nan")
    line = r[0] - drop * (k - k[0]) / (k[-1] - k[0])
    dev = (r - line) / drop
    area = np.sum((dev[1:] + dev[:-1]) / 2 * np.diff(k))
    return float(area / (k[-1] - k[0]))


def crossing(k: np.ndarray, r: np.ndarray, level: float = 0.5) -> float:
    """First lesion size at which the curve falls to `level`, by linear interpolation; nan if it never does."""
    k = np.asarray(k, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    for i in range(1, len(k)):
        if r[i] <= level < r[i - 1] or r[i] == level:
            if r[i - 1] == r[i]:
                return float(k[i])
            return float(k[i - 1] + (r[i - 1] - level) / (r[i - 1] - r[i]) * (k[i] - k[i - 1]))
    return float("nan")
