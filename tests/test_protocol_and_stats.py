"""The code follows the pre-registered documents, and the rule statistics behave as the protocol defines them."""

import re
from pathlib import Path

import numpy as np

from engramscope import bench, lesion
from engramscope.scoring import crossing, curve_shape

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (ROOT / "bench" / "lesion-protocol.md").read_text(encoding="utf-8")
PLAN = (ROOT / "bench" / "README.md").read_text(encoding="utf-8")


def test_lesion_conditions_match_protocol():
    sizes = re.search(r"k ∈ \{([0-9, ]+)\}", PROTOCOL).group(1)
    assert tuple(int(x) for x in sizes.split(",")) == lesion.K_SIZES
    assert "50 random subsets of size k" in PROTOCOL and lesion.N_SUBSETS == 50
    assert f"default_rng({lesion.SEED})" in PROTOCOL
    prefixes = re.findall(r'^\s+\d\. `"([^"]+)"`', PROTOCOL, re.M)
    assert tuple(prefixes) == lesion.PREFIXES
    assert "at least 30 phrases" in PROTOCOL and lesion.POWER_GUARD == 30
    assert "[−0.1, 0.1]" in PROTOCOL and lesion.BAND == 0.1
    assert "10,000 resamples" in PROTOCOL and lesion.RESAMPLES_CORRECTED == 10_000


def test_condition_list_is_complete():
    names = lesion.condition_names()
    assert len(names) == 1 + len(lesion.K_SIZES) * 50 + 24 + 24 + 3 + 8 + 3
    subsets = {k: np.stack([np.arange(k)] * 50) for k in lesion.K_SIZES}
    masks = lesion.keep_masks(names, subsets)
    removed = 24 - masks.sum(axis=1)
    assert removed[0] == 0
    assert all(removed[1 + i * 50] == k for i, k in enumerate(lesion.K_SIZES))
    tail = dict(zip(names[-14:], removed[-14:], strict=True))
    assert tail[("B3", 2)] == 8 and tail[("B4", 0)] == 3 and tail[("C1", 4)] == 16


def test_subsets_are_drawn_in_registered_order():
    a = lesion.draw_subsets([1, 2], (1, 14))
    b = lesion.draw_subsets([1, 2], (1, 14))
    assert all(np.array_equal(a[key], b[key]) for key in a)
    assert all(len(set(row)) == key[2] for key, v in a.items() for row in v)
    assert list(a)[:3] == [(1, 1, 1), (1, 1, 2), (1, 1, 3)]


def test_bench_matches_analysis_plan():
    assert f"default_rng({bench.SEED})" in PLAN
    assert "+ 61) mod 123" in PLAN and bench.SHIFT == 61
    assert "ec0e429" in PLAN and bench.REGISTERED == "ec0e429"
    assert "2,000 resamples" in PLAN and bench.RESAMPLES == 2000


def test_curve_shape_reference_points():
    k = np.array([0, 1, 2, 3, 4, 8, 12, 16, 20, 22, 23], float)
    line = 1 - 0.8 * k / 23
    assert abs(curve_shape(k, line)) < 1e-12
    late = np.where(k < 23, 1.0, 0.2)
    early = np.where(k > 0, 0.2, 1.0)
    assert 0.45 < curve_shape(k, late) <= 0.5
    assert -0.5 <= curve_shape(k, early) < -0.45


def test_crossing_interpolates():
    k = np.array([0, 10, 20], float)
    assert crossing(k, np.array([1.0, 0.6, 0.4])) == 15.0
    assert np.isnan(crossing(k, np.array([1.0, 0.9, 0.8])))
