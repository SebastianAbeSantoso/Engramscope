"""Design properties 6-8 and 10: byte offsets, FP8 decoding, linearity, lesion additivity."""

import json
import struct

import ml_dtypes
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from engramscope import fp8, hub
from engramscope.readout import contributions, topk_ids


def test_e4m3_matches_ml_dtypes():
    ref = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    assert np.array_equal(np.isnan(ref), np.isnan(fp8.E4M3))
    ok = ~np.isnan(ref)
    assert np.array_equal(ref[ok], fp8.E4M3[ok])


def test_e8m0_is_powers_of_two():
    if hasattr(ml_dtypes, "float8_e8m0fnu"):
        ref = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e8m0fnu).astype(np.float32)
        assert np.array_equal(np.isnan(ref), np.isnan(fp8.E8M0))
        assert np.array_equal(ref[:255], fp8.E8M0[:255])
    assert np.isnan(fp8.E8M0[255]) and fp8.E8M0[127] == 1.0 and fp8.E8M0[128] == 2.0


@settings(max_examples=50, deadline=None)
@given(arrays(np.uint8, (3, 64)), arrays(np.uint8, (3, 2), elements=st.integers(100, 150)))
def test_decode_rows_scales_each_block(values, scales):
    out = fp8.decode_rows(values, scales)
    for r in range(3):
        for b in range(2):
            expected = fp8.E4M3[values[r, 32 * b : 32 * b + 32]] * fp8.E8M0[scales[r, b]]
            assert np.array_equal(out[r, 32 * b : 32 * b + 32], expected, equal_nan=True)


def test_decode_block_scaled_tiles():
    rng = np.random.default_rng(0)
    values = rng.integers(0, 120, (64, 96), dtype=np.uint8)
    scales = rng.integers(120, 135, (2, 3), dtype=np.uint8)
    out = fp8.decode_block_scaled(values, scales)
    for i in range(64):
        for j in range(96):
            assert out[i, j] == fp8.E4M3[values[i, j]] * fp8.E8M0[scales[i // 32, j // 32]]


def test_safetensors_header_and_row_ranges():
    meta = {"a": {"dtype": "F8_E4M3", "shape": [10, 4], "data_offsets": [0, 40]}}
    raw = json.dumps(meta).encode()
    data_start, parsed = hub.parse_header(struct.pack("<Q", len(raw)), raw)
    assert data_start == 8 + len(raw) and parsed == meta
    t = hub.Tensor("a", "s", "F8_E4M3", (10, 4), data_start, 40)
    assert hub.row_range(t, 0, 4) == (data_start, 4)
    assert hub.row_range(t, 9, 4) == (data_start + 36, 4)
    with pytest.raises(IndexError):
        hub.row_range(t, 10, 4)


@settings(max_examples=50, deadline=None)
@given(st.integers(0, 2**32 - 1), st.lists(st.booleans(), min_size=6, max_size=6))
def test_linearity_and_lesion_additivity(seed, keep):
    rng = np.random.default_rng(seed)
    wv = rng.standard_normal((16, 6 * 8)).astype(np.float64)
    rows = rng.standard_normal((6, 8))
    parts = contributions(wv, rows)
    np.testing.assert_allclose(parts.sum(axis=0), wv @ rows.reshape(-1), rtol=1e-10, atol=1e-10)
    kept = np.array(keep)
    zeroed = rows * kept[:, None]
    np.testing.assert_allclose(parts[kept].sum(axis=0), wv @ zeroed.reshape(-1), rtol=1e-10, atol=1e-10)


def test_topk_is_cosine_order():
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((50, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    vec = rng.standard_normal((3, 8)).astype(np.float32)
    got = topk_ids(emb, vec, k=5, batch=2)
    for i in range(3):
        assert list(got[i]) == list(np.argsort(-(emb @ vec[i]), kind="stable")[:5])
