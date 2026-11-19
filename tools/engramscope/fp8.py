"""FP8 decoding for the model files: E4M3 values and E8M0 power-of-two scales, one scale per 32 values."""

from __future__ import annotations

import numpy as np


def _e4m3(code: int) -> float:
    sign = -1.0 if code & 0x80 else 1.0
    exponent, mantissa = (code >> 3) & 0xF, code & 7
    if exponent == 15 and mantissa == 7:
        return float("nan")
    if exponent == 0:
        return sign * (mantissa / 8) * 2.0**-6
    return sign * (1 + mantissa / 8) * 2.0 ** (exponent - 7)


E4M3 = np.array([_e4m3(code) for code in range(256)], dtype=np.float32)
E8M0 = np.array([2.0 ** (code - 127) if code < 255 else np.nan for code in range(256)], dtype=np.float32)


def decode_rows(values: np.ndarray, scales: np.ndarray, block: int = 32) -> np.ndarray:
    """Rows with one scale per `block` values. values: uint8 [n, d]; scales: uint8 [n, d // block]."""
    n, d = values.shape
    if scales.shape != (n, d // block):
        raise ValueError(f"scales shape {scales.shape}, expected {(n, d // block)}")
    decoded = E4M3[values].reshape(n, d // block, block) * E8M0[scales].reshape(n, d // block, 1)
    return decoded.reshape(n, d)


def decode_block_scaled(values: np.ndarray, scales: np.ndarray, block: int = 32) -> np.ndarray:
    """A matrix with one scale per `block` x `block` tile. values: uint8 [r, c]; scales: uint8 [r // block, c // block]."""
    r, c = values.shape
    if scales.shape != (r // block, c // block):
        raise ValueError(f"scales shape {scales.shape}, expected {(r // block, c // block)}")
    tiles = E4M3[values].reshape(r // block, block, c // block, block) * E8M0[scales][:, None, :, None]
    return tiles.reshape(r, c)
