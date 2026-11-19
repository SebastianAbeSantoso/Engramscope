"""Access to the pinned model files on Hugging Face: small files, safetensors headers, byte ranges.

Every URL pins the revision. The hub redirect for each file is resolved once per process (the
resolver is rate-limited to 3,000 requests per 300 s); byte ranges then go to the CDN directly,
with back-off on HTTP 429 and 5xx, and a fresh redirect if a signed CDN URL has expired.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from . import REPO, REVISION

HUB = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"


def cache_dir() -> Path:
    """Local cache for model data. Never inside the repository."""
    root = Path(os.environ.get("ENGRAMSCOPE_CACHE", Path.home() / ".cache" / "engramscope"))
    path = root / REVISION
    path.mkdir(parents=True, exist_ok=True)
    return path


_local = threading.local()


def _http() -> requests.Session:
    session = getattr(_local, "session", None)
    if session is None:
        session = requests.Session()
        session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64))
        _local.session = session
    return session


def file(name: str) -> Path:
    """A small file of the model repo (config, tokenizer, index), downloaded once into the cache."""
    path = cache_dir() / name
    if not path.exists():
        response = _http().get(HUB + name, timeout=120)
        response.raise_for_status()
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(response.content)
        tmp.replace(path)
    return path


_cdn: dict[str, str] = {}
_cdn_lock = threading.Lock()


def cdn_url(shard: str, refresh: bool = False) -> str:
    """The CDN location of a model file, resolved once (one hub request per file)."""
    with _cdn_lock:
        if refresh or shard not in _cdn:
            response = _http().get(HUB + shard, headers={"Range": "bytes=0-0"}, allow_redirects=False, timeout=60)
            if response.status_code in (301, 302, 303, 307, 308):
                _cdn[shard] = response.headers["Location"]
            else:
                response.raise_for_status()
                _cdn[shard] = HUB + shard
        return _cdn[shard]


def read_range(shard: str, start: int, length: int) -> bytes:
    """Bytes [start, start + length) of a model file."""
    refresh = False
    for attempt in range(8):
        url = cdn_url(shard, refresh=refresh)
        refresh = False
        try:
            response = _http().get(url, headers={"Range": f"bytes={start}-{start + length - 1}"}, timeout=60)
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(1 + attempt)
            continue
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(float(response.headers.get("Retry-After", 2**attempt)))
            continue
        if response.status_code == 403:  # signed CDN URL expired
            refresh = True
            continue
        if response.status_code != 206:
            raise RuntimeError(f"HTTP {response.status_code} for {shard} bytes {start}+{length}")
        if len(response.content) != length:
            raise RuntimeError(f"got {len(response.content)} bytes for {shard} bytes {start}+{length}")
        return response.content
    raise RuntimeError(f"giving up on {shard} bytes {start}+{length}")


@dataclass(frozen=True)
class Tensor:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    start: int  # absolute byte offset of the tensor's data in the shard
    nbytes: int


def parse_header(first8: bytes, header_bytes: bytes) -> tuple[int, dict]:
    """Safetensors layout: 8-byte little-endian header length, JSON header, then the data."""
    (n,) = struct.unpack("<Q", first8)
    if len(header_bytes) != n:
        raise ValueError("header length mismatch")
    return 8 + n, json.loads(header_bytes)


def header(shard: str) -> tuple[int, dict]:
    """(data start, header JSON) of a shard, cached."""
    path = cache_dir() / f"{shard}.header.json"
    if not path.exists():
        first8 = read_range(shard, 0, 8)
        (n,) = struct.unpack("<Q", first8)
        data_start, parsed = parse_header(first8, read_range(shard, 8, n))
        path.write_text(json.dumps({"data_start": data_start, "header": parsed}))
    cached = json.loads(path.read_text())
    return cached["data_start"], cached["header"]


def tensor(name: str) -> Tensor:
    index = json.loads(file("model.safetensors.index.json").read_text())
    shard = index["weight_map"][name]
    data_start, parsed = header(shard)
    entry = parsed[name]
    a, b = entry["data_offsets"]
    return Tensor(name, shard, entry["dtype"], tuple(entry["shape"]), data_start + a, b - a)


def row_range(t: Tensor, row: int, row_bytes: int) -> tuple[int, int]:
    """Absolute (start, length) of one row of a row-major tensor."""
    start = t.start + row * row_bytes
    if row < 0 or start + row_bytes > t.start + t.nbytes:
        raise IndexError(f"row {row} outside {t.name}")
    return start, row_bytes


def read_tensor(t: Tensor, offset: int = 0, length: int | None = None, chunk: int = 32 << 20) -> bytes:
    """Bytes [offset, offset + length) of a tensor's data, fetched in parallel chunks."""
    length = t.nbytes - offset if length is None else length
    if offset < 0 or offset + length > t.nbytes:
        raise IndexError(f"range outside {t.name}")
    out = bytearray(length)
    parts = [(o, min(chunk, length - o)) for o in range(0, length, chunk)]

    def fetch(part: tuple[int, int]) -> None:
        o, n = part
        out[o : o + n] = read_range(t.shard, t.start + offset + o, n)

    with cf.ThreadPoolExecutor(16) as pool:
        list(pool.map(fetch, parts))
    return bytes(out)
