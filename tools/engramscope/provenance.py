"""Provenance for results: commits, file hashes, and the guards that enforce pre-registration."""

from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

from . import REPO, REVISION

ROOT = Path(__file__).resolve().parents[2]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def last_commit(path: str) -> str:
    return git("log", "-1", "--format=%H", "--", path)


def is_clean(path: str) -> bool:
    return git("status", "--porcelain", "--", path) == ""


def require_registered(path: str, commit_prefix: str) -> dict:
    """Refuse to run unless `path` is committed, unchanged, and last changed by the registered commit."""
    if not is_clean(path):
        raise SystemExit(f"{path} has uncommitted changes; refusing to run")
    last = last_commit(path)
    if not last.startswith(commit_prefix):
        raise SystemExit(f"{path} was last changed in {last[:12]}, not in the registered commit {commit_prefix}")
    return {"path": path, "commit": last, "blob": git("hash-object", path)}


def today() -> str:
    """The builder's local date (WITA), used in result file names."""
    return dt.datetime.now(ZoneInfo("Asia/Makassar")).date().isoformat()


def record(**registered: dict) -> dict:
    return {
        "model": REPO,
        "revision": REVISION,
        "code_commit": git("rev-parse", "HEAD"),
        "code_clean": is_clean("tools") and is_clean("pyproject.toml"),
        "run_at": dt.datetime.now(ZoneInfo("Asia/Makassar")).isoformat(timespec="seconds"),
        "registered": registered,
    }
