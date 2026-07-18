"""HTTP(S) download cache for Dataset / make_loader prefetch.

Local paths pass through. Remote URLs are fetched into
``{TORCHFITS_CACHE_DIR}/remote/`` (or ``TORCHFITS_REMOTE_CACHE``) so DataLoader
workers read from disk. Optional background prefetch overlaps the next GET
with the current train step.
"""

from __future__ import annotations

import hashlib
import threading
import urllib.request
from pathlib import Path
from typing import Iterable

from torchfits.cache import remote_cache_root

_REMOTE_PREFIXES = ("http://", "https://")
_prefetch_lock = threading.Lock()
_prefetch_threads: dict[str, threading.Thread] = {}


def is_http_url(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith(_REMOTE_PREFIXES)


def remote_cache_dir() -> Path:
    return remote_cache_root()


def cache_path_for_url(url: str, *, cache_dir: Path | None = None) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    suffix = Path(url.split("?", 1)[0]).suffix or ".fits"
    if len(suffix) > 16:
        suffix = ".fits"
    return (cache_dir or remote_cache_dir()) / f"{digest}{suffix}"


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=120) as response:
        with open(tmp, "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    tmp.replace(dest)
    return dest


def resolve_local_path(
    path: str,
    *,
    cache_dir: Path | None = None,
    download: bool = True,
) -> str:
    """Return a local filesystem path for *path* (download HTTP(S) if needed)."""
    if not is_http_url(path):
        return path
    dest = cache_path_for_url(path, cache_dir=cache_dir)
    if dest.is_file():
        return str(dest)
    if not download:
        return str(dest)
    return str(_download(path, dest))


def prefetch_urls(urls: Iterable[str], *, cache_dir: Path | None = None) -> None:
    """Start background downloads for missing HTTP(S) URLs (best-effort)."""
    for url in urls:
        if not is_http_url(url):
            continue
        dest = cache_path_for_url(url, cache_dir=cache_dir)
        if dest.is_file():
            continue
        with _prefetch_lock:
            existing = _prefetch_threads.get(url)
            if existing is not None and existing.is_alive():
                continue

            def _job(u: str = url, d: Path = dest) -> None:
                try:
                    _download(u, d)
                except Exception:
                    pass  # ponytail: best-effort prefetch; next resolve retries

            thread = threading.Thread(
                target=_job, name="torchfits-prefetch", daemon=True
            )
            _prefetch_threads[url] = thread
            thread.start()
