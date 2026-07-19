"""HTTP(S) / vos download cache for Dataset / make_loader prefetch.

Local paths pass through. Remote URLs are fetched into
``{TORCHFITS_CACHE_DIR}/remote/`` (or ``TORCHFITS_REMOTE_CACHE``) so DataLoader
workers read from disk. Optional background prefetch overlaps the next GET
with the current train step.

vos / vault short forms materialize via the optional ``vos`` client.
"""

from __future__ import annotations

import hashlib
import importlib
import threading
from pathlib import Path
from typing import Iterable

from torchfits.http_util import HttpBlockedError, http_open, http_timeout
from torchfits.vos_uri import is_vos_path as is_vos_path
from torchfits.vos_uri import normalize_vos_uri as normalize_vos_uri

from torchfits.cache import remote_cache_root

_REMOTE_PREFIXES = ("http://", "https://")
_prefetch_lock = threading.Lock()
_prefetch_threads: dict[str, threading.Thread] = {}


def is_http_url(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith(_REMOTE_PREFIXES)


def is_remote_url(path: str) -> bool:
    return is_http_url(path) or is_vos_path(path)


def remote_cache_dir() -> Path:
    return remote_cache_root()


def cache_path_for_url(url: str, *, cache_dir: Path | None = None) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    suffix = Path(url.split("?", 1)[0]).suffix or ".fits"
    if len(suffix) > 16:
        suffix = ".fits"
    return (cache_dir or remote_cache_dir()) / f"{digest}{suffix}"


def _download_http(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    existing = tmp.stat().st_size if tmp.is_file() else 0
    headers: dict[str, str] = {}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
    try:
        with http_open(
            url, headers=headers or None, timeout=http_timeout()
        ) as response:
            status = getattr(response, "status", None) or response.getcode()
            # Server ignored Range → restart from scratch.
            append = status == 206 and existing > 0
            if not append:
                existing = 0
            mode = "ab" if append else "wb"
            content_length = response.headers.get("Content-Length")
            expected = int(content_length) if content_length else None
            wrote = 0
            with open(tmp, mode) as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    wrote += len(chunk)
            if expected is not None and wrote != expected:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                raise OSError(
                    f"{url}: short download ({wrote} bytes, expected {expected})"
                )
    except HttpBlockedError:
        raise
    tmp.replace(dest)
    return dest


def _download_vos(path: str, dest: Path) -> Path:
    try:
        vos = importlib.import_module("vos")
    except ImportError as exc:
        raise ImportError(
            "vos/vault paths require the optional 'vos' package (pip/pixi install vos)"
        ) from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()
    uri = normalize_vos_uri(path)
    client = vos.Client()
    client.copy(uri, str(tmp))
    if not tmp.is_file() or tmp.stat().st_size == 0:
        raise OSError(f"{path}: vos copy produced empty file")
    tmp.replace(dest)
    return dest


def _download(url: str, dest: Path) -> Path:
    if is_vos_path(url):
        return _download_vos(url, dest)
    return _download_http(url, dest)


def resolve_local_path(
    path: str,
    *,
    cache_dir: Path | None = None,
    download: bool = True,
) -> str:
    """Return a local filesystem path for *path* (download HTTP(S)/vos if needed)."""
    if not is_remote_url(path):
        return path
    # Cache key: normalized vos URI so short and long forms share one file.
    cache_key = normalize_vos_uri(path) if is_vos_path(path) else path
    dest = cache_path_for_url(cache_key, cache_dir=cache_dir)
    if dest.is_file():
        return str(dest)
    # Prefetch threads are keyed by cache_key so vos:/vault: aliases share one
    # in-flight download and do not race the same ".partial".
    with _prefetch_lock:
        existing = _prefetch_threads.get(cache_key)
    if existing is not None and existing.is_alive():
        existing.join()
        if dest.is_file():
            return str(dest)
    if not download:
        return str(dest)
    return str(_download(path, dest))


def prefetch_urls(urls: Iterable[str], *, cache_dir: Path | None = None) -> None:
    """Start background downloads for missing HTTP(S)/vos URLs (best-effort)."""
    for url in urls:
        if not is_remote_url(url):
            continue
        cache_key = normalize_vos_uri(url) if is_vos_path(url) else url
        dest = cache_path_for_url(cache_key, cache_dir=cache_dir)
        if dest.is_file():
            continue
        with _prefetch_lock:
            existing = _prefetch_threads.get(cache_key)
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
            _prefetch_threads[cache_key] = thread
            thread.start()
