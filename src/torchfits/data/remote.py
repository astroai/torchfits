"""HTTP(S) / vos download cache for Dataset / make_loader prefetch.

Local paths pass through. Remote URLs are fetched into
``{TORCHFITS_CACHE_DIR}/remote/`` (or ``TORCHFITS_REMOTE_CACHE``) so DataLoader
workers read from disk. Optional background prefetch overlaps the next GET
with the current train step.

vos / vault short forms materialize via the optional ``vos`` client.
"""

from __future__ import annotations

import bz2
import gzip
import hashlib
import importlib
import json
import logging
import threading
import os
import tempfile
import urllib.error
import warnings
import zipfile
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

try:
    import fcntl as _fcntl
except ImportError:  # e.g. Windows: isolation comes from unique temp names
    _fcntl = None  # type: ignore[assignment]

from torchfits.http_util import (
    HttpBlockedError,
    _parse_http_content_range,
    http_open,
    http_timeout,
)
from torchfits.vos_uri import is_vos_path as is_vos_path
from torchfits.vos_uri import normalize_vos_uri as normalize_vos_uri

from torchfits.cache import remote_cache_root
from torchfits._fits_structure import FitsStructureError as _FitsStructureError
from torchfits._fits_structure import walk_fits_hdus as _walk_fits_hdus

_REMOTE_PREFIXES = ("http://", "https://")
# Prefetch/download bookkeeping is keyed by (cache key, destination path):
# the same URL staged into two cache directories is two independent
# transfers, and one's failure must not be reported for the other.
_StateKey = tuple[str, str]
_prefetch_lock = threading.Lock()
_prefetch_threads: dict[_StateKey, threading.Thread] = {}
_prefetch_errors: dict[_StateKey, BaseException] = {}
_prefetch_scopes: dict[_StateKey, "PrefetchScope"] = {}
_download_locks: dict[_StateKey, threading.Lock] = {}
_log = logging.getLogger(__name__)


class PrefetchCancelled(OSError):
    """A prefetch whose :class:`PrefetchScope` was retired before it finished."""


class PrefetchScope:
    """Cancellation token for the prefetches one consumer (an iterator) starts.

    :func:`retire_prefetches` cancels every transfer started under the scope:
    each checks the token before creating directories, between body chunks
    and before promoting, so a retired transfer never recreates a staging
    directory its owner removed, and its error is dropped instead of being
    stored for the next consumer of the same URL.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        # Held while a transfer creates a directory, so retirement cannot
        # interleave between the cancellation check and the ``mkdir``.
        self._lock = threading.Lock()
        self._keys: set[_StateKey] = set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise PrefetchCancelled("prefetch retired by its owner")

    def make_dir(self, directory: Path) -> None:
        with self._lock:
            self.check()
            directory.mkdir(parents=True, exist_ok=True)


def _make_dir(directory: Path, scope: PrefetchScope | None) -> None:
    if scope is None:
        directory.mkdir(parents=True, exist_ok=True)
    else:
        scope.make_dir(directory)


def _state_key(cache_key: str, dest: Path) -> _StateKey:
    return (cache_key, str(dest))


def is_http_url(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith(_REMOTE_PREFIXES)


def is_remote_url(path: str) -> bool:
    return is_http_url(path) or is_vos_path(path)


def remote_cache_dir() -> Path:
    return remote_cache_root()


def ephemeral_scratch_dir() -> Path:
    """Return local fast scratch directory for staging (e.g. SLURM_TMPDIR or TMPDIR)."""
    for env_var in ("SLURM_TMPDIR", "TMPDIR"):
        val = os.environ.get(env_var, "").strip()
        if val:
            path = Path(val) / "torchfits_staging"
            path.mkdir(parents=True, exist_ok=True)
            return path
    fallback = Path(tempfile.gettempdir()) / "torchfits_staging"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def cleanup_downloaded_file(path: str | Path) -> bool:
    """Safely unlink a staged/downloaded temporary file."""
    try:
        p = Path(path)
        if p.is_file():
            p.unlink(missing_ok=True)
            return True
    except OSError as exc:
        _log.debug("cleanup_downloaded_file failed for %s: %s", path, exc)
    return False


def cache_path_for_url(url: str, *, cache_dir: Path | None = None) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    suffix = Path(url.split("?", 1)[0]).suffix or ".fits"
    if len(suffix) > 16:
        suffix = ".fits"
    return (cache_dir or remote_cache_dir()) / f"{digest}{suffix}"


@contextmanager
def _cross_process_download_lock(
    dest: Path, scope: PrefetchScope | None = None
) -> Iterator[None]:
    """Serialize download-on-demand for *dest* across processes via ``flock``.

    Only one writer at a time may touch a destination's ``.partial`` file;
    every other process either finds the finished file after acquiring the
    lock or joins the in-flight transfer. On platforms without ``fcntl``
    (e.g. Windows) this is a no-op and writer isolation comes from unique
    per-attempt temp names instead.
    """
    if _fcntl is None:
        yield
        return
    _make_dir(dest.parent, scope)
    lock_path = dest.parent / f".{dest.name}.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        _fcntl.flock(fd, _fcntl.LOCK_EX)
        yield
    finally:
        try:
            _fcntl.flock(fd, _fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def _partial_path_for(dest: Path) -> Path:
    """In-flight download file for *dest*.

    Shared canonical name under the cross-process lock so later processes
    can resume an interrupted transfer; unique per-attempt names where no
    lock primitive exists.
    """
    if _fcntl is not None:
        return dest.with_suffix(dest.suffix + ".partial")
    token = f"{os.getpid()}.{threading.get_ident():x}"
    return dest.with_suffix(dest.suffix + f".{token}.partial")


def _read_resume_validators(meta_path: Path) -> dict[str, str]:
    try:
        data = json.loads(meta_path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


# Leading bytes of a body CFITSIO can open: a FITS primary header, or a
# whole-file compression wrapper it decompresses (gzip, bzip2, Unix compress,
# pack, pkzip).
_FITS_PAYLOAD_MAGIC = (
    b"SIMPLE  =",
    b"\x1f\x8b",
    b"BZh",
    b"\x1f\x9d",
    b"\x1f\x1e",
    b"PK\x03\x04",
)


def _looks_like_fits_payload(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            head = handle.read(9)
    except OSError:
        return False
    return any(head.startswith(magic) for magic in _FITS_PAYLOAD_MAGIC)


def _verify_fits_payload(path: Path, *, framed: bool) -> None:
    """Raise :class:`_FitsStructureError` unless *path* is a complete payload.

    A plain FITS body is always walked (a header-only scan with seeks). A
    compressed body is decompressed and walked only when the transfer had
    no length framing (*framed* false): a gzip/bzip2/pkzip stream that ends
    early fails its end-of-stream / CRC check; Unix ``compress`` and
    ``pack`` carry no such check and are refused as unverifiable.
    """
    with open(path, "rb") as handle:
        head = handle.read(9)
        if head.startswith(b"SIMPLE  ="):
            handle.seek(0)
            _walk_fits_hdus(handle, os.fstat(handle.fileno()).st_size)
            return
    if framed:
        return
    try:
        if head.startswith(b"\x1f\x8b") or head.startswith(b"BZh"):
            opener = gzip.open if head.startswith(b"\x1f\x8b") else bz2.open
            with opener(path, "rb") as stream:
                _walk_fits_hdus(stream, None)
                while stream.read(1 << 20):
                    pass
            return
        if head.startswith(b"PK\x03\x04"):
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                if not members:
                    raise _FitsStructureError("empty zip archive")
                with archive.open(members[0]) as stream:
                    _walk_fits_hdus(stream, None)
                    while stream.read(1 << 20):
                        pass
            return
    except (EOFError, OSError, zlib.error, zipfile.BadZipFile) as exc:
        raise _FitsStructureError(
            f"compressed stream is incomplete or corrupt ({exc})"
        ) from exc
    raise _FitsStructureError(
        "compression format carries no integrity check, so a body without "
        "Content-Length cannot be verified"
    )


# Cached files already checked by _cached_copy_is_complete, keyed by path
# with the (size, mtime_ns) they had then, so the header walk runs once.
_verified_cache_files: dict[str, tuple[int, int]] = {}


def _cached_copy_is_complete(dest: Path) -> bool:
    """Cheap re-check of an existing cache entry (plain FITS header walk).

    Compressed entries are only checked for their magic: decompressing on
    every resolve would cost a full read of the file.
    """
    try:
        stat = dest.stat()
    except OSError:
        return False
    stamp = (stat.st_size, stat.st_mtime_ns)
    key = str(dest)
    with _prefetch_lock:
        if _verified_cache_files.get(key) == stamp:
            return True
    if not _looks_like_fits_payload(dest):
        return False
    try:
        _verify_fits_payload(dest, framed=True)
    except _FitsStructureError as exc:
        _log.warning("cached copy %s is incomplete: %s", dest, exc)
        return False
    except OSError:
        return False
    with _prefetch_lock:
        _verified_cache_files[key] = stamp
    return True


def _download_http(url: str, dest: Path, *, scope: PrefetchScope | None = None) -> Path:
    _make_dir(dest.parent, scope)
    tmp = _partial_path_for(dest)
    meta_path = Path(str(tmp) + ".meta")
    validators = _read_resume_validators(meta_path) if meta_path.is_file() else {}

    existing = tmp.stat().st_size if tmp.is_file() else 0
    headers: dict[str, str] = {}
    if existing > 0:
        if not (validators.get("etag") or validators.get("last_modified")):
            # A leftover partial without a validator cannot be checked
            # against the current remote content: restart cleanly rather
            # than risk gluing old bytes onto new ones.
            tmp.unlink(missing_ok=True)
            existing = 0
        else:
            headers["Range"] = f"bytes={existing}-"
            headers["If-Range"] = validators.get("etag") or validators["last_modified"]
    try:
        with http_open(
            url, headers=headers or None, timeout=http_timeout()
        ) as response:
            status = getattr(response, "status", None) or response.getcode()
            content_range = _parse_http_content_range(
                response.headers.get("Content-Range")
            )
            # Server ignored Range → restart from scratch.
            append = (
                status == 206
                and existing > 0
                and content_range is not None
                and content_range[0] == existing
            )
            if status == 206 and content_range is None:
                raise OSError(f"{url}: invalid or missing Content-Range")
            if status == 206 and existing > 0 and not append:
                raise OSError(
                    f"{url}: resumed response starts at "
                    f"{None if content_range is None else content_range[0]}, "
                    f"expected {existing}"
                )
            if not append:
                existing = 0
            mode = "ab" if append else "wb"
            content_length = response.headers.get("Content-Length")
            try:
                expected = int(content_length) if content_length else None
            except ValueError:
                # Malformed/duplicated length: same trust level as an absent
                # one (connection-close framing, warned below).
                expected = None
            wrote = 0
            if not append:
                # Capture validators so a later interrupted attempt can resume
                # safely (If-Range makes the server reject stale partials).
                etag = response.headers.get("ETag")
                last_modified = response.headers.get("Last-Modified")
                if etag or last_modified:
                    meta_path.write_text(
                        json.dumps({"etag": etag, "last_modified": last_modified})
                    )
                else:
                    meta_path.unlink(missing_ok=True)
            with open(tmp, mode) as handle:
                while True:
                    if scope is not None:
                        scope.check()
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    wrote += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            transfer_encoding = (
                response.headers.get("Transfer-Encoding") or ""
            ).lower()
            # Connection-close framing (no usable Content-Length, not
            # chunked): a clean EOF is indistinguishable from a connection
            # dropped mid-body, so completeness is checked from the payload
            # itself before promotion below.
            framed = expected is not None or "chunked" in transfer_encoding
            if expected is not None and wrote != expected:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                raise OSError(
                    f"{url}: short download ({wrote} bytes, expected {expected})"
                )
            if not append and wrote == 0:
                # A zero-byte body cannot be distinguished from a connection
                # dropped before the first chunk; never promote it to the
                # permanent cache (a cached empty file would be trusted
                # forever). Mirrors the _download_vos empty-copy check.
                tmp.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                raise OSError(f"{url}: empty download")
            if status == 206 and content_range is not None:
                range_start, range_end, total = content_range
                range_size = range_end - range_start + 1
                final_size = existing + wrote
                if (
                    wrote != range_size
                    or total is None
                    or range_end + 1 != total
                    or final_size != total
                ):
                    raise OSError(
                        f"{url}: incomplete Range download "
                        f"({final_size} bytes, total={total})"
                    )
        if not _looks_like_fits_payload(tmp):
            # A 200 error page (CDN outage, login wall, quota notice) is not a
            # FITS file; caching it would be trusted forever.
            tmp.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            raise OSError(
                f"{url}: response is not a FITS file (no 'SIMPLE  =' header "
                "or gzip/bzip2 signature); not cached"
            )
        try:
            _verify_fits_payload(tmp, framed=framed)
        except _FitsStructureError as exc:
            # A truncated body promoted here would be trusted forever and
            # fail every later read (CFITSIO status 108) without the server
            # ever being asked again.
            tmp.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            raise OSError(
                f"{url}: incomplete FITS download ({exc}); not cached"
                + ("" if framed else " (server sent no usable Content-Length)")
            ) from exc
        if scope is not None:
            scope.check()
        tmp.replace(dest)
        try:
            meta_path.unlink(missing_ok=True)
        except OSError:
            pass
        return dest
    except HttpBlockedError:
        raise
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and "Range" in headers:
            # Resume offset at/behind EOF (interrupted after the last byte,
            # or the remote changed size): the partial is dead. Drop it and
            # fetch cleanly instead of retrying the same doomed Range.
            tmp.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return _download_http(url, dest, scope=scope)
        raise
    finally:
        if _fcntl is None:
            # Token-named temporaries are unique per attempt and cannot be
            # resumed: never leave them behind on failure (on success they
            # were already promoted/removed, so this is a no-op).
            tmp.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)


def _download_vos(path: str, dest: Path, *, scope: PrefetchScope | None = None) -> Path:
    try:
        vos = importlib.import_module("vos")
    except ImportError as exc:
        raise ImportError(
            "vos/vault paths require the optional 'vos' package (pip/pixi install vos)"
        ) from exc
    _make_dir(dest.parent, scope)
    tmp = _partial_path_for(dest)
    try:
        if tmp.exists():
            tmp.unlink()
        uri = normalize_vos_uri(path)
        client = vos.Client()
        client.copy(uri, str(tmp))
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise OSError(f"{path}: vos copy produced empty file")
        if scope is not None:
            scope.check()
        tmp.replace(dest)
        return dest
    finally:
        if _fcntl is None:
            # Token-named temporaries are unique per attempt and cannot be
            # resumed: never leave them behind on failure.
            tmp.unlink(missing_ok=True)


def _download(url: str, dest: Path, *, scope: PrefetchScope | None = None) -> Path:
    if is_vos_path(url):
        return _download_vos(url, dest, scope=scope)
    return _download_http(url, dest, scope=scope)


def _download_once(
    cache_key: str, url: str, dest: Path, *, scope: PrefetchScope | None = None
) -> Path:
    # Thread lock covers Dataset/prefetch threads; the flock serializes
    # DataLoader worker *processes* sharing one cache directory.
    with _prefetch_lock:
        lock = _download_locks.setdefault(_state_key(cache_key, dest), threading.Lock())
    if scope is not None:
        scope.check()
    with lock, _cross_process_download_lock(dest, scope):
        if scope is not None:
            scope.check()
        if dest.is_file():
            return dest
        return _download(url, dest, scope=scope)


def _cleanup_state(key: _StateKey) -> None:
    """Remove per-transfer bookkeeping after a completed transfer.

    Does NOT pop from ``_download_locks`` — those Lock objects are tiny and
    removing them creates a race window where two threads could enter
    ``_download_once`` with different locks and download the same file in
    parallel, corrupting ``.partial``.
    """
    with _prefetch_lock:
        _prefetch_threads.pop(key, None)
        _prefetch_errors.pop(key, None)
        _prefetch_scopes.pop(key, None)


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
    key = _state_key(cache_key, dest)
    if dest.is_file():
        if is_http_url(path) and not _cached_copy_is_complete(dest):
            # Poisoned by an earlier release that cached any 200 body, or a
            # truncated connection-close body: drop it and fetch again
            # instead of failing on it forever.
            _log.warning(
                "discarding non-FITS or incomplete cached copy of %s at %s",
                path,
                dest,
            )
            dest.unlink(missing_ok=True)
        else:
            _cleanup_state(key)
            return str(dest)
    # Prefetch threads are keyed by (cache_key, dest) so vos:/vault: aliases
    # share one in-flight download and do not race the same ".partial".
    with _prefetch_lock:
        existing = _prefetch_threads.get(key)
        prefetch_error = _prefetch_errors.pop(key, None)
    if existing is not None and existing.is_alive():
        existing.join()
        with _prefetch_lock:
            prefetch_error = _prefetch_errors.pop(key, None)
        if dest.is_file():
            _cleanup_state(key)
            return str(dest)
    if prefetch_error is not None:
        raise prefetch_error
    if not download:
        return str(dest)
    result = str(_download_once(cache_key, path, dest))
    _cleanup_state(key)
    return result


def prefetch_urls(
    urls: Iterable[str],
    *,
    cache_dir: Path | None = None,
    scope: PrefetchScope | None = None,
) -> None:
    """Start background downloads for missing HTTP(S)/vos URLs (best-effort).

    With *scope*, the transfers can later be cancelled together by
    :func:`retire_prefetches` (their results and errors are then dropped).
    """
    for url in urls:
        if not is_remote_url(url):
            continue
        if scope is not None and scope.cancelled:
            return
        cache_key = normalize_vos_uri(url) if is_vos_path(url) else url
        dest = cache_path_for_url(cache_key, cache_dir=cache_dir)
        if dest.is_file():
            continue
        key = _state_key(cache_key, dest)
        with _prefetch_lock:
            existing = _prefetch_threads.get(key)
            if existing is not None and existing.is_alive():
                continue

            def _job(
                u: str = url,
                d: Path = dest,
                ck: str = cache_key,
                k: _StateKey = key,
            ) -> None:
                me = threading.current_thread()
                try:
                    _download_once(ck, u, d, scope=scope)
                    with _prefetch_lock:
                        if _prefetch_threads.get(k) is me:
                            _prefetch_errors.pop(k, None)
                except Exception as exc:
                    if scope is not None and scope.cancelled:
                        # Retired by its owner: nobody will consume this
                        # transfer, and its error (often the removed
                        # staging directory) must not poison the next one.
                        _log.debug("retired prefetch for %s ended: %s", u, exc)
                        return
                    # NOTE: best-effort prefetch; resolve_local_path re-raises
                    # the stored error instead of retrying opaquely.
                    with _prefetch_lock:
                        if _prefetch_threads.get(k) is me:
                            _prefetch_errors[k] = exc
                    _log.error("prefetch failed for %s: %s", u, exc)
                    warnings.warn(
                        f"prefetch failed for {u}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            thread = threading.Thread(
                target=_job, name="torchfits-prefetch", daemon=True
            )
            _prefetch_threads[key] = thread
            _prefetch_errors.pop(key, None)
            if scope is not None:
                _prefetch_scopes[key] = scope
                scope._keys.add(key)
            else:
                _prefetch_scopes.pop(key, None)
            thread.start()


def retire_prefetches(scope: PrefetchScope) -> list[threading.Thread]:
    """Cancel every prefetch started under *scope* and forget its state.

    Returns the threads still running; they stop at their next cancellation
    check (or when their current network read returns) without creating
    directories, promoting files or recording errors.
    """
    with scope._lock:
        scope._event.set()
    alive: list[threading.Thread] = []
    with _prefetch_lock:
        for key in scope._keys:
            if _prefetch_scopes.get(key) is not scope:
                continue
            thread = _prefetch_threads.pop(key, None)
            _prefetch_errors.pop(key, None)
            _prefetch_scopes.pop(key, None)
            if thread is not None and thread.is_alive():
                alive.append(thread)
        scope._keys.clear()
    return alive
