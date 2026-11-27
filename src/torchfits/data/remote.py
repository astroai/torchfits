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
from typing import IO, Iterable, Iterator

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

_REMOTE_PREFIXES = ("http://", "https://")
_prefetch_lock = threading.Lock()
_prefetch_threads: dict[str, threading.Thread] = {}
_prefetch_errors: dict[str, BaseException] = {}
_download_locks: dict[str, threading.Lock] = {}
_log = logging.getLogger(__name__)


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
def _cross_process_download_lock(dest: Path) -> Iterator[None]:
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
    dest.parent.mkdir(parents=True, exist_ok=True)
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


_FITS_BLOCK = 2880
_FITS_CARD = 80
_ByteStream = IO[bytes] | gzip.GzipFile | bz2.BZ2File


class _FitsStructureError(ValueError):
    """A body that does not hold every HDU its own headers declare."""


def _header_int(cards: dict[str, bytes], key: str, default: int | None = None) -> int:
    raw = cards.get(key)
    if raw is None:
        if default is None:
            raise _FitsStructureError(f"header lacks {key}")
        return default
    text = raw.split(b"/", 1)[0].strip()
    try:
        return int(text)
    except ValueError:
        raise _FitsStructureError(f"header {key} is not an integer: {text!r}") from None


def _hdu_data_size(cards: dict[str, bytes], primary: bool) -> int:
    """Data-unit byte count from BITPIX/NAXISn/PCOUNT/GCOUNT (FITS 4.0 §4.4.1)."""
    bitpix = _header_int(cards, "BITPIX")
    naxis = _header_int(cards, "NAXIS")
    if bitpix not in (8, 16, 32, 64, -32, -64) or not 0 <= naxis <= 999:
        raise _FitsStructureError(f"invalid BITPIX={bitpix} / NAXIS={naxis}")
    if naxis == 0:
        return 0
    axes = [_header_int(cards, f"NAXIS{i}") for i in range(1, naxis + 1)]
    if any(n < 0 for n in axes):
        raise _FitsStructureError(f"negative axis length in {axes}")
    groups = cards.get("GROUPS", b"").split(b"/", 1)[0].strip() == b"T"
    if primary and groups and axes[0] == 0:
        axes = axes[1:]  # random groups: NAXIS1 = 0 is not a real axis
    count = 1
    for n in axes:
        count *= n
    pcount = _header_int(cards, "PCOUNT", 0)
    gcount = _header_int(cards, "GCOUNT", 1)
    return abs(bitpix) // 8 * gcount * (pcount + count)


def _skip_bytes(handle: _ByteStream, n: int, size: int | None) -> int:
    """Advance *n* bytes; return how many were actually there."""
    if n <= 0:
        return 0
    if size is not None:
        pos = handle.tell()
        got = max(0, min(n, size - pos))
        handle.seek(pos + got)
        return got
    got = 0
    while got < n:
        chunk = handle.read(min(n - got, 1 << 20))
        if not chunk:
            break
        got += len(chunk)
    return got


def _walk_fits_hdus(handle: _ByteStream, size: int | None) -> None:
    """Check that *handle* covers every HDU its headers declare.

    Header blocks are read up to each ``END`` card and each data unit is
    sized from the header; a body that ends inside a header or a data unit
    raises :class:`_FitsStructureError`. A missing final 2880-byte pad is
    tolerated (CFITSIO reads such files), and bytes after the last HDU that
    do not start an ``XTENSION`` header are ignored as trailing junk.
    *size* is the byte length for seekable files, ``None`` for streams.
    """
    index = 0
    while True:
        block = handle.read(_FITS_BLOCK)
        if index == 0:
            if not block.startswith(b"SIMPLE  ="):
                raise _FitsStructureError("no primary header")
        elif not block or not b"XTENSION=".startswith(block[:9]):
            return
        cards: dict[str, bytes] = {}
        found_end = False
        while not found_end:
            for off in range(0, len(block) - _FITS_CARD + 1, _FITS_CARD):
                card = block[off : off + _FITS_CARD]
                key = card[:8].rstrip().decode("ascii", "replace")
                if key == "END":
                    found_end = True
                    break
                if card[8:10] == b"= " and key not in cards:
                    cards[key] = card[10:]
            if found_end:
                break
            if len(block) < _FITS_BLOCK:
                raise _FitsStructureError(f"HDU {index}: header ends before END card")
            block = handle.read(_FITS_BLOCK)
        if len(block) < _FITS_BLOCK:
            # CFITSIO refuses a header whose last 2880-byte record is short,
            # even when the END card itself arrived.
            raise _FitsStructureError(f"HDU {index}: header record truncated")
        data_size = _hdu_data_size(cards, index == 0)
        got = _skip_bytes(handle, data_size, size)
        if got < data_size:
            raise _FitsStructureError(
                f"HDU {index}: data truncated ({got} of {data_size} bytes)"
            )
        _skip_bytes(handle, -data_size % _FITS_BLOCK, size)
        index += 1


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


def _download_http(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
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
            return _download_http(url, dest)
        raise
    finally:
        if _fcntl is None:
            # Token-named temporaries are unique per attempt and cannot be
            # resumed: never leave them behind on failure (on success they
            # were already promoted/removed, so this is a no-op).
            tmp.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)


def _download_vos(path: str, dest: Path) -> Path:
    try:
        vos = importlib.import_module("vos")
    except ImportError as exc:
        raise ImportError(
            "vos/vault paths require the optional 'vos' package (pip/pixi install vos)"
        ) from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = _partial_path_for(dest)
    try:
        if tmp.exists():
            tmp.unlink()
        uri = normalize_vos_uri(path)
        client = vos.Client()
        client.copy(uri, str(tmp))
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise OSError(f"{path}: vos copy produced empty file")
        tmp.replace(dest)
        return dest
    finally:
        if _fcntl is None:
            # Token-named temporaries are unique per attempt and cannot be
            # resumed: never leave them behind on failure.
            tmp.unlink(missing_ok=True)


def _download(url: str, dest: Path) -> Path:
    if is_vos_path(url):
        return _download_vos(url, dest)
    return _download_http(url, dest)


def _download_once(cache_key: str, url: str, dest: Path) -> Path:
    # Thread lock covers Dataset/prefetch threads; the flock serializes
    # DataLoader worker *processes* sharing one cache directory.
    with _prefetch_lock:
        lock = _download_locks.setdefault(cache_key, threading.Lock())
    with lock, _cross_process_download_lock(dest):
        if dest.is_file():
            return dest
        return _download(url, dest)


def _cleanup_cache_key(cache_key: str) -> None:
    """Remove per-key download bookkeeping after a completed transfer.

    Does NOT pop from ``_download_locks`` — those Lock objects are tiny and
    removing them creates a race window where two threads could enter
    ``_download_once`` with different locks and download the same file in
    parallel, corrupting ``.partial``.
    """
    with _prefetch_lock:
        _prefetch_threads.pop(cache_key, None)
        _prefetch_errors.pop(cache_key, None)


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
            _cleanup_cache_key(cache_key)
            return str(dest)
    # Prefetch threads are keyed by cache_key so vos:/vault: aliases share one
    # in-flight download and do not race the same ".partial".
    with _prefetch_lock:
        existing = _prefetch_threads.get(cache_key)
        prefetch_error = _prefetch_errors.pop(cache_key, None)
    if existing is not None and existing.is_alive():
        existing.join()
        with _prefetch_lock:
            prefetch_error = _prefetch_errors.pop(cache_key, None)
        if dest.is_file():
            _cleanup_cache_key(cache_key)
            return str(dest)
    if prefetch_error is not None:
        raise prefetch_error
    if not download:
        return str(dest)
    result = str(_download_once(cache_key, path, dest))
    _cleanup_cache_key(cache_key)
    return result


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

            def _job(u: str = url, d: Path = dest, key: str = cache_key) -> None:
                try:
                    _download_once(key, u, d)
                    with _prefetch_lock:
                        _prefetch_errors.pop(key, None)
                except Exception as exc:
                    # NOTE: best-effort prefetch; resolve_local_path re-raises
                    # the stored error instead of retrying opaquely.
                    with _prefetch_lock:
                        _prefetch_errors[key] = exc
                    _log.error("prefetch failed for %s: %s", u, exc)
                    warnings.warn(
                        f"prefetch failed for {u}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            thread = threading.Thread(
                target=_job, name="torchfits-prefetch", daemon=True
            )
            _prefetch_threads[cache_key] = thread
            thread.start()
