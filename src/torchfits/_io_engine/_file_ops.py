"""Torch-free file helpers shared by the write and HDU-edit paths.

Atomic publish of a rewritten file, cache invalidation after it, and the
``compress=`` flag normalisation. Kept apart from ``_write_helpers`` (which
imports torch) so metadata-only edits such as ``delete_hdu`` never load torch.
"""

from __future__ import annotations

import contextlib
import os
import stat
import tempfile
from collections.abc import Iterator
from typing import Optional, Union

from .caches import invalidate_path_caches as _invalidate_io_path_caches


def _invalidate_path_caches(path: str) -> None:
    """Invalidate Python-side caches and native metadata for a modified path."""
    _invalidate_io_path_caches(path)
    import torchfits._C as cpp

    # The native side used to expose a per-path invalidate_file_cache as well;
    # it was a no-op with a misleading name (shared handles were removed), so
    # the live native state is cleared globally here.
    cpp.clear_shared_read_meta_cache()


@contextlib.contextmanager
def _atomic_replace_target(path: str) -> Iterator[str]:
    """Yield a temp path that is ``os.replace``d over ``path`` on clean exit.

    Every whole-file writer here reaches CFITSIO's ``fits_create_file("!path")``,
    whose leading ``!`` unlinks the target before a single byte of the payload is
    validated -- even a *successful* rewrite changes the inode. So the payload
    goes to a hidden temp file in the same directory (same filesystem, so the
    rename is atomic) and the target is replaced only once the write is known
    good: a rejected, failed or crashed write leaves any existing file
    byte-identical, and no reader ever observes a partial file at ``path``.

    ``path`` must already exist -- otherwise there is nothing to protect, and a
    caller that may be creating a new file should yield ``path`` directly. The
    temp file is created in the realpath's directory and renamed onto the
    realpath, so a symlinked target keeps its symlink and only the file it
    points to is replaced; the target's mode is carried over.

    This is the single copy of the recipe:
    ``write_api.atomic_write_target``, ``_hdu_rewrite._write_hdus_uncompressed``
    and ``_hdu_splice.publish_spliced`` all route through it, so a fix to
    the window cannot reach one site and miss the others.

    Callers must invalidate the path caches *after* the block -- see
    ``_invalidate_written_target``. Inside the block the rename has not happened
    yet, and invalidating early lets a concurrent reader cache the old contents
    under the path and keep serving them after the rename.
    """
    target = os.path.realpath(path)
    original_mode = stat.S_IMODE(os.stat(target).st_mode)
    temp_path = _sibling_temp_path(target)
    publish = temp_path
    try:
        yield temp_path
        publish = _publishable_temp(temp_path, target)
        os.chmod(publish, original_mode)
        os.replace(publish, target)
    finally:
        for leftover in {temp_path, publish}:
            if os.path.exists(leftover):
                os.unlink(leftover)


def _sibling_temp_path(target: str) -> str:
    """A fresh, non-existent hidden path next to ``target`` (same filesystem,
    so publishing it is atomic). CFITSIO creates the file itself.

    The temp file is always plain FITS (no '.gz' anywhere in its name, which
    CFITSIO would act on) so header-card replay and checksums can reopen it
    for writing; ``_publishable_temp`` gzips it for a '.gz' target.
    """
    target_dir = os.path.dirname(target) or "."
    stem = os.path.basename(target).replace(".gz", "-gz")
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{stem}.", suffix=".tmp.fits", dir=target_dir
    )
    os.close(fd)
    os.unlink(temp_path)
    return temp_path


def _publishable_temp(temp_path: str, target: str) -> str:
    """The finished temp file to publish at ``target``: gzip-compressed when
    the target is named '.gz', so an overwritten or edited x.fits.gz stays
    gzip instead of becoming plain FITS under a '.gz' name."""
    if not target.lower().endswith(".gz"):
        return temp_path
    import gzip
    import shutil

    gz_path = temp_path + ".gz"
    with open(temp_path, "rb") as src, gzip.open(gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.unlink(temp_path)
    return gz_path


@contextlib.contextmanager
def _atomic_create_target(path: str, *, replace: bool) -> Iterator[str]:
    """Yield a temp path published at ``path`` on clean exit, for a new file.

    The payload is written to a sibling temp file, so a failed write never
    creates -- or deletes -- anything at ``path``. With ``replace=False`` the
    temp file is published with ``os.link``, which refuses an existing target:
    a file another writer created after the caller's existence check is never
    clobbered and raises ``FileExistsError`` instead. ``replace=True``
    (overwrite requested) publishes with ``os.replace``.
    """
    target = os.path.realpath(path)
    temp_path = _sibling_temp_path(target)
    publish = temp_path
    try:
        yield temp_path
        publish = _publishable_temp(temp_path, target)
        if replace:
            os.replace(publish, target)
        else:
            try:
                os.link(publish, target)
            except FileExistsError:
                raise FileExistsError(
                    f"File '{path}' already exists. Use overwrite=True to overwrite."
                ) from None
            except OSError:
                # No hard links on this filesystem: best effort, still never
                # replacing a file that is already there.
                if os.path.lexists(target):
                    raise FileExistsError(
                        f"File '{path}' already exists. "
                        "Use overwrite=True to overwrite."
                    ) from None
                os.rename(publish, target)
    finally:
        for leftover in {temp_path, publish}:
            if os.path.exists(leftover):
                os.unlink(leftover)


def _invalidate_written_target(path: str) -> None:
    """Invalidate the caches a just-renamed temp file has made stale.

    ``path`` and its realpath are the two spellings a reader can hold a cached
    handle under, and they differ whenever the target is a symlink, so both are
    invalidated. Must be called *after* ``_atomic_replace_target``'s block
    exits; see there for why.
    """
    target = os.path.realpath(path)
    _invalidate_path_caches(path)
    if target != path:
        _invalidate_path_caches(target)


def _resolve_compression_algorithm(compress: Union[bool, str]) -> Optional[str]:
    """Normalize compress flag to a backend algorithm string or None."""
    if compress is False:
        return None
    if compress is True:
        return "RICE_1"
    if isinstance(compress, str):
        algo = compress.strip()
        return algo if algo else "RICE_1"
    raise TypeError("compress must be bool or compression algorithm string")
