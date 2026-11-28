"""Cache clearing, statistics, and disk-root helpers for torchfits.

This is the public, torch-free façade of torchfits' caches. It owns the disk
layer (``cache_root`` / ``remote_cache_root`` / ``sample_cache_root`` and
their removal) and delegates the in-process layer to the I/O engine
(``torchfits._io_engine.caches``: cached reads, header/meta/HDU-type LRUs,
counters, and the native ``SharedReadMeta`` clear). The engine never touches
disk caches; this module never holds in-process state. See
docs/architecture.md, "Caching".
"""

from __future__ import annotations

import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict

CACHE_ENV_SENTINELS: tuple[str, ...] = (
    "SLURM_JOB_ID",
    "PBS_JOBID",
    "LSB_JOBID",
    "SGE_JOB_ID",
    "AWS_EXECUTION_ENV",
    "AWS_LAMBDA_FUNCTION_NAME",
    "GOOGLE_CLOUD_PROJECT",
    "GCLOUD_PROJECT",
    "AZURE_FUNCTIONS_ENVIRONMENT",
    "WEBSITE_SITE_NAME",
    "KUBERNETES_SERVICE_HOST",
    "K_SERVICE",
)


def cache_root() -> Path:
    """Filesystem root for torchfits disk caches (remote downloads, samples)."""
    override = os.environ.get("TORCHFITS_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg:
        return Path(xdg).expanduser() / "torchfits"
    return Path.home() / ".cache" / "torchfits"


def remote_cache_root() -> Path:
    """Directory for HTTP(S) Dataset prefetch materialization."""
    override = os.environ.get("TORCHFITS_REMOTE_CACHE", "").strip()
    if override:
        return Path(override).expanduser()
    return cache_root() / "remote"


def sample_cache_root() -> Path:
    """Directory for example/gallery sample downloads."""
    override = os.environ.get("TORCHFITS_SAMPLE_CACHE", "").strip()
    if override:
        return Path(override).expanduser()
    return cache_root() / "samples"


def get_cache_stats() -> Dict[str, Any]:
    """Return in-process I/O cache statistics."""
    try:
        from ._io_engine.caches import get_cache_performance

        return get_cache_performance()
    except (ImportError, AttributeError):
        return {}


def clear_cache(*, disk: bool = False) -> None:
    """Clear in-process caches and optionally downloaded files."""
    try:
        # One engine call clears every in-process layer, the native
        # SharedReadMeta included (it warns itself when the extension is
        # unavailable); it used to be cleared a second time here first.
        from ._io_engine.caches import clear_file_cache

        clear_file_cache()
    except Exception as exc:
        # Best effort: a failed in-process clear must not stop the disk clear.
        warnings.warn(
            f"clear_cache: I/O engine clear failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    if disk:
        root = cache_root()
        _remove_tree(root)
        # TORCHFITS_REMOTE_CACHE / TORCHFITS_SAMPLE_CACHE are user-configured
        # and may be shared (even $HOME): never delete them wholesale. Only
        # the hashed download files torchfits names itself are removed from a
        # remote override outside the root; a sample override is left alone.
        remote = remote_cache_root()
        if not _is_within(remote, root):
            _remove_remote_downloads(remote)


def clear_all_caches() -> None:
    """Clear in-process caches and downloaded files."""
    clear_cache(disk=True)


def _remove_tree(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def _is_within(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


# Names data.remote creates: ``<sha256[:32]><suffix>`` plus its
# ``.partial``/``.partial.meta`` siblings and the ``.<name>.lock`` file.
_REMOTE_DOWNLOAD_NAME = re.compile(r"^\.?[0-9a-f]{32}\.[^/]+$")


def _remove_remote_downloads(directory: Path) -> None:
    """Unlink torchfits download files directly inside *directory*."""
    try:
        entries = list(directory.iterdir())
    except OSError:
        return
    for entry in entries:
        if not _REMOTE_DOWNLOAD_NAME.match(entry.name):
            continue
        try:
            if entry.is_file() and not entry.is_symlink():
                entry.unlink()
        except OSError:
            pass


__all__ = [
    "clear_all_caches",
    "clear_cache",
    "get_cache_stats",
]
