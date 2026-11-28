"""Tests for clear_all_caches() convenience function."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchfits


@pytest.fixture(autouse=True)
def _private_cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test here clears disk caches, so none may reach the real root.

    clear_all_caches() deletes cache_root(), which is ~/.cache/torchfits by
    default -- including samples/, where scripts/fetch_example_samples.sh puts
    the example data. Tests that did not set TORCHFITS_CACHE_DIR themselves
    deleted the developer's (and a parallel run's) downloaded samples on
    every `pixi run test`. A test that sets its own override still wins: its
    setenv runs after this fixture.
    """
    monkeypatch.setenv("TORCHFITS_CACHE_DIR", str(tmp_path / "default_cache_root"))
    monkeypatch.delenv("TORCHFITS_REMOTE_CACHE", raising=False)
    monkeypatch.delenv("TORCHFITS_SAMPLE_CACHE", raising=False)


def test_module_never_clears_the_real_cache_root(tmp_path: Path) -> None:
    from torchfits.cache import cache_root, remote_cache_root, sample_cache_root

    for root in (cache_root(), remote_cache_root(), sample_cache_root()):
        assert root.is_relative_to(tmp_path), root


def test_clear_all_caches_smoke(tmp_path: Path) -> None:
    """clear_all_caches() should not raise and should leave torchfits usable."""
    # Write and read a file to populate caches.
    data = torch.ones((4, 4), dtype=torch.float32)
    path = str(tmp_path / "test.fits")
    torchfits.write(path, data, overwrite=True)
    _ = torchfits.read(path, hdu=0)

    # Clear everything.
    torchfits.clear_all_caches()

    # Should still work after clearing.
    result = torchfits.read(path, hdu=0)
    assert torch.equal(result, data)


def test_clear_all_caches_removes_disk_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """clear_all_caches() removes the cache_root directory."""
    cache_dir = tmp_path / "torchfits_cache"
    cache_dir.mkdir()
    (cache_dir / "remote").mkdir()
    (cache_dir / "samples").mkdir()
    (cache_dir / "remote" / "dummy.fits").write_bytes(b"fake")

    monkeypatch.setenv("TORCHFITS_CACHE_DIR", str(cache_dir))

    assert (cache_dir / "remote" / "dummy.fits").exists()

    torchfits.clear_all_caches()

    # The cache root directory should be removed (or at least empty).
    assert not cache_dir.exists() or not any(cache_dir.iterdir())


def test_clear_cache_disk_false_preserves_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """clear_cache() (default) should NOT remove disk cache files."""
    cache_dir = tmp_path / "torchfits_cache"
    cache_dir.mkdir()
    remote = cache_dir / "remote"
    remote.mkdir()
    (remote / "dummy.fits").write_bytes(b"fake")

    monkeypatch.setenv("TORCHFITS_CACHE_DIR", str(cache_dir))

    assert (remote / "dummy.fits").exists()

    torchfits.cache.clear_cache()

    # Disk files should still be there.
    assert (remote / "dummy.fits").exists()


def test_clear_all_caches_accessible_from_root() -> None:
    """clear_all_caches should be importable from the torchfits root."""
    assert hasattr(torchfits, "clear_all_caches")
    assert callable(torchfits.clear_all_caches)


def test_clear_cache_disk_true_parameter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """clear_cache(disk=True) removes the disk cache root.

    Deep-review unit 10, TE-004: this test was named and documented for
    ``disk=True`` and called ``disk=False``, so a reader counting coverage
    would credit the flag with a test that never exercised it. The branch is
    reachable directly, so it is now called directly.
    """
    cache_dir = tmp_path / "torchfits_cache"
    (cache_dir / "remote").mkdir(parents=True)
    (cache_dir / "remote" / "cached.fits").write_bytes(b"payload")

    monkeypatch.setenv("TORCHFITS_CACHE_DIR", str(cache_dir))
    assert (cache_dir / "remote" / "cached.fits").exists()

    torchfits.cache.clear_cache(disk=True)

    assert not cache_dir.exists() or not any(cache_dir.rglob("*"))


def test_cache_subsystem_still_works_after_clear() -> None:
    """Subsystem clear should not break after a full clear."""
    torchfits.clear_all_caches()
    # A full clear resets the counters, so this also proves the subsystem
    # accessor still answers rather than raising.
    stats = torchfits.get_cache_performance()
    assert stats["total_requests"] == 0, stats
    assert stats["hit_rate"] == 0.0, stats


def test_clear_cache_disk_keeps_user_files_in_overridden_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Override dirs outside cache_root() are user-owned: only torchfits
    download files go, the directories and anything else in them stay."""
    from torchfits.data.remote import cache_path_for_url

    root = tmp_path / "root"
    remote = tmp_path / "remote_elsewhere"
    samples = tmp_path / "samples_elsewhere"
    for d in (root, remote, samples):
        d.mkdir()
        (d / "f.fits").write_bytes(b"user data")
    (remote / "notes").mkdir()
    (remote / "notes" / "keep.txt").write_text("mine")
    monkeypatch.setenv("TORCHFITS_CACHE_DIR", str(root))
    monkeypatch.setenv("TORCHFITS_REMOTE_CACHE", str(remote))
    monkeypatch.setenv("TORCHFITS_SAMPLE_CACHE", str(samples))
    dl = cache_path_for_url("https://example.org/x.fits")
    assert dl.parent == remote
    dl.write_bytes(b"download")
    siblings = [
        dl.with_suffix(dl.suffix + ".partial"),
        Path(str(dl.with_suffix(dl.suffix + ".partial")) + ".meta"),
        dl.parent / f".{dl.name}.lock",
    ]
    for s in siblings:
        s.write_bytes(b"")

    torchfits.cache.clear_cache(disk=True)

    assert not root.exists()
    assert not dl.exists()
    assert not any(s.exists() for s in siblings)
    assert (remote / "f.fits").read_bytes() == b"user data"
    assert (remote / "notes" / "keep.txt").read_text() == "mine"
    assert (samples / "f.fits").read_bytes() == b"user data"


def test_clear_all_caches_never_removes_dir_containing_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A remote/sample override that contains the root (e.g. $HOME) survives."""
    home = tmp_path / "home"
    root = home / ".cache" / "torchfits"
    root.mkdir(parents=True)
    (home / "thesis.fits").write_bytes(b"precious")
    monkeypatch.setenv("TORCHFITS_CACHE_DIR", str(root))
    monkeypatch.setenv("TORCHFITS_REMOTE_CACHE", str(home))
    monkeypatch.setenv("TORCHFITS_SAMPLE_CACHE", str(home))

    torchfits.clear_all_caches()

    assert (home / "thesis.fits").read_bytes() == b"precious"
    assert home.is_dir()


def test_clear_cache_is_the_engine_clear_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """torchfits.cache.clear_cache() delegates the in-process layer once.

    It used to clear the native SharedReadMeta itself and then again through
    clear_file_cache(); the engine call now does it once, and the cached read
    results and counters are cleared with it.
    """
    import torchfits._C as cpp

    path = str(tmp_path / "t.fits")
    torchfits.write(path, torch.ones((2, 2)), overwrite=True)
    torchfits.read(path, hdu=0)
    torchfits.read(path, hdu=0)
    assert torchfits.get_cache_performance()["total_requests"] > 0

    calls: list[None] = []
    real = cpp.clear_shared_read_meta_cache
    monkeypatch.setattr(
        cpp, "clear_shared_read_meta_cache", lambda: (calls.append(None), real())[1]
    )
    torchfits.cache.clear_cache()
    assert len(calls) == 1
    assert torchfits.cache.get_cache_stats() == torchfits.get_cache_performance()
    assert torchfits.get_cache_performance()["total_requests"] == 0


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="needs /proc/self/fd")
def test_clear_all_caches_releases_cached_table_readers(tmp_path: Path) -> None:
    """Cached native table readers keep a CFITSIO handle and a pinned data
    descriptor open per file; a cache clear must close them, or a deleted
    file's disk space stays allocated until the process exits."""
    import os

    import numpy as np
    import torchfits.table as tf_table

    paths = []
    for i in range(4):
        path = tmp_path / f"t{i}.fits"
        tf_table.write(str(path), {"a": np.arange(10, dtype=np.int32)}, overwrite=True)
        paths.append(path)
        assert tf_table.read(str(path)).num_rows == 10

    def _open_fits_fds() -> int:
        count = 0
        for fd in os.listdir("/proc/self/fd"):
            try:
                target = os.readlink(f"/proc/self/fd/{fd}")
            except OSError:
                continue
            count += target.startswith(str(tmp_path))
        return count

    assert _open_fits_fds() > 0  # the reader cache is holding the files
    torchfits.clear_all_caches()
    assert _open_fits_fds() == 0
    assert tf_table.read(str(paths[0])).num_rows == 10
