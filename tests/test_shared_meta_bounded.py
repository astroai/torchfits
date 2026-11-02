"""The per-path metadata registry is bounded.

It gained one entry per distinct path and never shed any, so a long scan over
many files held every path it had ever touched for the life of the process.
"""

from __future__ import annotations

import pytest

core = pytest.importorskip("torchfits._core")


def test_distinct_paths_do_not_grow_the_registry_without_bound(tmp_path) -> None:
    core.clear_shared_read_meta_cache()
    try:
        # Missing files still register a path before the open fails, which
        # makes this cheap: no fixture files are needed.
        for i in range(6000):
            with pytest.raises(RuntimeError):
                core.read_nrows(str(tmp_path / f"missing{i}.fits"), 1)
        count = core.shared_meta_entry_count()
        assert 0 < count <= 4096, count
    finally:
        core.clear_shared_read_meta_cache()
    assert core.shared_meta_entry_count() == 0
