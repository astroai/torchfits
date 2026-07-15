"""Unit checks for modular bench suites, operation filters, and RSS timing."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_contract import write_summary
from benchmarks.bench_timing import time_median, time_medians_interleaved
from benchmarks.suites import DEFICIT_FOCUS_SUITES, list_suite_names, resolve_suite


def test_suite_registry_resolves_aliases() -> None:
    s = resolve_suite("hcompress")
    assert s.name == "compressed_hcompress"
    assert s.scope == "fits"
    assert "hcompress" in s.case_filter or "compressed_hcompress" in s.case_filter
    assert "release" in list_suite_names()
    assert "compressed_hcompress" in DEFICIT_FOCUS_SUITES


def test_fitstable_predicate_suite_has_operation_filter() -> None:
    s = resolve_suite("fitstable_predicate")
    assert s.scope == "fitstable"
    assert "predicate" in s.operation
    assert s.no_gpu is True


def test_gpu_transports_suite_is_gpu_only() -> None:
    s = resolve_suite("gpu_transports")
    assert s.gpu_only is True


def test_time_median_reports_peak_rss() -> None:
    payload = bytearray(2 * 1024 * 1024)

    def _alloc() -> int:
        # Touch the buffer so RSS samples see real residency.
        payload[0] = 1
        payload[-1] = 2
        return len(payload)

    median, peak_rss, _peak_cuda, err = time_median(_alloc, runs=3, warmup=1)
    assert err is None
    assert median is not None and median >= 0.0
    # psutil may be absent in minimal envs; when present RSS must be finite.
    if peak_rss is not None:
        assert peak_rss > 0.0


def test_interleaved_warmup_failure_soft_skips() -> None:
    def ok() -> int:
        return 1

    def boom() -> int:
        raise RuntimeError("peer_warmup_fail")

    out = time_medians_interleaved(
        {"ok": ok, "boom": boom},
        runs=2,
        warmup=1,
    )
    assert out["ok"][0] is not None and out["ok"][3] is None
    assert out["boom"][0] is None and out["boom"][3] is not None


def test_scorecard_counts_table_within_floor() -> None:
    rows = [
        {
            "domain": "fitstable",
            "case_id": "narrow::predicate_filter",
            "family": "smart",
            "library": "torchfits",
            "method": "torchfits",
            "comparable": True,
            "status": "OK",
            "time_s": 1.04,
            "mmap_target": "off",
            "n_points": 1000,
            "metadata": {},
        },
        {
            "domain": "fitstable",
            "case_id": "narrow::predicate_filter",
            "family": "smart",
            "library": "fitsio",
            "method": "fitsio",
            "comparable": True,
            "status": "OK",
            "time_s": 1.0,
            "mmap_target": "off",
            "n_points": 1000,
            "metadata": {},
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "summary.md"
        write_summary(path, run_id="t", scopes=["fitstable"], rows=rows, deficits=[])
        text = path.read_text(encoding="utf-8")
        assert "1/1" in text
