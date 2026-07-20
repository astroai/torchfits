"""Ranking groups must not mix mmap-on and mmap-off peers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_contract import annotate_rankings, compute_deficits


def _row(
    *,
    domain: str = "fits",
    case_id: str,
    family: str,
    library: str,
    method: str,
    mmap_target: str,
    time_s: float,
    comparable: bool = True,
) -> dict:
    return {
        "domain": domain,
        "case_id": case_id,
        "case_label": case_id,
        "operation": "read_full",
        "family": family,
        "library": library,
        "method": method,
        "status": "OK",
        "comparable": comparable,
        "mmap_target": mmap_target,
        "time_s": time_s,
        "n_points": 64,
        "metadata": {},
    }


def test_mmap_modes_rank_independently() -> None:
    rows = [
        _row(
            case_id="tiny_int8_1d::read_full_gpu",
            family="smart",
            library="torchfits",
            method="torchfits",
            mmap_target="off",
            time_s=7.26e-3,
        ),
        _row(
            case_id="tiny_int8_1d::read_full_gpu",
            family="smart",
            library="fitsio",
            method="fitsio_torch",
            mmap_target="off",
            time_s=7.40e-3,
        ),
        # Faster fitsio on mmap-on must not demote mmap-off torchfits.
        _row(
            case_id="tiny_int8_1d::read_full_gpu",
            family="smart",
            library="fitsio",
            method="fitsio_torch",
            mmap_target="on",
            time_s=7.20e-3,
        ),
        _row(
            case_id="tiny_int8_1d::read_full_gpu",
            family="smart",
            library="torchfits",
            method="torchfits",
            mmap_target="on",
            time_s=9.33e-3,
        ),
    ]
    annotate_rankings(rows)
    off_tf = next(
        r for r in rows if r["mmap_target"] == "off" and r["library"] == "torchfits"
    )
    on_tf = next(
        r for r in rows if r["mmap_target"] == "on" and r["library"] == "torchfits"
    )
    assert off_tf["rank_in_family"] == 1
    assert off_tf["best_in_family"] is True
    assert on_tf["rank_in_family"] == 2

    # Images: any meaningful lag is a deficit (~15.7% here).
    deficits = compute_deficits(rows, run_id="test")
    assert len(deficits) == 1
    assert deficits[0]["mmap_target"] == "on"


def test_image_deficits_count_any_lag() -> None:
    rows = [
        _row(
            case_id="noise::read_full",
            family="smart",
            library="torchfits",
            method="torchfits",
            mmap_target="off",
            time_s=2.20e-3,
        ),
        _row(
            case_id="noise::read_full",
            family="smart",
            library="fitsio",
            method="fitsio_torch",
            mmap_target="off",
            time_s=1.00e-3,
        ),
    ]
    annotate_rankings(rows)
    deficits = compute_deficits(rows, run_id="test")
    assert len(deficits) == 1
    assert deficits[0]["domain"] == "fits"


def test_timer_epsilon_absorbs_only_clock_noise() -> None:
    """Image float-timer ε is absolute (~0.2ms), never a percent-of-median floor."""
    rows = [
        _row(
            case_id="plain::read_full",
            family="smart",
            library="torchfits",
            method="torchfits",
            mmap_target="off",
            time_s=60.0e-3 + 1e-4,  # 0.1ms — under ε
        ),
        _row(
            case_id="plain::read_full",
            family="smart",
            library="fitsio",
            method="fitsio_torch",
            mmap_target="off",
            time_s=60.0e-3,
        ),
    ]
    annotate_rankings(rows)
    noise = compute_deficits(rows, run_id="test")
    assert len(noise) == 1
    assert noise[0]["significance"] == "noise"

    rows[0]["time_s"] = 60.0e-3 + 2.5e-4  # 0.25ms — above ε, is significant
    annotate_rankings(rows)
    deficits = compute_deficits(rows, run_id="test")
    assert len(deficits) == 1
    assert deficits[0]["significance"] == "significant"


def test_table_arrow_allows_1_05() -> None:
    rows = [
        _row(
            domain="fitstable",
            case_id="narrow::predicate",
            family="smart",
            library="torchfits",
            method="torchfits",
            mmap_target="off",
            time_s=1.05e-3,  # inclusive 1.05× slack
        ),
        _row(
            domain="fitstable",
            case_id="narrow::predicate",
            family="smart",
            library="fitsio",
            method="fitsio_torch",
            mmap_target="off",
            time_s=1.00e-3,
        ),
    ]
    annotate_rankings(rows)
    noise = compute_deficits(rows, run_id="test")
    assert len(noise) == 1
    assert noise[0]["significance"] == "noise"

    rows[0]["time_s"] = 1.06e-3
    annotate_rankings(rows)
    deficits = compute_deficits(rows, run_id="test")
    assert len(deficits) == 1
    assert deficits[0]["domain"] == "fitstable"
    assert deficits[0]["significance"] == "significant"


def test_deficits_require_external_peer() -> None:
    rows = [
        _row(
            case_id="gpu::read_full",
            family="specialized",
            library="torchfits",
            method="torchfits_specialized_device",
            mmap_target="off",
            time_s=2e-3,
        ),
        _row(
            case_id="gpu::read_full",
            family="specialized",
            library="torchfits",
            method="torchfits_dtype_fair_device",
            mmap_target="off",
            time_s=1e-3,
        ),
    ]
    annotate_rankings(rows)
    assert compute_deficits(rows, run_id="test") == []
