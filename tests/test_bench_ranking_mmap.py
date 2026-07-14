"""Ranking groups must not mix mmap-on and mmap-off peers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_contract import annotate_rankings, compute_deficits


def _row(
    *,
    case_id: str,
    family: str,
    library: str,
    method: str,
    mmap_target: str,
    time_s: float,
    comparable: bool = True,
) -> dict:
    return {
        "domain": "fits",
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
            time_s=7.26e-5,
        ),
        _row(
            case_id="tiny_int8_1d::read_full_gpu",
            family="smart",
            library="fitsio",
            method="fitsio_torch",
            mmap_target="off",
            time_s=7.40e-5,
        ),
        # Faster fitsio on mmap-on must not demote mmap-off torchfits.
        _row(
            case_id="tiny_int8_1d::read_full_gpu",
            family="smart",
            library="fitsio",
            method="fitsio_torch",
            mmap_target="on",
            time_s=7.20e-5,
        ),
        _row(
            case_id="tiny_int8_1d::read_full_gpu",
            family="smart",
            library="torchfits",
            method="torchfits",
            mmap_target="on",
            time_s=8.33e-5,
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

    # mmap-on is ~15% behind — above the 8% deficit noise floor.
    deficits = compute_deficits(rows, run_id="test")
    assert len(deficits) == 1
    assert deficits[0]["mmap_target"] == "on"
    assert abs(float(deficits[0]["lag_ratio"]) - (8.33e-5 / 7.20e-5)) < 1e-9


def test_deficit_noise_floor_skips_sub_8pct() -> None:
    rows = [
        _row(
            case_id="noise::read_full",
            family="smart",
            library="torchfits",
            method="torchfits",
            mmap_target="off",
            time_s=1.07e-3,
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
    assert compute_deficits(rows, run_id="test") == []
