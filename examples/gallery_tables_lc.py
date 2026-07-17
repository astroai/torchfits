#!/usr/bin/env python
"""Table meta-transforms + light-curve gallery with before/after plots."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._plotting import (  # noqa: E402
    save_lightcurve_before_after,
    save_spectrum_before_after,
)

from torchfits.transforms import (  # noqa: E402
    AsymmetricSigmaClip,
    FITSScaleColumns,
    PhaseFold,
    SavitzkyGolayFilter,
    SigmaClip,
    TNullToNan,
)


def _log(path: Path | None) -> None:
    print("wrote", path if path else "(figures skipped)")


def _demo_table_meta() -> None:
    stored = {
        "FLUX": torch.tensor([1000.0, 2000.0, 3000.0]),
        "COUNTS": torch.tensor([1, -999, 3], dtype=torch.int32),
    }
    scaled = FITSScaleColumns({"FLUX": (0.001, 10.0)})(
        {"FLUX": stored["FLUX"].clone(), "COUNTS": stored["COUNTS"].clone()}
    )
    cleaned = TNullToNan({"COUNTS": -999.0})(
        {"FLUX": stored["FLUX"].clone(), "COUNTS": stored["COUNTS"].clone()}
    )
    print(
        "FITSScaleColumns FLUX:",
        stored["FLUX"].tolist(),
        "->",
        scaled["FLUX"].tolist(),
    )
    print(
        "TNullToNan COUNTS NaNs:",
        int(torch.isnan(cleaned["COUNTS"]).sum().item()),
    )
    _log(
        save_spectrum_before_after(
            None,
            stored["FLUX"],
            scaled["FLUX"],
            "table_fits_scale_columns",
            titles=("stored", "TSCAL/TZERO"),
        )
    )


def _demo_lightcurve() -> None:
    rng = torch.Generator().manual_seed(0)
    n = 1200
    time = torch.linspace(0.0, 40.0, n)
    period = 5.0
    phase = (time / period) % 1.0
    flux = torch.ones(n)
    flux[(phase < 0.03) | (phase > 0.97)] -= 0.02
    flux = flux + 0.001 * torch.randn(n, generator=rng)
    spikes = torch.randint(0, n, (12,), generator=rng)
    flux[spikes] += 0.02

    clipped = AsymmetricSigmaClip(n_low=5.0, n_high=3.0, dim=(-1,))(flux.clone())
    _log(
        save_lightcurve_before_after(
            time,
            flux,
            time,
            clipped,
            "lightcurve_asymmetric_sigma_clip",
            titles=("raw", "asymmetric clip"),
        )
    )

    sym = SigmaClip(n_sigma=4.0, dim=(-1,))(flux.clone())
    _log(
        save_lightcurve_before_after(
            time,
            flux,
            time,
            sym,
            "lightcurve_sigma_clip",
            titles=("raw", "sigma clip"),
        )
    )

    folded = PhaseFold(period=period, n_bins=80)(flux.clone())
    phase_axis = torch.linspace(0.0, 1.0, folded.shape[-1])
    _log(
        save_lightcurve_before_after(
            time,
            flux,
            phase_axis,
            folded,
            "lightcurve_phase_fold",
            titles=("time series", "phase-folded"),
        )
    )

    smooth = SavitzkyGolayFilter(window_length=21, polyorder=3)(folded.clone())
    _log(
        save_lightcurve_before_after(
            phase_axis,
            folded,
            phase_axis,
            smooth,
            "lightcurve_savgol_folded",
            titles=("folded", "Savitzky–Golay"),
        )
    )


def main() -> int:
    _demo_table_meta()
    _demo_lightcurve()
    print("gallery_tables_lc OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
