"""Example: MaNGA DRP LOGCUBE — named-HDU reads, spaxel spectra, narrowband map.

FITS stores axes as (x, y, wave); torchfits/numpy report the reversed shape
``(wave, y, x)``. MASK bit ``2**10`` flags DONOTUSE-class pixels here.

Skips cleanly if the (~200MB) sample isn't cached — fetch via
``bash scripts/fetch_example_samples.sh --with-manga``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._sample_data import try_ensure_sample  # noqa: E402

import torchfits  # noqa: E402

HALPHA_AIR_A = 6562.8
DONOTUSE_BIT = 1 << 10


def main() -> int:
    path = try_ensure_sample("manga_logcube")
    if path is None:
        print(
            "SKIP: sample 'manga_logcube' not cached. "
            "Fetch via: bash scripts/fetch_example_samples.sh --with-manga"
        )
        return 0

    flux, ivar, mask = torchfits.read_hdus(str(path), hdus=["FLUX", "IVAR", "MASK"])
    wave = torchfits.read(str(path), hdu="WAVE", mode="image")
    print(
        f"FLUX (wave,y,x)={tuple(flux.shape)} IVAR={tuple(ivar.shape)} MASK={tuple(mask.shape)}"
    )
    print(f"WAVE: n={wave.numel()} range=[{wave.min():.1f}, {wave.max():.1f}] A")

    n_wave, ny, nx = flux.shape
    cy, cx = ny // 2, nx // 2
    spectrum = flux[:, cy, cx]
    print(
        f"central spaxel ({cy},{cx}): mean={spectrum.mean():.3f} "
        f"std={spectrum.std():.3f} finite={torch.isfinite(spectrum).sum().item()}/{n_wave}"
    )

    wave_idx = int(torch.argmin((wave - HALPHA_AIR_A).abs()))
    lo, hi = max(0, wave_idx - 2), min(n_wave, wave_idx + 3)
    band = flux[lo:hi]
    band_mask = mask[lo:hi]
    valid = (band_mask.to(torch.int64) & DONOTUSE_BIT) == 0
    band = torch.where(valid, band, torch.zeros_like(band))
    counts = valid.sum(dim=0).clamp(min=1)
    halpha_map = band.sum(dim=0) / counts
    print(
        f"Ha narrowband (idx {lo}:{hi}, ~{wave[wave_idx]:.1f}A): "
        f"shape={tuple(halpha_map.shape)} mean={halpha_map.mean():.3f} "
        f"max={halpha_map.max():.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
