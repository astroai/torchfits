#!/usr/bin/env python
"""Spectrum / continuum transform gallery — before/after PNGs (SDSS or synthetic)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._plotting import save_spectrum_before_after  # noqa: E402
from examples._sample_data import SampleUnavailable, try_ensure_sample  # noqa: E402

from torchfits.transforms import (  # noqa: E402
    AlphaShapeContinuum,
    AsymmetricLeastSquares,
    BandMath,
    ContinuumNormalize,
    ContinuumRemoval,
    DopplerShift,
    RunningPercentile,
    SavitzkyGolayFilter,
    SpectralBinning,
    UpperEnvelopeContinuum,
    WaveletDecompose,
)


def _synthetic_spectrum(n: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    wave = torch.linspace(4000.0, 7000.0, n)
    continuum = 1.0 + 0.15 * torch.sin((wave - 4000) / 800)
    lines = -0.4 * torch.exp(-0.5 * ((wave - 4861) / 8) ** 2)
    lines = lines - 0.55 * torch.exp(-0.5 * ((wave - 6563) / 10) ** 2)
    lines = lines + 0.25 * torch.exp(-0.5 * ((wave - 5007) / 6) ** 2)
    flux = continuum * (1.0 + lines) + 0.01 * torch.randn(n)
    return wave, flux.float()


def _load_spectrum() -> tuple[torch.Tensor | None, torch.Tensor]:
    path = try_ensure_sample("sdss_spectrum")
    if path is None:
        return _synthetic_spectrum()
    try:
        import numpy as np
        from astropy.io import fits
    except ImportError:
        return _synthetic_spectrum()

    with fits.open(path) as hdul:
        # SDSS coadd spectrum is typically HDU 1 with loglam / flux columns.
        data = hdul[1].data
        names = {n.lower(): n for n in data.dtype.names or ()}
        flux_key = names.get("flux")
        loglam_key = names.get("loglam")
        if flux_key is None:
            return _synthetic_spectrum()
        flux = torch.as_tensor(np.asarray(data[flux_key], dtype=np.float64)).float()
        wave = None
        if loglam_key is not None:
            wave = (10.0 ** torch.as_tensor(np.asarray(data[loglam_key], dtype=np.float64))).float()
        # Trim to a manageable window for plots.
        if flux.numel() > 2500:
            mid = flux.numel() // 2
            sl = slice(mid - 1000, mid + 1000)
            flux = flux[sl]
            if wave is not None:
                wave = wave[sl]
        return wave, flux


def main() -> int:
    wave, flux = _load_spectrum()
    print(f"spectrum n={flux.numel()} finite={torch.isfinite(flux).sum().item()}")

    # Continuum normalize / remove (specutils-shaped 3-panel).
    cn = ContinuumNormalize(order=3, n_sigma=2.0)
    normed = cn(flux.clone())
    assert cn._continuum is not None
    print(
        "wrote",
        save_spectrum_before_after(
            wave,
            flux,
            normed,
            "spectrum_continuum_normalize",
            continuum=cn._continuum,
            titles=("flux", "normalized"),
        ),
    )

    cr = ContinuumRemoval(order=3, n_sigma=2.0)
    removed = cr(flux.clone())
    assert cr._baseline is not None
    print(
        "wrote",
        save_spectrum_before_after(
            wave,
            flux,
            removed,
            "spectrum_continuum_removal",
            continuum=cr._baseline,
            titles=("flux", "residual"),
        ),
    )

    # Continuum estimators: forward returns residual/smoothed; continuum = flux - residuals.
    estimators = [
        ("savitzky_golay", SavitzkyGolayFilter(window_length=51, polyorder=3)),
        ("running_percentile", RunningPercentile(percentile=90.0, window_size=51)),
        ("upper_envelope", UpperEnvelopeContinuum(window=25)),
        ("als", AsymmetricLeastSquares(lam=1e5, p=0.01, max_iter=5)),
        ("alpha_shape", AlphaShapeContinuum(half_window=25, iterations=2)),
    ]
    for tag, xf in estimators:
        out = xf(flux.clone())
        residuals = getattr(xf, "_residuals", None)
        continuum = flux - residuals if residuals is not None else None
        print(
            "wrote",
            save_spectrum_before_after(
                wave,
                flux,
                out,
                f"spectrum_{tag}",
                continuum=continuum,
                titles=("flux", tag),
            ),
        )

    # Wavelet: show reconstructed approximation length side-by-side with flux.
    wv = WaveletDecompose(levels=3)
    coeffs = wv(flux.clone())
    print(
        "wrote",
        save_spectrum_before_after(
            None,
            flux,
            coeffs,
            "spectrum_wavelet",
            titles=("flux", "haar coeffs"),
        ),
    )

    doppler = DopplerShift(z=0.05)
    shifted = doppler(flux.clone())
    print(
        "wrote",
        save_spectrum_before_after(
            wave,
            flux,
            shifted,
            "spectrum_doppler_shift",
            titles=("rest", "z=0.05"),
        ),
    )

    binned = SpectralBinning(factor=4, mode="mean")(flux.clone())
    print(
        "wrote",
        save_spectrum_before_after(
            None,
            flux,
            binned,
            "spectrum_spectral_binning",
            titles=("native", "bin×4"),
        ),
    )

    # BandMath on a tiny synthetic cube (C, H, W).
    cube = torch.rand(4, 32, 32)
    cube[2] = cube[3] * 0.7 + 0.1  # correlated red/nir-ish
    ndvi = BandMath(lambda b: (b[3] - b[2]) / (b[3] + b[2] + 1e-8))(cube)
    from examples._plotting import save_image_before_after

    print(
        "wrote",
        save_image_before_after(
            cube[2],
            ndvi,
            "spectrum_bandmath_ndvi",
            titles=("red", "NDVI-like"),
            cmap="viridis",
        ),
    )
    print("gallery_spectra OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SampleUnavailable as exc:
        print(f"SKIP: {exc}")
        raise SystemExit(0) from exc
