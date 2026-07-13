"""Type annotation tests — verify mask parameter typing across all transforms.

These tests are designed to be checked with ``mypy --strict`` and
exercise every transform's ``forward`` / ``inverse`` with both
``mask=None`` and ``mask=tensor`` to ensure type compatibility.
"""

from __future__ import annotations

from typing import Any

import torch

from torchfits.transforms import (
    AlphaShapeContinuum,
    ArcsinhStretch,
    AsymmetricLeastSquares,
    AsymmetricSigmaClip,
    BackgroundSubtract,
    BandMath,
    Compose,
    ContinuumNormalize,
    ContinuumRemoval,
    DopplerShift,
    FITSHeaderNormalize,
    FITSHeaderScale,
    FITSScaleColumns,
    FITSTransform,
    GlobalScalarNorm,
    LogStretch,
    MinMaxNormalize,
    PercentileClipNormalize,
    PhaseFold,
    RobustNormalize,
    RunningPercentile,
    SavitzkyGolayFilter,
    SigmaClip,
    SpectralBinning,
    SqrtStretch,
    TNullToNan,
    UpperEnvelopeContinuum,
    WaveletDecompose,
    ZScaleNormalize,
    estimate_background,
    zscale_limits,
)


def _img() -> torch.Tensor:
    return torch.randn(4, 64, 64)


def _mask() -> torch.Tensor:
    return torch.ones(4, 64, 64, dtype=torch.bool)


def _spec() -> torch.Tensor:
    return torch.randn(4, 200)


def _spec_mask() -> torch.Tensor:
    return torch.ones(4, 200, dtype=torch.bool)


def _table() -> dict[str, torch.Tensor]:
    return {"FLUX": torch.randn(100), "QUAL": torch.zeros(100, dtype=torch.int32)}


_HEADER: dict[str, object] = {"BITPIX": -32, "BSCALE": 1.0, "BZERO": 0.0}


# ---- Base class & Compose ----

def test_fitstransform_call_accepts_mask() -> None:
    t: FITSTransform = ArcsinhStretch()
    img = _img()
    out = t(img, mask=None)
    out = t(img, mask=_mask())
    assert out.shape == img.shape


def test_compose_mask_threading() -> None:
    c = Compose([BackgroundSubtract(), ArcsinhStretch(a=0.1)])
    img = _img()
    m = _mask()
    fwd = c.forward(img.clone(), mask=None)
    fwd_m = c.forward(img.clone(), mask=m)
    _inv = c.inverse(fwd, mask=None)
    _inv_m = c.inverse(fwd, mask=m)
    assert fwd.shape == img.shape
    assert fwd_m.shape == img.shape


# ---- Stateless stretches ----

def test_arcsinh_stretch_mask() -> None:
    t = ArcsinhStretch(a=1.0)
    img = _img()
    out = t.forward(img, mask=None)
    _ = t.forward(img, mask=_mask())
    _ = t.inverse(out, mask=None)
    _ = t.inverse(out, mask=_mask())
    assert out.shape == img.shape


def test_log_stretch_mask() -> None:
    t = LogStretch()
    img = _img().abs()
    out = t.forward(img, mask=None)
    _ = t.forward(img, mask=_mask())
    _ = t.inverse(out, mask=None)
    _ = t.inverse(out, mask=_mask())
    assert out.shape == img.shape


def test_sqrt_stretch_mask() -> None:
    t = SqrtStretch()
    img = _img().abs()
    out = t.forward(img, mask=None)
    _ = t.forward(img, mask=_mask())
    _ = t.inverse(out, mask=None)
    _ = t.inverse(out, mask=_mask())
    assert out.shape == img.shape


# ---- Stateful normalizers ----

def test_zscale_normalize_mask() -> None:
    t = ZScaleNormalize()
    img = _img()
    out = t.forward(img.clone(), mask=None)
    _ = t.forward(img.clone(), mask=_mask())
    _ = t.inverse(out, mask=None)
    _ = t.inverse(out, mask=_mask())
    assert out.shape == img.shape


def test_robust_normalize_mask() -> None:
    t = RobustNormalize()
    img = _img()
    out = t.forward(img.clone(), mask=None)
    _ = t.forward(img.clone(), mask=_mask())
    _ = t.inverse(out, mask=None)
    _ = t.inverse(out, mask=_mask())
    assert out.shape == img.shape


def test_background_subtract_mask() -> None:
    t = BackgroundSubtract()
    img = _img()
    out = t.forward(img.clone(), mask=None)
    _ = t.forward(img.clone(), mask=_mask())
    _ = t.inverse(out, mask=None)
    _ = t.inverse(out, mask=_mask())
    assert out.shape == img.shape


def test_percentile_clip_normalize_mask() -> None:
    t = PercentileClipNormalize()
    img = _img()
    out = t.forward(img.clone(), mask=None)
    _ = t.forward(img.clone(), mask=_mask())
    _ = t.inverse(out, mask=None)
    _ = t.inverse(out, mask=_mask())
    assert out.shape == img.shape


def test_minmax_normalize_mask() -> None:
    t = MinMaxNormalize()
    img = _img()
    out = t.forward(img.clone(), mask=None)
    _ = t.forward(img.clone(), mask=_mask())
    _ = t.inverse(out, mask=None)
    _ = t.inverse(out, mask=_mask())
    assert out.shape == img.shape


# ---- FITS metadata-aware transforms ----

def test_fits_header_scale_mask() -> None:
    t = FITSHeaderScale(bscale=2.0, bzero=10.0)
    img = _img()
    out = t.forward(img, mask=None)
    _ = t.forward(img, mask=_mask())
    _ = t.inverse(out, mask=None)
    _ = t.inverse(out, mask=_mask())
    assert out.shape == img.shape


def test_fits_header_normalize_mask() -> None:
    t = FITSHeaderNormalize(_HEADER, scale_floats=True)
    img = _img()
    out = t.forward(img.clone(), mask=None)
    _ = t.forward(img.clone(), mask=_mask())
    _ = t.inverse(out, mask=None)
    _ = t.inverse(out, mask=_mask())
    assert out.shape == img.shape


def test_fits_scale_columns_mask() -> None:
    t = FITSScaleColumns({"FLUX": (2.0, 0.0)})
    tbl = _table()
    out = t.forward(tbl, mask=None)
    _ = t.forward(tbl, mask=_mask())
    _ = t.inverse(out, mask=None)
    _ = t.inverse(out, mask=_mask())
    assert set(out.keys()) == set(tbl.keys())


def test_tnull_to_nan_mask() -> None:
    t = TNullToNan({"FLUX": -999.0})
    tbl = _table()
    out = t.forward(tbl, mask=None)
    _ = t.forward(tbl, mask=_mask())
    assert set(out.keys()) == set(tbl.keys())


# ---- Spectral transforms ----

def test_continuum_normalize_mask() -> None:
    t = ContinuumNormalize(order=1)
    spec = _spec()
    out = t.forward(spec, mask=None)
    _ = t.forward(spec, mask=_spec_mask())
    _ = t.inverse(out, mask=None)
    _ = t.inverse(out, mask=_spec_mask())
    assert out.shape == spec.shape


def test_doppler_shift_mask() -> None:
    t = DopplerShift(z=0.1)
    spec = _spec()
    out = t.forward(spec, mask=None)
    _ = t.forward(spec, mask=_spec_mask())
    assert out.shape[-1] > spec.shape[-1]


# ---- Continuum / baseline estimators ----

def test_continuum_removal_mask() -> None:
    t = ContinuumRemoval(method="polynomial", order=1)
    spec = _spec()
    out = t.forward(spec, mask=None)
    _ = t.forward(spec, mask=_spec_mask())
    _ = t.inverse(out, mask=None)
    _ = t.inverse(out, mask=_spec_mask())
    assert out.shape == spec.shape


def test_savitzky_golay_filter_mask() -> None:
    t = SavitzkyGolayFilter(window_length=7, polyorder=3)
    spec = _spec()
    out = t.forward(spec, mask=None)
    _ = t.forward(spec, mask=_spec_mask())
    _ = t.inverse(out, mask=None)
    _ = t.inverse(out, mask=_spec_mask())
    assert out.shape == spec.shape


def test_running_percentile_mask() -> None:
    t = RunningPercentile(percentile=90, window_size=7)
    spec = _spec()
    out = t.forward(spec, mask=None)
    _ = t.forward(spec, mask=_spec_mask())
    _ = t.inverse(out, mask=None)
    _ = t.inverse(out, mask=_spec_mask())
    assert out.shape == spec.shape


def test_upper_envelope_continuum_mask() -> None:
    t = UpperEnvelopeContinuum(window=7)
    spec = _spec()
    out = t.forward(spec, mask=None)
    _ = t.forward(spec, mask=_spec_mask())
    _ = t.inverse(out, mask=None)
    _ = t.inverse(out, mask=_spec_mask())
    assert out.shape == spec.shape


def test_wavelet_decompose_mask() -> None:
    x = torch.randn(4, 64)  # power of 2
    m = torch.ones(4, 64, dtype=torch.bool)
    t = WaveletDecompose(levels=2)
    out = t.forward(x.clone(), mask=None)
    _ = t.forward(x.clone(), mask=m)
    _ = t.inverse(out, mask=None)
    _ = t.inverse(out, mask=m)
    assert out.shape == x.shape


def test_asymmetric_least_squares_mask() -> None:
    t = AsymmetricLeastSquares(lam=1e3, max_iter=3)
    spec = _spec()
    out = t.forward(spec, mask=None)
    _ = t.forward(spec, mask=_spec_mask())
    _ = t.inverse(out, mask=None)
    _ = t.inverse(out, mask=_spec_mask())
    assert out.shape == spec.shape


def test_alpha_shape_continuum_mask() -> None:
    t = AlphaShapeContinuum(half_window=5, iterations=1)
    spec = _spec()
    out = t.forward(spec, mask=None)
    _ = t.forward(spec, mask=_spec_mask())
    _ = t.inverse(out, mask=None)
    _ = t.inverse(out, mask=_spec_mask())
    assert out.shape == spec.shape


def test_band_math_mask() -> None:
    x = torch.randn(4, 32, 32)
    t = BandMath(lambda b: (b[1] - b[0]) / (b[1] + b[0] + 1e-8))
    out = t.forward(x, mask=None)
    _ = t.forward(x, mask=_mask())
    assert out.shape == (32, 32)


def test_global_scalar_norm_mask() -> None:
    t = GlobalScalarNorm(stat="median")
    img = _img()
    out = t.forward(img.clone(), mask=None)
    _ = t.forward(img.clone(), mask=_mask())
    _ = t.inverse(out, mask=None)
    _ = t.inverse(out, mask=_mask())
    assert out.shape == img.shape


# ---- Irreversible transforms (SigmaClip, PhaseFold, SpectralBinning) ----

def test_sigma_clip_mask() -> None:
    t = SigmaClip(n_sigma=3.0, max_iter=3)
    img = _img()
    out = t.forward(img.clone(), mask=None)
    _ = t.forward(img.clone(), mask=_mask())
    assert out.shape == img.shape


def test_asymmetric_sigma_clip_mask() -> None:
    t = AsymmetricSigmaClip(n_low=2.0, n_high=3.0)
    img = _img()
    out = t.forward(img.clone(), mask=None)
    _ = t.forward(img.clone(), mask=_mask())
    assert out.shape == img.shape


def test_phase_fold_mask() -> None:
    t = PhaseFold(period=10.0, n_bins=16)
    spec = _spec()
    out = t.forward(spec, mask=None)
    _ = t.forward(spec, mask=_spec_mask())
    assert out.shape[-1] == 16


def test_spectral_binning_mask() -> None:
    t = SpectralBinning(factor=2, mode="mean")
    spec = _spec()
    out = t.forward(spec, mask=None)
    _ = t.forward(spec, mask=_spec_mask())
    _ = t.inverse(out, mask=None)
    _ = t.inverse(out, mask=_spec_mask())
    assert out.shape[-1] == spec.shape[-1] // 2


# ---- Standalone helpers ----

def test_estimate_background_mask() -> None:
    img = _img()
    med, std = estimate_background(img, mask=None)
    _med_m, _std_m = estimate_background(img, mask=_mask())
    assert med.shape == (4, 1, 1) and std.shape == (4, 1, 1)


def test_zscale_limits_mask() -> None:
    img = _img()
    z1, z2 = zscale_limits(img, mask=None)
    _z1_m, _z2_m = zscale_limits(img, mask=_mask())
    assert z1.shape == (4, 1, 1) and z2.shape == (4, 1, 1)


# ---- Base class return type preservation ----

def test_base_class_returns_any() -> None:
    """FITSTransform.forward/inverse return Any, but subclasses refine it."""
    t: FITSTransform = ArcsinhStretch()
    img = _img()
    result: Any = t.forward(img, mask=None)
    result = t.forward(img, mask=_mask())
    result = t.inverse(img, mask=None)
    result = t.inverse(img, mask=_mask())
    assert result is not None
