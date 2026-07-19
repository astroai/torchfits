"""Type annotation tests — verify mask parameter typing across all transforms.

These tests are designed to be checked with ``mypy --strict`` and
exercise every transform's ``forward`` / ``inverse`` with both
``mask=None`` and ``mask=tensor`` to ensure type compatibility.
"""

from __future__ import annotations

from typing import Any

import torch

from torchfits.transforms import (
    ArcsinhStretch,
    AsymmetricSigmaClip,
    AsModule,
    BackgroundSubtract,
    Compose,
    FITSHeaderNormalize,
    FITSHeaderScale,
    FITSScaleColumns,
    FITSTransform,
    GlobalScalarNorm,
    LogStretch,
    MinMaxNormalize,
    PercentileClipNormalize,
    RobustNormalize,
    SigmaClip,
    SqrtStretch,
    TNullToNan,
    ZScaleNormalize,
    as_module,
    estimate_background,
    zscale_limits,
)


def _img() -> torch.Tensor:
    return torch.randn(4, 64, 64)


def _mask() -> torch.Tensor:
    return torch.ones(4, 64, 64, dtype=torch.bool)


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


def test_as_module_wraps_transform() -> None:
    wrapped = as_module(ArcsinhStretch(a=0.1))
    assert isinstance(wrapped, AsModule)
    img = _img()
    out = wrapped(img, mask=None)
    _ = wrapped(img, mask=_mask())
    assert out.shape == img.shape


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


def test_global_scalar_norm_mask() -> None:
    t = GlobalScalarNorm(stat="median")
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


# ---- Irreversible transforms (SigmaClip) ----


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
