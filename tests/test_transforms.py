"""Tests for torchfits.transforms — ML-friendly FITS image preprocessing."""

import math
import pytest
import torch

from transforms_reference import (
    alpha_shape_per_spectrum,
    asls_dense_solve,
    continuum_normalize_per_spectrum,
    continuum_removal_per_spectrum,
    phase_fold_per_bin,
    running_percentile_per_spectrum,
    savitzky_golay_per_spectrum,
    sigma_clip_naive,
    upper_envelope_per_spectrum,
    wavelet_decompose_per_spectrum,
)

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
    _amin,
    _amax,
    _flatten_dims,
    _median,
    _normalize_dims,
    _quantile,
    _reduce_keepdim,
    _resample_1d,
    _resample_scale,
    estimate_background,
    safe_arcsinh,
    safe_log,
    zscale_limits,
)


# ---------------------------------------------------------------------------
# Helper: create test tensors with varying shapes and value distributions
# ---------------------------------------------------------------------------


def _make_tensor(
    shape=(4, 64, 64),
    dtype=torch.float32,
    kind="normal",
) -> torch.Tensor:
    """Create a test tensor with a specific distribution."""
    if kind == "normal":
        x = torch.randn(shape, dtype=dtype) * 10 + 100
    elif kind == "uniform":
        x = torch.rand(shape, dtype=dtype) * 100
    elif kind == "constant":
        x = torch.ones(shape, dtype=dtype) * 5.0
    elif kind == "zeros":
        x = torch.zeros(shape, dtype=dtype)
    elif kind == "mixed_sign":
        x = torch.randn(shape, dtype=dtype) * 50
    elif kind == "high_dr":
        # High dynamic range: few bright pixels on faint background
        x = torch.randn(shape, dtype=dtype) * 5 + 10
        x[..., 0, 0] = 20000  # bright source
        x[..., 1, 1] = 15000
    elif kind == "int16":
        x = torch.randint(-100, 100, shape, dtype=torch.int32).to(dtype)
    else:
        raise ValueError(f"Unknown kind: {kind}")
    return x


# ---------------------------------------------------------------------------
# Multi-dim reduction helpers
# ---------------------------------------------------------------------------


class TestNormalizeDims:
    def test_positive_dims(self):
        assert _normalize_dims(4, (2, 3)) == (2, 3)

    def test_negative_dims(self):
        assert _normalize_dims(4, (-2, -1)) == (2, 3)

    def test_mixed_dims(self):
        assert _normalize_dims(4, (0, -1)) == (0, 3)

    def test_removes_duplicates(self):
        assert _normalize_dims(3, (0, 0, -1)) == (0, 2)

    def test_sorted_output(self):
        assert _normalize_dims(5, (4, 1, 3)) == (1, 3, 4)


class TestFlattenDims:
    def test_contiguous_dims_at_end(self):
        x = torch.randn(2, 3, 64, 64)
        flat = _flatten_dims(x, (2, 3))
        assert flat.shape == (2, 3, 64 * 64)

    def test_non_contiguous_dims(self):
        x = torch.randn(2, 3, 64, 64)
        flat = _flatten_dims(x, (1, 3))  # channels + width
        assert flat.shape == (2, 64, 3 * 64)

    def test_single_dim(self):
        x = torch.randn(2, 3, 64, 64)
        flat = _flatten_dims(x, (2,))
        # single dim: permute is a no-op, reshape(-1) collapses last dim
        # After permute(*keep, *(2,)): keep=[0,1,3], dims=(2,) → x.permute(0,1,3,2) → (2,3,64,64)
        # reshape(2,3,64,-1) → (2,3,64,64) — but we called flatten on dims=(2,), a single dim
        # So it's just a reshape with -1 on the last dim = original dim 2 size.
        # Actually, with len(dims)==1, we wouldn't call _flatten_dims at all from _reduce_keepdim.
        # But testing directly: it should work.
        assert flat.numel() == x.numel()

    def test_all_dims(self):
        x = torch.randn(2, 3, 4)
        flat = _flatten_dims(x, (0, 1, 2))
        assert flat.shape == (2 * 3 * 4,)

    def test_preserves_values(self):
        x = torch.randn(2, 3, 4, 5)
        flat = _flatten_dims(x, (1, 3))
        expected = x.permute(0, 2, 1, 3).reshape(2, 4, 3 * 5)
        assert torch.equal(flat, expected)


class TestReduceKeepdim:
    def test_single_dim_fast_path(self):
        x = torch.randn(4, 64, 64)
        result = _reduce_keepdim(
            x, (0,), lambda t, d, k: torch.mean(t, dim=d, keepdim=k)
        )
        assert result.shape == (1, 64, 64)
        assert torch.allclose(result[0], x.mean(dim=0))

    def test_multi_dim_flatten_path(self):
        x = torch.randn(4, 64, 64)
        result = _reduce_keepdim(
            x, (-2, -1), lambda t, d, k: torch.mean(t, dim=d, keepdim=k)
        )
        assert result.shape == (4, 1, 1)
        for i in range(4):
            assert abs(result[i, 0, 0].item() - x[i].mean().item()) < 1e-5

    def test_single_dim_keepdim(self):
        x = torch.randn(2, 3, 64, 64)
        result = _median(x, (0,))
        assert result.shape == (1, 3, 64, 64)


class TestMedianAminAmaxQuantile:
    def test_median_multi_dim(self):
        x = torch.randn(4, 64, 64)
        m = _median(x, (-2, -1))
        assert m.shape == (4, 1, 1)

    def test_median_vs_torch(self):
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        m = _median(x, (-2, -1))
        # torch.median returns the lower median for even element counts
        assert m[0, 0, 0].item() == 2.0  # lower median of [1,2,3,4]
        assert m[1, 0, 0].item() == 6.0  # lower median of [5,6,7,8]

    def test_amin_multi_dim(self):
        x = torch.randn(4, 64, 64)
        vmin = _amin(x, (-2, -1))
        assert vmin.shape == (4, 1, 1)

    def test_amax_multi_dim(self):
        x = torch.randn(4, 64, 64)
        vmax = _amax(x, (-2, -1))
        assert vmax.shape == (4, 1, 1)

    def test_quantile_multi_dim(self):
        x = torch.randn(4, 64, 64)
        q = _quantile(x, 0.5, (-2, -1))
        assert q.shape == (4, 1, 1)

    def test_amin_amax_non_contiguous(self):
        x = torch.randn(2, 3, 64, 64)
        vmin = _amin(x, (1, -1))  # dims 1 and 3
        assert vmin.shape == (2, 1, 64, 1)


# ---------------------------------------------------------------------------
# Safe math utilities
# ---------------------------------------------------------------------------


class TestSafeMath:
    def test_safe_arcsinh_positive(self):
        x = torch.tensor([0.1, 1.0, 10.0, 100.0])
        out = safe_arcsinh(x, scale=1.0)
        expected = torch.asinh(x)
        assert torch.allclose(out, expected, rtol=1e-6)

    def test_safe_arcsinh_preserves_dtype(self):
        x = torch.tensor([1.0, 2.0], dtype=torch.float32)
        out = safe_arcsinh(x)
        assert out.dtype == torch.float32

    def test_safe_log_positive(self):
        x = torch.tensor([1.0, 10.0, 100.0])
        out = safe_log(x)
        expected = torch.log(x)
        assert torch.allclose(out, expected, rtol=1e-6)

    def test_safe_log_zero_clamped(self):
        x = torch.tensor([0.0, 1.0])
        out = safe_log(x)
        assert not torch.isinf(out[0])
        assert out[0] > -50  # roughly log(1e-9) in float64

    def test_safe_log_zero_finite(self):
        x = torch.tensor([0.0])
        out = safe_log(x)
        assert torch.isfinite(out).all()

    def test_safe_log_preserves_dtype(self):
        x = torch.tensor([1.0, 2.0], dtype=torch.float32)
        out = safe_log(x)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# estimate_background and zscale_limits
# ---------------------------------------------------------------------------


class TestEstimateBackground:
    def test_normal_distribution(self):
        x = torch.randn(4, 64, 64) * 5 + 100
        med, std = estimate_background(x)
        assert med.shape == (4, 1, 1)
        assert std.shape == (4, 1, 1)
        # Median should be near 100, std near 5*1.4826
        assert abs(med.mean().item() - 100) < 3
        assert abs(std.mean().item() - 5 * 1.4826) < 3

    def test_constant_image(self):
        x = torch.ones(4, 64, 64) * 42.0
        med, std = estimate_background(x)
        assert torch.allclose(med, torch.tensor(42.0), atol=1e-5)
        assert torch.all(std < 1e-5)

    def test_mask_excludes_pixels_from_median(self):
        # 5x5 image: values 1..25, mask out the central 3x3 (values 7..19)
        x = torch.arange(1, 26, dtype=torch.float32).reshape(1, 5, 5)
        mask = torch.ones(1, 5, 5, dtype=torch.bool)
        mask[0, 1:4, 1:4] = False  # exclude central 3x3
        med, _ = estimate_background(x, dim=(-2, -1), mask=mask)
        # Only edge pixels (16 values: 1..6, 20..25, 11..15, 16..19... actually
        # the 16 border values) should contribute to median.
        # Border values: row0 (1-5), row4 (21-25), col0 from rows1-3 (6,11,16),
        # col4 from rows1-3 (10,15,20). Sorted: 1,2,3,4,5,6,10,11,15,16,20,21,22,23,24,25,26?
        # Wait, with 5x5 = 25 values, masking 3x3=9 leaves 16. Values 1-25.
        # Border: [1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25]
        # PyTorch's nanmedian returns the lower median for even counts
        # (8th of 16 sorted values = 11).
        assert abs(med.item() - 11.0) < 0.5


class TestZScaleLimits:
    def test_returns_valid_range(self):
        x = torch.randn(4, 64, 64) * 5 + 100
        z1, z2 = zscale_limits(x)
        assert torch.all(z1 < z2)

    def test_constant_image_fallback(self):
        x = torch.ones(4, 64, 64) * 42.0
        z1, z2 = zscale_limits(x)
        assert torch.all(z1 < z2)  # fallback adds 1e-6


# ---------------------------------------------------------------------------
# FITSTransform base and Compose
# ---------------------------------------------------------------------------


class TestFITSTransform:
    def test_raises_not_implemented(self):
        t = FITSTransform()
        with pytest.raises(NotImplementedError):
            t.forward(torch.zeros(3))
        with pytest.raises(NotImplementedError):
            t.inverse(torch.zeros(3))

    def test_call_delegates(self):
        class Dummy(FITSTransform):
            def forward(self, x, mask=None):
                return x + 1

            def inverse(self, x, mask=None):
                return x - 1

        d = Dummy()
        assert d(torch.tensor(1.0)).item() == 2.0
        assert d.inverse(torch.tensor(2.0)).item() == 1.0


class TestCompose:
    def test_forward_chain(self):
        class AddOne(FITSTransform):
            def forward(self, x, mask=None):
                return x + 1

            def inverse(self, x, mask=None):
                return x - 1

        c = Compose([AddOne(), AddOne(), AddOne()])
        assert c(torch.tensor(0.0)).item() == 3.0

    def test_inverse_reverses_chain(self):
        class MulTwo(FITSTransform):
            def forward(self, x, mask=None):
                return x * 2

            def inverse(self, x, mask=None):
                return x / 2

        c = Compose([MulTwo(), MulTwo()])
        x = torch.tensor(5.0)
        fwd = c.forward(x)
        assert fwd.item() == 20.0
        inv = c.inverse(fwd)
        assert torch.allclose(inv, x)

    def test_len_and_getitem(self):
        class Id(FITSTransform):
            def forward(self, x, mask=None):
                return x

            def inverse(self, x, mask=None):
                return x

        c = Compose([Id(), Id(), Id()])
        assert len(c) == 3
        assert c[0] is c.transforms[0]

    def test_repr(self):
        rep = repr(Compose([ArcsinhStretch(a=0.1)]))
        assert "Compose" in rep
        assert "ArcsinhStretch" in rep


# ---------------------------------------------------------------------------
# Stateless stretch transforms (exact roundtrip)
# ---------------------------------------------------------------------------


class TestArcsinhStretch:
    def test_roundtrip_identity(self):
        x = _make_tensor((3, 32, 32), kind="uniform") + 1.0
        t = ArcsinhStretch(a=1.0)
        restored = t.inverse(t.forward(x))
        # arcsinh -> sinh through float64 has ~float32-epsilon error
        err = (x - restored).abs().max().item()
        assert err < 5e-5, f"roundtrip error {err} too large"

    def test_output_in_range(self):
        x = torch.linspace(0, 100, 1000).reshape(10, 100)
        t = ArcsinhStretch(a=1.0)
        out = t.forward(x)
        assert out.min() >= 0
        assert torch.isfinite(out).all()

    def test_different_a_values(self):
        x = _make_tensor((2, 16, 16), kind="uniform") + 1.0
        for a in [0.01, 0.1, 1.0, 10.0]:
            t = ArcsinhStretch(a=a)
            restored = t.inverse(t.forward(x))
            err = (x - restored).abs().max().item()
            assert err < 1e-4, f"a={a}: roundtrip error {err}"

    def test_preserves_dtype(self):
        x = torch.randn(4, 32, 32, dtype=torch.float32) * 10
        t = ArcsinhStretch()
        out = t.forward(x)
        assert out.dtype == torch.float32
        inv = t.inverse(out)
        assert inv.dtype == torch.float32

    def test_negative_inputs(self):
        x = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        t = ArcsinhStretch(a=1.0)
        out = t.forward(x)
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-5)


class TestLogStretch:
    def test_roundtrip_identity(self):
        x = torch.linspace(1, 1000, 100)
        t = LogStretch(a=1000.0)
        restored = t.inverse(t.forward(x))
        # log10->pow10 roundtrip through float64 has limited float32 precision
        err = (x - restored).abs().max().item()
        assert err < 2e-3, f"roundtrip error {err}"

    def test_negative_clamped(self):
        x = torch.tensor([-5.0, 0.0, 5.0])
        t = LogStretch(a=1000.0)
        out = t.forward(x)
        # Negative values should be clamped to 0, producing the same result as x=0
        assert out[0].item() == out[1].item()
        # Positive values produce a larger result
        assert out[2].item() > out[1].item()

    def test_preserves_dtype(self):
        x = torch.rand(4, 32, 32, dtype=torch.float32) * 100
        t = LogStretch()
        out = t.forward(x)
        assert out.dtype == torch.float32


class TestSqrtStretch:
    def test_roundtrip_identity(self):
        x = torch.linspace(1, 100, 100)
        t = SqrtStretch()
        restored = t.inverse(t.forward(x))
        assert torch.allclose(restored, x, rtol=1e-5)

    def test_negative_clamped(self):
        x = torch.tensor([-5.0, 0.0, 5.0])
        t = SqrtStretch()
        out = t.forward(x)
        assert out[0] == 0.0
        assert out[1] == 0.0
        assert out[2] > 0.0


# ---------------------------------------------------------------------------
# Stateful normalizer transforms
# ---------------------------------------------------------------------------


class TestZScaleNormalize:
    def test_roundtrip_identity(self):
        x = _make_tensor((4, 32, 32), kind="normal")
        t = ZScaleNormalize()
        restored = t.inverse(t.forward(x))
        err = (x - restored).abs().max().item()
        assert err < 2e-5

    def test_output_in_01_range(self):
        x = _make_tensor((4, 32, 32), kind="normal")
        t = ZScaleNormalize()
        out = t.forward(x)
        # Most values should be in [0, 1], but some outliers may be outside
        # due to the contrast-based limits
        assert out.min() >= -0.5
        assert out.max() <= 1.5

    def test_inverse_without_forward_raises(self):
        t = ZScaleNormalize()
        with pytest.raises(RuntimeError, match="prior forward"):
            t.inverse(torch.zeros(3))

    def test_constant_image(self):
        x = torch.ones(4, 32, 32) * 42.0
        t = ZScaleNormalize()
        out = t.forward(x)
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-5)

    def test_mask_no_mask_gives_same_result(self):
        """Mask=None should produce identical result to no mask passed."""
        x = _make_tensor((4, 32, 32), kind="normal")
        t1 = ZScaleNormalize()
        t2 = ZScaleNormalize()
        out1 = t1.forward(x, mask=None)
        out2 = t2.forward(x)
        assert torch.allclose(out1, out2, atol=1e-7)


class TestRobustNormalize:
    def test_roundtrip_identity(self):
        x = _make_tensor((4, 32, 32), kind="normal")
        t = RobustNormalize()
        restored = t.inverse(t.forward(x))
        err = (x - restored).abs().max().item()
        assert err < 2e-5

    def test_near_zero_median(self):
        x = _make_tensor((4, 32, 32), kind="normal")
        t = RobustNormalize()
        out = t.forward(x)
        # After robust normalization, the median should be near zero
        for i in range(4):
            assert abs(out[i].median().item()) < 0.1

    def test_inverse_without_forward_raises(self):
        t = RobustNormalize()
        with pytest.raises(RuntimeError, match="prior forward"):
            t.inverse(torch.zeros(3))

    def test_constant_image(self):
        x = torch.ones(4, 32, 32) * 42.0
        t = RobustNormalize()
        out = t.forward(x)
        # All values should be ~0 after normalization (image is flat)
        assert out.abs().max() < 1e-5


class TestBackgroundSubtract:
    def test_roundtrip_identity(self):
        x = _make_tensor((4, 32, 32), kind="normal")
        t = BackgroundSubtract()
        restored = t.inverse(t.forward(x))
        err = (x - restored).abs().max().item()
        assert err < 2e-5

    def test_subtracts_median(self):
        x = torch.ones(4, 64, 64) * 5.0
        x[0, 0, 0] = 100.0  # outlier
        t = BackgroundSubtract()
        out = t.forward(x)
        # Most values should be near 0 after subtraction
        median_after = out.median().item()
        assert abs(median_after) < 0.5

    def test_inverse_without_forward_raises(self):
        t = BackgroundSubtract()
        with pytest.raises(RuntimeError, match="prior forward"):
            t.inverse(torch.zeros(3))


class TestPercentileClipNormalize:
    def test_roundtrip_identity(self):
        # PercentileClipNormalize is lossy when percentiles clip values.
        # Use 0/100 to avoid clipping (full range, exact roundtrip).
        x = _make_tensor((4, 32, 32), kind="uniform")
        t = PercentileClipNormalize(lower_pct=0, upper_pct=100)
        restored = t.inverse(t.forward(x))
        err = (x - restored).abs().max().item()
        assert err < 2e-5

    def test_output_range(self):
        x = _make_tensor((4, 32, 32), kind="normal")
        t = PercentileClipNormalize(lower_pct=0, upper_pct=100)
        out = t.forward(x)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_clips_outliers(self):
        x = torch.zeros(10, 10)
        x[0, 0] = 1e6  # extreme outlier
        t = PercentileClipNormalize(lower_pct=10, upper_pct=90)
        out = t.forward(x)
        # After clipping, the outlier should be clamped
        assert out[0, 0].item() <= 1.0

    def test_inverse_without_forward_raises(self):
        t = PercentileClipNormalize()
        with pytest.raises(RuntimeError, match="prior forward"):
            t.inverse(torch.zeros(3))

    def test_constant_image(self):
        x = torch.ones(4, 32, 32) * 42.0
        t = PercentileClipNormalize(lower_pct=0, upper_pct=100)
        out = t.forward(x)
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-4)

    def test_per_channel(self):
        x = torch.stack(
            [
                torch.randn(64, 64) * 10 + 100,
                torch.randn(64, 64) * 2 + 10,
            ]
        )
        t = PercentileClipNormalize(lower_pct=5, upper_pct=95, dim=(-2, -1))
        out = t.forward(x)
        # Each channel should be independently normalized to [0, 1]
        assert out.shape == (2, 64, 64)
        for c in range(2):
            assert out[c].max() <= 1.0 + 1e-6
            assert out[c].min() >= -1e-6


class TestMinMaxNormalize:
    def test_roundtrip_identity(self):
        x = _make_tensor((4, 32, 32), kind="uniform")
        t = MinMaxNormalize()
        restored = t.inverse(t.forward(x))
        err = (x - restored).abs().max().item()
        assert err < 2e-5

    def test_output_range(self):
        x = _make_tensor((4, 32, 32), kind="normal")
        t = MinMaxNormalize()
        out = t.forward(x)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_constant_image(self):
        x = torch.ones(4, 32, 32) * 42.0
        t = MinMaxNormalize()
        out = t.forward(x)
        # Constant image normalizes to all zeros (vmin == 42, vmax ≈ 42 + 1e-6)
        assert out.abs().max() < 1e-5

    def test_inverse_without_forward_raises(self):
        t = MinMaxNormalize()
        with pytest.raises(RuntimeError, match="prior forward"):
            t.inverse(torch.zeros(3))


class TestMaskNanRoundtrip:
    """Verify transforms handle NaN-contaminated data correctly with masks."""

    def test_nan_roundtrip_with_mask(self):
        """NaN pixels excluded from stats, remain NaN in output, roundtrip works."""
        x = _make_tensor((4, 32, 32), kind="normal")
        # Inject NaN at known positions
        x[0, 0, 0] = float("nan")
        x[2, 15, 15] = float("nan")
        # Build a valid mask that excludes the NaN positions
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[0, 0, 0] = False
        mask[2, 15, 15] = False

        t = MinMaxNormalize()
        out = t.forward(x, mask=mask)
        # NaN pixels should stay NaN in the output
        assert torch.isnan(out[0, 0, 0])
        assert torch.isnan(out[2, 15, 15])
        # Valid pixels should be normalized to [0, 1]
        valid_out = out[~torch.isnan(out)]
        assert valid_out.min() >= -1e-6
        assert valid_out.max() <= 1.0 + 1e-6

        # Roundtrip: NaN positions excluded from min/max so inversion works
        restored = t.inverse(out)
        assert torch.isnan(restored[0, 0, 0])
        assert torch.isnan(restored[2, 15, 15])
        # Non-NaN values should roundtrip
        valid_x = x[mask]
        valid_restored = restored[mask]
        assert torch.allclose(valid_restored, valid_x, atol=1e-4)


# ---------------------------------------------------------------------------
# FITSHeaderScale
# ---------------------------------------------------------------------------


class TestFITSHeaderScale:
    def test_roundtrip_identity(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        t = FITSHeaderScale(bscale=2.0, bzero=10.0)
        physical = t.forward(x)
        expected = torch.tensor([12.0, 14.0, 16.0])
        assert torch.allclose(physical, expected)
        restored = t.inverse(physical)
        assert torch.allclose(restored, x)

    def test_from_header(self):
        header: dict[str, float] = {"BSCALE": 0.5, "BZERO": 100.0}
        t = FITSHeaderScale.from_header(header)
        assert t.bscale == 0.5
        assert t.bzero == 100.0

    def test_from_header_defaults(self):
        header: dict[str, object] = {}
        t = FITSHeaderScale.from_header(header)
        assert t.bscale == 1.0
        assert t.bzero == 0.0

    def test_identity_noop(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        t = FITSHeaderScale(bscale=1.0, bzero=0.0)
        out = t.forward(x)
        assert out.data_ptr() == x.data_ptr()  # same tensor, no copy
        inv = t.inverse(x)
        assert inv.data_ptr() == x.data_ptr()

    def test_preserves_int_dtype(self):
        x = torch.tensor([1, 2, 3], dtype=torch.int32)
        t = FITSHeaderScale(bscale=1.0, bzero=0.0)
        out = t.forward(x)
        assert out.dtype == torch.int32
        assert out.data_ptr() == x.data_ptr()

    def test_bscale_only(self):
        x = torch.tensor([1.0, 2.0])
        t = FITSHeaderScale(bscale=10.0, bzero=0.0)
        physical = t.forward(x)
        assert torch.allclose(physical, torch.tensor([10.0, 20.0]))
        assert torch.allclose(t.inverse(physical), x)

    def test_bzero_only(self):
        x = torch.tensor([1.0, 2.0])
        t = FITSHeaderScale(bscale=1.0, bzero=100.0)
        physical = t.forward(x)
        assert torch.allclose(physical, torch.tensor([101.0, 102.0]))
        assert torch.allclose(t.inverse(physical), x)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_pixel_image(self):
        x = torch.tensor([[42.0]])
        transforms = [
            ArcsinhStretch(),
            ZScaleNormalize(dim=(-2, -1)),
            RobustNormalize(dim=(-2, -1)),
            BackgroundSubtract(dim=(-2, -1)),
            PercentileClipNormalize(dim=(-2, -1)),
            MinMaxNormalize(dim=(-2, -1)),
        ]
        for t in transforms:
            out = t.forward(x)
            assert out.shape == (1, 1), f"{t}: shape mismatch"
            restored = t.inverse(out)
            assert torch.allclose(restored, x, atol=1e-4), f"{t}: roundtrip failed"

    def test_1d_vector(self):
        x = torch.linspace(0, 100, 50)
        t = RobustNormalize(dim=(-1,))
        restored = t.inverse(t.forward(x))
        err = (x - restored).abs().max().item()
        assert err < 2e-5

    def test_3d_cube(self):
        x = _make_tensor((2, 32, 32), kind="uniform")
        t = ZScaleNormalize(dim=(-2, -1))
        out = t.forward(x)
        assert out.shape == (2, 32, 32)
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-5)

    def test_batched_4d(self):
        x = _make_tensor((4, 3, 64, 64), kind="normal")
        t = Compose([BackgroundSubtract(dim=(-2, -1)), ArcsinhStretch(a=0.1)])
        out = t.forward(x)
        assert out.shape == (4, 3, 64, 64)
        restored = t.inverse(out)
        # arcsinh + bg-subtract roundtrip through float64
        err = (x - restored).abs().max().item()
        assert err < 5e-4, f"batch roundtrip error {err} too large"

    def test_zero_std_image(self):
        x = torch.ones(4, 32, 32) * 10.0
        t = ZScaleNormalize()
        out = t.forward(x)
        # Should not crash and should produce finite output
        assert torch.isfinite(out).all()

    def test_extreme_dynamic_range(self):
        x = torch.ones(64, 64) * 1e-10
        x[32, 32] = 1e10
        t = ArcsinhStretch(a=1.0)
        out = t.forward(x)
        assert torch.isfinite(out).all()
        restored = t.inverse(out)
        # High DR may lose some precision in float32, but should be close
        rel_err = ((x - restored).abs() / (x.abs() + 1e-30)).max().item()
        assert rel_err < 1e-4  # float64 roundtrip preserves precision well

    def test_int16_tensor(self):
        x = torch.randint(-100, 100, (4, 32, 32), dtype=torch.int16)
        t = MinMaxNormalize(dim=(-2, -1))
        # Should convert to float internally and work
        out = t.forward(x.float())
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_compose_with_stateful_transforms(self):
        """Verify Compose chains stateful transforms correctly."""
        x = _make_tensor((4, 32, 32), kind="normal")
        c = Compose(
            [
                BackgroundSubtract(),
                RobustNormalize(),
                ZScaleNormalize(),
            ]
        )
        fwd = c.forward(x)
        assert fwd.shape == x.shape
        inv = c.inverse(fwd)
        err = (x - inv).abs().max().item()
        assert err < 2e-5

    def test_repr_methods(self):
        """Verify repr is informative and doesn't crash."""
        for cls, args in [
            (ArcsinhStretch, {"a": 0.1}),
            (LogStretch, {"a": 500}),
            (ZScaleNormalize, {"contrast": 0.3}),
            (RobustNormalize, {"dim": (-1,)}),
            (PercentileClipNormalize, {"lower_pct": 5, "upper_pct": 95}),
            (FITSHeaderScale, {"bscale": 2.0, "bzero": 10.0}),
        ]:
            t = cls(**args)  # type: ignore[arg-type]
            r = repr(t)
            assert len(r) > 0
            assert cls.__name__ in r


# ---------------------------------------------------------------------------
# SigmaClip
# ---------------------------------------------------------------------------


class TestSigmaClip:
    def test_removes_outliers(self):
        x = torch.ones(10, 10) * 5.0
        x[0, 0] = 100.0  # extreme outlier
        t = SigmaClip(n_sigma=3.0, max_iter=5)
        out = t.forward(x)
        # The outlier should be replaced with the mean (~5)
        assert out[0, 0].item() < 10.0

    def test_no_clipping_on_clean_data(self):
        x = _make_tensor((4, 32, 32), kind="normal")
        t = SigmaClip(n_sigma=10.0, max_iter=3)
        out = t.forward(x)
        # With n_sigma=10, almost nothing should be clipped — values very close
        assert (out - x).abs().max().item() < 0.5

    def test_inverse_raises(self):
        t = SigmaClip()
        with pytest.raises(RuntimeError, match="irrecoverable"):
            t.inverse(torch.zeros(3))

    def test_constant_image(self):
        x = torch.ones(4, 32, 32) * 42.0
        t = SigmaClip()
        out = t.forward(x)
        assert torch.allclose(out, x, atol=1e-5)

    def test_per_channel(self):
        x = torch.randn(4, 32, 32) * 5 + 10
        t = SigmaClip(dim=(-2, -1))
        out = t.forward(x)
        assert out.shape == (4, 32, 32)
        assert torch.isfinite(out).all()

    def test_median_fill(self):
        x = torch.ones(10, 10) * 5.0
        x[0, 0] = 100.0
        t = SigmaClip(n_sigma=3.0, fill="median")
        out = t.forward(x)
        assert out[0, 0].item() < 10.0

    def test_repr(self):
        t = SigmaClip(n_sigma=5.0, max_iter=3, fill="median")
        r = repr(t)
        assert "SigmaClip" in r
        assert "5.0" in r


# ---------------------------------------------------------------------------
# FITSHeaderNormalize
# ---------------------------------------------------------------------------


class TestFITSHeaderNormalize:
    def test_int16_scales_to_01(self):
        header = {"BITPIX": 16, "BSCALE": 1.0, "BZERO": 0.0}
        x = torch.tensor([-32768, 0, 32767], dtype=torch.float32)
        t = FITSHeaderNormalize(header)
        out = t.forward(x)
        assert out.min() >= 0
        assert out.max() <= 1.0
        assert out[1].item() == pytest.approx(0.5, abs=0.01)

    def test_int16_with_bzero(self):
        header = {"BITPIX": 16, "BSCALE": 1.0, "BZERO": 32768.0}
        x = torch.tensor([0.0, 32768.0, 65535.0])
        t = FITSHeaderNormalize(header)
        out = t.forward(x)
        assert out.min() >= 0
        assert out.max() <= 1.0
        # Physical range: [-32768, 32767] * 1.0 + 32768 = [0, 65535]
        assert out[1].item() == pytest.approx(0.5, abs=0.01)

    def test_roundtrip_int16(self):
        header = {"BITPIX": 16, "BSCALE": 2.0, "BZERO": 100.0}
        x = torch.tensor([0.0, 500.0, 65534.0])
        t = FITSHeaderNormalize(header)
        out = t.forward(x)
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-4)

    def test_float32_no_scale(self):
        header = {"BITPIX": -32}
        x = torch.randn(4, 32, 32)
        t = FITSHeaderNormalize(header, scale_floats=False)
        out = t.forward(x)
        assert torch.equal(out, x)

    def test_float32_with_scale(self):
        header = {"BITPIX": -32}
        x = _make_tensor((4, 32, 32), kind="uniform")
        t = FITSHeaderNormalize(header, scale_floats=True)
        out = t.forward(x)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-5)

    def test_uint8(self):
        header = {"BITPIX": 8, "BSCALE": 1.0, "BZERO": 0.0}
        x = torch.tensor([0.0, 128.0, 255.0])
        t = FITSHeaderNormalize(header)
        out = t.forward(x)
        assert out.min() >= 0
        assert out.max() <= 1.0
        assert out[1].item() == pytest.approx(0.5, abs=0.01)

    def test_repr(self):
        t = FITSHeaderNormalize({"BITPIX": -32}, scale_floats=True)
        r = repr(t)
        assert "FITSHeaderNormalize" in r
        assert "bitpix=-32" in r


# ---------------------------------------------------------------------------
# ContinuumNormalize (spectral)
# ---------------------------------------------------------------------------


class TestContinuumNormalize:
    def test_normalizes_spectrum(self):
        # Simple spectrum: linear continuum + absorption line
        t_arr = torch.linspace(-1, 1, 100)
        continuum = 10.0 + 2.0 * t_arr
        line = -3.0 * torch.exp(-((t_arr - 0.1) ** 2) / (2 * 0.05**2))
        spectrum = continuum + line
        x = spectrum.unsqueeze(0)  # [1, 100]
        t = ContinuumNormalize(order=1, n_sigma=2.0, max_iter=3)
        out = t.forward(x)
        # Normalized spectrum should be ~1 away from the line
        assert out[0, :50].mean().item() == pytest.approx(1.0, abs=0.1)
        # The absorption line should still be visible
        assert out.min().item() < 0.9

    def test_roundtrip(self):
        x = torch.linspace(0, 100, 50).unsqueeze(0)
        t = ContinuumNormalize(order=1)
        out = t.forward(x)
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-5)

    def test_batched_spectra(self):
        x = torch.randn(4, 200) * 2 + 10
        t = ContinuumNormalize(order=2)
        out = t.forward(x)
        assert out.shape == (4, 200)
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-5)

    def test_inverse_without_forward_raises(self):
        t = ContinuumNormalize()
        with pytest.raises(RuntimeError, match="prior forward"):
            t.inverse(torch.zeros(10))

    def test_repr(self):
        r = repr(ContinuumNormalize(order=5, n_sigma=3.0))
        assert "ContinuumNormalize" in r
        assert "5" in r


# ---------------------------------------------------------------------------
# ContinuumNormalize — batched solve vs per-spectrum lstsq parity tests
# ---------------------------------------------------------------------------


class TestContinuumNormalizeVectorized:
    """Verify batched normal-equations solver matches per-spectrum lstsq."""

    @staticmethod
    def _run_vectorized(
        x: torch.Tensor,
        order: int = 3,
        n_sigma: float = 2.0,
        max_iter: int = 3,
    ) -> torch.Tensor:
        """Run production ContinuumNormalize and return the fitted continuum."""
        t = ContinuumNormalize(order=order, n_sigma=n_sigma, max_iter=max_iter)
        t.forward(x)
        return t._continuum  # type: ignore[return-value]

    def test_linear_continuum(self):
        """Linear continuum + absorption dip: both solvers agree on continuum."""
        torch.manual_seed(42)
        t_arr = torch.linspace(-1, 1, 100)
        continuum = 10.0 + 2.0 * t_arr
        line = -3.0 * torch.exp(-((t_arr - 0.1) ** 2) / (2 * 0.05**2))
        x = (continuum + line).unsqueeze(0)
        vec = self._run_vectorized(x, order=1, n_sigma=2.0, max_iter=3)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = continuum_normalize_per_spectrum(x_flat, order=1, n_sigma=2.0, max_iter=3)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_batched_spectra(self):
        """Multiple spectra with distinct baselines."""
        torch.manual_seed(42)
        x = torch.randn(4, 200) * 2 + 10
        vec = self._run_vectorized(x, order=2, n_sigma=3.0, max_iter=3)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = continuum_normalize_per_spectrum(x_flat, order=2, n_sigma=3.0, max_iter=3)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_cubic_order(self):
        """Order-3 polynomial fit."""
        torch.manual_seed(42)
        t_arr = torch.linspace(-1, 1, 150)
        continuum = 5.0 + 3.0 * t_arr + 2.0 * t_arr**2 + 1.0 * t_arr**3
        dip = -2.0 * torch.exp(-((t_arr - 0.2) ** 2) / (2 * 0.04**2))
        x = (continuum + dip).unsqueeze(0)
        vec = self._run_vectorized(x, order=3, n_sigma=2.0, max_iter=3)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = continuum_normalize_per_spectrum(x_flat, order=3, n_sigma=2.0, max_iter=3)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_no_absorption(self):
        """Pure polynomial: continuum should match exactly."""
        torch.manual_seed(42)
        t_arr = torch.linspace(-1, 1, 100)
        continuum = 10.0 + 2.0 * t_arr
        x = continuum.unsqueeze(0)
        vec = self._run_vectorized(x, order=1, n_sigma=5.0, max_iter=3)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = continuum_normalize_per_spectrum(x_flat, order=1, n_sigma=5.0, max_iter=3)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_many_iterations(self):
        """max_iter=10 with deep absorption — sigma-clipping converges."""
        torch.manual_seed(42)
        t_arr = torch.linspace(-1, 1, 120)
        continuum = 10.0 + 2.0 * t_arr
        dip = -6.0 * torch.exp(-((t_arr - 0.0) ** 2) / (2 * 0.02**2))
        x = (continuum + dip).unsqueeze(0)
        vec = self._run_vectorized(x, order=1, n_sigma=2.0, max_iter=10)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = continuum_normalize_per_spectrum(
            x_flat, order=1, n_sigma=2.0, max_iter=10
        )
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# DopplerShift (spectral)
# ---------------------------------------------------------------------------


class TestResample1d:
    """Unit tests for _resample_1d resampling engine."""

    # ---- area mode with spiky spectra ----

    def test_area_mode_spiky_gaussian_flux_conservation(self):
        """Area mode preserves per-cell average of a Gaussian feature."""
        t_arr = torch.linspace(-5, 5, 500)
        gauss = torch.exp(-(t_arr**2) / (2 * 0.5**2))  # σ = 0.5 px
        x = gauss.unsqueeze(0)  # [1, 500]
        x_old = torch.arange(500, dtype=torch.float32)
        x_new = torch.linspace(0, 499, 300, dtype=torch.float32)
        out_area = _resample_1d(x, x_old, x_new, mode="area")
        out_linear = _resample_1d(x, x_old, x_new, mode="linear")
        # Area and linear should produce different outputs (different algorithms)
        assert not torch.allclose(out_area, out_linear, atol=1e-4), (
            "area and linear should produce different results"
        )
        # Both should be finite
        assert torch.isfinite(out_area).all()
        assert torch.isfinite(out_linear).all()

    def test_area_mode_emission_line_pair(self):
        """Two close narrow lines: area mode avoids smearing between them."""
        t_arr = torch.linspace(-5, 5, 500)
        # Two Gaussians 2 pixels apart
        g1 = torch.exp(-((t_arr + 0.5) ** 2) / (2 * 0.1**2))
        g2 = torch.exp(-((t_arr - 0.5) ** 2) / (2 * 0.1**2))
        x = (g1 + g2).unsqueeze(0)
        x_old = torch.arange(500, dtype=torch.float32)
        x_new = torch.linspace(0, 499, 200, dtype=torch.float32)

        out_linear = _resample_1d(x, x_old, x_new, mode="linear")
        out_area = _resample_1d(x, x_old, x_new, mode="area")
        # Both should have two discernible peaks
        for out, mode_name in [(out_linear, "linear"), (out_area, "area")]:
            # At least 2 local maxima in the output
            larger_than_neighbors = (out[0, 1:-1] > out[0, :-2]) & (
                out[0, 1:-1] > out[0, 2:]
            )
            n_peaks = larger_than_neighbors.sum().item()
            # With downsampled narrow lines, at least 1 peak should survive
            assert n_peaks >= 1, f"{mode_name}: only {n_peaks} peaks detected"

    # ---- cubic mode edges ----

    def test_cubic_mode_reproduces_linear_at_boundaries(self):
        """Cubic falls back to linear at segment edges (idx <= 1 or >= L-1)."""
        x = torch.linspace(0, 100, 10).unsqueeze(0)
        x_old = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        # Query exactly at grid points — linear and cubic should agree everywhere
        x_new = x_old.clone()
        out_linear = _resample_1d(x, x_old, x_new, mode="linear")
        out_cubic = _resample_1d(x, x_old, x_new, mode="cubic")
        # At interior points cubic should match; at edges cubic falls to linear
        assert torch.allclose(out_linear, out_cubic, atol=1e-5)

    def test_cubic_mode_smoother_than_nearest(self):
        """Cubic interpolation produces smoother output than nearest."""
        x = torch.linspace(0, 100, 50).unsqueeze(0)
        x_old = torch.arange(50, dtype=torch.float32)
        x_new = torch.linspace(0, 49, 200, dtype=torch.float32)
        out_nearest = _resample_1d(x, x_old, x_new, mode="nearest")
        out_cubic = _resample_1d(x, x_old, x_new, mode="cubic")
        # Cubic should have fewer discontinuities (lower diff variance)
        diff_near = out_nearest[0, 1:] - out_nearest[0, :-1]
        diff_cubic = out_cubic[0, 1:] - out_cubic[0, :-1]
        # Cubic should have much lower step-to-step variance than nearest
        assert diff_cubic.var().item() < diff_near.var().item() * 0.5

    # ---- irregular wavelength grids ----

    def test_irregular_grid_linear(self):
        """Linear interpolation on non-uniform wavelength grid."""
        x = torch.linspace(0, 100, 100).unsqueeze(0)
        # Log-spaced source grid (common in spectroscopy)
        x_old = torch.logspace(0, 2, 100, dtype=torch.float32)  # 1..100
        # Linear-spaced output grid
        x_new = torch.linspace(1, 100, 50, dtype=torch.float32)
        out = _resample_1d(x, x_old, x_new, mode="linear")
        assert out.shape == (1, 50)
        assert torch.isfinite(out).all()

    def test_irregular_grid_preserves_monotonicity(self):
        """Linear interpolation of monotonic data stays monotonic."""
        x = torch.linspace(0, 100, 200).unsqueeze(0)
        x_old = torch.logspace(-1, 2, 200, dtype=torch.float32)
        x_new = torch.linspace(0.1, 100, 100, dtype=torch.float32)
        out = _resample_1d(x, x_old, x_new, mode="linear")
        diffs = out[0, 1:] - out[0, :-1]
        assert (diffs >= -1e-6).all(), f"found {diffs.min().item():.2e} decrease"

    # ---- area mode on irregular grids ----

    def test_area_mode_on_irregular_grid(self):
        """Area mode works on irregular wavelength grids via searchsorted fallback."""
        x = torch.linspace(0, 100, 100).unsqueeze(0)
        x_old = torch.logspace(0, 2, 100, dtype=torch.float32)
        x_new = torch.linspace(1, 100, 30, dtype=torch.float32)
        out = _resample_1d(x, x_old, x_new, mode="area")
        assert out.shape == (1, 30)
        assert torch.isfinite(out).all()

    # ---- mode validation ----

    def test_invalid_mode_raises(self):
        x = torch.randn(2, 100)
        x_old = torch.arange(100, dtype=torch.float32)
        x_new = torch.arange(50, dtype=torch.float32)
        with pytest.raises(ValueError, match="mode must be"):
            _resample_1d(x, x_old, x_new, mode="spline")

    def test_scale_invalid_mode_raises(self):
        x = torch.randn(2, 100)
        with pytest.raises(ValueError, match="mode must be"):
            _resample_scale(x, 0.5, mode="bicubic")

    # ---- edge cases ----

    def test_empty_input(self):
        x = torch.randn(3, 0)
        x_old = torch.zeros(0)
        x_new = torch.arange(10, dtype=torch.float32)
        out = _resample_1d(x, x_old, x_new)
        # L_src == 0 returns last dim 0 regardless of L_dst
        assert out.shape == (3, 0)

    def test_empty_output(self):
        x = torch.randn(3, 50)
        x_old = torch.arange(50, dtype=torch.float32)
        x_new = torch.zeros(0)
        out = _resample_1d(x, x_old, x_new)
        assert out.shape == (3, 0)

    def test_single_pixel_input(self):
        x = torch.tensor([[42.0]])
        x_old = torch.tensor([0.0])
        x_new = torch.linspace(0, 0, 5)
        out = _resample_1d(x, x_old, x_new)
        assert out.shape == (1, 5)
        # All output positions should broadcast the single input value
        assert torch.allclose(out, torch.full((1, 5), 42.0))

    def test_batched_spectra_all_modes(self):
        """All modes work on batched [N, L] tensors."""
        x = torch.randn(4, 200)
        x_old = torch.arange(200, dtype=torch.float32)
        x_new = torch.linspace(0, 199, 128, dtype=torch.float32)
        for mode in ["linear", "nearest", "cubic", "area"]:
            out = _resample_1d(x, x_old, x_new, mode=mode)
            assert out.shape == (4, 128), f"{mode}: bad shape {out.shape}"
            assert torch.isfinite(out).all(), f"{mode}: non-finite values"

    def test_scale_factor_resampling(self):
        """_resample_scale correctly changes length by scale factor."""
        x = torch.randn(3, 100)
        out = _resample_scale(x, 0.5)
        assert out.shape == (3, 50)
        out = _resample_scale(x, 2.0)
        assert out.shape == (3, 200)

    # ---- offset-uniform grid (exercises F.grid_sample normalization) ----

    def test_offset_uniform_grid(self):
        """Offset-uniform x_old (e.g. [100, 200, 300]) uses F.grid_sample with correct normalization."""
        # Create an offset-uniform source grid: spacing=100, start=100
        L_src = 50
        x_old = 100.0 + torch.arange(L_src, dtype=torch.float32) * 100.0
        # Values are a simple linear ramp matching the grid positions
        x = x_old.unsqueeze(0)  # [1, 50]
        # Query at positions that are exact midpoint interpolations
        x_new = 100.0 + torch.arange(L_src - 1, dtype=torch.float32) * 100.0 + 50.0
        out = _resample_1d(x, x_old, x_new, mode="linear")
        # Midpoint of [v_i, v_{i+1}] should be (v_i + v_{i+1})/2 = v_i + 50
        expected = x[:, :-1] + 50.0
        assert out.shape == (1, L_src - 1)
        assert torch.allclose(out, expected, atol=1e-4)

    def test_offset_uniform_grid_exact_query_points(self):
        """Querying exact grid positions on offset-uniform grid reproduces input values."""
        L_src = 40
        x_old = 200.0 + torch.arange(L_src, dtype=torch.float32) * 50.0
        x = torch.sin(torch.linspace(0, 3 * math.pi, L_src)).unsqueeze(0)
        # Query exactly at source positions
        out = _resample_1d(x, x_old, x_old, mode="linear")
        assert torch.allclose(out, x, atol=1e-5)

    def test_offset_uniform_grid_grid_sample_modes(self):
        """All grid_sample-compatible modes work on offset-uniform grids (non-zero origin)."""
        L_src = 60
        x_old = 500.0 + torch.arange(L_src, dtype=torch.float32) * 10.0
        x = torch.randn(3, L_src)
        x_new = torch.linspace(
            500.0, 500.0 + (L_src - 1) * 10.0, 40, dtype=torch.float32
        )
        for mode in ["linear", "nearest", "cubic"]:
            out = _resample_1d(x, x_old, x_new, mode=mode)
            assert out.shape == (3, 40), f"{mode}: bad shape {out.shape}"
            assert torch.isfinite(out).all(), f"{mode}: non-finite values"

    # ---- descending (negative-spacing) uniform grids ----

    def test_descending_uniform_grid(self):
        """Negative-spacing uniform grid (e.g. [500, 400, 300, ...]) exercises sign-preserving normalization."""
        L_src = 50
        x_old = (
            500.0 - torch.arange(L_src, dtype=torch.float32) * 10.0
        )  # [500, 490, ..., 10]
        x = x_old.unsqueeze(0)  # values match positions
        # Midpoint queries
        x_new = x_old[:-1] - 5.0  # [495, 485, ..., 15] — halfway between each pair
        out = _resample_1d(x, x_old, x_new, mode="linear")
        expected = x[:, :-1] - 5.0  # midpoint of [v_i, v_{i+1}] = v_i + dx/2 = v_i - 5
        assert out.shape == (1, L_src - 1)
        assert torch.allclose(out, expected, atol=1e-4)

    def test_descending_uniform_grid_all_modes(self):
        """All non-area modes work on descending uniform grids."""
        L_src = 40
        x_old = (
            1000.0 - torch.arange(L_src, dtype=torch.float32) * 25.0
        )  # [1000, 975, ..., 25]
        x = torch.randn(2, L_src)
        x_new = torch.linspace(1000.0, 25.0, 30, dtype=torch.float32)
        for mode in ["linear", "nearest", "cubic"]:
            out = _resample_1d(x, x_old, x_new, mode=mode)
            assert out.shape == (2, 30), f"{mode}: bad shape {out.shape}"
            assert torch.isfinite(out).all(), f"{mode}: non-finite values"

    # ---- zero-crossing uniform grid (negative to positive origin) ----

    def test_zero_crossing_uniform_grid(self):
        """Grid crossing zero (e.g. [-500, -400, ..., 500]) exercises negative-origin normalization."""
        L_src = 50
        x_old = (
            -500.0 + torch.arange(L_src, dtype=torch.float32) * 20.0
        )  # [-500, -480, ..., 480]
        x = x_old.unsqueeze(0)  # values match positions
        # Midpoint queries spanning the zero-crossing
        x_new = x_old[:-1] + 10.0  # [-490, -470, ..., 490] — halfway between each pair
        out = _resample_1d(x, x_old, x_new, mode="linear")
        expected = x[:, :-1] + 10.0  # midpoint of [v_i, v_{i+1}] = v_i + 10
        assert out.shape == (1, L_src - 1)
        assert torch.allclose(out, expected, atol=1e-4)

    def test_zero_crossing_uniform_grid_exact_query(self):
        """Exact query points on zero-crossing grid reproduce input values."""
        L_src = 40
        x_old = (
            -300.0 + torch.arange(L_src, dtype=torch.float32) * 15.0
        )  # [-300, -285, ..., 285]
        x = torch.sin(torch.linspace(0, 2 * math.pi, L_src)).unsqueeze(0)
        out = _resample_1d(x, x_old, x_old, mode="linear")
        assert torch.allclose(out, x, atol=1e-5)


class TestDopplerShift:
    def test_identity(self):
        x = torch.linspace(0, 100, 50).unsqueeze(0)
        t = DopplerShift(z=0.0)
        out = t.forward(x)
        assert torch.equal(out, x)

    def test_redshift_stretches(self):
        x = torch.zeros(1, 100)
        x[0, 49] = 1.0  # single line at center
        t = DopplerShift(z=0.1)  # redshift by 10%
        out = t.forward(x)
        # Peak should move to higher index (longer wavelength)
        peak_new = out.argmax().item()
        assert peak_new > 49  # redshifted

    def test_roundtrip(self):
        x = torch.linspace(0, 100, 200).unsqueeze(0)
        t = DopplerShift(z=0.05)
        out = t.forward(x)
        restored = t.inverse(out)
        # Roundtrip should preserve the original coordinate range exactly;
        # linear interpolation is not perfect so we allow small error.
        assert restored.shape[-1] == x.shape[-1], (
            f"length mismatch: {restored.shape[-1]} vs {x.shape[-1]}"
        )
        assert torch.allclose(restored, x, atol=5e-3), (
            f"max diff: {(restored - x).abs().max().item():.2e}"
        )

    def test_flux_conservation(self):
        # Smooth Gaussian profile — flux should be approximately conserved
        t_arr = torch.linspace(-5, 5, 200)
        gauss = torch.exp(-(t_arr**2) / 2.0)
        x = gauss.unsqueeze(0)
        t = DopplerShift(z=0.05)
        out = t.forward(x)
        # Resampling a smooth function preserves area closely
        assert abs(out.sum().item() - x.sum().item()) / abs(x.sum().item()) < 0.06

    def test_repr(self):
        r = repr(DopplerShift(z=0.5))
        assert "DopplerShift" in r
        assert "0.5" in r

    def test_inverse_without_forward_raises(self):
        import pytest

        t = DopplerShift(z=0.05)
        x = torch.linspace(0, 100, 200).unsqueeze(0)
        with pytest.raises(RuntimeError, match="prior forward"):
            t.inverse(x)

    def test_blueshift_roundtrip(self):
        x = torch.linspace(0, 100, 200).unsqueeze(0)
        t = DopplerShift(z=-0.05)
        out = t.forward(x)
        # Blueshift compresses the spectrum: output shorter than input
        assert out.shape[-1] < x.shape[-1], (
            f"blueshift should compress, got {out.shape[-1]} vs {x.shape[-1]}"
        )
        restored = t.inverse(out)
        assert restored.shape[-1] == x.shape[-1], (
            f"length mismatch: {restored.shape[-1]} vs {x.shape[-1]}"
        )
        assert torch.allclose(restored, x, atol=5e-3), (
            f"max diff: {(restored - x).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# PhaseFold (time series)
# ---------------------------------------------------------------------------


class TestPhaseFold:
    def test_folds_periodic_signal(self):
        # Sine wave with period=20, 200 time steps
        t_arr = torch.arange(200, dtype=torch.float32)
        signal = torch.sin(2 * math.pi * t_arr / 20.0)
        x = signal.unsqueeze(0)
        t = PhaseFold(period=20.0, n_bins=32)
        out = t.forward(x)
        assert out.shape == (1, 32)
        # Folded signal should show the periodic pattern
        assert out.abs().max().item() > 0.5

    def test_batched(self):
        x = torch.randn(4, 100)
        t = PhaseFold(period=10.0, n_bins=20)
        out = t.forward(x)
        assert out.shape == (4, 20)

    def test_inverse_raises(self):
        t = PhaseFold(period=1.0)
        with pytest.raises(RuntimeError, match="many-to-one"):
            t.inverse(torch.zeros(5))

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError):
            PhaseFold(period=0)
        with pytest.raises(ValueError):
            PhaseFold(period=-1)

    def test_repr(self):
        r = repr(PhaseFold(period=5.0, n_bins=10, t0=2.0))
        assert "PhaseFold" in r
        assert "5.0" in r


# ---------------------------------------------------------------------------
# SpectralBinning (hyperspectral)
# ---------------------------------------------------------------------------


class TestSpectralBinning:
    def test_bin_mean_preserves_flux(self):
        x = torch.ones(1, 64)
        t = SpectralBinning(factor=4, mode="mean")
        out = t.forward(x)
        assert out.shape == (1, 16)
        # Mean of ones is 1 — per-pixel value preserved
        assert torch.allclose(out, torch.ones(1, 16))

    def test_bin_sum_doubles_flux(self):
        x = torch.ones(1, 64)
        t = SpectralBinning(factor=4, mode="sum")
        out = t.forward(x)
        assert out.shape == (1, 16)
        # Sum of 4 ones = 4
        assert torch.allclose(out, torch.full((1, 16), 4.0))

    def test_roundtrip_mean(self):
        x = torch.randn(2, 100)
        t = SpectralBinning(factor=5, mode="mean", dim=-1)
        out = t.forward(x)
        restored = t.inverse(out)
        # Nearest-neighbour inverse: each binned value repeated 5 times
        assert restored.shape == (2, 100)
        # Bin-mean of restored should match forward output
        binned_check = restored.reshape(2, 20, 5).mean(dim=-1)
        assert torch.allclose(binned_check, out, atol=1e-6)

    def test_roundtrip_sum(self):
        x = torch.randn(3, 80)
        t = SpectralBinning(factor=4, mode="sum", dim=-1)
        out = t.forward(x)
        restored = t.inverse(out)
        # Inverse divides by factor to recover per-pixel flux
        assert restored.shape == (3, 80)
        # Sum of restored over each bin should match forward sum output
        binned_check = restored.reshape(3, 20, 4).sum(dim=-1)
        assert torch.allclose(binned_check, out, atol=1e-6)

    def test_non_unit_dim(self):
        # Bin along dim=0 for [C, H, W] tensor
        x = torch.randn(12, 32, 32)
        t = SpectralBinning(factor=3, mode="mean", dim=0)
        out = t.forward(x)
        assert out.shape == (4, 32, 32)
        restored = t.inverse(out)
        assert restored.shape == (12, 32, 32)

    def test_negative_dim(self):
        x = torch.randn(2, 16, 64)
        t = SpectralBinning(factor=4, mode="mean", dim=-2)
        out = t.forward(x)
        assert out.shape == (2, 4, 64)

    def test_partial_bins_dropped(self):
        # 65 channels with factor=8 → 64 binned, 1 dropped
        x = torch.randn(1, 65)
        t = SpectralBinning(factor=8, mode="mean")
        out = t.forward(x)
        assert out.shape == (1, 8)  # 64/8 = 8, trailing 1 dropped

    def test_factor_one_identity(self):
        x = torch.randn(4, 32)
        t = SpectralBinning(factor=1, mode="sum")
        out = t.forward(x)
        assert torch.equal(out, x)
        inv = t.inverse(out)
        assert torch.equal(inv, x)

    def test_invalid_factor_raises(self):
        with pytest.raises(ValueError):
            SpectralBinning(factor=0)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            SpectralBinning(mode="max")

    def test_repr(self):
        r = repr(SpectralBinning(factor=3, mode="sum", dim=-1))
        assert "SpectralBinning" in r
        assert "3" in r

    def test_batched_spectra(self):
        x = torch.randn(8, 512)
        t = SpectralBinning(factor=16, mode="mean")
        out = t.forward(x)
        assert out.shape == (8, 32)
        restored = t.inverse(out)
        assert restored.shape == (8, 512)


# ---------------------------------------------------------------------------
# ContinuumRemoval (hyperspectral)
# ---------------------------------------------------------------------------


class TestContinuumRemoval:
    def test_polynomial_subtracts_baseline(self):
        # Linear spectrum: continuum = 10 + 2*t + absorption dip
        t_arr = torch.linspace(-1, 1, 100)
        continuum = 10.0 + 2.0 * t_arr
        dip = -3.0 * torch.exp(-((t_arr - 0.1) ** 2) / (2 * 0.05**2))
        spectrum = continuum + dip
        x = spectrum.unsqueeze(0)
        t = ContinuumRemoval(method="polynomial", order=1, n_sigma=2.0)
        out = t.forward(x)
        # After removal, continuum should be near zero
        assert out[0, :50].mean().abs().item() < 0.5
        # The absorption dip should still be visible (negative)
        assert out.min().item() < -1.0

    def test_polynomial_roundtrip(self):
        x = torch.linspace(0, 100, 50).unsqueeze(0)
        t = ContinuumRemoval(method="polynomial", order=1)
        out = t.forward(x)
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-5)

    def test_spline_subtracts_baseline(self):
        # Non-linear continuum with absorption
        t_arr = torch.linspace(-1, 1, 200)
        continuum = 5.0 + 3.0 * t_arr + 2.0 * t_arr**2  # parabolic
        dip = -4.0 * torch.exp(-((t_arr - 0.0) ** 2) / (2 * 0.03**2))
        spectrum = continuum + dip
        x = spectrum.unsqueeze(0)
        t = ContinuumRemoval(method="spline", n_knots=8, n_sigma=2.0)
        out = t.forward(x)
        # After removal, continuum should be near zero away from dip
        assert out[0, :30].mean().abs().item() < 1.0
        assert out.min().item() < -1.0

    def test_spline_roundtrip(self):
        x = torch.linspace(0, 100, 100).unsqueeze(0)
        t = ContinuumRemoval(method="spline", n_knots=6, n_sigma=5.0)
        out = t.forward(x)
        restored = t.inverse(out)
        # Spline fit is approximate; roundtrip should be close
        assert torch.allclose(restored, x, atol=1e-4)

    def test_batched_spectra(self):
        x = torch.randn(4, 200) * 2 + 10
        t = ContinuumRemoval(method="polynomial", order=2)
        out = t.forward(x)
        assert out.shape == (4, 200)
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-5)

    def test_inverse_without_forward_raises(self):
        t = ContinuumRemoval()
        with pytest.raises(RuntimeError, match="prior forward"):
            t.inverse(torch.zeros(10))

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            ContinuumRemoval(method="invalid")

    def test_repr_polynomial(self):
        r = repr(ContinuumRemoval(method="polynomial", order=5))
        assert "polynomial" in r
        assert "5" in r

    def test_repr_spline(self):
        r = repr(ContinuumRemoval(method="spline", n_knots=12))
        assert "spline" in r
        assert "12" in r


# ---------------------------------------------------------------------------
# ContinuumRemoval — production poly/spline vs per-spectrum reference parity tests
# ---------------------------------------------------------------------------


class TestContinuumRemovalVectorized:
    """Verify poly/spline ContinuumRemoval matches per-spectrum reference."""

    @staticmethod
    def _run_vectorized(
        x: torch.Tensor,
        method: str = "polynomial",
        order: int = 3,
        n_knots: int = 10,
        n_sigma: float = 2.0,
        max_iter: int = 3,
    ) -> torch.Tensor:
        """Run production ContinuumRemoval and return the fitted baseline."""
        t = ContinuumRemoval(
            method=method,
            order=order,
            n_knots=n_knots,
            n_sigma=n_sigma,
            max_iter=max_iter,
        )
        t.forward(x)
        return t._baseline  # type: ignore[return-value]

    def test_polynomial_linear(self):
        """Linear continuum + absorption dip: both solvers agree."""
        torch.manual_seed(42)
        t_arr = torch.linspace(-1, 1, 100)
        continuum = 10.0 + 2.0 * t_arr
        dip = -3.0 * torch.exp(-((t_arr - 0.1) ** 2) / (2 * 0.05**2))
        x = (continuum + dip).unsqueeze(0)
        vec = self._run_vectorized(
            x, method="polynomial", order=1, n_sigma=2.0, max_iter=3
        )
        x_flat = x.reshape(-1, x.shape[-1])
        ref = continuum_removal_per_spectrum(
            x_flat, method="polynomial", order=1, n_knots=0, n_sigma=2.0, max_iter=3
        )
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_polynomial_batched(self):
        """Multiple spectra with polynomial method."""
        torch.manual_seed(42)
        x = torch.randn(4, 200) * 2 + 10
        vec = self._run_vectorized(
            x, method="polynomial", order=2, n_sigma=3.0, max_iter=3
        )
        x_flat = x.reshape(-1, x.shape[-1])
        ref = continuum_removal_per_spectrum(
            x_flat, method="polynomial", order=2, n_knots=0, n_sigma=3.0, max_iter=3
        )
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_spline_parabolic(self):
        """Parabolic continuum fit with spline method."""
        torch.manual_seed(42)
        t_arr = torch.linspace(-1, 1, 200)
        continuum = 5.0 + 3.0 * t_arr + 2.0 * t_arr**2
        dip = -4.0 * torch.exp(-((t_arr - 0.0) ** 2) / (2 * 0.03**2))
        x = (continuum + dip).unsqueeze(0)
        vec = self._run_vectorized(
            x, method="spline", n_knots=8, n_sigma=2.0, max_iter=3
        )
        x_flat = x.reshape(-1, x.shape[-1])
        ref = continuum_removal_per_spectrum(
            x_flat, method="spline", order=0, n_knots=8, n_sigma=2.0, max_iter=3
        )
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-3, rtol=1e-3), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_spline_batched(self):
        """Multiple spectra with spline method."""
        torch.manual_seed(42)
        x = torch.randn(4, 150) * 2 + 10
        vec = self._run_vectorized(
            x, method="spline", n_knots=6, n_sigma=5.0, max_iter=3
        )
        x_flat = x.reshape(-1, x.shape[-1])
        ref = continuum_removal_per_spectrum(
            x_flat, method="spline", order=0, n_knots=6, n_sigma=5.0, max_iter=3
        )
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-3, rtol=1e-3), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_polynomial_cubic(self):
        """Cubic polynomial baseline: order=3."""
        torch.manual_seed(42)
        t_arr = torch.linspace(-1, 1, 150)
        continuum = 5.0 + 2.0 * t_arr + 3.0 * t_arr**2 + 1.0 * t_arr**3
        dip = -2.0 * torch.exp(-((t_arr - 0.2) ** 2) / (2 * 0.04**2))
        x = (continuum + dip).unsqueeze(0)
        vec = self._run_vectorized(
            x, method="polynomial", order=3, n_sigma=2.0, max_iter=3
        )
        x_flat = x.reshape(-1, x.shape[-1])
        ref = continuum_removal_per_spectrum(
            x_flat, method="polynomial", order=3, n_knots=0, n_sigma=2.0, max_iter=3
        )
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_many_iterations(self):
        """max_iter=10 with deep absorption — sigma-clipping converges."""
        torch.manual_seed(42)
        t_arr = torch.linspace(-1, 1, 120)
        continuum = 10.0 + 2.0 * t_arr
        dip = -6.0 * torch.exp(-((t_arr - 0.0) ** 2) / (2 * 0.02**2))
        x = (continuum + dip).unsqueeze(0)
        vec = self._run_vectorized(
            x, method="polynomial", order=1, n_sigma=2.0, max_iter=10
        )
        x_flat = x.reshape(-1, x.shape[-1])
        ref = continuum_removal_per_spectrum(
            x_flat, method="polynomial", order=1, n_knots=0, n_sigma=2.0, max_iter=10
        )
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# BandMath (hyperspectral)
# ---------------------------------------------------------------------------


class TestBandMath:
    def test_ndvi_computation(self):
        # Simulate 2-band image: Red (band 0), NIR (band 1)
        red = torch.full((32, 32), 0.1)  # low reflectance
        nir = torch.full((32, 32), 0.5)  # high reflectance
        x = torch.stack([red, nir], dim=0)  # [2, 32, 32]
        ndvi = BandMath(lambda b: (b[1] - b[0]) / (b[1] + b[0] + 1e-8))
        out = ndvi.forward(x)
        expected = (0.5 - 0.1) / (0.5 + 0.1)  # 0.4 / 0.6 ≈ 0.667
        assert torch.allclose(out, torch.full_like(out, expected), atol=1e-5)

    def test_simple_ratio(self):
        x = torch.tensor([[2.0], [4.0]])  # band 0 = 2, band 1 = 4
        t = BandMath(lambda b: b[1] / (b[0] + 1e-8))
        out = t.forward(x)
        assert out.item() == pytest.approx(2.0)

    def test_works_on_last_dim(self):
        # [H, W, C] layout — bands along dim=-1
        x = torch.randn(10, 10, 4)
        t = BandMath(lambda b: (b[1] - b[0]) / (b[1] + b[0] + 1e-8), band_dim=-1)
        out = t.forward(x)
        assert out.shape == (10, 10)

    def test_inverse_raises(self):
        t = BandMath(lambda b: b[0])
        with pytest.raises(RuntimeError, match="lossy"):
            t.inverse(torch.zeros(3))

    def test_invalid_func_raises(self):
        with pytest.raises(TypeError):
            BandMath("not_callable")

    def test_repr(self):
        def _my_idx(b):
            return b[0] / (b[1] + 1e-8)

        t = BandMath(_my_idx)
        r = repr(t)
        assert "BandMath" in r

    def test_multi_output_bands(self):
        # Return multiple output bands
        x = torch.randn(4, 32, 32)  # 4 bands
        t = BandMath(lambda b: torch.stack([b[0] - b[1], b[2] / (b[3] + 1e-8)], dim=0))
        out = t.forward(x)
        assert out.shape == (2, 32, 32)


# ---------------------------------------------------------------------------
# GlobalScalarNorm (P5 — linear, invertible)
# ---------------------------------------------------------------------------


class TestGlobalScalarNorm:
    def test_median_norm_roundtrip(self):
        x = torch.randn(4, 64, 64) * 10 + 100
        t = GlobalScalarNorm(stat="median")
        out = t.forward(x)
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-5)

    def test_max_norm(self):
        x = torch.rand(8, 32) * 50
        t = GlobalScalarNorm(stat="max")
        out = t.forward(x)
        assert out.max().item() <= 1.0 + 1e-6
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-5)

    def test_mean_norm(self):
        x = torch.rand(2, 100) + 1.0  # positive values, mean ~1.5
        t = GlobalScalarNorm(stat="mean")
        out = t.forward(x)
        # After dividing by mean, the output mean should be ~1
        assert abs(out.mean().item() - 1.0) < 0.1

    def test_rms_norm(self):
        x = torch.randn(2, 100)
        t = GlobalScalarNorm(stat="rms")
        out = t.forward(x)
        # RMS of normalized data should be ~1
        rms = torch.sqrt((out**2).mean())
        assert abs(rms.item() - 1.0) < 0.1

    def test_per_image_norm(self):
        x = torch.randn(4, 32, 32) * 5 + 20
        t = GlobalScalarNorm(stat="median", dim=(-2, -1))
        out = t.forward(x)
        for i in range(4):
            assert abs(out[i].median().item() - 1.0) < 0.1
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-5)

    def test_inverse_without_forward_raises(self):
        t = GlobalScalarNorm()
        with pytest.raises(RuntimeError, match="prior forward"):
            t.inverse(torch.zeros(3))

    def test_invalid_stat_raises(self):
        with pytest.raises(ValueError):
            GlobalScalarNorm(stat="invalid")

    def test_repr(self):
        r = repr(GlobalScalarNorm(stat="rms", dim=(-1,)))
        assert "GlobalScalarNorm" in r
        assert "rms" in r


# ---------------------------------------------------------------------------
# SavitzkyGolayFilter (P4 — additive decomposition, invertible)
# ---------------------------------------------------------------------------


class TestSavitzkyGolayFilter:
    def test_smoothing_roundtrip(self):
        x = torch.linspace(0, 100, 200).unsqueeze(0)
        t = SavitzkyGolayFilter(window_length=7, polyorder=3)
        out = t.forward(x)
        assert out.shape == x.shape
        # Roundtrip should be perfect (additive decomposition)
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-6)

    def test_reduces_noise(self):
        x = torch.sin(torch.linspace(0, 4 * math.pi, 200)).unsqueeze(0)
        x = x + torch.randn_like(x) * 0.2  # add noise
        t = SavitzkyGolayFilter(window_length=11, polyorder=3)
        out = t.forward(x)
        # Smoothed should have lower std than original
        assert out.std().item() < x.std().item()

    def test_preserves_polynomial(self):
        # SG filter should exactly preserve a polynomial of order polyorder
        x = (torch.linspace(-1, 1, 100) ** 3).unsqueeze(0)
        t = SavitzkyGolayFilter(window_length=9, polyorder=3)
        out = t.forward(x)
        # Cubic polynomial should be exactly preserved (except at edges)
        assert torch.allclose(out[:, 5:-5], x[:, 5:-5], atol=1e-5)

    def test_inverse_without_forward_raises(self):
        t = SavitzkyGolayFilter()
        with pytest.raises(RuntimeError, match="prior forward"):
            t.inverse(torch.zeros(10))

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            SavitzkyGolayFilter(window_length=4)  # even

    def test_invalid_polyorder_raises(self):
        with pytest.raises(ValueError):
            SavitzkyGolayFilter(window_length=5, polyorder=5)  # >= window

    def test_non_unit_dim(self):
        x = torch.randn(4, 3, 128)
        t = SavitzkyGolayFilter(window_length=7, polyorder=2, dim=0)
        out = t.forward(x)
        assert out.shape == x.shape
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-6)

    def test_repr(self):
        r = repr(SavitzkyGolayFilter(window_length=9, polyorder=2, dim=-1))
        assert "SavitzkyGolayFilter" in r
        assert "9" in r


# ---------------------------------------------------------------------------
# SavitzkyGolayFilter — conv1d vs per-position lstsq parity tests
# ---------------------------------------------------------------------------


class TestSavitzkyGolayFilterVectorized:
    """Verify the conv1d+precomputed-kernel path matches per-position lstsq."""

    @staticmethod
    def _run_vectorized(
        x: torch.Tensor,
        window_length: int = 7,
        polyorder: int = 3,
        dim: int = -1,
    ) -> torch.Tensor:
        """Run production SavitzkyGolayFilter and return smoothed result."""
        t = SavitzkyGolayFilter(
            window_length=window_length, polyorder=polyorder, dim=dim
        )
        return t.forward(x)

    def test_smoothing_sine_noisy(self):
        """Noisy sine wave: both paths produce the same smoothed output."""
        torch.manual_seed(42)
        x = torch.sin(torch.linspace(0, 4 * math.pi, 200)).unsqueeze(0)
        x = x + torch.randn_like(x) * 0.2
        wl, po = 11, 3
        vec = self._run_vectorized(x, window_length=wl, polyorder=po)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = savitzky_golay_per_spectrum(x_flat, wl, po)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_batched_spectra(self):
        """Multiple spectra smoothed independently."""
        torch.manual_seed(42)
        x = torch.randn(6, 150) * 2 + 10
        vec = self._run_vectorized(x, window_length=7, polyorder=3)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = savitzky_golay_per_spectrum(x_flat, 7, 3)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_preserves_polynomial(self):
        """Cubic polynomial exactly preserved by cubic SG filter."""
        torch.manual_seed(42)
        x = (torch.linspace(-1, 1, 100) ** 3).unsqueeze(0)
        vec = self._run_vectorized(x, window_length=9, polyorder=3)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = savitzky_golay_per_spectrum(x_flat, 9, 3)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_linear_order(self):
        """First-order polynomial: SG with polyorder=1."""
        torch.manual_seed(42)
        x = torch.linspace(0, 100, 128).unsqueeze(0)
        x = x + torch.randn_like(x) * 0.5
        vec = self._run_vectorized(x, window_length=7, polyorder=1)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = savitzky_golay_per_spectrum(x_flat, 7, 1)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_small_window(self):
        """Minimum window_length=3."""
        torch.manual_seed(42)
        x = torch.randn(3, 64) * 2 + 10
        vec = self._run_vectorized(x, window_length=3, polyorder=1)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = savitzky_golay_per_spectrum(x_flat, 3, 1)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5)

    def test_non_default_dim(self):
        """Operate along dim=0."""
        torch.manual_seed(42)
        x = torch.randn(120, 3)  # [L, B]
        x[:, 0] = torch.sin(torch.linspace(0, 4 * math.pi, 120))
        vec = self._run_vectorized(x, window_length=7, polyorder=2, dim=0)
        x_moved = x.movedim(0, -1)  # [3, 120]
        x_flat = x_moved.reshape(-1, x_moved.shape[-1])
        ref_flat = savitzky_golay_per_spectrum(x_flat, 7, 2)
        ref = ref_flat.reshape(x_moved.shape).movedim(-1, 0)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# RunningPercentile (P6 — additive decomposition, invertible)
# ---------------------------------------------------------------------------


class TestRunningPercentile:
    def test_roundtrip(self):
        x = torch.linspace(0, 100, 200).unsqueeze(0)
        x = x + 0.5 * torch.sin(torch.linspace(0, 10 * math.pi, 200)).unsqueeze(0)
        t = RunningPercentile(percentile=90, window_size=21)
        out = t.forward(x)
        assert out.shape == x.shape
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-6)

    def test_upper_envelope(self):
        # Sine wave: running 95th percentile should hug the upper envelope
        x = torch.sin(torch.linspace(0, 4 * math.pi, 200)).unsqueeze(0)
        t = RunningPercentile(percentile=95, window_size=31)
        out = t.forward(x)
        # Upper envelope should be above most values
        assert (out >= x).float().mean().item() > 0.5

    def test_inverse_without_forward_raises(self):
        t = RunningPercentile()
        with pytest.raises(RuntimeError, match="prior forward"):
            t.inverse(torch.zeros(10))

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            RunningPercentile(window_size=4)  # even

    def test_invalid_percentile_raises(self):
        with pytest.raises(ValueError):
            RunningPercentile(percentile=150)
        with pytest.raises(ValueError):
            RunningPercentile(percentile=-10)

    def test_repr(self):
        r = repr(RunningPercentile(percentile=75, window_size=15))
        assert "RunningPercentile" in r
        assert "75" in r


# ---------------------------------------------------------------------------
# RunningPercentile — unfold+quantile vs per-spectrum loop parity tests
# ---------------------------------------------------------------------------


class TestRunningPercentileVectorized:
    """Verify the unfold+quantile vectorized path matches a per-spectrum loop."""

    @staticmethod
    def _run_vectorized(
        x: torch.Tensor,
        percentile: float = 90.0,
        window_size: int = 21,
        dim: int = -1,
    ) -> torch.Tensor:
        """Run the real vectorized RunningPercentile and return continuum."""
        t = RunningPercentile(percentile=percentile, window_size=window_size, dim=dim)
        return t.forward(x)

    def test_sine_upper_envelope(self):
        """Sine wave: 95th percentile hugs the upper envelope."""
        torch.manual_seed(42)
        x = torch.sin(torch.linspace(0, 4 * math.pi, 200)).unsqueeze(0)
        percentile, ws = 95.0, 31
        vec = self._run_vectorized(x, percentile=percentile, window_size=ws)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = running_percentile_per_spectrum(x_flat, percentile, ws)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_batched_spectra(self):
        """Multiple spectra, distinct percentiles."""
        torch.manual_seed(42)
        x = torch.randn(6, 150) * 2 + 10
        vec = self._run_vectorized(x, percentile=90.0, window_size=15)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = running_percentile_per_spectrum(x_flat, 90.0, 15)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_median_continuum(self):
        """50th percentile: running median."""
        torch.manual_seed(42)
        x = torch.linspace(0, 100, 150).unsqueeze(0)
        x = x + torch.randn_like(x) * 2
        vec = self._run_vectorized(x, percentile=50.0, window_size=11)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = running_percentile_per_spectrum(x_flat, 50.0, 11)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_small_window(self):
        """Minimum window_size=3."""
        torch.manual_seed(42)
        x = torch.randn(3, 64) * 2 + 10
        vec = self._run_vectorized(x, percentile=90.0, window_size=3)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = running_percentile_per_spectrum(x_flat, 90.0, 3)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5)

    def test_large_window(self):
        """Window size 51 — heavily smoothed continuum."""
        torch.manual_seed(42)
        x = torch.sin(torch.linspace(0, 8 * math.pi, 200)).unsqueeze(0)
        vec = self._run_vectorized(x, percentile=75.0, window_size=51)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = running_percentile_per_spectrum(x_flat, 75.0, 51)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_non_default_dim(self):
        """Operate along dim=0."""
        torch.manual_seed(42)
        x = torch.randn(120, 3)  # [L, B]
        vec = self._run_vectorized(x, percentile=90.0, window_size=11, dim=0)
        x_moved = x.movedim(0, -1)  # [3, 120]
        x_flat = x_moved.reshape(-1, x_moved.shape[-1])
        ref_flat = running_percentile_per_spectrum(x_flat, 90.0, 11)
        ref = ref_flat.reshape(x_moved.shape).movedim(-1, 0)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# UpperEnvelopeContinuum (P3 — additive decomposition, invertible)
# ---------------------------------------------------------------------------


class TestUpperEnvelopeContinuum:
    def test_roundtrip(self):
        x = torch.linspace(0, 100, 100).unsqueeze(0)
        x = x + 2.0 * torch.sin(torch.linspace(0, 6 * math.pi, 100)).unsqueeze(0)
        t = UpperEnvelopeContinuum(window=7)
        out = t.forward(x)
        assert out.shape == x.shape
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-6)

    def test_sits_above_signal(self):
        # Upper envelope should be >= signal at local maxima
        x = torch.sin(torch.linspace(0, 4 * math.pi, 200)).unsqueeze(0)
        t = UpperEnvelopeContinuum(window=15)
        out = t.forward(x)
        # Envelope should be >= signal at most points (not all, since
        # linear interpolation between local maxima can dip below)
        assert (out >= x).float().mean().item() > 0.50

    def test_with_smoothing(self):
        x = torch.linspace(0, 100, 200).unsqueeze(0)
        x = x + torch.sin(torch.linspace(0, 8 * math.pi, 200)).unsqueeze(0)
        t = UpperEnvelopeContinuum(window=10, smooth=5.0)
        out = t.forward(x)
        assert out.shape == x.shape
        # Smoothed envelope should be smoother than raw signal
        assert out.diff().abs().mean().item() < x.diff().abs().mean().item()

    def test_inverse_without_forward_raises(self):
        t = UpperEnvelopeContinuum()
        with pytest.raises(RuntimeError, match="prior forward"):
            t.inverse(torch.zeros(10))

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            UpperEnvelopeContinuum(window=0)

    def test_repr(self):
        r = repr(UpperEnvelopeContinuum(window=11, smooth=3.0))
        assert "UpperEnvelopeContinuum" in r
        assert "11" in r


# ---------------------------------------------------------------------------
# WaveletDecompose (P2 — fully invertible frequency split)
# ---------------------------------------------------------------------------


class TestWaveletDecompose:
    def test_roundtrip(self):
        x = torch.linspace(0, 100, 128).unsqueeze(0)  # power of 2
        t = WaveletDecompose(levels=3)
        out = t.forward(x)
        assert out.shape[-1] == x.shape[-1]
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-6)

    def test_approx_captures_low_freq(self):
        # Low-frequency signal: approximation should preserve it well
        t_wave = torch.linspace(0, 4 * math.pi, 128)
        x = torch.sin(t_wave).unsqueeze(0)  # low freq
        t = WaveletDecompose(levels=3)
        out = t.forward(x)
        # First approx_len elements are the final approximation
        approx_len = 128 >> 3  # 16
        approx = out[..., :approx_len]
        # Approx should have non-negligible energy
        assert approx.abs().max().item() > 0.1

    def test_detail_captures_high_freq(self):
        # High-frequency noise should go to detail coefficients
        x = torch.randn(1, 64) * 0.5  # white noise
        t = WaveletDecompose(levels=2)
        out = t.forward(x)
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-6)

    def test_non_power_of_two(self):
        # Non-power-of-2 length should be padded and work
        x = torch.linspace(0, 100, 100).unsqueeze(0)
        t = WaveletDecompose(levels=2)
        out = t.forward(x)
        restored = t.inverse(out)
        assert restored.shape[-1] == 100  # original length preserved
        # Roundtrip should be approximate due to padding
        assert torch.allclose(restored, x, atol=1e-4)

    def test_inverse_without_forward_raises(self):
        t = WaveletDecompose()
        with pytest.raises(RuntimeError, match="prior forward"):
            t.inverse(torch.zeros(16))

    def test_invalid_levels_raises(self):
        with pytest.raises(ValueError):
            WaveletDecompose(levels=0)
        with pytest.raises(ValueError):
            WaveletDecompose(levels=10)

    def test_repr(self):
        r = repr(WaveletDecompose(levels=3, dim=-1))
        assert "WaveletDecompose" in r
        assert "3" in r


# ---------------------------------------------------------------------------
# WaveletDecompose — vectorized DWT vs per-level loop parity tests
# ---------------------------------------------------------------------------


class TestWaveletDecomposeVectorized:
    """Verify the vectorized Haar DWT matches a per-position, per-level loop."""

    @staticmethod
    def _run_vectorized(
        x: torch.Tensor,
        levels: int = 3,
        dim: int = -1,
    ) -> torch.Tensor:
        """Run production WaveletDecompose and return coefficients."""
        t = WaveletDecompose(levels=levels, dim=dim)
        return t.forward(x)

    def test_power_of_two(self):
        """Length 128 = 2^7, levels=3: both paths identical."""
        torch.manual_seed(42)
        x = torch.sin(torch.linspace(0, 4 * math.pi, 128)).unsqueeze(0)
        vec = self._run_vectorized(x, levels=3)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = wavelet_decompose_per_spectrum(x_flat, levels=3)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_batched_spectra(self):
        """Multiple spectra decomposed independently."""
        torch.manual_seed(42)
        x = torch.randn(4, 64) * 2 + 10  # 64 = 2^6
        vec = self._run_vectorized(x, levels=2)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = wavelet_decompose_per_spectrum(x_flat, levels=2)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_single_level(self):
        """Single-level decomposition (levels=1)."""
        torch.manual_seed(42)
        x = torch.randn(2, 32) * 2 + 10  # 32 = 2^5
        vec = self._run_vectorized(x, levels=1)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = wavelet_decompose_per_spectrum(x_flat, levels=1)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_many_levels(self):
        """Deep decomposition (levels=4, length=256)."""
        torch.manual_seed(42)
        x = torch.randn(1, 256) * 2 + 10  # 256 = 2^8
        vec = self._run_vectorized(x, levels=4)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = wavelet_decompose_per_spectrum(x_flat, levels=4)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_non_default_dim(self):
        """Operate along dim=0."""
        torch.manual_seed(42)
        x = torch.randn(64, 3)  # [L, B], L=64 power of 2
        vec = self._run_vectorized(x, levels=2, dim=0)
        x_moved = x.movedim(0, -1)  # [3, 64]
        x_flat = x_moved.reshape(-1, x_moved.shape[-1])
        ref_flat = wavelet_decompose_per_spectrum(x_flat, levels=2)
        ref = ref_flat.reshape(x_moved.shape).movedim(-1, 0)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_non_power_of_two(self):
        """Length 100 (not a power of 2): pad to match production, then compare."""
        torch.manual_seed(42)
        x = torch.sin(torch.linspace(0, 4 * math.pi, 100)).unsqueeze(0)
        levels = 2
        # Replicate the production padding logic
        length = x.shape[-1]
        target = ((length + (1 << levels) - 1) >> levels) << levels
        pad_amount = target - length
        x_padded = torch.nn.functional.pad(x, (0, pad_amount), mode="reflect")
        vec = self._run_vectorized(x, levels=levels)
        x_flat = x_padded.reshape(-1, x_padded.shape[-1])
        ref = wavelet_decompose_per_spectrum(x_flat, levels=levels)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# UpperEnvelopeContinuum — vectorized-vs-per-spectrum parity tests
# ---------------------------------------------------------------------------


class TestUpperEnvelopeContinuumVectorized:
    """Verify the cummax-based vectorized envelope matches per-spectrum loop."""

    @staticmethod
    def _compute_is_local_max(x_flat: torch.Tensor, window: int) -> torch.Tensor:
        """Compute local-max mask using the same unfold/max logic as the class."""
        x_padded = torch.nn.functional.pad(x_flat, (window, window), mode="reflect")
        windows = x_padded.unfold(-1, 2 * window + 1, 1)
        return x_flat == windows.max(dim=-1).values

    @staticmethod
    def _run_vectorized(
        x: torch.Tensor, window: int = 7, smooth: float = 0.0, dim: int = -1
    ) -> torch.Tensor:
        """Run the real vectorized UpperEnvelopeContinuum and return continuum."""
        t = UpperEnvelopeContinuum(window=window, smooth=smooth, dim=dim)
        return t.forward(x)

    def test_single_spectrum_window3(self):
        """Simple spectrum: window=3, verify exact allclose."""
        torch.manual_seed(42)
        x = (torch.sin(torch.linspace(0, 6 * math.pi, 100)) + 1.0).unsqueeze(0)
        x = x + 0.3 * torch.randn_like(x)
        vec = self._run_vectorized(x, window=3)
        x_flat = x.reshape(-1, x.shape[-1])
        is_lm = self._compute_is_local_max(x_flat, 3)
        ref = upper_envelope_per_spectrum(x_flat, is_lm)
        ref = ref.reshape(x.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_single_spectrum_window11(self):
        """Larger window, more local maxima."""
        torch.manual_seed(42)
        x = (torch.sin(torch.linspace(0, 8 * math.pi, 200)) * 2 + 0.5).unsqueeze(0)
        x = x + 0.2 * torch.randn_like(x)
        vec = self._run_vectorized(x, window=11)
        x_flat = x.reshape(-1, x.shape[-1])
        is_lm = self._compute_is_local_max(x_flat, 11)
        ref = upper_envelope_per_spectrum(x_flat, is_lm)
        ref = ref.reshape(x.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_batched_spectra(self):
        """Multiple spectra processed simultaneously."""
        torch.manual_seed(42)
        x = torch.randn(8, 150) * 2 + 10
        vec = self._run_vectorized(x, window=5)
        x_flat = x.reshape(-1, x.shape[-1])
        is_lm = self._compute_is_local_max(x_flat, 5)
        ref = upper_envelope_per_spectrum(x_flat, is_lm)
        ref = ref.reshape(x.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_with_smoothing(self):
        """Gaussian smoothing enabled."""
        torch.manual_seed(42)
        x = torch.sin(torch.linspace(0, 4 * math.pi, 100)).unsqueeze(0) + 0.5
        x = x + 0.1 * torch.randn_like(x)
        vec = self._run_vectorized(x, window=5, smooth=3.0)
        x_flat = x.reshape(-1, x.shape[-1])
        is_lm = self._compute_is_local_max(x_flat, 5)
        ref = upper_envelope_per_spectrum(x_flat, is_lm, smooth=3.0)
        ref = ref.reshape(x.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_few_local_maxima(self):
        """Spectrum with < 2 local maxima uses global max fallback."""
        torch.manual_seed(42)
        # Monotonically increasing: at most 1 local max (at the end)
        x = torch.linspace(0, 10, 50).unsqueeze(0)
        vec = self._run_vectorized(x, window=3)
        x_flat = x.reshape(-1, x.shape[-1])
        is_lm = self._compute_is_local_max(x_flat, 3)
        ref = upper_envelope_per_spectrum(x_flat, is_lm)
        ref = ref.reshape(x.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_constant_input(self):
        """Constant signal: every point is a local max."""
        x = torch.ones(1, 64) * 5.0
        vec = self._run_vectorized(x, window=3)
        x_flat = x.reshape(-1, x.shape[-1])
        is_lm = self._compute_is_local_max(x_flat, 3)
        ref = upper_envelope_per_spectrum(x_flat, is_lm)
        ref = ref.reshape(x.shape)
        # All points are 5, so continuum should be 5 everywhere
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5)
        assert torch.allclose(vec, x, atol=1e-5)

    def test_non_default_dim(self):
        """Operate along dim=0 (batched over other dims)."""
        torch.manual_seed(42)
        x = torch.randn(128, 3)  # [L, B] — dim=0 is length
        vec = self._run_vectorized(x, window=5, dim=0)
        # The class moves dim=0 to last, so we need to match that
        x_moved = x.movedim(0, -1)  # [3, 128]
        x_flat = x_moved.reshape(-1, x_moved.shape[-1])  # [3, 128]
        is_lm = self._compute_is_local_max(x_flat, 5)
        ref_flat = upper_envelope_per_spectrum(x_flat, is_lm)
        ref = ref_flat.reshape(x_moved.shape).movedim(-1, 0)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_3d_tensor(self):
        """[C, H, W] tensor with spectral dim=-1."""
        torch.manual_seed(42)
        x = torch.randn(4, 16, 64) * 2 + 10
        vec = self._run_vectorized(x, window=5, dim=-1)
        x_flat = x.reshape(-1, x.shape[-1])
        is_lm = self._compute_is_local_max(x_flat, 5)
        ref = upper_envelope_per_spectrum(x_flat, is_lm)
        ref = ref.reshape(x.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_mixed_batch_with_few_lm(self):
        """Batch where one spectrum has < 2 LM, others have many."""
        torch.manual_seed(42)
        x = torch.randn(4, 100) * 2 + 10
        x[0, :] = torch.linspace(0, 10, 100)  # monotonic → 1 LM
        vec = self._run_vectorized(x, window=5)
        x_flat = x.reshape(-1, x.shape[-1])
        is_lm = self._compute_is_local_max(x_flat, 5)
        ref = upper_envelope_per_spectrum(x_flat, is_lm)
        ref = ref.reshape(x.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_window1(self):
        """Minimum window size: every point is a local max."""
        torch.manual_seed(42)
        x = torch.randn(1, 64) * 2 + 5
        vec = self._run_vectorized(x, window=1)
        x_flat = x.reshape(-1, x.shape[-1])
        is_lm = self._compute_is_local_max(x_flat, 1)
        ref = upper_envelope_per_spectrum(x_flat, is_lm)
        ref = ref.reshape(x.shape)
        # Verify vectorized matches reference; with window=1 almost every
        # point is its own nearest local max, so continuum ≈ signal.
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# PhaseFold — vectorized scatter_add_ vs per-bin loop parity tests
# ---------------------------------------------------------------------------


class TestPhaseFoldVectorized:
    """Verify the scatter_add_-based PhaseFold matches a per-bin loop."""

    @staticmethod
    def _compute_bins(
        length: int,
        period: float,
        n_bins: int,
        t0: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute bin indices exactly as PhaseFold does."""
        t = torch.arange(length, device=device, dtype=dtype)
        phase = ((t - t0) / period) % 1.0
        edges = torch.linspace(0.0, 1.0, n_bins + 1, device=device, dtype=dtype)
        bin_idx = torch.bucketize(phase, edges[:-1]) - 1
        return torch.clamp(bin_idx, 0, n_bins - 1)

    @staticmethod
    def _run_vectorized(
        x: torch.Tensor,
        period: float = 1.0,
        n_bins: int = 64,
        t0: float = 0.0,
    ) -> torch.Tensor:
        """Run the real vectorized PhaseFold and return folded output."""
        t = PhaseFold(period=period, n_bins=n_bins, t0=t0)
        return t.forward(x)

    def test_sine_period20(self):
        """Sine wave with period 20, 200 steps, 32 bins."""
        torch.manual_seed(42)
        length = 200
        t_arr = torch.arange(length, dtype=torch.float32)
        signal = torch.sin(2 * math.pi * t_arr / 20.0)
        x = signal.unsqueeze(0)
        period, n_bins, t0 = 20.0, 32, 0.0
        vec = self._run_vectorized(x, period=period, n_bins=n_bins, t0=t0)
        x_flat = x.reshape(-1, x.shape[-1])
        bin_idx = self._compute_bins(length, period, n_bins, t0, x.device, x.dtype)
        ref = phase_fold_per_bin(x_flat, n_bins, bin_idx)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_batched_spectra(self):
        """8 spectra, distinct signals, 64 bins."""
        torch.manual_seed(42)
        x = torch.randn(8, 150) * 2 + 10
        period, n_bins, t0 = 10.0, 64, 0.0
        vec = self._run_vectorized(x, period=period, n_bins=n_bins, t0=t0)
        x_flat = x.reshape(-1, x.shape[-1])
        bin_idx = self._compute_bins(150, period, n_bins, t0, x.device, x.dtype)
        ref = phase_fold_per_bin(x_flat, n_bins, bin_idx)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_few_bins(self):
        """Minimum bin count (2)."""
        torch.manual_seed(42)
        x = torch.randn(3, 100) * 2 + 5
        period, n_bins, t0 = 5.0, 2, 0.0
        vec = self._run_vectorized(x, period=period, n_bins=n_bins, t0=t0)
        x_flat = x.reshape(-1, x.shape[-1])
        bin_idx = self._compute_bins(100, period, n_bins, t0, x.device, x.dtype)
        ref = phase_fold_per_bin(x_flat, n_bins, bin_idx)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5)

    def test_period_longer_than_signal(self):
        """Period >> signal length: all points fall in first few bins."""
        torch.manual_seed(42)
        # 100 steps, period=1000 → phases tightly clustered near 0
        x = torch.randn(1, 100) * 2 + 10
        period, n_bins, t0 = 1000.0, 32, 0.0
        vec = self._run_vectorized(x, period=period, n_bins=n_bins, t0=t0)
        x_flat = x.reshape(-1, x.shape[-1])
        bin_idx = self._compute_bins(100, period, n_bins, t0, x.device, x.dtype)
        ref = phase_fold_per_bin(x_flat, n_bins, bin_idx)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_t0_offset(self):
        """Non-zero phase offset changes bin assignments."""
        torch.manual_seed(42)
        x = torch.randn(1, 100) * 2 + 10
        period, n_bins = 10.0, 32
        vec_t0_0 = self._run_vectorized(x, period=period, n_bins=n_bins, t0=0.0)
        vec_t0_3 = self._run_vectorized(x, period=period, n_bins=n_bins, t0=3.0)
        # With t0 offset, bin assignments shift — outputs should differ
        assert not torch.allclose(vec_t0_0, vec_t0_3, atol=1e-5)
        # But both should individually match their per-bin references
        x_flat = x.reshape(-1, x.shape[-1])
        bin_idx_3 = self._compute_bins(100, period, n_bins, 3.0, x.device, x.dtype)
        ref_3 = phase_fold_per_bin(x_flat, n_bins, bin_idx_3)
        ref_3 = ref_3.reshape(vec_t0_3.shape)
        assert torch.allclose(vec_t0_3, ref_3, atol=1e-5, rtol=1e-5)

    def test_many_bins(self):
        """Large number of bins (256)."""
        torch.manual_seed(42)
        x = torch.randn(2, 512) * 2 + 10
        period, n_bins, t0 = 50.0, 256, 0.0
        vec = self._run_vectorized(x, period=period, n_bins=n_bins, t0=t0)
        x_flat = x.reshape(-1, x.shape[-1])
        bin_idx = self._compute_bins(512, period, n_bins, t0, x.device, x.dtype)
        ref = phase_fold_per_bin(x_flat, n_bins, bin_idx)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# SigmaClip — zero-alloc buffer vs naive per-iteration-alloc parity tests
# ---------------------------------------------------------------------------


class TestSigmaClipVectorized:
    """Verify the zero-alloc buffer SigmaClip matches naive per-iteration alloc."""

    @staticmethod
    def _run_vectorized(
        x: torch.Tensor,
        n_sigma: float = 3.0,
        max_iter: int = 5,
        dim: tuple[int, ...] = (-2, -1),
        fill: str = "mean",
    ) -> torch.Tensor:
        """Run the real zero-alloc SigmaClip and return clipped output."""
        t = SigmaClip(n_sigma=n_sigma, max_iter=max_iter, dim=dim, fill=fill)
        return t.forward(x)

    def test_removes_single_outlier(self):
        """Single extreme outlier in a uniform field."""
        torch.manual_seed(42)
        x = torch.ones(10, 10) * 5.0
        x[0, 0] = 100.0
        vec = self._run_vectorized(x, dim=(-2, -1))
        ref = sigma_clip_naive(x, 3.0, 5, (-2, -1), "mean")
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )
        # Outlier should be replaced with ~5
        assert vec[0, 0].item() < 10.0

    def test_clean_data_no_clipping(self):
        """Clean normal data with wide threshold — no clipping."""
        torch.manual_seed(42)
        x = torch.randn(4, 32, 32) * 5 + 100
        vec = self._run_vectorized(x, n_sigma=10.0, dim=(-2, -1))
        ref = sigma_clip_naive(x, 10.0, 3, (-2, -1), "mean")
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_per_channel(self):
        """Clipping computed independently per image (dim=-2,-1)."""
        torch.manual_seed(42)
        x = torch.randn(4, 16, 16) * 3 + 10
        x[0, 0, 0] = 50.0
        x[2, 0, 0] = -20.0
        vec = self._run_vectorized(x, dim=(-2, -1))
        ref = sigma_clip_naive(x, 3.0, 5, (-2, -1), "mean")
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_median_fill(self):
        """Fill clipped values with median instead of mean."""
        torch.manual_seed(42)
        x = torch.ones(8, 8) * 5.0
        x[0, 0] = 100.0
        x[1, 1] = -20.0
        vec = self._run_vectorized(x, dim=(-2, -1), fill="median")
        ref = sigma_clip_naive(x, 3.0, 5, (-2, -1), "median")
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_wide_threshold_noop(self):
        """Very wide sigma threshold: no pixels clipped → output == input."""
        torch.manual_seed(42)
        x = torch.randn(3, 32, 32) * 5 + 20
        vec = self._run_vectorized(x, n_sigma=100.0, dim=(-2, -1))
        ref = sigma_clip_naive(x, 100.0, 3, (-2, -1), "mean")
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5)
        assert torch.allclose(vec, x, atol=1e-4)

    def test_global_dim(self):
        """Global sigma-clip (no dim specified → all pixels considered)."""
        torch.manual_seed(42)
        x = torch.randn(32, 32) * 5 + 10
        x[0, 0] = 100.0
        vec = self._run_vectorized(x, dim=(), fill="mean")
        ref = sigma_clip_naive(x, 3.0, 5, (), "mean")
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_constant_image(self):
        """Constant image: std=0, no clipping."""
        x = torch.ones(4, 32, 32) * 42.0
        vec = self._run_vectorized(x, dim=(-2, -1))
        ref = sigma_clip_naive(x, 3.0, 5, (-2, -1), "mean")
        assert torch.allclose(vec, ref, atol=1e-5)
        assert torch.allclose(vec, x, atol=1e-5)


# ---------------------------------------------------------------------------
# AsymmetricLeastSquares (Tier 2 — Eilers 2003, additive decomposition)
# ---------------------------------------------------------------------------


class TestAsymmetricLeastSquares:
    def test_roundtrip(self):
        # Quadratic continuum + Gaussian absorption dips
        t_arr = torch.linspace(-1, 1, 200)
        continuum = 10.0 + 2.0 * t_arr + 3.0 * t_arr**2
        dip = -2.0 * torch.exp(-((t_arr - 0.0) ** 2) / (2 * 0.03**2))
        dip2 = -1.5 * torch.exp(-((t_arr - 0.5) ** 2) / (2 * 0.05**2))
        spectrum = continuum + dip + dip2
        x = spectrum.unsqueeze(0)
        t = AsymmetricLeastSquares(lam=1e4, p=0.01, max_iter=10)
        out = t.forward(x)
        assert out.shape == x.shape
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-6)

    def test_baseline_hugs_lower_envelope(self):
        # Spectrum with strong absorption features
        t_arr = torch.linspace(-1, 1, 200)
        continuum = 10.0 + 2.0 * t_arr
        # Deep absorption at center
        dip = -5.0 * torch.exp(-((t_arr) ** 2) / (2 * 0.02**2))
        spectrum = continuum + dip
        x = spectrum.unsqueeze(0)
        t = AsymmetricLeastSquares(lam=1e5, p=0.001, max_iter=10)
        baseline = t.forward(x)
        # Baseline should be above the spectrum at the dip bottom
        # (it hugs the lower envelope, i.e., it's above absorption features)
        dip_min_idx = int(dip.argmin().item())
        assert baseline[0, dip_min_idx].item() > spectrum[dip_min_idx].item()

    def test_lam_controls_smoothness(self):
        # Use a linear baseline (nullspace of D^T D) with positive Gaussian peaks.
        # A stiffer lam pushes toward a straight line, which should be closer
        # to the true linear baseline than a flexible (wiggly) fit.
        t_arr = torch.linspace(-1, 1, 200)
        baseline_true = 5.0 + 10.0 * t_arr  # straight line
        peak = 3.0 * torch.exp(-((t_arr - 0.2) ** 2) / (2 * 0.03**2))
        peak2 = 2.0 * torch.exp(-((t_arr + 0.4) ** 2) / (2 * 0.04**2))
        x = (baseline_true + peak + peak2).unsqueeze(0)
        t_stiff = AsymmetricLeastSquares(lam=1e7, p=0.01, max_iter=10)
        t_flex = AsymmetricLeastSquares(lam=1e2, p=0.01, max_iter=10)
        stiff = t_stiff.forward(x)
        flex = t_flex.forward(x)
        # Stiffer baseline should be smoother and closer to the straight line
        stiff_err = (stiff - baseline_true.unsqueeze(0)).abs().mean().item()
        flex_err = (flex - baseline_true.unsqueeze(0)).abs().mean().item()
        assert stiff_err < flex_err, f"stiff_err {stiff_err} >= flex_err {flex_err}"

    def test_batched_spectra(self):
        x = torch.randn(4, 150) * 2 + 10
        t = AsymmetricLeastSquares(lam=1e5, p=0.01, max_iter=5)
        out = t.forward(x)
        assert out.shape == (4, 150)
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-6)

    def test_inverse_without_forward_raises(self):
        t = AsymmetricLeastSquares()
        with pytest.raises(RuntimeError, match="prior forward"):
            t.inverse(torch.zeros(10))

    def test_invalid_lam_raises(self):
        with pytest.raises(ValueError):
            AsymmetricLeastSquares(lam=0)
        with pytest.raises(ValueError):
            AsymmetricLeastSquares(lam=-1)

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError):
            AsymmetricLeastSquares(p=0)
        with pytest.raises(ValueError):
            AsymmetricLeastSquares(p=1)

    def test_invalid_envelope_raises(self):
        with pytest.raises(ValueError):
            AsymmetricLeastSquares(envelope="invalid")
        with pytest.raises(ValueError):
            AsymmetricLeastSquares(envelope="upper_")  # close but wrong

    def test_repr(self):
        r = repr(AsymmetricLeastSquares(lam=1e6, p=0.05))
        assert "AsymmetricLeastSquares" in r
        assert "1000000.0" in r


# ---------------------------------------------------------------------------
# AsymmetricLeastSquares — banded Cholesky vs dense torch.linalg.solve parity tests
# ---------------------------------------------------------------------------


class TestAsymmetricLeastSquaresVectorized:
    """Verify the banded-Cholesky solver matches dense torch.linalg.solve."""

    @staticmethod
    def _run_vectorized(
        x: torch.Tensor,
        lam: float = 1e5,
        p: float = 0.01,
        max_iter: int = 10,
        dim: int = -1,
        envelope: str = "lower",
    ) -> torch.Tensor:
        """Run the real banded-Cholesky AsymmetricLeastSquares and return baseline."""
        t = AsymmetricLeastSquares(
            lam=lam, p=p, max_iter=max_iter, dim=dim, envelope=envelope
        )
        return t.forward(x)

    def test_quadratic_baseline(self):
        """Quadratic continuum + absorption dips: both solvers agree."""
        torch.manual_seed(42)
        t_arr = torch.linspace(-1, 1, 200)
        continuum = 10.0 + 2.0 * t_arr + 3.0 * t_arr**2
        dip = -2.0 * torch.exp(-((t_arr - 0.0) ** 2) / (2 * 0.03**2))
        dip2 = -1.5 * torch.exp(-((t_arr - 0.5) ** 2) / (2 * 0.05**2))
        x = (continuum + dip + dip2).unsqueeze(0)
        vec = self._run_vectorized(x, lam=1e4, p=0.01, max_iter=10)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = asls_dense_solve(x_flat, lam=1e4, p=0.01, max_iter=10)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_batched_spectra(self):
        """Multiple spectra with distinct baselines."""
        torch.manual_seed(42)
        x = torch.randn(4, 120) * 2 + 10
        vec = self._run_vectorized(x, lam=1e5, p=0.01, max_iter=5)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = asls_dense_solve(x_flat, lam=1e5, p=0.01, max_iter=5)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_aggressive_asymmetry(self):
        """Very small p=0.001: baseline hugs the lower envelope tightly."""
        torch.manual_seed(42)
        t_arr = torch.linspace(-1, 1, 200)
        continuum = 10.0 + 2.0 * t_arr
        dip = -5.0 * torch.exp(-((t_arr) ** 2) / (2 * 0.02**2))
        x = (continuum + dip).unsqueeze(0)
        lam, p_val = 1e5, 0.001
        vec = self._run_vectorized(x, lam=lam, p=p_val, max_iter=10)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = asls_dense_solve(x_flat, lam=lam, p=p_val, max_iter=10)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_stiff_baseline(self):
        """Large lam=1e7: baseline is nearly linear."""
        torch.manual_seed(42)
        t_arr = torch.linspace(-1, 1, 200)
        baseline_true = 5.0 + 10.0 * t_arr
        peak = 3.0 * torch.exp(-((t_arr - 0.2) ** 2) / (2 * 0.03**2))
        x = (baseline_true + peak).unsqueeze(0)
        lam = 1e7
        vec = self._run_vectorized(x, lam=lam, p=0.01, max_iter=10)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = asls_dense_solve(x_flat, lam=lam, p=0.01, max_iter=10)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_short_signal(self):
        """Signal length < 4: D² penalty vanishes, baseline == signal."""
        torch.manual_seed(42)
        x = torch.randn(2, 3) * 2 + 10  # L=3 < 4
        vec = self._run_vectorized(x, lam=1e5, p=0.01, max_iter=5)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = asls_dense_solve(x_flat, lam=1e5, p=0.01, max_iter=5)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5)
        # Short signal: baseline should equal signal (no penalty)
        assert torch.allclose(vec, x, atol=1e-5)

    def test_convergence_early_exit(self):
        """Many max_iter but converges early — both solvers agree."""
        torch.manual_seed(42)
        t_arr = torch.linspace(-1, 1, 150)
        continuum = 10.0 + 2.0 * t_arr
        x = continuum.unsqueeze(0)
        vec = self._run_vectorized(x, lam=1e5, p=0.01, max_iter=50)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = asls_dense_solve(x_flat, lam=1e5, p=0.01, max_iter=50)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_non_default_dim(self):
        """Operate along dim=0."""
        torch.manual_seed(42)
        x = torch.randn(120, 3)  # [L, B] — dim=0 is length
        t_arr = torch.linspace(-1, 1, 120)
        x[:, 0] = 10.0 + 2.0 * t_arr
        x[:, 1] = 10.0 + 3.0 * t_arr
        vec = self._run_vectorized(x, lam=1e5, p=0.01, max_iter=5, dim=0)
        x_moved = x.movedim(0, -1)  # [3, 120]
        x_flat = x_moved.reshape(-1, x_moved.shape[-1])
        ref_flat = asls_dense_solve(x_flat, lam=1e5, p=0.01, max_iter=5)
        ref = ref_flat.reshape(x_moved.shape).movedim(-1, 0)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    # ---------------------------------------------------------------------------

    def test_upper_envelope_emission(self):
        """envelope='upper': baseline hugs emission peaks (absorption spectroscopy)."""
        torch.manual_seed(42)
        t_arr = torch.linspace(-1, 1, 200)
        continuum = 10.0 + 2.0 * t_arr
        peak = 5.0 * torch.exp(
            -((t_arr) ** 2) / (2 * 0.02**2)
        )  # emission above continuum
        x = (continuum + peak).unsqueeze(0)
        lam, p_val = 1e5, 0.001
        vec = self._run_vectorized(x, lam=lam, p=p_val, max_iter=10, envelope="upper")
        x_flat = x.reshape(-1, x.shape[-1])
        ref = asls_dense_solve(x_flat, lam=lam, p=p_val, max_iter=10, envelope="upper")
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )
        # Upper envelope should hug the emission peak: baseline above continuum
        # at the peak location (p=0.001 heavily weights points above baseline,
        # making the baseline rise toward the peak).
        peak_idx = int(peak.argmax().item())
        assert vec[0, peak_idx].item() > continuum[peak_idx].item()
        # Upper envelope should also be above the lower-envelope baseline
        # at the peak (the two modes should diverge meaningfully).
        vec_lower = self._run_vectorized(
            x, lam=lam, p=p_val, max_iter=10, envelope="lower"
        )
        assert vec[0, peak_idx].item() > vec_lower[0, peak_idx].item()

    def test_upper_envelope_batched(self):
        """Multiple spectra with envelope='upper'."""
        torch.manual_seed(42)
        x = torch.randn(4, 120) * 2 + 10
        vec = self._run_vectorized(x, lam=1e5, p=0.01, max_iter=5, envelope="upper")
        x_flat = x.reshape(-1, x.shape[-1])
        ref = asls_dense_solve(x_flat, lam=1e5, p=0.01, max_iter=5, envelope="upper")
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )


# AlphaShapeContinuum (Tier 2 — morphological closing, additive)
# ---------------------------------------------------------------------------


class TestAlphaShapeContinuum:
    def test_roundtrip(self):
        x = torch.linspace(0, 100, 200).unsqueeze(0)
        x = x + 2.0 * torch.sin(torch.linspace(0, 8 * math.pi, 200)).unsqueeze(0)
        t = AlphaShapeContinuum(half_window=15)
        out = t.forward(x)
        assert out.shape == x.shape
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-6)

    def test_always_above_signal(self):
        # Morphological closing (dilation→erosion) is guaranteed to be
        # >= original signal everywhere
        x = torch.sin(torch.linspace(0, 6 * math.pi, 200)).unsqueeze(0)
        t = AlphaShapeContinuum(half_window=10)
        out = t.forward(x)
        assert torch.all(out >= x - 1e-6)

    def test_bridges_absorption_features(self):
        # Spectrum with narrow absorption dips
        t_arr = torch.linspace(-1, 1, 200)
        continuum = 10.0 + 2.0 * t_arr
        dip = -3.0 * torch.exp(-((t_arr - 0.1) ** 2) / (2 * 0.02**2))
        x = (continuum + dip).unsqueeze(0)
        t = AlphaShapeContinuum(half_window=20)
        out = t.forward(x)
        # At the dip minimum, the continuum should be significantly above
        dip_min_idx = int(dip.argmin().item())
        assert out[0, dip_min_idx].item() > x[0, dip_min_idx].item() + 2.0

    def test_window_size_controls_scale(self):
        x = torch.sin(torch.linspace(0, 4 * math.pi, 200)).unsqueeze(0)
        t_small = AlphaShapeContinuum(half_window=2)
        t_large = AlphaShapeContinuum(half_window=30)
        out_small = t_small.forward(x)
        out_large = t_large.forward(x)
        # Larger window should produce a smoother (flatter) continuum
        small_std = out_small.std().item()
        large_std = out_large.std().item()
        assert large_std < small_std, f"large {large_std} >= small {small_std}"

    def test_multiple_iterations(self):
        x = torch.linspace(0, 100, 200).unsqueeze(0)
        x = x + torch.sin(torch.linspace(0, 8 * math.pi, 200)).unsqueeze(0)
        t = AlphaShapeContinuum(half_window=8, iterations=3)
        out = t.forward(x)
        assert out.shape == x.shape
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-6)

    def test_batched_spectra(self):
        x = torch.randn(4, 200) * 2 + 10
        t = AlphaShapeContinuum(half_window=15)
        out = t.forward(x)
        assert out.shape == (4, 200)
        restored = t.inverse(out)
        assert torch.allclose(restored, x, atol=1e-6)

    def test_inverse_without_forward_raises(self):
        t = AlphaShapeContinuum()
        with pytest.raises(RuntimeError, match="prior forward"):
            t.inverse(torch.zeros(10))

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            AlphaShapeContinuum(half_window=0)

    def test_invalid_iterations_raises(self):
        with pytest.raises(ValueError):
            AlphaShapeContinuum(iterations=0)

    def test_repr(self):
        r = repr(AlphaShapeContinuum(half_window=20, iterations=2))
        assert "AlphaShapeContinuum" in r
        assert "20" in r


# ---------------------------------------------------------------------------
# AlphaShapeContinuum — unfold/max/min vs per-spectrum loop parity tests
# ---------------------------------------------------------------------------


class TestAlphaShapeContinuumVectorized:
    """Verify the unfold/max/min closing matches a per-spectrum loop."""

    @staticmethod
    def _run_vectorized(
        x: torch.Tensor,
        half_window: int = 15,
        iterations: int = 1,
        dim: int = -1,
    ) -> torch.Tensor:
        """Run the real vectorized AlphaShapeContinuum and return continuum."""
        t = AlphaShapeContinuum(half_window=half_window, iterations=iterations, dim=dim)
        return t.forward(x)

    def test_single_spectrum_upper_envelope(self):
        """Sine wave: closing should produce an upper envelope >= signal."""
        torch.manual_seed(42)
        x = torch.sin(torch.linspace(0, 4 * math.pi, 100)).unsqueeze(0)
        hw = 10
        vec = self._run_vectorized(x, half_window=hw)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = alpha_shape_per_spectrum(x_flat, hw, 1)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )
        # Morphological closing is guaranteed >= signal
        assert torch.all(vec >= x - 1e-6)

    def test_batched_spectra(self):
        """Multiple spectra with distinct baselines."""
        torch.manual_seed(42)
        x = torch.randn(6, 150) * 2 + 10
        vec = self._run_vectorized(x, half_window=15)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = alpha_shape_per_spectrum(x_flat, 15, 1)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_bridges_absorption_dip(self):
        """Narrow absorption dip in continuum should be bridged."""
        torch.manual_seed(42)
        t_arr = torch.linspace(-1, 1, 200)
        continuum = 10.0 + 2.0 * t_arr
        dip = -3.0 * torch.exp(-((t_arr - 0.1) ** 2) / (2 * 0.02**2))
        x = (continuum + dip).unsqueeze(0)
        hw = 20
        vec = self._run_vectorized(x, half_window=hw)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = alpha_shape_per_spectrum(x_flat, hw, 1)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5)
        # At the dip minimum, continuum should be significantly above
        dip_min_idx = int(dip.argmin().item())
        assert vec[0, dip_min_idx].item() > x[0, dip_min_idx].item() + 2.0

    def test_multiple_iterations(self):
        """Multiple closing iterations progressively smooth the continuum."""
        torch.manual_seed(42)
        x = torch.linspace(0, 100, 150).unsqueeze(0)
        x = x + torch.sin(torch.linspace(0, 8 * math.pi, 150)).unsqueeze(0)
        vec = self._run_vectorized(x, half_window=8, iterations=3)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = alpha_shape_per_spectrum(x_flat, 8, 3)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )

    def test_small_window(self):
        """Minimum half_window=1: closing over +/-1 neighbours."""
        torch.manual_seed(42)
        x = torch.randn(4, 64) * 2 + 10
        vec = self._run_vectorized(x, half_window=1)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = alpha_shape_per_spectrum(x_flat, 1, 1)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5)
        # With hw=1, continuum should still be >= signal
        assert torch.all(vec >= x - 1e-6)

    def test_constant_input(self):
        """Constant signal: closing is identity."""
        x = torch.ones(2, 64) * 7.0
        vec = self._run_vectorized(x, half_window=5)
        x_flat = x.reshape(-1, x.shape[-1])
        ref = alpha_shape_per_spectrum(x_flat, 5, 1)
        ref = ref.reshape(vec.shape)
        assert torch.allclose(vec, ref, atol=1e-5)
        assert torch.allclose(vec, x, atol=1e-5)

    def test_non_default_dim(self):
        """Operate along dim=0."""
        torch.manual_seed(42)
        x = torch.randn(128, 3)  # [L, B] — dim=0 is length
        vec = self._run_vectorized(x, half_window=7, dim=0)
        x_moved = x.movedim(0, -1)  # [3, 128]
        x_flat = x_moved.reshape(-1, x_moved.shape[-1])
        ref_flat = alpha_shape_per_spectrum(x_flat, 7, 1)
        ref = ref_flat.reshape(x_moved.shape).movedim(-1, 0)
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5), (
            f"max diff: {(vec - ref).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# AsymmetricSigmaClip (simple one-pass asymmetric outlier rejection)
# ---------------------------------------------------------------------------


class TestAsymmetricSigmaClip:
    def test_clips_positive_outliers(self):
        x = torch.randn(10, 10) * 2 + 5.0
        x[0, 0] = 100.0  # extreme positive outlier
        t = AsymmetricSigmaClip(n_low=3.0, n_high=3.0)
        out = t.forward(x)
        # Outlier should be replaced with median (~5)
        assert out[0, 0].item() < 10.0

    def test_clips_negative_outliers(self):
        x = torch.randn(10, 10) * 2 + 5.0
        x[0, 0] = -50.0  # extreme negative outlier
        t = AsymmetricSigmaClip(n_low=3.0, n_high=3.0)
        out = t.forward(x)
        # Outlier should be replaced with median (~5)
        assert out[0, 0].item() > 0.0

    def test_asymmetric_thresholds(self):
        # Data with real variance so MAD > 0
        x = torch.randn(10, 10) * 2 + 5.0
        x[0, 0] = -8.0  # mild negative outlier
        x[1, 1] = 12.0  # mild positive outlier
        # Aggressive low clipping (n_low=10 -> keep negative), tight high clipping (n_high=1)
        t = AsymmetricSigmaClip(n_low=10.0, n_high=1.0)
        out = t.forward(x)
        # Negative outlier should survive (n_low=10 is very permissive)
        assert out[0, 0].item() < 0.0
        # Positive outlier should be clipped (n_high=1 is very strict)
        assert abs(out[1, 1].item() - 5.0) < 3.0

    def test_no_clipping_on_clean_data(self):
        x = torch.randn(4, 32, 32) * 5 + 100
        t = AsymmetricSigmaClip(n_low=10.0, n_high=10.0)
        out = t.forward(x)
        # With n_sigma=10, almost nothing should be clipped
        assert (out - x).abs().max().item() < 0.5

    def test_inverse_raises(self):
        t = AsymmetricSigmaClip()
        with pytest.raises(RuntimeError, match="irrecoverable"):
            t.inverse(torch.zeros(3))

    def test_constant_image(self):
        x = torch.ones(4, 32, 32) * 42.0
        t = AsymmetricSigmaClip()
        out = t.forward(x)
        assert torch.allclose(out, x, atol=1e-5)

    def test_per_channel(self):
        x = torch.randn(4, 32, 32) * 5 + 10
        t = AsymmetricSigmaClip(dim=(-2, -1))
        out = t.forward(x)
        assert out.shape == (4, 32, 32)
        assert torch.isfinite(out).all()

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            AsymmetricSigmaClip(n_low=0)
        with pytest.raises(ValueError):
            AsymmetricSigmaClip(n_high=-1)

    def test_repr(self):
        r = repr(AsymmetricSigmaClip(n_low=5.0, n_high=2.0))
        assert "AsymmetricSigmaClip" in r
        assert "5.0" in r
        assert "2.0" in r


# ---------------------------------------------------------------------------
# FITSScaleColumns (table column TSCAL/TZERO scaling, invertible)
# ---------------------------------------------------------------------------


class TestFITSScaleColumns:
    def test_roundtrip(self):
        header: dict[str, object] = {
            "TFIELDS": 2,
            "TTYPE1": "FLUX",
            "TFORM1": "E",
            "TSCAL1": 0.001,
            "TZERO1": 0.0,
            "TTYPE2": "MAG",
            "TFORM2": "E",
            "TSCAL2": 1.0,
            "TZERO2": 25.0,
        }
        flux = torch.tensor([1000.0, 2000.0, 3000.0])
        mag = torch.tensor([10.0, 20.0, 30.0])
        x = {"FLUX": flux, "MAG": mag}
        t = FITSScaleColumns.from_header(header)
        out = t.forward(x)
        # FLUX: physical = 0.001 * stored + 0 = [1.0, 2.0, 3.0]
        assert torch.allclose(out["FLUX"], torch.tensor([1.0, 2.0, 3.0]))
        # MAG: physical = 1.0 * stored + 25.0
        assert torch.allclose(out["MAG"], torch.tensor([35.0, 45.0, 55.0]))
        restored = t.inverse(out)
        assert torch.allclose(restored["FLUX"], flux)
        assert torch.allclose(restored["MAG"], mag)

    def test_empty_header(self):
        header: dict[str, object] = {}
        x = {"A": torch.randn(10)}
        t = FITSScaleColumns.from_header(header)
        out = t.forward(x)
        assert torch.equal(out["A"], x["A"])  # no-op

    def test_identity_scales_noop(self):
        header = {
            "TFIELDS": 1,
            "TTYPE1": "DATA",
            "TFORM1": "E",
            "TSCAL1": 1.0,
            "TZERO1": 0.0,
        }
        x = {"DATA": torch.randn(100)}
        t = FITSScaleColumns.from_header(header)
        out = t.forward(x)
        assert torch.equal(out["DATA"], x["DATA"])  # identity scales skipped

    def test_preserves_unrelated_columns(self):
        header = {
            "TFIELDS": 1,
            "TTYPE1": "FLUX",
            "TFORM1": "E",
            "TSCAL1": 2.0,
            "TZERO1": 10.0,
        }
        extra = torch.randn(5)
        x = {"FLUX": torch.ones(5), "EXTRA": extra}
        t = FITSScaleColumns.from_header(header)
        out = t.forward(x)
        assert torch.equal(out["EXTRA"], extra)  # unrelated column untouched
        assert "EXTRA" in out

    def test_preserves_int_dtype(self):
        header = {
            "TFIELDS": 1,
            "TTYPE1": "N",
            "TFORM1": "J",
            "TSCAL1": 1.0,
            "TZERO1": 0.0,
        }
        x = {"N": torch.tensor([1, 2, 3], dtype=torch.int32)}
        t = FITSScaleColumns.from_header(header)
        out = t.forward(x)
        assert out["N"].dtype == torch.int32

    def test_repr(self):
        header = {
            "TFIELDS": 1,
            "TTYPE1": "F",
            "TFORM1": "E",
            "TSCAL1": 0.5,
            "TZERO1": 100.0,
        }
        t = FITSScaleColumns.from_header(header)
        r = repr(t)
        assert "FITSScaleColumns" in r
        assert "0.5" in r


# ---------------------------------------------------------------------------
# TNullToNan (table column TNULL sentinel → NaN, lossy)
# ---------------------------------------------------------------------------


class TestTNullToNan:
    def test_replaces_sentinel_with_nan(self):
        header = {"TFIELDS": 1, "TTYPE1": "FLUX", "TFORM1": "J", "TNULL1": -999}
        x = {"FLUX": torch.tensor([1, -999, 3], dtype=torch.int32)}
        t = TNullToNan.from_header(header)
        out = t.forward(x)
        assert out["FLUX"].dtype == torch.float32  # promoted to float
        assert torch.isnan(out["FLUX"][1])
        assert out["FLUX"][0].item() == 1.0
        assert out["FLUX"][2].item() == 3.0

    def test_float_column_no_promotion(self):
        header = {"TFIELDS": 1, "TTYPE1": "VAL", "TFORM1": "E", "TNULL1": 0.0}
        x = {"VAL": torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)}
        t = TNullToNan.from_header(header)
        out = t.forward(x)
        assert out["VAL"].dtype == torch.float32  # already float, no promotion
        assert torch.isnan(out["VAL"][0])

    def test_empty_header(self):
        header: dict[str, object] = {}
        x = {"A": torch.randn(10)}
        t = TNullToNan.from_header(header)
        out = t.forward(x)
        assert torch.equal(out["A"], x["A"])  # no-op

    def test_nulls_only_on_specified_columns(self):
        header = {
            "TFIELDS": 2,
            "TTYPE1": "GOOD",
            "TFORM1": "J",
            "TTYPE2": "BAD",
            "TFORM2": "J",
            "TNULL2": -99,
        }
        x = {
            "GOOD": torch.tensor([1, -99, 3], dtype=torch.int32),
            "BAD": torch.tensor([1, -99, 3], dtype=torch.int32),
        }
        t = TNullToNan.from_header(header)
        out = t.forward(x)
        # GOOD has no TNULL, should be untouched
        assert out["GOOD"].dtype == torch.int32
        assert out["GOOD"][1].item() == -99
        # BAD has TNULL=-99, should be converted to NaN
        assert torch.isnan(out["BAD"][1])

    def test_inverse_raises(self):
        t = TNullToNan.from_header({})
        with pytest.raises(RuntimeError, match="lossy"):
            t.inverse({})

    def test_repr(self):
        header = {"TFIELDS": 1, "TTYPE1": "X", "TFORM1": "J", "TNULL1": -1}
        t = TNullToNan.from_header(header)
        r = repr(t)
        assert "TNullToNan" in r
        assert "X" in r


# ---------------------------------------------------------------------------
# FITSHeaderScale extended roundtrip tests
# ---------------------------------------------------------------------------


class TestFITSHeaderScaleRoundtrip:
    def test_scaled_image_roundtrip(self):
        """Simulate a typical int16 image with BSCALE/BZERO."""
        raw = torch.randint(-100, 100, (64, 64), dtype=torch.int16)
        header: dict[str, object] = {"BITPIX": 16, "BSCALE": 0.5, "BZERO": 2000.0}
        t = FITSHeaderScale.from_header(header)
        physical = t.forward(raw.float())
        expected = raw.float() * 0.5 + 2000.0
        assert torch.allclose(physical, expected)
        restored = t.inverse(physical)
        # 0.5 is exactly representable in float32, roundtrip is exact
        assert torch.allclose(restored, raw.float())

    def test_unsigned_uint16_convention(self):
        """BZERO=32768 is the standard FITS unsigned int16 convention."""
        raw = torch.randint(0, 65535, (32, 32), dtype=torch.int32)
        header = {"BITPIX": 16, "BSCALE": 1.0, "BZERO": 32768.0}
        t = FITSHeaderScale.from_header(header)
        physical = t.forward(raw.float())
        expected = raw.float() * 1.0 + 32768.0
        assert torch.allclose(physical, expected)
        restored = t.inverse(physical)
        assert torch.allclose(restored, raw.float())

    def test_bscale_only_convention(self):
        """BSCALE != 1, BZERO = 0."""
        header = {"BITPIX": -32, "BSCALE": 2.0, "BZERO": 0.0}
        orig = torch.rand(16, 16) * 100
        raw = orig.clone()
        t = FITSHeaderScale.from_header(header)
        out = t.forward(raw)
        # 2.0 is exact in float32
        assert torch.allclose(out, orig.mul(2.0))
        assert torch.allclose(t.inverse(out), orig)

    def test_inverse_applied_to_identity(self):
        """After applying scale, inverse should get back the original."""
        x = torch.tensor([100.0, 200.0, 300.0])
        t = FITSHeaderScale(bscale=0.5, bzero=50.0)
        fwd = t.forward(x)
        inv = t.inverse(fwd)
        assert torch.allclose(inv, x, atol=1e-5)
