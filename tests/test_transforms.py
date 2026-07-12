"""Tests for torchfits.transforms — ML-friendly FITS image preprocessing."""

import math
import pytest
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
    _amin,
    _amax,
    _flatten_dims,
    _median,
    _normalize_dims,
    _quantile,
    _reduce_keepdim,
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
            def forward(self, x):
                return x + 1

            def inverse(self, x):
                return x - 1

        d = Dummy()
        assert d(torch.tensor(1.0)).item() == 2.0
        assert d.inverse(torch.tensor(2.0)).item() == 1.0


class TestCompose:
    def test_forward_chain(self):
        class AddOne(FITSTransform):
            def forward(self, x):
                return x + 1

            def inverse(self, x):
                return x - 1

        c = Compose([AddOne(), AddOne(), AddOne()])
        assert c(torch.tensor(0.0)).item() == 3.0

    def test_inverse_reverses_chain(self):
        class MulTwo(FITSTransform):
            def forward(self, x):
                return x * 2

            def inverse(self, x):
                return x / 2

        c = Compose([MulTwo(), MulTwo()])
        x = torch.tensor(5.0)
        fwd = c.forward(x)
        assert fwd.item() == 20.0
        inv = c.inverse(fwd)
        assert torch.allclose(inv, x)

    def test_len_and_getitem(self):
        class Id(FITSTransform):
            def forward(self, x):
                return x

            def inverse(self, x):
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
        header = {"BSCALE": 0.5, "BZERO": 100.0}
        t = FITSHeaderScale.from_header(header)
        assert t.bscale == 0.5
        assert t.bzero == 100.0

    def test_from_header_defaults(self):
        header = {}
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
            t = cls(**args)
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
# DopplerShift (spectral)
# ---------------------------------------------------------------------------


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
        # Roundtrip should be approximate due to resampling
        assert torch.allclose(restored, x, atol=1e-2)

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
# UpperEnvelopeContinuum — vectorized-vs-per-spectrum parity tests
# ---------------------------------------------------------------------------


def _upper_envelope_per_spectrum(
    x_flat: torch.Tensor,
    window: int,
    smooth: float,
    is_local_max: torch.Tensor,
) -> torch.Tensor:
    """Reference per-spectrum implementation of UpperEnvelopeContinuum.

    Mirrors the vectorized cummax-based algorithm using Python for-loops
    over spectra and positions.  Used to verify correctness of the
    batched cummax approach.

    Parameters
    ----------
    x_flat : [N, L]
        Flattened input tensor.
    window : int
        Half-width for local-max detection (unused here — passed for
        signature compatibility; is_local_max is pre-computed).
    smooth : float
        Gaussian sigma for optional smoothing (0 = no smoothing).
    is_local_max : [N, L]
        Boolean mask where local maxima occur (pre-computed batched).

    Returns
    -------
    continuum : [N, L]
    """
    n_spectra, length = x_flat.shape
    continuum = torch.empty_like(x_flat)

    for i in range(n_spectra):
        lm = is_local_max[i]
        lm_count = lm.sum().item()

        if lm_count < 2:
            # Fallback: use global max for this spectrum
            continuum[i] = x_flat[i].max()
            continue

        for j in range(length):
            # Find nearest local max to the left
            left_pos = float("-inf")
            for k in range(j, -1, -1):
                if lm[k]:
                    left_pos = float(k)
                    break

            # Find nearest local max to the right
            right_pos = float("-inf")
            for k in range(j, length):
                if lm[k]:
                    right_pos = float(k)
                    break

            # Clean up inf values (same logic as vectorized)
            if left_pos == float("-inf"):
                left_pos = right_pos
            if right_pos == float("-inf"):
                right_pos = left_pos
            if left_pos == float("-inf"):
                left_pos = 0.0
                right_pos = 0.0

            # Gather values and interpolate
            li = int(left_pos)
            ri = int(right_pos)
            left_val = x_flat[i, li].item()
            right_val = x_flat[i, ri].item()

            denom = right_pos - left_pos
            if denom < 1e-30:
                continuum[i, j] = left_val
            else:
                frac = (float(j) - left_pos) / denom
                continuum[i, j] = left_val + (right_val - left_val) * frac

    # Optional Gaussian smoothing (same as vectorized)
    if smooth > 0:
        half = int(math.ceil(3.0 * smooth))
        t_kernel = torch.arange(
            -half, half + 1, device=x_flat.device, dtype=x_flat.dtype
        )
        kernel = torch.exp(-0.5 * (t_kernel / smooth) ** 2)
        kernel = kernel / kernel.sum()
        kernel_1d = kernel.view(1, 1, -1)
        cont_padded = torch.nn.functional.pad(
            continuum.unsqueeze(1), (half, half), mode="reflect"
        )
        continuum = torch.nn.functional.conv1d(
            cont_padded, kernel_1d.to(device=x_flat.device, dtype=x_flat.dtype)
        ).squeeze(1)

    return continuum


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
        ref = _upper_envelope_per_spectrum(x_flat, 3, 0.0, is_lm)
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
        ref = _upper_envelope_per_spectrum(x_flat, 11, 0.0, is_lm)
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
        ref = _upper_envelope_per_spectrum(x_flat, 5, 0.0, is_lm)
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
        ref = _upper_envelope_per_spectrum(x_flat, 5, 3.0, is_lm)
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
        ref = _upper_envelope_per_spectrum(x_flat, 3, 0.0, is_lm)
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
        ref = _upper_envelope_per_spectrum(x_flat, 3, 0.0, is_lm)
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
        ref_flat = _upper_envelope_per_spectrum(x_flat, 5, 0.0, is_lm)
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
        ref = _upper_envelope_per_spectrum(x_flat, 5, 0.0, is_lm)
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
        ref = _upper_envelope_per_spectrum(x_flat, 5, 0.0, is_lm)
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
        ref = _upper_envelope_per_spectrum(x_flat, 1, 0.0, is_lm)
        ref = ref.reshape(x.shape)
        # Verify vectorized matches reference; with window=1 almost every
        # point is its own nearest local max, so continuum ≈ signal.
        assert torch.allclose(vec, ref, atol=1e-5, rtol=1e-5)


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
        dip_min_idx = dip.argmin().item()
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

    def test_repr(self):
        r = repr(AsymmetricLeastSquares(lam=1e6, p=0.05))
        assert "AsymmetricLeastSquares" in r
        assert "1000000.0" in r


# ---------------------------------------------------------------------------
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
        dip_min_idx = dip.argmin().item()
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
        header = {
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
        header = {}
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
        header = {}
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
        header = {"BITPIX": 16, "BSCALE": 0.5, "BZERO": 2000.0}
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
