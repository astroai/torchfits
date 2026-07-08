"""Tests for torchfits.transforms — ML-friendly FITS image preprocessing."""

import pytest
import torch

from torchfits.transforms import (
    ArcsinhStretch,
    BackgroundSubtract,
    Compose,
    Downsample,
    FITSHeaderNormalize,
    FITSHeaderScale,
    FITSTransform,
    Gaussian2D,
    LogStretch,
    MinMaxNormalize,
    PercentileClipNormalize,
    RobustNormalize,
    SigmaClip,
    SqrtStretch,
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
        assert err < 1e-5

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
        assert err < 1e-5

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
        assert err < 1e-5

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
        assert err < 1e-5

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
        assert err < 1e-5

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
        assert err < 1e-5

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
        assert err < 1e-5

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
# Gaussian2D
# ---------------------------------------------------------------------------


class TestGaussian2D:
    def test_smoothes_image(self):
        x = torch.zeros(1, 32, 32)
        x[0, 16, 16] = 1.0  # single bright pixel
        t = Gaussian2D(sigma=2.0)
        out = t.forward(x)
        # The peak should be spread out
        assert out[0, 16, 16] < 1.0
        assert out.sum() > 0.99  # nearly flux-conserving

    def test_circular_vs_elliptical(self):
        x = torch.zeros(1, 32, 32)
        x[0, 16, 16] = 1.0
        circ = Gaussian2D(sigma=2.0)
        ellip = Gaussian2D(sigma=2.0, sigma_y=4.0)
        out_c = circ.forward(x)
        out_e = ellip.forward(x)
        # Elliptical kernel spreads more in y, less in x
        assert out_c[0, 16, 16] != out_e[0, 16, 16]

    def test_rotation(self):
        x = torch.zeros(1, 32, 32)
        x[0, 16, 16] = 1.0
        t = Gaussian2D(sigma=2.0, sigma_y=4.0, theta=45.0)
        out = t.forward(x)
        assert torch.isfinite(out).all()

    def test_inverse_raises(self):
        t = Gaussian2D(sigma=1.0)
        with pytest.raises(RuntimeError, match="deconvolution"):
            t.inverse(torch.zeros(3, 3))

    def test_batched_4d(self):
        x = _make_tensor((4, 3, 32, 32), kind="normal")
        t = Gaussian2D(sigma=1.5)
        out = t.forward(x)
        assert out.shape == (4, 3, 32, 32)
        assert torch.isfinite(out).all()

    def test_flux_conservation(self):
        # Kernel is normalized to sum = 1.0
        t = Gaussian2D(sigma=1.5)
        assert abs(t._kernel.sum().item() - 1.0) < 1e-7
        # The interior of a uniform image is unchanged (away from edges)
        x = torch.ones(1, 64, 64)
        out = t.forward(x)
        # Interior pixels (10 pixels from each edge) should be exactly 1.0
        interior = out[0, 10:-10, 10:-10]
        assert (interior - 1.0).abs().max().item() < 1e-6

    def test_repr(self):
        t = Gaussian2D(sigma=1.5, sigma_y=2.0, theta=30.0)
        r = repr(t)
        assert "Gaussian2D" in r
        assert "1.5" in r
        assert "2.0" in r


# ---------------------------------------------------------------------------
# Downsample
# ---------------------------------------------------------------------------


class TestDownsample:
    def test_mean_downsample(self):
        x = torch.ones(1, 64, 64) * 10.0
        t = Downsample(factor=2, mode="mean")
        out = t.forward(x)
        assert out.shape == (1, 32, 32)
        # Mean pooling preserves the mean value
        assert abs(out.mean().item() - 10.0) < 1e-5

    def test_max_downsample(self):
        x = torch.zeros(1, 64, 64)
        x[0, 31, 31] = 100.0
        t = Downsample(factor=2, mode="max")
        out = t.forward(x)
        assert out.max().item() == 100.0

    def test_inverse_upsamples(self):
        x = torch.rand(1, 32, 32)
        t = Downsample(factor=2)
        down = t.forward(x)
        assert down.shape == (1, 16, 16)
        up = t.inverse(down)
        assert up.shape == (1, 32, 32)

    def test_factor_one_identity(self):
        x = torch.rand(3, 32, 32)
        t = Downsample(factor=1)
        out = t.forward(x)
        assert torch.equal(out, x)
        inv = t.inverse(x)
        assert torch.equal(inv, x)

    def test_factor_zero_raises(self):
        with pytest.raises(ValueError):
            Downsample(factor=0)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            Downsample(mode="sum")

    def test_batched(self):
        x = _make_tensor((4, 3, 64, 64), kind="uniform")
        t = Downsample(factor=4)
        out = t.forward(x)
        assert out.shape == (4, 3, 16, 16)

    def test_repr(self):
        t = Downsample(factor=4, mode="max")
        r = repr(t)
        assert "Downsample" in r
        assert "4" in r
        assert "max" in r


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
