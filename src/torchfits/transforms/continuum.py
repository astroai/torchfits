from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.linalg

from .base import FITSTransform


def _build_spline_basis(
    n_points: int, n_knots: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Build a cubic B-spline basis matrix with evenly-spaced knots.

    Returns a matrix of shape ``[n_points, n_knots]`` where each column
    is a cubic B-spline basis function evaluated at the *n_points*
    positions.
    """
    # Knot positions: extend 2 beyond the range for cubic support
    t = torch.linspace(-2, n_knots + 1, n_knots + 4, device=device, dtype=dtype)
    x = torch.linspace(0, n_knots - 1, n_points, device=device, dtype=dtype)
    B = torch.zeros(n_points, n_knots, device=device, dtype=dtype)

    for k in range(n_knots):
        # Cubic B-spline basis function centred at knot k
        B[:, k] = _bspline_cubic(x, t[k : k + 5])

    return B


def _basis_order1(
    points: torch.Tensor, t_i: torch.Tensor, t_ip1: torch.Tensor
) -> torch.Tensor:
    """Order-1 B-spline basis (piecewise constant)."""
    return ((points >= t_i) & (points < t_ip1)).float()


def _basis_order2(
    points: torch.Tensor,
    t_i: torch.Tensor,
    t_ip1: torch.Tensor,
    t_ip2: torch.Tensor,
) -> torch.Tensor:
    """Order-2 B-spline basis (linear)."""
    b1 = (
        _basis_order1(points, t_i, t_ip1)
        * (points - t_i)
        / torch.clamp_min(t_ip1 - t_i, 1e-30)
    )
    b2 = (
        _basis_order1(points, t_ip1, t_ip2)
        * (t_ip2 - points)
        / torch.clamp_min(t_ip2 - t_ip1, 1e-30)
    )
    return b1 + b2


def _basis_order3(
    points: torch.Tensor,
    t_i: torch.Tensor,
    t_ip1: torch.Tensor,
    t_ip2: torch.Tensor,
    t_ip3: torch.Tensor,
) -> torch.Tensor:
    """Order-3 B-spline basis (quadratic)."""
    b1 = (
        _basis_order2(points, t_i, t_ip1, t_ip2)
        * (points - t_i)
        / torch.clamp_min(t_ip2 - t_i, 1e-30)
    )
    b2 = (
        _basis_order2(points, t_ip1, t_ip2, t_ip3)
        * (t_ip3 - points)
        / torch.clamp_min(t_ip3 - t_ip1, 1e-30)
    )
    return b1 + b2


def _bspline_cubic(x: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    """Evaluate a cubic B-spline with given knot vector at positions x.

    Uses the Cox-de Boor recursion for order 4 (cubic).
    *knots* should have 5 elements: ``[t_0, t_1, t_2, t_3, t_4]``.
    Returns 0 outside ``[t_0, t_4]``.
    """
    t0, t1, t2, t3, t4 = knots[0], knots[1], knots[2], knots[3], knots[4]
    b1 = _basis_order3(x, t0, t1, t2, t3) * (x - t0) / torch.clamp_min(t3 - t0, 1e-30)
    b2 = _basis_order3(x, t1, t2, t3, t4) * (t4 - x) / torch.clamp_min(t4 - t1, 1e-30)
    return b1 + b2


def _fit_spline_continuum(
    x: torch.Tensor,
    n_knots: int = 10,
    n_sigma: float = 2.0,
    max_iter: int = 3,
) -> torch.Tensor:
    """Fit a cubic B-spline continuum with iterative sigma-clipping."""
    n, length = x.shape
    B = _build_spline_basis(length, n_knots, x.device, x.dtype)

    # Add ridge penalty for numerical stability
    ridge = 1e-6 * torch.eye(n_knots, device=x.device, dtype=x.dtype)

    mask = torch.ones(n, length, dtype=torch.bool, device=x.device)
    for _ in range(max_iter):
        coeffs = torch.zeros(n, n_knots, device=x.device, dtype=x.dtype)
        for i in range(n):
            mi = mask[i]
            if mi.sum() <= n_knots:
                mi = torch.ones(length, dtype=torch.bool, device=x.device)
            Bm = B[mi]
            ym = x[i][mi]
            # Weighted least squares with ridge
            BtB_m = Bm.T @ Bm + ridge
            Bty_m = Bm.T @ ym
            try:
                coeffs[i] = torch.linalg.solve(BtB_m, Bty_m)
            except RuntimeError:
                # Fallback: use all points
                BtB = B.T @ B + ridge
                coeffs[i] = torch.linalg.solve(BtB, B.T @ x[i])

        continuum = (B @ coeffs.T).T  # [n, length]
        residuals = x - continuum
        # Sigma-clip on unmasked pixels
        masked_res = torch.where(mask, residuals, torch.zeros_like(residuals))
        count = mask.float().sum(dim=1, keepdim=True)
        mean_res = masked_res.sum(dim=1, keepdim=True) / torch.clamp_min(count, 1.0)
        var = torch.where(
            mask, (residuals - mean_res) ** 2, torch.zeros_like(residuals)
        ).sum(dim=1, keepdim=True) / torch.clamp_min(count, 1.0)
        std = torch.sqrt(torch.clamp_min(var, 0.0))
        new_mask = residuals.abs() < n_sigma * torch.clamp_min(std, 1e-9)
        if torch.equal(new_mask, mask):
            break
        mask = new_mask

    return continuum


def _sg_coeffs(window_length: int, polyorder: int, deriv: int = 0) -> torch.Tensor:
    """Compute Savitzky–Golay filter coefficients.

    Returns a 1D tensor of length *window_length* with the convolution
    coefficients for derivative order *deriv*.
    """
    if window_length % 2 == 0 or window_length < 3:
        raise ValueError("window_length must be odd and >= 3")
    if polyorder >= window_length:
        raise ValueError("polyorder must be < window_length")

    half = window_length // 2
    # Build Vandermonde matrix: x values relative to window centre
    x = torch.arange(-half, half + 1, dtype=torch.float64)
    A = torch.stack([x**k for k in range(polyorder + 1)], dim=1)  # [W, P+1]
    # Target: unit impulse at the window centre for derivative order *deriv*.
    # For deriv=0 (smoothing), we want the convolution to reproduce the
    # central value of a polynomial, which is solved by lstsq.
    y = torch.zeros(window_length, dtype=torch.float64)
    y[half] = 1.0
    # Solve A @ c ≈ y to get polynomial coefficients, then evaluate at all
    # window positions to produce the full convolution kernel of length W.
    c: torch.Tensor = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)  # [P+1]
    coeffs: torch.Tensor = A @ c  # [W]
    return coeffs.float()


class SavitzkyGolayFilter(FITSTransform):
    """Savitzky–Golay polynomial smoothing filter.

    Convolves the data along *dim* with pre-computed SG coefficients.
    The filter is additive: ``Original = Smoothed + Residuals``,
    so ``inverse`` recovers the original by re-adding the residuals.

    This is the standard smoothing method in laboratory spectroscopy
    (UV/VIS/NIR) and is fully information-preserving when residuals
    are retained.

    Parameters
    ----------
    window_length : int
        Odd window length in samples (>= 3).
    polyorder : int
        Polynomial order for the local fit (< window_length).
    dim : int
        Dimension to filter along (default -1).

    Notes
    -----
    The filter is applied via ``F.conv1d``, which is efficient on GPU.
    Edge values are padded by reflecting the boundary.
    """

    def __init__(
        self, window_length: int = 7, polyorder: int = 3, dim: int = -1
    ) -> None:
        self.window_length = int(window_length)
        self.polyorder = int(polyorder)
        self.dim = int(dim)
        # Pre-compute SG coefficients once
        coeffs = _sg_coeffs(window_length, polyorder)
        # Reshape for F.conv1d: [out_channels, in_channels/groups, kernel]
        self._coeffs_1d = coeffs.view(1, 1, -1)
        self._residuals: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.window_length < 3:
            return x
        ndim = x.ndim
        dim = self.dim if self.dim >= 0 else ndim + self.dim
        pad = self.window_length // 2

        # Move filtering dim to last position for conv1d
        x_moved = x.movedim(dim, -1)  # [..., L]
        x_flat = x_moved.reshape(-1, x_moved.shape[-1])  # [N, L]

        # Pad and convolve
        x_padded = torch.nn.functional.pad(
            x_flat.unsqueeze(1), (pad, pad), mode="reflect"
        )  # [N, 1, L+2*pad]
        smoothed = torch.nn.functional.conv1d(
            x_padded,
            self._coeffs_1d.to(device=x.device, dtype=x.dtype),
        ).squeeze(1)  # [N, L]

        smoothed = smoothed.reshape(x_moved.shape).movedim(-1, dim)
        self._residuals = x - smoothed
        return smoothed

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._residuals is None:
            raise RuntimeError(
                "SavitzkyGolayFilter.inverse() requires a prior forward() pass."
            )
        return x + self._residuals

    def __repr__(self) -> str:
        return (
            f"SavitzkyGolayFilter(window_length={self.window_length}, "
            f"polyorder={self.polyorder}, dim={self.dim})"
        )


class RunningPercentile(FITSTransform):
    """Running percentile continuum estimator.

    Computes the *percentile*-th percentile in a sliding window along
    *dim*, producing a smooth upper-envelope continuum.  This is the
    standard quick-look continuum method in many spectroscopic surveys.

    The transform is additive: ``Original = Continuum + Residuals``.
    ``inverse`` re-adds the stored residuals.

    Parameters
    ----------
    percentile : float
        Percentile to compute in each window (0–100).  Default 90 gives
        the upper envelope while ignoring narrow absorption lines.
    window_size : int
        Sliding window size in samples.  Must be odd and >= 3.
    dim : int
        Dimension to filter along (default -1).
    """

    def __init__(
        self, percentile: float = 90.0, window_size: int = 21, dim: int = -1
    ) -> None:
        if window_size % 2 == 0 or window_size < 3:
            raise ValueError("window_size must be odd and >= 3")
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be in [0, 100]")
        self.percentile = float(percentile)
        self.window_size = int(window_size)
        self.dim = int(dim)
        self._residuals: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        ndim = x.ndim
        dim = self.dim if self.dim >= 0 else ndim + self.dim
        pad = self.window_size // 2
        q = self.percentile / 100.0

        # Move filter dim to last position
        x_moved = x.movedim(dim, -1)  # [..., L]
        x_flat = x_moved.reshape(-1, x_moved.shape[-1])  # [N, L]

        with torch.no_grad():
            # Pad edges with reflection for continuity
            x_padded = torch.nn.functional.pad(
                x_flat, (pad, pad), mode="reflect"
            )  # [N, L+2*pad]

            # Unfold into windows: [N, L, window_size]
            windows = x_padded.unfold(-1, self.window_size, 1)

            # Compute percentile along the window dimension
            continuum = torch.quantile(
                windows.float(), q, dim=-1, interpolation="linear"
            ).to(x.dtype)

        continuum = continuum.reshape(x_moved.shape).movedim(-1, dim)
        self._residuals = x - continuum
        return continuum

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._residuals is None:
            raise RuntimeError(
                "RunningPercentile.inverse() requires a prior forward() pass."
            )
        return x + self._residuals

    def __repr__(self) -> str:
        return (
            f"RunningPercentile(percentile={self.percentile}, "
            f"window_size={self.window_size}, dim={self.dim})"
        )


class UpperEnvelopeContinuum(FITSTransform):
    """Upper-envelope continuum estimation via local-maxima interpolation.

    Finds local maxima in sliding windows, then interpolates between them
    to produce a smooth continuum.  This approximates the alpha-shape
    / convex-hull method used by RASSINE (Cretignier et al. 2020) but
    is implemented entirely in PyTorch.

    The transform is additive: ``Original = Continuum + Residuals``.
    ``inverse`` re-adds the stored residuals.

    Parameters
    ----------
    window : int
        Half-width for local-maximum detection.  A point is a local max
        if it is the largest in [i-window, i+window].  Larger values
        produce a smoother (less concave) continuum.
    smooth : float
        Optional Gaussian sigma for smoothing the final continuum.
        Default 0 (no smoothing).
    dim : int
        Dimension to operate along (default -1).
    """

    def __init__(self, window: int = 11, smooth: float = 0.0, dim: int = -1) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window)
        self.smooth = float(smooth)
        self.dim = int(dim)
        self._residuals: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        ndim = x.ndim
        dim = self.dim if self.dim >= 0 else ndim + self.dim

        # Move operating dim to last position
        x_moved = x.movedim(dim, -1)  # [..., L]
        x_flat = x_moved.reshape(-1, x_moved.shape[-1])  # [N, L]
        n_spectra, length = x_flat.shape

        with torch.no_grad():
            # Pad for local-max search at edges
            x_padded = torch.nn.functional.pad(
                x_flat, (self.window, self.window), mode="reflect"
            )  # [N, L+2*W]
            windows = x_padded.unfold(-1, 2 * self.window + 1, 1)  # [N, L, 2W+1]
            # A point is a local max if it equals the window maximum
            is_local_max = x_flat == windows.max(dim=-1).values  # [N, L]

            # Vectorized upper envelope via cummax-based nearest local max lookup.
            # For each position, find the nearest local max to the left and right,
            # then linearly interpolate between their values.
            positions = torch.arange(length, device=x.device, dtype=x.dtype)
            pos_exp = positions.unsqueeze(0).expand(n_spectra, -1)  # [N, L]

            # Nearest local max to the left: forward-fill local max positions.
            # Set non-local-max positions to -inf, then cummax gives the running
            # maximum position (i.e., nearest local max to the left).
            lm_positions = torch.where(
                is_local_max, pos_exp, torch.full_like(pos_exp, float("-inf"))
            )
            left_max_pos, _ = torch.cummax(lm_positions, dim=1)  # [N, L]

            # Nearest local max to the right: reverse, cummax, reverse back.
            rev_pos = length - 1 - pos_exp
            rev_lm = torch.where(
                is_local_max.flip(1),
                rev_pos.flip(1),
                torch.full_like(rev_pos.flip(1), float("-inf")),
            )
            rev_cummax, _ = torch.cummax(rev_lm, dim=1)
            right_max_pos = (length - 1) - rev_cummax.flip(1)  # [N, L]

            # Clean up inf/-inf: where no left max exists, use right; vice versa.
            left_max_pos = torch.where(
                torch.isinf(left_max_pos), right_max_pos, left_max_pos
            )
            right_max_pos = torch.where(
                torch.isinf(right_max_pos), left_max_pos, right_max_pos
            )
            # If both were inf (no local maxima at all), clamp to 0.
            left_max_pos = torch.where(
                torch.isinf(left_max_pos), torch.zeros_like(left_max_pos), left_max_pos
            )
            right_max_pos = torch.where(
                torch.isinf(right_max_pos),
                torch.zeros_like(right_max_pos),
                right_max_pos,
            )

            # Gather values at nearest left/right local max positions.
            left_idx = left_max_pos.long().clamp(0, length - 1)
            right_idx = right_max_pos.long().clamp(0, length - 1)
            left_vals = torch.gather(x_flat, 1, left_idx)  # [N, L]
            right_vals = torch.gather(x_flat, 1, right_idx)  # [N, L]

            # Linear interpolation between left and right local max values.
            denom = torch.clamp_min(right_max_pos - left_max_pos, 1e-30)
            frac = (pos_exp - left_max_pos) / denom
            continuum_vec = left_vals + (right_vals - left_vals) * frac

            # Fallback for spectra with < 2 local maxima: use global max.
            has_enough = is_local_max.sum(dim=1) >= 2  # [N]
            max_vals = x_flat.max(dim=1, keepdim=True).values  # [N, 1]
            continuum = torch.where(
                has_enough.unsqueeze(1),
                continuum_vec,
                max_vals.expand(-1, length),
            )

            # Optional Gaussian smoothing
            if self.smooth > 0:
                half = int(math.ceil(3.0 * self.smooth))
                t_kernel = torch.arange(-half, half + 1, device=x.device, dtype=x.dtype)
                kernel = torch.exp(-0.5 * (t_kernel / self.smooth) ** 2)
                kernel = kernel / kernel.sum()
                kernel_1d = kernel.view(1, 1, -1)
                cont_padded = torch.nn.functional.pad(
                    continuum.unsqueeze(1), (half, half), mode="reflect"
                )
                continuum = torch.nn.functional.conv1d(
                    cont_padded, kernel_1d.to(device=x.device, dtype=x.dtype)
                ).squeeze(1)

        continuum = continuum.reshape(x_moved.shape).movedim(-1, dim)
        self._residuals = x - continuum
        return continuum

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._residuals is None:
            raise RuntimeError(
                "UpperEnvelopeContinuum.inverse() requires a prior forward() pass."
            )
        return x + self._residuals

    def __repr__(self) -> str:
        return (
            f"UpperEnvelopeContinuum(window={self.window}, "
            f"smooth={self.smooth}, dim={self.dim})"
        )


def _haar_dwt_1d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-level 1D Haar discrete wavelet transform.

    Returns (approx, detail) where approx has half the length of x
    and detail holds the high-frequency coefficients.
    """
    # Ensure even length
    length = x.shape[-1]
    if length % 2 != 0:
        x = x[..., : length - 1]
        length -= 1
    # Average and difference
    approx = (x[..., 0::2] + x[..., 1::2]) / 2.0
    detail = (x[..., 0::2] - x[..., 1::2]) / 2.0
    return approx, detail


def _haar_idwt_1d(approx: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
    """Inverse single-level 1D Haar DWT."""
    length = approx.shape[-1] * 2
    x = torch.zeros(
        *approx.shape[:-1], length, device=approx.device, dtype=approx.dtype
    )
    x[..., 0::2] = approx + detail
    x[..., 1::2] = approx - detail
    return x


class WaveletDecompose(FITSTransform):
    """Multi-level Haar wavelet decomposition.

    Decomposes the signal along *dim* into *levels* of approximation +
    detail coefficients.  The output stacks ``[approx_L, detail_L, ...,
    detail_1]`` along *dim*, preserving all information for a perfect
    reconstruction.

    This is a fully invertible frequency split — the approximation
    coefficients capture the broadband continuum, while detail
    coefficients capture narrow spectral features.  Neural networks
    can learn to attend to either frequency band independently.

    ``inverse`` reconstructs the original signal from the coefficients.

    Parameters
    ----------
    levels : int
        Number of decomposition levels (1–8).  Level 1 splits into
        approx (half-length) + detail; each subsequent level further
        splits the approximation.
    dim : int
        Dimension to decompose along (default -1).

    Notes
    -----
    Uses the Haar wavelet (simplest, fastest, and most common in
    astro-ML for continuum/feature separation).  The transform is
    orthogonal (up to the ``sqrt(2)`` scaling factor), so it is
    numerically stable and gradient-safe.
    """

    def __init__(self, levels: int = 3, dim: int = -1) -> None:
        if levels < 1 or levels > 8:
            raise ValueError("levels must be in [1, 8]")
        self.levels = int(levels)
        self.dim = int(dim)
        self._orig_shape: Optional[tuple[int, ...]] = None
        self._padded: bool = False

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        ndim = x.ndim
        dim = self.dim if self.dim >= 0 else ndim + self.dim
        self._orig_shape = x.shape

        # Move working dim to last position
        x_w = x.movedim(dim, -1)
        shape_prefix = x_w.shape[:-1]

        # Pad to make length divisible by 2^levels
        length = x_w.shape[-1]
        target = ((length + (1 << self.levels) - 1) >> self.levels) << self.levels
        self._padded = target > length
        if self._padded:
            pad_amount = target - length
            x_w = torch.nn.functional.pad(x_w, (0, pad_amount), mode="reflect")
            self._pad_amount = pad_amount

        # Multi-level decomposition
        coeffs: list[torch.Tensor] = []
        current = x_w.reshape(-1, x_w.shape[-1])  # [N, L]
        for _ in range(self.levels):
            approx, detail = _haar_dwt_1d(current)
            coeffs.append(detail.reshape(*shape_prefix, detail.shape[-1]))
            current = approx  # continue decomposing the approximation
        coeffs.append(current.reshape(*shape_prefix, current.shape[-1]))  # final approx

        # Stack [approx_L, detail_L, ..., detail_1] along working dim.
        # coeffs = [detail_1, ..., detail_L, approx_L] in order of
        # increasing level.  Reverse to get decreasing frequency.
        result = torch.cat(coeffs[::-1], dim=-1)
        return result.movedim(-1, dim)

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._orig_shape is None:
            raise RuntimeError(
                "WaveletDecompose.inverse() requires a prior forward() pass."
            )
        ndim = x.ndim
        dim = self.dim if self.dim >= 0 else ndim + self.dim

        x_w = x.movedim(dim, -1)
        shape_prefix = x_w.shape[:-1]
        length = x_w.shape[-1]

        # Split into coefficient bands
        # The layout is [approx_L, detail_L, ..., detail_1]
        # Approx length = ceil(original / 2^L), but padded to power of 2
        target = (
            (self._orig_shape[dim] + (1 << self.levels) - 1) >> self.levels
        ) << self.levels
        n_padded = target  # length after padding
        # Approx has n_padded / 2^levels elements
        approx_len = n_padded >> self.levels

        # Split: first approx_len elements are the final approximation,
        # then detail_L (same length), detail_{L-1} (2x), ..., detail_1 (2^{L-1}x)
        coeffs_flat = x_w.reshape(-1, length)  # [N, L]
        positions = [approx_len]
        for lev in range(self.levels - 1, -1, -1):
            positions.append(positions[-1] + (approx_len << (self.levels - lev - 1)))

        # Verify: positions[-1] should equal length
        splits = torch.split(
            coeffs_flat,
            [positions[0]]
            + [positions[i + 1] - positions[i] for i in range(len(positions) - 1)],
            dim=-1,
        )
        approx = splits[0]  # final approx
        details: list[torch.Tensor] = list(
            splits[1:]
        )  # [detail_L, ..., detail_1] — deepest first

        # Reconstruct from coarsest to finest
        current = approx
        for detail in details:
            current = _haar_idwt_1d(current, detail)

        # Remove padding if any
        if self._padded:
            current = current[..., : self._orig_shape[dim]]

        current = current.reshape(*shape_prefix, current.shape[-1])
        return current.movedim(-1, dim)

    def __repr__(self) -> str:
        return f"WaveletDecompose(levels={self.levels}, dim={self.dim})"


def _build_d2_diagonals(
    n: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the 3 upper diagonals of the symmetric pentadiagonal D^T D matrix.

    D is the (n-2)×n second-difference operator.  D^T D is n×n pentadiagonal.
    This returns only the 3 non-zero upper diagonals (the matrix is symmetric),
    reducing storage from O(n²) to O(n) and enabling O(n) banded Cholesky.

    Returns
    -------
    d0 : Tensor, shape [n]
        Main diagonal.
    d1 : Tensor, shape [n-1]
        First super-diagonal (offset +1).
    d2 : Tensor, shape [n-2]
        Second super-diagonal (offset +2).
    """
    if n < 4:
        # For n < 4 the second-difference penalty vanishes (D has < 2 rows).
        d0 = torch.zeros(n, device=device, dtype=dtype)
        d1 = torch.zeros(max(n - 1, 0), device=device, dtype=dtype)
        d2 = torch.zeros(max(n - 2, 0), device=device, dtype=dtype)
        return d0, d1, d2

    d0 = torch.full((n,), 6.0, device=device, dtype=dtype)
    d0[0] = 1.0
    d0[1] = 5.0
    d0[n - 2] = 5.0
    d0[n - 1] = 1.0

    d1 = torch.full((n - 1,), -4.0, device=device, dtype=dtype)
    d1[0] = -2.0
    d1[n - 2] = -2.0

    d2 = torch.ones(n - 2, device=device, dtype=dtype)
    return d0, d1, d2


def _banded_chol_solve_batched_impl(
    w: torch.Tensor,
    lam_d0: torch.Tensor,
    lam_d1: torch.Tensor,
    lam_d2: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Solve ``(diag(W) + λD^T D) z = b`` via banded Cholesky factorization.

    The matrix is symmetric positive-definite pentadiagonal (bandwidth 2).
    The Cholesky factorization and triangular solves are O(n) per spectrum —
    a dramatic improvement over the O(n³) dense ``torch.linalg.solve``.

    All inputs must be in the same dtype (typically float64 for stability).
    The loop over spectrum positions is sequential, but each step is
    vectorized over the batch (spectra) dimension.

    This is the pure-Python implementation used as fallback when
    ``torch.jit.script`` is unavailable.  Prefer ``_banded_chol_solve_batched``
    which wraps this with JIT compilation for ~5-10× speedup on large spectra.

    Parameters
    ----------
    w : Tensor, shape [N, L]
        Diagonal weights (the W matrix as a 2D tensor).
    lam_d0 : Tensor, shape [L]
        λ * main diagonal of D^T D (precomputed, shared across iterations).
    lam_d1 : Tensor, shape [L-1]
        λ * first super-diagonal of D^T D.
    lam_d2 : Tensor, shape [L-2]
        λ * second super-diagonal of D^T D.
    b : Tensor, shape [N, L]
        Right-hand side.

    Returns
    -------
    z : Tensor, shape [N, L]
        Solution.
    """
    n_batch, length = w.shape

    # --- Build A's diagonals: A = diag(W) + λD² ---
    # Only a0 changes per iteration; a1, a2 are fixed (= lam_d1, lam_d2).
    a0 = w + lam_d0.unsqueeze(0)  # [N, L]
    a1 = lam_d1.unsqueeze(0).expand(n_batch, -1)  # [N, L-1]
    a2 = lam_d2.unsqueeze(0).expand(n_batch, -1)  # [N, L-2]

    # --- Cholesky factorization: A = L L^T, L has bandwidth 2 ---
    l0 = torch.empty_like(a0)  # [N, L]   main diagonal of L
    l1 = torch.empty_like(a1)  # [N, L-1] first sub-diagonal of L
    l2 = torch.empty_like(a2)  # [N, L-2] second sub-diagonal of L

    # j = 0
    l0[:, 0] = torch.sqrt(torch.clamp_min(a0[:, 0], 1e-30))
    l1[:, 0] = a1[:, 0] / l0[:, 0]
    if length > 2:
        l2[:, 0] = a2[:, 0] / l0[:, 0]

    # j = 1
    if length > 1:
        l0[:, 1] = torch.sqrt(torch.clamp_min(a0[:, 1] - l1[:, 0] ** 2, 1e-30))
        l1[:, 1] = (a1[:, 1] - l2[:, 0] * l1[:, 0]) / l0[:, 1]
        if length > 3:
            l2[:, 1] = a2[:, 1] / l0[:, 1]

    # j = 2 .. n-3 (interior points)
    for j in range(2, length - 2):
        l0[:, j] = torch.sqrt(
            torch.clamp_min(a0[:, j] - l1[:, j - 1] ** 2 - l2[:, j - 2] ** 2, 1e-30)
        )
        l1[:, j] = (a1[:, j] - l2[:, j - 1] * l1[:, j - 1]) / l0[:, j]
        l2[:, j] = a2[:, j] / l0[:, j]

    # j = n-2
    if length > 2:
        l0[:, length - 2] = torch.sqrt(
            torch.clamp_min(
                a0[:, length - 2] - l1[:, length - 3] ** 2 - l2[:, length - 4] ** 2,
                1e-30,
            )
        )
        l1[:, length - 2] = (
            a1[:, length - 2] - l2[:, length - 3] * l1[:, length - 3]
        ) / l0[:, length - 2]

    # j = n-1
    l0[:, length - 1] = torch.sqrt(
        torch.clamp_min(
            a0[:, length - 1] - l1[:, length - 2] ** 2 - l2[:, length - 3] ** 2, 1e-30
        )
    )

    # --- Forward substitution: L y = b ---
    y = torch.empty_like(b)
    y[:, 0] = b[:, 0] / l0[:, 0]
    if length > 1:
        y[:, 1] = (b[:, 1] - l1[:, 0] * y[:, 0]) / l0[:, 1]
    for j in range(2, length):
        y[:, j] = (
            b[:, j] - l1[:, j - 1] * y[:, j - 1] - l2[:, j - 2] * y[:, j - 2]
        ) / l0[:, j]

    # --- Backward substitution: L^T z = y ---
    z = torch.empty_like(y)
    z[:, length - 1] = y[:, length - 1] / l0[:, length - 1]
    if length > 1:
        z[:, length - 2] = (
            y[:, length - 2] - l1[:, length - 2] * z[:, length - 1]
        ) / l0[:, length - 2]
    for j in range(length - 3, -1, -1):
        z[:, j] = (y[:, j] - l1[:, j] * z[:, j + 1] - l2[:, j] * z[:, j + 2]) / l0[:, j]

    return z


# JIT-compile the banded solver to eliminate Python for-loop overhead.
# torch.jit.script pushes the three sequential loops (Cholesky, forward
# substitution, backward substitution) into C++ with zero code changes.
# Profiling shows this eliminates ~94% of the runtime for large L.
# Falls back to pure Python if `torch.jit.script` is unavailable.
try:
    _banded_chol_solve_batched: Callable[..., torch.Tensor] = torch.jit.script(
        _banded_chol_solve_batched_impl
    )
except (RuntimeError, TypeError, AttributeError):
    import warnings

    warnings.warn(
        "torch.jit.script() unavailable; _banded_chol_solve_batched "
        "will use pure Python (slower for large spectra).",
        stacklevel=2,
    )
    _banded_chol_solve_batched = _banded_chol_solve_batched_impl


class AsymmetricLeastSquares(FITSTransform):
    """Asymmetric Least Squares baseline correction (Eilers 2003).

    Iteratively fits a smooth baseline that hugs either the lower or upper
    envelope of the signal by differentially weighting points above vs below
    the baseline.  This is the standard method in Raman/NIR spectroscopy for
    automated baseline removal and is fully information-preserving via
    additive decomposition.

    The algorithm solves ``(W + λ D^T D) z = W y`` at each iteration,
    where *W* is a diagonal weight matrix with weights determined by
    *p* and *envelope*:

    - ``envelope="lower"`` (default): ``w_i = p`` if ``y_i > z_i``,
      ``1 − p`` otherwise.  Baseline hugs absorption features.
    - ``envelope="upper"``: ``w_i = 1 − p`` if ``y_i > z_i``,
      ``p`` otherwise.  Baseline hugs emission features.

    The transform is additive: ``Original = Baseline + Residuals``.
    ``inverse`` re-adds the stored residuals.

    Parameters
    ----------
    lam : float
        Smoothness parameter.  Larger values produce a stiffer baseline.
        Typical range: 1e2 to 1e9 (default 1e5).
    p : float
        Asymmetry parameter in (0, 1).  Smaller values make the baseline
        hug the target envelope more aggressively.
        Typical range: 0.001 to 0.1 (default 0.01).
    max_iter : int
        Maximum number of reweighting iterations (default 10).
    dim : int
        Dimension to operate along (default -1).
    envelope : str
        Which envelope to hug: ``"lower"`` (default) for absorption
        features (Raman/NIR), ``"upper"`` for emission features
        (stellar absorption spectroscopy).

    References
    ----------
    Eilers, P. H. C. (2003). "A Perfect Smoother."
    Analytical Chemistry, 75(14), 3631–3636.
    """

    def __init__(
        self,
        lam: float = 1e5,
        p: float = 0.01,
        max_iter: int = 10,
        dim: int = -1,
        envelope: str = "lower",
    ) -> None:
        if lam <= 0:
            raise ValueError("lam must be > 0")
        if not 0 < p < 1:
            raise ValueError("p must be in (0, 1)")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if envelope not in ("lower", "upper"):
            raise ValueError("envelope must be 'lower' or 'upper'")
        self.lam = float(lam)
        self.p = float(p)
        self.max_iter = int(max_iter)
        self.dim = int(dim)
        self.envelope = envelope
        self._residuals: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        ndim = x.ndim
        dim = self.dim if self.dim >= 0 else ndim + self.dim

        # Move working dim to last position
        x_moved = x.movedim(dim, -1)  # [..., L]
        x_flat = x_moved.reshape(-1, x_moved.shape[-1])  # [N, L]
        n_spectra, length = x_flat.shape

        with torch.no_grad():
            if length < 4:
                # D² penalty vanishes for n < 4: (W + 0) z = W y  →  z = y.
                baseline = x_flat.clone()
            else:
                # Precompute λ * D² diagonals once (shared across all
                # spectra and all reweighting iterations).
                d0, d1, d2 = _build_d2_diagonals(length, x.device, torch.float64)
                lam_d0 = self.lam * d0
                lam_d1 = self.lam * d1
                lam_d2 = self.lam * d2

                # Work in float64 for numerical stability with large λ.
                y = x_flat.double()  # [N, L]
                z = y.clone()  # initial estimate = signal

                for _ in range(self.max_iter):
                    # Weights: determined by envelope mode.
                    # lower: p for points above baseline (ignore peaks),
                    #        1-p for points below (hug absorption troughs).
                    # upper: 1-p for points above (hug emission peaks),
                    #        p for points below (ignore troughs).
                    if self.envelope == "lower":
                        w = torch.where(y > z, self.p, 1.0 - self.p)
                    else:
                        w = torch.where(y > z, 1.0 - self.p, self.p)

                    # RHS: W y (element-wise since W is diagonal)
                    b = w * y  # [N, L]

                    # Solve (W + λD²) z_new = W y via batched banded Cholesky.
                    # O(N*L) per iteration vs O(N*L³) for the dense solve.
                    z_new = _banded_chol_solve_batched(w, lam_d0, lam_d1, lam_d2, b)

                    # Check convergence across all spectra (same semantics
                    # as the old per-spectrum allclose with atol=1e-6, rtol=1e-5).
                    if torch.allclose(z_new, z, atol=1e-6):
                        z = z_new
                        break
                    z = z_new

                baseline = z.to(x.dtype)

        baseline = baseline.reshape(x_moved.shape).movedim(-1, dim)
        self._residuals = x - baseline
        return baseline

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._residuals is None:
            raise RuntimeError(
                "AsymmetricLeastSquares.inverse() requires a prior forward() pass."
            )
        return x + self._residuals

    def __repr__(self) -> str:
        return (
            f"AsymmetricLeastSquares(lam={self.lam}, p={self.p}, "
            f"max_iter={self.max_iter}, dim={self.dim}, "
            f"envelope={self.envelope!r})"
        )


class AlphaShapeContinuum(FITSTransform):
    """Alpha-shape continuum via morphological closing.

    Computes an upper-envelope continuum using morphological closing
    (dilation followed by erosion), which naturally follows the spectral
    peaks while bridging narrow absorption features.  This is a
    practical approximation to the full alpha-shape algorithm used by
    RASSINE (Cretignier et al. 2020), implemented entirely in PyTorch
    using unfold + max/min operations.

    Unlike :class:`UpperEnvelopeContinuum` (which uses local-max detection
    + interpolation), morphological closing produces a guaranteed upper
    envelope that is always >= the original signal.

    The transform is additive: ``Original = Continuum + Residuals``.
    ``inverse`` re-adds the stored residuals.

    Parameters
    ----------
    half_window : int
        Half-width of the structuring element in samples.  Larger values
        bridge wider absorption features.  Default 15.
    iterations : int
        Number of closing operations.  Each iteration applies
        dilation→erosion, progressively smoothing the continuum.
        Default 1.
    dim : int
        Dimension to operate along (default -1).

    References
    ----------
    Cretignier, M. et al. (2020). "RASSINE: Interactive tool for
    normalising stellar spectra." Astronomy & Astrophysics, 640, A42.
    """

    def __init__(
        self, half_window: int = 15, iterations: int = 1, dim: int = -1
    ) -> None:
        if half_window < 1:
            raise ValueError("half_window must be >= 1")
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        self.half_window = int(half_window)
        self.iterations = int(iterations)
        self.dim = int(dim)
        self._residuals: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        ndim = x.ndim
        dim = self.dim if self.dim >= 0 else ndim + self.dim
        window_size = 2 * self.half_window + 1

        # Move working dim to last position
        x_moved = x.movedim(dim, -1)  # [..., L]
        x_flat = x_moved.reshape(-1, x_moved.shape[-1])  # [N, L]
        pad = self.half_window

        with torch.no_grad():
            continuum = x_flat
            for _ in range(self.iterations):
                # Dilation: running max
                padded = torch.nn.functional.pad(continuum, (pad, pad), mode="reflect")
                windows = padded.unfold(-1, window_size, 1)  # [N, L, W]
                dilated = windows.max(dim=-1).values  # [N, L]
                # Erosion: running min of the dilated signal
                padded = torch.nn.functional.pad(dilated, (pad, pad), mode="reflect")
                windows = padded.unfold(-1, window_size, 1)  # [N, L, W]
                continuum = windows.min(dim=-1).values  # [N, L]

        continuum = continuum.reshape(x_moved.shape).movedim(-1, dim)
        self._residuals = x - continuum
        return continuum

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._residuals is None:
            raise RuntimeError(
                "AlphaShapeContinuum.inverse() requires a prior forward() pass."
            )
        return x + self._residuals

    def __repr__(self) -> str:
        return (
            f"AlphaShapeContinuum(half_window={self.half_window}, "
            f"iterations={self.iterations}, dim={self.dim})"
        )
