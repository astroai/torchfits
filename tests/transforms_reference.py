"""Reference implementations of torchfits transforms for parity testing.

Each function mirrors a vectorized (batched) PyTorch implementation using
explicit per-spectrum/per-position Python for-loops, making them suitable
for verifying correctness of performant production code.

These are **test utilities only** — they are deliberately slow and allocate
freely to maximise clarity and auditability.
"""

from __future__ import annotations

import math

import torch

from torchfits.transforms.continuum import _build_d2_diagonals, _build_spline_basis
from torchfits.transforms.helpers import (
    _flatten_dims,
    _median,
    _normalize_dims,
    _unflatten_result,
)


def upper_envelope_per_spectrum(
    x_flat: torch.Tensor,
    is_local_max: torch.Tensor,
    *,
    smooth: float = 0.0,
) -> torch.Tensor:
    """Reference per-spectrum implementation of UpperEnvelopeContinuum.

    Mirrors the vectorized cummax-based algorithm using Python for-loops
    over spectra and positions.  Used to verify correctness of the
    batched cummax approach.

    Parameters
    ----------
    x_flat : [N, L]
        Flattened input tensor.
    is_local_max : [N, L]
        Boolean mask where local maxima occur (pre-computed via
        ``F.pad(…, mode='reflect').unfold(…).max(…)``).
    smooth : float
        Gaussian sigma for optional continuum smoothing (0 = no smoothing).

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


def phase_fold_per_bin(
    x_flat: torch.Tensor,
    n_bins: int,
    bin_idx: torch.Tensor,
) -> torch.Tensor:
    """Reference per-bin loop for PhaseFold.

    Mirrors the scatter_add_ + bincount vectorized approach using an
    explicit per-bin Python for-loop: for each bin, mask the values,
    sum them, and divide by the count.

    *bin_idx* is pre-computed externally (typically via
    ``PhaseFold.forward`` bin computation logic).
    """
    n_samples, _length = x_flat.shape
    folded = torch.zeros(n_samples, n_bins, device=x_flat.device, dtype=x_flat.dtype)
    for b in range(n_bins):
        mask = bin_idx == b
        count = mask.sum().item()
        if count > 0:
            folded[:, b] = x_flat[:, mask].sum(dim=1) / float(count)
    return folded


def sigma_clip_naive(
    x: torch.Tensor,
    n_sigma: float,
    max_iter: int,
    dims: tuple[int, ...],
    fill: str,
) -> torch.Tensor:
    """Reference naive SigmaClip: allocates fresh tensors each iteration.

    Mirrors the zero-alloc buffer implementation using the simpler
    (less memory-efficient) approach of freshly allocating zeros
    and intermediate tensors in every iteration.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    n_sigma : float
        Number of standard deviations for clipping threshold.
    max_iter : int
        Maximum number of clipping iterations.
    dims : tuple[int, ...]
        Dimensions along which statistics are computed; empty for global.
    fill : str
        Fill value for clipped pixels: ``"mean"`` or ``"median"``.

    Returns
    -------
    torch.Tensor
        Clipped tensor with same shape and dtype as *x*.
    """
    ndim = x.ndim
    dims_norm: tuple[int, ...] = ()
    if len(dims) > 0:
        dims_norm = _normalize_dims(ndim, dims)

    mask = torch.ones_like(x, dtype=torch.bool)

    for _ in range(max_iter):
        # Compute mean of unmasked values (naive: fresh allocations)
        zeros = torch.zeros_like(x)
        masked_values = torch.where(mask, x, zeros)
        mask_f = mask.to(x.dtype)

        if len(dims_norm) > 0:
            x_flat = _flatten_dims(masked_values, dims_norm)
            c_flat = _flatten_dims(mask_f, dims_norm)
            total_sum = x_flat.sum(dim=-1, keepdim=True)
            total_cnt = c_flat.sum(dim=-1, keepdim=True)
            mean_v = total_sum / torch.clamp_min(total_cnt, 1.0)
            mean_v_full = _unflatten_result(mean_v, x.shape, dims_norm)

            # Compute std: diff^2 only on unmasked pixels
            diff_sq = (x - mean_v_full) ** 2
            var_sum = torch.where(mask, diff_sq, zeros)
            d_flat = _flatten_dims(var_sum, dims_norm)
            var = d_flat.sum(dim=-1, keepdim=True) / torch.clamp_min(total_cnt, 1.0)
            std_v_full = _unflatten_result(
                torch.sqrt(torch.clamp_min(var, 0.0)), x.shape, dims_norm
            )
        else:
            cnt = mask_f.sum()
            mean_scalar_val = (masked_values.sum() / max(cnt.item(), 1.0)).item()
            mean_v_full = torch.full_like(x, mean_scalar_val)
            diff_sq = (x - mean_v_full) ** 2
            var = torch.where(mask, diff_sq, zeros).sum() / max(cnt.item(), 1.0)
            std_scalar = math.sqrt(max(var.item(), 0.0))
            std_v_full = torch.full_like(x, std_scalar)

        new_mask = (x >= mean_v_full - n_sigma * std_v_full) & (
            x <= mean_v_full + n_sigma * std_v_full
        )
        if torch.equal(new_mask, mask):
            break
        mask = new_mask

    # Fill clipped values
    if fill == "mean":
        zeros = torch.zeros_like(x)
        masked_values = torch.where(mask, x, zeros)
        mask_f = mask.to(x.dtype)
        if len(dims_norm) > 0:
            xf = _flatten_dims(masked_values, dims_norm)
            cf = _flatten_dims(mask_f, dims_norm)
            fill_val = _unflatten_result(
                xf.sum(dim=-1, keepdim=True)
                / torch.clamp_min(cf.sum(dim=-1, keepdim=True), 1.0),
                x.shape,
                dims_norm,
            )
        else:
            cnt = mask_f.sum()
            fill_val = masked_values.sum() / max(cnt.item(), 1.0)
    else:
        fill_val = _median(
            torch.where(
                mask,
                x,
                torch.tensor(float("inf"), device=x.device, dtype=x.dtype),
            ),
            dims_norm if dims_norm else (-1,),
        )
        fill_val = torch.where(
            torch.isinf(fill_val), torch.zeros_like(fill_val), fill_val
        )

    return torch.where(mask, x, fill_val)


def alpha_shape_per_spectrum(
    x_flat: torch.Tensor,
    half_window: int,
    iterations: int,
) -> torch.Tensor:
    """Reference per-spectrum dilation/erosion loop for AlphaShapeContinuum.

    Mirrors the vectorized unfold/max/min morphological closing using
    per-spectrum, per-position Python for-loops with explicit reflection
    padding at the edges.

    Parameters
    ----------
    x_flat : [N, L]
        Flattened input tensor.
    half_window : int
        Half-width of the structuring element.
    iterations : int
        Number of dilation→erosion closing operations.

    Returns
    -------
    continuum : [N, L]
    """
    n_spectra, length = x_flat.shape
    continuum = x_flat.clone()

    for _ in range(iterations):
        # Dilation: running max over window [j - hw, j + hw]
        dilated = torch.empty_like(continuum)
        for i in range(n_spectra):
            for j in range(length):
                vals = []
                for k in range(j - half_window, j + half_window + 1):
                    # Reflect padding
                    if k < 0:
                        idx = -k
                    elif k >= length:
                        idx = 2 * length - k - 2
                    else:
                        idx = k
                    # Clamp to valid range (belt-and-suspenders)
                    idx = max(0, min(idx, length - 1))
                    vals.append(continuum[i, idx].item())
                dilated[i, j] = max(vals)

        # Erosion: running min of dilated signal over same window
        eroded = torch.empty_like(dilated)
        for i in range(n_spectra):
            for j in range(length):
                vals = []
                for k in range(j - half_window, j + half_window + 1):
                    if k < 0:
                        idx = -k
                    elif k >= length:
                        idx = 2 * length - k - 2
                    else:
                        idx = k
                    idx = max(0, min(idx, length - 1))
                    vals.append(dilated[i, idx].item())
                eroded[i, j] = min(vals)

        continuum = eroded

    return continuum


def asls_dense_solve(
    x_flat: torch.Tensor,
    lam: float,
    p: float,
    max_iter: int,
    envelope: str = "lower",
) -> torch.Tensor:
    """Reference dense-matrix solver for AsymmetricLeastSquares.

    Mirrors the production banded-Cholesky algorithm using
    ``torch.linalg.solve`` on the dense pentadiagonal matrix
    ``A = diag(W) + λ D^T D`` at each iteration.  This is O(L³)
    per spectrum rather than O(L), but serves as an unambiguous
    correctness check for the banded solver.

    Implements the same iterative reweighting logic as
    :class:`~torchfits.transforms.AsymmetricLeastSquares`:

    1. Start with ``z = y``.
    2. Compute weights ``w_i = p`` if ``y_i > z_i`` else ``1 - p``.
    3. Solve ``(diag(w) + λ D^T D) z_new = w * y``.
    4. Repeat until convergence or *max_iter*.

    Parameters
    ----------
    x_flat : [N, L]
        Flattened input tensor (in float32 or float64).
    lam : float
        Smoothness parameter.
    p : float
        Asymmetry parameter in (0, 1).
    max_iter : int
        Maximum number of reweighting iterations.

    Returns
    -------
    baseline : [N, L]
        Estimated baseline in the same dtype as *x_flat*.
    """
    n_spectra, length = x_flat.shape

    if length < 4:
        return x_flat.clone()

    # Build the D^T D pentadiagonal matrix from the production diagonals.
    d0, d1, d2 = _build_d2_diagonals(length, x_flat.device, torch.float64)
    d2_dense = torch.zeros(length, length, device=x_flat.device, dtype=torch.float64)
    for i in range(length):
        d2_dense[i, i] = d0[i]
    for i in range(length - 1):
        d2_dense[i, i + 1] = d1[i]
        d2_dense[i + 1, i] = d1[i]
    for i in range(length - 2):
        d2_dense[i, i + 2] = d2[i]
        d2_dense[i + 2, i] = d2[i]

    lam_d2 = lam * d2_dense  # [L, L]

    # Work in float64 for stability
    y = x_flat.double()  # [N, L]
    z = y.clone()

    for _ in range(max_iter):
        w = torch.where(
            y > z,
            p if envelope == "lower" else 1.0 - p,
            1.0 - p if envelope == "lower" else p,
        )  # [N, L]

        # Solve for each spectrum independently (dense solve, O(L³))
        z_new = torch.empty_like(z)
        for i in range(n_spectra):
            A = torch.diag(w[i]) + lam_d2  # [L, L]
            b = w[i] * y[i]  # [L]
            z_new[i] = torch.linalg.solve(A, b)

        if torch.allclose(z_new, z, atol=1e-6):
            z = z_new
            break
        z = z_new

    return z.to(x_flat.dtype)


def running_percentile_per_spectrum(
    x_flat: torch.Tensor,
    percentile: float,
    window_size: int,
) -> torch.Tensor:
    """Reference per-spectrum, per-position percentile loop.

    Mirrors the vectorized unfold + ``torch.quantile`` implementation
    using explicit per-spectrum, per-position Python for-loops with
    reflection padding.

    Parameters
    ----------
    x_flat : [N, L]
        Flattened input tensor.
    percentile : float
        Percentile in [0, 100].
    window_size : int
        Sliding window size (odd, >= 3).

    Returns
    -------
    continuum : [N, L]
    """
    n_spectra, length = x_flat.shape
    pad = window_size // 2
    q = percentile / 100.0

    continuum = torch.empty_like(x_flat)
    for i in range(n_spectra):
        for j in range(length):
            # Build window with reflection padding
            vals = []
            for k in range(j - pad, j + pad + 1):
                if k < 0:
                    idx = -k
                elif k >= length:
                    idx = 2 * length - k - 2
                else:
                    idx = k
                idx = max(0, min(idx, length - 1))
                vals.append(x_flat[i, idx].item())
            vals_t = torch.tensor(vals, device=x_flat.device, dtype=torch.float64)
            continuum[i, j] = torch.quantile(vals_t, q, interpolation="linear").to(
                x_flat.dtype
            )

    return continuum


def savitzky_golay_per_spectrum(
    x_flat: torch.Tensor,
    window_length: int,
    polyorder: int,
) -> torch.Tensor:
    """Reference per-spectrum, per-position lstsq fit for Savitzky–Golay.

    Mirrors the production ``conv1d`` + precomputed kernel using an
    explicit per-position polynomial fit via ``torch.linalg.lstsq``
    with Vandermonde basis and reflection padding.

    Parameters
    ----------
    x_flat : [N, L]
        Flattened input tensor.
    window_length : int
        Odd window length (>= 3).
    polyorder : int
        Polynomial order (< window_length).

    Returns
    -------
    smoothed : [N, L]
    """
    n_spectra, length = x_flat.shape
    pad = window_length // 2

    # Pre-build Vandermonde basis (shared across all positions and spectra).
    xs = torch.arange(-pad, pad + 1, dtype=torch.float64)
    A = torch.stack([xs**k for k in range(polyorder + 1)], dim=1)  # [W, P+1]

    smoothed = torch.empty_like(x_flat)
    for i in range(n_spectra):
        for j in range(length):
            # Build window with reflect padding
            vals = []
            for k in range(j - pad, j + pad + 1):
                if k < 0:
                    idx = -k
                elif k >= length:
                    idx = 2 * length - k - 2
                else:
                    idx = k
                idx = max(0, min(idx, length - 1))
                vals.append(x_flat[i, idx].item())

            y = torch.tensor(vals, dtype=torch.float64)  # [W]
            c = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)  # [P+1]
            # Evaluate polynomial at the centre (x=0) → c[0]
            smoothed[i, j] = c[0].to(x_flat.dtype)

    return smoothed


def wavelet_decompose_per_spectrum(
    x_flat: torch.Tensor,
    levels: int,
) -> torch.Tensor:
    """Reference per-spectrum, per-level Haar wavelet decomposition.

    Mirrors the vectorized ``_haar_dwt_1d`` multi-level decomposition
    using explicit per-position averaging/differencing loops.  Output
    layout matches the production: ``[approx_L, detail_L, ..., detail_1]``
    stacked along the last dimension.

    Parameters
    ----------
    x_flat : [N, L]
        Flattened input tensor (length padded to be divisible by 2^levels).
    levels : int
        Number of decomposition levels (1–8).

    Returns
    -------
    coeffs : [N, L]
        Stacked wavelet coefficients in the same layout as production.
    """
    n_spectra, length = x_flat.shape

    # Multi-level decomposition: repeatedly decompose the approximation
    current = x_flat.clone()
    detail_bands: list[torch.Tensor] = []
    for _ in range(levels):
        clen = current.shape[-1]
        approx = torch.empty(
            n_spectra, clen // 2, device=x_flat.device, dtype=x_flat.dtype
        )
        detail = torch.empty_like(approx)
        for i in range(n_spectra):
            for j in range(clen // 2):
                a = current[i, 2 * j].item()
                b = current[i, 2 * j + 1].item()
                approx[i, j] = (a + b) / 2.0
                detail[i, j] = (a - b) / 2.0
        detail_bands.append(detail)
        current = approx  # continue decomposing the approximation

    # final approx = current
    # detail_bands = [detail_1, detail_2, ..., detail_L]
    # Production layout: [approx_L, detail_L, ..., detail_1]
    result = torch.cat([current] + detail_bands[::-1], dim=-1)
    return result


def continuum_normalize_per_spectrum(
    x_flat: torch.Tensor,
    order: int,
    n_sigma: float,
    max_iter: int,
) -> torch.Tensor:
    """Reference per-spectrum lstsq continuum fit for ContinuumNormalize.

    Mirrors the production batched normal-equations solver using
    per-spectrum ``torch.linalg.lstsq`` with the same iterative
    sigma-clipping logic and Vandermonde polynomial basis.

    Parameters
    ----------
    x_flat : [N, L]
        Flattened input tensor.
    order : int
        Polynomial order.
    n_sigma : float
        Sigma-clipping threshold.
    max_iter : int
        Maximum sigma-clipping iterations.

    Returns
    -------
    continuum : [N, L]
    """
    n_spectra, length = x_flat.shape
    t = torch.linspace(-1.0, 1.0, length, device=x_flat.device, dtype=x_flat.dtype)
    A = torch.stack([t**k for k in range(order + 1)], dim=1)  # [L, order+1]
    ridge = 1e-6 * torch.eye(order + 1, device=x_flat.device, dtype=x_flat.dtype)

    continuum = torch.empty_like(x_flat)
    for i in range(n_spectra):
        y = x_flat[i]
        mask = torch.ones(length, dtype=torch.bool, device=x_flat.device)
        for _ in range(max_iter):
            if mask.sum().item() <= order:
                mask = torch.ones(length, dtype=torch.bool, device=x_flat.device)
            Am = A[mask]
            ym = y[mask]
            # Ridge-regularized lstsq: same 1e-6 penalty as production
            AtA = Am.T @ Am + ridge
            Aty = Am.T @ ym
            c = torch.linalg.solve(AtA, Aty)  # [order+1]
            fit = A @ c  # [L]
            residuals = y - fit
            std = residuals[mask].std(unbiased=False)
            new_mask = residuals.abs() < n_sigma * max(std.item(), 1e-9)
            if torch.equal(new_mask, mask):
                break
            mask = new_mask
        continuum[i] = fit

    return continuum


def continuum_removal_per_spectrum(
    x_flat: torch.Tensor,
    method: str,
    order: int,
    n_knots: int,
    n_sigma: float,
    max_iter: int,
) -> torch.Tensor:
    """Reference per-spectrum continuum fit for ContinuumRemoval.

    Mirrors the production ``_fit_poly_continuum`` / ``_fit_spline_continuum``
    using per-spectrum ``torch.linalg.solve`` with the same iterative
    sigma-clipping logic.  Supports both polynomial and cubic B-spline
    methods.

    Parameters
    ----------
    x_flat : [N, L]
        Flattened input tensor.
    method : str
        ``"polynomial"`` or ``"spline"``.
    order : int
        Polynomial order (used when method="polynomial").
    n_knots : int
        Number of B-spline knots (used when method="spline").
    n_sigma : float
        Sigma-clipping threshold.
    max_iter : int
        Maximum sigma-clipping iterations.

    Returns
    -------
    baseline : [N, L]
    """
    n_spectra, length = x_flat.shape

    if method == "polynomial":
        t = torch.linspace(-1.0, 1.0, length, device=x_flat.device, dtype=x_flat.dtype)
        A = torch.stack([t**k for k in range(order + 1)], dim=1)  # [L, order+1]
        ridge = 1e-6 * torch.eye(order + 1, device=x_flat.device, dtype=x_flat.dtype)
    else:
        B = _build_spline_basis(length, n_knots, x_flat.device, x_flat.dtype)
        ridge = 1e-6 * torch.eye(n_knots, device=x_flat.device, dtype=x_flat.dtype)

    baseline = torch.empty_like(x_flat)
    for i in range(n_spectra):
        y = x_flat[i]
        mask = torch.ones(length, dtype=torch.bool, device=x_flat.device)
        for _ in range(max_iter):
            if method == "polynomial":
                if mask.sum().item() <= order:
                    mask = torch.ones(length, dtype=torch.bool, device=x_flat.device)
                Am = A[mask]
            else:
                if mask.sum().item() <= n_knots:
                    mask = torch.ones(length, dtype=torch.bool, device=x_flat.device)
                Am = B[mask]
            ym = y[mask]
            AtA = Am.T @ Am + ridge
            Aty = Am.T @ ym
            try:
                c = torch.linalg.solve(AtA, Aty)
            except RuntimeError:
                if method == "polynomial":
                    AtA_full = A.T @ A + ridge
                    Aty_full = A.T @ y
                else:
                    AtA_full = B.T @ B + ridge
                    Aty_full = B.T @ y
                c = torch.linalg.solve(AtA_full, Aty_full)
            fit = (A if method == "polynomial" else B) @ c
            residuals = y - fit
            masked_res = torch.where(mask, residuals, torch.zeros_like(residuals))
            count = mask.float().sum()
            mean_res = masked_res.sum() / max(count.item(), 1.0)
            var = torch.where(
                mask, (residuals - mean_res) ** 2, torch.zeros_like(residuals)
            ).sum() / max(count.item(), 1.0)
            std = math.sqrt(max(var.item(), 0.0))
            new_mask = residuals.abs() < n_sigma * max(std, 1e-9)
            if torch.equal(new_mask, mask):
                break
            mask = new_mask
        baseline[i] = fit

    return baseline
