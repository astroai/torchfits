# Type stubs for transforms_reference — reference implementations for parity testing.

from __future__ import annotations

import torch

def upper_envelope_per_spectrum(
    x_flat: torch.Tensor,
    is_local_max: torch.Tensor,
    *,
    smooth: float = 0.0,
) -> torch.Tensor: ...
def phase_fold_per_bin(
    x_flat: torch.Tensor,
    n_bins: int,
    bin_idx: torch.Tensor,
) -> torch.Tensor: ...
def sigma_clip_naive(
    x: torch.Tensor,
    n_sigma: float,
    max_iter: int,
    dims: tuple[int, ...],
    fill: str,
) -> torch.Tensor: ...
def alpha_shape_per_spectrum(
    x_flat: torch.Tensor,
    half_window: int,
    iterations: int,
) -> torch.Tensor: ...
def continuum_normalize_per_spectrum(
    x_flat: torch.Tensor,
    order: int,
    n_sigma: float,
    max_iter: int,
) -> torch.Tensor: ...
def asls_dense_solve(
    x_flat: torch.Tensor,
    lam: float,
    p: float,
    max_iter: int,
    envelope: str = "lower",
) -> torch.Tensor: ...
def running_percentile_per_spectrum(
    x_flat: torch.Tensor,
    percentile: float,
    window_size: int,
) -> torch.Tensor: ...
def savitzky_golay_per_spectrum(
    x_flat: torch.Tensor,
    window_length: int,
    polyorder: int,
) -> torch.Tensor: ...
def wavelet_decompose_per_spectrum(
    x_flat: torch.Tensor,
    levels: int,
) -> torch.Tensor: ...
def continuum_removal_per_spectrum(
    x_flat: torch.Tensor,
    method: str,
    order: int,
    n_knots: int,
    n_sigma: float,
    max_iter: int,
) -> torch.Tensor: ...
