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
