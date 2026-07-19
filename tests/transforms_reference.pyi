# Type stubs for transforms_reference — reference implementation for parity testing.

from __future__ import annotations

import torch

def sigma_clip_naive(
    x: torch.Tensor,
    n_sigma: float,
    max_iter: int,
    dims: tuple[int, ...],
    fill: str,
) -> torch.Tensor: ...
