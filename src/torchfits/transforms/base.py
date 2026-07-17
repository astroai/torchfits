from __future__ import annotations

from typing import Any, Sequence

import torch

class FITSTransform:
    """Protocol for astronomy transforms with forward and inverse passes.

    Subclasses should override ``forward`` and ``inverse``.
    Calling an instance directly delegates to :meth:`forward`.

    All transforms accept an optional ``mask`` parameter
    (``torch.Tensor | None``) on both :meth:`forward` and
    :meth:`inverse`.  The mask is a boolean tensor where ``True``
    indicates a valid pixel.  Transforms that compute statistics
    (median, min, max, etc.) use the mask to exclude invalid
    pixels; pointwise transforms can safely ignore it.
    """

    def forward(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        raise NotImplementedError

    def inverse(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        raise NotImplementedError

    def __call__(self, x: Any, mask: torch.Tensor | None = None) -> Any:
        return self.forward(x, mask=mask)


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------



class Compose(FITSTransform):
    """Chain transforms; ``.inverse()`` unwinds them in reverse order."""

    def __init__(self, transforms: Sequence[FITSTransform]) -> None:
        self.transforms = list(transforms)

    def __len__(self) -> int:
        return len(self.transforms)

    def __getitem__(self, idx: int) -> FITSTransform:
        return self.transforms[idx]

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for t in self.transforms:
            x = t(x, mask=mask)
        return x

    def inverse(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for t in reversed(self.transforms):
            x = t.inverse(x, mask=mask)
        return x

    def __repr__(self) -> str:
        inner = ",\n    ".join(repr(t) for t in self.transforms)
        return f"Compose([\n    {inner}\n])"


# ---------------------------------------------------------------------------
# Stretches (stateless, always invertible)
# ---------------------------------------------------------------------------


