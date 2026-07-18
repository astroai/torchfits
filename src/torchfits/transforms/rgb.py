"""Lupton asinh RGB (1.0 surface); richer multi-band RGB → 1.1."""

from __future__ import annotations

from typing import Any

from ..cli.rgb import lupton_rgb as _lupton_rgb


def lupton_rgb(
    r: Any,
    g: Any,
    b: Any,
    *,
    Q: float = 8.0,
    stretch: float = 0.5,
    minimum: float = 0.0,
) -> Any:
    """Lupton+ (2004) asinh RGB → float tensor ``(H, W, 3)`` in ``[0, 1]``.

    Thin wrapper around the CLI preview helper. Tune ``Q`` / ``stretch`` /
    ``minimum`` per survey. Additional multi-band stretches are deferred to
    1.1.
    """
    return _lupton_rgb(r, g, b, Q=Q, stretch=stretch, minimum=minimum)
