"""CLI shim: Lupton RGB helpers live in ``torchfits.transforms.rgb``.

Kept for one release so ``from torchfits.cli.rgb import lupton_rgb`` keeps working.
"""

from __future__ import annotations

from torchfits.transforms.rgb import lupton_rgb, write_rgb_image

__all__ = ["lupton_rgb", "write_rgb_image"]
