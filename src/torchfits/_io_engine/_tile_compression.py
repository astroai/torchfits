"""Tile-compression controls for ``write(compress=...)``.

fpack-style knobs for CFITSIO tile-compressed image writes: the tile geometry
(``fpack -t``). Options are validated here, before anything is written, and
resolved per image HDU into the ``_compression`` mapping the native writer
reads from each image payload dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple

# Torch-free at import: the CLI (`torchfits compress`) imports this module.
if TYPE_CHECKING:
    from torch import Tensor

# Canonical CFITSIO codec name for every spelling the native writer accepts.
_ALGORITHM_ALIASES = {
    "R": "RICE_1",
    "RICE": "RICE_1",
    "RICE_1": "RICE_1",
    "G": "GZIP_1",
    "GZIP": "GZIP_1",
    "GZIP_1": "GZIP_1",
    "GZIP_2": "GZIP_2",
    "H": "HCOMPRESS_1",
    "HCOMPRESS": "HCOMPRESS_1",
    "HCOMPRESS_1": "HCOMPRESS_1",
    "P": "PLIO_1",
    "PLIO": "PLIO_1",
    "PLIO_1": "PLIO_1",
    "B": "BZIP2_1",
    "BZIP2": "BZIP2_1",
    "BZIP2_1": "BZIP2_1",
    "NONE": "NONE",
}

# HCOMPRESS works on 2-D tiles of at least this many pixels per side.
_HCOMPRESS_MIN_TILE = 4


def canonical_algorithm(algorithm: str) -> str:
    """CFITSIO codec name for ``algorithm`` (unknown names pass through)."""
    key = algorithm.strip().upper()
    return _ALGORITHM_ALIASES.get(key, key)


@dataclass(frozen=True)
class CompressionOptions:
    """Validated tile-compression options for one ``write()`` call."""

    algorithm: str
    tile_shape: Optional[Tuple[int, ...]] = None

    def image_options(self, data: Tensor, idx: int) -> Dict[str, Any]:
        """Native ``_compression`` mapping for one image payload."""
        out: Dict[str, Any] = {}
        if self.tile_shape is not None:
            out["tile_dims"] = resolve_tile_dims(
                self.tile_shape, tuple(data.shape), self.algorithm, idx
            )
        return out


def _normalize_tile_shape(tile_shape: Any) -> Tuple[int, ...]:
    if isinstance(tile_shape, (str, bytes)) or not isinstance(tile_shape, Sequence):
        raise TypeError(
            "tile_shape must be a sequence of ints (one tile size per image axis, "
            f"in tensor axis order), got {type(tile_shape).__name__}"
        )
    dims: list[int] = []
    for value in tile_shape:
        if isinstance(value, bool) or not isinstance(value, int):
            try:
                import numpy as np

                if isinstance(value, np.integer):
                    value = int(value)
                else:
                    raise TypeError
            except TypeError:
                raise TypeError(
                    f"tile_shape entries must be ints, got {value!r}"
                ) from None
        if value == 0 or value < -1:
            raise ValueError(
                f"tile_shape entries must be positive (or -1 for the whole axis), "
                f"got {tuple(tile_shape)!r}"
            )
        dims.append(int(value))
    if not dims:
        raise ValueError("tile_shape must name at least one axis")
    if len(dims) > 6:
        raise ValueError(
            f"tile_shape has {len(dims)} axes; CFITSIO tiles at most 6 dimensions"
        )
    return tuple(dims)


def resolve_tile_dims(
    tile_shape: Tuple[int, ...],
    shape: Tuple[int, ...],
    algorithm: str,
    idx: int,
) -> list[int]:
    """Tile sizes for an image of ``shape``, in FITS axis order (ZTILE1 first).

    ``tile_shape`` is in tensor axis order and covers the trailing axes:
    leading axes it leaves out get tile size 1. ``-1`` and sizes past the axis
    length mean the whole axis.
    """
    ndim = len(shape)
    if len(tile_shape) > ndim:
        raise ValueError(
            f"tile_shape {tile_shape!r} has {len(tile_shape)} axes but image HDU "
            f"{idx} has {ndim} (shape {tuple(shape)})"
        )
    full = (1,) * (ndim - len(tile_shape)) + tuple(tile_shape)
    dims = [
        axis if size == -1 or size > axis else size for size, axis in zip(full, shape)
    ]
    if algorithm == "HCOMPRESS_1":
        spanning = [size for size in dims if size > 1]
        if len(spanning) != 2 or min(spanning) < _HCOMPRESS_MIN_TILE:
            raise ValueError(
                "HCOMPRESS_1 needs 2-D tiles: exactly two tile axes longer than 1, "
                f"each at least {_HCOMPRESS_MIN_TILE} pixels; tile_shape "
                f"{tile_shape!r} gives {tuple(dims)} on image HDU {idx} "
                f"(shape {tuple(shape)})"
            )
    return list(reversed(dims))


def compression_options(
    compress: Any,
    *,
    tile_shape: Any = None,
) -> Optional[CompressionOptions]:
    """Validate the tile-compression keywords of ``write()``.

    Returns ``None`` when none is set. Raises ``ValueError``/``TypeError``
    for a bad value or one that ``compress`` cannot use.
    """
    if tile_shape is None:
        return None
    from ._file_ops import _resolve_compression_algorithm

    resolved = _resolve_compression_algorithm(compress)
    if resolved is None:
        raise ValueError(
            "tile_shape= applies to tile-compressed writes: pass compress="
        )
    algorithm = canonical_algorithm(resolved)
    if algorithm == "NONE":
        raise ValueError(
            "tile_shape= applies to tile-compressed writes, not compress='NONE'"
        )
    return CompressionOptions(
        algorithm=algorithm,
        tile_shape=_normalize_tile_shape(tile_shape),
    )
