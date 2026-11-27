"""Tile-compression controls for ``write(compress=...)``.

fpack-style knobs for CFITSIO tile-compressed image writes: the tile geometry
(``fpack -t``) and float quantization / dithering (``fpack -q``, ``-qz``,
``-qn``, CFITSIO ``fits_set_quantize_*`` / ``fits_set_dither_seed``). Options are validated here, before anything is written, and
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

# quantize_method= names -> CFITSIO codes (fits_set_quantize_method).
QUANTIZE_METHODS = {
    "subtractive_dither_1": 1,
    "subtractive_dither_2": 2,
    "no_dither": -1,
}
_NO_DITHER = -1
_SUBTRACTIVE_DITHER_2 = 2

# Codecs that can store floating-point pixels losslessly (quantize_level=0).
_LOSSLESS_FLOAT_CODECS = ("GZIP_1", "GZIP_2")

# CFITSIO's quantize level when dithering is off (imcomp_init_table): finer
# than the dithered default of 4 so undithered quantization stays unbiased.
_NO_DITHER_DEFAULT_LEVEL = 16.0


def canonical_algorithm(algorithm: str) -> str:
    """CFITSIO codec name for ``algorithm`` (unknown names pass through)."""
    key = algorithm.strip().upper()
    return _ALGORITHM_ALIASES.get(key, key)


@dataclass(frozen=True)
class CompressionOptions:
    """Validated tile-compression options for one ``write()`` call."""

    algorithm: str
    tile_shape: Optional[Tuple[int, ...]] = None
    quantize_level: Optional[float] = None
    quantize_method: Optional[int] = None
    dither_seed: Optional[int] = None

    @property
    def quantize_requested(self) -> bool:
        """True when any float-quantization option was given."""
        return (
            self.quantize_level is not None
            or self.quantize_method is not None
            or self.dither_seed is not None
        )

    def image_options(self, data: Tensor, idx: int) -> Dict[str, Any]:
        """Native ``_compression`` mapping for one image payload."""
        out: Dict[str, Any] = {}
        if self.tile_shape is not None:
            out["tile_dims"] = resolve_tile_dims(
                self.tile_shape, tuple(data.shape), self.algorithm, idx
            )
        if data.is_floating_point():
            # Integer images are always compressed losslessly: CFITSIO ignores
            # these settings for them.
            level = self.quantize_level
            if level is None and self.quantize_method == _NO_DITHER:
                level = _NO_DITHER_DEFAULT_LEVEL
            if level is not None:
                out["quantize_level"] = float(level)
            if self.quantize_method is not None:
                out["quantize_method"] = self.quantize_method
            if self.dither_seed is not None:
                out["dither_seed"] = self.dither_seed
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


def _normalize_quantize_level(value: Any) -> float:
    import math
    import numbers

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"quantize_level must be a number, got {value!r}")
    level = float(value)
    if not math.isfinite(level):
        raise ValueError(f"quantize_level must be finite, got {value!r}")
    return level


def _normalize_quantize_method(value: Any) -> int:
    if not isinstance(value, str):
        raise TypeError(
            f"quantize_method must be one of {sorted(QUANTIZE_METHODS)}, got {value!r}"
        )
    key = value.strip().lower()
    if key not in QUANTIZE_METHODS:
        raise ValueError(
            f"quantize_method must be one of {sorted(QUANTIZE_METHODS)}, got {value!r}"
        )
    return QUANTIZE_METHODS[key]


def _normalize_dither_seed(value: Any) -> int:
    import numbers

    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"dither_seed must be an int, got {value!r}")
    seed = int(value)
    if not 1 <= seed <= 10000:
        raise ValueError(f"dither_seed must be in 1..10000, got {seed}")
    return seed


def _check_quantize_combination(options: CompressionOptions) -> None:
    algorithm = options.algorithm
    level = options.quantize_level
    method = options.quantize_method
    if algorithm == "PLIO_1":
        raise ValueError(
            "quantize_level/quantize_method/dither_seed quantize floating-point "
            "images, which PLIO_1 cannot store; use another compress= codec"
        )
    if level == 0 and algorithm not in _LOSSLESS_FLOAT_CODECS:
        raise ValueError(
            "quantize_level=0 (lossless floats) needs compress='GZIP_1' or "
            f"'GZIP_2'; {algorithm} always quantizes floating-point images"
        )
    dither_given = method is not None or options.dither_seed is not None
    # GZIP_1 floats default to lossless: nothing would be quantized.
    lossless = level == 0 or (level is None and algorithm == "GZIP_1")
    if dither_given and lossless:
        why = (
            "quantize_level=0 stores them losslessly"
            if level == 0
            else f"{algorithm} stores them losslessly unless quantize_level is set"
        )
        raise ValueError(
            f"quantize_method/dither_seed only apply when floats are quantized, "
            f"but {why}; set a non-zero quantize_level"
        )
    if method == _NO_DITHER and options.dither_seed is not None:
        raise ValueError("dither_seed has no effect with quantize_method='no_dither'")
    if method == _SUBTRACTIVE_DITHER_2 and algorithm == "HCOMPRESS_1":
        raise ValueError(
            "HCOMPRESS_1 does not support quantize_method='subtractive_dither_2'; "
            "use 'subtractive_dither_1' or 'no_dither'"
        )


def compression_options(
    compress: Any,
    *,
    tile_shape: Any = None,
    quantize_level: Any = None,
    quantize_method: Any = None,
    dither_seed: Any = None,
) -> Optional[CompressionOptions]:
    """Validate the tile-compression keywords of ``write()``.

    Returns ``None`` when none is set. Raises ``ValueError``/``TypeError``
    for a bad value or one that ``compress`` cannot use.
    """
    given = {
        "tile_shape": tile_shape,
        "quantize_level": quantize_level,
        "quantize_method": quantize_method,
        "dither_seed": dither_seed,
    }
    names = [name for name, value in given.items() if value is not None]
    if not names:
        return None
    from ._file_ops import _resolve_compression_algorithm

    label = "/".join(f"{name}=" for name in names)
    resolved = _resolve_compression_algorithm(compress)
    if resolved is None:
        raise ValueError(f"{label} applies to tile-compressed writes: pass compress=")
    algorithm = canonical_algorithm(resolved)
    if algorithm == "NONE":
        raise ValueError(
            f"{label} applies to tile-compressed writes, not compress='NONE'"
        )
    options = CompressionOptions(
        algorithm=algorithm,
        tile_shape=None if tile_shape is None else _normalize_tile_shape(tile_shape),
        quantize_level=(
            None
            if quantize_level is None
            else _normalize_quantize_level(quantize_level)
        ),
        quantize_method=(
            None
            if quantize_method is None
            else _normalize_quantize_method(quantize_method)
        ),
        dither_seed=None
        if dither_seed is None
        else _normalize_dither_seed(dither_seed),
    )
    if options.quantize_requested:
        _check_quantize_combination(options)
    return options


def require_quantizable_image(
    options: Optional[CompressionOptions], any_float: bool
) -> None:
    """Refuse quantize options on a write with no floating-point image."""
    if options is not None and options.quantize_requested and not any_float:
        raise ValueError(
            "quantize_level/quantize_method/dither_seed apply to floating-point "
            "images, but every image HDU in this write is integer (integer "
            "images are always compressed losslessly)"
        )
