"""Path helpers for CFITSIO extended filenames."""

from __future__ import annotations


def cfitsio_base_path(path: str) -> str:
    """Return the on-disk path, stripping a CFITSIO ``[...]`` filter if present.

    CFITSIO extended filenames look like ``image.fits[10:100,20:200]`` or
    ``file.fits[1]``. Existence checks must use the base file, not the filter.
    """
    bracket = path.find("[")
    if bracket < 0:
        return path
    return path[:bracket]


def has_cfitsio_filter(path: str) -> bool:
    """True when ``path`` includes any CFITSIO ``[...]`` extended-filename bracket.

    This is a bracket presence test, not an image-section detector: HDU selectors
    like ``file.fits[1]`` / ``[EVENTS]`` also match. Prefer ``hdu=`` / EXTNAME for
    those; only image pixel sections are a smoke-tested torchfits surface today.
    """
    return "[" in path
