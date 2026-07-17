"""Download and cache public FITS samples for gallery examples."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

CACHE_DIR = Path(
    os.environ.get("TORCHFITS_SAMPLE_CACHE", Path.home() / ".cache" / "torchfits" / "samples")
)

# Stable public tutorial / survey files (astropy-data + SDSS SAS).
SAMPLES: dict[str, str] = {
    "horsehead": "http://data.astropy.org/tutorials/FITS-images/HorseHead.fits",
    "chandra_events": "http://data.astropy.org/tutorials/FITS-tables/chandra_events.fits",
    # Same plate/mjd/fiber used in specutils Spectrum.read docs.
    "sdss_spectrum": (
        "https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/"
        "0751/spec-0751-52251-0160.fits"
    ),
}


class SampleUnavailable(RuntimeError):
    """Raised when network samples cannot be fetched (or FAST mode skips)."""


def _fast_mode() -> bool:
    return os.environ.get("TORCHFITS_EXAMPLE_FAST", "").strip() in ("1", "true", "TRUE", "yes")


def ensure_sample(name: str, *, allow_download: bool | None = None) -> Path:
    """Return a local path for a named sample, downloading once if needed.

    In ``TORCHFITS_EXAMPLE_FAST=1`` (CI), skips network and raises
    :class:`SampleUnavailable` unless the file is already cached.
    """
    if name not in SAMPLES:
        raise KeyError(f"unknown sample {name!r}; choose from {sorted(SAMPLES)}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / f"{name}.fits"
    if dest.is_file() and dest.stat().st_size > 0:
        return dest

    if allow_download is None:
        allow_download = not _fast_mode()
    if not allow_download:
        raise SampleUnavailable(
            f"sample {name!r} not cached at {dest} (TORCHFITS_EXAMPLE_FAST skips download)"
        )

    url = SAMPLES[name]
    tmp = dest.with_suffix(".fits.partial")
    try:
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 — fixed public URLs
        tmp.replace(dest)
    except (urllib.error.URLError, OSError) as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise SampleUnavailable(f"failed to download {name} from {url}: {exc}") from exc
    return dest


def try_ensure_sample(name: str) -> Path | None:
    """Like :func:`ensure_sample` but returns ``None`` when unavailable."""
    try:
        return ensure_sample(name)
    except SampleUnavailable:
        return None
