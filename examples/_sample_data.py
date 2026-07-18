"""Download and cache public FITS samples for gallery examples."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit


def _default_sample_cache() -> Path:
    override = os.environ.get("TORCHFITS_SAMPLE_CACHE", "").strip()
    if override:
        return Path(override).expanduser()
    try:
        from torchfits.cache import sample_cache_root

        return sample_cache_root()
    except Exception:
        xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
        if xdg:
            return Path(xdg).expanduser() / "torchfits" / "samples"
        return Path.home() / ".cache" / "torchfits" / "samples"


CACHE_DIR = _default_sample_cache()

# Stable public tutorial / survey files (astropy-data + SDSS SAS).
SAMPLES: dict[str, str] = {
    "horsehead": "http://data.astropy.org/tutorials/FITS-images/HorseHead.fits",
    "chandra_events": "http://data.astropy.org/tutorials/FITS-tables/chandra_events.fits",
    # Same plate/mjd/fiber used in specutils Spectrum.read docs.
    "sdss_spectrum": (
        "https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/"
        "0751/spec-0751-52251-0160.fits"
    ),
    "m13_blue_0001": "http://data.astropy.org/tutorials/FITS-images/M13_blue_0001.fits",
    "m13_blue_0002": "http://data.astropy.org/tutorials/FITS-images/M13_blue_0002.fits",
    "m13_blue_0003": "http://data.astropy.org/tutorials/FITS-images/M13_blue_0003.fits",
    "m13_blue_0004": "http://data.astropy.org/tutorials/FITS-images/M13_blue_0004.fits",
    "m13_blue_0005": "http://data.astropy.org/tutorials/FITS-images/M13_blue_0005.fits",
    "fits_header_mef": "http://data.astropy.org/tutorials/FITS-Header/input_file.fits",
    "sdss_lupton_g": "http://data.astropy.org/visualization/reprojected_sdss_g.fits.bz2",
    "sdss_lupton_r": "http://data.astropy.org/visualization/reprojected_sdss_r.fits.bz2",
    "sdss_lupton_i": "http://data.astropy.org/visualization/reprojected_sdss_i.fits.bz2",
    "spitzer_example": "http://data.astropy.org/photometry/spitzer_example_image.fits",
    "radio_cube_c14": "http://data.astropy.org/tutorials/FITS-cubes/reduced_TAN_C14.fits",
    "manga_logcube": (
        "https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/7443/"
        "stack/manga-7443-12703-LOGCUBE.fits.gz"
    ),
    "galaxy_zoo1_table2": (
        "https://galaxy-zoo-1.s3.amazonaws.com/GalaxyZoo1_DR_table2.fits"
    ),
}


class SampleUnavailable(RuntimeError):
    """Raised when network samples cannot be fetched (or FAST mode skips)."""


def megacam_dir() -> Path:
    """Local cache dir for CFHT MegaCam ``.fits.fz`` samples (see fetch script)."""
    return Path(__file__).resolve().parents[1] / "benchmarks_data" / "cfht_megacam"


def megapipe_dir() -> Path:
    """Local cache dir for CFHTLS-Deep D1 MegaPipe mosaics/catalog (see fetch script)."""
    return Path(__file__).resolve().parents[1] / "benchmarks_data" / "cfht_megapipe"


def gz_legacy_cutouts_dir() -> Path:
    """Cache dir for Legacy Survey grz cutouts keyed to Galaxy Zoo 1 rows."""
    return CACHE_DIR / "gz_legacy_cutouts"


def _dest_path(name: str) -> Path:
    """Cache path for ``name``, preserving the URL's (possibly compound) suffix."""
    url_name = Path(urlsplit(SAMPLES[name]).path).name
    suffix = "".join(Path(url_name).suffixes) or ".fits"
    return CACHE_DIR / f"{name}{suffix}"


def _fast_mode() -> bool:
    return os.environ.get("TORCHFITS_EXAMPLE_FAST", "").strip() in (
        "1",
        "true",
        "TRUE",
        "yes",
    )


def ensure_sample(name: str, *, allow_download: bool | None = None) -> Path:
    """Return a local path for a named sample, downloading once if needed.

    In ``TORCHFITS_EXAMPLE_FAST=1`` (CI), skips network and raises
    :class:`SampleUnavailable` unless the file is already cached.
    """
    if name not in SAMPLES:
        raise KeyError(f"unknown sample {name!r}; choose from {sorted(SAMPLES)}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = _dest_path(name)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest

    if allow_download is None:
        allow_download = not _fast_mode()
    if not allow_download:
        raise SampleUnavailable(
            f"sample {name!r} not cached at {dest} (TORCHFITS_EXAMPLE_FAST skips download)"
        )

    url = SAMPLES[name]
    tmp = dest.with_name(dest.name + ".partial")
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
