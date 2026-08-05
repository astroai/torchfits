"""macOS-only diagnostic (TEMPORARY): dump ground truth for the compressed
float parity failure. Deleted before landing; drives ci.yml in PR #DIAG only."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import struct
import sys

import numpy as np
import pytest
import torch

fitsio = pytest.importorskip("fitsio")
astropy_fits = pytest.importorskip("astropy.io.fits")
import torchfits  # noqa: E402


def _cfitsio_version_of_lib(soname: str) -> str:
    try:
        lib = ctypes.CDLL(soname)
        fn = lib.fits_get_version
        v = ctypes.c_float(0.0)
        fn(ctypes.byref(v))
        return repr(v.value)
    except Exception as exc:  # noqa: BLE001
        return f"<{type(exc).__name__}: {exc}>"


def _torchfits_cfitsio_version() -> str:
    return "vendored 4.6.4 (per CMake TORCHFITS_USE_VENDORED_CFITSIO=ON)"


def _astropy_cfitsio_version(astropy_fits) -> str:
    try:
        import glob
        import os

        mod = astropy_fits._utils
        so = getattr(mod, "__file__", None)
        if so is None:
            d = os.path.dirname(astropy_fits.__file__)
            hits = glob.glob(os.path.join(d, "_utils*.so")) + glob.glob(
                os.path.join(d, "*", "_utils*.so")
            )
            so = hits[0] if hits else None
        if so is None:
            return "<no _utils .so found>"
        return _cfitsio_version_of_lib(so)
    except Exception as exc:  # noqa: BLE001
        return f"<{type(exc).__name__}: {exc}>"


def _dump() -> dict:
    rng = np.random.default_rng(20260804)
    compressed = rng.normal(size=(128, 128)).astype(np.float32)
    info: dict = {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
            "node": platform.node(),
        },
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "astropy": astropy_fits.__version__ if hasattr(astropy_fits, "__version__") else "?",
            "fitsio": getattr(fitsio, "__version__", "?"),
        },
        "cfitsio_versions": {
            "fitsio_wheel": fitsio.cfitsio_version() if hasattr(fitsio, "cfitsio_version") else repr(fitsio),
            "torchfits_C": _torchfits_cfitsio_version(),
            "astropy_wheel": _astropy_cfitsio_version(astropy_fits),
        },
    }
    try:
        from astropy.io.fits._utils import get_cfitsio_version

        info["cfitsio_versions"]["astropy_wheel_py"] = get_cfitsio_version()
    except Exception as exc:  # noqa: BLE001
        pass
    codes = {"RICE_1": "rice", "GZIP_1": "gzip", "HCOMPRESS_1": "hcompress"}
    info.setdefault("codecs", {})
    for ctype in ("RICE_1", "GZIP_1", "HCOMPRESS_1"):
        hdu = astropy_fits.CompImageHDU(compressed.copy(), compression_type=ctype)
        path = os.path.join(os.getcwd(), f"_diag_{codes[ctype]}.fits")
        astropy_fits.HDUList([astropy_fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)

        header = {}
        try:
            disk_header = fitsio.read_header(path, ext=1)
            for key in (
                "ZCMPTYPE",
                "ZQUANTIZ",
                "ZBITPIX",
                "ZSCALE",
                "ZZERO",
                "ZVAL1",
                "ZVAL2",
                "ZVAL3",
                "ZVAL4",
                "ZTILELEN",
                "NAXIS",
                "NAXIS1",
                "NAXIS2",
                "BSCALE",
                "BZERO",
                "BLANK",
                "EXTNAME",
            ):
                if key in disk_header:
                    v = disk_header[key]
                    header[key] = str(v) if not isinstance(v, float) else v
        except Exception as exc:  # noqa: BLE001
            header["<error>"] = f"<{type(exc).__name__}: {exc}>"

        tf = np.asarray(torchfits.read(path, hdu=1, mmap=True)).copy()
        fi = fitsio.read(path, ext=1).copy()
        ap = astropy_fits.open(path)[1].data.copy()

        # raw compressed bytes (disable_image_compression shows the table)
        try:
            with astropy_fits.open(path, disable_image_compression=True) as hdul:
                raw = hdul[1].data["COMPRESSED_DATA"]
                rawhex = raw.tobytes()[:512].hex()
        except Exception as exc:  # noqa: BLE001
            rawhex = f"<{type(exc).__name__}: {exc}>"

        m_tf_fi = np.flatnonzero(tf != fi)
        m_tf_ap = np.flatnonzero(tf != ap)
        m_fi_ap = np.flatnonzero(fi != ap)
        zero_tf = int(np.count_nonzero(tf == 0.0))
        zero_fi = int(np.count_nonzero(fi == 0.0))

        sample: list[dict] = []
        for i in m_tf_fi[:10].tolist():
            sample.append(
                {
                    "idx": int(i),
                    "y": int(i // 128),
                    "x": int(i % 128),
                    "tf": float(tf.flat[i]),
                    "tf_hex": struct.pack("<f", tf.flat[i].item()).hex(),
                    "fi": float(fi.flat[i]),
                    "fi_hex": struct.pack("<f", fi.flat[i].item()).hex(),
                }
            )

        info["codecs"][codes[ctype]] = {
            "header": header,
            "raw_compressed_hex": rawhex,
            "file_size": os.path.getsize(path),
            "n_mismatch_tf_vs_fitsio": int(len(m_tf_fi)),
            "n_mismatch_tf_vs_astropy": int(len(m_tf_ap)),
            "n_mismatch_fitsio_vs_astropy": int(len(m_fi_ap)),
            "n_pixels_zero_torchfits": zero_tf,
            "n_pixels_zero_fitsio": zero_fi,
            "n_pixels_zero_astropy": int(np.count_nonzero(ap == 0.0)),
            "sample": sample,
        }
    return info


def test_dump_ground_truth() -> None:
    info = _dump()
    text = json.dumps(info, indent=2, sort_keys=True)
    out = os.path.join(os.getcwd(), "diag_macos_parity.json")
    with open(out, "w") as fh:
        fh.write(text)
    pytest.fail(f"DIAG_DUMP\n{text}")