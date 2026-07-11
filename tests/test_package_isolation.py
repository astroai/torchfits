from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "torchfits"


def test_torchfits_source_does_not_reference_torchsky() -> None:
    offenders: list[str] = []
    for path in PACKAGE_ROOT.rglob("*"):
        if path.suffix not in {".py", ".cpp", ".h"} and path.name != "CMakeLists.txt":
            continue
        if "torchsky" in path.read_text(encoding="utf-8", errors="ignore").lower():
            offenders.append(str(path.relative_to(PACKAGE_ROOT)))
    assert offenders == []


def test_torchfits_contains_only_fits_native_sources() -> None:
    native_root = PACKAGE_ROOT / "cpp_src"
    assert not (native_root / "wcs.cpp").exists()
    assert not (native_root / "healpix.cpp").exists()
    assert not (PACKAGE_ROOT / "wcs").exists()
    assert not (PACKAGE_ROOT / "sphere").exists()


def test_torchfits_python_sources_never_import_astropy_or_fitsio() -> None:
    """Runtime I/O must use vendored CFITSIO + _C, not Python astropy/fitsio."""
    forbidden = (
        "import astropy",
        "from astropy",
        "import fitsio",
        "from fitsio",
    )
    offenders: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pattern in forbidden:
                if pattern in stripped:
                    offenders.append(f"{path.relative_to(PACKAGE_ROOT)}: {stripped}")
    assert not offenders, "\n".join(offenders)
