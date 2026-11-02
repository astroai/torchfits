"""Path-level caches re-validate extended-syntax names and never cache URLs.

``"f.fits[1]"`` is not stat-able as written, so its cache entry was never
re-validated: after ``f.fits`` was replaced (or unlinked) ``read_nrows`` and
``read_shape`` on the extended name kept answering about the old file for the
life of the process. A ``file://`` URL had the same problem. Runs in a
subprocess because the validation interval is read once per process.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import os, sys
    import torchfits

    def card(k, v):
        return f"{k:<8}= {v:>20}".ljust(80)

    def block(cards):
        s = "".join(cards) + "END".ljust(80)
        return (s + " " * ((-len(s)) % 2880)).encode()

    def build(nrows, n1):
        prim = block([card("SIMPLE", "T"), card("BITPIX", "8"), card("NAXIS", "1"),
                      card("NAXIS1", str(n1)), card("EXTEND", "T")])
        prim += b"\\0" * n1 + b"\\0" * ((-n1) % 2880)
        tab = block([card("XTENSION", "'BINTABLE'"), card("BITPIX", "8"),
                     card("NAXIS", "2"), card("NAXIS1", "4"),
                     card("NAXIS2", str(nrows)), card("PCOUNT", "0"),
                     card("GCOUNT", "1"), card("TFIELDS", "1"),
                     card("TTYPE1", "'X'"), card("TFORM1", "'J'")])
        data = b"\\0" * (4 * nrows)
        return prim + tab + data + b"\\0" * ((-len(data)) % 2880)

    def replace(path, blob):
        with open(path + ".new", "wb") as fh:
            fh.write(blob)
        os.replace(path + ".new", path)

    path = sys.argv[1]
    replace(path, build(5, 8))
    ext, url = path + "[1]", "file://" + path
    assert torchfits.read_nrows(ext, 0) == 5
    assert torchfits.read_nrows(url, 1) == 5
    assert torchfits.read_shape(path + "[0]", 0)[1] == (8,)

    replace(path, build(9, 3))
    assert torchfits.read_nrows(ext, 0) == 9, torchfits.read_nrows(ext, 0)
    assert torchfits.read_nrows(url, 1) == 9, torchfits.read_nrows(url, 1)
    assert torchfits.read_shape(path + "[0]", 0)[1] == (3,)

    os.remove(path)
    try:
        torchfits.read_nrows(ext, 0)
    except (RuntimeError, OSError):
        pass
    else:
        raise AssertionError("extended-syntax probe answered for an unlinked file")
    print("OK")
    """
)


def test_extended_syntax_and_url_probes_see_replacements(tmp_path) -> None:
    env = dict(os.environ, TORCHFITS_SHARED_META_VALIDATE_INTERVAL_MS="0")
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path / "ext.fits")],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout
