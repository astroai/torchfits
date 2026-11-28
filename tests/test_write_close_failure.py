"""A write whose final flush/close fails must raise, never publish.

CFITSIO writes its last buffered blocks inside ``fits_close_file``. The
image writers dropped that status, so a full disk, quota, EIO or
``RLIMIT_FSIZE`` failure there went unnoticed: ``torchfits.write`` returned
normally, left a truncated file, and with ``overwrite=True`` the atomic
publish renamed the truncated temp file over the good original.

Each case runs in a subprocess with ``RLIMIT_FSIZE`` lowered (and
``SIGXFSZ`` ignored, so ``write`` fails with ``EFBIG`` instead of killing the
process). Every operation must raise, and an existing destination must
survive byte-identical.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import torchfits

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="RLIMIT_FSIZE is POSIX-only"
)

_PRELUDE = """
import os, resource, signal, sys
import numpy as np
import torchfits

path = sys.argv[1]
limit = int(sys.argv[2])
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (limit, resource.RLIM_INFINITY))
try:
{body}
except Exception as exc:  # noqa: BLE001
    print("RAISED", type(exc).__name__, exc)
else:
    print("RETURNED")
"""


def _run_limited(body: str, path: str, limit: int) -> str:
    script = _PRELUDE.format(body=textwrap.indent(textwrap.dedent(body), "    "))
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
    proc = subprocess.run(
        [sys.executable, "-c", script, path, str(limit)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _assert_raised(out: str) -> None:
    assert out.startswith("RAISED"), out


def _no_temp_files_left(directory) -> None:
    leftovers = [n for n in os.listdir(directory) if n.endswith(".tmp.fits")]
    assert leftovers == []


def test_new_image_write_raises_and_creates_nothing(tmp_path):
    path = str(tmp_path / "new.fits")
    out = _run_limited(
        "torchfits.write(path, np.ones((30, 30), np.float32))", path, 4000
    )
    _assert_raised(out)
    assert "Failed to finish writing" in out
    assert not os.path.exists(path)
    _no_temp_files_left(tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        "torchfits.write(path, np.ones((80, 80), np.float32), overwrite=True)",
        "torchfits.write(path, [np.ones((40, 40), np.float32)] * 2, overwrite=True)",
        "torchfits.write(path, np.random.default_rng(0).integers("
        "-2**31, 2**31 - 1, (60, 60), dtype=np.int32), compress='GZIP_1',"
        " overwrite=True)",
        "torchfits.write(path, {'A': np.arange(4000, dtype=np.float64)},"
        " overwrite=True)",
        "torchfits.table.write(path, {'A': np.arange(4000, dtype=np.float64)},"
        " overwrite=True)",
        """
        with torchfits.table.writer(path, overwrite=True) as w:
            w.write({'A': np.arange(4000, dtype=np.float64)})
        """,
        "torchfits.insert_hdu(path, np.ones((60, 60), np.float32), index=1)",
        "torchfits.write(path, np.ones((80, 80), np.float32), overwrite=True,"
        " checksum=True)",
    ],
    ids=[
        "image",
        "image-list",
        "compressed",
        "dict-table",
        "table-write",
        "table-writer",
        "insert-hdu",
        "checksum",
    ],
)
def test_overwrite_failure_keeps_destination_byte_identical(tmp_path, body):
    path = str(tmp_path / "dest.fits")
    torchfits.write(path, np.full((70, 70), 7, np.float32))
    original = open(path, "rb").read()
    # Larger than the limit: the existing file is unaffected, but the new
    # payload cannot be written in full.
    limit = len(original) - 2880
    out = _run_limited(body, path, limit)
    _assert_raised(out)
    assert open(path, "rb").read() == original
    _no_temp_files_left(tmp_path)


def _full_header_image(path: str) -> None:
    """An image whose primary header has no free card slot left."""
    data = np.ones((10, 10), np.float32)
    best = None
    for n in range(20, 40):
        torchfits.write(path, data, {f"K{i:02d}": i for i in range(n)}, overwrite=True)
        if os.path.getsize(path) == 2880 + 2880:
            best = n
    assert best is not None
    torchfits.write(path, data, {f"K{i:02d}": i for i in range(best)}, overwrite=True)


def test_in_place_checksum_stamping_reports_a_failed_flush(tmp_path):
    path = str(tmp_path / "full.fits")
    _full_header_image(path)
    # CHECKSUM/DATASUM need a new header block: the file must grow.
    out = _run_limited(
        "torchfits.write_checksums(path, 0)", path, os.path.getsize(path)
    )
    _assert_raised(out)


def test_in_place_header_card_write_reports_a_failed_flush(tmp_path):
    path = str(tmp_path / "full.fits")
    _full_header_image(path)
    out = _run_limited(
        "torchfits.io._write_header_cards_if_supported(path, 0, {'EXTRA1': 1})",
        path,
        os.path.getsize(path),
    )
    _assert_raised(out)


def test_cli_compress_failure_keeps_output_absent(tmp_path):
    src = str(tmp_path / "src.fits")
    rng = np.random.default_rng(1)
    torchfits.write(src, rng.integers(-(2**31), 2**31 - 1, (80, 80), dtype=np.int32))
    dest = str(tmp_path / "out.fits")
    out = _run_limited(
        f"""
        from torchfits.cli.main import main
        rc = main(['compress', {src!r}, '-o', path, '--algorithm', 'GZIP_1'])
        if rc != 0:
            raise RuntimeError(f'exit {{rc}}')
        """,
        dest,
        8000,
    )
    _assert_raised(out)
    assert not os.path.exists(dest)


def test_explicit_close_of_a_write_handle_raises(tmp_path):
    path = str(tmp_path / "handle.fits")
    out = _run_limited(
        """
        import torchfits._C as c
        f = c.open_fits_file(path, 'w')
        f.write_image(np.ones((30, 30), np.float32))
        f.close()
        """,
        path,
        4000,
    )
    _assert_raised(out)
    assert "Failed to finish writing" in out
