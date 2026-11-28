"""Writing through a closed native FITS handle raises instead of crashing.

``write_image``, ``write_hdus`` and ``write_hdus_compressed_images`` never
checked the handle: after ``close()`` they passed a NULL ``fitsfile*`` to
CFITSIO (SIGSEGV). The write paths also run Python code with the GIL held
(``to_tensor()``, header values), and the handle lock is recursive, so that
code could close the handle on the writing thread and leave the writer
using a freed ``fitsfile``. Each case runs in a subprocess so a crash shows
up as a failed test, not a dead test runner.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

_PRELUDE = """
import sys
import numpy as np
import torchfits
import torchfits._C as c

path = sys.argv[1]
"""

CASES = {
    "write_image_after_close": """
        f = c.open_fits_file(path, "w")
        f.close()
        f.write_image(np.zeros((4, 4), np.float32))
    """,
    "write_hdus_after_close": """
        f = c.open_fits_file(path, "w")
        f.close()
        f.write_hdus([{"data": np.zeros((4, 4), np.float32), "header": {}}], True)
    """,
    "to_tensor_closes_mid_write_hdus": """
        f = c.open_fits_file(path, "w")

        class Closing:
            header = {}

            def to_tensor(self):
                f.close()
                return np.zeros((4, 4), np.float32)

        f.write_hdus([{"data": np.ones((2, 2), np.float32), "header": {}},
                      Closing()], True)
    """,
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_write_on_a_closed_handle_raises(tmp_path, case):
    script = _PRELUDE + textwrap.dedent(CASES[case])
    script += "\n"
    wrapped = (
        "try:\n"
        + textwrap.indent(script, "    ")
        + "\nexcept RuntimeError as exc:\n    print('RAISED', exc)\n"
        + "else:\n    print('RETURNED')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", wrapped, str(tmp_path / "out.fits")],
        capture_output=True,
        text=True,
        timeout=120,
        env=dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE"),
    )
    assert proc.returncode == 0, (proc.returncode, proc.stderr[-2000:])
    assert proc.stdout.startswith("RAISED"), proc.stdout
    assert "FITSFile is closed" in proc.stdout


def test_closing_an_untouched_write_handle_is_quiet(tmp_path):
    import torchfits._C as c

    f = c.open_fits_file(str(tmp_path / "empty.fits"), "w")
    f.close()
    f.close()
