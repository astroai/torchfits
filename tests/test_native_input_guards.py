"""The native module must reject or handle inputs the Python layer normally guards.

These calls go straight to ``torchfits._C``. Crash cases run in a subprocess so
a regression fails instead of killing the test session.
"""

import subprocess
import sys
import textwrap

import numpy as np

import torchfits._C as C
from torchfits._table.write import write as write_table


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_zero_dim_bit_and_string_payloads_write_one_row(tmp_path):
    bit = tmp_path / "bit.fits"
    text = tmp_path / "text.fits"
    write_table(
        str(bit),
        {"F": np.array([True, False, True])},
        schema={"F": {"format": "1X"}},
        overwrite=True,
    )
    write_table(str(text), {"S": ["ab", "cd"]}, overwrite=True)
    res = _run(
        f"""
        import numpy as np
        import torchfits._C as C
        C.update_fits_table_rows({str(bit)!r}, 1, {{"F": np.array(False)}}, 1, 1)
        C.append_fits_table_rows({str(bit)!r}, 1, {{"F": np.array(True)}})
        C.update_fits_table_rows(
            {str(text)!r}, 1, {{"S": np.array(65, dtype=np.uint8)}}, 1, 1
        )
        print("ok")
        """
    )
    assert res.returncode == 0, res.stderr
    assert C.read_fits_table(str(bit), 1)["F"].flatten().tolist() == [
        False,
        False,
        True,
        True,
    ]
    assert C.read_fits_table(str(text), 1)["S"].tolist() == [[65, 32], [99, 100]]
