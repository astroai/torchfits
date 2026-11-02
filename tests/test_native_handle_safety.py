"""Handle-level calls must keep the HDU cursor honest and never touch a freed fitsfile.

- get_num_hdus walks CFITSIO to the last HDU; a later read on the same handle
  returned that HDU instead of the one asked for.
- A table read through a handle left the image cursor on the table HDU.
- A closed handle handed nullptr to CFITSIO (SIGSEGV), and a TableReader built
  on a handle kept a dangling pointer after the handle was closed.

The crash cases run in a subprocess so a regression fails instead of killing
the test session.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

import torchfits
import torchfits._C as C
from torchfits._table.write import write as write_table


@pytest.fixture
def image_file(tmp_path):
    path = tmp_path / "two.fits"
    torchfits.write(
        str(path),
        [torch.zeros(8, 8, dtype=torch.float32), torch.ones(32, 32, dtype=torch.int16)],
        overwrite=True,
    )
    return str(path)


def test_num_hdus_does_not_move_the_read_cursor(image_file):
    f = C.open_fits_file(image_file, "r")
    assert tuple(f.read_tensor(0).shape) == (8, 8)
    assert C.get_num_hdus(f) == 2
    assert f.get_num_hdus() == 2
    assert tuple(f.read_tensor(0).shape) == (8, 8)
    assert tuple(C.read_full(f, 0).shape) == (8, 8)
    f.close()


def test_table_read_through_handle_keeps_image_cursor(tmp_path):
    path = tmp_path / "img_tab.fits"

    class _TableHDU:  # shape understood by the native write_hdus
        _raw_data = {"A": torch.arange(3, dtype=torch.int32)}
        header = {}

    image = {"data": torch.arange(16, dtype=torch.int32).reshape(4, 4)}
    C.write_fits_file(str(path), [image, _TableHDU()], True)
    f = C.open_fits_file(str(path), "r")
    assert torch.equal(
        f.read_tensor(0), torch.arange(16, dtype=torch.int32).reshape(4, 4)
    )
    assert C.read_fits_table_from_handle(f, 1)["A"].tolist() == [0, 1, 2]
    assert torch.equal(
        f.read_tensor(0), torch.arange(16, dtype=torch.int32).reshape(4, 4)
    )
    f.close()


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_closed_handle_raises_instead_of_crashing(image_file):
    res = _run(
        f"""
        import torchfits._C as C
        f = C.open_fits_file({image_file!r}, "r")
        f.close()
        for call in (lambda: C.get_num_hdus(f), lambda: C.read_fits_table_from_handle(f, 1)):
            try:
                call()
            except RuntimeError as exc:
                assert "closed" in str(exc), exc
            else:
                raise SystemExit("closed handle did not raise")
        print("ok")
        """
    )
    assert res.returncode == 0, (res.returncode, res.stderr)
    assert res.stdout.strip() == "ok"


def test_table_reader_outlives_closed_handle(tmp_path):
    path = tmp_path / "tab.fits"
    write_table(str(path), {"A": np.arange(5, dtype=np.int32)}, overwrite=True)
    res = _run(
        f"""
        import gc
        import torchfits._C as C
        f = C.open_fits_file({str(path)!r}, "r")
        tr = C.TableReader(f, 1)
        f.close()
        del f
        gc.collect()
        print(tr.read_rows(["A"])["A"].tolist())
        """
    )
    assert res.returncode == 0, (res.returncode, res.stderr)
    assert res.stdout.strip() == "[0, 1, 2, 3, 4]"
