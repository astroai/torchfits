"""The native module must reject or handle inputs the Python layer normally guards.

These calls go straight to ``torchfits._C``. Crash cases run in a subprocess so
a regression fails instead of killing the test session.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

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


def test_mmap_reader_calls_reject_a_foreign_capsule(tmp_path):
    path = tmp_path / "cap.fits"
    write_table(str(path), {"A": np.arange(3, dtype=np.int32)}, overwrite=True)
    res = _run(
        f"""
        import ctypes
        import torchfits._C as C
        new = ctypes.pythonapi.PyCapsule_New
        new.restype = ctypes.py_object
        new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        buf = ctypes.create_string_buffer(4096)
        bogus = new(ctypes.addressof(buf), b"some.other.capsule", None)
        for fn in (
            C.read_fits_table_rows_mmap_from_reader,
            C.read_fits_table_rows_mmap_from_reader_raw,
        ):
            try:
                fn(bogus, ["A"], 1, -1)
            except TypeError:
                pass
            else:
                raise SystemExit("foreign capsule accepted")
        reader = C.open_fits_mmap_reader({str(path)!r}, 1)
        out = C.read_fits_table_rows_mmap_from_reader(reader, ["A"], 1, -1)
        assert out["A"].tolist() == [0, 1, 2], out
        print("ok")
        """
    )
    assert res.returncode == 0, res.stdout + res.stderr


_MPS = torch.backends.mps.is_available()


@pytest.mark.skipif(not _MPS, reason="needs a non-CPU torch device (MPS)")
def test_native_writes_refuse_device_arrays(tmp_path):
    image = str(tmp_path / "img.fits")
    table = str(tmp_path / "tab.fits")
    dev = torch.ones(4, 4, device="mps")
    col = torch.ones(4, device="mps")

    with pytest.raises(RuntimeError, match="CPU"):
        C.write_fits_file(image, [{"data": dev}], True)
    with pytest.raises(RuntimeError, match="CPU"):
        C.write_fits_file_compressed_images(image, [{"data": dev}], True, "GZIP")
    with pytest.raises(RuntimeError, match="CPU"):
        C.write_fits_table(table, {"a": col}, {}, True, None, "binary")
    with pytest.raises(RuntimeError, match="CPU"):
        C.write_fits_table(table, {"v": [col, col]}, {}, True, None, "binary")

    write_table(table, {"a": np.zeros(4, dtype=np.float32)}, overwrite=True)
    with pytest.raises(RuntimeError, match="CPU"):
        C.append_fits_table_rows(table, 1, {"a": col})
    with pytest.raises(RuntimeError, match="CPU"):
        C.update_fits_table_rows(table, 1, {"a": col}, 1, 4)
    assert C.read_fits_table(table, 1)["a"].tolist() == [0.0] * 4


def test_ascii_schema_width_follows_the_code_letter(tmp_path):
    path = str(tmp_path / "ascii.fits")
    C.write_fits_table(
        path, {"s": ["hello", "worlds"]}, {}, True, {"s": {"format": "A10"}}, "ascii"
    )
    rows = C.read_fits_table(path, 1)["s"]
    assert [bytes(r).rstrip().decode() for r in rows.tolist()] == ["hello", "worlds"]


class _TableHDU:  # shape understood by the native write_hdus
    def __init__(self, header):
        self._raw_data = {"a": torch.arange(5, dtype=torch.int32)}
        self.header = header


def test_table_header_cannot_rewrite_the_table_layout(tmp_path):
    path = str(tmp_path / "tab_hdr.fits")
    for header in (
        {"NAXIS2": 2},
        {"NAXIS1": 8},
        {"TFIELDS": 3},
        {"TFORM1": "1E"},
        {"TTYPE1": "b"},
        {"PCOUNT": 100},
    ):
        C.write_fits_file(path, [{"data": torch.zeros(0)}, _TableHDU(header)], True)
        cards = {k: v for k, v, _ in C.read_header_dict(path, 1)}
        assert cards["NAXIS1"] == "4", header
        assert cards["NAXIS2"] == "5", header
        assert cards["TFORM1"].strip("' ") == "1J", header
        assert C.read_fits_table(path, 1)["a"].tolist() == list(range(5)), header

    # A spelling of the same layout is kept, and ordinary cards still land.
    C.write_fits_file(
        path,
        [{"data": torch.zeros(0)}, _TableHDU({"TFORM1": "J", "OBSERVER": "me"})],
        True,
    )
    cards = {k: v for k, v, _ in C.read_header_dict(path, 1)}
    assert cards["TFORM1"].strip("' ") == "J"
    assert cards["OBSERVER"].strip("' ") == "me"


def test_bad_header_value_is_rejected_before_overwriting(tmp_path):
    path = str(tmp_path / "partial.fits")
    writers = (
        lambda hdus: C.write_fits_file(path, hdus, True),
        lambda hdus: C.write_fits_file_compressed_images(path, hdus, True, "GZIP"),
    )
    for write in writers:
        C.write_fits_file(path, [{"data": torch.ones(4, 4)}], True)
        hdus = [{"data": torch.zeros(4, 4), "header": {"FOO": [1]}}]
        with pytest.raises(RuntimeError, match="Unsupported FITS header value"):
            write(hdus)
        assert torch.equal(C.read_full(path, 0, True), torch.ones(4, 4))

    # A bad card on a later HDU must not leave earlier HDUs half-written.
    path = str(tmp_path / "handle.fits")
    f = C.open_fits_file(path, "w")
    f.write_hdus([{"data": torch.ones(4, 4)}], False)
    hdus = [{"data": torch.zeros(2, 2)}, {"data": torch.zeros(2), "header": {"X": [1]}}]
    with pytest.raises(RuntimeError, match="Unsupported FITS header value"):
        f.write_hdus(hdus, False)
    f.close()
    g = C.open_fits_file(path, "r")
    assert g.get_num_hdus() == 1
    g.close()
