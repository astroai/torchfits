"""replace_column takes physical values when the column keeps its TSCAL/TZERO.

It used to store them as raw codes: ``[1, 2, 3]`` on a uint16 column read back
as ``[32769, 32770, 32771]`` and on TSCAL=0.5/TZERO=10 as ``[10.5, 11, 11.5]``,
and an out-of-range value came back as a bare RuntimeError after the rewrite
had started.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchfits
from torchfits import table


def _col(path):
    return table.read_arrow(path).column("A").to_pylist()


def _card(path, key):
    return torchfits.read_header(path, hdu=1).get(key)


UNSIGNED = [
    (np.uint16, "I", 32768, [0, 40000, 65535]),
    (np.uint32, "J", 2**31, [0, 3_000_000_000, 2**32 - 1]),
    (np.uint64, "K", 2**63, [0, 2**63 + 1, 2**64 - 1]),
    (np.int8, "B", -128, [-128, -1, 127]),
]


@pytest.mark.parametrize(("dtype", "code", "tzero", "new"), UNSIGNED)
@pytest.mark.parametrize("kind", ["list", "int64", "tensor"])
def test_unsigned_column_takes_values(tmp_path, dtype, code, tzero, new, kind):
    path = str(tmp_path / "u.fits")
    table.write(path, {"A": np.array([1, 2, 3], dtype=dtype)})
    if kind == "list":
        values = list(new)
    elif max(new) >= 2**63:
        pytest.skip("int64 cannot hold the uint64 values")
    elif kind == "int64":
        values = np.array(new, dtype=np.int64)
    else:
        values = torch.tensor(new, dtype=torch.int64)
    table.replace_column(path, "A", values)
    assert _col(path) == new
    assert _card(path, "TFORM1").strip().endswith(code)
    assert int(_card(path, "TZERO1")) == tzero


@pytest.mark.parametrize(("dtype", "code", "tzero", "new"), UNSIGNED)
def test_unsigned_column_rejects_out_of_range(tmp_path, dtype, code, tzero, new):
    path = str(tmp_path / "u.fits")
    table.write(path, {"A": np.array([1, 2, 3], dtype=dtype)})
    before = open(path, "rb").read()
    info = np.iinfo(dtype)
    for bad in ([info.min - 1, 2, 3], [1.5, 2, 3]):
        with pytest.raises(ValueError, match="'A'"):
            table.replace_column(path, "A", bad)
    if info.max < 2**63:
        with pytest.raises(ValueError, match="'A'"):
            table.replace_column(path, "A", [info.max + 1, 2, 3])
    assert open(path, "rb").read() == before


def _scaled(tmp_path, tnull=None):
    path = str(tmp_path / "s.fits")
    schema = {"A": {"format": "I", "bscale": 0.5, "bzero": 10.0}}
    if tnull is not None:
        schema["A"]["tnull"] = tnull
    table.write(path, {"A": np.array([100, 200, 300], dtype=np.int16)}, schema=schema)
    return path


@pytest.mark.parametrize(
    "values", [[1, 2, 3], np.array([1.0, 2.0, 3.0]), torch.tensor([1, 2, 3])]
)
def test_scaled_column_takes_physical_values(tmp_path, values):
    path = _scaled(tmp_path)
    table.replace_column(path, "A", values)
    assert _col(path) == [1.0, 2.0, 3.0]
    assert (_card(path, "TSCAL1"), _card(path, "TZERO1")) == (0.5, 10)


def test_scaled_column_rounds_and_maps_nan_to_tnull(tmp_path):
    path = _scaled(tmp_path, tnull=-32768)
    table.replace_column(path, "A", [11.5, 11.74, float("nan")])
    got = _col(path)
    assert got[:2] == [11.5, 11.5]
    assert got[2] is None or np.isnan(got[2])


def test_scaled_column_rejects_out_of_range_and_nan(tmp_path):
    path = _scaled(tmp_path)
    before = open(path, "rb").read()
    with pytest.raises(ValueError, match="'A'"):
        table.replace_column(path, "A", [40000, 2, 3])
    with pytest.raises(ValueError, match="'A'.*TNULL"):
        table.replace_column(path, "A", [float("nan"), 2, 3])
    assert open(path, "rb").read() == before


def test_explicit_tscal_tzero_take_stored_values(tmp_path):
    """New scaling given: values are stored codes, as in table.write."""
    path = _scaled(tmp_path)
    table.replace_column(
        path, "A", np.array([100, 200, 300], np.int16), tscal=0.5, tzero=10
    )
    assert _col(path) == [60.0, 110.0, 160.0]


def test_dtype_driven_layout_replacement_unchanged(tmp_path):
    """uint16 values on a scaled column still replace the layout (dtype wins)."""
    path = _scaled(tmp_path)
    table.replace_column(path, "A", np.array([1, 40000, 65535], dtype=np.uint16))
    assert _col(path) == [1, 40000, 65535]
    assert int(_card(path, "TZERO1")) == 32768
    assert _card(path, "TSCAL1") in (None, 1, 1.0)


@pytest.mark.parametrize("fmt", ["I6", "F10.3"])
def test_ascii_scaled_column_takes_physical_values(tmp_path, fmt):
    path = str(tmp_path / "a.fits")
    dtype = np.int32 if fmt.startswith("I") else np.float64
    table.write(
        path,
        {"A": np.array([100, 200, 300], dtype=dtype)},
        schema={"A": {"format": fmt, "bscale": 0.5, "bzero": 10.0}},
        table_type="ascii",
    )
    table.replace_column(path, "A", [1, 2, 3.5])
    assert _col(path) == [1.0, 2.0, 3.5]
