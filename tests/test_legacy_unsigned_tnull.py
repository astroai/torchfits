"""Unsigned-convention TNULL written by torchfits 1.3.1 in the value domain.

1.3.1 wrote e.g. ``TNULL1 = 65535`` on an ``I`` / ``TZERO = 32768`` column
(the uint16 value) instead of the stored raw integer (32767). The FITS-correct
reader compared it as a raw value, which can never match, so those nulls read
back as real values. A TNULL outside the stored range but inside the decoded
range is now read as that legacy decoded value, on every read route.
"""

from __future__ import annotations

import numpy as np
import pytest

import torchfits
from torchfits import table
from torchfits.fits_schema import decoded_tnull


def _add_card(path: str, card: str) -> None:
    """Insert ``card`` before END of the first extension header."""
    data = bytearray(open(path, "rb").read())
    end = data.find(b"END" + b" " * 77, 2880)
    assert end > 0 and (end + 80) % 2880 != 0, "header block is full"
    data[end : end + 80] = card.ljust(80).encode()
    data[end + 80 : end + 160] = b"END".ljust(80)
    open(path, "wb").write(bytes(data))


CASES = [
    # dtype, values, legacy (value-domain) TNULL, its row
    (np.uint16, [0, 1, 65535, 7], 65535, 2),
    (np.uint32, [0, 1, 2**32 - 1, 7], 2**32 - 1, 2),
    (np.int8, [0, 1, -1, 7], -1, 2),
    (np.uint64, [0, 1, 2**64 - 1, 7], 2**64 - 1, 2),
]


def _as_list(result, name):
    col = result[name] if isinstance(result, dict) else result.column(name)
    return col.tolist() if hasattr(col, "tolist") else col.to_pylist()


@pytest.fixture(params=CASES, ids=lambda c: np.dtype(c[0]).name)
def legacy(tmp_path, request):
    dtype, values, tnull, row = request.param
    path = str(tmp_path / "legacy.fits")
    table.write(path, {"A": np.array(values, dtype=dtype)})
    _add_card(path, f"TNULL1  = {tnull:>20}")
    assert torchfits.read_header(path, hdu=1)["TNULL1"] == tnull
    expected = list(values)
    expected[row] = None
    return path, expected


@pytest.mark.parametrize("backend", ["auto", "cpp", "torch"])
def test_legacy_tnull_reads_as_null(legacy, backend):
    path, expected = legacy
    assert _as_list(table.read(path, backend=backend), "A") == expected
    assert _as_list(table.read(path, where="A IS NULL", backend=backend), "A") == [None]
    kept = [v for v in expected if v is not None and v != 0]
    got = _as_list(table.read(path, where="A > 0", backend=backend), "A")
    assert got == kept


def test_legacy_tnull_arrow(legacy):
    path, expected = legacy
    assert table.read_arrow(path).column("A").to_pylist() == expected


@pytest.mark.parametrize(
    ("tnull", "tform", "tzero", "decoded"),
    [
        (32767, "I", 32768, 65535),  # standard raw TNULL
        (65535, "I", 32768, 65535),  # legacy value-domain TNULL
        (-32768, "I", 32768, 0),
        (127, "B", -128, -1),  # raw B byte 127 -> int8 -1
        (-1, "B", -128, -1),  # legacy int8 value
        (2**63 - 1, "K", 2**63, 2**64 - 1),
        (2**64 - 1, "K", 2**63, 2**64 - 1),
        (70000, "I", 32768, 70000 + 32768),  # outside both: unchanged rule
    ],
)
def test_decoded_tnull_legacy_domain(tnull, tform, tzero, decoded):
    assert decoded_tnull(tnull, tform, 1.0, tzero) == decoded


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: table.insert_column(p, "C", np.arange(4)),
        lambda p: table.replace_column(p, "B", np.arange(4) + 1),
    ],
    ids=["insert_column", "replace_column"],
)
def test_rewrite_of_legacy_file_keeps_nulls(tmp_path, mutate):
    """A column rewrite moved the legacy TNULL by TZERO again and refused it."""
    path = str(tmp_path / "legacy.fits")
    table.write(
        path, {"A": np.array([0, 1, 65535, 7], dtype=np.uint16), "B": np.arange(4)}
    )
    _add_card(path, f"TNULL1  = {65535:>20}")
    mutate(path)
    assert table.read_arrow(path).column("A").to_pylist() == [0, 1, None, 7]
    # The rewrite stores the standard raw TNULL.
    assert torchfits.read_header(path, hdu=1)["TNULL1"] == 32767
