"""Variable-length 'PA' columns read as bytes; 'PC'/'PM' raise instead of returning floats.

The VLA type map fell back to float32 for every element type it did not list:
'PA' was sized at 4 bytes per character and rejected as overrunning the heap,
and 'PC' came back as truncated float32 values.
"""

import struct

import pytest

import torchfits._C as C


def _card(key, value=None):
    if value is None:
        return key.ljust(80)
    if isinstance(value, bool):
        text = ("T" if value else "F").rjust(20)
    elif isinstance(value, str):
        text = ("'" + value.ljust(8) + "'").ljust(20)
    else:
        text = str(value).rjust(20)
    return (key.ljust(8) + "= " + text).ljust(80)


def _block(data: bytes, fill: bytes) -> bytes:
    return data + fill * (-len(data) % 2880)


def _write_vla_table(path, tform, cells, heap):
    rows = b"".join(struct.pack(">ii", n, off) for n, off in cells)
    primary = [
        _card("SIMPLE", True),
        _card("BITPIX", 8),
        _card("NAXIS", 0),
        _card("EXTEND", True),
    ]
    ext = [
        _card("XTENSION", "BINTABLE"),
        _card("BITPIX", 8),
        _card("NAXIS", 2),
        _card("NAXIS1", 8),
        _card("NAXIS2", len(cells)),
        _card("PCOUNT", len(heap)),
        _card("GCOUNT", 1),
        _card("TFIELDS", 1),
        _card("TTYPE1", "V"),
        _card("TFORM1", tform),
    ]
    with open(path, "wb") as fh:
        for cards in (primary, ext):
            fh.write(_block(("".join(cards) + "END".ljust(80)).encode("ascii"), b" "))
        fh.write(_block(rows + heap, b"\0"))


def test_variable_length_string_column(tmp_path):
    path = str(tmp_path / "pa.fits")
    values = [b"ab", b"hello", b"x"]
    heap, cells = b"", []
    for v in values:
        cells.append((len(v), len(heap)))
        heap += v
    _write_vla_table(path, "1PA(5)", cells, heap)

    got = C.read_fits_table(path, 1)["V"]
    assert [bytes(t.tolist()) for t in got] == values


def test_variable_length_complex_column_is_rejected(tmp_path):
    path = str(tmp_path / "pc.fits")
    heap = struct.pack(">ffff", 1.0, 2.0, 3.0, 4.0)
    _write_vla_table(path, "1PC(2)", [(2, 0)], heap)
    with pytest.raises(RuntimeError, match="not supported"):
        C.read_fits_table(path, 1)
