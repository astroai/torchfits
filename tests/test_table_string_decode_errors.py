"""String columns decode strictly by default, the same on every read path.

Bytes the encoding cannot decode were dropped silently (errors="ignore"):
b'caf\\xe9XX' read back as 'cafXX', and where S == 'cafXX' matched it. The
Arrow fast path also validated UTF-8 instead of the requested encoding, so
UTF-8 bytes decoded under encoding='ascii' only when no other row ended in a
space. Decoding is now strict by default -- a UnicodeDecodeError names the
column and suggests encoding= / decode_errors= -- and decode_errors='replace'
or 'ignore' opts into a lossy decode, identically on every path.
"""

import numpy as np
import pytest

import torchfits

T = torchfits.table


def _make(path, rows, old, new):
    T.write(path, {"ID": np.arange(len(rows), dtype=np.int32), "S": np.array(rows)})
    raw = bytearray(open(path, "rb").read())
    i = raw.find(old)
    raw[i : i + len(new)] = new
    open(path, "wb").write(bytes(raw))
    return path


@pytest.fixture()
def latin1(tmp_path):
    return _make(
        str(tmp_path / "l.fits"), ["cafeXX", "abcdef"], b"cafeXX", b"caf\xe9XX"
    )


@pytest.fixture(params=["abcdef", "ab"], ids=["unpadded", "padded"])
def utf8(tmp_path, request):
    # 'ab' is space-padded, which used to switch the decode path.
    rows = ["cafeXX", request.param]
    return _make(str(tmp_path / "u.fits"), rows, b"cafeXX", b"caf\xc3\xa9X")


READS = {
    "auto": lambda p, **kw: T.read(p, **kw).column("S").to_pylist(),
    "torch": lambda p, **kw: T.read(p, backend="torch", **kw).column("S").to_pylist(),
    "cpp": lambda p, **kw: T.read(p, backend="cpp", **kw).column("S").to_pylist(),
    "no-mmap": lambda p, **kw: T.read(p, mmap=False, **kw).column("S").to_pylist(),
    "scan": lambda p, **kw: [
        v for b in T.scan(p, **kw) for v in b.column("S").to_pylist()
    ],
    "hdu": lambda p, **kw: _hdu_strings(p, **kw),
}


def _hdu_strings(path, **kw):
    with torchfits.open(path) as hdul:
        return hdul[1].get_string_column("S", **kw)


@pytest.mark.parametrize("path_name", sorted(READS))
def test_strict_by_default_on_every_path(latin1, utf8, path_name):
    read = READS[path_name]
    for path in (latin1, utf8):
        with pytest.raises(UnicodeDecodeError, match="'S'.*encoding='latin-1'"):
            read(path)


@pytest.mark.parametrize("path_name", sorted(READS))
def test_opt_in_and_encoding_agree_on_every_path(latin1, utf8, path_name):
    read = READS[path_name]
    assert read(latin1, encoding="latin-1")[0] == "caf\xe9XX"
    assert read(latin1, decode_errors="replace")[0] == "caf�XX"
    assert read(latin1, decode_errors="ignore")[0] == "cafXX"
    assert read(utf8, encoding="utf-8")[0] == "caf\xe9X"
    assert read(utf8, decode_errors="replace")[0] == "caf��X"
    with pytest.raises(UnicodeDecodeError):
        read(latin1, encoding="utf-8")


def test_where_sees_the_same_decode(latin1):
    with pytest.raises(UnicodeDecodeError):
        T.read(latin1, where="S == 'cafXX'")
    assert T.read(latin1, where="S == 'cafXX'", decode_errors="ignore").num_rows == 1
    assert T.read(latin1, where="S == 'caf\xe9XX'", encoding="latin-1").num_rows == 1
    # Rows that are never decoded do not fail the read.
    assert T.read(latin1, where="ID == 1").column("S").to_pylist() == ["abcdef"]


def test_decode_errors_is_validated(latin1):
    with pytest.raises(ValueError, match="decode_errors"):
        T.read(latin1, decode_errors="surrogateescape")
    with pytest.raises(ValueError, match="decode_errors"):
        T.schema(latin1, where="ID == 0", decode_errors="bogus")
