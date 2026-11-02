"""``read_keys`` returns the same typed values as ``read_header``.

The two readers parse the same cards along different paths (CFITSIO keyword
lookup vs. the Python header parser), so these cases pin the spellings where
they used to disagree: Fortran ``D`` exponents, integers past int64, subnormal
floats, CONTINUE long strings, and keyword names CFITSIO treats as wildcards.
"""

from __future__ import annotations

import pytest

import torchfits


def _card(text: str) -> bytes:
    return text.ljust(80).encode("ascii")


def _write(path, cards: list[str]) -> str:
    raw = b"".join(_card(c) for c in cards) + _card("END")
    raw += b" " * ((-len(raw)) % 2880)
    path.write_bytes(raw)
    return str(path)


@pytest.fixture
def header_file(tmp_path) -> str:
    return _write(
        tmp_path / "keys.fits",
        [
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "DVAL    =              1.5D+03",
            "DVAL2   =                 2D-1",
            "BIGI    = 99999999999999999999",
            "NEGBIG  = -99999999999999999999",
            "TINY    =             1.0E-320",
            "PLAIN   =                   17",
            "DATEISH = 1999-01-01",
            "LONGSTR = 'This is a long string value that &'",
            "CONTINUE  'continues here&'",
            "CONTINUE  ' and ends.'",
            "SHORT   = 'O''Brien  '",
        ],
    )


NUMERIC = ["DVAL", "DVAL2", "BIGI", "NEGBIG", "TINY", "PLAIN", "DATEISH"]


def test_numeric_values_match_read_header(header_file: str) -> None:
    got = torchfits.read_keys(header_file, NUMERIC)
    header = torchfits.read_header(header_file)
    for key in NUMERIC:
        assert got[key] == header[key], key
        assert type(got[key]) is type(header[key]), key
    assert got["DVAL"] == 1500.0
    assert got["BIGI"] == 99999999999999999999
    assert 0.0 < got["TINY"] < 1e-300


def test_long_string_follows_continue_cards(header_file: str) -> None:
    got = torchfits.read_keys(header_file, ["LONGSTR", "SHORT"])
    header = torchfits.read_header(header_file)
    assert got["LONGSTR"] == header["LONGSTR"]
    assert got["LONGSTR"] == "This is a long string value that continues here and ends."
    assert got["SHORT"] == "O'Brien"


@pytest.mark.parametrize("key", ["NAXIS*", "NAXI?", "DVAL#"])
def test_wildcard_key_names_are_rejected(header_file: str, key: str) -> None:
    with pytest.raises(ValueError, match="cannot contain"):
        torchfits.read_keys(header_file, [key])
