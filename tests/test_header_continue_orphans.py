"""A CONTINUE card only extends the string card right before it.

After a number or a COMMENT card, a CONTINUE is malformed and must stay a
visible orphan instead of being glued onto an earlier, unrelated string
value. HIERARCH long strings drop their LONGSTRN '&' like 8-char keywords.
"""

from __future__ import annotations

import numpy as np
import pytest

import torchfits
import torchfits._C as _C
import torchfits._core as core
from torchfits._hdu.card import Card, _reassemble_longstr_cards
from torchfits._io_engine import hdu_api
from torchfits.header_parser import fast_parse_header, fast_parse_header_cards

CARDS = [
    "L3      = 'x'",
    "NUM     =                    5",
    "CONTINUE  'after num'",
    "L4      = 'a&'",
    "COMMENT   interleaved",
    "CONTINUE  'b'",
    "HIERARCH ESO LONG VAL = 'hierarch long &'",
    "CONTINUE  'tail'",
]


@pytest.fixture()
def fits_path(tmp_path):
    cards = [
        "SIMPLE  =                    T",
        "BITPIX  =                   16",
        "NAXIS   =                    2",
        "NAXIS1  =                    2",
        "NAXIS2  =                    2",
        *CARDS,
        "END",
    ]
    raw = "".join(c.ljust(80) for c in cards).ljust(2880).encode("ascii")
    path = tmp_path / "cont.fits"
    path.write_bytes(raw + np.zeros(4, ">i2").tobytes().ljust(2880, b"\0"))
    return str(path)


def _check(values: dict, orphans: list) -> None:
    assert values["L3"] == "x"
    assert values["L4"] == "a&"
    assert values["ESO LONG VAL"] == "hierarch long tail"
    assert orphans == ["after num", "b"]


def test_read_header(fits_path):
    hdr = torchfits.read_header(fits_path)
    _check(hdr, [c.value for c in hdr.cards if c.key == "CONTINUE"])
    assert hdr["COMMENT"].strip() == "interleaved"


def test_open_header(fits_path):
    with torchfits.open(fits_path) as hdul:
        hdr = hdul[0].header
        _check(hdr, [c.value for c in hdr.cards if c.key == "CONTINUE"])


def test_card_parser(fits_path):
    cards = fast_parse_header_cards(core.read_header_string(fits_path, 0))
    _check(
        {k: v for k, v, _ in cards},
        [v for k, v, _ in cards if k == "CONTINUE"],
    )


def test_dict_parser(fits_path):
    hdr = fast_parse_header(core.read_header_string(fits_path, 0))
    assert hdr["L3"] == "x"
    assert hdr["L4"] == "a&"
    assert hdr["ESO LONG VAL"] == "hierarch long tail"
    assert hdr["COMMENT"] == "interleaved"


def test_native_triple_reassembly(fits_path):
    raw = _C.FITSFile(fits_path, 0).read_header(0)
    for rebuilt in (
        _reassemble_longstr_cards(raw),
        hdu_api._reassemble_longstr_cards([Card(*t) for t in raw]),
    ):
        values = {c.key: c.value for c in rebuilt if c.key != "CONTINUE"}
        assert values["L3"] == "x"
        assert values["L4"] == "a&"
        assert values["ESO LONG VAL"] == "hierarch long tail"
        assert values["COMMENT"].strip() == "interleaved"
        assert sum(c.key == "CONTINUE" for c in rebuilt) == 2
