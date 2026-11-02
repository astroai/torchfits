"""insert_column / replace_column keep every other column exactly as stored."""

from __future__ import annotations

import numpy as np
import pytest

import torchfits


def _hdu_layout(path: str) -> list[tuple[list[str], int, int]]:
    """``(cards, start, end)`` per HDU, parsed from the raw 2880-byte blocks."""
    raw = open(path, "rb").read()
    out = []
    pos = 0
    while pos < len(raw):
        start = pos
        cards: list[str] = []
        while not cards or not cards[-1].startswith("END     "):
            card = raw[pos : pos + 80].decode("ascii")
            pos += 80
            cards.append(card)
        pos = -(-pos // 2880) * 2880
        values = _card_values(cards)
        size = 0
        if int(values.get("NAXIS", 0)):
            size = abs(int(values["BITPIX"])) // 8
            for a in range(1, int(values["NAXIS"]) + 1):
                size *= int(values[f"NAXIS{a}"])
            size += int(values.get("PCOUNT", 0))
        pos += -(-size // 2880) * 2880
        out.append((cards, start, pos))
    return out


def _hdu_cards(path: str, index: int) -> list[str]:
    return _hdu_layout(path)[index][0]


def _hdu_bytes(path: str, index: int) -> bytes:
    _cards, start, end = _hdu_layout(path)[index]
    return open(path, "rb").read()[start:end]


def _card_values(cards: list[str]) -> dict[str, str]:
    return {
        c[:8].strip(): c[10:].split("/")[0].strip() for c in cards if c[8:10] == "= "
    }


@pytest.fixture
def scaled_table(tmp_path):
    path = str(tmp_path / "scaled.fits")
    torchfits.table.write(
        path,
        {
            "s": np.array([100, 200, 300], dtype=np.int16),
            "x": np.arange(3, dtype=np.int32),
        },
        schema={"s": {"format": "I", "bscale": 0.5, "bzero": 10.0}},
        header={"EXTNAME": "CAT"},
        overwrite=True,
    )
    # A second scaled table after the target must come through byte-identical.
    other = str(tmp_path / "other.fits")
    torchfits.table.write(
        other,
        {"y": np.array([1, 2], dtype=np.int16)},
        schema={"y": {"format": "I", "bscale": 2.0, "bzero": 1.0}},
        overwrite=True,
    )
    with open(path, "ab") as fh:
        fh.write(_hdu_bytes(other, 1))
    return path


@pytest.mark.parametrize("op", ["insert", "replace"])
def test_scaled_column_survives_column_rewrite(scaled_table, op):
    before_s = torchfits.table.read_torch(scaled_table, hdu=1)["s"].tolist()
    assert before_s == [60.0, 110.0, 160.0]
    tail_before = _hdu_bytes(scaled_table, 2)

    if op == "insert":
        torchfits.table.insert_column(
            scaled_table, "new", np.array([9, 8, 7], dtype=np.int32), index=0
        )
        s_idx = 2
    else:
        torchfits.table.replace_column(
            scaled_table, "x", np.array([7, 8, 9], dtype=np.int32)
        )
        s_idx = 1

    got = torchfits.table.read_torch(scaled_table, hdu=1)
    assert got["s"].tolist() == before_s
    cards = _card_values(_hdu_cards(scaled_table, 1))
    assert cards[f"TFORM{s_idx}"].strip("'").strip() == "I"
    assert float(cards[f"TSCAL{s_idx}"]) == 0.5
    assert float(cards[f"TZERO{s_idx}"]) == 10.0
    assert _hdu_bytes(scaled_table, 2) == tail_before
    assert torchfits.table.read_torch(scaled_table, hdu=2)["y"].tolist() == [3.0, 5.0]


def test_column_rewrite_keeps_typed_cards_and_renumbers_column_cards(tmp_path):
    path = str(tmp_path / "cards.fits")
    header = torchfits.Header(
        {"EXPTIME": 30.5, "NUM": 7, "FLAG": True, "OBJ": "M31", "TDISP2": "I6"}
    )
    header["TLMIN1"] = 0
    header.add_history("first history")
    header.add_history("second history")
    header.add_comment("a comment")
    torchfits.table.write(
        path,
        {"a": np.arange(3, dtype=np.float64), "b": np.arange(3, dtype=np.int32)},
        header=header,
        overwrite=True,
    )
    torchfits.table.insert_column(
        path, "new", np.array([9, 8, 7], dtype=np.int32), index=0
    )

    cards = _hdu_cards(path, 1)
    values = _card_values(cards)
    assert values["EXPTIME"] == "30.5"
    assert values["NUM"] == "7"
    assert values["FLAG"] == "T"
    assert [c[8:].strip() for c in cards if c.startswith("HISTORY ")] == [
        "first history",
        "second history",
    ]
    assert [c[8:].strip() for c in cards if c.startswith("COMMENT ")] == ["a comment"]
    assert not any(c.startswith(("HISTORY =", "COMMENT =")) for c in cards)
    # 'a' moved from column 1 to 2, 'b' from 2 to 3.
    assert values["TLMIN2"] == "0"
    assert values["TDISP3"].strip("'").strip() == "I6"
    assert "TLMIN1" not in values and "TDISP2" not in values

    got = torchfits.read_header(path, 1)
    assert got["NUM"] == 7 and got["EXPTIME"] == 30.5 and got["FLAG"] is True


def test_ascii_table_column_rewrite(tmp_path):
    path = str(tmp_path / "ascii.fits")
    torchfits.table.write(
        path,
        {"a": np.array([1, 2, 3], dtype=np.int32), "s": ["ab", "cd", "ef"]},
        table_type="ascii",
        overwrite=True,
    )
    torchfits.table.replace_column(path, "a", np.array([7, 8, 9], dtype=np.int32))
    torchfits.table.insert_column(
        path, "z", np.array([4, 5, 6], dtype=np.int32), format="I5", index=0
    )
    values = _card_values(_hdu_cards(path, 1))
    assert values["XTENSION"].strip("'").strip() == "TABLE"
    assert int(values["TBCOL1"]) == 1
    got = torchfits.table.read(path, hdu=1).to_pydict()
    assert got == {"z": [4, 5, 6], "a": [7, 8, 9], "s": ["ab", "cd", "ef"]}
