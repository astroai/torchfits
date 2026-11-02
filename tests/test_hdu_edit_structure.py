"""HDU edits keep every HDU they do not insert or replace byte-for-byte.

insert/replace/delete decoded the whole file and wrote it back: a RICE
compressed HDU came back decompressed after delete_hdu of another HDU, and
compress= recompressed every image of the file instead of only the new one.
"""

import pytest
import torch

import torchfits

BLOCK = 2880


def _hdus(path):
    """``(header_cards, raw_bytes)`` of every HDU, parsed from 2880-byte blocks."""
    data = open(path, "rb").read()
    out, pos = [], 0
    while pos < len(data):
        start, cards = pos, []
        while True:
            block = data[pos : pos + BLOCK]
            pos += BLOCK
            done = False
            for i in range(0, BLOCK, 80):
                card = block[i : i + 80].decode("ascii")
                if card.rstrip() == "END":
                    done = True
                    break
                cards.append(card)
            if done:
                break
        kv = {}
        for card in cards:
            if card[8:10] == "= ":
                kv.setdefault(card[:8].strip(), card[10:].split("/")[0].strip())
        naxis = int(kv.get("NAXIS", 0))
        n = 0
        if naxis:
            n = 1
            for k in range(1, naxis + 1):
                n *= int(kv[f"NAXIS{k}"])
            n = abs(int(kv["BITPIX"])) // 8 * (n + int(kv.get("PCOUNT", 0)))
        pos += -(-n // BLOCK) * BLOCK
        out.append((kv, data[start:pos]))
    return out


def _no_checksum(raw):
    """HDU bytes with CHECKSUM/DATASUM card values blanked out."""
    cards = [raw[i : i + 80] for i in range(0, len(raw), 80)]
    return b"".join(c[:8] if c[:8] in (b"CHECKSUM", b"DATASUM ") else c for c in cards)


def _checksums_ok(path):
    n = torchfits.read_num_hdus(path)
    return [torchfits.verify_checksums(path, hdu=i)["status"] for i in range(n)]


def test_insert_keeps_scaled_image_bytes(tmp_path):
    path = str(tmp_path / "scaled.fits")
    codes = torch.arange(400, dtype=torch.int16).reshape(20, 20)
    codes[0, 0] = -32768
    torchfits.write(
        path,
        [
            torch.zeros(1, 1),
            {"data": codes, "header": {"BSCALE": 0.5, "BZERO": 10.0, "BLANK": -32768}},
        ],
        checksum=True,
    )
    before = _hdus(path)

    torchfits.insert_hdu(path, torch.ones(2, 2), index=1, header={"EXTNAME": "NEW"})

    after = _hdus(path)
    assert len(after) == 3
    assert after[0][1] == before[0][1]
    assert after[2][1] == before[1][1]
    kv = after[2][0]
    assert (kv["BITPIX"], kv["BSCALE"], kv["BZERO"], kv["BLANK"]) == (
        "16",
        "0.5",
        "10.",
        "-32768",
    )
    assert _checksums_ok(path) == ["ok", "ok", "ok"]


def test_delete_keeps_rice_hdus_compressed(tmp_path):
    path = str(tmp_path / "rice.fits")
    img = torch.arange(10000, dtype=torch.int32).reshape(100, 100)
    torchfits.write(path, [img, img + 1, img + 2], compress="RICE", checksum=True)
    before = _hdus(path)
    assert [h[0].get("ZIMAGE") for h in before] == [None, "T", "T", "T"]

    torchfits.delete_hdu(path, 2)

    after = _hdus(path)
    assert [h[1] for h in after] == [before[0][1], before[1][1], before[3][1]]
    assert torch.equal(torchfits.read(path, hdu=2), img + 2)
    assert _checksums_ok(path) == ["ok", "ok", "ok"]


def test_replace_keeps_scaled_table_column(tmp_path):
    path = str(tmp_path / "table.fits")
    torchfits.table.write(
        path,
        {"q": torch.tensor([0, 1, 2, 3], dtype=torch.int16)},
        schema={"q": {"bscale": 0.25, "bzero": 100.0, "unit": "mag"}},
        overwrite=True,
    )
    torchfits.insert_hdu(path, torch.ones(3, 3), index=2, header={"EXTNAME": "IMG"})
    for i in range(3):
        torchfits.write_checksums(path, hdu=i)
    before = _hdus(path)
    assert before[1][0]["TSCAL1"] and before[1][0]["TZERO1"]

    torchfits.replace_hdu(path, "IMG", torch.zeros(4, 4))

    after = _hdus(path)
    assert after[0][1] == before[0][1]
    assert after[1][1] == before[1][1]
    assert torchfits.read(path, hdu=2).shape == (4, 4)
    assert _checksums_ok(path) == ["ok", "ok", "ok"]


def test_insert_compress_applies_to_new_hdu_only(tmp_path):
    path = str(tmp_path / "mixed.fits")
    torchfits.write(
        path,
        [torch.arange(100.0).reshape(10, 10), {"data": torch.ones(10, 10)}],
        checksum=True,
    )
    before = _hdus(path)
    new = torch.arange(10000, dtype=torch.int32).reshape(100, 100)

    torchfits.insert_hdu(path, new, index=2, header={"EXTNAME": "C"}, compress="RICE")

    after = _hdus(path)
    assert [h[1] for h in after[:2]] == [h[1] for h in before]
    assert "ZIMAGE" not in after[1][0]
    assert after[2][0]["ZIMAGE"] == "T"
    assert after[2][0]["ZCMPTYPE"] == "'RICE_1  '"
    assert torch.equal(torchfits.read(path, hdu="C"), new)
    assert _checksums_ok(path) == ["ok", "ok", "ok"]


def test_primary_moves_convert_headers_and_keep_data(tmp_path):
    path = str(tmp_path / "prim.fits")
    old = torch.arange(6, dtype=torch.int16).reshape(2, 3)
    torchfits.write(path, [old, torch.ones(2, 2)], checksum=True)
    before = _hdus(path)

    torchfits.insert_hdu(path, torch.zeros(5), index=0)
    after = _hdus(path)
    assert after[0][0]["SIMPLE"] == "T"
    assert after[1][0]["XTENSION"] == "'IMAGE   '"
    assert "EXTEND" not in after[1][0] and after[1][0]["PCOUNT"] == "0"
    data_start = len(before[0][1]) - BLOCK
    assert after[1][1][-BLOCK:] == before[0][1][data_start:]
    assert after[2][1] == before[1][1]
    assert torch.equal(torchfits.read(path, hdu=1), old)
    assert _checksums_ok(path) == ["ok", "ok", "ok"]

    torchfits.delete_hdu(path, 0)
    restored = _hdus(path)
    assert restored[0][0]["SIMPLE"] == "T" and "PCOUNT" not in restored[0][0]
    assert _no_checksum(restored[0][1]) == _no_checksum(before[0][1])
    assert restored[1][1] == before[1][1]
    assert _checksums_ok(path) == ["ok", "ok"]


def test_delete_refuses_compress_and_table_primary(tmp_path):
    path = str(tmp_path / "d.fits")
    torchfits.write(path, [torch.zeros(2, 2), torch.ones(2, 2), torch.ones(3)])
    before = open(path, "rb").read()
    with pytest.raises(ValueError, match="compress"):
        torchfits.delete_hdu(path, 1, compress=True)
    assert open(path, "rb").read() == before

    comp = str(tmp_path / "c.fits")
    torchfits.write(comp, [torch.ones(4, 4)], compress=True)
    with pytest.raises(ValueError, match="tile-compressed image"):
        torchfits.delete_hdu(comp, 0)
