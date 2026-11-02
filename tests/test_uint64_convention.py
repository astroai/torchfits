"""uint64 images and table columns use the FITS 4.0 BZERO/TZERO = 2**63 convention.

Reads used to go through float64, losing the low bits (5 read back as 0.0),
and writes raised. Values must now round-trip exactly on every read path.
"""

import numpy as np
import pytest
import torch

import torchfits
import torchfits._C as C
import torchfits.table as table

EDGES = [0, 1, 5, 2**53 + 1, 2**63 - 1, 2**63, 2**63 + 7, 2**64 - 1]


def _u64(values):
    return torch.from_numpy(np.array(values, dtype=np.uint64))


@pytest.mark.parametrize("use_mmap", [True, False])
def test_image_roundtrip(tmp_path, use_mmap):
    path = str(tmp_path / "img.fits")
    data = _u64(EDGES).reshape(2, 4)
    torchfits.write(path, data, overwrite=True)

    header = torchfits.read_header(path, 0)
    assert int(header["BITPIX"]) == 64
    assert float(header["BZERO"]) == 2.0**63

    got = torchfits.read(path, mmap=use_mmap)
    assert got.dtype == torch.uint64
    assert torch.equal(got, data)

    with torchfits.open(path) as hdul:
        assert hdul[0].data.dtype == torch.uint64

    cut = C.SubsetReader(path, 0).read(1, 0, 3, 2)
    assert cut.dtype == torch.uint64
    assert torch.equal(cut, data[:, 1:3])


@pytest.mark.parametrize("kind", ["torch", "numpy"])
def test_table_roundtrip(tmp_path, kind):
    path = str(tmp_path / "tab.fits")
    values = np.array(EDGES, dtype=np.uint64)
    col = torch.from_numpy(values) if kind == "torch" else values
    torchfits.write(path, {"U": col}, overwrite=True)

    got = torchfits.read(path, hdu=1)["U"]
    assert got.dtype == torch.uint64
    assert got.numpy().tolist() == values.tolist()

    via_mmap = C.read_fits_table_rows(path, 1, ["U"], 1, -1, True)["U"]
    assert via_mmap.numpy().tolist() == values.tolist()

    arrow = table.read(path, hdu=1)
    assert str(arrow.schema.field("U").type) == "uint64"
    assert arrow["U"].to_pylist() == values.tolist()


@pytest.mark.parametrize(
    ("where", "pick"),
    [
        ("U >= 9223372036854775808", lambda v: v >= 2**63),
        ("U < 9223372036854775808", lambda v: v < 2**63),
        ("U == 5", lambda v: v == 5),
        ("U != 5", lambda v: v != 5),
        ("U > -1", lambda v: True),
        ("U <= 2.5", lambda v: v <= 2.5),
        ("U > 18446744073709551614", lambda v: v > 2**64 - 2),
    ],
)
def test_table_where(tmp_path, where, pick):
    path = str(tmp_path / "tab.fits")
    torchfits.write(path, {"U": np.array(EDGES, dtype=np.uint64)}, overwrite=True)
    expected = [v for v in EDGES if pick(v)]
    for kwargs in ({}, {"backend": "torch"}):
        got = table.read(path, hdu=1, where=where, **kwargs)["U"]
        got = got.to_pylist() if hasattr(got, "to_pylist") else got.numpy().tolist()
        assert got == expected, (where, kwargs)


def _raw_cards(path, hdu):
    """Header cards of HDU *hdu* straight from the 2880-byte blocks."""
    data = open(path, "rb").read()
    off = 0
    for idx in range(hdu + 1):
        cards = []
        while True:
            block = data[off : off + 2880]
            off += 2880
            chunk = [block[i : i + 80].decode("ascii") for i in range(0, 2880, 80)]
            cards.extend(chunk)
            if any(c.startswith("END     ") for c in chunk):
                break
        kv = {
            c[:8].strip(): c[10:].split("/")[0].strip()
            for c in cards
            if c[8:10] == "= "
        }
        if idx == hdu:
            return kv
        naxis = int(kv["NAXIS"])
        n = 0
        if naxis:
            n = 1
            for k in range(1, naxis + 1):
                n *= int(kv[f"NAXIS{k}"])
            n = (n + int(kv.get("PCOUNT", 0))) * abs(int(kv["BITPIX"])) // 8
        off += (n + 2879) // 2880 * 2880
    raise IndexError(hdu)


def _write_direct(path, col):
    torchfits.write(path, {"U": col})


def _write_compressed(path, col):
    torchfits.write(path, {"U": col}, compress=True)


def _write_hdulist(path, col):
    torchfits.write(path, torchfits.HDUList([torchfits.TableHDU({"U": col})]))


def _insert_into(path, col):
    torchfits.write(path, [torch.zeros(1, 1)])
    torchfits.insert_hdu(path, {"U": col}, index=1)


def _edit_other_hdu(path, col):
    torchfits.write(path, {"U": col})
    torchfits.insert_hdu(path, torch.ones(2, 2), index=2)
    torchfits.replace_hdu(path, 2, torch.zeros(2, 2))
    torchfits.delete_hdu(path, 2)


@pytest.mark.parametrize(
    "writer",
    [_write_direct, _write_compressed, _write_hdulist, _insert_into, _edit_other_hdu],
)
def test_table_tzero_card_is_exactly_two_pow_63(tmp_path, writer):
    # TZERO = 2**63 used to be clamped to LLONG_MAX on the direct path and to
    # make every rewrite path raise "out of long long range".
    path = str(tmp_path / "u.fits")
    col = _u64(EDGES)
    writer(path, col)
    text = _raw_cards(path, 1)["TZERO1"]
    assert float(text) == 2.0**63
    assert "." in text or "E" in text.upper()  # not the clamped integer
    got = torchfits.read(path, hdu=1)["U"]
    assert got.dtype == torch.uint64
    assert got.tolist() == EDGES
