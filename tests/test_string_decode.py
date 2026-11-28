"""Fixed-width byte-column decode: correctness anchor for windowed views."""

import torch

from torchfits._string_decode import decode_byte_tensor


def test_decode_byte_tensor_full_column():
    t = torch.tensor(
        [[ord(c) for c in "STAR_A  "], [ord(c) for c in "QSO-B   "]],
        dtype=torch.uint8,
    )
    assert decode_byte_tensor(t) == ["STAR_A", "QSO-B"]


def test_decode_byte_tensor_offset_window():
    """A small row window of a large column decodes only its own rows.

    The window is contiguous with a large storage behind it; decoding must
    return exactly the window's rows (and must not copy the whole storage).
    """
    big = torch.zeros((1000, 20), dtype=torch.uint8)
    for row in (500, 501, 502):
        text = f"ROW{row}".ljust(20)
        big[row] = torch.tensor([ord(c) for c in text], dtype=torch.uint8)
    window = big[500:503]
    assert window.is_contiguous()
    assert decode_byte_tensor(window) == ["ROW500", "ROW501", "ROW502"]


def test_decode_byte_tensor_strided_view_and_width_zero():
    t = torch.zeros((4, 6), dtype=torch.uint8)
    t[0, 0:3] = ord("A")
    t[2, 0:3] = ord("B")
    assert decode_byte_tensor(t[::2]) == ["AAA", "BBB"]
    assert decode_byte_tensor(torch.zeros((3, 0), dtype=torch.uint8)) == ["", "", ""]


def test_space_padded_ascii_columns_decode_in_arrow(monkeypatch):
    # FITS pads strings with blanks, so nearly every real string column has a
    # row ending in a space; that used to send the whole column through numpy's
    # per-element rstrip + decode. Arrow trims the same " \x00" set.
    import numpy as np
    import pyarrow as pa

    from torchfits._table import arrow_convert

    rows = [b"ab  ", b"c\x00\x00\x00", b"  d ", b"e\x00 \x00", b"    ", b"wxyz"]
    matrix = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(len(rows), 4)

    def no_numpy_decode(*_args, **_kwargs):
        raise AssertionError("numpy decode used for 7-bit, space-padded data")

    monkeypatch.setattr(np.char, "decode", no_numpy_decode)
    for encoding in ("ascii", "utf-8"):
        out = arrow_convert._decode_uint8_matrix_to_arrow(pa, matrix, encoding, True)
        assert out.to_pylist() == ["ab", "c", "  d", "e", "", "wxyz"]
