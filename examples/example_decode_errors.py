"""
Example: string columns that are not ASCII -- ``encoding=`` and ``decode_errors=``.

FITS character columns are ASCII, and torchfits decodes them strictly: a byte
the encoding cannot decode raises ``UnicodeDecodeError`` naming the column
instead of silently turning into U+FFFD or vanishing. For a legacy file that
breaks the rule, either name the encoding it really uses or opt into a lossy
decode:

- ``encoding="latin-1"``      -- the bytes are Latin-1 text;
- ``decode_errors="replace"`` -- each bad byte becomes U+FFFD;
- ``decode_errors="ignore"``  -- bad bytes are dropped.

The same keywords work on ``table.read`` / ``scan`` / ``reader`` / ``schema``
and on the converters (``table.to_pandas``, ``table.read_polars``, ...).
torchfits never *writes* non-ASCII text, so the example patches one byte of
a file it wrote. Synthetic data only; every claim is an assertion.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

import torchfits


def _legacy_file(path: str) -> None:
    """A table whose NAME column holds one Latin-1 byte (0xE9, 'é')."""
    torchfits.table.write(
        path,
        {"ID": np.arange(3, dtype=np.int32), "NAME": ["M31", "Cafe#", "NGC 253"]},
        overwrite=True,
    )
    with open(path, "r+b") as f:
        raw = f.read()
        at = raw.index(b"Cafe#")  # the cell bytes in the data unit
        f.seek(at + 3)
        f.write(b"\xe9")  # 'Cafe#' -> b'Caf\xe9#' (Latin-1 'Café#')


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="torchfits_decode_") as tmp:
        path = os.path.join(tmp, "legacy.fits")
        _legacy_file(path)

        # Default: strict ASCII. The error names the column.
        try:
            torchfits.table.read(path)
        except UnicodeDecodeError as exc:
            assert "NAME" in str(exc)
            print(f"strict (default): UnicodeDecodeError: {str(exc).split(';')[0]}")
        else:
            raise AssertionError("a non-ASCII byte must not decode silently")

        # Numeric columns are unaffected: project them away.
        assert torchfits.table.read(path, columns=["ID"]).num_rows == 3

        names = torchfits.table.read(path, encoding="latin-1").column("NAME")
        assert names.to_pylist() == ["M31", "Café#", "NGC 253"]
        print(f"encoding='latin-1':      {names.to_pylist()}")

        names = torchfits.table.read(path, decode_errors="replace").column("NAME")
        assert names.to_pylist()[1] == "Caf�#"
        print(f"decode_errors='replace': {names.to_pylist()}")

        names = torchfits.table.read(path, decode_errors="ignore").column("NAME")
        assert names.to_pylist()[1] == "Caf#"
        print(f"decode_errors='ignore':  {names.to_pylist()}")

        # Streaming readers and converters take the same keywords.
        batch = next(iter(torchfits.table.scan(path, decode_errors="replace")))
        assert batch.column("NAME")[1].as_py() == "Caf�#"
        frame = torchfits.table.to_pandas(path, encoding="latin-1")
        assert frame["NAME"].tolist()[1] == "Café#"

        # Tensor readers return the raw bytes (uint8), so they never decode.
        raw = torchfits.table.read_torch(path, columns=["NAME"])["NAME"]
        assert bytes(raw[1].tolist()).rstrip() == b"Caf\xe9#"

    print("All decode_errors checks passed.")


if __name__ == "__main__":
    main()
