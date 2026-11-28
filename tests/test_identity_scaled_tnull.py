"""A TNULL next to identity TSCAL/TZERO cards reads as float64 everywhere.

The native readers scale such a column (TNULL cells decode to NaN / null), so
the header-only ``table.schema()`` must report float64 too, not the storage
integer type.
"""

from __future__ import annotations

import numpy as np
import pytest

import torchfits


@pytest.mark.parametrize(
    "spec",
    [
        {"format": "J", "bscale": 1.0, "bzero": 0.0, "tnull": 2},
        {"format": "J", "bzero": 0.0, "tnull": 2},
        {"format": "B", "bscale": 1.0, "tnull": 2},
        {"format": "3J", "bscale": 1.0, "tnull": 2},
    ],
)
def test_schema_matches_read_for_identity_scaled_tnull(tmp_path, spec):
    path = str(tmp_path / "s.fits")
    dtype = np.uint8 if spec["format"] == "B" else np.int32
    values = np.arange(6, dtype=dtype)
    if spec["format"] == "3J":
        values = values.reshape(2, 3)
    torchfits.table.write(path, {"S": values}, schema={"S": spec})
    read_type = torchfits.table.read(path).schema.field("S").type
    assert torchfits.table.schema(path).field("S").type == read_type
    assert "double" in str(read_type)
