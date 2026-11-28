"""``table.write_ipc`` writes Feather V2 with the IPC writer, not write_feather.

``pyarrow.feather.write_feather`` is deprecated (FutureWarning since pyarrow
24); the non-streaming ``write_ipc`` now writes the same Arrow IPC file with
``pyarrow.ipc.new_file`` and keeps write_feather's ``compression=`` meaning.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

import torchfits

pa = pytest.importorskip("pyarrow")
ipc = pytest.importorskip("pyarrow.ipc")


@pytest.fixture
def data():
    return pa.table(
        {"A": np.zeros(50_000, dtype=np.int64), "S": ["x"] * 50_000},
        metadata={b"k": b"v"},
    )


@pytest.mark.parametrize("compression", ["zstd", "lz4", "uncompressed", None])
def test_write_ipc_without_deprecation_warning(tmp_path, data, compression):
    out = str(tmp_path / "t.arrow")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        torchfits.table.write_ipc(out, data, compression=compression)
    with ipc.open_file(out) as reader:
        # 64Ki-row chunks, as write_feather wrote them.
        assert reader.num_record_batches == 1
        got = reader.read_all()
    assert got.equals(data)
    assert got.schema.metadata == {b"k": b"v"}


def test_write_ipc_compression_is_applied(tmp_path, data):
    sizes = {}
    for compression in ("zstd", "uncompressed"):
        out = str(tmp_path / f"{compression}.arrow")
        torchfits.table.write_ipc(out, data, compression=compression)
        sizes[compression] = os.path.getsize(out)
    assert sizes["zstd"] * 4 < sizes["uncompressed"]


def test_write_ipc_rejects_unknown_codec_and_keeps_destination(tmp_path, data):
    out = tmp_path / "t.arrow"
    out.write_bytes(b"keep")
    with pytest.raises(ValueError, match="not supported"):
        torchfits.table.write_ipc(str(out), data, compression="snappy")
    assert out.read_bytes() == b"keep"
