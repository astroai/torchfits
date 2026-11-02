"""CLI regression tests for audit findings (diff, stats, cutout, setkey, arith,
compress/decompress, convert).

Each test drives ``torchfits.cli.main.main`` in-process and checks exit codes
and written files.
"""

from __future__ import annotations

import numpy as np
import torch

import torchfits
import torchfits.table as tf_table
from torchfits.cli.main import main


# --- diff: exit 0 means identical data, not identical summaries -------------


def test_diff_detects_int_difference_below_float32_precision(tmp_path, capsys):
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    torchfits.write(str(a), torch.tensor([[16777217, 16777216]], dtype=torch.int32))
    torchfits.write(str(b), torch.tensor([[16777216, 16777216]], dtype=torch.int32))
    assert main(["diff", str(a), str(b)]) == 1
    assert "1 of 2 pixels differ" in capsys.readouterr().err


def test_diff_detects_permuted_pixels(tmp_path, capsys):
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    torchfits.write(str(a), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    torchfits.write(str(b), torch.tensor([[4.0, 3.0], [2.0, 1.0]]))
    assert main(["diff", str(a), str(b)]) == 1
    assert "4 of 4 pixels differ" in capsys.readouterr().err


def test_diff_identical_nan_images_are_equal(tmp_path):
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    data = torch.tensor([[1.0, float("nan")], [3.0, 4.0]])
    torchfits.write(str(a), data)
    torchfits.write(str(b), data.clone())
    assert main(["diff", str(a), str(b)]) == 0


def _write_cat(path, ra):
    tf_table.write(
        str(path),
        {
            "RA": np.array(ra),
            "NAME": np.array(["a", "bb", "ccc"]),
            "ID": np.array([1, 2, 3], np.int64),
        },
        overwrite=True,
    )


def test_diff_compares_table_data(tmp_path, capsys):
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    _write_cat(a, [1.0, 2.0, float("nan")])
    _write_cat(b, [1.0, 2.0, 5.0])
    assert main(["diff", str(a), str(b)]) == 1
    assert "column 'RA': 1 of 3 rows differ" in capsys.readouterr().err


def test_diff_identical_tables_with_nan_are_equal(tmp_path):
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    _write_cat(a, [1.0, 2.0, float("nan")])
    _write_cat(b, [1.0, 2.0, float("nan")])
    assert main(["diff", str(a), str(b)]) == 0


# --- stats: float64, finite pixels only, midpoint median --------------------


def _stats_json(capsys, path):
    import json

    capsys.readouterr()
    assert main(["stats", str(path), "-e", "0", "-f", "json"]) == 0
    (record,) = json.loads(capsys.readouterr().out)
    return record


def test_stats_excludes_nan_and_counts_it(tmp_path, capsys):
    path = tmp_path / "nan.fits"
    torchfits.write(str(path), torch.tensor([[1.0, float("nan")], [3.0, 4.0]]))
    record = _stats_json(capsys, path)
    assert record["nbad"] == 1
    assert record["min"] == 1.0
    assert record["max"] == 4.0
    assert record["mean"] == np.mean([1.0, 3.0, 4.0])
    assert record["median"] == 3.0


def test_stats_median_is_midpoint_and_float64(tmp_path, capsys):
    path = tmp_path / "big.fits"
    data = torch.tensor([[16777217, 16777216, 1, 2]], dtype=torch.int32)
    torchfits.write(str(path), data)
    record = _stats_json(capsys, path)
    assert record["nbad"] == 0
    assert record["max"] == 16777217
    assert record["median"] == np.median(data.numpy().astype(np.float64))
    assert record["mean"] == np.mean(data.numpy().astype(np.float64))


def test_stats_int64_extrema_are_exact(tmp_path, capsys):
    path = tmp_path / "i64.fits"
    torchfits.write(str(path), torch.tensor([[2**53 + 1, 2**53]], dtype=torch.int64))
    record = _stats_json(capsys, path)
    assert record["max"] == 2**53 + 1
    assert record["min"] == 2**53


# --- cutout ------------------------------------------------------------------


def _compressed_wide_image(tmp_path):
    """20 rows x 30 columns, tile-compressed into HDU 1 (tile table is 8x20)."""
    src = tmp_path / "wide.fits"
    data = torch.arange(20 * 30, dtype=torch.float32).reshape(20, 30)
    torchfits.write(str(src), data, header={"CRPIX1": 10.0, "CRPIX2": 20.0})
    packed = tmp_path / "wide_c.fits"
    assert main(["compress", str(src), str(packed)]) == 0
    return packed, data


def test_cutout_box_uses_compressed_image_geometry(tmp_path):
    packed, data = _compressed_wide_image(tmp_path)
    out = tmp_path / "stamp.fits"
    # x in [20, 25) is inside the 30-pixel-wide image but beyond the tile
    # table's NAXIS1 (8 bytes): it must not be refused.
    argv = ["cutout", str(packed), "-e", "1", "--box", "20,0,25,5", "-o", str(out)]
    assert main(argv) == 0
    assert torch.equal(torchfits.read_tensor(str(out), hdu=0), data[0:5, 20:25])
    # y in [25, 30) lies beyond the 20 image rows: no pixels, usage error.
    out2 = tmp_path / "miss.fits"
    argv = ["cutout", str(packed), "-e", "1", "--box", "0,25,5,30", "-o", str(out2)]
    assert main(argv) == 2
    assert not out2.exists()


def test_info_compressed_shape_is_row_major_like_uncompressed(tmp_path, capsys):
    import json

    packed, data = _compressed_wide_image(tmp_path)
    capsys.readouterr()
    assert main(["info", str(tmp_path / "wide.fits"), str(packed), "-f", "json"]) == 0
    records = json.loads(capsys.readouterr().out)
    shapes = [r["shape"] for r in records if r.get("type") == "IMAGE"]
    assert shapes == [str(tuple(data.shape))] * 2


def test_setkey_delete_and_rename_hierarch_keys(tmp_path):
    path = tmp_path / "h.fits"
    torchfits.write(str(path), torch.zeros(2, 2))
    assert main(["setkey", str(path), "-k", "ESO DET DIT", "--value", "1.5"]) == 0
    assert main(["setkey", str(path), "-k", "VERYLONGKEYWORD", "--value", "3"]) == 0
    # Updating an existing HIERARCH key must not add a second card.
    assert main(["setkey", str(path), "-k", "ESO DET DIT", "--value", "2.5"]) == 0
    keys = [c.key for c in torchfits.read_header(str(path), 0).cards]
    assert keys.count("ESO DET DIT") == 1
    assert main(["setkey", str(path), "--delete", "ESO DET DIT"]) == 0
    assert main(["setkey", str(path), "--rename", "VERYLONGKEYWORD=OTHERLONGKEY"]) == 0
    header = torchfits.read_header(str(path), 0)
    assert "ESO DET DIT" not in header
    assert "VERYLONGKEYWORD" not in header
    assert header["OTHERLONGKEY"] == 3


def test_cutout_section_with_extension_flag_is_usage_error(tmp_path):
    src = tmp_path / "mef.fits"
    torchfits.write(str(src), [torch.ones(4, 4), torch.zeros(4, 4)])
    out = tmp_path / "o.fits"
    assert main(["cutout", f"{src}[1:2,1:2]", "-e", "1", "-o", str(out)]) == 2
    assert not out.exists()
    # The supported spelling puts the extension in the path.
    assert main(["cutout", f"{src}[1][1:2,1:2]", "-o", str(out)]) == 0
    assert torch.equal(torchfits.read_tensor(str(out), hdu=0), torch.zeros(2, 2))
