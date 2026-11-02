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
