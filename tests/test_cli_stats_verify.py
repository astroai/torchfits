"""CLI regression tests for ``torchfits stats`` / ``torchfits verify`` (R7-B).

Covers the JSON non-finite contract (NaN/±inf serialize as ``null``), empty
region stats, per-command exit codes vs the docs/cli.md exit table, and I/O
error mapping for truncated inputs.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
import torch

import torchfits
import torchfits.table as tf_table


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "torchfits.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _write_empty_image(path) -> None:
    torchfits.write(str(path), torch.zeros((0, 4)), overwrite=True)


def test_stats_empty_region_json_yields_null(tmp_path):
    """Empty region stats must parse as JSON with None values (decision 3)."""
    path = tmp_path / "empty.fits"
    _write_empty_image(path)
    result = _run_cli("stats", str(path), "-f", "json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert len(payload) == 1
    for key in ("min", "max", "mean", "std", "median"):
        assert payload[0][key] is None


def test_stats_empty_region_jsonl_yields_null(tmp_path):
    path = tmp_path / "empty.fits"
    _write_empty_image(path)
    result = _run_cli("stats", str(path), "-f", "jsonl")
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    for key in ("min", "max", "mean", "std", "median"):
        assert record[key] is None


def test_stats_nonfinite_floats_serialize_as_null(tmp_path):
    """NaN statistics serialize as null: JSON has no NaN/Infinity.

    Non-finite pixels are excluded, so an image with no finite pixel has
    undefined (NaN -> null) statistics and counts every pixel in ``nbad``.
    """
    path = tmp_path / "nan.fits"
    torchfits.write(
        str(path),
        torch.tensor([[float("nan"), float("-inf")], [float("nan"), float("inf")]]),
        overwrite=True,
    )
    for fmt in ("json", "jsonl"):
        result = _run_cli("stats", str(path), "-f", fmt)
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)  # must not contain bare NaN tokens
        record = payload[0] if fmt == "json" else payload
        for key in ("min", "max", "mean", "std", "median"):
            assert record[key] is None
        assert record["nbad"] == 4


def test_table_json_preview_nonfinite_null(tmp_path):
    """table -f json/jsonl previews serialize non-finite floats as null."""
    path = tmp_path / "tbl.fits"
    tf_table.write(str(path), {"A": np.array([1.0, np.nan, np.inf])}, overwrite=True)
    result = _run_cli("table", str(path), "-n", "3", "-f", "json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    preview = payload[0]["preview"]
    assert preview[0]["A"] == 1.0
    assert preview[1]["A"] is None
    assert preview[2]["A"] is None

    result = _run_cli("table", str(path), "-n", "3", "-f", "jsonl")
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["preview"][1]["A"] is None
    assert record["preview"][2]["A"] is None


def test_table_empty_preview_ok(tmp_path):
    """Zero-row tables preview as an empty list, exit 0."""
    path = tmp_path / "emptytbl.fits"
    tf_table.write(str(path), {"A": np.array([], dtype=np.float64)}, overwrite=True)
    result = _run_cli("table", str(path), "-n", "5", "-f", "json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload[0]["nrows"] == 0
    assert payload[0]["preview"] == []


def test_table_text_preview_is_valid_json(tmp_path):
    """The text-mode preview block is json.dumps output and must be valid JSON."""
    path = tmp_path / "tbl.fits"
    tf_table.write(str(path), {"A": np.array([1.0, np.nan, np.inf])}, overwrite=True)
    result = _run_cli("table", str(path), "-n", "3")
    assert result.returncode == 0, result.stderr
    marker = "preview:\n"
    assert marker in result.stdout
    preview = json.loads(result.stdout.split(marker, 1)[1])
    assert preview[0]["A"] == 1.0
    assert preview[1]["A"] is None
    assert preview[2]["A"] is None


def test_stats_invalid_hdu_exits_usage(tmp_path):
    path = tmp_path / "img.fits"
    torchfits.write(str(path), torch.zeros((2, 2)), overwrite=True)
    result = _run_cli("stats", str(path), "-e", "99")
    assert result.returncode == 2, result.stderr
    assert "out of range" in result.stderr


def test_table_invalid_hdu_exits_usage(tmp_path):
    path = tmp_path / "tbl.fits"
    tf_table.write(str(path), {"A": np.array([1.0, 2.0])}, overwrite=True)
    result = _run_cli("table", str(path), "-e", "99")
    assert result.returncode == 2, result.stderr
    assert "out of range" in result.stderr


def test_verify_invalid_hdu_exits_usage(tmp_path):
    path = tmp_path / "img.fits"
    torchfits.write(str(path), torch.zeros((2, 2)), overwrite=True)
    result = _run_cli("verify", str(path), "-e", "99")
    assert result.returncode == 2, result.stderr
    assert "out of range" in result.stderr


def test_stats_truncated_image_is_io_error(tmp_path):
    """Unreadable image data maps to exit 3 (invalid FITS structure), not a crash."""
    path = tmp_path / "trunc.fits"
    full = tmp_path / "full.fits"
    torchfits.write(
        str(full), torch.arange(16, dtype=torch.float32).reshape(4, 4), overwrite=True
    )
    raw = full.read_bytes()
    assert len(raw) > 2880
    path.write_bytes(raw[:-2880])
    result = _run_cli("stats", str(path))
    assert result.returncode == 3, result.stderr
    assert "Traceback" not in result.stderr
    # The raw CFITSIO status (108) is explained, not left bare.
    assert "truncated or unreadable FITS file" in result.stderr


def test_truncated_file_errors_are_explained_exit_3(tmp_path, capsys):
    from torchfits.cli.main import main

    full = tmp_path / "full.fits"
    torchfits.write(
        str(full),
        [torch.zeros(10, 10), torch.ones(100, 100), torch.ones(50, 50)],
        overwrite=True,
    )
    raw = full.read_bytes()
    data_cut = tmp_path / "data_cut.fits"
    data_cut.write_bytes(raw[: 2880 * 3 + 1000])  # inside HDU 1's data
    header_cut = tmp_path / "header_cut.fits"
    header_cut.write_bytes(raw[:1000])  # inside the primary header
    scan_cut = tmp_path / "scan_cut.fits"
    scan_cut.write_bytes(raw[: 5760 + 1000])  # inside HDU 1's header
    cases = [
        (["diff", str(data_cut), str(full)], "truncated or unreadable"),
        (
            [
                "arith",
                str(data_cut),
                "--op",
                "add",
                "--value",
                "1",
                "-o",
                str(tmp_path / "o.fits"),
            ],
            "truncated or unreadable",
        ),
        (["info", str(header_cut)], "truncated or unreadable"),
        (["header", str(scan_cut)], "appears truncated or corrupt"),
    ]
    for argv, message in cases:
        assert main(argv) == 3, argv
        err = capsys.readouterr().err
        assert message in err, (argv, err)
        assert "Traceback" not in err


def test_verify_truncated_checksummed_is_io_error(tmp_path):
    full = tmp_path / "full.fits"
    torchfits.write(
        str(full), torch.arange(16, dtype=torch.float32).reshape(4, 4), overwrite=True
    )
    torchfits.write_checksums(str(full), hdu=0)
    raw = full.read_bytes()
    path = tmp_path / "trunc.fits"
    path.write_bytes(raw[:-2880])
    result = _run_cli("verify", str(path))
    assert result.returncode == 3, result.stderr
    assert "Traceback" not in result.stderr


def test_verify_truncated_without_checksums_is_io_error(tmp_path, capsys):
    """A data-truncated file with no checksum keywords is not "OK".

    The structural walk sees every HDU's declared extent, so a file cut in
    its last data unit (which ``torchfits.open`` still accepts) exits 3
    instead of reporting "OK (no checksum keywords)". Compressed and
    complete files keep their verdicts.
    """
    from torchfits.cli.main import main

    full = tmp_path / "full.fits"
    torchfits.write(
        str(full), [torch.zeros(10, 10), torch.ones(100, 100)], overwrite=True
    )
    raw = full.read_bytes()
    cut = tmp_path / "cut.fits"
    cut.write_bytes(raw[: len(raw) - 5000])
    assert main(["verify", str(cut)]) == 3
    err = capsys.readouterr().err
    assert "truncated or unreadable FITS file" in err
    assert "HDU 1: data truncated" in err
    assert "Traceback" not in err
    # Selecting another HDU does not hide a truncated file.
    assert main(["verify", str(cut), "--hdu", "0"]) == 3
    capsys.readouterr()
    # An unpadded final block is tolerated, as CFITSIO tolerates it.
    unpadded = tmp_path / "unpadded.fits"
    unpadded.write_bytes(raw[: len(raw) - (-(100 * 100 * 4) % 2880)])
    for path in (full, unpadded):
        assert main(["verify", str(path)]) == 0
        assert "OK (no checksum keywords)" in capsys.readouterr().out


def test_verify_one_corrupt_file_exits_4_under_file_jobs(tmp_path):
    """Any failing HDU across the batch exits 4 even with -J fan-out."""
    good = tmp_path / "good.fits"
    bad = tmp_path / "bad.fits"
    data = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    torchfits.write(str(good), data, header={"FOO": 1}, overwrite=True)
    torchfits.write(str(bad), data, header={"FOO": 1}, overwrite=True)
    torchfits.write_checksums(str(good), hdu=0)
    torchfits.write_checksums(str(bad), hdu=0)
    # Corrupt a non-structural keyword (FOO card): the file stays openable but
    # the stored CHECKSUM no longer matches the header bytes.
    raw = bytearray(bad.read_bytes())
    idx = raw.find(b"FOO     =")
    assert idx != -1
    one = raw[idx : idx + 80].find(b"1")
    assert one != -1
    raw[idx + one : idx + one + 1] = b"2"
    bad.write_bytes(raw)
    result = _run_cli("verify", str(good), str(bad), "-J", "2", "-f", "jsonl")
    assert result.returncode == 4, result.stderr


def _table_with_odd_values(path):
    tf_table.write(
        str(path),
        {
            "C": np.array([1 + 2j, np.nan - 4j], dtype=np.complex64),
            "V": np.ones((2, 3), dtype=np.complex128),
            "S": np.array(["ab", "Qz"]),
            "I": np.array([7, 8], dtype=np.int32),
        },
        overwrite=True,
    )
    raw = path.read_bytes()
    at = raw.rfind(b"Qz")
    path.write_bytes(raw[:at] + b"\xff\xfe" + raw[at + 2 :])


def test_table_preview_json_complex_and_undecodable_strings(tmp_path, capsys):
    import json

    from torchfits.cli.main import main

    path = tmp_path / "odd.fits"
    _table_with_odd_values(path)
    assert main(["table", str(path), "-n", "5", "-f", "json"]) == 0
    (record,) = json.loads(capsys.readouterr().out)
    assert record["schema"] == [
        {"name": "C", "type": "complex64"},
        {"name": "V", "type": "fixed_size_list<item: complex128>[3]"},
        {"name": "S", "type": "string"},
        {"name": "I", "type": "int32"},
    ]
    first, second = record["preview"]
    assert first == {
        "C": {"re": 1.0, "im": 2.0},
        "V": [{"re": 1.0, "im": 0.0}] * 3,
        "S": "ab",
        "I": 7,
    }
    assert second["C"] == {"re": None, "im": -4.0}
    assert second["S"] == "��"

    # Text mode prints the same JSON preview block.
    assert main(["table", str(path), "-n", "1"]) == 0
    out = capsys.readouterr().out
    assert '"re": 1.0' in out
    assert "(1+2j)" not in out


def test_json_safe_bytes_and_complex():
    from torchfits.cli.common import _json_safe

    assert _json_safe(b"ok\xff") == "ok�"
    assert _json_safe(np.complex64(1 - 2j)) == {"re": 1.0, "im": -2.0}
    assert _json_safe([complex(float("inf"), 0)]) == [{"re": None, "im": 0.0}]


@pytest.mark.parametrize(
    "argv",
    [
        ["info", "{img}", "-e", "99"],
        ["header", "{img}", "-e", "99"],
        ["stats", "{img}", "-e", "99"],
        ["verify", "{img}", "-e", "99"],
        ["probe", "{img}", "-e", "99"],
        ["table", "{tbl}", "-e", "99"],
        ["cutout", "{img}", "-o", "{out}.fits", "-e", "99", "--box", "0,0,2,2"],
        ["convert", "{tbl}", "{out}.csv", "-e", "99"],
        ["convert", "{tbl}", "{out}.fits", "-e", "99"],
        ["convert", "{img}", "-o", "{out}.png", "--to", "png", "--bands", "99"],
        [
            "arith",
            "{img}",
            "--op",
            "add",
            "--value",
            "1",
            "-o",
            "{out}.fits",
            "-e",
            "99",
        ],
        ["transform", "{img}", "-o", "{out}.fits", "-e", "99", "--name", "LogStretch"],
        ["compress", "{img}", "-e", "99", "--split", "hdu", "--out-dir", "{dir}"],
        ["decompress", "{img}", "-e", "99", "--split", "hdu", "--out-dir", "{dir}"],
    ],
    ids=lambda argv: "-".join(a for a in argv[:4] if not a.startswith("{")),
)
def test_every_subcommand_reports_a_missing_hdu_as_usage_error(tmp_path, argv, capsys):
    # cutout, convert and transform used to exit 3 for an -e index the file
    # does not have; info/header/stats/table/verify/... exit 2.
    from torchfits.cli.main import main

    img = tmp_path / "img.fits"
    torchfits.write(str(img), torch.zeros((4, 4)))
    tbl = tmp_path / "tbl.fits"
    tf_table.write(str(tbl), {"A": np.array([1.0, 2.0])})
    fields = {"img": img, "tbl": tbl, "out": tmp_path / "out", "dir": tmp_path / "d"}
    assert main([a.format(**fields) for a in argv]) == 2
    assert "out of range" in capsys.readouterr().err
