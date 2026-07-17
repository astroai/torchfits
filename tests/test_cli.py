"""Smoke tests for the torchfits CLI."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch

import torchfits


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "torchfits.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def image_fits(tmp_path):
    path = tmp_path / "image.fits"
    data = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    torchfits.write(str(path), data, header={"BITPIX": -32}, overwrite=True)
    return path


def test_help_lists_subcommands():
    result = _run_cli("--help")
    assert result.returncode == 0
    for name in (
        "info",
        "header",
        "verify",
        "stats",
        "table",
        "convert",
        "probe",
        "copy",
        "arith",
        "diff",
    ):
        assert name in result.stdout


def test_info_json(image_fits):
    result = _run_cli("info", str(image_fits), "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload[0]["file"] == str(image_fits)
    assert payload[0]["hdu"] == 0
    assert payload[0]["type"] == "IMAGE"


def test_header_keyword(image_fits):
    result = _run_cli("header", str(image_fits), "--keyword", "BITPIX", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert any(row["keyword"] == "BITPIX" for row in payload)


def test_verify_without_checksums(image_fits):
    result = _run_cli("verify", str(image_fits), "--json")
    assert result.returncode == 4, result.stderr
    payload = json.loads(result.stdout)
    assert payload[0]["ok"] is False


def test_verify_after_write_checksums(image_fits):
    torchfits.write_checksums(str(image_fits), hdu=0)
    result = _run_cli("verify", str(image_fits))
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_copy_roundtrip(image_fits, tmp_path):
    out = tmp_path / "copy.fits"
    result = _run_cli("copy", str(image_fits), str(out))
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    copied = torchfits.read_tensor(str(out), hdu=0)
    original = torchfits.read_tensor(str(image_fits), hdu=0)
    assert torch.equal(copied, original)


def test_arith_add(image_fits, tmp_path):
    out = tmp_path / "arith.fits"
    result = _run_cli(
        "arith",
        str(image_fits),
        "--op",
        "add",
        "--value",
        "1",
        "--out",
        str(out),
    )
    assert result.returncode == 0, result.stderr
    tensor = torchfits.read_tensor(str(out), hdu=0)
    expected = torchfits.read_tensor(str(image_fits), hdu=0) + 1.0
    assert torch.allclose(tensor, expected)


def test_diff_same_file(image_fits):
    result = _run_cli("diff", str(image_fits), str(image_fits))
    assert result.returncode == 0, result.stderr


def test_diff_detects_change(image_fits, tmp_path):
    other = tmp_path / "other.fits"
    data = torchfits.read_tensor(str(image_fits), hdu=0) + 10.0
    torchfits.write(str(other), data, overwrite=True)
    result = _run_cli("diff", str(image_fits), str(other))
    assert result.returncode == 1, result.stderr
    assert result.stderr.strip()


def test_convert_png(image_fits, tmp_path):
    out = tmp_path / "rgb.png"
    result = _run_cli(
        "convert",
        str(image_fits),
        str(out),
        "--to",
        "png",
        "--bands",
        "0,0,0",
    )
    assert result.returncode == 0, result.stderr
    assert out.read_bytes()[:4] == b"\x89PNG"


def test_convert_parquet(table_fits, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    out = tmp_path / "table.parquet"
    result = _run_cli(
        "convert",
        str(table_fits),
        str(out),
        "--to",
        "parquet",
        "--hdu",
        "1",
    )
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    assert pq.read_table(str(out)).num_rows == 3


@pytest.fixture
def table_fits(tmp_path):
    import numpy as np
    from astropy.io import fits
    from astropy.table import Table

    path = tmp_path / "table.fits"
    data = {
        "ra": np.array([200.0, 201.0, 202.0], dtype=np.float64),
        "dec": np.array([45.0, 46.0, 47.0], dtype=np.float64),
        "flux": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }
    fits.BinTableHDU(Table(data), name="CAT").writeto(str(path), overwrite=True)
    return path


def test_header_fitsort_table(image_fits):
    result = _run_cli(
        "header",
        str(image_fits),
        "--fitsort",
        "--keyword",
        "BITPIX",
        "--keyword",
        "NAXIS",
        "--hdu",
        "0",
    )
    assert result.returncode == 0, result.stderr
    assert "BITPIX" in result.stdout
    assert "NAXIS" in result.stdout


def test_header_fitsort_json(image_fits):
    result = _run_cli(
        "header",
        str(image_fits),
        "--fitsort",
        "--keyword",
        "BITPIX",
        "--hdu",
        "0",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert int(payload[0]["BITPIX"]) == -32


def test_invalid_hdu_is_usage_error(image_fits):
    result = _run_cli("info", str(image_fits), "--hdu", "not-an-int")
    assert result.returncode == 2, result.stderr


def test_vos_probe_missing_package_message(monkeypatch):
    import builtins

    from torchfits.cli import cmds_probe
    from torchfits.cli.common import UsageError

    real_import = builtins.__import__

    def _block_vos(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vos" or name.startswith("vos."):
            raise ImportError("vos blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _block_vos)
    with pytest.raises(UsageError, match="vos"):
        cmds_probe._probe_vos("vos:example/file.fits")


def test_vos_probe_bad_uri_is_io_error_when_vos_present():
    result = _run_cli("probe", "vos:sfabbro/example.fits", "--json")
    # vos may be installed (CANFAR lab) or missing; either way no traceback.
    assert result.returncode in (2, 3), result.stderr
    assert "vos" in result.stderr.lower() or "service" in result.stderr.lower()
