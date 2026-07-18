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
    # No checksum keywords → ok=True, exit 0 (not corrupt, just unverified).
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload[0]["ok"] is True
    assert payload[0]["status"] == "no_checksums"


def test_verify_after_write_checksums(image_fits):
    torchfits.write_checksums(str(image_fits), hdu=0)
    result = _run_cli("verify", str(image_fits))
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_verify_text_output_messaging_contract(tmp_path):
    """Lock in the three verify text-output labels in text mode:

    - "OK (no checksum keywords)" — file has no DATASUM/CHECKSUM keywords
    - "OK"                       — checksums present and valid
    - "FAIL"                     — checksums present but corrupt (exit 4)
    """
    path = tmp_path / "verify.fits"
    data = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    torchfits.write(str(path), data, header={"FOO": 1}, overwrite=True)

    # 1. No checksum keywords → OK (no checksum keywords), exit 0.
    result = _run_cli("verify", str(path))
    assert result.returncode == 0, result.stderr
    assert "OK (no checksum keywords)" in result.stdout
    assert "FAIL" not in result.stdout

    # 2. Write checksums → OK, exit 0.
    torchfits.write_checksums(str(path), hdu=0)
    result = _run_cli("verify", str(path))
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
    assert "no checksum keywords" not in result.stdout
    assert "FAIL" not in result.stdout

    # 3. Corrupt a non-structural header keyword (FOO) → CHECKSUM fails, exit 4.
    # Changing a non-structural keyword alters header bytes without making
    # the file unopenable, so CHECKSUM recomputation diverges from the stored value.
    with open(path, "r+b") as f:
        raw = f.read()
        needle = b"FOO     ="
        idx = raw.find(needle)
        assert idx != -1, "FOO card not found"
        card = bytearray(raw[idx : idx + 80])
        one = card.find(b"1")
        assert one != -1, "digit to corrupt not found in FOO card"
        card[one : one + 1] = b"2"
        f.seek(idx)
        f.write(card)

    result = _run_cli("verify", str(path))
    assert result.returncode == 4, result.stderr
    assert "FAIL" in result.stdout


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
        "--bands",
        "0,0,0",
    )
    assert result.returncode == 0, result.stderr
    assert out.read_bytes()[:4] == b"\x89PNG"


def test_convert_infers_format_from_extension(table_fits, tmp_path):
    out = tmp_path / "table.parquet"
    result = _run_cli("convert", str(table_fits), str(out), "--hdu", "1")
    assert result.returncode == 0, result.stderr
    assert out.is_file()


def test_convert_unknown_extension_needs_to(table_fits, tmp_path):
    out = tmp_path / "table.dat"
    result = _run_cli("convert", str(table_fits), str(out), "--hdu", "1")
    assert result.returncode == 2, result.stderr
    assert "--to" in result.stderr


def test_info_jsonl_format(image_fits):
    result = _run_cli("info", str(image_fits), "--format", "jsonl")
    assert result.returncode == 0, result.stderr
    line = result.stdout.strip().splitlines()[0]
    payload = json.loads(line)
    assert payload["type"] == "IMAGE"


def test_cutout_box(image_fits, tmp_path):
    out = tmp_path / "box.fits"
    result = _run_cli(
        "cutout", str(image_fits), str(out), "--box", "0,0,2,2", "--hdu", "0"
    )
    assert result.returncode == 0, result.stderr
    cut = torchfits.read_tensor(str(out), hdu=0)
    assert cut.shape == (2, 2)


def test_cutout_cfitsio_section(image_fits, tmp_path):
    out = tmp_path / "section.fits"
    # CFITSIO 1-based inclusive [1:2,1:2] ≈ torchfits --box 0,0,2,2
    sectioned = f"{image_fits}[1:2,1:2]"
    result = _run_cli("cutout", sectioned, str(out), "--hdu", "0")
    assert result.returncode == 0, result.stderr
    cut = torchfits.read_tensor(str(out), hdu=0)
    assert cut.shape == (2, 2)
    expected = torchfits.read_subset(str(image_fits), 0, 0, 0, 2, 2)
    assert torch.allclose(cut, expected)


def test_cutout_rejects_section_and_box(image_fits, tmp_path):
    out = tmp_path / "bad.fits"
    result = _run_cli(
        "cutout",
        f"{image_fits}[1:2,1:2]",
        str(out),
        "--box",
        "0,0,2,2",
    )
    assert result.returncode == 2, result.stderr


def test_open_cfitsio_section_exists(image_fits):
    sectioned = f"{image_fits}[1:2,1:2]"
    with torchfits.open(sectioned) as hdul:
        assert len(hdul) >= 1
    tensor = torchfits.read_tensor(sectioned, hdu=0)
    assert tensor.shape == (2, 2)


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


def test_convert_csv_tsv_arrow(table_fits, tmp_path):
    import pyarrow.csv as pacsv
    import pyarrow.feather as feather

    csv_out = tmp_path / "t.csv"
    tsv_out = tmp_path / "t.tsv"
    arrow_out = tmp_path / "t.arrow"
    for path, fmt in ((csv_out, "csv"), (tsv_out, "tsv"), (arrow_out, "arrow")):
        result = _run_cli(
            "convert", str(table_fits), str(path), "--to", fmt, "--hdu", "1"
        )
        assert result.returncode == 0, result.stderr
        assert path.is_file()
    assert pacsv.read_csv(csv_out).num_rows == 3
    assert (
        pacsv.read_csv(
            tsv_out, parse_options=pacsv.ParseOptions(delimiter="\t")
        ).num_rows
        == 3
    )
    assert feather.read_table(arrow_out).num_rows == 3


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


def test_header_keyword_table(image_fits):
    result = _run_cli(
        "header",
        str(image_fits),
        "--keyword-table",
        "-k",
        "BITPIX",
        "-k",
        "NAXIS",
        "-e",
        "0",
    )
    assert result.returncode == 0, result.stderr
    assert "BITPIX" in result.stdout
    assert "NAXIS" in result.stdout


def test_header_fitsort_alias_still_works(image_fits):
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


def test_header_keyword_table_json(image_fits):
    result = _run_cli(
        "header",
        str(image_fits),
        "--keyword-table",
        "-k",
        "BITPIX",
        "-e",
        "0",
        "-f",
        "json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert int(payload[0]["BITPIX"]) == -32


def test_info_short_format(image_fits):
    result = _run_cli("info", str(image_fits), "-f", "jsonl")
    assert result.returncode == 0, result.stderr
    assert "hdu" in result.stdout


def test_convert_where_filter(table_fits, tmp_path):
    out = tmp_path / "bright.parquet"
    result = _run_cli(
        "convert",
        str(table_fits),
        str(out),
        "-e",
        "1",
        "--where",
        "flux > 1.5",
        "--columns",
        "ra,dec,flux",
    )
    assert result.returncode == 0, result.stderr
    import pyarrow.parquet as pq

    table = pq.read_table(out)
    assert table.num_rows == 2
    assert table.column_names == ["ra", "dec", "flux"]


def test_convert_short_flags(table_fits, tmp_path):
    out = tmp_path / "short.parquet"
    result = _run_cli(
        "convert",
        str(table_fits),
        "-o",
        str(out),
        "-e",
        "1",
        "-w",
        "flux > 1.5",
        "-c",
        "ra,dec,flux",
    )
    assert result.returncode == 0, result.stderr
    import pyarrow.parquet as pq

    assert pq.read_table(out).num_rows == 2


def test_copy_dash_o(image_fits, tmp_path):
    out = tmp_path / "via_o.fits"
    result = _run_cli("copy", str(image_fits), "-o", str(out))
    assert result.returncode == 0, result.stderr
    assert out.is_file()


def test_probe_header_bytes_help():
    result = _run_cli("probe", "--help")
    assert result.returncode == 0
    assert "--header-bytes" in result.stdout


def test_invalid_hdu_is_usage_error(image_fits):
    result = _run_cli("info", str(image_fits), "-e", "not-an-int")
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
        cmds_probe._probe_vos("vos:example/file.fits", header_bytes=5760)


def test_vos_probe_bad_uri_is_io_error_when_vos_present():
    result = _run_cli("probe", "vos:sfabbro/example.fits", "--json")
    # vos may be installed (CANFAR lab) or missing; either way no traceback.
    assert result.returncode in (2, 3), result.stderr
    assert "vos" in result.stderr.lower() or "service" in result.stderr.lower()


def test_convert_png_invalid_bands_count(image_fits, tmp_path):
    out = tmp_path / "rgb.png"
    result = _run_cli(
        "convert", str(image_fits), str(out), "--to", "png", "--bands", "0,1"
    )
    assert result.returncode == 2, result.stderr
    assert "bands" in result.stderr.lower()


def test_convert_png_invalid_bands_non_integer(image_fits, tmp_path):
    out = tmp_path / "rgb.png"
    result = _run_cli(
        "convert", str(image_fits), str(out), "--to", "png", "--bands", "a,b,c"
    )
    assert result.returncode == 2, result.stderr


def test_fitsort_with_invalid_hdu(image_fits):
    result = _run_cli(
        "header",
        str(image_fits),
        "--keyword-table",
        "-k",
        "BITPIX",
        "-e",
        "abc",
    )
    assert result.returncode == 2, result.stderr
    assert "hdu" in result.stderr.lower() or "invalid" in result.stderr.lower()


def test_setkey_hierarch_and_rename(image_fits, tmp_path):
    out = tmp_path / "keyed.fits"
    result = _run_cli("copy", str(image_fits), str(out))
    assert result.returncode == 0, result.stderr
    result = _run_cli(
        "setkey",
        str(out),
        "--key",
        "ESO DET CHIP1 ID",
        "--value",
        "42",
    )
    assert result.returncode == 0, result.stderr
    hdr = torchfits.read_header(str(out), 0)
    # HIERARCH cards may surface as key="HIERARCH" with the long name in the value.
    blob = " ".join(f"{k}={v}" for k, v in hdr.items()).upper()
    assert "CHIP1" in blob
    result = _run_cli(
        "setkey",
        str(out),
        "--key",
        "OBJECT",
        "--value",
        "DEMO",
    )
    assert result.returncode == 0, result.stderr
    result = _run_cli("setkey", str(out), "--rename", "OBJECT=TARGET")
    assert result.returncode == 0, result.stderr
    hdr = torchfits.read_header(str(out), 0)
    assert "TARGET" in hdr
    assert hdr["TARGET"] == "DEMO"


def test_http_probe_blocks_internal_ssrf():
    result = _run_cli("probe", "http://127.0.0.1:8000/latest/meta-data/")
    assert result.returncode == 3
    assert "access to internal or private networks is blocked" in result.stderr

    result = _run_cli("probe", "http://localhost:8080/")
    assert result.returncode == 3
    assert "access to internal or private networks is blocked" in result.stderr

    result = _run_cli("probe", "http://169.254.169.254/")
    assert result.returncode == 3
    assert "access to internal or private networks is blocked" in result.stderr
