"""``arith`` evaluation edges: integer saturation must not wrap, fractions warn.

The integer accumulation path promises "no silent integer wraparound" (see
``cmds_arith._compute``): out-of-range results saturate to the output dtype
with a ``RuntimeWarning``. Values cast from float arithmetic to an integer
output cannot keep fractions, but the loss must be warned about, not silent.
"""

from __future__ import annotations

import subprocess
import sys

import torch

import torchfits


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "torchfits.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_arith_uint32_mul_saturates_instead_of_wrapping(tmp_path):
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    big = torch.tensor([[4294967290, 4294967290], [1, 4000000000]], dtype=torch.uint32)
    torchfits.write(str(a), big, overwrite=True)
    torchfits.write(str(b), big, overwrite=True)
    out = tmp_path / "prod.fits"
    result = _run_cli("arith", str(a), str(b), "--op", "mul", "-o", str(out))
    assert result.returncode == 0, result.stderr
    assert "saturated" in result.stderr
    got = torchfits.read_tensor(str(out), hdu=0).to(torch.int64)
    # 4294967290^2 and 4000000000^2 overflow uint32 and must saturate to the
    # type maximum; 1*1 is exact. A wrapping int64 accumulator produces
    # garbage (0s) here instead.
    assert got.flatten().tolist() == [4294967295, 4294967295, 1, 4294967295]


def test_arith_fractional_scalar_on_integer_warns(tmp_path):
    img = tmp_path / "img.fits"
    torchfits.write(
        str(img),
        torch.arange(100, 116, dtype=torch.int16).reshape(4, 4),
        overwrite=True,
    )
    out = tmp_path / "out.fits"
    result = _run_cli(
        "arith", str(img), "--op", "add", "--value", "0.5", "-o", str(out)
    )
    assert result.returncode == 0, result.stderr
    # --dtype auto keeps the input dtype: 100 + 0.5 cannot keep the .5 in an
    # int16 output, and the dropped fraction must be warned about.
    assert "fractional" in result.stderr
    got = torchfits.read_tensor(str(out), hdu=0)
    assert got.flatten()[:4].tolist() == [100, 101, 102, 103]


def test_arith_div_by_zero_refused(tmp_path):
    img = tmp_path / "img.fits"
    torchfits.write(str(img), torch.ones(2, 2), overwrite=True)
    zero = tmp_path / "zero.fits"
    torchfits.write(str(zero), torch.zeros(2, 2), overwrite=True)

    result = _run_cli(
        "arith",
        str(img),
        "--op",
        "div",
        "--value",
        "0",
        "-o",
        str(tmp_path / "o1.fits"),
    )
    assert result.returncode == 2, result.stderr
    assert "division by zero" in result.stderr

    result = _run_cli(
        "arith", str(img), str(zero), "--op", "div", "-o", str(tmp_path / "o2.fits")
    )
    assert result.returncode == 2, result.stderr
    assert "division by zero" in result.stderr


# -- 64-bit integer images: no wraparound, no float64 rounding ----------------

_I64 = torch.iinfo(torch.int64)
_U64_MAX = 2**64 - 1


def _arith_main(tmp_path, values, dtype, *argv, operand=None):
    import warnings

    from torchfits.cli.main import main

    a = tmp_path / "a.fits"
    torchfits.write(str(a), torch.tensor([values], dtype=dtype), overwrite=True)
    args = ["arith", str(a), *argv]
    if operand is not None:
        b = tmp_path / "b.fits"
        torchfits.write(
            str(b), torch.tensor([operand[0]], dtype=operand[1]), overwrite=True
        )
        args = ["arith", str(a), str(b), *argv]
    out = tmp_path / "out.fits"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert main([*args, "-o", str(out)]) == 0
    got = torchfits.read_tensor(str(out), hdu=0)
    saturated = sum(
        int(str(w.message).split()[1]) for w in caught if "saturated" in str(w.message)
    )
    return got, saturated


def test_arith_int64_add_scalar_saturates(tmp_path):
    got, saturated = _arith_main(
        tmp_path, [_I64.max, 5], torch.int64, "--op", "add", "--value", "1"
    )
    assert got.dtype == torch.int64
    assert got.flatten().tolist() == [_I64.max, 6]
    assert saturated == 1


def test_arith_int64_sub_scalar_saturates(tmp_path):
    got, saturated = _arith_main(
        tmp_path, [_I64.min, 5], torch.int64, "--op", "sub", "--value", "1"
    )
    assert got.flatten().tolist() == [_I64.min, 4]
    assert saturated == 1


def test_arith_int64_mul_scalar_saturates(tmp_path):
    got, _ = _arith_main(
        tmp_path, [2**62, -(2**62), 3], torch.int64, "--op", "mul", "--value", "2"
    )
    assert got.flatten().tolist() == [_I64.max, _I64.min, 6]


def test_arith_int64_mul_image_saturates(tmp_path):
    got, saturated = _arith_main(
        tmp_path,
        [2**62, 3],
        torch.int64,
        "--op",
        "mul",
        operand=([2, 3], torch.int64),
    )
    assert got.flatten().tolist() == [_I64.max, 9]
    assert saturated == 1


def test_arith_int64_exact_large_scalar_literal(tmp_path):
    # An integer literal is kept exact (not rounded through a float).
    got, _ = _arith_main(
        tmp_path, [1, -5], torch.int64, "--op", "add", "--value", str(2**62 + 1)
    )
    assert got.flatten().tolist() == [2**62 + 2, 2**62 - 4]


def test_arith_int64_huge_float_scalar_saturates_to_max(tmp_path):
    got, saturated = _arith_main(
        tmp_path, [_I64.max, 0], torch.int64, "--op", "add", "--value", "1e20"
    )
    assert got.flatten().tolist() == [_I64.max, _I64.max]
    assert saturated == 2


def test_arith_uint64_add_scalar(tmp_path):
    got, saturated = _arith_main(
        tmp_path, [1, _U64_MAX], torch.uint64, "--op", "add", "--value", "1"
    )
    assert got.dtype == torch.uint64
    assert got.flatten().tolist() == [2, _U64_MAX]
    assert saturated == 1


def test_arith_uint64_sub_below_zero_saturates(tmp_path):
    got, saturated = _arith_main(
        tmp_path, [1, _U64_MAX], torch.uint64, "--op", "sub", "--value", "2"
    )
    assert got.flatten().tolist() == [0, _U64_MAX - 2]
    assert saturated == 1


def test_arith_uint64_mul_is_exact_beyond_2_53(tmp_path):
    got, saturated = _arith_main(
        tmp_path, [2**53 + 1, 2**63 + 1], torch.uint64, "--op", "mul", "--value", "1"
    )
    assert got.flatten().tolist() == [2**53 + 1, 2**63 + 1]
    assert saturated == 0


def test_arith_uint64_image_operand(tmp_path):
    got, saturated = _arith_main(
        tmp_path,
        [2**63 + 1, 2],
        torch.uint64,
        "--op",
        "mul",
        operand=([1, 2**63], torch.uint64),
    )
    assert got.flatten().tolist() == [2**63 + 1, _U64_MAX]
    assert saturated == 1


def test_arith_int64_plus_uint64_image(tmp_path):
    got, saturated = _arith_main(
        tmp_path,
        [5, -5],
        torch.int64,
        "--op",
        "add",
        operand=([_U64_MAX, 3], torch.uint64),
    )
    assert got.dtype == torch.int64
    assert got.flatten().tolist() == [_I64.max, -2]
    assert saturated == 1


def test_arith_int16_plus_int64_image_saturates_not_wraps(tmp_path):
    got, saturated = _arith_main(
        tmp_path,
        [1, -1],
        torch.int16,
        "--op",
        "add",
        operand=([_I64.max, _I64.min], torch.int64),
    )
    assert got.dtype == torch.int16
    assert got.flatten().tolist() == [32767, -32768]
    assert saturated == 2
