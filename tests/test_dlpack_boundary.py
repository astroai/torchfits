"""The native -> torch/numpy boundary (DLPack capsules and buffer views).

Every decoded buffer leaves C++ without a copy: torch results are DLPack
capsules adopted by ``torch.utils.dlpack.from_dlpack``, numpy results are
nanobind arrays owning the native storage, and raw table transport exposes a
read-only memoryview over it. These tests pin that contract directly:
storage identity, lifetime after the native result and capsule are gone, the
unconsumed-capsule destructor, concurrent first use (which is what imports
torch), empty results, and the unsigned / complex / bool dtypes.

Only the public ``torchfits`` API and ``torchfits._C`` are used, and every
input file is built byte-by-byte here, so the file runs against an installed
wheel as well as a source checkout.
"""

from __future__ import annotations

import gc
import os
import struct
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

import torchfits
import torchfits._C as cpp

# ---------------------------------------------------------------------------
# Byte-level FITS builders (no writer under test is involved)
# ---------------------------------------------------------------------------


def _card(key: str, value: str) -> str:
    return f"{key:<8}= {value:>20}".ljust(80)


def _block(cards: list[str]) -> bytes:
    text = "".join(cards) + "END".ljust(80)
    return (text + " " * ((-len(text)) % 2880)).encode()


def _pad(data: bytes) -> bytes:
    return data + b"\0" * ((-len(data)) % 2880)


PRIMARY = _block(
    [
        _card("SIMPLE", "T"),
        _card("BITPIX", "8"),
        _card("NAXIS", "0"),
        _card("EXTEND", "T"),
    ]
)


def _image(bitpix: int, shape: tuple[int, ...], raw: bytes, extra=()) -> bytes:
    cards = [
        _card("XTENSION", "'IMAGE   '"),
        _card("BITPIX", str(bitpix)),
        _card("NAXIS", str(len(shape))),
    ]
    for i, n in enumerate(reversed(shape), 1):
        cards.append(_card(f"NAXIS{i}", str(n)))
    cards += [_card("PCOUNT", "0"), _card("GCOUNT", "1")]
    cards += [_card(k, v) for k, v in extra]
    return _block(cards) + _pad(raw)


def _bintable(columns, nrows: int, data: bytes, heap: bytes = b"") -> bytes:
    """columns: (name, tform, width_bytes, extra cards with {i} placeholders)."""
    width = sum(c[2] for c in columns)
    cards = [
        _card("XTENSION", "'BINTABLE'"),
        _card("BITPIX", "8"),
        _card("NAXIS", "2"),
        _card("NAXIS1", str(width)),
        _card("NAXIS2", str(nrows)),
        _card("PCOUNT", str(len(heap))),
        _card("GCOUNT", "1"),
        _card("TFIELDS", str(len(columns))),
    ]
    for i, (name, tform, _width, extra) in enumerate(columns, 1):
        cards += [_card(f"TTYPE{i}", f"'{name}'"), _card(f"TFORM{i}", f"'{tform}'")]
        cards += [_card(k.format(i=i), v) for k, v in extra]
    return _block(cards) + _pad(data + heap)


U16 = np.array([[0, 1], [65535, 32768]], dtype=np.uint16)
U32 = np.array([0, 1, 2**32 - 1, 2**31], dtype=np.uint32)
U64 = np.array([0, 1, 2**64 - 1, 2**63], dtype=np.uint64)
U8_IMG = np.arange(250, 256, dtype=np.uint8).reshape(2, 3)
F32_IMG = (np.arange(64 * 48, dtype=np.float32) * 0.25).reshape(64, 48)

# Image HDUs in the image file, with their expected arrays.
IMAGES = {
    1: U8_IMG,
    2: U16,
    3: U32,
    4: U64,
    5: F32_IMG,
    6: np.zeros((0, 3), dtype=np.int32),
}

COL_U8 = np.array([0, 1, 254, 255], dtype=np.uint8)
COL_U16 = np.array([0, 1, 65535, 32768], dtype=np.uint16)
COL_U32 = U32
COL_U64 = U64
COL_FLAG = np.array([True, False, True, False])
COL_C = np.array([1 + 2j, -3.5 + 0j, -1j, 7.25 + 8j], dtype=np.complex64)
COL_M = np.array([1e300 + 1j, -2 - 2j, 3j, 0], dtype=np.complex128)
FIXED_COLUMNS = {
    "U8": COL_U8,
    "U16": COL_U16,
    "U32": COL_U32,
    "U64": COL_U64,
    "FLAG": COL_FLAG,
    "C": COL_C,
    "M": COL_M,
}
TABLE_LAYOUT = [
    ("U8", "B", 1, []),
    ("U16", "I", 2, [("TZERO{i}", "32768")]),
    ("U32", "J", 4, [("TZERO{i}", "2147483648")]),
    ("U64", "K", 8, [("TZERO{i}", "9223372036854775808")]),
    ("FLAG", "L", 1, []),
    ("C", "C", 8, []),
    ("M", "M", 16, []),
]
VLA_ROWS = [[1, 2, 3], [], [4], [5, 6]]


def _fixed_rows() -> bytes:
    stored_u16 = (COL_U16.astype(np.int64) - 32768).astype(">i2")
    stored_u32 = (COL_U32.astype(np.int64) - 2**31).astype(">i4")
    stored_u64 = (COL_U64 ^ np.uint64(2**63)).view(np.int64).astype(">i8")
    out = []
    for i in range(len(COL_U8)):
        out.append(
            struct.pack(">B", int(COL_U8[i]))
            + stored_u16[i : i + 1].tobytes()
            + stored_u32[i : i + 1].tobytes()
            + stored_u64[i : i + 1].tobytes()
            + (b"T" if COL_FLAG[i] else b"F")
            + struct.pack(">ff", COL_C[i].real, COL_C[i].imag)
            + struct.pack(">dd", COL_M[i].real, COL_M[i].imag)
        )
    return b"".join(out)


def _vla_table() -> bytes:
    heap = b""
    descriptors = b""
    for row in VLA_ROWS:
        # (element count, byte offset into the heap)
        descriptors += struct.pack(">ii", len(row), len(heap))
        heap += b"".join(struct.pack(">i", v) for v in row)
    return _bintable([("V", "PJ(3)", 8, [])], len(VLA_ROWS), descriptors, heap)


def write_image_file(path: str) -> None:
    with open(path, "wb") as fh:
        fh.write(PRIMARY)
        fh.write(_image(8, U8_IMG.shape, U8_IMG.tobytes()))
        fh.write(
            _image(
                16,
                U16.shape,
                (U16.astype(np.int64) - 32768).astype(">i2").tobytes(),
                [("BZERO", "32768"), ("BSCALE", "1")],
            )
        )
        fh.write(
            _image(
                32,
                U32.shape,
                (U32.astype(np.int64) - 2**31).astype(">i4").tobytes(),
                [("BZERO", "2147483648"), ("BSCALE", "1")],
            )
        )
        fh.write(
            _image(
                64,
                U64.shape,
                (U64 ^ np.uint64(2**63)).view(np.int64).astype(">i8").tobytes(),
                [("BZERO", "9223372036854775808"), ("BSCALE", "1")],
            )
        )
        fh.write(_image(-32, F32_IMG.shape, F32_IMG.astype(">f4").tobytes()))
        fh.write(_image(32, (0, 3), b""))


def write_table_file(path: str) -> None:
    """HDU 1: fixed columns; HDU 2: same columns, zero rows; HDU 3: VLA."""
    with open(path, "wb") as fh:
        fh.write(PRIMARY)
        fh.write(_bintable(TABLE_LAYOUT, len(COL_U8), _fixed_rows()))
        fh.write(_bintable(TABLE_LAYOUT, 0, b""))
        fh.write(_vla_table())


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("dlpack")
    images = str(root / "images.fits")
    tables = str(root / "tables.fits")
    write_image_file(images)
    write_table_file(tables)
    return {"images": images, "tables": tables}


TORCH_DTYPE = {
    np.dtype("uint8"): torch.uint8,
    np.dtype("uint16"): torch.uint16,
    np.dtype("uint32"): torch.uint32,
    np.dtype("uint64"): torch.uint64,
    np.dtype("int32"): torch.int32,
    np.dtype("float32"): torch.float32,
    np.dtype("bool"): torch.bool,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
}


def _torch_to_numpy(t: torch.Tensor) -> np.ndarray:
    # torch's .numpy() rejects uint16/32/64 on some releases; go through bytes.
    t = t.contiguous()
    return np.frombuffer(
        bytes(t.view(torch.uint8).numpy()) if t.numel() else b"",
        dtype={v: k for k, v in TORCH_DTYPE.items()}[t.dtype],
    ).reshape(tuple(t.shape))


def _assert_tensor(t, expected: np.ndarray) -> None:
    assert isinstance(t, torch.Tensor), type(t)
    assert t.dtype == TORCH_DTYPE[expected.dtype], (t.dtype, expected.dtype)
    assert tuple(t.shape) == expected.shape
    np.testing.assert_array_equal(_torch_to_numpy(t), expected)


# ---------------------------------------------------------------------------
# dtypes and empty results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hdu", sorted(IMAGES))
def test_image_dtypes_cross_the_boundary(files, hdu) -> None:
    expected = IMAGES[hdu]
    _assert_tensor(cpp.read_full(files["images"], hdu, False), expected)
    _assert_tensor(cpp.read_full_cached(files["images"], hdu, True), expected)
    _assert_tensor(torchfits.read(files["images"], hdu=hdu), expected)


@pytest.mark.parametrize("hdu", [1, 2], ids=["rows", "empty"])
def test_table_dtypes_cross_the_boundary(files, hdu) -> None:
    n = len(COL_U8) if hdu == 1 else 0
    expect = {k: v[:n] for k, v in FIXED_COLUMNS.items()}

    as_torch = cpp.read_fits_table(files["tables"], hdu, [], False)
    assert list(as_torch) == list(expect)
    for name, values in expect.items():
        _assert_tensor(as_torch[name], values)

    as_numpy = cpp.read_fits_table_rows_numpy(files["tables"], hdu, [], 1, -1, False)
    for name, values in expect.items():
        arr = as_numpy[name]
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == values.dtype
        np.testing.assert_array_equal(arr, values)

    raw = cpp.read_fits_table_rows_raw(files["tables"], hdu, [], 1, -1, False)
    for name, values in expect.items():
        col = raw[name]
        assert col["kind"] == "fixed"
        assert np.dtype(col["dtype"]) == values.dtype
        assert list(col["shape"]) == [n]
        view = col["data"]
        assert isinstance(view, memoryview) and view.readonly
        assert view.nbytes == values.nbytes
        np.testing.assert_array_equal(np.frombuffer(view, dtype=values.dtype), values)


def test_vla_rows_including_empty_row(files) -> None:
    rows = cpp.read_fits_table(files["tables"], 3, [], False)["V"]
    assert [r.tolist() for r in rows] == VLA_ROWS
    assert all(r.dtype == torch.int32 for r in rows)
    assert rows[1].numel() == 0

    values, offsets = cpp.read_fits_table_rows_numpy(
        files["tables"], 3, [], 1, -1, False
    )["V"]
    assert offsets.tolist() == [0, 3, 3, 4, 6]
    assert values.tolist() == [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Zero copy / storage identity
# ---------------------------------------------------------------------------


def test_torch_result_adopts_the_native_buffer(files) -> None:
    t = cpp.read_full(files["images"], 5, False)
    storage = t.untyped_storage()
    # Adopted as-is: the storage is exactly the decoded buffer, not a larger
    # torch allocation the data was copied into.
    assert storage.data_ptr() == t.data_ptr()
    assert storage.nbytes() == t.numel() * t.element_size()
    assert t.is_contiguous()
    # Writable, and writes land in that same buffer.
    t[0, 0] = -1.0
    assert t.data_ptr() == storage.data_ptr()
    assert float(t[0, 0]) == -1.0


def test_vla_rows_are_views_of_one_buffer(files) -> None:
    # Each row is exported as its own capsule, but every one points into the
    # single decoded values buffer: no per-row copy.
    rows = cpp.read_fits_table(files["tables"], 3, [], False)["V"]
    base = rows[0].data_ptr()
    offsets = [0, 3, 3, 4]
    for row, offset in zip(rows, offsets):
        if row.numel():
            assert row.data_ptr() == base + 4 * offset


def test_numpy_result_does_not_own_a_copy(files) -> None:
    result = cpp.read_fits_table_rows_numpy(files["tables"], 1, [], 1, -1, False)
    for arr in result.values():
        assert not arr.flags.owndata
        assert arr.base is not None


def test_raw_view_is_stable_and_read_only(files) -> None:
    raw = cpp.read_fits_table_rows_raw(files["tables"], 1, ["M"], 1, -1, False)
    view = raw["M"]["data"]
    a = np.frombuffer(view, dtype=np.complex128)
    b = np.frombuffer(view, dtype=np.complex128)
    assert a.ctypes.data == b.ctypes.data
    with pytest.raises((TypeError, BufferError, ValueError)):
        a[0] = 0


# ---------------------------------------------------------------------------
# Lifetime
# ---------------------------------------------------------------------------


def _churn(files) -> None:
    """Free and reallocate native buffers so a dangling pointer gets reused."""
    for _ in range(5):
        cpp.read_full(files["images"], 5, False)
        cpp.read_fits_table(files["tables"], 1, [], False)
        bytearray(1 << 16)
    gc.collect()


def test_results_outlive_their_containers(files) -> None:
    table = cpp.read_fits_table(files["tables"], 1, [], False)
    kept_m = table["M"]
    rows = cpp.read_fits_table(files["tables"], 3, [], False)["V"]
    kept_row = rows[3]
    numpy_cols = cpp.read_fits_table_rows_numpy(files["tables"], 1, [], 1, -1, False)
    kept_u64 = numpy_cols["U64"]
    values, _offsets = cpp.read_fits_table_rows_numpy(
        files["tables"], 3, [], 1, -1, False
    )["V"]
    raw = cpp.read_fits_table_rows_raw(files["tables"], 1, ["U32"], 1, -1, False)
    kept_view = raw["U32"]["data"]
    image = cpp.read_full(files["images"], 5, False)
    image_slice = image[10:20, 5]

    del table, rows, numpy_cols, _offsets, raw, image
    _churn(files)

    _assert_tensor(kept_m, COL_M)
    assert kept_row.tolist() == [5, 6]
    np.testing.assert_array_equal(kept_u64, COL_U64)
    assert values.tolist() == [1, 2, 3, 4, 5, 6]
    np.testing.assert_array_equal(np.frombuffer(kept_view, dtype=np.uint32), COL_U32)
    np.testing.assert_array_equal(image_slice.numpy(), F32_IMG[10:20, 5])


def test_tensor_freed_on_another_thread(files) -> None:
    import threading

    tensors = [cpp.read_full(files["images"], 5, False) for _ in range(8)]
    keep = tensors[0][:2, :2].clone()

    def drop(t) -> None:
        del t

    threads = [threading.Thread(target=drop, args=(tensors.pop(),)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    gc.collect()
    np.testing.assert_array_equal(keep.numpy(), F32_IMG[:2, :2])


# ---------------------------------------------------------------------------
# Subprocess checks: first use, unconsumed capsules
# ---------------------------------------------------------------------------


def _run(script: str, *args: str, timeout: int = 120) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=dict(os.environ),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


FIRST_USE = """
import sys, threading
import torchfits._C as cpp
assert "torch" not in sys.modules, "importing torchfits._C must not import torch"

images, tables = sys.argv[1], sys.argv[2]
n = 16
barrier = threading.Barrier(n)
results = [None] * n
errors = []

def work(i):
    try:
        barrier.wait()
        kind = i % 4
        if kind == 0:
            results[i] = ("image", cpp.read_full(images, 3, False))
        elif kind == 1:
            results[i] = ("table", cpp.read_fits_table(tables, 1, ["U64", "C"], False))
        elif kind == 2:
            results[i] = ("vla", cpp.read_fits_table(tables, 3, [], False))
        else:
            results[i] = ("numpy", cpp.read_fits_table_rows_numpy(tables, 1, [], 1, -1, False))
    except BaseException as exc:
        errors.append(repr(exc))

threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=90)
assert not any(t.is_alive() for t in threads), "deadlock on first use"
assert not errors, errors

import torch
for kind, r in results:
    if kind == "image":
        assert isinstance(r, torch.Tensor) and r.dtype == torch.uint32
        assert r.tolist() == [0, 1, 2**32 - 1, 2**31]
    elif kind == "table":
        assert r["U64"].dtype == torch.uint64 and r["C"].dtype == torch.complex64
        assert r["U64"].tolist() == [0, 1, 2**64 - 1, 2**63]
    elif kind == "vla":
        assert [x.tolist() for x in r["V"]] == [[1, 2, 3], [], [4], [5, 6]]
    else:
        assert str(r["U16"].dtype) == "uint16" and r["U16"].tolist() == [0, 1, 65535, 32768]
print("OK")
"""


def test_concurrent_first_use_imports_torch_once(files) -> None:
    out = _run(FIRST_USE, files["images"], files["tables"])
    assert "OK" in out


CAPSULES = """
import gc, os, sys
import torchfits._C as cpp
import torch.utils.dlpack as tud

images, big = sys.argv[1], sys.argv[2]
real = tud.from_dlpack
held = []
mode = ["hold"]

def seam(capsule):
    held.append(capsule)
    if mode[0] == "raise":
        raise RuntimeError("consumer refused")
    return "sentinel"

tud.from_dlpack = seam
first = cpp.read_full(images, 3, False)
if not held:
    print("SEAM_UNUSED")
    sys.exit(0)
assert first == "sentinel", first
cap = held.pop()
assert type(cap).__name__ == "PyCapsule" and '"dltensor"' in repr(cap), repr(cap)

# A capsule consumed after the native call returned (and gc ran) still owns
# valid storage.
gc.collect()
t = real(cap)
assert t.tolist() == [0, 1, 2**32 - 1, 2**31], t.tolist()
assert '"used_dltensor"' in repr(cap)
del cap  # renamed by the consumer: its destructor must not free the data
gc.collect()
assert t.tolist() == [0, 1, 2**32 - 1, 2**31]
del t


def rss_mb():
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        return None

# Unconsumed capsules: dropping them must free the native buffer (16 MiB each).
for _ in range(3):
    cpp.read_full(big, 1, False)
    held.clear()
base = rss_mb()
for _ in range(60):
    cpp.read_full(big, 1, False)
    held.clear()
gc.collect()
after = rss_mb()
if base is not None:
    assert after - base < 256, f"unconsumed capsules leaked: {base:.0f} -> {after:.0f} MiB"

# A consumer that raises: the error propagates and the capsule is still freed.
mode[0] = "raise"
for _ in range(60):
    try:
        cpp.read_full(big, 1, False)
    except RuntimeError as exc:
        assert "consumer refused" in str(exc)
    else:
        raise AssertionError("consumer error was swallowed")
    held.clear()
gc.collect()
final = rss_mb()
if base is not None:
    assert final - base < 256, f"refused capsules leaked: {base:.0f} -> {final:.0f} MiB"

print("OK")
"""


def test_unconsumed_capsule_destructor(files, tmp_path) -> None:
    big = str(tmp_path / "big.fits")
    side = 2048
    with open(big, "wb") as fh:
        fh.write(PRIMARY)
        fh.write(_image(-32, (side, side), bytes(4 * side * side)))
    out = _run(CAPSULES, files["images"], big)
    if "SEAM_UNUSED" in out:
        pytest.skip(
            "torchfits no longer resolves torch.utils.dlpack.from_dlpack lazily"
        )
    assert "OK" in out


def test_adopted_buffers_are_released(tmp_path) -> None:
    """Consumed capsules hand their deleter to torch; dropping tensors frees."""
    if not os.path.exists("/proc/self/statm"):
        pytest.skip("RSS probe needs /proc")
    big = str(tmp_path / "big.fits")
    side = 2048
    with open(big, "wb") as fh:
        fh.write(PRIMARY)
        fh.write(_image(-32, (side, side), bytes(4 * side * side)))

    def rss_mb() -> float:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20

    for _ in range(3):
        cpp.read_full(big, 1, False)
    gc.collect()
    base = rss_mb()
    for _ in range(60):
        t = cpp.read_full(big, 1, False)
        del t
    gc.collect()
    assert rss_mb() - base < 256
