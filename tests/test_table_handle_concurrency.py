"""Table reads through one shared FITSFile handle are serialized.

CFITSIO keeps a single current-HDU cursor per ``fitsfile``. The ``*_from_handle``
table bindings borrowed the handle's ``fitsfile*`` and ran the table reader on
it with the GIL released but without the handle's I/O lock, so two threads
reading different HDUs through one handle moved the cursor under each other
and returned the other table's columns, wrong values, errors or crashed. The
bindings now hold the handle's lock for the whole read, and refuse a read once
the path names a different file than the one the handle opened.
"""

from __future__ import annotations

import os
import struct
import threading
import time

import numpy as np
import pytest

import torchfits._C as cpp


def _card(key: str, value: str) -> str:
    return f"{key:<8}= {value:>20}".ljust(80)


def _block(cards: list[str]) -> bytes:
    text = "".join(cards) + "END".ljust(80)
    return (text + " " * ((-len(text)) % 2880)).encode()


def _bintable(name: str, tform: str, fmt: str, values: list) -> bytes:
    width = struct.calcsize(">" + fmt)
    header = _block(
        [
            _card("XTENSION", "'BINTABLE'"),
            _card("BITPIX", "8"),
            _card("NAXIS", "2"),
            _card("NAXIS1", str(width)),
            _card("NAXIS2", str(len(values))),
            _card("PCOUNT", "0"),
            _card("GCOUNT", "1"),
            _card("TFIELDS", "1"),
            _card("TTYPE1", f"'{name}'"),
            _card("TFORM1", f"'{tform}'"),
        ]
    )
    data = b"".join(struct.pack(">" + fmt, v) for v in values)
    return header + data + b"\0" * ((-len(data)) % 2880)


N_A = 600
N_B = 350


@pytest.fixture()
def two_table_file(tmp_path) -> str:
    path = str(tmp_path / "two_tables.fits")
    primary = _block(
        [
            _card("SIMPLE", "T"),
            _card("BITPIX", "8"),
            _card("NAXIS", "0"),
            _card("EXTEND", "T"),
        ]
    )
    with open(path, "wb") as fh:
        fh.write(primary)
        fh.write(_bintable("A", "J", "i", list(range(N_A))))
        fh.write(_bintable("B", "D", "d", [0.5 * i for i in range(N_B)]))
    return path


def _check(result: dict, hdu: int) -> None:
    if hdu == 1:
        assert list(result) == ["A"], list(result)
        np.testing.assert_array_equal(
            np.asarray(result["A"]), np.arange(N_A, dtype=np.int32)
        )
    else:
        assert list(result) == ["B"], list(result)
        np.testing.assert_array_equal(np.asarray(result["B"]), 0.5 * np.arange(N_B))


def _raw_values(result: dict, hdu: int) -> dict:
    col = result["A" if hdu == 1 else "B"]
    dtype = np.dtype(col["dtype"])
    return {("A" if hdu == 1 else "B"): np.frombuffer(col["data"], dtype=dtype)}


READERS = {
    "full": lambda h, hdu: cpp.read_fits_table_from_handle(h, hdu),
    "rows": lambda h, hdu: cpp.read_fits_table_rows_from_handle(h, hdu, [], 1, -1),
    "numpy": lambda h, hdu: cpp.read_fits_table_rows_numpy_from_handle(
        h, hdu, [], 1, -1
    ),
    "raw": lambda h, hdu: _raw_values(
        cpp.read_fits_table_rows_raw_from_handle(h, hdu, [], 1, -1), hdu
    ),
}


@pytest.mark.parametrize("reader", sorted(READERS))
def test_shared_handle_reads_alternate_hdus_correctly(two_table_file, reader) -> None:
    read = READERS[reader]
    handle = cpp.open_fits_file(two_table_file, "r")
    try:
        # Sanity: sequential reads are right.
        _check(read(handle, 1), 1)
        _check(read(handle, 2), 2)

        n_threads = 8
        iterations = 40
        errors: list[BaseException] = []
        start = threading.Barrier(n_threads)

        def worker(idx: int) -> None:
            try:
                start.wait()
                for it in range(iterations):
                    hdu = 1 + (idx + it) % 2
                    _check(read(handle, hdu), hdu)
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"{len(errors)} failures, first: {errors[0]!r}"
    finally:
        handle.close()


def test_shared_handle_reads_interleave_with_handle_methods(two_table_file) -> None:
    """Handle methods that move the cursor must not corrupt a table read."""
    handle = cpp.open_fits_file(two_table_file, "r")
    try:
        stop = threading.Event()
        errors: list[BaseException] = []

        def mover() -> None:
            try:
                while not stop.is_set():
                    handle.get_num_hdus()
                    cpp.read_header(handle, 0)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def reader(idx: int) -> None:
            try:
                for it in range(40):
                    hdu = 1 + (idx + it) % 2
                    _check(
                        cpp.read_fits_table_rows_from_handle(handle, hdu, [], 1, -1),
                        hdu,
                    )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        bg = threading.Thread(target=mover)
        bg.start()
        readers = [threading.Thread(target=reader, args=(i,)) for i in range(4)]
        for t in readers:
            t.start()
        for t in readers:
            t.join()
        stop.set()
        bg.join()
        assert not errors, f"{len(errors)} failures, first: {errors[0]!r}"
    finally:
        handle.close()


def _write_two_tables(path: str, n_a: int, n_b: int, a_value=None) -> None:
    primary = _block(
        [
            _card("SIMPLE", "T"),
            _card("BITPIX", "8"),
            _card("NAXIS", "0"),
            _card("EXTEND", "T"),
        ]
    )
    a = list(range(n_a)) if a_value is None else [a_value] * n_a
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(primary)
        fh.write(_bintable("A", "J", "i", a))
        fh.write(_bintable("B", "D", "d", [0.5 * i for i in range(n_b)]))
    os.replace(tmp, path)


N_REPLACEMENTS = 120


def _replacement_sizes(i: int) -> tuple[int, int]:
    if i % 2:
        return 50 + i, 20 + i
    return 2 * N_A + i, 2 * N_B + i


def test_handle_reads_during_replacement_never_mix_files(two_table_file) -> None:
    """Old-handle readers racing out-of-band replacements of the path.

    Every read through the handle opened on the original file must return the
    original contents or fail cleanly; it must never return the replacement's
    bytes under the original's geometry (the table reader also reads the file
    by name, so before the fix it returned the replacement's values or
    garbage). Once the writer stops, path-based
    readers (and the shared metadata cache) must describe the final file.
    """
    handle = cpp.open_fits_file(two_table_file, "r")
    # Warm up (first tensor result imports torch) so the readers overlap the
    # writer instead of starting after it.
    _check(cpp.read_fits_table_rows_from_handle(handle, 1, [], 1, -1), 1)
    stop = threading.Event()
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            # Alternate replacements larger and smaller than the original, so
            # the old handle's geometry both fits and overruns the new file.
            for i in range(N_REPLACEMENTS):
                n_a, n_b = _replacement_sizes(i)
                _write_two_tables(two_table_file, n_a, n_b, a_value=-1)
                time.sleep(0.002)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            stop.set()

    def reader(idx: int) -> None:
        it = 0
        try:
            while not stop.is_set() or it < 4:
                hdu = 1 + (idx + it) % 2
                it += 1
                try:
                    result = cpp.read_fits_table_rows_from_handle(
                        handle, hdu, [], 1, -1
                    )
                except RuntimeError as exc:
                    # Refused: the path no longer names the handle's file.
                    assert "replaced" in str(exc) or "geometry" in str(exc), exc
                    continue
                _check(result, hdu)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    try:
        threads = [threading.Thread(target=reader, args=(i,)) for i in range(4)]
        threads.append(threading.Thread(target=writer))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        handle.close()
    assert not errors, f"{len(errors)} failures, first: {errors[0]!r}"

    # Quiescent: every reader of the path sees the last replacement.
    import torchfits

    final_a, final_b = _replacement_sizes(N_REPLACEMENTS - 1)
    assert torchfits.read_nrows(two_table_file, 1) == final_a
    assert torchfits.read_nrows(two_table_file, 2) == final_b
    got = cpp.read_fits_table_rows(two_table_file, 1, [], 1, -1, False)
    np.testing.assert_array_equal(
        np.asarray(got["A"]), np.full(final_a, -1, dtype=np.int32)
    )
    fresh = cpp.open_fits_file(two_table_file, "r")
    try:
        got = cpp.read_fits_table_rows_from_handle(fresh, 2, [], 1, -1)
        np.testing.assert_array_equal(np.asarray(got["B"]), 0.5 * np.arange(final_b))
    finally:
        fresh.close()
