"""The native table writers run their CFITSIO IO without holding the GIL.

``write_fits_table`` and the append/insert/update/drop/rename mutation
bindings used to hold the GIL for the whole call, so every other Python
thread stalled while CFITSIO wrote (or, for an insert near the start or a
column drop, shifted) the table. The Python payload is now converted to
native buffers with the GIL held and the file IO runs with it released, like
``delete_fits_table_rows`` already did.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Callable

import numpy as np
import pytest

import torchfits
import torchfits._C as cpp

N_ROWS = 2_000_000


def _ticks_during(call: Callable[[], None]) -> tuple[int, float]:
    ticks: list[float] = []
    stop = threading.Event()

    def ticker() -> None:
        while not stop.is_set():
            ticks.append(time.perf_counter())

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-4)
    t = threading.Thread(target=ticker)
    t.start()
    try:
        time.sleep(0.01)
        start = time.perf_counter()
        call()
        end = time.perf_counter()
    finally:
        stop.set()
        t.join()
        sys.setswitchinterval(old_interval)
    duration = end - start
    # Ignore the edges, where the ticker may run before the native call takes
    # (or after it gives back) the GIL even when the call holds it throughout.
    margin = min(0.002, duration / 4)
    return sum(1 for x in ticks if start + margin < x < end - margin), duration


@pytest.fixture
def table_path(tmp_path):
    path = str(tmp_path / "big.fits")
    torchfits.write(
        path,
        {
            "a": np.arange(N_ROWS, dtype=np.int64),
            "b": np.arange(N_ROWS, dtype=np.float64),
        },
        overwrite=True,
    )
    return path


def _payload() -> dict[str, np.ndarray]:
    return {
        "a": np.arange(N_ROWS, dtype=np.int64)[::-1].copy(),
        "b": np.zeros(N_ROWS, dtype=np.float64),
    }


def test_write_fits_table_lets_other_threads_run(tmp_path) -> None:
    path = str(tmp_path / "w.fits")
    payload = _payload()
    during, duration = _ticks_during(
        lambda: cpp.write_fits_table(path, payload, {}, False)
    )
    assert during, f"no other thread ran during a {duration * 1e3:.1f} ms write"
    assert torchfits.read_nrows(path, 1) == N_ROWS


@pytest.mark.parametrize("op", ["append", "insert", "update"])
def test_row_writers_let_other_threads_run(table_path, op) -> None:
    payload = _payload()
    calls = {
        "append": lambda: cpp.append_fits_table_rows(table_path, 1, payload),
        "insert": lambda: cpp.insert_fits_table_rows(table_path, 1, payload, 1),
        "update": lambda: cpp.update_fits_table_rows(table_path, 1, payload, 1, N_ROWS),
    }
    during, duration = _ticks_during(calls[op])
    assert during, f"no other thread ran during a {duration * 1e3:.1f} ms {op}"
    expected = N_ROWS if op == "update" else 2 * N_ROWS
    assert torchfits.read_nrows(table_path, 1) == expected
    head = cpp.read_fits_table_rows(table_path, 1, ["a"], 1, 2, False)["a"].tolist()
    if op == "append":
        assert head == [0, 1]
    else:
        assert head == [N_ROWS - 1, N_ROWS - 2]


def test_drop_columns_lets_other_threads_run(table_path) -> None:
    during, duration = _ticks_during(
        lambda: cpp.drop_fits_table_columns(table_path, 1, ["a"])
    )
    assert during, f"no other thread ran during a {duration * 1e3:.1f} ms drop"
    assert torchfits.read_header(table_path, 1)["TTYPE1"] == "b"


def test_rename_and_rejected_payloads_still_work(table_path) -> None:
    cpp.rename_fits_table_columns(table_path, 1, {"a": "A2", "b": "a"})
    hdr = torchfits.read_header(table_path, 1)
    assert (hdr["TTYPE1"], hdr["TTYPE2"]) == ("A2", "a")
    with pytest.raises(RuntimeError, match="Column not found for rename_columns"):
        cpp.rename_fits_table_columns(table_path, 1, {"nope": "x"})

    # A payload rejected while it is converted never touches the file.
    bad = {"A2": np.zeros(2, dtype=np.int64), "a": np.zeros((2, 3))}
    for call in (
        lambda: cpp.append_fits_table_rows(table_path, 1, bad),
        lambda: cpp.insert_fits_table_rows(table_path, 1, bad, 1),
        lambda: cpp.update_fits_table_rows(table_path, 1, bad, 1, 2),
    ):
        with pytest.raises(RuntimeError, match="repeat mismatch"):
            call()
    assert torchfits.read_nrows(table_path, 1) == N_ROWS
    head = cpp.read_fits_table_rows(table_path, 1, ["A2"], 1, 2, False)["A2"]
    assert head.tolist() == [0, 1]
