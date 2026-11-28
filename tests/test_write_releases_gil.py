"""The native image/HDU-list writers run their CFITSIO IO without the GIL.

``write_fits_file`` (``torchfits.write`` of images and HDU lists,
``insert_hdu`` / ``replace_hdu``), ``write_fits_file_compressed_images``
(``write(compress=...)``) and ``FITSFile.write_image`` used to hold the GIL
for the whole create/write/close, so every other Python thread stalled while
CFITSIO wrote (or tile-compressed) the pixels. The Python payload is now
converted with the GIL held and the file IO runs with it released, like the
table writers (tests/test_table_mutation_releases_gil.py).
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

import torchfits
import torchfits._C as cpp

SIDE = 4096


def _longest_stall(call: Callable[[], None]) -> tuple[float, float]:
    """Run ``call`` while another thread ticks; return (longest gap, duration).

    A native call that holds the GIL stops the ticker for its whole length, so
    the longest gap between ticks is about the call's duration. A call that
    releases it leaves only scheduler-sized gaps. (Counting ticks inside the
    call's window instead is unreliable on a loaded host: the main thread can
    wait milliseconds to get the GIL back after the call, and the ticker's
    ticks during that wait land inside the measured window.)
    """
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
    inside = [start] + [x for x in ticks if start < x < end] + [end]
    gap = max(b - a for a, b in zip(inside, inside[1:]))
    return gap, end - start


def _assert_other_thread_ran(
    tmp_path: Path, write: Callable[[str], None], what: str
) -> str:
    """Run ``write(path)`` on fresh paths; return the last path written."""
    # Best of three: a loaded host can stall the ticker on its own.
    results = []
    for i in range(3):
        path = str(tmp_path / f"{what.replace(' ', '_')}_{i}.fits")
        results.append(_longest_stall(lambda: write(path)))
    assert any(gap < duration / 2 for gap, duration in results), (
        f"another thread stalled during every {what}: "
        + ", ".join(f"{g * 1e3:.0f} of {d * 1e3:.0f} ms" for g, d in results)
    )
    return path


def _image() -> np.ndarray:
    return np.arange(SIDE * SIDE, dtype=np.float32).reshape(SIDE, SIDE)


class _TableProxy:
    """The shape of the payload ``_hdu_rewrite`` hands the native writer."""

    __slots__ = ("_raw_data", "header", "_schema")

    def __init__(self, raw_data: dict[str, Any]) -> None:
        self._raw_data = raw_data
        self.header: dict[str, Any] = {}
        self._schema = None


def test_write_fits_file_image_lets_other_threads_run(tmp_path) -> None:
    image = _image()
    path = _assert_other_thread_ran(
        tmp_path,
        lambda p: cpp.write_fits_file(p, [{"data": image, "header": {}}], False),
        "image write",
    )
    np.testing.assert_array_equal(torchfits.read(path).numpy(), image)


def test_write_fits_file_strided_image_lets_other_threads_run(tmp_path) -> None:
    """A non-contiguous view is packed outside the GIL too."""
    image = _image().T
    path = _assert_other_thread_ran(
        tmp_path,
        lambda p: cpp.write_fits_file(p, [{"data": image, "header": {}}], False),
        "strided image write",
    )
    np.testing.assert_array_equal(torchfits.read(path).numpy(), image)


def test_write_fits_file_table_hdu_lets_other_threads_run(tmp_path) -> None:
    """insert_hdu / replace_hdu / write(HDUList) tables go through write_hdus."""
    n = 4_000_000
    cols = {
        "a": np.arange(n, dtype=np.int64),
        "b": np.arange(n, dtype=np.float64),
    }
    payload: list[Any] = [{"header": {}}, _TableProxy(cols)]
    path = _assert_other_thread_ran(
        tmp_path, lambda p: cpp.write_fits_file(p, payload, False), "table HDU write"
    )
    out = torchfits.table.read(path, hdu=1)
    np.testing.assert_array_equal(np.asarray(out["a"]), cols["a"])
    np.testing.assert_array_equal(np.asarray(out["b"]), cols["b"])


def test_compressed_image_write_lets_other_threads_run(tmp_path) -> None:
    image = (np.arange(SIDE * SIDE, dtype=np.int32) % 1000).reshape(SIDE, SIDE)
    path = _assert_other_thread_ran(
        tmp_path,
        lambda p: cpp.write_fits_file_compressed_images(
            p, [{"data": image, "header": {}}], False, "GZIP_1"
        ),
        "compressed image write",
    )
    np.testing.assert_array_equal(torchfits.read(path, hdu=1).numpy(), image)


def test_fitsfile_write_image_lets_other_threads_run(tmp_path) -> None:
    image = _image()

    def write(path: str) -> None:
        handle = cpp.FITSFile(path, 1)
        handle.write_image(image)
        handle.close()

    path = _assert_other_thread_ran(tmp_path, write, "FITSFile.write_image")
    np.testing.assert_array_equal(torchfits.read(path).numpy(), image)


def test_shared_write_handle_is_serialized_across_threads(tmp_path) -> None:
    """Two threads writing through one FITSFile take turns, never deadlock."""
    path = str(tmp_path / "shared.fits")
    handle = cpp.FITSFile(path, 1)
    images = [np.full((512, 512), i, dtype=np.int16) for i in range(8)]
    errors: list[BaseException] = []

    def worker(chunk: list[np.ndarray]) -> None:
        try:
            for image in chunk:
                handle.write_image(image)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(images[i::2],)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive(), "shared write handle deadlocked"
    handle.close()
    assert not errors
    values = sorted(int(torchfits.read(path, hdu=i).numpy()[0, 0]) for i in range(8))
    assert values == list(range(8))


@pytest.mark.parametrize("compress", [False, "GZIP_1"])
def test_concurrent_image_writes_are_correct(tmp_path, compress) -> None:
    images = [
        (np.arange(1024 * 1024, dtype=np.int32) * (i + 1)).reshape(1024, 1024)
        for i in range(4)
    ]
    paths = [str(tmp_path / f"c{i}.fits") for i in range(4)]
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            for _ in range(3):
                torchfits.write(paths[i], images[i], overwrite=True, compress=compress)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    hdu = 1 if compress else 0
    for path, image in zip(paths, images):
        np.testing.assert_array_equal(torchfits.read(path, hdu=hdu).numpy(), image)
