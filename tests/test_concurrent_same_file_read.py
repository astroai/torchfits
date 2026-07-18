"""Concurrent reads of the same file across threads (CFITSIO §4 Option A).

Regression guard for sharing one cached ``fitsfile*`` across threads. Each read
path now opens a private per-call handle, so many threads hitting the same file
and alternating HDUs must return byte-identical data with no crash / corruption.
"""

import tempfile
import threading

import numpy as np
import torch

import torchfits


def _write_mef():
    """Write a small MEF: primary image + one named image extension."""
    from astropy.io import fits

    rng = np.random.default_rng(1234)
    d0 = rng.normal(0, 1, (48, 64)).astype(np.float32)
    d1 = rng.normal(0, 1, (32, 96)).astype(np.float32)
    f = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    f.close()
    hdus = [fits.PrimaryHDU(d0), fits.ImageHDU(d1, name="SCI")]
    fits.HDUList(hdus).writeto(f.name, overwrite=True)
    return f.name, d0, d1


def test_concurrent_same_file_read():
    path, d0, d1 = _write_mef()

    # Single-threaded references (also warms SharedReadMeta).
    ref0 = torchfits.read(path, hdu=0)
    ref1 = torchfits.read(path, hdu=1)
    assert np.allclose(ref0.numpy(), d0)
    assert np.allclose(ref1.numpy(), d1)

    n_threads = 8
    iters = 25
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker(tid: int) -> None:
        try:
            for i in range(iters):
                hdu = (tid + i) % 2
                # Alternate between the unified read and the tensor fast path.
                if i % 2 == 0:
                    t = torchfits.read(path, hdu=hdu)
                else:
                    t = torchfits.read_tensor(path, hdu=hdu)
                expected = ref0 if hdu == 0 else ref1
                if not torch.equal(t.cpu(), expected):
                    raise AssertionError(
                        f"thread {tid} iter {i} hdu {hdu}: data mismatch"
                    )
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent read failures: {errors[:3]}"


def test_concurrent_same_file_table_read():
    """Concurrent table reads of one file across threads must not crash/corrupt."""
    n = 500
    f = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    f.close()
    torchfits.write(
        f.name,
        {
            "ID": np.arange(n, dtype=np.int64),
            "VAL": np.arange(n, dtype=np.float32),
        },
        overwrite=True,
    )
    ref = torchfits.table.read(f.name, hdu=1)
    assert ref.num_rows == n

    errors: list[Exception] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            for _ in range(20):
                t = torchfits.table.read(f.name, hdu=1)
                if t.num_rows != n:
                    raise AssertionError("row count mismatch")
                if t.column("ID").to_pylist()[:3] != [0, 1, 2]:
                    raise AssertionError("data corruption")
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent table read failures: {errors[:3]}"


if __name__ == "__main__":
    test_concurrent_same_file_read()
    test_concurrent_same_file_table_read()
    print("ok")
