"""HCOMPRESS_1 never writes past its tile buffer (cfitsio-4.7.0-hcompress-bounds).

A small tile of high-entropy 32-bit integers encodes to more bytes than
CFITSIO's ``nx * 4.4 + 26`` tile buffer. The bit-plane writer stored past the
end of it and corrupted the heap before the encoder reported status 413, so
the process could abort later in an unrelated ``free()``. glibc's malloc
checker turns that heap corruption into an immediate abort, so the writes run
in a child process with it enabled where it is available.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import torchfits

_MALLOC_DEBUG = Path("/usr/lib/x86_64-linux-gnu/libc_malloc_debug.so.0")

_CHILD = """
import sys
import numpy as np
import torchfits

rng = np.random.default_rng(0)
path = sys.argv[1]
for _ in range(20):
    x = rng.integers(-2**31, 2**31 - 1, size=(8, 8), dtype=np.int32, endpoint=True)
    try:
        torchfits.write(path, x, compress="HCOMPRESS_1", tile_shape=(4, 4), overwrite=True)
    except RuntimeError:
        continue
    assert np.array_equal(torchfits.read(path, hdu=1).numpy(), x)
print("ok")
"""


def test_high_entropy_int32_tiles_fail_cleanly(tmp_path: Path) -> None:
    path = str(tmp_path / "h.fits")
    x = np.random.default_rng(1).integers(
        -(2**31), 2**31 - 1, size=(8, 8), dtype=np.int32, endpoint=True
    )
    try:
        torchfits.write(
            path, x, compress="HCOMPRESS_1", tile_shape=(4, 4), overwrite=True
        )
    except RuntimeError as exc:
        assert "compress" in str(exc).lower()
    else:
        assert np.array_equal(torchfits.read(path, hdu=1).numpy(), x)


@pytest.mark.skipif(
    not sys.platform.startswith("linux")
    or not _MALLOC_DEBUG.exists()
    or "asan" in os.environ.get("LD_PRELOAD", ""),
    reason="needs glibc's libc_malloc_debug (and no ASan, which checks this itself)",
)
def test_high_entropy_int32_tiles_do_not_corrupt_the_heap(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["LD_PRELOAD"] = f"{_MALLOC_DEBUG} {env.get('LD_PRELOAD', '')}".strip()
    env["MALLOC_CHECK_"] = "3"
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, str(tmp_path / "h.fits")],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("ok")
