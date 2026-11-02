"""A handle opened before its file was replaced must keep reading the file it opened.

The shared per-path metadata (raw fd, shape, scale) follows the path. After an
out-of-band replace, a fresh reader refreshed it in place, and an old handle
then read the new file's bytes at the old data offset, or picked up the new
file's BZERO.

Runs in a subprocess so the revalidation interval can be shortened.
"""

import os
import subprocess
import sys
import textwrap


def _run(code: str, tmp_path) -> str:
    env = dict(os.environ, TORCHFITS_SHARED_META_VALIDATE_INTERVAL_MS="0")
    res = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    return res.stdout.strip()


def test_old_handle_does_not_read_replacement_bytes(tmp_path):
    out = _run(
        """
        import os, sys, torch, torchfits._C as C
        d = sys.argv[1]
        p, newp = os.path.join(d, "a.fits"), os.path.join(d, "b.fits")
        old = torch.arange(64, dtype=torch.int16).reshape(8, 8)
        C.write_fits_file(p, [{"data": old}], True)
        header = {f"K{i:03d}": i for i in range(60)}  # moves the data offset
        C.write_fits_file(newp, [{"data": old + 1000, "header": header}], True)
        h = C.open_fits_file(p, "r")
        os.replace(newp, p)
        assert C.read_full_nocache(p, 0, True).flatten()[0].item() == 1000
        for use_mmap in (True, False):
            assert torch.equal(h.read_tensor(0, use_mmap), old), use_mmap
        print("ok")
        """,
        tmp_path,
    )
    assert out == "ok"


def test_old_handle_does_not_take_replacement_scale(tmp_path):
    out = _run(
        """
        import os, sys, torch, torchfits._C as C
        d = sys.argv[1]
        p, newp = os.path.join(d, "a.fits"), os.path.join(d, "b.fits")
        old = torch.arange(64, dtype=torch.int16).reshape(8, 8)
        C.write_fits_file(p, [{"data": old}], True)
        C.write_fits_file(newp, [{"data": old, "header": {"BZERO": 32768, "BSCALE": 1}}], True)
        h = C.open_fits_file(p, "r")
        os.replace(newp, p)
        C.read_full_nocache(p, 0, False)
        t = h.read_tensor(0, False)
        assert t.dtype == torch.int16, t.dtype
        assert torch.equal(t, old)
        print("ok")
        """,
        tmp_path,
    )
    assert out == "ok"
