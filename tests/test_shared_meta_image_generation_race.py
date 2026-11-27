"""Image readers must not publish a replaced file's metadata.

``read_full_cached``, ``read_full_nocache`` and ``resolve_hdu_name_cached``
look up the shared per-path metadata, open the file, scan to the HDU and then
store what they found (image shape, compression flag, BSCALE/BZERO, EXTNAME
index). When the file was replaced while the scan ran, the validator cleared
those caches and rotated the generation; the in-flight reader then stored the
old file's answers under the new generation, and every later read of the new
file resolved the old shape (or HDU index). Stores now happen only while the
generation the reader started under is still current.

The scan is made slow with many empty HDUs so the replace reliably lands
inside it. Runs in a subprocess because the validation interval is read once
per process.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import os, sys, threading, time
    import torchfits._C as cpp
    from torchfits import _core

    def card(k, v):
        return f"{k:<8}= {v:>20}".ljust(80)

    def block(cards):
        s = "".join(cards) + "END".ljust(80)
        return (s + " " * ((-len(s)) % 2880)).encode()

    def image(n, extname):
        hdr = block([card("XTENSION", "'IMAGE   '"), card("BITPIX", "16"),
                     card("NAXIS", "2"), card("NAXIS1", str(n)),
                     card("NAXIS2", str(n)), card("PCOUNT", "0"),
                     card("GCOUNT", "1"), card("EXTNAME", f"'{extname}'")])
        data = b"\\0" * (2 * n * n)
        return hdr + data + b"\\0" * ((-len(data)) % 2880)

    def build(n, n_empty):
        out = [block([card("SIMPLE", "T"), card("BITPIX", "8"),
                      card("NAXIS", "0"), card("EXTEND", "T")])]
        empty = block([card("XTENSION", "'IMAGE   '"), card("BITPIX", "8"),
                       card("NAXIS", "0"), card("PCOUNT", "0"), card("GCOUNT", "1")])
        out.append(empty * n_empty)
        out.append(image(n, "SCI"))
        return b"".join(out)

    path = sys.argv[1]
    n_empty = 20000
    last = n_empty + 1
    old, new = build(8, n_empty), build(32, n_empty)
    # Same EXTNAME at a different index in the replacement.
    new_named = build(32, 0)

    failures = []

    def run(name, reader, replacement, check):
        for trial in range(3):
            with open(path, "wb") as fh:
                fh.write(old)
            with open(path + ".new", "wb") as fh:
                fh.write(replacement)
            _core.clear_shared_read_meta_cache()
            _core.read_num_hdus(path)
            t = threading.Thread(target=reader)
            t.start()
            time.sleep(0.005)
            os.replace(path + ".new", path)
            _core.read_hdu_type(path, 0)  # validation notices the replace
            t.join()
            problem = check()
            if problem:
                failures.append(f"STALE {name} trial={trial} {problem}")
                return

    def shape_is_new():
        shape = tuple(_core.read_shape(path, last)[1])
        if shape != (32, 32):
            return f"read_shape={shape}"
        got = tuple(cpp.read_full_cached(path, last, False).shape)
        if got != (32, 32):
            return f"read_full_cached={got}"
        return None

    run("read_full_cached", lambda: cpp.read_full_cached(path, last, False),
        new, shape_is_new)
    run("read_full_nocache", lambda: cpp.read_full_nocache(path, last, False),
        new, shape_is_new)

    def name_is_new():
        got = cpp.resolve_hdu_name_cached(path, "SCI")
        return None if got == 1 else f"resolve_hdu_name_cached={got}"

    run("resolve_hdu_name_cached",
        lambda: cpp.resolve_hdu_name_cached(path, "SCI"), new_named, name_is_new)
    if failures:
        print("\\n".join(failures))
        sys.exit(1)
    print("OK")
    """
)


def test_image_readers_do_not_pin_a_replaced_files_metadata(tmp_path) -> None:
    env = dict(os.environ, TORCHFITS_SHARED_META_VALIDATE_INTERVAL_MS="0")
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path / "race.fits")],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout
