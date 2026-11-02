"""A path-level probe must not publish an answer from a replaced file.

``read_nrows`` (and its siblings) open the file, scan to the HDU, then store
the answer in the shared per-path cache. When the file is replaced while the
scan is running, the validator clears that cache and rotates its generation;
storing afterwards pinned the old file's answer under the new generation, and
every later probe returned it even though the file on disk had changed.

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
    from torchfits import _core

    def card(k, v):
        return f"{k:<8}= {v:>20}".ljust(80)

    def block(cards):
        s = "".join(cards) + "END".ljust(80)
        return (s + " " * ((-len(s)) % 2880)).encode()

    def build(nrows, n_empty):
        out = [block([card("SIMPLE", "T"), card("BITPIX", "8"),
                      card("NAXIS", "0"), card("EXTEND", "T")])]
        empty = block([card("XTENSION", "'IMAGE   '"), card("BITPIX", "8"),
                       card("NAXIS", "0"), card("PCOUNT", "0"), card("GCOUNT", "1")])
        out.append(empty * n_empty)
        out.append(block([card("XTENSION", "'BINTABLE'"), card("BITPIX", "8"),
                          card("NAXIS", "2"), card("NAXIS1", "4"),
                          card("NAXIS2", str(nrows)), card("PCOUNT", "0"),
                          card("GCOUNT", "1"), card("TFIELDS", "1"),
                          card("TTYPE1", "'X'"), card("TFORM1", "'J'")]))
        data = b"\\0" * (4 * nrows)
        out.append(data + b"\\0" * ((-len(data)) % 2880))
        return b"".join(out)

    path = sys.argv[1]
    n_empty = 20000
    old, new = build(111, n_empty), build(222, n_empty)
    last = n_empty + 1
    for trial in range(3):
        with open(path, "wb") as fh:
            fh.write(old)
        with open(path + ".new", "wb") as fh:
            fh.write(new)
        _core.clear_shared_read_meta_cache()
        _core.read_num_hdus(path)
        t = threading.Thread(target=lambda: _core.read_nrows(path, last))
        t.start()
        time.sleep(0.005)
        os.replace(path + ".new", path)
        _core.read_hdu_type(path, 0)  # validation notices the replace
        t.join()
        got = _core.read_nrows(path, last)
        if got != 222:
            print(f"STALE trial={trial} got={got}")
            sys.exit(1)
    print("OK")
    """
)


def test_probe_does_not_pin_a_replaced_files_answer(tmp_path) -> None:
    env = dict(os.environ, TORCHFITS_SHARED_META_VALIDATE_INTERVAL_MS="0")
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path / "race.fits")],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout
