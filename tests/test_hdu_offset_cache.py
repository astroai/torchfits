"""Learned HDU offsets must never send a read to the wrong HDU.

A read-only handle publishes the CFITSIO header offsets it learned when it
closes, and the next open of the same file is seeded with them so HDU n is one
seek away. A stale offset silently reads a *different* HDU, so the table is
tied to the exact file version it was read from (inode, size, mtime, ctime)
and only recorded once that version is a few seconds old (see fits_core.cpp).

Each scenario rewrites file A into file B of the same size with a different
layout: primary + three int16 2x3 extensions (pixel [0, 0] = i), where A pads
E3's header by two blocks and B pads E1's, so A's E3 starts exactly where B's
E2 does. Offsets learned from A and applied to B read E2 as "HDU 3".

The files are written once per module and left to age past the recording
threshold, so every test exercises a handle that does publish.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import pytest

import torchfits

BLOCK = 2880
# TORCHFITS_HDU_OFFSET_CACHE_MIN_AGE_MS default: a file version younger than
# this is never recorded.
MIN_AGE_S = int(os.environ.get("TORCHFITS_HDU_OFFSET_CACHE_MIN_AGE_MS", "3000")) / 1000

A = [0, 0, 2]  # extension sizes 2, 2, 4 blocks: E3 starts at block 5
B = [2, 0, 0]  # extension sizes 4, 2, 2 blocks: E2 starts at block 5


def _card(key: str, value: object) -> str:
    if isinstance(value, bool):
        text = f"{'T' if value else 'F':>20}"
    elif isinstance(value, int):
        text = f"{value:>20}"
    else:
        text = f"{repr(value).replace(chr(34), chr(39)):<20}"
    return f"{key:<8}= {text}".ljust(80)


def _header(cards: list[tuple[str, object]], extra_blocks: int = 0) -> bytes:
    text = "".join(_card(k, v) for k, v in cards)
    text += "".join(f"COMMENT pad {i}".ljust(80) for i in range(36 * extra_blocks))
    text += "END".ljust(80)
    return (text + " " * ((-len(text)) % BLOCK)).encode("ascii")


def _ext(i: int, pad: int, kind: str) -> bytes:
    if kind == "image":
        cards = [
            ("XTENSION", "IMAGE"),
            ("BITPIX", 16),
            ("NAXIS", 2),
            ("NAXIS1", 3),
            ("NAXIS2", 2),
            ("PCOUNT", 0),
            ("GCOUNT", 1),
        ]
        data = np.full((2, 3), i, dtype=">i2").tobytes()
    else:
        cards = [
            ("XTENSION", "BINTABLE"),
            ("BITPIX", 8),
            ("NAXIS", 2),
            ("NAXIS1", 2),
            ("NAXIS2", 1),
            ("PCOUNT", 0),
            ("GCOUNT", 1),
            ("TFIELDS", 1),
            ("TTYPE1", "V"),
            ("TFORM1", "I"),
        ]
        data = np.array([i], dtype=">i2").tobytes()
    cards.append(("EXTNAME", f"E{i}"))
    return _header(cards, pad) + data + b"\0" * ((-len(data)) % BLOCK)


def _write(
    path: str, pads: list[int], kind: str = "image", mode: str = "inplace"
) -> None:
    target = path + ".tmp" if mode == "replace" else path
    with open(target, "wb") as fh:
        fh.write(
            _header([("SIMPLE", True), ("BITPIX", 16), ("NAXIS", 0), ("EXTEND", True)])
        )
        for i, pad in enumerate(pads, start=1):
            fh.write(_ext(i, pad, kind))
    if mode == "replace":
        os.replace(target, path)


def _write_mef(path: str, n_ext: int) -> None:
    _write(path, [0] * n_ext)


def _value(path: str, hdu: int) -> int:
    out = torchfits.read(path, hdu=hdu)
    if isinstance(out, dict):
        return int(out["V"][0])
    return int(out[0, 0])


@pytest.fixture(scope="module")
def aged(tmp_path_factory):
    """Every file the module needs, written up front and left to age once."""
    root = tmp_path_factory.mktemp("hdu_offsets")
    paths: dict[str, str] = {}

    def add(name: str, writer) -> None:
        paths[name] = str(root / f"{name}.fits")
        writer(paths[name])

    for mode in ("inplace", "replace"):
        add(f"handle_{mode}", lambda p: _write(p, A))
        add(f"window_{mode}", lambda p: _write(p, A))
        add(f"table_{mode}", lambda p: _write(p, A, "table"))
    add("writer_thread", lambda p: _write(p, A))
    add("positive", lambda p: _write_mef(p, 200))
    for name, n_ext in (("cap_small_1", 40), ("cap_small_2", 40), ("cap_big", 300)):
        add(name, lambda p, n=n_ext: _write_mef(p, n))
    time.sleep(MIN_AGE_S + 0.5)
    return paths


def _assert_b_layout(path: str) -> None:
    assert _value(path, 3) == 3
    assert _value(path, 2) == 2
    assert _value(path, 1) == 1
    assert torchfits.read_num_hdus(path) == 4


@pytest.mark.parametrize("mode", ["inplace", "replace"])
def test_handle_open_across_a_same_size_rewrite_publishes_nothing(aged, mode):
    path = aged[f"handle_{mode}"]
    torchfits.clear_all_caches()
    hdul = torchfits.open(path)
    try:
        assert int(hdul[3].data[0, 0]) == 3  # the handle learns A's offsets
        _write(path, B, mode=mode)
        time.sleep(1.1)  # past the shared-metadata validation window
    finally:
        hdul.close()  # used to publish A's offsets for B
    _assert_b_layout(path)


@pytest.mark.parametrize("mode", ["inplace", "replace"])
def test_path_read_right_after_a_same_size_rewrite(aged, mode):
    path = aged[f"window_{mode}"]
    torchfits.clear_all_caches()
    assert _value(path, 3) == 3  # publishes A's offsets
    _write(path, B, mode=mode)
    # Inside the shared-metadata validation window: the seed used to trust it.
    _assert_b_layout(path)


@pytest.mark.parametrize("mode", ["inplace", "replace"])
def test_stale_cached_table_reader_publishes_nothing(aged, mode):
    import torchfits._C as cpp

    path = aged[f"table_{mode}"]
    torchfits.clear_all_caches()

    def value(hdu: int) -> int:
        return int(cpp.read_fits_table_rows(path, hdu, [], 1, -1, False)["V"][0])

    assert value(3) == 3  # caches a table reader that knows A's offsets
    _write(path, B, kind="table", mode=mode)
    time.sleep(1.1)
    # This acquire finds the cached reader stale and destroys it; it used to
    # publish A's offsets on the way out, and the fresh reader opened right
    # after was seeded with them.
    assert value(3) == 3
    assert value(2) == 2


def test_writer_invalidation_while_another_thread_holds_a_reader(aged):
    import torch

    path = aged["writer_thread"]
    torchfits.clear_all_caches()
    learned = threading.Event()
    rewritten = threading.Event()
    seen: list[int] = []

    def reader() -> None:
        # A long-lived handle that stays on HDU 3 (so it knows A's offsets).
        with torchfits.open_subset_reader(path, hdu=3) as subset:
            seen.append(int(subset(0, 0, 1, 1)[0, 0]))
            learned.set()
            rewritten.wait(30)

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        assert learned.wait(30)

        # In-process writers: grow E1's header by two blocks, then rebuild E3
        # without its two padding blocks -- the B layout, same size as A.
        def ext(i: int) -> torch.Tensor:
            return torch.full((2, 3), i, dtype=torch.int16)

        many = {f"K{i:03d}": i for i in range(80)}
        torchfits.replace_hdu(path, 1, ext(1), header=many)
        torchfits.delete_hdu(path, 3)
        torchfits.insert_hdu(path, ext(3), index=3)
    finally:
        rewritten.set()
        thread.join(30)
    assert seen == [3]
    assert os.path.getsize(path) == 9 * BLOCK
    _assert_b_layout(path)


def _rchar() -> int:
    with open("/proc/self/io") as fh:
        for line in fh:
            if line.startswith("rchar:"):
                return int(line.split()[1])
    raise AssertionError("no rchar in /proc/self/io")


needs_proc_io = pytest.mark.skipif(
    not sys.platform.startswith("linux") or not os.path.exists("/proc/self/io"),
    reason="needs Linux /proc/self/io read accounting",
)


@needs_proc_io
def test_an_unchanged_file_still_skips_the_header_walk(aged):
    path = aged["positive"]
    torchfits.clear_all_caches()
    assert _value(path, 200) == 200  # walks, then records
    before = _rchar()
    assert _value(path, 200) == 200
    assert _rchar() - before < 16 * BLOCK


@needs_proc_io
def test_fresh_file_versions_are_not_recorded(tmp_path):
    # A version younger than the threshold could still be rewritten within the
    # filesystem's timestamp granularity without its ctime changing.
    path = str(tmp_path / "fresh.fits")
    _write_mef(path, 200)
    torchfits.clear_all_caches()
    assert _value(path, 200) == 200
    before = _rchar()
    assert _value(path, 200) == 200
    assert _rchar() - before > 150 * BLOCK


_CAP_SCRIPT = textwrap.dedent(
    """
    import sys
    import torchfits

    BLOCK = 2880

    def rchar():
        with open("/proc/self/io") as fh:
            return next(int(l.split()[1]) for l in fh if l.startswith("rchar:"))

    def walks(path, hdu):
        assert int(torchfits.read(path, hdu=hdu)[0, 0]) == hdu
        before = rchar()
        assert int(torchfits.read(path, hdu=hdu)[0, 0]) == hdu
        return rchar() - before > hdu * BLOCK // 2

    small_1, small_2, big = sys.argv[1:4]
    # Budget 600 bytes = 75 offsets: one 40-HDU table (42 offsets) fits, two
    # do not, and a 300-HDU table never does.
    assert not walks(small_1, 40)
    assert walks(small_2, 40)
    assert walks(big, 300)
    torchfits.clear_all_caches()  # releases small_1's table
    assert not walks(small_2, 40)
    print("ok")
    """
)


@needs_proc_io
def test_offset_tables_share_a_bounded_budget(aged):
    env = dict(os.environ, TORCHFITS_HDU_OFFSET_CACHE_MAX_BYTES="600")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _CAP_SCRIPT,
            aged["cap_small_1"],
            aged["cap_small_2"],
            aged["cap_big"],
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip().endswith("ok")
