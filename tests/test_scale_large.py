"""Large-scale probes: >2 GiB / >4 GiB files, >2^31-row tables, 10k HDUs/files.

Marked ``performance``: skipped unless pytest runs with ``--run-performance``
(``pixi run test-scale``) and deselected in CI. Most files are sparse (zero
data plus marker bytes), so a 7 GB table costs kilobytes of disk; the
streaming-writer test really writes 2.1 GB. Expect a few minutes and up to
~2.5 GB RSS. The default suite keeps small proxies of each fix
(test_many_hdus_scale.py, test_table_scale.py, test_io_releases_gil.py).
"""

from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import pytest

import torchfits
import torchfits.table as T

pytestmark = [
    pytest.mark.performance,
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux /proc"),
]

BLOCK = 2880


def _card(key: str, value: object) -> str:
    if isinstance(value, bool):
        text = f"{'T' if value else 'F':>20}"
    elif isinstance(value, int):
        text = f"{value:>20}"
    else:
        text = f"{repr(value).replace(chr(34), chr(39)):<20}"
    return f"{key:<8}= {text}".ljust(80)


def _header(cards: list[tuple[str, object]]) -> bytes:
    text = "".join(_card(k, v) for k, v in cards) + "END".ljust(80)
    return (text + " " * ((-len(text)) % BLOCK)).encode("ascii")


def _image(bitpix: int, naxes: tuple[int, ...], primary: bool, extname=None):
    cards: list[tuple[str, object]] = (
        [("SIMPLE", True)] if primary else [("XTENSION", "IMAGE")]
    )
    cards += [("BITPIX", bitpix), ("NAXIS", len(naxes))]
    cards += [(f"NAXIS{i}", n) for i, n in enumerate(naxes, 1)]
    cards += [("EXTEND", True)] if primary else [("PCOUNT", 0), ("GCOUNT", 1)]
    if extname:
        cards.append(("EXTNAME", extname))
    nbytes = abs(bitpix) // 8 * int(np.prod(naxes)) if naxes else 0
    return _header(cards), nbytes


def _bintable(naxis1: int, naxis2: int, cols, extname=None):
    cards: list[tuple[str, object]] = [
        ("XTENSION", "BINTABLE"),
        ("BITPIX", 8),
        ("NAXIS", 2),
        ("NAXIS1", naxis1),
        ("NAXIS2", naxis2),
        ("PCOUNT", 0),
        ("GCOUNT", 1),
        ("TFIELDS", len(cols)),
    ]
    for i, (name, form) in enumerate(cols, 1):
        cards += [(f"TTYPE{i}", name), (f"TFORM{i}", form)]
    if extname:
        cards.append(("EXTNAME", extname))
    return _header(cards), naxis1 * naxis2


def _build_sparse(path, hdus) -> None:
    """hdus: [(header, nbytes, {data_offset: bytes})]; data left as holes."""
    with open(path, "wb") as fh:
        pos = 0
        for header, nbytes, pokes in hdus:
            fh.seek(pos)
            fh.write(header)
            data = pos + len(header)
            for offset, payload in (pokes or {}).items():
                fh.seek(data + offset)
                fh.write(payload)
            pos = data + nbytes + ((-nbytes) % BLOCK)
        fh.truncate(pos)


@pytest.fixture(autouse=True)
def _drop_big_files(tmp_path):
    """pytest keeps the last runs' tmp dirs; do not leave GBs behind."""
    yield
    shutil.rmtree(tmp_path, ignore_errors=True)


def _free_bytes(path) -> int:
    return shutil.disk_usage(path).free


# --- images ------------------------------------------------------------------


def test_image_hdus_beyond_4gib(tmp_path):
    """A 4.46 GB int16 primary followed by an extension at a >4 GiB offset."""
    nx, ny = 65536, 34000
    path = tmp_path / "img4g.fits"
    last = ((ny - 1) * nx + nx - 1) * 2
    _build_sparse(
        path,
        [
            (*_image(16, (nx, ny), True), {last: np.array([1234], ">i2").tobytes()}),
            (
                *_image(32, (10, 3), False, "AFTER"),
                {0: np.arange(30, dtype=">i4").tobytes()},
            ),
        ],
    )
    assert os.path.getsize(path) > 4 * 2**30
    p = str(path)
    for mmap in (True, False):
        ext = torchfits.read(p, hdu=1, mmap=mmap)
        assert ext.flatten().tolist() == list(range(30))
    assert torchfits.read(p, hdu="AFTER").shape == (3, 10)
    tail = torchfits.read_subset(p, 0, nx - 2, ny - 1, nx, ny)
    assert tail.tolist() == [[0, 1234]]
    for hdu in (0, 1):
        torchfits.write_checksums(p, hdu)
        assert torchfits.verify_checksums(p, hdu)["ok"] is True


def test_full_read_and_tile_compression_past_2_31_pixels(tmp_path):
    """2^31 + 65536 uint8 pixels: full read, RICE_1 compress, compressed read."""
    from torchfits.cli.main import main

    if _free_bytes(tmp_path) < 4 * 2**30:
        pytest.skip("needs ~4 GB free disk")
    nx, ny = 65536, 32769
    n = nx * ny
    path = tmp_path / "u8.fits"
    pokes = {0: b"\x05", 2**31: b"\x09", n - 1: b"\x07"}
    _build_sparse(path, [(*_image(8, (nx, ny), True), pokes)])

    def _check(image) -> None:
        flat = image.reshape(-1)
        assert (int(flat[0]), int(flat[2**31]), int(flat[-1])) == (5, 9, 7)

    for mmap in (True, False):
        image = torchfits.read(str(path), mmap=mmap)
        _check(image)
        del image
    packed = tmp_path / "u8c.fits"
    assert main(["compress", "--algorithm", "RICE_1", str(path), str(packed)]) == 0
    y, x = divmod(2**31, nx)
    assert torchfits.read_subset(str(packed), 1, x, y, x + 2, y + 1).tolist() == [
        [9, 0]
    ]
    _check(torchfits.read(str(packed), hdu=1))


# --- tables ------------------------------------------------------------------


def test_table_past_2_31_rows(tmp_path):
    n = 2**31 + 1000
    path = tmp_path / "t1b.fits"
    pokes = {0: b"\x01", 2**31 - 1: b"\x02", 2**31: b"\x03", n - 1: b"\x04"}
    _build_sparse(
        path,
        [
            (*_image(8, (), True), None),
            (*_bintable(1, n, [("X", "1B")], "BIG"), pokes),
        ],
    )
    p = str(path)
    assert torchfits.read_nrows(p, 1) == n
    assert T.read(p, 1, row_slice=(n - 3, n)).column("X").to_pylist() == [0, 0, 4]
    assert T.read(p, 1, row_slice=(2**31 - 2, 2**31 + 2)).column("X").to_pylist() == [
        0,
        2,
        3,
        0,
    ]
    rows = [0, 2**31 - 1, 2**31, n - 1]
    assert T.read(p, 1, rows=rows).column("X").to_pylist() == [1, 2, 3, 4]
    assert T.read_torch(p, 1, start_row=n - 2, num_rows=3)["X"].tolist() == [0, 0, 4]
    assert torchfits.read(p, hdu=1, start_row=2**31, num_rows=2)["X"].tolist() == [2, 3]
    with torchfits.open_table_reader(p, 1) as reader:
        assert reader.read_torch(start_row=n - 1, num_rows=2)["X"].tolist() == [0, 4]
    batches = [b.num_rows for b in T.scan(p, 1, row_slice=(n - 150_000, n))]
    assert sum(batches) == 150_000
    # where= over every row: the native scan's index used to reserve 17 GB.
    assert T.read(p, 1, where="X > 0").column("X").to_pylist() == [1, 2, 3, 4]
    found = [
        v
        for b in T.scan(p, 1, where="X > 0", batch_size=1 << 22)
        for v in b.column("X").to_pylist()
    ]
    assert found == [1, 2, 3, 4]


def test_table_rows_beyond_4gib_offsets(tmp_path):
    n = 600_000_000  # 12-byte rows: 7.2 GB
    path = tmp_path / "t4g.fits"

    def row(k: int, f: float) -> bytes:
        return np.array([k], ">i8").tobytes() + np.array([f], ">f4").tobytes()

    pokes = {0: row(11, 1.5), (n - 2) * 12: row(-5, 2.5), (n - 1) * 12: row(99, 9.5)}
    _build_sparse(
        path,
        [
            (*_image(8, (), True), None),
            (*_bintable(12, n, [("K", "1K"), ("F", "1E")]), pokes),
        ],
    )
    p = str(path)
    want = {"K": [-5, 99], "F": [2.5, 9.5]}
    for mmap in (True, False):
        assert T.read(p, 1, row_slice=(n - 2, n), mmap=mmap).to_pydict() == want
        got = T.read_torch(p, 1, start_row=n - 1, num_rows=2, mmap=mmap)
        assert {k: v.tolist() for k, v in got.items()} == want
        scanned = [b.to_pydict() for b in T.scan(p, 1, row_slice=(n - 2, n), mmap=mmap)]
        assert scanned == [want]
    assert T.read(p, 1, rows=[0, n - 1]).to_pydict() == {"K": [11, 99], "F": [1.5, 9.5]}
    assert T.read(p, 1, where="K != 0", row_slice=(n - 1000, n)).to_pydict() == want


def test_stream_writer_and_append_cross_2_31_rows(tmp_path):
    if _free_bytes(tmp_path) < 5 * 2**30:
        pytest.skip("needs ~5 GB free disk")
    n = 2**31 + 5
    path = str(tmp_path / "stream.fits")
    batch = 1 << 24
    with T.writer(path, overwrite=True) as writer:
        done = 0
        while done < n:
            k = min(batch, n - done)
            values = np.zeros(k, dtype=np.uint8)
            if done <= 2**31 < done + k:
                values[2**31 - done] = 3
            if done + k == n:
                values[-1] = 4
            writer.write({"X": values})
            done += k
        assert writer.num_rows == n
    assert torchfits.read_nrows(path, 1) == n
    assert T.read(path, 1, rows=[2**31, n - 1]).column("X").to_pylist() == [3, 4]
    T.append_rows(path, {"X": np.array([7, 8], dtype=np.uint8)}, hdu=1)
    assert torchfits.read_nrows(path, 1) == n + 2
    assert T.read(path, 1, row_slice=(n - 1, n + 2)).column("X").to_pylist() == [
        4,
        7,
        8,
    ]
    torchfits.write_checksums(path, 1)
    assert torchfits.verify_checksums(path, 1)["ok"] is True


# --- many HDUs / many files --------------------------------------------------


def _rchar() -> int:
    with open("/proc/self/io") as fh:
        return next(int(line.split()[1]) for line in fh if line.startswith("rchar:"))


def test_ten_thousand_hdu_file(tmp_path, capsys):
    import torch

    from torchfits.cli.main import main

    n_ext = 10_000
    path = tmp_path / "mef.fits"
    hdus = [(*_image(16, (), True), None)]
    for i in range(1, n_ext + 1):
        hdus.append(
            (*_image(16, (10, 10), False, f"E{i}"), {0: np.array([i], ">i2").tobytes()})
        )
    _build_sparse(path, hdus)
    p = str(path)
    torchfits.clear_all_caches()

    before = _rchar()
    with torchfits.open(p) as hdul:
        assert len(hdul) == n_ext + 1
        assert hdul[f"E{n_ext}"].header["EXTNAME"] == f"E{n_ext}"
    assert _rchar() - before < 8 * n_ext * BLOCK  # one pass, not n^2 / 2

    assert int(torchfits.read(p, hdu=n_ext)[0, 0]) == n_ext
    before = _rchar()
    for _ in range(100):
        assert int(torchfits.read(p, hdu=n_ext)[0, 0]) == n_ext
    assert _rchar() - before < 100 * 16 * BLOCK  # no header walk per read

    capsys.readouterr()
    assert main(["verify", p]) == 0
    assert len(capsys.readouterr().out.splitlines()) == n_ext + 1
    torchfits.insert_hdu(p, torch.ones(3, 3), index=1)
    torchfits.delete_hdu(p, n_ext + 1)
    assert torchfits.read_num_hdus(p) == n_ext + 1
    assert int(torchfits.read(p, hdu=n_ext)[0, 0]) == n_ext - 1


def test_ten_thousand_files_keep_descriptors_bounded(tmp_path):
    import torch

    n_files = 10_000
    paths = []
    for i in range(n_files):
        path = str(tmp_path / f"f{i:05d}.fits")
        torchfits.write(path, torch.full((8, 8), float(i)))
        paths.append(path)

    def fits_fds() -> int:
        count = 0
        for fd in os.listdir("/proc/self/fd"):
            try:
                count += os.readlink(f"/proc/self/fd/{fd}").startswith(str(tmp_path))
            except OSError:
                pass
        return count

    assert sum(float(torchfits.read(p)[0, 0]) for p in paths) == sum(range(n_files))
    assert len(torchfits.read_batch(paths)) == n_files
    assert fits_fds() <= 32  # TORCHFITS_MAX_CACHED_FDS default
    torchfits.clear_all_caches()
    assert fits_fds() == 0
