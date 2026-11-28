"""Table reads whose memory must follow the result, not the table size.

Small-scale proxies for the >2 GiB / >2^31-row probes: the tables are sparse
files (zero data, a few marker rows), and an address-space limit stands in
for "the table is bigger than RAM".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

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


def write_sparse_byte_table(path, nrows: int, markers: dict[int, int]) -> None:
    """One ``1B`` column of ``nrows`` zero rows except ``markers`` {row: value}.

    The data is a hole in a sparse file, so a 10^8-row table costs no disk.
    """
    primary = _header([("SIMPLE", True), ("BITPIX", 8), ("NAXIS", 0), ("EXTEND", True)])
    table = _header(
        [
            ("XTENSION", "BINTABLE"),
            ("BITPIX", 8),
            ("NAXIS", 2),
            ("NAXIS1", 1),
            ("NAXIS2", nrows),
            ("PCOUNT", 0),
            ("GCOUNT", 1),
            ("TFIELDS", 1),
            ("TTYPE1", "X"),
            ("TFORM1", "1B"),
        ]
    )
    start = len(primary) + len(table)
    with open(path, "wb") as fh:
        fh.write(primary + table)
        for row, value in markers.items():
            fh.seek(start + row)
            fh.write(bytes([value]))
        fh.truncate(start + nrows + ((-nrows) % BLOCK))


_LIMITED_WHERE = textwrap.dedent(
    """
    import json, resource, sys
    import torchfits.table as T

    path, tiny, budget = sys.argv[1], sys.argv[2], int(sys.argv[3])
    T.read(tiny, 1, where="X > 0")  # load every module before the limit
    with open("/proc/self/status") as fh:
        vm = next(int(l.split()[1]) * 1024 for l in fh if l.startswith("VmSize"))
    resource.setrlimit(resource.RLIMIT_AS, (vm + budget, vm + budget))
    out = {}
    try:
        out["arrow"] = T.read(path, 1, where="X > 0").column("X").to_pylist()
    except MemoryError as exc:
        out["arrow"] = f"MemoryError: {exc}"
    print(json.dumps(out))
    """
)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="RLIMIT_AS / VmSize are Linux"
)
def test_native_where_memory_follows_matches_not_window(tmp_path):
    """where= on a large table must not reserve 8 bytes per scanned row.

    The native predicate scan reserved its matched-row index for the whole
    window, so a 2^31-row table asked for 17 GB before reading a row. Here a
    2*10^8-row table (1.6 GB of reservation) runs under a 768 MiB address-space
    budget; the four matches need a few bytes.
    """
    nrows = 200_000_000
    big = tmp_path / "big.fits"
    tiny = tmp_path / "tiny.fits"
    markers = {0: 1, 12_345_678: 2, nrows - 2: 3, nrows - 1: 4}
    write_sparse_byte_table(big, nrows, markers)
    write_sparse_byte_table(tiny, 10, {3: 1})
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
    proc = subprocess.run(
        [sys.executable, "-c", _LIMITED_WHERE, str(big), str(tiny), str(768 << 20)],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["arrow"] == [1, 2, 3, 4]
