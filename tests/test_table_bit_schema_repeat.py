"""A schema 'rX' repeat that disagrees with a 2-D bit payload must be rejected.

The schema repeat overrode the payload width, and the strided copy of a
non-contiguous payload then wrote shape(0)*shape(1) items into a buffer sized
for rows*repeat: a heap overflow (SIGSEGV) through torchfits.table.write.
Runs in a subprocess so a regression fails instead of killing pytest.
"""

import subprocess
import sys
import textwrap


def test_mismatched_bit_repeat_raises(tmp_path):
    code = textwrap.dedent(
        f"""
        import torch, torchfits.table as T
        x = torch.ones(512, 512, dtype=torch.bool).T  # non-contiguous
        try:
            T.write({str(tmp_path / "b.fits")!r}, {{"B": x}},
                    schema={{"B": {{"format": "1X"}}}}, overwrite=True)
        except Exception as exc:
            assert "bits per row" in str(exc), exc
            print("raised")
        else:
            print("no error")
        """
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert res.returncode == 0, (res.returncode, res.stderr[-2000:])
    assert res.stdout.strip() == "raised"


def test_matching_bit_repeat_strided_roundtrip(tmp_path):
    import torch

    import torchfits

    path = str(tmp_path / "ok.fits")
    x = (torch.arange(6 * 10).reshape(10, 6) % 3 == 0).T  # (6, 10), non-contiguous
    import torchfits.table as T

    T.write(path, {"B": x}, schema={"B": {"format": "10X"}}, overwrite=True)
    got = torchfits.read(path, hdu=1)["B"]
    assert got.reshape(x.shape).tolist() == x.tolist()
