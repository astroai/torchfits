"""overwrite=False must never clobber or delete a file that appears mid-write.

write() checked for an existing file once, then let CFITSIO create the target;
a failed write removed whatever was at the path (another writer's file), and
the compressed branch always clobbered ('!').
"""

import pytest
import torch

import torchfits
import torchfits._C as cpp
import torchfits.table as table

OTHER = b"OTHER PROCESS DATA"


def _racing(monkeypatch, target, fail=False):
    """Another process creates ``target`` after write()'s existence check,
    just before the native writer runs (which then fails if ``fail``)."""
    for attr in (
        "write_fits_file",
        "write_fits_file_compressed_images",
        "write_fits_table",
    ):
        original = getattr(cpp, attr)

        def racing(path, *args, _original=original, **kwargs):
            with open(target, "wb") as fh:
                fh.write(OTHER)
            if fail:
                raise RuntimeError("native writer failed")
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(cpp, attr, racing)


@pytest.mark.parametrize(
    "data, kwargs",
    [
        (torch.zeros(4, 4), {}),
        (torch.zeros(4, 4), {"compress": True}),
        ({"a": torch.arange(3)}, {}),
    ],
)
def test_write_does_not_clobber_file_created_mid_write(
    tmp_path, monkeypatch, data, kwargs
):
    path = str(tmp_path / "race.fits")
    _racing(monkeypatch, path)
    with pytest.raises(FileExistsError):
        torchfits.write(path, data, overwrite=False, **kwargs)
    assert open(path, "rb").read() == OTHER
    assert [p.name for p in tmp_path.iterdir()] == ["race.fits"]


def test_failed_write_does_not_delete_file_created_mid_write(tmp_path, monkeypatch):
    path = str(tmp_path / "race.fits")
    _racing(monkeypatch, path, fail=True)
    with pytest.raises(RuntimeError):
        torchfits.write(path, torch.zeros(2, 2), overwrite=False)
    assert open(path, "rb").read() == OTHER


def test_table_write_does_not_clobber_file_created_mid_write(tmp_path, monkeypatch):
    path = str(tmp_path / "race.fits")
    _racing(monkeypatch, path)
    with pytest.raises(FileExistsError):
        table.write(path, {"a": torch.arange(3)}, schema={"a": {"format": "K"}})
    assert open(path, "rb").read() == OTHER


def test_new_file_written_and_failed_write_leaves_nothing(tmp_path):
    path = str(tmp_path / "new.fits")
    torchfits.write(path, torch.ones(2, 2))
    assert torchfits.read(path).tolist() == [[1.0, 1.0], [1.0, 1.0]]
    bad = str(tmp_path / "bad.fits")
    with pytest.raises(Exception):
        torchfits.write(bad, {"bad": object()})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["new.fits"]
