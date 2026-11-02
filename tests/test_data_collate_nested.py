"""``fits_collate_fn`` collates nested per-arm dicts (multi-arm spectra)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from torchfits import Header, HDUList, TensorHDU
from torchfits.data import FitsSpectrumDataset, fits_collate_fn, make_loader


def _spectra(tmp_path: Path) -> list[str]:
    paths = []
    for i in range(4):
        path = tmp_path / f"spec{i}.fits"
        HDUList(
            [
                TensorHDU(torch.empty(0), Header({"CLS": i % 2})),
                TensorHDU(torch.arange(10.0) + i, Header({"EXTNAME": "B"})),
                TensorHDU(torch.arange(12.0) + i, Header({"EXTNAME": "R"})),
            ]
        ).write(str(path), overwrite=True)
        paths.append(str(path))
    return paths


def test_multi_arm_spectra_collate(tmp_path: Path) -> None:
    ds = FitsSpectrumDataset(_spectra(tmp_path), hdu=["B", "R"])
    batch = next(iter(make_loader(ds, batch_size=2, shuffle=False)))
    assert set(batch) == {"B", "R"}
    assert batch["B"]["flux"].shape == (2, 10)
    assert batch["R"]["flux"].shape == (2, 12)
    assert torch.equal(batch["R"]["flux"][1], torch.arange(12.0) + 1)


def test_multi_arm_spectra_collate_with_labels(tmp_path: Path) -> None:
    ds = FitsSpectrumDataset(_spectra(tmp_path), hdu=["B", "R"], label_key="CLS")
    arms, labels = next(iter(make_loader(ds, batch_size=4, shuffle=False)))
    assert arms["B"]["flux"].shape == (4, 10)
    assert labels.tolist() == [0, 1, 0, 1]


def test_nested_non_tensor_leaf_names_full_path() -> None:
    batch = [{"B": {"name": "x"}}, {"B": {"name": "y"}}]
    with pytest.raises(ValueError, match=r"'B\.name'"):
        fits_collate_fn(batch)
