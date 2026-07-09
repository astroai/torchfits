"""
Example: PyTorch Dataset pattern using torchfits.data classes with DataLoader.
"""

import os
import shutil
import tempfile

import numpy as np
import torch
from astropy.io import fits

import torchfits
from torchfits.data import FitsImageDataset, make_loader


def _create_dummy_fits(
    data_dir: str, num_files: int = 10, size: tuple[int, int] = (64, 64)
) -> None:
    os.makedirs(data_dir, exist_ok=True)
    for i in range(num_files):
        data = np.random.rand(*size).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        hdu.header["LABEL"] = i % 2
        hdu.writeto(os.path.join(data_dir, f"image_{i:03d}.fits"), overwrite=True)


def main() -> None:
    data_dir = tempfile.mkdtemp(prefix="torchfits_dataset_")
    try:
        _create_dummy_fits(data_dir, num_files=8)
        file_pattern = os.path.join(data_dir, "*.fits")

        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        for device in devices:
            print(f"\n--- device={device} ---")

            # Use the built-in FitsImageDataset with label-from-header
            dataset = FitsImageDataset(
                file_pattern,
                hdu=0,
                label_key="LABEL",
                device=device,
            )

            # Use make_loader with automatic cache warm-up
            loader = make_loader(
                dataset,
                batch_size=4,
                num_workers=0,
                pin_memory=False,
            )

            for i, (images, labels) in enumerate(loader):
                print(
                    f"  batch {i}: images={images.shape} on {images.device}, "
                    f"labels={labels.tolist()}"
                )
                if i >= 1:
                    break

        # Batch read multiple files at once (useful for inference pipelines)
        dataset = FitsImageDataset(file_pattern, hdu=0, device="cpu")
        batch = torchfits.read_batch(dataset.files[:4], hdu=0)
        print(f"\nread_batch: {len(batch)} images, first shape={batch[0].shape}")
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
