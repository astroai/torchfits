"""
Example: FITS tables as dataframes.

Primary path: ``table.read`` → Arrow (portable dataframe).
Also: ``table.read_torch`` (tensor columns), ``table.read_polars`` (native DF).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from astropy.table import Table

import torchfits


def _create_test_file(path: str) -> None:
    from astropy.io import fits

    table = Table(
        {
            "ra": np.array([200.0, 201.0, 202.0], dtype=np.float64),
            "dec": np.array([45.0, 46.0, 47.0], dtype=np.float64),
            "flux": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "id": np.array([1, 2, 3], dtype=np.int32),
            "flag": np.array([True, False, True], dtype=bool),
        }
    )
    fits.BinTableHDU(table, name="MY_TABLE").writeto(path, overwrite=True)


def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = fh.name

    try:
        _create_test_file(path)

        # --- Primary: dataframe via Arrow (synonym: table.read_arrow) ---
        arrow_df = torchfits.table.read(
            path,
            hdu=1,
            columns=["ra", "dec", "flux"],
            where="flux >= 2.0",
        )
        print(f"table.read dataframe (where flux >= 2): {arrow_df.num_rows} rows")
        print(f"  flux values: {arrow_df.column('flux').to_pylist()}")
        assert torchfits.table.read_arrow is torchfits.table.read

        # --- Dataframe columns as tensors (root alias: read_table) ---
        tensors = torchfits.table.read_torch(path, hdu=1)
        print("read_torch columns:", list(tensors.keys()))
        print(f"  ra: {tensors['ra'].tolist()}")

        subset = torchfits.read_table_rows(
            path, hdu=1, start_row=1, num_rows=2, columns=["id", "flag"]
        )
        print(
            f"read_table_rows id={subset['id'].tolist()}, flag={subset['flag'].tolist()}"
        )

        chunks = list(
            torchfits.table.scan_torch(path, hdu=1, batch_size=2, columns=["id"])
        )
        print(
            f"scan_torch: {len(chunks)} chunk(s), "
            f"ids={[c['id'].tolist() for c in chunks]}"
        )

        # --- Native Polars dataframe (optional dep) ---
        try:
            pl_df = torchfits.table.read_polars(path, hdu=1)
            print(f"read_polars dataframe: {pl_df.shape[0]} rows, cols={pl_df.columns}")
        except ImportError:
            print("read_polars skipped (polars not installed)")

        # --- In-place mutations ---
        torchfits.table.append_rows(
            path,
            {
                "ra": np.array([203.0], dtype=np.float64),
                "dec": np.array([48.0], dtype=np.float64),
                "flux": np.array([4.0], dtype=np.float32),
                "id": np.array([4], dtype=np.int32),
                "flag": np.array([False], dtype=bool),
            },
            hdu=1,
        )
        print("\nAppended 1 row to FITS table.")

        torchfits.table.update_rows(
            path,
            {"flux": np.array([9.9, 9.9], dtype=np.float32)},
            row_slice=slice(1, 3),
            hdu=1,
        )
        print("Updated flux for rows index 1 to 3.")

        torchfits.table.insert_column(
            path,
            "quality",
            np.array([100, 100, 100, 100], dtype=np.int16),
            hdu=1,
            format="I",
        )
        print("Inserted new column 'quality'.")

        torchfits.table.rename_columns(path, {"ra": "right_ascension"}, hdu=1)
        print("Renamed column 'ra' to 'right_ascension'.")

        torchfits.table.drop_columns(path, ["flag"], hdu=1)
        print("Dropped column 'flag'.")

        modified = torchfits.table.read_torch(path, hdu=1)
        print("Modified dataframe columns:", list(modified.keys()))
        print(f"  right_ascension: {modified['right_ascension'].tolist()}")
        print(f"  flux (updated): {modified['flux'].tolist()}")
        print(f"  quality (inserted): {modified['quality'].tolist()}")

        out_path = path.replace(".fits", "_out.fits")
        new_data = {
            "ra": torch.tensor([300.0, 301.0], dtype=torch.float64),
            "dec": torch.tensor([50.0, 51.0], dtype=torch.float64),
        }
        torchfits.table.write(
            out_path,
            new_data,
            header={"EXTNAME": "FILTERED"},
            overwrite=True,
        )
        written = torchfits.table.read_torch(out_path, hdu=1)
        print(f"table.write round-trip: {written['ra'].tolist()}")
        os.unlink(out_path)
    finally:
        os.unlink(path)


if __name__ == "__main__":
    main()
