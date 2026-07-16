"""
Example: FITS table I/O — tensor reads, Arrow reads, predicate pushdown, and write.
"""

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

        # --- Tensor path: read_table returns dict[str, Tensor] ---
        tensors = torchfits.read_table(path, hdu=1)
        print("read_table columns:", list(tensors.keys()))
        print(f"  ra: {tensors['ra'].tolist()}")

        # Column projection and row slice on the tensor path
        subset = torchfits.read_table_rows(
            path, hdu=1, start_row=1, num_rows=2, columns=["id", "flag"]
        )
        print(
            f"read_table_rows id={subset['id'].tolist()}, flag={subset['flag'].tolist()}"
        )

        # --- Arrow path: table.read returns pyarrow.Table ---
        arrow_table = torchfits.table.read(
            path,
            hdu=1,
            columns=["ra", "dec", "flux"],
            where="flux >= 2.0",
        )
        print(f"table.read (where flux >= 2): {arrow_table.num_rows} rows")
        print(f"  flux values: {arrow_table.column('flux').to_pylist()}")

        # Stream large tables in fixed-size chunks
        chunks = list(torchfits.stream_table(path, hdu=1, chunk_rows=2, columns=["id"]))
        print(
            f"stream_table: {len(chunks)} chunk(s), ids={[c['id'].tolist() for c in chunks]}"
        )

        # --- In-place table mutations via torchfits.table ---
        # Append new rows to the existing FITS file in-place
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

        # Update specific rows (e.g., set flux to 9.9 for row indices 1 and 2)
        torchfits.table.update_rows(
            path,
            {"flux": np.array([9.9, 9.9], dtype=np.float32)},
            row_slice=slice(1, 3),
            hdu=1,
        )
        print("Updated flux for rows index 1 to 3.")

        # Insert a new column (quality)
        torchfits.table.insert_column(
            path,
            "quality",
            np.array([100, 100, 100, 100], dtype=np.int16),
            hdu=1,
            format="I",
        )
        print("Inserted new column 'quality'.")

        # Rename a column (ra -> right_ascension)
        torchfits.table.rename_columns(path, {"ra": "right_ascension"}, hdu=1)
        print("Renamed column 'ra' to 'right_ascension'.")

        # Drop a column (flag)
        torchfits.table.drop_columns(path, ["flag"], hdu=1)
        print("Dropped column 'flag'.")

        # Read modified table back to verify in-place changes
        modified = torchfits.read_table(path, hdu=1)
        print("Modified table columns:", list(modified.keys()))
        print(f"  right_ascension: {modified['right_ascension'].tolist()}")
        print(f"  flux (updated): {modified['flux'].tolist()}")
        print(f"  quality (inserted): {modified['quality'].tolist()}")

        # --- Write back with table.write ---
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
        written = torchfits.read_table(out_path, hdu=1)
        print(f"table.write round-trip: {written['ra'].tolist()}")
        os.unlink(out_path)
    finally:
        os.unlink(path)


if __name__ == "__main__":
    main()
