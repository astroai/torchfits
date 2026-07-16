import sys
from pathlib import Path

# Add project root to sys.path to allow imports from the 'benchmarks' package
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from benchmarks.config import DEFAULT_OUTPUT_DIR  # noqa: E402

import argparse
import gzip
import os
import re
import shutil
import socket
import tempfile
import time
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import fitsio
import numpy as np
import torch
from astropy.io import fits as astropy_fits

import torchfits

from benchmarks.bench_contract import (
    RESULT_COLUMNS,
    annotate_rankings,
    write_csv,
    write_json,
)  # noqa: E402
from benchmarks.bench_timing import time_median, time_medians_interleaved  # noqa: E402

_BENCH_HOST = socket.gethostname()

# Scorecard timings must be cold-open / no handle cache. Peers (astropy, fitsio)
# reopen every iteration; leaving TorchFits cache on measured cache hits.
_TF_NO_CACHE = {"cache_capacity": 0, "handle_cache_capacity": 0}
_TF_READ_NO_CACHE = {**_TF_NO_CACHE, "use_cache": False}


def _repeats_for_rows(nrows: int) -> int:
    if nrows <= 10_000:
        return 9
    if nrows <= 100_000:
        return 5
    return 3


def _time_median(fn, *, runs: int, warmup: int) -> tuple[float | None, str | None]:
    median, _rss, _cuda, err = time_median(fn, runs=runs, warmup=warmup)
    return median, err


def _time_median_mem(
    fn, *, runs: int, warmup: int
) -> tuple[float | None, float | None, float | None, str | None]:
    return time_median(fn, runs=runs, warmup=warmup)


def _dtype_values(dtype: str, nrows: int, rng: np.random.Generator):
    if dtype == "f4":
        return rng.normal(size=nrows).astype(np.float32)
    if dtype == "f8":
        return rng.normal(size=nrows).astype(np.float64)
    if dtype == "i4":
        return rng.integers(-1_000_000, 1_000_000, size=nrows, dtype=np.int32)
    if dtype == "i8":
        return rng.integers(-1_000_000, 1_000_000, size=nrows, dtype=np.int64)
    if dtype == "bool":
        return rng.random(size=nrows) > 0.5
    if dtype.startswith("S"):
        width = int(dtype[1:]) if len(dtype) > 1 else 8
        return np.array([f"s{i:08d}"[:width] for i in range(nrows)], dtype=f"S{width}")
    raise ValueError(f"unsupported dtype spec: {dtype}")


def _write_table_file(
    *,
    out_path: Path,
    nrows: int,
    schema_name: str,
    schema: list[tuple[str, str]],
    rng_seed: int,
) -> tuple[str, list[str]]:
    rng = np.random.default_rng(rng_seed)
    cols = []
    for col_name, dtype in schema:
        arr = _dtype_values(dtype, nrows, rng)
        cols.append(
            astropy_fits.Column(name=col_name, format=_to_tform(dtype), array=arr)
        )
    hdu = astropy_fits.BinTableHDU.from_columns(
        cols, name=f"TABLE_{schema_name.upper()}"
    )
    astropy_fits.HDUList([astropy_fits.PrimaryHDU(), hdu]).writeto(
        out_path, overwrite=True
    )
    return schema_name, [c[0] for c in schema]


def _write_varlen_file(*, out_path: Path, nrows: int, rng_seed: int) -> list[str]:
    rng = np.random.default_rng(rng_seed)
    ids = np.arange(nrows, dtype=np.int32)
    flux = rng.normal(size=nrows).astype(np.float32)
    var = np.empty(nrows, dtype=object)
    for i in range(nrows):
        width = int((i % 7) + 1)
        var[i] = np.arange(width, dtype=np.int32)

    cols = [
        astropy_fits.Column(name="id", format="J", array=ids),
        astropy_fits.Column(name="flux", format="E", array=flux),
        astropy_fits.Column(name="values", format="PJ()", array=var),
    ]
    hdu = astropy_fits.BinTableHDU.from_columns(cols, name="TABLE_VARLEN")
    astropy_fits.HDUList([astropy_fits.PrimaryHDU(), hdu]).writeto(
        out_path, overwrite=True
    )
    return ["id", "flux", "values"]


def _write_typed_file(*, out_path: Path, nrows: int, rng_seed: int) -> list[str]:
    """Binary table with BIT, fixed string, and complex columns."""
    rng = np.random.default_rng(rng_seed)
    ids = np.arange(nrows, dtype=np.int32)
    flags = rng.integers(0, 256, size=nrows, dtype=np.uint8)
    names = np.array([f"obj_{i:06d}"[:12] for i in range(nrows)], dtype="S12")
    cvals = (
        rng.normal(size=nrows).astype(np.float32)
        + 1j * rng.normal(size=nrows).astype(np.float32)
    ).astype(np.complex64)

    cols = [
        astropy_fits.Column(name="id", format="J", array=ids),
        astropy_fits.Column(name="flags", format="8X", array=flags),
        astropy_fits.Column(name="name", format="A12", array=names),
        astropy_fits.Column(name="cval", format="C", array=cvals),
    ]
    hdu = astropy_fits.BinTableHDU.from_columns(cols, name="TABLE_TYPED")
    astropy_fits.HDUList([astropy_fits.PrimaryHDU(), hdu]).writeto(
        out_path, overwrite=True
    )
    return ["id", "flags", "name", "cval"]


def _write_ascii_file(*, out_path: Path, nrows: int, rng_seed: int) -> list[str]:
    """ASCII table extension (TableHDU) for read/projection benchmarks."""
    rng = np.random.default_rng(rng_seed)
    ids = np.arange(nrows, dtype=np.int32)
    flux = rng.normal(size=nrows).astype(np.float64)
    labels = np.array([f"L{i:04d}"[:10] for i in range(nrows)], dtype="S10")
    cols = astropy_fits.ColDefs(
        [
            astropy_fits.Column(name="id", format="I6", array=ids),
            astropy_fits.Column(name="flux", format="F12.4", array=flux),
            astropy_fits.Column(name="label", format="A10", array=labels),
        ]
    )
    hdu = astropy_fits.TableHDU.from_columns(cols)
    astropy_fits.HDUList([astropy_fits.PrimaryHDU(), hdu]).writeto(
        out_path, overwrite=True
    )
    return ["id", "flux", "label"]


def _to_tform(dtype: str) -> str:
    mapping = {
        "f4": "E",
        "f8": "D",
        "i4": "J",
        "i8": "K",
        "bool": "L",
    }
    if dtype in mapping:
        return mapping[dtype]
    if dtype.startswith("S"):
        return dtype[1:] + "A"
    raise ValueError(f"unsupported dtype spec: {dtype}")


def _gzip_fits(in_path: Path, out_path: Path) -> None:
    with in_path.open("rb") as f_in:
        with gzip.open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def _numpy_native_endian(arr: np.ndarray) -> np.ndarray:
    """Match TorchFits: deliver host-endian values (fitsio often leaves '>')."""
    if not isinstance(arr, np.ndarray):
        return arr
    if arr.dtype.names:
        need = any(
            arr.dtype.fields[name][0].byteorder not in ("=", "|")
            for name in arr.dtype.names
        )
        return arr.astype(arr.dtype.newbyteorder("=")) if need else arr
    if arr.dtype.byteorder not in ("=", "|"):
        return arr.astype(arr.dtype.newbyteorder("="))
    return arr


def _fitsio_read_native(path: Path, **kwargs):
    return _numpy_native_endian(fitsio.read(str(path), **kwargs))


def _table_to_torch_dict(data) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    if isinstance(data, dict):
        items = data.items()
    else:
        items = ((name, data[name]) for name in list(data.dtype.names or []))
    for name, value in items:
        if isinstance(value, list):
            # VLA: leave object heap columns out of Tensor dict (parity with prior skip).
            continue
        arr = np.ascontiguousarray(np.asarray(value))
        if arr.dtype.byteorder not in ("=", "|"):
            arr = arr.astype(arr.dtype.newbyteorder("="))
        if arr.dtype.kind in {"S", "U"}:
            if arr.dtype.kind == "U":
                arr = np.char.encode(arr, "ascii")
            arr = np.ascontiguousarray(arr).view("uint8").reshape(len(arr), -1)
        elif arr.dtype.kind == "b":
            arr = arr.astype(bool)
        elif arr.dtype.kind == "O":
            continue
        out[name] = torch.from_numpy(arr)
    return out


def _choose_numeric_col(
    columns: list[str], schema: list[tuple[str, str]] | None
) -> str:
    if schema:
        for name, dtype in schema:
            if dtype in {"f4", "f8", "i4", "i8"}:
                return name
    return columns[0]


def _bench_case(
    *,
    run_id: str,
    case: dict[str, Any],
    use_mmap: bool,
    policy_profile: str,
    warmup: int,
    operation_filter: str = "",
) -> list[dict[str, Any]]:
    path: Path = case["path"]
    nrows: int = int(case["nrows"])
    columns: list[str] = list(case["columns"])
    schema: list[tuple[str, str]] | None = case.get("schema")
    case_name = str(case["name"])
    compressed = bool(case.get("compressed", False))
    variable = bool(case.get("variable", False))
    unsupported = bool(case.get("unsupported", False))

    proj_cols = columns[: min(3, len(columns))]
    num_col = _choose_numeric_col(columns, schema)
    row_slice_start = 1
    row_slice_n = min(10_000, max(100, nrows // 10))

    mmap_target = "on" if use_mmap else "off"
    target_memmap = use_mmap

    runs = _repeats_for_rows(nrows)
    op_rx = re.compile(operation_filter) if operation_filter else None

    if unsupported:
        rows: list[dict[str, Any]] = []
        operations = [
            "read_full",
            "projection",
            "row_slice",
            "predicate_filter",
            "scan_count",
        ]
        if op_rx is not None:
            operations = [op for op in operations if op_rx.search(op)]
        method_specs = [
            ("torchfits", "torchfits", "smart", "smart"),
            ("astropy_torch", "astropy", "smart", "smart"),
            ("fitsio_torch", "fitsio", "smart", "smart"),
            ("torchfits_specialized", "torchfits", "specialized", "specialized"),
            ("astropy", "astropy", "specialized", "specialized"),
            ("fitsio", "fitsio", "specialized", "specialized"),
        ]
        for op_name in operations:
            for method, library, family, mode in method_specs:
                rows.append(
                    _make_row(
                        run_id=run_id,
                        case_name=case_name,
                        case=case,
                        operation=op_name,
                        family=family,
                        method=method,
                        library=library,
                        mode=mode,
                        mmap_target=mmap_target,
                        status="SKIPPED",
                        comparable=False,
                        skip_reason="compressed_table_case_not_enabled_in_default_env",
                        time_s=None,
                        throughput=None,
                        unit="rows/s",
                        n_points=nrows,
                    )
                )
        return rows

    try:
        import pyarrow.dataset as ds  # noqa: F401

        has_pyarrow = True
    except Exception:
        has_pyarrow = False

    print(
        f"[fitstable] case={case_name} rows={nrows} cols={len(columns)} compressed={compressed} variable={variable} runs={runs}",
        flush=True,
    )

    operations = {
        "read_full": {
            "torchfits": lambda: torchfits.read(
                str(path),
                hdu=1,
                mode="table",
                policy="smart",
                mmap=target_memmap,
                **_TF_READ_NO_CACHE,
            ),
            "torchfits_specialized": lambda: torchfits.read_table(
                str(path), hdu=1, mmap=target_memmap, **_TF_NO_CACHE
            ),
            "astropy": lambda: _astropy_read_full(path, memmap=target_memmap),
            "astropy_torch": lambda: _table_to_torch_dict(
                _astropy_read_full(path, memmap=target_memmap)
            ),
            "fitsio": lambda: _fitsio_read_native(path, ext=1),
            "fitsio_torch": lambda: _table_to_torch_dict(fitsio.read(str(path), ext=1)),
        },
        "projection": {
            "torchfits": lambda: torchfits.read(
                str(path),
                hdu=1,
                mode="table",
                policy="smart",
                mmap=target_memmap,
                columns=proj_cols,
                **_TF_READ_NO_CACHE,
            ),
            "torchfits_specialized": lambda: torchfits.read_table(
                str(path),
                hdu=1,
                columns=proj_cols,
                mmap=target_memmap,
                **_TF_NO_CACHE,
            ),
            "astropy": lambda: _astropy_projection(
                path, proj_cols, memmap=target_memmap
            ),
            "astropy_torch": lambda: _table_to_torch_dict(
                _astropy_projection(path, proj_cols, memmap=target_memmap)
            ),
            "fitsio": lambda: _fitsio_read_native(path, ext=1, columns=proj_cols),
            "fitsio_torch": lambda: _table_to_torch_dict(
                fitsio.read(str(path), ext=1, columns=proj_cols)
            ),
        },
        "row_slice": {
            "torchfits": lambda: torchfits.read(
                str(path),
                hdu=1,
                mode="table",
                policy="smart",
                mmap=target_memmap,
                start_row=row_slice_start,
                num_rows=row_slice_n,
                **_TF_READ_NO_CACHE,
            ),
            "torchfits_specialized": lambda: torchfits.read_table_rows(
                str(path),
                hdu=1,
                start_row=row_slice_start,
                num_rows=row_slice_n,
                mmap=target_memmap,
                **_TF_NO_CACHE,
            ),
            "astropy": lambda: _astropy_row_slice(
                path, row_slice_start, row_slice_n, memmap=target_memmap
            ),
            "astropy_torch": lambda: _table_to_torch_dict(
                _astropy_row_slice(
                    path, row_slice_start, row_slice_n, memmap=target_memmap
                )
            ),
            "fitsio": lambda: _fitsio_row_slice(path, row_slice_start, row_slice_n),
            "fitsio_torch": lambda: _table_to_torch_dict(
                _fitsio_row_slice(path, row_slice_start, row_slice_n)
            ),
        },
        "predicate_filter": {
            "torchfits": lambda: _torchfits_filter_pushdown(
                path, col=num_col, mmap=target_memmap, has_pyarrow=has_pyarrow
            ),
            "torchfits_specialized": lambda: _torchfits_filter_col_local(
                path, col=num_col, mmap=target_memmap
            ),
            # Specialized peers: single-column project+filter (not full-table mask).
            "astropy": lambda: _astropy_filter_col(
                path, col=num_col, memmap=target_memmap
            ),
            "astropy_torch": lambda: torch.as_tensor(
                _astropy_filter_col(path, col=num_col, memmap=target_memmap)
            ),
            "fitsio": lambda: _fitsio_filter_col(path, col=num_col),
            "fitsio_torch": lambda: torch.as_tensor(
                _fitsio_filter_col(path, col=num_col)
            ),
        },
        "scan_count": {
            "torchfits": lambda: _torchfits_scan_count(
                path, col=num_col, mmap=target_memmap, has_pyarrow=has_pyarrow
            ),
            "torchfits_specialized": lambda: _torchfits_scan_count_local(
                path, col=num_col, mmap=target_memmap
            ),
            "astropy": lambda: _astropy_scan_count(
                path, col=num_col, memmap=target_memmap
            ),
            "astropy_torch": lambda: _astropy_scan_count(
                path, col=num_col, memmap=target_memmap
            ),
            "fitsio": lambda: _fitsio_scan_count(path, col=num_col),
            "fitsio_torch": lambda: _fitsio_scan_count(path, col=num_col),
        },
    }

    rows: list[dict[str, Any]] = []

    for op_name, method_map in operations.items():
        if op_rx is not None and not op_rx.search(op_name):
            continue
        smart = {
            m: fn
            for m, fn in method_map.items()
            if m in {"torchfits", "astropy_torch", "fitsio_torch"}
        }
        specialized = {
            m: fn
            for m, fn in method_map.items()
            if m in {"torchfits_specialized", "astropy", "fitsio"}
        }
        timed: dict[
            str, tuple[float | None, float | None, float | None, str | None]
        ] = {}
        if smart:
            timed.update(time_medians_interleaved(smart, runs=runs, warmup=warmup))
        if specialized:
            timed.update(
                time_medians_interleaved(specialized, runs=runs, warmup=warmup)
            )

        for method, (t_val, peak_rss, peak_cuda, err) in timed.items():
            status = "OK" if t_val is not None else "FAILED"
            comparable = status == "OK"
            skip_reason = ""

            library = (
                "torchfits"
                if method.startswith("torchfits")
                else "fitsio"
                if method.startswith("fitsio")
                else "astropy"
            )
            family = (
                "smart"
                if method in {"torchfits", "astropy_torch", "fitsio_torch"}
                else "specialized"
            )
            mode = "smart" if family == "smart" else "specialized"

            # fitsio has no mmap toggle — timing it under mmap-on unfairly
            # ranks a buffered peer against torchfits honoring mmap=True.
            if method in {"fitsio", "fitsio_torch"} and target_memmap:
                comparable = False
                skip_reason = "fitsio_no_mmap: not comparable under mmap-on"

            # If strict mmap parity cannot be honored by astropy in this case, mark SKIPPED.
            if library == "astropy" and err and target_memmap:
                status = "SKIPPED"
                comparable = False
                skip_reason = f"strict_mmap_fairness: astropy memmap={target_memmap} unavailable ({err})"
                t_val = None

            throughput = (nrows / t_val) if (t_val is not None and t_val > 0) else None
            rows.append(
                _make_row(
                    run_id=run_id,
                    case_name=case_name,
                    case=case,
                    operation=op_name,
                    family=family,
                    method=method,
                    library=library,
                    mode=mode,
                    mmap_target=mmap_target,
                    status=status,
                    comparable=comparable,
                    skip_reason=skip_reason,
                    time_s=t_val,
                    peak_rss_mb=peak_rss,
                    peak_cuda_alloc_mb=peak_cuda,
                    throughput=throughput,
                    unit="rows/s",
                    n_points=nrows,
                )
            )

    return rows


def _astropy_materialize_col(data, name: str):
    """Eager-copy one FITS_rec column, including VLA heap payloads."""
    col = np.asarray(data[name])
    if col.dtype == object:
        return [np.asarray(x).copy() for x in col]
    return np.ascontiguousarray(col).copy()


def _astropy_materialize_table(
    data, *, names: list[str] | None = None
) -> dict[str, Any]:
    cols = list(names) if names is not None else list(data.columns.names)
    return {name: _astropy_materialize_col(data, name) for name in cols}


def _astropy_read_full(path: Path, *, memmap: bool):
    """Materialize the full table, including VLA heap payloads.

    ``np.array(FITS_rec, copy=False)`` leaves object/VLA columns as lazy
    views and understates Astropy cost vs eager TorchFits/fitsio reads.
    """
    with astropy_fits.open(path, memmap=memmap) as hdul:
        return _astropy_materialize_table(hdul[1].data)


def _astropy_projection(path: Path, columns: list[str], *, memmap: bool):
    with astropy_fits.open(path, memmap=memmap) as hdul:
        data = hdul[1].data
        # FITS_rec does not support list-of-names indexing directly.
        return _astropy_materialize_table(data, names=columns)


def _astropy_row_slice(path: Path, start_row: int, num_rows: int, *, memmap: bool):
    start0 = max(0, int(start_row) - 1)
    stop0 = start0 + int(num_rows)
    with astropy_fits.open(path, memmap=memmap) as hdul:
        data = hdul[1].data[start0:stop0]
        return _astropy_materialize_table(data)


def _astropy_filter(path: Path, *, col: str, memmap: bool):
    with astropy_fits.open(path, memmap=memmap) as hdul:
        data = hdul[1].data
        mask = np.asarray(data[col]) > 0
        return _astropy_materialize_table(data[mask])


def _astropy_filter_col(path: Path, *, col: str, memmap: bool):
    """Smart-family peer: project one column then filter (matches torchfits)."""
    with astropy_fits.open(path, memmap=memmap) as hdul:
        values = np.asarray(hdul[1].data[col])
        return values[values > 0]


def _astropy_scan_count(path: Path, *, col: str, memmap: bool):
    _ = col
    with astropy_fits.open(path, memmap=memmap) as hdul:
        return int(hdul[1].header.get("NAXIS2", 0))


def _fitsio_row_slice(path: Path, start_row: int, num_rows: int):
    start0 = max(0, int(start_row) - 1)
    stop0 = start0 + int(num_rows)
    return _fitsio_read_native(path, ext=1, rows=range(start0, stop0))


def _fitsio_filter(path: Path, *, col: str):
    data = _fitsio_read_native(path, ext=1)
    mask = np.asarray(data[col]) > 0
    filtered = data[mask]
    # Column-dict like TorchFits specialized (not a single structured ndarray).
    return {
        name: np.ascontiguousarray(filtered[name])
        for name in (filtered.dtype.names or [])
    }


def _fitsio_filter_col(path: Path, *, col: str):
    """Smart-family peer: project one column then filter (matches torchfits)."""
    data = _fitsio_read_native(path, ext=1, columns=[col])
    values = np.asarray(data[col])
    return values[values > 0]


def _fitsio_scan_count(path: Path, *, col: str):
    _ = col
    with fitsio.FITS(str(path)) as f:
        return int(f[1].get_nrows())


def _torchfits_filter_pushdown(path: Path, *, col: str, mmap: bool, has_pyarrow: bool):
    # Smart family scores the Tensor contract against fitsio_torch peers —
    # not Arrow interchange latency vs torch.from_numpy().
    _ = has_pyarrow
    data = torchfits.read_table(
        str(path), hdu=1, columns=[col], mmap=mmap, **_TF_NO_CACHE
    )
    values = data[col]
    if isinstance(values, torch.Tensor):
        return values[values > 0]
    arr = np.asarray(values)
    return arr[arr > 0]


def _torchfits_filter_local(path: Path, *, col: str, mmap: bool):
    """Specialized peer: full-table read then row filter (matches astropy/fitsio)."""
    from torchfits import cpp as _cpp

    data = _cpp.read_fits_table_rows_numpy(str(path), 1, [], 1, -1, bool(mmap))
    values = np.asarray(data[col])
    mask = values > 0
    return {k: np.ascontiguousarray(np.asarray(v)[mask]) for k, v in data.items()}


def _torchfits_filter_col_local(path: Path, *, col: str, mmap: bool):
    """Specialized peer: one-column project + mask (matches fitsio columns=[col])."""
    from torchfits import cpp as _cpp

    data = _cpp.read_fits_table_rows_numpy(str(path), 1, [col], 1, -1, bool(mmap))
    values = np.ascontiguousarray(np.asarray(data[col]))
    return values[values > 0]


def _torchfits_scan_count(path: Path, *, col: str, mmap: bool, has_pyarrow: bool):
    _ = col, mmap, has_pyarrow
    # Peer contract: NAXIS2 / get_nrows — not a column materialize.
    return int(torchfits.get_header(str(path), hdu=1).get("NAXIS2", 0))


def _torchfits_scan_count_local(path: Path, *, col: str, mmap: bool):
    _ = col, mmap
    return int(torchfits.get_header(str(path), hdu=1).get("NAXIS2", 0))


def _make_row(
    *,
    run_id: str,
    case_name: str,
    case: dict[str, Any],
    operation: str,
    family: str,
    method: str,
    library: str,
    mode: str,
    mmap_target: str,
    status: str,
    comparable: bool,
    skip_reason: str,
    time_s: float | None,
    throughput: float | None,
    unit: str,
    n_points: int,
    peak_rss_mb: float | None = None,
    peak_cuda_alloc_mb: float | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "domain": "fitstable",
        "suite": "fitstable_io",
        "case_id": f"{case_name}::{operation}",
        "case_label": f"{case_name} [{operation}]",
        "operation": operation,
        "family": family,
        "library": library,
        "method": method,
        "mode": mode,
        "status": status,
        "skip_reason": skip_reason,
        "comparable": comparable,
        "mmap_target": mmap_target,
        "host": _BENCH_HOST,
        "time_s": time_s,
        "peak_rss_mb": peak_rss_mb,
        "peak_cuda_alloc_mb": peak_cuda_alloc_mb,
        "throughput": throughput,
        "unit": unit,
        "size_mb": case["size_mb"],
        "n_points": n_points,
        "metadata": {
            "schema": case["schema_name"],
            "nrows": case["nrows"],
            "ncols": case["ncols"],
            "variable": case.get("variable", False),
            "compressed": case.get("compressed", False),
            "profile": case.get("profile", "default"),
        },
    }


def _build_cases(temp_dir: Path, *, quick: bool = False) -> list[dict[str, Any]]:
    schemas: dict[str, list[tuple[str, str]]] = {
        "narrow": [
            ("id", "i4"),
            ("flux", "f4"),
            ("err", "f4"),
            ("flag", "bool"),
        ],
        "mixed": [
            ("id", "i8"),
            ("ra", "f8"),
            ("dec", "f8"),
            ("flux", "f4"),
            ("fluxerr", "f4"),
            ("class", "i4"),
            ("name", "S16"),
            ("quality", "bool"),
        ],
        "wide": [
            *[(f"f{i:02d}", "f4") for i in range(20)],
            *[(f"i{i:02d}", "i4") for i in range(10)],
            *[(f"d{i:02d}", "f8") for i in range(6)],
            *[(f"s{i:02d}", "S12") for i in range(4)],
            ("flag", "bool"),
        ],
    }

    row_scales = (
        [1_000, 10_000, 100_000] if quick else [1_000, 10_000, 100_000, 1_000_000]
    )
    cases: list[dict[str, Any]] = []

    seed = 123
    for nrows in row_scales:
        schema_order = ["narrow", "mixed"]
        if nrows <= 100_000:
            schema_order.append("wide")

        for schema_name in schema_order:
            schema = schemas[schema_name]
            path = temp_dir / f"table_{schema_name}_{nrows}.fits"
            _schema, columns = _write_table_file(
                out_path=path,
                nrows=nrows,
                schema_name=schema_name,
                schema=schema,
                rng_seed=seed,
            )
            seed += 1
            cases.append(
                {
                    "name": f"{schema_name}_{nrows}",
                    "schema_name": schema_name,
                    "path": path,
                    "nrows": nrows,
                    "ncols": len(columns),
                    "columns": columns,
                    "schema": schema,
                    "size_mb": path.stat().st_size / (1024.0 * 1024.0),
                    "variable": False,
                    "compressed": False,
                    "profile": "base",
                }
            )

    for nrows in [1_000, 10_000] if quick else [1_000, 10_000, 100_000]:
        path = temp_dir / f"table_varlen_{nrows}.fits"
        columns = _write_varlen_file(out_path=path, nrows=nrows, rng_seed=seed)
        seed += 1
        cases.append(
            {
                "name": f"varlen_{nrows}",
                "schema_name": "varlen",
                "path": path,
                "nrows": nrows,
                "ncols": len(columns),
                "columns": columns,
                "schema": None,
                "size_mb": path.stat().st_size / (1024.0 * 1024.0),
                "variable": True,
                "compressed": False,
                "profile": "varlen",
            }
        )

    for nrows in [10_000] if quick else [10_000, 100_000]:
        path = temp_dir / f"table_typed_{nrows}.fits"
        columns = _write_typed_file(out_path=path, nrows=nrows, rng_seed=seed)
        seed += 1
        cases.append(
            {
                "name": f"typed_{nrows}",
                "schema_name": "typed",
                "path": path,
                "nrows": nrows,
                "ncols": len(columns),
                "columns": columns,
                "schema": None,
                "size_mb": path.stat().st_size / (1024.0 * 1024.0),
                "variable": False,
                "compressed": False,
                "profile": "typed",
            }
        )

    for nrows in [1_000] if quick else [1_000, 10_000]:
        path = temp_dir / f"table_ascii_{nrows}.fits"
        columns = _write_ascii_file(out_path=path, nrows=nrows, rng_seed=seed)
        seed += 1
        cases.append(
            {
                "name": f"ascii_{nrows}",
                "schema_name": "ascii",
                "path": path,
                "nrows": nrows,
                "ncols": len(columns),
                "columns": columns,
                "schema": None,
                "size_mb": path.stat().st_size / (1024.0 * 1024.0),
                "variable": False,
                "compressed": False,
                "profile": "ascii",
            }
        )

    # Compressed table benchmark placeholder (marked SKIPPED if not enabled).
    cases.append(
        {
            "name": "compressed_table_placeholder",
            "schema_name": "mixed",
            "path": temp_dir / "compressed_table_placeholder.fits",
            "nrows": 100_000,
            "ncols": 8,
            "columns": [
                "id",
                "ra",
                "dec",
                "flux",
                "fluxerr",
                "class",
                "name",
                "quality",
            ],
            "schema": schemas["mixed"],
            "size_mb": 0.0,
            "variable": False,
            "compressed": True,
            "unsupported": True,
            "profile": "compressed_placeholder",
        }
    )

    return cases


def run_fitstable_domain(
    *,
    run_id: str,
    output_dir: Path,
    use_mmap: bool = True,
    profile: str = "user",
    warmup: int = 1,
    quick: bool = False,
    max_cases: int | None = None,
    keep_temp: bool = False,
    case_filter: str = "",
    operation_filter: str = "",
) -> list[dict[str, Any]]:
    _ = profile
    temp_root = Path(tempfile.mkdtemp(prefix="torchfits_fitstable_"))
    rows: list[dict[str, Any]] = []

    try:
        cases = _build_cases(temp_root, quick=quick)
        if case_filter:
            rx = re.compile(case_filter)
            cases = [
                c
                for c in cases
                if rx.search(str(c.get("name", ""))) or rx.search(f"{c.get('name')}::")
            ]
            print(
                f"[fitstable] case filter {case_filter!r} -> {len(cases)} case(s)",
                flush=True,
            )
        if max_cases is not None and max_cases > 0:
            supported_cases = [
                c for c in cases if not bool(c.get("unsupported", False))
            ]
            cases = supported_cases[:max_cases]
            print(
                f"[fitstable] quick case cap applied: {len(cases)} case(s)", flush=True
            )
        for case in cases:
            rows.extend(
                _bench_case(
                    run_id=run_id,
                    case=case,
                    use_mmap=use_mmap,
                    policy_profile=profile,
                    warmup=warmup,
                    operation_filter=operation_filter,
                )
            )

        annotate_rankings(rows)
        return rows
    finally:
        if keep_temp:
            print(f"[fitstable] temp files kept: {temp_root}", flush=True)
        else:
            torchfits.clear_file_cache()
            shutil.rmtree(temp_root, ignore_errors=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--mmap", action="store_true")
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--profile", choices=["user", "lab"], default="user")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--filter", type=str, default="", help="Regex case filter")
    parser.add_argument(
        "--operation", type=str, default="", help="Regex operation filter"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    use_mmap = not args.no_mmap
    if args.mmap:
        use_mmap = True

    run_id = args.run_id.strip() or time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id

    rows = run_fitstable_domain(
        run_id=run_id,
        output_dir=run_dir,
        use_mmap=use_mmap,
        profile=args.profile,
        warmup=args.warmup,
        quick=args.quick,
        max_cases=(args.max_cases if args.max_cases > 0 else None),
        keep_temp=args.keep_temp,
        case_filter=args.filter,
        operation_filter=args.operation,
    )

    out_csv = run_dir / "fitstable_results.csv"
    write_csv(out_csv, rows, RESULT_COLUMNS)
    if args.json_out:
        write_json(args.json_out, rows)
    print(f"[fitstable] wrote {len(rows)} rows to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
