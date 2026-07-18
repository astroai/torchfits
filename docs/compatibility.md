# Compatibility matrix

Supported combinations for **torchfits 1.0.0rc1** wheels and source builds.

| Component | Supported | Notes |
|-----------|-----------|--------|
| Python | **3.10 – 3.13** | Classifiers and cibuildwheel matrix |
| PyTorch | **2.10.x** (`>=2.10,<2.11`) | ABI-matched native extension |
| NumPy | **≥ 1.20** | Runtime dependency |
| PyArrow | **≥ 5.0** (dev pin often 24.x) | Table / dataframe path |
| Platforms (wheels) | **Linux x86_64**, **macOS arm64** | Other arches: build from source |
| Optional: Polars / Pandas / DuckDB | via extras / env | Not required for core I/O |
| Optional: astropy / fitsio | test / bench only | Not imported by runtime I/O |

## Downstream guidance

| Consumer | Guidance |
|----------|----------|
| ML training (`Dataset` / `make_loader`) | Same Python + PyTorch 2.10 minor as the wheel |
| Arrow catalogs | Prefer `torchfits.table.read` / `scan` |
| Tensor columns | Prefer `torchfits.table.read_torch` / `scan_torch` |
| Root `read_table` / `stream_table` / `read_table_rows` | Deprecated compatibility aliases (warn in 1.0.0rc1) |

## Wheel install smoke

Local gate (current interpreter + sibling Pythons when present):

```bash
bash scripts/clean_install_smoke.sh
```

CI publishes the full cibuildwheel matrix on tagged releases
(`.github/workflows/build_wheels.yml`).
