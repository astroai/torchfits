# Benchmark Summary

- Run ID: `exhaustive_cuda_20260719_083810`
- Scopes: `fits, fitstable`
- Total normalized rows: `4087`
- TorchFits deficit rows (all lags): `127`
- TorchFits significant deficits: `20`
- Hostname: `torchfits-gpu-exhaustive-cuda-20260719-083810`
- CPU count: `96`
- torch.get_num_threads(): `8`
- Peak RSS (median across timed rows): `723.5 MB` (max `962.4 MB`)

## Domain Coverage

| Domain | Rows | Skipped |
|---|---:|---:|
| fits | 2943 | 85 |
| fitstable | 1144 | 76 |

## Astronomer Scorecard

| Domain | Family | TorchFits First | Win Rate | Legacy In Ranking |
|---|---|---:|---:|---:|
| fits | smart | 333/333 | 100.0% | 0 |
| fits | specialized | 419/419 | 100.0% | 0 |
| fitstable | smart | 153/164 | 93.3% | 0 |
| fitstable | specialized | 171/180 | 95.0% | 0 |

- TorchFits devices observed in this run: `cpu, cuda`
- Smart-family tables are the primary adoption view for astronomers (performance + portability).

## Adoption Checks

- `large-N` threshold: `n_points >= 100000`
- `small-N perceived` threshold: `torchfits_time_s < 0.000500s`
- `small-N max lag` threshold: `lag_ratio < 10.0x`

### Large-N Leadership

| Domain | Family | TorchFits First (large-N) | Win Rate |
|---|---|---:|---:|
| fitstable | smart | 58/63 | 92.1% |
| fitstable | specialized | 67/70 | 95.7% |

Large-N deficits detected:

| Case | n_points | Lag (x) | Behind (%) |
|---|---:|---:|---:|
| narrow_100000 [read_full] | 100000 | 1.252 | 25.24 |
| narrow_1000000 [read_full] | 1000000 | 1.148 | 14.84 |
| narrow_1000000 [scan_count] | 1000000 | 1.142 | 14.24 |
| mixed_100000 [scan_count] | 100000 | 1.107 | 10.71 |
| typed_100000 [scan_count] | 100000 | 1.056 | 5.65 |
| narrow_1000000 [predicate_filter] | 1000000 | 1.115 | 11.55 |
| typed_100000 [scan_count] | 100000 | 1.103 | 10.35 |
| varlen_100000 [scan_count] | 100000 | 1.058 | 5.84 |

### Small-N Visible Deficits

No small-N visible deficits detected.

## TorchFits Deficits (Not First)

### FITS - smart

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| small_int8_2d [read_full @ cuda] | read_full | 0.000128 | 773.5 | fitsio:fitsio_torch_device | 0.000121 | 773.5 | 1.059 | 5.95 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_int8_3d [read_full @ cuda] | read_full | 0.000161 | 773.5 | fitsio:fitsio_torch_device | 0.000154 | 773.5 | 1.043 | 4.35 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| compressed_hcompress_1 [read_full] | read_full | 0.030441 | 779.8 | fitsio:fitsio_torch | 0.029231 | 779.8 | 1.041 | 4.14 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| compressed_hcompress_1 [read_full @ cuda] | read_full | 0.030559 | 741.7 | fitsio:fitsio_torch_device | 0.029494 | 741.7 | 1.036 | 3.61 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| compressed_hcompress_1 [read_full @ cuda] | read_full | 0.030559 | 714.0 | fitsio:fitsio_torch_device | 0.029515 | 714.0 | 1.035 | 3.54 | on | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| compressed_hcompress_1 [read_full] | read_full | 0.030260 | 619.1 | fitsio:fitsio_torch | 0.029403 | 619.1 | 1.029 | 2.92 | on | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_int32_1d [read_full @ cuda] | read_full | 0.000115 | 773.5 | fitsio:fitsio_torch_device | 0.000114 | 773.5 | 1.011 | 1.10 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_uint16_2d [read_full @ cuda] | read_full | 0.000144 | 773.5 | fitsio:fitsio_torch_device | 0.000143 | 773.5 | 1.008 | 0.77 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_float64_1d [read_full @ cuda] | read_full | 0.000129 | 773.5 | fitsio:fitsio_torch_device | 0.000128 | 773.5 | 1.004 | 0.37 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| tiny_float32_2d [read_full @ cuda] | read_full | 0.000106 | 773.5 | fitsio:fitsio_torch_device | 0.000106 | 773.5 | 1.001 | 0.14 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |

### FITS - specialized

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| small_int16_1d [header_read] | header_read | 0.000062 | 779.8 | fitsio:fitsio | 0.000038 | 779.8 | 1.629 | 62.95 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_float32_1d [header_read] | header_read | 0.000064 | 779.8 | fitsio:fitsio | 0.000041 | 779.8 | 1.579 | 57.87 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| tiny_float32_1d [header_read] | header_read | 0.000066 | 779.8 | fitsio:fitsio | 0.000043 | 779.8 | 1.552 | 55.23 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| timeseries_frame_001 [header_read] | header_read | 0.000068 | 779.8 | fitsio:fitsio | 0.000044 | 779.8 | 1.551 | 55.08 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| tiny_float32_2d [header_read] | header_read | 0.000067 | 779.8 | fitsio:fitsio | 0.000044 | 779.8 | 1.547 | 54.69 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| tiny_int16_2d [header_read] | header_read | 0.000062 | 779.8 | fitsio:fitsio | 0.000041 | 779.8 | 1.529 | 52.85 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_float64_1d [header_read] | header_read | 0.000064 | 779.8 | fitsio:fitsio | 0.000042 | 779.8 | 1.507 | 50.71 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| tiny_int64_2d [header_read] | header_read | 0.000063 | 779.8 | fitsio:fitsio | 0.000042 | 779.8 | 1.502 | 50.17 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_int32_2d [header_read] | header_read | 0.000067 | 779.8 | fitsio:fitsio | 0.000045 | 779.8 | 1.495 | 49.55 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| timeseries_frame_003 [header_read] | header_read | 0.000065 | 779.8 | fitsio:fitsio | 0.000044 | 779.8 | 1.487 | 48.70 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| medium_float32_2d [header_read] | header_read | 0.000084 | 779.8 | fitsio:fitsio | 0.000056 | 779.8 | 1.484 | 48.37 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| medium_int32_3d [header_read] | header_read | 0.000065 | 779.8 | fitsio:fitsio | 0.000044 | 779.8 | 1.479 | 47.85 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| large_float64_2d [header_read] | header_read | 0.000066 | 779.8 | fitsio:fitsio | 0.000045 | 779.8 | 1.472 | 47.21 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| tiny_int32_1d [header_read] | header_read | 0.000060 | 779.8 | fitsio:fitsio | 0.000041 | 779.8 | 1.471 | 47.08 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| large_int64_2d [header_read] | header_read | 0.000066 | 779.8 | fitsio:fitsio | 0.000045 | 779.8 | 1.468 | 46.80 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| large_int64_1d [header_read] | header_read | 0.000064 | 779.8 | fitsio:fitsio | 0.000044 | 779.8 | 1.465 | 46.54 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| large_int16_1d [header_read] | header_read | 0.000063 | 779.8 | fitsio:fitsio | 0.000043 | 779.8 | 1.461 | 46.12 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_int64_2d [header_read] | header_read | 0.000064 | 779.8 | fitsio:fitsio | 0.000045 | 779.8 | 1.447 | 44.69 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| timeseries_frame_000 [header_read] | header_read | 0.000065 | 779.8 | fitsio:fitsio | 0.000045 | 779.8 | 1.436 | 43.63 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| large_int16_2d [header_read] | header_read | 0.000068 | 779.8 | fitsio:fitsio | 0.000047 | 779.8 | 1.432 | 43.17 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| large_int32_1d [header_read] | header_read | 0.000062 | 779.8 | fitsio:fitsio | 0.000043 | 779.8 | 1.427 | 42.71 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| medium_int32_2d [header_read] | header_read | 0.000068 | 779.8 | fitsio:fitsio | 0.000048 | 779.8 | 1.422 | 42.24 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| large_float32_1d [header_read] | header_read | 0.000063 | 779.8 | fitsio:fitsio | 0.000044 | 779.8 | 1.421 | 42.08 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| medium_int64_3d [header_read] | header_read | 0.000070 | 779.8 | fitsio:fitsio | 0.000050 | 779.8 | 1.407 | 40.72 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| large_float64_1d [header_read] | header_read | 0.000063 | 779.8 | fitsio:fitsio | 0.000045 | 779.8 | 1.405 | 40.46 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| medium_int32_1d [header_read] | header_read | 0.000065 | 779.8 | fitsio:fitsio | 0.000046 | 779.8 | 1.402 | 40.16 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| medium_int16_3d [header_read] | header_read | 0.000069 | 779.8 | fitsio:fitsio | 0.000049 | 779.8 | 1.399 | 39.94 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| large_float32_2d [header_read] | header_read | 0.000068 | 779.8 | fitsio:fitsio | 0.000049 | 779.8 | 1.398 | 39.80 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| medium_float64_2d [header_read] | header_read | 0.000079 | 779.8 | fitsio:fitsio | 0.000057 | 779.8 | 1.387 | 38.66 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| tiny_int16_3d [header_read] | header_read | 0.000064 | 779.8 | fitsio:fitsio | 0.000046 | 779.8 | 1.384 | 38.41 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| large_uint16_2d [header_read] | header_read | 0.000094 | 779.8 | fitsio:fitsio | 0.000069 | 779.8 | 1.375 | 37.53 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| timeseries_frame_004 [header_read] | header_read | 0.000065 | 779.8 | fitsio:fitsio | 0.000047 | 779.8 | 1.373 | 37.27 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_int64_1d [header_read] | header_read | 0.000061 | 779.8 | fitsio:fitsio | 0.000045 | 779.8 | 1.372 | 37.21 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| medium_float64_1d [header_read] | header_read | 0.000076 | 779.8 | fitsio:fitsio | 0.000056 | 779.8 | 1.369 | 36.88 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_float64_2d [header_read] | header_read | 0.000062 | 779.8 | fitsio:fitsio | 0.000045 | 779.8 | 1.365 | 36.46 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| medium_int16_1d [header_read] | header_read | 0.000077 | 779.8 | fitsio:fitsio | 0.000056 | 779.8 | 1.364 | 36.41 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| tiny_int64_1d [header_read] | header_read | 0.000058 | 779.8 | fitsio:fitsio | 0.000042 | 779.8 | 1.361 | 36.12 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| tiny_float64_1d [header_read] | header_read | 0.000060 | 779.8 | fitsio:fitsio | 0.000044 | 779.8 | 1.358 | 35.76 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_uint32_2d [header_read] | header_read | 0.000069 | 779.8 | fitsio:fitsio | 0.000051 | 779.8 | 1.354 | 35.38 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| tiny_int16_1d [header_read] | header_read | 0.000059 | 779.8 | fitsio:fitsio | 0.000044 | 779.8 | 1.352 | 35.17 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| tiny_float64_2d [header_read] | header_read | 0.000062 | 779.8 | fitsio:fitsio | 0.000046 | 779.8 | 1.351 | 35.14 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_int16_2d [header_read] | header_read | 0.000064 | 779.8 | fitsio:fitsio | 0.000047 | 779.8 | 1.351 | 35.09 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| timeseries_frame_002 [header_read] | header_read | 0.000064 | 779.8 | fitsio:fitsio | 0.000048 | 779.8 | 1.350 | 35.02 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| tiny_int8_1d [header_read] | header_read | 0.000067 | 779.8 | fitsio:fitsio | 0.000050 | 779.8 | 1.348 | 34.84 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| large_int32_2d [header_read] | header_read | 0.000066 | 779.8 | fitsio:fitsio | 0.000049 | 779.8 | 1.348 | 34.83 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_int32_1d [header_read] | header_read | 0.000058 | 779.8 | fitsio:fitsio | 0.000043 | 779.8 | 1.345 | 34.54 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_int32_3d [header_read] | header_read | 0.000066 | 779.8 | fitsio:fitsio | 0.000049 | 779.8 | 1.344 | 34.39 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_int8_3d [header_read] | header_read | 0.000069 | 779.8 | fitsio:fitsio | 0.000052 | 779.8 | 1.343 | 34.28 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| tiny_float64_3d [header_read] | header_read | 0.000066 | 779.8 | fitsio:fitsio | 0.000049 | 779.8 | 1.333 | 33.34 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_uint16_2d [header_read] | header_read | 0.000069 | 779.8 | fitsio:fitsio | 0.000052 | 779.8 | 1.333 | 33.30 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| tiny_int32_2d [header_read] | header_read | 0.000062 | 779.8 | fitsio:fitsio | 0.000047 | 779.8 | 1.332 | 33.22 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| large_uint32_2d [header_read] | header_read | 0.000087 | 779.8 | fitsio:fitsio | 0.000065 | 779.8 | 1.332 | 33.17 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| large_int8_1d [header_read] | header_read | 0.000086 | 779.8 | fitsio:fitsio | 0.000065 | 779.8 | 1.329 | 32.93 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| tiny_int32_3d [header_read] | header_read | 0.000066 | 779.8 | fitsio:fitsio | 0.000050 | 779.8 | 1.328 | 32.76 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| medium_uint32_2d [header_read] | header_read | 0.000072 | 779.8 | fitsio:fitsio | 0.000054 | 779.8 | 1.325 | 32.52 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_float32_2d [header_read] | header_read | 0.000062 | 779.8 | fitsio:fitsio | 0.000047 | 779.8 | 1.321 | 32.11 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| large_int8_2d [header_read] | header_read | 0.000088 | 779.8 | fitsio:fitsio | 0.000068 | 779.8 | 1.304 | 30.39 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_float32_3d [header_read] | header_read | 0.000064 | 779.8 | fitsio:fitsio | 0.000049 | 779.8 | 1.301 | 30.08 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| medium_int8_2d [header_read] | header_read | 0.000070 | 779.8 | fitsio:fitsio | 0.000054 | 779.8 | 1.300 | 30.02 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| tiny_int64_3d [header_read] | header_read | 0.000065 | 779.8 | fitsio:fitsio | 0.000050 | 779.8 | 1.294 | 29.36 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| mef_medium [header_read] | header_read | 0.000081 | 779.8 | fitsio:fitsio | 0.000063 | 779.8 | 1.292 | 29.17 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| medium_uint16_2d [header_read] | header_read | 0.000069 | 779.8 | fitsio:fitsio | 0.000054 | 779.8 | 1.286 | 28.64 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| medium_float32_1d [header_read] | header_read | 0.000074 | 779.8 | fitsio:fitsio | 0.000058 | 779.8 | 1.281 | 28.12 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| tiny_float32_3d [header_read] | header_read | 0.000065 | 779.8 | fitsio:fitsio | 0.000051 | 779.8 | 1.277 | 27.74 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| tiny_int8_3d [header_read] | header_read | 0.000071 | 779.8 | fitsio:fitsio | 0.000056 | 779.8 | 1.264 | 26.44 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| medium_int64_1d [header_read] | header_read | 0.000057 | 779.8 | fitsio:fitsio | 0.000045 | 779.8 | 1.261 | 26.07 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_float64_3d [header_read] | header_read | 0.000065 | 779.8 | fitsio:fitsio | 0.000052 | 779.8 | 1.259 | 25.94 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| medium_float32_3d [header_read] | header_read | 0.000076 | 779.8 | fitsio:fitsio | 0.000061 | 779.8 | 1.247 | 24.65 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| medium_int8_1d [header_read] | header_read | 0.000064 | 779.8 | fitsio:fitsio | 0.000052 | 779.8 | 1.244 | 24.44 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| scaled_small [header_read] | header_read | 0.000070 | 779.8 | fitsio:fitsio | 0.000056 | 779.8 | 1.244 | 24.38 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| medium_int16_2d [header_read] | header_read | 0.000074 | 779.8 | fitsio:fitsio | 0.000060 | 779.8 | 1.235 | 23.49 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_int8_1d [header_read] | header_read | 0.000061 | 779.8 | fitsio:fitsio | 0.000049 | 779.8 | 1.228 | 22.82 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| medium_int64_2d [header_read] | header_read | 0.000059 | 779.8 | fitsio:fitsio | 0.000048 | 779.8 | 1.225 | 22.52 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| multi_mef_10ext [header_read] | header_read | 0.000081 | 779.8 | fitsio:fitsio | 0.000067 | 779.8 | 1.214 | 21.37 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_int16_3d [header_read] | header_read | 0.000063 | 779.8 | fitsio:fitsio | 0.000052 | 779.8 | 1.211 | 21.14 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| scaled_medium [header_read] | header_read | 0.000068 | 779.8 | fitsio:fitsio | 0.000056 | 779.8 | 1.203 | 20.26 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| medium_float64_3d [header_read] | header_read | 0.000075 | 779.8 | fitsio:fitsio | 0.000063 | 779.8 | 1.202 | 20.25 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_int64_3d [header_read] | header_read | 0.000062 | 779.8 | fitsio:fitsio | 0.000052 | 779.8 | 1.201 | 20.10 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| scaled_large [header_read] | header_read | 0.000066 | 779.8 | fitsio:fitsio | 0.000056 | 779.8 | 1.188 | 18.83 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| small_int8_2d [header_read] | header_read | 0.000063 | 779.8 | fitsio:fitsio | 0.000053 | 779.8 | 1.186 | 18.62 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| mef_small [header_read] | header_read | 0.000077 | 779.8 | fitsio:fitsio | 0.000066 | 779.8 | 1.165 | 16.51 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| tiny_int8_2d [header_read] | header_read | 0.000061 | 779.8 | fitsio:fitsio | 0.000053 | 779.8 | 1.150 | 15.01 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| medium_int8_3d [header_read] | header_read | 0.000070 | 779.8 | fitsio:fitsio | 0.000063 | 779.8 | 1.116 | 11.56 | n/a | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| large_int8_2d [read_full] | read_full | 0.000938 | 779.8 | fitsio:fitsio_torch | 0.000903 | 779.8 | 1.039 | 3.87 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| compressed_hcompress_1 [read_full @ cuda] | read_full | 0.030520 | 741.7 | fitsio:fitsio_torch_device_specialized | 0.029479 | 741.7 | 1.035 | 3.53 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| compressed_hcompress_1 [read_full @ cuda] | read_full | 0.030897 | 714.0 | fitsio:fitsio_torch_device_specialized | 0.029851 | 714.0 | 1.035 | 3.50 | on | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| compressed_hcompress_1 [read_full] | read_full | 0.030401 | 619.1 | fitsio:fitsio_torch | 0.029403 | 619.1 | 1.034 | 3.40 | on | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| compressed_hcompress_1 [read_full] | read_full | 0.030207 | 779.8 | fitsio:fitsio_torch | 0.029231 | 779.8 | 1.033 | 3.34 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |

### FITSTABLE - smart

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| narrow_100000 [read_full] | read_full | 0.000933 | 701.2 | fitsio:fitsio_torch | 0.000745 | 701.2 | 1.252 | 25.24 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| narrow_1000000 [read_full] | read_full | 0.007634 | 722.1 | fitsio:fitsio_torch | 0.006648 | 722.1 | 1.148 | 14.84 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| narrow_1000000 [scan_count] | scan_count | 0.000103 | 723.1 | fitsio:fitsio_torch | 0.000091 | 723.1 | 1.142 | 14.24 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| ascii_10000 [scan_count] | scan_count | 0.000096 | 747.4 | fitsio:fitsio_torch | 0.000086 | 747.4 | 1.116 | 11.64 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| mixed_100000 [scan_count] | scan_count | 0.000117 | 717.7 | fitsio:fitsio_torch | 0.000106 | 717.7 | 1.107 | 10.71 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| varlen_10000 [scan_count] | scan_count | 0.000093 | 719.2 | fitsio:fitsio_torch | 0.000086 | 719.2 | 1.085 | 8.45 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| mixed_10000 [scan_count] | scan_count | 0.000112 | 700.6 | fitsio:fitsio_torch | 0.000104 | 700.6 | 1.081 | 8.11 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| narrow_10000 [scan_count] | scan_count | 0.000095 | 700.2 | fitsio:fitsio_torch | 0.000088 | 700.2 | 1.080 | 7.96 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| typed_10000 [scan_count] | scan_count | 0.000114 | 747.3 | fitsio:fitsio_torch | 0.000107 | 747.3 | 1.073 | 7.25 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| typed_100000 [scan_count] | scan_count | 0.000102 | 747.3 | fitsio:fitsio_torch | 0.000096 | 747.3 | 1.056 | 5.65 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| ascii_1000 [scan_count] | scan_count | 0.000095 | 747.4 | fitsio:fitsio_torch | 0.000090 | 747.4 | 1.055 | 5.49 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| ascii_10000 [predicate_filter] | predicate_filter | 0.000495 | 747.4 | fitsio:fitsio_torch | 0.000481 | 747.4 | 1.029 | 2.92 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| narrow_1000 [scan_count] | scan_count | 0.000092 | 699.6 | fitsio:fitsio_torch | 0.000090 | 699.6 | 1.027 | 2.67 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| mixed_1000 [scan_count] | scan_count | 0.000108 | 699.7 | fitsio:fitsio_torch | 0.000106 | 699.7 | 1.019 | 1.86 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| varlen_100000 [scan_count] | scan_count | 0.000092 | 747.2 | fitsio:fitsio_torch | 0.000091 | 747.2 | 1.007 | 0.74 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |

### FITSTABLE - specialized

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| narrow_1000000 [predicate_filter] | predicate_filter | 0.015128 | 723.1 | astropy:astropy | 0.013562 | 723.1 | 1.115 | 11.55 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| typed_100000 [scan_count] | scan_count | 0.000102 | 747.3 | fitsio:fitsio | 0.000092 | 747.3 | 1.103 | 10.35 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| varlen_10000 [scan_count] | scan_count | 0.000095 | 719.2 | fitsio:fitsio | 0.000087 | 719.2 | 1.089 | 8.91 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| ascii_10000 [scan_count] | scan_count | 0.000097 | 747.4 | fitsio:fitsio | 0.000089 | 747.4 | 1.085 | 8.49 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| narrow_10000 [scan_count] | scan_count | 0.000096 | 700.2 | fitsio:fitsio | 0.000090 | 700.2 | 1.075 | 7.50 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| ascii_1000 [scan_count] | scan_count | 0.000092 | 747.4 | fitsio:fitsio | 0.000086 | 747.4 | 1.074 | 7.39 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| typed_10000 [scan_count] | scan_count | 0.000116 | 747.3 | fitsio:fitsio | 0.000108 | 747.3 | 1.073 | 7.27 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| narrow_1000 [scan_count] | scan_count | 0.000096 | 699.6 | fitsio:fitsio | 0.000091 | 699.6 | 1.060 | 6.01 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| varlen_100000 [scan_count] | scan_count | 0.000094 | 747.2 | fitsio:fitsio | 0.000088 | 747.2 | 1.058 | 5.84 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| mixed_10000 [scan_count] | scan_count | 0.000107 | 700.6 | fitsio:fitsio | 0.000103 | 700.6 | 1.035 | 3.51 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| varlen_1000 [scan_count] | scan_count | 0.000090 | 719.2 | fitsio:fitsio | 0.000087 | 719.2 | 1.028 | 2.83 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| narrow_100000 [scan_count] | scan_count | 0.000096 | 701.4 | fitsio:fitsio | 0.000094 | 701.4 | 1.025 | 2.47 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| mixed_1000 [scan_count] | scan_count | 0.000112 | 699.7 | fitsio:fitsio | 0.000109 | 699.7 | 1.024 | 2.37 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |
| narrow_1000000 [read_full] | read_full | 0.007889 | 723.1 | fitsio:fitsio | 0.007754 | 723.1 | 1.017 | 1.73 | off | torchfits-gpu-exhaustive-cuda-20260719-083810 |

## Notes

- Strict mmap fairness is enforced in comparable sets. Rows with unmatched mmap controls are marked `SKIPPED`.
- Rankings are family-specific and never mix smart vs specialized method families.
