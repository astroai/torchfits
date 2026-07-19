# Benchmark Summary

- Run ID: `exhaustive_cpu_20260719_083800`
- Scopes: `fits, fitstable`
- Total normalized rows: `2829`
- TorchFits deficit rows (all lags): `101`
- TorchFits significant deficits: `5`
- Hostname: `torchfits-gpu-exhaustive-cpu-20260719-083800`
- CPU count: `192`
- torch.get_num_threads(): `8`
- Peak RSS (median across timed rows): `290.1 MB` (max `620.1 MB`)

## Domain Coverage

| Domain | Rows | Skipped |
|---|---:|---:|
| fits | 1689 | 85 |
| fitstable | 1140 | 76 |

## Astronomer Scorecard

| Domain | Family | TorchFits First | Win Rate | Legacy In Ranking |
|---|---|---:|---:|---:|
| fits | smart | 156/156 | 100.0% | 0 |
| fits | specialized | 242/242 | 100.0% | 0 |
| fitstable | smart | 161/164 | 98.2% | 0 |
| fitstable | specialized | 178/180 | 98.9% | 0 |

- TorchFits devices observed in this run: `-`
- Smart-family tables are the primary adoption view for astronomers (performance + portability).

## Adoption Checks

- `large-N` threshold: `n_points >= 100000`
- `small-N perceived` threshold: `torchfits_time_s < 0.000500s`
- `small-N max lag` threshold: `lag_ratio < 10.0x`

### Large-N Leadership

| Domain | Family | TorchFits First (large-N) | Win Rate |
|---|---|---:|---:|
| fitstable | smart | 62/63 | 98.4% |
| fitstable | specialized | 69/70 | 98.6% |

Large-N deficits detected:

| Case | n_points | Lag (x) | Behind (%) |
|---|---:|---:|---:|
| narrow_1000000 [scan_count] | 1000000 | 1.061 | 6.06 |
| narrow_1000000 [predicate_filter] | 1000000 | 1.139 | 13.92 |

### Small-N Visible Deficits

No small-N visible deficits detected.

## TorchFits Deficits (Not First)

### FITS - smart

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| compressed_hcompress_1 [read_full] | read_full | 0.045524 | 289.3 | fitsio:fitsio_torch | 0.044323 | 289.3 | 1.027 | 2.71 | on | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| compressed_hcompress_1 [read_full] | read_full | 0.045476 | 290.6 | fitsio:fitsio_torch | 0.044321 | 290.6 | 1.026 | 2.60 | off | torchfits-gpu-exhaustive-cpu-20260719-083800 |

### FITS - specialized

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| large_int32_2d [header_read] | header_read | 0.000042 | 277.4 | fitsio:fitsio | 0.000028 | 277.4 | 1.516 | 51.65 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| tiny_float64_1d [header_read] | header_read | 0.000039 | 277.4 | fitsio:fitsio | 0.000026 | 277.4 | 1.514 | 51.37 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| small_int64_1d [header_read] | header_read | 0.000040 | 277.4 | fitsio:fitsio | 0.000027 | 277.4 | 1.490 | 49.04 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| medium_int32_3d [header_read] | header_read | 0.000043 | 277.4 | fitsio:fitsio | 0.000029 | 277.4 | 1.490 | 48.97 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| tiny_int16_1d [header_read] | header_read | 0.000037 | 277.4 | fitsio:fitsio | 0.000025 | 277.4 | 1.489 | 48.89 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| medium_int32_2d [header_read] | header_read | 0.000040 | 277.4 | fitsio:fitsio | 0.000027 | 277.4 | 1.478 | 47.83 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| small_int64_2d [header_read] | header_read | 0.000041 | 277.4 | fitsio:fitsio | 0.000028 | 277.4 | 1.475 | 47.47 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| medium_int32_1d [header_read] | header_read | 0.000037 | 277.4 | fitsio:fitsio | 0.000025 | 277.4 | 1.471 | 47.10 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| small_float32_1d [header_read] | header_read | 0.000037 | 277.4 | fitsio:fitsio | 0.000025 | 277.4 | 1.468 | 46.76 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| medium_int16_1d [header_read] | header_read | 0.000038 | 277.4 | fitsio:fitsio | 0.000026 | 277.4 | 1.448 | 44.77 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| large_int64_1d [header_read] | header_read | 0.000037 | 277.4 | fitsio:fitsio | 0.000026 | 277.4 | 1.446 | 44.60 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| large_float64_1d [header_read] | header_read | 0.000042 | 277.4 | fitsio:fitsio | 0.000029 | 277.4 | 1.446 | 44.58 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| small_int16_1d [header_read] | header_read | 0.000037 | 277.4 | fitsio:fitsio | 0.000026 | 277.4 | 1.437 | 43.68 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| large_float32_2d [header_read] | header_read | 0.000040 | 277.4 | fitsio:fitsio | 0.000028 | 277.4 | 1.433 | 43.31 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| tiny_float64_3d [header_read] | header_read | 0.000042 | 277.4 | fitsio:fitsio | 0.000030 | 277.4 | 1.432 | 43.23 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| small_float64_1d [header_read] | header_read | 0.000038 | 277.4 | fitsio:fitsio | 0.000026 | 277.4 | 1.431 | 43.08 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| medium_int64_1d [header_read] | header_read | 0.000038 | 277.4 | fitsio:fitsio | 0.000027 | 277.4 | 1.431 | 43.07 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| small_uint32_2d [header_read] | header_read | 0.000046 | 277.4 | fitsio:fitsio | 0.000032 | 277.4 | 1.417 | 41.69 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| small_float32_3d [header_read] | header_read | 0.000041 | 277.4 | fitsio:fitsio | 0.000029 | 277.4 | 1.411 | 41.09 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| large_int64_2d [header_read] | header_read | 0.000039 | 277.4 | fitsio:fitsio | 0.000028 | 277.4 | 1.397 | 39.74 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| medium_float32_2d [header_read] | header_read | 0.000040 | 277.4 | fitsio:fitsio | 0.000029 | 277.4 | 1.397 | 39.73 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| timeseries_frame_000 [header_read] | header_read | 0.000040 | 277.4 | fitsio:fitsio | 0.000029 | 277.4 | 1.388 | 38.81 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| medium_float64_1d [header_read] | header_read | 0.000038 | 277.4 | fitsio:fitsio | 0.000028 | 277.4 | 1.386 | 38.57 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| large_int16_1d [header_read] | header_read | 0.000037 | 277.4 | fitsio:fitsio | 0.000027 | 277.4 | 1.374 | 37.43 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| tiny_int32_1d [header_read] | header_read | 0.000036 | 277.4 | fitsio:fitsio | 0.000026 | 277.4 | 1.372 | 37.23 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| medium_float32_1d [header_read] | header_read | 0.000037 | 277.4 | fitsio:fitsio | 0.000027 | 277.4 | 1.372 | 37.15 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| tiny_int64_1d [header_read] | header_read | 0.000035 | 277.4 | fitsio:fitsio | 0.000025 | 277.4 | 1.366 | 36.58 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| large_int32_1d [header_read] | header_read | 0.000034 | 277.4 | fitsio:fitsio | 0.000025 | 277.4 | 1.362 | 36.15 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| medium_float64_2d [header_read] | header_read | 0.000040 | 277.4 | fitsio:fitsio | 0.000029 | 277.4 | 1.361 | 36.12 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| timeseries_frame_003 [header_read] | header_read | 0.000040 | 277.4 | fitsio:fitsio | 0.000029 | 277.4 | 1.358 | 35.83 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| tiny_int64_2d [header_read] | header_read | 0.000037 | 277.4 | fitsio:fitsio | 0.000027 | 277.4 | 1.357 | 35.74 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| small_int32_1d [header_read] | header_read | 0.000036 | 277.4 | fitsio:fitsio | 0.000027 | 277.4 | 1.354 | 35.44 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| large_int8_1d [header_read] | header_read | 0.000041 | 277.4 | fitsio:fitsio | 0.000031 | 277.4 | 1.348 | 34.80 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| timeseries_frame_001 [header_read] | header_read | 0.000039 | 277.4 | fitsio:fitsio | 0.000029 | 277.4 | 1.347 | 34.72 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| tiny_int8_2d [header_read] | header_read | 0.000047 | 277.4 | fitsio:fitsio | 0.000035 | 277.4 | 1.340 | 33.98 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| timeseries_frame_002 [header_read] | header_read | 0.000038 | 277.4 | fitsio:fitsio | 0.000029 | 277.4 | 1.336 | 33.60 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| tiny_int64_3d [header_read] | header_read | 0.000041 | 277.4 | fitsio:fitsio | 0.000030 | 277.4 | 1.336 | 33.60 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| tiny_float32_2d [header_read] | header_read | 0.000037 | 277.4 | fitsio:fitsio | 0.000028 | 277.4 | 1.335 | 33.50 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| large_float32_1d [header_read] | header_read | 0.000035 | 277.4 | fitsio:fitsio | 0.000026 | 277.4 | 1.333 | 33.32 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| small_int32_2d [header_read] | header_read | 0.000037 | 277.4 | fitsio:fitsio | 0.000028 | 277.4 | 1.331 | 33.05 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| tiny_int32_2d [header_read] | header_read | 0.000038 | 277.4 | fitsio:fitsio | 0.000028 | 277.4 | 1.328 | 32.80 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| medium_uint16_2d [header_read] | header_read | 0.000046 | 277.4 | fitsio:fitsio | 0.000035 | 277.4 | 1.319 | 31.88 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| small_float64_2d [header_read] | header_read | 0.000039 | 277.4 | fitsio:fitsio | 0.000030 | 277.4 | 1.319 | 31.86 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| large_int8_2d [header_read] | header_read | 0.000045 | 277.4 | fitsio:fitsio | 0.000034 | 277.4 | 1.318 | 31.79 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| tiny_float64_2d [header_read] | header_read | 0.000038 | 277.4 | fitsio:fitsio | 0.000029 | 277.4 | 1.311 | 31.15 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| tiny_float32_1d [header_read] | header_read | 0.000034 | 277.4 | fitsio:fitsio | 0.000026 | 277.4 | 1.309 | 30.86 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| large_float64_2d [header_read] | header_read | 0.000039 | 277.4 | fitsio:fitsio | 0.000030 | 277.4 | 1.306 | 30.59 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| medium_float64_3d [header_read] | header_read | 0.000040 | 277.4 | fitsio:fitsio | 0.000031 | 277.4 | 1.304 | 30.41 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| small_int16_2d [header_read] | header_read | 0.000036 | 277.4 | fitsio:fitsio | 0.000028 | 277.4 | 1.299 | 29.90 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| medium_int64_3d [header_read] | header_read | 0.000038 | 277.4 | fitsio:fitsio | 0.000030 | 277.4 | 1.296 | 29.55 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| small_float64_3d [header_read] | header_read | 0.000041 | 277.4 | fitsio:fitsio | 0.000032 | 277.4 | 1.294 | 29.39 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| small_int16_3d [header_read] | header_read | 0.000039 | 277.4 | fitsio:fitsio | 0.000030 | 277.4 | 1.294 | 29.39 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| large_uint16_2d [header_read] | header_read | 0.000043 | 277.4 | fitsio:fitsio | 0.000033 | 277.4 | 1.294 | 29.38 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| tiny_int32_3d [header_read] | header_read | 0.000039 | 277.4 | fitsio:fitsio | 0.000031 | 277.4 | 1.282 | 28.18 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| medium_float32_3d [header_read] | header_read | 0.000039 | 277.4 | fitsio:fitsio | 0.000030 | 277.4 | 1.278 | 27.84 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| small_int32_3d [header_read] | header_read | 0.000039 | 277.4 | fitsio:fitsio | 0.000030 | 277.4 | 1.278 | 27.80 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| large_int16_2d [header_read] | header_read | 0.000038 | 277.4 | fitsio:fitsio | 0.000030 | 277.4 | 1.272 | 27.15 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| timeseries_frame_004 [header_read] | header_read | 0.000036 | 277.4 | fitsio:fitsio | 0.000029 | 277.4 | 1.265 | 26.49 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| small_int8_1d [header_read] | header_read | 0.000041 | 277.4 | fitsio:fitsio | 0.000032 | 277.4 | 1.265 | 26.46 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| medium_int64_2d [header_read] | header_read | 0.000037 | 277.4 | fitsio:fitsio | 0.000029 | 277.4 | 1.258 | 25.83 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| tiny_int16_3d [header_read] | header_read | 0.000038 | 277.4 | fitsio:fitsio | 0.000030 | 277.4 | 1.255 | 25.49 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| multi_mef_10ext [header_read] | header_read | 0.000054 | 277.4 | fitsio:fitsio | 0.000043 | 277.4 | 1.255 | 25.48 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| scaled_medium [header_read] | header_read | 0.000042 | 277.4 | fitsio:fitsio | 0.000034 | 277.4 | 1.253 | 25.29 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| medium_uint32_2d [header_read] | header_read | 0.000043 | 277.4 | fitsio:fitsio | 0.000034 | 277.4 | 1.249 | 24.88 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| small_int64_3d [header_read] | header_read | 0.000038 | 277.4 | fitsio:fitsio | 0.000031 | 277.4 | 1.248 | 24.81 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| medium_int8_1d [header_read] | header_read | 0.000040 | 277.4 | fitsio:fitsio | 0.000032 | 277.4 | 1.245 | 24.54 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| tiny_int8_1d [header_read] | header_read | 0.000040 | 277.4 | fitsio:fitsio | 0.000032 | 277.4 | 1.245 | 24.47 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| scaled_large [header_read] | header_read | 0.000043 | 277.4 | fitsio:fitsio | 0.000035 | 277.4 | 1.245 | 24.47 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| small_int8_2d [header_read] | header_read | 0.000043 | 277.4 | fitsio:fitsio | 0.000035 | 277.4 | 1.242 | 24.17 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| medium_int8_2d [header_read] | header_read | 0.000042 | 277.4 | fitsio:fitsio | 0.000034 | 277.4 | 1.242 | 24.16 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| scaled_small [header_read] | header_read | 0.000041 | 277.4 | fitsio:fitsio | 0.000033 | 277.4 | 1.240 | 24.04 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| medium_int16_2d [header_read] | header_read | 0.000037 | 277.4 | fitsio:fitsio | 0.000030 | 277.4 | 1.236 | 23.56 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| medium_int16_3d [header_read] | header_read | 0.000038 | 277.4 | fitsio:fitsio | 0.000031 | 277.4 | 1.231 | 23.14 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| tiny_float32_3d [header_read] | header_read | 0.000039 | 277.4 | fitsio:fitsio | 0.000032 | 277.4 | 1.231 | 23.11 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| medium_int8_3d [header_read] | header_read | 0.000044 | 277.4 | fitsio:fitsio | 0.000035 | 277.4 | 1.231 | 23.07 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| mef_small [header_read] | header_read | 0.000054 | 277.4 | fitsio:fitsio | 0.000044 | 277.4 | 1.228 | 22.77 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| mef_medium [header_read] | header_read | 0.000054 | 277.4 | fitsio:fitsio | 0.000044 | 277.4 | 1.225 | 22.49 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| small_float32_2d [header_read] | header_read | 0.000037 | 277.4 | fitsio:fitsio | 0.000030 | 277.4 | 1.223 | 22.29 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| tiny_int16_2d [header_read] | header_read | 0.000036 | 277.4 | fitsio:fitsio | 0.000030 | 277.4 | 1.221 | 22.15 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| small_int8_3d [header_read] | header_read | 0.000043 | 277.4 | fitsio:fitsio | 0.000036 | 277.4 | 1.199 | 19.92 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| small_uint16_2d [header_read] | header_read | 0.000039 | 277.4 | fitsio:fitsio | 0.000033 | 277.4 | 1.180 | 17.95 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| large_uint32_2d [header_read] | header_read | 0.000043 | 277.4 | fitsio:fitsio | 0.000036 | 277.4 | 1.179 | 17.87 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| tiny_int8_3d [header_read] | header_read | 0.000042 | 277.4 | fitsio:fitsio | 0.000036 | 277.4 | 1.168 | 16.85 | n/a | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| compressed_hcompress_1 [read_full] | read_full | 0.045615 | 289.3 | fitsio:fitsio_torch | 0.044323 | 289.3 | 1.029 | 2.91 | on | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| compressed_hcompress_1 [read_full] | read_full | 0.045309 | 290.6 | fitsio:fitsio_torch | 0.044321 | 290.6 | 1.022 | 2.23 | off | torchfits-gpu-exhaustive-cpu-20260719-083800 |

### FITSTABLE - smart

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| ascii_10000 [predicate_filter] | predicate_filter | 0.000401 | 411.1 | fitsio:fitsio_torch | 0.000367 | 411.1 | 1.094 | 9.43 | off | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| narrow_1000000 [scan_count] | scan_count | 0.000066 | 392.3 | fitsio:fitsio_torch | 0.000062 | 392.3 | 1.061 | 6.06 | off | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| narrow_10000 [scan_count] | scan_count | 0.000067 | 369.4 | fitsio:fitsio_torch | 0.000063 | 369.4 | 1.060 | 5.99 | off | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| varlen_10000 [scan_count] | scan_count | 0.000065 | 375.9 | fitsio:fitsio_torch | 0.000062 | 375.9 | 1.042 | 4.17 | off | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| ascii_1000 [scan_count] | scan_count | 0.000065 | 411.1 | fitsio:fitsio_torch | 0.000064 | 411.1 | 1.013 | 1.34 | off | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| typed_100000 [scan_count] | scan_count | 0.000067 | 411.1 | fitsio:fitsio_torch | 0.000066 | 411.1 | 1.012 | 1.23 | off | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| varlen_100000 [scan_count] | scan_count | 0.000063 | 411.8 | fitsio:fitsio_torch | 0.000062 | 411.8 | 1.011 | 1.15 | off | torchfits-gpu-exhaustive-cpu-20260719-083800 |

### FITSTABLE - specialized

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| narrow_1000000 [predicate_filter] | predicate_filter | 0.009512 | 392.3 | astropy:astropy | 0.008350 | 392.3 | 1.139 | 13.92 | off | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| ascii_10000 [scan_count] | scan_count | 0.000068 | 411.1 | fitsio:fitsio | 0.000060 | 411.1 | 1.131 | 13.07 | off | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| varlen_100000 [scan_count] | scan_count | 0.000066 | 411.8 | fitsio:fitsio | 0.000063 | 411.8 | 1.049 | 4.89 | off | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| typed_10000 [scan_count] | scan_count | 0.000064 | 411.8 | fitsio:fitsio | 0.000063 | 411.8 | 1.025 | 2.51 | off | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| ascii_1000 [scan_count] | scan_count | 0.000064 | 411.1 | fitsio:fitsio | 0.000063 | 411.1 | 1.024 | 2.42 | off | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| mixed_1000 [scan_count] | scan_count | 0.000075 | 368.8 | fitsio:fitsio | 0.000074 | 368.8 | 1.020 | 2.00 | off | torchfits-gpu-exhaustive-cpu-20260719-083800 |
| varlen_1000 [scan_count] | scan_count | 0.000059 | 375.9 | fitsio:fitsio | 0.000059 | 375.9 | 1.001 | 0.15 | off | torchfits-gpu-exhaustive-cpu-20260719-083800 |

## Notes

- Strict mmap fairness is enforced in comparable sets. Rows with unmatched mmap controls are marked `SKIPPED`.
- Rankings are family-specific and never mix smart vs specialized method families.
