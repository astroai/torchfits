# Benchmark Summary

- Run ID: `exhaustive_cpu_20260719_144337`
- Scopes: `fits, fitstable`
- Total normalized rows: `2829`
- TorchFits deficit rows (all lags): `92`
- TorchFits significant deficits: `3`
- Hostname: `torchfits-gpu-exhaustive-cpu-20260719-144337`
- CPU count: `192`
- torch.get_num_threads(): `8`
- Peak RSS (median across timed rows): `289.4 MB` (max `612.3 MB`)

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
| fitstable | smart | 163/164 | 99.4% | 0 |
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
| fitstable | specialized | 68/70 | 97.1% |

Large-N deficits detected:

| Case | n_points | Lag (x) | Behind (%) |
|---|---:|---:|---:|
| narrow_100000 [read_full] | 100000 | 1.062 | 6.15 |
| narrow_1000000 [predicate_filter] | 1000000 | 1.283 | 28.32 |
| narrow_1000000 [predicate_filter] | 1000000 | 1.216 | 21.59 |

### Small-N Visible Deficits

No small-N visible deficits detected.

## TorchFits Deficits (Not First)

### FITS - smart

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| compressed_hcompress_1 [read_full] | read_full | 0.026283 | 290.7 | fitsio:fitsio_torch | 0.025505 | 290.7 | 1.031 | 3.05 | off | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| compressed_hcompress_1 [read_full] | read_full | 0.045488 | 289.4 | fitsio:fitsio_torch | 0.044232 | 289.4 | 1.028 | 2.84 | on | torchfits-gpu-exhaustive-cpu-20260719-144337 |

### FITS - specialized

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| medium_int8_1d [header_read] | header_read | 0.000043 | 278.1 | fitsio:fitsio | 0.000029 | 278.1 | 1.508 | 50.82 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| small_int32_1d [header_read] | header_read | 0.000037 | 278.1 | fitsio:fitsio | 0.000025 | 278.1 | 1.500 | 49.98 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| small_float32_1d [header_read] | header_read | 0.000036 | 278.1 | fitsio:fitsio | 0.000025 | 278.1 | 1.447 | 44.74 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| tiny_float32_1d [header_read] | header_read | 0.000036 | 278.2 | fitsio:fitsio | 0.000025 | 278.2 | 1.445 | 44.47 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| medium_int64_1d [header_read] | header_read | 0.000036 | 278.1 | fitsio:fitsio | 0.000025 | 278.1 | 1.444 | 44.40 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| large_int16_1d [header_read] | header_read | 0.000036 | 278.1 | fitsio:fitsio | 0.000025 | 278.1 | 1.440 | 44.00 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| small_float64_1d [header_read] | header_read | 0.000038 | 278.1 | fitsio:fitsio | 0.000026 | 278.1 | 1.438 | 43.79 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| tiny_int64_1d [header_read] | header_read | 0.000035 | 278.2 | fitsio:fitsio | 0.000024 | 278.2 | 1.429 | 42.87 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| large_float64_2d [header_read] | header_read | 0.000040 | 278.1 | fitsio:fitsio | 0.000028 | 278.1 | 1.425 | 42.54 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| timeseries_frame_000 [header_read] | header_read | 0.000050 | 278.2 | fitsio:fitsio | 0.000036 | 278.2 | 1.421 | 42.10 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| medium_int16_2d [header_read] | header_read | 0.000041 | 278.1 | fitsio:fitsio | 0.000029 | 278.1 | 1.413 | 41.26 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| large_float32_1d [header_read] | header_read | 0.000038 | 278.1 | fitsio:fitsio | 0.000028 | 278.1 | 1.399 | 39.87 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| tiny_float32_2d [header_read] | header_read | 0.000038 | 278.2 | fitsio:fitsio | 0.000027 | 278.2 | 1.389 | 38.86 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| large_float32_2d [header_read] | header_read | 0.000042 | 278.1 | fitsio:fitsio | 0.000030 | 278.1 | 1.387 | 38.71 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| medium_int32_3d [header_read] | header_read | 0.000043 | 278.1 | fitsio:fitsio | 0.000031 | 278.1 | 1.381 | 38.07 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| medium_int64_2d [header_read] | header_read | 0.000040 | 278.1 | fitsio:fitsio | 0.000029 | 278.1 | 1.377 | 37.75 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| timeseries_frame_001 [header_read] | header_read | 0.000038 | 278.2 | fitsio:fitsio | 0.000028 | 278.2 | 1.377 | 37.70 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| large_float64_1d [header_read] | header_read | 0.000037 | 278.1 | fitsio:fitsio | 0.000027 | 278.1 | 1.377 | 37.67 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| small_float32_3d [header_read] | header_read | 0.000039 | 278.1 | fitsio:fitsio | 0.000029 | 278.1 | 1.376 | 37.63 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| tiny_int32_1d [header_read] | header_read | 0.000037 | 278.2 | fitsio:fitsio | 0.000027 | 278.2 | 1.376 | 37.55 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| medium_float64_3d [header_read] | header_read | 0.000039 | 278.1 | fitsio:fitsio | 0.000028 | 278.1 | 1.374 | 37.36 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| small_int32_2d [header_read] | header_read | 0.000037 | 278.1 | fitsio:fitsio | 0.000027 | 278.1 | 1.366 | 36.63 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| small_float64_2d [header_read] | header_read | 0.000038 | 278.1 | fitsio:fitsio | 0.000028 | 278.1 | 1.362 | 36.18 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| small_float32_2d [header_read] | header_read | 0.000039 | 278.1 | fitsio:fitsio | 0.000029 | 278.1 | 1.352 | 35.18 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| tiny_float64_1d [header_read] | header_read | 0.000036 | 278.2 | fitsio:fitsio | 0.000026 | 278.2 | 1.351 | 35.13 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| large_int64_1d [header_read] | header_read | 0.000036 | 278.1 | fitsio:fitsio | 0.000027 | 278.1 | 1.350 | 34.99 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| small_int64_1d [header_read] | header_read | 0.000052 | 278.1 | fitsio:fitsio | 0.000038 | 278.1 | 1.346 | 34.55 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| medium_float64_1d [header_read] | header_read | 0.000036 | 278.1 | fitsio:fitsio | 0.000027 | 278.1 | 1.342 | 34.20 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| timeseries_frame_002 [header_read] | header_read | 0.000038 | 278.2 | fitsio:fitsio | 0.000029 | 278.2 | 1.342 | 34.17 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| medium_int16_3d [header_read] | header_read | 0.000040 | 278.1 | fitsio:fitsio | 0.000030 | 278.1 | 1.341 | 34.09 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| large_int8_1d [header_read] | header_read | 0.000041 | 278.1 | fitsio:fitsio | 0.000031 | 278.1 | 1.341 | 34.08 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| large_int32_1d [header_read] | header_read | 0.000036 | 278.1 | fitsio:fitsio | 0.000027 | 278.1 | 1.337 | 33.66 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| tiny_int16_2d [header_read] | header_read | 0.000038 | 278.2 | fitsio:fitsio | 0.000028 | 278.2 | 1.336 | 33.64 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| large_int64_2d [header_read] | header_read | 0.000037 | 278.1 | fitsio:fitsio | 0.000028 | 278.1 | 1.334 | 33.37 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| tiny_float64_2d [header_read] | header_read | 0.000038 | 278.2 | fitsio:fitsio | 0.000028 | 278.2 | 1.334 | 33.35 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| medium_float32_3d [header_read] | header_read | 0.000038 | 278.1 | fitsio:fitsio | 0.000028 | 278.1 | 1.331 | 33.06 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| large_int32_2d [header_read] | header_read | 0.000037 | 278.1 | fitsio:fitsio | 0.000028 | 278.1 | 1.326 | 32.62 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| medium_float64_2d [header_read] | header_read | 0.000042 | 278.1 | fitsio:fitsio | 0.000032 | 278.1 | 1.326 | 32.59 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| medium_int32_1d [header_read] | header_read | 0.000035 | 278.1 | fitsio:fitsio | 0.000027 | 278.1 | 1.323 | 32.28 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| timeseries_frame_004 [header_read] | header_read | 0.000036 | 278.2 | fitsio:fitsio | 0.000027 | 278.2 | 1.322 | 32.21 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| tiny_float32_3d [header_read] | header_read | 0.000039 | 278.2 | fitsio:fitsio | 0.000029 | 278.2 | 1.317 | 31.72 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| tiny_int64_2d [header_read] | header_read | 0.000036 | 278.2 | fitsio:fitsio | 0.000027 | 278.2 | 1.314 | 31.40 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| small_int32_3d [header_read] | header_read | 0.000039 | 278.1 | fitsio:fitsio | 0.000029 | 278.1 | 1.314 | 31.39 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| medium_int32_2d [header_read] | header_read | 0.000039 | 278.1 | fitsio:fitsio | 0.000030 | 278.1 | 1.313 | 31.33 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| tiny_int16_3d [header_read] | header_read | 0.000039 | 278.2 | fitsio:fitsio | 0.000029 | 278.2 | 1.313 | 31.27 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| timeseries_frame_003 [header_read] | header_read | 0.000039 | 278.2 | fitsio:fitsio | 0.000029 | 278.2 | 1.312 | 31.15 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| small_int16_1d [header_read] | header_read | 0.000035 | 278.1 | fitsio:fitsio | 0.000027 | 278.1 | 1.310 | 30.98 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| tiny_float64_3d [header_read] | header_read | 0.000039 | 278.2 | fitsio:fitsio | 0.000030 | 278.2 | 1.309 | 30.88 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| small_int64_2d [header_read] | header_read | 0.000038 | 278.1 | fitsio:fitsio | 0.000029 | 278.1 | 1.306 | 30.57 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| small_int16_2d [header_read] | header_read | 0.000036 | 278.1 | fitsio:fitsio | 0.000028 | 278.1 | 1.304 | 30.45 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| scaled_medium [header_read] | header_read | 0.000044 | 278.1 | fitsio:fitsio | 0.000034 | 278.1 | 1.303 | 30.31 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| tiny_int16_1d [header_read] | header_read | 0.000034 | 278.2 | fitsio:fitsio | 0.000026 | 278.2 | 1.303 | 30.29 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| tiny_int8_3d [header_read] | header_read | 0.000044 | 278.2 | fitsio:fitsio | 0.000034 | 278.2 | 1.301 | 30.12 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| large_int8_2d [header_read] | header_read | 0.000042 | 278.1 | fitsio:fitsio | 0.000032 | 278.1 | 1.297 | 29.72 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| large_int16_2d [header_read] | header_read | 0.000037 | 278.1 | fitsio:fitsio | 0.000029 | 278.1 | 1.291 | 29.11 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| medium_float32_2d [header_read] | header_read | 0.000037 | 278.1 | fitsio:fitsio | 0.000028 | 278.1 | 1.290 | 29.01 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| small_uint32_2d [header_read] | header_read | 0.000092 | 278.2 | fitsio:fitsio | 0.000072 | 278.2 | 1.290 | 28.99 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| medium_int16_1d [header_read] | header_read | 0.000034 | 278.1 | fitsio:fitsio | 0.000027 | 278.1 | 1.283 | 28.32 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| large_uint16_2d [header_read] | header_read | 0.000041 | 278.1 | fitsio:fitsio | 0.000032 | 278.1 | 1.282 | 28.20 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| tiny_int32_3d [header_read] | header_read | 0.000038 | 278.2 | fitsio:fitsio | 0.000030 | 278.2 | 1.273 | 27.28 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| large_uint32_2d [header_read] | header_read | 0.000042 | 278.1 | fitsio:fitsio | 0.000033 | 278.1 | 1.272 | 27.24 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| medium_int8_3d [header_read] | header_read | 0.000047 | 278.1 | fitsio:fitsio | 0.000037 | 278.1 | 1.263 | 26.29 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| medium_uint32_2d [header_read] | header_read | 0.000044 | 278.1 | fitsio:fitsio | 0.000035 | 278.1 | 1.261 | 26.08 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| small_int8_3d [header_read] | header_read | 0.000045 | 278.1 | fitsio:fitsio | 0.000036 | 278.1 | 1.259 | 25.91 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| scaled_large [header_read] | header_read | 0.000043 | 278.1 | fitsio:fitsio | 0.000034 | 278.1 | 1.259 | 25.90 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| small_int8_2d [header_read] | header_read | 0.000041 | 278.1 | fitsio:fitsio | 0.000033 | 278.1 | 1.252 | 25.18 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| small_uint16_2d [header_read] | header_read | 0.000041 | 278.1 | fitsio:fitsio | 0.000033 | 278.2 | 1.252 | 25.17 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| small_int8_1d [header_read] | header_read | 0.000040 | 278.1 | fitsio:fitsio | 0.000032 | 278.1 | 1.249 | 24.94 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| scaled_small [header_read] | header_read | 0.000043 | 278.1 | fitsio:fitsio | 0.000034 | 278.1 | 1.244 | 24.44 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| small_int16_3d [header_read] | header_read | 0.000039 | 278.1 | fitsio:fitsio | 0.000031 | 278.1 | 1.240 | 23.96 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| medium_int64_3d [header_read] | header_read | 0.000037 | 278.1 | fitsio:fitsio | 0.000030 | 278.1 | 1.238 | 23.81 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| mef_small [header_read] | header_read | 0.000051 | 278.1 | fitsio:fitsio | 0.000042 | 278.1 | 1.230 | 23.03 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| medium_float32_1d [header_read] | header_read | 0.000032 | 278.1 | fitsio:fitsio | 0.000026 | 278.1 | 1.222 | 22.24 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| tiny_int32_2d [header_read] | header_read | 0.000036 | 278.2 | fitsio:fitsio | 0.000029 | 278.2 | 1.220 | 22.00 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| medium_uint16_2d [header_read] | header_read | 0.000041 | 278.1 | fitsio:fitsio | 0.000034 | 278.1 | 1.217 | 21.69 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| medium_int8_2d [header_read] | header_read | 0.000041 | 278.1 | fitsio:fitsio | 0.000034 | 278.1 | 1.214 | 21.41 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| small_float64_3d [header_read] | header_read | 0.000038 | 278.1 | fitsio:fitsio | 0.000032 | 278.1 | 1.199 | 19.85 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| tiny_int8_2d [header_read] | header_read | 0.000040 | 278.2 | fitsio:fitsio | 0.000034 | 278.2 | 1.182 | 18.24 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| tiny_int64_3d [header_read] | header_read | 0.000036 | 278.2 | fitsio:fitsio | 0.000031 | 278.2 | 1.177 | 17.66 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| small_int64_3d [header_read] | header_read | 0.000038 | 278.1 | fitsio:fitsio | 0.000033 | 278.1 | 1.147 | 14.74 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| mef_medium [header_read] | header_read | 0.000050 | 278.1 | fitsio:fitsio | 0.000043 | 278.1 | 1.145 | 14.52 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| multi_mef_10ext [header_read] | header_read | 0.000049 | 278.1 | fitsio:fitsio | 0.000044 | 278.1 | 1.118 | 11.76 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| tiny_int8_1d [header_read] | header_read | 0.000037 | 278.2 | fitsio:fitsio | 0.000034 | 278.2 | 1.106 | 10.61 | n/a | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| compressed_hcompress_1 [read_full] | read_full | 0.026297 | 290.7 | fitsio:fitsio_torch | 0.025505 | 290.7 | 1.031 | 3.11 | off | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| compressed_hcompress_1 [read_full] | read_full | 0.045498 | 289.4 | fitsio:fitsio_torch | 0.044232 | 289.4 | 1.029 | 2.86 | on | torchfits-gpu-exhaustive-cpu-20260719-144337 |

### FITSTABLE - smart

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| narrow_100000 [read_full] | read_full | 0.000620 | 366.0 | fitsio:fitsio_torch | 0.000584 | 366.0 | 1.062 | 6.15 | off | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| narrow_1000000 [read_full] | read_full | 0.005443 | 384.0 | fitsio:fitsio_torch | 0.005233 | 384.0 | 1.040 | 4.00 | off | torchfits-gpu-exhaustive-cpu-20260719-144337 |

### FITSTABLE - specialized

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| narrow_1000000 [predicate_filter] | predicate_filter | 0.010053 | 390.6 | astropy:astropy | 0.007834 | 390.6 | 1.283 | 28.32 | on | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| narrow_1000000 [predicate_filter] | predicate_filter | 0.010117 | 388.1 | astropy:astropy | 0.008320 | 388.1 | 1.216 | 21.59 | off | torchfits-gpu-exhaustive-cpu-20260719-144337 |
| mixed_1000000 [predicate_filter] | predicate_filter | 0.012431 | 425.0 | astropy:astropy | 0.011848 | 425.0 | 1.049 | 4.92 | on | torchfits-gpu-exhaustive-cpu-20260719-144337 |

## Notes

- Strict mmap fairness is enforced in comparable sets. Rows with unmatched mmap controls are marked `SKIPPED`.
- Rankings are family-specific and never mix smart vs specialized method families.
