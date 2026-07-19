# Benchmark Summary

- Run ID: `exhaustive_cuda_20260719_144457`
- Scopes: `fits, fitstable`
- Total normalized rows: `4087`
- TorchFits deficit rows (all lags): `102`
- TorchFits significant deficits: `3`
- Hostname: `torchfits-gpu-exhaustive-cuda-20260719-144457`
- CPU count: `96`
- torch.get_num_threads(): `8`
- Peak RSS (median across timed rows): `729.8 MB` (max `958.0 MB`)

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
| fitstable | smart | 163/164 | 99.4% | 0 |
| fitstable | specialized | 178/180 | 98.9% | 0 |

- TorchFits devices observed in this run: `cpu, cuda`
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
| narrow_1000000 [read_full] | 1000000 | 1.068 | 6.82 |
| narrow_1000000 [predicate_filter] | 1000000 | 1.175 | 17.47 |
| narrow_1000000 [predicate_filter] | 1000000 | 1.055 | 5.46 |

### Small-N Visible Deficits

No small-N visible deficits detected.

## TorchFits Deficits (Not First)

### FITS - smart

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| small_int8_1d [read_full @ cuda] | read_full | 0.000102 | 772.3 | fitsio:fitsio_torch_device | 0.000097 | 772.3 | 1.046 | 4.59 | off | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| small_float32_1d [read_full @ cuda] | read_full | 0.000114 | 772.3 | fitsio:fitsio_torch_device | 0.000109 | 772.3 | 1.043 | 4.31 | off | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| compressed_hcompress_1 [read_full @ cuda] | read_full | 0.030502 | 740.4 | fitsio:fitsio_torch_device | 0.029290 | 740.4 | 1.041 | 4.14 | off | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| compressed_hcompress_1 [read_full @ cuda] | read_full | 0.030527 | 712.7 | fitsio:fitsio_torch_device | 0.029321 | 712.7 | 1.041 | 4.11 | on | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_int8_3d [read_full @ cuda] | read_full | 0.000099 | 772.3 | fitsio:fitsio_torch_device | 0.000096 | 772.3 | 1.032 | 3.24 | off | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| compressed_hcompress_1 [read_full] | read_full | 0.030377 | 619.3 | fitsio:fitsio_torch | 0.029457 | 619.3 | 1.031 | 3.12 | on | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| compressed_hcompress_1 [read_full] | read_full | 0.030120 | 779.6 | fitsio:fitsio_torch | 0.029229 | 779.6 | 1.031 | 3.05 | off | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_int16_1d [read_full @ cuda] | read_full | 0.000099 | 772.3 | fitsio:fitsio_torch_device | 0.000097 | 772.3 | 1.025 | 2.53 | off | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_float64_1d [read_full @ cuda] | read_full | 0.000103 | 772.3 | fitsio:fitsio_torch_device | 0.000101 | 772.3 | 1.022 | 2.15 | off | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_int32_1d [read_full @ cuda] | read_full | 0.000099 | 772.3 | fitsio:fitsio_torch_device | 0.000098 | 772.3 | 1.013 | 1.34 | off | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| small_int16_1d [read_full @ cuda] | read_full | 0.000100 | 772.3 | fitsio:fitsio_torch_device | 0.000099 | 772.3 | 1.010 | 0.98 | off | torchfits-gpu-exhaustive-cuda-20260719-144457 |

### FITS - specialized

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| small_float64_1d [header_read] | header_read | 0.000062 | 779.6 | fitsio:fitsio | 0.000037 | 779.6 | 1.663 | 66.34 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| medium_int64_1d [header_read] | header_read | 0.000061 | 779.6 | fitsio:fitsio | 0.000037 | 779.6 | 1.647 | 64.72 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_float64_1d [header_read] | header_read | 0.000060 | 779.6 | fitsio:fitsio | 0.000038 | 779.6 | 1.583 | 58.27 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| timeseries_frame_001 [header_read] | header_read | 0.000062 | 779.6 | fitsio:fitsio | 0.000040 | 779.6 | 1.569 | 56.88 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_float32_2d [header_read] | header_read | 0.000062 | 779.6 | fitsio:fitsio | 0.000039 | 779.6 | 1.566 | 56.65 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_int16_2d [header_read] | header_read | 0.000061 | 779.6 | fitsio:fitsio | 0.000040 | 779.6 | 1.534 | 53.37 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| small_int32_2d [header_read] | header_read | 0.000064 | 779.6 | fitsio:fitsio | 0.000042 | 779.6 | 1.531 | 53.13 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| small_float32_1d [header_read] | header_read | 0.000060 | 779.6 | fitsio:fitsio | 0.000039 | 779.6 | 1.530 | 53.01 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| medium_int8_1d [header_read] | header_read | 0.000065 | 779.6 | fitsio:fitsio | 0.000043 | 779.6 | 1.490 | 48.96 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| medium_float64_1d [header_read] | header_read | 0.000062 | 779.6 | fitsio:fitsio | 0.000042 | 779.6 | 1.487 | 48.72 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| large_float32_1d [header_read] | header_read | 0.000061 | 779.6 | fitsio:fitsio | 0.000041 | 779.6 | 1.487 | 48.66 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| medium_uint16_2d [header_read] | header_read | 0.000069 | 779.6 | fitsio:fitsio | 0.000047 | 779.6 | 1.470 | 46.95 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| timeseries_frame_000 [header_read] | header_read | 0.000064 | 779.6 | fitsio:fitsio | 0.000043 | 779.6 | 1.467 | 46.70 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| large_int16_1d [header_read] | header_read | 0.000061 | 779.6 | fitsio:fitsio | 0.000042 | 779.6 | 1.465 | 46.55 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| medium_float64_2d [header_read] | header_read | 0.000064 | 779.6 | fitsio:fitsio | 0.000044 | 779.6 | 1.464 | 46.39 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| small_float64_2d [header_read] | header_read | 0.000062 | 779.6 | fitsio:fitsio | 0.000042 | 779.6 | 1.456 | 45.62 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| large_int32_1d [header_read] | header_read | 0.000060 | 779.6 | fitsio:fitsio | 0.000042 | 779.6 | 1.454 | 45.38 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_int8_1d [header_read] | header_read | 0.000063 | 779.6 | fitsio:fitsio | 0.000044 | 779.6 | 1.450 | 44.95 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| medium_int64_2d [header_read] | header_read | 0.000060 | 779.6 | fitsio:fitsio | 0.000042 | 779.6 | 1.446 | 44.62 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| large_float64_1d [header_read] | header_read | 0.000062 | 779.6 | fitsio:fitsio | 0.000043 | 779.6 | 1.442 | 44.16 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| timeseries_frame_002 [header_read] | header_read | 0.000062 | 779.6 | fitsio:fitsio | 0.000043 | 779.6 | 1.440 | 44.00 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_float64_2d [header_read] | header_read | 0.000059 | 779.6 | fitsio:fitsio | 0.000041 | 779.6 | 1.437 | 43.66 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| medium_int32_2d [header_read] | header_read | 0.000062 | 779.6 | fitsio:fitsio | 0.000043 | 779.6 | 1.435 | 43.52 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| small_uint16_2d [header_read] | header_read | 0.000066 | 779.6 | fitsio:fitsio | 0.000046 | 779.6 | 1.426 | 42.58 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| large_uint32_2d [header_read] | header_read | 0.000074 | 779.6 | fitsio:fitsio | 0.000052 | 779.6 | 1.419 | 41.91 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| medium_uint32_2d [header_read] | header_read | 0.000069 | 779.6 | fitsio:fitsio | 0.000048 | 779.6 | 1.419 | 41.91 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| multi_mef_10ext [header_read] | header_read | 0.000082 | 779.6 | fitsio:fitsio | 0.000058 | 779.6 | 1.412 | 41.16 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| large_int16_2d [header_read] | header_read | 0.000063 | 779.6 | fitsio:fitsio | 0.000045 | 779.6 | 1.409 | 40.94 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_int64_3d [header_read] | header_read | 0.000060 | 779.6 | fitsio:fitsio | 0.000043 | 779.6 | 1.404 | 40.44 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| small_int8_1d [header_read] | header_read | 0.000069 | 779.6 | fitsio:fitsio | 0.000049 | 779.6 | 1.403 | 40.29 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| small_int32_3d [header_read] | header_read | 0.000063 | 779.6 | fitsio:fitsio | 0.000045 | 779.6 | 1.403 | 40.28 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| large_int32_2d [header_read] | header_read | 0.000063 | 779.6 | fitsio:fitsio | 0.000045 | 779.6 | 1.398 | 39.76 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_float32_1d [header_read] | header_read | 0.000058 | 779.6 | fitsio:fitsio | 0.000041 | 779.6 | 1.396 | 39.63 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| medium_int16_1d [header_read] | header_read | 0.000059 | 779.6 | fitsio:fitsio | 0.000042 | 779.6 | 1.393 | 39.27 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_float64_3d [header_read] | header_read | 0.000063 | 779.6 | fitsio:fitsio | 0.000046 | 779.6 | 1.393 | 39.25 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| medium_int16_3d [header_read] | header_read | 0.000064 | 779.6 | fitsio:fitsio | 0.000046 | 779.6 | 1.391 | 39.08 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_int64_1d [header_read] | header_read | 0.000057 | 779.6 | fitsio:fitsio | 0.000041 | 779.6 | 1.390 | 39.03 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| medium_int16_2d [header_read] | header_read | 0.000061 | 779.6 | fitsio:fitsio | 0.000044 | 779.6 | 1.386 | 38.58 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| medium_int32_1d [header_read] | header_read | 0.000059 | 779.6 | fitsio:fitsio | 0.000043 | 779.6 | 1.385 | 38.50 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| small_float32_2d [header_read] | header_read | 0.000062 | 779.6 | fitsio:fitsio | 0.000045 | 779.6 | 1.382 | 38.24 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| large_int64_1d [header_read] | header_read | 0.000060 | 779.6 | fitsio:fitsio | 0.000043 | 779.6 | 1.382 | 38.19 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| small_int16_2d [header_read] | header_read | 0.000061 | 779.6 | fitsio:fitsio | 0.000044 | 779.6 | 1.382 | 38.15 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| large_float64_2d [header_read] | header_read | 0.000062 | 779.6 | fitsio:fitsio | 0.000045 | 779.6 | 1.376 | 37.61 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_int32_1d [header_read] | header_read | 0.000058 | 779.6 | fitsio:fitsio | 0.000042 | 779.6 | 1.375 | 37.51 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| medium_int32_3d [header_read] | header_read | 0.000067 | 779.6 | fitsio:fitsio | 0.000049 | 779.6 | 1.374 | 37.42 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| medium_float32_1d [header_read] | header_read | 0.000060 | 779.6 | fitsio:fitsio | 0.000044 | 779.6 | 1.367 | 36.70 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| large_int64_2d [header_read] | header_read | 0.000062 | 779.6 | fitsio:fitsio | 0.000046 | 779.6 | 1.367 | 36.69 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_int16_3d [header_read] | header_read | 0.000063 | 779.6 | fitsio:fitsio | 0.000046 | 779.6 | 1.357 | 35.74 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| small_int16_1d [header_read] | header_read | 0.000055 | 779.6 | fitsio:fitsio | 0.000041 | 779.6 | 1.356 | 35.64 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_int32_2d [header_read] | header_read | 0.000061 | 779.6 | fitsio:fitsio | 0.000045 | 779.6 | 1.356 | 35.57 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_int8_3d [header_read] | header_read | 0.000072 | 779.6 | fitsio:fitsio | 0.000053 | 779.6 | 1.351 | 35.05 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_float32_3d [header_read] | header_read | 0.000063 | 779.6 | fitsio:fitsio | 0.000047 | 779.6 | 1.348 | 34.76 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_int64_2d [header_read] | header_read | 0.000059 | 779.6 | fitsio:fitsio | 0.000044 | 779.6 | 1.345 | 34.48 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| large_float32_2d [header_read] | header_read | 0.000061 | 779.6 | fitsio:fitsio | 0.000045 | 779.6 | 1.344 | 34.42 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| timeseries_frame_003 [header_read] | header_read | 0.000059 | 779.6 | fitsio:fitsio | 0.000044 | 779.6 | 1.342 | 34.25 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| medium_int8_2d [header_read] | header_read | 0.000071 | 779.6 | fitsio:fitsio | 0.000054 | 779.6 | 1.333 | 33.32 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| timeseries_frame_004 [header_read] | header_read | 0.000061 | 779.6 | fitsio:fitsio | 0.000046 | 779.6 | 1.330 | 32.96 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| small_int64_1d [header_read] | header_read | 0.000056 | 779.6 | fitsio:fitsio | 0.000042 | 779.6 | 1.329 | 32.93 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| small_int8_2d [header_read] | header_read | 0.000066 | 779.6 | fitsio:fitsio | 0.000050 | 779.6 | 1.321 | 32.13 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| medium_int8_3d [header_read] | header_read | 0.000068 | 779.6 | fitsio:fitsio | 0.000052 | 779.6 | 1.320 | 31.97 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| small_int64_3d [header_read] | header_read | 0.000062 | 779.6 | fitsio:fitsio | 0.000047 | 779.6 | 1.318 | 31.84 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_int16_1d [header_read] | header_read | 0.000060 | 779.6 | fitsio:fitsio | 0.000046 | 779.6 | 1.315 | 31.50 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| medium_float64_3d [header_read] | header_read | 0.000063 | 779.6 | fitsio:fitsio | 0.000048 | 779.6 | 1.312 | 31.21 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_int32_3d [header_read] | header_read | 0.000060 | 779.6 | fitsio:fitsio | 0.000046 | 779.6 | 1.306 | 30.61 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| medium_float32_3d [header_read] | header_read | 0.000064 | 779.6 | fitsio:fitsio | 0.000049 | 779.6 | 1.305 | 30.54 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| small_int64_2d [header_read] | header_read | 0.000060 | 779.6 | fitsio:fitsio | 0.000046 | 779.6 | 1.304 | 30.37 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| large_uint16_2d [header_read] | header_read | 0.000067 | 779.6 | fitsio:fitsio | 0.000052 | 779.6 | 1.300 | 29.99 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| mef_small [header_read] | header_read | 0.000083 | 779.6 | fitsio:fitsio | 0.000064 | 779.6 | 1.298 | 29.79 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| small_int32_1d [header_read] | header_read | 0.000056 | 779.6 | fitsio:fitsio | 0.000043 | 779.6 | 1.290 | 28.96 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| medium_float32_2d [header_read] | header_read | 0.000060 | 779.6 | fitsio:fitsio | 0.000047 | 779.6 | 1.289 | 28.89 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| large_int8_1d [header_read] | header_read | 0.000066 | 779.6 | fitsio:fitsio | 0.000052 | 779.6 | 1.272 | 27.25 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| scaled_small [header_read] | header_read | 0.000067 | 779.6 | fitsio:fitsio | 0.000053 | 779.6 | 1.270 | 26.98 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| scaled_large [header_read] | header_read | 0.000065 | 779.6 | fitsio:fitsio | 0.000052 | 779.6 | 1.264 | 26.38 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| small_int8_3d [header_read] | header_read | 0.000070 | 779.6 | fitsio:fitsio | 0.000055 | 779.6 | 1.260 | 26.01 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| small_int16_3d [header_read] | header_read | 0.000063 | 779.6 | fitsio:fitsio | 0.000051 | 779.6 | 1.249 | 24.94 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| tiny_int8_2d [header_read] | header_read | 0.000066 | 779.6 | fitsio:fitsio | 0.000053 | 779.6 | 1.244 | 24.44 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| large_int8_2d [header_read] | header_read | 0.000067 | 779.6 | fitsio:fitsio | 0.000054 | 779.6 | 1.236 | 23.59 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| scaled_medium [header_read] | header_read | 0.000063 | 779.6 | fitsio:fitsio | 0.000051 | 779.6 | 1.232 | 23.23 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| medium_int64_3d [header_read] | header_read | 0.000059 | 779.6 | fitsio:fitsio | 0.000048 | 779.6 | 1.225 | 22.54 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| small_uint32_2d [header_read] | header_read | 0.000064 | 779.6 | fitsio:fitsio | 0.000053 | 779.6 | 1.215 | 21.52 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| mef_medium [header_read] | header_read | 0.000075 | 779.6 | fitsio:fitsio | 0.000063 | 779.6 | 1.192 | 19.23 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| small_float32_3d [header_read] | header_read | 0.000057 | 779.6 | fitsio:fitsio | 0.000048 | 779.6 | 1.178 | 17.75 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| small_float64_3d [header_read] | header_read | 0.000057 | 779.6 | fitsio:fitsio | 0.000048 | 779.6 | 1.177 | 17.70 | n/a | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| medium_int16_3d [read_full] | read_full | 0.000689 | 779.6 | fitsio:fitsio_torch | 0.000655 | 779.6 | 1.052 | 5.21 | off | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| compressed_hcompress_1 [read_full @ cuda] | read_full | 0.030536 | 740.4 | fitsio:fitsio_torch_device_specialized | 0.029322 | 740.4 | 1.041 | 4.14 | off | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| compressed_hcompress_1 [read_full @ cuda] | read_full | 0.030492 | 712.7 | fitsio:fitsio_torch_device_specialized | 0.029301 | 712.7 | 1.041 | 4.07 | on | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| compressed_hcompress_1 [read_full] | read_full | 0.030166 | 779.6 | fitsio:fitsio_torch | 0.029229 | 779.6 | 1.032 | 3.21 | off | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| compressed_hcompress_1 [read_full] | read_full | 0.030212 | 619.3 | fitsio:fitsio_torch | 0.029457 | 619.3 | 1.026 | 2.56 | on | torchfits-gpu-exhaustive-cuda-20260719-144457 |

### FITSTABLE - smart

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| narrow_1000000 [read_full] | read_full | 0.007712 | 717.6 | fitsio:fitsio_torch | 0.007220 | 717.6 | 1.068 | 6.82 | off | torchfits-gpu-exhaustive-cuda-20260719-144457 |

### FITSTABLE - specialized

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| narrow_1000000 [predicate_filter] | predicate_filter | 0.011587 | 733.3 | astropy:astropy | 0.009864 | 733.3 | 1.175 | 17.47 | on | torchfits-gpu-exhaustive-cuda-20260719-144457 |
| narrow_1000000 [predicate_filter] | predicate_filter | 0.011679 | 731.0 | astropy:astropy | 0.011074 | 718.6 | 1.055 | 5.46 | off | torchfits-gpu-exhaustive-cuda-20260719-144457 |

## Notes

- Strict mmap fairness is enforced in comparable sets. Rows with unmatched mmap controls are marked `SKIPPED`.
- Rankings are family-specific and never mix smart vs specialized method families.
