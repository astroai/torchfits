# Benchmark Summary

- Run ID: `exhaustive_mps_20260719_065105`
- Scopes: `fits, fitstable`
- Total normalized rows: `3931`
- TorchFits deficit rows (all lags): `123`
- TorchFits significant deficits: `21`
- Hostname: `NRC-054711`
- CPU count: `8`
- torch.get_num_threads(): `4`
- Peak RSS (median across timed rows): `802.4 MB` (max `1264.9 MB`)

## Domain Coverage

| Domain | Rows | Skipped |
|---|---:|---:|
| fits | 2789 | 85 |
| fitstable | 1142 | 76 |

## Astronomer Scorecard

| Domain | Family | TorchFits First | Win Rate | Legacy In Ranking |
|---|---|---:|---:|---:|
| fits | smart | 307/311 | 98.7% | 0 |
| fits | specialized | 394/397 | 99.2% | 0 |
| fitstable | smart | 158/164 | 96.3% | 0 |
| fitstable | specialized | 172/180 | 95.6% | 0 |

- TorchFits devices observed in this run: `cpu, mps`
- Smart-family tables are the primary adoption view for astronomers (performance + portability).

## Adoption Checks

- `large-N` threshold: `n_points >= 100000`
- `small-N perceived` threshold: `torchfits_time_s < 0.000500s`
- `small-N max lag` threshold: `lag_ratio < 10.0x`

### Large-N Leadership

| Domain | Family | TorchFits First (large-N) | Win Rate |
|---|---|---:|---:|
| fitstable | smart | 61/63 | 96.8% |
| fitstable | specialized | 66/70 | 94.3% |

Large-N deficits detected:

| Case | n_points | Lag (x) | Behind (%) |
|---|---:|---:|---:|
| mixed_1000000 [scan_count] | 1000000 | 3.957 | 295.68 |
| varlen_100000 [scan_count] | 100000 | 1.234 | 23.39 |
| narrow_100000 [scan_count] | 100000 | 2.647 | 164.66 |
| mixed_1000000 [scan_count] | 1000000 | 1.418 | 41.83 |
| narrow_1000000 [predicate_filter] | 1000000 | 1.393 | 39.32 |
| mixed_1000000 [predicate_filter] | 1000000 | 1.231 | 23.10 |

### Small-N Visible Deficits

| Case | TorchFits (s) | Lag (x) | Behind (%) | Impact |
|---|---:|---:|---:|---|
| small_uint16_2d [read_full @ mps] | 0.000567 | 1.593 | 59.33 | visible |
| large_int16_1d [read_full @ mps] | 0.000826 | 1.383 | 38.29 | visible |
| large_int64_1d [read_full @ mps] | 0.002197 | 1.147 | 14.72 | visible |
| compressed_rice_1 [read_full] | 0.023684 | 1.076 | 7.56 | visible |
| compressed_rice_1 [read_full] | 0.025296 | 1.149 | 14.88 | visible |
| compressed_rice_1 [read_full @ mps] | 0.008067 | 1.026 | 2.63 | visible |
| ascii_10000 [predicate_filter] | 0.000624 | 1.387 | 38.69 | visible |

## TorchFits Deficits (Not First)

### FITS - smart

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| small_int32_1d [read_full] | read_full | 0.000177 | 814.8 | fitsio:fitsio_torch | 0.000093 | 814.8 | 1.915 | 91.54 | off | NRC-054711 |
| small_uint16_2d [read_full @ mps] | read_full | 0.000567 | 819.0 | fitsio:fitsio_torch_device | 0.000356 | 819.0 | 1.593 | 59.33 | off | NRC-054711 |
| tiny_int64_3d [read_full @ mps] | read_full | 0.000375 | 819.0 | fitsio:fitsio_torch_device | 0.000245 | 819.0 | 1.529 | 52.90 | off | NRC-054711 |
| small_int16_2d [read_full @ mps] | read_full | 0.000381 | 819.0 | fitsio:fitsio_torch_device | 0.000259 | 819.0 | 1.473 | 47.30 | off | NRC-054711 |
| large_int16_1d [read_full @ mps] | read_full | 0.000826 | 817.0 | fitsio:fitsio_torch_device | 0.000597 | 817.0 | 1.383 | 38.29 | off | NRC-054711 |
| small_int32_1d [read_full @ mps] | read_full | 0.000318 | 819.0 | fitsio:fitsio_torch_device | 0.000231 | 819.0 | 1.375 | 37.51 | off | NRC-054711 |
| multi_mef_10ext [cutout_100x100] | cutout_100x100 | 0.000277 | 814.8 | fitsio:fitsio_torch | 0.000203 | 814.8 | 1.362 | 36.23 | n/a | NRC-054711 |
| medium_int16_3d [read_full @ mps] | read_full | 0.000929 | 817.0 | fitsio:fitsio_torch_device | 0.000778 | 817.0 | 1.193 | 19.34 | off | NRC-054711 |
| large_int64_1d [read_full @ mps] | read_full | 0.002197 | 817.0 | fitsio:fitsio_torch_device | 0.001916 | 817.0 | 1.147 | 14.72 | off | NRC-054711 |
| timeseries_frame_001 [read_full @ mps] | read_full | 0.000324 | 819.0 | fitsio:fitsio_torch_device | 0.000284 | 819.0 | 1.139 | 13.87 | off | NRC-054711 |
| small_uint32_2d [read_full @ mps] | read_full | 0.000363 | 819.0 | fitsio:fitsio_torch_device | 0.000324 | 819.0 | 1.120 | 12.04 | off | NRC-054711 |
| medium_float32_1d [read_full @ mps] | read_full | 0.000359 | 817.0 | fitsio:fitsio_torch_device | 0.000320 | 817.0 | 1.119 | 11.91 | off | NRC-054711 |
| mef_small [read_full] | read_full | 0.000311 | 814.8 | fitsio:fitsio_torch | 0.000279 | 814.8 | 1.114 | 11.40 | off | NRC-054711 |
| small_float32_1d [read_full] | read_full | 0.000098 | 814.8 | fitsio:fitsio_torch | 0.000090 | 814.8 | 1.091 | 9.12 | off | NRC-054711 |
| medium_int32_3d [read_full @ mps] | read_full | 0.001737 | 817.0 | fitsio:fitsio_torch_device | 0.001593 | 817.0 | 1.090 | 9.00 | off | NRC-054711 |
| tiny_float32_1d [read_full @ mps] | read_full | 0.000358 | 819.0 | fitsio:fitsio_torch_device | 0.000330 | 819.0 | 1.084 | 8.37 | off | NRC-054711 |
| compressed_rice_1 [read_full] | read_full | 0.023684 | 487.4 | fitsio:fitsio_torch | 0.022019 | 487.4 | 1.076 | 7.56 | on | NRC-054711 |
| scaled_medium [read_full @ mps] | read_full | 0.001116 | 819.0 | fitsio:fitsio_torch_device | 0.001041 | 819.0 | 1.072 | 7.20 | off | NRC-054711 |
| small_int8_1d [read_full @ mps] | read_full | 0.000290 | 819.0 | fitsio:fitsio_torch_device | 0.000273 | 819.0 | 1.063 | 6.32 | off | NRC-054711 |
| tiny_float64_3d [read_full] | read_full | 0.000091 | 814.8 | fitsio:fitsio_torch | 0.000086 | 814.8 | 1.060 | 6.05 | off | NRC-054711 |
| scaled_small [read_full @ mps] | read_full | 0.000280 | 819.0 | fitsio:fitsio_torch_device | 0.000264 | 819.0 | 1.058 | 5.84 | off | NRC-054711 |
| large_float32_1d [read_full @ mps] | read_full | 0.001054 | 817.0 | fitsio:fitsio_torch_device | 0.000997 | 817.0 | 1.057 | 5.74 | off | NRC-054711 |
| small_int64_2d [read_full @ mps] | read_full | 0.000386 | 819.0 | fitsio:fitsio_torch_device | 0.000365 | 819.0 | 1.056 | 5.63 | off | NRC-054711 |
| medium_int64_1d [read_full @ mps] | read_full | 0.000427 | 817.5 | fitsio:fitsio_torch_device | 0.000406 | 817.5 | 1.053 | 5.27 | off | NRC-054711 |
| small_int64_3d [read_full @ mps] | read_full | 0.000504 | 819.0 | fitsio:fitsio_torch_device | 0.000481 | 819.0 | 1.046 | 4.58 | off | NRC-054711 |
| large_uint16_2d [read_full @ mps] | read_full | 0.003996 | 817.0 | fitsio:fitsio_torch_device | 0.003827 | 817.0 | 1.044 | 4.42 | off | NRC-054711 |
| medium_int8_1d [read_full @ mps] | read_full | 0.000346 | 817.5 | fitsio:fitsio_torch_device | 0.000333 | 817.5 | 1.039 | 3.88 | off | NRC-054711 |
| small_int32_2d [read_full @ mps] | read_full | 0.000292 | 819.0 | fitsio:fitsio_torch_device | 0.000282 | 819.0 | 1.036 | 3.60 | off | NRC-054711 |
| compressed_hcompress_1 [read_full] | read_full | 0.051359 | 475.2 | fitsio:fitsio_torch | 0.050202 | 475.2 | 1.023 | 2.30 | on | NRC-054711 |
| medium_float32_2d [read_full @ mps] | read_full | 0.000999 | 817.0 | fitsio:fitsio_torch_device | 0.000979 | 817.0 | 1.020 | 2.03 | off | NRC-054711 |
| repeated_cutouts_50x_100x100 [repeated_cutouts_50x_100x100] | repeated_cutouts_50x_100x100 | 0.007253 | 814.8 | fitsio:fitsio_torch | 0.007165 | 814.8 | 1.012 | 1.23 | n/a | NRC-054711 |
| compressed_hcompress_1 [read_full @ mps] | read_full | 0.025326 | 817.0 | fitsio:fitsio_torch_device | 0.025096 | 817.0 | 1.009 | 0.92 | off | NRC-054711 |
| small_float32_1d [read_full @ mps] | read_full | 0.000222 | 819.0 | fitsio:fitsio_torch_device | 0.000221 | 819.0 | 1.005 | 0.55 | off | NRC-054711 |
| compressed_rice_1 [read_full @ mps] | read_full | 0.008316 | 817.0 | fitsio:fitsio_torch_device | 0.008273 | 817.0 | 1.005 | 0.53 | off | NRC-054711 |
| medium_uint16_2d [read_full @ mps] | read_full | 0.001171 | 819.0 | fitsio:fitsio_torch_device | 0.001167 | 819.0 | 1.003 | 0.32 | off | NRC-054711 |
| small_int16_3d [read_full @ mps] | read_full | 0.000273 | 819.0 | fitsio:fitsio_torch_device | 0.000272 | 819.0 | 1.003 | 0.29 | off | NRC-054711 |

### FITS - specialized

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| small_float64_1d [read_full] | read_full | 0.000201 | 814.8 | fitsio:fitsio_torch | 0.000097 | 814.8 | 2.080 | 108.05 | off | NRC-054711 |
| small_float32_1d [read_full] | read_full | 0.000163 | 814.8 | fitsio:fitsio_torch | 0.000090 | 814.8 | 1.820 | 82.04 | off | NRC-054711 |
| small_float32_1d [read_full @ mps] | read_full | 0.000484 | 819.0 | fitsio:fitsio_torch_device_specialized | 0.000282 | 819.0 | 1.713 | 71.28 | off | NRC-054711 |
| small_int16_2d [read_full] | read_full | 0.000199 | 814.8 | fitsio:fitsio_torch | 0.000121 | 814.8 | 1.655 | 65.49 | off | NRC-054711 |
| medium_int8_1d [read_full] | read_full | 0.000369 | 813.2 | fitsio:fitsio_torch | 0.000229 | 813.2 | 1.611 | 61.06 | off | NRC-054711 |
| small_int32_1d [read_full @ mps] | read_full | 0.000367 | 819.0 | fitsio:fitsio_torch_device_specialized | 0.000235 | 819.0 | 1.559 | 55.95 | off | NRC-054711 |
| small_uint16_2d [read_full @ mps] | read_full | 0.000411 | 819.0 | fitsio:fitsio_torch_device_specialized | 0.000270 | 819.0 | 1.519 | 51.86 | off | NRC-054711 |
| small_uint16_2d [read_full] | read_full | 0.000217 | 814.8 | fitsio:fitsio_torch | 0.000157 | 814.8 | 1.385 | 38.51 | off | NRC-054711 |
| small_float32_2d [read_full @ mps] | read_full | 0.000334 | 819.0 | fitsio:fitsio_torch_device_specialized | 0.000264 | 819.0 | 1.268 | 26.83 | off | NRC-054711 |
| small_int32_1d [header_read] | header_read | 0.000071 | 814.9 | fitsio:fitsio | 0.000058 | 814.9 | 1.222 | 22.15 | n/a | NRC-054711 |
| tiny_float32_1d [header_read] | header_read | 0.000072 | 814.9 | fitsio:fitsio | 0.000060 | 814.9 | 1.196 | 19.63 | n/a | NRC-054711 |
| small_int16_2d [read_full @ mps] | read_full | 0.000426 | 819.0 | fitsio:fitsio_torch_device_specialized | 0.000360 | 819.0 | 1.182 | 18.23 | off | NRC-054711 |
| timeseries_frame_003 [header_read] | header_read | 0.000071 | 814.9 | fitsio:fitsio | 0.000061 | 814.9 | 1.172 | 17.24 | n/a | NRC-054711 |
| tiny_float64_3d [read_full] | read_full | 0.000100 | 814.8 | fitsio:fitsio_torch | 0.000086 | 814.8 | 1.159 | 15.87 | off | NRC-054711 |
| small_int16_1d [header_read] | header_read | 0.000064 | 814.9 | fitsio:fitsio | 0.000055 | 814.9 | 1.155 | 15.47 | n/a | NRC-054711 |
| tiny_int64_2d [header_read] | header_read | 0.000067 | 814.9 | fitsio:fitsio | 0.000058 | 814.9 | 1.150 | 15.03 | n/a | NRC-054711 |
| compressed_rice_1 [read_full] | read_full | 0.025296 | 487.4 | fitsio:fitsio_torch | 0.022019 | 487.4 | 1.149 | 14.88 | on | NRC-054711 |
| small_float64_2d [header_read] | header_read | 0.000072 | 814.9 | fitsio:fitsio | 0.000063 | 814.9 | 1.149 | 14.86 | n/a | NRC-054711 |
| tiny_int32_2d [header_read] | header_read | 0.000065 | 814.9 | fitsio:fitsio | 0.000057 | 814.9 | 1.144 | 14.40 | n/a | NRC-054711 |
| small_int32_2d [header_read] | header_read | 0.000080 | 814.9 | fitsio:fitsio | 0.000070 | 814.9 | 1.141 | 14.14 | n/a | NRC-054711 |
| small_float64_1d [header_read] | header_read | 0.000067 | 814.9 | fitsio:fitsio | 0.000060 | 814.9 | 1.123 | 12.31 | n/a | NRC-054711 |
| large_int16_1d [header_read] | header_read | 0.000093 | 814.9 | fitsio:fitsio | 0.000083 | 814.9 | 1.118 | 11.75 | n/a | NRC-054711 |
| medium_float32_3d [read_full @ mps] | read_full | 0.001653 | 817.0 | fitsio:fitsio_torch_device_specialized | 0.001483 | 817.0 | 1.115 | 11.52 | off | NRC-054711 |
| mef_medium [header_read] | header_read | 0.000082 | 814.9 | fitsio:fitsio | 0.000074 | 814.9 | 1.114 | 11.40 | n/a | NRC-054711 |
| medium_float64_1d [header_read] | header_read | 0.000085 | 814.9 | fitsio:fitsio | 0.000077 | 814.9 | 1.107 | 10.65 | n/a | NRC-054711 |
| tiny_int16_3d [header_read] | header_read | 0.000063 | 814.9 | fitsio:fitsio | 0.000058 | 814.9 | 1.094 | 9.45 | n/a | NRC-054711 |
| large_int8_2d [header_read] | header_read | 0.000076 | 814.9 | fitsio:fitsio | 0.000069 | 814.9 | 1.093 | 9.27 | n/a | NRC-054711 |
| scaled_medium [read_full @ mps] | read_full | 0.001160 | 819.0 | fitsio:fitsio_torch_device_specialized | 0.001061 | 819.0 | 1.093 | 9.26 | off | NRC-054711 |
| small_int64_2d [header_read] | header_read | 0.000071 | 814.9 | fitsio:fitsio | 0.000065 | 814.9 | 1.088 | 8.80 | n/a | NRC-054711 |
| tiny_int8_1d [header_read] | header_read | 0.000064 | 814.9 | fitsio:fitsio | 0.000059 | 814.9 | 1.087 | 8.73 | n/a | NRC-054711 |
| medium_uint32_2d [header_read] | header_read | 0.000070 | 814.9 | fitsio:fitsio | 0.000065 | 814.9 | 1.084 | 8.36 | n/a | NRC-054711 |
| scaled_small [header_read] | header_read | 0.000071 | 814.9 | fitsio:fitsio | 0.000065 | 814.9 | 1.082 | 8.22 | n/a | NRC-054711 |
| mef_small [header_read] | header_read | 0.000079 | 814.9 | fitsio:fitsio | 0.000073 | 814.9 | 1.079 | 7.94 | n/a | NRC-054711 |
| tiny_float32_3d [header_read] | header_read | 0.000065 | 814.9 | fitsio:fitsio | 0.000061 | 814.9 | 1.073 | 7.32 | n/a | NRC-054711 |
| medium_int8_1d [read_full @ mps] | read_full | 0.000349 | 817.5 | fitsio:fitsio_torch_device_specialized | 0.000326 | 817.5 | 1.070 | 6.99 | off | NRC-054711 |
| tiny_int64_1d [header_read] | header_read | 0.000068 | 814.9 | fitsio:fitsio | 0.000063 | 814.9 | 1.067 | 6.69 | n/a | NRC-054711 |
| timeseries_frame_003 [read_full @ mps] | read_full | 0.000325 | 819.0 | fitsio:fitsio_torch_device_specialized | 0.000306 | 819.0 | 1.063 | 6.29 | off | NRC-054711 |
| small_int8_2d [header_read] | header_read | 0.000065 | 814.9 | fitsio:fitsio | 0.000062 | 814.9 | 1.060 | 5.95 | n/a | NRC-054711 |
| small_float32_3d [header_read] | header_read | 0.000072 | 814.9 | fitsio:fitsio | 0.000068 | 814.9 | 1.058 | 5.79 | n/a | NRC-054711 |
| medium_int32_1d [read_full @ mps] | read_full | 0.000299 | 817.0 | fitsio:fitsio_torch_device_specialized | 0.000284 | 817.0 | 1.056 | 5.55 | off | NRC-054711 |
| small_uint16_2d [header_read] | header_read | 0.000071 | 814.9 | fitsio:fitsio | 0.000067 | 814.9 | 1.055 | 5.45 | n/a | NRC-054711 |
| medium_uint16_2d [header_read] | header_read | 0.000071 | 814.9 | fitsio:fitsio | 0.000068 | 814.9 | 1.053 | 5.34 | n/a | NRC-054711 |
| large_int64_1d [header_read] | header_read | 0.000071 | 814.9 | fitsio:fitsio | 0.000067 | 814.9 | 1.053 | 5.34 | n/a | NRC-054711 |
| small_int64_3d [header_read] | header_read | 0.000062 | 814.9 | fitsio:fitsio | 0.000059 | 814.9 | 1.052 | 5.16 | n/a | NRC-054711 |
| timeseries_frame_000 [header_read] | header_read | 0.000075 | 814.9 | fitsio:fitsio | 0.000072 | 814.9 | 1.049 | 4.88 | n/a | NRC-054711 |
| small_int64_2d [read_full @ mps] | read_full | 0.000369 | 819.0 | fitsio:fitsio_torch_device_specialized | 0.000351 | 819.0 | 1.049 | 4.87 | off | NRC-054711 |
| medium_int16_3d [read_full @ mps] | read_full | 0.000752 | 817.0 | fitsio:fitsio_torch_device_specialized | 0.000724 | 817.0 | 1.039 | 3.86 | off | NRC-054711 |
| large_uint16_2d [read_full @ mps] | read_full | 0.003846 | 817.0 | fitsio:fitsio_torch_device_specialized | 0.003710 | 817.0 | 1.037 | 3.66 | off | NRC-054711 |
| large_int32_2d [header_read] | header_read | 0.000087 | 814.9 | fitsio:fitsio | 0.000084 | 814.9 | 1.027 | 2.67 | n/a | NRC-054711 |
| compressed_rice_1 [read_full @ mps] | read_full | 0.008067 | 817.0 | fitsio:fitsio_torch_device_specialized | 0.007860 | 817.0 | 1.026 | 2.63 | off | NRC-054711 |
| medium_int32_3d [read_full @ mps] | read_full | 0.001899 | 817.5 | fitsio:fitsio_torch_device_specialized | 0.001851 | 817.5 | 1.026 | 2.63 | off | NRC-054711 |
| medium_uint32_2d [read_full @ mps] | read_full | 0.001556 | 819.0 | fitsio:fitsio_torch_device_specialized | 0.001522 | 819.0 | 1.022 | 2.21 | off | NRC-054711 |
| large_int32_1d [read_full @ mps] | read_full | 0.000867 | 817.0 | fitsio:fitsio_torch_device_specialized | 0.000849 | 817.0 | 1.021 | 2.12 | off | NRC-054711 |
| timeseries_frame_001 [read_full @ mps] | read_full | 0.000278 | 819.0 | fitsio:fitsio_torch_device_specialized | 0.000273 | 819.0 | 1.020 | 1.98 | off | NRC-054711 |
| timeseries_frame_004 [header_read] | header_read | 0.000064 | 814.9 | fitsio:fitsio | 0.000063 | 814.9 | 1.016 | 1.60 | n/a | NRC-054711 |
| compressed_hcompress_1 [read_full @ mps] | read_full | 0.026049 | 817.0 | fitsio:fitsio_torch_device_specialized | 0.025721 | 817.0 | 1.013 | 1.27 | off | NRC-054711 |
| medium_float32_1d [read_full @ mps] | read_full | 0.000296 | 817.0 | fitsio:fitsio_torch_device_specialized | 0.000292 | 817.0 | 1.012 | 1.23 | off | NRC-054711 |
| tiny_int16_1d [header_read] | header_read | 0.000069 | 814.9 | fitsio:fitsio | 0.000068 | 814.9 | 1.011 | 1.10 | n/a | NRC-054711 |
| medium_int8_3d [read_full @ mps] | read_full | 0.001217 | 819.0 | astropy:astropy_torch_device_specialized | 0.001205 | 819.0 | 1.011 | 1.05 | off | NRC-054711 |
| small_int64_1d [header_read] | header_read | 0.000062 | 814.9 | fitsio:fitsio | 0.000062 | 814.9 | 1.007 | 0.74 | n/a | NRC-054711 |
| repeated_cutouts_50x_100x100 [repeated_cutouts_50x_100x100] | repeated_cutouts_50x_100x100 | 0.007214 | 814.8 | fitsio:fitsio_torch | 0.007165 | 814.8 | 1.007 | 0.68 | n/a | NRC-054711 |
| medium_int64_2d [read_full @ mps] | read_full | 0.002436 | 817.5 | fitsio:fitsio_torch_device_specialized | 0.002422 | 817.5 | 1.006 | 0.58 | off | NRC-054711 |
| compressed_rice_1 [read_full @ mps] | read_full | 0.010767 | 508.6 | fitsio:fitsio_torch_device_specialized | 0.010715 | 508.6 | 1.005 | 0.49 | on | NRC-054711 |
| medium_float32_2d [read_full @ mps] | read_full | 0.000946 | 817.0 | fitsio:fitsio_torch_device_specialized | 0.000941 | 817.0 | 1.005 | 0.47 | off | NRC-054711 |
| small_int8_3d [header_read] | header_read | 0.000076 | 814.9 | fitsio:fitsio | 0.000076 | 814.9 | 1.004 | 0.44 | n/a | NRC-054711 |
| compressed_hcompress_1 [read_full] | read_full | 0.050346 | 475.2 | fitsio:fitsio_torch | 0.050202 | 475.2 | 1.003 | 0.29 | on | NRC-054711 |
| scaled_large [header_read] | header_read | 0.000080 | 814.9 | fitsio:fitsio | 0.000080 | 814.9 | 1.001 | 0.05 | n/a | NRC-054711 |

### FITSTABLE - smart

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| mixed_1000000 [scan_count] | scan_count | 0.000470 | 1165.2 | fitsio:fitsio_torch | 0.000119 | 1165.2 | 3.957 | 295.68 | off | NRC-054711 |
| varlen_10000 [predicate_filter] | predicate_filter | 0.000474 | 1174.9 | fitsio:fitsio_torch | 0.000337 | 1174.9 | 1.408 | 40.77 | off | NRC-054711 |
| ascii_10000 [predicate_filter] | predicate_filter | 0.000624 | 1180.0 | fitsio:fitsio_torch | 0.000450 | 1180.0 | 1.387 | 38.69 | off | NRC-054711 |
| varlen_100000 [scan_count] | scan_count | 0.000140 | 1180.1 | fitsio:fitsio_torch | 0.000113 | 1180.1 | 1.234 | 23.39 | off | NRC-054711 |
| narrow_10000 [predicate_filter] | predicate_filter | 0.000414 | 748.0 | fitsio:fitsio_torch | 0.000350 | 748.0 | 1.185 | 18.46 | off | NRC-054711 |
| mixed_1000 [scan_count] | scan_count | 0.000128 | 746.2 | fitsio:fitsio_torch | 0.000110 | 746.2 | 1.156 | 15.60 | off | NRC-054711 |
| narrow_1000 [scan_count] | scan_count | 0.000102 | 746.1 | fitsio:fitsio_torch | 0.000100 | 746.1 | 1.019 | 1.91 | off | NRC-054711 |
| narrow_100000 [read_full] | read_full | 0.000661 | 749.9 | fitsio:fitsio_torch | 0.000657 | 749.9 | 1.006 | 0.58 | off | NRC-054711 |

### FITSTABLE - specialized

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| narrow_100000 [scan_count] | scan_count | 0.000254 | 756.0 | fitsio:fitsio | 0.000096 | 756.0 | 2.647 | 164.66 | off | NRC-054711 |
| mixed_1000000 [scan_count] | scan_count | 0.000147 | 1165.2 | fitsio:fitsio | 0.000104 | 1165.2 | 1.418 | 41.83 | off | NRC-054711 |
| narrow_1000000 [predicate_filter] | predicate_filter | 0.013827 | 857.1 | astropy:astropy | 0.009925 | 857.1 | 1.393 | 39.32 | off | NRC-054711 |
| mixed_10000 [scan_count] | scan_count | 0.000140 | 748.3 | fitsio:fitsio | 0.000108 | 748.3 | 1.296 | 29.61 | off | NRC-054711 |
| ascii_1000 [scan_count] | scan_count | 0.000111 | 1180.0 | fitsio:fitsio | 0.000088 | 1180.0 | 1.268 | 26.79 | off | NRC-054711 |
| mixed_1000000 [predicate_filter] | predicate_filter | 0.022755 | 1165.2 | astropy:astropy | 0.018485 | 1165.2 | 1.231 | 23.10 | off | NRC-054711 |
| narrow_1000 [scan_count] | scan_count | 0.000107 | 746.1 | fitsio:fitsio | 0.000096 | 746.1 | 1.115 | 11.53 | off | NRC-054711 |
| narrow_10000 [scan_count] | scan_count | 0.000088 | 748.0 | fitsio:fitsio | 0.000081 | 748.0 | 1.086 | 8.64 | off | NRC-054711 |
| typed_10000 [scan_count] | scan_count | 0.000091 | 1180.2 | fitsio:fitsio | 0.000088 | 1180.2 | 1.035 | 3.47 | off | NRC-054711 |
| mixed_1000 [scan_count] | scan_count | 0.000106 | 746.2 | fitsio:fitsio | 0.000102 | 746.2 | 1.033 | 3.35 | off | NRC-054711 |
| varlen_100000 [scan_count] | scan_count | 0.000089 | 1180.1 | fitsio:fitsio | 0.000087 | 1180.1 | 1.012 | 1.19 | off | NRC-054711 |
| ascii_10000 [scan_count] | scan_count | 0.000087 | 1180.0 | fitsio:fitsio | 0.000087 | 1180.0 | 1.004 | 0.38 | off | NRC-054711 |

## Notes

- Strict mmap fairness is enforced in comparable sets. Rows with unmatched mmap controls are marked `SKIPPED`.
- Rankings are family-specific and never mix smart vs specialized method families.
