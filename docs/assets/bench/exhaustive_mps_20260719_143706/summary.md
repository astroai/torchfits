# Benchmark Summary

- Run ID: `exhaustive_mps_20260719_143706`
- Scopes: `fits, fitstable`
- Total normalized rows: `3931`
- TorchFits deficit rows (all lags): `127`
- TorchFits significant deficits: `8`
- Hostname: `NRC-054711`
- CPU count: `8`
- torch.get_num_threads(): `4`
- Peak RSS (median across timed rows): `854.9 MB` (max `1201.4 MB`)

## Domain Coverage

| Domain | Rows | Skipped |
|---|---:|---:|
| fits | 2789 | 85 |
| fitstable | 1142 | 76 |

## Astronomer Scorecard

| Domain | Family | TorchFits First | Win Rate | Legacy In Ranking |
|---|---|---:|---:|---:|
| fits | smart | 306/310 | 98.7% | 0 |
| fits | specialized | 393/396 | 99.2% | 0 |
| fitstable | smart | 127/128 | 99.2% | 0 |
| fitstable | specialized | 144/144 | 100.0% | 0 |

- TorchFits devices observed in this run: `cpu, mps`
- Smart-family tables are the primary adoption view for astronomers (performance + portability).

## Adoption Checks

- `large-N` threshold: `n_points >= 100000`
- `small-N perceived` threshold: `torchfits_time_s < 0.000500s`
- `small-N max lag` threshold: `lag_ratio < 10.0x`

### Large-N Leadership

| Domain | Family | TorchFits First (large-N) | Win Rate |
|---|---|---:|---:|
| fitstable | smart | 48/49 | 98.0% |
| fitstable | specialized | 56/56 | 100.0% |

Large-N deficits detected:

| Case | n_points | Lag (x) | Behind (%) |
|---|---:|---:|---:|
| narrow_100000 [read_full] | 100000 | 1.147 | 14.70 |

### Small-N Visible Deficits

| Case | TorchFits (s) | Lag (x) | Behind (%) | Impact |
|---|---:|---:|---:|---|
| timeseries_frame_001 [read_full @ mps] | 0.000676 | 1.887 | 88.74 | visible |
| scaled_large [read_full @ mps] | 0.003319 | 1.079 | 7.89 | visible |
| compressed_rice_1 [read_full @ mps] | 0.007671 | 1.075 | 7.53 | visible |
| compressed_rice_1 [read_full @ mps] | 0.007235 | 1.041 | 4.09 | visible |
| scaled_large [read_full @ mps] | 0.003317 | 1.067 | 6.72 | visible |
| large_float32_2d [read_full @ mps] | 0.005272 | 1.065 | 6.47 | visible |

## TorchFits Deficits (Not First)

### FITS - smart

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| small_float64_1d [read_full] | read_full | 0.000138 | 854.9 | fitsio:fitsio_torch | 0.000064 | 854.9 | 2.165 | 116.52 | off | NRC-054711 |
| timeseries_frame_001 [read_full @ mps] | read_full | 0.000676 | 856.5 | fitsio:fitsio_torch_device | 0.000358 | 856.5 | 1.887 | 88.74 | off | NRC-054711 |
| timeseries_frame_004 [read_full] | read_full | 0.000203 | 854.9 | fitsio:fitsio_torch | 0.000122 | 854.9 | 1.663 | 66.30 | off | NRC-054711 |
| small_uint16_2d [read_full @ mps] | read_full | 0.000458 | 856.5 | fitsio:fitsio_torch_device | 0.000314 | 856.5 | 1.458 | 45.77 | off | NRC-054711 |
| tiny_int64_2d [read_full @ mps] | read_full | 0.000338 | 856.5 | fitsio:fitsio_torch_device | 0.000241 | 856.5 | 1.399 | 39.92 | off | NRC-054711 |
| small_int8_2d [read_full] | read_full | 0.000152 | 854.9 | fitsio:fitsio_torch | 0.000110 | 854.9 | 1.387 | 38.66 | off | NRC-054711 |
| small_float32_1d [read_full] | read_full | 0.000132 | 854.9 | fitsio:fitsio_torch | 0.000097 | 854.9 | 1.350 | 34.99 | off | NRC-054711 |
| mef_small [read_full] | read_full | 0.000176 | 854.9 | fitsio:fitsio_torch | 0.000136 | 854.9 | 1.293 | 29.29 | off | NRC-054711 |
| small_int16_2d [read_full @ mps] | read_full | 0.000363 | 856.5 | fitsio:fitsio_torch_device | 0.000284 | 856.5 | 1.278 | 27.82 | off | NRC-054711 |
| timeseries_frame_002 [read_full @ mps] | read_full | 0.000376 | 856.5 | fitsio:fitsio_torch_device | 0.000295 | 856.5 | 1.273 | 27.29 | off | NRC-054711 |
| tiny_int64_1d [read_full @ mps] | read_full | 0.000626 | 856.5 | astropy:astropy_torch_device | 0.000492 | 856.5 | 1.271 | 27.06 | off | NRC-054711 |
| tiny_int32_3d [read_full @ mps] | read_full | 0.000519 | 856.5 | fitsio:fitsio_torch_device | 0.000418 | 856.5 | 1.241 | 24.12 | off | NRC-054711 |
| tiny_int16_2d [read_full @ mps] | read_full | 0.000309 | 856.5 | fitsio:fitsio_torch_device | 0.000249 | 856.5 | 1.239 | 23.89 | off | NRC-054711 |
| tiny_int16_3d [read_full @ mps] | read_full | 0.000306 | 856.5 | fitsio:fitsio_torch_device | 0.000247 | 856.5 | 1.235 | 23.52 | off | NRC-054711 |
| tiny_int64_3d [read_full] | read_full | 0.000136 | 854.9 | fitsio:fitsio_torch | 0.000111 | 854.9 | 1.223 | 22.30 | off | NRC-054711 |
| tiny_int32_1d [read_full @ mps] | read_full | 0.000285 | 856.5 | fitsio:fitsio_torch_device | 0.000237 | 856.5 | 1.203 | 20.28 | off | NRC-054711 |
| medium_int8_1d [read_full @ mps] | read_full | 0.000335 | 856.5 | fitsio:fitsio_torch_device | 0.000285 | 856.5 | 1.175 | 17.51 | off | NRC-054711 |
| compressed_rice_1 [cutout_100x100] | cutout_100x100 | 0.000870 | 854.9 | fitsio:fitsio_torch | 0.000766 | 854.9 | 1.135 | 13.55 | n/a | NRC-054711 |
| tiny_float32_2d [read_full @ mps] | read_full | 0.000296 | 856.5 | fitsio:fitsio_torch_device | 0.000265 | 856.5 | 1.117 | 11.67 | off | NRC-054711 |
| small_int32_2d [read_full @ mps] | read_full | 0.000614 | 574.9 | astropy:astropy_torch_device | 0.000551 | 574.9 | 1.115 | 11.47 | on | NRC-054711 |
| large_int64_1d [read_full @ mps] | read_full | 0.002026 | 856.5 | fitsio:fitsio_torch_device | 0.001849 | 856.5 | 1.096 | 9.56 | off | NRC-054711 |
| scaled_large [read_full @ mps] | read_full | 0.003319 | 856.5 | fitsio:fitsio_torch_device | 0.003076 | 856.5 | 1.079 | 7.89 | off | NRC-054711 |
| tiny_int64_3d [read_full @ mps] | read_full | 0.000272 | 856.5 | fitsio:fitsio_torch_device | 0.000253 | 856.5 | 1.076 | 7.59 | off | NRC-054711 |
| compressed_rice_1 [read_full @ mps] | read_full | 0.007671 | 856.5 | fitsio:fitsio_torch_device | 0.007134 | 856.5 | 1.075 | 7.53 | off | NRC-054711 |
| tiny_float32_3d [read_full @ mps] | read_full | 0.000267 | 856.5 | fitsio:fitsio_torch_device | 0.000250 | 856.5 | 1.069 | 6.94 | off | NRC-054711 |
| timeseries_frame_003 [read_full @ mps] | read_full | 0.000304 | 856.5 | fitsio:fitsio_torch_device | 0.000288 | 856.5 | 1.054 | 5.44 | off | NRC-054711 |
| large_float32_2d [read_full @ mps] | read_full | 0.003850 | 856.5 | fitsio:fitsio_torch_device | 0.003673 | 856.5 | 1.048 | 4.84 | off | NRC-054711 |
| large_int32_2d [read_full @ mps] | read_full | 0.003649 | 856.5 | fitsio:fitsio_torch_device | 0.003483 | 856.5 | 1.048 | 4.75 | off | NRC-054711 |
| compressed_rice_1 [read_full @ mps] | read_full | 0.007235 | 574.3 | fitsio:fitsio_torch_device | 0.006951 | 574.3 | 1.041 | 4.09 | on | NRC-054711 |
| scaled_medium [read_full @ mps] | read_full | 0.000984 | 856.5 | fitsio:fitsio_torch_device | 0.000953 | 856.5 | 1.032 | 3.24 | off | NRC-054711 |
| large_uint16_2d [read_full @ mps] | read_full | 0.003756 | 856.5 | fitsio:fitsio_torch_device | 0.003640 | 856.5 | 1.032 | 3.18 | off | NRC-054711 |
| tiny_float32_1d [read_full @ mps] | read_full | 0.000285 | 856.5 | fitsio:fitsio_torch_device | 0.000277 | 856.5 | 1.031 | 3.07 | off | NRC-054711 |
| medium_uint16_2d [read_full @ mps] | read_full | 0.000995 | 856.5 | fitsio:fitsio_torch_device | 0.000966 | 856.5 | 1.030 | 3.01 | off | NRC-054711 |
| medium_int64_3d [read_full @ mps] | read_full | 0.003016 | 856.5 | fitsio:fitsio_torch_device | 0.002936 | 856.5 | 1.027 | 2.74 | off | NRC-054711 |
| medium_int32_2d [read_full @ mps] | read_full | 0.000970 | 856.5 | fitsio:fitsio_torch_device | 0.000957 | 856.5 | 1.014 | 1.40 | off | NRC-054711 |
| compressed_hcompress_1 [read_full @ mps] | read_full | 0.022075 | 574.4 | fitsio:fitsio_torch_device | 0.021778 | 574.4 | 1.014 | 1.36 | on | NRC-054711 |
| small_float32_3d [read_full @ mps] | read_full | 0.000325 | 856.5 | fitsio:fitsio_torch_device | 0.000322 | 856.5 | 1.011 | 1.14 | off | NRC-054711 |
| compressed_hcompress_1 [read_full @ mps] | read_full | 0.022077 | 856.5 | fitsio:fitsio_torch_device | 0.021848 | 856.5 | 1.010 | 1.05 | off | NRC-054711 |
| medium_int64_2d [read_full @ mps] | read_full | 0.001906 | 856.5 | fitsio:fitsio_torch_device | 0.001887 | 856.5 | 1.010 | 1.01 | off | NRC-054711 |
| compressed_hcompress_1 [read_full] | read_full | 0.021619 | 854.9 | fitsio:fitsio_torch | 0.021616 | 854.9 | 1.000 | 0.01 | off | NRC-054711 |

### FITS - specialized

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| medium_int8_2d [header_read] | header_read | 0.000247 | 854.9 | fitsio:fitsio | 0.000096 | 854.9 | 2.581 | 158.12 | n/a | NRC-054711 |
| small_float32_1d [read_full @ mps] | read_full | 0.000451 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000224 | 856.5 | 2.013 | 101.32 | off | NRC-054711 |
| small_int64_1d [read_full] | read_full | 0.000133 | 854.9 | fitsio:fitsio_torch | 0.000070 | 854.9 | 1.907 | 90.69 | off | NRC-054711 |
| tiny_float64_3d [read_full] | read_full | 0.000134 | 854.9 | fitsio:fitsio_torch | 0.000078 | 854.9 | 1.724 | 72.41 | off | NRC-054711 |
| tiny_float32_2d [read_full @ mps] | read_full | 0.000393 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000253 | 856.5 | 1.556 | 55.55 | off | NRC-054711 |
| small_uint16_2d [read_full @ mps] | read_full | 0.000414 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000282 | 856.5 | 1.471 | 47.12 | off | NRC-054711 |
| tiny_int64_2d [read_full @ mps] | read_full | 0.000347 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000240 | 856.5 | 1.449 | 44.87 | off | NRC-054711 |
| timeseries_frame_000 [read_full] | read_full | 0.000137 | 854.9 | fitsio:fitsio_torch | 0.000101 | 854.9 | 1.352 | 35.18 | off | NRC-054711 |
| small_float32_1d [read_full] | read_full | 0.000132 | 854.9 | fitsio:fitsio_torch | 0.000097 | 854.9 | 1.351 | 35.12 | off | NRC-054711 |
| small_float64_1d [header_read] | header_read | 0.000061 | 854.9 | fitsio:fitsio | 0.000047 | 854.9 | 1.293 | 29.31 | n/a | NRC-054711 |
| tiny_float32_3d [read_full @ mps] | read_full | 0.000361 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000282 | 856.5 | 1.281 | 28.08 | off | NRC-054711 |
| tiny_int16_3d [read_full @ mps] | read_full | 0.000300 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000234 | 856.5 | 1.280 | 27.99 | off | NRC-054711 |
| small_int32_2d [read_full] | read_full | 0.000140 | 854.9 | fitsio:fitsio_torch | 0.000112 | 854.9 | 1.254 | 25.43 | off | NRC-054711 |
| compressed_rice_1 [cutout_100x100] | cutout_100x100 | 0.000952 | 854.9 | fitsio:fitsio_torch | 0.000766 | 854.9 | 1.243 | 24.25 | n/a | NRC-054711 |
| medium_int8_1d [read_full @ mps] | read_full | 0.000376 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000310 | 856.5 | 1.211 | 21.13 | off | NRC-054711 |
| medium_int64_2d [header_read] | header_read | 0.000056 | 854.9 | fitsio:fitsio | 0.000046 | 854.9 | 1.206 | 20.56 | n/a | NRC-054711 |
| small_int32_3d [read_full @ mps] | read_full | 0.000389 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000328 | 856.5 | 1.188 | 18.75 | off | NRC-054711 |
| medium_int16_3d [header_read] | header_read | 0.000061 | 854.9 | fitsio:fitsio | 0.000051 | 854.9 | 1.187 | 18.71 | n/a | NRC-054711 |
| tiny_float64_2d [header_read] | header_read | 0.000062 | 854.9 | fitsio:fitsio | 0.000053 | 854.9 | 1.169 | 16.93 | n/a | NRC-054711 |
| medium_float64_2d [header_read] | header_read | 0.000054 | 854.9 | fitsio:fitsio | 0.000047 | 854.9 | 1.148 | 14.80 | n/a | NRC-054711 |
| tiny_int32_1d [header_read] | header_read | 0.000055 | 854.9 | fitsio:fitsio | 0.000048 | 854.9 | 1.145 | 14.48 | n/a | NRC-054711 |
| large_int16_2d [header_read] | header_read | 0.000085 | 854.9 | fitsio:fitsio | 0.000075 | 854.9 | 1.134 | 13.37 | n/a | NRC-054711 |
| medium_int16_1d [header_read] | header_read | 0.000053 | 854.9 | fitsio:fitsio | 0.000047 | 854.9 | 1.131 | 13.12 | n/a | NRC-054711 |
| scaled_small [header_read] | header_read | 0.000075 | 854.9 | fitsio:fitsio | 0.000068 | 854.9 | 1.109 | 10.85 | n/a | NRC-054711 |
| large_int64_1d [header_read] | header_read | 0.000048 | 854.9 | fitsio:fitsio | 0.000044 | 854.9 | 1.108 | 10.79 | n/a | NRC-054711 |
| timeseries_frame_000 [header_read] | header_read | 0.000051 | 854.9 | fitsio:fitsio | 0.000046 | 854.9 | 1.107 | 10.73 | n/a | NRC-054711 |
| medium_int64_1d [header_read] | header_read | 0.000047 | 854.9 | fitsio:fitsio | 0.000043 | 854.9 | 1.096 | 9.63 | n/a | NRC-054711 |
| medium_int32_3d [header_read] | header_read | 0.000073 | 854.9 | fitsio:fitsio | 0.000066 | 854.9 | 1.095 | 9.48 | n/a | NRC-054711 |
| large_uint16_2d [header_read] | header_read | 0.000060 | 854.9 | fitsio:fitsio | 0.000055 | 854.9 | 1.088 | 8.81 | n/a | NRC-054711 |
| small_int16_2d [header_read] | header_read | 0.000053 | 854.9 | fitsio:fitsio | 0.000049 | 854.9 | 1.078 | 7.78 | n/a | NRC-054711 |
| tiny_float32_3d [header_read] | header_read | 0.000061 | 854.9 | fitsio:fitsio | 0.000057 | 854.9 | 1.076 | 7.58 | n/a | NRC-054711 |
| tiny_int32_3d [header_read] | header_read | 0.000053 | 854.9 | fitsio:fitsio | 0.000050 | 854.9 | 1.072 | 7.22 | n/a | NRC-054711 |
| large_int32_1d [read_full @ mps] | read_full | 0.000980 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000916 | 856.5 | 1.070 | 6.99 | off | NRC-054711 |
| scaled_large [read_full @ mps] | read_full | 0.003317 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.003108 | 856.5 | 1.067 | 6.72 | off | NRC-054711 |
| large_float32_2d [read_full @ mps] | read_full | 0.005272 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.004952 | 856.5 | 1.065 | 6.47 | off | NRC-054711 |
| large_int8_1d [header_read] | header_read | 0.000055 | 854.9 | fitsio:fitsio | 0.000052 | 854.9 | 1.064 | 6.42 | n/a | NRC-054711 |
| small_int64_1d [read_full @ mps] | read_full | 0.000361 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000340 | 856.5 | 1.063 | 6.29 | off | NRC-054711 |
| large_float32_1d [header_read] | header_read | 0.000056 | 854.9 | fitsio:fitsio | 0.000052 | 854.9 | 1.062 | 6.22 | n/a | NRC-054711 |
| small_int16_1d [header_read] | header_read | 0.000049 | 854.9 | fitsio:fitsio | 0.000046 | 854.9 | 1.062 | 6.20 | n/a | NRC-054711 |
| mef_small [header_read] | header_read | 0.000067 | 854.9 | fitsio:fitsio | 0.000063 | 854.9 | 1.060 | 6.04 | n/a | NRC-054711 |
| medium_float64_3d [header_read] | header_read | 0.000060 | 854.9 | fitsio:fitsio | 0.000057 | 854.9 | 1.060 | 6.03 | n/a | NRC-054711 |
| tiny_int16_1d [read_full @ mps] | read_full | 0.000249 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000235 | 856.5 | 1.059 | 5.91 | off | NRC-054711 |
| small_int16_3d [header_read] | header_read | 0.000053 | 854.9 | fitsio:fitsio | 0.000050 | 854.9 | 1.058 | 5.77 | n/a | NRC-054711 |
| tiny_int64_2d [header_read] | header_read | 0.000060 | 854.9 | fitsio:fitsio | 0.000057 | 854.9 | 1.057 | 5.68 | n/a | NRC-054711 |
| medium_int16_2d [header_read] | header_read | 0.000055 | 854.9 | fitsio:fitsio | 0.000052 | 854.9 | 1.056 | 5.55 | n/a | NRC-054711 |
| tiny_int8_3d [header_read] | header_read | 0.000057 | 854.9 | fitsio:fitsio | 0.000054 | 854.9 | 1.055 | 5.47 | n/a | NRC-054711 |
| small_int16_1d [read_full] | read_full | 0.000068 | 854.9 | fitsio:fitsio_torch | 0.000065 | 854.9 | 1.050 | 4.98 | off | NRC-054711 |
| small_int16_2d [read_full @ mps] | read_full | 0.000300 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000286 | 856.5 | 1.048 | 4.79 | off | NRC-054711 |
| scaled_medium [read_full @ mps] | read_full | 0.000987 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000943 | 856.5 | 1.047 | 4.72 | off | NRC-054711 |
| small_int64_3d [header_read] | header_read | 0.000051 | 854.9 | fitsio:fitsio | 0.000049 | 854.9 | 1.044 | 4.41 | n/a | NRC-054711 |
| tiny_int16_2d [header_read] | header_read | 0.000052 | 854.9 | fitsio:fitsio | 0.000050 | 854.9 | 1.040 | 4.05 | n/a | NRC-054711 |
| small_float32_3d [header_read] | header_read | 0.000054 | 854.9 | fitsio:fitsio | 0.000052 | 854.9 | 1.040 | 4.03 | n/a | NRC-054711 |
| small_uint16_2d [header_read] | header_read | 0.000055 | 854.9 | fitsio:fitsio | 0.000053 | 854.9 | 1.040 | 4.00 | n/a | NRC-054711 |
| timeseries_frame_001 [read_full @ mps] | read_full | 0.000291 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000280 | 856.5 | 1.038 | 3.81 | off | NRC-054711 |
| large_int8_2d [header_read] | header_read | 0.000058 | 854.9 | fitsio:fitsio | 0.000056 | 854.9 | 1.037 | 3.71 | n/a | NRC-054711 |
| timeseries_frame_001 [header_read] | header_read | 0.000057 | 854.9 | fitsio:fitsio | 0.000055 | 854.9 | 1.036 | 3.63 | n/a | NRC-054711 |
| tiny_float32_1d [read_full @ mps] | read_full | 0.000235 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000227 | 856.5 | 1.031 | 3.11 | off | NRC-054711 |
| tiny_int8_2d [header_read] | header_read | 0.000066 | 854.9 | fitsio:fitsio | 0.000064 | 854.9 | 1.031 | 3.10 | n/a | NRC-054711 |
| compressed_rice_1 [cutout_100x100 @ mps] | cutout_100x100 | 0.000962 | 575.0 | fitsio:fitsio_torch_device_specialized | 0.000933 | 856.5 | 1.031 | 3.07 | n/a | NRC-054711 |
| small_int8_2d [header_read] | header_read | 0.000060 | 854.9 | fitsio:fitsio | 0.000058 | 854.9 | 1.030 | 3.03 | n/a | NRC-054711 |
| tiny_int16_3d [header_read] | header_read | 0.000053 | 854.9 | fitsio:fitsio | 0.000051 | 854.9 | 1.030 | 3.01 | n/a | NRC-054711 |
| timeseries_frame_004 [header_read] | header_read | 0.000062 | 854.9 | fitsio:fitsio | 0.000061 | 854.9 | 1.030 | 2.96 | n/a | NRC-054711 |
| tiny_int32_3d [read_full @ mps] | read_full | 0.000246 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000239 | 856.5 | 1.029 | 2.94 | off | NRC-054711 |
| tiny_int32_1d [read_full @ mps] | read_full | 0.000229 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000223 | 856.5 | 1.029 | 2.94 | off | NRC-054711 |
| scaled_medium [header_read] | header_read | 0.000052 | 854.9 | fitsio:fitsio | 0.000051 | 854.9 | 1.028 | 2.78 | n/a | NRC-054711 |
| tiny_float32_1d [header_read] | header_read | 0.000057 | 854.9 | fitsio:fitsio | 0.000056 | 854.9 | 1.025 | 2.46 | n/a | NRC-054711 |
| compressed_rice_1 [read_full @ mps] | read_full | 0.007097 | 574.3 | fitsio:fitsio_torch_device_specialized | 0.006931 | 574.3 | 1.024 | 2.39 | on | NRC-054711 |
| small_int32_2d [header_read] | header_read | 0.000049 | 854.9 | fitsio:fitsio | 0.000048 | 854.9 | 1.024 | 2.36 | n/a | NRC-054711 |
| compressed_rice_1 [read_full @ mps] | read_full | 0.007123 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.006964 | 856.5 | 1.023 | 2.29 | off | NRC-054711 |
| compressed_hcompress_1 [read_full @ mps] | read_full | 0.022122 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.021632 | 856.5 | 1.023 | 2.26 | off | NRC-054711 |
| medium_uint16_2d [read_full @ mps] | read_full | 0.001026 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.001005 | 856.5 | 1.020 | 2.04 | off | NRC-054711 |
| large_int64_2d [read_full @ mps] | read_full | 0.008307 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.008145 | 856.5 | 1.020 | 1.99 | off | NRC-054711 |
| tiny_int64_1d [read_full @ mps] | read_full | 0.000254 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000250 | 856.5 | 1.019 | 1.85 | off | NRC-054711 |
| medium_float32_2d [read_full @ mps] | read_full | 0.000969 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000953 | 856.5 | 1.017 | 1.66 | off | NRC-054711 |
| timeseries_frame_000 [read_full @ mps] | read_full | 0.000317 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000314 | 856.5 | 1.012 | 1.20 | off | NRC-054711 |
| large_int64_1d [read_full @ mps] | read_full | 0.001846 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.001827 | 856.5 | 1.010 | 1.02 | off | NRC-054711 |
| medium_float32_1d [header_read] | header_read | 0.000047 | 854.9 | fitsio:fitsio | 0.000047 | 854.9 | 1.009 | 0.90 | n/a | NRC-054711 |
| multi_mef_10ext [header_read] | header_read | 0.000062 | 854.9 | fitsio:fitsio | 0.000062 | 854.9 | 1.007 | 0.67 | n/a | NRC-054711 |
| small_float32_2d [header_read] | header_read | 0.000056 | 854.9 | fitsio:fitsio | 0.000056 | 854.9 | 1.006 | 0.60 | n/a | NRC-054711 |
| small_uint32_2d [header_read] | header_read | 0.000052 | 854.9 | fitsio:fitsio | 0.000052 | 854.9 | 1.006 | 0.56 | n/a | NRC-054711 |
| medium_int8_3d [header_read] | header_read | 0.000063 | 854.9 | fitsio:fitsio | 0.000063 | 854.9 | 1.004 | 0.40 | n/a | NRC-054711 |
| compressed_hcompress_1 [read_full @ mps] | read_full | 0.022011 | 574.4 | fitsio:fitsio_torch_device_specialized | 0.021924 | 574.4 | 1.004 | 0.40 | on | NRC-054711 |
| small_int16_3d [read_full @ mps] | read_full | 0.000272 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000271 | 856.5 | 1.003 | 0.32 | off | NRC-054711 |
| small_int64_3d [read_full @ mps] | read_full | 0.000468 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000467 | 856.5 | 1.003 | 0.29 | off | NRC-054711 |
| tiny_int64_3d [read_full @ mps] | read_full | 0.000244 | 856.5 | fitsio:fitsio_torch_device_specialized | 0.000243 | 856.5 | 1.001 | 0.09 | off | NRC-054711 |

### FITSTABLE - smart

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| narrow_100000 [read_full] | read_full | 0.000667 | 783.7 | fitsio:fitsio_torch | 0.000581 | 783.7 | 1.147 | 14.70 | off | NRC-054711 |

### FITSTABLE - specialized

| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| narrow_1000000 [predicate_filter] | predicate_filter | 0.008115 | 856.6 | astropy:astropy | 0.007979 | 856.6 | 1.017 | 1.70 | on | NRC-054711 |

## Notes

- Strict mmap fairness is enforced in comparable sets. Rows with unmatched mmap controls are marked `SKIPPED`.
- Rankings are family-specific and never mix smart vs specialized method families.
