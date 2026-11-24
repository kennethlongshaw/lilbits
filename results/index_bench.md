| context | path | step ms | CPU ms | peak RSS growth | index size | max diff vs masked |
|---|---|---|---|---|---|---|
| 1024 | masked dense | 1.37 | 1.45 | 1.5 MiB | 80 KiB | 0.0e+00 |
| 1024 | sparse gather | 0.79 | 0.86 | 0.2 MiB | 80 KiB | 2.2e-08 |
| 4096 | masked dense | 4.79 | 4.97 | 6.4 MiB | 320 KiB | 0.0e+00 |
| 4096 | sparse gather | 0.96 | 0.90 | 0.1 MiB | 320 KiB | 3.4e-08 |
| 16384 | masked dense | 15.03 | 15.44 | 21.1 MiB | 1280 KiB | 0.0e+00 |
| 16384 | sparse gather | 0.93 | 0.97 | 0.3 MiB | 1280 KiB | 2.6e-08 |
| 65536 | masked dense | 70.55 | 72.66 | 83.5 MiB | 5120 KiB | 0.0e+00 |
| 65536 | sparse gather | 1.58 | 1.65 | 0.9 MiB | 5120 KiB | 3.0e-08 |
