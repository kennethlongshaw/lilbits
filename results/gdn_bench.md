| T | rule | fwd+bwd ms | CPU ms | peak RSS growth | max diff vs naive |
|---|---|---|---|---|---|
| 64 | naive loop | 29.0 | 30.0 | 20.2 MiB | 0.0e+00 |
| 64 | chunked (C=64) | 5.6 | 6.1 | 6.4 MiB | 4.8e-07 |
| 256 | naive loop | 180.2 | 192.0 | 80.0 MiB | 0.0e+00 |
| 256 | chunked (C=64) | 22.5 | 23.3 | 27.3 MiB | 6.6e-07 |
| 1024 | naive loop | 1365.8 | 1416.6 | 315.3 MiB | 0.0e+00 |
| 1024 | chunked (C=64) | 86.1 | 88.5 | 85.9 MiB | 8.9e-07 |
| 4096 | naive loop | 16450.0 | 16970.3 | 811.4 MiB | 0.0e+00 |
| 4096 | chunked (C=64) | 1109.0 | 1199.5 | 319.0 MiB | 1.2e-06 |
