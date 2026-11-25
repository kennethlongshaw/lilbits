| model | params | deployed | inference RSS | decode ms/byte (CPU) | bytes/body pass | prefill B/s | train step | train peak RSS | state |
|---|---|---|---|---|---|---|---|---|---|
| pocket d512 (all on, LUT) | 10.3 M | 4.3 MiB | 162 MiB | 2.06 (2.05) | 4.92 | 3755 | 1589 ms | 800 MiB | 1306 KiB |
| FP16 same size as pocket d512 (all on, LUT) | 1.8 M | 3.4 MiB | 10 MiB | 1.46 (1.46) | 1.00 | 17471 | 255 ms | 130 MiB |  |
| pocket d1024 (all on, LUT) | 41.1 M | 14.4 MiB | 186 MiB | 2.78 (2.78) | 4.92 | 1239 | 5813 ms | 2387 MiB | 2582 KiB |
| FP16 same size as pocket d1024 (all on, LUT) | 7.0 M | 13.3 MiB | 30 MiB | 3.33 (3.31) | 1.00 | 6914 | 518 ms | 269 MiB |  |
| pocket d2048 (all on, LUT) | 164.6 M | 53.4 MiB | 270 MiB | 4.61 (4.60) | 4.92 | 314 | 20326 ms | 7785 MiB | 5140 KiB |
| FP16 same size as pocket d2048 (all on, LUT) | 27.6 M | 52.6 MiB | 109 MiB | 8.92 (8.89) | 1.00 | 1871 | 1714 ms | 551 MiB |  |
|   ablation d1024: native kernel | 41.1 M | 14.4 MiB | 316 MiB | 8.05 (8.04) | 4.92 | 958 | 5171 ms | 2373 MiB | 2582 KiB |
|   ablation d1024: no patching (router none) | 41.1 M | 14.4 MiB | 186 MiB | 18.73 (18.69) | 1.00 | 296 | 20627 ms | 9215 MiB | 2654 KiB |
|   ablation d1024: no index | 41.1 M | 14.4 MiB | 186 MiB | 3.41 (3.41) | 4.92 | 1534 | 4166 ms | 2249 MiB | 2582 KiB |
|   ablation d1024: FP core (no LittleBit core) | 49.6 M | 47.2 MiB | 331 MiB | 7.06 (7.05) | 4.92 | 1013 | 4069 ms | 1848 MiB | 2582 KiB |
|   ablation d1024: 1 loop | 41.1 M | 14.3 MiB | 186 MiB | 1.65 (1.65) | 4.92 | 1917 | 2839 ms | 1674 MiB | 1558 KiB |
|   ablation d1024: FP coda | 41.5 M | 38.7 MiB | 236 MiB | 3.18 (3.18) | 4.92 | 1210 | 4237 ms | 1911 MiB | 2582 KiB |
