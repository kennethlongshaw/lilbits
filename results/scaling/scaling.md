# Scaling sweep

Matched on deployed MiB: [0.5, 1.0, 2.0]. Short runs: 20 steps x 4 x 128 bytes of Shakespeare, cosine; single-threaded CPU, one process per point. Smoke-scale: read slopes and orderings, not absolute bits per byte.

## Points

| curve | target | width | params | deployed MiB | disk MiB | train loss | eval loss | bpb | train step ms | train wall s | train CPU util | train peak RSS MiB | decode ms/B | decode CPU ms/B | prefill B/s | infer peak RSS MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 depth=2 | 0.5 | 96 | 0.27M | 0.51 | 1.03 | 3.612 | 3.591 | 5.181 | 27 | 0.6 | 1.03 | 45 | 0.85 | 0.88 | 63267 | 6 |
| fp16 depth=2 | 1.0 | 128 | 0.46M | 0.87 | 1.76 | 3.231 | 3.211 | 4.632 | 35 | 0.7 | 1.03 | 69 | 0.82 | 0.85 | 45353 | 13 |
| fp16 depth=2 | 2.0 | 192 | 1.01M | 1.93 | 3.87 | 2.894 | 2.886 | 4.164 | 64 | 1.3 | 1.03 | 109 | 1.31 | 1.31 | 22302 | 17 |
| fp16 depth=4 | 0.5 | 64 | 0.23M | 0.44 | 0.90 | 4.174 | 4.152 | 5.990 | 39 | 0.8 | 1.03 | 60 | 1.13 | 1.15 | 56036 | 9 |
| fp16 depth=4 | 1.0 | 96 | 0.51M | 0.97 | 1.96 | 3.774 | 3.738 | 5.393 | 47 | 1.0 | 1.03 | 77 | 1.58 | 1.71 | 33019 | 5 |
| fp16 depth=4 | 2.0 | 128 | 0.88M | 1.68 | 3.39 | 3.279 | 3.263 | 4.708 | 87 | 1.8 | 1.03 | 119 | 1.81 | 1.85 | 23372 | 8 |
| pocket loops=3 bpw=0.55 | 0.5 | 64 | 0.32M | 0.49 | 1.36 | 8.679 | 4.287 | 6.185 | 171 | 3.2 | 1.02 | 102 | 2.39 | 2.45 | 6424 | 6 |
| pocket loops=3 bpw=0.55 | 1.0 | 192 | 1.57M | 1.07 | 6.10 | 8.263 | 4.086 | 5.896 | 283 | 5.8 | 1.04 | 233 | 2.46 | 2.51 | 4757 | 12 |
| pocket loops=3 bpw=0.55 | 2.0 | 320 | 4.16M | 2.15 | 15.98 | 8.164 | 4.039 | 5.827 | 553 | 13.2 | 1.03 | 487 | 3.60 | 3.74 | 3150 | 27 |
| pocket loops=1 bpw=0.55 | 0.5 | 64 | 0.32M | 0.49 | 1.35 | 8.525 | 4.212 | 6.077 | 119 | 2.4 | 1.03 | 81 | 1.91 | 1.98 | 8702 | 6 |
| pocket loops=1 bpw=0.55 | 1.0 | 192 | 1.55M | 1.04 | 6.05 | 8.256 | 4.086 | 5.895 | 231 | 4.7 | 1.03 | 163 | 2.13 | 2.20 | 7605 | 12 |
| pocket loops=1 bpw=0.55 | 2.0 | 320 | 4.14M | 2.12 | 15.90 | 8.180 | 4.050 | 5.843 | 325 | 7.2 | 1.03 | 339 | 1.91 | 2.02 | 7218 | 25 |
| pocket loops=3 bpw=1.0 | 0.5 | 64 | 0.35M | 0.50 | 1.47 | 8.734 | 4.314 | 6.223 | 191 | 3.9 | 1.03 | 103 | 3.01 | 3.09 | 6366 | 6 |
| pocket loops=3 bpw=1.0 | 1.0 | 192 | 1.85M | 1.10 | 7.20 | 8.262 | 4.086 | 5.895 | 344 | 6.7 | 1.03 | 243 | 3.12 | 3.22 | 4229 | 14 |
| pocket loops=3 bpw=1.0 | 2.0 | 320 | 4.93M | 2.25 | 18.95 | 8.163 | 4.039 | 5.827 | 533 | 12.2 | 1.03 | 525 | 3.16 | 3.27 | 3250 | 31 |
| looped loops=3 bpw=0.55 | 1.0 | 256 | 2.52M | 1.21 | 9.70 | 3.090 | 3.065 | 4.422 | 1243 | 22.1 | 1.03 | 803 | 8.54 | 8.66 | 1027 | 89 |
| looped loops=3 bpw=0.55 | 2.0 | 320 | 3.99M | 1.83 | 15.31 | 3.024 | 3.003 | 4.332 | 1554 | 29.4 | 1.03 | 1029 | 12.28 | 12.43 | 662 | 101 |
| looped loops=1 bpw=0.55 | 1.0 | 256 | 2.50M | 1.18 | 9.64 | 3.101 | 3.065 | 4.422 | 922 | 18.4 | 1.01 | 465 | 5.57 | 5.69 | 1469 | 65 |
| looped loops=1 bpw=0.55 | 2.0 | 320 | 3.97M | 1.79 | 15.24 | 3.027 | 2.980 | 4.299 | 1071 | 22.0 | 1.02 | 641 | 4.09 | 4.22 | 2013 | 74 |
| looped loops=3 bpw=1.0 | 1.0 | 256 | 3.02M | 1.27 | 11.62 | 3.070 | 3.034 | 4.377 | 1362 | 23.2 | 1.02 | 861 | 8.41 | 8.60 | 1006 | 78 |
| looped loops=3 bpw=1.0 | 2.0 | 320 | 4.77M | 1.93 | 18.29 | 3.032 | 2.999 | 4.327 | 1929 | 33.4 | 1.02 | 1161 | 8.77 | 8.92 | 701 | 98 |
| oploop loops=4 | 1.0 | 128 | 0.61M | 1.16 | 2.34 | 3.282 | 3.294 | 4.753 | 721 | 14.3 | 1.03 | 350 | 49.91 | 51.05 | 987 | 320 |
| oploop loops=4 | 2.0 | 192 | 1.35M | 2.57 | 5.16 | 3.078 | 3.084 | 4.450 | 1151 | 23.0 | 1.03 | 499 | 84.22 | 86.68 | 755 | 470 |
| oploop loops=2 | 1.0 | 128 | 0.61M | 1.16 | 2.34 | 3.189 | 3.196 | 4.611 | 384 | 7.6 | 1.03 | 235 | 22.71 | 23.42 | 1774 | 311 |
| oploop loops=2 | 2.0 | 192 | 1.35M | 2.57 | 5.16 | 2.996 | 2.993 | 4.317 | 626 | 12.2 | 1.03 | 337 | 50.59 | 52.04 | 1176 | 454 |

Out of range (no width within 35% of the target): looped loops=3 bpw=0.55 @ 0.5 deployed MiB, looped loops=1 bpw=0.55 @ 0.5 deployed MiB, looped loops=3 bpw=1.0 @ 0.5 deployed MiB, oploop loops=4 @ 0.5 deployed MiB, oploop loops=2 @ 0.5 deployed MiB

## Fits (y = a * size^b, size in deployed MiB)

| curve | points | bits / byte (short run) | decode ms / byte | train step ms | prefill bytes / s | inference peak RSS growth MiB |
|---|---|---|---|---|---|---|
| fp16 depth=2 | 3 | b=-0.16 (r2 0.98) | b=+0.34 (r2 0.78) | b=+0.64 (r2 0.98) | b=-0.79 (r2 0.99) | b=+0.77 (r2 0.87) |
| fp16 depth=4 | 3 | b=-0.18 (r2 0.97) | b=+0.36 (r2 0.98) | b=+0.58 (r2 0.85) | b=-0.65 (r2 1.00) | b=-0.14 (r2 0.09) |
| pocket loops=3 bpw=0.55 | 3 | b=-0.04 (r2 0.91) | b=+0.27 (r2 0.78) | b=+0.80 (r2 0.99) | b=-0.48 (r2 0.99) | b=+1.05 (r2 1.00) |
| pocket loops=1 bpw=0.55 | 3 | b=-0.03 (r2 0.92) | b=+0.00 (r2 0.00) | b=+0.68 (r2 0.98) | b=-0.13 (r2 0.95) | b=+1.02 (r2 1.00) |
| pocket loops=3 bpw=1.0 | 3 | b=-0.04 (r2 0.90) | b=+0.03 (r2 0.92) | b=+0.68 (r2 1.00) | b=-0.45 (r2 0.99) | b=+1.11 (r2 1.00) |
| looped loops=3 bpw=0.55 | 2 | b=-0.05 (r2 1.00) | b=+0.88 (r2 1.00) | b=+0.54 (r2 1.00) | b=-1.05 (r2 1.00) | b=+0.29 (r2 1.00) |
| looped loops=1 bpw=0.55 | 2 | b=-0.07 (r2 1.00) | b=-0.74 (r2 1.00) | b=+0.35 (r2 1.00) | b=+0.75 (r2 1.00) | b=+0.30 (r2 1.00) |
| looped loops=3 bpw=1.0 | 2 | b=-0.03 (r2 1.00) | b=+0.10 (r2 1.00) | b=+0.84 (r2 1.00) | b=-0.87 (r2 1.00) | b=+0.54 (r2 1.00) |
| oploop loops=4 | 2 | b=-0.08 (r2 1.00) | b=+0.66 (r2 1.00) | b=+0.59 (r2 1.00) | b=-0.34 (r2 1.00) | b=+0.48 (r2 1.00) |
| oploop loops=2 | 2 | b=-0.08 (r2 1.00) | b=+1.01 (r2 1.00) | b=+0.62 (r2 1.00) | b=-0.52 (r2 1.00) | b=+0.48 (r2 1.00) |

## Crossovers against `fp16 depth=2`

Each cell: the curve / baseline ratio over the curve's measured sizes (baseline read off its fit), then the crossover size if the two fits meet within the measured range (2x margin), else whether it is better or worse at every measured size. For prefill, higher is better; for the rest, lower. Fits from 2 points always have r2 = 1; trust slopes from 3+ points and r2 near 1.

| curve | bits / byte (short run) | decode ms / byte | train step ms | prefill bytes / s | inference peak RSS growth MiB |
|---|---|---|---|---|---|
| fp16 depth=4 | x1.13-1.15; worse everywhere measured | x1.53-1.57; worse everywhere measured | x1.37-1.49; worse everywhere measured | x0.76-0.91; crosses at 3.3 deployed MiB, better above | x0.39-1.33; crosses at 0.6 deployed MiB, better above |
| pocket loops=3 bpw=0.55 | x1.19-1.42; worse everywhere measured | x2.63-2.89; worse everywhere measured | x6.38-8.03; worse everywhere measured | x0.10-0.15; worse everywhere measured | x0.86-1.29; crosses at 0.85 deployed MiB, better below |
| pocket loops=1 bpw=0.55 | x1.17-1.43; worse everywhere measured | x1.55-2.56; worse everywhere measured | x4.87-5.19; worse everywhere measured | x0.13-0.34; worse everywhere measured | x0.84-1.22; crosses at 0.96 deployed MiB, better below |
| pocket loops=3 bpw=1.0 | x1.20-1.43; worse everywhere measured | x2.43-3.87; worse everywhere measured | x7.47-7.94; worse everywhere measured | x0.09-0.16; worse everywhere measured | x0.85-1.42; crosses at 0.8 deployed MiB, better below |
| looped loops=3 bpw=0.55 | x0.99-1.04; crosses at 1.3 deployed MiB, better below | x8.09-10.10; worse everywhere measured | x25.97-27.11; worse everywhere measured | x0.03-0.03; worse everywhere measured | x5.50-6.70; worse everywhere measured |
| looped loops=1 bpw=0.55 | x0.99-1.03; crosses at 1.4 deployed MiB, better below | x3.39-5.32; worse everywhere measured | x18.12-20.42; worse everywhere measured | x0.04-0.08; worse everywhere measured | x4.08-4.97; worse everywhere measured |
| looped loops=3 bpw=1.0 | x0.99-1.05; crosses at 1.4 deployed MiB, better below | x7.09-7.83; worse everywhere measured | x28.76-31.20; worse everywhere measured | x0.03-0.03; worse everywhere measured | x5.13-5.63; worse everywhere measured |
| oploop loops=4 | x1.06-1.13; worse everywhere measured | x47.89-61.71; worse everywhere measured | x15.47-16.11; worse everywhere measured | x0.03-0.04; worse everywhere measured | x19.78-24.78; worse everywhere measured |
| oploop loops=2 | x1.03-1.09; crosses at 0.83 deployed MiB, better below | x21.80-37.08; worse everywhere measured | x8.42-8.57; worse everywhere measured | x0.05-0.06; worse everywhere measured | x19.09-24.03; worse everywhere measured |

![scaling](scaling.png)
