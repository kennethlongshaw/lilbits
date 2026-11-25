# Scaling sweep

Matched on M params: [0.5, 1.0, 2.0]. Short runs: 20 steps x 4 x 128 bytes of Shakespeare, cosine; single-threaded CPU, one process per point. Smoke-scale: read slopes and orderings, not absolute bits per byte.

## Points

| curve | target | width | params | deployed MiB | disk MiB | train loss | eval loss | bpb | train step ms | train wall s | train CPU util | train peak RSS MiB | decode ms/B | decode CPU ms/B | prefill B/s | infer peak RSS MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 depth=2 | 0.5 | 128 | 0.46M | 0.87 | 1.76 | 3.231 | 3.211 | 4.632 | 37 | 0.8 | 1.03 | 68 | 0.76 | 0.78 | 54832 | 14 |
| fp16 depth=2 | 1.0 | 192 | 1.01M | 1.93 | 3.87 | 2.894 | 2.886 | 4.164 | 83 | 1.7 | 1.03 | 109 | 1.22 | 1.26 | 21002 | 11 |
| fp16 depth=2 | 2.0 | 256 | 1.77M | 3.38 | 6.77 | 2.802 | 2.774 | 4.002 | 107 | 2.1 | 1.06 | 138 | 1.49 | 1.55 | 22843 | 25 |
| fp16 depth=4 | 0.5 | 96 | 0.51M | 0.97 | 1.96 | 3.774 | 3.738 | 5.393 | 47 | 1.0 | 1.03 | 75 | 1.34 | 1.40 | 40621 | 5 |
| fp16 depth=4 | 1.0 | 128 | 0.88M | 1.68 | 3.39 | 3.279 | 3.263 | 4.708 | 89 | 1.8 | 1.04 | 121 | 1.86 | 1.91 | 23983 | 20 |
| fp16 depth=4 | 2.0 | 192 | 1.98M | 3.77 | 7.56 | 2.930 | 2.927 | 4.223 | 156 | 3.2 | 1.02 | 194 | 2.10 | 2.17 | 14591 | 11 |
| pocket loops=3 bpw=0.55 | 1.0 | 128 | 0.77M | 0.71 | 3.07 | 8.528 | 4.212 | 6.076 | 223 | 4.6 | 1.03 | 158 | 2.18 | 2.23 | 6351 | 8 |
| pocket loops=3 bpw=0.55 | 2.0 | 192 | 1.57M | 1.07 | 6.10 | 8.263 | 4.086 | 5.896 | 288 | 6.0 | 1.03 | 234 | 3.19 | 3.26 | 4130 | 12 |
| pocket loops=1 bpw=0.55 | 1.0 | 128 | 0.77M | 0.70 | 3.05 | 8.541 | 4.218 | 6.086 | 165 | 3.4 | 1.03 | 113 | 1.47 | 1.50 | 11629 | 8 |
| pocket loops=1 bpw=0.55 | 2.0 | 192 | 1.55M | 1.04 | 6.05 | 8.256 | 4.086 | 5.895 | 196 | 4.0 | 1.03 | 162 | 1.53 | 1.58 | 9852 | 12 |
| pocket loops=3 bpw=1.0 | 1.0 | 128 | 0.89M | 0.73 | 3.53 | 8.527 | 4.211 | 6.075 | 232 | 4.7 | 1.03 | 162 | 2.99 | 3.10 | 4791 | 9 |
| pocket loops=3 bpw=1.0 | 2.0 | 192 | 1.85M | 1.10 | 7.20 | 8.262 | 4.086 | 5.895 | 318 | 7.2 | 1.03 | 241 | 2.28 | 2.33 | 5452 | 14 |
| looped loops=3 bpw=0.55 | 0.5 | 128 | 0.58M | 0.35 | 2.32 | 3.397 | 3.397 | 4.901 | 709 | 12.4 | 1.02 | 339 | 10.18 | 10.43 | 1074 | 31 |
| looped loops=3 bpw=0.55 | 2.0 | 256 | 2.52M | 1.21 | 9.70 | 3.090 | 3.065 | 4.422 | 1309 | 24.9 | 1.03 | 790 | 10.69 | 10.95 | 929 | 76 |
| looped loops=1 bpw=0.55 | 0.5 | 128 | 0.57M | 0.33 | 2.29 | 3.422 | 3.421 | 4.935 | 419 | 8.4 | 1.03 | 191 | 4.48 | 4.62 | 2708 | 24 |
| looped loops=1 bpw=0.55 | 2.0 | 256 | 2.50M | 1.18 | 9.64 | 3.101 | 3.065 | 4.422 | 687 | 13.8 | 1.02 | 470 | 4.00 | 4.11 | 2118 | 70 |
| looped loops=3 bpw=1.0 | 2.0 | 192 | 1.67M | 0.75 | 6.47 | 3.172 | 3.158 | 4.556 | 1301 | 23.8 | 1.03 | 589 | 10.09 | 10.47 | 953 | 54 |
| oploop loops=4 | 0.5 | 128 | 0.61M | 1.16 | 2.34 | 3.282 | 3.294 | 4.753 | 800 | 16.1 | 1.04 | 354 | 30.70 | 31.67 | 979 | 320 |
| oploop loops=4 | 1.0 | 192 | 1.35M | 2.57 | 5.16 | 3.078 | 3.084 | 4.450 | 1046 | 22.0 | 1.04 | 499 | 77.89 | 80.21 | 717 | 467 |
| oploop loops=4 | 2.0 | 256 | 2.37M | 4.52 | 9.06 | 2.982 | 2.975 | 4.292 | 1510 | 30.0 | 1.03 | 667 | 154.08 | 157.95 | 410 | 610 |
| oploop loops=2 | 0.5 | 128 | 0.61M | 1.16 | 2.34 | 3.189 | 3.196 | 4.611 | 395 | 7.8 | 1.03 | 235 | 18.95 | 19.48 | 1880 | 307 |
| oploop loops=2 | 1.0 | 192 | 1.35M | 2.57 | 5.16 | 2.996 | 2.993 | 4.317 | 577 | 11.6 | 1.04 | 338 | 56.00 | 57.19 | 1331 | 456 |
| oploop loops=2 | 2.0 | 256 | 2.37M | 4.52 | 9.05 | 2.907 | 2.894 | 4.176 | 813 | 15.6 | 1.03 | 454 | 74.63 | 77.03 | 805 | 600 |

Out of range (no width within 35% of the target): pocket loops=3 bpw=0.55 @ 0.5 M params, pocket loops=1 bpw=0.55 @ 0.5 M params, pocket loops=3 bpw=1.0 @ 0.5 M params, looped loops=3 bpw=0.55 @ 1.0 M params, looped loops=1 bpw=0.55 @ 1.0 M params, looped loops=3 bpw=1.0 @ 0.5 M params, looped loops=3 bpw=1.0 @ 1.0 M params

## Fits (y = a * size^b, size in M params)

| curve | points | bits / byte (short run) | decode ms / byte | train step ms | prefill bytes / s | inference peak RSS growth MiB |
|---|---|---|---|---|---|---|
| fp16 depth=2 | 3 | b=-0.11 (r2 0.97) | b=+0.50 (r2 0.98) | b=+0.80 (r2 0.96) | b=-0.68 (r2 0.77) | b=+0.38 (r2 0.36) |
| fp16 depth=4 | 3 | b=-0.18 (r2 0.97) | b=+0.32 (r2 0.87) | b=+0.87 (r2 0.98) | b=-0.74 (r2 0.98) | b=+0.48 (r2 0.23) |
| pocket loops=3 bpw=0.55 | 2 | b=-0.04 (r2 1.00) | b=+0.54 (r2 1.00) | b=+0.36 (r2 1.00) | b=-0.61 (r2 1.00) | b=+0.61 (r2 1.00) |
| pocket loops=1 bpw=0.55 | 2 | b=-0.05 (r2 1.00) | b=+0.06 (r2 1.00) | b=+0.24 (r2 1.00) | b=-0.23 (r2 1.00) | b=+0.58 (r2 1.00) |
| pocket loops=3 bpw=1.0 | 2 | b=-0.04 (r2 1.00) | b=-0.37 (r2 1.00) | b=+0.43 (r2 1.00) | b=+0.18 (r2 1.00) | b=+0.64 (r2 1.00) |
| looped loops=3 bpw=0.55 | 2 | b=-0.07 (r2 1.00) | b=+0.03 (r2 1.00) | b=+0.42 (r2 1.00) | b=-0.10 (r2 1.00) | b=+0.60 (r2 1.00) |
| looped loops=1 bpw=0.55 | 2 | b=-0.07 (r2 1.00) | b=-0.08 (r2 1.00) | b=+0.33 (r2 1.00) | b=-0.17 (r2 1.00) | b=+0.73 (r2 1.00) |
| looped loops=3 bpw=1.0 | 1 | - | - | - | - | - |
| oploop loops=4 | 3 | b=-0.08 (r2 1.00) | b=+1.19 (r2 1.00) | b=+0.46 (r2 0.97) | b=-0.63 (r2 0.93) | b=+0.48 (r2 1.00) |
| oploop loops=2 | 3 | b=-0.07 (r2 0.99) | b=+1.03 (r2 0.95) | b=+0.53 (r2 1.00) | b=-0.61 (r2 0.96) | b=+0.49 (r2 1.00) |

## Crossovers against `fp16 depth=2`

Each cell: the curve / baseline ratio over the curve's measured sizes (baseline read off its fit), then the crossover size if the two fits meet within the measured range (2x margin), else whether it is better or worse at every measured size. For prefill, higher is better; for the rest, lower. Fits from 2 points always have r2 = 1; trust slopes from 3+ points and r2 near 1.

| curve | bits / byte (short run) | decode ms / byte | train step ms | prefill bytes / s | inference peak RSS growth MiB |
|---|---|---|---|---|---|
| fp16 depth=4 | x1.07-1.17; worse everywhere measured | x1.35-1.72; worse everywhere measured | x1.18-1.30; worse everywhere measured | x0.79-0.86; worse everywhere measured | x0.63-0.72; better everywhere measured |
| pocket loops=3 bpw=0.55 | x1.40-1.46; worse everywhere measured | x2.15-2.21; worse everywhere measured | x2.77-3.77; worse everywhere measured | x0.19-0.20; worse everywhere measured | x0.56-0.66; better everywhere measured |
| pocket loops=1 bpw=0.55 | x1.40-1.46; worse everywhere measured | x1.07-1.45; crosses at 1.8 M params, better above | x1.89-2.81; worse everywhere measured | x0.34-0.47; worse everywhere measured | x0.57-0.66; better everywhere measured |
| pocket loops=3 bpw=1.0 | x1.42-1.49; worse everywhere measured | x1.45-2.74; crosses at 2.8 M params, better above | x2.67-3.49; worse everywhere measured | x0.16-0.29; worse everywhere measured | x0.57-0.68; better everywhere measured |
| looped loops=3 bpw=0.55 | x1.09-1.16; worse everywhere measured | x5.86-11.57; worse everywhere measured | x8.60-15.09; worse everywhere measured | x0.03-0.06; worse everywhere measured | x2.42-3.37; worse everywhere measured |
| looped loops=1 bpw=0.55 | x1.10-1.16; worse everywhere measured | x2.20-5.13; worse everywhere measured | x4.53-9.02; worse everywhere measured | x0.07-0.14; worse everywhere measured | x1.86-3.11; worse everywhere measured |
| looped loops=3 bpw=1.0 | - | - | - | - | - |
| oploop loops=4 | x1.06-1.11; worse everywhere measured | x33.93-86.54; worse everywhere measured | x10.00-15.88; worse everywhere measured | x0.03-0.03; worse everywhere measured | x24.21-27.65; worse everywhere measured |
| oploop loops=2 | x1.03-1.08; worse everywhere measured | x22.71-46.96; worse everywhere measured | x5.51-7.97; worse everywhere measured | x0.05-0.05; worse everywhere measured | x23.30-27.26; worse everywhere measured |

![scaling](scaling_params.png)
