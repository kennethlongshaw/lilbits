LUT kernel SIMD width: 512 bits; torch 2.15.0+cu132

**One-token decode through the linears of 4 BitNet-2B4T-shaped blocks** (q, k, v, o, gate, up, down; fp32 weights exceed the cache)

| variant | threads | wall ms / token | CPU ms | 30-layer linears, est. ms / token (tok/s) | weight RAM | peak RSS growth |
|---|---|---|---|---|---|---|
| fp32 dense | 1 | 68.94 | 71.63 | 517.0 (2) | 1,060.0 MiB | 0.0 MiB |
| fp32 dense | 4 | 27.28 | 108.26 | 204.6 (5) | 1,060.0 MiB | 0.0 MiB |
| bf16 dense | 1 | 51.67 | 52.23 | 387.5 (3) | 530.0 MiB | 0.6 MiB |
| bf16 dense | 4 | 17.65 | 69.10 | 132.4 (8) | 530.0 MiB | 0.6 MiB |
| LittleBit 1.0 bpw unpack | 1 | 641.44 | 660.17 | 4810.8 (0) | 33.2 MiB | 25.5 MiB |
| LittleBit 1.0 bpw unpack | 4 | 207.32 | 835.38 | 1554.9 (1) | 33.2 MiB | 25.5 MiB |
| LittleBit 1.0 bpw LUT | 1 | 3.98 | 4.37 | 29.9 (33) | 33.4 MiB | 1.7 MiB |
| LittleBit 1.0 bpw LUT | 4 | 2.38 | 9.49 | 17.9 (56) | 33.4 MiB | 0.2 MiB |
| LittleBit 0.55 bpw LUT | 1 | 2.37 | 2.56 | 17.8 (56) | 18.5 MiB | 1.6 MiB |
| LittleBit 0.55 bpw LUT | 4 | 2.00 | 8.82 | 15.0 (67) | 18.5 MiB | 0.2 MiB |
| BitNet b1.58 unpack | 1 | 1017.52 | 1051.92 | 7631.4 (0) | 66.4 MiB | 222.0 MiB |
| BitNet b1.58 unpack | 4 | 339.74 | 1375.96 | 2548.1 (0) | 66.4 MiB | 222.0 MiB |
| BitNet b1.58 LUT | 1 | 13.33 | 13.70 | 100.0 (10) | 66.3 MiB | 0.2 MiB |
| BitNet b1.58 LUT | 4 | 4.73 | 19.10 | 35.5 (28) | 66.3 MiB | 0.2 MiB |
| BitNet b1.58 dense ternary (cached fp32) | 1 | 85.14 | 88.83 | 638.5 (2) | 2,120.0 MiB | 0.1 MiB |
| BitNet b1.58 dense ternary (cached fp32) | 4 | 33.14 | 134.17 | 248.6 (4) | 2,120.0 MiB | 0.1 MiB |

**Single layers**

**2560x2560, 1 token, 1 thread**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 1.184 | 1.00× | 1.224 | 0.2 MiB | 25.00 MiB | 25.00 MiB | 0.0e+00 |
| bf16 dense | 0.652 | 1.82× | 0.672 | 0.0 MiB | 12.50 MiB | 12.50 MiB | 3.5e-03 |
| LittleBit 1.0 bpw unpack | 16.320 | 0.07× | 18.680 | 7.0 MiB | 0.78 MiB | 0.79 MiB | 2.6e-04 |
| LittleBit 1.0 bpw LUT | 0.080 | 14.88× | 0.097 | 0.1 MiB | 0.79 MiB | 0.80 MiB | 2.6e-04 |
| LittleBit 0.55 bpw unpack | 8.761 | 0.14× | 9.154 | 7.4 MiB | 0.43 MiB | 0.43 MiB | 7.5e-05 |
| LittleBit 0.55 bpw LUT | 0.048 | 24.48× | 0.057 | 0.1 MiB | 0.44 MiB | 0.44 MiB | 7.5e-05 |
| BitNet b1.58 unpack | 35.462 | 0.03× | 36.993 | 62.5 MiB | 1.57 MiB | 1.56 MiB | 7.0e-05 |
| BitNet b1.58 LUT | 0.294 | 4.03× | 0.306 | 0.1 MiB | 1.56 MiB | 1.56 MiB | 7.0e-05 |
| BitNet b1.58 dense ternary (cached fp32) | 1.541 | 0.77× | 1.608 | 0.0 MiB | 50.00 MiB | 25.00 MiB | 0.0e+00 |

**2560x2560, 4 tokens, 1 thread**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 2.729 | 1.00× | 2.831 | 0.0 MiB | 25.00 MiB | 25.00 MiB | 0.0e+00 |
| bf16 dense | 1.021 | 2.67× | 1.112 | 0.2 MiB | 12.50 MiB | 12.50 MiB | 3.2e-03 |
| LittleBit 1.0 bpw unpack | 16.561 | 0.16× | 17.289 | 7.1 MiB | 0.78 MiB | 0.79 MiB | 2.6e-04 |
| LittleBit 1.0 bpw LUT | 0.253 | 10.77× | 0.271 | 0.1 MiB | 0.79 MiB | 0.80 MiB | 2.6e-04 |
| LittleBit 0.55 bpw unpack | 9.042 | 0.30× | 9.538 | 7.5 MiB | 0.43 MiB | 0.43 MiB | 7.5e-05 |
| LittleBit 0.55 bpw LUT | 0.150 | 18.21× | 0.163 | 0.1 MiB | 0.44 MiB | 0.44 MiB | 7.5e-05 |
| BitNet b1.58 unpack | 34.976 | 0.08× | 36.757 | 62.7 MiB | 1.57 MiB | 1.56 MiB | 7.0e-05 |
| BitNet b1.58 LUT | 0.533 | 5.12× | 0.553 | 0.3 MiB | 1.56 MiB | 1.56 MiB | 7.0e-05 |
| BitNet b1.58 dense ternary (cached fp32) | 1.981 | 1.38× | 2.042 | 0.2 MiB | 50.00 MiB | 25.00 MiB | 0.0e+00 |

**2560x2560, 16 tokens, 1 thread**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 2.935 | 1.00× | 3.066 | 0.2 MiB | 25.00 MiB | 25.00 MiB | 0.0e+00 |
| bf16 dense | 0.841 | 3.49× | 0.888 | 0.2 MiB | 12.50 MiB | 12.50 MiB | 2.9e-03 |
| LittleBit 1.0 bpw unpack | 12.937 | 0.23× | 13.305 | 9.2 MiB | 0.78 MiB | 0.79 MiB | 2.6e-04 |
| LittleBit 1.0 bpw LUT | 0.877 | 3.35× | 0.909 | 0.2 MiB | 0.79 MiB | 0.80 MiB | 2.6e-04 |
| LittleBit 0.55 bpw unpack | 7.118 | 0.41× | 7.336 | 8.1 MiB | 0.43 MiB | 0.43 MiB | 7.5e-05 |
| LittleBit 0.55 bpw LUT | 0.534 | 5.50× | 0.553 | 0.2 MiB | 0.44 MiB | 0.44 MiB | 7.5e-05 |
| BitNet b1.58 unpack | 30.538 | 0.10× | 31.813 | 63.1 MiB | 1.57 MiB | 1.56 MiB | 7.0e-05 |
| BitNet b1.58 LUT | 1.819 | 1.61× | 1.866 | 1.1 MiB | 1.56 MiB | 1.56 MiB | 7.0e-05 |
| BitNet b1.58 dense ternary (cached fp32) | 3.163 | 0.93× | 3.255 | 0.6 MiB | 50.00 MiB | 25.00 MiB | 0.0e+00 |

**2560x2560, 1 token, 4 threads**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 0.394 | 1.00× | 1.954 | 0.0 MiB | 25.00 MiB | 25.00 MiB | 0.0e+00 |
| bf16 dense | 0.209 | 1.89× | 0.804 | 0.0 MiB | 12.50 MiB | 12.50 MiB | 3.4e-03 |
| LittleBit 1.0 bpw unpack | 4.206 | 0.09× | 17.244 | 7.4 MiB | 0.78 MiB | 0.79 MiB | 2.6e-04 |
| LittleBit 1.0 bpw LUT | 0.070 | 5.63× | 0.121 | 0.1 MiB | 0.79 MiB | 0.80 MiB | 2.6e-04 |
| LittleBit 0.55 bpw unpack | 2.254 | 0.17× | 8.920 | 8.1 MiB | 0.43 MiB | 0.43 MiB | 7.5e-05 |
| LittleBit 0.55 bpw LUT | 0.065 | 6.05× | 0.114 | 0.1 MiB | 0.44 MiB | 0.44 MiB | 7.5e-05 |
| BitNet b1.58 unpack | 11.543 | 0.03× | 47.379 | 62.5 MiB | 1.57 MiB | 1.56 MiB | 7.0e-05 |
| BitNet b1.58 LUT | 0.080 | 4.91× | 0.282 | 0.1 MiB | 1.56 MiB | 1.56 MiB | 7.0e-05 |
| BitNet b1.58 dense ternary (cached fp32) | 0.475 | 0.83× | 2.023 | 0.0 MiB | 50.00 MiB | 25.00 MiB | 0.0e+00 |

**2560x2560, 4 tokens, 4 threads**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 0.720 | 1.00× | 3.118 | 0.0 MiB | 25.00 MiB | 25.00 MiB | 0.0e+00 |
| bf16 dense | 0.317 | 2.27× | 1.040 | 0.7 MiB | 12.50 MiB | 12.50 MiB | 3.6e-03 |
| LittleBit 1.0 bpw unpack | 4.299 | 0.17× | 16.958 | 6.8 MiB | 0.78 MiB | 0.79 MiB | 2.6e-04 |
| LittleBit 1.0 bpw LUT | 0.234 | 3.07× | 0.543 | 0.1 MiB | 0.79 MiB | 0.80 MiB | 2.6e-04 |
| LittleBit 0.55 bpw unpack | 2.406 | 0.30× | 9.620 | 7.7 MiB | 0.43 MiB | 0.43 MiB | 7.5e-05 |
| LittleBit 0.55 bpw LUT | 0.195 | 3.70× | 0.647 | 0.1 MiB | 0.44 MiB | 0.44 MiB | 7.6e-05 |
| BitNet b1.58 unpack | 12.031 | 0.06× | 47.771 | 62.7 MiB | 1.57 MiB | 1.56 MiB | 7.0e-05 |
| BitNet b1.58 LUT | 0.263 | 2.74× | 0.886 | 0.3 MiB | 1.56 MiB | 1.56 MiB | 7.0e-05 |
| BitNet b1.58 dense ternary (cached fp32) | 1.139 | 0.63× | 4.810 | 0.2 MiB | 50.00 MiB | 25.00 MiB | 0.0e+00 |

**2560x2560, 16 tokens, 4 threads**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 1.018 | 1.00× | 4.211 | 0.2 MiB | 25.00 MiB | 25.00 MiB | 0.0e+00 |
| bf16 dense | 0.308 | 3.31× | 1.026 | 0.7 MiB | 12.50 MiB | 12.50 MiB | 3.8e-03 |
| LittleBit 1.0 bpw unpack | 5.696 | 0.18× | 23.140 | 9.4 MiB | 0.78 MiB | 0.79 MiB | 2.6e-04 |
| LittleBit 1.0 bpw LUT | 0.909 | 1.12× | 3.540 | 0.2 MiB | 0.79 MiB | 0.80 MiB | 2.6e-04 |
| LittleBit 0.55 bpw unpack | 3.348 | 0.30× | 14.098 | 9.1 MiB | 0.43 MiB | 0.43 MiB | 7.5e-05 |
| LittleBit 0.55 bpw LUT | 0.958 | 1.06× | 3.941 | 0.2 MiB | 0.44 MiB | 0.44 MiB | 7.5e-05 |
| BitNet b1.58 unpack | 13.434 | 0.08× | 55.109 | 63.0 MiB | 1.57 MiB | 1.56 MiB | 7.0e-05 |
| BitNet b1.58 LUT | 0.966 | 1.05× | 4.046 | 0.9 MiB | 1.56 MiB | 1.56 MiB | 7.0e-05 |
| BitNet b1.58 dense ternary (cached fp32) | 1.425 | 0.71× | 5.794 | 0.6 MiB | 50.00 MiB | 25.00 MiB | 0.0e+00 |

**2560x6912, 1 token, 1 thread**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 3.793 | 1.00× | 3.958 | 0.0 MiB | 67.50 MiB | 67.50 MiB | 0.0e+00 |
| bf16 dense | 2.043 | 1.86× | 2.103 | 0.0 MiB | 33.75 MiB | 33.75 MiB | 3.4e-03 |
| LittleBit 1.0 bpw unpack | 41.666 | 0.09× | 43.013 | 21.4 MiB | 2.12 MiB | 2.12 MiB | 3.0e-04 |
| LittleBit 1.0 bpw LUT | 0.288 | 13.19× | 0.332 | 0.1 MiB | 2.12 MiB | 2.12 MiB | 3.0e-04 |
| LittleBit 0.55 bpw unpack | 21.926 | 0.17× | 22.547 | 22.2 MiB | 1.16 MiB | 1.16 MiB | 5.8e-04 |
| LittleBit 0.55 bpw LUT | 0.098 | 38.56× | 0.115 | 0.1 MiB | 1.17 MiB | 1.17 MiB | 5.8e-04 |
| BitNet b1.58 unpack | 98.405 | 0.04× | 101.871 | 168.8 MiB | 4.22 MiB | 4.22 MiB | 7.2e-05 |
| BitNet b1.58 LUT | 0.626 | 6.06× | 0.646 | 0.1 MiB | 4.22 MiB | 4.22 MiB | 7.2e-05 |
| BitNet b1.58 dense ternary (cached fp32) | 3.912 | 0.97× | 4.092 | 0.1 MiB | 135.00 MiB | 67.50 MiB | 0.0e+00 |

**2560x6912, 4 tokens, 1 thread**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 7.672 | 1.00× | 7.913 | 0.1 MiB | 67.50 MiB | 67.50 MiB | 0.0e+00 |
| bf16 dense | 2.659 | 2.89× | 2.798 | 0.2 MiB | 33.75 MiB | 33.75 MiB | 3.7e-03 |
| LittleBit 1.0 bpw unpack | 44.057 | 0.17× | 46.006 | 21.7 MiB | 2.12 MiB | 2.12 MiB | 3.0e-04 |
| LittleBit 1.0 bpw LUT | 0.850 | 9.02× | 0.891 | 0.2 MiB | 2.12 MiB | 2.12 MiB | 3.0e-04 |
| LittleBit 0.55 bpw unpack | 23.473 | 0.33× | 24.463 | 22.4 MiB | 1.16 MiB | 1.16 MiB | 5.8e-04 |
| LittleBit 0.55 bpw LUT | 0.389 | 19.74× | 0.401 | 0.2 MiB | 1.17 MiB | 1.17 MiB | 5.8e-04 |
| BitNet b1.58 unpack | 103.229 | 0.07× | 107.816 | 169.0 MiB | 4.22 MiB | 4.22 MiB | 7.2e-05 |
| BitNet b1.58 LUT | 1.799 | 4.27× | 1.865 | 0.4 MiB | 4.22 MiB | 4.22 MiB | 7.2e-05 |
| BitNet b1.58 dense ternary (cached fp32) | 8.220 | 0.93× | 8.760 | 0.2 MiB | 135.00 MiB | 67.50 MiB | 0.0e+00 |

**2560x6912, 16 tokens, 1 thread**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 11.632 | 1.00× | 11.893 | 0.4 MiB | 67.50 MiB | 67.50 MiB | 0.0e+00 |
| bf16 dense | 2.863 | 4.06× | 2.960 | 0.4 MiB | 33.75 MiB | 33.75 MiB | 4.9e-03 |
| LittleBit 1.0 bpw unpack | 46.832 | 0.25× | 47.974 | 21.9 MiB | 2.12 MiB | 2.12 MiB | 3.0e-04 |
| LittleBit 1.0 bpw LUT | 3.075 | 3.78× | 3.178 | 0.5 MiB | 2.12 MiB | 2.12 MiB | 3.0e-04 |
| LittleBit 0.55 bpw unpack | 24.678 | 0.47× | 25.438 | 21.5 MiB | 1.16 MiB | 1.16 MiB | 5.8e-04 |
| LittleBit 0.55 bpw LUT | 1.410 | 8.25× | 1.460 | 0.5 MiB | 1.17 MiB | 1.17 MiB | 5.8e-04 |
| BitNet b1.58 unpack | 107.991 | 0.11× | 111.488 | 170.0 MiB | 4.22 MiB | 4.22 MiB | 7.2e-05 |
| BitNet b1.58 LUT | 6.427 | 1.81× | 6.615 | 1.8 MiB | 4.22 MiB | 4.22 MiB | 7.2e-05 |
| BitNet b1.58 dense ternary (cached fp32) | 11.267 | 1.03× | 11.809 | 1.2 MiB | 135.00 MiB | 67.50 MiB | 0.0e+00 |

**2560x6912, 1 token, 4 threads**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 1.073 | 1.00× | 4.700 | 0.0 MiB | 67.50 MiB | 67.50 MiB | 0.0e+00 |
| bf16 dense | 0.638 | 1.68× | 2.492 | 0.0 MiB | 33.75 MiB | 33.75 MiB | 2.9e-03 |
| LittleBit 1.0 bpw unpack | 15.013 | 0.07× | 63.097 | 18.5 MiB | 2.12 MiB | 2.12 MiB | 3.0e-04 |
| LittleBit 1.0 bpw LUT | 0.111 | 9.63× | 0.155 | 0.1 MiB | 2.12 MiB | 2.12 MiB | 3.0e-04 |
| LittleBit 0.55 bpw unpack | 7.173 | 0.15× | 28.774 | 18.5 MiB | 1.16 MiB | 1.16 MiB | 5.8e-04 |
| LittleBit 0.55 bpw LUT | 0.092 | 11.62× | 0.135 | 0.1 MiB | 1.17 MiB | 1.17 MiB | 5.8e-04 |
| BitNet b1.58 unpack | 31.546 | 0.03× | 126.449 | 168.8 MiB | 4.22 MiB | 4.22 MiB | 7.2e-05 |
| BitNet b1.58 LUT | 0.150 | 7.17× | 0.760 | 0.1 MiB | 4.22 MiB | 4.22 MiB | 7.2e-05 |
| BitNet b1.58 dense ternary (cached fp32) | 1.261 | 0.85× | 5.499 | 0.1 MiB | 135.00 MiB | 67.50 MiB | 0.0e+00 |

**2560x6912, 4 tokens, 4 threads**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 2.090 | 1.00× | 8.198 | 0.1 MiB | 67.50 MiB | 67.50 MiB | 0.0e+00 |
| bf16 dense | 0.846 | 2.47× | 3.822 | 0.7 MiB | 33.75 MiB | 33.75 MiB | 3.7e-03 |
| LittleBit 1.0 bpw unpack | 14.460 | 0.14× | 59.446 | 18.3 MiB | 2.12 MiB | 2.12 MiB | 3.0e-04 |
| LittleBit 1.0 bpw LUT | 0.424 | 4.92× | 1.650 | 0.2 MiB | 2.12 MiB | 2.12 MiB | 3.0e-04 |
| LittleBit 0.55 bpw unpack | 6.396 | 0.33× | 26.136 | 18.3 MiB | 1.16 MiB | 1.16 MiB | 5.8e-04 |
| LittleBit 0.55 bpw LUT | 0.252 | 8.28× | 1.015 | 0.2 MiB | 1.17 MiB | 1.17 MiB | 5.8e-04 |
| BitNet b1.58 unpack | 32.514 | 0.06× | 131.280 | 169.1 MiB | 4.22 MiB | 4.22 MiB | 7.2e-05 |
| BitNet b1.58 LUT | 0.559 | 3.74× | 2.352 | 0.5 MiB | 4.22 MiB | 4.22 MiB | 7.2e-05 |
| BitNet b1.58 dense ternary (cached fp32) | 2.373 | 0.88× | 9.570 | 0.3 MiB | 135.00 MiB | 67.50 MiB | 0.0e+00 |

**2560x6912, 16 tokens, 4 threads**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 2.380 | 1.00× | 9.795 | 0.4 MiB | 67.50 MiB | 67.50 MiB | 0.0e+00 |
| bf16 dense | 0.732 | 3.25× | 3.163 | 0.9 MiB | 33.75 MiB | 33.75 MiB | 4.5e-03 |
| LittleBit 1.0 bpw unpack | 13.104 | 0.18× | 51.891 | 20.1 MiB | 2.12 MiB | 2.12 MiB | 3.0e-04 |
| LittleBit 1.0 bpw LUT | 1.325 | 1.80× | 5.502 | 0.5 MiB | 2.12 MiB | 2.12 MiB | 3.0e-04 |
| LittleBit 0.55 bpw unpack | 7.245 | 0.33× | 30.226 | 20.4 MiB | 1.16 MiB | 1.16 MiB | 5.8e-04 |
| LittleBit 0.55 bpw LUT | 1.074 | 2.22× | 4.634 | 0.5 MiB | 1.17 MiB | 1.17 MiB | 5.8e-04 |
| BitNet b1.58 unpack | 33.673 | 0.07× | 136.419 | 169.5 MiB | 4.22 MiB | 4.22 MiB | 7.2e-05 |
| BitNet b1.58 LUT | 1.704 | 1.40× | 7.068 | 1.5 MiB | 4.22 MiB | 4.22 MiB | 7.2e-05 |
| BitNet b1.58 dense ternary (cached fp32) | 2.989 | 0.80× | 12.687 | 0.9 MiB | 135.00 MiB | 67.50 MiB | 0.0e+00 |

**4096x4096, 1 token, 1 thread**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 3.500 | 1.00× | 3.611 | 0.0 MiB | 64.00 MiB | 64.00 MiB | 0.0e+00 |
| bf16 dense | 1.932 | 1.81× | 1.994 | 0.0 MiB | 32.00 MiB | 32.00 MiB | 3.3e-03 |
| LittleBit 1.0 bpw unpack | 42.536 | 0.08× | 43.626 | 11.3 MiB | 2.00 MiB | 2.01 MiB | 3.7e-05 |
| LittleBit 1.0 bpw LUT | 0.281 | 12.44× | 0.297 | 0.1 MiB | 2.02 MiB | 2.02 MiB | 3.6e-05 |
| LittleBit 0.55 bpw unpack | 18.570 | 0.19× | 19.636 | 13.4 MiB | 1.11 MiB | 1.11 MiB | 1.1e-04 |
| LittleBit 0.55 bpw LUT | 0.101 | 34.50× | 0.122 | 0.1 MiB | 1.13 MiB | 1.13 MiB | 1.1e-04 |
| BitNet b1.58 unpack | 85.651 | 0.04× | 88.109 | 160.1 MiB | 4.00 MiB | 4.00 MiB | 1.0e-04 |
| BitNet b1.58 LUT | 0.467 | 7.49× | 0.502 | 0.1 MiB | 4.00 MiB | 4.00 MiB | 1.0e-04 |
| BitNet b1.58 dense ternary (cached fp32) | 2.937 | 1.19× | 3.027 | 0.1 MiB | 128.00 MiB | 64.00 MiB | 0.0e+00 |

**4096x4096, 4 tokens, 1 thread**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 7.383 | 1.00× | 7.583 | 0.1 MiB | 64.00 MiB | 64.00 MiB | 0.0e+00 |
| bf16 dense | 2.587 | 2.85× | 2.670 | 0.3 MiB | 32.00 MiB | 32.00 MiB | 3.6e-03 |
| LittleBit 1.0 bpw unpack | 43.339 | 0.17× | 42.888 | 11.4 MiB | 2.00 MiB | 2.01 MiB | 3.6e-05 |
| LittleBit 1.0 bpw LUT | 0.836 | 8.83× | 0.879 | 0.2 MiB | 2.02 MiB | 2.02 MiB | 3.6e-05 |
| LittleBit 0.55 bpw unpack | 23.315 | 0.32× | 24.174 | 12.8 MiB | 1.11 MiB | 1.11 MiB | 1.1e-04 |
| LittleBit 0.55 bpw LUT | 0.393 | 18.79× | 0.417 | 0.1 MiB | 1.13 MiB | 1.13 MiB | 1.1e-04 |
| BitNet b1.58 unpack | 98.724 | 0.07× | 103.980 | 160.3 MiB | 4.00 MiB | 4.00 MiB | 1.0e-04 |
| BitNet b1.58 LUT | 1.731 | 4.27× | 1.812 | 0.6 MiB | 4.00 MiB | 4.00 MiB | 1.0e-04 |
| BitNet b1.58 dense ternary (cached fp32) | 7.616 | 0.97× | 7.851 | 0.2 MiB | 128.00 MiB | 64.00 MiB | 0.0e+00 |

**4096x4096, 16 tokens, 1 thread**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 10.703 | 1.00× | 11.269 | 0.3 MiB | 64.00 MiB | 64.00 MiB | 0.0e+00 |
| bf16 dense | 2.322 | 4.61× | 2.404 | 0.4 MiB | 32.00 MiB | 32.00 MiB | 3.0e-03 |
| LittleBit 1.0 bpw unpack | 38.653 | 0.28× | 39.422 | 12.1 MiB | 2.00 MiB | 2.01 MiB | 3.6e-05 |
| LittleBit 1.0 bpw LUT | 2.327 | 4.60× | 2.447 | 0.3 MiB | 2.02 MiB | 2.02 MiB | 3.6e-05 |
| LittleBit 0.55 bpw unpack | 17.980 | 0.60× | 18.759 | 13.0 MiB | 1.11 MiB | 1.11 MiB | 1.1e-04 |
| LittleBit 0.55 bpw LUT | 1.266 | 8.45× | 1.326 | 0.3 MiB | 1.13 MiB | 1.13 MiB | 1.1e-04 |
| BitNet b1.58 unpack | 91.820 | 0.12× | 95.525 | 160.8 MiB | 4.00 MiB | 4.00 MiB | 1.0e-04 |
| BitNet b1.58 LUT | 4.489 | 2.38× | 4.650 | 1.8 MiB | 4.00 MiB | 4.00 MiB | 1.0e-04 |
| BitNet b1.58 dense ternary (cached fp32) | 9.656 | 1.11× | 10.959 | 1.0 MiB | 128.00 MiB | 64.00 MiB | 0.0e+00 |

**4096x4096, 1 token, 4 threads**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 1.495 | 1.00× | 6.295 | 0.0 MiB | 64.00 MiB | 64.00 MiB | 0.0e+00 |
| bf16 dense | 0.853 | 1.75× | 3.319 | 0.0 MiB | 32.00 MiB | 32.00 MiB | 3.0e-03 |
| LittleBit 1.0 bpw unpack | 13.035 | 0.11× | 53.413 | 12.0 MiB | 2.00 MiB | 2.01 MiB | 3.6e-05 |
| LittleBit 1.0 bpw LUT | 0.130 | 11.52× | 0.347 | 0.1 MiB | 2.02 MiB | 2.02 MiB | 3.6e-05 |
| LittleBit 0.55 bpw unpack | 7.240 | 0.21× | 28.900 | 13.0 MiB | 1.11 MiB | 1.11 MiB | 1.1e-04 |
| LittleBit 0.55 bpw LUT | 0.118 | 12.69× | 0.401 | 0.1 MiB | 1.13 MiB | 1.13 MiB | 1.1e-04 |
| BitNet b1.58 unpack | 32.990 | 0.05× | 130.262 | 160.0 MiB | 4.00 MiB | 4.00 MiB | 1.0e-04 |
| BitNet b1.58 LUT | 0.157 | 9.51× | 0.177 | 0.1 MiB | 4.00 MiB | 4.00 MiB | 1.0e-04 |
| BitNet b1.58 dense ternary (cached fp32) | 1.273 | 1.17× | 5.353 | 0.1 MiB | 128.00 MiB | 64.00 MiB | 0.0e+00 |

**4096x4096, 4 tokens, 4 threads**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 2.124 | 1.00× | 9.237 | 0.1 MiB | 64.00 MiB | 64.00 MiB | 0.0e+00 |
| bf16 dense | 0.773 | 2.75× | 3.208 | 1.0 MiB | 32.00 MiB | 32.00 MiB | 3.3e-03 |
| LittleBit 1.0 bpw unpack | 13.899 | 0.15× | 52.779 | 11.6 MiB | 2.00 MiB | 2.01 MiB | 3.6e-05 |
| LittleBit 1.0 bpw LUT | 0.448 | 4.75× | 1.932 | 0.2 MiB | 2.02 MiB | 2.02 MiB | 3.6e-05 |
| LittleBit 0.55 bpw unpack | 6.653 | 0.32× | 26.347 | 12.1 MiB | 1.11 MiB | 1.11 MiB | 1.1e-04 |
| LittleBit 0.55 bpw LUT | 0.414 | 5.13× | 1.628 | 0.1 MiB | 1.13 MiB | 1.13 MiB | 1.1e-04 |
| BitNet b1.58 unpack | 38.269 | 0.06× | 153.138 | 160.3 MiB | 4.00 MiB | 4.00 MiB | 1.0e-04 |
| BitNet b1.58 LUT | 0.848 | 2.51× | 3.685 | 0.6 MiB | 4.00 MiB | 4.00 MiB | 1.0e-04 |
| BitNet b1.58 dense ternary (cached fp32) | 3.568 | 0.60× | 14.552 | 0.3 MiB | 128.00 MiB | 64.00 MiB | 0.0e+00 |

**4096x4096, 16 tokens, 4 threads**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 3.767 | 1.00× | 15.120 | 0.3 MiB | 64.00 MiB | 64.00 MiB | 0.0e+00 |
| bf16 dense | 0.792 | 4.75× | 3.297 | 1.1 MiB | 32.00 MiB | 32.00 MiB | 3.8e-03 |
| LittleBit 1.0 bpw unpack | 15.101 | 0.25× | 60.465 | 16.2 MiB | 2.00 MiB | 2.01 MiB | 3.6e-05 |
| LittleBit 1.0 bpw LUT | 1.611 | 2.34× | 6.550 | 0.3 MiB | 2.02 MiB | 2.02 MiB | 3.6e-05 |
| LittleBit 0.55 bpw unpack | 7.863 | 0.48× | 31.664 | 15.8 MiB | 1.11 MiB | 1.11 MiB | 1.1e-04 |
| LittleBit 0.55 bpw LUT | 1.371 | 2.75× | 6.164 | 0.3 MiB | 1.13 MiB | 1.13 MiB | 1.1e-04 |
| BitNet b1.58 unpack | 35.428 | 0.11× | 144.307 | 161.5 MiB | 4.00 MiB | 4.00 MiB | 1.0e-04 |
| BitNet b1.58 LUT | 1.972 | 1.91× | 7.889 | 1.5 MiB | 4.00 MiB | 4.00 MiB | 1.0e-04 |
| BitNet b1.58 dense ternary (cached fp32) | 3.511 | 1.07× | 13.986 | 1.0 MiB | 128.00 MiB | 64.00 MiB | 0.0e+00 |

**6912x2560, 1 token, 1 thread**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 2.550 | 1.00× | 2.651 | 0.0 MiB | 67.50 MiB | 67.50 MiB | 0.0e+00 |
| bf16 dense | 1.293 | 1.97× | 1.358 | 0.0 MiB | 33.75 MiB | 33.75 MiB | 2.9e-03 |
| LittleBit 1.0 bpw unpack | 37.824 | 0.07× | 39.931 | 22.5 MiB | 2.12 MiB | 2.12 MiB | 1.9e-04 |
| LittleBit 1.0 bpw LUT | 0.241 | 10.58× | 0.262 | 0.2 MiB | 2.13 MiB | 2.13 MiB | 1.9e-04 |
| LittleBit 0.55 bpw unpack | 20.343 | 0.13× | 20.978 | 21.7 MiB | 1.16 MiB | 1.16 MiB | 8.5e-05 |
| LittleBit 0.55 bpw LUT | 0.102 | 25.02× | 0.118 | 0.1 MiB | 1.18 MiB | 1.19 MiB | 8.5e-05 |
| BitNet b1.58 unpack | 88.910 | 0.03× | 96.049 | 168.9 MiB | 4.22 MiB | 4.22 MiB | 2.1e-04 |
| BitNet b1.58 LUT | 0.553 | 4.61× | 0.555 | 0.2 MiB | 4.22 MiB | 4.22 MiB | 2.1e-04 |
| BitNet b1.58 dense ternary (cached fp32) | 3.409 | 0.75× | 3.578 | 0.1 MiB | 135.00 MiB | 67.50 MiB | 0.0e+00 |

**6912x2560, 4 tokens, 1 thread**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 6.239 | 1.00× | 6.455 | 0.0 MiB | 67.50 MiB | 67.50 MiB | 0.0e+00 |
| bf16 dense | 2.148 | 2.90× | 2.209 | 0.4 MiB | 33.75 MiB | 33.75 MiB | 3.0e-03 |
| LittleBit 1.0 bpw unpack | 38.961 | 0.16× | 40.313 | 23.1 MiB | 2.12 MiB | 2.12 MiB | 1.9e-04 |
| LittleBit 1.0 bpw LUT | 0.764 | 8.17× | 0.799 | 0.2 MiB | 2.13 MiB | 2.13 MiB | 1.9e-04 |
| LittleBit 0.55 bpw unpack | 20.583 | 0.30× | 22.347 | 21.4 MiB | 1.16 MiB | 1.16 MiB | 8.5e-05 |
| LittleBit 0.55 bpw LUT | 0.405 | 15.42× | 0.419 | 0.2 MiB | 1.18 MiB | 1.19 MiB | 8.6e-05 |
| BitNet b1.58 unpack | 91.933 | 0.07× | 94.287 | 169.1 MiB | 4.22 MiB | 4.22 MiB | 2.1e-04 |
| BitNet b1.58 LUT | 1.343 | 4.65× | 1.427 | 0.8 MiB | 4.22 MiB | 4.22 MiB | 2.1e-04 |
| BitNet b1.58 dense ternary (cached fp32) | 5.160 | 1.21× | 5.306 | 0.4 MiB | 135.00 MiB | 67.50 MiB | 0.0e+00 |

**6912x2560, 16 tokens, 1 thread**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 8.442 | 1.00× | 8.510 | 0.2 MiB | 67.50 MiB | 67.50 MiB | 0.0e+00 |
| bf16 dense | 2.110 | 4.00× | 2.171 | 0.5 MiB | 33.75 MiB | 33.75 MiB | 3.0e-03 |
| LittleBit 1.0 bpw unpack | 34.430 | 0.25× | 35.451 | 21.5 MiB | 2.12 MiB | 2.12 MiB | 1.9e-04 |
| LittleBit 1.0 bpw LUT | 2.371 | 3.56× | 2.447 | 0.3 MiB | 2.13 MiB | 2.13 MiB | 1.9e-04 |
| LittleBit 0.55 bpw unpack | 18.472 | 0.46× | 19.123 | 19.8 MiB | 1.16 MiB | 1.16 MiB | 8.6e-05 |
| LittleBit 0.55 bpw LUT | 1.390 | 6.07× | 1.437 | 0.3 MiB | 1.18 MiB | 1.19 MiB | 8.5e-05 |
| BitNet b1.58 unpack | 111.767 | 0.08× | 113.952 | 170.6 MiB | 4.22 MiB | 4.22 MiB | 2.1e-04 |
| BitNet b1.58 LUT | 6.500 | 1.30× | 6.716 | 3.1 MiB | 4.22 MiB | 4.22 MiB | 2.1e-04 |
| BitNet b1.58 dense ternary (cached fp32) | 12.078 | 0.70× | 12.957 | 1.8 MiB | 135.00 MiB | 67.50 MiB | 0.0e+00 |

**6912x2560, 1 token, 4 threads**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 0.984 | 1.00× | 3.988 | 0.0 MiB | 67.50 MiB | 67.50 MiB | 0.0e+00 |
| bf16 dense | 0.646 | 1.52× | 2.585 | 0.0 MiB | 33.75 MiB | 33.75 MiB | 3.4e-03 |
| LittleBit 1.0 bpw unpack | 18.186 | 0.05× | 71.749 | 21.7 MiB | 2.12 MiB | 2.12 MiB | 1.9e-04 |
| LittleBit 1.0 bpw LUT | 0.147 | 6.68× | 0.168 | 0.2 MiB | 2.13 MiB | 2.13 MiB | 1.9e-04 |
| LittleBit 0.55 bpw unpack | 8.762 | 0.11× | 34.993 | 19.8 MiB | 1.16 MiB | 1.16 MiB | 8.5e-05 |
| LittleBit 0.55 bpw LUT | 0.145 | 6.80× | 0.176 | 0.1 MiB | 1.18 MiB | 1.19 MiB | 8.6e-05 |
| BitNet b1.58 unpack | 32.469 | 0.03× | 129.162 | 168.9 MiB | 4.22 MiB | 4.22 MiB | 2.1e-04 |
| BitNet b1.58 LUT | 0.234 | 4.20× | 0.787 | 0.3 MiB | 4.22 MiB | 4.22 MiB | 2.1e-04 |
| BitNet b1.58 dense ternary (cached fp32) | 1.271 | 0.77× | 5.010 | 0.1 MiB | 135.00 MiB | 67.50 MiB | 0.0e+00 |

**6912x2560, 4 tokens, 4 threads**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 2.083 | 1.00× | 8.819 | 0.0 MiB | 67.50 MiB | 67.50 MiB | 0.0e+00 |
| bf16 dense | 0.865 | 2.41× | 3.367 | 1.7 MiB | 33.75 MiB | 33.75 MiB | 3.2e-03 |
| LittleBit 1.0 bpw unpack | 16.387 | 0.13× | 65.928 | 21.0 MiB | 2.12 MiB | 2.12 MiB | 1.9e-04 |
| LittleBit 1.0 bpw LUT | 0.624 | 3.34× | 2.475 | 0.2 MiB | 2.13 MiB | 2.13 MiB | 1.9e-04 |
| LittleBit 0.55 bpw unpack | 8.863 | 0.24× | 35.853 | 19.0 MiB | 1.16 MiB | 1.16 MiB | 8.5e-05 |
| LittleBit 0.55 bpw LUT | 0.563 | 3.70× | 2.130 | 0.2 MiB | 1.18 MiB | 1.19 MiB | 8.5e-05 |
| BitNet b1.58 unpack | 35.926 | 0.06× | 144.842 | 169.1 MiB | 4.22 MiB | 4.22 MiB | 2.1e-04 |
| BitNet b1.58 LUT | 0.797 | 2.61× | 3.194 | 0.8 MiB | 4.22 MiB | 4.22 MiB | 2.1e-04 |
| BitNet b1.58 dense ternary (cached fp32) | 2.511 | 0.83× | 9.776 | 0.4 MiB | 135.00 MiB | 67.50 MiB | 0.0e+00 |

**6912x2560, 16 tokens, 4 threads**

| variant | wall ms | vs fp32 dense | CPU ms | peak RSS growth | weight RAM | disk | max rel err |
|---|---|---|---|---|---|---|---|
| fp32 dense | 3.049 | 1.00× | 12.034 | 0.2 MiB | 67.50 MiB | 67.50 MiB | 0.0e+00 |
| bf16 dense | 0.818 | 3.73× | 3.613 | 1.8 MiB | 33.75 MiB | 33.75 MiB | 3.4e-03 |
| LittleBit 1.0 bpw unpack | 19.041 | 0.16× | 77.769 | 20.1 MiB | 2.12 MiB | 2.12 MiB | 1.9e-04 |
| LittleBit 1.0 bpw LUT | 2.153 | 1.42× | 9.246 | 0.3 MiB | 2.13 MiB | 2.13 MiB | 1.9e-04 |
| LittleBit 0.55 bpw unpack | 9.011 | 0.34× | 36.280 | 19.7 MiB | 1.16 MiB | 1.16 MiB | 8.6e-05 |
| LittleBit 0.55 bpw LUT | 2.057 | 1.48× | 8.554 | 0.3 MiB | 1.18 MiB | 1.19 MiB | 8.6e-05 |
| BitNet b1.58 unpack | 36.733 | 0.08× | 148.794 | 170.6 MiB | 4.22 MiB | 4.22 MiB | 2.1e-04 |
| BitNet b1.58 LUT | 2.382 | 1.28× | 9.690 | 3.1 MiB | 4.22 MiB | 4.22 MiB | 2.1e-04 |
| BitNet b1.58 dense ternary (cached fp32) | 3.441 | 0.89× | 14.148 | 1.8 MiB | 135.00 MiB | 67.50 MiB | 0.0e+00 |
