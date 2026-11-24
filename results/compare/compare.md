| | FP16 d128 (full size) | LittleBit 1.40 bpw | BitNet d128 | BitNet d112 | FP16 d40 | LittleBit 0.79 bpw | BitNet d96 | LittleBit 0.55 bpw (1 path) | BitNet d80 | FP16 d32 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Quality** |  |  |  |  |  |  |  |  |  |  |
| Eval loss, mean ± sd of 3 seeds | 0.957 ± 0.019 | 1.004 ± 0.018 | 0.995 ± 0.021 | 1.004 ± 0.018 | 0.988 ± 0.021 | 1.081 ± 0.021 | 1.020 ± 0.019 | 1.205 ± 0.033 | 1.053 ± 0.013 | 1.009 ± 0.022 |
| Eval loss, seed 0 (this run) | 0.978 | 1.025 | 1.019 | 1.024 | 1.011 | 1.103 | 1.041 | 1.231 | 1.068 | 1.034 |
| Final train loss, seed 0 | 1.014 | 1.065 | 1.060 | 1.078 | 1.053 | 1.149 | 1.069 | 1.294 | 1.103 | 1.086 |
| **Size** |  |  |  |  |  |  |  |  |  |  |
| d_model / block bpw | 128 / 16.00 | 128 / 1.40 | 128 / 1.59 | 112 / 1.59 | 40 / 16.00 | 128 / 0.79 | 96 / 1.59 | 128 / 0.55 | 80 / 1.59 | 32 / 16.00 |
| Trainable params | 412,288 | 410,256 | 412,288 | 314,160 | 43,080 | 177,312 | 233,952 | 155,404 | 165,520 | 29,344 |
| Stored block linears | 772.0 KiB | 67.7 KiB | 76.5 KiB | 57.9 KiB | 73.8 KiB | 38.1 KiB | 42.8 KiB | 26.5 KiB | 30.0 KiB | 49.0 KiB |
| Stored whole model | 805.2 KiB | 101.0 KiB | 109.8 KiB | 87.0 KiB | 84.1 KiB | 71.4 KiB | 67.8 KiB | 59.7 KiB | 50.8 KiB | 57.3 KiB |
| vs FP16 d128 (whole model) | 1.0x smaller | 8.0x smaller | 7.3x smaller | 9.3x smaller | 9.6x smaller | 11.3x smaller | 11.9x smaller | 13.5x smaller | 15.9x smaller | 14.1x smaller |
| **Training** (1,500 steps, 3.07M tokens, 1 CPU thread) |  |  |  |  |  |  |  |  |  |  |
| Optimizer | muon | muon_ds, 600 FP warm-up steps | muon_ds | muon_ds | muon | muon_ds, 600 FP warm-up steps | muon_ds | muon_ds, 600 FP warm-up steps | muon_ds | muon |
| Wall time / CPU time | 129 s / ≈ wall (1 thread) | 153 s / ≈ wall (1 thread) | 203 s / ≈ wall (1 thread) | 171 s / ≈ wall (1 thread) | 43 s / ≈ wall (1 thread) | 130 s / ≈ wall (1 thread) | 139 s / ≈ wall (1 thread) | 124 s / ≈ wall (1 thread) | 113 s / ≈ wall (1 thread) | 39 s / ≈ wall (1 thread) |
| ms per step | 86 | 102 | 136 | 114 | 29 | 86 | 92 | 82 | 76 | 26 |
| Steps / seconds to eval loss ≤ 1.1 | 450 / 39 s | 500 / 51 s | 850 / 115 s | 1000 / 114 s | 900 / 26 s | 500 / 43 s | 1100 / 102 s | 500 / 41 s | 1200 / 91 s | 1050 / 27 s |
| Peak RSS growth while training (sampled) | 98 MiB | 116 MiB | 144 MiB | 125 MiB | 44 MiB | 110 MiB | 104 MiB | 103 MiB | 93 MiB | 40 MiB |
| Weights + grads + optimizer state | 4,898.0 KiB | 4,916.2 KiB | 4,898.0 KiB | 3,739.8 KiB | 525.6 KiB | 2,184.0 KiB | 2,791.5 KiB | 1,908.1 KiB | 1,981.2 KiB | 360.5 KiB |
| **Inference, fast path** (FP32 weights / cached factors) |  |  |  |  |  |  |  |  |  |  |
| Weights in RAM | 1,610.5 KiB | 3,066.6 KiB | 3,154.5 KiB | 2,396.2 KiB | 168.3 KiB | 1,279.1 KiB | 1,777.9 KiB | 1,127.2 KiB | 1,251.6 KiB | 114.6 KiB |
| state_dict file (training format) | 1,618.0 KiB | 1,647.8 KiB | 1,618.0 KiB | 1,234.7 KiB | 175.7 KiB | 737.8 KiB | 921.4 KiB | 632.0 KiB | 654.0 KiB | 122.1 KiB |
| Batch 32×64: ms (tokens/s) | 14.6 (139,920) | 17.7 (115,822) | 37.6 (54,516) | 32.1 (63,768) | 4.8 (426,834) | 13.9 (147,573) | 22.4 (91,264) | 12.4 (165,710) | 21.6 (94,887) | 3.7 (553,384) |
| 1×64 sequence: ms | 0.80 | 1.05 | 1.78 | 1.51 | 0.34 | 0.72 | 1.30 | 0.66 | 1.25 | 0.33 |
| 1 token: ms | 0.28 | 0.36 | 0.75 | 0.72 | 0.22 | 0.31 | 0.63 | 0.31 | 0.67 | 0.24 |
| Peak RSS growth, batch 32×64 | 17.1 MiB | 14.8 MiB | 22.8 MiB | 18.3 MiB | 4.1 MiB | 15.6 MiB | 19.4 MiB | 14.8 MiB | 13.3 MiB | 3.1 MiB |
| **Inference, packed** (LittleBit: 1-bit signs; BitNet: 2-bit codes) |  |  |  |  |  |  |  |  |  |  |
| Weights in RAM | n/a | 136.7 KiB | 219.0 KiB | 187.3 KiB | n/a | 110.9 KiB | 159.9 KiB | 96.5 KiB | 135.4 KiB | n/a |
| state_dict file | n/a | 159.9 KiB | 174.6 KiB | 142.6 KiB | n/a | 134.2 KiB | 115.4 KiB | 119.5 KiB | 90.7 KiB | n/a |
| Batch 32×64: ms | n/a | 25.0 | 32.9 | 27.2 | n/a | 19.6 | 19.0 | 16.3 | 19.9 | n/a |
| 1×64 sequence: ms | n/a | 3.04 | 2.71 | 2.22 | n/a | 2.62 | 1.81 | 1.70 | 1.68 | n/a |
| 1 token: ms | n/a | 2.17 | 1.50 | 1.48 | n/a | 1.80 | 1.16 | 1.20 | 1.14 | n/a |
| Peak RSS growth, batch 32×64 | n/a | 15.3 MiB | 20.3 MiB | 19.6 MiB | n/a | 14.8 MiB | 15.4 MiB | 15.1 MiB | 12.9 MiB | n/a |
