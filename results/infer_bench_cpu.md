## TinyLM forward (d_model 128, 2 blocks), cpu, 4 threads

| model | input | ms |
|---|---|---|
| FP | prefill 32x64 | 8.59 |
| FP | decode-like 1x1 | 0.28 |
| LittleBit 1.41 bpw (old: rebuild factors each call) | prefill 32x64 | 15.53 |
| LittleBit 1.41 bpw (old: rebuild factors each call) | decode-like 1x1 | 3.63 |
| LittleBit 1.41 bpw | prefill 32x64 | 9.09 |
| LittleBit 1.41 bpw | decode-like 1x1 | 0.30 |
| LittleBit 1.0 bpw (old: rebuild factors each call) | prefill 32x64 | 11.13 |
| LittleBit 1.0 bpw (old: rebuild factors each call) | decode-like 1x1 | 2.66 |
| LittleBit 1.0 bpw | prefill 32x64 | 8.73 |
| LittleBit 1.0 bpw | decode-like 1x1 | 0.35 |

## Single layer, cpu, 4 threads

| in x out | tokens | LittleBit bpw (rank/path) | path | MACs vs dense | nn.Linear ms | LittleBit ms | speedup |
|---|---|---|---|---|---|---|---|
| 1024x1024 | 1 | 1.00 (238) | lowrank | 0.93x | 0.022 | 0.030 | 0.72x |
| 1024x1024 | 1 | 0.55 (123) | lowrank | 0.48x | 0.022 | 0.017 | 1.24x |
| 1024x1024 | 1 | 0.10 (9) | lowrank | 0.04x | 0.022 | 0.013 | 1.67x |
| 1024x1024 | 2048 | 1.00 (238) | lowrank | 0.93x | 6.632 | 6.539 | 1.01x |
| 1024x1024 | 2048 | 0.55 (123) | lowrank | 0.48x | 6.632 | 3.581 | 1.85x |
| 1024x1024 | 2048 | 0.10 (9) | lowrank | 0.04x | 6.632 | 0.479 | 13.85x |
| 4096x4096 | 1 | 1.00 (1006) | lowrank | 0.98x | 0.773 | 0.778 | 0.99x |
| 4096x4096 | 1 | 0.55 (546) | lowrank | 0.53x | 0.773 | 0.452 | 1.71x |
| 4096x4096 | 1 | 0.10 (86) | lowrank | 0.08x | 0.773 | 0.036 | 21.71x |
| 4096x4096 | 2048 | 1.00 (1006) | lowrank | 0.98x | 120.450 | 124.656 | 0.97x |
| 4096x4096 | 2048 | 0.55 (546) | lowrank | 0.53x | 120.450 | 71.658 | 1.68x |
| 4096x4096 | 2048 | 0.10 (86) | lowrank | 0.08x | 120.450 | 15.569 | 7.74x |
| 4096x11008 | 1 | 1.00 (1475) | lowrank | 0.99x | 2.620 | 2.592 | 1.01x |
| 4096x11008 | 1 | 0.55 (804) | lowrank | 0.54x | 2.620 | 1.095 | 2.39x |
| 4096x11008 | 1 | 0.10 (133) | lowrank | 0.09x | 2.620 | 0.216 | 12.12x |
| 4096x11008 | 2048 | 1.00 (1475) | lowrank | 0.99x | 286.643 | 299.119 | 0.96x |
| 4096x11008 | 2048 | 0.55 (804) | lowrank | 0.54x | 286.643 | 174.881 | 1.64x |
| 4096x11008 | 2048 | 0.10 (133) | lowrank | 0.09x | 286.643 | 40.831 | 7.02x |
