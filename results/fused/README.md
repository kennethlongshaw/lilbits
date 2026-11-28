# Cutting inference RAM and CPU at the same loss (Example 14 FP16 baseline, d200 x 8 GDN layers)

Runtime: `src/lilbits/inference/fused.py` + the Rust engine `rust/lilbits_fused` (fearless_simd). Bench: `benchmarks/fused_bench.py`.
The tables below were measured with the original C kernel; the Rust engine that replaced it is compared at the end.

Baseline: Example 14's `fp16_size` model (width 200, 8 GDN layers, 1 head, 4.2M params), retrained with the same
settings and RNG (1000 steps x 8 x 128 B Shakespeare, seed 0): held-out bpb 2.3313, identical to the earlier run.
bpb64 = the same held-out sampler at 64 batches (64 KB, 8x the old 8 KB eval) for tighter comparisons.
Sapphire Rapids Xeon, 4 cores; each row is the median of 3 fresh processes; decode = 256 greedy bytes after a
512-byte prompt; CPU = process CPU time per byte.

| | bpb (8 KB) | bpb64 | decode ms/B | decode CPU ms/B | prefill B/s | prefill CPU ms/B | weights | recurrent state | process RSS |
|---|---|---|---|---|---|---|---|---|---|
| PyTorch FP32, 4 threads (baseline path) | 2.3313 | 2.3635 | 4.40 | 17.37 | 7,701 | 0.51 | 16.8 MB | 1.28 MB | 561 MiB |
| PyTorch FP32, 1 thread | 2.3313 | 2.3635 | 4.31 | 4.56 | 3,308 | 0.32 | 16.8 MB | 1.28 MB | 559 MiB |
| C runtime FP32 | 2.3313 | 2.3635 | 0.78 | 0.83 | 4,174 | 0.26 | 19.5 MB | 1.28 MB | 53 MiB |
| C runtime FP16 | 2.3313 | 2.3635 | 0.49 | 0.53 | 4,098 | 0.26 | 9.7 MB | 1.28 MB | 44 MiB |
| C runtime int8 weights | 2.3312 | 2.3634 | 0.35 | 0.38 | 4,414 | 0.24 | 5.0 MB | 1.28 MB | 39 MiB |
| **C runtime W8A8 (int8 x int8, VNNI)** | **2.3310** | **2.3630** | **0.28** | **0.30** | **8,331** | **0.13** | **5.0 MB** | 1.28 MB | **39 MiB** |
| C runtime int4 weights (group 32) | 2.3345 | 2.3659 | 0.42 | 0.45 | 4,543 | 0.23 | 3.0 MB | 1.28 MB | 37 MiB |
| **C W8A8 + 4 GDN heads (retrained)** | **2.2954** | **2.3472** | **0.22** | **0.24** | **13,228** | **0.08** | 5.0 MB | **0.32 MB** | 38 MiB |
| same, prefill on 4 threads | 2.2954 | 2.3472 | 0.21 | 0.27 | 16,172 | 0.25 | 5.0 MB | 0.32 MB | 38 MiB |

Recommended: W8A8 C runtime (same weights, same loss): decode 16x faster with 57x less CPU per byte, prefill
faster on 1 thread than PyTorch on 4, weights 3.3x smaller, process RSS 14x smaller (no torch import).
With 4 GDN heads instead of 1 (same params, retrain): 20x / 73x, and the recurrent state is 4x smaller.
Weights exclude ~0.3 MB padding (rows to 32, columns to 32). Peak RSS growth during prefill + decode is < 0.1 MiB
for every runtime (buffers preallocated); the older 12 MiB figure was first-call allocation.

## What changed
1. Fused native runtime (first C, now Rust; numpy + ctypes, no torch at inference). Each RMSNorm gain, the kernel-1
   "conv" scales and the GDN o_norm gain are folded into the weights; q/k/v/gate/a/b are one matrix and SwiGLU
   gate/up are one, so a layer is 4 matmuls + the recurrence. Decode is a C loop: no Python or dispatcher per op,
   which was most of PyTorch's 4.4 ms/byte (the matmuls are 4.2M MACs/byte).
2. Tile-major weights ([32-row tile][column][32]): matmuls stream weights once, no horizontal sums; 4 tokens
   share each weight load in prefill.
3. W8A8: int8 per-row weights x per-token int8 activations on AVX-512 VNNI (vpdpbusd). No loss change.
   int4 (MSE-searched group scales) costs +0.003 bpb, so it doesn't meet "same loss".
4. GDN recurrence reordered to one read pass (S^T k and S^T q together) and one update pass.
5. Architecture: GDN heads. 1 head of 200 was what `--head-dim 128` gave at width 200.

## Architecture sweep (bpb64, PyTorch, two seeds where run; seed noise ~0.01-0.03)
| d / heads | params | seed 0 | seed 1 |
|---|---|---|---|
| 200 / 1 (baseline) | 4.20M | 2.3635 (2.3619 other init) | 2.3366 |
| 200 / 2 | 4.20M | 2.3628 | |
| 200 / 4 | 4.21M | **2.3467** | **2.3357** |
| 200 / 8 | 4.22M | 2.3653 | |
| 176 / 4 | 3.26M | 2.3571 | 2.3638 |
| 160 / 4 | 2.71M | 2.3607 | 2.3717 |
| 144 / 4 | 2.21M | 2.3610 | |
4 heads is same-or-better on both seeds. Shrinking width looked free on seed 0 but costs 0.027-0.035 on seed 1,
so it doesn't reliably hold the loss; width stays 200.

## Correctness
`tests/test_fused.py`: FP32 logits match PyTorch to 1e-4 (1, 3 and 4 heads); greedy output equals `LeanLM.generate`
byte for byte, also with the prefill split in two; every quantized format stays within a small KL of FP32.
Quantized formats were checked on the trained model's bpb (tables).

## Limits
- Fast path needs AVX-512 (+VNNI for W8A8); other CPUs run the portable path (correct, slower matmuls).
- Batch 1; decode matmuls run on one thread; no speculative decoding (example 14 found it doesn't pay on CPU).
- Training is unchanged (PyTorch); `lilbits.inference.fused.export` converts a model to the runtime's arrays.

## Reproduce
    uv run python benchmarks/fused_bench.py train runs/base.pt                     # exact fp16_size baseline
    uv run python benchmarks/fused_bench.py train runs/h4.pt --variant 200,8,4     # 4 heads
    uv run python benchmarks/fused_bench.py export runs/h4.pt w8a8 runs/h4_w8a8.npz
    uv run python benchmarks/fused_bench.py suite "PyTorch 4t|torch|runs/base.pt|4" "W8A8|fused|runs/h4_w8a8.npz|1"
Example 14 now gives `fp16_size` at least 4 heads (`--min-heads 4`; `--min-heads 1` for the old 1-head baseline).

## Scaled to width 512 (8 layers, 27.4M params; same data and schedule: 1000 steps x 8 x 128 B, seed 0)
Medians of 3 fresh processes. Weights: PyTorch FP32 vs packed runtime file.

| | bpb (8 KB) | bpb64 | decode ms/B | decode CPU ms/B | prefill B/s | prefill CPU ms/B | weights | state | process RSS |
|---|---|---|---|---|---|---|---|---|---|
| 1 head, PyTorch FP32 4 threads | 2.3490 | 2.3882 | 10.46 | 41.3 | 785 | 4.99 | 109 MB | 8.4 MB | 680 MiB |
| 4 heads, PyTorch FP32 4 threads | 2.2988 | 2.3442 | 9.09 | 35.5 | 1,991 | 1.97 | 109 MB | 2.1 MB | 675 MiB |
| 4 heads, PyTorch FP32 1 thread | 2.2988 | 2.3442 | 19.22 | 21.0 | 928 | 1.20 | 109 MB | 2.1 MB | 678 MiB |
| 4 heads, C FP16 | 2.2988 | 2.3442 | 2.88 | 3.21 | 744 | 1.52 | 55 MB | 2.1 MB | 89 MiB |
| 4 heads, C int8 weights | 2.2988 | 2.3443 | 2.01 | 2.22 | 667 | 1.65 | 28 MB | 2.1 MB | 63 MiB |
| **4 heads, C W8A8** | 2.2996 | 2.3453 | **1.53** | **1.70** | 2,078 | **0.52** | **28 MB** | 2.1 MB | **63 MiB** |
| 4 heads, C W8A8, prefill 4 threads | 2.2996 | 2.3453 | 1.48 | 1.70 | 3,515 | 1.13 | 28 MB | 2.1 MB | 63 MiB |
| 1 head, C W8A8 | 2.3498 | 2.3896 | 2.21 | 2.39 | 1,228 | 0.90 | 28 MB | 8.4 MB | 69 MiB |

- 4 heads vs 1 at d512: -0.044 bpb64, 4x smaller state, faster. The head change holds (stronger) at scale.
- W8A8 costs +0.001 bpb at this width (was 0.000 at d200). int8 weight-only is exact (+0.0001) if that matters.
- C W8A8 vs the PyTorch 4-thread path: 5.9x faster decode, 21x less CPU per byte, 3.9x smaller weights,
  10.7x smaller process. Smaller ratio than d200 because PyTorch's per-op overhead is amortized over bigger
  matmuls; decode is now bound by streaming 28 MB of weights per byte through one core.
- d512 barely beats d200 at this training budget (bpb64 2.3442 vs 2.3467 with 4 heads): ~1 MB of training
  bytes is the limit, not model size. A bigger model needs more training tokens to pay off.
- Next speedups at this size: multi-thread decode matmuls (the runtime keeps decode on 1 thread), int4 for
  some layers if +0.003 bpb is acceptable.

## C kernel vs Rust engine (fearless_simd)
Same exported files, same benchmark, 3 fresh processes each (medians), run back to back; "pf4" = prefill on 4
threads. Each cell is C / Rust. Greedy text is byte-identical in every row.

| config | bpb64 | decode ms/B | decode CPU ms/B | prefill B/s | process RSS MiB |
|---|---|---|---|---|---|
| d200h4 W8A8 | 2.3472 / 2.3472 | 0.327 / 0.300 | 0.370 / 0.343 | 7,905 / 12,586 | 40 / 41 |
| d200h4 W8A8 pf4 | 2.3472 / 2.3472 | 0.292 / 0.259 | 0.390 / 0.287 | 12,044 / 7,798 | 41 / 42 |
| d200 f32 | 2.3635 / 2.3635 | 0.990 / 1.201 | 1.098 / 1.308 | 2,504 / 3,199 | 55 / 55 |
| d200 i8 | 2.3634 / 2.3634 | 0.425 / 0.480 | 0.460 / 0.531 | 3,849 / 4,432 | 41 / 41 |
| d512h4 W8A8 | 2.3453 / 2.3452 | 1.526 / 1.602 | 1.682 / 1.804 | 2,189 / 2,445 | 65 / 65 |
| d512h4 W8A8 pf4 | 2.3453 / 2.3452 | 1.578 / 1.543 | 1.773 / 1.730 | 3,253 / 2,631 | 65 / 69 |

Within run-to-run noise on this shared VM (the same C config measured 0.25-0.33 ms/B across sessions), Rust is at
parity: ahead on d200 W8A8, ~10-20% behind on the f32 / int8 rows and on 4-thread prefill (rayon vs OpenMP), so it
replaced the C kernel. What it took to get there: activation quantization and SiLU's exp() as explicit SIMD
(the C build got vector exp and vectorized reductions from -ffast-math; Rust has no fast-math). The Rust engine also
runs on CPUs without AVX-512 (portable path, verified by forcing it with `--cfg disable_dispatch_avx512`).

## Cache behaviour
`perf stat` can't read hardware counters here (Firecracker VM: "no PMU driver, software events only"), so:

Single-core read bandwidth on this Xeon (AVX-512 loads, `bw.c`-style microbenchmark):
| buffer | 64 KB | 256 KB-1 MB | 1.5 MB | 2 MB | 3-64 MB | 256 MB+ |
|---|---|---|---|---|---|---|
| GB/s | 117 | 137-140 | 111 | 55 | 24-31 | 12.5 |
L2 is 2 MB per core (fast up to ~1.5 MB), L3 260 MB shared.

- Decode streams every weight once per byte. d200 W8A8 = 5.0 MB in ~0.21 ms of matmul = ~24 GB/s: exactly the
  L3 rate. d512 (28 MB) runs at ~19 GB/s. Decode is L3-bandwidth bound, not compute bound.
- Cachegrind (D1 48 KB, LL = one 2 MB L2): 16 extra decode bytes add 1.30M LL read misses = 81K x 64 B = 5.2 MB per
  byte, i.e. the whole weight set misses L2 every step. The recurrent state (0.32 MB with 4 heads) and activations
  stay cached: no other misses.
- Same run with LL = 8 MB (the 4 cores' L2s together): 16 extra decode bytes add ~0 misses.
So the next speedup is splitting the decode matmul rows across the 4 cores with persistent pinned threads, so each
core keeps its 1.25 MB slice of the d200 weights in its own L2 (~110+ GB/s instead of 24).
