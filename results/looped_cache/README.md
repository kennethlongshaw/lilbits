# Looped depth that keeps its weights in cache (example 15)

Question: looped (weight-shared) depth lost on CPU decode because every loop re-streamed weights from L3, and decode
is weight-bandwidth bound. If a looped block's unique weights fit in L2 and its loops run back to back per byte,
do loops 2..R run at L2 speed, and does that make looped depth beat a plain GDN stack of the same quality?

Short answer: **the cache fix works, but looped depth still loses at matched quality.** Sharing one weight copy
across loops makes the same looped model up to 2x faster per byte on one thread when its unrolled weights don't fit
in L2. But a plain stack of the same held-out bpb is always cheaper still, because it has 2-4x fewer MACs per byte, and
at the 20 bytes/param training ratio every model this CPU can train fits in L2 anyway. Looping does buy quality per
*stored* parameter at that ratio, so it helps only where weight bytes are the constraint (flash, download, RAM), and
never decode speed or CPU per byte.

Setup: `examples/15_looped_cache_lm.py`. LeanLM GDN blocks, 4 heads; `loop:D,K,R` = K unique blocks applied R times
per byte, each application with its own GDN state; `plain:D,L` = the plain stack. Inference is the fused Rust engine
(W8A8). Looped layers' weight pointers alias one copy (**hot**), or each loop gets its own copy (**cold**, the same math
and identical greedy text, only the memory differs), so hot vs cold isolates the cache effect. No engine changes:
`FusedLM` layers are already pointers into numpy arrays. Decode = 2048 greedy bytes after a 512-byte prompt; medians
of 7 fresh processes run round-robin; CPU = process CPU per byte; RSS = whole process (numpy + ctypes, no torch).
Intel Xeon (Sapphire Rapids class), 4 vCPUs, 2 MB L2 per core.

## 1. The cache effect (sweep at ~1 byte/param, 1000 steps x 8 x 128 B Shakespeare)

Unrolled weights 5-11 MB, so cold reads come from L3. Decode ms/B, hot / cold:

| model | unique MB | streamed MB/B | 1 thread | 2 threads | 4 threads |
|---|---|---|---|---|---|
| loop:200,2,4 | 1.35 | 5.03 | **0.138** / 0.256 | 0.132 / 0.190 | 0.142 / 0.168 |
| loop:200,1,8 | 0.73 | 5.03 | **0.142** / 0.293 | 0.121 / 0.189 | 0.139 / 0.155 |
| loop:200,4,2 | 2.57 | 5.03 | 0.229 / 0.284 | **0.134** / 0.186 | 0.140 / 0.161 |
| loop:256,2,4 | 1.92 | 7.27 | 0.237 / 0.375 | **0.144** / 0.248 | 0.161 / 0.215 |
| loop:256,1,8 | 1.03 | 7.27 | **0.179** / 0.391 | 0.145 / 0.273 | 0.164 / 0.217 |
| loop:320,2,4 | 2.91 | 11.13 | 0.522 / 0.593 | **0.243** / 0.364 | 0.207 / 0.291 |

- When the unique block fits one core's L2 (≤ ~1.5 MB), sharing gives 1.9-2.2x on one thread: 5 MB per byte at
  ~36 GB/s effective instead of ~20 GB/s. The weights do stay hot across loops.
- Unique blocks bigger than one L2 (2.6-2.9 MB) gain little on 1 thread and a lot on 2 threads: the lockstep threads
  each own the same rows in every block application, so the block splits across the cores' L2s.
- At 4 threads the gap mostly closes for 5 MB models: the plain unrolled weights already fit in 4 x 2 MB of L2.
- CPU/B follows decode time; RSS drops by the copies not made (35 vs 39 MiB at 5 MB, 38 vs 46 MiB at 11 MB).

But quality (bpb64, held out) at this ratio, against plain stacks:

| model | unique params | bpb64 | decode ms/B, best thread count |
|---|---|---|---|
| plain:200,8 | 4.21M | **2.344** | 0.162 (4t) |
| plain:160,8 | 2.71M | 2.361 | 0.110 (2t) |
| plain:128,8 | 1.73M | 2.382 | **0.079** (2t) |
| loop:256,2,4 | 1.77M | 2.388 | 0.144 (2t, hot) |
| loop:200,4,2 | 2.13M | 2.392 | 0.134 (2t, hot) |
| plain:200,4 | 2.13M | 2.398 | 0.064 (2t) |
| loop:320,2,4 | 2.74M | 2.407 | 0.207 (4t, hot) |
| plain:200,2 | 1.09M | 2.409 | 0.030 (1t) |
| loop:200,2,4 | 1.09M | 2.430 | 0.121 (2t, hot) |
| loop:256,1,8 | 0.92M | 2.441 | 0.145 (2t, hot) |
| loop:200,1,8 | 0.57M | 2.476 | 0.121 (2t, hot) |

All of these models are undertrained (0.25-1.8 bytes/param). Every looped model is beaten by a plain one that is both
better and faster: plain:128,8 beats the best looped model (loop:256,2,4) on bpb and runs 1.8x faster. At equal unique
params, looping 4x was *worse* than not looping (loop:200,2,4 2.430 vs plain:200,2 2.409).

## 2. At 20 bytes per parameter (Chinchilla's 20 tokens/param, bytes as tokens)

Each model trains on 20 x its unique params bytes (8 x 128 B batches; steps = 20 P / 1024), so the looped and plain
models in a pair see the same data. Two corpora: Shakespeare (1.03 MB train, so 1.4-5.2 epochs), and the ~10 MB mixed
corpus (`lilbits.corpus`: Shakespeare + Gutenberg + webtext, 0.25-0.54 epochs, no repeats). The ratio caps models at
~270K params here, so every model's weights (<= 1 MB unrolled) sit in L2 hot or cold, and hot vs cold is within noise.

bpb64 (held out), with decode on 1 thread (the fastest at this size; more threads only add hand-off cost):

| model | params | MACs/B | Shakespeare bpb64 | mixed bpb64 | decode ms/B | CPU ms/B |
|---|---|---|---|---|---|---|
| plain:128,1 | 246K | 0.25M | 2.413 | 2.471 | **0.007-0.009** | 0.007-0.009 |
| loop:128,1,4 | 246K | 0.88M | 2.284 | 2.345 | 0.024 | 0.024 |
| plain:96,2 | 267K | 0.26M | 2.304 | 2.349 | 0.009-0.011 | 0.009-0.011 |
| loop:96,2,2 | 267K | 0.50M | **2.257** | **2.318** | 0.017-0.019 | 0.017-0.019 |
| plain:64,4 | 231K | 0.23M | 2.287 | 2.353 | 0.010-0.013 | 0.010-0.013 |
| plain:64,2 | 124K | 0.12M | 2.411 | 2.525 | 0.005 | 0.005 |
| loop:64,2,4 | 124K | 0.44M | 2.432 | 2.540 | 0.020-0.023 | 0.020-0.023 |
| loop:80,2,4 | 187K | 0.68M | 2.323 | | 0.030 | 0.030 |
| loop:96,1,4 | 146K | 0.50M | 2.406 | | 0.017 | 0.017 |
| loop:64,1,8 | 70K | 0.44M | 2.613 | | 0.020 | 0.020 |

(MACs/B = block applications x ~13 d^2 + 256 d for the head. Decode ranges are the two sweeps.)

- At this ratio looping helps per stored parameter, on both corpora: loop:128,1,4 beats plain:128,1 by 0.13 bpb,
  loop:96,2,2 beats plain:96,2 by 0.03-0.05. Narrow width is the exception: loop:64,2,4 is worse than plain:64,2.
- At matched quality, plain still wins on speed: plain:64,4 ties loop:128,1,4 (2.287 vs 2.284; 2.353 vs 2.345) at
  2-2.5x less decode time and CPU, because it has ~4x fewer MACs. The best looped model, loop:96,2,2, is 0.03 bpb
  better than plain:64,4 at ~1.7x the decode cost. Seed noise on bpb64 here was 0.01-0.03 (results/fused), so that
  0.03 is at the edge of noise; one seed was run.
- RSS is 33-34 MiB for every model at this size; the weights are < 1 MB.

## What this means for the flagship
- If the constraint is **decode speed or CPU per byte**, don't loop: at any quality we measured, a plain GDN stack is
  cheaper. Shared weights fix the re-streaming, but the loops' MACs stay.
- If the constraint is **stored weight bytes** (flash, download size) and the model is trained near 20 bytes/param, a
  wide block looped 2-4x gives better bpb per stored byte. Keep loops' weights aliased (one copy) in the engine: it
  costs nothing and saves the 2x on any model whose unrolled weights exceed L2.
- **Batching loop iterations doesn't apply to decode.** Loop r+1 of a byte needs loop r's output, and the next byte
  needs the last loop's argmax, so there is nothing independent to batch. Prefill already batches across bytes (64
  per weight pass), and looped prefill was 1.6-3x slower than plain at matched quality too.
- Not tested: the bandwidth-bound regime *at* 20 bytes/param (needs > ~1.5M params, so > 30 MB of unique text and
  hours of CPU training per model); input injection per loop (Huginn style); multiple seeds.

## Reproduce
    R="--bytes-per-param 20 --batch 8 --seq 128 --mode train"
    uv run python examples/15_looped_cache_lm.py $R --models plain:96,2 loop:96,2,2 plain:64,4 --ckpt runs/a.pt
    uv run python examples/15_looped_cache_lm.py $R --corpus mixed --models plain:128,1 loop:128,1,4 --ckpt runs/b.pt
    uv run python examples/15_looped_cache_lm.py _merge runs/all.pt runs/a.pt ...     # same --corpus only
    uv run python examples/15_looped_cache_lm.py --mode test --ckpt runs/all.pt
    uv run python examples/15_looped_cache_lm.py --mode infer --ckpt runs/all.pt --threads-list 1,2,4 --repeats 7 \
        --gen 2048 --out results/looped_cache/rows.json
Sweep 1: `--steps 1000 --batch 8 --seq 128` with the models in table 1. Each model's init is seeded from (seed, spec),
so groups trained in parallel and merged equal one process training them all. Raw rows: `rows_s0.json` (sweep 1),
`rows_r20_shakespeare.json`, `rows_r20_mixed.json`.

Side note: in the bench worker, numpy's idle OpenBLAS threads spun during engine decode and inflated CPU per byte
~4x at 1 thread (0.16 vs 0.03 ms/B on plain:200,2). The worker sets OPENBLAS_NUM_THREADS=1.
