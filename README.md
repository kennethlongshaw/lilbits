# littlebit-scratch

A small PyTorch prototype of **LittleBit-style sub-1-bit linear layers** inside a
transformer block, trained **from scratch** (no pretrained teacher, no distillation).

Reference: Lee et al., *LittleBit: Ultra Low-Bit Quantization via Latent Factorization*
(NeurIPS 2025, arXiv:2506.13771) and LittleBit-2 (arXiv:2603.00042).

## The layer

Every block linear (`q, k, v, o, gate, up, down`) stores

```
W_hat = diag(h) · sign(U) · diag(l) · sign(V)^T · diag(g)        U: out×r, V: in×r
```

- `U`, `V` are full-precision *latent* factors, binarized to ±1 in the forward pass.
- `h` (row), `g` (column) and `l` (latent) are learned FP16 scale vectors.
- Optional residual factorization adds a second path with its own factors (`--model.residual false` turns it off).
- The forward pass never builds `W`: scales fold into the factors, `A = g·sign(V)·l`, `B = h·sign(U)` (paths concatenated), then `y = x @ A @ Bᵀ`.
- Bits/weight = paths · (r·(in+out) + 16·(in+out+r)) / (in·out). `rank_for_bpw` picks the rank for a target.
- Sign gradients use a straight-through estimator: `clipped` (identity inside |x|≤1, the default here) or
  `smooth` (SmoothSign: sign forward, d/dx tanh(kx) backward, as in the paper's QAT).

Embeddings, RMSNorms and the LM head stay full precision, as in the paper.

### How this differs from the paper

The paper compresses a *pretrained* model: SVD of the dense weight initializes `U, V` and the scales
(LittleBit-2 adds a Joint-ITQ latent rotation), then QAT with knowledge distillation from the FP teacher.
Here there is no teacher: latents are random, scales are set so each layer starts roughly
variance-preserving, and the model is trained on the next-token loss directly. SVD/ITQ init does not
apply, so it's left out.

Two things mattered for from-scratch training: a lower learning rate on the latent factors than on
everything else (`--train.latent_lr_mult 0.1`, the default; otherwise signs churn ~20% per 100 steps and nothing learns),
and the clipped STE, which beat SmoothSign (k=100 or 10) in short CPU sweeps.

## Setup

```bash
uv sync                 # installs torch from PyPI (CUDA wheels on Linux)
uv run pytest
```

## Training

```bash
uv run lb-train --linear both --train.steps 1500             # FP baseline, then LittleBit @ ~1 bpw
uv run lb-train --linear littlebit --model.bpw 0.55 --model.d_model 1024 --model.n_layers 6 \
    --train.steps 20000 --train.accelerator cuda --train.precision bf16-mixed
uv run lb-train --text input.txt --linear both               # char-level LM on any text file
uv run lb-train --print_config > run.yaml && uv run lb-train --config run.yaml   # YAML configs
uv run lb-train --help                                       # every model.* / train.* option
```

The default data is a random sparse order-2 Markov chain (vocab 32, 4 successors per context), so the
best achievable loss (its entropy rate) is known and printed. `--out run.json` saves the loss history.

### CPU results (d_model 128, 2 blocks, seq 64, batch 32, 1,500 steps, seed 0)

Entropy floor ≈ 0.894 nats, uniform = 3.466.

| block linears       | bpw   | storage  | final eval loss |
|---------------------|-------|----------|-----------------|
| FP16                | 16    | 772 KiB  | 0.984           |
| LittleBit, residual | 0.99  | 47.8 KiB | 1.386           |
| LittleBit, 1 path   | 0.99  | 47.9 KiB | 1.397           |
| LittleBit, 1 path   | 1.72* | 83.2 KiB | 1.213           |

\* asked for 2 bpw; the 128→344 MLP layers hit full rank first.

At these tiny widths the FP16 scales eat a big part of the bit budget (a 128×128 layer at 1 bpw only gets
rank 15 per residual path), so sub-1-bit is much more forgiving at real model widths: a 4096×4096 layer at
0.1 bpw still gets rank 86 per path. Scale up `--d-model` on a GPU to test that regime.

### Lightning Fabric

Training runs on [Lightning Fabric](https://lightning.ai/docs/fabric/stable/): Fabric owns device placement,
`fabric.backward`, gradient clipping, seeding, precision and multi-device setup, so the loop has no `.to(device)`
calls. Options live in two dataclasses, `Config` (model) and `TrainConfig` (optimization + Fabric), and the
`lb-train` CLI is generated from them by jsonargparse (also gives `--config run.yaml` / `--print_config`).
Programmatic use: `train(Config(...), TrainConfig(...), source)`.

Fabric options: `--train.accelerator auto|cpu|cuda|mps`, `--train.devices N`, `--train.precision 32-true|bf16-mixed|16-mixed`.

## Initialization (`--model.init`, `--model.itq_iters`, `--train.fp_warmup`)

- `small` (default): latents ~ N(0, 0.1), scales set for unit variance.
- `bimodal`: latents at ±0.5 plus noise, so none start near the sign-flip point at zero.
- `svd`: no teacher, so take the dense matrix `nn.Linear` would have started from, and initialize from its
  SVD the way the paper does (Dual-SVID: `U√S`, `V√S`, rank-1 fits of their magnitudes give `h, g, l`;
  the residual path fits what's left). The result is rescaled to the dense init's norm.
- `--model.itq_iters N`: LittleBit-2's Joint-ITQ. Before binarizing, rotate both factors by one orthogonal `R`
  (`UVᵀ = (UR)(VR)ᵀ`, so the FP product is unchanged) chosen to push entries toward ±1 and away from zero.
- `--train.fp_warmup N`: train a full-precision copy for the first N steps, then convert it to LittleBit with the
  SVD(+ITQ) init and keep training. Still from scratch, no external teacher, but it reproduces the paper's
  "compress a trained model" pipeline. The N steps count toward `--train.steps`.

Logs show `sign-flips` (fraction of latent signs that changed since the last log) and `near-zero`
(fraction of latents within 10% of their rms from zero; a Gaussian gives about 8%).

### Init comparison (same setup as above, residual LittleBit at 0.99 bpw, 1,500 steps, seed 0)

| init                                | final eval | near-zero at start | flips/100 steps at end |
|-------------------------------------|------------|--------------------|------------------------|
| small (old default)                 | 1.386      | 7.9%               | 0.68%                  |
| bimodal                             | 1.287      | 0%                 | 0% (signs never move)  |
| bimodal, latent LR ×1               | 1.353      | 0%                 | 0.24%                  |
| svd                                 | 1.226      | 7.8%               | 0.26%                  |
| svd + ITQ                           | 1.179      | 3.2%               | 0.10%                  |
| svd + ITQ, latent LR ×1             | 1.201      | 3.2%               | 1.37%                  |
| FP warm start 300 steps + SVD       | 1.140      | 8.0%               | 0.30%                  |
| FP warm start 300 steps + SVD + ITQ | **1.089**  | 3.2%               | 0.11%                  |

FP16 reference: 0.984. Entropy floor: 0.894. These are single-seed CPU runs, so treat gaps under ~0.05 as
possibly noise. Run them yourself with e.g. `uv run lb-train --linear littlebit --train.fp_warmup 300 --model.itq_iters 50`.

## Optimizer (`--train.optim muon`)

`--train.optim muon` puts the 2D matrices inside the blocks (FP linear weights, or the LittleBit latents `U`, `V`)
on `torch.optim.Muon` (PyTorch ≥ 2.9) with `adjust_lr_fn="match_rms_adamw"`, so the same `--lr` applies.
Embeddings, the LM head, LittleBit scales and norms stay on AdamW (`--train.muon_embed true` moves embeddings/head
to Muon too). `--train.latent_lr_mult` and `--train.muon_lr_mult` still apply.

Same setup as above, 1,500 steps, seed 0 (final eval loss):

| model / init                    | AdamW (latent LR ×0.1) | Muon (latent LR ×0.1) | Muon (latent LR ×1) |
|---------------------------------|------------------------|-----------------------|---------------------|
| FP16                            | 0.984                  | 0.980 (0.978 at 3× lr)| n/a                 |
| LittleBit, small init           | 1.386                  | 1.470                 | 1.251               |
| LittleBit, svd + ITQ            | 1.179                  | 1.198                 | 1.217               |
| LittleBit, FP warm 300 + ITQ    | **1.089**              | 1.115                 | 1.187               |

Muon doesn't beat AdamW at this size. It does tolerate the full latent learning rate with the small init
(1.251, its best LittleBit result without SVD init), but with the SVD/ITQ inits the bigger latent steps undo the
rotation's work: near-zero latents climb from 3% back to ~10%.

### Weight decay 0.1 (default is 0)

Same runs with `--train.weight_decay 0.1` (previous tables used 0). Decay applies to the 2D weights and LittleBit
latents; scales and norms are never decayed.

| model / init                 | AdamW wd 0 → 0.1 | Muon wd 0 → 0.1 |
|------------------------------|------------------|-----------------|
| FP16                         | 0.984 → 0.986    | 0.980 → 0.981   |
| LittleBit, small init        | 1.386 → 1.357    | 1.470 → 1.438   |
| LittleBit, svd + ITQ         | 1.179 → 1.180    | 1.198 → 1.199   |
| LittleBit, FP warm 300 + ITQ | 1.089 → 1.102    | 1.115 → 1.127   |

### DeepSeek-V4 Muon recipe (`--train.optim muon_ds`, `--model.qk_norm true`)

`src/littlebit_scratch/optim.py` implements DeepSeek-V4's Muon: 10 Newton-Schulz steps (8 with the fast
coefficients, 2 with (2, -1.5, 0.5) to pin singular values at 1), update RMS 0.18, Nesterov 0.95.
`--model.qk_norm true` adds RMSNorm on per-head queries/keys (DeepSeek's replacement for QK-Clip).
`--train.muon_ns_steps 10` runs stock `torch.optim.Muon` with 10 steps of the fast coefficients only.

Matched params (`--model.bpw 1.41`), weight decay 0, 1,500 steps, seed 0. "Steps to" = first eval (every 100 steps)
at or below the target. LittleBit rows use the 300-step FP warm start + ITQ (the optimizer also runs the FP phase).

| run | final eval | steps to 1.05 / 1.00 (FP) or 1.15 / 1.10 (LittleBit) | wall time* |
|---|---|---|---|
| FP, AdamW | 0.984 | 800 / 1,300 | 109 s |
| FP, torch Muon 5 steps | 0.980 | 700 / 1,100 | 126 s |
| FP, torch Muon 10 steps | 0.980 | 700 / 1,100 | 131 s† |
| FP, DeepSeek Muon | 0.979 | 700 / 1,100 | 143 s |
| FP, DeepSeek Muon + QK-norm | 0.980 | 700 / 1,100 | 156 s |
| LittleBit, AdamW | **1.057** | **600 / 900** | 132 s |
| LittleBit, torch Muon 5 steps | 1.084 | 900 / 1,200 | 169 s |
| LittleBit, torch Muon 10 steps | 1.066 | 700 / 1,100 | 186 s† |
| LittleBit, DeepSeek Muon | 1.065 | 700 / 1,100 | 170 s |
| LittleBit, DeepSeek Muon + QK-norm | 1.080 | 800 / 1,200 | 179 s |

\* 4 runs shared the CPU, 1 thread each. † These two ran 2 at a time, so their times aren't directly comparable.

On FP weights every Muon variant needs ~15% fewer steps than AdamW to reach 1.00, but costs more per step on CPU,
so wall-clock is a wash at this size. More Newton-Schulz steps and the two-phase schedule don't change FP
results. On LittleBit latents AdamW still wins; 10 NS steps (stock or DeepSeek) close part of the gap vs
5 steps, and the DeepSeek two-phase schedule adds nothing over plain 10 steps. QK-norm didn't help either.

### Full benchmark across optimizers (`lb-bench --configs fp:muon_ds,...`)

`uv run lb-bench --bpw 1.41 --parallel --configs fp:adamw,fp:muon,fp:muon10,fp:muon_ds` (and the same with
`lb_warm`). Matched params, weight decay 0, 4 runs at a time on 1 CPU thread each (both batches the same way).
Saved as `results/bench_d128_matched_params_optimizers.json`.

| run | final loss | steps / tokens | wall time (per step) | trainable params | weights+grads+opt state | peak train memory | stored blocks | stored model | inference/batch |
|---|---|---|---|---|---|---|---|---|---|
| FP, AdamW | 0.984 | 1,500 / 3.07M | 109 s (73 ms) | 412,288 | 6.29 MiB | 230 MiB | 772.0 KiB (16.00 bpw) | 805.2 KiB | 16.2 ms |
| FP, PyTorch Muon 5 steps | 0.980 | 1,500 / 3.07M | 126 s (84 ms) | 412,288 | 4.78 MiB | 236 MiB | 772.0 KiB (16.00 bpw) | 805.2 KiB | 16.1 ms |
| FP, PyTorch Muon 10 steps | 0.980 | 1,500 / 3.07M | 146 s (97 ms) | 412,288 | 4.78 MiB | 234 MiB | 772.0 KiB (16.00 bpw) | 805.2 KiB | 15.8 ms |
| FP, DeepSeek Muon | 0.979 | 1,500 / 3.07M | 139 s (92 ms) | 412,288 | 4.78 MiB | 227 MiB | 772.0 KiB (16.00 bpw) | 805.2 KiB | 16.5 ms |
| LittleBit, AdamW | 1.056 | 1,500 (300 FP) / 3.07M | 144 s (96 ms) | 410,256 | 6.26 MiB | 237 MiB | 67.7 KiB (1.40 bpw) | 101.0 KiB | 21.3 ms |
| LittleBit, PyTorch Muon 5 steps | 1.084 | 1,500 (300 FP) / 3.07M | 185 s (123 ms) | 410,256 | 4.80 MiB | 251 MiB | 67.7 KiB (1.40 bpw) | 101.0 KiB | 23.0 ms |
| LittleBit, PyTorch Muon 10 steps | 1.066 | 1,500 (300 FP) / 3.07M | 203 s (135 ms) | 410,256 | 4.80 MiB | 253 MiB | 67.7 KiB (1.40 bpw) | 101.0 KiB | 25.8 ms |
| LittleBit, DeepSeek Muon | 1.065 | 1,500 (300 FP) / 3.07M | 182 s (121 ms) | 410,256 | 4.80 MiB | 235 MiB | 67.7 KiB (1.40 bpw) | 101.0 KiB | 27.4 ms |

## Baseline vs LittleBit on the same data (`uv run lb-bench`)

All three train on the identical Markov token stream (same seed), d_model 128, 2 blocks, batch 32 × seq 64,
AdamW, weight decay 0.1 (`--weight-decay 0.1`; default is now 0), 1 CPU thread each. Saved as `results/bench_d128.json`. This table was measured
*before* the fused forward (below), so its LittleBit wall time, peak memory and inference time are the old, slower ones.

| metric | FP16 baseline | LittleBit ~1 bpw, SVD+ITQ init (pure from scratch) | LittleBit ~1 bpw, 300-step FP warm start + SVD+ITQ |
|---|---|---|---|
| Final eval loss (floor 0.895) | 0.986 | 1.180 | 1.102 |
| Training steps / tokens seen | 1,500 / 3.07M | 1,500 / 3.07M | 1,500 / 3.07M (300 FP) |
| Wall time (ms/step) | 108 s (72) | 216 s (144) | 200 s (133) |
| Trainable params (FP32 while training) | 412,288 | 253,068 | 253,068 |
| Train state: weights+grads+AdamW | 6.29 MiB | 3.86 MiB | 3.86 MiB |
| Train peak memory (RSS delta / CUDA peak) | 228 MiB | 363 MiB | 356 MiB |
| Deployed block linears | 772.0 KiB (16.00 bpw) | 47.8 KiB (0.99 bpw) | 47.8 KiB (0.99 bpw) |
| Deployed whole model | 805.2 KiB | 81.0 KiB | 81.0 KiB |
| Compression vs FP16 (blocks / whole) | 1.0x / 1.0x | 16.2x / 9.9x | 16.2x / 9.9x |
| Inference ms per batch (unpacked) | 17.0 | 31.5 | 30.5 |

### Same trainable-parameter count (412k)

`uv run lb-bench --bpw 1.41` raises LittleBit's ranks (attention 15→27, MLP 29→48 per residual path) until it has as
many trainable FP32 parameters as the baseline (410,256 vs 412,288). Saved as `results/bench_d128_matched_params.json`.
Measured with the fused forward, configs run one after another (not concurrently).

| metric | FP16 baseline | LittleBit ~1.41 bpw, SVD+ITQ init (pure from scratch) | LittleBit ~1.41 bpw, 300-step FP warm start + SVD+ITQ |
|---|---|---|---|
| Final eval loss (floor 0.895) | 0.986 | 1.155 | 1.050 |
| Training steps / tokens seen | 1,500 / 3.07M | 1,500 / 3.07M | 1,500 / 3.07M (300 FP) |
| Wall time (ms/step) | 104 s (70) | 148 s (98) | 134 s (90) |
| Trainable params (FP32 while training) | 412,288 | 410,256 | 410,256 |
| Train state: weights+grads+AdamW | 6.29 MiB | 6.26 MiB | 6.26 MiB |
| Train peak memory (RSS delta / CUDA peak) | 228 MiB | 240 MiB | 239 MiB |
| Deployed block linears | 772.0 KiB (16.00 bpw) | 67.7 KiB (1.40 bpw) | 67.7 KiB (1.40 bpw) |
| Deployed whole model | 805.2 KiB | 101.0 KiB | 101.0 KiB |
| Compression vs FP16 (blocks / whole) | 1.0x / 1.0x | 11.4x / 8.0x | 11.4x / 8.0x |
| Inference ms per batch (unpacked) | 20.0 | 21.5 | 22.8 |

Same, with weight decay 0 (the default), saved as `results/bench_d128_matched_params_wd0.json`:

| metric | FP16 baseline | LittleBit ~1.41 bpw, SVD+ITQ init (pure from scratch) | LittleBit ~1.41 bpw, 300-step FP warm start + SVD+ITQ |
|---|---|---|---|
| Final eval loss (floor 0.895) | 0.984 | 1.146 | 1.056 |
| Training steps / tokens seen | 1,500 / 3.07M | 1,500 / 3.07M | 1,500 / 3.07M (300 FP) |
| Wall time (ms/step) | 98 s (66) | 134 s (89) | 124 s (83) |
| Trainable params (FP32 while training) | 412,288 | 410,256 | 410,256 |
| Train state: weights+grads+AdamW | 6.29 MiB | 6.26 MiB | 6.26 MiB |
| Train peak memory (RSS delta / CUDA peak) | 230 MiB | 244 MiB | 239 MiB |
| Deployed block linears | 772.0 KiB (16.00 bpw) | 67.7 KiB (1.40 bpw) | 67.7 KiB (1.40 bpw) |
| Deployed whole model | 805.2 KiB | 101.0 KiB | 101.0 KiB |
| Compression vs FP16 (blocks / whole) | 1.0x / 1.0x | 11.4x / 8.0x | 11.4x / 8.0x |
| Inference ms per batch (unpacked) | 17.9 | 22.9 | 18.7 |

Notes:
- "Deployed" counts LittleBit block weights as packed sign bits + FP16 scales, everything else as FP16.
  The code doesn't actually pack bits yet; that's the storage format the numbers assume.
- Training keeps FP32 latents, so training memory does not shrink with bpw: at ~1 bpw there are ~0.6 latent
  values per original weight.
- Fused forward: the scales are folded into the weight-sized factors (`A = g·sign(V)·l`, `B = h·sign(U)`) and
  residual paths are concatenated along the rank axis, so a layer is `x @ A @ Bᵀ`, two matmuls. Before this,
  scales were applied to activations per path: training steps were ~2× the baseline's and peak memory ~1.6×.
  Now it's ~1.4× time and about the same memory. What's left is op overhead (sign/STE and scale ops per
  layer), which matters at this tiny size and should shrink relative to matmuls at real widths.
- Inference time uses the same unpacked FP32 forward, not a bit-packed kernel, so it says nothing about deployed speed.
- The FP warm-start row's train-state line covers only the LittleBit phase; its FP phase needs the baseline's.

## Inference speed (`uv run lb-infer-bench`)

In `eval()` + `no_grad`, `LittleBitLinear` now builds its scale-folded ±1 factors once and caches them
(cleared on `.train()` / re-init), so inference is just `x @ A @ Bᵀ`. If the low-rank pair would cost more
MACs than a dense matmul, it multiplies them out into one dense matrix instead. Before this, every inference
call re-binarized the latents and rebuilt the factors, which is what made LittleBit look slow.

Full CPU output (4 threads): `results/infer_bench_cpu.md`. Highlights:

- TinyLM (d_model 128), batch 32×64: FP 8.6 ms; LittleBit 1.41 bpw 15.5 → 9.1 ms; 1.0 bpw 11.1 → 8.7 ms.
  Single-token: FP 0.28 ms; LittleBit 1.41 bpw 3.6 → 0.30 ms. At width 128 and ~1.4 bpw the rank is close to
  break-even, so parity is the best possible.
- Single layers at real widths: speed tracks MACs. ~1 bpw ≈ dense rank, so ≈1× speed; 0.55 bpw ≈ 1.7-2.7×;
  0.1 bpw ≈ 7-22× (4096×4096: 0.77 → 0.036 ms per token, 120 → 16 ms for 2,048 tokens).

These are FP32 cached factors, i.e. the speedup comes only from the low rank. A bit-packed kernel (1-bit
factors, which is how the paper deploys) would also cut weight memory traffic ~32×, which should help most in
memory-bound batch-1 decoding, especially on GPU. Not implemented here. The FP32 cache also means inference
RAM for the factors is rank·(in+out)·4 bytes, not the packed size in the tables above.

## Bit-packed inference export (`pack_model`, `uv run lb-mem-bench`)

`littlebit_scratch.packed.pack_model(model)` swaps each `LittleBitLinear` for a `PackedLittleBitLinear`:
signs stored 1 bit each (uint8, 8 per byte), `h, g, l` in FP16, FP32 latents dropped. The forward unpacks
128 rank columns at a time through a 256-entry lookup table, folds the scales into that small chunk, and
accumulates into the output in place, so no full-size FP matrix is ever built.
`PackedLittleBitLinear.empty(...)` + `load_state_dict` loads a packed checkpoint without ever allocating latents.

CPU, 1 thread, each case in its own process (`results/mem_bench_cpu.json`). "Extra peak RAM" is the
high-water RSS above the pre-forward level during the timed forwards.

| case | weights in RAM | on disk | small input: time / extra peak RAM | large input: time / extra peak RAM |
|---|---|---|---|---|
| TinyLM, FP32 baseline | 1,610 KiB | 1,618 KiB | 0.32 ms / 0.1 MiB (1 token) | 23.67 ms / 23.1 MiB (batch 32x64) |
| TinyLM, LittleBit 1.41 bpw, training format + FP32 cache | 3,067 KiB | 1,648 KiB | 0.35 ms / 0.1 MiB (1 token) | 25.16 ms / 14.5 MiB (batch 32x64) |
| TinyLM, LittleBit 1.41 bpw, **packed** | 137 KiB | 160 KiB | 2.07 ms / 0.4 MiB (1 token) | 29.98 ms / 22.0 MiB (batch 32x64) |
| 4096×4096, FP32 nn.Linear | 65,536 KiB | 65,538 KiB | 2.69 ms / 1.8 MiB (1 token) | 395.31 ms / 38.5 MiB (2048 tokens) |
| 4096×4096, LittleBit 1.0 bpw, training format + FP32 cache | 128,840 KiB | 64,459 KiB | 2.76 ms / 0.0 MiB (1 token) | 385.89 ms / 73.7 MiB (2048 tokens) |
| 4096×4096, LittleBit 1.0 bpw, **packed** | 2,052 KiB | 2,054 KiB | 36.41 ms / 16.3 MiB (1 token) | 475.13 ms / 49.6 MiB (2048 tokens) |
| 4096×4096, LittleBit 0.55 bpw, **packed** | 1,138 KiB | 1,141 KiB | 15.92 ms / 17.0 MiB (1 token) | 223.21 ms / 49.4 MiB (2048 tokens) |
| 4096×4096, LittleBit 0.1 bpw, **packed** | 208 KiB | 211 KiB | 3.10 ms / 9.7 MiB (1 token) | 56.86 ms / 41.6 MiB (2048 tokens) |

Packing gives the memory the bpw numbers promise (4096×4096 at 1 bpw: 64 MiB FP32 → 2 MiB, 32× less;
whole TinyLM 1.6 MiB → 0.13 MiB). The cost is speed without a fused kernel: every call unpacks all the signs,
so single-token decode is ~13× slower than dense at 1 bpw and ~1.2× slower at 0.1 bpw; with many tokens the
unpack amortizes (0.55 bpw: 1.8× faster, 0.1 bpw: 7× faster than dense for 2,048 tokens). A real kernel
would unpack in registers and turn the 32× smaller weight reads into decode speed.

## Helion kernel for packed inference (`backend="helion"`, `uv run lb-gpu-bench`)

`src/littlebit_scratch/helion_kernels.py` has two [Helion](https://pytorch.org/blog/helion/) kernels that
run a packed layer straight from the 1-bit storage, unpacking bits inside each tile (no unpacked weight
ever hits memory):

- `littlebit_down`: `zp[j, m, b] = l[8b+j] · Σ_k (x·g)[m, k] · sign_bit_j(Vb[k, b])`
- `littlebit_up`: `y[m, n] = y_in[m, n] + h[n] · Σ_b Σ_j zp[j, m, b] · sign_bit_j(Ub[n, b])`

The rank activation is kept as 8 bit planes (`zp[j]` = rank columns j, 8+j, ...), so each byte tile is
loaded once and shifted 8 times; no tile reshapes. Padding bits get `l = 0`. Residual paths are two launches
each, accumulating into `y`.

```bash
uv sync --extra gpu                  # installs helion (+ triton)
pack_model(model, backend="helion")  # or set_backend(model, "helion") on an already packed model
uv run lb-gpu-bench                  # FP16 nn.Linear vs packed torch vs packed Helion, 4096-wide layers
```

On first use per shape Helion picks the fastest of 4 candidate configs per kernel; `LB_HELION_AUTOTUNE=1`
runs Helion's full autotuner instead (~minutes per shape; it prints a config you can hardcode).

End-to-end CPU demo (`HELION_INTERPRET=1 uv run --extra gpu lb-demo-infer`): trains a LittleBit TinyLM,
packs it, then runs held-out eval and greedy generation through the unpacked model, the packed torch backend
and the packed Helion kernels. All three give the same held-out loss and identical generations; output in
`results/demo_infer_cpu_helion_interpret.txt`.

**Status: not yet run on a GPU** (this was built in a CPU-only sandbox). What's verified: Triton code
generates for every candidate config, and in Helion's eager reference mode on CPU (`HELION_INTERPRET=1`)
the kernels match the PyTorch packed layer to ~1e-6, for single layers and a whole packed TinyLM
(`tests/test_helion.py`). Speed is unmeasured. Likely next tuning steps once it runs: feed `tl.dot` FP16/BF16
operands instead of FP32 (±1 is exact in FP16), and fuse the two residual paths into one launch.

## Layout

- `src/littlebit_scratch/littlebit.py`: `LittleBitLinear`, STEs, bpw accounting
- `src/littlebit_scratch/model.py`: attention, SwiGLU, `TransformerBlock`, `TinyLM`
- `src/littlebit_scratch/data.py`: Markov source with known entropy floor, char-level text source
- `src/littlebit_scratch/train.py`: Lightning Fabric training loop, `TrainConfig`, `lb-train` CLI (jsonargparse)
- `src/littlebit_scratch/optim.py`: DeepSeek-V4-style Muon (hybrid Newton-Schulz)
- `src/littlebit_scratch/infer_bench.py`: `lb-infer-bench`, inference speed vs nn.Linear
- `src/littlebit_scratch/packed.py`: bit-packed inference export
- `src/littlebit_scratch/mem_bench.py`: `lb-mem-bench`, inference memory/speed of FP vs cached vs packed
- `src/littlebit_scratch/helion_kernels.py`: Helion kernels for packed inference
- `src/littlebit_scratch/demo_infer.py`: `lb-demo-infer`, train → pack → inference with torch / Helion backends
- `src/littlebit_scratch/gpu_bench.py`: `lb-gpu-bench`, GPU speed of dense vs packed torch vs packed Helion
- `src/littlebit_scratch/bench.py`: `lb-bench`, FP vs LittleBit size/loss/time/memory comparison
- `results/`: logs from the comparison runs above
- `tests/`: layer math, STE gradients, bpw budget, causality, SVD/ITQ init, and from-scratch training smoke tests
