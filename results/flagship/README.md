# Flagship edge LM: current best and the runs behind it

`examples/flagship_edge_lm.py` is a living example: its `BEST` config is the model with the lowest CPU and memory
per byte at about equal held-out bits per byte (bpb) found so far. When an experiment beats it, update `BEST` and add a row here.

## Current best (2026-11-28)

**Space-patched H-Net of Gated DeltaNet blocks** (`models/space_hnet.py`), trained from scratch, W8A8, run by the
standalone Rust CLI (`rust/lilbits_fused/src/bin/lilbits.rs`, one `.lbm` file, no Python):

    bytes -> embed (64, tied head) -> encoder: 1 GDN block x 64            every byte
          -> main: 4 GDN blocks x 96, one step per word (SpaceByte rule)    ~0.19 steps per byte
          -> decoder: 3 GDN blocks x 64 (input: main output + residual)     every byte

| | best: H-Net (enc 1x64, main 4x96, dec 3x64) | plain GDN d64 x 8 at the same bpb |
|---|---|---|
| params | 0.73M | 0.45M |
| held-out bpb, W8A8 on the engine | **2.158** | 2.163 |
| decode, 1 thread | **0.012 ms/B (~83K B/s)** | 0.017 ms/B (~59K B/s) |
| decode CPU / byte | **0.010 ms** | 0.015 ms |
| prefill, 1 thread | **133K B/s** | 85K B/s |
| linear MACs / byte | **0.33M** | 0.44M |
| bundle (weights on disk) | 0.84 MB | 0.55 MB |
| process RSS (whole CLI process) | 4.3 MiB | 3.8 MiB |
| recurrent state | 54 KB | 33 KB |

Same quality, 1.4x faster decode and 1.6x faster prefill on one core. It costs 0.3 MB more weights and 0.5 MiB more RSS.
Checkpoint and bundle: `/mnt/project-files/flagship/best_hnet_o64m96d3.{pt,w8a8.lbm}` in the project files.

    uv run python examples/flagship_edge_lm.py train runs/flagship/best.pt     # ~35 min on one core
    uv run python examples/flagship_edge_lm.py export runs/flagship/best.pt
    cargo build --release --manifest-path rust/lilbits_fused/Cargo.toml
    rust/lilbits_fused/target/release/lilbits runs/flagship/best.w8a8.lbm --prompt "HAMLET:" -n 240 --temperature 0.7

Sample (temperature 0.7, top-k 40):

    HAMLET:
    The hole of the two bellow them in the content.

    BECKINGHAD IV:
    The copest of them to everybody on the book of the
        offered with a father's cool and the king.

## Setup

- Data: `lilbits.corpus`, ~10 MB of English: tiny-shakespeare, the NLTK gutenberg books (bible and Moby Dick capped
  at 1 MB each) and NLTK webtext. The last 5% of every document is held out.
- Budget: **20 training bytes per parameter** (`--bytes-per-param 20`, bytes counted as tokens, Chinchilla style),
  capped at 4 epochs. That sets each model's steps: 0.5-1.7 epochs here. Batch 16 x 256 bytes, AdamW lr 3e-3
  cosine, wd 0.01, seed 0, one CPU thread per run, 4 runs in parallel.
- bpb: 64 x 8 x 256 held-out bytes (the same rows for every model), in torch and through the engine (W8A8).
- Speed: `examples/flagship_edge_lm.py bench`, which runs the CLI in 3 fresh processes (median) on one thread.
  Each process does a 512-byte prompt, then 2,048 greedy decode bytes, best of 5. RSS is the whole CLI process.
  Sapphire Rapids-class Xeon VM.
- Single seed. Seed noise on similar runs was 0.01-0.03 bpb (results/fused), so gaps under ~0.03 are not settled.

## All runs at 20 bytes / param (W8A8 on the engine)

| model | params | linear MACs / byte | held-out bpb | decode ms / B | decode CPU ms / B | prefill B/s | bundle | RSS MiB |
|---|---|---|---|---|---|---|---|---|
| plain GDN d48 x 8 | 0.26M | 0.25M | 2.443 | 0.015 | 0.015 | 88,844 | 0.43 MB | 3.5 |
| plain GDN d56 x 8 | 0.34M | 0.34M | 2.275 | 0.018 | 0.019 | 72,539 | 0.48 MB | 3.6 |
| plain GDN d64 x 8 | 0.45M | 0.44M | 2.163 | 0.017 | 0.015 | 85,282 | 0.55 MB | 3.8 |
| H-Net enc 1x48, main 4x80, dec 3x48 | 0.48M | 0.20M | 2.354 | 0.011 | 0.010 | 140,108 | 0.70 MB | 4.1 |
| H-Net enc 1x64, main 4x80, dec 2x64 | 0.52M | 0.24M | 2.265 | **0.010** | 0.010 | **173,607** | 0.70 MB | 4.1 |
| **H-Net enc 1x64, main 4x96, dec 3x64** | 0.73M | 0.33M | **2.158** | 0.012 | 0.010 | 133,481 | 0.84 MB | 4.3 |
| H-Net enc 1x80, main 4x96, dec 2x80 | 0.78M | 0.37M | **2.149** | 0.014 | 0.015 | 115,449 | 0.95 MB | 4.5 |
| H-Net enc 1x64, main 4x112, dec 2x64 | 0.85M | 0.30M | 2.196 | 0.011 | 0.010 | 146,329 | 1.09 MB | 4.6 |

All H-Nets use `align="start"`: the main network steps on a word's first byte, as in SpaceByte. W8A8 cost 0.0007-0.002 bpb on every model.

- **At equal speed the H-Net wins clearly.** H-Net 64/80 decodes at 0.010 ms/B with 2.265 bpb. The looped-depth thread's
  plain d64 x 4 on the same corpus (results/looped_cache) also decodes at 0.010 ms/B, with 2.353 bpb.
- **At equal bpb the H-Net is about 1.4x faster** (best row vs plain d64). It needs ~60% more params, so at 20 bytes/param
  it also trains on more bytes (14.6 vs 8.9 MB). Training wall time per run was about the same (0.56 vs 0.84 s/step).
- **Decoder depth beat encoder width.** dec 3x64 (2.158) beat dec 2x64 with a wider main (2.196), and nearly matched
  2x80 (2.149) at lower cost. Widening the main network bought the least.
- Everything fits in ~4 MiB of process memory, and the weights are 0.4-1.1 MB.

## Earlier rounds (kept for the record)

- **Under-trained (1,500 steps = 6 MB, 1-4 bytes/param).** Plain d128 x 8: 2.106 bpb, 0.056 ms/B, 5.5 MiB RSS.
  H-Net enc 2x128, main 6x256, dec 2x128 (6.1M params, 1 byte/param): 2.174 bpb, 0.084 ms/B, 10.6 MiB. "start"
  alignment beat "end" (main step on the trailing space) by 0.015. Too little data for the bigger model, which is why the
  budget is now set in bytes per parameter.

## Not in the flagship (measured elsewhere)

- Speculative / multi-byte decoding: slower on CPU at this size (results/byte_patching.md).
- LittleBit and looped depth: worse at matched quality (results/lean, results/looped_cache).
- Learned (H-Net router) and entropy patches: space patches were cheaper at equal loss (results/byte_patching.md).

## Next

- Multi-thread decode for the H-Net path (the engine's lockstep threads only cover the plain stack).
- int4 / ternary QAT for the main network, which is most of the weights.
- More data, so bigger models stay at 20 bytes/param without passing 4 epochs.
