# Primitives for a lean train + inference model

Goal: a model that is cheap both to train and to run. This repo is meant to be a collection of building
blocks for the architecture below (prelude → index → looped core → coda → verify). Each block is
testable on its own and has a dense FP reference.

## Status by architecture component

| component | primitive | status | where |
|---|---|---|---|
| Compressed linears | LittleBit low-rank binarized linear (residual paths, STE, SVD+ITQ init) | ✅ | `littlebit.py` |
| | BitNet b1.58 ternary linear (absmean weights, 8-bit activations) | ✅ | `bitnet.py` |
| | Bit-packed export: 1-bit signs (LittleBit), 2-bit codes (BitNet) | ✅ | `packed.py`, `bitnet.py` |
| | Helion kernels for packed LittleBit | ✅ interpret mode only, GPU speed unmeasured | `helion_kernels.py` |
| | Cheaper scales (INT8 h/g), uneven bpw allocation per layer type | ⬜ | |
| Prelude / core mixer | Gated DeltaNet (fixed-size state, FP gates option) | ✅ naive per-token loop | `mixers.py` |
| | Other recurrent mixers: Mamba2 (SSD), GLA / RWKV-6 style, xLSTM mLSTM, minGRU | ✅ naive loops, smoke-tested only | `mixers.py` |
| | Chunked GDN for training speed (or `fla` kernel hook) | ⬜ | |
| | State carry / rollback API (`initial_state` in, final state out) | ✅ every mixer; conv state not carried (use `conv_size=1`) | `mixers.py`, `examples/03_state_rollback.py` |
| Embed | Tied embedding / LM head | ⬜ | |
| Looped core | Shared-weight block applied N times | ⬜ | |
| | Per-loop FP scales (h, g, l) over shared LittleBit signs | ⬜ (LittleBit already separates signs from scales, so this is a small change) | |
| | Per-loop GDN state | ⬜ | |
| | Loop memory: attend over passes 1..i of the same token | ⬜ | |
| | Halting for decode, copying state forward into skipped loops | ⬜ | |
| Index / retrieval | 4-bit K/V cache with GQA | ⬜ | |
| | sign(K) signatures + XOR/popcount scoring | ⬜ (reuses `pack_signs` bit layout) | `packed.py` |
| | Block summaries every 64 tokens (shallow from prelude, deep from final loop) | ⬜ | |
| | Top-k exact attention with dense fallback when scores are flat | ⬜ | |
| | Indexer trained to match dense attention | ⬜ | |
| Coda | Higher-precision tail layers | ✅ (any `linear` choice per block) | `model.py` |
| | Sequential MTP heads for k drafts | ⬜ | |
| Verify | k-position parallel verify, accept the longest prefix, roll back index/state | ⬜ | |
| | Adaptive k from acceptance rate | ⬜ | |
| Decision models (Jev-style) | Stateful blocks, state fork per question, typed readout over tied code rows, per-type temperature | ✅ | `decision.py`, `examples/04_decision_model.py` |
| | Proper scoring (log loss, CRPS, Brier), ECE, code-swap flip rate | ✅ | `decision.py` |
| | MoE FFN with FP router, per-question halting | ⬜ | |
| Training | Fabric loop, Muon / DeepSeek Muon, FP warm start → LittleBit conversion, Optuna | ✅ | `train.py`, `optim.py`, `tune.py` |
| | Staged curriculum (looped FP → LittleBit → index → MTP → halting) | ⬜ | |
| Eval | Wall, CPU, peak RSS, RAM, disk for any code block (`Meter`, `size_report`) | ✅ | `measure.py` |
| | Loss vs stored size frontier, full train/inference compare, all examples side by side | ✅ | `benchmarks/` |
| | Long-context recall and multi-hop tasks | ⬜ (current Markov data can't measure them) | |

## What the measurements so far say for this design

- At d_model 128, LittleBit below 1 bpw loses to BitNet b1.58 at matched size (`results/frontier`). The
  FP16 scale vectors cost 16·(in+out)/(in·out) bpw per path, which is 0.25 at 128 wide and 0.008 at 4096.
  Sub-1-bit cores need wide layers, or cheaper scales.
- A looped core with shared signs and per-loop FP scales fits LittleBit well: the signs are the large
  part and get stored once, and the per-loop scales are what makes each pass different.
- `pack_signs` already produces the bit layout needed for sign(K) signatures, so XOR/popcount retrieval
  can reuse it.

## Suggested build order (matches the training stages)

1. Tied embeddings; looped GDN core with a fixed N; loop memory; chunked GDN. Train in FP.
2. LittleBit core with shared signs and per-loop scales; INT8 scales.
3. 4-bit GQA K/V cache, sign signatures, block summaries, top-k retrieval layer, indexer distillation.
4. MTP heads and the verify / rollback loop.
5. Halting for decode.

Every stage needs a long-context recall / multi-hop synthetic task to be measurable, so that task
generator should come first.
