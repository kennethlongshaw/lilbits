# Primitives for a lean train + inference model

Goal: a model that is cheap both to train and to run. This repo is meant to be a collection of building
blocks for the architecture below (prelude → index → looped core → coda → verify). Each block is
testable on its own and has a dense FP reference.

## Status by architecture component

| component | primitive | status | where |
|---|---|---|---|
| Compressed linears | LittleBit low-rank binarized linear (residual paths, STE, SVD+ITQ init) | ✅ | `layers/littlebit.py` |
| | BitNet b1.58 ternary linear (absmean weights, 8-bit activations) | ✅ | `layers/bitnet.py` |
| | Bit-packed export: 1-bit signs (LittleBit), 2-bit codes (BitNet) | ✅ | `inference/packed.py`, `layers/bitnet.py` |
| | Lookup-table (T-MAC style) CPU kernels for packed LittleBit and BitNet decode: AVX-512 / AVX2 / scalar, fused scales | ✅ 5–35× faster than FP32 dense at 1 token | `inference/lut.py`, `inference/lut_kernel.c`, `benchmarks/lut_bench.py` |
| | Helion kernels for packed LittleBit | ✅ interpret mode only, GPU speed unmeasured | `inference/helion_kernels.py` |
| | Cheaper scales (INT8 h/g), uneven bpw allocation per layer type | ⬜ | |
| Prelude / core mixer | Gated DeltaNet (fixed-size state, FP gates option) | ✅ naive per-token loop | `mixers/recurrent.py` |
| | Other recurrent mixers: Mamba2 (SSD), GLA / RWKV-6 style, xLSTM mLSTM, minGRU | ✅ naive loops, smoke-tested only | `mixers/recurrent.py` |
| | Chunked GDN for training speed (or `fla` kernel hook) | ⬜ | |
| | State carry / rollback API (`initial_state` in, final state out) | ✅ every mixer; conv state not carried (use `conv_size=1`) | `mixers/recurrent.py`, `examples/03_state_rollback.py` |
| Embed | Tied embedding / LM head | ✅ (looped LM) | `models/recurrent_depth.py` |
| | Byte-level input with dynamic patching instead of a tokenizer: BLT entropy patches, H-Net learned chunks | ✅ | `models/byte_latent.py`, `examples/07_byte_level_patching.py` |
| Looped core | Shared-weight block applied N times (prelude / core / coda, input injection, sandwich norm) | ✅ | `models/recurrent_depth.py`, `examples/06_looped_reasoning_lm.py` |
| | Sampled depth + truncated backprop (Huginn); learned exit gate (Ouro) | ✅ | `models/recurrent_depth.py` |
| | Per-loop FP scales (h, g, l) over shared LittleBit signs | ✅ | `layers/loop_scaled.py` |
| | Per-loop GDN state (shared weights, FP gates) | ✅ | `models/looped_index.py` |
| | Loop memory: attend over passes 1..i of the same token | ✅ | `mixers/loop_memory.py` |
| | Adaptive exit: freeze a position's state when KL between steps is small, or gate CDF ≥ q | ✅ (full-sequence; no KV-cache sharing across loops yet) | `models/recurrent_depth.py` |
| | Halting for decode, copying state forward into skipped loops' GDN | ✅ (relative state change threshold) | `models/looped_index.py` |
| Index / retrieval | 4-bit K/V cache with GQA (one KV head) | ✅ | `mixers/sparse_index.py` |
| | sign(K) signatures + XOR/popcount scoring | ✅ | `mixers/sparse_index.py` |
| | Block summaries every 64 tokens (shallow from prelude, deep from final loop) | ✅ | `mixers/sparse_index.py`, `models/looped_index.py` |
| | Top-k exact attention with dense fallback when scores are flat | ✅ (masked dense compute; no sparse gather kernel yet) | `mixers/sparse_index.py` |
| | Indexer trained to match dense attention (KL on block mass) | ✅ | `mixers/sparse_index.py` |
| Coda | Higher-precision tail layers | ✅ FP GDN coda | `models/looped_index.py` |
| | Sequential MTP heads for k drafts | ✅ | `models/looped_index.py` |
| Verify | k-position parallel verify, accept the longest prefix, roll back index/state | ✅ output identical to greedy (tested) | `models/looped_index.py`, `examples/09_looped_index_lm.py` |
| | Adaptive k from acceptance rate | ✅ | `models/looped_index.py` |
| Decision models (Jev-style) | Stateful blocks, state fork per question, typed readout over tied code rows, per-type temperature | ✅ | `models/decision.py`, `examples/04_decision_model.py` |
| | Proper scoring (log loss, CRPS, Brier), ECE, code-swap flip rate | ✅ | `models/decision.py` |
| | MoE FFN with FP router, per-question halting | ⬜ | |
| Training objectives | LLM-JEPA: JEPA embedding loss next to next-token loss, [PRED]-token predictor, block-causal packing, EMA target, loss dropout | ✅ | `models/jepa.py`, `examples/08_llm_jepa.py` |
| Training | Fabric loop, Muon / DeepSeek Muon, FP warm start → LittleBit conversion, Optuna | ✅ | `train.py`, `training/optim.py`, `training/tune.py` |
| | Staged curriculum (looped FP → LittleBit → index → MTP → halting) with a dense FP reference | ✅ smoke-tested | `examples/09_looped_index_lm.py` |
| Eval | Wall, CPU, peak RSS, RAM, disk for any code block (`Meter`, `size_report`) | ✅ | `measure.py` |
| | Loss vs stored size frontier, full train/inference compare, all examples side by side | ✅ | `benchmarks/` |
| | Long-context recall and multi-hop tasks | ✅ generator; nothing trained long enough to pass them yet | `data.py` (`RecallSource`) |

## What the measurements so far say for this design

- At d_model 128, LittleBit below 1 bpw loses to BitNet b1.58 at matched size (`results/frontier`). The
  FP16 scale vectors cost 16·(in+out)/(in·out) bpw per path, which is 0.25 at 128 wide and 0.008 at 4096.
  Sub-1-bit cores need wide layers, or cheaper scales.
- A looped core with shared signs and per-loop FP scales fits LittleBit well: the signs are the large
  part and get stored once, and the per-loop scales are what makes each pass different.
- `pack_signs` already produces the bit layout needed for sign(K) signatures, so XOR/popcount retrieval
  can reuse it.

The whole design is assembled in `models/looped_index.py` (`LoopedIndexLM`) and trained stage by stage in
`examples/09_looped_index_lm.py`. Still open: chunked GDN, INT8 scales, a sparse-gather retrieval kernel, LUT
kernels inside the looped core, and a training run long enough to measure recall.

## Build order (matches the training stages)

1. Tied embeddings; looped GDN core with a fixed N; loop memory; chunked GDN. Train in FP.
2. LittleBit core with shared signs and per-loop scales; INT8 scales.
3. 4-bit GQA K/V cache, sign signatures, block summaries, top-k retrieval layer, indexer distillation.
4. MTP heads and the verify / rollback loop.
5. Halting for decode.

Every stage needs a long-context recall / multi-hop synthetic task to be measurable, so that task
generator should come first.
