# Looped depth over MiniMax Sparse Attention (example 17)

Question: example 15 found that looping a GDN block buys little, because every loop only sees its own compressed
state. If the looped core is instead an attention block that reads a K/V cache **sparsely** (MiniMax Sparse
Attention, arXiv 2606.13392), each loop can fetch different context blocks. Does that make looping pay off, and what
does it cost against a plain GDN stack?

Short answer: **no, at this scale.** Sparse looped attention was the worst model in the comparison. Looping dense
attention 3x gained 0.022 bpb over 1 loop but only tied a plain GDN stack with the same params, at about 1.5x its
decode time and with a cache that grows with context. Training sparse cost about 0.05 bpb compared with dense, and the
indexer never learned to match the main attention well (index KL ~1.1 nats at the end). One seed.

## Setup

`examples/17_looped_msa.py`, model in `src/lilbits/models/looped_msa.py`.

- msa_l3: embed, 2 GDN blocks (prelude), then one shared core looped 3x: MSA attention over a K/V cache plus SwiGLU,
  with a learned per-loop embedding. K, V and the index keys are written to the cache once per byte from the
  prelude output, so a loop costs Wq, Wq_idx, Wo and the MLP plus the attended keys, never another K/V projection.
- MSA as in the paper: one index query head per GQA group (2 groups, 4 heads, head dim 32), one shared index key
  head (d_idx 32), token index scores max-pooled per key block, top-k blocks per query and group with the query's own
  block forced. The index branch is trained by KL to the group-averaged, detached main attention, with stop-gradient
  on its inputs. Dense attention + KL for the first 20% of steps (the paper's warmup), then sparse.
- Byte-scale sizes: block 16 bytes, k = 4, so at most 64 of up to 512 cached bytes per query (mean 53.5 in training).
- msa_l1: the same model, 1 loop. dense_l3: msa_l3 with dense attention throughout. gdn_p: plain GDN at matched
  params (3 layers). gdn_c: plain GDN with 5 layers, the same block applications per byte as msa_l3.
- Width 128. **20 bytes per parameter** (Chinchilla's 20 tokens/param, bytes as tokens): msa_l3 has 650K params, so
  every model trains on the same 13.0 MB (3172 steps x 8 x 512 bytes, 1.31 epochs of the ~10 MB `lilbits.corpus`
  train split). gdn_c has more params, so it sees 11.9 bytes/param.
- Held-out bpb: 16 x 8 windows of 512 bytes from the test split. Decode: batch 1, PyTorch eager, 1 thread, 256 greedy
  bytes after a 512-byte prompt, prefill excluded, 3 fresh processes. Intel Xeon 2.1 GHz, 4 vCPUs, 2 MB L2 per core.

## Results

| model | params | bytes/param | held-out bpb | decode ms/B | CPU ms/B | memory that grows with context |
|---|---|---|---|---|---|---|
| gdn_c (5 GDN) | 1.10M | 11.9 | **2.008** | 2.8-3.2 | 3.1-3.5 | none (80 KB state) |
| dense_l3 | 650K | 20 | 2.034 | 2.4-3.3 | 2.6-3.6 | 320 B/byte K/V + index keys (FP16) |
| gdn_p (3 GDN) | 671K | 19.4 | 2.040 | **1.9-2.1** | **2.0-2.2** | none (48 KB state) |
| msa_l1 | 649K | 20 | 2.056 | 2.1-2.7 | 2.3-3.0 | 320 B/byte |
| msa_l3 | 650K | 20 | 2.085 | 3.8-4.3 | 4.1-4.7 | 320 B/byte |

msa_l3 at other test-time loop counts: 1: 2.312, 2: 2.117, **3: 2.085**, 4: 2.100, 5: 2.134 (no extrapolation past the
trained depth). msa_l3 with sparsity switched off at test time: 2.077, so the selection itself costs only 0.008; most of
the 0.05 gap to dense_l3 comes from training under sparse attention.

Peak process RSS for infer was 850-877 MiB for every model, nearly all torch and the corpus; the MSA models used
about 25 MiB more than GDN.

## What this means
- Looping attention adds a little quality (dense_l3 beats msa_l1 by 0.022), but not more than spending the same
  params on GDN layers does (gdn_p 2.040). That matches example 15: at 20 bytes/param, looping helps per stored
  parameter at best slightly, and never on decode time.
- Sparsity doesn't help here and can't on CPU decode. At 512-768 bytes of context, a dense pass over the whole cache
  is cheaper than the index scores + top-k + gather, so msa_l3 decoded slower than dense_l3. MSA pays off when the
  context is far longer than k x block (the paper's 1M tokens vs a 2,048-token budget). Byte LMs on this CPU don't
  get there, and GDN's fixed state already makes decode cost independent of context length.
- Decode is weight-bandwidth bound, and these timings are PyTorch eager, so they are mostly per-op overhead (a few
  ms/B, against 0.01-0.03 ms/B for example 15's models in the fused Rust engine). The ordering is meaningful, the
  absolute numbers are not. Attention isn't in the Rust engine, so no engine numbers here.

## Not tested
- Several seeds (example 15 measured 0.01-0.03 bpb seed noise, so the msa_l1 vs gdn_p gap is within noise; the
  msa_l3 gap is not).
- Long-context tasks (needle recall, example 12) where attention over far context should beat a fixed GDN state.
- A longer dense warmup or a larger k: the indexer's KL stayed high, so better selection may close part of the
  sparse-vs-dense gap.

## Reproduce
    for m in msa_l3 msa_l1 dense_l3 gdn_p gdn_c; do
      uv run python examples/17_looped_msa.py --mode train --models $m --d 128 --bytes-per-param 20 --ckpt runs/$m.pt
      uv run python examples/17_looped_msa.py --mode test --ckpt runs/$m.pt
      uv run python examples/17_looped_msa.py --mode infer --gen 256 --ckpt runs/$m.pt
    done
