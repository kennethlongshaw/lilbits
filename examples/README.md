# Examples

Small, runnable scripts showing how the primitives fit together. Run from the repo root with `uv run python examples/<file>`.

| file | shows |
|---|---|
| `01_train_and_pack.py` | train a tiny LM with LittleBit / BitNet / FP linears, then pack it (1-bit signs or 2-bit ternary codes) |
| `02_recurrent_mixers.py` | Gated DeltaNet, Mamba2, GLA, mLSTM and minGRU mixers: params and fixed state size, FP vs LittleBit |
| `03_state_rollback.py` | snapshot / verify k drafts / roll back a recurrent state, as speculative decoding needs |
| `04_decision_model.py` | Jev-style decision model: state prefilled once, batched question forks, typed Choice / Score / Noul readout, calibration and code-swap eval (`--linear`, `--mixer`, `--loops`) |
| `05_pack_and_helion.py` | train → pack to 1-bit → inference on the torch and Helion backends, with logits and generation agreement |
| `06_looped_reasoning_lm.py` | recurrent-depth LM (Huginn; Ouro exit gate optional): latent loops instead of thinking tokens, test-time depth scaling, adaptive exit that freezes converged states, loop-vs-token cost (`--core-linear`, `--exit-gate`) |
| `07_byte_level_patching.py` | tokenizer-free byte models: BLT with entropy patches or SpaceByte space patches (no patcher model), hash n-gram embeddings, H-Net (learned chunks, ratio loss) vs a plain byte transformer: held-out bits/byte, bytes per patch, batch-1 cached decode (bytes/s, MACs/byte, inner steps/byte), full cost metrics (`--latent-linear`, `--max-patch`) |
| `08_llm_jepa.py` | LLM-JEPA: next-token loss plus a JEPA embedding loss on paired views (Text, Code), [PRED] tokens as the predictor, block-causal packing in one pass, EMA or online target, loss dropout; compares all variants on quality and full train/infer cost (`--linear`, `--k-pred`, `--ema-decay`) |
| `09_looped_index_lm.py` | the full lean-decoder design: LittleBit GDN prelude, looped core with shared signs + per-loop scales, per-loop GDN state, loop memory, sparse index (4-bit K/V, sign signatures, block summaries, dense fallback), FP coda with MTP drafts, parallel verify with rollback and adaptive k, decode halting; trained in 5 stages next to a dense FP reference on recall / two-hop data, with memory-tier bytes and full cost metrics (`--steps`, `--length`, `--loops`, `--bpw-alloc greedy` for calibrated uneven core ranks, `--kernel lut`) |
| `10_pocket_lm.py` | Shakespeare bytes + QA instruct stage; byte-level pocket model: byte encoder, exact byte-pair router (space-patching init) so the body runs once per patch, example 09's looped index body with a sub-1-bit INT8-scale LittleBit core, byte decoder, MTP drafts with exact verify; every piece a switch (`--router pair|hnet|none`, `--core fp|littlebit`, `--no-index`, `--mtp`, `--kernel lut`), next to an FP16 byte GDN LM matched on deployed bytes |
| `11_energy_thinking.py` | energy-based next-byte prediction (after Energy-Based Transformers, 2025): predict by gradient descent on a candidate under a learned energy, more steps = more thinking, lowest-energy-of-M = self-verification; lean form where a GDN backbone encodes context once and only a small energy head runs per step; next to a plain LM on the same backbone, on Shakespeare bytes + QA |
| `12_shakespeare_tasks.py` | toy Shakespeare tasks with exact scores, one per run (`--task`): needle-in-haystack recall of a hidden password, copy / sort of character-name lists, tool calling (`search("line")` / `next("line")` run over the corpus), DPO on right-vs-wrong QA answers, GRPO with exact-match rewards; `--model gdn` (GDN byte LM) or `looped` (example 09's looped index LM on bytes) |
| `13_operator_loop.py` | looped model where each loop is one routed instruction over the context and a per-position register workspace: typed ops (filter, point, topk, sum, mean, max, count, attend, mix, noop) with routed input / destination slots, gated write-back, Gumbel-soft training annealed to hard one-op-per-loop inference with a readable op trace, load-balance loss; vs the attention-only loop on Shakespeare sort / count / who-said / who-answers tasks, exact match per loop count |
| `14_lean_lm.py` | the measured wins in one byte-level model (models/lean_lm.py): GDN blocks (fixed-size state, chunked training kernel), LittleBit block linears after an FP phase (SVD+ITQ), low-rank MTP draft heads, LUT-packed inference and self-speculative decoding whose rollback keeps the verify call's per-byte GDN states instead of replaying; vs FP16 at the same deployed size and at the same width, every dimension |
| `15_looped_cache_lm.py` | looped (weight-shared) GDN depth on the fused Rust engine: K unique blocks x R loops with the loops' weights aliased to one copy (hot, stays in L2) vs copied (cold), against plain GDN stacks at matched bpb; decode ms/B, CPU/B, prefill, RSS; `--bytes-per-param 20` Chinchilla-ratio training, `--corpus mixed` (results/looped_cache) |
| `16_sleep_lm.py` | in-context adaptation via sleep (Lee et al. 2026, models/sleep_lm.py): every W tokens the whole stack loops N - 1 more times over the finished window with the same weights and gates the window's KV cache toward the refined keys / values; wake-only loss, backprop through all loops and windows; N = 1 / 2 / 4 side by side on Shakespeare bytes + few-shot key → value maps: bits per byte by context position, context efficiency vs N = 1, accuracy by shot count, decode and per-sleep cost, KV bytes (unchanged) and gate overhead (`--window`, `--loops`, `--data corpus`, `--bytes-per-param`, `--linear littlebit`, `--sleep-causal`); at 20 bytes/param on the 10 MB corpus N = 2 / 4 learn few-shot recall (97% at 4 shots) that N = 1 does not (17%), `results/sleep/` |
| `flagship_edge_lm.py` | **the current best edge LM, end to end** (living: `BEST` is updated as experiments win; results/flagship): space-patched H-Net of GDN blocks (narrow encoder / decoder per byte, main network once per word), trained from scratch on ~10 MB of English at 20 bytes / param, exported W8A8 to one `.lbm` file and run by the standalone Rust CLI `lilbits` (no Python, ~4 MiB RSS, ~83K bytes/s decode on one core) |
| `17_looped_msa.py` | looped depth over MiniMax Sparse Attention (models/looped_msa.py): GDN prelude, then one shared MSA+SwiGLU core looped over a K/V cache written once per byte; MSA index branch (per-group index query, shared index key, max-pooled block scores, top-k blocks with the own block forced) trained by KL to the detached main attention, dense warmup then sparse; vs 1 loop, dense attention, and plain GDN matched on params and on depth; held-out bpb, test-time loop curve, decode ms/byte, CPU, cache vs state bytes, RSS (`--bytes-per-param` sizes training to a data:param ratio) |
| `18_sleep_consolidation.py` | continual learning with a sleep phase (Behrouz et al. 2026, models/sleep_consolidation.py): each session's new facts go into fast memory (low-rank MLP adapters) while awake; sleep unlocks a new pre-allocated low-rank routed expert per block, distills fast → slow on the model's own dreams (GKD on- and off-policy, learning-to-imitate edit-distance reward), dreams with random old experts forced on and keeps the highest-gradient dreams for SFT, then resets fast memory; vs fast-only, continual SFT and a no-dreaming ablation: no-context recall of every session's facts, retention after each session, base drift in bits per byte (`--sessions`, `--facts`, `--lam`, `--alpha`, `--anchor`); at 20 bytes/param wake learns every fact and SFT forgets all but the last session, but sleep does not yet retain exact facts (`results/sleep/`) |

## Train / test / infer

Every example runs on the same harness (`lilbits.runner`, built on Lightning Fabric) and has the same options:

```bash
uv run python examples/04_decision_model.py                     # --mode all: train, save, test, infer
uv run python examples/04_decision_model.py --mode train        # train and save results/ckpt/04_decision_model.pt
uv run python examples/04_decision_model.py --mode test         # load the checkpoint, held-out quality
uv run python examples/04_decision_model.py --mode infer        # load the checkpoint, deployment cost
uv run python examples/01_train_and_pack.py --mode infer --kernel lut   # runtime options can change after training
```

- **train** goes through Fabric (`fab.setup`, `fab.backward`, or `lilbits.training.train`), with `--accelerator`,
  `--devices` and `--precision` passed through. Checkpoints are written and read with `fab.save` / `fab.load` and
  store the arguments that shaped the models, so test and infer rebuild exactly the same models.
- **test** is held-out quality: loss, accuracy, calibration, bits per byte, depth curves, whatever fits the example.
- **infer** is deployment: packing to 1- or 2-bit weights, LUT kernels, decode latency, cache and state sizes.
- Every phase reports wall time, CPU time and peak RSS growth, and every run reports params, weight RAM, deployed
  size and the checkpoint file size, ending with a `METRICS {json}` line.

Every example defaults to a short smoke run that proves the code path works; pass `--steps` to actually train.
`benchmarks/examples_bench.py` runs train, test and infer as three separate processes per example and tabulates
process wall, CPU and peak RSS next to each phase's own numbers (`results/examples_bench.md`).

See `docs/primitives.md` for how these map onto the target architecture.
