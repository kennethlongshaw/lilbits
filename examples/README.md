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
| `07_byte_level_patching.py` | tokenizer-free byte models: BLT (entropy patches, hash n-gram embeddings) and H-Net (learned chunks, ratio loss) vs a plain byte transformer: bits/byte, bytes per patch, full cost metrics (`--latent-linear`) |
| `08_llm_jepa.py` | LLM-JEPA: next-token loss plus a JEPA embedding loss on paired views (Text, Code), [PRED] tokens as the predictor, block-causal packing in one pass, EMA or online target, loss dropout; compares all variants on quality and full train/infer cost (`--linear`, `--k-pred`, `--ema-decay`) |
| `09_looped_index_lm.py` | the full lean-decoder design: LittleBit GDN prelude, looped core with shared signs + per-loop scales, per-loop GDN state, loop memory, sparse index (4-bit K/V, sign signatures, block summaries, dense fallback), FP coda with MTP drafts, parallel verify with rollback and adaptive k, decode halting; trained in 5 stages next to a dense FP reference on recall / two-hop data, with memory-tier bytes and full cost metrics (`--steps`, `--length`, `--loops`, `--bpw-alloc greedy` for calibrated uneven core ranks, `--kernel lut`) |
| `10_pocket_lm.py` | byte-level pocket model: byte encoder, exact byte-pair router (space-patching init) so the body runs once per patch, example 09's looped index body with a sub-1-bit INT8-scale LittleBit core, byte decoder, MTP drafts with exact verify; every piece a switch (`--router pair|hnet|none`, `--core fp|littlebit`, `--no-index`, `--mtp`, `--kernel lut`), next to an FP16 byte GDN LM matched on deployed bytes |

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
