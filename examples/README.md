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

Every example defaults to a short smoke run that proves the code path works; pass `--steps` to actually train.
Examples that train print a `METRICS {json}` line with train wall/CPU/peak RSS, inference cost, RAM weights,
checkpoint and stored size. `benchmarks/examples_bench.py` runs them all and tabulates it
(`results/examples_bench.md`).

See `docs/primitives.md` for how these map onto the target architecture.
