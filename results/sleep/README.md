# Sleep: examples 16 and 17

## Example 16: in-context adaptation via sleep (Lee et al. 2026, arXiv:2611.32802)

Every W = 32 bytes, the model loops its whole stack N - 1 more times over the window it just read and gates that
window's KV cache toward the refined keys and values (`models/sleep_lm.py`).

**Setup.** Run on CPU, one seed. The data is the ~10 MB corpus (`lilbits.corpus`: Shakespeare, Gutenberg and
webtext), with 25% of steps on few-shot key → value maps. Each map is a fresh random map over 26 × 26 letters per
sequence, and its 8 keys are shown back to back as "cKaF...". The model is d = 96 with 4 layers, 4 query heads and
1 KV head: 412,704 parameters without the gates. Training is batch 16 × 256 bytes for 2,016 steps, which is 8.26 MB:
**20 training bytes per parameter**, a Chinchilla-like data to weights ratio (`--bytes-per-param 20`).

    uv run python examples/16_sleep_lm.py --data corpus --d 96 --batch 16 --bytes-per-param 20 --keys 8 \
        --fewshot-frac 0.25 --loops 1   # and --loops 2, --loops 4; each N trained in its own process
    # test / infer: the three checkpoints merged into one, then --mode test --test-batches 16, --mode infer

| | N = 1 (no sleep) | N = 2 | N = 4 |
|---|---|---|---|
| final train loss (mean of the last 10% of steps, nats/byte) | 1.808 | 1.573 | **1.555** |
| step where the loss curve drops (few-shot task learned) | never in 2,016 steps | ~1,300 | ~1,000 |
| few-shot map accuracy, 1 / 2 / 4 / 6+ shots | 11% / 14% / 17% / 23% | 40% / 72% / 97% / 99.6% | 40% / 72% / 97% / **99.8%** |
| held-out bits per byte (full 256-byte windows) | 2.351 | 2.342 | **2.317** |
| last-window bpb with 32 / 224 bytes of prefix | 2.344 / 2.316 | 2.342 / 2.326 | 2.313 / 2.298 |
| sleep gate parameters | 0 | 9,408 (+2.3%) | 9,408 (+2.3%) |
| KV cache bytes per token | 768 | 768 | 768 |
| decode, bytes/s (1 thread, 64-byte prompt + 128 bytes) | 470 | 419 | 286 |
| one sleep (one window) | n/a | 4.6 ms | 11.7 ms |
| training step | 0.50 s | 0.88 s | 1.33 s (2 threads) |

0-shot accuracy is about 3% for every N (chance is 1/26).

**Read.** At the same data and the same size, sleep made the model learn the in-context lookup far earlier. N = 2
and N = 4 both learned it, and N = 4 learned it first. N = 1 had not learned it by the end of training: its
accuracy creeps up with the number of shots but stays under 25%. On held-out text, N = 4 is 0.034 bpb better,
N = 2 only 0.009 better. Test-time context efficiency is hard to read off the text curves. Even at 224 bytes of
prefix, N = 1 only gets 0.03 bpb below its 32-byte loss, so the `context_efficiency_vs_N=1` value of 7.1x for N = 4
just means that N = 4 with 32 bytes of prefix already beats N = 1 with 224. The paper's 2.1x / 2.4x is a
different measurement, made on 1.2B models with 16k contexts.

Caveats: one seed. The step at which a model learns the few-shot task jumps around between seeds, so N = 1 may
learn it with more steps; what this run shows is that sleep got there with less data. Training costs about N
times as much compute per step (0.50 / 0.88 / 1.33 s). Decode gets slower than the per-sleep cost alone would
explain, because this Python decode path spends overhead on the cache concatenations.

Files: `16_train_N{1,2,4}.json` (training metrics), `16_test_corpus_20bpp.json`, `16_infer.json`.
