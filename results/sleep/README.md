# Sleep: examples 16 and 18

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

## Example 18: continual learning with sleep consolidation (Behrouz et al. 2026, arXiv:2606.03979)

The model lives through 3 sessions of 6 new facts each, such as "Kalomi Tenvor's number is 4821.". While awake it
writes them into fast memory, a rank-16 adapter on every MLP. Sleep unlocks a new rank-16 routed expert per block
and seeds it from the fast-memory model (GKD on- and off-policy plus the LTI edit-distance reward, 200 steps). Then
it dreams, selects dreams by gradient norm and runs SFT on them, and finally resets the fast memory
(`models/sleep_consolidation.py`).

**Setup.** The base is d = 64 with 4 layers (236,224 parameters), pretrained on 4.7 MB of the corpus mixed with
sentences about random names: **20 bytes per parameter**. Recall is greedy and exact on all 4 digits, with no
context and fast memory off.

    uv run python examples/18_sleep_consolidation.py --d 64 --bytes-per-param 20 --wake-steps 300 \
        --consolidate-steps 200 --dream-steps 60 --expert-rank 16 [--anchor 0]

| | recall while awake (fast memory on) | recall of sessions 1 / 2 / 3 after all sessions | held-out bpb (base 2.622) |
|---|---|---|---|
| fast only (memory reset after each session) | 100% / 100% / 100% | 0 / 0 / 0 | 2.622 |
| continual SFT (one slow adapter) | 100% / 100% / 100% | 0 / 0 / **100%** | 5.397 |
| sleep (with anchor) | 100% / 100% / 100% | 0 / 0 / 0 | 3.484 |
| sleep, no dreaming (with anchor) | 100% / 100% / 100% | 0 / 0 / 0 | **2.658** |
| sleep, without anchor | 100% / 100% / 83% | 0 / 0 / 0 | 7.591 |
| sleep, no dreaming, without anchor | 100% / 100% / 100% | 0 / 0 / 0 | 5.737 |

**Read: consolidation does not work yet at this size.** Wake writes every fact into fast memory, and continual SFT
shows the catastrophic forgetting the paper targets: it keeps only the last session. Sleep kept no exact numbers,
not even right after its own session. Distilling into the new expert stalls at a KL of about 0.12 to 0.15 nats per
byte. That gets the format right but not the 4 digits. The "anchor", which is an addition of ours and not in the
paper, adds a KL to the pre-session model on general text. It is what keeps the base intact (2.66 bpb vs 5.74
without it), while dreaming hurts the base (3.48). Slow memory costs 16.9 KB per session (3.6% of the base in
FP16), and decode with 3 experts on runs about as fast as the base.

Likely next steps: more seeding steps or a higher expert rank; dream prompts that end exactly at "number is "; or
the paper's in-context teacher (the facts in context) in place of the fast adapter. One seed.

Files: `18_anchor1.json`, `18_anchor0.json`.
