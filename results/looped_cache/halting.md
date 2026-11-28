# Halting a looped block: what the literature says, and what held up on our GDN byte models

Context: example 15's `rloop:` models (one or two shared GDN blocks, trained with a random loop count per step, 20
bytes/param, mixed corpus; see README section 4). Here a byte may stop looping early. Two questions: which exit signal,
and what happens to the per-loop GDN states of the loops a byte skips.

## Methods, in brief

| family | how it decides | training needed | notes |
|---|---|---|---|
| ACT (Graves 2016), Universal Transformer (Dehghani 2018) | learned halting unit, stop when cumulative p > 1 - eps, ponder cost | yes, joint | sensitive to the ponder-cost weight |
| PonderNet (Banino 2021) | per-step halting prob = geometric exit distribution, KL to a geometric prior | yes, joint | unbiased gradients, prior sets the expected depth |
| Ouro / LoopLM exit gate ([Zhu et al. 2025](https://arxiv.org/pdf/2510.25741)) | gate lambda = sigmoid(w.h), exit at CDF >= q; entropy-regularized expected loss, then a stage that trains the gate to predict whether another step lowers loss | yes | the repo's `recurrent_depth.py` / example 06 has it |
| Huginn convergence ([Geiping et al. 2025](https://arxiv.org/pdf/2502.05171)) | KL(p_r ‖ p_{r-1}) < t | no (random-depth training) | needs the head every loop (O(V d)) |
| Relative state change (repo: `looped_index.py`, example 09) | ‖h_r − h_{r-1}‖ / ‖h_r‖ < t | no | O(d) |
| Acceleration / two-scale ([2509.23314](https://arxiv.org/html/2509.23314v2)) | ‖Δ_r − Δ_{r-1}‖ < t for 2 steps (Δ_r = h_r − h_{r-1}) | no | O(d); the step norm plateaus while the direction still rotates, so plain step-norm mis-halts; at matched quality lower latency than KL |
| Confidence (CALM, Schuster 2022) | top-1 prob / entropy / margin > t, calibrated with Learn-then-Test | no | strong on classification-like outputs |
| TRM ([Jolicoeur-Martineau 2025](https://arxiv.org/html/2510.04871v1)) | one BCE head on y: "is the current answer correct?"; stop when q > 0 | yes, deep supervision | HRM used Q-learning halt + continue heads (2 forward passes); TRM drops the continue head (1 pass). 2-layer, 5-7M params: Sudoku-Extreme 87.4% (HRM 55.0%), ARC-AGI-1 45%, ARC-AGI-2 8% |

Recent findings that matter for us:
- **Train the trajectory, read it post hoc.** [Diagnosing learned halting gates (2607.20519)](https://arxiv.org/html/2607.20519v1):
  a jointly trained gate both picks the exit and weights the per-depth losses, so it degrades the early loops it is
  supposed to read. Fixed-prior (random-depth) training with a simple confidence or KL readout needed far fewer loops
  (D@95 1.38 vs 2.80 for a learned linear gate); on pretrained Ouro, post-hoc readouts beat the gate about half the time.
- **Untrained KL is as good as a trained probe.** [Selection rules (2611.02333)](https://arxiv.org/pdf/2611.02333), on
  Ouro-1.4B: KL between successive loops tied a trained probe (macro saving 0.238 both); how the threshold is chosen
  flips which method "wins", and 4 of 9 test upper bounds broke the 1 pp budget. They recommend formal risk control
  (Learn-then-Test, conformal) on held-out data, plus an out-of-distribution check.
- **Recurrent states of skipped loops ("cache holes").** [CHASE, looped Mamba-2 (2607.10110)](https://www.alphaxiv.org/abs/2607.10110.md):
  if an exited token simply stops updating the deeper loops' SSM states, perplexity can explode (+4788% in one
  setting). Their fix trains with the holes, and then stays within +3-4% perplexity at ~2 of 4 loops. For attention, Ragged KV
  ([2610.09013](https://arxiv.org/pdf/2610.09013)) reuses the exited token's last K/V for deeper loops: 30% fewer
  FLOPs, GSM8K 75.3 vs 76.6 at 3.05 of 4 average loops. They also note a per-token exit penalty pushes every token to
  the same shallow depth; put the prior on the batch-average exit distribution instead.
- **Batching.** [Continuous depth batching (2608.09444)](https://arxiv.org/pdf/2608.09444) gets 1.3-1.5x throughput on
  GPUs by regrouping tokens between loop steps. Not needed for us: at batch 1 on CPU a skipped loop is simply not run.
- **Past the training horizon.** TRM-style models trained with deep supervision (each step refines a carried answer)
  keep improving past their 16 training steps on Sudoku: [2610.16427](https://arxiv.org/pdf/2610.16427) finds that
  continuing recovers 73.8% of failures (15.6% on another benchmark), and resampling the answer state y is a better use
  of extra compute than perturbing z. Our plain loops get worse past the trained count (README section 4), so deep
  supervision is the known recipe if we want open-ended thinking.
- TRM on text: one forum write-up ([TinyStories](https://discuss.google.dev/t/tiny-recursive-models-for-text-generation/318588))
  with causal attention and no baseline. Nothing shows it beats a plain LM on text yet.
- Looped transformers more broadly (Saunshi et al. 2025, "Reasoning with latent thoughts"; Yang et al. 2023; Fan et al.
  2024, length generalization) show looping helps algorithmic and reasoning tasks more than perplexity. That
  matches what we saw on bpb.

## What we measured

Byte-by-byte decode of 8 held-out batches (8 x 128 bytes, mixed corpus), PyTorch, every byte allowed 1..R loops.
Signals and thresholds swept (`examples/15_looped_cache_lm.py --mode test --halt 8`): KL, relative state change
`dh`, acceleration `accel`, top-1 confidence `conf`. Two state policies for loops a byte skips:
- **freeze**: their GDN states are not updated for that byte (cache holes);
- **write-through**: they still advance their states from the exited hidden state (outputs discarded), as
  `looped_index.py` does. Cost: with one shared block, the skipped loops all see the same frozen input with the same
  weights, so k, v, a, b are computed once (~15% of a block) and only the cheap per-loop state updates repeat.

"fixed@same" interpolates the fixed-depth curve at the same mean loops: a halting rule only helps if it beats that.

rloop:128,1,8 (fixed depth: 1: 2.647, 2: 2.444, 3: 2.398, 4: 2.380, 5: 2.374, **6: 2.373**, 7: 2.375, 8: 2.380):

| signal, threshold | mean loops | bpb freeze | bpb write-through | fixed@same |
|---|---|---|---|---|
| KL 3e-3 | 4.5-4.9 | 2.417 | **2.379** | 2.375-2.377 |
| KL 1e-2 | 3.4-3.6 | 2.421 | **2.385** | 2.387-2.390 |
| KL 3e-2 | 2.8 | 2.432 | **2.398** | 2.405-2.409 |
| accel 0.03 | 5.0-5.6 | 2.384 | **2.375** | 2.373-2.374 |
| accel 0.1 | 3.6-3.9 | 2.393 | 2.385 | 2.382-2.387 |
| dh 0.2 | 4.3 | 2.389 | 2.380 | 2.378 |
| conf 0.9 | 7.2 | 2.414 | 2.384 | 2.376 |
| conf 0.7 | 5.8-6.0 | 2.548 | 2.426 | 2.373 |

rloop:160,1,6 (fixed: 1: 2.531, 2: 2.330, 3: 2.292, **4-5: 2.282**, 6: 2.288):

| signal, threshold | mean loops | bpb freeze | bpb write-through | fixed@same |
|---|---|---|---|---|
| KL 3e-3 | 4.3-4.5 | 2.304 | 2.288 | 2.282 |
| KL 1e-2 | 3.4-3.6 | 2.316 | 2.288 | 2.286-2.288 |
| accel 0.1 | 3.7-4.0 | 2.297 | 2.284 | 2.282-2.285 |
| dh 0.2 | 4.3 | 2.292 | 2.284 | 2.282 |
| conf 0.9 | 5.3 | 2.341 | 2.295 | 2.284 |

rloop:96,2,4 and rloop:128,1,4 behave the same way (`halting_rows.json`).

- **Advance the skipped loops' states.** Freezing them costs 0.01-0.04 bpb at every threshold, the same failure CHASE
  reports for SSMs. With write-through, KL / accel / dh land within ~0.006 bpb of fixed depth at the same average loops (sometimes
  slightly better).
- **KL and acceleration are the good signals; confidence is the worst.** Byte prediction is high-entropy, so a
  confident early byte is not a converged one: conf 0.7 costs 0.05-0.17 bpb. Acceleration matches KL at O(d) cost
  instead of a 256 x d head per loop (15% of a d128 block, 4% at d512).
- **On next-byte bpb, halting ≈ picking the right fixed depth.** The best rules land on the fixed-depth curve, with no
  clear gain beyond it (differences ≤ 0.005, within noise). Per-byte adaptivity has little to exploit in this text at this size;
  it saves loops relative to the max trained depth (KL 3e-3 runs 4.5 of 8 loops at equal bpb), which a fixed depth of 5 also
  does.

## Top 3 methods head to head: quality per unit of compute

The three most promising rules from the survey and the test above, all with write-through states, against plain fixed
depth, scored on bpb vs **MACs per byte with every cost counted**: block applications, the exit signal (KL: the
256 x d head at every loop; acceleration: O(d); probe: O(d)), the write-only k/v/a/b projection of each exited byte
and the skipped loops' state updates, embedding and readout.
- **acceleration** `|Δh_r − Δh_{r-1}| / |h_r| < t` (untrained; first possible exit after loop 3)
- **KL** between successive loops' next-byte distributions `< t` (untrained, Huginn)
- **learned exit probe** (Ouro stage 2 / TRM-style, fit post hoc): ridge regression on O(d) features (normed hidden,
  |Δh|, |accel|, loop index) of the *remaining gain* `CE_r − min_{r'>r} CE_r'`, fit on 32 x 8 x 128 training bytes of
  full-depth trajectories; exit when the predicted gain is `< t` nats

Same 8 held-out batches (8 KB) as above, paired. "vs fixed" = bpb of the rule minus fixed depth interpolated at the same
MACs (negative = the rule is better); best point on each rule's curve:

| model (fixed-depth curve, MACs/B: bpb) | acceleration | KL | learned probe |
|---|---|---|---|
| rloop:128,1,8 (261K: 2.647, 946K: 2.380, 1175K: 2.374, 1403K: **2.373**) | +0.001 (2.378 @ 1076K) | +0.005 (2.385 @ 963K) | +0.003 (2.380 @ 1055K) |
| same, min 3 loops before any exit | +0.001 | +0.001 | +0.003 |
| rloop:160,1,6 (400K: 2.531, 1116K: 2.292, 1475K: **2.282**) | +0.000 (2.286 @ 1346K) | +0.003 (2.285 @ 1622K) | +0.005 (2.287 @ 2018K) |
| rloop:96,2,4 (284K: 2.380, 544K: 2.341, 803K: **2.336**) | +0.000 (2.336 @ 854K) | +0.002 (2.340 @ 645K) | +0.002 (2.341 @ 648K) |

MACs to get within 0.005 bpb of the best fixed depth: rloop:128,1,8 fixed 1043K, accel 1086K, probe 1199K, KL never;
rloop:160,1,6 fixed 1297K, accel 1318K, KL 1508K, probe never; rloop:96,2,4 fixed 548K, KL 640K, probe 684K, accel 854K.

Ranking, most compute-efficient first:
1. **Fixed depth at the right loop count.** Nothing beat it once all costs are counted. Almost all of the quality
   per MAC is in loops 1 → 3-4 (rloop:128,1,8: −0.27 bpb for +685K MACs from 1 to 4 loops, then −0.007 more for
   +457K to 6).
2. **Acceleration** ties fixed depth (+0.000 to +0.001) at almost no signal cost. It is the one to use if adaptivity
   is wanted, e.g. when the right depth varies by input in a way bpb doesn't show.
3. **Learned probe** (+0.002 to +0.005). It is cheap at inference, but on this data it predicts remaining gain no better than the
   untrained signals.
4. **KL** (+0.001 to +0.005): the head at every loop is 15% of a d128 block, which eats its savings. At d512 the head
   is ~4%, so KL would close most of that gap.

Why adaptivity doesn't pay here: the write-through cost (~13% of a block per exited byte, plus state updates) and the
signal's cost roughly cancel the loops saved, and on next-byte text the bytes that stop early are not much easier than
a fixed depth already handles. Differences of ≤ 0.005 bpb are at the edge of this 8 KB eval's resolution; one seed.

## Recommendation for the flagship loop
1. Train the trajectory with random depth (fixed prior), not a jointly trained gate. Add a gate later only if it beats
   the post-hoc readouts on held-out data.
2. Default to a fixed loop count picked on held-out data (the best compute per quality measured). If per-input
   adaptivity is needed, exit on acceleration (or KL if the head is computed anyway), with a minimum of 2-3 loops, thresholds calibrated on
   held-out data with a risk-control rule (target: ≤ X bpb loss vs full depth), and checked on a second corpus.
3. Always advance skipped loops' GDN states (write-through). In the Rust engine that means: after a byte exits at loop
   r, run one k/v/a/b projection of the exited hidden state and the remaining loops' state updates. That's an engine change, not
   done yet.
4. If extra thinking past the trained depth is the goal, the evidence points to TRM-style deep supervision (carry an
   answer state across supervised refinement steps, a correctness head for halting), tested on a task where compute
   should help (example 12), not on bpb.
