# Example 13: soft vs straight-through router (smoke, count task)

20 pretrain + 40 warm-up + 20 SFT steps, batch 8, d 96, 4 loops, 1 thread (2026-11-25).

| ops model | pretrain loss (start -> end) | warm-up router CE | SFT task loss | count score | pretrain wall s |
|---|---|---|---|---|---|
| soft (every op runs) | 5.40 -> 3.19 | 0.10 | 0.96 | 0.35 | 52.5 |
| straight-through (one-hot forward) | 5.56 -> 5.52 (does not learn) | 1.86 | 3.40 | 0.00 | 47.8 |

Straight-through saves under 10% of time: across B x T positions almost every op is chosen somewhere, so nearly all
ops still run. Real savings need per-position dispatch (gather each op's positions). It also stops learning from
the first step (one-hot routing sends positions into ops that read still-empty registers). Kept as `--route st`
for further work; the default stays soft.
