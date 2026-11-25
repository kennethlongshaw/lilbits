# Example 13 single-epoch runs (2026-11-25)

d 96, 4 loops, batch 8, seq 200, cosine. Shared seed stage, run once and loaded by every task (`--init`):
Shakespeare LM pretraining (644 steps = 1 pass over the train bytes), then router warm-up on mixed problems from all
five tasks with teacher-forced op chains (500 steps). Then 500 SFT steps per task (4,000 problems = 1 "epoch"),
router free. Path adds 100 GRPO steps (group 4, KL 0.05, fixed k3 estimator). Count and path ran on 4 threads;
lookup, argmax and sort ran two at a time on 2 threads each, so compare wall times within a task, not across tasks.

## Seed stage (from the count run, 4 threads)

| model | stage | end loss | wall s | CPU (cores) | peak RAM growth MiB |
|---|---|---|---|---|---|
| ops | pretrain | 1.89 | 588 | 3.85 | 1,061 |
| ops | warm-up | 0.585 (router CE 0.0007) | 500 | 3.86 | 1,026 |
| attend | pretrain | 1.77 | 53 | 3.98 | 203 |
| attend | warm-up | 0.566 | 43 | 3.96 | 189 |

## Tasks

| task | model | score (4 loops) | expected op chain | SFT answer loss | SFT wall s | SFT CPU | SFT peak RAM MiB | GRPO | answer latency ms | test wall s | params | deployed KiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| count | ops | 0.46 | 1.0 | 0.826 | 537 | 3.8 | 1027 | - | 15 | 2.9 | 352k | 687 |
| count | attend | 0.46 | - | 0.827 | 72 | 4.0 | 208 | - | 12 | 2.9 | 351k | 685 |
| lookup | ops | 0.00 | 1.0 | 1.447 | 933 | 1.9 | 1063 | - | 13 | 3.1 | 352k | 687 |
| lookup | attend | 0.02 | - | 1.410 | 76 | 2.0 | 208 | - | 14 | 3.1 | 351k | 685 |
| argmax | ops | 0.04 | 0.1 | 0.411 | 917 | 1.9 | 1060 | - | 41 | 7.5 | 352k | 687 |
| argmax | attend | 0.04 | - | 0.427 | 74 | 2.0 | 202 | - | 46 | 7.5 | 351k | 685 |
| sort | ops | 0.00 | 1.0 | 0.292 | 930 | 2.0 | 1061 | - | 73 | 26.9 | 352k | 687 |
| sort | attend | 0.06 | - | 0.227 | 65 | 2.0 | 211 | - | 70 | 26.9 | 351k | 685 |
| path | ops | 0.00 | 0.74 | 0.579 | 555 | 3.8 | 1059 | 587 s, reward 0, 4,115 MiB | 116 | 36.2 | 352k | 687 |
| path | attend | 0.00 | - | 0.518 | 46 | 4.0 | 192 | 188 s, reward 0, 720 MiB | 122 | 36.2 | 351k | 685 |

Checkpoint files: 2.8 MB per task (both models).

## Findings

- Neither model learns any task at this budget, including lookup, the control that attention alone should solve.
  Count sits at 0.46 for both: always answering the most common count.
- With the router task summary, routing follows the intended chain (count: filter -> count on 100%; lookup, sort:
  pointer/attend on 100%). Without it (13_count_before_router_summary.log) every task got the same chain.
- Right routing is not enough: on count the filter marks almost every byte (set mass ~55 of ~60 bytes), not the
  queried name, so count returns a length, not a count. Byte-level keys cannot match a multi-byte name in one step.
- Argmax drifts off `max` during router-free SFT (expected chain 0.1).
- GRPO on path never sampled a valid path (reward 0 throughout), so it cannot learn. Before the KL fix it also
  destroyed the policy (13_path_kl_bug.log: output became noise); after the fix it stays put (KL 0.0045).
- Cost: the operator model's SFT is 7 to 14x the attention loop's wall time and 5x its peak RAM; GRPO peaks at
  4.1 GiB vs 0.7 GiB.
