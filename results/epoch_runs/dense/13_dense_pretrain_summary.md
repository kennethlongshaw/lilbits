# Ex13 with dense next-token pretraining (easy lookup, level 0)

Run: `--task lookup --level 0 --pretrain-steps 2000 --warmup-steps 300 --steps 1500 --grpo-steps 50 --eval 100`, 4 threads, batch 4 x 200 bytes, d=96, 4 loops.

| stage | model | steps | final loss (CE) | wall s | CPU s | peak RAM MiB |
|---|---|---|---|---|---|---|
| pretrain | ops | 2000 | 1.910 | 2642 | 2886 | 544 |
| pretrain | attend | 2000 | 1.911 | 277 | 302 | 105 |
| warm-up | ops | 300 | 1.923 (router CE 0.001) | 407 | 445 | 521 |
| warm-up | attend | 300 | 1.757 | 32 | 35 | 98 |
| SFT | ops | 1500 | 1.573 | 2122 | 2307 | 519 |
| SFT | attend | 1500 | 2.082 | 152 | 165 | 103 |
| GRPO | both | 0 | skipped by gate: SFT sample reward 0.0 | | | |

Held-out score (100 problems, 1 to 4 loops): ops 0.00, attend 0.00 (word-key run without dense pretraining: ops 0.00 to 0.05, attend 0.11 to 0.13).
Ops router sends every lookup problem through count -> count -> count -> count.
Inference: ops 17.9 ms, attend 11.7 ms per answer; eval peak extra RAM under 1 MiB.
Size: ops 360,840 params (722 KB deployed), attend 359,967 params (720 KB); checkpoint file 2.9 MB.

Teacher-forced byte accuracy on trained bytes after pretraining + warm-up:

| model | dense recall streams | easy lookup answers (digits + newline) |
|---|---|---|
| ops | 0.026 | 0.385 |
| attend | 0.043 | 0.401 |

Random digit guessing would score about 0.10 on recall values, so neither model learned in-context retrieval during pretraining.
