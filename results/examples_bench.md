| example | ok | process wall | process CPU (util) | process peak RSS | train wall / CPU | train peak RSS growth | inference (example's own measure) | RAM weights | checkpoint file | stored size |
|---|---|---|---|---|---|---|---|---|---|---|
| 01 train+pack, LittleBit | ✅ | 12.6 s | 12.8 s (1.02) | 899 MiB | 8.0 s / 8.3 s (60 steps) | 105 MiB | 31.0 ms wall, 31.9 ms CPU per 32×63 batch | 137 KiB | 160 KiB | 101 KiB |
| 01 train+pack, BitNet | ✅ | 15.1 s | 15.3 s (1.02) | 902 MiB | 10.0 s / 10.4 s (60 steps) | 134 MiB | 40.5 ms wall, 41.8 ms CPU per 32×63 batch | 219 KiB | 175 KiB | 110 KiB |
| 01 train+pack, FP | ✅ | 9.0 s | 9.0 s (1.01) | 898 MiB | 5.2 s / 5.4 s (60 steps) | 91 MiB | 16.1 ms wall, 16.7 ms CPU per 32×63 batch | 1,610 KiB | 1,618 KiB | 805 KiB |
| 02 recurrent mixers | ✅ | 1.2 s | 1.1 s (0.97) | 641 MiB |  |  |  |  |  |  |
| 03 state rollback | ✅ | 1.1 s | 1.1 s (1.00) | 630 MiB |  |  |  |  |  |  |
| 04 decision, LittleBit GDN | ✅ | 28.4 s | 29.3 s (1.03) | 1207 MiB | 23.3 s / 24.3 s (30 steps) | 436 MiB | 15.7 ms prefill / state, 0.65 ms / question | 962 KiB | 1,003 KiB | 282 KiB |
| 04 decision, BitNet GDN | ✅ | 27.2 s | 27.9 s (1.03) | 1224 MiB | 22.4 s / 23.2 s (30 steps) | 458 MiB | 15.5 ms prefill / state, 0.61 ms / question | 1,463 KiB | 1,460 KiB | 299 KiB |
| 04 decision, FP GDN | ✅ | 24.7 s | 25.2 s (1.02) | 1191 MiB | 19.0 s / 19.7 s (30 steps) | 426 MiB | 16.9 ms prefill / state, 0.81 ms / question | 1,463 KiB | 1,460 KiB | 721 KiB |
| 04 decision, LittleBit Mamba2 | ✅ | 20.2 s | 20.7 s (1.02) | 1141 MiB | 15.1 s / 15.7 s (30 steps) | 370 MiB | 10.7 ms prefill / state, 0.71 ms / question | 1,123 KiB | 1,149 KiB | 382 KiB |
| 05 pack + Helion (interpret) | ✅ | 15.4 s | 15.6 s (1.01) | 1054 MiB | 5.3 s / 5.4 s (60 steps) | 106 MiB | eval + generate: torch 0.11 s, Helion (interpret) 5.71 s wall | 137 KiB | 160 KiB | 101 KiB |
| 06 looped LM, FP | ✅ | 13.0 s | 13.3 s (1.03) | 924 MiB | 9.3 s / 9.7 s (40 steps) | 265 MiB | adaptive exit: 6.0 loops / position, 148 ms per 32×63 batch | 1,843 KiB | 1,859 KiB | 921 KiB |
| 06 looped LM, LittleBit core | ✅ | 15.3 s | 15.6 s (1.02) | 959 MiB | 10.8 s / 11.2 s (40 steps) | 292 MiB | adaptive exit: 6.5 loops / position, 156 ms per 32×63 batch | 1,390 KiB | 1,444 KiB | 516 KiB |
| 06 looped LM, Ouro exit gate | ✅ | 15.2 s | 15.4 s (1.02) | 990 MiB | 11.3 s / 11.6 s (40 steps) | 330 MiB | adaptive exit: 2.1 loops / position, 64 ms per 32×63 batch | 1,843 KiB | 1,860 KiB | 922 KiB |
| 07 byte-level: baseline / BLT / H-Net | ✅ | 46.3 s | 47.7 s (1.03) | 1034 MiB |  |  | byte transformer: 4.54 bpb, 1.0 B/patch, train 7.7s, infer 89 ms, 1,352 KiB; BLT: 4.78 bpb, 4.4 B/patch, train 8.3s, infer 93 ms, 3,394 KiB; H-Net: 4.25 bpb, 2.6 B/patch, train 26.8s, infer 91 ms, 1,198 KiB |  |  |  |
