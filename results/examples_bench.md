| example | ok | process wall | process CPU (util) | process peak RSS | train wall / CPU | train peak RSS growth | inference (example's own measure) | RAM weights | checkpoint file | stored size |
|---|---|---|---|---|---|---|---|---|---|---|
| 01 train+pack, LittleBit | ✅ | 11.3 s | 11.4 s (1.01) | 903 MiB | 6.6 s / 6.8 s (60 steps) | 101 MiB | 32.0 ms wall, 33.8 ms CPU per 32×63 batch | 137 KiB | 160 KiB | 101 KiB |
| 01 train+pack, BitNet | ✅ | 15.5 s | 15.8 s (1.02) | 904 MiB | 10.3 s / 10.7 s (60 steps) | 131 MiB | 42.0 ms wall, 43.2 ms CPU per 32×63 batch | 219 KiB | 175 KiB | 110 KiB |
| 01 train+pack, FP | ✅ | 9.0 s | 9.1 s (1.01) | 893 MiB | 5.0 s / 5.1 s (60 steps) | 92 MiB | 15.9 ms wall, 16.3 ms CPU per 32×63 batch | 1,610 KiB | 1,618 KiB | 805 KiB |
| 02 recurrent mixers | ✅ | 1.6 s | 1.5 s (0.97) | 641 MiB |  |  |  |  |  |  |
| 03 state rollback | ✅ | 1.5 s | 1.5 s (1.00) | 630 MiB |  |  |  |  |  |  |
| 04 decision, LittleBit GDN | ✅ | 23.7 s | 24.2 s (1.02) | 1207 MiB | 18.7 s / 19.3 s (30 steps) | 435 MiB | 14.2 ms prefill / state, 0.53 ms / question | 962 KiB | 1,003 KiB | 282 KiB |
| 04 decision, BitNet GDN | ✅ | 25.3 s | 26.0 s (1.02) | 1242 MiB | 20.4 s / 21.1 s (30 steps) | 476 MiB | 16.0 ms prefill / state, 0.66 ms / question | 1,463 KiB | 1,460 KiB | 299 KiB |
| 04 decision, FP GDN | ✅ | 22.9 s | 23.4 s (1.02) | 1186 MiB | 17.0 s / 17.6 s (30 steps) | 420 MiB | 16.6 ms prefill / state, 0.89 ms / question | 1,463 KiB | 1,460 KiB | 721 KiB |
| 04 decision, LittleBit Mamba2 | ✅ | 22.5 s | 22.9 s (1.02) | 1135 MiB | 17.0 s / 17.5 s (30 steps) | 364 MiB | 11.6 ms prefill / state, 0.79 ms / question | 1,123 KiB | 1,149 KiB | 382 KiB |
| 05 pack + Helion (interpret) | ✅ | 18.0 s | 18.2 s (1.02) | 1061 MiB | 6.3 s / 6.6 s (60 steps) | 103 MiB | eval + generate: torch 0.15 s, Helion (interpret) 7.08 s wall | 137 KiB | 160 KiB | 101 KiB |
| 06 looped LM, FP | ✅ | 14.8 s | 15.1 s (1.02) | 928 MiB | 10.8 s / 11.2 s (40 steps) | 269 MiB | adaptive exit: 6.0 loops / position, 159 ms per 32×63 batch | 1,843 KiB | 1,859 KiB | 921 KiB |
| 06 looped LM, LittleBit core | ✅ | 13.5 s | 13.7 s (1.02) | 958 MiB | 9.8 s / 10.1 s (40 steps) | 291 MiB | adaptive exit: 6.5 loops / position, 167 ms per 32×63 batch | 1,390 KiB | 1,444 KiB | 516 KiB |
| 06 looped LM, Ouro exit gate | ✅ | 13.9 s | 14.2 s (1.02) | 989 MiB | 10.4 s / 10.7 s (40 steps) | 330 MiB | adaptive exit: 2.1 loops / position, 67 ms per 32×63 batch | 1,843 KiB | 1,860 KiB | 922 KiB |
| 07 byte-level: baseline / BLT / H-Net | ✅ | 46.6 s | 47.8 s (1.03) | 1030 MiB |  |  | byte transformer: 4.45 bpb, 1.0 B/patch, train 6.9s, infer 95 ms, 1,352 KiB; BLT: 4.79 bpb, 3.7 B/patch, train 7.9s, infer 94 ms, 3,394 KiB; H-Net: 4.19 bpb, 3.0 B/patch, train 28.2s, infer 126 ms, 1,198 KiB |  |  |  |
| 08 LLM-JEPA (LM / online / EMA / EMA+dropout) | ✅ | 4.8 s | 4.9 s (1.01) | 784 MiB |  |  | LM loss only: code loss 0.075, retrieval 0.00, train 0.25s, infer 14 ms, EMA copy 0 KiB; LLM-JEPA, online target: code loss 0.190, retrieval 0.02, train 0.46s, infer 15 ms, EMA copy 0 KiB; LLM-JEPA, EMA target: code loss 0.083, retrieval 0.08, train 0.32s, infer 13 ms, EMA copy 394 KiB; LLM-JEPA, EMA + loss dropout 0.5: code loss 0.100, retrieval 0.06, train 0.26s, infer 13 ms, EMA copy 394 KiB |  |  |  |
