| example | ok | process wall | process CPU (util) | process peak RSS | train wall / CPU | train peak RSS growth | inference (example's own measure) | RAM weights | checkpoint file | stored size |
|---|---|---|---|---|---|---|---|---|---|---|
| 01 train+pack, LittleBit | ✅ | 11.8 s | 12.0 s (1.02) | 898 MiB | 7.5 s / 7.8 s (60 steps) | 103 MiB | 27.9 ms wall, 28.9 ms CPU per 32×63 batch | 137 KiB | 160 KiB | 101 KiB |
| 01 train+pack, BitNet | ✅ | 12.5 s | 12.6 s (1.01) | 901 MiB | 8.0 s / 8.2 s (60 steps) | 133 MiB | 36.7 ms wall, 38.5 ms CPU per 32×63 batch | 219 KiB | 175 KiB | 110 KiB |
| 01 train+pack, FP | ✅ | 10.4 s | 10.5 s (1.01) | 894 MiB | 6.0 s / 6.1 s (60 steps) | 89 MiB | 19.7 ms wall, 20.5 ms CPU per 32×63 batch | 1,610 KiB | 1,618 KiB | 805 KiB |
| 02 recurrent mixers | ✅ | 1.5 s | 1.5 s (0.98) | 641 MiB |  |  |  |  |  |  |
| 03 state rollback | ✅ | 1.3 s | 1.3 s (1.00) | 630 MiB |  |  |  |  |  |  |
| 04 decision, LittleBit GDN | ✅ | 24.5 s | 25.0 s (1.02) | 1202 MiB | 19.0 s / 19.7 s (30 steps) | 430 MiB | 16.4 ms prefill / state, 0.65 ms / question | 962 KiB | 1,003 KiB | 282 KiB |
| 04 decision, BitNet GDN | ✅ | 28.0 s | 28.7 s (1.02) | 1223 MiB | 22.4 s / 23.3 s (30 steps) | 458 MiB | 20.2 ms prefill / state, 1.03 ms / question | 1,463 KiB | 1,460 KiB | 299 KiB |
| 04 decision, FP GDN | ✅ | 23.5 s | 24.1 s (1.03) | 1190 MiB | 18.6 s / 19.2 s (30 steps) | 425 MiB | 15.4 ms prefill / state, 0.70 ms / question | 1,463 KiB | 1,460 KiB | 721 KiB |
| 04 decision, LittleBit Mamba2 | ✅ | 19.8 s | 20.1 s (1.02) | 1144 MiB | 14.5 s / 15.0 s (30 steps) | 373 MiB | 9.5 ms prefill / state, 0.56 ms / question | 1,123 KiB | 1,149 KiB | 382 KiB |
| 05 pack + Helion (interpret) | ✅ | 16.8 s | 17.1 s (1.01) | 1060 MiB | 5.9 s / 6.1 s (60 steps) | 107 MiB | eval + generate: torch 0.11 s, Helion (interpret) 6.16 s wall | 137 KiB | 160 KiB | 101 KiB |
