# Example 14, lean LM: 150 steps x 8 x 256 bytes of Shakespeare (small sample), 1 CPU thread

| | lean (GDN + LittleBit 0.55 bpw + MTP + LUT) | fp16_size (same deployed bytes) | fp16_width (same width) |
|---|---|---|---|
| width / layers | 256 / 4 | 68 / 4 | 256 / 4 |
| params | 128,544 | 257,208 | 3,479,072 |
| deployed | 549 KB | 514 KB | 6.96 MB |
| final train CE | 2.028 (MTP CE 3.00) | 2.101 | 1.743 |
| held-out bpb | 3.073 (LUT-packed 3.091) | 3.101 | 2.759 |
| train wall / CPU | 127 s / 138 s | 31 s / 33 s | 139 s / 152 s |
| train peak RAM growth | 831 MiB | 239 MiB | 792 MiB |
| prefill | 6,573 B/s | 25,220 B/s | 5,864 B/s |
| greedy decode | 2.72 ms/B | 1.50 ms/B | 3.67 ms/B |
| speculative decode (kept states) | 2.74 ms/B, 2.0 B/call, draft acceptance 33%, output = greedy | | |
| speculative decode (replay rollback) | 4.15 ms/B, 1.0 B/call, output = greedy | | |
| recurrent state at 768 bytes | 256 KiB (FP32) | 18 KiB | 256 KiB |
| attention K/V at the same length (FP16) | 3.0 MiB | 0.8 MiB | 3.0 MiB |
| infer peak RAM growth | 52 MiB | 5 MiB | 13 MiB |
