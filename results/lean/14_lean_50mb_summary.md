# Example 14 at ~50 MB deployed: lean vs FP16 of the same size

300 steps x 8 x 128 bytes of Shakespeare (307 KB, small sample), 4 CPU threads, cosine, AdamW (single-tensor).
Lean: width 2560, 8 layers, 20 heads of 128, LittleBit 0.55 bpw from the SVD of the init (no FP phase, no ITQ:
an FP phase at this width needs ~11 GB of optimizer state), per-block gradient checkpointing.
FP16: width 496, 8 layers, 3 heads, solved to the same deployed bytes.

| | lean | fp16_size |
|---|---|---|
| deployed | 51.9 MB | 51.4 MB |
| trainable values (LittleBit latents vs FP weights) | 362.8M | 25.7M |
| final train CE (last step) | 1.640 (MTP 2.77) | 1.688 |
| held-out bpb | 2.688 (LUT-packed 2.677) | 2.700 |
| train wall / CPU | 3,784 s / 14,766 s (+302 s conversion inside) | 288 s / 1,135 s |
| train RAM growth (process peak) | 4.6 GiB (~8.2 GB) | 1.7 GiB |
| prefill | 352 B/s | 2,309 B/s |
| greedy decode | 24.7 ms/B | 9.5 ms/B |
| speculative decode (kept states) | 32.1 ms/B, 1.83 B/call, 28% drafts accepted, = greedy | |
| speculative decode (replay) | 42.3 ms/B, 0.92 B/call | |
| GDN state at 640 bytes | 10.0 MiB | 3.8 MiB |
| attention K/V at the same length (FP16) | 50 MiB | 9.7 MiB |
| infer RAM growth | 534 MiB | 28 MiB |

Same quality at the same bytes; lean costs 13x the training wall time, 2.6x the decode time and 6.6x the prefill
time. At matched bytes LittleBit's model is 5x wider, so it does ~12x the multiply-adds per byte; the LUT kernel
does not recover that on CPU.
